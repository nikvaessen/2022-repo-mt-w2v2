################################################################################
#
# Implement wav2vec2 as speech recognition module
#
# Author(s): Anonymous
################################################################################

import math

from typing import Optional, Callable, List, Tuple

import torch as t

from dataclasses import dataclass
from omegaconf import DictConfig

from src.evaluation.speech.tokenizer import BaseTokenizer
from src.networks import Wav2Vec2Network, Wav2Vec2NetworkRegularisationConfig
from src.networks.heads import SpeechRecognitionHead
from src.networks.wav2vec2 import (
    freeze_wav2vec2_on_after_backward,
    freeze_wav2vec2_on_train_start,
)
from src.pl_modules.speech_recognition_module import SpeechRecognitionLightningModule

################################################################################
# config


@dataclass
class Wav2vec2SpeechRecognitionModuleConfig:
    # pretrained weights of wav2vec model
    wav2vec_huggingface_id: str

    # whether to use reset the pretrained weights
    # and start from a fresh initialization
    reset_weights: bool

    # initially freeze wav2vec model
    wav2vec_initially_frozen: bool

    # whether to freeze the feature encoder part
    # of the network for the whole training run
    completely_freeze_feature_extractor: bool

    # number of steps before the wav2vec model is unfrozen
    # (if initially frozen at all)
    # if set to null, wav2vec will never be unfrozen
    num_frozen_steps: int

    # settings for regularization
    regularisation: Wav2Vec2NetworkRegularisationConfig

    # if enabled, gradient checkpointing slows down iteration speed but saves memory
    use_gradient_checkpointing: bool

    # language recognition head pretrained weights
    speech_head_huggingface_id: Optional[str] = None


################################################################################
# Lightning module


class Wav2vec2SpeechRecognitionModule(SpeechRecognitionLightningModule):
    def __init__(
        self,
        cfg: Wav2vec2SpeechRecognitionModuleConfig,
        hyperparameter_config: DictConfig,
        loss_fn_constructor: Callable[[], Callable[[t.Tensor, t.Tensor], t.Tensor]],
        tokenizer: BaseTokenizer,
    ):
        self.cfg = cfg

        super().__init__(hyperparameter_config, loss_fn_constructor, tokenizer)

        # load base network
        self.wav2vec2 = Wav2Vec2Network(
            wav2vec2_huggingface_id=cfg.wav2vec_huggingface_id,
            reset_weights=self.cfg.reset_weights,
            reg_cfg=self.cfg.regularisation,
            insert_cls_token=False,
            learnable_cls_token=False,
            gradient_checkpointing=self.cfg.use_gradient_checkpointing,
        )

        # fc layer for speech recognition
        self.speech_recognition_head = SpeechRecognitionHead(
            embedding_size=self.wav2vec2.num_embedding_features,
            vocab_size=tokenizer.vocabulary_size(),
            dropout_prob=self.cfg.regularisation.final_dropout,
            wav2vec2_huggingface_id=cfg.speech_head_huggingface_id,
            skip_first_token=False,
        )

    def compute_embedding_sequence(
        self, wav_input: t.Tensor, lengths: List[int]
    ) -> Tuple[t.Tensor, List[int]]:
        # transform input
        # (of shape [BS, 1, NUM_AUDIO_SAMPLES] or [1, NUM_AUDIO_SAMPLES])
        # to the required [BS, NUM_AUDIO_SAMPLES]
        if len(wav_input.shape) == 3 and wav_input.shape[1] == 1:
            wav_input = t.squeeze(wav_input)
        if len(wav_input.shape) == 1:
            wav_input = t.stack([wav_input])

        # first compute the wav2vec embeddings: will be shape [BS, NUM_WINDOWS, EMBEDDING_SIZE]
        num_audio_samples = [wav_input.shape[1] for _ in range(wav_input.shape[0])]
        (
            audio_features,
            num_audio_features,
            attention_mask,
        ) = self.wav2vec2.extract_features(wav_input, num_audio_samples)

        wav2vec2_embeddings = self.wav2vec2.compute_wav2vec2_embeddings(
            audio_features, attention_mask
        )

        # we end with all the operations to get to the speaker embeddings
        return wav2vec2_embeddings, num_audio_features

    def compute_vocabulary_prediction(
        self, embedding_tensor: t.Tensor, lengths: List[int]
    ) -> Tuple[t.Tensor, List[int]]:
        # lengths is not modified by head
        letter_prediction = self.speech_recognition_head(embedding_tensor)

        return letter_prediction, lengths

    def generate_example_input(
        self, include_batch_dimension: bool, batch_size: Optional[int] = None
    ):
        if include_batch_dimension:
            # [BATCH_SIZE, NUMBER_OF_AUDIO_SAMPLES]
            # the `16000` varies depending on length of audio file
            # (1 second in this case)
            shape = [batch_size, 16000]
        else:
            # [BATCH_SIZE, NUMBER_OF_AUDIO_SAMPLES]
            # the `16000` varies depending on length of audio file
            # (1 second in this case)
            shape = [
                16000,
            ]

        return t.rand(size=shape), [16000]

    def on_train_start(self) -> None:
        freeze_wav2vec2_on_train_start(
            self,
            self.wav2vec2,
            self.cfg.wav2vec_initially_frozen,
            self.cfg.completely_freeze_feature_extractor,
        )

    def on_after_backward(self) -> None:
        freeze_wav2vec2_on_after_backward(
            self,
            self.wav2vec2,
            self.cfg.num_frozen_steps,
            self.cfg.completely_freeze_feature_extractor,
        )
