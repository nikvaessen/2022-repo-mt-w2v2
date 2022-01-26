################################################################################
#
# Implement the wav2vec2 + fc network head for speaker recognition as a
# SpeakerRecognitionModule.
#
# Author(s): Anonymous
################################################################################

from dataclasses import dataclass
from typing import List, Optional, Callable

import torch as t

from omegaconf import DictConfig

from src.evaluation.speaker.speaker_recognition_evaluator import EvaluationPair
from src.networks.heads import SpeakerRecognitionHead
from src.networks.wav2vec2 import (
    Wav2Vec2NetworkRegularisationConfig,
    Wav2Vec2Network,
    freeze_wav2vec2_on_train_start,
    freeze_wav2vec2_on_after_backward,
)
from src.networks.wav2vec2_components.random_permutation_encoder import (
    Wav2vec2RandomPermutationEncoder,
)
from src.pl_modules.speaker_recognition_module import SpeakerRecognitionLightningModule

################################################################################
# Implementation of wav2vec with x-vector network head


@dataclass
class Wav2vec2SpeakerRecognitionModuleConfig:
    # settings for wav2vec architecture
    wav2vec_huggingface_id: str
    reset_weights: bool

    # settings related to training wav2vec2
    wav2vec_initially_frozen: bool
    num_frozen_steps: Optional[int]
    completely_freeze_feature_extractor: bool

    # settings for fc head
    stat_pooling_type: str
    use_cosine_linear: bool

    # probability of regularization techniques during training

    # shuffle input to encoder (before, or after rel. pos. emd.)
    random_permutation_mode: str  # one of 'before', 'after', or 'None'

    # settings for regularization
    regularisation: Wav2Vec2NetworkRegularisationConfig

    # if enabled, gradient checkpointing slows down iteration speed but saves memory
    use_gradient_checkpointing: bool

    # optional explicit overwrite of embedding size (e.g if you
    # need to load finetuned weights but want to experiment with another
    # pooling type in the evaluation)
    explicit_stat_pool_embedding_size: Optional[int]
    explicit_num_speakers: Optional[int]


class Wav2vec2SpeakerRecognitionModule(SpeakerRecognitionLightningModule):
    def __init__(
        self,
        hyperparameter_config: DictConfig,
        cfg: Wav2vec2SpeakerRecognitionModuleConfig,
        loss_fn_constructor: Callable[[], Callable[[t.Tensor, t.Tensor], t.Tensor]],
        num_speakers: int,
        validation_pairs: List[EvaluationPair],
        test_pairs: List[List[EvaluationPair]],
        test_names: List[str],
        auto_lr_find: Optional[float] = None,
    ):
        self.cfg = cfg

        # initialize as super class
        super().__init__(
            hyperparameter_config=hyperparameter_config,
            num_speakers=num_speakers,
            loss_fn_constructor=loss_fn_constructor,
            validation_pairs=validation_pairs,
            test_pairs=test_pairs,
            test_names=test_names,
            auto_lr_find=auto_lr_find,
        )

        if (
            self.cfg.random_permutation_mode
            not in Wav2vec2RandomPermutationEncoder.shuffle_input_sequence_options
        ):
            raise ValueError(
                f"{self.cfg.random_permutation_mode=} should be one of "
                f"{Wav2vec2RandomPermutationEncoder.shuffle_input_sequence_options}"
            )

        # create base wav2vec model
        self.wav2vec2 = Wav2Vec2Network(
            wav2vec2_huggingface_id=cfg.wav2vec_huggingface_id,
            reset_weights=self.cfg.reset_weights,
            reg_cfg=self.cfg.regularisation,
            gradient_checkpointing=self.cfg.use_gradient_checkpointing,
            insert_cls_token="first+cls" == self.cfg.stat_pooling_type
            or "first+cls+learnable" == self.cfg.stat_pooling_type,
            learnable_cls_token="first+cls+learnable" == self.cfg.stat_pooling_type,
            random_permutation_mode=self.cfg.random_permutation_mode,
        )

        # pooling + fc layer for speaker recognition
        self.speaker_recognition_head = SpeakerRecognitionHead(
            stat_pooling_type=self.cfg.stat_pooling_type,
            num_speakers=num_speakers
            if self.cfg.explicit_num_speakers is None
            else self.cfg.explicit_num_speakers,
            wav2vec2_embedding_size=self.wav2vec2.num_embedding_features,
            dropout_prob=self.cfg.regularisation.final_dropout,
            use_cosine_linear=self.cfg.use_cosine_linear,
        )

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

        return t.rand(size=shape)

    @property
    def speaker_embedding_size(self):
        return self.speaker_recognition_head.stat_pool_dimension

    def compute_speaker_embedding(self, wav_input: t.Tensor) -> t.Tensor:
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
        speaker_embedding, _ = self.speaker_recognition_head(
            wav2vec2_embeddings, lengths=num_audio_features, skip_prediction=True
        )

        return speaker_embedding

    def compute_speaker_prediction(self, embedding_tensor: t.Tensor) -> t.Tensor:
        # we apply all operations we need to apply on the speaker
        # embedding to get to the classification prediction
        _, speaker_prediction = self.speaker_recognition_head(
            embedding_tensor, skip_embedding=True
        )

        return speaker_prediction

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
