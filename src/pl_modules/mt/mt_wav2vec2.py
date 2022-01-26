################################################################################
#
# Implement a wav2vec2 model which does speech and speaker recognition.
#
# Author(s): Anonymous
################################################################################

import math

from typing import Optional, Callable, List, Tuple

import torch as t

from dataclasses import dataclass
from omegaconf import DictConfig

from src.evaluation.speaker.speaker_recognition_evaluator import EvaluationPair
from src.evaluation.speech.tokenizer import BaseTokenizer
from src.networks import Wav2Vec2Network, Wav2Vec2NetworkRegularisationConfig
from src.networks.branched_wav2vec2 import BranchedWav2Vec2Network
from src.networks.heads import SpeechRecognitionHead, SpeakerRecognitionHead
from src.networks.wav2vec2 import (
    freeze_wav2vec2_on_train_start,
    freeze_wav2vec2_on_after_backward,
)
from src.pl_modules.mt_speech_speaker_lightning_module import (
    MtSpeechSpeakerLightningModule,
)

################################################################################
# config


@dataclass
class MultiTaskWav2vec2ModuleConfig:
    # pretrained weights of wav2vec model
    wav2vec_huggingface_id: str

    # whether to use reset the pretrained weights
    # and start from a fresh initialization
    reset_weights: bool

    # initially freeze wav2vec model
    wav2vec_initially_frozen: bool

    # number of steps before the wav2vec model is unfrozen
    # (if initially frozen at all)
    # if set to null, wav2vec will never be unfrozen
    num_frozen_steps: int

    # whether to freeze the feature encoder part
    # of the network for the whole training run
    completely_freeze_feature_extractor: bool

    # whether to use a split encoder, and if so, how many layers to still share
    # before branching
    use_multi_branch_encoder: bool
    num_shared_transformers: int

    # attend on output with spiked ctc pooling (either blank or non-blank token)
    enable_ctc_pool_speaker_embedding: bool
    enable_ctc_pool_no_grad: bool
    ctc_pool_on_blank_embeddings: bool
    ctc_blank_idx: int

    # settings for fc head which does speaker classification
    stat_pooling_type: str
    use_cosine_linear: bool
    num_speaker_dimensions: int

    # settings for regularization
    regularisation: Wav2Vec2NetworkRegularisationConfig

    # if enabled, gradient checkpointing slows down iteration speed but saves memory
    use_gradient_checkpointing: bool

    # language recognition head pretrained weights
    speech_head_huggingface_id: Optional[str] = None

    # optional explicit overwrite of embedding size (e.g if you
    # need to load finetuned weights but want to experiment with another
    # pooling type in the evaluation)
    explicit_stat_pool_embedding_size: Optional[int] = None
    explicit_num_speakers: Optional[int] = None


################################################################################
#


class MultiTaskWav2vec2Module(MtSpeechSpeakerLightningModule):
    def __init__(
        self,
        cfg: MultiTaskWav2vec2ModuleConfig,
        hyperparameter_config: DictConfig,
        loss_fn_constructor: Callable[[], Callable[[t.Tensor, t.Tensor], t.Tensor]],
        tokenizer: BaseTokenizer,
        num_speakers: int,
        speaker_validation_pairs: List[EvaluationPair],
        speaker_test_pairs: List[List[EvaluationPair]],
        test_names: List[str],
        auto_lr_find: Optional[float] = None,
    ):
        self.cfg = cfg

        super().__init__(
            hyperparameter_config=hyperparameter_config,
            loss_fn_constructor=loss_fn_constructor,
            num_speakers=num_speakers,
            speaker_validation_pairs=speaker_validation_pairs,
            speaker_test_pairs=speaker_test_pairs,
            test_names=test_names,
            tokenizer=tokenizer,
            auto_lr_find=auto_lr_find,
            num_speaker_dimensions=self.cfg.num_speaker_dimensions,
        )

        # load network
        insert_class_token = (
            "first+cls" == self.cfg.stat_pooling_type
            or "first+cls+learnable" == self.cfg.stat_pooling_type
        )
        if self.cfg.use_multi_branch_encoder:
            self.wav2vec2 = BranchedWav2Vec2Network(
                wav2vec2_huggingface_id=cfg.wav2vec_huggingface_id,
                reset_weights=self.cfg.reset_weights,
                reg_cfg=self.cfg.regularisation,
                gradient_checkpointing=self.cfg.use_gradient_checkpointing,
                insert_cls_token=insert_class_token,
                learnable_cls_token="first+cls+learnable" == self.cfg.stat_pooling_type,
                branch_idx=self.cfg.num_shared_transformers,
            )
        else:
            self.wav2vec2 = Wav2Vec2Network(
                wav2vec2_huggingface_id=cfg.wav2vec_huggingface_id,
                reset_weights=self.cfg.reset_weights,
                reg_cfg=self.cfg.regularisation,
                gradient_checkpointing=self.cfg.use_gradient_checkpointing,
                insert_cls_token=insert_class_token,
                learnable_cls_token="first+cls+learnable" == self.cfg.stat_pooling_type,
            )

        if (
            self.cfg.enable_ctc_pool_speaker_embedding
            and self.cfg.stat_pooling_type not in ["mean", "mean&std"]
        ):
            raise ValueError(
                "enable_ctc_pool_speaker_embedding only valid for pooling mean and mean&std"
            )

        if (
            self.cfg.num_speaker_dimensions is not None
            and self.cfg.num_speaker_dimensions >= self.wav2vec2.num_embedding_features
        ):
            raise ValueError(
                f"{self.cfg.num_speaker_dimensions=} >= {self.wav2vec2.num_embedding_features=}"
            )

        # fc layer for speech recognition
        self.speech_recognition_head = SpeechRecognitionHead(
            embedding_size=self.wav2vec2.num_embedding_features
            if self.cfg.num_speaker_dimensions is None
            else self.wav2vec2.num_embedding_features - self.num_speaker_dimensions,
            vocab_size=tokenizer.vocabulary_size(),
            dropout_prob=self.cfg.regularisation.final_dropout,
            wav2vec2_huggingface_id=cfg.speech_head_huggingface_id,
            skip_first_token=insert_class_token,
        )

        # pooling + fc layer for speaker recognition
        self.speaker_recognition_head = SpeakerRecognitionHead(
            stat_pooling_type=self.cfg.stat_pooling_type,
            num_speakers=num_speakers
            if cfg.explicit_num_speakers is None
            else cfg.explicit_num_speakers,
            wav2vec2_embedding_size=self.wav2vec2.num_embedding_features
            if self.cfg.num_speaker_dimensions is None
            else self.cfg.num_speaker_dimensions,
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
            lengths = [16_000 for _ in range(batch_size)]
        else:
            # [BATCH_SIZE, NUMBER_OF_AUDIO_SAMPLES]
            # the `16000` varies depending on length of audio file
            # (1 second in this case)
            shape = [1, 16000]
            lengths = [16_000]

        return t.rand(size=shape), lengths

    @property
    def speaker_embedding_size(self):
        return self.speaker_recognition_head.stat_pool_dimension

    @property
    def speech_embedding_size(self):
        return self.speech_recognition_head.embedding_size

    def compute_input_token_sequence(self, input_tensor: t.Tensor, lengths: List[int]):
        # transform input
        # (of shape [BS, 1, NUM_AUDIO_SAMPLES] or [NUM_AUDIO_SAMPLES])
        # to the required [BS, NUM_AUDIO_SAMPLES]
        if len(input_tensor.shape) == 3 and input_tensor.shape[1] == 1:
            input_tensor = t.squeeze(input_tensor)
        if len(input_tensor.shape) == 1:
            input_tensor = t.stack([input_tensor])

        # compute features
        (
            input_feature_tokens,
            feature_lengths,
            attention_mask,
        ) = self.wav2vec2.extract_features(input_tensor, lengths)

        return input_feature_tokens, feature_lengths, attention_mask

    def compute_output_token_sequence(
        self, input_tensor: t.Tensor, attention_mask: t.Tensor
    ) -> t.Tensor:
        if self.cfg.use_multi_branch_encoder:
            # feed input tokens through encoder
            (
                output_sequence1,
                output_sequence2,
            ) = self.wav2vec2.compute_wav2vec2_embeddings(input_tensor, attention_mask)

            return output_sequence1, output_sequence2
        else:
            # feed input tokens through encoder
            output_sequence = self.wav2vec2.compute_wav2vec2_embeddings(
                input_tensor, attention_mask
            )

            return output_sequence

    def compute_speaker_embedding(
        self,
        sequence_tensor: t.Tensor,
        lengths: List[int],
        letter_predictions: Optional[t.Tensor] = None,
    ) -> t.Tensor:
        # if ctc_based speaker embedding pooling is enabled,
        # we modify the embedding time-axis order and 'max' length
        # such that mean pooling only uses the embedding which
        # are either blank or non-blank
        if self.cfg.enable_ctc_pool_speaker_embedding:
            if letter_predictions is None:
                raise ValueError(
                    "if ctc_based_speaker_embedding_pool, "
                    "letter_predictions should not be None"
                )

            sequence_tensor, lengths = _reorder_embeddings_based_on_blank_idx(
                embedding_sequence=sequence_tensor,
                lengths=lengths,
                letter_predictions=letter_predictions,
                blank_idx=self.cfg.ctc_blank_idx,
                blanks_valid=self.cfg.ctc_pool_on_blank_embeddings,
            )

        embedding, _ = self.speaker_recognition_head(
            sequence_tensor, lengths, skip_prediction=True
        )

        return embedding

    def compute_speaker_prediction(self, embedding_tensor: t.Tensor) -> t.Tensor:
        _, prediction = self.speaker_recognition_head(
            embedding_tensor, skip_embedding=True
        )

        return prediction

    def compute_vocabulary_prediction(
        self, embedding_tensor: t.Tensor, lengths: List[int]
    ) -> Tuple[t.Tensor, List[int]]:
        letter_predictions = self.speech_recognition_head(embedding_tensor)

        return letter_predictions, lengths

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


def _reorder_embeddings_based_on_blank_idx(
    embedding_sequence: t.Tensor,
    lengths: List[int],
    letter_predictions: t.Tensor,
    blank_idx: int,
    blanks_valid: bool,
) -> Tuple[t.Tensor, List[int]]:
    with t.no_grad():
        peek_idx = t.argmax(letter_predictions, dim=2)

    for batch_idx in range(embedding_sequence.shape[0]):
        valid_idx = []
        invalid_idx = []

        for time_idx in range(embedding_sequence.shape[1]):
            if time_idx >= lengths[batch_idx]:
                invalid_idx.append(time_idx)
                continue

            if blanks_valid:
                if peek_idx[batch_idx, time_idx] == blank_idx:
                    valid_idx.append(time_idx)
                else:
                    invalid_idx.append(time_idx)
            else:
                if peek_idx[batch_idx, time_idx] != blank_idx:
                    valid_idx.append(time_idx)
                else:
                    invalid_idx.append(time_idx)

        if len(valid_idx) == 0:
            continue

        new_length = len(valid_idx)
        new_idx_order = [] + valid_idx + invalid_idx

        embedding_sequence[batch_idx, :] = embedding_sequence[batch_idx, new_idx_order]
        lengths[batch_idx] = new_length

    return embedding_sequence, lengths
