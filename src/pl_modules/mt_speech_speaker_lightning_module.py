################################################################################
#
# Define a base lightning module for speech and/or speaker recognition network.
#
# Author(s): Anonymous
################################################################################

import logging
import pathlib

from abc import abstractmethod
from typing import Callable, Optional, List, Tuple, Any, Dict, Union

import torch as t
import torchmetrics

from omegaconf import DictConfig

from src.data.batches import SpeechRecognitionDataBatch, SpeakerClassificationDataBatch
from src.evaluation.speaker.cosine_distance import CosineDistanceEvaluator
from src.evaluation.speaker.speaker_recognition_evaluator import EvaluationPair
from src.evaluation.speech.tokenizer import BaseTokenizer
from src.evaluation.speech.wer import calculate_wer
from src.pl_modules import BaseLightningModule
from src.pl_modules.speaker_recognition_module import evaluate_embeddings
from src.pl_modules.speech_recognition_module import (
    decode_letter_prediction,
    calculate_wer_on_collected_output,
)

################################################################################
# Definition of speaker recognition API

# A logger for this file

log = logging.getLogger(__name__)


class MtSpeechSpeakerLightningModule(BaseLightningModule):
    def __init__(
        self,
        hyperparameter_config: DictConfig,
        loss_fn_constructor: Callable[[], Callable[[t.Tensor, t.Tensor], t.Tensor]],
        num_speakers: int,
        speaker_validation_pairs: List[EvaluationPair],
        speaker_test_pairs: List[List[EvaluationPair]],
        test_names: List[str],
        tokenizer: BaseTokenizer,
        auto_lr_find: Optional[float] = None,
        num_speaker_dimensions: Optional[int] = None,
    ):
        super().__init__(
            hyperparameter_config=hyperparameter_config,
            loss_fn_constructor=loss_fn_constructor,
            auto_lr_find=auto_lr_find,
        )

        # input arguments for speaker part
        self.num_speakers = num_speakers
        self.speaker_validation_pairs = speaker_validation_pairs
        self.speaker_test_pairs = speaker_test_pairs
        self.test_names = test_names
        self.num_speaker_dimensions = num_speaker_dimensions

        # multi-task metric
        self.metric_train_loss = torchmetrics.MeanMetric()

        # used to keep track of training/val accuracy
        self.metric_train_speaker_acc = torchmetrics.Accuracy()
        self.metric_train_speaker_loss = torchmetrics.MeanMetric()

        self.metric_valid_speaker_acc = torchmetrics.Accuracy()

        # evaluator
        self.evaluator = CosineDistanceEvaluator(
            center_before_scoring=False,
            length_norm_before_scoring=False,
        )

        # required for decoding to text
        self.tokenizer = tokenizer
        self.vocab_size = tokenizer.vocabulary_size()

        # keep track of metrics for speech
        self.metric_train_speech_loss = torchmetrics.MeanMetric()
        self.metric_train_speech_wer = torchmetrics.MeanMetric()
        self.metric_train_speech_wer_min = torchmetrics.MinMetric()
        self.metric_train_speech_wer_max = torchmetrics.MaxMetric()

        self.metric_val_loss_clean = torchmetrics.MeanMetric()
        self.metric_val_loss_other = torchmetrics.MeanMetric()

        # set on first call to self#_log_transcription_progress step
        self.tracking_audio_sample: t.Tensor = None
        self.tracking_transcription: str = None
        self.tracking_sequence_length: int = None

    @property
    @abstractmethod
    def speaker_embedding_size(self):
        pass

    @property
    @abstractmethod
    def speech_embedding_size(self):
        pass

    def forward(
        self,
        input_tensor: t.Tensor,
        lengths: List[int],
        skip_speech: bool = False,
        skip_speaker: bool = False,
    ):
        # assume input tensor of shape [BS, NUM_AUDIO_SAMPLES]
        return_dict = {}

        (
            input_token_sequence,
            input_lengths,
            attention_mask,
        ) = self.compute_input_token_sequence(input_tensor, lengths)

        embedding_sequence = self.compute_output_token_sequence(
            input_token_sequence, attention_mask
        )

        if isinstance(embedding_sequence, tuple):
            speaker_embedding_sequence = embedding_sequence[0]
            speech_embedding_sequence = embedding_sequence[1]
        else:
            speaker_embedding_sequence = embedding_sequence
            speech_embedding_sequence = embedding_sequence

        return_dict["speaker_embedding_sequence"] = embedding_sequence
        return_dict["speech_embedding_sequence"] = embedding_sequence
        return_dict["embedding_lengths"] = input_lengths

        # speaker branch
        if not skip_speaker:
            if self.num_speaker_dimensions is not None:
                speaker_embedding_sequence = speaker_embedding_sequence[
                    :, :, 0 : self.num_speaker_dimensions
                ]

            speaker_embedding = self.compute_speaker_embedding(
                speaker_embedding_sequence, input_lengths
            )
            speaker_prediction = self.compute_speaker_prediction(speaker_embedding)

            return_dict["speaker_embedding"] = speaker_embedding
            return_dict["speaker_prediction"] = speaker_prediction

        # speech branch
        if not skip_speech:
            if self.num_speaker_dimensions is not None:
                speech_embedding_sequence = embedding_sequence[
                    :, :, self.num_speaker_dimensions :
                ]

            vocabulary_prediction, vocab_lengths = self.compute_vocabulary_prediction(
                speech_embedding_sequence, input_lengths
            )

            return_dict["vocabulary_prediction"] = vocabulary_prediction
            return_dict["vocab_lengths"] = vocab_lengths

        return return_dict

    @abstractmethod
    def compute_input_token_sequence(
        self, input_tensor: t.Tensor, lengths: List[int]
    ) -> Tuple[t.Tensor, List[int], t.Tensor]:
        # transform:
        # 1) input_tensor with shape [BATCH_SIZE, NUM_SAMPLES]
        # 2) where 0:lengths[BATCH_IDX] are non-padded frames
        # into:
        # 1) an embedding of shape [BATCH_SIZE, SEQUENCE_LENGTH, EMBEDDING_SIZE]
        # 2) a list of lengths which represents frames which are non-padded
        #    (index 0:length_value is non-padded)
        # 3) an attention mask of shape [BATCH_SIZE, SEQUENCE_LENGTH] where '0' means
        #    tokens are not masked and '1' means that tokens are masked
        # which will be used as input to the encoder network
        pass

    @abstractmethod
    def compute_output_token_sequence(
        self, input_tensor: t.Tensor, attention_mask: t.Tensor
    ) -> t.Tensor:
        # transform:
        # 1) input_tensor with shape [BATCH_SIZE, SEQUENCE_LENGTH, EMBEDDING_SIZE]
        # 2) an attention mask of shape [BATCH_SIZE, SEQUENCE_LENGTH] where '0' means
        #    tokens are not masked and '1' means that tokens are masked
        # into:
        # 1) an embedding of shape [BATCH_SIZE, SEQUENCE_LENGTH, EMBEDDING_SIZE]
        # which is a self-attention encoded output sequence useful for downstream
        # tasks
        pass

    @abstractmethod
    def compute_speaker_embedding(
        self,
        sequence_tensor: t.Tensor,
        lengths: List[int],
        letter_predictions: Optional[t.Tensor] = None,
    ):
        # transform input_tensor with shape [BATCH_SIZE, SEQUENCE_LENGTH, EMBEDDING_SIZE]
        # into an embedding of shape [BATCH_SIZE, EMBEDDING_SIZE]
        pass

    @abstractmethod
    def compute_speaker_prediction(self, embedding_tensor: t.Tensor) -> t.Tensor:
        # transform embedding tensor with shape [BATCH_SIZE, EMBEDDING_SIZE]
        # into a speaker prediction of shape [BATCH_SIZE, NUM_SPEAKERS]
        pass

    @abstractmethod
    def compute_vocabulary_prediction(
        self, embedding_tensor: t.Tensor, lengths: List[int]
    ) -> Tuple[t.Tensor, List[int]]:
        # transform embedding tensor with shape [BATCH_SIZE, SEQUENCE_LENGTH, EMBEDDING_SIZE]
        # and list of non-padded range for each batch dimension
        # into a speaker prediction of shape [BATCH_SIZE, SEQUENCE_LENGTH, VOCAB_SIZE]
        # and a list of non-padded range for each batch dimension
        pass

    def training_step(
        self,
        batch: Dict[
            str, Union[SpeechRecognitionDataBatch, SpeakerClassificationDataBatch]
        ],
        batch_idx: int,
    ):
        speech_batch = batch["speech"]
        speaker_batch = batch["speaker"]

        assert isinstance(speaker_batch, SpeakerClassificationDataBatch)
        assert isinstance(speech_batch, SpeechRecognitionDataBatch)

        # first forward pass on speech batch
        speech_prediction, speech_pred_lengths = self._step_speech(speech_batch)
        speech_gt = speech_batch.ground_truth
        speech_gt_length = speech_batch.ground_truth_sequence_length

        # second forward pass on speaker batch
        _, speaker_logits_prediction = self._step_speaker(speaker_batch)
        speaker_labels = speaker_batch.ground_truth

        # compute loss
        (
            loss,
            speech_loss,
            speaker_loss,
            speaker_prediction,
            weight_speech,
            weight_speaker,
        ) = self.loss_fn(
            speech_predictions=speech_prediction,
            speech_prediction_lengths=speech_pred_lengths,
            speech_ground_truths=speech_gt,
            speech_ground_truth_lengths=speech_gt_length,
            speaker_logits=speaker_logits_prediction,
            speaker_labels=speaker_labels,
        )

        # log metrics
        self._log_train_speaker_acc(speaker_prediction, speaker_labels, batch_idx)
        self._log_train_speaker_loss(speaker_loss, weight_speaker, batch_idx)
        self._log_train_speech_loss_and_wer(
            loss=speech_loss,
            loss_weight=weight_speech,
            letter_prediction=speech_prediction,
            letter_prediction_lengths=speech_pred_lengths,
            speech_batch=speech_batch,
            batch_idx=batch_idx,
        )
        self._log_train_loss(loss, batch_idx)

        return {"loss": loss}

    def _step_speaker_and_speech(
        self,
        speaker_batch: SpeakerClassificationDataBatch,
        speech_batch: SpeechRecognitionDataBatch,
    ):
        speaker_audio_input = speaker_batch.audio_input
        speaker_audio_input_lengths = speaker_batch.audio_input_lengths

        speech_audio_input = speech_batch.audio_input
        speech_audio_input_lengths = speech_batch.audio_input_lengths

        # compute features for both batches independently
        (
            speaker_input_token_sequence,
            speaker_input_lengths,
            speaker_attention_mask,
        ) = self.compute_input_token_sequence(
            speaker_audio_input, speaker_audio_input_lengths
        )
        (
            speech_input_token_sequence,
            speech_input_lengths,
            speech_attention_mask,
        ) = self.compute_input_token_sequence(
            speech_audio_input, speech_audio_input_lengths
        )

        # merge results
        max_seq_length = max(speech_input_lengths + speaker_input_lengths)

        # pad
        speaker_attention_mask = self._pad_attention_mask_to_sequence_length(
            speaker_attention_mask, max_seq_length
        )
        speech_attention_mask = self._pad_attention_mask_to_sequence_length(
            speech_attention_mask, max_seq_length
        )
        merged_attention_mask = t.cat([speaker_attention_mask, speech_attention_mask])

        speech_input_token_sequence = self._pad_input_tokens_to_sequence_length(
            speech_input_token_sequence, max_seq_length
        )
        speaker_input_token_sequence = self._pad_input_tokens_to_sequence_length(
            speaker_input_token_sequence, max_seq_length
        )
        merged_input_token_sequence = t.cat(
            [speaker_input_token_sequence, speech_input_token_sequence]
        )

        # do encoder step once
        merged_embedding_tokens = self.compute_output_token_sequence(
            merged_input_token_sequence, merged_attention_mask
        )

        # separate back into speech and speaker results
        speaker_output_tokens = merged_embedding_tokens[0 : speaker_batch.batch_size]
        speech_output_tokens = merged_embedding_tokens[speaker_batch.batch_size :]

        # apply speaker head
        speaker_embedding = self.compute_speaker_embedding(
            speaker_output_tokens, speaker_input_lengths
        )

        assert len(speaker_embedding.shape) == 2
        assert speaker_embedding.shape[-1] == self.speaker_embedding_size

        speaker_logits_prediction = self.compute_speaker_prediction(speaker_embedding)

        # apply speech head
        assert len(speech_output_tokens.shape) == 3
        assert speech_output_tokens.shape[-1] == self.speaker_embedding_size

        speech_prediction, speech_pred_lengths = self.compute_vocabulary_prediction(
            speech_output_tokens, speech_input_lengths
        )

        # return results
        return (
            speaker_embedding,
            speaker_logits_prediction,
            speech_prediction,
            speech_pred_lengths,
        )

    @staticmethod
    def _pad_attention_mask_to_sequence_length(
        attention_mask: t.Tensor, sequence_length: int
    ):
        # assume shape [bs, sequence_length]
        bs, current_length = attention_mask.shape

        assert attention_mask.dtype == t.bool
        assert current_length <= sequence_length

        time_dim_to_add = sequence_length - current_length

        if time_dim_to_add == 0:
            return attention_mask

        to_add = t.zeros(
            (bs, time_dim_to_add), dtype=t.bool, device=attention_mask.device
        )

        return t.cat([attention_mask, to_add], dim=1)

    @staticmethod
    def _pad_input_tokens_to_sequence_length(
        input_tokens: t.Tensor, sequence_length: int
    ):
        # assume shape [bs, sequence_length, num_features]
        bs, current_length, num_features = input_tokens.shape

        assert current_length <= sequence_length

        time_dim_to_add = sequence_length - current_length

        if time_dim_to_add == 0:
            return input_tokens

        to_add = t.zeros(
            (bs, time_dim_to_add, num_features),
            dtype=input_tokens.dtype,
            device=input_tokens.device,
        )

        return t.cat([input_tokens, to_add], dim=1)

    def _step_speaker(self, speaker_batch: SpeakerClassificationDataBatch):
        speaker_audio_input = speaker_batch.audio_input
        speaker_audio_input_lengths = speaker_batch.audio_input_lengths

        (
            speaker_input_token_sequence,
            input_lengths,
            attention_mask,
        ) = self.compute_input_token_sequence(
            speaker_audio_input, speaker_audio_input_lengths
        )
        speaker_embedding_sequence = self.compute_output_token_sequence(
            speaker_input_token_sequence, attention_mask
        )

        if isinstance(speaker_embedding_sequence, tuple):
            speaker_embedding_sequence = speaker_embedding_sequence[0]

        if self.num_speaker_dimensions is not None:
            speaker_embedding_sequence = speaker_embedding_sequence[
                :, :, 0 : self.num_speaker_dimensions
            ]

        if self.cfg.enable_ctc_pool_speaker_embedding:
            if self.cfg.enable_ctc_pool_no_grad:
                with t.no_grad():
                    speech_prediction, _ = self.compute_vocabulary_prediction(
                        speaker_embedding_sequence, input_lengths
                    )
            else:
                speech_prediction, _ = self.compute_vocabulary_prediction(
                    speaker_embedding_sequence, input_lengths
                )
        else:
            speech_prediction = None

        speaker_embedding = self.compute_speaker_embedding(
            speaker_embedding_sequence, input_lengths, speech_prediction
        )

        assert len(speaker_embedding.shape) == 2
        assert speaker_embedding.shape[-1] == self.speaker_embedding_size

        speaker_logits_prediction = self.compute_speaker_prediction(speaker_embedding)

        return speaker_embedding, speaker_logits_prediction

    def _step_speech(self, speech_batch: SpeechRecognitionDataBatch):
        speech_audio_input = speech_batch.audio_input
        speech_audio_input_lengths = speech_batch.audio_input_lengths

        (
            speech_input_token_sequence,
            input_lengths,
            attention_mask,
        ) = self.compute_input_token_sequence(
            speech_audio_input, speech_audio_input_lengths
        )
        speech_embedding_sequence = self.compute_output_token_sequence(
            speech_input_token_sequence, attention_mask
        )

        if isinstance(speech_embedding_sequence, tuple):
            speech_embedding_sequence = speech_embedding_sequence[1]

        if self.num_speaker_dimensions is not None:
            speech_embedding_sequence = speech_embedding_sequence[
                :, :, self.num_speaker_dimensions :
            ]

        assert len(speech_embedding_sequence.shape) == 3
        assert speech_embedding_sequence.shape[-1] == self.speech_embedding_size

        speech_prediction, speech_pred_lengths = self.compute_vocabulary_prediction(
            speech_embedding_sequence, input_lengths
        )

        return speech_prediction, speech_pred_lengths

    def _log_train_loss(self, loss: t.Tensor, batch_idx: int):
        self.metric_train_loss(loss)

        if batch_idx % 100 == 0:
            self.log(
                "train_loss",
                self.metric_train_loss.compute(),
                on_step=True,
                on_epoch=False,
                prog_bar=True,
            )
            self.metric_train_loss.reset()

    def _log_train_speaker_acc(
        self, prediction: t.Tensor, label: t.Tensor, batch_idx: int
    ):
        self.metric_train_speaker_acc(prediction, label)

        if batch_idx % 100 == 0:
            self.log(
                "train_acc",
                self.metric_train_speaker_acc.compute(),
                on_step=True,
                on_epoch=False,
                prog_bar=True,
            )
            self.metric_train_speaker_acc.reset()

    def _log_train_speaker_loss(self, loss: t.Tensor, weight: t.Tensor, batch_idx: int):
        self.metric_train_speaker_loss(loss)

        self.log(
            "train_speaker_loss_weight",
            weight,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
        )
        if batch_idx % 100 == 0:
            self.log(
                "train_speaker_loss",
                self.metric_train_speaker_loss.compute(),
                on_step=True,
                on_epoch=False,
                prog_bar=True,
            )
            self.metric_train_speaker_loss.reset()

    def _log_train_speech_loss_and_wer(
        self,
        loss: t.Tensor,
        loss_weight: t.Tensor,
        letter_prediction: t.Tensor,
        letter_prediction_lengths: List[int],
        speech_batch: SpeechRecognitionDataBatch,
        batch_idx: int,
    ):
        with t.no_grad():
            transcription = decode_letter_prediction(
                self.tokenizer, letter_prediction, letter_prediction_lengths
            )
            ground_truth_transcription = speech_batch.ground_truth_strings
            train_wer = calculate_wer(transcription, ground_truth_transcription)

            # log training loss
            self.metric_train_speech_loss(loss.detach().cpu().item())
            self.metric_train_speech_wer(train_wer)
            self.metric_train_speech_wer_min(train_wer)
            self.metric_train_speech_wer_max(train_wer)

            self.log(
                "train_speech_loss_weight",
                loss_weight,
                on_step=True,
                on_epoch=False,
                prog_bar=True,
            )
            if batch_idx % 100 == 0:
                self.log_dict(
                    {
                        "train_speech_loss": self.metric_train_speech_loss.compute(),
                        "train_wer": self.metric_train_speech_wer.compute(),
                        "train_wer_min": self.metric_train_speech_wer_min.compute(),
                        "train_wer_max": self.metric_train_speech_wer_max.compute(),
                    },
                    on_step=True,
                    on_epoch=False,
                    prog_bar=True,
                )

                self.metric_train_speech_loss.reset()
                self.metric_train_speech_wer.reset()

            if self.global_step % 1000 == 0:
                log_transcription_progress(self, speech_batch)

    def validation_step(
        self,
        batch: Union[SpeechRecognitionDataBatch, SpeakerClassificationDataBatch],
        batch_idx: int,
        dataloader_idx: Optional[int] = None,
    ):
        if isinstance(batch, SpeakerClassificationDataBatch):
            return self._val_step_speaker(batch)
        else:
            return self._val_step_speech(batch)

    def _val_step_speech(self, speech_batch: SpeechRecognitionDataBatch):
        # forward pass on speech batch
        speech_prediction, speech_pred_lengths = self._step_speech(speech_batch)
        speech_gt = speech_batch.ground_truth
        speech_gt_length = speech_batch.ground_truth_sequence_length

        # compute loss
        loss = self.loss_fn.speech_loss(
            predictions=speech_prediction,
            ground_truths=speech_gt,
            prediction_lengths=speech_pred_lengths,
            ground_truth_lengths=speech_gt_length,
        )

        # compute transcription for val WER
        with t.no_grad():
            transcription = decode_letter_prediction(
                self.tokenizer, speech_prediction, speech_pred_lengths
            )
            ground_truth_transcription = speech_batch.ground_truth_strings

        return {
            "val_loss": loss.detach().to("cpu"),
            "transcription": transcription,
            "ground_truth": ground_truth_transcription,
        }

    def _val_step_speaker(self, speaker_batch: SpeakerClassificationDataBatch):
        # forward pass on speaker batch
        speaker_embedding, speaker_logits_prediction = self._step_speaker(speaker_batch)
        speaker_labels = speaker_batch.ground_truth

        # loss
        loss, prediction = self.loss_fn.speaker_loss(
            speaker_logits_prediction, speaker_labels
        )

        self.metric_valid_speaker_acc(prediction, speaker_labels)

        return {
            "val_loss": loss.detach().to("cpu"),
            "embedding": speaker_embedding.detach().to("cpu"),
            "sample_id": speaker_batch.keys,
        }

    def validation_epoch_end(self, outputs: List[Any]) -> None:
        # idx 0 is 'librispeech-dev-clean', idx 1 'librispeech-dev-other',
        # idx 2 voxceleb

        # speech val
        def collect_loss(output: List[Dict[str, List[str]]]):
            return t.mean(t.Tensor([d["val_loss"] for d in output]))

        wer_clean = calculate_wer_on_collected_output(outputs[0])
        wer_other = calculate_wer_on_collected_output(outputs[1])

        # speaker val
        results = evaluate_embeddings(
            self.evaluator, outputs[2], self.speaker_validation_pairs, False
        )

        # log results
        self.log_dict(
            {
                "val_loss_speech_clean": collect_loss(outputs[0]),
                "val_loss_speech_other": collect_loss(outputs[1]),
                "val_loss_speaker": collect_loss(outputs[2]),
                "val_wer_clean": wer_clean,
                "val_wer_other": wer_other,
                "val_speaker_acc": self.metric_valid_speaker_acc.compute(),
                "val_eer": results["eer"],
            },
            on_epoch=True,
            prog_bar=True,
        )
        self.metric_valid_speaker_acc.reset()

    def test_step(
        self,
        batch: SpeechRecognitionDataBatch,
        batch_idx: int,
        dataloader_idx: Optional[int] = None,
    ):
        if isinstance(batch, SpeakerClassificationDataBatch):
            return self._test_step_speaker(batch)
        else:
            return self._test_step_speech(batch)

    def _test_step_speaker(self, speaker_batch: SpeakerClassificationDataBatch):
        # forward pass on speaker batch
        speaker_embedding, _ = self._step_speaker(speaker_batch)

        return {
            "embedding": speaker_embedding.detach().to("cpu"),
            "sample_id": speaker_batch.keys,
        }

    def _test_step_speech(self, speech_batch: SpeechRecognitionDataBatch):
        # forward pass on speech batch
        speech_prediction, speech_pred_lengths = self._step_speech(speech_batch)

        # compute transcription for val WER
        transcription = decode_letter_prediction(
            self.tokenizer, speech_prediction, speech_pred_lengths
        )
        ground_truth_transcription = speech_batch.ground_truth_strings

        return {
            "letter_prediction": speech_prediction.detach().to("cpu"),
            "transcription": transcription,
            "ground_truth": ground_truth_transcription,
        }

    def test_epoch_end(self, outputs: List[Any]) -> None:
        if isinstance(outputs[0], dict):
            raise ValueError("did not expect only one test set")

        # log dict
        result_dict = {}
        num_speaker_sets = 0

        # speech test
        for idx, test_dict_list in enumerate(outputs):
            test_name = self.test_names[idx]

            sample = test_dict_list[0]

            if "letter_prediction" in sample:
                result = calculate_wer_on_collected_output(
                    test_dict_list,
                    log_to_dir=pathlib.Path.cwd() / f"wer_{test_name}",
                    vocab_map=self.tokenizer.vocabulary_dictionary(),
                )
                result_dict[f"test_wer_{test_name}"] = result
            elif "embedding" in sample:
                result = evaluate_embeddings(
                    self.evaluator,
                    test_dict_list,
                    self.speaker_test_pairs[num_speaker_sets],
                    False,
                )
                num_speaker_sets += 1

                result_dict[f"test_eer_{test_name}"] = result["eer"]

        # log results
        self.log_dict(
            result_dict,
            on_epoch=True,
            prog_bar=True,
        )


def log_transcription_progress(
    module: MtSpeechSpeakerLightningModule,
    current_training_batch: SpeechRecognitionDataBatch,
):
    if module.tracking_transcription is None:
        module.tracking_transcription = current_training_batch.ground_truth_strings[0]
        module.tracking_audio_sample = t.clone(
            current_training_batch.audio_input[
                0,
            ]
        ).detach()
        module.tracking_sequence_length = current_training_batch.audio_input_lengths[0]

        if hasattr(module.logger, "experiment") and hasattr(
            module.logger.experiment, "log_text"
        ):
            module.logger.experiment.log_text(
                f"ground_truth={module.tracking_transcription}",
            )
            module.logger.experiment.log_audio(
                module.tracking_audio_sample.cpu().numpy(),
                sample_rate=16000,
                file_name="ground_truth.wav",
            )

    with t.no_grad():
        tmp_training = module.training

        module.training = False
        return_dict = module.forward(
            module.tracking_audio_sample,
            [module.tracking_sequence_length],
            skip_speaker=True,
        )

        module.training = tmp_training

        vocabulary_prediction = return_dict["vocabulary_prediction"]
        vocab_lengths = return_dict["vocab_lengths"]

        # calculate transcription
        transcription = decode_letter_prediction(
            module.tokenizer, vocabulary_prediction, vocab_lengths
        )[0]

        if hasattr(module.logger, "experiment") and hasattr(
            module.logger.experiment, "log_text"
        ):
            module.logger.experiment.log_text(
                f"`{transcription}` len={len(transcription)}", step=module.global_step
            )
