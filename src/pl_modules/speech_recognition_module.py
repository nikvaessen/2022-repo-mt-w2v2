################################################################################
#
# Define a base lightning module for a speech recognition network.
#
# Author(s): Anonymous
################################################################################

import json
import logging
import pathlib

from abc import abstractmethod
from typing import Callable, Optional, List, Dict, Tuple

import numpy
import torch.nn
import torchmetrics

import torch as t
import wandb

from pytorch_lightning import LightningModule
from pytorch_lightning.loggers import WandbLogger

from pytorch_lightning.utilities.types import STEP_OUTPUT
from omegaconf import DictConfig

from src.data.batches import SpeechRecognitionDataBatch
from src.evaluation.speech.tokenizer.base import BaseTokenizer
from src.evaluation.speech.wer import calculate_wer
from src.optim.loss.ctc_loss import CtcLoss
from src.pl_modules import BaseLightningModule

################################################################################
# Definition of speaker recognition API

# A logger for this file

log = logging.getLogger(__name__)


class SpeechRecognitionLightningModule(BaseLightningModule):
    def __init__(
        self,
        hyperparameter_config: DictConfig,
        loss_fn_constructor: Callable[[], Callable[[t.Tensor, t.Tensor], t.Tensor]],
        tokenizer: BaseTokenizer,
        auto_lr_find: Optional[float] = None,
    ):
        super().__init__(hyperparameter_config, loss_fn_constructor, auto_lr_find)

        if not isinstance(self.loss_fn, CtcLoss):
            raise ValueError(
                f"expected loss class {CtcLoss}, " f"got {self.loss_fn.__class__}"
            )

        # required for decoding to text
        self.tokenizer = tokenizer
        self.vocab_size = tokenizer.vocabulary_size()

        # keep track of metrics
        self.metric_train_loss = torchmetrics.MeanMetric()
        self.metric_train_wer = torchmetrics.MeanMetric()

        self.metric_val_loss_clean = torchmetrics.MeanMetric()
        self.metric_val_loss_other = torchmetrics.MeanMetric()

        # set on first call to self#_log_transcription_progress step
        self.tracking_audio_sample: t.Tensor = None
        self.tracking_transcription: str = None
        self.tracking_sequence_length: int = None

    @abstractmethod
    def compute_embedding_sequence(
        self, input_tensor: t.Tensor, lengths: List[int]
    ) -> Tuple[t.Tensor, List[int]]:
        # transform:
        # 1) input_tensor with shape [BATCH_SIZE, NUM_SAMPLES]
        # 2) where 0:lengths[BATCH_IDX] are non-padded frames
        # into:
        # 1) an embedding of shape [BATCH_SIZE, SEQUENCE_LENGTH, EMBEDDING_SIZE]
        # 2) a list of lengths which represents frames which are (non-padded)
        #    lengths (index 0:length_value is non-padded)
        pass

    @abstractmethod
    def compute_vocabulary_prediction(
        self, embedding_tensor: t.Tensor, lengths: List[int]
    ) -> Tuple[t.Tensor, List[int]]:
        # transform embedding tensor with shape [BATCH_SIZE, EMBEDDING_SIZE]
        # and list of non-padded range for each batch dimension
        # into a speaker prediction of shape [BATCH_SIZE, SEQUENCE_LENGTH, VOCAB_SIZE]
        # and a list of non-padded range for each batch dimension
        pass

    def forward(self, input_tensor: torch.Tensor, lengths: List[int]):
        embedding, emb_lengths = self.compute_embedding_sequence(input_tensor, lengths)
        prediction, pred_lengths = self.compute_vocabulary_prediction(
            embedding, emb_lengths
        )

        return (embedding, emb_lengths), (prediction, pred_lengths)

    def training_step(
        self,
        batch: SpeechRecognitionDataBatch,
        batch_idx: int,
        optimized_idx: Optional[int] = None,
    ) -> STEP_OUTPUT:
        assert isinstance(batch, SpeechRecognitionDataBatch)
        _, (
            letter_prediction,
            letter_prediction_lengths,
        ) = self.forward(batch.audio_input, batch.audio_input_lengths)

        loss = self.loss_fn(
            predictions=letter_prediction,
            ground_truths=batch.ground_truth,
            prediction_lengths=letter_prediction_lengths,
            ground_truth_lengths=batch.ground_truth_sequence_length,
        )

        with torch.no_grad():
            transcription = decode_letter_prediction(
                self.tokenizer, letter_prediction, letter_prediction_lengths
            )
            ground_truth_transcription = batch.ground_truth_strings
            train_wer = calculate_wer(transcription, ground_truth_transcription)

            # log training loss
            self.metric_train_loss(loss.detach().cpu().item())
            self.metric_train_wer(train_wer)

            if batch_idx % 100 == 0:
                self.log_dict(
                    {
                        "train_loss": self.metric_train_loss.compute(),
                        "train_wer": self.metric_train_wer.compute(),
                    },
                    on_step=True,
                    on_epoch=False,
                    prog_bar=True,
                )

                self.metric_train_loss.reset()
                self.metric_train_wer.reset()

            if self.global_step % 1000 == 0:
                log_transcription_progress(self, batch)

        return loss

    def validation_step(
        self,
        batch: SpeechRecognitionDataBatch,
        batch_idx: int,
        dataloader_idx: Optional[int] = None,
    ) -> Optional[STEP_OUTPUT]:
        assert isinstance(batch, SpeechRecognitionDataBatch)

        _, (
            letter_prediction,
            letter_prediction_lengths,
        ) = self.forward(batch.audio_input, batch.audio_input_lengths)

        loss = self.loss_fn(
            predictions=letter_prediction,
            ground_truths=batch.ground_truth,
            prediction_lengths=letter_prediction_lengths,
            ground_truth_lengths=batch.ground_truth_sequence_length,
        )

        with torch.no_grad():
            transcription = decode_letter_prediction(
                self.tokenizer, letter_prediction, letter_prediction_lengths
            )
            ground_truth_transcription = batch.ground_truth_strings

        return {
            "val_loss": loss.detach().to("cpu"),
            "transcription": transcription,
            "ground_truth": ground_truth_transcription,
        }

    def validation_epoch_end(self, outputs: List[List[Dict[str, List[str]]]]) -> None:
        def collect_loss(output: List[Dict[str, List[str]]]):
            return t.mean(t.Tensor([d["val_loss"] for d in output]))

        wer_clean = calculate_wer_on_collected_output(outputs[0])
        wer_other = calculate_wer_on_collected_output(outputs[1])

        self.log_dict(
            {
                "val_loss_clean": collect_loss(outputs[0]),
                "val_loss_other": collect_loss(outputs[1]),
                "val_wer_clean": wer_clean,
                "val_wer_other": wer_other,
            },
            on_epoch=True,
            prog_bar=True,
        )

    def test_step(
        self,
        batch: SpeechRecognitionDataBatch,
        batch_idx: int,
        dataloader_idx: Optional[int] = None,
    ) -> Optional[STEP_OUTPUT]:
        assert isinstance(batch, SpeechRecognitionDataBatch)

        _, (letter_prediction, letter_prediction_lengths) = self.forward(
            batch.audio_input, batch.audio_input_lengths
        )

        with torch.no_grad():
            transcription = decode_letter_prediction(
                self.tokenizer, letter_prediction, letter_prediction_lengths
            )
            ground_truth_transcription = batch.ground_truth_strings

        return {
            "transcription": transcription,
            "ground_truth": ground_truth_transcription,
        }

    def test_epoch_end(self, outputs: List[List[Dict[str, List[str]]]]) -> None:
        wer_clean = calculate_wer_on_collected_output(outputs[0])
        wer_other = calculate_wer_on_collected_output(outputs[1])

        self.log_dict({"test_wer_clean": wer_clean, "test_wer_other": wer_other})


########################################################################################
# utility methods


def calculate_wer_on_collected_output(
    output: List[Dict[str, List[str]]],
    log_to_dir: pathlib.Path = None,
    vocab_map: Dict[str, int] = None,
):
    transcriptions = []
    ground_truths = []
    letter_prediction_tensors = []

    for d in output:
        transcriptions.extend(d["transcription"])
        ground_truths.extend(d["ground_truth"])

        if "letter_prediction" in d:
            letter_prediction_tensors.append(d["letter_prediction"])

    wer = calculate_wer(transcriptions, ground_truths)

    if log_to_dir is not None:
        log_to_dir.mkdir(exist_ok=True, parents=True)

        with (log_to_dir / "ground_truths.txt").open("w") as f:
            f.writelines("\n".join(ground_truths))

        with (log_to_dir / "predictions.txt").open("w") as f:
            f.writelines("\n".join(transcriptions))

        with (log_to_dir / "wer.txt").open("w") as f:
            f.write(f"wer={wer}\n")

        if len(letter_prediction_tensors) > 0:
            (log_to_dir / "tensors").mkdir(exist_ok=True, parents=True)
            (log_to_dir / "ndarrays").mkdir(exist_ok=True, parents=True)

            with (log_to_dir / "vocabulary.json").open("w") as f:
                json.dump(vocab_map, f)

            numpy.savez_compressed(
                str(log_to_dir / "letter_predictions.npz"),
                **{
                    str(idx): tensor.numpy()
                    for idx, tensor in enumerate(letter_prediction_tensors)
                },
            )
            for idx, tensor in enumerate(letter_prediction_tensors):
                torch.save(tensor, str(log_to_dir / "tensors" / f"{idx}.pt"))
                numpy.save(str(log_to_dir / "ndarrays" / f"{idx}.npy"), tensor.numpy())

    return wer


def decode_letter_prediction(
    tokenizer: BaseTokenizer, letter_prediction: t.Tensor, lengths: List[int]
) -> List[str]:
    # letter prediction has shape [BATCH_SIZE, MAX_SEQUENCE_LENGTH
    batch_size = letter_prediction.shape[0]
    transcriptions = []

    for bs in range(0, batch_size):
        batch_seq = letter_prediction[bs, 0 : lengths[bs], :]

        highest_letter_idx = t.argmax(batch_seq, dim=1)

        transcription = tokenizer.decode_tensor(highest_letter_idx)
        transcriptions.append(transcription)

    return transcriptions


def log_transcription_progress(
    module: LightningModule,
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

        if isinstance(module.logger, WandbLogger):
            module.logger.experiment.log(
                {
                    "sample_input_audio": wandb.Audio(
                        module.tracking_audio_sample.cpu().numpy(),
                        sample_rate=16000,
                    )
                }
            )

    with torch.no_grad():
        # compute embedding
        embedding, embedding_lengths = module.compute_embedding_sequence(
            module.tracking_audio_sample, [module.tracking_sequence_length]
        )

        # compute prediction
        (
            letter_prediction,
            letter_prediction_length,
        ) = module.compute_vocabulary_prediction(embedding, embedding_lengths)

        # calculate transcription
        transcription = decode_letter_prediction(
            module.tokenizer, letter_prediction, letter_prediction_length
        )[0]

        if isinstance(module.logger, WandbLogger):
            module.logger.experiment.log(
                {
                    "transcription_progress": wandb.Table(
                        columns=["ground_truth", "prediction"],
                        data=[
                            [
                                module.tracking_transcription,
                                transcription,
                            ]
                        ],
                    )
                }
            )
