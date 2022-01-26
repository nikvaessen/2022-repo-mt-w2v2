################################################################################
#
# This file provides a datamodule which joins the voxceleb and librispeech
# dataset in a multi-task learning setting.
#
# Author(s): Anonymous
################################################################################

from dataclasses import dataclass
from typing import List, Optional

from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS

from src.data.loading_config import SpeechDataLoaderConfig, SpeakerDataLoaderConfig
from src.data.modules import (
    LibriSpeechDataModuleConfig,
    VoxCelebDataModuleConfig,
    LibriSpeechDataModule,
    VoxCelebDataModule,
)
from src.data.modules.abstract import (
    SpeakerLightningDataModule,
    SpeechLightningDataModule,
)
from src.data.pipeline.base import Preprocessor
from src.evaluation.speaker.speaker_recognition_evaluator import EvaluationPair
from src.evaluation.speech.tokenizer import BaseTokenizer


################################################################################
# implement the data module


@dataclass
class JointLibrispeechVoxcelebDataModuleConfig:
    ls_cfg: LibriSpeechDataModuleConfig
    vc_cfg: VoxCelebDataModuleConfig

    test_voxceleb: bool
    test_librispeech: bool


class JointLibrispeechVoxcelebDataModule(
    SpeakerLightningDataModule, SpeechLightningDataModule
):
    def __init__(
        self,
        cfg: JointLibrispeechVoxcelebDataModuleConfig,
        speaker_dl_cfg: SpeakerDataLoaderConfig,
        speaker_train_pipeline: List[Preprocessor],
        speaker_val_pipeline: List[Preprocessor],
        speaker_test_pipeline: List[Preprocessor],
        speech_dl_cfg: SpeechDataLoaderConfig,
        tokenizer: BaseTokenizer,
        speech_train_pipeline: List[Preprocessor],
        speech_val_pipeline: List[Preprocessor],
        speech_test_pipeline: List[Preprocessor],
    ):
        super().__init__()

        self.cfg = cfg

        self.ls_dm = LibriSpeechDataModule(
            cfg=self.cfg.ls_cfg,
            dl_cfg=speech_dl_cfg,
            tokenizer=tokenizer,
            train_pipeline=speech_train_pipeline,
            val_pipeline=speech_val_pipeline,
            test_pipeline=speech_test_pipeline,
        )
        self.vox_dm = VoxCelebDataModule(
            cfg=self.cfg.vc_cfg,
            dl_cfg=speaker_dl_cfg,
            train_pipeline=speaker_train_pipeline,
            val_pipeline=speaker_val_pipeline,
            test_pipeline=speaker_test_pipeline,
        )

    @property
    def num_speakers(self) -> int:
        return self.vox_dm.num_speakers

    @property
    def val_pairs(self) -> List[EvaluationPair]:
        return self.vox_dm.val_pairs

    @property
    def test_pairs(self) -> List[List[EvaluationPair]]:
        return self.vox_dm.test_pairs

    @property
    def test_names(self) -> List[str]:
        names = []

        if self.cfg.test_librispeech:
            names += ["clean", "other"]

        if self.cfg.test_voxceleb:
            names += self.vox_dm.test_names

        return names

    @property
    def vocabulary(self) -> List[str]:
        return self.ls_dm.vocabulary

    @property
    def tokenizer(self) -> BaseTokenizer:
        return self.ls_dm.tokenizer

    def prepare_data(self) -> None:
        self.ls_dm.prepare_data()
        self.vox_dm.prepare_data()

    def setup(self, stage: Optional[str] = None) -> None:
        self.ls_dm.setup()
        self.vox_dm.setup()

    def summary(self):
        self.ls_dm.summary()
        self.vox_dm.summary()

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        vox_dm_loader = self.vox_dm.train_dataloader().repeat().__iter__()
        ls_dm_loader = self.ls_dm.train_dataloader().repeat().__iter__()

        while True:
            speaker_batch = next(vox_dm_loader)
            speech_batch = next(ls_dm_loader)

            yield {"speech": speech_batch, "speaker": speaker_batch}

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return [] + self.ls_dm.val_dataloader() + [self.vox_dm.val_dataloader()]

    def test_dataloader(self) -> EVAL_DATALOADERS:
        loaders = []

        if self.cfg.test_librispeech:
            loaders += self.ls_dm.test_dataloader()

        if self.cfg.test_voxceleb:
            loaders += (
                [self.vox_dm.test_dataloader()]
                if len(self.vox_dm.additional_test_sets) == 0
                else self.vox_dm.test_dataloader()
            )

        return loaders

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        raise NotImplemented()
