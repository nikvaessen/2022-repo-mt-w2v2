################################################################################
#
# This file implements the interface of different kinds of
# data modules.
#
# Author(s): Anonymous
################################################################################

from abc import abstractmethod, ABCMeta
from typing import List

from pytorch_lightning import LightningDataModule

################################################################################
# define a data module with a summarization method
from src.evaluation.speaker.speaker_recognition_evaluator import EvaluationPair
from src.evaluation.speech.tokenizer.base import BaseTokenizer


class SummarizableLightningDataModule(LightningDataModule, metaclass=ABCMeta):
    @abstractmethod
    def summary(self):
        pass


################################################################################
# implement a data module for speech (ASR) data


class SpeechLightningDataModule(SummarizableLightningDataModule):
    @property
    @abstractmethod
    def vocabulary(self) -> List[str]:
        pass

    @property
    @abstractmethod
    def tokenizer(self) -> BaseTokenizer:
        pass


################################################################################
# implement a data module for speaker recognition data


class SpeakerLightningDataModule(LightningDataModule):
    @property
    @abstractmethod
    def num_speakers(self) -> int:
        pass

    @property
    @abstractmethod
    def val_pairs(self) -> List[EvaluationPair]:
        pass

    @property
    @abstractmethod
    def test_pairs(self) -> List[EvaluationPair]:
        pass

    @property
    @abstractmethod
    def test_names(self) -> List[str]:
        pass
