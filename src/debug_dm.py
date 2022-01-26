########################################################################################
#
# This file is the main entrypoint of the train/eval loop based on the
# hydra configuration.
#
# Author(s): Anonymous
########################################################################################

import logging
import time

from abc import abstractmethod
from typing import Any, List, Union

import numpy as np

import torch as t
import pytorch_lightning as pl

from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.modules import LibriSpeechDataModule, VoxCelebDataModule
from src.main import construct_data_module
from src.data.batches import SpeechRecognitionDataBatch, SpeakerClassificationDataBatch

################################################################################
# stat collector interface


class StatCollector:
    @abstractmethod
    def _clear(self):
        pass

    @abstractmethod
    def _pass_batch(self, x: Any):
        pass

    @abstractmethod
    def _aggregate_batches(self) -> str:
        pass

    def summarize(self, dataloader: Union[DataLoader, List[DataLoader]]) -> str:
        self._clear()

        if isinstance(dataloader, list):
            dl_list = dataloader
        else:
            dl_list = [dataloader]

        start_time = time.time()

        num_batches = 0
        for dl in dl_list:
            for x in tqdm(dl):
                num_batches += 1
                self._pass_batch(x)

        end_time = time.time()

        result_string = ""
        result_string += f"total loop time: {round(end_time-start_time, 2)} seconds\n"
        result_string += f"{num_batches=}\n"
        result_string += self._aggregate_batches()

        return result_string

    @staticmethod
    def _stat_string(values: List[Union[float, int]]) -> str:
        min = np.min(values)
        max = np.max(values)
        mean = round(np.mean(values), 2)
        std = round(np.std(values), 2)
        sum = np.sum(values)

        return f"{min=} {max=} {mean=} {std=} {sum=}"


class LibriSpeechStatCollector(StatCollector):
    def _clear(self):
        self.bs = []
        self.frames = []
        self.gt_lengths = []

    def _pass_batch(self, x: Any):
        assert isinstance(x, SpeechRecognitionDataBatch)

        self.bs.append(x.batch_size)
        self.gt_lengths.extend(x.ground_truth_sequence_length)
        self.frames.extend(x.audio_input_lengths)

    def _aggregate_batches(self) -> str:
        return (
            f"number of samples: {np.sum(self.bs)}\n"
            f"batch size; min={min(self.bs)} max={max(self.bs)}\n"
            f"frame length; {self._stat_string(self.frames)}\n"
            f"audio length (seconds); {self._stat_string(np.array(self.frames) / 16_000)}\n"
            f"gt length; {self._stat_string(self.gt_lengths)}\n"
        )


class VoxcelebStatsCollector(StatCollector):
    def _clear(self):
        self.bs = []
        self.frames = []

    def _pass_batch(self, x: Any):
        assert isinstance(x, SpeakerClassificationDataBatch)

        self.bs.append(x.batch_size)
        self.frames.extend(x.audio_input_lengths)

    def _aggregate_batches(self) -> str:
        return (
            f"number of samples: {np.sum(self.bs)}\n"
            f"batch size; min={min(self.bs)} max={max(self.bs)}\n"
            f"frame length; {self._stat_string(self.frames)}\n"
            f"audio length (seconds); {self._stat_string(np.array(self.frames)/16_000)}\n"
        )


########################################################################################
# implement the main script

log = logging.getLogger(__name__)


def run_debug_dm_script(cfg: DictConfig):
    # print config
    print(OmegaConf.to_yaml(cfg))
    print(f"PyTorch version is {t.__version__}")
    print(f"PyTorch Lightning version is {pl.__version__}")

    # construct data module
    dm = construct_data_module(cfg)

    for x in dm.train_dataloader():
        print(x)
        break

    # construct the stat collector
    if isinstance(dm, LibriSpeechDataModule):
        collector = LibriSpeechStatCollector()
    elif isinstance(dm, VoxCelebDataModule):
        collector = VoxcelebStatsCollector()
    else:
        raise ValueError(f"cannot collect stats over {dm}")

    # loop over the data while collecting statistics and print the aggregated results
    print("### train data ###")
    print(collector.summarize(dm.train_dataloader()))

    print("### val data ###")
    print(collector.summarize(dm.val_dataloader()))

    print("### test data ###")
    print(collector.summarize(dm.test_dataloader()))
