################################################################################
#
# Lightning data module for librispeech (using webdataset shards)
#
# Author(s): Anonymous
################################################################################

import pathlib
import json
import random

from typing import Optional, Union, List, Dict, Callable, Generator

import torch as t
import webdataset as wds

from dataclasses import dataclass
from torch.utils.data import DataLoader

from src.data.batches import SpeechRecognitionDataBatch
from src.data.loading_config import SpeechDataLoaderConfig
from src.data.modules.abstract import SpeechLightningDataModule
from src.data.modules.wds_util import init_webdataset
from src.data.pipeline.base import Preprocessor, AudioDataSample
from src.data.pipeline.debug import BatchDebugInfo
from src.evaluation.speech.tokenizer.base import BaseTokenizer
from src.util.config_util import CastingConfig


################################################################################
# config for lightning data module


@dataclass
class LibriSpeechDataModuleConfig(CastingConfig):
    # path to folders containing train, val and test shards
    train_c100_shard_path: pathlib.Path
    train_c360_shard_path: pathlib.Path
    train_o500_shard_path: pathlib.Path
    val_clean_shard_path: pathlib.Path
    val_other_shard_path: pathlib.Path
    test_clean_shard_path: pathlib.Path
    test_other_shard_path: pathlib.Path

    # path to json containing the vocabulary
    vocabulary_json_path: pathlib.Path

    # how to collate the data when creating a batch
    # one of `default` (assumes same size)
    train_collate_fn: str
    val_collate_fn: str
    test_collate_fn: str

    # whether to keep debug info in data pipeline
    # (which can have serious performance slowdown)
    include_debug_info_in_data_pipeline: bool

    # train set options"
    # 960h: clean-100, clean-360 and other-500 subsets
    # 100h: clean-100 subset
    train_set: str


################################################################################
# data module


class LibriSpeechDataModule(SpeechLightningDataModule):
    def __init__(
        self,
        cfg: LibriSpeechDataModuleConfig,
        dl_cfg: SpeechDataLoaderConfig,
        tokenizer: BaseTokenizer,
        train_pipeline: List[Preprocessor],
        val_pipeline: List[Preprocessor],
        test_pipeline: List[Preprocessor],
    ):
        super().__init__()

        self.cfg = cfg
        self.dl_cfg = dl_cfg
        self.train_pipeline = train_pipeline
        self.val_pipeline = val_pipeline
        self.test_pipeline = test_pipeline
        self._tokenizer = tokenizer

        if self.cfg.train_set == "960h":
            self.train_set_pattern = "train_*_*-shard-*.tar*"
        elif self.cfg.train_set == "100h":
            self.train_set_pattern = "train_clean_100-shard-*.tar*"
        else:
            raise ValueError(f"{self.cfg.train_set=} should be one of [100h, 960h]")

        # values set in self#setup()
        self._vocabulary: List[str] = None

        self.train_ds: t.utils.data.Dataset = None

        self.val_ds_clean: t.utils.data.Dataset = None
        self.val_ds_other: t.utils.data.Dataset = None

        self.test_ds_clean: t.utils.data.Dataset = None
        self.test_ds_other: t.utils.data.Dataset = None

    def summary(self):
        print("librispeech is ready for use")

    @property
    def vocabulary(self) -> List[str]:
        if self._vocabulary is None:
            raise ValueError("vocabulary is accessible after setup() is called")

        return self._vocabulary

    @property
    def tokenizer(self) -> BaseTokenizer:
        return self._tokenizer

    def prepare_data(self) -> None:
        pass

    def _verify_tokenizer_matches_vocabulary(self):
        tokenizer_vocab = self._tokenizer.vocabulary_dictionary()
        for char in self.vocabulary:
            if char == " ":
                # space characters are always supported :)
                continue

            if char not in tokenizer_vocab:
                raise ValueError(
                    f"given tokenizer cannot handle char {char} in vocabulary"
                )

    def _load_vocabulary(self):
        with self.cfg.vocabulary_json_path.open("r") as f:
            return json.load(f)["vocabulary"]

    def setup(self, stage: Optional[str] = None) -> None:
        # verify vocabulary
        self._vocabulary = self._load_vocabulary()
        self._verify_tokenizer_matches_vocabulary()

        # setup train dataset
        self.train_ds: wds.Processor = init_webdataset(
            path_to_shards=[
                self.cfg.train_c100_shard_path,
                self.cfg.train_c360_shard_path,
                self.cfg.train_o500_shard_path,
            ],
            pattern=self.train_set_pattern,
            pipeline=self.train_pipeline,
            decode_fn=wds.torch_audio,
            map_decode_fn=self.construct_fn_decoded_dict_to_sample(
                self._tokenizer, self.cfg.include_debug_info_in_data_pipeline
            ),
        )

        # setup validation datasets
        self.val_ds_clean: wds.Processor = init_webdataset(
            path_to_shards=self.cfg.val_clean_shard_path,
            pattern="dev_clean-shard-*.tar*",
            pipeline=self.val_pipeline,
            decode_fn=wds.torch_audio,
            map_decode_fn=self.construct_fn_decoded_dict_to_sample(
                self._tokenizer, self.cfg.include_debug_info_in_data_pipeline
            ),
        )
        self.val_ds_other: wds.Processor = init_webdataset(
            path_to_shards=self.cfg.val_other_shard_path,
            pattern="dev_other-shard-*.tar*",
            pipeline=self.val_pipeline,
            decode_fn=wds.torch_audio,
            map_decode_fn=self.construct_fn_decoded_dict_to_sample(
                self._tokenizer, self.cfg.include_debug_info_in_data_pipeline
            ),
        )

        # setup test datasets
        self.test_ds_clean: wds.Processor = init_webdataset(
            path_to_shards=self.cfg.test_clean_shard_path,
            pattern="test_clean-shard-*.tar*",
            pipeline=self.test_pipeline,
            decode_fn=wds.torch_audio,
            map_decode_fn=self.construct_fn_decoded_dict_to_sample(
                self._tokenizer, self.cfg.include_debug_info_in_data_pipeline
            ),
        )
        self.test_ds_other: wds.Processor = init_webdataset(
            path_to_shards=self.cfg.test_other_shard_path,
            pattern="test_other-shard-*.tar*",
            pipeline=self.test_pipeline,
            decode_fn=wds.torch_audio,
            map_decode_fn=self.construct_fn_decoded_dict_to_sample(
                self._tokenizer, self.cfg.include_debug_info_in_data_pipeline
            ),
        )

    @staticmethod
    def construct_fn_decoded_dict_to_sample(
        tokenizer: BaseTokenizer, add_debug_info: bool
    ):
        def fn(d: dict):
            # create the sample
            sample = AudioDataSample(
                key=d["__key__"],
                audio=d["wav"][0],
                sample_rate=d["wav"][1],
                audio_length_frames=d["wav"][0].shape[-1],
                debug_info=BatchDebugInfo(
                    original_tensor=d["wav"][0],
                    pipeline_progress=[],
                    meta=d["json"],
                )
                if add_debug_info
                else None,
            )

            # get the ground truth
            transcription: str = d["json"]["transcription"]

            # convert transcription into sequence of integers
            transcription_int_sequence = tokenizer.encode_string(transcription)

            if len(transcription_int_sequence.shape) == 0:
                # 1 letter transcriptions
                transcription_int_sequence = t.stack([transcription_int_sequence])

            SpeechRecognitionDataBatch.set_gt_container(
                sample, transcription, transcription_int_sequence
            )

            return sample

        return fn

    def train_dataloader(
        self,
    ) -> Union[DataLoader, List[DataLoader], Dict[str, DataLoader]]:
        return wds.WebLoader(
            self.train_ds.then(
                dynamic_batches(
                    max_samples_in_batch=self.dl_cfg.train_max_num_samples,
                    max_queue_size=self.dl_cfg.max_queue_size,
                    collate_fn=self._determine_collate_fn(self.cfg.train_collate_fn),
                )
            ),
            num_workers=self.dl_cfg.num_workers,
            pin_memory=self.dl_cfg.pin_memory,
            batch_size=None,
        )

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        def wrap_dataset(val_ds):
            return wds.WebLoader(
                val_ds.batched(
                    self.dl_cfg.val_batch_size,
                    collation_fn=self._determine_collate_fn(self.cfg.val_collate_fn),
                ),
                num_workers=self.dl_cfg.num_workers,
                pin_memory=self.dl_cfg.pin_memory,
                batch_size=None,
            )

        return [wrap_dataset(self.val_ds_clean), wrap_dataset(self.val_ds_other)]

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        def wrap_dataset(test_ds):
            return wds.WebLoader(
                test_ds.batched(
                    self.dl_cfg.test_batch_size,
                    collation_fn=self._determine_collate_fn(self.cfg.test_collate_fn),
                ),
                num_workers=self.dl_cfg.num_workers,
                pin_memory=self.dl_cfg.pin_memory,
                batch_size=None,
            )

        return [wrap_dataset(self.test_ds_clean), wrap_dataset(self.test_ds_other)]

    @staticmethod
    def _determine_collate_fn(name: str):
        if name == "default":
            return SpeechRecognitionDataBatch.default_collate_fn
        else:
            raise ValueError(f"cannot determine collate_fn {name}")


################################################################################
# custom preprocessor which creates dynamically-sized batches with a certain
# length of samples


def dynamic_batches(
    max_samples_in_batch: int,
    max_queue_size: int,
    collate_fn: Callable[[List[AudioDataSample]], SpeechRecognitionDataBatch],
):
    def _fn(sample_iterator: Generator[AudioDataSample, None, None]):
        p = DynamicSpeechBatchProcessor(
            max_samples_in_batch=max_samples_in_batch,
            max_queue_size=max_queue_size,
            collate_fn=collate_fn,
        )

        return p(sample_iterator)

    return _fn


class DynamicSpeechBatchProcessor:
    def __init__(
        self,
        max_samples_in_batch: int,
        max_queue_size: int,
        collate_fn: Callable[[List[AudioDataSample]], SpeechRecognitionDataBatch],
    ):
        self.max_samples_in_batch = max_samples_in_batch
        self.max_queue_size = max_queue_size
        self.collate_fn = collate_fn

        self.queue: List[AudioDataSample] = []

    def __call__(
        self, sample_iterator: Generator[AudioDataSample, None, None]
    ) -> Generator[SpeechRecognitionDataBatch, None, None]:
        self.queue.clear()

        for sample in sample_iterator:
            if not isinstance(sample, AudioDataSample):
                raise ValueError(f"batch is expected to be of type {AudioDataSample}")

            self.queue.append(sample)

            if len(self.queue) == self.max_queue_size:
                yield self.get_batch()

        while len(self.queue) > 0:
            yield self.get_batch()

    def get_batch(self) -> SpeechRecognitionDataBatch:
        if len(self.queue) == 0:
            raise ValueError("cannot get a batch while queue is empty")
        if len(self.queue) == 1:
            batch = self.collate_fn(self.queue)
            self.queue.clear()
            return batch

        # sort the queue by length of the samples
        self.queue = sorted(self.queue, key=lambda b: b.audio_length_frames)

        # create state for batch creation
        sample_indexes = []
        current_batch_size = 0

        # select a random sample to start with
        prime_sample_idx = random.randint(0, len(self.queue) - 1)
        prime_sample: AudioDataSample = self.queue[prime_sample_idx]

        sample_indexes.append(prime_sample_idx)
        current_idx_min = prime_sample_idx
        current_idx_max = prime_sample_idx
        current_batch_size += 1
        current_max_sample_length = prime_sample.audio_length_frames
        current_min_sample_length = prime_sample.audio_length_frames

        # keep selecting samples while limiting the distance between
        # `min_sample_length` and `max_sample_length` until adding another sample
        # would results in a batch with more than `max_samples_in_batch`
        # or the whole queue is selected
        while True:
            # retrieve the two potential candidate samples
            candidate_idx_min = current_idx_min - 1
            candidate_idx_max = current_idx_max + 1

            if candidate_idx_min >= 0:
                min_sample: AudioDataSample = self.queue[candidate_idx_min]
            else:
                min_sample = None
            if candidate_idx_max < len(self.queue):
                max_sample: AudioDataSample = self.queue[candidate_idx_max]
            else:
                max_sample = None

            if min_sample is None and max_sample is None:
                break

            # check which candidate minimizes the distance between
            # max_target_length and min_target_length
            min_sample_distance = (
                current_max_sample_length - min_sample.audio_length_frames
                if min_sample is not None
                else float("inf")
            )
            max_sample_distance = (
                max_sample.audio_length_frames - current_min_sample_length
                if max_sample is not None
                else float("inf")
            )

            if min_sample_distance < max_sample_distance:
                # add min_sample to batch
                sample_to_add = min_sample
                sample_idx_to_add = candidate_idx_min
                current_idx_min = candidate_idx_min
            else:
                # add max_sample to batch
                sample_to_add = max_sample
                sample_idx_to_add = candidate_idx_max
                current_idx_max = candidate_idx_max

            # check if adding this sample to the batch
            # would exceed the max length; if it does we break instead
            if (current_batch_size + 1) * max(
                sample_to_add.audio_length_frames, current_max_sample_length
            ) > self.max_samples_in_batch:
                break

            # add the sample to the batch
            sample_indexes.append(sample_idx_to_add)
            current_batch_size += 1
            current_max_sample_length = max(
                current_max_sample_length, sample_to_add.audio_length_frames
            )
            current_min_sample_length = min(
                current_min_sample_length, sample_to_add.audio_length_frames
            )

        # retrieve the samples and remove them from the queue
        # in reverse order (high to low) in order to keep indexes valid
        # as samples are removed
        batch_samples = [
            self.queue.pop(idx) for idx in sorted(sample_indexes, reverse=True)
        ]

        return self.collate_fn(batch_samples)
