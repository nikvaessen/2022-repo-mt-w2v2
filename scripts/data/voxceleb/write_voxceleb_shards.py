#!/usr/bin/env python
################################################################################
#
# This file provides a CLI script for sharding the voxceleb data based
# on the webdataset API.
#
# Author(s): Anonymous
################################################################################

import pathlib
import random
import json
import multiprocessing
import subprocess

from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

import click

import webdataset
import torch

from torchaudio.backend.sox_io_backend import load as load_audio

################################################################################
# method to write shards

ID_SEPARATOR = "/"


def write_shards(
    voxceleb_folder_path: pathlib.Path,
    shards_path: pathlib.Path,
    compress_in_place: bool,
    shard_name_pattern: str = "shard-{idx:06d}",
    samples_per_shard: int = 5000,
    sequential_same_speaker_samples: int = 4,
    min_unique_speakers_per_shard: int = 32,
    ensure_all_data_in_shards: bool = False,
    discard_partial_shards: bool = True,
):
    """
    Transform a voxceleb-structured folder of .wav files to WebDataset shards.
    :param voxceleb_folder_path: folder where extracted voxceleb data is located
    :param shards_path: folder to write shards of data to
    :param compress_in_place: boolean value determining whether the shards will
                              be compressed with the `gpig` utility.
    :param samples_per_shard: number of data samples to store in each shards.
    :param shard_name_pattern: pattern of name to give to each shard
    """
    # make sure output folder exist
    shards_path.mkdir(parents=True, exist_ok=True)

    # find all audio files
    audio_files = sorted([f for f in voxceleb_folder_path.rglob("*.wav")])

    # create data dictionary {speaker id: List[file_path, sample_key]}}
    data: Dict[str, List[Tuple[str, str, pathlib.Path]]] = defaultdict(list)

    # track statistics on data
    all_speaker_ids = set()
    all_youtube_ids = set()
    all_sample_ids = set()
    youtube_id_per_speaker = defaultdict(list)
    sample_keys_per_speaker = defaultdict(list)
    num_samples = 0
    all_keys = set()

    for f in audio_files:
        # path should be
        # ${voxceleb_folder_path}/wav/speaker_id/youtube_id/utterance_id.wav
        speaker_id = f.parent.parent.name
        youtube_id = f.parent.name
        utterance_id = f.stem

        # create a unique key for this sample
        key = f"{speaker_id}{ID_SEPARATOR}{youtube_id}{ID_SEPARATOR}{utterance_id}"

        if key in all_keys:
            raise ValueError("found sample with duplicate key")
        else:
            all_keys.add(key)

        # store statistics
        num_samples += 1

        all_speaker_ids.add(speaker_id)
        all_youtube_ids.add(youtube_id)
        all_sample_ids.add(key)

        youtube_id_per_speaker[speaker_id].append(youtube_id)
        sample_keys_per_speaker[speaker_id].append(key)

        # store data in dict
        data[speaker_id].append((key, speaker_id, f))

    # randomly shuffle the list of all samples for each speaker
    for speaker_id in data.keys():
        random.shuffle(data[speaker_id])

    # determine a specific speaker_id label for each speaker_id
    speaker_id_to_idx = {
        speaker_id: idx for idx, speaker_id in enumerate(sorted(all_speaker_ids))
    }

    # write a meta.json file which contains statistics on the data
    # which will be written to shards
    all_speaker_ids = list(all_speaker_ids)
    all_youtube_ids = list(all_youtube_ids)
    all_sample_ids = list(all_sample_ids)

    meta_dict = {
        "speaker_ids": all_speaker_ids,
        "youtube_ids": all_youtube_ids,
        "sample_ids": all_sample_ids,
        "speaker_id_to_idx": speaker_id_to_idx,
        "youtube_ids_per_speaker": youtube_id_per_speaker,
        "sample_ids_per_speaker": sample_keys_per_speaker,
        "num_samples": num_samples,
        "num_speakers": len(all_speaker_ids),
    }

    with (shards_path / "meta.json").open("w") as f:
        json.dump(meta_dict, f)

    # split the data into shards such that each shard has at most
    # `samples_per_shard` samples and that the sequential order in the
    # shard is:
    # 1 = sample of speaker id `i`
    # ...
    # sequential_same_speaker_samples =sample of speaker id `i`
    # sequential_same_speaker_samples + 1 = sample of speaker id `j`
    # etc
    shards_list = []

    def samples_left():
        num_samples_left = sum(len(v) for v in data.values())
        num_valid_speakers = sum(
            len(v) >= sequential_same_speaker_samples for v in data.values()
        )

        # a shard should contain at least 2 different speakers
        if num_valid_speakers >= 2 or ensure_all_data_in_shards:
            return num_samples_left
        else:
            return 0

    def valid_speakers(n: int, previous_id: Optional[str] = None):
        return [k for k in data.keys() if len(data[k]) >= n and k != previous_id]

    def pop_n_samples(
        n: int, current_speakers_in_shard: Set[str], previous_id: Optional[str] = None
    ):
        valid_speaker_ids = valid_speakers(n, previous_id)

        if len(current_speakers_in_shard) < min_unique_speakers_per_shard:
            valid_speaker_ids = [
                sid for sid in valid_speaker_ids if sid not in current_speakers_in_shard
            ]

        if len(valid_speaker_ids) == 0:
            raise ValueError(
                f"shard cannot be guaranteed to have {min_unique_speakers_per_shard=}"
            )

        samples_per_speaker = [len(data[k]) for k in valid_speaker_ids]
        random_speaker_id = random.choices(valid_speaker_ids, samples_per_speaker)[0]
        current_speakers_in_shard.add(random_speaker_id)
        popped_samples = []

        for _ in range(n):
            sample_list = data[random_speaker_id]
            popped_samples.append(
                sample_list.pop(random.randint(0, len(sample_list) - 1))
            )

        return popped_samples, random_speaker_id, current_speakers_in_shard

    # write shards
    while samples_left() > 0:
        shard = []
        speakers_in_shard = set()
        previous = None

        print(
            f"determined shards={len(shards_list):>4}\t"
            f"samples left={samples_left():>9,d}\t"
            f"speakers left="
            f"{len(valid_speakers(sequential_same_speaker_samples, previous)):>4,d}"
        )
        while len(shard) < samples_per_shard and samples_left() > 0:
            samples, previous, speakers_in_shard = pop_n_samples(
                n=sequential_same_speaker_samples,
                current_speakers_in_shard=speakers_in_shard,
                previous_id=previous,
            )
            for key, speaker_id, f in samples:
                shard.append((key, speaker_id_to_idx[speaker_id], f))

        shards_list.append(shard)

    # assert all data is in a shard
    if ensure_all_data_in_shards:
        assert sum(len(v) for v in data.values()) == 0

    # remove any shard which does share the majority amount of samples
    if discard_partial_shards:
        unique_len_count = defaultdict(int)
        for lst in shards_list:
            unique_len_count[len(lst)] += 1

        if len(unique_len_count) > 2:
            raise ValueError("expected at most 2 unique lengths")

        if len(unique_len_count) == 0:
            raise ValueError("expected at least 1 unique length")

        majority_len = -1
        majority_count = -1
        for unique_len, count in unique_len_count.items():
            if count > majority_count:
                majority_len = unique_len
                majority_count = count

        shards_list = [lst for lst in shards_list if len(lst) == majority_len]

    # write shards
    shards_path.mkdir(exist_ok=True, parents=True)

    # seems like disk write speed only allows for 1 process anyway :/
    with multiprocessing.Pool(processes=1) as p:
        for idx, shard_content in enumerate(shards_list):
            args = {
                "shard_name": shard_name_pattern.format(idx=idx),
                "shards_path": shards_path,
                "data_tpl": shard_content,
                "compress": compress_in_place,
            }
            p.apply_async(
                _write_shard,
                kwds=args,
                error_callback=lambda x: print(
                    f"error in apply_async ``_write_shard!\n{x}"
                ),
            )

        p.close()
        p.join()


# function to write shards to disk, used internally
def _write_shard(
    shard_name: str, shards_path: pathlib.Path, data_tpl: List, compress: bool = True
):
    if shard_name.endswith(".tar.gz"):
        # `pigz` will automatically add extension (and would break if it's
        # already there)
        shard_name = shard_name.split(".tar.gz")[0]

    if not shard_name.endswith(".tar"):
        shard_name += ".tar"

    shard_path = str(shards_path / shard_name)
    print(f"writing shard {shard_path}")
    # note that we manually compress with `pigz` which is a lot faster than python
    with webdataset.TarWriter(shard_path) as sink:
        for key, speaker_id_idx, f in data_tpl:
            f: pathlib.Path = f
            # load the audio tensor to verify sample rate
            tensor, sample_rate = load_audio(str(f))

            if torch.any(torch.isnan(tensor)):
                raise ValueError(f"NaN value in wav file of {key=} at {f=}")
            if sample_rate != 16_000:
                raise ValueError(f"audio file {key} has {sample_rate=}")

            # extract speaker_id, youtube_id and utterance_id from key
            speaker_id, youtube_id, utterance_id = key.split(ID_SEPARATOR)

            # read file as binary blob
            with f.open("rb") as handler:
                binary_wav = handler.read()

            # create sample to write
            sample = {
                "__key__": key,
                "wav": binary_wav,
                "json": {
                    "speaker_id": speaker_id,
                    "youtube_id": youtube_id,
                    "utterance_id": utterance_id,
                    "speaker_id_idx": speaker_id_idx,
                    "num_frames": len(tensor.squeeze()),
                    "sampling_rate": sample_rate,
                },
            }

            # write sample to sink
            sink.write(sample)

    if compress:
        subprocess.call(
            ["pigz", shard_path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )


################################################################################
# entrypoint of script


@click.command()
@click.option("--root_data_path", required=True, type=pathlib.Path)
@click.option("--output_folder", required=True, type=pathlib.Path)
@click.option("--name", required=True, type=str)
@click.option("--compress", type=bool, default=True)
@click.option("--samples_per_shard", type=int, default=5000)
@click.option("--sequential_same_speaker_samples", type=int, default=1)
@click.option("--min_unique_speakers_per_shard", type=int, default=100)
@click.option("--ensure_all_data_in_shards", type=bool, default=True)
@click.option("--discard_partial_shards", type=bool, default=False)
def main(
    root_data_path: pathlib.Path,
    output_folder: pathlib.Path,
    name: str,
    compress: bool,
    samples_per_shard: int,
    sequential_same_speaker_samples: int,
    min_unique_speakers_per_shard: int,
    ensure_all_data_in_shards: bool,
    discard_partial_shards: bool,
):
    print(f"{root_data_path=}")
    print(f"{output_folder=}")

    write_shards(
        voxceleb_folder_path=root_data_path,
        shards_path=output_folder,
        compress_in_place=compress,
        shard_name_pattern=f"{name}_shard_" + "{idx:06d}",
        samples_per_shard=samples_per_shard,
        sequential_same_speaker_samples=sequential_same_speaker_samples,
        min_unique_speakers_per_shard=min_unique_speakers_per_shard,
        ensure_all_data_in_shards=ensure_all_data_in_shards,
        discard_partial_shards=discard_partial_shards,
    )


if __name__ == "__main__":
    main()
