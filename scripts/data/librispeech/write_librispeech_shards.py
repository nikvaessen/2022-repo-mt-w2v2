################################################################################
#
# This file provides a CLI script for sharding the librispeech data based
# on the webdataset API.
#
# Author(s): Anonymous
################################################################################

import pathlib
import json
import subprocess

import torch
import click
import yaspin

import webdataset as wds
import torchaudio.backend.sox_io_backend as tab

################################################################################
# implement the sharding logic


def write_librispeech_shards(
    librispeech_folder_path: pathlib.Path,
    shards_path: pathlib.Path,
    compress_in_place: bool,
    samples_per_shard: int,
    shard_name_pattern: str = "shard-%06d.tar",
):
    """
    Transform a librispeech-structured folder of .flac files to WebDataset shards.

    :param librispeech_folder_path: folder where extracted librespeech data is located
    :param shards_path: folder to write shards of data to
    :param compress_in_place: boolean value determining whether the shards will
                              be compressed with the `gpig` utility.
    :param samples_per_shard: number of data samples to store in each shards.
    :param shard_name_pattern: pattern of name to give to each shard
    """
    # make sure output folder exist
    shards_path.mkdir(parents=True, exist_ok=True)

    # find all audio files
    audio_files = sorted([f for f in librispeech_folder_path.rglob("*.wav")])

    # store statistics
    all_reader_ids = set()
    all_chapter_ids = set()
    all_keys = set()

    # create tuples
    # (unique_sample_id, transcription string, path_to_audio_file, num_audio_frames)
    data_tuples = []

    for file in audio_files:
        # path should be
        # ${librispeech_folder_path}/<reader_id>/<chapter_id>/<reader_id>-<chapter_id>-<utterance_id>.wav
        reader_id = file.parent.parent.name
        chapter_id = file.parent.name

        # create a unique key for this sample
        key = file.stem

        # store statistics
        all_reader_ids.add(reader_id)
        all_chapter_ids.add(chapter_id)

        if key in all_keys:
            raise ValueError(f"duplicate key {key}")
        else:
            all_keys.add(key)

        # load the transcription of the audio file
        with (file.parent / f"{reader_id}-{chapter_id}.trans.txt").open("r") as f:
            lines = [line.strip() for line in f.readlines()]
            transcription = None

            for line in lines:
                split_line = line.split(" ")
                utterance_key = split_line[0]

                if utterance_key == key:
                    transcription = " ".join(split_line[1:])
                    break

        if transcription is None:
            raise ValueError(f"unable to find transcription for {file}")

        # load num_frames in audio file
        num_frames = tab.info(str(file)).num_frames

        tup = (key, transcription, file, num_frames)
        data_tuples.append(tup)

    # write a meta.json file which contains statistics on the data
    # which will be written to shards
    meta_dict = {
        "reader_ids": list(all_reader_ids),
        "chapter_ids": list(all_chapter_ids),
        "keys": list(all_keys),
        "num_samples": len(data_tuples),
        "num_speakers": len(all_reader_ids),
    }

    with (
        shards_path / f"meta_{'_'.join(shard_name_pattern.split('_')[0:-1])}.json"
    ).open("w") as f:
        json.dump(meta_dict, f)

    # sort the tuples by length of audio file so that a batch does not need
    # a lot of padding
    data_tuples = sorted(data_tuples, key=lambda tupl: tupl[3])

    # write shards
    all_keys = set()
    shards_path.mkdir(exist_ok=True, parents=True)
    pattern = str(shards_path / shard_name_pattern)

    # optionally compress the .tar shards
    def compress(file_name: str):
        if compress_in_place:
            with yaspin.yaspin() as spinner:
                spinner.write(f"> compressing {file_name}")
                subprocess.call(
                    ["pigz", file_name],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )

    with wds.ShardWriter(
        pattern, maxsize=5e9, maxcount=samples_per_shard, post=compress
    ) as sink:
        for key, transcription, f, num_frames in sorted(
            data_tuples, key=lambda t: t[3]
        ):
            # load the audio tensor
            tensor, sample_rate = tab.load(str(f))

            if torch.any(torch.isnan(tensor)):
                raise ValueError(f"NaN value in wav file of {key=} at {f=}")

            # verify key is unique
            assert key not in all_keys
            all_keys.add(key)

            # extract speaker_id, youtube_id and utterance_id from key
            reader_id, chapter_id, utterance_id = key.split("-")

            # read file as binary blob
            with f.open("rb") as handler:
                binary_wav = handler.read()

            # create sample to write
            sample = {
                "__key__": f"{reader_id}/{chapter_id}/{utterance_id}",
                "wav": binary_wav,
                "json": {
                    "reader_id": reader_id,
                    "chapter_id": chapter_id,
                    "utterance_id": utterance_id,
                    "transcription": transcription,
                    "num_frames": num_frames,
                    "sampling_rate": sample_rate,
                },
            }

            # write sample to sink
            sink.write(sample)


################################################################################
# implement the CLI entrypoint


@click.command()
@click.option("--root_data_path", type=pathlib.Path, required=True)
@click.option("--output_folder", type=pathlib.Path, required=True)
@click.option("--name", type=str, required=True)
@click.option("--compress", type=bool, default=False)
@click.option("--samples_per_shard", type=int, default=5000)
def main(
    root_data_path: pathlib.Path,
    output_folder: pathlib.Path,
    compress: bool,
    samples_per_shard: int,
    name: str,
):
    write_librispeech_shards(
        librispeech_folder_path=root_data_path,
        shards_path=output_folder,
        compress_in_place=compress,
        samples_per_shard=samples_per_shard,
        shard_name_pattern=f"{name}-shard-%06d.tar",
    )


if __name__ == "__main__":
    main()
