########################################################################################
#
# Script for generating a vocabulary file (letter-based) on the transcriptions
# found in *.trans.txt files in one or more root directories.
#
# Author(s): Anonymous
########################################################################################

import json
import pathlib
import click

########################################################################################
# function which loops over all transcription files


def determine_char_vocabulary(root_folder: pathlib.Path):
    transcription_file_paths = []

    for f in root_folder.rglob("*.trans.txt"):
        transcription_file_paths.append(f)

    vocab = set()

    for transcription_file in transcription_file_paths:
        with transcription_file.open("r") as f:
            lines = [" ".join(line.strip().split(" ")[1:]) for line in f.readlines()]

            for line in lines:
                for char in line:
                    vocab.update(char)

    sorted_vocab = sorted(list(vocab))

    return sorted_vocab


########################################################################################
# entrypoint of script


@click.command()
@click.option("--root_folder", type=pathlib.Path, required=True)
@click.option("--output_json_path", type=pathlib.Path, required=True)
def main(root_folder: pathlib.Path, output_json_path: pathlib.Path):
    char_vocabulary = determine_char_vocabulary(root_folder)

    with output_json_path.open("w") as f:
        json.dump(
            {"vocabulary": char_vocabulary},
            f,
        )


if __name__ == "__main__":
    main()
