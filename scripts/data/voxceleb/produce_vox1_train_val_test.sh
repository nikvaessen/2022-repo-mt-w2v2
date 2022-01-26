#! /usr/bin/env bash
set -e

SCRIPT_DIR=$(dirname -- "$(readlink -f -- "${BASH_SOURCE[0]}")")
cd "$SCRIPT_DIR"

### set environment variables
source ../../../.env 2> /dev/null || source .env

# extract archives
echo "unzipping archives"
poetry run ./unzip_voxceleb_archives.sh

# create train/val/test split
if [ -d "$TEMP_FOLDER"/vox1_split ]; then
    rm -r "$TEMP_FOLDER"/vox1_split
fi

# (oriqginal test set)
echo "creating train/val/test split"
poetry run python split_voxceleb.py \
    --root_folder "$TEMP_FOLDER"/extracted_voxceleb \
    --output_folder "$TEMP_FOLDER"/vox1_split \
    --test_trials_path "$DATA_FOLDER"/voxceleb_meta/veri_test2.txt \
    --train_voxceleb1_dev true \
    --train_voxceleb2_dev false \
    --val_split_mode equal \
    --val_ratio 0.01

# prepare to write shards
SHARD_FOLDER=$DATA_FOLDER/vox1_shards

mkdir -p "$SHARD_FOLDER"

# generate val trials
echo "generating val trial list"
poetry run python generate_trials.py \
    --data_folder "$TEMP_FOLDER"/vox1_split/val \
    --vox_meta_path "$DATA_FOLDER/voxceleb_meta/vox1_meta.csv" \
    --save_path "$SHARD_FOLDER"/val_trials.txt \
    --num_pairs 5000 \
    --ensure_same_sex_trials true


# write train shards
echo "writing train shards"
poetry run python write_voxceleb_shards.py \
    --root_data_path "$TEMP_FOLDER"/vox1_split/train \
    --output_folder "$SHARD_FOLDER"/train \
    --name train \
    --compress false \
    --samples_per_shard 5000 \
    --sequential_same_speaker_samples 1 \
    --min_unique_speakers_per_shard 50 \
    --ensure_all_data_in_shards true \
    --discard_partial_shards false

# write val shards
echo "writing val shards"
poetry run python write_voxceleb_shards.py \
    --root_data_path "$TEMP_FOLDER"/vox1_split/val \
    --output_folder "$SHARD_FOLDER"/val \
    --name val \
    --compress false \
    --samples_per_shard 5000 \
    --sequential_same_speaker_samples 1 \
    --min_unique_speakers_per_shard 50 \
    --ensure_all_data_in_shards true \
    --discard_partial_shards false

# write test (original) shards
echo "writing test shards"
poetry run python write_voxceleb_shards.py \
    --root_data_path "$TEMP_FOLDER"/vox1_split/test \
    --output_folder "$SHARD_FOLDER"/test \
    --name test \
    --compress false \
    --samples_per_shard 5000 \
    --sequential_same_speaker_samples 1 \
    --min_unique_speakers_per_shard 40 \
    --ensure_all_data_in_shards true \
    --discard_partial_shards false