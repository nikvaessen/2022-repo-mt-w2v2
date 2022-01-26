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
VOX2_SPLIT="$TEMP_FOLDER"/vox2_split
VOX2_SPLIT_ALL="$TEMP_FOLDER"/vox2_split_all
VOX2_SPLIT_HARD="$TEMP_FOLDER"/vox2_split_hard


if [ -d "$VOX2_SPLIT" ]; then
    rm -r "$VOX2_SPLIT"
fi
if [ -d "$VOX2_SPLIT_ALL" ]; then
    rm -r "$VOX2_SPLIT_ALL"
fi
if [ -d "$VOX2_SPLIT_HARD" ]; then
    rm -r "$VOX2_SPLIT_HARD"
fi

# (original test set)
echo "creating train/val/test split"
poetry run python split_voxceleb.py \
    --root_folder "$TEMP_FOLDER"/extracted_voxceleb \
    --output_folder "$VOX2_SPLIT" \
    --test_trials_path "$DATA_FOLDER"/voxceleb_meta/veri_test2.txt \
    --train_voxceleb1_dev false \
    --train_voxceleb2_dev true \
    --val_split_mode equal \
    --val_ratio 0.01

# (extended test set)
echo "creating extended test split"
poetry run python split_voxceleb.py \
    --root_folder "$TEMP_FOLDER"/extracted_voxceleb \
    --output_folder "$VOX2_SPLIT_ALL" \
    --test_trials_path "$DATA_FOLDER"/voxceleb_meta/list_test_all2.txt \
    --train_voxceleb1_dev false \
    --train_voxceleb2_dev true \
    --val_split_mode equal \
    --val_ratio 0.01

# (hard test set)
echo "creating hard test split"
poetry run python split_voxceleb.py \
    --root_folder "$TEMP_FOLDER"/extracted_voxceleb \
    --output_folder "$VOX2_SPLIT_HARD" \
    --test_trials_path "$DATA_FOLDER"/voxceleb_meta/list_test_hard2.txt \
    --train_voxceleb1_dev false \
    --train_voxceleb2_dev true \
    --val_split_mode equal \
    --val_ratio 0.01

# prepare writing shards
SHARD_FOLDER=$DATA_FOLDER/vox2_shards
SHARD_FOLDER_ALL=$DATA_FOLDER/vox2_shards_all
SHARD_FOLDER_HARD=$DATA_FOLDER/vox2_shards_hard
 
mkdir -p "$SHARD_FOLDER" "$SHARD_FOLDER_ALL" "$SHARD_FOLDER_HARD"

# generate val trials
echo "generating val trial list"
poetry run python generate_trials.py \
    --data_folder "$VOX2_SPLIT"/val \
    --vox_meta_path "$DATA_FOLDER/voxceleb_meta/vox2_meta.csv" \
    --save_path "$SHARD_FOLDER"/val_trials.txt \
    --num_pairs 30000 \
    --ensure_same_sex_trials true

cp "$SHARD_FOLDER"/val_trials.txt "$SHARD_FOLDER_ALL"/val_trials.txt
cp "$SHARD_FOLDER"/val_trials.txt "$SHARD_FOLDER_HARD"/val_trials.txt

# write train shards
echo "writing train shards"
poetry run python write_voxceleb_shards.py \
    --root_data_path "$VOX2_SPLIT"/train \
    --output_folder "$SHARD_FOLDER"/train \
    --name train \
    --compress false \
    --samples_per_shard 5000 \
    --sequential_same_speaker_samples 1 \
    --min_unique_speakers_per_shard 500 \
    --ensure_all_data_in_shards true \
    --discard_partial_shards true

# write val shards
echo "writing val shards"
poetry run python write_voxceleb_shards.py \
    --root_data_path "$VOX2_SPLIT"/val \
    --output_folder "$SHARD_FOLDER"/val \
    --name val \
    --compress false \
    --samples_per_shard 5000 \
    --sequential_same_speaker_samples 1 \
    --min_unique_speakers_per_shard 100 \
    --ensure_all_data_in_shards true \
    --discard_partial_shards false

# write test (original) shards
echo "writing test shards"
poetry run python write_voxceleb_shards.py \
    --root_data_path "$VOX2_SPLIT"/test \
    --output_folder "$SHARD_FOLDER"/test \
    --name test \
    --compress false \
    --samples_per_shard 5000 \
    --sequential_same_speaker_samples 1 \
    --min_unique_speakers_per_shard 1 \
    --ensure_all_data_in_shards true \
    --discard_partial_shards false 

# write test (extended) shards
echo "writing extended test shards"
ln -s "$SHARD_FOLDER"/train "$SHARD_FOLDER_ALL"/train
ln -s "$SHARD_FOLDER"/val "$SHARD_FOLDER_ALL"/val

poetry run python write_voxceleb_shards.py \
    --root_data_path "$VOX2_SPLIT_ALL"/test \
    --output_folder "$SHARD_FOLDER_ALL"/test \
    --name test \
    --compress false \
    --samples_per_shard 5000 \
    --sequential_same_speaker_samples 1 \
    --min_unique_speakers_per_shard 1 \
    --ensure_all_data_in_shards true \
    --discard_partial_shards false 


# write test (hard) shards
echo "writing hard test shards"
ln -s "$SHARD_FOLDER"/train "$SHARD_FOLDER_HARD"/train
ln -s "$SHARD_FOLDER"/val "$SHARD_FOLDER_HARD"/val

poetry run python write_voxceleb_shards.py \
    --root_data_path "$VOX2_SPLIT_HARD"/test \
    --output_folder "$SHARD_FOLDER_HARD"/test \
    --name test \
    --compress false \
    --samples_per_shard 5000 \
    --sequential_same_speaker_samples 1 \
    --min_unique_speakers_per_shard 1 \
    --ensure_all_data_in_shards true \
    --discard_partial_shards false 
