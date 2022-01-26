#! /usr/bin/env bash
set -e

SCRIPT_DIR=$(dirname -- "$(readlink -f -- "${BASH_SOURCE[0]}")")
cd "$SCRIPT_DIR"

### set environment variables
source ../../../.env 2> /dev/null || source .env

# extract archives
echo "unzipping archives"
EXTRACT_DIR="$TEMP_FOLDER"/extracted_librispeech
poetry run ./untar_librispeech_archives.sh

# convert to wav
echo 'converting from flac to wav'
poetry run python librispeech_convert_to_wav.py "$EXTRACT_DIR" --num_workers "$(nproc --all)"

# write shards
SHARD_FOLDER=$DATA_FOLDER/librispeech_shards
SHARD_FOLDER_TRAIN_100=$SHARD_FOLDER/train_clean_100
SHARD_FOLDER_TRAIN_360=$SHARD_FOLDER/train_clean_360
SHARD_FOLDER_TRAIN_500=$SHARD_FOLDER/train_other_500
SHARD_FOLDER_DEV_CLEAN=$SHARD_FOLDER/dev_clean
SHARD_FOLDER_DEV_OTHER=$SHARD_FOLDER/dev_other
SHARD_FOLDER_TEST_CLEAN=$SHARD_FOLDER/test_clean
SHARD_FOLDER_TEST_OTHER=$SHARD_FOLDER/test_other
 
mkdir -p "$SHARD_FOLDER_TRAIN_100" "$SHARD_FOLDER_TRAIN_360" "$SHARD_FOLDER_TRAIN_500" \
    "$SHARD_FOLDER_DEV_CLEAN" "$SHARD_FOLDER_DEV_OTHER" "$SHARD_FOLDER_TEST_CLEAN" "$SHARD_FOLDER_TEST_OTHER"

# train clean 100
echo "writing train shards - clean 100h"
poetry run python write_librispeech_shards.py \
    --root_data_path "$EXTRACT_DIR"/train_clean_100 \
    --output_folder "$SHARD_FOLDER_TRAIN_100" \
    --name train_clean_100 \
    --compress false \
    --samples_per_shard 5000 \

# train clean 360
echo "writing train shards - clean 360H"
poetry run python write_librispeech_shards.py \
    --root_data_path "$EXTRACT_DIR"/train_clean_360 \
    --output_folder "$SHARD_FOLDER_TRAIN_360" \
    --name train_clean_360 \
    --compress false \
    --samples_per_shard 5000 \


# train other 500
echo "writing train shards - other 500H"
poetry run python write_librispeech_shards.py \
    --root_data_path "$EXTRACT_DIR"/train_other_500 \
    --output_folder "$SHARD_FOLDER_TRAIN_500" \
    --name train_other_500 \
    --compress false \
    --samples_per_shard 5000 \


# dev clean
echo "writing dev shards - clean"
poetry run python write_librispeech_shards.py \
    --root_data_path "$EXTRACT_DIR"/dev_clean \
    --output_folder "$SHARD_FOLDER_DEV_CLEAN" \
    --name dev_clean \
    --compress false \
    --samples_per_shard 5000 \



# dev other
echo "writing dev shards - other"
poetry run python write_librispeech_shards.py \
    --root_data_path "$EXTRACT_DIR"/dev_other \
    --output_folder "$SHARD_FOLDER_DEV_OTHER" \
    --name dev_other \
    --compress false \
    --samples_per_shard 5000 \


# test clean
echo "writing test shards - clean"
poetry run python write_librispeech_shards.py \
    --root_data_path "$EXTRACT_DIR"/test_clean \
    --output_folder "$SHARD_FOLDER_TEST_CLEAN" \
    --name test_clean \
    --compress false \
    --samples_per_shard 5000 \

# test other
echo "writing test shards - other"
poetry run python write_librispeech_shards.py \
    --root_data_path "$EXTRACT_DIR"/test_other \
    --output_folder "$SHARD_FOLDER_TEST_OTHER" \
    --name test_other \
    --compress false \
    --samples_per_shard 5000 \

# create vocabulary file
poetry run python generate_vocabulary.py \
  --root_folder "$EXTRACT_DIR" \
  --output_json_path "$SHARD_FOLDER"/vocabulary.json