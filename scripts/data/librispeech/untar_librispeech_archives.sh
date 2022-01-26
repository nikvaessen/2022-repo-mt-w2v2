#! /usr/bin/env bash
set -e

### set environment variables
source ../../../.env 2> /dev/null || source .env

UNZIP_DIR="$TEMP_FOLDER"/extracted_librispeech
TRAIN_CLEAN_100="$DATA_FOLDER"/librispeech/train-clean-100.tar.gz
TRAIN_CLEAN_360="$DATA_FOLDER"/librispeech/train-clean-360.tar.gz
TRAIN_OTHER_500="$DATA_FOLDER"/librispeech/train-other-500.tar.gz
DEV_CLEAN="$DATA_FOLDER"/librispeech/dev-clean.tar.gz
DEV_OTHER="$DATA_FOLDER"/librispeech/dev-other.tar.gz
TEST_CLEAN="$DATA_FOLDER"/librispeech/test-clean.tar.gz
TEST_OTHER="$DATA_FOLDER"/librispeech/test-other.tar.gz


# train clean 100
if [ -f "$TRAIN_CLEAN_100" ]; then
    echo "extracting $TRAIN_CLEAN_100"
    # NUM_FILES=$(gzip -cd "$TRAIN_CLEAN_100" | tar -tvv | grep -c ^-)
    NUM_FILES=29966 # 29129
    mkdir -p "$UNZIP_DIR"/train_clean_100
    tar xzfv "$TRAIN_CLEAN_100" -C "$UNZIP_DIR"/train_clean_100 | tqdm --total "$NUM_FILES" >> /dev/null
else
   echo "file $TRAIN_CLEAN_100 does not exist."
fi

# train clean 360
if [ -f "$TRAIN_CLEAN_360" ]; then
    echo "extracting $TRAIN_CLEAN_360"
    # NUM_FILES=$(gzip -cd "$TRAIN_CLEAN_360" | tar -tvv | grep -c ^-)
    NUM_FILES=109135 # 106116
    mkdir -p "$UNZIP_DIR"/train_clean_360
    tar xzfv "$TRAIN_CLEAN_360" -C "$UNZIP_DIR"/train_clean_360 | tqdm --total "$NUM_FILES" >> /dev/null
else
   echo "file $TRAIN_CLEAN_360 does not exist."
fi

# train other 500
if [ -f "$TRAIN_OTHER_500" ]; then
    echo "extracting $TRAIN_OTHER_500"
    # NUM_FILES=$(gzip -cd "$TRAIN_OTHER_500" | tar -tvv | grep -c ^-)
    NUM_FILES=155428 # 151477
    mkdir -p "$UNZIP_DIR"/train_other_500
    tar xzfv "$TRAIN_OTHER_500" -C "$UNZIP_DIR"/train_other_500| tqdm --total "$NUM_FILES" >> /dev/null
else
   echo "file $TRAIN_OTHER_500 does not exist."
fi

# dev clean
if [ -f "$DEV_CLEAN" ]; then
    echo "extracting $DEV_CLEAN"
    # NUM_FILES=$(gzip -cd "$DEV_CLEAN" | tar -tvv | grep -c ^-)
    NUM_FILES=2943 # 2805
    mkdir -p "$UNZIP_DIR"/dev_clean
    tar xzfv "$DEV_CLEAN" -C "$UNZIP_DIR"/dev_clean | tqdm --total "$NUM_FILES" >> /dev/null
else
   echo "file $DEV_CLEAN does not exist."
fi

# dev other
if [ -f "$DEV_OTHER" ]; then
    echo "extracting $DEV_OTHER"
    # NUM_FILES=$(gzip -cd "$DEV_OTHER" | tar -tvv | grep -c ^-)
    NUM_FILES=3085 # 2960
    mkdir -p "$UNZIP_DIR"/dev_other
    tar xzfv "$DEV_OTHER" -C "$UNZIP_DIR"/dev_other | tqdm --total "$NUM_FILES" >> /dev/null
else
   echo "file $DEV_OTHER does not exist."
fi

# test clean
if [ -f "$TEST_CLEAN" ]; then
    echo "extracting $TEST_CLEAN"
    # NUM_FILES=$(gzip -cd "$TEST_CLEAN" | tar -tvv | grep -c ^-)
    NUM_FILES=2840 # 2712
    mkdir -p "$UNZIP_DIR"/test_clean
    tar xzfv "$TEST_CLEAN" -C "$UNZIP_DIR"/test_clean| tqdm --total "$NUM_FILES" >> /dev/null
else
   echo "file $TEST_CLEAN does not exist."
fi

# test other
if [ -f "$TEST_OTHER" ]; then
    echo "extracting $TEST_OTHER"
    # NUM_FILES=$(gzip -cd "$TEST_OTHER" | tar -tvv | grep -c ^-)
    NUM_FILES=3158 # 3034
    mkdir -p "$UNZIP_DIR"/test_other
    tar xzfv "$TEST_OTHER" -C "$UNZIP_DIR"/test_other | tqdm --total "$NUM_FILES" >> /dev/null
else
   echo "file $TEST_OTHER does not exist."
fi
