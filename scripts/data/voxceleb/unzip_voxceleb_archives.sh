#! /usr/bin/env bash
set -e

### set environment variables
source ../../../.env 2> /dev/null || source .env

UNZIP_DIR="$TEMP_FOLDER"/extracted_voxceleb

# voxceleb1 train
if [ -f "$DATA_FOLDER"/voxceleb_archives/vox1_dev_wav.zip ]; then
    echo "unzipping $DATA_FOLDER/voxceleb_archives/vox1_dev_wav.zip"
    NUM_FILES=$(zipinfo -h "$DATA_FOLDER"/voxceleb_archives/vox1_dev_wav.zip | grep -oiP '(?<=entries: )[[:digit:]]+')
    mkdir -p "$UNZIP_DIR"/voxceleb1/train
    unzip -o "$DATA_FOLDER"/voxceleb_archives/vox1_dev_wav.zip \
        -d "$UNZIP_DIR"/voxceleb1/train | tqdm --total "$NUM_FILES" >> /dev/null
else
   echo "file $DATA_FOLDER/voxceleb_archives/vox1_dev_wav.zip does not exist."
fi

# voxceleb1 test
if [ -f "$DATA_FOLDER"/voxceleb_archives/vox1_test_wav.zip ]; then
    echo "unzipping $DATA_FOLDER/voxceleb_archives/vox1_test_wav.zip"
    NUM_FILES=$(zipinfo -h "$DATA_FOLDER"/voxceleb_archives/vox1_test_wav.zip | grep -oiP '(?<=entries: )[[:digit:]]+')
    mkdir -p "$UNZIP_DIR"/voxceleb1/test
    unzip -o "$DATA_FOLDER"/voxceleb_archives/vox1_test_wav.zip \
        -d "$UNZIP_DIR"/voxceleb1/test | tqdm --total "$NUM_FILES" >> /dev/null
else
   echo "file $DATA_FOLDER/voxceleb_archives/vox1_test_wav.zip does not exist."
fi

# voxceleb2 train
if [ -f "$DATA_FOLDER"/voxceleb_archives/vox2_dev_wav.zip ]; then
    echo "unzipping $DATA_FOLDER/voxceleb_archives/vox2_dev_wav.zip"
    NUM_FILES=$(zipinfo -h "$DATA_FOLDER"/voxceleb_archives/vox2_dev_wav.zip | grep -oiP '(?<=entries: )[[:digit:]]+')
    mkdir -p "$UNZIP_DIR"/voxceleb2/train
    unzip -o "$DATA_FOLDER"/voxceleb_archives/vox2_dev_wav.zip \
        -d "$UNZIP_DIR"/voxceleb2/train | tqdm --total "$NUM_FILES" >> /dev/null
else
   echo "file $DATA_FOLDER/voxceleb_archives/vox2_dev_wav.zip does not exist."
fi

# voxceleb2 test
if [ -f "$DATA_FOLDER"/voxceleb_archives/vox2_test_wav.zip ]; then
    echo "unzipping $DATA_FOLDER/voxceleb_archives/vox2_test_wav.zip"
    NUM_FILES=$(zipinfo -h "$DATA_FOLDER"/voxceleb_archives/vox2_test_wav.zip | grep -oiP '(?<=entries: )[[:digit:]]+')
    mkdir -p "$UNZIP_DIR"/voxceleb2/test
    unzip -o "$DATA_FOLDER"/voxceleb_archives/vox2_test_wav.zip \
        -d "$UNZIP_DIR"/voxceleb2/test | tqdm --total "$NUM_FILES" >> /dev/null
else
   echo "File $DATA_FOLDER/voxceleb_archives/vox2_test_wav.zip does not exist."
fi


