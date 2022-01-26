# Multi-task learning with wav2vec2

This repository implements multi-task learning for speech and speaker recognition with wav2vec2.

## Setting up the project dependencies

If poetry is not installed, see https://python-poetry.org/docs/. We also
expect at least python 3.8 on the system. If this is not the case, look into
https://github.com/pyenv/pyenv for an easy tool to install a specific
python version on your system. 


The python dependencies can be installed (in a project-specific virtual environment) by:

```bash
$ ./scripts/setup_dependencies.sh
```

To access the virtual environment, activate it with `poetry shell` or prepend `poetry run` to the command you want to run in the virtual environment.

## Setting up the environment

Copy the example environment variables:

```bash
$ cp .env.example .env 
```

You can than fill in `.env` accordingly. 

## Data preparation

### Voxceleb

#### Collecting the data archives

I've experienced that the download links for voxceleb1/2 can be unstable.
I recommend manually downloading the dataset from the google drive link displayed 
on https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1.html.

You should end up 4 zip files, which are expected to be placed in `$DATA_FOLDER/voxceleb_archives`. 
1. `vox1_dev_wav.zip` 
2. `vox1_test_wav.zip`
3. `vox2_dev_aac.zip`
4. `vox2_test_aac.zip`

You should also download the meta files of voxceleb. You can use 
`./scripts/download/download_voxceleb_meta.sh` to download them
to the expected location `$DATA_FOLDER/voxceleb_meta`.

#### Transforming data to wav

This requires ffmpeg to be installed on the machine. Check with `ffmpeg -version`.
Assuming the voxceleb2 data archives are found at `$DATA_FOLDER/voxceleb_archives/vox2_dev_aac.zip` and `$DATA_FOLDER/voxceleb_archives/vox2_test_aac.zip`, run the following commands, starting
from the root project directory.

```bash
source .env

PDIR=$PWD # folder where this README is located
D=$DATA_FOLDER # location of data - should be set in .env file
WORKERS=$(nproc --all) # number of CPUs available

# extract voxceleb 2 data
cd "$D" || exit
mkdir -p convert_tmp/train convert_tmp/test

unzip voxceleb_archives/vox2_dev_aac.zip -d convert_tmp/train
unzip voxceleb_archives/vox2_test_aac.zip -d convert_tmp/test

# run the conversion script
cd "$PDIR" || exit
poetry run python scripts/voxceleb2_convert_to_wav.py \
    "$D"/convert_tmp \
    --num_workers "$WORKERS"

# rezip the converted data
cd "$D"/convert_tmp/train || exit
zip "$D"/voxceleb_archives/vox2_dev_wav.zip wav -r

cd "$D"/convert_tmp/test || exit
zip "$D"/voxceleb_archives/vox2_test_wav.zip wav -r

# delete the unzipped .m4a files
cd "$D" || exit
rm -r convert_tmp
```

Note that this process can take a few hours on a fast machine and day(s) on a single (slow) cpu.
Make sure to save the `vox2_dev_wav.zip` and `vox2_test_wav.zip` files somewhere secure, so you don't have redo
this process :).

#### Writing shards

After having done the steps above, you can use `scripts/data/voxceleb/produce_vox2_train_val_test3.sh` to:

1. extract the archives (with wav files)
2. create train/val split
3. process data into webdataset shards.

The script should place the data in the correct location for `config/data/mt_vox2_libri.yaml` and `config/data/voxceleb2.yaml`.

### Librispeech

#### Downloading

You can use `scripts/download/download_librispeech.sh` to download the LibriSpeech dataset.

#### Writing shards

You can use `scripts/data/librispeech.produce_librispeech_shards.sh` to:

1. extract the archives
2. convert to wav
3. process data into webdataset shards

The script should place the data in the correct location for `config/data/mt_vox2_libri.yaml` and `config/data/librispeech.yaml`.

## Experiments

The following section provides the commands to reproduce the experiments. You can remove the `hydra/launcher=...` argument if you intend to run the experiment on a machine without SLURM.
If you do intend to use SLURM, you should edit `config/hydra/launcher/slurm.yaml` accordingly (namely, the `partition` parameter is left blank).

If you do not have access to a GPU with 24GB of VRAM, you can reduce the batch size by e.g. a factor 2, 
(See `speaker/dataloader/speaker.yaml` and `speech/dataloader/speech.yaml`) and add, 
e.g. `trainer.accumulate_grad_batches=2`.

### Baseline speech

All experiments were logged at https://wandb.ai/w2v2-mt-learning/baseline-speech.

#### learning rate hparam search

without regularization:
```
python run_speech.py -m +experiment=baseline_speech \
optim.algo.lr=1E-6,1.78E-6,3.16E-6,5.62E-6,1E-5,1.78E-5,3.16E-5,5.62E-5,1E-4,1.78E-4,3.16E-4,5.62E-4,1E-3 \
hydra/launcher=slurm tag=[no_reg,lr_grid]
```

with regularization:
```
python run_speech.py -m +experiment=baseline_speech \
optim.algo.lr=1E-6,1.78E-6,3.16E-6,5.62E-6,1E-5,1.78E-5,3.16E-5,5.62E-5,1E-4,1.78E-4,3.16E-4,5.62E-4,1E-3 \
hydra/launcher=slurm tag=[with_reg,lr_grid] network/regularisation=wav2vec2_default_asr
```

#### longer train session (320k steps):

without regularization:

```
python run_speech.py -m +experiment=baseline_speech \
optim.algo.lr=5.62E-5 trainer.max_steps=320_000 callbacks=speech_default \
hydra/launcher=slurm tag=[no_reg,320k]
```

with regularization:

```
python run_speech.py -m +experiment=baseline_speech \
optim.algo.lr=5.62E-5 trainer.max_steps=320_000 callbacks=speech_default \
hydra/launcher=slurm tag=[with_reg,320k] network/regularisation=wav2vec2_default_asr
```

### baseline speaker

All experiments were logged at https://wandb.ai/w2v2-mt-learning/baseline-speaker.

#### learning rate hparam search

first+cls pooling, 3 seconds

```
python run_speaker.py -m +experiment=baseline_speaker_first+cls \
speaker.dataloader.train_batch_size=66 speaker.pipeline.selector_train.desired_chunk_length_sec=3 \
optim.algo.lr=1E-6,1.78E-6,3.16E-6,5.62E-6,1E-5,1.78E-5,3.16E-5,5.62E-5,1E-4,1.78E-4,3.16E-4,5.62E-4,1E-3 \
hydra/launcher=slurm_11vram tag=[lr_grid,no_reg]
```

first+cls pooling, 9 seconds

```
python run_speaker.py -m +experiment=baseline_speaker_first+cls \
speaker.dataloader.train_batch_size=22 speaker.pipeline.selector_train.desired_chunk_length_sec=9 \
optim.algo.lr=1E-6,1.78E-6,3.16E-6,5.62E-6,1E-5,1.78E-5,3.16E-5,5.62E-5,1E-4,1.78E-4,3.16E-4,5.62E-4,1E-3 \
hydra/launcher=slurm_11vram tag=[lr_grid,no_reg]
```

mean pooling, 3 seconds

```
python run_speaker.py -m +experiment=baseline_speaker_mean \
speaker.dataloader.train_batch_size=66 speaker.pipeline.selector_train.desired_chunk_length_sec=3 \
optim.algo.lr=1E-6,1.78E-6,3.16E-6,5.62E-6,1E-5,1.78E-5,3.16E-5,5.62E-5,1E-4,1.78E-4,3.16E-4,5.62E-4,1E-3 \
hydra/launcher=slurm_11vram tag=[lr_grid,no_reg]
```

mean pooling, 3 seconds

```
python run_speaker.py -m +experiment=baseline_speaker_mean \
speaker.dataloader.train_batch_size=22 speaker.pipeline.selector_train.desired_chunk_length_sec=9 \
optim.algo.lr=1E-6,1.78E-6,3.16E-6,5.62E-6,1E-5,1.78E-5,3.16E-5,5.62E-5,1E-4,1.78E-4,3.16E-4,5.62E-4,1E-3 \
hydra/launcher=slurm_11vram tag=[lr_grid,no_reg]
```

#### longer train session (320k steps):

first+cls pooling, 3 seconds
```
python run_speaker.py -m +experiment=baseline_speaker_first+cls \
speaker.dataloader.train_batch_size=66 speaker.pipeline.selector_train.desired_chunk_length_sec=3 \
optim.algo.lr=3.16E-5 callbacks=speaker_default trainer.max_steps=320_000 \
hydra/launcher=slurm tag=[lr_grid,no_reg,320k]
```

first+cls pooling, 9 seconds

```
python run_speaker.py -m +experiment=baseline_speaker_first+cls \
speaker.dataloader.train_batch_size=22 speaker.pipeline.selector_train.desired_chunk_length_sec=9 \
optim.algo.lr=3.16E-5 callbacks=speaker_default trainer.max_steps=320_000 \
hydra/launcher=slurm tag=[lr_grid,no_reg,320k]
```

mean, 3 seconds

```
python run_speaker.py -m +experiment=baseline_speaker_mean \
speaker.dataloader.train_batch_size=66 speaker.pipeline.selector_train.desired_chunk_length_sec=3 \
optim.algo.lr=3.16E-5 callbacks=speaker_default trainer.max_steps=320_000 \
hydra/launcher=slurm tag=[lr_grid,no_reg,320k]
```

mena pooling, 9 seconds

```
python run_speaker.py -m +experiment=baseline_speaker_mean \
speaker.dataloader.train_batch_size=22 speaker.pipeline.selector_train.desired_chunk_length_sec=9 \
optim.algo.lr=5.62E-5 callbacks=speaker_default trainer.max_steps=320_000 \
hydra/launcher=slurm tag=[lr_grid,no_reg,320k]
```

### Multi-task learning experiments

All experiments were logged at https://wandb.ai/w2v2-mt-learning/baseline-mt-speech-speaker.

#### vanilla head, mean and first+cls pooling

3 seconds, dynamic weighting, static weighting with lambda=0.50

```
python run_mt_speech_speaker.py -m +experiment=multi_task_2heads_vanilla \
optim.loss.scale_method=min,null network.stat_pooling_type=mean,first+cls \
speaker.dataloader.train_batch_size=66 speaker.pipeline.selector_train.desired_chunk_length_sec=3 \
optim.algo.lr=1E-5,1.78E-5,3.16E-5,5.62E-5,1E-4,1.78E-4 \
hydra/launcher=slurm tag=[lr_grid,no_reg,3_sec]
```

9 seconds, dynamic weighting, static weighting with lambda=0.50

```
python run_mt_speech_speaker.py -m +experiment=multi_task_2heads_vanilla \
optim.loss.scale_method=min,null network.stat_pooling_type=mean,first+cls \
speaker.dataloader.train_batch_size=22 speaker.pipeline.selector_train.desired_chunk_length_sec=9 \
optim.algo.lr=1E-5,1.78E-5,3.16E-5,5.62E-5,1E-4,1.78E-4 \
hydra/launcher=slurm tag=[lr_grid,no_reg,9_sec]
```

3 seconds, static weighting with lambda=0.88

```
python run_mt_speech_speaker.py -m +experiment=multi_task_2heads_vanilla \
optim.loss.scale_method=static optim.loss.static_speech_weight=0.88 network.stat_pooling_type=mean,first+cls \
speaker.dataloader.train_batch_size=66 speaker.pipeline.selector_train.desired_chunk_length_sec=3 \
optim.algo.lr=1E-5,1.78E-5,3.16E-5,5.62E-5,1E-4,1.78E-4 \
hydra/launcher=slurm tag=[lr_grid,no_reg,heuristic]
```

9 seconds, static weighting with lambda=0.88

```
python run_mt_speech_speaker.py -m +experiment=multi_task_2heads_vanilla \
optim.loss.scale_method=static optim.loss.static_speech_weight=0.88 network.stat_pooling_type=mean,first+cls \
speaker.dataloader.train_batch_size=22 speaker.pipeline.selector_train.desired_chunk_length_sec=9 \
optim.algo.lr=1E-5,1.78E-5,3.16E-5,5.62E-5,1E-4,1.78E-4 \
hydra/launcher=slurm_snellius tag=[lr_grid,no_reg,heuristic]
```

##### 320k steps


3 seconds, static weighting with lambda=0.50
```
python run_mt_speech_speaker.py -m +experiment=multi_task_2heads_vanilla tag=[320k,no_reg] \
trainer.max_steps=320_000 callbacks=mt_default network.stat_pooling_type=mean \
optim.loss.scale_method=null optim.loss.static_speech_weight=0.5  \
speaker.dataloader.train_batch_size=66 speaker.pipeline.selector_train.desired_chunk_length_sec=3 \
optim.algo.lr=5.62E-05 \
hydra/launcher=slurm_24vram 
```

static weighting with lambda=0.88

```
python run_mt_speech_speaker.py -m +experiment=multi_task_2heads_vanilla tag=[320k,no_reg] \
trainer.max_steps=320_000 callbacks=mt_default network.stat_pooling_type=mean \
optim.loss.scale_method=static optim.loss.static_speech_weight=0.88  \
speaker.dataloader.train_batch_size=66 speaker.pipeline.selector_train.desired_chunk_length_sec=3 \
optim.algo.lr=1.00E-04 \
hydra/launcher=slurm_24vram 
```

static weighting with dynamic weighting

```
python run_mt_speech_speaker.py -m +experiment=multi_task_2heads_vanilla tag=[320k,no_reg] \
trainer.max_steps=320_000 callbacks=mt_default network.stat_pooling_type=mean \
optim.loss.scale_method=min optim.loss.static_speech_weight=0.5  \
speaker.dataloader.train_batch_size=66 speaker.pipeline.selector_train.desired_chunk_length_sec=3 \
optim.algo.lr=5.62E-05 \
hydra/launcher=slurm_24vram 
```

9 seconds,  static weighting with lambda=0.50

```
python run_mt_speech_speaker.py -m +experiment=multi_task_2heads_vanilla tag=[320k,no_reg] \
trainer.max_steps=320_000 callbacks=mt_default network.stat_pooling_type=mean \
optim.loss.scale_method=null optim.loss.static_speech_weight=0.5  \
speaker.dataloader.train_batch_size=22 \
speaker.pipeline.selector_train.desired_chunk_length_sec=9 \
optim.algo.lr=3.16E-05 \
hydra/launcher=slurm_24vram 
```

9 seconds,  static weighting with lambda=0.88

```
python run_mt_speech_speaker.py -m +experiment=multi_task_2heads_vanilla tag=[320k,no_reg] \
trainer.max_steps=320_000 callbacks=mt_default network.stat_pooling_type=mean \
optim.loss.scale_method=static optim.loss.static_speech_weight=0.88  \
speaker.dataloader.train_batch_size=22 \
speaker.pipeline.selector_train.desired_chunk_length_sec=9 \
optim.algo.lr=1.00E-04,5.62E-05,3.16E-05 hydra/launcher=slurm_24vram 
```

9 seconds,  dynamic weighting

```
python run_mt_speech_speaker.py -m +experiment=multi_task_2heads_vanilla tag=[320k,no_reg] \
trainer.max_steps=320_000 callbacks=mt_default \
network.stat_pooling_type=mean \
optim.loss.scale_method=min optim.loss.static_speech_weight=0.5  \
speaker.dataloader.train_batch_size=22 \
speaker.pipeline.selector_train.desired_chunk_length_sec=9 \
optim.algo.lr=1.00E-04 \
hydra/launcher=slurm_24vram 
```


### branched encoder

```
python run_mt_speech_speaker.py -m +experiment=multi_task_2heads_vanilla \
optim.loss.scale_method=min network.stat_pooling_type=mean \
speaker.dataloader.train_batch_size=22 speaker.pipeline.selector_train.desired_chunk_length_sec=9 \
optim.algo.lr=1E-5,1.78E-5,3.16E-5,5.62E-5,1E-4,1.78E-4 \
network.use_multi_branch_encoder=true network.num_shared_transformers=3,6,9 \
hydra/launcher=slurm tag=[lr_grid,no_reg,branched]
```

### explicit-dimension-split

```
python run_mt_speech_speaker.py -m +experiment=multi_task_2heads_vanilla \
optim.loss.scale_method=min network.stat_pooling_type=mean \
speaker.dataloader.train_batch_size=22 speaker.pipeline.selector_train.desired_chunk_length_sec=9 \
optim.algo.lr=1E-5,1.78E-5,3.16E-5,5.62E-5,1E-4,1.78E-4 \
network.num_speaker_dimensions=192,384,576 \
hydra/launcher=slurm tag=[lr_grid,no_reg,output_split]
```

### CTC-based pooling


```
python run_mt_speech_speaker.py -m +experiment=multi_task_2heads_vanilla \
optim.loss.scale_method=min network.stat_pooling_type=mean \
speaker.dataloader.train_batch_size=22 speaker.pipeline.selector_train.desired_chunk_length_sec=9 \
optim.algo.lr=1E-5,1.78E-5,3.16E-5,5.62E-5,1E-4,1.78E-4 \
network.enable_ctc_pool_speaker_embedding=true network.ctc_pool_on_blank_embeddings=false,true \
hydra/launcher=slurm tag=[lr_grid,no_reg,ctc_pool]
```


## Checkpoints

Checkpoints of each experiment will be made available after publication (hosted at institutional URL).
