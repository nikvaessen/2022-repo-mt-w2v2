########################################################################################
#
# This file is the main entrypoint of the train/eval loop based on the
# hydra configuration.
#
# Author(s): Anonymous
########################################################################################

import logging

from typing import Union, Callable, List, Dict

import torch as t
import pytorch_lightning as pl
import transformers
import wandb

from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
from torch.distributed import destroy_process_group
from pytorch_lightning import Callback
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CometLogger, WandbLogger

from src.auto_lr_find import run_lr_range_test
from src.data.loading_config import SpeakerDataLoaderConfig, SpeechDataLoaderConfig
from src.data.modules import (
    JointLibrispeechVoxcelebDataModuleConfig,
    JointLibrispeechVoxcelebDataModule,
)
from src.data.modules.librispeech import (
    LibriSpeechDataModuleConfig,
    LibriSpeechDataModule,
)
from src.data.modules.voxceleb import VoxCelebDataModuleConfig, VoxCelebDataModule
from src.data.modules.abstract import (
    SpeakerLightningDataModule,
    SpeechLightningDataModule,
)
from src.evaluation.speech.tokenizer.tokenizer_wav2vec2 import (
    Wav2vec2TokenizerConfig,
    Wav2vec2Tokenizer,
)
from src.pl_modules import (
    SpeakerRecognitionLightningModule,
    Wav2vec2SpeakerRecognitionModuleConfig,
    Wav2vec2SpeakerRecognitionModule,
    Wav2vec2SpeechRecognitionModuleConfig,
    Wav2vec2SpeechRecognitionModule,
    SpeechRecognitionLightningModule,
    MtSpeechSpeakerLightningModule,
    MultiTaskWav2vec2ModuleConfig,
    MultiTaskWav2vec2Module,
)
from src.util.system import get_git_revision_hash

########################################################################################
# implement constructing data module

log = logging.getLogger(__name__)


def load_pipeline(cfg: DictConfig, task_name: str):
    task_cfg = cfg[task_name]

    # load data pipelines
    if task_cfg.pipeline.get("augmentations", None) is not None:
        augment_wrappers = [
            instantiate(task_cfg.pipeline[n])
            for n in task_cfg.pipeline.get("augmentations")
        ]
    else:
        augment_wrappers = None

    train_pipeline = [
        instantiate(task_cfg.pipeline[n])
        if task_cfg.pipeline[n]["_target_"] != "src.data.preprocess.augment.Augmenter"
        else instantiate(task_cfg.pipeline[n], augmenters=augment_wrappers)
        for n in task_cfg.pipeline.train_pipeline
    ]
    val_pipeline = [
        instantiate(task_cfg.pipeline[n]) for n in task_cfg.pipeline.val_pipeline
    ]
    test_pipeline = [
        instantiate(task_cfg.pipeline[n]) for n in task_cfg.pipeline.test_pipeline
    ]

    return train_pipeline, val_pipeline, test_pipeline


def construct_speech_data_module(cfg: DictConfig):
    # load data module config
    dm_cfg = instantiate(cfg.data)

    # load dataloader config
    dl_cfg = instantiate(cfg.speech.dataloader)

    if not isinstance(dl_cfg, SpeechDataLoaderConfig):
        raise ValueError(
            f"LibriSpeechLightningDataModule expects {SpeechDataLoaderConfig},"
            f" got {dl_cfg}"
        )

    # load pipeline
    train_pipeline, val_pipeline, test_pipeline = load_pipeline(cfg, "speech")

    # construct tokenizer
    tokenizer = construct_tokenizer(cfg)

    return LibriSpeechDataModule(
        cfg=dm_cfg,
        dl_cfg=dl_cfg,
        tokenizer=tokenizer,
        train_pipeline=train_pipeline,
        val_pipeline=val_pipeline,
        test_pipeline=test_pipeline,
    )


def construct_speaker_data_module(cfg: DictConfig):
    # load data module config
    dm_cfg = instantiate(cfg.data)

    # load dataloader config
    dl_cfg = instantiate(cfg.speaker.dataloader)

    if not isinstance(dl_cfg, SpeakerDataLoaderConfig):
        raise ValueError(
            f"VoxCelebDataModule expects {SpeakerDataLoaderConfig}," f" got {dl_cfg}"
        )

    # load pipeline
    train_pipeline, val_pipeline, test_pipeline = load_pipeline(cfg, "speaker")

    return VoxCelebDataModule(
        cfg=dm_cfg,
        dl_cfg=dl_cfg,
        train_pipeline=train_pipeline,
        val_pipeline=val_pipeline,
        test_pipeline=test_pipeline,
    )


def construct_speech_and_speaker_data_module(cfg: DictConfig):
    # load data module config
    dm_cfg = instantiate(cfg.data)

    # load speaker dataloader config
    speaker_dl_cfg = instantiate(cfg.speaker.dataloader)

    if not isinstance(speaker_dl_cfg, SpeakerDataLoaderConfig):
        raise ValueError(
            f"VoxCelebDataModule expects {SpeakerDataLoaderConfig},"
            f" got {speaker_dl_cfg}"
        )

    # load speech dataloader config
    speech_dl_cfg = instantiate(cfg.speech.dataloader)

    if not isinstance(speech_dl_cfg, SpeechDataLoaderConfig):
        raise ValueError(
            f"LibriSpeechLightningDataModule expects {SpeechDataLoaderConfig},"
            f" got {speech_dl_cfg}"
        )

    # load pipeline for speaker
    speaker_train_pipeline, speaker_val_pipeline, speaker_test_pipeline = load_pipeline(
        cfg, "speaker"
    )

    # load pipeline for speech
    speech_train_pipeline, speech_val_pipeline, speech_test_pipeline = load_pipeline(
        cfg, "speech"
    )

    # construct tokenizer
    tokenizer = construct_tokenizer(cfg)

    return JointLibrispeechVoxcelebDataModule(
        cfg=dm_cfg,
        speaker_dl_cfg=speaker_dl_cfg,
        speaker_train_pipeline=speaker_train_pipeline,
        speaker_val_pipeline=speaker_val_pipeline,
        speaker_test_pipeline=speaker_test_pipeline,
        speech_dl_cfg=speech_dl_cfg,
        tokenizer=tokenizer,
        speech_train_pipeline=speech_train_pipeline,
        speech_val_pipeline=speech_val_pipeline,
        speech_test_pipeline=speech_test_pipeline,
    )


def construct_data_module(
    cfg: DictConfig,
) -> Union[SpeechLightningDataModule, SpeakerLightningDataModule]:
    # load data module config
    dm_cfg = instantiate(cfg.data)

    # create data module
    if isinstance(dm_cfg, VoxCelebDataModuleConfig):
        dm = construct_speaker_data_module(cfg)
    elif isinstance(dm_cfg, LibriSpeechDataModuleConfig):
        dm = construct_speech_data_module(cfg)
    elif isinstance(dm_cfg, JointLibrispeechVoxcelebDataModuleConfig):
        dm = construct_speech_and_speaker_data_module(cfg)
    else:
        raise ValueError(f"cannot load data module from {dm_cfg}")

    dm.prepare_data()
    dm.setup()
    dm.summary()

    return dm


########################################################################################
# implement the construction of network modules


def construct_speech_recognition_module(
    cfg: DictConfig,
    network_cfg: DictConfig,
    dm: SpeechLightningDataModule,
    loss_fn_constructor: Callable[[], Callable[[t.Tensor, t.Tensor], t.Tensor]],
) -> SpeechRecognitionLightningModule:
    # every network needs these variables
    tokenizer = dm.tokenizer

    if isinstance(network_cfg, Wav2vec2SpeechRecognitionModuleConfig):
        network_class = Wav2vec2SpeechRecognitionModule
    else:
        raise ValueError(f"cannot load network from {network_cfg}")

    # init model
    kwargs = {
        "hyperparameter_config": cfg,
        "cfg": network_cfg,
        "loss_fn_constructor": loss_fn_constructor,
        "tokenizer": tokenizer,
    }

    return init_model(cfg, network_class, kwargs)


def construct_speaker_recognition_module(
    cfg: DictConfig,
    network_cfg: DictConfig,
    dm: SpeakerLightningDataModule,
    loss_fn_constructor: Callable[[], Callable[[t.Tensor, t.Tensor], t.Tensor]],
) -> SpeakerRecognitionLightningModule:
    # every network needs to be given these variables
    num_speakers = dm.num_speakers
    validation_pairs = dm.val_pairs
    test_pairs = dm.test_pairs
    test_names = dm.test_names

    # get init function based on config type
    if isinstance(network_cfg, Wav2vec2SpeakerRecognitionModuleConfig):
        network_class = Wav2vec2SpeakerRecognitionModule
    else:
        raise ValueError(f"cannot load network from {network_cfg}")

    # init model
    kwargs = {
        "hyperparameter_config": cfg,
        "cfg": network_cfg,
        "num_speakers": num_speakers,
        "loss_fn_constructor": loss_fn_constructor,
        "validation_pairs": validation_pairs,
        "test_pairs": test_pairs,
        "test_names": test_names,
    }

    return init_model(cfg, network_class, kwargs)


def construct_multitask_module(
    cfg: DictConfig,
    network_cfg: DictConfig,
    speaker_dm: SpeakerLightningDataModule,
    speech_dm: SpeechLightningDataModule,
    loss_fn_constructor: Callable[[], Callable[[t.Tensor, t.Tensor], t.Tensor]],
) -> MtSpeechSpeakerLightningModule:
    # every network needs to be given these variables
    num_speakers = speaker_dm.num_speakers
    validation_pairs = speaker_dm.val_pairs
    test_pairs = speaker_dm.test_pairs
    test_names = speaker_dm.test_names
    tokenizer = speech_dm.tokenizer

    # get init function based on config type
    if isinstance(network_cfg, MultiTaskWav2vec2ModuleConfig):
        network_class = MultiTaskWav2vec2Module
    else:
        raise ValueError(f"cannot load network from {network_cfg}")

    # init model
    kwargs = {
        "hyperparameter_config": cfg,
        "cfg": network_cfg,
        "num_speakers": num_speakers,
        "loss_fn_constructor": loss_fn_constructor,
        "speaker_validation_pairs": validation_pairs,
        "speaker_test_pairs": test_pairs,
        "test_names": test_names,
        "tokenizer": tokenizer,
    }

    return init_model(cfg, network_class, kwargs)


def init_model(cfg: DictConfig, network_class, kwargs: Dict):
    # load model weights from checkpoint
    potential_checkpoint_path = cfg.get("load_network_from_checkpoint", None)

    if potential_checkpoint_path is not None:
        log.info(
            f"reloading {network_class.__class__} from {potential_checkpoint_path}"
        )
        network = network_class.load_from_checkpoint(
            cfg.load_network_from_checkpoint, strict=False, **kwargs
        )
    else:
        network = network_class(**kwargs)

    return network


def construct_network_module(
    cfg: DictConfig,
    dm: Union[SpeakerLightningDataModule, SpeechLightningDataModule],
) -> Union[SpeakerRecognitionLightningModule, SpeechLightningDataModule]:
    # load loss function
    def loss_fn_constructor():
        # should be instantiated in the network
        # so that potential parameters are properly
        # registered
        return instantiate(cfg.optim.loss)

    # load network config
    network_cfg = instantiate(cfg.network)

    if isinstance(dm, SpeakerLightningDataModule) and isinstance(
        dm, SpeechLightningDataModule
    ):
        network = construct_multitask_module(
            cfg, network_cfg, dm, dm, loss_fn_constructor
        )
    elif isinstance(dm, SpeakerLightningDataModule):
        network = construct_speaker_recognition_module(
            cfg, network_cfg, dm, loss_fn_constructor
        )
    elif isinstance(dm, SpeechLightningDataModule):
        network = construct_speech_recognition_module(
            cfg, network_cfg, dm, loss_fn_constructor
        )
    else:
        raise ValueError(
            f"can not construct network for data module type {dm.__class__}"
        )

    # set optimizer and learning rate schedule
    optimizer = instantiate(cfg.optim.algo, params=network.parameters())
    schedule = {
        "scheduler": instantiate(cfg.optim.schedule.scheduler, optimizer=optimizer),
        "monitor": cfg.optim.schedule.monitor,
        "interval": cfg.optim.schedule.interval,
        "frequency": cfg.optim.schedule.frequency,
        "name": cfg.optim.schedule.name,
    }
    # remove None values from dict
    schedule = {k: v for k, v in schedule.items() if v is not None}

    network.set_optimizer(optimizer)
    network.set_lr_schedule(schedule)

    return network


########################################################################################
# initialize tokenizer (for ASR data)


def construct_tokenizer(cfg: DictConfig):
    tokenizer_cfg = instantiate(cfg.speech.tokenizer)

    if isinstance(tokenizer_cfg, Wav2vec2TokenizerConfig):
        tokenizer = Wav2vec2Tokenizer(tokenizer_cfg)
    else:
        raise ValueError(f"cannot construct a tokenizer based on {tokenizer_cfg}")

    return tokenizer


########################################################################################
# implement construction of callbacks, profiler and logger


def construct_callbacks(cfg: DictConfig) -> List[Callback]:
    callbacks = []

    callback_cfg: DictConfig = cfg.callbacks

    ModelCheckpoint.CHECKPOINT_NAME_LAST = callback_cfg.get(
        "last_checkpoint_pattern", "last"
    )

    for cb_key in callback_cfg.to_add:
        if cb_key is None:
            continue

        if cb_key in callback_cfg:
            cb = instantiate(callback_cfg[cb_key])
            log.info(f"Using callback <{cb}>")

            callbacks.append(instantiate(callback_cfg[cb_key]))

    return callbacks


def construct_profiler(cfg: DictConfig):
    profile_cfg = cfg.get("profiler", None)

    if profile_cfg is None:
        return None
    else:
        return instantiate(profile_cfg)


def construct_logger(cfg: DictConfig):
    if cfg.use_wandb:
        if isinstance(cfg.tag, str):
            cfg.tag = [cfg.tag]
        if cfg.run_lr_range_test:
            cfg.tag.append("lr_range_test")

        logger = WandbLogger(
            entity="w2v2-mt-learning",
            project=cfg.project_name,
            name=cfg.experiment_name,
            tags=cfg.tag,
        )
        # init the wandb agent
        _ = logger.experiment
    else:
        logger = True

    return logger


########################################################################################
# implement the main function based on the whole config


def run_train_eval_script(cfg: DictConfig):
    if cfg.anonymous_mode:
        import warnings

        # warnings leak absolute path of python files (and thus username)
        warnings.filterwarnings("ignore")

        # pytorch lightning might log absolute path of checkpoint files, and thus
        # leak username
        logging.getLogger("pytorch_lightning").setLevel(logging.CRITICAL)

    # create logger
    logger = construct_logger(cfg)

    # print config
    print(OmegaConf.to_yaml(cfg))
    print(f"current git commit hash: {get_git_revision_hash()}")
    print(f"PyTorch version is {t.__version__}")
    print(f"PyTorch Lightning version is {pl.__version__}")
    print(f"transformers version is {transformers.__version__}")
    print()

    # construct data module
    dm = construct_data_module(cfg)

    # create callbacks
    callbacks = construct_callbacks(cfg)

    # construct profiler
    profiler = construct_profiler(cfg)

    # create training/evaluator
    trainer: pl.Trainer = instantiate(
        cfg.trainer,
        logger=logger,
        callbacks=callbacks,
        profiler=profiler,
        auto_lr_find="auto_lr_find" if cfg.run_lr_range_test else None,
    )

    # construct lighting module for train/test
    network = construct_network_module(cfg, dm)

    # tune model
    if cfg.run_lr_range_test:
        trainer.logger.log_hyperparams(cfg)
        run_lr_range_test(
            trainer,
            network,
            dm,
            tune_iterations=cfg.lr_range_iterations,
        )

        return

    # train model
    if cfg.fit_model:
        trainer.fit(network, datamodule=dm)

    # test model
    if cfg.trainer.accelerator == "ddp":
        destroy_process_group()

        if not trainer.global_rank == 0:
            return

    # create a new trainer which uses at most 1 gpu
    trainer: pl.Trainer = instantiate(
        cfg.trainer,
        gpus=min(1, int(cfg.trainer.get("gpus"))),
        accelerator=None,
        logger=logger,
        callbacks=callbacks,
        profiler=profiler,
    )

    result = None
    if cfg.eval_model and cfg.fit_model:
        # this will select the checkpoint with the best validation metric
        # according to the ModelCheckpoint callback
        try:
            result = trainer.test(datamodule=dm)
        except:
            # there might not have been a validation epoch
            result = trainer.test(network, datamodule=dm)
    elif cfg.eval_model:
        # this will simply test the given model weights (when it's e.g
        # manually loaded from a checkpoint)
        result = trainer.test(network, datamodule=dm)

    if result is not None:
        if isinstance(result, list):
            result_obj = result[0]

            if "test_eer_val" in result_obj:
                objective = result_obj["test_eer_val"]
            elif "test_wer_other" in result_obj:
                objective = result_obj["test_wer_other"]
            else:
                raise ValueError(
                    f"unknown objective value out of keys "
                    f"{[k for k in result_obj.keys()]}"
                )

            return objective
        else:
            raise ValueError(f"result object has unknown type {type(result)=}")

    return None
