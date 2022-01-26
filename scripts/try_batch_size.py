########################################################################################
#
# Use a fake batch to determine the maximum size without CUDA vram issues.
#
# Author(s): Anonymous
########################################################################################


import hydra
from omegaconf import DictConfig

from src.pl_modules import MultiTaskWav2vec2Module, MultiTaskWav2vec2ModuleConfig

########################################################################################
# fake batch generation settings

speech_max_frames = 32000
speech_gt_max_length = 400

speaker_max_frames = 16000 * 3


########################################################################################
# fake train loop


@hydra.main(config_path="..config", config_name="train_speech")
def run(cfg: DictConfig):
    # we import here such that tab-completion in bash
    # does not need to import everything (which slows it down
    # significantly)
    from src.main import construct_network_module, construct_data_module

    network = construct_network_module(cfg, construct_data_module(cfg))


if __name__ == "__main__":
    run()
