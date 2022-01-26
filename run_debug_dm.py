################################################################################
#
# This run script encapsulates the debugging script of a data module.
#
# Author(s): Anonymous
################################################################################

import hydra

from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf

from src.util.hydra_resolvers import (
    division_resolver,
    integer_division_resolver,
    random_uuid,
    random_experiment_id,
)

################################################################################
# set custom resolvers

OmegaConf.register_new_resolver("divide", division_resolver)
OmegaConf.register_new_resolver("idivide", integer_division_resolver)
OmegaConf.register_new_resolver("random_uuid", random_uuid)
OmegaConf.register_new_resolver("random_name", random_experiment_id)

################################################################################
# wrap around main hydra script


@hydra.main(config_path="config", config_name="train_mt_speaker_speech")
def run(cfg: DictConfig):
    # we import here such that tab-completion in bash
    # does not need to import everything (which slows it down
    # significantly)
    from src.debug_dm import run_debug_dm_script

    return run_debug_dm_script(cfg)


################################################################################
# execute hydra application

if __name__ == "__main__":
    load_dotenv()
    run()
