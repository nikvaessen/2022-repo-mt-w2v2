########################################################################################
#
# This file holds the logic to do an LR range test with PyTorch Lightning
#
# Author(s): Anonymous
########################################################################################

import json

from datetime import datetime

import pytorch_lightning as pl

########################################################################################
# the method to to lr range tests with a pytorch lightning trainer and module


def run_lr_range_test(
    trainer: pl.Trainer,
    network: pl.LightningModule,
    dm: pl.LightningDataModule,
    tune_iterations: int,
    log_metrics: bool = True,
):
    print(f"tuning for {tune_iterations} iterations")

    results = trainer.tune(
        network,
        datamodule=dm,
        lr_find_kwargs={
            "num_training": tune_iterations,
            "mode": "exponential",
            "early_stop_threshold": 3,
            "update_attr": True,
        },
    )

    if "lr_find" in results:
        result = results["lr_find"]
        filename = f"lr_find_{datetime.now().strftime('%Y%m%d-%H%M%S')}"

        result_dict = {
            "lr_min": result.lr_min,
            "lr_max": result.lr_max,
            "num_training": result.num_training,
            "data": {**result.results},
        }

        lr_list = result_dict["data"]["lr"]
        loss_list = result_dict["data"]["loss"]

        # save data points
        with open(f"{filename}.json", "w") as f:
            json.dump(
                result_dict,
                f,
            )

        # save figure
        fig = result.plot(suggest=True, show=False)
        fig.savefig(f"{filename}.png")

        if log_metrics:
            trainer.logger.log_metrics(
                {
                    "lr_min": result_dict["lr_min"],
                    "lr_max": result_dict["lr_max"],
                }
            )

            assert len(lr_list) == len(loss_list)

            for idx in range(len(lr_list)):
                trainer.logger.agg_and_log_metrics(
                    {"lr": lr_list[idx], "loss": loss_list[idx]}, step=idx
                )
