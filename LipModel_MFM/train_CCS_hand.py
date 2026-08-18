"""Train the lip-reading backbone and Minimalist Fusion Module."""

import os

import hydra
import torch
from omegaconf import OmegaConf
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from avg_ckpts import ensemble
from datamodule.data_module_CCS import DataModule_CCS
from lightning_CCS_Hand import ModelModule_CCS_hand


def _trainer_devices(gpu):
    if not torch.cuda.is_available():
        return 0
    gpu = int(gpu)
    if gpu < 0 or gpu >= torch.cuda.device_count():
        raise ValueError(
            f"GPU {gpu} is unavailable; found {torch.cuda.device_count()} devices."
        )
    return [gpu]


@hydra.main(
    version_base="1.3",
    config_path="datamodule/configs",
    config_name="config_CCS_hand",
)
def main(cfg):
    seed_everything(int(cfg.seed), workers=True)
    experiment_dir = os.path.join(str(cfg.exp_dir), str(cfg.exp_name))
    os.makedirs(experiment_dir, exist_ok=True)

    checkpoint = ModelCheckpoint(
        monitor="loss_val",
        mode="min",
        dirpath=experiment_dir,
        save_last=True,
        filename="{epoch}-{loss_val:.4f}",
        save_top_k=int(cfg.save_top_k),
    )
    callbacks = [checkpoint, LearningRateMonitor(logging_interval="step")]

    trainer_kwargs = OmegaConf.to_container(cfg.trainer, resolve=True)
    trainer = Trainer(
        **trainer_kwargs,
        callbacks=callbacks,
        gpus=_trainer_devices(cfg.gpu),
    )
    trainer.fit(
        model=ModelModule_CCS_hand(cfg),
        datamodule=DataModule_CCS(cfg),
    )

    average_count = int(getattr(cfg, "average_checkpoints", 0))
    if average_count > 0:
        ensemble(cfg, count=average_count)


if __name__ == "__main__":
    main()
