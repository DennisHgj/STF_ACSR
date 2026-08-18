"""Evaluate an STF-ACSR MFM checkpoint on a configured CCS split."""

import os

import hydra
import torch
from pytorch_lightning import Trainer, seed_everything

from checkpoint_utils import load_lightning_weights
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
    config_name="config_CCS_hand_test",
)
def main(cfg):
    seed_everything(int(cfg.seed), workers=True)
    checkpoint_path = os.path.join(
        str(cfg.exp_dir), str(cfg.exp_name), str(cfg.ckpt_path)
    )
    if not cfg.ckpt_path or not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            "Set exp_dir, exp_name and ckpt_path to an existing checkpoint; "
            f"resolved path: {checkpoint_path}"
        )

    model = ModelModule_CCS_hand(cfg)
    missing, unexpected = load_lightning_weights(
        model, checkpoint_path, strict=True
    )
    if missing or unexpected:
        raise RuntimeError(
            f"Checkpoint mismatch; missing={missing}, unexpected={unexpected}"
        )

    trainer_kwargs = {
        "num_nodes": 1,
        "gpus": _trainer_devices(cfg.gpu),
        "logger": False,
    }
    if cfg.limit_test_batches is not None:
        trainer_kwargs["limit_test_batches"] = int(cfg.limit_test_batches)
    trainer = Trainer(**trainer_kwargs)
    trainer.test(model=model, datamodule=DataModule_CCS(cfg))


if __name__ == "__main__":
    main()
