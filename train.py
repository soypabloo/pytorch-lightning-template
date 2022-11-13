"""System module."""
import argparse
import functools
import os
import random

import numpy as np
import pytorch_lightning as pl
import torch
import torch.backends.cudnn
from omegaconf import OmegaConf, DictConfig, ListConfig
import pytorch_lightning.callbacks
from pytorch_lightning.loggers.wandb import WandbLogger

from datamodule.dataloader import DataLoader
from model.model import Model
from typing import Any, Union


def rsetattr(obj: Union[DictConfig, ListConfig], attr: str, val: Any) -> None:
    """
    recursion을 이용하여,
    중첩된 형태로 구조화되어있는 값을 val로 수정한다.

    Returns:
        None
    """
    pre, _, post = attr.rpartition(".")
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)


def rgetattr(obj: Union[DictConfig, ListConfig], attr: str, *args) -> Any:
    """
    중첩되어 있는 config의 attribute를 반환한다.
    Args:
        obj (_type_): 접근하고자 하는 객체의
        attr (_type_): _description_
    """

    def _getattr(obj, attr):
        return getattr(obj, attr, *args)

    return functools.reduce(_getattr, [obj] + attr.split("."))


def seed_everything(seed):
    """set seed"""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    pl.seed_everything(seed, workers=True)


def main(cfg):
    seed_everything(cfg.train.seed)

    exp_name = "_".join(
        map(
            str,
            [
                cfg.name,
                cfg.model.model_name,
                cfg.optimizer.name,
                cfg.lr_scheduler.name,
                cfg.train.learning_rate,
                cfg.train.batch_size,
            ],
        )
    )
    print("set Logger")
    wandb_logger = WandbLogger(name=exp_name, project=cfg.project, log_model="all")

    call_backs = [
        getattr(pytorch_lightning.callbacks, call_back.name)(**call_back.args)
        for call_back in cfg.call_back
    ]
    print("Load DataLoader...")
    dataloader = DataLoader(
        cfg.train.batch_size,
        cfg.data.shuffle,
        cfg.path.train_path,
        cfg.path.dev_path,
        cfg.path.test_path,
    )
    print("Load model...")
    model = Model(cfg)
    print("set trainer...")
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=cfg.train.gpus,
        max_epochs=cfg.train.max_epoch,
        logger=wandb_logger,
        log_every_n_steps=cfg.train.logging_step,
        callbacks=call_backs,
    )

    trainer.fit(model=model, datamodule=dataloader)
    trainer.test(model=model, datamodule=dataloader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="base_config", required=True)

    args, options = parser.parse_known_args()
    cfg = OmegaConf.load(f"./config/{args.config}.yaml")
    # sweep에서 --config 이외의 cli argument가 들어오면 입력해준다.
    for option in options:
        arg_name, value = option.split("=")
        try:  # value가 int인지, float인지, string인지 체크
            value = int(value) if float(value) == int(float(value)) else float(value)
        except ValueError:
            pass
        # options에 추가로 적용한 args를 적용.
        rsetattr(cfg, arg_name, value)
    main(cfg)
