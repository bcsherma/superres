import argparse
from os import path

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
import torch
from torch.utils.data import DataLoader

import wandb
from data import Div2K
from model import ResidualDenseNetwork


def parse_args():
    parser = argparse.ArgumentParser("train.py")
    parser.add_argument("--dl_workers", type=int, default=4)
    parser.add_argument("--growth_rate", type=int, default=64)
    parser.add_argument("--log_grad", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max_epochs", type=int, default=200)
    parser.add_argument("--nblocks", type=int, default=16)
    parser.add_argument("--nchannels", type=int, default=3)
    parser.add_argument("--nfeatures", type=int, default=64)
    parser.add_argument("--nlayers", type=int, default=8)
    parser.add_argument("--patch_size", type=int, default=32)
    parser.add_argument("--precision", type=int, default=32)
    parser.add_argument("--save_code", action="store_true")
    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--val_batch_size", type=int, default=4)
    parser.add_argument("--val_im_size", type=int, default=128)
    parser.add_argument("dataset", type=str)
    return parser.parse_args()


def main():

    args = parse_args()

    with wandb.init(job_type="train", config=args) as run:

        config = run.config

        if config.save_code:
            run.log_code()

        data_source = run.use_artifact(config.dataset)
        config["scale_factor"] = data_source.metadata["scale_factor"]

        data_root = data_source.download()
        train_lr_dir = path.join(data_root, "train_lr")
        train_hr_dir = path.join(data_root, "train_hr")
        valid_lr_dir = path.join(data_root, "valid_lr")
        valid_hr_dir = path.join(data_root, "valid_hr")

        train_ds = Div2K(
            config.scale_factor, train_lr_dir, train_hr_dir, size=config.patch_size
        )
        val_ds = Div2K(
            config.scale_factor,
            valid_lr_dir,
            valid_hr_dir,
            size=config.val_im_size,
            center=True,
        )

        train_dl = DataLoader(
            train_ds, batch_size=config.train_batch_size, num_workers=config.dl_workers
        )
        val_dl = DataLoader(
            val_ds, batch_size=config.val_batch_size, num_workers=config.dl_workers
        )

        model = ResidualDenseNetwork(
            config.nfeatures,
            config.growth_rate,
            config.nblocks,
            config.nlayers,
            config.nchannels,
            config.scale_factor,
            config.lr,
        )

        logger = WandbLogger(experiment=run, log_model="all")
        if config.log_grad:
            logger.watch(model, log_freq=config.log_grad, log_graph=False)

        if torch.cuda.is_available():
            accelerator = "gpu"
            devices = -1
        else:
            accelerator = None
            devices = None

        trainer = pl.Trainer(
            precision=config.precision,
            logger=logger,
            accelerator=accelerator,
            devices=devices,
            max_epochs=config.max_epochs,
            check_val_every_n_epoch=10,
            callbacks=[
                LearningRateMonitor(),
                ModelCheckpoint(every_n_epochs=10),
                EarlyStopping(monitor="val/loss", mode="min")
            ],
        )
        trainer.fit(model, train_dl, val_dl)


if __name__ == "__main__":
    main()
