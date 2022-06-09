from os import path

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

import wandb
from data import Div2K
from model import ResidualDenseNetwork


def main():
    
    with wandb.init() as run:

        run.log_code()

        data_source = run.use_artifact("div2k:latest")
        data_root = data_source.download()
        train_lr_dir =  path.join(data_root, "train_lr")
        train_hr_dir =  path.join(data_root, "train_hr")
        valid_lr_dir =  path.join(data_root, "valid_lr")
        valid_hr_dir =  path.join(data_root, "valid_hr")

        train_ds = Div2K(4, train_lr_dir, train_hr_dir, size=30)
        val_ds = Div2K(4, valid_lr_dir, valid_hr_dir, size=90, center=True)

        train_dl = DataLoader(train_ds, batch_size=8, num_workers=8)
        val_dl = DataLoader(val_ds, batch_size=1, num_workers=8)

        model = ResidualDenseNetwork(64, 64, 16, 8, 3, 4)
        logger = WandbLogger()
        logger.watch(model, log_freq=200, log="all")
        trainer = pl.Trainer(gpus=1, logger=WandbLogger())
        trainer.fit(model, train_dl, val_dl)


if __name__ == "__main__":
    main()
