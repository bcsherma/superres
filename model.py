"""
PyTorch Lightning implementation of the neural network described in Residual 
Dense Network for Image Super-Resolution by Zhang et al. in CVPR 2018

Vanilla PyTorch implementation from https://github.com/yjn870/RDN-pytorch
was used as starter code.
"""

import pytorch_lightning as pl
import torch
from torch import nn

import wandb


class ResidualDenseNetwork(pl.LightningModule):
    """ """

    def __init__(
        self,
        num_features,
        growth_rate,
        num_blocks,
        num_layers,
        num_channels,
        scale_factor,
    ) -> None:
        super().__init__()
        self.num_features = num_features
        self.growth_rate = growth_rate
        self.num_blocks = num_blocks
        self.num_layers = num_layers
        self.loss = nn.L1Loss()

        # shallow feature extraction
        self.sfe1 = nn.Conv2d(num_channels, num_features, kernel_size=3, padding=1)
        self.sfe2 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)

        # residual dense blocks
        self.rdbs = nn.ModuleList(
            [ResidualDenseBlock(self.num_features, self.growth_rate, self.num_layers)]
        )
        for _ in range(self.num_blocks - 1):
            self.rdbs.append(
                ResidualDenseBlock(self.growth_rate, self.growth_rate, self.num_layers)
            )

        # global feature fusion
        self.gff = nn.Sequential(
            nn.Conv2d(
                self.growth_rate * self.num_blocks, self.num_features, kernel_size=1
            ),
            nn.Conv2d(self.num_features, self.num_features, kernel_size=3, padding=1),
        )

        # up-sampling
        assert 2 <= scale_factor <= 4
        if scale_factor == 2 or scale_factor == 4:
            self.upscale = []
            for _ in range(scale_factor // 2):
                self.upscale.extend(
                    [
                        nn.Conv2d(
                            self.num_features,
                            self.num_features * (2**2),
                            kernel_size=3,
                            padding=1,
                        ),
                        nn.PixelShuffle(2),
                    ]
                )
            self.upscale = nn.Sequential(*self.upscale)
        else:
            self.upscale = nn.Sequential(
                nn.Conv2d(
                    self.num_features,
                    self.num_features * (scale_factor**2),
                    kernel_size=3,
                    padding=1,
                ),
                nn.PixelShuffle(scale_factor),
            )

        self.output = nn.Conv2d(
            self.num_features, num_channels, kernel_size=3, padding=1
        )

    def forward(self, x):
        sfe1 = self.sfe1(x)
        sfe2 = self.sfe2(sfe1)

        x = sfe2
        local_features = []
        for i in range(self.num_blocks):
            x = self.rdbs[i](x)
            local_features.append(x)

        x = self.gff(torch.cat(local_features, 1)) + sfe1
        x = self.upscale(x)
        x = self.output(x)
        return x

    def training_step(self, batch):
        inputs = batch["lowres"]
        labels = batch["highres"]
        preds = self(inputs)
        loss = self.loss(preds, labels)
        self.log("train/loss", loss, on_step=True)
        return loss

    def validation_step(self, batch, _):
        lowres = batch["lowres"]
        highres = batch["highres"]
        preds = self(lowres)
        loss = self.loss(preds, highres)
        for lr, hr, pr in zip(lowres, highres, preds):
            self.table.add_data(wandb.Image(lr), wandb.Image(pr), wandb.Image(hr))
        self.log("val/loss", loss)

    def on_validation_start(self) -> None:
        self.table = wandb.Table(columns=["lowres", "superres", "highres"])

    def on_validation_end(self):
        self.logger.experiment.log({"predictions": self.table}, commit=False)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())


class ResidualDenseBlock(nn.Module):
    """ """

    def __init__(self, in_channels, growth_rate, num_layers) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            *[
                DenseLayer(in_channels + growth_rate * i, growth_rate)
                for i in range(num_layers)
            ]
        )
        self.local_feature_fusion = nn.Conv2d(
            in_channels + growth_rate * num_layers, growth_rate, kernel_size=1
        )

    def forward(self, x):
        return x + self.local_feature_fusion(self.layers(x))


class DenseLayer(nn.Module):
    """ """

    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size, padding=padding
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return torch.cat([x, self.relu(self.conv(x))], 1)
