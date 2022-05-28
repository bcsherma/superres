import random
from glob import glob
from os import path

from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.transforms import CenterCrop


class Div2K(Dataset):
    def __init__(self, lowres_folder, highres_folder, size=30, center=False):
        super().__init__()
        self.size = size
        self.center = center
        self.lowres_root = lowres_folder
        self.highres_root = highres_folder
        self.lowres_manifest = sorted(glob(path.join(self.lowres_root, "*.png")))
        self.highres_manifest = sorted(glob(path.join(self.highres_root, "*.png")))
        assert len(self.lowres_manifest) == len(self.highres_manifest)

    def __len__(self):
        return len(self.lowres_manifest)

    def __getitem__(self, index):
        lowres = read_image(self.lowres_manifest[index])
        highres = read_image(self.highres_manifest[index])

        if self.center:
            lowres = CenterCrop(size=self.size)(lowres)
            highres = CenterCrop(size=2 * self.size)(highres)
        else:
            x = random.randint(0, lowres.shape[2] - self.size)
            y = random.randint(0, lowres.shape[1] - self.size)
            lowres = lowres[..., y : y + self.size, x : x + self.size]
            highres = highres[
                ..., 2 * y : 2 * y + 2 * self.size, 2 * x : 2 * x + 2 * self.size
            ]

        return {
            "highres": highres.float(),
            "lowres": lowres.float(),
        }
