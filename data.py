from glob import glob
from os import path

from torch.utils.data import Dataset
from torchvision.io import read_image


class Div2K(Dataset):
    def __init__(self, lowres_folder, highres_folder):
        super().__init__()
        self.lowres_root = lowres_folder
        self.highres_root = highres_folder
        self.lowres_manifest = sorted(glob(path.join(self.lowres_root, "*.png")))
        self.highres_manifest = sorted(glob(path.join(self.highres_root, "*.png")))
        assert len(self.lowres_manifest) == len(self.highres_manifest)

    def __len__(self):
        return len(self.lowres_manifest)

    def __getitem__(self, index):
        return {
            "highres": read_image(self.highres_manifest[index]).float(),
            "lowres": read_image(self.lowres_manifest[index]).float(),
        }
