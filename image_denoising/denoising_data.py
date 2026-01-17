#定义模块的公开接口，仅暴露FolderDataset类
__all__ = ['FolderDataset']

from PIL import Image
import os
from torch.utils.data import Dataset
import denoising_config
import torch
import numpy as np
#正则表达式
import re


def sorted_alphanumeric(data):
    """Sort the given iterable in the way that humans expect."""
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(data, key=alphanum_key)

class FolderDataset(Dataset):
    """Custom Dataset for loading images from a folder."""
    def __init__(self, img_dir, transform=None):
        """
        Args:
            img_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.img_dir = img_dir
        self.transform = transform
        self.img_names = sorted_alphanumeric(os.listdir(img_dir))

    def __len__(self):
        return len(self.all_imgs)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_loc = os.path.join(self.img_dir, self.img_names[idx])
        image = Image.open(img_loc).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image

    noisy_imgs=tensor_image + denoising_config.NOISE_FACTOR * torch.randn(*tensor_image.shape)
        noisy_imgs=torch.clip(noisy_imgs,0.,1.)
        return noisy_imgs, tensor_image

