import os
from collections import Callable
from functools import partial
from typing import Optional, Any
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.datasets.folder import pil_loader


class ImagesDataset(Dataset):
    """
    load unlabeled image data from a folder

    inside the folder
        * meta.csv: a file contains a list of image names
        * images
    """

    def __init__(
            self,
            root_dir: str,
            transform: Optional[Callable] = None,
            loader: Callable = pil_loader,
    ) -> None:

        meta_file = os.path.join(root_dir, 'meta.csv')
        self.image_files = pd.read_csv(meta_file)
        self.root_dir = root_dir
        self.transform = transform
        self.loader = loader

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, self.image_files.iloc[idx, 0])
        image = self.loader(img_name)

        if self.transform:
            image = self.transform(image)

        return image


def np_loader(var: str, path: str) -> np.array:
    with np.load(path) as d:
        return d[var]


class NumpyDataset(Dataset):
    """
    load numpy array data from a folder

    inside the folder
        * meta.csv: a file contains a list of image names
        * numpy arrays
    """

    def __init__(
            self,
            root_dir: str,
            transform: Optional[Callable] = None,
            loader: Callable = np_loader,
            var: str = 'z',
    ) -> None:

        meta_file = os.path.join(root_dir, 'meta.csv')
        self.np_files = pd.read_csv(meta_file)
        self.root_dir = root_dir
        self.transform = transform
        self.loader = partial(loader, var)

    def __len__(self):
        return len(self.np_files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        np_file = os.path.join(self.root_dir, self.np_files.iloc[idx, 0])
        z = torch.from_numpy(self.loader(np_file))

        if self.transform:
            z = self.transform(z)

        return z
