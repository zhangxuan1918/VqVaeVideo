import os
from collections import Callable
from typing import Optional, Any

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
