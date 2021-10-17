import gzip
import os
from collections import Callable
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class NumpyDataset(Dataset):
    """
    load unlabeled numpy data from a folder

    inside the folder
        * meta.csv: a file contains a list of image names
        * numpy gziped files
    """

    def __init__(
            self,
            root_dir: str,
            max_seq_length: int,
            padding_file: str, # numpy file for black image code, we use it to pad in case the sequence length is small
            transform: Optional[Callable] = None
    ) -> None:

        meta_file = os.path.join(root_dir, 'meta.csv')
        self.numpy_files = pd.read_csv(meta_file)
        self.root_dir = root_dir
        self.transform = transform

        # for video codes, the np array has shape [#frame, 32, 32]
        # we only keep #max_seq_length frames
        self.max_seq_length = max_seq_length

        # if the #frame < #max_seq_length, we append padding
        # padding is normally the code of black images after passing through vqvae for images
        # padding shape [1, 32, 32]
        with gzip.GzipFile(padding_file, 'r') as f:
            self.padding = np.load(f)

    def __len__(self):
        return len(self.numpy_files)

    def __getitem__(self, idx: int) -> np.ndarray:
        if torch.is_tensor(idx):
            idx = idx.tolist()

        file_name = os.path.join(self.root_dir, self.numpy_files.iloc[idx, 0])
        with gzip.GzipFile(file_name, 'r') as f:
            codes = np.load(f)

        n = codes.shape[0]
        codes = codes[:min(n, self.max_seq_length)]
        if n < self.max_seq_length:
            # pad more frames
            pad = np.repeat(self.padding, self.max_seq_length - n, axis=0)
            codes = np.concatenate([codes, pad], axis=0)

        if self.transform:
            codes = self.transform(codes)
        return codes