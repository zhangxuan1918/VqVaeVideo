import gzip
import os
from collections import Callable
from typing import Optional, Tuple

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
            max_frame_length: int,
            vqvae_vocab_size: int,
            transform: Optional[Callable] = None
    ) -> None:

        meta_file = os.path.join(root_dir, 'meta.csv')
        self.numpy_files = pd.read_csv(meta_file)
        self.root_dir = root_dir
        self.transform = transform

        # for video codes, the np array has shape [#frame, 32, 32]
        # we only keep #max_seq_length frames
        self.max_frame_length = max_frame_length
        self.vqvae_vocab_size = vqvae_vocab_size

    def __len__(self):
        return len(self.numpy_files)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, int]:
        if torch.is_tensor(idx):
            idx = idx.tolist()

        file_name = os.path.join(self.root_dir, self.numpy_files.iloc[idx, 0])
        with gzip.GzipFile(file_name, 'r') as f:
            codes = np.load(f).astype(np.long)

        n, h, w = codes.shape
        if n < self.max_frame_length:
            # pad more frames
            # if the #frame < #max_frame_length, we append padding
            # we pad 8192 because vqvae has code 0 to 8191
            padding = np.ones((1, h, w), dtype=codes.dtype) * self.vqvae_vocab_size
            pad = np.repeat(padding, self.max_frame_length - n, axis=0)
            codes = np.concatenate([codes, pad], axis=0)
        # random sample
        s = np.random.randint(0, max(0, n - self.max_frame_length) + 1)
        codes = codes[s: s+self.max_frame_length]
        if self.transform:
            codes = self.transform(codes)
        return codes
