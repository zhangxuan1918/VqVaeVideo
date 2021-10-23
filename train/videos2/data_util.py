import gzip
import os
import random

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset


class VideoNumpyDataset(Dataset):
    """
    load unlabeled numpy data from a folder

    inside the folder
        * meta.csv: a file contains a list of image names
        * numpy gziped files
    """

    def __init__(
            self,
            root_dir: str,
            sequence_length: int,
            padding_file: str,  # numpy file for black image code, we use it to pad in case the sequence length is small
    ) -> None:

        meta_file = os.path.join(root_dir, 'meta.csv')
        self.numpy_files = pd.read_csv(meta_file)
        self.root_dir = root_dir

        # for video codes, the np array has shape [#frame, 32, 32]
        # we only keep #max_seq_length frames
        self.sequence_length = sequence_length

        with gzip.GzipFile(padding_file, 'r') as f:
            self.padding = np.load(f).astype(np.int)

    def __len__(self):
        return self.numpy_files.shape[0]

    def __getitem__(self, idx: int) -> np.ndarray:
        if torch.is_tensor(idx):
            idx = idx.tolist()

        file_name = os.path.join(self.root_dir, self.numpy_files.iloc[idx, 0])
        with gzip.GzipFile(file_name, 'r') as f:
            codes = np.load(f).astype(np.int)

        n = codes.shape[0]

        if n < self.sequence_length:
            # pad more frames
            pad = np.repeat(self.padding, self.sequence_length - n, axis=0)
            codes = np.concatenate([codes, pad], axis=0)
        elif n > self.sequence_length:
            # randomly select `sequence_length` frames
            s = random.randint(0, n-self.sequence_length)
            codes = codes[s:s + self.sequence_length]

        return codes
