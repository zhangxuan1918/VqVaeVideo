import os
from collections import Callable
from typing import Optional

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.datasets.folder import pil_loader


class VideoDataset(Dataset):
    """
    load unlabeled video data from a folder

    inside the folder
        * meta.csv: a file contains a list of video names
        * videos
    """

    def __init__(
            self,
            root_dir: str,
            max_seq_length: int,
            transform: Optional[Callable] = None
    ) -> None:

        meta_file = os.path.join(root_dir, 'meta.csv')
        self.video_files = pd.read_csv(meta_file)
        self.root_dir = root_dir
        self.max_seq_length = max_seq_length
        self.transform = transform
        # black images as padding
        # if the #frames in the video < max_seq_length, we pad black images
        self.padding = np.zeros((1, 256, 256, 3))

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        video_file = os.path.join(self.root_dir, self.video_files.iloc[idx, 0])
        video_in = cv2.VideoCapture(video_file)
        frames = []

        while video_in.isOpened():
            ret, frame = video_in.read()
            if ret:
                frames.append(frame)
            else:
                break
            if len(frames) == self.max_seq_length:
                break
        video_in.release()
        cv2.destroyAllWindows()

        video = np.stack(frames, 0)
        n = video.shape[0]
        if n < self.max_seq_length:
            # pad more frames
            pad = np.repeat(self.padding, self.max_seq_length - n, axis=0)
            video = np.concatenate([video, pad], axis=0)

        if self.transform:
            video = self.transform(video)
        video = np.transpose(video, (0, 3, 1, 2))
        return video
