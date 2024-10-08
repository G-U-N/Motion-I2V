import os, io, csv, math, random
import numpy as np
from einops import rearrange
from decord import VideoReader

import torch
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset
from animation.utils.util import zero_rank_print

from petrel_client.client import Client

_client = Client()
from io import BytesIO
import torchvision.transforms as transforms
import pandas as pd


class WebVid10M(Dataset):
    def __init__(
        self,
        csv_path,
        video_folder,
        sample_size=256,
        sample_stride=4,
        sample_n_frames=16,
        is_image=False,
    ):

        csv_path = "crawler:s3://webvid/meta/results_2M_train.csv"
        csv_data = BytesIO(_client.get(csv_path))
        data = pd.read_csv(csv_data)
        data = data.values.tolist()

        self.dataset = []
        for _d in data:
            self.dataset.append([str(_d[0]) + ".mp4", _d[1]])

        self.length = len(self.dataset)
        print(f"data scale: {self.length}")

        self.video_folder = video_folder
        self.sample_stride = sample_stride
        self.sample_n_frames = sample_n_frames
        self.is_image = is_image

        sample_size = (
            tuple(sample_size)
            if not isinstance(sample_size, int)
            else (sample_size, sample_size)
        )
        self.pixel_transforms = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.Resize(sample_size[0]),
                transforms.CenterCrop(sample_size),
                transforms.Normalize(
                    mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True
                ),
            ]
        )

        self.root_dir = "crawler:s3://webvid/data/2M/"

    def get_batch(self, idx):

        video_path = self.root_dir + self.dataset[idx][0]
        name = self.dataset[idx][1]
        video_data = BytesIO(_client.get(video_path))
        video_reader = VideoReader(video_data)

        video_length = len(video_reader)

        if not self.is_image:
            clip_length = min(
                video_length, (self.sample_n_frames - 1) * self.sample_stride + 1
            )
            start_idx = random.randint(0, video_length - clip_length)
            batch_index = np.linspace(
                start_idx, start_idx + clip_length - 1, self.sample_n_frames, dtype=int
            )
        else:
            batch_index = [random.randint(0, video_length - 1)]

        pixel_values = (
            torch.from_numpy(video_reader.get_batch(batch_index).asnumpy())
            .permute(0, 3, 1, 2)
            .contiguous()
        )
        pixel_values = pixel_values / 255.0
        del video_reader

        if self.is_image:
            pixel_values = pixel_values[0]

        return pixel_values, name

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        while True:
            try:
                pixel_values, name = self.get_batch(idx)
                break

            except Exception as e:
                print(e)
                idx = random.randint(0, self.length - 1)

        pixel_values = self.pixel_transforms(pixel_values)
        sample = dict(pixel_values=pixel_values, text=name)
        return sample


if __name__ == "__main__":
    pass
