import os, io, csv, math, random
import numpy as np
from einops import rearrange
from decord import VideoReader

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset
from io import BytesIO
import torchvision.transforms as transforms
import pandas as pd
import json
import pickle


class WebVid10M_Single_Image(Dataset):
    def __init__(
        self,
        sample_size=128,
        max_sample_stride=90,
        sample_n_frames=1,
    ):

        self.video_folder = video_folder = "/mnt/data/data/webvid"
        self.dataset = [
            os.path.join(video_folder, video_path)
            for video_path in os.listdir(video_folder)
            if video_path.endswith(("mp4",))
        ]
        self.length = len(self.dataset)
        print(f"data scale: {self.length}")

        self.max_sample_stride = max_sample_stride
        self.sample_n_frames = sample_n_frames

        self.sample_size = sample_size = (
            tuple(sample_size)
            if not isinstance(sample_size, int)
            else (sample_size, sample_size)
        )
        self.pixel_transforms = transforms.Compose(
            [
                transforms.Resize(sample_size),
            ]
        )

        self.root_dir = "crawler:s3://"

    def get_batch(self, idx):

        video_dir = self.dataset[idx]
        name = open(video_dir.replace("mp4", "txt"), "r").readline().strip()
        video_reader = VideoReader(video_dir)
        video_length = len(video_reader)

        stride = random.randint(1, self.max_sample_stride)
        # stride = 150
        start_idx = random.randint(0, video_length - stride)
        # start_idx = 0

        pixel_values = (
            torch.from_numpy(
                video_reader.get_batch([start_idx, start_idx + stride]).asnumpy()
            )
            .permute(0, 3, 1, 2)
            .contiguous()
        )
        pixel_values = pixel_values / 255.0
        del video_reader

        return pixel_values, name, stride

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        while True:
            try:
                pixel_values, name, stride = self.get_batch(idx)
                break
            except Exception as e:
                # print(e)
                idx = random.randint(0, self.length - 1)

        pixel_values = self.pixel_transforms(pixel_values)
        sample = dict(
            pixel_values=pixel_values, text=name, stride=torch.tensor(stride).float()
        )
        return sample


class WebVid10M_Multiple_Image_NormandUnnorm_Anime(Dataset):
    def __init__(
        self, sample_size=256, sample_stride=6, sample_n_frames=16, use_norm=False
    ):

        anno_path = "merged_output_with_contents_filtered_wofolder.jsonl"
        video_folder = "/mnt/data_15/"

        if anno_path is not None:
            self.dataset = []
            if isinstance(anno_path, str):
                with open(anno_path, "r") as f:
                    for line in f:
                        data = json.loads(line)
                        if data.get("ocr_valid", True):
                            self.dataset.append(data)
            else:
                # is tuple or list
                for path, folder in zip(anno_path, video_folder):
                    with open(path, "r") as f:
                        for line in f:
                            raw = json.loads(line)
                            raw["video_path"] = os.path.join(folder, raw["video_path"])
                            self.dataset.append(raw)
                video_folder = ""
        else:
            # there are really too many videos to read them from another server.
            # WORKAROUND
            self.dataset = []
            with open("/mnt/data_58/video_data/test.txt", "r") as f:
                for line in f:
                    self.dataset.append(os.path.join(video_folder, line.strip()))
            # self.dataset = [os.path.join(video_folder,video_path) for video_path in os.listdir(video_folder) if video_path.endswith(("mp4",))]
        self.anno_path = anno_path
        self.length = len(self.dataset)

        self.video_folder = video_folder

        self.sample_stride = sample_stride
        self.sample_n_frames = sample_n_frames

        sample_size = (
            tuple(sample_size)
            if not isinstance(sample_size, int)
            else (sample_size, sample_size)
        )
        self.pixel_transforms_norm = transforms.Compose(
            [
                transforms.Resize(sample_size),
                transforms.Normalize(
                    mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True
                ),
            ]
        )

        self.pixel_transforms = transforms.Compose(
            [
                transforms.Resize(sample_size),
                # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
            ]
        )

    def get_batch(self, idx):
        # video_dict = self.dataset[idx]

        motion_caption = ""

        if self.anno_path is not None:
            video_dir = os.path.join(self.video_folder, self.dataset[idx]["video_path"])
            name = self.dataset[idx]["caption"]
            if "motion_caption" in self.dataset[idx]:
                motion_caption = self.dataset[idx]["motion_caption"]
        else:
            video_dir = self.dataset[idx]
            name = open(video_dir.replace("mp4", "txt"), "r").readline().strip()

        video_reader = VideoReader(video_dir)
        video_length = len(video_reader)

        clip_length = min(
            video_length, (self.sample_n_frames - 1) * self.sample_stride + 1
        )
        start_idx = random.randint(0, video_length - clip_length)
        batch_index = np.linspace(
            start_idx, start_idx + clip_length - 1, self.sample_n_frames, dtype=int
        )

        pixel_values = (
            torch.from_numpy(video_reader.get_batch(batch_index).asnumpy())
            .permute(0, 3, 1, 2)
            .contiguous()
        )
        pixel_values = pixel_values / 255.0
        del video_reader

        return pixel_values, name, batch_index

    def __len__(self):
        return self.length

    def process_stride(self, batch_index):
        batch_index = torch.from_numpy(batch_index)
        batch_index = batch_index - batch_index[0]

        return batch_index

    def __getitem__(self, idx):
        while True:
            try:
                pixel_values, name, batch_index = self.get_batch(idx)
                break

            except Exception as e:
                print(e)
                idx = random.randint(0, self.length - 1)

        pixel_values_norm = self.pixel_transforms_norm(pixel_values.clone())
        pixel_values = self.pixel_transforms(pixel_values)
        batch_index = self.process_stride(batch_index)
        sample = dict(
            pixel_values=pixel_values,
            pixel_values_norm=pixel_values_norm,
            text=name,
            stride=batch_index.float(),
        )
        return sample


class WebVid10M_Multiple_Image_NormandUnnorm(Dataset):
    def __init__(
        self,
        sample_size=128,
        sample_stride=90,
        sample_n_frames=1,
        use_norm=False,
    ):

        self.video_folder = video_folder = "/mnt/data/data/webvid"
        self.dataset = [
            os.path.join(video_folder, video_path)
            for video_path in os.listdir(video_folder)
            if video_path.endswith(("mp4",))
        ]
        self.length = len(self.dataset)
        print(f"data scale: {self.length}")

        self.sample_stride = sample_stride
        self.sample_n_frames = sample_n_frames

        self.sample_size = sample_size = (
            tuple(sample_size)
            if not isinstance(sample_size, int)
            else (sample_size, sample_size)
        )

        self.pixel_transforms_norm = transforms.Compose(
            [
                transforms.Resize(sample_size),
                transforms.Normalize(
                    mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True
                ),
            ]
        )

        self.pixel_transforms = transforms.Compose(
            [
                transforms.Resize(sample_size),
                # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
            ]
        )

        self.root_dir = "crawler:s3://"

    def get_batch(self, idx):

        video_dir = self.dataset[idx]
        name = open(video_dir.replace("mp4", "txt"), "r").readline().strip()
        video_reader = VideoReader(video_dir)
        video_length = len(video_reader)

        stride = self.sample_stride
        clip_length = min(video_length, (self.sample_n_frames - 1) * stride + 1)
        start_idx = random.randint(0, video_length - clip_length)
        batch_index = np.linspace(
            start_idx, start_idx + clip_length - 1, self.sample_n_frames, dtype=int
        )

        pixel_values = (
            torch.from_numpy(video_reader.get_batch(batch_index).asnumpy())
            .permute(0, 3, 1, 2)
            .contiguous()
        )
        pixel_values = pixel_values / 255.0
        del video_reader

        return pixel_values, name, batch_index

    def __len__(self):
        return self.length

    def process_stride(self, batch_index):
        batch_index = torch.from_numpy(batch_index)
        batch_index = batch_index - batch_index[0]

        return batch_index

    def __getitem__(self, idx):
        while True:
            try:
                pixel_values, name, batch_index = self.get_batch(idx)
                break
            except Exception as e:
                # print(e)
                idx = random.randint(0, self.length - 1)

        pixel_values_norm = self.pixel_transforms_norm(pixel_values.clone())
        pixel_values = self.pixel_transforms(pixel_values)
        batch_index = self.process_stride(batch_index)
        sample = dict(
            pixel_values=pixel_values,
            pixel_values_norm=pixel_values_norm,
            text=name,
            stride=batch_index.float(),
        )
        return sample


class WebVid10M_Multiple_Image_Anime(Dataset):
    def __init__(
        self, sample_size=256, sample_stride=6, sample_n_frames=16, use_norm=False
    ):

        anno_path = "merged_output_with_contents_filtered_wofolder.jsonl"
        video_folder = "/mnt/data_15/"

        if anno_path is not None:
            self.dataset = []
            if isinstance(anno_path, str):
                with open(anno_path, "r") as f:
                    for line in f:
                        data = json.loads(line)
                        if data.get("ocr_valid", True):
                            self.dataset.append(data)
            else:
                # is tuple or list
                for path, folder in zip(anno_path, video_folder):
                    with open(path, "r") as f:
                        for line in f:
                            raw = json.loads(line)
                            raw["video_path"] = os.path.join(folder, raw["video_path"])
                            self.dataset.append(raw)
                video_folder = ""
        else:
            # there are really too many videos to read them from another server.
            # WORKAROUND
            self.dataset = []
            with open("/mnt/data_58/video_data/test.txt", "r") as f:
                for line in f:
                    self.dataset.append(os.path.join(video_folder, line.strip()))
            # self.dataset = [os.path.join(video_folder,video_path) for video_path in os.listdir(video_folder) if video_path.endswith(("mp4",))]
        self.anno_path = anno_path
        self.length = len(self.dataset)

        self.video_folder = video_folder

        self.sample_stride = sample_stride
        self.sample_n_frames = sample_n_frames

        sample_size = (
            tuple(sample_size)
            if not isinstance(sample_size, int)
            else (sample_size, sample_size)
        )
        if use_norm:
            self.pixel_transforms = transforms.Compose(
                [
                    transforms.Resize(sample_size),
                    transforms.Normalize(
                        mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True
                    ),
                ]
            )
        else:
            self.pixel_transforms = transforms.Compose(
                [
                    transforms.Resize(sample_size),
                    # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
                ]
            )

    def get_batch(self, idx):
        # video_dict = self.dataset[idx]

        motion_caption = ""

        if self.anno_path is not None:
            video_dir = os.path.join(self.video_folder, self.dataset[idx]["video_path"])
            name = self.dataset[idx]["caption"]
            if "motion_caption" in self.dataset[idx]:
                motion_caption = self.dataset[idx]["motion_caption"]
        else:
            video_dir = self.dataset[idx]
            name = open(video_dir.replace("mp4", "txt"), "r").readline().strip()

        video_reader = VideoReader(video_dir)
        video_length = len(video_reader)

        clip_length = min(
            video_length, (self.sample_n_frames - 1) * self.sample_stride + 1
        )
        start_idx = random.randint(0, video_length - clip_length)
        batch_index = np.linspace(
            start_idx, start_idx + clip_length - 1, self.sample_n_frames, dtype=int
        )

        pixel_values = (
            torch.from_numpy(video_reader.get_batch(batch_index).asnumpy())
            .permute(0, 3, 1, 2)
            .contiguous()
        )
        pixel_values = pixel_values / 255.0
        del video_reader

        return pixel_values, name, batch_index

    def __len__(self):
        return self.length

    def process_stride(self, batch_index):
        batch_index = torch.from_numpy(batch_index)
        batch_index = batch_index - batch_index[0]

        return batch_index

    def __getitem__(self, idx):
        while True:
            try:
                pixel_values, name, batch_index = self.get_batch(idx)
                break

            except Exception as e:
                print(e)
                idx = random.randint(0, self.length - 1)

        pixel_values = self.pixel_transforms(pixel_values)
        batch_index = self.process_stride(batch_index)
        sample = dict(pixel_values=pixel_values, text=name, stride=batch_index.float())
        return sample


class WebVid10M_Multiple_Image(Dataset):
    def __init__(
        self,
        sample_size=128,
        sample_stride=6,
        sample_n_frames=16,
        use_norm=False,
    ):

        self.video_folder = video_folder = "/mnt/data/data/webvid"
        self.dataset = [
            os.path.join(video_folder, video_path)
            for video_path in os.listdir(video_folder)
            if video_path.endswith(("mp4",))
        ]
        self.length = len(self.dataset)
        print(f"data scale: {self.length}")

        self.sample_n_frames = sample_n_frames

        self.sample_size = sample_size = (
            tuple(sample_size)
            if not isinstance(sample_size, int)
            else (sample_size, sample_size)
        )

        self.sample_stride = sample_stride
        self.sample_n_frames = sample_n_frames

        if use_norm:
            self.pixel_transforms = transforms.Compose(
                [
                    transforms.Resize(sample_size),
                    transforms.Normalize(
                        mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True
                    ),
                ]
            )
        else:
            self.pixel_transforms = transforms.Compose(
                [
                    transforms.Resize(sample_size),
                    # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
                ]
            )

        self.root_dir = "crawler:s3://"

    def get_batch(self, idx):

        video_dir = self.dataset[idx]
        name = open(video_dir.replace("mp4", "txt"), "r").readline().strip()
        video_reader = VideoReader(video_dir)
        video_length = len(video_reader)

        clip_length = min(
            video_length, (self.sample_n_frames - 1) * self.sample_stride + 1
        )
        start_idx = random.randint(0, video_length - clip_length)
        batch_index = np.linspace(
            start_idx, start_idx + clip_length - 1, self.sample_n_frames, dtype=int
        )
        pixel_values = (
            torch.from_numpy(video_reader.get_batch(batch_index).asnumpy())
            .permute(0, 3, 1, 2)
            .contiguous()
        )
        pixel_values = pixel_values / 255.0
        del video_reader

        return pixel_values, name, batch_index

    def __len__(self):
        return self.length

    def process_stride(self, batch_index):
        batch_index = torch.from_numpy(batch_index)
        batch_index = batch_index - batch_index[0]

        return batch_index

    def __getitem__(self, idx):
        while True:
            try:
                pixel_values, name, batch_index = self.get_batch(idx)
                break
            except Exception as e:
                print(e)
                idx = random.randint(0, self.length - 1)

        pixel_values = self.pixel_transforms(pixel_values)
        batch_index = self.process_stride(batch_index)
        sample = dict(pixel_values=pixel_values, text=name, stride=batch_index.float())
        return sample


if __name__ == "__main__":

    dataset = WebVid10M_Single_Image()

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=4,
        num_workers=1,
    )
    for idx, batch in enumerate(dataloader):
        print(batch["pixel_values"].shape, len(batch["text"]))

        if idx > 1000:
            exit()
