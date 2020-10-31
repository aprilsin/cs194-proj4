# This file is for loading images and keypoints customized for the Danes dataset.
# data source: http://www2.imm.dtu.dk/~aam/datasets/datasets.html.
import argparse
import functools
import itertools
import logging
import os
import random
import re
import shutil
import subprocess
import sys
import typing
from copy import deepcopy
from functools import reduce
from logging import debug, info, log
from pathlib import Path
from typing import (
    Callable,
    Dict,
    FrozenSet,
    Iterable,
    List,
    NamedTuple,
    NewType,
    Optional,
    Sequence,
    Set,
    Tuple,
    TypeVar,
    Union,
)

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import torchvision.io as TIO
import torchvision.transforms as TT
import torchvision.transforms.functional as TF
from skimage import io, transform
from skimage.util import img_as_float
from torch import Tensor, distributions, nn, tensor
from torch.nn import Linear, ReLU, Sequential, Softmax
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision import utils


def load_asf(file: os.PathLike) -> Tensor:
    file = Path(file)
    with file.open() as f:
        lines_read = f.readlines()

    num_pts = int(lines_read[9])
    assert num_pts == 58, num_pts

    lines = lines_read[16 : num_pts + 16]  # basically should be [16, 74]
    points = []
    for line in lines:
        data = line.split("\t")
        x, y = float(data[2]), float(data[3])
        points.append([x, y])

    points = torch.as_tensor(points, dtype=torch.float32)
    assert len(points) == num_pts, len(points)
    assert points.shape == (num_pts, 2)
    return points


def load_nose(file: os.PathLike) -> Tensor:
    points = load_asf(file)
    NOSE_INDEX = 53 - 1  # nose is 53rd keypoint, minus 1 for zero-index
    nose_point = points[NOSE_INDEX].reshape(1, 2)
    return nose_point


def load_img(img_file: Path):
    t = torchvision.io.read_image(str(img_file))
    pipeline = TT.Compose(
        [
            TT.ToPILImage(),
            TT.ToTensor(),
            TT.Grayscale(),
            # TODO add Resize
        ]
    )
    img = pipeline(t)
    return img


class FaceKeypointsDataset(Dataset):
    def __init__(
        self,
        idxs: Sequence[int],
        root_dir: Path,
        transform: Optional[Callable] = None,
    ) -> None:
        self.root_dir = root_dir
        self.img_files = sorted(
            f for f in self.root_dir.glob("*.jpg") if int(f.name.split("-")[0]) in idxs
        )
        self.asf_files = sorted(
            f for f in self.root_dir.glob("*.asf") if int(f.name.split("-")[0]) in idxs
        )
        self.len = len(self.img_files)
        self.transform = transform

    def __len__(self) -> int:
        return self.len

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        # TODO add augmentations with if random.random()<THRESHOLD

        img_name, asf_name = self.img_files[idx], self.asf_files[idx]
        img = load_img(img_name)
        h, w = img.shape[-2:]
        points = load_asf(asf_name)
        points[:, 0] *= w
        points[:, 1] *= h
        # TODO is rounding necessary?
        # points=points.round()

        if self.transform is not None:
            img, points = self.transform(img, points)

        assert isinstance(img, Tensor), type(img)
        assert isinstance(points, Tensor), type(points)
        return img, points


class NoseKeypointDataset(FaceKeypointsDataset):
    def __init__(
        self,
        idxs: Sequence[int],
        root_dir: Path = ROOT_DIR,
        transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(idxs, root_dir, transform)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        # TODO add augmentations with if random.random()<THRESHOLD

        img, points = super().__getitem__(idx)

        NOSE_INDEX = 53 - 1  # nose is 53rd keypoint, minus 1 for zero-index
        nose_point = points[NOSE_INDEX].reshape(1, 2)

        return img, nose_point


if __name__ == "__main__":
    train = FaceKeypointsDataset(idxs=range(1, 33))
    val = FaceKeypointsDataset(idxs=range(33, 41))
