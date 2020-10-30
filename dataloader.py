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
from logging import (
    debug,
    info,
    log,
)
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
from skimage import (
    io,
    transform,
)
from skimage.util import img_as_float
from torch import (
    Tensor,
    distributions,
    nn,
    tensor,
)
from torch.nn import (
    Linear,
    ReLU,
    Sequential,
    Softmax,
)
from torch.optim import Adam
from torch.utils.data import (
    DataLoader,
    Dataset,
    TensorDataset,
)
from torchvision import utils


ROOT_DIR = Path("imm_face_db")


def get_gender(person_idx: int) -> str:
    """For filename handling."""
    assert 1 <= person_idx <= 40, person_idx

    female_idx = [8, 12, 14, 15, 22, 30, 35]

    return "f" if person_idx in female_idx else "m"


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
    def __init__(self, idxs: Sequence[int], root_dir: Path = ROOT_DIR) -> None:
        self.root_dir = root_dir
        self.img_files = sorted(
            f for f in self.root_dir.glob("*.jpg") if int(f.name.split("-")[0]) in idxs
        )
        self.asf_files = sorted(
            f for f in self.root_dir.glob("*.asf") if int(f.name.split("-")[0]) in idxs
        )
        self.len = len(self.img_files)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        # TODO add augmentations with if random.random()<THRESHOLD

        img_name, asf_name = self.img_files[idx], self.asf_files[idx]
        img = load_img(img_name)
        h, w = img.shape[-2:]
        points = load_asf(asf_name)
        # N (x,y)
        points[:, 0] *= w
        points[:, 1] *= h
        # TODO is rounding necessary?
        # points=points.round()

        return img, points

    def __len__(self) -> int:
        return self.len


if __name__ == "__main__":
    train = FaceKeypointsDataset(idxs=range(1, 33))
    val = FaceKeypointsDataset(idxs=range(33, 41))
