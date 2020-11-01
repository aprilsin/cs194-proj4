# This file is for loading images and keypoints customized for the Danes dataset.
# data source: http://www2.imm.dtu.dk/~aam/datasets/datasets.html.
import argparse
import functools
import itertools
import logging
import math
import os
import random
import re
import shutil
import subprocess
import sys
import typing
import xml.etree.ElementTree as ET
from copy import deepcopy
from functools import reduce
from logging import debug, info, log
from pathlib import Path
from typing import (Callable, Dict, FrozenSet, Iterable, List, NamedTuple,
                    NewType, Optional, Sequence, Set, Tuple, TypeVar, Union)
from xml.etree import Element

import numpy as np
import numpy as np
import pandas as pd
import skimage.transform as ST
import torch
import torch.nn.functional as F
import torchvision
import torchvision.io as TIO
import torchvision.transforms as TT
import torchvision.transforms.functional as TF
from skimage.util import img_as_float
from torch import Tensor, distributions, nn, tensor
from torch.nn import Linear, ReLU, Sequential, Softmax
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision import utils

DANES_ROOT = Path("imm_face_db")
IBUG_ROOT = Path("ibug_300W_large_face_landmark_dataset")
# assert DANES_ROOT.exists()
# assert IBUG_ROOT.exists()


def assert_points(pts):
    assert isinstance(pts, Tensor), type(pts)
    assert pts.ndim == 2, pts.shape
    assert pts.shape[1] == 2, pts.shape

    # make sure that the keypoints are ratios
    rows = pts[:, 0]
    cols = pts[:, 1]
    # leave some wiggle room so use 1.5 instead of 1
    assert rows.max() <= 1.5 and cols.max() <= 1.5, f"{rows.max()}, {cols.max()}"

    return True


def assert_img(img):
    assert isinstance(img, Tensor), type(img)
    assert img.ndim == 3, img.shape
    assert list(img.shape)[0] == 1, f"{img.shape} is not grayscale"
    return True


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
    assert_points(points)
    return points


def load_nose(file: os.PathLike) -> Tensor:
    points = load_asf(file)
    NOSE_INDEX = 53 - 1  # nose is 53rd keypoint, minus 1 for zero-index
    nose_point = points[NOSE_INDEX].reshape(1, 2)
    assert_points(nose_point)
    return nose_point


def load_img(img_file: Path):
    t = torchvision.io.read_image(str(img_file))
    pipeline = TT.Compose(
        [
            TT.ToPILImage(),
            TT.ToTensor(),
            TT.Grayscale(),
        ]
    )
    img = pipeline(t)
    assert_img(img)
    return img


def load_xml(root_dir: Path, filename: Element) -> Tuple[Tensor, Tensor]:

    img_name = root_dir / filename.attrib["file"]
    img = load_img(img_name)

    h, w = img.shape[-2:]

    keypts = []
    for num in range(68):
        x_coordinate = int(filename[0][num].attrib["x"])
        y_coordinate = int(filename[0][num].attrib["y"])
        keypts.append([x_coordinate, y_coordinate])
    keypts = torch.as_tensor(keypts, dtype=torch.float32)

    # crop image background

    box = filename[0].attrib
    left, top, width, height = (
        int(box["left"]),
        int(box["top"]),
        int(box["width"]),
        int(box["height"]),
    )

    # ratio for adjusting box
    vr = 1.4
    hr = 1.2
    ver_shift = int(round(height * (vr - 1) / 2))
    hor_shift = int(round(width * (hr - 1) / 2))

    # x, y for the top left corner of the box, w, h for box width and height
    img = TT.functional.crop(
        img,
        top=top - ver_shift,
        left=left - hor_shift,
        height=int(round(height * vr)),
        width=int(round(width * hr)),
    )

    # fix keypoints according to crop
    keypts[:, 0] -= left - hor_shift
    keypts[:, 1] -= top - ver_shift

    # make keypoints ratios
    h, w = img.shape[-2:]
    keypts[:, 0] /= w
    keypts[:, 1] /= h

    assert_img(img)
    assert_points(keypts)
    return img, keypts


def part1_augment(image, keypoints) -> Tuple[Tensor, Tensor]:
    h, w = image.shape[-2:]

    # resize
    out_h, out_w = 240, 320
    image = TT.Resize((out_h, out_w))(image)
    # keypoints[..., 0] = keypoints[..., 0] * out_h / h
    # keypoints[..., 1] = keypoints[..., 1] * out_w / w

    return image, keypoints


def part2_augment(image, keypoints) -> Tuple[Tensor, Tensor]:
    # print(image.shape, keypoints.shape)

    # TODO make dataloader for colors for this to work
    # jitter = TT.ColorJitter(brightness=0.3, saturation=0.2)
    # image = jitter(image)

    # convert tensors to numpy arrays to use skimage
    image, keypoints = image.squeeze().numpy(), keypoints.numpy()
    h, w = image.shape
    center = (h / 2, w / 2)

    rotate_deg = np.random.randint(-12, 12)
    image = ST.rotate(image, angle=rotate_deg, center=(0, 0))
    for i in range(len(keypoints)):
        point = keypoints[i]
        qx, qy = rotate(point, (0, 0), -rotate_deg)
        keypoints[i][0] = qx
        keypoints[i][1] = qy

    # convert back to tensors
    image = torch.from_numpy(image)
    image = torch.unsqueeze(image, 0)
    keypoints = torch.from_numpy(keypoints)
    # print(image.shape, keypoints.shape)
    return image, keypoints


def rotate(point, origin, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in degrees.
    """
    angle = np.radians(angle)
    ox, oy = origin
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return qx, qy


class FaceKeypointsDataset(Dataset):
    def __init__(
        self,
        idxs: Sequence[int],
        root_dir: Path = DANES_ROOT,
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
        points = load_asf(asf_name)

        if self.transform is not None:
            img, points = self.transform(img, points)

        assert_img(img)
        assert_points(points)
        return img, points


class NoseKeypointDataset(FaceKeypointsDataset):
    def __init__(
        self,
        idxs: Sequence[int],
        root_dir: Path = DANES_ROOT,
        transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(idxs, root_dir, transform)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        # TODO add augmentations with if random.random()<THRESHOLD

        img, points = super().__getitem__(idx)

        NOSE_INDEX = 53 - 1  # nose is 53rd keypoint, minus 1 for zero-index
        nose_point = points[NOSE_INDEX].reshape(1, 2)
        return img, nose_point


class LargeDataset(Dataset):
    def __init__(
        self,
        idxs: Sequence[int],
        root_dir: Path = IBUG_ROOT,
        transform: Optional[Callable] = None,
    ) -> None:
        self.root_dir = root_dir
        self.tree = ET.parse(self.root_dir / "labels_ibug_300W_train.xml")
        # root = tree.getroot()
        self.files = self.tree.getroot()[2]  # should be 6666
        self.len = len(self.files)

    def __len__(self):
        return self.len  # should be 6666

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:

        filename = self.files[idx]
        img, keypts = load_xml(self.root_dir, filename)
        assert_img(img)
        assert_points(keypts)
        return img, keypts


def save_kaggle(keypoints):
    # TODO
    """
    Saves predicted keypoints of Part 3 test set as a csv file
    test set source = https://inst.eecs.berkeley.edu/~cs194-26/fa20/hw/proj4/labels_ibug_300W_test_parsed.xml
    """
    pass
