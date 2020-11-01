# This file is for loading images and keypoints customized for the Danes and ibug dataset.
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
    image = ST.rotate(image, angle=rotate_deg, center=(h / 2, w / 2))
    for i in range(len(keypoints)):
        point = keypoints[i]
        x, y = point[0] * w, point[1] * h
        qx, qy = rotate(point=(x, y), origin=(h / 2, w / 2), angle=-rotate_deg)
        keypoints[i][0] = qx / w
        keypoints[i][1] = qy / h

    # convert back to tensors
    image = torch.from_numpy(image)
    image = torch.unsqueeze(image, 0)
    keypoints = torch.from_numpy(keypoints)
    # print(image.shape, keypoints.shape)

    assert_img(image)
    assert_points(keypoints)
    return image, keypoints


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
        root_dir: Path,
        transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(idxs, root_dir, transform)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        # TODO add augmentations with if random.random()<THRESHOLD

        img, points = super().__getitem__(idx)

        NOSE_INDEX = 53 - 1  # nose is 53rd keypoint, minus 1 for zero-index
        nose_point = points[NOSE_INDEX].reshape(1, 2)
        return img, nose_point


class XmlSample:
    def __init__(
        self, root_dir: Path, xml_file: Path, filename: ET.Element, hr=1.4, wr=1.2
    ):
        self.root = root_dir
        self.source = xml_file
        self.file = filename
        self.hr, self.wr = hr, wr

    def load_img(self):
        # load image from file
        img_name = self.root / self.file.attrib["file"]
        img = load_img(img_name)
        assert_img(img)
        return img

    def get_box(self, adjust=True):

        box = self.file[0].attrib
        left, top, width, height = (
            int(box["left"]),
            int(box["top"]),
            int(box["width"]),
            int(box["height"]),
        )

        # ratio for adjusting box
        row_shift = int(round(height * (self.hr - 1) / 2))
        col_shift = int(round(width * (self.wr - 1) / 2))

        # x, y for the top left corner of the box, w, h for box width and height
        if adjust:
            top -= row_shift
            left -= col_shift
            height = int(round(height * self.hr))
            width = int(round(width * self.wr))
        return top, left, height, width


class XmlTrainSample(XmlSample):
    def __init__(
        self, root_dir: Path, xml_file: Path, filename: ET.Element, hr:int, wr:int
    ):
        super().__init__(root_dir, xml_file, filename, hr, wr)

    def load_pts(self):
        # load keypoints from file
        keypts = []
        for num in range(68):
            x_coordinate = int(self.file[0][num].attrib["x"])
            y_coordinate = int(self.file[0][num].attrib["y"])
            keypts.append([x_coordinate, y_coordinate])
        keypts = torch.as_tensor(keypts, dtype=torch.float32)
        return keypts

    def get_train_sample(self):
        img = self.load_img()
        keypts = self.load_pts()

        top, left, height, width = self.get_box()
        img = TT.functional.crop(
            img,
            top=top,
            left=left,
            height=height,
            width=width,
        )

        # fix keypoints according to crop
        keypts[:, 0] -= left
        keypts[:, 1] -= top

        # make keypoints ratios
        h, w = img.shape[-2:]
        keypts[:, 0] /= w
        keypts[:, 1] /= h

        # resize to 224x224
        img = TT.Resize((224, 224))(img)

        assert_img(img)
        assert_points(keypts)
        return img, keypts

    def get_original_pts(self, pts: Tensor) -> Tensor:
        assert_points(pts)

        # revert ratios keypoints to actual coordinates
        img = self.load_img()
        h, w = img.shape[-2:]
        pts[:, 0] *= w
        pts[:, 1] *= h

        # fix keypoints according to crop
        top, left, height, width = self.get_box()
        pts[:, 0] += left
        pts[:, 1] += top

        assert_points(pts)
        return pts


class XmlTestSample(XmlSample):
    def __init__(
        self, root_dir: Path, xml_file: Path, filename: ET.Element, hr:int, wr:int
    ):
        super().__init__(root_dir, xml_file, filename, hr, wr)

    def get_original_pts(self, pts: Tensor) -> Tensor:
        assert_points(pts)

        # revert ratios keypoints to actual coordinates
        img = self.load_img()
        h, w = img.shape[-2:]
        pts[:, 0] *= w
        pts[:, 1] *= h

        # fix keypoints according to crop
        top, left, height, width = self.get_box()
        pts[:, 0] += left
        pts[:, 1] += top

        assert_points(pts)
        return pts


class LargeDataset(Dataset):  # loads xml files
    def __init__(
        self,
        data_dir: Path,
        xml_file: Path,
        transform: Optional[Callable] = None,
    ) -> None:

        self.data_dir = data_dir
        self.xml = xml_file

        tree = ET.parse(self.xml)
        all_files = tree.getroot()[2]

        # initialize all samples in dataset as XmlSample
        self.samples = []
        for f in all_files:
            sample = XmlSample(
                root_dir=data_dir, xml_file=self.xml, filename=f, hr=1, wr=1
            )
            self.samples.append(sample)

    def __len__(self):
        return len(self.samples)  # should be 6666

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        sample = self.samples[idx]
        return sample.get_train_sample()


class LargeTrainDataset(LargeDataset):  # loads xml files
    def __init__(
        self,
        data_dir: Path,
        xml_file: Path,
        transform: Optional[Callable] = None,
    ) -> None:

        self.data_dir = data_dir
        self.xml = xml_file

        tree = ET.parse(self.xml)
        all_files = tree.getroot()[2]

        # initialize all samples in dataset as XmlSample
        self.samples = []
        for f in all_files:
            sample = XmlTrainSample(
                root_dir=data_dir, xml_file=self.xml, filename=f, hr=1, wr=1
            )
            self.samples.append(sample)

    def __len__(self):
        return len(self.samples)  # should be 6666

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        sample = self.samples[idx]
        return sample.get_train_sample()


class LargeTestDataset(LargeDataset):  # loads xml files
    def __init__(
        self,
        data_dir: Path,
        xml_file: Path,
        transform: Optional[Callable] = None,
    ) -> None:

        self.data_dir = data_dir
        self.xml = xml_file

        tree = ET.parse(self.xml)
        all_files = tree.getroot()[2]

        # initialize all samples in dataset as XmlSample
        self.samples = []
        for f in all_files:
            sample = XmlTestSample(
                root_dir=data_dir, xml_file=self.xml, filename=f, hr=1, wr=1
            )
            self.samples.append(sample)

    def __len__(self):
        return len(self.samples)  # should be 6666

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        sample = self.samples[idx]
        # TODO check empty Tensor
        return sample.load_img(), torch.empty(0) # there are no keypoints in test set


def to_panda(filename, keypts: Tensor):
    return True


def save_kaggle(keypts: Tensor) -> bool:
    # TODO
    """
    Saves predicted keypoints of Part 3 test set as a csv file
    """

    return True
