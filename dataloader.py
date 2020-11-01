# This file is for loading images and keypoints customized for the Danes and ibug dataset.
# data source: http://www2.imm.dtu.dk/~aam/datasets/datasets.html.
import math
import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
import skimage.transform as ST
import torch
import torchvision
import torchvision.transforms as TT
from torch import Tensor
from torch.utils.data import Dataset

from my_types import assert_img, assert_points


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
            # TT.Grayscale(),
        ]
    )
    img = pipeline(t)
    assert_img(img)
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
        self, root_dir: Path, xml_file: Path, filename: ET.Element, hr: int, wr: int
    ):
        self.root = root_dir
        self.source = xml_file
        self.filename = filename
        self.hr, self.wr = hr, wr

    def load_img(self):
        # load image from file
        img_name = self.root / self.filename.attrib["file"]
        img = load_img(img_name)
        assert_img(img)
        return img

    def load_pts(self):
        return torch.empty(0)

    def get_box(self, adjust=True):

        box = self.filename[0].attrib
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

        # make sure there's no negative indices
        if top <= 0:
            height -= abs(top)
            top = 0
        if left <= 0:
            width -= abs(left)
            left = 0

        return top, left, height, width

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


class XmlTrainSample(XmlSample):
    def __init__(
        self, root_dir: Path, xml_file: Path, filename: ET.Element, hr: int, wr: int
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


class XmlTestSample(XmlSample):
    def __init__(
        self, root_dir: Path, xml_file: Path, filename: ET.Element, hr: int, wr: int
    ):
        super().__init__(root_dir, xml_file, filename, hr, wr)


class LargeDataset(Dataset):  # loads xml files
    def __init__(
        self,
        data_dir: Path,
        xml_file: Path,
        sample_class: type = XmlSample,
        transform: Optional[Callable] = None,
    ) -> None:

        self.data_dir = data_dir
        self.xml = xml_file
        self.sample_class = sample_class
        self.transform = transform

        tree = ET.parse(self.xml)
        all_files = tree.getroot()[2]

        # initialize all samples in dataset as XmlSample
        self.samples = []
        for f in all_files:
            sample = sample_class(
                root_dir=data_dir, xml_file=self.xml, filename=f, hr=1, wr=1
            )
            self.samples.append(sample)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:

        sample = self.samples[idx]
        filename = sample.filename
        img = sample.load_img()
        keypts = sample.load_pts()

        if self.transform is not None:
            img, keypts = self.transform(img, keypts)

        # assert_img(img)
        # assert_points(keypts)
        return img, keypts  # , filename


class LargeTrainDataset(LargeDataset):  # loads xml files
    def __init__(
        self,
        data_dir: Path,
        xml_file: Path,
        transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(data_dir, xml_file, XmlTrainSample, transform)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        sample = self.samples[idx]
        img = sample.load_img()
        keypts = sample.load_pts()

        # crop image
        top, left, height, width = sample.get_box()
        img = TT.functional.crop(
            img,
            top=top,
            left=left,
            height=height,
            width=width,
        )
        assert_img(img)

        # fix keypoints according to crop
        keypts[:, 0] -= left
        keypts[:, 1] -= top

        # make keypoints ratios
        h, w = img.shape[-2:]
        keypts[:, 0] /= w
        keypts[:, 1] /= h

        # resize to 224x224
        img = TT.Resize((224, 224))(img)

        if self.transform is not None:
            img, keypts = self.transform(img, keypts)

        assert_img(img)
        assert_points(keypts)

        return img, keypts


class LargeTestDataset(LargeDataset):  # loads xml files
    def __init__(
        self,
        data_dir: Path,
        xml_file: Path,
        transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(data_dir, xml_file, XmlTestSample, transform)

def get_id(filename: ET.Element):
    
def to_panda(filename: ET.Element, keypts: Tensor):
    return True


def save_kaggle(keypts1008: List) -> bool:
    """
    Saves predicted keypoints of Part 3 test set as a csv file

    keypts1008: List of 1008 tensors.
        Each tensor contains the 68 predicted keypoints of a test sample.
        Each tensor is of shape (68, 2).

    """
    # TODO

    assert len(keypts1008) == 1008
    assert all(assert_points(keypts) for keypts in keypts1008)
    all_pts = keypts1008
    all_pds = []
    for keypts in all_pts:
        all_pds.append(to_panda(keypts))
    return True
