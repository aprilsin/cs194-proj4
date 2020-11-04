# This file is for loading images and keypoints customized for the Danes and ibug dataset.
# danes data set: http://www2.imm.dtu.dk/~aam/datasets/datasets.html.
import math
import os
import time
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import skimage as sk
import skimage.io as skio
import skimage.transform as ST
import torch
import torchvision.transforms as TT
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

from my_types import assert_img, assert_img_type, assert_points, to_img_arr

DATA_DIR = Path("data")
OUT_DIR = Path("output")

DANES_ROOT = DATA_DIR / "imm_face_db"
IBUG_ROOT = DATA_DIR / "ibug_300W_large_face_landmark_dataset"
train_xml = IBUG_ROOT / "labels_ibug_300W_train.xml"
test_xml = IBUG_ROOT / "labels_ibug_300W_test_parsed.xml"

MY_DIR = DATA_DIR / "my_collection"
my_test_xml = MY_DIR / "my_samples.xml"

ME_DIR = DATA_DIR / "me"
me_xml = ME_DIR / "me.xml"

OUT_DIR.mkdir(exist_ok=True)
assert DATA_DIR.exists()
assert DANES_ROOT.exists()
assert IBUG_ROOT.exists()
assert MY_DIR.exists()


def load_img(img_file: Path) -> Tensor:
    t = Image.open((img_file))
    pipeline = TT.Compose(
        [
            TT.ToTensor(),
        ]
    )
    img = pipeline(t)
    if img.shape[0] == 3:
        img = TT.Grayscale()(img)

    assert_img(img)
    return img


def load_asf(file: os.PathLike) -> Tensor:  # for part 1 and 2
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

#
# Part 1
#


def part1_transform(image, keypoints) -> Tuple[Tensor, Tensor]:
    # hardcoded
    out_h, out_w = 240, 320
    image = TT.Resize((out_h, out_w))(image)

    assert_img(image)
    assert_points(keypoints)  # do nothing, should be ratios
    return image, keypoints


def part1_augment(image, keypoints) -> Tuple[Tensor, Tensor]:

    return image, keypoints  # do nothing


class NoseKeypointTrainDataset(Dataset):
    def __init__(self) -> None:
        super().__init__()
        # Use all 6 images of the first 32 persons (index 1-32) as the training set
        # (total 32 x 6 = 192 images)
        root_dir = DANES_ROOT
        idxs = torch.arange(1, 33)
        self.img_files = sorted(
            f for f in root_dir.glob("*.jpg") if int(f.name.split("-")[0]) in idxs
        )
        self.asf_files = sorted(
            f for f in root_dir.glob("*.asf") if int(f.name.split("-")[0]) in idxs
        )
        assert len(self.img_files) == len(
            self.asf_files
        ), f"{len(self.img_files) = },  {len(self.asf_files) = }"

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:

        img_name, asf_name = self.img_files[idx], self.asf_files[idx]
        img = load_img(img_name)
        points = load_asf(asf_name)

        img, points = part1_transform(img, points)
        # TODO add augmentations with if random.random()<THRESHOLD
        img, points = part1_augment(img, points)

        NOSE_INDEX = 53 - 1  # nose is 53rd keypoint, minus 1 for zero-index
        nose_point = points[NOSE_INDEX].reshape(1, 2)

        assert_img(img)
        assert_points(nose_point)
        return img, nose_point


class NoseKeypointValidDataset(Dataset):
    def __init__(self) -> None:
        super().__init__()
        # Use all 6 images of the first 32 persons (index 1-32) as the training set
        # (total 32 x 6 = 192 images)
        idxs = torch.arange(33, 40 + 1)
        root_dir = DANES_ROOT
        self.img_files = sorted(
            f for f in root_dir.glob("*.jpg") if int(f.name.split("-")[0]) in idxs
        )
        self.asf_files = sorted(
            f for f in root_dir.glob("*.asf") if int(f.name.split("-")[0]) in idxs
        )
        assert len(self.img_files) == len(
            self.asf_files
        ), f"{len(self.img_files) = },  {len(self.asf_files) = }"

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        img_name, asf_name = self.img_files[idx], self.asf_files[idx]
        img = load_img(img_name)
        points = load_asf(asf_name)
        # no augmentation since we're not training
        img, points = part1_transform(img, points)

        NOSE_INDEX = 53 - 1  # nose is 53rd keypoint, minus 1 for zero-index
        nose_point = points[NOSE_INDEX].reshape(1, 2)

        assert_img(img)
        assert_points(nose_point)
        return img, nose_point


#
# Part 2
#


def rotate(point, origin, angle):
    """Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in degrees.
    """
    angle = np.radians(angle)
    ox, oy = origin
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return qx, qy


def part2_transform(image, keypoints) -> Tuple[Tensor, Tensor]:
    # hardcoded
    out_h, out_w = 240, 320
    image = TT.Resize((out_h, out_w))(image)

    assert_img(image)
    assert_points(keypoints)  # do nothing, should be ratios
    return image, keypoints


def part2_augment(image, keypoints) -> Tuple[Tensor, Tensor]:

    # convert tensors to numpy arrays to use skimage
    image, keypoints = image.squeeze().numpy(), keypoints.numpy()
    h, w = image.shape[-2:]

    rotate_deg = np.random.randint(-12, 12)
    image = ST.rotate(image, angle=rotate_deg, center=(h / 2, w / 2))
    for i, point in enumerate(keypoints):
        x, y = point[0] * w, point[1] * h
        qx, qy = rotate(point=(x, y), origin=(h / 2, w / 2), angle=-rotate_deg)
        keypoints[i][0] = qx / w
        keypoints[i][1] = qy / h

    # convert back to tensors
    image = torch.from_numpy(image)
    image = image.unsqueeze(0)
    keypoints = torch.from_numpy(keypoints)

    assert_img(image)
    assert_points(keypoints)
    return image, keypoints


class FaceKeypointsTrainDataset(Dataset):
    def __init__(self) -> None:
        super().__init__()
        # Use all 6 images of the first 32 persons (index 1-32) as the training set
        # (total 32 x 6 = 192 images)
        idxs = torch.arange(1, 33)
        root_dir = DANES_ROOT
        self.img_files = sorted(
            f for f in root_dir.glob("*.jpg") if int(f.name.split("-")[0]) in idxs
        )
        self.asf_files = sorted(
            f for f in root_dir.glob("*.asf") if int(f.name.split("-")[0]) in idxs
        )
        assert len(self.img_files) == len(
            self.asf_files
        ), f"{len(self.img_files) = },  {len(self.asf_files) = }"

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        img_name, asf_name = self.img_files[idx], self.asf_files[idx]
        img = load_img(img_name)
        points = load_asf(asf_name)

        img, points = part2_transform(img, points)
        # TODO add augmentations with if random.random()<THRESHOLD
        img, points = part2_augment(img, points)

        assert_img(img)
        assert_points(points)
        return img, points


class FaceKeypointsValidDataset(Dataset):  # works the same as training set
    def __init__(self) -> None:
        super().__init__()
        # Use all 6 images of the first 32 persons (index 1-32) as the training set
        # (total 32 x 6 = 192 images)
        idxs = torch.arange(33, 40 + 1)
        root_dir = DANES_ROOT
        self.img_files = sorted(
            f for f in root_dir.glob("*.jpg") if int(f.name.split("-")[0]) in idxs
        )
        self.asf_files = sorted(
            f for f in root_dir.glob("*.asf") if int(f.name.split("-")[0]) in idxs
        )
        assert len(self.img_files) == len(
            self.asf_files
        ), f"{len(self.img_files) = },  {len(self.asf_files) = }"

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        img_name, asf_name = self.img_files[idx], self.asf_files[idx]
        img = load_img(img_name)
        points = load_asf(asf_name)

        img, points = part2_transform(img, points)

        assert_img(img)
        assert_points(points)
        return img, points


#
# Part 3
#


def part3_augment(img, keypts):
    # do nothing
    return img, keypts


def part3_transform(img: Tensor) -> Tensor:
    # resnet expects input of size 224 x 224
    return TT.Resize((224, 224))(img)


class XmlSample:
    def __init__(
        self, xml_file: Path, filename: ET.Element, height_ratio: int, width_ratio: int
    ):
        self.root = IBUG_ROOT
        self.source = xml_file
        self.filename = filename
        self.hr, self.wr = height_ratio, width_ratio

    def get_name(self):
        return self.filename.attrib["file"]

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

    def load_img(self):
        # load image from file
        img_name = self.root / self.filename.attrib["file"]
        img = load_img(img_name)

        # crop image
        top, left, height, width = self.get_box()
        img = TT.functional.crop(
            img,
            top=top,
            left=left,
            height=height,
            width=width,
        )

        assert_img(img)
        return img

    def load_pts(self):
        # load keypoints from file
        keypts = []
        for num in range(68):
            x_coordinate = int(self.filename[0][num].attrib["x"])
            y_coordinate = int(self.filename[0][num].attrib["y"])
            keypts.append([x_coordinate, y_coordinate])
        keypts = torch.as_tensor(keypts, dtype=torch.float32)

        # fix keypoints according to crop
        top, left, _, _ = self.get_box()
        keypts[:, 0] -= left
        keypts[:, 1] -= top

        # make keypoints ratios
        img = self.get_original_img()
        h, w = img.shape[-2:]
        keypts[:, 0] /= w
        keypts[:, 1] /= h
        return keypts

    def get_original_img(self):
        img_name = self.root / self.filename.attrib["file"]
        img = skio.imread(img_name)
        img = sk.img_as_float(img)
        if img.ndim == 2:
           return np.dstack((img, img, img))
        assert_img_type(img)
        return img

    def get_original_pts(self, pts: Tensor) -> Tensor:
        assert_points(pts, ratio=False)
        pts = pts.cpu().detach()
        # revert ratios keypoints to actual coordinates
        img = self.get_original_img()
        h, w = img.shape[-2:]
        pts[:, 0] *= w
        pts[:, 1] *= h

        # fix keypoints according to crop
        top, left, _, _ = self.get_box()
        pts[:, 0] += left
        pts[:, 1] += top
        return pts


class XmlTrainSample(XmlSample):
    def __init__(self, filename: ET.Element, hr: float, wr: float):
        super().__init__(train_xml, filename, hr, wr)


class XmlValidSample(XmlSample):
    def __init__(self, filename: ET.Element, hr: float, wr: float):
        super().__init__(train_xml, filename, hr, wr)


class XmlTestSample(XmlSample):
    def __init__(self, filename: ET.Element, hr: float, wr: float):
        super().__init__(test_xml, filename, hr, wr)

    def load_pts(self):
        raise ValueError("Test Set has no keypoints.")


class LargeTrainDataset(Dataset):
    def __init__(self) -> None:
        super().__init__()

        tree = ET.parse(train_xml)

        all_files = tree.getroot()[2]
        train_files = all_files[:6_000]

        assert len(all_files) == 6_666, len(all_files)
        assert len(train_files) == 6_000, len(train_files)

        # initialize all samples in dataset as XmlSample
        self.samples = [XmlTrainSample(filename=f, hr=1.4, wr=1.2) for f in train_files]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        img = sample.load_img()
        keypts = sample.load_pts()

        img = part3_transform(img)

        # TODO may want to augment randomly
        img, keypts = part3_augment(img, keypts)

        assert_img(img)
        assert_points(keypts)
        return img, keypts

    def get_original_img(self, idx: int):
        sample = self.samples[idx]
        return sample.get_original_img()

    def get_original_pts(self, idx: int):
        sample = self.samples[idx]
        return sample.get_original_pts()


class LargeValidDataset(Dataset):
    def __init__(self) -> None:
        super().__init__()

        tree = ET.parse(train_xml)
        all_files = tree.getroot()[2]
        assert len(all_files) == 6_666, len(all_files)
        valid_files = all_files[6_000:]
        assert len(valid_files) == 666, len(valid_files)

        self.samples = [XmlValidSample(filename=f, hr=1.4, wr=1.2) for f in valid_files]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        img = sample.load_img()
        keypts = sample.load_pts()

        img = part3_transform(img)
        # no augmentation for validation set

        assert_img(img)
        assert_points(keypts)
        return img, keypts

    def get_original_img(self, idx: int):
        sample = self.samples[idx]
        return sample.get_original_img()

    def get_original_pts(self, idx: int):
        sample = self.samples[idx]
        return sample.get_original_pts()


class LargeTestDataset(Dataset):  # works the same as training set
    def __init__(self) -> None:
        super().__init__()

        tree = ET.parse(test_xml)
        test_files = tree.getroot()[2]
        assert len(test_files) == 1_008, len(test_files)

        self.samples = [XmlTestSample(filename=f, hr=1.4, wr=1.2) for f in test_files]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        img = sample.load_img()
        # test set has no keypoints

        img = part3_transform(img)
        # no augmentation for test set

        assert_img(img)
        return img

    def get_original_img(self, idx: int):
        sample = self.samples[idx]
        return sample.get_original_img()


def save_kaggle(keypts1008: List) -> None:
    """Saves predicted keypoints of Part 3 test set as a csv file.

    keypts1008: List of 1008 tensors.
        Each tensor contains the 68 predicted keypoints of a test sample.
        Each tensor is of shape (68, 2).
    """
    N = 1_008

    assert len(keypts1008) == N, len(keypts1008)
    keypts1008 = torch.stack(keypts1008).cpu().numpy()

    df = pd.DataFrame(keypts1008.reshape(-1, 1), columns=["Predicted"])
    df.index.name = "Id"
    df.to_csv(OUT_DIR / f"{time.time():.0f}.csv")


#
# My Collection
#


class MyXmlTestSample(XmlSample):
    def __init__(self, filename: ET.Element, hr: float, wr: float):
        super().__init__(my_test_xml, filename, hr, wr)
        self.root = MY_DIR


class MyTestSet(Dataset):
    def __init__(self) -> None:
        super().__init__()

        tree = ET.parse(my_test_xml)
        test_files = tree.getroot()[1]
        assert len(test_files) == 4, len(test_files)

        self.samples = [MyXmlTestSample(filename=f, hr=1.0, wr=1.0) for f in test_files]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        img = sample.load_img()
        # test set has no keypoints

        img = part3_transform(img)
        # no augmentation for test set

        assert_img(img)
        return img

    def get_original_img(self, idx: int):
        sample = self.samples[idx]
        return sample.get_original_img()


#
# Me Growing Up - Morph Sequence
#


class MeXmlSample(XmlSample):
    def __init__(self, filename: ET.Element, hr: float, wr: float):
        super().__init__(me_xml, filename, hr, wr)
        self.root = ME_DIR

    def get_crop_box(self):
        crop_box = self.filename[1].attrib
        left, top, width, height = (
            int(crop_box["left"]),
            int(crop_box["top"]),
            int(crop_box["width"]),
            int(crop_box["height"]),
        )
        return top, left, height, width

    def get_cropped_img(self) -> np.ndarray:

        img_name = self.root / self.filename.attrib["file"]
        img = to_img_arr(img_name)

        top, left, height, width = self.get_crop_box()
        cropped = img[top : top + height, left : left + width, :]

        assert_img_type(cropped)  # returns colored image
        return cropped

    def get_cropped_pts(self, pts):
        pts = self.get_original_pts(pts)
        assert_points(pts, ratio=False)
        top, left, _, _ = self.get_crop_box()

        # TODO should it be switched?
        pts[:, 0] += left
        pts[:, 1] += top
        assert_points(pts, ratio=False)
        return pts
    
    def get_original_pts(self, pts):
        # TODO
        pass
        

class MePicsSet(Dataset):
    def __init__(self) -> None:
        super().__init__()

        tree = ET.parse(me_xml)
        test_files = tree.getroot()[1]
        assert len(test_files) == 16, len(test_files)

        self.samples = [MeXmlSample(filename=f, hr=1.0, wr=1.0) for f in test_files]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        img = sample.load_img()
        img = part3_transform(img)
        return img
    
    def get_original_img(self, idx:int):
        sample = self.samples[idx]
        return sample.get_original_img()
    
    def get_cropped_img(self, idx:int):
        sample = self.samples[idx]
        return sample.get_cropped_img()
    
    def get_original_pts(self, idx:int, pts):
        sample = self.samples[idx]
        return sample.get_original_pts(pts)
    
    def get_cropped_pts(self, idx:int, pts):
        sample = self.samples[idx]
        return sample.get_cropped_pts(pts)

    def get_morph_img(self, idx: int):
        sample = self.samples[idx]
        cropped = sample.get_cropped_img()
        resized = ST.resize(cropped, (500, 500))
        return resized

    def get_morph_pts(self, idx: int, keypts):
        sample = self.samples[idx]
        pts = sample.get_cropped_pts(keypts)

        # turn into ratios
        pts[:, 0] /= 500
        pts[:, 1] /= 500

        assert_points(pts, ratio=True)
        return pts
