# This file is for loading images and keypoints customized for the Danes and ibug dataset.
# danes data set: http://www2.imm.dtu.dk/~aam/datasets/datasets.html.
import math
import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
import numpy as np
import skimage.transform as ST
import skimage.transform as ST
import torch
import torchvision
import torchvision.transforms as TT
import torchvision.transforms as TT
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

from my_types import assert_img, assert_points

DATA_DIR = Path("data")
DANES_ROOT = DATA_DIR / Path("imm_face_db")
IBUG_ROOT = DATA_DIR / Path("ibug_300W_large_face_landmark_dataset")
train_xml = IBUG_ROOT / Path("labels_ibug_300W_train.xml")
test_xml = IBUG_ROOT / Path("labels_ibug_300W_test_parsed.xml")
assert DATA_DIR.exists()
assert DANES_ROOT.exists()
assert IBUG_ROOT.exists()


def load_img(img_file: Path):
    t = Image.open((img_file))
    pipeline = TT.Compose(
        [
            # TT.ToPILImage(),
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


# def load_nose(file: os.PathLike) -> Tensor:
#     points = load_asf(file)
#     NOSE_INDEX = 53 - 1  # nose is 53rd keypoint, minus 1 for zero-index
#     nose_point = points[NOSE_INDEX].reshape(1, 2)
#     assert_points(nose_point)
#     return nose_point


#
# Part 1
#


def part1_transform(image, keypoints) -> Tuple[Tensor, Tensor]:
    out_h, out_w = 240, 320
    image = TT.Resize((out_h, out_w))(image)

    assert_img(image)
    assert_points(keypoints)  # do nothing, should be ratios
    return image, keypoints


def part1_augment(image, keypoints) -> Tuple[Tensor, Tensor]:
    # do nothing
    return image, keypoints


class NoseKeypointTrainDataset(Dataset):
    def __init__(
        self,
    ) -> None:
        super().__init__()
        # Use all 6 images of the first 32 persons (index 1-32) as the training set
        # (total 32 x 6 = 192 images)
        idxs = torch.arange(1, 33)
        self.img_files = sorted(
            f for f in self.root_dir.glob("*.jpg") if int(f.name.split("-")[0]) in idxs
        )
        self.asf_files = sorted(
            f for f in self.root_dir.glob("*.asf") if int(f.name.split("-")[0]) in idxs
        )
        assert len(self.img_files) == len(
            self.asf_files
        ), f"{len(self.img_files) = },  {len(self.asf_files) = }"

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        # TODO add augmentations with if random.random()<THRESHOLD

        img_name, asf_name = self.img_files[idx], self.asf_files[idx]
        img = load_img(img_name)
        points = load_asf(asf_name)
        img, points = part1_transform(img, points)
        img, points = part1_augment(img, points)

        NOSE_INDEX = 53 - 1  # nose is 53rd keypoint, minus 1 for zero-index
        nose_point = points[NOSE_INDEX].reshape(1, 2)

        assert_img(img)
        assert_points(nose_point)
        return img, nose_point


class NoseKeypointValidDataset(Dataset):
    def __init__(
        self,
        idxs: Sequence[int],
        root_dir: Path,
    ) -> None:
        super().__init__()
        # Use all 6 images of the first 32 persons (index 1-32) as the training set
        # (total 32 x 6 = 192 images)
        idxs = torch.arange(1, 33)
        self.img_files = sorted(
            f for f in self.root_dir.glob("*.jpg") if int(f.name.split("-")[0]) in idxs
        )
        self.asf_files = sorted(
            f for f in self.root_dir.glob("*.asf") if int(f.name.split("-")[0]) in idxs
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

        NOSE_INDEX = 53 - 1  # nose is 53rd keypoint, minus 1 for zero-index
        nose_point = points[NOSE_INDEX].reshape(1, 2)

        assert_img(img)
        assert_points(nose_point)
        return img, nose_point


class NoseKeypointTestDataset(Dataset):  # not used
    def __init__(
        self,
    ) -> None:

        super().__init__()
        # Use images of the remaining 8 persons (index 33-40) as the validation set
        # (total 8 * 6 = 48 images)
        idxs = torch.arange(32, 40)
        self.img_files = sorted(
            f for f in self.root_dir.glob("*.jpg") if int(f.name.split("-")[0]) in idxs
        )
        self.asf_files = sorted(
            f for f in self.root_dir.glob("*.asf") if int(f.name.split("-")[0]) in idxs
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

        NOSE_INDEX = 53 - 1  # nose is 53rd keypoint, minus 1 for zero-index
        nose_point = points[NOSE_INDEX].reshape(1, 2)

        assert_img(img)
        assert_points(nose_point)
        return img, nose_point


#
# Part 2
#


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


def part2_transform(image, keypoints) -> Tuple[Tensor, Tensor]:
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

    assert_img(image)
    assert_points(keypoints)
    return image, keypoints


class FaceKeypointsTrainDataset(Dataset):
    def __init__(self) -> None:
        super().__init__()
        # Use all 6 images of the first 32 persons (index 1-32) as the training set
        # (total 32 x 6 = 192 images)
        idxs = torch.arange(1, 33)
        self.img_files = sorted(
            f for f in self.root_dir.glob("*.jpg") if int(f.name.split("-")[0]) in idxs
        )
        self.asf_files = sorted(
            f for f in self.root_dir.glob("*.asf") if int(f.name.split("-")[0]) in idxs
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
        idxs = torch.arange(1, 33)
        self.img_files = sorted(
            f for f in self.root_dir.glob("*.jpg") if int(f.name.split("-")[0]) in idxs
        )
        self.asf_files = sorted(
            f for f in self.root_dir.glob("*.asf") if int(f.name.split("-")[0]) in idxs
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


class FaceKeypointsTestDataset(Dataset):
    def __init__(
        self,
    ) -> None:

        super().__init__()
        # Use images of the remaining 8 persons (index 33-40) as the validation set
        # (total 8 * 6 = 48 images)
        idxs = torch.arange(32, 40)
        self.img_files = sorted(
            f for f in self.root_dir.glob("*.jpg") if int(f.name.split("-")[0]) in idxs
        )
        self.asf_files = sorted(
            f for f in self.root_dir.glob("*.asf") if int(f.name.split("-")[0]) in idxs
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
        return img


#
# Part 3
#


def part3_augment(img, keypts):
    # do nothing
    return img, keypts


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
        raise ValueError("This function should not be called")

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
            x_coordinate = int(self.filename[0][num].attrib["x"])
            y_coordinate = int(self.filename[0][num].attrib["y"])
            keypts.append([x_coordinate, y_coordinate])
        keypts = torch.as_tensor(keypts, dtype=torch.float32)
        return keypts


class XmlValidSample(XmlSample):  # works exactly the same as XmlTestSample
    def __init__(
        self, root_dir: Path, xml_file: Path, filename: ET.Element, hr: int, wr: int
    ):
        super().__init__(root_dir, xml_file, filename, hr, wr)

    def load_pts(self):
        # load keypoints from file
        keypts = []
        for num in range(68):
            x_coordinate = int(self.filename[0][num].attrib["x"])
            y_coordinate = int(self.filename[0][num].attrib["y"])
            keypts.append([x_coordinate, y_coordinate])
        keypts = torch.as_tensor(keypts, dtype=torch.float32)
        return keypts


class XmlTestSample(XmlSample):
    def __init__(
        self, root_dir: Path, xml_file: Path, filename: ET.Element, hr: int, wr: int
    ):
        super().__init__(root_dir, xml_file, filename, hr, wr)


class LargeTrainDataset(Dataset):
    def __init__(self) -> None:
        super().__init__()
        self.root = IBUG_ROOT
        self.xml = train_xml

        tree = ET.parse(self.xml)
        all_files = tree.getroot()[2]

        # initialize all samples in dataset as XmlSample
        self.samples = []
        for f in all_files:
            sample = XmlTrainSample(
                root_dir=self.root, xml_file=self.xml, filename=f, hr=1.4, wr=1.2
            )
            self.samples.append(sample)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
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

        if self.augment is not None:
            img, keypts = self.augment(img, keypts)

        assert_img(img)
        assert_points(keypts)
        return img, keypts


class LargeValidDataset(LargeDataset):  # works the same as training set
    def __init__(self) -> None:
        super().__init__()
        self.root = IBUG_ROOT
        self.xml = train_xml

        tree = ET.parse(self.xml)
        all_files = tree.getroot()[2]

        # initialize all samples in dataset as XmlSample
        self.samples = []
        for f in all_files:
            sample = XmlValidSample(
                root_dir=self.root, xml_file=self.xml, filename=f, hr=1.4, wr=1.2
            )
            self.samples.append(sample)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
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

        if self.augment is not None:
            img, keypts = self.augment(img, keypts)

        assert_img(img)
        assert_points(keypts)
        return img, keypts


class LargeTestDataset(LargeDataset):  # works the same as training set
    def __init__(self) -> None:
        super().__init__()
        self.root = IBUG_ROOT
        self.xml = test_xml

        tree = ET.parse(self.xml)
        all_files = tree.getroot()[2]

        # initialize all samples in dataset as XmlSample
        self.samples = []
        for f in all_files:
            sample = XmlTestSample(
                root_dir=self.root, xml_file=self.xml, filename=f, hr=1.4, wr=1.2
            )
            self.samples.append(sample)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        img = sample.load_img()

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

        # resize to 224x224
        img = TT.Resize((224, 224))(img)

        assert_img(img)
        return img


# def get_id(filename: ET.Element):
#     img_name = filename.attrib["file"]
#     return img_name

# def to_panda(filename: ET.Element, keypts: Tensor):
#     return True


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
    for i in range(1008):
        id = i
        pred_keypts = all_pts[i]

    return True
