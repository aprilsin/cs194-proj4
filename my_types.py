import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import numpy as np
import skimage.io as io
from skimage.util import img_as_float

from constants import Array, List, img

#
# IMAGE ARRAYS
#
ZeroOneFloatArray = np.ndarray


def assert_zero_one_img(img: np.ndarray):
    assert isinstance(img, np.ndarray), f"expect ndarray but got {type(img)}"
    assert img.dtype == np.float64, img.dtype
    assert img.max() <= 1.0 and img.min() >= 0.0, (img.min(), img.max())
    return True


UbyteArray = np.ndarray


def assert_ubyte_img(img: np.ndarray):
    assert isinstance(img, np.ndarray), f"expect ndarray but got {type(img)}"
    assert img.dtype == int, img.dtype
    assert img.max() <= 255 and img.min() >= 0, (img.min(), img.max())
    return True


def assert_img_type(img: np.ndarray) -> bool:
    """ Check image data type """
    assert_zero_one_img(img)
    assert img.ndim == 2 or img.ndim == 3, img.ndim
    return True


ToImgArray = Union[os.PathLike, np.ndarray]


def to_img_arr(x: ToImgArray, as_gray=False) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return img_as_float(x).clip(0, 1)
    elif isinstance(x, (str, Path, os.PathLike)):
        x = Path(x)
        if x.suffix in (".jpeg", ".jpg"):
            img = io.imread(x, as_gray=as_gray)
            img = img_as_float(img)
            assert_img_type(img)
            return img
        else:
            raise ValueError(f"Didn't expect type {type(x)}")


#
# PIXEL VALUES / POINTS
#
def assert_points_part1(x: np.ndarray) -> bool:
    assert isinstance(x, np.ndarray)
    assert x.shape[1] == 2
    # range is normalized to within -0.5 and 0.5
    assert x.dtype == np.float64, img.dtype
    assert x.max() <= 0.5 and x.min() >= -0.5, (x.min(), img.max())
    return True


def assert_points(points: np.ndarray) -> bool:
    assert isinstance(points, np.ndarray)
    assert points.shape[1] == 2
    assert (points >= 0).all()
    return True


def assert_is_point(point) -> bool:
    assert isinstance(point, Union[List, Array, np.ndarray])
    assert point.shape == (2,)
    assert (point >= 0).all()
    return True


ToPoints = Union[os.PathLike, np.ndarray]


def to_points(x: ToPoints) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    elif isinstance(x, (str, Path, os.PathLike)):
        x = Path(x)
        if x.suffix in (".pkl", ".p"):
            points = pickle.load(open(x, "rb"))
            assert_points(points)
            return points
        elif x.suffix == ".asf":
            asf = open(x, "r")
            lines_read = asf.readlines()
            num_pts = int(lines_read[9])
            lines = []
            for i in range(16, num_pts + 16):
                lines.append(lines_read[i])

            points = []
            for line in lines:
                data = line.split(" \t")
                c = float(data[2])  # x coordinates = cols
                r = float(data[3])  # y coordinates = rows
                points.append((r, c))
            points = np.array(points)
            assert_points(points)
            return points
        else:
            raise ValueError(f"Didn't expect type {type(x)}")


#
# triangle
#
Triangle = np.ndarray


def assert_is_triangle(triangle: np.ndarray) -> bool:
    """ Check image data type """
    assert triangle.shape == (3, 2), triangle.shape
    assert (triangle >= 0).all(), triangle.nonzero()
    return True


#
# Indices
#
def assert_indices(indices: np.ndarray) -> bool:
    assert isinstance(indices, np.ndarray)
    #     assert indices.dtype == object, indices.dtype # should be of type Index
    assert (indices >= 0).all(), indices.nonzero()
    assert indices.shape[1] == 2
    return True


# ToIndex = Union[np.ndarray, List, Array, Tuple]
# def to_index(coord: ToIndex, (h, w)):
#     if not isinstance(coord, ToIndex):
#         raise ValueError(f"Didn't expect type {type(x)}")
#     coord.dtype == int:
#         return coord
#     else:
#         return np.int(np.round(num))
#     else:
@dataclass
class Index:
    row: int
    col: int


def index_to_arr(idx: Index) -> np.ndarray:
    assert isinstance(idx, Index)
    return np.ndarray


#
# Dataset
#
class Data:
    def __init__(self, img: np.ndarray, indices: np.ndarray):
        ## FIXME: make this work for tensors
        assert_img_type(img)
        assert_indices(indices)
        for idx in indices:
            self.assert_valid_index(img, idx)
        self.img = img
        self.indices = indices  # a list of keypoints

    def assert_valid_index(self, img: np.ndarray, idx: np.ndarray):
        h, w = img.shape
        assert (
            idx.row >= 0 and idx.row < h
        ), f"invalid row index {idx.row} for image of shape {img.shape}"
        assert (
            idx.col >= 0 and idx.col < w
        ), f"invalid col index {idx.row} for image of shape {img.shape}"
        return True


# def unnormalize(points, base):
#     assert_
#     assert num.dtype == float, num.dtype
#     return num * base
