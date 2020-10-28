import os
import pickle
from pathlib import Path
from typing import Sequence, Union
from dataclasses import dataclass

import numpy as np
import skimage.io as io
from skimage.util import img_as_float

from constants import *

ToImgArray = Union[os.PathLike, np.ndarray]
ZeroOneFloatArray = np.ndarray
UbyteArray = np.ndarray

ToPoints = Union[os.PathLike, np.ndarray]
Triangle = np.ndarray

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


def assert_img_type(img: np.ndarray) -> bool:
    """ Check image data type """
    assert isinstance(img, np.ndarray), f"expect ndarray but got {type(img)}"
    assert img.dtype == np.float64, img.dtype
    assert img.max() <= 1.0 and img.min() >= 0.0, (img.min(), img.max())
    assert img.ndim == 2 or img.ndim == 3, img.ndim
    return True


def assert_is_triangle(triangle: np.ndarray) -> bool:
    """ Check image data type """
    assert triangle.shape == (3, 2), triangle.shape
    assert (triangle >= 0).all(), triangle.nonzero()
    return True


def assert_indices(indices: np.ndarray) -> bool:
    assert isinstance(indices, np.ndarray)
    assert indices.dtype == Index, indices.dtype
    assert (indices >= 0).all(), indices.nonzero()
    assert indices.shape[1] == 2
    return True


def assert_points(points: np.ndarray) -> bool:
    assert isinstance(points, np.ndarray)
    assert points.shape[1] == 2
    assert (points >= 0).all()
    return True

def assert_is_point(point: np.ndarray) -> bool:
    assert isinstance(point, np.ndarray)
    assert point.shape == (2,)
    assert (point >= 0).all()
    return True

class Data:
    def __init__(self, img:np.ndarray, indices:np.ndarray):
        assert_img_type(img)
        assert_indices(indices)
        self.img = img
        self.indices = indices
        
@dataclass
class Index:
    row : int
    col : int

def valid_index(img:np.ndarray, idx:Index):
    pass

def to_int(num):
    if num.dtype == int:
        return num
    else:
        return np.int(np.round(num))

# def unnormalize(points, base):
#     assert_
#     assert num.dtype == float, num.dtype
#     return num * base