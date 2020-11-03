import os
import pickle
from pathlib import Path
from typing import Sequence, Union

import numpy as np
import skimage.io as io
from skimage.util import img_as_float
from torch import Tensor

from constants import *


def assert_points(pts, ratio=True) -> bool:
    if isinstance(pts, np.ndarray):
        assert pts.shape[1] == 2
        assert (pts >= 0).all()

    elif isinstance(pts, Tensor):
        assert pts.ndim == 2, pts.shape
        assert pts.shape[1] == 2, pts.shape

    if ratio:
        # make sure that the keypoints are ratios
        rows = pts[:, 0]
        cols = pts[:, 1]
        # leave some wiggle room so use 1.5 instead of 1
        assert (
            rows.max() <= 1.5 and cols.max() <= 1.5
        ), f"points are not ratios {rows.max()}, {cols.max()}"
        return True

    else:
        raise ValueError(f"points of type {type(pts)} not expected.")


def assert_img(img):
    assert isinstance(img, Tensor), type(img)
    assert img.ndim == 3, img.shape
    # assert list(img.shape)[0] == 1, f"{img.shape} is not grayscale"
    assert all(x > 0 for x in list(img.shape)), img.shape
    return True


ToImgArray = Union[os.PathLike, np.ndarray]
ZeroOneFloatArray = np.ndarray
UbyteArray = np.ndarray

ToPoints = Union[os.PathLike, np.ndarray]
Triangle = np.ndarray


def to_img_arr(x: ToImgArray) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return img_as_float(x).clip(0, 1)
    elif isinstance(x, torch.Tensor):
        if x.ndim == 4:  # batch
            # nchw to nhwc
            return x.cpu().detach().permute(0, 2, 3, 1).numpy()
        elif x.ndim == 3:  # single image
            return x.cpu().detach().permute(1, 2, 0).numpy()
    elif isinstance(x, (str, Path, os.PathLike)):
        x = Path(x)
        if x.suffix in (".jpeg", ".jpg"):
            img = io.imread(x)
            img = img_as_float(img)
            assert_img_type(img)
            return img
        else:
            raise ValueError(f"Didn't expect type {type(x)}")


def to_points(x: ToPoints) -> np.ndarray:
    if isinstance(x, np.ndarray):
        assert_points(x)
        return x
    elif isinstance(x, torch.Tensor):
        return x.cpu().detach().numpy()
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
    assert img.dtype == "float64", img.dtype
    assert np.max(img) <= 1.0 and np.min(img) >= 0.0, (np.min(img), np.max(img))
    assert np.ndim(img) == 3
    return True


def assert_is_triangle(triangle: np.ndarray) -> bool:
    """ Check image data type """
    assert triangle.shape == (3, 2), triangle.shape
    assert (triangle >= 0).all(), triangle
    return True


def assert_indices(indices: np.ndarray) -> bool:
    assert isinstance(indices, np.ndarray)
    assert indices.dtype == "int"
    assert (indices >= 0).all()
    assert indices.shape[1] == 2
    return True
