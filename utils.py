import argparse
import math
import os
import pickle
import re
import sys
from pathlib import Path
from typing import Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import skimage as sk
import skimage.io as io
from scipy.interpolate import interp2d
from scipy.spatial import Delaunay
from skimage import transform
from skimage.util import img_as_float, img_as_ubyte

from constants import *
from my_types import *

#######################
#      FIX INDEX      #
#######################


def ifloor(x: np.ndarray) -> np.ndarray:
    """Avoid rounding up by taking floor before int so you can't index out of bounds."""
    return np.int64(np.floor(x))


#######################
#   INPUT AND OUPUT   #
#######################


def pick_points(img: ToImgArray, num_pts: int, APPEND_CORNERS=True) -> np.ndarray:
    """
    Returns an array of points for one image with ginput
    """
    img = to_img_arr(img)
    print(f"Please select {num_pts} points in image.")
    plt.imshow(img)
    points = plt.ginput(num_pts, timeout=0)  # never timeout
    plt.close()

    if APPEND_CORNERS:
        y, x, _ = img.shape
        points.extend(
            [
                (0, 0),
                (0, y - 1),
                (x - 1, 0),
                (x - 1, y - 1),
            ]
        )
    print(f"Picked {num_pts} points successfully.")
    return np.array(points)


# def add_corners(img, points:np.ndarray):
#     points.extend(
#         [
#             (0, 0),
#             (0, img.shape[1] - 1),
#             (img.shape[0] - 1, 0),
#             (img.shape[0] - 1, img.shape[1] - 1),
#         ]
#     )


def save_points(points: np.ndarray, name: os.PathLike) -> None:
    """
    Saves points as Pickle
    """
    name = Path(name)
    pickle_name = name.with_suffix(".pkl")
    pickle.dump(points, open(pickle_name, "wb"))


def load_points(name: os.PathLike) -> np.ndarray:
    """
    Loads an array of points saved as Pickle
    """
    name = Path(name)
    pickle_name = name.with_suffix(".pkl")
    return pickle.load(open(pickle_name, "rb"))


def load_points_from_asf(file_name, APPEND_CORNERS=False) -> np.ndarray:
    asf = open(file_name, "r")
    lines_read = asf.readlines()
    num_pts = int(lines_read[9])
    lines = []
    for i in range(16, num_pts + 16):
        lines.append(lines_read[i])

    points = []
    for line in lines:
        data = line.split(" \t")
        points.append((float(data[2]), float(data[3])))
    points = np.array(points)
    points[:, 0] *= DANES_WIDTH
    points[:, 1] *= DANES_HEIGHT
    if APPEND_CORNERS:
        y, x = DANES_HEIGHT - 1, DANES_WIDTH - 1
        corners = np.array(((0.0, 0.0), (x, 0.0), (0.0, y), (x, y)))
        points = np.vstack((points, corners))
    return points


#######################
#      Alignment      #
#######################„


def find_centers(p1, p2) -> Tuple[int, int]:
    cr = int(np.round(np.mean([p1[0], p2[0]])))
    cc = int(np.round(np.mean([p1[1], p2[1]])))
    return cr, cc


def recenter(im, r, c):
    R, C, _ = im.shape
    rpad = (int)(np.abs(2 * r + 1 - R))
    cpad = (int)(np.abs(2 * c + 1 - C))
    return np.pad(
        im,
        [
            (0 if r > (R - 1) / 2 else rpad, 0 if r < (R - 1) / 2 else rpad),
            (0 if c > (C - 1) / 2 else cpad, 0 if c < (C - 1) / 2 else cpad),
            (0, 0),
        ],
        "constant",
    )


def align_img(
    img: ToImgArray,
    points: Optional[ToImgArray] = None,
    left_idx=0,
    right_idx=1,
    target_h=DEFAULT_HEIGHT,
    target_w=DEFAULT_WIDTH,
    SUPPRESS_DISPLAY=False,
) -> np.ndarray:

    img = to_img_arr(img)
    if points is None:
        print("Please select the eyes for alignment.")
        points = pick_points(img, 2)
    points = to_points(points)
    left_eye, right_eye = points[left_idx], points[right_idx]
    if not SUPPRESS_DISPLAY:
        print("eye coordinates:", left_eye, right_eye)

    # rescale
    actual_eye_len = np.sqrt(
        (right_eye[1] - left_eye[1]) ** 2 + (right_eye[0] - left_eye[0]) ** 2
    )
    diff = abs(actual_eye_len - DEFAULT_EYE_LEN) / DEFAULT_EYE_LEN
    scale = DEFAULT_EYE_LEN / actual_eye_len

    if diff > 0.2:
        assert not np.isnan(img).any()
        assert scale < 5, f"unreasonable scale of {scale}"
        if not SUPPRESS_DISPLAY:
            print(f"scaling by {scale}")
        scaled = transform.rescale(
            img,
            scale=scale,
            preserve_range=True,
            multichannel=True,
            mode=PAD_MODE,
        ).clip(0, 1)
    else:
        scaled = img
    assert_img_type(scaled)

    # do crop
    scaled_h, scaled_w = scaled.shape[0], scaled.shape[1]
    col_center, row_center = find_centers(left_eye * scale, right_eye * scale)
    row_center += 50

    col_shift = int(target_w // 2)
    row_shift = int(target_h // 2)

    col_start = col_center - col_shift
    col_end = col_center + col_shift
    row_start = row_center - row_shift
    row_end = row_center + row_shift

    rpad_before, rpad_after, cpad_before, cpad_after = 0, 0, 0, 0
    if target_h % 2 != 0:
        rpad_after = 1
    if target_w % 2 != 0:
        cpad_after = 1

    if row_start < 0:
        rpad_before += abs(row_start)
        row_start = 0
        row_end += rpad_before
    if row_end > scaled_h:
        rpad_after += row_end - scaled_h
    if col_start < 0:
        cpad_before += abs(col_start)
        col_start = 0
        col_end += cpad_before
    if col_end > scaled_w:
        cpad_after += col_end - scaled_w
    padded = np.pad(
        scaled,
        pad_width=((rpad_before, rpad_after), (cpad_before, cpad_after), (0, 0)),
        mode=PAD_MODE,
    )
    assert row_start >= 0 and row_end >= 0 and col_start >= 0 and col_end >= 0
    if target_h % 2 != 0:
        row_end += 1
    if target_w % 2 != 0:
        col_end += 1
    aligned = padded[row_start:row_end, col_start:col_end, :]

    assert aligned.shape[0] == DEFAULT_HEIGHT and aligned.shape[1] == DEFAULT_WIDTH
    assert_img_type(aligned)
    return aligned, points


def match_img_size(im1: np.ndarray, im2: np.ndarray):
    # Make images the same size
    h1, w1, c1 = im1.shape
    h2, w2, c2 = im2.shape
    assert c1 == c2

    if h1 < h2:
        im2 = im2[int(np.floor((h2 - h1) / 2.0)) : -int(np.ceil((h2 - h1) / 2.0)), :, :]
    elif h1 > h2:
        im1 = im1[int(np.floor((h1 - h2) / 2.0)) : -int(np.ceil((h1 - h2) / 2.0)), :, :]
    if w1 < w2:
        im2 = im2[:, int(np.floor((w2 - w1) / 2.0)) : -int(np.ceil((w2 - w1) / 2.0)), :]
    elif w1 > w2:
        im1 = im1[:, int(np.floor((w1 - w2) / 2.0)) : -int(np.ceil((w1 - w2) / 2.0)), :]
    assert im1.shape == im2.shape
    return im1, im2


###################
#     DISPLAY     #
###################


def plot_points(img: ToImgArray, points: ToPoints, annotate=True) -> None:
    img = to_img_arr(img)
    points = to_points(points)

    fig, ax = plt.subplots()
    plt.imshow(img)
    ax.scatter(points[:, 0], points[:, 1])
    n = np.arange(0, len(points))
    for i, txt in enumerate(n):
        ax.annotate(txt, (points[:, 0][i], points[:, 1][i]))
    plt.show()
