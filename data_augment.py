import typing
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
import skimage.transform as ST
from skimage import io
import torch
from torch import Tensor
import torchvision.transforms as TT


def part1_augment(image, keypoints) -> Tuple[Tensor, Tensor]:
    h, w = image.shape[-2:]

    # resize
    out_h, out_w = 240, 320
    image = TT.Resize((out_h, out_w))(image)
    keypoints[..., 0] = keypoints[..., 0] * out_h / h
    keypoints[..., 1] = keypoints[..., 1] * out_w / w

    return image, keypoints


def part2_augment(image, keypoints) -> Tuple[Tensor, Tensor]:
    # print(image.shape, keypoints.shape)
    # pts = torch.unsqueeze(keypoints, 1)
    # print(pts)
    # jitter = TT.ColorJitter(brightness=0.3)
    # image = jitter(image)

    # convert tensors to numpy arrays to use skimage
    image, keypoints = image.squeeze().numpy(), keypoints.numpy()
    h, w = image.shape
    center = (h / 2, w / 2)

    rotate_deg = np.random.randint(-15, 15)
    print(rotate_deg)
    image = ST.rotate(image, angle=rotate_deg, center=(0, 0))

    print(keypoints[52])
    for i in range(len(keypoints)):
        point = keypoints[i]
        qx, qy = rotate(point, (0, 0), -rotate_deg)
        keypoints[i][0] = qx
        keypoints[i][1] = qy
    print(keypoints[52])

    # convert back to tensors
    image = torch.from_numpy(image)
    image = torch.unsqueeze(image, 0)
    keypoints = torch.from_numpy(keypoints)
    print(image.shape, keypoints.shape)
    return image, keypoints


def rotation_mat(rot_deg, center):
    center_h, center_w = center
    center_x, center_y = center_w, center_h
    theta = np.radians(rot_deg)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array(((c, -s, center_x - center_y), (s, c, center_x + center_y)))
    return R


import math


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
