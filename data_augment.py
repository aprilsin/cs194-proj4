import math
import typing
from pathlib import Path
from typing import (Callable, Dict, FrozenSet, Iterable, List, NamedTuple,
                    NewType, Optional, Sequence, Set, Tuple, TypeVar, Union)

import numpy as np
import skimage.transform as ST
import torch
import torchvision.transforms as TT
from skimage import io
from torch import Tensor

from my_types import assert_img, assert_points


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

    out_h, out_w = 240, 320
    image = TT.Resize((out_h, out_w))(image)

    assert_img(image)
    assert_points(keypoints) # do nothing, should be ratios
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

def part3_augment(img, keypts):
    return img, keypts
