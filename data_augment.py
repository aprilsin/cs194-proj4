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
    image = ST.rotate(image, angle=rotate_deg, center=(0, 0))
    for i in range(len(keypoints)):
        point = keypoints[i]
        qx, qy = rotate(point, (0, 0), -rotate_deg)
        keypoints[i][0] = qx
        keypoints[i][1] = qy

    # convert back to tensors
    image = torch.from_numpy(image)
    image = torch.unsqueeze(image, 0)
    keypoints = torch.from_numpy(keypoints)
    # print(image.shape, keypoints.shape)
    return image, keypoints


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
