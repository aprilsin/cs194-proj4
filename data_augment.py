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

    # jitter = TT.ColorJitter(brightness=0.3)
    # image = jitter(image)

    # convert tensors to numpy arrays to use skimage
    image, keypoints = image.squeeze().numpy(), keypoints.numpy()
    h, w = image.shape

    rotate_deg = np.random.randint(-15, 15)
    image = ST.rotate(image, rotate_deg, center=(h / 2, w / 2))

    R = rotation_mat(rotate_deg)
    # print(R.shape, keypoints.T)
    keypoints = np.flip(keypoints, axis=0)
    print(keypoints.T.shape)
    keypoints = (R @ keypoints.T).T
    cx, cy = w / 2, h / 2
    for i in range(len(keypoints)):
        
    new_x = cx + (x - cx) * np.cos(angle) - (y - cy) * np.cos(angle),
    new_y = cy + (x - cx) * np.sin(angle) + (y - cy) * np.sin(angle)
    print(keypoints.shape)
    keypoints[:, 0] += h / 2
    keypoints[:, 1] += w / 2

    # convert back to tensors
    image = torch.from_numpy(image)
    image = torch.unsqueeze(image, 0)
    keypoints = torch.from_numpy(keypoints)
    # print(image.shape, keypoints.shape)
    return image, keypoints


def rotation_mat(rot_deg):
    theta = np.radians(rot_deg)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array(((c, -s), (s, c)))
    return R
