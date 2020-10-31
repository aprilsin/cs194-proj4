import typing
from pathlib import Path
from typing import (Callable, Dict, FrozenSet, Iterable, List, NamedTuple,
                    NewType, Optional, Sequence, Set, Tuple, TypeVar, Union)

import numpy as np
import skimage.transform as ST
import torch
import torchvision.transforms as TT


def part1_augment(image, keypoints):
    h, w = image.shape[-2:]

    # resize
    out_h, out_w = 240, 320
    image = TT.Resize((out_h, out_w))(image)
    keypoints[..., 0] = keypoints[..., 0] * out_h / h
    keypoints[..., 1] = keypoints[..., 1] * out_w / w

    return image, keypoints


def part2_augment(image, keypoints):

    print(image.shape, keypoints.shape)
    
    jitter = TT.ColorJitter(brightness=0.3)
    image = jitter(image)

    # convert tensors to numpy arrays to use skimage
    image, keypoints = image.numpy(), keypoints.numpy()

    rotate_deg = np.randint(-15, 15)
    image = ST.rotate(image, rotate_deg)
    keypoints = rotation_mat(rotate_deg) * keypoints.T
    # rotate = TT.RandomRotation((-15, 15))
    # print(rotate.get_params())
    # image = rotate(image)
    print(image.shape)
    # keypoints = rotate(keypoints)

    print(image.shape, keypoints.shape)
    return image, keypoints

def rotation_mat(rot_deg):
    theta = np.radians(rot_deg)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array(((c, -s), (s, c)))
    return R
