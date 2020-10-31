import typing
from pathlib import Path
from typing import (Callable, Dict, FrozenSet, Iterable, List, NamedTuple,
                    NewType, Optional, Sequence, Set, Tuple, TypeVar, Union)

import numpy as np
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


def part2_augment(image, keypoitns):

    rotate = TT.RandomRotation((-15, 15))
    image = rotate(image)
    keypoints = rotate(keypoints)

    return image, keypoints
