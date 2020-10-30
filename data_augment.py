import argparse
import functools
import itertools
import logging
import os
import random
import re
import shutil
import subprocess
import sys
import typing
from copy import deepcopy
from functools import reduce
from logging import (
    debug,
    info,
    log,
)
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
import torch
import torch.nn.functional as F
import torchvision
import torchvision.io as TIO
import torchvision.transforms as TT
import torchvision.transforms.functional as TF
from skimage import (
    io,
    transform,
)
from skimage.util import img_as_float
from torch import (
    Tensor,
    distributions,
    nn,
    tensor,
)
from torch.nn import (
    Linear,
    ReLU,
    Sequential,
    Softmax,
)
from torch.optim import Adam
from torch.utils.data import (
    DataLoader,
    Dataset,
    TensorDataset,
)
from torchvision import utils


class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, image, landmarks):

        h, w = image.shape[2:]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        image = transform.resize(image, (new_h, new_w))

        # h and w are swapped for landmarks because for images,
        # x and y axes are axis 1 and 0 respectively
        landmarks = landmarks * [new_w / w, new_h / h]

        return image, landmarks


class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, image, landmarks):

        h, w = image.shape[2:]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top : top + new_h, left : left + new_w]

        landmarks = landmarks - [left, top]

        return image, landmarks