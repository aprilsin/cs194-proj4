import typing
from pathlib import Path
from typing import (Callable, Dict, FrozenSet, Iterable, List, NamedTuple,
                    NewType, Optional, Sequence, Set, Tuple, TypeVar, Union)

import numpy as np
import torch
import torchvision.transforms as TT



class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, out_h, out_w):
        assert isinstance(out_h, int)
        assert isinstance(out_w, int)
        self.out_h = out_h
        self.out_w = out_w

    def __call__(self, image, keypoints):
        assert isinstance(image, Tensor), type(image)
        assert isinstance(keypoints, Tensor), type(keypoints)

        h, w = image.shape[-2:]
        image = TT.Resize((self.out_h, self.out_w))(image)
        keypoints[..., 0] = keypoints[..., 0] * self.out_h / h
        keypoints[..., 1] = keypoints[..., 1] * self.out_w / w

        image = torch.as_tensor(image)
        # keypoints = torch.as_tensor(keypoints)
        assert isinstance(image, Tensor), type(image)
        assert isinstance(keypoints, Tensor), type(keypoints)
        return image, keypoints


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