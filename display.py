from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor


def valid_keypoints(image, keypoints):
    h, w = image.shape
    rows = keypoints[:, 0]
    cols = keypoints[:, 1]

    assert (0 <= rows).all() and (
        rows < h
    ).all(), f"{rows.min() = }, {rows.max() = } out of bounds for {image.shape = }"
    assert (0 <= cols).all() and (
        cols < h
    ).all(), f"{cols.min() = }, {cols.max() = } out of bounds for {image.shape = }"
    return True


ToDisplayImage = Union[Tensor, np.ndarray]


def to_display_img(img: ToDisplayImage) -> np.ndarray:
    if img.ndim == 2 and isinstance(img, np.ndarray):
        return img

    assert isinstance(img, Tensor), type(img)

    if img.ndim == 2:
        return img.numpy()

    if img.ndim == 3 or img.ndim == 4:
        # remove dimension of size 1 for plt
        img = torch.squeeze(img)
        img = torch.detach(img).numpy()
        return img

    else:
        raise ValueError()


def assert_points(pts):
    assert pts.ndim == 2, pts.shape
    assert pts.shape[1] == 2, pts.shape


ToDisplayPoints = Union[Tensor, np.ndarray]


def to_display_pts(pts: ToDisplayPoints) -> np.ndarray:
    if pts.ndim == 2 and pts.shape[1] == 2 and isinstance(pts, np.ndarray):
        return pts

    assert isinstance(pts, Tensor), type(pts)

    if pts.ndim == 2 and pts.shape[1] == 2:
        return pts.numpy()

    if pts.ndim == 3:
        # remove dimension of size 1 for plt
        pts = torch.squeeze(pts, 0)
        pts = torch.detach(pts).numpy()
        return pts

    else:
        raise ValueError()


def show_keypoints(
    image: ToDisplayImage,
    truth_points: Union[Tensor, np.ndarray],
    pred_points: Union[Tensor, np.ndarray] = None,
) -> None:
    """Show image with keypoints"""

    # make everything numpy arrays
    image = to_display_img(image)
    assert image.ndim == 2, image.shape

    truth_points = to_display_pts(truth_points)
    assert truth_points.ndim == 2 and truth_points.shape[1] == 2, truth_points.shape
    assert valid_keypoints(image, truth_points)

    if pred_points is not None:
        pred_points = to_display_pts(pred_points)
        assert pred_points.ndim == 2 and pred_points.shape[1] == 2, pred_points.shape
        assert valid_keypoints(image, pred_points)

    h, w = image.shape
    print(f"{image.shape = }")

    # show image and plot keypoints
    plt.figure()
    # print(truth_points[:, 0], truth_points[:, 1])
    # print(truth_points[:, 0] * h, truth_points[:, 1] * w)
    # print(truth_points[:, 0] * w, truth_points[:, 1] * h)
    plt.imshow(image, cmap="gray")
    plt.scatter(truth_points[:, 0], truth_points[:, 1], s=35, c="g", marker="x")
    # plt.scatter(truth_points[:, 0] * w, truth_points[:, 1] * h, s=35, c="g", marker="x")
    if pred_points is not None:
        plt.scatter(
            pred_points[:, 0] * h, pred_points[:, 1] * w, s=35, c="r", marker="x"
        )

    plt.pause(0.001)  # pause a bit so that plots are updated
    plt.show()

def show_training_progress():
    pass

def show_epoch():
    pass
