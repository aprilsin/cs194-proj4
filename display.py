from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor


def valid_keypoints(image, keypoints):
    h, w = image.shape
    rows = keypoints[:, 0]
    cols = keypoints[:, 1]

    # make sure that the keypoints are indices and not ratios
    assert rows.max() >= 1 and cols.max() >= 1, f"{rows.max()}, {cols.max()}"

    # make sure that the keypoints are in bounds
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
    # print(pts.ndim, list(pts.shape), type(pts))

    if isinstance(pts, np.ndarray):
        assert pts.ndim == 2, pts.shape
        return pts

    assert isinstance(pts, Tensor), type(pts)
    if pts.ndim == 2:
        assert list(pts.shape)[1] == 2
        pts = torch.detach(pts).numpy()
        return pts
    if pts.ndim == 3:
        # remove dimension of size 1 for plt
        pts = torch.squeeze(pts, 0)
        pts = torch.detach(pts).numpy()
        return pts
    else:
        raise ValueError(type(pts), pts.shape)


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
    # assert valid_keypoints(image, truth_points)

    if pred_points is not None:
        pred_points = to_display_pts(pred_points)
        assert pred_points.ndim == 2 and pred_points.shape[1] == 2, pred_points.shape
        # assert valid_keypoints(image, pred_points)

    h, w = image.shape
    # print(f"{image.shape = }")

    # show image and plot keypoints
    # plt.figure(figsize=image.shape)
    plt.figure()
    plt.imshow(image, cmap="gray")
    plt.scatter(truth_points[:, 0], truth_points[:, 1], s=35, c="g", marker="x")
    # plt.scatter(truth_points[:, 0] * w, truth_points[:, 1] * h, s=35, c="g", marker="x")
    if pred_points is not None:
        plt.scatter(
            pred_points[:, 0], pred_points[:, 1], s=35, c="r", marker="x"
        )
    plt.pause(0.001)  # pause a bit so that plots are updated
    plt.show()


def show_training_progress():
    pass


def print_epoch(ep, train_loss, valid_loss) -> None:
    print()
    print(f"========== Epoch {ep} ==========")
    print(f"{train_loss = }")
    print(f"{valid_loss = }")

def show_sucess(ep, results):
    for (img, true_pts, pred_pts) in results:
        pass
