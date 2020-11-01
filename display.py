from typing import Union

import matplotlib.pyplot as plt
import numpy as np
from torch import Tensor

ToDisplayImage = Union[Tensor, np.ndarray]


def to_display_img(img: ToDisplayImage, color=False) -> np.ndarray:
    if img.ndim == 2 and isinstance(img, np.ndarray):
        return img

    assert isinstance(img, Tensor), type(img)
    img = img.detach().cpu()
    if img.ndim == 2:
        return img.numpy()

    if img.ndim == 3:

        # change channels first to channels last format
        img = img.permute(1, 2, 0)

        if not color:
            # remove dimension of size 1 for plt
            img = img.squeeze()
        return img.numpy()

    else:
        raise ValueError(type(img), img.shape)


ToDisplayPoints = Union[Tensor, np.ndarray]


def to_display_pts(pts: ToDisplayPoints) -> np.ndarray:
    # print(pts.ndim, list(pts.shape), type(pts))

    if isinstance(pts, np.ndarray):
        assert pts.ndim == 2, pts.shape
        return pts
    pts = pts.detach().cpu()

    assert isinstance(pts, Tensor), type(pts)
    # if pts.numel() == 0:
    #     return pts

    if pts.ndim == 2:
        assert list(pts.shape)[1] == 2
        pts = pts.numpy()
        return pts
    if pts.ndim == 3:
        # remove dimension of size 1 for plt
        pts = pts.squeeze(0)
        return pts.numpy()
    else:
        raise ValueError(type(pts), pts.shape)


def show_keypoints(
    image: ToDisplayImage,
    truth_points: Union[Tensor, np.ndarray] = None,
    pred_points: Union[Tensor, np.ndarray] = None,
    color: bool = False,
) -> None:
    """Show image with keypoints."""

    # make everything numpy arrays with correct shape
    image = to_display_img(image)
    assert image.ndim == 2 or image.ndim == 3, image.shape

    if truth_points is not None:
        truth_points = to_display_pts(truth_points)
        assert truth_points.ndim == 2 and truth_points.shape[1] == 2, truth_points.shape

    if pred_points is not None:
        pred_points = to_display_pts(pred_points)
        assert pred_points.ndim == 2 and pred_points.shape[1] == 2, pred_points.shape

    h, w = image.shape[0], image.shape[1]

    # show image and plot keypoints
    plt.figure()
    cmap = "viridis" if color else "gray"
    plt.imshow(image, cmap)
    if truth_points is not None:
        plt.scatter(
            truth_points[:, 0] * w, truth_points[:, 1] * h, s=35, c="g", marker="x"
        )
    if pred_points is not None:
        plt.scatter(
            pred_points[:, 0] * w, pred_points[:, 1] * h, s=35, c="r", marker="x"
        )
    plt.pause(0.001)  # pause a bit so that plots are updated
    plt.show()


def show_progress(loss_per_ep:List, title="Loss over Epochs"):
    loss_per_epoch = np.array(loss_per_epoch)
    x = np.arange(len(loss_per_ep))
    plt.figure()
    plt.title(title)
    plt.ylabel("loss")
    plt.xlabel("epochs")
    plt.xticks(x)
    plt.plot(loss_per_ep[:, 0])


def print_epoch(ep, train_loss, valid_loss) -> None:
    print()
    print(f"========== Epoch {ep} Results ==========")
    print(f"{train_loss = }")
    print(f"{valid_loss = }")


def show_sucess(ep, results):
    for (img, true_pts, pred_pts) in results:
        pass
