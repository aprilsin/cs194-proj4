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


def show_progress(loss_per_ep: Union[list, np.ndarray], title="Loss over Epochs"):
    if isinstance(loss_per_ep, list):
        loss_per_epoch = np.array(loss_per_epoch)

    assert loss_per_ep.ndim == 2

    x = int(loss_per_ep[:, 0])
    plt.figure()
    plt.title(title)
    plt.ylabel("loss")
    plt.xlabel("epochs")
    plt.xticks(x)
    plt.plot(loss_per_ep[:, 1])


def show_progress_both(
    training_loss: Union[list, np.ndarray],
    validation_loss: Union[list, np.ndarray],
    title="Loss over Epochs",
):
    if isinstance(training_loss, list):
        training_loss = np.array(training_loss)
    if isinstance(validation_loss, list):
        validation_loss = np.array(validation_loss)

    assert training_loss.ndim == 2
    assert validation_loss.ndim == 2

    x = np.int64(training_loss[:, 0])
    x = np.int64(validation_loss[:, 0])
    plt.figure()
    plt.title(title)
    plt.ylabel("loss")
    plt.xlabel("epochs")
    plt.xticks(x)
    plt.plot(training_loss[:, 1])
    plt.plot(validation_loss[:, 1])
    plt.show()


# def print_epoch(ep, train_loss, valid_loss) -> None:
#     print()
#     print(f"========== Epoch {ep} Results ==========")
#     print(f"{train_loss = }")
#     print(f"{valid_loss = }")


# def show_sucess(ep, results):
#     for (img, true_pts, pred_pts) in results:
#         pass
