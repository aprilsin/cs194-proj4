from typing import Union

import matplotlib.pyplot as plt
import numpy as np
from torch import Tensor

import cnn

ToDisplayImage = Union[Tensor, np.ndarray]


def to_display_img(img: ToDisplayImage, color=False) -> np.ndarray:
    if color:
        assert isinstance(img, np.ndarray), type(img)
        assert img.ndim == 3, img.shape
        return img

    if isinstance(img, np.ndarray):
        assert img.ndim == 2, img.shape
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
    image = to_display_img(image, color=color)

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


def show_progress(loss_per_epoch: Union[list, np.ndarray], title="Loss over Epochs"):
    if isinstance(loss_per_epoch, list):
        loss_per_epoch = np.array(loss_per_epoch)

    assert loss_per_epoch.ndim == 2

    x = np.int64(loss_per_epoch[:, 0])
    plt.figure()
    plt.title(title)
    plt.ylabel("loss")
    plt.xlabel("epochs")
    plt.xticks(x)
    plt.plot(loss_per_epoch[:, 1])


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
    fig, ax = plt.subplots()

    ax.set_title(title)
    ax.set_ylabel("loss")
    ax.set_xlabel("epochs")
    ax.set_xticks(x)
    ax.plot(np.int64(training_loss[:, 0]), training_loss[:, 1], label="Train")
    ax.plot(np.int64(validation_loss[:, 0]), validation_loss[:, 1], label="Val")
    ax.legend()
    return fig


# def print_epoch(ep, train_loss, valid_loss) -> None:
#     print()
#     print(f"========== Epoch {ep} Results ==========")
#     print(f"{train_loss = }")
#     print(f"{valid_loss = }")


# def show_sucess(ep, results):
#     for (img, true_pts, pred_pts) in results:
#         pass


def make_filter_fig(conv_layer, num_x, num_y):

    # permute is necessary to get channels in the right place for matplotlib
    w = conv_layer.weight.detach().clone().permute(0, 2, 3, 1).numpy()

    w = (w - w.min()) / (w.max() - w.min())
    # reduce across channels (grayscale it) so you can plot more than 1D and 3D input.
    w = w.mean(-1)
    num_filters = len(w)
    assert num_filters == num_x * num_y  # to make sure the filters will fit

    fig, axes = plt.subplots(num_x, num_y, sharex=True, sharey=True, figsize=(10, 10))
    for filter, ax in zip(w, axes.flat):
        ax.imshow(filter)
        ax.set_axis_off()
    return fig


def show_filters_part2(model: cnn.FaceFinder):
    model = model.cpu()
    return [make_filter_fig(model.C1, 6, 3), make_filter_fig(model.C2, 6, 4)]
