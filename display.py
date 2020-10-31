import numpy as np
import matplotlib.pyplot as plt
import torch


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


def show_keypoints(
    image: Union[Tensor, np.ndarray],
    truth_points: Union[Tensor, np.ndarray],
    pred_points: Union[Tensor, np.ndarray] = None,
)->None:
    """Show image with keypoints"""
    
    # check inputs have the correct shape
    assert image.ndim == 2, image.shape
    assert truth_points.ndim == 2 and truth_point.shape[1] == 2, truth_points.shape
    if pred_points is not None:
        assert pred_points.ndim == 2 and pred_point.shape[1] == 2, pred_points.shape


    assert image.ndim == 2, image.shape
    assert truth_points.ndim == 2, truth_points.shape
    # # change channels-first to channels-last format
    # image = image.permute(1, 2, 0)

    # # remove dimension of size 1 for plt
    # image = torch.squeeze(image)

    # show image and plot keypoints
    plt.figure()
    plt.imshow(image, cmap="gray")
    assert valid_keypoints(image, truth_points)
    plt.scatter(truth_points[:, 0], truth_points[:, 1], s=35, c="g", marker="x")

    if pred_points is not None:
        assert torch.is_tensor(pred_points), type(image)
        pred_points = pred_points.detach().numpy()
        assert pred_points.ndim == 2
        assert valid_keypoints(image, pred_points)
        plt.scatter(pred_points[:, 0], pred_points[:, 1], s=35, c="r", marker="x")

    plt.pause(0.001)  # pause a bit so that plots are updated
    plt.show()
