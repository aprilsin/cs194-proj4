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

def show_keypoints(data):
    """Show image with keypoints"""
    assert type(data) == tuple, type(data)
    assert len(data) == 2, len(data)

    # unpack a sample data
    image, keypoints = data
    assert torch.is_tensor(image), type(image)

    # change channels-first to channels-last format
    image = image.permute(1, 2, 0)

    # remove dimension of size 1 for plt
    image = torch.squeeze(image)

    assert valid_keypoints(image, keypoints)

    # show image and plot keypoints
    plt.imshow(image, cmap="gray")
    plt.scatter(keypoints[:, 0], keypoints[:, 1], s=10, marker=".", c="g")
    plt.pause(0.001)  # pause a bit so that plots are updated

