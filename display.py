import numpy as np
import matplotlib.pyplot as plt
import torch


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

    # show image and plot keypoints
    plt.imshow(image, cmap="gray")
    plt.scatter(keypoints[:, 0], keypoints[:, 1], s=10, marker=".", c="r")
    plt.pause(0.001)  # pause a bit so that plots are updated
