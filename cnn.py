# Convolutional Neural Networks

import torch
from torch.nn import Module, Sequential, Conv2d, ReLU, MaxPool2d, Linear, Flatten
import torch.nn.functional as F

# CNN for Part 1
nosefinder = Sequential(
    # layer 1
    Conv2d(1, 15, 5),  # 56,76,
    ReLU(),
    MaxPool2d(5),  # 52,72
    # layer 2
    Conv2d(15, 28, 3),  # 50,70
    ReLU(),
    MaxPool2d(3),  # 48,68
    # layer 3
    Conv2d(28, 20, 5),  # 44,64
    ReLU(),
    MaxPool2d(5),  # 40,60
    # layer 4 (skip for now)
    Flatten(),
    # fully connected layer 1
    Linear(20 * 40 * 60, 8),
    ReLU(),
    # fully connected layer 2
    Linear(8, 2),
)

# class NoseFinder(nn.Module):
#         def __init__(self):
#         super().__init__()
#         self.conv1 =
