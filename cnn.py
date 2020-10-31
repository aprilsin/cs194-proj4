# Convolutional Neural Networks

import torch
import torch.nn.functional as F
from torch.nn import (
    Conv2d,
    Flatten,
    Identity,
    Linear,
    MaxPool2d,
    Module,
    ReLU,
    Sequential,
)


class NoseFinder(Module):
    def __init__(self):
        super().__init__()

        # # unused variable, only for reference
        # self.model = Sequential(
        #     # layer 1
        #     Conv2d(1, 15, 5),  # 56,76,
        #     ReLU(),
        #     MaxPool2d(5),  # 52,72
        #     # layer 2
        #     Conv2d(15, 28, 3),  # 50,70
        #     ReLU(),
        #     MaxPool2d(3),  # 48,68
        #     # layer 3
        #     Conv2d(28, 20, 5),  # 44,64
        #     ReLU(),
        #     MaxPool2d(5),  # 40,60
        #     # layer 4 (skip for now)
        #     Flatten(),
        #     # fully connected layer 1
        #     Linear(20 * 40 * 60, 8),
        #     ReLU(),
        #     # fully connected layer 2
        #     Linear(8, 2 * 58),
        # )

        # formula: (N - F) / stride + 1
        self.C1 = Conv2d(1, 15, 3)
        self.C2 = Conv2d(15, 28, 3)
        self.C3 = Conv2d(28, 20, 3)
        self.C4 = Identity()  # do nothing for now
        self.FC1 = Linear(20 * 5 * 7, 128)
        self.FC2 = Linear(128, 2 * 58)

    def forward(self, img):
        print(img.shape)  # (b, 1, h=480, w=640)
        
        r = ReLU()
        mp3 = MaxPool2d(3)
        mp5 = MaxPool2d(5)
        mp7 = MaxPool2d(7)

        x = self.C1(img)
        print("C1: ", x.shape)
        x = r(x)
        x = mp5(x)
        print("C1: ", x.shape)

        x = self.C2(x)
        print("C2: ", x.shape)
        x = r(x)
        x = mp3(x)
        print("C2: ", x.shape)

        x = self.C3(x)
        print("C3: ", x.shape)
        x = r(x)
        x = mp5(x)
        print("C3: ", x.shape)

        x = self.C4(x)
        # x = r(x)
        # x = mp5(x)
        print("C4: ", x.shape)

        x = Flatten()(x)

        x = self.FC1(x)
        print("FC1: ", x.shape)

        x = r(x)
        print(x.shape)

        x = self.FC2(x)
        # x=torch.sigmoid(x)
        x = x.reshape(-1, 58, 2)
        print("FC2: ", x.shape)

        return x
