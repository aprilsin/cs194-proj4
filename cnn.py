# Convolutional Neural Networks

import torch
import torch.nn.functional as F
from torch.nn import (Conv2d, Flatten, Identity, Linear, MaxPool2d, Module,
                      ReLU, Sequential)


class NoseFinder(Module):
    def __init__(self):
        super().__init__()

        # formula: (N - F) / stride + 1
        self.C1 = Conv2d(1, 24, 3) # Conv2d(1, 15, 3)
        self.C2 = Conv2d(24, 24, 3) # Conv2d(15, 28, 3)
        self.C3 = Conv2d(24, 20, 3) # Conv2d(28, 20, 3)

        self.FC1 = Linear(20 * 7 * 10, 128)
        self.FC2 = Linear(128, 2 * 1)

    def forward(self, img):
        # print(img.shape)

        r = ReLU()
        mp3 = MaxPool2d(3)
        # mp5 = MaxPool2d(5)
        # mp7 = MaxPool2d(7)

        x = self.C1(img)
        # print("C1: ", x.shape)
        x = r(x)
        x = mp3(x)
        # print("m1: ", x.shape)

        x = self.C2(x)
        # print("C2: ", x.shape)
        x = r(x)
        x = mp3(x)
        # print("m2: ", x.shape)

        x = self.C3(x)
        # print("C3: ", x.shape)
        x = r(x)
        x = mp3(x)
        # print("m3: ", x.shape)

        x = Flatten()(x)

        x = self.FC1(x)
        # print("FC1: ", x.shape)

        x = r(x)
        # print(x.shape)

        x = self.FC2(x)
        # x=torch.sigmoid(x)
        x = x.reshape(-1, 1, 2)
        # print("FC2: ", x.shape)

        return x


class FaceFinder(Module):
    def __init__(self):
        super().__init__()

        # formula: (N - F) / stride + 1
        self.C1 = Conv2d(1, 15, 3)
        self.C2 = Conv2d(15, 28, 3)
        self.C3 = Conv2d(28, 20, 3)

        # FIXME
        self.C4 = Conv2d(20, 32, 3)
        self.C5 = Conv2d(32, 25, 3)

        self.FC1 = Linear(25 * 7 * 10, 128)
        self.FC2 = Linear(128, 2 * 58)

    def forward(self, img):
        # print(img.shape)

        r = ReLU()
        mp3 = MaxPool2d(3)
        # mp5 = MaxPool2d(5)
        # mp7 = MaxPool2d(7)

        x = self.C1(img)
        # print("C1: ", x.shape)
        x = r(x)
        x = mp3(x)
        # print("m1: ", x.shape)

        x = self.C2(x)
        # print("C2: ", x.shape)
        x = r(x)
        x = mp3(x)
        # print("m2: ", x.shape)

        x = self.C3(x)
        # print("C3: ", x.shape)
        x = r(x)
        x = mp3(x)
        # print("m3: ", x.shape)

        x = self.C4(x)
        # print("C4: ", x.shape)
        x = r(x)
        x = mp3(x)
        # print("m4: ", x.shape)

        x = self.C5(x)
        # print("C5: ", x.shape)
        x = r(x)
        x = mp3(x)
        print("m5: ", x.shape)

        x = Flatten()(x)

        x = self.FC1(x)
        # print("FC1: ", x.shape)

        x = r(x)
        # print(x.shape)

        x = self.FC2(x)
        # x=torch.sigmoid(x)
        x = x.reshape(-1, 58, 2)
        # print("FC2: ", x.shape)

        return x
