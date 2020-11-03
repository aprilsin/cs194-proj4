# Convolutional Neural Networks
import torch
import torchvision
from antialiased_cnns import BlurPool
from torch import nn
from torch.nn import Conv2d, Flatten, Linear, MaxPool2d, Module, ReLU


class NoseFinder(Module):
    def __init__(self):
        super().__init__()

        # formula: (N - F) / stride + 1
        self.C1 = Conv2d(1, 24, 3)  # Conv2d(1, 15, 3)
        self.C2 = Conv2d(24, 30, 3)  # Conv2d(15, 28, 3)
        self.C3 = Conv2d(30, 20, 3)  # Conv2d(28, 20, 3)
        # self.C4 = Conv2d(30, 20, 3)

        self.FC1 = Linear(20 * 7 * 10, 128)
        self.FC2 = Linear(128, 2 * 1)

        self.model = nn.Sequential(
            Conv2d(1, 24, 3),
            ReLU(),
            MaxPool2d(3),
            Conv2d(24, 30, 3),
            ReLU(),
            MaxPool2d(3),
            Conv2d(30, 20, 3),
            ReLU(),
            MaxPool2d(3),
            Flatten(),
            Linear(20 * 7 * 10, 128),
            ReLU(),
            Linear(128, 2 * 1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        return self.model(img).reshape(-1, 1, 2)


class BlurFaceFinder(Module):
    def __init__(self):
        super().__init__()

        # formula: (N - F) / stride + 1
        self.C1 = Conv2d(1, 18, 3)
        self.C2 = Conv2d(18, 24, 3)
        self.C3 = Conv2d(24, 30, 3)

        self.C4 = Conv2d(30, 30, 3)
        self.C5 = Conv2d(30, 25, 3)

        self.FC1 = Linear(25 * 54 * 74, 128)
        self.FC2 = Linear(128, 2 * 58)

    def forward(self, img):

        dev = self.C1.weight.device
        r = ReLU().to(dev)

        x = self.C1(img)
        x = r(x)
        x = BlurPool(self.C1.out_channels).to(dev)(x)
        x = self.C2(x)
        x = r(x)
        # x = mp3(x)

        x = self.C3(x)
        x = r(x)
        x = BlurPool(self.C3.out_channels).to(dev)(x)

        x = self.C4(x)
        x = r(x)
        # x = mp3(x)

        x = self.C5(x)
        x = r(x)
        # x = mp3(x)

        x = Flatten()(x)

        x = self.FC1(x)
        x = r(x)

        x = self.FC2(x)
        x = torch.sigmoid(x)
        x = x.reshape(-1, 58, 2)

        return x


class FaceFinder(Module):
    def __init__(self):
        super().__init__()

        # formula: (N - F) / stride + 1
        self.C1 = Conv2d(1, 18, 3)
        self.C2 = Conv2d(18, 24, 3)
        self.C3 = Conv2d(24, 30, 3)

        self.C4 = Conv2d(30, 30, 3)
        self.C5 = Conv2d(30, 25, 3)

        self.FC1 = Linear(25 * 21 * 30, 128)
        self.FC2 = Linear(128, 2 * 58)

    def forward(self, img):

        dev = self.C1.weight.device
        r = ReLU().to(dev)
        mp3 = MaxPool2d(3).to(dev)

        x = self.C1(img)
        x = r(x)
        x = mp3(x)
        x = self.C2(x)
        x = r(x)
        # x = mp3(x)

        x = self.C3(x)
        x = r(x)
        x = mp3(x)

        x = self.C4(x)
        x = r(x)
        # x = mp3(x)

        x = self.C5(x)
        x = r(x)
        # x = mp3(x)

        x = Flatten()(x)

        x = self.FC1(x)
        x = r(x)

        x = self.FC2(x)
        x = torch.sigmoid(x)
        x = x.reshape(-1, 58, 2)

        return x


class ResNet(Module):
    def __init__(self):
        super().__init__()

        self.model = torchvision.models.resnet18()
        self.model.conv1 = Conv2d(
            1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
        )
        # self.model.conv1.in_channels = 1
        # self.model.fc.out_features = 68 * 2
        self.model.fc = Linear(in_features=512, out_features=136, bias=True)

    def forward(self, x):
        x = self.model(x)
        x = torch.sigmoid(x)
        x = x.reshape(-1, 68, 2)
        return x
