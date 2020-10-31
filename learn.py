import torch
import torch.nn.functional as F
import torch.nn.functional as F
from torch.nn import (
    Conv2d,
    Flatten,
    Identity,
    Linear,
    MaxPool2d,
    Module,
    MSELoss,
    ReLU,
    Sequential,
)
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm.contrib import tenumerate

from display import show_keypoints


def train(train_loader, model, learning_rate):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    loss_fn = F.mse_loss
    optimizer = Adam(model.parameters(), lr=learning_rate)

    for i, (batched_imgs, batched_keypts) in tenumerate(train_loader):
        batched_imgs, batched_keypts = batched_imgs.to(device), batched_keypts.to(
            device
        )
        pred_keypts = model(batched_imgs)

        # Compute loss
        loss = loss_fn(pred_keypts, batched_keypts)

        # Print loss of current batch
        print(i, loss.item())  # .item() to look pretty

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return model


def test(test_loader, trained_model, show_every=1):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    loss_fn = F.mse_loss

    loss_per_batch = []
    for i, (img, keypts) in tenumerate(test_loader):
        img, keypts = img.to(device), keypts.to(device)
        pred_keypts = trained_model(img)

        # Compute and print loss.
        loss = loss_fn(pred_keypts, keypts)
        loss_per_batch.append(loss)

        if i % show_every == 0:
            show_keypoints(img, keypts, pred_keypts)
            # print(i, loss.item())

    return sum(loss_per_batch) / len(loss_per_batch)  # return the average loss
