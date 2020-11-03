import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from tqdm import trange
from tqdm.contrib import tenumerate

import display
from constants import DEVICE


# do training for one epoch
def train(train_loader, model, learning_rate):
    model = model.to(DEVICE)

    optimizer = Adam(model.parameters(), lr=learning_rate)

    loss_per_batch = []
    for i, (batched_imgs, batched_keypts) in tenumerate(train_loader):
        batched_imgs = batched_imgs.to(DEVICE)
        batched_keypts = batched_keypts.to(DEVICE)

        # predict keypoints with current model
        pred_keypts = model(batched_imgs)

        # Compute loss
        loss = F.mse_loss(pred_keypts, batched_keypts)
        loss_per_batch.append(loss.item())  # .item() to look pretty

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return model, mean(loss_per_batch)


def mean(xs):
    return sum(xs) / len(xs)


# do validation
def validate(valid_loader, model, show_every=1):  # default: show every batch
    model = model.to(DEVICE)
    loss_per_batch = []
    imgs, keypts, pred_pts = [], [], []
    with torch.no_grad():
        for i, (batched_imgs, batched_keypts) in tenumerate(valid_loader):

            batched_imgs = batched_imgs.to(DEVICE)
            batched_keypts = batched_keypts.to(DEVICE)

            # predict keypoints with trained model
            batched_pred_keypts = model(batched_imgs)

            # compute and print loss.
            loss = F.mse_loss(batched_pred_keypts, batched_keypts)

            if show_every is not None and i % show_every == 0:
                chosen = [1, 12, 18]
                for i in chosen:
                    display.show_keypoints(
                        batched_imgs[i], batched_keypts[i], batched_pred_keypts[i]
                    )

            loss_per_batch.append(loss.item())
            imgs.extend(batched_imgs)
            keypts.extend(batched_keypts)
            pred_pts.extend(batched_pred_keypts)

    return [imgs, keypts, pred_pts], mean(loss_per_batch)


def test(test_loader, model):
    model = model.to(DEVICE)
    imgs, pred_pts = [], []
    with torch.no_grad():
        for i, batched_imgs in tenumerate(test_loader):

            batched_imgs = batched_imgs.to(DEVICE)

            # predict keypoints with trained model
            pred_keypts = model(batched_imgs)
            imgs.extend(batched_imgs)
            pred_pts.extend(pred_keypts)

    return imgs, pred_pts


def train_and_validate(
    train_loader, valid_loader, model, epochs, learn_rate, show_every=10
):
    model = model.to(DEVICE)
    all_train_loss = []
    all_valid_loss = []
    for epoch in trange(epochs):

        model, train_loss = train(train_loader, model, learn_rate)
        print(f"{epoch = }: {train_loss = }")

        if epoch % show_every == 0:
            _, valid_loss = validate(valid_loader, model, show_every)
            all_valid_loss.append([epoch, valid_loss])

        all_train_loss.append([epoch, train_loss])

    return np.array(all_train_loss), np.array(all_valid_loss)
