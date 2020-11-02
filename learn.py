import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from tqdm.contrib import tenumerate

import display
from constants import DEVICE


# do training for one epoch
def train(train_loader, model, learning_rate):
    model = model.to(DEVICE)

    loss_fn = F.mse_loss
    optimizer = Adam(model.parameters(), lr=learning_rate)

    loss_per_batch = []
    for i, (batched_imgs, batched_keypts) in tenumerate(train_loader):
        # use GPU if available
        batched_imgs = batched_imgs.to(DEVICE)
        batched_keypts = batched_keypts.to(DEVICE)

        # predict keypoints with current model
        pred_keypts = model(batched_imgs)

        # Compute loss
        loss = loss_fn(pred_keypts, batched_keypts)
        loss_per_batch.append(loss.item())  # .item() to look pretty

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        break

    return model, sum(loss_per_batch) / len(loss_per_batch)  # return the average loss


# do validation
def validate(
    valid_loader, trained_model, show_every=1
):  # default: show every batch
    loss_fn = F.mse_loss
    loss_per_batch = []
    imgs, keypts, pred_pts = [], [], []
    with torch.no_grad():
        for i, (batched_imgs, batched_keypts) in tenumerate(valid_loader):

            # use GPU if available
            batched_imgs = batched_imgs.to(DEVICE)
            batched_keypts = batched_keypts.to(DEVICE)

            # predict keypoints with trained model
            batched_pred_keypts = trained_model(batched_imgs)

            # compute and print loss.
            loss = loss_fn(batched_pred_keypts, batched_keypts)
            print(f"batch{i}", loss.item())

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

    results = [imgs, keypts, pred_pts]
    return results, sum(loss_per_batch) / len(loss_per_batch)  # return the average loss


# do testing (no loss)
def test(test_loader, trained_model, save=False):
    loss_fn = F.mse_loss
    imgs, keypts, pred_pts = [], [], []
    with torch.no_grad():
        for i, (batched_imgs) in tenumerate(test_loader):

            # use GPU if available
            batched_imgs = batched_imgs.to(DEVICE)

            # predict keypoints with trained model
            pred_keypts = trained_model(batched_imgs)

    results = [imgs, keypts, pred_pts]
    return results


def train_and_validate(
    train_loader, test_loader, model, epochs, learn_rate, show_every=10
):
    all_train_loss = []
    all_valid_loss = []
    for ep in range(epochs):

        print(f"========== Start Epoch {ep} ==========")

        trained_model, train_loss = train(train_loader, model, learn_rate)

        if ep % show_every == 0:
            _, valid_loss = validate(test_loader, trained_model, show_every)
            all_valid_loss.append([ep, valid_loss])

        all_train_loss.append([ep, train_loss])

        display.print_epoch(ep, train_loss, valid_loss)

    return np.array(all_train_loss), np.array(all_valid_loss)
