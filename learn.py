import torch
import torch.nn.functional as F
import torch.nn.functional as F
from skimage import io
from torch.nn import (Conv2d, Flatten, Identity, Linear, MaxPool2d, Module,
                      MSELoss, ReLU, Sequential)
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm.contrib import tenumerate

from display import show_keypoints


# do training for one epoch
def train(train_loader, model, learning_rate):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    loss_fn = F.mse_loss
    optimizer = Adam(model.parameters(), lr=learning_rate)

    loss_per_batch = []
    for i, (batched_imgs, batched_keypts) in tenumerate(train_loader):
        # use GPU if available
        batched_imgs = batched_imgs.to(device)
        batched_keypts = batched_keypts.to(device)

        # predict keypoints with current model
        pred_keypts = model(batched_imgs)

        # Compute loss
        loss = loss_fn(pred_keypts, batched_keypts)
        loss_per_batch.append(loss.item())  # .item() to look pretty

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return model, sum(loss_per_batch) / len(loss_per_batch)  # return the average loss


# do testing
def test(test_loader, trained_model, show_every=1, save=False):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    loss_fn = F.mse_loss
    loss_per_batch = []
    imgs, keypts, pred_pts = [], [], []
    for i, (batched_imgs, batched_keypts) in tenumerate(test_loader):
        
        # use GPU if available
        batched_imgs = batched_imgs.to(device)
        batched_keypts = batched_keypts.to(device)

        # predict keypoints with trained model
        pred_keypts = trained_model(batched_imgs)

        # compute and print loss.
        loss = loss_fn(pred_keypts, batched_keypts)
        print(f"batch{i}", loss.item())

        if i % show_every == 0:
            chosen = [1, 7, 15]
            for i in chosen:
                show_keypoints(batched_imgs[i], batched_keypts[i], pred_keypts[i])

        loss_per_batch.append(loss.item())
        
        # if save:
        #     results.extend((batched_imgs, batched_keypts, pred_keypts))
        
    results = [imgs, keypts, pred_pts]
    return results, sum(loss_per_batch) / len(loss_per_batch)  # return the average loss
