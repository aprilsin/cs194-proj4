get_ipython().run_line_magic("load_ext", " autoreload")
get_ipython().run_line_magic("autoreload", " 2")


import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import skimage.io as io
import torch
import torchvision.transforms as TT
from torch.utils.data import DataLoader

import cnn
from data import (
    FaceKeypointsTrainDataset,
    FaceKeypointsValidDataset,
    LargeTestDataset,
    LargeTrainDataset,
    LargeValidDataset,
    NoseKeypointTrainDataset,
    NoseKeypointValidDataset,
    part1_augment,
    part2_augment,
    part3_augment,
    save_kaggle,
)
from display import show_filters_part2, show_keypoints, show_progress
from learn import test, train, train_and_validate, validate


DATA_DIR = Path("data")
DANES_ROOT = DATA_DIR / Path("imm_face_db")
IBUG_ROOT = DATA_DIR / Path("ibug_300W_large_face_landmark_dataset")
train_xml = IBUG_ROOT / Path("labels_ibug_300W_train.xml")
test_xml = IBUG_ROOT / Path("labels_ibug_300W_test_parsed.xml")

assert DATA_DIR.exists()
assert DANES_ROOT.exists()
assert IBUG_ROOT.exists()


get_ipython().run_cell_magic("capture", "", """if not IBUG_ROOT.exists():
    !wget https://people.eecs.berkeley.edu/~zhecao/ibug_300W_large_face_landmark_dataset.zip
    !unzip 'data ibug_300W_large_face_landmark_dataset.zip'    
    !rm -r 'ibug_300W_large_face_landmark_dataset.zip'
    !mv ibug_300W_large_face_landmark_dataset data""")


training_set1 = NoseKeypointTrainDataset()
validation_set1 = NoseKeypointValidDataset()


assert len(training_set1) == 192, len(training_set1)
assert len(validation_set1) == 48, len(validation_set1)


batch_size = 64
train_loader1 = DataLoader(training_set1, batch_size=batch_size, shuffle=True)
valid_loader1 = DataLoader(validation_set1, batch_size=batch_size, shuffle=False)


show_keypoints(training_set1[2][0], training_set1[2][1])
show_keypoints(training_set1[134][0], training_set1[134][1])


# Training and Testing

model1 = cnn.NoseFinder()
epochs = 20
learn_rate = 3e-4
show_every = 1

train_loss1, valid_loss1 = train_and_validate(
    train_loader1, valid_loader1, model1, epochs, learn_rate, show_every
)


# Plot training and validation loss progress
show_progress(train_loss1)
show_progress(valid_loss1)


results1, _ = validate(valid_loader1, model1, show_every=None)


valid_imgs1, valid_keypts1, valid_preds1 = (
    torch.stack(results1[0]),
    torch.stack(results1[1]),
    torch.stack(results1[2]),
)

for i in range(len(validation_set1)):
    show_keypoints(
        image=valid_imgs1[i],
        truth_points=valid_keypts1[i],
        pred_points=valid_preds1[i],
    )


valid_keypts1.shape


# TODO pick success and failure cases to display


torch.save(model1,'model1.pt')


training_set2 = FaceKeypointsTrainDataset()
validation_set2 = FaceKeypointsValidDataset()

assert len(training_set2) == 192
assert len(validation_set2) == 48
# Initialize Dataloaders

batch_size = 64
train_loader2 = DataLoader(training_set2, batch_size=batch_size, shuffle=True)
valid_loader2 = DataLoader(validation_set2, batch_size=batch_size, shuffle=False)


# Plotting a few input images and their face keypoints.

sample = training_set2[3]
image, points = sample
show_keypoints(image, points)


# Training and Testing

model2 = cnn.FaceFinder()
epochs = 50
learn_rate = 3e-4
show_every = 3

train_loss2, valid_loss2 = train_and_validate(
    train_loader2, valid_loader2, model2, epochs, learn_rate, show_every
)


# Plot training and validation loss progress
show_progress(train_loss2)
show_progress(valid_loss2)


results2, _ = validate(valid_loader2, model2, show_every=None)


# TODO pick success and failure cases to display


valid_imgs2, valid_keypts2, valid_preds2 = (
    torch.stack(results2[0]),
    torch.stack(results2[1]),
    torch.stack(results2[2]),
)

for i in random.sample(range(len(validation_set2)), k=10):
    show_keypoints(
        image=validation_set2[i],
        truth_points=valid_keypts2[i],
        pred_points=valid_preds2[i],
    )


figs = show_filters_part2(model2)


torch.save(model2,'model2.pt')


training_set3 = LargeTrainDataset()
validation_set3 = LargeValidDataset()
test_set3 = LargeTestDataset()

# Initialize Dataloaders
batch_size = 512
train_loader3 = DataLoader(training_set3, batch_size=batch_size, shuffle=True)
valid_loader3 = DataLoader(validation_set3, batch_size=batch_size, shuffle=False)
test_loader3 = DataLoader(test_set3, batch_size=batch_size, shuffle=False)


show_keypoints(training_set3[2][0], training_set3[2][1])
show_keypoints(training_set3[134][0], training_set3[134][1])


# Training and Testing
model3 = cnn.ResNet()
epochs = 5
learn_rate = 3e-4
show_every = 1

train_loss3, valid_loss3 = train_and_validate(
    train_loader3, valid_loader3, model3, epochs, learn_rate, show_every
)


# Plot training and validation loss progress
show_progress(train_loss3)
show_progress(valid_loss3)


results3 = test(test_loader3, model3)


test_imgs3, test_preds3 = torch.stack(results3[0]), torch.stack(results3[1])

for i in random.sample(range(1008), k=6):
    show_keypoints(image=test_imgs3[i], truth_points=None, pred_points=test_preds3[i])


torch.save(model3,'model3.pt')


original_pts = []
for pts, xml_sample in zip(test_preds3, test_loader3.samples):
    original_pts.apppend(xml_sample.get_original_pts(pts))
save_kaggle(original_pts)
