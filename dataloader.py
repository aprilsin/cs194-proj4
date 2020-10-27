# This file is for loading images and keypoints of dataset.
import numpy as np

# Dataloader for Part 1
def load_nosepoint(person_idx, viewpt_idx, gender):
    root_dir = "./imm_face_db/"

    # load all facial keypoints/landmarks
    file = open(root_dir + "{:02d}-{:d}{}.asf".format(person_idx, viewpt_idx, gender))
    points = file.readlines()[16:74]
    landmark = []
    for point in points:
        x, y = point.split("\t")[2:4]
        landmark.append([float(x), float(y)])

    # the nose keypoint
    nose_keypoint = np.array(landmark).astype("float32")[-5]
    return nose_keypoint

# Dataloader for Part 2
def load_keypoints(person_idx, viewpt_idx, gender):
    root_dir = "./imm_face_db/"

    # load all facial keypoints/landmarks
    file = open(root_dir + "{:02d}-{:d}{}.asf".format(person_idx, viewpt_idx, gender))
    points = file.readlines()[16:74]
    landmark = []

    for point in points:
        x, y = point.split("\t")[2:4]
        landmark.append([float(x), float(y)])

    # the nose keypoint
    keypoints = np.array(landmark).astype("float32")
    return keypoints