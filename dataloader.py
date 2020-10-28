# This file is for loading images and keypoints customized for the Danes dataset.
# data source: http://www2.imm.dtu.dk/~aam/datasets/datasets.html.

import numpy as np
import my_types

root_dir = "./imm_face_db/"
	
def get_gender(person_idx):
	assert person_idx in np.arange(1, 40+1)
	female_idx = [8, 12, 14, 15, 22, 30, 35]
	if person_idx in female_idx:
		return "f"
	return "m"

# Dataloader for Part 1
def load_nosepoint(person_idx, viewpt_idx):
    # load all facial keypoints/landmarks
    gender = get_gender(person_idx)
    file = open(root_dir + "{:02d}-{:d}{}.asf".format(person_idx, viewpt_idx, gender))
    points = file.readlines()[16:74]
    landmark = []
    for point in points:
        x, y = point.split("\t")[2:4]
        landmark.append([float(x), float(y)])

    # the nose keypoint
    nose_keypoint = np.array(landmark).astype("float32")[-5]
    return nose_keypoint

def read_img_part1(person_idx, viewpt_idx):
    gender = get_gender(person_idx)
    file_name = root_dir + "{:02d}-{:d}{}.jpg".format(person_idx, viewpt_idx, gender)
    return my_types.to_img_arr(file_name, as_gray=True)

# Dataloader for Part 2
def load_keypoints(person_idx, viewpt_idx):
    # load all facial keypoints/landmarks
    gender = get_gender(person_idx)
    file = open(root_dir + "{:02d}-{:d}{}.asf".format(person_idx, viewpt_idx, gender))
    points = file.readlines()[16:74]
    landmark = []

    for point in points:
        x, y = point.split("\t")[2:4]
        landmark.append([float(x), float(y)])

    # the nose keypoint
    keypoints = np.array(landmark).astype("float32")
    return keypoints