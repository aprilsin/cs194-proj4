# This file is for loading images and keypoints customized for the Danes dataset.
# data source: http://www2.imm.dtu.dk/~aam/datasets/datasets.html.
import torch
import numpy as np
import my_types
import skimage.transform

ROOT_DIR = "./imm_face_db/"
	
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
    file = open(ROOT_DIR + "{:02d}-{:d}{}.asf".format(person_idx, viewpt_idx, gender))
    points = file.readlines()[16:74]
    landmark = []
    for point in points:
        x, y = point.split("\t")[2:4]
        landmark.append([float(x), float(y)])
    # the nose keypoint
    NOSE_INDEX = -5
    nose_keypoint = torch.tensor(landmark,dtype=torch.float32)[NOSE_INDEX]
    return nose_keypoint

def data_augmentation_part1(img:np.ndarray, nose_point):
    my_types.assert_img_type(img)
    my_types.assert_is_point(nose_point)
    
    img = skimage.transform.imresize(img, output_shape=(80, 60), preserve_range=True)
    my_types.assert_img_type(img)
    img = img - 0.5 # normalize values to range [-0.5, 0.5]
    
#     my_types.assert_img_type(img)
    return img

def read_img_part1(person_idx, viewpt_idx):
    gender = get_gender(person_idx)
    file_name = ROOT_DIR + "{:02d}-{:d}{}.jpg".format(person_idx, viewpt_idx, gender)
    img = my_types.to_img_arr(file_name, as_gray=True)
#     augmented = data_augmentation(img)
#     return augmented
    return img

# Dataloader for Part 2
def load_keypoints(person_idx, viewpt_idx):
    # load all facial keypoints/landmarks
    gender = get_gender(person_idx)
    file = open(ROOT_DIR + "{:02d}-{:d}{}.asf".format(person_idx, viewpt_idx, gender))
    points = file.readlines()[16:74]
    landmark = []

    for point in points:
        x, y = point.split("\t")[2:4]
        landmark.append([float(x), float(y)])

    # the nose keypoint
    keypoints = np.array(landmark).astype("float32")
    return keypoints