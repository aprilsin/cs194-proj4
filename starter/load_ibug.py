import os
import xml.etree.ElementTree as ET

import numpy as np


tree = ET.parse("ibug_300W_large_face_landmark_dataset/labels_ibug_300W_train.xml")
root = tree.getroot()
root_dir = "ibug_300W_large_face_landmark_dataset"

bboxes = []  # face bounding box used to crop the image
landmarks = []  # the facial keypoints/landmarks for the whole training dataset
img_filenames = []  # the image names for the whole dataset

for filename in root[2]:
    img_filenames.append(os.path.join(root_dir, filename.attrib["file"]))
    box = filename[0].attrib
    # x, y for the top left corner of the box, w, h for box width and height
    bboxes.append([box["left"], box["top"], box["width"], box["height"]])

    landmark = []
    for num in range(68):
        x_coordinate = int(filename[0][num].attrib["x"])
        y_coordinate = int(filename[0][num].attrib["y"])
        landmark.append([x_coordinate, y_coordinate])
    landmarks.append(landmark)

landmarks = np.array(landmarks).astype("float32")
bboxes = np.array(bboxes).astype("float32")
