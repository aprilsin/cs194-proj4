import numpy as np
import matplotlib.pyplot as plt

def show_landmarks(image, landmarks):
    """Show image with landmarks"""
    plt.imshow(image, cmap='gray')
    plt.scatter(landmarks[:, 0], landmarks[:, 1], s=10, marker='.', c='r')
    plt.pause(0.001)  # pause a bit so that plots are updated

# plt.figure()
# show_landmarks(io.imread(os.path.join('data/faces/', img_name)),
#                landmarks)
# plt.show()