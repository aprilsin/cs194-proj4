import numpy as np
import matplotlib.pyplot as plt
from my_types import *

# def plot_tri_mesh(img: np.ndarray, points: np.ndarray, triangulation) -> None:
#     """
#     Displays the triangular mesh of an image
#     """
#     plt.imshow(img)
#     plt.triplot(points[:, 0], points[:, 1], triangulation.simplices)
#     plt.plot(points[:, 0], points[:, 1], "o")
#     plt.show()

def plot_points(img: np.ndarray, indices: np.ndarray):
    assert_img_type(img)
    assert_indices(indices)
    
    plt.imshow(img, cmap='gray')
    plt.plot(points, "o")
    plt.show()