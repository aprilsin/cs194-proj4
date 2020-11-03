""" Global Variables """
import torch

NUM_POINTS = 41
NUM_CHANNELS = 3
NUM_FRAMES = 20

DEFAULT_HEIGHT = 575
DEFAULT_WIDTH = 547
DEFAULT_EYE_LEN = DEFAULT_WIDTH * 0.25

PAD_MODE = "edge"

DANES_HEIGHT = 480
DANES_WIDTH = 640

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"