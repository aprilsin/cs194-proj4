# This file is for loading images and keypoints customized for the Danes dataset.
# data source: http://www2.imm.dtu.dk/~aam/datasets/datasets.html.

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from skimage import io, transform
from skimage.util import img_as_float

ROOT_DIR = "./imm_face_db/"
	
def get_gender(person_idx):
    assert 1 <= person_idx <= 40, person_idx
    female_idx = [8, 12, 14, 15, 22, 30, 35]
    if person_idx in female_idx:
        return "f"
    return "m"

def load_points_from_asf(file) -> np.ndarray:
    lines_read = file.readlines()
    
    num_pts = int(lines_read[9])
    assert num_pts == 58, num_pts
    
    lines = lines_read[16:num_pts+16] # basically should be [16, 74]
    points = []
    IMG_WIDTH = 640
    IMG_HEIGHT = 480
    for line in lines:
        data = line.split("\t")
        x, y = float(data[2]) * IMG_WIDTH, float(data[3]) * IMG_HEIGHT
        points.append([x, y])
    
    points = np.array(points).astype(np.float32)
    print("row:", points[:, 0].min(), points[:, 0].max())
    print("col:", points[:, 1].min(), points[:, 1].max())
    assert len(points) == num_pts, len(points)
    return points

# Dataloader for Part 1
def load_nosepoint(person_idx, viewpt_idx):
    """ input person_idx and viewpt_idx should be zero indexed"""
    person_idx += 1
    viewpt_idx += 1
    assert 1 <= person_idx <= 40, person_idx
    assert 1 <= viewpt_idx <= 6, viewpt_idx
    
    # load all facial keypoints/keypoints
    gender = get_gender(person_idx)
    file = open(ROOT_DIR + "{:02d}-{:d}{}.asf".format(person_idx, viewpt_idx, gender))
    landmark = load_points_from_asf(file)

    # the nose keypoint
    NOSE_INDEX = 53
    nose_keypoint = landmark[NOSE_INDEX]
    
    print("input nose keypoint = ", nose_keypoint)
    
    return np.array([nose_keypoint])

def read_img_part1(person_idx, viewpt_idx):
    """ input person_idx and viewpt_idx should be zero indexed"""
    person_idx += 1
    viewpt_idx += 1
    assert 1 <= person_idx <= 40, person_idx
    assert 1 <= viewpt_idx <= 6, viewpt_idx
    gender = get_gender(person_idx)
    file_name = ROOT_DIR + "{:02d}-{:d}{}.jpg".format(person_idx, viewpt_idx, gender)
    img = io.imread(file_name, as_gray=True)
    img = img_as_float(img)
    img = img - 0.5
    assert img.max() <= 0.5 and img.min() >= -0.5, (img.min(), img.max())
    return img

#
# Data Fix / Augmentation for Part 1
#
class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, keypoints = sample['image'], sample['keypoints']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))

        # h and w are swapped for keypoints because for images,
        # x and y axes are axis 1 and 0 respectively
        keypoints = keypoints * [new_w / w, new_h / h]

        return {'image': img, 'keypoints': keypoints}


class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, keypoints = sample['image'], sample['keypoints']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                      left: left + new_w]

        keypoints = keypoints - [left, top]

        return {'image': image, 'keypoints': keypoints}

class CenterCrop(object):
    """Crop center of the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, keypoints = sample['image'], sample['keypoints']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size
        
        ## FIXME
        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                      left: left + new_w]

        keypoints = keypoints - [left, top]

        return {'image': image, 'keypoints': keypoints}
    
class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, keypoints = sample['image'], sample['keypoints']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'keypoints': torch.from_numpy(keypoints)}

class NoseKeypointDataset(Dataset):
    """Nose Keypoint dataset."""

    def __init__(self, person_idx, viewpt_idx, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images and annotations.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """     
        self.root_dir = ROOT_DIR
        self.data = {} # a dictionary in the form {(peron, viewpoint): (image, keypoint)}
        self.transform = transform
        
        for p in person_idx:
            for v in viewpt_idx:
                image = read_img_part1(p, v)
                nosepoint = load_nosepoint(p, v) # point loaded from asf file is in range [0, 1]
                key = (p, v)
                self.data[key] = (image, nosepoint)

    def __len__(self):
        return len(self.keypoints)

    def __getitem__(self, idx):
        if type(idx) == int:
            key = list(self.data.keys())[idx]
            image, keypoints = self.data[key]
        else:
            if torch.is_tensor(idx):
                idx = idx.tolist()
            image = []
            keypoints = []
            for i in idx:
                key = self.data.keys()[i]
                img, pts = self.data[key]
                image.append(img)
                keypoints.append(pts)

        sample = {'image': image, 'keypoints': keypoints}

        if self.transform:
            sample = self.transform(sample)
            
        return sample
    

# Dataloader for Part 2
def load_keypoints(person_idx, viewpt_idx):
    """ input person_idx and viewpt_idx should be zero indexed"""
    person_idx += 1
    viewpt_idx += 1
    assert 1 <= person_idx <= 40, person_idx
    assert 1 <= viewpt_idx <= 6, viewpt_idx
    # load all facial keypoints/keypoints
    gender = get_gender(person_idx)
    file = open(ROOT_DIR + "{:02d}-{:d}{}.asf".format(person_idx, viewpt_idx, gender))
    keypoints = load_points_from_asf(file)
    return keypoints

class FaceKeypointsDataset(Dataset):
    """All Face Keypoints dataset."""

    def __init__(self, person_idx, viewpt_idx, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images and annotations.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """     
        self.root_dir = ROOT_DIR
        self.data = {} # a dictionary in the form {(peron, viewpoint): (image, keypoint)}
        self.transform = transform
        
        for p in person_idx:
            for v in viewpt_idx:
                image = read_img_part1(p, v)
                keypoints = load_keypoints(p, v) # point loaded from asf file is in range [0, 1]
#                 print("input nose point is: ", nosepoint)
                key = (p, v)
                self.data[key] = (image, keypoints)

    def __len__(self):
        return len(self.keypoints)

    def __getitem__(self, idx):
        if type(idx) == int:
            key = list(self.data.keys())[idx]
            image, keypoints = self.data[key]
        else:
            if torch.is_tensor(idx):
                idx = idx.tolist()
            image = []
            keypoints = []
            for i in idx:
                key = self.data.keys()[i]
                img, pts = self.data[key]
                image.append(img)
                keypoints.append(pts)

        sample = {'image': image, 'keypoints': keypoints}

        if self.transform:
            sample = self.transform(sample)
            
        return sample