from torch import Tensor


def assert_points(pts):
    assert isinstance(pts, Tensor), type(pts)
    assert pts.ndim == 2, pts.shape
    assert pts.shape[1] == 2, pts.shape

    # make sure that the keypoints are ratios
    rows = pts[:, 0]
    cols = pts[:, 1]
    # leave some wiggle room so use 1.5 instead of 1
    assert rows.max() <= 1.5 and cols.max() <= 1.5, f"{rows.max()}, {cols.max()}"

    return True


def assert_img(img):
    assert isinstance(img, Tensor), type(img)
    assert img.ndim == 3, img.shape
    # assert list(img.shape)[0] == 1, f"{img.shape} is not grayscale"
    assert all(x > 0 for x in list(img.shape)), img.shape
    return True
