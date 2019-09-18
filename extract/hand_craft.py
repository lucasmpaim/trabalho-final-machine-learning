import cv2
from skimage import feature
import numpy as np


# img = cv2.imread(filename)
# height, width, _ = img.shape


def convert_images(img):
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return grey, rgb


def calculate_color_histogram(img):
    height, width, _ = img.shape

    r_histogram = cv2.calcHist([img], [0], None, [256], [0, 256]) / (height * width)
    g_histogram = cv2.calcHist([img], [1], None, [256], [0, 256]) / (height * width)
    b_histogram = cv2.calcHist([img], [2], None, [256], [0, 256]) / (height * width)

    return r_histogram, g_histogram, b_histogram


def calculate_lbp(img):
    lbp = feature.local_binary_pattern(img, 59, 1, method="uniform")
    (lbp_histogram, _) = np.histogram(lbp.ravel(), bins=59, range=(0, 59))
    lbp_histogram = lbp_histogram.astype("float")
    lbp_histogram /= (lbp_histogram.sum())
    return lbp_histogram


def calculate_hog(img):
    return feature.hog(img, orientations=8,
                       pixels_per_cell=(32, 32),
                       cells_per_block=(8, 8),
                       block_norm='L2-Hys')
