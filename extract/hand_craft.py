import collections

import cv2
from skimage import feature
import numpy as np
import os
import pandas as pd

from utils.base_utils import get_classes

Histogram = collections.namedtuple('ColorHistogram', 'r g b')
CVImage = collections.namedtuple('CVImage', 'rgb grey')
ImageFeatures = collections.namedtuple('ImageFeatures', 'X Y')


def extract_hand_craft(base_location='base',
                       output_dir='output/training/hand_craft'):

    classes = get_classes(base_location)
    features = []
    Y = []

    for cl in classes:
        cl_features = extract_class_features(cl.dir)
        features.append(cl_features)
        Y.append([cl.Y] * len(cl_features))

    Y = np.ravel(Y)
    features = np.array(features)

    shape = features.shape
    features = features.reshape((shape[0] * shape[1], shape[2]))

    df = pd.DataFrame(features)
    df.to_csv(f'{output_dir}/features.csv',
              header=False, index=False, sep=';')

    df = pd.DataFrame(Y)
    df.to_csv(f'{output_dir}/Y.csv',
              header=False, index=False, sep=';')


def extract_class_features(class_dir):
    images = np.ravel([img[2] for img in os.walk(class_dir)])
    return [
        extract_image_features(f'{class_dir}/{img}')
        for img in images
    ]


def extract_image_features(image_dir):
    img = cv2.imread(image_dir)
    grey, rgb = convert_image(img)
    color_histogram = calculate_color_histogram(rgb)
    lbp = calculate_lbp(grey)
    hog = calculate_hog(grey)

    features = [image_dir, lbp, hog, color_histogram.r, color_histogram.g, color_histogram.b]
    X_aux = []
    for aux in features:
        X_aux = np.append(X_aux, np.ravel(aux))
    return X_aux


def convert_image(img):
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return CVImage(grey, rgb)


def calculate_color_histogram(img):
    height, width, _ = img.shape

    r_histogram = cv2.calcHist([img], [0], None, [256], [0, 256]) / (height * width)
    g_histogram = cv2.calcHist([img], [1], None, [256], [0, 256]) / (height * width)
    b_histogram = cv2.calcHist([img], [2], None, [256], [0, 256]) / (height * width)

    return Histogram(r=r_histogram, g=g_histogram, b=b_histogram)


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
