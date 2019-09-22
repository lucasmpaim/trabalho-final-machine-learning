import os

import cv2
import numpy as np
import pandas as pd
from keras.applications.inception_v3 import preprocess_input
from keras.applications.xception import Xception
from keras.preprocessing import image

from utils.base_utils import get_classes

model = Xception(include_top=False, weights='imagenet', pooling='avg')


def extract_deep(base_location='base',
                 output_dir='output/training/deep'):

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

    print(features.shape)

    df = pd.DataFrame(features)
    df.to_csv(f'{output_dir}/features.csv',
              header=False, index=False, sep=';')

    df = pd.DataFrame(Y)
    df.to_csv(f'{output_dir}/Y.csv',
              header=False, index=False, sep=';')


def extract_class_features(class_dir):
    images = np.ravel([img[2] for img in os.walk(class_dir)])
    return [
        deep_extract_features_from_image(f'{class_dir}/{img}')
        for img in images
    ]


def deep_extract_features_from_image(img_dir):
    print(img_dir)
    img = cv2.imread(img_dir)
    img = cv2.resize(img, (299, 299))
    xd = image.img_to_array(img)
    xd = np.expand_dims(xd, axis=0)
    xd = preprocess_input(xd)
    return np.append([img_dir], model.predict(xd))
