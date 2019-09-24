import collections

import cv2
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

GraphicData = collections.namedtuple('GraphicData', 'x error_on_val error_on_train')


def chunk_generator(dataframe, chunk_size):
    l = dataframe.shape[0]
    for ndx in range(0, l, chunk_size):
        yield dataframe.iloc[ndx:ndx + chunk_size, :-1], dataframe.iloc[ndx:ndx + chunk_size, -1:]


def train_holdout(base_dir, classifier_name='Naive Bayes', classifier=GaussianNB()):
    base = pd.read_csv(f'{base_dir}/features.csv', sep=';', header=None)
    # fetch folder from first index of base
    images_dirs = base.iloc[:, 0]
    # Separate X to a new DataFrame and convert to numpy array
    X = base.iloc[:, 1:]

    # Load classes names
    y = pd.read_csv(f'{base_dir}/Y.csv', sep=';', header=None)
    # HOLDOUT
    # separate the base in 70% to train and 30% to test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3,
                                                        random_state=42, stratify=y)

    # separate the train base in 70% to train and 30% do validation
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=.3,
                                                                    random_state=42, stratify=y_train)

    chunk_size = 10
    error_on_validation = 0.0
    error_on_training = 0.0

    graphic_data = []
    current_epoch = 0

    all_classes = np.unique(y.to_numpy())

    for chunk in chunk_generator(pd.concat([X_train, y_train], axis=1), chunk_size=chunk_size):
        current_epoch += 1
        classifier.partial_fit(chunk[0], chunk[1], classes=all_classes)

        current_error_on_training = 1 - classifier.score(X_test, y_test)
        current_error_on_validation = 1 - classifier.score(X_validation, y_validation)

        graphic_data.append(
            GraphicData(x=current_epoch,
                        error_on_val=current_error_on_validation,
                        error_on_train=current_error_on_training)
        )

        if (current_error_on_training < error_on_training and
                current_error_on_validation > error_on_validation):
            # STOP! OVERFITTING
            print(f'Stop training {classifier_name}, overfitting detected')
            break

        error_on_training = current_error_on_training
        error_on_validation = current_error_on_validation

    predicated_rows = classifier.predict(X_test)
    for predicated, expected in zip(predicated_rows, y_test.iterrows()):
        if expected[1][0] != predicated:
            img_dir = images_dirs[expected[0]]
            print(f'Confundiu {img_dir} com {predicated}')

    cm = confusion_matrix(y_test, predicated_rows)
    df_cm = pd.DataFrame(cm, index=all_classes, columns=all_classes)
    sns.heatmap(df_cm, annot=True)
    plt.title(f'Confusion Matrix --> {classifier_name} --> {base_dir}')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

    plt.title(f'Error Compare --> {classifier_name} --> {base_dir}')
    plt.plot([data.x for data in graphic_data],
             [data.error_on_train for data in graphic_data], label='Error on Train')
    plt.plot([data.x for data in graphic_data],
             [data.error_on_val for data in graphic_data], label='Error on Validation')
    plt.ylabel('Error')
    plt.xlabel('Epoch')
    plt.legend()
    plt.show()
