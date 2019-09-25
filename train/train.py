import collections

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

GraphicData = collections.namedtuple('GraphicData', 'x error_on_val error_on_train')


def chunk_generator(dataframe, chunk_size):
    l = dataframe.shape[0]
    for ndx in range(0, l, chunk_size):
        yield dataframe.iloc[ndx:ndx + chunk_size, :-1], dataframe.iloc[ndx:ndx + chunk_size, -1:]


def train_holdout(base_dir, classifier_name, classifier):
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

    all_classes = np.unique(y.to_numpy())

    def overfitting_prevent_train(X_train, y_train):
        # separate the train base in 70% to train and 30% do validation
        X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=.3,
                                                                        random_state=42, stratify=y_train)

        chunk_size = 10

        graphic_data = []
        current_epoch = 0

        alpha = 0.8
        smooth_error_training = []
        smooth_error_validation = []
        overfitting_count = 0

        for chunk in chunk_generator(pd.concat([X_train, y_train], axis=1), chunk_size=chunk_size):
            classifier.partial_fit(chunk[0], np.ravel(chunk[1]), classes=all_classes)

            current_error_on_training = 1 - classifier.score(X_test, np.ravel(y_test))
            current_error_on_validation = 1 - classifier.score(X_validation, np.ravel(y_validation))

            graphic_data.append(
                GraphicData(x=current_epoch,
                            error_on_val=current_error_on_validation,
                            error_on_train=current_error_on_training)
            )

            if current_epoch == 0:
                smooth_error_training.append(current_error_on_training)
                smooth_error_validation.append(current_error_on_validation)
            else:
                smooth_error_training.append(
                    alpha * smooth_error_training[current_epoch - 1] + (1 - alpha) * graphic_data[
                        current_epoch - 1].error_on_train)
                smooth_error_validation.append(
                    alpha * smooth_error_validation[current_epoch - 1] + (1 - alpha) * graphic_data[
                        current_epoch - 1].error_on_val)

            if current_epoch > 8:
                x1 = [data.x for data in graphic_data[-4:]]
                x2 = [data.x for data in graphic_data[-8:-4]]

                now = abs(np.trapz(smooth_error_training[-4:], x1) - np.trapz(smooth_error_validation[-4:], x1))
                before = abs(np.trapz(smooth_error_training[-8:-4], x2) - np.trapz(smooth_error_validation[-8:-4], x2))

                if now > before and abs(now - before) > 0.01:
                    overfitting_count += 1
                    if overfitting_count >= 5:
                        print(f'Overfitting Detected on {classifier_name}, epoch: {current_epoch}')
                        break
                else:
                    overfitting_count = 0

            current_epoch += 1

        print(f'Acurácia {classifier_name} Validação: {classifier.score(X_validation, np.ravel(y_validation))}')

        plt.title(f'Error Compare --> {classifier_name} --> {base_dir}')
        # plt.plot([data.x for data in graphic_data],
        #          [data.error_on_train for data in graphic_data], label='Error on Train')

        # plt.plot([data.x for data in graphic_data],
        #          [data.error_on_val for data in graphic_data], label='Error on Validation')

        plt.plot([data.x for data in graphic_data],
                 smooth_error_training, label='Error on Train Smooth')
        plt.plot([data.x for data in graphic_data],
                 smooth_error_validation, label='Error on Validation Smooth')

        plt.ylabel('Error')
        plt.xlabel('Epoch')
        plt.legend()
        plt.show()

    def normal_fit():
        classifier.fit(X_train, y_train)

    if hasattr(classifier, 'partial_fit'):
        overfitting_prevent_train(X_train, y_train)
    else:
        normal_fit()

    predicated_rows = classifier.predict(X_test)
    # for predicated, expected in zip(predicated_rows, y_test.iterrows()):
    #     if expected[1][0] != predicated:
    #         img_dir = images_dirs[expected[0]]
    #         print(f'Confundiu {img_dir} com {predicated}')

    print(f'Acurácia {classifier_name}: {classifier.score(X_test, y_test)}')

    cm = confusion_matrix(y_test, predicated_rows)
    df_cm = pd.DataFrame(cm, index=all_classes, columns=all_classes)
    sns.heatmap(df_cm, annot=True)
    plt.title(f'Confusion Matrix --> {classifier_name} --> {base_dir}')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
