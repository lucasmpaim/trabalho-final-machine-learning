import collections

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.manifold.t_sne import TSNE

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
        X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=.2,
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

            current_error_on_training = 1 - classifier.score(X_train, np.ravel(y_train))
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

            if current_epoch >= 1:
                if smooth_error_validation[-2] - smooth_error_validation[-1] < 1e-03:
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

    def visualizate_data():
        x_tsne = TSNE(n_components=2).fit_transform(X)

        y_aux = pd.DataFrame()
        y_aux['classes'] = y[0]

        tsne_df = pd.DataFrame()
        tsne_df['tsne-x'] = x_tsne[:, 0]
        tsne_df['tsne-y'] = x_tsne[:, 1]
        tsne_df = pd.concat([tsne_df, y_aux], axis=1)

        plt.figure(figsize=(16, 10))
        sns.scatterplot('tsne-x', 'tsne-y',
                        hue="classes",
                        legend='full',
                        palette=sns.color_palette("hls", 10),
                        alpha=0.3,
                        data=tsne_df)
        plt.show()

    if hasattr(classifier, 'partial_fit'):
        overfitting_prevent_train(X_train, y_train)
    else:
        normal_fit()

    predicated_rows = classifier.predict(X_test)
    predicated_proba = classifier.predict_proba(X_test)

    # aux for find proba for class
    aux = 0
    for predicated, expected in zip(predicated_rows, y_test.iterrows()):
        if expected[1][0] != predicated:
            img_dir = images_dirs[expected[0]]
            percent = predicated_proba[aux][np.where(all_classes == predicated)]
            print(f'Confundiu {img_dir} com {predicated}, proba: {percent}')
        aux += 1

    print(f'Acurácia {classifier_name} Teste : {classifier.score(X_test, y_test)}')

    cm = confusion_matrix(y_test, predicated_rows)
    df_cm = pd.DataFrame(cm, index=all_classes, columns=all_classes)
    sns.heatmap(df_cm, annot=True)
    plt.title(f'Confusion Matrix --> {classifier_name} --> {base_dir}')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

    # visualizate_data()

    if isinstance(classifier, DecisionTreeClassifier):
        from sklearn.externals.six import StringIO
        from sklearn.tree import export_graphviz
        import pydot

        dot_data = StringIO()
        export_graphviz(classifier, out_file=dot_data, rounded=True,
                        filled=True)
        graph = pydot.graph_from_dot_data(dot_data.getvalue())[0]
        graph.write_pdf(f'{classifier_name}.pdf')

    return classifier, X_test, y_test
