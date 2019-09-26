from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from train.train import *

deep_base_dir = 'output/training/deep'
hand_craft_dir = 'output/training/hand_craft'
# Load and plot datasets
rng = np.random.RandomState(123)


def train_all_for(base_dir):
    with np.errstate(divide='ignore'):
        # train_holdout(base_dir, 'Naive Bayes', GaussianNB(var_smoothing=1e-09))
        # train_holdout(base_dir, 'Decision Tree', DecisionTreeClassifier(criterion='entropy'))
        # train_holdout(base_dir, 'KNN', KNeighborsClassifier())
        train_holdout(base_dir, 'Neural Network - adam',
                      MLPClassifier(solver='adam', hidden_layer_sizes=(100,),
                                    activation='logistic', batch_size=100,
                                    max_iter=1000000,
                                    learning_rate_init=0.1,
                                    momentum=0.2, tol=1e-10,
                                    random_state=rng))

        # # parameters for SVM
        # parameters = [
        #     {'C': [0.1, 0.5, 1, 10, 100, 500, 1000], 'kernel': ['poly']},
        #     {'C': [0.1, 0.5, 1, 10, 100, 500, 1000], 'gamma': [0.1, 0.001, 0.0001, 0.00001], 'kernel': ['rbf']},
        # ]
        # svm = SVC(gamma='scale')
        # svm = GridSearchCV(svm, parameters, scoring='accuracy', cv=10, iid=False)
        # train_holdout(base_dir, 'SVM', svm)


# train_all_for(deep_base_dir)
train_all_for(hand_craft_dir)
