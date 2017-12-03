'''
class kNN to predict the class of a point x:
    for every point in the dataset:
        1. calculate the distance between the point and x
        2. take the k points with the smallest distances to x
        (**hint: use numpy's argsort() function**)

    return the majority class among these items

Distance:
 - euclidean_distance
 - cosine_distance

KNearestNeighbors class has three methods:
 - fit
 - predict
 - accuracy score
'''
import numpy as np
import pandas as pd
from collections import Counter
from itertools import izip
from sklearn.datasets import make_classification
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import sys
from itertools import combinations

def plot_decision_boundary(clf, X, y, n_classes):
    """Plot the decision boundary of a kNN classifier.
    Plots decision boundary for up to 4 classes.
    Colors have been specifically chosen to be color blindness friendly.
    Assumes classifier, clf, has a .predict() method that follows the
    sci-kit learn functionality.
    X must contain only 2 continuous features.
    Function modeled on sci-kit learn example.
    Parameters
    ----------
    clf: instance of classifier object
        A fitted classifier with a .predict() method.
    X: numpy array, shape = [n_samples, n_features]
        Test data.
    y: numpy array, shape = [n_samples,]
        Target labels.
    n_classes: int
        The number of classes in the target labels.
    """
    mesh_step_size = .2

    # Colors are in the order [red, yellow, blue, cyan]
    cmap_light = ListedColormap(['#FFAAAA', '#FFFFAA', '#AAAAFF', '#AAFFFF'])
    cmap_bold = ListedColormap(['#FF0000', '#FFFF00', '#0000FF', '#00FFFF'])

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, m_max]x[y_min, y_max].
    feature_1 = X[:, 0]
    feature_2 = X[:, 1]
    x_min, x_max = feature_1.min() - 1, feature_1.max() + 1
    y_min, y_max = feature_2.min() - 1, feature_2.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, mesh_step_size),
                         np.arange(y_min, y_max, mesh_step_size))
    dec_boundary = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    dec_boundary = dec_boundary.reshape(xx.shape)
    plt.figure()
    plt.pcolormesh(xx, yy, dec_boundary, cmap=cmap_light)

    # Plot also the training points
    plt.scatter(feature_1, feature_2, c=y, cmap=cmap_bold)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())

    plt.title(
              "{0}-Class classification (k = {1}, metric = '{2}')"
              .format(n_classes, clf.k, clf.distance))
    plt.show()

def euclidean_distance(arr1, arr2):
    """Compute the euclidean_distance between two numpy arrays.
    Parameters
    ----------
    a: numpy array
    b: numpy array
    Returns
    -------
    numpy array
    """
    diff = arr1 - arr2
    return np.sqrt(diff.dot(diff))

def cosine_distance(arr1, arr2):
    """Compute the cosine_distance between two numpy arrays.
    Parameters
    ----------
    a: numpy array
    b: numpy array
    Returns
    -------
    """
    num = arr1.dot(arr2)
    denom = np.sqrt(arr1.dot(arr1) * arr2.dot(arr2))
    return 1 - num / float(denom)

class KNearestNeighbors(object):
    def __init__(self, k=5, distance=euclidean_distance):
        ''' k: number of points with the smallest distances to each data point
            distance: Euclidean distance or cosine distance
            Initialize a KNearestNeighbors object.
        '''
        self.k = k
        self.distance = distance

    def fit(self, X, y):
        """Fit the model using X as training data and y as target labels.
        According to kNN algorithm, the training data is simply stored.
        Parameters
        ----------
        X: numpy array, shape = [n_samples, n_features]
            Training data.
        y: numpy array, shape = [n_samples,]
            Target labels.
        Returns
        -------
        None
        """
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        num_train_rows, num_train_cols = self.X_train.shape
        num_X_rows, _ = X.shape
        X = X.reshape((-1, num_train_cols))
        distances = np.zeros((num_X_rows, num_train_rows))
        for i, x in enumerate(X):
            for j, x_train in enumerate(self.X_train):
                distances[i, j] = self.distance(x_train, x)
        # Sort and take top k
        top_k = self.y_train[distances.argsort()[:, :self.k]]
        result = np.zeros(num_X_rows)
        for i, values in enumerate(top_k):
            top_voted_label, _ = Counter(values).most_common(1)[0]
            result[i] = top_voted_label
        return result

    def score(self, X, y_true):
        """Return the mean accuracy on the given data and true labels.
        Parameters
        ----------
        X: numpy array, shape = [n_samples, n_features]
            Test data.
        y_true: numpy array, shape = [n_samples,]
            True labels for given test data, X.
        Returns
        -------
        score: float
            Mean accuracy of self.predict(X) given true labels, y_true.
        """
        y_pred = self.predict(X)
        score = y_true == y_pred
        return np.average(score)


if __name__ == '__main__':
    X, y = make_classification(n_classes=3, n_features=2, n_redundant=0,
                               n_informative=2, n_clusters_per_class=1,
                               class_sep=1, random_state=5)
    print y.shape

    knn = KNearestNeighbors(4, cosine_distance)
    knn.fit(X, y)
    print "This is the accuracy: {}".format(knn.score(X, y))
    print "\tactual\tpredict\tcorrect?"
    for i, (actual, predicted) in enumerate(izip(y, knn.predict(X))):
        print "%d\t%d\t%d\t%d" % (i,
                                  actual,
                                  predicted,
                                  int(actual == predicted))
    print (actual - predicted).sum()
    # This loop plots the decision boundaries for different decision metrics
    for metric in [euclidean_distance, cosine_distance]:
        # we create an instance of Neighbours Classifier and fit the data.
        clf = KNearestNeighbors(k=3, distance=metric)
        clf.fit(X, y)
        plot_decision_boundary(clf, X, y, n_classes=3)
