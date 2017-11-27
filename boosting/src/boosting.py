import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import clone


class AdaBoostBinaryClassifier(object):
    '''
    INPUT:
    - n_estimator (int)
      * The number of estimators to use in boosting
      * Default: 50

    - learning_rate (float)
      * Determines how fast the error would shrink
      * Lower learning rate means more accurate decision boundary,
        but slower to converge
      * Default: 1
    '''

    def __init__(self,
                 n_estimators=50,
                 learning_rate=1):

        self.base_estimator = DecisionTreeClassifier(max_depth=1)
        self.n_estimator = n_estimators
        self.learning_rate = learning_rate

        # Will be filled-in in the fit() step
        self.estimators_ = []
        self.estimator_weight_ = np.zeros(self.n_estimator, dtype=np.float)

    def fit(self, X, y):
        '''
        INPUT:
        - x: 2d numpy array, feature matrix
        - y: numpy array, labels

        Build the estimators for the AdaBoost estimator.
        '''
        '''You should have a for loop that calls your _boost method n_estimators
        times. Make sure to save all the estimators in self.estimators_.
        You also need to save all the estimator weights in self.estimator_weight_.
        '''

        sample_weight = np.ones(len(y))
        for n in range(self.n_estimator):
            est, sample_weight, estimator_weight = self._boost(X, y, sample_weight)
            self.estimators_.append(est)
            self.estimator_weight_[n] = estimator_weight


    def _boost(self, X, y, sample_weight):
        '''
        INPUT:
        - x: 2d numpy array, feature matrix
        - y: numpy array, labels
        - sample_weight: numpy array

        OUTPUT:
        - estimator: DecisionTreeClassifier
        - sample_weight: numpy array (updated weights)
        - estimator_weight: float (weight of estimator)

        Go through one iteration of the AdaBoost algorithm. Build one estimator.
        '''

        estimator = clone(self.base_estimator)
        estimator.fit(X, y, sample_weight=sample_weight)
        estimator_error = sample_weight[estimator.predict(X) != y].sum() / sample_weight.sum()
        estimator_weight = np.log10((1 - estimator_error) / estimator_error)
        new_sample_weight = sample_weight * np.exp(estimator_weight * (estimator.predict(X) != y))
        return estimator, new_sample_weight, estimator_weight

    def predict(self, X):
        '''
        INPUT:
        - x: 2d numpy array, feature matrix

        OUTPUT:
        - labels: numpy array of predictions (0 or 1)
        '''
        '''
        Implement the predict method. This is step 3 from the algorithm.
        Note that the algorithm considers the predictions to be either -1 or 1.
        So once you get predictions back from your Decision Trees, change the 0's
        to -1's.'''
        prediction = [estimator.predict(X)*2-1 for estimator in self.estimators_]
        prod = zip(self.estimator_weight_, prediction)
        G = np.sign(sum((a*b) for a, b in prod))
        return (G > 0).astype(int)

    def score(self, X, y):
        '''
        INPUT:
        - x: 2d numpy array, feature matrix
        - y: numpy array, labels

        OUTPUT:
        - score: float (accuracy score between 0 and 1)
        '''
        prediction = self.predict(X)
        correct = prediction == y.astype(int)
        return correct.sum() / float(len(y.astype(int)))
