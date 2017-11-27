from DecisionTree import DecisionTree
import numpy as np


class RandomForest(object):
    '''A Random Forest class'''

    def __init__(self, num_trees, num_features):
        '''
           num_trees:  number of trees to create in the forest:
        num_features:  the number of features to consider when choosing the
                           best split for each node of the decision trees
        '''
        self.num_trees = num_trees
        self.num_features = num_features
        self.forest = None

    def fit(self, X, y):
        '''
        X:  two dimensional numpy array representing feature matrix
                for test data
        y:  numpy array representing labels for test data
        '''
        self.forest = self.build_forest(X, y, self.num_trees, X.shape[0], \
                                        self.num_features)

    def build_forest(self, X, y, num_trees, num_samples, num_features):
        '''
        Return a list of num_trees DecisionTrees.

         Repeat num_trees times:
         Create a random sample of the data with replacement
         Build a decision tree with that sample
         Return the list of the decision trees created
        '''
        decisiontree_list = []
        for n in range(self.num_trees):
            bs_index = np.random.choice(X.shape[0], num_samples)
            bs_X = X[bs_index]
            bs_y = y[bs_index]
            tree = DecisionTree(self.num_features)
            tree.fit(bs_X, bs_y)
            decisiontree_list.append(tree)
        return decisiontree_list

    def predict(self, X):
        '''
        Return a numpy array of the labels predicted for the given test data.
        '''
        output = []
        for tree in self.build_forest:
            output.append(tree.predict(X))
        return output


    def score(self, X, y):
        '''
        Return the accuracy of the Random Forest for the given test data and
        labels.
        '''
        pass


'''
from RandomForest import RandomForest
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

df = pd.read_csv('../data/playgolf.csv')
y = df.pop('Result').values
X = df.values
X_train, X_test, y_train, y_test = train_test_split(X, y)

rf = RandomForest(num_trees=3, num_features=2)
rf.fit(X_train, y_train)
rf.build_forest(X_train, y_train, num_trees=3, num_samples=len(X_train), num_features=2)
y_predict = rf.predict(X_test)
print "predict", rf.predict(X_test)
print "score:", rf.score(X_test, y_test)
'''
