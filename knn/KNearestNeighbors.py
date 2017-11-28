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
from itertools import combinations

class kNN(object):
    def __init__(self, k, distance):
        ''' k: number of points with the smallest distances to each data point
            distance: Euclidean distance or cosine distance
        '''
        self.k = k
        self.distance = distance

    def _euclidean_distance(self, arr1, arr2):
        diff = arr1 - arr2
        return np.sqrt(diff.dot(diff))

    def _cosine_distance(self, arr1, arr2):
        num = arr1.dot(arr2)
        denom = np.sqrt(arr1.dot(arr1) * arr2.got(arr2))
        return 1 - num / float(denom)

    def fit(self, X, y):
        '''     X:  two dimensional numpy array representing feature matrix
                for test data
                y:  numpy array representing labels for test data'''

        distance = []
        index_set = set()
        for (idx1, arr1), (idx2,arr2) in combinations(enumerate(X), 2):
            if self.distance == 'euclidean_distance':
                lst = [idx1, idx2, self._euclidean_distance(arr1, arr2)]
            elif self.distance == 'cosine_distance':
                lst = [idx1, idx2, self._cosine_distance(arr1, arr2)]
            distance.append(lst)
            index_set.add(idx1)
            index_set.add(idx2)

        vote_X = []
        for ind in index_set:
            vote_id1 = []
            for other_pair in distance:
                if ind == other_pair[0]:
                    vote_id1.append(other_pair[1:])
            new_vote = sorted(vote_id1, key=lambda x: x[1])[:((self.k)+1)]
            tot_votes = 0
            for vote in new_vote:
                tot_votes += y[vote[0]]
            if tot_votes >= self.k * 0.5:
                response = 1
            else:
                response = 0
            vote_X.append([X[ind], response])
        return vote_X



    def predict(self, X, y):
        pass

    def score(self, X, y):
        pass








if __name__ == '__main__':
    '''making fake data'''
    from sklearn.datasets import make_classification

    X, y = make_classification(n_features=4, n_redundant=0, n_informative=1,
                               n_clusters_per_class=1, class_sep=5, random_state=5)

    '''implement class KNearestNeighbors'''
    from KNearestNeighbors import kNN

    knn = kNN(k=3, distance='euclidean_distance')
    pred = knn.fit(X, y)
    output = [a[1] for a in pred]
    print output
    print sum(output - y)
    #print len(pred), len(X), len(y)
    #y_predict = knn.predict(X_new)
