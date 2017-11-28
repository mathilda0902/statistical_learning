'''making fake data'''
from sklearn.datasets import make_classification

X, y = make_classification(n_features=4, n_redundant=0, n_informative=1,
                           n_clusters_per_class=1, class_sep=5, random_state=5)

'''implement class KNearestNeighbors'''
from KNearestNeighbors import kNN

knn = KNearestNeighbors(k=3, distance=euclidean_distance)
knn.fit(X, y)


def euclidean_distance(arr1, arr2):
    diff = arr1 - arr2
    return np.sqrt(diff.dot(diff))

from itertools import combinations
def fit(self, X, y):
    '''     X:  two dimensional numpy array representing feature matrix
            for test data
            y:  numpy array representing labels for test data'''

    distance = []
    index_set = set()
    for (idx1, arr1), (idx2,arr2) in combinations(enumerate(X), 2):
        lst = [idx1, idx2, euclidean_distance(arr1, arr2)]
        distance.append(lst)
        index_set.add(idx1)
        index_set.add(idx2)



        vote_X = []
        for ind in index_set:
            vote_id1 = []
            for other_pair in distance:
                if ind == other_pair[0]:
#                    if other_pair[1:] not in vote_id1:
                    vote_id1.append(other_pair[1:])
            new_vote = sorted(vote_id1, key=lambda x: x[1])
            new_vote = new_vote[:(k+1)]
            tot_votes = 0
            for vote in new_vote:
                tot_votes += y[vote[0]]
            if tot_votes >= k * 0.5:
                response = 1
            else:
                response = 0
            vote_X.append([X[ind], response])
        print vote_X

'''
 [6, 7, 9.9670862109088674],
 [6, 8, 3.6912029491501208],
 [6, 9, 3.55331641620189],
 [7, 8, 10.617304253480958],
 [7, 9, 9.9656107612802796],
 [8, 9, 3.3924810065994215]]
 '''

    for vote in vote_subX:
        tot_votes += vote[0]
    if tot_votes >= k * 0.5:
        response = 1
    else:
        response = 0
vote_X.extend(X[id1], response)
print vote_X
