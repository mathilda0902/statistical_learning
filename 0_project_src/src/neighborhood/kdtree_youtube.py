# knn on dimensions: (40 major regions)
# hotels (high dimensions)
# sub ratings (only 7), similarity among the hotels
# high frequency words: similarity among the travelers


import math
import pprint
import numpy as np
import sklearn.preprocessing as pp

ppr = pprint.PrettyPrinter(indent=4)

def euc_distance(point1, point2):
    diff = np.subtract(point1, point2)
    dist = np.linalg.norm(diff)
    return dist

def _userItemMatrix(arr):
	df = df[['user', 'hotel id', 'ratings']]
	pdf = pd.pivot_table(df, index=['user'], columns = 'hotel id', values = "ratings").fillna(0)
	mat = csr_matrix(pdf)
    mat = df.astype(float).values
	return mat

def userCosineSim(arr):
    # base similarity matrix (all dot products)
    # replace this with A.dot(A.T).toarray() for sparse representation
    similarity = np.dot(arr, arr.T)
    # squared magnitude of preference vectors (number of occurrences)
    square_mag = np.diag(similarity)
    # inverse squared magnitude
    inv_square_mag = 1 / square_mag
    # if it doesn't occur, set it's inverse magnitude to zero (instead of inf)
    inv_square_mag[np.isinf(inv_square_mag)] = 0
    # inverse of the magnitude
    inv_mag = np.sqrt(inv_square_mag)
    # cosine similarity (elementwise multiply by inverse magnitudes)
    cosine = similarity * inv_mag
    cosine = cosine.T * inv_mag
    return cosine

def itemCosineSim(arr):
    arr = arr.T
    # base similarity matrix (all dot products)
    # replace this with A.dot(A.T).toarray() for sparse representation
    similarity = np.dot(arr, arr.T)
    # squared magnitude of preference vectors (number of occurrences)
    square_mag = np.diag(similarity)
    # inverse squared magnitude
    inv_square_mag = 1 / square_mag
    # if it doesn't occur, set it's inverse magnitude to zero (instead of inf)
    inv_square_mag[np.isinf(inv_square_mag)] = 0
    # inverse of the magnitude
    inv_mag = np.sqrt(inv_square_mag)
    # cosine similarity (elementwise multiply by inverse magnitudes)
    cosine = similarity * inv_mag
    cosine = cosine.T * inv_mag
    return cosine

'''
A = np.array([[ 4.,  5.,  4.],
       [ 3.,  4.,  3.],
       [ 1.,  1.,  1.],
       [ 2.,  5.,  5.],
       [ 4.,  2.,  2.]], dtype=np.longdouble)
In [181]: cosineSim(mat)
Out[181]:
array([[ 1.        ,  0.99948387,  0.99413485,  0.95530392,  0.91925472],
       [ 0.99948387,  1.        ,  0.99014754,  0.95685806,  0.91018205],
       [ 0.99413485,  0.99014754,  1.        ,  0.94280904,  0.94280904],
       [ 0.95530392,  0.95685806,  0.94280904,  1.        ,  0.77777778],
       [ 0.91925472,  0.91018205,  0.94280904,  0.77777778,  1.        ]])
'''
'''
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cosine
dist_out = 1-pairwise_distances(A, metric="cosine")
'''
'''
def closest_point(all_points, new_point):
    best_point = None
    best_distance = None

    for current_point in all_points:
        current_distance = distance(new_point, current_point)

        if best_distance is None or current_distance < best_distance:
            best_distance = current_distance
            best_point = current_point

    return best_point
'''

def angDist(arr):
    pass

def build_kdtree(points, depth=0):
    n = len(points)

    if n <= 0:
        return None

    axis = depth % k

    sorted_points = sorted(points, key=lambda point: point[axis])

    return {
        'point': sorted_points[n / 2],
        'left': build_kdtree(sorted_points[:n / 2], depth + 1),
        'right': build_kdtree(sorted_points[n/2 + 1:], depth + 1)
    }

kdtree = build_kdtree(points)

def kdtree_naive_closest_point(root, point, depth=0, best=None):
    if root is None:
        return best

    axis = depth % k

    next_best = None
    next_branch = None

    if best is None or distance(point, best) > distance(point, root['point']):
        next_best = root['point']
    else:
        next_best = best

    if point[axis] < root['point'][axis]:
        next_branch = root['left']
    else:
        next_branch = root['right']

    return kdtree_naive_closest_point(next_branch, point, depth + 1, next_best)

def closer_distance(pivot, p1, p2):
    if p1 is None:
        return p2

    if p2 is None:
        return p1

    d1 = euc_distance(pivot, p1)
    d2 = euc_distance(pivot, p2)

    if d1 < d2:
        return p1
    else:
        return p2


def kdtree_closest_point(root, point, depth=0):
    if root is None:
        return None

    axis = depth % k

    next_branch = None
    opposite_branch = None

    if point[axis] < root['point'][axis]:
        next_branch = root['left']
        opposite_branch = root['right']
    else:
        next_branch = root['right']
        opposite_branch = root['left']

    best = closer_distance(point,
                           kdtree_closest_point(next_branch,
                                                point,
                                                depth + 1),
                           root['point'])

    if euc_distance(point, best) > abs(point[axis] - root['point'][axis]):
        best = closer_distance(point,
                               kdtree_closest_point(opposite_branch,
                                                    point,
                                                    depth + 1),
                               best)

    return best

# run on hotel info with 7d tree:
'''
In [84]: test
Out[84]:
         hotel id  rooms  service  cleanliness  front desk  business service  \
0           99774    3.0      5.0          5.0         0.0               0.0
1           99774    4.0      5.0          5.0         0.0               0.0
2           99774    3.0      3.0          3.0         0.0               0.0
3           99774    5.0      5.0          5.0         0.0               0.0
4           99774    4.0      4.0          4.0         0.0               0.0
5           99774    4.0      4.0          4.0         0.0               0.0
'''

test.groupby('hotel id').mean()
training = test[:100]
test = test[100:200]
training_points = training.as_matrix()
test_points = test.as_matrix()
test_hotel1 = test_points[0]
test_hotel2 = test_points[1]
test_hotel3 = test_points[2]

k = 7
kdtree = build_kdtree(training_points)
rec_hotel1 = kdtree_closest_point(kdtree, test_hotel1)

# rec 85 in training_points = nyc[85]: hotel id == 217622
# test[0] in nyc[200]: hotel id == 2514392

'''
In [120]: hotel_names[hotel_names['hotel id'] == 79868]
Out[120]:
                    hotel name  hotel id       city     state zip code  \
11709  Bay Club Hotel & Marina     79868  San Diego  CA 92106    92106

      low price high price
11709      $129       $217

In [121]: hotel_names[hotel_names['hotel id'] == 81394]
Out[121]:
                hotel name  hotel id           city     state zip code  \
11905  The Donatello Hotel     81394  San Francisco  CA 94102    94102

      low price high price
11905      $171       $710

In [132]: hotels[hotels['hotel id'] == 79868]
Out[132]:
    hotel id  Unnamed: 0   ratings     rooms   service  cleanliness  \
45     79868    155255.5  4.443262  3.801418  4.400709     4.329787

    front desk  business service     value  location
45    0.765957          0.439716  4.230496   4.06383

In [133]: hotels[hotels['hotel id'] == 81394]
Out[133]:
     hotel id  Unnamed: 0  ratings     rooms   service  cleanliness  \
100     81394    134616.5  4.53753  3.924939  4.470944      4.48184

     front desk  business service    value  location
100    0.652542           0.35109  4.37046  4.146489
'''


# FLANN
import cProfile
from numpy import random
from pyflann import *
from scipy import spatial

# Config params
dim = 4
knn = 5
dataSize = 1000
testSize = 1

# Generate data
random.seed(1)
dataset = random.rand(dataSize, dim)
testset = random.rand(testSize, dim)

def test1(numIter=1000):
    '''Test tree build time.'''
    flann = FLANN()
    for k in range(numIter):
        kdtree = spatial.cKDTree(dataset, leafsize=10)
        params = flann.build_index(dataset, target_precision=0.0, log_level = 'info')

def test2(numIter=100):
    kdtree = spatial.cKDTree(dataset, leafsize=10)
    flann = FLANN()
    params = flann.build_index(dataset, target_precision=0.0, log_level = 'info')
    for k in range(numIter):
        result1 = kdtree.query(testset, 5)
        result2 = flann.nn_index(testset, 5, checks=params['checks'])

import cProfile
cProfile.run('test2()', 'out.prof')
