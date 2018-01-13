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

    d1 = distance(pivot, p1)
    d2 = distance(pivot, p2)

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

    if distance(point, best) > abs(point[axis] - root['point'][axis]):
        best = closer_distance(point,
                               kdtree_closest_point(opposite_branch,
                                                    point,
                                                    depth + 1),
                               best)

    return best
