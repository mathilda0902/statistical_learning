# knn on dimensions: (40 major regions)
# hotels (high dimensions)
# sub ratings (only 7), similarity among the hotels
# high frequency words: similarity among the travelers

import pprint
import numpy as np
import pandas as pd
import random
import sklearn.preprocessing as pp
from numpy import linalg


ppr = pprint.PrettyPrinter(indent=4)

def euc_distance(point1, point2):
    diff = np.subtract(point1, point2)
    dist = np.linalg.norm(diff)
    return dist

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

def accuracy(rec, test):
    return np.sqrt(np.linalg.norm((rec - test))/7.0)


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
test = pd.read_csv('hotel/unique_popular_3k_hotels.csv')
test = test[['hotel id', 'rooms', 'service', 'cleanliness', 'front desk', 'business service', 'value', 'location']]
test = test.groupby('hotel id').mean()

training = test[:2000]
test = test[2000:]
training_points = training.as_matrix()
test_points = test.as_matrix()
test_hotel1 = test_points[0]
test_hotel2 = test_points[1]
test_hotel3 = test_points[50]

k = 7
kdtree = build_kdtree(training_points)
rec_hotel1 = kdtree_closest_point(kdtree, test_hotel1)

rmse = (accuracy(rec_hotel1, test_hotel1) + accuracy(rec_hotel2, test_hotel2)
        + accuracy(rec_hotel3, test_hotel3)) / 3.0

np.argwhere(training_points == rec_hotel1)
np.argwhere(test_points == test_hotel1)

test.iloc[150]
test.iloc[78]

hotel_names[hotel_names['hotel id'] == 80747]
hotel_names[hotel_names['hotel id'] == 85031]

# rec 85 in training_points = nyc[90]: hotel id == 79868
# test[0] in nyc[101]: hotel id == 81394
#hotel id == 81087
#hotel id == 81397

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
