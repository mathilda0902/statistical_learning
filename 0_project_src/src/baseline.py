import pandas as pd
from __future__ import (absolute_import, division, print_function, unicode_literals)
from surprise import SVD
from surprise import Dataset
from surprise import evaluate, print_perf
from surprise import Reader
from surprise import GridSearch

ratings = pd.read_csv('project/popular_3k_hotels.csv')
df = ratings[['user', 'hotel id', 'ratings']].sample(frac=0.05, replace=False)
# A reader is still needed but only the rating_scale param is requiered.
reader = Reader(rating_scale=(1, 5))
# The columns must correspond to user id, item id and ratings (in that order).
data = Dataset.load_from_df(ratings[['user', 'hotel id', 'ratings']], reader)

#data.split(2)  # data can now be used normally

param_grid = {'n_epochs': [5, 10], 'lr_all': [0.002, 0.005],
              'reg_all': [0.4, 0.6]}
grid_search = GridSearch(SVD, param_grid, measures=['RMSE', 'FCP'])

data.split(n_folds=3)
grid_search.evaluate(data)

'''------------
Parameters combination 1 of 8
params:  {u'lr_all': 0.002, u'reg_all': 0.4, u'n_epochs': 5}
------------
Mean RMSE: 1.0422
Mean FCP : 0.5478
------------
------------
Parameters combination 2 of 8
params:  {u'lr_all': 0.002, u'reg_all': 0.4, u'n_epochs': 10}
------------
Mean RMSE: 1.0391
Mean FCP : 0.5482
------------
------------
Parameters combination 3 of 8
params:  {u'lr_all': 0.002, u'reg_all': 0.6, u'n_epochs': 5}
------------
Mean RMSE: 1.0475
Mean FCP : 0.5483
------------
------------
Parameters combination 4 of 8
params:  {u'lr_all': 0.002, u'reg_all': 0.6, u'n_epochs': 10}
------------
Mean RMSE: 1.0451
Mean FCP : 0.5485
------------
------------
Parameters combination 5 of 8
params:  {u'lr_all': 0.005, u'reg_all': 0.4, u'n_epochs': 5}
------------
Mean RMSE: 1.0392
Mean FCP : 0.5481
------------
------------
Parameters combination 6 of 8
params:  {u'lr_all': 0.005, u'reg_all': 0.4, u'n_epochs': 10}
------------
Mean RMSE: 1.0399
Mean FCP : 0.5487
------------
------------
Parameters combination 7 of 8
params:  {u'lr_all': 0.005, u'reg_all': 0.6, u'n_epochs': 5}
------------
Mean RMSE: 1.0452
Mean FCP : 0.5484
------------
------------
Parameters combination 8 of 8
params:  {u'lr_all': 0.005, u'reg_all': 0.6, u'n_epochs': 10}
------------
Mean RMSE: 1.0461
Mean FCP : 0.5485
------------'''
























if __name__ == '__main__':
    pass
