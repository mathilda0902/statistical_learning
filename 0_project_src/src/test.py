import pandas as pd
from surprise import Dataset, Reader, KNNBasic

ratings = pd.read_csv('dataset/popular_3k_hotels.csv', index_col=False)
df = ratings[['user', 'hotel id', 'ratings']].sample(frac=0.05, replace=False)
bsl_options = {'method': 'simple_gd', 'n_epochs': 20}
sim_options = {'name': 'pearson_baseline'}
algo = KNNBasic(k=10, bsl_options=bsl_options, sim_options=sim_options)

reader = Reader(rating_scale=(1,5))
data = Dataset.load_from_df(df, reader)
trainset = data.build_full_trainset()

algo.train(trainset)
algo.sim

bsl_options = {'method': 'simple_gd',
           'n_epochs': 20
           }
sim_options = {'name': 'pearson_baseline'}
algo = KNNBasic(k=10, bsl_options=bsl_options, sim_options=sim_options)
df = ratings[['user', 'hotel id', 'ratings']].sample(frac=0.05, replace=False)
reader = Reader(rating_scale=(1,5))
data = Dataset.load_from_df(df, reader)
trainset = data.build_full_trainset()

algo.train(trainset)
algo.sim

'''
Out[12]:
array([[ 1.,  0.,  0., ...,  0.,  0.,  0.],
   [ 0.,  1.,  0., ...,  0.,  0.,  0.],
   [ 0.,  0.,  1., ...,  0.,  0.,  0.],
   ...,
   [ 0.,  0.,  0., ...,  1.,  0.,  0.],
   [ 0.,  0.,  0., ...,  0.,  1.,  0.],
   [ 0.,  0.,  0., ...,  0.,  0.,  1.]])
'''
'''
In [16]: algo.sim.sum()
Out[16]: 50688.464135821167

In [17]: df.shape
Out[17]: (53614, 3)
'''


sklearn.metrics.pairwise.pairwise_distances(X, Y=None, metric='euclidean', n_jobs=1, **kwds)
