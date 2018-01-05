#http://surprise.readthedocs.io/en/stable/getting_started.html#basic-usage
import pandas as pd
from surprise import SVD
from surprise import Reader, Dataset
from surprise import evaluate, print_perf
from surprise.evaluate import GridSearch

# entire hotel ratings set
hotel_ratings = pd.read_csv('dataset/hotel_ratings.csv')
hotel_id = hotel_ratings['hotel id'].astype(int)
user = hotel_ratings['user']
rating = hotel_ratings['ratings'].astype(int)
ratings = pd.concat([user, hotel_id, rating], axis=1)

# make sample ratings of 100,000 reviews:
hotel_ratings.iloc[:100000].to_csv('dataset/sample_ratings.csv')

# sample hotel ratings
sample = pd.read_csv('dataset/sample_ratings.csv')
hotel_id = sample['hotel id'].astype(int)
user = sample['user']
rating = sample['ratings'].astype(int)
sample_ratings = pd.concat([user, hotel_id, rating], axis=1)
sample0 = sample_ratings.iloc[:10]


reader = Reader(rating_scale=(1,5))
data = Dataset.load_from_df(sample_ratings, reader)
data.split(5)

param_grid = {'n_epochs': [5, 10], 'lr_all': [0.002, 0.005],
                'reg_all': [0.4, 0.6]}
grid_search = GridSearch(SVD, param_grid, measures=['RMSE', 'FCP'])
grid_search.evaluate(data)

# train data to KNN basic model
reader = Reader(rating_scale=(1,5))
data = Dataset.load_from_df(sample_ratings, reader)
trainset = data.build_full_trainset()

algo = KNNBasic()
algo.train(trainset)

# predict
uid = str('strawberryshtc')
iid = int(99774)
pred = algo.predict(uid, iid, r_ui=4, verbose=True)

# train KNN baseline with pearson baseline
trainset = data.build_full_trainset()
sim_options = {'name': 'pearson_baseline', 'user_based': False}
algo = KNNBaseline(sim_options=sim_options)
algo.train(trainset)

testset = trainset.build_anti_testset()
predictions = algo.test(testset)

# get top 10 recommendation for each user
top_n = get_top_n(predictions, n=10)


















if __name__ == '__main__':
    pass
