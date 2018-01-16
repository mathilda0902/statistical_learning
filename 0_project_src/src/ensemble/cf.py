'''
ratings = pd.read_csv('dataset/unique_popular_3k_hotels.csv')
user_item = pd.read_csv('dataset/user_item_pop.csv')
user_split = user_item.groupby('user country')
hotel_split = user_item.groupby(['city'])
sub_user_geo = [user_split.get_group(x) for x in user_split.groups]
sub_hotel_geo = [hotel_split.get_group(x) for x in hotel_split.groups]
'''

import pandas as pd
from surprise import NormalPredictor
from surprise import Dataset
from surprise import Reader
from surprise.model_selection import cross_validate

# Creation of the dataframe. Column names are irrelevant.
ratings_dict = {'itemID': [1, 1, 1, 2, 2],
                'userID': [9, 32, 2, 45, 'user_foo'],
                'rating': [3, 2, 4, 3, 1]}
df = ratings[['user', 'hotel id', 'ratings']]

# A reader is still needed but only the rating_scale param is requiered.
reader = Reader(rating_scale=(1, 5))

# The columns must correspond to user id, item id and ratings (in that order).
data = Dataset.load_from_df(df[['user', 'hotel id', 'ratings']], reader)

# We can now use this dataset as we please, e.g. calling cross_validate
cross_validate(NormalPredictor(), data, cv=2)



from surprise import SVD
from surprise import accuracy
from surprise.model_selection import KFold
from surprise.model_selection import GridSearchCV

# CV
# define a cross-validation iterator
kf = KFold(n_splits=3)

algo = SVD()

for trainset, testset in kf.split(data):

    # train and test algorithm.
    algo.fit(trainset)
    predictions = algo.test(testset)

    # Compute and print Root Mean Squared Error
    accuracy.rmse(predictions, verbose=True)


# grid search for SVD:
param_grid = {'n_epochs': [5, 10], 'lr_all': [0.002, 0.005],
              'reg_all': [0.4, 0.6]}
gs = GridSearchCV(SVD, param_grid, measures=['rmse', 'mae'], cv=3)

gs.fit(data)

# best RMSE score
print(gs.best_score['rmse'])

# combination of parameters that gave the best RMSE score
print(gs.best_params['rmse'])

algo = gs.best_estimator['rmse']
algo.fit(data.build_full_trainset())
