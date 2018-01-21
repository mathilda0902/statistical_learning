'''
ratings = pd.read_csv('dataset/unique_popular_3k_hotels.csv')
user_item = pd.read_csv('dataset/user_item_pop.csv')
user_split = user_item.groupby('user country')
hotel_split = user_item.groupby(['city'])
sub_user_geo = [user_split.get_group(x) for x in user_split.groups]
sub_hotel_geo = [hotel_split.get_group(x) for x in hotel_split.groups]
'''
import random
import pandas as pd
from surprise import NormalPredictor
from surprise import Dataset
from surprise import Reader
from surprise.model_selection import cross_validate
from surprise.model_selection import train_test_split

from surprise import BaselineOnly
from surprise import SVD
from surprise import NMF
from surprise import SlopeOne
from surprise import CoClustering

from surprise import accuracy
from surprise.model_selection import KFold
from surprise.model_selection import GridSearchCV


# Creation of the dataframe. Column names are irrelevant.
ratings_dict = {'itemID': [1, 1, 1, 2, 2],
                'userID': [9, 32, 2, 45, 'user_foo'],
                'rating': [3, 2, 4, 3, 1]}
df = ratings[['user', 'hotel id', 'ratings']]


ratings = pd.read_csv('hotel/user_item_pop.csv')
ratings = ratings[['user', 'hotel id', 'ratings']]
# A reader is still needed but only the rating_scale param is requiered.
reader = Reader(rating_scale=(1, 5))

# The columns must correspond to user id, item id and ratings (in that order).
data = Dataset.load_from_df(ratings, reader)
raw_ratings = data.raw_ratings
# shuffle ratings if you want
random.shuffle(raw_ratings)
# A = 90% of the data, B = 10% of the data
threshold = int(.9 * len(raw_ratings))
A_raw_ratings = raw_ratings[:threshold]
B_raw_ratings = raw_ratings[threshold:]

data.raw_ratings = B_raw_ratings





# We can now use this dataset as we please, e.g. calling cross_validate
cross_validate(NormalPredictor(), data, cv=2)

#estimate normal distribution mu and sigma of all ratings:
# major_region_ratings
t = ratings['ratings']
arr = t.as_matrix()
#In [50]: arr[:10]
#Out[50]: array([5., 3., 4., 4., 4., 4., 2., 1., 4., 5.])
In [53]: np.mean(arr)
Out[53]: 3.9996730626141463
In [54]: np.var(arr)
Out[54]: 1.3228787458999327
normal(4, 1.322879)


# baseline only:
algo =

# knn basic:
param_grid_knn = {'k': [30, 40, 50]}
gs_knn = GridSearchCV(KNNBasic, param_grid_knn, measures=['rmse'], cv=3)
gs_knn.fit(data)
# best RMSE score
print(gs_knn.best_score['rmse'])
# combination of parameters that gave the best RMSE score
print(gs_knn.best_params['rmse'])
results_df = pd.DataFrame.from_dict(gs_knn.cv_results)



# grid search for SVD: 'n_epochs': [26],
param_grid_svd = {'lr_all': [0.0019], 'biased': [True], 'reg_all': [0.39],
                    'n_epochs': [10, 20, 30, 40, 50, 60, 70, 80]}
param_grid_svd = {'lr_all': [0.0019], 'biased': [True], 'reg_all': [0.39],
                    'n_epochs': [26], 'n_factors': [50, 100, 200]}
param_grid_svd = {'lr_all': [0.0019], 'n_epochs': [26, 50, 70],
                'biased': [True], 'reg_all': [0.385]}
gs_svd = GridSearchCV(SVD, param_grid_svd, measures=['rmse'], cv=3)
gs_svd.fit(data)

# best RMSE score
print(gs_svd.best_score['rmse'])
# combination of parameters that gave the best RMSE score
print(gs_svd.best_params['rmse'])
results_df = pd.DataFrame.from_dict(gs_svd.cv_results)


algo = gs_svd.best_estimator['rmse']
testset = data.construct_testset(B_raw_ratings)  # testset is now the set B
predictions = algo.test(testset)
accuracy.rmse(predictions)


#param_grid_nmf = {'n_epochs': [42, 43, 44], 'reg_pu': [0.039, 0.040, 0.041]}
#gs_nmf = GridSearchCV(NMF, param_grid_nmf, measures=['rmse', 'mae'], cv=3)
#gs_nmf.fit(data)

# grid search for CoClustering
param_grid_cocluster = {'n_cltr_u': [3,5,7], 'n_cltr_i': [3,5,7],
                        'n_epochs': [10, 20, 30, 40, 50]}
'''
n_cltr_u (int) – Number of user clusters. Default is 3.
n_cltr_i (int) – Number of item clusters. Default is 3.
n_epochs (int) – Number of iteration of the optimization loop. Default is 20.
'''
gs_cltr = GridSearchCV(CoClustering, param_grid_cocluster, measures=['rmse'], cv=3)
gs_cltr.fit(data)

results_df = pd.DataFrame.from_dict(gs_cltr.cv_results)

# best RMSE score
print(gs_cltr.best_score['rmse'])
# combination of parameters that gave the best RMSE score
print(gs_cltr.best_params['rmse'])

algo = gs_cltr.best_estimator['rmse']

# retrain on the whole set A
trainset = data.build_full_trainset()
algo.fit(trainset)

# Compute biased accuracy on A
predictions = algo.test(trainset.build_testset())
print('Biased accuracy on A,', end='   ')
accuracy.rmse(predictions)

# Compute unbiased accuracy on B
testset = data.construct_testset(B_raw_ratings)  # testset is now the set B
predictions = algo.test(testset)
print('Unbiased accuracy on B,', end=' ')
accuracy.rmse(predictions)


# comparing models:
# Spot Check Algorithms
models = []
models.append(('AlgoBase', BaselineOnly()))
models.append(('BaselineOnly', KNNBasic()))
models.append(('KNNBasic', KNNWithMeans()))
models.append(('KNNWithMeans', KNNWithZScore()))
models.append(('KNNWithZScore', SVD()))
models.append(('SVD', NMF()))
models.append(('NMF', SlopeOne()))
models.append(('SlopeOne', CoClustering()))

# evaluate each model in turn
results = []
names = []
for name, model in models:

    # define a cross-validation iterator
    kf = KFold(n_splits=3)
    algo = model
    for trainset, testset in kf.split(data):
        # train and test algorithm.
        algo.fit(trainset)
        predictions = algo.test(testset)
        # Compute and print Root Mean Squared Error
        accuracy.rmse(predictions, verbose=True)

	kfold = model_selection.KFold(n_splits=10, random_state=seed)
	cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)
