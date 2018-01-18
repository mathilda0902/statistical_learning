import operator
import csv
import random
import pandas as pd
import sklearn.preprocessing as pp
from scipy.sparse import csr_matrix
from multiprocessing import Pool
from sklearn.metrics import mean_squared_error
from math import sqrt
# from .kdtree import node
# https://machinelearningmastery.com/tutorial-to-implement-k-nearest-neighbors-in-python-from-scratch/

class knn_baseline(object):
	def __init__(self):
		pass

	def _tt_split(data, test_size=0.2):
		trainset = []
		testset = []
		for x in range(data.shape[0]):
			if random.random() < 1 - test_size:
				trainset.append(data.iloc[x])
			else:
				testset.append(data.iloc[x])
		return trainset, testset

	def _userItemMatrix(df):
		df = df[['user', 'hotel id', 'ratings']]
		pdf = pd.pivot_table(df, index=['user'], columns = 'hotel id', values = "ratings").fillna(0)
		mat = csr_matrix(pdf)
		return mat

	# item cosine similarity matrix
	def _itemSimilarities(mat):
    	col_normed_mat = pp.normalize(mat.tocsc(), axis=0)
    	return col_normed_mat.T * col_normed_mat

	# user cosine similarity matrix
	def _userSimilarities(mat):
    	col_normed_mat = pp.normalize(mat.tocsc(), axis=0)
    	return col_normed_mat * col_normed_mat.T

	def predict(ratings, similarity, type='user'):
    	if type == 'user':
        	mean_user_rating = ratings.mean(axis=1)
        	# use np.newaxis to ensure that mean_user_rating has same format as ratings
        	ratings_diff = (ratings - mean_user_rating[:, np.newaxis])
        	pred = mean_user_rating[:, np.newaxis] + similarity.dot(ratings_diff) / np.array([np.abs(similarity).sum(axis=1)]).T
    	elif type == 'item':
        	pred = ratings.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)])
    	return pred

	def evaluation(pred_rating, true_rating):
    	pred_rating = prediction[ground_truth.nonzero()].flatten()
    	true_rating = ground_truth[ground_truth.nonzero()].flatten()
    	return sqrt(mean_squared_error(prediction, ground_truth))

	# mega function for pipelining helpers to be sent to multiprocessors
	def matToSim(df):
		pdf = getMatrix(df)
		sim_mat = cosineSimilarities(pdf)
		return sim_mat

	# helper functin convert arrays to tuples:
	def totuple(a):
	    try:
	        return tuple(totuple(i) for i in a)
	    except TypeError:
	        return a



if __name__ == '__main__':
	pool = Pool(processes=5)
	user_item = pd.read_csv('dataset/user_item_pop.csv')
	#user_split = user_item.groupby('user country')
	hotel_split = user_item.groupby(['city'])
	#sub_user_geo = [user_split.get_group(x) for x in user_split.groups]
	sub_hotel_geo = [hotel_split.get_group(x) for x in hotel_split.groups]
	print pool.map(matToSim, sub_user_geo)
	print pool.map(predict, sub_user_geo)
	print pool.map(evaluation, sub_user_geo)
