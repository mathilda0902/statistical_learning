import operator
import csv
import random
import pandas as pd
import sklearn.preprocessing as pp
from scipy.sparse import csr_matrix
from multiprocessing import Pool
# https://machinelearningmastery.com/tutorial-to-implement-k-nearest-neighbors-in-python-from-scratch/

def loadDataset(filename, split, trainingSet=[] , testSet=[]):
	df = pd.read_csv(filename)

def getMatrix(df):
	df = df[['user', 'hotel id', 'ratings']]
	pdf = pd.pivot_table(df,index=['user'], columns = 'hotel id', values = "ratings").fillna(0)
	mat = csr_matrix(pdf)
	return mat

def cosineSimilarities(mat):
    col_normed_mat = pp.normalize(mat.tocsc(), axis=0)
    #proc_name = os.getid()
    return col_normed_mat.T * col_normed_mat

def matToSim(df):
	pdf = getMatrix(df)
	sim_mat = cosineSimilarities(pdf)
	return sim_mat

def getNeighbors(trainingSet, testInstance, k):
	distances = []
	length = len(testInstance)-1
	for x in range(len(trainingSet)):
		dist = euclideanDistance(testInstance, trainingSet[x], length)
		distances.append((trainingSet[x], dist))
	distances.sort(key=operator.itemgetter(1))
	neighbors = []
	for x in range(k):
		neighbors.append(distances[x][0])
	return neighbors

if __name__ == '__main__':
	pool = Pool(processes=5)
	user_item = pd.read_csv('dataset/user_item_pop.csv')
	user_split = user_item.groupby('user country')
	sub_user_geo = [user_split.get_group(x) for x in user_split.groups]
	print pool.map(matToSim, sub_user_geo)
