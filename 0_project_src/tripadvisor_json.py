import pandas as pd
import numpy as np


'''read tripadvisor_json: 12774 hotels'''

'''codes for reading one .json, create one dataframe with user-item info'''
data = json.load(open('dataset/tripadvisor_json/72572.json'))
data.keys()
'''[u'Reviews', u'HotelInfo']'''

user = []
ratings = []
for review in data['Reviews']:
    user.append(review['Author'])
    ratings.append(review['Ratings']['Overall'])

user = pd.DataFrame(user, columns=['user'])
ratings = pd.DataFrame(ratings, columns=['ratings'])
user_mat = pd.concat([user, ratings], axis=1)

'''define read-in function'''
def read_json(file):
    data = json.load(open(file))
    user = []
    ratings = []
    for review in data['Reviews']:
        user.append(review['Author'])
        ratings.append(review['Ratings']['Overall'])
    user = pd.DataFrame(user, columns=['user'])
    ratings = pd.DataFrame(ratings, columns=['ratings'])
    user_mat = pd.concat([user, ratings], axis=1)
    return user_mat

file2 = 'dataset/tripadvisor_json/72579.json'
file3 = 'dataset/tripadvisor_json/72586.json'


































if __name__ == '__main__':
    pass
