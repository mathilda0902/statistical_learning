import pandas as pd
import numpy as np

hotel_ratings = pd.read_csv('dataset/tripadvisor_json/hotel_ratings.csv')
hotel_ratings.replace(-1, 0, inplace=True)
hotel_ratings.drop(1621956, inplace=True)
hotel_ratings.fillna(0, inplace=True)
#In [37]: hotel_ratings.shape
#Out[37]: (1621956, 15)


sample_hotel = pd.read_csv('dataset/tripadvisor_json/sample_hotel.csv')

'''handling user geographics'''
g = hotel_ratings[['user', 'user location']]
grouped = g.groupby('user location').count().reset_index()
user_geo = grouped.sort_values('user', ascending=False)
major_loc = user_geo[user_geo['user'] >= 50]
#[2994 rows x 2 columns]
major_loc.to_csv('dataset/tripadvisor_json/user_location.csv')

'''how many anonymous users?'''
hotel_ratings[hotel_ratings['user'] == 'A TripAdvisor Member']['user'].count()
#Out[90]: 75503
# about 4.655% of total reviewers
'''mask anonymous users by unique location'''
anon = g[g['user'] == 'A TripAdvisor Member']
#[75503 rows x 2 columns]
#                         user             user location
#352      A TripAdvisor Member             Green Bay, WI
#355      A TripAdvisor Member                   England

'''from hotel_ratings drop all anonymous user names with NaN user location'''
mask1 = hotel_ratings['user'] != 'A TripAdvisor Member'
mask2 = hotel_ratings['user location'] != 0
mask = pd.concat([mask1, mask2], axis=1)
slct = mask.all(axis=1)
hotel_ratings_new = pd.concat([hotel_ratings, slct], axis=1)
hotel_ratings = hotel_ratings_new[hotel_ratings_new[0] != False]
hotel_ratings.drop([0], axis=1, inplace=True)

hotel_ratings.to_csv('dataset/tripadvisor_json/hotel_ratings.csv')





'''matrix factorization'''
mat_rec = hotel_ratings[['user', 'hotel id', 'ratings', 'review date']]
#In [20]: mat_rec.shape
#Out[20]: (1621956, 4)

#In [21]: mat_rec['review date'].min()
#Out[21]: 'Apr 1, 2002'

#In [22]: mat_rec['review date'].max()
#Out[22]: 'September 9, 2012'
