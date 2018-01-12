# correct dataset contains dimensions (1362204, 15)
import pandas as pd
import numpy as np
from datetime import datetime

unique_hotel_ratings = pd.read_csv('dataset/hotel_unique_ratings.csv')

hotel_ratings = pd.read_csv('dataset/hotel_ratings.csv')
# https://s3.amazonaws.com/multifacetedhotelrecommender/hotel_ratings.csv
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

hotel_ratings.to_csv('dataset/hotel_ratings.csv')

# after cleaning, matrix shape is:
#In [9]: rdf.shape
#Out[9]: (1374162, 16)



# rating dates spans: 'Apr 1, 2002' to 'September 9, 2012'
mat_rec = hotel_ratings[['user', 'hotel id', 'ratings', 'review date']]
#In [20]: mat_rec.shape
#Out[20]: (1621956, 4)

#In [21]: mat_rec['review date'].min()
#Out[21]: 'Apr 1, 2002'

#In [22]: mat_rec['review date'].max()
#Out[22]: 'September 9, 2012'

# clean user names: delete all non-english
# -*- coding: utf-8 -*-
def isEnglish(s):
    try:
        s.encode(encoding='utf-8').decode('ascii')
    except UnicodeDecodeError:
        return False
    else:
        return True

# complete ratings, after dropping anonymous users without login locations:
#In [20]: ratings.head()
#Out[20]:
#             user  hotel id  ratings        review date
#0  strawberryshtc     99774      4.0     March 14, 2012
#1    travelingTch     99774      5.0     March 11, 2012
#2        Dorit147     99774      3.0      March 8, 2012
#3        ashley r     99774      4.0  February 28, 2012
#4    kacunningham     99774      4.0  February 27, 2012


# delete non-english user names:
msk = [isEnglish(ele) for ele in rdf['user']]

rdf = rdf[msk]
rdf.shape
#Out[52]: (1373744, 16)

user = [e.decode('utf-8') for e in rdf['user']]
user = pd.DataFrame(user, columns=['user'])
rdf = pd.concat([user, rdf[['hotel id', 'ratings', 'review date', 'review id',
                'user location', 'title', 'content', 'rooms', 'service',
                'cleanliness', 'front desk', 'business service', 'value', 'location']]],axis=1)

rdf = rdf[msk]
#In [93]: rdf.shape
#Out[93]: (1373744, 15)

# write into csv:
rdf.to_csv('dataset/hotel_encoded_ratings.csv')
# upload to s3:
aws s3 cp dataset/hotel_encoded_ratings.csv s3://multifacetedhotelrecommender


# for each user, keep the most recent rating for each hotel:
date = pd.to_datetime(rdf['review date'])
rdf.drop(['review date'], axis=1, inplace=True)
hotel_id = rdf['hotel id'].astype(int)
rdf.drop(['hotel id', 'Unnamed: 0'], axis=1, inplace=True)
new_rdf = pd.concat([hotel_id, date, rdf], axis=1)

idx = new_rdf.groupby(['user', 'hotel id'])['review date'].transform(max) == new_rdf['review date']
unique = new_rdf[idx]
output = unique[unique['user'] != 'A TripAdvisor Member']
unique.shape
# (1362204, 15)

#write in to csv:
unique.to_csv('dataset/hotel_unique_ratings.csv')
# rank hotel id by the number of unique reviews/ratings in descending order:
unique_hotel_ratings = pd.read_csv('dataset/hotel_unique_ratings.csv')
hotel_popularity = unique_hotel_ratings.groupby(['hotel id']).size().reset_index(name='counts').sort_values('counts', ascending=False)

hotel_popularity[hotel_popularity['counts'] == 1]
# total reviews for each hotel:
hotel_popularity['counts'].describe()
#Out[19]:
#count    10006.000000
#mean       136.138917
#std        250.149549
#min          1.000000
#25%         21.000000
#50%         58.000000
#75%        147.000000
#max       4505.000000
#Name: counts, dtype: float64

# hotels that received under 21 unique ratings/reviews:
hotel_popularity[hotel_popularity['counts'] <= 21].shape
#Out[20]: (2515, 2)
# so there are 7491 hotels that have been rated by unique visitors for more than 21 times.

# most popular 3000 hotels:
most_pop = hotel_popularity.head(3000)
pop_list = most_pop['hotel id'].astype(int)

pop_hotels = unique_hotel_ratings[unique_hotel_ratings['hotel id'].isin(pop_list)]
pop_hotels.shape
#Out[33]: (1072270, 16)

pop_hotels.groupby('hotel id').mean()
#[3000 rows x 10 columns]

pop_hotels.to_csv('dataset/popular_3k_hotels.csv')


# anonymous:
'Posted by an Accorhotels.com traveler'
new_df = df.ix[~(df['user'] == 'Posted by an Accorhotels.com traveler')]
'Posted by a hotelsgrandparis.com traveler'
new_df = new_df.ix[~(df['user'] == 'Posted by a hotelsgrandparis.com traveler')]
'Posted by an Easytobook.com traveler'
new_df = new_df.ix[~(df['user'] == 'Posted by an Easytobook.com traveler')]
#David S              96
#David B              90
#David M              89
#John C               85
#John S               83
#David H              78
#John B               74
#ITA_And_RE_a         69
#John M               68
count = new_df['user'].value_counts()
dup_names = count.head(100000)
dup_names = dup_names.reset_index()
'''Out[82]:
                      index  user
0                   David S    96
1                   David B    90
2                   David M    89
3                    John C    85
4                    John S    83
5                   David H    78
6                    John B    74
7              ITA_And_RE_a    69
8                    John M    68
9                Pawel_EPWR    65
10                   John R    65
11                   Paul B    64
12                  David C    64
13                  Chris B    63'''

magic_barrier = new_df[new_df['user'].isin(dup_names['index'])]
output = magic_barrier.groupby(['user', 'hotel id']).mean()


reg = pd.read_csv('dataset/major_region_ratings.csv')
hotel = pd.read_csv('dataset/hotel_info.csv')
hotel_set = set(reg['hotel id'].tolist())
pop_hotel = hotel.loc[hotel['hotel id'].isin(hotel_set)]
# In [233]: pop_hotel.shape
# Out[233]: (3000, 7)
# selecting one location
# pop_hotel = pd.read_csv('dataset/user_item_pop.csv', index=False)
# user_item = reg.merge(pop_hotel, 'inner', on='hotel id')
# user_item.drop(['Unnamed: 0', 'Unnamed: 0.1'], axis=1, inplace=True)
# user_item.to_csv('dataset/user_item_pop.csv', index=False)
#In [42]: user_item.shape
#Out[42]: (532212, 24)
user_item = pd.read_csv('dataset/user_item_pop.csv')
pop_hotel_loc = set(pop_hotel['state'])
msk = pop_hotel['state'].str.contains('Spain') == True
pop_hotel[msk]













if __name__ == '__main__':
    pass
