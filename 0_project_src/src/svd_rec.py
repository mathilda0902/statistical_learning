# correct dataset contains dimensions (1362204, 15)
import pandas as pd
import numpy as np
from datetime import datetime
from scipy.sparse.linalg import svds
from scipy import sparse

# working on the most popular 3000 hotels with unique 1072272 ratings,
# rated by users with English login names,
# for duplicate ratings, the most recent rating has been kept on record
ratings = pd.read_csv('dataset/popular_3k_hotels.csv')

ratings = ratings.groupby(['user', 'hotel id']).mean()
#hotel_id = ratings['hotel id'].astype(int)
ratings.drop(['Unnamed: 0', 'Unnamed: 0.1'], axis=1, inplace=True)
#ratings = pd.concat([hotel_id, ratings], axis=1)

# after taking average of duplicate ratings, we have a reduced matrix: rating
#Out[22]:
#                                ratings
#user                  hotel id
#!!                    256618        5.0
#                      503464        5.0
#!!!!!!?               93561         4.0
#                      2514403       4.0
#                      2514639       4.0
##1 Bay Area Traveler  78607         2.0
#                      91967         4.0
#                      111849        5.0
#                      2515773       4.0
#                      2516236       2.0
##1Cubsfan             87833         5.0
#                      2515843       5.0
#                      2515923       5.0

#In [13]: rating.shape
#Out[13]: (1360474, 1)

#In [14]: rating_df.shape
#Out[14]: (1374162, 4)

# pivoting ratings table: take 2
rating.reset_index(inplace=True)
#Out[24]:
#                          user  hotel id  ratings
#0                           !!    256618      5.0
#1                           !!    503464      5.0
#2                      !!!!!!?     93561      4.0
#3                      !!!!!!?   2514403      4.0
#4                      !!!!!!?   2514639      4.0

# for each user, keep the most recent rating for each hotel:
idx = ratings.groupby(['user', 'hotel id'])['review date'].transform(max) == ratings['review date']
new_ratings = ratings[idx]
new_ratings.shape

df = ratings[['user', 'hotel id', 'ratings']]
#In [15]: df.shape
#Out[15]: (1072270, 3)
new_df = pd.pivot_table(df,index=['user'], columns = 'hotel id', values = "ratings").fillna(0)

'''In this case, it might help to prune out the rare users and rare items and try
again. Also, re-examine the data collection and data cleaning process to see
if mistakes were made. Try to get more observation data per user and per item,
if you can.'''

rating_df.groupby('user').count()['ratings'].nlargest(20000)
'''Out[13]:
user
Posted by an Accorhotels.com traveler        4095
Posted by a hotelsgrandparis.com traveler     166
Posted by an Easytobook.com traveler          121
David S                                        96
David B                                        90
David M                                        89
John C                                         85
John S                                         83
David H                                        78
John B                                         74
ITA_And_RE_a                                   69
John M                                         68
John R                                         65
Pawel_EPWR                                     65
David C                                        64
Paul B                                         64
Chris B                                        63
John L                                         60
Paul M                                         60
John H                                         59
David L                                        58
John D                                         58
John W                                         57
'''




df = ratings[['user', 'hotel id', 'ratings']].sample(frac=0.05, replace=False)
df = df.pivot(index='user',columns='hotel id', values='ratings').fillna(0)

X = scipy.sparse.csc_matrix(df)
u, s, vt = svds(X, 10, which = 'LM')


























if __name__ == '__main__':
    pass
