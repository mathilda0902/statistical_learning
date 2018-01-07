import pandas as pd
import numpy as np

rating_df = pd.read_csv('dataset/hotel_ratings.csv')
#

rating_df = rating_df[['user', 'hotel id', 'ratings', 'review date']]
rating_df.head()

#Out[4]:
#             user  hotel id  ratings        review date
#0  strawberryshtc     99774      4.0     March 14, 2012
#1    travelingTch     99774      5.0     March 11, 2012
#2        Dorit147     99774      3.0      March 8, 2012
#3        ashley r     99774      4.0  February 28, 2012
#4    kacunningham     99774      4.0  February 27, 2012

#R_df = rating_df.pivot(index='user', columns='hotel id', values='ratings').fillna(0)
#ValueError: Index contains duplicate entries, cannot reshape
# users might rated multiple times for the same hotel, choose the average rating

rating = rating_df.groupby(['user', 'hotel id']).mean()

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

R_df = rating.pivot(index='user', columns='hotel id', values='ratings').fillna(0)



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

from scipy.sparse.linalg import svds
from scipy import sparse


df = rating[['user', 'hotel id', 'ratings']].sample(frac=0.05, replace=False)
df = df.pivot(index='user',columns='hotel id', values='ratings').fillna(0)

X = scipy.sparse.csc_matrix(df)
u, s, vt = svds(X, 10, which = 'LM')


























if __name__ == '__main__':
    pass
