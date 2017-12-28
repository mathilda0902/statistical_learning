'''reading .dat files in Review_Texts:'''
import numpy as np
import pandas as pd
from collections import defaultdict

def read_user_rating(filepath, hotel):
    with open(filepath, 'r') as f:
        d = f.readlines()
        num_review = 0
        user = []
        rating = []
        # getting user names, counting number of reviews
        for el in d:
            if '<Author>' in el:
                num_review += 1
                user.append(''.join(el[8:].rstrip().split(",")))
            if '<Overall>' in el:
                rating.append(float(el[9]))
        user = pd.DataFrame(user, columns=['user'])
        rating = pd.DataFrame(rating, columns=['{} rating'.format(str(hotel))])
    return user, rating

# manual selection of 10 hotel reviews for sampling:
hotel_sample = ['hotel_72572.dat', 'hotel_72579.dat', 'hotel_72586.dat',
                'hotel_73718.dat', 'hotel_73727.dat', 'hotel_73739.dat',
                'hotel_73757.dat', 'hotel_73787.dat', 'hotel_73799.dat',
                'hotel_73821.dat']

path = 'dataset/Review_Texts'

hotel_names = ['Western_Pioneer_Square_Hotel',
                'Western_Loyal_Inn', 'Western_Executive_Inn',
                'Western_Central_Phoenix_Inn_Suites', 'Grace_Inn_at_Ahwatukee',
                'Western_InnSuites_Hotels', 'Wyndham_Phoenix',
                'Embassy_Suites_Phoenix_Airport',
                'Embassy_Suites_Hotel_Phoenix_North',
                'Hilton_Phoenix_Airport']

rev_dict = defaultdict(list)

for i in xrange(len(hotel_sample)):
    user, rating = read_user_rating(path+'/'+hotel_sample[i], hotel_names[i])
    user_mat = pd.concat([user, rating], axis=1)
    print user_mat.head(2), user_mat.shape
    rev_dict[hotel_names[i]] = user_mat

keys = rev_dict.keys()
mat1 = rev_dict[keys[0]]
mask1 = mat1['user'] == 'A TripAdvisor Member'
mat_12 = mat1[~mask1]

mat2 = rev_dict[keys[1]]
mat_22 = mat2[~mask1]
mat3 = rev_dict[keys[2]]

user_mat0 = mat1.join(mat2.set_index('user'), on='user')
user_mat0 = mat1.set_index('user').join(mat2.set_index('user'))
# join on:
user_mat = user_mat1.join(user_mat2.set_index('user'), on='user')

df1 = pd.DataFrame({'a': [1,2,3,4,7],
                    'b': [1,2,3,1,2]})
df1['id'] = np.ones(df1.shape[0]).astype(int)

df2 = pd.DataFrame({'a': [1,3,4,5,6],
                    'c': [1,2,1,2,2]})
df2['id'] = 2 * np.ones(df2.shape[0]).astype(int)

'''8 rating metrics:
 '<Overall>5\r\n',
 '<Value>5\r\n',
 '<Rooms>5\r\n',
 '<Location>5\r\n',
 '<Cleanliness>5\r\n',
 '<Check in / front desk>5\r\n',
 '<Service>5\r\n',
 '<Business service>5\r\n''''



# read in users and ratings from each hotel:
user1, rating1 = read_user_rating('dataset/Review_Texts/hotel_72572.dat',
                    'Western Pioneer Square')
user_mat1 = pd.concat([user1, rating1], axis=1)

user2, rating2 = read_user_rating('dataset/Review_Texts/hotel_72579.dat',
                    'Loyal Inn')
user_mat2 = pd.concat([user2, rating2], axis=1)
