# compute cosine similarities
import pandas as pd
import numpy as np
from collections import defaultdict
from scipy.sparse import csr_matrix
import sklearn
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KDTree

#ratings = pd.read_csv('dataset/popular_3k_hotels.csv')
#idx = ratings.groupby(['user', 'hotel id'])['review date'].transform(max) == ratings['review date']
#new_ratings = ratings[idx]
#new_ratings.to_csv('dataset/unique_popular_3k_hotels.csv')

# map user locations to dictionary:
'''loc = pd.read_csv('dataset/user_location.csv')
cols = ['user location', 'City', 'State', 'Country']
location = loc['user location'].tolist()
city = loc['City'].tolist()
state = loc['State'].tolist()
country = loc['Country'].tolist()
loc_dict = defaultdict()
for place, ci, st, co in zip(location, city, state, country):
    loc_dict[place] = (ci, st, co) #loc[loc['user location'] == place][cols[1:]]

ratings['user_loc'] = ratings['user location'].apply(lambda x: loc_dict.get(x, (None, None, None)))
ratings[['user city', 'user state', 'user country']] = ratings['user_loc'].apply(pd.Series)
ratings[['user city', 'user state', 'user country']] = pd.DataFrame(ratings['user_loc'].values.tolist())
'''

'''ratings = pd.read_csv('dataset/unique_popular_3k_hotels.csv', index=False)
regions = ratings.groupby('user country').count().sort_values('title', ascending=False)
regions = regions.reset_index()
country_names = regions['user country'][:-1]'''

a = ratings['user country'].unique()
user_country = a[1:-1]
user_country = ['USA', 'Israel', 'Canada', 'Italy', 'Singapore', 'Spain',
       'Australia', 'UK', 'Turkey', 'Russia', 'Greece', 'Austria',
       'Scotland', 'Germany', 'Norway', 'United Arab Emirates', 'Ireland',
       'Denmark', 'Argentina', 'New Zealand', 'Sweden', 'India', 'Romania',
       'France', 'Brazil', 'Netherlands', 'Thailand', 'Finland', 'Belgium',
       'China', 'Puerto Rico', 'South Africa', 'Switzerland', 'Portugal',
       'Malaysia', 'Philippines', 'Indonesia', 'Japan', 'Scoreland',
       'Copenhagen']

msk1 = ratings['user location'].isin(country_names)
msk2 = ratings['user country'].isin(country_names)
major_region_ratings = ratings[msk1|msk2]
regional_ratings = major_region_ratings[['user', 'hotel id', 'ratings']]
major_region_ratings.to_csv('dataset/major_region_ratings.csv', index=False)

# major regions, most popular 3000 hotels, unique user ratings, users with encoded names,
reg = pd.read_csv('dataset/major_region_ratings.csv', index_col=False)
reg.groupby('user country').size()
'''
user country
Argentina                  474
Australia                36675
Austria                    386
Belgium                   1540
Brazil                     509
Canada                   53410
China                     2696
'''
user_split = reg.groupby('user country')
sub_user_geo = [user_split.get_group(x) for x in user_split.groups]

new_df = pd.pivot_table(reg,index=['user'], columns = 'hotel id', values = "ratings").fillna(0)
new_df.to_csv('dataset/pivot_ratings.csv')

major_region_rating_matrix = csr_matrix(new_df)

model_knn = NearestNeighbors(algorithm='kd_tree', n_jobs=-1)
model_knn.fit(major_region_rating_matrix)

# compute cosine similarities the fast way
import sklearn.preprocessing as pp

def cosine_similarities(mat):
    col_normed_mat = pp.normalize(mat.tocsc(), axis=0)
    return col_normed_mat.T * col_normed_mat















if __name__ == '__main__':
    pass
