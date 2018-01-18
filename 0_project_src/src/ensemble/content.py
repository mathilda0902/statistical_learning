import os
from __future__ import print_function
import codecs
import pickle
import json

import numpy as np
import pandas as pd

from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.externals import joblib
from sklearn.manifold import MDS

from scipy.cluster.hierarchy import ward, dendrogram

import nltk
from nltk.stem.snowball import SnowballStemmer
import re
import mpld3

import matplotlib.pyplot as plt
import matplotlib as mpl

#filepath = 'hotel/3k_hotel_reviews.csv'
#def read_reviews(filepath):
#    ratings = pd.read_csv(filepath)
#    reviews = ratings.groupby('hotel name')['content'].apply(list)
#    reviews = reviews.reset_index()
#    hotel_names = reviews['hotel name'].tolist()
#    all_reviews = reviews['content'].tolist()
#    hotel_reviews = []
#    for review in all_reviews:
#        hotel_reviews.append('.'.join(review))
#    return hotel_names, hotel_reviews

# read review data and get the hotel names and review corpus:
def read_reviews(namesfile, reviewsfile):
    with open(namesfile, 'r') as outfile:
        hotel_names = json.load(outfile)
    with open(reviewsfile, 'r', encoding='utf-8', errors='ignore') as outfile:
        hotel_reviews = json.load(outfile)
    return hotel_names, hotel_reviews
hotel_names, hotel_reviews = read_reviews('hotel/hotel_names.txt', 'hotel/hotel_reviews.txt')

hotel_names, hotel_reviews = read_reviews('hotel/3k_hotel_reviews.csv')

with open('hotel/hotel_names.txt', 'w') as outfile:
    json.dump(hotel_names, outfile)

with open('hotel/hotel_reviews.txt', 'w') as outfile:
    json.dump(hotel_reviews, outfile)

with open('hotel/hotel_names.txt', 'r') as outfile:
    hotel_names = json.load(outfile)

with open('hotel/hotel_reviews.txt', 'r') as outfile:
    hotel_reviews = json.load(outfile)

with open('hotel/hotel_vocab_stemmed.txt', 'r') as outfile:
    hotel_vocab_stemmed = json.load(outfile)

with open('hotel/hotel_vocab_tokenized.txt', 'r') as outfile:
    hotel_vocab_tokenized = json.load(outfile)


# load nltk's English stopwords as variable called 'stopwords'
stopwords = nltk.corpus.stopwords.words('english')
stemmer = SnowballStemmer("english")

# define a tokenizer and stemmer which returns the set of stems in the text that it is passed
def tokenize_and_stem(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    stems = [stemmer.stem(t) for t in filtered_tokens]
    return stems

def tokenize_only(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    return filtered_tokens

'''
hotel_vocab_stemmed = []
hotel_vocab_tokenized = []
for i in hotel_reviews[:200]:
    allwords_stemmed = tokenize_and_stem(i)
    hotel_vocab_stemmed.extend(allwords_stemmed)
    allwords_tokenized = tokenize_only(i)
    hotel_vocab_tokenized.extend(allwords_tokenized)
'''

with open('hotel/hotel_vocab_tokenized.txt', 'r') as f:
    hotel_vocab_tokenized = json.load(f)
with open('hotel/hotel_vocab_stemmed.txt', 'r') as f:
    hotel_vocab_stemmed = json.load(f)

hotel_vocab = pd.DataFrame({'words': hotel_vocab_tokenized}, index = hotel_vocab_stemmed)
print ('there are ' + str(hotel_vocab.shape[0]) + ' items in hotel_vocab')
# there are 6188088 items in vocab_frame

user_vocab = pd.DataFrame({'words': user_vocab_tokenized}, index = user_vocab_stemmed)
print 'there are ' + str(user_vocab.shape[0])  + ' items in user_vocab'
# there are 6188088 items in vocab_frame

#define vectorizer parameters
tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=200000,
                                 min_df=0.2, stop_words='english',
                                 use_idf=True, tokenizer=tokenize_and_stem, ngram_range=(1,3))

%time tfidf_matrix = tfidf_vectorizer.fit_transform(hotel_reviews) #fit the vectorizer to reviews
print(tfidf_matrix.shape)
#CPU times: user 58.6 s, sys: 500 ms, total: 59.1 s
#Wall time: 59.4 s
#(100, 2934)

terms = tfidf_vectorizer.get_feature_names()
dist = 1 - cosine_similarity(tfidf_matrix)

joblib.dump(terms, 'hotel/features.pkl')
terms = joblib.load('hotel/features.pkl')


# k-means clustering
num_clusters = 5
km = KMeans(n_clusters=num_clusters)
%time km.fit(tfidf_matrix)
clusters = km.labels_.tolist()


# save model to pickle
joblib.dump(km,  'doc_cluster.pkl')
km = joblib.load('doc_cluster.pkl')

clusters = km.labels_.tolist()


# multi-dimensional scaling
MDS()

# convert two components as we're plotting points in a two-dimensional plane
# "precomputed" because we provide a distance matrix
# we will also specify `random_state` so the plot is reproducible.
mds = MDS(n_components=2, dissimilarity="precomputed", random_state=1)

pos = mds.fit_transform(dist)  # shape (n_components, n_samples)

xs, ys = pos[:, 0], pos[:, 1]


# cluster centroid
print("Top terms per cluster:")
print()
#sort cluster centers by proximity to centroid
order_centroids = km.cluster_centers_.argsort()[:, ::-1]

for i in range(num_clusters):
    print("Cluster %d words:" % i, end='')

    for ind in order_centroids[i, :20]: #replace 6 with n words per cluster
        print(' %s' % vocab_frame.ix[terms[ind].split(' ')].values.tolist()[0][0].encode('utf-8', 'ignore'), end=',')
    print() #add whitespace
    print() #add whitespace

    print("Cluster %d hotels:" % i, end='')
    for title in vocab_frame.ix[i]['words'].values.tolist():
        print(' %s,' % title, end='')
    print() #add whitespace
    print() #add whitespace

print()
print()

'''
Top terms per cluster:

Cluster 0 words: phoenix, motel, suite, dirty, downtown, light, airport, charges, bars, resort, center, hilton, kitchen, wall, wonderful, enjoyed, building, microwave, elevator, pool,

Cluster 1 words: spa, wine, palms, resort, season, beautiful, enjoyed, relax, bath, valley, treatment, wonderful, lodging, hills, beverly, b, romantic, charm, dinner, countries,

Cluster 2 words: best, western, square, space, downtown, waffles, city, indoor, center, enjoyed, stay, market, block, motel, shops, stay, continental, reservation, distance, clerk,

Cluster 3 words: hollywood, la, blvd, beverly, shops, sunset, hills, university, los, motel, los, theatres, studios, metro, angeles, bus, stars, bars, plaza, microwave,

Cluster 4 words: airport, shuttle, phoenix, flight, close, marriott, drivers, hilton, airport, early, shuttle, flying, suite, courtyard, shuttle, rental, bars, picked, free, strip,
'''


# visualizing:
#set up colors per clusters using a dict
cluster_colors = {0: '#1b9e77', 1: '#d95f02', 2: '#7570b3', 3: '#e7298a', 4: '#66a61e'}

#set up cluster names using a dict
cluster_names = {0: 'suite, downtown, light, airport, charges, bars, resort, kitchen, building, microwave, elevator, pool',
                 1: 'spa, wine, palms, relax, bath, valley, treatment, lodging, hills, beverly, romantic, charm, dinner, countries',
                 2: 'space, waffles, city, indoor, market, shops, continental, reservation, clerk',
                 3: 'sunset, hills, university, theatres, studios, metro, bus, plaza',
                 4: 'airport, shuttle, flight, drivers, early, courtyard, rental, picked, free'}

#some ipython magic to show the matplotlib plots inline

#create data frame that has the result of the MDS plus the cluster numbers and titles
df = pd.DataFrame(dict(x=xs, y=ys, label=clusters, title=hotel_names[:100]))

#group by cluster
groups = df.groupby('label')


# set up plot
fig, ax = plt.subplots(figsize=(17, 9)) # set size
ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling

#iterate through groups to layer the plot
#note that I use the cluster_name and cluster_color dicts with the 'name' lookup to return the appropriate color/label
for name, group in groups:
    ax.plot(group.x, group.y, marker='o', linestyle='', ms=12,
            label=cluster_names[name], color=cluster_colors[name],
            mec='none')
    ax.set_aspect('auto')
    ax.tick_params(\
        axis= 'x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom='off',      # ticks along the bottom edge are off
        top='off',         # ticks along the top edge are off
        labelbottom='off')
    ax.tick_params(\
        axis= 'y',         # changes apply to the y-axis
        which='both',      # both major and minor ticks are affected
        left='off',      # ticks along the bottom edge are off
        top='off',         # ticks along the top edge are off
        labelleft='off')

ax.legend(numpoints=1)  #show legend with only 1 point

#add label in x,y position with the label as the film title
for i in range(len(df)):
    ax.text(df.ix[i]['x'], df.ix[i]['y'], df.ix[i]['title'], size=8)



plt.show() #show the plot

#uncomment the below to save the plot if need be
#plt.savefig('clusters_small_noaxes.png', dpi=200)



#define custom toolbar location
class TopToolbar(mpld3.plugins.PluginBase):
    """Plugin for moving toolbar to top of figure"""

    JAVASCRIPT = """
    mpld3.register_plugin("toptoolbar", TopToolbar);
    TopToolbar.prototype = Object.create(mpld3.Plugin.prototype);
    TopToolbar.prototype.constructor = TopToolbar;
    function TopToolbar(fig, props){
        mpld3.Plugin.call(this, fig, props);
    };

    TopToolbar.prototype.draw = function(){
      // the toolbar svg doesn't exist
      // yet, so first draw it
      this.fig.toolbar.draw();

      // then change the y position to be
      // at the top of the figure
      this.fig.toolbar.toolbar.attr("x", 150);
      this.fig.toolbar.toolbar.attr("y", 400);

      // then remove the draw function,
      // so that it is not called again
      this.fig.toolbar.draw = function() {}
    }
    """
    def __init__(self):
        self.dict_ = {"type": "toptoolbar"}


#create data frame that has the result of the MDS plus the cluster numbers and titles
df = pd.DataFrame(dict(x=xs, y=ys, label=clusters, title=hotel_names[:100]))

#group by cluster
groups = df.groupby('label')

#define custom css to format the font and to remove the axis labeling
css = """
text.mpld3-text, div.mpld3-tooltip {
  font-family:Arial, Helvetica, sans-serif;
}

g.mpld3-xaxis, g.mpld3-yaxis {
display: none; }

svg.mpld3-figure {
margin-left: -200px;}
"""

# Plot
fig, ax = plt.subplots(figsize=(14,6)) #set plot size
ax.margins(0.03) # Optional, just adds 5% padding to the autoscaling

#iterate through groups to layer the plot
#note that I use the cluster_name and cluster_color dicts with the 'name' lookup to return the appropriate color/label
for name, group in groups:
    points = ax.plot(group.x, group.y, marker='o', linestyle='', ms=18,
                     label=cluster_names[name], mec='none',
                     color=cluster_colors[name])
    ax.set_aspect('auto')
    labels = [i for i in group.title]

    #set tooltip using points, labels and the already defined 'css'
    tooltip = mpld3.plugins.PointHTMLTooltip(points[0], labels,
                                       voffset=10, hoffset=10, css=css)
    #connect tooltip to fig
    mpld3.plugins.connect(fig, tooltip, TopToolbar())

    #set tick marks as blank
    ax.axes.get_xaxis().set_ticks([])
    ax.axes.get_yaxis().set_ticks([])

    #set axis as blank
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)


ax.legend(numpoints=1) #show legend with only one dot

mpld3.display() #show the plot

#uncomment the below to export to html
#html = mpld3.fig_to_html(fig)
#print(html)



linkage_matrix = ward(dist) #define the linkage_matrix using ward clustering pre-computed distances

fig, ax = plt.subplots(figsize=(5, 7)) # set size
ax = dendrogram(linkage_matrix, orientation="right", labels=hotel_names[:100]);

plt.tick_params(\
    axis= 'x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom='off',      # ticks along the bottom edge are off
    top='off',         # ticks along the top edge are off
    labelbottom='off')

plt.tight_layout() #show plot with tight layout
plt.show()

#uncomment below to save figure
plt.savefig('ward_clusters.png', dpi=200) #save figure as ward_clusters
