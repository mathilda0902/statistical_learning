# https://github.com/gSchool/dsi-solns-g55/tree/master/nlp
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

vect = TfidfVectorizer(sublinear_tf=True, max_df=0.5, analyzer='word', stop_words='english')
vect.fit(corpus)
corpus_tf_idf = vect.transform(corpus)
vectors = corpus_tf_idf.toarray()

def get_top_values(lst, n, labels):
    '''
    INPUT: LIST, INTEGER, LIST
    OUTPUT: LIST

    Given a list of values, find the indices with the highest n values.
    Return the labels for each of these indices.
    e.g.
    lst = [7, 3, 2, 4, 1]
    n = 2
    labels = ["cat", "dog", "mouse", "pig", "rabbit"]
    output: ["cat", "pig"]
    '''
    return [labels[i] for i in np.argsort(lst)[-1:-n-1:-1]]

avg = np.sum(vectors, axis=0) / np.sum(vectors > 0, axis=0)
get_top_values(avg, 10, vect.get_feature_names())

'''
[u'buildind',
 u'neer',
 u'brekfest',
 u'macyes',
 u'assets',
 u'experiencing',
 u'ppl',
 u'difference',
 u'recomend',
 u'definitley',
 u'furniture',
 u'challenge',
 u'private',
 u'definetly',
 u'upgraded',
 u'royal',
 u'rec',
 u'seeing',
 u'stops',
 u'daily',
 u'year',
 u'ice',
 u'gentleman',
 u'engine',
 u'workdesk',
 u'cute',
 u'helpfull',
 u'cosy',
 u'refurbishment',
 u'church',
 u'choices',
 u'carbs',
 u'winter',
 u'spreads',
 u'entry',
 u'cozy',
 u'restraurant',
 u'fishy',
 u'penalty',
 u'solo',
 u'female',
 u'10min',
 u'traveller',
 u'surrounding',
 u'esb',
 u'general',
 u'stare',
 u'neck',
 u'ordinary',
 u'enabled',
 u'unremarkable',
 u'wrenching',
 u'proportional',
 u'rip',
 u'numbers',
 u'thanksgiving',
 u'untold',
 u'abandoned',
 u'reported',
 u'economy',
 u'outrageous',
 u'ritz',
 u'home',
 u'invalid',
 u'washroom',
 u'dresser',
 u'endtables',
 u'dated',
 u'sea',
 u'suggest',
 u'frontdesk',
 u'incompetent',
 u'needless',
 u'fabulous',
 u'unforgettable',
 u'inconveniences',
 u'satisfaction',
 u'william',
 u'willingness',
 u'pool',
 u'resolve',
 u'centrally',
 u'ideal',
 u'facilities',
 u'mailed',
 u'spots',
 u'favorite',
 u'maintained',
 u'lived',
 u'sides',
 u'recently',
 u'nrf',
 u'conference',
 u'refurbished',
 u'attending',
 u'stores',
 u'charged',
 u'letter',
 u'allows',
 u'heart']

'''
