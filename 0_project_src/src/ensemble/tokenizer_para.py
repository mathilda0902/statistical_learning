import csv
import pandas as pd
import sklearn.preprocessing as pp
from multiprocessing import Pool
from random import shuffle, choice
import nltk
from nltk.stem.snowball import SnowballStemmer


def read_reviews(namesfile, reviewsfile):
    with open(namesfile, 'r') as outfile:
        hotel_names = json.load(outfile)
    with open(reviewsfile, 'r', encoding='utf-8', errors='ignore') as outfile:
        hotel_reviews = json.load(outfile)
    return hotel_names, hotel_reviews

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

def tfidf(mat):
	tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=200000,
                                 min_df=0.2, stop_words='english',
                                 use_idf=True, tokenizer=tokenize_and_stem, ngram_range=(1,3))
	tfidf_matrix = tfidf_vectorizer.fit_transform(mat)
	return tfidf_matrix.shape

if __name__ == '__main__':
	pool = Pool(processes=5)

	with open('hotel_vocab_tokenized.txt', 'r') as f:
	    hotel_vocab_tokenized = json.load(f)
	with open('hotel_vocab_stemmed.txt', 'r') as f:
	    hotel_vocab_stemmed = json.load(f)

	stemmer = SnowballStemmer("english")
	stopwords = nltk.corpus.stopwords.words('english')

	hotel_names, hotel_reviews = read_reviews('hotel_names.txt', 'hotel_reviews.txt')
	combined = list(zip(hotel_names, hotel_reviews))
	random.shuffle(combined)
	hotel_names[:], hotel_reviews[:] = zip(*combined)

	reviews_grouped = [hotel_reviews[i:i + 100] for i in range(0, len(l), 100)]

	print (pool.map(tfidf, review_grouped))
