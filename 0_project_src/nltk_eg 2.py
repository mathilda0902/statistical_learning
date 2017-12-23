import nltk
import re

exampleArray = ['The incredibly intimidating NLP scares people away who are sissies.']

# eg1
def processLanguage(content):
    try:
        for item in content:
            tokenized = nltk.word_tokenize(item)
            tagged = nltk.pos_tag(tokenized)
            print tagged

            chunkGram = r"""
            NP:
                {<.*>+}          # Chunk everything
                }<VBD|IN>+{
            """
            chunkParser = nltk.RegexpParser(chunkGram)
            chunked = chunkParser.parse(tagged)
            print chunked
            chunked.draw()

    except Exception, e:
        print str(e)

processLanguage(exampleArray)

'''read AmazonReviews:'''
import json
from pprint import pprint

data = json.load(open('dataset/AmazonReviews/cameras/143546026X.json'))
content = data['Reviews'][0]['Content']


'''reading .dat files in Review_Texts:'''
import numpy as np

def is_float(string):
    """ True if given string is float else False"""
    try:
        return float(string)
    except ValueError:
        return False

data1 = []
with open('dataset/hotel_72572.dat', 'r') as f:
    d = f.readlines()
    for i in d:
        k = i.rstrip().split(",")
        data1.append([float(i) if is_float(i) else i for i in k])

data1 = np.array(data1, dtype='O')


'''read processed_stars:'''
data2 = []
with open('dataset/processed_stars/books/all_balanced.review', 'r') as f:
    d = f.readlines()
    for i in d:
        k = i.rstrip().split(",")
        data2.append([float(i) if is_float(i) else i for i in k])

data2 = np.array(data2, dtype='O')

'''read tripadvisor_json:'''
data3 = json.load(open('dataset/tripadvisor_json/72572.json'))
reviews = data3['Reviews'][0]
hotel = data3['HotelInfo'][0]
content3 = data3['Reviews'][0]['Content']


# eg2
def processLanguage2(content):
    try:
        for item in content:
            tokenized = nltk.word_tokenize(item)
            tagged = nltk.pos_tag(tokenized)

            nameEnt = nltk.ne_chunk(tagged, binary=True)
            nameEnt.draw()

    except Exception, e:
        print str(e)

content = [str(content3)]
processLanguage2(content)


#eg:
from nltk import sent_tokenize, word_tokenize
from nltk import pos_tag, ne_chunk

exampleArray = 'The incredibly intimidating NLP scares people away who are sissies. Mr and Mrs Hills went up hill.'
sent = sent_tokenize(exampleArray)
print sent

words = word_tokenize(exampleArray)
tagged1 = nltk.pos_tag(words)
print words

words2 = nltk.wordpunct_tokenize(exampleArray)
tagged2 = nltk.pos_tag(words2)



for i in range(1, 20):
    for tweet in twitter.search('#win OR #fail', start=i, count=1000):
        s = tweet.text.lower()
        p = '#win' in s and 'WIN' or 'FAIL'
        v = tag(s)
        v = [word for word, pos in v if pos == 'JJ'] # JJ = aejective
        v = count(v)
        if v:
            knn.train(v, type=p)


print knn.classify('sweet potato burger')
print knn.classify('bitter')

for i in range(10):
    for result in Bing().search('"more important than"', start=i+1, count=50):
        s = r.text.lower()
        s = plaintext(s)
        s = parsetree(s)
        p = '{NP} (VP) more important than {NP}'
        for m in search(p, s):
            x = m.group(1).string
            y = m.group(2).string
            if x not in g:
                g.add_node(x)
            if y not in g:
                g.add_node(y)
            g.add_edge(g[x], g[y], stroke=(0,0,0,0.75))













'''#build knowledge base:
import time
import urllib2
from urllib2 import urlopen
import re
import cookielib
from cookielib import CookieJar
import datetime
import sqlite3

cj = CookieJar()
opener = urllib2.build_opener(urllib2.HTTPCookieProcessor(cj))


conn = sqlite3.connect('knowledgeBase.db')
c = conn.cursor()
'''
'''# one-time function
def createDB():
    c.execute("")


def reutersRSSvisit():
    try:
        page = 'http://feeds.reuters.com/reuters/topNews'
        sourceCode = opener.open(page).read()
        try:
            links = re.findall(r'<link.*href=\"(.*?)\"', sourceCode)
            for link in links:
                if '.rdf' in link:
                    pass
                else:
                    print 'Visiting the link.'
                    print '#####################'
                    linkSource= opener.open(linke).read()
                    linesOfInterest = re.findall(r'<p>(.*?)</p>', str(linkSource))
                    print 'Content: '
                    for eachLine in linesOfInterest:
                        print eachLine

        except Exception, e:
            print "Failed 2nd loop of huffingtonRSS."
            print str(e)

    except Exception, e:
        print "Failed main loop of huffingtonRSS."
        print str(e)

reutersRSSvisit()
'''



if __name__ == '__main__':
    pass
