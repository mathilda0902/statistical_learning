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

































if __name__ == '__main__':
    pass
