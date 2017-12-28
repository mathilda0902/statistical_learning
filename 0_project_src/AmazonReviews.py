'''AmazonReviews:
laptops: 2,474
TVs: 2,366'''
import json

# ProductID: 1440490899
ar_lap1 = json.load(open('dataset/AmazonReviews/laptops/1440490899.json'))
ar_lap1.keys()
'''
[u'Reviews', u'ProductInfo']
'''
len(ar_lap1['Reviews'])
'''
10
'''
rev1 = ar_lap1['Reviews']
rev1[0]
'''
{u'Author': u'Walt Baty "Walt 0000"',
 u'Content': u'Google is on to something here, but without a little luck, I would not have known about it. I now own 2 chrome books (one for my wife and one for me) and find the experience delightful. In trying to learn and become more productive with it, I have searched the web for how-to articles, etc. Quite by accident, I came across "The Chrome Book." It is well written, very informative, and a real good overview of the "operating system." It also treats some of the higher level uses.This was my best find yet, and one that I am buying as a reference book. I just hope Google keeps going with this concept with enhancements, improvements, and more advertising. Again, well written book and on target!',
 u'Date': u'January 10, 2012',
 u'Overall': u'5.0',
 u'ReviewID': u'RUSGOFLB5XRWR',
 u'Title': u'The most underrated and un-hyped new system'}'''

 rev1[0].keys()
'''[u'Author', u'ReviewID', u'Overall', u'Content', u'Title', u'Date']'''

prod1 = ar_lap1['ProductInfo']
prod1.keys()
'''[u'Price', u'ProductID', u'Features', u'ImgURL', u'Name']'''

# ProductID: B00A0Z3LDI
ar_lap2 = json.load(open('dataset/AmazonReviews/laptops/B00A0Z3LDI.json'))
len(ar_lap2['Reviews'])
'''2'''












































if __name__ == '__main__':
    pass
