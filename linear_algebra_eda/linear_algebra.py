import numpy as np
import matplotlib.pyplot as plt
'''Part 1.1'''
'''Question 1'''
s_2004 = np.array([0.25, 0.2, 0.55])
A = np.array([[0.7, 0.1, 0], [0.2, 0.9, 0.2], [0.1, 0, 0.8]])
s_2009 = np.dot(A, s_2004)
s_2014 = np.dot(A, s_2009)

'''result'''
'''In [9]: s_2009
Out[9]: array([ 0.195,  0.34 ,  0.465])

In [10]: s_2014
Out[10]: array([ 0.1705,  0.438 ,  0.3915])
'''

'''Part 1.2'''
'''Question 1, 2'''
from sklearn import datasets
import matplotlib.pyplot as plt
slw = datasets.load_iris().data[:, :2]
sl = slw[:, 0]
sw = slw[:, 1]
mean_vec = np.mean(slw, axis=0)

plt.scatter(sw, sl, color='b')
plt.plot(mean_vec[1], mean_vec[0], color='r', marker='x', markersize=20)
plt.xlabel('Sepal Width')
plt.ylabel('Sepal Length')
plt.savefig('iris_scatter')
plt.show()


'''Question 3'''
def euclidean_dist(col1, col2):
    if np.allclose(col1, col1.flatten()) and np.allclose(col2, col2.flatten()):
        if col1.shape != col2.shape:
            raise ValueError
        else:
            c = abs(col1 - col2)
            return np.sqrt(np.dot(c, c))

'''Question 4'''
def cosine_sim(col1, col2):
    if np.allclose(col1, col1.flatten()) and np.allclose(col2, col2.flatten()):
        if col1.shape != col2.shape:
            raise ValueError
        else:
            num = np.dot(col1, col2)
            len_1 = np.dot(col1, col1)
            len_2 = np.dot(col2, col2)
            return num/np.sqrt(len_1 * len_2)

'''Question 5'''
def compute_dist(data, metric):
    if metric == euclidean_dist:
        output = []
        for i in range(150):
            output.append(metric(data[i], mean_vec))
        return np.array(output).reshape(-1,1)
    elif metric == cosine_sim:
        output = []
        for i in range(150):
            output.append(metric(data[i], mean_vec))
        return np.array(output).reshape(-1,1)

euclidean_dists = compute_dist(slw, euclidean_dist)
cosine_sims = compute_dist(slw, cosine_sim)

'''In [163]: euclidean_dists
Out[163]:
array([[ 0.86686818],
       [ 0.94487765],
       [ 1.1526175 ],
       [ 1.24418398],
       [ 1.00465273],
       [ 0.95512326],...])'''

'''In [164]: cosine_sims
Out[164]:
array([[ 0.99282608],
       [ 0.99770423],
       [ 0.99326126],
       [ 0.99380121],
       [ 0.98987505],...])'''

'''
In [167]: euclidean_dists.shape
Out[167]: (150, 1)

In [168]: cosine_sims.shape
Out[168]: (150, 1)'''

'''Question 6'''
'''histograms for Euclidean distances and cosine similarities between
the (lengths, widths) and the mean vector'''
plt.hist(euclidean_dists, bins=20, normed=1)
plt.xlim(0, 2.5)
plt.ylim(0, 1.4)
plt.xlabel('Euclidean Distances')
plt.ylabel('Probability Density')
plt.savefig('hist_euclidean')
plt.show()

plt.hist(cosine_sims, bins=20, normed=1)
plt.xlim(0.98, 1)
plt.ylim(0, 350)
plt.xlabel('Cosine Similarities')
plt.ylabel('Probability Density')
plt.savefig('hist_cosine')
plt.show()



'''Extra credit'''
