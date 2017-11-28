import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
import logistic_regression_functions as f
from GradientDescent import GradientDescent
import numpy as np
X, y = make_classification(n_samples=100,
                            n_features=2,
                            n_informative=2,
                            n_redundant=0,
                            n_classes=2,
                            random_state=0)


gd = GradientDescent(f.cost, f.gradient, f.predict)
gd.fit(X, y)
print "coeffs: ", gd.coeffs
predictions = gd.predict(X, gd.coeffs, thresh=0.5)
print 'predictions: ', predictions
plt.scatter(X[:, 0], X[:, 1], alpha=0.5, c=y>0)
plt.show()

'''regr = LogisticRegression()
regr.fit(X[:, 0:2], y)
print regr.coef_
'''
