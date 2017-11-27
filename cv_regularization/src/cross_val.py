from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.datasets import load_boston
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

boston = load_boston()
X = boston.data
y = boston.target

'''Part 2'''

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

'''
Write a function rmse(true, predicted) that takes your true and predicted values
and calculates the RMSE. You should use sklearn.metrics.mean_squared_error() to
confirm your results.
'''
def rmse(true, predicted):
    diff = true - predicted
    return np.sqrt((diff.dot(diff))/float(len(true)))

linear = LinearRegression()
linear.fit(X_train, y_train)

# Call predict to get the predicted values for training and test set
train_predicted = linear.predict(X_train)
test_predicted = linear.predict(X_test)

'''
RMSE for training set  4.4101049988639929
RMSE for test set  5.5896366433351163
MSE for training set 19.449026101005177
MSE for test set 31.244037804514665
Model will be biased to the data set that we selected for training purpose.
Need real unseen data to test the true predictability of fitted model.
'''

'''Part 3'''
def crossVal(folds, X, y):
    test_mse = 0
    kf = KFold(n_splits=folds, shuffle=True)
    fold = 0
    sk_mse_all = []
    for train_index, test_index in kf.split(X):
        fold += 1
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        test_mse += (rmse(y_test, y_pred) ** 2)
        sk_mse = mean_squared_error(y_pred, y_test)
        sk_mse_all.append(sk_mse)
    return test_mse / float(folds), sk_mse_all.mean()

'''Checking with cross_val_score'''
'''model = LinearRegression()
cv_result = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
b=[-a for a in cv_result]
sum(b)/5'''

'''
In [44]: a = cross_validate(model, X, y, cv=5, scoring='n
    ...: eg_mean_squared_error')

In [45]: a
Out[45]:
{'fit_time': array([ 0.00161195,  0.0011518 ,  0.00118017,  0.0009551 ,  0.00071716]),
 'score_time': array([ 0.00046897,  0.0003562 ,  0.0003159 ,  0.00023794,  0.00023699]),
 'test_score': array([-12.48065021, -26.09620267, -33.11995587, -80.83305378, -33.58435565]),
 'train_score': array([-24.58896264, -22.24206903, -21.18704266, -12.91745434, -22.73718934])}

In [46]: x=a['test_score']

In [47]: b=[-i for i in x]

In [48]: b
Out[48]:
[12.480650212260288,
 26.096202669732676,
 33.119955872692486,
 80.833053778700943,
 33.584355652305646]

In [49]: sum(b)/5
Out[49]: 37.222843637138411

In [50]: x=a['train_score']

In [51]: b=[-i for i in x]

In [52]: sum(b)/5
Out[52]: 20.734543603202525

'''


'''plot'''
# input: cv as a range of k folds
def plot_learning_curve(estimator, label=None):
    ''' Plot learning curve with varying training sizes'''
    scores = list()
    train_sizes = np.linspace(10,100,10).astype(int)
    for train_size in train_sizes:
        cv_shuffle = cross_validation.ShuffleSplit(train_size=train_size,
                                test_size=200, n=len(y), random_state=0)
        test_error = cross_validation.cross_val_score(estimator, X, y, cv=cv_shuffle)
    scores.append(test_error)

    plt.plot(train_sizes, np.mean(scores, axis=1), label=label or estimator.__class__.__name__)
    plt.ylim(0,1)
    plt.title('Learning Curve')
    plt.ylabel('Explained variance on test set (R^2)')
    plt.xlabel('Training test size')
    plt.legend(loc='best')
    plt.show()


'''Part 4 stepwise selection'''
def adj_r2(X, predictors, r2):
	sampleSize = len(X)
	num = (1-r2)*(sampleSize - 1)
	den = sampleSize - predictors - 1
	return 1 - (num/den)

def rfe():
	score = np.zeros(len(colNames)+1)
	for i in xrange(1, len(colNames)+1):
		est = LinearRegression()
		selector = RFE(est, i).fit(X_train, y_train)
		score[i] = adj_r2(X_train, i, selector.score(X_test, y_test))
	plt.figure()
	plt.plot(np.arange(1, len(score)+1), score)
	plt.title('Adjusted R^2 vs. Feature Count')
	plt.xlabel('Number of features')
	plt.ylabel('Adjusted R^2')
	plt.legend()
	plt.show()


class forward_select(object):
    def __init__(self, verbose=True):
        self.verbose = verbose
        self.best_score = 0.
        self.column_names = []
        self.keep = []
        self.X_orig = None
        self.X = None
        self.y = None

    def fit_transform(self, df, target):
        self.y = target
        self.column_names = df.columns.tolist()
        self.X_orig = df.values
        self.X = np.ones([self.X_orig.shape[0],1])

        while self.X_orig.shape[1] > 0:
            scores = []
            for feature in xrange(self.X_orig.shape[1]):
                X_temp = np.concatenate((self.X, self.X_orig[:, feature, None]), axis=1)
                scores.append(sm.OLS(self.y, X_temp).fit().rsquared_adj)
            best_idx = np.argmax(np.asanyarray(scores))

            if scores[best_idx] <= self.best_score:
                if self.verbose:
                	print 'Removed columns ->', self.column_names
                	print '-> All done!'
                return self.X[:,1:]
            else:
                self.X = np.concatenate((self.X, self.X_orig[:, best_idx, None]), axis=1)
                self.X_orig = np.delete(self.X_orig, best_idx, axis=1)
                self.keep.append(self.column_names.pop(best_idx))
                self.best_score = scores[best_idx]
                if self.verbose: print 'Kept \'%s\' for a best score of %s' % (self.keep[-1], self.best_score)

def gen_forward():
	df = pd.DataFrame(X, columns=colNames)
	forward = forward_select(verbose=True)
	forward.fit_transform(df, y)




if __name__ == '__main__':
    print 'RMSE for training set ', rmse(train_predicted, y_train)
    print 'RMSE for test set ', rmse(test_predicted, y_test)
    print 'MSE for training set', mean_squared_error(train_predicted, y_train)
    print 'MSE for test set', mean_squared_error(test_predicted, y_test)
    print 'RMSE for 5-folds test ', crossVal(5, X, y)
    print 'Decrease in RMSE with 5-folds ', rmse(test_predicted, y_test) - crossVal(5, X, y)
