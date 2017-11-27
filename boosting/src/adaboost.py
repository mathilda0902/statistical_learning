from boosting import AdaBoostBinaryClassifier
import numpy as np
from sklearn.model_selection import train_test_split
from matplotlib.lines import Line2D
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
import requests
import itertools


'''
Use stage_score_plot to plot the test error curve for AdaBoostClassifier
and GradientBoostingClassifier.
In addition, include two more GradientBoostingClassifier models
where the max_depth argument is 10 and 100 respectively.
'''

'''1. abc:'''
abc = AdaBoostClassifier(base_estimator=None, n_estimators=100,
learning_rate=1.0, algorithm='SAMME.R', random_state=None)
'''2. gdbr_3:
learning_rate
max_depth
min_samples_leaf
max_features
n_estimators'''
gdbc_3 = GradientBoostingClassifier(learning_rate=0.1, max_depth=3,
                            min_samples_leaf=1, max_features=None,
                            n_estimators=100)
'''3. gdbr_10'''
gdbc_10 = GradientBoostingClassifier(learning_rate=0.1, max_depth=10,
                            min_samples_leaf=1, max_features=None,
                            n_estimators=100)
'''4. gdbr_100'''
gdbc_100 = GradientBoostingClassifier(learning_rate=0.1, max_depth=100,
                            min_samples_leaf=1, max_features=None,
                            n_estimators=100)


def staged_misclass(model, train_array, response_array, test_array, test_response):
    est = model.fit(train_array, response_array)
    predictions = list(est.staged_predict(test_array))
    misclass_rate = []
    for i in range(100):
        t = [(a-b) for a, b in zip(test_response, predictions[i])]
        misclass_rate.append(np.mean(t))
    return np.array(misclass_rate)


''' grid search cv GradientBoostingClassifier:'''
gdbc_param_grid = {'n_estimators': [100, 500],
                  'min_samples_leaf': [1, 3, 10],
                  'max_depth': [1, 3, 10],
                  'subsample': [1.0, 0.5],
                  'max_features': ['auto', 'sqrt', 'log2', None],
                  'learning_rate': np.logspace(-2, 0, num = 3),
                  'random_state': [1]}

'''Part 3: feature importances and partial dependency plots'''

gdbc_3 = GradientBoostingClassifier(learning_rate=0.1, min_samples_leaf=10,
n_estimators=500, subsample=1.0, random_state=1,
max_features='sqrt', max_depth=10)

def get_feature_names():
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.names'
    names = requests.get(url)
    names = names.text.split('\r\n')
    names = itertools.ifilter(lambda x: not x.startswith('|'), names)
    names = [str(name.split(':')[0]) for name in names]
    return np.array(list(names)[4:-1])

def bar_plot(feature_names, feature_importances):
    y_ind = np.arange(9, -1, -1) # 9 to 0
    fig = plt.figure(figsize=(8, 8))
    plt.barh(y_ind, feature_importances, height = 0.3, align='center')
    plt.ylim(y_ind.min() + 0.5, y_ind.max() + 0.5)
    plt.yticks(y_ind, feature_names)
    plt.xlabel('Relative feature importances')
    plt.ylabel('Features')
    figname = '3_1_feature_importance_bar_plot.png'
    plt.tight_layout()
    plt.savefig(figname, dpi = 100)
    plt.close()
    print "\n1) {0} plot saved.".format(figname)


if __name__=='__main__':
    data = np.genfromtxt('data/spam.csv', delimiter=',')

    y = data[:, -1]
    X = data[:, 0:-1]

    X_train, X_test, y_train, y_test = train_test_split(X, y)


    '''
    my_ada = AdaBoostBinaryClassifier(n_estimators=50)
    my_ada.fit(X_train, y_train)
    print "Accuracy:", my_ada.score(X_test, y_test)
    '''

    '''
    gdbc_gridsearch = GridSearchCV(GradientBoostingClassifier(), gdbc_param_grid, n_jobs=-1)
    gdbc_gridsearch.fit(X_train, y_train)
    print "Starting grid search - coarse (will take several minutes)"
    gdbc_gridsearch.fit(X_train, y_train)
    gdbc_params = gdbc_gridsearch.best_params_
    gdbc_score = gdbc_gridsearch.best_score_
    print "Coarse search best parameters:"
    for param, val in gdbc_params.iteritems():
        print "{0:<20s} | {1}".format(param, val)
    print "Coarse search best score: {0:0.3f}".format(gdbc_score)
    best_gdbc_model = gdbc_gridsearch.best_estimator_
    print 'best parameters for Gradient Boosting Grid Search: ', gdbc_gridsearch.best_params_
    '''



    '''
    a = staged_misclass(abc, X_train, y_train, X_test, y_test)
    b = staged_misclass(gdbc_3, X_train, y_train, X_test, y_test)
    c = staged_misclass(gdbc_10, X_train, y_train, X_test, y_test)
    d = staged_misclass(gdbc_100, X_train, y_train, X_test, y_test)
    x_axis = np.linspace(0,100,100)
    plt.figure(figsize=(6,4))
    plt.plot(x_axis, a, alpha=0.5, color='b', label='AdaBoost')
    plt.plot(x_axis, b, alpha=0.5, color='g', label='Gradient Boosting, max depth: 3')
    plt.plot(x_axis, c, alpha=0.5, color='r', label='Gradient Boosting, max depth: 10')
    plt.plot(x_axis, d, alpha=0.5, color='turquoise', label='Gradient Boosting, max depth: 100')
    plt.ylabel('Misclassification Rate')
    plt.xlabel('Iterations')
    plt.legend()
    plt.title('Investigate Model Complexity on Test Set')
    plt.show()'''


    model = gdbc_3.fit(X_train, y_train)
    t = model.feature_importances_

    feature_importances = model.feature_importances_
    top10_colindex = np.argsort(feature_importances)[::-1][0:10]
    feature_importances = feature_importances[top10_colindex]
    feature_importances = feature_importances / float(feature_importances.max()) #normalize

    all_feature_names = get_feature_names()
    feature_names = list(all_feature_names[top10_colindex])
    bar_plot(feature_names, feature_importances)

'''
Accuracy: 0.930495221546
'''
'''
Coarse search best parameters:
learning_rate        | 0.1
min_samples_leaf     | 10
n_estimators         | 500
subsample            | 1.0
random_state         | 1
max_features         | sqrt
max_depth            | 10
Coarse search best score: 0.957
best parameters for Gradient Boosting Grid Search:
{'learning_rate': 0.10000000000000001, 'min_samples_leaf': 10,
'n_estimators': 500, 'subsample': 1.0, 'random_state': 1,
'max_features': 'sqrt', 'max_depth': 10}
'''
