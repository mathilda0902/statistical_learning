from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from roc import plot_roc

# from sklearn.datasets import make_classification

df = pd.read_csv('../data/churn.csv')

d = {'no': False, 'yes': True, 'False.': False, 'True.': True}

col_sel = df.select_dtypes(include=['object'])
'''
     State     Phone Int'l Plan VMail Plan  Churn?
0       KS  382-4657         no        yes  False.
1       OH  371-7191         no        yes  False.
2       NJ  358-1921         no         no  False.
3       OH  375-9999        yes         no  False.
4       OK  330-6626        yes         no  False.
'''

col_sel = ["Int'l Plan", 'VMail Plan', 'Churn?']

df_churn = pd.concat([df.select_dtypes(exclude=['object']), df["Int'l Plan"].map(d),
            df['VMail Plan'].map(d), df['Churn?'].map(d)], axis=1)

y = df_churn.pop('Churn?').values
feature_names = df_churn.columns
X = df_churn.values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
y_test = y_test.astype('int64')
y_train = y_train.astype('int64')

'''fitting random forest with default arguments'''
rf = RandomForestClassifier()
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

acc_score = accuracy_score(y_test, y_pred)

conf_m = confusion_matrix(y_test, y_pred)

prec_score = precision_score(y_test, y_pred)

rec_score = recall_score(y_test, y_pred)

'''
Accuracy score for random forest:  0.953523238381
Precision score:  0.944444444444
Recall score:  0.715789473684
Confusion matrix:  [[568   4]
 [ 27  68]]
'''

'''change oob to True. compare out-of-bag training accurary score to test set'''
rf_oob = RandomForestClassifier(n_estimators = 20, oob_score=True)
rf_oob.fit(X_train, y_train)
acc_score_oob = rf_oob.oob_score_

''' comparing oob score to test accuracy score, with n_estimators=20:
Accuracy score for random forest:  0.937031484258
Out-of-bag score:  0.932108027007
'''

'''feature importance'''


''' higher, the more important
Feature importances:  [ 0.0318187   0.01019233  0.03312049  0.14704773  0.03478957  0.12286013
  0.08679852  0.02708882  0.05627226  0.04064982  0.03213551  0.03575091
  0.03750813  0.05287414  0.04655295  0.10589015  0.07941634  0.0192335 ]
  '''

'''
In [12]: feature_names
Out[12]:
Index([u'Account Length', u'Area Code', u'VMail Message', u'Day Mins',
       u'Day Calls', u'Day Charge', u'Eve Mins', u'Eve Calls', u'Eve Charge',
       u'Night Mins', u'Night Calls', u'Night Charge', u'Intl Mins',
       u'Intl Calls', u'Intl Charge', u'CustServ Calls', u'Int'l Plan',
       u'VMail Plan'],
      dtype='object')
'''

importances = rf.feature_importances_
imp_features = feature_names[importances > np.mean(importances)]

'''
Important features:  Index([u'Day Mins', u'Day Charge', u'Eve Mins', u'Eve Charge',
       u'CustServ Calls', u'Int'l Plan'],
      dtype='object')
'''

'''13. Calculate the standard deviation for feature importances across all trees'''

n = 10 # top 10 features

#importances = forest_fit.feature_importances_[:n]
importances = rf.feature_importances_[:n]
std = np.std([tree.feature_importances_ for tree in rf.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]
features = list(df.columns[indices])

# Print the feature ranking
print("\n13. Feature ranking:")

for f in range(n):
    print("%d. %s (%f)" % (f + 1, features[f], importances[indices[f]]))


'''graph relations between # of trees and accuracy score'''
list_accuracy_score = []
x_axis = range(5, 50, 5)
for t in x_axis:
    tot = 0
    for i in xrange(5):
        rf = RandomForestClassifier(n_estimators=t)
        rf.fit(X_train, y_train)
        tot += rf.score(X_test, y_test)
    list_accuracy_score.append(tot/5)

'''Try modifying the max features parameter'''
num_features = range(1, len(X_train[0]) + 1)
accuracies = []
for n in num_features:
    tot = 0
    for i in xrange(5):
        rf = RandomForestClassifier(max_features=n)
        rf.fit(X_train, y_train)
        tot += rf.score(X_test, y_test)
    accuracies.append(tot / 5)


''' Run all the other classifiers that we have learned so far in class'''

def get_scores(classifier, X_train, X_test, y_train, y_test, **kwargs):
    model = classifier(**kwargs)
    model.fit(X_train, y_train)
    y_predict = model.predict(X_test)
    return model.score(X_test, y_test), \
           precision_score(y_test, y_predict), \
           recall_score(y_test, y_predict)


'''roc curves'''
logit_model = LogisticRegression()
logit_model.fit(X_train, y_train)

'''With a fpr of 0.2, we should prefer RandomForest model over logistic regression,
because RandomForest has a higher tpr of 0.85 to 0.64.'''


if __name__ == '__main__':
    print "Accuracy score for random forest: ", acc_score
    print "Precision score: ", prec_score
    print "Recall score: ", rec_score
    print "Confusion matrix: ", conf_m
    print "Out-of-bag score: ", acc_score_oob
    print "Feature importances: ", importances
    print "Important features: ", imp_features

    plot_roc(X_train, y_train, RandomForestClassifier, n_estimators=20)
    plot_roc(X_train, y_train, LogisticRegression)
    plt.show()

    # Plot the feature importances of the forest
    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(10), importances[indices], yerr=std[indices], color="r", align="center")
    plt.xticks(range(10), indices)
    plt.xlim([-1, 10])
    plt.savefig('13_Feature_ranking.png')
    plt.close()
    print('\nPlotted 13) feature importances')

    plt.plot(num_features, accuracies)
    plt.savefig('15_accuracy_vs_num_features.png')
    plt.close()

    plt.plot(range(5, 50, 5), list_accuracy_score)
    plt.show()

    print "\n16. Model, Accuracy, Precision, Recall"
    print "    Random Forest:", get_scores(RandomForestClassifier, X_train, X_test, y_train, y_test, n_estimators=25, max_features=5)
    print "    Logistic Regression:", get_scores(LogisticRegression, X_train, X_test, y_train, y_test)
    print "    Decision Tree:", get_scores(DecisionTreeClassifier, X_train, X_test, y_train, y_test)
    print "    Naive Bayes:", get_scores(MultinomialNB, X_train, X_test, y_train, y_test)
