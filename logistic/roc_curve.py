from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
import statsmodels.formula.api as sm
from statsmodels.discrete.discrete_model import Logit
from statsmodels.tools import add_constant

'''Part 1'''
def get_key(item):
    return item[0]

def roc_curve(probs, labels, pos_label):
    sample = zip(probs, labels)
    sample_new = sorted(sample, key=get_key, reverse=True)
    tpr = []
    fpr = []
    th = []
    pos_class = len([q for q in labels if q == pos_label])
    neg_class = len(labels) - pos_class
    for index, s in enumerate(sample_new):
        th.append(s[0])
        c1 = 0
        c2 = 0
        for i in range(index+1):
            if sample_new[i][1] == pos_label:
                c1 += 1
            else:
                c2 += 1
        tpr.append(c1/float(pos_class))
        fpr.append(c2/float(neg_class))
    return fpr, tpr, th

X, y = make_classification(n_features=2, n_redundant=0, n_informative=2,
                           n_clusters_per_class=2, n_samples=1000)
X_train, X_test, y_train, y_test = train_test_split(X, y)

model = LogisticRegression()
model.fit(X_train, y_train)
probs = model.predict_proba(X_test)[:, 1]

fpr, tpr, thresholds = roc_curve(probs, y_test, 1)

fpr, tpr, thresholds = metrics.roc_curve(y_test, probs, pos_label=1)

plt.plot(fpr, tpr)
plt.xlabel("False Positive Rate (1 - Specificity)")
plt.ylabel("True Positive Rate (Sensitivity, Recall)")
plt.title("ROC plot of fake data")
plt.savefig('metrics_roc_fake_data')
plt.show()

def fitting(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    model = LogisticRegression()
    model.fit(X_train, y_train)
    probs = model.predict_proba(X_test)[:, 1]


'''Question 3'''
df = pd.read_csv('data/loanf.csv')
y = (df['Interest.Rate'] <= 12).values
y = y * 1
X = df[['FICO.Score', 'Loan.Length', 'Loan.Amount']].values


fitting(X, y)
fpr, tpr, thresholds = roc_curve(probs, y_test, 1)

fpr, tpr, thresholds = metrics.roc_curve(y_test, probs, pos_label=1)

a = np.linspace(0,1,100)
plt.plot(fpr, tpr)
plt.plot(a, a)
plt.xlabel("False Positive Rate (1 - Specificity)")
plt.ylabel("True Positive Rate (Sensitivity, Recall)")
plt.title("ROC plot of Interest Rate")
plt.savefig('metrics_roc_interest')
plt.show()


'''Part 2'''
grad_school = pd.read_csv('data/grad.csv')
pd.crosstab(grad_school['admit'], grad_school['rank'])
grad_school['rank'].plot(kind='bar')
df = grad_school.set_index('rank')
x = []
for i in range(1,5):
    x.append(sum(df.loc[i]['admit']) / float(len(df.loc[i]['admit'])))

plt.bar(range(1,5), x)
plt.xlabel('Rank')
plt.ylabel('Percent of Admitted')
plt.savefig('percent_rank_bar')
plt.show()

plt.hist(grad_school['gpa'])
plt.savefig('hist_gpa')
plt.show()

plt.hist(grad_school['gre'])
plt.savefig('hist_gre')
plt.show()

'''Question 1'''
grad_school = grad_school.reset_index()
y = grad_school['admit']
X = grad_school[['gre', 'gpa', 'rank']]
X_constant = add_constant(X, prepend=True)

model = Logit(y, X_constant).fit()
model.summary()

'''Question 2'''
"""
                           Logit Regression Results
==============================================================================
Dep. Variable:                  admit   No. Observations:                  400
Model:                          Logit   Df Residuals:                      396
Method:                           MLE   Df Model:                            3
Date:                Sun, 19 Nov 2017   Pseudo R-squ.:                 0.08107
Time:                        13:03:31   Log-Likelihood:                -229.72
converged:                       True   LL-Null:                       -249.99
                                        LLR p-value:                 8.207e-09
==============================================================================
                 coef    std err          z      P>|z|      [0.025      0.975]
------------------------------------------------------------------------------
const         -3.4495      1.133     -3.045      0.002      -5.670      -1.229
gre            0.0023      0.001      2.101      0.036       0.000       0.004
gpa            0.7770      0.327      2.373      0.018       0.135       1.419
rank          -0.5600      0.127     -4.405      0.000      -0.809      -0.311
==============================================================================
"""

'''Question 3'''
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
logit_model = LogisticRegression()
logit_model = logit_model.fit(X_train, y_train)
logit_model.score(X_test, y_test)
y_pred = logit_model.predict(X_test)

metrics.accuracy_score(y_test, y_pred)
metrics.recall_score(y_test, y_pred)
metrics.precision_score(y_test, y_pred)

'''
In [93]: metrics.accuracy_score(y_test, y_pred)
Out[93]: 0.68000000000000005

In [94]: metrics.recall_score(y_test, y_pred)
Out[94]: 0.17647058823529413

In [95]: metrics.precision_score(y_test, y_pred)
Out[95]: 0.59999999999999998
'''

'''Question 4'''
grad_school_rank = pd.get_dummies(grad_school['rank'], prefix='rank')
grad_school_rank = grad_school_rank.drop(['rank_1'], axis=1)
grad_school = grad_school.drop(['rank'], axis=1)
gs_new = pd.concat([grad_school, grad_school_rank], axis=1)
X_new = gs_new[['gre', 'gpa', 'rank_2', 'rank_3', 'rank_4']]

X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.25)
logit_model = LogisticRegression()
logit_model = logit_model.fit(X_train, y_train)
logit_model.score(X_test, y_test)
y_pred = logit_model.predict(X_test)

'''Question 5'''
metrics.accuracy_score(y_test, y_pred)
metrics.recall_score(y_test, y_pred)
metrics.precision_score(y_test, y_pred)

'''In [116]: metrics.accuracy_score(y_test, y_pred)
Out[116]: 0.69999999999999996
In [117]: metrics.recall_score(y_test, y_pred)
Out[117]: 0.31034482758620691
In [118]: metrics.precision_score(y_test, y_pred)
Out[118]: 0.47368421052631576
'''

'''Question 6'''
probs = logit_model.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(probs, y_test, 1)

a = np.linspace(0, 1, 50)
plt.plot(fpr, tpr)
plt.plot(a,a)
plt.xlabel("False Positive Rate (1 - Specificity)")
plt.ylabel("True Positive Rate (Sensitivity, Recall)")
plt.title("ROC plot of Grad School Admission")
plt.savefig('metrics_roc_grad_school')
plt.show()

'''Part 4'''
'''Question 1'''
y = grad_school['admit']
X = grad_school[['gre', 'gpa', 'rank']]

model_odds = sm.Logit(y, X)
result2 = model_odds.fit()
result2.summary()

"""
                           Logit Regression Results
==============================================================================
Dep. Variable:                  admit   No. Observations:                  400
Model:                          Logit   Df Residuals:                      397
Method:                           MLE   Df Model:                            2
Date:                Thu, 16 Nov 2017   Pseudo R-squ.:                 0.06176
Time:                        22:31:39   Log-Likelihood:                -234.55
converged:                       True   LL-Null:                       -249.99
                                        LLR p-value:                 1.971e-07
==============================================================================
                 coef    std err          z      P>|z|      [0.025      0.975]
------------------------------------------------------------------------------
gre            0.0015      0.001      1.420      0.155      -0.001       0.004
gpa           -0.0042      0.201     -0.021      0.983      -0.398       0.390
rank          -0.6695      0.121     -5.527      0.000      -0.907      -0.432
==============================================================================
"""

''' For gre, 1-unit change in feature will result in an odds ratio of:
exp(0.0015) = 1.0015
1-unit change in feature, gpa, will result in an oddds ratio of:
exp(-0.0042) = 0.99581
1-unit change in feature, rank, will result in an odds ratio of:
exp(-0.6695) = 0.51196
'''

'''
Increasing gre by 1 point increases the odds by a factor of 1.0015.
Increasing gpa by 1 point increases the odds by a factor of 0.99581.
Increasing rank by 1 point increases the odds by a factor of 0.51196.
'''

'''
ln(2) / 0.0015 = 462.098. Increasing gre by 462.098 will double my odds.
ln(2) / (-0.0042) = -165.035. Decreasing gpa by
'''

'''Part 5'''
'''Question 1'''
gre_m = grad_school['gre'].mean()
gpa_m = grad_school['gpa'].mean()

feature_m = pd.DataFrame({'gre': gre_m * np.ones(4),
                        'gpa': gpa_m * np.ones(4),
                         'rank': np.arange(1,5)})
fm = feature_m[['gre', 'gpa', 'rank']]

'''Question 2'''
model5 = LogisticRegression()
model5 = logit_model.fit(X, y)

fpred= model5.predict_proba(fm)
p_vec = [b for a,b in fpred]
odds = [b/(1-b) for b in p_vec]

output = np.vstack((fm['rank'], p_vec, odds)).T
for a in output:
    print 'rank: {}, probability: {}, odds: {}'. format(int(a[0]), round(a[1], 6), round(a[2], 6))

'''
rank: 1, probability: 0.518633, odds: 1.077417
rank: 2, probability: 0.370328, odds: 0.588129
rank: 3, probability: 0.243022, odds: 0.321042
rank: 4, probability: 0.149115, odds: 0.175247
'''

'''Quetion 3'''
plt.scatter(fm['rank'], p_vec)
plt.ylim(0,1)
plt.xlabel('Rank')
plt.ylabel('Probability')
plt.savefig('rank_probs')
plt.show()

plt.scatter(fm['rank'], odds)
plt.ylim(0,1)
plt.xlabel('Rank')
plt.ylabel('Odds')
plt.savefig('rank_odds')
plt.show()

plt.scatter(fm['rank'], odds)
plt.yscale('log')
plt.xlabel('Rank')
plt.ylabel('Odds')
plt.savefig('rank_log_odds')
plt.show()






if __name__ == '__main__':
    pass
