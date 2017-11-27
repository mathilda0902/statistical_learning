import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import scipy.stats as scs
from pandas.plotting import scatter_matrix

'''Question 1 - 3'''
df = pd.read_csv('data/balance.csv')
def scatter_m(data):
    scatter_matrix(data, alpha=0.6, figsize=(6,6), diagonal='kde')

scatter_m(df[['Income', 'Limit', 'Rating', 'Cards', 'Age', 'Education', 'Gender',
            'Student', 'Married', 'Ethnicity', 'Balance']])
plt.savefig('balance_scatter_matrix')
plt.show()

df.head()
'''   Unnamed: 0   Income  Limit  Rating  Cards  Age  Education  Gender Student  \
0           1   14.891   3606     283      2   34         11    Male      No
1           2  106.025   6645     483      3   82         15  Female     Yes
2           3  104.593   7075     514      4   71         11    Male      No
3           4  148.924   9504     681      3   36         11  Female      No
4           5   55.882   4897     357      2   68         16    Male      No

  Married  Ethnicity  Balance
0     Yes  Caucasian      333
1     Yes      Asian      903
2      No      Asian      580
3      No      Asian      964
4     Yes  Caucasian      331
'''

df_gender = pd.get_dummies(df['Gender'])
#df_gender.drop('Male', axis=1, inplace=True)

df_student = pd.get_dummies(df['Student'], prefix='Student')
df_married = pd.get_dummies(df['Married'], prefix='Married')
df_eth = pd.get_dummies(df['Ethnicity'])
df_new = pd.concat([df, df_gender, df_student, df_married, df_eth], axis=1)
df_new.drop(['Gender', 'Student', 'Married', 'Ethnicity', 'African American',
            'Student_No', 'Married_No'],
        axis=1, inplace=True)
df_new = df_new.applymap(np.int)

'''Question 4'''
x = df_new[['Income', 'Limit', 'Rating', 'Cards', 'Age', 'Education',
            'Female', 'Student_Yes', 'Married_Yes', 'Asian', 'Caucasian']].astype(float)
result = sm.OLS(df_new['Balance'], sm.add_constant(x)).fit()

result.summary()
# <class 'statsmodels.iolib.summary.Summary'>
"""
                            OLS Regression Results
==============================================================================
Dep. Variable:                Balance   R-squared:                       0.955
Model:                            OLS   Adj. R-squared:                  0.954
Method:                 Least Squares   F-statistic:                     749.9
Date:                Tue, 14 Nov 2017   Prob (F-statistic):          1.23e-253
Time:                        22:58:35   Log-Likelihood:                -2398.8
No. Observations:                 400   AIC:                             4822.
Df Residuals:                     388   BIC:                             4869.
Df Model:                          11
Covariance Type:            nonrobust
===============================================================================
                  coef    std err          t      P>|t|      [0.025      0.975]
-------------------------------------------------------------------------------
const        -483.7704     35.803    -13.512      0.000    -554.163    -413.378
Income         -7.7937      0.234    -33.302      0.000      -8.254      -7.334
Limit           0.1903      0.033      5.804      0.000       0.126       0.255
Rating          1.1427      0.491      2.327      0.020       0.177       2.108
Cards          17.7790      4.342      4.095      0.000       9.242      26.316
Age            -0.6126      0.294     -2.083      0.038      -1.191      -0.034
Education      -1.0304      1.598     -0.645      0.520      -4.173       2.112
Female        -10.6962      9.917     -1.079      0.281     -30.193       8.801
Student_Yes   425.4984     16.727     25.438      0.000     392.612     458.385
Married_Yes    -8.3440     10.366     -0.805      0.421     -28.724      12.036
Asian          16.2895     14.123      1.153      0.249     -11.477      44.056
Caucasian      10.0089     12.213      0.820      0.413     -14.003      34.021
==============================================================================
Omnibus:                       35.031   Durbin-Watson:                   1.965
Prob(Omnibus):                  0.000   Jarque-Bera (JB):               41.958
Skew:                           0.784   Prob(JB):                     7.74e-10
Kurtosis:                       3.242   Cond. No.                     3.87e+04
==============================================================================

Warnings:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
[2] The condition number is large, 3.87e+04. This might indicate that there are
strong multicollinearity or other numerical problems.
"""

model_res = result.outlier_test()['student_resid']
balance_pred = result.predict(sm.add_constant(x))

plt.scatter(balance_pred, model_res, color='b', alpha=0.6, edgecolor=None)
plt.axhline(0, color='r', linestyle='dashed')
plt.xlabel('Predicted Balance', fontsize=10, weight='bold')
plt.ylabel('Studentized Residuals', fontsize=10, weight='bold')
plt.xlim(-500, 2000)
plt.ylim(-4,5)
plt.tight_layout()
plt.savefig('residual_balance')
plt.show()

'''Question 5'''
plt.hist(df['Balance'], bins=100, normed=1)
plt.xlabel('Balance', weight=20)
plt.ylabel('Probablity Density', weight=20)
plt.savefig('balance_histogram')
plt.show()

'''Question 6'''
features = ['Income', 'Limit', 'Rating', 'Cards', 'Age', 'Education',
            'Female', 'Student_Yes', 'Married_Yes', 'Asian', 'Caucasian']

for f in features:
    df.plot(kind='scatter', y='Balance', x=f, edgecolor='none',
            figsize=(12, 5))
    plt.show()

'''Limit and Rating affect zero-Balance and non-zero-Balance'''
df.plot(kind='scatter', y='Balance', x='Limit', edgecolor='none',
        figsize=(12, 5))
plt.savefig('balance_limit')
plt.show()

df.plot(kind='scatter', y='Balance', x='Rating', edgecolor='none',
        figsize=(12, 5))
plt.savefig('balace_rating')
plt.show()

'''Question 7'''
'''Decision: Limit >= 3000 & Rating >= 220'''

'''Question 8: Remove the data points below the decided threshold of your chosen
variable and examine the number of zero observations that remain.'''
'''Cleaning zero balances'''
mask_limit = df_new['Limit'] >= 3000
mask_rating = df_new['Rating'] >= 220
# df_new['Balance'][(df_new['Limit'] >= 3000) & (df_new['Rating'] >= 220)]
df_step2 = df_new.query('Limit >= 3000 and Rating >= 220')

'''       Unnamed: 0      Income         Limit      Rating       Cards  \
count  303.000000  303.000000    303.000000  303.000000  303.000000
mean   201.551155   51.468647   5590.574257  412.108911    2.986799
std    117.585575   37.743622   1973.883212  132.793987    1.418827
min      1.000000   10.000000   3000.000000  224.000000'''


'''Question 9'''
x2 = df_step2[['Income', 'Limit', 'Rating', 'Cards', 'Age', 'Education',
            'Female', 'Student_Yes', 'Married_Yes', 'Asian', 'Caucasian']].astype(float)
result2 = sm.OLS(df_step2['Balance'], sm.add_constant(x2)).fit()
result2.summary()

# <class 'statsmodels.iolib.summary.Summary'>
"""
                            OLS Regression Results
==============================================================================
Dep. Variable:                Balance   R-squared:                       0.996
Model:                            OLS   Adj. R-squared:                  0.995
Method:                 Least Squares   F-statistic:                     5939.
Date:                Tue, 14 Nov 2017   Prob (F-statistic):               0.00
Time:                        23:35:03   Log-Likelihood:                -1437.9
No. Observations:                 303   AIC:                             2900.
Df Residuals:                     291   BIC:                             2944.
Df Model:                          11
Covariance Type:            nonrobust
===============================================================================
                  coef    std err          t      P>|t|      [0.025      0.975]
-------------------------------------------------------------------------------
const        -665.6484     12.057    -55.211      0.000    -689.377    -641.919
Income         -9.5495      0.076   -126.276      0.000      -9.698      -9.401
Limit           0.3144      0.011     29.221      0.000       0.293       0.336
Rating          0.0367      0.160      0.230      0.819      -0.278       0.351
Cards          25.1984      1.371     18.382      0.000      22.500      27.896
Age            -1.0907      0.097    -11.251      0.000      -1.282      -0.900
Education       0.0185      0.521      0.035      0.972      -1.008       1.045
Female         -2.3982      3.280     -0.731      0.465      -8.853       4.057
Student_Yes   493.6077      5.578     88.494      0.000     482.630     504.586
Married_Yes    -3.8309      3.458     -1.108      0.269     -10.636       2.974
Asian           3.6271      4.707      0.771      0.442      -5.637      12.891
Caucasian       0.0699      4.015      0.017      0.986      -7.833       7.973
==============================================================================
Omnibus:                      326.813   Durbin-Watson:                   1.912
Prob(Omnibus):                  0.000   Jarque-Bera (JB):            13009.221
Skew:                           4.642   Prob(JB):                         0.00
Kurtosis:                      33.729   Cond. No.                     4.42e+04
==============================================================================

Warnings:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
[2] The condition number is large, 4.42e+04. This might indicate that there are
strong multicollinearity or other numerical problems.
"""

'''residual plot'''
model_res2 = result2.outlier_test()['student_resid']
balance_pred2 = result2.predict(sm.add_constant(x2))

plt.scatter(balance_pred2, model_res2, color='b', alpha=0.6, edgecolor=None)
plt.axhline(0, color='r', linestyle='dashed')
plt.xlabel('Predicted Balance', fontsize=10, weight='bold')
plt.ylabel('Studentized Residuals', fontsize=10, weight='bold')
plt.xlim(-500, 2500)
plt.ylim(-6,6)
plt.tight_layout()
plt.savefig('selected_residual_balance')
plt.show()
