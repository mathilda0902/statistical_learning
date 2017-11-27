import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import scipy.stats as scs
from pandas.tools.plotting import scatter_matrix
from statsmodels.stats.outliers_influence import variance_inflation_factor
import plotly.plotly as py
import plotly.graph_objs as go

'''Part 1'''

prestige = sm.datasets.get_rdataset("Duncan", "car", cache=True).data

credit_card = sm.datasets.ccard.load_pandas().data

'''Question 1'''
'''scatter matrices'''
def scatter_m(data):
    scatter_matrix(data, alpha=0.6, figsize=(6,6), diagonal='kde')


scatter_m(prestige)
plt.savefig('scatter_m_prestige')
plt.show()

scatter_m(credit_card)
plt.savefig('scatter_m_cc')
plt.show()

'''fitting models'''
y_p = prestige['prestige']
x_p = prestige[['income', 'education']].astype(float)
x_p = sm.add_constant(x_p)

model_p = sm.OLS(y_p, x_p).fit()
summary_p = model_p.summary()

'''<class 'statsmodels.iolib.summary.Summary'>
"""
                            OLS Regression Results
==============================================================================
Dep. Variable:               prestige   R-squared:                       0.828
Model:                            OLS   Adj. R-squared:                  0.820
Method:                 Least Squares   F-statistic:                     101.2
Date:                Tue, 14 Nov 2017   Prob (F-statistic):           8.65e-17
Time:                        18:12:34   Log-Likelihood:                -178.98
No. Observations:                  45   AIC:                             364.0
Df Residuals:                      42   BIC:                             369.4
Df Model:                           2
Covariance Type:            nonrobust
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
const         -6.0647      4.272     -1.420      0.163     -14.686       2.556
income         0.5987      0.120      5.003      0.000       0.357       0.840
education      0.5458      0.098      5.555      0.000       0.348       0.744
==============================================================================
Omnibus:                        1.279   Durbin-Watson:                   1.458
Prob(Omnibus):                  0.528   Jarque-Bera (JB):                0.520
Skew:                           0.155   Prob(JB):                        0.771
Kurtosis:                       3.426   Cond. No.                         163.
==============================================================================

Warnings:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
"""'''

'''
Model interpretation:
The linear model of prestige = beta_0 + beta_1 * income + beta_2 * education
has a relatively high R^2, 0.828. The fitted model is:
prestige = -6.0647 + 0.5987 * income + 0.5458 * education
This means that for each increase in income, given other factors constant,
the prestige value increase by 0.5987. And for every increase in education value,
given other factors constant, the prestige score increases by 0.5458.
p values for both slope coefficient estimates are significant (less than 0.05, if we
set alpha = 0.05). This means that we reject the null hypothesis of each of these
slope estimates equal to zero. We conclude that the slopes of income and of education
are statistically significant and do not equal zero.
This indicates that both income and education are linear to the prestige score.
However, the intercept has a larger p-value (0.163), thus is not significantly
different from 0.
We can also come to the same conclusion when looking at the coefficient confidence
intervals. ( 0.357, 0.840) and (0.348, 0.744) each excludes zero. We reject the null
hypothesis as stated above.
'''


y_c = credit_card['AVGEXP']
x_c = credit_card[['AGE', 'INCOME', 'INCOMESQ', 'OWNRENT']].astype(float)
x_c = sm.add_constant(x_c)

model_c = sm.OLS(y_c, x_c).fit(fit_intercept=True)
summary_c = model_c.summary()
'''
<class 'statsmodels.iolib.summary.Summary'>
"""
                            OLS Regression Results
==============================================================================
Dep. Variable:                 AVGEXP   R-squared:                       0.244
Model:                            OLS   Adj. R-squared:                  0.198
Method:                 Least Squares   F-statistic:                     5.394
Date:                Tue, 14 Nov 2017   Prob (F-statistic):           0.000795
Time:                        18:12:04   Log-Likelihood:                -506.49
No. Observations:                  72   AIC:                             1023.
Df Residuals:                      67   BIC:                             1034.
Df Model:                           4
Covariance Type:            nonrobust
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
const       -237.1465    199.352     -1.190      0.238    -635.054     160.761
AGE           -3.0818      5.515     -0.559      0.578     -14.089       7.926
INCOME       234.3470     80.366      2.916      0.005      73.936     394.758
INCOMESQ     -14.9968      7.469     -2.008      0.049     -29.906      -0.088
OWNRENT       27.9409     82.922      0.337      0.737    -137.573     193.455
==============================================================================
Omnibus:                       69.024   Durbin-Watson:                   1.640
Prob(Omnibus):                  0.000   Jarque-Bera (JB):              497.349
Skew:                           2.844   Prob(JB):                    1.00e-108
Kurtosis:                      14.551   Cond. No.                         227.
==============================================================================

Warnings:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
"""
'''

'''
This model is a poor fit firstly due to the low R^2 (0.244).
Seconldy, only one of the 4 slope estimates and intercept has a significantly small
p-value. The INCOME is the only variable that has a slope that is significantly
different from zero. We reject the null hypothesis of the slope being identical to zero,
and conclude that the INCOME variable is of linear relation to the AVGEXP, our dependent
variable. The confidence interval of this coefficient also supports this conclusion.
However, the intercept, and the three other slope estimates are not so much different
from zero. Except maybe the INCOMESQ, which has a p-value very close to our significant level,
if we set alpha = 0.05. All the other three (constant intercept, slope of AGE, slope of
OWNRENT) have p-values that are much larger than 0.05 (0.238, 0.578, 0.737, respectively).
Therefore, we cannot reject the null hypothesis to each that states that these estimates
are significantly different than zero. We conclude that the AGE and the OWNRENT variables
are not of significant linear relations to our dependent variable, AVGEXP.
'''

'''Question 2'''
p_res = model_p.outlier_test()['student_resid']
y_p_pred = model_p.predict(x_p)

c_res = model_c.outlier_test()['student_resid']
y_c_pred = model_c.predict(x_c)

'''plotting'''
plt.scatter(y_p_pred, p_res, color='b', alpha=0.6, edgecolor=None)
plt.xlim(-20, 120)
plt.ylim(-3, 4)
plt.axhline(0, color='r', linestyle='dashed')
plt.xlabel('Predicted Prestige', fontsize=10, weight='bold')
plt.ylabel('Studentized Residuals', fontsize=10, weight='bold')
plt.tight_layout()
plt.savefig('residual_prestige')
plt.show()

plt.scatter(y_c_pred, c_res, color='b', alpha=0.6, edgecolor=None)
plt.xlim(0, 700)
plt.ylim(-2, 8)
plt.axhline(0, color='r', linestyle='dashed')
plt.xlabel('Predicted Average Expenditure', fontsize=10, weight='bold')
plt.ylabel('Studentized Residuals', fontsize=10, weight='bold')
plt.tight_layout()
plt.savefig('residual_credit_card')
plt.show()


'''Question 3'''
'''
The credit_card dataset has a more obvious trend of heteroscedasticity in the way
that residuals spread out with 'predicted average expenditure' increases. When the
'predicted average expenditure' is small, the residuals seem to cram together.
When the 'predicted average expenditure' gets large, the residuals seem to spread out,
instead of remaining evenly randomly of both sides of zero.
The prestige dataset however has a very nicely randomly residual plot, in which
the residuals evenly spread out on both sides of zero line.
'''

'''????????diagnostic.HetGoldfeldQuandt'''

'''We use the studentized residuals as opposed to the outright residuals, because this
looks at how extreme a residual is after accounting for the standard error of the
residuals.  Where simply comparing does not account for this.'''

'''If we are doing inferential statistics, the
assumption of equal variance is used to assure the accuracy of
statements regarding 95% confidence intervals (similar with
hypothesis testing).'''

'''Question 4'''
y_c_log = np.log(credit_card['AVGEXP'])
model_c_log = sm.OLS(y_c_log, x_c).fit(fit_intercept=True)
summary_c_log = model_c_log.summary()

'''
<class 'statsmodels.iolib.summary.Summary'>
"""
                            OLS Regression Results
==============================================================================
Dep. Variable:                 AVGEXP   R-squared:                       0.280
Model:                            OLS   Adj. R-squared:                  0.237
Method:                 Least Squares   F-statistic:                     6.501
Date:                Tue, 14 Nov 2017   Prob (F-statistic):           0.000176
Time:                        21:13:22   Log-Likelihood:                -98.157
No. Observations:                  72   AIC:                             206.3
Df Residuals:                      67   BIC:                             217.7
Df Model:                           4
Covariance Type:            nonrobust
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
const          3.6601      0.686      5.332      0.000       2.290       5.030
AGE           -0.0243      0.019     -1.279      0.205      -0.062       0.014
INCOME         0.7741      0.277      2.797      0.007       0.222       1.326
INCOMESQ      -0.0467      0.026     -1.814      0.074      -0.098       0.005
OWNRENT        0.3357      0.286      1.176      0.244      -0.234       0.906
==============================================================================
Omnibus:                        6.271   Durbin-Watson:                   1.901
Prob(Omnibus):                  0.043   Jarque-Bera (JB):                5.592
Skew:                          -0.546   Prob(JB):                       0.0611
Kurtosis:                       3.819   Cond. No.                         227.
==============================================================================

Warnings:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
"""
'''



c_log_res = model_c_log.outlier_test()['student_resid']
y_c_log_pred = model_c_log.predict(x_c)

plt.scatter(y_c_log_pred, c_log_res, color='b', alpha=0.6, edgecolor=None)
plt.xlim(3.5, 7)
plt.ylim(-4, 4)
plt.axhline(0, color='r', linestyle='dashed')
plt.xlabel('Predicted Log Average Expenditure', fontsize=10, weight='bold')
plt.ylabel('Studentized Residuals', fontsize=10, weight='bold')
plt.tight_layout()
plt.savefig('log_residual_credit_card')
plt.show()


'''Question 5'''
def fit_qq(res):
    fig = sm.qqplot(res, scs.norm, line='45', fit=True)

qq_prestige = fit_qq(p_res)
plt.savefig('qq_plot_prestige')
plt.show()

qq_cc = fit_qq(c_res)
plt.savefig('qq_plot_credit_card')
plt.show()

qq_c_log = fit_qq(c_log_res)
plt.savefig('qq_plot_log_credit_card')
plt.show()

'''
qq plot for the log AVGEXP shows a much better fit for the residuals to the normal
distribution. The Jarque-Bera test statistic tests the null that the data is
normally distributed against an alternative that the data follow some other
distribution. We notice an increase in the JB probability from the test summary,
1.00e-108 to 0.0611, indicating a decrease in the confidence of rejecting the null.
This means that with log transformation of the dependent variable, AVGEXP, we
fail to reject the null hypothesis of JB test, and conclude that the residuals
are normally distributed.
'''


'''Question 6
Checking multicollinearity.VIF's
Multicollinearity is when we have x-variables that are correlated with one another.
Commonly, if multicollinearity is present, then we might have cases where two x-variables
have a positive relationship with a response, but because they are related to one another,
when both are placed in the same linear model, we might see a negative coefficient on
one of the x-variables, when it truly should have a positive relationship with the response.
'''


def vifs(x):
	'''
	Input x as a DataFrame, calculates the vifs for each variable in the DataFrame.
	DataFrame should not have response variable.
	Returns dictionary where key is column name and value is the vif for that column.
	Requires scipy.stats be imported as scs
	'''
    vifs = []
    for index in range(x.shape[1]):
	       vifs.append(round(variance_inflation_factor(x.values, index),2))
    return vifs

vifs(x_c)
'''[35.29, 1.36, 16.33, 15.21, 1.43]'''
vifs(x_p)
'''[4.59, 2.1, 2.1]'''


'''Part 2'''
'''Question 1'''
'''
dataset name:
    prestige
variables:
    y_p = prestige['prestige']
    x_p = prestige[['income', 'education']]
fitted model:
    model_p
predicted values:
    y_p_pred = model_p.predict(x_p)
residuals:
    p_res = model_p.outlier_test()['student_resid']
'''
'''importing plotly offline'''
from plotly import __version__
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot


trace = go.Scatter(
    x = y_p_pred,
    y = p_res,
    mode = 'markers',
    name = 'markers'
)

data = [trace]
py.image.save_as({'data':data}, 'scatter_plot', format='png')
#py.iplot(data, filename='scatter-mode')
