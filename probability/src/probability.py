# Part 2
# Question 1
import pandas as pd
from scipy.stats.kde import gaussian_kde
import matplotlib.pyplot as plt
import scipy.stats as sc


df = pd.read_csv("data/admissions.csv")

'''
Out[4]:
   family_income   gpa  parent_avg_age
0        31402.0  3.18              32
1        32247.0  2.98              48
2        34732.0  2.85              61
3        53759.0  3.39              62
4        50952.0  3.10              45
'''

'''
In [5]: df.shape
Out[5]: (10992, 3)'''

# Question 2

# covariance function:
def covariance(x1, x2):
    return np.sum((x1 - np.mean(x1)) * (x2 - np.mean(x2))) / (len(x1) - 1.)

#Check your results
df.cov()

np.cov([df['family_income'], df['gpa'], df['parent_avg_age']], ddof=0)
'''
Out[10]:
array([[  3.32910757e+08,   4.01493379e+03,  -1.22621471e+03],
       [  4.01493379e+03,   8.78831966e-02,  -2.87826412e-02],
       [ -1.22621471e+03,  -2.87826412e-02,   1.12967163e+02]])
'''

df.cov()
'''
Out[7]:
                family_income          gpa  parent_avg_age
family_income    3.329410e+08  4015.299085    -1226.326280
gpa              4.015299e+03     0.087891       -0.028785
parent_avg_age  -1.226326e+03    -0.028785      112.977442
'''

# Question 3
# correlation function:
def correlation(x1, x2):
    std_prod = np.std(x1) * np.std(x2)
    covar = covariance(x1, x2)
    return covar / std_prod

#Check your results
df.corr()

In [11]: df.corr()
'''
Out[11]:
                family_income       gpa  parent_avg_age
family_income        1.000000  0.742269       -0.006323
gpa                  0.742269  1.000000       -0.009135
parent_avg_age      -0.006323 -0.009135        1.000000
'''

# Question 4
max_income = df['family_income'].max()
df['family_income_cat']=pd.cut(np.array(df['family_income']), [0,26832,37510,max_income],
                                    labels=["low","medium","high"])

# Get the conditional distribution of GPA given an income class
low_income_gpa = df[df['family_income_cat'] == 'low'].gpa
medium_income_gpa = df[df['family_income_cat'] == 'medium'].gpa
high_income_gpa = df[df['family_income_cat'] == 'high'].gpa

# Plot the distributions

def plot_smooth(gpa_samp, label):
    my_pdf = gaussian_kde(gpa_samp)
    x = np.linspace(min(gpa_samp) , max(gpa_samp))
    plt.plot(x, my_pdf(x), label=label)



# Part 3
# Question 1
import scipy.stats as scs
df2 = pd.read_csv('data/admissions_with_study_hrs_and_sports.csv')
'''
In [56]: df2.head()
Out[56]:
   family_income   gpa family_income_cat  parent_avg_age  hrs_studied  \
0        31402.0  3.18            medium              32    49.463745
1        32247.0  2.98            medium              48    16.414467
2        34732.0  2.85            medium              61     4.937079
3        53759.0  3.39              high              62   160.210286
4        50952.0  3.10            medium              45    36.417860

   sport_performance
0           0.033196
1           0.000317
2           0.021845
3           0.153819
4           0.010444
'''

# Question 2
plt.scatter(df2['gpa'], df2['hrs_studied'], alpha=.01, edgecolor='none')
slope, intercept, r_value, p_value, std_err = sc.linregress(df2['gpa'], df2['hrs_studied'])
plt.plot(df2['gpa'], slope * df2['gpa'] + intercept, color='r', alpha=.4)
plt.xlabel('GPA', fontsize=14, fontweight='bold')
plt.ylabel('Hours Studied', fontsize=14, fontweight='bold')

# Question 3
sc.pearsonr(df2['gpa'], df2['hrs_studied'])
sc.spearmanr(df2['gpa'], df2['hrs_studied'])
'''
The spearman correlation shows a more positive coefficient since it
captures the non-linear relationship.'''

sc.pearsonr(df2['gpa'], df2['sport_performance'])
sc.spearmanr(df2['gpa'], df2['sport_performance'])
# There is a strong relationship between gpa and sports perf. , but the values of the
# two variables are not monotonically increasing together. Therefore, the coefficients are low

'''Distribution simulation'''
# Define number of sales to be a uniform from 5000 to 6000
sales = sc.uniform(5000, 1000)
# Define conversion percent as a binomial distribution
conversion = sc.binom
# Profit PMF
profit_ = sc.binom


def simulate_sales():
    sales_draw = sales.rvs()
    conversion_draw = conversion(sales_draw, 0.12).rvs()
    wholesale_proportion = profit_(conversion_draw, .2).rvs()
    profit = conversion_draw * wholesale_proportion * 50
    return  profit + (conversion_draw-wholesale_proportion)*60




if __name__ == '__main__':
    print covariance(df.family_income, df.gpa)
    print covariance(df.family_income, df.parent_avg_age)
    print covariance(df.gpa, df.parent_avg_age)

    print correlation(df.family_income, df.gpa)
    print correlation(df.family_income, df.parent_avg_age)
    print correlation(df.gpa, df.parent_avg_age)

    fig = plt.figure(figsize=(12, 5))
    plot_smooth(low_income_gpa, 'low income')
    plot_smooth(medium_income_gpa, 'medium income')
    plot_smooth(high_income_gpa, 'high income')
    plt.xlabel('GPA', fontsize=14, fontweight='bold')
    plt.ylabel('Probability Density', fontsize=14, fontweight='bold')
    plt.legend()

    # The 90th percentile GPA for each class
    print '90th percentile GPA for low income class', np.percentile(low_income_gpa, 90)
    print '90th percentile GPA for medium income class', np.percentile(medium_income_gpa, 90)
    print '90th percentile GPA for high income class', np.percentile(high_income_gpa, 90)

    print sc.pearsonr(df2['gpa'], df2['hrs_studied'])
    print sc.spearmanr(df2['gpa'], df2['hrs_studied'])
    print 'The spearman correlation shows a more positive coefficient since it captures the non-linear relationship.'

    print sc.pearsonr(df2['gpa'], df2['sport_performance'])
    print sc.spearmanr(df2['gpa'], df2['sport_performance'])
    print 'There is s strong relationship between gpa and sports perf., but the values of thetwo variables are not monotonically increasing together. Therefore, the coefficients are low.'

    dist = [simulate_sales() for _ in range(10000)]
    plt.hist(dist)
    plt.xlabel('Profit', fontsize=14, fontweight='bold')
    plt.ylabel('Freq.', fontsize=14, fontweight='bold')
    print '2.5% percentile', np.percentile(dist, 2.5)
    print '97.5% percentile', np.percentile(dist, 97.5)
    # 2.5% percentile 33750.0
    # 97.5% percentile 42930.0

























1
