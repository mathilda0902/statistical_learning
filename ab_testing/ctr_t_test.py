'''Part 1
# question 1
# two sample comparison of means
# two sample comparison of proportions
question 2
'''

# Part 2
import pandas as pd
import scipy.stats as scs
df = pd.read_csv('data/nyt1.csv')
df_ctr = df.drop(df['Impressions'] == 0)

'''In [20]: df.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 458441 entries, 0 to 458440
Data columns (total 5 columns):
Age            458441 non-null int64
Gender         458441 non-null int64
Impressions    458441 non-null int64
Clicks         458441 non-null int64
Signed_In      458441 non-null int64
dtypes: int64(5)
memory usage: 17.5 MB'''

'''In [19]: df_ctr.info()
<class 'pandas.core.frame.DataFrame'>
Int64Index: 458439 entries, 2 to 458440
Data columns (total 5 columns):
Age            458439 non-null int64
Gender         458439 non-null int64
Impressions    458439 non-null int64
Clicks         458439 non-null int64
Signed_In      458439 non-null int64
dtypes: int64(5)
memory usage: 21.0 MB'''

df_ctr['CTR'] = df_ctr.Clicks/df_ctr.Impressions.astype(float)

# question 4
def plot_hist(df, title, color):
    df.hist(figsize=(12, 5), sharey=True, grid=False, color=color, alpha=0.5)
    plt.suptitle(title, size=18, weight='bold')
    plt.tight_layout()
    plt.show()

plot_hist(df, 'Click Through Rate Data', 'g')

# 5
df_ctr_signed_in = df_ctr[df_ctr['Signed_In'] == 1]
df_ctr_not_signed = df_ctr[df_ctr['Signed_In'] == 0]

# v1 -- a quick and dirty approach
df.groupby('Signed_In').hist(alpha=0.5)
plt.show()

# v2 -- a more satisfying plot
fig, axs = plt.subplots(2, 3, figsize=(12, 5))
for col_name, ax in zip(df_signed_in.columns, axs.flatten()):
    bins = np.linspace(df[col_name].min(), df[col_name].max(), 20)
    ax.hist(df_signed_in[col_name], bins=bins, alpha=0.5,
            normed=1, label="Signed In", color='g')
    ax.hist(df_not_signed_in[col_name], bins=bins, alpha=0.5,
            normed=1, label="Not Signed In", color='b')
    ax.set_title(col_name)
    ax.legend(loc='best')

plt.tight_layout()
plt.show()

# 5
'''not signed in group has ages of only 0!'''
scs.ttest_ind(df_ctr_signed_in['CTR'].dropna(), df_ct
    ...: r_not_signed['CTR'].dropna(), equal_var = False)
'''Out[88]: Ttest_indResult(statistic=-55.37570800349517, pvalue=0.0)'''

def plot_t_test(group_1_df, group_2_df, group_1_name, group_2_name):
    fig = plt.figure()
    group_1_mean = group_1_df['CTR'].mean()
    group_2_mean = group_2_df['CTR'].mean()

    print '%s Mean CTR: %s' % (group_1_name, group_1_mean)
    print '%s Mean CTR: %s' % (group_2_name, group_2_mean)
    print 'diff in mean:', abs(group_2_mean-group_1_mean)
    p_val = stats.ttest_ind(group_1_df['CTR'], group_2_df['CTR'], equal_var=False)[1]
    print 'p value is:', p_val

    group_1_df['CTR'].hist(normed=True, label=group_1_name, color='g', alpha=0.3)
    group_2_df['CTR'].hist(normed=True, label=group_2_name, color='r', alpha=0.3)
    plt.axvline(group_1_mean, color='r', alpha=0.6, lw=2)
    plt.axvline(group_2_mean, color='g', alpha=0.6, lw=2)

    plt.ylabel('Probability Density')
    plt.xlabel('CTR')
    plt.legend()
    plt.grid('off')
    plt.show()

plot_t_test(df_ctr_signed_in, df_ctr_not_signed_in, 'Signed In', 'Not Signed In')

# 6
'''since the p value is close to 0, the means of the two groups are significantly differnt.'''
df_male = df_ctr_signed_in[df_ctr_signed_in['Gender'] == 1]
df_female = df_ctr_signed_in[df_ctr_signed_in['Gender'] == 0]

scs.ttest_ind(df_male['CTR'].dropna(), df_female['CTR'].dropna(), equal_var = True)

plot_t_test(df_male, df_female, 'M', 'F')

'''Out[99]: Ttest_indResult(statistic=-3.2931510934680692,
pvalue=0.00099081984898803936)'''

'''since p value is less than alpha = 0.05, we reject the null
hypothesis and conclude that the male group and the female group
have significantly different mean CTR's.
'''

# 7
df_ctr_signed_in['age_groups'] = pd.cut(df_ctr_signed_in['Age'],
                                    [7, 18, 24, 34, 44, 54, 64, 1000],
                                    include_lowest=True)

df_ctr_signed_in['age_groups'].value_counts().sort_index().plot(kind='bar',
                                                            grid=False)
plt.xlabel('Age Group')
plt.ylabel('Number of users')
plt.tight_layout()
plt.show()
