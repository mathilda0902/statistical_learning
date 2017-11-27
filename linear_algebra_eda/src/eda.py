import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set(color_codes=True)

'''Part 1'''
'''Question 1'''
df = pd.read_csv('data/201402_trip_data.csv', parse_dates=['start_date', 'end_date'])

df['month'] = df['start_date'].dt.month
df['dayofweek'] = df['start_date'].dt.dayofweek
df['date'] = df['start_date'].dt.date
df['hour'] = df['start_date'].dt.hour

'''df.head()
   trip_id  duration          start_date             start_station  \
0     4576        63 2013-08-29 14:13:00  South Van Ness at Market
1     4607        70 2013-08-29 14:42:00        San Jose City Hall
2     4130        71 2013-08-29 10:16:00   Mountain View City Hall
3     4251        77 2013-08-29 11:29:00        San Jose City Hall
4     4299        83 2013-08-29 12:02:00  South Van Ness at Market

   start_terminal            end_date               end_station  end_terminal  \
0              66 2013-08-29 14:14:00  South Van Ness at Market            66
1              10 2013-08-29 14:43:00        San Jose City Hall            10
2              27 2013-08-29 10:17:00   Mountain View City Hall            27
3              10 2013-08-29 11:30:00        San Jose City Hall            10
4              66 2013-08-29 12:04:00            Market at 10th            67

   bike_# subscription_type zip_code  month  dayofweek        date  hour
0     520        Subscriber    94127      8          3  2013-08-29    14
1     661        Subscriber    95138      8          3  2013-08-29    14
2      48        Subscriber    97214      8          3  2013-08-29    10
3      26        Subscriber    95060      8          3  2013-08-29    11
4     319        Subscriber    94103      8          3  2013-08-29    12

'''

num_users = df.groupby('month')['date'].count()
num_users.plot.bar()
plt.savefig('number_of_users_by_month')
plt.show()

'''observations: no data for Mar, Apr, May, Jun, Jul. Few for Aug.'''

'''Question 3'''
mask = df['month'] >= 9
daily_users = df[mask][['date', 'hour', 'month']].groupby('date').count()
daily_users = daily_users.reset_index()
m = [a.month for a in daily_users['date']]
daily_users['month'] = m
'''
         date  hour  month
0  2013-09-01   706      9
1  2013-09-02   661      9
2  2013-09-03   597      9
3  2013-09-04   606      9
4  2013-09-05   677      9'''

'''calculate mean and sigma:'''
mu = daily_users['hour'].mean()
sig = np.std(daily_users['hour'], ddof=1)

'''plot'''

plt.figure(figsize=(12,5))
col = ['b', 'g', 'r', 'c']
mon = [9, 10, 11, 12]
mask = daily_users['month']
for i in range(4):
    plt.plot(daily_users['date'][mask == mon[i]], daily_users['hour'][mask == mon[i]],
        markevery=range(len(daily_users['date'][mask == mon[i]])), marker='o', color=col[i])
for t in ['2013-09-05', '2013-09-19', '2013-10-03', '2013-10-17', '2013-10-31', '2013-11-14', '2013-11-28', '2013-12-12', '2013-12-26']:
    plt.axvline(t, linestyle='dotted')
for t in [400, 600, 800, 1000, 1200]:
    plt.axhline(t, linestyle='dotted')
plt.axhline(mu, c='k', ls='solid')
plt.axhline(mu + 1.5*sig, c='k', ls='dashed')
plt.axhline(mu - 1.5*sig, c='k', ls='dashed')
plt.xlim('2013-09-01', '2013-12-31')
plt.xlabel('Month')
plt.xticks(rotation=90)
plt.ylim(200, 1400)
plt.ylabel('Number of Users')
plt.savefig('daily_sept_dec')
plt.tight_layout()
plt.show()


'''Question 4'''
daily_users_all_mon = df[['date', 'hour', 'month']].groupby('date').count()
daily_users_all_mon = daily_users_all_mon.reset_index()
m = [a.month for a in daily_users_all_mon['date']]
daily_users_all_mon['month'] = m

'''plotting'''
sns.set_style('whitegrid')
sns.kdeplot(daily_users_all_mon['hour'])
plt.hist(daily_users_all_mon['hour'], bins=15, normed=1)
plt.xlim(0, 1400)
plt.ylim(0, 0.0030)
plt.xlabel('Number of Users')
plt.ylabel('Probability Density')
plt.savefig('user_prob_den_kde')
plt.legend()
plt.tight_layout()
plt.show()

'''weekend vs weekday'''

dow = [a.weekday() for a in daily_users_all_mon['date']]
daily_users_all_mon['dayofweek'] = dow
mask_weekend = daily_users_all_mon['dayofweek'] > 4
mask_weekday = daily_users_all_mon['dayofweek'] < 5
df_weekend = daily_users_all_mon[mask_weekend]
df_weekday = daily_users_all_mon[mask_weekday]

plt.figure(figsize=(6,4))
sns.set_style('whitegrid')
sns.kdeplot(df_weekend['hour'], color='g', label='')
sns.kdeplot(df_weekday['hour'], color='b', label='')
plt.hist(df_weekend['hour'], bins=15, normed=1, label='weekend', color='g', alpha=0.5)
plt.hist(df_weekday['hour'], bins=15, normed=1, label='weekday', color='b', alpha=0.5)
plt.xlim(0, 1400)
plt.ylim(0, 0.0040)
plt.xlabel('Number of Users')
plt.ylabel('Probability Density')
plt.legend()
plt.savefig('user_prob_den_kde')
plt.tight_layout()
plt.show()


'''Question 5'''
df_hr = df[['date', 'hour', 'dayofweek', 'subscription_type']]
df_hr_date = df_hr.groupby(['hour','date']).count()
df_hr_date = df_hr_date.reset_index()
'''     hour        date  dayofweek  subscription_type
0        0  2013-08-30          1                  1
1        0  2013-08-31          6                  6
2        0  2013-09-01         10                 10
3        0  2013-09-02         24                 24
4        0  2013-09-03          6                  6
5        0  2013-09-04          2                  2
6        0  2013-09-05          2                  2
7        0  2013-09-06          2                  2
8        0  2013-09-07          5                  5
'''

df_hr_date.boxplot(column='dayofweek', by='hour')
plt.ylim(0,200)
plt.xlabel('Hour of the Day')
plt.ylabel('User Freq')
plt.savefig('freq_hour_boxplots')
plt.tight_layout()
plt.show()


'''Question 7'''
'''Replot the boxplot in 6. after binning your data into weekday and weekend'''
'''In [12]: df_hr.head()
Out[12]:
         date  hour  dayofweek subscription_type
0  2013-08-29    14          3        Subscriber
1  2013-08-29    14          3        Subscriber
2  2013-08-29    10          3        Subscriber
3  2013-08-29    11          3        Subscriber
4  2013-08-29    12          3        Subscriber'''


mask_weekend = df_hr['dayofweek'] > 4
mask_weekday = df_hr['dayofweek'] < 5
df_hr_weekend = df_hr[mask_weekend]
df_hr_weekday = df_hr[mask_weekday]

df_hr_weekend1 = df_hr_weekend.groupby(['hour', 'date']).count()
df_hr_weekend1 = df_hr_weekend.reset_index()
df_hr_weekend.boxplot(column='dayofweek', by='hour')
plt.ylim(0,200)
plt.xlabel('Hour of the Day')
plt.ylabel('User Freq')
plt.title('Weekend')
plt.savefig('weekend_boxplots')
plt.tight_layout()
plt.show()

df_hr_weekday1 = df_hr_weekday.groupby(['hour', 'date']).count()
df_hr_weekday1 = df_hr_weekday.reset_index()
df_hr_weekday.boxplot(column='dayofweek', by='hour')
plt.ylim(0,200)
plt.xlabel('Hour of the Day')
plt.ylabel('User Freq')
plt.title('Weekend')
plt.savefig('weekday_boxplots')
plt.tight_layout()
plt.show()

'''Question 8'''
'''Subscription Type: Subscriber and Customer
'''
'''In [39]: df_hr_weekend.head()
Out[39]:
            date  hour  dayofweek subscription_type
1462  2013-08-31    20          5          Customer
1463  2013-08-31    16          5        Subscriber
1464  2013-08-31    19          5          Customer
1465  2013-08-31    16          5        Subscriber
1466  2013-08-31    18          5          Customer

In [40]: df_hr_weekday.head()
Out[40]:
         date  hour  dayofweek subscription_type
0  2013-08-29    14          3        Subscriber
1  2013-08-29    14          3        Subscriber
2  2013-08-29    10          3        Subscriber
3  2013-08-29    11          3        Subscriber
4  2013-08-29    12          3        Subscriber'''

df_hr_weekend_c = df_hr_weekend[df_hr_weekend['subscription_type'] == 'Customer']
df_hr_weekend_s = df_hr_weekend[df_hr_weekend['subscription_type'] == 'Subscriber']
df_hr_weekday_c = df_hr_weekday[df_hr_weekday['subscription_type'] == 'Customer']
df_hr_weekday_s = df_hr_weekday[df_hr_weekday['subscription_type'] == 'Subscriber']

df_hr_weekend_c1 = df_hr_weekend_c.groupby(['hour', 'date']).count()
df_hr_weekend_s1 = df_hr_weekend_s.groupby(['hour', 'date']).count()
df_hr_weekday_c1 = df_hr_weekday_c.groupby(['hour', 'date']).count()
df_hr_weekday_s1 = df_hr_weekday_s.groupby(['hour', 'date']).count()
df_hr_weekend_c1 = df_hr_weekend_c1.reset_index()
df_hr_weekend_s1 = df_hr_weekend_s1.reset_index()
df_hr_weekday_c1 = df_hr_weekday_c1.reset_index()
df_hr_weekday_s1 = df_hr_weekday_s1.reset_index()


df_hr_weekend_c1.boxplot(column='dayofweek', by='hour')
plt.ylim(0,200)
plt.xlabel('Hour of the Day')
plt.ylabel('User Freq')
plt.title('Customer -- Weekend')
plt.savefig('weekend_c_boxplots')
plt.tight_layout()
plt.show()

df_hr_weekend_s1.boxplot(column='dayofweek', by='hour')
plt.ylim(0,200)
plt.xlabel('Hour of the Day')
plt.ylabel('User Freq')
plt.title('Subscriber -- Weekend')
plt.savefig('weekend_s_boxplots')
plt.tight_layout()
plt.show()

df_hr_weekday_c1.boxplot(column='dayofweek', by='hour')
plt.ylim(0,200)
plt.xlabel('Hour of the Day')
plt.ylabel('User Freq')
plt.title('Customer -- Weekday')
plt.savefig('weekday_c_boxplots')
plt.tight_layout()
plt.show()

df_hr_weekday_s1.boxplot(column='dayofweek', by='hour')
plt.ylim(0,200)
plt.xlabel('Hour of the Day')
plt.ylabel('User Freq')
plt.title('Subscriber -- Weekday')
plt.savefig('weekday_s_boxplots')
plt.tight_layout()
plt.show()



'''Part 2'''
import statsmodels.api as sm
from pandas.tools.plotting import scatter_matrix

prestige = sm.datasets.get_rdataset("Duncan", "car", cache=True).data
y = prestige['prestige']
x = prestige[['income', 'education']].astype(float)

scatter_matrix(prestige, alpha=0.2, figsize=(6, 6), diagonal='kde')
plt.savefig('scatter_matrix')
plt.show()


model = sm.OLS(y, x).fit()
summary = model.summary()
