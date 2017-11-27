'''plotting'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

'''plt.figure() before figure
plt.show() after figure
'''
'''You can change the default style of your plots with plt.style.use().
You can view your options with plt.style.available
'''
a = np.random.randint(1000, size = 50)
b = np.random.randint(1000, size = 50)
even_sum = (a+b) % 2

plt.figure(figsize = (8,8))
plt.subplot(221)
plt.scatter(a, b, c = even_sum, s=50, alpha = 0.7)
plt.axis([-10, 1010, -10, 1010])
plt.colorbar()
plt.xlabel('a')
plt.ylabel('b')
plt.title('fake data')

x = np.linspace(0, 5, 200)
y1 = 3*x + 0.5
y2 = 5 * np.sqrt(x)

plt.subplot(222)
plt.plot(x, y1, 'k--*', label = 'linear')
plt.plot(x, y2, 'g-^', label = 'squareroot')
plt.legend(loc='best')
plt.xlabel('x')
plt.ylabel('y')
plt.title('linear sqrt')

plt.subplot(223)
plt.plot(x, y1, 'k--*', label = 'linear')
plt.plot(x, y2, 'g-^', label = 'squareroot')
plt.legend(loc='best')
plt.xscale('log')
plt.xlabel('x')
plt.ylabel('y')
plt.title('log linear sqrt')

plt.subplot(224)
barheights = [3,5,1]
barlabels = ['grapes', 'oranges', 'hockey pucks']
x_pos = np.arange(len(barheights))
plt.bar(x_pos, barheights)
plt.xticks(x_pos, barlabels, rotation = 45)
plt.title('bar')

plt.tight_layout()
plt.savefig('four_figures')
plt.show()

'''Bikeshare'''
with open('data/bay_area_bikeshare/201402_weather_data_v2.csv') as f:
    labels = f.readline().strip().split(',')
[(i, label) for i, label in enumerate(labels)]

cols = [2, 5, 8, 11, 14, 17]
filepath = 'data/bay_area_bikeshare/201402_weather_data_v2.csv'
weather = np.loadtxt(filepath, delimiter=',', skiprows=1, usecols=cols)

df_weather = pd.read_csv(filepath)
df_weather.plot(kind = 'scatter', x = 'cloud_cover', y = 'min_humidity')
plt.show()
df_weather.plot(kind = 'hexbin', x = 'cloud_cover', y = 'min_humidity', gridsize=10)
plt.show()
df_weather.plot(kind='hexbin', x = 'mean_sea_level_pressure_in', y = 'min_humidity', gridsize=25)
plt.show()

df_weather = pd.read_csv(filepath,parse_dates=['date'], index_col='date')
df_weather['max_dew_point_f'].hist()
plt.show()
df_weather['zip'].unique()
df_94107 = df_weather[df_weather['zip'] == 94107]
df_94107['max_temperature_f'].plot(figsize=(10,5), marker='o')
plt.show()
df_94107['cloud_cover'].plot(figsize=(10,5), marker='o')
plt.show()
df_94107[df_94107.index.month == 4]['mean_sea_level_pressure_in'].plot(figsize=(10,5), marker='o')
plt.show()
