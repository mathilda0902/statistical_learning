# Plotting
## Objectives:

- random.randint
- plt.subplot
- bar plot, bar heights, bar labels
- x ticks
- tight_layout()
- savefig
- print variable names from csv:

```
with open('data/bay_area_bikeshare/201402_weather_data_v2.csv') as f:
    labels = f.readline().strip().split(',')
[(i, label) for i, label in enumerate(labels)]
```

- load numpy file, with selected columns:

```
cols = [2, 5, 8, 11, 14, 17]
filepath = 'data/bay_area_bikeshare/201402_weather_data_v2.csv'
weather = np.loadtxt(filepath, delimiter=',', skiprows=1, usecols=cols)
```
