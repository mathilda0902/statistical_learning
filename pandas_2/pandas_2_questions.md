# Moneyball

The ultimate goal for this assignment is to build a moneyball calculator for the Oakland A's head coach Billy Beane (played by Brad Pitt in the movie).   This calculator/software will find the data driven best players for the least amount of money.

Time permitting, we will re-appropriate this calculator to create data driven charts that illustrate why we made our decisions. As if we needed the ammo to bring to the big board meeting with the owner of the baseball team to convince them to make data-driven changes.

This may sound a bit complex and/or difficult...but you'll be surprised just how easy it will be!!!
<br>
<br>

## Overview
To do this assignment, you will have to know Python's Pandas.  You *could* say this assignment is just a trick to teach you the wonders of Pandas.

The tools and all the data to build this calculator are freely available.  What is difficult is getting the data into the format you want it.  What is complex is putting it all together.  

First, we need to feature engineer our 'moneyball' statistics.  These statistics are built from the standard, existing baseball statistics that Billy Beane believes will lead to earning more runs, and thus, winning more games.  

## Digging into hitting stats with the Batting.csv
** Inside the data folders is a text file named `readme2013.txt`, this is the documentation for the data files. **

The Batting.csv file contains what are known as 'back of the card' stats.  These are the baseball stats that were considered important because they are on the back of every single baseball card.

We will combine, twist, and turn the old features to create new features in an attempt to find better link to winning games.  This practice is called feature engineering.

For example, a players batting average is calculated by taking the number of hits they've had, and dividing it by the number of times they were at bat.  

Pandas is great, because if we want to create a new column in our DataFrame, we can do so on the fly just like this.  
``` df['new col'] = df['colA'] / df['colB'] ```

## Your first job is to feature engineer three new statistics.
The new stats you are calculating are Batting Average, OnBase Percentage, and Slugging Percentage.  
Each stat should be added as a new column in dfBatting.  
To get you started, I have done Batting Average for you perfectly.  
If you do not know **how** to calculate the new stats, use the links provided to figure out their equations.  You might need to reference the data's readme file to figure out what the abbreviations stand for.

1.  **Batting Average:**
  ``` dfBatting['BA'] =  dfBatting['H'] / dfBatting['AB'] ```
1.  **On Base Percentage (OBP)**: <br> http://en.wikipedia.org/wiki/On-base_percentage
1.  **Slugging Percentage (SLG)** <br> http://en.wikipedia.org/wiki/Slugging_percentage

If you print dfBatting.tail() it should look like this.

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>playerID</th>
      <th>yearID</th>
      <th>stint</th>
      <th>teamID</th>
      <th>lgID</th>
      <th>G</th>
      <th>G_batting</th>
      <th>AB</th>
      <th>R</th>
      <th>H</th>
      <th>2B</th>
      <th>3B</th>
      <th>HR</th>
      <th>RBI</th>
      <th>SB</th>
      <th>CS</th>
      <th>BB</th>
      <th>SO</th>
      <th>IBB</th>
      <th>HBP</th>
      <th>SH</th>
      <th>SF</th>
      <th>GIDP</th>
      <th>G_old</th>
      <th>BA</th>
      <th>OBP</th>
      <th>1B</th>
      <th>SLG</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>97884</th>
      <td> zimmejo02</td>
      <td> 2013</td>
      <td> 1</td>
      <td> WAS</td>
      <td> NL</td>
      <td>  32</td>
      <td>  32</td>
      <td>  65</td>
      <td>  4</td>
      <td>   8</td>
      <td>  1</td>
      <td> 0</td>
      <td>  0</td>
      <td>  2</td>
      <td>  0</td>
      <td> 0</td>
      <td>  1</td>
      <td>  20</td>
      <td> 0</td>
      <td> 0</td>
      <td> 6</td>
      <td> 1</td>
      <td>  0</td>
      <td>NaN</td>
      <td> 0.123077</td>
      <td> 0.134328</td>
      <td>   7</td>
      <td> 0.138462</td>
    </tr>
    <tr>
      <th>97885</th>
      <td> zimmery01</td>
      <td> 2013</td>
      <td> 1</td>
      <td> WAS</td>
      <td> NL</td>
      <td> 147</td>
      <td> 147</td>
      <td> 568</td>
      <td> 84</td>
      <td> 156</td>
      <td> 26</td>
      <td> 2</td>
      <td> 26</td>
      <td> 79</td>
      <td>  6</td>
      <td> 0</td>
      <td> 60</td>
      <td> 133</td>
      <td> 2</td>
      <td> 2</td>
      <td> 0</td>
      <td> 3</td>
      <td> 16</td>
      <td>NaN</td>
      <td> 0.274648</td>
      <td> 0.344392</td>
      <td> 102</td>
      <td> 0.464789</td>
    </tr>
    <tr>
      <th>97886</th>
      <td>  zitoba01</td>
      <td> 2013</td>
      <td> 1</td>
      <td> SFN</td>
      <td> NL</td>
      <td>  30</td>
      <td>  30</td>
      <td>  34</td>
      <td>  3</td>
      <td>   5</td>
      <td>  0</td>
      <td> 0</td>
      <td>  0</td>
      <td>  2</td>
      <td>  0</td>
      <td> 0</td>
      <td>  0</td>
      <td>   8</td>
      <td> 0</td>
      <td> 0</td>
      <td> 9</td>
      <td> 0</td>
      <td>  1</td>
      <td>NaN</td>
      <td> 0.147059</td>
      <td> 0.147059</td>
      <td>   5</td>
      <td> 0.147059</td>
    </tr>
    <tr>
      <th>97887</th>
      <td> zobribe01</td>
      <td> 2013</td>
      <td> 1</td>
      <td> TBA</td>
      <td> AL</td>
      <td> 157</td>
      <td> 157</td>
      <td> 612</td>
      <td> 77</td>
      <td> 168</td>
      <td> 36</td>
      <td> 3</td>
      <td> 12</td>
      <td> 71</td>
      <td> 11</td>
      <td> 3</td>
      <td> 72</td>
      <td>  91</td>
      <td> 4</td>
      <td> 7</td>
      <td> 1</td>
      <td> 6</td>
      <td> 18</td>
      <td>NaN</td>
      <td> 0.274510</td>
      <td> 0.354376</td>
      <td> 117</td>
      <td> 0.401961</td>
    </tr>
    <tr>
      <th>97888</th>
      <td> zuninmi01</td>
      <td> 2013</td>
      <td> 1</td>
      <td> SEA</td>
      <td> AL</td>
      <td>  52</td>
      <td>  52</td>
      <td> 173</td>
      <td> 22</td>
      <td>  37</td>
      <td>  5</td>
      <td> 0</td>
      <td>  5</td>
      <td> 14</td>
      <td>  1</td>
      <td> 0</td>
      <td> 16</td>
      <td>  49</td>
      <td> 0</td>
      <td> 3</td>
      <td> 0</td>
      <td> 1</td>
      <td>  5</td>
      <td>NaN</td>
      <td> 0.213873</td>
      <td> 0.290155</td>
      <td>  27</td>
      <td> 0.329480</td>
    </tr>
  </tbody>
</table>

___

# Joining batting data with salary data.
Now that we have our batting stats all clean and tight,  we want to be able to look up a player's batting stat and see how much they are making. Since our dfBatting doesn't have any salary information...

### Your next job is to merge the Batting.csv with the Salaries.csv
1.  Load in Salaries.csv as ```dfSals```
2.  If you print ```dfSals.yearID.min()```, you will notice that it only contains data from 1985 and above.  Since our dfBatting contains data from the 1800's, a merge may get messy.  
3.  **From dfBatting, remove all data before 1985**
4.  Now that our data files are nice and clean, find a column or columns that you can use to merge the two dataframes with.  Use ```pd.merge()``` to merge the dfBatting and dfSals together as a new dataframe called ```mergeddf```.  
5. Drop any columns that are repetitive, and also get rid of `G_old`.

Expected results:
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>playerID</th>
      <th>yearID</th>
      <th>stint</th>
      <th>teamID_x</th>
      <th>lgID_x</th>
      <th>G</th>
      <th>G_batting</th>
      <th>AB</th>
      <th>R</th>
      <th>H</th>
      <th>2B</th>
      <th>3B</th>
      <th>HR</th>
      <th>RBI</th>
      <th>SB</th>
      <th>CS</th>
      <th>BB</th>
      <th>SO</th>
      <th>IBB</th>
      <th>HBP</th>
      <th>SH</th>
      <th>SF</th>
      <th>GIDP</th>
      <th>BA</th>
      <th>OBP</th>
      <th>1B</th>
      <th>SLG</th>
      <th>salary</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>25392</th>
      <td> zieglbr01</td>
      <td> 2013</td>
      <td> 1</td>
      <td> ARI</td>
      <td> NL</td>
      <td>  78</td>
      <td>  78</td>
      <td>   2</td>
      <td>  0</td>
      <td>   0</td>
      <td>  0</td>
      <td> 0</td>
      <td>  0</td>
      <td>  0</td>
      <td>  0</td>
      <td> 0</td>
      <td>  0</td>
      <td>   0</td>
      <td> 0</td>
      <td> 0</td>
      <td> 0</td>
      <td> 0</td>
      <td>  0</td>
      <td> 0.000000</td>
      <td> 0.000000</td>
      <td>   0</td>
      <td> 0.000000</td>
      <td>  3150000</td>
    </tr>
    <tr>
      <th>25393</th>
      <td> zimmejo02</td>
      <td> 2013</td>
      <td> 1</td>
      <td> WAS</td>
      <td> NL</td>
      <td>  32</td>
      <td>  32</td>
      <td>  65</td>
      <td>  4</td>
      <td>   8</td>
      <td>  1</td>
      <td> 0</td>
      <td>  0</td>
      <td>  2</td>
      <td>  0</td>
      <td> 0</td>
      <td>  1</td>
      <td>  20</td>
      <td> 0</td>
      <td> 0</td>
      <td> 6</td>
      <td> 1</td>
      <td>  0</td>
      <td> 0.123077</td>
      <td> 0.134328</td>
      <td>   7</td>
      <td> 0.138462</td>
      <td>  5350000</td>
    </tr>
    <tr>
      <th>25394</th>
      <td> zimmery01</td>
      <td> 2013</td>
      <td> 1</td>
      <td> WAS</td>
      <td> NL</td>
      <td> 147</td>
      <td> 147</td>
      <td> 568</td>
      <td> 84</td>
      <td> 156</td>
      <td> 26</td>
      <td> 2</td>
      <td> 26</td>
      <td> 79</td>
      <td>  6</td>
      <td> 0</td>
      <td> 60</td>
      <td> 133</td>
      <td> 2</td>
      <td> 2</td>
      <td> 0</td>
      <td> 3</td>
      <td> 16</td>
      <td> 0.274648</td>
      <td> 0.344392</td>
      <td> 102</td>
      <td> 0.464789</td>
      <td> 14100000</td>
    </tr>
    <tr>
      <th>25395</th>
      <td>  zitoba01</td>
      <td> 2013</td>
      <td> 1</td>
      <td> SFN</td>
      <td> NL</td>
      <td>  30</td>
      <td>  30</td>
      <td>  34</td>
      <td>  3</td>
      <td>   5</td>
      <td>  0</td>
      <td> 0</td>
      <td>  0</td>
      <td>  2</td>
      <td>  0</td>
      <td> 0</td>
      <td>  0</td>
      <td>   8</td>
      <td> 0</td>
      <td> 0</td>
      <td> 9</td>
      <td> 0</td>
      <td>  1</td>
      <td> 0.147059</td>
      <td> 0.147059</td>
      <td>   5</td>
      <td> 0.147059</td>
      <td> 20000000</td>
    </tr>
    <tr>
      <th>25396</th>
      <td> zobribe01</td>
      <td> 2013</td>
      <td> 1</td>
      <td> TBA</td>
      <td> AL</td>
      <td> 157</td>
      <td> 157</td>
      <td> 612</td>
      <td> 77</td>
      <td> 168</td>
      <td> 36</td>
      <td> 3</td>
      <td> 12</td>
      <td> 71</td>
      <td> 11</td>
      <td> 3</td>
      <td> 72</td>
      <td>  91</td>
      <td> 4</td>
      <td> 7</td>
      <td> 1</td>
      <td> 6</td>
      <td> 18</td>
      <td> 0.274510</td>
      <td> 0.354376</td>
      <td> 117</td>
      <td> 0.401961</td>
      <td>  5687300</td>
    </tr>
  </tbody>
</table>

___
<br>
<br>
# Summing our losses.  
During the 2001-02 offseason, the Oakland A's lost three key free agents to larger market teams:  

1. first baseman 2000 AL MVP Jason Giambi (`giambja01`) to the New York Yankees  
2. outfielder Johnny Damon (`damonjo01`) to the Boston Red Sox  
3. closing pitcher Jason Isringhausen (`isrinja01`) to the St. Louis Cardinals  
4. as well as infielder Rainer Gustavo "Ray" Olmedo (``'saenzol01'``)

![](images/3ballplayers.png)

#### Now, tell me what our three offensive players average OPB was.  


* Using `mergeddf`, create a new dataframe that is just of the Oakland A's 2001 team (call it `oak2001`).
* Tell me the stats for each of these key players during their 2001 season at the Oakland A's
* Put all of the players data we are losing into a new dataframe called `lostboysdf`. hint:  
* ```my_mask = oak2001['playerID'].isin(['list', 'of', 'names'])``` *
* ```lostboysdf = oak2001[my_mask] ```

Expected Output: (Your df may have additional stats, I just cut the df width for presentation purposes):
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>playerID</th>
      <th>H</th>
      <th>2B</th>
      <th>3B</th>
      <th>HR</th>
      <th>OBP</th>
      <th>SLG</th>
      <th>BA</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>4955 </th>
      <td> damonjo01</td>
      <td> 165</td>
      <td> 34</td>
      <td> 4</td>
      <td>  9</td>
      <td> 0.?????</td>
      <td> 0.363354</td>
      <td> 0.256211</td>
    </tr>
    <tr>
      <th>7586 </th>
      <td> giambja01</td>
      <td> 178</td>
      <td> 47</td>
      <td> 2</td>
      <td> 38</td>
      <td> 0.?????</td>
      <td> 0.659615</td>
      <td> 0.342308</td>
    </tr>
    <tr>
      <th>10580</th>
      <td> isrinja01</td>
      <td>   0</td>
      <td>  0</td>
      <td> 0</td>
      <td>  0</td>
      <td>      NaN</td>
      <td>      NaN</td>
      <td>      NaN</td>
    </tr>
    <tr>
      <th>19422</th>
      <td> saenzol01</td>
      <td>  67</td>
      <td> 21</td>
      <td> 1</td>
      <td>  9</td>
      <td> 0.?????</td>
      <td> 0.383607</td>
      <td> 0.219672</td>
    </tr>
  </tbody>
</table>

Also, return their total at bats ('AB')
```python
        playerID teamID_x   AB  HR       SLG       OBP   salary
4955   damonjo01      OAK  '???'   9  0.363354  0.??????  7100000
7586   giambja01      OAK  '???'  38  0.659615  0.??????  4103333
19422  saenzol01      OAK  '???'   9  0.383607  0.??????   290000
```

___

# Now all we have to do is find the right players to replace them.

We need to find three players that have an average OBP of ???? or higher and the following
constraints:
* together have at least as many at bats as the players we lost
* together cost less than 15 million bucks

Selection Process

1. Let's assume 2001 players are being paid the same amount for the 2002 season
2. Find  three players that have an average OBP that is higher than ??? and will cost less than 15,000,000 USD.
  * Use the salary data from 2001 as your roster.  For the purpose of this exercise, assume all players are free agents (i.e. available to be traded to the A's).  
  * Create a new dataframe called all2001 from your ```mergeddf``` that contains ALL players.
  * Narrow down the stats to just keep ```['playerID', 'teamID_x','AB','HR', 'OBP', 'SLG', 'salary']```
  * This is actually a lot easier than it sounds if you use ```.sort()``` and your eyes.
  * Eliminate any players that have less than 50 at-bats...any fewer and they aren't *real* contributors to the team's offense 
 
**There are many correct answers** here is just one: **I encourage you to do much better**

 <table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>playerID</th>
      <th>teamID_x</th>
      <th>AB</th>
      <th>HR</th>
      <th>SLG</th>
      <th>OBP</th>
      <th>salary</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>18011</th>
      <td> raineti01</td>
      <td> MON</td>
      <td>  78</td>
      <td>  0</td>
      <td> 0.435897</td>
      <td> 0.432990</td>
      <td>  350000</td>
    </tr>
    <tr>
      <th>1756 </th>
      <td> berkmla01</td>
      <td> HOU</td>
      <td> 577</td>
      <td> 34</td>
      <td> 0.620451</td>
      <td> 0.430233</td>
      <td>  305000</td>
    </tr>
    <tr>
      <th>9400 </th>
      <td> heltoto01</td>
      <td> COL</td>
      <td> 587</td>
      <td> 49</td>
      <td> 0.684838</td>
      <td> 0.431655</td>
      <td> 4950000</td>
    </tr>
  </tbody>
</table>

<br>

# EXTRA CREDIT:
**MEDIUM:** Find all 3-player combinations that have MORE at bats, in addition to the the rest of the moneyball criteria.  

**HARD:**:  Find *all* the positions the players we lost played, then return only 3-player combinations that also played the same positions and fit our other criteria.

**Hint:** You might want to use this python library:  
```python
from itertools import combinations
```
---
* Python for Data Analysis: Chapter 5, 8, 9
* [pandas homepage](http://pandas.pydata.org/)
* [pandas documentation](http://pandas.pydata.org/pandas-docs/stable/index.html)
* [pandas tutorial](http://www.randalolson.com/2012/08/06/statistical-analysis-made-easy-in-python/)
* [pandas: Merge, Join, and Concatenate](http://pandas.pydata.org/pandas-docs/stable/merging.html)
* [pandas: Selecting Data](http://pandas.pydata.org/pandas-docs/stable/indexing.html)
* [pandas Group By: split-apply-combine](http://pandas.pydata.org/pandas-docs/stable/groupby.html)
* [pandas `apply`](http://pandas.pydata.org/pandas-docs/stable/groupby.html#flexible-apply)
* [pandas `sort_index`](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.sort_index.html#pandas.DataFrame.sort_index)
* [pandas `head`](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.head.html#pandas.DataFrame.head)
