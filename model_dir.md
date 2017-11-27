# ab_testing - A/B Testing
## CTR t-test:

- Click Through Rate: Clicks / Impressions
- Independent t test:
  - scs.ttest_ind(df_ctr_signed_in['CTR'].dropna(), df_ctr_not_signed['CTR'].dropna(), equal_var = False)
- p value:
  - p_val = stats.ttest_ind(group_1_df['CTR'], group_2_df['CTR'], equal_var=False)[1]
- By gender, by signed in vs. not signed in, and by age groups.


# ab_testing - Experimental Design
## Conversion rate z-test:


# bayesian - Bayesian Analysis

- Flip a fair coin: flip_coin.py
- Flip a biased coin: flip_biased_coin.py
- Bayesian update: bayes.py
    1. update priors
    2. normalize
    3. print distribution
    4. plot



# cv_regularization - Cross Validation

- **Training Set** - Used to train one, or more, models.
- **Validation Set** - Used to tune hyperparameters of different models and choose the best performing model.
- **Test Set** - Used to test the predictive performance of the best scoring model.

1. Use `train_test_split()` in scikit learn to make a training and test dataset.

2. Function `rmse(true, predicted)` that takes your true and predicted values and calculates the RMSE.

3. Fit linear regression model to training data:

```
from sklearn.linear_model import LinearRegression
linear = LinearRegression()
linear.fit(X_train, y_train)
```
4. Calculate predicted responses:

```
train_predicted = linear.predict(X_train)
test_predicted = linear.predict(X_test)
```

5. Create the function `crossVal(X_train, y_train)` using K-fold validation on the training dataset. sklearn has its own implementation of K-fold (sklearn.cross_validation.cross_val_score()). My own implementation contains following steps:

    - Randomly split the dataset into k folds.

    - For each fold:

        - Train the model with the (k-1) other folds

        - Use the trained model to predict the target values of each example in the current fold

        - Calculate the RMSE of the current fold's predicted values

        - Store the RMSE for this fold

    - Average the `k` results of your error metric. Return the average error metric.

6. Checking with `cross_val_score`:

    ```
    model = LinearRegression()
    cv_result = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
    b=[-a for a in cv_result]
    ```

7. Stepwise forward selection:

    - Adjusted R^2
    - RFE, Recursive Feature Elimination


# cv_regularization - Regularization: Ridge (Shrinkage) and Lasso
## Functions:

1. `mse_ridge` and `mse_lasso`: mean squared error of Ridge / Lasso Regression
2. `plot_ridge`: Plot the parameters (coefficients) of the Ridge regression (y-axis) versus the value of various alpha parameter.
3. `plot_error`: Plot the validation error and training error curves for Ridge regression with different alpha parameters.
4. `plot_lass`: Make a plot of the training error and the validation error as a function of the alpha parameter.
5. `best_choice`: Select a model based on the validation and training error curves.


# linear_algebra_eda - Exploratory Data Analysis (EDA)

1. Clean columns and special data formats (datetime).
2. Group by categorical variables. Check.
3. Plot group means. Mark the mean and mean +/- 1.5 * Standard Deviation as horizontal lines on the plot.
4. Plot distributions using histograms. With kde.
5. Boxplots for sub-groups.


# linear_algebra_eda - Linear Algebra

1. Stochastic process (matrix)
2. Euclidean Distance
3. Cosine similarity


# linear_regression - Linear Regression Diagnosis

1. Import statsmodels.api as sm
2. Scatter matrices
3. Fit linear regression model with constants:

```
y_p = prestige['prestige']
x_p = prestige[['income', 'education']].astype(float)
x_p = sm.add_constant(x_p)

model_p = sm.OLS(y_p, x_p).fit()
summary_p = model_p.summary()
```
4. Model interpretation
5. Outlier test:

```
p_res = model_p.outlier_test()['student_resid']
```

6. Residual plots
7. log transformation: y2 = np.log(y1)
8. QQ plots:

```
sm.qqplot(res, scs.norm, line='45', fit=True)
```
9. Checking multicollinearity using VIF's:
  - Function: def vif(x), x: dataframe



# linear_regression - Linear Regression Case Study

1. Dummy variables. Need to drop base level.
2. Fit model.
3. Outliers and residual study.
4. Feature scatter plots.
5. Remove the data points below the decided threshold of your chosen variable and examine the number of zero observations that remain.
6. Fit model again.

# logistic - ROC (Receiver Operating Characteristic) curve

1. ROC curve function `roc_curve`:

```
function ROC_curve(probabilities, labels):
    Sort instances by their prediction strength (the probabilities)
    For every instance in increasing order of probability:
        Set the threshold to be the probability
        Set everything above the threshold to the positive class
        Calculate the True Positive Rate (aka sensitivity or recall)
        Calculate the False Positive Rate (1 - specificity)
    Return three lists: TPRs, FPRs, thresholds
```
Recall that the *true positive rate* is

```
 number of true positives     number correctly predicted positive
-------------------------- = -------------------------------------
 number of positive cases           number of positive cases
```
and the *false positive rate* is

```
 number of false positives     number incorrectly predicted positive
--------------------------- = ---------------------------------------
  number of negative cases           number of negative cases
```

2. Use pandas `crosstab` to get a pivot table: `pd.crosstab(col1, col2)`

3. Use statsmodels to fit a Logistic Regression:

```
from sklearn.linear_model import LogisticRegression

model5 = LogisticRegression()
model5 = logit_model.fit(X, y)
fpred= model5.predict_proba(fm)

```

4. Output in format:

```
output = np.vstack((fm['rank'], p_vec, odds)).T
for a in output:
    print 'rank: {}, probability: {}, odds: {}'. format(int(a[0]), round(a[1], 6), round(a[2], 6))
```
    Display:
        rank: 1, probability: 0.518633, odds: 1.077417
        rank: 2, probability: 0.370328, odds: 0.588129
        rank: 3, probability: 0.243022, odds: 0.321042
        rank: 4, probability: 0.149115, odds: 0.175247

5. Metrics scores:

```
metrics.accuracy_score(y_test, y_pred)
metrics.recall_score(y_test, y_pred)
metrics.precision_score(y_test, y_pred)
```

# multi_armed_bandit - Multi-armed Bandit Problem

## Bayesian A/B testing:
While A/B testing with frequentist and Bayesian methods can be incredibly useful for determining the effectiveness of various changes to your products, better algorithms exist for making educated decision on-the-fly. Two such algorithms that typically out-perform A/B tests are extensions of the Multi-armed bandit problem which uses an epsilon-greedy strategy. Using a combination of exploration and exploitation, this strategy updates the model with each successive test, leading to higher overall click-through rate. An improvement on this algorithm uses an epsilon-first strategy called UCB1. Both can be used in lieu of traditional A/B testing to optimize products and click-through rates.

1. Posterior after n views, updated by prior distributions beta
2. simulating 10,000 points from site A's & B's beta distributions

# pandas_1 - Pandas 1
## Objectives:

- Creating new columns
- sort by columns, ascending and descending
- group by: by total, and by average of groups
- order: group by, then aggregate function (.count(), .mean(), .sum()), lastly sort_values()

# pandas_2 - Pandas 2
## Objectives:

- merge DataFrames on some columns
- more mask/conditions


# plotting - Plotting
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

# power_analysis - Power Analysis

## Functions:
### One sample:
- CI for mean
- Power for alternative hyppothesis
- Power vs effect effect size

### Two sample pooled:
- Null and alternative hypotheses distributions
- Minimum sample size

##  Part 0:
7. Under the null specified in part 2, using a 5% type I error rate, and considering the true mean being equal to the one found in our sample; compute the power of the test. Explain what power means in the context of this problem.

   ```
   calc_power(coke_weights, 20.4)
   # 0.29549570806327596
   ```
   ```
   Power, in this context, is the probability of detecting the mean weight of a
   bottle of coke is different from 20.4 given the that the weight of a bottle of
   coke is indeed different from 20.4.  In this case, we have a 29.5% probability
   of choosing the alternative correctly if the true mean value is larger by 0.12 ounces.
   ```

# probability - Probability
## Objectives:

The following is a short review of the distributions.

```
   Discrete:

       - Bernoulli
           * Model one instance of a success or failure trial (p)

       - Binomial
           * Number of successes out of a number of trials (n), each with probability of success (p)

       - Poisson
           * Model the number of events occurring in a fixed interval
           * Events occur at an average rate (lambda) independently of the last event

       - Geometric
           * Sequence of Bernoulli trials until first success (p)


   Continuous:

       - Uniform
           * Any of the values in the interval of a to b are equally likely

       - Gaussian
           * Commonly occurring distribution shaped like a bell curve
           * Often comes up because of the Central Limit Theorem (to be discussed later)

       - Exponential
           * Model time between Poisson events
           * Events occur continuously and independently
```

## functions:

- def covariance(x1, x2)
- def correlation(x1, x2)
- pd.cut
- plot Gausian kde
- scatter plot imposed on linear regression line
- pearson and spearman: correlations
- distribution simulations
- histogram
- np.percentile


# sampling_estimation - Bootstrapping Method
## Functions:

- Draw samples of random variables from a specified distribution, dist, with given parameters, params, return these in an array.
- Plot distribtuion of sample means for repeated draws from distribution. Draw samples of specified size from Scipy.stats distribution and calculate the sample mean. Repeat this a specified number of times to build out a sampling distribution of the sample mean. Plot the results.
- Sample standard deviation
- Sample standard error (SE)
- Create a series of bootstrapped samples of an input array.
- Calculate the confidence interval (C.I.) of chosen sample statistic using bootstrap sampling.
- Pearson correlation matrix.


# sampling_estimation - Parametric Estimates
## Functions:

- Sample mean and variance
- Methond of moments:
  - Fit Gamma and Normal distributions.
  - Plot pdf's.
- Likelihood estimation:
  - Fit.
  - Maximum likelihood Estimates.
  - Plot likelihood functions.
- maximum likelihood estimation:
  - stats.gamma.fit
  - stats.norm.fit
  - Plot pdf's.
- Plot kde.


- [x] completed:
  1. ab_testing
  2. bayesian
  3. cv_regularization
  4. linear_algebra_eda
  5. linear_regression
  6. logistic
  7. multi_armed_bandit
  8. pandas_1
  9. pandas_2
  10. plotting
  11. power_analysis
  12. probability
  13. sampling_estimation

- [ ] uncompleted:
  1. Decision Trees
  2. Bagging and Random Forests
  3. Boosting
