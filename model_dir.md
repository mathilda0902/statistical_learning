# General Workflow for working with sklearn:
```
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Load data from csv file
df = pd.read_csv('data/housing_prices.csv')
X = df[['square_feet', 'num_rooms']].values
y = df['price'].values

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)

# Run Linear Regression
regr = LinearRegression()
regr.fit(X_train, y_train)
print "Intercept:", regr.intercept_
print "Coefficients:", regr.coef_
print "R^2 error:", regr.score(X_test, y_test)
predicted_y = regr.predict(X_test)
```

# Example using KFold:
```
from sklearn import model_selection
kf = model_selection.KFold(X.shape[0], n_folds=5, shuffle=True)
results = []
for train_index, test_index in kf:
    regr = LinearRegression()
    regr.fit(X[train_index], y[train_index])
    results.append(regr.score(X[test_index], y[test_index]))
print "average score:", np.mean(results)
```

 - `sklearn.metrics.mean_squared_error()`
 - `sklearn.cross_validation.cross_val_score()`. ref: http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.cross_val_score.html
 - `sklearn.feature_selection.RFE`. ref: http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFE.html
 - mse of Ridge:
  ```
  import numpy as np
  import matplotlib.pyplot as plt
  from sklearn.datasets import load_diabetes
  from sklearn.preprocessing import scale
  from sklearn.cross_validation import KFold
  from sklearn.metrics import mean_squared_error
  import matplotlib.pyplot as plt
  from sklearn.cross_validation import train_test_split
  from sklearn.linear_model import Ridge, Lasso, RidgeCV, \
  										LassoCV, LinearRegression

  def mse_ridge(alpha=0.5):
  	ridge = Ridge(alpha=alpha)
  	fit = ridge.fit(X_train, y_train)
  	y_pred = ridge.predict(X_test)
  	return mean_squared_error(y_test, y_pred)
  ```
  - mse of Lasso:
  ```
  def mse_lasso(alpha=0.5):
	lasso = Lasso(alpha=alpha)
	fit = lasso.fit(X_train, y_train)
	y_pred = lasso.predict(X_test)
	return mean_squared_error(y_test, y_pred)
  ```
  - KFold cross validation model:
  ```
  def KFoldCVModel(model, kf, n_folds=10):
      train_error = np.empty(n_folds)
      val_error = np.empty(n_folds)
      for i, (train, validation) in enumerate(kf):
          model.fit(X_train[train], y_train[train])
          train_error[i] = mean_squared_error(y_train[train],
  										model.predict(X_train[train]))
          val_error[i] = mean_squared_error(y_train[validation],
  										model.predict(X_train[validation]))
      return train_error, val_error
  ```
  - cross validation to test best alpha:
  ```
  def test_alphas(model, alphas = np.linspace(0, 20, 400)):
  	k_fold_train_error = np.zeros(len(alphas))
  	k_fold_test_error = np.zeros(len(alphas))
  	for i, a in enumerate(alphas):
  	    ridge = model(alpha=a, normalize=True)
  	    kf = KFold(X_train.shape[0], n_folds=10)
  	    train_error, test_error = KFoldCVModel(ridge, kf)
  	    k_fold_train_error[i] = np.mean(train_error)
  	    k_fold_test_error[i] = np.mean(test_error)
  	return k_fold_train_error, k_fold_test_error
  ```


# ab_testing - A/B Testing
## CTR t-test:

- Click Through Rate: Clicks / Impressions
- Independent t test:
  - `scs.ttest_ind(df_ctr_signed_in['CTR'].dropna(), df_ctr_not_signed['CTR'].dropna(), equal_var = False)`
- p value:
  - `p_val = stats.ttest_ind(group_1_df['CTR'], group_2_df['CTR'], equal_var=False)[1]`
- By gender, by signed in vs. not signed in, and by age groups.
- Calculate and return an estimated probability that SiteA performs better
(has a higher click-through rate) than SiteB. Hint: Use Bayesian A/B Testing
```
    def calculate_clickthrough_prob(clicks_A, views_A, clicks_B, views_B):
        sample_size = 10000
        A_samples = beta_dist(1 + clicks_A, 1 + views_A - clicks_A, sample_size)
        B_samples = beta_dist(1 + clicks_B, 1 + views_B - clicks_B, sample_size)
        return np.mean(A_samples > B_samples)
```

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
- Difference between Bayesian and Frequentist inference:
  - Frequentist: Sampling is infinite and decision rules can be sharp. Data are a repeatable random sample - there is a frequency. Underlying parameters are fixed i.e. they remain constant during this repeatable sampling process.
  - Bayesian: Unknown quantities are treated probabilistically and the state of the world can always be updated. Data are observed from the realized sample. Parameters are unknown and described probabilistically. It is the data which are fixed.

- Difference between a frequentist A/B test and a Bayesian A/B test:
  - https://conversionxl.com/blog/bayesian-frequentist-ab-testing/
  - Basically, using a Frequentist method means making predictions on underlying truths of the experiment using only data from the current experiment.
  - So, the biggest distinction is that Bayesian probability specifies that there is some prior probability. The Bayesian approach goes something like this (summarized from this discussion):
    - Define the prior distribution that incorporates your subjective beliefs about a parameter. The prior can be uninformative or informative.
    - Gather data.
    - Update your prior distribution with the data using Bayes’ theorem (though you can have Bayesian methods without explicit use of Bayes’ rule to obtain a posterior distribution. The posterior distribution is a probability distribution that represents your updated beliefs about the parameter after having seen the data.
    - Analyze the posterior distribution and summarize it (mean, median, sd, quantiles,…).


# Bias and Variances
- If you get low training error and high testing error, what do you think is happening and what can you do about it?
- The variance of the training model is getting high. This is an indicator that the model that is being tested fits very well to the training data set, with possibly minimum bias. The low bias is reflected in the low training errors. If we would like to lower our testing error when applying our model to the testing set, we will have to sacrifice some variance of our model to achieve this.

- What is the RSS for a model's predictions?
- eg: The difference between each point and its prediction, the residual, is: [4.7, 0.1, -5.7, -7.8, 8.6], which we will then square to get [22.09, 0.01, 32.49, 60.84, 73.96], and sum, to arrive at an RSS of 189.89.

# cv_regularization - Cross Validation
- Why do we often prefer to use k-Fold Cross Validation instead of a simple train/test split?
- In k-fold cross-validation, the original sample is randomly partitioned into k equal sized subsamples. Of the k subsamples, a single subsample is retained as the validation data for testing the model, and the remaining k − 1 subsamples are used as training data. The cross-validation process is then repeated k times (the folds), with each of the k subsamples used exactly once as the validation data. The k results from the folds can then be averaged to produce a single estimation. The advantage of this method over repeated random sub-sampling is that all observations are used for both training and validation, and each observation is used for validation exactly once. 10-fold cross-validation is commonly used, but in general k remains an unfixed parameter.

# Cross Validation
```
    from sklearn.model_selection import KFold # import KFold
    X = np.array([[1, 2], [3, 4], [1, 2], [3, 4]]) # create an array
    y = np.array([1, 2, 3, 4]) # Create another array
    kf = KFold(n_splits=2) # Define the split - into 2 folds
    kf.get_n_splits(X) # returns the number of splitting iterations in the cross-validator
    print(kf)
    KFold(n_splits=2, random_state=None, shuffle=False)
```
The resulted folds:
```
    for train_index, test_index in kf.split(X):
     print(“TRAIN:”, train_index, “TEST:”, test_index)
     X_train, X_test = X[train_index], X[test_index]
     y_train, y_test = y[train_index], y[test_index]
    ('TRAIN:', array([2, 3]), 'TEST:', array([0, 1]))
    ('TRAIN:', array([0, 1]), 'TEST:', array([2, 3]))
```
LeaveOneOut:
```
    from sklearn.model_selection import LeaveOneOut
    X = np.array([[1, 2], [3, 4]])
    y = np.array([1, 2])
    loo = LeaveOneOut()
    loo.get_n_splits(X)

    for train_index, test_index in loo.split(X):
       print("TRAIN:", train_index, "TEST:", test_index)
       X_train, X_test = X[train_index], X[test_index]
       y_train, y_test = y[train_index], y[test_index]
       print(X_train, X_test, y_train, y_test)
```
Source: https://towardsdatascience.com/train-test-split-and-cross-validation-in-python-80b61beca4b6

- explain train/test splits:
  - Fitting our model to a given dataset is not enough. We would also like to know the predictive power of our model. To save time and money, we could reserve a smaller portion of our dataset for testing purposes. We then use the training portion as we desire to look for an optimal model, followed by testing our model on the testing portion. This split among the dataset is called a train/test split.
- explain how to get an honest sense of how well a model is doing
  - The predictive accuracy among other metrics shows how well a model is doing in terms of predicting some outcomes based on our hypotheses and selected features. We usually test this accuracy by fitting our model to the test dataset, and calculate a chosen error. Oftentimes, errors can be calculated using mean squared errors.
- state the purposes of Cross Validation
  - To re-use our data when the situation is not 'data-rich', cross validation provides us with a more efficient way of fitting model to the data that's available to us, compared with the traditional one split between training and testing data. Cross validation requires us to partition our datasets into k parts (called k-fold), then train-test our models with iterations. At each iteration, we train on the (k-1) parts of our data, and test it on the remaining part. Each of the k folds is used exactly once, thus providing a robust testing environment for our model.
- explain k in “k-fold cross validation”
  - See above.
- describe the sources of model error
  - There are two sources of model error: 1) from inevitable external noises, 2) from systematic errors of our model.
- describe overfitting and underfitting
  - When a model has been trained for too many times or its complexity is too high, overfitting happens. In this case, the model can very well predict the training data, but when applied to testing data, it's expected to deviate/have large variances from the true results. The degrees that model fits very closely to the training set is called a low bias, while the large variances of the testing data is called high variance. This is a bias-variance trade-off. Underfitting is the reverse.
- explain how to compare models and select the best one


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
- compare and contrast Ridge and Lasso:
- Ridge:
  - The assumptions of Ridge regression is same as least squared regression except normality is not to be assumed
  - It shrinks the value of coefficients but doesn’t reaches zero, which suggests no feature selection feature
  - This is a regularization method and uses l2 regularization.
- Lasso:
  - The assumptions of this regression is same as least squared regression except normality is not to be assumed
  - It shrinks coefficients to zero (exactly zero), which certainly helps in feature selection
  - This is a regularization method and uses l1 regularization
  - If group of predictors are highly correlated, lasso picks only one of them and shrinks the others to zero


- explain ridge regression and lasso regression:
  - Ridge regression and lasso regression both minimizes the model rss while penalizing the magnitudes of coefficients. Ridge regression takes an L2-norm on the beta coefficients while lasso takes L1-norm. Ridge regression works well for preventing overfitting. It takes all features in thus not computationally cheap for datasets with features of millions or more. Lasso regression works for cases in which features are of millions and we are looking for a sparse solution. This has a computational advantage over ridge regression.
- tune the bias/variance of a regression model:
  - We would try to eliminate some of the beta coefficients which are very much close to zero. This means that the variables associated with these coefficients are of little predicting power to our responses. We can reduce our model complexity in this way to avoid a large variance from our model, thus avoiding overfitting.
- choose the best regularization hyper-parameter for regression:

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
6. Find missing data:
  ```
  df.apply(lambda x: np.sum(pd.isnull(x)))
  ```
eg.: So, ~16% of avg_rating_of_driver is missing, which means we probably should not drop this data. Would be nice to just do a simple t-test to see if distribution of features is different for rows with/without missing values. Here is a helper function:

  ```
  def ttest_by(vals, by):
    '''Compute a t-test on a column based on an indicator for which sample the values are in.'''
    vals1 = vals[by]
    vals2 = vals[-by]

    return sp.stats.ttest_ind(vals1, vals2)
  ```
  ```
  ttest_by(df.avg_dist, pd.isnull(df.avg_rating_of_driver))
  ```
  -  We have a couple options for handling missing data: drop the rows if there are only a few or they are missing at random, impute the missing values, or bin the feature by quantiles (typically deciles) + a bin for missing.

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
- Logistic in taxonomy of Machine Learning algorithms:
  - Although it confusingly includes 'regression' in the name, logistic regression is actually a powerful tool for two-class and multiclass classification. It's fast and simple. The fact that it uses an 'S'-shaped curve instead of a straight line makes it a natural fit for dividing data into groups. Logistic regression gives linear class boundaries, so when you use it, make sure a linear approximation is something you can live with.
- Difference between linear regression and logistic regression:
  - Linear Regression is used to establish a relationship between Dependent and Independent variables, which is useful in estimating the resultant dependent variable in case independent variable change.
  - Logistic Regression on the other hand is used to ascertain the probability of an event. And this event is captured in binary format, i.e. 0 or 1.


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
output = np.vstack((fm['rank'], p_vec, )).T
for a in output:
    print 'rank: {}, probability: {}, : {}'. format(int(a[0]), round(a[1], 6), round(a[2], 6))
```
    Display:
        rank: 1, probability: 0.518633, : 1.077417
        rank: 2, probability: 0.370328, : 0.588129
        rank: 3, probability: 0.243022, : 0.321042
        rank: 4, probability: 0.149115, : 0.175247

5. Metrics scores:

```
metrics.accuracy_score(y_test, y_pred)
metrics.recall_score(y_test, y_pred)
metrics.precision_score(y_test, y_pred)
```

6. Beta coefficient interpretation:
  - Increasing the `GRE score` by `1 point` increases the chance of getting in `by a factor` of `1.00189`, or an increase of 0.189%.
  -  for a one-unit increase in the `continuous variable`, the expected change in log  is log (p/(1-p))
  - What change is required to double my chances of admission?
    - `log(2) / coef`: Increasing the GRE score by 367 points doubles the chance of getting in.
  - more: https://stats.idre.ucla.edu/other/mult-pkg/faq/general/faq-how-do-i-interpret--ratios-in-logistic-regression/

7. Compute the  (p/(1-p)):
  ```
  probabilities_rank = model.predict_proba(X_rank)[:, 1]
  for rank, prob in izip(ranks, probabilities_rank):
    print "rank: %d, probability: %f, : %f" % (rank, prob, prob / (1 - prob))
  ```
  - note: logit = log of  = log (p / (1-p))

# multi_armed_bandit - Multi-armed Bandit Problem
- The multi-armed bandits in particular focus on the question of exploration vs. exploitation trade-off - how many resources should be spent in trial and error vs. maximizing the benefit.
- We can calculate the expected clicks (rewards) from each strategy and see how many fewer clicks we got using the epsilon-greedy vs. the epsilon-first (A/B testing). These differences are known as "regrets".

## Bayesian A/B testing: (/Users/Victoria/galvanize/week2/dsi-power-bayesian/bayesian_ab_testing.pdf)
While A/B testing with frequentist and Bayesian methods can be incredibly useful for determining the effectiveness of various changes to your products, better algorithms exist for making educated decision on-the-fly. Two such algorithms that typically out-perform A/B tests are extensions of the Multi-armed bandit problem which uses an epsilon-greedy strategy. Using a combination of exploration and exploitation, this strategy updates the model with each successive test, leading to higher overall click-through rate. An improvement on this algorithm uses an epsilon-first strategy called UCB1. Both can be used in lieu of traditional A/B testing to optimize products and click-through rates.

1. Posterior after n views, updated by prior distributions beta
2. simulating 10,000 points from site A's & B's beta distributions

```
from numpy.random import beta
num samples = 10000
alpha=beta=1
site a simulation = beta(num conv a + alpha,
num views a − num conv a + beta,
size=num samples) site b simulation = beta(num conv b + alpha,
num views b − num conv b + beta, size=num samples)
```

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
- calculate sample size for power and significant level:
`dsi-warm-ups/answer/frequentist_ab_testing.py`
- ways to increase power:
  - Increase alpha
  - Conduct a one-tailed test
  - Increase the effect size
  - Decrease random error
  - Increase sample size
  - https://www.theanalysisfactor.com/5-ways-to-increase-power-in-a-study/
  - lecture notes: /Users/Victoria/galvanize/week2/dsi-power-bayesian/power_lecture.pdf

- Bonferroni: dsi-warm-ups/ab_testing.py


## Functions:
### One sample:
- CI for mean
- Power for alternative hypothesis
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


# decision_trees - Decision Trees
1. Decision trees implementation
2. Decision trees regressor
3. knn
4. Recursion practice
5. http://dataaspirant.com/2017/01/30/how-decision-tree-algorithm-works/
6. https://www.analyticsvidhya.com/blog/2016/04/complete-tutorial-tree-based-modeling-scratch-in-python/

7. Decision trees pros and cons:
  - Regular (not bagged or boosted)
    - Pros:
      - easy to interpret visually when the trees only contain several levels
      - Can easily handle qualitative (categorical) features
      - Works well with decision boundaries parellel to the feature axis
    - Cons:
      - prone to overfitting
      - possible issues with diagonal decision boundaries


9. notes on sklearn and its practical use:
  - Remember that the number of samples required to populate the tree doubles for each additional level the tree grows to. Use max_depth to control the size of the tree to prevent overfitting.
  - Use `min_samples_split` or `min_samples_leaf` to control the number of samples at a leaf node. A very small number will usually mean the tree will overfit, whereas a large number will prevent the tree from learning the data. Try `min_samples_leaf=5` as an initial value. If the sample size varies greatly, a float number can be used as percentage in these two parameters. The main difference between the two is that `min_samples_leaf` guarantees a minimum number of samples in a leaf, while `min_samples_split` can create arbitrary small leaves, though `min_samples_split` is more common in the literature.
  - Balance your dataset before training to prevent the tree from being biased toward the classes that are dominant. Class balancing can be done by sampling an equal number of samples from each class, or preferably by normalizing the sum of the sample weights (`sample_weight`) for each class to the same value. Also note that weight-based pre-pruning criteria, such as `min_weight_fraction_leaf`, will then be less biased toward dominant classes than criteria that are not aware of the sample weights, like `min_samples_leaf`.
  - If the samples are weighted, it will be easier to optimize the tree structure using weight-based pre-pruning criterion such as `min_weight_fraction_leaf`, which ensure that leaf nodes contain at least a fraction of the overall sum of the sample weights.
  - All decision trees use `np.float32` arrays internally. If training data is not in this format, a copy of the dataset will be made.
  - If the input matrix X is very sparse, it is recommended to convert to sparse `csc_matrix` before calling fit and sparse `csr_matrix` before calling predict. Training time can be orders of magnitude faster for a sparse matrix input compared to a dense matrix when features have zero values in most of the samples.
  - More on pre-/post-pruning: https://blog.nelsonliu.me/2016/08/05/gsoc-week-10-scikit-learn-pr-6954-adding-pre-pruning-to-decisiontrees/


# random_forest - Random forest

1. roc.py:
  - using KFold
  - input: X, y, clf_class, `**kwargs`
  - generate: fpr, tpr, thresholds
  - plot: mean_fpr, mean_tpr

2. sklearn_rf.py:
  - fitting random forest using default keywords.
  - get: accuracy score, confusion matrix, precision score, recall score.
  - change `oob` to True (`oob_score=True`). compare out-of-bag training accuracy score to test set.
  - feature importances
  - Calculate the standard deviation for feature importances across all trees
  - number of trees and accuracy score
  - max features parameter and accuracy score
  - function `get_score(classifier, X_train, X_test, y_train, y_test, **kwargs)`:
    - return model.score(X_test, y_test), precision_score(y_test, y_predict), recall_score(y_test, y_predict)
  - `n_estimators`: number of trees. `max_features`: the size of the random subsets of features to consider when splitting a node.

3. Confusion matrix:
  - `confusion_matrix(y_test, y_predict)`:
      ```
      answer:  716   6
                40  72
      ```
  - What is the precision? Recall?
    - precision: 0.923076923077
    - recall: 0.642857142857
  - `tn, fp, fn, tp = confusion_matrix([0, 1, 0, 1], [1, 1, 1, 0]).ravel()`
  - Thus in binary classification, the count of true negatives is `C_{0,0}`, false negatives is `C_{1,0}`, true positives is `C_{1,1}` and false positives is `C_{0,1}`.

4. Relation between precision and recall:
  - Note that the precision may not decrease with recall. The definition of precision `(\frac{T_p}{T_p + F_p})` shows that lowering the threshold of a classifier may increase the denominator, by increasing the number of results returned. If the threshold was previously set too high, the new results may all be true positives, which will increase precision. If the previous threshold was about right or too low, further lowering the threshold will introduce false positives, decreasing precision.
  - Recall is defined as `\frac{T_p}{T_p+F_n}`, where `T_p+F_n` does not depend on the classifier threshold. This means that lowering the classifier threshold may increase recall, by increasing the number of true positive results. It is also possible that lowering the threshold may leave recall unchanged, while the precision fluctuates.


# boosting - Boosting regressor

1. boosting.py: `class AdaBoostBinaryClassifier`, features:
    - fit
    - predict
    - score

2. boosting_regressor.py:
    - function `load_and_split_data`
    - `def cv_mse_r2(model)`:
        - Takes an instantiated model (estimator) and returns the average mean square error (mse) and coefficient of determination (r2) from kfold cross-validation.
        - Parameters: estimator: model object
                      X_train: 2d numpy array
                      y_train: 1d numpy array
                      nfolds: the number of folds in the kfold cross-validation
        - Returns:  mse: average mean_square_error of model over number of folds
                    r2: average coefficient of determination over number of folds
    - fit Random Forest:
        `rf = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=1)`
    - fit Gradient Boosting Regressor:
        `GradientBoostingRegressor(learning_rate=0.1, loss='ls', n_estimators=100, random_state=1)`
    - fit Ada Boost Regressor:
        `AdaBoostRegressor(DecisionTreeRegressor(), learning_rate=0.1, loss='linear', n_estimators=100, andom_state=1)`
    - `def stage_score_plot(estimator, X_train, y_train, X_test, y_test)`:
        - Parameters: estimator: GradientBoostingRegressor or AdaBoostRegressor
                      X_train: 2d numpy array
                      y_train: 1d numpy array
                      X_test: 2d numpy array
                      y_test: 1d numpy array
        - Returns: A plot of the number of iterations vs the MSE for the model for both the training set and test set.
    - Grid search
      ```
      def do_grid_search():

      # Split it up into our training and testing sets
      X_train, X_test, y_train, y_test = train_test_split(X,y)

      # Initalize our model here
      est = GradientBoostingClassifier()

      # Here are the params we are tuning, ie,
      # if you look in the docs, all of these are 'nobs' within the GradientBoostingClassifier algo.
      param_grid = {'learning_rate': [0.1, 0.05, 0.02],
                  'max_depth': [2, 3],
                  'min_samples_leaf': [3, 5],
                  }

      # Plug in our model, params dict, and the number of jobs, then .fit()
      gs_cv = GridSearchCV(est, param_grid, n_jobs=2).fit(X_train, y_train)

      # return the best score and the best params
      return gs_cv.best_score_, gs_cv.best_params_
      ```

3. adaboost.py:
    - staged misclassification rate
    - grid search
    - feature importances

4. What is boosting technique?
  - Boosting is an ensemble technique with which we can start our model predictions with a set of weak learners, whose predictions are only a little better than random guesses. These weak learns have high biases and low variances. After training iterations, we will arrive at a strong learner whose predictions are arbitrarily close to the real responses. This process is expected to reduce bias and variance in supervised learning.

5. Boosting for regression problems (GradientBoosting Regressor):
  - Gradient boosting is an gradient descent algorithm with boosting technique. Gradient tree boosting utilizes decision trees as its weak learners. In the regression cases, the weak learners were fitted to data with minimized MSE (mean squared error). Each of the following weak learns 'learns' a better model by correcting its predecessor's MSE.

6. Boosting for classification problems (GradientBoostingClassifier)
  - In the case of classification problems, instead of MSE, we can test models for metrics including precision, recall, accuracy, f1 score, etc.

7. Difference between AdaBoost and Gradient Boosting
  - They both use weak learners to arrive at a strong learner by aggregating (addition). They differ at the iterative process where the weak learners are created. Adaboost creates weak learners by changing the weights attached to each instance. By increasing the weights of wrong predictions (difficult ones) and decreasing the weights of correct predictions (easy ones), Adaboost aggregate weak learners by their (contribution) votes to the strong learner. If a weak learner has strong performance (a high alpha-weight), it has more contribution to the strong learner. These updated weights indicates a changing sample distribution.
  - During the gradient boosting iterative process, instead of changing sample distribution, the weak learners are trained on the remaining errors (pseudo-residuals) of the strong learner. The weak learners' contributions are not evaluated on its performance on the sample distribution, but by using the gradient descent optimization process. The final outcome is the one minimizing the overall error of the strong learner.

8. Understand hyperparameters associated to Boosting (learning rate, number of estimators ...)

9. GridSearch for optimal hyperparameters for Boosting:
10. Performance of Boosting compared to other machine learning algorithms

11. Difference between gradient boosting and random forest:
  - Gradient boosting is a sequential algorithm while random forest grows trees parallel. In random forests each tree is growing independently. To achieve high accucary, RF uses BAGGING (bootstrap aggregating) but is prone to overfitting.
    - BAGGING: drawing sample from training data uniformly and with replacement.
  - Gradient boosting has shallow trees as its weak learners (stumps with only 2 leaves). It reduces error by reducing bias. RF uses full grown trees with low bias and high variance. RF reduces error by reducing variance. The trees in RF are designed to be uncorrelated, thus maximizing the decrease in variance.


# K Nearest Neighbors (knn):
source: https://medium.com/@adi.bronshtein/a-quick-introduction-to-k-nearest-neighbors-algorithm-62214cea29c7
  - Be able to describe the KNN algorithm:
    - Answer:
      - KNN is a non-parametric approach for classification problems. It starts with storing all data points. Before prediction, we need to define a metric for distance. Distance metrics include Euclidean distance, Manhattan distance, cosine distance, etc. For each data point, we calculate the desired distance between this point and all the rest data from our set. We predict the label of this data point by taking the majority votes on the k-nearest points. We do this for all data points in our set. Typical k can be 5 or 10.
      - The weights of each vote can be scaled by the inverse of the pair distance, thus signing higher votes to the points that are nearer.
      - For regression problems, instead of votes, we apply mean of continuous target.

  - Describe the curse of dimensionality:
    - Answer:
      - The curse of dimensionality describes the sparsity in available data, when dimensionality increases drastically.
  - Recognize the conditions under which the curse may be problematic:
    - Answer:
      - In order to obtain a statistically sound and reliable result, the amount of data needed to support the result often grows exponentially with the dimensionality.
      - Organizing and searching data often relies on detecting areas where objects form groups with similar properties; in high dimensional data, however, all objects appear to be sparse and dissimilar in many ways, which prevents common data organization strategies from being efficient.
  - Enumerate strengths and weaknesses of KNN:
    - Advantage:
      1. Robust to noisy training data (especially if we use inverse square of weighted distance as the “distance”).  
      2. Can do well in practice with enough representative data.
      3. Flexible to feature / distance choices.
      4. Naturally handles multi-class cases.
    - Disadvantage:
      1. Need to determine value of parameter K (number of nearest neighbors).
      2. Distance based learning is not clear which type of distance to use and which attribute to use to produce the best results.
      3. Computation cost is quite high because we need to compute distance of each query instance to all training samples. Large search problem to find nearest neighbours.


  - Room for improvement:
    - KD tree for faster generalized N-point problems.
      http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KDTree.html#sklearn.neighbors.KDTree
    - `class sklearn.neighbors.KNeighborsClassifier`:
      - algorithm : {‘auto’, ‘ball_tree’, ‘kd_tree’, ‘brute’}, optional
      - leaf_size : int, optional (default = 30) Leaf size passed to BallTree or KDTree. This can affect the speed of the construction and query, as well as the memory required to store the tree. The optimal value depends on the nature of the problem.

# web_scraping.py:
- Define the MongoDB database and table
- Query the NYT API once
- Determine if the results are more than 100 pages
- Looping through the pages give the number of pages
- Scrape the meta data (link to article and put it into Mongo)
- Get all the links, visit the page and scrape the content

# web scraping with Python:

This is a tutorial stepping through the steps needed for you to get started
with scraping. Feel free to skip it if you are already comfortable with scraping.

1. Install the library needed from web scraping:

   - Beautiful Soup – A library designed for
     screen-scraping HTML and XML in Python

   ```
   pip install beautifulsoup4
   ```
2. We are going to scrape the BCS college football rankings as they appeared on the front page of [http://bcsfootball.org/](http://bcsfootball.org/) on December 8th, 2013. (The current version of the site does not include rankings.)
   - Fortunately, the Internet Archive Project keeps point-in-time snapshots of most popular web sites.
        - To see whether historical snapshots of `bcsfootball.org` are available, visit [http://web.archive.org/](http://web.archive.org/) and enter `http://www.bcsfootball.org/` into the search box.
        - Navigate to the correct date:
            1. Select the year 2013 on the activity graph.
            2. Choose December 8th on the calendar.
       - You should end up at this URL, which includes the date and time of the snapshot:
         [http://web.archive.org/web/20131208113724/http://www.bcsfootball.org/](http://web.archive.org/web/20131208113724/http://www.bcsfootball.org/)

3. Identify the table on the left hand side of the webpage, right click anywhere on it,
   and then select inspect element from the dropdown menu.

   ![football](images/football.png)

4. Now we are going to scrape that table using the following code. Type the code out line by line
   and run it yourself. Make sure you understand what each line is doing.

    ```python
    # Import a library to fetch web content using HTTP requests.
    import requests

    # Import a library to parse the HTML content of web pages.
    from bs4 import BeautifulSoup

    import pandas as pd

    # Use a snapshot of http://www.bcsfootball.org/ taken on December 8th, 2013.
    URL = "http://web.archive.org/web/20131208113724/http://www.bcsfootball.org/"

    # Get the HTML content of the web page as a string.
    content = requests.get(URL).content

    # Use a BeatifulSoup object to parse the HTML with "html.parser".
    soup = BeautifulSoup(content, "html.parser")

    # Find all <tr> elements (table rows) in the <tbody>
    # of the <table class="mod-data"> element.
    rows = soup.select('table.mod-data tbody tr')

    # Extract the text in each cell and put into a list of lists,
    # such that each list in the list represents content in a row.
    table_lst = []
    for row in rows:
        cell_lst = [cell.text for cell in row]
        table_lst.append(cell_lst)

    ranking = pd.DataFrame(table_lst)
    ranking.columns = ['ranking', 'state', 'score']
    print ranking.head()
    ```

5. Extra credit (optional):
    - Scrape the historical BCS Football rankings for a different date and time.
    - Given the name of a football stadium, scrape its location (latitude and longitude) from the stadium's Wikipedia page?

# ebay_scraping.py:
- helper function for getting class information out of the soup for a page
- get the source link for all images in a soup results object
- update image paths with a new prefix this is a function I use to make the code runnable from a remote directory
- Helper function to get soup from a live url, as opposed to a local copy
- downloads and opens an image from a url
- save images to specified directory (save_dir), if the directory does not exist yet it is created
- get the images from the soup of an ebay page, then save them locally

# Gradient Descent:
- GD example code for linear regression:
  - /Users/Victoria/galvanize/0_statistical_learning/gradient_descent/src
  - source: https://spin.atomicobject.com/2014/06/24/gradient-descent-linear-regression/

- Use cases for gradient descent:
  - Goal of gradient descent: to update a set of parameters in an iterative manner to minimize some cost/loss function.
  - Efficiency: if we consider linear regression, the explicit solution requires inverting a matrix which has complexity O(N3). This becomes prohibitive in the context of big data.
  - Convexity: a lot of problems in machine learning are convex, so using gradients ensure that we will get to the extrema. This is because for a convex function, gradient descent will always eventual converge given a small enough step size and infinite time.
  - Data streams: when data comes in as a continuous stream of items, without possibility to buffer all the samples, the closed form solution is not applicable.
  - Concept drift: when statistical properties of the data evolve over time, iterative training allows for smooth adaptation to new characteristics without completely forgetting the old ones.
  - Generalization: iterative method gives approximate solution and this is a feature, not a bug, because this makes the algorithm more robust against outliers and improves accuracy on new unseen examples.

- Compare and contrast gradient descent and stochastic gradient descent:
  - While in GD, you have to run through all the samples in your training set to do a single update for a parameter in a particular iteration, in SGD, on the other hand, you use ONLY ONE or SUBSET of training sample from your training set to do the update for a parameter in a particular iteration. If you use SUBSET, it is called Minibatch Stochastic gradient Descent.
  - SGD often converges much faster compared to GD but the error function is not as well minimized as in the case of GD. Often in most cases, the close approximation that you get in SGD for the parameter values are enough because they reach the optimal values and keep oscillating there.
  - Pseudocode and python code: https://www.pyimagesearch.com/2016/10/17/stochastic-gradient-descent-sgd-with-python/

# Dimensionality Reduction
- Reasons for reducing the dimensions:
  - It helps in data compressing and reducing the storage space required.
  - It fastens the time required for performing same computations. Less dimensions leads to less computing, also less dimensions can allow usage of algorithms unfit for a large number of dimensions.
  - It takes care of multi-collinearity that improves the model performance. It removes redundant features. For example: there is no point in storing a value in two different units (meters and inches).
  - Reducing the dimensions of data to 2D or 3D may allow us to plot and visualize it precisely. You can then observe patterns more clearly. Below you can see that, how a 3D data is converted into 2D. First it has identified the 2D plane then represented the points on these two new axis z1 and z2.
  - It is helpful in noise removal also and as result of that we can improve the performance of models.

- PCA in reducing dimensionality:
  -  Principal Component Analysis (PCA): In this technique, variables are transformed into a new set of variables, which are linear combination of original variables. These new set of variables are known as principle components. They are obtained in such a way that first principle component accounts for most of the possible variation of original data after which each succeeding component has the highest possible variance.
  - The second principal component must be orthogonal to the first principal component. In other words, it does its best to capture the variance in the data that is not captured by the first principal component. For two-dimensional dataset, there can be only two principal components. Below is a snapshot of the data and its first and second principal components. You can notice that second principle component is orthogonal to first principle component.
  - The principal components are sensitive to the scale of measurement, now to fix this issue we should always standardize variables before applying PCA. Applying PCA to your data set loses its meaning. If interpretability of the results is important for your analysis, PCA is not the right technique for your project.

- How to determine how many principal components to keep?
  - To decide how many eigenvalues/eigenvectors to keep, you should consider your reason for doing PCA in the first place. Are you doing it for reducing storage requirements, to reduce dimensionality for a classification algorithm, or for some other reason? If you don't have any strict constraints, I recommend plotting the cumulative sum of eigenvalues (assuming they are in descending order). If you divide each value by the total sum of eigenvalues prior to plotting, then your plot will show the fraction of total variance retained vs. number of eigenvalues. The plot will then provide a good indication of when you hit the point of diminishing returns (i.e., little variance is gained by retaining additional eigenvalues).

- Relationship between PCA and SVD:
  - PCA is map the data to lower dimensional. In order for PCA to do that it should calculate and rank the importance of features/dimensions. There are 2 ways to do so.
  - Using eigenvalue and eigenvector in covariance matrix to calculate and rank the importance of features
  - Using SVD on covariance matrix to calculate and rank the importance of the features SVD (covariance matrix) = [U S V']
  - After ranking the features/ dimensions then it will choose the most important ones (k) and map the actual data to k dimension.
  - In case PCA used SVD to rank the importance of features, then U matrix will have all features ranked, we choose the first k columns which represent the most important one.

# Tuning hyper-parameters
  - General Approach for Parameter Tuning
    - There are two types of parameter to be tuned here – tree based and boosting parameters.
      1. Choose a relatively high learning rate. Generally the default value of 0.1 works but somewhere between 0.05 to 0.2 should work for different problems
      2. Determine the optimum number of trees for this learning rate. This should range around 40-70. Remember to choose a value on which your system can work fairly fast. This is because it will be used for testing various scenarios and determining the tree parameters.
      3. Tune tree-specific parameters for decided learning rate and number of trees. Note that we can choose different parameters to define a tree and I’ll take up an example here.
      4. Lower the learning rate and increase the estimators proportionally to get more robust models.
      5. Fix learning rate and number of estimators for tuning tree-based parameters
      6. In order to decide on boosting parameters, we need to set some initial values of other parameters. Lets take the following values:
          1. min_samples_split = 500 : This should be ~0.5-1% of total values. Since this is imbalanced class problem, we’ll take a small value from the range.
          2. min_samples_leaf = 50 : Can be selected based on intuition. This is just used for preventing overfitting and again a small value because of imbalanced classes.
          3. max_depth = 8 : Should be chosen (5-8) based on the number of observations and predictors. This has 87K rows and 49 columns so lets take 8 here.
          4. max_features = ‘sqrt’ : Its a general thumb-rule to start with square root.
          5. subsample = 0.8 : This is a commonly used used start value.
    Please note that all the above are just initial estimates and will be tuned later. Lets take the default learning rate of 0.1 here and check the optimum number of trees for that. For this purpose, we can do a grid search and test out values from 20 to 80 in steps of 10.

# Classification Model Pros and Cons (Generalized)

  * Logistic Regression
  	* Pros
  		* low variance
  		* provides probabilities for outcomes
  		* works well with diagonal (feature) decision boundaries
  		* NOTE: logistic regression can also be used with kernel methods
  	* Cons
  		* high bias
  * Decision Trees
  	* Regular (not bagged or boosted)
  		* Pros
  			* easy to interpret visually when the trees only
  				contain several levels
  			* Can easily handle qualitative (categorical) features
  			* Works well with decision boundaries parellel to the feature axis
  		* Cons
  			* prone to overfitting
  			* possible issues with diagonal decision boundaries
  	* Bagged Trees : train multiple trees using bootstrapped data
  		to reduce variance and prevent overfitting
  		* Pros
  			* reduces variance in comparison to regular decision trees
  			* Can provide variable importance measures
  				* classification: Gini index
  				* regression: RSS
  			* Can easily handle qualitative (categorical) features
  			* Out of bag (OOB) estimates can be used for model validation
  		* Cons
  			* Not as easy to visually interpret
  			* Does not reduce variance if the features are correlated
  	* Boosted Trees : Similar to bagging, but learns sequentially and builds off
  		previous trees
  		* Pros
  			* Somewhat more interpretable than boosted trees/random forest
  				as the user can define the size of each tree resulting in
  				a collection of stumps (1 level) which can be viewed as an additive model
  			* Can easily handle qualitative (categorical) features
  		* Cons
  			* Unlike boosting and random forests, can overfit if number of trees is too large
  * Random Forest
  	* Pros
  		* Decorrelates trees (relative to boosted trees)
  			* important when dealing with mulitple features which may be correlated
  		* reduced variance (relative to regular trees)
  	* Cons
  		* Not as easy to visually interpret
  * SVM
  	* Pros
  		* Performs similarly to logistic regression when linear separation
  		* Performs well with non-linear boundary depending on the kernel used
  		* Handle high dimensional data well
  	* Cons
  		* Susceptible to overfitting/training issues depending on kernel
  * Neural Network (This section needs further information based on
  	different types of NN's)
  * Naive Bayes
  	* Pros
  		* Computationally fast
  		* Simple to implement
  		* Works well with high dimensions
  	* Cons
  		* Relies on independence assumption and will perform
  			badly if this assumption is not met


# Summary of Progress:
- [x] completed:
  1. sklearn Workflow
  2. KFold
  3. ab_testing
  4. bayesian
  5. cv_regularization
  6. linear_algebra_eda
  7. linear_regression
  8. logistic
  9. multi_armed_bandit
  10. pandas_1
  11. pandas_2
  12. plotting
  13. power_analysis
  14. probability
  15. sampling_estimation
  16. decision_trees
  17. random_forest
  18. Boosting
  19. knn
  20. gradient descent
  21. web scraping


- [ ] uncompleted:
  1. nlp
  2. cost-benefit matrix
  3. naive Bayes
  4. Clustering
  5. dimensionality reduction
  6. Non-negative matrix factorization
  7. Recommender system
