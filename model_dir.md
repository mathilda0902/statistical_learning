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

# Bias and Variances
- If you get low training error and high testing error, what do you think is happening and what can you do about it?
- The variance of the training model is getting high. This is an indicator that the model that is being tested fits very well to the training data set, with possibly minimum bias. The low bias is reflected in the low training errors. If we would like to lower our testing error when applying our model to the testing set, we will have to sacrifice some variance of our model to achieve this.

- What is the RSS for a model's predictions?
- eg: The difference between each point and its prediction, the residual, is: [4.7, 0.1, -5.7, -7.8, 8.6], which we will then square to get [22.09, 0.01, 32.49, 60.84, 73.96], and sum, to arrive at an RSS of 189.89.

# cv_regularization - Cross Validation
- Why do we often prefer to use k-Fold Cross Validation instead of a simple train/test split?
- In k-fold cross-validation, the original sample is randomly partitioned into k equal sized subsamples. Of the k subsamples, a single subsample is retained as the validation data for testing the model, and the remaining k − 1 subsamples are used as training data. The cross-validation process is then repeated k times (the folds), with each of the k subsamples used exactly once as the validation data. The k results from the folds can then be averaged to produce a single estimation. The advantage of this method over repeated random sub-sampling is that all observations are used for both training and validation, and each observation is used for validation exactly once. 10-fold cross-validation is commonly used, but in general k remains an unfixed parameter.


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

6. Beta coefficient interpretation:
  - Increasing the `GRE score` by `1 point` increases the chance of getting in `by a factor` of `1.00189`.
  - What change is required to double my chances of admission?
  - `log(2) / coef`: Increasing the GRE score by 367 points doubles the chance of getting in.

7. Compute the odds (p/(1-p)):
  ```
  probabilities_rank = model.predict_proba(X_rank)[:, 1]
  for rank, prob in izip(ranks, probabilities_rank):
    print "rank: %d, probability: %f, odds: %f" % (rank, prob, prob / (1 - prob))
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


# decision_trees - Decision Trees
1. Decision trees implementation
2. Decision trees regressor
3. knn
4. Recursion practice
5. notes on sklearn and its practical use:
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

# K Nearest Neighbors (knn):
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
      2. Effective if the training data is large.
    - Disadvantage:
      1. Need to determine value of parameter K (number of nearest neighbors).
      2. Distance based learning is not clear which type of distance to use and which attribute to use to produce the best results.
      3. Computation cost is quite high because we need to compute distance of each query instance to all training samples.

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


# ebay_scraping.py:
- helper function for getting class information out of the soup for a page
- get the source link for all images in a soup results object
- update image paths with a new prefix this is a function I use to make the code runnable from a remote directory
- Helper function to get soup from a live url, as opposed to a local copy
- downloads and opens an image from a url
- save images to specified directory (save_dir), if the directory does not exist yet it is created
- get the images from the soup of an ebay page, then save them locally





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
  1. gradient descent (to do)
  2. nlp
  3. web scraping
  4. cost-benefit matrix
  5. naive Bayes
  6. Clustering
