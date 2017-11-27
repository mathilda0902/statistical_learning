# A/B Testing
## CTR t-test:

- Click Through Rate: Clicks / Impressions
- Independent t test:
  - scs.ttest_ind(df_ctr_signed_in['CTR'].dropna(), df_ctr_not_signed['CTR'].dropna(), equal_var = False)
- p value:
  - p_val = stats.ttest_ind(group_1_df['CTR'], group_2_df['CTR'], equal_var=False)[1]
- By gender, by signed in vs. not signed in, and by age groups.


# Experimental Design
## Conversion rate z-test:



# Bayesian Analysis

- Flip a fair coin: flip_coin.py
- Flip a biased coin: flip_biased_coin.py
- Bayesian update: bayes.py
    1. update priors
    2. normalize
    3. print distribution
    4. plot



# Cross Validation

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


# Regularization: Ridge (Shrinkage) and Lasso
## Functions:

1. `mse_ridge` and `mse_lasso`: mean squared error of Ridge / Lasso Regression
2. `plot_ridge`: Plot the parameters (coefficients) of the Ridge regression (y-axis) versus the value of various alpha parameter.
3. `plot_error`: Plot the validation error and training error curves for Ridge regression with different alpha parameters.
4. `plot_lass`: Make a plot of the training error and the validation error as a function of the alpha parameter.
5. `best_choice`: Select a model based on the validation and training error curves.
