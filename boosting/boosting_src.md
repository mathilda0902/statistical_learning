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

3. adaboost.py:
    - staged misclassification rate
    - grid search
    - feature importances
