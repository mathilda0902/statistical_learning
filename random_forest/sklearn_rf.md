# random_forest - Random forest

1. roc.py:
  - using KFold
  - input: X, y, clf_class, `**kwargs`
  - generate: fpr, tpr, thresholds
  - plot: mean_fpr, mean_tpr

2. sklearn_rf.py:
  - fitting random forest using default keywords.
  - get: accuracy score, confusion matrix, precision score, recall score.
  - change `oob` to True. compare out-of-bag training accuracy score to test set.
  - feature importances
  - Calculate the standard deviation for feature importances across all trees
  - # of trees and accuracy score
  - max features parameter and accuracy score
  - function `get_score(classifier, X_train, X_test, y_train, y_test, **kwargs)`:
    - return model.score(X_test, y_test), precision_score(y_test, y_predict), recall_score(y_test, y_predict)
