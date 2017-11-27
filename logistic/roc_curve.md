# ROC (Receiver Operating Characteristic) curve

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
