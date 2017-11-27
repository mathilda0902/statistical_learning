from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np

'''Question 2: importing data and train split:'''
def load_and_split_data():
    ''' Loads sklearn's boston dataset and splits it into train:test datasets
        in a ratio of 80:20. Also sets the random_state for reproducible
        results each time model is run.

        Parameters: None
        Returns:  (X_train, X_test, y_train, y_test):  tuple of numpy arrays
                  column_names: numpy array containing the feature names
    '''
    boston = load_boston() #load sklearn's dataset
    X, y = boston.data, boston.target
    column_names = boston.feature_names
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                       test_size = 0.2,
                                       random_state = 1)
    return (X_train, X_test, y_train, y_test), column_names


'''Question 3: 3 classes of algorithm'''
rf = RandomForestRegressor(n_estimators=100,
                            n_jobs=-1,
                            random_state=1)

gdbr = GradientBoostingRegressor(learning_rate=0.1,
                                  loss='ls',
                                  n_estimators=100,
                                  random_state=1)

abr = AdaBoostRegressor(DecisionTreeRegressor(),
                         learning_rate=0.1,
                         loss='linear',
                         n_estimators=100,
                         random_state=1)

k_fold = KFold(n_splits=5, shuffle=True)

def cv_mse_r2(model):
    ''' Takes an instantiated model (estimator) and returns the average
        mean square error (mse) and coefficient of determination (r2) from
        kfold cross-validation.
        Parameters: estimator: model object
                    X_train: 2d numpy array
                    y_train: 1d numpy array
                    nfolds: the number of folds in the kfold cross-validation
        Returns:  mse: average mean_square_error of model over number of folds
                  r2: average coefficient of determination over number of folds
    '''

    mse_score = cross_val_score(model.fit(X, y), X, y, cv=k_fold, n_jobs=-1, scoring='neg_mean_squared_error')
    r2_score = cross_val_score(model.fit(X, y), X, y, cv=k_fold, n_jobs=-1, scoring='r2')
    name = estimator.__class__.__name__
    mean_mse = mse_score.mean()*(-1)
    mean_r2 = r2_score.mean()
    print "{0:<25s} Train CV | MSE: {1:0.3f} | R2: {2:0.3f}".format(name,
                                                        mean_mse, mean_r2)
    return mean_mse, mean_r2



'''
Run 1:
RandomForestRegressor     Train CV | MSE: 13.4483234028 | R2: 0.849900348747
GradientBoostingRegressor Train CV | MSE: 11.2118353539 | R2: 0.889732149272
AdaBoostRegressor         Train CV | MSE: 12.4464181712 | R2: 0.887662144835

Run 2:
RandomForestRegressor     Train CV | MSE: 10.2677996721 | R2: 0.85277329344
GradientBoostingRegressor Train CV | MSE: 23.208326717 | R2: 0.760150589534
AdaBoostRegressor         Train CV | MSE: 10.2089708794 | R2: 0.88166211799
'''

'''Question 5'''
gdbr2 = GradientBoostingRegressor(learning_rate=1,
                                  loss='ls',
                                  n_estimators=100,
                                  random_state=1)

mse_score2 = cross_val_score(gdbr2.fit(X, y), X, y, cv=k_fold, n_jobs=-1, scoring='neg_mean_squared_error')

'''
New GradientBoostingRegressor MSE:  21.4560657512.
New GradientBoostingRegressor MSE:  18.5662682919
New GradientBoostingRegressor MSE:  22.5586078845
New GradientBoostingRegressor MSE:  16.0524301292
> GradientBoostingRegressor Train CV | MSE: 9.01474037521
> GradientBoostingRegressor Train CV | MSE: 11.2118353539
increased MSE.
'''

'''Question 6'''
def stage_score_plot(estimator, X_train, y_train, X_test, y_test):
    '''
    Parameters: estimator: GradientBoostingRegressor or AdaBoostRegressor
                X_train: 2d numpy array
                y_train: 1d numpy array
                X_test: 2d numpy array
                y_test: 1d numpy array

    Returns: A plot of the number of iterations vs the MSE for the model for
    both the training set and test set.
    '''
    if estimator == gdbr:
        model = gdbr.fit(X_train, y_train)
    elif estimator == abr:
        model = abr.fit(X_train, y_train)
    predictions_train = list(model.staged_predict(X_train))
    predictions_test = list(model.staged_predict(X_test))
    mse_train = []
    mse_test = []
    for i in range(100):
        mse_train.append(mean_squared_error(y_train, predictions_train[i]))
        mse_test.append(mean_squared_error(y_test, predictions_test[i]))
    x_axis = np.linspace(0,100,100)
    mse_rf = mean_squared_error(y_test, rf.fit(X_train, y_train).predict(X_test))
    plt.figure(figsize=(6,4))
    plt.plot(x_axis, mse_train, 'b--', alpha=0.3,
                    label='{} - {}'.format(model.__class__.__name__, model.learning_rate))

    plt.plot(x_axis, mse_test, alpha=0.3, color='g',
                    label='{} - {}'.format(model.__class__.__name__, model.learning_rate))
    plt.axhline(y=mse_rf, color='y', linestyle='-.', label='Random Forest Test')
    plt.xlim(0, 100)
    plt.ylim(0, 90)
    plt.ylabel('MSE')
    plt.xlabel('Iterations')
    plt.title('{}'.format(model.__class__.__name__))



gdbr = GradientBoostingRegressor(learning_rate=0.1, loss='ls',
                    n_estimators=100, random_state=1)

gdbr = GradientBoostingRegressor(learning_rate=1, loss='ls',
                    n_estimators=100, random_state=1)

abr = AdaBoostRegressor(DecisionTreeRegressor(),
                         learning_rate=0.1,
                         loss='linear',
                         n_estimators=100,
                         random_state=1)

rf = RandomForestRegressor(n_estimators=100,
                            n_jobs=-1,
                            random_state=1)
stage_score_plot(abr, X_train, y_train, X_test, y_test)
plt.legend()
plt.show()


'''Grid search'''
random_forest_grid = {'max_depth': [3, None],
                      'max_features': ['sqrt', 'log2', None],
                      'min_samples_split': [2, 4],
                      'min_samples_leaf': [1, 2, 4],
                      'bootstrap': [True, False],
                      'n_estimators': [10, 20, 40, 80],
                      'random_state': [1]}

rf_gridsearch = GridSearchCV(RandomForestRegressor(),
                             random_forest_grid,
                             n_jobs=-1,
                             verbose=True,
                             scoring='mean_squared_error')
rf_gridsearch.fit(X_train, y_train)

print "best parameters:", rf_gridsearch.best_params_

best_rf_model = rf_gridsearch.best_estimator_

mean_squared_error(y_test, best_rf_model.predict(X_test))

'''result:
[Parallel(n_jobs=-1)]: Done 864 out of 864 | elapsed:  1.2min finished
best parameters: {'bootstrap': True, 'min_samples_leaf': 1,
'n_estimators': 40, 'min_samples_split': 2, 'random_state': 1,
'max_features': None, 'max_depth': None}
'''
'''
mse: 8.816969301470591
'''


'''base GradinetBoosting grid search:'''
gdbr_param_grid = {'n_estimators': [10000],
                  'min_samples_leaf': [7, 9, 13],
                  'max_depth': [4, 5, 6, 7],
                  'max_features': [5, 8, 13],
                  'learning_rate': [0.05, 0.02, 0.01],
                  }
gdbr_gridsearch = GridSearchCV(GradientBoostingRegressor(),
                    gdbr_param_grid,
                    n_jobs=-1,
                    verbose=True,
                    scoring='mean_squared_error')
gdbr_gridsearch.fit(X_train, y_train)

print 'best parameters for Gradient Boosting Grid Search: ', gdbr_gridsearch.best_params_

best_gdbr_model = gdbr_gridsearch.best_estimator_

mse = mean_squared_error(y_test, best_gdbr_model.predict(X_test))

'''
mse: 10.009659860183021
'''



if __name__ == '__main__':
#    print "RandomForestRegressor     Train CV | MSE: {} | R2: {}".format(cv_mse_r2(rf)[0], cv_mse_r2(rf)[1])
#    print "GradientBoostingRegressor Train CV | MSE: {} | R2: {}".format(cv_mse_r2(gdbr)[0], cv_mse_r2(gdbr)[1])
#    print "AdaBoostRegressor         Train CV | MSE: {} | R2: {}".format(cv_mse_r2(abr)[0], cv_mse_r2(abr)[1])
    print "New GradientBoostingRegressor MSE: ", mse_score2.mean()*(-1)
    print "staged_predict: ", gdbr.staged_predict(X_train)
