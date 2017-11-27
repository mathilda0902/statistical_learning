# Linear Regression Diagnosis

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
