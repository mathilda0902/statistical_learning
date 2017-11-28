import numpy as np


class GradientDescent(object):
    """Preform the gradient descent optimization algorithm for an arbitrary
    cost function.
    """

    def __init__(self, cost, gradient, predict, fit_intercept=True,
                 alpha=0.01,
                 num_iterations=10000):
        """Initialize the instance attributes of a GradientDescent object.

        Parameters
        ----------
        cost: The cost function to be minimized.
        gradient: The gradient of the cost function.
        predict_func: A function to make predictions after the optimization has
            converged.
        alpha: The learning rate.
        num_iterations: Number of iterations to use in the descent.

        Returns
        -------
        self: The initialized GradientDescent object.
        """
        # Initialize coefficients in run method once you know how many features
        # you have.
        self.coeffs = None
        self.cost = cost
        self.gradient = gradient
        self.predict = predict
        self.alpha = alpha
        self.num_iterations = num_iterations
        self.fit_intercept = fit_intercept

    def fit(self, X, y):
        """Run the gradient descent algorithm for num_iterations repetitions.

        Parameters
        ----------
        X: A two dimenstional numpy array.  The training data for the
            optimization.
        y: A one dimenstional numpy array.  The training response for the
            optimization.

        Returns
        -------
        self:  The fit GradientDescent object.
        """
        self.coeffs = np.zeros(len(X[0]))
        for k in range(self.num_iterations):
            self.coeffs = self.coeffs - self.alpha * self.gradient(X, y, self.coeffs)
        return self

    def predict(self, X):
        """Call self.predict_func to return predictions.

        Parameters
        ----------
        X: Data to make predictions on.

        Returns
        -------
        preds: A one dimensional numpy array of predictions.
        """
        return self.predict

    def add_intercept(self, X):
        '''Add an intercept column to a matrix X.

        Parameters
        ----------
        X: A two dimensional numpy array.

        Returns
        ----------
        X: The original matrix X, but with a constant column of 1's appended.
        '''
        new_col = np.ones(len(X))
        new_col = new_col.reshape((len(X),1))
        X = np.append(X, new_col, 1)
        return X

if __name__ == '__main__':
    pass
