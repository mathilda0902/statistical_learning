import numpy as np


def predict_proba(X, coeffs):
    """Calculate the predicted conditional probabilities (floats between 0 and
    1) for the given data with the given coefficients.

    Parameters
    ----------
    X: A 2 dimensional numpy array.  The data (independent variables) to use
        for prediction.
    coeffs: A 1 dimensional numpy array, the hypothesised coefficients.  Note
        that the shape of X and coeffs must align.

    Returns
    -------
    predicted_probabilities: The conditional probabilities from the logistic
        hypothosis function given the data and coefficients.

    """
    if len(X[0]) == len(coeffs):
        n_th = X.dot(coeffs)
        return 1/(1+np.exp(- n_th))
    else:
        return 'Dimensional Error.'


def predict(X, coeffs, thresh=0.5):
    """
    Calculate the predicted class values (0 or 1) for the given data with the
    given coefficients by comparing the predicted probabilities to a given
    threshold.

    Parameters
    ----------
    X: A 2 dimensional numpy array.  The data (independent variables) to use
        for prediction.
    coeffs: A 1 dimensional numpy array, the hypothesised coefficients.  Note
        that the shape of X and coeffs must align.
    thresh: Threshold for comparison of probabilities.

    Returns
    -------
    predicted_class: The predicted class.
    """
    probs = predict_proba(X, coeffs)
    return probs >= thresh

def cost(X, y, coeffs):
    """
    Calculate the logistic cost function of the data with the given
    coefficients.

    Parameters
    ----------
    X: A 2 dimensional numpy array.  The data (independent variables) to use
        for prediction.
    y: A 1 dimensional numpy array.  The actual class values of the response.
        Must be encoded as 0's and 1's.  Also, must align properly with X and
        coeffs.
    coeffs: A 1 dimensional numpy array, the hypothesised coefficients.  Note
        that the shape of X, y, and coeffs must align.

    Returns
    -------
    logistic_cost: The computed logistic cost.
    """
    cost1 = - np.log(predict_proba(X, coeffs)) * y
    cost2 = - np.log(1 - predict_proba(X, coeffs)) * (1 - y)
    cost = cost1 + cost2
    cost = cost.sum() / float(len(X[0]))
    return cost

def gradient(X, y, coeffs):
    gradient = []
    for i in range(len(coeffs)):
        derivative_i = (predict_proba(X, coeffs) - y) * X[:, i]
        derivative = derivative_i.sum()
        gradient.append(derivative)
    return np.array(gradient)



if __name__ == '__main__':
    pass
