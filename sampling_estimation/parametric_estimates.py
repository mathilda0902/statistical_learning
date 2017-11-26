import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns


def plot_hist_basic(df, col):
    """Return a Matplotlib axis object with a histogram of the data in col.
    Plots a histogram from the column col of dataframe df.
    Parameters
    ----------
    df: Pandas DataFrame
    col: str
        Column from df with numeric data to be plotted
    Returns
    -------
    ax: Matplotlib axis object
    """
    data = df[col]
    ax = data.hist(bins=20, normed=1, edgecolor='none', figsize=(10, 7), alpha=.5)
    ax.set_ylabel('Probability Density')
    ax.set_title(col)

    return ax


def get_sample_mean_var(df, col):
    """Calculate sample mean and sample variance of a 1-d array (or Series).
    Parameters
    ----------
    df: Pandas DataFrame
    col: str
        Column from df with numeric data to be plotted
    Returns
    -------
    tuple: (sample mean (float), sample variance (float))
    """

    # by default np.var returns population variance.
    # ddof=1 to get sample var (ddof: delta degrees of freedom)
    data = df[col]
    return data.mean(), data.var(ddof=1)

class methond_of_moments(object):
    """Cacluates and plots gamma and normal model method of moments estimates"""

    def __init__(self):
        """Construct methond_of_moments class"""
        pass

    def fit(self, df, col):
        """Fit Normal and Gamma models to the data using Method of Moments
        Parameters
        ----------
        df: Pandas DataFrame
        col: str
             Column from df with numeric data for Method of Moments
             distribution estimation and plotting
        """

        self.df = df
        self.col = col
        self.samp_mean, self.samp_var = get_sample_mean_var(self.df, self.col)
        self._fit_gamma()
        self._fit_normal()

    def _fit_gamma(self):
        """Fit Normal and Gamma models to the data using Method of Moments"""
        self.alpha = self.samp_mean**2 / self.samp_var
        self.beta = self.samp_mean / self.samp_var

    def _fit_normal(self):
        """Fit Normal and Gamma models to the data using Method of Moments"""
        self.samp_std = self.samp_var**0.5

    def plot_pdf(self, ax=None, gamma=True, normal=True, xlim=None, ylim=None):
        """Plot distribution PDFs using MOM against histogram of data in df[col].
        Parameters
        ----------
        ax: Matplotlib axis object, optional (default=None)
            Used for creating multiple subplots
        gamma: boolean, optional (default=True)
               Fit and plot a Gamma Distribution
        normal: boolean, optional (default=True)
                Fit and plot a Normal Distribution
        xlim: None, or two element tuple
              If not 'None', these limits are used for the x-axis
        ylim: None, or two element tuple
              If not 'None', these limits are used for the y-axis
        Returns
        -------
        ax: Matplotlib axis object
        """

        if ax is None:
            ax = plot_hist_basic(self.df, self.col)

        x_vals = np.linspace(self.df[self.col].min(), self.df[self.col].max())

        if gamma:
            gamma_rv = stats.gamma(a=self.alpha, scale=1/self.beta)
            gamma_p = gamma_rv.pdf(x_vals)
            ax.plot(x_vals, gamma_p, color='b', label='Gamma MOM', alpha=0.6)

        if normal:
            # scipy's scale parameter is standard dev.
            normal_rv = stats.norm(loc=self.samp_mean, scale=self.samp_std)
            normal_p = normal_rv.pdf(x_vals)
            ax.plot(x_vals, normal_p, color='k', label='Normal MOM', alpha=0.6)

        ax.set_ylabel('Probability Density')
        ax.legend()

        if not xlim is None:
            ax.set_xlim(*xlim)

        if not ylim is None:
            ax.set_ylim(*ylim)

        return ax



def plot_year(df, cols, estimation_method_list, gamma=True, normal=True):
    """Loop over 12 columns of data and plot fits to each column.
    Requires plot_method_of_moments or maximum_likelihood_estimation objects
    Parameters
    ----------
    df: Pandas DataFrame
    cols: list of str
        Columns from df with numeric data to be plotted
    estimation_method_list: list of model estimation objects
        plot_method_of_moments or maximum_likelihood_estimation objects
    gamma: boolean, optional (default=True)
        Fit and plot a Gamma Distribution
    normal: boolean, optional (default=True)
        Fit and plot a Normal Distribution
    Returns
    -------
    ax: 4x3 list of Matplotlib axis objects
    """
    # assuming we are plotting a 3x4 grid, so check
    if len(cols) != 12:
        ex_str = 'Expecting 12 monthly columns, got: {}'.format(len(cols))
        raise Exception(ValueError)

    # Pandas does a lexicographic sort on the column names when plotting
    # multiple histograms (with no obvious way to turn that off).
    # So to ensure consistency between the histograms and the fits,
    # use the sorted version of the columns here and in plotting the fits.
    #
    # The three lines indicated below for removal and the  inclusion
    # of the following two will implement the above discussed solution.
    # cols_srt = sorted(cols)
    # axes = df[cols_srt].hist(bins=20, normed=1,
    #                          grid=0, edgecolor='none',
    #                          figsize=(15, 10),
    #                          layout=(3,4))

    # The following two lines removed if lexographical ordering preferred
    fig, axes = plt.subplots(4,3, figsize=(15,10))
    cols_srt = cols
    for month, ax in zip(cols_srt, axes.flatten()):
        # the following line removed if lexographical ordering preferred
        ax.hist(df[month], bins=20, normed=1, edgecolor='none')
        for estimation_method in estimation_method_list:
            estimation_method.fit(df, month)
            estimation_method.plot_pdf(ax=ax, normal=normal, gamma=gamma,
                                       xlim=[0, 16], ylim=[0, 0.35])
    plt.tight_layout()

    return axes


class likelihood_estimation(object):
    """Calculates and plots likelihoods for the Poisson distribution paramter"""

    def __init__(self):
        """Construct likelihood_estimation class"""
        pass

    def _calculate_likelihood(self, lam, ks):
        """Compute the poisson log likelihood of observing ks for paramater lam.
        Parameters
        ----------
        lam: float
            Lambda rate parameter of a Poisson distribution
        ks: Numpy array
            Discrete count data observations assumed to be from a Poisson distribution
            Returns
            -------
            likelihood: float
            """
        return stats.poisson(lam).pmf(ks)


    def fit(self, data, lambda_range):
        """Approximate the log likelihood function for Poisson distrubtion given data
        Parameters
        ----------
        data: Numpy array
            Discrete count data observations assumed to be from a Poisson distribution
        lambda_range: Numpy array
            Different rate parameters (lambda values) at which to evaluate likelihood
            Possibly created using np.linspace
        """

        self.lambda_range = lambda_range
        self.sum_logs = []
        for lam in lambda_range:
            likes = self._calculate_likelihood(lam, data)
            sum_log_liklihood = np.sum(np.log10(likes))
            self.sum_logs.append(sum_log_liklihood)


    def get_maximum_likelihood_estimate(self):
        """Returns the MLE for the poisson distribution of the current fit of the data
        This method should be called after the .fit method
        Returns
        -------
        float: lambda value corresponding to the Maximum Likelihood of the data
            or a message to use the .fit method first.
        """

        maxlik_ind = np.argmax(self.sum_logs)
        return self.lambda_range[maxlik_ind]


    def plot_likelihood_function(self):
        """Plot the estimated log likelihood function
        Returns
        -------
        ax: Matplotlib axis object
        """

        fig, ax = plt.subplots(figsize=(7, 6))
        ax.plot(self.lambda_range, self.sum_logs)
        ax.set_ylabel('$log(L(x | \lambda))$', fontsize=14)
        ax.set_xlabel('$\lambda$', fontsize=14)

        return ax


class maximum_likelihood_estimation(object):
    """Cacluates and plots gamma and normal model maximum likelihood estimation"""

    def __init__(self):
        """Construct maximum_likelihood_estimation class"""
        pass

    def fit(self, df, col):
        """Fit Normal/Gamma models to the data using Maximum Likelihood Estimation
        Parameters
        ----------
        df: Pandas DataFrame
        col: str
             Column from df with numeric data for Maximum Likelihood
             Estimation distribution estimation and plotting
        """
        self.df = df
        self.col = col
        self._fit_gamma()
        self._fit_normal()

    def _fit_gamma(self):
        """Fit Gamma model to the data using Maximum Likelihood Estimation"""
        self.ahat, self.loc, self.bhat = stats.gamma.fit(self.df[self.col], floc=0)

    def _fit_normal(self):
        """Fit Normal model to the data using Maximum Likelihood Estimation"""
        self.mean_mle, self.std_mle = stats.norm.fit(self.df[self.col])


    def plot_pdf(self, ax=None, gamma=True, normal=True, xlim=None, ylim=None):
        """Plot distribution PDFs using MLE against histogram of data in df[col].
        Use Maximum Likelihood Estimation to fit Normal and/or Gamma Distributions
        to the data in df[col] and plot their PDFs against a histogram of the data.
        Parameters
        ----------
        ax: Matplotlib axis object, optional (default=None)
            Used for creating multiple subplots
        gamma: boolean, optional (default=True)
               Fit and plot a Gamma Distribution
        normal: boolean, optional (default=True)
                Fit and plot a Normal Distribution
        xlim: None, or two element tuple
              If not 'None', these limits are used for the x-axis
        ylim: None, or two element tuple
              If not 'None', these limits are used for the y-axis
        Returns
        -------
        ax: Matplotlib axis object
        """

        if ax is None:
            print 'sup'
            ax = plot_hist_basic(self.df, self.col)

        x_vals = np.linspace(self.df[self.col].min(), self.df[self.col].max())

        if gamma:
            gamma_rv = stats.gamma(a=self.ahat, loc=self.loc, scale=self.bhat)
            gamma_p = gamma_rv.pdf(x_vals)
            ax.plot(x_vals, gamma_p, color='k', alpha=0.7, label='Gamma MLE')

        if normal:
            normal_rv = stats.norm(loc=self.mean_mle, scale=self.std_mle)
            normal_p = normal_rv.pdf(x_vals)
            ax.plot(x_vals, normal_p, color='b', label='Normal MLE', alpha=0.6)

        ax.set_ylabel('Probability Density')
        ax.legend()

        if not xlim is None:
            ax.set_xlim(*xlim)

        if not ylim is None:
            ax.set_ylim(*ylim)

        return ax


def plot_kde(df, col):
    """Fit a Gaussian KDE to input data, plot fit over histogram of the data.
    Parameters
    ----------
    df: Pandas DataFrame
    col: str
        Column from df with numeric data to be plotted
    Returns
    -------
    ax: Matplotlib axis object
    """
    ax = plot_hist_basic(df, col)
    data = df[col]
    density = stats.kde.gaussian_kde(data)
    x_vals = np.linspace(data.min(), data.max(), 100)
    kde_vals = density(x_vals)

    ax.plot(x_vals, kde_vals, 'b-')

    return ax
