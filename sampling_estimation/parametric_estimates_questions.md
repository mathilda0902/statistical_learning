**Include your answers in** `morning_answer.py`.

## Part 1: Distribution Selection

In some scenarios selecting a distribution that will appropriately model the 
random variable we are interested in is easy. E.g. Binomial distributions 
model the number of heads in 50 flips of a fair coin. In other scenarios, 
however, distribution selection is not as obvious. In such cases, plotting 
the distribution often helps. For example, in this exercise the random 
variable is the **amount of rainfall in a given month** and the mechanism 
that controls the random variable is unclear. 

<br>

1. The file `data/rainfall.csv` contains rainfall data for Nashville
   measured since 1871. Load the data into a dataframe.

2. Plot a histogram of *January's* rainfall. After evaluating the data
   characteristics, identify which two of candidate distributions below would 
   be appropriate models for the data and explain why.

   <div align="center">
      <img width=600, height=500 src="/images/distributions/choose_distributions.png">
   </div>

## Part 2: Method of Moments

**Method of Moments (MOM)** estimates the parameters of a distribution (modelling a population) based on sample moments (such as the sample mean and sample variance).  The ***method of moments*** simply assigns the empirical moments to their theoretical counterparts, providing plug-in estimation of distribution parameters.  In this way we can use our sample (of many years worth of rain data) to determine a model for the population (of the true distribution of rain each month). To carry out this procedure we **must first specify** a distributional family which we will use to model the population in question. We should strive to pick a distriubtion that provides an appropriately useful approximation to the true data generating mechanism even thouugh it is unavoidable that any choice we make will not be perfect for real-world data for one reason or another.  A distribution that appears to not be "definitely wrong" -- i.e., minimizes apparent outliers or samples that look erroneous relative to the chosen distribution -- for the data on hand is probably a good choice. "All models are wrong, but some are useful" (and it's not that hard to make useful models!). 

<br>

1. Use **MOM** to compute the parameters for the two distributions selected
   in the previous question as follows:

   - Calculate the **sample mean** and **sample variance**

   - For the two distributions (you previously determined to be) under
     consideration as models for rainfall, locate the formulae relating the 
     distribution parameters to the population mean and population variance. 

   - The sample mean and the sample variance are obviously not the population 
     mean and the population variance, but we can use the sample versions as 
     estimators of the population versions; therefore, using the sample mean
     and variance in place of the population mean and variance, compute and 
     state the **MOM** estimate of the distribution parameters. (Some data
     models such as the normal distribution are parameterized by mean and 
     variance directly; whereas other data models such as the Gamma 
     distribution have different but related parameters.  The parameters 
     of the Gamma distribution are described below.)

   - Plot the two distributions with the computed parameters over the
     histogram of January rainfall data.
        - Use `normed=1` to plot a histogram on a probability density scale.
        - Use, e.g., `norm(mean, stdev).pdf(np.linspace(x, y))` to trace 
	  the distributions probability density function (pdf).

   **Hint:**
   - The pdf for the gamma function can be written as follows:
     ![](images/gamma_pdf.png)
     
   - The mean and variance of this parameterization are:
   
     ![](images/gamma_p1.png)
     
   - The population parameters can therefore be estimated using 
     **MOM** by substituting the sample mean and sample variance 
     into the formulas and solving for alpha and beta: 
     
     ![](images/gamma_p2.png)

   - NOTE: The Gamma distribution has two common parameterizations. The Gamma [Wikipedia](https://en.wikipedia.org/wiki/Gamma_distribution) page has a table comparing the two versions. They are *almost* equivalent, but instead of alpha and beta notation the other parameterization uses k and theta. The k and alpha parameter are equivalent between the two parameterizations and can be thought of as 'shape' parameters, and theta is the inverse of beta, so beta parameterizes the 'rate' of the distribution while theta is the more familiar 'scale' of the distribution. The `scipy.stats.gamma` uses the k and theta version, but refers to these as `a` and `scale`.

2. It is common (and effective) practice to decide how well a distribution fits data via graphing. Do you feel the gamma or the normal distribution is a better fit for the data on hand? 

   We can use the [Kolmogorov-Smirov](http://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Smirnov_test) 
   test to evaluate your fit using [scipy.stats.kstest](http://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.stats.kstest.html).
   If you have time after the other sprint problems come back and try this 
   out to see if it confirms your feelings regarding distrbution choice for the
   rainfall data. 

3. Now let's do what we have done for January for every month: loop over all 
   the months and create the same plot (of a histogram with an overlay of the 
   prefered data model) for each month. 
   
   **Here is some [code](plot_grid_stub.md) to help get you started with a 
   grid plot structure for the monthly plots.  Make sure you read through 
   and understand the code and thoroughly if you're not quite sure how to  
   get multiple subplots on one figure.**

   ![rainfall](images/rainfall.png)


## Part 3: Maximum likelihood Estimation

**Maximum likelihood Estimation (MLE)** chooses the parameter(s) that maximize
the likelihood of observing our given sample (under our choice of 
distributional family). The likelihood function is just the data model but it 
is instead evaluated with respect to the model parameters rather than the data 
itself (which is just plugged into the PDF/PMF equation and taken to be fixed 
as is).  The likelihood is a function that (often) looks like a PDF, but it 
does not actually capture any theoretical notion of a probability distribution 
for a random variable; instead, it simply shows the *relative "likelihood"* of 
different choices of the parameters given the data we observed.  The higher 
the likelihood of a specific parameter value the better the distribution fits
the observed data.  Hence, we assume that the parameter values that maximize 
the likelihood are good estimates of the underlying population distribution.
All that being said, however... you may recall that the likelihood term 
appears in Bayes' theorem... When the likelihood is combined with the prior 
(and normalized by the marginal likelihood) the result is a probability 
distribution for the parameter itself. That is, the parameter is treated as a 
random variable rather than a fixed constant and so it has a distribution 
which, given the data, can be calculated using Bayes' theorem.
Given this distribution, uncertainty and beliefs about the parameter can be
made using probability statements (which is the basis of Bayesian analysis).


MLE fitting is more computationally intensive than MOM (but often less 
computationally intensive than Bayesian model fitting). However, MLE is 
generally preferred over MOM because the resulting estimator(s) definitionally 
provide a better fit to the data. Further, parameter inference (i.e., 
uncertainty characterization) on MLE estimates from a frequentist perspective
has been substantially developed and is ubiquitously available.  These results
typically rely on sample sizes greater than 30 and distributional assymptotic 
results therein (such as, e.g., the central limit theorem). 


**Note:** MLE and MOM give you estimates of parameters under a given assumed 
distributional family. If the chosen distribution inherently does not fit the
data well, conducting MLE and MOM are a fools errand. 

<br>

1. Given data drawn from a Poisson distribution, where
  <img height='15px' src='images/MLE-lambda-equals-5.png'>, calculate
  the probability of observing a value of **6**. Use ``.pmf()``.

  Create a function `likelihood(x, lambda)` that takes in a lambda and
  a discrete data value, and returns the probability of observing the 
  discrete value given the particular lambda.

  **Note:** This function is returning probabilities of data given a parameter 
  choice. When we hold the data fixed and try different values for the 
  parameter, the comparision of the values is called the likelihood.  That is 
  why our function is called `likelihood()`.

2. Using numpy,
  [read](http://docs.scipy.org/doc/numpy/reference/routines.io.html)
  the file `data/poisson.txt` into a numpy array.

3. For a continuous range of lambda from 1 - 17 (use `np.linspace()`)
   calculate the likelihood (of the parameter) given the data. 

   You may assume the data are iid (independent and identically distributed)
   so that, e.g., (the probability of two data points can be calculated as) 
   `Pr(X=x,Y=y) = Pr(X=x)Pr(Y=y)` for a given choice of the parameter. 

   Note that as the number of data points increases, more and more 
   probabilities will need to be multiplied together.  While this is 
   theoretically apporpriate, it will eventually be numerical infeasible 
   (imprecise) A clever solution to this is to take the log (`np.log10()`) 
   of the product and instead sum log probabilities rather than multiply 
   probabilities.  
   
   Since the log function is monotonically increasing, the biggest possible
   sum of log probabilities will have a one to one correspondence with the 
   biggest possible product of probabilities. Specifically, it is the same 
   value of the parameter that will acheive maximums in both settings.  I.e., 
   if you have a concave function (e.g., a hill) if you log the heights of the 
   function (e.g., use log hill heights) the function (hill) will still 
   acheive it's maximum in the same place.   
   
4. Keep in mind that the objective is to compare the likelihoods for 
   different values of the parameters and choose the parameter that has 
   the highest likelihood... Plot the log likelihoods over the range of 
   considered lambda values and determine the lambda which produces the 
   maximum likelihood: this is our MLE point estimate.

   <div align="center">
     <img height="330" src="images/mle_poisson.png">
   </div>
   
   Is this what you expected?  See [9.0.1 Poisson Example](http://statweb.stanford.edu/~susan/courses/s200/lectures/lect11.pdf)

5. Returning to the rainfall data, find the maximum likelihood estimates of the
   alpha and beta for the gamma distributions.

   Scipy has a `fit()` method to find the MLE estimates for our data:

   ```python
    # Set the floc=0 for the method, this simply keeps the gamma
    # distribution's location fixed at 0

    ahat, loc, bhat = scipy.stats.gamma.fit(df[month], floc=0)
    alpha_mle, beta_mle = ahat, 1./bhat
    ```
   Using the MLE estimates for alpha and beta, plot the MLE estimated Gamma
   distribution on the same plot as the method of moments.

   Would you use the MOM or MLE estimates to parameterize your distributions
   for this data set?

   Explain your answer.

   <div align="center">
     <img src="images/mom_mle.png">
   </div>

   [**Save**](http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.savefig)
   **this figure** as ``mom_mle.png``**

## Part 4: Non-parametric Estimation

In some instances we may not be interested in parameterizing a particular
distribution of data, but rather just want a smoothed representation of the 
data at hand.

In such cases we can estimate the distribution *non-parametrically*
(i.e. making no assumptions about the form of the underlying distribution)
using kernel density estimation.

<br>

To see how this works let us create a bi-modal normal distribution that is
not easily described by any single (parametric) distributions.

1. Using scipy, create two normal distributions with different locations (mean)
   and scales (standard deviation):  
   ```Norm(0, 2)``` and ```Norm(4, 1)```.

2. Take 500 samples from each distribution and store them into a single array
   (or list). This has the effect of 'merging' our two distributions.

3. Plot a histogram of this merged distribution and try to identify if the 
   two peaks are where you would expect them to be.

   <div align="center">
     <img src="images/bimodal.png">
   </div>

4. Using the scipy
   [kde](http://docs.scipy.org/doc/scipy-0.13.0/reference/generated/scipy.stats.gaussian_kde.html)
   function [calculate a nonparametric representation of the data distribution](http://glowingpython.blogspot.com/2012/08/kernel-density-estimation-with-scipy.html). Overlay this non-parametric distribution on top of the histogram
   of the data.

   <div align="center">
     <img src="images/kde_bimodal.png">
   </div>

5. Consider some of the advantages and disadvantages of using a non-parametric 
   model as opposed to a parametric model. State at least one application where
   each is more appropriate than the other.

