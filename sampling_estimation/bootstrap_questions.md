**Include your code/answers in `afternoon_answers.py`**

**You can include word answers as string literals. e.g. `'''answer here'''`**

## Part 0: Central Limit Theorem

The Central Limit Theorem states that the mean of a sufficiently large number of samples of independent random variables, each drawn from a distribution with a well-defined expected value and well-defined variance, will be approximately normally distributed. Most distributions meet these criteria, so the CLT applies to them. Therefore the application of CLT is ubiquitous in statistics. Here we will explore the CLT via simulation.
 
<br>

1. As we explore the CLT we will be plotting means of samples drawn from several different distributions.  In order to do this efficiently we need to pass the Scipy distribution objects ([docs here](http://docs.scipy.org/doc/scipy-0.17.1/reference/stats.html)) and their parameters to this plotting function so that we can generate the data to be plotted. Here is a function that generates draws from a distribution by calling the `.rvs()` method.  Use this code to generate data from at least the **Poisson, Binomial, Exponential, Geometric, and Uniform** distributions. 

  ```python
  import scipy.stats as stats

  def make_draws(dist, params, size=200):
    """
    Draw samples of random variables from a specified distribution
    with given parameters and return these in an array.

    INPUT:
    dist: (Scipy.stats distribution object) Distribution with a .rvs method
    params: (dict) Parameters to define the distribution dist.
                  e.g. if dist = scipy.stats.binom then params could be:
                  {'n': 100, 'p': 0.25}
    size: (int) Number of samples to draw

    OUTPUT:
    (Numpy array) Sample of random variables 
    """
    return dist(**params).rvs(size) 

  # Generate draws from the Binomial Distribution, using Scipy's binom object.  
  binomial_samp = make_draws(stats.binom, {'n': 100, 'p':0.25}, size=200)

  ```

  NOTES:
    The `**params` notation unpacks the `params` dictionary and passes the items in the dict as keyword arguments to define the Scipy distribution instance.  If this is unfamiliar, check out the [docs](https://docs.python.org/2/tutorial/controlflow.html#unpacking-argument-lists) and/or [stackoverflow](http://stackoverflow.com/questions/1179223/in-python-when-passing-arguments-what-does-before-an-argument-do).

Beware that Scipy aims for consistency in what it calls the parameters that define a distribution and this results in some discrepancies between how we have presented these distributions in class and the definitions that Scipy uses.  We already saw this with Scipy's implementation of the Gamma distribution in the individual sprint.  Another example is with the [Uniform distribution](http://docs.scipy.org/doc/scipy-0.17.1/reference/generated/scipy.stats.uniform.html#scipy.stats.uniform): typically the distribution is parameterized by its minimum and maximum bounds, while Scipy uses `loc` and `scale` where min=`loc` and max=`loc+scale`. Take some time to scan through the documentation for each distribution you consider.

2. Now that you are comfortable drawing samples from various distributions, let's explore some behaviour of samples.  Implement a `plot_means` function that takes the same arguments as `make_draws()` but adds an extra argument `repeat` which determines the number of times to redraw a random sample of `size` from the distribution (and then plots the means). (Since `make_draws` instantiates a distribution object each time it is called and this involves quite a bit of overhead, for the sake of efficiency DO NOT just call `make_draws` from the `plot_means` function, but rather directly create the necessary functionality from scratch.) 

3. Call `plot_means()` with at least each of the following distributions: 
   - Binomial
   - Poisson 
   - Exponential
   - Geometric
   - Uniform

   Try out different parameter settings for each of the distributions, as well 
   as varying the `size` and `repeat` choices. Initial choices for the `size` 
   and `repeat` might be 200 and 5000. If the distribution of means resembles 
   a normal distribution regardless of the distribution you have specified, 
   you have demonstrated the CLT! (Failure to observe this phenomenon means 
   because of the assymetry and skew of the distribution of the data being 
   sampled more samples are needed before the CLT effect will begin to emerge!)

4. What do you observe if you change the sample `size` to 10 instead of 200,
   keeping `repeat` constant at 5000? Explain your observation on a high 
   level. Should the CLT apply when your sample `size` is small with a large 
   value for `repeat`?  Hint: this is best answered by distinguishing
   between the role of the of the `size` and the `repeat` variable.

5. Instead of taking the mean of the samples, take the maximum of each of the 
   samples and plot the histograms again. Do they resemble the normal 
   distribution? Do all sample statistics follow a normal distribution?  In
   your answer clarify the sample statistics to which the CLT applies. 

<br>

## Part 1: Population Inference and Confidence Interval


1. Suppose Google sampled 200 of its employees and measured how long they are gone for lunch. Load the data `data/lunch_hour.txt` into a Numpy ndarray and compute the mean lunch hour of the sample.

2. What is the sampling distribution of the sample mean? Justify your answer.

3. Compute the [standard error](http://en.wikipedia.org/wiki/Standard_error) of the sample mean. Based on the standard error and the sample mean, compute the [95% confidence interval](http://dsearls.org/courses/M120Concepts/ClassNotes/Statistics/530G_Derivation.htm) for the population mean lunch break length.

4. Interpret what the 95% confidence interval implies about the lunch hours of Google employees in general.

5. If the sample size were smaller, how would that affect the 95% CI? Explain your answer.  Suppose the sample size was 10, does your assumption from `2.` still hold? Explain your answer.

<br>

## Part 2: What is Bootstrapping and Why Do We Use It?

A common use of bootstrapping is to approximate the sampling distribution of a sample statistic (which is probably a point estimate of some unknown population parameter). Bootstrapping is especially useful when the sample size is small or when the underlying population distribution is unknown.

The general concept of bootstrapping is to create many samples from the one sample you actually have.  Hypothetically, randomly redrawing new samples out of the original sample itself (with replacement) can simulate random samples from the original population.  (Of course, this is only as good a representation of samples from the population as the original samples representation of the population).  The statistic of interest can be computed on each created (bootstrapped) sample in order to provide an empirical approximation of the sampling distribution of that statistic (under samples from the original population). The empirical quantiles of this bootstrapped sampling distribution can be interpreted as confidence intervals called bootstrapped confidence intervals. 

<br>

1. Why, given a smaller sample size, are we able to use bootstrapping to compute a more precise 95% CI as compared to the approach we used in `Part 1`?

2. Implement a `bootstrap` function to randomly draw with replacement from a given sample. The function should take a sample as a `numpy ndarray` and the number of resamples as an integer  (`default: 10000`). The function should return a list of `numpy ndarray` objects, each ndarray is one bootstrap sample. 
   
   **Hint:**
   - Use `np.random.randint` to randomly draw the row indexes with replacement
   - Create the bootstrap sample by indexing the original sample with the randomly drawn row indexes
 
<br>

## Part 3: Bootstrap to find Confidence Interval of Mean

Company X wants to find out if changing to Apple monitors increases its programmers' productivity. A random sample of 25 people is chosen and their monitors are switched. The difference between their productivity before and after the monitor switch is recorded in `data/productivity.txt`.
 
<br>

1. Load the data `data/productivity.txt`.

2. Why is it inappropriate to only report the mean difference in productivity as evidence to support the decision of changing all the monitors to Apple monitors in the company?

3. According to the Central Limit Theorem, the mean of N iid random variables converges to a normal distribution as N approaches infinity. Hence we can calculate the 95% confidence interval by `mean +/- 1.96 * standard error`.  Compute the 95% confidence interval using the sample's standard error.
   
4. Why might the confidence interval computed in the previous question be unreliable? 

5. Implement a `bootstrap_ci` function to calculate the confidence interval of any sample statistic (in this case the `mean`). The function should take a sample, a function that computes the sample statistic, the number of resamples (`default: 10000`), and the confidence interval (`default :95%`). The function should return the lower and upper bounds of the confidence interval and the bootstrap distribution of the test statistic.
   
   You should be able to call the function in the following manner:
   
   ```python
   bootstrap_ci(sample, stat_function=np.mean, iterations=1000, ci=95)
   ```

   **Hint:**
   - `bootstrap_ci` should call `bootstrap` to create the bootstrap samples.
   - Apply (map) the test statistic function to the bootstrap samples.
   - Use the **percentile interval method**: e.g., take the empirical 2.5% percentile and 97.5% percentile of the bootstrapped sampling distribution to create a 95% bootstrapped confidence interval.

6. Plot the bootstrap distribution of the means in a histogram. 

7. Based on the bootstrap confidence interval, what conclusions can you draw? What about if a 90% confidence interval were used instead? 
   
   Suppose there are 100 programmers in the company. The cost of changing a monitor is $500 and the increase of one unit of productivity is worth $2,000, would you recommend switching the monitors? State the assumptions you are making and show your work.
  
<br>

## Part 4: Bootstrap to find Confidence Interval of Correlation

You are interested if there is a positive correlation between the LSAT admission exam score and the first year GPA achieved in law schools. You are given the mean LSAT and mean GPA scores for the students from a sample of 15 law schools.

<br>

1. Load the data `data/law_sample.txt`

2. Use `scipy.stats.pearsonr` to compute the correlation.
   
3. Use the `bootstrap_ci` function to compute the confidence interval for the correlation coefficient.  Based on the bootstrap confidence interval, what conclusions can you draw?

4. Plot the bootstrap distribution of the correlation in a histogram. The percentile interval method (as implemented in `bootstrap_ci`) is less than optimal for building a confidence interval when the bootstrap distribution of the statistic is not nearly symmetrical. There are methods to correct for the confidence interval in such cases which will not be covered here. ([Studentized Bootstrap & Bias-Corrected Bootstrap](http://en.wikipedia.org/wiki/Bootstrapping_%28statistics%29#Methods_for_bootstrap_confidence_intervals)) 

5. Load in the LSAT and GPA data for all the law school (`data/law_all.txt`) and verify the population correlation is within the confidence interval.


