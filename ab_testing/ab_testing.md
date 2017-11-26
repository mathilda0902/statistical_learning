# A/B Testing
## CTR t-test:

- Click Through Rate: Clicks / Impressions
- Independent t test:
  - scs.ttest_ind(df_ctr_signed_in['CTR'].dropna(), df_ctr_not_signed['CTR'].dropna(), equal_var = False)
- p value:
  - p_val = stats.ttest_ind(group_1_df['CTR'], group_2_df['CTR'], equal_var=False)[1]
- By gender, by signed in vs. not signed in, and by age groups.

## Part 1: Hypothesis Testing Recap

1. State which test should be used for the following scenarios to calculate p-values.
Explain your choice.

   a. You want to know if dogs and cats have the same weight.
      To examine this issue you collect data by randomly selecting
      50 dogs and 80 cats from a large animal shelter.

   b. You want to determine if the same proportion of San Franciscans and
      Oaklanders like the SF Giants.  To examine this issue you select a
      random sample of San Franciscans and Oaklanders and ask them about
      their favorite baseball team.

      ```
      a.
      A two sample t-test would be appropriate here _provided_ we can assume the weights of
      dogs (and cats) is normally distributed as this is an assumption of the t-test; however...
      with sample sizes above 30 the t-test will give nearly identical results to a z-test.
      So the particulars of a z-test or a t-test here are going to be inconsequential.
      The z-test is appropriate because the sample sizes appear to be large enough for the
      central limit to apply to the averages of dog (and cat) weights.

      The above comments notwithstanding, an assumption on the variances of weights of dogs
      and cats must be considered. I.e., we can either enforce an assumption of equal variance,
      or not. Our choice will influence the specific details of the calculations we carry out
      for the degrees of freedom and the variance estimates associated with the test.  The test
      that *does not assume the variances are equal* is called *Welsh's test*.

      Note that it does not make sense to do any sort of paired testing here
      as there is no natural pairing or dependency structure between samples.

      b.
      Liking the giants or not is a binary (Bernoulli) random variable.
      The population of San Franciscans and Oaklanders is made of populations of
      Bernoulli random variables. Samples of size *n* and *m* from each of these
      populations are single samples from Binomial distributions
      each of which is of course the sum of some Bernoulli random variables
      (specifically *n* and *m* Bernoulli random variables, respectively).
      If we divide the number of people in each sample that like the Giants
      by the number of people in each sample, respectively, then we have a couple
      averages.  I.e., two x-bars.

      An average (x-bar) is *approximately* normally distributed (by the central limit
      theorem) regardless of the distribution of the original variables making up the
      sample if the sample size used to make the average is 'large enough'
      where 'large enough' depends on what the distribution of the samples actually are.
      (Also, these samples need to be 'independent and identically distributed,
      as in our case here they are).

      Note that if we *don't know* the variance of the population from which the sample
      that was used to calculate x-bar was drawn, then we will have to estimate it, and
      we therefore technically cannot use a z-test for hypothesis testing.  If the data is
      normally distributed then, mathematically speaking, we can use a t-test.  A t-test is
      fairly robust and can still be applied usefully to data that is *approximately*
      normally distributed.   

      The difference (i.e. a type of "sum") of two (or more) *independently* normally
      distributed random variables is also normally distributed.  Here, we can use a two sample
      t-test for our hypothesis testing.  For proportions, there are some convenient formulas
      for calculating the variance.  These formulas are true because (a) the samples are
      Binomial samples and (b) because the null hypothesis will likely assume that the
      proportions are equal which allows us to leverage all the data to calculate what that
      estimated proportion is using *all* of the data.
      ```


2. A study attempted to measure the influence of patients' astrological signs
   on their risk for heart failure.
   12 groups of patients (1 group for each astrological sign) were reviewed
   and the incidence of heart failure in each group was recorded.

   For each of the 12 groups, the researchers performed a z-test comparing
   the incidence of heart failure in one group to the incidence among the patients
   in each of the other groups (one-versus-one as opposed to one-versus-rest,
   which might be another approach to consider...).

   The group with the highest rate of heart failure was Pisces,
   which had a p-value of .026 when assessing the null hypothesis that it had
   the same heart failure rate as the group with the lowest heart failure rate, Leo.

   What is the the problem with concluding from this p-value that
   Pisces have a higher rate of heart failure than Leos at a significance level of 0.05?
   How might you adjust your interpretation of this p-value?

   ```
   The problem here is that we are _actually_ performing multiple tests at the 0.05 level
   by cherry picking _the best_ test to examine *even if we don't even "count" the other tests!*
   Even if birth sign has *no* effect on heart failure outcomes, by definition there's a 5
   percent probability we'll say it does *each time we test it*.

   There are 12 choose 2 pairwise combinations here.  That's 66 comparisons.
   If birth sign has no relationship with heart failure (i.e., the null hypothesis is
   true) each of these 66 comparisons has a 5 percent chance of randomly getting a
   p-value *less than 0.05* (again, by definition of the 0.05 significance level test).
   It is therefore not surprising that we would see a p-value in this family of tests
   that is less than 0.05. I.e., 'the best' one is less than 0.05.

   There are two common ways to deal with this (and many more options as well); namely,
   Bonferonni or False Discovery Rate (FDR) correction. The former simply divides the
   alpha significance level that we are testing at by the number of tests we are doing
   and makes the resulting value the level at which p-values must be tested against for
   significance (i.e., reject or fail to reject the null hypothesis at the stated significance
   level). What this does is makes the expected number of tests _accross the whole family of
   tests_ that will be wrong by chance be 0.05. So in this example, to control the *family
   wise* error at the 0.05 level _using the Bonferonni method_, we should compare the p-values
   to 0.05/66.  If all p-values are not less than this testing threshold, then the entire
   family of tests is _not significant_ at the family wise error level of 0.05. Alternatively,
   if some tests _are significant_ then they are significant at _the family wise error level_
   of 0.05.

   The Bonferroni family wise error procedure (and all family wise error rate procedures for
   that matter) are _very_ conservative.  They account for multiple testing by increasing the
   stringency of the testing threshold to reflect the extent of multiple testing being
   employed.  This means that each tests significance is dependent upon the number of tests
   being performed and as the number of tests increases the effect size has to be
   correspondingly larger to be able to be detected in the face of the increased stringency.

   An alternative to the Bonferroni correction is the False Discovery Rate (FDR).
   For a subset of tests, this reports the rate of false discoveries expected in the subset.
   So an FDR rate of 0.05 means that 5% of your tests are expected to be false positives.
   This is much less conservative than the Bonferroni correction because it does not
   *FORCE* every test to meet some multiple testing threshold.  Instead, it just reports
   the likely proportion of tests that are false positives as a result multiple testing.
   This is of course dependent upon the overall proportion of false hypothesis being
   tested (since those are the hypotheses that could potentially produce false positives).
   FDR is carried out by converting the p-value to a so called q-value, which indicates
   the FDR cutoff.  So if you took all tests with q-values less than 0.05, then 5% of those
   tests would be expected to be false positives. There is a python package (of course)
   called StatsModels which can convert p-values to q-values.  
   ```

3. Explain the difference between _statistical_ and _practical_ significance.

   ```
   Statistical significance is simply a decision as to weather or not an observed test
   statistic have happened at a certain level of chance (which the *you* choose as the
   *significance level* in setting up the test).  Formally, it can be executed by
   observing if the test statistic falls in a (null hypothesis's) rejection region or
   if the p-value is less than your alpha significance threshold level.  Remember that
   a p-value is "the probability of observing a test statistic as or more extreme than
   what you observed if the null hypothesis is true".  A little consideration will
   clarify that a p-value less than a significance level corresponds exactly to a test
   statistic which falls into the rejection region.  

   A statistical hypothesis test does not take into account weather or not the observed
   difference is big enough to have any practical real-world, business, or scientific
   ramifications.  E.g., if we show statistically that website version B is just marginally
   better than website version B, but the costs associated with maintaining website B are not
   warranted by the gains, then obviously statistical significance is not justification
   for pursing bad business practices.
   ```

## Part 2: Analyzing Click Through Rate


We will use hypothesis testing to analyze **Click Through Rate (CTR)**
on the New York Times website. CTR is defined as the number of clicks a
user makes per impression that is made upon the user. We are going to
determine if there is statistically significant difference between the
mean CTR for the following groups:
```
1. Signed in users v.s. Not signed in users
2. Male v.s. Female
3. Each of 7 age groups against each other (7 choose 2 = 21 tests)
```

1. Calculate the Bonferroni adjusted threshold need to control the
   family-wise error of all these tests at the 0.05 significance level.  

   ```
   All together there are 1+1+21=23 tests.
   So get a 0.05 family wise error rate we would compare the tested p-values to 0.05/23.
   ```


2. Load `data/nyt1.csv` in a pandas dataframe.

   Use `data.info()` to make sure the data types are valid and there are no null values.
   This data has been cleaned for you, but generally it is very prudent to make sure
   nothing unexpected is hiding in the data before you plunge forward with analysis.


3. Make a new column `CTR` using the `Impressions` and the `Clicks` columns.
   Remember to remove the rows with `0` impressions.


4. Plot the distribution of each column in the dataframe. Do that using `data.hist()`.
   Check out the arguments you can use with the function
   [here](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.hist.html).
   Set the `figsize=(12,5)` to make sure the graph is readable.


4. Make two dataframes - one a dataframe of 'users who are signed in'
   and a second of 'users who are not signed in'.
   Plot the distributions of the columns in each of the dataframes.
   By visually inspecting the two sets of distributions,
   describe the differences between users who are signed in and not signed in?

   ```
   While there are fewer individuals _not_ logged in,
   they actually appear to perhaps have a higher CTR
   compared to those who _are_ logged in.
   ```


5. Someone proposes using Welch's t-test to determine if the mean CTR between
   signed-in users and non-signed-in users is statistically different, i.e.,

   ```python
   scipy.stats.ttest_ind(a, b, equal_var = False)
   ```

   where equal_var = False because the Welch's t-test _does not_ assume the two populations
   in which the samples are drawn from have different variances.

   (a) Interpret the results of this test as they relate to a null hypothesis; however,
   additionally critique (b) the appropriateness (or lack thereof) of the t-test for CTR data,
   (c) the necessity of a t-test specifically (as opposed to a z-test) in this
   situation given the number of samples we have, and (d) the correctness (or lack thereof)
   of the assumption of equal variance.

   Overlay histograms (using the alpha transparency parameter) of the two samples and
   annotate their means with vertical lines in the overlayed histograms.

   ```
   (a) The result Ttest_indResult(statistic=-55.376117934260868, pvalue=0.0)
   tests the null hypothesis that the mean CTRs of the populations (signed in
   users versus not signed in users) are equal. Here, it shows that indeed CTR is
   _lower_ (since the sign is negative) for users who are logged on as opposed to
   not logged on and shows that this is extremely unlikely to have been observed
   by chance if the null hypothesis was true (because the p-value is so small that
   it registers as 0...); however...

   (b) The t-test was created to deal with small samples and is *only*
   strictly appropriate for data that is normally distributed (here the data is
   a binomial sample converted to a rate... so not normally distributed); however...

   (c) The t-test will have results that are *extremely* similar
   to z-tests when the df freedom (i.e., sample sizes) are no longer "small"
   (like, e.g., for samples sizes greater than n=30), and the z-test
   is likely appropriate here as a result of the Central Limit Theorem, so
   in the end this is a reasonable way to implement testing in this situation.

   (d) The null hypothesis is about the mean CTRs for signed in and not signed in
   users.  This is different than testing parameters of a bernoulli distribution
   (or a binomial distribution).  It's about the difference of means of populations
   of *x_i/n_i* where *x_i* is an *n_i* trial binomial random variable and *n_i* is
   potentially changing from sample to sample based on the number of impressions individual
   *i* sees. If we were testing binomial/bernoulli parameters then the null hypothesis
   would likely that the proportions are _the same_, in which case the binary
   outcome implies that the variances of the two samples will be equal.  This
   equality assumption which follows from the null hypothesis can thus be leveraged
   to make the test more powerful (i.e., more likely to select significant
   differences if they are there) *in the binary variable context*.  

   With CTR it is perhaps less clear weather or not we should assume that the variances
   of the two populations are equal; however, if we assume that the chance that
   someone will click is the same for any impression in both populations, and that
   the distribution of impressions is the same in both populations, then indeed the
   variances will be equal.  Allowing that the variances may not be equal means we are
   testing if the populations average CTR is the same, but there is more spread (variance)
   in one population over the other population around that mean. So the latter is a
   less strict hypothesis, while the former is a more powerful test (if the
   assumption of equal variances is made *and is reasonable/correct*).
   ```

6. Determine if the mean CTR between male users and female users is
   statistically different. Is the difference in mean CTR between signed-in users
   and non-signed-in users more worthy of further investigation than that between
   male and female? Explain your answer. `Male: 1, Female: 0`

   ```
   The difference in CTR between signed in and non-signed users is more
   worthy of further investigation since the difference in CTR is greater.
   The female/male CTR difference is only marginally significant (0.0010 < 0.00217).
   The signed-in/non_signed CTR difference is more significant than that of
   male/female (0.0 < 0.00217).
   ```

7. Calculate a new column called AgeGroup, which bins Age into the following buckets
   `'(18, 24]', '(24, 34]', '(34, 44]', '(44, 54]', '(54, 64]', '(64, 1000]', '(7, 18]'`

   Use only the rows where the users are signed in. The non-signed in users
   all have age 0, which indicates the data is not available.

   Use pandas' function
    [cut](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.cut.html)
   `pandas.cut(signin_data['Age'], [7, 18, 24, 34, 44, 54, 64, 1000])`

   Now make a barplot showing how many users fall into each age category.


8. Determine the pairs of age groups where the difference in mean CTR is
   statistically significant at a Bonferroni corrected family wise error
   rate of 0.05. Collect the p-values and the difference of the means for
   each pair and store these results in a `DataFrame`.

   Sort the data frame by (a) the absolute differences in mean CTR and then by
   (b) the p-values *for the pairs that are statistically significant*.  Comment
   on the trend you observe for groups `(64, 1000]`, `(54, 64]` and `(7, 18]`
   relative to the other groups and each other.  Provide some explanation as to
   the trends you observe.    

   Sort the data frame by (a) the absolute differences in mean CTR and then by
   (b) the p-values *for the pairs that _are statistically insignificant_*.  
   State the 3 groups that are the least different in mean CTR and provide an
   explanation for why you think this is true.


   ```
   Each '(64, 1000]', '(7, 18]' and '(54, 64]' age group
   has CTR significantly greater than than the other 4 age groups
   (i.e., '(18, 24]', '(24, 34]', '(34, 44]', and '(44, 54]').
   So the oldest 2 groups and the youngest group are the mostly
   likely to click through.

   While '(64, 1000]' has a higher CTR than '(7, 18]' which in turn has a
   higher CTR than '(54, 64]' and the differences are all statistically
   significant, they are less significant than that between the others.

   Perhaps the oldest groups are more prone to click through to read an
   article,  but do less browsing overall. On the other hand, the youngest
   group may be less directed but more active and superficial in their clicking.

   The differences in CTR between '(18, 24]', '(24, 34]', '(34, 44]' and '(44, 54]'
   are not significant. This indicates the users aged from 18 - 54 are clicking
   through at similar ratios. This may be due to similar browsing behavior. I.e.,
   they may be at work and scanning through a lot of articles in between tasks,
   but not clicking through into the articles to actually read them.
   ```
