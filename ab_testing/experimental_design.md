# Experimental Design
## Conversion rate z-test:

## Part 1: Experimental Design

**A properly designed experiment is key to understanding causation**. For the
following questions, you will think through how you would design experiments
to establish if an activity **causes** the business to be more effective.

<br>

1. You are a data scientist at Etsy, a peer-to-peer e-commerce platform.
   The marketing department has been organizing meetups among its sellers
   in various markets, and your task is to determine if attending these
   meetups causes sellers to generate more revenue.

  - Your boss suggests comparing the past sales from meetup attendees to
    the sales from sellers who did not attend a meetup. Will this allow us
    to conclude attending meetups cause sellers to generate more revenue?
    If so, why, otherwise, why not?

  - What is the purpose of planning and running an experiment (A/B testing) as
    opposed to just collecting data we can observe passively?  

  - Outline the steps to design an experiment that would allow us to determine
    if local meetups _**cause**_ a seller to sell more?  

  ```
  Observational data is prone to [confounding](https://en.wikipedia.org/wiki/Confounding).
  While statistical techniques can be used in an attempt to control and account for potential
  sources of bias, there can rarely ever be complete certainty that every source of potential
  bias has been accounted for.  Thus, results from observational data are suggestive, but never
  definitive. For example, more competent sellers (with higher sales) might just be more
  inclined to go to local meetups so the higher sales cannot be attributed to going to
  local meetups alone.

  Designing an experiment which manipulates only a treatment effect while
  randomizing sample characteristics across the assigned treatment classes
  can be used to provide definitive statistical evidence of a cause and
  effect impact that is driven only by the treatment, and not some unobserved
  confounding variable.

  A good experimental design for an A/B test might proceed as follows.

  Our first task it to figure out the actions that Etsy can or would like to take.
  For example, let's say that upon discussion with stakeholders, we find that Etsy
  is actually interested in whether a program that creates and then invites Etsy
  sellers to meetups would be effective in driving sales. It is important to take
  note of whether these meetup actually exist already.

  In case the meetups already exist, and the only change Etsy will potentially make
  is increased invitation activity to sellers, we could proceed as follows. Randomly
  partition our sellers into two groups, a treatment and control. After partitioning,
  we should check that our random groups are equally spread across geography, age groups,
  gender, markets, etc, as we are attempting to control for the confounding of these
  effects by randomly assigning treatment and control. Then we would invite sellers in
  the treatment group to the created meetups, while for the control group we would proceed
  as if the program did not exist (simply not invite the control sellers). After collecting
  data, we will be able to compare the sales record of treatment and control groups, and any
  statistical difference will be attributable to the effect of the treatment.

  Note that it is a mistake to evaluate the results based on whether the treated sellers
  actually attend the meetups, as this is not what Etsy is interested in measuring the
  effect of, and they almost certainly cannot expect 100% attendance upon the implementation
  of this program.

  If the plan includes the creation of meetups, our job is more difficult. It is impossible
  to create a meetup, and then only expose knowledge of its existence to certain sellers, we
  must make use of some natural barrier to this information spread. A plan may be to randomly
  segment geographies, and then assign geographic regions to treatment and control. This would
  necessitate somehow controlling for differences between region, possibly with some matching
  scheme between regions with similar characteristics. Once this is carefully set up, we would
  proceed as usual.
  ```

  <br>

2. Suppose you have built a book recommendation system at Company X using the
   latest package from cutting edge methodology research development work
   (recommendation techniques are covered later), and you are interested in
   finding out if the book recommender makes good recommendations. The quality
   of the book recommendations is determined by the click through rate (CTR) of
   the recommendations.

   Design an **experiment** that allows us to conclude if latest recommender
   methodologies are justifiably worthy of being implemented and put into
   production. State what the control group should be and justify your choice.

   ```
   We could randomly split users into two groups, one of which is given recommendations
   from the new recommender system (treatment group) and another which receives either
   your current recommendation system (control group).  It is a good idea to compare
   against the current recommendation system, as opposed to, e.g., a random recommendation
   straw man since you can already achieve the level of results provided by the current
   recommendation system and this is the system you are potentially going to replace if you
   can demonstrate improved CTR.
   ```    
<br>

## Part 2: A/B Testing

Designers at Etsy have created a **new landing page** in
an attempt to improve sign-up rate for local meetups.

The historic sign-up rate for the **old landing page** is 10%. An
improvement to only 10.1% would provide a lift of 1%. A 1% lift would
translate to an absolute difference of 0.1% difference in conversion.
If statistically significant, the new landing page would be considered
a success. The product manager will not consider implementing the new
page if the lift is not greater than or equal to 1%.

Your task is to determine if the new landing page can provide a 1% or
more lift to the sign-up rate. You are also given the understanding
that there is a very different user base on weekends and a significant
amount of the revenue comes from those weekend users.

<br>
For problems 4, 5, 8 and 9 see pair_answers.py for the code and
run `python pair_answers.py` to see the results referenced in the solutions
<br>

1. Design an experiment in order to decide if the new page has a 1% lift
in sign-up rate as compared to the old page. Describe in detail the data
collection scheme you would use for the experiment. Justify why the data
will be collected that way.

     ```
     We have a historical expectation regarding the success rate of the old landing
     page; however, since there appear to be some caveats to page performance (e.g.,
     weekend rates differ from weekday rates) it is best to compare the two
     pages head to head, as opposed to just comparing the new page against the 10%
     historical rate.  Therefore, we could randomly divert 50% of incoming users to
     the new page, and direct the other 50% of users to the old page.

     We need to be careful to not count users multiple times, as this would results in
     samples that are not independent samples but are *dependent* because they are from
     the same user.

     Approximately equal sized groups will allow us to commit equal resources towards
     learning about the effectiveness of each of the two landing pages. And sending users
     to both webpages at the same time allows us to account for the dependency of signup rates
     on time and day of the week (as opposed to, e.g., testing one page on the weekend
     and the other page on weekdays, which would be totally confounded by day of the
     week!). We should keep in mind that it's possible for one landing page to be better on
     weekdays while the other is better on weekends since we have prior information that
     weekday and weekend populations are distinct. In light of this we might consider
     carrying out two tests -- one for weekdays and one for weekends -- and entertain the
     notion of different websites for the weekdays and weekends.  

     Landing page assignment and sign-up result can be recorded for eventual testing.
     We will plan to use a two sample z-test here since we will be comparing averages
     and the number of samples will be quite large. Addressing the question of 1% lift
     (which is different from testing if the landing page sign-up rates are not equal)
     is somewhat complicated (see parts 3 and 5 below).  
     ```

2. Why is it useful to report the change in conversion in terms of lift
instead of absolute difference in conversion?

    ```
    Conversion lift provides a measure of relative improvement, as opposed to
    absolute improvement.  When we do classical statistical hypothesis testing
    we are examining the statistical significance of *differences* between
    statistics in an absolute sense.  But not all differences of the same size
    have the same *impact* on a business.  Lift on the other had shows you
    improvement (or decline) relative to your current performance levels.
    ```

3. State your null hypothesis and alternative hypothesis? Is your alternative
hypothesis set up for a one-tailed or two-tailed test? Explain your choice.

   ```
   Initially, we might by default consider testing if the difference in
   sign-up rate proportions between the two landing pages is 0. I.e., the
   null hypothesis would be that there is no difference in landing page
   sign-up rates.  However, we can actually increase our power if we are
   willing to convert this two-tailed test into a one-tailed test. That is,
   we instead aim to rather test only that the *new* landing page is *better*
   than the old landing page, as opposed to that they are simply *different*.
   If we do this we can extend our rejection region in one tail of the testing
   distribution -- and ignore the other tail of the testing distribution -- while
   still maintaining the same overall Type I alpha significance level.  

   But we can go even further than this. We are actually potentially interested in
   testing if we can observe a 1% lift in sign-up rate.  Based on our historical (10%)
   trends, we tentatively expect this to equate to an increase of 0.001 in the observed
   conversion rates.  So, we might propose the following hypotheses to actually examine
   the statistical significance not of just (a) a difference and (b) an increase, but of
   the actual business question we have on hand. Thus, we might propose:  

   - H0: new_conversion_rate - old_conversion_rate = 0.001
   - H1: new_conversion_rate - old_conversion_rate > 0.001

   These null and alternative hypothesis are the result of considering the two independent
   binomial samples (sign ups for old site and sign ups with new site) and the resulting
   (independent and normally distributed) sampling distributions for the corresponding
   conversion proportions.  The null hypothesis here assumes there is exactly a 1% lift for
   the new landing page compared to the old landing page so that the expected value of the
   difference between random variables of the two (normal) sampling distributions is 0.001.
   (The distribution of the difference will also be a normal distribution since it is the
   sum of two normal random variables).  The alternative hypothesis states that there is a
   *greater* than 1% lift.

   Now, since the measured outcomes are binary, each user result can be modeled
   as a Bernoulli random variable and all together the entire collection of user
   results for a given landing page make up a single Binomial sample.  Notice, howerver,
   that by assuming a relationship between the proportions underlying the Bernoulli
   samples associated with each landing page (via the null hypothesis assumption)
   we are actually implicitly specifying that the mean of each individual Bernoulli
   sample is informative about the other and that the means (and hence the variances)
   *can be estimated jointly using all the data*.  I.e., if the conversion rate for the
   old landing page is $p_o$ and the conversion rate for the new landing page is $p_n$,
   then the variance of number of
   conversions out of $n_o$ visits for the old landing page can be estimated as
   $n_o((n_o(p_o+0.001) +  n_n p_n)/(n_o + n_n))(1-(n_o(p_o+0.001) +  n_n p_n)/(n_o + n_n))$
   and the variance of the number of conversions out of $n_n$ visits for the new landing
   page can be estimated as
   $((n_op_o +  n_n(p_n-0.001))/(n_o + n_n))(1-(n_op_o +  n_n(p_n-0.001))/(n_o + n_n))/n_o$.
   And correspondingly, the variance of the sampling proportions for each landing page are
   $((n_o(p_o+0.001) +  n_n p_n)/(n_o + n_n))(1-(n_o(p_o+0.001) +  n_n p_n)/(n_o + n_n))/n_n$
   and
   $n_n((n_op_o +  n_n(p_n-0.001))/(n_o + n_n))(1-(n_op_o +  n_n(p_n-0.001))/(n_o + n_n))$,
   respectively.

   Note that a simplified alternative to these standard error calculations is given in
   `z_test.py`. Actually, for equal sample sizes in the two groups
   (and unequal sample sizes as well -- but it's easier to see for equal sample sizes)
   the simplification is _conservative_.  That is, it will overestimate -- not underestimate --
   the standard error associated with the null hypothesis for testing purposes.
   Thus, the simplification provides an easier calculation for the null hypothesis test at
   the cost of slightly reduced power.

   An alternative hypothesis construction and test -- namely, assuming the proportions are
   equal and not just testing for a "statistically significant increase", but instead
   testing for a "statistically significant increase, after subtration of some offset" --
   provides an alternative interpretation and rational to the problem at hand and the
   simplified standard error calculations therein.  We return to this alternative
   specification again in part 5 below.
   ```

4. You ran a pilot experiment according to ``Question 1`` for ~1 day (Tuesday). The
collected data is in ``data/experiment.csv``. Import the data into a pandas dataframe.

**Check the data for duplicates and clean the data as appropriate.**

**Hint: Write a function to check (a) column consistency between
	ab and landing_page and (b) ensure duplicate row removal**


5. Calculate a p-value for a 1% lift from using the new page compared to the
   old page. Use `from z_test import z_test`

   What assumptions are going into this test? (Hint: examine z_test/z_test.py)

   Use the test to either justify or reject adoption of the new page.  Provide
   an interpretation of the p-value to help the management understand your finding.

   ```
   In `pair_answers.py`, we assume the two proportions are equal, which allows us to
   estimate the conversion rate using the data from both pages together.  This provides
   estimates for the variances of the sampling distributions (which could be different
   under our "equal proportions" null hypothesis if the group sizes are different).
   I.e., as seen in `pair_answers.py`,
   `z_score = (ctr_new - ctr_old - effect_size) / se` where
   `se = sqrt(conversion * (1 - conversion) * (1 / nobs_old + 1 / nobs_new))`
   with `conversion` being the overall conversion rate averaging across pages.

   This `z_score` will be normally distributed provided the number of users
   used to estimate the sign-in rate proportions is large enough (to allow the
   Central Limit Theorem to kick in so that the distribution of the proportion
   test statistics -- averages -- are normally distributed) so that the distribution of
   the difference then is just a difference of independent normally distributed
   random variables which is itself normally distributed.

   A one-tailed test is used to examine if there is evidence of an
   increase in sign up rate for the new versus the old landing page at the 0.05 alpha
   significance level. However, the tested increase is not just "new beats old" but instead
   (by virtue of the "effect size" handicap included in the z-score calculation) it is
   "new beats old by at least 0.001".  I.e., not only does the z-statistic have to pass
   the "greater than 0" threshold in a statistically significant manner; it has to further
   beat that threshold by an additional 0.001. A one-tailed test is appropriate here since
   we will only adopt the new page if the sign up rate is shown to be (statistically)
   significantly greater than the old page; otherwise, we will simply retain the old page.  

   It turns out that our data indicates we should keep the _old_ page. The p-value (~0.76)
   does not meet the 0.05 alpha significance level threshold, and so we cannot
   reject the null hypothesis that the sign-up rate does not exceed a 0.001.

   A p-value is *the probability of observing a test statistic as or more extreme
   than what we saw in the data given the null hypothesis.  At at the 0.05
   significance level, even if the null hypothesis is true, there is a 5% chance
   that we may mistakenly reject it in favor of the alternative.
   ```

5 (extra).

  ```
  Running a chi-square test would give you equivalent results to the z-test
  above.  To do the chi-sqare test, we construct a contingency table of
  observed and expected data

   Observed Data (`obs`)

   |               | Converted    | Not Converted|
   |---------------|--------------|--------------|
   | Old Landing   |  9409        | 81766        |
   | New Landing   |  9527        | 86047        |


   Expected Data (`exp`)

   |               | Converted    | Not Converted|
   |---------------|--------------|--------------|
   | Old Landing   |  9409        | 81766        |
   | New Landing   |  9527        | 86047        |


   and then run the chi-squared test, e.g.,
   ```
   ```
   import scipy.stats as stats
   stats.chisquare(obs, exp)[1] / 2
   ```


6. Assume your test was insignificant. Given the setting of the experiment and
   the context of the problem, why might you be hesitant to make the conclusion
   to not use the new landing page. What would you do instead?

   ```
   We know that (a) the user base may be different on the weekend, and (b) that
   sign-up rates are actually driven more by weekend users than weekday users. In
   A/B testing we often have to take into account other confounding factors that
   may be influential in driving response and carefully weigh these into our
   analyses informing adoption of a new landing page.  Here, the landing
   page may work for weekend users, but not 'Tuesday' users.
   ```


7. Why might it be a problem if you keep evaluating the p-value as more data comes
   in and stop when the p-value is significant?
   See [this](http://www.evanmiller.org/how-not-to-run-an-ab-test.html) article.

   ```
   If the test is at some point going to be significant it seems intuitive to keep
   doing tests and watching results so that you can stop as soon as it becomes significant
   (and the improvement is "proven").  However, each time you do a classical statistical
   hypothesis test you have some chance of wrongly rejecting the null hypothesis.  By
   virtue of random chance a page may get a lucky string and CTR may appear to be improved,
   but given more  views it may "regress back to the mean" and return to unremarkable
   performance levels.  This is a form of the so-called "multiple testing problem". To avoid
   this pitfall, a predefined number of views must be fixed (to achieve a desired
   test power) and then the experiment must be carried out to completion, at which
   point the hypothesis test is run (at the desired alpha significance threshold)
   and the results evaluated.
   ```

8. Some perennial problems encountered in A/B testing are repeated testing (or "p-value
   monitoring") and inefficient or inappropriate stopping rules. Repeated testing renders
   statistical significance testing unreliable and must be approached and used with great
   care, if at all. On the other hand, premature test evaluation before enough data has
   been collected to appropriately power the experiment, or overly long running experiments
   that fail to take advantage of real and available improvements in favor of continued
   data collection, are simply inefficient (if not ineffective or even inappropriate).  
   What is instead desired is the proper balance of exploration and exploitation.

   Given these challenges, it is very important that an A/B testing experiment is designed
   with the appropriate statistical power. I.e., a proper "power analysis" must be done to
   determine the scope and parameters of the hypothesis test. One semi-quantitative way to
   access if your conclusion is going to change if you have had run the experiment longer is
   to (AFTER THE HYPOTHESIS TEST HAS BEEN CONDUCTED) plot the progression of the p-value as
   a time series. If the p-value is consistent upon the collection of more data over an
   extended period of time, then you are more confident that your conclusion would stay the
   same even if the experiment had run on longer.  

   See the Airbnb talk on this technique [here](http://nerds.airbnb.com/experiments-airbnb/)
   and [here](http://nerds.airbnb.com/experiments-at-airbnb/).

   Let's try this approach out.  Plot the time series of the p-values using hourly intervals
   such that the p-value at the second hour would be evaluating data up to second hour and
   the p-value at the third hour would be evaluated using the data up to the third hour,
   and so on. Describe the insights you gain from the plot.

   ```
   The p-values appear to be fairly volatile -- certainly they are not consistently
   trending below the 0.05 cutoff.  This is quite suggestive that indeed this hypothesis
   test *would not* be expected to provide significance (and reject the null hypothesis)
   if we allowed it to run longer.  It is interesting to notice that at hour 15 the test
   was marginally significant, but then the p-values increased returned to distinctly
   non-significant levels.  This is exactly the type of problem that can happen if you
   monitor p-values and then stop running your test when you see something "interesting".
   ```

9. There is an additional file `data/country.csv` providing country information about
   the clicks.  Does the country data offer any additional insights?

   ```
   Using a Bonferonni adjusted family-wise alpha significance level of 0.05 we will
   reject the tests if there original p-value is less than 0.05/4 = 0.0125 since
   we are doing four tests (one for each countries). Only the UK has a p-value
   less than this threshold (0.0026). The p-value for Canada (0.0176) is suggestive
   but it is *not* significant at the family-wise significant level, so we do not
   reject the null hypothesis (at this significance level) that there is a difference
   between landing page click through rates in Canada.  

   Family-wise error control is a nice idea, but it is *very* conservative.  In fact,
   it is so conservative that when the number of tests gets quite large, it will be
   virtually impossible to reject the null hypothesis for *ANY* of the tests.  An
   alternative approach to the multiple testing situation is to control the False
   Discovery Rate (FDR) as opposed to the family-wise error.  In this approach one
   provides a set of tests results and reports the rate of false positive among
   those results.  If you are in a situation with a large number of tests, FDR is
   something that should be seriously considered to account for multiple testing
   in a way that still allows you to identify the most unusual results in your tests.
   ```
