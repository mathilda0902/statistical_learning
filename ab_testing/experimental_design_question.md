Include your answers in `pair_answers.md`.

## Part 1: Experimental Design

**A properly designed experiment is key to understanding causation**. For the following questions, you
will think through how you would design experiments to establish if an activity **causes** the business
to be more effective.

<br>

1. You are a data scientist at Etsy, a peer-to-peer e-commerce platform. The marketing department has been
   organizing meetups among its sellers in various markets, and your task is to determine if attending these
   meetups causes sellers to generate more revenue.

  - Your boss suggests comparing the past sales from meetup attendees to the sales from sellers who did not attend 
    a meetup. Will this allow us to conclude attending meetups cause sellers to generate more revenue?
    If so, why, otherwise, why not?

  - Outline the steps to design an experiment that would allow us to determine if local meetups _**cause**_ a
    seller to sell more?  

  - What is the purpose of running an experiment (A/B testing) as opposed to applying various statistical techniques
    to your existing data?

  <br>

2. Suppose you have built a book recommendation system at Company X using the latest discoveries in research 
   (recommendation techniques are covered later). You are interested in finding out if the book recommender makes 
   good recommendations. The quality of the book recommendations is determined by the click through rate (CTR) of     the recommendations.
    
   Design an **experiment** that allows us to conclude if latest discoveries in recommenders are worthy of 
   being implemented and put into production. State what the control group should be and justify your choice.
    
<br>

## Part 2: A / B Testing

Include your answers in `pair_answers.py`.

Designers at Etsy have created a **new landing page** in an attempt to
improve sign-up rate for local meetups.

The historic sign-up rate for the **old landing page** is 10%.
An improvement in the sign-up rate to only 10.1%, while just a .1% absolute improvement, actually means a
lift in conversion of 1% (because the percent change from 10% to 10.1% is 1%).
If statistically significant, the new landing page would be considered a success.
The product manager will not consider implementing the new page if
the lift is not greater than or equal to 1%.

Your task is to determine if the new landing page can provide a 1% or more
lift to the sign-up rate. You are also given the understanding that there is a very different
user base on weekends and a significant amount of the revenue comes from those weekend users.

<br>

1. Design an experiment in order to decide if the new page has a 1% lift in sign-up rate as compared to the old page? 
   Describe in detail the data collection scheme you would use for the experiment. Justify why the data will be       collected that way.
   
2. Why is it useful to report the change in conversion in terms of lift instead of absolute difference in             conversion?

2. State your null hypothesis and alternative hypothesis? Is your alternative hypothesis set up for a one-tailed
   or two-tailed test? Explain your choice.

3. You ran a pilot experiment according to ``Question 1`` for ~1 day (Tuesday). The
   collected data is in ``data/experiment.csv``. Import the data into a pandas
   dataframe. Check the data for duplicates and clean the data as appropriate. 

4. Calculate a p-value for a 1% lift from using the new page compare to the
   old page. 
   - Use `from z_test import z_test`

   What assumptions are you making when you use a z-test.

   Interpret the p-value and explain your decision whether to adopt the new page or not.

6. Assume your test was insignificant. Given the setting of the experiment and the context of the problem, why might you be hesitant to make the conclusion to not use the new landing page. What would you do instead? 
   
7. Why might it be a problem if you keep evaluating the p-value as more data comes in and stop when the p-value is significant?
   See [this](http://www.evanmiller.org/how-not-to-run-an-ab-test.html) article.

8. One perennial problem for A/B testing is when to stop your test. We will cover a more in-depth treatment of calculating statistical power of your experiment. One semi-quantitative way to assess if your conclusion is going to change if you have had run the experiment longer is to plot the progression of the p-value as a time series. If the p-value is consistent upon the collection of more data over an extended period of time, then you are more confident that your conclusion would stay the same even if the experiment had run on longer.  
   - See Airbnb's talk on this technique [here](http://nerds.airbnb.com/experiments-airbnb/)
   and [here](http://nerds.airbnb.com/experiments-at-airbnb/).
   
   Plot the time series of the p-values using hourly intervals, such that the p-value at the second hour would be evaluating data up to second hour and at the third hour would be evaluating the data up to the third hour. Describe the insights you gain from the plot.

9. There is an additional file `data/country.csv` providing country information about the clicks.
   Does the country data offer any additional insights?
