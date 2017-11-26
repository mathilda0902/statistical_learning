# Power Analysis

## Functions:
### One sample:
- CI for mean
- Power for alternative hyppothesis
- Power vs effect effect size

### Two sample pooled:
- Null and alternative hypotheses distributions
- Minimum sample size

##  Part 0:
7. Under the null specified in part 2, using a 5% type I error rate, and considering the true mean being equal to the one found in our sample; compute the power of the test. Explain what power means in the context of this problem.

   ```
   calc_power(coke_weights, 20.4)
   # 0.29549570806327596
   ```
   ```
   Power, in this context, is the probability of detecting the mean weight of a
   bottle of coke is different from 20.4 given the that the weight of a bottle of
   coke is indeed different from 20.4.  In this case, we have a 29.5% probability
   of choosing the alternative correctly if the true mean value is larger by 0.12 ounces.
   ```
