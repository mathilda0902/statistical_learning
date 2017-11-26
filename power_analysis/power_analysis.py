'''Part 0: question 1 - 6'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as scs
import matplotlib.mlab as mlab
from math import sqrt

coke = np.loadtxt('data/coke_weights.txt')

'''
H0: weight of a bottle of coke = 20.4 oz
H1: weight of a bottle of coke != 20.4 oz
'''
n = len(coke)
mu_1 = np.mean(coke)
'''Out[7]: 20.519861441022606
'''
var = np.var(coke, ddof=1)
'''Out[8]: 0.9242649555542054
'''
mu_0 = 20.4

sigma = sqrt(var)/(sqrt(n))
x = np.linspace(mu_0 - 4*sigma, mu_0 + 4*sigma, 1000)
plt.plot(x,mlab.normpdf(x, mu_1, sigma), label = 'sample')
plt.axvline(mu_0)
plt.plot(x,mlab.normpdf(x, mu_0, sigma), label = 'null')
plt.axvline(mu_1)
plt.axvline(20.2347373706)
plt.axvline(20.5652626294)
plt.legend()
#plt.savefig('sample_mean_testing')
plt.show()

'''calculate confidence interval'''
def sample_sd(arr):
   return np.sqrt(np.sum((arr - np.mean(arr)) ** 2) / (len(arr) - 1))

def standard_error(arr):
   return sample_sd(arr) / np.sqrt(len(arr))

m = coke_weights.mean()
se = standard_error(coke_weights)
print 'The sample mean is', m
print 'The standard error is', se


'''Part 0 Question 7:
Under the null specified in part 2, using a 5% type I error rate, and considering
the true mean being equal to the one found in our sample; compute the power of
the test. Explain what power means in the context of this problem.'''
'''solution:'''
def calc_power(data, null_mean, ci=0.95):
   m = data.mean()
   se = standard_error(data)
   z1 = scs.norm(null_mean, se).ppf(ci + (1 - ci) / 2)
   z2 = scs.norm(null_mean, se).ppf((1 - ci) / 2)
   return 1 - scs.norm(data.mean(), se).cdf(z1) + scs.norm(data.mean(), se).cdf(z2)

'''Power, in this context, is the probability of detecting the mean weight of a
bottle of coke is different from 20.4 given the that the weight of a bottle of
coke is indeed different from 20.4.  In this case, we have a 29.5% probability
of choosing the alternative correctly if the true mean value is larger by 0.12 ounces.'''




'''Part I'''
'''question 1'''
'''solution'''
def explore_power(data, null_mean, ci=0.95):

   # Calculate the mean, se and me-(4 std)
   data_mean = np.mean(data)
   data_se = np.std(data, ddof=1) / np.sqrt(len(data))

   # Create a normal distribution based on mean and se
   null_norm = scs.norm(null_mean, data_se)
   data_norm = scs.norm(data_mean, data_se)

   # Calculate the rejection values (X*)
   reject_low = null_norm.ppf((1 - ci) / 2)
   reject_high = null_norm.ppf(ci + (1 - ci) / 2)

   # Calculate power
   power_lower = data_norm.cdf(reject_low)
   power_higher = 1 - data_norm.cdf(reject_high)
   power = (power_lower + power_higher) * 100
   return power

'''solution result'''
'''Power increased. Changing the null hypothesis here to 20.2 ounces increases the
effect size, which increases our ability to detect a shift and thus our power.
Specifically, we now have a power of 96.6%.'''


'''my attempt'''
def explore_power(mu_0, mu_1, data, alpha):
    z_beta = (mu_0 - mu_1)/float(scs.sem(data)) - scs.norm.ppf(alpha)
    power = 1 - scs.norm.cdf(z_beta)
    return power
'''my result'''
explore_power(20.4, np.mean(coke), 0.05)
'''returns 0.41164'''
explore_power(20.2, np.mean(coke), 0.05)
'''returns: 0.98417. Much increase from explore_power(20.4)'''

'''question 2'''
def power_plot(effect_size, data, alpha):
    z_beta = - (effect_size / float(scs.sem(data)) + scs.norm.ppf(alpha))
    power = 1 - scs.norm.cdf(z_beta)
    plt.plot(effect_size, power)
    plt.xlabel('effect size')
    plt.ylabel('power')
    plt.ylim(0,1.02)
    plt.show()

power_plot(np.linspace(0, 1.2, 20), coke, 0.05)

'''question 3'''
large_coke = np.loadtxt('data/coke_weights_1000.txt')
explore_power(20.360001135793635,np.mean(large_coke), large_coke, 0.05)
'''power is 0.986437 compared to power of 0.41164 for the same effect size (mean of coke - 20.4)'''

'''question 4'''
'''using the same coke data with 130 bottles.'''
def power_plot2(effect_size, data, alpha):
    z_beta = - (effect_size / float(scs.sem(data)) + scs.norm.ppf(alpha))
    power = 1 - scs.norm.cdf(z_beta)
    plt.plot(alpha, power)
    plt.xlabel('significance level')
    plt.ylabel('power')
    plt.ylim(0,1.02)
    plt.savefig("power_alpha")
    plt.show()

power_plot2(np.mean(coke)-20.4, coke, np.linspace(0.01, 0.3, 20))


'''Part II'''
'''preparing data'''
data = pd.read_csv('data/experiment.csv')
old_data = data[data['landing_page'] == 'old_page']['converted']
new_data = data[data['landing_page'] == 'new_page']['converted']

'''hypotheses
Set X as a random variable which is the (new conversion - old conversion)
X ~ p_new - p_old

H0: \pi_new - \pi_old = 0.001
H1: \pi_new - \pi_old > 0.001'''

p_old = old_data.mean()
p_new = new_data.mean()
n_1 = len(old_data)
n_2 = len(new_data)
p = (p_old + p_new)/(2.0)
s_pool = sqrt(p*(1-p)/n_1 + p*(1-p)/n_2)

'''Part 2.1'''
'''Question 1'''
#Data
data = pd.read_csv('data/experiment.csv')
old_data = data[data['landing_page'] == 'old_page']['converted']
new_data = data[data['landing_page'] == 'new_page']['converted']

#Definitions
n_old = old_data.count()
n_new = new_data.count()
p_old = old_data.sum()/float(n_old)
p_new = new_data.sum()/float(n_new)
p = (n_old*p_old + p_new*n_new)/(n_old+n_new)
se = np.sqrt(p*(1-p)/n_old + p*(1-p)/n_new)
center = p_new-p_old
sample = scs.norm(loc=center,scale=se)

#Plot
fig, ax = plt.subplots(1,1)
xvals2 = np.linspace(center-4*se,center+4*se,200)
ax.plot(xvals2,sample.pdf(xvals2),color='r', label='sample')
plt.legend()
plt.show()

'''Question 2'''
#Null Distribution
null = scs.norm(loc=0.001,scale=se)

#Define axes
xvals = np.linspace(.001-4*se,.001+4*se,200)
xvals2 = np.linspace(center-4*se,center+4*se,200)

#Plot
fig, ax = plt.subplots(1,1)
ax.plot(xvals,null.pdf(xvals),color='b', label='null')
ax.plot(xvals2,sample.pdf(xvals2),color='r', label='sample')
alphahighval = null.ppf(0.975)
ax.vlines(alphahighval,0,300,linestyle='--',color='g')
plt.legend()
plt.show()






'''Part 2.2'''
'''solution'''
def calc_min_sample_size(a1, a2, eff_size, alpha=0.05, two_tail=True, power=0.8):
   av1 = np.mean(a1)
   av2 = np.mean(a2)
   n1 = len(a1)
   n2 = len(a2)
   p = (av1 * n1 + av2 * n2)/(n1 + n2)
   sem = np.sqrt(p * (1-p) / n1 + p * (1-p)/ n2)
   pdiff = abs(av2 - av1)

   beta = 1-power
   if two_tail:
      alpha = alpha/2
   if pdiff >= eff_size:
      b = scs.norm(pdiff, sem).ppf(beta) + pdiff
      z_sem = b - eff_size
      z = scs.norm().ppf(1-alpha)
   else:
      b = scs.norm(pdiff, sem).ppf(1-beta) + pdiff
      z_sem = eff_size - b
      z = scs.norm().ppf(alpha)
   sem_des = z_sem / z
   n_des = (p * (1-p) / sem_des) ** 2
   return int(math.ceil(n_des))
