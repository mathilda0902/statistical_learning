import numpy as np
import scipy.stats as scs
import matplotlib.pyplot as plt

site_a = np.loadtxt('data/siteA.txt')
site_b = np.loadtxt('data/siteB.txt')

x = np.arange(0, 1.01, 0.01)
y0 = scs.uniform().pdf(x)

def plot_with_fill(x, y, label):
    lines = plt.plot(x, y, label=label, lw=2)
    plt.fill_between(x, 0, y, alpha=0.2, color=lines[0].get_c())

plot_with_fill(x, y, 'Prior')
plt.legend()
plt.show()

y1 = scs.beta(a=2, b=1).pdf(x)

'''site A after different views'''

alpha = sum(site_a[:50])
beta = 50 - alpha
y2 = scs.beta(alpha, beta).pdf(x)
plot_with_fill(x, y0 , 'Prior')
plot_with_fill(x, y2, 'Posterior after 50 views')

for i in range(4):
    alpha = sum(site_a[:100*(2**i)])
    beta = 100*(2**i) - alpha
    y = scs.beta(alpha, beta).pdf(x)
    plot_with_fill(x,y, 'Posterior after {} views'.format(100*(2**i)))

plt.legend()
plt.show()

'''site A & B after 800 views'''
alpha_a = sum(site_a)
beta_a = len(site_a) - alpha_a
alpha_b = sum(site_b)
beta_b = len(site_b) - alpha_b

y_a = scs.beta(alpha_a, beta_a).pdf(x)
y_b = scs.beta(alpha_b, beta_b).pdf(x)

'''plotting'''
plot_with_fill(x, y0, 'Prior')
label_str='Posterior after 800 views.'
plot_with_fill(x, y_a, label='Site A'+label_str)
plot_with_fill(x, y_b, label='Site B'+label_str)
plt.xlim([0, 0.2])
plt.savefig('site_a_site_b_posteriors')
plt.legend()
plt.show()

'''simulating 10,000 points from site A's & B's beta distributions'''
size = 10000
sim_a = np.random.beta(alpha_a, beta_a, size)
sim_b = np.random.beta(alpha_b, beta_b, size)
percent_likelihood = sum(np.greater(sim_b,sim_a)) / 10000.

'''In [69]: percent_likelihood
Out[69]: 0.99509999999999998'''

lower_a = scs.beta(alpha_a, beta_a).ppf(0.025)
upper_a = scs.beta(alpha_a, beta_a).ppf(0.975)
print("A's 95% HDI is {:.5f} to {:.5f}".format(lower_a, upper_a))

lower_b = scs.beta(alpha_b,beta_b).ppf(0.025)
upper_b = scs.beta(alpha_b,beta_b).ppf(0.975)
print("B's 95% HDI is {:.5f} to {:.5f}".format(lower_b, upper_b))

'''B > A + 0.02'''
new_arr = sim_b - sim_a - 0.02
plt.hist(new_arr, bins=20, normed=1)
plt.savefig('site_difference')
plt.show()

'''null hypothesis: site A has same conversion rate as site B'''
'''alternative hypothesis: site A has significanty different rate as site B'''
t, p = scs.ttest_ind(site_a + 0.02, site_b, equal_var=False)
if p < 0.05:
    print "P value: {}. Reject null hypothesis and conclude that site A has a significantly different conversion rate than site B.".format(p)
else:
    print "P value: {}. Fail to reject null hypothsis and conclude that there is not significant evidence that site A has a different rate than site B.".format(p)

'''profits: a click on site A is worth $1.00
a click on site B is worth $1.05'''
expect_a = sim_a.mean()
expect_b = sim_b.mean()

def exp_profit(exp_a, exp_b, hits):
    return (exp_b * 1.05 - exp_a * 1)*hits

hits = [10000* (10**i) for i in range(6)]
diff_profit = [exp_profit(expect_a, expect_b, hit) for hit in hits]
for hit, diff in zip(hits, diff_profit):
    print "Expected difference in profits of ${} for {} hits for each site.".format(diff, hit)
