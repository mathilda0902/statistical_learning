# Multi-armed Bandit Problem

## Bayesian A/B testing:
While A/B testing with frequentist and Bayesian methods can be incredibly useful for determining the effectiveness of various changes to your products, better algorithms exist for making educated decision on-the-fly. Two such algorithms that typically out-perform A/B tests are extensions of the Multi-armed bandit problem which uses an epsilon-greedy strategy. Using a combination of exploration and exploitation, this strategy updates the model with each successive test, leading to higher overall click-through rate. An improvement on this algorithm uses an epsilon-first strategy called UCB1. Both can be used in lieu of traditional A/B testing to optimize products and click-through rates.

1. Posterior after n views, updated by prior distributions beta
2. simulating 10,000 points from site A's & B's beta distributions
