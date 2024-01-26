import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

# Load the dataset
data_x = pd.read_csv('problem2_x.csv')
data_x1 = pd.read_csv('problem2_x1.csv')

# Calculate the sample mean vector and covariance matrix for MLE of the multivariate normal distribution
mean_vector = data_x.mean(axis=0)
cov_matrix = data_x.cov()

# For a multivariate normal distribution X = [X1, X2], the conditional distribution of X2 given X1 is normal with:
# mean(X2 | X1) = μ2 + Σ21 * Σ11^(-1) * (X1 - μ1)
# var(X2 | X1) = Σ22 - Σ21 * Σ11^(-1) * Σ12
mu1, mu2 = mean_vector[0], mean_vector[1]
sigma11 = cov_matrix.iloc[0, 0]
sigma22 = cov_matrix.iloc[1, 1]
sigma12 = cov_matrix.iloc[0, 1]
sigma21 = cov_matrix.iloc[1, 0]

# Invert sigma11
sigma11_inv = 1 / sigma11

# Calculate the conditional mean and variance
conditional_mean = mu2 + sigma21 * sigma11_inv * (data_x1['x1'] - mu1)
conditional_variance = sigma22 - sigma21 * sigma11_inv * sigma12

# Calculate the 95% confidence intervals for the conditional distribution of X2
z = norm.ppf(0.975)  # 97.5% percentile of the standard normal distribution for a 95% CI
ci_lower = conditional_mean - z * np.sqrt(conditional_variance)
ci_upper = conditional_mean + z * np.sqrt(conditional_variance)

# Plot the expected value of X2 with the 95% confidence interval
plt.figure(figsize=(10, 6))
plt.plot(data_x1['x1'], conditional_mean, label='Expected value of X2 given X1')
plt.fill_between(data_x1['x1'], ci_lower, ci_upper, color='gray', alpha=0.2, label='95% CI')
plt.xlabel('X1')
plt.ylabel('Expected X2 | X1')
plt.title('Conditional Expectation of X2 given X1 with 95% Confidence Interval')
plt.legend()
plt.show()
