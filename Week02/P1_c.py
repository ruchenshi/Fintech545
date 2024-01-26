import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, skew, kurtosis

# Load the original dataset
data = pd.read_csv('problem1.csv')

# Define the number of bootstrap samples
n_bootstrap = 1000

# Initialize lists to store the moments for manual and package calculations
manual_means, package_means = [], []
manual_variances, package_variances = [], []
manual_skewnesses, package_skewnesses = [], []
manual_kurtoses, package_kurtoses = [], []

for _ in range(n_bootstrap):
    # Create a bootstrap sample with replacement
    n = len(data)
    bootstrap_sample = data.sample(n, replace=True)
    
    # Calculate the moments manually
    # Calculate the mean (1st Moment)
    mean_manual = bootstrap_sample['x'].sum() / n

    # Calculate the variance (2nd Moment) - using the unbiased estimator with n-1 in the denominator
    variance_manual = ((bootstrap_sample['x'] - mean_manual) ** 2).sum() / (n - 1)

    # Calculate the standard deviation (for use in skewness and kurtosis calculations)
    std_dev = variance_manual ** 0.5

    # Calculate the skewness (3rd Moment) - this is the third standardized moment
    skewness_manual = ((bootstrap_sample['x'] - mean_manual) ** 3).sum() / n
    skewness_manual /= std_dev ** 3

    # Calculate the kurtosis (4th Moment) - this is the fourth standardized moment, minus 3 to get excess kurtosis
    kurtosis_manual = ((bootstrap_sample['x'] - mean_manual) ** 4).sum() / n
    kurtosis_manual /= std_dev ** 4
    kurtosis_manual -= 3
    
    # Store the manual calculations
    manual_means.append(mean_manual)
    manual_variances.append(variance_manual)
    manual_skewnesses.append(skewness_manual)
    manual_kurtoses.append(kurtosis_manual)
    
    # Use package functions for comparison (scipy.stats)
    mean_package = np.mean(bootstrap_sample['x'])
    variance_package = np.var(bootstrap_sample['x'], ddof=1)
    skewness_package = skew(bootstrap_sample['x'], bias=False)
    kurtosis_package = kurtosis(bootstrap_sample['x'], fisher=True, bias=False)
    
    # Store the package calculations
    package_means.append(mean_package)
    package_variances.append(variance_package)
    package_skewnesses.append(skewness_package)
    package_kurtoses.append(kurtosis_package)

# Perform paired t-tests for each moment
t_test_mean = ttest_rel(manual_means, package_means)
t_test_variance = ttest_rel(manual_variances, package_variances)
t_test_skewness = ttest_rel(manual_skewnesses, package_skewnesses)
t_test_kurtosis = ttest_rel(manual_kurtoses, package_kurtoses)

# Output the t-test results
print('Paired t-test results:')
print(f'Mean: statistic={t_test_mean.statistic}, p-value={t_test_mean.pvalue}')
print(f'Variance: statistic={t_test_variance.statistic}, p-value={t_test_variance.pvalue}')
print(f'Skewness: statistic={t_test_skewness.statistic}, p-value={t_test_skewness.pvalue}')
print(f'Kurtosis: statistic={t_test_kurtosis.statistic}, p-value={t_test_kurtosis.pvalue}')

