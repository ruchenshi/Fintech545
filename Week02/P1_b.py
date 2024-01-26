import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis

# Assuming 'problem1.csv' is in the current working directory
data = pd.read_csv('problem1.csv')

# Calculate the mean (1st Moment)
mean_value = np.mean(data['x'])

# Calculate the variance (2nd Moment)
# ddof=1 specifies that we want the sample variance
variance_value = np.var(data['x'], ddof=1)

# Calculate the skewness (3rd Moment)
# bias=False specifies that we want the unbiased estimator of skewness
skewness_value = skew(data['x'], bias=False)

# Calculate the kurtosis (4th Moment)
# fisher=True specifies that we want the excess kurtosis
# bias=False specifies that we want the unbiased estimator of kurtosis
kurtosis_value = kurtosis(data['x'], fisher=True, bias=False)

# Print the results
print(f"Mean: {mean_value}")
print(f"Variance: {variance_value}")
print(f"Skewness: {skewness_value}")
print(f"Kurtosis: {kurtosis_value}")
