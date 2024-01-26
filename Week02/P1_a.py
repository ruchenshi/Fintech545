import pandas as pd

# Load the data
data = pd.read_csv('problem1.csv')

# Calculate the sample size
n = len(data)

# Calculate the mean (1st Moment)
mean = data['x'].sum() / n

# Calculate the variance (2nd Moment) - using the unbiased estimator with n-1 in the denominator
variance = ((data['x'] - mean) ** 2).sum() / (n - 1)

# Calculate the standard deviation (for use in skewness and kurtosis calculations)
std_dev = variance ** 0.5

# Calculate the skewness (3rd Moment) - this is the third standardized moment
skewness = ((data['x'] - mean) ** 3).sum() / n
skewness /= std_dev ** 3

# Calculate the kurtosis (4th Moment) - this is the fourth standardized moment, minus 3 to get excess kurtosis
kurtosis = ((data['x'] - mean) ** 4).sum() / n
kurtosis /= std_dev ** 4
kurtosis -= 3

# Output the results
print(f"Mean: {mean}")
print(f"Variance: {variance}")
print(f"Skewness: {skewness}")
print(f"Kurtosis: {kurtosis}")
