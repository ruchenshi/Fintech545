import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Function to read data and drop the first column (which seems to contain dates or an index)
def read_data(file_path):
    rets = pd.read_csv(file_path)
    rets.drop(rets.columns[0], axis=1, inplace=True)
    return rets

# Function to calculate exponentially weighted covariance matrix
def ew_covar(x, lambda_):
    m, n = x.shape
    weights = np.array([(1 - lambda_) * lambda_ ** (m - i) for i in range(1, m + 1)])
    weights /= weights.sum()
    x -= x.mean()
    return np.cov(x.T, aweights=weights)

# Function to calculate the weights for the exponential weighting
def exp_w(m, lambda_):
    weights = np.array([(1 - lambda_) * lambda_ ** (m - i) for i in range(1, m + 1)])
    weights /= weights.sum()
    return weights

# Function to calculate the percentage of variance explained by the PCA components
def pca_pct_explained(a):
    pca = PCA()
    pca.fit(a)
    return np.cumsum(pca.explained_variance_ratio_)

# Read the dataset
rets = read_data("DailyReturn.csv")

# Apply PCA and calculate the percentage explained for different lambda values
lambda_values = [0.75, 0.85, 0.90, 0.95, 0.99]
pct_explained = {}

for lambda_ in lambda_values:
    covar = ew_covar(rets.to_numpy(), lambda_)
    pct_explained[lambda_] = pca_pct_explained(covar)

# Plot the results
plt.figure(figsize=(10, 7))
for lambda_, values in pct_explained.items():
    plt.plot(values, label=f'λ={lambda_}')

plt.title('% Explained by Eigenvalue')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.legend(loc='lower right')
plt.show()
