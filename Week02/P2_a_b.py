import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.optimize import minimize
from scipy import stats

# Load the data
data_path = 'problem2.csv'
data = pd.read_csv(data_path)

# Assuming 'x' is the independent variable and 'y' is the dependent variable
X = data['x']
y = data['y']

# Add a constant to the independent variable for the intercept
X_with_constant = sm.add_constant(X)

# Fit the OLS model
model_OLS = sm.OLS(y, X_with_constant).fit()

# Extracting the beta coefficients and the standard deviation of the errors (residuals)
beta_OLS = model_OLS.params
std_dev_OLS = model_OLS.resid.std()

# Define the likelihood function for MLE
def normal_log_likelihood(params, X, y):
    beta = params[:-1]
    sigma = params[-1]
    y_pred = X.dot(beta)
    neg_log_likelihood = -np.sum(stats.norm.logpdf(y, y_pred, sigma))
    return neg_log_likelihood

# Initial guess for the parameters (beta_0, beta_1, ..., beta_p, sigma)
# We start with the OLS estimates for beta and an arbitrary value for sigma
initial_guess = np.append(beta_OLS.values, std_dev_OLS)

# Minimize the negative log-likelihood
result = minimize(
    fun=normal_log_likelihood, 
    x0=initial_guess, 
    args=(X_with_constant, y), 
    method='L-BFGS-B', 
    bounds=[(None, None)] * (len(X_with_constant.columns)) + [(1e-8, None)]
)

# The MLE estimates for beta coefficients and sigma (standard deviation of errors)
beta_MLE = result.x[:-1]
sigma_MLE = result.x[-1]

# Print the results
print("OLS Estimates:")
print("Beta coefficients:", beta_OLS)
print("Standard deviation of the residuals:", std_dev_OLS)

print("\nMLE Estimates:")
print("Beta coefficients:", beta_MLE)
print("Sigma (standard deviation of errors):", sigma_MLE)

from scipy.stats import t as t_dist

# Define the likelihood function for MLE with T-distributed errors
def t_dist_log_likelihood(params, X, y, df):
    beta = params[:-1]
    sigma = params[-1]
    y_pred = X.dot(beta)
    neg_log_likelihood = -np.sum(t_dist.logpdf(y, df, loc=y_pred, scale=sigma))
    return neg_log_likelihood

# Degrees of freedom for the T-distribution
# Common practice is to set degrees of freedom to a value that indicates heavy tails
# Here we will use 3 as a starting point which indicates heavier tails than the normal distribution
df_t = 3

# Initial guess for the parameters (beta_0, beta_1, ..., beta_p, sigma)
# We start with the OLS estimates for beta and an arbitrary value for sigma
initial_guess_t = np.append(beta_OLS.values, std_dev_OLS)

# Minimize the negative log-likelihood for T-distribution
result_t = minimize(
    fun=t_dist_log_likelihood, 
    x0=initial_guess_t, 
    args=(X_with_constant, y, df_t), 
    method='L-BFGS-B', 
    bounds=[(None, None)] * (len(X_with_constant.columns)) + [(1e-8, None)]
)

# The MLE estimates for beta coefficients and sigma (standard deviation of errors) for T-distribution
beta_MLE_t = result_t.x[:-1]
sigma_MLE_t = result_t.x[-1]
log_likelihood_t = -result_t.fun  # The log-likelihood for the T-distribution

# Compare the log-likelihoods for Normal and T distributions
log_likelihood_normal = -normal_log_likelihood(np.append(beta_MLE, sigma_MLE), X_with_constant, y)

print("beta_MLE_t, sigma_MLE_t, log_likelihood_t, log_likelihood_normal: ",beta_MLE_t, sigma_MLE_t, log_likelihood_t, log_likelihood_normal)
