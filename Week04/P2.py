import pandas as pd
import numpy as np
from scipy.stats import norm, t
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.stattools import adfuller

# Function to calculate returns based on the specified method
def return_calculate(prices_df, method="DISCRETE", date_column="Date"):
    if date_column not in prices_df.columns:
        raise ValueError(f"dateColumn: {date_column} not in DataFrame columns")

    prices = prices_df.drop(columns=[date_column])
    if method.upper() == "DISCRETE":
        returns = prices.pct_change().dropna()
    elif method.upper() == "LOG":
        returns = np.log(prices / prices.shift(1)).dropna()
    else:
        raise ValueError("method must be in (\"LOG\",\"DISCRETE\")")
    
    returns[date_column] = prices_df[date_column].iloc[1:]  # Add dates back excluding the first NaN row
    return returns

# Function to remove the mean from the series
def demean_series(series):
    return series - series.mean()

# Function to calculate the Value at Risk (VaR)
def calculate_var(returns, level=5):
    """
    Calculate VaR using different methods:
    1. Normal distribution.
    2. Normal distribution with exponentially weighted variance.
    3. MLE fitted T distribution.
    4. Fitted AR(1) model.
    5. Historical Simulation.
    """
    # Convert the level to a probability
    alpha = level / 100

    # 1. Normal distribution
    var_normal = norm.ppf(alpha, returns.mean(), returns.std())
    
    # 2. Normal distribution with exponentially weighted variance
    lambda_param = 0.94
    ewm_variance = returns.ewm(alpha=(1 - lambda_param)).var().iloc[-1]
    var_ewma = norm.ppf(alpha, returns.mean(), np.sqrt(ewm_variance))
    
    # 3. MLE fitted T distribution
    params = t.fit(returns.dropna())
    var_t = t.ppf(alpha, *params)
    
    # 4. Fitted AR(1) model - check for stationarity first
    if adfuller(returns.dropna())[1] < 0.05:  # p-value < 0.05 implies stationarity
        ar_model = AutoReg(returns.dropna(), lags=1).fit()
        forecast = ar_model.predict(start=len(returns), end=len(returns), dynamic=False)
        var_ar1 = forecast - norm.ppf(1-alpha) * returns.std()
    else:
        var_ar1 = np.nan  # Non-stationary, AR(1) model not suitable
    
    # 5. Historical Simulation
    sorted_returns = np.sort(returns)
    var_hist = np.percentile(sorted_returns, alpha*100)
    
    
    return {
        'VaR_Normal': var_normal,
        'VaR_EWMA': var_ewma,
        'VaR_T': var_t,
        'VaR_AR1': var_ar1,
        'VaR_Historical': var_hist
    }

# Load the data
file_path = 'Week04/DailyPrices.csv'
prices_df = pd.read_csv(file_path)

# Calculate the arithmetic returns for all prices
returns_df = return_calculate(prices_df, method="DISCRETE")

# Demean the META series
returns_df['META'] = demean_series(returns_df['META'])

# Calculate VaR for META
var_results = calculate_var(returns_df['META'])

# Display the results
print("Value at Risk (VaR) for META at the 5% level using different methods:")
for method, var in var_results.items():
    if isinstance(var, pd.Series):
        # If the value is a Series, get the scalar value
        var = var.iloc[0]
    print(f"{method}: {var:.4f}")
