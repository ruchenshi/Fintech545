import numpy as np
import pandas as pd

# Function to calculate historical VaR for a given portfolio
def calculate_historical_var(values, confidence_level=0.95):
    # Calculate daily portfolio changes
    portfolio_changes = np.diff(values, axis=0)
    # Calculate the VaR at the specified confidence level
    # Since we're looking at losses, we take the lower end of the sorted changes (percentile)
    var = -np.percentile(portfolio_changes, (1 - confidence_level) * 100)
    return var

# Load the portfolio and daily prices data
portfolio = pd.read_csv('Week04/portfolio.csv')
prices = pd.read_csv('Week04/DailyPrices.csv')

# Calculate historical portfolio values for each portfolio
portfolios = ['A', 'B', 'C']
historical_vars = {}

# Calculate historical VaR for each portfolio
for portfolio_name in portfolios:
    stock_list = portfolio[portfolio['Portfolio'] == portfolio_name]['Stock'].tolist()
    holdings = portfolio[portfolio['Portfolio'] == portfolio_name]['Holding'].values
    portfolio_prices = prices[stock_list].values
    portfolio_values = portfolio_prices @ holdings
    historical_vars[portfolio_name] = calculate_historical_var(portfolio_values)

# Calculate total historical VaR (combining all portfolios)
all_stocks = portfolio['Stock'].unique()
all_holdings = portfolio.groupby('Stock')['Holding'].sum().reindex(all_stocks, fill_value=0).values
all_prices = prices[all_stocks].values
total_values = all_prices @ all_holdings
historical_vars['Total'] = calculate_historical_var(total_values)

print(f"Portfolio A VaR: ${historical_vars['A']:.2f}")
print(f"Portfolio B VaR: ${historical_vars['B']:.2f}")
print(f"Portfolio C VaR: ${historical_vars['C']:.2f}")
print(f"Total VaR: ${historical_vars['Total']:.2f}")
