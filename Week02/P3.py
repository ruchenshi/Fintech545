import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv('problem3.csv')

# Assuming the time series data is in the first column
ts = data.iloc[:, 0]

# Fit AR and MA models of orders 1 through 3
orders = range(1, 4)
ar_models = {order: ARIMA(ts, order=(order, 0, 0)).fit() for order in orders}
ma_models = {order: ARIMA(ts, order=(0, 0, order)).fit() for order in orders}

# Plotting the fits
for order in orders:
    plt.figure(figsize=(12, 4))
    plt.plot(ts, label='Original')
    plt.plot(ar_models[order].fittedvalues, label=f'AR({order}) Fit')
    plt.title(f'AR({order}) Time Series Fit')
    plt.legend()
    plt.show()

for order in orders:
    plt.figure(figsize=(12, 4))
    plt.plot(ts, label='Original')
    plt.plot(ma_models[order].fittedvalues, label=f'MA({order}) Fit')
    plt.title(f'MA({order}) Time Series Fit')
    plt.legend()
    plt.show()

# Evaluate the models and collect AIC and BIC
model_metrics = {
    f'AR({order})': (model.aic, model.bic) for order, model in ar_models.items()
}
model_metrics.update({
    f'MA({order})': (model.aic, model.bic) for order, model in ma_models.items()
})

# Determine the best fit based on AIC and BIC
best_aic = min(model_metrics, key=lambda x: model_metrics[x][0])
best_bic = min(model_metrics, key=lambda x: model_metrics[x][1])

# Print out the AIC and BIC for each model and the best ones
print("AIC and BIC for each model:")
for model, metrics in model_metrics.items():
    print(f"{model}: AIC={metrics[0]}, BIC={metrics[1]}")

print(f"Best fit based on AIC: {best_aic}")
print(f"Best fit based on BIC: {best_bic}")
