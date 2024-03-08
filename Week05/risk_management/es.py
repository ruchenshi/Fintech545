import numpy as np
from scipy.stats import norm
from scipy.stats import t
from scipy import optimize

def es_normal(data, alpha=0.05):
    
    # Assuming the data is in the first column, otherwise, adjust 'iloc[:,0]' to the correct column index
    a = data.iloc[:, 0].values
    a.sort()
    v = np.quantile(a, alpha)
    es = a[a <= v].mean()
    es_absolute = -es  # ES is negative for losses; we report as positive
    mean_of_a = a.mean()
    es_diff_from_mean = mean_of_a - es
    return es_absolute, es_diff_from_mean

# t distribution:
def MLE_t(pars, x):
    df = pars[0]
    loc=pars[1]
    scale = pars[2]
    ll = np.log(t.pdf(x, df=df,loc=loc,scale=scale)) 
    return -ll.sum()

def es_t_dist(data, alpha=0.05):
    x = np.array(data)
    mean_x = x.mean()
    std_x = x.std()
    cons = ({'type': 'ineq', 'fun': lambda x: x[0] - 2}, {'type': 'ineq', 'fun': lambda x: x[2]})
    model = optimize.minimize(fun=MLE_t, x0=[2, mean_x, std_x], constraints=cons, args=x).x
    
    df = model[0]
    loc = model[1]
    scale = model[2]
    t_sample = t.rvs(df=df, loc=loc, scale=scale, size=10000)
    
    t_VaR = -t.ppf(alpha, df=df, loc=loc, scale=scale)
    ES_t = -t_sample[t_sample < -t_VaR].mean()
    
    es_absolute = ES_t
    es_diff_from_mean = mean_x + es_absolute
    
    return es_absolute, es_diff_from_mean

def es_simulation(data, alpha=0.05, iterations=100000):
    returns = data.values
    simulated_returns = np.random.choice(returns, size=iterations, replace=True)
    sorted_returns = np.sort(simulated_returns)
    var_position = int(alpha * iterations)
    es = np.mean(sorted_returns[:var_position])
    return abs(es), abs(es - np.mean(returns))