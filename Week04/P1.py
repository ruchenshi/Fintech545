import numpy as np


# Set simulation parameters
T = 1000  # number of time steps
sigma = 0.01
n = 10000
# Initialize arrayP
P_0 = 100
P1_sum = []

for i in range(n):
    r = np.random.normal(0, sigma, size = T)
    P = P_0 + np.cumsum(r)
    P1_sum.append(P[-1])

# Calculate expected std
expected_classical_std = sigma * np.sqrt(T)

# Print mean and standard deviation of P

print("1. Classical:")
print("Mean of P:", np.mean(P1_sum))
print("Expected Mean of P:", 100)
print("Standard deviation of P:", np.std(P1_sum))
print("Expected Standard deviation of P:", expected_classical_std)


P2_sum = []
P_2 = []

for i in range(n):
    r = np.random.normal(0, sigma, size = T)
    P = P_0 * np.cumprod(1+r)
    P2_sum.append(P[-1])
    P_2.append(P[-2])

    

# Print mean and standard deviation of P
print("2. Arithmatic:")
print("Mean of P:", np.mean(P2_sum))
print("Standard deviation of P:", np.std(P2_sum))
expected_arithmatic_std = np.mean(np.std(P_2))
print("Expected Mean of P:", 100)
print("Expected Standard deviation of P:", expected_arithmatic_std)


P3_sum = []
for i in range(n):
    r = np.random.normal(0, sigma, size = T)
    P = P_0 * np.exp(np.cumsum(r))
    P3_sum.append(P[-1])
# Print mean and standard deviation of P
print("3. Log Return:")
print("Mean of P:", np.mean(P3_sum))
print("Standard deviation of P:", np.std(P3_sum))
# Expected value
mu = (sigma**2/2)
expected_mean = P_0 * np.exp(mu*T)
expected_std = P_0 * np.sqrt((np.exp(sigma**2 * T) - 1) * np.exp(2 * mu * T))
print("Expected Mean of P:", expected_mean)
print("Expected Standard deviation of P:", expected_std)