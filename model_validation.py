import numpy as np
import pandas as pd
from scipy.integrate import quad
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Heston model functions
def heston_char_func(params, u, T, S0, r):
    kappa, theta, sigma, rho, v0 = params
    i = 1j
    d = np.sqrt((rho * sigma * i * u - kappa)**2 + (u**2 + i * u) * sigma**2)
    g = (kappa - rho * sigma * i * u - d) / (kappa - rho * sigma * i * u + d)
    
    C = r * i * u * T + (kappa * theta) / (sigma**2) * ((kappa - rho * sigma * i * u - d) * T - 2 * np.log((1 - g * np.exp(-d * T)) / (1 - g)))
    D = (kappa - rho * sigma * i * u - d) / (sigma**2) * ((1 - np.exp(-d * T)) / (1 - g * np.exp(-d * T)))
    
    return np.exp(C + D * v0 + i * u * np.log(S0))

def heston_price(params, S0, K, T, r):
    def integrand(u):
        return np.exp(-1j * u * np.log(K)) * heston_char_func(params, u - 1j, T, S0, r) / (1j * u * S0)

    price = 0.5 * S0 - np.real(quad(integrand, 0, np.inf)[0]) / np.pi
    return price

# Example parameters
params = (2.0, 0.01, 0.1, -0.5, 0.01)
S0 = 100  # Initial stock price
r = 0.05  # Risk-free rate

# Load the merged data
merged_data = pd.read_csv('merged_data.csv')

# Extract relevant columns
X_market = merged_data[['Strike', 'Time_to_Maturity', 'Log_Return', 'Volatility']].values
market_prices = merged_data['Option_Price'].values

# Predict option prices using the Heston model
heston_prices = []
for i in range(len(X_market)):
    K = X_market[i, 0]  # Strike price
    T = X_market[i, 1]  # Time to maturity
    heston_price_val = heston_price(params, S0, K, T, r)
    heston_prices.append(heston_price_val)

# Calculate metrics
mse = mean_squared_error(market_prices, heston_prices)
r2 = r2_score(market_prices, heston_prices)
mae = mean_absolute_error(market_prices, heston_prices)

print(f"Market Validation - MSE: {mse}, R^2: {r2}, MAE: {mae}")

# Plot results
plt.scatter(market_prices, heston_prices, color='blue', label='Predicted')
plt.plot([min(market_prices), max(market_prices)], [min(market_prices), max(market_prices)], color='red', label='Actual')
plt.xlabel('Actual Option Prices')
plt.ylabel('Predicted Option Prices')
plt.title('Predicted vs Actual Option Prices')
plt.legend()
plt.show()
