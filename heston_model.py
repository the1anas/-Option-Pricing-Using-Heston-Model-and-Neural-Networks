import numpy as np
from scipy.integrate import quad

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
