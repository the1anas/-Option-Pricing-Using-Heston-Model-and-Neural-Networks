import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from heston_model import heston_price  # Ensure this import is correct
import numpy as np

# Load and prepare data
data = pd.read_csv('merged_data.csv')
X = data[['Strike', 'Time_to_Maturity', 'Log_Return', 'Volatility', 'Last Price', 'Bid', 'Ask']].values
y = np.array([[2.0, 0.01, 0.1, -0.5, 0.01]] * len(data))  # Dummy Heston parameters for example

# Split data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# Load model
model = tf.keras.models.load_model('best_stock_prediction_model.keras')

# Training history
history = model.history.history

# Plot training & validation loss values
plt.figure(figsize=(10, 6))
plt.plot(history['loss'])
plt.plot(history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.savefig('training_validation_loss.png')
plt.show()


def validate_model(model, market_data_file):
    market_data = pd.read_csv(market_data_file)
    X_market = market_data[['Strike', 'Time_to_Maturity', 'Log_Return', 'Volatility', 'Last Price', 'Bid', 'Ask']].values
    market_prices = market_data['Option_Price'].values

    # Normalize the input features
    scaler = StandardScaler()
    X_market_scaled = scaler.fit_transform(X_market)

    # Predict Heston parameters
    predicted_params = model.predict(X_market_scaled)
    predicted_params = np.nan_to_num(predicted_params, nan=0.0, posinf=1e10, neginf=-1e10)

    # Calculate Heston option prices
    S0 = market_data['Close'].values  # Assuming the close price is the current stock price
    r = 0.05  # Risk-free rate, adjust as needed
    heston_prices = []
    for params, S, K, T in zip(predicted_params, S0, market_data['Strike'], market_data['Time_to_Maturity']):
        kappa, theta, sigma, rho, v0 = params
        heston_params = (kappa, theta, sigma, rho, v0)
        price = heston_price(heston_params, S, K, T, r)
        heston_prices.append(price)
    heston_prices = np.array(heston_prices)

    # Evaluate the model
    mse = mean_squared_error(market_prices, heston_prices)
    r2 = r2_score(market_prices, heston_prices)
    mae = mean_absolute_error(market_prices, heston_prices)

    # Plot actual vs predicted prices
    plt.figure(figsize=(10, 6))
    plt.scatter(market_prices, heston_prices)
    plt.plot([min(market_prices), max(market_prices)], [min(market_prices), max(market_prices)], 'r')
    plt.title('Predicted vs Actual Option Prices')
    plt.xlabel('Actual Option Prices')
    plt.ylabel('Predicted Option Prices')
    plt.savefig('predicted_vs_actual_prices.png')
    plt.show()

    return mse, r2, mae

# Example usage
model = tf.keras.models.load_model('best_stock_prediction_model.keras')
mse, r2, mae = validate_model(model, 'merged_data.csv')
print(f"Market Validation - MSE: {mse}, R^2: {r2}, MAE: {mae}")
