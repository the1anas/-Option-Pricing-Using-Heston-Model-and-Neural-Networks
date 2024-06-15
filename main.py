import pandas as pd
from tensorflow.keras.models import load_model

# Load the saved model
model = load_model('best_stock_prediction_model.keras')

# Prepare input data for prediction
def predict_heston_params(model, input_data):
    predictions = model.predict(input_data)
    return predictions

# Example usage
X_val = pd.read_csv('processed_data.csv')  
predicted_params = predict_heston_params(model, X_val)

# Calculate option prices
S0 = 100  # Example spot price
K = 110  # Example strike price
T = 1  # Example time to maturity
r = 0.05  # Example risk-free rate

predicted_prices = [heston_price(params, S0, K, T, r) for params in predicted_params]
print(predicted_prices)
