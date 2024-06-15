import pandas as pd
import numpy as np

def preprocess_historical_data(file_path):
    data = pd.read_csv(file_path, parse_dates=True)
    
    # Handle NaN values
    data = data.dropna()

    # Calculate Log Return and Volatility
    data['Log_Return'] = np.log(data['Close'] / data['Close'].shift(1))
    data['Volatility'] = data['Log_Return'].rolling(window=21).std() * np.sqrt(252)
    data = data.dropna()
    
    data.to_csv("processed_historical_data.csv", index=False)
    print("Preprocessed historical data saved to processed_historical_data.csv")
    return data

def preprocess_option_data(file_path):
    data = pd.read_csv(file_path)
    
    # Handle NaN values
    data = data.dropna()

    data.to_csv("processed_market_prices.csv", index=False)
    print("Preprocessed market prices data saved to processed_market_prices.csv")
    return data

def merge_data(hist_data_path, option_data_path):
    hist_data = preprocess_historical_data(hist_data_path)
    option_data = preprocess_option_data(option_data_path)

    # Ensure matching dates or keys
    merged_data = pd.merge(option_data, hist_data, left_index=True, right_index=True, how='inner')
    
    merged_data.to_csv('merged_data.csv', index=False)
    print("Merged data saved to merged_data.csv")
    return merged_data

# Merge data
merged_data = merge_data('historical_data.csv', 'market_prices_googl.csv')
print(merged_data.head())

import pandas as pd

def verify_data(file_path):
    data = pd.read_csv(file_path)
    print(data.head())
    print(data.columns)

# Verify merged data
verify_data('merged_data.csv')
