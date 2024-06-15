import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def perform_eda(file_path):
    data = pd.read_csv(file_path, parse_dates=True)

    # Plotting the option prices
    plt.figure(figsize=(10, 6))
    plt.plot(data['Option_Price'])
    plt.title('Option Prices')
    plt.xlabel('Index')
    plt.ylabel('Option Price')
    plt.savefig('option_prices.png')
    plt.show()

    # Correlation matrix
    corr = data.corr(numeric_only=True)
    plt.figure(figsize=(10, 6))
    sns.heatmap(corr, annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.savefig('correlation_matrix.png')
    plt.show()

if __name__ == "__main__":
    file_path = "processed_market_prices.csv"
    perform_eda(file_path)
