import yfinance as yf
import pandas as pd

def collect_google_option_data():
    ticker = 'GOOGL'
    stock = yf.Ticker(ticker)
    options = stock.options
    option_data = []

    for expiry in options:
        opt = stock.option_chain(expiry)
        for call in opt.calls.itertuples():
            option_data.append([
                expiry,
                call.strike,
                call.lastPrice,
                call.bid,
                call.ask,
                call.volume,
                call.openInterest
            ])

    option_df = pd.DataFrame(option_data, columns=['Expiry', 'Strike', 'Last Price', 'Bid', 'Ask', 'Volume', 'Open Interest'])
    option_df['Time_to_Maturity'] = (pd.to_datetime(option_df['Expiry']) - pd.to_datetime('today')).dt.days / 365.25
    option_df['Option_Price'] = option_df['Last Price']
    option_df = option_df.dropna()
    option_df.to_csv('market_prices_googl.csv', index=False)
    print("GOOGL option data collected and saved to market_prices_googl.csv")

# Run data collection
collect_google_option_data()

