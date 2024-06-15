import yfinance as yf
import pandas as pd

def fetch_historical_data(ticker, start_date, end_date):
    stock = yf.Ticker(ticker)
    hist = stock.history(start=start_date, end=end_date)
    hist.to_csv('historical_data.csv')
    return hist

def fetch_option_data(ticker):
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
    option_df.to_csv('market_prices_googl.csv', index=False)
    return option_df

# Fetch data
fetch_historical_data('GOOGL', '2010-01-01', '2023-01-01')
fetch_option_data('GOOGL')
