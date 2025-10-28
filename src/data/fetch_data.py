import yfinance as yf
import pandas as pd

def fetch_data(ticker: str, start_date="2010-01-01", end_date=None):
    if end_date is None:
        end_date = pd.Timestamp.today().strftime("%Y-%m-%d")

    df = yf.download(
        ticker,
        start=start_date,
        end=end_date,
        progress=False,
        auto_adjust=False  # keep raw OHLC
    )
    if df.empty:
        raise ValueError(f"No data found for ticker {ticker}")

    df = df[["Open", "High", "Low", "Close"]].dropna()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df
