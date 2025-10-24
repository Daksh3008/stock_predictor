import yfinance as yf
import pandas as pd

def fetch_data(ticker: str, start_date="2015-01-01", end_date=None, freq="weekly"):
    if end_date is None:
        end_date = pd.Timestamp.today().strftime("%Y-%m-%d")

    df = yf.download(ticker, start=start_date, end=end_date, progress=False)
    if df.empty:
        raise ValueError(f"No data found for ticker {ticker}")

    df = df[["Open", "High", "Low", "Close"]]

    freq_map = {
        "daily": "D",
        "weekly": "W-FRI",
        "monthly": "M",
        "quarterly": "Q",
        "yearly": "A"
    }
    rule = freq_map.get(freq.lower())
    if rule is None:
        raise ValueError(f"Unsupported frequency '{freq}'")

    resampled = pd.DataFrame()
    resampled["Open"] = df["Open"].resample(rule).first()
    resampled["High"] = df["High"].resample(rule).max()
    resampled["Low"] = df["Low"].resample(rule).min()
    resampled["Close"] = df["Close"].resample(rule).last()
    return resampled.dropna()
