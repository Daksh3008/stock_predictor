import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from src.data.fetch_data import fetch_data
from src.data.preprocess import auto_lookback, scale_series, make_sequences
from src.model.lstm_model import build_lstm
from src.model.train import train_model
from src.model.predict import predict_for_date
from src.utils.config import get_default_config
from src.utils.helpers import find_ticker

def main():
    cfg = get_default_config()

    company_name = input("Enter company name: ").strip()
    ticker = find_ticker(company_name)
    price_type = input("Predict which price (Open/High/Low/Close): ").strip().capitalize()
    freq = input("Select frequency (daily/weekly/monthly/quarterly/yearly): ").strip().lower()
    target_date = input("Enter target date (YYYY-MM-DD): ").strip()

    df = fetch_data(ticker, start_date=cfg["start_date"], freq=freq)
    lookback = auto_lookback(freq)
    scaled, scaler = scale_series(df[price_type])
    X, y = make_sequences(scaled, lookback)
    split = int(len(X) * 0.8)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    model = build_lstm((lookback, 1), units=cfg["lstm_units"], dropout=cfg["dropout"])
    train_model(model, X_train, y_train, X_val, y_val,
                epochs=cfg["epochs"], batch_size=cfg["batch_size"])

    predicted_value = predict_for_date(model, scaler, scaled, lookback, df, target_date, freq)
    print(f"\n🔮 Predicted {price_type} price for {ticker} on {target_date}: ₹{predicted_value:,.2f}\n")


if __name__ == "__main__":
    main()
