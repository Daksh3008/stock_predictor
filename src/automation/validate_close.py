# src/automation/validate_close.py
import os
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import sys, os

# 🟢 Ensure project root is in Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


LOG_FILE = os.path.join(os.path.dirname(__file__), "../../predictions_log.xlsx")

df = pd.read_excel(LOG_FILE)
today_str = datetime.now().strftime("%Y-%m-%d")

# Get today’s predictions only
today_preds = df[df["Date"] == today_str].copy()

actuals = []
for _, row in today_preds.iterrows():
    ticker = row["Ticker"]
    try:
        data = yf.download(ticker, period="2d", progress=False)
        if not data.empty:
            actual_close = data["Close"].iloc[-1]
            pred_close = row["Predicted_Close_Today"]
            pct_error = abs((actual_close - pred_close) / actual_close) * 100
            actuals.append({
                "Ticker": ticker,
                "Date": today_str,
                "Predicted_Close_Today": pred_close,
                "Actual_Close": actual_close,
                "Pct_Error": pct_error
            })
    except Exception as e:
        print(f"⚠️ {ticker} fetch failed: {e}")

# Append results to Excel
if actuals:
    actual_df = pd.DataFrame(actuals)
    df_merged = pd.merge(df, actual_df, on=["Date", "Ticker"], how="left")
    df_merged.to_excel(LOG_FILE, index=False)
    print(f"✅ Validation appended for {len(actuals)} stocks")
else:
    print("⚠️ No actuals fetched.")
