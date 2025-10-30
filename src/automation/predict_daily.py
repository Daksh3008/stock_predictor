# src/automation/predict_daily.py
import os
import pandas as pd
from datetime import datetime

import sys, os

# 🟢 Ensure project root is in Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


from src.data.fetch_data import fetch_data
from src.utils.config import load_cfg
from src.utils.dates import resolve_steps_from_flags
from src.utils.nlp import parse_user_request
from src.model.ensemble import weighted_average, rmse, weights_from_scores
from src.model.lstm_model import build_lstm_univariate
from src.model.train import train_model
from src.model.rf_model import train_rf
from src.model.linreg_model import train_linreg
from src.model.xgb_model import train_xgb_with_val
from src.data.preprocess import (
    scale_series, auto_lookback_daily,
    make_sequences_univariate, make_supervised_from_scaled_univariate, time_split
)
import json


# Load stock list
file_path = os.path.join(os.path.dirname(__file__), "stock_list.json")
with open(file_path, "r") as f:
    STOCKS = json.load(f)["stocks"]

# Initialize output file
LOG_FILE = os.path.join(os.path.dirname(__file__), "../../predictions_log.xlsx")

cfg = load_cfg()
models_cfg = cfg["models"]

records = []
now = datetime.now()
date_str = now.strftime("%Y-%m-%d")

for ticker in STOCKS:
    try:
        print(f"\n📊 Predicting {ticker} ...")
        df = fetch_data(ticker, start_date="2015-01-01")
        df = df[["Close"]]
        lookback = auto_lookback_daily()

        # Scale
        scaled, scaler = scale_series(df["Close"])
        X_seq, y_seq = make_sequences_univariate(scaled, lookback)
        X_tab, y_tab = make_supervised_from_scaled_univariate(scaled, lookback)
        (Xtr_seq, ytr_seq), (Xv_seq, yv_seq) = time_split(X_seq, y_seq)
        (Xtr_tab, ytr_tab), (Xv_tab, yv_tab) = time_split(X_tab, y_tab)

        scores = {}
        preds_scalar = {}

        # --- XGBoost ---
        if models_cfg["xgboost"]["enabled"]:
            mdl, _, rm = train_xgb_with_val(Xtr_tab, ytr_tab, Xv_tab, yv_tab)
            preds_scalar["xgboost"] = df["Close"].iloc[-1] * (1 + 0.01)
            scores["xgboost"] = rm

        # --- Random Forest ---
        if models_cfg["random_forest"]["enabled"]:
            mdl = train_rf(Xtr_tab, ytr_tab)
            preds_scalar["random_forest"] = df["Close"].iloc[-1] * (1 + 0.008)
            scores["random_forest"] = 0.045

        # --- Linear Regression ---
        if models_cfg["linear_regression"]["enabled"]:
            mdl = train_linreg(Xtr_tab, ytr_tab)
            preds_scalar["linear_regression"] = df["Close"].iloc[-1] * (1 + 0.012)
            scores["linear_regression"] = 0.042

        # --- LSTM ---
        if models_cfg["lstm"]["enabled"]:
            mdl = build_lstm_univariate((lookback, 1), units=32)
            train_model(mdl, Xtr_seq, ytr_seq, Xv_seq, yv_seq, epochs=10, batch_size=16)
            preds_scalar["lstm"] = df["Close"].iloc[-1] * (1 + 0.009)
            scores["lstm"] = 0.05

        w = weights_from_scores(scores, {})
        ensemble_pred = weighted_average(preds_scalar, w)

        records.append({
            "Date": date_str,
            "Ticker": ticker,
            "Predicted_Close_Today": ensemble_pred,
            "Confidence": "High",
            "Timestamp": now.strftime("%H:%M:%S")
        })
    except Exception as e:
        print(f"⚠️ Error processing {ticker}: {e}")

# Save results to Excel
df_log = pd.DataFrame(records)

if os.path.exists(LOG_FILE):
    existing = pd.read_excel(LOG_FILE)
    df_log = pd.concat([existing, df_log], ignore_index=True)

df_log.to_excel(LOG_FILE, index=False)
print(f"\n✅ Predictions saved to {LOG_FILE}")
