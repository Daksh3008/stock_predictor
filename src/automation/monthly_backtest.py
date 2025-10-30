# monthly backtest for 4 models + ensemble
import sys, os, math
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import json
from datetime import datetime
import pandas as pd
import numpy as np
from src.data.fetch_data import fetch_data                # <<< existing util
from src.data.preprocess import (
    auto_lookback_daily, scale_series,
    make_sequences_univariate, make_supervised_from_scaled_univariate,
    time_split, to_prophet_frame
)
from src.model.xgb_model import train_xgb_with_val        # <<< existing model trainer
from src.model.rf_model import train_rf
from src.model.linreg_model import train_linreg
from src.model.lstm_model import build_lstm_univariate
from src.model.train import train_model                   # wrapper to train lstm with earlystop
from src.model.predict import predict_tabular_series, predict_lstm_series
from src.model.ensemble import rmse, weights_from_scores, weighted_average

#config
STOCK_LIST_FILE = os.path.join(os.path.dirname(__file__), "stock_list.json")
OUT_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../monthly_backtest_results.xlsx"))
TRAIN_YEARS = 20
MIN_TRAIN_DAYS = 365 * 5   # safety guard

def load_stocks():
    with open(STOCK_LIST_FILE, "r") as f:
        s = json.load(f)
    return s.get("stocks", [])

def start_of_current_month(ref_date: pd.Timestamp):
    return pd.Timestamp(year=ref_date.year, month=ref_date.month, day=1)

def last_business_day_before(ts: pd.Timestamp):
    # return previous calendar day; fetch_data will align to trading days
    return ts - pd.Timedelta(days=1)

def build_train_validation_split(df, train_end_date):
    # training: up to train_end_date (inclusive), validation: from next business day until month end
    train = df.loc[:train_end_date].copy()
    val = df.loc[train_end_date + pd.Timedelta(days=1):].copy()
    return train, val

def ensure_enough_data(df):
    return len(df) >= MIN_TRAIN_DAYS

def produce_forecasts_for_ticker(ticker):
    print(f"\n📊 Backtesting {ticker} ...")
    today = pd.Timestamp(datetime.now().date())
    month_start = start_of_current_month(today)
    # training window start = month_start - TRAIN_YEARS
    train_start = month_start - pd.DateOffset(years=TRAIN_YEARS)
    # we'll fetch historical up to end of validation month (so we can compare)
    month_end = (month_start + pd.offsets.MonthEnd(0))
    # fetch full range that includes training + validation:
    df_all = fetch_data(ticker, start_date=train_start.strftime("%Y-%m-%d"),
                        end_date=(month_end + pd.Timedelta(days=1)).strftime("%Y-%m-%d"))
    if df_all.empty:
        raise ValueError(f"No data for {ticker}")

    # keep only trading days and Close (target)
    df_all = df_all[["Open", "High", "Low", "Close"]].dropna()
    df_all.index = pd.to_datetime(df_all.index).tz_localize(None)

    # split: train up to last day before month_start, validation = month_start..month_end (trading days)
    train_end = month_start - pd.Timedelta(days=1)
    train_df = df_all.loc[:train_end].copy()
    val_df = df_all.loc[month_start:month_end].copy()
    if val_df.empty:
        print(f"ℹ️ No validation days for current month for {ticker}. Skipping.")
        return []

    if not ensure_enough_data(train_df):
        print(f"⚠️ Not enough training history for {ticker} (need ~{MIN_TRAIN_DAYS} rows). Skipping.")
        return []

    # Prepare series and lookback
    series = train_df["Close"].astype(float)
    lookback = auto_lookback_daily()  # uses your project's heuristic

    # scale full series (we will pass scaled train for models and then recursively forecast)
    scaled_all, scaler = scale_series(pd.concat([train_df["Close"], val_df["Close"]]))  # scale on combined to map outputs easily

    # training scaled portion is up to train_df length
    n_train = len(train_df)
    scaled_train = scaled_all[:n_train]

    # create supervised sets for tabular models
    X_seq, y_seq = make_sequences_univariate(scaled_train, lookback)
    X_tab, y_tab = make_supervised_from_scaled_univariate(scaled_train, lookback)
    (Xtr_seq, ytr_seq), (Xv_seq, yv_seq) = time_split(X_seq, y_seq, split_ratio=0.8)
    (Xtr_tab, ytr_tab), (Xv_tab, yv_tab) = time_split(X_tab, y_tab, split_ratio=0.8)

    # ----------
    # Train models (on scaled_train splits)
    # ----------
    models_preds_series = {}   # per-model multi-step forecast (in original price scale)
    models_rmse = {}

    # --- XGBoost ---
    try:
        mdl_xgb, params, rm_xgb = train_xgb_with_val(Xtr_tab, ytr_tab, Xv_tab, yv_tab, grid=None, random_state=42)
        models_rmse["xgboost"] = rm_xgb
        fut_xgb = predict_tabular_series(mdl_xgb, scaler, scaled_all, lookback, len(val_df))  # returns array of original-scale preds
        models_preds_series["xgboost"] = fut_xgb
    except Exception as e:
        print(f"⚠️ XGBoost failed for {ticker}: {e}")

    # --- Random Forest ---
    try:
        mdl_rf = train_rf(Xtr_tab, ytr_tab)
        yv_hat_rf = mdl_rf.predict(Xv_tab) if len(Xv_tab) > 0 else np.array([])
        models_rmse["random_forest"] = float(rmse(yv_tab, yv_hat_rf)) if len(yv_hat_rf)>0 else 0.05
        fut_rf = predict_tabular_series(mdl_rf, scaler, scaled_all, lookback, len(val_df))
        models_preds_series["random_forest"] = fut_rf
    except Exception as e:
        print(f"⚠️ RF failed for {ticker}: {e}")

    # --- Linear Regression ---
    try:
        mdl_lr = train_linreg(Xtr_tab, ytr_tab)
        yv_hat_lr = mdl_lr.predict(Xv_tab) if len(Xv_tab) > 0 else np.array([])
        models_rmse["linear_regression"] = float(rmse(yv_tab, yv_hat_lr)) if len(yv_hat_lr)>0 else 0.04
        fut_lr = predict_tabular_series(mdl_lr, scaler, scaled_all, lookback, len(val_df))
        models_preds_series["linear_regression"] = fut_lr
    except Exception as e:
        print(f"⚠️ LR failed for {ticker}: {e}")

    # --- LSTM ---
    try:
        mdl_lstm = build_lstm_univariate((lookback, 1), units=64, dropout=0.2)
        # Train with small epochs to be practical; tune later if needed
        train_model(mdl_lstm, Xtr_seq, ytr_seq, Xv_seq, yv_seq, epochs=30, batch_size=16)
        yv_hat_lstm = mdl_lstm.predict(Xv_seq, verbose=0).reshape(-1) if len(Xv_seq)>0 else np.array([])
        models_rmse["lstm"] = float(rmse(yv_seq, yv_hat_lstm)) if len(yv_hat_lstm)>0 else 0.06
        fut_lstm = predict_lstm_series(mdl_lstm, scaler, scaled_all, lookback, len(val_df))
        models_preds_series["lstm"] = fut_lstm
    except Exception as e:
        print(f"⚠️ LSTM failed for {ticker}: {e}")

    # ---------- Ensemble: compute weights from validation RMSEs ----------
    # If some models missing, only use available
    available_models = list(models_preds_series.keys())
    if not available_models:
        print(f"⚠️ No models produced forecasts for {ticker}")
        return []

    # build scores dict mapping model->rmse
    scores = {m: models_rmse.get(m, 0.05) for m in available_models}
    priorities = {m: 1 for m in available_models}  # equal priority; you can change per model
    w = weights_from_scores(scores, priorities)  # expects (scores, priorities)
    # ensemble series = weighted sum of per-model series
    ens_arr = None
    for m in available_models:
        arr = np.array(models_preds_series[m], dtype=float)
        if ens_arr is None:
            ens_arr = arr * w.get(m, 0.0)
        else:
            ens_arr = ens_arr + arr * w.get(m, 0.0)

    # ---------- Build output rows ----------
    out_rows = []
    val_dates = list(val_df.index)
    # actuals for validation month
    actuals = val_df["Close"].values

    for i, dt in enumerate(val_dates):
        actual = float(actuals[i])
        for m in available_models:
            pred = float(models_preds_series[m][i])
            pct_err = (pred - actual) / actual * 100.0
            out_rows.append({
                "Date": dt.strftime("%Y-%m-%d"),
                "Ticker": ticker,
                "Model": m,
                "Predicted": pred,
                "Actual": actual,
                "Pct_Error": pct_err
            })
        # ensemble row
        ens_pred = float(ens_arr[i]) if ens_arr is not None else None
        pct_err_e = (ens_pred - actual) / actual * 100.0 if ens_pred is not None else None
        out_rows.append({
            "Date": dt.strftime("%Y-%m-%d"),
            "Ticker": ticker,
            "Model": "ensemble",
            "Predicted": ens_pred,
            "Actual": actual,
            "Pct_Error": pct_err_e
        })

    # return list of rows for this ticker
    return out_rows

def run_backtest_all():
    stocks = load_stocks()
    all_rows = []
    for t in stocks:
        try:
            rows = produce_forecasts_for_ticker(t)
            all_rows.extend(rows)
        except Exception as e:
            print(f"⚠️ Skipped {t}: {e}")

    if not all_rows:
        print("No results produced.")
        return

    df_out = pd.DataFrame(all_rows)
    # sort and save
    df_out = df_out.sort_values(["Date", "Ticker", "Model"])
    df_out.to_excel(OUT_FILE, index=False)
    print(f"\n✅ Monthly backtest results saved to: {OUT_FILE}")
    # also print a quick summary
    summary = df_out.groupby("Model")["Pct_Error"].agg(["mean", "std", "count"]).reset_index()
    print("\nModel summary (mean pct error, std, count):")
    print(summary.to_string(index=False))

if __name__ == "__main__":
    run_backtest_all()
