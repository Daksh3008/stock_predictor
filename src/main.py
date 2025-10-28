import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np

from src.utils.config import load_cfg
from src.utils.nlp import parse_user_query
from src.utils.formatter import pretty_card
from src.utils.dates import (
    end_of_next_fiscal_quarter, end_of_next_month,
    end_of_next_year, end_of_next_week, parse_explicit_days
)
from src.data.fetch_data import fetch_data
from src.data.preprocess import (
    auto_lookback_daily, scale_series,
    make_sequences_univariate, make_supervised_from_scaled_univariate,
    to_prophet_frame, time_split
)
from src.model.train import train_model
from src.model.lstm_model import build_lstm_univariate
from src.model.xgb_model import train_xgb_with_val
from src.model.rf_model import train_rf
from src.model.linreg_model import train_linreg
from src.model.prophet_model import train_prophet, prophet_forecast_values
from src.model.predict import predict_lstm_series, predict_tabular_series
from src.model.ensemble import rmse, weights_from_scores, weighted_average

def resolve_target_date(text: str, last_date: pd.Timestamp) -> tuple[int, str]:
    """Return (steps, label_for_output). Calendar-based, Indian fiscal quarters."""
    txt = text.lower()
    # explicit "next N days"
    days = parse_explicit_days(txt)
    if days is not None:
        return max(1, days), f"next_{days}_days"

    if "quarter" in txt:
        td = end_of_next_fiscal_quarter(last_date); lbl = "next_quarter"
    elif "year" in txt:
        td = end_of_next_year(last_date); lbl = "next_year"
    elif "month" in txt:
        td = end_of_next_month(last_date); lbl = "next_month"
    elif "week" in txt:
        td = end_of_next_week(last_date); lbl = "next_week"
    else:
        td = last_date + pd.Timedelta(days=30); lbl = "next_30_days"
    steps = max(1, (td - last_date).days)
    return steps, lbl

def main():
    cfg = load_cfg()

    user_text = input("Enter request (e.g., 'Predict AAPL highest price next quarter'): ").strip()
    parsed = parse_user_query(user_text)
    ticker = parsed["ticker"]
    if not ticker:
        raise ValueError("Could not detect ticker. Please include it (e.g., AAPL, RELIANCE.NS).")
    target_col = parsed["target_col"]   # Open/High/Low/Close
    aggregation = parsed["aggregation"] # 'last' | 'max' | 'min'

    # fetch daily OHLC
    df = fetch_data(ticker, start_date=cfg["prediction"]["start_date"])
    datapoints = len(df)
    if datapoints < cfg["prediction"]["min_data_points"]:
        raise ValueError(f"Not enough data ({datapoints}) for reliable forecast.")

    last_date = df.index[-1]
    steps, horizon_label = resolve_target_date(user_text, last_date)

    # target series
    series = df[target_col]
    lookback = auto_lookback_daily()
    scaled, scaler = scale_series(series)

    # build datasets
    X_seq, y_seq = make_sequences_univariate(scaled, lookback)
    X_tab, y_tab = make_supervised_from_scaled_univariate(scaled, lookback)
    (Xtr_seq, ytr_seq), (Xv_seq, yv_seq) = time_split(X_seq, y_seq, split_ratio=0.8)
    (Xtr_tab, ytr_tab), (Xv_tab, yv_tab) = time_split(X_tab, y_tab, split_ratio=0.8)

    models_cfg = cfg["models"]
    preds_scalar = {}     # one scalar per model after aggregation
    preds_series = {}     # future daily series per model (for debug & max/min)
    scores = {}           # RMSE on validation
    priorities = {m: models_cfg[m]["priority"] for m in models_cfg if "priority" in models_cfg[m]}

    # Prophet
    if models_cfg["prophet"]["enabled"]:
        dfp = to_prophet_frame(df, target_col).dropna()

        # Ensure continuous numeric column
        dfp["y"] = pd.to_numeric(dfp["y"], errors="coerce")
        dfp = dfp.dropna().reset_index(drop=True)

        if len(dfp) < 30:  # Prophet minimum stability
            print("Prophet skipped: insufficient data for seasonal modeling")
        else:
            cut = max(1, int(len(dfp) * 0.8))
            train_df = dfp.iloc[:cut]
            val_df = dfp.iloc[cut:]

            mdl_p = train_prophet(train_df)

            if len(val_df) > 0:
                val_steps = len(val_df)
                val_forecast = prophet_forecast_values(mdl_p, val_steps)
                yv_true = val_df["y"].values
                yv_hat = val_forecast
                scores["prophet"] = rmse(yv_true, yv_hat)
            else:
            # fallback if validation too short
                scores["prophet"] = 0.05

                fut = prophet_forecast_values(mdl_p, steps)
                preds_series["prophet"] = fut


    # XGBoost
    if models_cfg["xgboost"]["enabled"]:
        mdl_xgb, params, rm = train_xgb_with_val(Xtr_tab, ytr_tab, Xv_tab, yv_tab, grid=None, random_state=42)
        scores["xgboost"] = rm
        fut = predict_tabular_series(mdl_xgb, scaler, scaled, lookback, steps)
        preds_series["xgboost"] = fut

    # Random Forest
    if models_cfg["random_forest"]["enabled"]:
        mdl_rf = train_rf(Xtr_tab, ytr_tab)
        yv_hat = mdl_rf.predict(Xv_tab); scores["random_forest"] = rmse(yv_tab, yv_hat)
        fut = predict_tabular_series(mdl_rf, scaler, scaled, lookback, steps)
        preds_series["random_forest"] = fut

    # Linear Regression
    if models_cfg["linear_regression"]["enabled"]:
        mdl_lr = train_linreg(Xtr_tab, ytr_tab)
        yv_hat = mdl_lr.predict(Xv_tab); scores["linear_regression"] = rmse(yv_tab, yv_hat)
        fut = predict_tabular_series(mdl_lr, scaler, scaled, lookback, steps)
        preds_series["linear_regression"] = fut

    # LSTM
    if models_cfg["lstm"]["enabled"]:
        mdl_lstm = build_lstm_univariate((lookback, 1), units=64, dropout=0.2)
        train_model(mdl_lstm, Xtr_seq, ytr_seq, Xv_seq, yv_seq, epochs=100, batch_size=16)
        yv_hat = mdl_lstm.predict(Xv_seq, verbose=0).reshape(-1)
        scores["lstm"] = rmse(yv_seq, yv_hat)
        fut = predict_lstm_series(mdl_lstm, scaler, scaled, lookback, steps)
        preds_series["lstm"] = fut

    # aggregate per model based on user intent
    def aggregate(series_vals: np.ndarray) -> float:
        if aggregation == "max":
            return float(np.max(series_vals))
        if aggregation == "min":
            return float(np.min(series_vals))
        return float(series_vals[-1])  # 'last'

    for m, s in preds_series.items():
        preds_scalar[m] = aggregate(s)

    # weights & ensemble
    w = weights_from_scores(scores, priorities)
    ensemble_pred = weighted_average(preds_scalar, w)

    # CI heuristic: best model residual std ~ rmse, plus disagreement
    best_model = min(scores, key=scores.get)
    best_rmse = scores[best_model]
    model_spread = np.std(list(preds_scalar.values())) if len(preds_scalar) > 1 else 0.0
    sigma = float(np.sqrt(best_rmse**2 + model_spread**2))
    ci_low, ci_high = ensemble_pred - 1.96 * sigma, ensemble_pred + 1.96 * sigma

    curr_price = float(df[target_col].iloc[-1])
    pct_change = (ensemble_pred - curr_price) / curr_price
    confidence = "High" if best_rmse < 0.02 else ("Medium" if best_rmse < 0.05 else "Low")

    # -----------------------
    # --- DEBUG MODE START ---
    # Remove this entire block after you are satisfied with outputs
    print("\n===== DEBUG: MODEL VALIDATION RMSE =====")
    for k, v in scores.items():
        print(f"{k:16s} : {v:.6f}")
    print("\n===== DEBUG: MODEL WEIGHTS (normalized) =====")
    for k, v in w.items():
        print(f"{k:16s} : {v:.4f}")
    print("\n===== DEBUG: PER-MODEL SCALAR PREDICTIONS =====")
    for k, v in preds_scalar.items():
        print(f"{k:16s} : {v:,.4f}")
    print("\n===== DEBUG: ENSEMBLE =====")
    print(f"ensemble_pred: {ensemble_pred:,.4f}, sigma: {sigma:.4f}, CI: [{ci_low:,.4f}, {ci_high:,.4f}]")
    # --- DEBUG MODE END ---
    # -----------------------

    print("\n" + pretty_card(
        ticker=ticker,
        curr_price=curr_price,
        pred_price=ensemble_pred,
        pct_change=pct_change,
        ci_low=ci_low, ci_high=ci_high,
        models_used=len(preds_scalar),
        best_model_name=best_model.capitalize(),
        best_score=1 / (1 + best_rmse),
        confidence=confidence,
        datapoints=datapoints,
        horizon_label=horizon_label
    ))

if __name__ == "__main__":
    main()
