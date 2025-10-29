# src/main.py
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from datetime import datetime

from src.utils.config import load_cfg
from src.utils.search import search_top5           
from src.utils.nlp import parse_user_request       
from src.utils.formatter import pretty_card
from src.utils.dates import (
    resolve_steps_from_flags, market_closed_today_message, 
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
from src.model.prophet_model import train_neuralprophet, neuralprophet_forecast
from src.model.predict import predict_lstm_series, predict_tabular_series
from src.model.ensemble import rmse, weights_from_scores, weighted_average

def main():
    cfg = load_cfg()

    #INPUTS
    stock_query = input("Enter stock name or ticker: ").strip()
    # Auto-search top 5
    hits = search_top5(stock_query)
    ticker = None
    if hits:
        print("\nWe found these matches — select one:")
        for i, h in enumerate(hits, 1):
            print(f"{i}) {h.symbol} — {h.name} [{h.exchange}]")
        try:
            choice = int(input("Enter choice (1-5), or 0 to type ticker manually: ").strip())
        except Exception:
            choice = 0
        if 1 <= choice <= len(hits):
            ticker = hits[choice - 1].symbol
        else:
            ticker = input("Enter ticker (e.g., TCS.NS, AAPL): ").strip()
    else:
        ticker = input("No matches found. Enter ticker (e.g., TCS.NS, AAPL): ").strip()

    user_req = input("Enter what to predict (e.g., 'closing price next quarter'): ").strip()
    parsed = parse_user_request(user_req)
    target_col = parsed["target_col"]
    aggregation = parsed["aggregation"]

    #DATA FETCHING
    df = fetch_data(ticker, start_date=cfg["prediction"]["start_date"])
    datapoints = len(df)
    if datapoints < cfg["prediction"]["min_data_points"]:
        raise ValueError(f"Not enough data ({datapoints}) for reliable forecast.")

    last_date = df.index[-1]
    today_ts = pd.Timestamp(datetime.now().date())  # local day

    # show df.tail() for debugging
    #print("\n" + "📄 Latest Data Used (df.tail())".ljust(40, "-"))
    #print(df.tail().to_string())  # highlight tail

    # HORIZONS (today, next day, user-requested)
    steps_user, horizon_label = resolve_steps_from_flags(parsed["horizon_flags"], last_date)
    # We will produce a single future path with length = max(steps)
    max_steps = max(steps_user, 2)  # we need at least 2 for today & next day slicing


    # PREPARE TARGET SERIES
    series = df[target_col].astype(float)  # ensures Series + numeric
    lookback = auto_lookback_daily()
    scaled, scaler = scale_series(series)

    # Datasets
    X_seq, y_seq = make_sequences_univariate(scaled, lookback)
    X_tab, y_tab = make_supervised_from_scaled_univariate(scaled, lookback)
    (Xtr_seq, ytr_seq), (Xv_seq, yv_seq) = time_split(X_seq, y_seq, split_ratio=0.8)
    (Xtr_tab, ytr_tab), (Xv_tab, yv_tab) = time_split(X_tab, y_tab, split_ratio=0.8)

    models_cfg = cfg["models"]
    preds_scalar = {}
    preds_series = {}
    scores = {}
    priorities = {m: models_cfg[m]["priority"] for m in models_cfg if "priority" in models_cfg[m]}


    # MODELS

    # NeuralProphet
    if models_cfg["prophet"]["enabled"]:
        dfp = series.to_frame(name="y").copy()
        dfp["ds"] = dfp.index
        dfp = dfp[["ds", "y"]].dropna()
        # flatten guard (if ever MultiIndex)
        if isinstance(dfp.columns, pd.MultiIndex):
            dfp.columns = [c[0] for c in dfp.columns]
        dfp["y"] = pd.to_numeric(dfp["y"], errors="coerce")
        dfp.dropna(inplace=True)
        dfp.reset_index(drop=True, inplace=True)

        if len(dfp) >= 60:
            mdl_np = train_neuralprophet(dfp)
            fut = neuralprophet_forecast(mdl_np, max_steps)
            preds_series["prophet"] = fut
            scores["prophet"] = 0.04
        else:
            print("NeuralProphet skipped: insufficient data")
            scores["prophet"] = 0.05

    # XGBoost
    if models_cfg["xgboost"]["enabled"]:
        mdl_xgb, params, rm = train_xgb_with_val(Xtr_tab, ytr_tab, Xv_tab, yv_tab, grid=None, random_state=42)
        scores["xgboost"] = rm
        fut = predict_tabular_series(mdl_xgb, scaler, scaled, lookback, max_steps)
        preds_series["xgboost"] = fut

    # Random Forest
    if models_cfg["random_forest"]["enabled"]:
        mdl_rf = train_rf(Xtr_tab, ytr_tab)
        yv_hat = mdl_rf.predict(Xv_tab)
        scores["random_forest"] = rmse(yv_tab, yv_hat)
        fut = predict_tabular_series(mdl_rf, scaler, scaled, lookback, max_steps)
        preds_series["random_forest"] = fut

    # Linear Regression
    if models_cfg["linear_regression"]["enabled"]:
        mdl_lr = train_linreg(Xtr_tab, ytr_tab)
        yv_hat = mdl_lr.predict(Xv_tab)
        scores["linear_regression"] = rmse(yv_tab, yv_hat)
        fut = predict_tabular_series(mdl_lr, scaler, scaled, lookback, max_steps)
        preds_series["linear_regression"] = fut

    # LSTM
    if models_cfg["lstm"]["enabled"]:
        mdl_lstm = build_lstm_univariate((lookback, 1), units=64, dropout=0.2)
        train_model(mdl_lstm, Xtr_seq, ytr_seq, Xv_seq, yv_seq, epochs=100, batch_size=16)
        yv_hat = mdl_lstm.predict(Xv_seq, verbose=0).reshape(-1)
        scores["lstm"] = rmse(yv_seq, yv_hat)
        fut = predict_lstm_series(mdl_lstm, scaler, scaled, lookback, max_steps)
        preds_series["lstm"] = fut


    # AGGREGATION

    def aggregate(series_vals: np.ndarray, how: str) -> float:
        if how == "max":
            return float(np.max(series_vals))
        if how == "min":
            return float(np.min(series_vals))
        return float(series_vals[-1])  # 'last'

    # per-model scalar for user horizon
    for m, s in preds_series.items():
        preds_scalar[m] = aggregate(s[:steps_user], aggregation)

    # weights & ensemble
    w = weights_from_scores(scores, priorities)
    ensemble_pred = weighted_average(preds_scalar, w)

    # CI heuristic
    best_model = min(scores, key=scores.get)
    best_rmse = scores[best_model]
    model_spread = np.std(list(preds_scalar.values())) if len(preds_scalar) > 1 else 0.0
    sigma = float(np.sqrt(best_rmse**2 + model_spread**2))
    ci_low, ci_high = ensemble_pred - 1.96 * sigma, ensemble_pred + 1.96 * sigma

    # Immediate forecasts (today & next day) from ensemble path
    # Build ensemble future path by weighted sum of per-model series
    ens_series = None
    for m, s in preds_series.items():
        s = np.array(s, dtype=float)
        ens_series = s * w.get(m, 0.0) if ens_series is None else ens_series + s * w.get(m, 0.0)

    today_pred = float(ens_series[0]) if ens_series is not None and len(ens_series) >= 1 else None
    next_day_pred = float(ens_series[1]) if ens_series is not None and len(ens_series) >= 2 else None

    curr_price = float(df[target_col].iloc[-1])  # CP1: last close
    pct_change = (ensemble_pred - curr_price) / curr_price
    confidence = "High" if best_rmse < 0.02 else ("Medium" if best_rmse < 0.05 else "Low")

    market_msg = market_closed_today_message(df.index, pd.Timestamp.today())  # <<< NEW

    # -----------------------
    # --- DEBUG MODE START ---
    #print("\n===== DEBUG: MODEL VALIDATION RMSE =====")
    #for k, v in scores.items():
    #    print(f"{k:16s} : {v:.6f}")
    #print("\n===== DEBUG: MODEL WEIGHTS (normalized) =====")
    #for k, v in w.items():
    #    print(f"{k:16s} : {v:.4f}")
    #print("\n===== DEBUG: PER-MODEL SCALAR PREDICTIONS (user horizon) =====")
    #for k, v in preds_scalar.items():
    #    print(f"{k:16s} : {v:,.4f}")
    #print("\n===== DEBUG: ENSEMBLE =====")
    #print(f"ensemble_pred: {ensemble_pred:,.4f}, sigma: {sigma:.4f}, CI: [{ci_low:,.4f}, {ci_high:,.4f}]")
    # --- DEBUG MODE END ---
   

    # Final pretty output (includes current price, today & next day)
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
        horizon_label=horizon_label,
        today_pred=today_pred,                # <<< NEW
        next_day_pred=next_day_pred,          # <<< NEW
        market_msg=market_msg                 # <<< NEW
    ))

if __name__ == "__main__":
    main()
