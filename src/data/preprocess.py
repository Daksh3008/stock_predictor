import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

def auto_lookback_daily() -> int:
    return 60  # good default; adjust if needed

def scale_series(series, scaler=None):
    arr = series.values.reshape(-1, 1)
    if scaler is None:
        scaler = MinMaxScaler((0, 1))
        scaled = scaler.fit_transform(arr)
    else:
        scaled = scaler.transform(arr)
    return scaled, scaler

def make_sequences_univariate(data, lookback):
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i - lookback:i, 0])
        y.append(data[i, 0])
    X, y = np.array(X), np.array(y)
    # LSTM expects (samples, timesteps, features=1)
    return X.reshape(X.shape[0], X.shape[1], 1), y

def make_supervised_from_scaled_univariate(data, lookback):
    # For tree/linear models: features = last 'lookback' target values
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i - lookback:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

def to_prophet_frame(df: pd.DataFrame, col: str) -> pd.DataFrame:
    out = df[[col]].copy()
    out = out.rename(columns={col: "y"})
    out["ds"] = out.index
    return out[["ds", "y"]]

def time_split(*arrays, split_ratio=0.8):
    n = len(arrays[0])
    cut = int(n * split_ratio)
    return tuple(a[:cut] for a in arrays), tuple(a[cut:] for a in arrays)
