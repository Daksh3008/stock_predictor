import numpy as np
from sklearn.preprocessing import MinMaxScaler

def auto_lookback(freq: str) -> int:
    mapping = {
        "daily": 60,
        "weekly": 26,
        "monthly": 24,
        "quarterly": 12,
        "yearly": 5
    }
    return mapping.get(freq.lower(), 10)

def scale_series(series, scaler=None):
    arr = series.values.reshape(-1, 1)
    if scaler is None:
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler((0, 1))
        scaled = scaler.fit_transform(arr)
    else:
        scaled = scaler.transform(arr)
    return scaled, scaler

def make_sequences(data, lookback):
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i-lookback:i, 0])
        y.append(data[i, 0])
    X, y = np.array(X), np.array(y)
    return X.reshape(X.shape[0], X.shape[1], 1), y
