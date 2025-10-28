import numpy as np

def predict_lstm_series(model, scaler, scaled_series, lookback, steps):
    X = scaled_series[-lookback:].reshape(1, lookback, 1)
    preds_scaled = []
    for _ in range(steps):
        p = model.predict(X, verbose=0)
        preds_scaled.append(p[0, 0])
        X = np.append(X[:, 1:, :], p.reshape(1, 1, 1), axis=1)
    preds_scaled = np.array(preds_scaled).reshape(-1, 1)
    return scaler.inverse_transform(preds_scaled).reshape(-1)

def predict_tabular_series(model, scaler, scaled_series, lookback, steps):
    window = scaled_series[-lookback:, 0].copy()
    preds = []
    for _ in range(steps):
        x = window.reshape(1, -1)
        p_scaled = model.predict(x)[0]
        preds.append(p_scaled)
        window = np.append(window[1:], p_scaled)
    preds = np.array(preds).reshape(-1, 1)
    return scaler.inverse_transform(preds).reshape(-1)
