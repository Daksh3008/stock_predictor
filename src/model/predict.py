import numpy as np
import pandas as pd

def predict_for_date(model, scaler, scaled_data, lookback, df, target_date, freq):
    target_date = pd.to_datetime(target_date)
    last_date = df.index[-1]
    freq_map = {
        "daily": "D",
        "weekly": "W-FRI",
        "monthly": "M",
        "quarterly": "Q",
        "yearly": "A"
    }
    rule = freq_map.get(freq.lower())
    steps = len(pd.date_range(start=last_date, end=target_date, freq=rule)) - 1

    if steps < 1:
        raise ValueError("Target date must be after last available date.")

    X_input = scaled_data[-lookback:].reshape(1, lookback, 1)
    for _ in range(steps):
        pred = model.predict(X_input, verbose=0)
        X_input = np.append(X_input[:, 1:, :], pred.reshape(1, 1, 1), axis=1)

    return scaler.inverse_transform(pred)[0, 0]
