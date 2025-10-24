import matplotlib.pyplot as plt
import numpy as np

def plot_actual_vs_predicted(df, price_type, scaler, model, lookback):
    scaled = scaler.transform(df[[price_type]].values)
    X, y = [], []
    for i in range(lookback, len(scaled)):
        X.append(scaled[i-lookback:i, 0])
        y.append(scaled[i, 0])
    X, y = np.array(X), np.array(y)
    preds = model.predict(X, verbose=0)
    preds_inv = scaler.inverse_transform(preds)
    y_inv = scaler.inverse_transform(y.reshape(-1, 1))

    plt.figure(figsize=(10, 5))
    plt.plot(y_inv, label="Actual", color="black")
    plt.plot(preds_inv, label="Predicted", color="orange", linestyle="--")
    plt.title(f"Actual vs Predicted {price_type}")
    plt.legend()
    plt.show()
