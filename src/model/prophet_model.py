from prophet import Prophet
import pandas as pd

def train_prophet(df_prophet_train: pd.DataFrame):
    m = Prophet(
        daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=True
    )
    m.fit(df_prophet_train)
    return m

def prophet_forecast_values(model, steps: int):
    # daily steps ahead
    future = model.make_future_dataframe(periods=steps, freq="D")
    fc = model.predict(future)
    # return only the new tail (future part)
    return fc["yhat"].tail(steps).values
