from neuralprophet import NeuralProphet
import pandas as pd

def train_neuralprophet(df_prophet_train: pd.DataFrame):
    m = NeuralProphet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        epochs=50,
        learning_rate=1.0,
    )
    m.fit(df_prophet_train, freq="D", minimal=True)

    # ✅ Store training data inside model for future prediction calls
    m.input_df = df_prophet_train.copy()

    return m

def neuralprophet_forecast(model, steps: int):
    # ✅ Pass input df explicitly
    future = model.make_future_dataframe(
        model.input_df,
        periods=steps,
        n_historic_predictions=False
    )
    forecast = model.predict(future)
    return forecast["yhat1"].tail(steps).values
