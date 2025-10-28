from itertools import product
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

DEFAULT_XGB_GRID = {
    "n_estimators": [300, 500],
    "learning_rate": [0.03, 0.06],
    "max_depth": [3, 5],
    "subsample": [0.8, 1.0],
}

def train_xgb_with_val(X_train, y_train, X_val, y_val, grid=None, random_state=42):
    if grid is None:
        grid = DEFAULT_XGB_GRID
    best_rmse, best_params, best_model = float("inf"), None, None
    for combo in product(*grid.values()):
        params = dict(zip(grid.keys(), combo))
        model = XGBRegressor(
            objective="reg:squarederror",
            random_state=random_state,
            n_jobs=-1,
            **params
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        mse = mean_squared_error(y_val, preds)
        rmse = float(np.sqrt(mse))
        if rmse < best_rmse:
            best_rmse, best_params, best_model = rmse, params, model
    return best_model, best_params, best_rmse
