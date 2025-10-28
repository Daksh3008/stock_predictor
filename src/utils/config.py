import os, yaml

_DEFAULT = {
    "models": {
        "lstm": {"enabled": True, "priority": 1},
        "prophet": {"enabled": True, "priority": 1},
        "xgboost": {"enabled": True, "priority": 2},
        "random_forest": {"enabled": True, "priority": 3},
        "linear_regression": {"enabled": True, "priority": 4},
    },
    "prediction": {
        "ensemble_method": "weighted_average",
        "min_data_points": 30,
        "start_date": "2010-01-01",
    },
}

def load_cfg(path="config/config.yaml"):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            user = yaml.safe_load(f) or {}
        cfg = _DEFAULT.copy()
        for k, v in user.items():
            if isinstance(v, dict) and k in cfg:
                cfg[k].update(v)
            else:
                cfg[k] = v
        return cfg
    return _DEFAULT
