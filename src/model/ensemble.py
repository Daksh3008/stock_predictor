import numpy as np

def rmse(y_true, y_pred):
    y_true = np.array(y_true).reshape(-1)
    y_pred = np.array(y_pred).reshape(-1)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def weights_from_scores(scores: dict, priorities: dict):
    skill = {}
    for m, rmse_val in scores.items():
        if np.isfinite(rmse_val) and rmse_val > 0:
            skill[m] = 1.0 / rmse_val
        else:
            skill[m] = 0.0
    adj = {}
    for m, sc in skill.items():
        p = max(1, int(priorities.get(m, 3)))
        adj[m] = sc * (1.0 / p)
    total = sum(adj.values())
    if total == 0:
        n = len(adj); return {m: 1.0 / n for m in adj}
    return {m: v / total for m, v in adj.items()}

def weighted_average(preds: dict, weights: dict):
    return float(sum(preds[m] * weights.get(m, 0.0) for m in preds))
