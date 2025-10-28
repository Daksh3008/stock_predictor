from sklearn.ensemble import RandomForestRegressor

def train_rf(X_train, y_train, n_estimators=400, max_depth=None, random_state=42):
    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        n_jobs=-1,
        random_state=random_state
    )
    rf.fit(X_train, y_train)
    return rf
