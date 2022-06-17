from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from lightgbm.sklearn import LGBMRegressor


def valid_lightgbm_model(X, y, params=False):
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.3, random_state=0xC0FFEE)
    if not params:
        model = LGBMRegressor()
    else:
        model = LGBMRegressor(**params)
    model.fit(X_train, y_train)
    pred = model.predict(X_valid)
    mae = mean_absolute_error(y_valid, pred)
    print(f"MAE : {mae:.3}")
    return model


def train_lightgbm_model(X, y, params=False):
    if not params:
        model = LGBMRegressor()
    else:
        model = LGBMRegressor(**params)
    model.fit(X, y)
    return model
