import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import joblib


def evaluate(y_true, y_pred):
    return {
        "R2": r2_score(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAE": mean_absolute_error(y_true, y_pred)
    }


def save_model(model, path):
    joblib.dump(model, path)
