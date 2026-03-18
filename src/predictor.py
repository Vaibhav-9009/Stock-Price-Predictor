"""
predictor.py — Train models, generate predictions, and compute evaluation metrics.
"""

import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from src.models import build_lstm, build_random_forest, build_linear_regression


# ── Metrics ──────────────────────────────────────────────────────────

def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Percentage Error (%)."""
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Return MAE, RMSE, R², MAPE as a dict."""
    return {
        "MAE": round(float(mean_absolute_error(y_true, y_pred)), 4),
        "RMSE": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 4),
        "R²": round(float(r2_score(y_true, y_pred)), 4),
        "MAPE (%)": round(_mape(y_true, y_pred), 2),
    }


# ── Training & Evaluation ───────────────────────────────────────────

def train_and_predict(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    target_scaler=None,
    progress_callback=None,
):
    """
    Train a model and return predictions + metrics.

    Parameters
    ----------
    model_name : one of config.AVAILABLE_MODELS
    target_scaler : MinMaxScaler to inverse-transform predictions
    progress_callback : optional callable(epoch, total_epochs) for UI

    Returns
    -------
    dict with keys: predictions, y_true, metrics, model
    """
    if model_name == config.MODEL_LSTM:
        model = build_lstm(input_shape=(X_train.shape[1], X_train.shape[2]))
        model.fit(X_train, y_train, progress_callback=progress_callback)
        y_pred_scaled = model.predict(X_test)

    elif model_name == config.MODEL_RF:
        model = build_random_forest()
        model.fit(X_train, y_train)
        y_pred_scaled = model.predict(X_test)

    elif model_name == config.MODEL_LR:
        model = build_linear_regression()
        model.fit(X_train, y_train)
        y_pred_scaled = model.predict(X_test)

    else:
        raise ValueError(f"Unknown model: {model_name}")

    # Inverse-transform to original price scale
    if target_scaler is not None:
        y_pred = target_scaler.inverse_transform(
            y_pred_scaled.reshape(-1, 1)
        ).flatten()
        y_true = target_scaler.inverse_transform(
            y_test.reshape(-1, 1)
        ).flatten()
    else:
        y_pred = y_pred_scaled
        y_true = y_test

    metrics = compute_metrics(y_true, y_pred)

    return {
        "predictions": y_pred,
        "y_true": y_true,
        "metrics": metrics,
        "model": model,
    }


def forecast_future(
    model,
    model_name: str,
    last_sequence: np.ndarray,
    feature_scaler,
    target_scaler,
    days: int | None = None,
):
    """
    Generate future price forecasts.

    Parameters
    ----------
    model        : trained model instance
    model_name   : name of the model
    last_sequence: the most recent data window (scaled)
    days         : number of days to forecast

    Returns
    -------
    np.ndarray of shape (days,) — predicted prices in original scale
    """
    if days is None:
        days = config.FORECAST_DAYS

    if model_name == config.MODEL_LSTM:
        predictions = []
        current_seq = last_sequence.copy()

        for _ in range(days):
            pred = model.predict(
                current_seq.reshape(1, current_seq.shape[0], current_seq.shape[1]),
            )[0]
            predictions.append(pred)

            # Slide window: drop first row, append new row with pred
            new_row = current_seq[-1].copy()
            # Update the Close column (index 3) with the prediction
            new_row[3] = pred
            current_seq = np.vstack([current_seq[1:], new_row])

        predictions = np.array(predictions)
        return target_scaler.inverse_transform(
            predictions.reshape(-1, 1)
        ).flatten()

    else:
        # For sklearn models, use the last feature vector and iterate
        last_features = last_sequence[-1].reshape(1, -1) if last_sequence.ndim > 1 else last_sequence.reshape(1, -1)
        predictions = []
        current_features = last_features.copy()

        for _ in range(days):
            pred = model.predict(current_features)[0]
            predictions.append(pred)
            # Update Close feature for next step
            new_feat = current_features.copy()
            new_feat[0, 3] = pred
            current_features = new_feat

        predictions = np.array(predictions)
        return target_scaler.inverse_transform(
            predictions.reshape(-1, 1)
        ).flatten()
