"""
preprocessor.py — Feature engineering, scaling, and dataset preparation.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import ta

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add technical analysis features to the OHLCV DataFrame.

    Added columns:
      SMA_20, SMA_50, EMA_20, RSI, MACD, MACD_Signal,
      Daily_Return, Volatility
    """
    df = df.copy()

    # Simple Moving Averages
    for window in config.SMA_WINDOWS:
        df[f"SMA_{window}"] = ta.trend.sma_indicator(df["Close"], window=window)

    # Exponential Moving Average
    df[f"EMA_{config.EMA_WINDOW}"] = ta.trend.ema_indicator(
        df["Close"], window=config.EMA_WINDOW
    )

    # Relative Strength Index
    df["RSI"] = ta.momentum.rsi(df["Close"], window=config.RSI_WINDOW)

    # MACD
    macd = ta.trend.MACD(df["Close"])
    df["MACD"] = macd.macd()
    df["MACD_Signal"] = macd.macd_signal()

    # Daily return (percentage)
    df["Daily_Return"] = df["Close"].pct_change() * 100

    # Rolling volatility (std of daily returns)
    df["Volatility"] = df["Daily_Return"].rolling(
        window=config.VOLATILITY_WINDOW
    ).std()

    # Drop rows with NaN created by rolling calculations
    df.dropna(inplace=True)

    return df


def scale_data(
    df: pd.DataFrame,
    feature_columns: list[str] | None = None,
    target_column: str | None = None,
):
    """
    Scale features and target with MinMaxScaler.

    Returns
    -------
    X_scaled : np.ndarray  — scaled feature matrix
    y_scaled : np.ndarray  — scaled target vector
    feature_scaler : MinMaxScaler
    target_scaler  : MinMaxScaler
    """
    if feature_columns is None:
        feature_columns = config.FEATURE_COLUMNS
    if target_column is None:
        target_column = config.TARGET_COLUMN

    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()

    X_scaled = feature_scaler.fit_transform(df[feature_columns].values)
    y_scaled = target_scaler.fit_transform(df[[target_column]].values)

    return X_scaled, y_scaled, feature_scaler, target_scaler


def create_sequences(X: np.ndarray, y: np.ndarray, seq_length: int):
    """
    Build sliding-window sequences for LSTM.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
    y : np.ndarray of shape (n_samples, 1)
    seq_length : int — number of past time-steps per sample

    Returns
    -------
    X_seq : np.ndarray of shape (n_samples - seq_length, seq_length, n_features)
    y_seq : np.ndarray of shape (n_samples - seq_length,)
    """
    X_seq, y_seq = [], []
    for i in range(seq_length, len(X)):
        X_seq.append(X[i - seq_length : i])
        y_seq.append(y[i, 0])
    return np.array(X_seq), np.array(y_seq)


def train_test_split_ts(
    X: np.ndarray,
    y: np.ndarray,
    train_ratio: float | None = None,
):
    """
    Time-series aware train/test split (no shuffling).
    """
    if train_ratio is None:
        train_ratio = config.TRAIN_TEST_SPLIT

    split = int(len(X) * train_ratio)
    return X[:split], X[split:], y[:split], y[split:]


def prepare_data(df: pd.DataFrame, for_lstm: bool = False):
    """
    End-to-end data preparation pipeline.

    Returns
    -------
    dict with keys:
        X_train, X_test, y_train, y_test,
        feature_scaler, target_scaler,
        dates_test (DatetimeIndex of the test portion)
    """
    df_feat = add_technical_indicators(df)

    X_scaled, y_scaled, feat_scaler, tgt_scaler = scale_data(df_feat)

    if for_lstm:
        seq_len = config.LSTM_SEQUENCE_LENGTH
        X_seq, y_seq = create_sequences(X_scaled, y_scaled, seq_len)
        X_train, X_test, y_train, y_test = train_test_split_ts(X_seq, y_seq)
        # Adjust date index for sequences
        dates = df_feat.index[seq_len:]
        split = int(len(X_seq) * config.TRAIN_TEST_SPLIT)
        dates_test = dates[split:]
    else:
        X_train, X_test, y_train, y_test = train_test_split_ts(
            X_scaled, y_scaled.ravel()
        )
        split = int(len(X_scaled) * config.TRAIN_TEST_SPLIT)
        dates_test = df_feat.index[split:]

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "feature_scaler": feat_scaler,
        "target_scaler": tgt_scaler,
        "dates_test": dates_test,
        "df_featured": df_feat,
    }
