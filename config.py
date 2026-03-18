"""
Configuration and constants for Stock Price Prediction project.
"""

# ── Default Parameters ──────────────────────────────────────────────
DEFAULT_TICKER = "RELIANCE.NS"
DEFAULT_START_DATE = "2020-01-01"
DEFAULT_END_DATE = "2025-12-31"

# ── Feature Engineering ─────────────────────────────────────────────
SMA_WINDOWS = [20, 50]
EMA_WINDOW = 20
RSI_WINDOW = 14
VOLATILITY_WINDOW = 20

FEATURE_COLUMNS = [
    "Open", "High", "Low", "Close", "Volume",
    "SMA_20", "SMA_50", "EMA_20", "RSI",
    "MACD", "MACD_Signal",
    "Daily_Return", "Volatility",
]

TARGET_COLUMN = "Close"

# ── Model Hyper-parameters ──────────────────────────────────────────
TRAIN_TEST_SPLIT = 0.8
LSTM_SEQUENCE_LENGTH = 60
LSTM_EPOCHS = 50
LSTM_BATCH_SIZE = 32
LSTM_UNITS_1 = 128
LSTM_UNITS_2 = 64
LSTM_DROPOUT = 0.2

RF_N_ESTIMATORS = 200
RF_MAX_DEPTH = 15
RF_RANDOM_STATE = 42

# ── Forecast ────────────────────────────────────────────────────────
FORECAST_DAYS = 7

# ── Model Names ─────────────────────────────────────────────────────
MODEL_LSTM = "LSTM"
MODEL_RF = "Random Forest"
MODEL_LR = "Linear Regression"
AVAILABLE_MODELS = [MODEL_LSTM, MODEL_RF, MODEL_LR]
