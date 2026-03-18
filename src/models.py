"""
models.py — Build ML model instances: LSTM (PyTorch), Random Forest, Linear Regression.
"""

import numpy as np
import torch
import torch.nn as nn

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


# ── PyTorch LSTM ─────────────────────────────────────────────────────

class LSTMModel(nn.Module):
    """Two-layer LSTM for time-series regression."""

    def __init__(self, input_size: int, hidden1: int, hidden2: int, dropout: float):
        super().__init__()
        self.lstm1 = nn.LSTM(input_size, hidden1, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.lstm2 = nn.LSTM(hidden1, hidden2, batch_first=True)
        self.dropout2 = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden2, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        out, _ = self.lstm1(x)
        out = self.dropout1(out)
        out, _ = self.lstm2(out)
        out = self.dropout2(out[:, -1, :])  # last time-step
        out = self.relu(self.fc1(out))
        return self.fc2(out).squeeze(-1)


class LSTMWrapper:
    """
    Scikit-learn-style wrapper around the PyTorch LSTM so the rest of the
    codebase can call .fit() / .predict() uniformly.
    """

    def __init__(self, input_size: int, epochs: int = None, batch_size: int = None, lr: float = 1e-3):
        self.epochs = epochs or config.LSTM_EPOCHS
        self.batch_size = batch_size or config.LSTM_BATCH_SIZE
        self.lr = lr
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = LSTMModel(
            input_size=input_size,
            hidden1=config.LSTM_UNITS_1,
            hidden2=config.LSTM_UNITS_2,
            dropout=config.LSTM_DROPOUT,
        ).to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def fit(self, X: np.ndarray, y: np.ndarray, progress_callback=None):
        """Train the LSTM model."""
        self.model.train()

        X_t = torch.FloatTensor(X).to(self.device)
        y_t = torch.FloatTensor(y).to(self.device)

        dataset = torch.utils.data.TensorDataset(X_t, y_t)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=False
        )

        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for X_batch, y_batch in loader:
                self.optimizer.zero_grad()
                pred = self.model(X_batch)
                loss = self.criterion(pred, y_batch)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()

            if progress_callback:
                progress_callback(epoch + 1, self.epochs)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions."""
        self.model.eval()
        with torch.no_grad():
            X_t = torch.FloatTensor(X).to(self.device)
            return self.model(X_t).cpu().numpy()


def build_lstm(input_shape: tuple) -> LSTMWrapper:
    """
    Build a PyTorch LSTM model wrapped with fit/predict interface.

    Parameters
    ----------
    input_shape : (sequence_length, n_features)
    """
    return LSTMWrapper(input_size=input_shape[1])


def build_random_forest():
    """Build a scikit-learn RandomForestRegressor."""
    from sklearn.ensemble import RandomForestRegressor

    return RandomForestRegressor(
        n_estimators=config.RF_N_ESTIMATORS,
        max_depth=config.RF_MAX_DEPTH,
        random_state=config.RF_RANDOM_STATE,
        n_jobs=-1,
    )


def build_linear_regression():
    """Build a scikit-learn LinearRegression."""
    from sklearn.linear_model import LinearRegression

    return LinearRegression()
