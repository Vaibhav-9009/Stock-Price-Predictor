"""
visualizer.py — Matplotlib chart utilities for the Streamlit UI.
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd


# ── Style defaults ───────────────────────────────────────────────────

COLORS = {
    "actual": "#2196F3",
    "predicted": "#FF5722",
    "forecast": "#4CAF50",
    "volume": "#90CAF9",
    "sma20": "#FF9800",
    "sma50": "#9C27B0",
}

plt.rcParams.update({
    "figure.facecolor": "#0E1117",
    "axes.facecolor": "#0E1117",
    "axes.edgecolor": "#333333",
    "axes.labelcolor": "#FAFAFA",
    "text.color": "#FAFAFA",
    "xtick.color": "#CCCCCC",
    "ytick.color": "#CCCCCC",
    "grid.color": "#1E1E1E",
    "grid.alpha": 0.5,
    "font.size": 11,
})


def plot_stock_history(df: pd.DataFrame, ticker: str = "") -> plt.Figure:
    """
    Plot historical Close price with volume bars.
    """
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(12, 6), gridspec_kw={"height_ratios": [3, 1]}, sharex=True
    )

    ax1.plot(df.index, df["Close"], color=COLORS["actual"], linewidth=1.5, label="Close")

    if "SMA_20" in df.columns:
        ax1.plot(df.index, df["SMA_20"], color=COLORS["sma20"], linewidth=1, alpha=0.7, label="SMA 20")
    if "SMA_50" in df.columns:
        ax1.plot(df.index, df["SMA_50"], color=COLORS["sma50"], linewidth=1, alpha=0.7, label="SMA 50")

    ax1.set_title(f"📈 {ticker} — Historical Price", fontsize=14, fontweight="bold")
    ax1.set_ylabel("Price (₹)")
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(True)

    ax2.bar(df.index, df["Volume"], color=COLORS["volume"], alpha=0.6, width=1.5)
    ax2.set_ylabel("Volume")
    ax2.grid(True)

    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


def plot_predictions(
    dates,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "",
) -> plt.Figure:
    """
    Actual vs Predicted line chart.
    """
    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(dates, y_true, color=COLORS["actual"], linewidth=1.5, label="Actual")
    ax.plot(dates, y_pred, color=COLORS["predicted"], linewidth=1.5, linestyle="--", label="Predicted")

    ax.fill_between(
        dates, y_true, y_pred,
        alpha=0.1, color=COLORS["predicted"],
    )

    ax.set_title(f"🔮 {model_name} — Actual vs Predicted", fontsize=14, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (₹)")
    ax.legend(fontsize=10)
    ax.grid(True)

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


def plot_model_comparison(metrics_dict: dict) -> plt.Figure:
    """
    Grouped bar chart comparing metrics across models.

    Parameters
    ----------
    metrics_dict : {model_name: {metric_name: value, ...}, ...}
    """
    models = list(metrics_dict.keys())
    metric_names = list(next(iter(metrics_dict.values())).keys())
    n_metrics = len(metric_names)
    n_models = len(models)

    fig, axes = plt.subplots(1, n_metrics, figsize=(4 * n_metrics, 5))
    if n_metrics == 1:
        axes = [axes]

    bar_colors = ["#2196F3", "#FF5722", "#4CAF50", "#FF9800", "#9C27B0"]

    for i, metric in enumerate(metric_names):
        values = [metrics_dict[m][metric] for m in models]
        bars = axes[i].bar(
            models, values,
            color=[bar_colors[j % len(bar_colors)] for j in range(n_models)],
            edgecolor="white", linewidth=0.5, alpha=0.85,
        )
        axes[i].set_title(metric, fontsize=12, fontweight="bold")
        axes[i].grid(axis="y", alpha=0.3)

        # Value labels on bars
        for bar, val in zip(bars, values):
            axes[i].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{val:.2f}",
                ha="center", va="bottom", fontsize=9,
            )

    fig.suptitle("📊 Model Comparison", fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    return fig


def plot_forecast(
    last_dates,
    last_prices: np.ndarray,
    forecast_dates,
    forecast_prices: np.ndarray,
    model_name: str = "",
) -> plt.Figure:
    """
    Plot recent history plus future forecast.
    """
    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(last_dates, last_prices, color=COLORS["actual"], linewidth=1.5, label="Recent Prices")
    ax.plot(forecast_dates, forecast_prices, color=COLORS["forecast"], linewidth=2,
            marker="o", markersize=5, linestyle="--", label="Forecast")

    # Connect the two lines
    ax.plot(
        [last_dates[-1], forecast_dates[0]],
        [last_prices[-1], forecast_prices[0]],
        color=COLORS["forecast"], linewidth=1.5, linestyle=":",
    )

    ax.fill_between(
        forecast_dates, forecast_prices,
        alpha=0.15, color=COLORS["forecast"],
    )

    ax.set_title(f"🚀 {model_name} — Future Forecast", fontsize=14, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (₹)")
    ax.legend(fontsize=10)
    ax.grid(True)

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b %Y"))
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig
