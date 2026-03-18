# 📈 Stock Price Prediction — ML Project

A beginner-friendly machine learning project that predicts stock prices using **LSTM**, **Random Forest**, and **Linear Regression** models with an interactive **Streamlit** dashboard.

---

## ✨ Features

| Feature | Description |
|---|---|
| **Multi-model prediction** | Compare LSTM, Random Forest & Linear Regression side-by-side |
| **Technical indicators** | SMA, EMA, RSI, MACD, Volatility as engineered features |
| **Interactive UI** | Streamlit dashboard with charts and metrics |
| **Model comparison** | MAE, RMSE, R², MAPE across all models |
| **Future forecast** | Predict stock prices for the next 7 days |

---

## 🛠️ Tech Stack

- **Python 3.10+**
- **yfinance** — historical stock data
- **Pandas / NumPy** — data manipulation
- **Scikit-learn** — Random Forest, Linear Regression, metrics
- **TensorFlow / Keras** — LSTM neural network
- **Matplotlib** — visualization
- **Streamlit** — web UI
- **ta** — technical analysis indicators

---

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Launch the app
streamlit run app.py
```

Then open **http://localhost:8501** in your browser.

---

## 📁 Project Structure

```
├── app.py               # Streamlit entry point
├── config.py            # Configuration & constants
├── requirements.txt     # Dependencies
├── src/
│   ├── data_loader.py   # Fetch data via yfinance
│   ├── preprocessor.py  # Feature engineering & scaling
│   ├── models.py        # LSTM, RF, LR model builders
│   ├── predictor.py     # Training & evaluation pipeline
│   └── visualizer.py    # Chart utilities
└── README.md
```

---

## 📊 Usage

1. Enter a stock ticker in the sidebar (e.g., `RELIANCE.NS`, `TCS.NS`, `INFY.NS`)
2. Pick a date range
3. Select one or more models
4. Click **🚀 Train & Predict**
5. View predictions, metrics, and model comparison charts

---

## 📝 License

This project is for educational purposes.
