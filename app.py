"""
app.py — Streamlit dashboard for Stock Price Prediction.

Run with:  streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

import config
from src.data_loader import fetch_stock_data, get_stock_info
from src.preprocessor import prepare_data, add_technical_indicators
from src.predictor import train_and_predict, forecast_future
from src.visualizer import (
    plot_stock_history,
    plot_predictions,
    plot_model_comparison,
    plot_forecast,
)

# ── Page Config ──────────────────────────────────────────────────────

st.set_page_config(
    page_title="📈 Stock Price Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────

st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #2196F3, #4CAF50);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .sub-header {
        color: #888;
        font-size: 1rem;
        margin-top: -10px;
        margin-bottom: 20px;
    }
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border-radius: 12px;
        padding: 20px;
        border: 1px solid #333;
    }
    .stMetric > div {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border-radius: 10px;
        padding: 12px;
        border: 1px solid #2a2a3e;
    }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ──────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    st.markdown("---")

    ticker = st.text_input(
        "🏷️ Stock Ticker",
        value=config.DEFAULT_TICKER,
        help="Indian stocks: append .NS (NSE) or .BO (BSE). E.g. TCS.NS, INFY.NS",
    )

    st.markdown("##### 📅 Date Range")
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input(
            "Start",
            value=pd.to_datetime(config.DEFAULT_START_DATE),
        )
    with col2:
        end_date = st.date_input(
            "End",
            value=pd.to_datetime(config.DEFAULT_END_DATE),
        )

    st.markdown("---")
    st.markdown("##### 🤖 Models")
    selected_models = st.multiselect(
        "Select models to train",
        options=config.AVAILABLE_MODELS,
        default=config.AVAILABLE_MODELS,
    )

    st.markdown("---")
    train_button = st.button("🚀 Train & Predict", use_container_width=True, type="primary")

    st.markdown("---")
    st.markdown(
        "<div style='text-align:center; color:#666; font-size:0.8rem;'>"
        "Built with ❤️ using Python & Streamlit"
        "</div>",
        unsafe_allow_html=True,
    )

# ── Main Area ────────────────────────────────────────────────────────

st.markdown('<p class="main-header">📈 Stock Price Predictor</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Predict stock prices using LSTM, Random Forest & Linear Regression</p>', unsafe_allow_html=True)

# ── Load & Display Stock Data ────────────────────────────────────────

try:
    with st.spinner("Fetching stock data..."):
        df = fetch_stock_data(ticker, str(start_date), str(end_date))

    # Stock info cards
    try:
        info = get_stock_info(ticker)
        info_cols = st.columns(5)
        info_cols[0].metric("🏢 Company", info["name"][:25])
        info_cols[1].metric("💰 Currency", info["currency"])
        info_cols[2].metric("📈 52W High", f"₹{info['52w_high']:,.2f}" if isinstance(info['52w_high'], (int, float)) else "N/A")
        info_cols[3].metric("📉 52W Low", f"₹{info['52w_low']:,.2f}" if isinstance(info['52w_low'], (int, float)) else "N/A")
        info_cols[4].metric("📊 Data Points", f"{len(df):,}")
    except Exception:
        st.info(f"Loaded **{len(df):,}** data points for **{ticker}**")

    # Add features for the history chart
    df_with_indicators = add_technical_indicators(df)

    # Tabs for overview
    tab_chart, tab_data = st.tabs(["📊 Price Chart", "📋 Raw Data"])

    with tab_chart:
        fig_history = plot_stock_history(df_with_indicators, ticker)
        st.pyplot(fig_history)

    with tab_data:
        st.dataframe(
            df.tail(100).style.format({
                "Open": "₹{:.2f}",
                "High": "₹{:.2f}",
                "Low": "₹{:.2f}",
                "Close": "₹{:.2f}",
                "Volume": "{:,.0f}",
            }),
            use_container_width=True,
        )

except Exception as e:
    st.error(f"❌ Error fetching data: {e}")
    st.stop()

# ── Model Training ───────────────────────────────────────────────────

if train_button:
    if not selected_models:
        st.warning("⚠️ Please select at least one model.")
        st.stop()

    st.markdown("---")
    st.markdown("## 🧠 Model Training & Predictions")

    all_results = {}
    all_metrics = {}

    for model_name in selected_models:
        st.markdown(f"### {model_name}")

        is_lstm = model_name == config.MODEL_LSTM
        progress_bar = st.progress(0, text=f"Preparing data for {model_name}...")

        try:
            # Prepare data
            data = prepare_data(df, for_lstm=is_lstm)
            progress_bar.progress(20, text=f"Training {model_name}...")

            # Progress callback for LSTM
            def make_progress_cb(pbar):
                def cb(epoch, total):
                    pct = 20 + int((epoch / total) * 60)
                    pbar.progress(pct, text=f"Training {model_name}... Epoch {epoch}/{total}")
                return cb

            result = train_and_predict(
                model_name=model_name,
                X_train=data["X_train"],
                y_train=data["y_train"],
                X_test=data["X_test"],
                y_test=data["y_test"],
                target_scaler=data["target_scaler"],
                progress_callback=make_progress_cb(progress_bar) if is_lstm else None,
            )

            progress_bar.progress(90, text=f"Generating charts for {model_name}...")

            all_results[model_name] = result
            all_results[model_name]["dates_test"] = data["dates_test"]
            all_results[model_name]["data"] = data
            all_metrics[model_name] = result["metrics"]

            # Metrics row
            mcols = st.columns(4)
            for i, (metric_name, metric_val) in enumerate(result["metrics"].items()):
                mcols[i].metric(metric_name, f"{metric_val:.4f}")

            # Prediction chart
            fig_pred = plot_predictions(
                dates=data["dates_test"],
                y_true=result["y_true"],
                y_pred=result["predictions"],
                model_name=model_name,
            )
            st.pyplot(fig_pred)

            progress_bar.progress(100, text=f"✅ {model_name} complete!")

        except Exception as e:
            st.error(f"❌ Error training {model_name}: {e}")
            import traceback
            st.code(traceback.format_exc())
            continue

    # ── Model Comparison ─────────────────────────────────────────────

    if len(all_metrics) > 1:
        st.markdown("---")
        st.markdown("## 📊 Model Comparison")

        # Metrics table
        metrics_df = pd.DataFrame(all_metrics).T
        st.dataframe(
            metrics_df.style.highlight_min(axis=0, subset=["MAE", "RMSE", "MAPE (%)"], props="background-color: #1b5e20; color: white;")
                           .highlight_max(axis=0, subset=["R²"], props="background-color: #1b5e20; color: white;"),
            use_container_width=True,
        )

        # Comparison chart
        fig_comp = plot_model_comparison(all_metrics)
        st.pyplot(fig_comp)

        # Best model
        best_model = min(all_metrics, key=lambda m: all_metrics[m]["RMSE"])
        st.success(f"🏆 **Best Model (lowest RMSE):** {best_model} — RMSE: {all_metrics[best_model]['RMSE']:.4f}")

    # ── Future Forecast ──────────────────────────────────────────────

    if all_results:
        st.markdown("---")
        st.markdown("## 🚀 Future Price Forecast")

        # Use the best model (or first available)
        forecast_model_name = min(all_metrics, key=lambda m: all_metrics[m]["RMSE"]) if len(all_metrics) > 1 else list(all_results.keys())[0]

        st.info(f"Forecasting next **{config.FORECAST_DAYS} days** using **{forecast_model_name}**")

        try:
            result = all_results[forecast_model_name]
            data = result["data"]

            is_lstm = forecast_model_name == config.MODEL_LSTM

            if is_lstm:
                last_seq = data["X_test"][-1]
            else:
                last_seq = data["X_test"][-1]

            forecast_prices = forecast_future(
                model=result["model"],
                model_name=forecast_model_name,
                last_sequence=last_seq,
                feature_scaler=data["feature_scaler"],
                target_scaler=data["target_scaler"],
            )

            # Create forecast dates
            last_date = data["dates_test"][-1]
            forecast_dates = pd.date_range(
                start=last_date + timedelta(days=1),
                periods=config.FORECAST_DAYS,
                freq="B",  # Business days
            )

            # Recent 30 days for context
            recent_n = min(30, len(result["y_true"]))
            recent_dates = data["dates_test"][-recent_n:]
            recent_prices = result["y_true"][-recent_n:]

            fig_forecast = plot_forecast(
                last_dates=recent_dates,
                last_prices=recent_prices,
                forecast_dates=forecast_dates,
                forecast_prices=forecast_prices,
                model_name=forecast_model_name,
            )
            st.pyplot(fig_forecast)

            # Forecast table
            forecast_df = pd.DataFrame({
                "Date": forecast_dates.strftime("%d %b %Y"),
                "Predicted Price (₹)": [f"₹{p:,.2f}" for p in forecast_prices],
                "Change (₹)": [f"₹{forecast_prices[i] - (forecast_prices[i-1] if i > 0 else recent_prices[-1]):+,.2f}" for i in range(len(forecast_prices))],
            })
            st.dataframe(forecast_df, use_container_width=True, hide_index=True)

        except Exception as e:
            st.warning(f"⚠️ Could not generate forecast: {e}")

    st.markdown("---")
    st.markdown(
        "<div style='text-align:center; color:#666;'>"
        "⚠️ <b>Disclaimer:</b> This is for educational purposes only. "
        "Stock predictions are not financial advice."
        "</div>",
        unsafe_allow_html=True,
    )
