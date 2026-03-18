"""
data_loader.py — Fetch historical stock data using Yahoo Finance API directly.
No external dependencies beyond requests + pandas (both already installed).
"""

import requests
import pandas as pd
import time
import io


def fetch_stock_data(
    ticker: str,
    start: str,
    end: str,
) -> pd.DataFrame:
    """
    Download OHLCV stock data from Yahoo Finance via direct API.

    Parameters
    ----------
    ticker : str   — Stock ticker symbol (e.g. "RELIANCE.NS").
    start  : str   — Start date in "YYYY-MM-DD" format.
    end    : str   — End date in "YYYY-MM-DD" format.

    Returns
    -------
    pd.DataFrame with columns: Open, High, Low, Close, Volume
    """
    # Convert dates to Unix timestamps
    start_ts = int(pd.Timestamp(start).timestamp())
    end_ts = int(pd.Timestamp(end).timestamp())

    # Yahoo Finance CSV download URL
    url = (
        f"https://query1.finance.yahoo.com/v7/finance/download/{ticker}"
        f"?period1={start_ts}&period2={end_ts}&interval=1d"
        f"&events=history&includeAdjustedClose=true"
    )

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }

    try:
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
    except requests.exceptions.RequestException:
        # Fallback: try v8 chart API
        return _fetch_via_chart_api(ticker, start_ts, end_ts, headers)

    df = pd.read_csv(io.StringIO(response.text), parse_dates=["Date"], index_col="Date")

    if df.empty:
        raise ValueError(
            f"No data returned for ticker '{ticker}' "
            f"between {start} and {end}. "
            "Check the ticker symbol — for Indian stocks append .NS or .BO."
        )

    # Keep only the columns we need
    keep = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    df = df[keep].copy()
    df.dropna(inplace=True)
    df.index.name = "Date"

    return df


def _fetch_via_chart_api(ticker: str, start_ts: int, end_ts: int, headers: dict) -> pd.DataFrame:
    """Fallback: use Yahoo Finance v8 chart API (returns JSON)."""
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
        f"?period1={start_ts}&period2={end_ts}&interval=1d"
    )

    resp = requests.get(url, headers=headers, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    result = data["chart"]["result"][0]
    timestamps = result["timestamp"]
    quote = result["indicators"]["quote"][0]

    df = pd.DataFrame({
        "Open": quote["open"],
        "High": quote["high"],
        "Low": quote["low"],
        "Close": quote["close"],
        "Volume": quote["volume"],
    }, index=pd.to_datetime(timestamps, unit="s"))

    df.index.name = "Date"
    df.dropna(inplace=True)

    if df.empty:
        raise ValueError(f"No data returned for ticker '{ticker}'.")

    return df


def get_stock_info(ticker: str) -> dict:
    """Return basic stock metadata (name, sector, currency, etc.)."""
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }

    try:
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}?interval=1d&range=1d"
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        meta = data["chart"]["result"][0]["meta"]

        return {
            "name": meta.get("longName", meta.get("shortName", ticker)),
            "sector": "N/A",
            "industry": "N/A",
            "currency": meta.get("currency", "INR"),
            "market_cap": "N/A",
            "52w_high": meta.get("fiftyTwoWeekHigh", "N/A"),
            "52w_low": meta.get("fiftyTwoWeekLow", "N/A"),
        }
    except Exception:
        return {
            "name": ticker,
            "sector": "N/A",
            "industry": "N/A",
            "currency": "INR",
            "market_cap": "N/A",
            "52w_high": "N/A",
            "52w_low": "N/A",
        }
