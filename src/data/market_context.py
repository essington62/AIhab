"""
src/data/market_context.py — Daily market context (VIX, DXY, Oil, S&P500).
Uses yfinance (no API key required).

Writes to: data/01_raw/market/
  - vix_daily.parquet       (^VIX)
  - dxy_daily.parquet       (DX-Y.NYB)
  - oil_daily.parquet       (CL=F)
  - sp500_daily.parquet     (^GSPC)

Used for dashboard context only — not wired into gate scoring.
Incremental: fetches only missing days from yfinance.
"""

import logging
from pathlib import Path

import pandas as pd
import yfinance as yf

from .utils import append_and_save, enforce_utc

logger = logging.getLogger("data_layer.market_context")

RAW_DIR = Path("data/01_raw/market")

TICKERS = {
    "vix_daily.parquet":   "^VIX",
    "dxy_daily.parquet":   "DX-Y.NYB",
    "oil_daily.parquet":   "CL=F",
    "sp500_daily.parquet": "^GSPC",
}

DEFAULT_LOOKBACK_DAYS = 365 * 3  # 3 years on first run


def _get_last_date(filepath: Path) -> pd.Timestamp | None:
    if not filepath.exists():
        return None
    df = pd.read_parquet(filepath)
    df = enforce_utc(df, "timestamp")
    return df["timestamp"].max()


def fetch_ticker(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Download OHLCV from yfinance. Returns DataFrame with timestamp col."""
    raw = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    if raw.empty:
        logger.warning(f"{ticker}: yfinance returned empty DataFrame")
        return pd.DataFrame()

    # yfinance returns MultiIndex columns when single ticker with auto_adjust
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    raw = raw.reset_index()
    # Date column name varies (Date or Datetime)
    date_col = "Date" if "Date" in raw.columns else "Datetime"
    raw = raw.rename(columns={date_col: "timestamp"})
    raw["timestamp"] = pd.to_datetime(raw["timestamp"], utc=True)

    # Keep only Close + Volume (context is close-price driven)
    keep = ["timestamp", "Close"]
    if "Volume" in raw.columns:
        keep.append("Volume")
    raw = raw[keep].rename(columns={"Close": "close", "Volume": "volume"})
    raw = raw.dropna(subset=["close"])
    return raw


def run() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    for filename, ticker in TICKERS.items():
        filepath = RAW_DIR / filename
        last_ts = _get_last_date(filepath)

        if last_ts is not None:
            # +1 day to avoid re-fetching last known row
            start = (last_ts + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
            logger.info(f"{ticker}: incremental fetch from {start}")
        else:
            start = (pd.Timestamp.utcnow() - pd.Timedelta(days=DEFAULT_LOOKBACK_DAYS)).strftime("%Y-%m-%d")
            logger.info(f"{ticker}: full fetch from {start}")

        end = (pd.Timestamp.utcnow() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

        try:
            df = fetch_ticker(ticker, start, end)
            if df.empty:
                logger.info(f"{filename}: no new data")
                continue
            df["source"] = ticker
            append_and_save(df, filepath, freq="daily")
        except Exception as e:
            logger.error(f"{filename} ({ticker}): fetch failed — {e}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S UTC",
    )
    run()
