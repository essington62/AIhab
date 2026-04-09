"""
Binance Spot OHLCV ingest — 1h candles.
Writes to: data/01_raw/spot/btc_1h.parquet
"""

import logging
from pathlib import Path

import pandas as pd

from .utils import append_and_save, enforce_utc, fetch_with_retry

logger = logging.getLogger("data_layer.binance_spot")

BASE_URL = "https://api.binance.com"
RAW_PATH = Path("data/01_raw/spot/btc_1h.parquet")
SYMBOL = "BTCUSDT"
INTERVAL = "1h"
LIMIT = 1000  # max per request


def fetch_spot_klines(
    symbol: str = SYMBOL,
    interval: str = INTERVAL,
    start_ms: int | None = None,
    end_ms: int | None = None,
    limit: int = LIMIT,
) -> pd.DataFrame:
    """Fetch raw klines from Binance Spot API."""
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    if start_ms:
        params["startTime"] = start_ms
    if end_ms:
        params["endTime"] = end_ms

    data = fetch_with_retry(f"{BASE_URL}/api/v3/klines", params=params)

    df = pd.DataFrame(data, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "num_trades",
        "taker_buy_base_vol", "taker_buy_quote_vol", "ignore",
    ])
    df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df = df[["timestamp", "open", "high", "low", "close", "volume", "num_trades"]].copy()
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    df["num_trades"] = df["num_trades"].astype(int)
    df["source"] = "binance_spot"  # Fix 8
    return df


def get_last_timestamp(filepath: Path) -> pd.Timestamp | None:
    """Return last timestamp in existing parquet, or None."""
    if not filepath.exists():
        return None
    df = pd.read_parquet(filepath)
    df = enforce_utc(df, "timestamp")
    return df["timestamp"].max()


def run(start_ms: int | None = None) -> None:
    """
    Incremental ingest: fetch from last known timestamp forward.
    Fetches up to LIMIT=1000 candles per call (covers ~41 days of 1h data).
    """
    last_ts = get_last_timestamp(RAW_PATH)

    if last_ts is not None:
        # +1ms to avoid re-fetching last known candle
        start_ms = int(last_ts.timestamp() * 1000) + 1
        logger.info(f"Incremental fetch from {last_ts}")
    elif start_ms is None:
        # Default: last 90 days
        start_ms = int(
            (pd.Timestamp.utcnow() - pd.Timedelta(days=90)).timestamp() * 1000
        )
        logger.info("No existing data — fetching last 90 days")

    new_df = fetch_spot_klines(start_ms=start_ms)

    if new_df.empty:
        logger.info("No new spot data")
        return

    append_and_save(new_df, RAW_PATH, freq="1h")


if __name__ == "__main__":
    run()
