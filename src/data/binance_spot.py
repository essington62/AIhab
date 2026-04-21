"""
Binance Spot OHLCV ingest — 1h candles.
Writes to: data/01_raw/spot/{symbol}_1h.parquet  (e.g. btc_1h.parquet, eth_1h.parquet)

Public API:
  fetch_spot_1h(symbol, start_time) — multi-symbol, batched, bootstrap-capable
  run()                             — backward compat BTC-only incremental
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
    df = df[["timestamp", "open", "high", "low", "close", "volume", "num_trades",
             "taker_buy_base_vol", "taker_buy_quote_vol"]].copy()
    for col in ["open", "high", "low", "close", "volume",
                "taker_buy_base_vol", "taker_buy_quote_vol"]:
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


def fetch_spot_1h(symbol: str = "BTC", start_time=None) -> None:
    """
    Multi-symbol 1h spot ingest with batch pagination.

    Args:
        symbol:     BTC, ETH, SOL, etc. (converted to {symbol}USDT for API)
        start_time: pd.Timestamp or datetime — start of range for bootstrap.
                    If None, resumes from last saved timestamp or defaults 90 days.

    Output: data/01_raw/spot/{symbol.lower()}_1h.parquet
    Loops in batches of LIMIT candles — supports full-year bootstrap (~9 calls).
    """
    pair = f"{symbol.upper()}USDT"
    raw_path = Path(f"data/01_raw/spot/{symbol.lower()}_1h.parquet")

    last_ts = get_last_timestamp(raw_path)
    if last_ts is not None:
        cur_ms = int(last_ts.timestamp() * 1000) + 1
        logger.info(f"{symbol} 1h: incremental from {last_ts}")
    elif start_time is not None:
        ts = pd.Timestamp(start_time)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        cur_ms = int(ts.timestamp() * 1000)
        logger.info(f"{symbol} 1h: bootstrap from {ts}")
    else:
        cur_ms = int((pd.Timestamp.utcnow() - pd.Timedelta(days=90)).timestamp() * 1000)
        logger.info(f"{symbol} 1h: default 90 days")

    now_ms = int(pd.Timestamp.utcnow().timestamp() * 1000)
    all_batches: list[pd.DataFrame] = []

    while cur_ms < now_ms:
        batch = fetch_spot_klines(symbol=pair, start_ms=cur_ms)
        if batch.empty:
            break
        all_batches.append(batch)
        if len(batch) < LIMIT:
            break
        cur_ms = int(batch["timestamp"].max().timestamp() * 1000) + 1

    if not all_batches:
        logger.info(f"{symbol} 1h: no new data")
        return

    df = pd.concat(all_batches, ignore_index=True)
    append_and_save(df, raw_path, freq="1h")
    logger.info(f"{symbol} 1h: +{len(df)} new rows → {raw_path}")


if __name__ == "__main__":
    run()
