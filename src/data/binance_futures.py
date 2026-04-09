"""
Binance Futures ingest: OI (1h), Taker ratio (1h), Funding rate (8h),
L/S account ratio (1h), L/S position ratio (1h).

Writes to: data/01_raw/futures/
  - oi_1h.parquet
  - taker_1h.parquet
  - funding_8h.parquet
  - ls_account_1h.parquet
  - ls_position_1h.parquet

Note: Binance Futures historical data = last 30 days only.
"""

import logging
from pathlib import Path

import pandas as pd

from .utils import append_and_save, enforce_utc, fetch_with_retry

logger = logging.getLogger("data_layer.binance_futures")

BASE_URL = "https://fapi.binance.com"
SYMBOL = "BTCUSDT"
PERIOD = "1h"
LIMIT = 500  # max per endpoint

RAW_DIR = Path("data/01_raw/futures")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_last_ts(filepath: Path) -> pd.Timestamp | None:
    if not filepath.exists():
        return None
    df = pd.read_parquet(filepath)
    df = enforce_utc(df, "timestamp")
    return df["timestamp"].max()


def _start_ms(filepath: Path, default_days: int = 30) -> int:
    last = _get_last_ts(filepath)
    if last is not None:
        return int(last.timestamp() * 1000) + 1
    return int(
        (pd.Timestamp.utcnow() - pd.Timedelta(days=default_days)).timestamp() * 1000
    )


# ---------------------------------------------------------------------------
# OI — /futures/data/openInterestHist
# ---------------------------------------------------------------------------

def fetch_oi(start_ms: int) -> pd.DataFrame:
    params = {
        "symbol": SYMBOL,
        "period": PERIOD,
        "limit": LIMIT,
        "startTime": start_ms,
    }
    data = fetch_with_retry(f"{BASE_URL}/futures/data/openInterestHist", params=params)
    if not data:
        return pd.DataFrame()
    df = pd.DataFrame(data)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df["sumOpenInterest"] = df["sumOpenInterest"].astype(float)
    df["sumOpenInterestValue"] = df["sumOpenInterestValue"].astype(float)
    df = df[["timestamp", "sumOpenInterest", "sumOpenInterestValue"]].copy()
    df["source"] = "binance_futures_oi"
    return df


# ---------------------------------------------------------------------------
# Taker Long/Short Ratio — /futures/data/takerlongshortRatio
# ---------------------------------------------------------------------------

def fetch_taker(start_ms: int) -> pd.DataFrame:
    params = {
        "symbol": SYMBOL,
        "period": PERIOD,
        "limit": LIMIT,
        "startTime": start_ms,
    }
    data = fetch_with_retry(
        f"{BASE_URL}/futures/data/takerlongshortRatio", params=params
    )
    if not data:
        return pd.DataFrame()
    df = pd.DataFrame(data)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df["buySellRatio"] = df["buySellRatio"].astype(float)
    df["buyVol"] = df["buyVol"].astype(float)
    df["sellVol"] = df["sellVol"].astype(float)
    df = df[["timestamp", "buySellRatio", "buyVol", "sellVol"]].copy()
    df["source"] = "binance_futures_taker"
    return df


# ---------------------------------------------------------------------------
# Funding Rate — /fapi/v1/fundingRate (events ~8h)
# ---------------------------------------------------------------------------

def fetch_funding(start_ms: int) -> pd.DataFrame:
    params = {
        "symbol": SYMBOL,
        "limit": 1000,
        "startTime": start_ms,
    }
    data = fetch_with_retry(f"{BASE_URL}/fapi/v1/fundingRate", params=params)
    if not data:
        return pd.DataFrame()
    df = pd.DataFrame(data)
    df["timestamp"] = pd.to_datetime(df["fundingTime"], unit="ms", utc=True)
    df["fundingRate"] = df["fundingRate"].astype(float)
    df = df[["timestamp", "fundingRate"]].copy()
    df["source"] = "binance_futures_funding"
    return df


# ---------------------------------------------------------------------------
# L/S Account Ratio — /futures/data/globalLongShortAccountRatio
# ---------------------------------------------------------------------------

def fetch_ls_account(start_ms: int) -> pd.DataFrame:
    params = {
        "symbol": SYMBOL,
        "period": PERIOD,
        "limit": LIMIT,
        "startTime": start_ms,
    }
    data = fetch_with_retry(
        f"{BASE_URL}/futures/data/globalLongShortAccountRatio", params=params
    )
    if not data:
        return pd.DataFrame()
    df = pd.DataFrame(data)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df["longShortRatio"] = df["longShortRatio"].astype(float)
    df["longAccount"] = df["longAccount"].astype(float)
    df["shortAccount"] = df["shortAccount"].astype(float)
    df = df[["timestamp", "longShortRatio", "longAccount", "shortAccount"]].copy()
    df["source"] = "binance_futures_ls_account"
    return df


# ---------------------------------------------------------------------------
# L/S Position Ratio — /futures/data/topLongShortPositionRatio
# ---------------------------------------------------------------------------

def fetch_ls_position(start_ms: int) -> pd.DataFrame:
    params = {
        "symbol": SYMBOL,
        "period": PERIOD,
        "limit": LIMIT,
        "startTime": start_ms,
    }
    data = fetch_with_retry(
        f"{BASE_URL}/futures/data/topLongShortPositionRatio", params=params
    )
    if not data:
        return pd.DataFrame()
    df = pd.DataFrame(data)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df["longShortRatio"] = df["longShortRatio"].astype(float)
    df["longAccount"] = df["longAccount"].astype(float)
    df["shortAccount"] = df["shortAccount"].astype(float)
    df = df[["timestamp", "longShortRatio", "longAccount", "shortAccount"]].copy()
    df["source"] = "binance_futures_ls_position"
    return df


# ---------------------------------------------------------------------------
# Run all
# ---------------------------------------------------------------------------

def run() -> None:
    tasks = [
        ("oi_1h.parquet", fetch_oi),
        ("taker_1h.parquet", fetch_taker),
        ("ls_account_1h.parquet", fetch_ls_account),
        ("ls_position_1h.parquet", fetch_ls_position),
    ]
    for filename, fetch_fn in tasks:
        filepath = RAW_DIR / filename
        start = _start_ms(filepath)
        logger.info(f"Fetching {filename} from {pd.Timestamp(start, unit='ms', tz='UTC')}")
        try:
            df = fetch_fn(start)
            if df.empty:
                logger.info(f"{filename}: no new data")
            else:
                append_and_save(df, filepath, freq="1h")
        except Exception as e:
            logger.error(f"{filename}: fetch failed — {e}")

    # Funding rate stored as 8h events
    funding_path = RAW_DIR / "funding_8h.parquet"
    start = _start_ms(funding_path)
    logger.info(f"Fetching funding_8h.parquet from {pd.Timestamp(start, unit='ms', tz='UTC')}")
    try:
        df = fetch_funding(start)
        if df.empty:
            logger.info("funding_8h.parquet: no new data")
        else:
            append_and_save(df, funding_path, freq="8h")
    except Exception as e:
        logger.error(f"funding_8h.parquet: fetch failed — {e}")


if __name__ == "__main__":
    run()
