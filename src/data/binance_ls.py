"""
Binance Futures L/S ratios — top traders only (account + position).
Writes to: data/01_raw/futures/
  - ls_account_1h.parquet   (topLongShortAccountRatio)
  - ls_position_1h.parquet  (topLongShortPositionRatio)

Endpoints:
  GET /futures/data/topLongShortAccountRatio
  GET /futures/data/topLongShortPositionRatio
  params: symbol=BTCUSDT, period=1h, limit=500

Note: CoinGlass does not carry these endpoints — Binance is the only source.
Max history per request: 500 hours (~21 days). Run hourly to accumulate.
"""

import logging
from pathlib import Path

import pandas as pd

from .utils import append_and_save, enforce_utc, fetch_with_retry

logger = logging.getLogger("data_layer.binance_ls")

BASE_URL = "https://fapi.binance.com"
SYMBOL = "BTCUSDT"
PERIOD = "1h"
LIMIT = 500

RAW_DIR = Path("data/01_raw/futures")


def _start_ms(filepath: Path, default_days: int = 21) -> int:
    if filepath.exists():
        df = pd.read_parquet(filepath)
        df = enforce_utc(df, "timestamp")
        last = df["timestamp"].max()
        return int(last.timestamp() * 1000) + 1
    return int(
        (pd.Timestamp.utcnow() - pd.Timedelta(days=default_days)).timestamp() * 1000
    )


def _parse_ls(data: list, source: str) -> pd.DataFrame:
    if not data:
        return pd.DataFrame()
    df = pd.DataFrame(data)
    df["timestamp"]     = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df["longShortRatio"] = pd.to_numeric(df["longShortRatio"], errors="coerce")
    df["longAccount"]    = pd.to_numeric(df["longAccount"],    errors="coerce")
    df["shortAccount"]   = pd.to_numeric(df["shortAccount"],   errors="coerce")
    df = df[["timestamp", "longShortRatio", "longAccount", "shortAccount"]].copy()
    df["source"] = source
    return df.dropna(subset=["longShortRatio"])


def fetch_ls_account(start_ms: int) -> pd.DataFrame:
    data = fetch_with_retry(
        f"{BASE_URL}/futures/data/topLongShortAccountRatio",
        params={"symbol": SYMBOL, "period": PERIOD, "limit": LIMIT, "startTime": start_ms},
    )
    return _parse_ls(data or [], "binance_ls_account_top")


def fetch_ls_position(start_ms: int) -> pd.DataFrame:
    data = fetch_with_retry(
        f"{BASE_URL}/futures/data/topLongShortPositionRatio",
        params={"symbol": SYMBOL, "period": PERIOD, "limit": LIMIT, "startTime": start_ms},
    )
    return _parse_ls(data or [], "binance_ls_position_top")


def run() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    tasks = [
        ("ls_account_1h.parquet",  fetch_ls_account),
        ("ls_position_1h.parquet", fetch_ls_position),
    ]
    for filename, fetch_fn in tasks:
        filepath = RAW_DIR / filename
        start = _start_ms(filepath)
        logger.info(
            f"Fetching {filename} from {pd.Timestamp(start, unit='ms', tz='UTC')}"
        )
        try:
            df = fetch_fn(start)
            if df.empty:
                logger.info(f"{filename}: no new data")
            else:
                append_and_save(df, filepath, freq="1h")
                logger.info(f"{filename}: +{len(df)} rows")
        except Exception as e:
            logger.error(f"{filename}: fetch failed — {e}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run()
