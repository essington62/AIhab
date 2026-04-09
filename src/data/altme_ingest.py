"""
Alt.me Fear & Greed Index ingest (daily).
Writes to: data/01_raw/sentiment/fear_greed_daily.parquet

API: https://api.alternative.me/fng/?limit=N — public, no key required.
"""

import logging
from pathlib import Path

import pandas as pd

from .utils import append_and_save, enforce_utc, fetch_with_retry

logger = logging.getLogger("data_layer.altme")

API_URL = "https://api.alternative.me/fng/"
RAW_PATH = Path("data/01_raw/sentiment/fear_greed_daily.parquet")


def _get_last_date(filepath: Path) -> pd.Timestamp | None:
    if not filepath.exists():
        return None
    df = pd.read_parquet(filepath)
    df = enforce_utc(df, "timestamp")
    return df["timestamp"].max()


def fetch_fear_greed(limit: int = 365) -> pd.DataFrame:
    data = fetch_with_retry(API_URL, params={"limit": limit, "format": "json"})
    rows = data.get("data", [])
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    # API returns Unix timestamp as string
    df["timestamp"] = pd.to_datetime(df["timestamp"].astype(int), unit="s", utc=True)
    df["fg_value"] = df["value"].astype(int)
    df["fg_classification"] = df["value_classification"]
    df = df[["timestamp", "fg_value", "fg_classification"]].copy()
    df["source"] = "altme_fear_greed"
    return df


def run() -> None:
    last_ts = _get_last_date(RAW_PATH)

    if last_ts is not None:
        days_missing = (pd.Timestamp.utcnow() - last_ts).days + 2
        limit = min(days_missing, 365)
        logger.info(f"Incremental F&G fetch: last={last_ts}, limit={limit}")
    else:
        limit = 365
        logger.info("Full F&G fetch (365 days)")

    df = fetch_fear_greed(limit=limit)
    if df.empty:
        logger.info("Fear & Greed: no new data")
        return

    append_and_save(df, RAW_PATH, freq="daily")


if __name__ == "__main__":
    run()
