"""
FRED macro data ingest: DGS10, DGS2, RRPONTSYD.
Writes to: data/01_raw/macro/fred_daily.parquet

Fix 5: incremental fetch uses last_date + 1 day to avoid duplicating last row.
Requires: pip install fredapi
Credentials: conf/credentials.yml → fred_api_key
"""

import logging
from pathlib import Path

import pandas as pd
import yaml
from fredapi import Fred

from .utils import append_and_save, enforce_utc

logger = logging.getLogger("data_layer.fred")

RAW_PATH = Path("data/01_raw/macro/fred_daily.parquet")
CREDENTIALS_PATH = Path("conf/credentials.yml")

SERIES = {
    "DGS10": "dgs10",       # 10Y Treasury yield
    "DGS2": "dgs2",         # 2Y Treasury yield
    "RRPONTSYD": "rrp",     # Reverse Repo
}


def _load_api_key() -> str:
    with open(CREDENTIALS_PATH) as f:
        creds = yaml.safe_load(f)
    return creds["fred_api_key"]


def _get_last_date(filepath: Path) -> pd.Timestamp | None:
    if not filepath.exists():
        return None
    df = pd.read_parquet(filepath)
    df = enforce_utc(df, "timestamp")
    return df["timestamp"].max()


def fetch_series(fred: Fred, series_id: str, last_date: pd.Timestamp | None) -> pd.Series:
    """Fetch FRED series incrementally. Fix 5: start = last_date + 1 day."""
    if last_date is not None:
        start = (last_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    else:
        start = "2020-01-01"

    logger.info(f"Fetching FRED {series_id} from {start}")
    return fred.get_series(series_id, observation_start=start)


def run() -> None:
    api_key = _load_api_key()
    fred = Fred(api_key=api_key)

    last_date = _get_last_date(RAW_PATH)

    series_list = []
    for series_id, col_name in SERIES.items():
        try:
            s = fetch_series(fred, series_id, last_date)
            if s.empty:
                logger.info(f"FRED {series_id}: no new data")
                continue
            df = s.reset_index()
            df.columns = ["date", col_name]
            # Fix 1: convert `date` → `timestamp` UTC
            df["timestamp"] = pd.to_datetime(df["date"], utc=True)
            df = df.drop(columns=["date"])
            df["source"] = "fred"
            series_list.append(df.set_index("timestamp"))
        except Exception as e:
            logger.error(f"FRED {series_id}: fetch failed — {e}")

    if not series_list:
        logger.info("No new FRED data")
        return

    # Outer join: all series aligned on date index
    combined = series_list[0]
    for s in series_list[1:]:
        combined = combined.join(s.drop(columns=["source"], errors="ignore"), how="outer")
    combined = combined.reset_index()

    # Forward-fill NaN (weekends/holidays) after sort
    combined = combined.sort_values("timestamp")
    fill_cols = [c for c in combined.columns if c not in ("timestamp", "source")]
    combined[fill_cols] = combined[fill_cols].ffill()
    combined["source"] = "fred"

    append_and_save(combined, RAW_PATH, freq="daily")


if __name__ == "__main__":
    run()
