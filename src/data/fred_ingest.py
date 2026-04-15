"""
FRED macro data ingest: DGS10, DGS2, RRPONTSYD (combined) +
Fed Observatory series: EFFR, DFEDTARU, DFEDTARL, T5YIE, T10YIE (individual parquets).

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

# Fed Observatory series — each stored as individual parquet
OBSERVATORY_SERIES = {
    "EFFR":     Path("data/01_raw/macro/effr.parquet"),       # Effective Fed Funds Rate
    "DFEDTARU": Path("data/01_raw/macro/dfedtaru.parquet"),   # Target Upper
    "DFEDTARL": Path("data/01_raw/macro/dfedtarl.parquet"),   # Target Lower
    "T5YIE":    Path("data/01_raw/macro/t5yie.parquet"),      # Breakeven Inflation 5Y
    "T10YIE":   Path("data/01_raw/macro/t10yie.parquet"),     # Breakeven Inflation 10Y
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


def _run_observatory(fred: Fred) -> None:
    """Fetch Fed Observatory series, each to its own parquet."""
    RAW_DIR = Path("data/01_raw/macro")
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    for series_id, filepath in OBSERVATORY_SERIES.items():
        try:
            last_date = _get_last_date(filepath)
            s = fetch_series(fred, series_id, last_date)
            if s.empty:
                logger.info(f"FRED {series_id}: no new data")
                continue
            df = s.reset_index()
            df.columns = ["date", "value"]
            df["timestamp"] = pd.to_datetime(df["date"], utc=True)
            df = df.drop(columns=["date"])
            df["source"] = "fred"
            df = df.sort_values("timestamp")
            append_and_save(df, filepath, freq="daily")
            logger.info(f"FRED {series_id}: +{len(df)} rows → {filepath.name}")
        except Exception as e:
            logger.error(f"FRED {series_id} (observatory): fetch failed — {e}")


def run() -> None:
    api_key = _load_api_key()
    fred = Fred(api_key=api_key)

    # --- Combined series (fred_daily.parquet) ---
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
    else:
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

    # --- Observatory series (individual parquets) ---
    _run_observatory(fred)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S UTC",
    )
    run()
