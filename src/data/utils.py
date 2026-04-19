"""
Shared utilities for the data layer.
Covers: Fix 1 (UTC), Fix 2 (window), Fix 7 (retry), Fix 8 (source col),
        Fix 9 (monotonicity), Fix 10 (logging).
"""

import hashlib
import logging
import time
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Fix 10: Structured logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S UTC",
)
logger = logging.getLogger("data_layer")


# ---------------------------------------------------------------------------
# Fix 7: Retry with exponential backoff
# ---------------------------------------------------------------------------
def fetch_with_retry(
    url: str,
    params: Optional[dict] = None,
    headers: Optional[dict] = None,
    max_retries: int = 3,
    timeout: int = 30,
) -> dict | list:
    """GET request with exponential backoff. Raises on final failure."""
    for attempt in range(max_retries):
        try:
            r = requests.get(url, params=params, headers=headers, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except requests.exceptions.RequestException as e:
            wait = 2 ** attempt
            logger.warning(
                f"Attempt {attempt + 1}/{max_retries} failed for {url}: {e}. "
                f"Retry in {wait}s"
            )
            if attempt < max_retries - 1:
                time.sleep(wait)
            else:
                logger.error(f"All {max_retries} attempts failed for {url}")
                raise


# ---------------------------------------------------------------------------
# Fix 1: UTC enforcement
# ---------------------------------------------------------------------------
def enforce_utc(df: pd.DataFrame, col: str = "timestamp") -> pd.DataFrame:
    """Ensure `col` is timezone-aware UTC. Raises if col missing."""
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found. Columns: {list(df.columns)}")
    df[col] = pd.to_datetime(df[col], utc=True)
    assert df[col].dt.tz is not None, f"Timezone must be UTC in column '{col}'"
    return df


# ---------------------------------------------------------------------------
# Fix 2: Windowed parquet save
# ---------------------------------------------------------------------------
MAX_ROWS = {
    "1h": 8760,    # ~1 year
    "daily": 1095,  # ~3 years
    "8h": 3285,    # ~3 years of 8h events
}


def save_with_window(
    df: pd.DataFrame,
    filepath: str | Path,
    freq: str = "1h",
    ts_col: str = "timestamp",
) -> None:
    """Sort by timestamp, deduplicate, truncate to max window, save parquet."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    max_rows = MAX_ROWS.get(freq, 8760)
    df = df.sort_values(ts_col).drop_duplicates(subset=[ts_col]).tail(max_rows)

    # Fix 9: monotonicity check
    assert df[ts_col].is_monotonic_increasing, (
        f"Non-monotonic timestamps in {filepath}"
    )

    df.to_parquet(filepath, index=False)
    logger.info(
        f"{filepath.name}: saved {len(df)} rows, "
        f"last={df[ts_col].max()}"
    )


# ---------------------------------------------------------------------------
# Append-and-save: load existing + concat + window + save
# ---------------------------------------------------------------------------
def append_and_save(
    new_df: pd.DataFrame,
    filepath: str | Path,
    freq: str = "1h",
    ts_col: str = "timestamp",
) -> pd.DataFrame:
    """Load existing parquet (if any), concat with new_df, dedup, window, save."""
    filepath = Path(filepath)
    if filepath.exists():
        existing = pd.read_parquet(filepath)
        existing = enforce_utc(existing, ts_col)
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df.copy()

    combined = enforce_utc(combined, ts_col)
    n_before = len(combined)
    save_with_window(combined, filepath, freq=freq, ts_col=ts_col)
    combined = combined.sort_values(ts_col).drop_duplicates(subset=[ts_col])
    logger.info(
        f"{filepath.name}: +{len(new_df)} new rows, "
        f"total after dedup={len(combined)} (was {n_before})"
    )
    return combined


# ---------------------------------------------------------------------------
# Timestamp dedup — keep latest on collision
# ---------------------------------------------------------------------------
def dedup_by_timestamp(df: pd.DataFrame, ts_col: str = "timestamp") -> pd.DataFrame:
    """
    Remove duplicate timestamps, keeping the LAST row (by current sort order).
    Sort ascending first so that after concat(old, new) the new data wins.
    """
    if df.empty or ts_col not in df.columns:
        return df
    return (
        df.sort_values(ts_col)
        .drop_duplicates(subset=[ts_col], keep="last")
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# Fix 6: News dedup hash
# ---------------------------------------------------------------------------
def news_hash(title: str, source: str) -> str:
    """Deterministic MD5 hash for news deduplication."""
    raw = (title[:100].lower().strip() + "|" + source.lower()).encode()
    return hashlib.md5(raw).hexdigest()
