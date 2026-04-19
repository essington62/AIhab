"""
CoinGlass L/S ratio — historical bootstrap source.

Used ONE-TIME to populate deep historical L/S data (up to ~365 days).
After bootstrap, incremental updates come from Binance (free).

Endpoints (CoinGlass v4, Hobbyist plan):
  /api/futures/top-long-short-account-ratio/history   — whale accounts
  /api/futures/top-long-short-position-ratio/history  — whale positions

Output paths match binance_ls.py convention:
  BTC (legacy):  data/01_raw/futures/ls_account_1h.parquet
  ETH:           data/01_raw/futures/eth_ls_account_1h.parquet

If endpoints are not available on the Hobbyist plan, errors are logged
and empty DataFrames returned (caller handles gracefully).
"""
import logging
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

from src.config import get_credential
from src.data.utils import append_and_save, dedup_by_timestamp, enforce_utc, fetch_with_retry

logger = logging.getLogger("data_layer.coinglass_ls")

BASE_URL = "https://open-api-v4.coinglass.com"
RAW_DIR = Path("data/01_raw/futures")

# L/S endpoints — verified CoinGlass v4 paths
_ENDPOINT_ACCOUNT  = "/api/futures/top-long-short-account-ratio/history"
_ENDPOINT_POSITION = "/api/futures/top-long-short-position-ratio/history"


def _load_api_key() -> str:
    try:
        key = get_credential("coinglass_api_key")
    except KeyError:
        raise ValueError("coinglass_api_key not found in credentials.yml")
    if not key or str(key).startswith("YOUR_"):
        raise ValueError("coinglass_api_key not configured")
    return key


def _headers(api_key: str) -> dict:
    return {"CG-API-KEY": api_key, "accept": "application/json"}


def _check_response(data: dict, endpoint: str) -> bool:
    if not isinstance(data, dict):
        return False
    code = str(data.get("code", ""))
    if code != "0":
        logger.warning(
            f"{endpoint}: API code={code} msg={data.get('msg', '')} — "
            f"endpoint may not be on Hobbyist plan"
        )
        return False
    return True


def _ls_raw_path(symbol: str, base_name: str) -> Path:
    if symbol.upper() == "BTC":
        return RAW_DIR / base_name
    return RAW_DIR / f"{symbol.lower()}_{base_name}"


def _fetch_ls_paginated(
    endpoint: str,
    symbol: str,
    api_key: str,
    start_time: Optional[datetime] = None,
    interval: str = "1h",
) -> list[dict]:
    """
    Paginated CoinGlass L/S fetch using forward-advancing startTime.
    Returns list of raw record dicts. Empty list if endpoint unavailable.
    """
    pair = f"{symbol.upper()}USDT"

    if start_time is None:
        start_time = datetime.now(timezone.utc) - timedelta(days=180)

    cur_start_ms = int(start_time.timestamp() * 1000)
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    headers = _headers(api_key)

    all_records: list[dict] = []
    limit = 500
    max_pages = 100  # safety — 100 × 500 × 1h = 50000h ≈ 5.7 years

    for page in range(max_pages):
        params = {
            "symbol": pair,
            "interval": interval,
            "startTime": cur_start_ms,
            "endTime": now_ms,
            "limit": limit,
        }
        try:
            data = fetch_with_retry(
                f"{BASE_URL}{endpoint}",
                params=params,
                headers=headers,
            )
        except Exception as e:
            logger.error(f"CoinGlass L/S fetch failed (page {page}): {e}")
            break

        if not _check_response(data, endpoint):
            break

        records = data.get("data", [])
        if not records:
            logger.debug(f"Page {page}: no more records")
            break

        all_records.extend(records)
        logger.debug(f"Page {page}: +{len(records)} records (total {len(all_records)})")

        if len(records) < limit:
            break

        # Advance startTime to newest received + 1ms
        latest_ms = max(
            int(r.get("time", r.get("timestamp", cur_start_ms))) for r in records
        )
        next_start = latest_ms + 1
        if next_start <= cur_start_ms:
            break
        cur_start_ms = next_start

        time.sleep(0.3)  # rate limit courtesy

    return all_records


def _normalize_ls(records: list[dict], source_tag: str) -> pd.DataFrame:
    """
    Normalize CoinGlass L/S records to AI.hab schema.
    Schema (matches binance_ls.py):
        timestamp, longShortRatio, longAccount, shortAccount, source
    """
    rows = []
    for r in records:
        ts_ms = r.get("time") or r.get("timestamp") or r.get("createTime")
        if not ts_ms:
            continue
        try:
            rows.append({
                "timestamp":     pd.Timestamp(int(ts_ms), unit="ms", tz="UTC"),
                "longShortRatio": float(r.get("longShortRatio", 0) or 0),
                "longAccount":    float(r.get("longAccount", r.get("long", 0)) or 0),
                "shortAccount":   float(r.get("shortAccount", r.get("short", 0)) or 0),
                "source":         source_tag,
            })
        except (ValueError, TypeError):
            continue

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df.dropna(subset=["longShortRatio"])


def fetch_ls_accounts_cg(
    symbol: str,
    start_time: Optional[datetime] = None,
) -> pd.DataFrame:
    """
    Fetch L/S account ratio history from CoinGlass.
    Returns normalized DataFrame or empty DataFrame if unavailable.
    """
    try:
        api_key = _load_api_key()
    except ValueError as e:
        logger.warning(f"fetch_ls_accounts_cg: {e}")
        return pd.DataFrame()

    logger.info(f"Fetching {symbol} L/S accounts from CoinGlass...")
    records = _fetch_ls_paginated(
        _ENDPOINT_ACCOUNT, symbol, api_key, start_time
    )
    if not records:
        logger.warning(f"{symbol} L/S accounts: no data from CoinGlass")
        return pd.DataFrame()

    df = _normalize_ls(records, "coinglass_ls_account")
    logger.info(f"{symbol} L/S accounts: {len(df)} records")
    return df


def fetch_ls_positions_cg(
    symbol: str,
    start_time: Optional[datetime] = None,
) -> pd.DataFrame:
    """
    Fetch L/S position ratio history from CoinGlass.
    Returns normalized DataFrame or empty DataFrame if unavailable.
    """
    try:
        api_key = _load_api_key()
    except ValueError as e:
        logger.warning(f"fetch_ls_positions_cg: {e}")
        return pd.DataFrame()

    logger.info(f"Fetching {symbol} L/S positions from CoinGlass...")
    records = _fetch_ls_paginated(
        _ENDPOINT_POSITION, symbol, api_key, start_time
    )
    if not records:
        logger.warning(f"{symbol} L/S positions: no data from CoinGlass")
        return pd.DataFrame()

    df = _normalize_ls(records, "coinglass_ls_position")
    logger.info(f"{symbol} L/S positions: {len(df)} records")
    return df


def bootstrap_ls_to_parquet(
    symbol: str,
    start_time: Optional[datetime] = None,
) -> dict:
    """
    Bootstrap L/S accounts + positions from CoinGlass and merge with any
    existing parquets (idempotent, dedup by timestamp).

    Returns status dict:
        {"accounts": {"rows": N, "path": ...} | {"error": ...},
         "positions": {"rows": N, "path": ...} | {"error": ...}}
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    result: dict = {}

    tasks = [
        ("accounts", "ls_account_1h.parquet", fetch_ls_accounts_cg),
        ("positions", "ls_position_1h.parquet", fetch_ls_positions_cg),
    ]
    for key, base_name, fetch_fn in tasks:
        filepath = _ls_raw_path(symbol, base_name)
        logger.info(f"Bootstrap {symbol} L/S {key} → {filepath}")

        df_new = fetch_fn(symbol, start_time)

        if df_new.empty:
            logger.error(f"❌ {symbol} L/S {key}: empty result")
            result[key] = {"error": "empty result from CoinGlass"}
            continue

        # Merge with existing data (CoinGlass bootstrap + future Binance incremental)
        if filepath.exists():
            df_existing = pd.read_parquet(filepath)
            df_existing = enforce_utc(df_existing, "timestamp")
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
            df_combined = dedup_by_timestamp(df_combined)
        else:
            df_combined = df_new

        # Save — use save_with_window via append_and_save logic
        from src.data.utils import save_with_window
        save_with_window(df_combined, filepath, freq="1h")

        logger.info(f"✅ {symbol} L/S {key}: {len(df_combined)} rows → {filepath.name}")
        result[key] = {"rows": len(df_combined), "path": str(filepath)}

    return result
