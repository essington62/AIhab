"""
Binance Futures L/S ratios — top traders only (account + position).
Writes to: data/01_raw/futures/
  BTC (backward compat):  ls_account_1h.parquet, ls_position_1h.parquet
  ETH (and other coins):  eth_ls_account_1h.parquet, eth_ls_position_1h.parquet

Public API (multi-symbol):
  fetch_ls_accounts(symbol, start_time)  — batched, bootstrap-capable
  fetch_ls_positions(symbol, start_time) — same
  run()                                  — backward compat BTC-only

Endpoints:
  GET /futures/data/topLongShortAccountRatio
  GET /futures/data/topLongShortPositionRatio
  params: symbol=BTCUSDT, period=1h, limit=500

Note: CoinGlass does not carry these endpoints — Binance is the only source.
Max history per request: 500 hours (~21 days). Run hourly to accumulate.
Binance only keeps ~3 months of L/S history — bootstrap may get partial data.
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


# ---------------------------------------------------------------------------
# Multi-symbol public API
# ---------------------------------------------------------------------------

def _ls_path(symbol: str, base_name: str) -> Path:
    """BTC keeps legacy name (no prefix); other symbols get symbol prefix."""
    if symbol.upper() == "BTC":
        return RAW_DIR / base_name
    return RAW_DIR / f"{symbol.lower()}_{base_name}"


def _fetch_ls_batched(
    endpoint: str,
    symbol: str,
    source: str,
    filepath: Path,
    start_time=None,
) -> None:
    """
    Paginated L/S fetch with startTime. Loops in batches of LIMIT=500.
    Binance history is limited (~3 months) so bootstrap may get partial data.
    """
    pair = f"{symbol.upper()}USDT"

    last_ts = None
    if filepath.exists():
        df_ex = pd.read_parquet(filepath)
        df_ex = enforce_utc(df_ex, "timestamp")
        last_ts = df_ex["timestamp"].max()

    if last_ts is not None:
        cur_ms = int(last_ts.timestamp() * 1000) + 1
        logger.info(f"{symbol} {source}: incremental from {last_ts}")
    elif start_time is not None:
        ts = pd.Timestamp(start_time)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        cur_ms = int(ts.timestamp() * 1000)
        logger.info(f"{symbol} {source}: bootstrap from {ts}")
    else:
        cur_ms = int((pd.Timestamp.utcnow() - pd.Timedelta(days=21)).timestamp() * 1000)
        logger.info(f"{symbol} {source}: default 21 days")

    now_ms = int(pd.Timestamp.utcnow().timestamp() * 1000)
    all_batches: list[pd.DataFrame] = []

    while cur_ms < now_ms:
        data = fetch_with_retry(
            f"{BASE_URL}{endpoint}",
            params={"symbol": pair, "period": PERIOD, "limit": LIMIT, "startTime": cur_ms},
        )
        batch = _parse_ls(data or [], source)
        if batch.empty:
            break
        all_batches.append(batch)
        if len(batch) < LIMIT:
            break
        cur_ms = int(batch["timestamp"].max().timestamp() * 1000) + 1

    if not all_batches:
        logger.info(f"{symbol} {source}: no new data")
        return

    df = pd.concat(all_batches, ignore_index=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    append_and_save(df, filepath, freq="1h")
    logger.info(f"{symbol} {source}: +{len(df)} rows → {filepath}")


def fetch_ls_accounts(symbol: str = "BTC", start_time=None) -> None:
    """
    Fetch top L/S account ratio 1h for any symbol. Loops in batches.
    Output: data/01_raw/futures/{symbol}_ls_account_1h.parquet  (BTC: ls_account_1h.parquet)
    """
    filepath = _ls_path(symbol, "ls_account_1h.parquet")
    try:
        _fetch_ls_batched(
            "/futures/data/topLongShortAccountRatio",
            symbol,
            "binance_ls_account_top",
            filepath,
            start_time,
        )
    except Exception as e:
        logger.error(f"{symbol} L/S accounts: failed — {e}")


def fetch_ls_positions(symbol: str = "BTC", start_time=None) -> None:
    """
    Fetch top L/S position ratio 1h for any symbol. Loops in batches.
    Output: data/01_raw/futures/{symbol}_ls_position_1h.parquet  (BTC: ls_position_1h.parquet)
    """
    filepath = _ls_path(symbol, "ls_position_1h.parquet")
    try:
        _fetch_ls_batched(
            "/futures/data/topLongShortPositionRatio",
            symbol,
            "binance_ls_position_top",
            filepath,
            start_time,
        )
    except Exception as e:
        logger.error(f"{symbol} L/S positions: failed — {e}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run()
