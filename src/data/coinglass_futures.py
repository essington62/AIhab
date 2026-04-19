"""
CoinGlass v4 futures ingest — cross-exchange aggregated derivatives.
Writes to: data/01_raw/futures/
  BTC (backward compat):  oi_4h.parquet, funding_4h.parquet, taker_4h.parquet
  ETH (and other coins):  eth_oi_4h.parquet, eth_funding_4h.parquet, eth_taker_4h.parquet

Public API (multi-symbol):
  fetch_oi_4h(symbol, start_time)
  fetch_funding_4h(symbol, start_time)
  fetch_taker_4h(symbol, start_time)
  run()  — backward compat BTC-only

Replaces: src/data/binance_futures.py (Binance-only, 1h)
Rationale: CoinGlass aggregated OI = $51.8B vs Binance-only $6.5B (12%)

API: open-api-v4.coinglass.com  Header: CG-API-KEY
Credentials: conf/credentials.yml → coinglass_api_key

Note on history depth: CoinGlass 4h endpoints return last N candles (limit=1080 ≈ 180 days).
No startTime pagination available — bootstrap limited to ~180 days for OI/Funding/Taker.

Confirmed field schemas (tested 2026-04-08):
  OI agg:       time(ms), open, high, low, close  [close = OI USD]
  Funding OIw:  time(ms), open, high, low, close  [close = funding rate str]
  Taker v2:     time(ms), taker_buy_volume_usd, taker_sell_volume_usd
"""

import logging
from pathlib import Path

import pandas as pd
import yaml

from .utils import append_and_save, enforce_utc, fetch_with_retry

logger = logging.getLogger("data_layer.coinglass_futures")

BASE_URL = "https://open-api-v4.coinglass.com"
CREDENTIALS_PATH = Path("conf/credentials.yml")
RAW_DIR = Path("data/01_raw/futures")

_EXPECTED_SOURCES = {
    "oi_4h.parquet": "coinglass_oi_agg",
    "funding_4h.parquet": "coinglass_funding_oi",
    "taker_4h.parquet": "coinglass_taker",
}


def _load_api_key() -> str:
    with open(CREDENTIALS_PATH) as f:
        creds = yaml.safe_load(f)
    key = creds.get("coinglass_api_key", "")
    if not key or key.startswith("YOUR_"):
        raise ValueError("coinglass_api_key not configured in credentials.yml")
    return key


def _headers(api_key: str) -> dict:
    return {"CG-API-KEY": api_key, "Accept": "application/json"}


def _check_response(data: dict, endpoint: str) -> bool:
    if not isinstance(data, dict):
        return False
    code = str(data.get("code", ""))
    if code != "0":
        logger.error(f"{endpoint}: API error code={code} msg={data.get('msg', '')}")
        return False
    return True


def _get_last_ts(filepath: Path) -> pd.Timestamp | None:
    if not filepath.exists():
        return None
    df = pd.read_parquet(filepath)
    df = enforce_utc(df, "timestamp")
    return df["timestamp"].max()


def _source_changed(filepath: Path, expected_source: str) -> bool:
    """Return True if file exists but was written by a different source."""
    if not filepath.exists():
        return False
    try:
        df = pd.read_parquet(filepath)
        if "source" in df.columns and len(df) > 0:
            return str(df["source"].iloc[0]) != expected_source
    except Exception:
        pass
    return False


# ---------------------------------------------------------------------------
# OI Aggregated — /api/futures/open-interest/aggregated-history
# symbol=BTC (no exchange), 4h, limit=1080
# Fields: time (ms), open, high, low, close  [close = aggregated OI USD]
# ---------------------------------------------------------------------------

def fetch_oi_aggregated(api_key: str) -> pd.DataFrame:
    data = fetch_with_retry(
        f"{BASE_URL}/api/futures/open-interest/aggregated-history",
        headers=_headers(api_key),
        params={"symbol": "BTC", "interval": "4h", "limit": 1080, "unit": "usd"},
    )
    if not _check_response(data, "oi/aggregated-history"):
        return pd.DataFrame()

    rows = data.get("data", [])
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    if "time" not in df.columns or "close" not in df.columns:
        logger.error(f"oi_agg: unexpected cols={list(df.columns)}")
        return pd.DataFrame()

    out = pd.DataFrame()
    out["timestamp"] = pd.to_datetime(df["time"].astype("int64"), unit="ms", utc=True)
    out["open_interest"] = df["close"].astype(float)
    out["source"] = "coinglass_oi_agg"
    return out.dropna(subset=["open_interest"])


# ---------------------------------------------------------------------------
# Funding Rate OI-Weighted — /api/futures/funding-rate/oi-weight-history
# symbol=BTC (no exchange), 4h, limit=1080
# Fields: time (ms), open, high, low, close  [close = funding rate str]
# ---------------------------------------------------------------------------

def fetch_funding_oi_weighted(api_key: str) -> pd.DataFrame:
    data = fetch_with_retry(
        f"{BASE_URL}/api/futures/funding-rate/oi-weight-history",
        headers=_headers(api_key),
        params={"symbol": "BTC", "interval": "4h", "limit": 1080},
    )
    if not _check_response(data, "funding-rate/oi-weight-history"):
        return pd.DataFrame()

    rows = data.get("data", [])
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    if "time" not in df.columns or "close" not in df.columns:
        logger.error(f"funding_oi: unexpected cols={list(df.columns)}")
        return pd.DataFrame()

    out = pd.DataFrame()
    out["timestamp"] = pd.to_datetime(df["time"].astype("int64"), unit="ms", utc=True)
    out["funding_rate"] = pd.to_numeric(df["close"], errors="coerce")
    out["source"] = "coinglass_funding_oi"
    return out.dropna(subset=["funding_rate"])


# ---------------------------------------------------------------------------
# Taker Buy/Sell — /api/futures/v2/taker-buy-sell-volume/history
# exchange=Binance, symbol=BTCUSDT, 4h, limit=1080
# Fields: time (ms), taker_buy_volume_usd, taker_sell_volume_usd
# (cross-exchange taker not available on Hobbyist plan)
# ---------------------------------------------------------------------------

def fetch_taker(api_key: str) -> pd.DataFrame:
    data = fetch_with_retry(
        f"{BASE_URL}/api/futures/v2/taker-buy-sell-volume/history",
        headers=_headers(api_key),
        params={"exchange": "Binance", "symbol": "BTCUSDT", "interval": "4h", "limit": 1080},
    )
    if not _check_response(data, "taker-buy-sell-volume/history"):
        return pd.DataFrame()

    rows = data.get("data", [])
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    if "time" not in df.columns:
        logger.error(f"taker: unexpected cols={list(df.columns)}")
        return pd.DataFrame()

    out = pd.DataFrame()
    out["timestamp"] = pd.to_datetime(df["time"].astype("int64"), unit="ms", utc=True)
    out["buy_volume_usd"] = pd.to_numeric(df.get("taker_buy_volume_usd"), errors="coerce")
    out["sell_volume_usd"] = pd.to_numeric(df.get("taker_sell_volume_usd"), errors="coerce")

    total = out["buy_volume_usd"] + out["sell_volume_usd"]
    out["buy_sell_ratio"] = (out["buy_volume_usd"] / total.replace(0, float("nan"))).round(6)
    out["source"] = "coinglass_taker"
    return out.dropna(subset=["buy_volume_usd"])


# ---------------------------------------------------------------------------
# Run all
# ---------------------------------------------------------------------------

def run() -> None:
    try:
        api_key = _load_api_key()
    except ValueError as e:
        logger.warning(f"CoinGlass futures skipped: {e}")
        return

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    tasks = [
        ("oi_4h.parquet",      fetch_oi_aggregated,     "4h"),
        ("funding_4h.parquet", fetch_funding_oi_weighted, "4h"),
        ("taker_4h.parquet",   fetch_taker,              "4h"),
    ]

    for filename, fetch_fn, freq in tasks:
        filepath = RAW_DIR / filename
        expected_src = _EXPECTED_SOURCES[filename]
        logger.info(f"Fetching {filename}")
        try:
            # Source change detection: clear if previous data is from Binance
            if _source_changed(filepath, expected_src):
                logger.info(f"{filename}: source changed → clearing old Binance data")
                filepath.unlink()

            df = fetch_fn(api_key)
            if df.empty:
                logger.info(f"{filename}: empty response")
                continue

            last_ts = _get_last_ts(filepath)
            if last_ts is not None:
                df = df[df["timestamp"] > last_ts]

            if df.empty:
                logger.info(f"{filename}: already up to date")
                continue

            append_and_save(df, filepath, freq=freq)
            logger.info(f"{filename}: +{len(df)} rows")
        except Exception as e:
            logger.error(f"{filename}: fetch failed — {e}")


# ---------------------------------------------------------------------------
# Multi-symbol public API
# ---------------------------------------------------------------------------

def _futures_path(symbol: str, base_name: str) -> Path:
    """BTC keeps legacy name (no prefix); other symbols get symbol prefix."""
    if symbol.upper() == "BTC":
        return RAW_DIR / base_name
    return RAW_DIR / f"{symbol.lower()}_{base_name}"


def fetch_oi_4h(symbol: str = "BTC", start_time=None) -> None:
    """
    Fetch cross-exchange aggregated OI 4h for any symbol.
    Output: data/01_raw/futures/{symbol}_oi_4h.parquet  (BTC: oi_4h.parquet)
    Note: CoinGlass limit=1080 ≈ last 180 days; no startTime pagination.
    """
    try:
        api_key = _load_api_key()
    except ValueError as e:
        logger.warning(f"fetch_oi_4h({symbol}) skipped: {e}")
        return

    filepath = _futures_path(symbol, "oi_4h.parquet")
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    logger.info(f"Fetching {symbol} OI 4h")
    try:
        data = fetch_with_retry(
            f"{BASE_URL}/api/futures/open-interest/aggregated-history",
            headers=_headers(api_key),
            params={"symbol": symbol.upper(), "interval": "4h", "limit": 1080, "unit": "usd"},
        )
        if not _check_response(data, f"oi/{symbol}"):
            return

        rows = data.get("data", [])
        if not rows:
            logger.info(f"{symbol} OI: empty response")
            return

        df = pd.DataFrame(rows)
        out = pd.DataFrame()
        out["timestamp"] = pd.to_datetime(df["time"].astype("int64"), unit="ms", utc=True)
        out["open_interest"] = df["close"].astype(float)
        out["source"] = f"coinglass_oi_agg_{symbol.lower()}"
        out = out.dropna(subset=["open_interest"])

        last_ts = _get_last_ts(filepath)
        if last_ts is not None:
            out = out[out["timestamp"] > last_ts]

        if out.empty:
            logger.info(f"{symbol} OI: already up to date")
            return

        append_and_save(out, filepath, freq="4h")
        logger.info(f"{symbol} OI 4h: +{len(out)} rows → {filepath}")
    except Exception as e:
        logger.error(f"{symbol} OI 4h: failed — {e}")


def fetch_funding_4h(symbol: str = "BTC", start_time=None) -> None:
    """
    Fetch OI-weighted funding rate 4h for any symbol.
    Output: data/01_raw/futures/{symbol}_funding_4h.parquet  (BTC: funding_4h.parquet)
    """
    try:
        api_key = _load_api_key()
    except ValueError as e:
        logger.warning(f"fetch_funding_4h({symbol}) skipped: {e}")
        return

    filepath = _futures_path(symbol, "funding_4h.parquet")
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    logger.info(f"Fetching {symbol} Funding 4h")
    try:
        data = fetch_with_retry(
            f"{BASE_URL}/api/futures/funding-rate/oi-weight-history",
            headers=_headers(api_key),
            params={"symbol": symbol.upper(), "interval": "4h", "limit": 1080},
        )
        if not _check_response(data, f"funding/{symbol}"):
            return

        rows = data.get("data", [])
        if not rows:
            logger.info(f"{symbol} Funding: empty response")
            return

        df = pd.DataFrame(rows)
        out = pd.DataFrame()
        out["timestamp"] = pd.to_datetime(df["time"].astype("int64"), unit="ms", utc=True)
        out["funding_rate"] = pd.to_numeric(df["close"], errors="coerce")
        out["source"] = f"coinglass_funding_oi_{symbol.lower()}"
        out = out.dropna(subset=["funding_rate"])

        last_ts = _get_last_ts(filepath)
        if last_ts is not None:
            out = out[out["timestamp"] > last_ts]

        if out.empty:
            logger.info(f"{symbol} Funding: already up to date")
            return

        append_and_save(out, filepath, freq="4h")
        logger.info(f"{symbol} Funding 4h: +{len(out)} rows → {filepath}")
    except Exception as e:
        logger.error(f"{symbol} Funding 4h: failed — {e}")


def fetch_taker_4h(symbol: str = "BTC", start_time=None) -> None:
    """
    Fetch Binance taker buy/sell volume 4h for any symbol.
    Output: data/01_raw/futures/{symbol}_taker_4h.parquet  (BTC: taker_4h.parquet)
    """
    try:
        api_key = _load_api_key()
    except ValueError as e:
        logger.warning(f"fetch_taker_4h({symbol}) skipped: {e}")
        return

    filepath = _futures_path(symbol, "taker_4h.parquet")
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    pair = f"{symbol.upper()}USDT"
    logger.info(f"Fetching {symbol} Taker 4h")
    try:
        data = fetch_with_retry(
            f"{BASE_URL}/api/futures/v2/taker-buy-sell-volume/history",
            headers=_headers(api_key),
            params={"exchange": "Binance", "symbol": pair, "interval": "4h", "limit": 1080},
        )
        if not _check_response(data, f"taker/{symbol}"):
            logger.warning(
                f"{symbol} Taker: API error — may not be available on current plan"
            )
            return

        rows = data.get("data", [])
        if not rows:
            logger.info(f"{symbol} Taker: empty response")
            return

        df = pd.DataFrame(rows)
        out = pd.DataFrame()
        out["timestamp"] = pd.to_datetime(df["time"].astype("int64"), unit="ms", utc=True)
        out["buy_volume_usd"] = pd.to_numeric(df.get("taker_buy_volume_usd"), errors="coerce")
        out["sell_volume_usd"] = pd.to_numeric(df.get("taker_sell_volume_usd"), errors="coerce")
        total = out["buy_volume_usd"] + out["sell_volume_usd"]
        out["buy_sell_ratio"] = (out["buy_volume_usd"] / total.replace(0, float("nan"))).round(6)
        out["source"] = f"coinglass_taker_{symbol.lower()}"
        out = out.dropna(subset=["buy_volume_usd"])

        last_ts = _get_last_ts(filepath)
        if last_ts is not None:
            out = out[out["timestamp"] > last_ts]

        if out.empty:
            logger.info(f"{symbol} Taker: already up to date")
            return

        append_and_save(out, filepath, freq="4h")
        logger.info(f"{symbol} Taker 4h: +{len(out)} rows → {filepath}")
    except Exception as e:
        logger.error(f"{symbol} Taker 4h: failed — {e}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run()
