"""
CoinGlass v4 daily ingest.
Writes to: data/01_raw/coinglass/
  - bubble_index_daily.parquet
  - etf_flows_daily.parquet
  - liquidations_4h.parquet    (long/short liquidation USD — 4h bars)

API: open-api-v4.coinglass.com  Header: CG-API-KEY
Credentials: conf/credentials.yml → coinglass_api_key

Confirmed working endpoints (tested 2026-04-08):
  /api/index/bitcoin/bubble-index
  /api/etf/bitcoin/flow-history
  /api/futures/liquidation/history        (Binance BTCUSDT 4h)
  /api/futures/open-interest/history      (available but sourced from Binance)
  /api/futures/funding-rate/history       (available but sourced from Binance)
  /api/futures/v2/taker-buy-sell-volume/history  (available but sourced from Binance)

Stablecoin mcap endpoint (case-sensitive):
  /api/index/stableCoin-marketCap-history
  Response: data.data_list (list of dicts per coin), data.time_list (ms ints)
  stablecoin_mcap_usd = sum(row.values()) for each row in data_list
"""

import logging
from pathlib import Path

import pandas as pd
import yaml

from .utils import append_and_save, enforce_utc, fetch_with_retry

logger = logging.getLogger("data_layer.coinglass")

BASE_URL = "https://open-api-v4.coinglass.com"
CREDENTIALS_PATH = Path("conf/credentials.yml")
RAW_DIR = Path("data/01_raw/coinglass")


def _load_api_key() -> str:
    with open(CREDENTIALS_PATH) as f:
        creds = yaml.safe_load(f)
    key = creds.get("coinglass_api_key", "")
    if not key or key.startswith("YOUR_"):
        raise ValueError("coinglass_api_key not configured in credentials.yml")
    return key


def _get_last_date(filepath: Path) -> pd.Timestamp | None:
    if not filepath.exists():
        return None
    df = pd.read_parquet(filepath)
    df = enforce_utc(df, "timestamp")
    return df["timestamp"].max()


def _headers(api_key: str) -> dict:
    return {"CG-API-KEY": api_key, "Accept": "application/json"}


def _check_response(data: dict, endpoint: str) -> bool:
    """CoinGlass v4 returns code='0' on success."""
    if not isinstance(data, dict):
        return False
    code = str(data.get("code", ""))
    if code != "0":
        logger.error(f"{endpoint}: API error code={code} msg={data.get('msg','')}")
        return False
    return True


# ---------------------------------------------------------------------------
# Bubble Index — /api/index/bitcoin/bubble-index
# Fields: date_string (str), bubble_index (float)
# ---------------------------------------------------------------------------

def fetch_bubble_index(api_key: str) -> pd.DataFrame:
    data = fetch_with_retry(
        f"{BASE_URL}/api/index/bitcoin/bubble-index",
        headers=_headers(api_key),
    )
    if not _check_response(data, "bubble-index"):
        return pd.DataFrame()

    rows = data.get("data", [])
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    if "date_string" not in df.columns or "bubble_index" not in df.columns:
        logger.error(f"bubble-index: unexpected cols={list(df.columns)}")
        return pd.DataFrame()

    df["timestamp"] = pd.to_datetime(df["date_string"], utc=True)
    df = df[["timestamp"]].copy()
    df["bubble_index"] = pd.DataFrame(rows)["bubble_index"].astype(float)
    df["source"] = "coinglass_bubble"
    return df.dropna(subset=["bubble_index"])


# ---------------------------------------------------------------------------
# ETF Flows — /api/etf/bitcoin/flow-history
# Fields: timestamp (ms int), flow_usd (float)
# ---------------------------------------------------------------------------

def fetch_etf_flows(api_key: str) -> pd.DataFrame:
    data = fetch_with_retry(
        f"{BASE_URL}/api/etf/bitcoin/flow-history",
        headers=_headers(api_key),
    )
    if not _check_response(data, "etf/bitcoin/flow-history"):
        return pd.DataFrame()

    rows = data.get("data", [])
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    if "timestamp" not in df.columns or "flow_usd" not in df.columns:
        logger.error(f"etf-flows: unexpected cols={list(df.columns)}")
        return pd.DataFrame()

    # timestamp is ms integer (e.g. 1704931200000)
    df["timestamp"] = pd.to_datetime(df["timestamp"].astype("int64"), unit="ms", utc=True)
    out = df[["timestamp"]].copy()
    out["etf_flow_usd"] = df["flow_usd"].astype(float)
    out["source"] = "coinglass_etf"
    return out.dropna(subset=["etf_flow_usd"])


# ---------------------------------------------------------------------------
# Stablecoin Mcap — /api/index/stableCoin-marketCap-history
# Response: data = {time_list: [ms, ...], data_list: [{USDT: x, USDC: y, ...}, ...]}
# stablecoin_mcap_usd = sum of all stablecoin values per row
# ---------------------------------------------------------------------------

def fetch_stablecoin_mcap(api_key: str) -> pd.DataFrame:
    data = fetch_with_retry(
        f"{BASE_URL}/api/index/stableCoin-marketCap-history",
        headers=_headers(api_key),
    )
    if not _check_response(data, "stableCoin-marketCap-history"):
        return pd.DataFrame()

    payload = data.get("data", {})
    time_list = payload.get("time_list", [])
    data_list = payload.get("data_list", [])

    if not time_list or not data_list or len(time_list) != len(data_list):
        logger.error(f"stablecoin: unexpected payload keys={list(payload.keys())}")
        return pd.DataFrame()

    rows = []
    for ts_ms, coin_dict in zip(time_list, data_list):
        mcap = sum(float(v) for v in coin_dict.values() if v is not None)
        rows.append({
            "timestamp": pd.Timestamp(int(ts_ms), unit="ms", tz="UTC"),
            "stablecoin_mcap_usd": mcap,
            "source": "coinglass_stablecoin",
        })

    df = pd.DataFrame(rows)
    logger.info(f"stablecoin_mcap: {len(df)} rows fetched")
    return df.dropna(subset=["stablecoin_mcap_usd"])


# ---------------------------------------------------------------------------
# Liquidations — /api/futures/liquidation/history
# Fields: time (ms int), long_liquidation_usd, short_liquidation_usd
# Interval: 4h  (max 1080 candles ≈ 6 months)
# ---------------------------------------------------------------------------

def fetch_liquidations(api_key: str) -> pd.DataFrame:
    data = fetch_with_retry(
        f"{BASE_URL}/api/futures/liquidation/history",
        headers=_headers(api_key),
        params={"exchange": "Binance", "symbol": "BTCUSDT", "interval": "4h", "limit": 1080},
    )
    if not _check_response(data, "futures/liquidation/history"):
        return pd.DataFrame()

    rows = data.get("data", [])
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    if "time" not in df.columns:
        logger.error(f"liquidations: unexpected cols={list(df.columns)}")
        return pd.DataFrame()

    df["timestamp"] = pd.to_datetime(df["time"].astype("int64"), unit="ms", utc=True)
    out = df[["timestamp"]].copy()
    out["long_liq_usd"]  = df["long_liquidation_usd"].astype(float)
    out["short_liq_usd"] = df["short_liquidation_usd"].astype(float)
    out["liq_ratio"]     = (out["long_liq_usd"] / (out["long_liq_usd"] + out["short_liq_usd"])).round(4)
    out["source"] = "coinglass_liq"
    return out.dropna(subset=["long_liq_usd"])


# ---------------------------------------------------------------------------
# Order Book — /api/futures/orderbook/ask-bids-history
# Fields: time (ms), bids_usd, bids_quantity, asks_usd, asks_quantity
# Pair-level: exchange=Binance, symbol=BTCUSDT
# ---------------------------------------------------------------------------

def fetch_orderbook(api_key: str) -> pd.DataFrame:
    data = fetch_with_retry(
        f"{BASE_URL}/api/futures/orderbook/ask-bids-history",
        headers=_headers(api_key),
        params={"exchange": "Binance", "symbol": "BTCUSDT", "interval": "4h", "limit": 1080},
    )
    if not _check_response(data, "orderbook/ask-bids-history"):
        return pd.DataFrame()

    rows = data.get("data", [])
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    if "time" not in df.columns:
        logger.error(f"orderbook: unexpected cols={list(df.columns)}")
        return pd.DataFrame()

    df["timestamp"]    = pd.to_datetime(df["time"].astype("int64"), unit="ms", utc=True)
    out                = df[["timestamp"]].copy()
    out["bids_usd"]    = df["bids_usd"].astype(float)
    out["asks_usd"]    = df["asks_usd"].astype(float)
    out["bids_qty"]    = df["bids_quantity"].astype(float)
    out["asks_qty"]    = df["asks_quantity"].astype(float)
    out["bid_ask_ratio"] = (out["bids_usd"] / out["asks_usd"].replace(0, float("nan"))).round(4)
    out["source"]      = "coinglass_ob"
    return out.dropna(subset=["bids_usd"])


# ---------------------------------------------------------------------------
# Order Book Aggregated — /api/futures/orderbook/aggregated-ask-bids-history
# Fields: time (ms), aggregated_bids_usd, aggregated_asks_usd, ...
# Multi-exchange: exchange_list=Binance, symbol=BTC
# ---------------------------------------------------------------------------

def fetch_orderbook_agg(api_key: str) -> pd.DataFrame:
    data = fetch_with_retry(
        f"{BASE_URL}/api/futures/orderbook/aggregated-ask-bids-history",
        headers=_headers(api_key),
        params={"exchange_list": "Binance", "symbol": "BTC", "interval": "4h", "limit": 1080},
    )
    if not _check_response(data, "orderbook/aggregated-ask-bids-history"):
        return pd.DataFrame()

    rows = data.get("data", [])
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    if "time" not in df.columns:
        logger.error(f"orderbook_agg: unexpected cols={list(df.columns)}")
        return pd.DataFrame()

    df["timestamp"]    = pd.to_datetime(df["time"].astype("int64"), unit="ms", utc=True)
    out                = df[["timestamp"]].copy()
    out["bids_usd"]    = df["aggregated_bids_usd"].astype(float)
    out["asks_usd"]    = df["aggregated_asks_usd"].astype(float)
    out["bids_qty"]    = df["aggregated_bids_quantity"].astype(float)
    out["asks_qty"]    = df["aggregated_asks_quantity"].astype(float)
    out["bid_ask_ratio"] = (out["bids_usd"] / out["asks_usd"].replace(0, float("nan"))).round(4)
    out["source"]      = "coinglass_ob_agg"
    return out.dropna(subset=["bids_usd"])


# ---------------------------------------------------------------------------
# Run all
# ---------------------------------------------------------------------------

def run() -> None:
    try:
        api_key = _load_api_key()
    except ValueError as e:
        logger.warning(f"CoinGlass skipped: {e}")
        return

    tasks = [
        ("bubble_index_daily.parquet",    fetch_bubble_index,    "daily"),
        ("etf_flows_daily.parquet",       fetch_etf_flows,       "daily"),
        ("stablecoin_mcap_daily.parquet", fetch_stablecoin_mcap, "daily"),
        ("liquidations_4h.parquet",       fetch_liquidations,    "4h"),
        ("orderbook_4h.parquet",          fetch_orderbook,       "4h"),
        ("orderbook_agg_4h.parquet",      fetch_orderbook_agg,   "4h"),
    ]

    for filename, fetch_fn, freq in tasks:
        filepath = RAW_DIR / filename
        logger.info(f"Fetching {filename}")
        try:
            df = fetch_fn(api_key)
            if df.empty:
                logger.info(f"{filename}: empty response")
                continue
            last_ts = _get_last_date(filepath)
            if last_ts is not None:
                df = df[df["timestamp"] > last_ts]
            if df.empty:
                logger.info(f"{filename}: already up to date")
                continue
            append_and_save(df, filepath, freq=freq)
            logger.info(f"{filename}: +{len(df)} rows")
        except Exception as e:
            logger.error(f"{filename}: fetch failed — {e}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run()
