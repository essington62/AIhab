"""ETH data ingestion — Binance spot 1h."""
import logging
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[2]
OUT_PATH = ROOT / "data/01_raw/spot/eth_1h.parquet"

logger = logging.getLogger("eth_ingestion")


def ingest_eth_spot_1h(hours_back: int = 168) -> pd.DataFrame:
    """Ingest last N hours of ETH/USDT 1h candles from Binance."""
    symbol = "ETHUSDT"
    interval = "1h"
    limit = 1000

    url = "https://api.binance.com/api/v3/klines"

    # Load existing
    if OUT_PATH.exists():
        existing = pd.read_parquet(OUT_PATH)
        existing["timestamp"] = pd.to_datetime(existing["timestamp"], utc=True)
    else:
        existing = pd.DataFrame()

    params = {"symbol": symbol, "interval": interval, "limit": limit}
    r = requests.get(url, params=params, timeout=10)
    r.raise_for_status()
    data = r.json()

    df = pd.DataFrame(data, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades", "taker_buy_base",
        "taker_buy_quote", "ignore",
    ])

    df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["open"] = df["open"].astype(float)
    df["high"] = df["high"].astype(float)
    df["low"] = df["low"].astype(float)
    df["close"] = df["close"].astype(float)
    df["volume"] = df["volume"].astype(float)
    df = df[["timestamp", "open", "high", "low", "close", "volume"]]

    if not existing.empty:
        combined = pd.concat([existing, df])
        combined = combined.drop_duplicates(subset=["timestamp"], keep="last")
        combined = combined.sort_values("timestamp").reset_index(drop=True)
    else:
        combined = df.sort_values("timestamp").reset_index(drop=True)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(OUT_PATH, index=False)

    logger.info(f"ETH spot 1h: {len(combined)} rows saved to {OUT_PATH}")
    return combined


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ingest_eth_spot_1h()
