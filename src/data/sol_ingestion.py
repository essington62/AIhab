"""SOL data ingestion — Binance spot 1h + CoinGlass OI/Taker 4h."""
import logging
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[2]
SPOT_PATH = ROOT / "data/01_raw/spot/sol_1h.parquet"

logger = logging.getLogger("sol_ingestion")


def ingest_sol_spot_1h() -> pd.DataFrame:
    """Incremental ingest of SOL/USDT 1h candles from Binance."""
    url = "https://api.binance.com/api/v3/klines"

    existing = pd.DataFrame()
    if SPOT_PATH.exists():
        existing = pd.read_parquet(SPOT_PATH)
        existing["timestamp"] = pd.to_datetime(existing["timestamp"], utc=True)

    params = {"symbol": "SOLUSDT", "interval": "1h", "limit": 1000}
    r = requests.get(url, params=params, timeout=10)
    r.raise_for_status()

    df = pd.DataFrame(r.json(), columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades", "taker_buy_base",
        "taker_buy_quote", "ignore",
    ])
    df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    df = df[["timestamp", "open", "high", "low", "close", "volume"]]

    combined = (
        pd.concat([existing, df])
        .drop_duplicates(subset=["timestamp"], keep="last")
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    SPOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(SPOT_PATH, index=False)
    logger.info(f"SOL spot 1h: {len(combined)} rows → {SPOT_PATH}")
    return combined


def ingest_sol_derivatives() -> None:
    """Incremental ingest of SOL OI + Taker 4h from CoinGlass."""
    try:
        from src.data.coinglass_futures import fetch_oi_4h, fetch_taker_4h
        fetch_oi_4h("SOL")
        fetch_taker_4h("SOL")
    except Exception as e:
        logger.error(f"SOL derivatives ingest failed: {e}")


def run() -> None:
    ingest_sol_spot_1h()
    ingest_sol_derivatives()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run()
