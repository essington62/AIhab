"""
Download ETH + SOL histórico de data.binance.vision (1h, 7 meses).
V2: detecta unit (ms vs us) automaticamente.
"""
import io
import logging
import zipfile
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("download_history")

BASE_URL = "https://data.binance.vision/data/spot/monthly/klines"
OUT_DIR = Path("/tmp")
TARGET_COLS = ["timestamp", "open", "high", "low", "close", "volume"]


def generate_months_back(n_months=7):
    today = datetime.utcnow()
    months = []
    year, month = today.year, today.month
    for _ in range(n_months):
        months.append((year, month))
        month -= 1
        if month == 0:
            month = 12
            year -= 1
    return list(reversed(months))


def download_monthly_zip(symbol, interval, year, month):
    filename = f"{symbol}-{interval}-{year}-{month:02d}.zip"
    url = f"{BASE_URL}/{symbol}/{interval}/{filename}"
    logger.info(f"Fetching: {filename}")
    try:
        r = requests.get(url, timeout=30)
        if r.status_code == 404:
            logger.warning(f"  Not found (current month)")
            return None
        r.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
            with zf.open(zf.namelist()[0]) as f:
                df = pd.read_csv(f, header=None, names=[
                    "open_time", "open", "high", "low", "close", "volume",
                    "close_time", "quote_volume", "trades",
                    "taker_buy_base", "taker_buy_quote", "ignore",
                ])
        logger.info(f"  OK: {len(df)} rows")
        return df
    except Exception as e:
        logger.error(f"  Failed: {e}")
        return None


def fetch_recent_klines_api(symbol, interval="1h", limit=1000):
    url = "https://api.binance.com/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    logger.info(f"API fetch: {symbol}")
    r = requests.get(url, params=params, timeout=10)
    r.raise_for_status()
    data = r.json()
    df = pd.DataFrame(data, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades",
        "taker_buy_base", "taker_buy_quote", "ignore",
    ])
    logger.info(f"  API: {len(df)} rows")
    return df


def detect_time_unit(sample):
    n = len(str(int(sample)))
    if n <= 13: return "ms"
    if n <= 16: return "us"
    return "ns"


def convert_to_aihab_schema(df):
    df = df.copy()
    sample = df["open_time"].iloc[0]
    unit = detect_time_unit(sample)
    logger.info(f"  unit={unit} (sample: {sample})")
    df["timestamp"] = pd.to_datetime(df["open_time"], unit=unit, utc=True)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    return df[TARGET_COLS]


def download_symbol_full_history(symbol, output_path, n_months=7):
    logger.info("=" * 60)
    logger.info(f"Downloading {symbol} — {n_months} months")
    logger.info("=" * 60)
    
    all_dfs = []
    for year, month in generate_months_back(n_months):
        df_month = download_monthly_zip(symbol, "1h", year, month)
        if df_month is not None:
            try:
                df_month = convert_to_aihab_schema(df_month)
                all_dfs.append(df_month)
            except Exception as e:
                logger.error(f"  Conversion failed {year}-{month:02d}: {e}")
    
    try:
        df_api = fetch_recent_klines_api(symbol, "1h", 1000)
        df_api = convert_to_aihab_schema(df_api)
        all_dfs.append(df_api)
    except Exception as e:
        logger.warning(f"API fallback failed: {e}")
    
    if not all_dfs:
        logger.error(f"Nenhum dado para {symbol}!")
        return
    
    df = pd.concat(all_dfs, ignore_index=True)
    df = df.drop_duplicates(subset=["timestamp"], keep="last")
    df = df.sort_values("timestamp").reset_index(drop=True)
    
    logger.info(f"Total: {len(df):,} rows")
    logger.info(f"Period: {df['timestamp'].min()} → {df['timestamp'].max()}")
    span = (df["timestamp"].max() - df["timestamp"].min()).days
    logger.info(f"Span: {span} days")
    
    assert span >= 60, f"INSUFICIENTE: {span} dias"
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    logger.info(f"✅ Saved: {output_path} ({output_path.stat().st_size / 1024:.1f} KB)")


def main():
    download_symbol_full_history("ETHUSDT", OUT_DIR / "eth_1h.parquet", 7)
    print()
    download_symbol_full_history("SOLUSDT", OUT_DIR / "sol_1h.parquet", 7)
    print(f"\n✅ COMPLETO — arquivos em {OUT_DIR}")


if __name__ == "__main__":
    main()
