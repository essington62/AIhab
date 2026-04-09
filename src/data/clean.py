"""
raw → 02_intermediate transformation layer.
The feature pipeline reads from 02_intermediate/, never from 01_raw/.

Transforms:
  - spot:    btc_1h.parquet → 02_intermediate/spot/btc_1h_clean.parquet
             (adds BB, RSI, MAs)
  - futures: oi_4h.parquet, funding_4h.parquet, taker_4h.parquet
             → forward-filled 1H series (CoinGlass 4h → Opção A ffill)
  - macro:   fred_daily.parquet → fred_daily_clean.parquet (ffill, aligned)

Binance Futures (1h) kept as legacy / backup below but not called by run().
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .utils import enforce_utc, save_with_window

logger = logging.getLogger("data_layer.clean")

RAW = Path("data/01_raw")
INTER = Path("data/02_intermediate")


# ---------------------------------------------------------------------------
# Fix 3: Align to 1H grid
# ---------------------------------------------------------------------------

def align_to_hourly_grid(df: pd.DataFrame, value_cols: list[str]) -> pd.DataFrame:
    """Resample to 1H UTC grid. Gaps stay as NaN (z-score handles them)."""
    df = enforce_utc(df, "timestamp")
    df = df.set_index("timestamp")[value_cols]
    df = df.resample("1h").last()
    df = df.reset_index()
    return df


# ---------------------------------------------------------------------------
# Fix 4: Funding rate forward-fill to 1H
# ---------------------------------------------------------------------------

def funding_to_1h(df: pd.DataFrame) -> pd.DataFrame:
    """Convert 8h funding events to 1H series via forward-fill."""
    df = enforce_utc(df, "timestamp")
    df = df.set_index("timestamp")[["fundingRate"]]
    df = df.resample("1h").ffill()
    df = df.reset_index()
    df["source"] = "binance_futures_funding"
    return df


# ---------------------------------------------------------------------------
# Technical indicators (BB, RSI, MAs) for spot
# ---------------------------------------------------------------------------

def _bollinger_pct(close: pd.Series, window: int = 20) -> pd.Series:
    """BB %B: position within Bollinger Band (0=lower, 1=upper)."""
    ma = close.rolling(window).mean()
    std = close.rolling(window).std()
    upper = ma + 2 * std
    lower = ma - 2 * std
    bb_pct = (close - lower) / (upper - lower + 1e-10)
    return bb_pct.clip(0, 1)


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / (loss + 1e-10)
    return 100 - 100 / (1 + rs)


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    close = df["close"].astype(float)
    df["bb_pct"] = _bollinger_pct(close, 20)
    df["rsi_14"] = _rsi(close, 14)
    df["ma_21"] = close.rolling(21).mean()
    df["ma_50"] = close.rolling(50).mean()
    df["ma_200"] = close.rolling(200).mean()
    return df


# ---------------------------------------------------------------------------
# Clean: Spot
# ---------------------------------------------------------------------------

def clean_spot() -> None:
    src = RAW / "spot" / "btc_1h.parquet"
    dst = INTER / "spot" / "btc_1h_clean.parquet"

    if not src.exists():
        logger.warning(f"clean_spot: {src} not found, skipping")
        return

    df = pd.read_parquet(src)
    df = enforce_utc(df, "timestamp")
    df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"])
    df = add_technical_indicators(df)
    save_with_window(df, dst, freq="1h")
    logger.info(f"clean_spot: {len(df)} rows → {dst}")


# ---------------------------------------------------------------------------
# Clean: CoinGlass Futures 4h → 1h ffill (primary, Opção A)
# ---------------------------------------------------------------------------

def clean_futures_4h_ffill(src_name: str, value_cols: list[str], dst_name: str) -> None:
    """Forward-fill 4h CoinGlass data to 1h grid. Preserves last known value
    across the 3 intermediate hours between 4h candles."""
    src = RAW / "futures" / src_name
    dst = INTER / "futures" / dst_name

    if not src.exists():
        logger.warning(f"clean_futures_4h: {src} not found, skipping")
        return

    df = pd.read_parquet(src)
    df = enforce_utc(df, "timestamp")
    df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"])
    df = df.set_index("timestamp")[value_cols]

    # Resample to 1h grid, forward-fill the 3 empty slots between each 4h candle
    df = df.resample("1h").ffill()
    df = df.reset_index()
    df["source"] = src.stem

    save_with_window(df, dst, freq="1h")
    logger.info(f"clean_futures_4h {src_name}: {len(df)} rows → {dst}")


# ---------------------------------------------------------------------------
# Legacy: Binance Futures 1h (kept as backup — not called by run())
# ---------------------------------------------------------------------------

def clean_futures_1h(filename: str, value_cols: list[str]) -> None:
    src = RAW / "futures" / filename
    dst = INTER / "futures" / filename.replace(".parquet", "_clean.parquet")

    if not src.exists():
        logger.warning(f"clean_futures: {src} not found, skipping")
        return

    df = pd.read_parquet(src)
    df = align_to_hourly_grid(df, value_cols)
    df["source"] = src.stem
    save_with_window(df, dst, freq="1h")
    logger.info(f"clean_futures {filename}: {len(df)} rows → {dst}")


def clean_funding() -> None:
    """Legacy: Binance 8h funding → 1h ffill. Not called by run()."""
    src = RAW / "futures" / "funding_8h.parquet"
    dst = INTER / "futures" / "funding_1h_clean.parquet"

    if not src.exists():
        logger.warning(f"clean_funding: {src} not found, skipping")
        return

    df = pd.read_parquet(src)
    df = funding_to_1h(df)
    save_with_window(df, dst, freq="1h")
    logger.info(f"clean_funding: {len(df)} rows → {dst}")


# ---------------------------------------------------------------------------
# Clean: FRED macro
# ---------------------------------------------------------------------------

def clean_macro() -> None:
    src = RAW / "macro" / "fred_daily.parquet"
    dst = INTER / "macro" / "fred_daily_clean.parquet"

    if not src.exists():
        logger.warning(f"clean_macro: {src} not found, skipping")
        return

    df = pd.read_parquet(src)
    df = enforce_utc(df, "timestamp")
    df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"])

    # Forward-fill all FRED series (weekends / holidays)
    fill_cols = [c for c in df.columns if c not in ("timestamp", "source")]
    df[fill_cols] = df[fill_cols].ffill()

    save_with_window(df, dst, freq="daily")
    logger.info(f"clean_macro: {len(df)} rows → {dst}")


# ---------------------------------------------------------------------------
# Clean: Spot 1D (aggregate 1h → daily OHLCV for R5C HMM)
# ---------------------------------------------------------------------------

def clean_spot_1d() -> None:
    """Aggregate 1h clean spot data to daily OHLCV. Saves btc_1d_clean.parquet."""
    src = INTER / "spot" / "btc_1h_clean.parquet"
    dst = INTER / "spot" / "btc_1d_clean.parquet"

    if not src.exists():
        logger.warning(f"clean_spot_1d: {src} not found, run clean_spot() first")
        return

    df = pd.read_parquet(src)
    df = enforce_utc(df, "timestamp")
    df = df.sort_values("timestamp")

    # Resample 1h → 1D (Binance daily closes at 00:00 UTC)
    df = df.set_index("timestamp")
    daily = pd.DataFrame({
        "open":   df["open"].resample("1D").first(),
        "high":   df["high"].resample("1D").max(),
        "low":    df["low"].resample("1D").min(),
        "close":  df["close"].resample("1D").last(),
        "volume": df["volume"].resample("1D").sum(),
    })
    daily = daily.dropna(subset=["close"])
    daily["source"] = "binance_spot_1h_aggregated"
    daily = daily.reset_index()

    save_with_window(daily, dst, freq="daily")
    logger.info(f"clean_spot_1d: {len(daily)} days → {dst}")


# ---------------------------------------------------------------------------
# Run all
# ---------------------------------------------------------------------------

def run() -> None:
    clean_spot()
    clean_spot_1d()
    # CoinGlass aggregated futures 4h → 1h ffill
    clean_futures_4h_ffill("oi_4h.parquet",      ["open_interest"],              "oi_1h_clean.parquet")
    clean_futures_4h_ffill("funding_4h.parquet",  ["funding_rate"],               "funding_1h_clean.parquet")
    clean_futures_4h_ffill("taker_4h.parquet",    ["buy_volume_usd", "sell_volume_usd", "buy_sell_ratio"], "taker_1h_clean.parquet")
    # L/S account still from Binance 1h
    clean_futures_1h("ls_account_1h.parquet", ["longShortRatio", "longAccount", "shortAccount"])
    clean_futures_1h("ls_position_1h.parquet", ["longShortRatio", "longAccount", "shortAccount"])
    clean_macro()


if __name__ == "__main__":
    run()
