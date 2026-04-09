"""
src/features/technical.py — Technical indicators (BB, RSI, MAs).

Reads: 02_intermediate/spot/btc_1h_clean.parquet (already has bb_pct, rsi_14, MAs
       computed by clean.py — this module exposes them as a typed dict for the
       trading pipeline and can recompute if needed).

Entry point: get_latest_technical() → dict with current values.
             run() → recomputes and saves back to clean parquet (if raw updated).
"""

import logging

import numpy as np
import pandas as pd

from src.config import get_params, get_path

logger = logging.getLogger("features.technical")


# ---------------------------------------------------------------------------
# Indicator computation (standalone, no I/O)
# ---------------------------------------------------------------------------

def compute_bollinger_pct(close: pd.Series, window: int = 20, std: float = 2.0) -> pd.Series:
    """BB %B: 0 = lower band, 1 = upper band. Clipped [0, 1]."""
    ma = close.rolling(window).mean()
    std_dev = close.rolling(window).std()
    upper = ma + std * std_dev
    lower = ma - std * std_dev
    pct = (close - lower) / (upper - lower).replace(0, np.nan)
    return pct.clip(0, 1)


def compute_rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def compute_ma(close: pd.Series, window: int) -> pd.Series:
    return close.rolling(window).mean()


def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add bb_pct, rsi_14, and MAs to a OHLCV DataFrame. Returns modified copy."""
    params = get_params()["technical"]
    close = df["close"].astype(float)

    df = df.copy()
    df["bb_pct"] = compute_bollinger_pct(close, params["bb_window"], params["bb_std"])
    df["rsi_14"] = compute_rsi(close, params["rsi_window"])
    for w in params["ma_windows"]:
        df[f"ma_{w}"] = compute_ma(close, w)

    return df


# ---------------------------------------------------------------------------
# Latest-value extraction (used by paper_trader each cycle)
# ---------------------------------------------------------------------------

def get_latest_technical() -> dict:
    """
    Load last row of clean spot data, return dict with current indicator values.
    Returns NaN-safe dict — caller must check for NaN before using in scoring.
    """
    path = get_path("clean_spot_1h")
    if not path.exists():
        logger.warning(f"clean_spot_1h not found at {path}")
        return {}

    df = pd.read_parquet(path)
    if df.empty:
        return {}

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp")
    latest = df.iloc[-1]

    result = {
        "timestamp": latest["timestamp"],
        "close": float(latest["close"]),
        "bb_pct": float(latest["bb_pct"]) if not pd.isna(latest.get("bb_pct")) else None,
        "rsi_14": float(latest["rsi_14"]) if not pd.isna(latest.get("rsi_14")) else None,
    }
    params = get_params()["technical"]
    for w in params["ma_windows"]:
        col = f"ma_{w}"
        result[col] = float(latest[col]) if col in latest.index and not pd.isna(latest[col]) else None

    logger.info(
        f"Latest technical: close={result['close']:.2f}, "
        f"bb_pct={result['bb_pct']:.3f}, rsi={result['rsi_14']:.1f}"
    )
    return result


# ---------------------------------------------------------------------------
# Standalone run: recompute indicators and save
# ---------------------------------------------------------------------------

def run() -> None:
    """Recompute all technical indicators on the clean spot parquet and overwrite."""
    from src.data.utils import save_with_window

    path = get_path("clean_spot_1h")
    if not path.exists():
        logger.warning(f"clean_spot_1h not found — run clean.py first")
        return

    df = pd.read_parquet(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"])
    df = add_all_indicators(df)
    save_with_window(df, path, freq="1h")
    logger.info(f"technical.run(): recomputed indicators, {len(df)} rows")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run()
