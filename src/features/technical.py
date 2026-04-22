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

def compute_bollinger(close: pd.Series, window: int = 20, std: float = 2.0) -> pd.DataFrame:
    """Return DataFrame with bb_upper, bb_middle, bb_lower, bb_pct."""
    ma = close.rolling(window).mean()
    std_dev = close.rolling(window).std()
    upper = ma + std * std_dev
    lower = ma - std * std_dev
    pct = (close - lower) / (upper - lower).replace(0, np.nan)
    return pd.DataFrame({
        "bb_upper":  upper,
        "bb_middle": ma,
        "bb_lower":  lower,
        "bb_pct":    pct.clip(0, 1),
    })


def compute_bollinger_pct(close: pd.Series, window: int = 20, std: float = 2.0) -> pd.Series:
    """BB %B: 0 = lower band, 1 = upper band. Clipped [0, 1]. (Legacy — kept for compat.)"""
    return compute_bollinger(close, window, std)["bb_pct"]


def compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    """Average True Range (Wilder's smoothing via EWM span=window)."""
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(span=window, min_periods=window, adjust=False).mean()


def compute_rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def compute_ma(close: pd.Series, window: int) -> pd.Series:
    return close.rolling(window).mean()


def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add BB bands, RSI, MAs, ATR, and rolling highs/lows to a OHLCV DataFrame."""
    params = get_params()["technical"]
    close = df["close"].astype(float)
    high  = df["high"].astype(float)  if "high"  in df.columns else close
    low   = df["low"].astype(float)   if "low"   in df.columns else close

    df = df.copy()

    # Bollinger Bands (all four columns)
    bb = compute_bollinger(close, params["bb_window"], params["bb_std"])
    df["bb_upper"]  = bb["bb_upper"]
    df["bb_middle"] = bb["bb_middle"]
    df["bb_lower"]  = bb["bb_lower"]
    df["bb_pct"]    = bb["bb_pct"]

    # RSI
    df["rsi_14"] = compute_rsi(close, params["rsi_window"])

    # Moving averages
    for w in params["ma_windows"]:
        df[f"ma_{w}"] = compute_ma(close, w)

    # ATR-14
    df["atr_14"] = compute_atr(high, low, close, 14)

    # Rolling high/low lookback windows (in hourly candles: 7d=168h, 30d=720h)
    df["high_7d"]  = high.rolling(168).max()
    df["low_7d"]   = low.rolling(168).min()
    df["high_30d"] = high.rolling(720).max()
    df["low_30d"]  = low.rolling(720).min()

    return df


# ---------------------------------------------------------------------------
# Live price (sub-candle resolution for stop checks)
# ---------------------------------------------------------------------------

def get_live_price(symbol: str = "BTCUSDT", timeout: float = 5.0):
    """
    Fetch current price from Binance public REST API (no auth required).
    Used by check_stops_only() to get sub-candle resolution (15min checks).

    Returns float or None on failure.
    """
    import requests
    try:
        r = requests.get(
            "https://api.binance.com/api/v3/ticker/price",
            params={"symbol": symbol},
            timeout=timeout,
        )
        r.raise_for_status()
        return float(r.json()["price"])
    except Exception as e:
        logger.warning(f"get_live_price({symbol}) failed: {e}")
        return None


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

    # ret_1d — 24 hourly candles = 1 day
    if len(df) >= 25:
        close_now = float(df.iloc[-1]["close"])
        close_24h_ago = float(df.iloc[-25]["close"])
        ret_1d = (close_now - close_24h_ago) / close_24h_ago
    else:
        ret_1d = None

    latest = df.iloc[-1]

    def _f(col):
        return float(latest[col]) if col in latest.index and not pd.isna(latest[col]) else None

    # volume_z: rolling 7d (168h) z-score of current candle volume
    volume_z = None
    if "volume" in df.columns and len(df) >= 14:
        vol_s   = df["volume"].astype(float)
        vol_win = min(168, len(vol_s))
        vol_mean = vol_s.rolling(vol_win, min_periods=14).mean()
        vol_std  = vol_s.rolling(vol_win, min_periods=14).std()
        vol_z_s  = (vol_s - vol_mean) / vol_std.replace(0, np.nan)
        _vz = vol_z_s.iloc[-1]
        volume_z = float(_vz) if not pd.isna(_vz) else None

    result = {
        "timestamp":  latest["timestamp"],
        "close":      float(latest["close"]),
        "bb_pct":     _f("bb_pct"),
        "bb_upper":   _f("bb_upper"),
        "bb_middle":  _f("bb_middle"),
        "bb_lower":   _f("bb_lower"),
        "rsi_14":     _f("rsi_14"),
        "atr_14":     _f("atr_14"),
        "high_7d":    _f("high_7d"),
        "low_7d":     _f("low_7d"),
        "high_30d":   _f("high_30d"),
        "low_30d":    _f("low_30d"),
        "ret_1d":     round(ret_1d, 6) if ret_1d is not None else None,
        "volume_z":   volume_z,
    }
    params = get_params()["technical"]
    for w in params["ma_windows"]:
        col = f"ma_{w}"
        result[col] = _f(col)

    _ret1d_str = f"{result['ret_1d']:.4f}" if result.get("ret_1d") is not None else "N/A"
    logger.info(
        f"Latest technical: close={result['close']:.2f}, "
        f"bb_pct={result['bb_pct']:.3f}, rsi={result['rsi_14']:.1f}, "
        f"ret_1d={_ret1d_str}"
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
