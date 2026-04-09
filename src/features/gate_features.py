"""
src/features/gate_features.py — Z-scores for all gates (G3–G10).

Reads from 02_intermediate/ (1h-aligned, clean) and 01_raw/ (coinglass, sentiment).
Output: data/02_features/gate_zscores.parquet

Column map (from actual parquet column names produced by data layer):
  clean_futures_oi      → open_interest         (gate G4, CoinGlass aggregated)
  clean_futures_taker   → buy_sell_ratio        (gate G9, CoinGlass Binance)
  clean_futures_funding → funding_rate          (gate G10, CoinGlass OI-weighted)
  clean_macro           → dgs10, dgs2, rrp      (gate G3)
  coinglass_stablecoin  → stablecoin_mcap_usd   (gate G5)
  coinglass_bubble      → bubble_index          (gate G6)
  coinglass_etf         → etf_flow_usd          (gate G7)
  sentiment_fg          → fg_value              (gate G8)
"""

import logging

import numpy as np
import pandas as pd

from src.config import get_params, get_path

logger = logging.getLogger("features.gate_features")


# ---------------------------------------------------------------------------
# Core z-score
# ---------------------------------------------------------------------------

def compute_zscore(series: pd.Series, window: int) -> pd.Series:
    """Rolling z-score with min_periods = max(window//2, 5)."""
    min_p = max(window // 2, 5)
    mean = series.rolling(window, min_periods=min_p).mean()
    std = series.rolling(window, min_periods=min_p).std()
    return (series - mean) / std.replace(0, np.nan)


# ---------------------------------------------------------------------------
# Loader helpers
# ---------------------------------------------------------------------------

def _load_ts(name: str, ts_col: str = "timestamp") -> pd.DataFrame:
    path = get_path(name)
    if not path.exists():
        logger.warning(f"{name}: file not found at {path}")
        return pd.DataFrame()
    df = pd.read_parquet(path)
    df[ts_col] = pd.to_datetime(df[ts_col], utc=True)
    return df.sort_values(ts_col).set_index(ts_col)


def _series(name: str, col: str) -> pd.Series:
    df = _load_ts(name)
    if df.empty or col not in df.columns:
        logger.warning(f"{name}: column '{col}' not found. Cols: {list(df.columns)}")
        return pd.Series(dtype=float)
    return df[col].astype(float)


# ---------------------------------------------------------------------------
# Futures z-scores (1h)
# ---------------------------------------------------------------------------

def _futures_zscores(windows: dict) -> dict[str, pd.Series]:
    oi_z = compute_zscore(_series("clean_futures_oi", "open_interest"), windows["oi"])
    taker_z = compute_zscore(_series("clean_futures_taker", "buy_sell_ratio"), windows["taker"])
    funding_z = compute_zscore(_series("clean_futures_funding", "funding_rate"), windows["funding"])
    return {"oi_z": oi_z, "taker_z": taker_z, "funding_z": funding_z}


# ---------------------------------------------------------------------------
# Macro z-scores (daily → forward-fill to 1h base index)
# ---------------------------------------------------------------------------

def _macro_zscores(windows: dict) -> dict[str, pd.Series]:
    macro = _load_ts("clean_macro")
    if macro.empty:
        return {}

    dgs10 = macro["dgs10"].astype(float) if "dgs10" in macro.columns else pd.Series(dtype=float)
    dgs2  = macro["dgs2"].astype(float)  if "dgs2"  in macro.columns else pd.Series(dtype=float)
    rrp   = macro["rrp"].astype(float)   if "rrp"   in macro.columns else pd.Series(dtype=float)
    curve = dgs10 - dgs2

    return {
        "dgs10_z":  compute_zscore(dgs10, windows["dgs10"]),
        "dgs2_z":   compute_zscore(dgs2,  windows["dgs2"]),
        "rrp_z":    compute_zscore(rrp,   windows["rrp"]),
        "curve_z":  compute_zscore(curve, windows["yield_curve"]),
    }


# ---------------------------------------------------------------------------
# CoinGlass + Sentiment z-scores (daily)
# ---------------------------------------------------------------------------

def _daily_zscores(windows: dict, etf_roll: int) -> dict[str, pd.Series]:
    stable_z = compute_zscore(
        _series("coinglass_stablecoin", "stablecoin_mcap_usd"), windows["stablecoin_mcap"]
    )

    bubble_z = compute_zscore(
        _series("coinglass_bubble", "bubble_index"), windows["bubble_index"]
    )

    etf_raw = _series("coinglass_etf", "etf_flow_usd")
    etf_cum = etf_raw.rolling(etf_roll, min_periods=3).sum()
    etf_z = compute_zscore(etf_cum, windows["etf_flows"])

    fg_z = compute_zscore(
        _series("sentiment_fg", "fg_value"), windows["fear_greed"]
    )

    return {
        "stablecoin_z": stable_z,
        "bubble_z": bubble_z,
        "etf_z": etf_z,
        "fg_z": fg_z,
    }


# ---------------------------------------------------------------------------
# Merge: 1h futures base + daily ffill
# ---------------------------------------------------------------------------

def _ffill_daily_to_1h(base_idx: pd.DatetimeIndex, series: pd.Series) -> pd.Series:
    """Reindex daily series to 1h index via forward-fill."""
    if series.empty:
        return pd.Series(np.nan, index=base_idx)
    combined = series.reindex(base_idx.union(series.index)).sort_index().ffill()
    return combined.reindex(base_idx)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def compute_all_zscores() -> pd.DataFrame:
    params = get_params()
    windows = params["zscore_windows"]
    etf_roll = params["etf_flow_rolling"]

    futures = _futures_zscores(windows)
    macro = _macro_zscores(windows)
    daily = _daily_zscores(windows, etf_roll)

    # Use OI index as 1h base
    oi_series = _series("clean_futures_oi", "open_interest")
    if oi_series.empty:
        logger.error("OI data missing — cannot compute gate z-scores")
        return pd.DataFrame()

    base_idx = oi_series.index

    result = pd.DataFrame(index=base_idx)

    # 1h futures series: reindex + ffill (taker/funding may lag OI by 1-4h)
    for name, series in futures.items():
        result[name] = series.reindex(base_idx).ffill()

    # Daily series: ffill to 1h
    for name, series in {**macro, **daily}.items():
        result[name] = _ffill_daily_to_1h(base_idx, series)

    result = result.reset_index().rename(columns={"timestamp": "timestamp"})
    result["timestamp"] = pd.to_datetime(result["timestamp"], utc=True)

    return result


def run() -> None:
    result = compute_all_zscores()
    if result.empty:
        logger.error("gate_features: empty result — check data sources")
        return

    out_path = get_path("gate_zscores")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Window: keep last 8760 rows (~1 year 1h)
    result = result.sort_values("timestamp").tail(8760)
    result.to_parquet(out_path, index=False)

    logger.info(
        f"gate_zscores: {len(result)} rows saved, "
        f"last={result['timestamp'].max()}, "
        f"cols={[c for c in result.columns if c != 'timestamp']}"
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run()
