"""
src/models/r5c_hmm.py — R5C HMM regime classification.

3 states: Bull / Sideways / Bear
Features: log_return, vol_short, vol_ratio, drawdown, volume_z, slope_21d

CRITICAL — Day-Shift Contract:
  Regime computed on day D (using D's closing candle) is applied to
  candles from D+1 00:00 UTC onwards. NEVER apply regime of day D
  to candles of day D itself.

Daily run: python -m src.models.r5c_hmm
Output: data/03_models/r5c_regime_history.parquet
        data/05_output/portfolio_state.json (regime field updated)
"""

import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

from src.config import get_params, get_path

logger = logging.getLogger("models.r5c_hmm")


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def compute_r5c_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute R5C HMM features from daily OHLCV DataFrame.
    Requires columns: timestamp, open, high, low, close, volume.
    Returns DataFrame with feature columns (NaN rows at edges, handled by caller).
    """
    params = get_params()["r5c"]
    vol_short = params["vol_short_window"]
    vol_long = params["vol_long_window"]
    slope_w = params["slope_window"]

    df = df.copy().sort_values("timestamp")
    close = df["close"].astype(float)
    volume = df["volume"].astype(float)

    # log_return
    df["log_return"] = np.log(close / close.shift(1))

    # vol_short: rolling std of log_return
    df["vol_short"] = df["log_return"].rolling(vol_short).std()

    # vol_ratio: vol_short / vol_long
    vol_long_s = df["log_return"].rolling(vol_long).std()
    df["vol_ratio"] = df["vol_short"] / vol_long_s.replace(0, np.nan)

    # drawdown: fraction from rolling max
    roll_max = close.cummax()
    df["drawdown"] = (close - roll_max) / roll_max

    # volume_z: rolling z-score
    vol_mean = volume.rolling(vol_long).mean()
    vol_std = volume.rolling(vol_long).std()
    df["volume_z"] = (volume - vol_mean) / vol_std.replace(0, np.nan)

    # slope_21d: linear regression slope of log(close) over last N days
    log_close = np.log(close)

    def linreg_slope(arr: np.ndarray) -> float:
        if np.isnan(arr).any():
            return np.nan
        x = np.arange(len(arr))
        slope, _, _, _, _ = stats.linregress(x, arr)
        return float(slope)

    df["slope_21d"] = log_close.rolling(slope_w).apply(linreg_slope, raw=True)

    feature_cols = ["log_return", "vol_short", "vol_ratio", "drawdown", "volume_z", "slope_21d"]
    return df[["timestamp"] + feature_cols]


# ---------------------------------------------------------------------------
# Model load
# ---------------------------------------------------------------------------

def load_model():
    """Load R5C HMM pickle. Raises FileNotFoundError if not found."""
    path = get_path("r5c_model")
    if not path.exists():
        raise FileNotFoundError(f"R5C model not found at {path}. Run migration script first.")
    with open(path, "rb") as f:
        model = pickle.load(f)
    logger.info(f"R5C model loaded from {path}")
    return model


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def predict_regime(model, features_df: pd.DataFrame) -> pd.DataFrame:
    """
    Run HMM inference on feature DataFrame.
    Returns DataFrame with: timestamp, state_idx, regime, prob_bull, prob_sideways, prob_bear.
    State label mapping from parameters.yml["r5c"]["state_labels"].
    """
    params = get_params()["r5c"]
    label_map: dict = {int(k): v for k, v in params["state_labels"].items()}
    feature_cols = params["features"]

    df = features_df.dropna(subset=feature_cols).copy()
    if df.empty:
        logger.warning("predict_regime: all rows have NaN features")
        return pd.DataFrame()

    X = df[feature_cols].values
    state_seq = model.predict(X)
    probs = model.predict_proba(X)

    result = pd.DataFrame({
        "timestamp": df["timestamp"].values,
        "state_idx": state_seq,
    })
    result["regime"] = result["state_idx"].map(label_map).fillna("Unknown")

    # Assign prob columns dynamically based on label_map
    for idx, label in label_map.items():
        col = f"prob_{label.lower()}"
        result[col] = probs[:, idx] if idx < probs.shape[1] else np.nan

    return result


# ---------------------------------------------------------------------------
# Get current regime (Day-Shift: regime of D used from D+1)
# ---------------------------------------------------------------------------

def get_current_regime(today: Optional[pd.Timestamp] = None) -> dict:
    """
    Return regime applicable RIGHT NOW.
    Day-shift: use regime from yesterday's candle (D-1), applied today (D).
    Falls back to 'Sideways' if data unavailable.
    """
    today = today or pd.Timestamp.utcnow().normalize()

    path = get_path("r5c_regime_history")
    if not path.exists():
        logger.warning("r5c_regime_history not found — defaulting to Sideways")
        return {"regime": "Sideways", "prob_sideways": 1.0, "timestamp": str(today)}

    df = pd.read_parquet(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp")

    # Day-shift: use regime where timestamp < today
    applicable = df[df["timestamp"] < today]
    if applicable.empty:
        logger.warning("No applicable regime found — defaulting to Sideways")
        return {"regime": "Sideways", "prob_sideways": 1.0, "timestamp": str(today)}

    latest = applicable.iloc[-1].to_dict()
    logger.info(
        f"Current regime: {latest.get('regime')} "
        f"(from {latest.get('timestamp')}, "
        f"applied from {today})"
    )
    return latest


# ---------------------------------------------------------------------------
# Daily run: compute features + predict + append to history
# ---------------------------------------------------------------------------

def run() -> None:
    """Load daily candles, compute features, predict regime, append to history."""
    clean_1d = get_path("clean_spot_1d")
    if not clean_1d.exists():
        logger.error(f"clean_spot_1d not found at {clean_1d} — run daily ingest first")
        return

    df = pd.read_parquet(clean_1d)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp")

    features = compute_r5c_features(df)

    try:
        model = load_model()
    except FileNotFoundError as e:
        logger.error(str(e))
        return

    regimes = predict_regime(model, features)
    if regimes.empty:
        logger.error("R5C inference returned empty result")
        return

    # Append to history (dedup on timestamp)
    hist_path = get_path("r5c_regime_history")
    hist_path.parent.mkdir(parents=True, exist_ok=True)

    if hist_path.exists():
        existing = pd.read_parquet(hist_path)
        existing["timestamp"] = pd.to_datetime(existing["timestamp"], utc=True)
        regimes["timestamp"] = pd.to_datetime(regimes["timestamp"], utc=True)
        combined = pd.concat([existing, regimes], ignore_index=True)
        combined["timestamp"] = pd.to_datetime(combined["timestamp"], utc=True)
        combined = combined.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
        combined = combined.tail(1095)  # ~3 years daily
    else:
        combined = regimes.sort_values("timestamp")

    combined.to_parquet(hist_path, index=False)
    latest = combined.iloc[-1]
    logger.info(
        f"R5C run complete: {len(combined)} rows, "
        f"latest regime={latest['regime']} on {latest['timestamp']}"
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run()
