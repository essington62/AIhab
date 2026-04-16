#!/usr/bin/env python3
"""
scripts/daily_observation.py — Daily state snapshot for paper-trading observation period.

Read-only: reads parquets, appends one row to data/06_observation/daily_observation.csv.
Designed to run at 01:00 UTC via supercronic.

Run manually:
    conda run -n btc_trading_v1 python scripts/daily_observation.py
    # or inside container:
    python scripts/daily_observation.py
"""

import csv
import json
import logging
import math
import pathlib
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

# ── Path setup ────────────────────────────────────────────────────────────────
_REPO = pathlib.Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.config import get_params, get_path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("daily_observation")

OBS_DIR = _REPO / "data" / "06_observation"
OBS_CSV = OBS_DIR / "daily_observation.csv"

CSV_FIELDS = [
    "timestamp", "btc_price", "btc_change_24h",
    "score_adjusted", "score_raw", "regime_multiplier",
    "regime", "signal", "threshold", "block_reason",
    "fg_raw", "fg_z", "bb_pct", "rsi_14",
    "oi_z", "funding_z", "taker_z",
    "cluster_technical", "cluster_positioning", "cluster_macro",
    "cluster_liquidity", "cluster_sentiment", "cluster_news",
    "ma200", "ma200_distance_pct",
    "kill_switches",
    "flag_near_enter", "flag_extreme_negative", "flag_signal_changed",
    "flag_kill_switch_active", "flag_bb_extreme_high", "flag_bb_extreme_low",
    "delta_score", "delta_price_pct", "regime_changed", "signal_changed",
    "summary",
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _safe_read_parquet(key: str) -> pd.DataFrame:
    try:
        p = get_path(key)
        if not p.exists():
            logger.warning(f"{key}: parquet not found at {p}")
            return pd.DataFrame()
        df = pd.read_parquet(p)
        return df
    except Exception as e:
        logger.warning(f"{key}: read error — {e}")
        return pd.DataFrame()


def _latest(df: pd.DataFrame, col: str, default=None):
    if df.empty or col not in df.columns:
        return default
    val = df[col].dropna()
    return float(val.iloc[-1]) if not val.empty else default


def _nan(v):
    return v is None or (isinstance(v, float) and math.isnan(v))


# ── Cluster recompute (read-only, from gate_zscores + params) ─────────────────

def _compute_clusters_from_zscores(zs: dict) -> dict:
    """Lightweight cluster recompute for observation log. No I/O side effects."""
    from src.models.gate_scoring import (
        evaluate_g1, evaluate_g3, evaluate_g4, evaluate_g5,
        evaluate_g6, evaluate_g7, evaluate_g8, evaluate_g9, evaluate_g10,
        aggregate_clusters,
    )
    # stale_days=0 for all gates → assume fresh (conservative: use all available data)
    stale = {k: 0 for k in get_params().get("stale_tolerance_days", {})}

    g1  = evaluate_g1(zs.get("bb_pct", 0.5), zs.get("rsi_14", 50.0))
    g3  = evaluate_g3(zs)
    g4  = evaluate_g4(zs, stale)
    g5  = evaluate_g5(zs, stale)
    g6  = evaluate_g6(zs, stale)
    g7  = evaluate_g7(zs, stale)
    g8  = evaluate_g8(zs, stale)
    g9  = evaluate_g9(zs, stale)
    g10 = evaluate_g10(zs, stale)

    gates = {
        "g1": g1["g1"],   # evaluate_g1 returns {"g1": score, ...}
        "g3": g3, "g4": g4, "g5": g5,
        "g6": g6, "g7": g7, "g8": g8, "g9": g9, "g10": g10,
        "g2": 0.0,        # news not recomputed in observation (async DeepSeek)
    }
    result = aggregate_clusters(gates)   # returns {"clusters": {...}, "total_score": ...}
    return result["clusters"]


# ── Main ──────────────────────────────────────────────────────────────────────

def collect_snapshot() -> dict:
    ts = datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    # Portfolio state
    portfolio = {}
    try:
        p = get_path("portfolio_state")
        if p.exists():
            portfolio = json.loads(p.read_text())
    except Exception as e:
        logger.warning(f"portfolio_state: {e}")

    # Spot data
    spot_df = _safe_read_parquet("clean_spot_1h")
    if not spot_df.empty and "timestamp" in spot_df.columns:
        spot_df["timestamp"] = pd.to_datetime(spot_df["timestamp"], utc=True)
        spot_df = spot_df.sort_values("timestamp")

    btc_price    = _latest(spot_df, "close",  None)
    bb_pct       = _latest(spot_df, "bb_pct", None)
    rsi_14       = _latest(spot_df, "rsi_14", None)
    ma200        = _latest(spot_df, "ma_200", None)
    btc_change_24h = None
    if not spot_df.empty and len(spot_df) > 24 and btc_price:
        prev = float(spot_df["close"].iloc[-25])
        btc_change_24h = (btc_price - prev) / prev * 100

    ma200_distance_pct = None
    if btc_price and ma200:
        ma200_distance_pct = (btc_price - ma200) / ma200 * 100

    # Gate z-scores
    zs_df = _safe_read_parquet("gate_zscores")
    zs = {}
    if not zs_df.empty:
        last_z = zs_df.iloc[-1]
        zs = {c: float(last_z[c]) for c in zs_df.columns if c != "timestamp" and not _nan(last_z[c])}

    fg_z = zs.get("fg_z")
    oi_z = zs.get("oi_z")
    funding_z = zs.get("funding_z")
    taker_z = zs.get("taker_z")

    # F&G raw
    fg_df = _safe_read_parquet("sentiment_fg")
    fg_raw = _latest(fg_df, "fg_value", None)

    # Score history
    sh_df = _safe_read_parquet("score_history")
    score_adjusted    = portfolio.get("last_score")
    score_raw_val     = portfolio.get("last_score_raw")
    regime_multiplier = portfolio.get("last_regime_multiplier")
    regime            = portfolio.get("last_regime", "Unknown")
    signal            = portfolio.get("last_signal", "—")
    threshold         = portfolio.get("last_threshold")
    block_reason      = portfolio.get("last_block_reason", "")

    # Clusters (recomputed from z-scores)
    zs_for_clusters = {**zs, "bb_pct": bb_pct or 0.5, "rsi_14": rsi_14 or 50.0}
    clusters = {}
    try:
        clusters = _compute_clusters_from_zscores(zs_for_clusters)
    except Exception as e:
        logger.warning(f"cluster recompute failed: {e}")

    # Kill switches active
    kill_switches = []
    if block_reason:
        kill_switches = [block_reason]
    elif not sh_df.empty and "block_reason" in sh_df.columns:
        br = sh_df["block_reason"].iloc[-1]
        if br and not (isinstance(br, float) and math.isnan(br)):
            kill_switches = [str(br)]

    # Flags
    sa = score_adjusted or 0.0
    flag_near_enter        = sa > 1.5 and signal == "HOLD"
    flag_extreme_negative  = sa < -2.5
    flag_kill_switch_active = len(kill_switches) > 0
    flag_bb_extreme_high   = (bb_pct or 0) > 0.90
    flag_bb_extreme_low    = (bb_pct or 1) < 0.10

    # Previous row for delta
    delta_score = delta_price_pct = None
    regime_changed = signal_changed = False
    flag_signal_changed = False
    if OBS_CSV.exists():
        try:
            prev_df = pd.read_csv(OBS_CSV)
            if not prev_df.empty:
                prev = prev_df.iloc[-1]
                prev_score = prev.get("score_adjusted")
                prev_price = prev.get("btc_price")
                prev_regime = prev.get("regime", "")
                prev_signal = prev.get("signal", "")
                if prev_score is not None and not _nan(float(prev_score)) and not _nan(sa):
                    delta_score = sa - float(prev_score)
                if prev_price and btc_price:
                    delta_price_pct = (btc_price - float(prev_price)) / float(prev_price) * 100
                regime_changed = str(prev_regime) != str(regime)
                signal_changed = str(prev_signal) != str(signal)
                flag_signal_changed = signal_changed
        except Exception as e:
            logger.warning(f"delta calc: {e}")

    # Summary line
    flags = []
    if flag_near_enter:        flags.append("⚠️ near-enter")
    if flag_signal_changed:    flags.append(f"🔴 signal→{signal}")
    if flag_kill_switch_active: flags.append(f"🛑 {kill_switches[0]}")
    if flag_bb_extreme_high:   flags.append("📈 BB_TOP")
    if flag_bb_extreme_low:    flags.append("📉 BB_BOTTOM")
    if flag_extreme_negative:  flags.append("🔴 score_extremo")

    price_str = f"${btc_price:,.0f}" if btc_price else "N/A"
    chg_str   = f"({btc_change_24h:+.1f}% 24h)" if btc_change_24h is not None else ""
    raw_str   = f"{score_raw_val:.2f}" if score_raw_val is not None else "?"
    mult_str  = f"{regime_multiplier}" if regime_multiplier is not None else "?"
    summary = (
        f"BTC {price_str} {chg_str} | {signal} score {sa:.2f} "
        f"(raw {raw_str} × {mult_str}) | {regime} | "
        f"F&G {fg_raw:.0f}" if fg_raw else f"F&G ?"
    )
    if flags:
        summary += " | " + " ".join(flags)

    return {
        "timestamp":            ts,
        "btc_price":            round(btc_price, 2) if btc_price else None,
        "btc_change_24h":       round(btc_change_24h, 3) if btc_change_24h is not None else None,
        "score_adjusted":       round(sa, 4),
        "score_raw":            round(score_raw_val, 4) if score_raw_val is not None else None,
        "regime_multiplier":    regime_multiplier,
        "regime":               regime,
        "signal":               signal,
        "threshold":            round(threshold, 4) if threshold else None,
        "block_reason":         block_reason or "",
        "fg_raw":               int(fg_raw) if fg_raw is not None else None,
        "fg_z":                 round(fg_z, 4) if fg_z is not None else None,
        "bb_pct":               round(bb_pct, 4) if bb_pct is not None else None,
        "rsi_14":               round(rsi_14, 2) if rsi_14 is not None else None,
        "oi_z":                 round(oi_z, 4) if oi_z is not None else None,
        "funding_z":            round(funding_z, 4) if funding_z is not None else None,
        "taker_z":              round(taker_z, 4) if taker_z is not None else None,
        "cluster_technical":    round(clusters.get("technical", 0), 4),
        "cluster_positioning":  round(clusters.get("positioning", 0), 4),
        "cluster_macro":        round(clusters.get("macro", 0), 4),
        "cluster_liquidity":    round(clusters.get("liquidity", 0), 4),
        "cluster_sentiment":    round(clusters.get("sentiment", 0), 4),
        "cluster_news":         round(clusters.get("news", 0), 4),
        "ma200":                round(ma200, 2) if ma200 else None,
        "ma200_distance_pct":   round(ma200_distance_pct, 3) if ma200_distance_pct is not None else None,
        "kill_switches":        ";".join(kill_switches) if kill_switches else "",
        "flag_near_enter":      flag_near_enter,
        "flag_extreme_negative": flag_extreme_negative,
        "flag_signal_changed":  flag_signal_changed,
        "flag_kill_switch_active": flag_kill_switch_active,
        "flag_bb_extreme_high": flag_bb_extreme_high,
        "flag_bb_extreme_low":  flag_bb_extreme_low,
        "delta_score":          round(delta_score, 4) if delta_score is not None else None,
        "delta_price_pct":      round(delta_price_pct, 3) if delta_price_pct is not None else None,
        "regime_changed":       regime_changed,
        "signal_changed":       signal_changed,
        "summary":              summary,
    }


def append_to_csv(row: dict) -> None:
    OBS_DIR.mkdir(parents=True, exist_ok=True)
    write_header = not OBS_CSV.exists()
    with open(OBS_CSV, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in CSV_FIELDS})


if __name__ == "__main__":
    logger.info("daily_observation: collecting snapshot...")
    row = collect_snapshot()
    append_to_csv(row)
    logger.info(f"daily_observation: appended row to {OBS_CSV}")
    print()
    print("=== SNAPSHOT ===")
    print(row["summary"])
    print()
    for k in CSV_FIELDS:
        v = row.get(k)
        if v not in (None, "", False):
            print(f"  {k:28s}: {v}")
