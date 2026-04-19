"""
src/models/gate_scoring.py — Gate Scoring v2 engine.

11 gates → 6 clusters → total score → threshold → decision.
ALL parameters read from parameters.yml via src/config.get_params().

Entry point: run_scoring_pipeline(...) → dict with signal, score, breakdown.
MA200 override: if close < MA200 and slope < 0 for N consecutive days → force Bear,
bypassing R5C latency in sustained downtrends.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

from src.config import get_params

logger = logging.getLogger("models.gate_scoring")


# ---------------------------------------------------------------------------
# Continuous tanh scoring (G3–G10)
# ---------------------------------------------------------------------------

def gate_score_continuous(z: float, corr: float, sensitivity: float, max_score: float) -> float:
    """Tanh scoring: corr * tanh(z * sensitivity) * max_score, clipped to ±max_score."""
    if z is None or (isinstance(z, float) and np.isnan(z)):
        return 0.0
    raw = corr * np.tanh(z * sensitivity) * max_score
    return float(np.clip(raw, -max_score, max_score))


# ---------------------------------------------------------------------------
# MA200 override — bypass slow R5C in sustained downtrends
# ---------------------------------------------------------------------------

def check_ma200_override(spot_df: pd.DataFrame) -> dict:
    """
    If close < MA200 AND slope_MA200 < 0 for N consecutive days → force Bear.
    N is read from parameters.yml ma200_override.consecutive_days (default 5).
    """
    cfg = get_params().get("ma200_override", {})
    if not cfg.get("enabled", True):
        return {"force_bear": False, "consecutive_days_below": 0,
                "close_vs_ma200_pct": None, "ma200_slope_5d": None}

    n = cfg.get("consecutive_days", 5)
    close = spot_df["close"].astype(float)
    ma200 = close.rolling(200, min_periods=100).mean()
    ma200_slope = ma200 - ma200.shift(5)

    below = close < ma200
    neg_slope = ma200_slope < 0
    combined = below & neg_slope

    # Count consecutive True at the end of the series
    consecutive = 0
    for val in reversed(combined.values):
        if val:
            consecutive += 1
        else:
            break

    last_close = close.iloc[-1]
    last_ma200 = ma200.iloc[-1]
    last_slope = ma200_slope.iloc[-1]

    close_vs_ma200_pct = (
        (last_close / last_ma200 - 1) * 100 if pd.notna(last_ma200) else None
    )

    return {
        "force_bear": consecutive >= n,
        "consecutive_days_below": consecutive,
        "close_vs_ma200_pct": round(close_vs_ma200_pct, 2) if close_vs_ma200_pct is not None else None,
        "ma200_slope_5d": round(float(last_slope), 1) if pd.notna(last_slope) else None,
    }


# ---------------------------------------------------------------------------
# G0 — Regime multiplier / BLOCK
# ---------------------------------------------------------------------------

def evaluate_g0(regime: str) -> dict:
    """Bear → BLOCK. Sideways → multiplier from params. Bull → 1.0."""
    regime = (regime or "Sideways").strip()
    if regime == "Bear":
        return {"block": True, "block_reason": "BLOCK_BEAR_REGIME", "multiplier": 0.0}
    elif regime == "Sideways":
        mult = float(get_params().get("sideways_multiplier", 0.5))
        return {"block": False, "multiplier": mult}
    else:  # Bull
        return {"block": False, "multiplier": 1.0}


# ---------------------------------------------------------------------------
# G1 — Technical (BB + RSI bucket scoring)
# ---------------------------------------------------------------------------

def _apply_buckets(value: float, buckets: list[dict]) -> float:
    """Apply ordered bucket rules to a value. Returns score of first matching bucket."""
    for bucket in buckets:
        cond = bucket["condition"]
        thr = bucket["threshold"]
        score = bucket["score"]
        if cond == "gte" and value >= thr:
            return float(score)
        elif cond == "gt" and value > thr:
            return float(score)
        elif cond == "lt" and value < thr:
            return float(score)
        elif cond == "lte" and value <= thr:
            return float(score)
        elif cond == "else":
            return float(score)
    return 0.0


def evaluate_g1(bb_pct: float, rsi: float) -> dict:
    """
    Bucket scoring — validated walk-forward (208 signals).
    DO NOT convert to tanh. Reads thresholds from parameters.yml.
    """
    params = get_params()
    bb_buckets = params["g1_bb_scores"]
    rsi_buckets = params["g1_rsi_scores"]

    bb_score = _apply_buckets(bb_pct, bb_buckets) if bb_pct is not None else 0.0
    rsi_score = _apply_buckets(rsi, rsi_buckets) if rsi is not None else 0.0

    # BB kill switch check (returned separately for paper_trader)
    ks_params = params["kill_switches"]
    bb_kill = bb_pct is not None and bb_pct >= ks_params["bb_top_threshold"]

    return {
        "g1": bb_score + rsi_score,
        "bb_score": bb_score,
        "rsi_score": rsi_score,
        "bb_kill": bb_kill,
    }


# ---------------------------------------------------------------------------
# G2 — News (crypto + fed split)
# ---------------------------------------------------------------------------

def evaluate_g2(news_crypto_score: float, fed_sentiment: dict) -> dict:
    """
    G2 = crypto (50%) + fed (50%).
    Kill switch if fed_score < threshold near FOMC — handled via fomc_kill_switch
    flag in fed_sentinel output (paper_trader checks it upstream).
    """
    params = get_params()
    w_crypto = params["news"]["crypto_weight"]   # 0.5
    w_fed = params["news"]["fed_weight"]         # 0.5

    fed_score = fed_sentiment.get("fed_score", 0.0) if fed_sentiment else 0.0
    crypto_score = news_crypto_score or 0.0

    g2 = w_crypto * crypto_score + w_fed * fed_score
    return {
        "g2": g2,
        "g2_crypto": crypto_score,
        "g2_fed": fed_score,
    }


# ---------------------------------------------------------------------------
# G3 — Macro rates (DGS10, DGS2, curve, RRP)
# ---------------------------------------------------------------------------

def evaluate_g3(zscores: dict, effective_weights: Optional[dict] = None) -> float:
    params = get_params()["gate_params"]
    eff = effective_weights or {}
    g3 = 0.0
    for key, z_col in [
        ("g3_dgs10", "dgs10_z"),
        ("g3_curve", "curve_z"),
        ("g3_rrp", "rrp_z"),
        ("g3_dgs2", "dgs2_z"),
    ]:
        if key in params:
            corr, sens, max_s = params[key]
            max_s = eff.get(key, max_s)
            g3 += gate_score_continuous(zscores.get(z_col, np.nan), corr, sens, max_s)
    return g3


# ---------------------------------------------------------------------------
# G4 — Open Interest
# ---------------------------------------------------------------------------

def evaluate_g4(zscores: dict, stale_days: dict, effective_weights: Optional[dict] = None) -> float:
    params = get_params()
    gp = params["gate_params"]
    stale_tol = params["stale_tolerance_days"]

    if stale_days.get("g4_oi", 0) > stale_tol["g4_oi"]:
        logger.warning("G4 OI stale — using 0.0")
        return 0.0

    corr, sens, max_s = gp["g4_oi"]
    max_s = (effective_weights or {}).get("g4_oi", max_s)
    return gate_score_continuous(zscores.get("oi_z", np.nan), corr, sens, max_s)


# ---------------------------------------------------------------------------
# G5–G10 — Remaining continuous gates
# ---------------------------------------------------------------------------

def _stale_gate(gate_key: str, stale_days: dict) -> bool:
    tol = get_params()["stale_tolerance_days"]
    days = stale_days.get(gate_key, 0)
    limit = tol.get(gate_key, 7)
    return days > limit


def evaluate_g5(zscores: dict, stale_days: dict, effective_weights: Optional[dict] = None) -> float:
    if _stale_gate("g5_stablecoin", stale_days):
        return 0.0
    corr, sens, max_s = get_params()["gate_params"]["g5_stable"]
    max_s = (effective_weights or {}).get("g5_stable", max_s)
    return gate_score_continuous(zscores.get("stablecoin_z", np.nan), corr, sens, max_s)


def evaluate_g6(zscores: dict, stale_days: dict, effective_weights: Optional[dict] = None) -> float:
    if _stale_gate("g6_bubble", stale_days):
        return 0.0
    corr, sens, max_s = get_params()["gate_params"]["g6_bubble"]
    max_s = (effective_weights or {}).get("g6_bubble", max_s)
    return gate_score_continuous(zscores.get("bubble_z", np.nan), corr, sens, max_s)


def evaluate_g7(zscores: dict, stale_days: dict, effective_weights: Optional[dict] = None) -> float:
    if _stale_gate("g7_etf", stale_days):
        return 0.0
    corr, sens, max_s = get_params()["gate_params"]["g7_etf"]
    max_s = (effective_weights or {}).get("g7_etf", max_s)
    return gate_score_continuous(zscores.get("etf_z", np.nan), corr, sens, max_s)


def evaluate_g8(zscores: dict, stale_days: dict, effective_weights: Optional[dict] = None) -> float:
    if _stale_gate("g8_fg", stale_days):
        return 0.0
    corr, sens, max_s = get_params()["gate_params"]["g8_fg"]
    max_s = (effective_weights or {}).get("g8_fg", max_s)
    return gate_score_continuous(zscores.get("fg_z", np.nan), corr, sens, max_s)


def evaluate_g9(zscores: dict, stale_days: dict, effective_weights: Optional[dict] = None) -> float:
    if _stale_gate("g9_taker", stale_days):
        return 0.0
    corr, sens, max_s = get_params()["gate_params"]["g9_taker"]
    max_s = (effective_weights or {}).get("g9_taker", max_s)
    return gate_score_continuous(zscores.get("taker_z", np.nan), corr, sens, max_s)


def evaluate_g10(zscores: dict, stale_days: dict, effective_weights: Optional[dict] = None) -> float:
    if _stale_gate("g10_funding", stale_days):
        return 0.0
    corr, sens, max_s = get_params()["gate_params"]["g10_funding"]
    max_s = (effective_weights or {}).get("g10_funding", max_s)
    return gate_score_continuous(zscores.get("funding_z", np.nan), corr, sens, max_s)


# ---------------------------------------------------------------------------
# Cluster aggregation with caps
# ---------------------------------------------------------------------------

def aggregate_clusters(gates: dict) -> dict:
    """Apply cluster caps from parameters.yml and return cluster scores + total."""
    caps = get_params()["cluster_caps"]

    clusters = {
        "technical":   float(np.clip(gates["g1"],                         *caps["technical"])),
        "news":        float(np.clip(gates["g2"],                         *caps["news"])),
        "macro":       float(np.clip(gates["g3"],                         *caps["macro"])),
        "positioning": float(np.clip(gates["g4"] + gates["g10"],          *caps["positioning"])),
        "liquidity":   float(np.clip(gates["g5"] + gates["g7"],           *caps["liquidity"])),
        "sentiment":   float(np.clip(gates["g6"] + gates["g8"] + gates["g9"], *caps["sentiment"])),
    }
    total = sum(clusters.values())
    return {"clusters": clusters, "total_score": round(total, 4)}


# ---------------------------------------------------------------------------
# Dynamic threshold
# ---------------------------------------------------------------------------

def compute_threshold(score_history: list[float], fed_proximity_adj: float = 0.0) -> float:
    """
    Quantile-based dynamic threshold with floor/ceiling + Fed Sentinel overlay.
    Uses warmup_value if fewer than min_history_days entries.
    """
    params = get_params()["threshold"]
    warmup = params["warmup_value"]
    floor = params["floor"]
    ceiling = params["ceiling"]
    quantile = params["quantile"]
    min_hist = params["min_history_days"]
    hist_window = params["history_window"]

    if len(score_history) < min_hist:
        base = warmup
    else:
        window = score_history[-hist_window:]
        base = float(np.quantile(window, quantile))
        base = float(np.clip(base, floor, ceiling))

    return round(base + fed_proximity_adj, 4)


# ---------------------------------------------------------------------------
# Kill switches
# ---------------------------------------------------------------------------

def check_kill_switches(
    bb_pct: float,
    oi_z: float,
    news_score: float,
    fed_context: dict,
    oi_stale: bool,
) -> dict:
    """Return dict: {blocked, reason}. Checks all kill conditions."""
    ks = get_params()["kill_switches"]

    if bb_pct is not None and bb_pct >= ks["bb_top_threshold"]:
        return {"blocked": True, "reason": "BLOCK_BB_TOP"}

    if not oi_stale and oi_z is not None and not np.isnan(oi_z):
        if oi_z > ks["oi_extreme_z"]:
            return {"blocked": True, "reason": "BLOCK_OI_EXTREME"}

    if news_score is not None and news_score < ks["news_bear_score"]:
        return {"blocked": True, "reason": "BLOCK_NEWS_BEAR"}

    if fed_context.get("fomc_kill_switch"):
        return {"blocked": True, "reason": "BLOCK_FED_HAWKISH"}

    return {"blocked": False, "reason": None}


# ---------------------------------------------------------------------------
# Master pipeline
# ---------------------------------------------------------------------------

def run_scoring_pipeline(
    regime: str,
    bb_pct: float,
    rsi: float,
    zscores: dict,
    stale_days: dict,
    news_crypto_score: float,
    fed_context: dict,
    score_history: list[float],
    spot_df: Optional[pd.DataFrame] = None,
    zs_daily: Optional[pd.DataFrame] = None,
    spot_daily: Optional[pd.Series] = None,
) -> dict:
    """
    Full pipeline: gates → clusters → threshold → kill switches → decision.

    Returns dict:
      signal:      ENTER | HOLD | BLOCK
      score:       float
      threshold:   float
      block_reason: str | None
      gate_scores: dict
      clusters:    dict
      proximity_adj: float
      ma200_override: dict
    """
    # Adaptive weights — confidence weighting + kill switch graduado (G3–G10)
    adaptive_details: dict = {}
    adaptive_summary: dict = {}
    effective_weights: Optional[dict] = None
    global_conf_mult: float = 1.0
    global_conf_label: str = "no_data"
    try:
        from src.models.adaptive_weights import compute_adaptive_weights, get_global_multiplier
        params = get_params()
        if zs_daily is not None and spot_daily is not None and not zs_daily.empty:
            adaptive = compute_adaptive_weights(zs_daily, spot_daily, params)
            effective_weights = adaptive["weights"]
            adaptive_details = adaptive["details"]
            adaptive_summary = adaptive["summary"]
            global_conf_mult, global_conf_label = get_global_multiplier(adaptive, params)
            logger.info(
                f"Adaptive weights: mean_conf={adaptive_summary['mean_confidence']:.3f} "
                f"wt_conf={adaptive_summary['weighted_mean_confidence']:.3f} "
                f"global_mult={global_conf_mult:.3f} "
                f"ok={adaptive_summary['n_ok']} reduced={adaptive_summary['n_reduced']} "
                f"severe={adaptive_summary['n_severe']} extreme={adaptive_summary['n_extreme']}"
            )
    except Exception as e:
        logger.warning(f"Adaptive weights failed (using base weights): {e}")

    # MA200 override: bypass slow R5C in sustained downtrends
    ma200 = {}
    if spot_df is not None and len(spot_df) >= 100:
        ma200 = check_ma200_override(spot_df)
        if ma200["force_bear"] and regime != "Bear":
            logger.info(
                f"MA200 override: forcing Bear "
                f"(close {ma200['close_vs_ma200_pct']:+.1f}% vs MA200, "
                f"{ma200['consecutive_days_below']}d below, "
                f"slope={ma200['ma200_slope_5d']})"
            )
            regime = "Bear"

    # G0 — regime check
    g0 = evaluate_g0(regime)
    if g0["block"]:
        return {
            "signal": "BLOCK",
            "block_reason": g0["block_reason"],
            "score": None,
            "score_raw": None,
            "regime_multiplier": 0.0,
            "threshold": None,
            "gate_scores": {},
            "clusters": {},
            "proximity_adj": fed_context.get("proximity_adjustment", 0.0),
            "ma200_override": ma200,
            "adaptive_weights": adaptive_details,
        }

    # G1
    g1_result = evaluate_g1(bb_pct, rsi)

    # G2
    g2_result = evaluate_g2(news_crypto_score, fed_context)

    # G3–G10 (with effective_weights from adaptive module when available)
    ew = effective_weights
    gates = {
        "g1":  g1_result["g1"],
        "g2":  g2_result["g2"],
        "g3":  evaluate_g3(zscores, ew),
        "g4":  evaluate_g4(zscores, stale_days, ew),
        "g5":  evaluate_g5(zscores, stale_days, ew),
        "g6":  evaluate_g6(zscores, stale_days, ew),
        "g7":  evaluate_g7(zscores, stale_days, ew),
        "g8":  evaluate_g8(zscores, stale_days, ew),
        "g9":  evaluate_g9(zscores, stale_days, ew),
        "g10": evaluate_g10(zscores, stale_days, ew),
    }

    # Stale data block: >50% of CoinGlass gates stale
    coinglass_gates = ["g5_stablecoin", "g6_bubble", "g7_etf", "g8_fg"]
    n_stale = sum(1 for k in coinglass_gates if _stale_gate(k, stale_days))
    if n_stale > len(coinglass_gates) // 2:
        return {
            "signal": "BLOCK",
            "block_reason": "BLOCK_STALE_DATA",
            "score": None,
            "score_raw": None,
            "regime_multiplier": g0["multiplier"],
            "threshold": None,
            "gate_scores": gates,
            "clusters": {},
            "proximity_adj": fed_context.get("proximity_adjustment", 0.0),
            "ma200_override": ma200,
            "adaptive_weights": adaptive_details,
        }

    # Cluster aggregation
    cluster_result = aggregate_clusters(gates)
    score_raw = cluster_result["total_score"]

    # Apply G0 sideways multiplier
    multiplier = g0["multiplier"]
    score_after_regime = round(score_raw * multiplier, 4) if multiplier != 1.0 else score_raw

    # Apply global confidence multiplier (pós-regime, pré-threshold)
    total_score = round(score_after_regime * global_conf_mult, 4)

    # Threshold
    prox_adj = fed_context.get("proximity_adjustment", 0.0)
    threshold = compute_threshold(score_history, prox_adj)

    # Kill switches
    oi_stale = stale_days.get("g4_oi", 0) > get_params()["stale_tolerance_days"]["g4_oi"]
    kill = check_kill_switches(
        bb_pct=bb_pct,
        oi_z=zscores.get("oi_z"),
        news_score=g2_result["g2"],
        fed_context=fed_context,
        oi_stale=oi_stale,
    )

    if kill["blocked"]:
        signal = "BLOCK"
        block_reason = kill["reason"]
    elif total_score >= threshold:
        signal = "ENTER"
        block_reason = None
    else:
        signal = "HOLD"
        block_reason = None

    gate_scores = {
        **gates,
        "g2_crypto": g2_result["g2_crypto"],
        "g2_fed": g2_result["g2_fed"],
        "bb_score": g1_result["bb_score"],
        "rsi_score": g1_result["rsi_score"],
        "regime_multiplier": multiplier,
    }

    ma200_tag = f" [MA200:{ma200.get('close_vs_ma200_pct'):+.1f}%]" if ma200.get("close_vs_ma200_pct") is not None else ""
    logger.info(
        f"Gate scoring: score={total_score:.3f} (raw={score_raw:.3f} ×regime={multiplier} "
        f"×gc={global_conf_mult:.3f}) vs thr={threshold:.3f} "
        f"→ {signal} (prox_adj={prox_adj:+.1f}, regime={regime}{ma200_tag})"
    )

    return {
        "signal": signal,
        "block_reason": block_reason,
        "score": total_score,                    # pós-regime × pós-global-conf (decision)
        "score_raw": score_raw,                  # Σ clusters, pré-multiplicadores
        "score_after_regime": score_after_regime, # pós-regime, pré-global-conf
        "regime_multiplier": multiplier,
        "global_confidence_multiplier": global_conf_mult,
        "global_confidence_source": global_conf_label,
        "threshold": threshold,
        "gate_scores": gate_scores,
        "clusters": cluster_result["clusters"],
        "proximity_adj": prox_adj,
        "ma200_override": ma200,
        "adaptive_weights": adaptive_details,
    }
