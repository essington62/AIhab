"""
Adaptive Weights — Confidence weighting + kill switch graduado.

Três níveis de adaptação:
1. Confidence weighting (delta < 0.5): redução suave
2. Kill switch severo (delta >= 0.5): reduz para 30%
3. Kill switch extremo (delta >= 0.6): zera completamente
"""
import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


GATE_MAP = {
    "oi_z":         ("g4_oi",       "G4 OI"),
    "taker_z":      ("g9_taker",    "G9 Taker"),
    "funding_z":    ("g10_funding", "G10 Funding"),
    "dgs10_z":      ("g3_dgs10",    "G3 DGS10"),
    "curve_z":      ("g3_curve",    "G3 Curve"),
    "stablecoin_z": ("g5_stable",   "G5 Stablecoin"),
    "bubble_z":     ("g6_bubble",   "G6 Bubble"),
    "etf_z":        ("g7_etf",      "G7 ETF"),
    "fg_z":         ("g8_fg",       "G8 F&G"),
}


def compute_rolling_correlations(
    zs_daily: pd.DataFrame,
    spot_daily: pd.Series,
    windows: list = [30, 60],
) -> dict:
    """Correlação rolling de cada gate com forward return 3d."""
    ret_3d = spot_daily.pct_change(3).shift(-3) * 100

    results = {}
    for zcol, (gkey, gname) in GATE_MAP.items():
        if zcol not in zs_daily.columns:
            continue

        df = pd.concat([zs_daily[zcol], ret_3d.rename("ret_3d")], axis=1).dropna()
        results[zcol] = {}

        for w in windows:
            if len(df) < w:
                results[zcol][w] = None
                continue

            rolling = df[zcol].rolling(w).corr(df["ret_3d"])
            latest = rolling.dropna().iloc[-1] if not rolling.dropna().empty else None
            results[zcol][w] = float(latest) if latest is not None else None

    return results


def compute_delta_smooth(
    corr_cfg: float,
    corr_short: Optional[float],
    corr_long: Optional[float],
    smooth_short: float = 0.3,
    smooth_long: float = 0.7,
) -> Optional[float]:
    """
    Delta suavizado (ponderado entre janelas short e long).

    Returns:
        delta smooth, ou None se sem dados
    """
    if corr_short is None and corr_long is None:
        return None

    if corr_long is None:
        return abs(corr_cfg - corr_short)
    if corr_short is None:
        return abs(corr_cfg - corr_long)

    delta_short = abs(corr_cfg - corr_short)
    delta_long = abs(corr_cfg - corr_long)
    return smooth_long * delta_long + smooth_short * delta_short


def compute_confidence(delta: Optional[float], min_confidence: float = 0.0) -> float:
    """Confidence = 1 - min(delta, 1.0)."""
    if delta is None:
        return 1.0
    confidence = 1.0 - min(delta, 1.0)
    return max(confidence, min_confidence)


def apply_kill_switch(
    effective_weight: float,
    delta: Optional[float],
    severe_threshold: float = 0.5,
    severe_multiplier: float = 0.3,
    extreme_threshold: float = 0.6,
) -> tuple[float, str]:
    """
    Aplica kill switch graduado sobre effective_weight.

    Returns:
        (new_weight, status_label)
        status_label: "ok" | "severe" | "extreme"
    """
    if delta is None:
        return effective_weight, "ok"

    if delta >= extreme_threshold:
        return 0.0, "extreme"

    if delta >= severe_threshold:
        return effective_weight * severe_multiplier, "severe"

    return effective_weight, "ok"


def compute_adaptive_weights(
    zs_daily: pd.DataFrame,
    spot_daily: pd.Series,
    params: dict,
) -> dict:
    """
    Computa pesos adaptativos para todos os gates.

    Returns:
        dict com keys: weights, details, summary, enabled
    """
    aw_cfg = params.get("adaptive_weights", {})

    if not aw_cfg.get("enabled", False):
        gp = params.get("gate_params", {})
        base_weights = {k: float(cfg[2]) if len(cfg) >= 3 else 1.0 for k, cfg in gp.items()}
        return {
            "weights": base_weights,
            "details": {k: {
                "gate": k, "base_weight": v, "effective_weight": v, "confidence": 1.0,
                "kill_status": "ok", "delta": None, "corr_cfg": None, "corr_long": None,
            } for k, v in base_weights.items()},
            "summary": {"n_ok": len(base_weights), "n_reduced": 0, "n_severe": 0,
                        "n_extreme": 0, "mean_confidence": 1.0},
            "enabled": False,
        }

    conf_cfg = aw_cfg.get("confidence", {})
    ks_cfg = aw_cfg.get("kill_switch", {})

    w_short = conf_cfg.get("window_short", 30)
    w_long = conf_cfg.get("window_long", 60)
    smooth_short = conf_cfg.get("smooth_short", 0.3)
    smooth_long = conf_cfg.get("smooth_long", 0.7)
    min_conf = conf_cfg.get("min_confidence", 0.0)

    severe_th = ks_cfg.get("severe_delta_threshold", 0.5)
    severe_mult = ks_cfg.get("severe_multiplier", 0.3)
    extreme_th = ks_cfg.get("extreme_delta_threshold", 0.6)

    rolling_corrs = compute_rolling_correlations(zs_daily, spot_daily, windows=[w_short, w_long])

    gp = params.get("gate_params", {})
    weights = {}
    details = {}

    for zcol, (gkey, gname) in GATE_MAP.items():
        if gkey not in gp:
            continue

        cfg = gp[gkey]
        corr_cfg = float(cfg[0])
        base_weight = float(cfg[2]) if len(cfg) >= 3 else 1.0

        corr_short = rolling_corrs.get(zcol, {}).get(w_short)
        corr_long = rolling_corrs.get(zcol, {}).get(w_long)

        if corr_short is None and corr_long is None:
            weights[gkey] = base_weight
            details[gkey] = {
                "gate": gname, "base_weight": base_weight, "effective_weight": base_weight,
                "confidence": 1.0, "kill_status": "ok", "delta": None,
                "corr_cfg": corr_cfg, "corr_short": None, "corr_long": None,
                "reason": "insufficient_data",
            }
            continue

        delta = compute_delta_smooth(corr_cfg, corr_short, corr_long, smooth_short, smooth_long)

        confidence = compute_confidence(delta, min_conf) if conf_cfg.get("enabled", True) else 1.0
        effective_weight = base_weight * confidence

        if ks_cfg.get("enabled", True):
            effective_weight, kill_status = apply_kill_switch(
                effective_weight, delta, severe_th, severe_mult, extreme_th
            )
        else:
            kill_status = "ok"

        if kill_status == "extreme":
            reason = f"EXTREME (delta={delta:.3f} >= {extreme_th})"
        elif kill_status == "severe":
            reason = f"SEVERE (delta={delta:.3f} >= {severe_th}, weight × {severe_mult})"
        elif confidence < 0.8:
            reason = f"REDUCED (delta={delta:.3f}, conf={confidence:.2f})"
        else:
            reason = f"OK (delta={delta:.3f}, conf={confidence:.2f})"

        weights[gkey] = effective_weight
        details[gkey] = {
            "gate": gname,
            "base_weight": base_weight,
            "effective_weight": round(effective_weight, 4),
            "confidence": round(confidence, 4),
            "kill_status": kill_status,
            "delta": round(delta, 4) if delta is not None else None,
            "corr_cfg": round(corr_cfg, 4),
            "corr_short": round(corr_short, 4) if corr_short is not None else None,
            "corr_long": round(corr_long, 4) if corr_long is not None else None,
            "reason": reason,
        }

    n_ok = sum(1 for d in details.values() if d["kill_status"] == "ok" and d.get("confidence", 0) > 0.8)
    n_reduced = sum(1 for d in details.values() if d["kill_status"] == "ok" and d.get("confidence", 0) <= 0.8)
    n_severe = sum(1 for d in details.values() if d["kill_status"] == "severe")
    n_extreme = sum(1 for d in details.values() if d["kill_status"] == "extreme")
    mean_conf = float(np.mean([d["confidence"] for d in details.values()])) if details else 1.0

    return {
        "weights": weights,
        "details": details,
        "summary": {
            "n_ok": n_ok,
            "n_reduced": n_reduced,
            "n_severe": n_severe,
            "n_extreme": n_extreme,
            "mean_confidence": round(mean_conf, 4),
        },
        "enabled": True,
    }
