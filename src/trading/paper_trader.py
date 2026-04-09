"""
src/trading/paper_trader.py — Hourly trading cycle.

Pipeline:
  load data → R5C regime → technical → z-scores → stale check →
  news sentiment → Fed Sentinel → Gate Scoring → decision → execution → log

Hardened: gate scoring wrapped in try/except — cycle never breaks the loop.
Each step is independently logged.
"""

import json
import logging
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.config import get_params, get_path
from src.features.fed_sentinel import get_fed_context
from src.features.gate_features import compute_all_zscores
from src.features.technical import get_latest_technical
from src.models.gate_scoring import run_scoring_pipeline
from src.models.r5c_hmm import get_current_regime
from src.trading.execution import (
    atomic_write_json,
    check_stops,
    execute_entry,
    execute_exit,
    load_portfolio,
)

logger = logging.getLogger("trading.paper_trader")


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def load_latest_zscores() -> dict:
    """Return latest row of gate_zscores as dict. Empty dict if unavailable."""
    path = get_path("gate_zscores")
    if not path.exists():
        logger.warning("gate_zscores.parquet not found")
        return {}
    df = pd.read_parquet(path)
    if df.empty:
        return {}
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    latest = df.sort_values("timestamp").iloc[-1]
    return latest.to_dict()


def compute_stale_days(zscores: dict) -> dict:
    """
    Estimate staleness for each gate based on last update timestamp in parquets.
    Returns {gate_key: days_since_last_update}.
    """
    now = pd.Timestamp.utcnow()
    stale = {}

    gate_paths = {
        "g4_oi":       "clean_futures_oi",
        "g9_taker":    "clean_futures_taker",
        "g10_funding": "clean_futures_funding",
        "g5_stablecoin": "coinglass_stablecoin",
        "g6_bubble":   "coinglass_bubble",
        "g7_etf":      "coinglass_etf",
        "g8_fg":       "sentiment_fg",
        "g3_macro":    "clean_macro",
    }
    for gate_key, catalog_name in gate_paths.items():
        try:
            path = get_path(catalog_name)
            if not path.exists():
                stale[gate_key] = 999
                continue
            df = pd.read_parquet(path)
            if df.empty:
                stale[gate_key] = 999
                continue
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            last_ts = df["timestamp"].max()
            stale[gate_key] = (now - last_ts).total_seconds() / 86400
        except Exception as e:
            logger.warning(f"stale check {gate_key}: {e}")
            stale[gate_key] = 999

    return stale


def load_news_crypto_score(lookback_hours: int = 4) -> float:
    """Load latest crypto news score from news_scores parquet, or 0.0."""
    path = get_path("news_scores")
    if not path.exists():
        return 0.0
    df = pd.read_parquet(path)
    if df.empty:
        return 0.0
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    cutoff = pd.Timestamp.utcnow() - pd.Timedelta(hours=lookback_hours)
    recent = df[df["timestamp"] >= cutoff]
    if recent.empty:
        return 0.0
    # Take mean of crypto scores in window
    if "crypto_score" in recent.columns:
        return float(recent["crypto_score"].mean())
    return 0.0


def load_score_history() -> list[float]:
    """Load historical total scores from score_history parquet."""
    path = get_path("score_history")
    if not path.exists():
        return []
    df = pd.read_parquet(path)
    if "total_score" not in df.columns:
        return []
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp")
    # Last 90 days
    cutoff = pd.Timestamp.utcnow() - pd.Timedelta(days=90)
    return df[df["timestamp"] >= cutoff]["total_score"].dropna().tolist()


def append_score_history(result: dict) -> None:
    """Append scoring result to score_history parquet."""
    if result.get("score") is None:
        return

    path = get_path("score_history")
    path.parent.mkdir(parents=True, exist_ok=True)

    row = pd.DataFrame([{
        "timestamp": pd.Timestamp.utcnow(),
        "total_score": result["score"],
        "threshold": result.get("threshold"),
        "signal": result.get("signal"),
        "block_reason": result.get("block_reason"),
        "proximity_adj": result.get("proximity_adj", 0.0),
    }])

    if path.exists():
        existing = pd.read_parquet(path)
        combined = pd.concat([existing, row], ignore_index=True)
        combined = combined.tail(8760)  # ~1 year
    else:
        combined = row

    combined.to_parquet(path, index=False)


def log_cycle(
    result: dict,
    technical: dict,
    portfolio: dict,
    fed_context: dict,
    cycle_ts: pd.Timestamp,
) -> None:
    """Append full cycle snapshot to cycle_log.parquet and print summary."""
    path = get_path("cycle_log")
    path.parent.mkdir(parents=True, exist_ok=True)

    gate_scores = result.get("gate_scores", {})
    row = {
        "timestamp": cycle_ts,
        "signal": result.get("signal"),
        "score": result.get("score"),
        "threshold": result.get("threshold"),
        "block_reason": result.get("block_reason"),
        "bb_pct": technical.get("bb_pct"),
        "rsi_14": technical.get("rsi_14"),
        "close": technical.get("close"),
        "fed_score": fed_context.get("fed_score"),
        "proximity_adj": fed_context.get("proximity_adjustment"),
        "is_blackout": fed_context.get("is_blackout"),
        "has_position": portfolio.get("has_position"),
        "capital_usd": portfolio.get("capital_usd"),
        **{f"gate_{k}": v for k, v in gate_scores.items() if isinstance(v, (int, float))},
        **{f"cluster_{k}": v for k, v in result.get("clusters", {}).items()},
    }

    if path.exists():
        existing = pd.read_parquet(path)
        combined = pd.concat([existing, pd.DataFrame([row])], ignore_index=True)
        combined = combined.tail(8760)
    else:
        combined = pd.DataFrame([row])

    combined.to_parquet(path, index=False)

    logger.info(
        f"CYCLE [{cycle_ts}]: {result.get('signal')} "
        f"score={result.get('score')} vs thr={result.get('threshold')} | "
        f"close={technical.get('close')} bb={technical.get('bb_pct')} "
        f"rsi={technical.get('rsi_14')} | "
        f"capital=${portfolio.get('capital_usd'):.2f} pos={portfolio.get('has_position')}"
    )


# ---------------------------------------------------------------------------
# Main cycle
# ---------------------------------------------------------------------------

def run_cycle() -> dict:
    """
    Execute one full hourly cycle.
    Returns result dict (useful for testing / direct calls).
    Gate scoring is wrapped in try/except — cycle never breaks the loop.
    """
    cycle_ts = pd.Timestamp.utcnow()
    params = get_params()

    # 1. Load regime (day-shifted: yesterday's regime applied today)
    regime_data = get_current_regime(today=cycle_ts.normalize())
    regime = regime_data.get("regime", "Sideways")

    # 2. Technical
    technical = get_latest_technical()
    bb_pct = technical.get("bb_pct")
    rsi_14 = technical.get("rsi_14")

    # 3. Z-scores
    zscores = load_latest_zscores()

    # 4. Stale days
    stale_days = compute_stale_days(zscores)

    # 5. News
    lookback_h = params["news"]["lookback_hours"]
    news_crypto_score = load_news_crypto_score(lookback_h)

    # 6. Fed Sentinel
    fed_context = get_fed_context(today=cycle_ts.date())

    # 7. Score history
    score_history = load_score_history()

    # 8. Gate Scoring (hardened)
    try:
        result = run_scoring_pipeline(
            regime=regime,
            bb_pct=bb_pct,
            rsi=rsi_14,
            zscores=zscores,
            stale_days=stale_days,
            news_crypto_score=news_crypto_score,
            fed_context=fed_context,
            score_history=score_history,
        )
    except Exception as e:
        logger.error(f"Gate scoring failed: {e}", exc_info=True)
        result = {
            "signal": "HOLD",
            "block_reason": f"SCORING_ERROR: {e}",
            "score": None,
            "threshold": None,
            "gate_scores": {},
            "clusters": {},
            "proximity_adj": 0.0,
        }

    # 9. Portfolio
    portfolio = load_portfolio()
    current_price = technical.get("close")

    # 10. Execution
    if current_price is not None:
        # Check stops first (position management takes priority)
        if portfolio["has_position"]:
            exit_triggered, exit_reason = check_stops(current_price, portfolio)
            if exit_triggered:
                portfolio = execute_exit(current_price, portfolio, exit_reason)
            # Reload after potential state change
            portfolio = load_portfolio()

        # Entry decision
        if result["signal"] == "ENTER" and not portfolio["has_position"]:
            portfolio = execute_entry(current_price, portfolio)

    # 11. Append score history
    append_score_history(result)

    # 11b. Stamp last_signal/score/threshold/regime onto portfolio state
    portfolio["last_signal"] = result.get("signal")
    portfolio["last_score"] = result.get("score")
    portfolio["last_threshold"] = result.get("threshold")
    portfolio["last_regime"] = regime
    portfolio["updated_at"] = str(cycle_ts)
    atomic_write_json(portfolio, get_path("portfolio_state"))

    # 12. Log cycle
    log_cycle(result, technical, portfolio, fed_context, cycle_ts)

    return result


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S UTC",
    )
    run_cycle()
