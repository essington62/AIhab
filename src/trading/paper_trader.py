"""
src/trading/paper_trader.py — Hourly trading cycle.

Pipeline:
  load data → R5C regime → technical → z-scores → stale check →
  news sentiment → Fed Sentinel → Gate Scoring → decision → execution → log

Hardened: gate scoring wrapped in try/except — cycle never breaks the loop.
Each step is independently logged.

MAE/MFE: while a position is open, each cycle records the price path and
tracks max_favorable (MFE) and max_adverse (MAE). On exit, the completed
trade is persisted to data/05_output/trades.parquet and
data/05_output/trade_paths.parquet.
"""

import fcntl
import json
import logging
import uuid
from datetime import datetime, timezone
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
from src.trading.capital_manager import (
    bot_to_bucket_key,
    check_and_pause_if_needed,
    cm_can_enter,
    init_buckets,
    reset_daily_counters_if_needed,
    sync_capital_for_entry,
    sync_entry_to_bucket,
    sync_exit_to_bucket,
)
from src.trading.execution import (
    atomic_write_json,
    check_stops,
    execute_entry,
    execute_exit,
    load_portfolio,
    parse_utc,
)

logger = logging.getLogger("trading.paper_trader")

_LOCK_FILE = "/tmp/aihab_trading.lock"


# ---------------------------------------------------------------------------
# Lock helpers — prevent concurrent access to portfolio_state.json
# ---------------------------------------------------------------------------

def acquire_lock():
    """Try to acquire exclusive lock. Returns file handle or None if busy."""
    try:
        lock_fd = open(_LOCK_FILE, "w")
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return lock_fd
    except (IOError, OSError):
        return None


def release_lock(lock_fd) -> None:
    """Release lock and close file handle."""
    if lock_fd:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        lock_fd.close()


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
        "score_raw": result.get("score_raw"),
        "regime_multiplier": result.get("regime_multiplier"),
        "threshold": result.get("threshold"),
        "signal": result.get("signal"),
        "block_reason": result.get("block_reason"),
        "proximity_adj": result.get("proximity_adj", 0.0),
        "entry_bot": result.get("entry_bot"),
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

    _sig = result.get("signal")
    if _sig == "FILTERED":
        logger.info(
            f"CYCLE [{cycle_ts}]: FILTERED ({result.get('filter_reason', '?')}) "
            f"score={result.get('score')} vs thr={result.get('threshold')} | "
            f"RSI={result.get('filter_rsi')} ret_1d={result.get('filter_ret_1d')} | "
            f"capital=${portfolio.get('capital_usd'):.2f} pos={portfolio.get('has_position')}"
        )
    elif _sig == "ENTER":
        logger.info(
            f"CYCLE [{cycle_ts}]: ENTER "
            f"score={result.get('score')} vs thr={result.get('threshold')} | "
            f"RSI={technical.get('rsi_14')} ret_1d={technical.get('ret_1d')} | "
            f"capital=${portfolio.get('capital_usd'):.2f} pos={portfolio.get('has_position')}"
        )
    elif _sig == "ENTER_BOT2":
        logger.info(
            f"CYCLE [{cycle_ts}]: ENTER_BOT2 "
            f"stablecoin_z={portfolio.get('entry_stablecoin_z', '?')} | "
            f"close={technical.get('close')} bb={technical.get('bb_pct')} "
            f"rsi={technical.get('rsi_14')} | "
            f"capital=${portfolio.get('capital_usd'):.2f} pos={portfolio.get('has_position')}"
        )
    else:
        logger.info(
            f"CYCLE [{cycle_ts}]: {_sig} "
            f"score={result.get('score')} vs thr={result.get('threshold')} | "
            f"close={technical.get('close')} bb={technical.get('bb_pct')} "
            f"rsi={technical.get('rsi_14')} | "
            f"capital=${portfolio.get('capital_usd'):.2f} pos={portfolio.get('has_position')}"
        )


# ---------------------------------------------------------------------------
# MAE/MFE — trade tracking helpers
# ---------------------------------------------------------------------------

def _update_excursions(portfolio: dict, current_price: float, cycle_ts: pd.Timestamp) -> None:
    """
    Update price_path, max_favorable (MFE), max_adverse (MAE) in-place.
    Persists portfolio to disk after update.
    """
    entry_price = portfolio.get("entry_price")
    if not entry_price:
        return

    current_return = (current_price - entry_price) / entry_price
    entry_time = parse_utc(portfolio.get("entry_time", str(cycle_ts)))
    hours_elapsed = (cycle_ts - entry_time).total_seconds() / 3600

    # MFE — maximum favorable excursion
    if current_return > portfolio.get("max_favorable", 0.0):
        portfolio["max_favorable"] = current_return
        portfolio["mfe_time"] = str(cycle_ts)

    # MAE — maximum adverse excursion
    if current_return < portfolio.get("max_adverse", 0.0):
        portfolio["max_adverse"] = current_return
        portfolio["mae_time"] = str(cycle_ts)

    portfolio.setdefault("price_path", []).append({
        "timestamp": str(cycle_ts),
        "price": current_price,
        "return_pct": round(current_return * 100, 4),
        "hours_since_entry": round(hours_elapsed, 2),
    })

    atomic_write_json(portfolio, get_path("portfolio_state"))


def _init_trade_tracking(
    portfolio: dict,
    result: dict,
    regime: str,
    technical: dict,
    zscores: dict,
) -> None:
    """
    Stamp scoring context + init MAE/MFE fields onto portfolio (in-place).
    Call immediately after execute_entry, before saving portfolio.
    """
    params = get_params()["execution"]
    clusters = result.get("clusters", {})

    portfolio["trade_id"] = str(uuid.uuid4())
    portfolio["max_favorable"] = 0.0
    portfolio["max_adverse"] = 0.0
    portfolio["mfe_time"] = None
    portfolio["mae_time"] = None
    portfolio["price_path"] = []

    # Scoring context at entry
    portfolio["entry_score_raw"] = result.get("score_raw")
    portfolio["entry_score_adjusted"] = result.get("score")
    portfolio["entry_regime"] = regime
    portfolio["entry_bb_pct"] = technical.get("bb_pct")
    portfolio["entry_rsi"] = technical.get("rsi_14")
    portfolio["entry_atr"] = technical.get("atr_14")
    portfolio["entry_ret_1d"] = technical.get("ret_1d")

    # Z-score snapshot at entry
    portfolio["entry_oi_z"] = zscores.get("oi_z")
    portfolio["entry_fg_raw"] = zscores.get("fg_z")

    # Cluster scores at entry
    for name in ["technical", "positioning", "macro", "liquidity", "sentiment", "news"]:
        portfolio[f"entry_cluster_{name}"] = clusters.get(name)

    # Stops configuration at entry (actual values used, dynamic or fixed)
    portfolio["entry_stop_gain_pct"] = portfolio.get("take_profit_price") and round(
        portfolio["take_profit_price"] / portfolio["entry_price"] - 1, 6
    ) or params["take_profit_pct"]
    portfolio["entry_stop_loss_pct"] = portfolio.get("stop_loss_price") and round(
        1 - portfolio["stop_loss_price"] / portfolio["entry_price"], 6
    ) or params["stop_loss_pct"]
    portfolio["entry_trailing_stop_pct"] = portfolio.get("trailing_stop_pct_actual") or params["trailing_stop_pct"]
    portfolio["entry_stops_mode"] = portfolio.get("stops_mode", "fixed")
    portfolio["entry_atr_pct"] = portfolio.get("entry_atr_pct")
    portfolio.setdefault("entry_bot", "bot1")


def _hours_since_entry(entry_time: pd.Timestamp, ts_str: Optional[str]) -> Optional[float]:
    if not ts_str:
        return None
    try:
        t = parse_utc(ts_str)
        return round((t - entry_time).total_seconds() / 3600, 2)
    except Exception:
        return None


def _build_trade_record(portfolio: dict, exit_price: float, exit_reason: str) -> dict:
    """
    Assemble completed trade record from portfolio state.
    Includes price_path (to be split out by _save_completed_trade).
    """
    entry_price = portfolio["entry_price"]
    entry_time = parse_utc(portfolio.get("entry_time", ""))
    exit_time = pd.Timestamp.utcnow()
    duration_h = (exit_time - entry_time).total_seconds() / 3600
    return_pct = (exit_price - entry_price) / entry_price

    return {
        "trade_id": portfolio.get("trade_id", str(uuid.uuid4())),
        "entry_time": str(entry_time),
        "exit_time": str(exit_time),
        "duration_hours": round(duration_h, 2),
        "entry_price": entry_price,
        "exit_price": exit_price,
        "return_pct": round(return_pct * 100, 4),
        "stop_gain_pct": portfolio.get("entry_stop_gain_pct"),
        "stop_loss_pct": portfolio.get("entry_stop_loss_pct"),
        "trailing_stop_pct": portfolio.get("entry_trailing_stop_pct"),
        "exit_reason": exit_reason,
        # MAE/MFE
        "mae_pct": round(portfolio.get("max_adverse", 0.0) * 100, 4),
        "mfe_pct": round(portfolio.get("max_favorable", 0.0) * 100, 4),
        "mae_time": portfolio.get("mae_time"),
        "mfe_time": portfolio.get("mfe_time"),
        "hours_to_mfe": _hours_since_entry(entry_time, portfolio.get("mfe_time")),
        # Entry context
        "entry_score_raw": portfolio.get("entry_score_raw"),
        "entry_score_adjusted": portfolio.get("entry_score_adjusted"),
        "entry_regime": portfolio.get("entry_regime"),
        "entry_bb_pct": portfolio.get("entry_bb_pct"),
        "entry_rsi": portfolio.get("entry_rsi"),
        "entry_atr": portfolio.get("entry_atr"),
        "entry_ret_1d": portfolio.get("entry_ret_1d"),
        "entry_filter_passed": portfolio.get("entry_filter_passed", True),
        "entry_bot": portfolio.get("entry_bot", "bot1"),
        "entry_stablecoin_z": portfolio.get("entry_stablecoin_z"),
        "entry_max_hold_hours": portfolio.get("entry_max_hold_hours"),
        "entry_oi_z": portfolio.get("entry_oi_z"),
        "entry_fg_raw": portfolio.get("entry_fg_raw"),
        "entry_cluster_technical": portfolio.get("entry_cluster_technical"),
        "entry_cluster_positioning": portfolio.get("entry_cluster_positioning"),
        "entry_cluster_macro": portfolio.get("entry_cluster_macro"),
        "entry_cluster_liquidity": portfolio.get("entry_cluster_liquidity"),
        "entry_cluster_sentiment": portfolio.get("entry_cluster_sentiment"),
        "entry_cluster_news": portfolio.get("entry_cluster_news"),
        # Dynamic stops info
        "stops_mode": portfolio.get("stops_mode", "fixed"),
        "entry_atr_pct": portfolio.get("entry_atr_pct"),
        "actual_stop_loss_pct": portfolio.get("entry_stop_loss_pct"),
        "actual_take_profit_pct": portfolio.get("entry_stop_gain_pct"),
        "actual_trailing_pct": portfolio.get("trailing_stop_pct_actual"),
        # Price path — extracted by _save_completed_trade
        "_price_path": portfolio.get("price_path", []),
    }


def _save_completed_trade(record: dict) -> None:
    """
    Persist completed trade to trades.parquet.
    Splits price_path into trade_paths.parquet.
    """
    price_path_data = record.pop("_price_path", [])

    # trades.parquet — one scalar row per trade
    trades_path = get_path("trades")
    trades_path.parent.mkdir(parents=True, exist_ok=True)
    trade_row = pd.DataFrame([record])

    if trades_path.exists():
        existing = pd.read_parquet(trades_path)
        combined = pd.concat([existing, trade_row], ignore_index=True)
    else:
        combined = trade_row

    combined.to_parquet(trades_path, index=False)
    logger.info(
        f"Trade saved: id={record.get('trade_id')} "
        f"return={record.get('return_pct'):+.3f}% "
        f"mae={record.get('mae_pct'):.3f}% mfe={record.get('mfe_pct'):.3f}% "
        f"reason={record.get('exit_reason')}"
    )

    # trade_paths.parquet — one row per price observation
    if price_path_data:
        paths_path = get_path("trade_paths")
        paths_path.parent.mkdir(parents=True, exist_ok=True)
        for obs in price_path_data:
            obs["trade_id"] = record.get("trade_id")
        paths_df = pd.DataFrame(price_path_data)
        if paths_path.exists():
            existing_paths = pd.read_parquet(paths_path)
            paths_df = pd.concat([existing_paths, paths_df], ignore_index=True)
        paths_df.to_parquet(paths_path, index=False)


# ---------------------------------------------------------------------------
# Cooldown (prevents immediate reentry after a stop loss)
# ---------------------------------------------------------------------------

def check_cooldown(portfolio: dict, current_price: float, bot: str, params: dict) -> dict:
    """
    Check if cooldown is active after a stop loss.

    Returns dict with keys: can_enter (bool), reason (str),
    hours_remaining (float), consecutive_sl (int), price_above_exit (bool|None).
    """
    if bot == "bot1":
        cd_cfg = params.get("reversal_filter", {}).get("cooldown", {})
    else:
        cd_cfg = params.get("momentum_filter", {}).get("cooldown", {})

    if not cd_cfg.get("enabled", False):
        return {"can_enter": True, "reason": "cooldown_disabled"}

    last_sl_time  = portfolio.get("last_sl_time")
    last_sl_price = portfolio.get("last_sl_price")
    last_sl_bot   = portfolio.get("last_sl_bot")
    consecutive_sl = portfolio.get("consecutive_sl_count", 0)

    if not last_sl_time:
        return {"can_enter": True, "reason": "no_previous_sl"}

    # SL from a different bot does not block this bot
    if last_sl_bot and last_sl_bot != bot:
        return {"can_enter": True, "reason": f"sl_was_{last_sl_bot}_not_{bot}"}

    try:
        sl_time = pd.Timestamp(last_sl_time)
        if sl_time.tzinfo is None:
            sl_time = sl_time.tz_localize("UTC")
        now = pd.Timestamp.now("UTC")
        hours_since_sl = (now - sl_time).total_seconds() / 3600
    except Exception:
        return {"can_enter": True, "reason": "invalid_sl_time"}

    hours_required = cd_cfg.get("hours_after_sl", 12)
    max_consecutive = cd_cfg.get("max_consecutive_sl", 3)
    pause_hours = cd_cfg.get("consecutive_sl_pause_hours", 24)

    if consecutive_sl >= max_consecutive:
        hours_required = pause_hours

    if hours_since_sl < hours_required:
        hours_remaining = hours_required - hours_since_sl
        consec_suffix = f", CONSECUTIVE_SL={consecutive_sl}" if consecutive_sl >= max_consecutive else ""
        return {
            "can_enter": False,
            "reason": (
                f"COOLDOWN ({hours_since_sl:.1f}h / {hours_required}h required{consec_suffix})"
            ),
            "hours_remaining": hours_remaining,
            "consecutive_sl": consecutive_sl,
            "price_above_exit": None,
        }

    require_above = cd_cfg.get("require_price_above_exit", True)
    if require_above and last_sl_price:
        bounce_pct = cd_cfg.get("bounce_pct", 0.003)
        min_reentry_price = last_sl_price * (1 + bounce_pct)
        if current_price <= min_reentry_price:
            return {
                "can_enter": False,
                "reason": (
                    f"PRICE_BELOW_BOUNCE (current={current_price:.0f} <= "
                    f"min={min_reentry_price:.0f} [{last_sl_price:.0f} × {1 + bounce_pct:.3f}])"
                ),
                "hours_remaining": 0,
                "consecutive_sl": consecutive_sl,
                "price_above_exit": False,
            }

    return {
        "can_enter": True,
        "reason": "cooldown_passed",
        "hours_remaining": 0,
        "consecutive_sl": consecutive_sl,
        "price_above_exit": True,
    }


# ---------------------------------------------------------------------------
# Reversal filter (entry confirmation — applied after scoring says ENTER)
# ---------------------------------------------------------------------------

def check_momentum_filter(technical: dict, zscores: dict, params: dict) -> dict:
    """
    Bot 2 — Momentum/Liquidez filter.
    Entry when stablecoin liquidity is flowing in + market in uptrend.
    Independent of gate scoring — runs as alternative entry path.

    Conditions (ALL must be true):
      1. stablecoin_z > 1.3 (strong liquidity inflow)
      2. ret_1d > 0 (positive momentum)
      3. RSI > 50 (bullish zone)
      4. close > MA21 (short-term uptrend)
      5. BB% < 0.98 (not a blow-off top)
    """
    mf = params.get("momentum_filter", {})

    if not mf.get("enabled", False):
        return {"passed": False, "reason": "filter_disabled"}

    stablecoin_z = zscores.get("stablecoin_z")
    ret_1d = technical.get("ret_1d")
    rsi = technical.get("rsi_14")
    bb_pct = technical.get("bb_pct")
    close = technical.get("close")
    ma_21 = technical.get("ma_21")

    sz_min = mf.get("stablecoin_z_min", 1.3)
    ret_min = mf.get("ret_1d_min", 0.0)
    rsi_min = mf.get("rsi_min", 50)
    bb_max = mf.get("bb_pct_max", 0.98)
    require_ma21 = mf.get("require_above_ma21", True)

    result = {
        "stablecoin_z": stablecoin_z,
        "ret_1d": ret_1d,
        "rsi": rsi,
        "bb_pct": bb_pct,
        "close": close,
        "ma_21": ma_21,
        "sz_min": sz_min,
        "ret_min": ret_min,
        "rsi_min": rsi_min,
        "bb_max": bb_max,
    }

    if stablecoin_z is None or ret_1d is None or rsi is None or bb_pct is None:
        result["passed"] = False
        result["reason"] = "MISSING_DATA"
        return result

    if require_ma21 and (close is None or ma_21 is None):
        result["passed"] = False
        result["reason"] = "MISSING_MA21"
        return result

    reasons = []

    if stablecoin_z <= sz_min:
        reasons.append(f"LOW_LIQUIDITY (stable_z={stablecoin_z:.2f} <= {sz_min})")

    if ret_1d <= ret_min:
        reasons.append(f"NEG_MOMENTUM (ret_1d={ret_1d:.4f} <= {ret_min})")

    if rsi <= rsi_min:
        reasons.append(f"RSI_LOW (RSI={rsi:.1f} <= {rsi_min})")

    if bb_pct >= bb_max:
        reasons.append(f"BLOW_OFF_TOP (BB={bb_pct:.3f} >= {bb_max})")

    if require_ma21 and close <= ma_21:
        reasons.append(f"BELOW_MA21 (close={close:.0f} <= MA21={ma_21:.0f})")

    spike_cfg = mf.get("spike_guard", {})
    if spike_cfg.get("enabled", False):
        spike_ret_max = spike_cfg.get("spike_ret_max", 0.03)
        spike_rsi_max = spike_cfg.get("spike_rsi_max", 65)
        if ret_1d > spike_ret_max and rsi > spike_rsi_max:
            reasons.append(
                f"LATE_SPIKE (ret_1d={ret_1d:.4f} > {spike_ret_max} "
                f"AND RSI={rsi:.1f} > {spike_rsi_max})"
            )

    if reasons:
        result["passed"] = False
        result["reason"] = " & ".join(reasons)
        return result

    result["passed"] = True
    result["reason"] = "momentum_confirmed"
    return result


def check_momentum_filter_v2(technical: dict, zscores: dict, params: dict) -> dict:
    """
    Bot 2 v2 — Momentum with Early Reversal.

    Two clauses (entry if EITHER passes):
      CLASSIC:  ret_1d > 0, RSI > 50, close > MA21
      EARLY:    ret_1d > -1.5%, trend_improving (3h), delta_ret_3h > 0.5%, RSI > 35

    Global filters always apply: stablecoin_z > sz_min, BB% < bb_max.
    """
    mf = params.get("momentum_filter_v2", {})

    if not mf.get("enabled", False):
        return {"passed": False, "reason": "v2_disabled", "entry_mode": None}

    stablecoin_z = zscores.get("stablecoin_z")
    ret_1d = technical.get("ret_1d")
    ret_1d_1h = technical.get("ret_1d_1h_ago")
    ret_1d_3h = technical.get("ret_1d_3h_ago")
    rsi = technical.get("rsi_14")
    bb_pct = technical.get("bb_pct")
    close = technical.get("close")
    ma_21 = technical.get("ma_21")

    sz_min = mf.get("stablecoin_z_min", 1.3)
    bb_max = mf.get("bb_pct_max", 0.98)
    classic_cfg = mf.get("classic", {})
    early_cfg = mf.get("early_reversal", {})

    result = {
        "stablecoin_z": stablecoin_z,
        "ret_1d": ret_1d,
        "rsi": rsi,
        "bb_pct": bb_pct,
        "close": close,
        "ma_21": ma_21,
        "classic_pass": False,
        "early_pass": False,
        "entry_mode": None,
    }

    if any(v is None for v in [stablecoin_z, ret_1d, rsi, bb_pct, close, ma_21]):
        result["passed"] = False
        result["reason"] = "MISSING_DATA"
        return result

    if stablecoin_z <= sz_min:
        result["passed"] = False
        result["reason"] = f"LOW_LIQUIDITY (sz={stablecoin_z:.2f})"
        return result

    if bb_pct >= bb_max:
        result["passed"] = False
        result["reason"] = f"BLOW_OFF_TOP (bb={bb_pct:.3f})"
        return result

    classic_pass = (
        ret_1d > classic_cfg.get("ret_1d_min", 0.0)
        and rsi > classic_cfg.get("rsi_min", 50)
        and close > ma_21
    )
    result["classic_pass"] = classic_pass

    early_pass = False
    if early_cfg.get("enabled", True) and ret_1d_1h is not None and ret_1d_3h is not None:
        trend_improving = (ret_1d > ret_1d_1h) and (ret_1d_1h > ret_1d_3h)
        delta_3h = ret_1d - ret_1d_3h
        early_pass = (
            ret_1d > early_cfg.get("ret_1d_floor", -0.015)
            and trend_improving
            and delta_3h > early_cfg.get("delta_ret_3h_min", 0.005)
            and rsi > early_cfg.get("rsi_floor", 35)
        )
        result["early_pass"] = early_pass
        result["trend_improving"] = trend_improving
        result["delta_3h"] = round(delta_3h, 4)

    if classic_pass:
        result["passed"] = True
        result["reason"] = "classic"
        result["entry_mode"] = "classic"
    elif early_pass:
        result["passed"] = True
        result["reason"] = "early_reversal"
        result["entry_mode"] = "early"
    else:
        result["passed"] = False
        result["reason"] = "no_trigger"

    return result


def check_reversal_filter(technical: dict, params: dict) -> dict:
    """
    Verify reversal conditions before executing an entry.
    Scoring already decided ENTER; this adds technical confirmation:
    - RSI < rsi_max (oversold)
    - ret_1d > ret_1d_min (sell-off decelerating, not a falling knife)
    - Exception: RSI < rsi_extreme_override overrides ret_1d check (extreme capitulation)

    Accumulates ALL failure reasons (not short-circuit).

    Returns dict with {passed, reason, rsi, ret_1d, rsi_max, ret_1d_min}.
    """
    rf = params.get("reversal_filter", {})

    if not rf.get("enabled", False):
        return {
            "passed": True,
            "reason": "filter_disabled",
            "rsi": technical.get("rsi_14"),
            "ret_1d": technical.get("ret_1d"),
            "rsi_max": None,
            "ret_1d_min": None,
        }

    rsi_max = rf.get("rsi_max", 35)
    ret_1d_min = rf.get("ret_1d_min", -0.01)
    rsi_extreme = rf.get("rsi_extreme_override", 25)

    rsi = technical.get("rsi_14")
    ret_1d = technical.get("ret_1d")

    if rsi is None or ret_1d is None:
        return {
            "passed": False,
            "reason": (
                f"FILTER_DATA_MISSING (rsi={'N/A' if rsi is None else f'{rsi:.1f}'}, "
                f"ret_1d={'N/A' if ret_1d is None else f'{ret_1d:.4f}'})"
            ),
            "rsi": rsi,
            "ret_1d": ret_1d,
            "rsi_max": rsi_max,
            "ret_1d_min": ret_1d_min,
        }

    reasons = []

    if rsi >= rsi_max:
        reasons.append(f"RSI_TOO_HIGH ({rsi:.1f} >= {rsi_max})")

    if ret_1d <= ret_1d_min:
        if rsi_extreme > 0 and rsi < rsi_extreme:
            logger.info(
                f"EXTREME_CAPITULATION: RSI={rsi:.1f} < {rsi_extreme} → "
                f"overriding ret_1d filter (ret_1d={ret_1d:.4f})"
            )
        else:
            reasons.append(f"FALLING_KNIFE (ret_1d={ret_1d:.4f} <= {ret_1d_min})")

    if reasons:
        return {
            "passed": False,
            "reason": " & ".join(reasons),
            "rsi": rsi,
            "ret_1d": ret_1d,
            "rsi_max": rsi_max,
            "ret_1d_min": ret_1d_min,
        }

    return {
        "passed": True,
        "reason": "reversal_confirmed",
        "rsi": rsi,
        "ret_1d": ret_1d,
        "rsi_max": rsi_max,
        "ret_1d_min": ret_1d_min,
    }


# ---------------------------------------------------------------------------
# Lightweight stop check (15-min cycle)
# ---------------------------------------------------------------------------

def check_stops_only() -> dict:
    """
    Lightweight stop check — runs every 15min.
    No gate/regime recalculation. Only reads current price vs stops.
    Updates trailing_high and MAE/MFE if no stop triggered.
    Exits the position if any stop is reached.

    Returns:
        dict with {action: 'no_position'|'hold'|'exit'|'error', ...}
    """
    cycle_ts = pd.Timestamp.utcnow()

    portfolio = load_portfolio()
    if not portfolio.get("has_position"):
        logger.info("[STOPS-15m] No open position — skipping")
        return {"action": "no_position"}

    try:
        technical = get_latest_technical()
        current_price = technical.get("close")
        if current_price is None:
            raise ValueError("close price is None")
    except Exception as e:
        logger.warning(f"[STOPS-15m] WARN: could not fetch price, skipping cycle: {e}")
        return {"action": "error", "error": str(e)}

    # Update MAE/MFE and price path (saves portfolio)
    _update_excursions(portfolio, current_price, cycle_ts)

    # Check stops (also updates trailing_high in portfolio dict + saves if moved)
    exit_triggered, exit_reason = check_stops(current_price, portfolio)

    if exit_triggered:
        completed_trade = _build_trade_record(portfolio, current_price, exit_reason)
        portfolio = execute_exit(current_price, portfolio, exit_reason)
        _save_completed_trade(completed_trade)
        return_pct = (current_price - completed_trade["entry_price"]) / completed_trade["entry_price"]
        logger.info(
            f"[STOPS-15m] EXIT by {exit_reason} | price=${current_price:,.0f} | "
            f"return={return_pct:+.2%}"
        )
        return {
            "action": "exit",
            "reason": exit_reason,
            "price": current_price,
            "return_pct": return_pct,
        }

    trailing_high = portfolio.get("trailing_high") or 0.0
    sg = portfolio.get("take_profit_price") or 0.0
    sl = portfolio.get("stop_loss_price") or 0.0
    logger.info(
        f"[STOPS-15m] HOLD | price=${current_price:,.0f} | "
        f"SG=${sg:,.0f} | SL=${sl:,.0f} | trailing_high=${trailing_high:,.0f}"
    )
    return {
        "action": "hold",
        "price": current_price,
        "trailing_high": trailing_high,
        "stop_gain": sg,
        "stop_loss": sl,
    }


# ---------------------------------------------------------------------------
# Bot 2 entry execution
# ---------------------------------------------------------------------------

def _execute_bot2_entry(current_price: float, portfolio: dict, mf_params: dict) -> dict:
    """Execute Bot 2 entry with its own fixed stops (not ATR-based)."""
    params = get_params()["execution"]
    capital = portfolio["capital_usd"]
    size_pct = params["position_size_pct"]

    position_value = capital * size_pct
    quantity = position_value / current_price

    sl_pct = mf_params.get("stop_loss_pct", 0.015)
    tp_pct = mf_params.get("take_profit_pct", 0.02)
    trail_pct = mf_params.get("trailing_stop_pct", 0.01)

    portfolio["has_position"] = True
    portfolio["entry_price"] = round(current_price, 2)
    portfolio["entry_time"] = str(pd.Timestamp.utcnow())
    portfolio["quantity"] = round(quantity, 6)
    portfolio["trailing_high"] = round(current_price, 2)
    portfolio["stop_loss_price"] = round(current_price * (1 - sl_pct), 2)
    portfolio["take_profit_price"] = round(current_price * (1 + tp_pct), 2)
    portfolio["trailing_stop_pct_actual"] = trail_pct
    portfolio["stops_mode"] = "bot2_fixed"
    portfolio["entry_atr_pct"] = None
    portfolio["last_updated"] = str(pd.Timestamp.utcnow())

    atomic_write_json(portfolio, get_path("portfolio_state"))
    logger.info(
        f"BOT2 ENTRY: price={current_price:.2f}, qty={quantity:.6f}, "
        f"SL={portfolio['stop_loss_price']}, TP={portfolio['take_profit_price']} [bot2_fixed]"
    )
    return portfolio


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

    # 7b. Spot df for MA200 override (1h clean → enough history)
    spot_df = None
    try:
        _sp = pd.read_parquet(get_path("clean_spot_1h"))
        _sp["timestamp"] = pd.to_datetime(_sp["timestamp"], utc=True)
        spot_df = _sp.sort_values("timestamp").reset_index(drop=True)
    except Exception as e:
        logger.warning(f"spot_df load failed (MA200 override disabled): {e}")

    # 7c. Daily data for adaptive weights (zs_daily + spot_daily)
    zs_daily = None
    spot_daily = None
    try:
        _zs_df = pd.read_parquet(get_path("gate_zscores"))
        _zs_df["timestamp"] = pd.to_datetime(_zs_df["timestamp"], utc=True)
        zs_daily = _zs_df.set_index("timestamp").resample("1D").last()

        _spot_raw = pd.read_parquet(get_path("clean_spot_1h"))
        _spot_raw["timestamp"] = pd.to_datetime(_spot_raw["timestamp"], utc=True)
        spot_daily = _spot_raw.set_index("timestamp").resample("1D")["close"].last()
    except Exception as e:
        logger.warning(f"adaptive weights data load failed (using base weights): {e}")

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
            spot_df=spot_df,
            zs_daily=zs_daily,
            spot_daily=spot_daily,
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
    # 9b. Capital manager: init buckets (idempotent) + reset daily counters
    cm_params = params.get("capital_management", {})
    if cm_params.get("enabled", False):
        init_buckets(portfolio, params)
        if reset_daily_counters_if_needed(portfolio):
            atomic_write_json(portfolio, get_path("portfolio_state"))

    current_price = technical.get("close")

    # 10. Execution
    if current_price is not None:
        # Check stops first (position management takes priority)
        if portfolio["has_position"]:
            # Update MAE/MFE and price path before stop evaluation
            _update_excursions(portfolio, current_price, cycle_ts)

            exit_triggered, exit_reason = check_stops(current_price, portfolio)
            if exit_triggered:
                # Build completed trade record BEFORE execute_exit clears portfolio state
                completed_trade = _build_trade_record(portfolio, current_price, exit_reason)
                _exiting_bot = portfolio.get("entry_bot", "bot1")
                portfolio = execute_exit(current_price, portfolio, exit_reason)
                _save_completed_trade(completed_trade)
                # Capital manager: update bucket capital + check safety limits
                if cm_params.get("enabled", False):
                    _bucket_key = bot_to_bucket_key(_exiting_bot)
                    sync_exit_to_bucket(portfolio, _bucket_key)
                    check_and_pause_if_needed(portfolio, _bucket_key, params)
                # Cooldown tracking: set on SL, reset counter on successful exit
                if exit_reason == "STOP_LOSS":
                    portfolio["last_sl_time"]  = str(pd.Timestamp.now("UTC"))
                    portfolio["last_sl_price"] = current_price
                    portfolio["last_sl_bot"]   = _exiting_bot
                    portfolio["consecutive_sl_count"] = portfolio.get("consecutive_sl_count", 0) + 1
                    logger.info(
                        f"COOLDOWN SET: bot={_exiting_bot} "
                        f"consecutive_sl={portfolio['consecutive_sl_count']} "
                        f"price={current_price:.0f}"
                    )
                else:
                    portfolio["consecutive_sl_count"] = 0
                atomic_write_json(portfolio, get_path("portfolio_state"))

            # Reload after potential state change
            portfolio = load_portfolio()

        # Entry decision
        bot_entered = None
        mf_check = {"passed": False, "reason": "not_evaluated", "stablecoin_z": None}

        # Bot 2 max hold timeout check (runs before entry, while position is open)
        if portfolio["has_position"]:
            entry_bot = portfolio.get("entry_bot", "bot1")
            if entry_bot == "bot2":
                max_hold_h = portfolio.get("entry_max_hold_hours", 120)
                entry_time_str = portfolio.get("entry_time", "")
                if entry_time_str:
                    entry_time_dt = parse_utc(entry_time_str)
                    hours_in_trade = (cycle_ts - entry_time_dt).total_seconds() / 3600
                    if hours_in_trade >= max_hold_h:
                        completed_trade = _build_trade_record(portfolio, current_price, "bot2_timeout")
                        portfolio = execute_exit(current_price, portfolio, "bot2_timeout")
                        _save_completed_trade(completed_trade)
                        if cm_params.get("enabled", False):
                            sync_exit_to_bucket(portfolio, "btc_bot2")
                            check_and_pause_if_needed(portfolio, "btc_bot2", params)
                        portfolio["consecutive_sl_count"] = 0
                        atomic_write_json(portfolio, get_path("portfolio_state"))
                        portfolio = load_portfolio()
                        logger.info(
                            f"BOT2 TIMEOUT: {hours_in_trade:.0f}h >= {max_hold_h}h | "
                            f"exit=${current_price:,.0f}"
                        )

        # Bot 1: Reversal — scoring says ENTER → check cooldown → apply reversal filter
        cd_check: dict = {"can_enter": True, "reason": "not_evaluated"}
        if result["signal"] == "ENTER" and not portfolio["has_position"]:
            cd_check = check_cooldown(portfolio, current_price, "bot1", params)
            if not cd_check["can_enter"]:
                logger.info(f"BOT1 COOLDOWN ACTIVE: {cd_check['reason']}")
                result["signal"] = "COOLDOWN"
                result["cooldown_reason"] = cd_check["reason"]
            else:
                # Capital manager gate
                cm_check_b1: dict = {"can_enter": True, "reason": "cm_disabled"}
                if cm_params.get("enabled", False):
                    cm_check_b1 = cm_can_enter(portfolio, "btc_bot1", params)
                    if not cm_check_b1["can_enter"]:
                        logger.info(f"BOT1 CM BLOCKED: {cm_check_b1['reason']}")
                        result["signal"] = "HOLD"

                if cm_check_b1["can_enter"]:
                    rf_check = check_reversal_filter(technical, params)

                    if rf_check["passed"]:
                        atr_14 = technical.get("atr_14")
                        if cm_params.get("enabled", False):
                            sync_capital_for_entry(portfolio, "btc_bot1")
                        portfolio = execute_entry(current_price, portfolio, atr_14=atr_14)
                        if cm_params.get("enabled", False):
                            sync_entry_to_bucket(portfolio, "btc_bot1")
                        _init_trade_tracking(portfolio, result, regime, technical, zscores)
                        portfolio["entry_ret_1d"] = rf_check["ret_1d"]
                        portfolio["entry_filter_passed"] = True
                        portfolio["entry_bot"] = "bot1"
                        atomic_write_json(portfolio, get_path("portfolio_state"))
                        bot_entered = "bot1"
                        logger.info(
                            f"BOT1 ENTRY CONFIRMED: score={result.get('score', 0):.3f} | "
                            f"RSI={rf_check['rsi']:.1f} (<{rf_check['rsi_max']}) | "
                            f"ret_1d={rf_check['ret_1d']:.4f} (>{rf_check['ret_1d_min']})"
                        )
                    else:
                        logger.info(
                            f"BOT1 ENTRY FILTERED: score={result.get('score', 0):.3f} | "
                            f"RSI={rf_check['rsi']} | ret_1d={rf_check['ret_1d']} → "
                            f"{rf_check['reason']}"
                        )
                        result["signal"] = "FILTERED"
                        result["filter_reason"] = rf_check["reason"]
                        result["filter_rsi"] = rf_check["rsi"]
                        result["filter_ret_1d"] = rf_check["ret_1d"]

        # Bot 2: Momentum — independent of gate scoring, runs if no position and not Bear
        if bot_entered is None and not portfolio["has_position"] and regime != "Bear":
            cd_check_b2 = check_cooldown(portfolio, current_price, "bot2", params)
            if not cd_check_b2["can_enter"]:
                logger.info(f"BOT2 COOLDOWN ACTIVE: {cd_check_b2['reason']}")
                mf_check = {"passed": False, "reason": cd_check_b2["reason"], "stablecoin_z": None}
            else:
                # Capital manager gate
                if cm_params.get("enabled", False):
                    cm_check_b2 = cm_can_enter(portfolio, "btc_bot2", params)
                    if not cm_check_b2["can_enter"]:
                        logger.info(f"BOT2 CM BLOCKED: {cm_check_b2['reason']}")
                        mf_check = {"passed": False, "reason": cm_check_b2["reason"], "stablecoin_z": None}
                    else:
                        mf_check = check_momentum_filter(technical, zscores, params)
                else:
                    mf_check = check_momentum_filter(technical, zscores, params)

            if mf_check["passed"]:
                mf_params = params.get("momentum_filter", {})
                if cm_params.get("enabled", False):
                    sync_capital_for_entry(portfolio, "btc_bot2")
                portfolio = _execute_bot2_entry(current_price, portfolio, mf_params)
                if cm_params.get("enabled", False):
                    sync_entry_to_bucket(portfolio, "btc_bot2")
                _init_trade_tracking(portfolio, result, regime, technical, zscores)
                portfolio["entry_bot"] = "bot2"
                portfolio["entry_stablecoin_z"] = mf_check["stablecoin_z"]
                portfolio["entry_filter_passed"] = True
                portfolio["entry_max_hold_hours"] = mf_params.get("max_hold_hours", 120)
                atomic_write_json(portfolio, get_path("portfolio_state"))
                bot_entered = "bot2"
                result["signal"] = "ENTER_BOT2"
                logger.info(
                    f"BOT2 ENTRY CONFIRMED: stablecoin_z={mf_check['stablecoin_z']:.2f} | "
                    f"ret_1d={mf_check['ret_1d']:.4f} | RSI={mf_check['rsi']:.1f} | "
                    f"BB={mf_check['bb_pct']:.3f}"
                )
            else:
                if mf_check.get("stablecoin_z") and mf_check["stablecoin_z"] > 0.5:
                    logger.info(
                        f"BOT2 FILTERED: {mf_check['reason']} | "
                        f"stable_z={mf_check.get('stablecoin_z')}"
                    )

    # 11. Append score history
    append_score_history(result)

    # 11b. Stamp last_signal/score/threshold/regime onto portfolio state
    portfolio["last_signal"] = result.get("signal")
    portfolio["last_score"] = result.get("score")
    portfolio["last_score_raw"] = result.get("score_raw")
    portfolio["last_regime_multiplier"] = result.get("regime_multiplier")
    portfolio["last_threshold"] = result.get("threshold")
    portfolio["last_regime"] = regime
    portfolio["updated_at"] = str(cycle_ts)
    # Reversal filter state — persisted every cycle for dashboard/debug
    portfolio["last_filter_passed"] = result.get("signal") != "FILTERED"
    portfolio["last_filter_reason"] = result.get("filter_reason")
    portfolio["last_filter_rsi"] = technical.get("rsi_14")
    portfolio["last_filter_ret_1d"] = technical.get("ret_1d")
    # Momentum filter state — persisted every cycle for dashboard/debug
    portfolio["last_momentum_passed"] = result.get("signal") == "ENTER_BOT2"
    portfolio["last_momentum_reason"] = mf_check.get("reason")
    portfolio["last_momentum_stablecoin_z"] = zscores.get("stablecoin_z")
    # Cooldown state — persisted every cycle for dashboard
    portfolio["last_cooldown_can_enter"] = cd_check.get("can_enter")
    portfolio["last_cooldown_reason"] = cd_check.get("reason")
    # Adaptive weights state — persisted for dashboard
    if result.get("adaptive_weights"):
        portfolio["last_adaptive_weights"] = result["adaptive_weights"]
    portfolio["last_global_confidence_multiplier"] = result.get("global_confidence_multiplier", 1.0)
    portfolio["last_global_confidence_source"] = result.get("global_confidence_source", "")
    portfolio["last_score_after_regime"] = result.get("score_after_regime")
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
