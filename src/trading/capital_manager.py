"""
src/trading/capital_manager.py — Multi-bucket capital management.

Two buckets: btc_bot1 (Bot 1 Reversal) and btc_bot2 (Bot 2 Momentum).
Each bucket has independent capital, drawdown tracking, and safety limits.
Mutex: only one bot can hold a BTC position at a time (enforced by paper_trader).

Portfolio structure when enabled:
  portfolio["capital_usd"]        — root capital (last active bucket, backward compat)
  portfolio["total_capital_usd"]  — sum of all bucket capitals
  portfolio["buckets"]["btc_bot1"] / ["btc_bot2"]  — per-bucket state

Safety limits (per bucket):
  max_drawdown_pct   — pauses bucket if cumulative DD >= threshold
  max_daily_loss_pct — pauses bucket if daily PnL loss >= threshold
"""

import logging
from typing import Optional

import pandas as pd

logger = logging.getLogger("trading.capital_manager")

BUCKET_KEYS = ["btc_bot1", "btc_bot2"]
_BOT_TO_BUCKET = {"bot1": "btc_bot1", "bot2": "btc_bot2"}


def bot_to_bucket_key(bot: str) -> str:
    return _BOT_TO_BUCKET.get(bot, f"btc_{bot}")


def get_bucket(portfolio: dict, bucket_key: str) -> Optional[dict]:
    return portfolio.get("buckets", {}).get(bucket_key)


def init_buckets(portfolio: dict, params: dict) -> dict:
    """Initialize bucket structure if not present. Idempotent."""
    cm = params.get("capital_management", {})
    if not cm.get("enabled", False):
        return portfolio

    if "buckets" in portfolio:
        return portfolio

    total = portfolio.get("capital_usd", params.get("execution", {}).get("paper_capital_usd", 10000.0))
    buckets_cfg = cm.get("buckets", {})

    portfolio["buckets"] = {}
    for key, cfg in buckets_cfg.items():
        alloc = cfg.get("allocation_pct", 0.5)
        cap = round(float(total) * alloc, 2)
        portfolio["buckets"][key] = _new_bucket(cfg.get("name", key), cap)

    portfolio["total_capital_usd"] = float(total)
    return portfolio


def _new_bucket(name: str, capital: float) -> dict:
    return {
        "name": name,
        "current_capital": capital,
        "initial_capital": capital,
        "peak_capital": capital,
        "has_position": False,
        "entry_price": None,
        "entry_time": None,
        "quantity": 0.0,
        "trailing_high": None,
        "stop_loss_price": None,
        "take_profit_price": None,
        "daily_pnl": 0.0,
        "daily_pnl_date": None,
        "daily_capital_base": capital,
        "paused_until": None,
        "pause_reason": None,
    }


def cm_can_enter(portfolio: dict, bucket_key: str, params: dict) -> dict:
    """
    Check if bucket is allowed to enter a trade.
    Returns {can_enter: bool, reason: str}.
    """
    cm = params.get("capital_management", {})
    if not cm.get("enabled", False):
        return {"can_enter": True, "reason": "cm_disabled"}

    bucket = get_bucket(portfolio, bucket_key)
    if bucket is None:
        return {"can_enter": False, "reason": f"bucket_{bucket_key}_not_found"}

    # Pause check
    paused_until = bucket.get("paused_until")
    if paused_until:
        try:
            pause_ts = pd.Timestamp(paused_until)
            if pause_ts.tzinfo is None:
                pause_ts = pause_ts.tz_localize("UTC")
            now = pd.Timestamp.now("UTC")
            if now < pause_ts:
                remaining = (pause_ts - now).total_seconds() / 3600
                return {
                    "can_enter": False,
                    "reason": (
                        f"PAUSED ({remaining:.1f}h remaining: "
                        f"{bucket.get('pause_reason', 'unknown')})"
                    ),
                }
        except Exception:
            pass

    safety = cm.get("safety", {})

    # Max drawdown check
    max_dd_pct = safety.get("max_drawdown_pct", 0.15)
    initial = bucket.get("initial_capital", 1.0)
    current = bucket.get("current_capital", initial)
    dd_pct = (initial - current) / initial if initial > 0 else 0.0
    if dd_pct >= max_dd_pct:
        return {
            "can_enter": False,
            "reason": f"MAX_DD_REACHED ({dd_pct:.1%} >= {max_dd_pct:.1%})",
        }

    # Daily loss check
    max_daily_pct = safety.get("max_daily_loss_pct", 0.05)
    daily_pnl = bucket.get("daily_pnl", 0.0)
    daily_base = bucket.get("daily_capital_base", initial)
    if daily_base > 0 and daily_pnl < 0:
        daily_loss_pct = -daily_pnl / daily_base
        if daily_loss_pct >= max_daily_pct:
            return {
                "can_enter": False,
                "reason": (
                    f"DAILY_LOSS_LIMIT ({daily_loss_pct:.1%} >= {max_daily_pct:.1%})"
                ),
            }

    return {"can_enter": True, "reason": "ok"}


def sync_capital_for_entry(portfolio: dict, bucket_key: str) -> None:
    """
    Copy bucket's current_capital to portfolio["capital_usd"] before execute_entry.
    Ensures position sizing uses the bucket's capital, not the aggregate.
    """
    bucket = get_bucket(portfolio, bucket_key)
    if bucket is not None:
        portfolio["capital_usd"] = bucket["current_capital"]


def sync_entry_to_bucket(portfolio: dict, bucket_key: str) -> None:
    """Copy root entry fields to bucket after execute_entry."""
    bucket = get_bucket(portfolio, bucket_key)
    if bucket is None:
        return
    for field in [
        "has_position", "entry_price", "entry_time", "quantity",
        "trailing_high", "stop_loss_price", "take_profit_price",
        "trailing_stop_pct_actual", "stops_mode", "entry_atr_pct",
    ]:
        bucket[field] = portfolio.get(field)


def sync_exit_to_bucket(portfolio: dict, bucket_key: str) -> None:
    """
    After execute_exit: update bucket's current_capital from root portfolio["capital_usd"],
    clear position fields, update peak and daily PnL tracking.
    Also updates portfolio["total_capital_usd"].
    """
    bucket = get_bucket(portfolio, bucket_key)
    if bucket is None:
        return

    old_cap = bucket["current_capital"]
    new_cap = portfolio["capital_usd"]
    pnl = new_cap - old_cap

    bucket["current_capital"] = round(new_cap, 2)
    bucket["has_position"] = False
    bucket["entry_price"] = None
    bucket["entry_time"] = None
    bucket["quantity"] = 0.0
    bucket["trailing_high"] = None
    bucket["stop_loss_price"] = None
    bucket["take_profit_price"] = None

    if new_cap > bucket.get("peak_capital", 0.0):
        bucket["peak_capital"] = round(new_cap, 2)

    bucket["daily_pnl"] = round(bucket.get("daily_pnl", 0.0) + pnl, 2)

    portfolio["total_capital_usd"] = round(
        sum(b["current_capital"] for b in portfolio.get("buckets", {}).values()), 2
    )


def check_and_pause_if_needed(portfolio: dict, bucket_key: str, params: dict) -> None:
    """
    After exit: breach safety limits → set paused_until on bucket.
    Modifies bucket in-place (no disk write — caller saves portfolio).
    """
    cm = params.get("capital_management", {})
    if not cm.get("enabled", False):
        return

    bucket = get_bucket(portfolio, bucket_key)
    if bucket is None:
        return

    safety = cm.get("safety", {})
    max_dd_pct = safety.get("max_drawdown_pct", 0.15)
    max_daily_pct = safety.get("max_daily_loss_pct", 0.05)
    pause_dd_h = safety.get("pause_hours_after_dd", 72)
    pause_daily_h = safety.get("pause_hours_after_daily_loss", 24)

    initial = bucket.get("initial_capital", 1.0)
    current = bucket["current_capital"]

    dd_pct = (initial - current) / initial if initial > 0 else 0.0
    if dd_pct >= max_dd_pct:
        pause_until = pd.Timestamp.now("UTC") + pd.Timedelta(hours=pause_dd_h)
        bucket["paused_until"] = str(pause_until)
        bucket["pause_reason"] = f"MAX_DD {dd_pct:.1%}"
        logger.warning(
            f"[CM] {bucket_key} PAUSED {pause_dd_h}h: "
            f"MAX_DD {dd_pct:.1%} >= {max_dd_pct:.1%}"
        )
        return

    daily_pnl = bucket.get("daily_pnl", 0.0)
    daily_base = bucket.get("daily_capital_base", initial)
    if daily_base > 0 and daily_pnl < 0:
        daily_loss_pct = -daily_pnl / daily_base
        if daily_loss_pct >= max_daily_pct:
            pause_until = pd.Timestamp.now("UTC") + pd.Timedelta(hours=pause_daily_h)
            bucket["paused_until"] = str(pause_until)
            bucket["pause_reason"] = f"DAILY_LOSS {daily_loss_pct:.1%}"
            logger.warning(
                f"[CM] {bucket_key} PAUSED {pause_daily_h}h: "
                f"DAILY_LOSS {daily_loss_pct:.1%} >= {max_daily_pct:.1%}"
            )


def reset_daily_counters_if_needed(portfolio: dict) -> bool:
    """
    Reset daily_pnl counters if the day changed.
    Call at the start of each cycle (before entry decisions).
    Returns True if any bucket was reset.
    """
    today = str(pd.Timestamp.now("UTC").date())
    changed = False

    for bucket in portfolio.get("buckets", {}).values():
        if bucket.get("daily_pnl_date") != today:
            bucket["daily_pnl"] = 0.0
            bucket["daily_pnl_date"] = today
            bucket["daily_capital_base"] = bucket.get("current_capital", 0.0)
            changed = True

            # Expire any elapsed pauses
            paused_until = bucket.get("paused_until")
            if paused_until:
                try:
                    pause_ts = pd.Timestamp(paused_until)
                    if pause_ts.tzinfo is None:
                        pause_ts = pause_ts.tz_localize("UTC")
                    if pd.Timestamp.now("UTC") >= pause_ts:
                        bucket["paused_until"] = None
                        bucket["pause_reason"] = None
                except Exception:
                    pass

    return changed
