"""
src/trading/execution.py — Atomic state management and order execution.

Provides:
  atomic_write_json()       — stale-write protected JSON persistence
  load_portfolio()          — load portfolio state (idempotent)
  execute_entry()           — open paper position
  execute_exit()            — close paper position
  check_stops()             — SL / TP / trailing stop logic
  parse_utc()               — safe UTC timestamp parsing

All execution parameters from parameters.yml["execution"].
Portfolio state: data/05_output/portfolio_state.json (atomic writes only).
"""

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Optional

import pandas as pd

from src.config import get_params, get_path

logger = logging.getLogger("trading.execution")


# ---------------------------------------------------------------------------
# Atomic JSON write (stale-write protected)
# ---------------------------------------------------------------------------

def atomic_write_json(data: dict, filepath: Path) -> None:
    """
    Write dict to JSON atomically: write to temp file, then os.replace.
    Prevents partial writes corrupting portfolio state.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    tmp = filepath.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2, default=str)
    os.replace(tmp, filepath)
    logger.debug(f"atomic_write_json: {filepath}")


def parse_utc(ts_str: str) -> pd.Timestamp:
    """Parse ISO timestamp string to UTC Timestamp."""
    return pd.Timestamp(ts_str, tz="UTC") if ts_str else pd.Timestamp.utcnow()


# ---------------------------------------------------------------------------
# Portfolio state
# ---------------------------------------------------------------------------

DEFAULT_PORTFOLIO = {
    "has_position": False,
    "entry_price": None,
    "entry_time": None,
    "quantity": 0.0,
    "capital_usd": None,       # filled from params on first load
    "trailing_high": None,
    "stop_loss_price": None,
    "take_profit_price": None,
    "last_updated": None,
}


def load_portfolio() -> dict:
    """Load portfolio state. Returns defaults if file doesn't exist yet."""
    path = get_path("portfolio_state")
    params = get_params()["execution"]

    if not path.exists():
        portfolio = DEFAULT_PORTFOLIO.copy()
        portfolio["capital_usd"] = params["paper_capital_usd"]
        portfolio["last_updated"] = str(pd.Timestamp.utcnow())
        atomic_write_json(portfolio, path)
        return portfolio

    with open(path) as f:
        portfolio = json.load(f)

    # Backfill capital if missing (first-time migration)
    if portfolio.get("capital_usd") is None:
        portfolio["capital_usd"] = params["paper_capital_usd"]

    return portfolio


# ---------------------------------------------------------------------------
# Dynamic stops calculation
# ---------------------------------------------------------------------------

def compute_dynamic_stops(entry_price: float, atr_14: float, params: dict) -> dict:
    """
    Calculate SL, TP and trailing dynamically based on ATR.
    ATR% is fixed once at entry — does not change mid-trade.

    Returns dict with stop percentages, prices, atr_pct, and stops_mode.
    """
    atr_pct = atr_14 / entry_price

    sl_pct = params["atr_multiplier_sl"] * atr_pct
    tp_pct = params["atr_multiplier_tp"] * atr_pct
    trail_pct = params["atr_multiplier_trail"] * atr_pct

    sl_pct    = max(params["min_stop_loss_pct"],    min(params["max_stop_loss_pct"],    sl_pct))
    tp_pct    = max(params["min_take_profit_pct"],  min(params["max_take_profit_pct"],  tp_pct))
    trail_pct = max(params["min_trailing_pct"],     min(params["max_trailing_pct"],     trail_pct))

    return {
        "stop_loss_pct":      round(sl_pct, 6),
        "take_profit_pct":    round(tp_pct, 6),
        "trailing_stop_pct":  round(trail_pct, 6),
        "stop_loss_price":    round(entry_price * (1 - sl_pct), 2),
        "take_profit_price":  round(entry_price * (1 + tp_pct), 2),
        "atr_pct":            round(atr_pct, 6),
        "stops_mode":         "dynamic",
    }


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------

def execute_entry(current_price: float, portfolio: dict, atr_14: Optional[float] = None) -> dict:
    """Open a paper long position. Modifies and saves portfolio state.

    If use_dynamic_stops=true and atr_14 is available, computes ATR-based stops.
    Falls back to fixed stops from parameters.yml if ATR is unavailable.
    """
    params = get_params()["execution"]
    capital = portfolio["capital_usd"]
    size_pct = params["position_size_pct"]

    position_value = capital * size_pct
    quantity = position_value / current_price

    # Determine stops: dynamic (ATR-based) or fixed fallback
    if params.get("use_dynamic_stops", False) and atr_14 and atr_14 > 0:
        dynamic = compute_dynamic_stops(current_price, atr_14, params)
        sl_pct    = dynamic["stop_loss_pct"]
        tp_pct    = dynamic["take_profit_pct"]
        trail_pct = dynamic["trailing_stop_pct"]
        stops_mode = "dynamic"
        atr_pct   = dynamic["atr_pct"]
        logger.info(
            f"DYNAMIC STOPS: ATR%={atr_pct:.3%} → SL={sl_pct:.2%} TP={tp_pct:.2%} Trail={trail_pct:.2%}"
        )
    else:
        sl_pct    = params["stop_loss_pct"]
        tp_pct    = params["take_profit_pct"]
        trail_pct = params["trailing_stop_pct"]
        stops_mode = "fixed"
        atr_pct   = None
        logger.info(f"FIXED STOPS (ATR unavailable): SL={sl_pct:.2%} TP={tp_pct:.2%}")

    portfolio["has_position"] = True
    portfolio["entry_price"] = round(current_price, 2)
    portfolio["entry_time"] = str(pd.Timestamp.utcnow())
    portfolio["quantity"] = round(quantity, 6)
    portfolio["trailing_high"] = round(current_price, 2)
    portfolio["stop_loss_price"] = round(current_price * (1 - sl_pct), 2)
    portfolio["take_profit_price"] = round(current_price * (1 + tp_pct), 2)
    portfolio["trailing_stop_pct_actual"] = trail_pct
    portfolio["stops_mode"] = stops_mode
    portfolio["entry_atr_pct"] = atr_pct
    portfolio["last_updated"] = str(pd.Timestamp.utcnow())

    atomic_write_json(portfolio, get_path("portfolio_state"))
    logger.info(
        f"ENTRY: price={current_price:.2f}, qty={quantity:.6f}, "
        f"SL={portfolio['stop_loss_price']}, TP={portfolio['take_profit_price']} [{stops_mode}]"
    )
    return portfolio


# ---------------------------------------------------------------------------
# Exit
# ---------------------------------------------------------------------------

def execute_exit(current_price: float, portfolio: dict, reason: str) -> dict:
    """Close paper position. Realise PnL and reset state."""
    if not portfolio["has_position"]:
        logger.warning("execute_exit called but no position open")
        return portfolio

    entry = portfolio["entry_price"]
    qty = portfolio["quantity"]
    pnl = (current_price - entry) * qty
    pnl_pct = (current_price / entry - 1) * 100

    old_capital = portfolio["capital_usd"]
    portfolio["capital_usd"] = round(old_capital + pnl, 2)
    portfolio["has_position"] = False
    portfolio["entry_price"] = None
    portfolio["entry_time"] = None
    portfolio["quantity"] = 0.0
    portfolio["trailing_high"] = None
    portfolio["stop_loss_price"] = None
    portfolio["take_profit_price"] = None
    portfolio["last_updated"] = str(pd.Timestamp.utcnow())

    atomic_write_json(portfolio, get_path("portfolio_state"))
    logger.info(
        f"EXIT [{reason}]: price={current_price:.2f}, pnl=${pnl:+.2f} ({pnl_pct:+.2f}%), "
        f"capital=${portfolio['capital_usd']:.2f}"
    )
    return portfolio


# ---------------------------------------------------------------------------
# Stop management
# ---------------------------------------------------------------------------

def check_stops(current_price: float, portfolio: dict) -> tuple[bool, str]:
    """
    Check SL, TP, and trailing stop.
    Returns (exit_triggered: bool, reason: str).
    Updates trailing_high and stop_loss_price if trailing stop moves up.
    """
    if not portfolio["has_position"]:
        return False, ""

    params = get_params()["execution"]
    # Use trailing % fixed at entry (dynamic or fixed), fall back to global param
    trailing_pct = portfolio.get("trailing_stop_pct_actual") or params["trailing_stop_pct"]

    # Update trailing high
    trailing_high = portfolio.get("trailing_high") or portfolio["entry_price"]
    if current_price > trailing_high:
        trailing_high = current_price
        trailing_stop = round(trailing_high * (1 - trailing_pct), 2)
        # Only move stop up, never down
        if trailing_stop > portfolio["stop_loss_price"]:
            portfolio["trailing_high"] = trailing_high
            portfolio["stop_loss_price"] = trailing_stop
            portfolio["last_updated"] = str(pd.Timestamp.utcnow())
            atomic_write_json(portfolio, get_path("portfolio_state"))
            logger.debug(f"Trailing stop updated to {trailing_stop}")

    # Take profit
    if current_price >= portfolio["take_profit_price"]:
        return True, "TAKE_PROFIT"

    # Stop loss (includes trailing)
    if current_price <= portfolio["stop_loss_price"]:
        return True, "STOP_LOSS"

    return False, ""
