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
# Entry
# ---------------------------------------------------------------------------

def execute_entry(current_price: float, portfolio: dict) -> dict:
    """Open a paper long position. Modifies and saves portfolio state."""
    params = get_params()["execution"]
    capital = portfolio["capital_usd"]
    size_pct = params["position_size_pct"]
    sl_pct = params["stop_loss_pct"]
    tp_pct = params["take_profit_pct"]

    position_value = capital * size_pct
    quantity = position_value / current_price

    portfolio["has_position"] = True
    portfolio["entry_price"] = round(current_price, 2)
    portfolio["entry_time"] = str(pd.Timestamp.utcnow())
    portfolio["quantity"] = round(quantity, 6)
    portfolio["trailing_high"] = round(current_price, 2)
    portfolio["stop_loss_price"] = round(current_price * (1 - sl_pct), 2)
    portfolio["take_profit_price"] = round(current_price * (1 + tp_pct), 2)
    portfolio["last_updated"] = str(pd.Timestamp.utcnow())

    atomic_write_json(portfolio, get_path("portfolio_state"))
    logger.info(
        f"ENTRY: price={current_price:.2f}, qty={quantity:.6f}, "
        f"SL={portfolio['stop_loss_price']}, TP={portfolio['take_profit_price']}"
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
    trailing_pct = params["trailing_stop_pct"]

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
