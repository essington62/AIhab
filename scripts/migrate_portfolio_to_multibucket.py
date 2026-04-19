#!/usr/bin/env python3
"""
Migrate portfolio_state.json from single-capital to multi-bucket (Opção A).

Creates:
  portfolio["buckets"]["btc_bot1"] — 50% of current capital
  portfolio["buckets"]["btc_bot2"] — 50% of current capital
  portfolio["total_capital_usd"]   — current total

Idempotent: exits cleanly if already migrated.
Backup: creates portfolio_state.json.backup_premigration before modifying.
"""
import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.config import get_path


def _new_bucket(name: str, cap: float, has_pos: bool, portfolio: dict) -> dict:
    return {
        "name": name,
        "current_capital": cap,
        "initial_capital": cap,
        "peak_capital": cap,
        "has_position": has_pos,
        "entry_price": portfolio.get("entry_price") if has_pos else None,
        "entry_time": portfolio.get("entry_time") if has_pos else None,
        "quantity": portfolio.get("quantity", 0.0) if has_pos else 0.0,
        "trailing_high": portfolio.get("trailing_high") if has_pos else None,
        "stop_loss_price": portfolio.get("stop_loss_price") if has_pos else None,
        "take_profit_price": portfolio.get("take_profit_price") if has_pos else None,
        "daily_pnl": 0.0,
        "daily_pnl_date": None,
        "daily_capital_base": cap,
        "paused_until": None,
        "pause_reason": None,
    }


def main():
    path = get_path("portfolio_state")

    if not path.exists():
        print("portfolio_state.json not found — nothing to migrate.")
        return

    with open(path) as f:
        portfolio = json.load(f)

    if "buckets" in portfolio:
        print("Already migrated (buckets key exists). Exiting.")
        return

    # Backup
    backup = path.with_suffix(".json.backup_premigration")
    shutil.copy2(path, backup)
    print(f"Backup saved: {backup}")

    total = float(portfolio.get("capital_usd", 10000.0))
    half = round(total / 2, 2)
    # Ensure buckets add up exactly to total (handle rounding)
    b1_cap = half
    b2_cap = round(total - half, 2)

    entry_bot = portfolio.get("entry_bot", "bot1") if portfolio.get("has_position") else None
    b1_has_pos = entry_bot == "bot1"
    b2_has_pos = entry_bot == "bot2"

    portfolio["total_capital_usd"] = total
    portfolio["buckets"] = {
        "btc_bot1": _new_bucket("Bot 1 (Reversal)", b1_cap, b1_has_pos, portfolio),
        "btc_bot2": _new_bucket("Bot 2 (Momentum)", b2_cap, b2_has_pos, portfolio),
    }

    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(portfolio, f, indent=2, default=str)
    os.replace(tmp, path)

    print(
        f"Migration complete.\n"
        f"  total_capital_usd = ${total:,.2f}\n"
        f"  btc_bot1 capital  = ${b1_cap:,.2f} (has_position={b1_has_pos})\n"
        f"  btc_bot2 capital  = ${b2_cap:,.2f} (has_position={b2_has_pos})\n"
        f"  capital_management.enabled is still false in parameters.yml\n"
        f"  Set enabled: true when ready to activate."
    )


if __name__ == "__main__":
    main()
