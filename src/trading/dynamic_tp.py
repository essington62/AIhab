"""
src/trading/dynamic_tp.py

Dynamic TP v2 — 3-rule system for Bot 2 BTC.

Status: applied 22/04/2026 without prior backtest study.
User accepted risk of deviation from baseline (Fixed 2% Sharpe 2.71).

History:
- v1 (20/04): rejected. Complex buckets. Sharpe -0.15.
- v2 (22/04): simplified. Default = baseline. Only modifies edge cases.
"""

import logging

logger = logging.getLogger(__name__)


def get_dynamic_tp(rsi: float | None, bb_pct: float | None,
                   volume_z: float | None) -> tuple[float, str]:
    """
    Calculate dynamic TP for Bot 2 based on entry context.

    Rules (priority order):
      1. volume_z > 1.0  → TP 1.0%  (volume exhaustion signal)
      2. RSI > 75 AND bb_pct > 0.95 → TP 1.5% (overbought context)
      3. default → TP 2.0% (baseline)

    Args:
        rsi:      RSI at entry (0–100)
        bb_pct:   Bollinger Band %B (0–1)
        volume_z: Volume z-score rolling 7d (168h)

    Returns:
        (tp_pct, reason): TP as fraction (0.02 = 2%), reason label
    """
    if volume_z is not None and volume_z > 1.0:
        return 0.010, "volume_exhaustion"
    if rsi is not None and bb_pct is not None and rsi > 75 and bb_pct > 0.95:
        return 0.015, "overbought"
    return 0.020, "default"


def log_tp_decision(entry_price: float, tp_pct: float, reason: str,
                    rsi: float | None, bb_pct: float | None,
                    volume_z: float | None) -> None:
    rsi_s    = f"{rsi:.1f}"    if rsi    is not None else "N/A"
    bb_s     = f"{bb_pct:.3f}" if bb_pct is not None else "N/A"
    volz_s   = f"{volume_z:+.2f}" if volume_z is not None else "N/A"
    tp_price = entry_price * (1 + tp_pct)
    logger.info(
        f"BOT2 Dynamic TP: {tp_pct*100:.1f}% ({reason}) | "
        f"TP ${tp_price:,.2f} | RSI={rsi_s} BB={bb_s} VolZ={volz_s}"
    )
