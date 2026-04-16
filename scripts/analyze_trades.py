#!/usr/bin/env python
"""Analisa trades do paper trading — MAE/MFE, win rate, expectancy."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from src.config import get_path


def _load_trades() -> pd.DataFrame:
    path = get_path("trades")
    if not path.exists():
        print("No trades found. Run paper trading first.")
        sys.exit(0)
    df = pd.read_parquet(path)
    if df.empty:
        print("trades.parquet is empty.")
        sys.exit(0)
    df["entry_time"] = pd.to_datetime(df["entry_time"], utc=True, errors="coerce")
    df["exit_time"] = pd.to_datetime(df["exit_time"], utc=True, errors="coerce")
    return df


def _pct_str(values: pd.Series, pcts: list[int]) -> str:
    return "  ".join(f"p{p}: {np.percentile(values.dropna(), p):+.2f}%" for p in pcts)


def main() -> int:
    df = _load_trades()
    n = len(df)
    winners = df[df["return_pct"] > 0]
    losers = df[df["return_pct"] <= 0]
    win_rate = len(winners) / n if n else 0

    avg_win = winners["return_pct"].mean() if len(winners) else 0.0
    avg_loss = abs(losers["return_pct"].mean()) if len(losers) else 0.0
    expectancy = win_rate * avg_win - (1 - win_rate) * avg_loss

    total_wins = winners["return_pct"].sum()
    total_losses = abs(losers["return_pct"].sum())
    profit_factor = (total_wins / total_losses) if total_losses > 0 else float("inf")

    pcts = [25, 50, 75, 90, 95]

    print(f"\n{'═' * 56}")
    print(f" Trade Analysis  ({n} trades)")
    print(f"{'═' * 56}")
    print(f"  Win Rate:      {win_rate:.0%}  ({len(winners)}/{n})")
    print(f"  Avg Win:       {avg_win:+.3f}%")
    print(f"  Avg Loss:      {-avg_loss:+.3f}%")
    print(f"  Expectancy:    {expectancy:+.4f}% per trade")
    print(f"  Profit Factor: {profit_factor:.2f}")
    print(f"  Avg Duration:  {df['duration_hours'].mean():.1f}h")
    print()

    print("  MAE Distribution (all trades):")
    print(f"    {_pct_str(df['mae_pct'], pcts)}")
    print("  MAE Distribution (winners only):")
    if len(winners):
        print(f"    {_pct_str(winners['mae_pct'], pcts)}")
    else:
        print("    (no winners yet)")
    print()

    print("  MFE Distribution (all trades):")
    print(f"    {_pct_str(df['mfe_pct'], pcts)}")
    print()

    # Optimal stop recommendations
    if len(winners) >= 5:
        sl_rec = abs(np.percentile(winners["mae_pct"].dropna(), 90))
        sg_rec = np.percentile(df["mfe_pct"].dropna(), 75)
        print("  Optimal Stops (based on MAE/MFE):")
        print(f"    SL recommendation: {sl_rec:.2f}%  (MAE p90 of winners)")
        print(f"    SG recommendation: {sg_rec:.2f}%  (MFE p75 of all trades)")

        # Compare to configured stops
        params_sl = df["stop_loss_pct"].iloc[-1] * 100 if "stop_loss_pct" in df.columns else None
        params_sg = df["stop_gain_pct"].iloc[-1] * 100 if "stop_gain_pct" in df.columns else None
        if params_sl and params_sg:
            print(f"    Configured: SL={params_sl:.2f}% SG={params_sg:.2f}%")
    print()

    # Exit reason breakdown
    if "exit_reason" in df.columns:
        print("  Exit Reasons:")
        for reason, count in df["exit_reason"].value_counts().items():
            avg_ret = df[df["exit_reason"] == reason]["return_pct"].mean()
            print(f"    {reason:<20} {count:>4}x  avg return: {avg_ret:+.3f}%")
        print()

    # Context analysis
    if "entry_score_raw" in df.columns and len(df) >= 5:
        print("  Context Analysis:")
        top_q = df.nlargest(max(1, n // 3), "return_pct")
        bot_q = df.nsmallest(max(1, n // 3), "return_pct")
        for label, subset in [("Best entries", top_q), ("Worst entries", bot_q)]:
            score = subset["entry_score_raw"].mean() if "entry_score_raw" in subset else None
            bb = subset["entry_bb_pct"].mean() if "entry_bb_pct" in subset else None
            rsi = subset["entry_rsi"].mean() if "entry_rsi" in subset else None
            parts = []
            if score is not None:
                parts.append(f"score={score:.2f}")
            if bb is not None:
                parts.append(f"BB={bb:.2f}")
            if rsi is not None:
                parts.append(f"RSI={rsi:.0f}")
            print(f"    {label}: {', '.join(parts)}")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
