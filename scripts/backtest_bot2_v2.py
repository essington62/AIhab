#!/usr/bin/env python3
"""
Backtest comparativo Bot 2 original vs Bot 2 v2 (Early Reversal).

Métricas estruturais: entry quality, win rate, profit factor, drawdown,
false signal rate, entry mode breakdown.

Usage:
    python scripts/backtest_bot2_v2.py
"""
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("backtest")

OUT_DIR = ROOT / "prompts"
PLOTS_DIR = OUT_DIR / "plots"
TABLES_DIR = OUT_DIR / "tables"
REPORT_PATH = OUT_DIR / "bot2_v2_backtest_report.md"

for d in [PLOTS_DIR, TABLES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

SL_PCT = 0.015
TP_PCT = 0.020
TRAIL_PCT = 0.010
MAX_HOLD_HOURS = 120


def load_data() -> pd.DataFrame:
    spot = pd.read_parquet(ROOT / "data/02_intermediate/spot/btc_1h_clean.parquet")
    spot["timestamp"] = pd.to_datetime(spot["timestamp"], utc=True)
    spot = spot.sort_values("timestamp").reset_index(drop=True)

    # Forward returns (24h rolling, shifted for signal timing)
    spot["ret_1d"] = spot["close"].pct_change(24)
    spot["ret_1d_1h_ago"] = spot["ret_1d"].shift(1)
    spot["ret_1d_3h_ago"] = spot["ret_1d"].shift(3)
    spot["low_24h"] = spot["low"].rolling(24, min_periods=1).min()

    zs = pd.read_parquet(ROOT / "data/02_features/gate_zscores.parquet")
    zs["timestamp"] = pd.to_datetime(zs["timestamp"], utc=True)

    df = spot.merge(zs[["timestamp", "stablecoin_z"]], on="timestamp", how="left")
    df["stablecoin_z"] = df["stablecoin_z"].ffill()

    logger.info(
        f"Loaded {len(df)} rows: "
        f"{df['timestamp'].min().date()} → {df['timestamp'].max().date()}"
    )
    return df


def _eval_baseline(row) -> tuple[bool, str | None]:
    """Bot 2 original — all conditions must pass."""
    if any(pd.isna(row.get(c)) for c in ["stablecoin_z", "ret_1d", "rsi_14", "bb_pct", "close", "ma_21"]):
        return False, None
    # spike guard
    if row["ret_1d"] > 0.03 and row["rsi_14"] > 65:
        return False, None
    if (
        row["stablecoin_z"] > 1.3
        and row["ret_1d"] > 0
        and row["rsi_14"] > 50
        and row["close"] > row["ma_21"]
        and row["bb_pct"] < 0.98
    ):
        return True, "classic"
    return False, None


def _eval_v2(row) -> tuple[bool, str | None]:
    """Bot 2 v2 — classic OR early_reversal."""
    if any(pd.isna(row.get(c)) for c in ["stablecoin_z", "ret_1d", "rsi_14", "bb_pct", "close", "ma_21"]):
        return False, None

    if row["stablecoin_z"] <= 1.3 or row["bb_pct"] >= 0.98:
        return False, None

    # Classic
    if row["ret_1d"] > 0 and row["rsi_14"] > 50 and row["close"] > row["ma_21"]:
        return True, "classic"

    # Early reversal
    r1h = row.get("ret_1d_1h_ago")
    r3h = row.get("ret_1d_3h_ago")
    if pd.isna(r1h) or pd.isna(r3h):
        return False, None

    trend_improving = (row["ret_1d"] > r1h) and (r1h > r3h)
    delta_3h = row["ret_1d"] - r3h
    if (
        row["ret_1d"] > -0.015
        and trend_improving
        and delta_3h > 0.005
        and row["rsi_14"] > 35
    ):
        return True, "early"
    return False, None


def simulate(df: pd.DataFrame, evaluator, label: str) -> pd.DataFrame:
    """Sequential simulation — one position at a time."""
    trades = []
    position = None

    for i in range(len(df)):
        row = df.iloc[i]

        if position is None:
            passed, mode = evaluator(row)
            if passed:
                low_24h = row["low_24h"] if not pd.isna(row["low_24h"]) else row["close"]
                entry_delay = (row["close"] / low_24h - 1) if low_24h > 0 else 0
                position = {
                    "entry_idx": i,
                    "entry_price": row["close"],
                    "entry_ts": row["timestamp"],
                    "entry_mode": mode,
                    "entry_rsi": row["rsi_14"],
                    "entry_bb": row["bb_pct"],
                    "entry_ret_1d": row["ret_1d"],
                    "entry_sz": row["stablecoin_z"],
                    "entry_delay_pct": entry_delay,
                    "trailing_high": row["close"],
                    "sl": row["close"] * (1 - SL_PCT),
                    "tp": row["close"] * (1 + TP_PCT),
                }
        else:
            # Update trailing
            if row["high"] > position["trailing_high"]:
                position["trailing_high"] = row["high"]

            trail_stop = position["trailing_high"] * (1 - TRAIL_PCT)
            hold_h = i - position["entry_idx"]

            exit_reason = exit_price = None
            if row["low"] <= position["sl"]:
                exit_reason, exit_price = "SL", position["sl"]
            elif row["high"] >= position["tp"]:
                exit_reason, exit_price = "TP", position["tp"]
            elif row["high"] > position["entry_price"] and row["low"] <= trail_stop:
                exit_reason = "TRAIL"
                exit_price = max(trail_stop, position["entry_price"])
            elif hold_h >= MAX_HOLD_HOURS:
                exit_reason, exit_price = "TIMEOUT", row["close"]

            if exit_reason:
                pnl_pct = exit_price / position["entry_price"] - 1
                trades.append({
                    **position,
                    "exit_ts": row["timestamp"],
                    "exit_price": exit_price,
                    "exit_reason": exit_reason,
                    "hold_hours": hold_h,
                    "pnl_pct": pnl_pct,
                    "fast_sl": exit_reason == "SL" and hold_h < 6,
                })
                position = None

    out = pd.DataFrame(trades)
    out["label"] = label
    return out


def metrics(trades: pd.DataFrame, label: str) -> dict:
    if trades.empty:
        return {"label": label, "n_trades": 0}

    wins = trades[trades["pnl_pct"] > 0]
    losses = trades[trades["pnl_pct"] <= 0]

    wr = len(wins) / len(trades)
    avg_r = float(trades["pnl_pct"].mean())
    total_r = float((1 + trades["pnl_pct"]).prod() - 1)
    std_r = float(trades["pnl_pct"].std())
    sharpe = (avg_r / std_r * np.sqrt(len(trades))) if std_r > 0 else 0

    sum_w = float(wins["pnl_pct"].sum()) if len(wins) else 0
    sum_l = float(abs(losses["pnl_pct"].sum())) if len(losses) else 0
    pf = sum_w / sum_l if sum_l > 0 else float("inf")

    equity = (1 + trades["pnl_pct"]).cumprod()
    max_dd = float((equity / equity.cummax() - 1).min())

    fsr = float(trades["fast_sl"].sum() / len(trades))
    avg_delay = float(trades["entry_delay_pct"].mean())
    avg_hold = float(trades["hold_hours"].mean())

    return {
        "label": label,
        "n_trades": len(trades),
        "winrate": wr,
        "avg_return": avg_r,
        "total_return": total_r,
        "sharpe": sharpe,
        "profit_factor": pf,
        "max_drawdown": max_dd,
        "avg_entry_delay_pct": avg_delay,
        "false_signal_rate": fsr,
        "avg_hold_hours": avg_hold,
    }


def generate_report(m_bl, m_v2, entry_modes, early_adv, df, trades_bl, trades_v2):
    def pct(v):
        return f"{v*100:.2f}%" if isinstance(v, float) else str(v)

    def diff(a, b):
        if isinstance(a, float) and isinstance(b, float):
            return f"{(b-a)*100:+.2f}pp"
        return ""

    lines = [
        "# 🔬 Bot 2 v2 — Structural Backtest Report",
        "",
        f"**Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        f"**Period:** {df['timestamp'].min().date()} → {df['timestamp'].max().date()}",
        f"**Rows analyzed:** {len(df)} (1h candles)",
        "",
        "## 🎯 Critério de Aceite",
        "",
        "| Critério | Threshold | Resultado |",
        "|----------|-----------|-----------|",
    ]

    wr_drop = m_bl.get("winrate", 0) - m_v2.get("winrate", 0)
    dd_ratio = (
        abs(m_v2.get("max_drawdown", 0)) / abs(m_bl.get("max_drawdown", -0.001))
        if m_bl.get("max_drawdown", 0) != 0 else 1.0
    )
    fsr_v2 = m_v2.get("false_signal_rate", 0)
    ea_ok = early_adv is not None and early_adv > 0.003
    wr_ok = wr_drop < 0.05
    dd_ok = dd_ratio <= 1.2
    fsr_ok = fsr_v2 < 0.30

    lines += [
        f"| early_advantage > 0.3% | 0.3% | {early_adv*100:.2f}% {'✅' if ea_ok else '❌'} |",
        f"| winrate_drop < 5pp | 5pp | {wr_drop*100:.2f}pp {'✅' if wr_ok else '❌'} |",
        f"| dd_ratio ≤ 1.2x | 1.2x | {dd_ratio:.2f}x {'✅' if dd_ok else '❌'} |",
        f"| false_signal_rate < 30% | 30% | {fsr_v2*100:.1f}% {'✅' if fsr_ok else '❌'} |",
        "",
    ]

    verdict = "✅ ACEITAR V2" if all([ea_ok, wr_ok, dd_ok, fsr_ok]) else "❌ REJEITAR V2"
    lines += [f"### Veredicto: {verdict}", ""]

    # Global comparison
    lines += [
        "## 📊 Comparação Global",
        "",
        "| Métrica | Baseline | V2 | Δ |",
        "|---------|----------|-----|---|",
    ]
    for k, fmt in [
        ("n_trades", "int"),
        ("winrate", "pct"),
        ("avg_return", "pct"),
        ("total_return", "pct"),
        ("sharpe", "float"),
        ("profit_factor", "float"),
        ("max_drawdown", "pct"),
        ("avg_entry_delay_pct", "pct"),
        ("false_signal_rate", "pct"),
        ("avg_hold_hours", "float"),
    ]:
        vb = m_bl.get(k, 0)
        vv = m_v2.get(k, 0)
        if fmt == "int":
            lines.append(f"| {k} | {vb} | {vv} | {vv-vb:+d} |")
        elif fmt == "pct":
            lines.append(f"| {k} | {vb*100:.2f}% | {vv*100:.2f}% | {(vv-vb)*100:+.2f}pp |")
        else:
            lines.append(f"| {k} | {vb:.3f} | {vv:.3f} | {(vv-vb):+.3f} |")

    lines.append(f"\n**Early Advantage:** `{early_adv*100:.2f}%`\n" if early_adv else "")

    # Entry mode breakdown
    if entry_modes:
        lines += [
            "## 🎯 V2 por Entry Mode",
            "",
            "| Mode | N | WR | AvgRet | PF | MaxDD | FalseSig | AvgHold |",
            "|------|---|-----|--------|-----|-------|----------|---------|",
        ]
        for mode, m in entry_modes.items():
            lines.append(
                f"| {mode} | {m.get('n_trades',0)} "
                f"| {m.get('winrate',0)*100:.1f}% "
                f"| {m.get('avg_return',0)*100:.2f}% "
                f"| {m.get('profit_factor',0):.2f} "
                f"| {m.get('max_drawdown',0)*100:.2f}% "
                f"| {m.get('false_signal_rate',0)*100:.1f}% "
                f"| {m.get('avg_hold_hours',0):.0f}h |"
            )
        lines += [
            "",
            "**Interpretação:**",
            "- `classic`: deve ser idêntico ao baseline (mesma lógica)",
            "- `early`: early entries trazem edge ou destroem risco?",
        ]

    # Exit reason breakdown
    lines += ["", "## 📤 Distribuição de Saídas", ""]
    for trades, label in [(trades_bl, "Baseline"), (trades_v2, "V2")]:
        if not trades.empty:
            dist = trades["exit_reason"].value_counts()
            lines.append(f"**{label}:** " + ", ".join(f"{k}={v}" for k, v in dist.items()))

    lines += [
        "",
        "## 📊 Plots",
        "",
        "![Equity Curves](plots/bot2_equity_curves.png)",
        "![Entry Delay](plots/bot2_entry_delay.png)",
    ]

    REPORT_PATH.write_text("\n".join(lines))
    logger.info(f"Report: {REPORT_PATH}")


def plot_equity_curves(trades_bl: pd.DataFrame, trades_v2: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, trades, label, color in [
        (axes[0], trades_bl, "Baseline", "#2196F3"),
        (axes[0], trades_v2, "V2 (Early Reversal)", "#4CAF50"),
    ]:
        if trades.empty:
            continue
        t = trades.sort_values("entry_ts").reset_index(drop=True)
        equity = (1 + t["pnl_pct"]).cumprod()
        ax.plot(range(len(equity)), equity, label=label, lw=2, color=color)

    axes[0].axhline(1.0, color="gray", ls="--", alpha=0.5, lw=0.8)
    axes[0].set_title("Equity Curves")
    axes[0].set_xlabel("Trade #")
    axes[0].set_ylabel("Equity (1.0 = start)")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Entry delay distribution
    data = []
    labels_d = []
    for trades, label in [(trades_bl, "Baseline"), (trades_v2, "V2 all")]:
        if not trades.empty:
            data.append(trades["entry_delay_pct"] * 100)
            labels_d.append(label)
    if "entry_mode" in trades_v2.columns:
        early = trades_v2[trades_v2["entry_mode"] == "early"]
        if not early.empty:
            data.append(early["entry_delay_pct"] * 100)
            labels_d.append("V2 early only")

    if data:
        axes[1].boxplot(data, labels=labels_d, patch_artist=True)
        axes[1].set_title("Entry Delay vs 24h Low (%)")
        axes[1].set_ylabel("% above 24h low")
        axes[1].grid(alpha=0.3)

    plt.tight_layout()
    path = PLOTS_DIR / "bot2_equity_curves.png"
    plt.savefig(path, dpi=110, bbox_inches="tight")
    plt.close()
    logger.info(f"Plot: {path.name}")


def plot_entry_delay(trades_bl: pd.DataFrame, trades_v2: pd.DataFrame):
    if trades_v2.empty or "entry_mode" not in trades_v2.columns:
        return
    fig, ax = plt.subplots(figsize=(10, 4))
    for trades, label, color in [
        (trades_bl, "Baseline", "#2196F3"),
        (trades_v2[trades_v2["entry_mode"] == "classic"], "V2 classic", "#4CAF50"),
        (trades_v2[trades_v2["entry_mode"] == "early"], "V2 early", "#FF9800"),
    ]:
        if trades.empty:
            continue
        ax.scatter(
            range(len(trades)), trades["entry_delay_pct"] * 100,
            label=label, s=20, alpha=0.6, color=color,
        )
    ax.axhline(0, color="gray", ls="--", alpha=0.5, lw=0.8)
    ax.set_title("Entry Delay per Trade")
    ax.set_ylabel("% above 24h low")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = PLOTS_DIR / "bot2_entry_delay.png"
    plt.savefig(path, dpi=110, bbox_inches="tight")
    plt.close()
    logger.info(f"Plot: {path.name}")


def main():
    logger.info("=" * 60)
    logger.info("Bot 2 v2 Early Reversal — Structural Backtest")
    logger.info("=" * 60)

    df = load_data()

    logger.info("\n── Simulating BASELINE ──")
    trades_bl = simulate(df, _eval_baseline, "baseline")
    logger.info(f"Baseline: {len(trades_bl)} trades")

    logger.info("\n── Simulating V2 ──")
    trades_v2 = simulate(df, _eval_v2, "v2")
    logger.info(f"V2: {len(trades_v2)} trades")

    trades_bl.to_csv(TABLES_DIR / "bot2_trades_baseline.csv", index=False)
    trades_v2.to_csv(TABLES_DIR / "bot2_trades_v2.csv", index=False)

    m_bl = metrics(trades_bl, "baseline")
    m_v2 = metrics(trades_v2, "v2")

    # By entry_mode
    entry_mode_metrics = {}
    if "entry_mode" in trades_v2.columns:
        for mode in ["classic", "early"]:
            t = trades_v2[trades_v2["entry_mode"] == mode]
            if not t.empty:
                entry_mode_metrics[mode] = metrics(t, f"v2_{mode}")

    # Early advantage: avg entry price comparison
    early_adv = None
    if not trades_bl.empty and not trades_v2.empty:
        avg_bl = float(trades_bl["entry_price"].mean())
        avg_v2 = float(trades_v2["entry_price"].mean())
        early_adv = (avg_bl - avg_v2) / avg_bl

    generate_report(m_bl, m_v2, entry_mode_metrics, early_adv, df, trades_bl, trades_v2)
    plot_equity_curves(trades_bl, trades_v2)
    plot_entry_delay(trades_bl, trades_v2)

    # Terminal summary
    print("\n" + "=" * 60)
    print("RESULTADO FINAL — Bot 2 v2 Structural Backtest")
    print("=" * 60)
    for label, m in [("BASELINE", m_bl), ("V2", m_v2)]:
        print(f"\n{label}:")
        print(f"  Trades: {m.get('n_trades',0)} | WR: {m.get('winrate',0)*100:.1f}% | "
              f"PF: {m.get('profit_factor',0):.2f} | Sharpe: {m.get('sharpe',0):.2f}")
        print(f"  AvgRet: {m.get('avg_return',0)*100:.2f}% | TotalRet: {m.get('total_return',0)*100:.2f}%")
        print(f"  MaxDD: {m.get('max_drawdown',0)*100:.2f}% | "
              f"FalseSig: {m.get('false_signal_rate',0)*100:.1f}% | "
              f"EntryDelay: {m.get('avg_entry_delay_pct',0)*100:.2f}%")

    print(f"\nEarly Advantage: {early_adv*100:.2f}%" if early_adv else "\nEarly Advantage: N/A")

    if entry_mode_metrics:
        print("\nV2 por entry_mode:")
        for mode, m in entry_mode_metrics.items():
            print(f"  [{mode}] n={m.get('n_trades',0)} WR={m.get('winrate',0)*100:.1f}% "
                  f"PF={m.get('profit_factor',0):.2f} FalseSig={m.get('false_signal_rate',0)*100:.1f}%")

    print(f"\nReport: {REPORT_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()
