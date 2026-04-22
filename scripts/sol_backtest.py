"""
SOL Bot 4 — Walk-Forward Backtest (Phase 2).

Testa estratégia baseada em evidências do Phase 1 EDA:
  H1: taker_z como gate primário (Cohen's d=0.51-0.61)
  H1: OI_z bipolar (continuation 1h, reversion 24h warning)
  H3: ETH momentum como contexto (β_ETH=0.629 > β_BTC=0.516)
  H2: volume_z auxiliar (1h/4h useful, 24h noise)

Split: 70% train / 30% test (walk-forward, sem look-ahead).

Outputs:
  conf/parameters_sol.yml   (best params do TEST set)
  prompts/sol_phase2_report.md
  prompts/tables/sol_backtest_*.csv
  prompts/plots/sol_backtest/*.png
"""

import logging
import sys
from datetime import datetime, timezone
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("sol_backtest")

OUT_DIR = ROOT / "prompts"
PLOTS_DIR = OUT_DIR / "plots" / "sol_backtest"
TABLES_DIR = OUT_DIR / "tables"
REPORT_PATH = OUT_DIR / "sol_phase2_report.md"
PARAMS_PATH = ROOT / "conf" / "parameters_sol.yml"

for d in [OUT_DIR, PLOTS_DIR, TABLES_DIR]:
    d.mkdir(parents=True, exist_ok=True)


def _df_to_md(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    header = "| " + " | ".join(str(c) for c in cols) + " |"
    sep = "|" + "|".join(["---"] * len(cols)) + "|"
    rows = ["| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join([header, sep] + rows)


# ==================================================================
# DATA LOADING + FEATURE ENGINEERING
# ==================================================================

def _load_ohlcv(asset: str) -> pd.DataFrame:
    df = pd.read_parquet(ROOT / f"data/01_raw/spot/{asset}_1h.parquet")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)


def _load_derivative(asset: str, dtype: str) -> pd.DataFrame:
    prefix = "" if asset == "btc" else f"{asset}_"
    df = pd.read_parquet(ROOT / f"data/01_raw/futures/{prefix}{dtype}_4h.parquet")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)


def _zscore(s: pd.Series, window: int) -> pd.Series:
    m = s.rolling(window, min_periods=max(10, window // 4)).mean()
    sd = s.rolling(window, min_periods=max(10, window // 4)).std()
    return (s - m) / sd.replace(0, np.nan)


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def build_features() -> pd.DataFrame:
    """
    Construção do dataset de features com anti look-ahead (shift(1)).

    Base: SOL 1h OHLCV
    Context: ETH 1h (G0, H3)
    Gates: SOL taker 4h, SOL OI 4h (ffill → 1h, shift(1) antes de usar como feature)
    """
    logger.info("Building features dataset...")

    # OHLCV
    sol = _load_ohlcv("sol")
    eth = _load_ohlcv("eth")

    # SOL technicals
    sol["rsi"] = _rsi(sol["close"])
    sol["ma21"] = sol["close"].rolling(21).mean()
    sol["ret_1d"] = sol["close"].pct_change(24)
    sol["volume_z"] = _zscore(sol["volume"], 168)

    # ETH context features (G0)
    eth["eth_ret_1h"] = eth["close"].pct_change()

    # SOL taker 4h
    taker = _load_derivative("sol", "taker")
    buy_col = next((c for c in ["taker_buy_volume_usd", "buy_volume_usd"] if c in taker.columns), None)
    sell_col = next((c for c in ["taker_sell_volume_usd", "sell_volume_usd"] if c in taker.columns), None)
    if buy_col and sell_col:
        total = taker[buy_col] + taker[sell_col]
        taker["taker_ratio"] = taker[buy_col] / total.replace(0, np.nan)
        taker["taker_z"] = _zscore(taker["taker_ratio"], 42)
    elif "buy_sell_ratio" in taker.columns:
        taker["taker_ratio"] = taker["buy_sell_ratio"]
        taker["taker_z"] = _zscore(taker["taker_ratio"], 42)

    # SOL OI 4h — z-score + 24h rolling max (bipolar H1 insight)
    oi = _load_derivative("sol", "oi")
    oi["oi_z"] = _zscore(oi["open_interest"], 42)
    # 24h max: rolling 6 candles × 4h = 24h (worst-case OI level in last 24h)
    oi["oi_z_24h_max"] = oi["oi_z"].rolling(6, min_periods=1).max()

    # Merge: SOL base
    df = sol[["timestamp", "open", "high", "low", "close", "volume",
              "rsi", "ma21", "ret_1d", "volume_z"]].copy()

    # ETH context
    df = df.merge(
        eth[["timestamp", "eth_ret_1h"]],
        on="timestamp", how="left"
    )

    # Taker 4h → ffill to 1h
    df = df.merge(taker[["timestamp", "taker_z"]], on="timestamp", how="left")
    df["taker_z"] = df["taker_z"].ffill()

    # OI 4h → ffill to 1h
    df = df.merge(oi[["timestamp", "oi_z", "oi_z_24h_max"]], on="timestamp", how="left")
    df["oi_z"] = df["oi_z"].ffill()
    df["oi_z_24h_max"] = df["oi_z_24h_max"].ffill()

    # Anti look-ahead: shift(1) em todas as features preditoras
    for col in ["volume_z", "taker_z", "oi_z", "eth_ret_1h"]:
        if col in df.columns:
            df[f"{col}_prev"] = df[col].shift(1)
    # oi_z_24h_max usada tanto em entry (shifted) quanto em exit logic (atual, durante trade)
    df["oi_z_24h_max_prev"] = df["oi_z_24h_max"].shift(1)

    # Drop warmup (RSI+MA21 precisam de ~21-28 candles)
    warmup_cols = ["rsi", "ma21", "ret_1d", "volume_z_prev", "taker_z_prev", "oi_z_prev"]
    df = df.dropna(subset=[c for c in warmup_cols if c in df.columns]).reset_index(drop=True)

    # Restrict to period where derivatives exist (OI/taker start ~Oct 24)
    has_deriv = df["taker_z_prev"].notna() & df["oi_z_prev"].notna()
    df = df[has_deriv].reset_index(drop=True)

    logger.info(f"Dataset: {len(df):,} rows  {df['timestamp'].min().date()} → {df['timestamp'].max().date()}")
    return df


# ==================================================================
# ENTRY SIGNAL
# ==================================================================

def entry_signal(row, params: dict) -> tuple[bool, str]:
    """Returns (allowed, reason)."""
    g0 = params.get("g0_eth_regime", {})
    if g0.get("enabled", True):
        eth_ret = row.get("eth_ret_1h_prev")
        if pd.notna(eth_ret) and eth_ret < g0.get("eth_ret_1h_min", 0.0):
            return False, "eth_bearish"

    g1 = params.get("g1_taker", {})
    if g1.get("enabled", True):
        tz = row.get("taker_z_prev")
        if pd.notna(tz) and tz < g1.get("taker_z_4h_min", 0.0):
            return False, "taker_negative"

    g2 = params.get("g2_oi_bipolar", {})
    if g2.get("enabled", True):
        oi1h = row.get("oi_z_prev")
        if pd.notna(oi1h) and oi1h < g2.get("oi_z_1h_min", 0.0):
            return False, "oi_1h_negative"
        oi24 = row.get("oi_z_24h_max_prev")
        if pd.notna(oi24) and oi24 > g2.get("oi_z_24h_block", 2.5):
            return False, "oi_24h_extreme"

    g3 = params.get("g3_volume_aux", {})
    if g3.get("enabled", True):
        vz = row.get("volume_z_prev")
        if pd.notna(vz) and vz < g3.get("volume_z_prev_min", -0.5):
            return False, "volume_crashed"

    flt = params.get("filters", {})
    if row.get("ret_1d", 0) < flt.get("ret_1d_min", 0.0):
        return False, "ret_1d_negative"
    rsi = row.get("rsi", 50.0)
    if rsi < flt.get("rsi_min", 50.0):
        return False, "rsi_too_low"
    if rsi > flt.get("rsi_max", 85.0):
        return False, "rsi_too_high"
    if flt.get("close_above_ma21", True):
        if pd.notna(row.get("ma21")) and row["close"] < row["ma21"]:
            return False, "below_ma21"

    return True, "ok"


# ==================================================================
# TRADE SIMULATION
# ==================================================================

def simulate_trade(df: pd.DataFrame, entry_idx: int, params: dict) -> dict:
    stops = params.get("stops", {})
    sl_pct = stops.get("stop_loss_pct", 0.015)
    tp_pct = stops.get("take_profit_pct", 0.020)
    trail_pct = stops.get("trailing_pct", 0.010)
    max_hold = stops.get("max_hold_hours", 120)

    entry_row = df.iloc[entry_idx]
    entry_price = entry_row["close"]
    entry_time = entry_row["timestamp"]

    sl_price = entry_price * (1 - sl_pct)
    tp_price = entry_price * (1 + tp_pct)
    trailing_high = entry_price
    trailing_sl = sl_price

    oi_exit = params.get("early_exits", {}).get("oi_24h_reversion", {})
    oi_exit_on = oi_exit.get("enabled", True)
    oi_exit_thr = oi_exit.get("oi_z_threshold", 2.0)
    oi_exit_min_h = oi_exit.get("min_hours_since_entry", 12)

    max_idx = min(entry_idx + max_hold, len(df) - 1)

    for i in range(entry_idx + 1, max_idx + 1):
        row = df.iloc[i]
        hours_held = i - entry_idx

        # Update trailing
        if row["high"] > trailing_high:
            trailing_high = row["high"]
            trailing_sl = max(trailing_sl, trailing_high * (1 - trail_pct))

        # TP
        if row["high"] >= tp_price:
            return dict(entry_time=entry_time, entry_price=entry_price,
                        exit_time=row["timestamp"], exit_price=tp_price,
                        return_pct=tp_pct * 100, exit_reason="TP", hours_held=hours_held)

        # Trailing/fixed SL
        if row["low"] <= trailing_sl:
            ret = (trailing_sl / entry_price - 1) * 100
            reason = "TRAIL" if trailing_sl > sl_price else "SL"
            return dict(entry_time=entry_time, entry_price=entry_price,
                        exit_time=row["timestamp"], exit_price=trailing_sl,
                        return_pct=ret, exit_reason=reason, hours_held=hours_held)

        # OI early exit (H1 bipolar: high OI_z_24h → reversal coming)
        if oi_exit_on and hours_held >= oi_exit_min_h:
            oi24 = row.get("oi_z_24h_max")
            if pd.notna(oi24) and oi24 > oi_exit_thr:
                ret = (row["close"] / entry_price - 1) * 100
                return dict(entry_time=entry_time, entry_price=entry_price,
                            exit_time=row["timestamp"], exit_price=row["close"],
                            return_pct=ret, exit_reason="OI_EXIT", hours_held=hours_held)

    # Max hold
    last = df.iloc[max_idx]
    return dict(entry_time=entry_time, entry_price=entry_price,
                exit_time=last["timestamp"], exit_price=last["close"],
                return_pct=(last["close"] / entry_price - 1) * 100,
                exit_reason="MAX_HOLD", hours_held=max_idx - entry_idx)


# ==================================================================
# BACKTEST RUNNER
# ==================================================================

def run_backtest(df: pd.DataFrame, params: dict, cooldown_h: int = 4) -> pd.DataFrame:
    trades = []
    i = 0
    last_exit_i = -cooldown_h - 1

    while i < len(df):
        if i <= last_exit_i + cooldown_h:
            i += 1
            continue
        allowed, _ = entry_signal(df.iloc[i], params)
        if allowed:
            trade = simulate_trade(df, i, params)
            trades.append(trade)
            exit_ts = trade["exit_time"]
            while i < len(df) and df.iloc[i]["timestamp"] <= exit_ts:
                i += 1
            last_exit_i = i
        else:
            i += 1

    return pd.DataFrame(trades) if trades else pd.DataFrame(
        columns=["entry_time", "entry_price", "exit_time", "exit_price",
                 "return_pct", "exit_reason", "hours_held"]
    )


# ==================================================================
# METRICS
# ==================================================================

def compute_metrics(trades: pd.DataFrame) -> dict:
    if len(trades) == 0:
        return dict(n_trades=0, sharpe=0.0, win_rate=0.0,
                    avg_return=0.0, total_return=0.0, max_dd=0.0, profit_factor=0.0)

    rets = trades["return_pct"].values / 100
    wins = rets > 0
    std = rets.std()
    sharpe = (rets.mean() / std) * np.sqrt(52) if std > 0 else 0.0

    cum = np.cumprod(1 + rets)
    peak = np.maximum.accumulate(cum)
    max_dd = ((cum - peak) / peak).min() * 100

    gross_win = rets[wins].sum()
    gross_loss = abs(rets[~wins].sum())
    pf = gross_win / gross_loss if gross_loss > 0 else (gross_win if gross_win > 0 else 0.0)

    return dict(
        n_trades=len(trades),
        sharpe=sharpe,
        win_rate=wins.mean() * 100,
        avg_return=rets.mean() * 100,
        total_return=(cum[-1] - 1) * 100,
        max_dd=max_dd,
        profit_factor=pf,
    )


# ==================================================================
# GRID SEARCH (train only)
# ==================================================================

def default_params() -> dict:
    return {
        "g0_eth_regime": {"enabled": True, "eth_ret_1h_min": 0.0},
        "g1_taker": {"enabled": True, "taker_z_4h_min": 0.0},
        "g2_oi_bipolar": {"enabled": True, "oi_z_1h_min": 0.0,
                          "oi_z_24h_warning": 2.0, "oi_z_24h_block": 2.5},
        "g3_volume_aux": {"enabled": True, "volume_z_prev_min": -0.5},
        "filters": {"ret_1d_min": 0.0, "rsi_min": 50.0, "rsi_max": 85.0,
                    "close_above_ma21": True},
        "stops": {"stop_loss_pct": 0.015, "take_profit_pct": 0.020,
                  "trailing_pct": 0.010, "max_hold_hours": 120},
        "early_exits": {"oi_24h_reversion": {"enabled": True,
                                              "oi_z_threshold": 2.0,
                                              "min_hours_since_entry": 12}},
    }


def _set_param(params: dict, dotkey: str, value) -> dict:
    import copy
    p = copy.deepcopy(params)
    section, key = dotkey.split(".", 1)
    p[section][key] = value
    return p


def grid_search(df_train: pd.DataFrame) -> pd.DataFrame:
    grid = {
        "g1_taker.taker_z_4h_min": [-0.5, 0.0, 0.3],
        "g2_oi_bipolar.oi_z_1h_min": [-0.5, 0.0, 0.3],
        "g2_oi_bipolar.oi_z_24h_block": [2.0, 2.5, 3.0],
        "filters.rsi_min": [50, 55, 60],
        "filters.rsi_max": [80, 85],
    }
    keys = list(grid.keys())
    combos = list(product(*[grid[k] for k in keys]))
    logger.info(f"Grid search: {len(combos)} combos on TRAIN set...")

    rows = []
    for i, combo in enumerate(combos):
        params = default_params()
        for k, v in zip(keys, combo):
            params = _set_param(params, k, v)
        trades = run_backtest(df_train, params)
        m = compute_metrics(trades)
        row = {k: v for k, v in zip(keys, combo)}
        row.update(m)
        rows.append(row)
        if (i + 1) % 50 == 0:
            logger.info(f"  {i+1}/{len(combos)} done...")

    return pd.DataFrame(rows).sort_values("sharpe", ascending=False)


# ==================================================================
# PLOTS
# ==================================================================

def generate_plots(df_train: pd.DataFrame, df_test: pd.DataFrame,
                   trades_test: pd.DataFrame, grid: pd.DataFrame,
                   test_results: pd.DataFrame):
    logger.info("Generating plots...")

    # Plot 1: Equity curve (test set)
    if len(trades_test) > 0:
        fig, axes = plt.subplots(2, 1, figsize=(14, 8))
        rets = trades_test["return_pct"].values / 100
        cum = np.cumprod(1 + rets)
        axes[0].plot(range(len(cum)), cum * 10000, color="green", linewidth=1.5)
        axes[0].axhline(10000, color="gray", linestyle="--", linewidth=0.5)
        axes[0].set_title(f"SOL Bot 4 — Equity Curve (TEST set, {len(trades_test)} trades)")
        axes[0].set_ylabel("Portfolio ($)")
        axes[0].grid(alpha=0.3)

        colors = ["green" if r > 0 else "red" for r in rets]
        axes[1].bar(range(len(rets)), rets * 100, color=colors, alpha=0.7)
        axes[1].axhline(0, color="black", linewidth=0.5)
        axes[1].set_title("Return per Trade (%)")
        axes[1].set_ylabel("Return (%)")
        axes[1].grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "equity_curve_test.png", dpi=100, bbox_inches="tight")
        plt.close()

    # Plot 2: Train Sharpe vs Test Sharpe (top 10)
    if len(test_results) > 0:
        fig, ax = plt.subplots(figsize=(10, 5))
        x = range(len(test_results))
        ax.scatter(test_results["train_sharpe"], test_results["test_sharpe"],
                   s=80, alpha=0.8, c=test_results["test_sharpe"],
                   cmap="RdYlGn", vmin=0, vmax=3)
        ax.plot([0, 4], [0, 4], "k--", linewidth=0.5, label="y=x (no overfitting)")
        ax.set_xlabel("Train Sharpe")
        ax.set_ylabel("Test Sharpe")
        ax.set_title("Train vs Test Sharpe (top 10 configs)")
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "train_vs_test_sharpe.png", dpi=100, bbox_inches="tight")
        plt.close()

    # Plot 3: Grid search distribution
    if len(grid) > 0:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(grid["sharpe"].dropna(), bins=30, color="#2196F3", alpha=0.7, edgecolor="black")
        if len(test_results) > 0:
            best_test = test_results["test_sharpe"].max()
            ax.axvline(best_test, color="green", linestyle="--",
                       label=f"Best TEST Sharpe={best_test:.2f}")
        ax.set_title("Grid Search: Sharpe Distribution (TRAIN)")
        ax.set_xlabel("Sharpe")
        ax.set_ylabel("N configs")
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "grid_sharpe_distribution.png", dpi=100, bbox_inches="tight")
        plt.close()

    logger.info(f"Plots saved to {PLOTS_DIR}")


# ==================================================================
# REPORT + YAML SAVE
# ==================================================================

def _native(obj):
    """Recursively convert numpy scalars → Python native types for YAML."""
    if isinstance(obj, dict):
        return {k: _native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_native(v) for v in obj]
    if hasattr(obj, "item"):  # numpy scalar
        return obj.item()
    return obj


def save_params_yaml(params: dict):
    final = {
        "sol_bot": {
            "enabled": True,
            "capital_usd": 10000,
            "gates": {
                "g0_eth_regime": params["g0_eth_regime"],
                "g1_taker": params["g1_taker"],
                "g2_oi_bipolar": params["g2_oi_bipolar"],
                "g3_volume_aux": params["g3_volume_aux"],
            },
            "filters": params["filters"],
            "stops": params["stops"],
            "early_exits": params["early_exits"],
        }
    }
    with open(PARAMS_PATH, "w") as f:
        yaml.dump(_native(final), f, default_flow_style=False, sort_keys=False)
    logger.info(f"parameters_sol.yml saved: {PARAMS_PATH}")


def generate_report(df: pd.DataFrame, best_test: dict, best_params: dict,
                    grid: pd.DataFrame, test_results: pd.DataFrame,
                    trades_test: pd.DataFrame, train_days: int, test_days: int):
    test_sharpe = best_test["sharpe"]
    overfitting = best_test.get("train_sharpe", 0) - test_sharpe

    if test_sharpe >= 2.0 and best_test["n_trades"] >= 5 and overfitting < 1.0:
        decision = "✅ GO — Implementar sol_bot.py (Phase 3)"
        decision_body = "Sharpe ≥ 2.0, N ≥ 5, overfitting < 1.0."
    elif test_sharpe >= 1.0 and best_test["n_trades"] >= 5:
        decision = "⚖️ PAPER ONLY — Monitorar sem live capital"
        decision_body = "Sharpe 1.0-2.0: edge existe mas insuficiente para live."
    elif best_test["n_trades"] < 5:
        decision = "⚠️ INSUFFICIENT DATA — Poucos trades no TEST"
        decision_body = f"N={best_test['n_trades']} trades no TEST set. Coletar mais dados."
    elif overfitting >= 1.0:
        decision = "⚠️ OVERFITTING — Train/Test gap > 1.0"
        decision_body = f"Overfitting={overfitting:.2f}. Params instáveis."
    else:
        decision = "❌ NO-GO — Sharpe insuficiente"
        decision_body = "Sharpe < 1.0. Arquivar e revisitar em 3 meses."

    lines = []
    lines.append("# SOL Bot Phase 2 — Walk-Forward Backtest")
    lines.append(f"\n**Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    lines.append(f"**Dataset:** {len(df):,} rows  |  Train: {train_days}d  |  Test: {test_days}d\n")

    lines.append("## Estratégia testada\n")
    lines.append("- **G0:** ETH momentum filter (H3: β_ETH > β_BTC)")
    lines.append("- **G1:** taker_z_prev > thr (H1: Cohen's d=0.51-0.61)")
    lines.append("- **G2:** OI_z bipolar — continuation 1h, reversion block 24h (H1)")
    lines.append("- **G3:** volume_z auxiliar (H2)")
    lines.append("- **Exit:** TP 2% / SL 1.5% / Trail 1% / OI_early_exit / max 120h")
    lines.append("")

    lines.append(f"## {decision}\n")
    lines.append(decision_body)
    lines.append("")

    lines.append("## Best Config — TEST set (out-of-sample)\n")
    lines.append(f"| Métrica | Valor |")
    lines.append("|---------|-------|")
    lines.append(f"| **Test Sharpe** | **{test_sharpe:.3f}** |")
    lines.append(f"| Train Sharpe | {best_test.get('train_sharpe', '-'):.3f} |")
    lines.append(f"| Overfitting (Δ) | {overfitting:.3f} {'⚠️' if overfitting > 1.0 else '✅'} |")
    lines.append(f"| N trades (test) | {best_test['n_trades']} |")
    lines.append(f"| Win Rate | {best_test['win_rate']:.1f}% |")
    lines.append(f"| Avg Return | {best_test['avg_return']:+.3f}% |")
    lines.append(f"| Total Return | {best_test['total_return']:+.2f}% |")
    lines.append(f"| Max DD | {best_test['max_dd']:.1f}% |")
    lines.append(f"| Profit Factor | {best_test['profit_factor']:.2f} |")
    lines.append("")

    if len(trades_test) > 0:
        exit_counts = trades_test["exit_reason"].value_counts()
        lines.append("### Exit reasons (TEST)\n")
        lines.append("| Reason | N |")
        lines.append("|--------|---|")
        for reason, n in exit_counts.items():
            lines.append(f"| {reason} | {n} |")
        lines.append("")

    lines.append("## Best Parameters\n")
    lines.append("```yaml")
    lines.append(yaml.dump(_native(best_params), default_flow_style=False).rstrip())
    lines.append("```\n")

    lines.append("## Grid Search — Top 10 TRAIN\n")
    gcols = [c for c in grid.columns if "." in c] + ["sharpe", "n_trades", "win_rate", "total_return"]
    gcols = [c for c in gcols if c in grid.columns]
    top10 = grid.head(10)[gcols].copy()
    top10 = top10.round(4)
    lines.append(_df_to_md(top10))
    lines.append("")

    lines.append("## TEST Results — Top 10 (out-of-sample)\n")
    tcols = ["train_sharpe", "test_sharpe", "test_n_trades", "test_wr",
             "test_total_ret", "overfitting"]
    if len(test_results) > 0:
        tr_display = test_results[[c for c in tcols if c in test_results.columns]].round(3)
        lines.append(_df_to_md(tr_display))
    lines.append("")

    lines.append("## Arquivos\n")
    lines.append(f"- `conf/parameters_sol.yml` — best params")
    lines.append(f"- `prompts/tables/sol_backtest_grid.csv`")
    lines.append(f"- `prompts/tables/sol_backtest_test_results.csv`")
    lines.append(f"- `prompts/plots/sol_backtest/`")

    with open(REPORT_PATH, "w") as f:
        f.write("\n".join(lines))
    logger.info(f"Report: {REPORT_PATH}")


# ==================================================================
# MAIN
# ==================================================================

def main():
    logger.info("=" * 60)
    logger.info("SOL BOT 4 — PHASE 2 WALK-FORWARD BACKTEST")
    logger.info("=" * 60)

    df = build_features()

    # 70/30 split
    split_idx = int(len(df) * 0.70)
    df_train = df.iloc[:split_idx].reset_index(drop=True)
    df_test = df.iloc[split_idx:].reset_index(drop=True)
    train_days = (df_train["timestamp"].max() - df_train["timestamp"].min()).days
    test_days = (df_test["timestamp"].max() - df_test["timestamp"].min()).days
    logger.info(f"TRAIN: {len(df_train):,} rows  {df_train['timestamp'].min().date()} → {df_train['timestamp'].max().date()} ({train_days}d)")
    logger.info(f"TEST:  {len(df_test):,} rows  {df_test['timestamp'].min().date()} → {df_test['timestamp'].max().date()} ({test_days}d)")

    # Grid search on TRAIN
    grid = grid_search(df_train)
    grid.to_csv(TABLES_DIR / "sol_backtest_grid.csv", index=False)

    top10 = grid.head(10)
    logger.info(f"\nTop 10 TRAIN configs (Sharpe):")
    print(top10[["sharpe", "n_trades", "win_rate", "total_return"]].to_string(index=False))

    # Test top 10 on TEST set
    import copy
    test_rows = []
    best_test_trades = pd.DataFrame()
    best_test_params = default_params()

    for _, row in top10.iterrows():
        params = default_params()
        for col in row.index:
            if "." in col:
                params = _set_param(params, col, row[col])

        trades_t = run_backtest(df_test, params)
        m = compute_metrics(trades_t)
        test_rows.append({
            "train_sharpe": round(row["sharpe"], 3),
            "test_sharpe": round(m["sharpe"], 3),
            "test_n_trades": m["n_trades"],
            "test_wr": round(m["win_rate"], 1),
            "test_total_ret": round(m["total_return"], 2),
            "overfitting": round(row["sharpe"] - m["sharpe"], 3),
            **{c: row[c] for c in row.index if "." in c},
        })

    test_results = pd.DataFrame(test_rows).sort_values("test_sharpe", ascending=False)
    test_results.to_csv(TABLES_DIR / "sol_backtest_test_results.csv", index=False)

    # Best test config
    best_row = test_results.iloc[0]
    best_params_used = default_params()
    for col in best_row.index:
        if "." in col:
            best_params_used = _set_param(best_params_used, col, best_row[col])

    best_test_trades = run_backtest(df_test, best_params_used)
    best_metrics = compute_metrics(best_test_trades)
    best_metrics["train_sharpe"] = best_row["train_sharpe"]

    logger.info(f"\n=== BEST CONFIG ===")
    logger.info(f"Train Sharpe: {best_row['train_sharpe']:.3f}")
    logger.info(f"Test  Sharpe: {best_metrics['sharpe']:.3f}")
    logger.info(f"Overfitting:  {best_row['overfitting']:.3f}")
    logger.info(f"Test N trades: {best_metrics['n_trades']}")
    logger.info(f"Test WR:       {best_metrics['win_rate']:.1f}%")
    logger.info(f"Test Total:    {best_metrics['total_return']:+.2f}%")

    # Decision
    test_sharpe = best_metrics["sharpe"]
    overfitting = best_row["overfitting"]
    if test_sharpe >= 2.0 and best_metrics["n_trades"] >= 5 and overfitting < 1.0:
        decision = "✅ GO"
    elif test_sharpe >= 1.0 and best_metrics["n_trades"] >= 5:
        decision = "⚖️ PAPER ONLY"
    else:
        decision = "❌ NO-GO"

    # Save yaml + plots + report
    save_params_yaml(best_params_used)
    generate_plots(df_train, df_test, best_test_trades, grid, test_results)
    generate_report(df, best_metrics, best_params_used, grid, test_results,
                    best_test_trades, train_days, test_days)

    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    print(f"\nDecision: {decision}")
    print(f"Test Sharpe: {test_sharpe:.3f}  |  Overfitting: {overfitting:.3f}  |  N trades: {best_metrics['n_trades']}")
    print(f"\nReport: {REPORT_PATH}")
    print(f"Params: {PARAMS_PATH}")

    print("\nTest results (top 10):")
    print(test_results[["train_sharpe", "test_sharpe", "test_n_trades",
                          "test_wr", "overfitting"]].to_string(index=False))


if __name__ == "__main__":
    main()
