"""
SOL Bot 4 v2 "Sweet Spot" — Backtest Rigoroso
Trigger: filter study revelou Sharpe ~0.08 no histórico completo vs 2.03 reportado.

Hipótese v2: manter hard gates v1 + dois filtros conservadores:
  1.03 <= close/MA21 <= 1.05   (sweet spot de trend)
  0.0  <= volume_z   <= 0.5    (volume saudável, não extremo)
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DATA_SOL   = ROOT / "data/01_raw/spot/sol_1h.parquet"
DATA_ETH   = ROOT / "data/01_raw/spot/eth_1h.parquet"
DATA_OI    = ROOT / "data/01_raw/futures/sol_oi_4h.parquet"
DATA_TAKER = ROOT / "data/01_raw/futures/sol_taker_4h.parquet"

OUT_PLOTS  = ROOT / "prompts/plots/sol_v2_sweet_spot"
OUT_TABLES = ROOT / "prompts/tables/sol_v2_sweet_spot"
OUT_REPORT = ROOT / "prompts/sol_v2_sweet_spot_report.md"

OUT_PLOTS.mkdir(parents=True, exist_ok=True)
OUT_TABLES.mkdir(parents=True, exist_ok=True)

PARAMS = {
    "taker_z_4h_min":  0.3,
    "oi_z_24h_block":  2.0,
    "eth_ret_1h_min":  0.0,
    "ret_1d_min":      0.0,
    "rsi_min":        60.0,
    "rsi_max":        80.0,
    "sl_pct":          0.015,
    "tp_pct":          0.020,
    "trail_pct":       0.010,
    "max_hold_hours": 120,
    "cooldown_hours":   4,
}

# ─────────────────────────────────────────────
# Feature engineering helpers
# ─────────────────────────────────────────────
def _zscore(s: pd.Series, window: int) -> pd.Series:
    m  = s.rolling(window, min_periods=max(10, window // 4)).mean()
    sd = s.rolling(window, min_periods=max(10, window // 4)).std()
    return (s - m) / sd.replace(0, np.nan)


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


# ─────────────────────────────────────────────
# Build unified 1h DataFrame (full history)
# ─────────────────────────────────────────────
def build_dataset() -> pd.DataFrame:
    sol = pd.read_parquet(DATA_SOL)
    sol["timestamp"] = pd.to_datetime(sol["timestamp"], utc=True)
    sol = sol.sort_values("timestamp").drop_duplicates("timestamp").set_index("timestamp")

    eth = pd.read_parquet(DATA_ETH)
    eth["timestamp"] = pd.to_datetime(eth["timestamp"], utc=True)
    eth = eth.sort_values("timestamp").drop_duplicates("timestamp").set_index("timestamp")
    eth["eth_ret_1h"] = eth["close"].pct_change()
    sol["eth_ret_1h_prev"] = eth["eth_ret_1h"].reindex(sol.index, method="ffill").shift(1)

    oi = pd.read_parquet(DATA_OI)
    oi["timestamp"] = pd.to_datetime(oi["timestamp"], utc=True)
    oi = oi.sort_values("timestamp").drop_duplicates("timestamp").set_index("timestamp")
    oi["oi_z"] = _zscore(oi["open_interest"], 42)
    oi["oi_z_24h_max"] = oi["oi_z"].rolling(6, min_periods=1).max()
    sol["oi_z_24h_max_prev"] = oi["oi_z_24h_max"].reindex(sol.index, method="ffill").shift(1)

    taker = pd.read_parquet(DATA_TAKER)
    taker["timestamp"] = pd.to_datetime(taker["timestamp"], utc=True)
    taker = taker.sort_values("timestamp").drop_duplicates("timestamp").set_index("timestamp")
    buy_col  = next((c for c in ["buy_volume_usd", "taker_buy_volume_usd"]  if c in taker.columns), None)
    sell_col = next((c for c in ["sell_volume_usd", "taker_sell_volume_usd"] if c in taker.columns), None)
    if buy_col and sell_col:
        total = taker[buy_col] + taker[sell_col]
        taker["taker_ratio"] = taker[buy_col] / total.replace(0, np.nan)
    else:
        ratio_col = next((c for c in ["taker_ratio", "buy_sell_ratio"] if c in taker.columns), None)
        taker["taker_ratio"] = taker[ratio_col]
    taker["taker_z"] = _zscore(taker["taker_ratio"], 42)
    sol["taker_z_prev"] = taker["taker_z"].reindex(sol.index, method="ffill").shift(1)

    sol["rsi"]          = _rsi(sol["close"])
    sol["ma21"]         = sol["close"].rolling(21).mean()
    sol["ret_1d"]       = sol["close"].pct_change(24)
    sol["ret_1h"]       = sol["close"].pct_change(1)
    sol["volume_z_168"] = _zscore(sol["volume"], 168)
    sol["close_ma21_ratio"] = sol["close"] / sol["ma21"]

    return sol


# ─────────────────────────────────────────────
# Backtest engine
# ─────────────────────────────────────────────
def run_backtest(sol: pd.DataFrame, params: dict,
                 sweet_spot: dict | None = None,
                 start_date: str | None = None,
                 end_date:   str | None = None) -> list[dict]:
    """
    sweet_spot: {"cm_min", "cm_max", "vz_min", "vz_max"} or None (v1 only)
    """
    sl_pct     = params["sl_pct"]
    tp_pct     = params["tp_pct"]
    trail_pct  = params["trail_pct"]
    max_hold   = params["max_hold_hours"]
    cooldown_h = params["cooldown_hours"]

    data = sol.copy()
    if start_date:
        data = data[data.index >= start_date]
    if end_date:
        data = data[data.index <= end_date]

    in_trade = False
    entry_ts = entry_price = trail_high = sl_price = tp_price = max_hold_ts = None
    cooldown_until = None
    trades = []

    for ts, row in data.iterrows():
        # ── Exit ────────────────────────────────────────────────────
        if in_trade:
            h, l, c = row["high"], row["low"], row["close"]
            if c > trail_high:
                trail_high = c
            trail_sl  = trail_high * (1 - trail_pct)
            eff_sl    = max(sl_price, trail_sl)
            exit_r = exit_p = None
            if h >= tp_price:
                exit_r, exit_p = "TP",    tp_price
            elif l <= eff_sl:
                exit_r = "TRAIL" if eff_sl > sl_price else "SL"
                exit_p = eff_sl
            elif ts >= max_hold_ts:
                exit_r, exit_p = "MAX_HOLD", c
            if exit_r:
                pnl = (exit_p - entry_price) / entry_price
                trades.append({
                    "entry_ts": entry_ts, "exit_ts": ts,
                    "entry_price": entry_price, "exit_price": exit_p,
                    "exit_reason": exit_r,
                    "return_pct": pnl * 100,
                    "hold_hours": (ts - entry_ts).total_seconds() / 3600,
                    "close_ma21_ratio": float(data.loc[entry_ts, "close_ma21_ratio"]) if pd.notna(data.loc[entry_ts, "close_ma21_ratio"]) else None,
                    "volume_z": float(data.loc[entry_ts, "volume_z_168"]) if pd.notna(data.loc[entry_ts, "volume_z_168"]) else None,
                })
                in_trade = False
                cooldown_until = ts + pd.Timedelta(hours=cooldown_h)
            continue

        # ── Entry gate check ────────────────────────────────────────
        if cooldown_until and ts < cooldown_until:
            continue

        eth_ret = row.get("eth_ret_1h_prev")
        taker_z = row.get("taker_z_prev")
        oi24    = row.get("oi_z_24h_max_prev")
        ret_1d  = row.get("ret_1d")
        rsi     = row.get("rsi")
        ma21    = row.get("ma21")
        close   = row["close"]

        if pd.isna(eth_ret) or eth_ret <= params["eth_ret_1h_min"]: continue
        if pd.isna(taker_z) or taker_z  < params["taker_z_4h_min"]: continue
        if pd.notna(oi24) and oi24 >= params["oi_z_24h_block"]:     continue
        if pd.isna(ret_1d) or ret_1d <= params["ret_1d_min"]:       continue
        if pd.isna(rsi) or rsi < params["rsi_min"] or rsi > params["rsi_max"]: continue
        if pd.isna(ma21) or close < ma21:                            continue

        # ── v2 Sweet Spot filters ───────────────────────────────────
        if sweet_spot is not None:
            cm_ratio = row.get("close_ma21_ratio")
            vol_z    = row.get("volume_z_168")
            if pd.isna(cm_ratio) or not (sweet_spot["cm_min"] <= cm_ratio <= sweet_spot["cm_max"]):
                continue
            if pd.isna(vol_z) or not (sweet_spot["vz_min"] <= vol_z <= sweet_spot["vz_max"]):
                continue

        # ── Enter ───────────────────────────────────────────────────
        in_trade       = True
        entry_ts       = ts
        entry_price    = close
        trail_high     = close
        sl_price       = close * (1 - sl_pct)
        tp_price       = close * (1 + tp_pct)
        max_hold_ts    = ts + pd.Timedelta(hours=max_hold)

    return trades


# ─────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────
def metrics(trades: list[dict]) -> dict:
    if not trades:
        return {"n_trades": 0, "win_rate": 0, "avg_return": 0,
                "sharpe": 0, "max_dd": 0, "total_return": 0, "profit_factor": 0}
    rets   = pd.Series([t["return_pct"] for t in trades])
    wins   = rets[rets > 0]
    losses = rets[rets <= 0]
    eq     = (1 + rets / 100).cumprod()
    dd     = ((eq - eq.cummax()) / eq.cummax()).min() * 100
    total  = (eq.iloc[-1] - 1) * 100
    sharpe = float(rets.mean() / rets.std() * np.sqrt(52)) if rets.std() > 0 else 0
    pf = (wins.sum() / abs(losses.sum())) if (len(losses) > 0 and abs(losses.sum()) > 0) else float("inf")
    return {
        "n_trades":     len(trades),
        "win_rate":     float((rets > 0).mean()),
        "avg_return":   float(rets.mean()),
        "sharpe":       sharpe,
        "max_dd":       float(dd),
        "total_return": float(total),
        "profit_factor": float(pf),
    }


# ─────────────────────────────────────────────
# Plotting helpers
# ─────────────────────────────────────────────
DARK_BG   = "#0e1117"
CARD_BG   = "#1a1d23"
GREEN     = "#21c55d"
RED       = "#ff4b4b"
BLUE      = "#4a90d9"
YELLOW    = "#f0c040"
GREY_TXT  = "#cccccc"
GREY_LINE = "#444444"

def _fig(w=12, h=5):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(CARD_BG)
    for spine in ax.spines.values():
        spine.set_edgecolor(GREY_LINE)
    ax.tick_params(colors=GREY_TXT)
    return fig, ax


def plot_equity_curves(v1_trades, v2_trades, best_label):
    fig, ax = _fig(13, 5)
    for trades, label, color, lw in [
        (v1_trades, "v1 Baseline", GREY_LINE, 1.2),
        (v2_trades, f"v2 {best_label}", GREEN, 2.0),
    ]:
        if not trades:
            continue
        rets = pd.Series([t["return_pct"] for t in trades])
        eq   = (1 + rets / 100).cumprod()
        ax.plot(range(len(eq)), eq.values, label=label, color=color, linewidth=lw)
    ax.axhline(1.0, color=YELLOW, linewidth=0.6, linestyle="--")
    ax.set_xlabel("Trade #", color=GREY_TXT)
    ax.set_ylabel("Portfolio (normalised)", color=GREY_TXT)
    ax.set_title("SOL Bot 4 — Equity Curve: v1 vs v2 Sweet Spot", color="white", fontsize=13)
    ax.legend(facecolor=CARD_BG, labelcolor=GREY_TXT, fontsize=9)
    plt.tight_layout()
    plt.savefig(OUT_PLOTS / "equity_curve.png", dpi=100, bbox_inches="tight",
                facecolor=DARK_BG)
    plt.close()
    print("  → equity_curve.png")


def plot_returns_distribution(v1_trades, v2_trades):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor(DARK_BG)
    for ax, trades, label in [
        (axes[0], v1_trades, "v1 Baseline"),
        (axes[1], v2_trades, "v2 Sweet Spot"),
    ]:
        ax.set_facecolor(CARD_BG)
        for spine in ax.spines.values():
            spine.set_edgecolor(GREY_LINE)
        ax.tick_params(colors=GREY_TXT)
        if not trades:
            ax.set_title(f"{label}\n(no trades)", color="white")
            continue
        rets = [t["return_pct"] for t in trades]
        colors_bar = [GREEN if r > 0 else RED for r in rets]
        ax.bar(range(len(rets)), rets, color=colors_bar, alpha=0.8, width=0.7)
        ax.axhline(0, color=YELLOW, linewidth=0.8, linestyle="--")
        m = metrics(trades)
        ax.set_title(f"{label}  N={m['n_trades']}  WR={m['win_rate']:.0%}  "
                     f"Sharpe={m['sharpe']:.2f}", color="white", fontsize=10)
        ax.set_xlabel("Trade #", color=GREY_TXT)
        ax.set_ylabel("Return %", color=GREY_TXT)
    plt.suptitle("SOL Bot 4 — Returns Distribution", color="white", fontsize=12)
    plt.tight_layout()
    plt.savefig(OUT_PLOTS / "trades_distribution.png", dpi=100, bbox_inches="tight",
                facecolor=DARK_BG)
    plt.close()
    print("  → trades_distribution.png")


def plot_threshold_heatmap(sol, params, cm_vals, vz_vals):
    sharpe_grid = np.zeros((len(vz_vals), len(cm_vals)))
    for i, vz in enumerate(vz_vals):
        for j, cm in enumerate(cm_vals):
            ss = {"cm_min": cm[0], "cm_max": cm[1], "vz_min": -0.1, "vz_max": vz}
            t  = run_backtest(sol, params, sweet_spot=ss)
            sharpe_grid[i, j] = metrics(t)["sharpe"] if metrics(t)["n_trades"] >= 3 else np.nan

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(CARD_BG)
    im = ax.imshow(sharpe_grid, cmap="RdYlGn", aspect="auto",
                   vmin=-2, vmax=4)
    plt.colorbar(im, ax=ax, label="Sharpe")
    ax.set_xticks(range(len(cm_vals)))
    ax.set_xticklabels([f"{v[0]:.3f}-{v[1]:.3f}" for v in cm_vals],
                       rotation=30, ha="right", color=GREY_TXT, fontsize=8)
    ax.set_yticks(range(len(vz_vals)))
    ax.set_yticklabels([f"vz_max={v:.1f}" for v in vz_vals], color=GREY_TXT, fontsize=8)
    ax.set_xlabel("Close/MA21 range", color=GREY_TXT)
    ax.set_ylabel("Volume Z max", color=GREY_TXT)
    ax.set_title("Sharpe Heatmap — Threshold Grid", color="white", fontsize=12)
    for i in range(len(vz_vals)):
        for j in range(len(cm_vals)):
            v = sharpe_grid[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        color="white", fontsize=7)
    plt.tight_layout()
    plt.savefig(OUT_PLOTS / "threshold_heatmap.png", dpi=100, bbox_inches="tight",
                facecolor=DARK_BG)
    plt.close()
    print("  → threshold_heatmap.png")


def plot_subperiods(subperiod_results):
    periods = [r["period"] for r in subperiod_results]
    sharpes = [r["sharpe"] for r in subperiod_results]
    ns      = [r["n_trades"] for r in subperiod_results]

    fig, ax1 = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor(DARK_BG)
    ax1.set_facecolor(CARD_BG)
    for spine in ax1.spines.values():
        spine.set_edgecolor(GREY_LINE)
    ax1.tick_params(colors=GREY_TXT)

    ax2 = ax1.twinx()
    x   = np.arange(len(periods))
    colors_bar = [GREEN if s > 1.0 else (YELLOW if s > 0 else RED) for s in sharpes]
    bars = ax1.bar(x, sharpes, color=colors_bar, alpha=0.8, width=0.5)
    ax2.plot(x, ns, "o-", color=BLUE, linewidth=1.5, label="N trades")
    ax1.axhline(0,   color=GREY_LINE, linewidth=0.6, linestyle="--")
    ax1.axhline(1.5, color=GREEN,     linewidth=0.8, linestyle="--", alpha=0.5, label="Target 1.5")

    ax1.set_xticks(x)
    ax1.set_xticklabels(periods, color=GREY_TXT, fontsize=9)
    ax1.set_ylabel("Sharpe", color=GREY_TXT)
    ax2.set_ylabel("N Trades", color=BLUE)
    ax2.tick_params(colors=BLUE)
    ax1.set_title("SOL Bot 4 v2 — Robustez por Sub-período", color="white", fontsize=12)
    ax1.legend(facecolor=CARD_BG, labelcolor=GREY_TXT, loc="upper left", fontsize=8)

    for bar, s in zip(bars, sharpes):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                 f"{s:.2f}", ha="center", va="bottom", color="white", fontsize=9)
    plt.tight_layout()
    plt.savefig(OUT_PLOTS / "subperiods_comparison.png", dpi=100, bbox_inches="tight",
                facecolor=DARK_BG)
    plt.close()
    print("  → subperiods_comparison.png")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    print("=" * 65)
    print("SOL Bot 4 v2 'Sweet Spot' — Backtest Rigoroso")
    print("=" * 65)

    # ── 1. Build dataset ──────────────────────────────────────────
    print("\n[1/9] Building dataset...")
    sol = build_dataset()
    print(f"  Total rows: {len(sol)}  ({sol.index.min()} → {sol.index.max()})")

    # ── 2. Baseline v1 (full history) ─────────────────────────────
    print("\n[2/9] Baseline v1 (full history, no sweet spot)...")
    v1_trades  = run_backtest(sol, PARAMS)
    v1_metrics = metrics(v1_trades)
    print(f"  Sharpe:  {v1_metrics['sharpe']:.2f}  (expected ~0.08)")
    print(f"  Trades:  {v1_metrics['n_trades']}")
    print(f"  WR:      {v1_metrics['win_rate']:.1%}")
    print(f"  Avg Ret: {v1_metrics['avg_return']:.3f}%")
    print(f"  Max DD:  {v1_metrics['max_dd']:.2f}%")

    # ── 3. v2 Central Sweet Spot ──────────────────────────────────
    CENTRAL = {"cm_min": 1.03, "cm_max": 1.05, "vz_min": 0.0, "vz_max": 0.5}
    print("\n[3/9] v2 Central Sweet Spot (close_ma21 1.03-1.05, vol_z 0-0.5)...")
    v2_trades  = run_backtest(sol, PARAMS, sweet_spot=CENTRAL)
    v2_metrics = metrics(v2_trades)
    print(f"  Sharpe:  {v2_metrics['sharpe']:.2f}")
    print(f"  Trades:  {v2_metrics['n_trades']}")
    print(f"  WR:      {v2_metrics['win_rate']:.1%}")
    print(f"  Avg Ret: {v2_metrics['avg_return']:.3f}%")
    print(f"  Max DD:  {v2_metrics['max_dd']:.2f}%")
    print(f"  PF:      {v2_metrics['profit_factor']:.2f}")
    delta_v1_v2 = v2_metrics["sharpe"] - v1_metrics["sharpe"]
    print(f"  Δ vs v1: {delta_v1_v2:+.2f}")

    # ── 4. Grid Search — 5 variations ────────────────────────────
    print("\n[4/9] Grid search (5 variations + central)...")
    variations = [
        ("v2_permissivo",    {"cm_min": 1.025, "cm_max": 1.055, "vz_min": -0.1, "vz_max": 0.6}),
        ("v2_central",       {"cm_min": 1.030, "cm_max": 1.050, "vz_min":  0.0, "vz_max": 0.5}),
        ("v2_restritivo",    {"cm_min": 1.035, "cm_max": 1.045, "vz_min":  0.0, "vz_max": 0.3}),
        ("v2_cm_permissivo", {"cm_min": 1.030, "cm_max": 1.060, "vz_min": -0.2, "vz_max": 0.7}),
        ("v2_vz_restritivo", {"cm_min": 1.025, "cm_max": 1.050, "vz_min":  0.1, "vz_max": 0.5}),
    ]

    grid_results = []
    all_var_trades = {}
    for name, ss in variations:
        t = run_backtest(sol, PARAMS, sweet_spot=ss)
        m = metrics(t)
        all_var_trades[name] = t
        keep = m["n_trades"] / v1_metrics["n_trades"] * 100 if v1_metrics["n_trades"] else 0
        print(f"  {name:<22} n={m['n_trades']:2d}  wr={m['win_rate']:.0%}  "
              f"sharpe={m['sharpe']:.2f}  Δ={m['sharpe']-v1_metrics['sharpe']:+.2f}  "
              f"kept={keep:.0f}%")
        grid_results.append({
            "config": name,
            "cm_min": ss["cm_min"], "cm_max": ss["cm_max"],
            "vz_min": ss["vz_min"], "vz_max": ss["vz_max"],
            "n_trades": m["n_trades"],
            "win_rate": round(m["win_rate"], 3),
            "avg_return": round(m["avg_return"], 4),
            "sharpe": round(m["sharpe"], 3),
            "max_dd": round(m["max_dd"], 3),
            "total_return": round(m["total_return"], 3),
            "profit_factor": round(m["profit_factor"], 3),
            "trades_kept_pct": round(keep, 1),
        })

    grid_df = pd.DataFrame(grid_results)
    grid_df.to_csv(OUT_TABLES / "grid_search.csv", index=False)
    print(f"  → grid_search.csv saved")

    # ── 5. Train / Test split (70/30 temporal) ────────────────────
    print("\n[5/9] Train/Test split (70/30 temporal)...")
    split_idx = int(len(sol) * 0.70)
    split_ts  = sol.index[split_idx]
    print(f"  Train: {sol.index.min()} → {split_ts}")
    print(f"  Test:  {split_ts} → {sol.index.max()}")

    # Find best config on TRAIN
    best_train_sharpe = -999
    best_train_config = None
    best_train_name   = None
    for name, ss in variations:
        t_train = run_backtest(sol, PARAMS, sweet_spot=ss, end_date=str(split_ts))
        m_train = metrics(t_train)
        if m_train["sharpe"] > best_train_sharpe and m_train["n_trades"] >= 5:
            best_train_sharpe = m_train["sharpe"]
            best_train_config = ss
            best_train_name   = name

    # v1 train
    v1_train = run_backtest(sol, PARAMS, end_date=str(split_ts))
    v1_test  = run_backtest(sol, PARAMS, start_date=str(split_ts))
    v1_m_train = metrics(v1_train)
    v1_m_test  = metrics(v1_test)

    # Apply best config to TEST
    if best_train_config:
        v2_test   = run_backtest(sol, PARAMS, sweet_spot=best_train_config,
                                 start_date=str(split_ts))
        v2_m_test = metrics(v2_test)
        best_train_n = metrics(run_backtest(sol, PARAMS, sweet_spot=best_train_config,
                                            end_date=str(split_ts)))["n_trades"]
    else:
        v2_test     = []
        v2_m_test   = metrics([])
        best_train_n = 0

    overfit_delta = best_train_sharpe - v2_m_test["sharpe"]
    print(f"  Best config (train): {best_train_name}")
    print(f"  Train Sharpe: {best_train_sharpe:.2f}  (N={best_train_n})")
    print(f"  Test  Sharpe: {v2_m_test['sharpe']:.2f}  (N={v2_m_test['n_trades']})")
    print(f"  Overfitting Δ: {overfit_delta:+.2f}")
    print(f"  v1 Train/Test: {v1_m_train['sharpe']:.2f} / {v1_m_test['sharpe']:.2f}")

    # ── 6. Sub-period robustness ──────────────────────────────────
    print("\n[6/9] Sub-period robustness (v2 central)...")
    periods = [
        ("Out-Dez 2025", "2025-10-01", "2025-12-31"),
        ("Jan-Feb 2026", "2026-01-01", "2026-02-28"),
        ("Mar-Abr 2026", "2026-03-01", "2026-04-22"),
    ]
    subperiod_results = []
    for name, start, end in periods:
        t_v1 = run_backtest(sol, PARAMS,                     start_date=start, end_date=end)
        t_v2 = run_backtest(sol, PARAMS, sweet_spot=CENTRAL, start_date=start, end_date=end)
        m_v1 = metrics(t_v1)
        m_v2 = metrics(t_v2)
        print(f"  {name}: v1 Sharpe={m_v1['sharpe']:.2f} N={m_v1['n_trades']}  |  "
              f"v2 Sharpe={m_v2['sharpe']:.2f} N={m_v2['n_trades']}")
        subperiod_results.append({
            "period": name, "start": start, "end": end,
            "v1_n": m_v1["n_trades"], "v1_sharpe": round(m_v1["sharpe"], 3),
            "v1_wr": round(m_v1["win_rate"], 3),
            "v2_n": m_v2["n_trades"], "v2_sharpe": round(m_v2["sharpe"], 3),
            "v2_wr": round(m_v2["win_rate"], 3),
            "n_trades": m_v2["n_trades"],  # for plot
            "sharpe":   round(m_v2["sharpe"], 3),  # for plot
        })

    sp_df = pd.DataFrame(subperiod_results)
    sp_df.to_csv(OUT_TABLES / "subperiods_results.csv", index=False)

    # Count how many periods v2 has Sharpe > 1.0
    n_good_periods = sum(1 for r in subperiod_results if r["v2_sharpe"] > 1.0)
    print(f"  Periods with Sharpe > 1.0: {n_good_periods}/3")

    # ── 7. Trade 22/04 validation ─────────────────────────────────
    print("\n[7/9] Trade 22/04 validation (entry $88.23)...")
    close_22 = 88.33   # from filter study (closest available candle)
    ma21_22  = 87.45   # from entry_features
    vol_z_22 = 0.218   # from filter study
    cm_22    = close_22 / ma21_22

    in_cm_central  = CENTRAL["cm_min"] <= cm_22 <= CENTRAL["cm_max"]
    in_vz_central  = CENTRAL["vz_min"] <= vol_z_22 <= CENTRAL["vz_max"]
    v2_would_block = not (in_cm_central and in_vz_central)

    print(f"  close/MA21: {cm_22:.4f}  in [{CENTRAL['cm_min']},{CENTRAL['cm_max']}]? {in_cm_central}")
    print(f"  volume_z:   {vol_z_22:.3f}  in [{CENTRAL['vz_min']},{CENTRAL['vz_max']}]? {in_vz_central}")
    print(f"  v2 central would block: {v2_would_block}")
    if not in_cm_central:
        print(f"  Reason: close_ma21={cm_22:.4f} outside [{CENTRAL['cm_min']},{CENTRAL['cm_max']}]")
    if not in_vz_central:
        print(f"  Reason: volume_z={vol_z_22:.3f} outside [{CENTRAL['vz_min']},{CENTRAL['vz_max']}]")

    # ── 8. Decision ───────────────────────────────────────────────
    print("\n[8/9] Decision criteria...")
    c1 = v2_metrics["sharpe"]    > 1.5
    c2 = v2_metrics["n_trades"]  >= 15
    c3 = abs(v2_metrics["max_dd"]) < 5.0
    c4 = abs(overfit_delta)       < 0.5
    c5 = n_good_periods           >= 2

    print(f"  C1 Sharpe > 1.5:        {c1}  ({v2_metrics['sharpe']:.2f})")
    print(f"  C2 N ≥ 15:              {c2}  ({v2_metrics['n_trades']})")
    print(f"  C3 Max DD < 5%:         {c3}  ({v2_metrics['max_dd']:.2f}%)")
    print(f"  C4 Overfitting Δ < 0.5: {c4}  ({overfit_delta:+.2f})")
    print(f"  C5 Robust (2/3 periodos): {c5}  ({n_good_periods}/3)")

    n_criteria_met = sum([c1, c2, c3, c4, c5])
    if n_criteria_met >= 4 and c1 and c2:
        verdict = "APROVADO"
        action  = "Implementar v2 em sol_bot4.py + reativar crontab"
    elif v2_metrics["sharpe"] < 1.0 or v2_metrics["n_trades"] < 10 or abs(v2_metrics["max_dd"]) > 10:
        verdict = "REJEITADO"
        action  = "Não reativar Bot 4 v2. Considerar re-EDA ou abandono SOL."
    else:
        verdict = "INCONCLUSIVO"
        action  = "Acumular mais dados. Revisitar com N ≥ 30 trades históricos."

    print(f"\n  → VEREDITO: {verdict}")
    print(f"  → AÇÃO:    {action}")

    # ── 9. Plots & Tables ─────────────────────────────────────────
    print("\n[9/9] Generating plots and tables...")

    # Best v2 config for plots
    best_v2_for_plot = max(all_var_trades.items(),
                           key=lambda x: metrics(x[1])["sharpe"] if metrics(x[1])["n_trades"] >= 5 else -99)
    best_v2_name = best_v2_for_plot[0]
    best_v2_t    = best_v2_for_plot[1]

    plot_equity_curves(v1_trades, v2_trades, "central (1.03-1.05 / 0-0.5)")
    plot_returns_distribution(v1_trades, v2_trades)

    # Heatmap grid (7x5 smaller grid to keep runtime reasonable)
    cm_ranges = [(1.020, 1.040), (1.025, 1.045), (1.030, 1.050),
                 (1.035, 1.055), (1.040, 1.060)]
    vz_maxes  = [0.3, 0.5, 0.7, 1.0, 1.5]
    plot_threshold_heatmap(sol, PARAMS, cm_ranges, vz_maxes)
    plot_subperiods(subperiod_results)

    # Trades comparison table
    all_trades = (
        [{"config": "v1", **t} for t in v1_trades] +
        [{"config": "v2_central", **t} for t in v2_trades]
    )
    pd.DataFrame(all_trades).to_csv(OUT_TABLES / "trades_v1_vs_v2.csv", index=False)
    print(f"  → trades_v1_vs_v2.csv ({len(all_trades)} rows)")

    # ── Report ─────────────────────────────────────────────────────
    print(f"\n[REPORT] {OUT_REPORT}")

    def _row(r):
        pf = "∞" if r["profit_factor"] >= 99 else f"{r['profit_factor']:.2f}"
        return (f"| {r['config']:<22} | {r['n_trades']:3d} | {r['win_rate']:.0%} | "
                f"{r['avg_return']:+.3f}% | {r['sharpe']:.2f} | "
                f"{r['max_dd']:.2f}% | {r['total_return']:+.2f}% | {pf} |")

    grid_header = ("| Config | N | WR | Avg Ret | Sharpe | Max DD | Total Ret | PF |\n"
                   "|--------|---|-----|---------|--------|--------|-----------|----|\n")
    grid_body = (grid_header +
                 _row({"config": "v1_baseline", **{k: round(v, 4) if isinstance(v, float) else v
                                                   for k, v in v1_metrics.items()}}) + "\n" +
                 "\n".join(_row(r) for r in grid_results))

    sp_rows = "\n".join(
        f"| {r['period']:<15} | {r['v1_n']:2d} | {r['v1_sharpe']:.2f} | {r['v1_wr']:.0%} "
        f"| {r['v2_n']:2d} | {r['v2_sharpe']:.2f} | {r['v2_wr']:.0%} |"
        for r in subperiod_results
    )

    # Implementation code (only if approved)
    impl_code = ""
    if verdict == "APROVADO":
        best_cfg = best_train_config or CENTRAL
        impl_code = f"""
## 8. Implementação

```python
# Em src/trading/sol_bot4.py — adicionar em check_entry_signal():

def check_sweet_spot_v2(features: dict, df: pd.DataFrame) -> tuple[bool, str | None]:
    \"\"\"v2 Sweet Spot guard — close/MA21 and volume_z range.\"\"\"
    cm_ratio = features.get("close_ma21_ratio")
    vol_z    = features.get("volume_z_168")

    if cm_ratio is None or not ({best_cfg['cm_min']} <= cm_ratio <= {best_cfg['cm_max']}):
        return False, f"close_ma21_outside_{{cm_ratio:.3f}}"
    if vol_z is None or not ({best_cfg['vz_min']} <= vol_z <= {best_cfg['vz_max']}):
        return False, f"volume_z_outside_{{vol_z:.2f}}"
    return True, None
```

```yaml
# Em conf/parameters_sol.yml — adicionar em filters:
  sweet_spot_v2:
    enabled: true
    close_ma21_min: {best_cfg['cm_min']}
    close_ma21_max: {best_cfg['cm_max']}
    volume_z_min: {best_cfg['vz_min']}
    volume_z_max: {best_cfg['vz_max']}
```
"""

    report = f"""# SOL Bot 4 v2 "Sweet Spot" — Backtest Report

**Data:** 2026-04-22
**Objetivo:** Validar strategy v2 antes de reativar Bot 4 (pausado desde 22/04)

---

## 1. v1 Baseline (replicação)

| Metric | v1 Full History |
|--------|----------------|
| Sharpe | {v1_metrics['sharpe']:.2f} |
| Trades | {v1_metrics['n_trades']} |
| Win Rate | {v1_metrics['win_rate']:.0%} |
| Avg Return | {v1_metrics['avg_return']:.3f}% |
| Max DD | {v1_metrics['max_dd']:.2f}% |

> Filter study confirmado: Sharpe {v1_metrics['sharpe']:.2f} (Phase 2 reportou 2.03 em test set N=21 — lucky sampling)

---

## 2. v2 Sweet Spot Central

Config: `close/MA21 ∈ [1.03, 1.05]` e `volume_z ∈ [0.0, 0.5]`

| Metric | v2 Central |
|--------|-----------|
| Sharpe | **{v2_metrics['sharpe']:.2f}** |
| Trades | {v2_metrics['n_trades']} |
| Win Rate | {v2_metrics['win_rate']:.0%} |
| Avg Return | {v2_metrics['avg_return']:.3f}% |
| Max DD | {v2_metrics['max_dd']:.2f}% |
| Total Return | {v2_metrics['total_return']:+.2f}% |
| Profit Factor | {v2_metrics['profit_factor']:.2f} |
| Δ vs v1 | {delta_v1_v2:+.2f} |

---

## 3. Grid Search — Variações

{grid_body}

---

## 4. Train / Test Split (70/30 temporal)

- **Train:** {sol.index.min().date()} → {split_ts.date()}
- **Test:**  {split_ts.date()} → {sol.index.max().date()}
- **Best config (train):** `{best_train_name}` — Sharpe {best_train_sharpe:.2f} (N={best_train_n})
- **Test Sharpe:** {v2_m_test['sharpe']:.2f} (N={v2_m_test['n_trades']})
- **Overfitting Δ:** {overfit_delta:+.2f}

{'✅ Boa generalização (Δ < 0.5)' if abs(overfit_delta) < 0.5 else '⚠️ Over-fit suspeito (Δ ≥ 0.5)' if abs(overfit_delta) < 1.0 else '🔴 Over-fit provável (Δ ≥ 1.0)'}

---

## 5. Robustez por Sub-período

| Período | v1 N | v1 Sharpe | v1 WR | v2 N | v2 Sharpe | v2 WR |
|---------|------|-----------|-------|------|-----------|-------|
{sp_rows}

**Períodos v2 com Sharpe > 1.0:** {n_good_periods}/3

{'✅ Strategy regime-robust' if n_good_periods >= 2 else '⚠️ Regime-dependent' if n_good_periods == 1 else '🔴 Sem edge consistente'}

---

## 6. Trade 22/04 Validation

| Feature | Valor real | Range v2 central | Passa? |
|---------|-----------|-------------------|--------|
| close/MA21 | {cm_22:.4f} | [{CENTRAL['cm_min']}, {CENTRAL['cm_max']}] | {'✅' if in_cm_central else '🛡️ BLOQUEADO'} |
| volume_z | {vol_z_22:.3f} | [{CENTRAL['vz_min']}, {CENTRAL['vz_max']}] | {'✅' if in_vz_central else '🛡️ BLOQUEADO'} |

**v2 teria bloqueado a trade 22/04?** {'✅ SIM' if v2_would_block else '❌ NÃO'}

---

## 7. Critérios de Decisão

| Critério | Meta | Resultado | Status |
|----------|------|-----------|--------|
| Sharpe > 1.5 | > 1.5 | {v2_metrics['sharpe']:.2f} | {'✅' if c1 else '❌'} |
| N trades ≥ 15 | ≥ 15 | {v2_metrics['n_trades']} | {'✅' if c2 else '❌'} |
| Max DD < 5% | < 5% | {v2_metrics['max_dd']:.2f}% | {'✅' if c3 else '❌'} |
| Overfitting Δ < 0.5 | < 0.5 | {overfit_delta:+.2f} | {'✅' if c4 else '❌'} |
| Robusto 2/3 períodos | ≥ 2/3 | {n_good_periods}/3 | {'✅' if c5 else '❌'} |

**Critérios atendidos:** {n_criteria_met}/5

---

## **VEREDITO: {verdict}**

**Ação:** {action}
{impl_code}

---

## 9. Plots gerados

- `plots/sol_v2_sweet_spot/equity_curve.png`
- `plots/sol_v2_sweet_spot/trades_distribution.png`
- `plots/sol_v2_sweet_spot/threshold_heatmap.png`
- `plots/sol_v2_sweet_spot/subperiods_comparison.png`

## 10. Tables geradas

- `tables/sol_v2_sweet_spot/grid_search.csv`
- `tables/sol_v2_sweet_spot/trades_v1_vs_v2.csv`
- `tables/sol_v2_sweet_spot/subperiods_results.csv`
"""

    OUT_REPORT.write_text(report)
    print(f"  → {OUT_REPORT} saved")

    print("\n" + "=" * 65)
    print(f"DONE — VEREDITO: {verdict}")
    print("=" * 65)

    return verdict, v2_metrics, overfit_delta


if __name__ == "__main__":
    main()
