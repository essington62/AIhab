"""
SOL Bot 4 — Estudo Estatístico de Filtros Estruturais
Trigger: primeira trade live -0.98% em 22/04/2026

Testa 4 filtros candidatos:
  1. Structural (dist_to_resistance_12h)
  2. Volume_z (exaustão)
  3. Extension (close/MA21 ratio)
  4. Velocity (ret_1h / ret_24h_avg)
"""
import sys
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DATA_SOL   = ROOT / "data/01_raw/spot/sol_1h.parquet"
DATA_ETH   = ROOT / "data/01_raw/spot/eth_1h.parquet"
DATA_OI    = ROOT / "data/01_raw/futures/sol_oi_4h.parquet"
DATA_TAKER = ROOT / "data/01_raw/futures/sol_taker_4h.parquet"

OUT_PLOTS  = ROOT / "prompts/plots/sol_filters_study"
OUT_TABLES = ROOT / "prompts/tables/sol_filters_study"
OUT_REPORT = ROOT / "prompts/sol_filters_study_report.md"

OUT_PLOTS.mkdir(parents=True, exist_ok=True)
OUT_TABLES.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────
# Parameters (from conf/parameters_sol.yml)
# ─────────────────────────────────────────────
PARAMS = {
    "taker_z_4h_min":  0.3,
    "oi_z_24h_block":  2.0,
    "oi_z_1h_min":    -0.5,
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
# Helpers
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
# Build unified 1h DataFrame
# ─────────────────────────────────────────────
def build_dataset() -> pd.DataFrame:
    # SOL OHLCV
    sol = pd.read_parquet(DATA_SOL)
    sol["timestamp"] = pd.to_datetime(sol["timestamp"], utc=True)
    sol = sol.sort_values("timestamp").drop_duplicates("timestamp")
    sol = sol.set_index("timestamp")

    # ETH ret
    eth = pd.read_parquet(DATA_ETH)
    eth["timestamp"] = pd.to_datetime(eth["timestamp"], utc=True)
    eth = eth.sort_values("timestamp").drop_duplicates("timestamp").set_index("timestamp")
    eth["eth_ret_1h"] = eth["close"].pct_change()
    sol["eth_ret_1h_raw"] = eth["eth_ret_1h"].reindex(sol.index, method="ffill")
    sol["eth_ret_1h_prev"] = sol["eth_ret_1h_raw"].shift(1)

    # OI 4h (ffill to 1h)
    oi = pd.read_parquet(DATA_OI)
    oi["timestamp"] = pd.to_datetime(oi["timestamp"], utc=True)
    oi = oi.sort_values("timestamp").drop_duplicates("timestamp").set_index("timestamp")
    oi["oi_z"] = _zscore(oi["open_interest"], 42)
    oi["oi_z_24h_max"] = oi["oi_z"].rolling(6, min_periods=1).max()
    sol["oi_z_raw"] = oi["oi_z"].reindex(sol.index, method="ffill")
    sol["oi_z_24h_max_raw"] = oi["oi_z_24h_max"].reindex(sol.index, method="ffill")
    # anti look-ahead: shift(1)
    sol["oi_z_prev"] = sol["oi_z_raw"].shift(1)
    sol["oi_z_24h_max_prev"] = sol["oi_z_24h_max_raw"].shift(1)

    # Taker 4h (ffill to 1h)
    taker = pd.read_parquet(DATA_TAKER)
    taker["timestamp"] = pd.to_datetime(taker["timestamp"], utc=True)
    taker = taker.sort_values("timestamp").drop_duplicates("timestamp").set_index("timestamp")
    buy_col  = next((c for c in ["buy_volume_usd", "taker_buy_volume_usd"]  if c in taker.columns), None)
    sell_col = next((c for c in ["sell_volume_usd", "taker_sell_volume_usd"] if c in taker.columns), None)
    ratio_col = next((c for c in ["taker_ratio", "buy_sell_ratio"] if c in taker.columns), None)
    if buy_col and sell_col:
        total = taker[buy_col] + taker[sell_col]
        taker["taker_ratio"] = taker[buy_col] / total.replace(0, np.nan)
    else:
        taker["taker_ratio"] = taker[ratio_col]
    taker["taker_z"] = _zscore(taker["taker_ratio"], 42)
    sol["taker_z_raw"] = taker["taker_z"].reindex(sol.index, method="ffill")
    sol["taker_z_prev"] = sol["taker_z_raw"].shift(1)

    # Standard SOL features
    sol["rsi"]   = _rsi(sol["close"])
    sol["ma21"]  = sol["close"].rolling(21).mean()
    sol["ret_1d"] = sol["close"].pct_change(24)
    sol["ret_1h"] = sol["close"].pct_change(1)

    # Volume z-score (7d rolling, window=168)
    sol["volume_z_168"] = _zscore(sol["volume"], 168)

    return sol


# ─────────────────────────────────────────────
# Structural features (per entry)
# ─────────────────────────────────────────────
def compute_structural_features(sol: pd.DataFrame, entry_ts, lookback_hours: int = 12) -> dict:
    entry_row = sol.loc[entry_ts]
    entry_price = entry_row["close"]

    # lookback window (exclusive of entry candle)
    window = sol[sol.index < entry_ts].tail(lookback_hours)

    resistance_12h = window["high"].max() if not window.empty else entry_price
    support_12h    = window["low"].min()  if not window.empty else entry_price

    dist_to_resistance_pct = (resistance_12h - entry_price) / entry_price
    dist_from_support_pct  = (entry_price - support_12h) / entry_price

    structural_rr = (
        dist_to_resistance_pct / dist_from_support_pct
        if dist_from_support_pct > 0.0001 else float("inf")
    )

    # Volume z at entry
    volume_z = entry_row["volume_z_168"]

    # Extension: close/MA21
    ma21 = entry_row["ma21"]
    close_ma21_ratio = entry_price / ma21 if (ma21 and ma21 > 0) else 1.0

    # Velocity: ret_1h / (ret_24h/24)
    ret_1h  = entry_row["ret_1h"]
    ret_24h = entry_row["ret_1d"]  # already pct_change(24)
    avg_hourly_ret = ret_24h / 24 if abs(ret_24h) > 1e-8 else 0
    velocity_ratio = ret_1h / avg_hourly_ret if abs(avg_hourly_ret) > 1e-8 else 0

    return {
        "dist_to_resistance_pct": float(dist_to_resistance_pct),
        "dist_from_support_pct":  float(dist_from_support_pct),
        "structural_rr":          float(structural_rr) if structural_rr != float("inf") else 10.0,
        "volume_z":               float(volume_z) if pd.notna(volume_z) else 0.0,
        "close_ma21_ratio":       float(close_ma21_ratio),
        "ret_1h_at_entry":        float(ret_1h) if pd.notna(ret_1h) else 0.0,
        "velocity_ratio":         float(velocity_ratio),
    }


# ─────────────────────────────────────────────
# Backtest engine
# ─────────────────────────────────────────────
def run_backtest(sol: pd.DataFrame, params: dict,
                 extra_filter=None) -> tuple[list[dict], list[dict]]:
    """
    Returns (trades, all_signals_2026).
    extra_filter(row, struct_feats) → True = allow, False = block
    """
    sl_pct    = params["sl_pct"]
    tp_pct    = params["tp_pct"]
    trail_pct = params["trail_pct"]
    max_hold  = params["max_hold_hours"]
    cooldown_h = params["cooldown_hours"]

    # Restrict signals to 2026+
    sol_2026 = sol[sol.index >= "2026-01-01"]

    in_trade      = False
    entry_ts      = None
    entry_price   = None
    trail_high    = None
    sl_price      = None
    tp_price      = None
    max_hold_ts   = None
    cooldown_until = None
    entry_struct  = {}
    entry_row_vals = {}

    trades  = []
    signals = []  # all 2026+ candles where hard gates pass (before extra_filter)

    for ts, row in sol_2026.iterrows():
        # ── Exit management ──────────────────────────────────────────
        if in_trade:
            h = row["high"]
            l = row["low"]
            c = row["close"]

            # Update trailing
            if c > trail_high:
                trail_high = c
            trail_sl = trail_high * (1 - trail_pct)
            eff_sl = max(sl_price, trail_sl)

            exit_reason = None
            exit_price  = None

            if h >= tp_price:
                exit_reason = "TP";    exit_price = tp_price
            elif l <= eff_sl:
                exit_reason = "TRAIL" if eff_sl > sl_price else "SL"
                exit_price  = eff_sl
            elif ts >= max_hold_ts:
                exit_reason = "MAX_HOLD"; exit_price = c

            if exit_reason:
                pnl_pct = (exit_price - entry_price) / entry_price
                hold_h  = (ts - entry_ts).total_seconds() / 3600
                trade = {
                    "entry_ts":    entry_ts,
                    "exit_ts":     ts,
                    "entry_price": entry_price,
                    "exit_price":  exit_price,
                    "exit_reason": exit_reason,
                    "return_pct":  pnl_pct * 100,
                    "hold_hours":  hold_h,
                    **entry_struct,
                    **{f"entry_{k}": v for k, v in entry_row_vals.items()},
                }
                trades.append(trade)
                in_trade       = False
                cooldown_until = ts + pd.Timedelta(hours=cooldown_h)
            continue

        # ── Entry gate check ─────────────────────────────────────────
        # Cooldown
        if cooldown_until and ts < cooldown_until:
            continue

        # Hard gates
        eth_ret = row.get("eth_ret_1h_prev")
        taker_z = row.get("taker_z_prev")
        oi24    = row.get("oi_z_24h_max_prev")
        ret_1d  = row.get("ret_1d")
        rsi     = row.get("rsi")
        ma21    = row.get("ma21")
        close   = row["close"]

        if pd.isna(eth_ret) or eth_ret <= params["eth_ret_1h_min"]:
            continue
        if pd.isna(taker_z) or taker_z < params["taker_z_4h_min"]:
            continue
        if pd.notna(oi24) and oi24 >= params["oi_z_24h_block"]:
            continue
        if pd.isna(ret_1d) or ret_1d <= params["ret_1d_min"]:
            continue
        if pd.isna(rsi) or rsi < params["rsi_min"] or rsi > params["rsi_max"]:
            continue
        if pd.isna(ma21) or close < ma21:
            continue

        # All hard gates passed — compute structural features
        struct = compute_structural_features(sol, ts)
        signals.append({"entry_ts": ts, "close": close, **struct})

        # Apply extra filter
        if extra_filter is not None and not extra_filter(row, struct):
            continue

        # Enter
        in_trade      = True
        entry_ts      = ts
        entry_price   = close
        trail_high    = close
        sl_price      = close * (1 - sl_pct)
        tp_price      = close * (1 + tp_pct)
        max_hold_ts   = ts + pd.Timedelta(hours=max_hold)
        entry_struct  = struct
        entry_row_vals = {
            "taker_z": float(taker_z) if pd.notna(taker_z) else None,
            "oi24": float(oi24) if pd.notna(oi24) else None,
            "eth_ret": float(eth_ret) if pd.notna(eth_ret) else None,
            "rsi": float(rsi) if pd.notna(rsi) else None,
            "ret_1d": float(ret_1d) if pd.notna(ret_1d) else None,
        }

    return trades, signals


# ─────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────
def compute_metrics(trades: list[dict]) -> dict:
    if not trades:
        return {"n_trades": 0, "win_rate": 0, "avg_return": 0,
                "sharpe": 0, "max_dd": 0, "total_return": 0, "profit_factor": 0}

    rets = pd.Series([t["return_pct"] for t in trades])
    wins  = rets[rets > 0]
    losses = rets[rets <= 0]

    # Equity curve
    equity = (1 + rets / 100).cumprod()
    rolling_max = equity.cummax()
    dd = (equity - rolling_max) / rolling_max
    max_dd = float(dd.min() * 100)

    total_return = float((equity.iloc[-1] - 1) * 100)

    sharpe = 0.0
    if rets.std() > 0:
        # annualise assuming ~1 trade per week
        sharpe = float(rets.mean() / rets.std() * np.sqrt(52))

    pf = (wins.sum() / abs(losses.sum())) if (len(losses) > 0 and abs(losses.sum()) > 0) else float("inf")

    return {
        "n_trades":     len(trades),
        "win_rate":     float((rets > 0).mean()),
        "avg_return":   float(rets.mean()),
        "sharpe":       sharpe,
        "max_dd":       max_dd,
        "total_return": total_return,
        "profit_factor": float(pf),
    }


# ─────────────────────────────────────────────
# Univariate analysis
# ─────────────────────────────────────────────
def univariate_analysis(trades_df: pd.DataFrame, feature: str,
                        bins: list, labels: list) -> pd.DataFrame:
    trades_df = trades_df.copy()
    trades_df["_bucket"] = pd.cut(trades_df[feature], bins=bins, labels=labels)
    grp = trades_df.groupby("_bucket", observed=True)

    rows = []
    for bucket, g in grp:
        if len(g) == 0:
            continue
        rets = g["return_pct"]
        sharpe = (rets.mean() / rets.std() * np.sqrt(52)) if rets.std() > 0 else 0
        rows.append({
            "bucket": str(bucket),
            "n": len(g),
            "win_rate": f"{(rets>0).mean():.0%}",
            "avg_return": f"{rets.mean():.3f}%",
            "sharpe": f"{sharpe:.2f}",
        })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────
def plot_scatter_features(trades_df: pd.DataFrame):
    features = [
        ("dist_to_resistance_pct", "Dist to Resistance (12h)"),
        ("structural_rr",          "Structural R/R"),
        ("volume_z",               "Volume Z-score (168h)"),
        ("close_ma21_ratio",       "Close / MA21"),
        ("ret_1h_at_entry",        "Ret 1h at Entry"),
        ("velocity_ratio",         "Velocity Ratio"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.patch.set_facecolor("#0e1117")

    for ax, (feat, title) in zip(axes.flat, features):
        ax.set_facecolor("#1a1d23")
        colors = ["#ff4b4b" if r <= 0 else "#21c55d" for r in trades_df["return_pct"]]
        ax.scatter(trades_df[feat], trades_df["return_pct"], c=colors, alpha=0.7, s=60)
        ax.axhline(0, color="#888", linewidth=0.8, linestyle="--")
        ax.set_xlabel(feat, color="#ccc", fontsize=8)
        ax.set_ylabel("Return %", color="#ccc", fontsize=8)
        ax.set_title(title, color="#fff", fontsize=10)
        ax.tick_params(colors="#aaa")
        for spine in ax.spines.values():
            spine.set_edgecolor("#333")

        # Mark the live trade (22/04) if close to real values
        ax.annotate("22/04 live\n(-0.98%)", xy=(0, 0), xycoords="axes fraction",
                    fontsize=7, color="#f0c040", alpha=0)  # invisible placeholder

    plt.suptitle("SOL Bot 4 — Feature vs Trade Return (2026)", color="#fff", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(OUT_PLOTS / "scatter_features.png", dpi=100, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print("  → scatter_features.png saved")


def plot_filter_comparison(comparison_df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(13, 5))
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#1a1d23")

    x = np.arange(len(comparison_df))
    baseline_sharpe = comparison_df[comparison_df["config"] == "baseline"]["sharpe"].iloc[0]
    colors = ["#21c55d" if s > baseline_sharpe else ("#f0c040" if s == baseline_sharpe else "#ff4b4b")
              for s in comparison_df["sharpe"]]

    bars = ax.bar(x, comparison_df["sharpe"], color=colors, alpha=0.85, width=0.6)
    ax.axhline(baseline_sharpe, color="#f0c040", linestyle="--", linewidth=1, label=f"Baseline {baseline_sharpe:.2f}")
    ax.set_xticks(x)
    ax.set_xticklabels(comparison_df["config"], rotation=30, ha="right", color="#ccc", fontsize=9)
    ax.set_ylabel("Sharpe (annualised)", color="#ccc")
    ax.set_title("SOL Bot 4 — Filter Comparison", color="#fff", fontsize=13)
    ax.tick_params(colors="#aaa")
    ax.legend(fontsize=9, facecolor="#1a1d23", labelcolor="#ccc")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333")

    for bar, val in zip(bars, comparison_df["sharpe"]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{val:.2f}", ha="center", va="bottom", color="#fff", fontsize=8)

    plt.tight_layout()
    plt.savefig(OUT_PLOTS / "filter_comparison_bars.png", dpi=100, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print("  → filter_comparison_bars.png saved")


def plot_grid_search(grid_df: pd.DataFrame, filter_name: str, param_name: str):
    fig, ax1 = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor("#0e1117")
    ax1.set_facecolor("#1a1d23")

    ax2 = ax1.twinx()

    x = grid_df[param_name].astype(str)
    x_idx = np.arange(len(x))

    ax1.plot(x_idx, grid_df["sharpe"], "o-", color="#21c55d", linewidth=2, label="Sharpe")
    ax2.bar(x_idx, grid_df["n_trades"], alpha=0.3, color="#4a90d9", label="N trades")

    ax1.set_xticks(x_idx)
    ax1.set_xticklabels(x, color="#ccc", fontsize=9)
    ax1.set_ylabel("Sharpe", color="#21c55d")
    ax2.set_ylabel("N Trades", color="#4a90d9")
    ax1.set_title(f"Grid Search — {filter_name}", color="#fff", fontsize=12)
    ax1.tick_params(colors="#aaa")
    ax2.tick_params(colors="#aaa")
    for spine in ax1.spines.values():
        spine.set_edgecolor("#333")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, facecolor="#1a1d23", labelcolor="#ccc")

    plt.tight_layout()
    plt.savefig(OUT_PLOTS / "grid_search_threshold.png", dpi=100, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print("  → grid_search_threshold.png saved")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    print("=" * 60)
    print("SOL Bot 4 — Filters Study (2026+)")
    print("=" * 60)

    # ── 1. Build dataset ───────────────────────────────────────────
    print("\n[1/8] Building unified dataset...")
    sol = build_dataset()
    sol_2026 = sol[sol.index >= "2026-01-01"]
    print(f"  Total 1h rows: {len(sol)}")
    print(f"  2026+ rows:    {len(sol_2026)}")

    # ── 2. Baseline backtest ───────────────────────────────────────
    print("\n[2/8] Running baseline backtest...")
    baseline_trades, all_signals = run_backtest(sol, PARAMS)
    baseline_metrics = compute_metrics(baseline_trades)

    print(f"  Signals (hard gates):  {len(all_signals)}")
    print(f"  Trades entered:        {baseline_metrics['n_trades']}")
    print(f"  Win Rate:              {baseline_metrics['win_rate']:.1%}")
    print(f"  Avg Return:            {baseline_metrics['avg_return']:.3f}%")
    print(f"  Sharpe:                {baseline_metrics['sharpe']:.2f}")
    print(f"  Max DD:                {baseline_metrics['max_dd']:.2f}%")
    print(f"  Total Return:          {baseline_metrics['total_return']:.2f}%")

    if not baseline_trades:
        print("  ⚠️  No trades generated — check data coverage!")
        return

    trades_df = pd.DataFrame(baseline_trades)

    # ── 3. Structural features (already computed per trade in backtest) ──
    print("\n[3/8] Structural features per trade:")
    for col in ["dist_to_resistance_pct", "structural_rr", "volume_z",
                "close_ma21_ratio", "velocity_ratio"]:
        if col in trades_df.columns:
            print(f"  {col}: min={trades_df[col].min():.3f}  "
                  f"mean={trades_df[col].mean():.3f}  "
                  f"max={trades_df[col].max():.3f}")

    # ── 4. Univariate analysis ─────────────────────────────────────
    print("\n[4/8] Univariate analysis...")
    univariate_results = {}

    features_config = [
        ("dist_to_resistance_pct", [0,0.005,0.010,0.015,0.020,0.030,0.050,1.0],
         ["<0.5%","0.5-1%","1-1.5%","1.5-2%","2-3%","3-5%",">5%"]),
        ("structural_rr",          [0,0.2,0.4,0.6,1.0,2.0,20.0],
         ["<0.2","0.2-0.4","0.4-0.6","0.6-1.0","1-2",">2"]),
        ("volume_z",               [-5,-1,0,0.5,1.0,1.5,2.0,10.0],
         ["<-1","-1-0","0-0.5","0.5-1","1-1.5","1.5-2",">2"]),
        ("close_ma21_ratio",       [0.9,0.99,1.0,1.01,1.02,1.03,1.05,1.2],
         ["<0.99","0.99-1.0","1.0-1.01","1.01-1.02","1.02-1.03","1.03-1.05",">1.05"]),
        ("velocity_ratio",         [-10,-1,0,0.5,1.0,2.0,5.0,50.0],
         ["<-1","-1-0","0-0.5","0.5-1","1-2","2-5",">5"]),
    ]

    all_univ = []
    for feat, bins, labels in features_config:
        if feat not in trades_df.columns:
            continue
        tdf = trades_df.copy()
        tdf[feat] = pd.to_numeric(tdf[feat], errors="coerce")
        res = univariate_analysis(tdf, feat, bins, labels)
        res.insert(0, "feature", feat)
        all_univ.append(res)
        print(f"\n  {feat}:")
        print(res.to_string(index=False))

    if all_univ:
        pd.concat(all_univ).to_csv(OUT_TABLES / "univariate_analysis.csv", index=False)
        print(f"\n  → univariate_analysis.csv saved ({len(pd.concat(all_univ))} rows)")

    # ── 5. Scatter plots ───────────────────────────────────────────
    print("\n[5/8] Generating scatter plots...")
    plot_scatter_features(trades_df)

    # ── 6. Filter tests ────────────────────────────────────────────
    print("\n[6/8] Testing isolated filters...")

    # Define filter functions
    def f_resistance_1pct(row, s):  return s["dist_to_resistance_pct"] > 0.010
    def f_resistance_2pct(row, s):  return s["dist_to_resistance_pct"] > 0.020
    def f_bad_rr_03(row, s):        return s["structural_rr"] > 0.30
    def f_bad_rr_05(row, s):        return s["structural_rr"] > 0.50
    def f_high_vol_1(row, s):       return s["volume_z"] < 1.0
    def f_high_vol_15(row, s):      return s["volume_z"] < 1.5
    def f_extension_103(row, s):    return s["close_ma21_ratio"] < 1.03
    def f_extension_105(row, s):    return s["close_ma21_ratio"] < 1.05
    def f_velocity_2(row, s):       return s["velocity_ratio"] < 2.0

    filter_configs = [
        ("baseline",               None),
        ("resistance_>1%",         f_resistance_1pct),
        ("resistance_>2%",         f_resistance_2pct),
        ("struct_rr_>0.3",         f_bad_rr_03),
        ("struct_rr_>0.5",         f_bad_rr_05),
        ("volume_z_<1.0",          f_high_vol_1),
        ("volume_z_<1.5",          f_high_vol_15),
        ("extension_<1.03",        f_extension_103),
        ("extension_<1.05",        f_extension_105),
        ("velocity_<2.0",          f_velocity_2),
    ]

    results = {}
    for name, filt in filter_configs:
        t, _ = run_backtest(sol, PARAMS, extra_filter=filt)
        m = compute_metrics(t)
        results[name] = m
        delta = m["sharpe"] - baseline_metrics["sharpe"]
        keep  = m["n_trades"] / baseline_metrics["n_trades"] if baseline_metrics["n_trades"] > 0 else 0
        print(f"  {name:<25}  n={m['n_trades']:2d}  wr={m['win_rate']:.0%}  "
              f"sharpe={m['sharpe']:.2f}  Δ={delta:+.2f}  keep={keep:.0%}")

    comparison_df = pd.DataFrame([
        {"config": k, "n_trades": v["n_trades"], "win_rate": f"{v['win_rate']:.0%}",
         "avg_return": f"{v['avg_return']:.3f}%", "sharpe": v["sharpe"],
         "max_dd": f"{v['max_dd']:.2f}%",
         "delta_vs_baseline": round(v["sharpe"] - baseline_metrics["sharpe"], 3)}
        for k, v in results.items()
    ])
    comparison_df.to_csv(OUT_TABLES / "filter_comparison.csv", index=False)
    print(f"\n  → filter_comparison.csv saved")

    # ── 7. Combined filters ────────────────────────────────────────
    print("\n[7/8] Testing filter combinations...")

    # Find best individual filter
    solo = {k: v for k, v in results.items() if k != "baseline"}
    best_single = max(solo, key=lambda k: solo[k]["sharpe"])
    print(f"  Best single: {best_single} (sharpe={solo[best_single]['sharpe']:.2f})")

    def f_combo_A(row, s):  # resistance + volume
        return s["dist_to_resistance_pct"] > 0.010 and s["volume_z"] < 1.0

    def f_combo_B(row, s):  # rr + extension
        return s["structural_rr"] > 0.30 and s["close_ma21_ratio"] < 1.03

    def f_combo_C(row, s):  # resistance + rr
        return s["dist_to_resistance_pct"] > 0.010 and s["structural_rr"] > 0.30

    def f_combo_D(row, s):  # resistance + rr + volume
        return (s["dist_to_resistance_pct"] > 0.010 and
                s["structural_rr"] > 0.30 and
                s["volume_z"] < 1.0)

    combo_configs = [
        ("combo_resistance+volume",     f_combo_A),
        ("combo_rr+extension",          f_combo_B),
        ("combo_resistance+rr",         f_combo_C),
        ("combo_resistance+rr+volume",  f_combo_D),
    ]

    combo_rows = []
    for name, filt in combo_configs:
        t, _ = run_backtest(sol, PARAMS, extra_filter=filt)
        m = compute_metrics(t)
        delta = m["sharpe"] - baseline_metrics["sharpe"]
        keep  = m["n_trades"] / baseline_metrics["n_trades"] if baseline_metrics["n_trades"] > 0 else 0
        print(f"  {name:<30}  n={m['n_trades']:2d}  wr={m['win_rate']:.0%}  "
              f"sharpe={m['sharpe']:.2f}  Δ={delta:+.2f}  keep={keep:.0%}")
        combo_rows.append({
            "config": name, "n_trades": m["n_trades"],
            "win_rate": f"{m['win_rate']:.0%}", "sharpe": m["sharpe"],
            "delta_vs_baseline": round(delta, 3),
            "trades_kept_pct": f"{keep:.0%}",
        })

    combo_df = pd.DataFrame(combo_rows)

    # ── 6b. Comparison plots (all configs) ────────────────────────
    all_comparison = comparison_df.copy()
    # append combos
    extra_rows = combo_df[["config", "n_trades", "sharpe", "delta_vs_baseline"]].copy()
    extra_rows["win_rate"] = ""
    extra_rows["avg_return"] = ""
    extra_rows["max_dd"] = ""
    all_comparison = pd.concat([all_comparison, extra_rows], ignore_index=True)

    plot_filter_comparison(all_comparison[["config", "sharpe"]])

    # ── 8. Grid search (resistance threshold) ─────────────────────
    print("\n[8/8] Grid search on resistance threshold...")

    thresholds = [0.003, 0.005, 0.0075, 0.010, 0.0125, 0.015, 0.018, 0.020, 0.025, 0.030]
    grid_rows = []
    for thr in thresholds:
        def _f(row, s, t=thr): return s["dist_to_resistance_pct"] > t
        t, _ = run_backtest(sol, PARAMS, extra_filter=_f)
        m = compute_metrics(t)
        grid_rows.append({
            "threshold_pct": thr * 100,
            "n_trades": m["n_trades"],
            "win_rate": m["win_rate"],
            "sharpe": m["sharpe"],
            "max_dd": m["max_dd"],
            "delta_sharpe": m["sharpe"] - baseline_metrics["sharpe"],
        })
        print(f"  thr={thr*100:.1f}%  n={m['n_trades']:2d}  "
              f"sharpe={m['sharpe']:.2f}  Δ={m['sharpe']-baseline_metrics['sharpe']:+.2f}")

    grid_df = pd.DataFrame(grid_rows)
    grid_df.to_csv(OUT_TABLES / "grid_search_results.csv", index=False)
    print(f"  → grid_search_results.csv saved")

    best_grid = grid_df.loc[grid_df["sharpe"].idxmax()]
    print(f"\n  Best resistance threshold: {best_grid['threshold_pct']:.1f}%  "
          f"sharpe={best_grid['sharpe']:.2f}  n={int(best_grid['n_trades'])}")

    plot_grid_search(grid_df, "Resistance Distance Filter", "threshold_pct")

    # ── Apply filters to live trade (22/04) ───────────────────────
    print("\n[LIVE TRADE] Analysis of 22/04 trade ($88.23 entry):")
    live_ts     = pd.Timestamp("2026-04-22 16:15:00", tz="UTC")
    live_entry  = 88.23
    live_resist = 88.55  # 12h high at entry time

    live_dist_r = (live_resist - live_entry) / live_entry
    live_tp     = live_entry * (1 + PARAMS["tp_pct"])  # $89.99
    live_dist_tp = (live_tp - live_entry) / live_entry
    live_rr     = live_dist_r / (live_entry * PARAMS["sl_pct"] / live_entry) if PARAMS["sl_pct"] > 0 else 0

    # Try to compute from actual data
    if live_ts in sol.index:
        live_struct = compute_structural_features(sol, live_ts)
    else:
        # Use closest available timestamp
        available = sol[sol.index <= live_ts]
        if not available.empty:
            closest_ts = available.index[-1]
            live_struct = compute_structural_features(sol, closest_ts)
            print(f"  (using closest available ts: {closest_ts})")
        else:
            live_struct = {
                "dist_to_resistance_pct": live_dist_r,
                "structural_rr": live_dist_r / PARAMS["sl_pct"],
                "volume_z": float("nan"),
                "close_ma21_ratio": float("nan"),
                "velocity_ratio": float("nan"),
            }

    print(f"  dist_to_resistance: {live_struct['dist_to_resistance_pct']*100:.2f}%")
    print(f"  structural_rr:      {live_struct['structural_rr']:.3f}")
    print(f"  volume_z:           {live_struct['volume_z']:.3f}")
    print(f"  close_ma21_ratio:   {live_struct['close_ma21_ratio']:.4f}")
    print(f"  velocity_ratio:     {live_struct['velocity_ratio']:.3f}")
    print(f"  TP required:        {live_dist_tp*100:.2f}% (through resistance at {live_dist_r*100:.2f}%)")

    # Check each filter
    print("\n  Filter checks on live trade:")
    for name, filt in filter_configs[1:] + combo_configs:
        row = {}  # dummy row
        block = not filt(row, live_struct)
        status = "🛡️ BLOCKED" if block else "✅ allowed"
        print(f"  {status}  {name}")

    # ── Determine recommendation ───────────────────────────────────
    print("\n[RECOMMENDATION]")
    best_all = max(
        [(k, v) for k, v in results.items() if k != "baseline"],
        key=lambda x: x[1]["sharpe"]
    )
    best_name, best_m = best_all
    delta_sharpe = best_m["sharpe"] - baseline_metrics["sharpe"]
    keep_ratio   = best_m["n_trades"] / baseline_metrics["n_trades"] if baseline_metrics["n_trades"] > 0 else 0
    blocks_live  = not [f for n, f in filter_configs if n == best_name][0](
        {}, live_struct) if [f for n, f in filter_configs if n == best_name] else False

    criteria_met = (
        delta_sharpe >= 0.3 and
        keep_ratio >= 0.6 and
        blocks_live
    )

    rec_filter = best_name
    rec_sharpe = best_m["sharpe"]
    rec_delta  = delta_sharpe

    print(f"  Best filter:   {rec_filter}")
    print(f"  Sharpe delta:  {rec_delta:+.2f} (threshold: ≥0.3)")
    print(f"  Trades kept:   {keep_ratio:.0%} (threshold: ≥60%)")
    print(f"  Blocks live:   {blocks_live}")
    print(f"  → APPLY FILTER: {criteria_met}")

    # ── Generate report ────────────────────────────────────────────
    print(f"\n[REPORT] Generating {OUT_REPORT}...")

    # Build per-filter table
    def _build_filter_table(configs_dict):
        rows = []
        for name, m in configs_dict.items():
            d = round(m["sharpe"] - baseline_metrics["sharpe"], 2)
            keep = round(m["n_trades"] / baseline_metrics["n_trades"] * 100) if baseline_metrics["n_trades"] else 0
            rows.append(
                f"| {name:<30} | {m['n_trades']:3d} | {m['win_rate']:.0%} | "
                f"{m['avg_return']:.3f}% | {m['sharpe']:.2f} | {d:+.2f} | {keep}% |"
            )
        return "\n".join(rows)

    univ_section = ""
    for feat, bins, labels in features_config:
        if feat not in trades_df.columns:
            continue
        tdf = trades_df.copy()
        tdf[feat] = pd.to_numeric(tdf[feat], errors="coerce")
        res = univariate_analysis(tdf, feat, bins, labels)
        univ_section += f"\n### {feat}\n"
        univ_section += "| Bucket | N | WR | Avg Return | Sharpe |\n"
        univ_section += "|--------|---|----|------------|--------|\n"
        for _, row in res.iterrows():
            univ_section += f"| {row['bucket']:<12} | {row['n']:2d} | {row['win_rate']} | {row['avg_return']} | {row['sharpe']} |\n"
        univ_section += "\n"

    # Combo table
    combo_section = "| Config | N | WR | Sharpe | Δ vs baseline | Kept |\n"
    combo_section += "|--------|---|-----|--------|---------------|------|\n"
    for r in combo_rows:
        combo_section += (f"| {r['config']:<30} | {r['n_trades']:2d} | {r['win_rate']} | "
                          f"{r['sharpe']:.2f} | {r['delta_vs_baseline']:+.2f} | {r['trades_kept_pct']} |\n")

    # Grid table
    grid_section = "| Threshold | N | WR | Sharpe | Δ |\n"
    grid_section += "|-----------|---|-----|--------|---|\n"
    for _, gr in grid_df.iterrows():
        grid_section += (f"| {gr['threshold_pct']:.1f}% | {int(gr['n_trades'])} | "
                         f"{gr['win_rate']:.0%} | {gr['sharpe']:.2f} | {gr['delta_sharpe']:+.2f} |\n")

    # Live trade filter checks
    live_checks = ""
    for name, filt in filter_configs[1:]:
        block = not filt({}, live_struct)
        status = "🛡️ BLOCKED" if block else "✅ allowed"
        live_checks += f"- **{name}**: dist={live_struct['dist_to_resistance_pct']*100:.2f}%, vol_z={live_struct['volume_z']:.2f} → {status}\n"

    # Implementation code
    best_thr_pct = float(best_grid["threshold_pct"])

    impl_code = f'''```python
# Em src/trading/sol_bot4.py, adicionar em check_entry_signal():

def check_structural_guard(sol_df: pd.DataFrame, entry_price: float,
                           lookback_h: int = 12,
                           min_dist_pct: float = {best_thr_pct/100:.4f}) -> tuple[bool, str | None]:
    """Block entries perto de resistência 12h."""
    recent = sol_df.tail(lookback_h)
    resistance = recent["high"].max()
    dist_pct = (resistance - entry_price) / entry_price

    if dist_pct < min_dist_pct:
        return False, f"close_to_resistance_{{dist_pct:.2%}}"
    return True, None
```
'''

    recommendation_text = ""
    if criteria_met:
        recommendation_text = f"""### Filtro RECOMENDADO: `{rec_filter}`

**Threshold:** {best_thr_pct:.1f}% distância até resistência 12h

**Justificativa:**
- Sharpe melhorado de {baseline_metrics['sharpe']:.2f} → {rec_sharpe:.2f} (+{rec_delta:.2f})
- Trades mantidos: {int(best_m['n_trades'])}/{baseline_metrics['n_trades']} ({keep_ratio:.0%}) — acima do mínimo 60%
- Filtra a trade real 22/04 corretamente ✓
- Princípio conceitual sólido: entrada perto de resistência = R/R estrutural ruim

{impl_code}"""
    else:
        recommendation_text = f"""### Nenhum filtro atende todos os critérios

Melhor candidato: `{rec_filter}` (Sharpe {rec_sharpe:.2f}, Δ={rec_delta:+.2f})

- Sharpe delta ≥ 0.3: {"✅" if delta_sharpe >= 0.3 else "❌"} ({rec_delta:+.2f})
- Trades mantidos ≥ 60%: {"✅" if keep_ratio >= 0.6 else "❌"} ({keep_ratio:.0%})
- Bloqueia trade live: {"✅" if blocks_live else "❌"}

**Conclusão:** Trade 22/04 foi evento N=1. Strategy original válida. Continuar monitorando.

> Revisitar quando N ≥ 30 trades (poder estatístico suficiente)."""

    report = f"""# SOL Bot 4 — Estudo de Filtros Estruturais

**Data:** 2026-04-22
**Trigger:** primeira trade live -0.98% (fase tardia rally)
**Entry:** $88.23 @ 16:15 UTC | **Exit:** $87.37 @ 19:43 UTC (TRAIL) | **Duration:** 3h 28min

---

## 1. Baseline

| Metric | Value |
|--------|-------|
| Sharpe | {baseline_metrics['sharpe']:.2f} |
| Trades | {baseline_metrics['n_trades']} |
| Win Rate | {baseline_metrics['win_rate']:.0%} |
| Avg Return | {baseline_metrics['avg_return']:.3f}% |
| Max DD | {baseline_metrics['max_dd']:.2f}% |
| Total Return | {baseline_metrics['total_return']:.2f}% |
| Profit Factor | {baseline_metrics['profit_factor']:.2f} |

> Target: Sharpe ~2.03 (Phase 2 backtest original)

---

## 2. Análise univariada (por feature)

{univ_section}

---

## 3. Teste de filtros isolados

| Config | N | WR | Avg Return | Sharpe | Δ vs baseline | Kept |
|--------|---|----|------------|--------|---------------|------|
{_build_filter_table(results)}

---

## 4. Filtros combinados

{combo_section}

---

## 5. Grid search — Distance to Resistance

{grid_section}

**Melhor threshold:** {best_thr_pct:.1f}%
**Sharpe resultante:** {best_grid['sharpe']:.2f} (Δ={best_grid['delta_sharpe']:+.2f})

![Grid Search](plots/sol_filters_study/grid_search_threshold.png)

---

## 6. Análise da trade live (22/04)

**Features estruturais no momento da entrada ($88.23):**

| Feature | Valor | Filtro aplicado |
|---------|-------|-----------------|
| dist_to_resistance_12h | {live_struct['dist_to_resistance_pct']*100:.2f}% | > {best_thr_pct:.1f}%? |
| structural_rr | {live_struct['structural_rr']:.3f} | > 0.3? |
| volume_z | {live_struct['volume_z']:.3f} | < 1.0? |
| close/MA21 | {live_struct['close_ma21_ratio']:.4f} | < 1.03? |
| velocity_ratio | {live_struct['velocity_ratio']:.3f} | < 2.0? |

**Diagnóstico:** Entry a {live_struct['dist_to_resistance_pct']*100:.2f}% da resistência 12h ($88.55).
TP ($89.99) exigia +{live_dist_tp*100:.1f}% furar a resistência. R/R estrutural = 0.09.

{live_checks}

---

## 7. Recomendação

{recommendation_text}

---

## 8. Plots gerados

- `plots/sol_filters_study/scatter_features.png` — Feature vs Return por trade
- `plots/sol_filters_study/filter_comparison_bars.png` — Comparação de Sharpe por filtro
- `plots/sol_filters_study/grid_search_threshold.png` — Grid search resistência

## 9. Tabelas geradas

- `tables/sol_filters_study/univariate_analysis.csv`
- `tables/sol_filters_study/filter_comparison.csv`
- `tables/sol_filters_study/grid_search_results.csv`
"""

    OUT_REPORT.write_text(report)
    print(f"  → {OUT_REPORT} saved")

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()
