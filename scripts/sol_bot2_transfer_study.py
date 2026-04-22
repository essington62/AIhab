"""
SOL Bot 5 Study — Bot 2 BTC Strategy Transfer (2026 ONLY)

Pergunta: a strategy de Bot 2 BTC (momentum + stablecoin fuel) transfere para SOL?

Regras:
  - Dados 2026+ APENAS
  - Strategy IDÊNTICA ao Bot 2 BTC (sem tuning)
  - Stops idênticos: SL 1.5%, TP 2%, Trail 1%, max 120h
  - Sem Dynamic TP (testar baseline primeiro)
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

DATA_SOL      = ROOT / "data/01_raw/spot/sol_1h.parquet"
DATA_BTC      = ROOT / "data/02_intermediate/spot/btc_1h_clean.parquet"
DATA_ZSCORES  = ROOT / "data/02_features/gate_zscores.parquet"

OUT_PLOTS  = ROOT / "prompts/plots/sol_bot2_transfer"
OUT_TABLES = ROOT / "prompts/tables/sol_bot2_transfer"
OUT_REPORT = ROOT / "prompts/sol_bot2_transfer_report.md"

OUT_PLOTS.mkdir(parents=True, exist_ok=True)
OUT_TABLES.mkdir(parents=True, exist_ok=True)

# ── Bot 2 BTC stops (idênticos, sem Dynamic TP) ──────────────────
STOPS = {
    "sl_pct":        0.015,
    "tp_pct":        0.020,
    "trail_pct":     0.010,
    "max_hold_hours": 120,
    "cooldown_hours":  4,
}

# ── Bot 2 BTC filter params (copy-paste) ─────────────────────────
FILTERS = {
    "stablecoin_z_min": 1.3,
    "ret_1d_min":       0.0,
    "rsi_min":         60.0,
    "rsi_max":         80.0,  # nota: prompt usa RSI < 80 implicitamente
    "bb_pct_max":       0.98,
    "spike_ret_max":    0.03,
    "spike_rsi_max":   65.0,
}

DARK_BG  = "#0e1117"
CARD_BG  = "#1a1d23"
GREEN    = "#21c55d"
RED      = "#ff4b4b"
BLUE     = "#4a90d9"
YELLOW   = "#f0c040"
GREY     = "#cccccc"


# ─────────────────────────────────────────────
# Feature computation
# ─────────────────────────────────────────────
def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add RSI, BB, MA21, ret_1d to any OHLCV dataframe."""
    df = df.copy()
    close = df["close"]

    # RSI 14
    df["rsi"] = _rsi(close, 14)

    # Bollinger Bands 20 (same as Bot 2 BTC uses bb_window=20)
    ma20  = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    bb_upper = ma20 + 2 * std20
    bb_lower = ma20 - 2 * std20
    df["bb_pct"] = ((close - bb_lower) / (bb_upper - bb_lower)).clip(0, 1)

    # MA21
    df["ma21"] = close.rolling(21).mean()

    # 24h return
    df["ret_1d"] = close.pct_change(24)

    return df


# ─────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────
def load_stablecoin_z() -> pd.Series:
    gz = pd.read_parquet(DATA_ZSCORES)
    gz["timestamp"] = pd.to_datetime(gz["timestamp"], utc=True)
    gz = gz.sort_values("timestamp").drop_duplicates("timestamp")
    gz = gz.set_index("timestamp")
    return gz["stablecoin_z"]


def load_sol_2026() -> pd.DataFrame:
    df = pd.read_parquet(DATA_SOL)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").drop_duplicates("timestamp").set_index("timestamp")
    df = compute_features(df)
    return df[df.index >= "2026-01-01"]


def load_btc_2026() -> pd.DataFrame:
    df = pd.read_parquet(DATA_BTC)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").drop_duplicates("timestamp").set_index("timestamp")
    # BTC clean has rsi_14 and ma_21 pre-computed; add bb_pct alias and ret_1d
    if "rsi_14" in df.columns and "rsi" not in df.columns:
        df["rsi"] = df["rsi_14"]
    if "ma_21" in df.columns and "ma21" not in df.columns:
        df["ma21"] = df["ma_21"]
    # Recompute bb_pct from scratch for consistency with SOL
    close  = df["close"].astype(float)
    ma20   = close.rolling(20).mean()
    std20  = close.rolling(20).std()
    bb_upper = ma20 + 2 * std20
    bb_lower = ma20 - 2 * std20
    df["bb_pct"] = ((close - bb_lower) / (bb_upper - bb_lower)).clip(0, 1)
    df["ret_1d"] = close.pct_change(24)
    return df[df.index >= "2026-01-01"]


def attach_stablecoin_z(df: pd.DataFrame, sz: pd.Series) -> pd.DataFrame:
    df = df.copy()
    df["stablecoin_z"] = sz.reindex(df.index, method="ffill")
    return df


# ─────────────────────────────────────────────
# Bot 2 filter check (exact copy)
# ─────────────────────────────────────────────
def check_bot2_entry(row, f=FILTERS) -> tuple[bool, str | None]:
    sz    = row.get("stablecoin_z")
    ret1d = row.get("ret_1d")
    rsi   = row.get("rsi")
    bb    = row.get("bb_pct")
    close = row.get("close")
    ma21  = row.get("ma21")

    # Missing data guard
    if any(v is None or (isinstance(v, float) and np.isnan(v))
           for v in [sz, ret1d, rsi, bb, close, ma21]):
        return False, "missing_data"

    if sz <= f["stablecoin_z_min"]:
        return False, f"stablecoin_low_{sz:.2f}"
    if ret1d <= f["ret_1d_min"]:
        return False, "ret_1d_negative"
    if rsi < f["rsi_min"]:
        return False, f"rsi_low_{rsi:.1f}"
    if rsi > f["rsi_max"]:
        return False, f"rsi_high_{rsi:.1f}"
    if bb >= f["bb_pct_max"]:
        return False, f"bb_top_{bb:.3f}"
    if close <= ma21:
        return False, "below_ma21"
    # Spike guard
    if ret1d > f["spike_ret_max"] and rsi > f["spike_rsi_max"]:
        return False, f"spike_guard_ret{ret1d:.3f}_rsi{rsi:.1f}"

    return True, None


# ─────────────────────────────────────────────
# Backtest engine
# ─────────────────────────────────────────────
def run_backtest(df: pd.DataFrame, stops: dict = STOPS) -> list[dict]:
    sl_pct     = stops["sl_pct"]
    tp_pct     = stops["tp_pct"]
    trail_pct  = stops["trail_pct"]
    max_hold   = stops["max_hold_hours"]
    cooldown_h = stops["cooldown_hours"]

    in_trade = False
    entry_ts = entry_price = trail_high = sl_price = tp_price = max_hold_ts = None
    cooldown_until = None
    entry_feats = {}
    trades = []

    for ts, row in df.iterrows():
        # ── Exit ────────────────────────────────────────────────
        if in_trade:
            h = row["high"] if "high" in row else row["close"]
            l = row["low"]  if "low"  in row else row["close"]
            c = row["close"]

            if c > trail_high:
                trail_high = c
            trail_sl  = trail_high * (1 - trail_pct)
            eff_sl    = max(sl_price, trail_sl)

            exit_r = exit_p = None
            if h >= tp_price:
                exit_r, exit_p = "TP", tp_price
            elif l <= eff_sl:
                exit_r = "TRAIL" if eff_sl > sl_price else "SL"
                exit_p = eff_sl
            elif ts >= max_hold_ts:
                exit_r, exit_p = "TIMEOUT", c

            if exit_r:
                pnl = (exit_p - entry_price) / entry_price
                hold_h = (ts - entry_ts).total_seconds() / 3600
                trades.append({
                    "entry_ts":    entry_ts,
                    "exit_ts":     ts,
                    "entry_price": entry_price,
                    "exit_price":  exit_p,
                    "exit_reason": exit_r,
                    "return_pct":  pnl * 100,
                    "hold_hours":  hold_h,
                    **entry_feats,
                })
                in_trade = False
                cooldown_until = ts + pd.Timedelta(hours=cooldown_h)
            continue

        # ── Entry ────────────────────────────────────────────────
        if cooldown_until and ts < cooldown_until:
            continue

        passed, reason = check_bot2_entry(row)
        if not passed:
            continue

        in_trade      = True
        entry_ts      = ts
        entry_price   = float(row["close"])
        trail_high    = entry_price
        sl_price      = entry_price * (1 - sl_pct)
        tp_price      = entry_price * (1 + tp_pct)
        max_hold_ts   = ts + pd.Timedelta(hours=max_hold)
        entry_feats   = {
            "rsi":          float(row.get("rsi", np.nan)),
            "bb_pct":       float(row.get("bb_pct", np.nan)),
            "ret_1d":       float(row.get("ret_1d", np.nan)),
            "stablecoin_z": float(row.get("stablecoin_z", np.nan)),
            "close_ma21":   float(row["close"] / row["ma21"]) if row.get("ma21") else np.nan,
        }

    return trades


# ─────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────
def compute_metrics(trades: list[dict]) -> dict:
    if not trades:
        return {"n_trades": 0, "sharpe": 0.0, "win_rate": 0.0,
                "avg_return": 0.0, "total_return": 0.0, "max_dd": 0.0,
                "profit_factor": 0.0}
    rets  = pd.Series([t["return_pct"] for t in trades])
    wins  = rets[rets > 0]
    losses = rets[rets <= 0]

    eq    = (1 + rets / 100).cumprod()
    dd    = ((eq - eq.cummax()) / eq.cummax()).min() * 100
    total = (eq.iloc[-1] - 1) * 100

    # Sharpe annualised (consistent with filter study: sqrt(52))
    sharpe = float(rets.mean() / rets.std() * np.sqrt(52)) if rets.std() > 0 else 0.0
    pf = (wins.sum() / abs(losses.sum())) if (len(losses) > 0 and abs(losses.sum()) > 0) else float("inf")

    return {
        "n_trades":     len(trades),
        "sharpe":       sharpe,
        "win_rate":     float((rets > 0).mean()),
        "avg_return":   float(rets.mean()),
        "total_return": float(total),
        "max_dd":       float(dd),
        "profit_factor": float(pf),
    }


def metrics_subperiod(trades: list[dict], start: str, end: str) -> dict:
    sub = [t for t in trades
           if pd.Timestamp(start, tz="UTC") <= t["entry_ts"] <= pd.Timestamp(end, tz="UTC")]
    return compute_metrics(sub)


# ─────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────
def _ax(w=12, h=5):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(CARD_BG)
    for sp in ax.spines.values():
        sp.set_edgecolor("#333")
    ax.tick_params(colors=GREY)
    return fig, ax


def plot_equity_curves(btc_trades, sol_trades):
    fig, ax = _ax(13, 5)
    for trades, label, color, lw in [
        (btc_trades, "BTC Bot 2 (2026)", BLUE,  2.0),
        (sol_trades, "SOL Bot 2 transfer (2026)", GREEN, 2.0),
    ]:
        if not trades:
            continue
        rets = pd.Series([t["return_pct"] for t in trades])
        eq   = (1 + rets / 100).cumprod()
        ax.plot(range(len(eq)), eq.values, label=label, color=color, linewidth=lw)
    ax.axhline(1.0, color=YELLOW, linewidth=0.6, linestyle="--", alpha=0.6)
    ax.set_xlabel("Trade #", color=GREY)
    ax.set_ylabel("Portfolio (normalised)", color=GREY)
    ax.set_title("Bot 2 Strategy: BTC vs SOL (2026)", color="white", fontsize=13)
    ax.legend(facecolor=CARD_BG, labelcolor=GREY, fontsize=9)
    plt.tight_layout()
    plt.savefig(OUT_PLOTS / "equity_curves.png", dpi=100, facecolor=DARK_BG,
                bbox_inches="tight")
    plt.close()
    print("  → equity_curves.png")


def plot_returns_bars(btc_trades, sol_trades):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor(DARK_BG)
    for ax, trades, label in [(axes[0], btc_trades, "BTC"), (axes[1], sol_trades, "SOL")]:
        ax.set_facecolor(CARD_BG)
        for sp in ax.spines.values():
            sp.set_edgecolor("#333")
        ax.tick_params(colors=GREY)
        if not trades:
            ax.set_title(f"{label} — no trades", color="white")
            continue
        rets   = [t["return_pct"] for t in trades]
        colors = [GREEN if r > 0 else RED for r in rets]
        ax.bar(range(len(rets)), rets, color=colors, alpha=0.85, width=0.7)
        ax.axhline(0, color=YELLOW, linewidth=0.8, linestyle="--")
        m = compute_metrics(trades)
        ax.set_title(f"{label}  N={m['n_trades']}  WR={m['win_rate']:.0%}  "
                     f"Sharpe={m['sharpe']:.2f}", color="white", fontsize=10)
        ax.set_xlabel("Trade #", color=GREY)
        ax.set_ylabel("Return %", color=GREY)
    plt.suptitle("Bot 2 Strategy — Trade Returns Distribution", color="white", fontsize=12)
    plt.tight_layout()
    plt.savefig(OUT_PLOTS / "returns_distribution.png", dpi=100, facecolor=DARK_BG,
                bbox_inches="tight")
    plt.close()
    print("  → returns_distribution.png")


def plot_subperiods(subperiod_rows):
    periods = [r["period"] for r in subperiod_rows]
    btc_sharpe = [r["btc_sharpe"] for r in subperiod_rows]
    sol_sharpe = [r["sol_sharpe"] for r in subperiod_rows]
    x = np.arange(len(periods))
    w = 0.35

    fig, ax = _ax(10, 5)
    bars1 = ax.bar(x - w/2, btc_sharpe, w, label="BTC",    color=BLUE,  alpha=0.85)
    bars2 = ax.bar(x + w/2, sol_sharpe, w, label="SOL",    color=GREEN, alpha=0.85)
    ax.axhline(0,   color="#444", linewidth=0.7, linestyle="--")
    ax.axhline(1.5, color=GREEN,  linewidth=0.6, linestyle="--", alpha=0.4, label="Target 1.5")
    ax.set_xticks(x)
    ax.set_xticklabels(periods, color=GREY, fontsize=9)
    ax.set_ylabel("Sharpe", color=GREY)
    ax.set_title("Bot 2 Strategy — Sub-period Comparison BTC vs SOL", color="white", fontsize=12)
    ax.legend(facecolor=CARD_BG, labelcolor=GREY, fontsize=8)
    for bar in bars1:
        v = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.05, f"{v:.2f}",
                ha="center", va="bottom", color="white", fontsize=8)
    for bar in bars2:
        v = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.05, f"{v:.2f}",
                ha="center", va="bottom", color="white", fontsize=8)
    plt.tight_layout()
    plt.savefig(OUT_PLOTS / "subperiods_comparison.png", dpi=100, facecolor=DARK_BG,
                bbox_inches="tight")
    plt.close()
    print("  → subperiods_comparison.png")


def plot_feature_scatter(trades, asset_label):
    if not trades:
        return
    df_t = pd.DataFrame(trades)
    features = [
        ("stablecoin_z", "Stablecoin Z"),
        ("rsi",          "RSI at entry"),
        ("bb_pct",       "BB% at entry"),
        ("ret_1d",       "ret_1d at entry"),
        ("close_ma21",   "Close/MA21"),
    ]
    fig, axes = plt.subplots(1, 5, figsize=(18, 4))
    fig.patch.set_facecolor(DARK_BG)
    for ax, (feat, title) in zip(axes, features):
        ax.set_facecolor(CARD_BG)
        for sp in ax.spines.values():
            sp.set_edgecolor("#333")
        ax.tick_params(colors=GREY)
        if feat not in df_t.columns:
            continue
        colors = [GREEN if r > 0 else RED for r in df_t["return_pct"]]
        ax.scatter(df_t[feat], df_t["return_pct"], c=colors, alpha=0.7, s=50)
        ax.axhline(0, color="#555", linewidth=0.7, linestyle="--")
        ax.set_xlabel(feat, color=GREY, fontsize=7)
        ax.set_ylabel("Return %" if feat == features[0][0] else "", color=GREY, fontsize=7)
        ax.set_title(title, color="white", fontsize=8)
    plt.suptitle(f"{asset_label} — Entry Features vs Trade Return", color="white", fontsize=11)
    plt.tight_layout()
    fname = f"feature_scatter_{'sol' if 'SOL' in asset_label else 'btc'}.png"
    plt.savefig(OUT_PLOTS / fname, dpi=100, facecolor=DARK_BG, bbox_inches="tight")
    plt.close()
    print(f"  → {fname}")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    print("=" * 65)
    print("SOL Bot 5 — Bot 2 BTC Strategy Transfer (2026 ONLY)")
    print("=" * 65)

    # ── 1. Load data ─────────────────────────────────────────────
    print("\n[1/7] Loading data...")
    stablecoin_z = load_stablecoin_z()
    btc = attach_stablecoin_z(load_btc_2026(), stablecoin_z)
    sol = attach_stablecoin_z(load_sol_2026(), stablecoin_z)

    print(f"  BTC 2026: {len(btc)} rows  ({btc.index.min().date()} → {btc.index.max().date()})")
    print(f"  SOL 2026: {len(sol)} rows  ({sol.index.min().date()} → {sol.index.max().date()})")
    print(f"  stablecoin_z: non-null BTC {btc['stablecoin_z'].notna().sum()}, SOL {sol['stablecoin_z'].notna().sum()}")
    print(f"  stablecoin_z last: {stablecoin_z.dropna().index[-1].date()} ({stablecoin_z.dropna().iloc[-1]:.3f})")
    print(f"  Note: stablecoin_z coverage up to Apr 15 — last week ffilled")

    # ── 2. BTC backtest ───────────────────────────────────────────
    print("\n[2/7] Bot 2 on BTC 2026...")
    btc_trades  = run_backtest(btc)
    btc_metrics = compute_metrics(btc_trades)
    print(f"  Trades: {btc_metrics['n_trades']}")
    print(f"  Sharpe: {btc_metrics['sharpe']:.2f}")
    print(f"  WR:     {btc_metrics['win_rate']:.1%}")
    print(f"  Avg:    {btc_metrics['avg_return']:.3f}%")
    print(f"  MaxDD:  {btc_metrics['max_dd']:.2f}%")
    print(f"  PF:     {btc_metrics['profit_factor']:.2f}")
    print(f"  Total:  {btc_metrics['total_return']:+.2f}%")

    # ── 3. SOL backtest ───────────────────────────────────────────
    print("\n[3/7] Bot 2 on SOL 2026...")
    sol_trades  = run_backtest(sol)
    sol_metrics = compute_metrics(sol_trades)
    print(f"  Trades: {sol_metrics['n_trades']}")
    print(f"  Sharpe: {sol_metrics['sharpe']:.2f}")
    print(f"  WR:     {sol_metrics['win_rate']:.1%}")
    print(f"  Avg:    {sol_metrics['avg_return']:.3f}%")
    print(f"  MaxDD:  {sol_metrics['max_dd']:.2f}%")
    print(f"  PF:     {sol_metrics['profit_factor']:.2f}")
    print(f"  Total:  {sol_metrics['total_return']:+.2f}%")

    # Δ vs BTC
    delta_sharpe = sol_metrics["sharpe"] - btc_metrics["sharpe"]
    print(f"\n  Δ Sharpe SOL vs BTC: {delta_sharpe:+.2f}")

    # ── 4. Sub-period analysis ────────────────────────────────────
    print("\n[4/7] Sub-period analysis...")
    subperiods = [
        ("Jan-Feb 2026", "2026-01-01", "2026-02-28"),
        ("Mar-Abr 2026", "2026-03-01", "2026-04-22"),
    ]
    subperiod_rows = []
    for name, start, end in subperiods:
        mb = metrics_subperiod(btc_trades, start, end)
        ms = metrics_subperiod(sol_trades, start, end)
        print(f"  {name}:")
        print(f"    BTC  N={mb['n_trades']:2d}  Sharpe={mb['sharpe']:.2f}  WR={mb['win_rate']:.0%}")
        print(f"    SOL  N={ms['n_trades']:2d}  Sharpe={ms['sharpe']:.2f}  WR={ms['win_rate']:.0%}")
        subperiod_rows.append({
            "period": name,
            "btc_n": mb["n_trades"], "btc_sharpe": round(mb["sharpe"], 3), "btc_wr": round(mb["win_rate"], 3),
            "sol_n": ms["n_trades"], "sol_sharpe": round(ms["sharpe"], 3), "sol_wr": round(ms["win_rate"], 3),
        })

    n_sol_good = sum(1 for r in subperiod_rows if r["sol_sharpe"] > 1.0)
    print(f"\n  SOL periods with Sharpe > 1.0: {n_sol_good}/{len(subperiods)}")

    # ── 5. Decision ───────────────────────────────────────────────
    print("\n[5/7] Decision criteria (SOL)...")
    c1 = sol_metrics["sharpe"]    > 1.5
    c2 = sol_metrics["n_trades"]  >= 15
    c3 = sol_metrics["win_rate"]  > 0.50
    c4 = abs(sol_metrics["max_dd"]) < 5.0
    c5 = n_sol_good >= 2

    print(f"  C1 Sharpe > 1.5:        {'✅' if c1 else '❌'}  ({sol_metrics['sharpe']:.2f})")
    print(f"  C2 N ≥ 15:              {'✅' if c2 else '❌'}  ({sol_metrics['n_trades']})")
    print(f"  C3 WR > 50%:            {'✅' if c3 else '❌'}  ({sol_metrics['win_rate']:.1%})")
    print(f"  C4 Max DD < 5%:         {'✅' if c4 else '❌'}  ({sol_metrics['max_dd']:.2f}%)")
    print(f"  C5 Robusto 2/2 períodos:{'✅' if c5 else '❌'}  ({n_sol_good}/2)")

    n_met = sum([c1, c2, c3, c4, c5])
    if n_met >= 4 and c1 and c2:
        verdict = "APROVADO"
        action  = "Implementar Bot 5 SOL (Bot 2 transfer) e reativar crontab"
    elif (sol_metrics["sharpe"] < 1.0 or sol_metrics["n_trades"] < 10
          or abs(sol_metrics["max_dd"]) > 10):
        verdict = "REJEITADO"
        action  = "SOL não tem edge com Bot 2 strategy em 2026. Abandonar SOL por agora."
    else:
        verdict = "INCONCLUSIVO"
        action  = (f"Borderline ({n_met}/5 critérios). Sharpe {sol_metrics['sharpe']:.2f}. "
                   "Acumular mais dados 2026.")

    print(f"\n  → VEREDITO: {verdict}  ({n_met}/5 critérios)")
    print(f"  → AÇÃO:    {action}")

    # ── 6. Plots & Tables ─────────────────────────────────────────
    print("\n[6/7] Plots & tables...")
    plot_equity_curves(btc_trades, sol_trades)
    plot_returns_bars(btc_trades, sol_trades)
    plot_subperiods(subperiod_rows)
    plot_feature_scatter(btc_trades, "BTC Bot 2")
    plot_feature_scatter(sol_trades, "SOL Bot 2 Transfer")

    # Tables
    all_trades_df = pd.concat([
        pd.DataFrame([{"asset": "btc", **t} for t in btc_trades]),
        pd.DataFrame([{"asset": "sol", **t} for t in sol_trades]),
    ], ignore_index=True) if btc_trades or sol_trades else pd.DataFrame()
    if not all_trades_df.empty:
        all_trades_df.to_csv(OUT_TABLES / "trades_btc_vs_sol.csv", index=False)

    pd.DataFrame(subperiod_rows).to_csv(OUT_TABLES / "subperiods.csv", index=False)

    comparison = pd.DataFrame([
        {"asset": "BTC Bot 2 (2026)", **btc_metrics},
        {"asset": "SOL Bot 2 transfer (2026)", **sol_metrics},
    ])
    comparison.to_csv(OUT_TABLES / "comparison_summary.csv", index=False)
    print(f"  → CSVs saved ({len(list(OUT_TABLES.glob('*.csv')))} files)")

    # ── 7. Report ─────────────────────────────────────────────────
    print(f"\n[7/7] Writing report {OUT_REPORT}...")

    def _pf(v): return "∞" if v >= 99 else f"{v:.2f}"

    sp_rows_md = "\n".join(
        f"| {r['period']:<15} | {r['btc_n']:2d} | {r['btc_sharpe']:.2f} | {r['btc_wr']:.0%} "
        f"| {r['sol_n']:2d} | {r['sol_sharpe']:.2f} | {r['sol_wr']:.0%} |"
        for r in subperiod_rows
    )

    impl_code = ""
    if verdict == "APROVADO":
        impl_code = """
## 8. Implementação Bot 5 SOL

Bot 5 = copy do Bot 2 com dados SOL. Criar `src/trading/sol_bot5.py`:

```python
# Filtros IDÊNTICOS ao Bot 2 BTC (sem modificação)
FILTERS = {
    "stablecoin_z_min": 1.3,
    "ret_1d_min":       0.0,
    "rsi_min":         60.0,
    "rsi_max":         80.0,
    "bb_pct_max":       0.98,
    "spike_ret_max":    0.03,
    "spike_rsi_max":   65.0,
}

STOPS = {
    "sl_pct":         0.015,
    "tp_pct":         0.020,
    "trail_pct":      0.010,
    "max_hold_hours": 120,
    "cooldown_hours":   4,
}
```

Diferenças de Bot 2:
- Lê `data/01_raw/spot/sol_1h.parquet` (não BTC)
- Computa RSI/BB/MA21 em SOL (não vem do clean parquet)
- stablecoin_z = mesmo path (macro, asset-agnostic)
- Portfolio em `data/04_scoring/portfolio_sol5.json`
"""

    report = f"""# SOL Bot 5 — Bot 2 BTC Strategy Transfer (2026 ONLY)

**Data:** 2026-04-22
**Objetivo:** Testar se Bot 2 BTC strategy transfere para SOL em 2026

---

## 1. Dados

| Asset | Período | Rows |
|-------|---------|------|
| BTC (clean) | Jan-Abr 2026 | {len(btc)} |
| SOL (raw) | Jan-Abr 2026 | {len(sol)} |
| stablecoin_z | Oct 2025 - Abr 15 2026 | {sol['stablecoin_z'].notna().sum()} |

> stablecoin_z com forward-fill para Abr 22 (última atualização: Abr 15)

---

## 2. Resultado BTC Bot 2 (2026 — baseline)

| Metric | BTC Bot 2 |
|--------|-----------|
| N trades | {btc_metrics['n_trades']} |
| Sharpe | {btc_metrics['sharpe']:.2f} |
| Win Rate | {btc_metrics['win_rate']:.0%} |
| Avg Return | {btc_metrics['avg_return']:.3f}% |
| Total Return | {btc_metrics['total_return']:+.2f}% |
| Max DD | {btc_metrics['max_dd']:.2f}% |
| Profit Factor | {_pf(btc_metrics['profit_factor'])} |

> Reference: live performance BTC Bot 2 Mar-Abr 2026 — WR 80%, PF 2.07, +1.83%

---

## 3. Resultado SOL Bot 2 Transfer (2026)

| Metric | SOL Bot 2 |
|--------|-----------|
| N trades | {sol_metrics['n_trades']} |
| Sharpe | **{sol_metrics['sharpe']:.2f}** |
| Win Rate | {sol_metrics['win_rate']:.0%} |
| Avg Return | {sol_metrics['avg_return']:.3f}% |
| Total Return | {sol_metrics['total_return']:+.2f}% |
| Max DD | {sol_metrics['max_dd']:.2f}% |
| Profit Factor | {_pf(sol_metrics['profit_factor'])} |

---

## 4. Comparação BTC vs SOL

| Metric | BTC Bot 2 | SOL Bot 2 | Δ |
|--------|-----------|-----------|---|
| N trades | {btc_metrics['n_trades']} | {sol_metrics['n_trades']} | — |
| Sharpe | {btc_metrics['sharpe']:.2f} | {sol_metrics['sharpe']:.2f} | {delta_sharpe:+.2f} |
| Win Rate | {btc_metrics['win_rate']:.0%} | {sol_metrics['win_rate']:.0%} | {sol_metrics['win_rate']-btc_metrics['win_rate']:+.0%} |
| Avg Return | {btc_metrics['avg_return']:.3f}% | {sol_metrics['avg_return']:.3f}% | {sol_metrics['avg_return']-btc_metrics['avg_return']:+.3f}% |
| Max DD | {btc_metrics['max_dd']:.2f}% | {sol_metrics['max_dd']:.2f}% | — |
| Profit Factor | {_pf(btc_metrics['profit_factor'])} | {_pf(sol_metrics['profit_factor'])} | — |

---

## 5. Sub-períodos (SOL)

| Período | BTC N | BTC Sharpe | BTC WR | SOL N | SOL Sharpe | SOL WR |
|---------|-------|------------|--------|-------|------------|--------|
{sp_rows_md}

**SOL períodos com Sharpe > 1.0:** {n_sol_good}/{len(subperiods)}

---

## 6. Critérios de Decisão

| Critério | Meta | Resultado | Status |
|----------|------|-----------|--------|
| Sharpe > 1.5 | > 1.5 | {sol_metrics['sharpe']:.2f} | {'✅' if c1 else '❌'} |
| N ≥ 15 trades | ≥ 15 | {sol_metrics['n_trades']} | {'✅' if c2 else '❌'} |
| WR > 50% | > 50% | {sol_metrics['win_rate']:.0%} | {'✅' if c3 else '❌'} |
| Max DD < 5% | < 5% | {sol_metrics['max_dd']:.2f}% | {'✅' if c4 else '❌'} |
| Robusto 2/2 | ≥ 2/2 | {n_sol_good}/2 | {'✅' if c5 else '❌'} |

**Critérios atendidos:** {n_met}/5

---

## 7. **VEREDITO: {verdict}**

**Ação:** {action}

### Interpretação

{'**Hipótese CONFIRMADA:** o problema com SOL v1 eram as features SOL-specific (taker_z/oi_z), não o asset. Bot 2 puro tem edge em SOL em 2026.' if verdict == 'APROVADO' else '**Hipótese REJEITADA:** Bot 2 strategy não transfere para SOL. O problema é o regime SOL em 2026, não as features específicas. Qualquer strategy de momentum enfrenta dificuldades.' if verdict == 'REJEITADO' else f'**Hipótese INCONCLUSIVA:** Sharpe {sol_metrics["sharpe"]:.2f} é borderline. Pode haver edge real mas N={sol_metrics["n_trades"]} é insuficiente para conclusão sólida. BTC tem Sharpe {btc_metrics["sharpe"]:.2f} com mesma strategy — a diferença de {delta_sharpe:+.2f} sugere atrito de asset.'}
{impl_code}

---

## 9. Plots gerados

- `plots/sol_bot2_transfer/equity_curves.png`
- `plots/sol_bot2_transfer/returns_distribution.png`
- `plots/sol_bot2_transfer/subperiods_comparison.png`
- `plots/sol_bot2_transfer/feature_scatter_btc.png`
- `plots/sol_bot2_transfer/feature_scatter_sol.png`

## 10. Tables geradas

- `tables/sol_bot2_transfer/comparison_summary.csv`
- `tables/sol_bot2_transfer/trades_btc_vs_sol.csv`
- `tables/sol_bot2_transfer/subperiods.csv`
"""

    OUT_REPORT.write_text(report)
    print(f"  → {OUT_REPORT} saved")

    print("\n" + "=" * 65)
    print(f"DONE — VEREDITO: {verdict}  ({n_met}/5)")
    print("=" * 65)
    return verdict, sol_metrics


if __name__ == "__main__":
    main()
