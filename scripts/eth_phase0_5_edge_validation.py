"""
ETH Phase 0.5 — Quick Edge Validation.

Testa rapidamente se Volume + OI (descobertos na Phase 0) geram edge acionável.

Output:
  - prints estruturados no terminal
  - prompts/eth_phase0_5_summary.md (resumo one-pager)

Usage:
  python scripts/eth_phase0_5_edge_validation.py
"""
import sys
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUT_DIR = ROOT / "prompts"
OUT_DIR.mkdir(exist_ok=True)
SUMMARY_PATH = OUT_DIR / "eth_phase0_5_summary.md"


# ==========================================================
# DATA LOADING
# ==========================================================

def load_data():
    """Carrega spot + OI, calcula features, retorna df daily."""
    # Spot ETH
    spot = pd.read_parquet(ROOT / "data/01_raw/spot/eth_1h.parquet")
    spot["timestamp"] = pd.to_datetime(spot["timestamp"], utc=True)

    spot_d = spot.set_index("timestamp").resample("D").agg({
        "close": "last",
        "volume": "sum"
    }).dropna().reset_index()

    # Forward returns
    for h in [1, 3, 7, 14]:
        spot_d[f"fwd_{h}d"] = spot_d["close"].shift(-h) / spot_d["close"] - 1

    # Volume z-score (rolling 30d)
    spot_d["volume_z"] = (
        (spot_d["volume"] - spot_d["volume"].rolling(30).mean()) /
        spot_d["volume"].rolling(30).std()
    )

    # OI data
    oi = pd.read_parquet(ROOT / "data/01_raw/futures/eth_oi_4h.parquet")
    oi["timestamp"] = pd.to_datetime(oi["timestamp"], utc=True)
    oi_d = oi.set_index("timestamp").resample("D").mean(numeric_only=True)

    df = spot_d.merge(oi_d, on="timestamp", how="left")

    # Identify OI column
    oi_col = None
    for col in df.columns:
        if "oi" in col.lower() or "open_interest" in col.lower():
            if col in ["oi_z"]:
                continue
            oi_col = col
            break

    if oi_col:
        df["oi_z"] = (
            (df[oi_col] - df[oi_col].rolling(30).mean()) /
            df[oi_col].rolling(30).std()
        )
        print(f"[Info] Using OI column: {oi_col}")
    else:
        print("[Warn] No OI column found")

    return df


# ==========================================================
# ANALYSIS FUNCTIONS
# ==========================================================

def conditional_returns(df, condition, label, baseline_avg=None):
    """
    Print forward returns for a subset defined by condition.
    Also shows delta vs baseline if provided.
    """
    sub = df[condition].copy()

    if len(sub) < 20:
        print(f"  {label}: too few samples ({len(sub)})")
        return None

    print(f"\n  === {label} ===")
    print(f"  N = {len(sub)} ({len(sub)/len(df)*100:.1f}% of data)")

    results = {}
    for h in [1, 3, 7, 14]:
        col = f"fwd_{h}d"
        valid = sub[col].dropna()
        if len(valid) < 10:
            continue

        avg = valid.mean()
        med = valid.median()
        win = (valid > 0).mean()

        delta_str = ""
        if baseline_avg and h in baseline_avg:
            delta = (avg - baseline_avg[h]) * 100
            delta_str = f" | Δ baseline: {delta:+.2f}pp"

        print(f"  {h}d → avg: {avg*100:+.2f}% | med: {med*100:+.2f}% | WR: {win*100:.0f}%{delta_str}")

        results[h] = {
            "avg": avg,
            "med": med,
            "win_rate": win,
            "n": len(valid),
        }

    return results


def baseline_returns(df):
    """Return baseline forward returns (overall)."""
    baseline = {}
    for h in [1, 3, 7, 14]:
        col = f"fwd_{h}d"
        valid = df[col].dropna()
        baseline[h] = valid.mean()
    return baseline


def test_correlations_multi_window(df, feature, target="fwd_7d"):
    """Test if correlation is stable across different windows."""
    print(f"\n  Correlation {feature} → {target} (multi-window):")

    results = []
    for window in [90, 180, 270, 365]:
        sub = df.tail(window)[[feature, target]].dropna()
        if len(sub) < 30:
            continue

        r, p = pearsonr(sub[feature], sub[target])
        sig = "✅" if p < 0.05 else "❌"
        print(f"  {window}d: corr={r:+.3f} | p={p:.4f} {sig} | N={len(sub)}")
        results.append({
            "window": window,
            "correlation": r,
            "p_value": p,
            "significant": p < 0.05,
            "n": len(sub),
        })

    return results


def quartile_analysis(df, feature, target="fwd_7d"):
    """Analyze forward returns by quartiles of a feature."""
    sub = df[[feature, target]].dropna()
    if len(sub) < 40:
        print(f"  {feature}: too few samples for quartile analysis")
        return

    sub = sub.copy()
    sub["quartile"] = pd.qcut(sub[feature], q=4, labels=["Q1 (low)", "Q2", "Q3", "Q4 (high)"])

    print(f"\n  Quartile analysis: {feature} → {target}")
    print(f"  {'Quartile':<12} {'Range':<20} {'N':<5} {'Avg':>10} {'WR':>6}")
    print(f"  {'-'*60}")

    for q in sub["quartile"].cat.categories:
        q_data = sub[sub["quartile"] == q]
        range_str = f"[{q_data[feature].min():+.2f}, {q_data[feature].max():+.2f}]"
        avg = q_data[target].mean() * 100
        win = (q_data[target] > 0).mean() * 100
        print(f"  {str(q):<12} {range_str:<20} {len(q_data):<5} {avg:>+8.2f}% {win:>5.0f}%")


def simple_backtest(df):
    """Backtest minimal rule-based strategy."""
    print("\n" + "="*60)
    print("SIMPLE BACKTEST — Rule-based")
    print("="*60)

    if "oi_z" not in df.columns:
        print("  [Skip] No OI data")
        return

    # Regra simples: entra em "low volume + high OI" (acumulação silenciosa)
    entry_signal = (df["volume_z"] < -0.5) & (df["oi_z"] > 0.5)

    trades = []
    in_position = False
    entry_idx = None
    entry_price = None

    for i in range(len(df)):
        row = df.iloc[i]

        if not in_position:
            if entry_signal.iloc[i]:
                entry_idx = i
                entry_price = row["close"]
                in_position = True
        else:
            hold_days = i - entry_idx
            current_price = row["close"]
            ret = (current_price - entry_price) / entry_price

            # Exit conditions: 7 days OR -2% stop OR +4% target
            exit_reason = None
            if ret <= -0.02:
                exit_reason = "STOP"
            elif ret >= 0.04:
                exit_reason = "TARGET"
            elif hold_days >= 7:
                exit_reason = "TIME"

            if exit_reason:
                trades.append({
                    "entry_date": df.iloc[entry_idx]["timestamp"],
                    "exit_date": row["timestamp"],
                    "entry_price": entry_price,
                    "exit_price": current_price,
                    "return": ret,
                    "hold_days": hold_days,
                    "reason": exit_reason,
                })
                in_position = False

    if not trades:
        print("  No trades triggered")
        return

    td = pd.DataFrame(trades)

    wr = (td["return"] > 0).mean() * 100
    avg_ret = td["return"].mean() * 100
    total_ret = ((1 + td["return"]).prod() - 1) * 100
    sharpe = td["return"].mean() / td["return"].std() * np.sqrt(52) if td["return"].std() > 0 else 0

    # Max drawdown
    cum = (1 + td["return"]).cumprod()
    max_dd = ((cum / cum.cummax() - 1).min()) * 100

    print(f"\n  Rule: volume_z < -0.5 AND oi_z > +0.5")
    print(f"  Exit: 7d TIME | -2% STOP | +4% TARGET")
    print(f"  ")
    print(f"  N trades:        {len(trades)}")
    print(f"  Win rate:        {wr:.1f}%")
    print(f"  Avg return:      {avg_ret:+.2f}%")
    print(f"  Total return:    {total_ret:+.2f}%")
    print(f"  Sharpe (annual): {sharpe:.2f}")
    print(f"  Max drawdown:    {max_dd:.2f}%")
    print(f"  ")
    print(f"  Exit reasons: {dict(td['reason'].value_counts())}")

    return {
        "n_trades": len(trades),
        "win_rate": wr,
        "avg_return": avg_ret,
        "total_return": total_ret,
        "sharpe": sharpe,
        "max_dd": max_dd,
        "trades": trades,
    }


# ==========================================================
# MAIN
# ==========================================================

def run_tests(df):
    print("\n" + "="*60)
    print("BASELINE — Overall forward returns")
    print("="*60)

    baseline = baseline_returns(df)
    for h, avg in baseline.items():
        print(f"  {h}d: {avg*100:+.2f}%")

    # ------------------------------------------------------
    # H4 — Multi-window correlations (stability test)
    # ------------------------------------------------------
    print("\n" + "="*60)
    print("H4 — MULTI-WINDOW CORRELATIONS")
    print("="*60)

    vol_corr = test_correlations_multi_window(df, "volume_z", "fwd_7d")

    oi_corr = None
    if "oi_z" in df.columns:
        oi_corr = test_correlations_multi_window(df, "oi_z", "fwd_7d")

    # ------------------------------------------------------
    # Quartile analyses (edge intensity)
    # ------------------------------------------------------
    print("\n" + "="*60)
    print("QUARTILE ANALYSIS (last 180d)")
    print("="*60)

    df180 = df.tail(180).copy()
    quartile_analysis(df180, "volume_z")

    if "oi_z" in df180.columns:
        quartile_analysis(df180, "oi_z")

    # ------------------------------------------------------
    # H1 — Volume tests
    # ------------------------------------------------------
    print("\n" + "="*60)
    print("H1 — VOLUME TESTS (last 180d)")
    print("="*60)

    conditional_returns(df180, df180["volume_z"] > 1.0, "volume_z > +1 (high)", baseline)
    conditional_returns(df180, df180["volume_z"] < -1.0, "volume_z < -1 (low)", baseline)
    conditional_returns(df180, df180["volume_z"] > 2.0, "volume_z > +2 (extreme high)", baseline)
    conditional_returns(df180, df180["volume_z"] < -2.0, "volume_z < -2 (extreme low)", baseline)

    # ------------------------------------------------------
    # H2 — OI tests (inverted signal)
    # ------------------------------------------------------
    if "oi_z" in df180.columns:
        print("\n" + "="*60)
        print("H2 — OI TESTS (ETH LOGIC — inverted)")
        print("="*60)

        conditional_returns(df180, df180["oi_z"] > 1.0, "oi_z > +1 (ETH bullish)", baseline)
        conditional_returns(df180, df180["oi_z"] < -1.0, "oi_z < -1 (ETH bearish)", baseline)

    # ------------------------------------------------------
    # H3 — Combined signal
    # ------------------------------------------------------
    if "oi_z" in df180.columns:
        print("\n" + "="*60)
        print("H3 — COMBINED SIGNAL (last 180d)")
        print("="*60)

        # Acumulação silenciosa: volume baixo + OI alto
        combo1 = (df180["volume_z"] < -0.5) & (df180["oi_z"] > 0.5)
        conditional_returns(df180, combo1, "volume_z <-0.5 AND oi_z >+0.5 (silent accum)", baseline)

        # Distribuição: volume alto + OI alto (topo)
        combo2 = (df180["volume_z"] > 1.0) & (df180["oi_z"] > 1.0)
        conditional_returns(df180, combo2, "volume_z >+1 AND oi_z >+1 (distribution?)", baseline)

        # Capitulação: volume alto + OI baixo
        combo3 = (df180["volume_z"] > 1.0) & (df180["oi_z"] < -1.0)
        conditional_returns(df180, combo3, "volume_z >+1 AND oi_z <-1 (capitulation?)", baseline)

    # ------------------------------------------------------
    # Simple backtest
    # ------------------------------------------------------
    bt_results = simple_backtest(df180)

    # ------------------------------------------------------
    # Generate summary
    # ------------------------------------------------------
    generate_summary(vol_corr, oi_corr, bt_results, baseline)


def generate_summary(vol_corr, oi_corr, bt_results, baseline):
    """Generate a one-pager summary."""
    lines = []
    lines.append("# ETH Phase 0.5 — Edge Validation Summary")
    lines.append(f"\n**Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    lines.append("")

    # Baseline
    lines.append("## Baseline\n")
    lines.append("Forward returns on all data:")
    for h, avg in baseline.items():
        lines.append(f"- {h}d: `{avg*100:+.2f}%`")
    lines.append("")

    # Volume multi-window
    lines.append("## H4 — Volume correlation stability\n")
    lines.append("| Window | Corr | p-value | Significant |")
    lines.append("|--------|------|---------|-------------|")
    for r in vol_corr:
        sig = "✅" if r["significant"] else "❌"
        lines.append(f"| {r['window']}d | {r['correlation']:+.3f} | {r['p_value']:.4f} | {sig} |")

    all_consistent = all(np.sign(r["correlation"]) == np.sign(vol_corr[0]["correlation"]) for r in vol_corr)
    all_significant = all(r["significant"] for r in vol_corr)
    lines.append(f"\n**Structural?** {'✅ Sign consistent across windows' if all_consistent else '❌ Sign flips'}")
    lines.append(f"**All significant?** {'✅' if all_significant else '❌'}")
    lines.append("")

    # OI multi-window
    if oi_corr:
        lines.append("## H4 — OI correlation stability\n")
        lines.append("| Window | Corr | p-value | Significant |")
        lines.append("|--------|------|---------|-------------|")
        for r in oi_corr:
            sig = "✅" if r["significant"] else "❌"
            lines.append(f"| {r['window']}d | {r['correlation']:+.3f} | {r['p_value']:.4f} | {sig} |")

        all_consistent = all(np.sign(r["correlation"]) == np.sign(oi_corr[0]["correlation"]) for r in oi_corr)
        lines.append(f"\n**Structural?** {'✅ Sign consistent' if all_consistent else '❌ Sign flips'}")
        lines.append("")

    # Backtest
    if bt_results:
        lines.append("## Backtest — Simple rule\n")
        lines.append("**Rule:** `volume_z < -0.5 AND oi_z > +0.5` (silent accumulation)")
        lines.append("**Exit:** 7d TIME / -2% STOP / +4% TARGET\n")
        lines.append(f"- N trades: {bt_results['n_trades']}")
        lines.append(f"- Win rate: {bt_results['win_rate']:.1f}%")
        lines.append(f"- Avg return: {bt_results['avg_return']:+.2f}%")
        lines.append(f"- Total return: {bt_results['total_return']:+.2f}%")
        lines.append(f"- Sharpe (annualized): {bt_results['sharpe']:.2f}")
        lines.append(f"- Max drawdown: {bt_results['max_dd']:.2f}%")
        lines.append("")

    # Decision guide
    lines.append("## Decision Guide\n")
    lines.append("### Proceed to Phase 1 (implement ETH trading) IF:")
    lines.append("- ✅ Volume corr significant in 3+ windows")
    lines.append("- ✅ Backtest WR > 55%")
    lines.append("- ✅ Backtest Sharpe > 1.0")
    lines.append("- ✅ Backtest max DD < 15%")
    lines.append("")
    lines.append("### Continue research IF:")
    lines.append("- ⚠️ Some signals strong but not all criteria met")
    lines.append("- ⚠️ Sinais inconsistentes em diferentes windows")
    lines.append("")
    lines.append("### Abandon ETH IF:")
    lines.append("- ❌ Backtest WR < 45%")
    lines.append("- ❌ Correlações viram sinal entre windows")
    lines.append("- ❌ N trades insuficiente")

    with open(SUMMARY_PATH, "w") as f:
        f.write("\n".join(lines))

    print(f"\nSummary: {SUMMARY_PATH}")


def main():
    df = load_data()
    print(f"\n[Info] Loaded {len(df)} days of ETH data")
    print(f"[Info] Period: {df['timestamp'].min().date()} → {df['timestamp'].max().date()}")

    run_tests(df)


if __name__ == "__main__":
    main()
