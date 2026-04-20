"""
ETH Phase 0.5 v2 — Quick Edge Validation (2026 ONLY).

Valida tese: 'passado distante não vale em crypto'.
Re-roda análise da v1 mas filtrando apenas dados de 2026.

Output:
  - prints terminal
  - prompts/eth_phase0_5_v2_2026_summary.md

Usage:
  python scripts/eth_phase0_5_v2_2026_only.py
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
SUMMARY_PATH = OUT_DIR / "eth_phase0_5_v2_2026_summary.md"


def load_data_2026():
    """Carrega spot + OI, calcula features, FILTRA APENAS 2026."""
    spot = pd.read_parquet(ROOT / "data/01_raw/spot/eth_1h.parquet")
    spot["timestamp"] = pd.to_datetime(spot["timestamp"], utc=True)

    spot_d = spot.set_index("timestamp").resample("D").agg({
        "close": "last",
        "volume": "sum"
    }).dropna().reset_index()

    for h in [1, 3, 7, 14]:
        spot_d[f"fwd_{h}d"] = spot_d["close"].shift(-h) / spot_d["close"] - 1

    # Z-score com rolling 30d (calculado ANTES do filtro para ter histórico)
    spot_d["volume_z"] = (
        (spot_d["volume"] - spot_d["volume"].rolling(30).mean()) /
        spot_d["volume"].rolling(30).std()
    )

    oi = pd.read_parquet(ROOT / "data/01_raw/futures/eth_oi_4h.parquet")
    oi["timestamp"] = pd.to_datetime(oi["timestamp"], utc=True)
    oi_d = oi.set_index("timestamp").resample("D").mean(numeric_only=True)

    df = spot_d.merge(oi_d, on="timestamp", how="left")

    oi_col = None
    for col in df.columns:
        if ("oi" in col.lower() or "open_interest" in col.lower()) and col not in ["oi_z"]:
            oi_col = col
            break

    if oi_col:
        # Calcula z-score ANTES do filtro 2026 (usa histórico disponível)
        df["oi_z"] = (
            (df[oi_col] - df[oi_col].rolling(30).mean()) /
            df[oi_col].rolling(30).std()
        )
        print(f"[Info] Using OI column: {oi_col}")

    # FILTRO CRÍTICO — apenas 2026
    cutoff = pd.Timestamp("2026-01-01", tz="UTC")
    df_2026 = df[df["timestamp"] >= cutoff].reset_index(drop=True)

    print(f"[Info] Total days in 2026: {len(df_2026)}")
    print(f"[Info] Period: {df_2026['timestamp'].min().date()} → {df_2026['timestamp'].max().date()}")

    return df_2026


def conditional_returns(df, condition, label, baseline_avg=None):
    sub = df[condition].copy()

    if len(sub) < 10:
        print(f"  {label}: too few samples ({len(sub)})")
        return None

    print(f"\n  === {label} ===")
    print(f"  N = {len(sub)} ({len(sub)/len(df)*100:.1f}% of data)")

    results = {}
    for h in [1, 3, 7, 14]:
        col = f"fwd_{h}d"
        valid = sub[col].dropna()
        if len(valid) < 5:
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
    baseline = {}
    for h in [1, 3, 7, 14]:
        col = f"fwd_{h}d"
        valid = df[col].dropna()
        baseline[h] = valid.mean()
    return baseline


def test_correlations_sub_windows(df, feature, target="fwd_7d"):
    """Testa correlação em sub-janelas dentro de 2026."""
    print(f"\n  Correlation {feature} → {target} (sub-windows within 2026):")

    results = []
    total_days = len(df)
    windows = [30, 60, 90, total_days]

    for window in windows:
        if window > total_days:
            window = total_days

        sub = df.tail(window)[[feature, target]].dropna()
        if len(sub) < 15:
            continue

        r, p = pearsonr(sub[feature], sub[target])
        sig = "✅" if p < 0.05 else "❌"
        label = f"last {window}d" if window < total_days else f"all 2026 ({window}d)"
        print(f"  {label}: corr={r:+.3f} | p={p:.4f} {sig} | N={len(sub)}")
        results.append({
            "window": window,
            "label": label,
            "correlation": r,
            "p_value": p,
            "significant": p < 0.05,
            "n": len(sub),
        })

    return results


def quartile_analysis(df, feature, target="fwd_7d"):
    sub = df[[feature, target]].dropna()
    if len(sub) < 30:
        print(f"  {feature}: too few samples for quartile analysis")
        return

    sub = sub.copy()
    sub["quartile"] = pd.qcut(sub[feature], q=4, labels=["Q1 (low)", "Q2", "Q3", "Q4 (high)"], duplicates="drop")

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
    print("\n" + "="*60)
    print("SIMPLE BACKTEST — Rule-based (2026 only)")
    print("="*60)

    if "oi_z" not in df.columns:
        print("  [Skip] No OI data")
        return

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
        return None

    td = pd.DataFrame(trades)

    wr = (td["return"] > 0).mean() * 100
    avg_ret = td["return"].mean() * 100
    total_ret = ((1 + td["return"]).prod() - 1) * 100
    sharpe = td["return"].mean() / td["return"].std() * np.sqrt(52) if td["return"].std() > 0 else 0

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
    print(f"  Exit reasons: {dict(td['reason'].value_counts())}")

    return {
        "n_trades": len(trades),
        "win_rate": wr,
        "avg_return": avg_ret,
        "total_return": total_ret,
        "sharpe": sharpe,
        "max_dd": max_dd,
    }


def compare_with_v1(corr_vol_2026, corr_oi_2026):
    """Print comparison 2026-only vs v1 (180d)."""
    print("\n" + "="*60)
    print("COMPARACAO: 2026-only vs v1 (180 dias)")
    print("="*60)

    print("\nVolume:")
    print(f"  v1 (180d):  corr = -0.387  ✅")
    for r in corr_vol_2026:
        if "all 2026" in r["label"]:
            sig = "✅" if r["significant"] else "❌"
            print(f"  v2 (2026):  corr = {r['correlation']:+.3f}  {sig}")
            break

    if corr_oi_2026:
        print("\nOI:")
        print(f"  v1 (90d):   corr = +0.338  ✅ (hipotese 'artefato')")
        print(f"  v1 (180d):  insignificante  ❌")
        for r in corr_oi_2026:
            if "all 2026" in r["label"]:
                sig = "✅" if r["significant"] else "❌"
                print(f"  v2 (2026):  corr = {r['correlation']:+.3f}  {sig}")
                print(f"\n  Conclusao:")
                if r["significant"] and abs(r["correlation"]) > 0.2:
                    print("  ✅ OI REAL: padrao e de 2026, nao artefato 90d")
                    print("  → tese de 'memoria curta' confirmada")
                    print("  → OI invertido e legitimo em ETH")
                else:
                    print("  ❌ OI NAO confirma em 2026-only")
                    print("  → 90d realmente foi artefato")
                    print("  → focar so em volume")
                break


def run_tests_2026(df):
    print("\n" + "="*60)
    print("ETH Phase 0.5 v2 — 2026 ONLY")
    print("="*60)
    print(f"Dataset: {len(df)} days")
    print(f"Period: {df['timestamp'].min().date()} → {df['timestamp'].max().date()}")

    print("\n" + "="*60)
    print("BASELINE — Forward returns in 2026")
    print("="*60)

    baseline = baseline_returns(df)
    for h, avg in baseline.items():
        print(f"  {h}d: {avg*100:+.2f}%")

    print("\n" + "="*60)
    print("VOLUME CORRELATIONS — Sub-windows within 2026")
    print("="*60)

    vol_corr = test_correlations_sub_windows(df, "volume_z", "fwd_7d")

    oi_corr = None
    if "oi_z" in df.columns:
        print("\n" + "="*60)
        print("OI CORRELATIONS — Sub-windows within 2026")
        print("="*60)
        oi_corr = test_correlations_sub_windows(df, "oi_z", "fwd_7d")

    print("\n" + "="*60)
    print("QUARTILE ANALYSIS")
    print("="*60)

    quartile_analysis(df, "volume_z")
    if "oi_z" in df.columns:
        quartile_analysis(df, "oi_z")

    print("\n" + "="*60)
    print("CONDITIONAL RETURNS (2026)")
    print("="*60)

    conditional_returns(df, df["volume_z"] > 1.0, "volume_z > +1", baseline)
    conditional_returns(df, df["volume_z"] < -1.0, "volume_z < -1", baseline)
    conditional_returns(df, df["volume_z"] > 1.5, "volume_z > +1.5 (extreme high)", baseline)

    if "oi_z" in df.columns:
        conditional_returns(df, df["oi_z"] > 1.0, "oi_z > +1 (ETH bullish?)", baseline)
        conditional_returns(df, df["oi_z"] < -1.0, "oi_z < -1 (ETH bearish?)", baseline)

        combo = (df["volume_z"] < -0.5) & (df["oi_z"] > 0.5)
        conditional_returns(df, combo, "volume_z <-0.5 AND oi_z >+0.5", baseline)

    bt = simple_backtest(df)

    compare_with_v1(vol_corr, oi_corr)

    generate_summary_v2(vol_corr, oi_corr, bt, baseline, len(df))


def generate_summary_v2(vol_corr, oi_corr, bt_results, baseline, n_days):
    lines = []
    lines.append("# ETH Phase 0.5 v2 — Only 2026 Window")
    lines.append(f"\n**Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    lines.append(f"**Window:** 2026-01-01 → today ({n_days} days)")
    lines.append("")

    lines.append("## Tese testada\n")
    lines.append("> Em crypto, passado distante (2025, 2024) dilui o sinal real do regime atual.")
    lines.append("> Rodar apenas com 2026 deve revelar edges que a janela de 180d (misturada) escondia.")
    lines.append("")

    lines.append("## Baseline 2026\n")
    for h, avg in baseline.items():
        lines.append(f"- {h}d: `{avg*100:+.2f}%`")
    lines.append("")

    lines.append("## Volume correlations (sub-windows)\n")
    lines.append("| Window | Corr | p-value | Significant |")
    lines.append("|--------|------|---------|-------------|")
    for r in vol_corr:
        sig = "✅" if r["significant"] else "❌"
        lines.append(f"| {r['label']} | {r['correlation']:+.3f} | {r['p_value']:.4f} | {sig} |")
    lines.append("")

    if oi_corr:
        lines.append("## OI correlations (sub-windows)\n")
        lines.append("| Window | Corr | p-value | Significant |")
        lines.append("|--------|------|---------|-------------|")
        for r in oi_corr:
            sig = "✅" if r["significant"] else "❌"
            lines.append(f"| {r['label']} | {r['correlation']:+.3f} | {r['p_value']:.4f} | {sig} |")
        lines.append("")

    if bt_results:
        lines.append("## Backtest 2026\n")
        lines.append(f"- N trades: {bt_results['n_trades']}")
        lines.append(f"- Win rate: {bt_results['win_rate']:.1f}%")
        lines.append(f"- Avg return: {bt_results['avg_return']:+.2f}%")
        lines.append(f"- Sharpe: {bt_results['sharpe']:.2f}")
        lines.append(f"- Max DD: {bt_results['max_dd']:.2f}%")
        lines.append("")

    lines.append("## Comparacao com v1 (180 dias)\n")
    lines.append("| Metrica | v1 (180d) | v2 (2026) | Mudanca |")
    lines.append("|---------|-----------|-----------|---------|")

    v1_vol = -0.387
    v2_vol = None
    for r in vol_corr:
        if "all 2026" in r["label"]:
            v2_vol = r["correlation"]
            break

    if v2_vol is not None:
        diff = v2_vol - v1_vol
        lines.append(f"| Volume corr | {v1_vol:+.3f} | {v2_vol:+.3f} | {diff:+.3f} |")

    if oi_corr:
        v2_oi = None
        for r in oi_corr:
            if "all 2026" in r["label"]:
                v2_oi = r["correlation"]
                break
        if v2_oi is not None:
            lines.append(f"| OI corr (2026) | insig | {v2_oi:+.3f} | — |")

    lines.append("")

    lines.append("## Conclusao e decisao\n")

    v2_vol_significant = v2_vol is not None and abs(v2_vol) > 0.2

    v2_oi_significant = False
    if oi_corr:
        v2_oi_significant = any(
            r["significant"] and abs(r["correlation"]) > 0.2
            for r in oi_corr if "all 2026" in r["label"]
        )

    if v2_vol_significant and v2_oi_significant:
        lines.append("### TESE CONFIRMADA")
        lines.append("- Volume mantém edge em 2026-only ✅")
        lines.append("- OI agora também tem edge (era regime 2026, não artefato 90d) ✅")
        lines.append("- 'Memória curta' valida: dados 2025 estavam diluindo sinais")
        lines.append("")
        lines.append("**Próxima ação:** Phase 1 ETH com volume + OI (sinais limpos)")
    elif v2_vol_significant:
        lines.append("### TESE PARCIAL")
        lines.append("- Volume mantém edge ✅")
        lines.append("- OI continua incerto ❌")
        lines.append("- 2026-only confirma volume mas não OI")
        lines.append("")
        lines.append("**Próxima ação:** Phase 1 ETH apenas com volume (OI fica de fora)")
    else:
        lines.append("### TESE NAO CONFIRMADA")
        lines.append("- Sinal volume enfraquece em 2026-only")
        lines.append("- Talvez 2026 seja curto demais (só ~110 dias)")
        lines.append("")
        lines.append("**Próxima ação:** Considerar voltar para 180d ou coletar mais dados 2026")

    with open(SUMMARY_PATH, "w") as f:
        f.write("\n".join(lines))

    print(f"\nSummary: {SUMMARY_PATH}")


def main():
    df = load_data_2026()
    run_tests_2026(df)


if __name__ == "__main__":
    main()
