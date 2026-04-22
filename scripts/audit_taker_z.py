"""
Auditoria Anti Look-Ahead Bias — taker_z.

O pipeline produz taker_1h_clean via resample("1h").ffill() sobre taker_4h.
Cada linha 1h herda o valor da 4h candle que INCLUI esse timestamp.
Se essa 4h candle ainda não estava completa no momento da decisão → look-ahead.

Testa empiricamente:
  - taker_z[t=0]  → valor que o backtest usa (potencial look-ahead)
  - taker_z[t-4]  → valor que produção teria disponível (4h anterior completo)

Compara Sharpe com cada versão para quantificar o impacto.
"""
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("audit_taker_z")

REPORT_PATH = ROOT / "prompts/audit_taker_z_report.md"


# ==================================================================
# LOAD DATA
# ==================================================================

def load_all():
    """Returns (df_signals, df_taker_4h, df_gate_1h)."""
    # Error analysis dataset (136 trades)
    ea_path = ROOT / "prompts/tables/error_analysis_full_dataset.csv"
    if not ea_path.exists():
        raise FileNotFoundError(f"Run error_analysis_losers.py first: {ea_path}")
    df_sig = pd.read_csv(ea_path)
    df_sig["entry_time"] = pd.to_datetime(df_sig["entry_time"], utc=True)
    df_sig["is_loser"] = ~df_sig["is_winner"]

    # Raw 4h taker data
    t4_path = ROOT / "data/01_raw/futures/taker_4h.parquet"
    df_4h = pd.read_parquet(t4_path)
    df_4h["timestamp"] = pd.to_datetime(df_4h["timestamp"], utc=True)
    df_4h = df_4h.sort_values("timestamp").reset_index(drop=True)

    # Gate z-scores (1h, already ffilled)
    gz_path = ROOT / "data/02_features/gate_zscores.parquet"
    df_gate = pd.read_parquet(gz_path)
    df_gate["timestamp"] = pd.to_datetime(df_gate["timestamp"], utc=True)
    df_gate = df_gate.sort_values("timestamp").reset_index(drop=True)

    logger.info(f"Signals: {len(df_sig)} | taker_4h: {len(df_4h)} rows | gate_1h: {len(df_gate)} rows")
    logger.info(f"taker_4h range: {df_4h['timestamp'].min().date()} → {df_4h['timestamp'].max().date()}")
    return df_sig, df_4h, df_gate


# ==================================================================
# TEMPORAL ANALYSIS: WHICH 4H CANDLE DOES EACH SIGNAL USE?
# ==================================================================

def analyze_4h_alignment(df_sig: pd.DataFrame, df_4h: pd.DataFrame, df_gate: pd.DataFrame) -> pd.DataFrame:
    """
    For each signal at time T, determine:
      1. taker_z[t]    — value backtest uses (from gate_zscores at T)
      2. taker_4h_current — which 4h candle timestamp is active at T (via ffill)
      3. hours_into_4h — how many hours into the current 4h candle is T?
      4. taker_z_prev  — taker_z from the PREVIOUS 4h candle (safe lag)
      5. taker_z_lag1h — gate_zscores at T-1 (safe for production)
    """
    records = []
    gate_by_ts = df_gate.set_index("timestamp")

    for _, row in df_sig.iterrows():
        et = row["entry_time"]

        # 1. taker_z[t=0] — what error_analysis used
        tz_t0 = row["taker_z"]

        # 2. Which 4h candle is "active" at T (via ffill)?
        # The active candle is the last 4h timestamp <= T
        mask_4h = df_4h["timestamp"] <= et
        if not mask_4h.any():
            records.append({"entry_time": et, "taker_z_t0": tz_t0,
                             "note": "no_4h_data"})
            continue

        idx_curr = df_4h[mask_4h].index[-1]
        ts_curr = df_4h.loc[idx_curr, "timestamp"]
        hours_into_candle = (et - ts_curr).total_seconds() / 3600

        # 3. taker_z from gate_zscores at T (should match taker_z[t0])
        try:
            tz_gate = float(gate_by_ts.loc[et, "taker_z"]) if et in gate_by_ts.index else np.nan
        except Exception:
            tz_gate = np.nan

        # 4. PREVIOUS 4h candle (last completed before current candle opened)
        if idx_curr > 0:
            ts_prev = df_4h.loc[idx_curr - 1, "timestamp"]
            # Get gate_zscores at the moment the previous 4h candle was active
            # (which is at ts_curr - 1h, i.e., the hour before current 4h started)
            ts_prev_read = ts_curr - pd.Timedelta(hours=1)
            mask_prev = df_gate["timestamp"] <= ts_prev_read
            if mask_prev.any():
                tz_prev_4h = float(df_gate[mask_prev].iloc[-1]["taker_z"])
            else:
                tz_prev_4h = np.nan
        else:
            tz_prev_4h = np.nan
            ts_prev = pd.NaT

        # 5. gate taker_z at T-1h (lag 1 candle — production-safe)
        ts_lag1 = et - pd.Timedelta(hours=1)
        mask_lag = df_gate["timestamp"] <= ts_lag1
        tz_lag1h = float(df_gate[mask_lag].iloc[-1]["taker_z"]) if mask_lag.any() else np.nan

        records.append({
            "entry_time": et,
            "return_pct": row["return_pct"],
            "is_loser": row["is_loser"],
            "taker_z_t0": tz_t0,
            "taker_z_gate_t0": tz_gate,
            "ts_curr_4h": ts_curr,
            "hours_into_4h_candle": round(hours_into_candle, 1),
            "ts_prev_4h": ts_prev,
            "taker_z_prev_4h": tz_prev_4h,
            "taker_z_lag1h": tz_lag1h,
        })

    return pd.DataFrame(records)


# ==================================================================
# METRICS
# ==================================================================

def compute_metrics(returns: np.ndarray) -> dict:
    r = np.asarray([x for x in returns if not np.isnan(x)])
    if len(r) == 0:
        return dict(n=0, wr=0, expectancy=0, sharpe=0, total_ret=0, max_dd=0)

    wins = r[r > 0]
    wr = len(wins) / len(r) * 100
    std = r.std()
    sharpe = (r.mean() / std) * np.sqrt(52) if std > 0 else 0.0
    cum = np.cumprod(1 + r)
    max_dd = ((cum - np.maximum.accumulate(cum)) / np.maximum.accumulate(cum)).min() * 100
    total = (np.prod(1 + r) - 1) * 100

    return dict(n=len(r), wr=round(wr, 1), expectancy=round(r.mean() * 100, 4),
                sharpe=round(sharpe, 3), total_ret=round(total, 2),
                max_dd=round(max_dd, 2))


def apply_filter_on_col(df: pd.DataFrame, taker_col: str, threshold: float) -> dict:
    valid = df.dropna(subset=[taker_col])
    mask = valid[taker_col] < threshold
    kept = valid[~mask]
    blocked = valid[mask]

    m = compute_metrics(kept["return_pct"].values)
    n_bl = len(blocked)
    n_bl_losers = int(blocked["is_loser"].sum())

    return {
        "col": taker_col,
        "threshold": threshold,
        "n_signals_with_data": len(valid),
        "n_blocked": n_bl,
        "n_blocked_losers": n_bl_losers,
        "n_blocked_winners": n_bl - n_bl_losers,
        **m,
    }


# ==================================================================
# EMPIRICAL LOOK-AHEAD TEST
# ==================================================================

def run_empirical_test(df_aligned: pd.DataFrame):
    """
    Compare filter outcomes using t=0 vs prev_4h vs lag1h.
    If Sharpe drops significantly with lagged data → look-ahead bias confirmed.
    """
    results = {}

    # Baseline (no filter)
    baseline = compute_metrics(df_aligned["return_pct"].values)
    results["BASELINE"] = {**baseline, "n_blocked": 0, "n_blocked_losers": 0}

    for col, label in [
        ("taker_z_t0",       "t=0 (backtest, potencial look-ahead)"),
        ("taker_z_prev_4h",  "prev_4h (produção-safe)"),
        ("taker_z_lag1h",    "lag_1h (produção-safe)"),
    ]:
        r = apply_filter_on_col(df_aligned, col, -1.0)
        results[label] = r

    return results


# ==================================================================
# CAUSAL ANALYSIS: IS taker_z CORRELATED WITH RETURN DIRECTION?
# ==================================================================

def causal_test(df_aligned: pd.DataFrame):
    """
    Check: does taker_z[t=0] correlate with return_pct more than taker_z[prev_4h]?

    If taker_z[t=0] is look-ahead, it should be MORE correlated with outcomes
    than the lag version (which doesn't know the future).
    """
    results = {}
    for col in ["taker_z_t0", "taker_z_prev_4h", "taker_z_lag1h"]:
        valid = df_aligned.dropna(subset=[col])
        if len(valid) < 10:
            continue
        corr = valid[col].corr(valid["return_pct"])
        # Correlation: positive taker (buyers) → positive return (expected)
        results[col] = {
            "corr_with_return": round(float(corr), 4),
            "n": len(valid),
        }
    return results


# ==================================================================
# REPORT
# ==================================================================

def generate_report(df_aligned: pd.DataFrame, filter_results: dict,
                    causal: dict, hours_stats: dict):
    lines = [
        "# Auditoria Look-Ahead Bias — taker_z",
        f"\n**Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        "",
    ]

    # Verdict
    baseline_sharpe = filter_results["BASELINE"]["sharpe"]
    t0_sharpe = filter_results.get("t=0 (backtest, potencial look-ahead)", {}).get("sharpe", 0)
    prev_sharpe = filter_results.get("prev_4h (produção-safe)", {}).get("sharpe", 0)

    delta_t0 = t0_sharpe - baseline_sharpe
    delta_prev = prev_sharpe - baseline_sharpe
    delta_inflation = delta_t0 - delta_prev

    lines += ["## 🎯 Veredito", ""]
    if delta_prev <= 0:
        verdict = "❌ LOOK-AHEAD CONFIRMADO"
        msg = (f"Com dados production-safe (prev_4h), Sharpe={prev_sharpe:.2f} "
               f"({delta_prev:+.2f} vs baseline). O ganho de Sharpe {delta_t0:+.2f} "
               f"visto no Filter Validation desaparece. **Resultado era ilusório.**")
    elif delta_inflation > 0.5:
        verdict = "⚠️ LOOK-AHEAD PARCIAL"
        msg = (f"Com dados production-safe, Sharpe melhora {delta_prev:+.2f} (real). "
               f"Mas o backtest infla em {delta_inflation:.2f} extra (bias). "
               f"Resultado real={prev_sharpe:.2f}, não {t0_sharpe:.2f}.")
    elif delta_inflation < 0.2:
        verdict = "✅ CLEAN"
        msg = (f"Com dados production-safe, Sharpe={prev_sharpe:.2f} ({delta_prev:+.2f}). "
               f"Diferença vs t=0: apenas {delta_inflation:.2f}. Bias é negligenciável. "
               f"**Filtro é robusto.**")
    else:
        verdict = f"⚠️ BIAS LEVE ({delta_inflation:.2f})"
        msg = (f"Algum inflação detectada mas moderada. "
               f"Resultado real ≈ Sharpe {prev_sharpe:.2f}.")

    lines += [f"### {verdict}", "", msg, ""]

    # Temporal structure
    lines += [
        "## 1. Estrutura Temporal: Qual 4h Candle Cada Sinal Usa?",
        "",
        f"- Sinais analisados: {len(df_aligned)}",
    ]
    for k, v in hours_stats.items():
        lines.append(f"- {k}: {v}")
    lines.append("")

    lines += [
        "## 2. Comparação de Filtros: t=0 vs Production-Safe",
        "",
        "| Versão | N kept | WR% | Sharpe | Expectância | Max DD | N bloq | L/W bloq |",
        "|--------|--------|-----|--------|-------------|--------|--------|----------|",
    ]
    for label, r in filter_results.items():
        lines.append(
            f"| {label} | {r['n']} | {r['wr']:.1f}% | {r['sharpe']:.2f} "
            f"| {r['expectancy']:+.3f}% | {r['max_dd']:.1f}% "
            f"| {r.get('n_blocked', 0)} "
            f"| {r.get('n_blocked_losers',0)}L/{r.get('n_blocked_winners',0)}W |"
        )
    lines.append("")

    # Sharpe comparison
    lines += [
        "## 3. Impacto do Look-Ahead no Sharpe",
        "",
        f"| Métrica | Valor |",
        f"|---------|-------|",
        f"| Baseline Sharpe | {baseline_sharpe:.2f} |",
        f"| Filter t=0 Sharpe | {t0_sharpe:.2f} (Δ={delta_t0:+.2f}) |",
        f"| Filter prev_4h Sharpe | {prev_sharpe:.2f} (Δ={delta_prev:+.2f}) |",
        f"| Inflação por look-ahead | {delta_inflation:.2f} |",
        "",
    ]

    # Causal test
    lines += [
        "## 4. Teste Causal: Correlação taker_z vs return_pct",
        "",
        "Se taker_z[t=0] tem correlação MUITO MAIOR com return do que prev_4h,",
        "é evidência de look-ahead (usando futuro para prever futuro).",
        "",
        "| Versão | Corr(taker_z, return_pct) | N |",
        "|--------|--------------------------|---|",
    ]
    for col, c in causal.items():
        lines.append(f"| {col} | {c['corr_with_return']:.4f} | {c['n']} |")
    lines.append("")
    if causal:
        corr_t0 = causal.get("taker_z_t0", {}).get("corr_with_return", 0)
        corr_prev = causal.get("taker_z_prev_4h", {}).get("corr_with_return", 0)
        if abs(corr_t0) > abs(corr_prev) * 1.3:
            lines.append("⚠️ taker_z[t=0] tem correlação muito maior — sinal de look-ahead.")
        else:
            lines.append("✅ Correlações similares — sem sinal de look-ahead por causalidade.")
    lines.append("")

    # Next steps
    lines += [
        "## 5. Próximos Passos",
        "",
    ]
    if "CONFIRMADO" in verdict or "PARCIAL" in verdict or "BIAS LEVE" in verdict:
        real_sharpe = prev_sharpe
        lines += [
            f"1. **Sharpe real do filtro: {real_sharpe:.2f}** (não {t0_sharpe:.2f})",
            f"2. Regerar error_analysis com `taker_z_prev_4h` ao invés de `taker_z_t0`",
            f"3. Regerar filter_validation com dados corrigidos",
            "4. Verificar se other features (funding_z, oi_z) têm o mesmo bias",
            f"5. Sharpe {real_sharpe:.2f} {'ainda vale integrar' if real_sharpe > baseline_sharpe + 0.3 else 'não justifica integração'} (baseline={baseline_sharpe:.2f})",
        ]
    else:
        lines += [
            "1. ✅ Auditoria passou — look-ahead negligenciável",
            f"2. Filtro production-safe tem Sharpe {prev_sharpe:.2f} vs baseline {baseline_sharpe:.2f}",
            "3. Prosseguir com shadow mode / paper monitoring",
        ]

    REPORT_PATH.write_text("\n".join(lines))
    logger.info(f"Report → {REPORT_PATH}")


# ==================================================================
# MAIN
# ==================================================================

def run_audit():
    logger.info("=" * 60)
    logger.info("Auditoria Look-Ahead Bias — taker_z")
    logger.info("=" * 60)

    df_sig, df_4h, df_gate = load_all()

    # Temporal alignment analysis
    logger.info("Analyzing 4h temporal alignment...")
    df_aligned = analyze_4h_alignment(df_sig, df_4h, df_gate)

    # Hours-into-candle stats
    valid_hours = df_aligned["hours_into_4h_candle"].dropna()
    hours_stats = {
        "Média horas no 4h candle ativo na entrada": f"{valid_hours.mean():.1f}h",
        "Distribuição (0=início, 3=fim do candle)": f"min={valid_hours.min():.0f}h max={valid_hours.max():.0f}h",
        "Sinais com taker_z[prev_4h] disponível": int(df_aligned["taker_z_prev_4h"].notna().sum()),
        "Diff média |t0 - prev_4h|": f"{(df_aligned['taker_z_t0'] - df_aligned['taker_z_prev_4h']).abs().mean():.3f}",
        "Diff máxima |t0 - prev_4h|": f"{(df_aligned['taker_z_t0'] - df_aligned['taker_z_prev_4h']).abs().max():.3f}",
    }

    # Print key stats
    print("\nHours into 4h candle distribution:")
    print(valid_hours.value_counts().sort_index().to_string())

    print("\ntaker_z_t0 vs taker_z_prev_4h differences:")
    diff = (df_aligned["taker_z_t0"] - df_aligned["taker_z_prev_4h"]).abs()
    print(f"  mean diff: {diff.mean():.3f}")
    print(f"  % signals where diff > 1.0: {(diff > 1.0).mean()*100:.1f}%")
    print(f"  % signals where filter decision changes (at threshold -1.0):")
    t0_blocked = df_aligned["taker_z_t0"] < -1.0
    prev_blocked = df_aligned["taker_z_prev_4h"] < -1.0
    both_agree = (t0_blocked == prev_blocked).sum()
    print(f"    Agreement: {both_agree}/{len(df_aligned)} ({both_agree/len(df_aligned)*100:.1f}%)")
    print(f"    t0 blocks but prev doesn't: {(t0_blocked & ~prev_blocked).sum()}")
    print(f"    prev blocks but t0 doesn't: {(~t0_blocked & prev_blocked).sum()}")

    # Empirical test
    filter_results = run_empirical_test(df_aligned)

    print("\n" + "=" * 60)
    print("FILTER COMPARISON")
    print("=" * 60)
    for label, r in filter_results.items():
        print(f"  {label}")
        print(f"    N={r['n']}, WR={r['wr']:.1f}%, Sharpe={r['sharpe']:.3f}, "
              f"Expect={r['expectancy']:+.3f}%")

    # Causal test
    causal = causal_test(df_aligned)
    print("\nCausal correlations:")
    for col, c in causal.items():
        print(f"  {col}: corr={c['corr_with_return']:.4f}")

    # Generate report
    generate_report(df_aligned, filter_results, causal, hours_stats)

    # Summary
    t0_sharpe = filter_results.get("t=0 (backtest, potencial look-ahead)", {}).get("sharpe", 0)
    prev_sharpe = filter_results.get("prev_4h (produção-safe)", {}).get("sharpe", 0)
    base_sharpe = filter_results["BASELINE"]["sharpe"]
    inflation = t0_sharpe - prev_sharpe

    print("\n" + "=" * 60)
    print("VERDICT")
    print("=" * 60)
    print(f"  Baseline:            Sharpe {base_sharpe:.2f}")
    print(f"  Filter t=0:          Sharpe {t0_sharpe:.2f} (+{t0_sharpe-base_sharpe:.2f})")
    print(f"  Filter prev_4h safe: Sharpe {prev_sharpe:.2f} (+{prev_sharpe-base_sharpe:.2f})")
    print(f"  Look-ahead inflation: {inflation:.2f} Sharpe points")
    if inflation > 0.5:
        print("  ❌ LOOK-AHEAD BIAS DETECTADO")
    elif inflation > 0.2:
        print("  ⚠️ BIAS LEVE")
    else:
        print("  ✅ CLEAN — bias negligenciável")
    print(f"\nReport: {REPORT_PATH}")


if __name__ == "__main__":
    run_audit()
