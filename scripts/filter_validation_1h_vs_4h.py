"""
Comparação: taker_z 4h CoinGlass vs 1h Binance nativo.

Cada sinal recebe:
  taker_z_t0       — 4h CoinGlass t=0 (biased — usado no filter_validation original)
  taker_z_prev_4h  — 4h CoinGlass candle anterior (production-safe)
  taker_z_1h_t0    — 1h Binance candle que contém T (biased)
  taker_z_1h_prev  — 1h Binance candle T-1h (production-safe)

Pergunta central: qual fonte tem melhor Sharpe production-safe?

Outputs:
  prompts/filter_validation_1h_vs_4h_report.md
  prompts/tables/filter_validation_1h_vs_4h.csv
  prompts/plots/filter_1h_vs_4h/
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
logger = logging.getLogger("filter_1h_vs_4h")

PLOTS_DIR = ROOT / "prompts/plots/filter_1h_vs_4h"
TABLES_DIR = ROOT / "prompts/tables"
REPORT_PATH = ROOT / "prompts/filter_validation_1h_vs_4h_report.md"
SCENARIOS_CSV = TABLES_DIR / "filter_validation_1h_vs_4h.csv"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)


# ==================================================================
# LOAD + ENRICH
# ==================================================================

def load_and_enrich() -> pd.DataFrame:
    ea_path = ROOT / "prompts/tables/error_analysis_full_dataset.csv"
    if not ea_path.exists():
        raise FileNotFoundError(f"Run error_analysis_losers.py first: {ea_path}")

    df_sig = pd.read_csv(ea_path)
    df_sig["entry_time"] = pd.to_datetime(df_sig["entry_time"], utc=True)
    df_sig["is_loser"] = ~df_sig["is_winner"]
    df_sig = df_sig.sort_values("entry_time").reset_index(drop=True)
    logger.info(f"Signals: {len(df_sig)} | WR={df_sig['is_winner'].mean()*100:.1f}%")

    # Raw 4h taker
    t4_path = ROOT / "data/01_raw/futures/taker_4h.parquet"
    df_4h = pd.read_parquet(t4_path)
    df_4h["timestamp"] = pd.to_datetime(df_4h["timestamp"], utc=True)
    df_4h = df_4h.sort_values("timestamp").reset_index(drop=True)
    logger.info(f"taker_4h: {len(df_4h)} rows | {df_4h['timestamp'].min().date()} → {df_4h['timestamp'].max().date()}")

    # Gate z-scores (has taker_z_1h now)
    gz_path = ROOT / "data/02_features/gate_zscores.parquet"
    df_gate = pd.read_parquet(gz_path)
    df_gate["timestamp"] = pd.to_datetime(df_gate["timestamp"], utc=True)
    df_gate = df_gate.sort_values("timestamp").reset_index(drop=True)
    logger.info(f"gate_zscores cols: {df_gate.columns.tolist()}")

    gate_by_ts = df_gate.set_index("timestamp")

    records = []
    for _, row in df_sig.iterrows():
        et = row["entry_time"]
        rec = {
            "entry_time": et,
            "return_pct": row["return_pct"],
            "is_loser": row["is_loser"],
            "taker_z_t0": row["taker_z"],    # 4h CoinGlass biased (original dataset)
        }

        # ── 4h CoinGlass prev (production-safe) ──────────────────────────
        mask_4h = df_4h["timestamp"] <= et
        if mask_4h.any():
            idx_curr = df_4h[mask_4h].index[-1]
            ts_curr = df_4h.loc[idx_curr, "timestamp"]
            if idx_curr > 0:
                ts_prev_read = ts_curr - pd.Timedelta(hours=1)
                mask_prev = df_gate["timestamp"] <= ts_prev_read
                rec["taker_z_prev_4h"] = float(df_gate[mask_prev].iloc[-1]["taker_z"]) \
                    if mask_prev.any() else np.nan
            else:
                rec["taker_z_prev_4h"] = np.nan
        else:
            rec["taker_z_prev_4h"] = np.nan

        # ── 1h Binance t=0 (biased) ──────────────────────────────────────
        if et in gate_by_ts.index and "taker_z_1h" in gate_by_ts.columns:
            rec["taker_z_1h_t0"] = float(gate_by_ts.loc[et, "taker_z_1h"])
        else:
            # nearest 1h candle at or before entry_time
            mask_1h = df_gate["timestamp"] <= et
            if mask_1h.any() and "taker_z_1h" in df_gate.columns:
                rec["taker_z_1h_t0"] = float(df_gate[mask_1h].iloc[-1]["taker_z_1h"])
            else:
                rec["taker_z_1h_t0"] = np.nan

        # ── 1h Binance prev T-1h (production-safe) ───────────────────────
        ts_lag1 = et - pd.Timedelta(hours=1)
        mask_lag = df_gate["timestamp"] <= ts_lag1
        if mask_lag.any() and "taker_z_1h" in df_gate.columns:
            rec["taker_z_1h_prev"] = float(df_gate[mask_lag].iloc[-1]["taker_z_1h"])
        else:
            rec["taker_z_1h_prev"] = np.nan

        records.append(rec)

    df = pd.DataFrame(records)

    # Coverage check
    for col in ["taker_z_t0", "taker_z_prev_4h", "taker_z_1h_t0", "taker_z_1h_prev"]:
        nn = df[col].notna().sum()
        logger.info(f"  {col}: {nn}/{len(df)} non-null")

    return df


# ==================================================================
# METRICS
# ==================================================================

def compute_metrics(returns: np.ndarray, tpy: float = 52.0) -> dict:
    r = np.asarray([x for x in returns if x is not None and not np.isnan(x)])
    if len(r) == 0:
        return dict(n=0, wr=0, expect=0, sharpe=0, total_ret=0, max_dd=0, pf=0)
    wins = r[r > 0]
    losses = r[r < 0]
    wr = len(wins) / len(r) * 100
    std = r.std()
    sharpe = (r.mean() / std) * np.sqrt(tpy) if std > 0 else 0.0
    pf = wins.sum() / abs(losses.sum()) if losses.sum() != 0 else float("inf")
    cum = np.cumprod(1 + r)
    max_dd = ((cum - np.maximum.accumulate(cum)) / np.maximum.accumulate(cum)).min() * 100
    return dict(
        n=len(r),
        wr=round(wr, 1),
        expect=round(r.mean() * 100, 4),
        sharpe=round(sharpe, 3),
        total_ret=round((np.prod(1 + r) - 1) * 100, 2),
        max_dd=round(max_dd, 2),
        pf=round(pf, 3),
    )


def apply_filter(df: pd.DataFrame, col: str, threshold: float, label: str) -> dict:
    valid = df.dropna(subset=[col])
    mask = valid[col] < threshold
    kept = valid[~mask]
    blocked = valid[mask]
    m = compute_metrics(kept["return_pct"].values)
    return {
        "scenario": label,
        "col": col,
        "threshold": threshold,
        "n_blocked": len(blocked),
        "n_blocked_losers": int(blocked["is_loser"].sum()),
        "n_blocked_winners": len(blocked) - int(blocked["is_loser"].sum()),
        **m,
    }


# ==================================================================
# CENÁRIOS
# ==================================================================

def run_scenarios(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    # Baseline
    m = compute_metrics(df["return_pct"].values)
    rows.append({"scenario": "BASELINE", "col": "—", "threshold": None,
                 "n_blocked": 0, "n_blocked_losers": 0, "n_blocked_winners": 0, **m})

    # 4h CoinGlass t=0 (biased — reference from original study)
    for t in [-0.5, -0.8, -1.0, -1.5]:
        rows.append(apply_filter(df, "taker_z_t0", t, f"4h CoinGlass t=0 BIASED < {t}"))

    # 4h CoinGlass prev (production-safe)
    for t in [-0.5, -0.8, -1.0, -1.5]:
        rows.append(apply_filter(df, "taker_z_prev_4h", t, f"4h CoinGlass prev SAFE < {t}"))

    # 1h Binance t=0 (biased)
    for t in [-0.5, -0.8, -1.0, -1.5]:
        rows.append(apply_filter(df, "taker_z_1h_t0", t, f"1h Binance t=0 BIASED < {t}"))

    # 1h Binance prev (production-safe)
    for t in [-0.5, -0.8, -1.0, -1.5]:
        rows.append(apply_filter(df, "taker_z_1h_prev", t, f"1h Binance prev SAFE < {t}"))

    return pd.DataFrame(rows)


# ==================================================================
# SUBPERIODS (tercis)
# ==================================================================

def test_subperiods(df: pd.DataFrame) -> pd.DataFrame:
    n = len(df)
    t = n // 3
    rows = []
    for p_name, sub in [("T1", df.iloc[:t]), ("T2", df.iloc[t:2*t]), ("T3", df.iloc[2*t:])]:
        m = compute_metrics(sub["return_pct"].values)
        rows.append({"period": p_name, "scenario": "BASELINE", **m, "n_blocked": 0})

        for col, label in [("taker_z_prev_4h", "4h SAFE -1.0"),
                           ("taker_z_1h_prev", "1h SAFE -1.0")]:
            valid = sub.dropna(subset=[col])
            mask = valid[col] < -1.0
            m = compute_metrics(valid[~mask]["return_pct"].values)
            rows.append({"period": p_name, "scenario": f"{label}",
                         **m, "n_blocked": int(mask.sum())})

    return pd.DataFrame(rows)


# ==================================================================
# CAUSAL CORRELATION TEST
# ==================================================================

def causal_test(df: pd.DataFrame) -> dict:
    results = {}
    for col in ["taker_z_t0", "taker_z_prev_4h", "taker_z_1h_t0", "taker_z_1h_prev"]:
        valid = df.dropna(subset=[col])
        if len(valid) < 10:
            continue
        corr = valid[col].corr(valid["return_pct"])
        results[col] = {"corr": round(float(corr), 4), "n": len(valid)}
    return results


# ==================================================================
# PLOTS
# ==================================================================

def generate_plots(df: pd.DataFrame, df_sc: pd.DataFrame):
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Plot 1: Sharpe comparison (SAFE scenarios only + baseline) ────────
    safe = df_sc[
        df_sc["scenario"].str.contains("SAFE|BASELINE")
    ].copy()

    fig, ax = plt.subplots(figsize=(14, 6))
    colors = []
    for s in safe["scenario"]:
        if "BASELINE" in s:
            colors.append("gray")
        elif "4h" in s:
            colors.append("#1f77b4")
        else:
            colors.append("#2ca02c")

    bars = ax.barh(safe["scenario"], safe["sharpe"], color=colors, alpha=0.78)
    baseline_sh = df_sc[df_sc["scenario"] == "BASELINE"]["sharpe"].iloc[0]
    ax.axvline(baseline_sh, color="red", linestyle="--", lw=1.5, label=f"Baseline {baseline_sh:.2f}")
    ax.set_xlabel("Sharpe")
    ax.set_title("Sharpe — Production-Safe: 4h CoinGlass vs 1h Binance")

    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(color="gray", label="Baseline"),
        Patch(color="#1f77b4", label="4h CoinGlass prev (safe)"),
        Patch(color="#2ca02c", label="1h Binance prev (safe)"),
        plt.Line2D([0], [0], color="red", lw=1.5, ls="--", label="Baseline Sharpe"),
    ])
    ax.grid(alpha=0.3, axis="x")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "sharpe_1h_vs_4h.png", dpi=100, bbox_inches="tight")
    plt.close()

    # ── Plot 2: threshold sensitivity — both safe ─────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, suffix, color, label in [
        (axes[0], "4h CoinGlass prev SAFE", "#1f77b4", "4h CoinGlass prev"),
        (axes[1], "1h Binance prev SAFE", "#2ca02c", "1h Binance prev"),
    ]:
        subset = df_sc[df_sc["scenario"].str.contains(re.escape(suffix))].copy()
        subset["threshold"] = subset["threshold"].astype(float)
        subset = subset.sort_values("threshold")
        ax.plot(subset["threshold"], subset["sharpe"], "o-", color=color, lw=2, ms=9)
        ax.axhline(baseline_sh, color="red", ls="--", lw=1.5, label="Baseline")
        ax.set_xlabel("Threshold")
        ax.set_ylabel("Sharpe")
        ax.set_title(f"Threshold Sensitivity — {label}")
        ax.legend()
        ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "threshold_sensitivity.png", dpi=100, bbox_inches="tight")
    plt.close()

    # ── Plot 3: Scatter 4h vs 1h ─────────────────────────────────────────
    both = df.dropna(subset=["taker_z_prev_4h", "taker_z_1h_prev"])
    if len(both) > 5:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        cs = both["is_loser"].map({True: "red", False: "green"})
        axes[0].scatter(both["taker_z_prev_4h"], both["taker_z_1h_prev"],
                        alpha=0.65, s=45, c=cs)
        axes[0].plot([-4, 4], [-4, 4], "k--", alpha=0.25)
        axes[0].axhline(-1.0, color="black", linestyle=":", alpha=0.4)
        axes[0].axvline(-1.0, color="black", linestyle=":", alpha=0.4)
        corr = both[["taker_z_prev_4h", "taker_z_1h_prev"]].corr().iloc[0, 1]
        axes[0].text(0.05, 0.95, f"Corr: {corr:.3f}",
                    transform=axes[0].transAxes, va="top",
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
        axes[0].set_xlabel("taker_z 4h CoinGlass prev (safe)")
        axes[0].set_ylabel("taker_z 1h Binance prev (safe)")
        axes[0].set_title("Correlação: 4h vs 1h (production-safe)")
        axes[0].grid(alpha=0.3)

        axes[1].hist(both[both["is_loser"]]["taker_z_1h_prev"], bins=25,
                    alpha=0.6, color="red", label="Losers", density=True)
        axes[1].hist(both[~both["is_loser"]]["taker_z_1h_prev"], bins=25,
                    alpha=0.6, color="green", label="Winners", density=True)
        axes[1].axvline(-1.0, color="black", linestyle="--", label="-1.0 threshold")
        axes[1].set_xlabel("taker_z 1h Binance prev")
        axes[1].set_title("Distribuição: 1h Binance prev — Losers vs Winners")
        axes[1].legend()
        axes[1].grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "correlation_4h_vs_1h.png", dpi=100, bbox_inches="tight")
        plt.close()

    # ── Plot 4: Equity curves ─────────────────────────────────────────────
    d = df.sort_values("entry_time").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(14, 6))
    xi = np.arange(len(d))

    ax.plot(xi, (np.cumprod(1 + d["return_pct"]) - 1) * 100,
            color="red", lw=2, label="Baseline")

    for col, t, label, color in [
        ("taker_z_prev_4h", -1.0, "4h CoinGlass prev SAFE -1.0", "#1f77b4"),
        ("taker_z_1h_prev", -1.0, "1h Binance prev SAFE -1.0", "#2ca02c"),
        ("taker_z_t0",      -1.0, "4h CoinGlass t=0 BIASED -1.0", "#aec7e8"),
    ]:
        valid = d.copy()
        valid.loc[valid[col].isna() | (valid[col] >= t), "filtered_ret"] = valid["return_pct"]
        valid.loc[valid[col].notna() & (valid[col] < t), "filtered_ret"] = 0
        ax.plot(xi, (np.cumprod(1 + valid["filtered_ret"].fillna(valid["return_pct"])) - 1) * 100,
                lw=2, label=label, color=color)

    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel("Trade #")
    ax.set_ylabel("Retorno Acumulado %")
    ax.set_title("Equity Curves: Baseline vs Filtros (4h e 1h)")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "equity_curves.png", dpi=100, bbox_inches="tight")
    plt.close()

    logger.info(f"Plots → {PLOTS_DIR}")


# ==================================================================
# REPORT
# ==================================================================

def generate_report(df: pd.DataFrame, df_sc: pd.DataFrame,
                    df_sub: pd.DataFrame, causal: dict):
    baseline = df_sc[df_sc["scenario"] == "BASELINE"].iloc[0]
    base_sh = baseline["sharpe"]

    def best(suffix):
        s = df_sc[df_sc["scenario"].str.contains(re.escape(suffix))]
        return s.loc[s["sharpe"].idxmax()] if len(s) else None

    b4h_safe = best("4h CoinGlass prev SAFE")
    b1h_safe = best("1h Binance prev SAFE")
    b4h_bias = best("4h CoinGlass t=0 BIASED")

    lines = [
        "# Filter Validation 1h vs 4h — Comparação Final",
        f"\n**Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        f"**Dataset:** {len(df)} trades",
        "",
        "## 1. Todos os Cenários",
        "",
        "| Cenário | N kept | WR% | Expect | Sharpe | Total Ret% | Max DD% | Bloq |",
        "|---------|--------|-----|--------|--------|------------|---------|------|",
    ]
    for _, row in df_sc.iterrows():
        lines.append(
            f"| {row['scenario']} | {row['n']} | {row['wr']:.1f}% | "
            f"{row['expect']:+.3f}% | {row['sharpe']:.2f} | "
            f"{row['total_ret']:+.2f}% | {row['max_dd']:.1f}% | "
            f"{int(row.get('n_blocked', 0))} |"
        )

    lines += ["", "## 2. Resumo Executivo", ""]
    lines.append(f"| Métrica | 4h Biased | 4h SAFE | 1h SAFE |")
    lines.append(f"|---------|-----------|---------|---------|")

    def fmt_row(label, attr):
        v4b = getattr(b4h_bias, attr, 0) if b4h_bias is not None else 0
        v4s = getattr(b4h_safe, attr, 0) if b4h_safe is not None else 0
        v1s = getattr(b1h_safe, attr, 0) if b1h_safe is not None else 0
        return f"| {label} | {v4b:.2f} | {v4s:.2f} | {v1s:.2f} |"

    for label, attr in [("Best Sharpe", "sharpe"), ("N kept", "n"),
                         ("WR%", "wr"), ("Expect%", "expect")]:
        v4b = b4h_bias[attr] if b4h_bias is not None else 0
        v4s = b4h_safe[attr] if b4h_safe is not None else 0
        v1s = b1h_safe[attr] if b1h_safe is not None else 0
        lines.append(f"| {label} | {v4b:.2f} | {v4s:.2f} | {v1s:.2f} |")

    lines += ["", "## 3. Causal Correlation Test", ""]
    lines.append("| Versão | Corr(taker_z, return_pct) | N |")
    lines.append("|--------|--------------------------|---|")
    for col, c in causal.items():
        lines.append(f"| {col} | {c['corr']:.4f} | {c['n']} |")

    if causal:
        corr_t0 = causal.get("taker_z_t0", {}).get("corr", 0)
        corr_1h_prev = causal.get("taker_z_1h_prev", {}).get("corr", 0)
        corr_4h_prev = causal.get("taker_z_prev_4h", {}).get("corr", 0)
        lines.append("")
        if abs(corr_t0) > abs(corr_4h_prev) * 1.3:
            lines.append("⚠️ taker_z_t0 tem correlação muito maior — look-ahead no 4h.")
        else:
            lines.append("✅ Correlações similares — look-ahead negligenciável no 4h.")
        if abs(corr_1h_prev) > 0.05:
            lines.append(f"✅ taker_z_1h_prev correlaciona com retorno: {corr_1h_prev:.4f}")

    lines += ["", "## 4. Sub-Períodos (Robustez)", ""]
    lines.append("| Período | Cenário | N | WR% | Sharpe | Bloq |")
    lines.append("|---------|---------|---|-----|--------|------|")
    for _, row in df_sub.iterrows():
        lines.append(
            f"| {row['period']} | {row['scenario']} | {row['n']} | "
            f"{row['wr']:.1f}% | {row['sharpe']:.2f} | {row.get('n_blocked', 0)} |"
        )

    # Veredicto
    lines += ["", "## 5. 🎯 Veredito", ""]

    sh_4h = b4h_safe["sharpe"] if b4h_safe is not None else 0
    sh_1h = b1h_safe["sharpe"] if b1h_safe is not None else 0

    inflation_4h = (b4h_bias["sharpe"] if b4h_bias is not None else 0) - sh_4h

    lines.append(f"**Baseline:** Sharpe {base_sh:.2f}")
    lines.append(f"**4h CoinGlass biased (t=0):** Sharpe {b4h_bias['sharpe']:.2f} (inflação: {inflation_4h:+.2f})")
    lines.append(f"**4h CoinGlass prev (safe):** Sharpe {sh_4h:.2f} (Δ vs baseline: {sh_4h-base_sh:+.2f})")
    lines.append(f"**1h Binance prev (safe):** Sharpe {sh_1h:.2f} (Δ vs baseline: {sh_1h-base_sh:+.2f})")
    lines.append("")

    diff_1h_vs_4h = sh_1h - sh_4h

    if diff_1h_vs_4h > 0.3:
        verdict = "✅ BINANCE 1H SUPERIOR"
        msg = (f"1h Binance prev bate 4h CoinGlass prev por {diff_1h_vs_4h:+.2f} Sharpe. "
               f"Latência menor, fonte própria. **Migrar para 1h em produção.**")
    elif diff_1h_vs_4h > -0.2:
        verdict = "⚖️ EMPATE — manter 4h CoinGlass"
        msg = (f"Diff 1h vs 4h: {diff_1h_vs_4h:+.2f} (não significativo). "
               f"4h CoinGlass tem agregação multi-exchange. **Manter arquitetura atual.**")
    else:
        verdict = "🔴 4H COINGLASS SUPERIOR"
        msg = (f"1h Binance prev é {diff_1h_vs_4h:.2f} Sharpe abaixo do 4h. "
               f"Agregação multi-exchange do CoinGlass é crítica. **Manter 4h.**")

    lines += [f"### {verdict}", "", msg, "",
              "## 6. Visualizações", "",
              f"- `{PLOTS_DIR}/sharpe_1h_vs_4h.png`",
              f"- `{PLOTS_DIR}/threshold_sensitivity.png`",
              f"- `{PLOTS_DIR}/correlation_4h_vs_1h.png`",
              f"- `{PLOTS_DIR}/equity_curves.png`"]

    REPORT_PATH.write_text("\n".join(lines))
    logger.info(f"Report → {REPORT_PATH}")


# ==================================================================
# MAIN
# ==================================================================

def run():
    import re
    globals()["re"] = re

    logger.info("=" * 60)
    logger.info("Filter Validation 1h vs 4h")
    logger.info("=" * 60)

    df = load_and_enrich()
    df_sc = run_scenarios(df)
    df_sc.to_csv(SCENARIOS_CSV, index=False)

    df_sub = test_subperiods(df)
    causal = causal_test(df)

    print("\n" + "=" * 70)
    print("SCENARIOS")
    print("=" * 70)
    cols = ["scenario", "n", "wr", "expect", "sharpe", "total_ret", "n_blocked"]
    print(df_sc[cols].to_string(index=False))

    print("\n" + "=" * 70)
    print("SUBPERIODS")
    print("=" * 70)
    print(df_sub[["period", "scenario", "n", "wr", "sharpe", "n_blocked"]].to_string(index=False))

    print("\n" + "=" * 70)
    print("CAUSAL CORRELATIONS")
    print("=" * 70)
    for col, c in causal.items():
        print(f"  {col}: corr={c['corr']:.4f} (n={c['n']})")

    generate_plots(df, df_sc)
    generate_report(df, df_sc, df_sub, causal)

    logger.info(f"\nDone. Report: {REPORT_PATH}")


if __name__ == "__main__":
    run()
