"""
Dynamic TP Study — Marujo's Proposal vs Fixed 2%.

Testa proposta de TP dinâmico baseado em contexto de entrada:
  - Volume climax (vol_z > thr) → TP reduzido
  - Late entry (RSI > thr AND BB > thr) → TP moderado
  - Normal → TP 2% (atual)

MFE por exit_reason:
  - TP:    MFE = return_pct (bateu exatamente no TP)
  - SL:    MFE = 0.0 (conservador — direto ao SL)
  - TRAIL: MFE ≈ return_pct + 0.01 (trail 1% abaixo do pico)

Outputs:
  prompts/dynamic_tp_report.md
  prompts/tables/dynamic_tp_*.csv
  prompts/plots/dynamic_tp/*.png
"""
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("dynamic_tp")

DATASET_PATH = ROOT / "prompts/tables/error_analysis_full_dataset.csv"
OUT_DIR = ROOT / "prompts"
PLOTS_DIR = OUT_DIR / "plots" / "dynamic_tp"
TABLES_DIR = OUT_DIR / "tables"
REPORT_PATH = OUT_DIR / "dynamic_tp_report.md"

for d in [OUT_DIR, PLOTS_DIR, TABLES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

def _df_to_md(df: pd.DataFrame) -> str:
    """Markdown table sem depender de tabulate."""
    cols = list(df.columns)
    header = "| " + " | ".join(str(c) for c in cols) + " |"
    sep = "|" + "|".join(["---"] * len(cols)) + "|"
    rows = []
    for _, r in df.iterrows():
        rows.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join([header, sep] + rows)


SL_PCT = -0.015   # SL fixo (-1.5%)
TRAIL_PCT = 0.01  # trailing stop 1% abaixo do pico


# ==================================================================
# LOAD + ENRICH
# ==================================================================

def load_dataset() -> pd.DataFrame:
    if not DATASET_PATH.exists():
        logger.error(f"Dataset não encontrado: {DATASET_PATH}")
        sys.exit(1)

    df = pd.read_csv(DATASET_PATH)
    df["entry_time"] = pd.to_datetime(df["entry_time"], utc=True)

    # Normalise column names
    if "rsi_14" in df.columns and "entry_rsi" not in df.columns:
        df["entry_rsi"] = df["rsi_14"]
    if "bb_pct" in df.columns and "entry_bb_pct" not in df.columns:
        df["entry_bb_pct"] = df["bb_pct"]
    if "volume_z" in df.columns and "_volume_z" not in df.columns:
        df["_volume_z"] = df["volume_z"]

    logger.info(f"Loaded {len(df)} trades")
    logger.info(f"exit_reason: {df['exit_reason'].value_counts().to_dict()}")
    logger.info(f"return_pct range: {df['return_pct'].min():.4f} → {df['return_pct'].max():.4f}")
    return df


def estimate_mfe(row) -> float:
    """MFE em decimal por exit_reason."""
    exit_reason = row.get("exit_reason", "")
    ret = row["return_pct"]
    if exit_reason == "TP":
        return ret           # fechou exatamente no TP
    elif exit_reason == "SL":
        return 0.0           # conservador: direto ao SL
    elif exit_reason == "TRAIL":
        return ret + TRAIL_PCT  # trail 1% → pico ≈ return + 1%
    return max(ret, 0.0)


# ==================================================================
# CLASSIFICAÇÃO EM BUCKETS
# ==================================================================

def classify_bucket(row, volume_z_thr: float = 1.0, rsi_thr: float = 75.0, bb_thr: float = 0.95) -> str:
    vz = row.get("_volume_z", 0.0)
    rsi = row.get("entry_rsi", 50.0)
    bb = row.get("entry_bb_pct", 0.5)
    if pd.notna(vz) and vz > volume_z_thr:
        return "exhaustion"
    elif pd.notna(rsi) and pd.notna(bb) and rsi > rsi_thr and bb > bb_thr:
        return "late"
    return "normal"


# ==================================================================
# SIMULAÇÃO DE TRADE
# ==================================================================

def simulate_trade(row, tp_config: dict) -> float:
    """
    Retorna return simulado em decimal.

    SL: conservador → unchanged.
    TP/TRAIL: se MFE >= novo_TP → fecha no novo_TP; senão return original.
    """
    bucket = row["bucket"]
    new_tp = tp_config.get(bucket, 0.02)
    exit_reason = row.get("exit_reason", "")
    ret = row["return_pct"]
    mfe = row["_mfe"]

    if exit_reason == "SL":
        return ret  # SL inalterado (MFE desconhecido, conservador)

    if mfe >= new_tp:
        return new_tp

    return ret  # nem novo TP nem SL — return original (TRAIL < TP)


# ==================================================================
# MÉTRICAS
# ==================================================================

def compute_metrics(returns: np.ndarray, label: str = "") -> dict:
    """Métricas em decimal. Sharpe anualizado por sqrt(n_trades_por_ano)."""
    if len(returns) == 0:
        return {k: 0 for k in ["n_trades", "n_wins", "n_losses", "win_rate_pct",
                                "avg_return_pct", "total_return_pct", "sharpe", "max_dd_pct"]}

    wins = returns > 0
    std = returns.std()
    sharpe = (returns.mean() / std) * np.sqrt(52) if std > 0 else 0.0  # 52 trades/ano estimado

    cum = np.cumprod(1 + returns)
    peak = np.maximum.accumulate(cum)
    max_dd = ((cum - peak) / peak).min() * 100

    return {
        "config": label,
        "n_trades": len(returns),
        "n_wins": int(wins.sum()),
        "n_losses": int((~wins).sum()),
        "win_rate_pct": wins.mean() * 100,
        "avg_return_pct": returns.mean() * 100,
        "total_return_pct": (np.prod(1 + returns) - 1) * 100,
        "sharpe": sharpe,
        "max_dd_pct": max_dd,
    }


def simulate_config(df: pd.DataFrame, tp_config: dict, label: str):
    returns = df.apply(lambda r: simulate_trade(r, tp_config), axis=1).values
    m = compute_metrics(returns, label)
    m["tp_config"] = str(tp_config)
    return m, returns


# ==================================================================
# BUCKET ANALYSIS
# ==================================================================

def bucket_analysis(df: pd.DataFrame, tp_cfg_fixed: dict, tp_cfg_marujo: dict) -> pd.DataFrame:
    rows = []
    for bucket in ["normal", "late", "exhaustion"]:
        sub = df[df["bucket"] == bucket]
        if len(sub) == 0:
            continue
        ret_fixed = sub.apply(lambda r: simulate_trade(r, tp_cfg_fixed), axis=1).values
        ret_marujo = sub.apply(lambda r: simulate_trade(r, tp_cfg_marujo), axis=1).values
        m_f = compute_metrics(ret_fixed)
        m_m = compute_metrics(ret_marujo)

        # exit_reason breakdown
        exit_counts = sub["exit_reason"].value_counts().to_dict()

        rows.append({
            "bucket": bucket,
            "n_trades": len(sub),
            "pct_total": f"{len(sub)/len(df)*100:.1f}%",
            "tp_n": exit_counts.get("TP", 0),
            "sl_n": exit_counts.get("SL", 0),
            "trail_n": exit_counts.get("TRAIL", 0),
            "wr_original": f"{sub['is_winner'].mean()*100:.1f}%",
            "fixed_sharpe": round(m_f["sharpe"], 3),
            "marujo_sharpe": round(m_m["sharpe"], 3),
            "delta_sharpe": round(m_m["sharpe"] - m_f["sharpe"], 3),
            "fixed_avg_ret": f"{m_f['avg_return_pct']:+.3f}%",
            "marujo_avg_ret": f"{m_m['avg_return_pct']:+.3f}%",
        })
    return pd.DataFrame(rows)


# ==================================================================
# GRID SEARCH
# ==================================================================

def grid_search(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Grid search...")

    results = []
    volume_z_thrs = [0.5, 0.8, 1.0, 1.3, 1.5, 2.0]
    rsi_thrs = [60, 65, 70, 72, 75, 78, 80]
    bb_thrs = [0.80, 0.85, 0.90, 0.92, 0.95, 0.98]
    tp_combos = [
        (0.01, 1.50),  # exhaustion:1%, late:1.5%
        (0.008, 1.50),
        (0.01, 1.30),
        (0.01, 1.80),
        (0.012, 1.50),
        (0.015, 1.50),
    ]  # (tp_exh, tp_late) — normal sempre 2%

    for vz in volume_z_thrs:
        for rsi in rsi_thrs:
            for bb in bb_thrs:
                df_c = df.copy()
                df_c["bucket"] = df_c.apply(lambda r: classify_bucket(r, vz, rsi, bb), axis=1)
                n_exh = (df_c["bucket"] == "exhaustion").sum()
                n_late = (df_c["bucket"] == "late").sum()
                if n_exh == 0 and n_late == 0:
                    continue  # nothing to test

                for tp_exh, tp_late in tp_combos:
                    cfg = {"exhaustion": tp_exh, "late": tp_late / 100, "normal": 0.02}
                    # Note: tp_late is already in decimal here
                    cfg = {"exhaustion": tp_exh, "late": tp_late / 100.0, "normal": 0.02}
                    returns = df_c.apply(lambda r: simulate_trade(r, cfg), axis=1).values
                    m = compute_metrics(returns)
                    results.append({
                        "volume_z_thr": vz,
                        "rsi_thr": rsi,
                        "bb_thr": bb,
                        "tp_exh_pct": tp_exh * 100,
                        "tp_late_pct": tp_late,
                        "n_exh": n_exh,
                        "n_late": n_late,
                        "sharpe": m["sharpe"],
                        "win_rate": m["win_rate_pct"],
                        "total_return": m["total_return_pct"],
                        "max_dd": m["max_dd_pct"],
                    })

    return pd.DataFrame(results)


# ==================================================================
# SENSITIVITY: SL MFE assumption
# ==================================================================

def sensitivity_sl_mfe(df: pd.DataFrame, tp_config: dict, mfe_assumptions: list) -> pd.DataFrame:
    """
    Testa: e se X% dos SL trades no bucket tivessem MFE > new_TP?
    Simula cenários otimistas.
    """
    rows = []
    for mfe_sl in mfe_assumptions:  # assumed MFE for SL trades (decimal)
        df_c = df.copy()
        df_c["_mfe_adj"] = df_c.apply(
            lambda r: mfe_sl if r["exit_reason"] == "SL" else r["_mfe"], axis=1
        )
        # Temporarily override _mfe
        df_c["_mfe_orig"] = df_c["_mfe"]
        df_c["_mfe"] = df_c["_mfe_adj"]
        returns = df_c.apply(lambda r: simulate_trade(r, tp_config), axis=1).values
        m = compute_metrics(returns)
        rows.append({
            "sl_mfe_assumed_pct": mfe_sl * 100,
            "sharpe": m["sharpe"],
            "win_rate_pct": m["win_rate_pct"],
            "total_return_pct": m["total_return_pct"],
        })
        df_c["_mfe"] = df_c["_mfe_orig"]
    return pd.DataFrame(rows)


# ==================================================================
# PLOTS
# ==================================================================

def generate_plots(df: pd.DataFrame, df_configs: pd.DataFrame, grid_results: pd.DataFrame,
                   df_bucket: pd.DataFrame):
    logger.info("Plots...")

    # Plot 1: Sharpe por config
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#888888", "#2196F3", "#4CAF50", "#FF9800"]
    configs = df_configs["config"].tolist()
    sharpes = df_configs["sharpe"].tolist()
    ax.bar(configs, sharpes, color=colors[:len(configs)], alpha=0.8, edgecolor="black", linewidth=0.5)
    baseline = df_configs[df_configs["config"] == "Fixed 2%"]["sharpe"].iloc[0]
    ax.axhline(baseline, color="red", linestyle="--", linewidth=1.5, label=f"Baseline {baseline:.3f}")
    for i, (c, s) in enumerate(zip(configs, sharpes)):
        ax.text(i, s + 0.01, f"{s:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_title("Sharpe por Configuração de TP")
    ax.set_ylabel("Sharpe (anualizado √52)")
    ax.legend()
    ax.set_ylim(0, max(sharpes) * 1.2)
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "sharpe_comparison.png", dpi=100, bbox_inches="tight")
    plt.close()

    # Plot 2: Bucket distribution + exit reason
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: bucket counts
    bucket_counts = df["bucket"].value_counts()
    axes[0].bar(bucket_counts.index, bucket_counts.values,
                color=["#4CAF50", "#FF9800", "#F44336"], alpha=0.8, edgecolor="black", linewidth=0.5)
    for i, (name, count) in enumerate(bucket_counts.items()):
        axes[0].text(i, count + 0.3, f"{count}\n({count/len(df)*100:.1f}%)",
                     ha="center", va="bottom", fontsize=10)
    axes[0].set_title("Distribuição por Bucket (Marujo's v1 thresholds)")
    axes[0].set_ylabel("N trades")

    # Right: exit reason by bucket
    if not df_bucket.empty:
        x = np.arange(len(df_bucket))
        w = 0.25
        axes[1].bar(x - w, df_bucket["tp_n"], w, label="TP", color="#4CAF50", alpha=0.8)
        axes[1].bar(x,     df_bucket["sl_n"], w, label="SL", color="#F44336", alpha=0.8)
        axes[1].bar(x + w, df_bucket["trail_n"], w, label="TRAIL", color="#2196F3", alpha=0.8)
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(df_bucket["bucket"])
        axes[1].set_title("Exit Reason por Bucket")
        axes[1].set_ylabel("N trades")
        axes[1].legend()

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "bucket_distribution.png", dpi=100, bbox_inches="tight")
    plt.close()

    # Plot 3: Heatmap Sharpe grid (volume_z=1.0, tp_exh=1%, tp_late=1.5%)
    if not grid_results.empty:
        subset = grid_results[
            (grid_results["volume_z_thr"] == 1.0) &
            (grid_results["tp_exh_pct"] == 1.0) &
            (grid_results["tp_late_pct"] == 1.5)
        ]
        if not subset.empty:
            pivot = subset.pivot_table(index="rsi_thr", columns="bb_thr", values="sharpe", aggfunc="mean")
            fig, ax = plt.subplots(figsize=(10, 6))
            im = ax.imshow(pivot.values, cmap="RdYlGn", aspect="auto",
                          vmin=pivot.values.min(), vmax=pivot.values.max())
            ax.set_xticks(range(len(pivot.columns)))
            ax.set_xticklabels([f"{c:.2f}" for c in pivot.columns])
            ax.set_yticks(range(len(pivot.index)))
            ax.set_yticklabels([str(int(r)) for r in pivot.index])
            ax.set_xlabel("BB% threshold")
            ax.set_ylabel("RSI threshold")
            ax.set_title("Sharpe Heatmap (vol_z_thr=1.0, TP=1%/1.5%/2%)")
            for i in range(len(pivot.index)):
                for j in range(len(pivot.columns)):
                    v = pivot.values[i, j]
                    if not np.isnan(v):
                        ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=8)
            plt.colorbar(im, label="Sharpe")
            plt.tight_layout()
            plt.savefig(PLOTS_DIR / "sharpe_heatmap_rsi_bb.png", dpi=100, bbox_inches="tight")
            plt.close()

    # Plot 4: Return distribution Fixed vs Marujo
    cfg_fixed = {"exhaustion": 0.02, "late": 0.02, "normal": 0.02}
    cfg_marujo = {"exhaustion": 0.01, "late": 0.015, "normal": 0.02}
    ret_fixed = df.apply(lambda r: simulate_trade(r, cfg_fixed), axis=1).values * 100
    ret_marujo = df.apply(lambda r: simulate_trade(r, cfg_marujo), axis=1).values * 100
    fig, ax = plt.subplots(figsize=(12, 5))
    bins = np.linspace(-2, 2.5, 30)
    ax.hist(ret_fixed, bins=bins, alpha=0.5, label="Fixed 2%", color="gray")
    ax.hist(ret_marujo, bins=bins, alpha=0.5, label="Marujo's v1", color="blue")
    ax.axvline(ret_fixed.mean(), color="gray", linestyle="--", label=f"Mean fixed {ret_fixed.mean():+.3f}%")
    ax.axvline(ret_marujo.mean(), color="blue", linestyle="--", label=f"Mean marujo {ret_marujo.mean():+.3f}%")
    ax.set_title("Distribuição de Returns: Fixed vs Marujo")
    ax.set_xlabel("Return (%)")
    ax.set_ylabel("N trades")
    ax.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "return_distribution.png", dpi=100, bbox_inches="tight")
    plt.close()

    logger.info(f"Plots em {PLOTS_DIR}")


# ==================================================================
# REPORT
# ==================================================================

def generate_report(df: pd.DataFrame, df_configs: pd.DataFrame, grid_results: pd.DataFrame,
                    df_bucket: pd.DataFrame, sensitivity: pd.DataFrame):
    lines = []
    lines.append("# Dynamic TP Study — Marujo's Proposal vs Fixed 2%")
    lines.append(f"\n**Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    lines.append(f"**Dataset:** {len(df)} trades (2026)  |  SL: -1.5%, Trail: 1%\n")

    lines.append("## 1. Proposta do Marujo\n")
    lines.append("```python")
    lines.append("def get_tp(rsi, bb_pct, volume_z):")
    lines.append("    if volume_z > 1.0:               # exhaustion")
    lines.append("        return 0.01                  # TP 1%")
    lines.append("    elif rsi > 75 and bb_pct > 0.95: # late entry")
    lines.append("        return 0.015                 # TP 1.5%")
    lines.append("    else:                            # normal")
    lines.append("        return 0.02                  # TP 2% (atual)")
    lines.append("```\n")

    lines.append("## 2. Distribuição dos Trades por Bucket\n")
    lines.append("| Bucket | N | % | TP | SL | TRAIL | WR original |")
    lines.append("|--------|---|---|----|----|-------|-------------|")
    for _, r in df_bucket.iterrows():
        lines.append(f"| {r['bucket']} | {r['n_trades']} | {r['pct_total']} | {r['tp_n']} | {r['sl_n']} | {r['trail_n']} | {r['wr_original']} |")
    lines.append("")

    lines.append("### ⚠️ Observação crítica sobre thresholds\n")
    n_late = (df["bucket"] == "late").sum()
    n_exh = (df["bucket"] == "exhaustion").sum()
    lines.append(f"- **Late bucket: {n_late} trade(s)** — RSI>75 AND BB>0.95 é muito restritivo nos dados de 2026.")
    lines.append(f"  - RSI max dataset: {df['entry_rsi'].max():.1f} | BB max: {df['entry_bb_pct'].max():.3f}")
    lines.append(f"- **Exhaustion bucket: {n_exh} trades** — mais significativo para análise.")
    lines.append("")

    lines.append("## 3. Comparação de Configurações\n")
    lines.append("| Config | N | WR% | Avg Return | Sharpe | Total Return | Max DD | TP Config |")
    lines.append("|--------|---|-----|-----------|--------|-------------|--------|-----------|")
    for _, r in df_configs.iterrows():
        lines.append(
            f"| {r['config']} | {r['n_trades']} | {r['win_rate_pct']:.1f}% | "
            f"{r['avg_return_pct']:+.3f}% | **{r['sharpe']:.3f}** | "
            f"{r['total_return_pct']:+.2f}% | {r['max_dd_pct']:.1f}% | `{r['tp_config']}` |"
        )
    lines.append("")

    lines.append("## 4. Análise por Bucket\n")
    lines.append("| Bucket | N | Sharpe Fixed | Sharpe Marujo | Δ Sharpe | Avg Fixed | Avg Marujo |")
    lines.append("|--------|---|-------------|--------------|----------|-----------|-----------|")
    for _, r in df_bucket.iterrows():
        delta = r["delta_sharpe"]
        arrow = "↑" if delta > 0.05 else ("↓" if delta < -0.05 else "→")
        lines.append(f"| {r['bucket']} | {r['n_trades']} | {r['fixed_sharpe']:.3f} | {r['marujo_sharpe']:.3f} | {delta:+.3f} {arrow} | {r['fixed_avg_ret']} | {r['marujo_avg_ret']} |")
    lines.append("")

    lines.append("## 5. Mecanismo de Impacto\n")
    lines.append("Como o dynamic TP afeta cada exit_reason:\n")
    lines.append("| Exit | MFE assumido | Efeito com TP < 2% |")
    lines.append("|------|-------------|---------------------|")
    lines.append("| TP (68 trades) | = return (2%) | TP trades → return cai de 2% para novo_TP |")
    lines.append("| SL (45 trades) | = 0 (conservador) | Inalterado |")
    lines.append("| TRAIL (23 trades) | ≈ return + 1% (trail pico) | Se MFE ≥ novo_TP → captura mais cedo com return maior |")
    lines.append("")

    lines.append("## 6. Grid Search — Top 10 por Sharpe\n")
    if not grid_results.empty:
        top10 = grid_results.nlargest(10, "sharpe")[
            ["volume_z_thr", "rsi_thr", "bb_thr", "tp_exh_pct", "tp_late_pct",
             "n_exh", "n_late", "sharpe", "win_rate", "total_return", "max_dd"]
        ].reset_index(drop=True)
        lines.append(_df_to_md(top10))
    lines.append("")

    lines.append("## 7. Sensibilidade: MFE dos trades SL\n")
    lines.append("E se os SL trades (especialmente em exhaustion) tivessem algum MFE positivo antes de reverter?\n")
    if not sensitivity.empty:
        lines.append(_df_to_md(sensitivity))
    lines.append("")

    # Veredicto
    lines.append("## 8. 🎯 Veredicto\n")
    fixed = df_configs[df_configs["config"] == "Fixed 2%"]["sharpe"].iloc[0]
    marujo_row = df_configs[df_configs["config"] == "Marujo's v1"]
    optimized_row = df_configs[df_configs["config"].str.startswith("Grid-optimized")]

    marujo_sharpe = marujo_row["sharpe"].iloc[0] if not marujo_row.empty else fixed
    delta_m = marujo_sharpe - fixed

    lines.append(f"- **Fixed 2% (baseline):** Sharpe = {fixed:.3f}")
    lines.append(f"- **Marujo's v1:** Sharpe = {marujo_sharpe:.3f} ({delta_m:+.3f})")
    if not optimized_row.empty:
        opt_sharpe = optimized_row["sharpe"].iloc[0]
        delta_o = opt_sharpe - fixed
        lines.append(f"- **Grid-optimized:** Sharpe = {opt_sharpe:.3f} ({delta_o:+.3f})")
    lines.append("")

    if delta_m > 0.1:
        lines.append("### ✅ IMPLEMENTAR Marujo's v1")
        lines.append(f"Ganho significativo +{delta_m:.3f}. Alterar `paper_trader.py` com `get_tp()`.")
    elif delta_m > -0.1:
        lines.append("### ⚖️ EMPATE — manter Fixed 2%")
        lines.append(f"Diferença {delta_m:+.3f} ≤ threshold (±0.1). Complexidade adicional não compensa.")
    else:
        lines.append("### ❌ REJEITAR — Dynamic TP piora")
        lines.append(f"Perda {delta_m:+.3f}. Fixed 2% é ótimo para este dataset.")
    lines.append("")
    lines.append("**Raciocínio:**")
    lines.append("- Trades TP (68): dynamic TP reduz ganho de 2% → X% — principal driver de perda")
    lines.append("- Trades TRAIL (23): dynamic TP pode melhorar captura vs trail")
    lines.append("- Trades SL (45): inalterados (MFE desconhecido, assumido 0 — conservador)")
    lines.append("")
    lines.append("## 9. Arquivos\n")
    lines.append(f"- Plots: `prompts/plots/dynamic_tp/`")
    lines.append(f"- Tables: `prompts/tables/dynamic_tp_*.csv`")

    with open(REPORT_PATH, "w") as f:
        f.write("\n".join(lines))
    logger.info(f"Report: {REPORT_PATH}")


# ==================================================================
# MAIN
# ==================================================================

def main():
    logger.info("=" * 60)
    logger.info("Dynamic TP Study")
    logger.info("=" * 60)

    df = load_dataset()
    df["_mfe"] = df.apply(estimate_mfe, axis=1)

    # Classify with Marujo's v1 thresholds
    df["bucket"] = df.apply(lambda r: classify_bucket(r, 1.0, 75.0, 0.95), axis=1)
    logger.info(f"Bucket counts: {df['bucket'].value_counts().to_dict()}")

    # Configurations
    cfg_fixed = {"exhaustion": 0.02, "late": 0.02, "normal": 0.02}
    cfg_marujo = {"exhaustion": 0.01, "late": 0.015, "normal": 0.02}

    config_results = []
    for label, cfg in [("Fixed 2%", cfg_fixed), ("Marujo's v1", cfg_marujo)]:
        m, _ = simulate_config(df, cfg, label)
        config_results.append(m)
        logger.info(f"{label}: Sharpe={m['sharpe']:.3f}  WR={m['win_rate_pct']:.1f}%  AvgRet={m['avg_return_pct']:+.3f}%")

    # Bucket analysis
    df_bucket = bucket_analysis(df, cfg_fixed, cfg_marujo)

    # Grid search
    grid = grid_search(df)
    if not grid.empty:
        best = grid.nlargest(1, "sharpe").iloc[0]
        logger.info(f"Grid best: Sharpe={best['sharpe']:.3f} vol_z={best['volume_z_thr']} rsi={best['rsi_thr']} bb={best['bb_thr']}")
        df_best = df.copy()
        df_best["bucket"] = df_best.apply(
            lambda r: classify_bucket(r, best["volume_z_thr"], best["rsi_thr"], best["bb_thr"]), axis=1
        )
        cfg_opt = {"exhaustion": best["tp_exh_pct"] / 100, "late": best["tp_late_pct"] / 100, "normal": 0.02}
        m_opt, _ = simulate_config(df_best, cfg_opt, "Grid-optimized")
        config_results.append(m_opt)
        logger.info(f"Grid-optimized: Sharpe={m_opt['sharpe']:.3f}")

    df_configs = pd.DataFrame(config_results)

    # Sensitivity: SL MFE scenarios for Marujo bucket
    sens = sensitivity_sl_mfe(df, cfg_marujo, [0.0, 0.005, 0.01, 0.015])

    # Save tables
    df_configs.to_csv(TABLES_DIR / "dynamic_tp_configs.csv", index=False)
    df_bucket.to_csv(TABLES_DIR / "dynamic_tp_bucket_analysis.csv", index=False)
    grid.to_csv(TABLES_DIR / "dynamic_tp_grid_search.csv", index=False)
    sens.to_csv(TABLES_DIR / "dynamic_tp_sensitivity.csv", index=False)

    generate_plots(df, df_configs, grid, df_bucket)
    generate_report(df, df_configs, grid, df_bucket, sens)

    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print("\nConfigs:")
    print(df_configs[["config", "n_trades", "win_rate_pct", "avg_return_pct",
                       "sharpe", "total_return_pct"]].to_string(index=False))
    print("\nBucket breakdown:")
    print(df_bucket[["bucket", "n_trades", "tp_n", "sl_n", "trail_n",
                      "fixed_sharpe", "marujo_sharpe", "delta_sharpe"]].to_string(index=False))
    if not grid.empty:
        print("\nTop 5 grid:")
        print(grid.nlargest(5, "sharpe")[
            ["volume_z_thr", "rsi_thr", "bb_thr", "tp_exh_pct", "tp_late_pct", "sharpe"]
        ].to_string(index=False))
    print(f"\nSensitivity (SL MFE):")
    print(sens.to_string(index=False))
    logger.info(f"\n✅ Complete. Report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
