#!/usr/bin/env python3
"""
Estudo Estatístico MFE/MAE — Bot 2 Entry Signals.

Para cada momento no histórico em que os filtros do Bot 2 teriam disparado,
mede MFE (Maximum Favorable Excursion) e MAE (Maximum Adverse Excursion)
em janelas de 4h, 12h, 24h, 48h, 72h, 120h.

Saída:
  prompts/mfe_mae_study_bot2.md
  prompts/plots/mfe_mae_*.png
  prompts/tables/mfe_mae_raw.csv
  prompts/tables/mfe_mae_percentiles.csv
  prompts/tables/tp_sl_expectancy.csv

Usage:
    python scripts/mfe_mae_study_bot2.py
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
logger = logging.getLogger("mfe_mae")

OUT_DIR = ROOT / "prompts"
PLOTS_DIR = OUT_DIR / "plots"
TABLES_DIR = OUT_DIR / "tables"
REPORT_PATH = OUT_DIR / "mfe_mae_study_bot2.md"

for d in [PLOTS_DIR, TABLES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

TIME_WINDOWS = [4, 12, 24, 48, 72, 120]
PERCENTILES = [10, 25, 50, 75, 90]


def load_data() -> pd.DataFrame:
    spot = pd.read_parquet(ROOT / "data/02_intermediate/spot/btc_1h_clean.parquet")
    spot["timestamp"] = pd.to_datetime(spot["timestamp"], utc=True)
    spot = spot.sort_values("timestamp").reset_index(drop=True)

    spot["ret_1d"] = spot["close"].pct_change(24)

    zs = pd.read_parquet(ROOT / "data/02_features/gate_zscores.parquet")
    zs["timestamp"] = pd.to_datetime(zs["timestamp"], utc=True)

    df = spot.merge(zs[["timestamp", "stablecoin_z"]], on="timestamp", how="left")
    df["stablecoin_z"] = df["stablecoin_z"].ffill()
    df["regime"] = "Unknown"

    df = df[df["timestamp"] >= pd.Timestamp("2026-01-01", tz="UTC")].reset_index(drop=True)
    logger.info(f"Loaded {len(df)} rows: {df['timestamp'].min().date()} → {df['timestamp'].max().date()}")
    return df


def _signal_passes(row) -> bool:
    """Bot 2 current filter (production)."""
    for c in ["stablecoin_z", "ret_1d", "rsi_14", "bb_pct", "close", "ma_21"]:
        if pd.isna(row.get(c)):
            return False
    if row["ret_1d"] > 0.03 and row["rsi_14"] > 65:
        return False
    return (
        row["stablecoin_z"] > 1.3
        and row["ret_1d"] > 0
        and row["rsi_14"] > 50
        and row["close"] > row["ma_21"]
        and row["bb_pct"] < 0.98
    )


def compute_mfe_mae(df: pd.DataFrame, signal_idx: int, window_hours: int) -> dict | None:
    entry_price = df.at[signal_idx, "close"]
    end_idx = min(signal_idx + window_hours, len(df) - 1)
    window = df.iloc[signal_idx + 1 : end_idx + 1]

    if window.empty:
        return None

    max_high_pos = window["high"].values.argmax()
    min_low_pos = window["low"].values.argmin()

    mfe_pct = (window["high"].iloc[max_high_pos] - entry_price) / entry_price
    mae_pct = (window["low"].iloc[min_low_pos] - entry_price) / entry_price

    return {
        "mfe_pct": mfe_pct,
        "mae_pct": mae_pct,
        "time_to_mfe_h": max_high_pos + 1,
        "time_to_mae_h": min_low_pos + 1,
    }


def collect_signals(df: pd.DataFrame) -> pd.DataFrame:
    signals = []
    for i in range(len(df) - max(TIME_WINDOWS)):
        row = df.iloc[i]
        if not _signal_passes(row):
            continue

        rec = {
            "timestamp": row["timestamp"],
            "idx": i,
            "entry_price": float(row["close"]),
            "bb_pct_entry": float(row["bb_pct"]),
            "rsi_entry": float(row["rsi_14"]),
            "ret_1d_entry": float(row["ret_1d"]),
            "stablecoin_z_entry": float(row["stablecoin_z"]),
            "regime": row.get("regime", "Unknown"),
        }

        for w in TIME_WINDOWS:
            result = compute_mfe_mae(df, i, w)
            if result:
                for k, v in result.items():
                    rec[f"{k}_{w}h"] = v

        signals.append(rec)

    return pd.DataFrame(signals)


def compute_percentiles(df_signals: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for w in TIME_WINDOWS:
        mfe_col, mae_col = f"mfe_pct_{w}h", f"mae_pct_{w}h"
        if mfe_col not in df_signals.columns:
            continue
        mfe = df_signals[mfe_col].dropna()
        mae = df_signals[mae_col].dropna()
        row = {"window_hours": w, "n_signals": len(mfe)}
        for p in PERCENTILES:
            row[f"mfe_p{p}"] = float(np.percentile(mfe, p))
            row[f"mae_p{p}"] = float(np.percentile(mae, p))
        rows.append(row)
    return pd.DataFrame(rows)


def compute_conditional_stats(df_signals: pd.DataFrame, col: str, bins, labels) -> pd.DataFrame:
    df = df_signals.copy()
    df["_bin"] = pd.cut(df[col], bins=bins, labels=labels, include_lowest=True)
    rows = []
    for label in labels:
        sub = df[df["_bin"] == label]
        if len(sub) < 5:
            continue
        row = {"condition": str(label), "n": len(sub)}
        for w in [24, 48]:
            mfe_col, mae_col = f"mfe_pct_{w}h", f"mae_pct_{w}h"
            if mfe_col in sub.columns:
                row[f"mfe_p50_{w}h"] = float(sub[mfe_col].median())
                row[f"mae_p50_{w}h"] = float(sub[mae_col].median())
        rows.append(row)
    return pd.DataFrame(rows)


def compute_hit_probabilities(df_signals: pd.DataFrame) -> pd.DataFrame:
    """Grid search TP × SL — para cada par, simula qual veio primeiro em 24h."""
    tp_candidates = [0.003, 0.005, 0.008, 0.010, 0.015, 0.020]
    sl_candidates = [-0.005, -0.008, -0.010, -0.015, -0.020]
    mfe_col = "mfe_pct_24h"
    mae_col = "mae_pct_24h"
    t_mfe_col = "time_to_mfe_h_24h"
    t_mae_col = "time_to_mae_h_24h"

    valid = df_signals[[mfe_col, mae_col, t_mfe_col, t_mae_col]].dropna()

    rows = []
    for tp in tp_candidates:
        for sl in sl_candidates:
            wins = losses = timeouts = 0
            returns = []
            for _, r in valid.iterrows():
                hit_tp = r[mfe_col] >= tp
                hit_sl = r[mae_col] <= sl
                if hit_tp and hit_sl:
                    if r[t_mfe_col] <= r[t_mae_col]:
                        wins += 1
                        returns.append(tp)
                    else:
                        losses += 1
                        returns.append(sl)
                elif hit_tp:
                    wins += 1
                    returns.append(tp)
                elif hit_sl:
                    losses += 1
                    returns.append(sl)
                else:
                    timeouts += 1
                    returns.append(0.0)

            total = wins + losses + timeouts
            if total == 0:
                continue
            rows.append({
                "tp_pct": tp,
                "sl_pct": sl,
                "rr_ratio": round(abs(tp / sl), 2),
                "win_rate": wins / total,
                "loss_rate": losses / total,
                "timeout_rate": timeouts / total,
                "expectancy": float(np.mean(returns)),
                "n": total,
            })

    return pd.DataFrame(rows).sort_values("expectancy", ascending=False).reset_index(drop=True)


def plot_distribution(df_signals: pd.DataFrame):
    mfe = df_signals["mfe_pct_24h"].dropna() * 100
    mae = df_signals["mae_pct_24h"].dropna() * 100

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(mfe, bins=40, color="#4CAF50", alpha=0.75, edgecolor="black", linewidth=0.4)
    for p, ls in [(50, "--"), (75, ":")]:
        v = float(np.percentile(mfe, p))
        axes[0].axvline(v, linestyle=ls, color="red", label=f"P{p}: +{v:.2f}%")
    axes[0].set_title("MFE 24h (Maximum Favorable Excursion)")
    axes[0].set_xlabel("Ganho máximo (%)")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].hist(mae, bins=40, color="#f44336", alpha=0.75, edgecolor="black", linewidth=0.4)
    for p, ls in [(50, "--"), (25, ":")]:
        v = float(np.percentile(mae, p))
        axes[1].axvline(v, linestyle=ls, color="blue", label=f"P{p}: {v:.2f}%")
    axes[1].set_title("MAE 24h (Maximum Adverse Excursion)")
    axes[1].set_xlabel("Perda máxima (%)")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    path = PLOTS_DIR / "mfe_mae_distribution.png"
    plt.savefig(path, dpi=110, bbox_inches="tight")
    plt.close()
    logger.info(f"Plot: {path.name}")


def plot_scatter(df_signals: pd.DataFrame):
    mfe = df_signals["mfe_pct_24h"].dropna() * 100
    mae = df_signals["mae_pct_24h"].dropna() * 100
    idx = mfe.index.intersection(mae.index)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(mae.loc[idx], mfe.loc[idx], alpha=0.5, s=25, color="#1565C0")

    # Reference lines: current stops
    ax.axhline(2.0, color="green", ls="--", lw=1, alpha=0.6, label="TP atual +2.0%")
    ax.axhline(1.0, color="green", ls=":", lw=1, alpha=0.5, label="TP +1.0%")
    ax.axhline(0.5, color="green", ls=":", lw=1, alpha=0.4, label="TP +0.5%")
    ax.axvline(-1.5, color="red", ls="--", lw=1, alpha=0.6, label="SL atual -1.5%")
    ax.axvline(-1.0, color="red", ls=":", lw=1, alpha=0.5, label="SL -1.0%")
    ax.axhline(0, color="gray", lw=0.8, alpha=0.4)

    ax.set_xlabel("MAE 24h % (perda máxima)")
    ax.set_ylabel("MFE 24h % (ganho máximo)")
    ax.set_title("MFE vs MAE por Sinal Bot 2 (24h)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    path = PLOTS_DIR / "mfe_mae_scatter.png"
    plt.savefig(path, dpi=110, bbox_inches="tight")
    plt.close()
    logger.info(f"Plot: {path.name}")


def plot_heatmap(hit_probs: pd.DataFrame):
    if hit_probs.empty:
        return

    pivot = hit_probs.pivot(index="sl_pct", columns="tp_pct", values="expectancy")

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    # Expectancy
    mat = pivot.values * 100
    im = axes[0].imshow(mat, cmap="RdYlGn", aspect="auto",
                        vmin=np.nanmin(mat), vmax=np.nanmax(mat))
    axes[0].set_xticks(range(len(pivot.columns)))
    axes[0].set_xticklabels([f"+{c*100:.1f}%" for c in pivot.columns], fontsize=8)
    axes[0].set_yticks(range(len(pivot.index)))
    axes[0].set_yticklabels([f"{r*100:.1f}%" for r in pivot.index], fontsize=8)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if not np.isnan(mat[i, j]):
                color = "white" if abs(mat[i, j]) > 0.05 else "black"
                axes[0].text(j, i, f"{mat[i,j]:+.3f}%", ha="center", va="center",
                             color=color, fontsize=8)
    plt.colorbar(im, ax=axes[0], label="Expectancy %")
    axes[0].set_title("Expectancy por TP × SL (janela 24h)")
    axes[0].set_xlabel("TP")
    axes[0].set_ylabel("SL")

    # Win rate
    pivot_wr = hit_probs.pivot(index="sl_pct", columns="tp_pct", values="win_rate")
    mat_wr = pivot_wr.values * 100
    im2 = axes[1].imshow(mat_wr, cmap="Blues", aspect="auto", vmin=0, vmax=100)
    axes[1].set_xticks(range(len(pivot_wr.columns)))
    axes[1].set_xticklabels([f"+{c*100:.1f}%" for c in pivot_wr.columns], fontsize=8)
    axes[1].set_yticks(range(len(pivot_wr.index)))
    axes[1].set_yticklabels([f"{r*100:.1f}%" for r in pivot_wr.index], fontsize=8)
    for i in range(mat_wr.shape[0]):
        for j in range(mat_wr.shape[1]):
            if not np.isnan(mat_wr[i, j]):
                color = "white" if mat_wr[i, j] > 60 else "black"
                axes[1].text(j, i, f"{mat_wr[i,j]:.0f}%", ha="center", va="center",
                             color=color, fontsize=8)
    plt.colorbar(im2, ax=axes[1], label="Win Rate %")
    axes[1].set_title("Win Rate por TP × SL (janela 24h)")
    axes[1].set_xlabel("TP")
    axes[1].set_ylabel("SL")

    plt.tight_layout()
    path = PLOTS_DIR / "tp_sl_expectancy.png"
    plt.savefig(path, dpi=110, bbox_inches="tight")
    plt.close()
    logger.info(f"Plot: {path.name}")


def plot_percentile_curves(percentiles: pd.DataFrame):
    """MFE/MAE percentis por janela de tempo."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    windows = percentiles["window_hours"].values
    colors_mfe = {"p25": "#81C784", "p50": "#388E3C", "p75": "#1B5E20"}
    colors_mae = {"p25": "#EF9A9A", "p50": "#E53935", "p75": "#7f0000"}

    for p_key, color in colors_mfe.items():
        p = p_key.replace("p", "")
        col = f"mfe_{p_key}"
        if col in percentiles.columns:
            axes[0].plot(windows, percentiles[col] * 100, "o-", color=color,
                         label=f"P{p}", lw=2)
    axes[0].axhline(2.0, color="gray", ls="--", alpha=0.5, label="TP atual +2%")
    axes[0].axhline(1.0, color="gray", ls=":", alpha=0.5)
    axes[0].set_title("MFE por Janela de Tempo")
    axes[0].set_xlabel("Horas após entrada")
    axes[0].set_ylabel("MFE %")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    for p_key, color in colors_mae.items():
        p = p_key.replace("p", "")
        col = f"mae_{p_key}"
        if col in percentiles.columns:
            axes[1].plot(windows, percentiles[col] * 100, "o-", color=color,
                         label=f"P{p}", lw=2)
    axes[1].axhline(-1.5, color="gray", ls="--", alpha=0.5, label="SL atual -1.5%")
    axes[1].axhline(-1.0, color="gray", ls=":", alpha=0.5)
    axes[1].set_title("MAE por Janela de Tempo")
    axes[1].set_xlabel("Horas após entrada")
    axes[1].set_ylabel("MAE %")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    path = PLOTS_DIR / "mfe_mae_by_window.png"
    plt.savefig(path, dpi=110, bbox_inches="tight")
    plt.close()
    logger.info(f"Plot: {path.name}")


def generate_report(df_signals, percentiles, bb_cond, hit_probs):
    mfe_24 = df_signals["mfe_pct_24h"].dropna() * 100
    mae_24 = df_signals["mae_pct_24h"].dropna() * 100

    best = hit_probs.iloc[0] if not hit_probs.empty else None
    current = hit_probs[(hit_probs["tp_pct"] == 0.020) & (hit_probs["sl_pct"] == -0.015)]
    cur = current.iloc[0] if not current.empty else None

    lines = [
        "# 📊 MFE/MAE Study — Bot 2 Entry Signals",
        "",
        f"**Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        f"**Period:** {df_signals['timestamp'].min().date()} → {df_signals['timestamp'].max().date()}",
        f"**Signals analyzed:** {len(df_signals)}",
        "",
        "## 🎯 Sumário Executivo",
        "",
        f"Em {len(df_signals)} sinais históricos do Bot 2, nas 24h seguintes à entrada:",
        f"- **MFE P50:** +{mfe_24.median():.2f}%  |  **MFE P75:** +{np.percentile(mfe_24, 75):.2f}%  |  **MFE P90:** +{np.percentile(mfe_24, 90):.2f}%",
        f"- **MAE P50:** {mae_24.median():.2f}%  |  **MAE P25:** {np.percentile(mae_24, 25):.2f}%  |  **MAE P10:** {np.percentile(mae_24, 10):.2f}%",
        "",
        f"**Interpretação:** o TP atual de +2.0% é atingido em apenas {(mfe_24 >= 2.0).mean()*100:.0f}% dos trades.",
        f"O SL atual de -1.5% é atingido em {(mae_24 <= -1.5).mean()*100:.0f}% dos trades nas primeiras 24h.",
        "",
        "## ⏱️ Percentis por Janela de Tempo",
        "",
        "### MFE (ganho máximo atingido)",
        "",
        "| Janela | n | P10 | P25 | P50 | P75 | P90 |",
        "|--------|---|-----|-----|-----|-----|-----|",
    ]
    for _, row in percentiles.iterrows():
        lines.append(
            f"| {int(row['window_hours'])}h | {int(row['n_signals'])} "
            f"| +{row['mfe_p10']*100:.2f}% | +{row['mfe_p25']*100:.2f}% "
            f"| +{row['mfe_p50']*100:.2f}% | +{row['mfe_p75']*100:.2f}% "
            f"| +{row['mfe_p90']*100:.2f}% |"
        )
    lines += [
        "",
        "### MAE (perda máxima atingida)",
        "",
        "| Janela | n | P10 | P25 | P50 | P75 | P90 |",
        "|--------|---|-----|-----|-----|-----|-----|",
    ]
    for _, row in percentiles.iterrows():
        lines.append(
            f"| {int(row['window_hours'])}h | {int(row['n_signals'])} "
            f"| {row['mae_p10']*100:.2f}% | {row['mae_p25']*100:.2f}% "
            f"| {row['mae_p50']*100:.2f}% | {row['mae_p75']*100:.2f}% "
            f"| {row['mae_p90']*100:.2f}% |"
        )

    if not bb_cond.empty:
        lines += [
            "",
            "## 📈 Análise Condicional por BB% na Entrada",
            "",
            "| BB% Range | n | MFE P50 24h | MAE P50 24h | MFE P50 48h | MAE P50 48h |",
            "|-----------|---|-------------|-------------|-------------|-------------|",
        ]
        for _, row in bb_cond.iterrows():
            lines.append(
                f"| {row['condition']} | {int(row['n'])} "
                f"| +{row.get('mfe_p50_24h', 0)*100:.2f}% | {row.get('mae_p50_24h', 0)*100:.2f}% "
                f"| +{row.get('mfe_p50_48h', 0)*100:.2f}% | {row.get('mae_p50_48h', 0)*100:.2f}% |"
            )

    lines += [
        "",
        "## 🎯 Grid TP × SL — Top 10 por Expectancy (janela 24h)",
        "",
        "| # | TP | SL | R:R | Win Rate | Loss Rate | Timeout | Expectancy |",
        "|---|-----|-----|-----|----------|-----------|---------|------------|",
    ]
    for i, (_, row) in enumerate(hit_probs.head(10).iterrows(), 1):
        lines.append(
            f"| {i} | +{row['tp_pct']*100:.1f}% | {row['sl_pct']*100:.1f}% "
            f"| {row['rr_ratio']:.2f}:1 | {row['win_rate']*100:.1f}% "
            f"| {row['loss_rate']*100:.1f}% | {row['timeout_rate']*100:.1f}% "
            f"| {row['expectancy']*100:+.3f}% |"
        )

    lines += ["", "## ⚖️ Config Atual vs Melhor Encontrada", ""]
    if cur is not None:
        lines += [
            "### Atual: TP +2.0% / SL -1.5%",
            f"- Win Rate: **{cur['win_rate']*100:.1f}%**",
            f"- Loss Rate: {cur['loss_rate']*100:.1f}%",
            f"- Timeout: {cur['timeout_rate']*100:.1f}%",
            f"- Expectancy: **{cur['expectancy']*100:+.3f}%**",
            "",
        ]
    if best is not None:
        lines += [
            f"### Melhor: TP +{best['tp_pct']*100:.1f}% / SL {best['sl_pct']*100:.1f}%",
            f"- Win Rate: **{best['win_rate']*100:.1f}%**",
            f"- Loss Rate: {best['loss_rate']*100:.1f}%",
            f"- Timeout: {best['timeout_rate']*100:.1f}%",
            f"- Expectancy: **{best['expectancy']*100:+.3f}%**",
            "",
        ]
        if cur is not None:
            delta_exp = (best["expectancy"] - cur["expectancy"]) * 100
            delta_wr = (best["win_rate"] - cur["win_rate"]) * 100
            lines += [
                f"**Δ Expectancy: {delta_exp:+.3f}pp** | **Δ Win Rate: {delta_wr:+.1f}pp**",
                "",
            ]

    lines += [
        "## 📊 Plots",
        "",
        "- [Distribuição MFE/MAE 24h](plots/mfe_mae_distribution.png)",
        "- [Scatter MFE vs MAE](plots/mfe_mae_scatter.png)",
        "- [Heatmap Expectancy + WR](plots/tp_sl_expectancy.png)",
        "- [Percentis por janela](plots/mfe_mae_by_window.png)",
        "",
        "## 💡 Interpretação",
        "",
        "**Regra de ouro:**",
        "- TP ideal ≈ P25–P50 do MFE → ganhar o que o mercado dá na metade dos trades",
        "- SL ideal ≈ P40–P50 do MAE → aceitar a perda que já acontece na metade dos casos",
        "- Timeout alto (> 30%) indica que TP muito distante → mais trades ficam presos sem direção",
        "",
        "## 🎯 Próximos Passos",
        "",
        "1. Se melhor config tiver Expectancy significativamente maior → ajustar `parameters.yml`",
        "2. Se BB% condicional mostrar grande variação → considerar stops por zona (BB%)",
        "3. Rodar `backtest_bot2_v2.py` com nova config para confirmar",
    ]

    REPORT_PATH.write_text("\n".join(lines))
    logger.info(f"Report: {REPORT_PATH}")


def main():
    logger.info("=" * 60)
    logger.info("MFE/MAE Study — Bot 2 Entry Signals")
    logger.info("=" * 60)

    df = load_data()

    logger.info("\n── Collecting signals + MFE/MAE ──")
    df_signals = collect_signals(df)
    logger.info(f"Signals found: {len(df_signals)}")

    if df_signals.empty:
        logger.error("No signals — aborting")
        return

    df_signals.to_csv(TABLES_DIR / "mfe_mae_raw.csv", index=False)

    logger.info("\n── Percentiles ──")
    percentiles = compute_percentiles(df_signals)
    percentiles.to_csv(TABLES_DIR / "mfe_mae_percentiles.csv", index=False)

    logger.info("\n── Conditional analysis (BB%) ──")
    bb_cond = compute_conditional_stats(
        df_signals, "bb_pct_entry",
        bins=[0, 0.3, 0.6, 0.98],
        labels=["BB<0.3 (fundo)", "BB 0.3-0.6 (mid)", "BB>0.6 (topo)"],
    )

    logger.info("\n── TP/SL expectancy grid ──")
    hit_probs = compute_hit_probabilities(df_signals)
    hit_probs.to_csv(TABLES_DIR / "tp_sl_expectancy.csv", index=False)

    logger.info("\n── Plots ──")
    plot_distribution(df_signals)
    plot_scatter(df_signals)
    plot_heatmap(hit_probs)
    plot_percentile_curves(percentiles)

    logger.info("\n── Report ──")
    generate_report(df_signals, percentiles, bb_cond, hit_probs)

    mfe_24 = df_signals["mfe_pct_24h"].dropna() * 100
    mae_24 = df_signals["mae_pct_24h"].dropna() * 100
    best = hit_probs.iloc[0] if not hit_probs.empty else None
    cur_row = hit_probs[(hit_probs["tp_pct"] == 0.020) & (hit_probs["sl_pct"] == -0.015)]

    print("\n" + "=" * 60)
    print("RESUMO MFE/MAE Study")
    print("=" * 60)
    print(f"Sinais: {len(df_signals)}")
    print(f"MFE 24h: P25={np.percentile(mfe_24,25):.2f}% | P50={mfe_24.median():.2f}% | P75={np.percentile(mfe_24,75):.2f}%")
    print(f"MAE 24h: P25={np.percentile(mae_24,25):.2f}% | P50={mae_24.median():.2f}% | P75={np.percentile(mae_24,75):.2f}%")
    print(f"TP atual +2.0% atingido em: {(mfe_24>=2.0).mean()*100:.0f}% dos sinais (24h)")
    print(f"SL atual -1.5% atingido em: {(mae_24<=-1.5).mean()*100:.0f}% dos sinais (24h)")
    if not cur_row.empty:
        c = cur_row.iloc[0]
        print(f"\nConfig atual  TP+2%/SL-1.5%: WR={c['win_rate']*100:.1f}% | Exp={c['expectancy']*100:+.3f}%")
    if best is not None:
        print(f"Melhor config TP+{best['tp_pct']*100:.1f}%/SL{best['sl_pct']*100:.1f}%: WR={best['win_rate']*100:.1f}% | Exp={best['expectancy']*100:+.3f}%")
    print(f"\nTop 3 por expectancy:")
    for i, (_, r) in enumerate(hit_probs.head(3).iterrows(), 1):
        print(f"  {i}. TP+{r['tp_pct']*100:.1f}%/SL{r['sl_pct']*100:.1f}% → WR {r['win_rate']*100:.1f}%, Exp {r['expectancy']*100:+.3f}%")
    print(f"\nReport: {REPORT_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()
