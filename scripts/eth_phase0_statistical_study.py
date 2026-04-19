#!/usr/bin/env python3
"""
Fase 0 — Estudo Estatístico Descritivo do ETH

Analisa correlações, estabilidade e alinhamento dos gates atuais
(calibrados para BTC) aplicados a dados de ETH.

Output:
  prompts/eth_phase0_report.md
  prompts/plots/eth_phase0_*.png
  prompts/tables/eth_correlations_*.csv

Usage:
  python scripts/eth_phase0_statistical_study.py
"""
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import get_params

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("eth_phase0")

OUT_DIR = ROOT / "prompts"
PLOTS_DIR = OUT_DIR / "plots"
TABLES_DIR = OUT_DIR / "tables"
REPORT_PATH = OUT_DIR / "eth_phase0_report.md"

for d in [PLOTS_DIR, TABLES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Gates: ETH-specific (recomputed from raw) + global (reused from BTC zscores)
GATE_MAP = {
    "oi_z":         {"name": "G4 OI",         "cfg_key": "g4_oi",      "source": "eth"},
    "funding_z":    {"name": "G10 Funding",    "cfg_key": "g10_funding", "source": "eth"},
    "taker_z":      {"name": "G9 Taker",       "cfg_key": "g9_taker",   "source": "eth"},
    "stablecoin_z": {"name": "G5 Stablecoin",  "cfg_key": "g5_stable",  "source": "global"},
    "bubble_z":     {"name": "G6 Bubble",      "cfg_key": "g6_bubble",  "source": "global"},
    "etf_z":        {"name": "G7 ETF",         "cfg_key": "g7_etf",     "source": "global"},
    "fg_z":         {"name": "G8 F&G",         "cfg_key": "g8_fg",      "source": "global"},
    "dgs10_z":      {"name": "G3 DGS10",       "cfg_key": "g3_dgs10",   "source": "global"},
    "curve_z":      {"name": "G3 Curve",       "cfg_key": "g3_curve",   "source": "global"},
}


def _zscore_rolling(series: pd.Series, window: int = 30) -> pd.Series:
    m = series.rolling(window, min_periods=window // 2).mean()
    s = series.rolling(window, min_periods=window // 2).std()
    return (series - m) / s.replace(0, np.nan)


def load_eth_data() -> pd.DataFrame:
    """Load ETH spot + recompute ETH z-scores + attach global z-scores → daily DataFrame."""

    # --- Spot 1h → daily close ---
    spot = pd.read_parquet(ROOT / "data/01_raw/spot/eth_1h.parquet")
    spot["timestamp"] = pd.to_datetime(spot["timestamp"], utc=True)
    spot = spot.sort_values("timestamp")
    close_daily = spot.set_index("timestamp")["close"].resample("1D").last().dropna()
    logger.info(f"Spot ETH: {len(spot)} 1h rows, {len(close_daily)} daily candles")
    logger.info(f"  Range: {close_daily.index.min().date()} → {close_daily.index.max().date()}")

    # --- ETH OI 4h → daily z-score ---
    oi = pd.read_parquet(ROOT / "data/01_raw/futures/eth_oi_4h.parquet")
    oi["timestamp"] = pd.to_datetime(oi["timestamp"], utc=True)
    oi_daily = oi.set_index("timestamp")["open_interest"].resample("1D").last()
    oi_z = _zscore_rolling(oi_daily, 30)

    # --- ETH Funding 4h → daily z-score ---
    fund = pd.read_parquet(ROOT / "data/01_raw/futures/eth_funding_4h.parquet")
    fund["timestamp"] = pd.to_datetime(fund["timestamp"], utc=True)
    fund_daily = fund.set_index("timestamp")["funding_rate"].resample("1D").last()
    funding_z = _zscore_rolling(fund_daily, 30)

    # --- ETH Taker 4h → daily z-score (buy_sell_ratio) ---
    taker = pd.read_parquet(ROOT / "data/01_raw/futures/eth_taker_4h.parquet")
    taker["timestamp"] = pd.to_datetime(taker["timestamp"], utc=True)
    taker_daily = taker.set_index("timestamp")["buy_sell_ratio"].resample("1D").last()
    taker_z = _zscore_rolling(taker_daily, 30)

    # --- Global z-scores (1h → daily last) ---
    global_cols = ["stablecoin_z", "bubble_z", "etf_z", "fg_z", "dgs10_z", "curve_z"]
    zs = pd.read_parquet(ROOT / "data/02_features/gate_zscores.parquet")
    zs["timestamp"] = pd.to_datetime(zs["timestamp"], utc=True)
    available_global = [c for c in global_cols if c in zs.columns]
    missing_global = [c for c in global_cols if c not in zs.columns]
    if missing_global:
        logger.warning(f"Global z-score cols not found: {missing_global}")
    zs_daily = zs.set_index("timestamp")[available_global].resample("1D").last()

    # --- Assemble daily DataFrame ---
    df = pd.DataFrame(index=close_daily.index)
    df["close"] = close_daily
    df["oi_z"] = oi_z.reindex(df.index, method="ffill")
    df["funding_z"] = funding_z.reindex(df.index, method="ffill")
    df["taker_z"] = taker_z.reindex(df.index, method="ffill")
    for col in available_global:
        df[col] = zs_daily[col].reindex(df.index, method="ffill")
    for col in missing_global:
        df[col] = np.nan

    # --- Forward returns (daily) ---
    df["ret_1d"] = df["close"].pct_change().shift(-1) * 100
    df["ret_3d"] = df["close"].pct_change(3).shift(-3) * 100
    df["ret_7d"] = df["close"].pct_change(7).shift(-7) * 100

    logger.info(f"Assembled: {len(df)} daily rows")
    return df


def _max_drawdown(series: pd.Series) -> float:
    peak = series.cummax()
    dd = (series - peak) / peak * 100
    return float(dd.min())


def analyze_regime(df: pd.DataFrame, label: str) -> dict:
    ret = df["close"].pct_change().dropna()
    return {
        "n_days": len(df),
        "price_start": round(float(df["close"].iloc[0]), 2),
        "price_end": round(float(df["close"].iloc[-1]), 2),
        "total_return_pct": round((df["close"].iloc[-1] / df["close"].iloc[0] - 1) * 100, 2),
        "daily_vol_pct": round(float(ret.std() * 100), 2),
        "annualized_vol_pct": round(float(ret.std() * 100 * np.sqrt(365)), 2),
        "max_drawdown_pct": round(_max_drawdown(df["close"]), 2),
        "autocorr_1d": round(float(ret.autocorr(lag=1)), 3) if len(ret) > 10 else None,
        "skew": round(float(ret.skew()), 3),
        "kurtosis": round(float(ret.kurtosis()), 3),
    }


def analyze_correlations(df: pd.DataFrame, label: str, params: dict) -> pd.DataFrame:
    gate_params = params.get("gate_params", {})
    rows = []

    for zcol, info in GATE_MAP.items():
        gname = info["name"]
        cfg_key = info["cfg_key"]
        corr_cfg = gate_params.get(cfg_key, [0, 0, 0])[0]

        if zcol not in df.columns or df[zcol].isna().all():
            rows.append({"gate": gname, "zcol": zcol, "source": info["source"],
                         "corr_cfg": corr_cfg, "status": "❌ missing"})
            continue

        sub = df[[zcol, "ret_1d", "ret_3d", "ret_7d"]].dropna()
        n = len(sub)
        if n < 15:
            rows.append({"gate": gname, "zcol": zcol, "source": info["source"],
                         "corr_cfg": corr_cfg, "n": n,
                         "status": f"⚠️ low sample (n={n})"})
            continue

        c1 = float(sub[zcol].corr(sub["ret_1d"]))
        c3 = float(sub[zcol].corr(sub["ret_3d"]))
        c7 = float(sub[zcol].corr(sub["ret_7d"]))

        delta_3d = abs(corr_cfg - c3)

        if delta_3d < 0.15:
            status = "✅ aligned"
        elif delta_3d < 0.30:
            status = "⚠️ attention"
        else:
            status = "🔴 broken"

        # flag sinal invertido
        if corr_cfg != 0 and not np.isnan(c3):
            if np.sign(corr_cfg) != np.sign(c3):
                status += " (inv)"

        rows.append({
            "gate": gname, "zcol": zcol, "source": info["source"],
            "corr_cfg": round(corr_cfg, 3),
            "corr_1d": round(c1, 3), "corr_3d": round(c3, 3), "corr_7d": round(c7, 3),
            "delta_3d": round(delta_3d, 3), "n": n, "status": status,
        })

    result = pd.DataFrame(rows)
    result.to_csv(TABLES_DIR / f"eth_correlations_{label}.csv", index=False)
    logger.info(f"Saved correlations table ({label}): {len(result)} gates")
    return result


def compute_alignment(df_corr: pd.DataFrame) -> dict:
    valid = df_corr.dropna(subset=["corr_3d", "corr_cfg"])
    if valid.empty:
        return {"alignment": None, "avg_delta": None, "distribution": {}, "n_gates": 0}
    deltas = (valid["corr_cfg"] - valid["corr_3d"]).abs()
    avg_delta = float(deltas.mean())
    return {
        "alignment": round(max(0.0, 1.0 - avg_delta), 3),
        "avg_delta": round(avg_delta, 3),
        "distribution": valid["status"].value_counts().to_dict(),
        "n_gates": len(valid),
    }


def plot_heatmap(df_corr: pd.DataFrame, label: str):
    cols = ["corr_cfg", "corr_1d", "corr_3d", "corr_7d"]
    valid = df_corr.dropna(subset=cols)
    if valid.empty:
        return

    fig, ax = plt.subplots(figsize=(8, max(4, len(valid) * 0.55 + 1)))
    mat = valid[cols].values.astype(float)
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-0.7, vmax=0.7, aspect="auto")

    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(["Config\n(BTC)", "ETH 1d", "ETH 3d", "ETH 7d"], fontsize=9)
    ax.set_yticks(range(len(valid)))
    labels = [f"{r['gate']} ({r['source']})" for _, r in valid.iterrows()]
    ax.set_yticklabels(labels, fontsize=8)

    for i in range(len(valid)):
        for j in range(len(cols)):
            v = mat[i, j]
            color = "white" if abs(v) > 0.35 else "black"
            ax.text(j, i, f"{v:+.2f}", ha="center", va="center", color=color, fontsize=9)

    plt.colorbar(im, ax=ax, label="Pearson r")
    ax.set_title(f"ETH Gate Correlations vs Forward Returns — {label}", fontsize=11)
    plt.tight_layout()
    path = PLOTS_DIR / f"eth_corr_heatmap_{label}.png"
    plt.savefig(path, dpi=110, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved heatmap: {path.name}")


def plot_stability(df: pd.DataFrame, label: str):
    gates_present = [(zcol, info) for zcol, info in GATE_MAP.items()
                     if zcol in df.columns and not df[zcol].isna().all()]
    n = len(gates_present)
    if n == 0:
        return

    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3.5 * rows), sharex=True)
    axes = np.array(axes).flatten() if n > 1 else [axes]

    for idx, (zcol, info) in enumerate(gates_present):
        ax = axes[idx]
        sub = df[[zcol, "ret_3d"]].dropna()
        if len(sub) < 20:
            ax.text(0.5, 0.5, f"n={len(sub)}", ha="center", transform=ax.transAxes)
        else:
            rc = sub[zcol].rolling(min(20, len(sub) // 3)).corr(sub["ret_3d"])
            ax.plot(rc.index, rc.values, lw=1.5, label="rolling corr 3d")
            ax.axhline(0, color="gray", lw=0.8, ls="--")
        ax.set_title(f"{info['name']} ({info['source']})", fontsize=9)
        ax.set_ylim(-1, 1)
        ax.grid(alpha=0.3)

    for ax in axes[n:]:
        ax.set_visible(False)

    plt.suptitle(f"ETH Rolling Correlation vs Forward 3d Return — {label}", fontsize=11)
    plt.tight_layout()
    path = PLOTS_DIR / f"eth_stability_{label}.png"
    plt.savefig(path, dpi=110, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved stability: {path.name}")


def generate_report(results: dict):
    lines = [
        "# 🔬 ETH Phase 0 — Statistical Descriptive Study",
        "",
        f"**Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        "",
        "**Objetivo:** Verificar transferibilidade dos gates BTC → ETH antes de calibrar `parameters_eth.yml`.",
        "",
    ]

    for label, data in results.items():
        regime = data["regime"]
        alignment = data["alignment"]
        df_corr = data["correlations"]

        lines += [
            f"## Janela {label}",
            "",
            "### Regime ETH",
            "",
            f"| Métrica | Valor |",
            f"|---------|-------|",
        ]
        for k, v in regime.items():
            if v is not None:
                lines.append(f"| {k} | {v} |")

        lines += [
            "",
            "### Alinhamento com BTC config",
            "",
            f"- **Model Alignment:** `{alignment.get('alignment')}` ({alignment.get('n_gates')} gates)",
            f"- **Avg Delta:** `{alignment.get('avg_delta')}`",
            f"- **Distribuição:** {alignment.get('distribution')}",
            "",
            "### Correlações por gate",
            "",
            "| Gate | Fonte | Config (BTC) | ETH 1d | ETH 3d | ETH 7d | Δ_3d | n | Status |",
            "|------|-------|-------------|--------|--------|--------|------|---|--------|",
        ]
        for _, row in df_corr.iterrows():
            lines.append(
                f"| {row.get('gate','—')} "
                f"| {row.get('source','—')} "
                f"| {row.get('corr_cfg','—')} "
                f"| {row.get('corr_1d','—')} "
                f"| {row.get('corr_3d','—')} "
                f"| {row.get('corr_7d','—')} "
                f"| {row.get('delta_3d','—')} "
                f"| {row.get('n','—')} "
                f"| {row.get('status','—')} |"
            )
        lines += [
            "",
            f"![Heatmap](plots/eth_corr_heatmap_{label.replace(' ', '_')}.png)",
            f"![Stability](plots/eth_stability_{label.replace(' ', '_')}.png)",
            "",
        ]

    lines += [
        "## 💡 Guia de decisão",
        "",
        "| Alignment | Decisão |",
        "|-----------|---------|",
        "| > 0.7 | Copiar parameters.yml como baseline ETH — ajuste mínimo |",
        "| 0.4 – 0.7 | Adaptive layer suficiente — ajustar corr_cfg dos gates ⚠️ |",
        "| < 0.4 | Recalibração manual necessária antes de paper trading |",
        "",
        "**Gates ✅ aligned:** transferíveis direto",
        "**Gates ⚠️ attention:** ajustar `corr_cfg` no parameters_eth.yml",
        "**Gates 🔴 broken:** remover ou desativar para ETH",
        "**Gates (inv):** sinal invertido — requer atenção especial",
        "",
        "## 🎯 Próximos passos",
        "",
        "1. Analisar quais gates são transferíveis",
        "2. Criar `conf/parameters_eth.yml` com ajustes necessários",
        "3. Ativar paper trading ETH quando alignment > 0.4",
    ]

    REPORT_PATH.write_text("\n".join(lines))
    logger.info(f"Report saved: {REPORT_PATH}")


def main():
    logger.info("=" * 60)
    logger.info("ETH Phase 0 — Statistical Descriptive Study")
    logger.info("=" * 60)

    params = get_params()
    df_full = load_eth_data()

    now = df_full.index.max()
    windows = {
        "28d": df_full.loc[now - timedelta(days=28):now],
        "180d": df_full.loc[now - timedelta(days=180):now],
    }

    results = {}
    for label, df_w in windows.items():
        logger.info(f"\n── Window {label} ({len(df_w)} days) ──")
        corr = analyze_correlations(df_w, label, params)
        results[label] = {
            "regime": analyze_regime(df_w, label),
            "correlations": corr,
            "alignment": compute_alignment(corr),
        }
        plot_heatmap(corr, label)
        plot_stability(df_w, label)

    generate_report(results)

    print("\n" + "=" * 60)
    print("RESUMO FINAL — ETH Phase 0")
    print("=" * 60)
    for label, data in results.items():
        a = data["alignment"]
        r = data["regime"]
        print(f"\nJanela {label} ({r['n_days']} dias):")
        print(f"  ETH: {r['price_start']} → {r['price_end']} USD  "
              f"({r['total_return_pct']:+.1f}%, vol {r['daily_vol_pct']:.1f}%/d, DD {r['max_drawdown_pct']:.1f}%)")
        print(f"  Model Alignment: {a['alignment']}  (avg_delta={a['avg_delta']}, n={a['n_gates']})")
        print(f"  Gates: {a['distribution']}")
    print(f"\nReport: {REPORT_PATH}")
    print(f"Plots:  {PLOTS_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
