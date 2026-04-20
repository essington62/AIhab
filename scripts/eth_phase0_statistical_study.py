#!/usr/bin/env python3
"""
Fase 0 ETH — Estudo Estatístico com Hipóteses Testáveis.

Responde:
  H1: Correlações ETH são similares a BTC?
  H2: Quais gates têm poder preditivo em ETH?
  H3: R5C HMM funciona em ETH?
  H4: Qual o model alignment ETH?

Critérios de decisão explícitos → COPY / ADAPT / RECALIBRATE / ABANDON.

Outputs:
  prompts/eth_phase0_report.md
  prompts/plots/fase0/
  prompts/tables/

Usage:
  python scripts/eth_phase0_statistical_study.py
"""
import logging
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("eth_phase0")

OUT_DIR = ROOT / "prompts"
PLOTS_DIR = OUT_DIR / "plots" / "fase0"
TABLES_DIR = OUT_DIR / "tables"
REPORT_PATH = OUT_DIR / "eth_phase0_report.md"

for d in [PLOTS_DIR, TABLES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Correlações BTC 2026 de referência (7d forward return)
BTC_CORRELATIONS = {
    "g4_oi_coin_margin": -0.472,
    "g5_stablecoin":     +0.326,
    "g7_etf":            +0.263,
    "g6_bubble":         -0.345,
    "g3_dgs10":          -0.315,
    "g3_curve":          -0.280,
    "g9_taker":          +0.060,
    "g10_funding":       +0.023,
    "g8_fg":             +0.150,
}


def load_eth_data() -> pd.DataFrame:
    # Spot 1h → daily
    spot = pd.read_parquet(ROOT / "data/01_raw/spot/eth_1h.parquet")
    spot["timestamp"] = pd.to_datetime(spot["timestamp"], utc=True)
    spot = spot.sort_values("timestamp")

    spot_d = spot.set_index("timestamp").resample("D").agg(
        open=("open", "first"), high=("high", "max"),
        low=("low", "min"), close=("close", "last"), volume=("volume", "sum"),
    ).reset_index()

    # Forward returns
    for h in [1, 3, 7, 14]:
        spot_d[f"fwd_return_{h}d"] = spot_d["close"].shift(-h) / spot_d["close"] - 1

    spot_d["ret_1d"] = spot_d["close"].pct_change(1)
    spot_d["ret_7d"] = spot_d["close"].pct_change(7)

    # RSI 14d
    delta = spot_d["close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    spot_d["rsi_14"] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))

    # MAs
    for w in [21, 50, 200]:
        spot_d[f"ma_{w}"] = spot_d["close"].rolling(w).mean()

    # BB%
    ma20 = spot_d["close"].rolling(20).mean()
    std20 = spot_d["close"].rolling(20).std()
    bb_upper = ma20 + 2 * std20
    bb_lower = ma20 - 2 * std20
    spot_d["bb_pct"] = (spot_d["close"] - bb_lower) / (bb_upper - bb_lower)

    # Derivatives 4h → daily mean
    oi = pd.read_parquet(ROOT / "data/01_raw/futures/eth_oi_4h.parquet")
    fund = pd.read_parquet(ROOT / "data/01_raw/futures/eth_funding_4h.parquet")
    taker = pd.read_parquet(ROOT / "data/01_raw/futures/eth_taker_4h.parquet")

    for dfr in [oi, fund, taker]:
        dfr["timestamp"] = pd.to_datetime(dfr["timestamp"], utc=True)

    oi_d = oi.set_index("timestamp")[["open_interest"]].resample("D").mean().reset_index()
    fund_d = fund.set_index("timestamp")[["funding_rate"]].resample("D").mean().reset_index()
    taker_d = taker.set_index("timestamp")[["buy_sell_ratio"]].resample("D").mean().reset_index()

    df = spot_d.copy()
    for right in [oi_d, fund_d, taker_d]:
        df = df.merge(right, on="timestamp", how="left")

    # Últimos 180 dias (overlap completo)
    df = df.tail(180).reset_index(drop=True)

    logger.info(f"Loaded {len(df)} daily rows")
    logger.info(f"Period: {df['timestamp'].min().date()} → {df['timestamp'].max().date()}")
    return df


def compute_zscores(df: pd.DataFrame, window: int = 30) -> pd.DataFrame:
    df = df.copy()

    def _z(series):
        m = series.rolling(window, min_periods=window // 2).mean()
        s = series.rolling(window, min_periods=window // 2).std()
        return (series - m) / s.replace(0, np.nan)

    df["oi_z"] = _z(df["open_interest"])
    df["funding_z"] = _z(df["funding_rate"])
    df["taker_z"] = _z(df["buy_sell_ratio"])
    df["volume_z"] = _z(df["volume"])
    df["rsi_z"] = (df["rsi_14"] - 50) / 20  # normalized, not rolling z

    logger.info("Z-scores computed: oi_z, funding_z, taker_z, volume_z, rsi_z")
    return df


def h1_correlation_analysis(df: pd.DataFrame) -> dict:
    """H1: Correlações ETH features vs forward returns."""
    feature_map = [
        ("oi_z",         "g4_oi_coin_margin"),
        ("funding_z",    "g10_funding"),
        ("taker_z",      "g9_taker"),
        ("volume_z",     "volume"),
        ("rsi_14",       "rsi"),
        ("bb_pct",       "bb_pct"),
        ("ret_1d",       "ret_1d"),
        ("ret_7d",       "ret_7d"),
    ]

    results = {}
    for h in [1, 3, 7, 14]:
        target = f"fwd_return_{h}d"
        if target not in df.columns:
            continue
        corrs = {}
        for col, label in feature_map:
            if col not in df.columns:
                continue
            valid = df[[col, target]].dropna()
            if len(valid) < 30:
                continue
            pr, pp = pearsonr(valid[col], valid[target])
            sr, sp = spearmanr(valid[col], valid[target])
            corrs[label] = {
                "pearson": float(pr), "pearson_p": float(pp),
                "spearman": float(sr), "n": len(valid),
                "significant": bool(pp < 0.05),
            }
        results[f"{h}d"] = corrs

    return results


def h2_gate_power(correlations: dict) -> dict:
    """H2: Classifica gates em STRONG / MODERATE / WEAK / NO_DATA."""
    corrs_7d = correlations.get("7d", {})
    gate_status = {}

    for gate, ref_corr in BTC_CORRELATIONS.items():
        eth_info = corrs_7d.get(gate, {})
        eth_corr = eth_info.get("pearson", np.nan)

        if pd.isna(eth_corr):
            gate_status[gate] = {
                "btc_corr": ref_corr, "eth_corr": None,
                "diff": None, "status": "NO_DATA", "action": "skip",
                "significant": False,
            }
            continue

        diff = abs(eth_corr - ref_corr)
        abs_eth = abs(eth_corr)

        if abs_eth > 0.20:
            status, action = "STRONG", "keep"
        elif abs_eth > 0.10:
            status, action = "MODERATE", "keep_reduced_weight"
        else:
            status, action = "WEAK", "discard"

        gate_status[gate] = {
            "btc_corr": ref_corr, "eth_corr": float(eth_corr),
            "diff": float(diff), "abs_eth": float(abs_eth),
            "status": status, "action": action,
            "significant": eth_info.get("significant", False),
        }

    return gate_status


def h3_regime_analysis(df: pd.DataFrame) -> dict | None:
    """H3: Aplica R5C HMM (treinado em BTC) ao ETH."""
    model_path = ROOT / "data/03_models/r5c_hmm.pkl"
    if not model_path.exists():
        logger.warning("R5C model not found")
        return None

    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
    except Exception as e:
        logger.warning(f"Failed to load R5C: {e}")
        return None

    df = df.copy()
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))
    df["log_return_21d"] = np.log(df["close"] / df["close"].shift(21))

    vol_short = df["log_return"].rolling(5).std()
    vol_long = df["log_return"].rolling(30).std()
    df["vol_ratio_log"] = np.log((vol_short / vol_long).replace(0, np.nan))
    df["vol_short_z"] = (vol_short - vol_short.rolling(60).mean()) / vol_short.rolling(60).std()

    df["volume_log"] = np.log(df["volume"].replace(0, np.nan))
    df["volume_log_z"] = (df["volume_log"] - df["volume_log"].rolling(60).mean()) / df["volume_log"].rolling(60).std()

    df["drawdown"] = df["close"] / df["close"].rolling(30).max() - 1
    df["drawdown_chg_5d"] = df["drawdown"] - df["drawdown"].shift(5)
    df["drawdown_chg_z"] = (df["drawdown_chg_5d"] - df["drawdown_chg_5d"].rolling(60).mean()) / df["drawdown_chg_5d"].rolling(60).std()

    features = ["log_return", "log_return_21d", "vol_ratio_log",
                "vol_short_z", "volume_log_z", "drawdown_chg_z"]

    valid = df[features].dropna()
    if len(valid) < 30:
        logger.warning("Not enough data for regime analysis")
        return None

    try:
        regimes = model.predict(valid.values)
        df_r = df.loc[valid.index].copy()
        df_r["regime"] = regimes

        # Map regime idx → name by mean return
        mean_ret = df_r.groupby("regime")["log_return"].mean().sort_values()
        regime_map = {}
        for i, (idx, _) in enumerate(mean_ret.items()):
            regime_map[idx] = ["Bear", "Sideways", "Bull"][i]
        df_r["regime_name"] = df_r["regime"].map(regime_map)

        stats = {}
        for name in ["Bull", "Sideways", "Bear"]:
            sub = df_r[df_r["regime_name"] == name]
            if sub.empty:
                continue
            fwd = sub["fwd_return_7d"].dropna() if "fwd_return_7d" in sub.columns else pd.Series()
            stats[name] = {
                "n": len(sub),
                "pct": len(sub) / len(df_r) * 100,
                "avg_fwd_7d": float(fwd.mean()) if len(fwd) else None,
                "median_fwd_7d": float(fwd.median()) if len(fwd) else None,
            }

        all_three = all(r in stats for r in ["Bull", "Sideways", "Bear"])
        regime_separates = (
            all_three
            and (stats["Bull"]["avg_fwd_7d"] or 0) > (stats["Sideways"]["avg_fwd_7d"] or 0)
            and (stats["Sideways"]["avg_fwd_7d"] or 0) > (stats["Bear"]["avg_fwd_7d"] or 0)
        )

        return {
            "regime_stats": stats,
            "regime_separates": regime_separates,
            "df_regime": df_r[["timestamp", "close", "log_return", "regime_name"]],
        }
    except Exception as e:
        logger.warning(f"Regime prediction failed: {e}")
        return None


def h4_alignment(gate_status: dict) -> float:
    """H4: Model alignment = 1 - avg_diff (for non-NO_DATA gates)."""
    diffs = [v["diff"] for v in gate_status.values() if v.get("diff") is not None]
    if not diffs:
        return 0.0
    return float(max(0.0, 1.0 - np.mean(diffs) / 0.5))


def decision_logic(alignment: float, gate_status: dict, regime_result) -> dict:
    strong = sum(1 for v in gate_status.values() if v.get("status") == "STRONG")
    moderate = sum(1 for v in gate_status.values() if v.get("status") == "MODERATE")
    weak = sum(1 for v in gate_status.values() if v.get("status") == "WEAK")
    regime_ok = bool(regime_result and regime_result.get("regime_separates", False))

    if alignment > 0.70 and strong >= 3 and regime_ok:
        decision = "COPY_BTC_CONFIG"
        desc = "ETH muito similar a BTC. Copiar parameters.yml com ajustes mínimos."
    elif alignment > 0.40 and (strong + moderate) >= 3:
        decision = "ADAPT_CONFIG"
        desc = "ETH parcialmente similar. Recalibrar gates fracos, manter arquitetura."
    elif alignment > 0.20:
        decision = "RECALIBRATE_DEEPLY"
        desc = "ETH diferente. Recalibração profunda de gates e pesos necessária."
    else:
        decision = "ABANDON_OR_REDESIGN"
        desc = "ETH não se encaixa no framework BTC. Considerar features ETH-específicas."

    return {
        "decision": decision, "description": desc,
        "alignment": alignment, "strong_gates": strong,
        "moderate_gates": moderate, "weak_gates": weak, "regime_ok": regime_ok,
    }


def plot_correlations(gate_status: dict):
    gates, btc_corrs, eth_corrs, statuses = [], [], [], []
    for gate, s in gate_status.items():
        if s.get("eth_corr") is None:
            continue
        gates.append(gate.replace("_", " "))
        btc_corrs.append(s["btc_corr"])
        eth_corrs.append(s["eth_corr"])
        statuses.append(s["status"])

    if not gates:
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(gates))
    w = 0.35

    ax.bar(x - w/2, btc_corrs, w, label="BTC 2026 ref", color="#FF9800", alpha=0.8)
    colors = {"STRONG": "#4CAF50", "MODERATE": "#2196F3", "WEAK": "#f44336"}
    for i, (ec, st) in enumerate(zip(eth_corrs, statuses)):
        ax.bar(x[i] + w/2, ec, w, color=colors.get(st, "gray"), alpha=0.8)

    # Legend proxies
    from matplotlib.patches import Patch
    legend = [Patch(color="#FF9800", label="BTC ref")] + [
        Patch(color=c, label=f"ETH {st}") for st, c in colors.items()
    ]
    ax.legend(handles=legend)
    ax.axhline(0, color="black", lw=0.5)
    ax.axhline(0.20, color="green", ls=":", alpha=0.5, lw=0.8)
    ax.axhline(-0.20, color="green", ls=":", alpha=0.5, lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(gates, rotation=40, ha="right", fontsize=9)
    ax.set_ylabel("Correlação com forward return 7d")
    ax.set_title("Gates: BTC ref vs ETH actual (7d forward return)")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = PLOTS_DIR / "correlations_eth_vs_btc.png"
    plt.savefig(path, dpi=110, bbox_inches="tight")
    plt.close()
    logger.info(f"Plot: {path.name}")


def plot_regimes(regime_result):
    if not regime_result:
        return
    df = regime_result["df_regime"]
    if df is None or df.empty:
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    colors = {"Bull": "#4CAF50", "Sideways": "#2196F3", "Bear": "#f44336"}

    for name, color in colors.items():
        mask = df["regime_name"] == name
        if mask.any():
            ax1.scatter(df.loc[mask, "timestamp"], df.loc[mask, "close"],
                        c=color, label=name, s=12, alpha=0.7)
    ax1.set_ylabel("ETH Price")
    ax1.set_title("ETH price colorido por regime R5C (modelo BTC)")
    ax1.legend()
    ax1.grid(alpha=0.3)

    for name, color in colors.items():
        mask = df["regime_name"] == name
        if mask.any():
            ax2.scatter(df.loc[mask, "timestamp"], df.loc[mask, "log_return"] * 100,
                        c=color, s=12, alpha=0.7)
    ax2.axhline(0, color="black", lw=0.5)
    ax2.set_ylabel("Log return %")
    ax2.grid(alpha=0.3)
    plt.tight_layout()
    path = PLOTS_DIR / "regimes_eth.png"
    plt.savefig(path, dpi=110, bbox_inches="tight")
    plt.close()
    logger.info(f"Plot: {path.name}")


def plot_correlation_matrix(df: pd.DataFrame):
    """Matriz de correlação entre features ETH."""
    cols = [c for c in ["oi_z", "funding_z", "taker_z", "volume_z", "rsi_14",
                         "bb_pct", "ret_1d", "ret_7d", "fwd_return_7d"] if c in df.columns]
    sub = df[cols].dropna()
    if len(sub) < 20:
        return
    corr = sub.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(cols)))
    ax.set_yticklabels(cols, fontsize=8)
    for i in range(len(cols)):
        for j in range(len(cols)):
            v = corr.iloc[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    color="white" if abs(v) > 0.5 else "black", fontsize=7)
    plt.colorbar(im, ax=ax)
    ax.set_title("ETH Features Correlation Matrix")
    plt.tight_layout()
    path = PLOTS_DIR / "correlation_matrix_eth.png"
    plt.savefig(path, dpi=110, bbox_inches="tight")
    plt.close()
    logger.info(f"Plot: {path.name}")


def generate_report(correlations, gate_status, regime_result, alignment, decision):
    emoji_map = {"COPY_BTC_CONFIG": "✅", "ADAPT_CONFIG": "⚠️",
                 "RECALIBRATE_DEEPLY": "🔴", "ABANDON_OR_REDESIGN": "❌"}

    lines = [
        "# 🔬 ETH Phase 0 — Statistical Study Report",
        "",
        f"**Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        "**Window:** 180 days  |  **Method:** Pearson + Spearman + HMM regime",
        "",
        "## 🎯 Decisão Final",
        "",
        f"### {emoji_map.get(decision['decision'], '❓')} {decision['decision']}",
        "",
        f"**{decision['description']}**",
        "",
        f"| Métrica | Valor | Threshold |",
        f"|---------|-------|-----------|",
        f"| Model Alignment | `{alignment:.3f}` | >0.70 copy / >0.40 adapt |",
        f"| Strong gates | `{decision['strong_gates']}` | ≥3 preferred |",
        f"| Moderate gates | `{decision['moderate_gates']}` | — |",
        f"| Weak gates | `{decision['weak_gates']}` | — |",
        f"| Regime separates | `{'✅' if decision['regime_ok'] else '❌'}` | Bull>Sideways>Bear |",
        "",
    ]

    # H1
    lines += ["## H1: Correlações ETH vs Forward Returns", ""]
    for horizon, corrs in correlations.items():
        if not corrs:
            continue
        lines += [
            f"### Horizonte {horizon}",
            "",
            "| Feature | Pearson r | p-value | Spearman r | n | Sig |",
            "|---------|-----------|---------|------------|---|-----|",
        ]
        for feat, c in sorted(corrs.items(), key=lambda x: -abs(x[1]["pearson"])):
            sig = "✅" if c["significant"] else "—"
            lines.append(
                f"| {feat} | {c['pearson']:+.3f} | {c['pearson_p']:.3f} | "
                f"{c['spearman']:+.3f} | {c['n']} | {sig} |"
            )
        lines.append("")

    # H2
    lines += [
        "## H2: Poder Preditivo dos Gates (7d forward return)",
        "",
        "| Gate | BTC ref | ETH actual | |Δ| | Power | Action | Sig |",
        "|------|---------|------------|-----|-------|--------|-----|",
    ]
    icons = {"STRONG": "🟢", "MODERATE": "🟡", "WEAK": "🔴", "NO_DATA": "⚪"}
    for gate, s in gate_status.items():
        btc = f"{s['btc_corr']:+.3f}"
        eth = f"{s['eth_corr']:+.3f}" if s.get("eth_corr") is not None else "N/A"
        diff = f"{s['diff']:.3f}" if s.get("diff") is not None else "—"
        sig = "✅" if s.get("significant") else "—"
        lines.append(
            f"| {gate} | {btc} | {eth} | {diff} | "
            f"{icons.get(s['status'])} {s['status']} | `{s['action']}` | {sig} |"
        )
    lines.append("")

    # H3
    lines += ["## H3: R5C HMM aplicado ao ETH", ""]
    if regime_result and regime_result.get("regime_stats"):
        lines += [
            "| Regime | n | % | Avg fwd 7d | Median fwd 7d |",
            "|--------|---|---|------------|---------------|",
        ]
        for name in ["Bull", "Sideways", "Bear"]:
            s = regime_result["regime_stats"].get(name, {})
            if not s:
                continue
            avg = f"{s['avg_fwd_7d']*100:+.2f}%" if s.get("avg_fwd_7d") is not None else "N/A"
            med = f"{s['median_fwd_7d']*100:+.2f}%" if s.get("median_fwd_7d") is not None else "N/A"
            lines.append(f"| {name} | {s['n']} | {s['pct']:.1f}% | {avg} | {med} |")
        sep = regime_result["regime_separates"]
        lines += ["", f"**Bull > Sideways > Bear:** {'✅ SIM' if sep else '❌ NÃO'}", ""]
    else:
        lines += ["*R5C não disponível ou falhou.*", ""]

    # H4
    lines += [
        "## H4: Model Alignment",
        "",
        f"**Alignment = {alignment:.3f}**",
        "",
        ("🟢 Similar → copy config" if alignment > 0.70 else
         "🟡 Parcialmente similar → adaptar" if alignment > 0.40 else
         "🔴 Diferente → recalibrar" if alignment > 0.20 else
         "⚫ Não se encaixa → redesign"),
        "",
    ]

    # Next steps
    lines += ["## 🎯 Próximos Passos", ""]
    d = decision["decision"]
    if d == "COPY_BTC_CONFIG":
        lines += [
            "1. Criar `conf/parameters_eth.yml` copiado de `parameters.yml`",
            "2. Ajustar símbolos, paths e stop sizes para ETH",
            "3. Paper trading paralelo — portfolio separado",
        ]
    elif d == "ADAPT_CONFIG":
        strong_g = [g for g, s in gate_status.items() if s.get("status") == "STRONG"]
        weak_g = [g for g, s in gate_status.items() if s.get("status") == "WEAK"]
        lines += [
            f"1. Manter gates STRONG: `{strong_g}`",
            f"2. Desabilitar gates WEAK: `{weak_g}`",
            "3. Recalibrar corr_cfg dos gates MODERATE no parameters_eth.yml",
        ]
    elif d == "RECALIBRATE_DEEPLY":
        lines += [
            "1. Retreinar R5C HMM com dados ETH (dados acumulam ao longo do tempo)",
            "2. Re-examinar correlações em janelas diferentes (90d, 360d)",
            "3. Considerar features ETH-específicas (staking yield, ETH/BTC ratio)",
        ]
    else:
        lines += [
            "1. Investigar features ETH-específicas: staking APY, L2 TVL, gas fees, burn rate",
            "2. Avaliar se ETH trading justifica o esforço vs focar só em BTC",
        ]

    lines += [
        "",
        "## 📊 Plots",
        "",
        "- [Correlações BTC vs ETH](plots/fase0/correlations_eth_vs_btc.png)",
        "- [Regimes R5C em ETH](plots/fase0/regimes_eth.png)",
        "- [Matriz de correlação ETH](plots/fase0/correlation_matrix_eth.png)",
    ]

    REPORT_PATH.write_text("\n".join(lines))
    logger.info(f"Report: {REPORT_PATH}")


def main():
    logger.info("=" * 60)
    logger.info("ETH Phase 0 — Statistical Study with Testable Hypotheses")
    logger.info("=" * 60)

    df = load_eth_data()
    df = compute_zscores(df)
    df.to_csv(TABLES_DIR / "eth_phase0_features.csv", index=False)

    logger.info("\n── H1: Correlations ──")
    correlations = h1_correlation_analysis(df)

    logger.info("\n── H2: Gate power ──")
    gate_status = h2_gate_power(correlations)

    logger.info("\n── H3: Regime (R5C HMM) ──")
    regime_result = h3_regime_analysis(df)

    logger.info("\n── H4: Alignment ──")
    alignment = h4_alignment(gate_status)
    decision = decision_logic(alignment, gate_status, regime_result)

    logger.info("\n── Plots ──")
    plot_correlations(gate_status)
    plot_regimes(regime_result)
    plot_correlation_matrix(df)

    logger.info("\n── Report ──")
    generate_report(correlations, gate_status, regime_result, alignment, decision)

    # Save gate status CSV
    gate_df = pd.DataFrame([
        {"gate": g, **{k: v for k, v in s.items() if k != "df_regime"}}
        for g, s in gate_status.items()
    ])
    gate_df.to_csv(TABLES_DIR / "eth_gate_status.csv", index=False)

    print("\n" + "=" * 60)
    print("ETH Phase 0 — RESULTADO")
    print("=" * 60)
    print(f"Decision:  {decision['decision']}")
    print(f"Alignment: {alignment:.3f}")
    print(f"Strong:    {decision['strong_gates']} | Moderate: {decision['moderate_gates']} | Weak: {decision['weak_gates']}")
    print(f"Regime OK: {decision['regime_ok']}")
    print()
    print("Gates:")
    for gate, s in gate_status.items():
        eth = f"{s['eth_corr']:+.3f}" if s.get("eth_corr") is not None else "N/A"
        print(f"  {s['status']:8s}  {gate:30s}  ETH={eth}  BTC={s['btc_corr']:+.3f}")
    print(f"\nReport: {REPORT_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()
