#!/usr/bin/env python3
"""
scripts/weekly_report.py — Weekly summary of paper-trading observation period.

Reads data/06_observation/daily_observation.csv and prints a structured report.

Run manually:
    conda run -n btc_trading_v1 python scripts/weekly_report.py
    # or inside container:
    python scripts/weekly_report.py
    # With explicit date range:
    python scripts/weekly_report.py 2026-04-17 2026-04-23
"""

import pathlib
import sys
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

_REPO = pathlib.Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

OBS_CSV = _REPO / "data" / "06_observation" / "daily_observation.csv"
LINE = "═" * 60


def _fmt_price(v):
    try:
        return f"${float(v):,.0f}"
    except Exception:
        return str(v)


def _pct(v, decimals=1):
    try:
        return f"{float(v):+.{decimals}f}%"
    except Exception:
        return "N/A"


def generate_report(start_date: str = None, end_date: str = None) -> str:
    if not OBS_CSV.exists():
        return f"Arquivo não encontrado: {OBS_CSV}\nRode daily_observation.py primeiro."

    df = pd.read_csv(OBS_CSV, parse_dates=["timestamp"])
    if df.empty:
        return "CSV vazio — nenhuma observação registrada ainda."

    # Date filter
    if start_date:
        df = df[df["timestamp"] >= pd.Timestamp(start_date, tz="UTC")]
    if end_date:
        df = df[df["timestamp"] <= pd.Timestamp(end_date, tz="UTC") + timedelta(days=1)]

    if df.empty:
        return f"Nenhuma observação no período {start_date} — {end_date}."

    period_start = df["timestamp"].min().strftime("%Y-%m-%d")
    period_end   = df["timestamp"].max().strftime("%Y-%m-%d")
    n_days       = len(df)

    # Price
    price_start = df["btc_price"].dropna().iloc[0]  if not df["btc_price"].dropna().empty else None
    price_end   = df["btc_price"].dropna().iloc[-1] if not df["btc_price"].dropna().empty else None
    price_chg   = (price_end - price_start) / price_start * 100 if price_start and price_end else None

    # Score stats
    scores = df["score_adjusted"].dropna()
    score_mean = scores.mean() if not scores.empty else None
    score_std  = scores.std()  if not scores.empty else None
    score_min  = scores.min()  if not scores.empty else None
    score_max  = scores.max()  if not scores.empty else None

    # Regime counts
    regime_counts = df["regime"].value_counts()
    regime_str = ", ".join(f"{r} ({c}/{n_days}d)" for r, c in regime_counts.items())

    # Signal counts
    signal_counts = df["signal"].value_counts()
    signal_str = ", ".join(f"{s} ({c}/{n_days}d)" for s, c in signal_counts.items())

    # Cluster means
    cluster_cols = [c for c in df.columns if c.startswith("cluster_")]
    cluster_means = {c.replace("cluster_", ""): df[c].mean() for c in cluster_cols}

    # Events (flags)
    events = []
    for _, row in df.iterrows():
        ts = row["timestamp"].strftime("%Y-%m-%d") if hasattr(row["timestamp"], "strftime") else str(row["timestamp"])
        flags = []
        if str(row.get("flag_near_enter", "")).lower() == "true":
            flags.append(f"⚠️  Near-ENTER (score {row.get('score_adjusted', '?'):.2f}, threshold {row.get('threshold', '?')})")
        if str(row.get("flag_signal_changed", "")).lower() == "true":
            flags.append(f"🔴 Sinal mudou → {row.get('signal', '?')}")
        if str(row.get("flag_kill_switch_active", "")).lower() == "true":
            ks = row.get("kill_switches", "?")
            flags.append(f"🛑 Kill switch: {ks}")
        if str(row.get("flag_bb_extreme_high", "")).lower() == "true":
            flags.append(f"📈 BB_TOP (bb_pct={row.get('bb_pct', '?'):.2f})")
        if str(row.get("flag_bb_extreme_low", "")).lower() == "true":
            flags.append(f"📉 BB_BOTTOM (bb_pct={row.get('bb_pct', '?'):.2f})")
        for f in flags:
            events.append(f"  - {ts}: {f}")

    events_str = "\n".join(events) if events else "  (nenhum evento relevante)"

    # F&G validation
    fg_check = "N/A"
    fg_df = df[["fg_raw", "fg_z"]].dropna()
    if not fg_df.empty:
        extreme_fear = fg_df[fg_df["fg_raw"] < 25]
        if not extreme_fear.empty:
            fg_neg_pct = (extreme_fear["fg_z"] < 0).mean() * 100
            fg_check = f"SIM ({fg_neg_pct:.0f}% dos dias de Extreme Fear com fg_z<0)"
        else:
            fg_check = "Sem dias de Extreme Fear no período"

    # Kill switch count
    ks_total = int(df["flag_kill_switch_active"].astype(str).str.lower().eq("true").sum())

    # Score vs price correlation
    score_price_corr = "N/A"
    if "btc_change_24h" in df.columns and not scores.empty:
        joint = df[["score_adjusted", "btc_change_24h"]].dropna()
        if len(joint) >= 5:
            corr = joint["score_adjusted"].corr(joint["btc_change_24h"])
            score_price_corr = f"{corr:.3f}"

    # Near-ENTER analysis
    near_enter_df = df[df["flag_near_enter"].astype(str).str.lower() == "true"]
    near_enter_notes = []
    if not near_enter_df.empty:
        for _, row in near_enter_df.iterrows():
            bb = row.get("bb_pct")
            oi = row.get("oi_z")
            ts = row["timestamp"].strftime("%Y-%m-%d") if hasattr(row["timestamp"], "strftime") else str(row["timestamp"])
            quality = "favorável" if (bb and float(bb) < 0.5 and oi and float(oi) < 1.0) else "misto"
            near_enter_notes.append(f"  {ts}: BB={bb:.2f} OI_z={oi:.2f} → setup {quality}")

    near_enter_str = "\n".join(near_enter_notes) if near_enter_notes else "  Nenhum near-ENTER registrado"

    # Threshold note
    thresholds = df["threshold"].dropna()
    thr_note = (f"range [{thresholds.min():.2f}, {thresholds.max():.2f}]"
                if not thresholds.empty else "N/A")

    lines = [
        LINE,
        f"  Relatório Semanal AI.hab",
        f"  Período: {period_start} a {period_end} ({n_days} observações)",
        LINE,
        "",
        "RESUMO",
        f"  BTC: {_fmt_price(price_start)} → {_fmt_price(price_end)} ({_pct(price_chg)})",
        f"  Score médio ajustado: {score_mean:.3f} (σ {score_std:.3f})" if score_mean is not None else "  Score: N/A",
        f"  Score range: [{score_min:.3f}, {score_max:.3f}]" if score_min is not None else "",
        f"  Regime: {regime_str}",
        f"  Sinal:  {signal_str}",
        f"  Threshold ativo: {thr_note}",
        "",
        "EVENTOS RELEVANTES",
        events_str,
        "",
        "CLUSTERS (média do período)",
    ] + [
        f"  {k:12s}: {v:+.4f}" for k, v in cluster_means.items()
    ] + [
        "",
        "VALIDAÇÃO DE CALIBRAÇÃO",
        f"  F&G z-score negativo em Extreme Fear? {fg_check}",
        f"  Kill switches dispararam: {ks_total} vez(es) em {n_days} dias",
        f"  Correlação score vs retorno 24h: {score_price_corr}",
        "",
        "PERGUNTAS PARA REVISÃO MANUAL",
        "  Near-ENTER — BB pct e OI z no momento:",
        near_enter_str,
        "",
        f"  Threshold {thr_note} está calibrado? Comparar score_mean ({score_mean:.3f}) vs threshold.",
        "",
        LINE,
    ]

    return "\n".join(lines)


if __name__ == "__main__":
    start = sys.argv[1] if len(sys.argv) > 1 else None
    end   = sys.argv[2] if len(sys.argv) > 2 else None

    if start is None:
        # Default: last 7 days
        end_dt   = datetime.now(tz=timezone.utc)
        start_dt = end_dt - timedelta(days=7)
        start = start_dt.strftime("%Y-%m-%d")
        end   = end_dt.strftime("%Y-%m-%d")

    print(generate_report(start, end))
