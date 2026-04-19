"""
Estudo de Adaptação de Gates — Fase 1 (Descritiva)

Analisa a estabilidade dos gates do AI.hab em 2026 e investiga se
o 'model alignment' pode ser usado como indicador de risco do sistema.

Saídas:
- prompts/estudo_adaptacao_gates_fase1.md
- prompts/plots/fase1/rolling_correlations.png
- prompts/plots/fase1/alignment_time_series.png
- prompts/tables/fase1/monthly_correlations.csv
- prompts/tables/fase1/gate_stability.csv
- prompts/tables/fase1/daily_alignment.csv
- prompts/tables/fase1/alignment_vs_performance.csv
- prompts/tables/fase1/regime_by_alignment.csv
- prompts/tables/fase1/performance_by_regime.csv
- prompts/tables/fase1/regime_transitions.csv
"""
import sys
from pathlib import Path
from datetime import date

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import yaml

matplotlib.use("Agg")  # non-interactive backend

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

OUTPUT_DIR = ROOT / "prompts"
PLOTS_DIR  = OUTPUT_DIR / "plots"  / "fase1"
TABLES_DIR = OUTPUT_DIR / "tables" / "fase1"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)

GATE_MAP = {
    "oi_z":         ("g4_oi",       "G4 OI"),
    "taker_z":      ("g9_taker",    "G9 Taker"),
    "funding_z":    ("g10_funding", "G10 Funding"),
    "dgs10_z":      ("g3_dgs10",    "G3 DGS10"),
    "curve_z":      ("g3_curve",    "G3 Curve"),
    "stablecoin_z": ("g5_stable",   "G5 Stablecoin"),
    "bubble_z":     ("g6_bubble",   "G6 Bubble"),
    "etf_z":        ("g7_etf",      "G7 ETF"),
    "fg_z":         ("g8_fg",       "G8 F&G"),
}

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data():
    zs   = pd.read_parquet(ROOT / "data/02_features/gate_zscores.parquet")
    spot = pd.read_parquet(ROOT / "data/02_intermediate/spot/btc_1h_clean.parquet")

    with open(ROOT / "conf/parameters.yml") as f:
        params = yaml.safe_load(f)

    trades_path = ROOT / "data/05_output/trades.parquet"
    trades = pd.read_parquet(trades_path) if trades_path.exists() else pd.DataFrame()

    zs["timestamp"] = pd.to_datetime(zs["timestamp"], utc=True)
    zs_daily = zs.set_index("timestamp").resample("1D").last()

    spot["timestamp"] = pd.to_datetime(spot["timestamp"], utc=True)
    spot_daily = spot.set_index("timestamp").resample("1D")["close"].last()

    # Forward return: 3d à frente (shift=-3 porque pct_change(3) olha 3 dias atrás)
    ret_3d = spot_daily.pct_change(3).shift(-3) * 100

    return zs_daily, spot_daily, ret_3d, params, trades


# ---------------------------------------------------------------------------
# CAMADA 1 — Estabilidade dos Gates
# ---------------------------------------------------------------------------

def _month_bounds(year_month: str):
    y, m = int(year_month[:4]), int(year_month[5:])
    start = f"{y}-{m:02d}-01"
    if m == 12:
        end = f"{y+1}-01-01"
    else:
        end = f"{y}-{m+1:02d}-01"
    return start, end


def compute_monthly_corr(zs_daily, ret_3d):
    rows   = []
    months = [("Jan", "2026-01"), ("Feb", "2026-02"), ("Mar", "2026-03"), ("Apr", "2026-04")]

    for zcol, (gkey, gname) in GATE_MAP.items():
        if zcol not in zs_daily.columns:
            continue
        row = {"gate": gname, "zcol": zcol, "param_key": gkey}
        for mname, ym in months:
            s, e = _month_bounds(ym)
            z_m     = zs_daily.loc[(zs_daily.index >= s) & (zs_daily.index < e), zcol]
            r_m     = ret_3d.loc[(ret_3d.index >= s) & (ret_3d.index < e)]
            merged  = pd.concat([z_m, r_m], axis=1, join="inner").dropna()
            n       = len(merged)
            row[f"{mname}_n"] = n
            if n >= 10:
                row[mname] = round(float(merged.iloc[:, 0].corr(merged.iloc[:, 1])), 3)
            else:
                row[mname] = None
        rows.append(row)

    return pd.DataFrame(rows)


def compute_rolling_corr(zs_daily, ret_3d, windows=(30, 60, 90)):
    results = {}
    for zcol, (gkey, gname) in GATE_MAP.items():
        if zcol not in zs_daily.columns:
            continue
        df = pd.concat([zs_daily[zcol], ret_3d], axis=1).dropna()
        df.columns = [zcol, "ret_3d"]
        results[gname] = {}
        for w in windows:
            results[gname][f"rolling_{w}d"] = df[zcol].rolling(w).corr(df["ret_3d"])
    return results


def classify_stability(monthly_df, gate_params):
    classifications = []

    for _, row in monthly_df.iterrows():
        gname  = row["gate"]
        gkey   = row["param_key"]
        gp     = gate_params.get("gate_params", {})
        corr_cfg = float(gp.get(gkey, [0])[0]) if gkey in gp else 0.0

        months_valid = [row.get(m) for m in ("Jan", "Feb", "Mar", "Apr") if row.get(m) is not None]

        if len(months_valid) < 2:
            status, reason = "INSUFFICIENT_DATA", "Menos de 2 meses com dados"
        else:
            signs   = [np.sign(v) for v in months_valid]
            n_flips = sum(1 for i in range(1, len(signs))
                         if signs[i] != signs[i-1] and signs[i] != 0 and signs[i-1] != 0)
            mean_abs = np.mean([abs(v) for v in months_valid])
            std_abs  = np.std([abs(v) for v in months_valid])
            last     = months_valid[-1]
            cfg_sign = np.sign(corr_cfg)

            if abs(last) < 0.05:
                status = "BROKEN"
                reason = f"Correlação atual próxima de zero ({last:+.3f})"
            elif n_flips >= 2:
                status = "BROKEN"
                reason = f"{n_flips} inversões de sinal em 2026"
            elif cfg_sign != 0 and np.sign(last) != cfg_sign and abs(last) > 0.10:
                status = "BROKEN"
                reason = f"Sinal invertido vs config (cfg={corr_cfg:+.3f}, atual={last:+.3f})"
            elif mean_abs > 0 and std_abs / mean_abs > 0.40:
                status = "UNSTABLE"
                reason = f"Alta variância ({std_abs:.3f}/{mean_abs:.3f})"
            elif (len(months_valid) >= 3
                  and all(abs(months_valid[i]) < abs(months_valid[i-1])
                          for i in range(1, len(months_valid)))):
                status = "WEAKENING"
                reason = "Magnitude decaindo progressivamente"
            else:
                status = "STABLE"
                reason = "Direção consistente, magnitude estável"

        classifications.append({
            "gate": gname, "corr_cfg": corr_cfg,
            "Jan": row.get("Jan"), "Feb": row.get("Feb"),
            "Mar": row.get("Mar"), "Apr": row.get("Apr"),
            "status": status, "reason": reason,
        })

    return pd.DataFrame(classifications)


def plot_rolling_correlations(rolling_results, output_path):
    gates  = list(rolling_results.keys())
    n_cols = 3
    n_rows = int(np.ceil(len(gates) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, n_rows * 4), sharex=True)
    axes = axes.flatten()

    colors = {"rolling_30d": "#58a6ff", "rolling_60d": "#d29922", "rolling_90d": "#f85149"}
    for i, gname in enumerate(gates):
        ax = axes[i]
        for wname, series in rolling_results[gname].items():
            ax.plot(series.index, series.values,
                    label=wname, color=colors.get(wname), alpha=0.85, linewidth=1.2)
        ax.axhline(0, color="#8b949e", linestyle="--", alpha=0.4, linewidth=0.8)
        ax.set_title(gname, fontsize=11, fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2)
        ax.tick_params(axis="x", rotation=45, labelsize=7)
        ax.set_ylim(-1.1, 1.1)

    for j in range(len(gates), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Rolling Correlations por Gate (30d / 60d / 90d) — 2026", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"   Salvo: {output_path}")


# ---------------------------------------------------------------------------
# CAMADA 4 — Model Health vs PnL
# ---------------------------------------------------------------------------

def compute_daily_alignment(zs_daily, ret_3d, gate_params, window=30):
    gp     = gate_params.get("gate_params", {})
    rows   = []
    dates  = zs_daily.index[window:]

    for date in dates:
        deltas = []
        window_start = date - pd.Timedelta(days=window)
        for zcol, (gkey, gname) in GATE_MAP.items():
            if zcol not in zs_daily.columns or gkey not in gp:
                continue
            cfg      = gp[gkey]
            corr_cfg = float(cfg[0])
            weight   = float(cfg[2]) if len(cfg) >= 3 else 1.0

            z_win  = zs_daily.loc[window_start:date, zcol]
            r_win  = ret_3d.loc[window_start:date]
            merged = pd.concat([z_win, r_win], axis=1).dropna()
            if len(merged) < 10:
                continue

            corr_real = float(merged.iloc[:, 0].corr(merged.iloc[:, 1]))
            deltas.append((abs(corr_real - corr_cfg), weight))

        if deltas:
            total_w  = sum(w for _, w in deltas)
            weighted = sum(d * w for d, w in deltas) / total_w
            rows.append({"date": date, "alignment": weighted, "n_gates": len(deltas)})

    return pd.DataFrame(rows)


def alignment_vs_performance(alignment_df, spot_daily):
    df = alignment_df.copy().set_index("date")

    spot_reindexed      = spot_daily.reindex(df.index)
    df["spot_price"]    = spot_reindexed
    df["forward_ret_7d"]  = ((spot_daily.shift(-7)  / spot_daily - 1) * 100).reindex(df.index)
    df["forward_ret_30d"] = ((spot_daily.shift(-30) / spot_daily - 1) * 100).reindex(df.index)

    daily_ret           = spot_daily.pct_change()
    df["forward_vol_7d"] = (daily_ret.rolling(7).std().shift(-7) * np.sqrt(7) * 100).reindex(df.index)

    def _fwd_dd(idx, n=7):
        try:
            future = spot_daily.loc[idx: idx + pd.Timedelta(days=n)]
            if len(future) < 2:
                return None
            return float((future / future.cummax() - 1).min() * 100)
        except Exception:
            return None

    df["forward_dd_7d"] = [_fwd_dd(d) for d in df.index]

    corrs = {
        "alignment_vs_ret_7d":  float(df["alignment"].corr(df["forward_ret_7d"])),
        "alignment_vs_ret_30d": float(df["alignment"].corr(df["forward_ret_30d"])),
        "alignment_vs_vol_7d":  float(df["alignment"].corr(df["forward_vol_7d"])),
        "alignment_vs_dd_7d":   float(df["alignment"].corr(df["forward_dd_7d"])),
    }
    return df, corrs


def plot_alignment_with_price(alignment_df, spot_daily, output_path):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True,
                                    gridspec_kw={"height_ratios": [1, 1.2]})
    df = alignment_df.set_index("date")

    ax1.plot(df.index, df["alignment"], color="#d29922", linewidth=1.5, label="Alignment 30d")
    ax1.axhline(0.15, color="#3fb950", linestyle="--", alpha=0.6, label="Saudável (<0.15)")
    ax1.axhline(0.30, color="#f85149", linestyle="--", alpha=0.6, label="Desalinhado (>0.30)")
    _ymax = max(df["alignment"].max() * 1.1, 0.35)
    ax1.fill_between(df.index, 0, 0.15,  alpha=0.07, color="#3fb950")
    ax1.fill_between(df.index, 0.15, 0.30, alpha=0.07, color="#d29922")
    ax1.fill_between(df.index, 0.30, _ymax, alpha=0.07, color="#f85149")
    ax1.set_ylim(0, _ymax)
    ax1.set_ylabel("Alignment (|Δcorr| ponderado)")
    ax1.set_title("Model Health Alignment — 2026", fontsize=12, fontweight="bold")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.2)

    spot_plot = spot_daily.reindex(df.index)
    ax2.plot(spot_plot.index, spot_plot.values, color="#58a6ff", linewidth=1.5)
    ax2.set_ylabel("BTC Price (USD)")
    ax2.set_title("BTC Price", fontsize=12)
    ax2.grid(True, alpha=0.2)
    ax2.tick_params(axis="x", rotation=45, labelsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"   Salvo: {output_path}")


# ---------------------------------------------------------------------------
# CAMADA 6 — Detecção de Regime via Alignment
# ---------------------------------------------------------------------------

def classify_regime_by_alignment(alignment_df, thresholds=(0.20, 0.35)):
    df = alignment_df.copy()
    lo, hi = thresholds
    df["regime_alignment"] = df["alignment"].apply(
        lambda a: "STABLE" if a < lo else ("TRANSITION" if a <= hi else "UNSTABLE")
    )
    return df


def performance_by_regime(df):
    required = {"forward_ret_7d", "forward_ret_30d", "forward_vol_7d", "forward_dd_7d"}
    avail    = [c for c in required if c in df.columns]
    agg_cfg  = {c: ["mean", "median", "std", "count"] if c == "forward_ret_7d"
                else ["mean", "std"] for c in avail}
    return df.groupby("regime_alignment").agg(agg_cfg).round(3)


def analyze_regime_transitions(regime_df):
    df = regime_df.copy()
    if "date" not in df.columns:
        df = df.reset_index().rename(columns={"index": "date"})
    df = df.sort_values("date")
    df["regime_change"] = df["regime_alignment"] != df["regime_alignment"].shift(1)
    df["regime_id"]     = df["regime_change"].cumsum()

    periods = (
        df.groupby("regime_id")
        .agg(
            regime        = ("regime_alignment", "first"),
            start         = ("date", "first"),
            end           = ("date", "last"),
            duration_days = ("date", lambda x: (x.max() - x.min()).days + 1),
        )
        .reset_index(drop=True)
    )
    return periods


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def _fmt_corr(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    return f"{v:+.3f}"


def generate_report(stability_df, alignment_df, perf_corrs,
                    perf_by_regime, transitions, monthly_df, params):
    today = date.today().isoformat()
    gp    = params.get("gate_params", {})

    # ── Resumo executivo ──────────────────────────────────────────────────
    n_stable     = (stability_df["status"] == "STABLE").sum()
    n_weakening  = (stability_df["status"] == "WEAKENING").sum()
    n_unstable   = (stability_df["status"] == "UNSTABLE").sum()
    n_broken     = (stability_df["status"] == "BROKEN").sum()

    mean_aln = alignment_df["alignment"].mean()
    max_aln  = alignment_df["alignment"].max()
    last_aln = alignment_df["alignment"].iloc[-1] if len(alignment_df) > 0 else float("nan")

    corr_vs_ret7  = perf_corrs.get("alignment_vs_ret_7d", float("nan"))
    corr_vs_dd7   = perf_corrs.get("alignment_vs_dd_7d", float("nan"))
    corr_vs_vol7  = perf_corrs.get("alignment_vs_vol_7d", float("nan"))

    # ── Camada 1 table ────────────────────────────────────────────────────
    c1_rows = []
    for _, r in stability_df.iterrows():
        gkey     = monthly_df.loc[monthly_df["gate"] == r["gate"], "param_key"].values
        gkey     = gkey[0] if len(gkey) > 0 else ""
        cfg_list = gp.get(gkey, [])
        weight   = f"{cfg_list[2]:.1f}" if len(cfg_list) >= 3 else "—"
        c1_rows.append(
            f"| {r['gate']} | {r['corr_cfg']:+.3f} | {weight} "
            f"| {_fmt_corr(r.get('Jan'))} | {_fmt_corr(r.get('Feb'))} "
            f"| {_fmt_corr(r.get('Mar'))} | {_fmt_corr(r.get('Apr'))} "
            f"| **{r['status']}** | {r['reason']} |"
        )
    c1_table = "\n".join(c1_rows)

    # ── Camada 4 table ────────────────────────────────────────────────────
    c4_rows = "\n".join([
        f"| {k.replace('alignment_vs_', '').replace('_', ' ')} | {v:+.3f} |"
        for k, v in perf_corrs.items()
    ])

    # ── Camada 6 tables ───────────────────────────────────────────────────
    regime_dist = ""
    perf_tbl    = ""
    trans_tbl   = ""
    if not perf_by_regime.empty:
        total_days = perf_by_regime[("forward_ret_7d", "count")].sum() if ("forward_ret_7d", "count") in perf_by_regime.columns else 1
        for reg in ["STABLE", "TRANSITION", "UNSTABLE"]:
            if reg in perf_by_regime.index:
                n_days = int(perf_by_regime.loc[reg, ("forward_ret_7d", "count")]) if ("forward_ret_7d", "count") in perf_by_regime.columns else "?"
                pct    = f"{n_days/total_days*100:.0f}%" if isinstance(n_days, int) else "?"
                regime_dist += f"| {reg} | {n_days} | {pct} |\n"
        for reg in ["STABLE", "TRANSITION", "UNSTABLE"]:
            if reg in perf_by_regime.index:
                r7m = perf_by_regime.loc[reg, ("forward_ret_7d", "mean")] if ("forward_ret_7d", "mean") in perf_by_regime.columns else float("nan")
                r30 = perf_by_regime.loc[reg, ("forward_ret_30d", "mean")] if ("forward_ret_30d", "mean") in perf_by_regime.columns else float("nan")
                vol = perf_by_regime.loc[reg, ("forward_vol_7d", "mean")] if ("forward_vol_7d", "mean") in perf_by_regime.columns else float("nan")
                dd  = perf_by_regime.loc[reg, ("forward_dd_7d", "mean")] if ("forward_dd_7d", "mean") in perf_by_regime.columns else float("nan")
                perf_tbl += f"| {reg} | {_fmt_corr(r7m)} | {_fmt_corr(r30)} | {_fmt_corr(vol)} | {_fmt_corr(dd)} |\n"

    if not transitions.empty:
        for _, t in transitions.iterrows():
            trans_tbl += f"| {t['regime']} | {str(t['start'])[:10]} | {str(t['end'])[:10]} | {t['duration_days']} |\n"

    # ── Interpretações ────────────────────────────────────────────────────
    if abs(corr_vs_ret7) > 0.20:
        c4_interp = (
            f"Alignment apresenta correlação {'negativa' if corr_vs_ret7 < 0 else 'positiva'} "
            f"moderada com retorno 7d ({corr_vs_ret7:+.3f}). "
            "Períodos de desalinhamento tendem a preceder performance pior, sugerindo valor como indicador de risco."
        )
    else:
        c4_interp = (
            f"Correlação entre alignment e retorno 7d é fraca ({corr_vs_ret7:+.3f}). "
            "O alignment ainda pode ser útil como proxy de volatilidade/risco, "
            f"dado que correlaciona {corr_vs_vol7:+.3f} com vol 7d."
        )

    if abs(corr_vs_dd7) > 0.15:
        c4_interp += (
            f" Correlação positiva com drawdown 7d ({corr_vs_dd7:+.3f}) indica que "
            "alignment alto precede drawdowns maiores — útil para position sizing."
        )

    c6_interp = (
        "O alignment como detector de regime separa períodos em que o modelo opera "
        "dentro ou fora de suas premissas de calibração. "
    )
    stable_rows = transitions[transitions["regime"] == "STABLE"]
    if not stable_rows.empty:
        avg_stable = stable_rows["duration_days"].mean()
        c6_interp += f"Regimes STABLE duram em média {avg_stable:.0f} dias. "
    unstable_rows = transitions[transitions["regime"] == "UNSTABLE"]
    if not unstable_rows.empty:
        avg_unstable = unstable_rows["duration_days"].mean()
        c6_interp += f"Regimes UNSTABLE duram {avg_unstable:.0f} dias em média — "
        c6_interp += ("curtos o suficiente para serem detectados antes de causarem perda significativa."
                      if avg_unstable < 14 else
                      "longos, indicando necessidade de adaptação ativa.")

    # ── Recomendações Fase 2 ──────────────────────────────────────────────
    broken_gates = stability_df[stability_df["status"] == "BROKEN"]["gate"].tolist()
    recs = []
    if broken_gates:
        recs.append(
            f"- [ ] Recalibrar gates BROKEN: {', '.join(broken_gates)} — "
            "atualizar `corr_cfg` em `parameters.yml` com valores de 2026"
        )
    weakening_gates = stability_df[stability_df["status"] == "WEAKENING"]["gate"].tolist()
    if weakening_gates:
        recs.append(
            f"- [ ] Monitorar gates WEAKENING: {', '.join(weakening_gates)} — "
            "candidatos a redução de peso (`max_score`) se tendência continuar"
        )
    if abs(corr_vs_vol7) > 0.15:
        recs.append(
            "- [ ] Considerar usar `alignment` como fator de position sizing: "
            "reduzir tamanho quando alignment > 0.30"
        )
    recs.append(
        "- [ ] Implementar re-calibração automática mensal com rolling 90d "
        "(substituir `corr_cfg` se |Δ| > 0.20 por 60+ dias consecutivos)"
    )
    recs_str = "\n".join(recs) if recs else "- [ ] Ver análise acima para detalhes"

    # ── Assemble markdown ─────────────────────────────────────────────────
    report = f"""# Estudo de Adaptação de Gates — Fase 1 (Descritiva)

**Data:** {today}
**Período analisado:** 2026-01-01 → {today}

---

## Resumo Executivo

- **{n_stable} gates STABLE**, {n_weakening} WEAKENING, {n_unstable} UNSTABLE, **{n_broken} BROKEN** dos {len(stability_df)} analisados
- Model Alignment atual: **{last_aln:.3f}** (média 2026: {mean_aln:.3f}, máximo: {max_aln:.3f})
- Correlação alignment vs retorno 7d: **{corr_vs_ret7:+.3f}** — {"prediz performance" if abs(corr_vs_ret7) > 0.15 else "fraca relação com retorno"}
- Correlação alignment vs drawdown 7d: **{corr_vs_dd7:+.3f}** — {"útil como indicador de risco" if abs(corr_vs_dd7) > 0.15 else "relação fraca com drawdown"}
- {"⚠️ Gates BROKEN exigem recalibração antes da Fase 2" if n_broken > 0 else "✅ Nenhum gate em status BROKEN"}

---

## Camada 1 — Estabilidade dos Gates

### Classificação por Gate

| Gate | Config | Peso | Jan | Feb | Mar | Apr | Status | Razão |
|------|--------|------|-----|-----|-----|-----|--------|-------|
{c1_table}

**Legenda:** STABLE = direção consistente | WEAKENING = magnitude decaindo | UNSTABLE = alta variância | BROKEN = sinal invertido ou ~0

### Plot: Rolling Correlations

![Rolling Correlations](plots/fase1/rolling_correlations.png)

### Interpretação

- Gates com `corr_cfg` negativo (OI, Funding, Bubble, F&G) devem manter sinal negativo para funcionar como esperado
- Gates com `corr_cfg` positivo (Stablecoin, ETF, RRP) devem manter sinal positivo
- Magnitude acima de 0.20 indica sinal de entrada robusto; abaixo de 0.05 = ruído

---

## Camada 4 — Model Health vs PnL

### Plot: Alignment + BTC Price

![Alignment 2026](plots/fase1/alignment_time_series.png)

### Correlações Alignment vs Performance

| Métrica | Correlação |
|---------|-----------|
{c4_rows}

### Interpretação

{c4_interp}

---

## Camada 6 — Detecção de Regime via Alignment

### Distribuição de Regimes em 2026

| Regime | N dias | % tempo |
|--------|--------|---------|
{regime_dist.rstrip()}

### Performance Forward por Regime

| Regime | Ret 7d | Ret 30d | Vol 7d | DD 7d |
|--------|--------|---------|--------|-------|
{perf_tbl.rstrip()}

### Durações de Regime

| Regime | Início | Fim | Duração (dias) |
|--------|--------|-----|----------------|
{trans_tbl.rstrip()}

### Interpretação

{c6_interp}

---

## Conclusões e Próximos Passos

### O que aprendemos

1. **Estabilidade dos gates em 2026:** {n_stable}/{len(stability_df)} estáveis — o modelo {'mantém base sólida' if n_stable > n_broken else 'precisa de recalibração significativa'}
2. **Model Alignment como indicador de risco:** correlação de {corr_vs_vol7:+.3f} com volatilidade forward sugere {'valor real como proxy de risco' if abs(corr_vs_vol7) > 0.15 else 'uso limitado como preditor isolado'}
3. **Regime detection via Alignment:** o separador 0.20/0.35 {'discrimina bem STABLE vs UNSTABLE' if not transitions.empty else 'requer mais dados'} — {'útil para position sizing adaptativo' if abs(corr_vs_dd7) > 0.15 else 'mais estudo necessário'}

### Recomendações para Fase 2

{recs_str}

### Questões em aberto

- [ ] Qual threshold de alignment deve disparar redução de capital exposto?
- [ ] Vale substituir `corr_cfg` automaticamente ou manter controle manual?
- [ ] O alignment de curto prazo (14d) seria mais sensível a mudanças de regime?
"""

    report_path = OUTPUT_DIR / "estudo_adaptacao_gates_fase1.md"
    report_path.write_text(report, encoding="utf-8")
    print(f"   Relatório: {report_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("Estudo de Adaptação de Gates — Fase 1 (Descritiva)")
    print("=" * 60)

    print("\n1. Carregando dados...")
    zs_daily, spot_daily, ret_3d, params, trades = load_data()
    print(f"   Z-scores: {len(zs_daily)} dias ({zs_daily.index[0].date()} → {zs_daily.index[-1].date()})")
    print(f"   Spot:     {len(spot_daily)} dias ({spot_daily.index[0].date()} → {spot_daily.index[-1].date()})")
    print(f"   Trades:   {len(trades)}")

    # ── Camada 1 ──────────────────────────────────────────────────────────
    print("\n2. CAMADA 1 — Estabilidade dos Gates...")
    monthly_df   = compute_monthly_corr(zs_daily, ret_3d)
    rolling_res  = compute_rolling_corr(zs_daily, ret_3d)
    stability_df = classify_stability(monthly_df, params)

    monthly_df.to_csv(TABLES_DIR / "monthly_correlations.csv", index=False)
    stability_df.to_csv(TABLES_DIR / "gate_stability.csv", index=False)
    plot_rolling_correlations(rolling_res, PLOTS_DIR / "rolling_correlations.png")
    print(f"   Status: {stability_df['status'].value_counts().to_dict()}")
    for _, r in stability_df.iterrows():
        emoji = {"STABLE": "✅", "WEAKENING": "🟡", "UNSTABLE": "⚠️", "BROKEN": "🔴"}.get(r["status"], "?")
        print(f"   {emoji} {r['gate']}: {r['status']} — {r['reason']}")

    # ── Camada 4 ──────────────────────────────────────────────────────────
    print("\n3. CAMADA 4 — Model Health vs PnL...")
    alignment_df              = compute_daily_alignment(zs_daily, ret_3d, params, window=30)
    alignment_with_perf, corrs = alignment_vs_performance(alignment_df, spot_daily)

    alignment_df.to_csv(TABLES_DIR / "daily_alignment.csv", index=False)
    alignment_with_perf.to_csv(TABLES_DIR / "alignment_vs_performance.csv")
    plot_alignment_with_price(alignment_df, spot_daily, PLOTS_DIR / "alignment_time_series.png")
    print(f"   Alignment atual: {alignment_df['alignment'].iloc[-1]:.3f}")
    for k, v in corrs.items():
        print(f"   {k}: {v:+.3f}")

    # ── Camada 6 ──────────────────────────────────────────────────────────
    print("\n4. CAMADA 6 — Detecção de Regime...")
    regime_df    = classify_regime_by_alignment(alignment_with_perf.reset_index())
    perf_regime  = performance_by_regime(regime_df)
    transitions  = analyze_regime_transitions(regime_df)

    regime_df.to_csv(TABLES_DIR / "regime_by_alignment.csv", index=False)
    perf_regime.to_csv(TABLES_DIR / "performance_by_regime.csv")
    transitions.to_csv(TABLES_DIR / "regime_transitions.csv", index=False)
    print(f"   Distribuição: {regime_df['regime_alignment'].value_counts().to_dict()}")

    # ── Relatório ─────────────────────────────────────────────────────────
    print("\n5. Gerando relatório markdown...")
    generate_report(stability_df, alignment_df, corrs,
                    perf_regime, transitions, monthly_df, params)

    print(f"\n✅ Estudo completo.")
    print(f"   Relatório: {OUTPUT_DIR}/estudo_adaptacao_gates_fase1.md")
    print(f"   Plots:     {PLOTS_DIR}/")
    print(f"   Tabelas:   {TABLES_DIR}/")


if __name__ == "__main__":
    main()
