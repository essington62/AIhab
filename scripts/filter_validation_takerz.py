"""
Mini-Backtest: Validação filtro taker_z.

Carrega dataset do error_analysis_losers (136 trades, 2026).
Responde:
  1. Filtro taker_z < -0.96 melhora Sharpe? Expectância? P&L?
  2. Threshold é overfit (sensível ao valor exato)?
  3. Combinação com funding_z adiciona valor?
  4. Filtro funciona em sub-períodos (tercis temporais)?

Outputs:
  prompts/filter_validation_report.md
  prompts/tables/filter_validation_scenarios.csv
  prompts/plots/filter_validation/
    pnl_comparison.png
    threshold_sensitivity.png
    subperiod_robustness.png
    equity_curves.png
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
logger = logging.getLogger("filter_validation")

PLOTS_DIR = ROOT / "prompts/plots/filter_validation"
TABLES_DIR = ROOT / "prompts/tables"
REPORT_PATH = ROOT / "prompts/filter_validation_report.md"
SCENARIOS_CSV = TABLES_DIR / "filter_validation_scenarios.csv"

for d in [PLOTS_DIR, TABLES_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ==================================================================
# LOAD DATASET
# ==================================================================

def load_dataset() -> pd.DataFrame:
    path = ROOT / "prompts/tables/error_analysis_full_dataset.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}\nRun error_analysis_losers.py first."
        )
    df = pd.read_csv(path)
    df["entry_time"] = pd.to_datetime(df["entry_time"], utc=True)
    df = df.sort_values("entry_time").reset_index(drop=True)
    df["is_loser"] = ~df["is_winner"]
    logger.info(f"Loaded {len(df)} trades | WR={df['is_winner'].mean()*100:.1f}%")
    return df


# ==================================================================
# METRICS
# ==================================================================

def compute_metrics(returns: np.ndarray, trades_per_year: float = 52) -> dict:
    r = np.asarray([x for x in returns if x is not None and not np.isnan(x)])
    if len(r) == 0:
        return dict(n_trades=0, win_rate_pct=0, avg_return_pct=0,
                    avg_win_pct=0, avg_loss_pct=0, expectancy_pct=0,
                    profit_factor=0, total_return_pct=0,
                    sharpe=0, sortino=0, max_dd_pct=0)

    wins = r[r > 0]
    losses = r[r < 0]

    wr = len(wins) / len(r) * 100
    avg_win = wins.mean() * 100 if len(wins) else 0.0
    avg_loss = losses.mean() * 100 if len(losses) else 0.0
    pf = wins.sum() / abs(losses.sum()) if losses.sum() != 0 else float("inf")

    total_ret = (np.prod(1 + r) - 1) * 100

    std = r.std()
    sharpe = (r.mean() / std) * np.sqrt(trades_per_year) if std > 0 else 0.0

    d_std = losses.std() if len(losses) > 1 else 0.0
    sortino = (r.mean() / d_std) * np.sqrt(trades_per_year) if d_std > 0 else float("inf")

    cum = np.cumprod(1 + r)
    peak = np.maximum.accumulate(cum)
    max_dd = ((cum - peak) / peak).min() * 100

    return dict(
        n_trades=len(r),
        win_rate_pct=round(wr, 2),
        avg_return_pct=round(r.mean() * 100, 4),
        avg_win_pct=round(avg_win, 4),
        avg_loss_pct=round(avg_loss, 4),
        expectancy_pct=round(r.mean() * 100, 4),
        profit_factor=round(pf, 3),
        total_return_pct=round(total_ret, 3),
        sharpe=round(sharpe, 3),
        sortino=round(sortino, 3),
        max_dd_pct=round(max_dd, 3),
    )


def apply_filter(df: pd.DataFrame, mask: pd.Series, name: str) -> dict:
    """mask=True → trade is BLOCKED."""
    kept = df[~mask]
    blocked = df[mask]

    m = compute_metrics(kept["return_pct"].values)
    n_bl = int(mask.sum())
    n_bl_losers = int(blocked["is_loser"].sum())
    n_bl_winners = n_bl - n_bl_losers

    return {
        "scenario": name,
        "n_blocked": n_bl,
        "n_blocked_losers": n_bl_losers,
        "n_blocked_winners": n_bl_winners,
        "block_ratio": round(n_bl_losers / max(n_bl_winners, 1), 2),
        **m,
    }


# ==================================================================
# CENÁRIOS
# ==================================================================

def run_scenarios(df: pd.DataFrame) -> pd.DataFrame:
    scenarios = []

    # Baseline
    scenarios.append(apply_filter(df, pd.Series(False, index=df.index), "BASELINE"))

    # taker_z sweep
    for t in [-0.5, -0.8, -0.96, -1.0, -1.2, -1.5]:
        scenarios.append(apply_filter(df, df["taker_z"] < t, f"taker_z < {t}"))

    # funding_z sweep
    for t in [-0.8, -1.0, -1.37, -1.5]:
        scenarios.append(apply_filter(df, df["funding_z"] < t, f"funding_z < {t}"))

    # Combinações
    scenarios.append(apply_filter(
        df, (df["taker_z"] < -0.96) & (df["funding_z"] < -1.37),
        "taker_z<-0.96 AND funding_z<-1.37"
    ))
    scenarios.append(apply_filter(
        df, (df["taker_z"] < -0.96) | (df["funding_z"] < -1.37),
        "taker_z<-0.96 OR funding_z<-1.37"
    ))
    scenarios.append(apply_filter(
        df, (df["taker_z"] < -1.0) | (df["funding_z"] < -1.0),
        "taker_z<-1.0 OR funding_z<-1.0 (robusto)"
    ))

    # close_vs_ma200
    scenarios.append(apply_filter(df, df["close_vs_ma200"] > 0.04, "close_vs_ma200 > 0.04"))

    # Triplo
    scenarios.append(apply_filter(
        df,
        (df["taker_z"] < -0.96) & (df["funding_z"] < -1.37) & (df["close_vs_ma200"] > 0.04),
        "TRIPLE AND"
    ))

    return pd.DataFrame(scenarios)


# ==================================================================
# SUB-PERÍODOS
# ==================================================================

def test_subperiods(df: pd.DataFrame) -> pd.DataFrame:
    n = len(df)
    t = n // 3
    slices = {
        "T1 (trades 1-45)": df.iloc[:t],
        "T2 (trades 46-90)": df.iloc[t:2*t],
        "T3 (trades 91-136)": df.iloc[2*t:],
    }

    rows = []
    for label, sub in slices.items():
        base = compute_metrics(sub["return_pct"].values)
        base["scenario"] = f"{label} — BASELINE"
        base["n_blocked"] = 0
        rows.append(base)

        mask = sub["taker_z"] < -0.96
        kept = sub[~mask]
        blocked = sub[mask]
        m = compute_metrics(kept["return_pct"].values)
        m["scenario"] = f"{label} — taker_z<-0.96"
        m["n_blocked"] = int(mask.sum())
        m["n_blocked_losers"] = int(blocked["is_loser"].sum())
        m["n_blocked_winners"] = len(blocked) - int(blocked["is_loser"].sum())
        rows.append(m)

    return pd.DataFrame(rows)


# ==================================================================
# EQUITY CURVES
# ==================================================================

def compute_equity_curves(df: pd.DataFrame) -> pd.DataFrame:
    d = df.sort_values("entry_time").copy().reset_index(drop=True)

    d["eq_baseline"] = (np.cumprod(1 + d["return_pct"]) - 1) * 100

    m1 = d["taker_z"] < -0.96
    r1 = d["return_pct"].where(~m1, 0)
    d["eq_taker"] = (np.cumprod(1 + r1) - 1) * 100

    m2 = (d["taker_z"] < -0.96) | (d["funding_z"] < -1.37)
    r2 = d["return_pct"].where(~m2, 0)
    d["eq_or"] = (np.cumprod(1 + r2) - 1) * 100

    m3 = (d["taker_z"] < -1.0) | (d["funding_z"] < -1.0)
    r3 = d["return_pct"].where(~m3, 0)
    d["eq_robust"] = (np.cumprod(1 + r3) - 1) * 100

    return d


# ==================================================================
# PLOTS
# ==================================================================

def generate_plots(df_sc: pd.DataFrame, df_sub: pd.DataFrame, df_eq: pd.DataFrame):
    key = [
        "BASELINE",
        "taker_z < -0.96",
        "taker_z < -1.0",
        "funding_z < -1.37",
        "taker_z<-0.96 OR funding_z<-1.37",
        "taker_z<-0.96 AND funding_z<-1.37",
        "taker_z<-1.0 OR funding_z<-1.0 (robusto)",
    ]
    df_key = df_sc[df_sc["scenario"].isin(key)].reset_index(drop=True)

    # ── Plot 1: 4-panel comparison ─────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    x = np.arange(len(df_key))
    colors = ["#d62728" if r["scenario"] == "BASELINE" else "#1f77b4"
              for _, r in df_key.iterrows()]

    for ax, col, title, ylabel in [
        (axes[0, 0], "sharpe",          "Sharpe Ratio",            "Sharpe"),
        (axes[0, 1], "expectancy_pct",  "Expectância por Trade",   "Expectância %"),
        (axes[1, 0], "total_return_pct","Retorno Total Composto",  "Return %"),
        (axes[1, 1], "max_dd_pct",      "Max Drawdown",            "Max DD %"),
    ]:
        ax.bar(x, df_key[col], color=colors, alpha=0.85, edgecolor="white")
        baseline_val = df_key.loc[df_key["scenario"] == "BASELINE", col]
        if len(baseline_val):
            ax.axhline(baseline_val.iloc[0], color="red", lw=1.5, ls="--", label="Baseline")
        ax.set_xticks(x)
        ax.set_xticklabels(df_key["scenario"], rotation=40, ha="right", fontsize=8)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(alpha=0.3, axis="y")
        ax.legend(fontsize=8)

    plt.suptitle("Comparação de Cenários — Filtro Taker_Z", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "pnl_comparison.png", dpi=100, bbox_inches="tight")
    plt.close()

    # ── Plot 2: threshold sensitivity ──────────────────────────────
    taker = df_sc[df_sc["scenario"].str.match(r"taker_z < -[\d.]+")].copy()
    taker["threshold"] = taker["scenario"].str.extract(r"< (-[\d.]+)").astype(float)
    taker = taker.sort_values("threshold")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    base_sharpe = df_sc.loc[df_sc["scenario"] == "BASELINE", "sharpe"].iloc[0]
    base_exp = df_sc.loc[df_sc["scenario"] == "BASELINE", "expectancy_pct"].iloc[0]

    for ax, col, base_val, label, color in [
        (axes[0], "sharpe",         base_sharpe, "Sharpe",       "steelblue"),
        (axes[1], "expectancy_pct", base_exp,    "Expectância %","seagreen"),
    ]:
        ax.plot(taker["threshold"], taker[col], "o-", color=color, lw=2, ms=9)
        ax.axhline(base_val, color="red", ls="--", lw=1.5, label="Baseline")
        ax.axvline(-0.96, color="grey", ls=":", lw=1.2, label="threshold original")
        ax.set_xlabel("taker_z threshold")
        ax.set_ylabel(label)
        ax.set_title(f"Sensibilidade ao Threshold: {label}")
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "threshold_sensitivity.png", dpi=100, bbox_inches="tight")
    plt.close()

    # ── Plot 3: sub-period robustness ──────────────────────────────
    periods = ["T1 (trades 1-45)", "T2 (trades 46-90)", "T3 (trades 91-136)"]
    bs_sh, fl_sh, bs_ex, fl_ex = [], [], [], []
    for p in periods:
        b = df_sub[df_sub["scenario"] == f"{p} — BASELINE"]
        f = df_sub[df_sub["scenario"] == f"{p} — taker_z<-0.96"]
        if len(b) and len(f):
            bs_sh.append(b["sharpe"].iloc[0]); fl_sh.append(f["sharpe"].iloc[0])
            bs_ex.append(b["expectancy_pct"].iloc[0]); fl_ex.append(f["expectancy_pct"].iloc[0])

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    xi = np.arange(len(periods))
    w = 0.35
    for ax, bvals, fvals, title, ylabel in [
        (axes[0], bs_sh, fl_sh, "Robustez — Sharpe",       "Sharpe"),
        (axes[1], bs_ex, fl_ex, "Robustez — Expectância",  "Expectância %"),
    ]:
        ax.bar(xi - w/2, bvals, w, label="Baseline", color="#d62728", alpha=0.75)
        ax.bar(xi + w/2, fvals, w, label="taker_z<-0.96", color="#1f77b4", alpha=0.75)
        ax.set_xticks(xi); ax.set_xticklabels(periods)
        ax.set_ylabel(ylabel); ax.set_title(title)
        ax.legend(); ax.grid(alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "subperiod_robustness.png", dpi=100, bbox_inches="tight")
    plt.close()

    # ── Plot 4: equity curves ──────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 6))
    xi_eq = np.arange(len(df_eq))
    ax.plot(xi_eq, df_eq["eq_baseline"], color="red",      lw=2, label="Baseline")
    ax.plot(xi_eq, df_eq["eq_taker"],    color="steelblue",lw=2, label="taker_z < -0.96")
    ax.plot(xi_eq, df_eq["eq_or"],       color="green",    lw=2, label="taker_z<-0.96 OR funding_z<-1.37")
    ax.plot(xi_eq, df_eq["eq_robust"],   color="purple",   lw=2, ls="--", label="taker_z<-1.0 OR funding_z<-1.0")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel("Trade #")
    ax.set_ylabel("Retorno acumulado %")
    ax.set_title("Curvas de Capital: Baseline vs Filtros")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "equity_curves.png", dpi=100, bbox_inches="tight")
    plt.close()

    logger.info(f"Plots → {PLOTS_DIR}")


# ==================================================================
# REPORT
# ==================================================================

def generate_report(df_sc: pd.DataFrame, df_sub: pd.DataFrame):
    baseline = df_sc[df_sc["scenario"] == "BASELINE"].iloc[0]
    filter_only = df_sc[df_sc["scenario"] != "BASELINE"]

    best_sharpe_row = filter_only.loc[filter_only["sharpe"].idxmax()]
    best_exp_row = filter_only.loc[filter_only["expectancy_pct"].idxmax()]
    main_filter = df_sc[df_sc["scenario"] == "taker_z < -0.96"].iloc[0]
    robust_filter = df_sc[df_sc["scenario"] == "taker_z<-1.0 OR funding_z<-1.0 (robusto)"].iloc[0]

    lines = [
        "# Mini-Backtest: Validação Filtro Taker_Z",
        f"\n**Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        f"**Dataset:** 136 trades, 2026-02-24 → 2026-04-16",
        "",
        "## 1. Comparação de Cenários",
        "",
        "| Cenário | N | WR% | Expectância | Sharpe | Total Ret% | Max DD% | Bloq |",
        "|---------|---|-----|-------------|--------|------------|---------|------|",
    ]
    for _, row in df_sc.iterrows():
        lines.append(
            f"| {row['scenario']} | {row['n_trades']} | {row['win_rate_pct']:.1f}% "
            f"| {row['expectancy_pct']:+.3f}% | {row['sharpe']:.2f} "
            f"| {row['total_return_pct']:+.2f}% | {row['max_dd_pct']:.1f}% "
            f"| {int(row.get('n_blocked', 0))} |"
        )

    lines += [
        "",
        "## 2. Sensibilidade ao Threshold (Overfit Check)",
        "",
        "| Threshold | Sharpe | Δ Sharpe | Expectância | Δ Expect | Bloq (L/W) |",
        "|-----------|--------|----------|-------------|----------|------------|",
    ]
    taker = df_sc[df_sc["scenario"].str.match(r"taker_z < -[\d.]+")].copy()
    taker["threshold"] = taker["scenario"].str.extract(r"< (-[\d.]+)").astype(float)
    for _, row in taker.sort_values("threshold").iterrows():
        ds = row["sharpe"] - baseline["sharpe"]
        de = row["expectancy_pct"] - baseline["expectancy_pct"]
        lines.append(
            f"| {row['scenario']} | {row['sharpe']:.2f} | {ds:+.2f} "
            f"| {row['expectancy_pct']:+.3f}% | {de:+.3f}pp "
            f"| {int(row.get('n_blocked', 0))} ({int(row.get('n_blocked_losers',0))}L/{int(row.get('n_blocked_winners',0))}W) |"
        )

    lines += [
        "",
        "## 3. Robustez em Sub-Períodos",
        "",
        "| Período | N | WR% | Sharpe | Expectância | Bloqueados |",
        "|---------|---|-----|--------|-------------|------------|",
    ]
    for _, row in df_sub.iterrows():
        bl = f"{int(row.get('n_blocked', 0))}"
        if pd.notna(row.get("n_blocked_losers")) and row.get("n_blocked_losers", 0) > 0:
            bl += f" ({int(row['n_blocked_losers'])}L/{int(row.get('n_blocked_winners', 0))}W)"
        lines.append(
            f"| {row['scenario']} | {row['n_trades']} | {row['win_rate_pct']:.1f}% "
            f"| {row['sharpe']:.2f} | {row['expectancy_pct']:+.3f}% | {bl} |"
        )

    # Robustez CV
    f_sub = df_sub[df_sub["scenario"].str.contains("taker_z")]
    sharpe_cv = (f_sub["sharpe"].std() / f_sub["sharpe"].mean() * 100) if f_sub["sharpe"].mean() > 0 else 999

    # Threshold estabilidade
    taker_narrow = taker[(taker["threshold"] >= -1.2) & (taker["threshold"] <= -0.8)]
    sharpe_range = taker_narrow["sharpe"].max() - taker_narrow["sharpe"].min() if len(taker_narrow) else 0

    ds_main = main_filter["sharpe"] - baseline["sharpe"]
    de_main = main_filter["expectancy_pct"] - baseline["expectancy_pct"]
    dd_main = main_filter["max_dd_pct"] - baseline["max_dd_pct"]

    # Decision
    all_periods_improve = all(
        df_sub[df_sub["scenario"] == f"{p} — taker_z<-0.96"]["sharpe"].iloc[0] >
        df_sub[df_sub["scenario"] == f"{p} — BASELINE"]["sharpe"].iloc[0]
        for p in ["T1 (trades 1-45)", "T2 (trades 46-90)", "T3 (trades 91-136)"]
        if len(df_sub[df_sub["scenario"] == f"{p} — taker_z<-0.96"]) and
           len(df_sub[df_sub["scenario"] == f"{p} — BASELINE"])
    )

    if ds_main > 0.2 and de_main > 0 and sharpe_range < 0.5 and sharpe_cv < 50 and all_periods_improve:
        verdict = "✅ INTEGRAR"
        verdict_detail = (
            "Filtro melhora Sharpe (+{:.2f}), expectância ({:+.3f}pp), "
            "é robusto ao threshold (range Sharpe={:.2f} entre -0.8 e -1.2) "
            "e consistente em todos os sub-períodos (CV={:.0f}%).".format(
                ds_main, de_main, sharpe_range, sharpe_cv
            )
        )
        next_step = "Implementar `taker_z < -1.0` em `paper_trader.py` (threshold round, mais robusto que -0.96)."
    elif ds_main > 0 and de_main > 0:
        verdict = "⚠️ MARGINAL"
        verdict_detail = (
            "Melhoria existe (Sharpe {:+.2f}, expectância {:+.3f}pp) mas "
            "{}{}. Avaliar custo de adicionar complexidade.".format(
                ds_main, de_main,
                "threshold sensível (range={:.2f}) ".format(sharpe_range) if sharpe_range >= 0.5 else "",
                "inconsistente entre sub-períodos (CV={:.0f}%) ".format(sharpe_cv) if sharpe_cv >= 50 else "",
            ).strip()
        )
        next_step = "Paper monitor por 4 semanas antes de decidir."
    else:
        verdict = "❌ REJEITAR"
        verdict_detail = (
            "Filtro não melhora o sistema (Sharpe {:+.2f}, expectância {:+.3f}pp).".format(
                ds_main, de_main
            )
        )
        next_step = "Filtro não agrega valor."

    lines += [
        "",
        "## 4. Veredito",
        "",
        f"### {verdict}",
        "",
        f"**Filtro principal:** `taker_z < -0.96`",
        f"- Sharpe: {baseline['sharpe']:.2f} → {main_filter['sharpe']:.2f} ({ds_main:+.2f})",
        f"- Expectância: {baseline['expectancy_pct']:+.3f}% → {main_filter['expectancy_pct']:+.3f}% ({de_main:+.3f}pp)",
        f"- Total Return: {baseline['total_return_pct']:+.2f}% → {main_filter['total_return_pct']:+.2f}%",
        f"- Max DD: {baseline['max_dd_pct']:.1f}% → {main_filter['max_dd_pct']:.1f}% ({dd_main:+.1f}pp)",
        f"- Bloqueados: {int(main_filter['n_blocked'])} "
          f"({int(main_filter['n_blocked_losers'])}L / {int(main_filter['n_blocked_winners'])}W)",
        "",
        f"**Avaliação:**",
        f"{verdict_detail}",
        "",
        f"**Filtro robusto alternativo:** `taker_z<-1.0 OR funding_z<-1.0`",
        f"- Sharpe: {robust_filter['sharpe']:.2f} ({robust_filter['sharpe']-baseline['sharpe']:+.2f})",
        f"- Expectância: {robust_filter['expectancy_pct']:+.3f}% "
          f"({robust_filter['expectancy_pct']-baseline['expectancy_pct']:+.3f}pp)",
        "",
        f"**Melhor filtro geral (Sharpe):** `{best_sharpe_row['scenario']}`",
        f"- Sharpe: {best_sharpe_row['sharpe']:.2f} ({best_sharpe_row['sharpe']-baseline['sharpe']:+.2f})",
        "",
        f"**Próximo passo:** {next_step}",
        "",
        "## 5. Análise de Robustez",
        "",
        f"- **Threshold stability (range Sharpe -0.8→-1.2):** {sharpe_range:.3f} "
          f"{'✅ robusto' if sharpe_range < 0.5 else '⚠️ sensível'}",
        f"- **Consistência sub-períodos (CV):** {sharpe_cv:.1f}% "
          f"{'✅ consistente' if sharpe_cv < 50 else '⚠️ inconsistente'}",
        f"- **Melhora em todos os tercis:** {'✅ sim' if all_periods_improve else '❌ não'}",
        "",
        "## 6. Visualizações",
        "",
        "- `prompts/plots/filter_validation/pnl_comparison.png`",
        "- `prompts/plots/filter_validation/threshold_sensitivity.png`",
        "- `prompts/plots/filter_validation/subperiod_robustness.png`",
        "- `prompts/plots/filter_validation/equity_curves.png`",
    ]

    REPORT_PATH.write_text("\n".join(lines))
    logger.info(f"Report → {REPORT_PATH}")


# ==================================================================
# MAIN
# ==================================================================

def run_study():
    logger.info("=" * 60)
    logger.info("Mini-Backtest: Validação Filtro Taker_Z")
    logger.info("=" * 60)

    df = load_dataset()

    df_sc = run_scenarios(df)
    df_sc.to_csv(SCENARIOS_CSV, index=False)

    df_sub = test_subperiods(df)
    df_eq = compute_equity_curves(df)

    print("\n" + "=" * 60)
    print("SCENARIOS")
    print("=" * 60)
    cols = ["scenario", "n_trades", "win_rate_pct", "expectancy_pct",
            "sharpe", "total_return_pct", "max_dd_pct", "n_blocked"]
    print(df_sc[cols].to_string(index=False))

    print("\n" + "=" * 60)
    print("SUB-PERIODS")
    print("=" * 60)
    print(df_sub[["scenario", "n_trades", "win_rate_pct",
                  "expectancy_pct", "sharpe"]].to_string(index=False))

    generate_plots(df_sc, df_sub, df_eq)
    generate_report(df_sc, df_sub)

    logger.info("Done. Report: " + str(REPORT_PATH))


if __name__ == "__main__":
    run_study()
