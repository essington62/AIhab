"""
Adaptive Stops v1 — Estudo de Validação.

Identifica os mesmos sinais históricos do Bot 2 (mesma lógica de _signal_passes),
simula cada trade com stops fixos (atual) e adaptativos (ATR z-score),
e compara expectância, Sharpe e WR.

Dados:
  data/02_intermediate/spot/btc_1h_clean.parquet  — indicadores
  data/01_raw/spot/btc_1h.parquet                 — OHLCV cru (high/low para ATR)
  data/02_features/gate_zscores.parquet           — stablecoin_z

Outputs:
  prompts/adaptive_stops_v1_report.md
  prompts/tables/adaptive_stops_v1_trades.csv
  prompts/plots/adaptive_stops_v1/
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
logger = logging.getLogger("adaptive_stops")

PLOTS_DIR = ROOT / "prompts/plots/adaptive_stops_v1"
TABLES_DIR = ROOT / "prompts/tables"
REPORT_PATH = ROOT / "prompts/adaptive_stops_v1_report.md"
TRADES_CSV = TABLES_DIR / "adaptive_stops_v1_trades.csv"

for d in [PLOTS_DIR, TABLES_DIR, REPORT_PATH.parent]:
    d.mkdir(parents=True, exist_ok=True)

# Config atual (fixed)
FIXED_SL = 0.015
FIXED_TP = 0.020
FIXED_TRAIL = 0.010
MAX_HOLD_H = 120


# ==========================================================
# DATA LOADING
# ==========================================================

def load_data() -> pd.DataFrame:
    """
    Carrega clean parquet + gate z-scores.
    O clean parquet já tem high/low/close — não precisa de merge com raw.
    Retorna df 1h com todos os indicadores necessários.
    """
    df = pd.read_parquet(ROOT / "data/02_intermediate/spot/btc_1h_clean.parquet")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)

    # ret_1d
    df["ret_1d"] = df["close"].pct_change(24)

    # Gate z-scores
    zs = pd.read_parquet(ROOT / "data/02_features/gate_zscores.parquet")
    zs["timestamp"] = pd.to_datetime(zs["timestamp"], utc=True)
    df = df.merge(zs[["timestamp", "stablecoin_z"]], on="timestamp", how="left")
    df["stablecoin_z"] = df["stablecoin_z"].ffill()

    # ATR 14h — True Range com high/low/close do clean parquet
    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    df["atr_14"] = tr.rolling(14).mean()
    df["atr_pct"] = df["atr_14"] / close

    # Filtra a janela de estudo (2026)
    df = df[df["timestamp"] >= pd.Timestamp("2026-01-01", tz="UTC")].reset_index(drop=True)
    logger.info(f"Loaded {len(df)} rows: {df['timestamp'].min().date()} → {df['timestamp'].max().date()}")
    return df


# ==========================================================
# SIGNAL IDENTIFICATION (mesmo que mfe_mae_study_bot2.py)
# ==========================================================

def _signal_passes(row) -> bool:
    """Bot 2 current filter — produção."""
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


# ==========================================================
# VOLATILITY CLASSIFICATION
# ==========================================================

def classify_volatility(atr_pct_now: float, atr_pct_history: pd.Series) -> dict:
    history = atr_pct_history.dropna()

    if len(history) < 20 or pd.isna(atr_pct_now):
        return {
            "label": "NORMAL",
            "atr_pct": atr_pct_now,
            "z_score": 0.0,
            "sl_multiplier": 1.5,
            "tp_multiplier": 2.0,
        }

    mean = history.mean()
    std = history.std()
    z = (atr_pct_now - mean) / std if std > 0 else 0.0

    if atr_pct_now > 0.035:
        label, sl_m, tp_m = "EXTREME",      2.0, 3.0
    elif z > 1.5:
        label, sl_m, tp_m = "VOLATILE",     1.8, 2.8
    elif z > 0.5:
        label, sl_m, tp_m = "ABOVE_NORMAL", 1.6, 2.4
    elif z > -0.5:
        label, sl_m, tp_m = "NORMAL",       1.5, 2.0
    elif z > -1.5:
        label, sl_m, tp_m = "CALM",         1.3, 1.8
    else:
        label, sl_m, tp_m = "MUCH_CALMER",  1.2, 1.7

    return {
        "label": label,
        "atr_pct": atr_pct_now,
        "z_score": z,
        "sl_multiplier": sl_m,
        "tp_multiplier": tp_m,
    }


def compute_adaptive_stops(entry_price: float, regime: dict) -> dict:
    atr_pct = regime["atr_pct"]
    sl_pct = max(0.008, min(0.030, regime["sl_multiplier"] * atr_pct))
    tp_pct = max(0.012, min(0.050, regime["tp_multiplier"] * atr_pct))
    return {
        "sl_pct": sl_pct,
        "tp_pct": tp_pct,
        "sl_price": entry_price * (1 - sl_pct),
        "tp_price": entry_price * (1 + tp_pct),
    }


# ==========================================================
# TRADE SIMULATION
# ==========================================================

def simulate_trade(df: pd.DataFrame, entry_idx: int,
                   entry_price: float, sl_pct: float, tp_pct: float,
                   trail_pct: float = 0.010, max_hold: int = MAX_HOLD_H) -> dict:
    """
    Simula um trade usando os candles subsequentes (high/low para trigger preciso).
    Retorna exit_reason, exit_price, return_pct, hours_held.
    """
    sl_price = entry_price * (1 - sl_pct)
    tp_price = entry_price * (1 + tp_pct)
    trailing_high = entry_price

    end_idx = min(entry_idx + max_hold, len(df) - 1)

    for i in range(entry_idx + 1, end_idx + 1):
        row = df.iloc[i]
        high = row.get("high", row["close"])
        low = row.get("low", row["close"])
        close = row["close"]
        hours = i - entry_idx

        # Atualiza trailing high
        if high > trailing_high:
            trailing_high = high

        # TP (high do candle toca TP)
        if high >= tp_price:
            return {"exit_reason": "TP", "exit_price": tp_price,
                    "return_pct": tp_pct, "hours_held": hours}

        # SL (low do candle toca SL)
        if low <= sl_price:
            return {"exit_reason": "SL", "exit_price": sl_price,
                    "return_pct": -sl_pct, "hours_held": hours}

        # Trailing (só se em lucro)
        if close > entry_price:
            trailing_stop = trailing_high * (1 - trail_pct)
            if close <= trailing_stop and trailing_stop > entry_price:
                ret = (trailing_stop - entry_price) / entry_price
                return {"exit_reason": "TRAIL", "exit_price": trailing_stop,
                        "return_pct": ret, "hours_held": hours}

    # Timeout — fecha no close do último candle
    last = df.iloc[end_idx]
    ret = (last["close"] - entry_price) / entry_price
    return {"exit_reason": "TIMEOUT", "exit_price": last["close"],
            "return_pct": ret, "hours_held": end_idx - entry_idx}


# ==========================================================
# METRICS
# ==========================================================

def compute_metrics(returns: np.ndarray) -> dict:
    r = np.array([x for x in returns if x is not None and not np.isnan(x)])
    if len(r) == 0:
        return {}

    wr = (r > 0).mean() * 100
    avg = r.mean() * 100
    total = (np.prod(1 + r) - 1) * 100
    std = r.std()
    sharpe = (r.mean() / std) * np.sqrt(52) if std > 0 else 0.0

    cum = np.cumprod(1 + r)
    peak = np.maximum.accumulate(cum)
    max_dd = ((cum - peak) / peak).min() * 100

    return {
        "n_trades": len(r),
        "win_rate_pct": wr,
        "avg_return_pct": avg,
        "total_return_pct": total,
        "sharpe": sharpe,
        "max_dd_pct": max_dd,
    }


# ==========================================================
# MAIN STUDY
# ==========================================================

def run_study():
    logger.info("=" * 60)
    logger.info("Adaptive Stops v1 — Estudo de Validação")
    logger.info("=" * 60)

    df = load_data()

    # ATR diário para z-score histórico
    atr_daily = df.set_index("timestamp")["atr_pct"].resample("D").last().dropna()

    results = []
    skip_margin = MAX_HOLD_H + 10

    for i in range(len(df) - skip_margin):
        row = df.iloc[i]
        if not _signal_passes(row):
            continue

        entry_price = float(row["close"])
        entry_time = row["timestamp"]
        atr_pct_now = row["atr_pct"]

        if pd.isna(atr_pct_now):
            continue

        # Histórico 30 dias antes
        entry_date = entry_time.normalize()
        history = atr_daily.loc[
            entry_date - pd.Timedelta(days=31) : entry_date - pd.Timedelta(days=1)
        ]

        regime = classify_volatility(atr_pct_now, history)
        adaptive = compute_adaptive_stops(entry_price, regime)

        # Simulação fixed stops
        fixed = simulate_trade(df, i, entry_price, FIXED_SL, FIXED_TP, FIXED_TRAIL)
        # Simulação adaptive stops
        adap_out = simulate_trade(df, i, entry_price,
                                  adaptive["sl_pct"], adaptive["tp_pct"], FIXED_TRAIL)

        results.append({
            "entry_time": str(entry_time),
            "entry_price": entry_price,
            "atr_pct": round(atr_pct_now, 5),
            "z_score": round(regime["z_score"], 3),
            "vol_label": regime["label"],
            "sl_mult": regime["sl_multiplier"],
            "tp_mult": regime["tp_multiplier"],
            "adaptive_sl_pct": round(adaptive["sl_pct"], 4),
            "adaptive_tp_pct": round(adaptive["tp_pct"], 4),
            # Fixed
            "fixed_exit": fixed["exit_reason"],
            "fixed_return": round(fixed["return_pct"], 5),
            "fixed_hours": fixed["hours_held"],
            # Adaptive
            "adaptive_exit": adap_out["exit_reason"],
            "adaptive_return": round(adap_out["return_pct"], 5),
            "adaptive_hours": adap_out["hours_held"],
            # Diff
            "return_diff": round(adap_out["return_pct"] - fixed["return_pct"], 5),
        })

    if not results:
        logger.error("No signals found! Check data paths.")
        return

    df_res = pd.DataFrame(results)
    df_res.to_csv(TRADES_CSV, index=False)
    logger.info(f"Processed {len(df_res)} signals → {TRADES_CSV}")

    _print_comparison(df_res)
    _generate_plots(df_res)
    _generate_report(df_res)


def _print_comparison(df: pd.DataFrame):
    logger.info("\n" + "=" * 60)
    logger.info("COMPARISON — Fixed vs Adaptive")
    logger.info("=" * 60)

    m_fixed = compute_metrics(df["fixed_return"].values)
    m_adap = compute_metrics(df["adaptive_return"].values)

    print(f"\n{'Metric':<25} {'FIXED':>12} {'ADAPTIVE':>12} {'Diff':>10}")
    print("-" * 65)
    for key in ["n_trades", "win_rate_pct", "avg_return_pct", "sharpe", "max_dd_pct"]:
        f = m_fixed.get(key, 0)
        a = m_adap.get(key, 0)
        diff = a - f
        print(f"{key:<25} {f:>12.3f} {a:>12.3f} {diff:>+10.3f}")

    print(f"\n{'Regime':<20} {'N':>5} {'Fixed avg%':>12} {'Adaptive avg%':>14} {'Diff':>8}")
    print("-" * 65)
    for label in ["MUCH_CALMER", "CALM", "NORMAL", "ABOVE_NORMAL", "VOLATILE", "EXTREME"]:
        sub = df[df["vol_label"] == label]
        if len(sub) == 0:
            continue
        f_avg = sub["fixed_return"].mean() * 100
        a_avg = sub["adaptive_return"].mean() * 100
        print(f"{label:<20} {len(sub):>5} {f_avg:>11.2f}% {a_avg:>13.2f}% {a_avg-f_avg:>+7.2f}pp")

    print(f"\nExit distribution (fixed): {dict(df['fixed_exit'].value_counts())}")
    print(f"Exit distribution (adaptive): {dict(df['adaptive_exit'].value_counts())}")

    print(f"\nVolatility distribution:")
    print(df["vol_label"].value_counts().to_string())


def _generate_plots(df: pd.DataFrame):
    # Plot 1: distributions
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(df["fixed_return"] * 100, bins=30, alpha=0.6, label="Fixed", color="steelblue")
    axes[0].hist(df["adaptive_return"] * 100, bins=30, alpha=0.6, label="Adaptive", color="darkorange")
    axes[0].axvline(0, color="black", lw=0.8)
    axes[0].set_xlabel("Return %")
    axes[0].set_title("Distribuição de Retornos: Fixed vs Adaptive")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    lim = max(df["fixed_return"].abs().max(), df["adaptive_return"].abs().max()) * 105
    axes[1].scatter(df["fixed_return"] * 100, df["adaptive_return"] * 100, alpha=0.4, s=25)
    axes[1].plot([-lim, lim], [-lim, lim], "k--", alpha=0.3, lw=1)
    axes[1].axhline(0, color="black", lw=0.5)
    axes[1].axvline(0, color="black", lw=0.5)
    axes[1].set_xlabel("Fixed return %")
    axes[1].set_ylabel("Adaptive return %")
    axes[1].set_title("Trade-by-Trade: Fixed vs Adaptive")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "returns_distribution.png", dpi=100, bbox_inches="tight")
    plt.close()

    # Plot 2: by regime
    regimes = ["MUCH_CALMER", "CALM", "NORMAL", "ABOVE_NORMAL", "VOLATILE", "EXTREME"]
    f_avgs, a_avgs, labels = [], [], []

    for r in regimes:
        sub = df[df["vol_label"] == r]
        if len(sub) < 3:
            continue
        f_avgs.append(sub["fixed_return"].mean() * 100)
        a_avgs.append(sub["adaptive_return"].mean() * 100)
        labels.append(f"{r}\n(n={len(sub)})")

    if labels:
        fig, ax = plt.subplots(figsize=(12, 5))
        x = np.arange(len(labels))
        w = 0.35
        ax.bar(x - w/2, f_avgs, w, label="Fixed", color="steelblue", alpha=0.8)
        ax.bar(x + w/2, a_avgs, w, label="Adaptive", color="darkorange", alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.axhline(0, color="black", lw=0.8)
        ax.set_ylabel("Avg return %")
        ax.set_title("Performance por Regime de Volatilidade")
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "performance_by_regime.png", dpi=100, bbox_inches="tight")
        plt.close()

    # Plot 3: cumulative returns
    fixed_cum = (1 + df.sort_values("entry_time")["fixed_return"]).cumprod()
    adap_cum = (1 + df.sort_values("entry_time")["adaptive_return"]).cumprod()

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(fixed_cum.values, label="Fixed", color="steelblue")
    ax.plot(adap_cum.values, label="Adaptive", color="darkorange")
    ax.axhline(1.0, color="black", lw=0.8)
    ax.set_xlabel("Trade #")
    ax.set_ylabel("Cumulative return (1=inicial)")
    ax.set_title("Curva de Capital Acumulada: Fixed vs Adaptive")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "equity_curve.png", dpi=100, bbox_inches="tight")
    plt.close()

    logger.info(f"Plots saved to {PLOTS_DIR}")


def _generate_report(df: pd.DataFrame):
    m_fixed = compute_metrics(df["fixed_return"].values)
    m_adap = compute_metrics(df["adaptive_return"].values)

    lines = []
    lines.append("# Adaptive Stops v1 — Estudo de Validação")
    lines.append(f"\n**Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    lines.append(f"**Sinais analisados:** {len(df)}")
    lines.append(f"**Período:** {df['entry_time'].min()[:10]} → {df['entry_time'].max()[:10]}")
    lines.append("")

    lines.append("## Comparação de Estratégias\n")
    lines.append("| Métrica | FIXED (atual) | ADAPTIVE | Diff |")
    lines.append("|---------|---------------|----------|------|")
    for key, label in [
        ("n_trades", "N trades"),
        ("win_rate_pct", "Win Rate %"),
        ("avg_return_pct", "Avg Return %"),
        ("sharpe", "Sharpe"),
        ("max_dd_pct", "Max DD %"),
    ]:
        f = m_fixed.get(key, 0)
        a = m_adap.get(key, 0)
        diff = a - f
        lines.append(f"| {label} | {f:.3f} | {a:.3f} | {diff:+.3f} |")
    lines.append("")

    lines.append("## Performance por Regime\n")
    lines.append("| Regime | N | Fixed avg% | Adaptive avg% | Diff |")
    lines.append("|--------|---|-----------|--------------|------|")
    for label in ["MUCH_CALMER", "CALM", "NORMAL", "ABOVE_NORMAL", "VOLATILE", "EXTREME"]:
        sub = df[df["vol_label"] == label]
        if len(sub) == 0:
            continue
        f_avg = sub["fixed_return"].mean() * 100
        a_avg = sub["adaptive_return"].mean() * 100
        lines.append(f"| {label} | {len(sub)} | {f_avg:+.2f}% | {a_avg:+.2f}% | {a_avg-f_avg:+.2f}pp |")
    lines.append("")

    lines.append("## Distribuição de Volatilidade\n")
    vc = df["vol_label"].value_counts()
    for label, count in vc.items():
        pct = count / len(df) * 100
        lines.append(f"- {label}: {count} ({pct:.1f}%)")
    lines.append("")

    lines.append("## Saídas\n")
    lines.append("### Fixed\n")
    for r, c in df["fixed_exit"].value_counts().items():
        lines.append(f"- {r}: {c} ({c/len(df)*100:.1f}%)")
    lines.append("\n### Adaptive\n")
    for r, c in df["adaptive_exit"].value_counts().items():
        lines.append(f"- {r}: {c} ({c/len(df)*100:.1f}%)")
    lines.append("")

    # Veredicto
    lines.append("## Veredicto\n")
    diff_wr = m_adap.get("win_rate_pct", 0) - m_fixed.get("win_rate_pct", 0)
    diff_avg = m_adap.get("avg_return_pct", 0) - m_fixed.get("avg_return_pct", 0)
    diff_sharpe = m_adap.get("sharpe", 0) - m_fixed.get("sharpe", 0)
    diff_dd = m_adap.get("max_dd_pct", 0) - m_fixed.get("max_dd_pct", 0)

    wins = sum([diff_wr > 0, diff_avg > 0, diff_sharpe > 0, diff_dd > 0])

    if wins >= 3:
        lines.append("### ADAPTIVE VENCEU\n")
        lines.append(f"- WR diff: {diff_wr:+.2f}pp")
        lines.append(f"- Avg return diff: {diff_avg:+.3f}pp")
        lines.append(f"- Sharpe diff: {diff_sharpe:+.3f}")
        lines.append(f"- Max DD diff: {diff_dd:+.2f}pp")
        lines.append("\n**Recomendação:** integrar adaptive stops no paper_trader.py")
    elif wins <= 1:
        lines.append("### FIXED VENCEU\n")
        lines.append(f"- WR diff: {diff_wr:+.2f}pp")
        lines.append(f"- Avg return diff: {diff_avg:+.3f}pp")
        lines.append(f"- Sharpe diff: {diff_sharpe:+.3f}")
        lines.append("\n**Recomendação:** manter config fixa (SL 1.5% / TP 2.0%)")
    else:
        lines.append("### RESULTADO MISTO\n")
        lines.append(f"- WR diff: {diff_wr:+.2f}pp")
        lines.append(f"- Avg return diff: {diff_avg:+.3f}pp")
        lines.append(f"- Sharpe diff: {diff_sharpe:+.3f}")
        lines.append("\n**Recomendação:** análise mais profunda por regime antes de decidir")
    lines.append("")

    lines.append("## Plots\n")
    lines.append("- `prompts/plots/adaptive_stops_v1/returns_distribution.png`")
    lines.append("- `prompts/plots/adaptive_stops_v1/performance_by_regime.png`")
    lines.append("- `prompts/plots/adaptive_stops_v1/equity_curve.png`")

    with open(REPORT_PATH, "w") as f:
        f.write("\n".join(lines))

    logger.info(f"Report: {REPORT_PATH}")


if __name__ == "__main__":
    run_study()
