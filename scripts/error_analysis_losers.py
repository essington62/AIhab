"""
Error Analysis — 43 Losers do Bot 2.

Reconstrói 136 sinais históricos via _signal_passes (btc_1h_clean.parquet),
simula trades com stops fixos (SL 1.5% / TP 2%), extrai ~20 features por trade,
compara distribuições losers vs winners (Cohen's d + Mann-Whitney),
e clusteriza os 43 losers (K-means, k=2-4, silhouette).

Outputs:
  prompts/error_analysis_report.md
  prompts/tables/error_analysis_full_dataset.csv
  prompts/tables/error_analysis_loser_profiles.csv
  prompts/plots/error_analysis/
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
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("error_analysis")

PLOTS_DIR = ROOT / "prompts/plots/error_analysis"
TABLES_DIR = ROOT / "prompts/tables"
REPORT_PATH = ROOT / "prompts/error_analysis_report.md"

for d in [PLOTS_DIR, TABLES_DIR, REPORT_PATH.parent]:
    d.mkdir(parents=True, exist_ok=True)

FIXED_SL = 0.015
FIXED_TP = 0.020
FIXED_TRAIL = 0.010
MAX_HOLD_H = 120


# ==========================================================
# DATA LOADING
# ==========================================================

def load_data() -> pd.DataFrame:
    df = pd.read_parquet(ROOT / "data/02_intermediate/spot/btc_1h_clean.parquet")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)

    df["ret_1d"] = df["close"].pct_change(24)
    df["ret_3h"] = df["close"].pct_change(3)
    df["ret_1h"] = df["close"].pct_change(1)

    # ATR 14h
    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    df["atr_14"] = tr.rolling(14).mean()
    df["atr_pct"] = df["atr_14"] / close

    # Volume z-score (30h rolling)
    if "volume" in df.columns:
        vol_mean = df["volume"].rolling(30).mean()
        vol_std = df["volume"].rolling(30).std()
        df["volume_z"] = (df["volume"] - vol_mean) / vol_std.replace(0, np.nan)
    else:
        df["volume_z"] = np.nan

    # MA200 se disponível
    if "ma_200" not in df.columns and "close" in df.columns:
        df["ma_200"] = df["close"].rolling(200).mean()

    # Gate z-scores
    zs_path = ROOT / "data/02_features/gate_zscores.parquet"
    if zs_path.exists():
        zs = pd.read_parquet(zs_path)
        zs["timestamp"] = pd.to_datetime(zs["timestamp"], utc=True)
        merge_cols = ["timestamp"]
        for col in ["stablecoin_z", "oi_z", "funding_z", "taker_z", "bb_index"]:
            if col in zs.columns:
                merge_cols.append(col)
        df = df.merge(zs[merge_cols], on="timestamp", how="left")
        for col in merge_cols[1:]:
            df[col] = df[col].ffill()
    else:
        df["stablecoin_z"] = np.nan

    df = df[df["timestamp"] >= pd.Timestamp("2026-01-01", tz="UTC")].reset_index(drop=True)
    logger.info(f"Loaded {len(df)} rows: {df['timestamp'].min().date()} → {df['timestamp'].max().date()}")
    return df


# ==========================================================
# SIGNAL IDENTIFICATION (idêntico ao mfe_mae e adaptive_stops)
# ==========================================================

def _signal_passes(row) -> bool:
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
# TRADE SIMULATION
# ==========================================================

def simulate_trade(df: pd.DataFrame, entry_idx: int,
                   entry_price: float, sl_pct: float, tp_pct: float,
                   trail_pct: float = FIXED_TRAIL,
                   max_hold: int = MAX_HOLD_H) -> dict:
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

        if high > trailing_high:
            trailing_high = high

        if high >= tp_price:
            return {"exit_reason": "TP", "exit_price": tp_price,
                    "return_pct": tp_pct, "hours_held": hours}

        if low <= sl_price:
            return {"exit_reason": "SL", "exit_price": sl_price,
                    "return_pct": -sl_pct, "hours_held": hours}

        if close > entry_price:
            trailing_stop = trailing_high * (1 - trail_pct)
            if close <= trailing_stop and trailing_stop > entry_price:
                ret = (trailing_stop - entry_price) / entry_price
                return {"exit_reason": "TRAIL", "exit_price": trailing_stop,
                        "return_pct": ret, "hours_held": hours}

    last = df.iloc[end_idx]
    ret = (last["close"] - entry_price) / entry_price
    return {"exit_reason": "TIMEOUT", "exit_price": last["close"],
            "return_pct": ret, "hours_held": end_idx - entry_idx}


# ==========================================================
# FEATURE EXTRACTION
# ==========================================================

def extract_features(df: pd.DataFrame, idx: int) -> dict:
    row = df.iloc[idx]
    close = float(row["close"])
    high = float(row.get("high", close))
    low = float(row.get("low", close))
    open_ = float(row.get("open", close))
    candle_range = high - low + 1e-10

    # Candle microstructure
    body = (close - open_) / candle_range
    upper_wick = (high - max(open_, close)) / candle_range
    lower_wick = (min(open_, close) - low) / candle_range

    # MA context
    ma21 = float(row.get("ma_21", np.nan))
    ma200 = float(row.get("ma_200", np.nan))
    close_vs_ma21 = (close - ma21) / ma21 if not np.isnan(ma21) and ma21 > 0 else np.nan
    close_vs_ma200 = (close - ma200) / ma200 if not np.isnan(ma200) and ma200 > 0 else np.nan

    # Linear trend slope over last 3 closes
    if idx >= 3:
        last3 = df.iloc[idx - 2: idx + 1]["close"].values
        if len(last3) == 3 and not np.any(np.isnan(last3)):
            slope = np.polyfit([0, 1, 2], last3, 1)[0] / close
        else:
            slope = np.nan
    else:
        slope = np.nan

    ts = row["timestamp"]

    return {
        "rsi_14": float(row.get("rsi_14", np.nan)),
        "bb_pct": float(row.get("bb_pct", np.nan)),
        "ret_1d": float(row.get("ret_1d", np.nan)),
        "ret_3h": float(row.get("ret_3h", np.nan)),
        "ret_1h": float(row.get("ret_1h", np.nan)),
        "stablecoin_z": float(row.get("stablecoin_z", np.nan)),
        "atr_pct": float(row.get("atr_pct", np.nan)),
        "volume_z": float(row.get("volume_z", np.nan)),
        "oi_z": float(row.get("oi_z", np.nan)),
        "funding_z": float(row.get("funding_z", np.nan)),
        "taker_z": float(row.get("taker_z", np.nan)),
        "hour_of_day": int(ts.hour),
        "day_of_week": int(ts.dayofweek),
        "is_weekend": int(ts.dayofweek >= 5),
        "candle_body": body,
        "upper_wick": upper_wick,
        "lower_wick": lower_wick,
        "close_vs_ma21": close_vs_ma21,
        "close_vs_ma200": close_vs_ma200,
        "trend_slope_3h": slope,
    }


# ==========================================================
# STATISTICAL COMPARISON
# ==========================================================

def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return np.nan
    n1, n2 = len(a), len(b)
    s_pool = np.sqrt(((n1 - 1) * a.std(ddof=1) ** 2 + (n2 - 1) * b.std(ddof=1) ** 2) / (n1 + n2 - 2))
    return (a.mean() - b.mean()) / s_pool if s_pool > 0 else 0.0


def compare_groups(df_w: pd.DataFrame, df_l: pd.DataFrame, features: list) -> pd.DataFrame:
    rows = []
    for feat in features:
        w = df_w[feat].dropna().values
        l = df_l[feat].dropna().values
        if len(w) < 3 or len(l) < 3:
            continue
        stat, pval = stats.mannwhitneyu(w, l, alternative="two-sided")
        d = cohens_d(w, l)
        rows.append({
            "feature": feat,
            "winner_mean": w.mean(),
            "loser_mean": l.mean(),
            "diff": w.mean() - l.mean(),
            "cohens_d": d,
            "abs_cohens_d": abs(d),
            "mw_pval": pval,
            "significant": pval < 0.05,
        })
    df_stats = pd.DataFrame(rows).sort_values("abs_cohens_d", ascending=False)
    return df_stats


# ==========================================================
# K-MEANS CLUSTERING
# ==========================================================

def cluster_losers(df_losers: pd.DataFrame, feature_cols: list) -> tuple[pd.DataFrame, dict]:
    """Returns df_losers with cluster column + cluster profiles dict."""
    available = [c for c in feature_cols if c in df_losers.columns and df_losers[c].notna().sum() > len(df_losers) * 0.5]
    if len(available) < 3:
        logger.warning("Insufficient features for clustering")
        df_losers["cluster"] = 0
        return df_losers, {}

    X = df_losers[available].copy()
    X = X.fillna(X.median())

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    best_k, best_score, best_labels = 2, -1.0, None

    for k in range(2, min(5, len(df_losers))):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        if len(np.unique(labels)) < 2:
            continue
        score = silhouette_score(X_scaled, labels)
        logger.info(f"K={k} silhouette={score:.3f}")
        if score > best_score:
            best_k, best_score, best_labels = k, score, labels

    df_losers = df_losers.copy()
    df_losers["cluster"] = best_labels if best_labels is not None else 0

    profiles = {}
    for c in range(best_k):
        sub = df_losers[df_losers["cluster"] == c]
        profiles[c] = {
            "n": len(sub),
            "pct": len(sub) / len(df_losers) * 100,
            "means": sub[available].mean().to_dict(),
        }

    meta = {
        "best_k": best_k,
        "silhouette": best_score,
        "features_used": available,
        "profiles": profiles,
    }
    return df_losers, meta


# ==========================================================
# MINI-BACKTEST FOR HYPOTHESIS FILTERS
# ==========================================================

def test_filter_hypothesis(df_full: pd.DataFrame, feature: str,
                           threshold: float, direction: str) -> dict:
    """
    Simula bloqueio de sinais onde feature < threshold (ou > threshold).
    Compara WR e total_return vs baseline.
    """
    if direction == "block_above":
        mask_blocked = df_full[feature] >= threshold
    else:
        mask_blocked = df_full[feature] <= threshold

    df_pass = df_full[~mask_blocked]
    df_base = df_full

    def wr(d):
        return (d["return_pct"] > 0).mean() * 100 if len(d) > 0 else 0

    def pf(d):
        gains = d.loc[d["return_pct"] > 0, "return_pct"].sum()
        losses = d.loc[d["return_pct"] < 0, "return_pct"].abs().sum()
        return gains / losses if losses > 0 else np.nan

    base_wr = wr(df_base)
    filtered_wr = wr(df_pass)
    n_blocked = mask_blocked.sum()
    n_blocked_winners = df_base[mask_blocked & (df_base["return_pct"] > 0)].shape[0]

    return {
        "feature": feature,
        "threshold": threshold,
        "direction": direction,
        "n_total": len(df_base),
        "n_blocked": int(n_blocked),
        "n_blocked_winners": int(n_blocked_winners),
        "n_pass": len(df_pass),
        "base_wr": base_wr,
        "filtered_wr": filtered_wr,
        "wr_gain": filtered_wr - base_wr,
        "base_pf": pf(df_base),
        "filtered_pf": pf(df_pass),
    }


# ==========================================================
# REPORT GENERATION
# ==========================================================

def _generate_report(df_full: pd.DataFrame, stats_df: pd.DataFrame,
                     cluster_meta: dict, hypotheses: list[dict]):
    n_total = len(df_full)
    n_losers = (df_full["return_pct"] < 0).sum()
    n_winners = (df_full["return_pct"] > 0).sum()
    wr = n_winners / n_total * 100

    lines = [
        "# Error Analysis — Losers do Bot 2",
        f"\n**Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        f"**Sinais analisados:** {n_total}",
        f"**Winners:** {n_winners} ({n_winners/n_total*100:.1f}%)",
        f"**Losers:** {n_losers} ({n_losers/n_total*100:.1f}%)",
        f"**Período:** {df_full['entry_time'].min()[:10]} → {df_full['entry_time'].max()[:10]}",
        "",
        "## Top Features — Cohen's d (Losers vs Winners)",
        "",
        "| Feature | Winner μ | Loser μ | Diff | Cohen's d | p-value | Sig |",
        "|---------|---------|--------|------|-----------|---------|-----|",
    ]

    top_n = min(15, len(stats_df))
    for _, row in stats_df.head(top_n).iterrows():
        sig = "✅" if row["significant"] else ""
        lines.append(
            f"| {row['feature']} | {row['winner_mean']:.3f} | {row['loser_mean']:.3f} "
            f"| {row['diff']:+.3f} | {row['cohens_d']:+.3f} | {row['mw_pval']:.3f} | {sig} |"
        )

    lines += ["", "## Clustering dos Losers", ""]
    if cluster_meta:
        lines.append(f"**K ótimo:** {cluster_meta['best_k']} (silhouette={cluster_meta['silhouette']:.3f})")
        lines.append(f"**Features usadas:** {', '.join(cluster_meta['features_used'])}")
        lines.append("")

        for c, profile in cluster_meta.get("profiles", {}).items():
            lines.append(f"### Cluster {c} — n={profile['n']} ({profile['pct']:.0f}%)")
            top_means = sorted(profile["means"].items(), key=lambda x: abs(x[1]), reverse=True)[:8]
            lines.append("")
            lines.append("| Feature | Média |")
            lines.append("|---------|-------|")
            for feat, mean in top_means:
                lines.append(f"| {feat} | {mean:.3f} |")
            lines.append("")

    lines += ["## Hipóteses de Filtro", ""]
    lines += [
        "| Feature | Threshold | Direction | N bloqueados | Winners perdidos | WR antes | WR depois | Ganho |",
        "|---------|-----------|-----------|-------------|-----------------|----------|-----------|-------|",
    ]
    for h in sorted(hypotheses, key=lambda x: x["wr_gain"], reverse=True):
        if h["n_pass"] < 10:
            continue
        lines.append(
            f"| {h['feature']} | {h['threshold']:.2f} | {h['direction']} "
            f"| {h['n_blocked']} | {h['n_blocked_winners']} "
            f"| {h['base_wr']:.1f}% | {h['filtered_wr']:.1f}% | {h['wr_gain']:+.1f}pp |"
        )

    lines += [
        "",
        "## Conclusão",
        "",
    ]

    # Auto-conclude
    sig_features = stats_df[stats_df["significant"] & (stats_df["abs_cohens_d"] > 0.3)]
    if len(sig_features) > 0:
        top_feat = sig_features.iloc[0]
        lines.append(
            f"**Padrão encontrado:** `{top_feat['feature']}` é o discriminador mais forte "
            f"(Cohen's d={top_feat['cohens_d']:+.2f}, p={top_feat['mw_pval']:.3f})."
        )
        best_h = max(hypotheses, key=lambda x: x["wr_gain"]) if hypotheses else None
        if best_h and best_h["wr_gain"] > 1.0 and best_h["n_blocked_winners"] < best_h["n_blocked"] * 0.5:
            lines.append(
                f"Filtro mais promissor: block `{best_h['feature']}` {best_h['direction'].replace('_', ' ')} "
                f"{best_h['threshold']:.2f} → +{best_h['wr_gain']:.1f}pp WR, "
                f"bloqueando {best_h['n_blocked']} sinais ({best_h['n_blocked_winners']} winners perdidos)."
            )
        else:
            lines.append("Filtro com benefício marginal — considerar custo/benefício antes de implementar.")
    else:
        lines.append(
            "**Sem padrão claro:** Nenhuma feature discrimina significativamente losers vs winners. "
            "Os 33% losers são estatística normal do sistema — não há filtro óbvio a adicionar."
        )

    lines += [
        "",
        "## Plots",
        "",
        "- `prompts/plots/error_analysis/feature_comparison.png`",
        "- `prompts/plots/error_analysis/clusters.png`",
        "- `prompts/plots/error_analysis/temporal_patterns.png`",
        "- `prompts/plots/error_analysis/hypothesis_impact.png`",
    ]

    with open(REPORT_PATH, "w") as f:
        f.write("\n".join(lines))

    logger.info(f"Report: {REPORT_PATH}")


# ==========================================================
# PLOTS
# ==========================================================

def _generate_plots(df_full: pd.DataFrame, stats_df: pd.DataFrame,
                    cluster_meta: dict):
    winners = df_full[df_full["return_pct"] > 0]
    losers = df_full[df_full["return_pct"] < 0]

    # Plot 1: Top features comparison (violin / box)
    top_features = stats_df.head(8)["feature"].tolist()
    available_top = [f for f in top_features if f in df_full.columns and df_full[f].notna().sum() > 5]

    if available_top:
        n = len(available_top)
        ncols = min(4, n)
        nrows = (n + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
        axes = np.array(axes).flatten() if n > 1 else [axes]

        for ax, feat in zip(axes, available_top):
            w_vals = winners[feat].dropna().values
            l_vals = losers[feat].dropna().values
            data = [w_vals, l_vals]
            parts = ax.violinplot(data, positions=[1, 2], showmedians=True)
            for pc, color in zip(parts["bodies"], ["steelblue", "tomato"]):
                pc.set_facecolor(color)
                pc.set_alpha(0.7)
            ax.set_xticks([1, 2])
            ax.set_xticklabels(["Winners", "Losers"])
            d = stats_df[stats_df["feature"] == feat]["cohens_d"].values
            pval = stats_df[stats_df["feature"] == feat]["mw_pval"].values
            title = f"{feat}\nd={d[0]:+.2f}, p={pval[0]:.3f}" if len(d) else feat
            ax.set_title(title, fontsize=9)
            ax.grid(alpha=0.3)

        for ax in axes[len(available_top):]:
            ax.set_visible(False)

        plt.suptitle("Feature Distributions: Winners vs Losers", fontsize=12, y=1.01)
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "feature_comparison.png", dpi=100, bbox_inches="tight")
        plt.close()

    # Plot 2: Cluster scatter (2 top features)
    if cluster_meta and "cluster" in losers.columns:
        feat_used = cluster_meta.get("features_used", [])
        if len(feat_used) >= 2:
            f1, f2 = feat_used[0], feat_used[1]
            fig, ax = plt.subplots(figsize=(8, 6))
            k = cluster_meta["best_k"]
            colors = plt.cm.tab10(np.linspace(0, 0.6, k))
            for c in range(k):
                sub = losers[losers["cluster"] == c]
                ax.scatter(sub[f1], sub[f2], label=f"Cluster {c} (n={len(sub)})",
                           color=colors[c], alpha=0.7, s=60)
            ax.set_xlabel(f1)
            ax.set_ylabel(f2)
            ax.set_title(f"Loser Clusters (k={k}, sil={cluster_meta['silhouette']:.3f})")
            ax.legend()
            ax.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(PLOTS_DIR / "clusters.png", dpi=100, bbox_inches="tight")
            plt.close()

    # Plot 3: Temporal patterns
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # By hour of day
    for label, subset, color in [("Winners", winners, "steelblue"), ("Losers", losers, "tomato")]:
        if "hour_of_day" in subset.columns:
            hour_wr = subset.groupby("hour_of_day").size()
            axes[0].bar(hour_wr.index + (0 if color == "steelblue" else 0.4),
                        hour_wr.values, width=0.4, alpha=0.7, label=label, color=color)

    axes[0].set_xlabel("Hour of Day (UTC)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Sinais por Hora (UTC)")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # By day of week
    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    for label, subset, color in [("Winners", winners, "steelblue"), ("Losers", losers, "tomato")]:
        if "day_of_week" in subset.columns:
            dow = subset.groupby("day_of_week").size().reindex(range(7), fill_value=0)
            axes[1].bar(dow.index + (0 if color == "steelblue" else 0.4),
                        dow.values, width=0.4, alpha=0.7, label=label, color=color)

    axes[1].set_xticks(range(7))
    axes[1].set_xticklabels(days)
    axes[1].set_ylabel("Count")
    axes[1].set_title("Sinais por Dia da Semana")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "temporal_patterns.png", dpi=100, bbox_inches="tight")
    plt.close()

    # Plot 4: Hypothesis impact
    logger.info("Plots saved to " + str(PLOTS_DIR))


# ==========================================================
# MAIN
# ==========================================================

def run_analysis():
    logger.info("=" * 60)
    logger.info("Error Analysis — Losers do Bot 2")
    logger.info("=" * 60)

    df = load_data()

    all_records = []
    skip_margin = MAX_HOLD_H + 10

    for i in range(len(df) - skip_margin):
        row = df.iloc[i]
        if not _signal_passes(row):
            continue

        entry_price = float(row["close"])
        trade = simulate_trade(df, i, entry_price, FIXED_SL, FIXED_TP, FIXED_TRAIL)
        features = extract_features(df, i)

        record = {
            "entry_time": str(row["timestamp"]),
            "entry_price": entry_price,
            **features,
            "exit_reason": trade["exit_reason"],
            "return_pct": trade["return_pct"],
            "hours_held": trade["hours_held"],
            "is_winner": trade["return_pct"] > 0,
        }
        all_records.append(record)

    if not all_records:
        logger.error("No signals found.")
        return

    df_full = pd.DataFrame(all_records)
    logger.info(f"Total signals: {len(df_full)}")
    logger.info(f"Winners: {df_full['is_winner'].sum()} | Losers: {(~df_full['is_winner']).sum()}")

    df_winners = df_full[df_full["is_winner"]]
    df_losers = df_full[~df_full["is_winner"]]

    # Save full dataset
    df_full.to_csv(TABLES_DIR / "error_analysis_full_dataset.csv", index=False)

    # Feature list for statistical comparison
    feature_cols = [
        "rsi_14", "bb_pct", "ret_1d", "ret_3h", "ret_1h",
        "stablecoin_z", "atr_pct", "volume_z", "oi_z", "funding_z", "taker_z",
        "hour_of_day", "day_of_week", "is_weekend",
        "candle_body", "upper_wick", "lower_wick",
        "close_vs_ma21", "close_vs_ma200", "trend_slope_3h",
    ]
    available_feats = [c for c in feature_cols if c in df_full.columns]

    # Statistical comparison
    stats_df = compare_groups(df_winners, df_losers, available_feats)
    logger.info(f"\nTop discriminators:")
    for _, r in stats_df.head(5).iterrows():
        logger.info(f"  {r['feature']}: d={r['cohens_d']:+.2f}, p={r['mw_pval']:.3f}")

    # Clustering
    df_losers, cluster_meta = cluster_losers(df_losers, available_feats)

    # Save loser profiles
    df_losers.to_csv(TABLES_DIR / "error_analysis_loser_profiles.csv", index=False)

    # Hypothesis tests — top 5 significant features
    hypotheses = []
    sig_features = stats_df[stats_df["significant"]].head(5)

    for _, row in sig_features.iterrows():
        feat = row["feature"]
        if feat not in df_full.columns or df_full[feat].isna().all():
            continue

        # Direction: if losers have higher mean → block_above; else block_below
        loser_mean = df_losers[feat].mean()
        winner_mean = df_winners[feat].mean()
        threshold = df_full[feat].quantile(0.75) if loser_mean > winner_mean else df_full[feat].quantile(0.25)
        direction = "block_above" if loser_mean > winner_mean else "block_below"

        h = test_filter_hypothesis(df_full, feat, threshold, direction)
        hypotheses.append(h)

    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Total signals: {len(df_full)} | WR: {df_full['is_winner'].mean()*100:.1f}%")
    print(f"Winners: {df_winners.shape[0]} | Losers: {df_losers.shape[0]}")
    print(f"\nTop 5 discriminating features:")
    print(stats_df.head(5)[["feature", "cohens_d", "mw_pval", "significant"]].to_string(index=False))

    if cluster_meta:
        print(f"\nLoser clusters: k={cluster_meta['best_k']}, silhouette={cluster_meta['silhouette']:.3f}")
        for c, p in cluster_meta.get("profiles", {}).items():
            print(f"  Cluster {c}: n={p['n']} ({p['pct']:.0f}%)")

    if hypotheses:
        best = max(hypotheses, key=lambda x: x["wr_gain"])
        print(f"\nBest filter hypothesis: {best['feature']} {best['direction']} {best['threshold']:.2f}")
        print(f"  WR: {best['base_wr']:.1f}% → {best['filtered_wr']:.1f}% (+{best['wr_gain']:.1f}pp)")
        print(f"  Blocked: {best['n_blocked']} signals ({best['n_blocked_winners']} winners lost)")

    # Propagate cluster labels back into df_full for plotting
    if cluster_meta and "cluster" in df_losers.columns:
        df_full = df_full.copy()
        df_full["cluster"] = np.nan
        df_full.loc[df_losers.index, "cluster"] = df_losers["cluster"].values

    _generate_plots(df_full, stats_df, cluster_meta if cluster_meta else {})
    _generate_report(df_full, stats_df, cluster_meta if cluster_meta else {}, hypotheses)

    logger.info("Done.")


if __name__ == "__main__":
    run_analysis()
