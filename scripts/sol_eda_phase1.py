"""
SOL EDA Phase 1 — Teste das 4 hipóteses (H1-H4).

Responde:
  H1: Derivativos preveem retornos curtos?
  H2: Volume SOL discrimina mais que BTC/ETH?
  H3: SOL tem beta maior pra ETH ou pra BTC?
  H4: SOL reverte mais após shocks?

Outputs:
  prompts/sol_eda_phase1_report.md
  prompts/tables/sol_eda_*.csv
  prompts/plots/sol_eda/*.png
"""
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("sol_eda")

OUT_DIR = ROOT / "prompts"
PLOTS_DIR = OUT_DIR / "plots" / "sol_eda"
TABLES_DIR = OUT_DIR / "tables"
for d in [OUT_DIR, PLOTS_DIR, TABLES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

REPORT_PATH = OUT_DIR / "sol_eda_phase1_report.md"


# ==================================================================
# ETAPA 1: CARREGAR DADOS
# ==================================================================

def load_ohlcv(asset: str) -> pd.DataFrame:
    path = ROOT / f"data/01_raw/spot/{asset}_1h.parquet"
    df = pd.read_parquet(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)

    df["return_1h"] = df["close"].pct_change()
    df["log_return_1h"] = np.log(df["close"] / df["close"].shift(1))

    df["volume_z"] = (
        (df["volume"] - df["volume"].rolling(168, min_periods=48).mean()) /
        df["volume"].rolling(168, min_periods=48).std()
    )

    logger.info(f"Loaded {asset} OHLCV: {len(df):,} rows  {df['timestamp'].min().date()} → {df['timestamp'].max().date()}")
    return df


def load_derivative(asset: str, deriv_type: str) -> pd.DataFrame:
    if asset.lower() == "btc":
        path = ROOT / f"data/01_raw/futures/{deriv_type}_4h.parquet"
    else:
        path = ROOT / f"data/01_raw/futures/{asset.lower()}_{deriv_type}_4h.parquet"

    if not path.exists():
        logger.warning(f"{path} não existe")
        return pd.DataFrame()

    df = pd.read_parquet(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    logger.info(f"Loaded {asset} {deriv_type}: {len(df):,} rows")
    return df


def compute_forward_returns(df: pd.DataFrame, horizons: list = [1, 4, 24]) -> pd.DataFrame:
    df = df.copy()
    for h in horizons:
        df[f"fwd_return_{h}h"] = df["close"].shift(-h) / df["close"] - 1
    return df


def compute_zscore(series: pd.Series, window: int = 42) -> pd.Series:
    return (
        (series - series.rolling(window, min_periods=10).mean()) /
        series.rolling(window, min_periods=10).std()
    )


# ==================================================================
# H1: DERIVATIVOS PREVEEM RETORNOS?
# ==================================================================

def test_h1_derivatives_predict(sol_ohlcv: pd.DataFrame) -> dict:
    logger.info("=" * 60)
    logger.info("H1: Derivativos preveem retornos curtos?")
    logger.info("=" * 60)

    oi = load_derivative("sol", "oi")
    funding = load_derivative("sol", "funding")
    taker = load_derivative("sol", "taker")

    if oi.empty or funding.empty or taker.empty:
        logger.warning("Derivativos SOL faltando — H1 parcial")
        return {"status": "partial_data"}

    oi["oi_z"] = compute_zscore(oi["open_interest"], 42)
    funding["funding_z"] = compute_zscore(funding["funding_rate"], 42)

    # Taker: colunas buy_volume_usd / sell_volume_usd (schema CoinGlass AI.hab)
    buy_col = next((c for c in ["taker_buy_volume_usd", "buy_volume_usd"] if c in taker.columns), None)
    sell_col = next((c for c in ["taker_sell_volume_usd", "sell_volume_usd"] if c in taker.columns), None)
    ratio_col = next((c for c in ["taker_ratio", "buy_sell_ratio"] if c in taker.columns), None)

    if buy_col and sell_col:
        total = taker[buy_col] + taker[sell_col]
        taker["taker_ratio"] = taker[buy_col] / total.replace(0, np.nan)
        taker["taker_z"] = compute_zscore(taker["taker_ratio"], 42)
    elif ratio_col:
        taker["taker_ratio"] = taker[ratio_col]
        taker["taker_z"] = compute_zscore(taker["taker_ratio"], 42)

    # Merge OHLCV 1h + derivativos 4h (ffill)
    df = sol_ohlcv[["timestamp", "close"]].copy()
    for deriv_df, cols in [
        (oi[["timestamp", "oi_z"]], ["oi_z"]),
        (funding[["timestamp", "funding_z"]], ["funding_z"]),
        (taker[["timestamp"] + [c for c in ["taker_ratio", "taker_z"] if c in taker.columns]],
         [c for c in ["taker_ratio", "taker_z"] if c in taker.columns]),
    ]:
        df = df.merge(deriv_df, on="timestamp", how="left")

    feat_cols = [c for c in ["oi_z", "funding_z", "taker_ratio", "taker_z"] if c in df.columns]
    for col in feat_cols:
        df[col] = df[col].ffill()
        df[f"{col}_prev"] = df[col].shift(1)  # anti look-ahead

    df = compute_forward_returns(df)

    results = []
    prev_feats = [f"{c}_prev" for c in feat_cols]
    for feat in prev_feats:
        if feat not in df.columns:
            continue
        for h in [1, 4, 24]:
            ret_col = f"fwd_return_{h}h"
            valid = df.dropna(subset=[feat, ret_col])
            if len(valid) < 100:
                continue
            corr = valid[[feat, ret_col]].corr().iloc[0, 1]
            _, p_value = stats.pearsonr(valid[feat], valid[ret_col])
            results.append({
                "feature": feat,
                "horizon": f"{h}h",
                "n": len(valid),
                "corr": corr,
                "p_value": p_value,
                "significant": p_value < 0.05,
            })

    shock_results = []
    for feat in ["oi_z_prev", "funding_z_prev", "taker_z_prev"]:
        if feat not in df.columns:
            continue
        for h in [1, 4, 24]:
            ret_col = f"fwd_return_{h}h"
            valid = df.dropna(subset=[feat, ret_col])
            if len(valid) < 100:
                continue
            high = valid[valid[feat] > 2.0][ret_col]
            low = valid[valid[feat] < -2.0][ret_col]
            if len(high) < 5 or len(low) < 5:
                continue
            pooled_std = np.sqrt((high.std()**2 + low.std()**2) / 2)
            cohens_d = (high.mean() - low.mean()) / pooled_std if pooled_std > 0 else 0
            _, p_value = stats.ttest_ind(high, low, equal_var=False)
            shock_results.append({
                "feature": feat,
                "horizon": f"{h}h",
                "n_high_shock": len(high),
                "n_low_shock": len(low),
                "mean_return_high": high.mean() * 100,
                "mean_return_low": low.mean() * 100,
                "cohens_d": cohens_d,
                "p_value": p_value,
            })

    return {
        "status": "ok",
        "correlations": pd.DataFrame(results),
        "shocks": pd.DataFrame(shock_results),
        "df_enriched": df,
    }


# ==================================================================
# H2: SOL REAGE MAIS A VOLUME QUE BTC/ETH?
# ==================================================================

def test_h2_volume_flow(btc: pd.DataFrame, eth: pd.DataFrame, sol: pd.DataFrame) -> dict:
    logger.info("=" * 60)
    logger.info("H2: SOL reage mais a volume que BTC/ETH?")
    logger.info("=" * 60)

    results = []

    for asset_name, df_asset in [("BTC", btc), ("ETH", eth), ("SOL", sol)]:
        df = df_asset.copy()
        df["volume_z_prev"] = df["volume_z"].shift(1)
        df = compute_forward_returns(df)

        for h in [1, 4, 24]:
            valid = df.dropna(subset=["volume_z_prev", f"fwd_return_{h}h"])
            if len(valid) < 100:
                continue
            corr = valid[["volume_z_prev", f"fwd_return_{h}h"]].corr().iloc[0, 1]
            _, p_value = stats.pearsonr(valid["volume_z_prev"], valid[f"fwd_return_{h}h"])
            shock_mask = valid["volume_z_prev"].abs() > 2
            n_shocks = shock_mask.sum()
            shock_return_mean = valid[shock_mask][f"fwd_return_{h}h"].mean() * 100 if n_shocks > 0 else 0
            baseline_mean = valid[~shock_mask][f"fwd_return_{h}h"].mean() * 100
            results.append({
                "asset": asset_name,
                "horizon": f"{h}h",
                "n": len(valid),
                "corr": corr,
                "p_value": p_value,
                "n_shocks": n_shocks,
                "shock_return_mean_pct": shock_return_mean,
                "baseline_return_mean_pct": baseline_mean,
            })

    df_results = pd.DataFrame(results)

    comparison = []
    for h in ["1h", "4h", "24h"]:
        sub = df_results[df_results.horizon == h]
        if len(sub) < 3:
            continue
        sol_corr = sub[sub.asset == "SOL"]["corr"].values[0]
        btc_corr = sub[sub.asset == "BTC"]["corr"].values[0]
        eth_corr = sub[sub.asset == "ETH"]["corr"].values[0]
        avg_others = (abs(btc_corr) + abs(eth_corr)) / 2
        sol_abs = abs(sol_corr)
        comparison.append({
            "horizon": h,
            "sol_abs_corr": sol_abs,
            "btc_abs_corr": abs(btc_corr),
            "eth_abs_corr": abs(eth_corr),
            "avg_btc_eth": avg_others,
            "sol_vs_avg": f"{sol_abs / avg_others:.2f}x" if avg_others > 0 else "N/A",
            "h2_supported": sol_abs > avg_others * 1.2,
        })

    return {"detail": df_results, "comparison": pd.DataFrame(comparison)}


# ==================================================================
# H3: SOL TEM BETA MAIOR PRO ETH OU BTC?
# ==================================================================

def test_h3_beta_decomposition(btc: pd.DataFrame, eth: pd.DataFrame, sol: pd.DataFrame) -> dict:
    logger.info("=" * 60)
    logger.info("H3: SOL tem beta maior pro ETH que pro BTC?")
    logger.info("=" * 60)

    merged = pd.merge(
        btc[["timestamp", "return_1h"]].rename(columns={"return_1h": "btc_ret"}),
        eth[["timestamp", "return_1h"]].rename(columns={"return_1h": "eth_ret"}),
        on="timestamp", how="inner"
    )
    merged = pd.merge(
        merged,
        sol[["timestamp", "return_1h"]].rename(columns={"return_1h": "sol_ret"}),
        on="timestamp", how="inner"
    )
    merged = merged.dropna()
    logger.info(f"Overlap: {len(merged):,} 1h candles  {merged['timestamp'].min().date()} → {merged['timestamp'].max().date()}")

    X = merged[["btc_ret", "eth_ret"]].values
    y = merged["sol_ret"].values

    reg = LinearRegression()
    reg.fit(X, y)
    beta_btc, beta_eth = reg.coef_
    alpha = reg.intercept_
    r2 = reg.score(X, y)

    y_pred = reg.predict(X)
    residuals = y - y_pred
    n, k = len(y), 2
    residual_var = np.sum(residuals**2) / (n - k - 1)
    X_c = X - X.mean(axis=0)
    cov_matrix = residual_var * np.linalg.inv(X_c.T @ X_c)
    se_btc, se_eth = np.sqrt(np.diag(cov_matrix))
    t_btc = beta_btc / se_btc
    t_eth = beta_eth / se_eth

    r2_btc = LinearRegression().fit(merged[["btc_ret"]], y).score(merged[["btc_ret"]], y)
    r2_eth = LinearRegression().fit(merged[["eth_ret"]], y).score(merged[["eth_ret"]], y)

    corr_btc = merged[["sol_ret", "btc_ret"]].corr().iloc[0, 1]
    corr_eth = merged[["sol_ret", "eth_ret"]].corr().iloc[0, 1]

    # Rolling beta 168h, step 24h
    rolling_results = []
    window = 168
    for i in range(window, len(merged), 24):
        sub = merged.iloc[i - window:i]
        if len(sub) < int(window * 0.8):
            continue
        reg_r = LinearRegression().fit(sub[["btc_ret", "eth_ret"]].values, sub["sol_ret"].values)
        rolling_results.append({
            "timestamp": sub["timestamp"].iloc[-1],
            "beta_btc": reg_r.coef_[0],
            "beta_eth": reg_r.coef_[1],
        })

    return {
        "multivariate": {
            "beta_btc": beta_btc, "beta_eth": beta_eth,
            "se_btc": se_btc, "se_eth": se_eth,
            "t_stat_btc": t_btc, "t_stat_eth": t_eth,
            "alpha": alpha, "r2_combined": r2,
        },
        "univariate": {
            "r2_btc_only": r2_btc, "r2_eth_only": r2_eth,
            "corr_btc": corr_btc, "corr_eth": corr_eth,
        },
        "rolling_beta": pd.DataFrame(rolling_results),
        "n_observations": len(merged),
    }


# ==================================================================
# H4: SOL REVERTE MAIS APÓS SHOCKS?
# ==================================================================

def test_h4_reflexivity(btc: pd.DataFrame, eth: pd.DataFrame, sol: pd.DataFrame) -> pd.DataFrame:
    logger.info("=" * 60)
    logger.info("H4: SOL reverte mais após shocks?")
    logger.info("=" * 60)

    results = []

    for asset_name, df_asset in [("BTC", btc), ("ETH", eth), ("SOL", sol)]:
        df = df_asset.copy()
        rolling_std = df["return_1h"].rolling(168, min_periods=48).std()
        df["return_z"] = df["return_1h"] / rolling_std.replace(0, np.nan)
        df = compute_forward_returns(df)

        for threshold in [2.0, 3.0]:
            pos_shock = df[df["return_z"] > threshold]
            neg_shock = df[df["return_z"] < -threshold]

            for h in [1, 4, 24]:
                ret_col = f"fwd_return_{h}h"
                pos_fwd = pos_shock[ret_col].dropna()
                neg_fwd = neg_shock[ret_col].dropna()
                if len(pos_fwd) < 5 or len(neg_fwd) < 5:
                    continue

                results.append({
                    "asset": asset_name,
                    "shock_threshold": f"±{threshold}σ",
                    "horizon": f"{h}h",
                    "n_pos_shocks": len(pos_fwd),
                    "n_neg_shocks": len(neg_fwd),
                    "pos_shock_fwd_return_pct": pos_fwd.mean() * 100,
                    "neg_shock_fwd_return_pct": neg_fwd.mean() * 100,
                    "reversion_strength_pct": (neg_fwd.mean() - pos_fwd.mean()) * 100,
                    "pos_p_value": stats.ttest_1samp(pos_fwd, 0).pvalue,
                    "neg_p_value": stats.ttest_1samp(neg_fwd, 0).pvalue,
                })

    return pd.DataFrame(results)


# ==================================================================
# VISUALIZAÇÕES
# ==================================================================

def generate_plots(h1_result, h2_result, h3_result, h4_result, btc, eth, sol):
    logger.info("Gerando plots...")

    # Plot 1: H2 — Volume vs forward returns 4h (3 assets)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for idx, (name, df) in enumerate([("BTC", btc), ("ETH", eth), ("SOL", sol)]):
        df_plot = df.copy()
        df_plot["volume_z_prev"] = df_plot["volume_z"].shift(1)
        df_plot = compute_forward_returns(df_plot)
        valid = df_plot.dropna(subset=["volume_z_prev", "fwd_return_4h"])
        axes[idx].scatter(valid["volume_z_prev"], valid["fwd_return_4h"] * 100,
                          alpha=0.3, s=10)
        axes[idx].axhline(0, color="black", linewidth=0.3)
        axes[idx].axvline(0, color="black", linewidth=0.3)
        axes[idx].set_xlabel("volume_z (t-1)")
        axes[idx].set_ylabel("Forward return 4h (%)")
        axes[idx].set_title(name)
        axes[idx].set_xlim(-4, 4)
        axes[idx].grid(alpha=0.3)
        corr = valid[["volume_z_prev", "fwd_return_4h"]].corr().iloc[0, 1]
        axes[idx].text(0.05, 0.95, f"Corr: {corr:.3f}\nN: {len(valid):,}",
                       transform=axes[idx].transAxes, verticalalignment="top",
                       bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    plt.suptitle("H2: Volume (t-1) vs Forward Return 4h")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "h2_volume_vs_returns.png", dpi=100, bbox_inches="tight")
    plt.close()

    # Plot 2: H3 — Rolling beta
    rolling_df = h3_result.get("rolling_beta", pd.DataFrame())
    if not rolling_df.empty:
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(rolling_df["timestamp"], rolling_df["beta_btc"], label="β BTC", color="orange")
        ax.plot(rolling_df["timestamp"], rolling_df["beta_eth"], label="β ETH", color="purple")
        ax.axhline(1, color="gray", linestyle="--", alpha=0.5, label="β=1")
        ax.axhline(0, color="black", linewidth=0.3)
        ax.set_title("H3: SOL Rolling Beta (7d window)")
        ax.set_ylabel("Beta")
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "h3_rolling_beta.png", dpi=100, bbox_inches="tight")
        plt.close()

    # Plot 3: H4 — Mean reversion strength
    if isinstance(h4_result, pd.DataFrame) and not h4_result.empty:
        subset = h4_result[h4_result["shock_threshold"] == "±3.0σ"]
        pivot = subset.pivot(index="asset", columns="horizon", values="reversion_strength_pct")
        if not pivot.empty:
            fig, ax = plt.subplots(figsize=(10, 5))
            pivot.plot(kind="bar", ax=ax)
            ax.set_title("H4: Mean Reversion after ±3σ Shock (neg_fwd − pos_fwd)")
            ax.set_ylabel("Reversion strength (pp)")
            ax.axhline(0, color="black", linewidth=0.5)
            ax.legend(title="Forward horizon")
            ax.grid(alpha=0.3)
            ax.set_xlabel("")
            plt.xticks(rotation=0)
            plt.tight_layout()
            plt.savefig(PLOTS_DIR / "h4_reversion_strength.png", dpi=100, bbox_inches="tight")
            plt.close()

    # Plot 4: H1 — correlações derivativos vs horizonte (heatmap)
    if h1_result.get("status") == "ok" and not h1_result["correlations"].empty:
        corr_df = h1_result["correlations"]
        pivot = corr_df.pivot(index="feature", columns="horizon", values="corr")
        if not pivot.empty:
            fig, ax = plt.subplots(figsize=(8, max(4, len(pivot) * 0.8)))
            im = ax.imshow(pivot.values, cmap="RdYlGn", vmin=-0.15, vmax=0.15, aspect="auto")
            ax.set_xticks(range(len(pivot.columns)))
            ax.set_xticklabels(pivot.columns)
            ax.set_yticks(range(len(pivot.index)))
            ax.set_yticklabels(pivot.index, fontsize=9)
            for i in range(len(pivot.index)):
                for j in range(len(pivot.columns)):
                    val = pivot.values[i, j]
                    if not np.isnan(val):
                        ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=8)
            plt.colorbar(im, ax=ax, label="Correlation")
            ax.set_title("H1: SOL Derivatives Correlation (prev → forward return)")
            plt.tight_layout()
            plt.savefig(PLOTS_DIR / "h1_derivatives_corr_heatmap.png", dpi=100, bbox_inches="tight")
            plt.close()

    logger.info(f"Plots salvos em {PLOTS_DIR}")


# ==================================================================
# REPORT
# ==================================================================

def _verdict_h1(h1: dict) -> str:
    if h1.get("status") != "ok" or h1["correlations"].empty:
        return "⚠️ INCONCLUSIVA — dados insuficientes"
    sig = h1["correlations"][h1["correlations"]["significant"]]
    if len(sig) == 0:
        return "❌ REJEITADA — nenhuma correlação significativa"
    best = h1["correlations"].loc[h1["correlations"]["corr"].abs().idxmax()]
    return f"✅ PARCIALMENTE SUPORTADA — {len(sig)}/{len(h1['correlations'])} significativas (melhor: {best['feature']} {best['horizon']} r={best['corr']:+.3f})"


def _verdict_h2(h2: dict) -> str:
    if "comparison" not in h2 or h2["comparison"].empty:
        return "⚠️ INCONCLUSIVA"
    supported = h2["comparison"]["h2_supported"].sum()
    total = len(h2["comparison"])
    if supported == total:
        return "✅ SUPORTADA — SOL ≥1.2× avg(BTC,ETH) em todos horizontes"
    elif supported > 0:
        return f"⚖️ MISTA — suportada em {supported}/{total} horizontes"
    return "❌ REJEITADA — SOL não supera BTC/ETH em volume flow"


def _verdict_h3(h3: dict) -> str:
    if "multivariate" not in h3:
        return "⚠️ INCONCLUSIVA"
    mv = h3["multivariate"]
    if mv["beta_eth"] > mv["beta_btc"] and abs(mv["t_stat_eth"]) > 2:
        return f"✅ SUPORTADA — β ETH ({mv['beta_eth']:+.3f}) > β BTC ({mv['beta_btc']:+.3f}), t_eth={mv['t_stat_eth']:+.2f}"
    elif mv["beta_btc"] > mv["beta_eth"] and abs(mv["t_stat_btc"]) > 2:
        return f"❌ REJEITADA — β BTC ({mv['beta_btc']:+.3f}) > β ETH ({mv['beta_eth']:+.3f}), t_btc={mv['t_stat_btc']:+.2f}"
    return f"⚖️ INCONCLUSIVA — β ETH={mv['beta_eth']:+.3f} β BTC={mv['beta_btc']:+.3f} (sem diferença significativa)"


def _verdict_h4(h4: pd.DataFrame) -> str:
    if not isinstance(h4, pd.DataFrame) or h4.empty:
        return "⚠️ INCONCLUSIVA"
    sub = h4[(h4["shock_threshold"] == "±3.0σ") & (h4["horizon"] == "4h")]
    if sub.empty:
        return "⚠️ Dados insuficientes (shocks 3σ)"
    sol_rev = sub[sub.asset == "SOL"]["reversion_strength_pct"].values
    btc_rev = sub[sub.asset == "BTC"]["reversion_strength_pct"].values
    eth_rev = sub[sub.asset == "ETH"]["reversion_strength_pct"].values
    if len(sol_rev) == 0:
        return "⚠️ Sem dados SOL"
    sol_val = sol_rev[0]
    avg_others = np.mean([v for arr in [btc_rev, eth_rev] for v in arr] or [0])
    if sol_val > avg_others * 1.2 and sol_val > 0:
        return f"✅ SUPORTADA — SOL reversion {sol_val:+.3f}pp > avg outros {avg_others:+.3f}pp (±3σ 4h)"
    elif sol_val > 0:
        return f"⚖️ MISTA — SOL tem reversão ({sol_val:+.3f}pp) mas não significativamente maior que BTC/ETH"
    return f"❌ REJEITADA — SOL reversion {sol_val:+.3f}pp ≤ 0 ou abaixo de outros"


def generate_report(h1, h2, h3, h4):
    logger.info("Gerando report...")
    lines = []
    lines.append("# SOL EDA Phase 1 — Teste das 4 Hipóteses")
    lines.append(f"\n**Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}\n")

    lines.append("## Resumo — Veredictos\n")
    lines.append(f"- **H1 (Derivativos):** {_verdict_h1(h1)}")
    lines.append(f"- **H2 (Volume flow):** {_verdict_h2(h2)}")
    lines.append(f"- **H3 (Beta ETH vs BTC):** {_verdict_h3(h3)}")
    lines.append(f"- **H4 (Reflexividade):** {_verdict_h4(h4)}")
    lines.append("")

    # H1
    lines.append("## H1 — Derivativos preveem retornos curtos?\n")
    if h1.get("status") == "ok" and not h1["correlations"].empty:
        lines.append("### Correlações (feature prev × forward return)\n")
        lines.append("| Feature | Horizon | N | Corr | p-value | Sig |")
        lines.append("|---------|---------|---|------|---------|-----|")
        for _, row in h1["correlations"].sort_values(["feature", "horizon"]).iterrows():
            sig = "✅" if row["significant"] else "❌"
            lines.append(
                f"| {row['feature']} | {row['horizon']} | {row['n']:,} | "
                f"{row['corr']:+.4f} | {row['p_value']:.4f} | {sig} |"
            )
        lines.append("")

        if not h1["shocks"].empty:
            lines.append("### Shocks extremos (feature > 2σ vs < -2σ)\n")
            lines.append("| Feature | Horizon | N+ | N- | Ret high | Ret low | Cohen's d | p |")
            lines.append("|---------|---------|----|----|----------|---------|-----------|---|")
            for _, row in h1["shocks"].iterrows():
                lines.append(
                    f"| {row['feature']} | {row['horizon']} | {row['n_high_shock']} | {row['n_low_shock']} | "
                    f"{row['mean_return_high']:+.3f}% | {row['mean_return_low']:+.3f}% | "
                    f"{row['cohens_d']:+.3f} | {row['p_value']:.4f} |"
                )
            lines.append("")
    else:
        lines.append("⚠️ Dados parciais\n")

    # H2
    lines.append("## H2 — Volume flow: SOL vs BTC/ETH\n")
    if "comparison" in h2 and not h2["comparison"].empty:
        lines.append("| Horizon | SOL |corr| | BTC |corr| | ETH |corr| | SOL vs avg | H2 |")
        lines.append("|---------|-------------|-------------|-------------|------------|-----|")
        for _, row in h2["comparison"].iterrows():
            sup = "✅" if row["h2_supported"] else "❌"
            lines.append(
                f"| {row['horizon']} | {row['sol_abs_corr']:.4f} | "
                f"{row['btc_abs_corr']:.4f} | {row['eth_abs_corr']:.4f} | "
                f"{row['sol_vs_avg']} | {sup} |"
            )
        lines.append("")
        lines.append("### Detalhe por asset\n")
        lines.append("| Asset | Horizon | N | Corr | p-value | N shocks | Shock ret% | Baseline ret% |")
        lines.append("|-------|---------|---|------|---------|----------|------------|--------------|")
        for _, row in h2["detail"].iterrows():
            lines.append(
                f"| {row['asset']} | {row['horizon']} | {row['n']:,} | {row['corr']:+.4f} | "
                f"{row['p_value']:.4f} | {row['n_shocks']} | {row['shock_return_mean_pct']:+.3f}% | "
                f"{row['baseline_return_mean_pct']:+.3f}% |"
            )
        lines.append("")

    # H3
    lines.append("## H3 — Beta decomposition (SOL ~ BTC + ETH)\n")
    if "multivariate" in h3:
        mv = h3["multivariate"]
        univ = h3["univariate"]
        lines.append(f"**Observações:** {h3['n_observations']:,} candles 1h\n")
        lines.append("### Multivariate regression: SOL = α + β_btc·BTC + β_eth·ETH\n")
        lines.append(f"| Coef | Value | SE | t-stat |")
        lines.append(f"|------|-------|----|--------|")
        lines.append(f"| α | {mv['alpha']:+.6f} | — | — |")
        lines.append(f"| β BTC | {mv['beta_btc']:+.4f} | {mv['se_btc']:.4f} | {mv['t_stat_btc']:+.2f} |")
        lines.append(f"| β ETH | {mv['beta_eth']:+.4f} | {mv['se_eth']:.4f} | {mv['t_stat_eth']:+.2f} |")
        lines.append(f"\n**R² combined:** {mv['r2_combined']:.4f}\n")
        lines.append("### Univariate\n")
        lines.append(f"| Model | R² | Corr(SOL) |")
        lines.append(f"|-------|-----|-----------|")
        lines.append(f"| SOL ~ BTC | {univ['r2_btc_only']:.4f} | {univ['corr_btc']:.4f} |")
        lines.append(f"| SOL ~ ETH | {univ['r2_eth_only']:.4f} | {univ['corr_eth']:.4f} |")
        lines.append("")

    # H4
    lines.append("## H4 — Reflexividade / Mean reversion\n")
    if isinstance(h4, pd.DataFrame) and not h4.empty:
        lines.append("| Asset | Shock | Horizon | N+ | N- | Ret+ | Ret- | Reversão | p+ | p- |")
        lines.append("|-------|-------|---------|-----|-----|------|------|----------|----|----|")
        for _, row in h4.sort_values(["shock_threshold", "asset", "horizon"]).iterrows():
            lines.append(
                f"| {row['asset']} | {row['shock_threshold']} | {row['horizon']} | "
                f"{row['n_pos_shocks']} | {row['n_neg_shocks']} | "
                f"{row['pos_shock_fwd_return_pct']:+.3f}% | "
                f"{row['neg_shock_fwd_return_pct']:+.3f}% | "
                f"{row['reversion_strength_pct']:+.3f}pp | "
                f"{row['pos_p_value']:.4f} | {row['neg_p_value']:.4f} |"
            )
        lines.append("")

    lines.append("## Implicações estratégicas\n")
    lines.append("### Decisão: SOL Bot strategy\n")
    verdicts = {
        "H1": _verdict_h1(h1),
        "H2": _verdict_h2(h2),
        "H3": _verdict_h3(h3),
        "H4": _verdict_h4(h4),
    }
    n_confirmed = sum(1 for v in verdicts.values() if v.startswith("✅"))
    if n_confirmed >= 3:
        lines.append("**Multi-feature SOL Bot viável:** Volume filter + ETH context + Mean reversion + Derivative gates.")
    elif n_confirmed == 2:
        lines.append("**Strategy parcial:** usar apenas features das hipóteses confirmadas. Aguardar mais dados para features inconclusivas.")
    elif n_confirmed == 1 and "H3" in [k for k, v in verdicts.items() if v.startswith("✅")]:
        lines.append("**Pair trade SOL-ETH:** SOL Bot baseado em divergência SOL/ETH como estratégia principal.")
    else:
        lines.append("**Edge fraco:** manter BTC + ETH como ativos principais. SOL como análise periódica.")
    lines.append("")
    lines.append("## Arquivos gerados\n")
    lines.append(f"- Plots: `prompts/plots/sol_eda/`")
    lines.append(f"- Tables: `prompts/tables/sol_eda_*.csv`")

    with open(REPORT_PATH, "w") as f:
        f.write("\n".join(lines))
    logger.info(f"Report: {REPORT_PATH}")


# ==================================================================
# MAIN
# ==================================================================

def main():
    logger.info("=" * 60)
    logger.info("SOL EDA Phase 1 — Testando H1, H2, H3, H4")
    logger.info("=" * 60)

    btc = load_ohlcv("btc")
    eth = load_ohlcv("eth")
    sol = load_ohlcv("sol")

    h1 = test_h1_derivatives_predict(sol)
    h2 = test_h2_volume_flow(btc, eth, sol)
    h3 = test_h3_beta_decomposition(btc, eth, sol)
    h4 = test_h4_reflexivity(btc, eth, sol)

    # Save tables
    for name, obj in [
        ("sol_eda_h1_correlations", h1.get("correlations")),
        ("sol_eda_h1_shocks", h1.get("shocks")),
        ("sol_eda_h2_detail", h2.get("detail")),
        ("sol_eda_h2_comparison", h2.get("comparison")),
        ("sol_eda_h3_rolling_beta", h3.get("rolling_beta")),
        ("sol_eda_h4_reversion", h4 if isinstance(h4, pd.DataFrame) else None),
    ]:
        if obj is not None and isinstance(obj, pd.DataFrame) and not obj.empty:
            obj.to_csv(TABLES_DIR / f"{name}.csv", index=False)
            logger.info(f"Saved {TABLES_DIR / name}.csv")

    generate_plots(h1, h2, h3, h4, btc, eth, sol)
    generate_report(h1, h2, h3, h4)

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print("\n### Veredictos:")
    print(f"  H1 (Derivativos): {_verdict_h1(h1)}")
    print(f"  H2 (Volume):      {_verdict_h2(h2)}")
    print(f"  H3 (Beta ETH/BTC):{_verdict_h3(h3)}")
    print(f"  H4 (Reflexiv.):   {_verdict_h4(h4)}")

    print("\n### H2 comparison:")
    if "comparison" in h2 and not h2["comparison"].empty:
        print(h2["comparison"].to_string(index=False))

    print("\n### H3 multivariate betas:")
    if "multivariate" in h3:
        mv = h3["multivariate"]
        print(f"  β BTC: {mv['beta_btc']:+.4f}  (t={mv['t_stat_btc']:+.2f})")
        print(f"  β ETH: {mv['beta_eth']:+.4f}  (t={mv['t_stat_eth']:+.2f})")
        print(f"  R²:    {mv['r2_combined']:.4f}")

    print("\n### H4 reversion (3σ, 4h):")
    if isinstance(h4, pd.DataFrame) and not h4.empty:
        sub = h4[(h4["shock_threshold"] == "±3.0σ") & (h4["horizon"] == "4h")]
        if not sub.empty:
            print(sub[["asset", "n_pos_shocks", "pos_shock_fwd_return_pct",
                        "neg_shock_fwd_return_pct", "reversion_strength_pct"]].to_string(index=False))

    logger.info(f"\n✅ Complete. Report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
