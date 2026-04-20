# Prompt: Fase 0 ETH — Estudo Estatístico com Hipóteses Testáveis

## Contexto

O AI.hab BTC trading system está em paper trading na AWS (EC2 São Paulo) usando Gate Scoring v2 com 11 gates, R5C HMM, MA200 Override, Adaptive Weights e Capital Manager.

**Objetivo:** Avaliar se o framework desenvolvido para BTC se aplica ao ETH, antes de qualquer implementação.

**Filosofia:** Método científico — hipóteses testáveis com critérios de decisão explícitos antes de rodar análise. Aprendizado da sessão anterior: **intuição perde para estatística**.

## Dados já coletados (Fase -1 concluída)

```
✅ ETH spot 1h    → data/01_raw/spot/eth_1h.parquet          (8.781 rows, 1 ano)
✅ ETH OI 4h      → data/01_raw/futures/eth_oi_4h.parquet    (1.080 rows, 180d)
✅ ETH Funding 4h → data/01_raw/futures/eth_funding_4h.parquet (1.080 rows, 180d)
✅ ETH Taker 4h   → data/01_raw/futures/eth_taker_4h.parquet (1.080 rows, 180d)
❌ ETH L/S         → não temos (só 30d histórico, não é gate oficial)
```

**Janela de análise:** 180 dias (overlap completo OI/funding/taker + spot 1h).

## Hipóteses a testar

### H1 — Correlações fundamentais se aplicam ao ETH?

Hipótese nula: correlações dos gates BTC vs forward returns NÃO se aplicam ao ETH.

Hipótese alternativa: ETH tem estrutura de correlações similar ao BTC.

**Critério de decisão:**
- Se |corr_eth - corr_btc| < 0.15 para maioria dos gates → similar
- Se |corr_eth - corr_btc| > 0.25 para maioria → diferente
- Entre 0.15-0.25 → adaptação necessária

### H2 — Os mesmos gates têm poder preditivo?

**Correlações BTC 2026 de referência:**
```
G4 OI coin margin:    -0.472  (mais forte)
G5 Stablecoin mcap:   +0.326
G7 ETF flows:         +0.263
G6 Bubble:            -0.345
G3 DGS10:             -0.315
G9 Taker:             +0.06   (ruído)
G10 Funding:          +0.023
```

**Critério de decisão:**
- Gate ETH com |corr| > 0.20 → mantém (STRONG)
- Gate ETH com |corr| 0.10-0.20 → usa com peso reduzido (MODERATE)
- Gate ETH com |corr| < 0.10 → descarta (WEAK)

### H3 — R5C HMM (treinado em BTC) funciona em ETH?

Hipótese: os estados Bull/Sideways/Bear detectados em BTC também fazem sentido em ETH.

**Critério de decisão:**
- Forward return médio em Bull ETH > Sideways ETH > Bear ETH → SIM
- Se regimes não separam forward returns → retreinar HMM para ETH

### H4 — Model Alignment ETH está em qual faixa?

**Decisão final baseada em alignment:**
- **Alignment > 0.7:** copia config BTC com ajustes mínimos
- **Alignment 0.4-0.7:** adaptive layer compensa, ajustes localizados
- **Alignment < 0.4:** recalibração profunda ou descartar ETH

## Estrutura do script

### Criar `scripts/eth_phase0_statistical_study.py`

```python
"""
Fase 0 ETH — Estudo Estatístico com Hipóteses Testáveis.

Responde:
  H1: Correlações ETH são similares a BTC?
  H2: Quais gates têm poder preditivo em ETH?
  H3: R5C HMM funciona em ETH?
  H4: Qual o model alignment ETH?

Critérios de decisão explícitos no final → copia/adapta/recalibra.

Outputs:
  prompts/eth_phase0_report.md       — relatório principal
  prompts/plots/fase0/                — visualizações
  prompts/tables/                     — CSVs de suporte

Usage:
  python scripts/eth_phase0_statistical_study.py
"""
import logging
import sys
import pickle
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("eth_phase0")

OUT_DIR = ROOT / "prompts"
PLOTS_DIR = OUT_DIR / "plots" / "fase0"
TABLES_DIR = OUT_DIR / "tables"
for d in [OUT_DIR, PLOTS_DIR, TABLES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

REPORT_PATH = OUT_DIR / "eth_phase0_report.md"

# Correlações BTC 2026 de referência (para comparação)
BTC_CORRELATIONS = {
    "g4_oi_coin_margin":  -0.472,
    "g5_stablecoin":      +0.326,
    "g7_etf":             +0.263,
    "g6_bubble":          -0.345,
    "g3_dgs10":           -0.315,
    "g3_curve":           -0.280,
    "g9_taker":           +0.060,
    "g10_funding":        +0.023,
    "g8_fg":              +0.150,
}


def load_eth_data():
    """Carrega e consolida todos os dados ETH."""
    # Spot 1h
    spot = pd.read_parquet(ROOT / "data/01_raw/spot/eth_1h.parquet")
    spot["timestamp"] = pd.to_datetime(spot["timestamp"], utc=True)
    spot = spot.sort_values("timestamp").reset_index(drop=True)
    
    # Resample to daily for analysis
    spot_d = spot.set_index("timestamp").resample("D").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum"
    }).reset_index()
    
    # Calcular forward returns
    for horizon in [1, 3, 7, 14]:
        spot_d[f"fwd_return_{horizon}d"] = spot_d["close"].shift(-horizon) / spot_d["close"] - 1
    
    # Features técnicas
    spot_d["ret_1d"] = spot_d["close"].pct_change(1)
    spot_d["ret_7d"] = spot_d["close"].pct_change(7)
    spot_d["vol_ratio"] = spot_d["volume"] / spot_d["volume"].rolling(30).mean()
    
    # RSI (14d)
    delta = spot_d["close"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    spot_d["rsi_14"] = 100 - (100 / (1 + rs))
    
    # Moving averages
    spot_d["ma_21"] = spot_d["close"].rolling(21).mean()
    spot_d["ma_50"] = spot_d["close"].rolling(50).mean()
    spot_d["ma_200"] = spot_d["close"].rolling(200).mean()
    
    # BB
    ma20 = spot_d["close"].rolling(20).mean()
    std20 = spot_d["close"].rolling(20).std()
    spot_d["bb_upper"] = ma20 + 2 * std20
    spot_d["bb_lower"] = ma20 - 2 * std20
    spot_d["bb_pct"] = (spot_d["close"] - spot_d["bb_lower"]) / (spot_d["bb_upper"] - spot_d["bb_lower"])
    
    # Derivatives (resample 4h → 1d)
    oi = pd.read_parquet(ROOT / "data/01_raw/futures/eth_oi_4h.parquet")
    funding = pd.read_parquet(ROOT / "data/01_raw/futures/eth_funding_4h.parquet")
    taker = pd.read_parquet(ROOT / "data/01_raw/futures/eth_taker_4h.parquet")
    
    for df in [oi, funding, taker]:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    
    # Resample to daily
    oi_d = oi.set_index("timestamp").resample("D").mean(numeric_only=True).reset_index()
    funding_d = funding.set_index("timestamp").resample("D").mean(numeric_only=True).reset_index()
    taker_d = taker.set_index("timestamp").resample("D").mean(numeric_only=True).reset_index()
    
    # Merge tudo
    df = spot_d.merge(oi_d, on="timestamp", how="left", suffixes=("", "_oi"))
    df = df.merge(funding_d, on="timestamp", how="left", suffixes=("", "_funding"))
    df = df.merge(taker_d, on="timestamp", how="left", suffixes=("", "_taker"))
    
    # Filtrar janela overlap completo (últimos 180 dias)
    df = df.tail(180).reset_index(drop=True)
    
    logger.info(f"Loaded {len(df)} days of ETH data")
    logger.info(f"Period: {df['timestamp'].min().date()} → {df['timestamp'].max().date()}")
    logger.info(f"Columns: {list(df.columns)}")
    
    return df


def compute_zscores(df, window=30):
    """Compute z-scores for key features."""
    df = df.copy()
    
    # OI z-score (coin margin se disponível)
    oi_col = None
    for col in ["oi_coin_margin", "open_interest_usd", "oi_stablecoin", "oi_total"]:
        if col in df.columns and df[col].notna().sum() > 60:
            oi_col = col
            break
    if oi_col:
        df["oi_z"] = (df[oi_col] - df[oi_col].rolling(window).mean()) / df[oi_col].rolling(window).std()
        logger.info(f"Using OI column: {oi_col}")
    
    # Funding z-score
    funding_cols = [c for c in df.columns if "funding" in c.lower() and "rate" in c.lower()]
    if funding_cols:
        funding_col = funding_cols[0]
        df["funding_z"] = (df[funding_col] - df[funding_col].rolling(window).mean()) / df[funding_col].rolling(window).std()
        logger.info(f"Using funding column: {funding_col}")
    
    # Taker ratio z-score
    taker_cols = [c for c in df.columns if "taker" in c.lower()]
    if taker_cols:
        buy_cols = [c for c in taker_cols if "buy" in c.lower()]
        sell_cols = [c for c in taker_cols if "sell" in c.lower()]
        if buy_cols and sell_cols:
            df["taker_ratio"] = df[buy_cols[0]] / (df[buy_cols[0]] + df[sell_cols[0]])
        elif "taker_ratio" in df.columns:
            pass
        else:
            df["taker_ratio"] = df[taker_cols[0]]
        
        df["taker_z"] = (df["taker_ratio"] - df["taker_ratio"].rolling(window).mean()) / df["taker_ratio"].rolling(window).std()
    
    # Volume z-score
    df["volume_z"] = (df["volume"] - df["volume"].rolling(window).mean()) / df["volume"].rolling(window).std()
    
    # RSI z-score
    df["rsi_z"] = (df["rsi_14"] - 50) / 20
    
    return df


def h1_correlation_analysis(df):
    """H1: Correlações ETH vs forward returns."""
    correlations = {}
    
    feature_cols = [
        ("oi_z", "g4_oi_coin_margin"),
        ("funding_z", "g10_funding"),
        ("taker_z", "g9_taker"),
        ("volume_z", "volume"),
        ("rsi_14", "rsi"),
        ("bb_pct", "bb_pct"),
        ("ret_1d", "ret_1d"),
        ("ret_7d", "ret_7d"),
    ]
    
    for horizon in [1, 3, 7, 14]:
        target = f"fwd_return_{horizon}d"
        if target not in df.columns:
            continue
        
        corrs = {}
        for col, label in feature_cols:
            if col not in df.columns:
                continue
            
            valid = df[[col, target]].dropna()
            if len(valid) < 30:
                continue
            
            pearson_r, pearson_p = pearsonr(valid[col], valid[target])
            spearman_r, spearman_p = spearmanr(valid[col], valid[target])
            
            corrs[label] = {
                "pearson": pearson_r,
                "pearson_p": pearson_p,
                "spearman": spearman_r,
                "n": len(valid),
                "significant": pearson_p < 0.05,
            }
        
        correlations[f"{horizon}d"] = corrs
    
    return correlations


def h2_gate_power_analysis(correlations):
    """H2: Quais gates têm poder preditivo em ETH?"""
    corrs_7d = correlations.get("7d", {})
    
    gate_status = {}
    for gate, ref_corr in BTC_CORRELATIONS.items():
        eth_corr = corrs_7d.get(gate, {}).get("pearson", np.nan)
        
        if pd.isna(eth_corr):
            gate_status[gate] = {
                "btc_corr": ref_corr,
                "eth_corr": None,
                "diff": None,
                "status": "NO_DATA",
                "action": "skip",
            }
            continue
        
        diff = abs(eth_corr - ref_corr)
        abs_eth = abs(eth_corr)
        
        if abs_eth > 0.20:
            status = "STRONG"
            action = "keep"
        elif abs_eth > 0.10:
            status = "MODERATE"
            action = "keep_reduced_weight"
        else:
            status = "WEAK"
            action = "discard"
        
        gate_status[gate] = {
            "btc_corr": ref_corr,
            "eth_corr": eth_corr,
            "diff": diff,
            "abs_eth": abs_eth,
            "status": status,
            "action": action,
            "significant": corrs_7d.get(gate, {}).get("significant", False),
        }
    
    return gate_status


def h3_regime_analysis(df):
    """H3: R5C HMM (BTC) funciona em ETH?"""
    model_path = ROOT / "data/03_models/r5c_hmm.pkl"
    
    if not model_path.exists():
        logger.warning("R5C HMM model not found, skipping regime analysis")
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
    df["vol_ratio_log"] = np.log(vol_short / vol_long)
    df["vol_short_z"] = (vol_short - vol_short.rolling(60).mean()) / vol_short.rolling(60).std()
    
    df["volume_log"] = np.log(df["volume"])
    df["volume_log_z"] = (df["volume_log"] - df["volume_log"].rolling(60).mean()) / df["volume_log"].rolling(60).std()
    
    df["drawdown"] = df["close"] / df["close"].rolling(30).max() - 1
    df["drawdown_change_5d"] = df["drawdown"] - df["drawdown"].shift(5)
    df["drawdown_change_5d_z"] = (df["drawdown_change_5d"] - df["drawdown_change_5d"].rolling(60).mean()) / df["drawdown_change_5d"].rolling(60).std()
    
    features = [
        "log_return", "log_return_21d", "vol_ratio_log",
        "vol_short_z", "volume_log_z", "drawdown_change_5d_z"
    ]
    
    valid_df = df[features].dropna()
    if len(valid_df) < 30:
        logger.warning("Not enough data for regime analysis")
        return None
    
    try:
        regimes = model.predict(valid_df.values)
        
        df_regime = df.loc[valid_df.index].copy()
        df_regime["regime"] = regimes
        
        regime_returns = df_regime.groupby("regime")["log_return"].mean().sort_values()
        regime_map = {}
        for i, (reg_idx, _) in enumerate(regime_returns.items()):
            if i == 0:
                regime_map[reg_idx] = "Bear"
            elif i == 1:
                regime_map[reg_idx] = "Sideways"
            else:
                regime_map[reg_idx] = "Bull"
        
        df_regime["regime_name"] = df_regime["regime"].map(regime_map)
        
        regime_stats = {}
        for name in ["Bull", "Sideways", "Bear"]:
            sub = df_regime[df_regime["regime_name"] == name]
            if sub.empty:
                continue
            
            regime_stats[name] = {
                "n": len(sub),
                "pct": len(sub) / len(df_regime) * 100,
                "avg_fwd_7d": sub["fwd_return_7d"].mean() if "fwd_return_7d" in sub.columns else None,
                "median_fwd_7d": sub["fwd_return_7d"].median() if "fwd_return_7d" in sub.columns else None,
                "std_fwd_7d": sub["fwd_return_7d"].std() if "fwd_return_7d" in sub.columns else None,
            }
        
        if all(r in regime_stats for r in ["Bull", "Sideways", "Bear"]):
            bull_avg = regime_stats["Bull"].get("avg_fwd_7d", 0) or 0
            sideways_avg = regime_stats["Sideways"].get("avg_fwd_7d", 0) or 0
            bear_avg = regime_stats["Bear"].get("avg_fwd_7d", 0) or 0
            
            regime_separates = bull_avg > sideways_avg > bear_avg
        else:
            regime_separates = False
        
        return {
            "regime_stats": regime_stats,
            "regime_separates": regime_separates,
            "df_regime": df_regime[["timestamp", "close", "log_return", "regime", "regime_name"]],
        }
    except Exception as e:
        logger.warning(f"Regime prediction failed: {e}")
        return None


def h4_compute_alignment(gate_status):
    """H4: Model Alignment."""
    keep_gates = {k: v for k, v in gate_status.items() if v.get("action") in ["keep", "keep_reduced_weight"]}
    
    if not keep_gates:
        return 0.0
    
    diffs = [v["diff"] for v in keep_gates.values() if v.get("diff") is not None]
    
    if not diffs:
        return 0.0
    
    avg_diff = np.mean(diffs)
    alignment = max(0, 1 - (avg_diff / 0.5))
    
    return alignment


def decision_logic(alignment, gate_status, regime_result):
    """Decisão final baseada nos critérios."""
    strong_gates = sum(1 for v in gate_status.values() if v.get("status") == "STRONG")
    moderate_gates = sum(1 for v in gate_status.values() if v.get("status") == "MODERATE")
    weak_gates = sum(1 for v in gate_status.values() if v.get("status") == "WEAK")
    
    regime_ok = regime_result.get("regime_separates", False) if regime_result else False
    
    if alignment > 0.70 and strong_gates >= 3 and regime_ok:
        decision = "COPY_BTC_CONFIG"
        description = "ETH muito similar a BTC. Copiar parameters.yml com pequenos ajustes (símbolos, stop loss)."
    elif alignment > 0.40 and (strong_gates + moderate_gates) >= 3:
        decision = "ADAPT_CONFIG"
        description = "ETH parcialmente similar. Recalibrar gates fracos, manter arquitetura."
    elif alignment > 0.20:
        decision = "RECALIBRATE_DEEPLY"
        description = "ETH diferente. Recalibração profunda de gates e pesos necessária."
    else:
        decision = "ABANDON_OR_REDESIGN"
        description = "ETH não se encaixa no framework BTC. Considerar gates ETH-específicos (staking, burn, L2 TVL)."
    
    return {
        "decision": decision,
        "description": description,
        "alignment": alignment,
        "strong_gates": strong_gates,
        "moderate_gates": moderate_gates,
        "weak_gates": weak_gates,
        "regime_ok": regime_ok,
    }


def plot_correlations(gate_status):
    """Correlations BTC vs ETH."""
    gates = []
    btc_corrs = []
    eth_corrs = []
    
    for gate, status in gate_status.items():
        if status.get("eth_corr") is None:
            continue
        gates.append(gate.replace("g", "G").replace("_", " "))
        btc_corrs.append(status["btc_corr"])
        eth_corrs.append(status["eth_corr"])
    
    if not gates:
        return
    
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(gates))
    width = 0.35
    
    ax.bar(x - width/2, btc_corrs, width, label="BTC 2026", color="orange", alpha=0.7)
    ax.bar(x + width/2, eth_corrs, width, label="ETH 180d", color="blue", alpha=0.7)
    
    ax.axhline(0, color="black", linewidth=0.5)
    ax.axhline(0.20, color="green", linestyle=":", alpha=0.5, label="Strong threshold")
    ax.axhline(-0.20, color="green", linestyle=":", alpha=0.5)
    
    ax.set_xticks(x)
    ax.set_xticklabels(gates, rotation=45, ha="right")
    ax.set_ylabel("Correlação com forward return 7d")
    ax.set_title("Correlações Gates: BTC vs ETH")
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    path = PLOTS_DIR / "correlations_eth_vs_btc.png"
    plt.savefig(path, dpi=100, bbox_inches="tight")
    plt.close()
    logger.info(f"✅ {path}")


def plot_regimes(regime_result):
    """Regimes aplicados a ETH."""
    if not regime_result:
        return
    
    df = regime_result["df_regime"]
    if df is None or df.empty:
        return
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    
    colors = {"Bull": "green", "Sideways": "blue", "Bear": "red"}
    for name, color in colors.items():
        mask = df["regime_name"] == name
        if mask.any():
            ax1.scatter(df.loc[mask, "timestamp"], df.loc[mask, "close"],
                       c=color, label=name, s=15, alpha=0.7)
    
    ax1.set_ylabel("ETH Price")
    ax1.set_title("ETH price colorido por regime R5C (treinado em BTC)")
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    df_sorted = df.sort_values("timestamp").copy()
    
    for name, color in colors.items():
        mask = df_sorted["regime_name"] == name
        if mask.any():
            ax2.scatter(df_sorted.loc[mask, "timestamp"], df_sorted.loc[mask, "log_return"] * 100,
                       c=color, s=15, alpha=0.7)
    
    ax2.axhline(0, color="black", linewidth=0.5)
    ax2.set_ylabel("Log return %")
    ax2.set_xlabel("Date")
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    path = PLOTS_DIR / "regimes_eth.png"
    plt.savefig(path, dpi=100, bbox_inches="tight")
    plt.close()
    logger.info(f"✅ {path}")


def generate_report(correlations, gate_status, regime_result, alignment, decision):
    """Gera relatório markdown completo."""
    lines = []
    lines.append("# 🔬 ETH Phase 0 — Statistical Study Report")
    lines.append(f"\n**Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    lines.append(f"**Window:** 180 days")
    lines.append("")
    
    # Decisão final no topo
    lines.append("## 🎯 Decisão\n")
    decision_emoji = {
        "COPY_BTC_CONFIG": "✅",
        "ADAPT_CONFIG": "⚠️",
        "RECALIBRATE_DEEPLY": "🔴",
        "ABANDON_OR_REDESIGN": "❌",
    }.get(decision["decision"], "❓")
    
    lines.append(f"### {decision_emoji} {decision['decision']}\n")
    lines.append(f"**{decision['description']}**\n")
    lines.append(f"- Model Alignment: `{alignment:.2f}`")
    lines.append(f"- Strong gates: {decision['strong_gates']}")
    lines.append(f"- Moderate gates: {decision['moderate_gates']}")
    lines.append(f"- Weak gates: {decision['weak_gates']}")
    lines.append(f"- Regime separates returns: {'✅' if decision['regime_ok'] else '❌'}")
    lines.append("")
    
    # H1
    lines.append("## H1: Correlações ETH vs Forward Returns\n")
    for horizon, corrs in correlations.items():
        if not corrs:
            continue
        lines.append(f"### Horizonte {horizon}")
        lines.append("| Feature | Pearson | p-value | Spearman | N | Significant |")
        lines.append("|---------|---------|---------|----------|---|-------------|")
        for feat, c in corrs.items():
            sig = "✅" if c.get("significant") else "❌"
            lines.append(
                f"| {feat} | {c['pearson']:+.3f} | {c['pearson_p']:.3f} | "
                f"{c['spearman']:+.3f} | {c['n']} | {sig} |"
            )
        lines.append("")
    
    # H2
    lines.append("## H2: Poder Preditivo dos Gates\n")
    lines.append("| Gate | BTC corr | ETH corr | |Δ| | Status | Action |")
    lines.append("|------|----------|----------|-----|--------|--------|")
    for gate, s in gate_status.items():
        btc = s.get("btc_corr", "?")
        eth = s.get("eth_corr", "N/A")
        diff = s.get("diff", "?")
        
        btc_str = f"{btc:+.3f}" if isinstance(btc, (int, float)) else btc
        eth_str = f"{eth:+.3f}" if isinstance(eth, (int, float)) else eth
        diff_str = f"{diff:.3f}" if isinstance(diff, (int, float)) else diff
        
        emoji = {"STRONG": "🟢", "MODERATE": "🟡", "WEAK": "🔴", "NO_DATA": "⚪"}.get(s.get("status"), "?")
        lines.append(
            f"| {gate} | {btc_str} | {eth_str} | {diff_str} | "
            f"{emoji} {s.get('status')} | `{s.get('action')}` |"
        )
    lines.append("")
    
    # H3
    lines.append("## H3: R5C HMM em ETH\n")
    if regime_result and regime_result.get("regime_stats"):
        lines.append("| Regime | N | % | Avg fwd 7d | Median fwd 7d |")
        lines.append("|--------|---|---|------------|---------------|")
        for name, stats in regime_result["regime_stats"].items():
            avg = stats.get("avg_fwd_7d")
            med = stats.get("median_fwd_7d")
            avg_str = f"{avg*100:+.2f}%" if avg is not None else "N/A"
            med_str = f"{med*100:+.2f}%" if med is not None else "N/A"
            lines.append(
                f"| {name} | {stats['n']} | {stats['pct']:.1f}% | {avg_str} | {med_str} |"
            )
        lines.append("")
        lines.append(f"**Bull > Sideways > Bear?** {'✅ SIM' if regime_result['regime_separates'] else '❌ NÃO'}")
    else:
        lines.append("*R5C não disponível ou análise falhou.*")
    lines.append("")
    
    # H4
    lines.append("## H4: Model Alignment\n")
    lines.append(f"**Alignment = {alignment:.3f}**\n")
    if alignment > 0.70:
        lines.append("🟢 ETH estruturalmente similar a BTC → copiar config")
    elif alignment > 0.40:
        lines.append("🟡 ETH parcialmente similar → adaptar config")
    elif alignment > 0.20:
        lines.append("🔴 ETH bastante diferente → recalibrar profundamente")
    else:
        lines.append("⚫ ETH não se encaixa → considerar abandonar ou redesign completo")
    lines.append("")
    
    # Next
    lines.append("## 🎯 Próximos Passos\n")
    if decision["decision"] == "COPY_BTC_CONFIG":
        lines.append("1. Criar `conf/parameters_eth.yml` copiado de `parameters.yml`")
        lines.append("2. Ajustar símbolos (BTC → ETH) e thresholds de preço")
        lines.append("3. Rodar paper trading em paralelo (separate portfolio)")
        lines.append("4. Monitorar 2-4 semanas antes de dinheiro real")
    elif decision["decision"] == "ADAPT_CONFIG":
        lines.append(f"1. Manter gates STRONG: {[g for g, s in gate_status.items() if s.get('status') == 'STRONG']}")
        lines.append(f"2. Reduzir peso dos MODERATE: {[g for g, s in gate_status.items() if s.get('status') == 'MODERATE']}")
        lines.append(f"3. Desabilitar WEAK: {[g for g, s in gate_status.items() if s.get('status') == 'WEAK']}")
        lines.append("4. Calibrar threshold ETH-específico baseado nas correlações")
    elif decision["decision"] == "RECALIBRATE_DEEPLY":
        lines.append("1. Retreinar R5C HMM com dados ETH")
        lines.append("2. Re-examinar correlações em diferentes janelas (90d, 360d)")
        lines.append("3. Considerar features ETH-específicas (staking yield, ETH/BTC ratio)")
    else:
        lines.append("1. Investigar features ETH-específicas: ETH/BTC ratio, L2 TVL, staking APY, gas fees")
        lines.append("2. Construir modelo fundamentalmente diferente")
        lines.append("3. Considerar se vale o esforço vs focar só em BTC")
    
    with open(REPORT_PATH, "w") as f:
        f.write("\n".join(lines))
    logger.info(f"📄 Report: {REPORT_PATH}")


def main():
    logger.info("=" * 60)
    logger.info("ETH Phase 0 — Statistical Study")
    logger.info("=" * 60)
    
    df = load_eth_data()
    df = compute_zscores(df)
    df.to_csv(TABLES_DIR / "eth_phase0_features.csv", index=False)
    
    logger.info("\n── H1: Computing correlations ──")
    correlations = h1_correlation_analysis(df)
    
    logger.info("\n── H2: Evaluating gate power ──")
    gate_status = h2_gate_power_analysis(correlations)
    
    logger.info("\n── H3: Regime analysis (R5C HMM) ──")
    regime_result = h3_regime_analysis(df)
    
    logger.info("\n── H4: Computing model alignment ──")
    alignment = h4_compute_alignment(gate_status)
    
    decision = decision_logic(alignment, gate_status, regime_result)
    
    logger.info("\n── Generating plots ──")
    plot_correlations(gate_status)
    plot_regimes(regime_result)
    
    logger.info("\n── Generating report ──")
    generate_report(correlations, gate_status, regime_result, alignment, decision)
    
    print("\n" + "=" * 60)
    print("RESULTADO")
    print("=" * 60)
    print(f"Decision:   {decision['decision']}")
    print(f"Alignment:  {alignment:.3f}")
    print(f"Strong:     {decision['strong_gates']} gates")
    print(f"Moderate:   {decision['moderate_gates']} gates")
    print(f"Weak:       {decision['weak_gates']} gates")
    print(f"Regime OK:  {decision['regime_ok']}")
    print(f"\n📄 Report: {REPORT_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()
```

## Checklist

1. [ ] Criar `scripts/eth_phase0_statistical_study.py` (código acima)
2. [ ] Rodar: `python scripts/eth_phase0_statistical_study.py`
3. [ ] Verificar outputs:
   - `prompts/eth_phase0_report.md`
   - `prompts/plots/fase0/correlations_eth_vs_btc.png`
   - `prompts/plots/fase0/regimes_eth.png`
   - `prompts/tables/eth_phase0_features.csv`
4. [ ] Analisar decisão: COPY / ADAPT / RECALIBRATE / ABANDON
5. [ ] Se dados faltarem, rodar com o que tiver e documentar limitações
6. [ ] Git commit + push

## Critérios de decisão (resumo)

```
alignment > 0.70 AND strong_gates >= 3 AND regime_ok:
    → COPY_BTC_CONFIG (próxima fase: parameters_eth.yml)
    
alignment > 0.40 AND (strong+moderate) >= 3:
    → ADAPT_CONFIG (ajustes localizados)
    
alignment > 0.20:
    → RECALIBRATE_DEEPLY (retreinar HMM, refazer scoring)
    
alignment <= 0.20:
    → ABANDON_OR_REDESIGN (ETH requer features próprias)
```

## O que NÃO fazer nesta fase

- ❌ Implementar `parameters_eth.yml` agora (isso é próxima fase)
- ❌ Rodar paper trading ETH (isso é Fase 1+)
- ❌ Mudar o BTC paper trading (produção intocada)
- ❌ Discutir gates novos específicos para ETH (staking, burn, etc) — só se alignment < 0.40

## Filosofia aplicada

**Hipóteses testáveis + critérios objetivos = decisão informada.**

O estudo vai dizer **direto** se o framework funciona em ETH ou não. Sem feeling, sem achismo.

Se der COPY ou ADAPT → seguimos pra Fase 1 (implementação) com confiança.
Se der RECALIBRATE ou ABANDON → paramos e pensamos com cuidado antes de gastar tempo.
