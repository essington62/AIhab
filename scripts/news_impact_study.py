"""
news_impact_study.py
Estudo de impacto de notícias macro no preço do BTC — 2026.
Roda localmente: python scripts/news_impact_study.py
"""

import json
import os
import warnings
from datetime import timezone

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------
NEWS_PATH = "data/01_raw/news/macro_news.parquet"
BTC_PATH = "data/01_raw/spot/btc_1h.parquet"
OUT_MD = "prompts/news_impact_study.md"
OUT_CONTEXT = "prompts/news_regime_context.md"

os.makedirs("prompts", exist_ok=True)

# ---------------------------------------------------------------------------
# PARTE 1 — CARREGAMENTO DE DADOS
# ---------------------------------------------------------------------------
print("=" * 60)
print("PARTE 1 — CARREGAMENTO DE DADOS")
print("=" * 60)

# macro_news
news_raw = pd.read_parquet(NEWS_PATH)

# descobre timestamp dinamicamente
ts_col = next(
    (c for c in news_raw.columns if "time" in c.lower() or c in ("ts", "date", "published")),
    None,
)
if ts_col != "timestamp":
    news_raw = news_raw.rename(columns={ts_col: "timestamp"})

news_raw["timestamp"] = pd.to_datetime(news_raw["timestamp"], utc=True, errors="coerce")
news_raw = news_raw.dropna(subset=["timestamp"])

print(f"\nmacro_news: {news_raw.shape[0]} artigos × {news_raw.shape[1]} colunas")
print(f"Colunas: {news_raw.columns.tolist()}")
print(f"Range: {news_raw['timestamp'].min().date()} → {news_raw['timestamp'].max().date()}")
print(f"ds_score preenchidos: {news_raw['ds_score'].notna().sum()} de {len(news_raw)}")

# btc_1h
btc_raw = pd.read_parquet(BTC_PATH)
ts_col_btc = next(
    (c for c in btc_raw.columns if "time" in c.lower() or c in ("ts", "date")),
    None,
)
if ts_col_btc != "timestamp":
    btc_raw = btc_raw.rename(columns={ts_col_btc: "timestamp"})

btc_raw["timestamp"] = pd.to_datetime(btc_raw["timestamp"], utc=True, errors="coerce")
btc_raw = btc_raw.sort_values("timestamp").reset_index(drop=True)

# retornos futuros e pré-notícia
btc = btc_raw.copy()
btc["ret_1h_pre"] = btc["close"] / btc["close"].shift(1) - 1
btc["ret_1h"] = btc["close"].shift(-1) / btc["close"] - 1
btc["ret_4h"] = btc["close"].shift(-4) / btc["close"] - 1
btc["ret_12h"] = btc["close"].shift(-12) / btc["close"] - 1
btc["ret_24h"] = btc["close"].shift(-24) / btc["close"] - 1

baseline_ret_1h = btc["ret_1h"].mean()
baseline_ret_4h = btc["ret_4h"].mean()
baseline_ret_12h = btc["ret_12h"].mean()
baseline_ret_24h = btc["ret_24h"].mean()

print(f"\nbtc_1h: {btc.shape[0]} candles")
print(f"Range: {btc['timestamp'].min().date()} → {btc['timestamp'].max().date()}")
print(f"Baseline ret_1h:  {baseline_ret_1h*100:+.4f}%")
print(f"Baseline ret_4h:  {baseline_ret_4h*100:+.4f}%")
print(f"Baseline ret_12h: {baseline_ret_12h*100:+.4f}%")
print(f"Baseline ret_24h: {baseline_ret_24h*100:+.4f}%")

# filtra notícias dentro da janela btc
btc_start = btc["timestamp"].min()
btc_end = btc["timestamp"].max()
news = news_raw[
    (news_raw["timestamp"] >= btc_start) & (news_raw["timestamp"] <= btc_end)
].copy()
print(f"\nArtigos dentro da janela BTC ({btc_start.date()} → {btc_end.date()}): {len(news)}")

# ---------------------------------------------------------------------------
# PARTE 2 — CLASSIFICAÇÃO HIERÁRQUICA DE EVENTOS
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("PARTE 2 — CLASSIFICAÇÃO HIERÁRQUICA DE EVENTOS")
print("=" * 60)

KEYWORD_MAP = [
    # FED — ordem: hawkish/dovish antes do fallback neutral
    ("FED", "FOMC_HAWKISH", [
        "rate hike", "hawkish", "higher for longer", "tighten",
        "warsh", "regime change fed", "rate increase", "rate rises",
        "interest rate hike", "rate hike possible",
    ]),
    ("FED", "FOMC_DOVISH", [
        "rate cut", "dovish", "pivot", "easing", "pause", "hold rates",
        "rate-cut hopes", "rate cut hopes", "foresee rate cut",
    ]),
    ("FED", "FOMC_NEUTRAL", [
        "fed", "fomc", "powell", "federal reserve", "monetary policy",
        "jerome powell", "treasury", "inflation", "cpi", "ppi",
    ]),
    # GEO
    ("GEO", "HORMUZ_BLOCK", [
        "hormuz", "strait", "blockade", "tanker", "shipping lane", "naval",
    ]),
    ("GEO", "WAR_ESCALATION", [
        "airstrike", "missile", "invasion", "troops",
        "military strike", "attack", "escalat",
    ]),
    ("GEO", "WAR_DEESCALATION", [
        "ceasefire", "peace talks", "negotiat", "truce",
        "diplomatic", "agreement",
    ]),
    # ENERGY
    ("ENERGY", "OPEC_ACTION", [
        "opec", "oil cut", "production cut", "crude output", "energy supply",
    ]),
    ("ENERGY", "OIL_PRICE", [
        "oil price", "crude price", "wti", "brent", "energy crisis",
        "fuel price", "oil futures", "oil prices", "crude",
        "gas price", "gas prices",
    ]),
    # LIQUIDITY
    ("LIQUIDITY", "STABLECOIN", [
        "stablecoin", "usdt", "usdc", "tether", "circle", "stablecoin mcap",
    ]),
    ("LIQUIDITY", "CARRY_UNWIND", [
        "yen carry", "boj", "japan rate", "carry trade", "unwind",
    ]),
    # RISK
    ("RISK", "RECESSION", [
        "recession", "gdp", "contraction", "economic slowdown", "depression",
    ]),
    ("RISK", "BANK_CRISIS", [
        "bank fail", "bank run", "credit crisis", "contagion", "systemic",
    ]),
]


def classify_article(title: str):
    t = title.lower()
    for macro_cat, evento, keywords in KEYWORD_MAP:
        if any(kw in t for kw in keywords):
            return macro_cat, evento
    return "OTHER", "UNCATEGORIZED"


news[["macro_category", "evento_cluster"]] = news["title"].apply(
    lambda t: pd.Series(classify_article(t))
)

dist = news.groupby(["macro_category", "evento_cluster"]).size().reset_index(name="N")
print("\nDistribuição por categoria/evento:")
print(dist.to_string(index=False))

# ---------------------------------------------------------------------------
# PARTE 3 — FILTRO EX-ANTE
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("PARTE 3 — FILTRO EX-ANTE (abs(ret_1h_pre) < 0.3%)")
print("=" * 60)

# merge_asof: para cada notícia, candle mais próximo ≤ timestamp
btc_sorted = btc.sort_values("timestamp")
news_sorted = news.sort_values("timestamp").copy()

merged = pd.merge_asof(
    news_sorted,
    btc_sorted[["timestamp", "close", "ret_1h_pre", "ret_1h", "ret_4h", "ret_12h", "ret_24h"]],
    on="timestamp",
    direction="nearest",
    tolerance=pd.Timedelta("30min"),
)

merged["ex_ante_ok"] = merged["ret_1h_pre"].abs() < 0.003
kept = merged[merged["ex_ante_ok"]].copy()
dropped = merged[~merged["ex_ante_ok"]].copy()

print(f"\nTotal artigos na janela: {len(merged)}")
print(f"KEPT  (|ret_1h_pre| < 0.3%): {len(kept)}")
print(f"DROP  (|ret_1h_pre| >= 0.3%): {len(dropped)}")
print(f"Sem candle BTC (±30min): {merged['ret_1h_pre'].isna().sum()}")

print("\n% KEPT por categoria:")
for (mc, ev), grp in merged.groupby(["macro_category", "evento_cluster"]):
    ok = grp["ex_ante_ok"].sum()
    pct = 100 * ok / len(grp) if len(grp) > 0 else 0
    print(f"  {mc}/{ev}: {ok}/{len(grp)} ({pct:.0f}%)")

# ---------------------------------------------------------------------------
# PARTE 4 — CÁLCULO DE EDGE
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("PARTE 4 — CÁLCULO DE EDGE (ajustado pelo baseline)")
print("=" * 60)

df = kept.copy()
df["edge_1h"] = df["ret_1h"] - baseline_ret_1h
df["edge_4h"] = df["ret_4h"] - baseline_ret_4h
df["edge_12h"] = df["ret_12h"] - baseline_ret_12h
df["edge_24h"] = df["ret_24h"] - baseline_ret_24h

# direcional accuracy: só para artigos com ds_score
df["has_ds"] = df["ds_score"].notna()


def directional_accuracy(grp):
    sub = grp[grp["has_ds"] & grp["edge_4h"].notna() & grp["ds_score"].notna()]
    if len(sub) == 0:
        return np.nan
    correct = ((sub["ds_score"] < 0) & (sub["edge_4h"] < 0)) | (
        (sub["ds_score"] > 0) & (sub["edge_4h"] > 0)
    )
    return correct.mean() * 100


def peak_horizon(row):
    vals = {
        "1h": abs(row.get("edge_1h_mean", 0) or 0),
        "4h": abs(row.get("edge_4h_mean", 0) or 0),
        "12h": abs(row.get("edge_12h_mean", 0) or 0),
        "24h": abs(row.get("edge_24h_mean", 0) or 0),
    }
    return max(vals, key=vals.get)


summary_rows = []
for (mc, ev), grp in df.groupby(["macro_category", "evento_cluster"]):
    n = len(grp)
    n_ex_ante = grp["ex_ante_ok"].sum() if "ex_ante_ok" in grp.columns else n
    n_total_before = len(merged[(merged["macro_category"] == mc) & (merged["evento_cluster"] == ev)])
    ex_ante_pct = 100 * n / n_total_before if n_total_before > 0 else np.nan

    ds_mean = grp["ds_score"].mean()
    e1h_m = grp["edge_1h"].mean()
    e4h_m = grp["edge_4h"].mean()
    e4h_s = grp["edge_4h"].std()
    e12h_m = grp["edge_12h"].mean()
    e24h_m = grp["edge_24h"].mean()
    e24h_s = grp["edge_24h"].std()
    acc = directional_accuracy(grp)

    summary_rows.append({
        "macro": mc,
        "evento": ev,
        "N": n,
        "ex_ante_pct": ex_ante_pct,
        "ds_score_mean": ds_mean,
        "edge_1h_mean": e1h_m,
        "edge_4h_mean": e4h_m,
        "edge_4h_std": e4h_s,
        "edge_12h_mean": e12h_m,
        "edge_24h_mean": e24h_m,
        "edge_24h_std": e24h_s,
        "acuracia_4h": acc,
    })

summary = pd.DataFrame(summary_rows)
summary["horizonte_pico"] = summary.apply(peak_horizon, axis=1)

# ---------------------------------------------------------------------------
# PARTE 5 — ANÁLISE DE SATURAÇÃO
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("PARTE 5 — ANÁLISE DE SATURAÇÃO (N >= 5)")
print("=" * 60)

sat_rows = []
eligible = df.groupby(["macro_category", "evento_cluster"]).filter(lambda g: len(g) >= 5)

if len(eligible) == 0:
    print("Nenhum cluster com N >= 5 artigos KEPT. Saturação não calculável.")
    sat_table = pd.DataFrame()
else:
    for (mc, ev), grp in eligible.groupby(["macro_category", "evento_cluster"]):
        g = grp.sort_values("timestamp").reset_index(drop=True)
        e_1st = g.loc[0, "edge_4h"] if len(g) >= 1 else np.nan
        e_2nd = g.loc[1, "edge_4h"] if len(g) >= 2 else np.nan
        e_3rd_plus = g.loc[2:, "edge_4h"].mean() if len(g) >= 3 else np.nan

        satura = (
            (not np.isnan(e_1st)) and (not np.isnan(e_3rd_plus)) and
            (abs(e_3rd_plus) < 0.5 * abs(e_1st))
        ) if not (np.isnan(e_1st) or np.isnan(e_3rd_plus)) else False

        sat_rows.append({
            "Cluster": f"{mc}/{ev}",
            "N": len(g),
            "edge_1st": f"{e_1st*100:+.2f}%" if not np.isnan(e_1st) else "—",
            "edge_2nd": f"{e_2nd*100:+.2f}%" if not np.isnan(e_2nd) else "—",
            "edge_3rd+": f"{e_3rd_plus*100:+.2f}%" if not np.isnan(e_3rd_plus) else "—",
            "Satura?": "✅ sim" if satura else "❌ não",
            "_satura_bool": satura,
        })
        print(f"  {mc}/{ev}: 1st={e_1st*100:+.2f}% 2nd={e_2nd*100:+.2f}% "
              f"3rd+={e_3rd_plus*100:+.2f}% → {'✅ satura' if satura else '❌ não satura'}")

    sat_table = pd.DataFrame(sat_rows)
    print("\nTabela de saturação:")
    print(sat_table[["Cluster", "N", "edge_1st", "edge_2nd", "edge_3rd+", "Satura?"]].to_string(index=False))

# adiciona saturação no summary
def lookup_satura(mc, ev):
    if len(sat_rows) == 0:
        return "N<5"
    match = [r for r in sat_rows if r["Cluster"] == f"{mc}/{ev}"]
    if not match:
        return "N<5"
    return "sim" if match[0]["_satura_bool"] else "não"


summary["satura"] = summary.apply(lambda r: lookup_satura(r["macro"], r["evento"]), axis=1)

# ---------------------------------------------------------------------------
# PARTE 6 — TABELA FINAL DE CALIBRAÇÃO
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("PARTE 6 — TABELA FINAL DE CALIBRAÇÃO (ordenada por |edge_4h|)")
print("=" * 60)

summary_sorted = summary.reindex(
    summary["edge_4h_mean"].abs().sort_values(ascending=False).index
)

header = (
    f"{'macro':<10} {'evento':<18} {'N':>3} {'ex%':>5} {'DS':>6} "
    f"{'edge_4h':>8} {'edge_24h':>9} {'acc_4h':>7} {'satura':>6} {'pico':>5}"
)
sep = "─" * len(header)
print(f"\n{header}")
print(sep)
for _, r in summary_sorted.iterrows():
    acc_str = f"{r['acuracia_4h']:.0f}%" if not np.isnan(r["acuracia_4h"]) else "  N/A"
    ds_str = f"{r['ds_score_mean']:+.2f}" if not np.isnan(r["ds_score_mean"]) else "  N/A"
    print(
        f"{r['macro']:<10} {r['evento']:<18} {r['N']:>3} "
        f"{r['ex_ante_pct']:>4.0f}% {ds_str:>6} "
        f"{r['edge_4h_mean']*100:>+7.2f}% {r['edge_24h_mean']*100:>+8.2f}% "
        f"{acc_str:>7} {r['satura']:>6} {r['horizonte_pico']:>5}"
    )

# ---------------------------------------------------------------------------
# PARTE 7 — OUTPUT PARA PROMPT DO R1
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("PARTE 7 — GERANDO OUTPUTS")
print("=" * 60)

# --- context para R1 ---
lines_context = []
lines_context.append("═" * 55)
lines_context.append("CALIBRAÇÃO HISTÓRICA BTC 2026 — NEWS IMPACT")
lines_context.append("═" * 55)
lines_context.append(
    f"Baseline: BTC retorna {baseline_ret_4h*100:+.4f}% em 4h sem eventos (drift natural)"
)
lines_context.append(f"Período: {btc['timestamp'].min().date()} → {btc['timestamp'].max().date()}")
lines_context.append(f"Artigos analisados: {len(kept)} (ex-ante filter aplicado)")
lines_context.append("")
lines_context.append("Por categoria (edge = retorno ajustado pelo baseline):")
lines_context.append("")

for _, r in summary_sorted.iterrows():
    mc, ev, n = r["macro"], r["evento"], int(r["N"])
    e4 = r["edge_4h_mean"] * 100
    e24 = r["edge_24h_mean"] * 100
    acc = r["acuracia_4h"]
    pico = r["horizonte_pico"]
    sat = r["satura"]

    acc_str = f"{acc:.0f}%" if not np.isnan(acc) else "N/A (sem ds_score)"
    conf = (
        0.8 if (not np.isnan(acc) and acc >= 70) else
        0.6 if (not np.isnan(acc) and acc >= 55) else
        0.4
    )

    lines_context.append(f"{mc}/{ev} (N={n}, acurácia={acc_str}):")
    lines_context.append(f"  - edge_4h médio: {e4:+.2f}% | edge_24h: {e24:+.2f}% | horizonte pico: {pico}")
    lines_context.append(f"  - saturação: {sat} {'após 3º evento no mesmo episódio' if sat == 'sim' else '— cada evento mantém força'}")

    if e4 < -0.3:
        direction = "BEAR"
        threshold = "se ds_score < -1"
    elif e4 > 0.3:
        direction = "BULL"
        threshold = "se ds_score > +1"
    else:
        direction = "SIDEWAYS"
        threshold = "sinal fraco — manter classificação DeepSeek"

    if sat == "sim":
        instr = f"instrução: {direction} no 1º/2º evento, SIDEWAYS no 3º+"
    else:
        instr = f"instrução: classificar {direction} {threshold}"

    lines_context.append(f"  - {instr}")
    lines_context.append(f"  - confidence sugerida: {conf}")
    lines_context.append("")

lines_context.append("Confidence sugerida por acurácia:")
lines_context.append("  acurácia >= 70% → confidence 0.8")
lines_context.append("  acurácia 55-70% → confidence 0.6")
lines_context.append("  acurácia < 55%  → confidence 0.4 (ruído)")
lines_context.append("═" * 55)

context_text = "\n".join(lines_context)
with open(OUT_CONTEXT, "w") as f:
    f.write(context_text)
print(f"Salvo: {OUT_CONTEXT}")

# --- estudo completo ---
lines_md = []
lines_md.append("# News Impact Study — BTC 2026")
lines_md.append(f"\nGerado: {pd.Timestamp.now(tz='UTC').strftime('%Y-%m-%d %H:%M UTC')}")
lines_md.append("\n## Dados")
lines_md.append(f"- macro_news: {len(news_raw)} artigos totais, {len(news)} na janela BTC")
lines_md.append(f"- Classificados (ds_score): {news_raw['ds_score'].notna().sum()}")
lines_md.append(f"- btc_1h: {len(btc)} candles")
lines_md.append(f"- Artigos após ex-ante filter: {len(kept)}")
lines_md.append(f"- Baseline ret_4h: {baseline_ret_4h*100:+.4f}%")
lines_md.append(f"- Baseline ret_24h: {baseline_ret_24h*100:+.4f}%")

lines_md.append("\n## Tabela de Calibração\n")
lines_md.append(f"| macro | evento | N | ex_ante% | DS_score | edge_4h | edge_24h | acurácia | satura | horizonte_pico |")
lines_md.append("|---|---|---|---|---|---|---|---|---|---|")
for _, r in summary_sorted.iterrows():
    acc_s = f"{r['acuracia_4h']:.0f}%" if not np.isnan(r["acuracia_4h"]) else "N/A"
    ds_s = f"{r['ds_score_mean']:+.2f}" if not np.isnan(r["ds_score_mean"]) else "N/A"
    lines_md.append(
        f"| {r['macro']} | {r['evento']} | {int(r['N'])} | {r['ex_ante_pct']:.0f}% | "
        f"{ds_s} | {r['edge_4h_mean']*100:+.2f}% | {r['edge_24h_mean']*100:+.2f}% | "
        f"{acc_s} | {r['satura']} | {r['horizonte_pico']} |"
    )

if len(sat_rows) > 0:
    lines_md.append("\n## Tabela de Saturação\n")
    lines_md.append("| Cluster | N | edge_1st | edge_2nd | edge_3rd+ | Satura? |")
    lines_md.append("|---|---|---|---|---|---|")
    for r in sat_rows:
        lines_md.append(
            f"| {r['Cluster']} | {r['N']} | {r['edge_1st']} | "
            f"{r['edge_2nd']} | {r['edge_3rd+']} | {r['Satura?']} |"
        )

lines_md.append("\n## Contexto para Prompt R1\n")
lines_md.append("```")
lines_md.append(context_text)
lines_md.append("```")

with open(OUT_MD, "w") as f:
    f.write("\n".join(lines_md))
print(f"Salvo: {OUT_MD}")

# ---------------------------------------------------------------------------
# RESUMO EXECUTIVO
# ---------------------------------------------------------------------------
print("\n" + "═" * 60)
print("RESUMO EXECUTIVO")
print("═" * 60)

n_total = len(news)
n_kept = len(kept)
n_classified = kept["ds_score"].notna().sum()
top3 = summary_sorted.head(3)

print(f"\n• {n_total} artigos na janela BTC | {n_kept} passaram ex-ante filter ({100*n_kept/n_total:.0f}%)")
print(f"• {n_classified} artigos com ds_score para accuracy metrics")
print(f"• Baseline drift 4h: {baseline_ret_4h*100:+.4f}%")

print(f"\nTop 3 clusters por |edge_4h|:")
for i, (_, r) in enumerate(top3.iterrows(), 1):
    acc_str = f"acc={r['acuracia_4h']:.0f}%" if not np.isnan(r["acuracia_4h"]) else "acc=N/A"
    print(f"  {i}. {r['macro']}/{r['evento']}: edge_4h={r['edge_4h_mean']*100:+.2f}% | {acc_str} | pico={r['horizonte_pico']}")

print("\nTop 3 insights para o prompt do R1:")

# insight 1: maior edge negativo
bear_candidates = summary_sorted[summary_sorted["edge_4h_mean"] < 0].head(1)
if len(bear_candidates):
    r = bear_candidates.iloc[0]
    print(f"  1. {r['macro']}/{r['evento']} tem edge bearish mais forte ({r['edge_4h_mean']*100:+.2f}% em {r['horizonte_pico']}) — "
          f"priorizar score negativo")

# insight 2: maior edge positivo
bull_candidates = summary_sorted[summary_sorted["edge_4h_mean"] > 0].head(1)
if len(bull_candidates):
    r = bull_candidates.iloc[0]
    print(f"  2. {r['macro']}/{r['evento']} tem edge bullish mais forte ({r['edge_4h_mean']*100:+.2f}% em {r['horizonte_pico']}) — "
          f"não bloquear em exceção macro")

# insight 3: acurácia do DeepSeek
acc_candidates = summary_sorted[summary_sorted["acuracia_4h"].notna()].sort_values("acuracia_4h", ascending=False).head(1)
if len(acc_candidates):
    r = acc_candidates.iloc[0]
    print(f"  3. DeepSeek mais preciso em {r['macro']}/{r['evento']}: {r['acuracia_4h']:.0f}% accuracy direcional "
          f"(confidence sugerida: {0.8 if r['acuracia_4h'] >= 70 else 0.6 if r['acuracia_4h'] >= 55 else 0.4})")
elif len(summary_sorted):
    print(f"  3. Amostra pequena (N={n_classified} com ds_score) — accuracy metrics requerem mais dados classificados")

print(f"\nOutputs gerados:")
print(f"  {OUT_MD}")
print(f"  {OUT_CONTEXT}")
