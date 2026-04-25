"""
news_regime.py — Macro news regime classification for BTC via DeepSeek-R1.

Consumes classified news + market flow and produces a holistic regime signal
(BULL / SIDEWAYS / BEAR) with confidence, distinct from per-article classification.

Inputs:
    data/01_raw/news/{crypto,macro,fed}_news.parquet  (ds_score populated)
    data/01_raw/spot/btc_1h.parquet
    data/02_features/gate_zscores.parquet
    data/04_scoring/score_history.parquet

Output:
    data/02_features/news_regime.parquet
    columns: timestamp, regime_hint, confidence, reasoning, n_articles, dominant_cluster

Usage:
    python -m src.data.news_regime            # full run
    python -m src.data.news_regime --dry-run  # print prompt, no API call
"""

import json
import logging
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
import requests

from src.config import get_credential, get_path

logger = logging.getLogger("data_layer.news_regime")

DEEPSEEK_URL  = "https://api.deepseek.com/chat/completions"
MODEL_R1      = "deepseek-reasoner"
NEWS_WINDOW_H = 4        # hours of news to consider
RETENTION     = 8760     # rows cap for output parquet

# ---------------------------------------------------------------------------
# Event cluster keyword map (mirrors news_impact_study.py)
# Priority order: specific keywords before broad fallbacks.
# ---------------------------------------------------------------------------
KEYWORD_MAP = [
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
    ("ENERGY", "OPEC_ACTION", [
        "opec", "oil cut", "production cut", "crude output", "energy supply",
    ]),
    ("ENERGY", "OIL_PRICE", [
        "oil price", "crude price", "wti", "brent", "energy crisis",
        "fuel price", "oil futures", "oil prices", "crude",
        "gas price", "gas prices",
    ]),
    ("LIQUIDITY", "STABLECOIN", [
        "stablecoin", "usdt", "usdc", "tether", "circle", "stablecoin mcap",
    ]),
    ("LIQUIDITY", "CARRY_UNWIND", [
        "yen carry", "boj", "japan rate", "carry trade", "unwind",
    ]),
    ("RISK", "RECESSION", [
        "recession", "gdp", "contraction", "economic slowdown", "depression",
    ]),
    ("RISK", "BANK_CRISIS", [
        "bank fail", "bank run", "credit crisis", "contagion", "systemic",
    ]),
]


def classify_event_cluster(title: str) -> str:
    """Return 'MACRO/EVENTO' string for a given article title."""
    t = title.lower()
    for macro_cat, evento, keywords in KEYWORD_MAP:
        if any(kw in t for kw in keywords):
            return f"{macro_cat}/{evento}"
    return "OTHER/UNCATEGORIZED"


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def _load_market_context() -> dict:
    """Load BTC price context + flow z-scores. Returns flat dict."""
    ctx: dict = {
        "close": None, "ret_1h_pre": None, "ret_4h": None,
        "oi_z": None, "taker_z": None, "funding_z": None,
        "regime_hmm": "Unknown",
    }

    # BTC 1h — last 6 candles is enough for ret_4h + ret_1h_pre
    try:
        btc = pd.read_parquet("data/01_raw/spot/btc_1h.parquet")
        btc["timestamp"] = pd.to_datetime(btc["timestamp"], utc=True, errors="coerce")
        btc = btc.sort_values("timestamp").tail(8).reset_index(drop=True)
        if len(btc) >= 5:
            ctx["close"] = float(btc["close"].iloc[-1])
            ctx["ret_4h"] = float(btc["close"].iloc[-1] / btc["close"].iloc[-5] - 1)
        if len(btc) >= 3:
            # ret_1h_pre: candle at t-1h relative to snapshot (close[-2] / close[-3])
            ctx["ret_1h_pre"] = float(btc["close"].iloc[-2] / btc["close"].iloc[-3] - 1)
    except Exception as e:
        logger.warning(f"btc_1h load failed: {e}")

    # Z-scores
    try:
        zs = pd.read_parquet("data/02_features/gate_zscores.parquet")
        zs["timestamp"] = pd.to_datetime(zs["timestamp"], utc=True, errors="coerce")
        last = zs.sort_values("timestamp").iloc[-1]
        ctx["oi_z"]      = float(last.get("oi_z", 0) or 0)
        ctx["taker_z"]   = float(last.get("taker_z", 0) or 0)
        ctx["funding_z"] = float(last.get("funding_z", 0) or 0)
    except Exception as e:
        logger.warning(f"gate_zscores load failed: {e}")

    # HMM regime from score_history
    try:
        sh = pd.read_parquet("data/04_scoring/score_history.parquet")
        if "regime_multiplier" in sh.columns:
            rm = sh["regime_multiplier"].iloc[-1]
            if rm >= 1.0:
                ctx["regime_hmm"] = "Bull"
            elif rm <= 0.0:
                ctx["regime_hmm"] = "Bear"
            else:
                ctx["regime_hmm"] = "Sideways"
    except Exception as e:
        logger.warning(f"score_history load failed: {e}")

    return ctx


def _load_recent_articles(hours: int = NEWS_WINDOW_H) -> list[dict]:
    """Load classified news from last N hours, attach event_cluster."""
    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=hours)
    articles = []

    for path_str in [
        "data/01_raw/news/fed_news.parquet",
        "data/01_raw/news/macro_news.parquet",
        "data/01_raw/news/crypto_news.parquet",
    ]:
        p = Path(path_str)
        if not p.exists():
            continue
        try:
            df = pd.read_parquet(p)
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
            recent = df[
                (df["timestamp"] >= cutoff)
                & (df.get("ds_classified", pd.Series(False, index=df.index)) == True)  # noqa: E712
                & df["ds_score"].notna()
            ].copy()
            for _, row in recent.iterrows():
                articles.append({
                    "timestamp":     row["timestamp"],
                    "title":         str(row.get("title", "")),
                    "ds_score":      float(row["ds_score"]),
                    "event_cluster": classify_event_cluster(str(row.get("title", ""))),
                    "source":        str(row.get("source", "")),
                })
        except Exception as e:
            logger.warning(f"{path_str}: {e}")

    # dedup by title, keep latest
    seen: set[str] = set()
    deduped = []
    for a in sorted(articles, key=lambda x: x["timestamp"], reverse=True):
        key = a["title"][:60].lower()
        if key not in seen:
            seen.add(key)
            deduped.append(a)

    return sorted(deduped, key=lambda x: x["timestamp"])


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def build_prompt(ctx: dict, articles: list[dict]) -> str:
    """
    Build the full prompt sent to DeepSeek-R1.
    BLOCO A: system instruction
    BLOCO B: market data + news with event clusters
    BLOCO C: ordered rules
    """

    # ── BLOCO A ─────────────────────────────────────────────────────────────
    bloco_a = (
        "You are a senior Bitcoin macro analyst.\n"
        "Classify the BTC market regime for the NEXT 4 HOURS based on macro "
        "news events and market flow data.\n"
        "Return ONLY valid JSON — no markdown, no extra text:\n"
        '{"regime_hint": "BULL|SIDEWAYS|BEAR", '
        '"confidence": 0.0-1.0, '
        '"reasoning": "max 15 words"}'
    )

    # ── BLOCO B ─────────────────────────────────────────────────────────────
    close_str    = f"${ctx['close']:,.0f}"     if ctx["close"]    is not None else "N/A"
    ret4h_str    = f"{ctx['ret_4h']*100:+.2f}%"   if ctx["ret_4h"]   is not None else "N/A"
    ret_pre_str  = f"{ctx['ret_1h_pre']*100:+.2f}%" if ctx["ret_1h_pre"] is not None else "N/A"
    oi_str       = f"{ctx['oi_z']:+.2f}"       if ctx["oi_z"]     is not None else "N/A"
    taker_str    = f"{ctx['taker_z']:+.2f}"    if ctx["taker_z"]  is not None else "N/A"
    funding_str  = f"{ctx['funding_z']:+.2f}"  if ctx["funding_z"] is not None else "N/A"

    bloco_b_lines = [
        "=== MERCADO BTC ===",
        f"Preço atual:           {close_str}",
        f"ret_4h (momentum):     {ret4h_str}",
        f"BTC ret_1h_pre:        {ret_pre_str}  ← movimento ANTES do snapshot",
        f"OI_z (posicionamento): {oi_str}   (>1.5 sobreaquecido, <-1 desalavancando)",
        f"taker_z (fluxo):       {taker_str}  (>0 comprando, <0 vendendo)",
        f"funding_z:             {funding_str}",
        f"",
        f"REGIME HMM (contexto, não replicar): {ctx['regime_hmm']}",
        "",
        f"=== NOTÍCIAS ÚLTIMAS {NEWS_WINDOW_H}H (formato: HH:MM | score | cluster | título) ===",
    ]

    if articles:
        for a in articles:
            ts_str = a["timestamp"].strftime("%H:%M")
            score  = a["ds_score"]
            cluster = a["event_cluster"]
            title   = a["title"][:80]
            bloco_b_lines.append(f"{ts_str} | {score:+.0f} | {cluster} | {title}")
    else:
        bloco_b_lines.append("(nenhuma notícia classificada nas últimas 4h)")

    # Cluster counts
    cluster_counts: Counter = Counter(a["event_cluster"] for a in articles)
    bloco_b_lines.append("")
    bloco_b_lines.append("CONTAGEM POR CLUSTER:")
    if cluster_counts:
        for cluster, cnt in sorted(cluster_counts.items()):
            bloco_b_lines.append(f"  {cluster}: {cnt} artigo(s)")
    else:
        bloco_b_lines.append("  (sem artigos classificados)")

    n_relevant = sum(
        cnt for cl, cnt in cluster_counts.items() if cl != "OTHER/UNCATEGORIZED"
    )
    bloco_b_lines.append(f"\nN artigos relevantes (ex-OTHER): {n_relevant}")

    bloco_b = "\n".join(bloco_b_lines)

    # ── BLOCO C — regras em ordem de prioridade ──────────────────────────────
    bloco_c = """=== REGRAS OBRIGATÓRIAS (em ordem de prioridade) ===

1. FOMC_DOVISH → NUNCA classificar BULL.
   Histórico 2026: sell the news confirmado, accuracy=0% para BULL.
   Se notícias FED/FOMC_DOVISH dominam → SIDEWAYS no máximo.

2. SATURAÇÃO → se mesmo cluster aparece 3+ vezes nas últimas 4h
   → downgrade para SIDEWAYS (mercado já precificou eventos repetidos).

3. REGIME HMM → usar como contexto de fundo, NÃO replicar automaticamente.
   Se as notícias e o fluxo divergirem do regime atual, classifique a divergência.
   Exemplo: HMM=Sideways mas Fed hawkish forte + OI subindo → classificar BEAR.

4. DATA QUALITY → se N artigos relevantes (ex-OTHER/UNCATEGORIZED) < 3:
   - data_quality = "low"
   - confidence <= 0.4 (hard cap)
   - regime_hint deve ser "SIDEWAYS" salvo evidência muito clara (ex: Fed decision day)

5. HORIZONTE → classificar para as próximas 4h.
   Use edge_4h como métrica primária de decisão.
   edge_24h é contexto de tendência apenas — não determina a classificação sozinho.

6. FLUXO > HEADLINE — graduação obrigatória:

   FLUXO COMPLETO (3 de 3 condições) → fluxo domina:
     OI_z < 0 (desalavancando)
     AND taker_z > 0 (comprando)
     AND ret_4h > 0 (preço confirmando)
     → classificar SIDEWAYS ou BULL
     → confidence += 0.1 (bônus convergência)
     → reasoning: "mercado absorveu headline, fluxo domina"

   FLUXO PARCIAL (2 de 3 condições) → absorção sem confirmação:
     OI_z < 0 AND taker_z > 0 AND ret_4h <= 0
     → classificar SIDEWAYS com viés BEAR
     → confidence -= 0.1 (incerteza maior)
     → reasoning: "absorção tentando, sem confirmação de preço"
     → NÃO usar "neutral" — descrever o setup real

   FLUXO BEARISH COMPLETO (3 de 3):
     OI_z > 1.5 (sobreaquecido)
     AND taker_z < 0 (vendendo)
     AND ret_4h < -0.5%
     → classificar SIDEWAYS ou BEAR
     → reasoning: "headline positivo mas posicionamento deteriora"

   CONVERGÊNCIA (notícia e fluxo alinhados):
     → confidence += 0.1 (não exceder 1.0)

7. FUNDING_Z — contexto de posicionamento do crowd:

   funding_z < -1.0 → crowd está SHORT (pagando para shortar)
     → setup de squeeze potencial
     → sozinho não muda classificação
     → combinado com taker_z > 0 E OI_z < 0:
       reforça viés BULL/SIDEWAYS (shorts vão ser forçados a cobrir)
     → reasoning deve mencionar: "crowd short, squeeze risk"

   funding_z > 1.0 → crowd está LONG (pagando para comprar)
     → mercado sobreaquecido no lado comprador
     → combinado com OI_z > 1.5: reforça viés BEAR
     → reasoning deve mencionar: "longs sobrecarregados"

   funding_z entre -1.0 e +1.0 → neutro, ignorar

8. TAKER_Z — escala de intensidade obrigatória:

   taker_z > 1.5        → compra FORTE — mencionar explicitamente
   taker_z 0.3 a 1.5   → compra MODERADA — NÃO usar "neutral"
   taker_z -0.3 a 0.3  → neutro de fato
   taker_z -0.3 a -1.5 → venda MODERADA — NÃO usar "neutral"
   taker_z < -1.5       → venda FORTE — mencionar explicitamente

   PROIBIDO: usar a palavra "neutral" para taker_z fora do range -0.3 a +0.3.
   Se taker_z = +0.41 → descrever como "compra moderada".
   Se taker_z = -0.80 → descrever como "venda moderada"."""

    return f"{bloco_a}\n\n{bloco_b}\n\n{bloco_c}"


# ---------------------------------------------------------------------------
# DeepSeek-R1 call
# ---------------------------------------------------------------------------

def _call_r1(prompt: str, api_key: str) -> dict:
    payload = {
        "model": MODEL_R1,
        "max_tokens": 512,
        "temperature": 0,
        "messages": [{"role": "user", "content": prompt}],
    }
    resp = requests.post(
        DEEPSEEK_URL,
        json=payload,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        timeout=90,
    )
    resp.raise_for_status()
    msg = resp.json()["choices"][0]["message"]

    content   = msg.get("content", "").strip()
    reasoning = msg.get("reasoning_content", "").strip()

    def _extract_json(text: str) -> dict | None:
        """Try direct parse first, then regex for last {...} block."""
        clean = text.replace("```json", "").replace("```", "").strip()
        try:
            return json.loads(clean)
        except json.JSONDecodeError:
            pass
        # Last JSON object wins (R1 tends to emit it at the end of reasoning)
        matches = list(re.finditer(r'\{[^{}]+\}', clean, re.DOTALL))
        for m in reversed(matches):
            try:
                return json.loads(m.group())
            except json.JSONDecodeError:
                continue
        return None

    for source in (content, reasoning):
        if source:
            result = _extract_json(source)
            if result is not None:
                return result

    raise ValueError(f"Cannot parse R1 response — content={content[:100]!r}")


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def _save_result(result: dict) -> None:
    out_path = Path("data/02_features/news_regime.parquet")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    row = pd.DataFrame([result])
    if out_path.exists():
        existing = pd.read_parquet(out_path)
        combined = pd.concat([existing, row], ignore_index=True).tail(RETENTION)
    else:
        combined = row
    combined.to_parquet(out_path, index=False)
    logger.info(
        f"news_regime saved: regime={result['regime_hint']} "
        f"conf={result['confidence']:.2f} n={result['n_articles']} → {out_path}"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(dry_run: bool = False) -> dict | None:
    ctx      = _load_market_context()
    articles = _load_recent_articles()
    prompt   = build_prompt(ctx, articles)

    cluster_counts = Counter(a["event_cluster"] for a in articles)
    n_relevant = sum(cnt for cl, cnt in cluster_counts.items() if cl != "OTHER/UNCATEGORIZED")
    dominant = cluster_counts.most_common(1)[0][0] if cluster_counts else "OTHER/UNCATEGORIZED"

    if dry_run:
        print("\n" + "═" * 70)
        print("DRY-RUN — PROMPT QUE SERIA ENVIADO AO DeepSeek-R1:")
        print("═" * 70)
        print(prompt)
        print("═" * 70)
        print(f"\nN artigos relevantes: {n_relevant}")
        print(f"Cluster dominante:    {dominant}")
        print(f"Modelo:               {MODEL_R1}")
        print("(API NÃO chamada, nada salvo)")
        return None

    try:
        api_key = get_credential("deepseek_api_key")
    except Exception as e:
        logger.error(f"DeepSeek key missing: {e}")
        return None

    try:
        classification = _call_r1(prompt, api_key)
    except Exception as e:
        logger.error(f"R1 call failed: {e}")
        return None

    regime_hint = str(classification.get("regime_hint", "SIDEWAYS")).upper()
    confidence  = float(classification.get("confidence", 0.4))
    reasoning   = str(classification.get("reasoning", ""))[:120]

    if regime_hint not in ("BULL", "SIDEWAYS", "BEAR"):
        regime_hint = "SIDEWAYS"

    result = {
        "timestamp":        pd.Timestamp.now(tz="UTC"),
        "regime_hint":      regime_hint,
        "confidence":       round(confidence, 3),
        "reasoning":        reasoning,
        "n_articles":       len(articles),
        "n_relevant":       n_relevant,
        "dominant_cluster": dominant,
    }

    logger.info(
        f"R1 → regime={regime_hint} conf={confidence:.2f} | {reasoning}"
    )
    _save_result(result)
    return result


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S UTC",
    )
    dry_run = "--dry-run" in sys.argv
    result = run(dry_run=dry_run)
    if result and not dry_run:
        print(f"\nregime_hint:      {result['regime_hint']}")
        print(f"confidence:       {result['confidence']:.2f}")
        print(f"reasoning:        {result['reasoning']}")
        print(f"n_articles:       {result['n_articles']} ({result['n_relevant']} relevantes)")
        print(f"dominant_cluster: {result['dominant_cluster']}")
