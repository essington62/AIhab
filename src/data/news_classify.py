"""
news_classify.py — DeepSeek classifier for BTC + Macro news.

Pipeline:
  1. Relevance filter (MACRO only, before DeepSeek — saves tokens)
  2. Irrelevant → ds_classified=True, ds_score=0, ds_topic=noise
  3. Relevant → DeepSeek batch [CRYPTO]/[MACRO]/[FED] tags
  4. Source-weighted aggregation → news_scores.parquet

Inputs:
    data/01_raw/news/crypto_news.parquet
    data/01_raw/news/macro_news.parquet
    data/01_raw/news/fed_news.parquet
    conf/macro_relevance.json

Outputs:
    data/01_raw/news/crypto_news.parquet   (+ ds_* columns in-place)
    data/01_raw/news/macro_news.parquet    (+ ds_* columns in-place)
    data/01_raw/news/fed_news.parquet      (+ ds_* columns in-place)
    data/02_features/news_scores.parquet   (timestamp, crypto_score, macro_score, combined_score)

API: conf/credentials.yml → deepseek_api_key
"""

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

from src.config import get_credential, get_path

logger = logging.getLogger("data_layer.news_classify")

DEEPSEEK_URL    = "https://api.deepseek.com/chat/completions"
CLASSIFY_WINDOW = 6     # hours — window of news to classify
MAX_ITEMS       = 50    # max items per DeepSeek call
RETENTION_ROWS  = 8760  # rows cap for news_scores

RELEVANCE_CFG = Path("conf/macro_relevance.json")

DS_COLS = ["ds_classified", "ds_topic", "ds_regime", "ds_impact",
           "ds_score", "ds_reason", "ds_classified_at"]

SOURCE_WEIGHTS: dict[str, float] = {
    "reuters": 1.0, "bloomberg": 1.0, "wall street journal": 0.9,
    "wsj": 0.9, "financial times": 0.9, "ft": 0.9,
    "new york times": 0.9, "nyt": 0.9, "associated press": 0.9,
    "ap news": 0.9, "cnbc": 0.8, "bbc": 0.8,
    "al jazeera": 0.8, "cnn": 0.7, "yahoo finance": 0.6,
    "coindesk": 0.7, "cointelegraph": 0.6, "cryptocompare": 0.6,
    "federal reserve": 1.0,
}
DEFAULT_WEIGHT = 0.5

CLASSIFY_PROMPT = """You are a senior Bitcoin trading analyst.

Classify each news item by its REAL impact on Bitcoin price using 3 regimes:

BULL (score +3 to +10): Clearly positive for BTC price.
  Examples: ceasefire/peace, rate cut, ETF approval, institutional buying, adoption.

SIDEWAYS (score -2 to +2): Mixed, uncertain, contradictory, or already priced in.
  Rules:
  - Both positive AND negative signals in one article → SIDEWAYS
  - BTC rising despite negative news → market priced it → SIDEWAYS
  - Profit taking after rally → SIDEWAYS +1 (healthy, not bearish)

BEAR (score -3 to -10): Clearly negative for BTC price.
  Examples: war escalation, rate hike, hack, ban, forced sell-off.

IMPORTANT:
- [MACRO] articles: classify by INDIRECT impact via causal chain.
  Oil surge → inflation → rates up → risk-off → BTC down (BEAR)
  Ceasefire → risk-on → BTC up (BULL)
  Fed dovish → liquidity → BTC up (BULL)
- [FED] articles: classify by Fed tone (hawkish=BEAR, dovish=BULL).
- impact HIGH if changes macro narrative (war/peace, Fed, national regulation).
- impact MEDIUM if significant but doesn't change narrative.
- impact LOW if opinion, analysis, or generic.

Respond ONLY in valid JSON, no markdown:
[
  {{
    "index": 0,
    "regime": "BULL|SIDEWAYS|BEAR",
    "impact": "HIGH|MEDIUM|LOW",
    "score": -10 to +10,
    "topic": "geopolitical_war|trump_policy|fed_monetary|institutional_btc|oil_energy|regulatory|market_stress|mining|noise",
    "reason": "max 8 words"
  }}
]

NEWS:
{news_list}
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_relevance() -> dict:
    if RELEVANCE_CFG.exists():
        with open(RELEVANCE_CFG) as f:
            return json.load(f)
    return {"high_priority": [], "medium_priority": [], "negative_combos": {}}


def _classify_relevance(title: str, relevance: dict) -> str:
    t = title.lower()
    for kw in relevance.get("high_priority", []):
        if kw.lower() in t:
            return "high"
    for kw in relevance.get("medium_priority", []):
        if kw.lower() in t:
            return "medium"
    return "low"


def _combo_adjustment(title: str, relevance: dict) -> float:
    t = title.lower()
    adj = 0.0
    for name, combo in relevance.get("negative_combos", {}).items():
        if name.startswith("_"):
            continue
        kws = combo.get("keywords", [])
        min_match = combo.get("min_match", 1)
        if sum(1 for kw in kws if kw.lower() in t) >= min_match:
            adj += combo.get("score_adjustment", 0.0)
    return adj


def _source_weight(source: str) -> float:
    s = source.lower()
    for key, w in SOURCE_WEIGHTS.items():
        if key in s:
            return w
    return DEFAULT_WEIGHT


def _derive_regime(score: float) -> str:
    if score >= 3:
        return "BULL"
    if score <= -3:
        return "BEAR"
    return "SIDEWAYS"


def _load_parquet(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    # Ensure timestamp is tz-aware UTC
    if "timestamp" in df.columns and df["timestamp"].dt.tz is None:
        df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")
    # Ensure ds_* cols exist
    for col in DS_COLS:
        if col not in df.columns:
            df[col] = None if col != "ds_classified" else False
    # Ensure title_hash
    if "title_hash" not in df.columns:
        import hashlib
        df["title_hash"] = df["title"].apply(
            lambda t: hashlib.sha1(re.sub(r'[^a-z0-9 ]', '', str(t).lower()).strip().encode()).hexdigest()
        )
    return df


def _call_deepseek(items: list[dict], api_key: str) -> list[dict]:
    news_list = "\n".join(
        f"[{n['index']}] [{n['news_type'].upper()}] {n['title']}"
        for n in items
    )
    payload = {
        "model": "deepseek-chat",
        "max_tokens": 2000,
        "temperature": 0.1,
        "messages": [{"role": "user", "content": CLASSIFY_PROMPT.format(news_list=news_list)}],
    }
    resp = requests.post(
        DEEPSEEK_URL,
        json=payload,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        timeout=60,
    )
    resp.raise_for_status()
    raw = resp.json()["choices"][0]["message"]["content"]
    content = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`").strip()

    # Parse JSON robustly
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass
    m = re.search(r'\[.*\]', content, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass
    # Object-level fallback
    items_out = []
    for m in re.finditer(r'\{[^{}]+\}', content, re.DOTALL):
        try:
            obj = json.loads(m.group())
            if "index" in obj and "score" in obj:
                items_out.append(obj)
        except json.JSONDecodeError:
            continue
    if items_out:
        logger.warning(f"JSON recovery: extracted {len(items_out)} items")
        return items_out
    raise ValueError(f"Cannot parse DeepSeek response: {content[:200]}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run() -> None:
    try:
        api_key = get_credential("deepseek_api_key")
    except Exception as e:
        logger.error(f"DeepSeek key missing: {e}")
        return

    relevance = _load_relevance()
    classified_at = datetime.now(timezone.utc).isoformat()
    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=CLASSIFY_WINDOW)

    # Load parquets
    crypto_path = Path("data/01_raw/news/crypto_news.parquet")
    macro_path  = Path("data/01_raw/news/macro_news.parquet")
    fed_path    = Path("data/01_raw/news/fed_news.parquet")

    crypto_df = _load_parquet(crypto_path)
    macro_df  = _load_parquet(macro_path)
    fed_df    = _load_parquet(fed_path)

    if crypto_df is None:
        logger.warning("crypto_news.parquet not found — run news_ingest first")
        return

    # Global dedup: remove from macro/fed what's already in crypto
    crypto_hashes = set(crypto_df["title_hash"])
    if macro_df is not None and not macro_df.empty:
        macro_df = macro_df[~macro_df["title_hash"].isin(crypto_hashes)].copy()
    if fed_df is not None and not fed_df.empty:
        fed_df = fed_df[~fed_df["title_hash"].isin(crypto_hashes)].copy()

    # Pre-classify low-priority macro as NOISE (no API cost)
    noise_count = 0
    if macro_df is not None and not macro_df.empty:
        unclassified = (
            (macro_df["timestamp"] >= cutoff)
            & (macro_df["ds_classified"] != True)  # noqa: E712
        )
        for idx in macro_df[unclassified].index:
            title = str(macro_df.at[idx, "title"])
            if _classify_relevance(title, relevance) == "low":
                macro_df.at[idx, "ds_classified"]   = True
                macro_df.at[idx, "ds_score"]         = 0.0
                macro_df.at[idx, "ds_topic"]         = "noise"
                macro_df.at[idx, "ds_impact"]        = "LOW"
                macro_df.at[idx, "ds_regime"]        = "SIDEWAYS"
                macro_df.at[idx, "ds_reason"]        = "irrelevant"
                macro_df.at[idx, "ds_classified_at"] = classified_at
                noise_count += 1
    if noise_count:
        logger.info(f"Pre-classified {noise_count} macro articles as NOISE")

    # Collect pending items
    def _pending(df: pd.DataFrame | None, news_type: str) -> list[dict]:
        if df is None or df.empty:
            return []
        mask = (df["timestamp"] >= cutoff) & (df["ds_classified"] != True)  # noqa: E712
        return [
            {"index": -1, "title_hash": row["title_hash"], "title": row["title"], "news_type": news_type}
            for _, row in df[mask].iterrows()
        ]

    crypto_pending = _pending(crypto_df, "crypto")
    macro_high     = [p for p in _pending(macro_df, "macro") if _classify_relevance(p["title"], relevance) == "high"]
    macro_medium   = [p for p in _pending(macro_df, "macro") if _classify_relevance(p["title"], relevance) == "medium"]
    fed_pending    = _pending(fed_df, "fed")

    batch = []
    batch.extend(fed_pending)
    batch.extend(macro_high)
    batch.extend(macro_medium[:10])
    remaining = max(0, MAX_ITEMS - len(batch))
    batch.extend(crypto_pending[:remaining])

    logger.info(
        f"Batch: {len(batch)} items (fed={len(fed_pending)}, "
        f"macro_high={len(macro_high)}, macro_med={min(len(macro_medium),10)}, "
        f"crypto={min(len(crypto_pending),remaining)})"
    )

    if not batch:
        logger.info(f"Nothing to classify in last {CLASSIFY_WINDOW}h")
        _write_news_scores(crypto_df, macro_df)
        return

    for i, item in enumerate(batch):
        item["index"] = i

    # Call DeepSeek
    try:
        classifications = _call_deepseek(batch, api_key)
    except Exception as e:
        logger.error(f"DeepSeek failed: {e}")
        _write_news_scores(crypto_df, macro_df)
        return

    logger.info(f"Received {len(classifications)} classifications")

    cls_map = {int(c.get("index", -1)): c for c in classifications if "index" in c}

    def _apply(df: pd.DataFrame | None, news_type: str) -> pd.DataFrame:
        if df is None or df.empty:
            return df or pd.DataFrame()
        pending_type = [p for p in batch if p["news_type"] == news_type]
        updated = 0
        for item in pending_type:
            cls = cls_map.get(item["index"])
            if cls is None:
                continue
            mask = df["title_hash"] == item["title_hash"]
            if not mask.any():
                continue
            score = float(cls.get("score", 0))
            adj   = _combo_adjustment(item["title"], relevance)
            score += adj
            df.loc[mask, "ds_classified"]   = True
            df.loc[mask, "ds_topic"]        = str(cls.get("topic", "noise"))
            df.loc[mask, "ds_impact"]       = str(cls.get("impact", "LOW"))
            df.loc[mask, "ds_score"]        = score
            df.loc[mask, "ds_regime"]       = _derive_regime(score)
            df.loc[mask, "ds_reason"]       = str(cls.get("reason", ""))[:120]
            df.loc[mask, "ds_classified_at"] = classified_at
            updated += 1
        logger.info(f"[{news_type}] classified {updated} rows")
        return df

    crypto_df = _apply(crypto_df, "crypto")
    macro_df  = _apply(macro_df,  "macro")
    fed_df    = _apply(fed_df,    "fed")

    # Save source parquets with ds_* columns
    crypto_df.to_parquet(crypto_path, index=False)
    if macro_df is not None and not macro_df.empty:
        macro_df.to_parquet(macro_path, index=False)
    if fed_df is not None and not fed_df.empty:
        fed_df.to_parquet(fed_path, index=False)

    _write_news_scores(crypto_df, macro_df)


def _write_news_scores(
    crypto_df: pd.DataFrame | None,
    macro_df: pd.DataFrame | None,
    window_hours: int = 4,
) -> None:
    """Aggregate classified scores → data/02_features/news_scores.parquet."""
    now = pd.Timestamp.now(tz="UTC")
    cutoff = now - pd.Timedelta(hours=window_hours)

    def _agg_score(df: pd.DataFrame | None) -> float:
        if df is None or df.empty or "ds_score" not in df.columns:
            return 0.0
        recent = df[(df.get("timestamp", pd.Series(dtype="object")) >= cutoff) & (df["ds_classified"] == True)]  # noqa: E712
        if recent.empty:
            return 0.0
        recent = recent.copy()
        recent["sw"] = recent["source"].apply(_source_weight)
        recent["ws"] = recent["ds_score"] * recent["sw"]
        return float(recent["ws"].mean())

    crypto_score = _agg_score(crypto_df)
    macro_score  = _agg_score(macro_df)
    combined     = 0.4 * crypto_score + 0.6 * macro_score

    row = pd.DataFrame([{
        "timestamp":     now,
        "crypto_score":  round(crypto_score, 4),
        "macro_score":   round(macro_score,  4),
        "combined_score": round(combined,    4),
    }])

    path = get_path("news_scores")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = pd.read_parquet(path)
        combined_df = pd.concat([existing, row], ignore_index=True).tail(RETENTION_ROWS)
    else:
        combined_df = row

    combined_df.to_parquet(path, index=False)
    logger.info(
        f"news_scores: crypto={crypto_score:+.3f} macro={macro_score:+.3f} combined={combined:+.3f} → {path}"
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S UTC",
    )
    run()
