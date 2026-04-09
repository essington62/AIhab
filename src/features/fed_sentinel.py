"""
src/features/fed_sentinel.py — Fed Sentinel (3 layers).

Layer 1 — Static:   FOMC dates, hearings, transitions, blackout periods
                    from conf/fed_calendar.json
Layer 2 — Dynamic:  Classify fed_news parquet via DeepSeek (hawkish/dovish)
                    with member weights
Layer 3 — Adaptive: Proximity adjustment to scoring threshold

Entry points:
  get_fed_context(today=None) → dict with all output for paper_trader
  compute_fomc_proximity_adjustment(today) → float
  is_in_blackout(today) → bool
  get_next_fed_event(today) → dict
"""

import logging
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

from src.config import get_credential, get_fed_calendar, get_params, get_path

logger = logging.getLogger("features.fed_sentinel")


# ---------------------------------------------------------------------------
# Layer 1: Static calendar
# ---------------------------------------------------------------------------

def _parse_dates(date_strings: list[str]) -> list[date]:
    return [date.fromisoformat(d) for d in date_strings]


def get_next_fed_event(today: Optional[date] = None) -> dict:
    """Return the next upcoming FOMC decision, hearing, or transition."""
    today = today or date.today()
    cal = get_fed_calendar()

    events = []
    for d in cal["fomc_decisions"]:
        events.append({"date": date.fromisoformat(d), "type": "fomc_decision"})
    for h in cal.get("hearings", []):
        events.append({"date": date.fromisoformat(h["date"]), "type": h["type"], "member": h.get("member", "")})
    for t in cal.get("transitions", []):
        events.append({"date": date.fromisoformat(t["date"]), "type": t["type"], "member": t.get("member", "")})

    future = [e for e in events if e["date"] >= today]
    if not future:
        return {"date": None, "type": "none", "days_away": None}

    next_event = min(future, key=lambda e: e["date"])
    days_away = (next_event["date"] - today).days
    return {**next_event, "days_away": days_away}


def is_in_blackout(today: Optional[date] = None) -> bool:
    """True if today is within Fed blackout period (T-N to T-2 before FOMC)."""
    today = today or date.today()
    cal = get_fed_calendar()
    before = cal.get("blackout_days_before", 10)
    ends = cal.get("blackout_ends_days_before", 2)

    for d_str in cal["fomc_decisions"]:
        fomc = date.fromisoformat(d_str)
        blackout_start = fomc - timedelta(days=before)
        blackout_end = fomc - timedelta(days=ends)
        if blackout_start <= today <= blackout_end:
            return True
    return False


def compute_fomc_proximity_adjustment(today: Optional[date] = None) -> float:
    """
    Returns float to ADD to the scoring threshold.
    Higher value = harder to get ENTER signal near FOMC.
    """
    today = today or date.today()
    cal = get_fed_calendar()
    params = get_params()["fed"]["proximity_adjustments"]

    # Check against FOMC decisions
    for d_str in cal["fomc_decisions"]:
        fomc = date.fromisoformat(d_str)
        delta = (fomc - today).days

        if -0 <= delta <= 2:     # T-2 to T0
            return params["fomc_decision_t0_t2"]
        if 3 <= delta <= 5:      # T-5 to T-3
            return params["fomc_decision_t3_t5"]
        if delta == -1:          # T+1 after FOMC
            return params["fomc_post_t1"]

    # Check hearings / transitions
    all_events = cal.get("hearings", []) + cal.get("transitions", [])
    for event in all_events:
        ev_date = date.fromisoformat(event["date"])
        delta = (ev_date - today).days
        if 0 <= delta <= 1:
            return params["hearing_transition_t0_t1"]

    # Blackout
    if is_in_blackout(today):
        return params["blackout"]

    return params["normal"]


# ---------------------------------------------------------------------------
# Layer 2: Dynamic classification (DeepSeek)
# ---------------------------------------------------------------------------

DEEPSEEK_URL = "https://api.deepseek.com/chat/completions"

FED_CLASSIFY_PROMPT = """You are a Fed policy analyst. Classify the following news article about the Federal Reserve.

Article: {title}
{body}

Rate the article on a scale from -3 to +3:
  -3: Extremely hawkish (aggressive rate hikes, tight monetary policy)
  -1 to -2: Hawkish
   0: Neutral / unclear
  +1 to +2: Dovish
  +3: Extremely dovish (rate cuts, accommodation)

Also identify if any specific Fed member is mentioned from this list: {members}

Respond in JSON only:
{{"score": <float>, "member": "<name or null>", "surprise_factor": <0.0-1.0>, "reasoning": "<one sentence>"}}"""


def _classify_article(title: str, body: str, members: list[str]) -> dict:
    """Call DeepSeek to classify a Fed article. Returns dict with score, member, etc."""
    try:
        api_key = get_credential("deepseek_api_key")
    except KeyError:
        logger.warning("deepseek_api_key not set — skipping article classification")
        return {"score": 0.0, "member": None, "surprise_factor": 0.5}

    prompt = FED_CLASSIFY_PROMPT.format(
        title=title[:200],
        body=body[:400],
        members=", ".join(members),
    )
    payload = {
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
        "response_format": {"type": "json_object"},
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    import json, time
    for attempt in range(3):
        try:
            r = requests.post(DEEPSEEK_URL, json=payload, headers=headers, timeout=20)
            r.raise_for_status()
            content = r.json()["choices"][0]["message"]["content"]
            return json.loads(content)
        except Exception as e:
            wait = 2 ** attempt
            logger.warning(f"DeepSeek attempt {attempt+1}/3: {e}. Retry {wait}s")
            if attempt < 2:
                time.sleep(wait)

    return {"score": 0.0, "member": None, "surprise_factor": 0.5}


def classify_fed_news(lookback_hours: int = 4) -> dict:
    """
    Load recent fed_news, classify via DeepSeek, apply member weights.
    Returns: {fed_score, n_articles, sentiment_label, articles_classified}
    """
    cal = get_fed_calendar()
    member_weights: dict = cal["member_weights"]
    members = list(member_weights.keys())

    path = get_path("news_fed")
    if not path.exists():
        logger.warning("fed_news parquet not found")
        return {"fed_score": 0.0, "n_articles": 0, "sentiment_label": "neutral"}

    df = pd.read_parquet(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    cutoff = pd.Timestamp.utcnow() - pd.Timedelta(hours=lookback_hours)
    recent = df[df["timestamp"] >= cutoff].copy()

    if recent.empty:
        logger.info(f"classify_fed_news: no articles in last {lookback_hours}h")
        return {"fed_score": 0.0, "n_articles": 0, "sentiment_label": "neutral"}

    scores = []
    for _, row in recent.iterrows():
        result = _classify_article(row.get("title", ""), row.get("body", ""), members)
        raw_score = float(result.get("score", 0.0))
        surprise = float(result.get("surprise_factor", 0.5))
        member = result.get("member")
        weight = member_weights.get(member, 0.3) if member else 0.3
        weighted = raw_score * (1 + surprise * 0.5) * weight
        scores.append(weighted)

    if not scores:
        return {"fed_score": 0.0, "n_articles": 0, "sentiment_label": "neutral"}

    fed_score = sum(scores) / len(scores)
    label = "hawkish" if fed_score < -0.5 else "dovish" if fed_score > 0.5 else "neutral"

    logger.info(f"classify_fed_news: {len(scores)} articles, fed_score={fed_score:.2f}, {label}")
    return {
        "fed_score": round(fed_score, 3),
        "n_articles": len(scores),
        "sentiment_label": label,
    }


# ---------------------------------------------------------------------------
# Layer 3: Unified context (used by paper_trader)
# ---------------------------------------------------------------------------

def get_fed_context(today: Optional[date] = None) -> dict:
    """
    Full Fed Sentinel output for one cycle.
    Returns dict used directly by gate_scoring and paper_trader.
    """
    today = today or date.today()
    params = get_params()

    proximity_adj = compute_fomc_proximity_adjustment(today)
    blackout = is_in_blackout(today)
    next_event = get_next_fed_event(today)
    lookback_h = params["news"]["lookback_hours"]
    news_result = classify_fed_news(lookback_hours=lookback_h)

    fed_score = news_result["fed_score"]

    # Kill switch check
    kill_threshold = params["fed"]["g2_fed_kill_threshold"]
    fomc_kill = (
        fed_score < kill_threshold
        and next_event.get("days_away") is not None
        and next_event["days_away"] <= 2
        and next_event["type"] == "fomc_decision"
    )

    return {
        "fed_score": fed_score,
        "sentiment_label": news_result["sentiment_label"],
        "n_articles": news_result["n_articles"],
        "proximity_adjustment": proximity_adj,
        "is_blackout": blackout,
        "next_event": next_event,
        "fomc_kill_switch": fomc_kill,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    import json
    ctx = get_fed_context()
    print(json.dumps({k: str(v) if not isinstance(v, (int, float, bool, str, dict, type(None))) else v
                      for k, v in ctx.items()}, indent=2))
