"""
News ingest: Google News RSS (crypto + macro + Fed).
Writes to: data/01_raw/news/
  - crypto_news.parquet
  - macro_news.parquet
  - fed_news.parquet

Schema: timestamp (UTC), title, body, url, source, category, title_hash
Dedup: SHA1(normalize(title)) — robust against source variations.
Retention: 8760 rows (1h window cap).
"""

import hashlib
import logging
import re
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path

import pandas as pd

from .utils import append_and_save

logger = logging.getLogger("data_layer.news")

RAW_DIR = Path("data/01_raw/news")
RETENTION = 8760  # rows

# ---------------------------------------------------------------------------
# Google News RSS feeds
# ---------------------------------------------------------------------------

CRYPTO_RSS_FEEDS = {
    "btc_price":       "https://news.google.com/rss/search?q=bitcoin+BTC+price&hl=en-US&gl=US&ceid=US:en",
    "btc_institutional": "https://news.google.com/rss/search?q=bitcoin+institutional+adoption&hl=en-US&gl=US&ceid=US:en",
    "btc_etf":         "https://news.google.com/rss/search?q=bitcoin+ETF+spot&hl=en-US&gl=US&ceid=US:en",
    "crypto_regulation": "https://news.google.com/rss/search?q=crypto+regulation+SEC+bitcoin&hl=en-US&gl=US&ceid=US:en",
    "btc_onchain":     "https://news.google.com/rss/search?q=bitcoin+whale+on-chain+exchange&hl=en-US&gl=US&ceid=US:en",
}

MACRO_RSS_FEEDS = {
    "energy":      "https://news.google.com/rss/search?q=oil+energy+prices&hl=en-US&gl=US&ceid=US:en",
    "fed":         "https://news.google.com/rss/search?q=Federal+Reserve+interest+rates&hl=en-US&gl=US&ceid=US:en",
    "geopolitical": "https://news.google.com/rss/search?q=geopolitical+risk+war+sanctions&hl=en-US&gl=US&ceid=US:en",
    "inflation":   "https://news.google.com/rss/search?q=inflation+CPI+PPI+US+economy&hl=en-US&gl=US&ceid=US:en",
    "global_risk": "https://news.google.com/rss/search?q=global+recession+financial+crisis&hl=en-US&gl=US&ceid=US:en",
}

FED_RSS_FEEDS = {
    "fed_speeches": "https://www.federalreserve.gov/feeds/speeches.xml",
    "fed_press":    "https://www.federalreserve.gov/feeds/press_all.xml",
    "fomc_google":  "https://news.google.com/rss/search?q=FOMC+rate+decision+Federal+Reserve&hl=en-US&gl=US&ceid=US:en",
    "fed_speakers": "https://news.google.com/rss/search?q=Powell+Warsh+Williams+Fed+speech&hl=en-US&gl=US&ceid=US:en",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize_title(title: str) -> str:
    return re.sub(r'[^a-z0-9 ]', '', title.lower()).strip()


def title_hash(title: str) -> str:
    return hashlib.sha1(_normalize_title(title).encode()).hexdigest()


def _word_overlap(t1: str, t2: str) -> float:
    """Jaccard similarity on lowercase words ≥4 chars (ignores stop words)."""
    w1 = set(re.findall(r'\b[a-z]{4,}\b', t1.lower()))
    w2 = set(re.findall(r'\b[a-z]{4,}\b', t2.lower()))
    if not w1 or not w2:
        return 0.0
    return len(w1 & w2) / len(w1 | w2)


def _fuzzy_dedup_batch(df: pd.DataFrame, threshold: float = 0.70) -> pd.DataFrame:
    """Within-batch dedup: drop articles whose title is >threshold similar to an
    already-accepted article. Keeps the most recent of near-duplicates."""
    if df.empty:
        return df
    df = df.sort_values("timestamp", ascending=False).reset_index(drop=True)
    kept: list[int] = []
    kept_titles: list[str] = []
    for i, row in df.iterrows():
        t = row["title"]
        if any(_word_overlap(t, kt) >= threshold for kt in kept_titles):
            continue
        kept.append(i)
        kept_titles.append(t)
    dropped = len(df) - len(kept)
    if dropped:
        logger.info(f"fuzzy_dedup_batch: removed {dropped} near-duplicate(s)")
    return df.loc[kept].reset_index(drop=True)


def _fuzzy_dedup_against_existing(
    new_df: pd.DataFrame, filepath: Path, threshold: float = 0.70, recent_n: int = 200
) -> pd.DataFrame:
    """Drop new articles that are near-duplicates of recent existing ones."""
    if not filepath.exists() or new_df.empty:
        return new_df
    try:
        existing = pd.read_parquet(filepath, columns=["title"]).tail(recent_n)
        existing_titles = existing["title"].tolist()
        keep = [
            not any(_word_overlap(row["title"], et) >= threshold for et in existing_titles)
            for _, row in new_df.iterrows()
        ]
        dropped = len(new_df) - sum(keep)
        if dropped:
            logger.info(f"{filepath.name}: fuzzy_dedup removed {dropped} cross-batch duplicate(s)")
        return new_df[keep].reset_index(drop=True)
    except Exception as e:
        logger.warning(f"fuzzy_dedup_against_existing failed: {e}")
        return new_df


def _fetch_rss(url: str, max_results: int = 15) -> list[dict]:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        resp = urllib.request.urlopen(req, timeout=15)
        root = ET.fromstring(resp.read())
    except Exception as e:
        logger.warning(f"RSS failed for {url[:60]}: {e}")
        return []

    items = []
    for item in root.findall(".//item")[:max_results]:
        title_el  = item.find("title")
        link_el   = item.find("link")
        pub_el    = item.find("pubDate")
        source_el = item.find("source")

        title    = (title_el.text  or "").strip() if title_el  is not None else ""
        link     = (link_el.text   or "")         if link_el   is not None else ""
        pub_date = (pub_el.text    or "")         if pub_el    is not None else ""
        source   = (source_el.text or "")         if source_el is not None else ""

        if not title:
            continue

        try:
            ts = parsedate_to_datetime(pub_date)
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
        except Exception:
            ts = datetime.now(timezone.utc)

        items.append({
            "timestamp":   pd.Timestamp(ts),
            "title":       title,
            "body":        "",
            "url":         link,
            "source":      source,
            "title_hash":  title_hash(title),
        })

    return items


def _fetch_all(feeds: dict, category: str) -> pd.DataFrame:
    records = []
    for topic, url in feeds.items():
        rows = _fetch_rss(url)
        for r in rows:
            r["category"] = category
            r["topic"] = topic
        records.extend(rows)
        logger.info(f"RSS {topic}: {len(rows)} items")

    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records)
    # Within-batch fuzzy dedup: same story from multiple feeds
    df = _fuzzy_dedup_batch(df, threshold=0.70)
    return df


def _dedup_against_existing(new_df: pd.DataFrame, filepath: Path) -> pd.DataFrame:
    """Remove rows whose title_hash already exist in the parquet."""
    if not filepath.exists() or new_df.empty:
        return new_df
    try:
        existing = pd.read_parquet(filepath, columns=["title_hash"])
        existing_hashes = set(existing["title_hash"].astype(str))
        before = len(new_df)
        new_df = new_df[~new_df["title_hash"].isin(existing_hashes)]
        dropped = before - len(new_df)
        if dropped:
            logger.info(f"{filepath.name}: deduped {dropped} already-seen rows")
    except Exception:
        pass
    return new_df


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

def run() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    targets = [
        ("crypto_news.parquet", CRYPTO_RSS_FEEDS, "crypto"),
        ("macro_news.parquet",  MACRO_RSS_FEEDS,  "macro"),
        ("fed_news.parquet",    FED_RSS_FEEDS,    "fed"),
    ]

    for filename, feeds, category in targets:
        filepath = RAW_DIR / filename
        try:
            df = _fetch_all(feeds, category)
            if df.empty:
                logger.info(f"{filename}: no data fetched")
                continue
            df = _dedup_against_existing(df, filepath)
            df = _fuzzy_dedup_against_existing(df, filepath)
            if df.empty:
                logger.info(f"{filename}: all already seen")
                continue
            append_and_save(df, filepath, freq="1h")
            logger.info(f"{filename}: +{len(df)} new rows")
        except Exception as e:
            logger.error(f"{filename}: {e}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S UTC",
    )
    run()
