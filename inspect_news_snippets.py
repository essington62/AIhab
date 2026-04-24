"""
inspect_news_snippets.py — Inspeciona snippets brutos do Google News RSS.

Objetivo: verificar o que realmente vem no campo <description> para
entender se há conteúdo útil para classificação de sentimento.
"""

import html
import json
import re
import time
from pathlib import Path
from urllib.parse import quote_plus

import feedparser

ROOT = Path(__file__).resolve().parent
KEYWORDS_PATH = ROOT / "conf" / "macro_keywords.json"

# ── Config ────────────────────────────────────────────────────────────────────
MAX_PER_QUERY   = 3     # artigos por query
DELAY_SECONDS   = 1.0   # pausa entre requests
TOP_N           = 10    # quantos artigos finais mostrar
SNIPPET_MAXLEN  = 300   # chars do snippet limpo
RAW_MAXLEN      = 200   # chars do raw HTML


def clean_html(raw: str) -> str:
    """Strip HTML tags, decode entities, colapsar espaços."""
    if not raw:
        return ""
    text = re.sub(r"<[^>]+>", " ", raw)
    text = html.unescape(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def fetch_group(group_name: str, queries: list[str]) -> list[dict]:
    """Busca até MAX_PER_QUERY artigos para cada query do grupo."""
    articles = []
    for query in queries:
        url = (
            "https://news.google.com/rss/search"
            f"?q={quote_plus(query)}&hl=en-US&gl=US&ceid=US:en"
        )
        try:
            feed = feedparser.parse(url)
            entries = feed.entries[:MAX_PER_QUERY]
            for e in entries:
                raw = getattr(e, "summary", "") or ""
                clean = clean_html(raw)
                published = getattr(e, "published", "") or ""
                # parse published para ordenação
                try:
                    pub_ts = feedparser._parse_date(published) if published else None
                    pub_ts = time.mktime(pub_ts) if pub_ts else 0.0
                except Exception:
                    pub_ts = 0.0

                articles.append({
                    "group":        group_name,
                    "query":        query,
                    "title":        getattr(e, "title", ""),
                    "published":    published,
                    "pub_ts":       pub_ts,
                    "link":         getattr(e, "link", ""),
                    "snippet_raw":  raw,
                    "snippet_clean": clean,
                })
        except Exception as exc:
            print(f"  [WARN] Erro ao buscar '{query}': {exc}")
        time.sleep(DELAY_SECONDS)
    return articles


def deduplicate(articles: list[dict]) -> list[dict]:
    seen = set()
    unique = []
    for a in articles:
        key = a["link"] or a["title"]
        if key not in seen:
            seen.add(key)
            unique.append(a)
    return unique


def print_article(n: int, a: dict) -> None:
    raw_preview   = (a["snippet_raw"][:RAW_MAXLEN]   + "…") if len(a["snippet_raw"])   > RAW_MAXLEN   else a["snippet_raw"]
    clean_preview = (a["snippet_clean"][:SNIPPET_MAXLEN] + "…") if len(a["snippet_clean"]) > SNIPPET_MAXLEN else a["snippet_clean"]

    print(f"\n=== ARTIGO {n} ===")
    print(f"Grupo    : {a['group']}")
    print(f"Query    : {a['query']}")
    print(f"Título   : {a['title']}")
    print(f"Data     : {a['published']}")
    print(f"Link     : {a['link']}")
    print(f"Snippet  : {clean_preview or '(vazio)'}")
    print(f"RAW HTML : {raw_preview or '(vazio)'}")
    print("─" * 60)


def main():
    with open(KEYWORDS_PATH) as f:
        cfg = json.load(f)

    groups = cfg["search_groups"]

    # Ordem: fed_monetary primeiro, depois os demais
    order = ["fed_monetary"] + [k for k in groups if k != "fed_monetary"]
    enabled = [k for k in order if groups.get(k, {}).get("enabled", False)]

    print(f"Grupos habilitados: {enabled}")
    print(f"Buscando {MAX_PER_QUERY} artigos por query, delay {DELAY_SECONDS}s...\n")

    all_articles: list[dict] = []
    for group_name in enabled:
        queries = groups[group_name]["queries"]
        print(f"[{group_name}] {len(queries)} queries...")
        found = fetch_group(group_name, queries)
        print(f"  → {len(found)} artigos coletados")
        all_articles.extend(found)

    unique = deduplicate(all_articles)
    unique.sort(key=lambda a: a["pub_ts"], reverse=True)
    top = unique[:TOP_N]

    print(f"\n{'='*60}")
    print(f"TOP {len(top)} ARTIGOS (deduplicado, ordem: mais recente primeiro)")
    print(f"{'='*60}")

    for i, a in enumerate(top, 1):
        print_article(i, a)

    # ── Resumo ─────────────────────────────────────────────────────────────
    n_total      = len(unique)
    n_raw        = sum(1 for a in unique if a["snippet_raw"])
    n_rich       = sum(1 for a in unique if len(a["snippet_clean"]) > 50)

    print(f"\n{'='*60}")
    print(f"RESUMO")
    print(f"{'='*60}")
    print(f"Total artigos únicos  : {n_total}")
    print(f"Com snippet_raw       : {n_raw} / {n_total}")
    print(f"Com snippet rico >50c : {n_rich} / {n_total}")

    if n_raw == 0:
        print("\n⚠️  AVISO: snippet_raw sempre vazio — Google News pode ter mudado o schema RSS.")
        print("   Verifique se o campo <description> ainda existe no feed.")
    elif n_rich < n_raw // 2:
        print("\n⚠️  AVISO: maioria dos snippets é curta (<50 chars). Conteúdo pode ser só título.")
    else:
        print("\n✅ Snippets com conteúdo suficiente para classificação de sentimento.")


if __name__ == "__main__":
    main()
