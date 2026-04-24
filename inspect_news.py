#!/usr/bin/env python3
"""
inspect_news.py — snapshot completo do pipeline de news em um comando.
Uso: conda run -n btc_trading_v1 python inspect_news.py
"""
import os
import sys
from datetime import datetime, timezone
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))

SEP  = "─" * 62
SEP2 = "═" * 62


def find_ts_col(df):
    if pd.api.types.is_datetime64_any_dtype(df.index):
        return "__index__"
    for col in ["timestamp", "published", "date", "ts", "time"]:
        if col in df.columns:
            return col
    return None


def to_utc_series(series):
    s = pd.to_datetime(series, utc=True, errors="coerce")
    return s


def get_max_ts(df, ts_col):
    if ts_col == "__index__":
        series = to_utc_series(df.index.to_series())
    else:
        series = to_utc_series(df[ts_col])
    return series.max()


def hours_ago(ts):
    now = datetime.now(timezone.utc)
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return (now - ts).total_seconds() / 3600


def trunc(s, n=80):
    s = str(s) if s is not None else ""
    return s[:n] + "…" if len(s) > n else s


# ══════════════════════════════════════════════════════════════
# SEÇÃO 1 — news_scores.parquet
# ══════════════════════════════════════════════════════════════
print(f"\n{SEP2}")
print("  SEÇÃO 1 — data/02_features/news_scores.parquet")
print(SEP2)

ns_path = os.path.join(BASE, "data/02_features/news_scores.parquet")
ns_max_ts = None

if os.path.exists(ns_path):
    ns = pd.read_parquet(ns_path)
    ts_col = find_ts_col(ns)
    print(f"  Shape   : {ns.shape}")
    print(f"  Colunas : {list(ns.columns)}")
    print(f"  TS field: '{ts_col}'")
    print(f"\n  Últimas 6 linhas:")
    print(ns.tail(6).to_string(index=False))
    ns_max_ts = get_max_ts(ns, ts_col)
else:
    print(f"  [ERRO] Arquivo não encontrado: {ns_path}")

# ══════════════════════════════════════════════════════════════
# SEÇÃO 2 — data/01_raw/news/ (artigos brutos)
# ══════════════════════════════════════════════════════════════
print(f"\n{SEP2}")
print("  SEÇÃO 2 — data/01_raw/news/ (artigos brutos)")
print(SEP2)

news_dir = os.path.join(BASE, "data/01_raw/news")
news_summaries = {}  # fname -> max_ts

if not os.path.isdir(news_dir):
    print(f"  [ERRO] Diretório não encontrado: {news_dir}")
else:
    files = sorted(f for f in os.listdir(news_dir) if f.endswith(".parquet"))
    if not files:
        print("  [AVISO] Nenhum .parquet encontrado.")

    for fname in files:
        fpath = os.path.join(news_dir, fname)
        df = pd.read_parquet(fpath)
        ts_col = find_ts_col(df)

        print(f"\n{SEP}")
        print(f"  {fname}")
        print(SEP)
        print(f"  Shape   : {df.shape}")
        print(f"  Colunas : {list(df.columns)}")
        print(f"  TS field: '{ts_col}'")

        # Última linha — campo a campo
        print(f"\n  Última linha:")
        last = df.tail(1).iloc[0]
        for col in df.columns:
            val = last[col]
            print(f"    {col:<25}: {trunc(val, 90)}")

        # Ordena por timestamp para pegar os mais recentes
        if ts_col and ts_col != "__index__" and ts_col in df.columns:
            ts_s = to_utc_series(df[ts_col])
            df_sorted = df.copy()
            df_sorted["_ts"] = ts_s
            df_sorted = df_sorted.sort_values("_ts", ascending=False)
        elif ts_col == "__index__":
            df_sorted = df.copy()
            df_sorted["_ts"] = to_utc_series(df.index.to_series())
            df_sorted = df_sorted.sort_values("_ts", ascending=False)
        else:
            df_sorted = df.copy()
            df_sorted["_ts"] = pd.NaT

        max_ts = get_max_ts(df, ts_col)
        news_summaries[fname] = max_ts

        # 5 artigos mais recentes
        print(f"\n  5 artigos mais recentes:")
        for _, row in df_sorted.head(5).iterrows():
            ts_str = row["_ts"].strftime("%Y-%m-%d %H:%M UTC") if pd.notna(row.get("_ts")) else "?"
            title  = trunc(row.get("title", row.get("headline", "")), 72)
            score  = row.get("ds_score", row.get("score", None))
            score_str = f"  score={score:+.2f}" if score is not None and pd.notna(score) else ""
            print(f"    [{ts_str}]{score_str}")
            print(f"      {title}")

# ══════════════════════════════════════════════════════════════
# SEÇÃO 3 — Resumo de atualização
# ══════════════════════════════════════════════════════════════
print(f"\n{SEP2}")
print("  SEÇÃO 3 — Resumo de atualização das fontes")
print(SEP2)

all_sources = {}
for fname, max_ts in news_summaries.items():
    all_sources[fname] = max_ts
if ns_max_ts is not None:
    all_sources["news_scores.parquet (features)"] = ns_max_ts

for src, max_ts in all_sources.items():
    if pd.isna(max_ts):
        print(f"  {src:<40}  [sem timestamp]")
        continue
    h = hours_ago(max_ts)
    alert = "  ⚠️  ALERTA: > 6h sem atualizar" if h > 6 else ""
    ts_str = max_ts.strftime("%Y-%m-%d %H:%M UTC")
    print(f"  {src:<40}  última: {ts_str}  ({h:.1f}h){alert}")

print()
