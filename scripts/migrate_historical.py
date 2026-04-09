"""
scripts/migrate_historical.py — One-time migration from crypto-market-state.

Migrates:
  CoinGlass bubble_index      → data/01_raw/coinglass/bubble_index_daily.parquet
  CoinGlass stablecoin_mcap   → data/01_raw/coinglass/stablecoin_mcap_daily.parquet
  CoinGlass ETF flows         → data/01_raw/coinglass/etf_flows_daily.parquet
  FRED DGS10, DGS2, RRPONTSYD → data/01_raw/macro/fred_daily.parquet
  R5C HMM model               → data/03_models/r5c_hmm.pkl

Column mappings:
  bubble_index.parquet:    index(date) + bubble_index → timestamp, bubble_index
  stablecoin_mcap.parquet: index(date) + stablecoin_mcap_usd → timestamp, stablecoin_mcap_usd
  etf_total.parquet:       index(date) + flow_usd → timestamp, etf_flow_usd
  DGS10/DGS2/RRP:          date + value → timestamp, dgs10/dgs2/rrp
"""

import shutil
from pathlib import Path

import pandas as pd

OLD = Path("/Users/brown/Documents/MLGeral/crypto_v2/crypto-market-state/data")
NEW = Path("/Users/brown/Documents/MLGeral/btc_AI/data")


def save(df: pd.DataFrame, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"])
    df.to_parquet(dest, index=False)
    print(f"  OK  {dest.relative_to(NEW.parent.parent)}: {len(df)} rows, "
          f"{df['timestamp'].min().date()} → {df['timestamp'].max().date()}")


# ---------------------------------------------------------------------------
# CoinGlass — Bubble Index
# ---------------------------------------------------------------------------
src = OLD / "01_raw/derivatives/coinglass/indices/bubble_index.parquet"
dst = NEW / "01_raw/coinglass/bubble_index_daily.parquet"
df = pd.read_parquet(src)
df = df.reset_index().rename(columns={"date": "timestamp"})
df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
df = df[["timestamp", "bubble_index"]].copy()
df["source"] = "coinglass_bubble_migrated"
save(df, dst)

# ---------------------------------------------------------------------------
# CoinGlass — Stablecoin Mcap
# ---------------------------------------------------------------------------
src = OLD / "01_raw/derivatives/coinglass/indices/stablecoin_mcap.parquet"
dst = NEW / "01_raw/coinglass/stablecoin_mcap_daily.parquet"
df = pd.read_parquet(src)
df = df.reset_index().rename(columns={"date": "timestamp"})
df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
df = df[["timestamp", "stablecoin_mcap_usd"]].copy()
df["source"] = "coinglass_stablecoin_migrated"
save(df, dst)

# ---------------------------------------------------------------------------
# CoinGlass — ETF flows
# ---------------------------------------------------------------------------
src = OLD / "01_raw/derivatives/coinglass/etf/BTC_flows_total.parquet"
dst = NEW / "01_raw/coinglass/etf_flows_daily.parquet"
df = pd.read_parquet(src)
df = df.reset_index().rename(columns={"date": "timestamp", "flow_usd": "etf_flow_usd"})
df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
df = df[["timestamp", "etf_flow_usd"]].copy()
df["source"] = "coinglass_etf_migrated"
save(df, dst)

# ---------------------------------------------------------------------------
# FRED — DGS10, DGS2, RRPONTSYD → single parquet
# ---------------------------------------------------------------------------
series = {}
for series_id, col_name in [("DGS10", "dgs10"), ("DGS2", "dgs2"), ("RRPONTSYD", "rrp")]:
    src = OLD / f"01_raw/macro/daily/{series_id}.parquet"
    df = pd.read_parquet(src)
    # Old schema: date + value columns
    df = df.rename(columns={"date": "timestamp", "value": col_name})
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    series[col_name] = df.set_index("timestamp")[col_name]

macro = pd.DataFrame(series)
macro = macro.reset_index()
macro["source"] = "fred_migrated"
dst = NEW / "01_raw/macro/fred_daily.parquet"
save(macro, dst)

# ---------------------------------------------------------------------------
# R5C HMM model
# ---------------------------------------------------------------------------
src = OLD / "05_models/r5c_hmm.pkl"
dst = NEW / "03_models/r5c_hmm.pkl"
dst.parent.mkdir(parents=True, exist_ok=True)
shutil.copy2(src, dst)
print(f"  OK  models/r5c_hmm.pkl copied ({src.stat().st_size} bytes)")

print("\nMigration complete.")
