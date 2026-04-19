#!/usr/bin/env python3
"""
Check ETH data coverage after bootstrap.
Usage: python scripts/check_eth_data_coverage.py
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

ETH_FILES = {
    "Spot 1h":        "data/01_raw/spot/eth_1h.parquet",
    "OI 4h":          "data/01_raw/futures/eth_oi_4h.parquet",
    "Funding 4h":     "data/01_raw/futures/eth_funding_4h.parquet",
    "Taker 4h":       "data/01_raw/futures/eth_taker_4h.parquet",
    "L/S Account":    "data/01_raw/futures/eth_ls_account_1h.parquet",
    "L/S Position":   "data/01_raw/futures/eth_ls_position_1h.parquet",
}

GLOBAL_FILES = {
    "FRED Macro":     "data/02_intermediate/macro/fred_daily_clean.parquet",
    "Fear & Greed":   "data/01_raw/sentiment/fear_greed_daily.parquet",
    "Stablecoin":     "data/01_raw/coinglass/stablecoin_mcap_daily.parquet",
    "ETF Flows BTC":  "data/01_raw/coinglass/etf_flows_daily.parquet",
}

BTC_FILES = {
    "BTC Spot 1h":    "data/01_raw/spot/btc_1h.parquet",
    "BTC OI 4h":      "data/01_raw/futures/oi_4h.parquet",
    "BTC Funding 4h": "data/01_raw/futures/funding_4h.parquet",
    "BTC L/S Acct":   "data/01_raw/futures/ls_account_1h.parquet",
}


def _check(files: dict, section: str) -> None:
    print(f"\n{'='*72}")
    print(f" {section}")
    print(f"{'='*72}")
    print(f"{'Dataset':<20} {'Rows':<8} {'Start':<22} {'End':<22} {'Age'}")
    print(f"{'-'*72}")

    now = pd.Timestamp.now(tz="UTC")

    for name, path_rel in files.items():
        path = ROOT / path_rel
        if not path.exists():
            print(f"{name:<20} ❌ MISSING")
            continue
        try:
            df = pd.read_parquet(path)
            n = len(df)
            if "timestamp" in df.columns:
                ts = pd.to_datetime(df["timestamp"], utc=True)
                start = ts.min().strftime("%Y-%m-%d %H:%M")
                end   = ts.max().strftime("%Y-%m-%d %H:%M")
                age_h = (now - ts.max()).total_seconds() / 3600
                age_s = f"{age_h:.0f}h" if age_h < 48 else f"{age_h/24:.0f}d"
            else:
                start = end = age_s = "no ts col"
            print(f"{name:<20} {n:<8} {start:<22} {end:<22} {age_s}")
        except Exception as e:
            print(f"{name:<20} ⚠️  ERROR: {e}")


def main():
    _check(ETH_FILES, "ETH Data (new)")
    _check(BTC_FILES, "BTC Data (must still be current)")
    _check(GLOBAL_FILES, "Global Data (shared)")


if __name__ == "__main__":
    main()
