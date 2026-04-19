#!/usr/bin/env python3
"""
Bootstrap L/S historical data via CoinGlass (one-shot).

Strategy:
  - Bootstrap: CoinGlass (deep history, up to ~365 days)
  - Incremental: Binance (free, adds new data every cycle)
  Both sources converge in the same parquet (dedup by timestamp).

Usage:
    python scripts/bootstrap_ls_coinglass.py --symbols ETH
    python scripts/bootstrap_ls_coinglass.py --symbols BTC,ETH
    python scripts/bootstrap_ls_coinglass.py --symbols ETH --days 365

WARNING: Run ONCE per symbol — uses CoinGlass API quota.
If endpoint returns error, it may not be available on Hobbyist plan.
"""
import argparse
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S UTC",
)
logger = logging.getLogger("bootstrap_ls")


def main():
    parser = argparse.ArgumentParser(description="Bootstrap L/S ratio via CoinGlass")
    parser.add_argument(
        "--symbols", default="ETH",
        help="Comma-separated symbols (default: ETH)",
    )
    parser.add_argument(
        "--days", type=int, default=180,
        help="Days of history to fetch (default: 180)",
    )
    args = parser.parse_args()

    symbols = [s.strip().upper() for s in args.symbols.split(",")]
    start_time = datetime.now(timezone.utc) - timedelta(days=args.days)
    start_time = start_time.replace(hour=0, minute=0, second=0, microsecond=0)

    logger.info("=" * 60)
    logger.info("Bootstrap L/S Ratio via CoinGlass")
    logger.info(f"Symbols  : {symbols}")
    logger.info(f"Period   : {start_time.strftime('%Y-%m-%d')} → now ({args.days} days)")
    logger.info("WARNING  : Run ONCE per symbol — consumes CoinGlass API quota.")
    logger.info("=" * 60)

    from src.data.coinglass_ls import bootstrap_ls_to_parquet

    results: dict = {}
    for sym in symbols:
        logger.info(f"\n── {sym} ──")
        try:
            r = bootstrap_ls_to_parquet(sym, start_time=start_time)
            results[sym] = r
        except Exception as e:
            logger.error(f"Failed for {sym}: {e}", exc_info=True)
            results[sym] = {"accounts": {"error": str(e)}, "positions": {"error": str(e)}}

    logger.info("\n" + "=" * 60)
    logger.info("Summary:")
    for sym, r in results.items():
        for key in ["accounts", "positions"]:
            info = r.get(key, {})
            if "rows" in info:
                logger.info(f"  {sym} {key}: ✅ {info['rows']} rows → {Path(info['path']).name}")
            else:
                logger.info(f"  {sym} {key}: ❌ {info.get('error', 'unknown error')}")
    logger.info("")
    logger.info("After bootstrap: incremental updates use Binance (free, no quota).")
    logger.info("Verify: python scripts/check_eth_data_coverage.py")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
