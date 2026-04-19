#!/usr/bin/env python3
"""
Bootstrap ETH historical data — one-shot script to populate data lake with ~1 year of ETH data.

Usage:
    python scripts/bootstrap_eth_history.py
    # or on EC2:
    docker exec aihab-app python3 /app/scripts/bootstrap_eth_history.py

Fetches:
- Spot 1h (Binance)             — up to 1 year OHLCV, batched
- Futures OI 4h (CoinGlass)     — last ~180 days (API limit)
- Futures Funding 4h (CoinGlass)— last ~180 days
- Futures Taker 4h (CoinGlass)  — last ~180 days (may fail on Hobbyist plan)
- L/S Account 1h (Binance)      — up to ~3 months (Binance history limit)
- L/S Position 1h (Binance)     — up to ~3 months

Idempotent: each function resumes from last saved timestamp.
Global data (FRED, F&G, stablecoin) is shared with BTC — not collected here.
"""
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
logger = logging.getLogger("bootstrap_eth")

SYMBOL = "ETH"
HISTORY_DAYS = 365
START_DATE = datetime.now(timezone.utc) - timedelta(days=HISTORY_DAYS)
START_DATE = START_DATE.replace(hour=0, minute=0, second=0, microsecond=0)


def bootstrap_spot():
    """1h spot via Binance — full year, batched (~9 calls of 1000 candles)."""
    from src.data.binance_spot import fetch_spot_1h
    logger.info(f"── Spot 1h [{SYMBOL}] from {START_DATE.strftime('%Y-%m-%d')} ──")
    try:
        fetch_spot_1h(symbol=SYMBOL, start_time=START_DATE)
        logger.info(f"✅ {SYMBOL} spot 1h done")
    except Exception as e:
        logger.error(f"❌ {SYMBOL} spot 1h: {e}")


def bootstrap_futures_coinglass():
    """CoinGlass OI/Funding/Taker 4h — limited to last ~180 days (API limit=1080)."""
    from src.data.coinglass_futures import fetch_oi_4h, fetch_funding_4h, fetch_taker_4h
    logger.info(f"── CoinGlass Futures [{SYMBOL}] (max ~180 days) ──")

    for name, fn in [("OI", fetch_oi_4h), ("Funding", fetch_funding_4h), ("Taker", fetch_taker_4h)]:
        try:
            fn(symbol=SYMBOL, start_time=START_DATE)
            logger.info(f"✅ {SYMBOL} {name} 4h done")
        except Exception as e:
            logger.error(f"❌ {SYMBOL} {name} 4h: {e}")


def bootstrap_ls_ratio():
    """Binance L/S accounts/positions 1h — limited to ~3 months (Binance history)."""
    from src.data.binance_ls import fetch_ls_accounts, fetch_ls_positions
    logger.info(f"── L/S Ratios [{SYMBOL}] (max ~3 months Binance history) ──")

    for name, fn in [("L/S accounts", fetch_ls_accounts), ("L/S positions", fetch_ls_positions)]:
        try:
            fn(symbol=SYMBOL, start_time=START_DATE)
            logger.info(f"✅ {SYMBOL} {name} done")
        except Exception as e:
            logger.error(f"❌ {SYMBOL} {name}: {e}")


def main():
    logger.info("=" * 60)
    logger.info(f"Bootstrap {SYMBOL} Historical Data")
    logger.info(f"Target period: {START_DATE.strftime('%Y-%m-%d')} → now ({HISTORY_DAYS} days)")
    logger.info("=" * 60)

    bootstrap_spot()
    bootstrap_futures_coinglass()
    bootstrap_ls_ratio()

    logger.info("")
    logger.info("Note: FRED/F&G/Stablecoin/Bubble/ETF are global — shared with BTC.")
    logger.info("      Run scripts/check_eth_data_coverage.py to verify results.")
    logger.info("=" * 60)
    logger.info("Bootstrap complete")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
