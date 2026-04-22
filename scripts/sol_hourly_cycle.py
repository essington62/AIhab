#!/usr/bin/env python3
"""SOL Bot 4 hourly cycle — runs at :15 of each hour via cron."""
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.trading.sol_bot4 import run_hourly_cycle


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    try:
        run_hourly_cycle()
    except Exception as e:
        logging.error(f"SOL cycle failed: {e}", exc_info=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
