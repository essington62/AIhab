#!/usr/bin/env python3
"""Checagem rápida de stops — roda a cada 15min via cron."""

import logging
import sys

from src.trading.paper_trader import acquire_lock, check_stops_only, release_lock


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger("check_stops")

    lock = acquire_lock()
    if lock is None:
        logger.info("Another cycle is running — skipping this check")
        return 0

    try:
        result = check_stops_only()
    finally:
        release_lock(lock)

    action = result.get("action")
    if action == "hold":
        logger.info(
            f"HOLD — price=${result['price']:,.0f} trailing_high=${result.get('trailing_high', 0):,.0f}"
        )
    elif action == "exit":
        logger.info(
            f"EXIT by {result['reason']} — price=${result['price']:,.0f} "
            f"return={result.get('return_pct', 0):+.2%}"
        )
    elif action == "no_position":
        logger.info("No open position — nothing to check")
    elif action == "error":
        logger.warning(f"Could not check stops: {result.get('error', 'unknown')}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
