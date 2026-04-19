#!/bin/bash
# ETH hourly incremental data collection.
# TEMPLATE — NOT ACTIVE in crontab. Activate after Fase 0 (descritivo) validates ETH.
#
# To activate, add to /etc/cron.d/aihab-cron:
#   7 * * * * /app/scripts/hourly_cycle_eth.sh >> /app/logs/hourly_eth.log 2>&1
# (minute 7 = 2 min after BTC cycle to avoid simultaneous API rate limits)

set -e

if [ -d "/app" ]; then
    cd /app
else
    cd "$(dirname "$0")/.."
fi

echo "=== ETH hourly collection start: $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="

python -c "
from src.data.binance_spot import fetch_spot_1h
from src.data.coinglass_futures import fetch_oi_4h, fetch_funding_4h, fetch_taker_4h
from src.data.binance_ls import fetch_ls_accounts, fetch_ls_positions

SYMBOL = 'ETH'
fetch_spot_1h(symbol=SYMBOL)
fetch_oi_4h(symbol=SYMBOL)
fetch_funding_4h(symbol=SYMBOL)
fetch_taker_4h(symbol=SYMBOL)
fetch_ls_accounts(symbol=SYMBOL)
fetch_ls_positions(symbol=SYMBOL)
"

echo "=== ETH hourly collection done: $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
