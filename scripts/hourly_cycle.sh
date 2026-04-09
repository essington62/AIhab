#!/usr/bin/env bash
# Hourly cycle: ingest → clean → features → scoring → paper trade
# Cron: 5 * * * * /Users/brown/Documents/MLGeral/btc_AI/scripts/hourly_cycle.sh

set -euo pipefail
cd /Users/brown/Documents/MLGeral/btc_AI

LOG_DIR="logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/hourly_$(date -u +%Y%m%d_%H%M%S).log"

exec >> "$LOG_FILE" 2>&1
echo "=== Hourly cycle start: $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="

# 1. Data ingestion
python -m src.data.binance_spot
python -m src.data.coinglass_futures    # cross-exchange OI/funding/taker (replaces binance_futures)
python -m src.data.news_ingest
python -m src.data.news_classify   # DeepSeek classify → news_scores.parquet

# 2. Clean raw → intermediate
python -m src.data.clean

# 3. Features
python -m src.features.technical
python -m src.features.gate_features
# fed_sentinel runs inline in paper_trader (no parquet output needed)

# 4. Trading decision
python -m src.trading.paper_trader

echo "=== Hourly cycle done: $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
