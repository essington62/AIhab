#!/usr/bin/env bash
# Daily update: macro + sentiment + coinglass + R5C re-fit
# Cron: 0 7 * * * /Users/brown/Documents/MLGeral/btc_AI/scripts/daily_update.sh

set -euo pipefail
cd /Users/brown/Documents/MLGeral/btc_AI

LOG_DIR="logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/daily_$(date -u +%Y%m%d_%H%M%S).log"

exec >> "$LOG_FILE" 2>&1
echo "=== Daily update start: $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="

# 1. Daily data sources
python -m src.data.fred_ingest
python -m src.data.coinglass_ingest
python -m src.data.altme_ingest
python -m src.data.market_context

# 2. Daily candles (for R5C)
python -m src.data.binance_spot  # 1h candles (clean.py aggregates to 1d)

# 3. Clean (rebuild intermediates with fresh daily data)
python -m src.data.clean

# 4. R5C HMM daily re-fit (uses closed daily candle)
python -m src.models.r5c_hmm

# 5. Recompute features with new daily z-scores
python -m src.features.gate_features

echo "=== Daily update done: $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
