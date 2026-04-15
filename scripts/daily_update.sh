#!/usr/bin/env bash
# Daily update: macro + sentiment + coinglass + R5C re-fit
# Disparado por: supercronic (container) ou cron local (dev)

set -euo pipefail

# Detecta se está no container (/app) ou no Mac (repo local)
if [ -d "/app" ]; then
    cd /app
else
    cd "$(dirname "$0")/.."
fi

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

# 2. Daily candles (para R5C)
python -m src.data.binance_spot

# 3. Clean (rebuild intermediates com dados diários frescos)
python -m src.data.clean

# 4. R5C HMM daily re-fit (usa candle diário fechado)
python -m src.models.r5c_hmm

# 5. Recompute features com novos z-scores diários
python -m src.features.gate_features

echo "=== Daily update done: $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
