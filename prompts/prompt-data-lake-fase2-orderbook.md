# Task: Data Lake Fase 2 — Coleta de Order Book (6 exchanges)

## Contexto

Projeto btc-data-lake em `~/Documents/MLGeral/btc-data-lake/`.
Fase 1 completa: funding rates de 6 exchanges (Binance, OKX, Bybit, Gate.io, Bitget, KuCoin) coletando a cada 8h via cron na EC2.
Spec geral do projeto em `CLAUDE.md` na raiz do repo.
Assessment com endpoints pesquisados em `~/Documents/MLGeral/btc_AI/prompts/assessment-data-lake-6-exchanges.md`.

Objetivo da Fase 2: coletar order book snapshots das 6 exchanges para calcular spread bid/ask e slippage estimado — dados críticos para viabilidade da arbitragem (custo real de execução).

## O que implementar

### 1. Adicionar endpoints de order book ao conf/exchanges.yml

Para cada exchange, adicionar seção `order_book` nos endpoints. Referência do assessment:

```yaml
# Binance
order_book:
  path: "/fapi/v1/depth"
  method: GET
  params:
    symbol: "{symbol}"
    limit: 10           # top 10 níveis bid + ask
  response_mapping:
    bids: "bids"        # [[price, qty], ...]
    asks: "asks"        # [[price, qty], ...]
    timestamp: "T"      # ou usar server time

# OKX
order_book:
  path: "/api/v5/market/books"
  method: GET
  params:
    instId: "{symbol}"
    sz: "10"
  response_mapping:
    bids: "data[0].bids"
    asks: "data[0].asks"
    timestamp: "data[0].ts"

# Bybit
order_book:
  path: "/v5/market/orderbook"
  method: GET
  params:
    category: "linear"
    symbol: "{symbol}"
    limit: 10
  response_mapping:
    bids: "result.b"
    asks: "result.a"
    timestamp: "result.ts"

# Gate.io
order_book:
  path: "/api/v4/futures/usdt/order_book"
  method: GET
  params:
    contract: "{symbol}"
    limit: 10
  response_mapping:
    bids: "bids"
    asks: "asks"

# Bitget
order_book:
  path: "/api/v2/mix/market/depth"
  method: GET
  params:
    symbol: "{symbol}"
    productType: "usdt-futures"
    limit: "10"
  response_mapping:
    bids: "data.bids"
    asks: "data.asks"
    timestamp: "data.ts"

# KuCoin
order_book:
  path: "/api/v1/level2/depth20"
  method: GET
  params:
    symbol: "{symbol}"
  response_mapping:
    bids: "data.bids"
    asks: "data.asks"
    timestamp: "data.ts"
```

**IMPORTANTE:** Estes mappings são baseados na pesquisa do assessment. DEVEM ser validados com curl manual antes de implementar o parsing. Cada exchange tem formato diferente para o response JSON.

### 2. Adicionar conf/collection.yml — seção order_book

```yaml
collection:
  # ... funding_rate existente ...
  order_book:
    frequency: "1h"
    cron: "2 * * * *"            # minuto 2 de cada hora
    depth: 10                     # top 10 níveis
    retry_attempts: 3
    retry_delay_seconds: 10
    timeout_seconds: 15           # order book é rápido
```

### 3. Adicionar método ao BaseCollector

Em `src/collectors/base.py`, adicionar método abstrato:

```python
@abstractmethod
def collect_order_book(self) -> dict:
    """Retorna dict com schema normalizado:
    {
        "timestamp": datetime (UTC),
        "exchange": str,
        "symbol": str (normalizado, ex: BTCUSDT),
        "bids": list[list[float, float]],  # [[price, qty], ...] top N
        "asks": list[list[float, float]],  # [[price, qty], ...] top N
        "best_bid": float,                  # bids[0][0]
        "best_ask": float,                  # asks[0][0]
        "spread_pct": float,                # (best_ask - best_bid) / mid_price * 100
        "mid_price": float,                 # (best_bid + best_ask) / 2
        "bid_depth_usd": float,             # sum(price * qty for all bids)
        "ask_depth_usd": float,             # sum(price * qty for all asks)
    }
    """
    pass
```

### 4. Implementar collect_order_book() em cada collector

Para cada exchange (binance.py, okx.py, bybit.py, gateio.py, bitget.py, kucoin.py):

1. Fazer fetch do endpoint `order_book`
2. Parsear bids e asks do response JSON (formato varia por exchange)
3. Converter strings para float (muitas exchanges retornam preço/qty como string)
4. Calcular campos derivados: best_bid, best_ask, spread_pct, mid_price, bid_depth_usd, ask_depth_usd
5. Retornar dict normalizado

**Atenção ao parsing:**
- Binance: `bids` e `asks` são `[["price", "qty"], ...]` (strings)
- OKX: response wrappado em `data[0]`, bids/asks são `[["price", "qty", "0", "numOrders"], ...]`
- Bybit: `result.b` e `result.a`, formato `[["price", "qty"]]`
- Gate.io: pode ter formato diferente dos outros — verificar com curl
- Bitget: wrappado em `data`, verificar se bids/asks são strings ou floats
- KuCoin: `data.bids` e `data.asks`, verificar formato

### 5. Schema normalizado do order book

Em `src/normalizers/schema.py`, adicionar:

```python
ORDER_BOOK_SCHEMA = {
    "timestamp": "datetime64[ns, UTC]",
    "exchange": "string",
    "symbol": "string",
    "best_bid": "float64",
    "best_ask": "float64",
    "spread_pct": "float64",
    "mid_price": "float64",
    "bid_depth_usd": "float64",
    "ask_depth_usd": "float64",
    "bids_json": "string",       # JSON string dos top 10 levels (para análise futura)
    "asks_json": "string",       # JSON string dos top 10 levels
}
```

### 6. Validação em transforms.py

Adicionar `validate_order_book_records()`:

- `best_bid > 0` e `best_ask > 0`
- `best_ask > best_bid` (senão é crossed book, dado corrompido)
- `spread_pct` entre 0% e 1% (se >1% é suspeito, warning)
- `mid_price` entre $10k e $200k (sanity check BTC)
- depth_usd > 0

### 7. Storage — parquet writer

Adicionar `write_order_book()` em `src/storage/parquet_writer.py`:

```python
def write_order_book(records: list[dict], base_path: str = "data/normalized"):
    """Partição: data/normalized/order_book/exchange=binance/date=2026-04-16/data.parquet"""
    # Mesmo pattern do write_funding_rates
    # bids e asks salvos como JSON string (column bids_json, asks_json)
    # Campos numéricos como float64
```

### 8. Script de coleta

Criar `scripts/collect_orderbook.py`:

```python
#!/usr/bin/env python
"""Coleta order book snapshots de todas as exchanges habilitadas."""

import sys
import logging
from src.config import get_exchanges, get_instruments
from src.collectors import COLLECTOR_MAP
from src.normalizers.transforms import validate_order_book_records
from src.storage.parquet_writer import write_order_book

def main():
    logging.basicConfig(...)
    logger = logging.getLogger("collect_orderbook")

    exchanges = get_exchanges()
    instruments = get_instruments()
    all_records = []
    errors = []

    for inst in instruments:
        for name, exc_config in exchanges.items():
            if not exc_config.get("enabled", False):
                continue
            try:
                collector = COLLECTOR_MAP[name](exc_config, inst)
                book = collector.collect_order_book()
                all_records.append(book)
                logger.info(
                    f"✅ {name:8s}: {inst['id']} "
                    f"bid={book['best_bid']:.1f} ask={book['best_ask']:.1f} "
                    f"spread={book['spread_pct']:.4f}% "
                    f"depth=${book['bid_depth_usd']/1000:.0f}k/${book['ask_depth_usd']/1000:.0f}k"
                )
            except Exception as e:
                logger.error(f"❌ {name}: {inst['id']} failed: {e}")
                errors.append({"exchange": name, "error": str(e)})

    # Validar e persistir
    valid = validate_order_book_records(all_records)
    write_order_book(valid)

    # Resumo cross-exchange
    if len(valid) >= 2:
        sorted_by_bid = sorted(valid, key=lambda x: x['best_bid'], reverse=True)
        best_buy  = min(valid, key=lambda x: x['best_ask'])   # comprar mais barato
        best_sell = max(valid, key=lambda x: x['best_bid'])    # vender mais caro

        cross_spread = (best_sell['best_bid'] - best_buy['best_ask']) / best_buy['best_ask'] * 100
        logger.info(
            f"💰 CROSS-EXCHANGE: "
            f"Buy {best_buy['exchange']} ask=${best_buy['best_ask']:.1f} → "
            f"Sell {best_sell['exchange']} bid=${best_sell['best_bid']:.1f} → "
            f"spread={cross_spread:+.4f}%"
        )

        # Slippage estimado para diferentes tamanhos de posição
        for size_usd in [500, 1000, 5000]:
            logger.info(f"  Slippage ${size_usd}: (calcular a partir do book depth)")

    logger.info(f"Done: {len(valid)} records, {len(errors)} errors")
    return 0 if not errors else 1

if __name__ == "__main__":
    sys.exit(main())
```

### 9. Script de análise de slippage

Criar `scripts/show_slippage.py`:

```python
#!/usr/bin/env python
"""Calcula slippage estimado para diferentes tamanhos de posição em cada exchange."""

# Lê o último order book snapshot de cada exchange
# Para cada tamanho ($500, $1000, $5000):
#   - Simula compra: percorre asks até preencher o tamanho
#   - Simula venda: percorre bids até preencher o tamanho
#   - Calcula slippage = preço médio executado vs mid_price
#
# Output:
# Exchange  | Mid Price | Spread  | Slippage $500 | Slippage $1k | Slippage $5k
# Binance   | $74,300   | 0.01%   | 0.005%        | 0.008%       | 0.025%
# OKX       | $74,295   | 0.02%   | 0.007%        | 0.012%       | 0.035%
# ...
#
# Cross-exchange arbitrage viability:
# Buy OKX ($74,295) → Sell Binance ($74,305) = +$10 per BTC
# After fees (0.04% × 2 legs): -$59.4
# Net: -$49.4 → NOT PROFITABLE at current spread
# Break-even spread needed: 0.08% ($59.4)
```

### 10. Atualizar collect_all.sh

```bash
#!/usr/bin/env bash
set -euo pipefail

if [ -d "/app" ]; then
    cd /app
else
    cd "$(dirname "$0")/.."
fi

echo "[$(date -u)] Starting collection cycle..."
python3 scripts/collect_funding.py
python3 scripts/collect_orderbook.py
echo "[$(date -u)] Collection complete."
```

### 11. Testes

Adicionar em `tests/test_collectors.py`:

Para cada exchange:
- Mock do response JSON do order book (fixture)
- `collect_order_book()` retorna dict com todos os campos
- `best_ask > best_bid`
- `spread_pct > 0`
- `bids` e `asks` têm exatamente 10 níveis (ou menos se exchange retornar menos)
- Preços são float, não string

Adicionar `tests/test_slippage.py`:
- Calcula slippage correto em book sintético
- Edge case: book raso (menos de 10 níveis)
- Edge case: order size maior que depth total

### 12. Crontab atualizado

```
# Funding rates (a cada 8h, 10min após settlement)
10 0,8,16 * * * cd /home/ubuntu/Data_Lake_Arbitragem && python3 scripts/collect_funding.py >> logs/collection.log 2>&1

# Order book snapshots (a cada hora, minuto 2)
2 * * * * cd /home/ubuntu/Data_Lake_Arbitragem && python3 scripts/collect_orderbook.py >> logs/orderbook.log 2>&1
```

## Validação

1. Rodar localmente:
   ```bash
   cd ~/Documents/MLGeral/btc-data-lake
   python scripts/collect_orderbook.py
   ```
   Deve mostrar 6 exchanges com bid/ask/spread/depth.

2. Rodar análise de slippage:
   ```bash
   python scripts/show_slippage.py
   ```
   Deve mostrar tabela cross-exchange com viabilidade.

3. Testes:
   ```bash
   pytest tests/ -v
   ```

4. Commit + push:
   ```bash
   git add -A
   git commit -m "feat: phase 2 — order book collection + slippage analysis"
   git push origin master:main
   ```

## Restrições

- **NÃO alterar** a lógica de coleta de funding rates (Fase 1 validada e rodando)
- **NÃO instalar** novas dependências (requests + pandas + pyarrow já cobrem tudo)
- **Se endpoint não funcionar como documentado**: logar warning e pular (não crashar, não bloquear outras exchanges)
- **Order book é dado efêmero**: se uma coleta falhar, não tem backfill (ao contrário de funding rate que tem histórico). Logar o erro e seguir.
- **Manter pattern de parametrização via YAML**: adicionar novo depth level ou nova exchange deve ser só conf/
