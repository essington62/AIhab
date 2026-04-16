# Assessment: Data Lake Multi-Exchange para Funding Rate Arbitrage

**Data:** 2026-04-16
**Objetivo:** Avaliar viabilidade de coletar dados de derivativos diretamente das 6 maiores exchanges para suportar estratégia de funding rate arbitrage delta-neutral.
**Exchanges:** Binance, OKX, Bybit, Gate.io, Bitget, KuCoin

---

## 1. Resumo Executivo

**Viável? SIM.**

Todas as 6 exchanges oferecem APIs públicas gratuitas com endpoints de funding rate, order book, klines (candle mínimo 1m) e open interest. Não há necessidade de API key para dados de mercado público. O custo de infraestrutura é zero além da EC2 que já temos.

**Por que não ficar só no CoinGlass:**

- CoinGlass Hobbyist ($29/mês): candle mínimo 4h, 30 req/min, dados agregados
- Para arbitragem precisamos: funding rate **por exchange individual**, granularidade por período de funding (8h) ou menor, order book depth para calcular slippage
- CoinGlass agrega — pra arbitragem precisamos do oposto: ver a **diferença** entre exchanges

**Recomendação: modelo híbrido**

- CoinGlass continua no AI.hab (OI-weighted, contexto macro, dashboard)
- Data lake vai direto nas 6 exchanges (funding individual, spreads, books)

---

## 2. Comparativo de APIs — Dados Críticos para Arbitragem

### 2.1 Funding Rate

| Exchange | Endpoint | Frequência | Auth | Histórico | Rate Limit |
|----------|----------|-----------|------|-----------|------------|
| Binance | `/fapi/v1/fundingRate` (USDT-M) `/dapi/v1/fundingRate` (Coin-M) | 8h (00:00, 08:00, 16:00 UTC) | Público | Sim, com startTime/endTime | 500 weight/5min |
| OKX | `/api/v5/public/funding-rate` + WS channel `funding-rate` | 8h | Público | Sim | VIP tier-based |
| Bybit | V5 API funding rate history | 8h (00:00, 08:00, 16:00 UTC) | Público | Sim | Padrão V5 |
| Gate.io | Futures API funding rate | 8h | Público | Sim | Per-endpoint |
| Bitget | Perpetual futures funding | 8h | Público | Sim | 20 req/sec/IP |
| KuCoin | `/api/v1/funding-rate/{symbol}/current` | 8h | Público | Sim, com time ranges | Weight/VIP |

**Conclusão:** Todas oferecem funding rate gratuito com histórico. Diferença é só no rate limit e formato de resposta.

### 2.2 Klines (Candles)

| Exchange | Candle Mínimo | Max por Request | Base URL Futures |
|----------|--------------|-----------------|------------------|
| Binance | 1m | 1000 candles | `fapi.binance.com` |
| OKX | 1m | Não especificado | `okx.com/api/v5` |
| Bybit | 1m | 1000 candles | `api.bybit.com/v5` |
| Gate.io | 1m | 2000 candles | `fx-api.gateio.ws/api/v4` |
| Bitget | 1m | Não especificado | `api.bitget.com` |
| KuCoin | 1m | 500 candles | `api-futures.kucoin.com/api/v1` |

**Conclusão:** Todas suportam 1m. Para arbitragem de funding (decisão a cada 8h), candles de 1h ou até 4h seriam suficientes. 1m seria útil para timing de execução.

### 2.3 Order Book

| Exchange | REST | WebSocket | Profundidade | Update Freq WS |
|----------|------|-----------|-------------|----------------|
| Binance | `/fapi/v1/depth` | Sim | Até 5000 níveis | 100ms-1000ms |
| OKX | Sim | Sim (recomendado) | 5, 10, 20, 50, 100 | Real-time |
| Bybit | Sim | `stream.bybit.com/v5/public/linear` | 5, 10, 20, 50, 100 | Real-time |
| Gate.io | `list_futures_order_book()` | `futures.order_book` channel | Variável | Real-time |
| Bitget | Sim | Sim | Variável | Real-time |
| KuCoin | Sim | Sim | Variável | Real-time |

**Conclusão:** Order book disponível em todas. Para arbitragem, depth de 5-10 níveis é suficiente para calcular slippage em posições de $500-$5000.

### 2.4 Open Interest (por exchange)

| Exchange | Endpoint | Granularidade |
|----------|----------|---------------|
| Binance | Não tem endpoint dedicado (via fundingInfo) | Limitado |
| OKX | `GetContractsOpenInterestAndVolume` | Per-symbol |
| Bybit | `/v5/market/open-interest` | Per-symbol, com intervalos |
| Gate.io | Campo `open_interest` em ContractStat | Per-symbol |
| Bitget | `/api/v2/mix/market/open-interest` | Per-symbol |
| KuCoin | Endpoint dedicado (atualizado Jan 2026) | Per-symbol |

**Conclusão:** 5 de 6 têm OI dedicado. Binance é o gap — precisa extrair de outro endpoint. Para arbitragem, OI é secundário (mais relevante pro AI.hab).

### 2.5 Taker Buy/Sell Volume

| Exchange | Disponível | Granularidade |
|----------|-----------|---------------|
| Binance | Sim: `/futures/data/takerBuySellVol` | 5m, 15m, 30m, 1h, 4h, 1d |
| OKX | Limitado | Via market data |
| Bybit | Não documentado | — |
| Gate.io | Limitado | Via market data |
| Bitget | Não documentado | — |
| KuCoin | Não documentado | — |

**Conclusão:** Só Binance tem taker volume granular. Para arbitragem, não é crítico — mais relevante pro AI.hab (já usa via CoinGlass).

---

## 3. Rate Limits — Resumo Operacional

| Exchange | Modelo | Limite Principal | Recovery |
|----------|--------|-----------------|----------|
| Binance | IP-based weight | 1200 req/min por IP | Automático |
| OKX | User ID / VIP | VIP tier-dependent | — |
| Bybit | IP-based | 600 req/5sec por IP | — |
| Gate.io | Misto IP + endpoint | 10 req/10sec (orders); geral variável | — |
| Bitget | IP-based | 6000 req/min por IP; 20 req/sec para contratos | 5min após limit |
| KuCoin | UID weight-based | Weight por 30 seg | 429 error |

**Impacto para o data lake:** Com 6 exchanges × ~5 endpoints cada = ~30 chamadas por ciclo. A cada 8h = 90 chamadas/dia. Nenhum rate limit é sequer tocado. Mesmo coletando a cada 1h (180 chamadas/dia) é trivial.

---

## 4. Fees para Trading (Referência — Execução Futura)

| Exchange | Maker | Taker | Nota |
|----------|-------|-------|------|
| Binance | 0.020% | 0.040-0.050% | VIP9: 0%/0.017% |
| OKX | 0.020% | 0.050% | VIP scaling |
| Bybit | 0.020% | 0.055% | VIP: 0%/0.030% |
| Gate.io | 0.020% | 0.050% | VIP16: 0%/0.016% |
| Bitget | 0.020% | 0.060% | Promos ativas 2026 |
| KuCoin | 0.020% | 0.060% | VIP9 maker rebate removido Mar 2026 |

**Para arbitragem:** fees impactam diretamente o P&L. Com funding rate médio de 0.01-0.05%/8h e fee de 0.04% taker por leg (spot + futuro), o break-even exige funding rate > ~0.03% por período. Priorizar exchanges com menor taker fee (Binance, OKX).

---

## 5. CoinGlass vs Direct — Quando Usar Cada Um

| Necessidade | CoinGlass | Direct Exchange |
|-------------|-----------|-----------------|
| OI-weighted funding (benchmark) | ✅ Exclusivo | ❌ Precisa calcular |
| Funding rate por exchange | ✅ (4h mín) | ✅ (real-time) |
| Cross-exchange agregação | ✅ Pronto | ❌ Build próprio |
| Candle < 4h | ❌ Limite do plano | ✅ (1m disponível) |
| Order book depth | ❌ Não disponível | ✅ Real-time |
| Custo | $29/mês | $0 |
| Manutenção | Zero | Média (6 integrações) |

**Decisão:** Manter CoinGlass para AI.hab (macro, dashboard). Data lake direto nas exchanges para arbitragem.

---

## 6. Alternativas a CoinGlass (Referência)

| Plataforma | Pontos Fortes | Custo |
|------------|---------------|-------|
| Coinalyze | Free API, OI/funding agregado, 40 req/min | Grátis (API) |
| Laevitas | Institucional, options chains, analytics profundo | Enterprise $$ |
| Tardis.dev | Tick-level, 1min granularidade, CSV exports | Subscription $$ |
| CoinAPI.io | API unificada 100+ exchanges | Metered pricing |

**Para o futuro:** se o data lake próprio se tornar custoso de manter, Tardis.dev é a alternativa mais completa para dados de alta frequência.

---

## 7. Arquitetura Proposta — Data Lake (btc-data-lake/)

```
btc-data-lake/
├── conf/
│   ├── exchanges.yml          # Config por exchange (URL, endpoints, rate limits)
│   ├── instruments.yml        # Pares a monitorar (BTCUSDT perp, etc.)
│   ├── collection.yml         # Frequências de coleta por tipo de dado
│   └── credentials.yml        # API keys (se necessário no futuro para trading)
├── src/
│   ├── collectors/
│   │   ├── base.py            # Classe abstrata BaseCollector
│   │   ├── binance.py         # Implementação Binance
│   │   ├── okx.py             # Implementação OKX
│   │   ├── bybit.py           # Implementação Bybit
│   │   ├── gateio.py          # Implementação Gate.io
│   │   ├── bitget.py          # Implementação Bitget
│   │   └── kucoin.py          # Implementação KuCoin
│   ├── normalizers/
│   │   ├── schema.py          # Schema unificado (todos exchanges → mesmo formato)
│   │   └── transforms.py      # Limpeza, resampling, validação
│   ├── storage/
│   │   ├── parquet_writer.py  # Escrita particionada por exchange/date
│   │   └── s3_sync.py         # (futuro) sync para S3
│   └── signals/
│       ├── funding_spread.py  # Spread de funding entre exchanges
│       ├── entry_exit.py      # Regras de entrada/saída
│       └── risk.py            # Margem, liquidação, position sizing
├── scripts/
│   ├── collect_funding.sh     # Coleta funding rates (a cada 8h)
│   ├── collect_orderbook.sh   # Snapshot order book (a cada 1h)
│   └── collect_klines.sh      # Klines horários
├── data/
│   ├── raw/                   # Dados brutos por exchange
│   │   ├── binance/
│   │   ├── okx/
│   │   ├── bybit/
│   │   ├── gateio/
│   │   ├── bitget/
│   │   └── kucoin/
│   ├── normalized/            # Schema unificado, parquets particionados
│   ├── signals/               # Spreads calculados, oportunidades
│   └── reports/               # Relatórios diários/semanais
├── tests/
├── docker/
├── Dockerfile
├── docker-compose.yml
└── CLAUDE.md                  # Spec do projeto
```

### Princípio de parametrização (exchanges.yml):

```yaml
exchanges:
  binance:
    enabled: true
    base_url: "https://fapi.binance.com"
    funding_rate:
      endpoint: "/fapi/v1/fundingRate"
      params:
        symbol: "{instrument}"
        limit: 100
      frequency: "8h"
    klines:
      endpoint: "/fapi/v1/klines"
      params:
        symbol: "{instrument}"
        interval: "1h"
        limit: 500
    order_book:
      endpoint: "/fapi/v1/depth"
      params:
        symbol: "{instrument}"
        limit: 10
    rate_limit:
      max_requests_per_minute: 1200
      weight_per_request: 1

  okx:
    enabled: true
    base_url: "https://www.okx.com"
    funding_rate:
      endpoint: "/api/v5/public/funding-rate"
      params:
        instId: "{instrument}"
      frequency: "8h"
    # ... mesmo padrão
```

### Adicionar nova exchange = só YAML:

```yaml
  # Nova exchange — zero código
  mexc:
    enabled: true
    base_url: "https://futures.mexc.com"
    funding_rate:
      endpoint: "/api/v1/contract/funding_rate"
      params:
        symbol: "{instrument}"
    # ...
```

### Adicionar novo instrumento = só YAML:

```yaml
instruments:
  - symbol: BTCUSDT
    exchange_symbols:
      binance: "BTCUSDT"
      okx: "BTC-USDT-SWAP"
      bybit: "BTCUSDT"
      gateio: "BTC_USDT"
      bitget: "BTCUSDT"
      kucoin: "XBTUSDTM"
  - symbol: ETHUSDT
    exchange_symbols:
      binance: "ETHUSDT"
      # ... mesmo padrão
```

---

## 8. Fases de Implementação

### Fase 1 — Coleta de Funding Rates (1-2 semanas)

- Implementar BaseCollector + 6 collectors (funding rate apenas)
- Schema unificado: timestamp, exchange, symbol, funding_rate, next_funding_time
- Cron a cada 8h (sincronizado com períodos de funding)
- Persistência em parquet particionado por exchange/date
- Dashboard simples: tabela de funding rates por exchange + spread max

### Fase 2 — Order Book + Klines (2-3 semanas)

- Adicionar coleta de order book snapshot (top 10 níveis, 1h)
- Adicionar klines 1h para cada exchange
- Calcular spread implícito (bid/ask) e slippage estimado por tamanho de posição
- Normalização cross-exchange (timestamps, formatos)

### Fase 3 — Sinais de Arbitragem (2-3 semanas)

- Calcular funding spread entre pares de exchanges
- Implementar regras de entrada/saída do texto original
- Risk management: margem, liquidação, position sizing
- Paper trading: simular execução sem capital real

### Fase 4 — Execução (futuro)

- Integração com APIs de trading (requer API keys com permissão)
- AI.hab como orquestrador (gate scoring adaptado)
- Multi-bot: Grid, Arb, Trend

---

## 9. Riscos e Mitigações

| Risco | Probabilidade | Impacto | Mitigação |
|-------|--------------|---------|-----------|
| Exchange muda API sem aviso | Média | Alto | Schema validation + alertas de falha de coleta |
| Rate limit atingido | Baixa | Médio | Backoff exponencial + coleta conservadora |
| Funding rate vai a zero (oportunidade some) | Média | Alto | Monitorar tendência; só operar quando rate > 0.03% |
| Exchange quebra (risco FTX) | Baixa | Crítico | Nunca >30% do capital em 1 exchange; rebalancear |
| Dados inconsistentes entre exchanges | Média | Médio | Validação cruzada, alertas de anomalia |
| Latência de execução | Média | Médio | Para funding arb, latência de segundos é OK (janela de 8h) |

---

## 10. Decisão e Próximos Passos

**Decisão: PROSSEGUIR com Fase 1 (coleta de funding rates)**

Justificativa:
- Custo zero de API
- Infraestrutura já existe (EC2, Docker)
- Funding rate é o dado mais crítico para a estratégia
- 2 semanas de dados coletados já permitem análise de viabilidade da arbitragem

**Próximo passo imediato:**
Criar o repo btc-data-lake/ com a estrutura proposta, conf/exchanges.yml parametrizado, e o primeiro collector (Binance) como referência. Os outros 5 seguem o mesmo padrão.
