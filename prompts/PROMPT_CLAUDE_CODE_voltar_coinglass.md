# PROMPT — Voltar derivativos para CoinGlass (agregado cross-exchange)

## Contexto

Decidimos voltar a usar CoinGlass como fonte principal de derivativos.
Motivos:
- CoinGlass agrega todas as exchanges (Binance, OKX, Bybit, etc.)
- OI Binance = $6.5B, OI agregado = $51.8B — ver só Binance é 12% do mercado
- Preparação para arbitragem cross-exchange
- Plano Hobbyist contratado (annual)

Binance Futures fica como backup / dados complementares (order book).

## Mudanças necessárias

### 1. Atualizar binance_futures.py → coinglass_futures.py

Os endpoints CoinGlass estão mapeados em `conf/coinglass_endpoints.yml`.
API v4 base: `https://open-api-v4.coinglass.com`

Endpoints a usar (todos 4h, limit=1080 = ~6 meses):

```yaml
# OI agregado (cross-exchange)
open_interest_aggregated_history:
  path: /api/futures/open-interest/aggregated-history
  params: {symbol: BTC, interval: 4h, limit: 1080}
  output: data/01_raw/futures/oi_4h.parquet
  fields: time → timestamp, close → open_interest

# Funding rate OI-weighted (cross-exchange)
funding_rate_oi_weight_history:
  path: /api/futures/funding-rate/oi-weight-history
  params: {symbol: BTC, interval: 4h, limit: 1080}
  output: data/01_raw/futures/funding_4h.parquet
  fields: time → timestamp, close → funding_rate

# Taker buy/sell (Binance — agregado não disponível no Hobbyist)
taker_buy_sell_volume_history:
  path: /api/futures/v2/taker-buy-sell-volume/history
  params: {exchange: Binance, symbol: BTCUSDT, interval: 4h, limit: 1080}
  output: data/01_raw/futures/taker_4h.parquet
  fields: time → timestamp, buySellRatio → buy_sell_ratio

# Liquidações (se disponível no plano)
pair_liquidation_history:
  path: /api/futures/liquidation/history
  params: {exchange: Binance, symbol: BTCUSDT, interval: 4h, limit: 1080}
  output: data/01_raw/futures/liquidations_4h.parquet
  fields: time → timestamp, longLiquidationUsd, shortLiquidationUsd

# Long/Short ratio top accounts (Binance)
# Nota: CoinGlass retornava 404 antes — testar novamente
```

### 2. Atualizar clean.py

Mudar de resample 1H pra resample 4H (ou manter 1H com ffill):

```python
# Opção A: manter grid 1H com ffill dos dados 4H
# Pro: paper trader continua rodando a cada hora
# Con: valores ficam "flat" entre candles 4H
df = df.set_index("timestamp").resample("1h").ffill()

# Opção B: mudar grid pra 4H
# Pro: dados reais, sem ffill
# Con: z-scores com menos pontos, paper trader precisa ajustar
```

RECOMENDAÇÃO: Opção A (ffill pra 1H). O paper trader continua rodando 
de hora em hora, e os dados de derivativos atualizam a cada 4h. 
Entre candles, o valor do último candle 4h é usado (forward-fill).
É o mesmo que já fazíamos com funding 8h → ffill 1h.

### 3. Atualizar parameters.yml

```yaml
ingestion:
  coinglass_futures:
    base_url: "https://open-api-v4.coinglass.com"
    interval: "4h"
    limit: 1080
    symbol_aggregated: "BTC"        # pra endpoints agregados
    symbol_exchange: "BTCUSDT"      # pra endpoints por exchange
    exchange: "Binance"             # pra endpoints que pedem exchange
```

### 4. Atualizar z-score windows

Z-scores de 14 dias com dados 4h = 14 * 6 = 84 candles (suficiente).
Z-scores de 30 dias com dados 4h = 30 * 6 = 180 candles (suficiente).

Não precisa mudar os windows em dias — a função compute_zscore usa rolling
sobre o número de rows, que agora são 4h em vez de 1h.

MAS: se o z-score window está em dias, precisa converter pra número de candles:
```python
# Se window = 14 (dias) e freq = 4h:
window_candles = window_days * (24 // 4)  # 14 * 6 = 84
```

Verificar se gate_features.py usa window em dias ou em rows.
Ajustar conforme necessário.

### 5. Atualizar hourly_cycle.sh

```bash
# ANTES:
python -m src.data.binance_futures

# DEPOIS:
python -m src.data.coinglass_futures
```

Manter binance_spot.py (candles 1h pra BB/RSI — CoinGlass não tem spot).

### 6. Lidar com o gap de dados

Os parquets em data/01_raw/futures/ atualmente têm dados Binance (1h).
Os novos dados CoinGlass serão 4h e com volumes diferentes.

Opções:
A) Limpar os parquets antigos e começar do zero com CoinGlass
B) Manter antigos e concatenar — gap de volume mas z-score normaliza

RECOMENDAÇÃO: Opção A — limpar e começar do zero.
O z-score precisa de 14 dias de dados consistentes.
Com CoinGlass 4h e limit=1080, já traz ~180 dias de histórico.
Não precisa dos dados Binance anteriores.

```python
# No início do fetch, se arquivo existe e source é diferente:
if filepath.exists():
    existing = pd.read_parquet(filepath)
    if "source" in existing.columns and existing["source"].iloc[0] != "coinglass":
        # Dados antigos são de outra fonte — começar do zero
        logger.info(f"Replacing {filepath}: old source was {existing['source'].iloc[0]}")
        existing = pd.DataFrame()  # limpa
```

### 7. Manter Binance Futures como complemento

Não deletar binance_futures.py — manter como módulo separado pra:
- Order book depth (CoinGlass não tem em real-time)
- L/S ratio (se CoinGlass não retornar)
- Backup se CoinGlass cair

### 8. Dashboard — atualizar labels

Onde mostra "OI (Binance)" agora deve mostrar "OI (Agregado)".
Liquidações e taker podem ser "Binance" se o endpoint CoinGlass for por exchange.

### 9. Testar

```bash
# Rodar ingestão CoinGlass
python -m src.data.coinglass_futures

# Verificar dados
python -c "
import pandas as pd
for f in ['oi_4h', 'funding_4h', 'taker_4h', 'liquidations_4h']:
    path = f'data/01_raw/futures/{f}.parquet'
    try:
        df = pd.read_parquet(path)
        print(f'{f}: {len(df)} rows | last={df[\"timestamp\"].max()}')
    except Exception as e:
        print(f'{f}: {e}')
"

# Rodar clean + features
python -m src.data.clean
python -m src.features.gate_features

# Rodar paper trader
python -m src.trading.paper_trader
```

## CRITICAL

- API key CoinGlass deve estar em conf/credentials.yml
- Headers: {"accept": "application/json", "CG-API-KEY": key} 
  (verificar nome exato do header na API v4)
- Todos os timestamps CoinGlass são Unix seconds (não ms) — 
  converter: pd.to_datetime(ts, unit="s", utc=True)
- O campo de dados na response é data["data"] (não data diretamente)
- Rate limit CoinGlass Hobbyist: verificar docs
