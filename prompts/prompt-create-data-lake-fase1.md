# Task: Criar repo btc-data-lake e implementar Fase 1 (Coleta de Funding Rates)

## Contexto

Projeto separado do AI.hab. Objetivo: data lake multi-exchange para suportar estratégia de funding rate arbitrage delta-neutral.

Assessment completo com APIs pesquisadas está em `prompts/assessment-data-lake-6-exchanges.md` no repo btc_AI. Usar como referência para endpoints, rate limits e schemas.

Filosofia: scripts puros Python, sem Kedro, sem frameworks pesados. Organização inspirada no Kedro (parametrização YAML, separação clara de camadas). Parquets como interface entre módulos.

## 1. Criar Estrutura do Repo

Criar em `~/Documents/MLGeral/btc-data-lake/`:

```
btc-data-lake/
├── conf/
│   ├── exchanges.yml          # Config por exchange
│   ├── instruments.yml        # Pares a monitorar
│   ├── collection.yml         # Frequências, timeouts, retries
│   └── credentials.yml        # (gitignored) API keys futuras
├── src/
│   ├── __init__.py
│   ├── config.py              # Loader centralizado de YAML
│   ├── collectors/
│   │   ├── __init__.py
│   │   ├── base.py            # BaseCollector (classe abstrata)
│   │   ├── binance.py
│   │   ├── okx.py
│   │   ├── bybit.py
│   │   ├── gateio.py
│   │   ├── bitget.py
│   │   └── kucoin.py
│   ├── normalizers/
│   │   ├── __init__.py
│   │   ├── schema.py          # Schema unificado
│   │   └── transforms.py      # Validação, limpeza
│   └── storage/
│       ├── __init__.py
│       └── parquet_writer.py  # Escrita particionada
├── scripts/
│   ├── collect_funding.py     # Script principal de coleta
│   ├── collect_all.sh         # Wrapper shell pro cron
│   └── show_spreads.py        # Visualização rápida de spreads
├── data/
│   ├── raw/                   # Bruto por exchange (gitignored)
│   │   ├── binance/
│   │   ├── okx/
│   │   ├── bybit/
│   │   ├── gateio/
│   │   ├── bitget/
│   │   └── kucoin/
│   └── normalized/            # Schema unificado (gitignored)
├── tests/
│   ├── __init__.py
│   ├── test_config.py         # Testa consistência YAML
│   ├── test_collectors.py     # Testa parsing de cada exchange
│   └── test_normalizer.py     # Testa schema unificado
├── .gitignore
├── requirements.txt
├── CLAUDE.md                  # Spec do projeto
└── README.md
```

## 2. Configuração YAML

### conf/exchanges.yml

```yaml
# Configuração por exchange — adicionar nova exchange = só YAML
# Cada exchange segue o mesmo schema; BaseCollector lê este arquivo.

exchanges:
  binance:
    enabled: true
    name: "Binance"
    base_url: "https://fapi.binance.com"
    endpoints:
      funding_rate:
        path: "/fapi/v1/fundingRate"
        method: GET
        params:
          symbol: "{symbol}"
          limit: 100
        response_mapping:
          timestamp: "fundingTime"
          funding_rate: "fundingRate"
          symbol: "symbol"
      funding_rate_current:
        path: "/fapi/v1/premiumIndex"
        method: GET
        params:
          symbol: "{symbol}"
        response_mapping:
          next_funding_time: "nextFundingTime"
          current_rate: "lastFundingRate"
          mark_price: "markPrice"
          index_price: "indexPrice"
    rate_limit:
      max_requests_per_minute: 1200
      weight_per_request: 1
      backoff_seconds: 5
    symbol_format: "{base}{quote}"  # BTCUSDT

  okx:
    enabled: true
    name: "OKX"
    base_url: "https://www.okx.com"
    endpoints:
      funding_rate:
        path: "/api/v5/public/funding-rate"
        method: GET
        params:
          instId: "{symbol}"
        response_mapping:
          timestamp: "fundingTime"
          funding_rate: "fundingRate"
          symbol: "instId"
          next_funding_time: "nextFundingTime"
      funding_rate_history:
        path: "/api/v5/public/funding-rate-history"
        method: GET
        params:
          instId: "{symbol}"
          limit: 100
        response_mapping:
          timestamp: "fundingTime"
          funding_rate: "fundingRate"
    rate_limit:
      max_requests_per_minute: 600
      weight_per_request: 1
      backoff_seconds: 5
    symbol_format: "{base}-{quote}-SWAP"  # BTC-USDT-SWAP

  bybit:
    enabled: true
    name: "Bybit"
    base_url: "https://api.bybit.com"
    endpoints:
      funding_rate:
        path: "/v5/market/funding/history"
        method: GET
        params:
          category: "linear"
          symbol: "{symbol}"
          limit: 200
        response_mapping:
          timestamp: "fundingRateTimestamp"
          funding_rate: "fundingRate"
          symbol: "symbol"
    rate_limit:
      max_requests_per_minute: 120
      weight_per_request: 1
      backoff_seconds: 5
    symbol_format: "{base}{quote}"  # BTCUSDT

  gateio:
    enabled: true
    name: "Gate.io"
    base_url: "https://fx-api.gateio.ws"
    endpoints:
      funding_rate:
        path: "/api/v4/futures/usdt/funding_rate"
        method: GET
        params:
          contract: "{symbol}"
          limit: 100
        response_mapping:
          timestamp: "t"
          funding_rate: "r"
          symbol: "contract"  # pode não estar no response, inferir do request
    rate_limit:
      max_requests_per_minute: 300
      weight_per_request: 1
      backoff_seconds: 5
    symbol_format: "{base}_{quote}"  # BTC_USDT

  bitget:
    enabled: true
    name: "Bitget"
    base_url: "https://api.bitget.com"
    endpoints:
      funding_rate:
        path: "/api/v2/mix/market/history-fund-rate"
        method: GET
        params:
          symbol: "{symbol}"
          productType: "usdt-futures"
          pageSize: 100
        response_mapping:
          timestamp: "fundingTime"
          funding_rate: "fundingRate"
          symbol: "symbol"
      funding_rate_current:
        path: "/api/v2/mix/market/current-fund-rate"
        method: GET
        params:
          symbol: "{symbol}"
          productType: "usdt-futures"
        response_mapping:
          current_rate: "fundingRate"
    rate_limit:
      max_requests_per_minute: 600
      weight_per_request: 1
      backoff_seconds: 5
    symbol_format: "{base}{quote}"  # BTCUSDT

  kucoin:
    enabled: true
    name: "KuCoin"
    base_url: "https://api-futures.kucoin.com"
    endpoints:
      funding_rate:
        path: "/api/v1/funding-rate/{symbol}/current"
        method: GET
        params: {}
        response_mapping:
          current_rate: "value"
          timestamp: "timePoint"
      funding_rate_history:
        path: "/api/v1/contract/funding-rates"
        method: GET
        params:
          symbol: "{symbol}"
          from: "{start_time}"
          to: "{end_time}"
        response_mapping:
          timestamp: "timePoint"
          funding_rate: "fundingRate"
    rate_limit:
      max_requests_per_minute: 180
      weight_per_request: 1
      backoff_seconds: 5
    symbol_format: "{base}{quote}M"  # XBTUSDTM
```

**IMPORTANTE:** Este YAML é ponto de partida. Os endpoints e response_mappings DEVEM ser validados contra a documentação oficial de cada exchange antes de implementar. Se algum endpoint estiver incorreto, corrigir no YAML e documentar.

### conf/instruments.yml

```yaml
instruments:
  - id: BTCUSDT
    base: BTC
    quote: USDT
    exchange_symbols:
      binance: "BTCUSDT"
      okx: "BTC-USDT-SWAP"
      bybit: "BTCUSDT"
      gateio: "BTC_USDT"
      bitget: "BTCUSDT"
      kucoin: "XBTUSDTM"

  # Futuro — adicionar ETHUSDT é só copiar o bloco acima
  # - id: ETHUSDT
  #   base: ETH
  #   quote: USDT
  #   exchange_symbols:
  #     binance: "ETHUSDT"
  #     ...
```

### conf/collection.yml

```yaml
collection:
  funding_rate:
    frequency: "8h"           # Coleta sincronizada com períodos de funding
    cron: "10 0,8,16 * * *"   # 10min depois do settlement (dá tempo de propagar)
    history_days: 30           # Backfill inicial: 30 dias
    retry_attempts: 3
    retry_delay_seconds: 10
    timeout_seconds: 30

  # Fase 2 — descomentar quando implementar
  # order_book:
  #   frequency: "1h"
  #   cron: "5 * * * *"
  #   depth: 10
  # klines:
  #   frequency: "1h"
  #   cron: "3 * * * *"
  #   interval: "1h"
  #   limit: 24

storage:
  format: "parquet"
  partition_by: ["exchange", "date"]  # data/normalized/binance/2026-04-16.parquet
  compression: "snappy"

logging:
  level: "INFO"
  file: "logs/collection.log"
  max_size_mb: 50
  backup_count: 5
```

## 3. Implementação dos Módulos

### src/config.py

Loader centralizado que carrega os 3 YAMLs e expõe como dicts. Pattern idêntico ao do AI.hab.

```python
import yaml
from pathlib import Path

_CONF_DIR = Path(__file__).parent.parent / "conf"

def load_config(filename: str) -> dict:
    with open(_CONF_DIR / filename, "r") as f:
        return yaml.safe_load(f)

def get_exchanges() -> dict:
    return load_config("exchanges.yml")["exchanges"]

def get_instruments() -> list:
    return load_config("instruments.yml")["instruments"]

def get_collection_config() -> dict:
    return load_config("collection.yml")
```

### src/collectors/base.py

Classe abstrata que:
1. Lê config do exchange.yml
2. Monta URL com symbol substituído
3. Faz request com retry + backoff
4. Chama `parse_response()` (implementado por cada subclasse)
5. Retorna lista de dicts normalizados

```python
from abc import ABC, abstractmethod
import requests
import time
import logging

class BaseCollector(ABC):
    def __init__(self, exchange_config: dict, instrument_config: dict):
        self.config = exchange_config
        self.instrument = instrument_config
        self.name = exchange_config["name"]
        self.base_url = exchange_config["base_url"]
        self.rate_limit = exchange_config["rate_limit"]
        self.logger = logging.getLogger(f"collector.{self.name}")

    def get_symbol(self) -> str:
        """Retorna o symbol no formato da exchange."""
        return self.instrument["exchange_symbols"][self.name.lower()]

    def fetch(self, endpoint_key: str, extra_params: dict = None) -> dict:
        """Faz request com retry e backoff."""
        endpoint = self.config["endpoints"][endpoint_key]
        url = self.base_url + endpoint["path"].replace("{symbol}", self.get_symbol())
        params = {k: v.replace("{symbol}", self.get_symbol()) if isinstance(v, str) else v
                  for k, v in endpoint.get("params", {}).items()}
        if extra_params:
            params.update(extra_params)

        collection_config = ...  # carregar de collection.yml
        for attempt in range(collection_config.get("retry_attempts", 3)):
            try:
                resp = requests.get(url, params=params,
                                   timeout=collection_config.get("timeout_seconds", 30))
                resp.raise_for_status()
                return resp.json()
            except Exception as e:
                self.logger.warning(f"Attempt {attempt+1} failed: {e}")
                time.sleep(collection_config.get("retry_delay_seconds", 10))
        raise RuntimeError(f"Failed to fetch {endpoint_key} from {self.name} after retries")

    @abstractmethod
    def collect_funding_rates(self) -> list[dict]:
        """Retorna lista de dicts com schema normalizado:
        {
            "timestamp": datetime (UTC),
            "exchange": str,
            "symbol": str (normalizado, ex: BTCUSDT),
            "funding_rate": float,
            "next_funding_time": datetime or None,
            "mark_price": float or None,
            "index_price": float or None,
        }
        """
        pass

    @abstractmethod
    def collect_current_funding(self) -> dict:
        """Retorna dict com funding rate atual e próximo settlement."""
        pass
```

### src/collectors/binance.py (referência para os outros)

```python
from .base import BaseCollector
from datetime import datetime, timezone

class BinanceCollector(BaseCollector):
    def collect_funding_rates(self) -> list[dict]:
        data = self.fetch("funding_rate")
        mapping = self.config["endpoints"]["funding_rate"]["response_mapping"]
        results = []
        for item in data:
            results.append({
                "timestamp": datetime.fromtimestamp(
                    int(item[mapping["timestamp"]]) / 1000, tz=timezone.utc
                ),
                "exchange": "binance",
                "symbol": self.instrument["id"],  # normalizado: BTCUSDT
                "funding_rate": float(item[mapping["funding_rate"]]),
                "next_funding_time": None,
                "mark_price": None,
                "index_price": None,
            })
        return results

    def collect_current_funding(self) -> dict:
        data = self.fetch("funding_rate_current")
        # Binance retorna lista com 1 item ou dict direto
        item = data[0] if isinstance(data, list) else data
        mapping = self.config["endpoints"]["funding_rate_current"]["response_mapping"]
        return {
            "timestamp": datetime.now(timezone.utc),
            "exchange": "binance",
            "symbol": self.instrument["id"],
            "funding_rate": float(item[mapping["current_rate"]]),
            "next_funding_time": datetime.fromtimestamp(
                int(item[mapping["next_funding_time"]]) / 1000, tz=timezone.utc
            ),
            "mark_price": float(item.get(mapping.get("mark_price", ""), 0)),
            "index_price": float(item.get(mapping.get("index_price", ""), 0)),
        }
```

### src/collectors/okx.py, bybit.py, gateio.py, bitget.py, kucoin.py

Cada um implementa `collect_funding_rates()` e `collect_current_funding()` seguindo o pattern do Binance. As diferenças são:
- Parsing do response JSON (cada exchange tem formato diferente)
- Tratamento de timestamp (ms vs seconds vs ISO string)
- Campos disponíveis (nem todas retornam mark_price/index_price)

**OKX:** response vem em `data.data[0]` (wrapper). `instId` no formato `BTC-USDT-SWAP`.
**Bybit:** response vem em `result.list[]`. Timestamps em ms.
**Gate.io:** response é lista direta. Funding rate no campo `r`. Timestamps em seconds.
**Bitget:** response vem em `data.list[]`. `productType` obrigatório.
**KuCoin:** funding rate current usa path param `{symbol}`, não query param. Symbol é `XBTUSDTM`.

**Para cada exchange:**
1. Verificar a documentação oficial do endpoint
2. Fazer 1 request manual (curl ou browser) para ver o formato real do response
3. Implementar o parsing baseado no response real, não no YAML estimado
4. Testar com `pytest tests/test_collectors.py`

### src/normalizers/schema.py

Define o schema Pandas/Parquet unificado:

```python
FUNDING_RATE_SCHEMA = {
    "timestamp": "datetime64[ns, UTC]",
    "exchange": "string",
    "symbol": "string",
    "funding_rate": "float64",
    "next_funding_time": "datetime64[ns, UTC]",  # nullable
    "mark_price": "float64",                      # nullable
    "index_price": "float64",                     # nullable
}

FUNDING_RATE_COLUMNS = list(FUNDING_RATE_SCHEMA.keys())
```

### src/normalizers/transforms.py

Validação:
- Todos os campos obrigatórios presentes
- funding_rate dentro de range razoável (-1.0 a +1.0) — se fora, é bug de parsing (provavelmente % vs decimal)
- Timestamps no futuro → warning
- Duplicatas removidas (mesmo exchange + symbol + timestamp)

### src/storage/parquet_writer.py

Escrita particionada:

```python
def write_funding_rates(records: list[dict], base_path: str = "data/normalized"):
    df = pd.DataFrame(records)
    # Partição: data/normalized/funding_rates/exchange=binance/date=2026-04-16/data.parquet
    for (exchange, date), group in df.groupby([df.exchange, df.timestamp.dt.date]):
        path = Path(base_path) / "funding_rates" / f"exchange={exchange}" / f"date={date}"
        path.mkdir(parents=True, exist_ok=True)
        outfile = path / "data.parquet"
        if outfile.exists():
            existing = pd.read_parquet(outfile)
            group = pd.concat([existing, group]).drop_duplicates(
                subset=["timestamp", "exchange", "symbol"]
            )
        group.to_parquet(outfile, index=False, compression="snappy")
```

Também escrever raw por exchange:

```python
def write_raw(records: list[dict], exchange: str, base_path: str = "data/raw"):
    # data/raw/binance/funding_2026-04-16.parquet
    ...
```

### scripts/collect_funding.py

Script principal que orquestra a coleta:

```python
#!/usr/bin/env python
"""Coleta funding rates de todas as exchanges habilitadas."""

import sys
import logging
from datetime import datetime, timezone
from src.config import get_exchanges, get_instruments, get_collection_config
from src.collectors import COLLECTOR_MAP  # {name: CollectorClass}
from src.normalizers.transforms import validate_records
from src.storage.parquet_writer import write_funding_rates, write_raw

def main():
    logging.basicConfig(...)
    logger = logging.getLogger("collect_funding")

    exchanges = get_exchanges()
    instruments = get_instruments()
    config = get_collection_config()

    all_records = []
    errors = []

    for inst in instruments:
        for name, exc_config in exchanges.items():
            if not exc_config.get("enabled", False):
                continue
            if name not in inst.get("exchange_symbols", {}):
                continue

            try:
                collector = COLLECTOR_MAP[name](exc_config, inst)
                # Coleta rate atual
                current = collector.collect_current_funding()
                all_records.append(current)
                logger.info(f"✅ {name}: {inst['id']} funding={current['funding_rate']:.6f}")
            except Exception as e:
                logger.error(f"❌ {name}: {inst['id']} failed: {e}")
                errors.append({"exchange": name, "symbol": inst["id"], "error": str(e)})

    # Validar e persistir
    valid_records = validate_records(all_records)
    write_funding_rates(valid_records)
    write_raw(valid_records, ...)

    # Resumo
    logger.info(f"Collected {len(valid_records)} records, {len(errors)} errors")

    # Print spread summary
    if len(valid_records) >= 2:
        rates = {r["exchange"]: r["funding_rate"] for r in valid_records}
        max_ex = max(rates, key=rates.get)
        min_ex = min(rates, key=rates.get)
        spread = rates[max_ex] - rates[min_ex]
        logger.info(f"💰 Spread: {spread:.6f} ({max_ex} {rates[max_ex]:.6f} vs {min_ex} {rates[min_ex]:.6f})")

    return 0 if not errors else 1

if __name__ == "__main__":
    sys.exit(main())
```

### scripts/show_spreads.py

Utilitário para visualizar funding rates e spreads:

```python
#!/usr/bin/env python
"""Mostra funding rates atuais e spread entre exchanges."""

import pandas as pd
from pathlib import Path

def main():
    base = Path("data/normalized/funding_rates")
    if not base.exists():
        print("Sem dados ainda. Rode collect_funding.py primeiro.")
        return

    # Lê últimos registros de cada exchange
    # Mostra tabela formatada:
    #
    # Exchange   | Funding Rate | Next Settlement | Mark Price
    # Binance    | +0.0100%     | 2026-04-16 08:00| $74,300
    # OKX        | +0.0085%     | 2026-04-16 08:00| $74,295
    # Bybit      | +0.0120%     | 2026-04-16 08:00| $74,310
    # ...
    # Spread Max: 0.0035% (Bybit vs OKX)
    # Anualizado: ~16% (se mantido)

if __name__ == "__main__":
    main()
```

### scripts/collect_all.sh

```bash
#!/usr/bin/env bash
set -euo pipefail

if [ -d "/app" ]; then
    cd /app
else
    cd "$(dirname "$0")/.."
fi

echo "[$(date -u)] Starting funding rate collection..."
python scripts/collect_funding.py
echo "[$(date -u)] Collection complete."
```

## 4. Testes

### tests/test_config.py

- Todos os exchanges em exchanges.yml têm os campos obrigatórios
- Todos os instruments têm exchange_symbols para cada exchange habilitada
- symbol_format é consistente

### tests/test_collectors.py

Para cada exchange:
- Mock do response JSON (fixture com exemplo real)
- `collect_funding_rates()` retorna lista com schema correto
- `collect_current_funding()` retorna dict com schema correto
- Timestamps são UTC
- funding_rate é float entre -1.0 e +1.0

### tests/test_normalizer.py

- Validação rejeita funding_rate fora de range
- Duplicatas removidas corretamente
- Schema enforcement (columns, dtypes)

## 5. Backfill Inicial

Após os collectors funcionarem, rodar backfill de 30 dias para cada exchange:

```python
# scripts/backfill_funding.py
# Para cada exchange, busca funding rate history dos últimos 30 dias
# Persiste em parquets particionados
# Vai ser a base de dados para análise de viabilidade
```

## 6. Arquivos de Suporte

### .gitignore

```
data/
logs/
conf/credentials.yml
__pycache__/
*.pyc
.env
.venv/
```

### requirements.txt

```
requests>=2.31.0
pandas>=2.0.0
pyarrow>=14.0.0
pyyaml>=6.0.0
pytest>=7.0.0
```

### CLAUDE.md

Criar com:
- Descrição do projeto (data lake multi-exchange para arbitragem)
- Arquitetura (collectors → normalizers → storage)
- Como adicionar nova exchange (só YAML)
- Como adicionar novo instrumento (só YAML)
- Referência ao assessment (`prompts/assessment-data-lake-6-exchanges.md` no btc_AI)
- Relação com AI.hab (projetos irmãos, CoinGlass para AI.hab, direct para data lake)
- Comandos: `python scripts/collect_funding.py`, `python scripts/show_spreads.py`

## 7. Setup do Ambiente

Antes de rodar qualquer código:

```bash
cd ~/Documents/MLGeral/btc-data-lake

# Criar virtualenv
python3 -m venv .venv
source .venv/bin/activate

# Instalar dependências
pip install -r requirements.txt

# Verificar
python -c "import requests, pandas, pyarrow, yaml; print('✅ Todas as libs instaladas')"
```

Adicionar ao `.gitignore`:
```
.venv/
```

O `requirements.txt` já está definido na seção 6. Garantir que seja criado ANTES de tentar rodar qualquer script.

## 8. Validação da Fase 1

Após implementar tudo:

1. **Rodar coleta manualmente:**
   ```bash
   cd ~/Documents/MLGeral/btc-data-lake
   python scripts/collect_funding.py
   ```

2. **Verificar output:**
   ```bash
   python scripts/show_spreads.py
   ```
   Deve mostrar funding rate de pelo menos 5 das 6 exchanges (se alguma falhar, ok — logar erro e continuar).

3. **Verificar parquets:**
   ```bash
   python -c "
   import pandas as pd
   from pathlib import Path
   for f in sorted(Path('data/normalized').rglob('*.parquet')):
       df = pd.read_parquet(f)
       print(f'{f}: {len(df)} rows')
       print(df.head().to_string())
   "
   ```

4. **Rodar testes:**
   ```bash
   pytest tests/ -v
   ```

5. **Rodar backfill:**
   ```bash
   python scripts/backfill_funding.py
   ```

6. **Inicializar git:**
   ```bash
   git init
   git add -A
   git commit -m "feat: btc-data-lake phase 1 — funding rate collection from 6 exchanges"
   ```

## Restrições

- **Não criar conta em nenhuma exchange** — usar apenas endpoints públicos
- **Não usar API keys** nesta fase — tudo é market data público
- **Se um endpoint não funcionar como documentado**, logar warning e pular (não crashar)
- **Cada collector deve ser testável independentemente** — se Bitget estiver down, os outros 5 continuam
- **Não instalar frameworks pesados** (Kedro, Airflow, etc.) — Python puro + libs básicas
- **Responses das APIs podem mudar** — usar try/except generoso no parsing
