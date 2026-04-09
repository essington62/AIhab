# ADDENDUM — Fixes obrigatórios na Data Layer

## Aplicar a TODOS os módulos de ingestão antes de considerar pronto.

## Fix 1: Timezone enforcement (CRÍTICO)

TODOS os módulos devem garantir UTC em toda operação de timestamp.

```python
# Padrão obrigatório em TODO módulo:
df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
assert df["timestamp"].dt.tz is not None, "Timezone must be UTC"
```

Coluna de tempo é SEMPRE `timestamp` (nunca `date`, `time`, `dt`).
Exceção: FRED usa `date` no source, converter pra `timestamp` ao salvar.

## Fix 2: Janela máxima nos parquets (evitar inflação)

Parquets são append-only MAS com janela máxima:

```python
# Após concat + dedup, truncar:
MAX_ROWS = {
    "1h": 8760,    # ~1 ano de dados 1h
    "daily": 1095,  # ~3 anos de dados diários
    "8h": 3285,     # ~3 anos de funding events
}

def save_with_window(df, filepath, freq="1h"):
    max_rows = MAX_ROWS.get(freq, 8760)
    df = df.sort_values("timestamp").tail(max_rows)
    df.to_parquet(filepath, index=False)
```

## Fix 3: Resample Binance Futures pra grid consistente (CRÍTICO)

Binance Futures pode ter gaps e timestamps desalinhados.
Após fetch, forçar grid 1h:

```python
def align_to_hourly_grid(df, value_cols):
    """Resample para grid 1H consistente, preenchendo gaps."""
    df = df.set_index("timestamp")
    df = df.resample("1h").last()  # último valor de cada hora
    # NaN em gaps fica como NaN — o z-score lida com isso
    df = df.reset_index()
    return df
```

Aplicar em: oi_1h, taker_1h, ls_account, ls_position.

## Fix 4: Funding rate = forward-fill (CRÍTICO)

Funding é evento 8h, não 1h. Pra usar no ciclo 1h:

```python
def process_funding(df):
    """Converte funding 8h em série 1h via forward-fill."""
    df = df.set_index("timestamp")
    # Resample pra 1H e forward-fill
    df = df.resample("1h").ffill()
    df = df.reset_index()
    return df
```

O valor de funding fica constante entre eventos (correto semanticamente).

## Fix 5: FRED incremental sem duplicar

```python
# ERRADO:
new = fred.get_series(series_id, observation_start=last_date)
# → FRED inclui last_date de novo → duplica

# CORRETO:
observation_start = (last_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
new = fred.get_series(series_id, observation_start=observation_start)
```

## Fix 6: News dedup robusto

```python
import hashlib

def news_hash(title, source):
    """Hash determinístico pra dedup de news."""
    raw = (title[:100].lower().strip() + "|" + source.lower()).encode()
    return hashlib.md5(raw).hexdigest()

# Usar como coluna de dedup:
df["hash"] = df.apply(lambda r: news_hash(r["title"], r["source"]), axis=1)
df = df.drop_duplicates(subset=["hash"])
```

## Fix 7: Retry/backoff real (obrigatório)

Template pra TODAS as API calls:

```python
import time
import requests
import logging

logger = logging.getLogger(__name__)

def fetch_with_retry(url, params=None, max_retries=3, timeout=30):
    """Fetch com retry exponential backoff."""
    for attempt in range(max_retries):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except requests.exceptions.RequestException as e:
            wait = 2 ** attempt
            logger.warning(f"Attempt {attempt+1}/{max_retries} failed: {e}. Retry in {wait}s")
            if attempt < max_retries - 1:
                time.sleep(wait)
            else:
                logger.error(f"All {max_retries} attempts failed for {url}")
                raise
```

Usar `fetch_with_retry()` em vez de `requests.get()` direto em TODOS os módulos.

## Fix 8: Coluna source em todos os parquets

```python
df["source"] = "binance_futures"  # ou "fred", "coinglass", "alt_me", etc.
```

Facilita debug quando concatenar dados de fontes diferentes.

## Fix 9: Validar monotonicidade

```python
# Após sort e dedup, validar:
assert df["timestamp"].is_monotonic_increasing, f"Non-monotonic timestamps in {filepath}"
```

## Fix 10: Logging estruturado

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S UTC",
)

logger = logging.getLogger("data_layer")

# Em cada módulo:
logger.info(f"{filename}: +{len(new_data)} rows, total={len(combined)}, "
            f"last={combined['timestamp'].max()}")
```

## Fix 11: Camada intermediária (02_intermediate)

Separar raw (sujo) de clean (processado):

```
data/
├── 01_raw/          # Dados como vieram da API (append-only)
├── 02_intermediate/ # Dados limpos: resampled, aligned, ffill
│   ├── futures/
│   │   ├── oi_1h_clean.parquet      # resampled 1H grid
│   │   ├── taker_1h_clean.parquet   # resampled 1H grid
│   │   └── funding_1h_clean.parquet # forward-filled 1H
│   ├── macro/
│   │   └── fred_daily_clean.parquet # todas as séries FRED aligned
│   └── spot/
│       └── btc_1h_clean.parquet     # com BB, RSI, MAs calculados
├── 02_features/     # Z-scores (output do gate_features)
```

O pipeline de features lê de `02_intermediate/`, nunca de `01_raw/`.
Um script `src/data/clean.py` transforma raw → intermediate.

## Resumo: ordem de aplicação

1. Implementar `fetch_with_retry()` (Fix 7) — usar em todos os módulos
2. Timezone enforcement em todos os módulos (Fix 1)
3. Coluna `timestamp` padronizada + `source` (Fix 8)
4. Resample 1H grid nos Futures (Fix 3)
5. Forward-fill funding (Fix 4)
6. Janela máxima nos parquets (Fix 2)
7. Fix FRED incremental (Fix 5)
8. News dedup robusto (Fix 6)
9. Validar monotonicidade (Fix 9)
10. Logging estruturado (Fix 10)
11. Criar 02_intermediate (Fix 11)
