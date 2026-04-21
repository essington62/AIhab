# Diagnosis Phase 1 — binance_spot.py Fix

**Data:** 2026-04-21 | **Status:** Diagnóstico completo, zero modificações

---

## 1. `append_and_save` behavior

```python
# src/data/utils.py:106-129
def append_and_save(new_df, filepath, freq="1h", ts_col="timestamp"):
    if filepath.exists():
        existing = pd.read_parquet(filepath)
        existing = enforce_utc(existing, ts_col)
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df.copy()
    save_with_window(combined, filepath, freq=freq, ts_col=ts_col)

# save_with_window:
#   df.sort_values(ts_col).drop_duplicates(subset=[ts_col]).tail(max_rows)
#   drop_duplicates() default = keep='first'
```

**Schema mismatch behavior:**
- `pd.concat` com schemas diferentes: pandas alinha por nome de coluna, preenche NaN para colunas ausentes em qualquer lado — **sem erro, silencioso**.
- `drop_duplicates(keep='first')` com sort estável → **EXISTING ganha** em colisão de timestamps (existing é concat primeiro).
- `MAX_ROWS["1h"] = 8760` (1 ano).
- **Sem validação de schema** — nenhuma checagem de colunas esperadas.

**Consequência para seed:**
Se chamarmos `append_and_save(cms_seed, RAW_PATH)`:
- existing = btc_1h.parquet (sem taker cols) → vence nos 2,462 rows overlap
- cms_seed rows do overlap perdem taker data → ficam NaN
- cms_seed rows pré-overlap (3,120 rows) mantêm taker data

---

## 2. Parquet atual do btc_AI

| Atributo | Valor |
|----------|-------|
| Rows | 2,462 |
| Período | 2026-01-08 21:00 UTC → 2026-04-21 10:00 UTC |
| Size | 0.29 MB |
| NaN | 0 em todas as colunas |

**Colunas e dtypes:**
```
timestamp     datetime64[ns, UTC]
open          float64
high          float64
low           float64
close         float64
volume        float64
num_trades    int64
source        object
```

**Observação crítica:** `taker_buy_base_vol` e `taker_buy_quote_vol` estão **ausentes** do parquet salvo, mesmo sendo parseados pela API em `binance_spot.py:44-46`. Causa: linha 48 faz select explícito que os exclui.

---

## 3. Simulação de import crypto-market-state → btc_AI

### Schema crypto-market-state (raw)
```
Index: DatetimeIndex(name='timestamp'), UTC
Columns: open, high, low, close, volume, quote_volume, trades,
         taker_buy_volume, taker_buy_quote_volume
dtypes: todos float64 (incluindo 'trades')
Rows: 5,582 | Período: 2025-09-01 → 2026-04-21 13:00 UTC | NaN: 0
```

### Após conversão para schema btc_AI
```python
cms_converted = cms.reset_index()
cms_converted.rename(columns={
    'trades':                   'num_trades',
    'taker_buy_volume':         'taker_buy_base_vol',
    'taker_buy_quote_volume':   'taker_buy_quote_vol',
})
cms_converted['source'] = 'binance_spot'
```

Colunas resultantes:
```
timestamp, open, high, low, close, volume, quote_volume,
num_trades, taker_buy_base_vol, taker_buy_quote_vol, source
```

### Diff de colunas

| Situação | Colunas |
|----------|---------|
| Só em cms_converted | `taker_buy_base_vol`, `taker_buy_quote_vol`, `quote_volume` |
| Só em btc_AI atual | nenhuma |
| Em ambos | `timestamp`, `open`, `high`, `low`, `close`, `volume`, `num_trades`, `source` |

### Overlap temporal

| Dataset | Período | Rows |
|---------|---------|------|
| btc_AI atual | 2026-01-08 21:00 → 2026-04-21 10:00 | 2,462 |
| cms_converted | 2025-09-01 00:00 → 2026-04-21 13:00 | 5,582 |
| Em ambos | 2026-01-08 21:00 → 2026-04-21 10:00 | 2,462 |
| Só em btc_AI | — | **0** |
| Só em cms (histórico antigo) | 2025-09-01 → 2026-01-08 20:00 | 3,120 |

**Achado crítico:** btc_AI é um subconjunto completo de cms. Nenhuma row de btc_AI está fora do cms. Podemos substituir btc_1h.parquet por cms_converted sem perda de dados.

### dtype issue: num_trades
- btc_AI: `int64` (sem NaN)
- cms `trades`: `float64` → após `astype('Int64')` → `Int64` (nullable)
- Concat `int64 + Int64` produz `Int64` ou `float64` — inconsistência
- **Fix:** usar `astype(int)` no seed (não há NaN na coluna)

---

## 4. Callsites que leem btc_1h.parquet

### Leitura direta de `data/01_raw/spot/btc_1h.parquet`

| Arquivo | Linha | Uso |
|---------|-------|-----|
| `src/data/binance_spot.py` | 60 | `get_last_timestamp()` — só lê `timestamp` |
| `src/data/clean.py` | 93 | `clean_spot()` — lê tudo, passa para `add_technical_indicators` |
| `scripts/check_eth_data_coverage.py` | 31 | Só conta rows (coverage check) |
| `scripts/adaptive_stops_v1_study.py` | ~60 | Lê `high`/`low` para ATR |
| `scripts/audit_taker_sources.py` | 88 | Auditoria de colunas taker |

### Leitura de `02_intermediate/spot/btc_1h_clean.parquet` (derivado)

| Arquivo | Uso |
|---------|-----|
| `src/dashboard/app.py:195` | Dashboard |
| `scripts/error_analysis_losers.py:54` | Análise |
| `scripts/mfe_mae_study_bot2.py:49` | Estudo |
| `scripts/backtest_bot2_v2.py:43` | Backtest |
| `scripts/analysis/estudo_adaptacao_fase1.py:58` | Análise |
| `scripts/adaptive_stops_v1_study.py:60` | Backtest |

### Colunas acessadas por callsite

- `clean_spot()`: passa df inteiro para `add_technical_indicators` → usa só `close` → **tolerante a novas colunas**
- `adaptive_stops_v1_study.py`: usa `high`, `low` do raw → **tolerante**
- `audit_taker_sources.py:88`: lista colunas do arquivo → **tolerante** (apenas inspeciona)
- Nenhum callsite acessa colunas por posição ou depende de schema fixo

**Conclusão:** adicionar `taker_buy_base_vol` e `taker_buy_quote_vol` ao parquet **não quebra nenhum callsite existente**.

---

## 5. `quote_volume`: usado?

**Em `src/data/binance_spot.py`:** parseado na linha 44 mas **dropado na linha 48** (select explícito exclui).

**Em qualquer outro lugar:** `grep -rn "quote_volume" src/ scripts/` retornou zero matches fora de `binance_spot.py` e `eth_ingestion.py`.

`eth_ingestion.py` tem referência própria (schema ETH separado), não lê btc_1h.parquet.

**Decisão: Opção A — IGNORAR `quote_volume`.** Não adicionar ao schema btc_AI.

---

## 6. ⚠️ Risks identificados

### R1 — Root bug em binance_spot.py linha 48
```python
# ATUAL (bug): taker cols são parseadas mas descartadas
df = df[["timestamp", "open", "high", "low", "close", "volume", "num_trades"]].copy()

# CORRETO (fix):
df = df[["timestamp", "open", "high", "low", "close", "volume",
         "num_trades", "taker_buy_base_vol", "taker_buy_quote_vol"]].copy()
```

### R2 — `append_and_save` não é ideal para seed
Usar `append_and_save(cms_seed, RAW_PATH)` faz existing ganhar nos overlaps. Como btc_AI não tem taker, as 2,462 rows overlap ficariam com NaN para taker. Para o seed, é melhor **não usar append_and_save** e fazer replace direto.

### R3 — dtype `num_trades`: float64 (cms) → int64 (btc_AI)
`trades` em cms está como `float64` (sem casas decimais, mas float). Converter com `astype(int)` antes de salvar. Se não, `num_trades` no parquet final seria `float64` ou `Int64`, divergindo do schema atual.

### R4 — cms tem 3 candles mais recentes que btc_AI
cms vai até 2026-04-21 13:00, btc_AI até 10:00. Após seed, o parquet terá 3 horas extras. A próxima run incremental vai partir de 13:00 normalmente. **Sem risco.**

### R5 — window de 8760 rows (MAX_ROWS["1h"])
cms_converted tem 5,582 rows — dentro do limite. Sem truncamento.

### R6 — `save_with_window` exige monotonicity
O assert `df[ts_col].is_monotonic_increasing` vai passar após sort+dedup. **Sem risco.**

---

## 7. 🎯 Recomendações para FASE 2

### Fix 1 — binance_spot.py (cirúrgico)

Linha 48, adicionar as duas colunas ao select e garantir tipos corretos:

```python
# binance_spot.py linha 48-51 (FASE 2)
df = df[["timestamp", "open", "high", "low", "close", "volume",
         "num_trades", "taker_buy_base_vol", "taker_buy_quote_vol"]].copy()
for col in ["open", "high", "low", "close", "volume",
            "taker_buy_base_vol", "taker_buy_quote_vol"]:
    df[col] = df[col].astype(float)
df["num_trades"] = df["num_trades"].astype(int)
df["source"] = "binance_spot"
```

### Fix 2 — Seed script one-shot (não usar append_and_save)

```python
# scripts/seed_btc_from_cms.py (FASE 2 — one-shot, não adicionar ao cron)
import pandas as pd
from pathlib import Path
from src.data.utils import save_with_window

CMS_PATH = Path("/Users/brown/Documents/MLGeral/crypto_v2/crypto-market-state/"
                "data/01_raw/spot/crypto/1h/BTCUSDT_1h.parquet")
OUT_PATH = Path("data/01_raw/spot/btc_1h.parquet")

cms = pd.read_parquet(CMS_PATH)
df = cms.reset_index().rename(columns={
    'trades':                  'num_trades',
    'taker_buy_volume':        'taker_buy_base_vol',
    'taker_buy_quote_volume':  'taker_buy_quote_vol',
})
df = df.drop(columns=['quote_volume'])  # não usado no btc_AI
df['num_trades'] = df['num_trades'].astype(int)
df['source'] = 'binance_spot'
df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)

# save_with_window faz sort+dedup+window+save
save_with_window(df, OUT_PATH, freq="1h")
print(f"Seed completo: {len(df)} rows → {OUT_PATH}")
print(f"Período: {df['timestamp'].min()} → {df['timestamp'].max()}")
print(f"Colunas: {df.columns.tolist()}")
```

**Por que replace direto (não append_and_save):** evita existing ganhar nos overlaps e perder taker data do período 2026-01-08+. O replace usa cms como única fonte; como btc_AI tem 0 rows exclusivos (subconjunto total de cms), não há perda.

### Fix 3 — Backup antes do seed

```bash
cp data/01_raw/spot/btc_1h.parquet data/01_raw/spot/btc_1h.parquet.bak_pre_seed
```

### Ordem de execução na FASE 2

```
1. Aplicar fix em binance_spot.py (linha 48)
2. Backup do parquet atual
3. Executar seed_btc_from_cms.py
4. Verificar parquet resultante (rows, colunas, NaN counts, período)
5. Executar clean.py → verificar btc_1h_clean.parquet não quebrou
6. Rodar tests/
7. Próxima run incremental (hourly_cycle) fará append com schema novo
```

---

## Decisões resolvidas

| Decisão | Escolha | Justificativa |
|---------|---------|---------------|
| `quote_volume` | **Ignorar** (Opção A) | Nenhum callsite usa; parsing já o dropava |
| Colunas faltando no btc_AI antigo | **Seed cms** (Opção B) | btc_AI é subconjunto de cms; 0 rows exclusivos em btc_AI |
| Como fazer o append | **Replace direto** via `save_with_window` | `append_and_save` faz existing ganhar → perderia taker do overlap |
| Naming final | **btc_AI** (`taker_buy_base_vol`, `taker_buy_quote_vol`) | Já definido; converter cms ao importar |
