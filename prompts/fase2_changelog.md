# FASE 2 Changelog — Import + Fix binance_spot

**Data:** 2026-04-21
**Objetivo:** Habilitar taker volume em pipeline btc_AI

## Mudanças

### Arquivos renomeados
- `data/01_raw/spot/btc_1h.parquet` → `data/01_raw/spot/btc_1h_sem_taker_backup.parquet`
  - Backup do parquet antigo (2,462 rows, sem taker)

### Arquivos novos
- `data/01_raw/spot/btc_1h.parquet`
  - Importado do crypto-market-state (5,583 rows após 1 incremental, com taker em todos)
  - Período: 2025-09-01 00:00 UTC → 2026-04-21 14:00 UTC
  - Schema: timestamp, open, high, low, close, volume, num_trades, taker_buy_base_vol, taker_buy_quote_vol, source
- `scripts/import_cms_1h_to_btc_ai.py` (one-shot — não adicionar ao cron)

### Código modificado
- `src/data/binance_spot.py` — `fetch_spot_klines()` linhas 48-50
  - Incluído `taker_buy_base_vol`, `taker_buy_quote_vol` no select de colunas
  - Incluído taker cols no cast para float

## Resultados da validação

| Teste | Status |
|-------|--------|
| Schema parquet importado (expected_cols, dtypes, zero NaN) | ✅ |
| `clean_spot()` — 5,582 rows, `btc_1h_clean.parquet` com taker | ✅ |
| `python -m src.data.binance_spot` — incremental +1 row com taker | ✅ |
| Últimas 24 rows sem NaN em taker | ✅ |

## Impacto

### Antes
- btc_1h.parquet: 2,462 rows, período 2026-01-08 → 2026-04-21, sem taker
- Cron hourly puxava OHLCV + descartava taker silenciosamente

### Depois
- btc_1h.parquet: 5,583+ rows, período 2025-09-01 → atual, taker em todos
- Cron hourly puxa OHLCV + salva taker_buy_base_vol e taker_buy_quote_vol
- `btc_1h_clean.parquet` propaga taker para features pipeline

## Rollback (se necessário)

```bash
rm data/01_raw/spot/btc_1h.parquet
mv data/01_raw/spot/btc_1h_sem_taker_backup.parquet data/01_raw/spot/btc_1h.parquet
git checkout src/data/binance_spot.py
rm scripts/import_cms_1h_to_btc_ai.py
```

## Próximos passos (FASE 3)

1. Adicionar `taker_z_1h` em features pipeline (z-score do taker_buy_base_vol nativo 1h)
2. Re-rodar filter_validation com taker 1h nativo vs taker 4h CoinGlass
3. Decidir qual vence e ir para shadow mode em paper_trader
