# FASE 4 Changelog — Shadow Mode Taker_Z Filter

**Data:** 2026-04-21
**Objetivo:** Validar filtro taker_z out-of-sample antes de ativar em produção

## Novos arquivos

### `src/trading/shadow_filters.py`
- `evaluate_taker_z_shadow(entry_time, trade_id, bot_origin)` — avalia `taker_z < -1.0` para 4h CoinGlass e 1h Binance
- Anti look-ahead: usa `timestamp < entry_time` (strict)
- Persiste em JSONL append-only (`data/08_shadow/taker_z_shadow_log.jsonl`)
- Zero impacto no fluxo de trading

### `scripts/analyze_shadow_log.py`
- Carrega shadow log + trades.parquet
- Calcula ratio losers/winners dos bloqueios virtuais
- Compara com histórico (1.43)
- Decisão automática: ESPERAR / CONTINUAR SHADOW / PODE ATIVAR / REJEITAR

## Modificações

### `src/trading/paper_trader.py`
- Import adicionado: `from src.trading.shadow_filters import evaluate_taker_z_shadow`
- Bot 1 entry (após confirmação): try/except block → `evaluate_taker_z_shadow`
- Bot 2 entry (após confirmação): try/except block → `evaluate_taker_z_shadow`
- 8 linhas efetivas + 1 import

### `src/features/gate_features.py` (FASE 3, já deployado)
- `taker_z_1h` adicionado em `gate_zscores.parquet`

### `conf/parameters.yml` (FASE 3, já deployado)
- `taker_1h: 168` adicionado em `zscore_windows`

## Log

- `data/08_shadow/taker_z_shadow_log.jsonl` (append-only, cresce a cada trade)

Campos por entry:
```json
{
  "timestamp_utc": "...",
  "trade_id": "uuid",
  "bot_origin": "bot1|bot2",
  "entry_time": "...",
  "filter_version": "v1",
  "threshold": -1.0,
  "status": "ok|error_no_gate_zscores",
  "prev_candle_time": "...",
  "taker_z_4h": -2.11,
  "taker_z_1h": -1.11,
  "would_block_4h": true,
  "would_block_1h": true,
  "both_agree_block": true,
  "disagreement": false
}
```

## Monitoramento

```bash
# Semanal
python scripts/analyze_shadow_log.py
cat prompts/shadow_analysis_report.md
```

## Rollback (desativar shadow)

Comentar blocos `# SHADOW MODE` em `paper_trader.py` (2 blocos try/except).
Módulo `shadow_filters.py` pode ficar — sem impacto se não chamado.

```bash
# OU rollback via git:
git revert HEAD
git push origin main
# AWS: git pull && docker-compose restart aihab-app
```

## Critérios para ativação (4-6 semanas)

| Ratio losers/winners bloqueados | Decisão |
|--------------------------------|---------|
| ≥ 1.30 com 100+ trades | ✅ Ativar 4h em produção |
| 1.0–1.3 | ⚠️ Continuar shadow |
| < 1.0 | ❌ Rejeitar — filtro degradado |

## Próximo passo (FASE 5 — quando ativar)

Mudar `shadow_filters.py` de logging-only para bloqueio real:
1. Adicionar flag `ACTIVE_BLOCK = True`
2. Retornar `{"should_block": True}` para paper_trader verificar
3. Paper_trader: se `should_block`, não executar entrada + log `BLOCKED_TAKER_Z`
