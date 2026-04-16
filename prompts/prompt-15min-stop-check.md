# Task: Separar ciclo de stops (15min) do ciclo completo (1h)

## Contexto

AI.hab em produção (EC2 São Paulo, Docker Compose).
Primeiro trade do paper trading fechou com +1.57%.
Problema identificado: o ciclo atual roda a cada 1h (scoring + stops juntos). Com ciclo de 1h, o preço pode ultrapassar SG ou SL e o sistema só detecta até 59 minutos depois. No primeiro trade, o preço passou do Take Profit ($75,273) mas o sistema não vendeu na hora — só no ciclo seguinte.

**Solução:** separar em dois ciclos:
- **1h** — scoring completo (gates, regime, decisão ENTER/HOLD/EXIT) — já existe
- **15min** — só checagem de stops (SG, SL, trailing) — novo, leve, só lê preço atual e compara com stops configurados

Spec completa do projeto em `CLAUDE.md` na raiz do repo btc_AI.

## Parte 1 — Criar função check_stops_only()

Em `src/trading/paper_trader.py`, criar nova função que:

1. Verifica se há posição aberta (lê `portfolio_state.json`)
2. Se não há posição → return imediatamente (nada a fazer)
3. Se há posição aberta:
   a. Busca preço atual do BTC (mesmo método usado no run_cycle)
   b. Lê os stops configurados de `conf/parameters.yml` (stop_gain, stop_loss, trailing_stop)
   c. Lê dados da posição: entry_price, trailing_high
   d. Checa condições de saída:
      - **Stop Gain:** preço atual ≥ entry_price × (1 + stop_gain) → EXIT
      - **Stop Loss:** preço atual ≤ entry_price × (1 - stop_loss) → EXIT
      - **Trailing Stop:** preço atual ≤ trailing_high × (1 - trailing_stop) → EXIT
   e. Se nenhum stop atingido:
      - Atualiza trailing_high = max(trailing_high, preço_atual)
      - **Atualiza MAE/MFE** (chamar _update_excursions existente)
      - Salva portfolio_state.json atualizado
      - Loga: `"[STOPS-15m] HOLD | price=$XX,XXX | SG=$XX,XXX | SL=$XX,XXX | trailing_high=$XX,XXX"`
   f. Se algum stop atingido:
      - Chama _build_trade_record() para montar o registro com MAE/MFE
      - Executa exit (execute_exit ou equivalente)
      - Chama _save_completed_trade() para persistir em trades.parquet
      - Loga: `"[STOPS-15m] EXIT by {reason} | price=$XX,XXX | return={pct}%"`

```python
def check_stops_only(self) -> dict:
    """
    Checagem leve de stops — roda a cada 15min.
    Não recalcula scores, gates, regime. Só verifica preço vs stops.
    Se nenhum stop atingido, atualiza trailing_high e MAE/MFE.
    Se stop atingido, executa exit completo.
    
    Returns:
        dict com {action: 'hold'|'exit', reason: str, price: float, ...}
    """
```

### IMPORTANTE — Reutilizar código existente

A função check_stops_only() NÃO deve duplicar lógica. Deve:
- Usar o mesmo método de get_current_price() que run_cycle usa
- Usar o mesmo _update_excursions() existente
- Usar o mesmo _build_trade_record() e _save_completed_trade() existentes
- Usar o mesmo execute_exit() existente
- A única diferença é: NÃO recalcula scores/gates/regime

### Tratamento de erros

Se não conseguir buscar preço (API down, timeout):
- Logar warning: `"[STOPS-15m] WARN: could not fetch price, skipping cycle"`
- NÃO executar nenhum stop
- Return com action='error'
- O próximo ciclo de 15min (ou o ciclo completo de 1h) tentará de novo

## Parte 2 — Criar script scripts/check_stops.py

```python
#!/usr/bin/env python
"""Checagem rápida de stops — roda a cada 15min via cron."""

import sys
import logging
from src.trading.paper_trader import PaperTrader
from src.config import load_params

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    logger = logging.getLogger("check_stops")
    
    params = load_params()
    trader = PaperTrader(params)
    
    result = trader.check_stops_only()
    
    if result['action'] == 'hold':
        logger.info(f"HOLD — price=${result['price']:,.0f} trailing_high=${result.get('trailing_high', 0):,.0f}")
    elif result['action'] == 'exit':
        logger.info(f"EXIT by {result['reason']} — price=${result['price']:,.0f} return={result.get('return_pct', 0):.2%}")
    elif result['action'] == 'no_position':
        logger.info("No open position — nothing to check")
    elif result['action'] == 'error':
        logger.warning(f"Could not check stops: {result.get('error', 'unknown')}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
```

## Parte 3 — Atualizar hourly_cycle.sh

O ciclo horário (`scripts/hourly_cycle.sh`) continua rodando a cada hora com o scoring completo. Mas agora, quando ele roda, ele TAMBÉM faz a checagem de stops (como parte do run_cycle normal). Os dois ciclos são complementares:

- **15min:** só stops (leve, ~2s)
- **1h:** scoring completo + stops (pesado, ~30s com DeepSeek)

O hourly_cycle.sh NÃO precisa mudar. Ele já faz checagem de stops como parte do run_cycle.

## Parte 4 — Cron no Docker (supercronic)

Atualizar `crontab` no container Docker para incluir o ciclo de 15min:

```
# Ciclo completo (scoring + stops) — a cada hora
0 * * * * /app/scripts/hourly_cycle.sh >> /app/logs/hourly.log 2>&1

# Checagem rápida de stops — a cada 15min (nos minutos 15, 30, 45)
# NÃO roda no minuto 0 porque o ciclo completo já cobre
15,30,45 * * * * cd /app && python3 scripts/check_stops.py >> /app/logs/stops.log 2>&1
```

**Nota:** O cron roda nos minutos 15, 30, 45 (NÃO no minuto 0, que é quando o ciclo completo roda). Isso evita conflito de dois processos acessando portfolio_state.json ao mesmo tempo.

Se o projeto usa supercronic (verificar Dockerfile), o formato é o mesmo. Se usa crontab padrão, adaptar.

## Parte 5 — Lock file para evitar race condition

Criar um lock simples para evitar que check_stops e hourly_cycle rodem ao mesmo tempo:

```python
# Em src/trading/paper_trader.py ou utils
import fcntl
import os

LOCK_FILE = "/tmp/aihab_trading.lock"

def acquire_lock():
    """Tenta adquirir lock exclusivo. Retorna file handle ou None."""
    try:
        lock_fd = open(LOCK_FILE, 'w')
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return lock_fd
    except (IOError, OSError):
        return None

def release_lock(lock_fd):
    """Libera o lock."""
    if lock_fd:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        lock_fd.close()
```

Usar no check_stops.py:
```python
lock = acquire_lock()
if lock is None:
    logger.info("Another cycle is running — skipping this check")
    return 0
try:
    result = trader.check_stops_only()
    # ... process result
finally:
    release_lock(lock)
```

Usar também no hourly_cycle (ou no run_cycle dentro do paper_trader).

## Parte 6 — Logging separado

Os logs do check_stops devem ir para arquivo separado para facilitar debug:
- `logs/stops.log` — checagem de 15min
- `logs/hourly.log` — ciclo completo (já existe)

Formato do log do check_stops:
```
2026-04-16 20:45:00 [check_stops] HOLD — price=$75,338 trailing_high=$75,418
2026-04-16 21:00:00 [hourly_cycle] Full cycle — score=3.5 regime=Sideways action=HOLD
2026-04-16 21:15:00 [check_stops] HOLD — price=$75,200 trailing_high=$75,418
2026-04-16 21:30:00 [check_stops] EXIT by trailing_stop — price=$74,300 return=+0.19%
```

## Parte 7 — Testes

Adicionar em `tests/test_paper_trader.py`:

### TestCheckStopsOnly
1. `test_check_stops_no_position` — sem posição aberta → retorna action='no_position'
2. `test_check_stops_hold` — preço dentro dos stops → retorna action='hold', atualiza trailing_high
3. `test_check_stops_stop_gain` — preço acima de SG → retorna action='exit', reason='stop_gain'
4. `test_check_stops_stop_loss` — preço abaixo de SL → retorna action='exit', reason='stop_loss'
5. `test_check_stops_trailing` — preço caiu do trailing_high mais que trailing_pct → action='exit', reason='trailing_stop'
6. `test_check_stops_updates_trailing_high` — preço subiu → trailing_high atualizado no portfolio
7. `test_check_stops_updates_excursions` — MAE/MFE atualizados durante hold
8. `test_check_stops_api_error` — preço indisponível → action='error', nenhum stop executado
9. `test_check_stops_lock_conflict` — lock já adquirido → skip graceful

### TestLockMechanism
1. `test_acquire_release_lock` — lock funciona corretamente
2. `test_lock_conflict` — segundo acquire retorna None

## Entregáveis

1. `src/trading/paper_trader.py` — nova função `check_stops_only()` + lock helpers
2. `scripts/check_stops.py` — script standalone para cron
3. Crontab/supercronic atualizado com ciclo de 15min
4. `tests/test_paper_trader.py` — 11 novos testes
5. Logs separados: `logs/stops.log`
6. Commit + push:
   ```bash
   git add -A && git commit -m "feat: add 15min stop check cycle (separate from hourly scoring)"
   git push origin master:main
   ```

## Restrições

- **NÃO alterar** o run_cycle() existente — ele continua funcionando exatamente como está
- **NÃO duplicar** lógica de stops — check_stops_only() deve reutilizar as funções existentes
- **NÃO rodar** check_stops no minuto 0 — evita conflito com hourly_cycle
- **Lock obrigatório** — portfolio_state.json não pode ser escrito por dois processos
- **Falha silenciosa** — se preço não disponível ou lock ocupado, logar e pular (não crashar)
- **Manter leitura de parâmetros de conf/parameters.yml** — stops vêm do YAML, não hardcoded
