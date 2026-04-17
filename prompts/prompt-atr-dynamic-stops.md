# Task: ATR-based Dynamic Stops (substituir stops fixos por stops baseados em volatilidade)

## Contexto

AI.hab em produção (EC2 São Paulo, Docker Compose).
Primeiro trade do paper trading fechou com +1.57% usando stops fixos (SG 2%, SL 3%, trailing 1.5%).
Problema: stops fixos não se adaptam à volatilidade. Em mercado calmo, 3% de SL é largo demais (devolve lucro). Em mercado volátil, 2% de SG é apertado demais (vende cedo).

**Solução:** calcular SG, SL e trailing dinamicamente com base no ATR (Average True Range) no momento da entrada. ATR já é calculado pelo sistema (`atr_14` em `src/features/technical.py`) e está disponível no contexto do trade (`entry_atr` no portfolio).

Spec completa do projeto em `CLAUDE.md` na raiz do repo btc_AI.

## Como funciona

### Fórmula

```
ATR% = ATR_14 / current_price × 100    (volatilidade relativa em %)

Stop Loss    = K_sl   × ATR%           (ex: 2.0 × 1.2% = 2.4%)
Take Profit  = K_tp   × ATR%           (ex: 1.5 × 1.2% = 1.8%)
Trailing     = K_trail × ATR%          (ex: 1.0 × 1.2% = 1.2%)
```

Os multiplicadores K são configuráveis em `parameters.yml`. O ATR% é calculado uma vez no momento da entrada e fixado para a duração do trade (não muda mid-trade).

### Exemplo prático

Com BTC a $75,000 e ATR_14 = $900:
```
ATR% = 900 / 75000 = 1.20%

SL = 2.0 × 1.20% = 2.40%  → $75,000 × 0.976 = $73,200
TP = 1.5 × 1.20% = 1.80%  → $75,000 × 1.018 = $76,350
Trailing = 1.0 × 1.20% = 1.20%  → trailing de 1.20% abaixo do high
```

Se volatilidade dobrar (ATR_14 = $1,800, ATR% = 2.40%):
```
SL = 2.0 × 2.40% = 4.80%  → stop mais largo, não toma stop por noise
TP = 1.5 × 2.40% = 3.60%  → alvo mais ambicioso em mercado volátil
Trailing = 1.0 × 2.40% = 2.40%  → trailing mais solto, deixa correr
```

### Fallback

Se ATR_14 não estiver disponível (dados insuficientes, API down), usar os stops fixos como fallback. Nunca ficar sem stop.

## Parte 1 — Atualizar parameters.yml

Em `conf/parameters.yml`, adicionar seção de dynamic stops e manter os fixos como fallback:

```yaml
execution:
  position_size_pct: 1.0
  paper_capital_usd: 10000.0

  # --- Dynamic stops (ATR-based) ---
  use_dynamic_stops: true       # true = ATR-based, false = fixed stops
  atr_multiplier_sl: 2.0        # K_sl: Stop Loss = K × ATR%
  atr_multiplier_tp: 1.5        # K_tp: Take Profit = K × ATR%
  atr_multiplier_trail: 1.0     # K_trail: Trailing = K × ATR%

  # Clamps (min/max para stops dinâmicos, segurança)
  min_stop_loss_pct: 0.01       # SL nunca menor que 1%
  max_stop_loss_pct: 0.06       # SL nunca maior que 6%
  min_take_profit_pct: 0.01     # TP nunca menor que 1%
  max_take_profit_pct: 0.06     # TP nunca maior que 6%
  min_trailing_pct: 0.008       # Trailing nunca menor que 0.8%
  max_trailing_pct: 0.04        # Trailing nunca maior que 4%

  # --- Fixed stops (fallback se ATR indisponível) ---
  stop_loss_pct: 0.03           # 3% (usado se use_dynamic_stops=false ou ATR indisponível)
  take_profit_pct: 0.02         # 2%
  trailing_stop_pct: 0.015      # 1.5%
```

## Parte 2 — Criar função de cálculo de stops dinâmicos

Em `src/trading/execution.py`, adicionar:

```python
def compute_dynamic_stops(entry_price: float, atr_14: float, params: dict) -> dict:
    """
    Calcula SL, TP e trailing dinamicamente baseado no ATR.
    
    Args:
        entry_price: preço de entrada
        atr_14: ATR de 14 períodos (em USD, não percentual)
        params: execution params do parameters.yml
        
    Returns:
        dict com {
            'stop_loss_pct': float,      # percentual do SL
            'take_profit_pct': float,    # percentual do TP
            'trailing_stop_pct': float,  # percentual do trailing
            'stop_loss_price': float,    # preço do SL
            'take_profit_price': float,  # preço do TP
            'atr_pct': float,            # ATR% usado no cálculo
            'stops_mode': str,           # 'dynamic' ou 'fixed'
        }
    """
    # Calcular ATR como percentual do preço
    atr_pct = atr_14 / entry_price  # ex: 900/75000 = 0.012 (1.2%)
    
    # Aplicar multiplicadores
    sl_pct = params['atr_multiplier_sl'] * atr_pct
    tp_pct = params['atr_multiplier_tp'] * atr_pct
    trail_pct = params['atr_multiplier_trail'] * atr_pct
    
    # Clampar dentro dos limites de segurança
    sl_pct = max(params['min_stop_loss_pct'], min(params['max_stop_loss_pct'], sl_pct))
    tp_pct = max(params['min_take_profit_pct'], min(params['max_take_profit_pct'], tp_pct))
    trail_pct = max(params['min_trailing_pct'], min(params['max_trailing_pct'], trail_pct))
    
    return {
        'stop_loss_pct': round(sl_pct, 6),
        'take_profit_pct': round(tp_pct, 6),
        'trailing_stop_pct': round(trail_pct, 6),
        'stop_loss_price': round(entry_price * (1 - sl_pct), 2),
        'take_profit_price': round(entry_price * (1 + tp_pct), 2),
        'atr_pct': round(atr_pct, 6),
        'stops_mode': 'dynamic',
    }
```

## Parte 3 — Modificar execute_entry()

Em `src/trading/execution.py`, a função `execute_entry()` atualmente calcula stops assim:

```python
# ANTES (fixo)
portfolio["stop_loss_price"] = round(current_price * (1 - sl_pct), 2)
portfolio["take_profit_price"] = round(current_price * (1 + tp_pct), 2)
```

Modificar para:

```python
# DEPOIS (dinâmico com fallback)
atr_14 = kwargs.get("atr_14")  # passado pelo paper_trader no momento da entry

if params.get("use_dynamic_stops", False) and atr_14 and atr_14 > 0:
    dynamic = compute_dynamic_stops(current_price, atr_14, params)
    sl_pct = dynamic['stop_loss_pct']
    tp_pct = dynamic['take_profit_pct']
    trail_pct = dynamic['trailing_stop_pct']
    stops_mode = 'dynamic'
    atr_pct = dynamic['atr_pct']
    logger.info(
        f"DYNAMIC STOPS: ATR%={atr_pct:.3%} → SL={sl_pct:.2%} TP={tp_pct:.2%} Trail={trail_pct:.2%}"
    )
else:
    sl_pct = params["stop_loss_pct"]
    tp_pct = params["take_profit_pct"]
    trail_pct = params["trailing_stop_pct"]
    stops_mode = 'fixed'
    atr_pct = None
    logger.info(f"FIXED STOPS (ATR unavailable): SL={sl_pct:.2%} TP={tp_pct:.2%}")

portfolio["stop_loss_price"] = round(current_price * (1 - sl_pct), 2)
portfolio["take_profit_price"] = round(current_price * (1 + tp_pct), 2)
portfolio["trailing_stop_pct_actual"] = trail_pct   # salvar o trailing % usado
portfolio["stops_mode"] = stops_mode
portfolio["entry_atr_pct"] = atr_pct
```

**IMPORTANTE:** A assinatura de `execute_entry()` precisa aceitar o `atr_14` como parâmetro adicional. Usar `**kwargs` ou adicionar parâmetro explícito. Verificar todos os call sites.

## Parte 4 — Modificar check_stops()

Em `src/trading/execution.py`, a função `check_stops()` atualmente lê `trailing_stop_pct` dos params globais:

```python
# ANTES
params = get_params()["execution"]
trailing_pct = params["trailing_stop_pct"]
```

Modificar para usar o trailing_pct que foi calculado na entrada e salvo no portfolio:

```python
# DEPOIS
params = get_params()["execution"]
# Usar trailing calculado na entrada (dinâmico), com fallback para params global
trailing_pct = portfolio.get("trailing_stop_pct_actual") or params["trailing_stop_pct"]
```

Isso garante que o trailing % não muda mid-trade (é fixado no momento da entrada).

## Parte 5 — Modificar paper_trader.py (call site da entry)

Em `src/trading/paper_trader.py`, onde `execute_entry()` é chamado, passar o ATR:

```python
# No run_cycle(), quando decide ENTER:
technical = get_latest_technical()
atr_14 = technical.get("atr_14")

portfolio = execute_entry(current_price, portfolio, atr_14=atr_14)
```

Também em `_init_trade_tracking()`, registrar o stops_mode e atr_pct:

```python
portfolio["entry_stops_mode"] = portfolio.get("stops_mode", "fixed")
portfolio["entry_atr_pct"] = portfolio.get("entry_atr_pct")
```

## Parte 6 — Atualizar _build_trade_record()

Em `src/trading/paper_trader.py`, adicionar campos ao trade record:

```python
completed_trade = {
    # ... campos existentes ...
    
    # Dynamic stops info
    "stops_mode": portfolio.get("stops_mode", "fixed"),
    "entry_atr_pct": portfolio.get("entry_atr_pct"),
    "actual_stop_loss_pct": portfolio.get("entry_stop_loss_pct"),
    "actual_take_profit_pct": portfolio.get("entry_stop_gain_pct"),
    "actual_trailing_pct": portfolio.get("trailing_stop_pct_actual"),
}
```

## Parte 7 — Atualizar dashboard

Em `src/dashboard/app.py`, no card de Paper Trading, mostrar os stops dinâmicos:

Onde atualmente exibe:
```
Stop Loss: $74,275 | Take Profit: $75,273
```

Modificar para incluir o modo:
```
Stop Loss: $73,200 (2.4% ATR) | Take Profit: $76,350 (1.8% ATR) | Trailing: 1.2% ATR
```

Ou se fixo:
```
Stop Loss: $72,750 (3.0% fixed) | Take Profit: $75,600 (2.0% fixed) | Trailing: 1.5% fixed
```

Mostrar também o ATR% usado na decisão.

## Parte 8 — Testes

### TestComputeDynamicStops
1. `test_dynamic_stops_normal` — ATR normal → calcula SL/TP/trailing corretos
2. `test_dynamic_stops_high_volatility` — ATR alto → stops mais largos (mas dentro do max clamp)
3. `test_dynamic_stops_low_volatility` — ATR baixo → stops apertados (mas dentro do min clamp)
4. `test_dynamic_stops_clamp_max` — ATR extremo → clampado no max (6%)
5. `test_dynamic_stops_clamp_min` — ATR minúsculo → clampado no min (1%)

### TestExecuteEntryDynamic
6. `test_entry_with_atr_dynamic` — use_dynamic_stops=true + ATR disponível → stops dinâmicos no portfolio
7. `test_entry_without_atr_fallback` — use_dynamic_stops=true + ATR=None → fallback para fixos
8. `test_entry_dynamic_disabled` — use_dynamic_stops=false → usa fixos mesmo com ATR disponível

### TestCheckStopsDynamic
9. `test_check_stops_uses_portfolio_trailing` — trailing_pct vem do portfolio, não dos params globais
10. `test_check_stops_trailing_fallback` — se trailing_pct_actual não existe no portfolio, usa params global

### TestIntegration
11. `test_full_cycle_dynamic_stops` — entry com ATR → stops calculados → check_stops usa trailing correto → exit com registro completo

## Entregáveis

1. `conf/parameters.yml` — seção de dynamic stops (multiplicadores K, clamps, flag use_dynamic_stops)
2. `src/trading/execution.py` — `compute_dynamic_stops()`, `execute_entry()` modificado, `check_stops()` modificado
3. `src/trading/paper_trader.py` — passa ATR na entry, registra stops_mode no trade record
4. `src/dashboard/app.py` — mostra stops dinâmicos com ATR%
5. `tests/test_execution.py` ou `tests/test_paper_trader.py` — 11 novos testes
6. Commit + push:
   ```bash
   git add -A && git commit -m "feat: ATR-based dynamic stops (adaptive SL/TP/trailing)"
   git push origin master:main
   ```

## Restrições

- **NÃO alterar** lógica de gates, scoring, ou regime
- **NÃO alterar** lógica de MAE/MFE tracking
- **NÃO alterar** check_stops_only() (15min cycle) — ele já usa check_stops() que será atualizado
- **Fallback obrigatório** — se ATR indisponível, SEMPRE usar stops fixos. Nunca ficar sem stop.
- **Stops fixados na entrada** — ATR% calculado UMA VEZ na entrada, não muda durante o trade
- **Clamps de segurança** — SL/TP nunca menor que 1% nem maior que 6%
- **Backward compatible** — `use_dynamic_stops: false` mantém comportamento atual (stops fixos)
- **Manter leitura de parâmetros de conf/parameters.yml** — multiplicadores K e clamps vêm do YAML
