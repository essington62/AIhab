# Task: Implementar Reversal Filter — condições adicionais de entrada (RSI < 35 + ret_1d > -1%)

## Contexto

AI.hab em produção (EC2 São Paulo, Docker Compose).
Estudo estatístico detalhado (Jan 2024 — Abr 2026, 833 dias, 146 sinais de score > 2.5) demonstrou que o **score combinado** do gate scoring **sozinho** não gera edge suficiente para ser lucrativo (WR 42%, PF 0.96, total -5.5% com SG 2%/SL 1.5%).

Porém, adicionando dois filtros técnicos de reversão, o sistema se transforma:

**Score > 2.5 + RSI < 35 + ret_1d > -1%** (com SG 2% / SL 1.5%):
- 41 trades | WR 63% | PF 2.31 | Total +29.5% | MaxDD -4.5%

A lógica é:
- **Score > threshold** = "os fundamentos dizem que está barato" (gates macro, posicionamento, liquidez, sentimento)
- **RSI < 35** = "está profundamente oversold" (confirmação técnica de capitulação)
- **ret_1d > -1%** = "não é faca caindo" (o sell-off diminuiu, possível fundo)

Sem o filtro RSI+ret_1d, o scoring system entra em trades durante quedas ativas onde o preço continua caindo. O filtro espera pela desaceleração da queda.

Spec completa do projeto em `CLAUDE.md` na raiz do repo btc_AI.

## Arquitetura da mudança

O filtro **NÃO** altera o scoring pipeline (`gate_scoring.py`). O scoring continua gerando ENTER/HOLD/BLOCK como antes. A mudança é no `paper_trader.py` — após receber `signal == "ENTER"`, aplica os filtros adicionais antes de executar a entrada.

Fluxo atual:
```
scoring → signal=ENTER → execute_entry()
```

Fluxo novo:
```
scoring → signal=ENTER → check_reversal_filter(rsi, ret_1d) → se passa → execute_entry()
                                                              → se não passa → HOLD (log: FILTERED)
```

Isso mantém o scoring 100% intacto e adiciona uma camada de filtragem configurável.

## Parte 1 — Calcular ret_1d no get_latest_technical()

Em `src/features/technical.py`, na função `get_latest_technical()`, o DataFrame `df` já está carregado com todas as candles 1h. Calcular o retorno das últimas 24h:

```python
# Dentro de get_latest_technical(), após df = df.sort_values("timestamp"):

# Retorno 1d (24 candles de 1h)
if len(df) >= 25:
    close_now = float(df.iloc[-1]["close"])
    close_24h_ago = float(df.iloc[-25]["close"])  # 24h antes = 25ª candle de trás
    ret_1d = (close_now - close_24h_ago) / close_24h_ago
else:
    ret_1d = None

# Adicionar ao dict result:
result["ret_1d"] = round(ret_1d, 6) if ret_1d is not None else None
```

**ATENÇÃO:** Como os dados são candles de 1h, 24 candles atrás = 1 dia. Usar `df.iloc[-25]` (25 posições atrás = 24 intervalos de 1h).

Atualizar também o log:
```python
logger.info(
    f"Latest technical: close={result['close']:.2f}, "
    f"bb_pct={result['bb_pct']:.3f}, rsi={result['rsi_14']:.1f}, "
    f"ret_1d={result['ret_1d']:.4f}" if result.get('ret_1d') is not None else
    f"Latest technical: close={result['close']:.2f}, "
    f"bb_pct={result['bb_pct']:.3f}, rsi={result['rsi_14']:.1f}, ret_1d=N/A"
)
```

## Parte 2 — Atualizar parameters.yml

Adicionar seção de reversal filter em `conf/parameters.yml`:

```yaml
# ---------------------------------------------------------------------------
# Reversal filter (entry confirmation — applied AFTER scoring)
# ---------------------------------------------------------------------------
reversal_filter:
  enabled: true                   # true = filtro ativo, false = aceita qualquer ENTER do scoring
  rsi_max: 35                     # RSI deve estar ABAIXO deste valor (oversold confirmation)
  ret_1d_min: -0.01              # Retorno 1d deve estar ACIMA de -1% (não é faca caindo)
  rsi_extreme_override: 25        # RSI abaixo deste valor = capitulação extrema → ignora ret_1d_min
                                  # (set to 0 to disable override)
  # Nota: score threshold já é gerenciado pelo dynamic_threshold no scoring pipeline.
  # O reversal filter COMPLEMENTA o scoring, não substitui.
```

## Parte 3 — Criar função check_reversal_filter()

Em `src/trading/paper_trader.py`, criar nova função:

```python
def check_reversal_filter(technical: dict, params: dict) -> dict:
    """
    Verifica se condições de reversão estão atendidas para confirmar entrada.
    
    O scoring pipeline já decidiu ENTER (score >= threshold).
    Este filtro adiciona confirmação técnica:
    - RSI < rsi_max (oversold confirmation)
    - ret_1d > ret_1d_min (não é faca caindo)
    - Exceção: RSI < rsi_extreme_override ignora ret_1d (capitulação extrema)
    
    Acumula TODOS os motivos de falha (não para no primeiro).
    
    Args:
        technical: dict com indicadores técnicos (rsi_14, ret_1d)
        params: parameters.yml completo
        
    Returns:
        dict com {
            'passed': bool,
            'reason': str,        # motivo(s) se não passou, separados por " & "
            'rsi': float|None,
            'ret_1d': float|None,
            'rsi_max': float,
            'ret_1d_min': float,
        }
    """
    rf = params.get("reversal_filter", {})
    
    # Se filtro desabilitado, sempre passa
    if not rf.get("enabled", False):
        return {
            "passed": True,
            "reason": "filter_disabled",
            "rsi": technical.get("rsi_14"),
            "ret_1d": technical.get("ret_1d"),
            "rsi_max": None,
            "ret_1d_min": None,
        }
    
    rsi_max = rf.get("rsi_max", 35)
    ret_1d_min = rf.get("ret_1d_min", -0.01)
    rsi_extreme = rf.get("rsi_extreme_override", 25)  # 0 = disabled
    
    rsi = technical.get("rsi_14")
    ret_1d = technical.get("ret_1d")
    
    # Se dados indisponíveis, NÃO entra (segurança)
    if rsi is None or ret_1d is None:
        return {
            "passed": False,
            "reason": f"FILTER_DATA_MISSING (rsi={'N/A' if rsi is None else f'{rsi:.1f}'}, ret_1d={'N/A' if ret_1d is None else f'{ret_1d:.4f}'})",
            "rsi": rsi,
            "ret_1d": ret_1d,
            "rsi_max": rsi_max,
            "ret_1d_min": ret_1d_min,
        }
    
    # Acumular TODOS os motivos de falha (debug gold)
    reasons = []
    
    # Checar RSI
    if rsi >= rsi_max:
        reasons.append(f"RSI_TOO_HIGH ({rsi:.1f} >= {rsi_max})")
    
    # Checar ret_1d — COM exceção para capitulação extrema
    if ret_1d <= ret_1d_min:
        if rsi_extreme > 0 and rsi < rsi_extreme:
            # Capitulação extrema: RSI tão baixo que ignora a queda forte
            logger.info(
                f"EXTREME_CAPITULATION: RSI={rsi:.1f} < {rsi_extreme} → "
                f"overriding ret_1d filter (ret_1d={ret_1d:.4f})"
            )
        else:
            reasons.append(f"FALLING_KNIFE (ret_1d={ret_1d:.4f} <= {ret_1d_min})")
    
    if reasons:
        return {
            "passed": False,
            "reason": " & ".join(reasons),
            "rsi": rsi,
            "ret_1d": ret_1d,
            "rsi_max": rsi_max,
            "ret_1d_min": ret_1d_min,
        }
    
    return {
        "passed": True,
        "reason": "reversal_confirmed",
        "rsi": rsi,
        "ret_1d": ret_1d,
        "rsi_max": rsi_max,
        "ret_1d_min": ret_1d_min,
    }
```

## Parte 4 — Modificar run_cycle() para aplicar o filtro

Em `src/trading/paper_trader.py`, na seção "10. Execution", modificar a decisão de entrada (linhas ~585-591):

```python
# ANTES (linha 585-591):
        # Entry decision
        if result["signal"] == "ENTER" and not portfolio["has_position"]:
            atr_14 = technical.get("atr_14")
            portfolio = execute_entry(current_price, portfolio, atr_14=atr_14)
            _init_trade_tracking(portfolio, result, regime, technical, zscores)
            atomic_write_json(portfolio, get_path("portfolio_state"))

# DEPOIS:
        # Entry decision (scoring says ENTER → apply reversal filter)
        if result["signal"] == "ENTER" and not portfolio["has_position"]:
            rf_check = check_reversal_filter(technical, params)
            
            if rf_check["passed"]:
                atr_14 = technical.get("atr_14")
                portfolio = execute_entry(current_price, portfolio, atr_14=atr_14)
                _init_trade_tracking(portfolio, result, regime, technical, zscores)
                # Stamp reversal filter context
                portfolio["entry_ret_1d"] = rf_check["ret_1d"]
                portfolio["entry_filter_passed"] = True
                atomic_write_json(portfolio, get_path("portfolio_state"))
                logger.info(
                    f"ENTRY CONFIRMED: score={result.get('score', 0):.3f} | "
                    f"RSI={rf_check['rsi']:.1f} (<{rf_check['rsi_max']}) | "
                    f"ret_1d={rf_check['ret_1d']:.4f} (>{rf_check['ret_1d_min']})"
                )
            else:
                logger.info(
                    f"ENTRY FILTERED: score={result.get('score', 0):.3f} | "
                    f"RSI={rf_check['rsi']:.1f} | ret_1d={rf_check['ret_1d']:.4f} → "
                    f"{rf_check['reason']}"
                )
                # Override signal para log/dashboard
                result["signal"] = "FILTERED"
                result["filter_reason"] = rf_check["reason"]
                result["filter_rsi"] = rf_check["rsi"]
                result["filter_ret_1d"] = rf_check["ret_1d"]
```

**IMPORTANTE:** O sinal muda de "ENTER" para "FILTERED" no result dict — isso permite que o dashboard e os logs mostrem que o scoring queria entrar mas o filtro bloqueou.

## Parte 5 — Atualizar _init_trade_tracking()

Adicionar `ret_1d` ao contexto salvo na entrada:

```python
# Em _init_trade_tracking(), adicionar após entry_atr:
portfolio["entry_ret_1d"] = technical.get("ret_1d")
```

## Parte 6 — Atualizar _build_trade_record()

Adicionar campos do reversal filter ao registro do trade:

```python
# Em _build_trade_record(), adicionar após entry_atr:
"entry_ret_1d": portfolio.get("entry_ret_1d"),
"entry_filter_passed": portfolio.get("entry_filter_passed", True),
```

## Parte 7 — Atualizar log_cycle()

Localizar a função `log_cycle()` e adicionar informações do filtro quando aplicável.

Se `result["signal"] == "FILTERED"`, logar:
```
[CYCLE] score=3.245 vs threshold=2.8 → FILTERED (RSI_TOO_HIGH: rsi=42.3 >= 35) | regime=Sideways
```

Se `result["signal"] == "ENTER"`, logar (como já faz, mas incluir RSI e ret_1d):
```
[CYCLE] score=3.245 vs threshold=2.8 → ENTER (RSI=28.5, ret_1d=-0.003) | regime=Sideways
```

## Parte 8 — Atualizar dashboard

Em `src/dashboard/app.py`, no card de Paper Trading:

### 8a. Mostrar filtro no status

Onde mostra o último sinal (ENTER/HOLD/BLOCK), adicionar FILTERED como possível estado com cor amarela/laranja:
```python
if last_signal == "FILTERED":
    signal_color = "#FFA500"  # laranja
    signal_text = f"FILTERED ({portfolio.get('last_filter_reason', 'unknown')})"
```

### 8b. Mostrar RSI e ret_1d atuais

No card de indicadores, adicionar:
```
RSI: 28.5 (< 35 ✓) | ret_1d: -0.3% (> -1% ✓)
```

Ou se filtro não passa:
```
RSI: 42.3 (≥ 35 ✗) | ret_1d: -0.3% (> -1% ✓)
```

### 8c. Salvar filter state no portfolio (SEMPRE — pass ou block)

Na seção "11b" do `run_cycle()`, onde atualmente se salva `last_signal`, `last_score`, etc., adicionar SEMPRE (não apenas quando filtrado):

```python
# Reversal filter state (persisted for dashboard/debug)
portfolio["last_filter_passed"] = result.get("signal") != "FILTERED"
portfolio["last_filter_reason"] = result.get("filter_reason")
portfolio["last_filter_rsi"] = technical.get("rsi_14")
portfolio["last_filter_ret_1d"] = technical.get("ret_1d")
```

Isso garante que o dashboard SEMPRE tem os valores atuais de RSI e ret_1d, independente de o scoring ter dito ENTER ou não. É essencial para debugging — saber "o filtro teria passado?" mesmo quando o scoring disse HOLD.

## Parte 9 — Testes

### TestCheckReversalFilter (unitários)

1. `test_filter_disabled` — `reversal_filter.enabled: false` → always passes
2. `test_filter_passes_all` — RSI=28, ret_1d=-0.003 → passes
3. `test_filter_rsi_too_high` — RSI=42, ret_1d=-0.003 → blocked, reason contém RSI_TOO_HIGH
4. `test_filter_falling_knife` — RSI=28, ret_1d=-0.015 → blocked, reason contém FALLING_KNIFE
5. `test_filter_both_fail` — RSI=42, ret_1d=-0.015 → blocked, reason contém AMBOS ("RSI_TOO_HIGH" e "FALLING_KNIFE" separados por " & ")
6. `test_filter_rsi_exactly_35` — RSI=35.0 → blocked (condição é `<`, não `<=`)
7. `test_filter_ret_1d_exactly_minus_1pct` — ret_1d=-0.01 → blocked (condição é `>`, não `>=`)
8. `test_filter_rsi_none` — RSI=None → blocked, reason=FILTER_DATA_MISSING
9. `test_filter_ret_1d_none` — ret_1d=None → blocked, reason=FILTER_DATA_MISSING
10. `test_filter_edge_case_rsi_34_9` — RSI=34.9 → passes
11. `test_filter_edge_case_ret_minus_0_9pct` — ret_1d=-0.009 → passes
12. `test_filter_extreme_capitulation_override` — RSI=22, ret_1d=-0.02 → PASSES (RSI < 25 overrides ret_1d check)
13. `test_filter_extreme_override_disabled` — RSI=22, ret_1d=-0.02, rsi_extreme_override=0 → blocked (override disabled)
14. `test_filter_rsi_26_no_override` — RSI=26, ret_1d=-0.02 → blocked (RSI not extreme enough for override)

### TestRunCycleWithFilter (integração)

12. `test_entry_with_filter_pass` — scoring=ENTER + RSI<35 + ret_1d>-1% → posição aberta
13. `test_entry_with_filter_blocked_rsi` — scoring=ENTER + RSI=42 → sem posição, result["signal"]="FILTERED"
14. `test_entry_with_filter_blocked_ret` — scoring=ENTER + ret_1d=-1.5% → sem posição
15. `test_entry_with_filter_disabled` — scoring=ENTER + filter disabled → posição aberta (ignora RSI/ret_1d)
16. `test_hold_signal_unaffected` — scoring=HOLD → filter não é chamado, comportamento normal
17. `test_block_signal_unaffected` — scoring=BLOCK → filter não é chamado
18. `test_filter_context_saved` — após entry, portfolio tem entry_ret_1d e entry_filter_passed
19. `test_trade_record_has_filter_fields` — trade completo tem entry_ret_1d
20. `test_filter_state_persisted_on_block` — quando FILTERED, portfolio tem last_filter_passed=False e last_filter_reason
21. `test_filter_state_persisted_on_hold` — quando HOLD (scoring não disse ENTER), portfolio tem last_filter_rsi e last_filter_ret_1d atuais

### TestGetLatestTechnicalRet1d

22. `test_ret_1d_calculated` — DataFrame com 50+ rows → ret_1d calculado corretamente
23. `test_ret_1d_insufficient_data` — DataFrame com < 25 rows → ret_1d = None
24. `test_ret_1d_matches_manual` — Verificar cálculo: (close[-1] - close[-25]) / close[-25]

## Entregáveis

1. `src/features/technical.py` — `get_latest_technical()` retorna `ret_1d`
2. `conf/parameters.yml` — seção `reversal_filter` (enabled, rsi_max=35, ret_1d_min=-0.01, rsi_extreme_override=25)
3. `src/trading/paper_trader.py` — `check_reversal_filter()`, `run_cycle()` modificado, `_init_trade_tracking()` com ret_1d, `_build_trade_record()` com campos extras, filter state persistido sempre
4. `src/dashboard/app.py` — mostrar FILTERED como estado, exibir RSI e ret_1d atuais
5. Testes: 24 novos testes
6. Commit + push:
   ```bash
   git add -A && git commit -m "feat: reversal filter — RSI<35 + ret_1d>-1% entry confirmation (WR 63%, PF 2.31)"
   git push origin master:main
   ```

## Restrições

- **NÃO alterar** `gate_scoring.py` — o scoring pipeline continua idêntico
- **NÃO alterar** lógica de gates, clusters, threshold dinâmico, ou kill switches
- **NÃO alterar** `check_stops()`, `check_stops_only()`, `execute_entry()`, `execute_exit()`
- **NÃO alterar** MAE/MFE tracking
- **NÃO alterar** ATR-based dynamic stops
- **Filtro desabilitável** — `reversal_filter.enabled: false` restaura comportamento original
- **Dados indisponíveis = não entra** — se RSI ou ret_1d forem None, não executar entrada
- **Manter TODOS os logs existentes** — adicionar novos, não remover
- **Manter leitura de parâmetros de conf/parameters.yml** — thresholds do filtro vêm do YAML, não hardcoded
- **ret_1d é calculado das candles 1h** — 24 candles = 1 dia. Não precisa de fonte externa
- **O sinal "FILTERED"** é derivado de "ENTER" — para fins de score_history, usar o score original (não bloquear append_score_history)
- **Persistir filter state SEMPRE** — last_filter_rsi e last_filter_ret_1d são salvos em todo ciclo (não só quando FILTERED), para debug no dashboard

## Nota de Arquitetura Futura

> **G11 Reversal Gate** — Quando tivermos dados suficientes do paper trading (20+ trades), considerar transformar o filtro binário (passa/bloqueia) em um gate contínuo (G11) que entra no score total com peso configurável. Vantagem: permite weighting gradual ao invés de corte abrupto. Não implementar agora — o filtro binário é mais seguro com poucos dados.
