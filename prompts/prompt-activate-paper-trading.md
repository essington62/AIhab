# Task: Ativar paper trading agressivo + infraestrutura MAE/MFE

## Contexto

AI.hab em produção (EC2 São Paulo, Docker Compose).
Sistema em HOLD permanente porque o multiplicador Sideways (×0.5) impede qualquer ENTER.
Score bruto atual: +3.792. Score ajustado: +1.896 (abaixo de threshold 2.5).
Decisão: ativar paper trading com parâmetros menos conservadores para gerar dados de trades.

Spec completa em `CLAUDE.md` na raiz do repo btc_AI.

**IMPORTANTE:** Estas mudanças são para período de paper trading (2-4 semanas). Não é produção com dinheiro real. O objetivo é gerar trades para alimentar análise MAE/MFE posterior.

## Parte 1 — Ajustes de parâmetros

### 1.1 Multiplicador Sideways

Em `conf/parameters.yml`, mudar o multiplicador do regime Sideways:

```yaml
# ANTES
# sideways_multiplier: 0.5

# DEPOIS
sideways_multiplier: 1.0    # Paper trading: score bruto passa direto
```

Se o campo não existir explicitamente no YAML (pode estar hardcoded), procurar em:
- `src/models/gate_scoring.py` — onde o multiplicador é aplicado
- `src/dashboard/app.py` — onde o regime é exibido

Se estiver hardcoded, mover para parameters.yml e fazer o código ler de lá.

### 1.2 Zerar Bubble Index (G6)

Em `conf/parameters.yml`, zerar o max_score do g6_bubble:

```yaml
gate_params:
  # ...
  g6_bubble: [-0.345, 0.7, 0.0]    # max_score 0.0 (era 1.0)
```

Isso não remove o gate — apenas faz com que sua contribuição máxima seja zero. Mantém o cálculo do z-score (para monitoramento no dashboard) mas não afeta o score total.

### 1.3 Stops para "tiro curto"

Em `conf/parameters.yml`, ajustar stops:

```yaml
trading:
  stop_gain: 0.02        # 2% (era 1.5%)
  stop_loss: 0.03        # 3% (era 2%)
  trailing_stop: 0.015   # 1.5% trailing (era 1%)
```

Se os campos tiverem nomes diferentes, adaptar. Procurar em:
- `src/trading/paper_trader.py` — onde os stops são lidos/aplicados

### 1.4 Atualizar dashboard

Em `src/dashboard/app.py`, garantir que o breakdown mostra a mudança:
- Onde exibe `× Regime Sideways (0.5×)` deve agora mostrar `× Regime Sideways (1.0×)` — ou simplesmente não mostrar multiplicador quando é 1.0.
- O card de Sentiment deve mostrar G6 Bubble com contribuição 0.000 (não esconder, só zerar).

## Parte 2 — Infraestrutura MAE/MFE

### 2.1 O que é MAE/MFE

- **MAE (Maximum Adverse Excursion)**: maior drawdown que o trade sofreu antes de fechar
- **MFE (Maximum Favorable Excursion)**: maior runup que o trade teve antes de fechar

Estes dados permitem calibrar stops ótimos empiricamente: "se 90% dos trades que deram lucro tiveram MAE < 1.5%, então SL de 1.5% é ótimo".

### 2.2 Coletar dados durante o trade

Em `src/trading/paper_trader.py`, quando um trade está aberto (entre ENTER e EXIT):

A cada ciclo horário (cada vez que o paper_trader roda):

```python
if trade_open:
    current_price = get_current_price()
    entry_price = trade['entry_price']
    
    # Calcular excursões
    current_return = (current_price - entry_price) / entry_price
    
    # Atualizar max favorable e max adverse
    trade['max_favorable'] = max(trade.get('max_favorable', 0), current_return)
    trade['max_adverse'] = min(trade.get('max_adverse', 0), current_return)
    
    # Registrar série temporal do trade
    trade['price_path'].append({
        'timestamp': now_utc,
        'price': current_price,
        'return_pct': current_return,
        'hours_since_entry': hours_elapsed,
    })
```

### 2.3 Schema do trade completo

Quando o trade fecha (SG, SL, trailing, ou signal change), persistir:

```python
completed_trade = {
    # Identificação
    'trade_id': uuid4(),
    'entry_time': datetime,
    'exit_time': datetime,
    'duration_hours': float,
    
    # Preços
    'entry_price': float,
    'exit_price': float,
    'return_pct': float,             # (exit - entry) / entry
    
    # Stops usados
    'stop_gain_pct': float,          # o SG configurado
    'stop_loss_pct': float,          # o SL configurado
    'trailing_stop_pct': float,      # o trailing configurado
    'exit_reason': str,              # 'stop_gain' | 'stop_loss' | 'trailing' | 'signal_change' | 'timeout'
    
    # MAE/MFE
    'mae_pct': float,                # Maximum Adverse Excursion (negativo)
    'mfe_pct': float,                # Maximum Favorable Excursion (positivo)
    'mae_time': datetime,            # Quando atingiu MAE
    'mfe_time': datetime,            # Quando atingiu MFE
    'hours_to_mfe': float,           # Horas da entrada até MFE
    
    # Contexto na entrada
    'entry_score_raw': float,
    'entry_score_adjusted': float,
    'entry_regime': str,
    'entry_bb_pct': float,
    'entry_rsi': float,
    'entry_atr': float,
    'entry_oi_z': float,
    'entry_fg_raw': float,
    'entry_cluster_technical': float,
    'entry_cluster_positioning': float,
    'entry_cluster_macro': float,
    'entry_cluster_liquidity': float,
    'entry_cluster_sentiment': float,
    'entry_cluster_news': float,
    
    # Price path (série temporal completa)
    'price_path': list[dict],        # [{timestamp, price, return_pct, hours_since_entry}, ...]
}
```

### 2.4 Persistência

Criar novo parquet:

```python
# data/05_output/trades.parquet — um row por trade completo
# data/05_output/trade_paths.parquet — um row por observação (price path)
```

`trades.parquet` tem os campos escalares (MAE, MFE, return, exit_reason, etc.).
`trade_paths.parquet` tem o price path detalhado (trade_id + timestamp + price), para análise temporal posterior.

### 2.5 Script de análise MAE/MFE

Criar `scripts/analyze_trades.py`:

```python
#!/usr/bin/env python
"""Analisa trades do paper trading — MAE/MFE, win rate, expectancy."""

# Lê trades.parquet
# Calcula:
#   - Win rate (% de trades com return > 0)
#   - Avg win / Avg loss
#   - Expectancy = win_rate × avg_win - (1-win_rate) × avg_loss
#   - Profit factor = sum(wins) / sum(losses)
#   - MAE distribution: percentis 25, 50, 75, 90, 95
#   - MFE distribution: percentis 25, 50, 75, 90, 95
#   - Optimal SL (baseado em MAE p90 dos winners)
#   - Optimal SG (baseado em MFE p75 de todos os trades)
#   - Duration analysis: trades curtos vs longos
#
# Output:
# ═══ Trade Analysis (N trades) ═══
# Win Rate:     65% (13/20)
# Avg Win:      +1.8%
# Avg Loss:     -2.1%
# Expectancy:   +0.43% per trade
# Profit Factor: 1.52
#
# MAE Distribution (all trades):
#   p25: -0.5%  p50: -1.0%  p75: -1.8%  p90: -2.5%
# MAE Distribution (winners only):
#   p25: -0.3%  p50: -0.6%  p75: -1.2%  p90: -1.5%
#
# MFE Distribution (all trades):
#   p25: +0.8%  p50: +1.5%  p75: +2.2%  p90: +3.0%
#
# Optimal Stops (based on MAE/MFE):
#   SL recommendation: 1.5% (MAE p90 of winners)
#   SG recommendation: 2.2% (MFE p75 of all trades)
#   Expected improvement: +0.15% per trade vs current
#
# Context Analysis:
#   Best entries: score > 3.5, BB < 0.30, RSI < 40
#   Worst entries: score 2.5-3.0, BB > 0.60
```

### 2.6 Integração com dashboard

Na seção Paper Trading do dashboard (seção 9), adicionar:
- Tabela de trades recentes com MAE/MFE
- Win rate / Profit factor / Expectancy
- "Optimal stops" sugeridos vs configurados

Isso pode ser adicionado depois, quando houver dados. Por enquanto, só a coleta.

## Entregáveis

1. `conf/parameters.yml` atualizado (sideways ×1.0, bubble max 0.0, stops 2%/3%)
2. `src/trading/paper_trader.py` com coleta de MAE/MFE e price path
3. `src/dashboard/app.py` atualizado (display do multiplicador)
4. `scripts/analyze_trades.py` (análise pós-trades)
5. Novos parquets: `data/05_output/trades.parquet`, `data/05_output/trade_paths.parquet`
6. Testes para a lógica de MAE/MFE
7. Commit + push + deploy:
   ```bash
   git add -A && git commit -m "feat: activate paper trading (sideways x1.0, bubble zero) + MAE/MFE infra"
   git push origin main
   # EC2:
   ssh -i ~/.ssh/aihab-key-sp.pem ubuntu@54.232.162.161 "
     cd ~/AIhab && git pull && docker compose build --no-cache && docker compose up -d
   "
   ```

## Restrições

- **NÃO alterar** lógica do g1_technical (buckets validados, NÃO MEXER)
- **NÃO alterar** outros gate_params além do g6_bubble max_score
- **NÃO alterar** cluster_caps
- **NÃO alterar** lógica de kill switches (BLOCK_BB_TOP, BLOCK_OI_EXTREME, etc.)
- Manter leitura de todos os parâmetros de `conf/parameters.yml` (sem hardcode)
- Se o paper_trader não tiver estrutura para price_path (lista crescente), criar campo separado ou parquet à parte
