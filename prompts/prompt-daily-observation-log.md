# Task: Criar script de observação diária automática (modo paper trading 2 semanas)

## Contexto

AI.hab rodando em produção (EC2 São Paulo, Docker Compose). Sistema em modo observação / paper trading por 2 semanas. Precisamos de um log diário automático que capture o estado completo do sistema para análise posterior.

Arquitetura e spec em `CLAUDE.md` na raiz.

## O que criar

### 1. Script `scripts/daily_observation.py`

Script Python que:

**a) Lê o estado atual do sistema dos parquets:**

```python
# Fontes de dados:
portfolio = pd.read_parquet('data/05_output/portfolio.parquet')      # último estado
score_hist = pd.read_parquet('data/04_scoring/score_history.parquet') # histórico scores
gate_zscores = pd.read_parquet('data/02_features/gate_zscores.parquet')  # z-scores atuais
spot = pd.read_parquet('data/02_intermediate/spot/btc_1h_clean.parquet') # preço + indicadores
```

**b) Extrai os campos para o log:**

| Campo | Fonte | Descrição |
|-------|-------|-----------|
| timestamp | datetime.utcnow() | Momento da captura |
| btc_price | spot.close.iloc[-1] | Preço BTC atual |
| btc_change_24h | (close[-1] - close[-25]) / close[-25] * 100 | Variação 24h % |
| score_adjusted | portfolio.last_score | Score pós-regime |
| score_raw | portfolio.last_score_raw | Score pré-regime |
| regime_multiplier | portfolio.last_regime_multiplier | Multiplicador G0 |
| regime | portfolio.last_regime | Bull/Sideways/Bear |
| signal | portfolio.last_signal | ENTER/HOLD/BLOCK |
| threshold | 2.5 ou p75 dinâmico (como estiver implementado) | Threshold ativo |
| fg_raw | gate_zscores (coluna fg_raw ou fear_greed_raw).iloc[-1] | Fear & Greed raw 0-100 |
| fg_z | gate_zscores (coluna fg_z).iloc[-1] | F&G z-score (validar que negativo em Extreme Fear) |
| bb_pct | spot.bb_pct.iloc[-1] | Posição na Bollinger |
| rsi_14 | spot.rsi_14.iloc[-1] | RSI |
| oi_z | gate_zscores (coluna oi_z).iloc[-1] | OI z-score |
| cluster_technical | score_history (coluna correspondente).iloc[-1] | Score cluster Technical |
| cluster_positioning | idem | Score cluster Positioning |
| cluster_macro | idem | Score cluster Macro |
| cluster_liquidity | idem | Score cluster Liquidity |
| cluster_sentiment | idem | Score cluster Sentiment |
| cluster_news | idem | Score cluster News |
| kill_switches | portfolio (campo correspondente) ou lista vazia | Kill switches ativos |
| ma200 | spot.ma_200.iloc[-1] | MA200 |
| ma200_distance_pct | (close - ma200) / ma200 * 100 | Distância % do preço à MA200 |

**c) Calcula flags automáticas:**

```python
# Flags
flag_near_enter = score_adjusted > 1.5 and signal == 'HOLD'  # Perto de entrar
flag_extreme_negative = score_adjusted < -2.5                 # Score muito negativo
flag_signal_changed = signal != previous_signal               # Mudança de decisão
flag_kill_switch_active = len(kill_switches) > 0              # Algum kill switch ativo
flag_bb_extreme_high = bb_pct > 0.90                          # Topo da Bollinger
flag_bb_extreme_low = bb_pct < 0.10                           # Fundo da Bollinger
```

**d) Gera resumo de 1 linha automático:**

```python
# Template:
# "BTC $74,650 (+0.6% 24h) | HOLD score -0.65 (raw -1.30 × 0.5) | Sideways | F&G 23 | BB 0.91 | ⚠️ near-enter"
summary = f"BTC ${btc_price:,.0f} ({btc_change_24h:+.1f}% 24h) | {signal} score {score_adjusted:.2f} (raw {score_raw:.2f} × {regime_multiplier}) | {regime} | F&G {fg_raw:.0f} | BB {bb_pct:.2f}"
if any_flags:
    summary += f" | {'⚠️' if flag_near_enter else '🔴' if flag_signal_changed else '📊'} {flag_description}"
```

**e) Appenda ao CSV de observação:**

```python
obs_file = 'data/06_observation/daily_observation.csv'
# Se não existe, cria com headers
# Appenda nova linha
# Formato CSV para fácil abertura em Excel/Google Sheets
```

**f) Compara com dia anterior:**

Se o log tem pelo menos 2 registros, calcula:
- Delta score (melhorou/piorou)
- Delta preço
- Se regime mudou
- Se sinal mudou (CRÍTICO — logar em destaque)

### 2. Adicionar ao crontab

No arquivo `crontab` (supercronic), adicionar:

```
# Observação diária 01:00 UTC (22h BRT)
0 1 * * * cd /app && python scripts/daily_observation.py >> /app/logs/daily_observation.log 2>&1
```

### 3. Criar diretório de output

```bash
mkdir -p data/06_observation
```

Adicionar ao `Dockerfile` o `mkdir -p data/06_observation`.

### 4. Script de relatório semanal `scripts/weekly_report.py`

Script que lê `data/06_observation/daily_observation.csv` e gera um resumo semanal:

```
═══ Relatório Semanal AI.hab ═══
Período: 2026-04-17 a 2026-04-23

Resumo:
  BTC: $74,650 → $XX,XXX (variação X.X%)
  Score médio ajustado: X.XX (σ X.XX)
  Score range: [min, max]
  Regime predominante: Sideways (7/7 dias)
  Sinal predominante: HOLD (6/7 dias), BLOCK (1/7 dias)

Eventos relevantes:
  - 2026-04-19: ⚠️ Near-ENTER (score +1.8, threshold 2.5)
  - 2026-04-21: 🔴 Kill switch BLOCK_BB_TOP ativo

Clusters (média semanal):
  Technical:   +X.XX
  Positioning: -X.XX
  Macro:       +X.XX
  Liquidity:   +X.XX
  Sentiment:   +X.XX
  News:        +X.XX

Validação de calibração:
  F&G z-score consistentemente negativo em Extreme Fear? SIM/NÃO
  Kill switches dispararam com sentido? X disparos em Y dias
  Score oscila com sentido ou ruído? Correlação score vs retorno 24h: X.XX

Perguntas para revisão manual:
  - Algum near-ENTER coincidiu com setup favorável (BB baixo + OI desalavancado)?
  - O threshold atual (2.5) está calibrado ou deveria ser ajustado?
```

O script de relatório pode ser rodado manualmente (`python scripts/weekly_report.py`) ou agendado para domingo 01:00 UTC.

### 5. Volume Docker

Garantir que `data/06_observation/` está no volume `aihab-data` para persistência. Verificar no `docker-compose.yml`:

```yaml
volumes:
  - aihab-data:/app/data
```

Se `aihab-data` já monta `/app/data`, o subdiretório `06_observation` é automaticamente persistido. Só confirmar.

## Restrições

- **NÃO alterar** lógica de scoring, trading, ou parâmetros.
- **NÃO alterar** `parameters.yml`.
- Script é **read-only** nos parquets existentes — só lê e grava no próprio CSV de observação.
- Se algum parquet não existir ou estiver vazio, logar warning e continuar (não crashar o script).

## Entregáveis

1. `scripts/daily_observation.py` — script funcional
2. `scripts/weekly_report.py` — gerador de relatório semanal
3. `crontab` atualizado com a linha do daily_observation
4. `Dockerfile` atualizado com `mkdir -p data/06_observation`
5. Teste local: rodar `python scripts/daily_observation.py` e mostrar output do CSV gerado
6. Commit + push + deploy:
   ```bash
   git add -A && git commit -m "feat: daily observation log + weekly report for paper trading"
   git push origin main
   # EC2:
   ssh -i ~/.ssh/aihab-key-sp.pem ubuntu@54.232.162.161 "
     cd ~/AIhab && git pull && docker compose build --no-cache && docker compose up -d
   "
   ```

## Nota

Este script é temporário para o período de observação (2 semanas), mas pode ser mantido permanentemente como log de auditoria do sistema. O CSV gerado é exportável para Google Sheets ou Excel para análise visual.
