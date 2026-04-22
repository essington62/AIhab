# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.

## Projeto

**AI.hab** (btc-trading-v1) — Sistema de trading BTC para produção (AWS São Paulo).
Ciclo 1h. Scripts puros (sem Kedro). Parquets como interface entre módulos.
Nome: AI.hab — Capitão Ahab caçando baleias com inteligência artificial.

## Deploy

- **AWS EC2**: t3.small, sa-east-1 (São Paulo), IP: 54.232.162.161
- **Docker Compose**: 2 containers (aihab-app com cron + aihab-dashboard Streamlit)
- **Dashboard**: http://54.232.162.161:8501
- **GitHub**: https://github.com/essington62/AIhab.git
- **Instance ID**: i-06ddcf82415eaff56
- **Key pair**: aihab-key-sp (~/.ssh/aihab-key-sp.pem no Mac Mini)

## Arquitetura (v1.1 — Gate Scoring + R5C + Fed Sentinel + MA200 Override)

```
Ciclo Horário (1h):
    Binance Spot (1h candles) → BB, RSI, MAs
    CoinGlass Futures (4h, ffill→1h) → OI agregado, funding OI-weighted, taker
    Binance Futures (1h) → L/S ratio top accounts/positions (whale tracking)
    News RSS (crypto + macro + Fed) → DeepSeek classify
                ↓
    Clean (raw → intermediate, resample 1H grid, ffill)
    Gate Features → z-scores de todas as variáveis
    Fed Sentinel → proximity adjustment + fed sentiment
                ↓
    R5C Regime (daily) → Bull / Sideways / Bear
    MA200 Override → se close < MA200 e slope negativo 5+ dias → força Bear
                ↓
    Gate Scoring v2 → 6 clusters → total score vs threshold
                ↓
    ENTER / HOLD / BLOCK
                ↓
    Execução: stops (SL 2%, SG 1.5%, trailing 1%)

Ciclo Diário (07:00 UTC):
    Binance Spot (1d candles)
    FRED (DGS10, DGS2, RRP) → macro z-scores
    CoinGlass (stablecoin mcap, bubble index, ETF flows) → daily z-scores
    Alt.me (Fear & Greed)
    yfinance (VIX, DXY, Oil WTI, S&P500) → dashboard contexto
    R5C HMM re-fit (candle diário fechado)
    Recompute features
```

## Gate Scoring v2

### Os 11 Gates

| Gate | Nome | Fonte | Freq | Scoring |
|------|------|-------|------|---------|
| G0 | Regime (R5C HMM + MA200) | Binance daily | daily | Bear=BLOCK, Sideways=x0.5, Bull=x1.0 |
| G1 | Technical (BB+RSI) | Binance 1h | 1h | Buckets validados (208 sinais walk-forward) |
| G2 | News (crypto + Fed) | DeepSeek | 1h | Split: G2_crypto + G2_fed |
| G3 | Macro rates | FRED | daily | Contínuo tanh: DGS10, DGS2, curve 2y10y, RRP |
| G4 | Positioning (OI) | CoinGlass agregado | 4h | Contínuo tanh (corr -0.472, mais forte) |
| G5 | Crypto liquidity | CoinGlass | daily | Contínuo tanh: stablecoin mcap (max 1.5) |
| G6 | Bubble index | CoinGlass | daily | Contínuo tanh: overextension filter |
| G7 | ETF flows | CoinGlass | daily | Contínuo tanh: cum 7d (max 1.5) |
| G8 | Fear & Greed | Alt.me | daily | Contínuo tanh: contrarian |
| G9 | Taker ratio | CoinGlass | 4h | Contínuo tanh (max 0.3, correlação 2026 fraca) |
| G10 | Funding rate | CoinGlass OI-weighted | 4h | Contínuo tanh: conditional, extremos |

### Parâmetros (ajustados 2026-04-15 via análise retroativa)

```yaml
gate_params:
  g3_dgs10:    [-0.315, 0.7, 1.0]
  g3_curve:    [-0.282, 0.7, 0.8]
  g3_rrp:      [+0.212, 0.7, 0.7]
  g3_dgs2:     [-0.154, 0.7, 0.5]
  g4_oi:       [-0.472, 0.8, 2.0]    # MAIS FORTE
  g5_stable:   [+0.326, 0.6, 1.5]    # aumentado (melhor preditor 7d, corr +0.50)
  g6_bubble:   [-0.345, 0.7, 1.0]
  g7_etf:      [+0.263, 0.6, 1.5]    # aumentado (2º melhor preditor 7d, corr +0.48)
  g8_fg:       [-0.211, 0.7, 0.8]
  g9_taker:    [+0.143, 0.5, 0.3]    # reduzido (correlação 2026 = 0.06, ruído)
  g10_funding: [-0.064, 0.4, 0.5]
```

### Cluster Caps

```yaml
cluster_caps:
  technical:   [-2.0, 3.5]
  news:        [-1.5, 1.0]
  macro:       [-1.5, 1.0]
  positioning: [-2.0, 1.5]
  liquidity:   [-1.5, 2.5]    # cap aumentado pra acomodar g5+g7 com max 1.5
  sentiment:   [-1.5, 1.5]
```

### G1 Technical — NÃO MEXER

Buckets validados walk-forward (208 sinais, 2025+). Não converter pra tanh.

### MA200 Override (v1.1)

Se close < MA200 e slope negativo por 5+ dias → força Bear (bypass R5C lento).

### Kill Switches

- G0 regime = Bear (ou MA200 override) → BLOCK
- BB > 0.80 → BLOCK_BB_TOP
- OI z > 2.5 (se dados frescos) → BLOCK_OI_EXTREME
- News BEAR forte (score < -3) → BLOCK_NEWS_BEAR
- G2_fed < -1.0 perto de FOMC → BLOCK_FED_HAWKISH

## Data Sources

### CoinGlass (derivativos agregado, 4h, Hobbyist $29/mês)
- OI agregado cross-exchange (~$56B)
- Funding OI-weighted cross-exchange
- Taker buy/sell (Binance)
- Stablecoin mcap, Bubble index, ETF flows (daily)
- Liquidações, Order book bid/ask (dashboard)

### Binance (spot + whale tracking, grátis)
- Spot OHLCV 1h/1d → BB, RSI, MAs
- L/S ratio top accounts/positions 1h → whale tracking

### Outras (grátis)
- FRED: DGS10, DGS2, RRP
- Alt.me: Fear & Greed
- yfinance: VIX, DXY, Oil, S&P500
- Google News RSS + DeepSeek (~$2/mês)

## Project Structure

```
btc_AI/
├── conf/                          # Configuração (zero hardcode)
│   ├── parameters.yml             # Gates, stops, capital_management, momentum_filter_v2
│   └── credentials.yml            # API keys (gitignored)
├── data/                          # gitignored, volumes Docker
│   ├── 01_raw/                    # Dados brutos das APIs
│   │   ├── spot/                  # btc_1h, eth_1h, sol_1h
│   │   └── futures/               # oi, funding, taker, ls_account, ls_position (BTC+ETH)
│   ├── 02_intermediate/           # Clean: resampled, ffilled
│   ├── 02_features/               # Z-scores, news_scores
│   ├── 03_models/                 # R5C HMM pickle
│   ├── 04_scoring/                # Gate scores, regime history
│   ├── 05_output/                 # Portfolio, trades (parquet)
│   └── 05_trades/                 # Trades JSON por bot (completed_trades_sol.json)
├── src/
│   ├── config.py                  # Loader centralizado
│   ├── data/                      # Ingestão (10 módulos)
│   │   ├── binance_spot.py        # Spot 1h/1d — multi-symbol
│   │   ├── binance_ls.py          # L/S top accounts/positions — multi-symbol
│   │   ├── coinglass_futures.py   # OI/Funding/Taker 4h — multi-symbol
│   │   ├── coinglass_ls.py        # L/S bootstrap via CoinGlass (one-shot)
│   │   └── utils.py               # fetch_with_retry, dedup_by_timestamp, save_with_window
│   ├── features/
│   │   └── technical.py           # get_latest_technical() → rsi_14, bb_pct, ma_21, volume_z, rsi (alias)
│   ├── models/                    # r5c_hmm, gate_scoring
│   ├── trading/
│   │   ├── paper_trader.py        # run_cycle, check_momentum_filter, check_momentum_filter_v2
│   │   ├── dynamic_tp.py          # Dynamic TP v2: 3 regras (volume_z, RSI+BB, default)
│   │   ├── capital_manager.py     # Multi-bucket portfolio (btc_bot1 + btc_bot2, 50/50)
│   │   └── execution.py
│   └── dashboard/                 # Streamlit app (9 seções)
│       └── app.py                 # load_sol_trades_json() lê 05_trades/completed_trades_sol.json
├── scripts/
│   ├── hourly_cycle.sh / daily_update.sh
│   ├── bootstrap_eth_history.py
│   ├── bootstrap_ls_coinglass.py
│   ├── check_eth_data_coverage.py
│   ├── eth_phase0_statistical_study.py
│   ├── backtest_bot2_v2.py
│   ├── sol_filters_study.py       # Estudo filtros estruturais SOL (2026-04-22)
│   ├── sol_v2_sweet_spot_backtest.py  # Backtest sweet spot close/MA21 SOL (2026-04-22)
│   ├── sol_bot2_transfer_study.py # Estudo transfer Bot 2 → SOL (2026-04-22)
│   └── migrate_portfolio_to_multibucket.py
├── prompts/                       # Relatórios de análise e backtest
│   ├── eth_phase0_report.md
│   ├── bot2_v2_backtest_report.md
│   ├── sol_v2_sweet_spot_report.md
│   ├── sol_bot2_transfer_report.md
│   ├── plots/                     # Heatmaps, equity curves
│   └── tables/                    # CSVs de correlação e trades
├── tests/
│   ├── test_dynamic_tp.py         # 9 testes Dynamic TP v2
│   └── ...                        # 316+ testes total
├── docker/                        # environment_docker.txt, crontab
├── Dockerfile, docker-compose.yml
└── CLAUDE.md
```

## Dashboard (9 seções, dark theme CoinGlass style)

1. Header (BTC, OI, F&G, regime, score, MA200, Fed, cron)
2. Gate Scoring v2 (6 clusters + texto interpretativo)
3. AI Analyst (DeepSeek sob demanda)
4. Whale Tracking (L/S + gráfico divergência)
5. Derivativos (OI, funding, taker, liquidações, bid/ask, order book)
6. Macro (DGS10, DGS2, curve, VIX, DXY, Oil, S&P)
7. News & Sentiment (feed + scores + F&G + Fed Sentinel)
8. System Health (freshness + calibration alerts + score history)
9. Paper Trading (capital, P&L, equity curve, alpha — BTC + SOL separados)

## Calibration Alerts

Rolling 30d correlação vs retorno 3d forward. Compara com parameters.yml.
✅ Δ<0.15 | ⚠️ Δ>0.15 | 🔴 Δ>0.25

## Bots — Status (2026-04-22)

### Bot 1 BTC (Gate Scoring)
- **Status:** LIVE (produção)
- Estratégia: Gate Score threshold → ENTER/BLOCK
- Stops: SL 2%, SG 1.5%, trailing 1%

### Bot 2 BTC (Momentum + Stablecoin)
- **Status:** LIVE (produção) + Dynamic TP v2 ativo
- Estratégia: stablecoin_z > 1.3, ret_1d > 0, 60 ≤ RSI ≤ 80, BB < 0.98, close > MA21, spike guard
- Stops: SL 1.5%, Trail 1%, TP dinâmico (ver abaixo)
- **Live Mar-Abr 2026:** 5 trades, WR 80%, PF 2.07, +1.83%
- ⚠️ Backtest 2026 (N=25): Sharpe -0.90 — divergência live/backtest a monitorar (pequena amostra live)

### Bot 4 SOL
- **Status:** PAUSADO (2026-04-22)
- 1 trade fechado: -0.98% (TRAIL, 3h28m)
- **Conclusão: SOL ABANDONADO** — 3 estudos consecutivos rejeitados (ver seção SOL abaixo)

## Bot 2 — Dynamic TP v2 (live desde 2026-04-22)

`src/trading/dynamic_tp.py` — `get_dynamic_tp(rsi, bb_pct, volume_z) → (tp_pct, reason)`

| Regra | Condição | TP | Reason |
|-------|----------|----|--------|
| 1 | volume_z > 1.0 | 1.0% | volume_exhaustion |
| 2 | RSI > 75 AND BB > 0.95 | 1.5% | overbought |
| 3 | default | 2.0% | default |

- `volume_z`: rolling 168h z-score de volume, computado em `get_latest_technical()`
- `rsi`: alias adicionado em `get_latest_technical()` (chave real é `rsi_14`)
- `mf_check["volume_z"]` injetado em `paper_trader.py` antes de `_execute_bot2_entry()`

## Bot 2 v2 — Early Reversal (Backtest — REJEITADO)

`check_momentum_filter_v2` em paper_trader.py — duas cláusulas OR:
- **Classic:** ret_1d > 0, RSI > 50, close > MA21
- **Early:** ret_1d > -1.5%, trend_improving 3h, delta_ret_3h > 0.5%, RSI > 35

**Veredicto backtest (2026-01-08 → 2026-04-20): ❌ REJEITADO**

Early entries (n=16): WR 50% mas PF 0.79 — losses maiores que wins.
`momentum_filter_v2.enabled: false` — código mantido para referência.

## SOL — Estudos e Veredito Final (2026-04-22)

**3 estudos independentes. 3 rejeições. SOL abandonado.**

| Estudo | Script | Resultado | Sharpe 2026 |
|--------|--------|-----------|-------------|
| Filter Study | `sol_filters_study.py` | ❌ REJEITADO | 0.08 baseline |
| v2 Sweet Spot | `sol_v2_sweet_spot_backtest.py` | ❌ REJEITADO | 0.26 (N=9, overfit) |
| Bot 2 Transfer | `sol_bot2_transfer_study.py` | ❌ REJEITADO | -2.16 (WR 27%) |

**Conclusão:** Problema é o regime SOL em 2026, não a estratégia. Nenhuma strategy de momentum tem edge em SOL 2026.

### SOL Dashboard Fix
- Bot 4 escreve trades em `data/05_trades/completed_trades_sol.json`
- Dashboard lê via `load_sol_trades_json()` (normaliza schema: pnl_pct→return_pct×100, timestamps UTC)
- **Não** usar `data/05_output/trades_sol.parquet` (path legado, não existe)

## Multi-Symbol (ETH — Fase 0 → 1)

### Naming convention (multi-symbol)

BTC mantém paths legados sem prefixo. Outros símbolos ganham prefixo:
- BTC: `ls_account_1h.parquet`, `oi_4h.parquet`
- ETH: `eth_ls_account_1h.parquet`, `eth_oi_4h.parquet`

### Dados ETH disponíveis (2026-04-20)

| Dataset | Rows | Período |
|---------|------|---------|
| Spot 1h | 8760 | 365d |
| OI/Funding/Taker 4h | 1080 | 180d |
| L/S Account/Position 1h | 500 | 28d (cron acumula) |

### L/S híbrido (Binance + CoinGlass)

Binance retém apenas ~30 dias de L/S. Bootstrap via CoinGlass (one-shot, usa quota).
`binance_ls.py` clamps startTime a 28d automaticamente para evitar HTTP 400.
`dedup_by_timestamp` garante que Binance (novo) vence CoinGlass (antigo) em overlap.

### ETH Phase 0 — Resultados (2026-04-20)

**Model Alignment 28d: 0.664 | 180d: 0.782** → zona "adaptive layer suficiente"

| Gate | 180d ETH corr_3d | Config BTC | Status |
|------|-----------------|------------|--------|
| G4 OI | +0.019 | -0.472 | 🔴 broken (instável) |
| G10 Funding | +0.020 | -0.064 | ✅ aligned |
| G9 Taker | -0.007 | +0.143 | ✅ aligned |
| G8 F&G | -0.100 | -0.211 | ✅ aligned |
| G3 DGS10 | -0.159 | -0.315 | ⚠️ attention |
| G5 Stablecoin | +0.122 | +0.326 | ⚠️ attention |
| G7 ETF | +0.056 | +0.263 | ⚠️ attention |
| G3 Curve | -0.061 | -0.282 | ⚠️ attention |
| G6 Bubble | -0.011 | -0.345 | 🔴 broken |

**Próximo passo ETH:** `conf/parameters_eth.yml` com ajuste dos gates ⚠️/🔴.

## Capital Manager (Multi-Bucket)

`src/trading/capital_manager.py` — dois buckets independentes (btc_bot1 + btc_bot2, 50/50).
- `capital_management.enabled: false` por default (ativar após migração)
- Migração: `python scripts/migrate_portfolio_to_multibucket.py`
- Safety gates: max_drawdown 15%, max_daily_loss 5%, pause mechanics

## Roadmap

### BTC (prioridade imediata)
- Monitorar divergência live/backtest Bot 2 (live WR 80% N=5 vs backtest Sharpe -0.90 N=25)
- Acumular N=20+ trades live para validação estatística
- Rotina diária de recalibração (correlações rolling vs config)
- Elastic IP na EC2

### ETH (próximo)
- `conf/parameters_eth.yml` com corr_cfg recalibrados para ETH (G4 OI e G6 Bubble estão broken)
- Paper trading ETH (quando alignment > 0.4 confirmado)
- Deploy EC2: `git pull` + `bootstrap_eth_history.py`

### SOL
- **ABANDONADO** — re-avaliar somente se regime mudar (evidência externa necessária)

### Data Lake Multi-Exchange (btc-data-lake/ — projeto separado)
- Assessment 6 exchanges (Binance, OKX, Bybit, Gate, Bitget, KuCoin)
- Ingestão parametrizável via YAML
- Agregação caseira, deploy AWS S3

### Arbitragem (btc-arbitrage/ — projeto separado)
- Funding rate arbitrage delta-neutral
- AI.hab como orquestrador de bots (Grid, Arb, Trend)
- Gate scoring adaptado anti-whale

### Comercialização
- Signal service, Dashboard SaaS, Copy trading

## Filosofia

O mercado crypto não tem memória longa. ML supervisionado falhou.
Abordagem correta: regras estatísticas + pesos em YAML + validação contínua.
Adaptação > Previsão. O Robin Hood do BTC. 🐋
