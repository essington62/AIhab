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
├── data/                          # gitignored, volumes Docker
│   ├── 01_raw/                    # Dados brutos das APIs
│   ├── 02_intermediate/           # Clean: resampled, ffilled
│   ├── 02_features/               # Z-scores, news_scores
│   ├── 03_models/                 # R5C HMM pickle
│   ├── 04_scoring/                # Gate scores, regime history
│   └── 05_output/                 # Portfolio, trades
├── src/
│   ├── config.py                  # Loader centralizado
│   ├── data/                      # Ingestão (9 módulos)
│   ├── features/                  # Technical, gate_features, fed_sentinel
│   ├── models/                    # r5c_hmm, gate_scoring
│   ├── trading/                   # paper_trader, execution
│   └── dashboard/                 # Streamlit app (9 seções)
├── scripts/                       # hourly_cycle.sh, daily_update.sh
├── docker/                        # environment_docker.txt, crontab
├── Dockerfile, docker-compose.yml
├── tests/                         # 69 testes
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
9. Paper Trading (capital, P&L, equity curve, alpha)

## Calibration Alerts

Rolling 30d correlação vs retorno 3d forward. Compara com parameters.yml.
✅ Δ<0.15 | ⚠️ Δ>0.15 | 🔴 Δ>0.25

## Roadmap

### Curto prazo
- Rotina diária de recalibração (correlações rolling vs config)
- Elastic IP na EC2
- Acumular paper trading data

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
