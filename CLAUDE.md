# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.

## Projeto

**btc-trading-v1** — Sistema de trading BTC para produção (AWS).
Ciclo 1h. Scripts puros (sem Kedro). Parquets como interface entre módulos.

## Arquitetura (v1.0 — Gate Scoring + R5C + Fed Sentinel)

```
Ciclo Horário (1h):
    Binance Spot (1h candles) → BB, RSI, MAs
    Binance Futures (1h) → OI, taker ratio, funding rate
    News RSS (crypto + macro + Fed) → DeepSeek classify
                ↓
    Gate Features → z-scores de todas as variáveis
    Fed Sentinel → proximity adjustment + fed sentiment
                ↓
    R5C Regime (daily) → Bull / Sideways / Bear
                ↓
    Gate Scoring v2 → 6 clusters → total score vs threshold
                ↓
    ENTER / HOLD / BLOCK
                ↓
    Execução: stops (SL 2%, SG 1.5%, trailing 1%)

Ciclo Diário (07:00 UTC):
    FRED (DGS10, DGS2, RRP) → macro z-scores
    CoinGlass (stablecoin mcap, bubble index, ETF flows) → daily z-scores
    Alt.me (Fear & Greed)
    R5C HMM re-fit (candle diário fechado)
```

## Gate Scoring v2

### Os 11 Gates

| Gate | Nome | Fonte | Freq | Scoring |
|------|------|-------|------|---------|
| G0 | Regime (R5C HMM) | Binance daily | daily | Binário: Bear=BLOCK, Sideways=x0.5, Bull=x1.0 |
| G1 | Technical (BB+RSI) | Binance 1h | 1h | Buckets validados (208 sinais walk-forward) |
| G2 | News (crypto + Fed) | DeepSeek | 1h | Split: G2_crypto + G2_fed |
| G3 | Macro rates | FRED | daily | Contínuo tanh: DGS10, DGS2, curve 2y10y, RRP |
| G4 | Positioning (OI) | Binance Futures | 1h | Contínuo tanh (corr -0.472, mais forte) |
| G5 | Crypto liquidity | CoinGlass | daily | Contínuo tanh: stablecoin mcap |
| G6 | Bubble index | CoinGlass | daily | Contínuo tanh: overextension filter |
| G7 | ETF flows | CoinGlass | daily | Contínuo tanh: cum 7d, fluxo institucional |
| G8 | Fear & Greed | Alt.me | daily | Contínuo tanh: contrarian |
| G9 | Taker ratio | Binance Futures | 1h | Contínuo tanh: pressão compradora/vendedora |
| G10 | Funding rate | Binance Futures | 1h | Contínuo tanh: conditional, extremos |

### Scoring Contínuo (G3-G10)

```python
def gate_score_continuous(z, corr, sensitivity, max_score):
    raw = corr * np.tanh(z * sensitivity) * max_score
    return np.clip(raw, -max_score, max_score)
```

Parâmetros:
```python
GATE_PARAMS = {
    "g3_dgs10":    (-0.315, 0.7, 1.0),
    "g3_curve":    (-0.282, 0.7, 0.8),
    "g3_rrp":      (+0.212, 0.7, 0.7),
    "g3_dgs2":     (-0.154, 0.7, 0.5),
    "g4_oi":       (-0.472, 0.8, 2.0),   # MAIS FORTE
    "g5_stable":   (+0.326, 0.6, 1.0),
    "g6_bubble":   (-0.345, 0.7, 1.0),
    "g7_etf":      (+0.263, 0.6, 1.0),
    "g8_fg":       (-0.211, 0.7, 0.8),
    "g9_taker":    (+0.143, 0.5, 0.5),
    "g10_funding": (-0.064, 0.4, 0.5),
}
```

### G1 Technical — NÃO MEXER

Buckets validados walk-forward (208 sinais, 2025+):
```python
# BB (dominante)
if bb > 0.80:   bb_score = -2.0   # kill switch topo
elif bb < 0.20: bb_score = +3.0   # win 88%
elif bb < 0.30: bb_score = +2.0   # win 77%
elif bb < 0.40: bb_score = +0.5
else:           bb_score = +0.0

# RSI (complementar)
if rsi < 35:    rsi_score = +1.0
elif rsi < 45:  rsi_score = +0.5
elif rsi > 60:  rsi_score = -1.0
else:           rsi_score = +0.0
```

### Cluster Caps (anti-double counting)

```python
clusters = {
    "technical":   np.clip(g1,              -2.0, +3.5),
    "news":        np.clip(g2,              -1.5, +1.0),
    "macro":       np.clip(g3,              -1.5, +1.0),
    "positioning": np.clip(g4 + g10,        -2.0, +1.5),
    "liquidity":   np.clip(g5 + g7,         -1.5, +1.5),
    "sentiment":   np.clip(g6 + g8 + g9,    -1.5, +1.5),
}
total_score = sum(clusters.values())
```

### Threshold Dinâmico

```python
threshold = np.clip(
    np.quantile(score_history[-90:], 0.75),
    2.0,   # floor
    5.0    # ceiling
) + fed_proximity_adjustment  # Fed Sentinel overlay
```

Warmup (< 90 dias de histórico): threshold = 3.5

### Kill Switches

- G0 regime = Bear → BLOCK
- BB > 0.80 → BLOCK_BB_TOP
- OI z > 2.5 (se dados frescos) → BLOCK_OI_EXTREME
- News BEAR forte (score < -3) → BLOCK_NEWS_BEAR
- G2_fed < -1.0 perto de FOMC → BLOCK_FED_HAWKISH

### Stale Data

Cada gate tem tolerância de staleness:
```python
MAX_STALE_DAYS = {
    "g4_oi": 3, "g9_taker": 3, "g10_funding": 3,
    "g5_stablecoin": 7, "g6_bubble": 7, "g7_etf": 5,
    "g8_fg": 3, "g3_macro": 2,
}
```
Gate stale → score = 0.0 (neutro). Se >50% CoinGlass gates stale → BLOCK_STALE_DATA.

## Fed Sentinel

Módulo dedicado à política monetária do Fed.

### 3 Camadas

1. **Static** (`fed_calendar.json`): datas FOMC, hearings, transições, blackout periods
2. **Dynamic** (`fed_news_ingest.py` + DeepSeek): classificação hawkish/dovish com fator surpresa e peso por membro
3. **Threshold Adaptive**: proximity adjustment ao threshold do scoring

### Proximity Adjustment

```
T-2 a T0 de FOMC decision: +1.5
T-5 a T-3 de FOMC: +0.7
T-1 a T0 de hearing/transition: +0.5
T+1 de FOMC: +0.3
Blackout (T-10 a T-2): +0.3
Normal: +0.0
```

### FOMC Voting Members 2026

```python
MEMBER_WEIGHT = {
    "Jerome Powell": 1.0,       # Chair (term ends May 15)
    "Kevin Warsh": 0.8,         # Chair-designate
    "John Williams": 0.7,       # NY Fed (always votes)
    "Philip Jefferson": 0.6,    # Vice Chair
    "Christopher Waller": 0.6,  # hawkish
    "Michelle Bowman": 0.5,     # hawkish
    "Stephen Miran": 0.5,       # dovish, Trump ally
    "Michael Barr": 0.4,
    "Lisa Cook": 0.4,
    "Beth Hammack": 0.4,
    "Neel Kashkari": 0.4,
    "Lorie Logan": 0.4,
    "Anna Paulson": 0.3,
}
```

### G2 Split

G2 = G2_crypto (50%) + G2_fed (50%)
Se G2_fed < -1.0 perto de FOMC → kill switch temporário.

## R5C HMM

- 3 estados: Bull / Sideways / Bear (covariance_type=full)
- Features: log_return, vol_short, vol_ratio, drawdown, volume_z, slope_21d
- Roda daily. Regime do dia D aplicado candles D+1 00:00 UTC em diante.
- Distribuição histórica: Sideways 41%, Bull 31%, Bear 28%

## Data Sources

### Intraday (1h cycle — Binance)

| Endpoint | Path | Dados |
|----------|------|-------|
| Spot klines 1h | `/api/v3/klines` | OHLCV → BB, RSI |
| Futures OI | `/futures/data/openInterestHist` | OI statistics (1h) |
| Futures taker | `/futures/data/takerlongshortRatio` | Taker buy/sell (1h) |
| Futures funding | `/fapi/v1/fundingRate` | Funding rate history |
| Futures L/S ratio | `/futures/data/topLongShortPositionRatio` | Monitor |

Nota: Binance Futures historical data = últimos 30 dias apenas. Acumular no lake.

### Daily (07:00 UTC)

| Fonte | Dados | Gate |
|-------|-------|------|
| FRED | DGS10, DGS2, RRPONTSYD | G3 |
| CoinGlass | stablecoin mcap, bubble index, ETF flows | G5, G6, G7 |
| Alt.me | Fear & Greed | G8 |
| Yahoo Finance | VIX, DXY, Oil, S&P500 | Dashboard contexto |

### News (1h cycle)

| Fonte | Tipo |
|-------|------|
| CryptoCompare | Crypto news (50/hora) |
| Google News RSS | Macro (energy, fed, geopolitical, inflation, global_risk) |
| Fed Board RSS | Speeches, press releases |
| Google News RSS | Fed-specific (rate decisions, FOMC, speakers) |

Classificação: DeepSeek (~$2/mês), prompt separado para crypto vs Fed.

## Project Structure

```
btc-trading-v1/
├── conf/
│   ├── fed_calendar.json
│   ├── gate_params.yml
│   └── credentials.yml            # gitignored
├── data/
│   ├── 01_raw/
│   │   ├── spot/                   # Binance spot OHLCV
│   │   ├── futures/                # Binance Futures (OI, taker, funding)
│   │   ├── macro/                  # FRED
│   │   ├── coinglass/              # Daily (stablecoin, bubble, ETF)
│   │   ├── news/                   # All news parquets
│   │   └── sentiment/              # F&G
│   ├── 02_features/                # Z-scores
│   ├── 03_models/                  # R5C HMM pickle
│   ├── 04_scoring/                 # Gate scores, score history
│   └── 05_output/                  # Paper trading, shadow log
├── src/
│   ├── data/                       # Data ingestion modules
│   ├── features/                   # Feature engineering + Fed Sentinel
│   ├── models/                     # R5C HMM + Gate Scoring
│   ├── trading/                    # Paper trader + execution
│   └── dashboard/                  # Streamlit app
├── scripts/
│   ├── hourly_cycle.sh             # Cron: every hour
│   ├── daily_update.sh             # Cron: 07:00 UTC
│   ├── migrate_historical.py       # One-time: import old data
│   └── backfill_binance_futures.py # One-time: fill 30d history
├── tests/
├── CLAUDE.md
├── environment.yml
└── README.md
```

## Crontab

```
# Hourly cycle (minuto 5 de cada hora)
5 * * * *    /path/to/btc-trading-v1/scripts/hourly_cycle.sh

# Daily update (07:00 UTC)
0 7 * * *    /path/to/btc-trading-v1/scripts/daily_update.sh
```

## Execution Rules

- atomic_write_json() para todos os state files
- DatetimeIndex UTC always
- Parquets como interface entre módulos (write → read)
- Cada módulo funciona standalone (testável isoladamente)
- Todas as API calls com retry + timeout + logging
- Gate scoring dentro de try/except no paper trader (nunca quebra o loop)
- Portfolio state em JSON atômico com stale-write protection

## Walk-forward Evidence

Scoring v1 (BB + RSI, 2025+ Sideways):
- BB < 0.30: 51 sinais, win_3d=75%, ret_3d=+1.37%
- Score >= 2.5: 32 sinais, win_3d=72%, ret_3d=+1.18%
- BB > 0.30: retorno NEGATIVO

Correlações 2026+ (que geraram os gates):
- OI coin margin: corr=-0.472 (mais forte)
- Bubble index: corr=-0.345
- Stablecoin Mcap: corr=+0.326
- DGS10: corr=-0.315
- Yield curve 2y10y: corr=-0.282
- ETF flows cum 7d: corr=+0.263
- RRP: corr=+0.212
- F&G: corr=-0.211
- DGS2: corr=-0.154
- Taker ratio: corr=+0.143
- Funding rate: corr=-0.064

## Contexto de Mercado (abril 2026)

- BTC ~$71k, rally pós-ceasefire EUA-Irã (temporário, 2 semanas)
- R5C: Sideways (p=0.992)
- Oil caiu 15% ($95), yields caindo, DXY caindo
- Derivativos bearish: OI z=+1.82, taker z=-2.50
- Fed: rates 3.50-3.75%, Warsh hearing 16/04, FOMC 28-29/04
- Powell term expires 15/05
- Score atual: -2.005 / threshold 3.50 → HOLD robusto

## Migration Notes

Dados históricos importados de `crypto-market-state`:
- CoinGlass parquets → data/01_raw/coinglass/
- FRED parquets → data/01_raw/macro/
- Spot OHLCV → data/01_raw/spot/
- R5C HMM model → data/03_models/r5c_hmm.pkl

Binance Futures backfill: 30 dias de OI, taker, funding preenchidos.
Histórico anterior (pré-Binance) vem dos parquets CoinGlass migrados.
