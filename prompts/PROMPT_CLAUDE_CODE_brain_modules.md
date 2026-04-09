# PROMPT — Features + Scoring + Trading (o cérebro do sistema)

## Contexto

A camada de dados está pronta (src/data/*). Os parquets fluem:
01_raw/ → 02_intermediate/ (clean.py) → 02_features/ (próximo)

Agora construir: features → scoring → trading.
Ler conf/parameters.yml e src/config.py antes de começar.

## Módulo 1: src/features/gate_features.py

Lê de 02_intermediate/, calcula z-scores, salva em 02_features/gate_zscores.parquet.

```python
"""
Gate Features — Z-scores para todos os gates.
Lê dados limpos de 02_intermediate/, produz z-scores.
Output: data/02_features/gate_zscores.parquet

Todas as janelas vêm de parameters.yml["zscore_windows"].
Todos os paths vêm de catalog.yml.
"""

# Z-scores a calcular:
#
# De futures (1h, 02_intermediate):
#   oi_z          → zscore(open_interest_value, window=14)
#   taker_z       → zscore(buy_sell_ratio, window=14)
#   funding_z     → zscore(funding_rate, window=14)
#
# De macro (daily, 02_intermediate):
#   dgs10_z       → zscore(DGS10, window=30)
#   dgs2_z        → zscore(DGS2, window=30)
#   rrp_z         → zscore(RRPONTSYD, window=30)
#   curve_z       → zscore(DGS10 - DGS2, window=30)
#
# De coinglass (daily, 01_raw — já são clean):
#   stablecoin_z  → zscore(stablecoin_mcap, window=30)
#   bubble_z      → zscore(bubble_index, window=30)
#   etf_z         → zscore(rolling_sum(etf_flows, 7d), window=30)
#
# De sentiment (daily, 01_raw):
#   fg_z          → zscore(fear_greed, window=30)

def compute_zscore(series, window):
    """Rolling z-score."""
    mean = series.rolling(window, min_periods=max(window//2, 5)).mean()
    std = series.rolling(window, min_periods=max(window//2, 5)).std()
    z = (series - mean) / std.replace(0, np.nan)
    return z

def compute_all_zscores():
    """
    Compute z-scores from all intermediate/raw data.
    Returns DataFrame with DatetimeIndex (UTC) and z-score columns.
    """
    params = get_params()
    windows = params["zscore_windows"]
    
    # 1. Futures (1h) — ler de 02_intermediate
    oi = pd.read_parquet(get_path("clean_futures_oi"))
    taker = pd.read_parquet(get_path("clean_futures_taker"))
    funding = pd.read_parquet(get_path("clean_futures_funding"))
    
    oi_z = compute_zscore(oi.set_index("timestamp")["open_interest_value"], windows["oi"])
    taker_z = compute_zscore(taker.set_index("timestamp")["buy_sell_ratio"], windows["taker"])
    funding_z = compute_zscore(funding.set_index("timestamp")["funding_rate"], windows["funding"])
    
    # 2. Macro (daily) — ler de 02_intermediate
    macro = pd.read_parquet(get_path("clean_macro"))
    # macro deve ter colunas: timestamp, dgs10, dgs2, rrp
    dgs10_z = compute_zscore(macro.set_index("timestamp")["dgs10"], windows["dgs10"])
    dgs2_z = compute_zscore(macro.set_index("timestamp")["dgs2"], windows["dgs2"])
    rrp_z = compute_zscore(macro.set_index("timestamp")["rrp"], windows["rrp"])
    curve = macro.set_index("timestamp")["dgs10"] - macro.set_index("timestamp")["dgs2"]
    curve_z = compute_zscore(curve, windows["yield_curve"])
    
    # 3. CoinGlass daily — ler de 01_raw (já clean)
    stable = pd.read_parquet(get_path("coinglass_stablecoin"))
    bubble = pd.read_parquet(get_path("coinglass_bubble"))
    etf = pd.read_parquet(get_path("coinglass_etf"))
    fg = pd.read_parquet(get_path("sentiment_fg"))
    
    stable_z = compute_zscore(
        stable.set_index("timestamp")["value"], windows["stablecoin_mcap"]
    )
    bubble_z = compute_zscore(
        bubble.set_index("timestamp")["value"], windows["bubble_index"]
    )
    # ETF: rolling_sum 7d primeiro, depois z-score
    etf_series = etf.set_index("timestamp")["value"]
    etf_cum = etf_series.rolling(params["etf_flow_rolling"], min_periods=3).sum()
    etf_z = compute_zscore(etf_cum, windows["etf_flows"])
    
    fg_z = compute_zscore(
        fg.set_index("timestamp")["value"], windows["fear_greed"]
    )
    
    # 4. Merge tudo num DataFrame
    # Futures são 1h, macro/coinglass são daily
    # Usar merge_asof ou reindex pra alinhar daily → 1h (forward-fill)
    
    # Base: timestamps do OI (1h grid)
    base_idx = oi_z.index
    
    result = pd.DataFrame(index=base_idx)
    result["oi_z"] = oi_z
    result["taker_z"] = taker_z
    result["funding_z"] = funding_z
    
    # Daily z-scores: reindex pro grid 1h com ffill
    for name, series in [
        ("dgs10_z", dgs10_z), ("dgs2_z", dgs2_z),
        ("rrp_z", rrp_z), ("curve_z", curve_z),
        ("stablecoin_z", stable_z), ("bubble_z", bubble_z),
        ("etf_z", etf_z), ("fg_z", fg_z),
    ]:
        result[name] = series.reindex(base_idx, method="ffill")
    
    # 5. Salvar
    result.to_parquet(get_path("gate_zscores"))
    return result
```

**IMPORTANTE**: os nomes de colunas nos parquets intermediários podem ser
diferentes do que está aqui. O Claude Code DEVE verificar os nomes reais:
```python
for name in ["clean_futures_oi", "clean_futures_taker", "clean_futures_funding", "clean_macro"]:
    df = pd.read_parquet(get_path(name))
    print(f"{name}: cols={list(df.columns)}")
```

## Módulo 2: src/features/technical.py

Calcula BB, RSI, MAs a partir dos candles 1h.

```python
"""
Technical Indicators — BB, RSI, MAs
Lê spot 1h de 02_intermediate.
Parâmetros de parameters.yml["technical"].
"""

def compute_bollinger(close, window=20, std=2):
    ma = close.rolling(window).mean()
    std_dev = close.rolling(window).std()
    upper = ma + std * std_dev
    lower = ma - std * std_dev
    pct = (close - lower) / (upper - lower)
    return pct, upper, lower, ma

def compute_rsi(close, window=14):
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# Output: bb_pct, rsi, mas (7, 21, 50, 99, 200)
# Retorna dict com últimos valores pra usar no scoring
```

## Módulo 3: src/features/fed_sentinel.py

3 camadas. Ler de conf/fed_calendar.json e conf/parameters.yml.

```python
"""
Fed Sentinel — 3 camadas:
1. Static: Calendário FOMC, blackout detection
2. Dynamic: Classificação hawkish/dovish (DeepSeek)
3. Threshold Adaptive: proximity adjustment

Parâmetros de parameters.yml["fed"].
Calendário de conf/fed_calendar.json.
Members/weights de conf/fed_calendar.json.
"""

# Implementar:
#
# compute_fomc_proximity_adjustment(today) → float
#   Retorna valor a SOMAR ao threshold do scoring.
#   Usa parameters.yml["fed"]["proximity_adjustments"]
#
# compute_fed_sentiment(articles, today) → dict
#   Classifica artigos Fed via DeepSeek
#   Aplica MEMBER_WEIGHT
#   Retorna {fed_score, proximity_factor, n_articles, ...}
#
# is_in_blackout(today) → bool
#   True se estamos em blackout period (T-10 a T-2 de FOMC)
#
# get_next_fed_event(today) → dict
#   Retorna próximo evento do calendário
```

## Módulo 4: src/models/r5c_hmm.py

Inferência do R5C. Modelo já treinado (data/03_models/r5c_hmm.pkl).

```python
"""
R5C HMM — Regime classification
3 estados: Bull / Sideways / Bear
Features: log_return, vol_short, vol_ratio, drawdown, volume_z, slope_21d

Roda 1x/dia. Regime do dia D aplicado a partir de D+1 00:00 UTC.
Modelo pickle migrado do projeto antigo.
"""

# CRITICAL — Day-Shift Contract:
# Regime do dia D é calculado com candle diário de D (fechamento).
# Aplicado aos candles de D+1 00:00 UTC em diante.
# NUNCA aplicar regime do dia D aos candles do próprio dia D.

def load_model():
    """Load R5C HMM from pickle."""
    import pickle
    with open(get_path("r5c_model"), "rb") as f:
        return pickle.load(f)

def compute_features(daily_candles):
    """Compute R5C features from daily OHLCV."""
    # log_return, vol_short (7d), vol_ratio (7d/30d),
    # drawdown (from peak), volume_z (30d), slope_21d (linear reg)
    ...

def predict_regime(model, features):
    """Predict current regime."""
    # Returns: regime (str), probabilities (dict)
    probs = model.predict_proba(features)
    state = model.predict(features)
    # Map state index to label (Bull/Sideways/Bear)
    # IMPORTANT: state mapping depende de como o modelo foi treinado
    # Verificar no projeto antigo qual index = qual regime
    ...
```

## Módulo 5: src/models/gate_scoring.py

O engine de decisão. Migrar a lógica validada, adaptar pra ler de config.

```python
"""
Gate Scoring v2
11 gates → 6 clusters → total score → decision

TUDO vem de parameters.yml:
- gate_params (correlações, sensibilidades, max_scores)
- cluster_caps
- threshold (warmup, floor, ceiling, quantile, window)
- kill_switches
- stale_tolerance_days
- g1_bb_scores, g1_rsi_scores
"""

def gate_score_continuous(z, corr, sensitivity, max_score):
    """Tanh scoring. Sinal correto: corr * tanh(z * sens) * max."""
    raw = corr * np.tanh(z * sensitivity) * max_score
    return float(np.clip(raw, -max_score, max_score))

def evaluate_g0_regime(regime):
    """Bear=BLOCK, Sideways=0.5, Bull=1.0"""
    ...

def evaluate_g1_technical(bb_pct, rsi):
    """Bucket scoring — lê thresholds de parameters.yml."""
    # NÃO hardcodar os buckets. Ler de params["g1_bb_scores"]
    ...

def evaluate_g2_news(crypto_score, fed_sentiment, fed_proximity):
    """Split G2: crypto (50%) + fed (50%). Kill switch se fed < -1.0 perto FOMC."""
    params = get_params()["fed"]
    ...

def evaluate_g3_to_g10(zscores, stale_days):
    """Contínuo tanh pra todos os gates 3-10. Parâmetros de gate_params."""
    params = get_params()["gate_params"]
    ...

def aggregate_clusters(gate_results):
    """Agrupa em clusters e aplica caps de parameters.yml."""
    caps = get_params()["cluster_caps"]
    ...

def compute_threshold(score_history, fed_proximity_adj):
    """Threshold dinâmico + Fed proximity overlay."""
    params = get_params()["threshold"]
    ...

def check_kill_switches(bb_pct, oi_z, news_regime, news_score, oi_stale):
    """Kill switches de parameters.yml."""
    ks = get_params()["kill_switches"]
    ...

def run_scoring_pipeline(regime, bb_pct, rsi, zscores, stale_days,
                          news_crypto, fed_sentiment, fed_proximity_adj,
                          score_history):
    """
    Pipeline completo: gates → clusters → threshold → decision.
    Retorna dict com signal, score, threshold, sizing, breakdown.
    """
    ...
```

## Módulo 6: src/trading/paper_trader.py

Ciclo 1h. O mais simples possível.

```python
"""
Paper Trader v1.0
Ciclo: 1h (chamado pelo hourly_cycle.sh)
Pipeline: load data → R5C → features → scoring → decision → execution
"""

def run_cycle():
    """Um ciclo completo de trading."""
    
    # 1. Load regime (daily, último disponível)
    regime = load_r5c_regime()
    
    # 2. Load technical (1h candle mais recente)
    technical = compute_latest_technical()  # BB, RSI do último candle
    
    # 3. Load z-scores (último row do gate_zscores.parquet)
    zscores = load_latest_zscores()
    
    # 4. Compute stale_days pra cada gate
    stale_days = compute_stale_days(zscores)
    
    # 5. Load news sentiment
    news = load_news_sentiment()
    
    # 6. Load Fed Sentinel
    fed = compute_fed_context()  # sentiment + proximity
    
    # 7. Load score history
    score_history = load_score_history()
    
    # 8. Run Gate Scoring
    result = run_scoring_pipeline(
        regime=regime,
        bb_pct=technical["bb_pct"],
        rsi=technical["rsi"],
        zscores=zscores,
        stale_days=stale_days,
        news_crypto=news,
        fed_sentiment=fed["sentiment"],
        fed_proximity_adj=fed["proximity_adjustment"],
        score_history=score_history,
    )
    
    # 9. Load portfolio
    portfolio = load_portfolio()
    
    # 10. Decision
    if result["signal"] == "ENTER" and not portfolio["has_position"]:
        execute_entry(result["sizing"], portfolio)
    elif portfolio["has_position"]:
        check_stops(portfolio, technical)
    
    # 11. Append to score history
    append_score_history(result)
    
    # 12. Log (shadow-style pra análise)
    log_cycle(result, technical, portfolio, fed)

if __name__ == "__main__":
    run_cycle()
```

## Módulo 7: src/trading/execution.py

Migrar do projeto antigo (shared/execution.py). Manter:
- atomic_write_json()
- parse_utc()
- execute_buy(), execute_sell()
- idempotência, stale-write protection

## Scripts de orquestração

### scripts/hourly_cycle.sh (ATUALIZAR)

```bash
#!/bin/bash
cd /Users/brown/Documents/MLGeral/btc_AI

echo "$(date -u) ── Hourly cycle start"

# 1. Data ingestion
python -m src.data.binance_spot --timeframe 1h
python -m src.data.binance_futures
python -m src.data.news_ingest

# 2. Clean (raw → intermediate)
python -m src.data.clean

# 3. Features (intermediate → features)
python -m src.features.gate_features
python -m src.features.technical
python -m src.features.fed_sentinel

# 4. News classification (se classify_news existir)
# python -m src.data.classify_news

# 5. Trading decision
python -m src.trading.paper_trader

# 6. DQ check
python -m src.data.dq

echo "$(date -u) ── Hourly cycle done"
```

### scripts/daily_update.sh (ATUALIZAR)

```bash
#!/bin/bash
cd /Users/brown/Documents/MLGeral/btc_AI

echo "$(date -u) ── Daily update start"

# 1. Daily data
python -m src.data.binance_spot --timeframe 1d
python -m src.data.fred_ingest
python -m src.data.coinglass_ingest
python -m src.data.altme_ingest
python -m src.data.market_context

# 2. R5C HMM (precisa candle diário fechado)
python -m src.models.r5c_hmm

# 3. Reclean + recompute features com dados diários novos
python -m src.data.clean
python -m src.features.gate_features

echo "$(date -u) ── Daily update done"
```

## Testes

```python
# tests/test_gate_scoring.py
# Migrar os 34 testes do projeto antigo, adaptar pra ler de config

# tests/test_fed_sentinel.py
# Testar proximity adjustment com datas conhecidas

# tests/test_paper_trader.py
# Testar ciclo completo com dados mock
```

## Ordem de implementação

1. src/features/technical.py (simples, BB/RSI)
2. src/features/gate_features.py (z-scores)
3. src/features/fed_sentinel.py (3 camadas)
4. src/models/r5c_hmm.py (inferência, migrar pickle)
5. src/models/gate_scoring.py (engine de decisão)
6. src/trading/execution.py (migrar do antigo)
7. src/trading/paper_trader.py (ciclo 1h)
8. scripts/hourly_cycle.sh + daily_update.sh
9. Testes

## CRITICAL

- ZERO hardcode. Tudo de parameters.yml e catalog.yml via src/config.py.
- Verificar nomes de colunas reais nos parquets intermediários antes de codificar.
- Day-Shift Contract: regime D aplicado candles D+1.
- G1 bucket scoring: NÃO converter pra tanh. Validado empiricamente.
- Portfolio state: atomic_write_json() sempre.
- Cada módulo roda standalone: `python -m src.features.gate_features`
