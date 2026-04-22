# Prompt: SOL Bot 5 Study — Bot 2 Strategy Transfer (2026 ONLY)

## Contexto

**Bot 4 SOL v1 foi REJEITADO** (Sharpe 0.08 full history, edge gone em 2026).

**Hipótese nova:** Bot 2 BTC strategy (momentum + stablecoin fuel) pode funcionar em SOL se aplicada PURA, sem os filtros v1 que eram SOL-specific (taker_z, oi_z bipolar).

**Bot 2 BTC performance live (Mar-Abr 2026):**
- 5 trades
- WR 80%
- PF 2.07
- Total Return +1.83%
- Sharpe backtest 2.71 (Fixed 2% TP / 1.5% SL)

## Objetivo

Testar se a **strategy de Bot 2 BTC transfere para SOL** usando APENAS dados 2026 (pós-regime change).

**Não é re-invenção** — é transferência de strategy validada entre assets.

## Filosofia do teste

```
BTC Bot 2 dá Sharpe 2.71 em 2026+
SOL v1 dá Sharpe -0.26 em Mar-Abr 2026

Pergunta: o problema foi STRATEGY ou FEATURES específicas de SOL?

Se problema for features SOL-specific (taker_z / oi_z):
  Bot 2 DNA puro pode ter edge em SOL
  Porque Bot 2 não depende desses features
  
Se problema for SOL regime em si:
  Bot 2 tampouco vai funcionar
  Confirma: SOL não tem edge
```

## Regras rigorosas

### 1. APENAS dados 2026

```python
# Filtro obrigatório
df = df[df.index >= "2026-01-01"]
```

Justificativa: outros bots (BTC, ETH) usam apenas 2026. Consistência metodológica.

### 2. Strategy IDÊNTICA ao Bot 2 BTC

**Não adaptar, não tunar.** Aplicar exatamente os mesmos filtros:

```python
def check_bot2_filters_sol(df, row):
    """Mesmos filtros Bot 2 BTC aplicados a SOL."""
    
    # Stablecoin Z (macro, não é BTC-specific)
    if row["stablecoin_z"] <= 1.3:
        return False, "stablecoin_insufficient"
    
    # Momentum diário
    if row["ret_1d"] <= 0:
        return False, "ret_1d_negative"
    
    # RSI zone
    if not (60 <= row["rsi"] <= 80):
        return False, f"rsi_outside_{row['rsi']:.1f}"
    
    # BB não top
    if row["bb_pct"] >= 0.98:
        return False, f"bb_too_high_{row['bb_pct']:.3f}"
    
    # Trend
    if row["close"] <= row["ma21"]:
        return False, "below_ma21"
    
    # Spike guard (Bot 2 anti-euphoria)
    if row["ret_1d"] > 0.03 or row["rsi"] > 65:
        if row["ret_1d"] > 0.03 and row["rsi"] > 65:
            return False, "spike_guard_active"
    
    return True, None
```

### 3. Stops IDÊNTICOS

```python
# Mesma config Bot 2 BTC
sl_pct = 0.015      # SL 1.5%
tp_pct = 0.020      # TP 2% (default, sem Dynamic TP)
trail_pct = 0.010   # Trail 1%
max_hold_hours = 120
```

## Metodologia

### Fase 1: Preparar dados SOL 2026

```python
import pandas as pd
from pathlib import Path

# Load SOL spot
sol = pd.read_parquet("data/01_raw/spot/sol_1h.parquet")
sol["timestamp"] = pd.to_datetime(sol["timestamp"], utc=True)
sol = sol.set_index("timestamp").sort_index()

# Filtrar 2026
sol_2026 = sol[sol.index >= "2026-01-01"].copy()
print(f"SOL 2026: {len(sol_2026)} rows, {sol_2026.index[0]} to {sol_2026.index[-1]}")

# Load stablecoin_z (macro feature)
# Buscar em features/gate ou similar
# Stablecoin_z é universal — mesmo para SOL quer BTC
```

### Fase 2: Compute features Bot 2 para SOL

```python
import numpy as np

def compute_features(df):
    """RSI, BB, MA21, ret_1d para SOL."""
    df = df.copy()
    
    # RSI 14
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df["rsi"] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands 21
    ma21 = df["close"].rolling(21).mean()
    std21 = df["close"].rolling(21).std()
    df["ma21"] = ma21
    df["bb_upper"] = ma21 + 2 * std21
    df["bb_lower"] = ma21 - 2 * std21
    df["bb_pct"] = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])
    
    # Returns
    df["ret_1d"] = df["close"].pct_change(24)  # 24h
    
    return df

sol_2026 = compute_features(sol_2026)

# Adicionar stablecoin_z (macro)
stablecoin_z_df = load_stablecoin_z()  # verificar path
sol_2026 = sol_2026.join(stablecoin_z_df["stablecoin_z"], how="left")
sol_2026["stablecoin_z"] = sol_2026["stablecoin_z"].fillna(method="ffill")
```

### Fase 3: Backtest Bot 2 strategy no SOL

```python
def backtest_bot2_sol(df, sl_pct=0.015, tp_pct=0.020, trail_pct=0.010, max_hold=120):
    """Bot 2 BTC strategy aplicada em SOL 2026."""
    
    trades = []
    in_position = False
    entry_price = None
    entry_time = None
    trail_high = None
    position_count = 0
    
    for i in range(21, len(df)):  # pular warmup
        row = df.iloc[i]
        
        if not in_position:
            # Check entry
            passed, reason = check_bot2_filters_sol(df, row)
            if passed:
                # Enter
                in_position = True
                entry_price = row["close"]
                entry_time = df.index[i]
                trail_high = entry_price
                entry_features = {
                    "rsi": row["rsi"],
                    "bb_pct": row["bb_pct"],
                    "ret_1d": row["ret_1d"],
                    "stablecoin_z": row["stablecoin_z"],
                    "close_ma21": row["close"] / row["ma21"],
                }
        else:
            # Check exit
            current_price = row["close"]
            hours_held = (df.index[i] - entry_time).total_seconds() / 3600
            
            # Trail logic
            if current_price > trail_high:
                trail_high = current_price
            
            sl_price = entry_price * (1 - sl_pct)
            tp_price = entry_price * (1 + tp_pct)
            trail_stop = trail_high * (1 - trail_pct)
            
            # Exit conditions
            exit_reason = None
            exit_price = None
            
            if row["low"] <= sl_price:
                exit_reason = "SL"
                exit_price = sl_price
            elif row["high"] >= tp_price:
                exit_reason = "TP"
                exit_price = tp_price
            elif row["low"] <= trail_stop and trail_high > entry_price * 1.002:
                # Só aciona trail se teve algum ganho
                exit_reason = "TRAIL"
                exit_price = trail_stop
            elif hours_held >= max_hold:
                exit_reason = "TIMEOUT"
                exit_price = current_price
            
            if exit_reason:
                ret = (exit_price - entry_price) / entry_price
                trades.append({
                    "entry_time": entry_time,
                    "exit_time": df.index[i],
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "return_pct": ret * 100,
                    "exit_reason": exit_reason,
                    "hours_held": hours_held,
                    **entry_features,
                })
                in_position = False
                entry_price = None
                entry_time = None
                trail_high = None
    
    return pd.DataFrame(trades)

sol_bot2_trades = backtest_bot2_sol(sol_2026)
print(f"Total trades: {len(sol_bot2_trades)}")
```

### Fase 4: Métricas

```python
def compute_metrics(trades_df):
    """Sharpe, WR, DD, etc."""
    if len(trades_df) == 0:
        return {"n_trades": 0, "sharpe": 0, "win_rate": 0, "total_return": 0}
    
    returns = trades_df["return_pct"].values / 100
    
    # Sharpe (annualized para 24/7 crypto)
    # Assumindo ~1 trade/dia na média, ajustar
    mean_ret = returns.mean()
    std_ret = returns.std()
    
    # Ajuste temporal para annual
    days_in_test = (trades_df["exit_time"].max() - trades_df["entry_time"].min()).days
    trades_per_year = len(trades_df) / (days_in_test / 365)
    sharpe = (mean_ret / std_ret) * np.sqrt(trades_per_year) if std_ret > 0 else 0
    
    # Win rate
    wr = (returns > 0).mean()
    
    # Drawdown
    cum_returns = (1 + returns).cumprod()
    peak = cum_returns.cummax()
    dd = (cum_returns - peak) / peak
    max_dd = dd.min()
    
    # Profit factor
    wins = returns[returns > 0].sum()
    losses = abs(returns[returns < 0].sum())
    pf = wins / losses if losses > 0 else float('inf')
    
    # Total return
    total_return = cum_returns.iloc[-1] - 1 if len(cum_returns) > 0 else 0
    
    return {
        "n_trades": len(trades_df),
        "sharpe": sharpe,
        "win_rate": wr,
        "avg_return": mean_ret * 100,
        "total_return": total_return * 100,
        "max_dd": max_dd * 100,
        "profit_factor": pf,
    }

sol_metrics = compute_metrics(sol_bot2_trades)
print(sol_metrics)
```

### Fase 5: Validação sub-períodos

```python
# Split em 3 (consistente com study anterior)
periods = [
    ("Jan-Feb 2026", "2026-01-01", "2026-02-28"),
    ("Mar-Apr 2026", "2026-03-01", "2026-04-22"),
]

for name, start, end in periods:
    sub_trades = sol_bot2_trades[
        (sol_bot2_trades["entry_time"] >= start) &
        (sol_bot2_trades["entry_time"] <= end)
    ]
    m = compute_metrics(sub_trades)
    print(f"\n{name}:")
    print(f"  N: {m['n_trades']}, Sharpe: {m['sharpe']:.2f}, WR: {m['win_rate']:.1%}")
```

### Fase 6: Comparação com Bot 2 BTC 2026

```python
# Usar mesmo período 2026 em BTC
btc_2026 = load_btc_2026()
btc_bot2_trades = backtest_bot2_sol(btc_2026)  # mesma função
btc_metrics = compute_metrics(btc_bot2_trades)

comparison = pd.DataFrame({
    "metric": ["n_trades", "sharpe", "win_rate", "avg_return", "total_return", "max_dd", "profit_factor"],
    "BTC Bot 2": list(btc_metrics.values()),
    "SOL Bot 2 transfer": list(sol_metrics.values()),
})
print(comparison)
```

## Critérios de decisão

**APROVAR Bot 5 SOL (Bot 2 transfer) se TODOS:**
- Sharpe > 1.5 em 2026 full
- N ≥ 15 trades
- WR > 50%
- Max DD < 5%
- Consistente em ambos sub-períodos (Jan-Fev E Mar-Abr)

**REJEITAR se:**
- Sharpe < 1.0
- N < 10
- Mar-Abr 2026 sub-período negativo (mesmo problema SOL v1)

**INCONCLUSIVO se:**
- Sharpe 1.0-1.5 (borderline)
- Sub-períodos divergentes
- Precisa mais dados

## Output esperado

Relatório `prompts/sol_bot2_transfer_report.md`:

```markdown
# SOL Bot 5 — Bot 2 BTC Strategy Transfer (2026 ONLY)

## 1. Dados
- Período: Jan 2026 - Abr 2026
- N rows SOL: XXXX
- N rows BTC: XXXX

## 2. Resultado SOL (Bot 2 strategy)

| Metric | Value |
|--------|-------|
| N trades | XX |
| Sharpe | X.XX |
| WR | XX% |
| Total Return | X.XX% |
| Max DD | X.XX% |
| PF | X.XX |

## 3. Comparação BTC vs SOL

| Metric | BTC Bot 2 | SOL Bot 2 | Δ |
|--------|-----------|-----------|---|
| N | XX | XX | — |
| Sharpe | X.XX | X.XX | ±X.XX |
| ... | | | |

## 4. Sub-períodos SOL

| Período | N | Sharpe | WR |
|---------|---|--------|-----|
| Jan-Feb | XX | X.XX | XX% |
| Mar-Abr | XX | X.XX | XX% |

## 5. Insights

[Análise: transfere ou não?]

## 6. Recomendação

[Aprovar → implementar Bot 5 | Rejeitar → abandon SOL]

## 7. Se aprovado: implementação

Código sugerido para Bot 5 SOL baseado em Bot 2 BTC.
```

## Tarefas resumo

1. ✅ Script `scripts/sol_bot2_transfer_study.py`
2. ✅ Load SOL + BTC 2026
3. ✅ Compute features (RSI, BB, MA21, stablecoin_z)
4. ✅ Backtest Bot 2 strategy em ambos
5. ✅ Comparação head-to-head
6. ✅ Sub-períodos análise
7. ✅ Relatório com recomendação binária

## Tempo estimado

- Setup + data loading: 20 min
- Feature computation: 20 min
- Backtest implementation: 30 min
- Metrics + comparison: 20 min
- Sub-period analysis: 15 min
- Report + plots: 25 min
- **Total: ~2h10min**

## Critério importante

**NÃO tunar a strategy.** Aplicar EXATAMENTE como Bot 2 BTC. Se funcionar, funciona. Se não, rejeita.

Tuning = over-fit risk. Regra rigorosa: copy-paste do Bot 2 BTC config.

## Contexto importante

**Aprendizados das tentativas anteriores:**
- Phase 2 (21/04): Sharpe 2.03 test set pequeno = false positive
- Filter study (22/04): Sharpe 0.08 full, 2.23 Out-Dez 2025
- v2 Sweet Spot (22/04): Sharpe 0.26, REJEITADO
- Conclusion: edge SOL decay em 2026

**Esta tentativa é DIFERENTE:**
- Strategy BTC validada LIVE em 2026
- Não é SOL-specific features
- Não é sweet spot mining
- É transfer learning de strategy comprovada

Se FUNCIONAR: confirma que SOL tem edge, só strategy estava errada.
Se FALHAR: confirma que SOL não tem edge em 2026 (qualquer strategy).

## Notas técnicas

**Stablecoin_z:**
- É feature MACRO do mercado de stablecoins total
- Não varia por asset operado
- SOL e BTC usam MESMO stablecoin_z
- Path provável: `data/02_features/gate_zscores.parquet`

**Dynamic TP:** NÃO APLICAR
- Teste primeiro com Fixed 2%/1.5%/trail 1%
- Se strategy transfer funciona: depois avaliar Dynamic TP
- Uma variável de cada vez
