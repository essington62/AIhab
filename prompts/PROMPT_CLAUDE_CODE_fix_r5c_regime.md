# PROMPT — Fix R5C: Bear no projeto novo vs Sideways no antigo

## Problema

O projeto novo classifica R5C = Bear com BTC a $71.445.
O projeto antigo classifica R5C = Sideways (p=0.992) com o mesmo BTC.
Com Bear, o sistema bloqueia TUDO (BLOCK_BEAR_REGIME). Nada opera.

## Causa mais provável

O spot 1d do projeto novo tem apenas 91 dias (agregado de candles 1h).
O projeto antigo tem anos de histórico diário.
O HMM é sensível ao tamanho e formato da série — com 91 dias de dados
agregados (não candles diários reais) a distribuição das features muda,
e o modelo classifica diferente.

## Diagnóstico

### Passo 1: Comparar dados diários entre projetos

```python
import pandas as pd
from pathlib import Path

# Projeto antigo
old_path = Path("/Users/brown/Documents/MLGeral/crypto_v2/crypto-market-state/data")

# Encontrar candles diários no projeto antigo
print("=== Projeto antigo: spot daily ===")
for p in sorted(old_path.rglob("*.parquet")):
    name = str(p.relative_to(old_path)).lower()
    if "spot" in name and ("1d" in name or "daily" in name):
        df = pd.read_parquet(p)
        print(f"  {p.relative_to(old_path)}: {len(df)} rows | cols={list(df.columns)[:6]}")
        if len(df) > 0:
            ts_col = [c for c in df.columns if 'time' in c.lower() or 'date' in c.lower()]
            if ts_col:
                dates = pd.to_datetime(df[ts_col[0]])
                print(f"    Range: {dates.min()} → {dates.max()} ({(dates.max()-dates.min()).days} days)")

# Projeto novo
new_path = Path("/Users/brown/Documents/MLGeral/btc_AI/data")
print("\n=== Projeto novo: spot daily ===")
new_daily = new_path / "01_raw" / "spot" / "btc_1d.parquet"
if new_daily.exists():
    df = pd.read_parquet(new_daily)
    print(f"  {len(df)} rows | cols={list(df.columns)[:6]}")
    if "timestamp" in df.columns:
        print(f"  Range: {df['timestamp'].min()} → {df['timestamp'].max()}")
else:
    print("  NOT FOUND")
```

### Passo 2: Comparar features R5C entre projetos

```python
# No projeto antigo, verificar como as features são calculadas
# e qual o regime atual
import pickle
import numpy as np

# Modelo antigo
old_model_paths = list(old_path.rglob("*r5c*hmm*.pkl")) + list(old_path.rglob("*hmm*r5c*.pkl"))
print(f"\nModelos HMM encontrados: {old_model_paths}")

# Modelo novo
new_model = new_path / "03_models" / "r5c_hmm.pkl"
print(f"Modelo novo existe: {new_model.exists()}")

# Verificar se são o mesmo arquivo
if old_model_paths and new_model.exists():
    import hashlib
    old_hash = hashlib.md5(open(old_model_paths[0], 'rb').read()).hexdigest()
    new_hash = hashlib.md5(open(new_model, 'rb').read()).hexdigest()
    print(f"Same model? {old_hash == new_hash} (old={old_hash[:8]}, new={new_hash[:8]})")
```

### Passo 3: Comparar input features

```python
# Calcular features R5C com dados do projeto ANTIGO e do NOVO
# e comparar os valores

# Features R5C: log_return, vol_short, vol_ratio, drawdown, volume_z, slope_21d

def compute_r5c_features(daily_df):
    """Compute R5C features from daily OHLCV."""
    df = daily_df.copy()
    
    # log_return
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))
    
    # vol_short (7d rolling std of log_return)
    df["vol_short"] = df["log_return"].rolling(7).std()
    
    # vol_ratio (7d / 30d)
    vol_long = df["log_return"].rolling(30).std()
    df["vol_ratio"] = df["vol_short"] / vol_long
    
    # drawdown (from rolling max)
    rolling_max = df["close"].cummax()
    df["drawdown"] = (df["close"] - rolling_max) / rolling_max
    
    # volume_z (30d z-score of volume)
    vol_mean = df["volume"].rolling(30).mean()
    vol_std = df["volume"].rolling(30).std()
    df["volume_z"] = (df["volume"] - vol_mean) / vol_std
    
    # slope_21d (linear regression slope of close over 21d)
    from scipy import stats
    def rolling_slope(series, window=21):
        slopes = []
        for i in range(len(series)):
            if i < window - 1:
                slopes.append(np.nan)
            else:
                y = series.iloc[i-window+1:i+1].values
                x = np.arange(window)
                slope, _, _, _, _ = stats.linregress(x, y)
                slopes.append(slope)
        return pd.Series(slopes, index=series.index)
    
    df["slope_21d"] = rolling_slope(df["close"], 21)
    
    features = ["log_return", "vol_short", "vol_ratio", "drawdown", "volume_z", "slope_21d"]
    return df[features].dropna()

# Comparar últimas features
# (adaptar paths e nomes de colunas conforme encontrado no Passo 1)
```

### Passo 4: Rodar predição com dados do antigo e do novo

```python
# Carregar modelo
with open(new_model, 'rb') as f:
    model = pickle.load(f)

# Predizer com features do projeto ANTIGO (histórico longo)
# features_old = compute_r5c_features(old_daily_df)
# pred_old = model.predict(features_old.iloc[-30:].values)
# probs_old = model.predict_proba(features_old.iloc[-30:].values)

# Predizer com features do projeto NOVO (91 dias)
# features_new = compute_r5c_features(new_daily_df)
# pred_new = model.predict(features_new.iloc[-30:].values)
# probs_new = model.predict_proba(features_new.iloc[-30:].values)

# print("Last 5 days — OLD data:")
# for i in range(-5, 0):
#     print(f"  {pred_old[i]} probs={probs_old[i]}")
#
# print("Last 5 days — NEW data:")
# for i in range(-5, 0):
#     print(f"  {pred_new[i]} probs={probs_new[i]}")
```

## Fix

### Se o problema é dados insuficientes (91d vs anos):

Migrar o histórico diário COMPLETO do projeto antigo:

```python
# Copiar candles diários completos
import shutil

# Identificar o arquivo correto no projeto antigo (do Passo 1)
old_daily = "..."  # path encontrado
new_daily = "/Users/brown/Documents/MLGeral/btc_AI/data/01_raw/spot/btc_1d.parquet"

# Ler antigo, normalizar schema
old_df = pd.read_parquet(old_daily)
# Renomear colunas pra schema do projeto novo: timestamp, open, high, low, close, volume
# Garantir UTC
old_df["timestamp"] = pd.to_datetime(old_df["timestamp_col"], utc=True)
# ... normalizar

# Salvar
old_df.to_parquet(new_daily, index=False)
print(f"Migrated: {len(old_df)} rows, {old_df['timestamp'].min()} → {old_df['timestamp'].max()}")
```

Depois re-rodar:
```bash
python -m src.data.clean          # regenerar intermediate
python -m src.models.r5c_hmm      # recalcular regime
```

### Se o problema é state_labels:

O Claude Code já corrigiu por drawdown ordering ({0:Bull, 1:Bear, 2:Sideways}).
Mas se com dados completos ainda diverge, comparar state assignment:

```python
# No projeto antigo, como o regime é determinado?
grep -rn "predict\|regime\|state_label\|hmm" \
  /Users/brown/Documents/MLGeral/crypto_v2/crypto-market-state/src/ \
  --include="*.py" | head -20
```

### Validação final

Depois do fix, o regime deve bater com o projeto antigo:
```
Projeto antigo: Sideways (p=0.992)
Projeto novo:   Sideways (p≈0.99)  ← deve concordar
```

Se concordar → o sistema sai de BLOCK_BEAR e começa a operar.
