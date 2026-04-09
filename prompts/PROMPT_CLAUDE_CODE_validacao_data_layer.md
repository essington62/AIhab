# PROMPT — Validação completa da camada de dados

## Objetivo
Rodar cada módulo de ingestão, verificar que os parquets foram criados,
e confirmar que temos todos os dados necessários para os gates.

## Passo 1: Rodar cada módulo de ingestão

```bash
cd /Users/brown/Documents/MLGeral/btc_AI

# Ativar ambiente
conda activate btc_trading_v1

# Binance Spot
echo "=== Binance Spot ==="
python -m src.data.binance_spot --timeframe 1h
python -m src.data.binance_spot --timeframe 1d

# Binance Futures
echo "=== Binance Futures ==="
python -m src.data.binance_futures

# FRED
echo "=== FRED ==="
python -m src.data.fred_ingest

# CoinGlass
echo "=== CoinGlass ==="
python -m src.data.coinglass_ingest

# Fear & Greed
echo "=== Alt.me ==="
python -m src.data.altme_ingest

# News
echo "=== News ==="
python -m src.data.news_ingest

# Clean (raw → intermediate)
echo "=== Clean ==="
python -m src.data.clean
```

Se algum módulo falhar, anotar o erro e continuar com os outros.

## Passo 2: Inventário completo dos parquets

```python
import pandas as pd
from pathlib import Path

base = Path("data")

print("=" * 80)
print("INVENTÁRIO COMPLETO DE DADOS")
print("=" * 80)

for layer in ["01_raw", "02_intermediate"]:
    layer_path = base / layer
    if not layer_path.exists():
        print(f"\n⚠️  {layer}/ NÃO EXISTE")
        continue
    
    print(f"\n{'─' * 80}")
    print(f"📁 {layer}/")
    print(f"{'─' * 80}")
    
    for p in sorted(layer_path.rglob("*.parquet")):
        try:
            df = pd.read_parquet(p)
            rel = p.relative_to(base)
            
            # Encontrar coluna de timestamp
            ts_col = None
            for c in df.columns:
                if c == "timestamp":
                    ts_col = c
                    break
            
            if ts_col:
                first = df[ts_col].min()
                last = df[ts_col].max()
                tz = str(df[ts_col].dt.tz) if hasattr(df[ts_col].dt, 'tz') and df[ts_col].dt.tz else "NO TZ ⚠️"
                date_info = f"| {first} → {last} | tz={tz}"
            else:
                date_info = "| NO timestamp col ⚠️"
            
            nan_count = df.isna().sum().sum()
            nan_info = f"| NaN={nan_count}" if nan_count > 0 else ""
            
            print(f"  {str(rel):50s} {len(df):>6} rows | cols={list(df.columns)} {date_info} {nan_info}")
        except Exception as e:
            print(f"  ❌ {p.relative_to(base)}: {e}")

# Verificar JSONs
for p in sorted(base.rglob("*.json")):
    print(f"  {p.relative_to(base)}: JSON exists")
```

## Passo 3: Checklist — temos tudo pra cada gate?

```python
"""
Verificar que cada gate tem os dados que precisa.
"""

from src.config import get_path
import pandas as pd

GATE_REQUIREMENTS = {
    "G1 Technical (BB+RSI)": {
        "source": "clean_spot_1h" if "clean_spot_1h" in str(get_path) else "spot_1h",
        "columns_needed": ["close"],  # pra calcular BB e RSI
        "frequency": "1h",
    },
    "G3 Macro (FRED)": {
        "files": ["macro_dgs10", "macro_dgs2", "macro_rrp"],
        "frequency": "daily",
    },
    "G4 Positioning (OI)": {
        "source": "futures_oi",
        "columns_needed": ["open_interest_value", "open_interest"],
        "frequency": "1h",
    },
    "G5 Liquidity (Stablecoin)": {
        "source": "coinglass_stablecoin",
        "frequency": "daily",
    },
    "G6 Bubble Index": {
        "source": "coinglass_bubble",
        "frequency": "daily",
    },
    "G7 ETF Flows": {
        "source": "coinglass_etf",
        "frequency": "daily",
    },
    "G8 Fear & Greed": {
        "source": "sentiment_fg",
        "frequency": "daily",
    },
    "G9 Taker Ratio": {
        "source": "futures_taker",
        "columns_needed": ["buy_sell_ratio"],
        "frequency": "1h",
    },
    "G10 Funding Rate": {
        "source": "futures_funding",
        "columns_needed": ["funding_rate"],
        "frequency": "8h → ffill 1h",
    },
    "R5C HMM": {
        "source": "spot_1d",
        "columns_needed": ["open", "high", "low", "close", "volume"],
        "frequency": "daily",
    },
}

print("\n" + "=" * 80)
print("CHECKLIST POR GATE")
print("=" * 80)

for gate, req in GATE_REQUIREMENTS.items():
    print(f"\n  {gate}:")
    
    sources = req.get("files", [req["source"]]) if "source" in req else req.get("files", [])
    
    for src_name in sources:
        try:
            path = get_path(src_name)
            if Path(path).exists():
                df = pd.read_parquet(path)
                cols = list(df.columns)
                rows = len(df)
                
                # Verificar colunas necessárias
                needed = req.get("columns_needed", [])
                missing_cols = [c for c in needed if c not in cols]
                
                if missing_cols:
                    print(f"    ⚠️  {src_name}: {rows} rows, cols={cols}")
                    print(f"       MISSING COLUMNS: {missing_cols}")
                else:
                    # Verificar freshness
                    if "timestamp" in cols:
                        last = pd.to_datetime(df["timestamp"]).max()
                        hours_ago = (pd.Timestamp.now(tz="UTC") - last).total_seconds() / 3600
                        fresh = "✅ FRESH" if hours_ago < 48 else f"⚠️ {hours_ago:.0f}h ago"
                    else:
                        fresh = "⚠️ no timestamp"
                    
                    print(f"    ✅ {src_name}: {rows} rows, cols={cols[:5]}... {fresh}")
            else:
                print(f"    ❌ {src_name}: FILE NOT FOUND at {path}")
        except Exception as e:
            print(f"    ❌ {src_name}: {e}")
```

## Passo 4: Verificar dados intermediários (clean)

```python
print("\n" + "=" * 80)
print("02_INTERMEDIATE (clean)")
print("=" * 80)

clean_files = {
    "clean_futures_oi": "OI resampled 1H grid",
    "clean_futures_taker": "Taker resampled 1H grid",
    "clean_futures_funding": "Funding forward-filled 1H",
    "clean_macro": "FRED daily aligned",
}

for name, desc in clean_files.items():
    try:
        path = get_path(name)
        if Path(path).exists():
            df = pd.read_parquet(path)
            
            # Verificar grid 1H (pra futures)
            if "futures" in name and "timestamp" in df.columns:
                ts = pd.to_datetime(df["timestamp"])
                diffs = ts.diff().dropna()
                is_hourly = (diffs == pd.Timedelta(hours=1)).mean()
                grid_status = f"1H grid: {is_hourly:.0%}" if is_hourly > 0.9 else f"⚠️ gaps: {is_hourly:.0%} hourly"
            else:
                grid_status = ""
            
            # Verificar NaN
            nan_pct = df.isna().mean()
            nan_cols = [f"{c}={v:.0%}" for c, v in nan_pct.items() if v > 0]
            
            print(f"  ✅ {name}: {len(df)} rows | {desc}")
            print(f"     Cols: {list(df.columns)}")
            if grid_status:
                print(f"     {grid_status}")
            if nan_cols:
                print(f"     NaN: {', '.join(nan_cols)}")
        else:
            print(f"  ❌ {name}: NOT FOUND")
    except Exception as e:
        print(f"  ❌ {name}: {e}")
```

## Passo 5: Migração de dados históricos

```bash
# Verificar se a migração já rodou
echo "=== Dados migrados do projeto antigo ==="
ls -la data/01_raw/coinglass/ 2>/dev/null || echo "CoinGlass: NÃO MIGRADO"
ls -la data/03_models/ 2>/dev/null || echo "R5C model: NÃO MIGRADO"
```

Se NÃO migrou ainda:
```bash
python scripts/migrate_historical.py
```

Se o script não existe ainda, criar e rodar manualmente:
```python
import shutil
from pathlib import Path

OLD = Path("/Users/brown/Documents/MLGeral/crypto_v2/crypto-market-state/data")
NEW = Path("/Users/brown/Documents/MLGeral/btc_AI/data")

# Listar o que existe no projeto antigo
print("=== Projeto antigo ===")
for p in sorted(OLD.rglob("*.parquet"))[:30]:
    print(f"  {p.relative_to(OLD)}")
for p in sorted(OLD.rglob("*.pkl")):
    print(f"  {p.relative_to(OLD)}")
```

## Passo 6: Backfill Binance Futures

```bash
# Verificar se temos 30 dias de histórico
python -c "
import pandas as pd
from pathlib import Path

for f in ['oi_1h', 'taker_1h', 'funding']:
    p = Path(f'data/01_raw/futures/{f}.parquet')
    if p.exists():
        df = pd.read_parquet(p)
        ts = pd.to_datetime(df['timestamp'])
        days = (ts.max() - ts.min()).days
        print(f'{f}: {len(df)} rows, {days} days span ({ts.min()} → {ts.max()})')
    else:
        print(f'{f}: NOT FOUND')
"
```

Se < 14 dias de span:
```bash
python scripts/backfill_binance_futures.py
```

## Passo 7: DQ Report

```bash
python -m src.data.dq
```

## Resultado esperado

Preciso ver:
1. Todos os módulos rodam sem erro fatal
2. Parquets criados em 01_raw/ e 02_intermediate/
3. Timestamps em UTC (tz-aware)
4. Futures em grid 1H consistente (>90% hourly)
5. Funding forward-filled
6. Cada gate tem os dados necessários (checklist verde)
7. Dados históricos migrados (CoinGlass, R5C model)
8. Binance Futures com ≥14 dias de span
