# Prompt: Dynamic TP v2 (Bot 2 BTC) + Fix Dashboard (SOL trades)

## Contexto

Duas mudanças consolidadas em um único deploy:

1. **Dynamic TP v2** para Bot 2 BTC — aplicação direta (sem study, risco aceito)
2. **Fix dashboard** — SOL trades não aparecem (schema/path mismatch)

**SOL Bot 4 será pausado separadamente (via crontab).** Este prompt NÃO mexe em Bot 4.

---

# PARTE 1: Dynamic TP v2 — Bot 2 BTC

## Proposta

```python
def get_dynamic_tp(rsi, bb_pct, volume_z):
    """
    Dynamic TP para Bot 2 BTC baseado em contexto do entry.
    
    Rule 1: volume extreme → exaustão iminente
    Rule 2: overbought (RSI + BB) → continuation limitada
    Rule 3: default → baseline validado (2%)
    """
    if volume_z > 1.0:
        return 0.010, "volume_exhaustion"
    elif rsi > 75 and bb_pct > 0.95:
        return 0.015, "overbought"
    else:
        return 0.020, "default"
```

## Rationale

**Por que aplicar sem study:**
- Usuário assume risco (decisão informada)
- v1 (buckets complexos, 20/04) foi rejeitado
- v2 é DIFERENTE: 3 regras simples, default = baseline
- Trade 22/04 Bot 2 teria melhorado: MFE +1.88%, TRAIL +0.70% → v2 TP 1.5% = +1.50% (+0.80pp)
- Risco contido: SL/Trailing inalterados, só TP varia

**Reversibilidade:** commit separado pra rollback fácil.

## Tarefas Dynamic TP

### 1. Criar `src/trading/dynamic_tp.py`

```python
"""
src/trading/dynamic_tp.py

Dynamic TP v2 — 3-rule system for Bot 2 BTC.

Status: applied 22/04/2026 without prior backtest study.
User accepted risk of deviation from baseline (Fixed 2% Sharpe 2.71).

History:
- v1 (20/04): rejected. Complex buckets. Sharpe -0.15.
- v2 (22/04): simplified. Default = baseline. Only modifies edge cases.
"""

import logging

logger = logging.getLogger(__name__)


def get_dynamic_tp(rsi: float, bb_pct: float, volume_z: float) -> tuple[float, str]:
    """
    Calculate dynamic TP for Bot 2 based on entry context.
    
    Args:
        rsi: RSI at entry (0-100)
        bb_pct: Bollinger Band percentage (0-1)
        volume_z: Volume z-score rolling 7d
    
    Returns:
        (tp_pct, reason): TP as fraction (0.02 = 2%), reason label
    """
    if volume_z is not None and volume_z > 1.0:
        return 0.010, "volume_exhaustion"
    elif rsi is not None and bb_pct is not None and rsi > 75 and bb_pct > 0.95:
        return 0.015, "overbought"
    else:
        return 0.020, "default"


def log_tp_decision(entry_price, tp_pct, reason, rsi, bb_pct, volume_z):
    tp_price = entry_price * (1 + tp_pct)
    logger.info(
        f"BOT2 Dynamic TP: {tp_pct*100:.1f}% ({reason}) | "
        f"TP ${tp_price:,.2f} | RSI={rsi:.1f} BB={bb_pct:.3f} VolZ={volume_z:+.2f}"
    )
```

### 2. Integrar em `src/trading/paper_trader.py`

**Localizar** `_execute_bot2_entry()` e similar:

```bash
grep -n "_execute_bot2_entry\|take_profit.*bot2\|bot2.*take_profit" src/trading/paper_trader.py
```

**Modificar o entry** para usar TP dinâmico:

```python
# No topo do arquivo:
from src.trading.dynamic_tp import get_dynamic_tp, log_tp_decision

# Dentro da função de entry Bot 2:
def _execute_bot2_entry(current_price, portfolio, mf_params, mf_check):
    # Extrair features do mf_check
    rsi = mf_check.get("rsi")
    bb_pct = mf_check.get("bb_pct")
    volume_z = mf_check.get("volume_z")  # pode ser None
    
    # Dynamic TP
    tp_pct, tp_reason = get_dynamic_tp(rsi, bb_pct, volume_z)
    log_tp_decision(current_price, tp_pct, tp_reason, rsi, bb_pct, volume_z)
    
    # Fallback caso alguma feature seja None
    if tp_pct is None:
        tp_pct = mf_params.get("take_profit_pct", 0.02)
        tp_reason = "fallback_default"
    
    sl_pct = mf_params.get("stop_loss_pct", 0.015)
    
    tp_price = current_price * (1 + tp_pct)
    sl_price = current_price * (1 - sl_pct)
    
    # Persistir no trade record
    portfolio["tp_reason"] = tp_reason
    portfolio["dynamic_tp_pct"] = tp_pct
    
    # ... resto do fluxo existente (quantidade, logging, etc)
```

### 3. Garantir `volume_z` disponível no `mf_check`

**Verificar** se `check_bot2_filters()` (ou equivalente) computa volume_z:

```bash
grep -n "check_bot2\|momentum_filter\|volume_z" src/trading/*.py src/features/*.py
```

**Se NÃO computa, adicionar:**

```python
# Em src/features/momentum_filters.py (ou onde filters Bot 2 vivem):

def compute_volume_z(df, window=168):
    """Volume z-score rolling 7d (168 candles 1h)."""
    if len(df) < window:
        return 0.0  # neutro se insuficiente
    
    recent = df["volume"].tail(window)
    current = df["volume"].iloc[-1]
    
    mean = recent.mean()
    std = recent.std()
    
    if std == 0 or pd.isna(std):
        return 0.0
    
    return (current - mean) / std


# Integrar em check_bot2_filters:
def check_bot2_filters(df, params):
    # ... código existente ...
    
    # NOVO: volume_z
    volume_z = compute_volume_z(df)
    
    result = {
        "passed": all_passed,
        "stablecoin_z": stablecoin_z,
        "ret_1d": ret_1d,
        "rsi": rsi,
        "bb_pct": bb_pct,
        "volume_z": volume_z,  # NOVO
        "close_above_ma21": close_above_ma21,
    }
    return result
```

### 4. Testes unitários

```python
# tests/test_dynamic_tp.py
import pytest
from src.trading.dynamic_tp import get_dynamic_tp


def test_rule_1_volume_exhaustion():
    tp, reason = get_dynamic_tp(rsi=60, bb_pct=0.5, volume_z=1.5)
    assert tp == 0.010
    assert reason == "volume_exhaustion"


def test_rule_2_overbought():
    tp, reason = get_dynamic_tp(rsi=78, bb_pct=0.97, volume_z=0.5)
    assert tp == 0.015
    assert reason == "overbought"


def test_rule_2_only_rsi_not_enough():
    # RSI alto mas BB baixo: NÃO overbought
    tp, reason = get_dynamic_tp(rsi=78, bb_pct=0.8, volume_z=0.5)
    assert tp == 0.020
    assert reason == "default"


def test_rule_3_default():
    tp, reason = get_dynamic_tp(rsi=65, bb_pct=0.7, volume_z=0.2)
    assert tp == 0.020
    assert reason == "default"


def test_rule_1_dominates_rule_2():
    # Volume extreme + overbought: Rule 1 vence
    tp, reason = get_dynamic_tp(rsi=80, bb_pct=0.98, volume_z=1.5)
    assert tp == 0.010
    assert reason == "volume_exhaustion"


def test_none_values():
    # Features None: deve retornar default
    tp, reason = get_dynamic_tp(rsi=None, bb_pct=None, volume_z=None)
    assert tp == 0.020
    assert reason == "default"
```

---

# PARTE 2: Fix Dashboard — SOL trades

## Problema

Primeiro trade SOL fechou (22/04, -0.98%). Arquivo existe mas dashboard não mostra.

**Arquivo real:**
```
/app/data/05_trades/completed_trades_sol.json

Schema:
{
  "symbol": "SOLUSDT",
  "entry_price": 88.23,
  "exit_price": 87.3675,
  "quantity": 113.340134,
  "entry_timestamp": "2026-04-22T16:15:01+00:00",
  "exit_timestamp": "2026-04-22T19:43:00+00:00",
  "exit_reason": "TRAIL",
  "pnl_pct": -0.009776,
  "pnl_usd": -97.76,
  "entry_features": {
    "close": 88.33, "rsi": 62.4, "ma21": 87.45,
    "ret_1d": 0.0308, "taker_z_prev": 0.96,
    "oi_z_prev": 1.84, "oi_z_24h_max": 1.84,
    "oi_z_24h_max_prev": 1.84, "eth_ret_1h_prev": 0.0003
  }
}
```

**Dashboard procura em:** path diferente + schema diferente.

## Tarefas Fix Dashboard

### 1. Identificar função de load

```bash
grep -n "def load_trades\|load_trades_filtered\|load_sol_trades" src/dashboard/app.py
```

### 2. Criar/atualizar adapter para SOL

```python
# Em src/dashboard/app.py

def load_sol_trades():
    """
    Load SOL trades from /data/05_trades/ and unify schema.
    
    Returns DataFrame compatible with render_trades_table().
    """
    from pathlib import Path
    import json
    
    path = Path("data/05_trades/completed_trades_sol.json")
    if not path.exists():
        return pd.DataFrame()
    
    try:
        with open(path) as f:
            raw_trades = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        return pd.DataFrame()
    
    if not raw_trades:
        return pd.DataFrame()
    
    # Adapter: SOL schema → unified
    unified = []
    for t in raw_trades:
        entry_price = t["entry_price"]
        
        unified.append({
            "entry_time": pd.to_datetime(t["entry_timestamp"], utc=True),
            "exit_time": pd.to_datetime(t["exit_timestamp"], utc=True),
            "entry_price": entry_price,
            "exit_price": t["exit_price"],
            "return_pct": t["pnl_pct"] * 100,  # fração → %
            "exit_reason": t["exit_reason"],
            # SL/TP calculados retroativamente (SOL usa 1.5%/2%)
            "stop_loss_price": entry_price * 0.985,
            "take_profit_price": entry_price * 1.02,
            "trailing_high": entry_price * 1.002,  # aproximação
            # Metadata
            "bot_origin": "sol_bot4",
            "symbol": t.get("symbol", "SOLUSDT"),
            "pnl_usd": t.get("pnl_usd", 0),
            "quantity": t.get("quantity", 0),
            # Features (opcional, se render mostrar)
            "stablecoin_z": None,  # SOL não tem esse campo
            "rsi": t.get("entry_features", {}).get("rsi"),
        })
    
    return pd.DataFrame(unified)
```

### 3. Integrar no `load_trades_filtered()` ou similar

```python
# Atualizar load_trades_filtered para rotear por bot:

def load_trades_filtered(asset="btc", bot=None):
    """Load trades, route by asset/bot."""
    
    if asset == "sol" and bot == "bot4":
        return load_sol_trades()
    
    # ... código existente para BTC e ETH ...
```

### 4. Atualizar `render_trades_table()` para ser resiliente

```python
def render_trades_table(trades_df, include_stops=True):
    """Render unified trades table, resilient to missing fields."""
    if trades_df.empty:
        st.info("Sem trades")
        return
    
    # Garantir colunas obrigatórias
    for col in ["stop_loss_price", "take_profit_price", "trailing_high",
                "stablecoin_z", "rsi"]:
        if col not in trades_df.columns:
            trades_df[col] = None
    
    # ... resto do render ...
```

### 5. Performance Monitoring — Bot 4 atualizar

Na função `compute_bot_health()` ou similar, agora Bot 4 vai ter dados:

```python
# Em Performance Monitoring:
for bot_info in [
    {"key": "bot2", "emoji": "🚀", "name": "BOT 2 - Momentum (BTC)", "asset": "btc"},
    {"key": "bot3", "emoji": "⚡", "name": "BOT 3 - Volume Defensivo (ETH)", "asset": "eth"},
    {"key": "bot4", "emoji": "🟣", "name": "BOT 4 - Taker/Flow (SOL)", "asset": "sol"},
]:
    trades = load_trades_filtered(asset=bot_info["asset"], bot=bot_info["key"])
    # ... resto do código ...
```

Agora SOL vai mostrar:
- 1 trade
- WR 0% (1 loss)
- Avg Return -0.98%
- Total Return -0.98%

---

# PARTE 3: Deploy unificado

## Validação local

```bash
cd /Users/brown/Documents/MLGeral/btc_AI

# 1. Testes Dynamic TP
python -c "from src.trading.dynamic_tp import get_dynamic_tp; print(get_dynamic_tp(78, 0.97, 0.5))"
# Esperado: (0.015, 'overbought')

# 2. Rodar testes unitários
pytest tests/test_dynamic_tp.py -v

# 3. Streamlit local
streamlit run src/dashboard/app.py

# Browser: verificar SOL trade aparece
```

## Checklist visual

```
☐ Dashboard: SOL Bot 4 seção mostra 1 trade:
  Entry 22/04 16:15 | $88.23
  Exit $87.37 | TRAIL | -0.98%
  
☐ Performance Monitoring — Bot 4:
  Trades: 1
  WR: 0%
  Avg Return: -0.98%
  Total Return: -0.98%
  🟡 Início (n=1)
  
☐ Bot 2 BTC continua mostrando 5 trades (inalterado)
☐ Resto do dashboard OK
```

## Commit estruturado

**Dois commits separados pra rollback granular:**

```bash
# Commit 1: Dynamic TP
git add src/trading/dynamic_tp.py \
        src/trading/paper_trader.py \
        src/features/momentum_filters.py \
        tests/test_dynamic_tp.py

git commit -m "feat(bot2): Dynamic TP v2 — 3-rule system

Replace Fixed 2% TP with context-aware rules:
- volume_z > 1.0: TP 1% (exhaustion)
- RSI > 75 AND BB > 0.95: TP 1.5% (overbought)
- else: TP 2% (default = baseline)

Trade 22/04 case: MFE +1.88% but trailed +0.70%.
Dynamic v2 Rule 2 (RSI=78, BB=0.97) → TP 1.5% = +1.50%.

Risk acknowledged: applied without backtest.
Different from v1 (rejected): simpler rules, default = baseline.
Reversible via git revert."

# Commit 2: Dashboard fix
git add src/dashboard/app.py

git commit -m "fix(dashboard): SOL trades schema adapter

- Add load_sol_trades() reading /data/05_trades/completed_trades_sol.json
- Map SOL schema (pnl_pct, entry_timestamp) to unified format
- render_trades_table resilient to missing SL/TP
- First SOL trade now visible: 22/04 entry \$88.23 exit \$87.37 TRAIL -0.98%
- Performance Monitoring Bot 4 shows real metrics (1 trade, WR 0%)"

# Push
git push origin main
```

## Deploy AWS

**REBUILD obrigatório** (Dockerfile COPY):

```bash
ssh -i ~/.ssh/aihab-key-sp.pem ubuntu@54.232.162.161 << 'EOF'
cd ~/AIhab
git pull origin main
echo "=== Commits recentes ==="
git log --oneline -5

echo ""
echo "=== Rebuild app + dashboard ==="
docker compose build app dashboard

echo ""
echo "=== Recreate containers ==="
docker compose up -d --force-recreate --no-deps app dashboard

echo ""
echo "=== Aguardar startup ==="
sleep 8

echo ""
echo "=== Status ==="
docker compose ps

echo ""
echo "=== Logs app (últimas 10) ==="
docker compose logs --tail 10 app

echo ""
echo "=== Logs dashboard (últimas 10) ==="
docker compose logs --tail 10 dashboard

echo ""
echo "=== Verificar Dynamic TP dentro container ==="
docker exec aihab-app python3 -c "
from src.trading.dynamic_tp import get_dynamic_tp
print('Rule 1:', get_dynamic_tp(60, 0.5, 1.5))
print('Rule 2:', get_dynamic_tp(78, 0.97, 0.5))
print('Rule 3:', get_dynamic_tp(65, 0.7, 0.2))
"

echo ""
echo "=== Verificar SOL fix dentro container ==="
docker exec aihab-dashboard ls -la /app/data/05_trades/
EOF
```

## Validação pós-deploy

**Browser em `http://54.232.162.161:8501`:**

```
☐ SOL Bot 4 mostra trade (22/04 -0.98%)
☐ Performance Monitoring Bot 4 com métricas
☐ Bot 2 inalterado (5 trades)
☐ Outras seções OK
☐ Sem erros Python

Próximo cycle Bot 2 (minuto :05):
☐ Log mostra "BOT2 Dynamic TP: X% (reason)"
☐ Se entrar trade: usa TP dinâmico
```

---

# PARTE 4: Monitoring pós-deploy

## Dynamic TP — próximos 10 trades Bot 2

```
Tracking:
  - Qual regra disparou em cada trade?
  - Rule 1 (volume_z > 1): quantos trades?
  - Rule 2 (overbought): quantos trades?
  - Rule 3 (default): quantos trades?
  - Performance por regra: WR, avg return
```

## Fix dashboard — funcionando

```
Confirmar:
  - SOL trade aparece
  - Métricas Bot 4 corretas
  - Sem erros
  - Próximo trade SOL (quando reativar): aparece automaticamente
```

---

# Critérios de qualidade

- ✅ Commits separados (rollback granular)
- ✅ Testes unitários Dynamic TP
- ✅ Dashboard resiliente (missing fields)
- ✅ Deploy sem quebrar bots
- ✅ SOL trade visível
- ✅ Bot 2 usa Dynamic TP em próximo entry
- ✅ Monitoring facilitado (logs + dashboard)

# Tempo estimado

- Dynamic TP implementation: 25 min
- Testes unitários: 10 min
- Dashboard fix: 15 min
- Validação local: 15 min
- Deploy + validação AWS: 15 min
- **Total: ~1h20min**

# Rollback (se necessário)

```bash
# Revert Dynamic TP:
git revert <commit_dynamic_tp>
git push origin main
ssh AWS: git pull + docker compose build app + up -d --force-recreate

# Revert dashboard:
git revert <commit_dashboard>
git push origin main
ssh AWS: git pull + docker compose build dashboard + up -d --force-recreate
```

Commits separados permitem revert individual sem afetar o outro.
