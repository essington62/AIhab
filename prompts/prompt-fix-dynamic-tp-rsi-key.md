# Prompt: FIX — Dynamic TP bug (RSI None)

## Bug identificado

**Dynamic TP v2 está funcionando mas com bug de integração.**

Validação em produção mostrou:

```bash
docker exec aihab-app python3 -c "
from src.features.technical import get_latest_technical
result = get_latest_technical()
print('Volume_z:', result.get('volume_z'))  # ✅ -0.9025
print('RSI:', result.get('rsi'))             # ❌ None!
print('BB_pct:', result.get('bb_pct'))       # ✅ 0.673
print('Keys:', sorted(result.keys()))
"
```

**Output revelou:**
- `volume_z` funciona ✅
- `bb_pct` funciona ✅
- `rsi` retorna None ❌
- Chave correta no dict é `rsi_14` (não `rsi`)

## Impacto

```python
# Em _execute_bot2_entry (provável):
rsi = mf_check.get("rsi")      # ← sempre None
bb_pct = mf_check.get("bb_pct")
volume_z = mf_check.get("volume_z")

get_dynamic_tp(None, bb_pct, volume_z)
# Rule 2 (RSI > 75 AND BB > 0.95) NUNCA dispara
# Rule 3 (default) sempre dispara quando volume_z <= 1
```

**Consequência:** Bot 2 nunca usa TP 1.5% (overbought). v2 praticamente vira baseline.

## Fix simples

### Opção A (preferida): corrigir chave na origem

Em `src/features/technical.py`, adicionar alias ao final do dict retornado:

```python
def get_latest_technical():
    # ... código existente ...
    
    result = {
        "rsi_14": rsi,
        # ... outros ...
    }
    
    # NOVO: alias pra compatibilidade
    result["rsi"] = result.get("rsi_14")
    
    return result
```

### Opção B (alternativa): corrigir no consumer

Em `src/trading/paper_trader.py`, onde `_execute_bot2_entry` lê `mf_check`:

```python
# ANTES:
rsi = mf_check.get("rsi")

# DEPOIS:
rsi = mf_check.get("rsi_14", mf_check.get("rsi"))  # fallback
```

**Recomendação: Opção A** (mais cirúrgica, beneficia outros consumers).

## Tarefas

### 1. Confirmar onde está o bug

```bash
grep -n "mf_check.get\|rsi.*get\|get.*rsi" src/trading/paper_trader.py | head -10
grep -n "result.*rsi\|rsi_14\|return.*{" src/features/technical.py | head -10
```

### 2. Aplicar Opção A

Em `src/features/technical.py`, no return do `get_latest_technical()`:

```python
# Adicionar antes do return:
if "rsi_14" in result and "rsi" not in result:
    result["rsi"] = result["rsi_14"]

return result
```

### 3. Testar local

```bash
python -c "
from src.features.technical import get_latest_technical
r = get_latest_technical()
print('rsi:', r.get('rsi'))        # deve retornar valor, não None
print('rsi_14:', r.get('rsi_14'))  # deve retornar mesmo valor
assert r.get('rsi') == r.get('rsi_14'), 'Bug!'
print('OK!')
"
```

### 4. Commit e deploy

```bash
git add src/features/technical.py
git commit -m "fix(technical): add 'rsi' alias for 'rsi_14' in get_latest_technical

Dynamic TP v2 Rule 2 (RSI > 75 AND BB > 0.95) was never firing because
paper_trader.py looked up mf_check.get('rsi') but technical.py returns
'rsi_14'. Added alias for backward compat.

Tested: get_latest_technical()['rsi'] now returns valid value."

git push origin main

# Deploy AWS:
ssh AWS:
  cd ~/AIhab
  git pull
  docker compose build app
  docker compose up -d --force-recreate --no-deps app
```

### 5. Validar em produção

```bash
docker exec aihab-app python3 -c "
from src.features.technical import get_latest_technical
r = get_latest_technical()
print('RSI via rsi:', r.get('rsi'))
print('RSI via rsi_14:', r.get('rsi_14'))
print('Both should be the same and non-None')
"
```

## Impacto esperado

**Antes do fix:**
- Rule 2 nunca dispara
- v2 ≈ baseline
- Bot 2 sempre TP 2%

**Após fix:**
- Rule 1 funciona (volume_z)
- Rule 2 funciona (RSI + BB)
- Rule 3 funciona (default)
- v2 totalmente funcional

## Tempo

- Fix: 2 min
- Test: 1 min
- Deploy: 3 min
- **Total: ~6 min**

## Critério de sucesso

Após deploy:
```python
# Output esperado:
RSI via rsi: 68.5  # ou qualquer número não-None
RSI via rsi_14: 68.5
```

Próximo cycle Bot 2 (quando disparar), verificar log:
```bash
docker exec aihab-app grep "BOT2 Dynamic TP" /app/logs/hourly.log
```

Se log mostrar "Rule 2 (overbought)" algum dia = fix funcionou.
