# Prompt: Revisão SOL Bot 4 — Auditoria pós-deploy

## Contexto

SOL Bot 4 foi implementado e deployado na AWS **sem seguir algumas diretrizes críticas** do prompt original (Phase 3). Este prompt é uma **auditoria de conformidade** para identificar gaps e corrigir antes que comece a gerar trades em paper.

**Status:** já em produção paper trading AWS (54.232.162.161).

**Prioridade:** encontrar bugs/gaps ANTES do primeiro trade real para evitar:
1. Look-ahead bias em decisões
2. Capital contamination (afetar BTC/ETH)
3. Decisões com features erradas
4. Shadow scoring não funcionando

## Objetivo da revisão

Fazer **auditoria sistemática** em 8 pontos críticos, reportar discrepâncias, e gerar patches corretivos onde necessário.

## 8 Pontos críticos a auditar

### PONTO 1: Anti look-ahead em TODAS as features preditoras

**Especificação esperada:**
```python
# Features usadas como PREDITORAS devem usar shift(1)
taker_z_prev = df["taker_z"].shift(1)
eth_ret_1h_prev = df["eth_ret_1h"].shift(1)
oi_z_prev = df["oi_z"].shift(1)
stablecoin_z_prev = df["stablecoin_z"].shift(1)

# Features que descrevem o candle ATUAL (close, ma21, rsi, ret_1d)
# NÃO precisam shift — elas definem estado atual, não previsão
```

**O que verificar:**
```
☐ taker_z usado na decisão é o PREV (4h candle anterior fechado)?
☐ eth_ret_1h usado é o PREV (1h candle anterior fechado)?
☐ oi_z_24h usado é do candle anterior?
☐ stablecoin_z usado é do candle anterior?

CRÍTICO: Se feature CURRENT é usada, o bot "vê o futuro" e Sharpe 
backtest (2.03) NÃO refletirá performance real. Pode dar trades 
perfeitos no backtest e péssimos em produção.
```

**Como auditar:**
```bash
# Procurar onde features são USADAS na decisão
grep -n "taker_z\|eth_ret_1h\|oi_z\|stablecoin_z" src/trading/sol_bot4.py

# Cada ocorrência em check_entry_signal() deveria ter "_prev" no nome
```

---

### PONTO 2: Capital simulado separado

**Especificação esperada:**
```python
# portfolio_state_sol.json (SEPARADO de portfolio_state.json de BTC)
# NÃO deve aparecer em total_capital de BTC/ETH
# NÃO deve ser somado no MultiAssetManager no dashboard atual
```

**O que verificar:**
```
☐ portfolio_state_sol.json existe em /app/data/05_output/?
☐ É arquivo SEPARADO do BTC portfolio_state.json?
☐ Capital inicial é $10,000.00 (não tomou de BTC)?
☐ Se há bug: SOL Bot usando BTC capital?
☐ MultiAssetManager adicionou bucket "sol_bot4" sem mexer BTC/ETH buckets?
☐ Total capital BTC continua $9,859.32 (ou similar, unchanged)?
```

**Como auditar:**
```bash
ssh -i ~/.ssh/aihab-key-sp.pem ubuntu@54.232.162.161 << 'EOF'
cd ~/AIhab

echo "=== Portfolio states existentes ==="
ls -la data/05_output/portfolio_state*.json

echo ""
echo "=== BTC capital (deve ser ~$9,859) ==="
cat data/05_output/portfolio_state.json | python3 -c "
import json, sys
d = json.load(sys.stdin)
print(f'Capital BTC total: \${d.get(\"total_capital_usd\", 0):,.2f}')
print(f'Bot1: \${d[\"buckets\"][\"btc_bot1\"][\"current_capital\"]:,.2f}')
print(f'Bot2: \${d[\"buckets\"][\"btc_bot2\"][\"current_capital\"]:,.2f}')
"

echo ""
echo "=== SOL capital (deve ser exatamente $10,000) ==="
if [ -f data/05_output/portfolio_state_sol.json ]; then
    cat data/05_output/portfolio_state_sol.json | python3 -m json.tool | head -20
else
    echo "ARQUIVO NÃO EXISTE — problema?"
fi

echo ""
echo "=== ETH capital ==="
if [ -f data/05_output/portfolio_state_eth.json ] || [ -f data/05_output/portfolio_eth.json ]; then
    find data/05_output -name "portfolio*eth*.json" -exec cat {} \;
fi
EOF
```

---

### PONTO 3: Shadow scoring logging funcionando

**Especificação esperada:**
```python
# A cada ENTRY do bot, logar também a decisão do scoring:
#   score = taker(2) + eth(1) + oi(1)
#   if score >= 3: would_enter=True
#
# Path: data/08_shadow/sol_scoring_shadow_log.jsonl
# Format: JSONL (uma entry por linha)
```

**O que verificar:**
```
☐ Função log_scoring_shadow() existe em sol_bot4.py?
☐ É chamada a CADA entry (confirmada ou blocked)?
☐ Path correto: data/08_shadow/sol_scoring_shadow_log.jsonl?
☐ Formato JSONL válido?
☐ Entry tem: timestamp, trade_id, score_total, would_enter, breakdown?
```

**Como auditar:**
```bash
# 1. Verificar código
grep -n "log_scoring_shadow\|scoring_shadow" src/trading/sol_bot4.py

# 2. Verificar arquivo de log (pode ser vazio se sem trade ainda)
ls -la data/08_shadow/sol_scoring_shadow_log.jsonl

# 3. Se existir, validar formato
if [ -f data/08_shadow/sol_scoring_shadow_log.jsonl ]; then
    cat data/08_shadow/sol_scoring_shadow_log.jsonl | python3 -c "
import json, sys
for i, line in enumerate(sys.stdin, 1):
    try:
        entry = json.loads(line)
        print(f'Entry {i}: OK — score={entry.get(\"score_total\", \"MISSING\")}, would_enter={entry.get(\"would_enter\", \"MISSING\")}')
    except Exception as e:
        print(f'Entry {i}: INVALID JSON - {e}')
"
fi
```

---

### PONTO 4: Early exit OI 24h após 12h

**Especificação esperada:**
```python
# Se posição aberta há >= 12h E oi_z_24h > 2.0:
#   EXIT com reason="OI_EARLY_EXIT"
#
# Esse é edge proprietário (OI bipolar insight)
```

**O que verificar:**
```
☐ Função check_early_exit() ou equivalente existe?
☐ Verifica hours_held >= 12?
☐ Verifica oi_z_24h > 2.0?
☐ Retorna exit_reason="OI_EARLY_EXIT"?
☐ É chamada no ciclo de stops check?
```

**Como auditar:**
```bash
grep -n "OI_EARLY_EXIT\|oi_z_24h" src/trading/sol_bot4.py

# Procurar se lógica está no check_exits() ou em função separada
grep -A 10 "check_exits\|check_early_exit" src/trading/sol_bot4.py | head -30
```

---

### PONTO 5: Cooldown 4h pós-exit

**Especificação esperada:**
```python
# Após fechar posição, sol_bot4 não pode abrir outra por 4h
# cooldown_until = exit_time + 4h
```

**O que verificar:**
```
☐ Cooldown implementado após close_position()?
☐ cooldown_hours = 4 (não herda do BTC bot que é diferente)?
☐ check_entry respeita o cooldown?
```

**Como auditar:**
```bash
grep -n "cooldown" src/trading/sol_bot4.py
```

---

### PONTO 6: Logs separados (não contaminar outros bots)

**Especificação esperada:**
```
logs/sol_bot4.log (separado de btc e eth)
NÃO escrever em logs/hourly_cycle.log ou outros bots
```

**O que verificar:**
```
☐ Logger configurado para path sol_bot4.log?
☐ Não interfere em outros bot loggers?
☐ Format padrão do projeto?
```

**Como auditar:**
```bash
ssh -i ~/.ssh/aihab-key-sp.pem ubuntu@54.232.162.161 "
cd ~/AIhab
ls -la logs/sol_bot4.log 2>/dev/null || echo 'sol_bot4.log não existe'
ls -la logs/sol*.log 2>/dev/null
"

# Validar no código:
grep -n "FileHandler\|logging.*sol" scripts/sol_hourly_cycle.py
```

---

### PONTO 7: Timezones UTC em TUDO

**Especificação esperada:**
```python
# TUDO em UTC (tz-aware)
# pd.to_datetime(..., utc=True) sempre
# datetime.now(timezone.utc) sempre
```

**O que verificar:**
```
☐ pd.to_datetime() tem utc=True em TODOS os locais?
☐ datetime.now() tem timezone.utc?
☐ Timestamps salvos são ISO format UTC?
☐ Nenhum mistura naive + aware (causa erros silenciosos)?
```

**Como auditar:**
```bash
# Procurar timestamp handling
grep -n "pd.to_datetime\|datetime.now" src/trading/sol_bot4.py scripts/sol_*.py

# Cada ocorrência deveria ter utc=True ou timezone.utc
```

---

### PONTO 8: Integração MultiAssetManager sem quebrar BTC/ETH

**Especificação esperada:**
```python
# Adicionar "sol_bot4" bucket SEM modificar buckets existentes
# NÃO renomear chaves
# NÃO mexer em total_capital_usd dos outros
```

**O que verificar:**
```
☐ Bucket "sol_bot4" adicionado em multi_asset_manager.py?
☐ Buckets btc_bot1, btc_bot2, eth_bot3 inalterados?
☐ total_capital_usd não inclui SOL (isolado)?
☐ Sem breaking changes em API do manager?
```

**Como auditar:**
```bash
# Ver se buckets originais ainda estão lá
grep -A 3 "btc_bot1\|btc_bot2\|eth_bot3\|sol_bot4" src/trading/multi_asset_manager.py | head -40

# Testar que dashboard BTC continua funcional
curl -s http://localhost:8501/ 2>/dev/null || echo "dashboard offline, ver separado"
```

## Script de auditoria completo

Criar `scripts/audit_sol_bot4.py`:

```python
"""
Auditoria SOL Bot 4 — Verifica conformidade com specs críticas.

Rodar: python scripts/audit_sol_bot4.py
Output: AUDIT_SOL_BOT4.md com findings
"""

import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOL_BOT_PATH = ROOT / "src" / "trading" / "sol_bot4.py"
AUDIT_REPORT = ROOT / "AUDIT_SOL_BOT4.md"

issues = []

def check(condition, msg, severity="WARN"):
    """Log check result."""
    status = "✅" if condition else "❌"
    issues.append({
        "status": status,
        "severity": severity if not condition else "OK",
        "message": msg,
    })
    print(f"{status} {msg}")


def main():
    print("=" * 60)
    print("SOL Bot 4 — Auditoria de Conformidade")
    print("=" * 60)
    
    if not SOL_BOT_PATH.exists():
        print(f"❌ CRÍTICO: {SOL_BOT_PATH} não existe!")
        sys.exit(1)
    
    source = SOL_BOT_PATH.read_text()
    
    # ========== PONTO 1: Anti look-ahead ==========
    print("\n[1] Anti look-ahead em features preditoras")
    
    # Features que DEVEM ter _prev na decisão
    critical_features = ["taker_z", "eth_ret_1h", "oi_z", "stablecoin_z"]
    
    for feat in critical_features:
        # Procurar em check_entry_signal
        entry_section = extract_function(source, "check_entry_signal")
        if entry_section:
            has_prev = f"{feat}_prev" in entry_section
            check(has_prev, f"Feature '{feat}' usa versão _prev em check_entry_signal", severity="CRITICAL")
    
    # ========== PONTO 2: Capital separado ==========
    print("\n[2] Capital $10k separado de BTC/ETH")
    
    has_sol_state = "portfolio_state_sol.json" in source
    check(has_sol_state, "usa portfolio_state_sol.json (separado)", severity="CRITICAL")
    
    has_10k = "10000" in source or "10_000" in source
    check(has_10k, "Capital inicial $10,000 hardcoded", severity="HIGH")
    
    # ========== PONTO 3: Shadow scoring ==========
    print("\n[3] Shadow scoring logging")
    
    has_scoring_func = "log_scoring_shadow" in source or "scoring_shadow" in source
    check(has_scoring_func, "Função log_scoring_shadow existe", severity="HIGH")
    
    has_scoring_path = "sol_scoring_shadow_log.jsonl" in source
    check(has_scoring_path, "Path sol_scoring_shadow_log.jsonl", severity="HIGH")
    
    # Verificar se é CHAMADA (não só definida)
    # Procurar chamada em contexto de entry
    scoring_called = bool(re.search(r"log_scoring_shadow\s*\(", source))
    check(scoring_called, "log_scoring_shadow é CHAMADA no código", severity="CRITICAL")
    
    # ========== PONTO 4: Early exit OI 24h ==========
    print("\n[4] Early exit OI 24h")
    
    has_oi_exit = "OI_EARLY_EXIT" in source
    check(has_oi_exit, "Reason 'OI_EARLY_EXIT' implementado", severity="HIGH")
    
    # Verificar thresholds
    has_12h_check = "12" in source and ("hours_held" in source or "hours_since_entry" in source)
    check(has_12h_check, "Check de 12h mínimo implementado", severity="MEDIUM")
    
    has_oi_24h_threshold = "oi_z_24h" in source and "2.0" in source
    check(has_oi_24h_threshold, "Threshold oi_z_24h > 2.0", severity="HIGH")
    
    # ========== PONTO 5: Cooldown 4h ==========
    print("\n[5] Cooldown 4h pós-exit")
    
    has_cooldown = "cooldown" in source.lower()
    check(has_cooldown, "Cooldown implementado", severity="MEDIUM")
    
    cooldown_is_4h = "4" in source and "cooldown" in source.lower()
    check(cooldown_is_4h, "Cooldown parece ser 4h", severity="LOW")
    
    # ========== PONTO 6: Logs separados ==========
    print("\n[6] Logs separados")
    
    has_sol_log = "sol_bot4.log" in source or "sol_bot4" in source
    check(has_sol_log, "Log file 'sol_bot4.log' configurado", severity="MEDIUM")
    
    # ========== PONTO 7: Timezones UTC ==========
    print("\n[7] Timezones UTC")
    
    to_datetime_calls = re.findall(r"pd\.to_datetime\([^)]*\)", source)
    utc_aware = all("utc=True" in call for call in to_datetime_calls) if to_datetime_calls else True
    check(utc_aware, f"{len(to_datetime_calls)} calls pd.to_datetime, todos com utc=True", severity="HIGH")
    
    datetime_now_calls = re.findall(r"datetime\.now\([^)]*\)", source)
    utc_now = all("timezone.utc" in call for call in datetime_now_calls) if datetime_now_calls else True
    check(utc_now, f"{len(datetime_now_calls)} calls datetime.now, todos com timezone.utc", severity="HIGH")
    
    # ========== PONTO 8: MultiAssetManager ==========
    print("\n[8] MultiAssetManager integração")
    
    mam_path = ROOT / "src" / "trading" / "multi_asset_manager.py"
    if mam_path.exists():
        mam_source = mam_path.read_text()
        
        has_sol_bucket = '"sol_bot4"' in mam_source or "'sol_bot4'" in mam_source
        check(has_sol_bucket, "Bucket 'sol_bot4' adicionado em MultiAssetManager", severity="HIGH")
        
        # Buckets originais devem estar preservados
        for bot in ["btc_bot1", "btc_bot2", "eth_bot3"]:
            has_bot = f'"{bot}"' in mam_source or f"'{bot}'" in mam_source
            check(has_bot, f"Bucket '{bot}' preservado", severity="CRITICAL")
    else:
        check(False, "multi_asset_manager.py não encontrado", severity="CRITICAL")
    
    # ========== RESUMO ==========
    print("\n" + "=" * 60)
    print("RESUMO")
    print("=" * 60)
    
    critical_issues = [i for i in issues if i["severity"] == "CRITICAL"]
    high_issues = [i for i in issues if i["severity"] == "HIGH"]
    medium_issues = [i for i in issues if i["severity"] == "MEDIUM"]
    
    print(f"Total checks: {len(issues)}")
    print(f"CRITICAL issues: {len(critical_issues)}")
    print(f"HIGH issues: {len(high_issues)}")
    print(f"MEDIUM issues: {len(medium_issues)}")
    
    if critical_issues:
        print("\n🚨 AÇÃO IMEDIATA — Issues críticas:")
        for issue in critical_issues:
            print(f"  {issue['status']} {issue['message']}")
    
    # Generate markdown report
    lines = ["# SOL Bot 4 — Auditoria de Conformidade", ""]
    lines.append(f"**Data:** {datetime.now(timezone.utc).isoformat()}")
    lines.append(f"**Total checks:** {len(issues)}")
    lines.append(f"**Critical:** {len(critical_issues)} | **High:** {len(high_issues)} | **Medium:** {len(medium_issues)}")
    lines.append("")
    lines.append("## Resultados")
    lines.append("")
    for issue in issues:
        lines.append(f"- {issue['status']} **[{issue['severity']}]** {issue['message']}")
    
    lines.append("")
    lines.append("## Próximos passos")
    lines.append("")
    if critical_issues:
        lines.append("### 🚨 CORRIGIR AGORA (antes do primeiro trade)")
        for issue in critical_issues:
            if "❌" in issue["status"]:
                lines.append(f"- {issue['message']}")
    
    if high_issues:
        lines.append("")
        lines.append("### ⚠️ Corrigir em 24h")
        for issue in high_issues:
            if "❌" in issue["status"]:
                lines.append(f"- {issue['message']}")
    
    AUDIT_REPORT.write_text("\n".join(lines))
    print(f"\n✅ Report: {AUDIT_REPORT}")


def extract_function(source, func_name):
    """Extract function body from source."""
    pattern = rf"def {func_name}\([^)]*\).*?(?=\ndef |\Z)"
    match = re.search(pattern, source, re.DOTALL)
    return match.group(0) if match else None


if __name__ == "__main__":
    main()
```

## Execução da auditoria

### Passo 1: Local (código fonte)

```bash
cd /Users/brown/Documents/MLGeral/btc_AI
conda activate btc_trading_v1

# Criar e rodar script de auditoria
python scripts/audit_sol_bot4.py

# Ler report
cat AUDIT_SOL_BOT4.md
```

### Passo 2: AWS (estado de produção)

```bash
ssh -i ~/.ssh/aihab-key-sp.pem ubuntu@54.232.162.161 << 'EOF'
cd ~/AIhab

echo "=== 1. Crontab SOL entries ==="
docker exec aihab-app crontab -l | grep -i sol

echo ""
echo "=== 2. Logs SOL ==="
ls -la logs/sol*.log 2>/dev/null || echo "Sem logs SOL"

echo ""
echo "=== 3. Portfolio states ==="
ls -la data/05_output/portfolio*.json

echo ""
echo "=== 4. Shadow logs ==="
ls -la data/08_shadow/

echo ""
echo "=== 5. Últimos cycles SOL (se houver) ==="
if [ -f logs/sol_bot4.log ]; then
    tail -20 logs/sol_bot4.log
fi

echo ""
echo "=== 6. SOL decision recente ==="
docker exec aihab-app python3 -c "
import sys
sys.path.insert(0, '/app')
try:
    from src.trading.sol_bot4 import run_hourly_cycle
    print('Import OK')
    # DRY RUN (não executar se vai abrir posição)
except Exception as e:
    print(f'IMPORT FAILED: {e}')
"

echo ""
echo "=== 7. BTC capital unchanged ==="
cat data/05_output/portfolio_state.json | python3 -c "
import json, sys
d = json.load(sys.stdin)
print(f'BTC total capital: \${d.get(\"total_capital_usd\", 0):,.2f}')
"
EOF
```

### Passo 3: Interpretar findings e gerar patches

Baseado no output de ambos os steps:

**Se CRITICAL issues (anti look-ahead, capital misturado):**
```
AÇÃO IMEDIATA:
1. Pausar bot (comentar cron SOL)
2. Gerar patch
3. Testar localmente
4. Re-deploy
```

**Se HIGH issues (shadow scoring, early exit, UTC):**
```
AÇÃO EM 24H:
1. Gerar patches
2. Deploy em próxima janela de tempo ocioso
3. Manter bot rodando com limitações conhecidas
```

**Se MEDIUM issues:**
```
ACEITÁVEL por 1 semana:
1. Documentar
2. Corrigir em próximo release
```

## Patches comuns

### Patch 1: Adicionar _prev em features preditoras

```python
# ANTES (BUGGY):
def check_entry_signal(row, config):
    if row["taker_z"] < 0.3:  # ❌ CURRENT value
        return False, "taker_weak"

# DEPOIS (CORRETO):
def check_entry_signal(row, config):
    if row["taker_z_prev"] < 0.3:  # ✅ PREVIOUS value
        return False, "taker_weak"
```

### Patch 2: Capital isolado

```python
# Garantir que SOL NUNCA usa state de BTC:
def load_state():
    state_path = ROOT / "data/05_output/portfolio_state_sol.json"  # SOL-specific
    if state_path.exists():
        with open(state_path) as f:
            return json.load(f)
    return {
        "has_position": False,
        "capital_usd": 10000.0,  # Hard-coded $10k
        "bot_name": "sol_bot4",
    }
```

### Patch 3: Shadow scoring call

```python
# Dentro de enter_position(), ADICIONAR:
def enter_position(state, price, time, features, config):
    # ... código existente ...
    
    # Shadow scoring (sempre logar)
    log_scoring_shadow(
        trade_id=state["trade_id"],
        entry_time=time,
        features=features,
    )
```

## Critérios de GO/NO-GO após auditoria

```
🟢 GO - manter rodando:
  ✓ Zero critical issues
  ✓ < 3 high issues com plano de correção

🟡 CORRIGIR-E-CONTINUAR:
  - 1-2 critical issues
  - Patches aplicados em < 4h

🔴 PARAR BOT:
  - 3+ critical issues
  - Look-ahead confirmado
  - Capital contamination
  - Performance degradado
```

## Comandos de emergência

### Pausar SOL Bot (se problema crítico)

```bash
ssh -i ~/.ssh/aihab-key-sp.pem ubuntu@54.232.162.161

# Opção A: comentar crontab
docker exec aihab-app sed -i 's|^.*sol_hourly_cycle.*|# &|' /etc/cron.d/crontab
docker exec aihab-app service cron restart

# Opção B: disable via config
docker exec aihab-app python3 -c "
import yaml
with open('/app/conf/parameters_sol.yml') as f:
    cfg = yaml.safe_load(f)
cfg['sol_bot']['enabled'] = False
with open('/app/conf/parameters_sol.yml', 'w') as f:
    yaml.dump(cfg, f)
print('SOL Bot disabled via config')
"
```

### Fechar posição aberta se crítico

```bash
# Se SOL tem posição aberta e é bug crítico:
docker exec aihab-app python3 << 'PYEOF'
import json
from pathlib import Path

state_path = Path("/app/data/05_output/portfolio_state_sol.json")
state = json.load(state_path.open())

if state.get("has_position"):
    print("POSIÇÃO ABERTA — fechando manualmente")
    state["has_position"] = False
    state["manual_close_reason"] = "emergency_audit"
    state_path.write_text(json.dumps(state, indent=2, default=str))
    print("Position closed manually")
else:
    print("Sem posição aberta")
PYEOF
```

## Output esperado da auditoria

```
AUDIT_SOL_BOT4.md com:
  ✓ 8 pontos auditados
  ✓ Status de cada (✅/❌)
  ✓ Severity (CRITICAL/HIGH/MEDIUM/LOW)
  ✓ Próximos passos priorizados

Se tudo OK:
  → Bot continua rodando em paper
  → Monitor 14 dias

Se patches necessários:
  → Aplicar e re-deploy
  → Re-rodar auditoria
  → Confirmar 0 critical
```

## Checklist rápido (fast audit)

Se não quer rodar script completo, verifica esses 5 pontos rapidamente:

```bash
# 1. Arquivo existe e roda?
ls -la /Users/brown/Documents/MLGeral/btc_AI/src/trading/sol_bot4.py

# 2. Parameters_sol.yml existe?
cat /Users/brown/Documents/MLGeral/btc_AI/conf/parameters_sol.yml

# 3. AWS rodando?
ssh -i ~/.ssh/aihab-key-sp.pem ubuntu@54.232.162.161 \
  "docker exec aihab-app crontab -l | grep sol"

# 4. Portfolio SOL separado?
ssh -i ~/.ssh/aihab-key-sp.pem ubuntu@54.232.162.161 \
  "ls /home/ubuntu/AIhab/data/05_output/portfolio_state_sol.json"

# 5. Primeiro cycle rodou sem crash?
ssh -i ~/.ssh/aihab-key-sp.pem ubuntu@54.232.162.161 \
  "tail -5 /home/ubuntu/AIhab/logs/sol_bot4.log 2>/dev/null || echo 'sem logs ainda'"
```

Se todos OK → bot rodando, agendar auditoria completa depois.
Se algum falha → investigar urgente.
