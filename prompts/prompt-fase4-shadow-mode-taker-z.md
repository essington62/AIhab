# Prompt FASE 4: Shadow Mode — Filtro Taker_Z (4h + 1h)

## Contexto

**FASE 3 validou:**
- Filtro `taker_z < -1.0` melhora Sharpe: 2.71 → 3.61 (4h) / 3.71 (1h)
- 4h CoinGlass e 1h Binance empatam (+0.10 Sharpe, não significativo)
- Decisão: usar 4h em produção, manter 1h como backup/futuro

**Estado atual:**
- `gate_zscores.parquet` tem AMBAS colunas: `taker_z` (4h) e `taker_z_1h` (1h Binance nativo)
- `paper_trader.py` NÃO consome taker_z ainda (zero matches no grep)
- Sistema rodando com Sharpe 2.71 real

**Objetivo FASE 4:** implementar shadow mode **passivo**:
- Registra o que o filtro FARIA (bloquearia ou não)
- NÃO bloqueia trades (zero impacto em comportamento)
- Loga AMBAS versões (4h e 1h) para análise comparativa out-of-sample
- Após 2-4 semanas, decide se ativa filtro ou não

## Princípios críticos

1. **ZERO alteração** no comportamento de entrada de trades
2. **ADIÇÃO pura** — módulo separado, sem modificar lógica existente
3. **Registro robusto** — AMBAS versões (4h e 1h)
4. **Rollback trivial** — desativar = comentar 2 linhas
5. **Out-of-sample** — coleta dados que histórico não tinha

## Estrutura da implementação

```
src/trading/
  shadow_filters.py           ← NOVO módulo
  paper_trader.py             ← 3 linhas adicionadas

scripts/
  analyze_shadow_log.py       ← NOVO — análise semanal

data/08_shadow/                ← NOVO — logs separados
  taker_z_shadow_log.jsonl
```

## Passo 1 — Criar módulo shadow_filters

### `src/trading/shadow_filters.py`

```python
"""
Shadow filters — avaliação de filtros sem alterar comportamento.

Este módulo avalia filtros em paralelo ao paper_trader, registrando o que
o filtro FARIA sem de fato bloquear trades. Permite validação out-of-sample
antes de ativar o filtro em produção.

Uso:
    from src.trading.shadow_filters import evaluate_taker_z_shadow
    
    # Dentro do paper_trader, antes de abrir trade:
    shadow_log = evaluate_taker_z_shadow(entry_time, trade_id)
    # NÃO bloqueia — apenas registra
"""
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger("shadow_filters")

# Paths
GATE_ZSCORES_PATH = Path("data/02_features/gate_zscores.parquet")
SHADOW_LOG_DIR = Path("data/08_shadow")
SHADOW_LOG_FILE = SHADOW_LOG_DIR / "taker_z_shadow_log.jsonl"

# Filter config
FILTER_THRESHOLD = -1.0
FILTER_VERSION = "v1"


def _load_gate_zscores() -> pd.DataFrame | None:
    """Carrega gate_zscores.parquet. Retorna None se não existir."""
    if not GATE_ZSCORES_PATH.exists():
        logger.warning(f"gate_zscores.parquet not found at {GATE_ZSCORES_PATH}")
        return None
    
    try:
        df = pd.read_parquet(GATE_ZSCORES_PATH)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        return df.sort_values("timestamp").reset_index(drop=True)
    except Exception as e:
        logger.error(f"Failed to load gate_zscores: {e}")
        return None


def _get_prev_value(df: pd.DataFrame, entry_time: pd.Timestamp, column: str) -> float | None:
    """Retorna último valor ANTES de entry_time (safe, anti look-ahead)."""
    if column not in df.columns:
        return None
    
    prev_rows = df[df["timestamp"] < entry_time]
    if prev_rows.empty:
        return None
    
    val = prev_rows.iloc[-1][column]
    return float(val) if pd.notna(val) else None


def evaluate_taker_z_shadow(
    entry_time: pd.Timestamp,
    trade_id: str | int | None = None,
    bot_origin: str | None = None,
) -> dict[str, Any]:
    """
    Shadow evaluation do filtro taker_z.
    
    Args:
        entry_time: momento da entrada do trade (UTC)
        trade_id: identificador do trade (para rastreamento)
        bot_origin: "bot_1", "bot_2", "bot_3", etc
    
    Returns:
        dict com log estruturado (já persistido em arquivo)
    """
    # Normaliza entry_time
    if not isinstance(entry_time, pd.Timestamp):
        entry_time = pd.Timestamp(entry_time)
    if entry_time.tz is None:
        entry_time = entry_time.tz_localize("UTC")
    
    # Carrega features
    df = _load_gate_zscores()
    
    log_entry: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "trade_id": str(trade_id) if trade_id is not None else None,
        "bot_origin": bot_origin,
        "entry_time": entry_time.isoformat(),
        "filter_version": FILTER_VERSION,
        "threshold": FILTER_THRESHOLD,
        "status": "unknown",
    }
    
    if df is None:
        log_entry["status"] = "error_no_gate_zscores"
        _persist_log(log_entry)
        return log_entry
    
    # 4h CoinGlass (prev)
    taker_z_4h = _get_prev_value(df, entry_time, "taker_z")
    
    # 1h Binance (prev)
    taker_z_1h = _get_prev_value(df, entry_time, "taker_z_1h")
    
    # Timestamp do candle usado (para auditoria)
    prev_rows = df[df["timestamp"] < entry_time]
    prev_candle_time = (
        prev_rows.iloc[-1]["timestamp"].isoformat()
        if not prev_rows.empty else None
    )
    
    # Evaluation
    would_block_4h = (taker_z_4h is not None) and (taker_z_4h < FILTER_THRESHOLD)
    would_block_1h = (taker_z_1h is not None) and (taker_z_1h < FILTER_THRESHOLD)
    
    log_entry.update({
        "status": "ok",
        "prev_candle_time": prev_candle_time,
        "taker_z_4h": taker_z_4h,
        "taker_z_1h": taker_z_1h,
        "would_block_4h": would_block_4h,
        "would_block_1h": would_block_1h,
        "both_agree_block": would_block_4h and would_block_1h,
        "disagreement": would_block_4h != would_block_1h,
    })
    
    _persist_log(log_entry)
    return log_entry


def _persist_log(log_entry: dict[str, Any]) -> None:
    """Append log no JSONL file."""
    try:
        SHADOW_LOG_DIR.mkdir(parents=True, exist_ok=True)
        with open(SHADOW_LOG_FILE, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
    except Exception as e:
        logger.error(f"Failed to persist shadow log: {e}")
```

## Passo 2 — Integrar no `paper_trader.py`

**Localizar onde o trade é aberto** (provavelmente função `open_position` ou similar):

```bash
grep -n "open_position\|enter_trade\|entry_price\|trade_id" src/trading/paper_trader.py | head -20
```

**Encontrar o ponto certo:** logo antes de abrir a posição, após todas as validações normais.

**Adicionar (3 linhas + import):**

```python
# No topo do paper_trader.py, junto aos outros imports:
from src.trading.shadow_filters import evaluate_taker_z_shadow

# NO LOCAL onde o trade é aberto (exemplo):
def open_position(self, entry_price, entry_time, ...):
    # ... código existente ...
    
    # ============================================================
    # SHADOW MODE: avalia filtro sem bloquear (FASE 4)
    # ============================================================
    try:
        shadow_result = evaluate_taker_z_shadow(
            entry_time=entry_time,
            trade_id=trade_id,
            bot_origin=bot_origin,
        )
        logger.info(
            f"Shadow filter taker_z: 4h={shadow_result.get('taker_z_4h')} "
            f"(would_block={shadow_result.get('would_block_4h')}), "
            f"1h={shadow_result.get('taker_z_1h')} "
            f"(would_block={shadow_result.get('would_block_1h')})"
        )
    except Exception as e:
        logger.warning(f"Shadow filter evaluation failed: {e}")
        # Shadow mode NUNCA deve quebrar paper_trader
    
    # ... resto do código existente ...
```

**CRÍTICO:**
- Shadow mode NUNCA pode bloquear ou modificar fluxo de trade
- try/except garante que se falhar, paper_trader continua normal
- Logger.info mostra que está funcionando, não spammar

## Passo 3 — Script de análise

### `scripts/analyze_shadow_log.py`

```python
"""
Análise do shadow log — avaliar filtros out-of-sample.

Rodar semanalmente para verificar:
  1. Quantos trades ocorreram?
  2. Quantos o filtro TERIA bloqueado (4h e 1h)?
  3. Desses bloqueios virtuais, quantos foram losers?
  4. Ratio se mantém como histórico (1.43)?

Outputs:
  prompts/shadow_analysis_report.md
"""
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("shadow_analysis")

ROOT = Path(__file__).resolve().parents[1]
SHADOW_LOG = ROOT / "data/08_shadow/taker_z_shadow_log.jsonl"
TRADES_PATH = ROOT / "data/05_trades/completed_trades.json"  # ajuste se diferente
REPORT_PATH = ROOT / "prompts/shadow_analysis_report.md"


def load_shadow_log() -> pd.DataFrame:
    """Carrega log JSONL em DataFrame."""
    if not SHADOW_LOG.exists():
        raise FileNotFoundError(f"Shadow log not found: {SHADOW_LOG}")
    
    records = []
    with open(SHADOW_LOG) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    
    df = pd.DataFrame(records)
    df["entry_time"] = pd.to_datetime(df["entry_time"], utc=True)
    return df


def load_trades() -> pd.DataFrame:
    """Carrega completed trades."""
    if not TRADES_PATH.exists():
        # Tenta paths alternativos
        for alt in [
            ROOT / "data/05_output/completed_trades.json",
            ROOT / "data/04_scoring/completed_trades.json",
        ]:
            if alt.exists():
                with open(alt) as f:
                    data = json.load(f)
                return pd.DataFrame(data)
        raise FileNotFoundError("completed_trades.json not found")
    
    with open(TRADES_PATH) as f:
        data = json.load(f)
    return pd.DataFrame(data)


def main():
    logger.info("=" * 60)
    logger.info("Shadow Filter Analysis")
    logger.info("=" * 60)
    
    # Load data
    try:
        df_shadow = load_shadow_log()
    except FileNotFoundError:
        logger.error("Shadow log vazio — rode paper_trader por alguns dias primeiro")
        return
    
    logger.info(f"Shadow log: {len(df_shadow)} entries")
    logger.info(f"Period: {df_shadow['entry_time'].min()} → {df_shadow['entry_time'].max()}")
    
    df_trades = load_trades()
    logger.info(f"Completed trades: {len(df_trades)}")
    
    # Merge shadow log com trades (por trade_id)
    if "trade_id" in df_trades.columns and "trade_id" in df_shadow.columns:
        df = df_trades.merge(
            df_shadow[["trade_id", "taker_z_4h", "taker_z_1h",
                       "would_block_4h", "would_block_1h"]],
            on="trade_id",
            how="left",
        )
    else:
        logger.warning("Cannot match by trade_id — using time-based join")
        # Fallback: match por entry_time aproximado
        df = df_trades.copy()
    
    # Stats básicas
    report_lines = []
    report_lines.append("# 📊 Shadow Filter Analysis Report")
    report_lines.append(f"\n**Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    report_lines.append(f"**Shadow log period:** {df_shadow['entry_time'].min()} → {df_shadow['entry_time'].max()}")
    report_lines.append(f"**Total trades observed:** {len(df_shadow)}")
    report_lines.append(f"**Completed trades (with outcome):** {len(df[df.get('return_pct', 0).notna()])}")
    report_lines.append("")
    
    # 4h analysis
    report_lines.append("## 4h CoinGlass Filter (taker_z < -1.0)\n")
    n_block_4h = df_shadow["would_block_4h"].sum()
    report_lines.append(f"- Would block: {n_block_4h} / {len(df_shadow)} ({n_block_4h/len(df_shadow)*100:.1f}%)")
    
    if "return_pct" in df.columns:
        blocked_4h = df[df["would_block_4h"] == True]
        if len(blocked_4h) > 0:
            losers_blocked = (blocked_4h["return_pct"] < 0).sum()
            winners_blocked = (blocked_4h["return_pct"] > 0).sum()
            ratio_4h = losers_blocked / max(winners_blocked, 1)
            report_lines.append(f"- Of blocked: {losers_blocked} losers + {winners_blocked} winners")
            report_lines.append(f"- Ratio: {ratio_4h:.2f} losers/winner (histórico: 1.43)")
            
            if ratio_4h >= 1.3:
                report_lines.append(f"- ✅ Consistente com histórico")
            else:
                report_lines.append(f"- ⚠️ Degradado vs histórico")
    report_lines.append("")
    
    # 1h analysis
    report_lines.append("## 1h Binance Filter (taker_z_1h < -1.0)\n")
    n_block_1h = df_shadow["would_block_1h"].sum()
    report_lines.append(f"- Would block: {n_block_1h} / {len(df_shadow)} ({n_block_1h/len(df_shadow)*100:.1f}%)")
    
    if "return_pct" in df.columns:
        blocked_1h = df[df["would_block_1h"] == True]
        if len(blocked_1h) > 0:
            losers_blocked = (blocked_1h["return_pct"] < 0).sum()
            winners_blocked = (blocked_1h["return_pct"] > 0).sum()
            ratio_1h = losers_blocked / max(winners_blocked, 1)
            report_lines.append(f"- Of blocked: {losers_blocked} losers + {winners_blocked} winners")
            report_lines.append(f"- Ratio: {ratio_1h:.2f} losers/winner")
    report_lines.append("")
    
    # Agreement analysis
    report_lines.append("## Agreement 4h vs 1h\n")
    agree = (df_shadow["would_block_4h"] == df_shadow["would_block_1h"]).sum()
    disagree = len(df_shadow) - agree
    report_lines.append(f"- Concordaram: {agree} ({agree/len(df_shadow)*100:.1f}%)")
    report_lines.append(f"- Divergiram: {disagree} ({disagree/len(df_shadow)*100:.1f}%)")
    report_lines.append("")
    
    # Decision matrix
    report_lines.append("## Decisão: ativar filtro?\n")
    if len(df_shadow) < 50:
        report_lines.append("⏳ **ESPERAR** — amostra muito pequena (<50 trades)")
    elif len(df_shadow) < 100:
        report_lines.append("⏳ **CONTINUAR SHADOW** — 50-100 trades, tendência emergindo")
    else:
        if "return_pct" in df.columns:
            blocked_4h = df[df["would_block_4h"] == True]
            if len(blocked_4h) > 10:
                losers = (blocked_4h["return_pct"] < 0).sum()
                winners = (blocked_4h["return_pct"] > 0).sum()
                ratio = losers / max(winners, 1)
                if ratio >= 1.3:
                    report_lines.append(f"✅ **PODE ATIVAR** — ratio {ratio:.2f} confirmado out-of-sample")
                else:
                    report_lines.append(f"❌ **NÃO ATIVAR** — ratio {ratio:.2f} degradado (histórico 1.43)")
    
    report_lines.append("")
    
    # Save
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w") as f:
        f.write("\n".join(report_lines))
    
    logger.info(f"Report saved: {REPORT_PATH}")
    print("\n" + "\n".join(report_lines))


if __name__ == "__main__":
    main()
```

## Passo 4 — Validação local

### Teste 1 — Módulo carrega sem erro

```bash
cd /Users/brown/Documents/MLGeral/btc_AI
conda activate btc_trading_v1

python -c "
from src.trading.shadow_filters import evaluate_taker_z_shadow
import pandas as pd

# Teste com entry_time agora
now = pd.Timestamp.utcnow()
result = evaluate_taker_z_shadow(entry_time=now, trade_id='test_001', bot_origin='test')
print(result)
"
```

**Esperado:** log estruturado aparece + entrada no `data/08_shadow/taker_z_shadow_log.jsonl`

### Teste 2 — paper_trader não quebrou

```bash
# Teste dry-run do paper_trader
python -m src.trading.paper_trader
```

**Se não abrir trade:** OK (não tinha sinal)
**Se abrir trade:** verificar que log de shadow aparece

### Teste 3 — Log em JSONL é válido

```bash
python -c "
import json
from pathlib import Path
log_file = Path('data/08_shadow/taker_z_shadow_log.jsonl')
if log_file.exists():
    with open(log_file) as f:
        for line in f:
            record = json.loads(line.strip())
            print(record)
"
```

## Passo 5 — Deploy na AWS

**Arquivos a sincronizar:**

```bash
# No Mac:
cd /Users/brown/Documents/MLGeral/btc_AI

# Git commit + push
git add src/trading/shadow_filters.py
git add scripts/analyze_shadow_log.py
git add src/trading/paper_trader.py  # com as 3 linhas novas
git commit -m "FASE 4: shadow mode filtro taker_z (4h + 1h)"
git push origin main

# Na AWS EC2:
ssh -i ~/.ssh/aihab-key-sp.pem ubuntu@54.232.162.161
cd ~/btc_AI
git pull
docker-compose restart aihab-app
```

**Validar logs na AWS:**

```bash
# Após próximo hourly cycle:
docker logs aihab-app --tail 100 | grep "Shadow filter"
```

## Passo 6 — Documentação

Criar `prompts/fase4_changelog.md`:

```markdown
# FASE 4 Changelog — Shadow Mode Taker_Z Filter

**Data:** 2026-04-21
**Objetivo:** Validar filtro taker_z out-of-sample antes de ativar

## Novos arquivos

- `src/trading/shadow_filters.py` — módulo de avaliação shadow
- `scripts/analyze_shadow_log.py` — análise semanal

## Modificações

- `src/trading/paper_trader.py`:
  - Adicionado import de shadow_filters
  - Adicionado try/except block que chama evaluate_taker_z_shadow antes de abrir posição
  - 3 linhas efetivas + import

## Logs

- `data/08_shadow/taker_z_shadow_log.jsonl` (cresce append-only)

## Como usar

### Monitorar (semanalmente)

```bash
python scripts/analyze_shadow_log.py
cat prompts/shadow_analysis_report.md
```

### Rollback (desativar)

Comentar o bloco `try/except` do shadow_filter em paper_trader.py.
Módulo `shadow_filters.py` pode ficar (zero impacto se não chamado).

## Critérios para ativação

Após 2-4 semanas (50+ trades novos):

- ✅ Ativar 4h: ratio losers/winners entre bloqueados ≥ 1.3
- ⚠️ Continuar shadow: ratio 1.0-1.3
- ❌ Rejeitar: ratio < 1.0 (filtro não reproduz histórico)

## Próximos passos

- Analisar report semanalmente
- Após 2-4 semanas, decidir ativação
- Se ativar: mudar shadow_filters para bloqueio real (flag `ACTIVE_BLOCK=True`)
```

## Checklist completo

```
[ ] Criar src/trading/shadow_filters.py
[ ] Adicionar 3 linhas em paper_trader.py + import
[ ] Criar scripts/analyze_shadow_log.py
[ ] Teste 1: módulo carrega sem erro
[ ] Teste 2: paper_trader roda (dry-run se possível)
[ ] Teste 3: log JSONL é válido
[ ] Git commit + push
[ ] Deploy AWS (pull + restart)
[ ] Validar logs AWS após próximo ciclo
[ ] Criar fase4_changelog.md
```

## Critérios de sucesso

```
✅ shadow_filters.py importável sem erro
✅ paper_trader.py executa normalmente (sem mudança de comportamento)
✅ Log JSONL acumula entries a cada trade
✅ AMBAS versões (4h e 1h) aparecem em cada entry
✅ Análise script roda sem crash
✅ Rollback possível em 2 minutos (comentar try/except)
```

## Rollback se algo der errado

```bash
# 1. Comentar shadow filter em paper_trader.py (3 linhas)
# 2. Restart container AWS
docker-compose restart aihab-app

# 3. Módulo shadow_filters.py pode ficar (não é chamado)
# 4. Logs existentes podem ficar (útil mesmo assim)

# OU rollback completo via git:
git revert HEAD
git push origin main
# Depois AWS: git pull && docker-compose restart
```

## Cronograma sugerido

```
Dia 0 (hoje):       implementação + deploy AWS
Dia 7:              primeira análise (10-30 trades shadow)
Dia 14:             segunda análise (20-60 trades)
Dia 21:             decisão preliminar
Dia 28 (4 semanas): decisão final sobre ativação
```

## Princípios críticos

1. **Shadow mode é PASSIVO** — zero mudança de comportamento
2. **try/except sempre** — se shadow falhar, paper_trader continua
3. **AMBAS versões logadas** — maximize insight para decisão
4. **Análise semanal** — não esperar 4 semanas cegamente
5. **Critério claro de ativação** — ratio ≥ 1.3 out-of-sample
