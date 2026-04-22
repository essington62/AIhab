# Prompt: Adicionar Filtro de News ao Bot 2 (Momentum/Liquidez)

## Contexto

O Bot 2 (Momentum/Liquidez) está LIVE e funciona bem (WR 80%, PF 2.07, 5 trades).
Porém ele NÃO considera notícias antes de entrar — compra momentum mesmo com news bearish.

O Bot 1 já tem news via gate scoring (G2). O `news_crypto_score` já é computado
em `run_cycle()` (linha 973) via `load_news_crypto_score()`. Basta passar esse
valor para o `check_momentum_filter()`.

Bot 2 compra MOMENTUM — news bearish contradiz a tese. Deve ser mais conservador
que o kill switch do Bot 1 (que é -3.0). Threshold sugerido: **-1.0**.

## Alterações necessárias (3 arquivos)

---

### 1. `conf/parameters.yml` — Adicionar `news_score_min` ao momentum_filter

Localizar a seção `momentum_filter:` (linha ~206) e adicionar ANTES do bloco `spike_guard:`:

```yaml
  # News filter: bloqueia entrada se notícias muito bearish
  # Bot 2 compra momentum — news bearish contradiz a tese
  # Usa o mesmo news_crypto_score do gate scoring (G2)
  news_score_min: -1.0       # news_crypto_score >= -1.0 (mais conservador que Bot 1 kill switch de -3.0)
```

O bloco deve ficar assim após a alteração:

```yaml
momentum_filter:
  enabled: true
  stablecoin_z_min: 1.3
  ret_1d_min: 0.0
  rsi_min: 50
  bb_pct_max: 0.98
  require_above_ma21: true

  # Cooldown após stop loss
  cooldown:
    enabled: true
    hours_after_sl: 12
    require_price_above_exit: true
    bounce_pct: 0.003
    max_consecutive_sl: 3
    consecutive_sl_pause_hours: 24

  # News filter: bloqueia entrada se notícias muito bearish
  news_score_min: -1.0

  # Anti-spike guard
  spike_guard:
    enabled: true
    spike_ret_max: 0.03
    spike_rsi_max: 65

  # Stops específicos do Bot 2
  stop_loss_pct: 0.015
  take_profit_pct: 0.02
  trailing_stop_pct: 0.01
  max_hold_hours: 120
```

---

### 2. `src/trading/paper_trader.py` — Modificar `check_momentum_filter()`

#### 2a. Alterar a assinatura para receber `news_crypto_score`

De:
```python
def check_momentum_filter(technical: dict, zscores: dict, params: dict) -> dict:
```

Para:
```python
def check_momentum_filter(technical: dict, zscores: dict, params: dict, news_crypto_score: float = 0.0) -> dict:
```

#### 2b. Atualizar o docstring — adicionar condição 6

Adicionar na lista de condições:
```
      6. news_crypto_score >= -1.0 (no strong bearish news)
```

#### 2c. Ler o threshold do YAML

Após a linha `require_ma21 = mf.get("require_above_ma21", True)`, adicionar:

```python
    news_min = mf.get("news_score_min", -1.0)
```

#### 2d. Adicionar `news_crypto_score` e `news_min` ao result dict

No dict `result = { ... }`, adicionar:

```python
        "news_crypto_score": news_crypto_score,
        "news_min": news_min,
```

#### 2e. Adicionar o check de news — APÓS os checks existentes (antes do spike_guard)

Antes do bloco `spike_cfg = mf.get("spike_guard", {})`, adicionar:

```python
    if news_crypto_score < news_min:
        reasons.append(f"BEARISH_NEWS (news={news_crypto_score:.2f} < {news_min})")
```

Nota: `news_crypto_score` é um float (default 0.0 se não disponível), então
não precisa de null check — 0.0 é neutro e passa.

---

### 3. `src/trading/paper_trader.py` — Atualizar chamadas em `run_cycle()`

Em `run_cycle()`, o `news_crypto_score` já é computado na linha ~973:
```python
news_crypto_score = load_news_crypto_score(lookback_h)
```

Localizar as duas chamadas a `check_momentum_filter()` (linhas ~1179 e ~1181):

```python
mf_check = check_momentum_filter(technical, zscores, params)
```

Substituir AMBAS por:

```python
mf_check = check_momentum_filter(technical, zscores, params, news_crypto_score)
```

---

### 4. `src/dashboard/app.py` — Mostrar news no painel Bot 2 (OPCIONAL)

Se existir um expander/painel de Bot 2 no dashboard, adicionar uma métrica
mostrando o news_crypto_score atual e se está acima do threshold.
Isso é opcional — se não encontrar o painel de Bot 2 no dashboard, pular.

---

### 5. Testes — `tests/test_momentum_filter.py`

Adicionar estes testes ao arquivo de testes existente do momentum filter:

```python
class TestMomentumFilterNews:
    """News filter tests for Bot 2."""

    def test_bearish_news_blocks_entry(self):
        """news_crypto_score < -1.0 should block."""
        r = check_momentum_filter(_tech(), _zscores(), PARAMS_ENABLED, news_crypto_score=-1.5)
        assert r["passed"] is False
        assert "BEARISH_NEWS" in r["reason"]

    def test_neutral_news_passes(self):
        """news_crypto_score = 0.0 should pass."""
        r = check_momentum_filter(_tech(), _zscores(), PARAMS_ENABLED, news_crypto_score=0.0)
        assert r["passed"] is True

    def test_positive_news_passes(self):
        """news_crypto_score = 1.5 should pass."""
        r = check_momentum_filter(_tech(), _zscores(), PARAMS_ENABLED, news_crypto_score=1.5)
        assert r["passed"] is True

    def test_news_exactly_at_threshold_blocks(self):
        """news_crypto_score == -1.0 should block (need > -1.0, not >=)."""
        # Nota: o check é news_crypto_score < news_min, então -1.0 NÃO bloqueia (não é < -1.0)
        r = check_momentum_filter(_tech(), _zscores(), PARAMS_ENABLED, news_crypto_score=-1.0)
        assert r["passed"] is True  # -1.0 is NOT < -1.0

    def test_slightly_below_threshold_blocks(self):
        """news_crypto_score = -1.01 should block."""
        r = check_momentum_filter(_tech(), _zscores(), PARAMS_ENABLED, news_crypto_score=-1.01)
        assert r["passed"] is False
        assert "BEARISH_NEWS" in r["reason"]

    def test_news_combined_with_other_failures(self):
        """Multiple failures including news should all appear in reason."""
        r = check_momentum_filter(
            _tech(rsi=40), _zscores(stable_z=0.5), PARAMS_ENABLED, news_crypto_score=-2.0
        )
        assert r["passed"] is False
        assert "BEARISH_NEWS" in r["reason"]
        assert "LOW_LIQUIDITY" in r["reason"]
        assert "RSI_LOW" in r["reason"]

    def test_default_news_score_is_neutral(self):
        """Without news_crypto_score arg, default 0.0 should pass."""
        r = check_momentum_filter(_tech(), _zscores(), PARAMS_ENABLED)
        assert r["passed"] is True
```

Use os helpers `_tech()`, `_zscores()` e `PARAMS_ENABLED` que já existem no
arquivo de testes. Se `PARAMS_ENABLED` não tiver `news_score_min`, adicione:

```python
PARAMS_ENABLED = {
    "momentum_filter": {
        "enabled": True,
        "stablecoin_z_min": 1.3,
        "ret_1d_min": 0.0,
        "rsi_min": 50,
        "bb_pct_max": 0.98,
        "require_above_ma21": True,
        "news_score_min": -1.0,        # ← ADICIONAR
    }
}
```

---

### 6. Log — Adicionar news ao log de entry do Bot 2

Na mensagem de log `BOT2 ENTRY CONFIRMED:` (em run_cycle), adicionar o news_crypto_score:

De:
```python
f"BOT2 ENTRY CONFIRMED: stablecoin_z={mf_check['stablecoin_z']:.2f} | "
f"ret_1d={mf_check['ret_1d']:.4f} | RSI={mf_check['rsi']:.1f} | "
f"BB={mf_check['bb_pct']:.3f}"
```

Para:
```python
f"BOT2 ENTRY CONFIRMED: stablecoin_z={mf_check['stablecoin_z']:.2f} | "
f"ret_1d={mf_check['ret_1d']:.4f} | RSI={mf_check['rsi']:.1f} | "
f"BB={mf_check['bb_pct']:.3f} | news={news_crypto_score:.2f}"
```

---

## Resumo das mudanças

| Arquivo | O que muda |
|---------|-----------|
| `conf/parameters.yml` | +1 parâmetro: `news_score_min: -1.0` |
| `src/trading/paper_trader.py` | Assinatura + 1 check + 2 call sites |
| `tests/test_momentum_filter.py` | +7 testes |
| `src/dashboard/app.py` | Opcional: métrica news no painel Bot 2 |

## Notas

- NÃO alterar `check_reversal_filter()` — Bot 1 fica intocado
- NÃO alterar `gate_scoring.py` — o news score é lido, não computado
- NÃO alterar `load_news_crypto_score()` — a função já existe e funciona
- O default `news_crypto_score=0.0` na assinatura garante backward compatibility
- Threshold -1.0 é mais conservador que o kill switch do Bot 1 (-3.0) porque
  Bot 2 compra momentum — faz sentido ser mais sensível a news negativas
- Rodar `pytest` ao final — todos os testes existentes + novos devem passar
