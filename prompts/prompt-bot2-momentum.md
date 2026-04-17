# Prompt: Implementar Bot 2 — Momentum/Liquidez no AI.hab

## Contexto

O AI.hab já tem o **Bot 1 (Reversal Filter)** que compra fundos: score > threshold + RSI < 35 + ret_1d > -1%.

O **Bot 2 (Momentum/Liquidez)** é complementar — surfa ondas de liquidez quando stablecoins estão entrando no mercado.

**Estudo estatístico validado** (estudo_bot2_momentum.py):
- PF 1.39–1.54, WR 50–52%, Total +27–32%, MaxDD -10 a -15%
- **Complementaridade 100%** com Bot 1 (zero overlap nos sinais)
- Combined Bot1+Bot2: PF 1.70, +57.0%, WR 52% vs Bot1 sozinho PF 1.32, +30.8%
- Edge crescente: 2023 PF 1.15 → 2024 PF 1.21 → 2025 PF 1.71 → 2026 PF 2.58
- Stops ótimos: SG 2% / SL 1.5% / max 5d (120h) / trailing 1%

## Regra de Entrada (Bot 2)

```python
# Todas as condições devem ser TRUE simultaneamente:
stablecoin_z > 1.3      # Liquidez forte entrando via stablecoins
ret_1d > 0               # Mercado em alta (momentum positivo)
rsi_14 > 50              # RSI em zona bullish (não oversold)
close > ma_21            # Preço acima da MA21 (tendência curta de alta)
bb_pct < 0.98            # Filtro anti blow-off top
```

## Regra de Mutex Bot1/Bot2

```python
# Bot 2 NUNCA entra se Bot 1 tem posição aberta (e vice-versa)
# Ambos compartilham o mesmo portfolio_state.json
# A verificação é feita via portfolio["has_position"]
# Identificador do bot que abriu: portfolio["entry_bot"] = "bot1" ou "bot2"
```

## Arquitetura de Implementação

O Bot 2 NÃO é um segundo paper_trader. Ele é integrado no `run_cycle()` existente como um **caminho alternativo de entrada**:

```
Gate Scoring → ENTER?
  ├─ YES → check_reversal_filter() (Bot 1)
  │    ├─ passed → ENTRY (bot1)
  │    └─ filtered → fallthrough (don't enter)
  │
  └─ NO (HOLD/BLOCK) → check_momentum_filter() (Bot 2) ← NOVO
       ├─ passed → ENTRY (bot2, com stops próprios)
       └─ filtered → HOLD (nenhum bot entra)
```

**IMPORTANTE**: Bot 2 roda INDEPENDENTE do gate scoring. Ele NÃO precisa que o score > threshold. O gate scoring pode dizer HOLD ou BLOCK (exceto Bear regime), e o Bot 2 ainda assim pode entrar se as condições de liquidez forem atendidas.

Revisão do fluxo:

```
1. Regime check → Bear? → BLOCK total (ambos bots bloqueados)
2. Technical + zscores carregados normalmente
3. Gate Scoring roda → result["signal"]
4. Se has_position → check_stops (igual hoje, ambos bots)
5. Se NOT has_position:
   a. Se signal == ENTER → check_reversal_filter (Bot 1)
      - passed → entry bot1
      - filtered → continua para (b)
   b. Se regime != Bear → check_momentum_filter (Bot 2)
      - passed → entry bot2 (stops diferentes)
      - filtered → HOLD
```

---

## PARTE 1: parameters.yml — Adicionar seção momentum_filter

Adicionar APÓS a seção `reversal_filter:`:

```yaml
# ---------------------------------------------------------------------------
# Momentum filter (Bot 2 — entry confirmation, independent of gate scoring)
# ---------------------------------------------------------------------------
momentum_filter:
  enabled: true
  stablecoin_z_min: 1.3     # Stablecoin z-score mínimo (liquidez entrando)
  ret_1d_min: 0.0            # Retorno 1d mínimo (momentum positivo, > 0%)
  rsi_min: 50                # RSI mínimo (zona bullish)
  bb_pct_max: 0.98           # BB% máximo (filtro anti blow-off top)
  require_above_ma21: true   # Preço deve estar acima da MA21

  # Stops específicos do Bot 2 (mais apertados que Bot 1)
  stop_loss_pct: 0.015       # SL 1.5%
  take_profit_pct: 0.02      # SG 2%
  trailing_stop_pct: 0.01    # Trailing 1%
  max_hold_hours: 120        # Max 5 dias (120 horas)
```

---

## PARTE 2: check_momentum_filter() em paper_trader.py

Adicionar a função `check_momentum_filter()` APÓS `check_reversal_filter()` no paper_trader.py:

```python
def check_momentum_filter(technical: dict, zscores: dict, params: dict) -> dict:
    """
    Bot 2 — Momentum/Liquidez filter.
    Entry when stablecoin liquidity is flowing in + market in uptrend.
    Independent of gate scoring — runs as alternative entry path.

    Conditions (ALL must be true):
      1. stablecoin_z > 1.3 (strong liquidity inflow)
      2. ret_1d > 0 (positive momentum)
      3. RSI > 50 (bullish zone)
      4. close > MA21 (short-term uptrend)
      5. BB% < 0.98 (not a blow-off top)

    Returns:
        dict with keys: passed (bool), reason (str), stablecoin_z, ret_1d,
                        rsi, bb_pct, above_ma21
    """
    mf = params.get("momentum_filter", {})

    if not mf.get("enabled", False):
        return {"passed": False, "reason": "filter_disabled"}

    # Extract values
    stablecoin_z = zscores.get("stablecoin_z")
    ret_1d = technical.get("ret_1d")
    rsi = technical.get("rsi_14")
    bb_pct = technical.get("bb_pct")
    close = technical.get("close")
    ma_21 = technical.get("ma_21")

    # Thresholds from params
    sz_min = mf.get("stablecoin_z_min", 1.3)
    ret_min = mf.get("ret_1d_min", 0.0)
    rsi_min = mf.get("rsi_min", 50)
    bb_max = mf.get("bb_pct_max", 0.98)
    require_ma21 = mf.get("require_above_ma21", True)

    # Build result dict (always populated for logging)
    result = {
        "stablecoin_z": stablecoin_z,
        "ret_1d": ret_1d,
        "rsi": rsi,
        "bb_pct": bb_pct,
        "close": close,
        "ma_21": ma_21,
        "sz_min": sz_min,
        "ret_min": ret_min,
        "rsi_min": rsi_min,
        "bb_max": bb_max,
    }

    # Null checks — any missing data → block
    if stablecoin_z is None or ret_1d is None or rsi is None or bb_pct is None:
        result["passed"] = False
        result["reason"] = "MISSING_DATA"
        return result

    if require_ma21 and (close is None or ma_21 is None):
        result["passed"] = False
        result["reason"] = "MISSING_MA21"
        return result

    # Check conditions — aggregate ALL failure reasons
    reasons = []

    if stablecoin_z <= sz_min:
        reasons.append(f"LOW_LIQUIDITY (stable_z={stablecoin_z:.2f} <= {sz_min})")

    if ret_1d <= ret_min:
        reasons.append(f"NEG_MOMENTUM (ret_1d={ret_1d:.4f} <= {ret_min})")

    if rsi <= rsi_min:
        reasons.append(f"RSI_LOW (RSI={rsi:.1f} <= {rsi_min})")

    if bb_pct >= bb_max:
        reasons.append(f"BLOW_OFF_TOP (BB={bb_pct:.3f} >= {bb_max})")

    if require_ma21 and close <= ma_21:
        reasons.append(f"BELOW_MA21 (close={close:.0f} <= MA21={ma_21:.0f})")

    if reasons:
        result["passed"] = False
        result["reason"] = " & ".join(reasons)
        return result

    result["passed"] = True
    result["reason"] = "momentum_confirmed"
    return result
```

---

## PARTE 3: Modificar run_cycle() em paper_trader.py

### 3a. Após o bloco do Bot 1 (Entry decision), adicionar Bot 2

Localizar o bloco (linhas ~686-711):
```python
        # Entry decision (scoring says ENTER → apply reversal filter)
        if result["signal"] == "ENTER" and not portfolio["has_position"]:
            rf_check = check_reversal_filter(technical, params)
            ...
```

Substituir por:

```python
        # Entry decision
        bot_entered = None

        # Bot 1: Reversal — scoring says ENTER → apply reversal filter
        if result["signal"] == "ENTER" and not portfolio["has_position"]:
            rf_check = check_reversal_filter(technical, params)

            if rf_check["passed"]:
                atr_14 = technical.get("atr_14")
                portfolio = execute_entry(current_price, portfolio, atr_14=atr_14)
                _init_trade_tracking(portfolio, result, regime, technical, zscores)
                portfolio["entry_ret_1d"] = rf_check["ret_1d"]
                portfolio["entry_filter_passed"] = True
                portfolio["entry_bot"] = "bot1"
                atomic_write_json(portfolio, get_path("portfolio_state"))
                bot_entered = "bot1"
                logger.info(
                    f"BOT1 ENTRY CONFIRMED: score={result.get('score', 0):.3f} | "
                    f"RSI={rf_check['rsi']:.1f} (<{rf_check['rsi_max']}) | "
                    f"ret_1d={rf_check['ret_1d']:.4f} (>{rf_check['ret_1d_min']})"
                )
            else:
                logger.info(
                    f"BOT1 ENTRY FILTERED: score={result.get('score', 0):.3f} | "
                    f"RSI={rf_check['rsi']} | ret_1d={rf_check['ret_1d']} → "
                    f"{rf_check['reason']}"
                )
                result["signal"] = "FILTERED"
                result["filter_reason"] = rf_check["reason"]
                result["filter_rsi"] = rf_check["rsi"]
                result["filter_ret_1d"] = rf_check["ret_1d"]

        # Bot 2: Momentum — independent of gate scoring, runs if no position
        if bot_entered is None and not portfolio["has_position"] and regime != "Bear":
            mf_check = check_momentum_filter(technical, zscores, params)

            if mf_check["passed"]:
                # Bot 2 uses its OWN stop configuration
                mf_params = params.get("momentum_filter", {})
                portfolio = _execute_bot2_entry(current_price, portfolio, mf_params)
                _init_trade_tracking(portfolio, result, regime, technical, zscores)
                portfolio["entry_bot"] = "bot2"
                portfolio["entry_stablecoin_z"] = mf_check["stablecoin_z"]
                portfolio["entry_filter_passed"] = True
                portfolio["entry_max_hold_hours"] = mf_params.get("max_hold_hours", 120)
                atomic_write_json(portfolio, get_path("portfolio_state"))
                bot_entered = "bot2"
                result["signal"] = "ENTER_BOT2"
                logger.info(
                    f"BOT2 ENTRY CONFIRMED: stablecoin_z={mf_check['stablecoin_z']:.2f} | "
                    f"ret_1d={mf_check['ret_1d']:.4f} | RSI={mf_check['rsi']:.1f} | "
                    f"BB={mf_check['bb_pct']:.3f}"
                )
            else:
                # Log Bot 2 check only if it's close to triggering (stablecoin_z > 0.5)
                if mf_check.get("stablecoin_z") and mf_check["stablecoin_z"] > 0.5:
                    logger.info(
                        f"BOT2 FILTERED: {mf_check['reason']} | "
                        f"stable_z={mf_check.get('stablecoin_z')}"
                    )
```

### 3b. Adicionar _execute_bot2_entry() — entrada com stops fixos do Bot 2

Adicionar como nova função no paper_trader.py (ANTES de run_cycle):

```python
def _execute_bot2_entry(current_price: float, portfolio: dict, mf_params: dict) -> dict:
    """
    Execute Bot 2 entry with its own fixed stops (not ATR-based).
    Bot 2 stops: SG 2% / SL 1.5% / trailing 1% / max 120h.
    """
    params = get_params()["execution"]
    capital = portfolio["capital_usd"]
    size_pct = params["position_size_pct"]

    position_value = capital * size_pct
    quantity = position_value / current_price

    sl_pct = mf_params.get("stop_loss_pct", 0.015)
    tp_pct = mf_params.get("take_profit_pct", 0.02)
    trail_pct = mf_params.get("trailing_stop_pct", 0.01)

    portfolio["has_position"] = True
    portfolio["entry_price"] = round(current_price, 2)
    portfolio["entry_time"] = str(pd.Timestamp.utcnow())
    portfolio["quantity"] = round(quantity, 6)
    portfolio["trailing_high"] = round(current_price, 2)
    portfolio["stop_loss_price"] = round(current_price * (1 - sl_pct), 2)
    portfolio["take_profit_price"] = round(current_price * (1 + tp_pct), 2)
    portfolio["trailing_stop_pct_actual"] = trail_pct
    portfolio["stops_mode"] = "bot2_fixed"
    portfolio["entry_atr_pct"] = None
    portfolio["last_updated"] = str(pd.Timestamp.utcnow())

    atomic_write_json(portfolio, get_path("portfolio_state"))
    logger.info(
        f"BOT2 ENTRY: price={current_price:.2f}, qty={quantity:.6f}, "
        f"SL={portfolio['stop_loss_price']}, TP={portfolio['take_profit_price']} [bot2_fixed]"
    )
    return portfolio
```

### 3c. Max hold timeout — Adicionar check no bloco de stops

No bloco de check_stops (dentro de `if portfolio["has_position"]:`), ANTES de `check_stops()`, adicionar:

```python
            # Bot 2 max hold timeout check
            entry_bot = portfolio.get("entry_bot", "bot1")
            if entry_bot == "bot2":
                max_hold_h = portfolio.get("entry_max_hold_hours", 120)
                entry_time = parse_utc(portfolio.get("entry_time", ""))
                hours_in_trade = (cycle_ts - entry_time).total_seconds() / 3600
                if hours_in_trade >= max_hold_h:
                    completed_trade = _build_trade_record(portfolio, current_price, "bot2_timeout")
                    portfolio = execute_exit(current_price, portfolio, "bot2_timeout")
                    _save_completed_trade(completed_trade)
                    portfolio = load_portfolio()
                    logger.info(
                        f"BOT2 TIMEOUT: {hours_in_trade:.0f}h >= {max_hold_h}h | "
                        f"exit=${current_price:,.0f}"
                    )
```

---

## PARTE 4: _init_trade_tracking — Adicionar entry_bot

No `_init_trade_tracking()`, adicionar ao final (a linha `portfolio["entry_bot"]` já é setada no run_cycle, mas queremos garantir default):

```python
    # Bot identifier (set by caller, default to bot1 for backward compat)
    portfolio.setdefault("entry_bot", "bot1")
```

---

## PARTE 5: _build_trade_record — Adicionar entry_bot ao trade record

Localizar o `return` dict em `_build_trade_record()` e adicionar:

```python
        "entry_bot": portfolio.get("entry_bot", "bot1"),
        "entry_stablecoin_z": portfolio.get("entry_stablecoin_z"),
        "entry_max_hold_hours": portfolio.get("entry_max_hold_hours"),
```

---

## PARTE 6: log_cycle — Adicionar log do Bot 2

Adicionar caso ENTER_BOT2 no log_cycle():

```python
    elif _sig == "ENTER_BOT2":
        logger.info(
            f"CYCLE [{cycle_ts}]: ENTER_BOT2 "
            f"stablecoin_z={result.get('entry_stablecoin_z', '?')} | "
            f"close={technical.get('close')} bb={technical.get('bb_pct')} "
            f"rsi={technical.get('rsi_14')} | "
            f"capital=${portfolio.get('capital_usd'):.2f} pos={portfolio.get('has_position')}"
        )
```

---

## PARTE 7: Portfolio state — Persistir estado do Bot 2

No bloco final de `run_cycle()` (step 11b), adicionar:

```python
    # Momentum filter state — persisted every cycle for dashboard/debug
    portfolio["last_momentum_passed"] = result.get("signal") == "ENTER_BOT2"
    portfolio["last_momentum_reason"] = mf_check.get("reason") if 'mf_check' in dir() else None
    portfolio["last_momentum_stablecoin_z"] = zscores.get("stablecoin_z")
```

**IMPORTANTE**: A variável `mf_check` precisa existir fora do bloco condicional. Inicializar no topo da seção de entry decision:

```python
        mf_check = {"passed": False, "reason": "not_evaluated"}
```

---

## PARTE 8: Dashboard — Adicionar indicação Bot 2

No `app.py`, na seção 1 (Header), onde mostra o signal:

### 8a. Signal display — diferenciar Bot 1 e Bot 2

Onde exibe o signal com cor, adicionar caso para ENTER_BOT2:

```python
elif signal == "ENTER_BOT2":
    st.markdown(f"🟢 **ENTER (Bot2 Momentum)**")
```

### 8b. Na seção 2 (Gate Scoring), adicionar painel Bot 2

Após o painel de Gate Scoring, adicionar um expander:

```python
with st.expander("🚀 Bot 2 — Momentum/Liquidez", expanded=False):
    mf_params = params.get("momentum_filter", {})
    if not mf_params.get("enabled", False):
        st.info("Bot 2 desabilitado")
    else:
        col1, col2, col3, col4, col5 = st.columns(5)
        stable_z = zs.get("stablecoin_z", 0)
        ret_1d_val = portfolio.get("last_filter_ret_1d")
        rsi_val = latest_tech.get("rsi_14", 0)
        bb_val = latest_tech.get("bb_pct", 0)
        close_val = latest_tech.get("close", 0)
        ma21_val = latest_tech.get("ma_21", 0)

        sz_min = mf_params.get("stablecoin_z_min", 1.3)
        above_ma21 = close_val > ma21_val if close_val and ma21_val else False

        col1.metric("Stablecoin Z", f"{stable_z:.2f}",
                     delta="✓" if stable_z > sz_min else "✗")
        col2.metric("ret_1d", f"{ret_1d_val:.4f}" if ret_1d_val else "N/A",
                     delta="✓" if ret_1d_val and ret_1d_val > 0 else "✗")
        col3.metric("RSI", f"{rsi_val:.1f}",
                     delta="✓" if rsi_val > 50 else "✗")
        col4.metric("BB%", f"{bb_val:.3f}",
                     delta="✓" if bb_val < 0.98 else "✗")
        col5.metric(">MA21", "Yes" if above_ma21 else "No",
                     delta="✓" if above_ma21 else "✗")

        # Status
        all_pass = (stable_z > sz_min and ret_1d_val and ret_1d_val > 0 and
                    rsi_val > 50 and bb_val < 0.98 and above_ma21)
        if portfolio.get("has_position") and portfolio.get("entry_bot") == "bot2":
            st.success("🟢 Bot 2 POSIÇÃO ABERTA")
        elif portfolio.get("has_position"):
            st.warning("Bot 1 tem posição — Bot 2 bloqueado (mutex)")
        elif all_pass:
            st.success("✅ Todas as condições atendidas — aguardando próximo ciclo")
        else:
            reason = portfolio.get("last_momentum_reason", "")
            st.info(f"Bot 2 aguardando condições | {reason}")
```

### 8c. Na seção 8 (Paper Trading), diferenciar trades Bot 1 vs Bot 2

Onde mostra trades, adicionar coluna `entry_bot`:

```python
# Na tabela de trades, incluir a coluna entry_bot
if "entry_bot" in trades_df.columns:
    # Colorir Bot 1 = azul, Bot 2 = verde
    ...
```

---

## PARTE 9: append_score_history — Registrar Bot 2

No `append_score_history()`, adicionar:

```python
        "entry_bot": result.get("entry_bot"),
```

E no `log_cycle`, adicionar ao `row`:

```python
        "entry_bot": portfolio.get("entry_bot") if portfolio.get("has_position") else None,
```

---

## PARTE 10: Testes

### 10a. test_momentum_filter.py

```python
"""Tests for check_momentum_filter()."""
import pytest
from src.trading.paper_trader import check_momentum_filter

PARAMS_ENABLED = {
    "momentum_filter": {
        "enabled": True,
        "stablecoin_z_min": 1.3,
        "ret_1d_min": 0.0,
        "rsi_min": 50,
        "bb_pct_max": 0.98,
        "require_above_ma21": True,
    }
}

PARAMS_DISABLED = {
    "momentum_filter": {"enabled": False}
}


def _tech(close=75000, rsi=65, bb=0.85, ret=0.01, ma21=73000):
    return {
        "close": close,
        "rsi_14": rsi,
        "bb_pct": bb,
        "ret_1d": ret,
        "ma_21": ma21,
    }


def _zscores(stable_z=1.5):
    return {"stablecoin_z": stable_z}


class TestMomentumFilterBasic:
    """Basic pass/fail tests."""

    def test_all_conditions_pass(self):
        r = check_momentum_filter(_tech(), _zscores(), PARAMS_ENABLED)
        assert r["passed"] is True
        assert r["reason"] == "momentum_confirmed"

    def test_filter_disabled(self):
        r = check_momentum_filter(_tech(), _zscores(), PARAMS_DISABLED)
        assert r["passed"] is False
        assert r["reason"] == "filter_disabled"

    def test_low_stablecoin_z(self):
        r = check_momentum_filter(_tech(), _zscores(stable_z=0.5), PARAMS_ENABLED)
        assert r["passed"] is False
        assert "LOW_LIQUIDITY" in r["reason"]

    def test_negative_ret_1d(self):
        r = check_momentum_filter(_tech(ret=-0.01), _zscores(), PARAMS_ENABLED)
        assert r["passed"] is False
        assert "NEG_MOMENTUM" in r["reason"]

    def test_low_rsi(self):
        r = check_momentum_filter(_tech(rsi=40), _zscores(), PARAMS_ENABLED)
        assert r["passed"] is False
        assert "RSI_LOW" in r["reason"]

    def test_blow_off_top(self):
        r = check_momentum_filter(_tech(bb=0.99), _zscores(), PARAMS_ENABLED)
        assert r["passed"] is False
        assert "BLOW_OFF_TOP" in r["reason"]

    def test_below_ma21(self):
        r = check_momentum_filter(_tech(close=72000, ma21=73000), _zscores(), PARAMS_ENABLED)
        assert r["passed"] is False
        assert "BELOW_MA21" in r["reason"]


class TestMomentumFilterEdgeCases:
    """Edge cases and multiple failures."""

    def test_multiple_failures(self):
        r = check_momentum_filter(
            _tech(rsi=40, ret=-0.01, bb=0.99),
            _zscores(stable_z=0.5),
            PARAMS_ENABLED
        )
        assert r["passed"] is False
        assert "LOW_LIQUIDITY" in r["reason"]
        assert "NEG_MOMENTUM" in r["reason"]
        assert "RSI_LOW" in r["reason"]
        assert "BLOW_OFF_TOP" in r["reason"]

    def test_missing_stablecoin_z(self):
        r = check_momentum_filter(_tech(), {"stablecoin_z": None}, PARAMS_ENABLED)
        assert r["passed"] is False
        assert "MISSING_DATA" in r["reason"]

    def test_missing_rsi(self):
        tech = _tech()
        tech["rsi_14"] = None
        r = check_momentum_filter(tech, _zscores(), PARAMS_ENABLED)
        assert r["passed"] is False
        assert "MISSING_DATA" in r["reason"]

    def test_exact_threshold_stablecoin(self):
        """stablecoin_z == 1.3 should NOT pass (need > 1.3)."""
        r = check_momentum_filter(_tech(), _zscores(stable_z=1.3), PARAMS_ENABLED)
        assert r["passed"] is False

    def test_exact_threshold_bb(self):
        """bb_pct == 0.98 should NOT pass (need < 0.98)."""
        r = check_momentum_filter(_tech(bb=0.98), _zscores(), PARAMS_ENABLED)
        assert r["passed"] is False

    def test_ret_1d_zero_should_fail(self):
        """ret_1d == 0 should NOT pass (need > 0)."""
        r = check_momentum_filter(_tech(ret=0.0), _zscores(), PARAMS_ENABLED)
        assert r["passed"] is False

    def test_rsi_exactly_50_should_fail(self):
        """RSI == 50 should NOT pass (need > 50)."""
        r = check_momentum_filter(_tech(rsi=50), _zscores(), PARAMS_ENABLED)
        assert r["passed"] is False

    def test_missing_ma21_data(self):
        tech = _tech()
        tech["ma_21"] = None
        r = check_momentum_filter(tech, _zscores(), PARAMS_ENABLED)
        assert r["passed"] is False
        assert "MISSING_MA21" in r["reason"]


class TestMomentumFilterBoundary:
    """Just-passing thresholds."""

    def test_stablecoin_just_above(self):
        r = check_momentum_filter(_tech(), _zscores(stable_z=1.31), PARAMS_ENABLED)
        assert r["passed"] is True

    def test_ret_just_above_zero(self):
        r = check_momentum_filter(_tech(ret=0.0001), _zscores(), PARAMS_ENABLED)
        assert r["passed"] is True

    def test_bb_just_below_threshold(self):
        r = check_momentum_filter(_tech(bb=0.979), _zscores(), PARAMS_ENABLED)
        assert r["passed"] is True

    def test_rsi_just_above_50(self):
        r = check_momentum_filter(_tech(rsi=50.1), _zscores(), PARAMS_ENABLED)
        assert r["passed"] is True
```

### 10b. test_bot2_integration.py

```python
"""Integration tests for Bot 2 in run_cycle flow."""
import pytest
from unittest.mock import patch, MagicMock

class TestBot2Integration:
    """Test Bot 2 entry path in run_cycle."""

    def test_bot2_does_not_enter_during_bear_regime(self):
        """Bear regime blocks both bots."""
        # Mock regime as Bear, momentum conditions met
        # Verify no entry

    def test_bot2_enters_when_scoring_says_hold(self):
        """Bot 2 can enter even if gate scoring says HOLD."""
        # Mock scoring result as HOLD, momentum conditions met
        # Verify entry with entry_bot="bot2"

    def test_bot2_blocked_when_bot1_has_position(self):
        """Mutex: Bot 2 cannot enter if Bot 1 has open position."""
        # Mock portfolio with has_position=True, entry_bot="bot1"
        # Verify Bot 2 does not enter

    def test_bot1_has_priority_over_bot2(self):
        """If scoring says ENTER and reversal passes, Bot 1 takes precedence."""
        # Mock scoring ENTER + reversal passes + momentum passes
        # Verify entry_bot="bot1"

    def test_bot2_entry_uses_own_stops(self):
        """Bot 2 entry should use momentum_filter stops, not execution stops."""
        # Verify stop_loss_price = entry * (1 - 0.015)
        # Verify take_profit_price = entry * (1 + 0.02)
        # Verify trailing = 0.01
        # Verify stops_mode = "bot2_fixed"

    def test_bot2_timeout_exit(self):
        """Bot 2 exits after max_hold_hours."""
        # Mock entry_time 121h ago, entry_bot="bot2"
        # Verify exit with reason "bot2_timeout"

    def test_bot2_timeout_does_not_affect_bot1(self):
        """Bot 1 trades do NOT have max_hold_hours timeout."""
        # Mock entry_time 200h ago, entry_bot="bot1"
        # Verify no timeout exit
```

### 10c. test_bot2_parameters.py

```python
"""Test Bot 2 parameters in parameters.yml."""
import pytest
from src.config import get_params

class TestBot2Parameters:
    def test_momentum_filter_exists(self):
        params = get_params()
        assert "momentum_filter" in params

    def test_momentum_filter_has_all_keys(self):
        mf = get_params()["momentum_filter"]
        required = ["enabled", "stablecoin_z_min", "ret_1d_min", "rsi_min",
                     "bb_pct_max", "require_above_ma21",
                     "stop_loss_pct", "take_profit_pct", "trailing_stop_pct",
                     "max_hold_hours"]
        for key in required:
            assert key in mf, f"Missing key: {key}"

    def test_momentum_filter_values(self):
        mf = get_params()["momentum_filter"]
        assert mf["stablecoin_z_min"] == 1.3
        assert mf["ret_1d_min"] == 0.0
        assert mf["rsi_min"] == 50
        assert mf["bb_pct_max"] == 0.98
        assert mf["stop_loss_pct"] == 0.015
        assert mf["take_profit_pct"] == 0.02
        assert mf["trailing_stop_pct"] == 0.01
        assert mf["max_hold_hours"] == 120
```

---

## Checklist de Implementação

1. [ ] `conf/parameters.yml` — Adicionar seção `momentum_filter`
2. [ ] `src/trading/paper_trader.py` — `check_momentum_filter()`
3. [ ] `src/trading/paper_trader.py` — `_execute_bot2_entry()`
4. [ ] `src/trading/paper_trader.py` — Modificar `run_cycle()` (Bot 2 entry path)
5. [ ] `src/trading/paper_trader.py` — Bot 2 timeout em check_stops
6. [ ] `src/trading/paper_trader.py` — `_init_trade_tracking()` default entry_bot
7. [ ] `src/trading/paper_trader.py` — `_build_trade_record()` entry_bot field
8. [ ] `src/trading/paper_trader.py` — `log_cycle()` ENTER_BOT2 case
9. [ ] `src/trading/paper_trader.py` — Persistir momentum filter state
10. [ ] `src/dashboard/app.py` — Signal display ENTER_BOT2
11. [ ] `src/dashboard/app.py` — Bot 2 panel no Gate Scoring
12. [ ] `src/dashboard/app.py` — Diferenciar trades Bot 1/Bot 2
13. [ ] `tests/test_momentum_filter.py` — 17+ testes unitários
14. [ ] `tests/test_bot2_integration.py` — 7 testes de integração
15. [ ] `tests/test_bot2_parameters.py` — 3 testes de parâmetros
16. [ ] Rodar `pytest` — todos os testes passando
17. [ ] Git commit + push
18. [ ] Docker build + deploy EC2

## Notas de Cuidado

- **NÃO alterar** o gate scoring (gate_scoring.py) — Bot 2 é 100% no paper_trader
- **NÃO alterar** as regras do Bot 1 (reversal filter) — tudo existente fica intocado
- **NÃO alterar** execution.py — Bot 2 usa `_execute_bot2_entry()` com stops próprios
- O `check_stops()` existente funciona para ambos os bots (SL/TP/trailing)
- O timeout do Bot 2 é verificado ANTES do check_stops no ciclo
- Backward compatibility: trades antigos sem `entry_bot` → default "bot1"
