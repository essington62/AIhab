# Prompt: Dashboard Consolidation — ETH, SOL, Cleanup

## Contexto

Continuando refactor do dashboard `src/dashboard/app.py`. Duas iterações já aplicadas:

**Iteração 1:** Bot 1 Conservador + trades enriquecidos + Whale comentado
**Iteração 2:** Bot 2 Momentum (filtros + trades) adicionado

**Screenshot atual (confirmado pelo usuário):**
```
1. Header (price, regime, capital)
2. 🛡️ BOT 1 - Conservador (Gate Scoring v2 completo)
3. 📊 Histórico Bot 1 — Trades
4. 🚀 BOT 2 - Momentum (filtros + trades)
5. 🤖 AI Analyst (DeepSeek)
6. ⚙️ System Health
7. 📐 Model Health
8. ⚖️ Adaptive Weights
9. 🔴 Paper Trading Bot 1 — Trades  ← DUPLICADO
10. 🔴 Paper Trading Bot 2 — Trades  ← DUPLICADO
11. (Fed Observatory, Macro, Derivativos, News — provavelmente mais acima)
```

**Sistema SOL Bot 4 em produção:** primeira posição aberta ($88.23), Sharpe backtest 2.03.

## Objetivo

Esta iteração faz 5 mudanças:

1. **Adicionar seção ETH - Bot 3** após Bot 2 (filtros + trades)
2. **Adicionar seção SOL - Bot 4** após ETH (filtros + trades)
3. **Remover Derivativos + Macro** (display duplicado, não agrega)
4. **Enriquecer Fed Sentinel** com cenários de corte (screenshot referência)
5. **Remover Paper Trading duplicados** no fim (Bot 1/Bot 2)

## Estrutura desejada após mudanças

```
1. Header (price, regime, capital) — sem mudança
2. 🛡️ BOT 1 - Conservador — sem mudança
3. 📊 Histórico Bot 1 — Trades — sem mudança
4. 🚀 BOT 2 - Momentum — sem mudança
5. 🆕 ⚡ ETH - Bot 3 (Volume Defensivo) ← NOVO
   ├── Filtros atuais
   └── Histórico trades
6. 🆕 🟣 SOL - Bot 4 (Taker/Flow) ← NOVO (ASSET NOVO)
   ├── Filtros atuais
   ├── Shadow scoring alternative
   └── Histórico trades
7. 🤖 AI Analyst (DeepSeek) — sem mudança
8. ⚙️ System Health — sem mudança
9. 📐 Model Health — sem mudança
10. ⚖️ Adaptive Weights — sem mudança
11. ❌ Paper Trading Bot 1/Bot 2 (duplicados) — REMOVER
12. 📰 News + 🏛️ Fed Sentinel ENRIQUECIDO ← MODIFICAR
13. ❌ Derivativos — REMOVER
14. ❌ Macro — REMOVER
```

---

## Mudança 1: Seção ETH - Bot 3 (Volume Defensivo)

### Filosofia Bot 3

Bot 3 é **volume-only defensivo** para ETH. Entra no Q2 (quantile 2) do volume histórico — zona de volume **baixo-médio**, esperando mean reversion suave.

**Filtros Bot 3:**

| Filtro | Condição | Fonte |
|--------|----------|-------|
| Volume Q2 | `vol_z` entre -0.75 e -0.30 | OHLCV ETH |
| RSI range | `RSI < 60` | OHLCV ETH |
| MA200 trend | `close > MA200` | OHLCV ETH |

(Confirmar valores exatos em `src/trading/eth_bot3.py` e `conf/parameters_eth.yml`)

### Implementação

Onde buscar valores:
- `data/01_raw/spot/eth_1h.parquet` (OHLCV)
- `data/05_output/portfolio_state_eth.json` OU `portfolio_eth.json` (state)
- `src/trading/eth_bot3.py` (lógica de filtros)

**Código sugerido:**

```python
# ========================================
# ⚡ ETH - BOT 3 (Volume Defensivo)
# ========================================
st.markdown("---")
st.markdown("## ⚡ ETH - Bot 3 (Volume Defensivo)")
st.caption("Filosofia: mean reversion em volume baixo-médio (Q2). Conservador por natureza.")

# Header com preço ETH + capital
eth_price = get_latest_price("eth")  # implementar helper
eth_portfolio = load_portfolio_state_eth()  # implementar helper

col1, col2, col3 = st.columns(3)
col1.metric("💰 ETH Price", f"${eth_price:,.2f}" if eth_price else "—")
col2.metric("📊 Capital", f"${eth_portfolio.get('current_capital', 10000):,.2f}")
col3.metric("🎯 Posição", "ABERTA" if eth_portfolio.get('has_position') else "—")

# Filtros
vol_z_current = eth_portfolio.get("last_vol_z") or compute_vol_z_eth()
rsi_eth = eth_portfolio.get("last_rsi") or compute_rsi_eth()
close_above_ma200_eth = eth_portfolio.get("last_close_above_ma200", None)

# Status header
filter_results_eth = {
    "vol_q2": (-0.75 < vol_z_current < -0.30 if vol_z_current else False, vol_z_current),
    "rsi": (rsi_eth < 60 if rsi_eth else False, rsi_eth),
    "ma200": (close_above_ma200_eth if close_above_ma200_eth is not None else False, None),
}

total_passed_eth = sum(1 for passed, _ in filter_results_eth.values() if passed)
total_filters_eth = len(filter_results_eth)

if total_passed_eth == total_filters_eth:
    st.success(f"✅ ENTRY ELEGÍVEL — {total_passed_eth}/{total_filters_eth} filtros")
else:
    st.warning(f"🛑 WAIT — {total_passed_eth}/{total_filters_eth} filtros")

# Filter cards
cols = st.columns(3)

with cols[0]:
    passed, value = filter_results_eth["vol_q2"]
    badge = "✅" if passed else "❌"
    st.markdown(f"**📊 Volume Q2**  {badge}")
    st.metric(
        label="vol_z",
        value=f"{value:.2f}" if value else "—",
        delta="-0.75 < z < -0.30" if passed else "fora Q2"
    )

with cols[1]:
    passed, value = filter_results_eth["rsi"]
    badge = "✅" if passed else "❌"
    st.markdown(f"**📉 RSI < 60**  {badge}")
    st.metric(
        label="RSI",
        value=f"{value:.1f}" if value else "—",
        delta="não sobrecomprado" if passed else "sobrecomprado"
    )

with cols[2]:
    passed, value = filter_results_eth["ma200"]
    badge = "✅" if passed else "❌"
    st.markdown(f"**📈 Trend MA200**  {badge}")
    st.metric(
        label="close vs MA200",
        value="ACIMA" if passed else "ABAIXO",
        delta="uptrend confirmado" if passed else "downtrend"
    )

# Histórico Trades Bot 3
st.markdown("### 📊 Histórico Bot 3 — Trades")
trades_bot3 = load_trades_filtered(asset="eth", bot="bot3")

if trades_bot3.empty:
    st.info("Nenhum trade Bot 3 completado ainda.")
else:
    render_trades_table(trades_bot3, include_stops=True)
    
    # Métricas
    n_total = len(trades_bot3)
    n_wins = (trades_bot3["return_pct"] > 0).sum()
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Trades", n_total)
    col2.metric("Win Rate", f"{n_wins/n_total*100:.1f}%")
    col3.metric("Avg Return", f"{trades_bot3['return_pct'].mean():+.2f}%")
    col4.metric("Total Return", f"{((1+trades_bot3['return_pct']/100).prod()-1)*100:+.2f}%")
```

---

## Mudança 2: Seção SOL - Bot 4 (Taker/Flow)

### Filosofia Bot 4

Bot 4 é **taker aggression + flow** para SOL. Entra quando compra agressiva confirmada + OI saudável + ETH context positivo + Bot 2 DNA filters passam.

**Filtros Bot 4 (Hard gates):**

| Filtro | Condição | Fonte |
|--------|----------|-------|
| Taker Z | `taker_z_4h > 0.3` | derivatives SOL 4h |
| OI Block | `oi_z_24h < 2.0` | derivatives SOL 4h |
| ETH Context | `eth_ret_1h > 0` | OHLCV ETH |
| RSI range | `60 < RSI < 80` | OHLCV SOL |
| Stablecoin | `stablecoin_z > 1.3` | features |
| Trend MA21 | `close > MA21 SOL` | OHLCV SOL |

**Shadow Scoring Alternative:**
```
score = taker*2 + eth*1 + oi*1
if score >= 3: would_enter=True
```

Loga em `data/08_shadow/sol_scoring_shadow_log.jsonl` (já implementado).

### Implementação

Onde buscar valores:
- `data/01_raw/spot/sol_1h.parquet` (OHLCV SOL)
- `data/01_raw/futures/sol_oi_4h.parquet`, `sol_taker_4h.parquet`
- `data/05_output/portfolio_state_sol.json` (state Bot 4)
- `data/08_shadow/sol_scoring_shadow_log.jsonl` (shadow scoring)
- `src/trading/sol_bot4.py` (lógica hard gates)

**Código sugerido:**

```python
# ========================================
# 🟣 SOL - BOT 4 (Taker/Flow)
# ========================================
st.markdown("---")
st.markdown("## 🟣 SOL - Bot 4 (Taker/Flow)")
st.caption("Filosofia: compra agressiva + fluxo saudável + contexto ETH. Hard gates + Bot 2 DNA.")

# Header
sol_price = get_latest_price("sol")
sol_portfolio = load_portfolio_state_sol()

col1, col2, col3 = st.columns(3)
col1.metric("💰 SOL Price", f"${sol_price:,.2f}" if sol_price else "—")
col2.metric("📊 Capital", f"${sol_portfolio.get('current_capital', 10000):,.2f}")

position_status = "ABERTA" if sol_portfolio.get('has_position') else "—"
col3.metric("🎯 Posição", position_status)

# Se posição aberta, destacar
if sol_portfolio.get('has_position'):
    st.success(
        f"🟢 **POSIÇÃO ABERTA** — Entry: ${sol_portfolio['entry_price']:.2f} | "
        f"Current P&L: {((sol_price/sol_portfolio['entry_price'])-1)*100:+.2f}% | "
        f"Trail: ${sol_portfolio.get('trailing_stop_price', 0):.2f} | "
        f"TP: ${sol_portfolio.get('take_profit_price', 0):.2f}"
    )

# Hard gates (3 principais)
taker_z_sol = sol_portfolio.get("last_taker_z") or load_feature("sol", "taker_z")
oi_z_24h_sol = sol_portfolio.get("last_oi_z_24h") or load_feature("sol", "oi_z_24h")
eth_ret_1h = sol_portfolio.get("last_eth_ret_1h") or load_feature("eth", "ret_1h")

st.markdown("### 🎯 Hard Gates")

hard_gates = {
    "taker": (taker_z_sol > 0.3 if taker_z_sol else False, taker_z_sol, "> 0.3"),
    "oi_block": (oi_z_24h_sol < 2.0 if oi_z_24h_sol else True, oi_z_24h_sol, "< 2.0"),
    "eth_context": (eth_ret_1h > 0 if eth_ret_1h else False, eth_ret_1h, "> 0"),
}

cols = st.columns(3)
with cols[0]:
    passed, val, thr = hard_gates["taker"]
    badge = "✅" if passed else "❌"
    st.markdown(f"**🔥 Taker Z**  {badge}")
    st.metric("taker_z_4h", f"{val:+.2f}" if val else "—", thr)

with cols[1]:
    passed, val, thr = hard_gates["oi_block"]
    badge = "✅" if passed else "❌"
    st.markdown(f"**📊 OI Block**  {badge}")
    st.metric("oi_z_24h", f"{val:+.2f}" if val else "—", thr)

with cols[2]:
    passed, val, thr = hard_gates["eth_context"]
    badge = "✅" if passed else "❌"
    st.markdown(f"**🌉 ETH Context**  {badge}")
    st.metric("eth_ret_1h", f"{val*100:+.2f}%" if val else "—", thr)

# Bot 2 DNA filters (herdados do Bot 2 BTC adaptados)
st.markdown("### 🧬 Bot 2 DNA (Momentum)")

# Reusar lógica do Bot 2 mas pra SOL
stablecoin_z_sol = sol_portfolio.get("last_stablecoin_z")
rsi_sol = sol_portfolio.get("last_rsi")
bb_pct_sol = sol_portfolio.get("last_bb_pct")
close_above_ma21_sol = sol_portfolio.get("last_close_above_ma21")
ret_1d_sol = sol_portfolio.get("last_ret_1d")

cols = st.columns(5)
# ... (similar ao Bot 2 BTC, cada um com seu card + badge)

# Shadow Scoring Alternative
st.markdown("### 👻 Shadow Scoring Alternative")
st.caption("Proposta alternativa: score = taker×2 + eth + oi. Entry se score ≥ 3.")

# Ler últimas entries do shadow log
shadow_path = "data/08_shadow/sol_scoring_shadow_log.jsonl"
try:
    import json
    shadow_entries = []
    with open(shadow_path) as f:
        for line in f:
            shadow_entries.append(json.loads(line))
    
    if shadow_entries:
        shadow_df = pd.DataFrame(shadow_entries)
        # Mostrar as últimas 5 entries
        st.dataframe(shadow_df.tail(5)[["timestamp", "score_total", "would_enter", "breakdown"]], use_container_width=True)
        
        # Métricas
        total_shadow = len(shadow_df)
        would_enter_count = shadow_df["would_enter"].sum()
        agree_with_hard = (shadow_df["would_enter"] == shadow_df["hard_gate_entered"]).mean() * 100 if "hard_gate_entered" in shadow_df.columns else None
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Shadow Entries", total_shadow)
        col2.metric("Would-Enter", int(would_enter_count))
        if agree_with_hard is not None:
            col3.metric("Agreement %", f"{agree_with_hard:.1f}%")
    else:
        st.info("Shadow log vazio — aguardando primeiros cycles.")
except FileNotFoundError:
    st.info("Shadow log não encontrado.")

# Histórico Trades Bot 4
st.markdown("### 📊 Histórico Bot 4 — Trades")
trades_bot4 = load_trades_filtered(asset="sol", bot="bot4")

if trades_bot4.empty:
    st.info("Nenhum trade Bot 4 completado ainda. Primeira posição aberta hoje 22/04 às 16:15 UTC!")
else:
    render_trades_table(trades_bot4, include_stops=True)
    # ... métricas
```

---

## Mudança 3: Enriquecer Fed Sentinel com cenários

### Contexto

Fed Sentinel atual mostra apenas:
- Próximo evento (chair_term_end, em 23 dias)
- Proximity adjustment
- Blackout status

**Screenshot referência mostra cenários adicionais:**

```
Corte 25bps — 35%
Impacto BTC: Bullish | Ação: Reduzir threshold, aumentar sizing em Sideways
Historicamente, primeiro corte após ciclo de alta = BTC sobe 30-50% em 6 meses.
Dólar enfraquece, capital migra pra risk-on. Stablecoins entram forte.

Manutenção — 65%
Impacto BTC: Neutro | Ação: Manter scoring atual
Já precificado pelo mercado. Impacto depende do guidance e dot plot.
Atenção ao tom da coletiva — hawkish surprise pode derrubar,
dovish surprise pode subir.
```

### Implementação

Localizar seção Fed Sentinel em `src/dashboard/app.py` e **adicionar abaixo do estado atual**:

```python
# Dentro da seção Fed Sentinel existente, APÓS informações atuais:

st.markdown("#### 📋 Cenários prováveis")
st.caption("Estimativas baseadas em DGS2 (proxy de expectativa de juros)")

# Pegar probabilidades do fed_observatory
prob_cut = fed_data.get("prob_cut", 0.35)
prob_hold = fed_data.get("prob_hold", 0.65)
prob_hike = fed_data.get("prob_hike", 0.0)

# Cenário 1: Corte 25bps
if prob_cut > 0:
    with st.container():
        st.markdown(f"**🟢 Corte 25bps — {prob_cut*100:.0f}%**")
        st.caption(
            "**Impacto BTC:** Bullish | **Ação:** Reduzir threshold, aumentar sizing em Sideways\n\n"
            "Historicamente, primeiro corte após ciclo de alta = BTC sobe 30-50% em 6 meses. "
            "Dólar enfraquece, capital migra pra risk-on. Stablecoins entram forte."
        )

# Cenário 2: Manutenção
if prob_hold > 0:
    with st.container():
        st.markdown(f"**🟡 Manutenção — {prob_hold*100:.0f}%**")
        st.caption(
            "**Impacto BTC:** Neutro | **Ação:** Manter scoring atual\n\n"
            "Já precificado pelo mercado. Impacto depende do guidance e dot plot. "
            "Atenção ao tom da coletiva — hawkish surprise pode derrubar, "
            "dovish surprise pode subir."
        )

# Cenário 3: Alta (só se prob > 0)
if prob_hike > 0:
    with st.container():
        st.markdown(f"**🔴 Alta 25bps — {prob_hike*100:.0f}%**")
        st.caption(
            "**Impacto BTC:** Bearish | **Ação:** Elevar threshold, reduzir sizing\n\n"
            "Hike surprise em ciclo de corte = forte repricing. "
            "Dollar strengthening, crypto sell-off típico. "
            "Preparar Bot 1 pra BLOCK mais rígido."
        )
```

### Importante sobre textos

Os textos dos cenários acima são os mesmos do screenshot. Usar **exatamente como estão** pra manter consistência visual.

Probabilidades vêm de `fed_observatory.py` (DGS2 proxy):
- Verificar se `get_fed_observatory_data()` retorna `prob_cut`, `prob_hold`, `prob_hike`
- Se não, adicionar computação

---

## Mudança 4: REMOVER seções

### 4a. Remover Derivativos
Localizar seção Derivativos (OI, Funding, Taker, Liquidações, Bid/Ask) e **remover completamente**.

Motivo: 
- Já aparecem implicitamente nos bots (Bot 1 Gate Scoring, Bot 4 Hard gates)
- Display duplicado
- Polui dashboard

**Comentar código ou deletar:**
```python
# REMOVED — derivativos já implícitos nos bots
# Ver git history se precisar reverter
```

### 4b. Remover Macro
Localizar seção Macro (DGS10, DGS2, 2/10Y, VIX, DXY, OIL, S&P500) e **remover completamente**.

Motivo:
- Só Bot 1 usa em decisão
- Já aparece no cluster Macro do Gate Scoring
- Display duplicado

### 4c. Remover Paper Trading duplicados

**No final do dashboard (após Adaptive Weights), remover:**
- 🔴 Paper Trading Bot 1 — Trades
- 🔴 Paper Trading Bot 2 — Trades

Motivo: já aparecem DEPOIS do BOT 1 Conservador e DEPOIS do BOT 2 Momentum (iterações anteriores). Duplicação visual.

---

## Tarefas em ordem

1. **Ler código atual** `src/dashboard/app.py`:
   - Mapear onde cada seção está (linhas)
   - Identificar helpers existentes

2. **Criar helpers novos (se não existem):**
   - `load_portfolio_state_eth()` 
   - `load_portfolio_state_sol()`
   - `get_latest_price(asset)` — BTC, ETH, SOL
   - `load_feature(asset, feature_name)` — pra pegar valores atuais
   - `load_trades_filtered(asset="btc", bot="bot1")` — expandir parâmetro asset

3. **Adicionar seções ETH e SOL** após Bot 2

4. **Enriquecer Fed Sentinel** com 2-3 cenários

5. **Remover seções duplicadas** (Derivativos, Macro, Paper Trading duplicado)

6. **Testar local** `streamlit run src/dashboard/app.py`

7. **Validar estrutura final** (ver checklist abaixo)

## Validação

```bash
# Sintaxe
python -c "import ast; ast.parse(open('src/dashboard/app.py').read())"

# Run
streamlit run src/dashboard/app.py
```

**Checklist:**
```
☐ BOT 1 Conservador (inalterado)
☐ Trades Bot 1 (inalterado)
☐ BOT 2 Momentum (inalterado)
☐ Trades Bot 2 (inalterado)
☐ ⚡ ETH - Bot 3 (NOVO — com filtros e trades)
☐ 🟣 SOL - Bot 4 (NOVO — com hard gates, Bot 2 DNA, shadow scoring, trades)
☐ SOL posição aberta destacada (entry $88.23)
☐ AI Analyst (inalterado)
☐ System Health (inalterado)
☐ Model Health (inalterado)
☐ Adaptive Weights (inalterado)
☐ Fed Sentinel ENRIQUECIDO com cenários (corte/manutenção)
☐ News (ajustada para englobar Fed Sentinel)
☐ Derivativos REMOVIDO
☐ Macro REMOVIDO
☐ Paper Trading duplicados REMOVIDOS
```

## Helpers técnicos

```python
# Helper exemplo para load SOL state
def load_portfolio_state_sol():
    path = "data/05_output/portfolio_state_sol.json"
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        return {
            "has_position": False,
            "current_capital": 10000,
            "bot_name": "sol_bot4",
        }

# Helper para get latest price
def get_latest_price(asset):
    path_map = {
        "btc": "data/01_raw/spot/btc_1h.parquet",
        "eth": "data/01_raw/spot/eth_1h.parquet",
        "sol": "data/01_raw/spot/sol_1h.parquet",
    }
    try:
        df = pd.read_parquet(path_map[asset])
        return float(df["close"].iloc[-1])
    except Exception:
        return None

# Helper para load feature
@st.cache_data(ttl=300)
def load_feature_value(asset, feature_name, source="current"):
    """Load most recent value of a feature."""
    # Logic to read from appropriate parquet
    pass
```

## Tempo estimado

- Leitura código: 15 min
- ETH Bot 3 section: 20 min
- SOL Bot 4 section: 30 min (mais complexo com hard gates + Bot 2 DNA + shadow)
- Fed Sentinel enrichment: 10 min
- Remover duplicados: 10 min
- Teste + ajustes: 15 min
- **Total: ~1h40min**

## O que NÃO fazer

- ❌ Modificar DeepSeek (AI Analyst) — próxima iteração
- ❌ Modificar System Health, Model Health, Adaptive Weights
- ❌ Refactor lógica backend dos bots
- ❌ Deploy AWS automaticamente
- ❌ Criar páginas separadas (continua single page)

## Estilo visual

**Consistência por asset:**
- BTC (padrão): 🛡️ e 🚀 (Bot 1 e Bot 2)
- ETH: ⚡ (raio, energia, Ethereum)
- SOL: 🟣 (roxo, cor clássica Solana)

**Posição aberta:** sempre destacar com `st.success()` ou container verde

**Cards de filtros:** mesmo padrão do Bot 2 (emoji + nome + badge + valor + threshold)

## Commit sugerido

```
feat(dashboard): multi-asset consolidation — ETH, SOL, cleanup

- Add ETH Bot 3 section (volume Q2 filters + trades)
- Add SOL Bot 4 section (hard gates + Bot 2 DNA + shadow scoring)
- Highlight SOL first position ($88.23 paper)
- Enrich Fed Sentinel with cut/hold/hike scenarios
- Remove duplicate sections: Derivativos, Macro, Paper Trading
- Single consolidated view of all 3 assets
```

---

## Notas importantes

**Sobre SOL primeira posição:**
O trade aberto hoje (22/04, 16:15 UTC, $88.23) deve aparecer DESTACADO na seção SOL. Usar `st.success()` com detalhes completos (entry, current P&L, trail, TP, max hold).

**Sobre ETH zero trades:**
Bot 3 ETH pode não ter trades ainda (tava aguardando setup). Mostrar "Nenhum trade completado" sem alarme.

**Sobre performance:**
Múltiplos parquets sendo carregados. Usar `@st.cache_data(ttl=300)` nos helpers de load.

**Sobre dados faltantes:**
Se portfolio_state_eth/sol não têm todos os campos `last_filter_*`, carregar dos parquets de features direto. Consistente com estratégia Opção A da iteração anterior.
