# Prompt: Finalização Dashboard — Compactar Bot 1 Monitoring + Performance Bots 2/3/4

## Contexto

Última iteração do dashboard antes do deploy AWS. Estado atual:

```
1. Header (price, regime, capital)
2. 🛡️ BOT 1 - Conservador (Gate Scoring)
3. 📊 Trades Bot 1
4. 🚀 BOT 2 - Momentum (filtros + trades)
5. ⚡ ETH - Bot 3 (iteração anterior)
6. 🟣 SOL - Bot 4 (iteração anterior)
7. 🤖 AI Analyst (DeepSeek) — inalterado
8. ⚙️ System Health
9. 📐 Model Health     ← MUDAR (bot 1 only, compactar)
10. 📊 Calibration Alerts  ← MUDAR (bot 1 only, compactar)
11. ⚖️ Adaptive Weights    ← MUDAR (bot 1 only, compactar)
12. 📰 News + Fed Sentinel
```

**Problema atual:** Seções 9, 10, 11 aparecem como **gerais/sistema** mas são exclusivas do Bot 1. Confunde leitura.

**Solução:** deixar escopo claro + compactar + adicionar Performance simples pra Bots 2/3/4.

## Objetivo

**Duas mudanças:**

1. **Compactar Model Health + Calibration Alerts + Adaptive Weights** (Bot 1 only)
   - Consolidar em UMA seção claramente marcada "BOT 1 - Monitoramento"
   - Cards resumo visíveis, detalhes em expander colapsado
   - Reduzir 3 seções longas em 1 compacta

2. **Adicionar Performance Monitoring** para Bots 2/3/4
   - Seção nova após Bot 1 Monitoring
   - Indicadores simples de saúde (WR, PF, Sharpe quando possível)
   - Alertas se degradação detectada

## Mudança 1: Compactar Bot 1 Monitoring

### Estrutura atual (expandida, 3 seções)

```
📐 Model Health
  Alignment: 0.244
  Distribuição: 4 saudável / 2 atenção / 3 desalinhado
  Extremos: G10 Funding (0.054) / G4 OI (0.446)
  Média simples: 0.233
  Leitura: "Alguns gates começando a divergir..."
  
📊 Calibration Alerts (rolling 30d corr vs parameters.yml)
  🔴 G4 OI: corr_cfg=-0.472 corr_30d=-0.026 Δ=0.446 (n=30)
  🔴 G6 Bubble: ...
  🔴 G8 F&G: ...
  🔴 G5 Stablecoin: ...
  ⚠️ G3 Curve: ...
  ✅ G9 Taker: ...
  ✅ G7 ETF: ...
  ✅ G3 DGS10: ...
  ✅ G10 Funding: ...

⚖️ Adaptive Weights
  Global Confidence Multiplier: ×0.617
  Mean Confidence: 0.56
  OK: 2 | Reduced: 4 | Severe: 0 | Extreme: 3
  Detalhe por Gate (9 gates com base/eff/conf/delta)
```

### Estrutura desejada (compacta, 1 seção)

```
🛡️ BOT 1 - Monitoramento (Gate Scoring Health)

┌──────────────────┬──────────────────┬──────────────────┐
│ 🎯 ALIGNMENT     │ ⚖️ GLOBAL CONF   │ 💀 KILL SWITCH  │
│                  │                  │                  │
│ 0.244            │ ×0.617           │ 3 gates mortos  │
│ 🟡 Atenção       │ 🟡 Reduzido -38% │ 🔴 Taker/Funding│
│                  │                  │   /Curve        │
└──────────────────┴──────────────────┴──────────────────┘

[ ℹ️ Detalhes completos (expandir) ]
  └── Ao clicar: aparece
       - Gates status (9 cards)
       - Calibration alerts table
       - Detailed por gate table
```

### Implementação

```python
# ========================================
# 🛡️ BOT 1 - MONITORAMENTO
# ========================================
st.markdown("---")
st.markdown("## 🛡️ BOT 1 - Monitoramento")
st.caption("Saúde estatística do Gate Scoring v2 — auditoria de calibração e kill switches")

# Dados (já computados em código existente, só reorganizar)
# Assumir variáveis: model_alignment, gconf_mult, kill_count, kill_gates_list

# Cards resumo (sempre visíveis)
col1, col2, col3 = st.columns(3)

with col1:
    alignment = model_alignment  # já computado
    if alignment >= 0.6:
        badge, status = "🟢", "Saudável"
    elif alignment >= 0.3:
        badge, status = "🟡", "Atenção"
    else:
        badge, status = "🔴", "Desalinhado"
    
    st.metric(
        label="🎯 MODEL ALIGNMENT",
        value=f"{alignment:.3f}",
        delta=f"{badge} {status}",
        delta_color="off"
    )

with col2:
    reduction = (1 - gconf_mult) * 100
    if gconf_mult >= 0.8:
        badge = "🟢 OK"
    elif gconf_mult >= 0.6:
        badge = "🟡 Reduzido"
    else:
        badge = "🔴 Severo"
    
    st.metric(
        label="⚖️ GLOBAL CONFIDENCE",
        value=f"×{gconf_mult:.3f}",
        delta=f"{badge} -{reduction:.1f}%",
        delta_color="off"
    )

with col3:
    kill_count = sum(1 for g in adaptive_details.values() 
                     if isinstance(g, dict) and g.get('kill_status') == 'extreme')
    kill_gates = [name for name, info in adaptive_details.items() 
                  if isinstance(info, dict) and info.get('kill_status') == 'extreme']
    
    if kill_count == 0:
        badge, color_text = "🟢 Nenhum", ""
    elif kill_count <= 2:
        badge = "🟡 Alguns"
        color_text = ", ".join(kill_gates[:3])
    else:
        badge = "🔴 Múltiplos"
        color_text = ", ".join(kill_gates[:3])
    
    st.metric(
        label="💀 KILL SWITCHES",
        value=f"{kill_count} gates",
        delta=f"{badge}: {color_text}" if color_text else badge,
        delta_color="off"
    )

# Leitura interpretativa (uma linha)
if alignment < 0.3 or gconf_mult < 0.5:
    st.warning("⚠️ Bot 1 operando com capacidade reduzida. Considerar recalibrar gates em 2-4 semanas.")
elif alignment < 0.6 or gconf_mult < 0.8:
    st.info("ℹ️ Bot 1 operando com leve degradação. Monitorar evolução.")
else:
    st.success("✅ Bot 1 saudável. Gates alinhados com correlações configuradas.")

# Detalhes completos (colapsado)
with st.expander("🔍 Detalhes completos (Calibration + Adaptive por gate)", expanded=False):
    
    # TAB 1: Calibration Alerts (tabela)
    st.markdown("### 📊 Calibration Alerts (rolling 30d)")
    st.caption("Divergência entre correlação configurada e observada")
    
    calibration_data = []
    for gate_name, info in adaptive_details.items():
        if isinstance(info, dict):
            cfg = info.get('corr_cfg', 0)
            real = info.get('corr_30d', 0)
            delta = abs(cfg - real)
            
            if delta >= 0.35:
                status = "🔴"
            elif delta >= 0.20:
                status = "⚠️"
            else:
                status = "✅"
            
            calibration_data.append({
                "Status": status,
                "Gate": gate_name,
                "corr_cfg": f"{cfg:+.3f}",
                "corr_30d": f"{real:+.3f}",
                "Δ": f"{delta:.3f}",
                "n": info.get('n', 30),
            })
    
    if calibration_data:
        cal_df = pd.DataFrame(calibration_data)
        st.dataframe(cal_df, use_container_width=True, hide_index=True)
    
    # TAB 2: Adaptive Details (tabela)
    st.markdown("### ⚖️ Adaptive Weights Details")
    st.caption("Pesos efetivos após confidence weighting + kill switch")
    
    adaptive_data = []
    for gate_name, info in adaptive_details.items():
        if isinstance(info, dict):
            kill_status = info.get('kill_status', 'ok')
            if kill_status == 'extreme':
                status = "🔴"
            elif kill_status == 'severe':
                status = "⚠️"
            elif info.get('confidence', 1.0) < 0.8:
                status = "🟡"
            else:
                status = "✅"
            
            adaptive_data.append({
                "Status": status,
                "Gate": gate_name,
                "Base": f"{info.get('base_weight', 0):.2f}",
                "Eff": f"{info.get('effective_weight', 0):.2f}",
                "Conf": f"{info.get('confidence', 0):.2f}",
                "Δ": f"{abs(info.get('corr_cfg', 0) - info.get('corr_30d', 0)):.3f}",
            })
    
    if adaptive_data:
        adp_df = pd.DataFrame(adaptive_data)
        st.dataframe(adp_df, use_container_width=True, hide_index=True)
    
    # Resumo adaptive
    n_ok = sum(1 for d in adaptive_data if d["Status"] == "✅")
    n_reduced = sum(1 for d in adaptive_data if d["Status"] == "🟡")
    n_severe = sum(1 for d in adaptive_data if d["Status"] == "⚠️")
    n_extreme = sum(1 for d in adaptive_data if d["Status"] == "🔴")
    
    cols = st.columns(4)
    cols[0].metric("OK", n_ok)
    cols[1].metric("Reduced", n_reduced, delta="conf<0.8")
    cols[2].metric("Severe", n_severe, delta="weight ×0.3")
    cols[3].metric("Extreme", n_extreme, delta="weight=0")
```

### Importante

- **NÃO DELETAR** as seções antigas (Model Health, Calibration Alerts, Adaptive Weights) antes de criar a nova
- Consolidar em UM único bloco
- Remover as 3 seções antigas APÓS testar que a nova funciona

## Mudança 2: Performance Monitoring (Bots 2/3/4)

### Filosofia

Métricas diferentes das do Bot 1. Bots 2/3/4 usam filtros binários, então não têm "calibration" matemática. Em vez disso, medimos **saúde de performance**:

- Está ganhando? (WR)
- Quanto ganha vs perde? (PF)
- Com que consistência? (Sharpe rolling quando N ≥ 20)
- Está degradando? (comparar últimos 10 vs últimos 30)

### Estrutura

```
📊 Performance Monitoring (Bots 2/3/4)

Para cada bot:

┌─────────────────────────────────────────────────────┐
│ 🚀 BOT 2 - Momentum (BTC)                           │
│                                                      │
│ Trades: 2   WR: 100%   PF: ∞   Sharpe: (n<20)       │
│ Últimos 30d: +18.48%   Max DD: 0%                   │
│ 🟢 Saudável — performance alinhada com expectativa   │
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│ ⚡ BOT 3 - Volume Defensivo (ETH)                   │
│                                                      │
│ Trades: 0   Aguardando primeiro setup               │
│ ⏳ Insuficiente — precisa de trades para análise    │
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│ 🟣 BOT 4 - Taker/Flow (SOL)                         │
│                                                      │
│ Trades: 0 completos (1 aberto)                      │
│ Posição ativa: $88.23 @ 16:15 UTC                   │
│ ⏳ Aguardando primeiro fechamento                    │
└─────────────────────────────────────────────────────┘
```

### Indicadores simples sugeridos

Para cada bot:

**Se N ≥ 1 trade:**
- Trades count
- Win Rate (%)
- Avg Return per trade
- Total Return accumulated

**Se N ≥ 10:**
- Profit Factor (sum wins / sum losses)
- Max Drawdown
- Last 10 vs previous 10 trend (se aplicável)

**Se N ≥ 20:**
- Sharpe rolling (anualizado)
- Comparison vs backtest expectation

**Alertas:**
```
Condições de ALERT 🔴:
  - Sharpe live < 50% do Sharpe backtest
  - WR < 40% em 10+ trades
  - DD > 10%
  - 5+ losses consecutivos

Condições de ATENÇÃO 🟡:
  - Sharpe live entre 50-80% do backtest
  - WR entre 40-50%
  - DD entre 5-10%

Condições OK 🟢:
  - Todas métricas acima dos thresholds

Condições AGUARDANDO ⏳:
  - N < 10 trades
```

### Implementação

```python
# ========================================
# 📊 PERFORMANCE MONITORING (Bots 2/3/4)
# ========================================
st.markdown("---")
st.markdown("## 📊 Performance Monitoring")
st.caption("Saúde de performance dos bots momentum-based (Bot 1 tem monitoring próprio acima)")

# Expected Sharpe por bot (do backtest)
EXPECTED_SHARPE = {
    "bot2": 2.71,  # BTC Bot 2 (live atual)
    "bot3": 0.64,  # ETH Bot 3 (backtest Phase 0.5)
    "bot4": 2.03,  # SOL Bot 4 (backtest Phase 2)
}

def compute_bot_health(trades_df, bot_name, expected_sharpe=None):
    """Compute health status for a bot."""
    if trades_df.empty:
        return {
            "status": "⏳",
            "label": "Sem trades",
            "message": "Aguardando primeiro trade",
            "n_trades": 0,
        }
    
    n = len(trades_df)
    
    if n < 10:
        # Pouco dados, só mostrar absolutos
        wr = (trades_df["return_pct"] > 0).sum() / n * 100
        avg_ret = trades_df["return_pct"].mean()
        total_ret = ((1 + trades_df["return_pct"]/100).prod() - 1) * 100
        
        return {
            "status": "⏳",
            "label": f"Início de histórico (n={n})",
            "message": f"Precisa de {10-n} trades para análise completa",
            "n_trades": n,
            "win_rate": wr,
            "avg_return": avg_ret,
            "total_return": total_ret,
        }
    
    # N >= 10: análise completa
    wins = trades_df[trades_df["return_pct"] > 0]
    losses = trades_df[trades_df["return_pct"] <= 0]
    
    wr = len(wins) / n * 100
    pf = wins["return_pct"].sum() / abs(losses["return_pct"].sum()) if len(losses) > 0 else float('inf')
    
    # Compute DD
    returns = trades_df["return_pct"] / 100
    equity = (1 + returns).cumprod()
    dd = (equity / equity.cummax() - 1).min() * 100
    
    # Sharpe (se n >= 20)
    sharpe = None
    if n >= 20:
        # Simplificado: Sharpe annual assuming 1 trade per day avg
        sharpe = (returns.mean() / returns.std()) * (252 ** 0.5)
    
    # Classify health
    if wr < 40 or (sharpe and sharpe < 0):
        status = "🔴"
        label = "Crítico"
    elif wr < 50 or dd < -10 or (sharpe and expected_sharpe and sharpe < expected_sharpe * 0.5):
        status = "🟡"
        label = "Atenção"
    else:
        status = "🟢"
        label = "Saudável"
    
    # Custom message
    msg_parts = []
    if sharpe and expected_sharpe:
        ratio = sharpe / expected_sharpe * 100
        msg_parts.append(f"Sharpe {sharpe:.2f} ({ratio:.0f}% do backtest)")
    
    if dd < -5:
        msg_parts.append(f"DD {dd:.1f}%")
    
    message = " | ".join(msg_parts) if msg_parts else "Performance alinhada com expectativa"
    
    return {
        "status": status,
        "label": label,
        "message": message,
        "n_trades": n,
        "win_rate": wr,
        "profit_factor": pf,
        "max_dd": dd,
        "sharpe": sharpe,
        "total_return": ((1 + returns).prod() - 1) * 100,
    }


# Render cada bot
for bot_info in [
    {"key": "bot2", "emoji": "🚀", "name": "BOT 2 - Momentum (BTC)", "asset": "btc"},
    {"key": "bot3", "emoji": "⚡", "name": "BOT 3 - Volume Defensivo (ETH)", "asset": "eth"},
    {"key": "bot4", "emoji": "🟣", "name": "BOT 4 - Taker/Flow (SOL)", "asset": "sol"},
]:
    trades = load_trades_filtered(asset=bot_info["asset"], bot=bot_info["key"])
    health = compute_bot_health(
        trades,
        bot_info["key"],
        expected_sharpe=EXPECTED_SHARPE.get(bot_info["key"])
    )
    
    with st.container():
        st.markdown(f"### {bot_info['emoji']} {bot_info['name']}")
        
        if health["n_trades"] == 0:
            # Special case: SOL has open position
            if bot_info["key"] == "bot4":
                sol_state = load_portfolio_state_sol()
                if sol_state.get("has_position"):
                    st.info(
                        f"🟡 Posição aberta: ${sol_state['entry_price']:.2f} @ "
                        f"{sol_state.get('entry_time', 'N/A')} | "
                        f"Aguardando primeiro fechamento para métricas"
                    )
                else:
                    st.info("⏳ Aguardando primeiro trade")
            else:
                st.info(f"⏳ {health['message']}")
        else:
            # Mostrar métricas
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Trades", health["n_trades"])
            col2.metric("Win Rate", f"{health['win_rate']:.1f}%")
            
            if "profit_factor" in health:
                pf_display = f"{health['profit_factor']:.2f}" if health['profit_factor'] != float('inf') else "∞"
                col3.metric("Profit Factor", pf_display)
            else:
                col3.metric("Avg Return", f"{health.get('avg_return', 0):+.2f}%")
            
            if "sharpe" in health and health["sharpe"]:
                col4.metric("Sharpe (annual)", f"{health['sharpe']:.2f}")
            else:
                col4.metric("Total Return", f"{health.get('total_return', 0):+.2f}%")
            
            # Status + message
            if health["status"] == "🔴":
                st.error(f"{health['status']} {health['label']} — {health['message']}")
            elif health["status"] == "🟡":
                st.warning(f"{health['status']} {health['label']} — {health['message']}")
            elif health["status"] == "🟢":
                st.success(f"{health['status']} {health['label']} — {health['message']}")
            else:
                st.info(f"{health['status']} {health['label']} — {health['message']}")
```

## Estrutura final desejada

```
1. Header
2. 🛡️ BOT 1 - Conservador (Gate Scoring)
3. 📊 Trades Bot 1
4. 🚀 BOT 2 - Momentum
5. ⚡ ETH - Bot 3
6. 🟣 SOL - Bot 4
7. 🤖 AI Analyst (DeepSeek) — inalterado
8. ⚙️ System Health
9. 🆕 🛡️ BOT 1 - Monitoramento (consolidado, Model + Calibration + Adaptive)
10. 🆕 📊 Performance Monitoring (Bots 2/3/4 — simples)
11. 📰 News + Fed Sentinel
```

## Tarefas Claude Code

1. **Ler** `src/dashboard/app.py` — localizar:
   - Model Health section
   - Calibration Alerts section  
   - Adaptive Weights section
   - Helpers existentes (load_trades_filtered, etc)

2. **Consolidar** Model + Calibration + Adaptive em UMA seção "Bot 1 - Monitoramento":
   - 3 cards resumo no topo
   - Expander com detalhes completos
   - Leitura interpretativa (uma linha)

3. **Adicionar** Performance Monitoring section:
   - Helper function `compute_bot_health()`
   - Cards por bot (Bot 2, Bot 3, Bot 4)
   - Alertas automáticos baseados em WR/Sharpe/DD
   - Handle de casos especiais (SOL posição aberta, ETH sem trades)

4. **Remover** as 3 seções antigas (após testar nova funciona)

5. **Testar local:**
   ```bash
   streamlit run src/dashboard/app.py
   ```

## Validação

**Checklist após mudança:**

```
☐ Seção antiga Model Health REMOVIDA
☐ Seção antiga Calibration Alerts REMOVIDA
☐ Seção antiga Adaptive Weights REMOVIDA
☐ Nova "🛡️ BOT 1 - Monitoramento" aparece
☐ 3 cards resumo: Alignment + GConf + Kill Switches
☐ Expander com detalhes funciona
☐ Tabelas Calibration + Adaptive dentro do expander
☐ Nova "📊 Performance Monitoring" aparece
☐ Bot 2: mostra métricas com 2 trades atuais
☐ Bot 3: mostra "Aguardando primeiro trade"
☐ Bot 4: mostra "Posição aberta" com detalhes
☐ Resto do dashboard inalterado
☐ Streamlit roda sem erro
```

## Deploy process (após validação local)

```bash
# === LOCAL ===
cd /Users/brown/Documents/MLGeral/btc_AI
streamlit run src/dashboard/app.py  # testar UMA ÚLTIMA vez

# Quando OK:
git status
git add src/dashboard/app.py
git commit -m "feat(dashboard): compact Bot 1 monitoring + performance for Bots 2/3/4

- Consolidate Model Health + Calibration + Adaptive into single Bot 1 section
- Summary cards + expander for details
- Add Performance Monitoring for Bots 2/3/4 (WR, PF, Sharpe, DD)
- Health alerts based on live vs backtest comparison
- Handle SOL open position and ETH zero trades gracefully"

git push origin main

# === AWS ===
ssh -i ~/.ssh/aihab-key-sp.pem ubuntu@54.232.162.161
cd ~/AIhab
git pull origin main

# Ver se dashboard roda em container separado ou junto
docker compose ps

# Restart do serviço correto:
docker compose restart <nome_do_servico_dashboard>
# OU
docker compose up -d --force-recreate --no-deps <service>

# Verificar no browser (provavelmente :8501)
```

## O que NÃO fazer

- ❌ Modificar DeepSeek (próxima iteração separada)
- ❌ Deletar seções antigas ANTES de criar as novas (risco)
- ❌ Deploy AUTOMÁTICO sem validação local
- ❌ Mexer em Paper Trader (`src/trading/paper_trader.py`)

## Tempo estimado

- Leitura + planejamento: 10 min
- Compactação Bot 1 Monitoring: 30 min
- Performance Monitoring: 30 min
- Teste local: 15 min
- **Total: ~1h25min**
