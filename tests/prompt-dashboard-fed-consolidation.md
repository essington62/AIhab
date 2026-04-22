# Prompt: Consolidar Fed Sentinel + Fed Observatory (último ajuste pré-deploy)

## Contexto

Último ajuste do dashboard BTC antes do deploy AWS.

**Estado atual (screenshot referência):**
```
📰 News & Sentiment
  ├── Scores DeepSeek
  ├── Notícias Recentes
  ├── 🙂 Fear & Greed (card lateral)
  └── 🏛️ Fed Sentinel (card lateral)
        Próx evento: chair_term_end
        Em: 23 dias
        Proximity adj: +0.0
        Blackout: Não

🏛️ Fed Observatory  ← SEÇÃO SEPARADA
  Probabilidades estimadas (proxy DGS2, não FedWatch)
  ├── Corte 25bps - 35%
  ├── Manutenção - 65%
  ├── Alta 25bps - 0%
  ├── DGS2: 3.72%
  ├── EFFR: 3.64%
  ├── Spread vs Fed: +10bps
  ├── Inflation 5Y: 2.56%
  ├── Inflation 10Y: 2.38%
  └── Cenários por decisão
        Corte 25bps - 35% | Impacto BTC: Bullish | Ação: ...
        Manutenção - 65% | Impacto BTC: Neutro | Ação: ...
```

**Problema:**
- Fed Sentinel (card simples) + Fed Observatory (seção gigante) são **redundantes**
- Ambos tratam do mesmo tema (Fed policy)
- Fed Sentinel mostra APENAS o básico; Fed Observatory tem os detalhes
- Ocupa 2 seções quando poderia ser 1 consolidada

## Objetivo

Consolidar em **UMA seção única** dentro do card Fed Sentinel:

```
🏛️ FED SENTINEL (expandido)
├── Status atual (já existe)
│   Próx evento, dias, proximity adj, blackout
├── 🆕 Probabilidades (do Fed Observatory)
│   Corte 25bps / Manutenção / Alta
├── 🆕 Indicadores (do Fed Observatory)
│   DGS2, EFFR, Spread, Inflation 5Y/10Y
└── 🆕 Cenários por decisão (do Fed Observatory)
    Textos detalhados por cenário
```

**REMOVER completamente** a seção `🏛️ Fed Observatory` após consolidação.

## Instruções detalhadas

### Onde está cada coisa no código

Buscar em `src/dashboard/app.py`:

```bash
grep -n "Fed Sentinel\|fed_sentinel" src/dashboard/app.py
grep -n "Fed Observatory\|fed_observatory\|prob_cut\|prob_hike" src/dashboard/app.py
```

Identificar:
- Linhas onde Fed Sentinel card renderiza (dentro de News & Sentiment)
- Linhas onde Fed Observatory section começa e termina
- Variáveis/dados usados por ambos (podem vir da mesma source)

### Layout consolidado proposto

**Fed Sentinel vira uma seção dedicada** (não mais card lateral de News):

```
🏛️ Fed Sentinel
Probabilidades estimadas (proxy DGS2, não FedWatch)

┌─────────────────┬─────────────────┬─────────────────┐
│ 🟢 Corte 25bps  │ 🟡 Manutenção   │ 🔴 Alta 25bps   │
│ 35%             │ 65%             │ 0%              │
│ BTC: Bullish    │ BTC: Neutro     │ BTC: Bearish    │
└─────────────────┴─────────────────┴─────────────────┘

Próximo evento:
  • chair_term_end em 23 dias
  • Proximity adjustment: +0.0
  • Blackout period: Não

┌──────────┬──────────┬──────────┬──────────┬──────────┐
│ DGS2     │ EFFR     │ SPREAD   │ INFL 5Y  │ INFL 10Y │
│ 3.72%    │ 3.64%    │ +10bps   │ 2.56%    │ 2.38%    │
│ -0.16    │          │          │ -0.10    │          │
└──────────┴──────────┴──────────┴──────────┴──────────┘

Cenários por decisão:

📗 Corte 25bps — 35%
   Impacto BTC: Bullish | Ação: Reduzir threshold, aumentar sizing em Sideways
   
   Historicamente, primeiro corte após ciclo de alta = BTC sobe 30-50% em 6 meses. 
   Dólar enfraquece, capital migra pra risk-on. Stablecoins entram forte.

📘 Manutenção — 65%
   Impacto BTC: Neutro | Ação: Manter scoring atual
   
   Já precificado pelo mercado. Impacto depende do guidance e dot plot. 
   Atenção ao tom da coletiva — hawkish surprise pode derrubar, 
   dovish surprise pode subir.
```

### Implementação

```python
# ========================================
# 🏛️ FED SENTINEL (consolidado com Fed Observatory)
# ========================================
st.markdown("---")
st.markdown("## 🏛️ Fed Sentinel")
st.caption("Probabilidades estimadas (proxy DGS2, não FedWatch)")

# Dados (do fed_observatory existente)
fed_obs = get_fed_observatory_data()  # função já existe
fed_sent = get_fed_sentinel_status()  # função já existe

prob_cut = fed_obs.get("prob_cut", 0)
prob_hold = fed_obs.get("prob_hold", 0)
prob_hike = fed_obs.get("prob_hike", 0)

# ===== Probabilidades (3 cards) =====
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("#### 🟢 Corte 25bps")
    st.metric(
        label="Probabilidade",
        value=f"{prob_cut*100:.0f}%",
        delta="BTC: Bullish",
        delta_color="normal" if prob_cut > 0.3 else "off"
    )

with col2:
    st.markdown("#### 🟡 Manutenção")
    st.metric(
        label="Probabilidade",
        value=f"{prob_hold*100:.0f}%",
        delta="BTC: Neutro",
        delta_color="off"
    )

with col3:
    st.markdown("#### 🔴 Alta 25bps")
    st.metric(
        label="Probabilidade",
        value=f"{prob_hike*100:.0f}%",
        delta="BTC: Bearish",
        delta_color="inverse" if prob_hike > 0.3 else "off"
    )

# ===== Status Sentinel (próximo evento) =====
st.markdown("#### 📅 Próximo evento")
event_col1, event_col2, event_col3 = st.columns(3)

next_event = fed_sent.get("next_event", "N/A")
days_to_event = fed_sent.get("days_to_event", "—")
proximity_adj = fed_sent.get("proximity_adjustment", 0.0)
blackout = fed_sent.get("is_blackout", False)

event_col1.metric("Evento", next_event)
event_col2.metric("Em", f"{days_to_event} dias" if days_to_event != "—" else "—")
event_col3.metric(
    "Proximity adj",
    f"{proximity_adj:+.2f}",
    delta="🔒 Blackout" if blackout else "—",
    delta_color="off"
)

# ===== Indicadores macro (5 cards) =====
st.markdown("#### 📊 Indicadores Macro")

# Dados do FRED e outras sources
dgs2 = fed_obs.get("dgs2")
effr = fed_obs.get("effr")
spread_vs_fed = fed_obs.get("spread_vs_fed_bps", 0)
inflation_5y = fed_obs.get("inflation_5y")
inflation_10y = fed_obs.get("inflation_10y")

# Deltas (mudanças recentes)
dgs2_delta = fed_obs.get("dgs2_delta", 0)
inflation_5y_delta = fed_obs.get("inflation_5y_delta", 0)

ind_col1, ind_col2, ind_col3, ind_col4, ind_col5 = st.columns(5)
ind_col1.metric(
    "DGS2",
    f"{dgs2:.2f}%" if dgs2 else "—",
    delta=f"{dgs2_delta:+.2f}" if dgs2_delta else None,
    delta_color="inverse"  # queda de juros = bom pra BTC
)
ind_col2.metric("EFFR", f"{effr:.2f}%" if effr else "—")
ind_col3.metric("Spread vs Fed", f"{spread_vs_fed:+.0f}bps")
ind_col4.metric(
    "Inflation 5Y",
    f"{inflation_5y:.2f}%" if inflation_5y else "—",
    delta=f"{inflation_5y_delta:+.2f}" if inflation_5y_delta else None,
    delta_color="inverse"  # inflação caindo = bom
)
ind_col5.metric("Inflation 10Y", f"{inflation_10y:.2f}%" if inflation_10y else "—")

# ===== Cenários por decisão =====
st.markdown("#### 🎯 Cenários por decisão")

# Cenário 1: Corte 25bps (só se prob > 0)
if prob_cut > 0:
    with st.container(border=True):
        st.markdown(f"**📗 Corte 25bps — {prob_cut*100:.0f}%**")
        st.caption("**Impacto BTC:** 🟢 Bullish | **Ação:** Reduzir threshold, aumentar sizing em Sideways")
        st.markdown(
            "Historicamente, primeiro corte após ciclo de alta = BTC sobe 30-50% em 6 meses. "
            "Dólar enfraquece, capital migra pra risk-on. Stablecoins entram forte."
        )

# Cenário 2: Manutenção (só se prob > 0)
if prob_hold > 0:
    with st.container(border=True):
        st.markdown(f"**📘 Manutenção — {prob_hold*100:.0f}%**")
        st.caption("**Impacto BTC:** 🟡 Neutro | **Ação:** Manter scoring atual")
        st.markdown(
            "Já precificado pelo mercado. Impacto depende do guidance e dot plot. "
            "Atenção ao tom da coletiva — hawkish surprise pode derrubar, "
            "dovish surprise pode subir."
        )

# Cenário 3: Alta 25bps (só se prob > 0)
if prob_hike > 0:
    with st.container(border=True):
        st.markdown(f"**📕 Alta 25bps — {prob_hike*100:.0f}%**")
        st.caption("**Impacto BTC:** 🔴 Bearish | **Ação:** Elevar threshold, reduzir sizing")
        st.markdown(
            "Hike surprise em ciclo de corte = forte repricing. "
            "Dólar fortalece, crypto sell-off típico. "
            "Bot 1 pode ativar BLOCK mais rígido automaticamente."
        )
```

### Remoção da seção Fed Observatory antiga

Localizar a seção inteira `## 🏛️ Fed Observatory` em `src/dashboard/app.py` e **remover completamente**.

A seção provavelmente tem:
- `st.markdown("## 🏛️ Fed Observatory")`
- Cards de probabilidades
- Cards de indicadores (DGS2, EFFR, etc)
- Cenários por decisão

**Tudo isso vai pra dentro do Fed Sentinel consolidado acima.**

### Remoção do card Fed Sentinel pequeno (dentro de News)

Na seção `News & Sentiment`, o card lateral pequeno do Fed Sentinel (próx evento, blackout, etc) DEVE SER REMOVIDO, já que essa info agora aparece dentro da nova seção Fed Sentinel consolidada.

Manter em News & Sentiment:
- ✅ Scores DeepSeek
- ✅ Notícias Recentes
- ✅ Fear & Greed (card lateral)
- ❌ ~~Fed Sentinel card pequeno~~ (removido)

## Estrutura final do dashboard (após esta mudança)

```
1. Header (price, regime, capital)
2. 🛡️ BOT 1 - Conservador
3. 📊 Histórico Bot 1
3b. 🚀 BOT 2 - Momentum
3c. ⚡ ETH - Bot 3
3d. 🟣 SOL - Bot 4
4. 🤖 AI Analyst
5. ⚙️ System Health
9. 🛡️ BOT 1 - Monitoramento
10. 📊 Performance Monitoring
11. 📰 News & Sentiment (sem Fed Sentinel pequeno)
12. 🆕 🏛️ Fed Sentinel (CONSOLIDADO — probabilidades + indicadores + cenários)
```

❌ **REMOVIDO:** 🏛️ Fed Observatory (seção inteira, tudo migrado pro Fed Sentinel)

## Validação local

```bash
cd /Users/brown/Documents/MLGeral/btc_AI

# Sintaxe
python -c "import ast; ast.parse(open('src/dashboard/app.py').read())"

# Run
streamlit run src/dashboard/app.py
```

**Checklist visual:**
```
☐ Seção 🏛️ Fed Observatory NÃO APARECE (removida)
☐ Card pequeno "Fed Sentinel" em News & Sentiment NÃO APARECE
☐ Nova seção 🏛️ Fed Sentinel aparece ao fim do dashboard
☐ 3 cards de probabilidades (Corte/Manutenção/Alta)
☐ Status próximo evento (chair_term_end, dias, blackout)
☐ 5 cards de indicadores (DGS2, EFFR, Spread, Inflation 5Y/10Y)
☐ Cenários com textos detalhados (Corte + Manutenção + Alta se > 0)
☐ Resto do dashboard inalterado
☐ Sem erros no console do Streamlit
```

## Testing extra

**Verificar que não quebrou outras features:**
```
☐ News & Sentiment ainda renderiza (scores DeepSeek + notícias)
☐ Fear & Greed card lateral ainda aparece
☐ AI Analyst ainda funciona (botão Gerar Análise)
☐ System Health não teve mudança
☐ Performance Monitoring não teve mudança
```

## Deploy process (após validação local)

```bash
# === LOCAL ===
cd /Users/brown/Documents/MLGeral/btc_AI

# Última verificação
streamlit run src/dashboard/app.py
# (abrir browser, confirmar tudo OK)

# Commit
git add src/dashboard/app.py
git commit -m "feat(dashboard): consolidate Fed Sentinel + Fed Observatory

- Merge Fed Observatory into Fed Sentinel section
- Single source of Fed-related info (probabilities + indicators + scenarios)
- Remove duplicate small Fed Sentinel card from News & Sentiment
- Cleaner structure: one comprehensive Fed section at the bottom"

# Push
git push origin main

# === AWS ===
ssh -i ~/.ssh/aihab-key-sp.pem ubuntu@54.232.162.161
cd ~/AIhab
git pull origin main

# Restart dashboard
docker compose restart aihab-dashboard

# Verificar
docker compose ps | grep dashboard
docker compose logs --tail 20 aihab-dashboard
```

### Comando consolidado (copy-paste) pra deploy

```bash
# No Mac, após validar local:
cd /Users/brown/Documents/MLGeral/btc_AI && \
git add src/dashboard/app.py && \
git commit -m "feat(dashboard): consolidate Fed Sentinel + Fed Observatory" && \
git push origin main

# No AWS:
ssh -i ~/.ssh/aihab-key-sp.pem ubuntu@54.232.162.161 "cd ~/AIhab && git pull origin main && docker compose restart aihab-dashboard && sleep 3 && docker compose ps | grep dashboard"
```

## O que NÃO fazer

- ❌ Deletar seção Fed Observatory antes de criar Fed Sentinel consolidado
- ❌ Remover funções helper (`get_fed_observatory_data`, etc) — ainda são usadas
- ❌ Modificar outros helpers (fed_sentinel.py, fed_observatory.py em src/)
- ❌ Deploy automaticamente sem validar local primeiro

## Tempo estimado

- Leitura código: 10 min
- Criar nova seção Fed Sentinel consolidada: 25 min
- Remover Fed Observatory antigo + card pequeno: 10 min
- Teste local: 10 min
- Deploy: 5 min
- **Total: ~1h**

## Importante sobre helpers

As funções `get_fed_observatory_data()` e `get_fed_sentinel_status()` em `src/features/fed_observatory.py` e `src/features/fed_sentinel.py` **NÃO devem ser modificadas**. Só o dashboard muda.

Se alguma função não retornar todos os campos esperados, adicionar fallbacks no dashboard:

```python
prob_cut = fed_obs.get("prob_cut", 0)  # default 0 se não existe
dgs2 = fed_obs.get("dgs2")
if dgs2 is None:
    st.caption("⚠️ DGS2 não disponível")
```

## Resultado final esperado

Dashboard BTC consolidado, limpo, sem duplicações Fed-related. **Pronto para produção AWS.**
