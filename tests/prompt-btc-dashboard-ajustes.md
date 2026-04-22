# Prompt: Ajustes BTC Dashboard — 3 mudanças focadas

## Contexto

Ajustes cirúrgicos no dashboard BTC atual (`src/dashboard/app.py`), sem refactor geral. Três mudanças específicas:

1. **Reposicionar histórico de trades Bot 1**: mover para cima da seção AI Analyst (logo após Gate Scoring v2)
2. **Enriquecer informações de trades**: adicionar SL, TP e Trailing (já tem entry)
3. **Remover temporariamente Whale Tracking**: comentar código, não deletar (voltar no futuro com estatística)

## Contexto técnico

**Arquivo:** `src/dashboard/app.py`

**Estrutura atual (screenshot referência):**
```
1. Header (price, regime, capital)
2. Gate Scoring v2 (Bot 1 gatekeeper)
3. AI Analyst (DeepSeek — botão manual)
4. Whale Tracking (L/S, display only)
5. (seções abaixo: Derivativos, Macro, News, etc)
```

**Estrutura desejada:**
```
1. Header (price, regime, capital) — sem mudança
2. Gate Scoring v2 — sem mudança
3. 🆕 TRADES BOT 1 (novo posicionamento, com SL/TP/Trail completos)
4. AI Analyst — sem mudança
5. ❌ Whale Tracking — comentado (bloco preservado pra futuro)
6. (demais seções) — sem mudança
```

## Instruções detalhadas

### Mudança 1: Reposicionar trades Bot 1

**Problema atual:**
- Histórico de trades Bot 1 está provavelmente no fim do dashboard ou em outra aba
- Usuário quer ver trades PRÓXIMO da lógica dos gates (Gate Scoring v2)
- Racional: ver a decisão do gate + resultado dos trades numa olhada só

**Ação:**
1. Localizar seção de trades Bot 1 atual (procurar por "trades" + "bot1" em app.py)
2. Criar nova seção "📊 Histórico Bot 1 (Reversal)" imediatamente APÓS Gate Scoring v2
3. Mover (ou duplicar com reorganização) a lógica de trades Bot 1 pra esta posição
4. Se já existe histórico misturado Bot 1 + Bot 2, SEPARAR:
   - Bot 1 vai pra cima (após Gate Scoring)
   - Bot 2 continua onde está ou também move pra nova posição

**Código sugerido (após o bloco de Gate Scoring v2):**
```python
# ==========================================
# BOT 1 TRADES HISTORY (Reversal)
# ==========================================
st.markdown("### 📊 Histórico Bot 1 (Reversal)")
st.caption("Trades baseados em Gate Scoring v2 acima")

# Load trades
trades_bot1 = load_trades_filtered(bot="bot1")  # função a criar ou usar existente

if trades_bot1.empty:
    st.info("Sem trades do Bot 1 registrados ainda")
else:
    # Mostrar em formato estruturado (ver Mudança 2)
    render_trades_table(trades_bot1, include_stops=True)
```

### Mudança 2: Enriquecer informações dos trades

**Informações ATUAIS que já aparecem em trades:**
- ✓ Entrada (entry_price, entry_time)
- ✓ Saída (exit_price, exit_time, exit_reason)
- ✓ Return %
- ✓ Features entry (stablecoin_z, RSI, BB para Bot 2)

**Informações FALTANTES que devem ser adicionadas:**
- ❌ Stop Loss price (configurado no entry)
- ❌ Take Profit price (configurado no entry)
- ❌ Trailing Stop (se ativo, qual foi o trailing high?)

**Onde buscar essas informações:**
- `portfolio_state.json`: contém SL/TP de posição ATIVA
- `data/05_output/trades_*.jsonl` ou similar: histórico de trades finalizados
- Se trades finalizados NÃO têm SL/TP salvos, precisa verificar código em `src/trading/paper_trader.py` e adicionar ao dict de trade:
  ```python
  trade_record = {
      ... (campos existentes)
      "stop_loss_price": state["stop_loss_price"],
      "take_profit_price": state["take_profit_price"],
      "trailing_high": state.get("trailing_high"),
      "trailing_stop_final": state.get("trailing_stop_price"),
  }
  ```

**Layout sugerido da tabela enriquecida:**

```
| Entry Time | Entry $ | SL $ | TP $ | Trail High $ | Exit $ | Exit Reason | Return % | Features |
|------------|---------|------|------|--------------|--------|-------------|----------|----------|
| 4/22 02:05 | 76,291  | 75,146 | 77,817 | 77,940 | 77,940 | TP | +2.16% | stb=2.47 RSI=54 |
| 4/22 07:05 | 78,004  | 76,834 | 79,564 | 79,236 | (aberto) | — | +0.95% | stb=2.26 RSI=78 |
```

**Tratamento especial para posição aberta:**
- Se `has_position=True`, mostrar linha com Exit = "POSIÇÃO ABERTA" (verde/destaque)
- Current price vs entry mostra P&L não-realizado
- Trailing stop atualizado dinamicamente

**Código sugerido (helper function):**
```python
def render_trades_table(trades_df, include_stops=True):
    """Render tabela de trades com SL/TP/Trail."""
    if trades_df.empty:
        st.info("Sem trades")
        return
    
    # Columns a mostrar
    cols = ["entry_time", "entry_price"]
    if include_stops:
        cols.extend(["stop_loss_price", "take_profit_price", "trailing_high"])
    cols.extend(["exit_price", "exit_reason", "return_pct"])
    
    # Formatar valores
    display_df = trades_df[cols].copy()
    
    # Price columns com $
    for col in ["entry_price", "stop_loss_price", "take_profit_price",
                "trailing_high", "exit_price"]:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(
                lambda x: f"${x:,.2f}" if pd.notna(x) else "—"
            )
    
    # Return com cor
    if "return_pct" in display_df.columns:
        display_df["return_pct"] = display_df["return_pct"].apply(
            lambda x: f"{x:+.2f}%" if pd.notna(x) else "—"
        )
    
    # Exit reason com emoji
    if "exit_reason" in display_df.columns:
        emoji_map = {
            "TP": "🎯 TP",
            "SL": "🛑 SL",
            "TRAILING": "📉 Trail",
            "MAX_HOLD": "⏰ Max Hold",
            "OI_EARLY_EXIT": "🚨 OI Exit",
        }
        display_df["exit_reason"] = display_df["exit_reason"].map(
            lambda x: emoji_map.get(x, x) if pd.notna(x) else "—"
        )
    
    # Render
    st.dataframe(display_df, use_container_width=True, hide_index=True)
```

### Mudança 3: Remover Whale Tracking (comentado)

**Problema:**
- Whale Tracking (L/S ratio, top accounts) é display-only
- Não alimenta decisão de nenhum bot
- Ocupa espaço visual sem agregar valor atual
- Mas queremos PRESERVAR o código pra voltar no futuro

**Ação:**
1. Localizar seção Whale Tracking em `src/dashboard/app.py`
2. Comentar bloco inteiro usando docstring `"""..."""` ou `# -` ao invés de deletar
3. Adicionar marcador claro de TODO

**Código sugerido:**

```python
# ==========================================
# 🐋 WHALE TRACKING — DISABLED (temp)
# ==========================================
# TODO (futuro): reativar quando tivermos validação estatística
# do valor preditivo do L/S ratio.
# 
# Análise a fazer antes de reativar:
#   - Correlação L/S ratio vs forward returns (SOL EDA style)
#   - Cohen's d em shocks ±2σ
#   - Comparar com outras features
#   - Se p<0.05 e |corr|>0.1 → considerar adicionar ao Bot 1 ou Bot 4
#
# Código preservado abaixo (não deletar — reativar quando estatística justificar):

"""
st.markdown("---")
st.markdown("## 🐋 Whale Tracking")

# (todo o código original aqui, dentro da docstring)
# Whale signal, L/S ratio, top accounts, gráfico, etc.
"""
```

**Ou alternativa com flag:**
```python
ENABLE_WHALE_TRACKING = False  # TODO: reativar após validação estatística

if ENABLE_WHALE_TRACKING:
    # ... (código original)
```

**Escolher opção mais limpa para este projeto.**

## Requisitos técnicos

### Compatibilidade
- ✅ Não quebrar funcionamento atual
- ✅ Rodar Streamlit local sem erro após mudança
- ✅ Nenhum dado deletado (Whale preservado comentado)

### Performance
- ✅ `load_trades_filtered()` deve usar cache Streamlit se possível (@st.cache_data)
- ✅ Não adicionar queries extras se dados já carregados em memória

### UX
- ✅ Tabela de trades deve ter fonte legível (números alinhados à direita)
- ✅ Posição aberta DESTACADA (cor diferente ou ícone)
- ✅ Exit reasons com emojis (facilita scanning visual)

## Validação pós-mudança

Rodar localmente e verificar:

```bash
cd /Users/brown/Documents/MLGeral/btc_AI

# Verificar sintaxe Python
python -c "import ast; ast.parse(open('src/dashboard/app.py').read())"

# Rodar dashboard local
streamlit run src/dashboard/app.py

# Abrir browser e verificar:
# ☐ Gate Scoring v2 no mesmo lugar
# ☐ Histórico Bot 1 APARECE após Gate Scoring
# ☐ Trades mostram SL/TP/Trailing (colunas novas)
# ☐ Posição aberta destacada (se houver)
# ☐ AI Analyst abaixo dos trades Bot 1
# ☐ Whale Tracking NÃO APARECE
# ☐ Resto do dashboard funciona igual
```

## Tarefas em ordem

1. **Ler `src/dashboard/app.py`** completo pra entender estrutura
2. **Identificar linhas** de:
   - Gate Scoring v2 (final)
   - AI Analyst (início)
   - Whale Tracking (início/fim)
   - Seção atual de trades Bot 1 (se existir)
3. **Verificar `paper_trader.py`** se SL/TP são persistidos em trades
4. **Aplicar mudanças** (refactor localmente, não deploy)
5. **Testar** Streamlit local
6. **Commit** com mensagem clara:
   ```
   feat(dashboard): reorganizar BTC com trades Bot 1 após gates
   
   - Mover histórico Bot 1 pra acima do AI Analyst
   - Enriquecer trades com SL/TP/Trailing stops
   - Desabilitar Whale Tracking temporariamente (preservado comentado)
   ```

## Não fazer nesse prompt

- ❌ Refactor completo do dashboard
- ❌ Criar páginas separadas (pages/)
- ❌ Mover Bot 2 (continua onde está, ou em outra iteração)
- ❌ Mudar lógica de scoring
- ❌ Deploy AWS automaticamente (só local test)

## Esperado ao final

- `src/dashboard/app.py` modificado
- Streamlit rodando local
- Screenshot ou descrição do novo layout
- Commit pendente (você decide se faz push)

## Tempo estimado

- Leitura código: 10 min
- Mudança 1 (mover trades): 15 min
- Mudança 2 (SL/TP/Trail): 20 min (pode precisar mexer em paper_trader.py)
- Mudança 3 (comentar Whale): 5 min
- Teste local: 10 min
- Total: ~1h

## Nota sobre SL/TP em trades históricos

Se `paper_trader.py` NÃO salva SL/TP/Trail atualmente:

**Opção A (preferida):** adicionar campos no dict de trade quando close_position é chamado
```python
trade_record["stop_loss_price"] = state["stop_loss_price"]
trade_record["take_profit_price"] = state["take_profit_price"]
trade_record["trailing_high"] = state["trailing_high"]
```

**Opção B (fallback):** calcular retroativamente com entry_price × percentuais fixos
```python
# Se não tem stop_loss_price gravado:
sl_calc = entry_price * (1 - 0.015)  # -1.5%
tp_calc = entry_price * (1 + 0.020)  # +2%
# Mostrar com asterisco "*calculado"
```

Escolher Opção A e gravar em trades NOVOS. Histórico antigo fica sem SL/TP (mostra "—") ou usa fallback com aviso "*".
