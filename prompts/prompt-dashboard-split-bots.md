# Prompt: Separar Dashboard Paper Trading em Bot 1 e Bot 2

## Contexto

O dashboard (`src/dashboard/app.py`) tem uma seção única "Paper Trading" que mistura Bot 1 (Reversal) e Bot 2 (Momentum). O Bot 2 está escondido dentro de um `st.expander`. Queremos separar em **duas seções visuais independentes**, cada uma com seus próprios cards, filtros, equity curve e stats.

**Ambos os bots compartilham o mesmo `portfolio_state.json` e `cycle_log.parquet`** — não há portfolios separados. A diferenciação é feita pelo campo `entry_bot` ("bot1" ou "bot2") presente no portfolio e nos trade records.

**Regra de mutex:** só um bot pode ter posição aberta por vez. O capital é compartilhado.

## O que mudar

### 1. Renomear seção atual para "📋 Paper Trading — Bot 1 (Reversal)"

A seção atual (SECTION 8) passa a ser explicitamente do Bot 1:

```python
st.markdown("### 📋 Paper Trading — Bot 1 (Reversal)")
```

**Manter tudo que já existe nessa seção**, com as seguintes adições:
- No card "Posição": se `entry_bot == "bot2"`, mostrar "Bot 2 tem posição" em vez dos detalhes de entrada
- No card "Sinais ENTER": filtrar apenas sinais `"ENTER"` (não `"ENTER_BOT2"`)
- O Reversal Filter state bar: manter como está
- **Remover** o `st.expander("🚀 Bot 2 — Momentum/Liquidez")` inteiro desta seção — vai pra seção nova

### 2. Criar nova seção "🚀 Paper Trading — Bot 2 (Momentum)"

Adicionar **APÓS** a seção do Bot 1 (após a equity curve e stats do Bot 1), uma seção nova:

```python
st.markdown("---")
st.markdown("### 🚀 Paper Trading — Bot 2 (Momentum)")
```

#### 2a. Cards de status (4 colunas, mesmo estilo visual do Bot 1)

| Card | Conteúdo |
|------|----------|
| Capital | Mesmo capital compartilhado ($capital) |
| Posição | Se `entry_bot == "bot2"`: ABERTA + preço entrada. Se `entry_bot == "bot1"`: "Bot 1 tem posição (mutex)". Se sem posição: "SEM POSIÇÃO" |
| P&L Não Realizado | Se `entry_bot == "bot2"`: calcular P&L. Senão: "—" |
| Sinais ENTER_BOT2 | Contar apenas `signal == "ENTER_BOT2"` no cycle_log |

#### 2b. Momentum Filter status bar (tirar do expander, sempre visível)

Mostrar as 5 condições do Bot 2 em formato compacto (similar ao Reversal Filter bar do Bot 1):

```html
<div class="cg-card" style="padding:8px 16px; font-size:12px;">
  <span style="color:#8b949e;">Momentum Filter: </span>
  <span style="color:{pass_color};">✓ PASS / ✗ FILTERED (reason)</span>
  | Stablecoin Z: <span>{value} ({check})</span>
  | ret_1d: <span>{value} ({check})</span>
  | RSI: <span>{value} ({check})</span>
  | BB%: <span>{value} ({check})</span>
  | >MA21: <span>{value} ({check})</span>
</div>
```

Usar mesma lógica de cores do Reversal Filter:
- Verde (#3fb950) quando condição atendida (✓)
- Vermelho (#f85149) quando não atendida (✗)

Valores e thresholds:
- `stablecoin_z` do portfolio (`last_momentum_stablecoin_z`) ou `zs.get("stablecoin_z")` → precisa ser > 1.3
- `ret_1d` do portfolio (`last_filter_ret_1d`) → precisa ser > 0
- `rsi_14` do spot → precisa ser > 50
- `bb_pct` do spot → precisa ser < 0.98
- `close > ma_21` do spot → precisa ser True

#### 2c. Stops do Bot 2 (se posição aberta com entry_bot == "bot2")

Mesmo formato da barra de stops do Bot 1, mas usando os stops do Bot 2:
- SL: 1.5% fixo
- TP: 2% fixo
- Trailing: 1% fixo
- Max hold: 120h (mostrar horas restantes)

Adicionar indicador de tempo restante:
```
Tempo no trade: 48h / 120h max (72h restantes)
```

#### 2d. Equity curve e stats separados — NÃO IMPLEMENTAR AGORA

Como ambos os bots compartilham o mesmo capital e portfolio, a equity curve é uma só. Manter a equity curve existente na seção do Bot 1 como "equity combinada". No Bot 2, apenas mostrar:

```python
st.caption("Equity curve combinada (Bot 1 + Bot 2) disponível acima.")
```

#### 2e. Histórico de trades Bot 2

Se houver trades completados com `entry_bot == "bot2"` no `completed_trades`:

```python
trades_path = ROOT / "data/05_output/completed_trades.json"
if trades_path.exists():
    trades = json.loads(trades_path.read_text())
    bot2_trades = [t for t in trades if t.get("entry_bot") == "bot2"]
    if bot2_trades:
        # Mostrar tabela com: entry_time, entry_price, exit_price, pnl_pct, exit_reason
```

Se não houver trades: `st.info("Nenhum trade Bot 2 completado ainda.")`

### 3. Ajustar equity curve do Bot 1

Renomear título do gráfico:
```python
title="Equity Curve Combinada (Bot 1 + Bot 2) vs Buy & Hold"
```

Nos stats abaixo da equity curve, adicionar um breakdown:
```python
# Contar trades por bot
bot1_trades = [t for t in all_trades if t.get("entry_bot", "bot1") == "bot1"]
bot2_trades = [t for t in all_trades if t.get("entry_bot") == "bot2"]

# Adicionar na linha de stats
st.metric("Trades Bot 1", len(bot1_trades))
st.metric("Trades Bot 2", len(bot2_trades))
```

## Dados disponíveis

### portfolio_state.json (campos relevantes)
```json
{
  "has_position": true/false,
  "entry_bot": "bot1" ou "bot2",
  "entry_price": 77000.0,
  "entry_time": "2026-04-17T...",
  "capital_usd": 10156.96,
  "stop_loss_price": 75845.0,
  "take_profit_price": 78540.0,
  "stops_mode": "dynamic" ou "bot2_fixed",
  "entry_max_hold_hours": 120,
  "last_momentum_passed": false,
  "last_momentum_reason": "LOW_LIQUIDITY (...)",
  "last_momentum_stablecoin_z": 0.8,
  "last_filter_ret_1d": 0.029,
  "last_filter_rsi": 82.4,
  "last_filter_passed": false
}
```

### cycle_log.parquet (campos relevantes)
```
timestamp, signal ("ENTER", "ENTER_BOT2", "HOLD", "BLOCK", "FILTERED", "EXIT"),
capital_usd, price, score, entry_bot
```

### completed_trades.json
```json
[
  {
    "entry_bot": "bot1",
    "entry_price": 71000,
    "exit_price": 72114,
    "pnl_pct": 1.57,
    "exit_reason": "take_profit",
    "entry_time": "...",
    "exit_time": "..."
  }
]
```

## Estilo visual

Manter exatamente o mesmo estilo CoinGlass dark theme. As classes CSS já existem:
- `.cg-card`, `.cg-card-title`, `.cg-card-value`, `.cg-card-sub`
- `.pos` (verde), `.neg` (vermelho), `.neut` (cinza), `.warn` (amarelo)
- Cores: GREEN="#3fb950", RED="#f85149", AMBER="#d29922", GREY="#8b949e", BLUE="#58a6ff"

## Notas de cuidado

- **NÃO alterar** nenhuma outra seção do dashboard (header, gate scoring, whale, derivativos, macro, news, system health)
- **NÃO alterar** paper_trader.py — apenas app.py
- **NÃO criar** portfolio separado para Bot 2 — capital é compartilhado
- Backward compatibility: trades sem `entry_bot` → default "bot1"
- O Bot 2 pode estar desabilitado (`momentum_filter.enabled: false`) — nesse caso, mostrar `st.info("Bot 2 desabilitado")` na seção inteira
- Testar visualmente que ambas as seções renderizam corretamente quando:
  - Sem posição aberta
  - Bot 1 com posição aberta
  - Bot 2 com posição aberta
  - Nenhum trade completado ainda

## Checklist

1. [ ] Renomear seção Paper Trading existente para "Bot 1 (Reversal)"
2. [ ] Remover expander do Bot 2 da seção do Bot 1
3. [ ] Criar seção "Bot 2 (Momentum)" com 4 cards de status
4. [ ] Momentum Filter status bar (sempre visível, formato inline)
5. [ ] Stops bar do Bot 2 (quando entry_bot == "bot2")
6. [ ] Tempo restante no trade Bot 2 (horas vs max 120h)
7. [ ] Histórico de trades Bot 2
8. [ ] Renomear equity curve para "Combinada"
9. [ ] Breakdown de trades por bot nos stats
10. [ ] Testar com Bot 2 desabilitado
11. [ ] Git commit + push + docker rebuild
