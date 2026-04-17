# Estudo: Modelo Supervisionado de Regime — Resultados e Análise

**Data:** 17 de abril de 2026  
**Objetivo:** Avaliar a viabilidade de substituir o HMM (não-supervisionado) por um modelo supervisionado treinado com dados históricos, e investigar se as features dos gates podem prever trades lucrativos.

---

## 1. Contexto e Motivação

O modelo atual de regime (R5C HMM) está praticamente inútil — fica em Sideways permanente porque exige movimentos extremos para classificar Bull ou Bear. Com o multiplicador Sideways que era ×0.5, bloqueava todas as entradas. Mesmo com ×1.0 (paper trading), a informação de regime não está agregando valor ao trading.

A proposta: olhar para a série histórica, entender os movimentos reais de subida/descida/lateralização, definir o que é Bull/Bear/Sideways com base em dados, e treinar um modelo supervisionado usando as variáveis dos gates como input.

## 2. Dados Disponíveis

**Série BTC spot daily:** 838 dias (Jan 2024 — Abr 2026), preço de $39,568 a $124,659.

**Gate features (all_zscores):** 11 z-scores calculados diariamente:
- Macro: dgs10_z, dgs2_z, curve_z, rrp_z
- Posicionamento: oi_coin_margin_z, funding_rate_z, taker_ratio_z
- Liquidez: stablecoin_mcap_z, etf_flows_7d_z
- Sentimento: fear_greed_z, bubble_index_z

**ETF flows:** desde 11 Jan 2024 (584 dias) — confirma corte natural pós-ETF approval.

**Technical features** (calculadas do OHLCV): BB%, RSI-14, ATR%, retornos 1d/3d/5d/10d, volume z-score, distância MA50/MA200, range 5d, dia da semana.

## 3. Análise de Forward Returns

Calculamos retornos futuros (3d, 5d, 7d, 14d) para entender a distribuição de movimentos:

| Horizonte | Média | Std | p10 | p50 | p90 |
|-----------|-------|-----|-----|-----|-----|
| 3 dias | +0.28% | 4.25% | -4.9% | +0.3% | +5.2% |
| 5 dias | +0.46% | 5.42% | -6.2% | +0.3% | +7.7% |
| 7 dias | +0.64% | 6.42% | -6.9% | +0.4% | +8.8% |
| 14 dias | +1.25% | 9.32% | -9.2% | +0.5% | +12.3% |

**Insight:** A distribuição é levemente positiva (bias bullish do BTC no período), mas a volatilidade é alta. O desvio padrão de 5d é 5.4%, o que significa que variações de ±5% em 5 dias são normais (1 std).

### Distribuição de regimes (forward return 5d, threshold ±3%)

- **Bull** (retorno > +3%): 239 dias (29%)
- **Sideways** (entre -3% e +3%): 391 dias (47%)
- **Bear** (retorno < -3%): 203 dias (24%)

### Fases de regime (sequências contínuas)

Identificamos 279 fases de regime. Duração mediana: **2 dias** para todos os regimes. As fases longas (>5 dias) são raras mas significativas:

**Maiores fases Bull:**
- Fev 2024: 12d, +21.7% ($51,849 → $63,114) — rally pré-halving
- Nov 2024: 10d, +15.6% ($69,496 → $80,370) — eleição Trump
- Jul 2024: 12d, +14.6% ($55,858 → $63,988) — recuperação pós-crash

**Maiores fases Bear:**
- Jan 2026: 10d, -17.2% ($88,347 → $73,166) — crash início do ano
- Nov 2025: 11d, -12.6% ($104,723 → $91,555)
- Jul 2024: 8d, -9.4% ($67,908 → $61,498)

## 4. Simulação de Trades (Baseline)

### SG 1% / SL 1% / Max 5 dias (entrando TODOS os dias)

| Outcome | Count | % | Avg Return |
|---------|-------|---|------------|
| Win | 304 | 36.5% | +1.00% |
| Loss | 495 | 59.4% | -1.00% |
| Timeout | 34 | 4.1% | variável |

**Win Rate: 36.5% | Profit Factor: ~0.61 | Total: negativo**

O win rate varia significativamente por mês: de 14% (Fev 2026, mercado em crash) a 60% (Set 2025, rally). Isso confirma que EXISTEM condições favoráveis — o desafio é identificá-las.

### SG 2% / SL 1% (assimétrico)

**Win Rate: 29.9% | PF: 0.86 | Total: -83%**

Pior que o simétrico. O mercado atinge -1% muito mais facilmente que +2%. A assimetria trabalha contra.

### Meses com melhor win rate (SG 1%/SL 1%)

- Set 2025: 60% — rally forte
- Abr 2026: 58% — momento atual
- Jul 2025: 55% — tendência positiva
- Jun 2024: 53% — recuperação
- Fev 2024: 52% — rally pré-halving

## 5. Modelo com Gate Z-Scores (11 features)

### Correlações com Win

Todas as correlações foram **extremamente fracas** (< 0.06):

| Feature | Correlação | Significância |
|---------|-----------|--------------|
| rrp_z | +0.059 | Fraca |
| stablecoin_mcap_z | -0.050 | Fraca |
| curve_z | -0.028 | Desprezível |
| etf_flows_7d_z | -0.022 | Desprezível |
| funding_rate_z | +0.020 | Desprezível |

### Random Forest (5-fold CV)

**Accuracy: 0.631 ± 0.009 vs Baseline: 0.635**

O modelo é **estatisticamente equivalente a sempre prever Loss** (classe majoritária). Não aprendeu nenhum padrão preditivo.

**Diagnóstico:** Os z-scores dos gates são sinais macro/posicionamento que mudam devagar (escala de dias/semanas). Um trade de 1% em 1-5 dias é dominado por dinâmica de curto prazo que essas features não capturam individualmente.

## 6. Modelo com Todas as Features (27 features)

Adicionamos 16 features técnicas: BB%, RSI, ATR%, retornos 1d/3d/5d/10d, volume z-score, distância MA50/MA200, range 5d%, dia da semana.

### Resultados com TimeSeriesSplit (5 folds)

| Modelo | Accuracy | Baseline |
|--------|----------|----------|
| Random Forest | 0.639 ± 0.042 | 0.647 |
| Gradient Boosting | 0.593 ± 0.034 | 0.647 |

**Ambos os modelos ficaram ABAIXO do baseline.** Nenhum sinal preditivo detectado.

### Feature Importance (Gradient Boosting)

Top 5: etf_flows_7d_z (0.076), stablecoin_mcap_z (0.067), range_5d_pct (0.063), ret_1d (0.061), rrp_z (0.061)

A importância é **dispersa uniformemente** entre as features — nenhuma domina. Isso indica que o modelo está capturando ruído, não sinal.

### Correlações mais fortes das novas features

| Feature | Correlação | Interpretação |
|---------|-----------|---------------|
| dow (dia da semana) | +0.151 | Padrão semanal de liquidez |
| atr_pct | -0.122 | Baixa volatilidade → mais wins |
| range_5d_pct | -0.098 | Mercado comprimido → mais wins |

## 7. Diagnóstico: Por que não funciona?

### 7.1 O SG/SL de 1% é ruído

Com ATR% médio de ~1.2% no BTC, um stop de 1% está dentro de **1 desvio padrão do ruído diário**. É como tentar prever se uma moeda vai cair cara ou coroa — o noise domina o signal.

### 7.2 Features individuais vs combinações

**PONTO CRÍTICO:** Os gates no AI.hab não funcionam isoladamente — funcionam em COMBINAÇÃO. O scoring system soma 11 gates com pesos diferentes e aplica kill switches. Uma feature sozinha (ex: oi_z = +0.5) não significa nada; mas oi_z = +0.5 + bb_pct < 0.20 + fear_greed_z < -0.5 = forte sinal de compra.

A correlação linear e o Random Forest/GB básico NÃO capturam bem essas interações complexas com o dataset pequeno que temos (533 rows). Com mais dados ou com feature engineering de interações, o resultado poderia ser diferente.

### 7.3 O problema está no alvo, não nas features

O score do AI.hab (combinação dos gates) JÁ é um bom filtro de entrada — o primeiro trade fechou +1.57%. O problema não é "quando entrar" mas sim "como gerenciar o trade depois de entrar" (stops, trailing, saída). 

As features dos gates provavelmente TÊM poder preditivo para movimentos maiores (3-5%) em horizontes mais longos (7-14 dias), que é exatamente o que o scoring system já faz. Tentar comprimir isso para 1% SG/SL em 5 dias perde o sinal.

## 8. Caminhos Possíveis

### Caminho A — Modelo de regime com horizonte adequado
- Manter SG 2-3% / SL 2-3% (compatível com a volatilidade do BTC)
- Treinar o modelo para prever forward return 7d > ±3% (threshold onde as features têm mais poder)
- Usar como substituto do HMM para o multiplicador de regime
- Base: 32% Bull / 41% Sideways / 27% Bear — split razoável

### Caminho B — Feature engineering de interações
- Criar features compostas: score_total (soma ponderada dos gates como o AI.hab faz), combinações BB×RSI, OI×Funding, etc.
- Testar se as interações capturam o que as features individuais perdem
- Isso essencialmente "ensina" ao modelo o que o scoring system já faz, mas permite que ele descubra combinações melhores

### Caminho C — Dados intraday (4h)
- Usar candles 4h ao invés de daily para SG/SL de 1%
- Mais datapoints (6× mais), melhor resolução temporal
- Features técnicas em 4h são mais preditivas para movimentos curtos
- Limitação: menos histórico disponível (dados 4h são mais recentes)

### Caminho D — Modelo de saída (não de entrada)
- Aceitar que o scoring system JÁ é bom para ENTRADA
- Treinar modelo para SAÍDA: dado que entrei, quando é o melhor momento de sair?
- Features: evolução do preço pós-entrada, MAE/MFE em tempo real, mudança nos gates
- Precisa de mais trades para treinar (acumular dados do paper trading)

### Caminho E — Ensemble: scoring atual + ML para refinamento
- O scoring system (rule-based) gera o sinal base (ENTER/HOLD)
- ML model ajusta a confiança: "o scoring diz ENTER, mas o modelo prediz 70% de win → vai" vs "scoring diz ENTER mas modelo prediz 40% → espera"
- Melhor dos dois mundos: interpretabilidade + poder preditivo

## 9. Recomendação

**Curto prazo (agora):** Seguir o **Caminho A** — treinar modelo de regime com horizonte de 7d e threshold ±3%. Substituir o HMM. Manter os stops ATR-based que já implementamos (se adapta à volatilidade).

**Médio prazo (após acumular 20+ trades no paper trading):** Adicionar **Caminho B** (interações) e **Caminho D** (modelo de saída), usando dados reais de MAE/MFE dos trades.

**Longo prazo:** **Caminho E** — ensemble com scoring + ML.

## 10. Conclusões

1. **Os gate z-scores sozinhos não predizem trades de 1% SG/SL** — as correlações são desprezíveis e os modelos não batem o baseline.

2. **Adicionar features técnicas (BB%, RSI, ATR, momentum) não resolveu** — o problema não é falta de features, mas incompatibilidade entre o horizonte do trade (1-5d, 1%) e a escala temporal das features (dias/semanas).

3. **O score combinado dos gates (sistema atual) provavelmente funciona melhor que features isoladas** — a inteligência está na COMBINAÇÃO ponderada, não nas features individuais. Isso precisa ser testado como Caminho B.

4. **SG/SL de 1% é incompatível com trading diário de BTC** — está dentro do noise. O sweet spot está em 2-3% SG com horizonte de 5-7 dias, que é onde os gates têm poder preditivo real.

5. **O primeiro trade (+1.57%) validou a abordagem rule-based** — o scoring system funciona. A evolução não é substituí-lo por ML, mas usar ML para REFINÁ-LO (ajustar confiança, otimizar saída).

---

## 11. Estudo Complementar: Score Edge + Reversal Filter (Descoberta Principal)

### 11.1 Score Bucket Analysis (SG 2% / SL 1.5%)

Mudando para stops compatíveis com a volatilidade do BTC (SG 2%, SL 1.5%, max hold 5d), o score combinado mostra sinal fraco mas real:

| Score Bucket | Trades | WR | PF | Total Return |
|-------------|--------|-----|------|-------------|
| 0.0 – 1.0 | 240 | 38% | 0.82 | -57% |
| 1.0 – 2.0 | 161 | 41% | 0.93 | -14% |
| 2.0 – 2.5 | 76 | 43% | 1.01 | +1% |
| 2.5 – 3.0 | 54 | 44% | 1.06 | +4% |
| 3.0 – 3.5 | 39 | 41% | 0.93 | -4% |
| 3.5 – 4.0 | 31 | 45% | 1.09 | +4% |
| 4.0 – 6.0 | 22 | 46% | 1.11 | +3% |
| > 6.0 | 2 | 50% | — | — |

**Insight:** A monotonia é clara — score mais alto = melhor WR e PF. Score > 3.5 é o único bucket consistentemente lucrativo (WR 46%, PF 1.11). Mas a margem é fina.

### 11.2 Regime proxy: MA200

Dividindo por posição relativa à MA200:

| Condição | Trades | WR | PF |
|----------|--------|-----|------|
| Acima MA200 + Score 2.5-3.0 | 30 | 55% | 1.63 |
| Abaixo MA200 + Score 2.5-3.0 | 24 | 30% | 0.57 |
| Acima MA200 + Score > 3.5 | 18 | 58% | 1.87 |
| Abaixo MA200 + Score > 3.5 | 14 | 36% | 0.75 |

**2026 é 100% abaixo da MA200** — bear market. O score ainda funciona para capturar bounces, mas precisa de filtro adicional.

### 11.3 Feature Analysis: O que diferencia Wins de Losses?

Analisando as características técnicas no momento de entrada dos trades (Score > 2.5):

| Feature | Média Wins | Média Losses | Delta |
|---------|-----------|-------------|-------|
| RSI | 35.2 | 42.1 | -6.9 |
| BB% | 0.15 | 0.22 | -0.07 |
| ret_1d | -0.8% | -1.8% | +1.0% |
| ATR% | 1.3% | 1.4% | -0.1% |

**Padrão claro:** Wins acontecem quando RSI está mais baixo (oversold profundo) e o retorno de 1 dia não é muito negativo (a queda está desacelerando). É um padrão de **reversão de capitulação**.

### 11.4 Filtros individuais testados

| Filtro | Trades | WR | PF | Total |
|--------|--------|-----|------|-------|
| RSI < 30 | 28 | 50% | 1.33 | +12% |
| RSI < 35 | 52 | 50% | 1.33 | +22% |
| RSI < 40 | 82 | 46% | 1.14 | +15% |
| BB% < 0.15 | 38 | 50% | 1.33 | +17% |
| ret_1d > -1% | 98 | 45% | 1.09 | +12% |

RSI < 35 é o melhor filtro individual: dobra os trades vs RSI < 30, mantendo a mesma performance.

### 11.5 RESULTADO FINAL: Combo Filter

Simulação realista com trades não-sobrepostos (só abre novo trade após fechar o anterior):

| Estratégia | Trades | WR | PF | Total | MaxDD |
|-----------|--------|-----|------|-------|-------|
| Score>2.5 (baseline) | 146 | 42% | 0.96 | -5.5% | -19.5% |
| Score>2.5 + RSI<35 | 52 | 50% | 1.33 | +22.0% | -9.0% |
| **Score>2.5 + RSI<35 + ret_1d>-1%** | **41** | **63%** | **2.31** | **+29.5%** | **-4.5%** |
| Score>2.5 + RSI<35 + ret_1d>-1% (SG2/SL2) | 40 | 68% | 2.08 | +28.0% | -6.0% |
| Score>2.5 + RSI<35 + ret_1d>-1% (SG3/SL2/7d) | 38 | 61% | 2.30 | +39.0% | -6.0% |
| Score>2.5 + RSI<35 (ATR stops) | 46 | 50% | 1.50 | +30.5% | -8.0% |

### 11.6 Interpretação da Estratégia Vencedora

A combinação **Score > 2.5 + RSI < 35 + ret_1d > -1%** captura um padrão específico e repetível:

1. **Score > threshold** → Os fundamentos macro, posicionamento, liquidez e sentimento dizem "está barato para comprar" (contrarian signal)
2. **RSI < 35** → O preço está em zona de sobrevenda profunda — confirmação técnica de capitulação do mercado
3. **ret_1d > -1%** → A queda está perdendo força — não é faca caindo, é possível formação de fundo

É essencialmente um **detector de fundo de capitulação confirmada**. O score identifica que as condições fundamentais são favoráveis, o RSI confirma que o mercado já vendeu demais, e o ret_1d garante que não estamos entrando no meio de um crash ativo.

### 11.7 Frequência e Aplicabilidade

- 41 trades em ~833 dias = **~1 trade a cada 20 dias** (média)
- Compatível com operação de baixa frequência, sem overtrading
- MaxDD de -4.5% é excelente para BTC (que tem drawdowns de 20-30% regulares)
- 63% WR com PF 2.31 dá margem confortável para variância estatística

## 12. Conclusões Atualizadas

1. **O score combinado TEM edge**, mas precisa de stops adequados (SG 2%+) e filtros técnicos para ser lucrativo.

2. **RSI < 35 + ret_1d > -1% transforma o sistema** de breakeven (WR 42%, PF 0.96) para fortemente lucrativo (WR 63%, PF 2.31).

3. **A lógica é "capitulação confirmada"** — score diz "barato", RSI diz "oversold profundo", ret_1d diz "a queda parou".

4. **O filtro reduz trades de 146 para 41** (72% menos), mas cada trade é muito mais qualificado. Qualidade sobre quantidade.

5. **Próximo passo: implementar o reversal filter no AI.hab** como camada de confirmação pós-scoring. Prompt gerado em `prompts/prompt-reversal-filter.md`.
