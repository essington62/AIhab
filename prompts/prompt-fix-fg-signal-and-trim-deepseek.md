# Task: Corrigir inversão de sinal do F&G + enxugar system prompt do DeepSeek + pequenos bugs

## Contexto

Dashboard AI.hab em produção (EC2 São Paulo, http://54.232.162.161:8501).
Arquitetura e especificação em `CLAUDE.md` na raiz. Deploy via Docker Compose (aihab-app + aihab-dashboard).

Bugs detectados via inspeção visual do dashboard e da análise gerada pelo DeepSeek. Quatro correções agrupadas (são pequenas e têm relação entre si).

## Bug 1 — Inversão de sinal do F&G (CRÍTICO)

### Sintomas

- **Header**: F&G = 23 (Extreme Fear)
- **Card Sentiment** (seção Gate Scoring v2): texto interpretativo diz "Greed elevado — mercado complacente, cautela" (oposto do que o valor real indica)
- **Payload do DeepSeek**: `fg_z = +1.71`, e o modelo interpretou como "viés otimista"
- **Contribuição do card G8 F&G**: `-0.141`

### Hipótese

O z-score do F&G está sendo calculado/persistido com sinal invertido. Com `fg_raw = 23` e média histórica (~45-50 em regime normal), o z deveria ser **negativo** (~-1.5 a -2.0). Está saindo **positivo** (+1.71).

Isso importa porque:
- G8 tem `sign = -0.211` (contrarian) em `conf/parameters.yml`
- Com z invertido, a contribuição final ao cluster Sentiment está com sinal errado
- O sistema está penalizando Extreme Fear ao invés de premiá-lo (contra a filosofia contrarian)

### Investigação

1. **Confirmar inversão empiricamente:**
   ```bash
   docker exec aihab-app python -c "
   import pandas as pd
   df = pd.read_parquet('/app/data/02_features/gate_zscores.parquet')
   cols = [c for c in df.columns if 'fg' in c.lower() or 'fear' in c.lower()]
   print(df[['timestamp'] + cols].tail(10).to_string())
   "
   ```
   Se `fg_raw = 23` e `fg_z > 0` → confirmado invertido.

2. **Rastrear cálculo:**
   - Procurar em `src/features/gate_features.py` (ou equivalente) onde o z-score do F&G é computado.
   - Verificar a fórmula: deveria ser `(raw - rolling_mean) / rolling_std`.
   - Possíveis causas: ordem invertida na subtração, sinal errado aplicado no persist, ou rolling window com bug.

3. **Conferir todos os outros z-scores simultaneamente:**
   - Este mesmo bug pode estar em outros gates (positioning, liquidity, etc.)
   - Fazer sanity check: para cada z-score, confirmar que o sinal observado é consistente com a direção esperada (ex: OI alto → z positivo; stablecoin mcap crescendo → z positivo).

### Correção

- Corrigir a fórmula no local que gera o z invertido.
- Re-rodar o ciclo horário para atualizar parquets: `docker exec aihab-app bash -l -c 'cd /app && bash scripts/hourly_cycle.sh'`
- Validar que após correção: com F&G raw = 23, `fg_z` agora é negativo e o card Sentiment contribui positivamente (porque `z negativo × sign negativo = positivo`).

### NÃO mexer

- Valores de `conf/parameters.yml` (signs, thresholds, max_scores).
- Lógica de cluster_caps.
- Só corrigir a geração do z-score.

## Bug 2 — Texto interpretativo do card Sentiment ignora valor atual

Mesmo que o Bug 1 esteja corrigido, o texto do card Sentiment parece ser estático ("Greed elevado — mercado complacente, cautela") ou está mapeado errado para o valor do cluster.

### Investigação

Localizar em `src/dashboard/app.py` a renderização do card Sentiment. Verificar se existe lógica condicional tipo:

```python
if sentiment_score > X:
    text = "Greed elevado..."
elif sentiment_score < Y:
    text = "Fear extremo..."
else:
    text = "Sentimento neutro..."
```

Se a lógica existe, validar os thresholds. Se não existe, criar.

### Correção

O texto deve refletir o valor atual do cluster Sentiment **E** o F&G raw (não só o z). Sugestão:

```
Se F&G raw < 25 → "Extreme Fear — contrarian bullish (retail aterrorizado)"
Se F&G raw 25-45 → "Fear — cautela acumulando"
Se F&G raw 45-55 → "Neutro"
Se F&G raw 55-75 → "Greed — otimismo moderado"
Se F&G raw > 75 → "Extreme Greed — contrarian bearish (euforia, risco de correção)"
```

Aplicar padrão similar para os outros cards com texto interpretativo (Technical, Positioning, Macro, Liquidity, News) — garantir que cada um lê o valor atual e escreve mensagem condizente.

## Bug 3 — Enxugar system prompt do DeepSeek

### Problema

O system prompt atual gera análise em 6 seções: Regime/Macro, Estrutura de Preço (S&R), Posicionamento Institucional, Sentimento/Notícias, Leitura do Sinal Atual, Gatilhos. O usuário considerou a análise longa demais, especialmente porque as seções 1, 3 e 4 apenas recapitulam informação que já está visível nos cards do dashboard.

### Correção

Simplificar o system prompt para gerar análise em **3 seções**:

1. **Leitura do sinal atual** (2-3 linhas): síntese da decisão (ENTER/HOLD/BLOCK), score ajustado vs threshold, tensão narrativa principal do mercado no momento.

2. **Suportes e Resistências numéricos**: lista concreta com níveis específicos. Formato:
   ```
   Resistência imediata: $XX,XXX (descrição)
   Resistência estrutural: $XX,XXX (descrição)
   Suporte imediato: $XX,XXX (descrição)
   Suporte estrutural: $XX,XXX (descrição)
   Zona de invalidação: $XX,XXX (descrição)
   ATR(14): $XXX (magnitude média dos movimentos)
   ```

3. **Gatilhos para mudar o sinal**: condições multi-variável objetivas.
   ```
   Para HOLD → ENTER:
   - Condição técnica específica (ex: fechamento 1h > $X + RSI < Y)
   - Condição de posicionamento/liquidez (ex: funding z > 0.5 + baleia acumulando)

   Para HOLD → BLOCK:
   - Condição de kill switch iminente
   - Condição de ruptura de suporte
   ```

### Restrições do novo prompt

- **Proibir recapitulação** do que já está visível nos cards (regime, macro, sentiment, news cluster scores).
- **Proibir markdown** (sem `**`, `##`, bullets com `*`). Usar texto plano com quebras de linha. Isso resolve também o artefato visual `∗∗eotopodorangede7diasem∗∗` que apareceu na última execução.
- **Limitar tamanho:** ~300-500 palavras no total.
- **Exigir números concretos** em todos os S&R e gatilhos — nunca "perto da resistência" sem número, nunca "melhora do momentum" sem threshold.

### Arquivo

O system prompt provavelmente está em `src/dashboard/app.py` (seção AI Analyst) ou em um arquivo separado tipo `src/llm/deepseek_prompt.py` ou `conf/prompts/deepseek_system.txt`. Localizar, substituir, commitar.

### Parâmetros do call

- Manter `max_tokens=2000` (dá folga mesmo com prompt menor).
- Manter `finish_reason=stop` check.

## Bug 4 — Fed Observatory ausente no payload

A última análise do DeepSeek mencionou: *"A ausência de dados do Fed Observatory limita a leitura do posicionamento interno do banco central."*

### Investigação

1. Verificar se `src/features/fed_observatory.py` está sendo chamado no ciclo horário ou diário.
2. Conferir se os outputs (probabilidades cut/hold/hike) estão sendo persistidos em parquet.
3. Validar se o payload enviado ao DeepSeek inclui esses campos.

### Correção

Se o módulo não está rodando: adicionar ao `scripts/hourly_cycle.sh` ou `scripts/daily_update.sh` conforme frequência apropriada.
Se está rodando mas não chega no payload: adicionar os campos em `src/dashboard/app.py` na montagem do payload do DeepSeek.

## Entregáveis

1. **Diagnóstico** (5-10 linhas): onde estava cada bug, causa raiz de cada um.
2. **Patches** (diffs) dos arquivos afetados. Esperado:
   - `src/features/gate_features.py` (ou equivalente) — correção do z-score F&G
   - `src/dashboard/app.py` — texto interpretativo dos cards + payload DeepSeek com Fed Observatory
   - Arquivo do system prompt DeepSeek — versão enxuta
   - `scripts/hourly_cycle.sh` ou `daily_update.sh` — se precisar adicionar Fed Observatory
3. **Verificação**:
   - Output do comando `gate_zscores.parquet` mostrando `fg_z` negativo com F&G raw baixo.
   - Screenshot do dashboard com card Sentiment mostrando texto correto.
   - Exemplo de análise do DeepSeek no formato novo (3 seções, plain text, ~400 palavras).
4. **Commit + push + deploy** na EC2:
   ```bash
   git add -A && git commit -m "fix: F&G z-score sign + card texts + slim DeepSeek prompt + Fed Obs payload"
   git push origin main

   # Na EC2:
   ssh -i ~/.ssh/aihab-key-sp.pem ubuntu@54.232.162.161 "
     cd ~/AIhab && git pull && docker compose build --no-cache && docker compose up -d
   "
   ```

## Arquivos relevantes

- `CLAUDE.md` (spec geral, convenções de sign dos gates)
- `conf/parameters.yml` (gate_params com signs — NÃO mexer)
- `src/features/gate_features.py` (z-scores)
- `src/features/fed_observatory.py` (Fed probabilities)
- `src/dashboard/app.py` (cards + payload DeepSeek + system prompt, se inline)
- `src/models/gate_scoring.py` (consumo dos z-scores — verificar se o bug não tá aqui também)
- `scripts/hourly_cycle.sh`, `scripts/daily_update.sh`
- `data/02_features/gate_zscores.parquet` (validação)

## Nota final

O usuário vai deixar o sistema rodar 1-2 semanas em paper trading observação após essas correções. Não haverá mais ajustes de parâmetros durante esse período. Portanto esta é a **última janela** para consertar bugs antes da amostra de validação. Vale dobrar o cuidado com regressions — rodar os testes existentes (`pytest tests/`) após as correções.
