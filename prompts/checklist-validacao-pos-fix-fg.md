# Checklist de validação — pós-fix F&G + DeepSeek

Antes de entrar em modo "deixar rodar 2 semanas sem mexer", validar 4 coisas.
Estimativa: 10 minutos.

## 1. Diff do parameters.yml

```bash
ssh -i ~/.ssh/aihab-key-sp.pem ubuntu@54.232.162.161 \
  "cd ~/AIhab && git log -1 --stat conf/parameters.yml && git diff HEAD~1 HEAD -- conf/parameters.yml"
```

**Ok se mudou:** janela rolling do F&G (ex: window: 30 → window: 180), ou movido pra config separada com baseline absoluta.

**NÃO ok se mudou:** sign, threshold, max_score do g8_fg ou de qualquer outro gate.

Se NÃO ok → reverter `git revert <commit>` e refazer apenas a parte legítima.

## 2. Origem do threshold 3.50 vs 2.5

```bash
ssh -i ~/.ssh/aihab-key-sp.pem ubuntu@54.232.162.161 \
  "cd ~/AIhab && grep -rn 'threshold' conf/ src/models/ src/dashboard/app.py | grep -v '#' | head -20"
```

Procurar de onde vem o 3.50. Se for p75 dinâmico calculado dos últimos 30d de score_history.parquet → ok, autoajuste é feature. Se for hardcoded em algum lugar novo → revisar.

## 3. Sanity check de outros z-scores

Mesmo argumento do F&G (rolling 30d em regime extremo distorce baseline) pode estar afetando outros gates contrarian. Conferir:

```bash
docker exec aihab-app python -c "
import pandas as pd
df = pd.read_parquet('/app/data/02_features/gate_zscores.parquet')
last = df.iloc[-1]
print('Last z-scores:')
for col in df.columns:
    if '_z' in col or '_raw' in col:
        print(f'  {col}: {last[col]:.3f}')
"
```

Cross-check manual: cada z faz sentido com o raw correspondente?
- OI raw alto → OI z deveria ser positivo ✓
- Stablecoin mcap subindo → stable z positivo ✓
- Bubble index alto → bubble z positivo ✓
- F&G raw 23 → fg z negativo (agora deveria estar) ✓

Se algum z e raw apontarem direções opostas: investigar.

## 4. Texto dos cards do dashboard

Abrir http://54.232.162.161:8501 e verificar:

- Card **Sentiment**: com F&G 23, texto deveria dizer algo como "Extreme Fear — contrarian bullish (retail aterrorizado)". NÃO mais "Greed elevado".
- Card **Liquidity**: bate com os números (G5/G7 positivos → "Liquidez entrando")
- Card **Macro**: texto consistente com sub-gates DGS10/DGS2/curve/RRP
- Card **Technical**: texto consistente com BB pct e RSI atuais
- Card **Positioning**: texto consistente com OI z e funding z
- Card **News**: texto consistente com Crypto e Fed

Se algum card ainda mostrar texto contradizendo os números: criar issue específica.

## 5. Análise do DeepSeek formato novo

Triggar nova análise no dashboard (botão "AI Analyst") e validar:

- 3 seções (não 6)
- Texto plano sem `**` ou `##` ou `*`
- Números concretos em todos os S&R
- Gatilhos com condições objetivas
- ~300-500 palavras

Se vier longa de novo ou com markdown: o system prompt não foi enxugado o suficiente, voltar pro Claude Code.

## 6. Testes

```bash
ssh -i ~/.ssh/aihab-key-sp.pem ubuntu@54.232.162.161 \
  "cd ~/AIhab && docker exec aihab-app pytest tests/ -v 2>&1 | tail -30"
```

69 testes deveriam continuar passando. Se algum falhar (especialmente os 4 de consistência YAML/code que pegariam drift): investigar antes de declarar fechado.

## Critério de "pronto pra deixar rodar"

Todos os 6 itens acima checados. Se 1 ou mais falharem: não entra em modo observação ainda.
