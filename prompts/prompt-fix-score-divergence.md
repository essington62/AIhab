# Task: Investigar e corrigir divergência de Score (header vs corpo) no dashboard AI.hab

## Contexto

Dashboard Streamlit do AI.hab em produção (EC2 São Paulo, http://54.232.162.161:8501).
Arquitetura e especificação completa estão em `CLAUDE.md` na raiz do repo.

## Problema observado

Na tela renderizada do dashboard, o **Score** aparece em dois lugares com valores diferentes:

- **Header** (topo da página): `Score: 1.964 / 2.5`
- **Seção "Gate Scoring v2"** (logo abaixo do header): `Score Total: +4.049` | `Threshold: 2.5`

Verificação manual: a soma dos 6 clusters exibidos no corpo bate com 4.049:

```
Technical   +2.500
Positioning +0.836
Macro       -0.094
Liquidity   +0.593
Sentiment   -0.287
News        +0.500
─────────────────
Total       +4.048  ✓ (bate com o corpo)
```

O valor do **header (1.964) não corresponde** nem ao score bruto, nem ao score multiplicado pelo regime Sideways (4.049 × 0.5 = 2.02), nem a nenhum cálculo óbvio a partir dos clusters exibidos.

Além disso:

- Gate decision exibida: **HOLD**
- Score corpo (4.049) > Threshold (2.5) → pela leitura ingênua deveria ser ENTER
- Se o HOLD vem do multiplicador Sideways do G0 (x0.5 conforme CLAUDE.md), isso precisa ser **transparente na UI**

## O que investigar

1. **Encontrar as duas fontes do Score:**
   - No arquivo `src/dashboard/app.py`, localize onde o Score é lido/calculado para:
     - a) O componente do header
     - b) A seção "Gate Scoring v2" (Score Total)
   - Identifique se lêem do mesmo parquet / mesma função ou de fontes diferentes.
   - Provável hipótese: o header lê de um parquet stale (ex: último snapshot salvo pelo cron) e o corpo recalcula em memória — ou vice-versa.

2. **Rastrear o pipeline do score:**
   - `src/models/gate_scoring.py` (ou equivalente) é onde o score final é computado.
   - Verifique se existe: score_bruto, score_ajustado_por_regime, score_com_kill_switches — quais são persistidos e onde.
   - Conferir se o multiplicador do G0 (Bear=BLOCK, Sideways=x0.5, Bull=x1.0) está sendo aplicado no local correto e se o valor persistido é pré ou pós-multiplicador.

3. **Reproduzir o valor 1.964:**
   - Rode os scripts de ingestão e scoring localmente (ou via `docker exec aihab-app`) e verifique qual combinação de cálculos produz 1.964.
   - Se não for reproduzível, é bug de staleness (header lendo arquivo antigo).

## O que corrigir

Depois de diagnosticar, implementar UMA das duas soluções (prefira a primeira):

**Opção A — Fonte única (preferida):**

- Garantir que header e corpo leiam do **mesmo** objeto/parquet, com o mesmo timestamp.
- O valor exibido deve ser o **score final de decisão** (pós-regime, pós-kill-switches).
- Na seção "Gate Scoring v2" mostrar explicitamente o cálculo:

  ```
  Score bruto (soma clusters): +4.049
  × Regime Sideways (x0.5):    +2.025
  = Score ajustado:            +2.025
  Threshold:                    2.500
  Decisão:                     HOLD  (score < threshold)
  ```

- Header passa a mostrar apenas o score ajustado: `Score: 2.025 / 2.5`.

**Opção B — Dois scores com rótulos inequívocos:**

- Header: `Score (ajustado): X.XXX / threshold`
- Corpo: `Score bruto: Y.YYY` + linha separada `Ajuste de regime: x0.5` + `Score ajustado: X.XXX`
- Ambos devem vir da mesma fonte — nunca de arquivos/funções diferentes.

## Restrições

- **Não alterar** a lógica de `g1_technical` (buckets walk-forward validados — CLAUDE.md diz "NÃO MEXER").
- **Não alterar** os valores em `conf/parameters.yml` (gate_params, cluster_caps).
- Mudanças devem ser em `src/dashboard/app.py` e, se necessário, em `src/models/gate_scoring.py` para expor os valores intermediários.
- Manter compatibilidade com os outros consumidores dos parquets de scoring (se houver).

## Entregáveis

1. **Diagnóstico** em 5-10 linhas: onde estavam as duas fontes, por que divergiam.
2. **Patch** (diff) nos arquivos afetados.
3. **Verificação**: screenshot ou log do dashboard mostrando header e corpo com valores consistentes, e o breakdown do cálculo visível.
4. Caso o bug seja de staleness de parquet (header lendo arquivo velho), incluir nota sobre como o cron quebrado (Cron: N/A ⚠️ na tela) contribui pro problema — mas **não** tentar consertar o cron nesta task.

## Arquivos relevantes para começar

- `CLAUDE.md` (spec geral, regra do G0 Sideways x0.5)
- `src/dashboard/app.py` (header + seção Gate Scoring v2)
- `src/models/gate_scoring.py`
- `conf/parameters.yml` (gate_params, cluster_caps, threshold)
- `data/04_scoring/` (parquets de output do scoring — verificar nomes e schemas)

Comece pelo `app.py` e siga o fio até encontrar as duas leituras de score.
