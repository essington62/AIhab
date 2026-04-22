# System Health + FRED Usage Audit

**Data:** 2026-04-22 15:00 UTC

---

## 1. FRED Usage em Gate Scoring

### Gates que consomem FRED Macro

Fonte raiz: `data/02_intermediate/macro/fred_daily_clean.parquet`  
Construídos em: `src/features/gate_features.py:91-106` (`_macro_zscores()`)  
Consumidos em: `src/models/gate_scoring.py:174-188` (`evaluate_g3()`)

| Gate | Feature | Derivação | Arquivo (linha) | corr | sens | max_score | Cluster cap |
|------|---------|-----------|-----------------|------|------|-----------|-------------|
| G3 DGS10 | `dgs10_z` | rolling z-score de `dgs10` | gate_features.py:102 | -0.315 | 0.7 | 1.0 | macro [-1.5, 1.0] |
| G3 Curve | `curve_z` | rolling z-score de `dgs10 - dgs2` | gate_features.py:105 | -0.282 | 0.7 | 0.8 | macro [-1.5, 1.0] |
| G3 RRP | `rrp_z` | rolling z-score de `rrp` | gate_features.py:104 | +0.212 | 0.7 | 0.7 | macro [-1.5, 1.0] |
| G3 DGS2 | `dgs2_z` | rolling z-score de `dgs2` | gate_features.py:103 | -0.154 | 0.7 | 0.5 | macro [-1.5, 1.0] |

**Soma bruta máxima G3 = ±3.0, mas cluster cap = [-1.5, 1.0] → impacto real máximo no score total = +1.0 / -1.5**

### Como FRED stale se propaga na decisão

**Tracking:** `paper_trader.py:98-133` (`compute_stale_days()`) lê `clean_macro`, calcula `stale_days["g3_macro"]` em dias.

**Tolerância configurada:** `parameters.yml:stale_tolerance_days.g3_macro = 2` dias.

**Bug de design:** `evaluate_g3()` (gate_scoring.py:174-188) **NÃO checa staleness** — não tem `_stale_gate("g3_macro", stale_days)`. Diferente de G4-G10 que todos têm guarda de stale.

Consequência: com FRED 64h stale:
- `stale_days["g3_macro"] = 2.67` (>2 dias de tolerância)
- G3 **continua computando** com os últimos valores via forward-fill da feature pipeline
- Não há block, não há zero-out, não há warning no score
- O alerta no System Health (⚠️) é cosmético — não afeta a decisão

### Avaliação de risco com 64h stale

| Variável | Velocidade de mudança | Risco 64h stale | Justificativa |
|---------|----------------------|-----------------|---------------|
| DGS10 | ~1–5 bp/dia | 🟢 BAIXO | Moves slowly; 64h ≈ 2–10bp, z-score rolling 252d não mexe |
| DGS2 | ~2–8 bp/dia | 🟢 BAIXO | Similar; curto prazo mais volátil mas z-score absorve |
| Curve (10y-2y) | ~3–12 bp/dia | 🟡 ATENÇÃO | Pode ter inversão/normalização em eventos de risco |
| RRP | Intraday spikes | 🟡 ATENÇÃO | Mais sensível a QE/QT e quarter-end. Porém peso 0.7 (menor) |

**Conclusão FRED stale 64h: PREOCUPANTE, não CRÍTICO**

- DGS10/Curve movem devagar: valores de 2.67d atrás são representativos em 99% dos casos
- RRP é o vetor de risco real, especialmente quarter-end ou mudança Fed
- Cluster cap em 1.0 limita o impacto máximo mesmo que G3 calcule errado
- O cenário crítico seria uma virada sharp de política Fed intra-semana (raro)

### Ações recomendadas (FRED)

1. **Imediato — threshold System Health:** Reduzir de 48h → 36h para FRED Macro (FRED publica ~21:15 UTC; 36h garante alerta se 1 ciclo perdido)
2. **Próximo ciclo — enforce stale em G3:** Adicionar `_stale_gate("g3_macro", stale_days)` em `evaluate_g3()`, retornando 0.0 se stale (mesmo padrão de G4-G10)
3. **Investigar causa do stale:** FRED 64h significa que `daily_update.sh` falhou ontem (07:00 UTC). Ver logs AWS: `tail -50 /app/logs/daily.log`

---

## 2. System Health Coverage

### Fontes atualmente monitoradas

`src/dashboard/app.py:1612-1620`

| Fonte | DataFrame carregado | Arquivo parquet | Threshold | Observação |
|-------|--------------------|-----------------|-----------|-|
| Binance Spot | `spot_df` | `data/02_intermediate/spot/btc_1h_clean.parquet` | 3h | BTC only |
| Futures OI | `oi_df` | `data/02_intermediate/futures/oi_1h_clean.parquet` | 3h | BTC only |
| FRED Macro | `macro_df` | `data/02_intermediate/macro/fred_daily_clean.parquet` | 48h | Mudar pra 36h |
| CoinGlass | `bubble_df` | `data/01_raw/coinglass/bubble_index_daily.parquet` | 72h | Proxy pra todos CG daily |
| Fear & Greed | `fg_df` | `data/01_raw/sentiment/fear_greed_daily.parquet` | 48h | OK |
| News | `load_news("crypto")` | `data/01_raw/news/crypto_news.parquet` | 4h | OK |
| Z-scores | `zs_df` | `data/02_features/gate_zscores.parquet` | 3h | Derivado, não raw |

**7 fontes monitoradas — todas BTC/macro. Zero cobertura de ETH e SOL.**

### Fontes NOVAS não-monitoradas

| Fonte | Arquivo | Usado por | Prioridade | Threshold sugerido |
|-------|---------|-----------|------------|--------------------|
| Binance Spot ETH | `data/01_raw/spot/eth_1h.parquet` | Bot 3 (entry signal) | 🔴 HIGH | 2h |
| Binance Spot SOL | `data/01_raw/spot/sol_1h.parquet` | Bot 4 (entry signal) | 🔴 HIGH | 2h |
| CoinGlass OI ETH | `data/01_raw/futures/eth_oi_4h.parquet` | Bot 3 (G4 equivalent) | 🔴 HIGH | 6h |
| CoinGlass OI SOL | `data/01_raw/futures/sol_oi_4h.parquet` | Bot 4 (G2 OI gate) | 🔴 HIGH | 6h |
| CoinGlass Taker ETH | `data/01_raw/futures/eth_taker_4h.parquet` | Bot 3 (volume gate) | 🔴 HIGH | 6h |
| CoinGlass Taker SOL | `data/01_raw/futures/sol_taker_4h.parquet` | Bot 4 (G1 taker gate) | 🔴 HIGH | 6h |
| CoinGlass Funding ETH | `data/01_raw/futures/eth_funding_4h.parquet` | Bot 3 (G10 equiv) | 🟡 MEDIUM | 8h |
| CoinGlass Funding SOL | `data/01_raw/futures/sol_funding_4h.parquet` | Bot 4 (ingestado, não usado) | 🟡 MEDIUM | 8h |
| Portfolio SOL | `data/04_scoring/portfolio_sol.json` | Bot 4 (estado) | 🟡 MEDIUM | N/A — verificar existe |
| Shadow BTC | `data/08_shadow/taker_z_shadow_log.jsonl` | Shadow mode BTC | 🟢 LOW | N/A — só cresce com trades |
| Shadow SOL | `data/08_shadow/sol_scoring_shadow_log.jsonl` | Shadow scoring SOL | 🟢 LOW | N/A — só cresce com trades |

**Impacto de não monitorar HIGH:**  
Se `sol_taker_4h.parquet` ficar stale, Bot 4 computa `taker_z_prev = None` → bloqueia entry (fail-safe). Mas não há alerta visível antes de investigar manualmente. O dashboard não reflete a saúde dos bots de outros assets.

---

## 3. Patch Sugerido para src/dashboard/app.py

### Localização

Linhas 1612–1620 — dicionário `stale_checks`:

```python
# ANTES (app.py:1612-1620):
stale_checks = {
    "Binance Spot":  (_age_h(spot_df),    3),
    "Futures OI":    (_age_h(oi_df),       3),
    "FRED Macro":    (_age_h(macro_df),   48),
    "CoinGlass":     (_age_h(bubble_df),  72),
    "Fear & Greed":  (_age_h(fg_df),      48),
    "News":          (_age_h(load_news("crypto")), 4),
    "Z-scores":      (_age_h(zs_df),       3),
}
```

```python
# DEPOIS:
# Carregar parquets dos novos assets (antes do dicionário)
_eth_spot   = load_parquet("data/01_raw/spot/eth_1h.parquet")
_sol_spot   = load_parquet("data/01_raw/spot/sol_1h.parquet")
_eth_oi     = load_parquet("data/01_raw/futures/eth_oi_4h.parquet")
_sol_oi     = load_parquet("data/01_raw/futures/sol_oi_4h.parquet")
_eth_taker  = load_parquet("data/01_raw/futures/eth_taker_4h.parquet")
_sol_taker  = load_parquet("data/01_raw/futures/sol_taker_4h.parquet")

stale_checks = {
    # BTC (existentes — renomear para clareza)
    "Spot BTC":      (_age_h(spot_df),    2),   # 3 → 2h (mais apertado)
    "Futures OI":    (_age_h(oi_df),      3),
    "FRED Macro":    (_age_h(macro_df),  36),   # 48 → 36h (catch stale mais cedo)
    "CoinGlass":     (_age_h(bubble_df), 72),
    "Fear & Greed":  (_age_h(fg_df),     48),
    "News":          (_age_h(load_news("crypto")), 4),
    "Z-scores":      (_age_h(zs_df),      3),
    # ETH (Bot 3)
    "Spot ETH":      (_age_h(_eth_spot),  2),
    "OI ETH":        (_age_h(_eth_oi),    6),
    "Taker ETH":     (_age_h(_eth_taker), 6),
    # SOL (Bot 4)
    "Spot SOL":      (_age_h(_sol_spot),  2),
    "OI SOL":        (_age_h(_sol_oi),    6),
    "Taker SOL":     (_age_h(_sol_taker), 6),
}
```

**Nota sobre shadow logs e portfolio_sol.json:**  
`_age_h()` espera DataFrame com coluna `timestamp`. Shadow logs são JSONL e portfolio é JSON. Precisam de helper separado se quiser incluir. Sugestão: verificar apenas se o arquivo existe (sem threshold de staleness), pois crescem somente quando há trades.

Helper para portfolio JSON (se quiser adicionar):
```python
def _portfolio_age_h(path: str) -> float:
    """Retorna age em horas do last_update de um portfolio JSON."""
    p = ROOT / path
    if not p.exists():
        return 9999
    try:
        d = json.loads(p.read_text())
        ts = pd.to_datetime(d.get("last_update"), utc=True)
        return (pd.Timestamp.now(tz="UTC") - ts).total_seconds() / 3600
    except Exception:
        return 9999
```

### Patch em evaluate_g3 (stale guard)

`src/models/gate_scoring.py:174` — adicionar stale check antes de computar:

```python
# ANTES:
def evaluate_g3(zscores: dict, effective_weights: Optional[dict] = None) -> float:
    params = get_params()["gate_params"]
    eff = effective_weights or {}
    g3 = 0.0
    for key, z_col in [...]:

# DEPOIS:
def evaluate_g3(zscores: dict, stale_days: dict = None, effective_weights: Optional[dict] = None) -> float:
    if stale_days and _stale_gate("g3_macro", stale_days):
        logger.warning("G3 macro FRED stale — using 0.0")
        return 0.0
    params = get_params()["gate_params"]
    eff = effective_weights or {}
    g3 = 0.0
    for key, z_col in [...]:
```

E atualizar a chamada em `run_scoring_pipeline()` (linha 441):
```python
# ANTES:
"g3":  evaluate_g3(zscores, ew),

# DEPOIS:
"g3":  evaluate_g3(zscores, stale_days, ew),
```

---

## 4. Recomendações Gerais

| Prioridade | Ação | Impacto |
|------------|------|---------|
| 🔴 Imediato | Investigar causa do FRED 64h stale: `tail -50 logs/daily.log` na AWS | Entender se daily_update.sh falhou |
| 🔴 Imediato | Aplicar patch `stale_checks` no dashboard (adicionar ETH/SOL) | Visibilidade dos bots novos |
| 🟡 Esta semana | Reduzir FRED threshold 48h → 36h no patch | Catch stale antes de 2 ciclos perdidos |
| 🟡 Esta semana | Adicionar `_stale_gate` em `evaluate_g3()` | G3 zeraria em vez de usar dados obsoletos |
| 🟢 Futuro | Dashboard por asset (BTC/ETH/SOL tabs) com System Health por bot | Visão completa do sistema multi-asset |
| 🟢 Futuro | Alert consolidado: se N bots com stale > threshold → push notification | Proatividade vs reatividade |

### Sobre a ausência de ETH/SOL no dashboard atual

O dashboard foi projetado para BTC. A adição de ETH e SOL ao System Health é um pré-requisito para o refactor por tabs. Enquanto não houver tabs, o patch de `stale_checks` cobre o mínimo viável: visibilidade de falha de ingestão dos novos bots.

Os shadow logs (`taker_z_shadow_log.jsonl`, `sol_scoring_shadow_log.jsonl`) não precisam de staleness check — crescem apenas quando trades acontecem, e podem ficar dias sem novas entradas em regime bloqueado. O correto é monitorar apenas se o arquivo existe.
