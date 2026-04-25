═══════════════════════════════════════════════════════
CALIBRAÇÃO HISTÓRICA BTC 2026 — NEWS IMPACT
═══════════════════════════════════════════════════════
Baseline: BTC retorna -0.0183% em 4h sem eventos (drift natural)
Período: 2025-09-01 → 2026-04-25
Artigos analisados: 62 (ex-ante filter aplicado)

Por categoria (edge = retorno ajustado pelo baseline):

FED/FOMC_DOVISH (N=4, acurácia=0%):
  - edge_4h médio: -0.78% | edge_24h: +1.46% | horizonte pico: 24h
  - saturação: N<5 — cada evento mantém força
  - instrução: classificar BEAR se ds_score < -1
  - confidence sugerida: 0.4

FED/FOMC_HAWKISH (N=4, acurácia=100%):
  - edge_4h médio: -0.78% | edge_24h: +1.78% | horizonte pico: 24h
  - saturação: N<5 — cada evento mantém força
  - instrução: classificar BEAR se ds_score < -1
  - confidence sugerida: 0.8

GEO/HORMUZ_BLOCK (N=1, acurácia=N/A (sem ds_score)):
  - edge_4h médio: -0.76% | edge_24h: -2.57% | horizonte pico: 24h
  - saturação: N<5 — cada evento mantém força
  - instrução: classificar BEAR se ds_score < -1
  - confidence sugerida: 0.4

GEO/WAR_ESCALATION (N=1, acurácia=N/A (sem ds_score)):
  - edge_4h médio: -0.61% | edge_24h: -0.71% | horizonte pico: 12h
  - saturação: N<5 — cada evento mantém força
  - instrução: classificar BEAR se ds_score < -1
  - confidence sugerida: 0.4

LIQUIDITY/STABLECOIN (N=1, acurácia=N/A (sem ds_score)):
  - edge_4h médio: -0.35% | edge_24h: +5.05% | horizonte pico: 24h
  - saturação: N<5 — cada evento mantém força
  - instrução: classificar BEAR se ds_score < -1
  - confidence sugerida: 0.4

FED/FOMC_NEUTRAL (N=13, acurácia=0%):
  - edge_4h médio: +0.20% | edge_24h: +0.54% | horizonte pico: 12h
  - saturação: não — cada evento mantém força
  - instrução: classificar SIDEWAYS sinal fraco — manter classificação DeepSeek
  - confidence sugerida: 0.4

RISK/RECESSION (N=9, acurácia=N/A (sem ds_score)):
  - edge_4h médio: -0.11% | edge_24h: +0.77% | horizonte pico: 24h
  - saturação: sim após 3º evento no mesmo episódio
  - instrução: SIDEWAYS no 1º/2º evento, SIDEWAYS no 3º+
  - confidence sugerida: 0.4

GEO/WAR_DEESCALATION (N=4, acurácia=0%):
  - edge_4h médio: +0.10% | edge_24h: +0.64% | horizonte pico: 24h
  - saturação: N<5 — cada evento mantém força
  - instrução: classificar SIDEWAYS sinal fraco — manter classificação DeepSeek
  - confidence sugerida: 0.4

ENERGY/OIL_PRICE (N=7, acurácia=N/A (sem ds_score)):
  - edge_4h médio: -0.07% | edge_24h: +1.68% | horizonte pico: 24h
  - saturação: sim após 3º evento no mesmo episódio
  - instrução: SIDEWAYS no 1º/2º evento, SIDEWAYS no 3º+
  - confidence sugerida: 0.4

OTHER/UNCATEGORIZED (N=18, acurácia=50%):
  - edge_4h médio: +0.04% | edge_24h: -0.80% | horizonte pico: 24h
  - saturação: sim após 3º evento no mesmo episódio
  - instrução: SIDEWAYS no 1º/2º evento, SIDEWAYS no 3º+
  - confidence sugerida: 0.4

Confidence sugerida por acurácia:
  acurácia >= 70% → confidence 0.8
  acurácia 55-70% → confidence 0.6
  acurácia < 55%  → confidence 0.4 (ruído)
═══════════════════════════════════════════════════════