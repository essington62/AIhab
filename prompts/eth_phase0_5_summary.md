# ETH Phase 0.5 — Edge Validation Summary

**Generated:** 2026-04-20 23:03 UTC

## Baseline

Forward returns on all data:
- 1d: `+0.17%`
- 3d: `+0.55%`
- 7d: `+1.14%`
- 14d: `+2.12%`

## H4 — Volume correlation stability

| Window | Corr | p-value | Significant |
|--------|------|---------|-------------|
| 90d | -0.368 | 0.0006 | ✅ |
| 180d | -0.258 | 0.0006 | ✅ |
| 270d | -0.211 | 0.0006 | ✅ |
| 365d | -0.140 | 0.0111 | ✅ |

**Structural?** ✅ Sign consistent across windows
**All significant?** ✅

## H4 — OI correlation stability

| Window | Corr | p-value | Significant |
|--------|------|---------|-------------|
| 90d | +0.338 | 0.0018 | ✅ |
| 180d | +0.101 | 0.2263 | ❌ |
| 270d | +0.101 | 0.2263 | ❌ |
| 365d | +0.101 | 0.2263 | ❌ |

**Structural?** ✅ Sign consistent

## Backtest — Simple rule

**Rule:** `volume_z < -0.5 AND oi_z > +0.5` (silent accumulation)
**Exit:** 7d TIME / -2% STOP / +4% TARGET

- N trades: 12
- Win rate: 41.7%
- Avg return: +0.88%
- Total return: +9.09%
- Sharpe (annualized): 1.09
- Max drawdown: -8.57%

## Decision Guide

### Proceed to Phase 1 (implement ETH trading) IF:
- ✅ Volume corr significant in 3+ windows
- ✅ Backtest WR > 55%
- ✅ Backtest Sharpe > 1.0
- ✅ Backtest max DD < 15%

### Continue research IF:
- ⚠️ Some signals strong but not all criteria met
- ⚠️ Sinais inconsistentes em diferentes windows

### Abandon ETH IF:
- ❌ Backtest WR < 45%
- ❌ Correlações viram sinal entre windows
- ❌ N trades insuficiente