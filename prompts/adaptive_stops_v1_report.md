# Adaptive Stops v1 — Estudo de Validação

**Generated:** 2026-04-21 13:06 UTC
**Sinais analisados:** 136
**Período:** 2026-02-24 → 2026-04-16

## Comparação de Estratégias

| Métrica | FIXED (atual) | ADAPTIVE | Diff |
|---------|---------------|----------|------|
| N trades | 136.000 | 136.000 | +0.000 |
| Win Rate % | 66.912 | 64.706 | -2.206 |
| Avg Return % | 0.589 | 0.429 | -0.160 |
| Sharpe | 2.713 | 2.190 | -0.523 |
| Max DD % | -19.813 | -15.941 | +3.872 |

## Performance por Regime

| Regime | N | Fixed avg% | Adaptive avg% | Diff |
|--------|---|-----------|--------------|------|
| MUCH_CALMER | 16 | +2.00% | +1.07% | -0.93pp |
| CALM | 19 | +0.63% | +0.42% | -0.21pp |
| NORMAL | 85 | +0.27% | +0.29% | +0.02pp |
| ABOVE_NORMAL | 16 | +0.85% | +0.53% | -0.32pp |

## Distribuição de Volatilidade

- NORMAL: 85 (62.5%)
- CALM: 19 (14.0%)
- ABOVE_NORMAL: 16 (11.8%)
- MUCH_CALMER: 16 (11.8%)

## Saídas

### Fixed

- TP: 68 (50.0%)
- SL: 45 (33.1%)
- TRAIL: 23 (16.9%)

### Adaptive

- TP: 63 (46.3%)
- SL: 48 (35.3%)
- TRAIL: 25 (18.4%)

## Veredicto

### FIXED VENCEU

- WR diff: -2.21pp
- Avg return diff: -0.160pp
- Sharpe diff: -0.523

**Recomendação:** manter config fixa (SL 1.5% / TP 2.0%)

## Plots

- `prompts/plots/adaptive_stops_v1/returns_distribution.png`
- `prompts/plots/adaptive_stops_v1/performance_by_regime.png`
- `prompts/plots/adaptive_stops_v1/equity_curve.png`