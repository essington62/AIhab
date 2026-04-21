# Filter Validation 1h vs 4h — Comparação Final

**Generated:** 2026-04-21 15:01 UTC
**Dataset:** 136 trades

## 1. Todos os Cenários

| Cenário | N kept | WR% | Expect | Sharpe | Total Ret% | Max DD% | Bloq |
|---------|--------|-----|--------|--------|------------|---------|------|
| BASELINE | 136 | 66.9% | +0.589% | 2.71 | +118.70% | -19.8% | 0 |
| 4h CoinGlass t=0 BIASED < -0.5 | 93 | 75.3% | +0.863% | 4.24 | +120.08% | -10.0% | 43 |
| 4h CoinGlass t=0 BIASED < -0.8 | 101 | 76.2% | +0.845% | 4.23 | +131.60% | -10.0% | 35 |
| 4h CoinGlass t=0 BIASED < -1.0 | 102 | 75.5% | +0.822% | 4.09 | +128.13% | -10.0% | 34 |
| 4h CoinGlass t=0 BIASED < -1.5 | 116 | 72.4% | +0.753% | 3.62 | +135.74% | -11.4% | 20 |
| 4h CoinGlass prev SAFE < -0.5 | 107 | 70.1% | +0.704% | 3.30 | +109.27% | -18.6% | 29 |
| 4h CoinGlass prev SAFE < -0.8 | 116 | 71.6% | +0.749% | 3.55 | +134.47% | -18.6% | 20 |
| 4h CoinGlass prev SAFE < -1.0 | 117 | 71.8% | +0.760% | 3.61 | +139.16% | -18.6% | 19 |
| 4h CoinGlass prev SAFE < -1.5 | 136 | 66.9% | +0.589% | 2.71 | +118.70% | -19.8% | 0 |
| 1h Binance t=0 BIASED < -0.5 | 94 | 71.3% | +0.714% | 3.38 | +93.03% | -10.3% | 42 |
| 1h Binance t=0 BIASED < -0.8 | 105 | 73.3% | +0.794% | 3.83 | +126.86% | -10.3% | 31 |
| 1h Binance t=0 BIASED < -1.0 | 110 | 73.6% | +0.802% | 3.88 | +137.90% | -10.3% | 26 |
| 1h Binance t=0 BIASED < -1.5 | 127 | 68.5% | +0.637% | 2.96 | +120.76% | -16.1% | 9 |
| 1h Binance prev SAFE < -0.5 | 84 | 70.2% | +0.671% | 3.17 | +73.65% | -10.0% | 52 |
| 1h Binance prev SAFE < -0.8 | 102 | 72.5% | +0.757% | 3.63 | +113.40% | -10.0% | 34 |
| 1h Binance prev SAFE < -1.0 | 110 | 72.7% | +0.772% | 3.71 | +130.15% | -11.4% | 26 |
| 1h Binance prev SAFE < -1.5 | 128 | 68.0% | +0.635% | 2.93 | +121.38% | -17.5% | 8 |

## 2. Resumo Executivo

| Métrica | 4h Biased | 4h SAFE | 1h SAFE |
|---------|-----------|---------|---------|
| Best Sharpe | 4.24 | 3.61 | 3.71 |
| N kept | 93.00 | 117.00 | 110.00 |
| WR% | 75.30 | 71.80 | 72.70 |
| Expect% | 0.86 | 0.76 | 0.77 |

## 3. Causal Correlation Test

| Versão | Corr(taker_z, return_pct) | N |
|--------|--------------------------|---|
| taker_z_t0 | 0.2757 | 136 |
| taker_z_prev_4h | 0.1399 | 136 |
| taker_z_1h_t0 | 0.2497 | 136 |
| taker_z_1h_prev | 0.1753 | 136 |

⚠️ taker_z_t0 tem correlação muito maior — look-ahead no 4h.
✅ taker_z_1h_prev correlaciona com retorno: 0.1753

## 4. Sub-Períodos (Robustez)

| Período | Cenário | N | WR% | Sharpe | Bloq |
|---------|---------|---|-----|--------|------|
| T1 | BASELINE | 45 | 60.0% | 1.25 | 0 |
| T1 | 4h SAFE -1.0 | 33 | 69.7% | 3.00 | 12 |
| T1 | 1h SAFE -1.0 | 36 | 66.7% | 2.03 | 9 |
| T2 | BASELINE | 45 | 75.6% | 4.69 | 0 |
| T2 | 4h SAFE -1.0 | 41 | 80.5% | 5.83 | 4 |
| T2 | 1h SAFE -1.0 | 41 | 78.0% | 5.33 | 4 |
| T3 | BASELINE | 46 | 65.2% | 2.45 | 0 |
| T3 | 4h SAFE -1.0 | 43 | 65.1% | 2.37 | 3 |
| T3 | 1h SAFE -1.0 | 33 | 72.7% | 3.79 | 13 |

## 5. 🎯 Veredito

**Baseline:** Sharpe 2.71
**4h CoinGlass biased (t=0):** Sharpe 4.24 (inflação: +0.63)
**4h CoinGlass prev (safe):** Sharpe 3.61 (Δ vs baseline: +0.89)
**1h Binance prev (safe):** Sharpe 3.71 (Δ vs baseline: +0.99)

### ⚖️ EMPATE — manter 4h CoinGlass

Diff 1h vs 4h: +0.10 (não significativo). 4h CoinGlass tem agregação multi-exchange. **Manter arquitetura atual.**

## 6. Visualizações

- `/Users/brown/Documents/MLGeral/btc_AI/prompts/plots/filter_1h_vs_4h/sharpe_1h_vs_4h.png`
- `/Users/brown/Documents/MLGeral/btc_AI/prompts/plots/filter_1h_vs_4h/threshold_sensitivity.png`
- `/Users/brown/Documents/MLGeral/btc_AI/prompts/plots/filter_1h_vs_4h/correlation_4h_vs_1h.png`
- `/Users/brown/Documents/MLGeral/btc_AI/prompts/plots/filter_1h_vs_4h/equity_curves.png`