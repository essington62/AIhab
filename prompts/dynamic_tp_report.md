# Dynamic TP Study — Marujo's Proposal vs Fixed 2%

**Generated:** 2026-04-22 13:16 UTC
**Dataset:** 136 trades (2026)  |  SL: -1.5%, Trail: 1%

## 1. Proposta do Marujo

```python
def get_tp(rsi, bb_pct, volume_z):
    if volume_z > 1.0:               # exhaustion
        return 0.01                  # TP 1%
    elif rsi > 75 and bb_pct > 0.95: # late entry
        return 0.015                 # TP 1.5%
    else:                            # normal
        return 0.02                  # TP 2% (atual)
```

## 2. Distribuição dos Trades por Bucket

| Bucket | N | % | TP | SL | TRAIL | WR original |
|--------|---|---|----|----|-------|-------------|
| normal | 118 | 86.8% | 60 | 39 | 19 | 66.9% |
| late | 1 | 0.7% | 0 | 0 | 1 | 100.0% |
| exhaustion | 17 | 12.5% | 8 | 6 | 3 | 64.7% |

### ⚠️ Observação crítica sobre thresholds

- **Late bucket: 1 trade(s)** — RSI>75 AND BB>0.95 é muito restritivo nos dados de 2026.
  - RSI max dataset: 81.9 | BB max: 0.978
- **Exhaustion bucket: 17 trades** — mais significativo para análise.

## 3. Comparação de Configurações

| Config | N | WR% | Avg Return | Sharpe | Total Return | Max DD | TP Config |
|--------|---|-----|-----------|--------|-------------|--------|-----------|
| Fixed 2% | 136 | 66.9% | +0.589% | **2.713** | +118.70% | -19.8% | `{'exhaustion': 0.02, 'late': 0.02, 'normal': 0.02}` |
| Marujo's v1 | 136 | 66.9% | +0.543% | **2.559** | +105.66% | -19.8% | `{'exhaustion': 0.01, 'late': 0.015, 'normal': 0.02}` |
| Grid-optimized | 136 | 66.9% | +0.586% | **2.715** | +117.70% | -19.8% | `{'exhaustion': np.float64(0.01), 'late': np.float64(0.015), 'normal': 0.02}` |

## 4. Análise por Bucket

| Bucket | N | Sharpe Fixed | Sharpe Marujo | Δ Sharpe | Avg Fixed | Avg Marujo |
|--------|---|-------------|--------------|----------|-----------|-----------|
| normal | 118 | 2.791 | 2.791 | +0.000 → | +0.608% | +0.608% |
| late | 1 | 0.000 | 0.000 | +0.000 → | +0.183% | +0.183% |
| exhaustion | 17 | 2.220 | 0.710 | -1.509 ↓ | +0.486% | +0.118% |

## 5. Mecanismo de Impacto

Como o dynamic TP afeta cada exit_reason:

| Exit | MFE assumido | Efeito com TP < 2% |
|------|-------------|---------------------|
| TP (68 trades) | = return (2%) | TP trades → return cai de 2% para novo_TP |
| SL (45 trades) | = 0 (conservador) | Inalterado |
| TRAIL (23 trades) | ≈ return + 1% (trail pico) | Se MFE ≥ novo_TP → captura mais cedo com return maior |

## 6. Grid Search — Top 10 por Sharpe

| volume_z_thr | rsi_thr | bb_thr | tp_exh_pct | tp_late_pct | n_exh | n_late | sharpe | win_rate | total_return | max_dd |
|---|---|---|---|---|---|---|---|---|---|---|
| 2.0 | 75.0 | 0.85 | 1.0 | 1.5 | 6.0 | 8.0 | 2.7148528966786825 | 66.91176470588235 | 117.7046101125609 | -19.812250980588146 |
| 2.0 | 75.0 | 0.85 | 1.5 | 1.5 | 6.0 | 8.0 | 2.7136547190667715 | 66.91176470588235 | 117.97092001692704 | -19.81225098058817 |
| 2.0 | 78.0 | 0.8 | 1.0 | 1.8 | 6.0 | 6.0 | 2.7129509018330107 | 66.91176470588235 | 117.98709292163561 | -19.812250980588125 |
| 2.0 | 78.0 | 0.85 | 1.0 | 1.8 | 6.0 | 6.0 | 2.7129509018330107 | 66.91176470588235 | 117.98709292163561 | -19.812250980588125 |
| 2.0 | 75.0 | 0.85 | 1.0 | 1.8 | 6.0 | 8.0 | 2.708303852080118 | 66.91176470588235 | 117.55966724923982 | -19.81225098058815 |
| 1.5 | 75.0 | 0.85 | 1.5 | 1.5 | 7.0 | 8.0 | 2.701415083057175 | 66.91176470588235 | 116.90243511488335 | -19.81225098058813 |
| 2.0 | 75.0 | 0.8 | 1.0 | 1.8 | 6.0 | 10.0 | 2.698999913383408 | 66.91176470588235 | 116.70732852979793 | -19.812250980588146 |
| 2.0 | 78.0 | 0.8 | 1.0 | 1.5 | 6.0 | 6.0 | 2.6977711631696892 | 66.91176470588235 | 116.70418981437467 | -19.812250980588143 |
| 2.0 | 78.0 | 0.85 | 1.0 | 1.5 | 6.0 | 6.0 | 2.6977711631696892 | 66.91176470588235 | 116.70418981437467 | -19.812250980588143 |
| 2.0 | 75.0 | 0.85 | 1.2 | 1.5 | 6.0 | 8.0 | 2.696999239178341 | 66.91176470588235 | 116.68432615187534 | -19.812250980588164 |

## 7. Sensibilidade: MFE dos trades SL

E se os SL trades (especialmente em exhaustion) tivessem algum MFE positivo antes de reverter?

| sl_mfe_assumed_pct | sharpe | win_rate_pct | total_return_pct |
|---|---|---|---|
| 0.0 | 2.55857361105618 | 66.91176470588235 | 105.66065085442928 |
| 0.5 | 2.55857361105618 | 66.91176470588235 | 105.66065085442928 |
| 1.0 | 2.55857361105618 | 66.91176470588235 | 105.66065085442928 |
| 1.5 | 2.55857361105618 | 66.91176470588235 | 105.66065085442928 |

## 8. 🎯 Veredicto

- **Fixed 2% (baseline):** Sharpe = 2.713
- **Marujo's v1:** Sharpe = 2.559 (-0.155)
- **Grid-optimized:** Sharpe = 2.715 (+0.002)

### ❌ REJEITAR — Dynamic TP piora
Perda -0.155. Fixed 2% é ótimo para este dataset.

**Raciocínio:**
- Trades TP (68): dynamic TP reduz ganho de 2% → X% — principal driver de perda
- Trades TRAIL (23): dynamic TP pode melhorar captura vs trail
- Trades SL (45): inalterados (MFE desconhecido, assumido 0 — conservador)

## 9. Arquivos

- Plots: `prompts/plots/dynamic_tp/`
- Tables: `prompts/tables/dynamic_tp_*.csv`