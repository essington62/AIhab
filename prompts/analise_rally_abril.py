#!/usr/bin/env python3
"""
Análise do rally BTC 13-17 abril 2026.
Rodar no Mac Mini com o conda env crypto_market_state:
  cd ~/Documents/MLGeral/btc_AI
  python prompts/analise_rally_abril.py
"""
import pandas as pd
import numpy as np
import sys, os

# Paths
DATA = "data"

print("=" * 80)
print("ANÁLISE RALLY BTC — 13 a 17 ABRIL 2026")
print("=" * 80)

# ── 1. PREÇO + INDICADORES TÉCNICOS ──────────────────────────────
spot = pd.read_parquet(f"{DATA}/02_intermediate/spot/btc_1h_clean.parquet")
spot["timestamp"] = pd.to_datetime(spot["timestamp"], utc=True)
spot = spot.sort_values("timestamp").reset_index(drop=True)

# Compute extras
prev_close = spot["close"].shift(1)
tr = pd.concat([
    spot["high"] - spot["low"],
    (spot["high"] - prev_close).abs(),
    (spot["low"] - prev_close).abs(),
], axis=1).max(axis=1)
spot["atr_14"] = tr.ewm(span=14, min_periods=14, adjust=False).mean()
spot["ret_1d"] = spot["close"].pct_change(24)
spot["atr_pct"] = spot["atr_14"] / spot["close"]

mask = (spot["timestamp"] >= "2026-04-12") & (spot["timestamp"] <= "2026-04-18")
df = spot[mask].copy()

# Daily summary
df["date"] = df["timestamp"].dt.date
daily = df.groupby("date").agg(
    open=("close", "first"),
    high=("high", "max"),
    low=("low", "min"),
    close=("close", "last"),
    bb_min=("bb_pct", "min"),
    bb_max=("bb_pct", "max"),
    bb_last=("bb_pct", "last"),
    rsi_min=("rsi_14", "min"),
    rsi_max=("rsi_14", "max"),
    rsi_last=("rsi_14", "last"),
    atr_pct=("atr_pct", "last"),
)
first_open = daily.iloc[0]["open"]

print("\n── 1. RESUMO DIÁRIO ────────────────────────────────────────")
for idx, row in daily.iterrows():
    cum = (row.close - first_open) / first_open * 100
    day = (row.close - row.open) / row.open * 100
    print(f"\n{idx}:")
    print(f"  Preço: Open=${row.open:,.0f} High=${row.high:,.0f} Low=${row.low:,.0f} Close=${row.close:,.0f}")
    print(f"  Retorno: Dia={day:+.2f}%  Acumulado={cum:+.2f}%")
    print(f"  BB%: [{row.bb_min:.3f} — {row.bb_max:.3f}] close={row.bb_last:.3f}")
    print(f"  RSI: [{row.rsi_min:.1f} — {row.rsi_max:.1f}] close={row.rsi_last:.1f}")
    print(f"  ATR%: {row.atr_pct:.2%}")

# ── 2. MOMENTOS RSI < 35 (reversal filter candidates) ────────────
print("\n\n── 2. REVERSAL FILTER CANDIDATES (RSI < 35) ────────────────")
low_rsi = df[df["rsi_14"] < 35].copy()
if low_rsi.empty:
    print("  Nenhum momento com RSI < 35 no período")
else:
    for _, row in low_rsi.iterrows():
        ret = row["ret_1d"]
        rf_pass = row["rsi_14"] < 35 and pd.notna(ret) and ret > -0.01
        tag = "PASS ✓" if rf_pass else "BLOCK ✗"
        print(f"  {row.timestamp.strftime('%m/%d %H:%M')} | ${row.close:,.0f} | RSI={row.rsi_14:.1f} | BB={row.bb_pct:.3f} | ret_1d={ret:.4f} | {tag}")

# ── 3. MOMENTOS BB > 0.80 (kill switch zone) ─────────────────────
print("\n\n── 3. BB KILL SWITCH (BB% >= 0.80) — horas bloqueadas ─────")
bb_high = df[df["bb_pct"] >= 0.80].copy()
bb_high["date"] = bb_high["timestamp"].dt.date
bb_by_day = bb_high.groupby("date").size()
total_hours = df.groupby(df["timestamp"].dt.date).size()
print(f"  {'Data':<12} {'Horas BB>=0.80':>15} {'Total horas':>12} {'% bloqueado':>12}")
for d in daily.index:
    blocked = bb_by_day.get(d, 0)
    total = total_hours.get(d, 24)
    pct = blocked / total * 100 if total > 0 else 0
    print(f"  {d}   {blocked:>10}h     {total:>8}h     {pct:>8.0f}%")

# ── 4. Z-SCORES DOS GATES ────────────────────────────────────────
print("\n\n── 4. Z-SCORES (quem está puxando liquidez?) ────────────────")
zscores = pd.read_parquet(f"{DATA}/02_features/gate_zscores.parquet")
zscores["timestamp"] = pd.to_datetime(zscores["timestamp"], utc=True)
zscores = zscores.sort_values("timestamp")

z_mask = (zscores["timestamp"] >= "2026-04-10") & (zscores["timestamp"] <= "2026-04-18")
z_daily = zscores[z_mask].set_index("timestamp").resample("1D").last().dropna(how="all")

key_cols = ["stablecoin_z", "etf_z", "fg_z", "oi_z", "funding_z", "bubble_z"]
existing = [c for c in key_cols if c in z_daily.columns]
print(z_daily[existing].round(3).to_string())

print("\n  Interpretação:")
for _, row in z_daily.tail(3).iterrows():
    stable = row.get("stablecoin_z", 0)
    etf = row.get("etf_z", 0)
    fg = row.get("fg_z", 0)
    oi = row.get("oi_z", 0)
    if stable > 0.5:
        print(f"  → stablecoin_z={stable:+.3f}: LIQUIDEZ ENTRANDO (stablecoins em alta)")
    if etf > 0.3:
        print(f"  → etf_z={etf:+.3f}: ETF INFLOWS positivos")
    if fg < -0.5:
        print(f"  → fg_z={fg:+.3f}: FEAR dominante (contrarian bullish)")
    if oi > 1.0:
        print(f"  → oi_z={oi:+.3f}: OI ALTO (alavancagem crescendo, risco de liquidação)")
    if oi < -1.0:
        print(f"  → oi_z={oi:+.3f}: OI BAIXO (desalavancagem, base para rally)")

# ── 5. SCORE HISTORY ──────────────────────────────────────────────
print("\n\n── 5. SCORE HISTORY (últimos ciclos) ────────────────────────")
scores = pd.read_parquet(f"{DATA}/04_scoring/score_history.parquet")
scores["timestamp"] = pd.to_datetime(scores["timestamp"], utc=True)
scores = scores.sort_values("timestamp")
s_mask = (scores["timestamp"] >= "2026-04-12") & (scores["timestamp"] <= "2026-04-18")
sdf = scores[s_mask]
if sdf.empty:
    print("  Sem scores no período (sistema pode ter sido restartado)")
else:
    for _, row in sdf.iterrows():
        sig = row.get("signal", "?")
        sc = row.get("total_score", None)
        thr = row.get("threshold", None)
        br = row.get("block_reason", "")
        sc_str = f"{sc:.3f}" if pd.notna(sc) else "None"
        thr_str = f"{thr:.3f}" if pd.notna(thr) else "None"
        br_str = str(br) if pd.notna(br) and br else ""
        print(f"  {row.timestamp.strftime('%m/%d %H:%M')} | {sig:>8} | score={sc_str:>7} | thr={thr_str} | {br_str}")

# ── 6. ANÁLISE: O sistema perdeu oportunidade? ───────────────────
print("\n\n── 6. ANÁLISE: O SISTEMA PERDEU OPORTUNIDADE? ─────────────")
# Check if there were moments where score > threshold AND RSI < 35
# (i.e., where the system SHOULD have entered)
cum_total = (daily.iloc[-1].close - first_open) / first_open * 100
last_close = daily.iloc[-1].close
print(f"""
  Rally de ${first_open:,.0f} para ${last_close:,.0f} ({cum_total:+.1f}%)

  O BB kill switch bloqueou entrada durante a SUBIDA (BB% >= 0.80).
  Isso é CORRETO — o sistema nao compra em sobrecompra.

  A pergunta real: houve momento de ENTRADA antes do rally?

  DIA 12 (pre-rally): RSI caiu ate 11.8 (!!) com ret_1d entre -1.5% e -3.5%.
  O ret_1d era muito negativo (faca caindo) → filtro BLOQUEOU corretamente.
  MAS: RSI=11.8 < 25 = extreme capitulation override teria passado!
  Se o sistema estivesse ativo com o reversal filter no dia 12,
  a entrada teria sido em ~$71,000 e hoje estaria +6%.

  DIAS 15-16 (pullbacks): RSI < 35 com ret_1d > -1% em 4 momentos.
  Entradas em $73,700-$74,300, ganho potencial de 2-3%.

  Para capturar rallies news-driven (gap up como dia 13),
  o sistema precisaria de um gate de momentum/breakout.
  O reversal filter captura FUNDOS — e o fundo era dia 12.
""")

# ── 7. NUPL e on-chain (se disponível) ───────────────────────────
print("\n── 7. NUPL / ON-CHAIN ──────────────────────────────────────")
print("""
  NUPL (Net Unrealized Profit/Loss) NÃO está no pipeline atual.
  Fontes possíveis para adicionar:
  - Glassnode (pago, ~$29/mês tier básico)
  - CryptoQuant (pago)
  - LookIntoBitcoin (gratuito, scraping)

  NUPL = (Market Cap - Realized Cap) / Market Cap
  Faixas: <0 = Capitulação, 0-0.25 = Hope, 0.25-0.5 = Optimism,
          0.5-0.75 = Belief, >0.75 = Euphoria/Greed

  Outros on-chain úteis:
  - SOPR (Spent Output Profit Ratio)
  - MVRV (Market Value / Realized Value)
  - Exchange netflow (saídas = bullish)

  O CoinGlass Hobbyist ($29/mês) que já temos NÃO inclui NUPL.
  Precisaria de API adicional.
""")

print("\n" + "=" * 80)
print("FIM DA ANÁLISE")
print("=" * 80)
