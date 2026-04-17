#!/usr/bin/env python3
"""
Estudo Bot 2 — Momentum/Breakout Strategy
Rodar no Mac Mini com conda env crypto_market_state:
  cd ~/Documents/MLGeral/crypto_v2/crypto-market-state
  python ~/Documents/MLGeral/btc_AI/prompts/estudo_bot2_momentum.py

Usa dados do Data Lake original (mais histórico desde 2024).
"""
import pandas as pd
import numpy as np
import sys, os, warnings
warnings.filterwarnings("ignore")

# ── PATHS (Data Lake original) ─────────────────────────────
# Ajustar se necessário
DL = os.path.expanduser("~/Documents/MLGeral/crypto_v2/crypto-market-state")
AI = os.path.expanduser("~/Documents/MLGeral/btc_AI")

print("=" * 80)
print("ESTUDO BOT 2 — MOMENTUM/BREAKOUT STRATEGY")
print("=" * 80)

# ── 1. CARREGAR DADOS ─────────────────────────────────────
print("\n── 1. CARREGANDO DADOS ──────────────────────────────────────")

# BTC spot daily (Data Lake)
spot_path = f"{DL}/data/01_raw/spot/crypto/daily_24x7/BTCUSDT.parquet"
if not os.path.exists(spot_path):
    # Fallback to AI.hab data
    spot_path = f"{AI}/data/02_intermediate/spot/btc_1h_clean.parquet"
    spot = pd.read_parquet(spot_path)
    spot["timestamp"] = pd.to_datetime(spot["timestamp"], utc=True)
    spot = spot.sort_values("timestamp")
    # Resample to daily
    spot = spot.set_index("timestamp").resample("1D").agg({
        "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"
    }).dropna().reset_index()
    print(f"  Spot (1h→daily resampled): {len(spot)} days")
else:
    spot = pd.read_parquet(spot_path)
    # Data Lake uses 'open_time' instead of 'timestamp'
    if "open_time" in spot.columns:
        spot = spot.rename(columns={"open_time": "timestamp"})
    spot["timestamp"] = pd.to_datetime(spot["timestamp"], utc=True)
    spot = spot.sort_values("timestamp").reset_index(drop=True)
    print(f"  Spot daily (Data Lake): {len(spot)} days, {spot.timestamp.min().date()} to {spot.timestamp.max().date()}")

# Stablecoin mcap
stable_path = f"{AI}/data/01_raw/coinglass/stablecoin_mcap_daily.parquet"
stable = pd.read_parquet(stable_path)
stable["timestamp"] = pd.to_datetime(stable["timestamp"], utc=True)
print(f"  Stablecoin mcap: {len(stable)} days, {stable.timestamp.min().date()} to {stable.timestamp.max().date()}")

# ETF flows
etf_path = f"{AI}/data/01_raw/coinglass/etf_flows_daily.parquet"
etf = pd.read_parquet(etf_path)
etf["timestamp"] = pd.to_datetime(etf["timestamp"], utc=True)
print(f"  ETF flows: {len(etf)} days, {etf.timestamp.min().date()} to {etf.timestamp.max().date()}")

# Fear & Greed
fg_path = f"{AI}/data/01_raw/sentiment/fear_greed_daily.parquet"
fg = pd.read_parquet(fg_path)
fg["timestamp"] = pd.to_datetime(fg["timestamp"], utc=True)
print(f"  Fear & Greed: {len(fg)} days, {fg.timestamp.min().date()} to {fg.timestamp.max().date()}")

# Bubble index
bubble_path = f"{AI}/data/01_raw/coinglass/bubble_index_daily.parquet"
bubble = pd.read_parquet(bubble_path)
bubble["timestamp"] = pd.to_datetime(bubble["timestamp"], utc=True)
print(f"  Bubble index: {len(bubble)} days")

# ── 2. CONSTRUIR FEATURES ─────────────────────────────────
print("\n── 2. CONSTRUINDO FEATURES ──────────────────────────────────")

df = spot[["timestamp", "open", "high", "low", "close", "volume"]].copy()

# Technical
df["ret_1d"] = df["close"].pct_change(1)
df["ret_3d"] = df["close"].pct_change(3)
df["ret_5d"] = df["close"].pct_change(5)
df["ma_7"] = df["close"].rolling(7).mean()
df["ma_21"] = df["close"].rolling(21).mean()
df["ma_50"] = df["close"].rolling(50).mean()
df["ma_200"] = df["close"].rolling(200).mean()

# BB (20d, 2σ)
df["bb_mid"] = df["close"].rolling(20).mean()
df["bb_std"] = df["close"].rolling(20).std()
df["bb_upper"] = df["bb_mid"] + 2 * df["bb_std"]
df["bb_lower"] = df["bb_mid"] - 2 * df["bb_std"]
df["bb_pct"] = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])
df["bb_pct"] = df["bb_pct"].clip(0, 1)

# RSI 14d
delta = df["close"].diff()
gain = delta.clip(lower=0).rolling(14).mean()
loss = (-delta.clip(upper=0)).rolling(14).mean()
rs = gain / loss.replace(0, np.nan)
df["rsi_14"] = 100 - (100 / (1 + rs))

# ATR
prev_close = df["close"].shift(1)
tr = pd.concat([
    df["high"] - df["low"],
    (df["high"] - prev_close).abs(),
    (df["low"] - prev_close).abs(),
], axis=1).max(axis=1)
df["atr_14"] = tr.ewm(span=14, min_periods=14, adjust=False).mean()
df["atr_pct"] = df["atr_14"] / df["close"]

# Volume z-score
df["vol_z"] = (df["volume"] - df["volume"].rolling(30).mean()) / df["volume"].rolling(30).std()

# Above MA
df["above_ma21"] = (df["close"] > df["ma_21"]).astype(int)
df["above_ma50"] = (df["close"] > df["ma_50"]).astype(int)
df["above_ma200"] = (df["close"] > df["ma_200"]).astype(int)

# Merge external data
df["date"] = df["timestamp"].dt.normalize()

# Stablecoin z-score
stable_daily = stable[["timestamp", "stablecoin_mcap_usd"]].copy()
stable_daily["date"] = stable_daily["timestamp"].dt.normalize()
stable_daily["stablecoin_z"] = (
    (stable_daily["stablecoin_mcap_usd"] - stable_daily["stablecoin_mcap_usd"].rolling(30).mean())
    / stable_daily["stablecoin_mcap_usd"].rolling(30).std()
)
df = df.merge(stable_daily[["date", "stablecoin_z"]], on="date", how="left")

# ETF flow z-score (cum 7d)
etf_daily = etf[["timestamp", "etf_flow_usd"]].copy()
etf_daily["date"] = etf_daily["timestamp"].dt.normalize()
etf_daily["etf_cum_7d"] = etf_daily["etf_flow_usd"].rolling(7).sum()
etf_daily["etf_z"] = (
    (etf_daily["etf_cum_7d"] - etf_daily["etf_cum_7d"].rolling(30).mean())
    / etf_daily["etf_cum_7d"].rolling(30).std()
)
df = df.merge(etf_daily[["date", "etf_z", "etf_cum_7d"]], on="date", how="left")

# Fear & Greed
fg_daily = fg[["timestamp", "fg_value"]].copy()
fg_daily["date"] = fg_daily["timestamp"].dt.normalize()
fg_daily["fg_z"] = (
    (fg_daily["fg_value"] - fg_daily["fg_value"].rolling(252, min_periods=30).mean())
    / fg_daily["fg_value"].rolling(252, min_periods=30).std()
)
df = df.merge(fg_daily[["date", "fg_value", "fg_z"]], on="date", how="left")

# Forward returns (for analysis)
df["fwd_1d"] = df["close"].pct_change(1).shift(-1)
df["fwd_3d"] = df["close"].pct_change(3).shift(-3)
df["fwd_5d"] = df["close"].pct_change(5).shift(-5)
df["fwd_7d"] = df["close"].pct_change(7).shift(-7)

# Drop NaN
df = df.dropna(subset=["rsi_14", "bb_pct", "ret_1d", "atr_pct"]).reset_index(drop=True)
print(f"  Dataset final: {len(df)} days, {df.timestamp.min().date()} to {df.timestamp.max().date()}")
print(f"  Com stablecoin_z: {df.stablecoin_z.notna().sum()} days")
print(f"  Com etf_z: {df.etf_z.notna().sum()} days")
print(f"  Com fg_z: {df.fg_z.notna().sum()} days")

# ── 3. DEFINIR ESTRATÉGIAS DE MOMENTUM ────────────────────
print("\n── 3. ESTRATÉGIAS DE MOMENTUM ───────────────────────────────")

def simulate_trades(signals, df, sg_pct, sl_pct, max_hold, trailing_pct=None, label=""):
    """Simula trades não-sobrepostos com SG/SL/trailing/max_hold."""
    trades = []
    i = 0
    while i < len(signals):
        entry_idx = signals.iloc[i]  # index in df
        entry_row = df.loc[entry_idx]
        entry_price = entry_row["close"]
        entry_date = entry_row["timestamp"]

        sg_price = entry_price * (1 + sg_pct)
        sl_price = entry_price * (1 - sl_pct)
        trailing_high = entry_price

        exit_price = None
        exit_reason = None
        exit_date = None

        for j in range(1, max_hold + 1):
            if entry_idx + j >= len(df):
                break
            day = df.iloc[entry_idx + j]
            high = day["high"]
            low = day["low"]
            close = day["close"]

            # Update trailing
            if high > trailing_high:
                trailing_high = high

            # Check SL (intraday low)
            if low <= sl_price:
                exit_price = sl_price
                exit_reason = "stop_loss"
                exit_date = day["timestamp"]
                break

            # Check trailing
            if trailing_pct and trailing_high * (1 - trailing_pct) >= close:
                exit_price = trailing_high * (1 - trailing_pct)
                exit_reason = "trailing_stop"
                exit_date = day["timestamp"]
                break

            # Check SG (intraday high)
            if high >= sg_price:
                exit_price = sg_price
                exit_reason = "stop_gain"
                exit_date = day["timestamp"]
                break

        if exit_price is None:
            # Timeout
            last_idx = min(entry_idx + max_hold, len(df) - 1)
            exit_price = df.iloc[last_idx]["close"]
            exit_reason = "timeout"
            exit_date = df.iloc[last_idx]["timestamp"]

        ret = (exit_price - entry_price) / entry_price
        trades.append({
            "entry_date": entry_date,
            "exit_date": exit_date,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "return_pct": ret,
            "exit_reason": exit_reason,
        })

        # Skip to after exit (no overlap)
        exit_idx = df.index[df["timestamp"] >= exit_date].min() if exit_date is not None else entry_idx + max_hold
        i_next = signals[signals > exit_idx].index
        if len(i_next) > 0:
            i = i_next[0]
        else:
            break

    if not trades:
        print(f"  {label}: 0 trades")
        return pd.DataFrame()

    tdf = pd.DataFrame(trades)
    wins = tdf[tdf["return_pct"] > 0]
    losses = tdf[tdf["return_pct"] <= 0]
    wr = len(wins) / len(tdf) * 100
    avg_win = wins["return_pct"].mean() if len(wins) > 0 else 0
    avg_loss = abs(losses["return_pct"].mean()) if len(losses) > 0 else 0.001
    pf = (len(wins) * avg_win) / (len(losses) * avg_loss) if len(losses) > 0 and avg_loss > 0 else 999
    total = tdf["return_pct"].sum() * 100
    cum = (1 + tdf["return_pct"]).cumprod()
    maxdd = ((cum / cum.cummax()) - 1).min() * 100

    by_reason = tdf.groupby("exit_reason").size().to_dict()

    print(f"  {label}:")
    print(f"    Trades: {len(tdf)} | WR: {wr:.0f}% | PF: {pf:.2f} | Total: {total:+.1f}% | MaxDD: {maxdd:.1f}%")
    print(f"    Avg Win: {avg_win*100:+.2f}% | Avg Loss: {avg_loss*100:-.2f}% | Exits: {by_reason}")

    return tdf


# ── Strategy signals ──────────────────────────────────────

# A. Momentum puro: ret_1d > 2% + above MA21
sig_a = df.index[(df["ret_1d"] > 0.02) & (df["above_ma21"] == 1)]
sig_a_series = pd.Series(sig_a)

# B. Momentum + Liquidez: ret_1d > 2% + stablecoin_z > 1.0
sig_b = df.index[(df["ret_1d"] > 0.02) & (df["stablecoin_z"] > 1.0)]
sig_b_series = pd.Series(sig_b)

# C. Liquidez forte + RSI rising: stablecoin_z > 1.5 + ret_1d > 0 + RSI > 50
sig_c = df.index[(df["stablecoin_z"] > 1.5) & (df["ret_1d"] > 0) & (df["rsi_14"] > 50)]
sig_c_series = pd.Series(sig_c)

# D. ETF inflows + momentum: etf_z > 0.5 + ret_1d > 1%
sig_d = df.index[(df["etf_z"] > 0.5) & (df["ret_1d"] > 0.01)]
sig_d_series = pd.Series(sig_d)

# E. Breakout: BB% > 0.80 + volume spike (vol_z > 1) + ret_1d > 1%
sig_e = df.index[(df["bb_pct"] > 0.80) & (df["vol_z"] > 1.0) & (df["ret_1d"] > 0.01)]
sig_e_series = pd.Series(sig_e)

# F. Combo: stablecoin_z > 1.0 + etf_z > 0.3 + ret_1d > 1% + above MA21
sig_f = df.index[
    (df["stablecoin_z"] > 1.0) & (df["etf_z"] > 0.3) &
    (df["ret_1d"] > 0.01) & (df["above_ma21"] == 1)
]
sig_f_series = pd.Series(sig_f)

# G. Fear reverting to neutral + liquidity: fg_z entre -1 e 0 (saindo do medo) + stablecoin_z > 0.5 + ret_1d > 0
sig_g = df.index[
    (df["fg_z"] > -1.0) & (df["fg_z"] < 0) &
    (df["stablecoin_z"] > 0.5) & (df["ret_1d"] > 0)
]
sig_g_series = pd.Series(sig_g)

# H. Baseline: compra todo dia (random entry)
sig_h = df.index[df["ret_1d"].notna()]
sig_h_series = pd.Series(sig_h)

print("\n  Signal counts:")
print(f"    A. Momentum puro (ret_1d>2% + >MA21):         {len(sig_a)} sinais")
print(f"    B. Momentum+Liquidez (ret>2% + stable_z>1):   {len(sig_b)} sinais")
print(f"    C. Liquidez forte (stable>1.5 + ret>0 + RSI>50): {len(sig_c)} sinais")
print(f"    D. ETF inflows (etf_z>0.5 + ret>1%):          {len(sig_d)} sinais")
print(f"    E. Breakout (BB>0.80 + vol_z>1 + ret>1%):     {len(sig_e)} sinais")
print(f"    F. Combo (stable>1 + etf>0.3 + ret>1% + >MA21): {len(sig_f)} sinais")
print(f"    G. Fear reverting (fg -1..0 + stable>0.5):     {len(sig_g)} sinais")
print(f"    H. Baseline (todos os dias):                    {len(sig_h)} sinais")

# ── 4. SIMULAÇÃO ──────────────────────────────────────────
print("\n── 4. SIMULAÇÃO (SG 3% / SL 2% / max 7d / trailing 1.5%) ──")
configs = [
    (sig_a_series, "A. Momentum puro"),
    (sig_b_series, "B. Momentum+Liquidez"),
    (sig_c_series, "C. Liquidez forte"),
    (sig_d_series, "D. ETF inflows"),
    (sig_e_series, "E. Breakout BB+Vol"),
    (sig_f_series, "F. Combo completo"),
    (sig_g_series, "G. Fear reverting"),
    (sig_h_series, "H. Baseline (todos)"),
]

results = {}
for sig_series, label in configs:
    if len(sig_series) > 0:
        tdf = simulate_trades(sig_series, df, sg_pct=0.03, sl_pct=0.02, max_hold=7, trailing_pct=0.015, label=label)
        results[label] = tdf
    else:
        print(f"  {label}: 0 sinais")

# ── 5. VARIAÇÕES DE STOPS ────────────────────────────────
print("\n── 5. MELHOR ESTRATÉGIA COM DIFERENTES STOPS ────────────────")
# Pick the best strategy and test different stop configs
best_label = None
best_pf = 0
for label, tdf in results.items():
    if len(tdf) >= 5:
        wins = tdf[tdf["return_pct"] > 0]
        losses = tdf[tdf["return_pct"] <= 0]
        if len(losses) > 0:
            pf = (len(wins) * wins["return_pct"].mean()) / (len(losses) * abs(losses["return_pct"].mean()))
            if pf > best_pf:
                best_pf = pf
                best_label = label

if best_label:
    print(f"\n  Melhor estratégia: {best_label} (PF={best_pf:.2f})")
    # Find the original signal series
    sig_map = {
        "A. Momentum puro": sig_a_series,
        "B. Momentum+Liquidez": sig_b_series,
        "C. Liquidez forte": sig_c_series,
        "D. ETF inflows": sig_d_series,
        "E. Breakout BB+Vol": sig_e_series,
        "F. Combo completo": sig_f_series,
        "G. Fear reverting": sig_g_series,
    }
    best_sig = sig_map.get(best_label)
    if best_sig is not None:
        print("\n  Variações:")
        simulate_trades(best_sig, df, 0.02, 0.015, 5, 0.01, f"{best_label} | SG2/SL1.5/5d/trail1%")
        simulate_trades(best_sig, df, 0.03, 0.02, 7, 0.015, f"{best_label} | SG3/SL2/7d/trail1.5%")
        simulate_trades(best_sig, df, 0.04, 0.02, 10, 0.02, f"{best_label} | SG4/SL2/10d/trail2%")
        simulate_trades(best_sig, df, 0.05, 0.03, 14, 0.025, f"{best_label} | SG5/SL3/14d/trail2.5%")
else:
    print("  Nenhuma estratégia com >= 5 trades encontrada")

# ── 6. COMPARAÇÃO COM BOT 1 (Reversal) ──────────────────
print("\n── 6. COMPARAÇÃO COM BOT 1 (Reversal Filter) ────────────────")
# Bot 1 signal: RSI < 35 + ret_1d > -1%
sig_bot1 = df.index[(df["rsi_14"] < 35) & (df["ret_1d"] > -0.01)]
sig_bot1_series = pd.Series(sig_bot1)
print(f"  Bot 1 sinais (RSI<35 + ret_1d>-1%): {len(sig_bot1)}")
simulate_trades(sig_bot1_series, df, 0.02, 0.015, 5, 0.01, "Bot 1 Reversal | SG2/SL1.5/5d")
simulate_trades(sig_bot1_series, df, 0.03, 0.02, 7, 0.015, "Bot 1 Reversal | SG3/SL2/7d")

# ── 7. ANÁLISE DO RALLY 13 ABRIL ─────────────────────────
print("\n── 7. ANÁLISE: QUAIS ESTRATÉGIAS TERIAM ENTRADO NO RALLY? ──")
apr12 = df[df["timestamp"].dt.date == pd.Timestamp("2026-04-12").date()]
apr13 = df[df["timestamp"].dt.date == pd.Timestamp("2026-04-13").date()]

if len(apr12) > 0:
    r = apr12.iloc[-1]
    print(f"\n  04/12 (pre-rally): close=${r.close:,.0f} | RSI={r.rsi_14:.1f} | BB={r.bb_pct:.3f} | ret_1d={r.ret_1d:.4f}")
    print(f"    stablecoin_z={r.stablecoin_z:.3f}" if pd.notna(r.stablecoin_z) else "    stablecoin_z=N/A")
    print(f"    etf_z={r.etf_z:.3f}" if pd.notna(r.etf_z) else "    etf_z=N/A")
    print(f"    fg_z={r.fg_z:.3f}" if pd.notna(r.fg_z) else "    fg_z=N/A")
    print(f"    above_ma21={r.above_ma21} | above_ma200={r.above_ma200}")

    # Check each strategy
    for name, cond in [
        ("A", r.ret_1d > 0.02 and r.above_ma21 == 1),
        ("B", r.ret_1d > 0.02 and (pd.notna(r.stablecoin_z) and r.stablecoin_z > 1.0)),
        ("C", (pd.notna(r.stablecoin_z) and r.stablecoin_z > 1.5) and r.ret_1d > 0 and r.rsi_14 > 50),
        ("D", (pd.notna(r.etf_z) and r.etf_z > 0.5) and r.ret_1d > 0.01),
        ("E", r.bb_pct > 0.80 and r.vol_z > 1.0 and r.ret_1d > 0.01),
        ("F", (pd.notna(r.stablecoin_z) and r.stablecoin_z > 1.0) and (pd.notna(r.etf_z) and r.etf_z > 0.3) and r.ret_1d > 0.01 and r.above_ma21 == 1),
        ("G", (pd.notna(r.fg_z) and -1 < r.fg_z < 0) and (pd.notna(r.stablecoin_z) and r.stablecoin_z > 0.5) and r.ret_1d > 0),
        ("Bot1", r.rsi_14 < 35 and r.ret_1d > -0.01),
    ]:
        print(f"    {name}: {'ENTER ✓' if cond else 'NO ✗'}")

if len(apr13) > 0:
    r = apr13.iloc[-1]
    print(f"\n  04/13 (rally day): close=${r.close:,.0f} | RSI={r.rsi_14:.1f} | BB={r.bb_pct:.3f} | ret_1d={r.ret_1d:.4f}")
    print(f"    stablecoin_z={r.stablecoin_z:.3f}" if pd.notna(r.stablecoin_z) else "    stablecoin_z=N/A")

# ── 8. FORWARD RETURN POR CONDIÇÃO ───────────────────────
print("\n── 8. FORWARD RETURNS POR CONDIÇÃO ──────────────────────────")
conditions = {
    "stablecoin_z > 1.0": df["stablecoin_z"] > 1.0,
    "stablecoin_z > 1.5": df["stablecoin_z"] > 1.5,
    "etf_z > 0.5": df["etf_z"] > 0.5,
    "ret_1d > 2%": df["ret_1d"] > 0.02,
    "ret_1d > 3%": df["ret_1d"] > 0.03,
    "BB% > 0.80": df["bb_pct"] > 0.80,
    "RSI > 60": df["rsi_14"] > 60,
    "above MA200": df["above_ma200"] == 1,
    "fg_z < -0.5": df["fg_z"] < -0.5,
    "Todos os dias": pd.Series(True, index=df.index),
}

print(f"  {'Condição':<25} {'N':>5} {'fwd_1d':>8} {'fwd_3d':>8} {'fwd_5d':>8} {'fwd_7d':>8}")
print("  " + "-" * 65)
for name, mask in conditions.items():
    subset = df[mask]
    if len(subset) < 5:
        continue
    f1 = subset["fwd_1d"].mean() * 100
    f3 = subset["fwd_3d"].mean() * 100
    f5 = subset["fwd_5d"].mean() * 100
    f7 = subset["fwd_7d"].mean() * 100
    print(f"  {name:<25} {len(subset):>5} {f1:>+7.2f}% {f3:>+7.2f}% {f5:>+7.2f}% {f7:>+7.2f}%")

# ── 9. REGRA FINAL REFINADA ──────────────────────────────
print("\n── 9. REGRA FINAL REFINADA (ajustes do usuário) ─────────────")

# Testar 3 thresholds de stablecoin_z com filtro anti-topo
for sz_thr in [1.2, 1.3, 1.5]:
    sig = df.index[
        (df["stablecoin_z"] > sz_thr) &
        (df["ret_1d"] > 0) &
        (df["rsi_14"] > 50) &
        (df["above_ma21"] == 1) &
        (df["bb_pct"] < 0.98)
    ]
    sig_s = pd.Series(sig)
    label = f"Bot2 Final (stable>{sz_thr} + ret>0 + RSI>50 + >MA21 + BB<0.98)"
    if len(sig_s) > 0:
        simulate_trades(sig_s, df, 0.02, 0.015, 5, 0.01, label)
    else:
        print(f"  {label}: 0 sinais")

# ── 10. SIMULAÇÃO COMBINADA BOT1 + BOT2 ─────────────────
print("\n── 10. SIMULAÇÃO COMBINADA Bot1 + Bot2 ──────────────────────")

# Bot 1: Reversal (RSI < 35 + ret_1d > -1%)
# Bot 2: Liquidez (stablecoin_z > 1.3 + ret_1d > 0 + RSI > 50 + >MA21 + BB < 0.98)
# Regra: se Bot1 ativo, Bot2 bloqueado (e vice-versa)

bot1_signals = df.index[(df["rsi_14"] < 35) & (df["ret_1d"] > -0.01)]
bot2_signals = df.index[
    (df["stablecoin_z"] > 1.3) &
    (df["ret_1d"] > 0) &
    (df["rsi_14"] > 50) &
    (df["above_ma21"] == 1) &
    (df["bb_pct"] < 0.98)
]

# Check overlap
bot1_dates = set(df.loc[bot1_signals, "timestamp"].dt.date)
bot2_dates = set(df.loc[bot2_signals, "timestamp"].dt.date)
overlap = bot1_dates & bot2_dates
print(f"  Bot 1 signal days: {len(bot1_dates)}")
print(f"  Bot 2 signal days: {len(bot2_dates)}")
print(f"  Overlap days: {len(overlap)}")
print(f"  Complementaridade: {100 * (1 - len(overlap) / max(len(bot1_dates | bot2_dates), 1)):.0f}%")

# Simulate combined: merge signals, priority Bot1 if overlap
combined_signals = []
bot1_set = set(bot1_signals)
bot2_set = set(bot2_signals)
all_sigs = sorted(bot1_set | bot2_set)

in_trade_until = -1
for idx in all_sigs:
    if idx <= in_trade_until:
        continue
    # Bot 1 has priority
    bot = "Bot1" if idx in bot1_set else "Bot2"
    combined_signals.append({"idx": idx, "bot": bot})
    in_trade_until = idx + 5  # max_hold = 5 days

print(f"\n  Combined trades (non-overlapping): {len(combined_signals)}")
bot1_count = sum(1 for s in combined_signals if s["bot"] == "Bot1")
bot2_count = sum(1 for s in combined_signals if s["bot"] == "Bot2")
print(f"    Bot 1 entries: {bot1_count}")
print(f"    Bot 2 entries: {bot2_count}")

# Simulate combined
combined_idx = pd.Series([s["idx"] for s in combined_signals])
if len(combined_idx) > 0:
    simulate_trades(combined_idx, df, 0.02, 0.015, 5, 0.01, "COMBINED Bot1+Bot2 | SG2/SL1.5/5d/trail1%")

# Compare: Bot1 alone vs Combined
print("\n  ── Comparação direta ──")
simulate_trades(pd.Series(bot1_signals), df, 0.02, 0.015, 5, 0.01, "Bot 1 SOZINHO | SG2/SL1.5/5d/trail1%")

# ── 11. RALLY ABRIL COM REGRA FINAL ─────────────────────
print("\n── 11. RALLY ABRIL: REGRA FINAL TERIA ENTRADO? ──────────────")
for d_str in ["2026-04-11", "2026-04-12", "2026-04-13", "2026-04-14", "2026-04-15", "2026-04-16"]:
    d_rows = df[df["timestamp"].dt.date == pd.Timestamp(d_str).date()]
    if len(d_rows) == 0:
        print(f"  {d_str}: sem dados")
        continue
    r = d_rows.iloc[-1]
    sz = r.stablecoin_z if pd.notna(r.stablecoin_z) else None
    bot2_pass = (
        sz is not None and sz > 1.3 and
        r.ret_1d > 0 and
        r.rsi_14 > 50 and
        r.above_ma21 == 1 and
        r.bb_pct < 0.98
    )
    bot1_pass = r.rsi_14 < 35 and r.ret_1d > -0.01
    tag = ""
    if bot1_pass:
        tag = "Bot1 ENTER ✓"
    elif bot2_pass:
        tag = "Bot2 ENTER ✓"
    else:
        reasons = []
        if sz is None:
            reasons.append("stable=N/A")
        elif sz <= 1.3:
            reasons.append(f"stable={sz:.2f}≤1.3")
        if r.ret_1d <= 0:
            reasons.append(f"ret={r.ret_1d:.3f}≤0")
        if r.rsi_14 <= 50:
            reasons.append(f"RSI={r.rsi_14:.0f}≤50")
        if r.above_ma21 != 1:
            reasons.append("<MA21")
        if r.bb_pct >= 0.98:
            reasons.append(f"BB={r.bb_pct:.3f}≥0.98")
        tag = "BLOCK: " + " & ".join(reasons)

    print(f"  {d_str}: ${r.close:,.0f} | RSI={r.rsi_14:.1f} | BB={r.bb_pct:.3f} | ret={r.ret_1d:+.3f} | stable_z={sz if sz else 'N/A'} | {tag}")

# ── 12. ANÁLISE POR ANO ─────────────────────────────────
print("\n── 12. PERFORMANCE POR ANO (Bot2 regra final) ───────────────")
bot2_final_sig = df.index[
    (df["stablecoin_z"] > 1.3) &
    (df["ret_1d"] > 0) &
    (df["rsi_14"] > 50) &
    (df["above_ma21"] == 1) &
    (df["bb_pct"] < 0.98)
]
bot2_final_series = pd.Series(bot2_final_sig)

# Simulate and get trades
if len(bot2_final_series) > 0:
    trades_bot2 = simulate_trades(bot2_final_series, df, 0.02, 0.015, 5, 0.01, "Bot2 Final (all years)")
    if len(trades_bot2) > 0:
        trades_bot2["year"] = pd.to_datetime(trades_bot2["entry_date"]).dt.year
        print("\n  Por ano:")
        for yr, grp in trades_bot2.groupby("year"):
            wins = grp[grp["return_pct"] > 0]
            losses = grp[grp["return_pct"] <= 0]
            wr = len(wins) / len(grp) * 100
            total = grp["return_pct"].sum() * 100
            avg_w = wins["return_pct"].mean() * 100 if len(wins) > 0 else 0
            avg_l = abs(losses["return_pct"].mean()) * 100 if len(losses) > 0 else 0.001
            pf = (len(wins) * wins["return_pct"].mean()) / (len(losses) * abs(losses["return_pct"].mean())) if len(losses) > 0 else 999
            print(f"    {yr}: {len(grp)} trades | WR={wr:.0f}% | PF={pf:.2f} | Total={total:+.1f}%")

print("\n" + "=" * 80)
print("FIM DO ESTUDO")
print("=" * 80)
