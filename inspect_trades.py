#!/usr/bin/env python3
"""
inspect_trades.py — contexto completo de cada trade com dados de mercado.

Uso:
  conda run -n btc_trading_v1 python inspect_trades.py              # todos, últimos 5
  conda run -n btc_trading_v1 python inspect_trades.py --bot bot1
  conda run -n btc_trading_v1 python inspect_trades.py --bot bot2 --n 10
"""
import argparse
import os
import sys
import pandas as pd
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
SEP2 = "═" * 62
SEP  = "─" * 62


# ──────────────────────────────────────────────────────────────
# Utilidades
# ──────────────────────────────────────────────────────────────

def to_utc(val):
    """Converte escalar ou Series para datetime UTC."""
    if isinstance(val, pd.Series):
        return pd.to_datetime(val, utc=True, errors="coerce")
    return pd.to_datetime(val, utc=True)


def find_ts_col(df):
    """Descoberta dinâmica do campo de timestamp."""
    if pd.api.types.is_datetime64_any_dtype(df.index):
        return "__index__"
    for col in ["timestamp", "entry_time", "date", "published", "ts", "time"]:
        if col in df.columns:
            return col
    return None


def nearest_idx(df, target_ts):
    """Índice posicional da linha mais próxima de target_ts em df['_ts']."""
    diffs = (df["_ts"] - target_ts).abs()
    return diffs.idxmin()


def trunc(val, n=80):
    s = str(val) if val is not None else ""
    return s[:n] + "…" if len(s) > n else s


# ──────────────────────────────────────────────────────────────
# Loaders
# ──────────────────────────────────────────────────────────────

def load_trades(bot_filter=None):
    for path in [
        os.path.join(BASE, "data/05_output/trades_history.parquet"),
        os.path.join(BASE, "data/05_output/trades.parquet"),
    ]:
        if os.path.exists(path):
            df = pd.read_parquet(path)
            if bot_filter and "entry_bot" in df.columns:
                df = df[df["entry_bot"] == bot_filter].copy()
            elif bot_filter:
                print(f"  [AVISO] Coluna 'entry_bot' ausente — filtro ignorado.")
            return df, os.path.relpath(path, BASE)
    return None, None


def load_btc_1h():
    path = os.path.join(BASE, "data/01_raw/spot/btc_1h.parquet")
    if not os.path.exists(path):
        return None
    df = pd.read_parquet(path)
    ts_col = find_ts_col(df)
    if ts_col == "__index__":
        df["_ts"] = to_utc(df.index.to_series())
    else:
        df["_ts"] = to_utc(df[ts_col])
    return df.sort_values("_ts").reset_index(drop=True)


def load_score_history():
    path = os.path.join(BASE, "data/04_scoring/score_history.parquet")
    if not os.path.exists(path):
        return None
    df = pd.read_parquet(path)
    ts_col = find_ts_col(df)
    df["_ts"] = to_utc(df[ts_col]) if ts_col != "__index__" else to_utc(df.index.to_series())
    return df.sort_values("_ts").reset_index(drop=True)


def load_news_scores():
    path = os.path.join(BASE, "data/02_features/news_scores.parquet")
    if not os.path.exists(path):
        return None
    df = pd.read_parquet(path)
    ts_col = find_ts_col(df)
    df["_ts"] = to_utc(df[ts_col]) if ts_col != "__index__" else to_utc(df.index.to_series())
    return df.sort_values("_ts").reset_index(drop=True)


# ──────────────────────────────────────────────────────────────
# Colunas relevantes por dataset
# ──────────────────────────────────────────────────────────────

def score_display_cols(df):
    keep = []
    for c in df.columns:
        if c == "_ts":
            continue
        cl = c.lower()
        if any(x in cl for x in [
            "score", "regime", "signal", "block",
            "g0", "g1", "g2", "g3", "g4", "g5", "g6", "g7", "g8", "g9", "g10",
            "cluster", "threshold", "proximity",
        ]):
            keep.append(c)
    return keep if keep else [c for c in df.columns if c != "_ts"]


def news_score_cols(df):
    return [c for c in df.columns if c != "_ts" and (
        "score" in c.lower() or "sentiment" in c.lower()
    )]


# ──────────────────────────────────────────────────────────────
# Formatação de linhas de preço
# ──────────────────────────────────────────────────────────────

def fmt_price(row, label=None):
    ts   = row["_ts"].strftime("%m/%d %H:%M")
    close = row.get("close") if hasattr(row, "get") else row["close"]
    vol   = row.get("volume") if hasattr(row, "get") else row["volume"]
    vol_s = f"  vol={vol:>9.1f}" if vol is not None and pd.notna(vol) else ""
    price_s = f"${close:>10,.2f}" if pd.notna(close) else "   ?"
    label_s = f"  <<< {label}" if label else ""
    return f"  {ts} | {price_s}{vol_s}{label_s}"


def fmt_score_row(row, cols):
    ts = row["_ts"].strftime("%Y-%m-%d %H:%M UTC")
    parts = [f"ts={ts}"]
    for c in cols:
        val = row.get(c)
        if val is None or (isinstance(val, float) and np.isnan(val)):
            continue
        if isinstance(val, float):
            parts.append(f"{c}={val:.3f}")
        else:
            parts.append(f"{c}={val}")
    return "  " + "  |  ".join(parts)


# ──────────────────────────────────────────────────────────────
# Contexto por trade
# ──────────────────────────────────────────────────────────────

def print_trade(idx, trade, btc, scores, news):
    entry_ts  = to_utc(trade.get("entry_time"))
    exit_ts   = to_utc(trade.get("exit_time")) if trade.get("exit_time") else pd.NaT
    bot       = trade.get("entry_bot", "?")
    ret       = trade.get("return_pct", trade.get("return"))
    ret_s     = f"{ret:+.2f}%" if ret is not None and pd.notna(ret) else "?"
    exit_rsn  = trade.get("exit_reason", "?")
    score_adj = trade.get("entry_score_adjusted", trade.get("entry_score_raw"))
    score_s   = f"{score_adj:.3f}" if score_adj is not None and pd.notna(score_adj) else "?"
    ep        = trade.get("entry_price")
    xp        = trade.get("exit_price")
    entry_fmt = entry_ts.strftime("%d/%m %H:%M") if pd.notna(entry_ts) else "?"
    exit_fmt  = exit_ts.strftime("%d/%m %H:%M") if pd.notna(exit_ts) else "?"

    print(f"\n{SEP2}")
    print(f"  TRADE #{idx} — Bot {bot} | entrada: {entry_fmt} | saída: {exit_fmt}")
    print(f"  Resultado: {ret_s} ({exit_rsn}) | Score entrada: {score_s}")
    print(SEP2)

    # ── PREÇO ────────────────────────────────────────────────
    print(f"\n── PREÇO {SEP[8:]}")
    if btc is not None and pd.notna(entry_ts):
        ie = nearest_idx(btc, entry_ts)
        before = btc.iloc[max(0, ie - 10): ie]
        entry_row = btc.iloc[ie]

        if pd.notna(exit_ts):
            ix = nearest_idx(btc, exit_ts)
            during = btc.iloc[ie + 1: ix]
            after  = btc.iloc[ix + 1: ix + 11]
            exit_row = btc.iloc[ix]
        else:
            ix     = ie
            during = pd.DataFrame()
            after  = btc.iloc[ie + 1: ie + 11]
            exit_row = None

        print(f"  [antes — {len(before)} candles]")
        for _, r in before.iterrows():
            print(fmt_price(r))

        ep_lbl = f"ENTRADA @ ${ep:,.0f}" if ep and pd.notna(ep) else "ENTRADA"
        print(fmt_price(entry_row, ep_lbl))

        if len(during) > 0:
            print(f"  [durante — {len(during)} candles]")
            for _, r in during.iterrows():
                print(fmt_price(r))

        if exit_row is not None:
            xp_lbl = f"SAÍDA @ ${xp:,.0f} ({exit_rsn})" if xp and pd.notna(xp) else f"SAÍDA ({exit_rsn})"
            print(fmt_price(exit_row, xp_lbl))

        print(f"  [após — {len(after)} candles]")
        for _, r in after.iterrows():
            print(fmt_price(r))
    else:
        print("  [dados de preço não disponíveis]")

    # ── GATE SCORES ──────────────────────────────────────────
    print(f"\n── GATE SCORES NA ENTRADA {SEP[25:]}")
    if scores is not None and pd.notna(entry_ts):
        scols = score_display_cols(scores)
        ise = nearest_idx(scores, entry_ts)

        before_s = scores.iloc[max(0, ise - 10): ise]
        entry_s  = scores.iloc[ise]

        if pd.notna(exit_ts):
            isx = nearest_idx(scores, exit_ts)
            after_s = scores.iloc[isx + 1: isx + 11]
        else:
            after_s = scores.iloc[ise + 1: ise + 11]

        if len(before_s) > 0:
            print(f"  [10 linhas antes]")
            for _, r in before_s.iterrows():
                print(fmt_score_row(r, scols))
        print(f"  [na entrada]")
        print(fmt_score_row(entry_s, scols))
        if len(after_s) > 0:
            print(f"  [10 linhas após saída]")
            for _, r in after_s.iterrows():
                print(fmt_score_row(r, scols))
    else:
        print("  [dados de gate scores não disponíveis]")

    # ── NEWS ────────────────────────────────────────────────
    print(f"\n── NEWS NA ENTRADA {SEP[18:]}")
    if news is not None and pd.notna(entry_ts):
        ncols = news_score_cols(news)
        inp   = nearest_idx(news, entry_ts)
        rn    = news.iloc[inp]
        ts_n  = rn["_ts"].strftime("%Y-%m-%d %H:%M UTC")
        print(f"  timestamp : {ts_n}")
        for c in ncols:
            val = rn.get(c)
            v_s = f"{val:.4f}" if isinstance(val, float) and pd.notna(val) else str(val)
            print(f"  {c:<20}: {v_s}")
    else:
        print("  [dados de news não disponíveis]")


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Contexto completo de trades BTC")
    parser.add_argument("--bot", default=None, help="Filtrar por bot: bot1 | bot2")
    parser.add_argument("--n",   type=int, default=5, help="Número de trades (default: 5)")
    args = parser.parse_args()

    print(f"\n{SEP2}")
    print("  INSPECT TRADES — contexto completo de mercado")
    print(SEP2)

    trades_df, trades_path = load_trades(args.bot)
    if trades_df is None:
        print("[ERRO] Nenhum arquivo de trades encontrado em data/05_output/.")
        sys.exit(1)

    btc    = load_btc_1h()
    scores = load_score_history()
    news   = load_news_scores()

    print(f"\n  Fonte trades : {trades_path}")
    print(f"  Total trades : {len(trades_df)}" + (f"  (filtro: {args.bot})" if args.bot else ""))
    print(f"  Mostrando   : últimos {min(args.n, len(trades_df))}")
    print(f"  BTC 1h      : {len(btc)} candles" if btc is not None else "  BTC 1h: não disponível")
    print(f"  Scores      : {len(scores)} entradas" if scores is not None else "  Scores: não disponível")
    print(f"  News scores : {len(news)} entradas" if news is not None else "  News scores: não disponível")

    subset = trades_df.tail(args.n).reset_index(drop=True)
    for i, (_, trade) in enumerate(subset.iterrows()):
        print_trade(i + 1, trade, btc, scores, news)

    # ── RESUMO ───────────────────────────────────────────────
    print(f"\n{SEP2}")
    print("  RESUMO ESTATÍSTICO")
    print(SEP2)
    ret_col = next((c for c in ["return_pct", "return"] if c in subset.columns), None)
    if ret_col:
        rets = pd.to_numeric(subset[ret_col], errors="coerce").dropna()
        total = len(rets)
        wins  = (rets > 0).sum()
        wr    = wins / total * 100 if total > 0 else 0
        print(f"  Trades analisados : {total}")
        print(f"  Win Rate          : {wr:.1f}%  ({wins}W / {total - wins}L)")
        print(f"  Retorno médio     : {rets.mean():+.2f}%")
        print(f"  Melhor trade      : {rets.max():+.2f}%")
        print(f"  Pior trade        : {rets.min():+.2f}%")
        if total > 1:
            gross_profit = rets[rets > 0].sum()
            gross_loss   = abs(rets[rets < 0].sum())
            pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")
            print(f"  Profit Factor     : {pf:.2f}")
    else:
        print("  [coluna de retorno não encontrada]")
    print()


if __name__ == "__main__":
    main()
