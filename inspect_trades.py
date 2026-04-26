#!/usr/bin/env python3
"""
inspect_trades.py — ferramenta definitiva de análise de trades AI.hab.

Uso:
  python inspect_trades.py                        # --closed últimos 5
  python inspect_trades.py --closed --bot bot2 --n 5
  python inspect_trades.py --open
  python inspect_trades.py --summary
"""
import argparse
import json
import os
import sys
import pandas as pd
import numpy as np
from datetime import timezone

BASE = os.path.dirname(os.path.abspath(__file__))
SEP2 = "═" * 62
SEP  = "─" * 62


# ──────────────────────────────────────────────────────────────
# Utilidades
# ──────────────────────────────────────────────────────────────

def to_utc(val):
    if isinstance(val, pd.Series):
        return pd.to_datetime(val, utc=True, errors="coerce")
    return pd.to_datetime(val, utc=True)


def find_ts_col(df):
    if pd.api.types.is_datetime64_any_dtype(df.index):
        return "__index__"
    for col in ["timestamp", "entry_time", "date", "published", "ts", "time"]:
        if col in df.columns:
            return col
    return None


def nearest_idx(df, target_ts):
    diffs = (df["_ts"] - target_ts).abs()
    return diffs.idxmin()


def trunc(val, n=80):
    s = str(val) if val is not None else ""
    return s[:n] + "…" if len(s) > n else s


def fmt_pct(val, precision=2):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "?"
    return f"{val:+.{precision}f}%"


def fmt_usd(val):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "?"
    return f"${val:>10,.2f}"


# ──────────────────────────────────────────────────────────────
# Loaders
# ──────────────────────────────────────────────────────────────

def resolve_trades_path():
    """get_path("trades") via config; fallback para paths conhecidos."""
    # Tenta via src.config
    try:
        sys.path.insert(0, BASE)
        from src.config import get_path
        p = get_path("trades")
        if p.exists():
            return str(p), "src.config.get_path('trades')"
    except Exception:
        pass

    # Fallback paths
    for rel in [
        "data/05_output/trades.parquet",
        "data/05_output/trades_history.parquet",
    ]:
        p = os.path.join(BASE, rel)
        if os.path.exists(p):
            return p, f"fallback: {rel}"

    return None, None


def load_trades(bot_filter=None):
    path, source = resolve_trades_path()
    if path is None:
        return None, None, None
    df = pd.read_parquet(path)
    if bot_filter and "entry_bot" in df.columns:
        df = df[df["entry_bot"] == bot_filter].copy()
    elif bot_filter:
        print(f"  [AVISO] Coluna 'entry_bot' ausente — filtro ignorado.")
    return df, os.path.relpath(path, BASE), source


def load_portfolio_state():
    for rel in [
        "data/05_output/portfolio_state.json",
        "data/04_scoring/portfolio_state.json",
    ]:
        p = os.path.join(BASE, rel)
        if os.path.exists(p):
            with open(p) as f:
                return json.load(f), rel
    return None, None


def load_parquet_with_ts(rel_path):
    p = os.path.join(BASE, rel_path)
    if not os.path.exists(p):
        return None
    df = pd.read_parquet(p)
    ts_col = find_ts_col(df)
    if ts_col == "__index__":
        df["_ts"] = to_utc(df.index.to_series())
    elif ts_col:
        df["_ts"] = to_utc(df[ts_col])
    else:
        return None
    return df.sort_values("_ts").reset_index(drop=True)


def load_btc_1h():
    return load_parquet_with_ts("data/01_raw/spot/btc_1h.parquet")


def load_score_history():
    return load_parquet_with_ts("data/04_scoring/score_history.parquet")


def load_news_scores():
    return load_parquet_with_ts("data/02_features/news_scores.parquet")


def load_news_regime():
    return load_parquet_with_ts("data/02_features/news_regime.parquet")


# ──────────────────────────────────────────────────────────────
# Colunas relevantes
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
# Formatação
# ──────────────────────────────────────────────────────────────

def fmt_price(row, label=None):
    ts    = row["_ts"].strftime("%m/%d %H:%M")
    close = row.get("close") if hasattr(row, "get") else row["close"]
    vol   = row.get("volume") if hasattr(row, "get") else row["volume"]
    vol_s   = f"  vol={vol:>9.1f}" if vol is not None and pd.notna(vol) else ""
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
        parts.append(f"{c}={val:.3f}" if isinstance(val, float) else f"{c}={val}")
    return "  " + "  |  ".join(parts)


def bot_stats(df):
    """Retorna dict com métricas de um subset de trades."""
    ret_col = next((c for c in ["return_pct", "return"] if c in df.columns), None)
    if ret_col is None or len(df) == 0:
        return dict(n=0, wr=0, total_ret=0, pf=0, best=0, worst=0)
    rets  = pd.to_numeric(df[ret_col], errors="coerce").dropna()
    n     = len(rets)
    wins  = (rets > 0).sum()
    wr    = wins / n * 100 if n > 0 else 0
    gp    = rets[rets > 0].sum()
    gl    = abs(rets[rets < 0].sum())
    pf    = gp / gl if gl > 0 else float("inf")
    return dict(n=n, wr=wr, total_ret=rets.sum(), pf=pf,
                best=rets.max(), worst=rets.min())


# ──────────────────────────────────────────────────────────────
# Modo --open
# ──────────────────────────────────────────────────────────────

def cmd_open():
    state, state_path = load_portfolio_state()
    if state is None:
        print("[ERRO] portfolio_state.json não encontrado.")
        sys.exit(1)

    btc    = load_btc_1h()
    news   = load_news_scores()
    regime = load_news_regime()

    print(f"\n{SEP2}")
    print("  POSIÇÃO ABERTA — AI.hab")
    print(SEP2)
    print(f"  Fonte: {state_path}")

    has_pos = state.get("has_position", False)
    if not has_pos:
        print("\n  Status: SEM POSIÇÃO")
        cap = state.get("capital_usd")
        if cap:
            print(f"  Capital: {fmt_usd(cap)}")
        sl_time = state.get("last_sl_time")
        if sl_time:
            print(f"  Último SL: {sl_time}")
        cons_sl = state.get("consecutive_sl_count")
        if cons_sl is not None:
            print(f"  SL consecutivos: {cons_sl}")
        print()
        return

    entry_price  = state.get("entry_price")
    entry_time   = state.get("entry_time")
    entry_bot    = state.get("entry_bot", state.get("trade_id", "?"))
    sl_price     = state.get("stop_loss_price")
    tp_price     = state.get("take_profit_price")
    trail_high   = state.get("trailing_high")
    capital      = state.get("capital_usd")
    quantity     = state.get("quantity", 0)
    mae          = state.get("max_adverse", 0) or 0
    mfe          = state.get("max_favorable", 0) or 0

    # Preço atual via btc_1h
    current_price = None
    if btc is not None:
        last_row = btc.iloc[-1]
        current_price = last_row.get("close")

    # Retorno atual
    ret_now = None
    if entry_price and current_price:
        ret_now = (current_price - entry_price) / entry_price * 100

    # Tempo em aberto
    duration_h = None
    if entry_time:
        try:
            et = to_utc(entry_time)
            now = pd.Timestamp.now(tz=timezone.utc)
            duration_h = (now - et).total_seconds() / 3600
        except Exception:
            pass

    # Distância ao SL / TP
    def dist(target, current, capital_usd, qty):
        if target is None or current is None:
            return None, None
        pct = (target - current) / current * 100
        usd = (target - current) * qty if qty else None
        return pct, usd

    sl_pct, sl_usd = dist(sl_price, current_price, capital, quantity)
    tp_pct, tp_usd = dist(tp_price, current_price, capital, quantity)

    # News combined score na entrada
    news_score = None
    if news is not None and entry_time:
        try:
            et = to_utc(entry_time)
            idx = nearest_idx(news, et)
            row = news.iloc[idx]
            for col in ["news_combined_score", "combined_score", "score"]:
                if col in row.index and pd.notna(row[col]):
                    news_score = float(row[col])
                    break
        except Exception:
            pass

    # G2b regime_hint na entrada
    g2b_hint = None
    g2b_conf = None
    if regime is not None and entry_time:
        try:
            et = to_utc(entry_time)
            idx = nearest_idx(regime, et)
            row = regime.iloc[idx]
            g2b_hint = row.get("regime_hint")
            g2b_conf = row.get("confidence")
        except Exception:
            pass

    print(f"\n  Status       : ABERTA")
    print(f"  Bot          : {entry_bot}")
    print(f"  Entrada      : {entry_time}")
    print(f"  Entry price  : {fmt_usd(entry_price)}")
    if duration_h is not None:
        print(f"  Tempo aberto : {duration_h:.1f}h")
    print()
    print(f"  Preço atual  : {fmt_usd(current_price)}")
    print(f"  Retorno atual: {fmt_pct(ret_now)}")
    print(f"  MAE até agora: {fmt_pct(mae * 100 if mae and abs(mae) < 1 else mae)}")
    print(f"  MFE até agora: {fmt_pct(mfe * 100 if mfe and abs(mfe) < 1 else mfe)}")
    print()
    print(f"  Stop Loss    : {fmt_usd(sl_price)}")
    if sl_pct is not None:
        sl_usd_s = f"  ({fmt_usd(sl_usd)})" if sl_usd else ""
        print(f"  Dist. ao SL  : {fmt_pct(sl_pct)}{sl_usd_s}")
    print(f"  Take Profit  : {fmt_usd(tp_price)}")
    if tp_pct is not None:
        tp_usd_s = f"  ({fmt_usd(tp_usd)})" if tp_usd else ""
        print(f"  Dist. ao TP  : {fmt_pct(tp_pct)}{tp_usd_s}")
    print(f"  Trailing high: {fmt_usd(trail_high)}")
    print()
    print(f"  entry_bb_pct : {state.get('entry_bb_pct', '?')}")
    print(f"  entry_rsi    : {state.get('entry_rsi', '?')}")
    if news_score is not None:
        print(f"  news_score   : {news_score:+.4f}")
    if g2b_hint is not None:
        conf_s = f"  (conf={g2b_conf:.2f})" if g2b_conf else ""
        print(f"  G2b regime   : {g2b_hint}{conf_s}")
    print()


# ──────────────────────────────────────────────────────────────
# Modo --summary
# ──────────────────────────────────────────────────────────────

def cmd_summary():
    trades_df, trades_path, source = load_trades()
    state, _ = load_portfolio_state()
    btc = load_btc_1h()

    current_price = None
    if btc is not None:
        current_price = btc.iloc[-1].get("close")

    from datetime import date
    today = date.today().strftime("%Y-%m-%d")

    print(f"\n{SEP2}")
    print(f"  RESUMO GERAL — {today}")
    print(SEP2)

    if trades_df is None:
        print("  [ERRO] Nenhum arquivo de trades encontrado.")
        sys.exit(1)

    print(f"  Fonte: {trades_path}  [{source}]")
    print()

    all_bots = ["bot1", "bot2"]
    if "entry_bot" in trades_df.columns:
        all_bots = sorted(trades_df["entry_bot"].dropna().unique().tolist())

    labels = {
        "bot1": "BOT 1 (Reversal Gate)",
        "bot2": "BOT 2 (Momentum)",
        "bot3": "BOT 3 (ETH Volume)",
    }

    for bot in all_bots:
        sub = trades_df[trades_df["entry_bot"] == bot] if "entry_bot" in trades_df.columns else trades_df
        st  = bot_stats(sub)

        # Último trade
        last_ret   = "—"
        last_date  = "—"
        last_rsn   = "—"
        if len(sub) > 0:
            last = sub.iloc[-1]
            ret_col = next((c for c in ["return_pct", "return"] if c in last.index), None)
            if ret_col:
                last_ret = f"{last[ret_col]:+.2f}%"
            et = last.get("exit_time") or last.get("entry_time")
            if et:
                try:
                    last_date = to_utc(et).strftime("%d/%m")
                except Exception:
                    pass
            last_rsn = last.get("exit_reason", "?")

        label = labels.get(bot, bot.upper())
        pf_s  = f"{st['pf']:.2f}" if st["pf"] != float("inf") else "∞"
        print(f"  {label}:")
        print(f"  Trades: {st['n']} | WR: {st['wr']:.0f}% | Return: {fmt_pct(st['total_ret'])} | PF: {pf_s}")
        print(f"  Último: {last_date} {last_ret} ({last_rsn})")

        # Status posição aberta
        if state and state.get("has_position"):
            ep   = state.get("entry_price")
            ebot = state.get("entry_bot", "?")
            sl   = state.get("stop_loss_price")
            tp   = state.get("take_profit_price")
            if ebot == bot or (bot == "bot2" and "bot2" in str(ebot)):
                ret_now = None
                if ep and current_price:
                    ret_now = (current_price - ep) / ep * 100
                ret_s = fmt_pct(ret_now) if ret_now is not None else "?"
                sl_s  = f"SL {fmt_usd(sl)}" if sl else ""
                tp_s  = f"TP {fmt_usd(tp)}" if tp else ""
                price_s = fmt_usd(current_price) if current_price else "?"
                print(f"  Status: ABERTA @ {fmt_usd(ep)} | {ret_s} atual | {sl_s} | {tp_s}")
            else:
                print(f"  Status: SEM POSIÇÃO")
        else:
            print(f"  Status: SEM POSIÇÃO")
        print()

    # Total
    st_all = bot_stats(trades_df)
    cap = state.get("capital_usd") if state else None
    cap_s = f" | Capital: {fmt_usd(cap)}" if cap else ""
    pf_s = f"{st_all['pf']:.2f}" if st_all["pf"] != float("inf") else "∞"
    print(SEP2)
    print(f"  TOTAL:")
    print(f"  Trades: {st_all['n']} | WR: {st_all['wr']:.0f}% | Return: {fmt_pct(st_all['total_ret'])}{cap_s}")
    print(f"  Melhor: {fmt_pct(st_all['best'])} | Pior: {fmt_pct(st_all['worst'])} | PF: {pf_s}")
    print(SEP2)
    print()


# ──────────────────────────────────────────────────────────────
# Modo --closed (comportamento original aprimorado)
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
            ix       = ie
            during   = pd.DataFrame()
            after    = btc.iloc[ie + 1: ie + 11]
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


def cmd_closed(bot_filter, n):
    trades_df, trades_path, source = load_trades(bot_filter)
    if trades_df is None:
        print("[ERRO] Nenhum arquivo de trades encontrado em data/05_output/.")
        sys.exit(1)

    btc    = load_btc_1h()
    scores = load_score_history()
    news   = load_news_scores()

    print(f"\n{SEP2}")
    print("  INSPECT TRADES — contexto completo de mercado")
    print(SEP2)
    print(f"\n  Fonte trades : {trades_path}  [{source}]")
    print(f"  Total trades : {len(trades_df)}" + (f"  (filtro: {bot_filter})" if bot_filter else ""))
    print(f"  Mostrando    : últimos {min(n, len(trades_df))}")
    print(f"  BTC 1h       : {len(btc)} candles" if btc is not None else "  BTC 1h: não disponível")
    print(f"  Scores       : {len(scores)} entradas" if scores is not None else "  Scores: não disponível")
    print(f"  News scores  : {len(news)} entradas" if news is not None else "  News scores: não disponível")

    subset = trades_df.tail(n).reset_index(drop=True)
    for i, (_, trade) in enumerate(subset.iterrows()):
        print_trade(i + 1, trade, btc, scores, news)

    # Resumo
    print(f"\n{SEP2}")
    print("  RESUMO ESTATÍSTICO")
    print(SEP2)
    st = bot_stats(subset)
    pf_s = f"{st['pf']:.2f}" if st["pf"] != float("inf") else "∞"
    print(f"  Trades analisados : {st['n']}")
    print(f"  Win Rate          : {st['wr']:.1f}%  ({int(st['wr']*st['n']/100)}W / {st['n'] - int(st['wr']*st['n']/100)}L)")
    print(f"  Retorno médio     : {st['total_ret']/st['n']:+.2f}%" if st['n'] > 0 else "")
    print(f"  Melhor trade      : {fmt_pct(st['best'])}")
    print(f"  Pior trade        : {fmt_pct(st['worst'])}")
    print(f"  Profit Factor     : {pf_s}")
    print()


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Ferramenta definitiva de análise de trades AI.hab")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--open",    action="store_true", help="Mostra posição aberta atual")
    group.add_argument("--summary", action="store_true", help="Resumo geral por bot")
    group.add_argument("--closed",  action="store_true", help="Trades fechados (default)")
    parser.add_argument("--bot", default=None, help="Filtrar por bot: bot1 | bot2")
    parser.add_argument("--n",  type=int, default=5, help="Número de trades (default: 5)")
    args = parser.parse_args()

    if args.open:
        cmd_open()
    elif args.summary:
        cmd_summary()
    else:
        cmd_closed(args.bot, args.n)


if __name__ == "__main__":
    main()
