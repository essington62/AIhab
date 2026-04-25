"""
btc-trading-v1 Dashboard — CoinGlass style, dark theme.
Run: streamlit run src/dashboard/app.py --server.port 8501
"""

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import yaml
from plotly.subplots import make_subplots

# ---------------------------------------------------------------------------
# Path setup (streamlit runs from project root)
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.config import get_credential, get_params, get_path

try:
    from src.features.fed_observatory import estimate_rate_probability, load_fed_data
    _FED_OBS_AVAILABLE = True
except Exception:
    _FED_OBS_AVAILABLE = False

st.set_page_config(
    page_title="btc-trading-v1",
    page_icon="₿",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items={},
)

# ---------------------------------------------------------------------------
# CSS — Dark theme CoinGlass style
# ---------------------------------------------------------------------------
st.markdown("""
<style>
  /* ── Hide Streamlit chrome that pushes content down ── */
  #MainMenu { visibility: hidden; }
  header[data-testid="stHeader"] { display: none !important; }
  footer { visibility: hidden; }
  div[data-testid="stDecoration"] { display: none !important; }
  div[data-testid="stToolbar"] { display: none !important; }

  .stApp { background-color: #0d1117; color: #e6edf3; }
  section[data-testid="stSidebar"] { background-color: #161b22; }
  /* Zero top padding so header-bar is visible immediately */
  .block-container { padding: 0.5rem 2rem 2rem 2rem !important; max-width: 1600px; }
  h1, h2, h3 { color: #e6edf3; }
  hr { border-color: #30363d; }

  .cg-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 14px 16px;
    margin-bottom: 8px;
  }
  .cg-card-title {
    font-size: 11px;
    color: #8b949e;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 4px;
  }
  .cg-card-value {
    font-size: 22px;
    font-weight: 700;
    line-height: 1.2;
  }
  .cg-card-sub {
    font-size: 12px;
    color: #8b949e;
    margin-top: 3px;
  }
  .cg-card-interp {
    font-size: 12px;
    color: #8b949e;
    margin-top: 6px;
    border-top: 1px solid #30363d;
    padding-top: 6px;
    line-height: 1.5;
  }
  .pos  { color: #3fb950; }
  .neg  { color: #f85149; }
  .warn { color: #d29922; }
  .neut { color: #8b949e; }

  .header-bar {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 10px 20px;
    margin-top: 4px;
    display: flex;
    gap: 24px;
    align-items: center;
    flex-wrap: wrap;
    margin-bottom: 16px;
    font-size: 13px;
  }
  .header-item { color: #8b949e; }
  .header-value { color: #e6edf3; font-weight: 600; }
  .signal-enter  { color: #3fb950; font-weight: 700; }
  .signal-hold   { color: #d29922; font-weight: 700; }
  .signal-block  { color: #f85149; font-weight: 700; }
  .signal-filter { color: #FFA500; font-weight: 700; }

  .score-bar-wrap {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 14px 16px;
    margin-bottom: 12px;
  }

  .news-row {
    border-bottom: 1px solid #21262d;
    padding: 6px 0;
    font-size: 13px;
  }
  .tag-crypto { background:#1f4e7a; color:#79c0ff; border-radius:4px; padding:1px 5px; font-size:10px; }
  .tag-macro  { background:#3a1f5e; color:#d2a8ff; border-radius:4px; padding:1px 5px; font-size:10px; }
  .tag-fed    { background:#4a2010; color:#ffa657; border-radius:4px; padding:1px 5px; font-size:10px; }

  div[data-testid="stMetric"] { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 10px 14px; }
  div[data-testid="stMetric"] label { color: #8b949e !important; font-size: 11px !important; text-transform: uppercase; }
  div[data-testid="stMetricValue"] { color: #e6edf3 !important; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Plotly theme
# ---------------------------------------------------------------------------
PLOTLY = dict(
    template="plotly_dark",
    paper_bgcolor="#0d1117",
    plot_bgcolor="#161b22",
    font=dict(color="#8b949e", size=11),
    margin=dict(l=50, r=20, t=30, b=40),
    xaxis=dict(gridcolor="#21262d", zeroline=False),
    yaxis=dict(gridcolor="#21262d", zeroline=False),
)

GREEN = "#3fb950"
RED   = "#f85149"
AMBER = "#d29922"
GREY  = "#8b949e"
BLUE  = "#58a6ff"

# ---------------------------------------------------------------------------
# Data loading — cached 60s
# ---------------------------------------------------------------------------

@st.cache_data(ttl=60)
def load_portfolio() -> dict:
    path = ROOT / "data/05_output/portfolio_state.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text())

@st.cache_data(ttl=60)
def load_parquet(rel_path: str) -> pd.DataFrame:
    path = ROOT / rel_path
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df

@st.cache_data(ttl=60)
def load_score_history() -> pd.DataFrame:
    return load_parquet("data/04_scoring/score_history.parquet")

@st.cache_data(ttl=60)
def load_gate_zscores() -> pd.DataFrame:
    return load_parquet("data/02_features/gate_zscores.parquet")

@st.cache_data(ttl=60)
def load_regime_history() -> pd.DataFrame:
    return load_parquet("data/03_models/r5c_regime_history.parquet")

@st.cache_data(ttl=60)
def load_spot() -> pd.DataFrame:
    return load_parquet("data/02_intermediate/spot/btc_1h_clean.parquet")

@st.cache_data(ttl=60)
def load_news(category: str) -> pd.DataFrame:
    paths = {
        "crypto": "data/01_raw/news/crypto_news.parquet",
        "macro":  "data/01_raw/news/macro_news.parquet",
        "fed":    "data/01_raw/news/fed_news.parquet",
    }
    return load_parquet(paths.get(category, ""))

@st.cache_data(ttl=60)
def load_fed_calendar() -> dict:
    path = ROOT / "conf/fed_calendar.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text())

@st.cache_data(ttl=300)
def load_params() -> dict:
    return get_params()


@st.cache_data(ttl=3600)
def get_deepseek_balance() -> dict:
    """Query DeepSeek balance API. Cached 1h to avoid hammering on every refresh."""
    try:
        api_key = get_credential("deepseek_api_key")
        resp = requests.get(
            "https://api.deepseek.com/user/balance",
            headers={"Authorization": f"Bearer {api_key}", "Accept": "application/json"},
            timeout=5,
        )
        resp.raise_for_status()
        data = resp.json()
        # Response: {"is_available": bool, "balance_infos": [{"currency": "USD", "total_balance": "...", ...}]}
        is_available = data.get("is_available", False)
        balance_usd = 0.0
        for info in data.get("balance_infos", []):
            if info.get("currency", "").upper() == "USD":
                balance_usd = float(info.get("total_balance", 0))
                break
        return {"available": is_available, "balance_usd": balance_usd, "error": None}
    except Exception as e:
        return {"available": False, "balance_usd": 0.0, "error": str(e)}


@st.cache_data(ttl=60)
def load_sol_trades_json() -> pd.DataFrame:
    """Load SOL trades from JSON (data/05_trades/) and normalize to unified schema."""
    path = ROOT / "data/05_trades/completed_trades_sol.json"
    if not path.exists():
        return pd.DataFrame()
    try:
        raw = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return pd.DataFrame()
    if not raw:
        return pd.DataFrame()

    rows = []
    for t in raw:
        ep = t.get("entry_price", 0) or 0
        rows.append({
            "entry_time":        pd.to_datetime(t.get("entry_timestamp"), utc=True),
            "exit_time":         pd.to_datetime(t.get("exit_timestamp"),  utc=True),
            "entry_price":       ep,
            "exit_price":        t.get("exit_price"),
            "return_pct":        (t.get("pnl_pct") or 0) * 100,
            "exit_reason":       t.get("exit_reason"),
            "stop_loss_price":   round(ep * 0.985, 4) if ep else None,
            "take_profit_price": round(ep * 1.020, 4) if ep else None,
            "trailing_high":     None,
            "pnl_usd":           t.get("pnl_usd", 0),
            "quantity":          t.get("quantity", 0),
            "symbol":            t.get("symbol", "SOLUSDT"),
            "entry_bot":         "bot4",
            "rsi":               (t.get("entry_features") or {}).get("rsi"),
            "stablecoin_z":      None,
        })
    return pd.DataFrame(rows)

def _latest(df: pd.DataFrame, col: str, default=None):
    if df.empty or col not in df.columns:
        return default
    val = df[col].dropna()
    return val.iloc[-1] if not val.empty else default

def _age_h(df: pd.DataFrame) -> float:
    if df.empty or "timestamp" not in df.columns:
        return 9999
    last = df["timestamp"].max()
    return (pd.Timestamp.now(tz="UTC") - last).total_seconds() / 3600

def _color_val(v: float, good_positive: bool = True) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "neut"
    if good_positive:
        return "pos" if v > 0 else ("neg" if v < 0 else "neut")
    else:
        return "neg" if v > 0 else ("pos" if v < 0 else "neut")

def _fmt(v, fmt=".2f", suffix="") -> str:
    if v is None:
        return "—"
    try:
        return f"{v:{fmt}}{suffix}"
    except Exception:
        return str(v)

def _colored(v: float, fmt="+.3f", good_positive=True) -> str:
    c = _color_val(v, good_positive)
    return f'<span class="{c}">{_fmt(v, fmt)}</span>'

# ---------------------------------------------------------------------------
# Gate scoring helpers — compute cluster scores from z-scores
# ---------------------------------------------------------------------------

def _tanh_score(z: float, corr: float, sensitivity: float, max_score: float) -> float:
    if z is None or np.isnan(z):
        return 0.0
    return float(np.clip(corr * np.tanh(z * sensitivity) * max_score, -max_score, max_score))

def compute_clusters(zs: dict, bb_pct: float, rsi_14: float, portfolio: dict) -> dict:
    """Re-compute gate clusters from latest z-scores. Returns cluster scores + details."""
    params = load_params()
    gp = params.get("gate_params", {})

    def g(key): return zs.get(key, 0.0) or 0.0

    # G1 — Technical (bucket, validated)
    bb_cfg = params.get("g1_bb_scores", [])
    rsi_cfg = params.get("g1_rsi_scores", [])
    bb_s = rsi_s = 0.0
    for b in bb_cfg:
        cond = b.get("condition", "")
        thr  = b.get("threshold", 0)
        score = b.get("score", 0)
        if cond == "gte" and bb_pct >= thr:
            bb_s = score; break
        elif cond == "lt" and bb_pct < thr:
            bb_s = score; break
    for b in rsi_cfg:
        cond = b.get("condition", "")
        thr  = b.get("threshold", 0)
        score = b.get("score", 0)
        if cond == "lt" and rsi_14 < thr:
            rsi_s = score; break
        elif cond == "gt" and rsi_14 > thr:
            rsi_s = score; break
    g1 = bb_s + rsi_s

    # G3 macro
    g3_dgs10 = _tanh_score(g("dgs10_z"),  *gp.get("g3_dgs10",  [-0.315, 0.7, 1.0]))
    g3_curve = _tanh_score(g("curve_z"),  *gp.get("g3_curve",  [-0.282, 0.7, 0.8]))
    g3_rrp   = _tanh_score(g("rrp_z"),    *gp.get("g3_rrp",    [ 0.212, 0.7, 0.7]))
    g3_dgs2  = _tanh_score(g("dgs2_z"),   *gp.get("g3_dgs2",   [-0.154, 0.7, 0.5]))
    g3 = g3_dgs10 + g3_curve + g3_rrp + g3_dgs2

    # G4 OI, G9 taker, G10 funding
    g4  = _tanh_score(g("oi_z"),         *gp.get("g4_oi",       [-0.472, 0.8, 2.0]))
    g9  = _tanh_score(g("taker_z"),      *gp.get("g9_taker",    [ 0.143, 0.5, 0.3]))
    g10 = _tanh_score(g("funding_z"),    *gp.get("g10_funding",  [-0.064, 0.4, 0.5]))

    # G5 stable, G7 ETF, G6 bubble, G8 F&G
    g5  = _tanh_score(g("stablecoin_z"), *gp.get("g5_stable",   [ 0.326, 0.6, 1.5]))
    g7  = _tanh_score(g("etf_z"),        *gp.get("g7_etf",       [ 0.263, 0.6, 1.5]))
    g6  = _tanh_score(g("bubble_z"),     *gp.get("g6_bubble",    [-0.345, 0.7, 0.0]))
    g8  = _tanh_score(g("fg_z"),         *gp.get("g8_fg",        [-0.211, 0.7, 0.8]))

    # G2 news
    ns = load_parquet("data/02_features/news_scores.parquet")
    crypto_score = _latest(ns, "crypto_score", 0.0) or 0.0
    macro_score  = _latest(ns, "macro_score",  0.0) or 0.0
    fed_score    = portfolio.get("_fed_score", 0.0)
    g2 = 0.5 * float(crypto_score) + 0.5 * float(fed_score)

    caps = params.get("cluster_caps", {})
    clusters = {
        "technical":   float(np.clip(g1,           *caps.get("technical",   [-2.0,  3.5]))),
        "news":        float(np.clip(g2,           *caps.get("news",        [-1.5,  1.0]))),
        "macro":       float(np.clip(g3,           *caps.get("macro",       [-1.5,  1.0]))),
        "positioning": float(np.clip(g4 + g10,     *caps.get("positioning", [-2.0,  1.5]))),
        "liquidity":   float(np.clip(g5 + g7,      *caps.get("liquidity",   [-1.5,  2.5]))),
        "sentiment":   float(np.clip(g6 + g8 + g9, *caps.get("sentiment",   [-1.5,  1.5]))),
    }
    details = {
        "technical":   {"bb_pct": bb_pct, "rsi_14": rsi_14, "bb_s": bb_s, "rsi_s": rsi_s, "g1_raw": g1},
        "news":        {"crypto_score": crypto_score, "macro_score": macro_score, "fed_score": fed_score, "g2_raw": g2},
        "macro":       {"dgs10_z": g("dgs10_z"), "curve_z": g("curve_z"), "rrp_z": g("rrp_z"), "dgs2_z": g("dgs2_z"),
                        "g3_dgs10": g3_dgs10, "g3_curve": g3_curve, "g3_rrp": g3_rrp, "g3_dgs2": g3_dgs2, "g3_raw": g3},
        "positioning": {"oi_z": g("oi_z"), "funding_z": g("funding_z"), "g4": g4, "g10": g10},
        "liquidity":   {"stablecoin_z": g("stablecoin_z"), "etf_z": g("etf_z"), "g5": g5, "g7": g7},
        "sentiment":   {"bubble_z": g("bubble_z"), "fg_z": g("fg_z"), "taker_z": g("taker_z"), "g6": g6, "g8": g8, "g9": g9},
    }
    return clusters, details

# ---------------------------------------------------------------------------
# Interpretation text
# ---------------------------------------------------------------------------

def interpret_cluster(name: str, score: float, det: dict, zs: dict) -> str:
    """Linha 2 dos cards: narrativa pura, sem repetir números (esses ficam no sub/linha 1)."""
    def z(k): return zs.get(k) or 0.0

    if name == "technical":
        bb = det.get("bb_pct", 0.5)
        rsi = det.get("rsi_14", 50)
        if bb > 0.80:
            return "⛔ Kill switch ativado — preço no topo da banda, zona de sobrecompra extrema"
        if bb < 0.20:
            return "🟢 Base da banda — win rate histórico 88% em 3d"
        if bb < 0.30:
            return "🟢 Próximo da banda inferior — win rate histórico 77%"
        if rsi < 35:
            return "Sobrevenda — mercado stretched pra baixo"
        if rsi > 60:
            return "Momentum comprador elevado — monitorar reversão"
        return "Neutro — preço no meio da banda, sem sinal direcional"

    if name == "positioning":
        oi_z = det.get("oi_z", 0)
        fund_z = det.get("funding_z", 0)
        if oi_z > 2.0:
            return "⚠️ Alavancagem extrema — alto risco de liquidação em cascata se preço ceder"
        if oi_z > 1.0:
            return "Mercado alavancado — risco elevado de long squeeze"
        if oi_z < -1.0:
            return "Mercado desalavancado — base limpa, favorável pra subida"
        if fund_z > 1.5:
            return "Longs pagando muito — mercado crowded, risco de squeeze"
        if fund_z < -1.5:
            return "Shorts pagando — potencial short squeeze"
        return "Alavancagem dentro da normalidade"

    if name == "macro":
        dgs10_z = z("dgs10_z")
        curve_z = z("curve_z")
        if dgs10_z > 1.0 and curve_z < -0.5:
            return "Juros altos com curva invertendo — ambiente restritivo pra risco"
        if dgs10_z > 1.0:
            return "Juros acima da média — pressão sobre ativos de risco"
        if dgs10_z < -1.0:
            return "Juros em queda — expectativa de afrouxamento favorece BTC"
        if curve_z < -1.0:
            return "Curva de juros invertendo — sinal clássico de stress econômico"
        return "Macro dentro da normalidade"

    if name == "liquidity":
        stab_z = z("stablecoin_z")
        etf_z  = z("etf_z")
        if stab_z > 1.0 and etf_z > 0.5:
            return "🟢 Liquidez entrando — stablecoins crescendo e ETFs com inflows"
        if stab_z < -1.0:
            return "Stablecoins contraindo — capital saindo do ecossistema, menos combustível"
        if etf_z > 1.0:
            return "ETF inflows forte — institucional comprando"
        if etf_z < -1.0:
            return "ETF outflows — institucional reduzindo exposição"
        return "Liquidez dentro da normalidade"

    if name == "sentiment":
        taker_z  = z("taker_z")
        bubble_z = z("bubble_z")
        fg_raw   = det.get("fg_raw")  # raw 0-100 value from sentiment parquet
        if taker_z < -2.0:
            return "⚠️ Pressão vendedora extrema nos futuros — retail vendendo agressivamente"
        if taker_z > 2.0:
            return "Compra agressiva nos futuros — momentum comprador forte"
        if bubble_z > 1.5:
            return "Bubble index overextended — mercado sobreaquecido, risco de correção"
        # F&G text driven by raw value (0-100 fixed scale, not z-score)
        if fg_raw is not None:
            if fg_raw < 25:
                return "😱 Extreme Fear — contrarian bullish (retail aterrorizado)"
            if fg_raw < 45:
                return "😰 Fear — cautela acumulando, potencial oportunidade contrarian"
            if fg_raw < 55:
                return "😐 Neutro"
            if fg_raw < 75:
                return "😏 Greed — otimismo moderado, monitorar"
            return "🤑 Extreme Greed — contrarian bearish, risco de correção"
        # Fallback to z-score if raw not available
        if z("fg_z") < -1.0:
            return "Fear & Greed em medo — contrarian bullish"
        if z("fg_z") > 1.5:
            return "Greed elevado — mercado complacente, cautela"
        return "Sentimento neutro"

    if name == "news":
        cs = det.get("crypto_score", 0)
        ms = det.get("macro_score", 0)
        fs = det.get("fed_score", 0)
        if cs > 3.0:
            return "🟢 Notícias crypto muito bullish — catalisador positivo no curto prazo"
        if cs < -3.0:
            return "🔴 Notícias crypto bearish — sentimento negativo recente"
        if fs < -1.0:
            return "Fed hawkish — linguagem restritiva nos discursos recentes"
        if ms > 1.0:
            return "Macro positiva — ambiente macro favorável"
        return "Notícias mistas sem viés claro"

    return ""

# ---------------------------------------------------------------------------
# Fed Sentinel helpers
# ---------------------------------------------------------------------------

def get_fomc_proximity(calendar: dict) -> dict:
    today = pd.Timestamp.now(tz="UTC").date()
    result = {"next_event": None, "days_away": 9999, "event_type": None, "proximity_adj": 0.0, "in_blackout": False}
    events = calendar.get("fomc_dates", []) + calendar.get("hearings", []) + calendar.get("transitions", [])
    upcoming = []
    for ev in events:
        raw_date = ev.get("date") or ev.get("start") or ev.get("decision_date")
        if not raw_date:
            continue
        try:
            ev_date = pd.Timestamp(raw_date).date()
        except Exception:
            continue
        days = (ev_date - today).days
        if -3 <= days <= 30:
            upcoming.append((days, ev))
    if not upcoming:
        return result
    upcoming.sort(key=lambda x: abs(x[0]))
    days, ev = upcoming[0]
    result["days_away"] = days
    result["next_event"] = ev.get("description") or ev.get("type") or str(ev)
    result["event_type"] = ev.get("type", "fomc")
    adj = 0.0
    if -2 <= days <= 0:  adj = 1.5
    elif -5 <= days <= -3: adj = 0.7
    elif days == 1:      adj = 0.3
    result["proximity_adj"] = adj
    # blackout: T-10 to T-2
    result["in_blackout"] = -10 <= days <= -2
    return result

# ---------------------------------------------------------------------------
# Whale signal
# ---------------------------------------------------------------------------

def whale_signal(ls_account: float, price_chg_24h: float) -> tuple[str, str]:
    if ls_account < 0.95 and price_chg_24h > 0:
        return "🔴 DISTRIBUIÇÃO", "Baleias vendendo no rally — divergência bearish"
    if ls_account > 1.10 and price_chg_24h < 0:
        return "🟢 ACUMULAÇÃO", "Baleias comprando na queda — divergência bullish"
    if ls_account > 1.05:
        return "🟢 Baleias long", f"L/S Accounts={ls_account:.3f} — mais contas long que short entre top traders"
    if ls_account < 0.95:
        return "🔴 Baleias short", f"L/S Accounts={ls_account:.3f} — mais contas short entre top traders"
    return "⚪ Neutro", f"L/S Accounts={ls_account:.3f} — equilíbrio entre longs e shorts"

# ---------------------------------------------------------------------------
# Helpers for latest hourly log
# ---------------------------------------------------------------------------

def compute_bot_metrics(trades_df: pd.DataFrame) -> dict:
    """Compute trading metrics (WR, PF, DD, etc.) for a set of completed trades."""
    if trades_df.empty or "return_pct" not in trades_df.columns:
        return {
            "n_trades": 0, "wins": 0, "losses": 0,
            "win_rate": 0.0, "profit_factor": 0.0,
            "total_return": 0.0, "max_drawdown": 0.0,
            "avg_win": 0.0, "avg_loss": 0.0, "avg_duration_hours": 0.0,
        }
    returns = trades_df["return_pct"].astype(float)
    n = len(returns)
    wins    = int((returns > 0).sum())
    losses  = int((returns <= 0).sum())
    win_rate = wins / n * 100 if n > 0 else 0.0
    gross_profit = float(returns[returns > 0].sum())
    gross_loss   = float(abs(returns[returns <= 0].sum()))
    if gross_loss > 0:
        profit_factor = round(gross_profit / gross_loss, 2)
    elif gross_profit > 0:
        profit_factor = 99.0
    else:
        profit_factor = 0.0
    total_return = round(float(((1 + returns / 100).prod() - 1) * 100), 2)
    equity       = (1 + returns / 100).cumprod()
    rolling_max  = equity.cummax()
    max_drawdown = round(float(((equity / rolling_max) - 1).min() * 100), 2)
    avg_win  = round(float(returns[returns > 0].mean()), 2) if wins > 0 else 0.0
    avg_loss = round(float(returns[returns <= 0].mean()), 2) if losses > 0 else 0.0
    avg_dur  = 0.0
    if "duration_hours" in trades_df.columns:
        avg_dur = round(float(trades_df["duration_hours"].mean()), 1)
    return {
        "n_trades": n, "wins": wins, "losses": losses,
        "win_rate": round(win_rate, 1),
        "profit_factor": profit_factor,
        "total_return": total_return,
        "max_drawdown": max_drawdown,
        "avg_win": avg_win, "avg_loss": avg_loss,
        "avg_duration_hours": avg_dur,
    }


def get_last_cycle_info() -> dict:
    log_dir = ROOT / "logs"
    if not log_dir.exists():
        return {}
    logs = sorted(log_dir.glob("hourly_*.log"), reverse=True)
    if not logs:
        return {}
    try:
        text = logs[0].read_text()
        last_line = text.strip().split("\n")[-1]
        mtime = datetime.fromtimestamp(logs[0].stat().st_mtime, tz=timezone.utc)
        age_m = (datetime.now(timezone.utc) - mtime).total_seconds() / 60
        return {"last_log": logs[0].name, "age_min": age_m, "last_line": last_line}
    except Exception:
        return {}

# ---------------------------------------------------------------------------
# DeepSeek AI Analyst
# ---------------------------------------------------------------------------

_DEEPSEEK_SYSTEM_PROMPT = """Você é o analista quantitativo do sistema AI.hab. Recebe um snapshot do gate scoring v2 e produz leitura operacional em 3 seções, em texto plano.

REGRAS
- Proibido recapitular o que já está nos cards do dashboard (regime, cluster scores, macro z-scores, sentiment, news scores). O usuário já vê tudo isso. Foque APENAS no que os dados combinados implicam para estrutura de preço e gatilhos de sinal.
- Sem markdown. Sem asteriscos, hashtags, bullets com asterisco. Use apenas quebras de linha e numeração simples (1., 2., 3.) para subitens.
- Exija números concretos. Nunca escreva "perto da resistência" sem mencionar o valor em dólares. Nunca "melhora do momentum" sem threshold específico.
- Linguagem probabilística. Evite determinismos. Use "sugere", "indica", "é consistente com".
- Resposta entre 300 e 500 palavras total. Se dados incompletos, seja mais curto e sinalize.
- Português do Brasil. Sem emojis. Sem hype.

ESTRUTURA OBRIGATÓRIA

SEÇÃO 1 — LEITURA DO SINAL ATUAL (2-4 linhas)
Sintetize a decisão do sistema: sinal, score ajustado vs threshold, e a tensão narrativa central (por que o score está onde está — qual cluster domina, o que está travando ou liberando o sinal). Se kill switch ativo, por que domina. Se HOLD, qual cluster ou regime impede ENTER.

SEÇÃO 2 — SUPORTES E RESISTÊNCIAS
Seis linhas, uma por nível. Use exatamente este formato:
Resistência imediata: $XX,XXX — [origem: BB upper / high 7d / etc]
Resistência estrutural: $XX,XXX — [origem]
Suporte imediato: $XX,XXX — [origem: MA50 / BB middle / etc]
Suporte estrutural: $XX,XXX — [origem: MA200 / low 7d / etc]
Zona de invalidação (bearish): $XX,XXX — [condição: fechamento 1h abaixo de X invalida estrutura]
ATR(14): $XXX — amplitude média dos movimentos horários

SEÇÃO 3 — GATILHOS PARA MUDAR O SINAL
Para HOLD → ENTER: duas condições simultâneas mensuráveis (preço, RSI, z-score, regime).
Para HOLD → BLOCK: duas condições que ativariam kill switch ou ruptura de suporte.
Cada condição em uma linha, com threshold numérico explícito.

CONTEXTO
Gate scoring v2: 11 gates, 6 clusters. Score = Σ clusters × G0 (Bear=0, Sideways={sw_mult}×, Bull=1.0). ENTER se score ≥ threshold. Kill switches forçam BLOCK: BB_TOP≥0.80, OI z>2.5, NEWS_BEAR<-3, FED_HAWKISH (fed_score<-1 + FOMC≤T-2), BEAR_REGIME."""


def call_deepseek_analyst(context: dict) -> str:
    try:
        api_key = get_credential("deepseek_api_key")
    except Exception:
        return "DeepSeek API key não configurado."

    # ── Sinal e score ────────────────────────────────────────────────────────
    score_raw   = context.get("score_raw", 0) or 0
    mult        = context.get("regime_multiplier", 1.0) or 1.0
    score_adj   = context.get("score", 0) or 0
    block_rsn   = context.get("block_reason") or "nenhum"

    # ── MA200 ────────────────────────────────────────────────────────────────
    ma200_val   = context.get("ma200_val")
    ma200_pct   = context.get("ma200_pct")
    ma200_slope = context.get("ma200_slope")
    ma200_line  = (f"MA200: ${ma200_val:,.0f} ({ma200_pct:+.1f}% do preço, "
                   f"slope 5h {ma200_slope:+.0f})")  if ma200_val else "MA200: indisponível"

    # ── BB ───────────────────────────────────────────────────────────────────
    bb_upper    = context.get("bb_upper")
    bb_middle   = context.get("bb_middle")
    bb_lower    = context.get("bb_lower")
    bb_line     = (f"BB: lower=${bb_lower:,.0f} / mid=${bb_middle:,.0f} / upper=${bb_upper:,.0f} / "
                   f"pct={context.get('bb_pct',0):.3f}")  if bb_upper else \
                  f"BB: pct={context.get('bb_pct',0):.3f} (valores absolutos indisponíveis)"

    # ── MAs ──────────────────────────────────────────────────────────────────
    _ma50  = context.get("ma50")
    _ma100 = context.get("ma100")
    ma_line = "  ".join(
        f"MA{w}: ${v:,.0f}" for w, v in [("50", _ma50), ("100", _ma100), ("200", context.get("ma200_val"))]
        if v is not None
    ) or "MAs: indisponíveis"

    # ── Rolling High/Low ─────────────────────────────────────────────────────
    _h7  = context.get("high_7d");  _l7  = context.get("low_7d")
    _h30 = context.get("high_30d"); _l30 = context.get("low_30d")
    _atr = context.get("atr_14")
    sr_line = ""
    if _h7:
        sr_line += f"Range 7d:  ${_l7:,.0f} — ${_h7:,.0f}\n"
    if _h30:
        sr_line += f"Range 30d: ${_l30:,.0f} — ${_h30:,.0f}\n"
    if _atr:
        sr_line += f"ATR(14):   ${_atr:,.0f}"
    if not sr_line:
        sr_line = "Ranges: indisponíveis"

    # ── Fed events ≤14d ──────────────────────────────────────────────────────
    fed_events_14d = context.get("fed_events_14d", [])
    fed_events_str = "\n  ".join(fed_events_14d) if fed_events_14d else "Nenhum evento Fed nos próximos 14 dias"

    # ── Fed Observatory ──────────────────────────────────────────────────────
    fed_obs     = context.get("fed_obs", {})
    if fed_obs:
        fed_obs_line = (f"Prob corte: {fed_obs.get('prob_cut',0):.0%} | "
                        f"Manutenção: {fed_obs.get('prob_hold',0):.0%} | "
                        f"Alta: {fed_obs.get('prob_hike',0):.0%} "
                        f"(confiança: {fed_obs.get('confidence','?')})")
    else:
        fed_obs_line = "Fed Observatory: indisponível"

    user_prompt = f"""=== SNAPSHOT AI.hab — {context.get('ts', 'agora')} ===

## PREÇO E REGIME
Preço BTC: ${context.get('price', 0):,.0f} ({context.get('pct_24h', 0):+.2f}% 24h)
Regime R5C: {context.get('regime', 'N/A')}
RSI(14): {context.get('rsi', 50):.1f} | BB pct: {context.get('bb_pct', 0.5):.3f}
{bb_line}
{ma_line}
{ma200_line}

## SUPORTE E RESISTÊNCIA
{sr_line}

## SINAL E SCORE
Score bruto (Σ clusters): {score_raw:+.4f}
Multiplicador G0 ({context.get('regime','?')}): ×{mult}
Score ajustado: {score_adj:+.4f}
Threshold: {context.get('threshold', 3.5):.3f}
Decisão: **{context.get('signal', 'N/A')}**
Kill switch ativo: {block_rsn}

## CLUSTERS (contribuição ao score)
  Technical:   {context.get('c_technical', 0):+.3f}
  Positioning: {context.get('c_positioning', 0):+.3f}
  Macro:       {context.get('c_macro', 0):+.3f}
  Liquidity:   {context.get('c_liquidity', 0):+.3f}
  Sentiment:   {context.get('c_sentiment', 0):+.3f}
  News:        {context.get('c_news', 0):+.3f}

## DERIVATIVOS E POSICIONAMENTO
OI z-score:      {context.get('oi_z', 0):.2f}
Funding z-score: {context.get('funding_z', 0):.2f}
Taker z-score:   {context.get('taker_z', 0):.2f}
L/S top accounts:  {context.get('ls_account', 1):.3f}
L/S top positions: {context.get('ls_position', 1):.3f}
Whale signal: {context.get('whale_signal', 'N/A')}

## LIQUIDEZ E FLUXOS
Stablecoin mcap z-score: {context.get('stablecoin_z', 0):.2f}
ETF flows 7d z-score:    {context.get('etf_z', 0):.2f}

## MACRO (z-scores)
DGS10:      {context.get('dgs10_z', 0):.2f}
DGS2:       {context.get('dgs2_z', 0):.2f}
Curva 2y10y:{context.get('curve_z', 0):.2f}
RRP:        {context.get('rrp_z', 0):.2f}

## FED
Próximo evento: {context.get('fed_event', 'N/A')} em {context.get('fed_days', '?')} dias
Proximity adj threshold: {context.get('fed_adj', 0):+.1f}
{fed_obs_line}
Eventos Fed ≤14 dias:
  {fed_events_str}

## SENTIMENTO
Fear & Greed: {context.get('fg_val', 0):.0f}/100 ({context.get('fg_cls', 'N/A')}) | z-score: {context.get('fg_z', 0):.2f}
Bubble index z-score: {context.get('bubble_z', 0):.2f}
News crypto score:    {context.get('crypto_news_score', 0):+.2f}
News fed score:       {context.get('fed_news_score', 0):+.2f}

---
Gere a análise estruturada em 6 seções conforme instruído no system prompt."""

    try:
        resp = requests.post(
            "https://api.deepseek.com/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": "deepseek-chat",
                "max_tokens": 2000,
                "temperature": 0.3,
                "top_p": 0.9,
                "messages": [
                    {"role": "system", "content": _DEEPSEEK_SYSTEM_PROMPT.format(
                        sw_mult=float(load_params().get("sideways_multiplier", 0.5)))},
                    {"role": "user",   "content": user_prompt},
                ],
            },
            timeout=45,
        )
        resp.raise_for_status()
        data    = resp.json()
        content = data["choices"][0]["message"]["content"]
        usage   = data.get("usage", {})
        finish  = data["choices"][0].get("finish_reason", "")
        comp_tokens = usage.get("completion_tokens", 0)
        if finish == "length" or comp_tokens >= 1800:
            _warn = (
                f"DeepSeek truncation: finish_reason={finish}, "
                f"completion_tokens={comp_tokens}, max_tokens=2000"
            )
            import logging, pathlib, datetime
            logging.getLogger("dashboard.deepseek").warning(_warn)
            try:
                _log_dir = pathlib.Path("/app/logs") if pathlib.Path("/app").exists() \
                           else pathlib.Path("logs")
                _log_dir.mkdir(parents=True, exist_ok=True)
                _ts = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
                with open(_log_dir / "deepseek_truncation.log", "a") as _fh:
                    _fh.write(f"{_ts}  {_warn}\n")
            except Exception:
                pass
            if finish == "length":
                content = "⚠️ *Análise pode estar incompleta — max_tokens atingido.*\n\n" + content
        return content
    except Exception as e:
        return f"Erro ao chamar DeepSeek: {e}"

# ===========================================================================
# MAIN APP
# ===========================================================================

def main():
    # ── Load all data (cached 60s — spinner visible on first cold start) ──
    with st.spinner("Carregando dados..."):
        portfolio  = load_portfolio()
        zs_df      = load_gate_zscores()
        spot_df    = load_spot()
        regime_df  = load_regime_history()
        score_hist = load_score_history()
        macro_df   = load_parquet("data/02_intermediate/macro/fred_daily_clean.parquet")
        vix_df     = load_parquet("data/01_raw/market/vix_daily.parquet")
        dxy_df     = load_parquet("data/01_raw/market/dxy_daily.parquet")
        oil_df     = load_parquet("data/01_raw/market/oil_daily.parquet")
        sp500_df   = load_parquet("data/01_raw/market/sp500_daily.parquet")
        oi_df      = load_parquet("data/02_intermediate/futures/oi_1h_clean.parquet")
        taker_df   = load_parquet("data/02_intermediate/futures/taker_1h_clean.parquet")
        funding_df = load_parquet("data/02_intermediate/futures/funding_1h_clean.parquet")
        lsa_df     = load_parquet("data/01_raw/futures/ls_account_1h.parquet")
        lsp_df     = load_parquet("data/01_raw/futures/ls_position_1h.parquet")
        liq_df     = load_parquet("data/01_raw/coinglass/liquidations_4h.parquet")
        ob_df      = load_parquet("data/01_raw/coinglass/orderbook_4h.parquet")
        ob_agg_df  = load_parquet("data/01_raw/coinglass/orderbook_agg_4h.parquet")
        fg_df      = load_parquet("data/01_raw/sentiment/fear_greed_daily.parquet")
        etf_df     = load_parquet("data/01_raw/coinglass/etf_flows_daily.parquet")
        bubble_df  = load_parquet("data/01_raw/coinglass/bubble_index_daily.parquet")
        ns_df      = load_parquet("data/02_features/news_scores.parquet")
        cal        = load_fed_calendar()

    # ── Current values ─────────────────────────────────────────────────────
    price      = _latest(spot_df, "close", 0.0)
    bb_pct     = _latest(spot_df, "bb_pct", 0.5)
    rsi_14     = _latest(spot_df, "rsi_14", 50.0)
    regime     = portfolio.get("last_regime") or _latest(regime_df, "regime", "Unknown")
    signal     = portfolio.get("last_signal", "—")
    score      = portfolio.get("last_score") or 0.0
    threshold  = portfolio.get("last_threshold") or 3.5
    capital    = portfolio.get("total_capital_usd", portfolio.get("capital_usd", 10000.0))

    # 24h price change
    if len(spot_df) > 24:
        pct_24h = (price - spot_df["close"].iloc[-25]) / spot_df["close"].iloc[-25] * 100
    else:
        pct_24h = 0.0

    # MA200 info (from 1h spot — enough rows for rolling 200)
    ma200_val = None; ma200_pct = None; ma200_slope = None
    if not spot_df.empty and len(spot_df) >= 200:
        close_s = spot_df["close"].astype(float)
        ma200_s = close_s.rolling(200).mean()
        ma200_val = float(ma200_s.iloc[-1])
        ma200_pct = (price / ma200_val - 1) * 100
        ma200_slope = float(ma200_s.iloc[-1] - ma200_s.iloc[-6]) if len(ma200_s) >= 6 else 0.0  # 5h slope

    # Z-scores
    zs = {}
    if not zs_df.empty:
        last_zs = zs_df.iloc[-1].to_dict()
        zs = {k: v for k, v in last_zs.items() if k != "timestamp" and v is not None}

    fomc = get_fomc_proximity(cal)
    clusters, cdet = compute_clusters(zs, bb_pct or 0.5, rsi_14 or 50.0, portfolio)
    total_score_raw = round(sum(clusters.values()), 4)

    # Apply G0 regime multiplier (mirrors gate_scoring.py; Sideways reads from params)
    _sw_mult = float(load_params().get("sideways_multiplier", 0.5))
    regime_multiplier = {"Sideways": _sw_mult, "Bear": 0.0, "Bull": 1.0}.get(regime, 1.0)
    total_score_after_regime = round(total_score_raw * regime_multiplier, 4)

    # Apply global confidence multiplier (portfolio is source of truth — backend computed it)
    global_conf_mult = portfolio.get("last_global_confidence_multiplier", 1.0) or 1.0
    total_score = round(total_score_after_regime * global_conf_mult, 4)

    # Re-evaluate kill switches on fresh data (mirrors gate_scoring.check_kill_switches)
    # Result: signal_computed is consistent with total_score shown in the body
    _ks = load_params().get("kill_switches", {})
    _news_d = cdet.get("news", {})
    _g2_raw = (0.5 * float(_news_d.get("crypto_score") or 0)
               + 0.5 * float(_news_d.get("fed_score") or 0))
    _oi_z = float(zs.get("oi_z") or 0)
    _fomc_kill = (
        float(_news_d.get("fed_score") or 0) < _ks.get("g2_fed_fomc_threshold", -1.0)
        and fomc.get("days_away", 9999) <= 2
        and fomc.get("event_type") == "fomc_decision"
    )

    if regime == "Bear":
        signal_computed, block_reason_computed = "BLOCK", "BLOCK_BEAR_REGIME"
    elif (bb_pct or 0) >= _ks.get("bb_top_threshold", 0.80):
        signal_computed, block_reason_computed = "BLOCK", "BLOCK_BB_TOP"
    elif _oi_z > _ks.get("oi_extreme_z", 2.5):
        signal_computed, block_reason_computed = "BLOCK", "BLOCK_OI_EXTREME"
    elif _g2_raw < _ks.get("news_bear_score", -3.0):
        signal_computed, block_reason_computed = "BLOCK", "BLOCK_NEWS_BEAR"
    elif _fomc_kill:
        signal_computed, block_reason_computed = "BLOCK", "BLOCK_FED_HAWKISH"
    elif total_score >= threshold:
        signal_computed, block_reason_computed = "ENTER", None
    else:
        signal_computed, block_reason_computed = "HOLD", None

    # score and signal both freshly computed — single source of truth
    score = total_score
    signal = signal_computed

    # L/S whale
    ls_account_val = _latest(lsa_df, "longShortRatio", 1.0)
    ls_position_val = _latest(lsp_df, "longShortRatio", 1.0)
    ws_label, ws_text = whale_signal(ls_account_val or 1.0, pct_24h)

    # OI / funding / taker latest
    oi_val = _latest(oi_df, "open_interest", 0)
    fund_val = _latest(funding_df, "funding_rate", 0)
    taker_val = _latest(taker_df, "buy_sell_ratio", 1.0)

    # Macro latest
    dgs10 = _latest(macro_df, "dgs10", 0)
    dgs2  = _latest(macro_df, "dgs2", 0)
    rrp   = _latest(macro_df, "rrp", 0)
    fg_val = _latest(fg_df, "fg_value", 0)
    fg_cls = _latest(fg_df, "fg_classification", "")
    cdet["sentiment"]["fg_raw"] = fg_val  # raw 0-100 value for interpret_cluster
    vix_val   = _latest(vix_df, "close", 0)
    dxy_val   = _latest(dxy_df, "close", 0)
    oil_val   = _latest(oil_df, "close", 0)
    sp500_val = _latest(sp500_df, "close", 0)

    # News scores
    crypto_ns = _latest(ns_df, "crypto_score", 0.0)
    macro_ns  = _latest(ns_df, "macro_score", 0.0)
    combined_ns = _latest(ns_df, "combined_score", 0.0)

    # ── Auto-refresh toggle ────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### ⚙️ Settings")
        auto_refresh = st.checkbox("Auto refresh (60s)", value=False)
        if auto_refresh:
            time.sleep(60)
            st.rerun()
        st.markdown("---")
        st.markdown(f"**Last load:** {datetime.now(timezone.utc).strftime('%H:%M:%S UTC')}")

    # =========================================================================
    # SECTION 1: HEADER
    # =========================================================================
    sig_class = {"ENTER": "signal-enter", "ENTER_BOT2": "signal-enter", "HOLD": "signal-hold", "BLOCK": "signal-block", "FILTERED": "signal-filter", "COOLDOWN": "signal-filter"}.get(signal, "neut")
    price_c   = "pos" if pct_24h >= 0 else "neg"
    regime_c  = {"Bull": "pos", "Bear": "neg", "Sideways": "warn"}.get(regime, "neut")
    fed_note  = f"Fed: {fomc['next_event']} em {fomc['days_away']}d" if fomc["next_event"] else "Fed: sem eventos próximos"
    cycle_info = get_last_cycle_info()
    if not cycle_info:
        cycle_age = "nunca executou"
        cycle_ok  = "🔴"
    else:
        age_m = cycle_info["age_min"]
        cycle_age = f"{age_m:.0f}min atrás"
        cycle_ok  = "✅" if age_m < 90 else ("⚠️" if age_m < 180 else "🔴")

    # MA200 header snippet (pre-built to avoid backslash in f-string)
    if ma200_val:
        _ma200_c = "pos" if (ma200_pct or 0) >= 0 else "neg"
        _ma200_arrow = "↑" if (ma200_slope or 0) > 0 else "↓"
        _ma200_header_html = (
            f'<span class="header-item">MA200: <span class="header-value">${ma200_val:,.0f}</span>'
            f' <span class="{_ma200_c}">{ma200_pct:+.1f}%</span>'
            f' <span style="color:#8b949e">{_ma200_arrow}</span></span>'
        )
    else:
        _ma200_header_html = ""

    _hdr_gc_mult = portfolio.get("last_global_confidence_multiplier", 1.0) or 1.0
    if _hdr_gc_mult < 0.95:
        _hdr_gc_color = "neg" if _hdr_gc_mult < 0.6 else "warn"
        _hdr_gc_html = f'<span class="header-item">GConf: <span class="{_hdr_gc_color}">×{_hdr_gc_mult:.2f}</span></span>'
    else:
        _hdr_gc_html = ""

    st.markdown(f"""
<div class="header-bar">
  <span>₿ <span class="header-value">${price:,.0f}</span></span>
  <span class="{price_c}">{pct_24h:+.2f}%</span>
  <span class="header-item">OI: <span class="header-value">${oi_val/1e9:.1f}B</span></span>
  <span class="header-item">F&G: <span class="header-value">{fg_val:.0f}</span> {fg_cls}</span>
  <span class="header-item">R5C: <span class="{regime_c} header-value">{regime}</span></span>
  {_ma200_header_html}
  {_hdr_gc_html}
  <span class="header-item">Gate: <span class="{sig_class}">{"ENTER (Bot2 Mom)" if signal == "ENTER_BOT2" else signal}</span></span>
  <span class="header-item">Score: <span class="header-value">{score:.3f}</span> / {threshold:.1f}</span>
  <span class="header-item">Capital: <span class="header-value">${capital:,.0f}</span></span>
  <span class="{'warn' if fomc['days_away'] < 5 else 'header-item'}">{fed_note}</span>
  <span class="header-item">Cron: {cycle_age} {cycle_ok}</span>
</div>
""", unsafe_allow_html=True)

    # =========================================================================
    # SECTION 2: GATE SCORING
    # =========================================================================
    st.markdown("### BOT 1 - Conservador")

    # Score bar — color driven by signal_computed (kill switches already applied)
    score_pct = min(max((total_score / threshold) * 100, 0), 110) if threshold > 0 else 0
    if signal_computed == "BLOCK":
        bar_color = "#f85149"
    elif signal_computed == "ENTER":
        bar_color = "#3fb950"
    elif total_score > threshold * 0.6:
        bar_color = "#d29922"
    else:
        bar_color = "#f85149"

    # Score breakdown (regime mult + global conf, shown when either reduces score)
    # Note: global_conf_mult already computed above (same portfolio key)
    _score_gc_src  = portfolio.get("last_global_confidence_source", "") or ""
    _show_gc = global_conf_mult < 0.95
    if regime_multiplier != 1.0 or _show_gc:
        _parts = [f'<span>Σ clusters (bruto): {total_score_raw:+.3f}</span>']
        if regime_multiplier != 1.0:
            _parts.append(
                f'<span style="color:#d29922;">× Regime {regime} ({regime_multiplier}×)</span>'
            )
        if _show_gc:
            _gc_color_bd = "#f85149" if global_conf_mult < 0.5 else "#d29922"
            _parts.append(
                f'<span style="color:{_gc_color_bd};">× GConf ({global_conf_mult:.2f}×)</span>'
            )
        _parts.append(f'<b style="color:{bar_color};">= {total_score:+.3f}</b>')
        multiplier_html = (
            f'<div style="display:flex; justify-content:space-between; font-size:12px; '
            f'color:#8b949e; margin-top:4px; flex-wrap:wrap; gap:4px;">'
            + " | ".join(_parts) +
            f'</div>'
        )
    else:
        multiplier_html = ""

    # Kill switch banner (only when BLOCK is from a kill switch, not from Bear/score)
    if signal_computed == "BLOCK" and block_reason_computed and block_reason_computed != "BLOCK_BEAR_REGIME":
        ks_html = (
            f'<div style="margin-top:6px; padding:4px 8px; background:#3d1a1a; border-radius:4px; '
            f'font-size:12px; color:#f85149;">⛔ Kill switch: {block_reason_computed}</div>'
        )
    else:
        ks_html = ""

    # Threshold tooltip: quantile context from score history
    if not score_hist.empty and "total_score" in score_hist.columns:
        _hist_vals = score_hist["total_score"].dropna().tail(90).tolist()
        if len(_hist_vals) >= 5:
            _q75 = round(float(np.quantile(_hist_vals, 0.75)), 3)
            _prox = fomc.get("proximity_adj", 0.0)
            _threshold_tip = (f"p75 de {len(_hist_vals)} ciclos: {_q75:.3f}"
                              f" + adj Fed: {_prox:+.1f} = {threshold:.3f}")
        else:
            _threshold_tip = f"Warmup — apenas {len(_hist_vals)} ciclos (mín 90)"
    else:
        _threshold_tip = "Sem histórico de scores"

    st.markdown(f"""
<div class="score-bar-wrap">
  <div style="display:flex; justify-content:space-between; margin-bottom:6px;">
    <span style="font-size:13px; color:#8b949e;">Score Total</span>
    <span style="font-size:18px; font-weight:700; color:{bar_color};">{total_score:+.3f}</span>
    <span style="font-size:13px; color:#8b949e;" title="{_threshold_tip}">Threshold: {threshold:.1f} ⓘ</span>
  </div>
  <div style="background:#21262d; border-radius:4px; height:8px; width:100%;">
    <div style="background:{bar_color}; width:{min(score_pct,100):.1f}%; height:8px; border-radius:4px;"></div>
  </div>
  {multiplier_html}
  {ks_html}
</div>
""", unsafe_allow_html=True)

    # 6 cluster cards
    CLUSTER_ORDER = ["technical", "positioning", "macro", "liquidity", "sentiment", "news"]
    CLUSTER_ICONS = {"technical":"📊", "positioning":"📈", "macro":"🏛️", "liquidity":"💧", "sentiment":"🧠", "news":"📰"}
    cols_c = st.columns(3)
    for i, name in enumerate(CLUSTER_ORDER):
        sc = clusters.get(name, 0)
        sc_c = "pos" if sc > 0.1 else ("neg" if sc < -0.1 else "neut")
        d = cdet.get(name, {})
        interp = interpret_cluster(name, sc, d, zs)

        if name == "technical":
            sub = f"BB: {d.get('bb_s',0):+.1f} | RSI: {d.get('rsi_s',0):+.1f} → G1 bruto: {d.get('g1_raw',0):+.3f}"
        elif name == "positioning":
            g4v = d.get('g4', 0); g10v = d.get('g10', 0)
            sub = f"G4 OI: {g4v:+.3f} | G10 Fund: {g10v:+.3f} → raw: {g4v+g10v:+.3f}"
        elif name == "macro":
            sub = (f"DGS10: {d.get('g3_dgs10',0):+.3f} | DGS2: {d.get('g3_dgs2',0):+.3f} | "
                   f"Curve: {d.get('g3_curve',0):+.3f} | RRP: {d.get('g3_rrp',0):+.3f}")
        elif name == "liquidity":
            g5v = d.get('g5', 0); g7v = d.get('g7', 0)
            sub = f"G5 Stable: {g5v:+.3f} | G7 ETF: {g7v:+.3f} → raw: {g5v+g7v:+.3f}"
        elif name == "sentiment":
            sub = (f"G6 Bubble: {d.get('g6',0):+.3f} | G8 F&G: {d.get('g8',0):+.3f} | "
                   f"G9 Taker: {d.get('g9',0):+.3f}")
        else:  # news
            cs_ = d.get('crypto_score', 0)
            ms_ = d.get('macro_score',  0)
            fs_ = d.get('fed_score',    0)
            comb_ = cs_ * 0.4 + ms_ * 0.6
            g2r   = d.get('g2_raw', 0)
            cap_note = f" (cap: {sc:+.3f})" if abs(g2r - sc) > 0.005 else ""
            sub = (f"Crypto: {cs_:+.2f} | Macro: {ms_:+.2f} | Fed: {fs_:+.2f}"
                   f"<br>Combined = Crypto×0.4 + Macro×0.6 = {comb_:+.3f}{cap_note}")

        with cols_c[i % 3]:
            st.markdown(f"""
<div class="cg-card">
  <div class="cg-card-title">{CLUSTER_ICONS[name]} {name.upper()}</div>
  <div class="cg-card-value {sc_c}">{sc:+.3f}</div>
  <div class="cg-card-sub">{sub}</div>
  <div class="cg-card-interp">{interp}</div>
</div>""", unsafe_allow_html=True)

    # =========================================================================
    # SECTION 3: BOT 1 TRADES HISTORY
    # =========================================================================
    st.markdown("---")
    st.markdown("### 📊 Histórico Bot 1 — Trades")
    st.caption("Trades baseados no Gate Scoring acima")

    _th_hist = load_parquet("data/05_output/trades.parquet")

    def _filter_b1(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame()
        if "entry_bot" not in df.columns:
            return df.copy()
        return df[df["entry_bot"] == "bot1"].copy()

    _b1h = _filter_b1(_th_hist)

    # ── Posição aberta (se houver) ─────────────────────────────────────────
    _pos_open = portfolio.get("has_position", False)
    _ep_open  = portfolio.get("entry_price")
    _sl_open  = portfolio.get("stop_loss_price")
    _tp_open  = portfolio.get("take_profit_price")
    _tr_open  = portfolio.get("trailing_high")
    _et_open  = portfolio.get("entry_time")
    _bot_open = portfolio.get("entry_bot", "bot1")

    if _pos_open and _bot_open in ("bot1", None, "") and _ep_open:
        _ur_pct = ((price / _ep_open) - 1) * 100
        _ur_c   = "pos" if _ur_pct >= 0 else "neg"
        _et_str = pd.to_datetime(_et_open).strftime("%m/%d %H:%M") if _et_open else "?"
        _sl_str = f"${_sl_open:,.0f}" if _sl_open else "—"
        _tp_str = f"${_tp_open:,.0f}" if _tp_open else "—"
        _tr_str = f"${_tr_open:,.0f}" if _tr_open else "—"
        st.markdown(f"""
<div class="cg-card" style="padding:8px 16px; font-size:12px; border-left:3px solid #3fb950; margin-bottom:6px;">
  <span style="color:#3fb950; font-weight:700;">🟢 POSIÇÃO ABERTA</span>
  &nbsp;|&nbsp; <span style="color:#8b949e;">Entrada:</span> ${_ep_open:,.0f} ({_et_str})
  &nbsp;|&nbsp; <span style="color:#8b949e;">SL:</span> {_sl_str}
  &nbsp;|&nbsp; <span style="color:#8b949e;">TP:</span> {_tp_str}
  &nbsp;|&nbsp; <span style="color:#8b949e;">Trail High:</span> {_tr_str}
  &nbsp;|&nbsp; <span style="color:#8b949e;">Atual:</span> ${price:,.0f}
  &nbsp;|&nbsp; <span class="{_ur_c}" style="font-weight:700;">{_ur_pct:+.2f}%</span>
</div>""", unsafe_allow_html=True)

    # ── Histórico de trades completados ────────────────────────────────────
    if not _b1h.empty:
        _b1h_disp = _b1h.copy()

        # Calcular preços absolutos de SL/TP a partir dos percentuais salvos
        def _price_from_pct(row, pct_col, direction=1):
            ep = row.get("entry_price", 0) or 0
            pct = row.get(pct_col, 0) or 0
            return ep * (1 + direction * pct) if ep else None

        _b1h_disp["SL $"]       = _b1h_disp.apply(lambda r: _price_from_pct(r, "actual_stop_loss_pct", -1), axis=1)
        _b1h_disp["TP $"]       = _b1h_disp.apply(lambda r: _price_from_pct(r, "actual_take_profit_pct", 1), axis=1)
        _b1h_disp["Trail High"] = _b1h_disp.apply(
            lambda r: (r.get("entry_price", 0) or 0) * (1 + (r.get("mfe_pct", 0) or 0) / 100), axis=1
        )

        _exit_emoji = {
            "TP": "🎯 TP", "SL": "🛑 SL", "TRAILING": "📉 Trail",
            "MAX_HOLD": "⏰ MaxHold", "bot2_timeout": "⏰ MaxHold",
            "OI_EARLY_EXIT": "🚨 OI",
        }

        for _, _t in _b1h_disp.sort_values("entry_time", ascending=False).head(10).iterrows():
            _et  = pd.to_datetime(_t.get("entry_time"))
            _xt  = pd.to_datetime(_t.get("exit_time"))
            _ets = _et.strftime("%m/%d %H:%M") if pd.notna(_et) else "?"
            _xts = _xt.strftime("%m/%d %H:%M") if pd.notna(_xt) else "?"
            _ret = _t.get("return_pct", 0) or 0
            _rc  = "pos" if _ret > 0 else "neg"
            _ep  = _t.get("entry_price", 0) or 0
            _xp  = _t.get("exit_price", 0) or 0
            _xr  = _exit_emoji.get(_t.get("exit_reason", ""), _t.get("exit_reason", "?"))
            _sl  = _t.get("SL $")
            _tp  = _t.get("TP $")
            _th  = _t.get("Trail High")
            _dur = _t.get("duration_hours", 0) or 0
            _sc  = _t.get("entry_score_adjusted", 0) or 0
            _sl_s = f"${_sl:,.0f}" if _sl and _sl > 0 else "—"
            _tp_s = f"${_tp:,.0f}" if _tp and _tp > 0 else "—"
            _th_s = f"${_th:,.0f}" if _th and _th > _ep else "—"
            st.markdown(f"""
<div class="cg-card" style="padding:8px 16px; font-size:12px; margin-bottom:3px;">
  <span style="color:#8b949e;">{_ets} → {_xts} ({_dur:.0f}h)</span>
  &nbsp;|&nbsp; <span style="color:#8b949e;">In:</span> ${_ep:,.0f}
  &nbsp;|&nbsp; <span style="color:#8b949e;">SL:</span> {_sl_s}
  &nbsp;|&nbsp; <span style="color:#8b949e;">TP:</span> {_tp_s}
  &nbsp;|&nbsp; <span style="color:#8b949e;">Trail:</span> {_th_s}
  &nbsp;|&nbsp; <span style="color:#8b949e;">Out:</span> ${_xp:,.0f} <span style="color:#8b949e;">({_xr})</span>
  &nbsp;|&nbsp; <span class="{_rc}" style="font-weight:700;">{_ret:+.2f}%</span>
  &nbsp;|&nbsp; <span style="color:#8b949e;">Score:</span> {_sc:+.3f}
</div>""", unsafe_allow_html=True)
    else:
        st.info("Nenhum trade Bot 1 completado ainda.")

    # ── Métricas rápidas ───────────────────────────────────────────────────
    if not _b1h.empty:
        _m1 = compute_bot_metrics(_b1h)
        if _m1["n_trades"] < 3:
            st.caption("⚠️ Métricas preliminares — mínimo recomendado: 20 trades.")
        _mc = st.columns(5)
        _mc[0].markdown(f'<div class="cg-card" style="text-align:center;padding:8px;"><div class="cg-card-title">TRADES</div><div style="font-size:16px;font-weight:700;">{_m1["n_trades"]}</div><div class="cg-card-sub">{_m1["wins"]}W / {_m1["losses"]}L</div></div>', unsafe_allow_html=True)
        _wr_c = "pos" if _m1["win_rate"] >= 50 else "neg"
        _mc[1].markdown(f'<div class="cg-card" style="text-align:center;padding:8px;"><div class="cg-card-title">WIN RATE</div><div class="cg-card-value {_wr_c}">{_m1["win_rate"]:.0f}%</div></div>', unsafe_allow_html=True)
        _pf_c = "pos" if _m1["profit_factor"] >= 1.0 else "neg"
        _pf_v = "∞" if _m1["profit_factor"] >= 99 else f'{_m1["profit_factor"]:.2f}'
        _mc[2].markdown(f'<div class="cg-card" style="text-align:center;padding:8px;"><div class="cg-card-title">PROFIT FACTOR</div><div class="cg-card-value {_pf_c}">{_pf_v}</div></div>', unsafe_allow_html=True)
        _tr_c = "pos" if _m1["total_return"] >= 0 else "neg"
        _mc[3].markdown(f'<div class="cg-card" style="text-align:center;padding:8px;"><div class="cg-card-title">RETORNO</div><div class="cg-card-value {_tr_c}">{_m1["total_return"]:+.2f}%</div></div>', unsafe_allow_html=True)
        _mc[4].markdown(f'<div class="cg-card" style="text-align:center;padding:8px;"><div class="cg-card-title">MAX DD</div><div class="cg-card-value neg">{_m1["max_drawdown"]:.2f}%</div></div>', unsafe_allow_html=True)

    # =========================================================================
    # SECTION 3b: BOT 2 — MOMENTUM
    # =========================================================================
    st.markdown("---")
    st.markdown("### 🚀 BOT 2 - Momentum")
    st.caption("Filosofia: momentum alimentado por liquidez. Entra com stablecoin fuel + trend + sem blow-off.")

    # ── Valores atuais (todos já carregados no escopo) ─────────────────────
    _sz   = last_zs.get("stablecoin_z") if not zs_df.empty else None
    _r1d  = pct_24h / 100  # ret_1d = pct_24h / 100 (pct_24h já em %)
    _rsi2 = rsi_14
    _bb2  = bb_pct
    _ma21 = _latest(spot_df, "ma_21", 0.0) if not spot_df.empty else 0.0
    _above_ma21 = bool(price and _ma21 and price > _ma21)

    # Thresholds reais de parameters.yml
    _sz_min  = 1.3
    _r1d_min = 0.0
    _rsi_min = 50
    _bb_max  = 0.98   # bloqueia se >= 0.98 (anti blow-off top)
    _spike_ret_max = 0.03
    _spike_rsi_max = 65

    # ── Helper: check individual filter ────────────────────────────────────
    def _b2_chk(val, ok_condition, border_condition=None):
        """Returns (passed, badge_html) with ✅/⚠️/❌."""
        if val is None:
            return False, "⚪"
        if ok_condition(val):
            return True, "✅"
        if border_condition and border_condition(val):
            return False, "⚠️"
        return False, "❌"

    _f_sz   = _b2_chk(_sz,   lambda v: v > _sz_min,  lambda v: v > _sz_min - 0.3)
    _f_r1d  = _b2_chk(_r1d,  lambda v: v > _r1d_min, lambda v: v > -0.005)
    _f_rsi  = _b2_chk(_rsi2, lambda v: _rsi_min < v < 85, lambda v: (_rsi_min - 5) < v < 90)
    _f_bb   = _b2_chk(_bb2,  lambda v: v < _bb_max,  lambda v: v < _bb_max + 0.02)
    _f_ma21 = (_above_ma21, "✅" if _above_ma21 else "❌")

    # News filter
    _news_min = load_params().get("momentum_filter", {}).get("news_score_min", -1.0)
    _news_val = float(crypto_ns) if crypto_ns is not None else 0.0
    _f_news = _b2_chk(_news_val, lambda v: v >= _news_min, lambda v: v >= _news_min - 0.5)

    # Spike guard extra: ret_1d > 3% AND RSI > 65 → entrada tardia
    _spike_block = bool(_r1d and _rsi2 and _r1d > _spike_ret_max and _rsi2 > _spike_rsi_max)

    _filters = [_f_sz, _f_r1d, _f_rsi, _f_bb, _f_ma21, _f_news]
    _n_pass  = sum(1 for passed, _ in _filters if passed)
    _all_ok  = _n_pass == 6 and not _spike_block

    # ── Status header ──────────────────────────────────────────────────────
    _b2_enabled = load_params().get("momentum_filter", {}).get("enabled", False)
    if not _b2_enabled:
        st.info("⏸️ Bot 2 desabilitado (momentum_filter.enabled: false em parameters.yml)")
    elif _spike_block:
        st.warning(f"🛑 SPIKE GUARD — ret_1d={_r1d*100:+.1f}% > 3% + RSI={_rsi2:.0f} > 65 → entrada tardia bloqueada")
    elif _all_ok:
        st.success(f"✅ ENTRY ELEGÍVEL — {_n_pass}/6 filtros passam")
    else:
        _fail_names = []
        if not _f_sz[0]:    _fail_names.append("stablecoin")
        if not _f_r1d[0]:   _fail_names.append("momentum")
        if not _f_rsi[0]:   _fail_names.append("RSI")
        if not _f_bb[0]:    _fail_names.append("BB top")
        if not _f_ma21[0]:  _fail_names.append("MA21")
        if not _f_news[0]:  _fail_names.append("news bearish")
        st.warning(f"🛑 BLOCK — {_n_pass}/6 filtros ({', '.join(_fail_names)} {'⚠️ borderline' if any('⚠️' in b for _, b in _filters) else 'falhou'})")

    # ── 6 filter cards ─────────────────────────────────────────────────────
    _fc = st.columns(6)

    def _card(col, icon, label, val_str, badge, threshold_str, color_class):
        col.markdown(f"""
<div class="cg-card" style="text-align:center; padding:10px 6px;">
  <div class="cg-card-title">{icon} {label}</div>
  <div class="cg-card-value {color_class}" style="font-size:20px;">{val_str} {badge}</div>
  <div class="cg-card-sub">{threshold_str}</div>
</div>""", unsafe_allow_html=True)

    _sz_str   = f"{_sz:.2f}" if _sz is not None else "N/A"
    _r1d_str  = f"{_r1d*100:+.2f}%" if _r1d is not None else "N/A"
    _rsi_str  = f"{_rsi2:.1f}" if _rsi2 is not None else "N/A"
    _bb_str   = f"{_bb2:.3f}" if _bb2 is not None else "N/A"
    _ma21_str = "ACIMA" if _above_ma21 else "ABAIXO"

    _news_str = f"{_news_val:+.2f}" if _news_val is not None else "N/A"
    _card(_fc[0], "💧", "Stablecoin Z",  _sz_str,   _f_sz[1],   f"> {_sz_min}",    "pos" if _f_sz[0] else "neg")
    _card(_fc[1], "📈", "Momentum 24h",  _r1d_str,  _f_r1d[1],  "> 0%",            "pos" if _f_r1d[0] else "neg")
    _card(_fc[2], "📊", "RSI",           _rsi_str,  _f_rsi[1],  f"> {_rsi_min}",   "pos" if _f_rsi[0] else "neg")
    _card(_fc[3], "🎯", "BB (não top)",  _bb_str,   _f_bb[1],   f"< {_bb_max}",    "pos" if _f_bb[0] else "neg")
    _card(_fc[4], "🔄", "Trend MA21",    _ma21_str, _f_ma21[1], "close > MA21",    "pos" if _f_ma21[0] else "neg")
    _card(_fc[5], "📰", "News",          _news_str, _f_news[1], f">= {_news_min}", "pos" if _f_news[0] else "neg")

    # ── Posição Bot 2 aberta (se houver) ──────────────────────────────────
    _bot_open2 = portfolio.get("entry_bot")
    _has_pos2  = portfolio.get("has_position", False) and _bot_open2 == "bot2"
    if _has_pos2:
        _ep2  = portfolio.get("entry_price", 0)
        _sl2  = portfolio.get("stop_loss_price")
        _tp2  = portfolio.get("take_profit_price")
        _tr2  = portfolio.get("trailing_high")
        _et2  = portfolio.get("entry_time")
        _ur2  = ((price / _ep2) - 1) * 100 if _ep2 else 0
        _urc2 = "pos" if _ur2 >= 0 else "neg"
        _et2s = pd.to_datetime(_et2).strftime("%m/%d %H:%M") if _et2 else "?"
        st.markdown(f"""
<div class="cg-card" style="padding:8px 16px; font-size:12px; border-left:3px solid #3fb950; margin-top:8px;">
  <span style="color:#3fb950; font-weight:700;">🟢 POSIÇÃO ABERTA (Bot 2)</span>
  &nbsp;|&nbsp; <span style="color:#8b949e;">Entrada:</span> ${_ep2:,.0f} ({_et2s})
  &nbsp;|&nbsp; <span style="color:#8b949e;">SL:</span> {"$"+f"{_sl2:,.0f}" if _sl2 else "—"}
  &nbsp;|&nbsp; <span style="color:#8b949e;">TP:</span> {"$"+f"{_tp2:,.0f}" if _tp2 else "—"}
  &nbsp;|&nbsp; <span style="color:#8b949e;">Trail:</span> {"$"+f"{_tr2:,.0f}" if _tr2 else "—"}
  &nbsp;|&nbsp; <span style="color:#8b949e;">Atual:</span> ${price:,.0f}
  &nbsp;|&nbsp; <span class="{_urc2}" style="font-weight:700;">{_ur2:+.2f}%</span>
</div>""", unsafe_allow_html=True)

    # ── Histórico trades Bot 2 ─────────────────────────────────────────────
    st.markdown("**Histórico Bot 2 — Trades**")
    _b2h = _th_hist[_th_hist["entry_bot"] == "bot2"].copy() if (
        not _th_hist.empty and "entry_bot" in _th_hist.columns
    ) else pd.DataFrame()

    if not _b2h.empty:
        _exit_emoji2 = {
            "TP": "🎯 TP", "SL": "🛑 SL", "TRAILING": "📉 Trail",
            "MAX_HOLD": "⏰ MaxHold", "bot2_timeout": "⏰ MaxHold",
        }
        _b2h["_sl_price"] = _b2h.apply(
            lambda r: (r.get("entry_price", 0) or 0) * (1 - (r.get("actual_stop_loss_pct", 0) or 0)), axis=1)
        _b2h["_tp_price"] = _b2h.apply(
            lambda r: (r.get("entry_price", 0) or 0) * (1 + (r.get("actual_take_profit_pct", 0) or 0)), axis=1)
        _b2h["_th_price"] = _b2h.apply(
            lambda r: (r.get("entry_price", 0) or 0) * (1 + (r.get("mfe_pct", 0) or 0) / 100), axis=1)

        for _, _t in _b2h.sort_values("entry_time", ascending=False).head(10).iterrows():
            _et  = pd.to_datetime(_t.get("entry_time"))
            _xt  = pd.to_datetime(_t.get("exit_time"))
            _ets = _et.strftime("%m/%d %H:%M") if pd.notna(_et) else "?"
            _xts = _xt.strftime("%m/%d %H:%M") if pd.notna(_xt) else "?"
            _ret = _t.get("return_pct", 0) or 0
            _rc  = "pos" if _ret > 0 else "neg"
            _ep  = _t.get("entry_price", 0) or 0
            _xp  = _t.get("exit_price", 0) or 0
            _dur = _t.get("duration_hours", 0) or 0
            _xr  = _exit_emoji2.get(_t.get("exit_reason", ""), _t.get("exit_reason", "?"))
            _sl  = _t.get("_sl_price", 0)
            _tp  = _t.get("_tp_price", 0)
            _th  = _t.get("_th_price", 0)
            _sz2 = _t.get("entry_stablecoin_z", 0) or 0
            _sl_s = f"${_sl:,.0f}" if _sl and _sl > 0 else "—"
            _tp_s = f"${_tp:,.0f}" if _tp and _tp > 0 else "—"
            _th_s = f"${_th:,.0f}" if _th and _th > _ep else "—"
            st.markdown(f"""
<div class="cg-card" style="padding:8px 16px; font-size:12px; margin-bottom:3px;">
  <span style="color:#8b949e;">{_ets} → {_xts} ({_dur:.0f}h)</span>
  &nbsp;|&nbsp; <span style="color:#8b949e;">In:</span> ${_ep:,.0f}
  &nbsp;|&nbsp; <span style="color:#8b949e;">SL:</span> {_sl_s}
  &nbsp;|&nbsp; <span style="color:#8b949e;">TP:</span> {_tp_s}
  &nbsp;|&nbsp; <span style="color:#8b949e;">Trail:</span> {_th_s}
  &nbsp;|&nbsp; <span style="color:#8b949e;">Out:</span> ${_xp:,.0f} <span style="color:#8b949e;">({_xr})</span>
  &nbsp;|&nbsp; <span class="{_rc}" style="font-weight:700;">{_ret:+.2f}%</span>
  &nbsp;|&nbsp; <span style="color:#8b949e;">StbZ:</span> {_sz2:.2f}
</div>""", unsafe_allow_html=True)

        # Métricas resumo Bot 2
        if not _b2h.empty:
            _m2 = compute_bot_metrics(_b2h)
            if _m2["n_trades"] < 3:
                st.caption("⚠️ Métricas preliminares — mínimo 20 trades para avaliação confiável.")
            _mc2 = st.columns(5)
            _mc2[0].markdown(f'<div class="cg-card" style="text-align:center;padding:8px;"><div class="cg-card-title">TRADES</div><div style="font-size:16px;font-weight:700;">{_m2["n_trades"]}</div><div class="cg-card-sub">{_m2["wins"]}W / {_m2["losses"]}L</div></div>', unsafe_allow_html=True)
            _wr2c = "pos" if _m2["win_rate"] >= 50 else "neg"
            _mc2[1].markdown(f'<div class="cg-card" style="text-align:center;padding:8px;"><div class="cg-card-title">WIN RATE</div><div class="cg-card-value {_wr2c}">{_m2["win_rate"]:.0f}%</div></div>', unsafe_allow_html=True)
            _pf2c = "pos" if _m2["profit_factor"] >= 1.0 else "neg"
            _pf2v = "∞" if _m2["profit_factor"] >= 99 else f'{_m2["profit_factor"]:.2f}'
            _mc2[2].markdown(f'<div class="cg-card" style="text-align:center;padding:8px;"><div class="cg-card-title">PROFIT FACTOR</div><div class="cg-card-value {_pf2c}">{_pf2v}</div></div>', unsafe_allow_html=True)
            _tr2c = "pos" if _m2["total_return"] >= 0 else "neg"
            _mc2[3].markdown(f'<div class="cg-card" style="text-align:center;padding:8px;"><div class="cg-card-title">RETORNO</div><div class="cg-card-value {_tr2c}">{_m2["total_return"]:+.2f}%</div></div>', unsafe_allow_html=True)
            _mc2[4].markdown(f'<div class="cg-card" style="text-align:center;padding:8px;"><div class="cg-card-title">MAX DD</div><div class="cg-card-value neg">{_m2["max_drawdown"]:.2f}%</div></div>', unsafe_allow_html=True)
    else:
        st.info("Nenhum trade Bot 2 completado ainda.")

    # =========================================================================
    # SECTION 3c: ETH — BOT 3 (Volume Defensivo)
    # =========================================================================
    st.markdown("---")
    st.markdown("### ⚡ ETH - Bot 3 (Volume Defensivo)")
    st.caption("Filosofia: mean reversion em volume baixo-médio (Q2). Conservador por natureza.")

    _eth_port_path = ROOT / "data/04_scoring/portfolio_eth.json"
    try:
        _eth_port = json.loads(_eth_port_path.read_text()) if _eth_port_path.exists() else {}
    except Exception:
        _eth_port = {}

    try:
        from src.trading.eth_bot3 import compute_eth_features as _cef
        _ef3 = _cef()
    except Exception:
        _ef3 = {}

    _eth_spot_df  = load_parquet("data/01_raw/spot/eth_1h.parquet")
    _eth_price    = float(_eth_spot_df["close"].iloc[-1]) if not _eth_spot_df.empty else None
    _eth_capital  = float(_eth_port.get("capital_usd", 10000.0))
    _eth_has_pos  = bool(_eth_port.get("has_position", False))
    _eth_vol_z    = _ef3.get("volume_z")
    _eth_rsi_v    = _ef3.get("rsi_14")
    _eth_above200 = bool(_ef3.get("above_ma200", False))

    _ec_h = st.columns(3)
    _eth_px_s = f"${_eth_price:,.2f}" if _eth_price else "—"
    _eth_cc   = "pos" if _eth_capital >= 10000 else "neg"
    _ec_h[0].markdown(f'<div class="cg-card" style="text-align:center;padding:8px;"><div class="cg-card-title">ETH PRICE</div><div class="cg-card-value">{_eth_px_s}</div></div>', unsafe_allow_html=True)
    _ec_h[1].markdown(f'<div class="cg-card" style="text-align:center;padding:8px;"><div class="cg-card-title">CAPITAL</div><div class="cg-card-value {_eth_cc}">${_eth_capital:,.2f}</div><div class="cg-card-sub {_eth_cc}">{(_eth_capital/10000-1)*100:+.2f}% vs início</div></div>', unsafe_allow_html=True)
    _eth_pos_c = "pos" if _eth_has_pos else "neut"
    _ec_h[2].markdown(f'<div class="cg-card" style="text-align:center;padding:8px;"><div class="cg-card-title">POSIÇÃO</div><div class="cg-card-value {_eth_pos_c}">{"ABERTA" if _eth_has_pos else "—"}</div></div>', unsafe_allow_html=True)

    if _eth_has_pos:
        _ep_eth  = _eth_port.get("entry_price") or 0
        _sl_eth  = _eth_port.get("stop_loss_price")
        _tp_eth  = _eth_port.get("take_profit_price")
        _ur_eth  = ((_eth_price / _ep_eth) - 1) * 100 if (_eth_price and _ep_eth) else 0
        _urc_eth = "pos" if _ur_eth >= 0 else "neg"
        _sl_es   = f"${_sl_eth:,.2f}" if _sl_eth else "—"
        _tp_es   = f"${_tp_eth:,.2f}" if _tp_eth else "—"
        st.markdown(f'<div class="cg-card" style="padding:8px 16px;font-size:12px;border-left:3px solid #3fb950;margin-top:8px;"><span style="color:#3fb950;font-weight:700;">🟢 POSIÇÃO ABERTA (Bot 3)</span> &nbsp;|&nbsp; <span style="color:#8b949e;">Entrada:</span> ${_ep_eth:,.2f} &nbsp;|&nbsp; <span style="color:#8b949e;">SL:</span> {_sl_es} &nbsp;|&nbsp; <span style="color:#8b949e;">TP:</span> {_tp_es} &nbsp;|&nbsp; <span style="color:#8b949e;">Atual:</span> {_eth_px_s} &nbsp;|&nbsp; <span class="{_urc_eth}" style="font-weight:700;">{_ur_eth:+.2f}%</span></div>', unsafe_allow_html=True)

    _vol_z_min_e, _vol_z_max_e, _rsi_max_e = -0.75, -0.30, 60
    _ef3_volq2  = bool(_eth_vol_z is not None and _vol_z_min_e < _eth_vol_z < _vol_z_max_e)
    _ef3_rsi_ok = bool(_eth_rsi_v is not None and _eth_rsi_v < _rsi_max_e)
    _n_eth_pass = sum([_ef3_volq2, _ef3_rsi_ok, _eth_above200])

    if _n_eth_pass == 3:
        st.success("✅ ENTRY ELEGÍVEL — 3/3 filtros")
    elif _n_eth_pass == 0:
        st.error("🛑 BLOCK — 0/3 filtros")
    else:
        st.warning(f"⏳ WAIT — {_n_eth_pass}/3 filtros passam")

    _efc = st.columns(3)
    _vol3_s = f"{_eth_vol_z:.2f}" if _eth_vol_z is not None else "N/A"
    _rsi3_s = f"{_eth_rsi_v:.1f}" if _eth_rsi_v is not None else "N/A"
    _card(_efc[0], "📊", "Volume Q2",   _vol3_s,                                "✅" if _ef3_volq2 else "❌",  "-0.75 < z < -0.30", "pos" if _ef3_volq2 else "neg")
    _card(_efc[1], "📉", "RSI < 60",    _rsi3_s,                                "✅" if _ef3_rsi_ok else "❌", "< 60",              "pos" if _ef3_rsi_ok else "neg")
    _card(_efc[2], "📈", "MA200 Trend", "ACIMA" if _eth_above200 else "ABAIXO", "✅" if _eth_above200 else "❌", "close > MA200",  "pos" if _eth_above200 else "neg")

    st.markdown("#### 📊 Histórico Bot 3 — Trades")
    _eth_tdf = load_parquet("data/05_output/trades_eth.parquet")
    if _eth_tdf.empty:
        st.info("Nenhum trade Bot 3 completado ainda.")
    else:
        _m3 = compute_bot_metrics(_eth_tdf)
        if _m3["n_trades"] < 3:
            st.caption("⚠️ Métricas preliminares — mínimo 20 trades para avaliação confiável.")
        _mc3 = st.columns(5)
        _mc3[0].markdown(f'<div class="cg-card" style="text-align:center;padding:8px;"><div class="cg-card-title">TRADES</div><div style="font-size:16px;font-weight:700;">{_m3["n_trades"]}</div><div class="cg-card-sub">{_m3["wins"]}W / {_m3["losses"]}L</div></div>', unsafe_allow_html=True)
        _wr3c = "pos" if _m3["win_rate"] >= 50 else "neg"
        _mc3[1].markdown(f'<div class="cg-card" style="text-align:center;padding:8px;"><div class="cg-card-title">WIN RATE</div><div class="cg-card-value {_wr3c}">{_m3["win_rate"]:.0f}%</div></div>', unsafe_allow_html=True)
        _pf3c = "pos" if _m3["profit_factor"] >= 1.0 else "neg"
        _pf3v = "∞" if _m3["profit_factor"] >= 99 else f'{_m3["profit_factor"]:.2f}'
        _mc3[2].markdown(f'<div class="cg-card" style="text-align:center;padding:8px;"><div class="cg-card-title">PROFIT FACTOR</div><div class="cg-card-value {_pf3c}">{_pf3v}</div></div>', unsafe_allow_html=True)
        _tr3c = "pos" if _m3["total_return"] >= 0 else "neg"
        _mc3[3].markdown(f'<div class="cg-card" style="text-align:center;padding:8px;"><div class="cg-card-title">RETORNO</div><div class="cg-card-value {_tr3c}">{_m3["total_return"]:+.2f}%</div></div>', unsafe_allow_html=True)
        _mc3[4].markdown(f'<div class="cg-card" style="text-align:center;padding:8px;"><div class="cg-card-title">MAX DD</div><div class="cg-card-value neg">{_m3["max_drawdown"]:.2f}%</div></div>', unsafe_allow_html=True)

    # =========================================================================
    # SECTION 3d: SOL — BOT 4 (Taker/Flow)
    # =========================================================================
    st.markdown("---")
    st.markdown("### 🟣 SOL - Bot 4 (Taker/Flow)")
    st.caption("Filosofia: compra agressiva + fluxo saudável + contexto ETH. Hard gates + Bot 2 DNA.")

    _sol_port_path = ROOT / "data/04_scoring/portfolio_sol.json"
    try:
        _sol_port = json.loads(_sol_port_path.read_text()) if _sol_port_path.exists() else {}
    except Exception:
        _sol_port = {}

    try:
        from src.trading.sol_bot4 import compute_sol_features as _csf
        _sf4 = _csf()
    except Exception:
        _sf4 = {}

    _sol_spot_df  = load_parquet("data/01_raw/spot/sol_1h.parquet")
    _sol_price    = float(_sol_spot_df["close"].iloc[-1]) if not _sol_spot_df.empty else None
    _sol_capital  = float(_sol_port.get("capital_usd", 10000.0))
    _sol_has_pos  = bool(_sol_port.get("has_position", False))
    _sol_px_s     = f"${_sol_price:,.2f}" if _sol_price else "—"

    _sol_h = st.columns(3)
    _sol_cc = "pos" if _sol_capital >= 10000 else "neg"
    _sol_h[0].markdown(f'<div class="cg-card" style="text-align:center;padding:8px;"><div class="cg-card-title">SOL PRICE</div><div class="cg-card-value">{_sol_px_s}</div></div>', unsafe_allow_html=True)
    _sol_h[1].markdown(f'<div class="cg-card" style="text-align:center;padding:8px;"><div class="cg-card-title">CAPITAL</div><div class="cg-card-value {_sol_cc}">${_sol_capital:,.2f}</div><div class="cg-card-sub {_sol_cc}">{(_sol_capital/10000-1)*100:+.2f}% vs início</div></div>', unsafe_allow_html=True)
    _sol_pos_c = "pos" if _sol_has_pos else "neut"
    _sol_h[2].markdown(f'<div class="cg-card" style="text-align:center;padding:8px;"><div class="cg-card-title">POSIÇÃO</div><div class="cg-card-value {_sol_pos_c}">{"ABERTA" if _sol_has_pos else "—"}</div></div>', unsafe_allow_html=True)

    if _sol_has_pos:
        _ep_sol  = _sol_port.get("entry_price") or 0
        _sl_sol  = _sol_port.get("stop_loss_price")
        _tp_sol  = _sol_port.get("take_profit_price")
        _th_sol  = _sol_port.get("trailing_high")
        _et_sol  = _sol_port.get("entry_timestamp")
        _mh_sol  = _sol_port.get("max_hold_until")
        _ur_sol  = ((_sol_price / _ep_sol) - 1) * 100 if (_sol_price and _ep_sol) else 0
        _urc_sol = "pos" if _ur_sol >= 0 else "neg"
        _ets_sol = pd.to_datetime(_et_sol, utc=True).strftime("%m/%d %H:%M") if _et_sol else "?"
        _mhs_sol = pd.to_datetime(_mh_sol, utc=True).strftime("%m/%d %H:%M") if _mh_sol else "?"
        _sl_ss   = f"${_sl_sol:,.2f}" if _sl_sol else "—"
        _tp_ss   = f"${_tp_sol:,.2f}" if _tp_sol else "—"
        _th_ss   = f"${_th_sol:,.2f}" if _th_sol else "—"
        st.markdown(f"""
<div class="cg-card" style="padding:10px 16px; font-size:12px; border-left:3px solid #a371f7; margin-top:8px;">
  <span style="color:#a371f7; font-weight:700;">🟣 POSIÇÃO ABERTA (Bot 4)</span>
  &nbsp;|&nbsp; <span style="color:#8b949e;">Entrada:</span> ${_ep_sol:,.2f} ({_ets_sol})
  &nbsp;|&nbsp; <span style="color:#8b949e;">SL:</span> {_sl_ss}
  &nbsp;|&nbsp; <span style="color:#8b949e;">TP:</span> {_tp_ss}
  &nbsp;|&nbsp; <span style="color:#8b949e;">Trail:</span> {_th_ss}
  &nbsp;|&nbsp; <span style="color:#8b949e;">Max Hold:</span> {_mhs_sol}
  &nbsp;|&nbsp; <span style="color:#8b949e;">Atual:</span> {_sol_px_s}
  &nbsp;|&nbsp; <span class="{_urc_sol}" style="font-weight:700;">{_ur_sol:+.2f}%</span>
</div>""", unsafe_allow_html=True)

    # Hard gates
    _taker_z4 = _sf4.get("taker_z_prev")
    _oi_z24_4 = _sf4.get("oi_z_24h_max")
    _eth_rh4  = _sf4.get("eth_ret_1h_prev")
    _sol_rsi4 = _sf4.get("rsi")
    _sol_ma21 = _sf4.get("ma21")
    _sol_cl   = _sf4.get("close") or _sol_price or 0
    _sol_ret1d = _sf4.get("ret_1d")
    _sz_sol   = last_zs.get("stablecoin_z") if not zs_df.empty else None

    _g1_ok = bool(_taker_z4 is not None and _taker_z4 > 0.3)
    _g2_ok = bool(_oi_z24_4 is not None and _oi_z24_4 < 2.0)
    _g3_ok = bool(_eth_rh4  is not None and _eth_rh4  > 0)

    _hard_blocked = [n for n, p in [("Taker", _g1_ok), ("OI Block", _g2_ok), ("ETH", _g3_ok)] if not p]
    st.markdown("##### 🎯 Hard Gates")
    if not _hard_blocked:
        st.success("✅ Hard gates PASS — 3/3")
    else:
        st.warning(f"⏳ Hard gates — bloqueado: {', '.join(_hard_blocked)}")

    _hgc = st.columns(3)
    _tk_s  = f"{_taker_z4:+.2f}" if _taker_z4 is not None else "N/A"
    _oi_s  = f"{_oi_z24_4:+.2f}" if _oi_z24_4 is not None else "N/A"
    _er_s  = f"{_eth_rh4*100:+.3f}%" if _eth_rh4 is not None else "N/A"
    _card(_hgc[0], "🔥", "Taker Z",     _tk_s, "✅" if _g1_ok else "❌", "> 0.3",  "pos" if _g1_ok else "neg")
    _card(_hgc[1], "📊", "OI Block",    _oi_s, "✅" if _g2_ok else "❌", "< 2.0 (bipolar)", "pos" if _g2_ok else "neg")
    _card(_hgc[2], "🌉", "ETH Context", _er_s, "✅" if _g3_ok else "❌", "ret > 0", "pos" if _g3_ok else "neg")

    # Bot 2 DNA filters
    st.markdown("##### 🧬 Bot 2 DNA (Momentum)")
    _dna_sz   = bool(_sz_sol is not None and _sz_sol > 1.3)
    _dna_r1d  = bool(_sol_ret1d is not None and _sol_ret1d > 0)
    _dna_rsi  = bool(_sol_rsi4 is not None and 60 < _sol_rsi4 < 80)
    _dna_ma21 = bool(_sol_ma21 and _sol_cl and _sol_cl > _sol_ma21)
    _dna_cols = st.columns(4)
    _sz4_s    = f"{_sz_sol:.2f}" if _sz_sol is not None else "N/A"
    _r1d4_s   = f"{_sol_ret1d*100:+.2f}%" if _sol_ret1d is not None else "N/A"
    _rsi4_s   = f"{_sol_rsi4:.1f}" if _sol_rsi4 is not None else "N/A"
    _ma4_s    = "ACIMA" if _dna_ma21 else "ABAIXO"
    _card(_dna_cols[0], "💧", "Stablecoin Z", _sz4_s,  "✅" if _dna_sz else "❌",   "> 1.3",      "pos" if _dna_sz else "neg")
    _card(_dna_cols[1], "📈", "Momentum 1d",  _r1d4_s, "✅" if _dna_r1d else "❌",  "> 0",        "pos" if _dna_r1d else "neg")
    _card(_dna_cols[2], "📊", "RSI 60-80",    _rsi4_s, "✅" if _dna_rsi else "❌",  "60 < RSI < 80", "pos" if _dna_rsi else "neg")
    _card(_dna_cols[3], "🔄", "Trend MA21",   _ma4_s,  "✅" if _dna_ma21 else "❌", "close > MA21", "pos" if _dna_ma21 else "neg")

    # Shadow scoring log
    st.markdown("##### 👻 Shadow Scoring Alternative")
    st.caption("Score = taker×2 + eth×1 + oi×1 | Entry se score ≥ 3")
    _shd_path = ROOT / "data/08_shadow/sol_scoring_shadow_log.jsonl"
    try:
        if _shd_path.exists():
            _shd_lines = _shd_path.read_text().strip().splitlines()
            if _shd_lines:
                _shd_entries = [json.loads(l) for l in _shd_lines]
                _shd_df = pd.DataFrame(_shd_entries)
                _shd_total = len(_shd_df)
                _shd_would_enter = int(_shd_df.get("scoring_would_enter", pd.Series(dtype=bool)).sum()) if "scoring_would_enter" in _shd_df.columns else 0
                _shd_cols = st.columns(2)
                _shd_cols[0].markdown(f'<div class="cg-card" style="text-align:center;padding:8px;"><div class="cg-card-title">SHADOW ENTRIES</div><div style="font-size:18px;font-weight:700;">{_shd_total}</div></div>', unsafe_allow_html=True)
                _shd_cols[1].markdown(f'<div class="cg-card" style="text-align:center;padding:8px;"><div class="cg-card-title">WOULD-ENTER</div><div style="font-size:18px;font-weight:700;">{_shd_would_enter}</div></div>', unsafe_allow_html=True)
                _disp_cols = ["timestamp", "score_total", "scoring_would_enter", "breakdown"] if all(c in _shd_df.columns for c in ["timestamp", "score_total", "scoring_would_enter"]) else list(_shd_df.columns[:5])
                st.dataframe(_shd_df[_disp_cols].tail(5), use_container_width=True, hide_index=True)
            else:
                st.info("Shadow log existe mas está vazio.")
        else:
            st.info("Shadow log não encontrado — aguardando primeiro ciclo.")
    except Exception as _shd_e:
        st.caption(f"Shadow log indisponível: {_shd_e}")

    # Bot 4 trades
    st.markdown("#### 📊 Histórico Bot 4 — Trades")
    _sol_tdf = load_sol_trades_json()
    if _sol_tdf.empty:
        if _sol_has_pos:
            st.info(f"Nenhum trade Bot 4 completado. Primeira posição ativa: ${_sol_port.get('entry_price', 0):,.2f} (paper).")
        else:
            st.info("Nenhum trade Bot 4 completado ainda.")
    else:
        _m4 = compute_bot_metrics(_sol_tdf)
        if _m4["n_trades"] < 3:
            st.caption("⚠️ Métricas preliminares.")
        _mc4 = st.columns(5)
        _mc4[0].markdown(f'<div class="cg-card" style="text-align:center;padding:8px;"><div class="cg-card-title">TRADES</div><div style="font-size:16px;font-weight:700;">{_m4["n_trades"]}</div><div class="cg-card-sub">{_m4["wins"]}W / {_m4["losses"]}L</div></div>', unsafe_allow_html=True)
        _wr4c = "pos" if _m4["win_rate"] >= 50 else "neg"
        _mc4[1].markdown(f'<div class="cg-card" style="text-align:center;padding:8px;"><div class="cg-card-title">WIN RATE</div><div class="cg-card-value {_wr4c}">{_m4["win_rate"]:.0f}%</div></div>', unsafe_allow_html=True)
        _pf4c = "pos" if _m4["profit_factor"] >= 1.0 else "neg"
        _pf4v = "∞" if _m4["profit_factor"] >= 99 else f'{_m4["profit_factor"]:.2f}'
        _mc4[2].markdown(f'<div class="cg-card" style="text-align:center;padding:8px;"><div class="cg-card-title">PROFIT FACTOR</div><div class="cg-card-value {_pf4c}">{_pf4v}</div></div>', unsafe_allow_html=True)
        _tr4c = "pos" if _m4["total_return"] >= 0 else "neg"
        _mc4[3].markdown(f'<div class="cg-card" style="text-align:center;padding:8px;"><div class="cg-card-title">RETORNO</div><div class="cg-card-value {_tr4c}">{_m4["total_return"]:+.2f}%</div></div>', unsafe_allow_html=True)
        _mc4[4].markdown(f'<div class="cg-card" style="text-align:center;padding:8px;"><div class="cg-card-title">MAX DD</div><div class="cg-card-value neg">{_m4["max_drawdown"]:.2f}%</div></div>', unsafe_allow_html=True)

    # =========================================================================
    # SECTION 4: AI ANALYST (DeepSeek)
    # =========================================================================
    st.markdown("---")
    st.markdown("### 🤖 AI Analyst (DeepSeek)")

    if st.button("🔍 Gerar Análise", type="primary"):
        # Price structure fields from spot_df
        _bb_upper  = _latest(spot_df, "bb_upper",  None)
        _bb_middle = _latest(spot_df, "bb_middle", None)
        _bb_lower  = _latest(spot_df, "bb_lower",  None)
        _ma50      = _latest(spot_df, "ma_50",     None)
        _ma99      = _latest(spot_df, "ma_99",     None)
        _ma200_sp  = _latest(spot_df, "ma_200",    None)  # spot_df value (fallback for ma200_val)
        _atr_14    = _latest(spot_df, "atr_14",    None)
        _high_7d   = _latest(spot_df, "high_7d",   None)
        _low_7d    = _latest(spot_df, "low_7d",    None)
        _high_30d  = _latest(spot_df, "high_30d",  None)
        _low_30d   = _latest(spot_df, "low_30d",   None)

        # Fed events in next 14 days from calendar
        _today = pd.Timestamp.now(tz="UTC").date()
        _fed_events_14d = []
        for _ev in (cal.get("fomc_dates", []) + cal.get("hearings", [])
                    + cal.get("transitions", [])):
            _raw = _ev.get("date") or _ev.get("start") or _ev.get("decision_date")
            if _raw:
                try:
                    _d = pd.Timestamp(_raw).date()
                    _days = (_d - _today).days
                    if 0 <= _days <= 14:
                        _fed_events_14d.append(
                            f"{_d} ({_days}d): {_ev.get('description') or _ev.get('type','?')}"
                        )
                except Exception:
                    pass

        # Fed Observatory (fast — reads parquets, no API)
        _fed_obs = {}
        if _FED_OBS_AVAILABLE:
            try:
                _fed_obs = estimate_rate_probability(load_fed_data())
            except Exception:
                pass

        _news_det = cdet.get("news", {})
        context = {
            # Preço e regime
            "ts": pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d %H:%M UTC"),
            "price": price, "pct_24h": pct_24h, "regime": regime,
            "rsi": rsi_14 or 50.0, "bb_pct": bb_pct or 0.5,
            "bb_upper": _bb_upper, "bb_middle": _bb_middle, "bb_lower": _bb_lower,
            "ma50": _ma50, "ma100": _ma99,
            "ma200_val": ma200_val or _ma200_sp,
            "ma200_pct": ma200_pct, "ma200_slope": ma200_slope,
            "atr_14": _atr_14,
            "high_7d": _high_7d, "low_7d": _low_7d,
            "high_30d": _high_30d, "low_30d": _low_30d,
            # Sinal e score
            "signal": signal, "score": total_score,
            "score_raw": total_score_raw, "regime_multiplier": regime_multiplier,
            "threshold": threshold, "block_reason": block_reason_computed,
            # Clusters
            "c_technical": clusters["technical"], "c_positioning": clusters["positioning"],
            "c_macro": clusters["macro"], "c_liquidity": clusters["liquidity"],
            "c_sentiment": clusters["sentiment"], "c_news": clusters["news"],
            # Derivativos
            "oi_z": zs.get("oi_z", 0), "funding_z": zs.get("funding_z", 0),
            "taker_z": zs.get("taker_z", 0),
            "ls_account": ls_account_val, "ls_position": ls_position_val,
            "whale_signal": ws_label,
            # Liquidez
            "stablecoin_z": zs.get("stablecoin_z", 0), "etf_z": zs.get("etf_z", 0),
            # Macro z-scores
            "dgs10_z": zs.get("dgs10_z", 0), "dgs2_z": zs.get("dgs2_z", 0),
            "curve_z": zs.get("curve_z", 0), "rrp_z": zs.get("rrp_z", 0),
            # Fed
            "fed_event": fomc.get("next_event", "N/A"),
            "fed_days": fomc.get("days_away", "—"),
            "fed_adj": fomc.get("proximity_adj", 0),
            "fed_events_14d": _fed_events_14d,
            "fed_obs": _fed_obs,
            # Sentimento
            "fg_val": fg_val, "fg_cls": fg_cls,
            "fg_z": zs.get("fg_z", 0), "bubble_z": zs.get("bubble_z", 0),
            "crypto_news_score": _news_det.get("crypto_score", 0),
            "fed_news_score": _news_det.get("fed_score", 0),
            # Capital
            "capital": capital, "has_position": portfolio.get("has_position", False),
        }
        with st.spinner("Consultando DeepSeek..."):
            analysis = call_deepseek_analyst(context)
        st.markdown('<div class="cg-card" style="font-size:14px; line-height:1.7;">',
                    unsafe_allow_html=True)
        st.markdown(analysis)
        st.markdown('</div>', unsafe_allow_html=True)

    # =========================================================================
    # SECTION 4: WHALE TRACKING — DESABILITADO TEMPORARIAMENTE
    # =========================================================================
    # TODO (futuro): reativar após validação estatística do L/S ratio.
    # Análise pendente:
    #   - Correlação L/S ratio vs forward returns (estilo SOL EDA Phase 1)
    #   - Cohen's d em shocks ±2σ
    #   - Se |corr| > 0.1 e p < 0.05 → considerar gate no Bot 1 ou Bot 4
    # Código preservado abaixo — não deletar.
    #
    # ENABLE_WHALE_TRACKING = False
    #
    # st.markdown("---")
    # st.markdown("### 🐋 Whale Tracking")
    # col_w1, col_w2 = st.columns([1, 2])
    # with col_w1:
    #     ws_c = "neg" if "🔴" in ws_label else ("pos" if "🟢" in ws_label else "neut")
    #     lsa_delta = ""
    #     if len(lsa_df) > 24:
    #         lsa_prev = lsa_df["longShortRatio"].iloc[-25]
    #         lsa_d = (ls_account_val - lsa_prev)
    #         lsa_delta = f"({lsa_d:+.3f} vs 24h)"
    #     lsp_c = "pos" if (ls_position_val or 1) > 1.0 else "neg"
    #     st.markdown(f"""
    # <div class="cg-card">
    #   <div class="cg-card-title">Top Accounts L/S Ratio</div>
    #   <div class="cg-card-value {ws_c}">{ls_account_val:.3f} {lsa_delta}</div>
    #   <div class="cg-card-sub">Top Positions: <span class="{lsp_c}">{ls_position_val:.3f}</span></div>
    #   <div class="cg-card-interp"><strong>{ws_label}</strong><br>{ws_text}</div>
    # </div>""", unsafe_allow_html=True)
    # with col_w2:
    #     if not lsa_df.empty and len(lsa_df) > 48:
    #         lsa_7d = lsa_df.tail(168)
    #         spot_7d = spot_df.tail(168) if not spot_df.empty else pd.DataFrame()
    #         fig = make_subplots(specs=[[{"secondary_y": True}]])
    #         fig.add_trace(go.Scatter(
    #             x=lsa_7d["timestamp"], y=lsa_7d["longShortRatio"],
    #             name="L/S Accounts", line=dict(color=BLUE, width=1.5)
    #         ), secondary_y=False)
    #         if not spot_7d.empty:
    #             fig.add_trace(go.Scatter(
    #                 x=spot_7d["timestamp"], y=spot_7d["close"],
    #                 name="BTC Price", line=dict(color=AMBER, width=1.5, dash="dot")
    #             ), secondary_y=True)
    #         fig.add_hline(y=1.0, line_dash="dash", line_color=GREY, opacity=0.4)
    #         fig.update_layout(**PLOTLY, height=200, showlegend=True,
    #                           legend=dict(orientation="h", y=1.1))
    #         fig.update_yaxes(title_text="L/S Ratio", secondary_y=False)
    #         fig.update_yaxes(title_text="BTC $", secondary_y=True)
    #         st.plotly_chart(fig, use_container_width=True)

    # SECTION 5: DERIVATIVES — REMOVIDO (já implícito nos bots)

    # SECTION 6: MACRO — REMOVIDO (já implícito no Gate Scoring)

    # =========================================================================
    # SECTION 8: NEWS & SENTIMENT
    # =========================================================================
    st.markdown("---")
    st.markdown("### 📰 News & Sentiment")

    col_n1, col_n2 = st.columns([2, 1])

    with col_n1:
        # News scores bar
        c_ns = "pos" if (combined_ns or 0) > 0 else "neg"
        st.markdown(f"""
<div class="cg-card" style="margin-bottom:12px;">
  <div style="display:flex; gap:24px; align-items:center;">
    <span class="cg-card-title" style="margin:0;">SCORES DEEPSEEK</span>
    <span>Crypto: <b class="{'pos' if (crypto_ns or 0)>0 else 'neg'}">{crypto_ns:+.2f}</b></span>
    <span>Macro: <b class="{'pos' if (macro_ns or 0)>0 else 'neg'}">{macro_ns:+.2f}</b></span>
    <span>Combined: <b class="{c_ns}">{combined_ns:+.2f}</b></span>
  </div>
</div>""", unsafe_allow_html=True)

        # Recent news feed
        all_news = []
        for cat in ["crypto", "macro", "fed"]:
            df_n = load_news(cat)
            if not df_n.empty:
                all_news.append(df_n)
        if all_news:
            news_all = pd.concat(all_news).sort_values("timestamp", ascending=False).head(12)
            st.markdown("**Notícias Recentes**")
            for _, row in news_all.iterrows():
                cat = row.get("category", "")
                tag_class = {"crypto": "tag-crypto", "macro": "tag-macro", "fed": "tag-fed"}.get(cat, "tag-macro")
                ds_score = row.get("ds_score")
                score_str = ""
                if ds_score is not None and not (isinstance(ds_score, float) and np.isnan(ds_score)):
                    sc = float(ds_score)
                    sc_c = "pos" if sc > 1 else ("neg" if sc < -1 else "neut")
                    score_str = f' <span class="{sc_c}">({sc:+.0f})</span>'
                ts_str = row["timestamp"].strftime("%m/%d %H:%M") if pd.notna(row.get("timestamp")) else ""
                title_trunc = str(row.get("title", ""))[:90]
                st.markdown(f"""
<div class="news-row">
  <span class="{tag_class}">{cat.upper()}</span> {score_str}
  <span style="color:#e6edf3;"> {title_trunc}</span>
  <span style="color:#484f58; margin-left:8px; font-size:11px;">{ts_str}</span>
</div>""", unsafe_allow_html=True)

    with col_n2:
        # F&G + Fed Sentinel card
        fg_c = "neg" if (fg_val or 0) < 30 else ("warn" if fg_val < 60 else "pos")
        st.markdown(f"""
<div class="cg-card">
  <div class="cg-card-title">😱 Fear & Greed</div>
  <div class="cg-card-value {fg_c}">{fg_val:.0f}</div>
  <div class="cg-card-sub">{fg_cls}</div>
  <div class="cg-card-sub">z={zs.get('fg_z',0):.2f}</div>
</div>""", unsafe_allow_html=True)



    # =========================================================================
    # SECTION 9: FED SENTINEL (consolidado — Observatory + Sentinel)
    # =========================================================================
    st.markdown("---")
    st.markdown("### 🏛️ Fed Sentinel")
    st.caption("Probabilidades estimadas (proxy DGS2, não FedWatch)")

    try:
        from src.features.fed_observatory import load_fed_data, estimate_rate_probability, get_scenario_analysis

        _fed_data = load_fed_data()
        _prob     = estimate_rate_probability(_fed_data)
        _analysis = get_scenario_analysis(_prob)

        # --- Probability cards ---
        col_fo1, col_fo2, col_fo3 = st.columns(3)
        with col_fo1:
            _pct = _prob["prob_cut"] * 100
            _col = "#3fb950" if _pct > 30 else "#8b949e"
            st.markdown(f"""
<div class="cg-card">
  <div class="cg-card-title">🟢 CORTE 25bps</div>
  <div class="cg-card-value" style="color:{_col};">{_pct:.0f}%</div>
  <div class="cg-card-sub">BTC: Bullish</div>
</div>""", unsafe_allow_html=True)

        with col_fo2:
            _pct = _prob["prob_hold"] * 100
            st.markdown(f"""
<div class="cg-card">
  <div class="cg-card-title">🟡 MANUTENÇÃO</div>
  <div class="cg-card-value" style="color:#8b949e;">{_pct:.0f}%</div>
  <div class="cg-card-sub">BTC: Neutro</div>
</div>""", unsafe_allow_html=True)

        with col_fo3:
            _pct = _prob["prob_hike"] * 100
            _col = "#f85149" if _pct > 10 else "#8b949e"
            st.markdown(f"""
<div class="cg-card">
  <div class="cg-card-title">🔴 ALTA 25bps</div>
  <div class="cg-card-value" style="color:{_col};">{_pct:.0f}%</div>
  <div class="cg-card-sub">BTC: Bearish</div>
</div>""", unsafe_allow_html=True)

        # --- Sentinel status (next event) ---
        st.markdown("**📅 Próximo evento**")
        _fs_c1, _fs_c2, _fs_c3, _fs_c4 = st.columns(4)
        _fs_c1.metric("Evento", fomc.get("next_event", "N/A"))
        _fs_c2.metric("Em", f"{fomc.get('days_away', '—')} dias")
        _fs_c3.metric("Proximity adj", f"+{fomc.get('proximity_adj', 0):.1f}")
        _fs_c4.metric("Blackout", "Sim ⚠️" if fomc.get("in_blackout") else "Não")

        # --- Indicators ---
        _ind = _prob.get("indicators", {})
        _ind_cols = st.columns(5)
        _ind_list = [
            ("DGS2",          _ind.get("dgs2"),         "%",   _ind.get("dgs2_change_30d")),
            ("EFFR",          _ind.get("effr"),          "%",   None),
            ("Spread vs Fed", _ind.get("spread_vs_fed"), "bps", None),
            ("Inflation 5Y",  _ind.get("inflation_5y"),  "%",   _ind.get("inflation_5y_change_30d")),
            ("Inflation 10Y", _ind.get("inflation_10y"), "%",   None),
        ]
        for _i, (_name, _val, _unit, _change) in enumerate(_ind_list):
            with _ind_cols[_i]:
                if _val is not None:
                    if _unit == "bps":
                        st.metric(_name, f"{_val*100:+.0f}bps")
                    else:
                        st.metric(_name, f"{_val:.2f}%",
                                  delta=f"{_change:+.2f}" if _change is not None else None,
                                  delta_color="inverse")
                else:
                    st.metric(_name, "N/A")

        # --- Scenarios ---
        st.markdown("**🎯 Cenários por decisão:**")
        _color_map = {"green": "#3fb950", "gray": "#8b949e", "red": "#f85149"}
        for _sc in _analysis["scenarios"]:
            _bc = _color_map.get(_sc["color"], "#8b949e")
            st.markdown(f"""
<div style="background:#161b22; border-left:3px solid {_bc};
            padding:10px 15px; margin:5px 0; border-radius:4px;">
  <strong>{_sc["name"]}</strong> — {_sc["probability"]}
  <br><span style="font-size:0.85em; color:#8b949e;">
  Impacto BTC: {_sc["btc_impact"]} | Ação: {_sc["action"]}
  </span>
  <br><span style="font-size:0.8em; color:#6e7681;">{_sc["description"]}</span>
</div>""", unsafe_allow_html=True)

        # --- Member sentiment ---
        if _analysis.get("member_summary"):
            _ms = _analysis["member_summary"]
            st.markdown(
                f"**Membros Fed (últimos 30 dias):** "
                f"🔴 Hawkish: {_ms['hawkish']} | "
                f"🟢 Dovish: {_ms['dovish']} | "
                f"⚪ Neutro: {_ms['neutral']} "
                f"→ Tendência: **{_ms['trend']}**"
            )

        # --- Fed agenda ---
        try:
            with open("conf/fed_calendar.json") as _f:
                _cal = json.load(_f)
            _now_ts = pd.Timestamp.now(tz="UTC")
            _all_events = []
            for _d in _cal.get("fomc_decisions", []):
                _all_events.append({"date": _d, "type": "FOMC Decision"})
            for _e in _cal.get("hearings", []):
                _all_events.append({"date": _e["date"], "type": f"{_e['type']} ({_e.get('member', '')})"})
            for _e in _cal.get("transitions", []):
                _all_events.append({"date": _e["date"], "type": f"{_e['type']} ({_e.get('member', '')})"})
            _upcoming = sorted(
                [e for e in _all_events if pd.to_datetime(e["date"]) > _now_ts],
                key=lambda x: x["date"]
            )[:5]
            if _upcoming:
                st.markdown("**Agenda Fed:**")
                for _e in _upcoming:
                    _days = (pd.to_datetime(_e["date"]) - _now_ts).days
                    st.markdown(f"📅 {_e['date']} — {_e['type']} ({_days}d)")
        except Exception:
            pass

    except Exception as _e:
        st.warning(f"Fed Sentinel indisponível: {_e}")

        # =========================================================================
    # SECTION 7: SYSTEM HEALTH
    # =========================================================================
    st.markdown("---")
    st.markdown("### ⚙️ System Health")

    col_h1, col_h2 = st.columns([1, 2])

    with col_h1:
        # Staleness check
        stale_checks = {
            "Binance Spot":  (_age_h(spot_df),    3),
            "Futures OI":    (_age_h(oi_df),       3),
            "FRED Macro":    (_age_h(macro_df),   48),
            "CoinGlass":     (_age_h(bubble_df),  72),
            "Fear & Greed":  (_age_h(fg_df),      48),
            "News":          (_age_h(load_news("crypto")), 4),
            "Z-scores":      (_age_h(zs_df),       3),
        }
        stale_list = []
        st.markdown("**Fontes de dados:**")
        for name, (age, tol) in stale_checks.items():
            ok = age < tol
            icon = "✅" if ok else "⚠️"
            age_str = f"{age:.1f}h" if age < 9999 else "MISSING"
            color = "" if ok else "color:#d29922;"
            st.markdown(f'<span style="{color}">{icon} {name}: {age_str}</span>', unsafe_allow_html=True)
            if not ok:
                stale_list.append(name)

        st.markdown(f"**Stale:** {len(stale_list)} fonte(s)" + (f": {', '.join(stale_list)}" if stale_list else " — tudo OK"))

        # DeepSeek balance card
        _ds = get_deepseek_balance()
        _ds_bal = _ds["balance_usd"]
        _ds_avail = _ds["available"]
        _ds_err = _ds["error"]
        if _ds_err:
            _ds_color = "#f85149"
            _ds_status = "🔴 erro"
        elif _ds_bal > 1.00:
            _ds_color = "#3fb950"
            _ds_status = "✅ disponível"
        elif _ds_bal > 0.20:
            _ds_color = "#d29922"
            _ds_status = "🟡 baixo"
        else:
            _ds_color = "#f85149"
            _ds_status = "🔴 crítico"
        st.markdown("**DeepSeek API:**")
        st.markdown(
            f'<div style="border-left:3px solid {_ds_color};padding:6px 10px;margin:4px 0;'
            f'background:rgba(0,0,0,0.15);border-radius:4px;font-size:0.85em;">'
            f'💰 Saldo: <b style="color:{_ds_color};">${_ds_bal:.2f} USD</b> — {_ds_status}<br>'
            f'G2a (chat):&nbsp;&nbsp;ativo<br>'
            f'G2b (R1):&nbsp;&nbsp;&nbsp;&nbsp;shadow mode'
            f'{"<br><span style=\"color:#d29922;font-size:0.8em;\">⚠️ " + _ds_err[:60] + "</span>" if _ds_err else ""}'
            f'</div>',
            unsafe_allow_html=True,
        )

        # Cron info
        st.markdown("**Cron:**")
        st.markdown(f"Hourly: {cycle_age} {cycle_ok}")
        if cycle_info.get("last_line"):
            st.caption(cycle_info["last_line"][-80:])

    with col_h2:
        # Score history chart (7 days)
        if not score_hist.empty and len(score_hist) > 2:
            sh = score_hist.tail(168)
            regime_hist_7d = regime_df.tail(7) if not regime_df.empty else pd.DataFrame()

            fig = go.Figure()
            # Regime background bands
            if not regime_hist_7d.empty:
                for _, row in regime_hist_7d.iterrows():
                    r_color = {"Bull": "rgba(63,185,80,0.08)", "Bear": "rgba(248,81,73,0.08)"}.get(row["regime"], "rgba(210,153,34,0.08)")
                    next_day = row["timestamp"] + pd.Timedelta(days=1)
                    fig.add_vrect(x0=row["timestamp"], x1=next_day, fillcolor=r_color, layer="below", line_width=0)

            fig.add_trace(go.Scatter(
                x=sh["timestamp"], y=sh["total_score"],
                name="Score", line=dict(color=BLUE, width=2),
                fill="tozeroy", fillcolor="rgba(88,166,255,0.1)",
            ))
            if "threshold" in sh.columns:
                fig.add_trace(go.Scatter(
                    x=sh["timestamp"], y=sh["threshold"],
                    name="Threshold", line=dict(color=AMBER, width=1.5, dash="dash"),
                ))
            fig.add_hline(y=0, line_color=GREY, opacity=0.3)
            fig.update_layout(**PLOTLY, height=220, title="Score History (últimas 168 leituras)")
            st.plotly_chart(fig, use_container_width=True)

    # ── Calibration data (shared by Model Health + Calibration Alerts) ────────
    _GATE_CORR_MAP = {
        "oi_z":         ("g4_oi",       "G4 OI"),
        "taker_z":      ("g9_taker",    "G9 Taker"),
        "funding_z":    ("g10_funding", "G10 Funding"),
        "dgs10_z":      ("g3_dgs10",    "G3 DGS10"),
        "curve_z":      ("g3_curve",    "G3 Curve"),
        "stablecoin_z": ("g5_stable",   "G5 Stablecoin"),
        "bubble_z":     ("g6_bubble",   "G6 Bubble"),
        "etf_z":        ("g7_etf",      "G7 ETF"),
        "fg_z":         ("g8_fg",       "G8 F&G"),
    }
    calib_rows: list = []
    _calib_error: str = ""
    try:
        _params = load_params()
        _gp = _params.get("gate_params", {})
        _zs_cal = zs_df.copy()
        if "timestamp" not in _zs_cal.columns:
            _zs_cal = _zs_cal.reset_index()
        _zs_cal["timestamp"] = pd.to_datetime(_zs_cal["timestamp"], utc=True)
        _zs_cal = _zs_cal.set_index("timestamp").resample("1D").last()
        _spot_cal = spot_df.copy()
        if not _spot_cal.empty:
            _spot_cal["timestamp"] = pd.to_datetime(_spot_cal["timestamp"], utc=True)
            _spot_ret = _spot_cal.set_index("timestamp").resample("1D")["close"].last().pct_change(3) * 100
            _spot_ret = _spot_ret.shift(-1)
        else:
            _spot_ret = pd.Series(dtype=float)

        for zcol, (gkey, gname) in _GATE_CORR_MAP.items():
            if zcol not in _zs_cal.columns or _spot_ret.empty:
                continue
            _m = _zs_cal[zcol].to_frame().join(_spot_ret.rename("ret3d")).dropna()
            _m30 = _m.tail(30)
            if len(_m30) < 10:
                continue
            corr_now = float(_m30[zcol].corr(_m30["ret3d"]))
            corr_cfg = float(_gp.get(gkey, [0])[0]) if gkey in _gp else None
            if corr_cfg is None:
                continue
            diff = abs(corr_now - corr_cfg)
            _gcfg = _gp.get(gkey, [])
            weight = float(_gcfg[2]) if len(_gcfg) >= 3 else 1.0
            calib_rows.append({
                "gate": gname, "corr_cfg": corr_cfg, "corr_30d": corr_now,
                "diff": diff, "weight": weight, "n": len(_m30),
            })
    except Exception as _e:
        _calib_error = str(_e)

    # ── BOT 1 MONITORAMENTO (Model Health + Calibration + Adaptive — consolidado) ──
    st.markdown("---")
    st.markdown("### 🛡️ BOT 1 - Monitoramento")
    st.caption("Saúde estatística do Gate Scoring v2 — calibração, adaptive weights e kill switches")

    if _calib_error:
        st.warning(f"⚠️ Dados de calibração indisponíveis: {_calib_error}")
    elif not calib_rows:
        st.info("Dados insuficientes para análise (< 10 dias de z-scores)")
    else:
        _mh_cfg     = _params.get("model_health", {})
        _th_healthy = _mh_cfg.get("threshold_healthy", 0.15)
        _th_warning = _mh_cfg.get("threshold_warning", 0.30)

        _total_w   = sum(r["weight"] for r in calib_rows)
        _model_aln = sum(r["diff"] * r["weight"] for r in calib_rows) / _total_w

        _aw_cfg          = _params.get("adaptive_weights", {})
        _adaptive_details = portfolio.get("last_adaptive_weights", {})
        _gc_val          = portfolio.get("last_global_confidence_multiplier", 1.0) or 1.0

        _n_extreme = sum(1 for d in _adaptive_details.values() if d.get("kill_status") == "extreme") if _adaptive_details else 0
        _kill_gates = [d.get("gate", k) for k, d in _adaptive_details.items() if d.get("kill_status") == "extreme"] if _adaptive_details else []

        # ── 3 Summary Cards ────────────────────────────────────────────────
        _mon_c1, _mon_c2, _mon_c3 = st.columns(3)

        # Card 1: Model Alignment
        if _model_aln < _th_healthy:
            _aln_color, _aln_badge = "#3fb950", "🟢 Saudável"
        elif _model_aln < _th_warning:
            _aln_color, _aln_badge = "#d29922", "🟡 Atenção"
        else:
            _aln_color, _aln_badge = "#f85149", "🔴 Desalinhado"

        with _mon_c1:
            st.markdown(f"""
<div class="cg-card" style="text-align:center;padding:12px;border-left:4px solid {_aln_color};">
  <div class="cg-card-title">🎯 MODEL ALIGNMENT</div>
  <div class="cg-card-value" style="color:{_aln_color};">{_model_aln:.3f}</div>
  <div class="cg-card-sub">{_aln_badge}</div>
</div>""", unsafe_allow_html=True)

        # Card 2: Global Confidence
        _reduction = (1 - _gc_val) * 100
        if _gc_val >= 0.8:
            _gc_color, _gc_badge = "#3fb950", "🟢 Normal"
        elif _gc_val >= 0.5:
            _gc_color, _gc_badge = "#d29922", f"🟡 Reduzido -{_reduction:.0f}%"
        else:
            _gc_color, _gc_badge = "#f85149", f"🔴 Severo -{_reduction:.0f}%"

        with _mon_c2:
            _aw_enabled = _aw_cfg.get("enabled", False)
            _gc_disp = f"×{_gc_val:.3f}" if _aw_enabled else "OFF"
            st.markdown(f"""
<div class="cg-card" style="text-align:center;padding:12px;border-left:4px solid {_gc_color};">
  <div class="cg-card-title">⚖️ GLOBAL CONFIDENCE</div>
  <div class="cg-card-value" style="color:{_gc_color};">{_gc_disp}</div>
  <div class="cg-card-sub">{_gc_badge if _aw_enabled else "adaptive weights off"}</div>
</div>""", unsafe_allow_html=True)

        # Card 3: Kill Switches
        if _n_extreme == 0:
            _ks_color, _ks_badge, _ks_val = "#3fb950", "🟢 Nenhum", "0 gates"
        elif _n_extreme <= 2:
            _ks_color = "#d29922"
            _ks_badge = "🟡 Alguns"
            _ks_val = f"{_n_extreme} gates"
        else:
            _ks_color = "#f85149"
            _ks_badge = "🔴 Múltiplos"
            _ks_val = f"{_n_extreme} gates"

        with _mon_c3:
            _ks_names = ", ".join(_kill_gates[:3]) if _kill_gates else "—"
            st.markdown(f"""
<div class="cg-card" style="text-align:center;padding:12px;border-left:4px solid {_ks_color};">
  <div class="cg-card-title">💀 KILL SWITCHES</div>
  <div class="cg-card-value" style="color:{_ks_color};">{_ks_val}</div>
  <div class="cg-card-sub">{_ks_badge}: {_ks_names}</div>
</div>""", unsafe_allow_html=True)

        # Leitura interpretativa
        if _model_aln >= _th_warning or _gc_val < 0.5:
            st.warning("⚠️ Bot 1 com capacidade reduzida. Considerar recalibrar gates em 2-4 semanas.")
        elif _model_aln >= _th_healthy or _gc_val < 0.8:
            st.info("ℹ️ Bot 1 com leve degradação. Monitorar evolução.")
        else:
            st.success("✅ Bot 1 saudável. Gates alinhados com correlações configuradas.")

        # ── Detalhes completos em expander ─────────────────────────────────
        with st.expander("🔍 Detalhes completos (Calibration + Adaptive por gate)", expanded=False):

            # Calibration Alerts
            st.markdown("**📐 Calibration Alerts (rolling 30d corr vs parameters.yml)**")
            _cal_rows = sorted(calib_rows, key=lambda x: -x["diff"])
            _cal_data = []
            for r in _cal_rows:
                icon = "🔴" if r["diff"] > 0.25 else ("⚠️" if r["diff"] > 0.15 else "✅")
                _cal_data.append({
                    "": icon,
                    "Gate": r["gate"],
                    "corr_cfg": f"{r['corr_cfg']:+.3f}",
                    "corr_30d": f"{r['corr_30d']:+.3f}",
                    "Δ": f"{r['diff']:.3f}",
                    "n": r["n"],
                })
            if _cal_data:
                st.dataframe(pd.DataFrame(_cal_data), use_container_width=True, hide_index=True)

            # Adaptive Weights detail
            if _adaptive_details:
                st.markdown("**⚖️ Adaptive Weights por Gate**")
                _aw_data = []
                for _gkey, _d in sorted(_adaptive_details.items(), key=lambda x: x[1].get("delta") or 0, reverse=True):
                    _ks = _d.get("kill_status", "ok")
                    _cf = _d.get("confidence", 0)
                    icon = "⛔" if _ks == "extreme" else ("⚠️" if _ks == "severe" else ("🟡" if _cf < 0.8 else "✅"))
                    _aw_data.append({
                        "": icon,
                        "Gate": _d.get("gate", _gkey),
                        "Base": f"{_d.get('base_weight', 0):.2f}",
                        "Eff":  f"{_d.get('effective_weight', 0):.2f}",
                        "Conf": f"{_cf:.2f}",
                        "Δ":    f"{_d.get('delta') or 0:.3f}",
                        "cfg":  f"{_d.get('corr_cfg') or 0:+.3f}",
                        "real": f"{_d.get('corr_long') or 0:+.3f}",
                    })
                if _aw_data:
                    st.dataframe(pd.DataFrame(_aw_data), use_container_width=True, hide_index=True)

                # Mini summary
                _mean_conf = float(np.mean([d.get("confidence", 0) for d in _adaptive_details.values()]))
                _n_ok  = sum(1 for d in _adaptive_details.values() if d.get("kill_status") == "ok" and d.get("confidence", 0) > 0.8)
                _n_red = sum(1 for d in _adaptive_details.values() if d.get("kill_status") == "ok" and d.get("confidence", 0) <= 0.8)
                _n_sev = sum(1 for d in _adaptive_details.values() if d.get("kill_status") == "severe")
                _sc1, _sc2, _sc3, _sc4, _sc5 = st.columns(5)
                _sc1.metric("Mean Conf", f"{_mean_conf:.2f}")
                _sc2.metric("✅ OK", _n_ok)
                _sc3.metric("🟡 Reduced", _n_red)
                _sc4.metric("⚠️ Severe", _n_sev)
                _sc5.metric("⛔ Extreme", _n_extreme)

    # ── PERFORMANCE MONITORING (Bots 2/3/4) ───────────────────────────────────
    st.markdown("---")
    st.markdown("### 📊 Performance Monitoring")
    st.caption("Saúde de performance dos bots momentum-based (Bot 1 tem monitoring próprio acima)")

    # Expected Sharpe from backtests
    _EXPECTED_SHARPE = {"bot2": 2.71, "bot3": 0.64, "bot4": 2.03}

    def _compute_bot_health(trades_df, expected_sharpe=None):
        if trades_df.empty:
            return {"status": "⏳", "label": "Sem trades", "n": 0}
        n = len(trades_df)
        rets = trades_df["return_pct"].astype(float)
        wins = int((rets > 0).sum())
        wr   = wins / n * 100
        total_ret = float(((1 + rets / 100).prod() - 1) * 100)
        if n < 10:
            return {"status": "⏳", "label": f"Início (n={n})", "n": n, "wr": wr, "total": total_ret,
                    "msg": f"{10-n} trades para análise completa"}
        gross_win  = float(rets[rets > 0].sum())
        gross_loss = float(abs(rets[rets <= 0].sum()))
        pf  = gross_win / gross_loss if gross_loss > 0 else 99.0
        equity = (1 + rets / 100).cumprod()
        dd  = float((equity / equity.cummax() - 1).min() * 100)
        sharpe = None
        if n >= 20 and rets.std() > 0:
            sharpe = float((rets.mean() / rets.std()) * (252 ** 0.5))
        # health classification
        if wr < 40 or dd < -10 or (sharpe is not None and sharpe < 0):
            status, label = "🔴", "Crítico"
        elif wr < 50 or dd < -5 or (sharpe is not None and expected_sharpe and sharpe < expected_sharpe * 0.5):
            status, label = "🟡", "Atenção"
        else:
            status, label = "🟢", "Saudável"
        msg_parts = []
        if sharpe is not None and expected_sharpe:
            msg_parts.append(f"Sharpe {sharpe:.2f} ({sharpe/expected_sharpe*100:.0f}% do backtest {expected_sharpe:.2f})")
        if dd < -3:
            msg_parts.append(f"DD {dd:.1f}%")
        msg = " | ".join(msg_parts) if msg_parts else "Performance alinhada com expectativa"
        return {"status": status, "label": label, "n": n, "wr": wr, "pf": pf, "dd": dd,
                "sharpe": sharpe, "total": total_ret, "msg": msg}

    def _load_bot_trades(asset, bot_key):
        """Load trades for a given asset/bot."""
        if asset == "sol":
            return load_sol_trades_json()
        path_map = {"btc": "data/05_output/trades.parquet",
                    "eth": "data/05_output/trades_eth.parquet"}
        df = load_parquet(path_map.get(asset, "data/05_output/trades.parquet"))
        if df.empty:
            return df
        if "entry_bot" in df.columns:
            df = df[df["entry_bot"] == bot_key]
        return df.reset_index(drop=True)

    _perf_bots = [
        {"key": "bot2", "emoji": "🚀", "name": "BOT 2 - Momentum (BTC)", "asset": "btc",
         "open_port": None},
        {"key": "bot3", "emoji": "⚡", "name": "BOT 3 - Volume Defensivo (ETH)", "asset": "eth",
         "open_port": _eth_port if "_eth_port" in dir() else {}},
        {"key": "bot4", "emoji": "🟣", "name": "BOT 4 - Taker/Flow (SOL)",  "asset": "sol",
         "open_port": _sol_port if "_sol_port" in dir() else {}},
    ]

    for _pb in _perf_bots:
        st.markdown(f"#### {_pb['emoji']} {_pb['name']}")
        _pt = _load_bot_trades(_pb["asset"], _pb["key"])
        _ph = _compute_bot_health(_pt, _EXPECTED_SHARPE.get(_pb["key"]))

        if _ph["n"] == 0:
            # Check for open position
            _op = _pb.get("open_port") or {}
            if _op.get("has_position"):
                _ep_v = _op.get("entry_price", 0)
                st.info(f"⏳ Nenhum trade fechado — posição aberta: ${_ep_v:,.2f} (paper). Aguardando primeiro fechamento.")
            else:
                st.info("⏳ Aguardando primeiro trade.")
        else:
            _pc = st.columns(4)
            _pc[0].metric("Trades", _ph["n"])
            _wr_delta = f"🟢 OK" if _ph["wr"] >= 50 else "🔴 Baixo"
            _pc[1].metric("Win Rate", f"{_ph['wr']:.1f}%", delta=_wr_delta, delta_color="off")
            if "pf" in _ph:
                _pf_d = "∞" if _ph["pf"] >= 99 else f"{_ph['pf']:.2f}"
                _pc[2].metric("Profit Factor", _pf_d)
            else:
                _pc[2].metric("Avg Return", f"{_ph.get('avg', 0):+.2f}%")
            if _ph.get("sharpe") is not None:
                _pc[3].metric("Sharpe (anual)", f"{_ph['sharpe']:.2f}",
                              delta=f"{_ph['sharpe']/_EXPECTED_SHARPE[_pb['key']]*100:.0f}% do backtest" if _pb["key"] in _EXPECTED_SHARPE else None,
                              delta_color="off")
            else:
                _pc[3].metric("Total Retorno", f"{_ph['total']:+.2f}%")

            _status_fn = {"🔴": st.error, "🟡": st.warning, "🟢": st.success}.get(_ph["status"], st.info)
            _status_fn(f"{_ph['status']} {_ph['label']} — {_ph['msg']}")

    # SECTION 8: PAPER TRADING — REMOVIDO (trades exibidos em Bot 1/Bot 2 acima)
    # Variáveis necessárias para Safety Status abaixo
    _params     = load_params()
    _cm_params  = _params.get("capital_management", {})
    _cm_enabled = _cm_params.get("enabled", False)
    _buckets    = portfolio.get("buckets", {})
    _mf_params  = _params.get("momentum_filter", {})

    # =========================================================================
    # SECTION 9: SAFETY STATUS (Capital Manager — shown only when enabled)
    # =========================================================================
    if _cm_enabled and _buckets:
        st.markdown("---")
        st.markdown("### 🛡️ Safety Status — Capital Manager")

        _safe_cfg = _cm_params.get("safety", {})
        _max_dd   = _safe_cfg.get("max_drawdown_pct", 0.15)
        _max_dl   = _safe_cfg.get("max_daily_loss_pct", 0.05)

        _safe_cols = st.columns(len(_buckets))
        for _i, (_bkey, _bkt) in enumerate(_buckets.items()):
            with _safe_cols[_i]:
                _b_init    = _bkt.get("initial_capital", 1.0)
                _b_cur     = _bkt.get("current_capital", _b_init)
                _b_peak    = _bkt.get("peak_capital", _b_init)
                _b_dd      = (_b_init - _b_cur) / _b_init if _b_init > 0 else 0.0
                _b_dpnl    = _bkt.get("daily_pnl", 0.0)
                _b_dbase   = _bkt.get("daily_capital_base", _b_init) or _b_init
                _b_dloss   = -_b_dpnl / _b_dbase if _b_dpnl < 0 and _b_dbase > 0 else 0.0
                _b_paused  = _bkt.get("paused_until")
                _b_reason  = _bkt.get("pause_reason", "")

                _dd_color   = "#f85149" if _b_dd >= _max_dd else ("#d29922" if _b_dd > _max_dd * 0.7 else "#3fb950")
                _dl_color   = "#f85149" if _b_dloss >= _max_dl else ("#d29922" if _b_dloss > _max_dl * 0.7 else "#3fb950")
                _pause_html = ""
                if _b_paused:
                    try:
                        _pt = pd.Timestamp(_b_paused)
                        if _pt.tzinfo is None:
                            _pt = _pt.tz_localize("UTC")
                        _rem_h = max(0.0, (_pt - pd.Timestamp.now("UTC")).total_seconds() / 3600)
                        _pause_html = (
                            f'<div style="color:#f85149;font-size:11px;margin-top:4px;">'
                            f'⏸️ PAUSADO {_rem_h:.0f}h ({_b_reason})</div>'
                        )
                    except Exception:
                        _pause_html = f'<div style="color:#f85149;font-size:11px;margin-top:4px;">⏸️ PAUSADO</div>'

                st.markdown(f"""
<div class="cg-card" style="text-align:center;">
  <div class="cg-card-title">{_bkt.get("name", _bkey)}</div>
  <div class="cg-card-value">${_b_cur:,.2f}</div>
  <div style="font-size:11px;margin-top:4px;">
    <span style="color:#8b949e;">DD: </span>
    <span style="color:{_dd_color};">{_b_dd:.1%}</span>
    <span style="color:#8b949e;"> / {_max_dd:.0%} max</span>
    &nbsp;|&nbsp;
    <span style="color:#8b949e;">Dia: </span>
    <span style="color:{_dl_color};">{_b_dpnl:+.0f} ({_b_dloss:.1%})</span>
  </div>
  {_pause_html}
</div>""", unsafe_allow_html=True)

        st.markdown(
            f'<div class="cg-card" style="padding:8px 16px; font-size:12px; margin-top:4px;">'
            f'<span style="color:#8b949e;">Capital Total: </span>'
            f'<span style="color:#e6edf3;">${portfolio.get("total_capital_usd", capital):,.2f}</span>'
            f' &nbsp;|&nbsp; '
            f'<span style="color:#8b949e;">Safety: </span>'
            f'<span style="color:#8b949e;">DD max {_max_dd:.0%} / Perda diária max {_max_dl:.0%} por bucket</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.caption(f"btc-trading-v1 | {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} | Auto-refresh: {'ON' if auto_refresh else 'OFF'}")


if __name__ == "__main__" or True:
    main()
