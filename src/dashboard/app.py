# app.py (versão com Admin integrado)
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
import plotly.express as px

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
    initial_sidebar_state="expanded",
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

# ===========================================================================
# NOVO: Menu lateral (antes de qualquer conteúdo do dashboard)
# ===========================================================================
st.sidebar.title("Navegação")
view = st.sidebar.radio("Selecione a visão", ["Painel Principal", "Admin"])

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
        is_available = data.get("is_available", False)
        balance_usd = 0.0
        for info in data.get("balance_infos", []):
            if info.get("currency", "").upper() == "USD":
                balance_usd = float(info.get("total_balance", 0))
                break
        return {"available": is_available, "balance_usd": balance_usd, "error": None}
    except Exception as e:
        return {"available": False, "balance_usd": 0.0, "error": str(e)}

def load_analyst_context() -> dict | None:
    """Read conf/analyst_context.json. Returns None if missing."""
    path = ROOT / "conf/analyst_context.json"
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None

def save_analyst_context(data: dict) -> None:
    """Persist analyst context to conf/analyst_context.json with current UTC timestamp."""
    data["updated_at"] = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d %H:%M UTC")
    path = ROOT / "conf/analyst_context.json"
    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

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
    fed_score    = float(portfolio.get("_fed_score", 0.0))
    g2 = 0.5 * crypto_score + 0.5 * fed_score

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
        oi_df      = load_parquet("data/02_intermediate/futures/oi_1h_clean.parquet")
        taker_df   = load_parquet("data/02_intermediate/futures/taker_1h_clean.parquet")
        funding_df = load_parquet("data/02_intermediate/futures/funding_1h_clean.parquet")
        lsa_df     = load_parquet("data/01_raw/futures/ls_account_1h.parquet")
        lsp_df     = load_parquet("data/01_raw/futures/ls_position_1h.parquet")
        fg_df      = load_parquet("data/01_raw/sentiment/fear_greed_daily.parquet")
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

    # Re-evaluate kill switches on fresh data
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

    # MA200 header snippet
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

    # Score bar
    score_pct = min(max((total_score / threshold) * 100, 0), 110) if threshold > 0 else 0
    if signal_computed == "BLOCK":
        bar_color = "#f85149"
    elif signal_computed == "ENTER":
        bar_color = "#3fb950"
    elif total_score > threshold * 0.6:
        bar_color = "#d29922"
    else:
        bar_color = "#f85149"

    # Score breakdown
    if regime_multiplier != 1.0 or global_conf_mult < 0.95:
        _parts = [f'<span>Σ clusters (bruto): {total_score_raw:+.3f}</span>']
        if regime_multiplier != 1.0:
            _parts.append(
                f'<span style="color:#d29922;">× Regime {regime} ({regime_multiplier}×)</span>'
            )
        if global_conf_mult < 0.95:
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

    # Kill switch banner
    if signal_computed == "BLOCK" and block_reason_computed and block_reason_computed != "BLOCK_BEAR_REGIME":
        ks_html = (
            f'<div style="margin-top:6px; padding:4px 8px; background:#3d1a1a; border-radius:4px; '
            f'font-size:12px; color:#f85149;">⛔ Kill switch: {block_reason_computed}</div>'
        )
    else:
        ks_html = ""

    # Threshold tooltip
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
    _r1d  = pct_24h / 100  # ret_1d = pct_24h / 100
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

    # Spike guard extra
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
        # F&G card
        fg_c = "neg" if (fg_val or 0) < 30 else ("warn" if fg_val < 60 else "pos")
        st.markdown(f"""
<div class="cg-card">
  <div class="cg-card-title">😱 Fear & Greed</div>
  <div class="cg-card-value {fg_c}">{fg_val:.0f}</div>
  <div class="cg-card-sub">{fg_cls}</div>
  <div class="cg-card-sub">z={zs.get('fg_z',0):.2f}</div>
</div>""", unsafe_allow_html=True)

    # =========================================================================
    # SECTION 9: FED SENTINEL
    # =========================================================================
    st.markdown("---")
    st.markdown("### 🏛️ Fed Sentinel")
    st.caption("Probabilidades estimadas (proxy DGS2, não FedWatch)")

    try:
        from src.features.fed_observatory import load_fed_data, estimate_rate_probability, get_scenario_analysis

        _fed_data = load_fed_data()
        _prob     = estimate_rate_probability(_fed_data)
        _analysis = get_scenario_analysis(_prob)

        # Probability cards
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

        # Sentinel status
        st.markdown("**📅 Próximo evento**")
        _fs_c1, _fs_c2, _fs_c3, _fs_c4 = st.columns(4)
        _fs_c1.metric("Evento", fomc.get("next_event", "N/A"))
        _fs_c2.metric("Em", f"{fomc.get('days_away', '—')} dias")
        _fs_c3.metric("Proximity adj", f"+{fomc.get('proximity_adj', 0):.1f}")
        _fs_c4.metric("Blackout", "Sim ⚠️" if fomc.get("in_blackout") else "Não")

        # Indicators
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

        # Scenarios
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

        # Member sentiment
        if _analysis.get("member_summary"):
            _ms = _analysis["member_summary"]
            st.markdown(
                f"**Membros Fed (últimos 30 dias):** "
                f"🔴 Hawkish: {_ms['hawkish']} | "
                f"🟢 Dovish: {_ms['dovish']} | "
                f"⚪ Neutro: {_ms['neutral']} "
                f"→ Tendência: **{_ms['trend']}**"
            )

        # Fed agenda
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

    # ── Final footer ───────────────────────────────────────────────────────
    st.caption(f"btc-trading-v1 | Limpo: ETH/SOL removidos | {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} | Auto-refresh: {'ON' if auto_refresh else 'OFF'}")

# ===========================================================================
# NOVO: Condicional para selecionar visão (Painel Principal ou Admin)
# ===========================================================================
if view == "Painel Principal":
    if __name__ == "__main__" or True:
        main()
elif view == "Admin":
    # =========================================================================
    # ADMIN PANEL
    # =========================================================================
    st.header("🔧 Painel de Administração")
    st.caption(f"Última atualização: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")

    # ── 1. STATUS DOS BOTS ────────────────────────────────────────────────────
    st.subheader("🤖 Status dos Bots — BTC")

    # Source of truth: portfolio_state.json (capital_manager.json has stale n_trades)
    _adm_port   = load_portfolio()
    _adm_cap    = _adm_port.get("capital_usd", 10000.0)
    _adm_ini    = 10000.0
    _adm_pnl    = _adm_cap - _adm_ini
    _adm_pnl_pct = (_adm_cap / _adm_ini - 1) * 100
    _adm_has_pos = _adm_port.get("has_position", False)
    _adm_ebot    = _adm_port.get("entry_bot")
    _adm_ep      = _adm_port.get("entry_price")
    _adm_sl      = _adm_port.get("stop_loss_price")
    _adm_tp      = _adm_port.get("take_profit_price")
    _adm_et      = _adm_port.get("entry_time")
    _adm_paused  = _adm_port.get("paused_until")
    _adm_csl     = _adm_port.get("consecutive_sl_count", 0)
    _adm_lsig    = _adm_port.get("last_signal", "—")

    _ac1, _ac2, _ac3 = st.columns(3)

    _adm_pnl_c = "pos" if _adm_pnl_pct >= 0 else "neg"
    _ac1.metric(
        "💰 Capital Total BTC",
        f"${_adm_cap:,.2f}",
        delta=f"{_adm_pnl_pct:+.2f}% (${_adm_pnl:+.2f})",
    )

    _b1_pos    = _adm_has_pos and _adm_ebot == "bot1"
    _b1_status = ("🔒 Em posição" if _b1_pos
                  else "⏸ Pausado" if _adm_paused
                  else "🟢 Aguardando sinal")
    _ac2.metric(
        "🤖 Bot 1 — Reversal",
        _b1_status,
        delta=f"SL consecutivos: {_adm_csl}",
    )

    _b2_pos    = _adm_has_pos and _adm_ebot == "bot2"
    _b2_status = ("🔒 Em posição" if _b2_pos
                  else "⏸ Pausado" if _adm_paused
                  else "🟢 Aguardando sinal")
    _ac3.metric(
        "🚀 Bot 2 — Momentum",
        _b2_status,
        delta=f"Último sinal: {_adm_lsig}",
    )

    if _adm_has_pos and _adm_ep:
        _adm_et_s = str(_adm_et)[:16] if _adm_et else "—"
        _adm_sl_s = f"${_adm_sl:,.0f}" if _adm_sl else "—"
        _adm_tp_s = f"${_adm_tp:,.0f}" if _adm_tp else "—"
        st.info(
            f"📌 Posição aberta — {_adm_ebot} | "
            f"Entrada: ${_adm_ep:,.0f} | "
            f"SL: {_adm_sl_s} | "
            f"TP: {_adm_tp_s} | "
            f"Desde: {_adm_et_s} UTC"
        )

    st.caption(
        "⚠️ capital_management.enabled=false — bots compartilham pool único. "
        "Buckets separados pendentes de ativação."
    )

    # ── 2. SEMÁFORO DE FONTES ─────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("🚦 Status das Fontes de Dados")

    _FONTES = [
        ("BTC Spot 1h",      "data/01_raw/spot/btc_1h.parquet",                           2),
        ("Futures OI",       "data/01_raw/futures/oi_4h.parquet",                         6),
        ("Futures Taker",    "data/01_raw/futures/taker_4h.parquet",                      6),
        ("Futures Funding",  "data/01_raw/futures/funding_4h.parquet",                    6),
        ("News Crypto",      "data/01_raw/news/crypto_news.parquet",                      2),
        ("News Macro",       "data/01_raw/news/macro_news.parquet",                       2),
        ("News Scores G2a",  "data/02_features/news_scores.parquet",                      2),
        ("News Regime G2b",  "data/02_features/news_regime.parquet",                      5),
        ("Gate Zscores",     "data/02_features/gate_zscores.parquet",                     2),
        ("Score History",    "data/04_scoring/score_history.parquet",                     2),
        ("ETF Flows",        "data/01_raw/coinglass/etf_flows_daily.parquet",            26),
        ("FRED Macro",       "data/02_intermediate/macro/fred_daily_clean.parquet",      26),
    ]

    _fcols = st.columns(4)
    for _fi, (_fname, _frel, _fmax) in enumerate(_FONTES):
        _fpath = ROOT / _frel
        if _fpath.exists():
            _mtime = datetime.fromtimestamp(_fpath.stat().st_mtime, tz=timezone.utc)
            _age_h = (datetime.now(tz=timezone.utc) - _mtime).total_seconds() / 3600
            if _age_h < _fmax * 0.5:
                _ficon, _fdelta_c = "🟢", "normal"
            elif _age_h < _fmax:
                _ficon, _fdelta_c = "🟡", "off"
            else:
                _ficon, _fdelta_c = "🔴", "inverse"
            _fval, _fdelta = f"{_age_h:.1f}h", f"limite {_fmax}h"
        else:
            _ficon, _fdelta_c = "🔴", "inverse"
            _fval, _fdelta = "ausente", "não encontrado"
        _fcols[_fi % 4].metric(
            label=f"{_ficon} {_fname}",
            value=_fval,
            delta=_fdelta,
            delta_color=_fdelta_c,
        )

    # ── 3. GATE SCORES — ÚLTIMOS 7 DIAS ──────────────────────────────────────
    st.markdown("---")
    st.subheader("📊 Gate Scores — Últimos 7 dias")

    _sh_path = ROOT / "data/04_scoring/score_history.parquet"
    if _sh_path.exists():
        _sh = pd.read_parquet(_sh_path)
        if "timestamp" in _sh.columns:
            _sh["timestamp"] = pd.to_datetime(_sh["timestamp"], utc=True)
            _sh = _sh.set_index("timestamp").sort_index()
        _cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=7)
        _sh7 = _sh[_sh.index >= _cutoff]

        if not _sh7.empty:
            _cols_show = [c for c in ["total_score", "score_raw", "threshold",
                                       "signal", "block_reason", "regime_multiplier"]
                          if c in _sh7.columns]
            st.dataframe(
                _sh7[_cols_show].tail(10).sort_index(ascending=False),
                use_container_width=True,
            )
            if "total_score" in _sh7.columns:
                _fig_sh = go.Figure()
                _fig_sh.add_trace(go.Scatter(
                    x=_sh7.index, y=_sh7["total_score"],
                    name="Score Total", line=dict(color=GREEN, width=2),
                ))
                if "score_raw" in _sh7.columns:
                    _fig_sh.add_trace(go.Scatter(
                        x=_sh7.index, y=_sh7["score_raw"],
                        name="Score Raw", line=dict(color=BLUE, width=1, dash="dot"),
                    ))
                if "threshold" in _sh7.columns:
                    _fig_sh.add_trace(go.Scatter(
                        x=_sh7.index, y=_sh7["threshold"],
                        name="Threshold", line=dict(color=RED, width=1, dash="dash"),
                    ))
                _fig_sh.update_layout(
                    height=280, title="Score Total vs Threshold (7d)",
                    **{k: v for k, v in PLOTLY.items() if k != "margin"},
                    margin=dict(l=40, r=20, t=40, b=30),
                )
                st.plotly_chart(_fig_sh, use_container_width=True)
        else:
            st.info("Nenhum score nos últimos 7 dias.")
    else:
        st.warning(f"score_history.parquet não encontrado.")

    # ── 4. PARÂMETROS ATIVOS ──────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("⚙️ Parâmetros Ativos")

    _ap = load_params()
    _tab1, _tab2 = st.tabs(["Bot 1 — Reversal Gate", "Bot 2 — Momentum"])

    with _tab1:
        _ex  = _ap.get("execution", {})
        _rf  = _ap.get("reversal_filter", {})
        _rfc = _rf.get("cooldown", {})
        _c1, _c2 = st.columns(2)
        with _c1:
            st.markdown("**Stops Dinâmicos (ATR)**")
            st.table(pd.DataFrame({
                "Parâmetro": ["Dynamic stops", "Min SL%", "Max SL%",
                              "Min TP%", "Max TP%",
                              "ATR mult SL", "ATR mult TP", "ATR mult Trail"],
                "Valor": [
                    "✅" if _ex.get("use_dynamic_stops") else "❌",
                    f"{_ex.get('min_stop_loss_pct', 0)*100:.1f}%",
                    f"{_ex.get('max_stop_loss_pct', 0)*100:.1f}%",
                    f"{_ex.get('min_take_profit_pct', 0)*100:.1f}%",
                    f"{_ex.get('max_take_profit_pct', 0)*100:.1f}%",
                    _ex.get('atr_multiplier_sl', '-'),
                    _ex.get('atr_multiplier_tp', '-'),
                    _ex.get('atr_multiplier_trail', '-'),
                ]
            }))
        with _c2:
            st.markdown("**Reversal Filter + Cooldown**")
            st.table(pd.DataFrame({
                "Parâmetro": ["RSI max", "Ret 1d min", "RSI extremo override",
                              "Cooldown horas SL", "Max SLs consecutivos",
                              "Pausa após max SLs (h)"],
                "Valor": [
                    _rf.get("rsi_max", "-"),
                    f"{(_rf.get('ret_1d_min') or 0)*100:.1f}%",
                    _rf.get("rsi_extreme_override", "-"),
                    _rfc.get("hours_after_sl", "-"),
                    _rfc.get("max_consecutive_sl", "-"),
                    _rfc.get("consecutive_sl_pause_hours", "-"),
                ]
            }))

    with _tab2:
        _mf  = _ap.get("momentum_filter", {})
        _mfc = _mf.get("cooldown", {})
        _sg  = _mf.get("spike_guard", {})
        _c1, _c2 = st.columns(2)
        with _c1:
            st.markdown("**Filtros de Entrada**")
            st.table(pd.DataFrame({
                "Parâmetro": ["Enabled", "Stablecoin Z min", "RSI min", "RSI max",
                              "BB% max", "Dist high 7d min", "Ret 1d min",
                              "Require MA21", "Kill switches gate",
                              "News score min"],
                "Valor": [
                    "✅" if _mf.get("enabled") else "❌",
                    _mf.get("stablecoin_z_min", "-"),
                    _mf.get("rsi_min", "-"),
                    _mf.get("rsi_max", "-"),
                    f"{(_mf.get('bb_pct_max') or 0)*100:.0f}%",
                    f"{(_mf.get('dist_high_7d_min') or 0)*100:.0f}%",
                    f"{(_mf.get('ret_1d_min') or 0)*100:.1f}%",
                    "✅" if _mf.get("require_above_ma21") else "❌",
                    "✅" if _mf.get("respect_gate_kill_switches") else "❌",
                    _mf.get("news_score_min", "-"),
                ]
            }))
        with _c2:
            st.markdown("**Stops + Spike Guard**")
            st.table(pd.DataFrame({
                "Parâmetro": ["SL%", "TP%", "Trailing%", "Max hold horas",
                              "Spike guard", "Spike ret max", "Spike RSI max",
                              "Cooldown horas SL", "Max SLs consecutivos"],
                "Valor": [
                    f"{(_mf.get('stop_loss_pct') or 0)*100:.1f}%",
                    f"{(_mf.get('take_profit_pct') or 0)*100:.1f}%",
                    f"{(_mf.get('trailing_stop_pct') or 0)*100:.1f}%",
                    _mf.get("max_hold_hours", "-"),
                    "✅" if _sg.get("enabled") else "❌",
                    f"{(_sg.get('spike_ret_max') or 0)*100:.0f}%",
                    _sg.get("spike_rsi_max", "-"),
                    _mfc.get("hours_after_sl", "-"),
                    _mfc.get("max_consecutive_sl", "-"),
                ]
            }))

    # ── 5. CONTEXTO DO ANALISTA ───────────────────────────────────────────────
    st.markdown("---")
    st.subheader("📝 Contexto do Analista")
    st.caption("Percepção qualitativa de mercado — alimenta o G2b (DeepSeek-R1) no próximo ciclo de 4h.")

    _ctx_path = ROOT / "conf" / "analyst_context.json"
    _ctx = load_analyst_context() or {}

    if _ctx:
        _ctx_updated = _ctx.get("updated_at", "-")
        try:
            _ctx_dt = datetime.fromisoformat(_ctx_updated.replace(" UTC", "+00:00"))
            if _ctx_dt.tzinfo is None:
                _ctx_dt = _ctx_dt.replace(tzinfo=timezone.utc)
            _ctx_age_h  = (datetime.now(tz=timezone.utc) - _ctx_dt).total_seconds() / 3600
            _ctx_age_s  = f"{_ctx_age_h:.1f}h atrás"
            _ctx_expire = max(0.0, 12.0 - _ctx_age_h)
            _ctx_status = "🟢" if _ctx_age_h < 8 else ("🟡" if _ctx_age_h < 12 else "🔴 EXPIRADO")
        except Exception:
            _ctx_age_s, _ctx_expire, _ctx_status = "-", 0.0, "⚪"
        st.caption(f"{_ctx_status} Atualizado: {_ctx_updated} ({_ctx_age_s}) | Expira em: {_ctx_expire:.1f}h")
    else:
        st.caption("⚪ Nenhum contexto salvo ainda.")

    _bias_opt = ["BEAR", "SIDEWAYS", "BULL"]
    _conf_opt = ["high", "medium", "low"]
    _cc1, _cc2, _cc3 = st.columns(3)
    _ctx_bias    = _ctx.get("bias", "SIDEWAYS")
    _ctx_conf    = _ctx.get("confidence", "medium")
    _ctx_bias_i  = _bias_opt.index(_ctx_bias) if _ctx_bias in _bias_opt else 1
    _ctx_conf_i  = _conf_opt.index(_ctx_conf) if _ctx_conf in _conf_opt else 1
    _new_bias    = _cc1.selectbox("Viés", _bias_opt, index=_ctx_bias_i)
    _new_conf    = _cc2.selectbox("Confiança", _conf_opt, index=_ctx_conf_i)
    _new_horizon = _cc3.text_input("Horizonte", value=_ctx.get("horizon", "24-48h"))

    _new_context = st.text_area(
        "Percepção de mercado",
        value=_ctx.get("context", ""),
        height=120,
        placeholder="Ex: Trump otimista sobre acordo Iran mas negociações falharam no Paquistão. Hormuz ainda bloqueado...",
    )
    _new_tags_str = st.text_input(
        "Tags (separadas por vírgula)",
        value=", ".join(_ctx.get("tags", [])),
    )

    if st.button("💾 Salvar Contexto"):
        _new_ctx = {
            "updated_at": datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            "author":     "Edmundo",
            "horizon":    _new_horizon,
            "context":    _new_context,
            "bias":       _new_bias,
            "confidence": _new_conf,
            "tags":       [t.strip() for t in _new_tags_str.split(",") if t.strip()],
        }
        save_analyst_context(_new_ctx)
        st.success("✅ Contexto salvo — G2b usará na próxima execução (4h)")
        st.rerun()

    # ── Último output G2b ─────────────────────────────────────────────────
    _nr_path = ROOT / "data" / "02_features" / "news_regime.parquet"
    if _nr_path.exists():
        _nr_df = load_parquet("data/02_features/news_regime.parquet")
        if not _nr_df.empty:
            _nr_last = _nr_df.iloc[-1]
            st.markdown("**Último output G2b:**")
            _nr_c1, _nr_c2, _nr_c3 = st.columns(3)
            _nr_hint  = str(_nr_last.get("regime_hint", "-"))
            _nr_color = "🟢" if _nr_hint == "BULL" else ("🔴" if _nr_hint == "BEAR" else "🟡")
            _nr_c1.metric("Regime G2b", f"{_nr_color} {_nr_hint}")
            _nr_c2.metric("Confiança", f"{float(_nr_last.get('confidence', 0)):.2f}")
            _nr_c3.metric("Analyst bias usado", str(_nr_last.get("analyst_bias") or "nenhum"))
            _nr_reasoning = str(_nr_last.get("reasoning", "-"))
            st.caption(f"Reasoning: {_nr_reasoning[:200]}")

    # ── Footer ────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.caption(f"Admin | {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")