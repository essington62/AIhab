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

st.set_page_config(
    page_title="btc-trading-v1",
    page_icon="₿",
    layout="wide",
    initial_sidebar_state="collapsed",
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
  .signal-enter { color: #3fb950; font-weight: 700; }
  .signal-hold  { color: #d29922; font-weight: 700; }
  .signal-block { color: #f85149; font-weight: 700; }

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
    g9  = _tanh_score(g("taker_z"),      *gp.get("g9_taker",    [ 0.143, 0.5, 0.5]))
    g10 = _tanh_score(g("funding_z"),    *gp.get("g10_funding",  [-0.064, 0.4, 0.5]))

    # G5 stable, G7 ETF, G6 bubble, G8 F&G
    g5  = _tanh_score(g("stablecoin_z"), *gp.get("g5_stable",   [ 0.326, 0.6, 1.0]))
    g7  = _tanh_score(g("etf_z"),        *gp.get("g7_etf",       [ 0.263, 0.6, 1.0]))
    g6  = _tanh_score(g("bubble_z"),     *gp.get("g6_bubble",    [-0.345, 0.7, 1.0]))
    g8  = _tanh_score(g("fg_z"),         *gp.get("g8_fg",        [-0.211, 0.7, 0.8]))

    # G2 news
    ns = load_parquet("data/02_features/news_scores.parquet")
    crypto_score = _latest(ns, "crypto_score", 0.0) or 0.0
    macro_score  = _latest(ns, "macro_score",  0.0) or 0.0
    fed_score    = portfolio.get("_fed_score", 0.0)
    g2 = 0.5 * float(crypto_score) + 0.5 * float(fed_score)

    clusters = {
        "technical":   float(np.clip(g1,             -2.0,  3.5)),
        "news":        float(np.clip(g2,             -1.5,  1.0)),
        "macro":       float(np.clip(g3,             -1.5,  1.0)),
        "positioning": float(np.clip(g4 + g10,       -2.0,  1.5)),
        "liquidity":   float(np.clip(g5 + g7,        -1.5,  1.5)),
        "sentiment":   float(np.clip(g6 + g8 + g9,   -1.5,  1.5)),
    }
    details = {
        "technical":   {"bb_pct": bb_pct, "rsi_14": rsi_14, "bb_s": bb_s, "rsi_s": rsi_s, "g1_raw": g1},
        "news":        {"crypto_score": crypto_score, "macro_score": macro_score, "fed_score": fed_score},
        "macro":       {"dgs10_z": g("dgs10_z"), "curve_z": g("curve_z"), "rrp_z": g("rrp_z"), "dgs2_z": g("dgs2_z")},
        "positioning": {"oi_z": g("oi_z"), "funding_z": g("funding_z"), "g4": g4, "g10": g10},
        "liquidity":   {"stablecoin_z": g("stablecoin_z"), "etf_z": g("etf_z"), "g5": g5, "g7": g7},
        "sentiment":   {"bubble_z": g("bubble_z"), "fg_z": g("fg_z"), "taker_z": g("taker_z"), "g6": g6, "g8": g8, "g9": g9},
    }
    return clusters, details

# ---------------------------------------------------------------------------
# Interpretation text
# ---------------------------------------------------------------------------

def interpret_cluster(name: str, score: float, det: dict, zs: dict) -> str:
    def z(k): return zs.get(k) or 0.0

    if name == "technical":
        bb = det.get("bb_pct", 0.5)
        rsi = det.get("rsi_14", 50)
        if bb > 0.80:
            return f"⛔ BB={bb:.2f} — Kill switch ativado. Preço no topo da banda, zona de sobrecompra extrema"
        if bb < 0.20:
            return f"🟢 BB={bb:.2f} — Preço na base da banda. Win rate histórico 88% em 3d"
        if bb < 0.30:
            return f"🟢 BB={bb:.2f} — Próximo da banda inferior. Win rate histórico 77%"
        if rsi < 35:
            return f"RSI={rsi:.0f} — Sobrevenda. Mercado stretched pra baixo"
        if rsi > 60:
            return f"RSI={rsi:.0f} — Momentum comprador elevado, monitorar reversão"
        return f"BB={bb:.2f} RSI={rsi:.0f} — Neutro, preço no meio da banda sem sinal direcional"

    if name == "positioning":
        oi_z = det.get("oi_z", 0)
        fund_z = det.get("funding_z", 0)
        if oi_z > 2.0:
            return f"⚠️ Alavancagem extrema (OI {oi_z:.1f}σ). Alto risco de liquidação em cascata se preço ceder"
        if oi_z > 1.0:
            return f"OI {oi_z:.1f}σ acima da média — mercado alavancado, risco elevado de long squeeze"
        if oi_z < -1.0:
            return f"OI {oi_z:.1f}σ — mercado desalavancado. Base limpa, favorável pra subida"
        if fund_z > 1.5:
            return f"Funding {fund_z:.1f}σ — longs pagando muito. Mercado crowded long, risco de squeeze"
        if fund_z < -1.5:
            return f"Funding negativo — shorts pagando. Potencial short squeeze"
        return f"OI z={oi_z:.2f} | Funding z={fund_z:.2f} — alavancagem dentro da normalidade"

    if name == "macro":
        dgs10_z = z("dgs10_z")
        curve_z = z("curve_z")
        if dgs10_z > 1.0 and curve_z < -0.5:
            return f"Juros altos (DGS10 {dgs10_z:.1f}σ) com curva invertendo ({curve_z:.1f}σ) — ambiente restritivo pra risco"
        if dgs10_z > 1.0:
            return f"DGS10 {dgs10_z:.1f}σ acima da média — juros elevados pressionam ativos de risco"
        if dgs10_z < -1.0:
            return f"Juros caindo ({dgs10_z:.1f}σ) — expectativa de afrouxamento monetário favorece BTC"
        if curve_z < -1.0:
            return f"Curva de juros invertendo ({curve_z:.1f}σ) — sinal clássico de stress econômico"
        return f"DGS10 z={dgs10_z:.2f} | Curve z={curve_z:.2f} — macro dentro da normalidade"

    if name == "liquidity":
        stab_z = z("stablecoin_z")
        etf_z  = z("etf_z")
        if stab_z > 1.0 and etf_z > 0.5:
            return f"🟢 Liquidez entrando — stablecoins crescendo ({stab_z:.1f}σ) e ETFs com inflows ({etf_z:.1f}σ)"
        if stab_z < -1.0:
            return f"Stablecoins contraindo ({stab_z:.1f}σ) — capital saindo do ecossistema crypto. Menos combustível"
        if etf_z > 1.0:
            return f"ETF inflows forte ({etf_z:.1f}σ) — institucional comprando"
        if etf_z < -1.0:
            return f"ETF outflows ({etf_z:.1f}σ) — institucional reduzindo exposição"
        return f"Stablecoin z={stab_z:.2f} | ETF z={etf_z:.2f} — liquidez dentro da normalidade"

    if name == "sentiment":
        taker_z  = z("taker_z")
        bubble_z = z("bubble_z")
        fg_z     = z("fg_z")
        if taker_z < -2.0:
            return f"⚠️ Pressão vendedora extrema nos futuros (taker {taker_z:.1f}σ) — retail vendendo agressivamente"
        if taker_z > 2.0:
            return f"Compra agressiva nos futuros (taker {taker_z:.1f}σ) — momentum comprador forte"
        if bubble_z > 1.5:
            return f"Bubble index overextended ({bubble_z:.1f}σ) — mercado sobreaquecido, risco de correção"
        if fg_z < -1.0:
            return f"Fear & Greed em medo ({fg_z:.1f}σ) — contrarian bullish. Historicamente bom pra comprar"
        if fg_z > 1.5:
            return f"Greed elevado ({fg_z:.1f}σ) — mercado complacente, cautela"
        return f"Taker z={taker_z:.2f} | Bubble z={bubble_z:.2f} | F&G z={fg_z:.2f} — sentimento neutro"

    if name == "news":
        cs = det.get("crypto_score", 0)
        ms = det.get("macro_score", 0)
        fs = det.get("fed_score", 0)
        if cs > 3.0:
            return f"🟢 Notícias crypto muito bullish (score +{cs:.1f}). Catalisador positivo no curto prazo"
        if cs < -3.0:
            return f"🔴 Notícias crypto bearish (score {cs:.1f}). Sentimento negativo recente"
        if fs < -1.0:
            return f"Fed hawkish (score {fs:.2f}) — linguagem restritiva nos discursos recentes"
        if ms > 1.0:
            return f"Macro positiva (score +{ms:.1f}) — ambiente macro favorável"
        return f"Crypto {cs:+.1f} | Macro {ms:+.1f} | Fed {fs:+.2f} — notícias mistas sem viés claro"

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

def call_deepseek_analyst(context: dict) -> str:
    try:
        api_key = get_credential("deepseek_api_key")
    except Exception:
        return "DeepSeek API key não configurado."

    prompt = f"""Você é um analista quant sênior especializado em Bitcoin e macro.
Analise o estado atual do mercado BTC e forneça uma análise concisa e acionável.

=== DADOS ATUAIS ===
Preço BTC: ${context.get('price', 0):,.0f}
Variação 24h: {context.get('pct_24h', 0):+.2f}%
Regime R5C: {context.get('regime', 'N/A')}
Sinal Gate v2: {context.get('signal', 'N/A')}
Score: {context.get('score', 0):.3f} / Threshold: {context.get('threshold', 3.5):.1f}
Capital: ${context.get('capital', 10000):,.0f} | Posição: {context.get('has_position', False)}

Clusters:
  Technical:   {context.get('c_technical', 0):.3f}
  Positioning: {context.get('c_positioning', 0):.3f}
  Macro:       {context.get('c_macro', 0):.3f}
  Liquidity:   {context.get('c_liquidity', 0):.3f}
  Sentiment:   {context.get('c_sentiment', 0):.3f}
  News:        {context.get('c_news', 0):.3f}

Z-scores principais:
  OI: {context.get('oi_z', 0):.2f} | Taker: {context.get('taker_z', 0):.2f} | Funding: {context.get('funding_z', 0):.2f}
  DGS10: {context.get('dgs10_z', 0):.2f} | Curve: {context.get('curve_z', 0):.2f}
  Stablecoin: {context.get('stablecoin_z', 0):.2f} | ETF: {context.get('etf_z', 0):.2f}
  F&G: {context.get('fg_z', 0):.2f} | Bubble: {context.get('bubble_z', 0):.2f}

Baleias: L/S={context.get('ls_account', 1):.3f} | {context.get('whale_signal', 'N/A')}
Fed: {context.get('fed_event', 'N/A')} em {context.get('fed_days', 'N/A')} dias | Proximity adj: +{context.get('fed_adj', 0):.1f}

=== INSTRUÇÃO ===
Em 3-5 parágrafos curtos (máximo 300 palavras total):
1. Avaliação do regime e contexto macro atual
2. Principal risco e principal oportunidade
3. Recomendação: manter HOLD ou argumento pra mudar
4. Um gatilho específico que mudaria o sinal pra ENTER ou BLOCK

Seja específico com números. Não repita dados — interprete-os."""

    try:
        resp = requests.post(
            "https://api.deepseek.com/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": "deepseek-chat",
                "max_tokens": 600,
                "temperature": 0.3,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"Erro ao chamar DeepSeek: {e}"

# ===========================================================================
# MAIN APP
# ===========================================================================

def main():
    # ── Load all data ─────────────────────────────────────────────────────
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
    capital    = portfolio.get("capital_usd", 10000.0)

    # 24h price change
    if len(spot_df) > 24:
        pct_24h = (price - spot_df["close"].iloc[-25]) / spot_df["close"].iloc[-25] * 100
    else:
        pct_24h = 0.0

    # Z-scores
    zs = {}
    if not zs_df.empty:
        last_zs = zs_df.iloc[-1].to_dict()
        zs = {k: v for k, v in last_zs.items() if k != "timestamp" and v is not None}

    fomc = get_fomc_proximity(cal)
    clusters, cdet = compute_clusters(zs, bb_pct or 0.5, rsi_14 or 50.0, portfolio)
    total_score = sum(clusters.values())

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
            time.sleep(1)
            st.rerun()
        st.markdown("---")
        st.markdown(f"**Last load:** {datetime.now(timezone.utc).strftime('%H:%M:%S UTC')}")

    # =========================================================================
    # SECTION 1: HEADER
    # =========================================================================
    sig_class = {"ENTER": "signal-enter", "HOLD": "signal-hold", "BLOCK": "signal-block"}.get(signal, "neut")
    price_c   = "pos" if pct_24h >= 0 else "neg"
    regime_c  = {"Bull": "pos", "Bear": "neg", "Sideways": "warn"}.get(regime, "neut")
    fed_note  = f"Fed: {fomc['next_event']} em {fomc['days_away']}d" if fomc["next_event"] else "Fed: sem eventos próximos"
    cycle_info = get_last_cycle_info()
    cycle_age  = f"{cycle_info['age_min']:.0f}min atrás" if cycle_info else "N/A"
    cycle_ok   = "✅" if cycle_info and cycle_info.get("age_min", 99) < 75 else "⚠️"

    st.markdown(f"""
<div class="header-bar">
  <span>₿ <span class="header-value">${price:,.0f}</span></span>
  <span class="{price_c}">{pct_24h:+.2f}%</span>
  <span class="header-item">OI: <span class="header-value">${oi_val/1e9:.1f}B</span></span>
  <span class="header-item">F&G: <span class="header-value">{fg_val:.0f}</span> {fg_cls}</span>
  <span class="header-item">R5C: <span class="{regime_c} header-value">{regime}</span></span>
  <span class="header-item">Gate: <span class="{sig_class}">{signal}</span></span>
  <span class="header-item">Score: <span class="header-value">{score:.3f}</span> / {threshold:.1f}</span>
  <span class="header-item">Capital: <span class="header-value">${capital:,.0f}</span></span>
  <span class="{'warn' if fomc['days_away'] < 5 else 'header-item'}">{fed_note}</span>
  <span class="header-item">Cron: {cycle_age} {cycle_ok}</span>
</div>
""", unsafe_allow_html=True)

    # =========================================================================
    # SECTION 2: GATE SCORING
    # =========================================================================
    st.markdown("### Gate Scoring v2")

    # Score bar
    score_pct = min(max((total_score / threshold) * 100, 0), 110) if threshold > 0 else 0
    bar_color = "#3fb950" if total_score >= threshold else ("#d29922" if total_score > threshold * 0.6 else "#f85149")
    st.markdown(f"""
<div class="score-bar-wrap">
  <div style="display:flex; justify-content:space-between; margin-bottom:6px;">
    <span style="font-size:13px; color:#8b949e;">Score Total</span>
    <span style="font-size:18px; font-weight:700; color:{bar_color};">{total_score:+.3f}</span>
    <span style="font-size:13px; color:#8b949e;">Threshold: {threshold:.1f}</span>
  </div>
  <div style="background:#21262d; border-radius:4px; height:8px; width:100%;">
    <div style="background:{bar_color}; width:{min(score_pct,100):.1f}%; height:8px; border-radius:4px;"></div>
  </div>
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
            sub = f"BB={d.get('bb_pct',0):.3f} | RSI={d.get('rsi_14',0):.1f}"
        elif name == "positioning":
            sub = f"OI z={d.get('oi_z',0):.2f} | Fund z={d.get('funding_z',0):.2f}"
        elif name == "macro":
            sub = f"DGS10 z={d.get('dgs10_z',0):.2f} | Curve z={d.get('curve_z',0):.2f}"
        elif name == "liquidity":
            sub = f"Stable z={d.get('stablecoin_z',0):.2f} | ETF z={d.get('etf_z',0):.2f}"
        elif name == "sentiment":
            sub = f"Bubble z={d.get('bubble_z',0):.2f} | F&G z={d.get('fg_z',0):.2f} | Taker z={d.get('taker_z',0):.2f}"
        else:
            sub = f"Crypto {d.get('crypto_score',0):+.2f} | Macro {d.get('macro_score',0):+.2f} | Fed {d.get('fed_score',0):+.2f}"

        with cols_c[i % 3]:
            st.markdown(f"""
<div class="cg-card">
  <div class="cg-card-title">{CLUSTER_ICONS[name]} {name.upper()}</div>
  <div class="cg-card-value {sc_c}">{sc:+.3f}</div>
  <div class="cg-card-sub">{sub}</div>
  <div class="cg-card-interp">{interp}</div>
</div>""", unsafe_allow_html=True)

    # =========================================================================
    # SECTION 3: WHALE TRACKING
    # =========================================================================
    st.markdown("---")
    st.markdown("### 🐋 Whale Tracking")
    col_w1, col_w2 = st.columns([1, 2])

    with col_w1:
        ws_c = "neg" if "🔴" in ws_label else ("pos" if "🟢" in ws_label else "neut")
        lsa_delta = ""
        if len(lsa_df) > 24:
            lsa_prev = lsa_df["longShortRatio"].iloc[-25]
            lsa_d = (ls_account_val - lsa_prev)
            lsa_delta = f"({lsa_d:+.3f} vs 24h)"
        lsp_c = "pos" if (ls_position_val or 1) > 1.0 else "neg"
        st.markdown(f"""
<div class="cg-card">
  <div class="cg-card-title">Top Accounts L/S Ratio</div>
  <div class="cg-card-value {ws_c}">{ls_account_val:.3f} {lsa_delta}</div>
  <div class="cg-card-sub">Top Positions: <span class="{lsp_c}">{ls_position_val:.3f}</span></div>
  <div class="cg-card-interp"><strong>{ws_label}</strong><br>{ws_text}</div>
</div>""", unsafe_allow_html=True)

    with col_w2:
        if not lsa_df.empty and len(lsa_df) > 48:
            lsa_7d = lsa_df.tail(168)
            spot_7d = spot_df.tail(168) if not spot_df.empty else pd.DataFrame()
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(go.Scatter(
                x=lsa_7d["timestamp"], y=lsa_7d["longShortRatio"],
                name="L/S Accounts", line=dict(color=BLUE, width=1.5)
            ), secondary_y=False)
            if not spot_7d.empty:
                fig.add_trace(go.Scatter(
                    x=spot_7d["timestamp"], y=spot_7d["close"],
                    name="BTC Price", line=dict(color=AMBER, width=1.5, dash="dot")
                ), secondary_y=True)
            fig.add_hline(y=1.0, line_dash="dash", line_color=GREY, opacity=0.4)
            fig.update_layout(**PLOTLY, height=200, showlegend=True,
                              legend=dict(orientation="h", y=1.1))
            fig.update_yaxes(title_text="L/S Ratio", secondary_y=False)
            fig.update_yaxes(title_text="BTC $", secondary_y=True)
            st.plotly_chart(fig, use_container_width=True)

    # =========================================================================
    # SECTION 4: DERIVATIVES
    # =========================================================================
    st.markdown("---")
    st.markdown("### 📈 Derivativos")

    # 24h metrics
    oi_chg = 0.0
    if len(oi_df) > 24:
        oi_prev = oi_df["open_interest"].iloc[-25]
        oi_chg = (oi_val - oi_prev) / oi_prev * 100 if oi_prev else 0

    liq_24h = 0.0; long_liq_24h = 0.0; short_liq_24h = 0.0
    if not liq_df.empty:
        cutoff_24h = pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=24)
        liq_24h_df = liq_df[liq_df["timestamp"] >= cutoff_24h]
        long_liq_24h  = liq_24h_df["long_liq_usd"].sum() / 1e6
        short_liq_24h = liq_24h_df["short_liq_usd"].sum() / 1e6
        liq_24h = long_liq_24h + short_liq_24h

    fund_z = zs.get("funding_z", 0)
    taker_z_val = zs.get("taker_z", 0)

    # Bid/Ask ratio from orderbook
    ob_ratio     = _latest(ob_df,     "bid_ask_ratio", None)
    ob_agg_ratio = _latest(ob_agg_df, "bid_ask_ratio", None)
    ob_bids      = _latest(ob_df,     "bids_usd", 0) or 0
    ob_asks      = _latest(ob_df,     "asks_usd", 0) or 0
    ob_ratio_c   = "pos" if (ob_ratio or 1) > 1.05 else ("neg" if (ob_ratio or 1) < 0.95 else "neut")
    ob_interp = (
        "🟢 Bids dominam — pressão compradora no orderbook" if (ob_ratio or 1) > 1.1 else
        "🔴 Asks dominam — pressão vendedora no orderbook" if (ob_ratio or 1) < 0.9 else
        "⚪ Orderbook equilibrado"
    )

    d_cols = st.columns(5)
    metrics_d = [
        ("OI (Agregado)",   f"${oi_val/1e9:.1f}B",      f"{oi_chg:+.2f}%",                      oi_chg >= 0),
        ("Funding Rate",    f"{(fund_val or 0)*100:.4f}%", f"z={fund_z:.2f}",                    fund_z < 0),
        ("Taker Ratio",     f"{taker_val:.3f}",          f"z={taker_z_val:.2f}",                 taker_z_val > 0),
        ("Liquidações 24h", f"${liq_24h:.1f}M",          f"L:${long_liq_24h:.0f}M S:${short_liq_24h:.0f}M", None),
        ("Bid/Ask Ratio",   f"{ob_ratio:.3f}" if ob_ratio else "—",
                            f"Bids:${ob_bids/1e6:.0f}M Asks:${ob_asks/1e6:.0f}M", ob_ratio and ob_ratio > 1),
    ]
    for col, (title, val, sub, good) in zip(d_cols, metrics_d):
        sub_c = "" if good is None else ("pos" if good else "neg")
        with col:
            st.markdown(f"""
<div class="cg-card" style="text-align:center;">
  <div class="cg-card-title">{title}</div>
  <div class="cg-card-value">{val}</div>
  <div class="cg-card-sub {sub_c}">{sub}</div>
</div>""", unsafe_allow_html=True)

    # Bid/Ask interpretation bar
    if ob_ratio is not None:
        st.markdown(f"""
<div class="cg-card" style="padding:8px 16px; margin-bottom:8px;">
  <span style="font-size:12px; color:#8b949e;">ORDERBOOK (Binance BTCUSDT) </span>
  <span class="{ob_ratio_c}"> {ob_interp}</span>
  {'<span style="color:#8b949e; margin-left:16px;">Agg: <b>'+f"{ob_agg_ratio:.3f}"+'</b></span>' if ob_agg_ratio else ''}
</div>""", unsafe_allow_html=True)

    # Charts — expandable
    with st.expander("📊 Gráficos Derivativos", expanded=False):
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["OI vs Preço", "Funding Rate", "Liquidações", "Taker Ratio", "Bid/Ask Ratio"])

        with tab1:
            if not oi_df.empty and len(oi_df) > 24:
                oi_30d = oi_df.tail(720)
                sp_30d = spot_df.tail(720)
                fig = make_subplots(specs=[[{"secondary_y": True}]])
                fig.add_trace(go.Scatter(x=oi_30d["timestamp"], y=oi_30d["open_interest"]/1e9,
                    name="OI Agregado ($B)", line=dict(color=BLUE, width=1.5)), secondary_y=False)
                if not sp_30d.empty:
                    fig.add_trace(go.Scatter(x=sp_30d["timestamp"], y=sp_30d["close"],
                        name="BTC", line=dict(color=AMBER, width=1.5, dash="dot")), secondary_y=True)
                fig.update_layout(**PLOTLY, height=250, title="Open Interest vs BTC Price (30d)")
                st.plotly_chart(fig, use_container_width=True)

        with tab2:
            if not funding_df.empty and len(funding_df) > 10:
                fund_30d = funding_df.tail(180)
                fund_mean = fund_30d["funding_rate"].mean()
                fund_std  = fund_30d["funding_rate"].std()
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=fund_30d["timestamp"], y=fund_30d["funding_rate"]*100,
                    name="Funding %", line=dict(color=BLUE, width=1.5)))
                fig.add_hline(y=(fund_mean+fund_std)*100, line_dash="dash", line_color=AMBER, opacity=0.6, annotation_text="+1σ")
                fig.add_hline(y=(fund_mean-fund_std)*100, line_dash="dash", line_color=AMBER, opacity=0.6, annotation_text="-1σ")
                fig.add_hline(y=(fund_mean+2*fund_std)*100, line_dash="dot", line_color=RED, opacity=0.5, annotation_text="+2σ")
                fig.add_hline(y=0, line_color=GREY, opacity=0.4)
                fig.update_layout(**PLOTLY, height=250, title="Funding Rate com bandas ±1σ / ±2σ")
                st.plotly_chart(fig, use_container_width=True)

        with tab3:
            if not liq_df.empty and len(liq_df) > 10:
                liq_30d = liq_df.tail(180)
                fig = go.Figure()
                fig.add_trace(go.Bar(x=liq_30d["timestamp"], y=liq_30d["long_liq_usd"]/1e6,
                    name="Long Liqs ($M)", marker_color=RED, opacity=0.8))
                fig.add_trace(go.Bar(x=liq_30d["timestamp"], y=-liq_30d["short_liq_usd"]/1e6,
                    name="Short Liqs ($M)", marker_color=GREEN, opacity=0.8))
                fig.update_layout(**PLOTLY, height=250, barmode="relative",
                                  title="Liquidações Long (vermelho) vs Short (verde) ($M)")
                st.plotly_chart(fig, use_container_width=True)

        with tab4:
            if not taker_df.empty and len(taker_df) > 10:
                t_30d = taker_df.tail(720)
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=t_30d["timestamp"], y=t_30d["buy_sell_ratio"],
                    name="Taker Buy/Sell", line=dict(color=BLUE, width=1.5)))
                fig.add_hline(y=1.0, line_color=GREY, opacity=0.5)
                fig.update_layout(**PLOTLY, height=250, title="Taker Buy/Sell Ratio (30d)")
                st.plotly_chart(fig, use_container_width=True)

        with tab5:
            if not ob_df.empty and len(ob_df) > 10:
                ob_30d = ob_df.tail(180)
                sp_4h  = spot_df.tail(180) if not spot_df.empty else pd.DataFrame()

                fig = make_subplots(
                    rows=2, cols=1, shared_xaxes=True,
                    row_heights=[0.5, 0.5],
                    vertical_spacing=0.06,
                )
                # Top: bid_ask_ratio vs price
                fig.add_trace(go.Scatter(
                    x=ob_30d["timestamp"], y=ob_30d["bid_ask_ratio"],
                    name="Bid/Ask Ratio (Binance)", line=dict(color=BLUE, width=1.5),
                ), row=1, col=1)
                if not ob_agg_df.empty and len(ob_agg_df) > 10:
                    ob_agg_30d = ob_agg_df.tail(180)
                    fig.add_trace(go.Scatter(
                        x=ob_agg_30d["timestamp"], y=ob_agg_30d["bid_ask_ratio"],
                        name="Bid/Ask Ratio (Agg)", line=dict(color=AMBER, width=1.2, dash="dot"),
                    ), row=1, col=1)
                fig.add_hline(y=1.0, line_color=GREY, opacity=0.5, row=1, col=1)

                # Bottom: bids vs asks (stacked area)
                fig.add_trace(go.Scatter(
                    x=ob_30d["timestamp"], y=ob_30d["bids_usd"] / 1e6,
                    name="Bids ($M)", line=dict(color=GREEN, width=1),
                    fill="tozeroy", fillcolor="rgba(63,185,80,0.15)",
                ), row=2, col=1)
                fig.add_trace(go.Scatter(
                    x=ob_30d["timestamp"], y=ob_30d["asks_usd"] / 1e6,
                    name="Asks ($M)", line=dict(color=RED, width=1),
                    fill="tozeroy", fillcolor="rgba(248,81,73,0.15)",
                ), row=2, col=1)

                fig.update_layout(**PLOTLY, height=380,
                                  title="Bid/Ask Ratio + Volume ($M) — Binance BTCUSDT 4h",
                                  showlegend=True)
                fig.update_yaxes(title_text="Ratio", row=1, col=1)
                fig.update_yaxes(title_text="$M", row=2, col=1)
                st.plotly_chart(fig, use_container_width=True)

    # =========================================================================
    # SECTION 5: MACRO
    # =========================================================================
    st.markdown("---")
    st.markdown("### 🏛️ Macro")

    # Variações
    def _chg(df, col="close", days=1):
        if df.empty or len(df) < days+1:
            return 0.0
        return (df[col].iloc[-1] - df[col].iloc[-days-1]) / df[col].iloc[-days-1] * 100

    vix_chg   = _chg(vix_df)
    dxy_chg   = _chg(dxy_df)
    oil_chg   = _chg(oil_df)
    sp500_chg = _chg(sp500_df)

    curve = (dgs10 or 0) - (dgs2 or 0)
    macro_cols = st.columns(7)
    macro_data = [
        ("DGS10", f"{dgs10:.2f}%", f"z={zs.get('dgs10_z',0):.2f}", zs.get("dgs10_z",0) < 0),
        ("DGS2",  f"{dgs2:.2f}%",  f"z={zs.get('dgs2_z',0):.2f}",  zs.get("dgs2_z",0) < 0),
        ("2/10y", f"{curve*100:.0f}bps", f"z={zs.get('curve_z',0):.2f}", zs.get("curve_z",0) > 0),
        ("VIX",   f"{vix_val:.1f}", f"{vix_chg:+.2f}%", vix_chg < 0),
        ("DXY",   f"{dxy_val:.2f}", f"{dxy_chg:+.2f}%", dxy_chg < 0),
        ("Oil",   f"${oil_val:.1f}", f"{oil_chg:+.2f}%", oil_chg < 0),
        ("S&P500",f"{sp500_val:,.0f}", f"{sp500_chg:+.2f}%", sp500_chg > 0),
    ]
    for col, (title, val, sub, good) in zip(macro_cols, macro_data):
        sub_c = "pos" if good else "neg"
        with col:
            st.markdown(f"""
<div class="cg-card" style="text-align:center; padding:10px 8px;">
  <div class="cg-card-title">{title}</div>
  <div style="font-size:16px; font-weight:700;">{val}</div>
  <div class="cg-card-sub {sub_c}">{sub}</div>
</div>""", unsafe_allow_html=True)

    # =========================================================================
    # SECTION 6: NEWS & SENTIMENT
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

        st.markdown(f"""
<div class="cg-card" style="margin-top:8px;">
  <div class="cg-card-title">🏦 Fed Sentinel</div>
  <div style="font-size:13px; margin-top:4px;">
    <div>Próx evento: <b>{fomc.get('next_event','N/A')}</b></div>
    <div>Em: <b>{fomc.get('days_away','—')} dias</b></div>
    <div>Proximity adj: <b>+{fomc.get('proximity_adj',0):.1f}</b></div>
    <div>Blackout: <b>{'Sim ⚠️' if fomc.get('in_blackout') else 'Não'}</b></div>
  </div>
</div>""", unsafe_allow_html=True)

    # =========================================================================
    # SECTION 7: AI ANALYST
    # =========================================================================
    st.markdown("---")
    st.markdown("### 🤖 AI Analyst (DeepSeek)")

    if st.button("🔍 Gerar Análise", type="primary"):
        context = {
            "price": price, "pct_24h": pct_24h, "regime": regime,
            "signal": signal, "score": total_score, "threshold": threshold,
            "capital": capital, "has_position": portfolio.get("has_position", False),
            "c_technical": clusters["technical"], "c_positioning": clusters["positioning"],
            "c_macro": clusters["macro"], "c_liquidity": clusters["liquidity"],
            "c_sentiment": clusters["sentiment"], "c_news": clusters["news"],
            "oi_z": zs.get("oi_z",0), "taker_z": zs.get("taker_z",0),
            "funding_z": zs.get("funding_z",0), "dgs10_z": zs.get("dgs10_z",0),
            "curve_z": zs.get("curve_z",0), "stablecoin_z": zs.get("stablecoin_z",0),
            "etf_z": zs.get("etf_z",0), "fg_z": zs.get("fg_z",0),
            "bubble_z": zs.get("bubble_z",0), "ls_account": ls_account_val,
            "whale_signal": ws_label, "fed_event": fomc.get("next_event","N/A"),
            "fed_days": fomc.get("days_away","—"), "fed_adj": fomc.get("proximity_adj",0),
        }
        with st.spinner("Consultando DeepSeek..."):
            analysis = call_deepseek_analyst(context)
        st.markdown(f"""
<div class="cg-card" style="font-size:14px; line-height:1.7;">
{analysis.replace(chr(10), '<br>')}
</div>""", unsafe_allow_html=True)

    # =========================================================================
    # SECTION 8: SYSTEM HEALTH
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

    st.caption(f"btc-trading-v1 | {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} | Auto-refresh: {'ON' if auto_refresh else 'OFF'}")


if __name__ == "__main__" or True:
    main()
