# PROMPT — Fed Observatory (substituto gratuito do FedWatch)

## Objetivo
Adicionar seção "Fed Observatory" no dashboard, entre News & Sentiment 
e System Health. Usa dados gratuitos (FRED + Fed Sentinel) pra estimar 
probabilidades de corte/manutenção/alta e mostrar cenários.

## Passo 1: Adicionar séries FRED extras

Em `conf/parameters.yml`, adicionar:
```yaml
ingestion:
  fred:
    series:
      - DGS10
      - DGS2
      - RRPONTSYD
      - WALCL
      - EFFR        # NOVO: Effective Fed Funds Rate
      - DFEDTARU    # NOVO: Target Upper
      - DFEDTARL    # NOVO: Target Lower
      - T5YIE       # NOVO: Breakeven Inflation 5Y
      - T10YIE      # NOVO: Breakeven Inflation 10Y
```

Em `conf/catalog.yml`, adicionar:
```yaml
macro_effr: data/01_raw/macro/effr.parquet
macro_dfedtaru: data/01_raw/macro/dfedtaru.parquet
macro_dfedtarl: data/01_raw/macro/dfedtarl.parquet
macro_t5yie: data/01_raw/macro/t5yie.parquet
macro_t10yie: data/01_raw/macro/t10yie.parquet
```

Atualizar `src/data/fred_ingest.py` pra puxar as novas séries.
Mesmo padrão: incremental, UTC, retry.

## Passo 2: Criar src/features/fed_observatory.py

```python
"""
Fed Observatory — Estimativa de probabilidades de decisão do Fed.
Substituto gratuito do CME FedWatch ($25/mês).
Usa DGS2 como proxy de expectativa de juros curtos.
"""

import pandas as pd
import numpy as np
from src.config import get_params, get_path
import json
import logging

logger = logging.getLogger("features.fed_observatory")


def load_fed_data():
    """Carrega todos os dados necessários pro observatory."""
    data = {}
    
    for name in ["dgs10", "dgs2", "effr", "dfedtaru", "dfedtarl", "t5yie", "t10yie"]:
        try:
            key = f"macro_{name}"
            path = get_path(key)
            df = pd.read_parquet(path)
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            df = df.sort_values("timestamp")
            data[name] = df
        except Exception as e:
            logger.warning(f"Fed Observatory: {name} not available: {e}")
            data[name] = None
    
    return data


def estimate_rate_probability(data):
    """
    Estima probabilidade de corte/manutenção/alta usando DGS2 como proxy.
    
    Lógica: DGS2 precifica expectativa de juros curtos nos próximos 2 anos.
    Se DGS2 cai abaixo do fed rate atual → mercado antecipa corte.
    Velocidade da mudança indica urgência da expectativa.
    """
    params = get_params()
    fed_rate = params.get("fed", {}).get("current_rate_mid", 3.625)
    # current_rate_mid = (3.50 + 3.75) / 2
    
    result = {
        "fed_rate_current": f"{fed_rate - 0.125:.2f}-{fed_rate + 0.125:.2f}%",
        "fed_rate_mid": fed_rate,
        "prob_cut": 0.0,
        "prob_hold": 1.0,
        "prob_hike": 0.0,
        "confidence": "low",
        "indicators": {},
    }
    
    # DGS2 atual e 30d atrás
    if data.get("dgs2") is not None and len(data["dgs2"]) > 0:
        dgs2_df = data["dgs2"]
        dgs2_now = dgs2_df["value"].iloc[-1]
        
        # 30 dias atrás
        cutoff_30d = dgs2_df["timestamp"].iloc[-1] - pd.Timedelta(days=30)
        dgs2_30d = dgs2_df[dgs2_df["timestamp"] <= cutoff_30d]["value"]
        dgs2_prev = dgs2_30d.iloc[-1] if len(dgs2_30d) > 0 else dgs2_now
        
        dgs2_change_30d = dgs2_now - dgs2_prev
        spread_vs_fed = dgs2_now - fed_rate
        
        result["indicators"]["dgs2"] = dgs2_now
        result["indicators"]["dgs2_change_30d"] = dgs2_change_30d
        result["indicators"]["spread_vs_fed"] = spread_vs_fed
        
        # Probabilidade base pelo spread
        if spread_vs_fed < -0.25:
            prob_cut = 0.85
        elif spread_vs_fed < -0.10:
            prob_cut = 0.65
        elif spread_vs_fed < 0:
            prob_cut = 0.45
        elif spread_vs_fed < 0.15:
            prob_cut = 0.25
        elif spread_vs_fed < 0.30:
            prob_cut = 0.10
        else:
            prob_cut = 0.05
        
        # Ajustar pela velocidade de mudança
        if dgs2_change_30d < -0.30:
            prob_cut += 0.20
        elif dgs2_change_30d < -0.15:
            prob_cut += 0.10
        elif dgs2_change_30d > 0.30:
            prob_cut -= 0.15
        elif dgs2_change_30d > 0.15:
            prob_cut -= 0.08
        
        prob_cut = max(0.0, min(0.95, prob_cut))
        
        # Probabilidade de alta (raro)
        prob_hike = 0.0
        if spread_vs_fed > 0.50 and dgs2_change_30d > 0.30:
            prob_hike = 0.10
        
        result["prob_cut"] = round(prob_cut, 3)
        result["prob_hold"] = round(1.0 - prob_cut - prob_hike, 3)
        result["prob_hike"] = round(prob_hike, 3)
        result["confidence"] = "medium"
    
    # EFFR
    if data.get("effr") is not None and len(data["effr"]) > 0:
        effr = data["effr"]["value"].iloc[-1]
        result["indicators"]["effr"] = effr
    
    # Target bounds
    if data.get("dfedtaru") is not None and len(data["dfedtaru"]) > 0:
        result["indicators"]["target_upper"] = data["dfedtaru"]["value"].iloc[-1]
    if data.get("dfedtarl") is not None and len(data["dfedtarl"]) > 0:
        result["indicators"]["target_lower"] = data["dfedtarl"]["value"].iloc[-1]
    
    # Breakeven inflation
    if data.get("t5yie") is not None and len(data["t5yie"]) > 0:
        t5yie = data["t5yie"]["value"].iloc[-1]
        result["indicators"]["inflation_5y"] = t5yie
        
        # Inflação caindo → mais chance de corte
        cutoff = data["t5yie"]["timestamp"].iloc[-1] - pd.Timedelta(days=30)
        t5yie_30d = data["t5yie"][data["t5yie"]["timestamp"] <= cutoff]["value"]
        if len(t5yie_30d) > 0:
            t5yie_change = t5yie - t5yie_30d.iloc[-1]
            result["indicators"]["inflation_5y_change_30d"] = t5yie_change
    
    if data.get("t10yie") is not None and len(data["t10yie"]) > 0:
        result["indicators"]["inflation_10y"] = data["t10yie"]["value"].iloc[-1]
    
    return result


def get_scenario_analysis(prob_result, fed_sentinel_data=None):
    """
    Gera análise de cenários baseada nas probabilidades.
    """
    prob_cut = prob_result["prob_cut"]
    prob_hold = prob_result["prob_hold"]
    prob_hike = prob_result["prob_hike"]
    
    scenarios = []
    
    # Cenário corte
    scenarios.append({
        "name": "📉 Corte 25bps",
        "probability": f"{prob_cut*100:.0f}%",
        "btc_impact": "Bullish",
        "action": "Reduzir threshold, aumentar sizing em Sideways",
        "description": ("Historicamente, primeiro corte após ciclo de alta = BTC "
                       "sobe 30-50% em 6 meses. Dólar enfraquece, capital migra "
                       "pra risk-on. Stablecoins entram forte."),
        "color": "green",
    })
    
    # Cenário manutenção
    scenarios.append({
        "name": "➡️ Manutenção",
        "probability": f"{prob_hold*100:.0f}%",
        "btc_impact": "Neutro",
        "action": "Manter scoring atual",
        "description": ("Já precificado pelo mercado. Impacto depende do guidance "
                       "e dot plot. Atenção ao tom da coletiva — hawkish surprise "
                       "pode derrubar, dovish surprise pode subir."),
        "color": "gray",
    })
    
    # Cenário alta
    if prob_hike > 0.01:
        scenarios.append({
            "name": "📈 Alta 25bps",
            "probability": f"{prob_hike*100:.0f}%",
            "btc_impact": "Bearish",
            "action": "Kill switch ativado, forçar Bear regime",
            "description": ("Cenário de stress. BTC cairia 10-20%. "
                           "Trigger: inflação voltando por guerra/commodities."),
            "color": "red",
        })
    
    # Agregar sentiment dos membros (se disponível)
    member_summary = None
    if fed_sentinel_data:
        hawkish = sum(1 for a in fed_sentinel_data if a.get("sentiment") == "hawkish")
        dovish = sum(1 for a in fed_sentinel_data if a.get("sentiment") == "dovish")
        neutral = sum(1 for a in fed_sentinel_data if a.get("sentiment") == "neutral")
        member_summary = {
            "hawkish": hawkish,
            "dovish": dovish,
            "neutral": neutral,
            "trend": "hawkish" if hawkish > dovish else "dovish" if dovish > hawkish else "balanced",
        }
    
    return {
        "scenarios": scenarios,
        "member_summary": member_summary,
    }


def run():
    """Roda análise completa do Fed Observatory."""
    data = load_fed_data()
    prob = estimate_rate_probability(data)
    scenarios = get_scenario_analysis(prob)
    
    logger.info(f"Fed Observatory: cut={prob['prob_cut']:.0%} hold={prob['prob_hold']:.0%} "
                f"hike={prob['prob_hike']:.0%} | DGS2={prob['indicators'].get('dgs2', '?')}")
    
    return {**prob, **scenarios}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    result = run()
    print(json.dumps(result, indent=2, default=str))
```

## Passo 3: Adicionar no Dashboard

Em `src/dashboard/app.py`, adicionar seção "Fed Observatory" ENTRE 
"News & Sentiment" e "System Health".

```python
# ── SECTION: Fed Observatory ──

st.markdown("## 🏛️ Fed Observatory")

# Importar e rodar
from src.features.fed_observatory import load_fed_data, estimate_rate_probability, get_scenario_analysis

fed_data = load_fed_data()
prob = estimate_rate_probability(fed_data)
analysis = get_scenario_analysis(prob)

# --- Probabilidades ---
col1, col2, col3 = st.columns(3)
with col1:
    pct = prob["prob_cut"] * 100
    color = "#3fb950" if pct > 30 else "#8b949e"
    st.markdown(f"""
    <div class="metric-card">
        <div style="font-size:0.8em; color:#8b949e;">CORTE 25bps</div>
        <div style="font-size:1.8em; font-weight:bold; color:{color};">{pct:.0f}%</div>
        <div style="font-size:0.7em;">BTC: Bullish</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    pct = prob["prob_hold"] * 100
    st.markdown(f"""
    <div class="metric-card">
        <div style="font-size:0.8em; color:#8b949e;">MANUTENÇÃO</div>
        <div style="font-size:1.8em; font-weight:bold; color:#8b949e;">{pct:.0f}%</div>
        <div style="font-size:0.7em;">BTC: Neutro</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    pct = prob["prob_hike"] * 100
    color = "#f85149" if pct > 10 else "#8b949e"
    st.markdown(f"""
    <div class="metric-card">
        <div style="font-size:0.8em; color:#8b949e;">ALTA 25bps</div>
        <div style="font-size:1.8em; font-weight:bold; color:{color};">{pct:.0f}%</div>
        <div style="font-size:0.7em;">BTC: Bearish</div>
    </div>
    """, unsafe_allow_html=True)

# --- Indicadores ---
indicators = prob.get("indicators", {})
cols = st.columns(5)
indicator_list = [
    ("DGS2", indicators.get("dgs2"), "%", indicators.get("dgs2_change_30d")),
    ("EFFR", indicators.get("effr"), "%", None),
    ("Spread vs Fed", indicators.get("spread_vs_fed"), "bps", None),
    ("Inflation 5Y", indicators.get("inflation_5y"), "%", indicators.get("inflation_5y_change_30d")),
    ("Inflation 10Y", indicators.get("inflation_10y"), "%", None),
]

for i, (name, val, unit, change) in enumerate(indicator_list):
    with cols[i]:
        if val is not None:
            change_str = f" ({change:+.2f})" if change is not None else ""
            if unit == "bps":
                st.metric(name, f"{val*100:+.0f}bps")
            else:
                st.metric(name, f"{val:.2f}%", delta=f"{change:+.2f}" if change else None)
        else:
            st.metric(name, "N/A")

# --- Cenários ---
st.markdown("**Cenários por decisão:**")
for scenario in analysis["scenarios"]:
    color_map = {"green": "#3fb950", "gray": "#8b949e", "red": "#f85149"}
    border_color = color_map.get(scenario["color"], "#8b949e")
    st.markdown(f"""
    <div style="background:#161b22; border-left:3px solid {border_color}; 
                padding:10px 15px; margin:5px 0; border-radius:4px;">
        <strong>{scenario["name"]}</strong> — {scenario["probability"]}
        <br><span style="font-size:0.85em; color:#8b949e;">
        Impacto BTC: {scenario["btc_impact"]} | Ação: {scenario["action"]}
        </span>
        <br><span style="font-size:0.8em; color:#6e7681;">{scenario["description"]}</span>
    </div>
    """, unsafe_allow_html=True)

# --- Membros ---
if analysis.get("member_summary"):
    ms = analysis["member_summary"]
    st.markdown(f"""
    **Membros Fed (últimos 30 dias):**
    🔴 Hawkish: {ms['hawkish']} | 🟢 Dovish: {ms['dovish']} | ⚪ Neutro: {ms['neutral']}
    → Tendência: **{ms['trend']}**
    """)

# --- Agenda ---
# Ler de fed_calendar.json
import json
try:
    with open("conf/fed_calendar.json") as f:
        cal = json.load(f)
    events = cal.get("events", [])
    upcoming = [e for e in events 
                if pd.to_datetime(e["date"]) > pd.Timestamp.now(tz="UTC")][:5]
    if upcoming:
        st.markdown("**Agenda Fed:**")
        for e in upcoming:
            days = (pd.to_datetime(e["date"]) - pd.Timestamp.now(tz="UTC")).days
            st.markdown(f"📅 {e['date']} — {e['type']} ({days}d)")
except:
    pass
```

## Passo 4: Adicionar parâmetro do fed rate atual

Em `conf/parameters.yml`, dentro do bloco `fed:`:
```yaml
fed:
  current_rate: "3.50-3.75"
  current_rate_mid: 3.625
```

## Passo 5: Testar

```bash
# Puxar novas séries FRED
python -m src.data.fred_ingest

# Testar observatory standalone
python -m src.features.fed_observatory

# Rodar dashboard
streamlit run src/dashboard/app.py --server.port 8501
```

## IMPORTANT

- NÃO é tão preciso quanto o FedWatch ($25/mês) — é uma aproximação
- Mostrar no dashboard: "Probabilidades estimadas (proxy DGS2, não FedWatch)"
- Confidence level: low/medium dependendo dos dados disponíveis
- Se no futuro quiser FedWatch real, é $25/mês via API REST
