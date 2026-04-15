"""
Fed Observatory — Estimativa de probabilidades de decisão do Fed.
Substituto gratuito do CME FedWatch ($25/mês).
Usa DGS2 como proxy de expectativa de juros curtos.
"""

import json
import logging

import numpy as np
import pandas as pd

from src.config import get_params, get_path

logger = logging.getLogger("features.fed_observatory")


def load_fed_data():
    """Carrega todos os dados necessários pro observatory."""
    data = {}

    # DGS10 e DGS2 são colunas em clean_macro (fred_daily_clean.parquet)
    try:
        clean_path = get_path("clean_macro")
        clean_df = pd.read_parquet(clean_path)
        clean_df["timestamp"] = pd.to_datetime(clean_df["timestamp"], utc=True)
        clean_df = clean_df.sort_values("timestamp")

        for col in ["dgs10", "dgs2"]:
            col_upper = col.upper()
            if col_upper in clean_df.columns:
                data[col] = clean_df[["timestamp", col_upper]].rename(columns={col_upper: "value"}).dropna()
            elif col in clean_df.columns:
                data[col] = clean_df[["timestamp", col]].rename(columns={col: "value"}).dropna()
            else:
                logger.warning(f"Fed Observatory: {col} column not found in clean_macro")
                data[col] = None
    except Exception as e:
        logger.warning(f"Fed Observatory: clean_macro not available: {e}")
        data["dgs10"] = None
        data["dgs2"] = None

    # EFFR, target bounds e breakeven inflation — parquets individuais
    for name in ["effr", "dfedtaru", "dfedtarl", "t5yie", "t10yie"]:
        try:
            path = get_path(f"macro_{name}")
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
        result["indicators"]["effr"] = data["effr"]["value"].iloc[-1]

    # Target bounds
    if data.get("dfedtaru") is not None and len(data["dfedtaru"]) > 0:
        result["indicators"]["target_upper"] = data["dfedtaru"]["value"].iloc[-1]
    if data.get("dfedtarl") is not None and len(data["dfedtarl"]) > 0:
        result["indicators"]["target_lower"] = data["dfedtarl"]["value"].iloc[-1]

    # Breakeven inflation
    if data.get("t5yie") is not None and len(data["t5yie"]) > 0:
        t5yie = data["t5yie"]["value"].iloc[-1]
        result["indicators"]["inflation_5y"] = t5yie

        cutoff = data["t5yie"]["timestamp"].iloc[-1] - pd.Timedelta(days=30)
        t5yie_30d = data["t5yie"][data["t5yie"]["timestamp"] <= cutoff]["value"]
        if len(t5yie_30d) > 0:
            result["indicators"]["inflation_5y_change_30d"] = t5yie - t5yie_30d.iloc[-1]

    if data.get("t10yie") is not None and len(data["t10yie"]) > 0:
        result["indicators"]["inflation_10y"] = data["t10yie"]["value"].iloc[-1]

    return result


def get_scenario_analysis(prob_result, fed_sentinel_data=None):
    """Gera análise de cenários baseada nas probabilidades."""
    prob_cut = prob_result["prob_cut"]
    prob_hold = prob_result["prob_hold"]
    prob_hike = prob_result["prob_hike"]

    scenarios = []

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

    logger.info(
        f"Fed Observatory: cut={prob['prob_cut']:.0%} hold={prob['prob_hold']:.0%} "
        f"hike={prob['prob_hike']:.0%} | DGS2={prob['indicators'].get('dgs2', '?')}"
    )

    return {**prob, **scenarios}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    result = run()
    print(json.dumps(result, indent=2, default=str))
