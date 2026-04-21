"""
Shadow filters — avaliação passiva de filtros sem alterar comportamento.

Registra o que o filtro taker_z FARIA (bloquearia ou não) sem bloquear nada.
Permite validação out-of-sample antes de ativar em produção.

Uso:
    from src.trading.shadow_filters import evaluate_taker_z_shadow
    # Dentro do paper_trader, após confirmar entrada:
    evaluate_taker_z_shadow(entry_time, trade_id, bot_origin)
"""
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger("trading.shadow_filters")

GATE_ZSCORES_PATH = Path("data/02_features/gate_zscores.parquet")
SHADOW_LOG_DIR = Path("data/08_shadow")
SHADOW_LOG_FILE = SHADOW_LOG_DIR / "taker_z_shadow_log.jsonl"

FILTER_THRESHOLD = -1.0
FILTER_VERSION = "v1"


def _load_gate_zscores() -> pd.DataFrame | None:
    if not GATE_ZSCORES_PATH.exists():
        logger.warning(f"gate_zscores not found at {GATE_ZSCORES_PATH}")
        return None
    try:
        df = pd.read_parquet(GATE_ZSCORES_PATH)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        return df.sort_values("timestamp").reset_index(drop=True)
    except Exception as e:
        logger.error(f"Failed to load gate_zscores: {e}")
        return None


def _get_prev_value(df: pd.DataFrame, entry_time: pd.Timestamp, column: str) -> float | None:
    """Último valor ANTES de entry_time (strict <, anti look-ahead)."""
    if column not in df.columns:
        return None
    prev = df[df["timestamp"] < entry_time]
    if prev.empty:
        return None
    val = prev.iloc[-1][column]
    return float(val) if pd.notna(val) else None


def evaluate_taker_z_shadow(
    entry_time: pd.Timestamp,
    trade_id: str | int | None = None,
    bot_origin: str | None = None,
) -> dict[str, Any]:
    """
    Avalia filtro taker_z em shadow mode (sem bloquear).

    Args:
        entry_time:  timestamp UTC da entrada do trade
        trade_id:    UUID do trade (para rastreamento posterior)
        bot_origin:  "bot1" | "bot2" | "bot3"

    Returns:
        dict com avaliação — também persiste em JSONL.
    """
    if not isinstance(entry_time, pd.Timestamp):
        entry_time = pd.Timestamp(entry_time)
    if entry_time.tz is None:
        entry_time = entry_time.tz_localize("UTC")

    log_entry: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "trade_id": str(trade_id) if trade_id is not None else None,
        "bot_origin": bot_origin,
        "entry_time": entry_time.isoformat(),
        "filter_version": FILTER_VERSION,
        "threshold": FILTER_THRESHOLD,
        "status": "unknown",
    }

    df = _load_gate_zscores()
    if df is None:
        log_entry["status"] = "error_no_gate_zscores"
        _persist_log(log_entry)
        return log_entry

    taker_z_4h = _get_prev_value(df, entry_time, "taker_z")
    taker_z_1h = _get_prev_value(df, entry_time, "taker_z_1h")

    prev_rows = df[df["timestamp"] < entry_time]
    prev_candle_time = prev_rows.iloc[-1]["timestamp"].isoformat() if not prev_rows.empty else None

    would_block_4h = (taker_z_4h is not None) and (taker_z_4h < FILTER_THRESHOLD)
    would_block_1h = (taker_z_1h is not None) and (taker_z_1h < FILTER_THRESHOLD)

    log_entry.update({
        "status": "ok",
        "prev_candle_time": prev_candle_time,
        "taker_z_4h": taker_z_4h,
        "taker_z_1h": taker_z_1h,
        "would_block_4h": would_block_4h,
        "would_block_1h": would_block_1h,
        "both_agree_block": would_block_4h and would_block_1h,
        "disagreement": would_block_4h != would_block_1h,
    })

    _persist_log(log_entry)

    _4h_str = f"{taker_z_4h:.3f}" if taker_z_4h is not None else "N/A"
    _1h_str = f"{taker_z_1h:.3f}" if taker_z_1h is not None else "N/A"
    logger.info(
        f"SHADOW taker_z | trade={trade_id} bot={bot_origin} | "
        f"4h={_4h_str} would_block={would_block_4h} | "
        f"1h={_1h_str} would_block={would_block_1h}"
    )
    return log_entry


def _persist_log(entry: dict[str, Any]) -> None:
    try:
        SHADOW_LOG_DIR.mkdir(parents=True, exist_ok=True)
        with open(SHADOW_LOG_FILE, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as e:
        logger.error(f"Failed to persist shadow log: {e}")
