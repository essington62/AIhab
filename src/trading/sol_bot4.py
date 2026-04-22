"""
SOL Bot 4 — Taker Aggression + Flow Strategy.

Filosofia: Bot 2 Momentum adaptada para SOL (Phase 1 EDA + Phase 2 backtest).

Hard gates (Phase 2 validated, Test Sharpe 2.03):
  G0: ETH ret_1h_prev > 0         (H3: β_ETH=0.629 > β_BTC)
  G1: taker_z_prev > 0.3          (H1: Cohen's d=0.51-0.61)
  G2: oi_z_24h_max < 2.0          (H1 bipolar: block reversal)
  F:  60 <= RSI <= 80, MA21, ret_1d > 0

Exit: TP +2% / SL -1.5% / Trail -1% / OI early exit (>12h hold) / max 120h

Shadow scoring logged for future comparison (scoring v1 hypothesis).
"""
import json
import logging
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

logger = logging.getLogger("sol_bot4")

PARAMS_PATH = ROOT / "conf/parameters_sol.yml"
PORTFOLIO_PATH = ROOT / "data/04_scoring/portfolio_sol.json"
TRADES_PATH = ROOT / "data/05_trades/completed_trades_sol.json"
LOCK_PATH = ROOT / "data/04_scoring/sol_cycle.lock"
SHADOW_LOG_PATH = ROOT / "data/08_shadow/sol_scoring_shadow_log.jsonl"

for p in [PORTFOLIO_PATH.parent, TRADES_PATH.parent, SHADOW_LOG_PATH.parent]:
    p.mkdir(parents=True, exist_ok=True)


# ==================================================================
# CONFIG
# ==================================================================

def get_params() -> dict:
    with open(PARAMS_PATH) as f:
        return yaml.safe_load(f)["sol_bot"]


# ==================================================================
# LOCK
# ==================================================================

def acquire_lock() -> bool:
    if LOCK_PATH.exists():
        try:
            age = datetime.now(timezone.utc).timestamp() - LOCK_PATH.stat().st_mtime
            if age < 300:
                return False
            LOCK_PATH.unlink()
        except Exception:
            pass
    try:
        LOCK_PATH.touch()
        return True
    except Exception:
        return False


def release_lock():
    try:
        if LOCK_PATH.exists():
            LOCK_PATH.unlink()
    except Exception:
        pass


# ==================================================================
# PORTFOLIO STATE
# ==================================================================

def load_portfolio() -> dict:
    if not PORTFOLIO_PATH.exists():
        params = get_params()
        state = {
            "capital_usd": float(params.get("capital_usd", 10000)),
            "has_position": False,
            "entry_price": None, "quantity": None,
            "stop_loss_price": None, "take_profit_price": None,
            "trailing_high": None, "entry_timestamp": None,
            "entry_features": None, "max_hold_until": None,
            "last_update": datetime.now(timezone.utc).isoformat(),
        }
        save_portfolio(state)
        return state
    with open(PORTFOLIO_PATH) as f:
        return json.load(f)


def save_portfolio(portfolio: dict):
    portfolio["last_update"] = datetime.now(timezone.utc).isoformat()
    PORTFOLIO_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(PORTFOLIO_PATH, "w") as f:
        json.dump(portfolio, f, indent=2, default=str)


# ==================================================================
# LIVE PRICE
# ==================================================================

def get_live_price(timeout: float = 5.0) -> float | None:
    try:
        r = requests.get(
            "https://api.binance.com/api/v3/ticker/price",
            params={"symbol": "SOLUSDT"}, timeout=timeout,
        )
        r.raise_for_status()
        return float(r.json()["price"])
    except Exception as e:
        logger.warning(f"get_live_price(SOLUSDT) failed: {e}")
        return None


# ==================================================================
# FEATURE ENGINEERING
# ==================================================================

def _zscore(s: pd.Series, window: int) -> pd.Series:
    m = s.rolling(window, min_periods=max(10, window // 4)).mean()
    sd = s.rolling(window, min_periods=max(10, window // 4)).std()
    return (s - m) / sd.replace(0, np.nan)


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def compute_sol_features() -> dict:
    """
    Computa features do último candle fechado (shift(1) anti look-ahead).
    Returns dict com todos os valores necessários para decisão.
    """
    # SOL OHLCV
    sol_path = ROOT / "data/01_raw/spot/sol_1h.parquet"
    if not sol_path.exists():
        raise FileNotFoundError(f"SOL spot data missing: {sol_path}")
    sol = pd.read_parquet(sol_path)
    sol["timestamp"] = pd.to_datetime(sol["timestamp"], utc=True)
    sol = sol.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)

    sol["rsi"] = _rsi(sol["close"])
    sol["ma21"] = sol["close"].rolling(21).mean()
    sol["ret_1d"] = sol["close"].pct_change(24)

    # ETH context (G0 — H3)
    eth_path = ROOT / "data/01_raw/spot/eth_1h.parquet"
    eth_ret_1h_prev = None
    if eth_path.exists():
        eth = pd.read_parquet(eth_path)
        eth["timestamp"] = pd.to_datetime(eth["timestamp"], utc=True)
        eth = eth.sort_values("timestamp").reset_index(drop=True)
        eth["eth_ret_1h"] = eth["close"].pct_change()
        # shift(1): use candle before current
        eth["eth_ret_1h_prev"] = eth["eth_ret_1h"].shift(1)
        # Align with SOL last timestamp
        sol_last_ts = sol["timestamp"].iloc[-1]
        match = eth[eth["timestamp"] <= sol_last_ts]
        if not match.empty:
            eth_ret_1h_prev = float(match.iloc[-1]["eth_ret_1h_prev"] or 0)

    # SOL OI 4h
    oi_path = ROOT / "data/01_raw/futures/sol_oi_4h.parquet"
    oi_z_prev = None
    oi_z_24h_max = None
    if oi_path.exists():
        oi = pd.read_parquet(oi_path)
        oi["timestamp"] = pd.to_datetime(oi["timestamp"], utc=True)
        oi = oi.sort_values("timestamp").reset_index(drop=True)
        oi["oi_z"] = _zscore(oi["open_interest"], 42)
        oi["oi_z_24h_max"] = oi["oi_z"].rolling(6, min_periods=1).max()
        # shift(1) for entry features — last COMPLETED candle before now
        oi["oi_z_prev"] = oi["oi_z"].shift(1)
        oi["oi_z_24h_max_prev"] = oi["oi_z_24h_max"].shift(1)
        sol_last_ts = sol["timestamp"].iloc[-1]
        oi_match = oi[oi["timestamp"] <= sol_last_ts]
        if not oi_match.empty:
            last_oi = oi_match.iloc[-1]
            oi_z_prev = float(last_oi["oi_z_prev"]) if pd.notna(last_oi["oi_z_prev"]) else None
            oi_z_24h_max = float(last_oi["oi_z_24h_max"]) if pd.notna(last_oi["oi_z_24h_max"]) else None

    # SOL Taker 4h (G1)
    taker_path = ROOT / "data/01_raw/futures/sol_taker_4h.parquet"
    taker_z_prev = None
    if taker_path.exists():
        taker = pd.read_parquet(taker_path)
        taker["timestamp"] = pd.to_datetime(taker["timestamp"], utc=True)
        taker = taker.sort_values("timestamp").reset_index(drop=True)
        buy_col = next((c for c in ["buy_volume_usd", "taker_buy_volume_usd"] if c in taker.columns), None)
        sell_col = next((c for c in ["sell_volume_usd", "taker_sell_volume_usd"] if c in taker.columns), None)
        ratio_col = next((c for c in ["taker_ratio", "buy_sell_ratio"] if c in taker.columns), None)
        if buy_col and sell_col:
            total = taker[buy_col] + taker[sell_col]
            taker["taker_ratio"] = taker[buy_col] / total.replace(0, np.nan)
        elif ratio_col:
            taker["taker_ratio"] = taker[ratio_col]
        taker["taker_z"] = _zscore(taker["taker_ratio"], 42)
        taker["taker_z_prev"] = taker["taker_z"].shift(1)  # anti look-ahead
        sol_last_ts = sol["timestamp"].iloc[-1]
        taker_match = taker[taker["timestamp"] <= sol_last_ts]
        if not taker_match.empty:
            v = taker_match.iloc[-1]["taker_z_prev"]
            taker_z_prev = float(v) if pd.notna(v) else None

    # Get current (latest closed) candle
    latest = sol.iloc[-1]
    prev = sol.iloc[-2] if len(sol) >= 2 else latest  # prev candle for high/low checks

    return {
        "timestamp": latest["timestamp"],
        "close": float(latest["close"]),
        "high": float(latest["high"]),
        "low": float(latest["low"]),
        "rsi": float(latest["rsi"]) if pd.notna(latest.get("rsi")) else None,
        "ma21": float(latest["ma21"]) if pd.notna(latest.get("ma21")) else None,
        "ret_1d": float(latest["ret_1d"]) if pd.notna(latest.get("ret_1d")) else None,
        "taker_z_prev": taker_z_prev,
        "oi_z_prev": oi_z_prev,
        "oi_z_24h_max": oi_z_24h_max,         # current (for exit checks during trade)
        "oi_z_24h_max_prev": oi_z_24h_max,     # same value used for entry check
        "eth_ret_1h_prev": eth_ret_1h_prev,
    }


# ==================================================================
# ENTRY SIGNAL (hard gates)
# ==================================================================

def check_entry_signal(features: dict, params: dict) -> tuple[bool, list[str]]:
    reasons = []

    # G0: ETH momentum (H3)
    eth_ret = features.get("eth_ret_1h_prev")
    eth_min = params["gates"]["g0_eth_regime"]["eth_ret_1h_min"]
    if eth_ret is None or eth_ret <= eth_min:
        reasons.append(f"eth_bearish ({eth_ret})")

    # G1: taker_z (H1, Cohen's d=0.51)
    taker_z = features.get("taker_z_prev")
    taker_min = params["gates"]["g1_taker"]["taker_z_4h_min"]
    if taker_z is None or taker_z < taker_min:
        reasons.append(f"taker_weak ({taker_z})")

    # G2: OI 24h block (H1 bipolar)
    oi24 = features.get("oi_z_24h_max_prev")
    oi_block = params["gates"]["g2_oi_bipolar"]["oi_z_24h_block"]
    if oi24 is not None and oi24 >= oi_block:
        reasons.append(f"oi_extreme ({oi24:.2f})")

    # Filters
    ret_1d = features.get("ret_1d")
    if ret_1d is None or ret_1d <= params["filters"]["ret_1d_min"]:
        reasons.append(f"no_momentum (ret_1d={ret_1d})")

    rsi = features.get("rsi")
    if rsi is None:
        reasons.append("rsi_missing")
    elif rsi < params["filters"]["rsi_min"]:
        reasons.append(f"rsi_low ({rsi:.1f})")
    elif rsi > params["filters"]["rsi_max"]:
        reasons.append(f"rsi_high ({rsi:.1f})")

    close = features.get("close", 0)
    ma21 = features.get("ma21")
    if params["filters"].get("close_above_ma21", True) and (ma21 is None or close < ma21):
        reasons.append(f"below_ma21 ({close:.2f}<{ma21})")

    return len(reasons) == 0, reasons


# ==================================================================
# STOPS CHECK
# ==================================================================

def check_stops(current_price: float, high: float, low: float,
                portfolio: dict, params: dict) -> dict | None:
    """Returns exit dict or None."""
    stops = params["stops"]
    trail_pct = stops["trailing_pct"]
    max_hold_h = stops["max_hold_hours"]

    tp = portfolio["take_profit_price"]
    sl = portfolio["stop_loss_price"]
    trail_high = portfolio.get("trailing_high", portfolio["entry_price"])
    max_hold_until = pd.to_datetime(portfolio.get("max_hold_until"), utc=True) if portfolio.get("max_hold_until") else None

    # Update trailing high
    if current_price > trail_high:
        portfolio["trailing_high"] = round(current_price, 4)
        trail_high = current_price

    trailing_sl = round(trail_high * (1 - trail_pct), 4)
    effective_sl = max(sl, trailing_sl)

    # TP
    if high >= tp:
        return {"reason": "TP", "exit_price": tp}
    # SL / trailing
    if low <= effective_sl:
        reason = "TRAIL" if effective_sl > sl else "SL"
        return {"reason": reason, "exit_price": effective_sl}
    # Max hold
    if max_hold_until and pd.Timestamp.now("UTC") >= max_hold_until:
        return {"reason": "MAX_HOLD", "exit_price": current_price}

    return None


def check_oi_early_exit(features: dict, portfolio: dict, params: dict) -> dict | None:
    """OI 24h reversion early exit — only after min_hours held."""
    oi_cfg = params.get("early_exits", {}).get("oi_24h_reversion", {})
    if not oi_cfg.get("enabled", True):
        return None
    entry_ts = pd.to_datetime(portfolio.get("entry_timestamp"), utc=True)
    if entry_ts is None:
        return None
    hours_held = (pd.Timestamp.now("UTC") - entry_ts).total_seconds() / 3600
    if hours_held < oi_cfg.get("min_hours_since_entry", 12):
        return None
    oi24 = features.get("oi_z_24h_max")
    if oi24 is not None and oi24 > oi_cfg.get("oi_z_threshold", 2.0):
        return {"reason": "OI_EARLY_EXIT", "exit_price": features["close"]}
    return None


# ==================================================================
# POSITION MANAGEMENT
# ==================================================================

def execute_entry(price: float, features: dict, portfolio: dict, params: dict):
    stops = params["stops"]
    sl_pct = stops["stop_loss_pct"]
    tp_pct = stops["take_profit_pct"]
    cooldown_h = params.get("cooldown_hours", 4)

    qty = portfolio["capital_usd"] / price
    now = datetime.now(timezone.utc)

    portfolio.update({
        "has_position": True,
        "entry_price": round(price, 4),
        "quantity": round(qty, 6),
        "stop_loss_price": round(price * (1 - sl_pct), 4),
        "take_profit_price": round(price * (1 + tp_pct), 4),
        "trailing_high": round(price, 4),
        "entry_timestamp": now.isoformat(),
        "max_hold_until": (now + timedelta(hours=stops["max_hold_hours"])).isoformat(),
        "entry_features": {k: round(v, 6) if isinstance(v, float) else v
                           for k, v in features.items() if k != "timestamp"},
    })
    save_portfolio(portfolio)
    logger.info(
        f"SOL4 ENTRY: price=${price:,.3f} qty={qty:.4f} "
        f"SL=${portfolio['stop_loss_price']:,.3f} TP=${portfolio['take_profit_price']:,.3f} "
        f"taker_z={features.get('taker_z_prev')}"
    )


def execute_exit(exit_price: float, exit_reason: str, portfolio: dict, params: dict):
    entry_price = portfolio["entry_price"]
    qty = portfolio["quantity"]
    pnl_pct = (exit_price - entry_price) / entry_price
    pnl_usd = (exit_price - entry_price) * qty

    trade = {
        "symbol": "SOLUSDT",
        "entry_price": entry_price,
        "exit_price": round(exit_price, 4),
        "quantity": qty,
        "entry_timestamp": portfolio.get("entry_timestamp"),
        "exit_timestamp": datetime.now(timezone.utc).isoformat(),
        "exit_reason": exit_reason,
        "pnl_pct": round(pnl_pct, 6),
        "pnl_usd": round(pnl_usd, 2),
        "entry_features": portfolio.get("entry_features", {}),
    }

    trades = []
    if TRADES_PATH.exists():
        with open(TRADES_PATH) as f:
            trades = json.load(f)
    trades.append(trade)
    TRADES_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(TRADES_PATH, "w") as f:
        json.dump(trades, f, indent=2, default=str)

    portfolio["capital_usd"] = round(portfolio["capital_usd"] + pnl_usd, 2)
    for key in ["has_position", "entry_price", "quantity", "stop_loss_price",
                "take_profit_price", "trailing_high", "entry_timestamp",
                "max_hold_until", "entry_features"]:
        portfolio[key] = None if key != "has_position" else False
    save_portfolio(portfolio)
    logger.info(
        f"SOL4 EXIT ({exit_reason}): price=${exit_price:,.3f} "
        f"return={pnl_pct:+.2%} pnl=${pnl_usd:+.2f} capital=${portfolio['capital_usd']:,.2f}"
    )


# ==================================================================
# SHADOW SCORING
# ==================================================================

def log_shadow_scoring(features: dict, is_hard_gate_entry: bool):
    """
    Log shadow scoring hypothesis for future comparison.
    Scoring v1: taker(+2) + eth(+1) + oi_guard(+1) → entry if >= 3.
    """
    score = 0
    breakdown = {}
    taker_z = features.get("taker_z_prev") or 0
    eth_ret = features.get("eth_ret_1h_prev") or 0
    oi24 = features.get("oi_z_24h_max_prev")

    if taker_z > 0.3:
        score += 2
        breakdown["taker"] = 2
    else:
        breakdown["taker"] = 0

    if eth_ret > 0:
        score += 1
        breakdown["eth"] = 1
    else:
        breakdown["eth"] = 0

    if oi24 is None or oi24 < 2.0:
        score += 1
        breakdown["oi_guard"] = 1
    else:
        breakdown["oi_guard"] = 0

    entry = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "bot_origin": "sol_bot4",
        "filter_version": "scoring_v1",
        "hard_gate_entry": is_hard_gate_entry,
        "score_total": score,
        "score_threshold": 3,
        "scoring_would_enter": score >= 3,
        "breakdown": breakdown,
        "taker_z_prev": taker_z,
        "eth_ret_1h_prev": eth_ret,
        "oi_z_24h": oi24,
    }

    SHADOW_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(SHADOW_LOG_PATH, "a") as f:
        f.write(json.dumps(entry, default=str) + "\n")


# ==================================================================
# MAIN CYCLES
# ==================================================================

def run_hourly_cycle():
    """Full hourly cycle: feature compute → entry/exit decision."""
    if not acquire_lock():
        logger.warning("SOL cycle already running, skipping")
        return

    try:
        params = get_params()
        if not params.get("enabled", True):
            logger.info("SOL Bot 4 disabled in parameters_sol.yml")
            return

        portfolio = load_portfolio()

        # Compute features (uses parquet files — no live price needed here)
        try:
            features = compute_sol_features()
        except Exception as e:
            logger.error(f"SOL feature computation failed: {e}", exc_info=True)
            return

        current_price = get_live_price()
        if current_price is None:
            # Fall back to last close
            current_price = features["close"]
            logger.warning(f"Using last close as live price: ${current_price}")

        # === Position open: check exits ===
        if portfolio.get("has_position"):
            # Refresh features with current price for stops
            features["close"] = current_price

            exit_info = check_stops(
                current_price, features.get("high", current_price),
                features.get("low", current_price), portfolio, params
            )
            if exit_info is None:
                exit_info = check_oi_early_exit(features, portfolio, params)

            if exit_info:
                execute_exit(exit_info["exit_price"], exit_info["reason"], portfolio, params)
            else:
                save_portfolio(portfolio)
                entry_price = portfolio["entry_price"]
                ret = (current_price - entry_price) / entry_price
                logger.info(
                    f"SOL4 HOLD: ${current_price:,.3f} "
                    f"entry=${entry_price:,.3f} ret={ret:+.2%} "
                    f"trail_high=${portfolio.get('trailing_high', 0):,.3f}"
                )
            return

        # === No position: check entry ===
        allowed, reasons = check_entry_signal(features, params)

        # Always log shadow scoring (even on blocks)
        log_shadow_scoring(features, allowed)

        if allowed:
            execute_entry(current_price, features, portfolio, params)
        else:
            logger.info(f"SOL4 BLOCK: {', '.join(reasons[:3])}")

    finally:
        release_lock()


def check_stops_only():
    """Lightweight stops check (runs every 15 min via cron)."""
    if not acquire_lock():
        logger.info("SOL cycle running, skipping stops check")
        return

    try:
        params = get_params()
        portfolio = load_portfolio()

        if not portfolio.get("has_position"):
            return

        current_price = get_live_price()
        if current_price is None:
            logger.warning("SOL stops check: no live price")
            return

        try:
            features = compute_sol_features()
            features["close"] = current_price
        except Exception:
            # Minimal fallback for stops check
            features = {"close": current_price,
                        "high": current_price, "low": current_price,
                        "oi_z_24h_max": None}

        exit_info = check_stops(
            current_price, current_price, current_price, portfolio, params
        )
        if exit_info is None:
            exit_info = check_oi_early_exit(features, portfolio, params)

        if exit_info:
            execute_exit(exit_info["exit_price"], exit_info["reason"], portfolio, params)
        else:
            save_portfolio(portfolio)
            logger.info(
                f"SOL4 [STOPS] HOLD: ${current_price:,.3f} "
                f"trail_high=${portfolio.get('trailing_high', 0):,.3f}"
            )
    finally:
        release_lock()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
    run_hourly_cycle()
