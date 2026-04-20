"""
ETH Bot 3 — Volume-based strategy (Phase 1).

Lógica:
  - BLOCK: volume_z > +1.5 → NUNCA entra (gate de bloqueio)
  - ENTRY: volume_z em Q2 (-0.75 a -0.30) + RSI < 60 + close > MA200
  - EXIT: SL -2% / TP +4% / trailing 1.5% / timeout 168h

Isolado do BTC Bot 1/2. Portfolio, state, logs separados.
"""
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

logger = logging.getLogger("eth_bot3")

# Paths
PARAMS_PATH = ROOT / "conf/parameters_eth.yml"
PORTFOLIO_PATH = ROOT / "data/04_scoring/portfolio_eth.json"
TRADES_PATH = ROOT / "data/05_trades/completed_trades_eth.json"
SPOT_PATH = ROOT / "data/01_raw/spot/eth_1h.parquet"
LOCK_PATH = ROOT / "data/04_scoring/eth_cycle.lock"

for p in [PORTFOLIO_PATH.parent, TRADES_PATH.parent]:
    p.mkdir(parents=True, exist_ok=True)


def get_params() -> dict:
    with open(PARAMS_PATH) as f:
        return yaml.safe_load(f)


def load_portfolio() -> dict:
    if not PORTFOLIO_PATH.exists():
        params = get_params()
        initial = {
            "capital_usd": params["execution"]["initial_capital_usd"],
            "has_position": False,
            "entry_price": None,
            "quantity": None,
            "stop_loss_price": None,
            "take_profit_price": None,
            "trailing_high": None,
            "entry_timestamp": None,
            "entry_volume_z": None,
            "last_update": datetime.now(timezone.utc).isoformat(),
        }
        save_portfolio(initial)
        return initial

    with open(PORTFOLIO_PATH) as f:
        return json.load(f)


def save_portfolio(portfolio: dict):
    portfolio["last_update"] = datetime.now(timezone.utc).isoformat()
    with open(PORTFOLIO_PATH, "w") as f:
        json.dump(portfolio, f, indent=2, default=str)


def acquire_lock() -> bool:
    """ETH-specific lock (doesn't conflict with BTC)."""
    if LOCK_PATH.exists():
        try:
            mtime = LOCK_PATH.stat().st_mtime
            age = datetime.now(timezone.utc).timestamp() - mtime
            if age < 300:  # 5 min
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


def get_live_price(symbol: str = "ETHUSDT", timeout: float = 5.0):
    """Binance public API — same pattern as BTC fix."""
    try:
        r = requests.get(
            "https://api.binance.com/api/v3/ticker/price",
            params={"symbol": symbol},
            timeout=timeout,
        )
        r.raise_for_status()
        return float(r.json()["price"])
    except Exception as e:
        logger.warning(f"get_live_price({symbol}) failed: {e}")
        return None


def compute_eth_features() -> dict:
    """Compute features needed for ETH bot decision."""
    if not SPOT_PATH.exists():
        raise FileNotFoundError(f"ETH spot data missing: {SPOT_PATH}")

    df = pd.read_parquet(SPOT_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Resample to daily (we operate on daily signals but exec hourly)
    df_d = df.set_index("timestamp").resample("D").agg({
        "close": "last",
        "volume": "sum",
        "high": "max",
        "low": "min",
    }).dropna().reset_index()

    params = get_params()
    win = params["zscore"]["volume_window"]
    min_history = params["zscore"]["min_history"]

    if len(df_d) < min_history:
        raise ValueError(f"Insufficient ETH history: {len(df_d)} days < {min_history} required")

    # Volume z-score (rolling 30d)
    df_d["volume_z"] = (
        (df_d["volume"] - df_d["volume"].rolling(win).mean()) /
        df_d["volume"].rolling(win).std()
    )

    # RSI 14d (daily)
    delta = df_d["close"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df_d["rsi_14"] = 100 - (100 / (1 + rs))

    # MA200 (daily)
    df_d["ma_200"] = df_d["close"].rolling(200).mean()

    latest = df_d.iloc[-1]

    return {
        "close_daily": float(latest["close"]),
        "volume_z": float(latest["volume_z"]) if pd.notna(latest["volume_z"]) else None,
        "rsi_14": float(latest["rsi_14"]) if pd.notna(latest["rsi_14"]) else None,
        "ma_200": float(latest["ma_200"]) if pd.notna(latest["ma_200"]) else None,
        "above_ma200": bool(latest["close"] > latest["ma_200"]) if pd.notna(latest["ma_200"]) else False,
        "timestamp": latest["timestamp"],
    }


def check_block_rule(features: dict, params: dict) -> dict:
    """Aplica BLOCK rule: volume_z > +1.5 → nunca entra."""
    vol_z = features.get("volume_z")
    threshold = params["block_rules"]["volume_z_block"]

    if vol_z is None or (isinstance(vol_z, float) and pd.isna(vol_z)):
        return {"blocked": False, "reason": "volume_z_missing"}

    if vol_z > threshold:
        return {
            "blocked": True,
            "reason": f"HIGH_VOLUME_SPIKE (vol_z={vol_z:.2f} > {threshold})",
            "volume_z": vol_z,
        }

    return {"blocked": False, "reason": "", "volume_z": vol_z}


def check_entry_rule(features: dict, params: dict) -> dict:
    """Entry: volume_z em Q2 + filtros de sanidade."""
    rule = params["entry_rule"]

    vol_z = features.get("volume_z")
    rsi = features.get("rsi_14")
    above_ma200 = features.get("above_ma200", False)

    result = {
        "volume_z": vol_z,
        "rsi_14": rsi,
        "above_ma200": above_ma200,
        "passed": False,
        "reason": "",
    }

    if vol_z is None or rsi is None:
        result["reason"] = "MISSING_DATA"
        return result

    # Check volume in Q2 range
    if not (rule["volume_z_min"] < vol_z < rule["volume_z_max"]):
        result["reason"] = (
            f"VOLUME_OUT_OF_Q2 (vol_z={vol_z:.2f} not in "
            f"({rule['volume_z_min']}, {rule['volume_z_max']}))"
        )
        return result

    # RSI not overbought
    if rsi >= rule["rsi_max"]:
        result["reason"] = f"RSI_HIGH ({rsi:.1f} >= {rule['rsi_max']})"
        return result

    # Price above MA200
    if rule["price_above_ma200"] and not above_ma200:
        result["reason"] = "BELOW_MA200"
        return result

    result["passed"] = True
    result["reason"] = "Q2_VOLUME_MATCH"
    return result


def execute_entry(current_price: float, portfolio: dict, features: dict, params: dict) -> dict:
    exec_cfg = params["execution"]
    capital = portfolio["capital_usd"]
    size_pct = exec_cfg["position_size_pct"]

    position_value = capital * size_pct
    quantity = position_value / current_price

    sl_pct = exec_cfg["stop_loss_pct"]
    tp_pct = exec_cfg["take_profit_pct"]

    portfolio["has_position"] = True
    portfolio["entry_price"] = round(current_price, 2)
    portfolio["quantity"] = round(quantity, 6)
    portfolio["stop_loss_price"] = round(current_price * (1 - sl_pct), 2)
    portfolio["take_profit_price"] = round(current_price * (1 + tp_pct), 2)
    portfolio["trailing_high"] = round(current_price, 2)
    portfolio["entry_timestamp"] = datetime.now(timezone.utc).isoformat()
    portfolio["entry_volume_z"] = features.get("volume_z")
    portfolio["entry_rsi"] = features.get("rsi_14")

    save_portfolio(portfolio)

    logger.info(
        f"ETH3 ENTRY: price=${current_price:,.2f} qty={quantity:.4f} "
        f"SL=${portfolio['stop_loss_price']:,.2f} TP=${portfolio['take_profit_price']:,.2f} "
        f"vol_z={features.get('volume_z'):.2f}"
    )
    return portfolio


def check_stops(current_price: float, portfolio: dict, params: dict) -> dict:
    """Check SL/TP/trailing/timeout. Returns exit decision."""
    exec_cfg = params["execution"]
    trail_pct = exec_cfg["trailing_stop_pct"]
    max_hold = exec_cfg["max_hold_hours"]

    # Update trailing high
    if current_price > portfolio["trailing_high"]:
        portfolio["trailing_high"] = round(current_price, 2)

    exit_reason = None
    exit_price = current_price

    # SL
    if current_price <= portfolio["stop_loss_price"]:
        exit_reason = "SL"
        exit_price = portfolio["stop_loss_price"]

    # TP
    elif current_price >= portfolio["take_profit_price"]:
        exit_reason = "TP"
        exit_price = portfolio["take_profit_price"]

    # Trailing (only if in profit)
    elif current_price > portfolio["entry_price"]:
        trailing_stop = portfolio["trailing_high"] * (1 - trail_pct)
        if current_price <= trailing_stop:
            exit_reason = "TRAIL"
            exit_price = max(trailing_stop, portfolio["entry_price"])

    # Timeout
    if exit_reason is None and portfolio.get("entry_timestamp"):
        entry_ts = pd.Timestamp(portfolio["entry_timestamp"])
        now = pd.Timestamp.now("UTC")
        hold_hours = (now - entry_ts).total_seconds() / 3600
        if hold_hours >= max_hold:
            exit_reason = "TIMEOUT"
            exit_price = current_price

    if exit_reason:
        return {
            "exit": True,
            "reason": exit_reason,
            "price": exit_price,
            "return_pct": (exit_price - portfolio["entry_price"]) / portfolio["entry_price"],
        }

    return {"exit": False, "trailing_high": portfolio["trailing_high"]}


def execute_exit(exit_price: float, exit_reason: str, portfolio: dict) -> dict:
    entry_price = portfolio["entry_price"]
    quantity = portfolio["quantity"]

    pnl_pct = (exit_price - entry_price) / entry_price
    pnl_usd = (exit_price - entry_price) * quantity

    portfolio["capital_usd"] = portfolio["capital_usd"] + pnl_usd

    trade = {
        "symbol": "ETHUSDT",
        "entry_price": entry_price,
        "exit_price": exit_price,
        "quantity": quantity,
        "entry_timestamp": portfolio.get("entry_timestamp"),
        "exit_timestamp": datetime.now(timezone.utc).isoformat(),
        "exit_reason": exit_reason,
        "pnl_pct": round(pnl_pct, 4),
        "pnl_usd": round(pnl_usd, 2),
        "entry_volume_z": portfolio.get("entry_volume_z"),
        "entry_rsi": portfolio.get("entry_rsi"),
    }

    trades = []
    if TRADES_PATH.exists():
        with open(TRADES_PATH) as f:
            trades = json.load(f)
    trades.append(trade)
    with open(TRADES_PATH, "w") as f:
        json.dump(trades, f, indent=2, default=str)

    # Reset portfolio
    portfolio["has_position"] = False
    portfolio["entry_price"] = None
    portfolio["quantity"] = None
    portfolio["stop_loss_price"] = None
    portfolio["take_profit_price"] = None
    portfolio["trailing_high"] = None
    portfolio["entry_timestamp"] = None
    portfolio["entry_volume_z"] = None
    portfolio["entry_rsi"] = None

    save_portfolio(portfolio)

    logger.info(
        f"ETH3 EXIT by {exit_reason}: price=${exit_price:,.2f} "
        f"return={pnl_pct:+.2%} pnl=${pnl_usd:+.2f}"
    )
    return portfolio


def run_hourly_cycle():
    """Main cycle — runs every hour via cron."""
    if not acquire_lock():
        logger.warning("Another ETH cycle running, skipping")
        return

    try:
        params = get_params()
        portfolio = load_portfolio()

        current_price = get_live_price("ETHUSDT")
        if current_price is None:
            logger.warning("Could not get live ETH price")
            return

        # If position open, check stops only
        if portfolio.get("has_position"):
            stop_result = check_stops(current_price, portfolio, params)
            if stop_result["exit"]:
                execute_exit(stop_result["price"], stop_result["reason"], portfolio)
            else:
                save_portfolio(portfolio)
                logger.info(
                    f"ETH3 HOLD: price=${current_price:,.2f} "
                    f"entry=${portfolio['entry_price']:,.2f} "
                    f"ret={(current_price - portfolio['entry_price']) / portfolio['entry_price']:+.2%}"
                )
            return

        # Compute features
        features = compute_eth_features()

        # BLOCK rule first
        block = check_block_rule(features, params)
        if block["blocked"]:
            logger.info(f"ETH3 BLOCKED: {block['reason']}")
            return

        # Entry rule
        entry = check_entry_rule(features, params)
        if entry["passed"]:
            execute_entry(current_price, portfolio, features, params)
        else:
            logger.info(f"ETH3 WAIT: {entry['reason']}")

    finally:
        release_lock()


def check_stops_only():
    """Lightweight stops check (runs every 15min via cron)."""
    if not acquire_lock():
        logger.info("Another ETH cycle running, skip stops check")
        return

    try:
        params = get_params()
        portfolio = load_portfolio()

        if not portfolio.get("has_position"):
            return

        current_price = get_live_price("ETHUSDT")
        if current_price is None:
            return

        stop_result = check_stops(current_price, portfolio, params)
        if stop_result["exit"]:
            execute_exit(stop_result["price"], stop_result["reason"], portfolio)
        else:
            save_portfolio(portfolio)
            logger.info(
                f"ETH3 [STOPS-15m] HOLD: price=${current_price:,.2f} "
                f"trailing_high=${portfolio['trailing_high']:,.2f}"
            )

    finally:
        release_lock()
