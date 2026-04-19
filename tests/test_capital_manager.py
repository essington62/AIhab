"""
Tests for src/trading/capital_manager.py

Coverage:
  - init_buckets: creates correct bucket structure
  - cm_can_enter: disabled / ok / paused / max_dd / daily_loss
  - sync_capital_for_entry: sets correct capital on root
  - sync_entry_to_bucket: copies entry fields
  - sync_exit_to_bucket: updates bucket capital, PnL, total
  - check_and_pause_if_needed: sets paused_until on DD / daily loss breach
  - reset_daily_counters_if_needed: resets daily_pnl on new day
  - bot_to_bucket_key: mapping helper
"""
import pytest
import pandas as pd

from src.trading.capital_manager import (
    bot_to_bucket_key,
    check_and_pause_if_needed,
    cm_can_enter,
    init_buckets,
    reset_daily_counters_if_needed,
    sync_capital_for_entry,
    sync_entry_to_bucket,
    sync_exit_to_bucket,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _base_params(enabled=True) -> dict:
    return {
        "capital_management": {
            "enabled": enabled,
            "buckets": {
                "btc_bot1": {"name": "Bot 1 (Reversal)", "allocation_pct": 0.5},
                "btc_bot2": {"name": "Bot 2 (Momentum)", "allocation_pct": 0.5},
            },
            "safety": {
                "max_drawdown_pct": 0.15,
                "max_daily_loss_pct": 0.05,
                "pause_hours_after_dd": 72,
                "pause_hours_after_daily_loss": 24,
            },
        },
        "execution": {"paper_capital_usd": 10000.0},
    }


def _portfolio_with_capital(capital=10000.0) -> dict:
    return {"capital_usd": capital, "has_position": False}


# ---------------------------------------------------------------------------
# bot_to_bucket_key
# ---------------------------------------------------------------------------

def test_bot_to_bucket_key_bot1():
    assert bot_to_bucket_key("bot1") == "btc_bot1"


def test_bot_to_bucket_key_bot2():
    assert bot_to_bucket_key("bot2") == "btc_bot2"


# ---------------------------------------------------------------------------
# init_buckets
# ---------------------------------------------------------------------------

def test_init_buckets_creates_structure():
    portfolio = _portfolio_with_capital(10000.0)
    params = _base_params()
    init_buckets(portfolio, params)
    assert "buckets" in portfolio
    assert "btc_bot1" in portfolio["buckets"]
    assert "btc_bot2" in portfolio["buckets"]


def test_init_buckets_50_50_split():
    portfolio = _portfolio_with_capital(10000.0)
    params = _base_params()
    init_buckets(portfolio, params)
    assert portfolio["buckets"]["btc_bot1"]["current_capital"] == 5000.0
    assert portfolio["buckets"]["btc_bot2"]["current_capital"] == 5000.0


def test_init_buckets_total_capital_usd():
    portfolio = _portfolio_with_capital(10000.0)
    init_buckets(portfolio, _base_params())
    assert portfolio["total_capital_usd"] == 10000.0


def test_init_buckets_idempotent():
    portfolio = _portfolio_with_capital(10000.0)
    params = _base_params()
    init_buckets(portfolio, params)
    portfolio["buckets"]["btc_bot1"]["current_capital"] = 4000.0
    init_buckets(portfolio, params)  # second call — should not overwrite
    assert portfolio["buckets"]["btc_bot1"]["current_capital"] == 4000.0


def test_init_buckets_disabled_noop():
    portfolio = _portfolio_with_capital(10000.0)
    params = _base_params(enabled=False)
    init_buckets(portfolio, params)
    assert "buckets" not in portfolio


# ---------------------------------------------------------------------------
# cm_can_enter
# ---------------------------------------------------------------------------

def _portfolio_with_buckets(b1_cap=5000.0, b2_cap=5000.0) -> dict:
    return {
        "capital_usd": b1_cap + b2_cap,
        "total_capital_usd": b1_cap + b2_cap,
        "has_position": False,
        "buckets": {
            "btc_bot1": {
                "current_capital": b1_cap,
                "initial_capital": 5000.0,
                "peak_capital": 5000.0,
                "daily_pnl": 0.0,
                "daily_capital_base": 5000.0,
                "paused_until": None,
                "pause_reason": None,
            },
            "btc_bot2": {
                "current_capital": b2_cap,
                "initial_capital": 5000.0,
                "peak_capital": 5000.0,
                "daily_pnl": 0.0,
                "daily_capital_base": 5000.0,
                "paused_until": None,
                "pause_reason": None,
            },
        },
    }


def test_cm_can_enter_disabled():
    portfolio = _portfolio_with_buckets()
    params = _base_params(enabled=False)
    result = cm_can_enter(portfolio, "btc_bot1", params)
    assert result["can_enter"] is True
    assert result["reason"] == "cm_disabled"


def test_cm_can_enter_ok():
    portfolio = _portfolio_with_buckets()
    result = cm_can_enter(portfolio, "btc_bot1", _base_params())
    assert result["can_enter"] is True
    assert result["reason"] == "ok"


def test_cm_can_enter_bucket_not_found():
    portfolio = _portfolio_with_buckets()
    result = cm_can_enter(portfolio, "btc_bot3", _base_params())
    assert result["can_enter"] is False
    assert "not_found" in result["reason"]


def test_cm_can_enter_paused():
    portfolio = _portfolio_with_buckets()
    future = str(pd.Timestamp.now("UTC") + pd.Timedelta(hours=10))
    portfolio["buckets"]["btc_bot1"]["paused_until"] = future
    portfolio["buckets"]["btc_bot1"]["pause_reason"] = "test pause"
    result = cm_can_enter(portfolio, "btc_bot1", _base_params())
    assert result["can_enter"] is False
    assert "PAUSED" in result["reason"]


def test_cm_can_enter_expired_pause():
    portfolio = _portfolio_with_buckets()
    past = str(pd.Timestamp.now("UTC") - pd.Timedelta(hours=1))
    portfolio["buckets"]["btc_bot1"]["paused_until"] = past
    result = cm_can_enter(portfolio, "btc_bot1", _base_params())
    assert result["can_enter"] is True


def test_cm_can_enter_max_dd():
    # initial=5000, current=4100 → DD=18% > 15%
    portfolio = _portfolio_with_buckets(b1_cap=4100.0)
    result = cm_can_enter(portfolio, "btc_bot1", _base_params())
    assert result["can_enter"] is False
    assert "MAX_DD_REACHED" in result["reason"]


def test_cm_can_enter_daily_loss():
    portfolio = _portfolio_with_buckets()
    portfolio["buckets"]["btc_bot1"]["daily_pnl"] = -260.0  # 5.2% of 5000
    result = cm_can_enter(portfolio, "btc_bot1", _base_params())
    assert result["can_enter"] is False
    assert "DAILY_LOSS_LIMIT" in result["reason"]


# ---------------------------------------------------------------------------
# sync helpers
# ---------------------------------------------------------------------------

def test_sync_capital_for_entry():
    portfolio = _portfolio_with_buckets(b1_cap=4800.0)
    sync_capital_for_entry(portfolio, "btc_bot1")
    assert portfolio["capital_usd"] == 4800.0


def test_sync_entry_to_bucket():
    portfolio = _portfolio_with_buckets()
    portfolio.update({
        "has_position": True,
        "entry_price": 85000.0,
        "entry_time": "2026-04-19T12:00:00",
        "quantity": 0.058,
        "trailing_high": 85000.0,
        "stop_loss_price": 82450.0,
        "take_profit_price": 86700.0,
        "trailing_stop_pct_actual": 0.01,
        "stops_mode": "dynamic",
        "entry_atr_pct": 0.015,
    })
    sync_entry_to_bucket(portfolio, "btc_bot1")
    bucket = portfolio["buckets"]["btc_bot1"]
    assert bucket["has_position"] is True
    assert bucket["entry_price"] == 85000.0
    assert bucket["quantity"] == 0.058


def test_sync_exit_to_bucket_updates_capital():
    portfolio = _portfolio_with_buckets(b1_cap=5000.0, b2_cap=5000.0)
    portfolio["capital_usd"] = 5100.0  # after profitable trade
    sync_exit_to_bucket(portfolio, "btc_bot1")
    assert portfolio["buckets"]["btc_bot1"]["current_capital"] == 5100.0
    assert portfolio["total_capital_usd"] == 10100.0


def test_sync_exit_to_bucket_updates_daily_pnl():
    portfolio = _portfolio_with_buckets(b1_cap=5000.0, b2_cap=5000.0)
    portfolio["buckets"]["btc_bot1"]["daily_pnl"] = 50.0
    portfolio["capital_usd"] = 5100.0
    sync_exit_to_bucket(portfolio, "btc_bot1")
    assert portfolio["buckets"]["btc_bot1"]["daily_pnl"] == 150.0  # 50 + 100


def test_sync_exit_clears_position_fields():
    portfolio = _portfolio_with_buckets()
    portfolio["buckets"]["btc_bot1"]["has_position"] = True
    portfolio["buckets"]["btc_bot1"]["entry_price"] = 80000.0
    portfolio["capital_usd"] = 5000.0
    sync_exit_to_bucket(portfolio, "btc_bot1")
    assert portfolio["buckets"]["btc_bot1"]["has_position"] is False
    assert portfolio["buckets"]["btc_bot1"]["entry_price"] is None


# ---------------------------------------------------------------------------
# check_and_pause_if_needed
# ---------------------------------------------------------------------------

def test_pause_on_max_dd():
    portfolio = _portfolio_with_buckets(b1_cap=4200.0)  # 16% DD
    check_and_pause_if_needed(portfolio, "btc_bot1", _base_params())
    bucket = portfolio["buckets"]["btc_bot1"]
    assert bucket["paused_until"] is not None
    assert "MAX_DD" in bucket["pause_reason"]


def test_pause_on_daily_loss():
    portfolio = _portfolio_with_buckets()
    portfolio["buckets"]["btc_bot1"]["daily_pnl"] = -300.0  # 6% of 5000
    check_and_pause_if_needed(portfolio, "btc_bot1", _base_params())
    bucket = portfolio["buckets"]["btc_bot1"]
    assert bucket["paused_until"] is not None
    assert "DAILY_LOSS" in bucket["pause_reason"]


def test_no_pause_when_ok():
    portfolio = _portfolio_with_buckets()
    check_and_pause_if_needed(portfolio, "btc_bot1", _base_params())
    assert portfolio["buckets"]["btc_bot1"]["paused_until"] is None


# ---------------------------------------------------------------------------
# reset_daily_counters_if_needed
# ---------------------------------------------------------------------------

def test_reset_daily_counters_on_new_day():
    portfolio = _portfolio_with_buckets()
    portfolio["buckets"]["btc_bot1"]["daily_pnl"] = -100.0
    portfolio["buckets"]["btc_bot1"]["daily_pnl_date"] = "2020-01-01"  # stale
    changed = reset_daily_counters_if_needed(portfolio)
    assert changed is True
    assert portfolio["buckets"]["btc_bot1"]["daily_pnl"] == 0.0


def test_reset_daily_counters_same_day_noop():
    today = str(pd.Timestamp.now("UTC").date())
    portfolio = _portfolio_with_buckets()
    portfolio["buckets"]["btc_bot1"]["daily_pnl_date"] = today
    portfolio["buckets"]["btc_bot2"]["daily_pnl_date"] = today
    changed = reset_daily_counters_if_needed(portfolio)
    assert changed is False


def test_reset_clears_expired_pause():
    portfolio = _portfolio_with_buckets()
    past = str(pd.Timestamp.now("UTC") - pd.Timedelta(hours=1))
    portfolio["buckets"]["btc_bot1"]["paused_until"] = past
    portfolio["buckets"]["btc_bot1"]["pause_reason"] = "test"
    portfolio["buckets"]["btc_bot1"]["daily_pnl_date"] = "2020-01-01"
    reset_daily_counters_if_needed(portfolio)
    assert portfolio["buckets"]["btc_bot1"]["paused_until"] is None
