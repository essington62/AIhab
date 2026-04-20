"""Tests for ETH Bot 3 logic."""
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.trading.eth_bot3 import (
    check_block_rule,
    check_entry_rule,
    check_stops,
    execute_entry,
    execute_exit,
)

BASE_PARAMS = {
    "block_rules": {
        "enabled": True,
        "volume_z_block": 1.5,
    },
    "entry_rule": {
        "enabled": True,
        "volume_z_min": -0.75,
        "volume_z_max": -0.30,
        "rsi_max": 60,
        "price_above_ma200": True,
    },
    "execution": {
        "symbol": "ETHUSDT",
        "initial_capital_usd": 10000,
        "position_size_pct": 1.0,
        "stop_loss_pct": 0.02,
        "take_profit_pct": 0.04,
        "trailing_stop_pct": 0.015,
        "max_hold_hours": 168,
    },
}

BASE_PORTFOLIO = {
    "capital_usd": 10000.0,
    "has_position": True,
    "entry_price": 1800.0,
    "quantity": 5.555556,
    "stop_loss_price": 1764.0,
    "take_profit_price": 1872.0,
    "trailing_high": 1800.0,
    "entry_timestamp": "2026-04-20T10:00:00+00:00",
    "entry_volume_z": -0.55,
    "entry_rsi": 45.0,
    "last_update": "2026-04-20T10:00:00+00:00",
}


# ===========================================================
# TestBlockRule
# ===========================================================

class TestBlockRule:
    def test_high_volume_blocks(self):
        features = {"volume_z": 2.0}
        r = check_block_rule(features, BASE_PARAMS)
        assert r["blocked"] is True
        assert "HIGH_VOLUME_SPIKE" in r["reason"]

    def test_at_threshold_blocks(self):
        features = {"volume_z": 1.5}
        r = check_block_rule(features, BASE_PARAMS)
        assert r["blocked"] is False  # strictly greater

    def test_just_above_threshold_blocks(self):
        features = {"volume_z": 1.51}
        r = check_block_rule(features, BASE_PARAMS)
        assert r["blocked"] is True

    def test_normal_volume_passes(self):
        features = {"volume_z": 0.5}
        r = check_block_rule(features, BASE_PARAMS)
        assert r["blocked"] is False

    def test_negative_volume_passes(self):
        features = {"volume_z": -1.0}
        r = check_block_rule(features, BASE_PARAMS)
        assert r["blocked"] is False

    def test_missing_volume_does_not_block(self):
        features = {}
        r = check_block_rule(features, BASE_PARAMS)
        assert r["blocked"] is False
        assert r["reason"] == "volume_z_missing"

    def test_none_volume_does_not_block(self):
        features = {"volume_z": None}
        r = check_block_rule(features, BASE_PARAMS)
        assert r["blocked"] is False


# ===========================================================
# TestEntryRule
# ===========================================================

class TestEntryRule:
    def test_valid_entry(self):
        features = {
            "volume_z": -0.5,
            "rsi_14": 45,
            "above_ma200": True,
        }
        r = check_entry_rule(features, BASE_PARAMS)
        assert r["passed"] is True
        assert r["reason"] == "Q2_VOLUME_MATCH"

    def test_volume_too_low_rejects(self):
        features = {
            "volume_z": -2.0,
            "rsi_14": 45,
            "above_ma200": True,
        }
        r = check_entry_rule(features, BASE_PARAMS)
        assert r["passed"] is False
        assert "VOLUME_OUT_OF_Q2" in r["reason"]

    def test_volume_too_high_rejects(self):
        # Volume above Q2 range
        features = {
            "volume_z": 0.0,
            "rsi_14": 45,
            "above_ma200": True,
        }
        r = check_entry_rule(features, BASE_PARAMS)
        assert r["passed"] is False
        assert "VOLUME_OUT_OF_Q2" in r["reason"]

    def test_rsi_too_high_rejects(self):
        features = {
            "volume_z": -0.5,
            "rsi_14": 70,
            "above_ma200": True,
        }
        r = check_entry_rule(features, BASE_PARAMS)
        assert r["passed"] is False
        assert "RSI_HIGH" in r["reason"]

    def test_rsi_at_threshold_rejects(self):
        features = {
            "volume_z": -0.5,
            "rsi_14": 60,
            "above_ma200": True,
        }
        r = check_entry_rule(features, BASE_PARAMS)
        assert r["passed"] is False

    def test_below_ma200_rejects(self):
        features = {
            "volume_z": -0.5,
            "rsi_14": 45,
            "above_ma200": False,
        }
        r = check_entry_rule(features, BASE_PARAMS)
        assert r["passed"] is False
        assert "BELOW_MA200" in r["reason"]

    def test_missing_volume_rejects(self):
        features = {"rsi_14": 45, "above_ma200": True}
        r = check_entry_rule(features, BASE_PARAMS)
        assert r["passed"] is False
        assert "MISSING_DATA" in r["reason"]

    def test_missing_rsi_rejects(self):
        features = {"volume_z": -0.5, "above_ma200": True}
        r = check_entry_rule(features, BASE_PARAMS)
        assert r["passed"] is False
        assert "MISSING_DATA" in r["reason"]


# ===========================================================
# TestCheckStops
# ===========================================================

class TestCheckStops:
    def test_sl_hit(self):
        portfolio = dict(BASE_PORTFOLIO)
        r = check_stops(1763.0, portfolio, BASE_PARAMS)
        assert r["exit"] is True
        assert r["reason"] == "SL"

    def test_tp_hit(self):
        portfolio = dict(BASE_PORTFOLIO)
        r = check_stops(1873.0, portfolio, BASE_PARAMS)
        assert r["exit"] is True
        assert r["reason"] == "TP"

    def test_hold_no_exit(self):
        portfolio = dict(BASE_PORTFOLIO)
        r = check_stops(1820.0, portfolio, BASE_PARAMS)
        assert r["exit"] is False

    def test_trailing_stop_triggers(self):
        portfolio = dict(BASE_PORTFOLIO)
        portfolio["trailing_high"] = 1900.0  # up from 1800 entry
        # 1900 * (1 - 0.015) = 1871.5; price falls to 1860
        r = check_stops(1860.0, portfolio, BASE_PARAMS)
        assert r["exit"] is True
        assert r["reason"] == "TRAIL"

    def test_trailing_updates_high(self):
        portfolio = dict(BASE_PORTFOLIO)
        portfolio["trailing_high"] = 1800.0
        r = check_stops(1850.0, portfolio, BASE_PARAMS)
        assert r["exit"] is False
        assert portfolio["trailing_high"] == 1850.0

    def test_timeout_triggers(self):
        portfolio = dict(BASE_PORTFOLIO)
        portfolio["entry_timestamp"] = "2026-01-01T00:00:00+00:00"
        r = check_stops(1820.0, portfolio, BASE_PARAMS)
        assert r["exit"] is True
        assert r["reason"] == "TIMEOUT"


# ===========================================================
# TestExecuteEntryExit
# ===========================================================

class TestExecuteEntryExit:
    def test_execute_entry_sets_portfolio(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src.trading.eth_bot3.PORTFOLIO_PATH", tmp_path / "portfolio_eth.json")
        monkeypatch.setattr("src.trading.eth_bot3.TRADES_PATH", tmp_path / "trades.json")

        portfolio = {
            "capital_usd": 10000.0,
            "has_position": False,
        }
        features = {"volume_z": -0.5, "rsi_14": 45.0}

        result = execute_entry(1800.0, portfolio, features, BASE_PARAMS)

        assert result["has_position"] is True
        assert result["entry_price"] == 1800.0
        assert result["stop_loss_price"] == pytest.approx(1764.0, abs=1.0)
        assert result["take_profit_price"] == pytest.approx(1872.0, abs=1.0)
        assert result["entry_volume_z"] == -0.5

    def test_execute_exit_updates_capital(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src.trading.eth_bot3.PORTFOLIO_PATH", tmp_path / "portfolio_eth.json")
        monkeypatch.setattr("src.trading.eth_bot3.TRADES_PATH", tmp_path / "trades.json")

        portfolio = dict(BASE_PORTFOLIO)
        result = execute_exit(1872.0, "TP", portfolio)

        expected_pnl = (1872.0 - 1800.0) * 5.555556
        assert result["capital_usd"] == pytest.approx(10000.0 + expected_pnl, rel=1e-4)
        assert result["has_position"] is False
        assert result["entry_price"] is None

    def test_execute_exit_saves_trade(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src.trading.eth_bot3.PORTFOLIO_PATH", tmp_path / "portfolio_eth.json")
        trades_path = tmp_path / "trades.json"
        monkeypatch.setattr("src.trading.eth_bot3.TRADES_PATH", trades_path)

        portfolio = dict(BASE_PORTFOLIO)
        execute_exit(1764.0, "SL", portfolio)

        assert trades_path.exists()
        with open(trades_path) as f:
            trades = json.load(f)
        assert len(trades) == 1
        assert trades[0]["exit_reason"] == "SL"
        assert trades[0]["symbol"] == "ETHUSDT"
