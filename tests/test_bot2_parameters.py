"""Tests for Bot 2 parameters in parameters.yml."""
import pytest
from src.config import get_params


class TestBot2Parameters:
    def test_momentum_filter_exists(self):
        params = get_params()
        assert "momentum_filter" in params

    def test_momentum_filter_has_all_keys(self):
        mf = get_params()["momentum_filter"]
        required = [
            "enabled", "stablecoin_z_min", "ret_1d_min", "rsi_min",
            "bb_pct_max", "require_above_ma21",
            "rsi_max", "dist_high_7d_min", "respect_gate_kill_switches",
            "stop_loss_pct", "take_profit_pct", "trailing_stop_pct",
            "max_hold_hours",
        ]
        for key in required:
            assert key in mf, f"Missing key: {key}"

    def test_momentum_filter_values(self):
        mf = get_params()["momentum_filter"]
        assert mf["stablecoin_z_min"] == 1.3
        assert mf["ret_1d_min"] == 0.0
        assert mf["rsi_min"] == 50
        assert mf["bb_pct_max"] == 0.85
        assert mf["rsi_max"] == 65
        assert mf["dist_high_7d_min"] == 0.02
        assert mf["respect_gate_kill_switches"] is True
        assert mf["stop_loss_pct"] == 0.015
        assert mf["take_profit_pct"] == 0.02
        assert mf["trailing_stop_pct"] == 0.01
        assert mf["max_hold_hours"] == 120
