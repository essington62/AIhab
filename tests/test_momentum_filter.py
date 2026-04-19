"""Tests for check_momentum_filter()."""
import pytest
from src.trading.paper_trader import check_momentum_filter

PARAMS_ENABLED = {
    "momentum_filter": {
        "enabled": True,
        "stablecoin_z_min": 1.3,
        "ret_1d_min": 0.0,
        "rsi_min": 50,
        "bb_pct_max": 0.98,
        "require_above_ma21": True,
    }
}

PARAMS_DISABLED = {
    "momentum_filter": {"enabled": False}
}


def _tech(close=75000, rsi=65, bb=0.85, ret=0.01, ma21=73000):
    return {
        "close": close,
        "rsi_14": rsi,
        "bb_pct": bb,
        "ret_1d": ret,
        "ma_21": ma21,
    }


def _zscores(stable_z=1.5):
    return {"stablecoin_z": stable_z}


class TestMomentumFilterBasic:
    def test_all_conditions_pass(self):
        r = check_momentum_filter(_tech(), _zscores(), PARAMS_ENABLED)
        assert r["passed"] is True
        assert r["reason"] == "momentum_confirmed"

    def test_filter_disabled(self):
        r = check_momentum_filter(_tech(), _zscores(), PARAMS_DISABLED)
        assert r["passed"] is False
        assert r["reason"] == "filter_disabled"

    def test_low_stablecoin_z(self):
        r = check_momentum_filter(_tech(), _zscores(stable_z=0.5), PARAMS_ENABLED)
        assert r["passed"] is False
        assert "LOW_LIQUIDITY" in r["reason"]

    def test_negative_ret_1d(self):
        r = check_momentum_filter(_tech(ret=-0.01), _zscores(), PARAMS_ENABLED)
        assert r["passed"] is False
        assert "NEG_MOMENTUM" in r["reason"]

    def test_low_rsi(self):
        r = check_momentum_filter(_tech(rsi=40), _zscores(), PARAMS_ENABLED)
        assert r["passed"] is False
        assert "RSI_LOW" in r["reason"]

    def test_blow_off_top(self):
        r = check_momentum_filter(_tech(bb=0.99), _zscores(), PARAMS_ENABLED)
        assert r["passed"] is False
        assert "BLOW_OFF_TOP" in r["reason"]

    def test_below_ma21(self):
        r = check_momentum_filter(_tech(close=72000, ma21=73000), _zscores(), PARAMS_ENABLED)
        assert r["passed"] is False
        assert "BELOW_MA21" in r["reason"]


class TestMomentumFilterEdgeCases:
    def test_multiple_failures(self):
        r = check_momentum_filter(
            _tech(rsi=40, ret=-0.01, bb=0.99),
            _zscores(stable_z=0.5),
            PARAMS_ENABLED
        )
        assert r["passed"] is False
        assert "LOW_LIQUIDITY" in r["reason"]
        assert "NEG_MOMENTUM" in r["reason"]
        assert "RSI_LOW" in r["reason"]
        assert "BLOW_OFF_TOP" in r["reason"]

    def test_missing_stablecoin_z(self):
        r = check_momentum_filter(_tech(), {"stablecoin_z": None}, PARAMS_ENABLED)
        assert r["passed"] is False
        assert "MISSING_DATA" in r["reason"]

    def test_missing_rsi(self):
        tech = _tech()
        tech["rsi_14"] = None
        r = check_momentum_filter(tech, _zscores(), PARAMS_ENABLED)
        assert r["passed"] is False
        assert "MISSING_DATA" in r["reason"]

    def test_exact_threshold_stablecoin(self):
        """stablecoin_z == 1.3 should NOT pass (need > 1.3)."""
        r = check_momentum_filter(_tech(), _zscores(stable_z=1.3), PARAMS_ENABLED)
        assert r["passed"] is False

    def test_exact_threshold_bb(self):
        """bb_pct == 0.98 should NOT pass (need < 0.98)."""
        r = check_momentum_filter(_tech(bb=0.98), _zscores(), PARAMS_ENABLED)
        assert r["passed"] is False

    def test_ret_1d_zero_should_fail(self):
        """ret_1d == 0 should NOT pass (need > 0)."""
        r = check_momentum_filter(_tech(ret=0.0), _zscores(), PARAMS_ENABLED)
        assert r["passed"] is False

    def test_rsi_exactly_50_should_fail(self):
        """RSI == 50 should NOT pass (need > 50)."""
        r = check_momentum_filter(_tech(rsi=50), _zscores(), PARAMS_ENABLED)
        assert r["passed"] is False

    def test_missing_ma21_data(self):
        tech = _tech()
        tech["ma_21"] = None
        r = check_momentum_filter(tech, _zscores(), PARAMS_ENABLED)
        assert r["passed"] is False
        assert "MISSING_MA21" in r["reason"]


class TestMomentumFilterBoundary:
    def test_stablecoin_just_above(self):
        r = check_momentum_filter(_tech(), _zscores(stable_z=1.31), PARAMS_ENABLED)
        assert r["passed"] is True

    def test_ret_just_above_zero(self):
        r = check_momentum_filter(_tech(ret=0.0001), _zscores(), PARAMS_ENABLED)
        assert r["passed"] is True

    def test_bb_just_below_threshold(self):
        r = check_momentum_filter(_tech(bb=0.979), _zscores(), PARAMS_ENABLED)
        assert r["passed"] is True

    def test_rsi_just_above_50(self):
        r = check_momentum_filter(_tech(rsi=50.1), _zscores(), PARAMS_ENABLED)
        assert r["passed"] is True


class TestSpikeGuard:
    """Tests for spike guard (late entry protection)."""

    PARAMS_SPIKE = {
        "momentum_filter": {
            "enabled": True,
            "stablecoin_z_min": 1.3,
            "ret_1d_min": 0.0,
            "rsi_min": 50,
            "bb_pct_max": 0.98,
            "require_above_ma21": True,
            "spike_guard": {
                "enabled": True,
                "spike_ret_max": 0.03,
                "spike_rsi_max": 65,
            },
        }
    }

    def test_spike_blocks_high_ret_and_high_rsi(self):
        """ret_1d > 3% AND RSI > 65 → LATE_SPIKE."""
        r = check_momentum_filter(
            _tech(ret=0.035, rsi=72),
            _zscores(),
            self.PARAMS_SPIKE,
        )
        assert r["passed"] is False
        assert "LATE_SPIKE" in r["reason"]

    def test_spike_allows_high_ret_low_rsi(self):
        """ret_1d > 3% but RSI <= 65 → should pass (only one condition)."""
        r = check_momentum_filter(
            _tech(ret=0.035, rsi=55),
            _zscores(),
            self.PARAMS_SPIKE,
        )
        assert r["passed"] is True

    def test_spike_allows_low_ret_high_rsi(self):
        """RSI > 65 but ret_1d <= 3% → should pass (only one condition)."""
        r = check_momentum_filter(
            _tech(ret=0.01, rsi=72),
            _zscores(),
            self.PARAMS_SPIKE,
        )
        assert r["passed"] is True

    def test_spike_exact_thresholds_should_pass(self):
        """ret_1d == 3% AND RSI == 65 → should pass (need strictly > )."""
        r = check_momentum_filter(
            _tech(ret=0.03, rsi=65),
            _zscores(),
            self.PARAMS_SPIKE,
        )
        assert r["passed"] is True

    def test_spike_guard_disabled(self):
        """Spike guard disabled → should not block even with extreme values."""
        params_no_spike = {
            "momentum_filter": {
                "enabled": True,
                "stablecoin_z_min": 1.3,
                "ret_1d_min": 0.0,
                "rsi_min": 50,
                "bb_pct_max": 0.98,
                "require_above_ma21": True,
                "spike_guard": {"enabled": False},
            }
        }
        r = check_momentum_filter(
            _tech(ret=0.05, rsi=80),
            _zscores(),
            params_no_spike,
        )
        assert r["passed"] is True

    def test_spike_reproduces_failed_trade(self):
        """Reproduce the actual failed trade: ret_1d=3.2%, RSI=72.8."""
        r = check_momentum_filter(
            _tech(ret=0.032, rsi=72.8, bb=0.743, close=77459, ma21=76000),
            _zscores(stable_z=2.09),
            self.PARAMS_SPIKE,
        )
        assert r["passed"] is False
        assert "LATE_SPIKE" in r["reason"]
