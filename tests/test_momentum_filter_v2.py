"""Tests for check_momentum_filter_v2."""
import pytest
from src.trading.paper_trader import check_momentum_filter_v2

BASE_PARAMS = {
    "momentum_filter_v2": {
        "enabled": True,
        "stablecoin_z_min": 1.3,
        "bb_pct_max": 0.98,
        "classic": {"rsi_min": 50, "ret_1d_min": 0.0},
        "early_reversal": {
            "enabled": True,
            "ret_1d_floor": -0.015,
            "rsi_floor": 35,
            "delta_ret_3h_min": 0.005,
        },
    }
}


class TestDisabled:
    def test_disabled_returns_not_passed(self):
        params = {"momentum_filter_v2": {"enabled": False}}
        r = check_momentum_filter_v2({}, {}, params)
        assert r["passed"] is False
        assert r["entry_mode"] is None

    def test_missing_key_is_disabled(self):
        r = check_momentum_filter_v2({}, {}, {})
        assert r["passed"] is False


class TestGlobalFilters:
    def test_low_liquidity_blocks_before_classic(self):
        tech = {"close": 75000, "ma_21": 74000, "rsi_14": 60, "bb_pct": 0.5,
                "ret_1d": 0.02, "ret_1d_1h_ago": 0.01, "ret_1d_3h_ago": 0.0}
        r = check_momentum_filter_v2(tech, {"stablecoin_z": 1.0}, BASE_PARAMS)
        assert r["passed"] is False
        assert "LOW_LIQUIDITY" in r["reason"]

    def test_blow_off_top_blocks(self):
        tech = {"close": 75000, "ma_21": 74000, "rsi_14": 60, "bb_pct": 0.99,
                "ret_1d": 0.02, "ret_1d_1h_ago": 0.01, "ret_1d_3h_ago": 0.0}
        r = check_momentum_filter_v2(tech, {"stablecoin_z": 2.0}, BASE_PARAMS)
        assert r["passed"] is False
        assert "BLOW_OFF_TOP" in r["reason"]

    def test_missing_data_blocks(self):
        r = check_momentum_filter_v2(
            {"close": 75000},
            {"stablecoin_z": 2.0},
            BASE_PARAMS,
        )
        assert r["passed"] is False
        assert r["reason"] == "MISSING_DATA"


class TestClassic:
    def test_classic_passes(self):
        tech = {"close": 75000, "ma_21": 74000, "rsi_14": 55, "bb_pct": 0.5,
                "ret_1d": 0.01, "ret_1d_1h_ago": 0.005, "ret_1d_3h_ago": 0.0}
        r = check_momentum_filter_v2(tech, {"stablecoin_z": 1.5}, BASE_PARAMS)
        assert r["passed"] is True
        assert r["entry_mode"] == "classic"

    def test_classic_rsi_below_50_fails(self):
        tech = {"close": 75000, "ma_21": 74000, "rsi_14": 48, "bb_pct": 0.5,
                "ret_1d": 0.01, "ret_1d_1h_ago": None, "ret_1d_3h_ago": None}
        r = check_momentum_filter_v2(tech, {"stablecoin_z": 1.5}, BASE_PARAMS)
        assert r["classic_pass"] is False

    def test_classic_below_ma21_fails(self):
        tech = {"close": 73000, "ma_21": 74000, "rsi_14": 55, "bb_pct": 0.5,
                "ret_1d": 0.01, "ret_1d_1h_ago": None, "ret_1d_3h_ago": None}
        r = check_momentum_filter_v2(tech, {"stablecoin_z": 1.5}, BASE_PARAMS)
        assert r["classic_pass"] is False

    def test_classic_negative_ret_fails(self):
        tech = {"close": 75000, "ma_21": 74000, "rsi_14": 55, "bb_pct": 0.5,
                "ret_1d": -0.001, "ret_1d_1h_ago": None, "ret_1d_3h_ago": None}
        r = check_momentum_filter_v2(tech, {"stablecoin_z": 1.5}, BASE_PARAMS)
        assert r["classic_pass"] is False


class TestEarlyReversal:
    def _early_tech(self, **overrides):
        base = {
            "close": 74819, "ma_21": 74900, "rsi_14": 44.5, "bb_pct": 0.5,
            "ret_1d": -0.0076, "ret_1d_1h_ago": -0.0143, "ret_1d_3h_ago": -0.0144,
        }
        base.update(overrides)
        return base

    def test_real_bottom_20apr_passes(self):
        """Case from 2026-04-19 22:00 — RSI 44, stablecoin 2.25, trend improving."""
        r = check_momentum_filter_v2(
            self._early_tech(),
            {"stablecoin_z": 2.25},
            BASE_PARAMS,
        )
        assert r["passed"] is True
        assert r["entry_mode"] == "early"

    def test_free_fall_ret_below_floor(self):
        """ret_1d < -1.5% — free fall, block even if improving."""
        r = check_momentum_filter_v2(
            self._early_tech(ret_1d=-0.020, ret_1d_1h_ago=-0.025, ret_1d_3h_ago=-0.030),
            {"stablecoin_z": 2.0},
            BASE_PARAMS,
        )
        assert r["passed"] is False

    def test_trend_not_improving_rejected(self):
        """Descending trend: ret now < ret_1h (not improving)."""
        r = check_momentum_filter_v2(
            self._early_tech(ret_1d=-0.012, ret_1d_1h_ago=-0.008, ret_1d_3h_ago=-0.005),
            {"stablecoin_z": 1.5},
            BASE_PARAMS,
        )
        assert r["passed"] is False

    def test_delta_3h_too_small_rejected(self):
        """Improving but only by 0.1% — below delta_ret_3h_min=0.5%."""
        r = check_momentum_filter_v2(
            self._early_tech(ret_1d=-0.010, ret_1d_1h_ago=-0.0105, ret_1d_3h_ago=-0.011),
            {"stablecoin_z": 1.5},
            BASE_PARAMS,
        )
        assert r["passed"] is False

    def test_extreme_oversold_rsi_below_floor(self):
        """RSI < 35 — extreme oversold, block even with trend improving."""
        r = check_momentum_filter_v2(
            self._early_tech(rsi_14=30),
            {"stablecoin_z": 1.5},
            BASE_PARAMS,
        )
        assert r["passed"] is False

    def test_early_disabled_by_config(self):
        params_no_early = {
            "momentum_filter_v2": {
                **BASE_PARAMS["momentum_filter_v2"],
                "early_reversal": {"enabled": False},
            }
        }
        r = check_momentum_filter_v2(
            self._early_tech(),
            {"stablecoin_z": 2.25},
            params_no_early,
        )
        assert r["passed"] is False

    def test_missing_lag_data_skips_early(self):
        """Without ret_1d_1h_ago, early path is skipped."""
        tech = self._early_tech(ret_1d_1h_ago=None, ret_1d_3h_ago=None)
        r = check_momentum_filter_v2(tech, {"stablecoin_z": 2.25}, BASE_PARAMS)
        assert r["passed"] is False

    def test_classic_preferred_over_early(self):
        """If classic passes, entry_mode should be 'classic', not 'early'."""
        tech = {
            "close": 76000, "ma_21": 74000, "rsi_14": 55, "bb_pct": 0.5,
            "ret_1d": 0.01, "ret_1d_1h_ago": -0.005, "ret_1d_3h_ago": -0.010,
        }
        r = check_momentum_filter_v2(tech, {"stablecoin_z": 2.0}, BASE_PARAMS)
        assert r["passed"] is True
        assert r["entry_mode"] == "classic"
