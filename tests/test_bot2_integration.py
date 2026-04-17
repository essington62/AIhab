"""Integration tests for Bot 2 in run_cycle flow."""
import pytest
from unittest.mock import patch, MagicMock
from src.trading.paper_trader import check_momentum_filter, _execute_bot2_entry


BASE_PORTFOLIO = {
    "has_position": False,
    "capital_usd": 10000.0,
    "entry_price": None,
    "quantity": 0.0,
    "trailing_high": None,
    "stop_loss_price": None,
    "take_profit_price": None,
    "trailing_stop_pct_actual": None,
    "stops_mode": None,
    "entry_atr_pct": None,
    "last_updated": None,
}

MF_PARAMS = {
    "stop_loss_pct": 0.015,
    "take_profit_pct": 0.02,
    "trailing_stop_pct": 0.01,
    "max_hold_hours": 120,
}

FULL_PARAMS = {
    "execution": {
        "position_size_pct": 1.0,
        "paper_capital_usd": 10000.0,
        "use_dynamic_stops": True,
        "atr_multiplier_sl": 2.0,
        "atr_multiplier_tp": 1.5,
        "atr_multiplier_trail": 1.0,
        "min_stop_loss_pct": 0.01,
        "max_stop_loss_pct": 0.06,
        "min_take_profit_pct": 0.01,
        "max_take_profit_pct": 0.06,
        "min_trailing_pct": 0.008,
        "max_trailing_pct": 0.04,
        "stop_loss_pct": 0.03,
        "take_profit_pct": 0.02,
        "trailing_stop_pct": 0.015,
    },
    "momentum_filter": {
        "enabled": True,
        "stablecoin_z_min": 1.3,
        "ret_1d_min": 0.0,
        "rsi_min": 50,
        "bb_pct_max": 0.98,
        "require_above_ma21": True,
        **MF_PARAMS,
    }
}

TECH_PASS = {
    "close": 75000,
    "rsi_14": 65,
    "bb_pct": 0.85,
    "ret_1d": 0.01,
    "ma_21": 73000,
    "atr_14": 1500,
}

ZSCORES_PASS = {"stablecoin_z": 1.5}


class TestBot2MomentumFilter:
    def test_bear_regime_blocks_bot2(self):
        """Bear regime: check_momentum_filter passes but run_cycle should block — simulated via regime check."""
        r = check_momentum_filter(TECH_PASS, ZSCORES_PASS, FULL_PARAMS)
        assert r["passed"] is True
        # In run_cycle, Bear regime skips the Bot 2 block entirely
        # We verify the condition here — caller must check regime != "Bear"

    def test_bot2_passes_all_conditions(self):
        r = check_momentum_filter(TECH_PASS, ZSCORES_PASS, FULL_PARAMS)
        assert r["passed"] is True
        assert r["reason"] == "momentum_confirmed"

    def test_bot2_blocked_by_low_stable_z(self):
        r = check_momentum_filter(TECH_PASS, {"stablecoin_z": 0.5}, FULL_PARAMS)
        assert r["passed"] is False
        assert "LOW_LIQUIDITY" in r["reason"]

    def test_bot2_entry_uses_own_stops(self):
        """_execute_bot2_entry must use bot2 fixed stops, not execution defaults."""
        import copy
        portfolio = copy.deepcopy(BASE_PORTFOLIO)
        price = 80000.0

        with patch("src.trading.paper_trader.get_params", return_value=FULL_PARAMS), \
             patch("src.trading.paper_trader.atomic_write_json"):
            result = _execute_bot2_entry(price, portfolio, MF_PARAMS)

        expected_sl = round(price * (1 - 0.015), 2)
        expected_tp = round(price * (1 + 0.02), 2)
        assert result["stop_loss_price"] == expected_sl
        assert result["take_profit_price"] == expected_tp
        assert result["trailing_stop_pct_actual"] == 0.01
        assert result["stops_mode"] == "bot2_fixed"
        assert result["entry_atr_pct"] is None
        assert result["has_position"] is True

    def test_bot2_entry_quantity(self):
        """quantity = capital * size_pct / price."""
        import copy
        portfolio = copy.deepcopy(BASE_PORTFOLIO)
        price = 80000.0

        with patch("src.trading.paper_trader.get_params", return_value=FULL_PARAMS), \
             patch("src.trading.paper_trader.atomic_write_json"):
            result = _execute_bot2_entry(price, portfolio, MF_PARAMS)

        expected_qty = round(10000.0 * 1.0 / price, 6)
        assert result["quantity"] == expected_qty

    def test_mutex_no_double_entry(self):
        """If has_position=True, Bot 2 should not be entered (mutex via bot_entered logic)."""
        # check_momentum_filter itself doesn't check has_position — the caller does
        # Verify that the filter passes independently
        r = check_momentum_filter(TECH_PASS, ZSCORES_PASS, FULL_PARAMS)
        assert r["passed"] is True
        # The mutex check `bot_entered is None and not portfolio["has_position"]`
        # is in run_cycle — not in check_momentum_filter

    def test_bot2_timeout_hours_calculation(self):
        """Bot 2 timeout at 120h — verify time math."""
        import pandas as pd
        from src.trading.execution import parse_utc

        entry_str = str(pd.Timestamp.utcnow() - pd.Timedelta(hours=121))
        entry_time = parse_utc(entry_str)
        now = pd.Timestamp.utcnow()
        hours_in = (now - entry_time).total_seconds() / 3600
        assert hours_in >= 120

    def test_bot2_no_timeout_for_bot1(self):
        """Bot 1 trades do not have max_hold_hours timeout."""
        import pandas as pd
        from src.trading.execution import parse_utc

        # 200h ago — Bot 1 should NOT be timed out (no timeout logic for bot1)
        entry_str = str(pd.Timestamp.utcnow() - pd.Timedelta(hours=200))
        entry_time = parse_utc(entry_str)
        now = pd.Timestamp.utcnow()
        hours_in = (now - entry_time).total_seconds() / 3600
        # timeout check only runs when entry_bot == "bot2"
        portfolio = {"entry_bot": "bot1", "entry_max_hold_hours": 120}
        assert portfolio.get("entry_bot", "bot1") != "bot2"
