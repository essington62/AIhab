"""Tests for check_cooldown() — post-SL reentry protection."""
import pandas as pd
import pytest

from src.trading.paper_trader import check_cooldown

PARAMS = {
    "reversal_filter": {
        "cooldown": {
            "enabled": True,
            "hours_after_sl": 12,
            "require_price_above_exit": True,
            "max_consecutive_sl": 3,
            "consecutive_sl_pause_hours": 24,
        }
    },
    "momentum_filter": {
        "cooldown": {
            "enabled": True,
            "hours_after_sl": 12,
            "require_price_above_exit": True,
            "max_consecutive_sl": 3,
            "consecutive_sl_pause_hours": 24,
        }
    },
}


def _sl_portfolio(hours_ago: float, sl_price: float, bot: str = "bot1", consec: int = 1) -> dict:
    sl_time = str(pd.Timestamp.now("UTC") - pd.Timedelta(hours=hours_ago))
    return {
        "last_sl_time": sl_time,
        "last_sl_price": sl_price,
        "last_sl_bot": bot,
        "consecutive_sl_count": consec,
    }


class TestCooldownBasic:
    def test_no_previous_sl(self):
        r = check_cooldown({}, 75000, "bot1", PARAMS)
        assert r["can_enter"] is True
        assert r["reason"] == "no_previous_sl"

    def test_cooldown_disabled(self):
        params = {"reversal_filter": {"cooldown": {"enabled": False}}}
        portfolio = _sl_portfolio(hours_ago=1, sl_price=75000)
        r = check_cooldown(portfolio, 75000, "bot1", params)
        assert r["can_enter"] is True
        assert r["reason"] == "cooldown_disabled"

    def test_within_cooldown_period(self):
        """SL 2h ago, cooldown 12h → cannot enter."""
        r = check_cooldown(_sl_portfolio(2, 76000), 77000, "bot1", PARAMS)
        assert r["can_enter"] is False
        assert "COOLDOWN" in r["reason"]

    def test_after_cooldown_period_price_above(self):
        """SL 15h ago, price above exit → can enter."""
        r = check_cooldown(_sl_portfolio(15, 75000), 76000, "bot1", PARAMS)
        assert r["can_enter"] is True
        assert r["reason"] == "cooldown_passed"

    def test_after_cooldown_period_price_below(self):
        """SL 15h ago, price still below exit → cannot enter."""
        r = check_cooldown(_sl_portfolio(15, 76000), 75000, "bot1", PARAMS)
        assert r["can_enter"] is False
        assert "PRICE_BELOW_EXIT" in r["reason"]

    def test_after_cooldown_period_price_equal(self):
        """Price == sl_exit price → still blocked (need strictly >)."""
        r = check_cooldown(_sl_portfolio(15, 75000), 75000, "bot1", PARAMS)
        assert r["can_enter"] is False
        assert "PRICE_BELOW_EXIT" in r["reason"]


class TestCooldownConsecutive:
    def test_consecutive_sl_extended_pause(self):
        """3 consecutive SLs → 24h required instead of 12h."""
        r = check_cooldown(_sl_portfolio(15, 74000, consec=3), 75000, "bot1", PARAMS)
        assert r["can_enter"] is False
        assert "CONSECUTIVE_SL" in r["reason"]

    def test_consecutive_sl_after_extended_pause(self):
        """3 SLs, 25h ago, price above → can enter."""
        r = check_cooldown(_sl_portfolio(25, 74000, consec=3), 75000, "bot1", PARAMS)
        assert r["can_enter"] is True

    def test_two_consecutive_uses_normal_cooldown(self):
        """2 SLs (< max_consecutive=3) → still uses 12h, not 24h."""
        r = check_cooldown(_sl_portfolio(13, 74000, consec=2), 75000, "bot1", PARAMS)
        assert r["can_enter"] is True


class TestCooldownCrossBot:
    def test_bot1_sl_does_not_block_bot2(self):
        """Bot 1 SL should not block Bot 2."""
        r = check_cooldown(_sl_portfolio(2, 76000, bot="bot1"), 75000, "bot2", PARAMS)
        assert r["can_enter"] is True
        assert "bot1" in r["reason"]

    def test_bot2_sl_does_not_block_bot1(self):
        """Bot 2 SL should not block Bot 1."""
        r = check_cooldown(_sl_portfolio(2, 76000, bot="bot2"), 75000, "bot1", PARAMS)
        assert r["can_enter"] is True
        assert "bot2" in r["reason"]


class TestCooldownReproduceRealCase:
    def test_immediate_reentry_blocked(self):
        """Reproduce: Bot1 SL at 20:05, tries to reenter at 20:05 → blocked."""
        r = check_cooldown(
            _sl_portfolio(hours_ago=0.02, sl_price=75617),
            75600,
            "bot1",
            PARAMS,
        )
        assert r["can_enter"] is False

    def test_reentry_allowed_after_recovery(self):
        """After 13h + price recovered above SL exit → allowed."""
        r = check_cooldown(
            _sl_portfolio(hours_ago=13, sl_price=75617),
            76000,
            "bot1",
            PARAMS,
        )
        assert r["can_enter"] is True
