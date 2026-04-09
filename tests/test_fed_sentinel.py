"""
tests/test_fed_sentinel.py — Fed Sentinel unit tests.
Run: pytest tests/test_fed_sentinel.py -v
"""

from datetime import date

import pytest

from src.features.fed_sentinel import (
    compute_fomc_proximity_adjustment,
    get_next_fed_event,
    is_in_blackout,
)


# ---------------------------------------------------------------------------
# is_in_blackout
# ---------------------------------------------------------------------------

class TestIsInBlackout:
    def test_t10_before_fomc_is_blackout(self):
        # FOMC 2026-04-29 → blackout starts 2026-04-19
        assert is_in_blackout(date(2026, 4, 19)) is True

    def test_t5_before_fomc_is_blackout(self):
        assert is_in_blackout(date(2026, 4, 24)) is True

    def test_t2_before_fomc_is_blackout(self):
        # blackout_ends_days_before=2, so T-2 is still in blackout
        assert is_in_blackout(date(2026, 4, 27)) is True

    def test_t1_before_fomc_not_blackout(self):
        # T-1 (2026-04-28) is not in blackout (ends at T-2)
        assert is_in_blackout(date(2026, 4, 28)) is False

    def test_fomc_day_not_blackout(self):
        assert is_in_blackout(date(2026, 4, 29)) is False

    def test_normal_day_not_blackout(self):
        assert is_in_blackout(date(2026, 4, 10)) is False


# ---------------------------------------------------------------------------
# compute_fomc_proximity_adjustment
# ---------------------------------------------------------------------------

class TestProximityAdjustment:
    def test_t0_fomc_decision_max_adj(self):
        # FOMC 2026-04-29 → T-0 = day of decision → delta=0, falls in 0<=delta<=2
        adj = compute_fomc_proximity_adjustment(date(2026, 4, 29))
        assert adj == pytest.approx(1.5)

    def test_t2_before_fomc(self):
        # 2026-04-27, FOMC 2026-04-29 → delta=2
        adj = compute_fomc_proximity_adjustment(date(2026, 4, 27))
        assert adj == pytest.approx(1.5)

    def test_t3_before_fomc(self):
        adj = compute_fomc_proximity_adjustment(date(2026, 4, 26))
        assert adj == pytest.approx(0.7)

    def test_t5_before_fomc(self):
        adj = compute_fomc_proximity_adjustment(date(2026, 4, 24))
        assert adj == pytest.approx(0.7)

    def test_t1_after_fomc(self):
        # 2026-04-30: day after FOMC → delta=-1
        adj = compute_fomc_proximity_adjustment(date(2026, 4, 30))
        assert adj == pytest.approx(0.3)

    def test_blackout_adj(self):
        # T-10 = 2026-04-19: in blackout, not in 0-5 day window
        adj = compute_fomc_proximity_adjustment(date(2026, 4, 19))
        assert adj == pytest.approx(0.3)

    def test_normal_day_zero(self):
        adj = compute_fomc_proximity_adjustment(date(2026, 4, 10))
        assert adj == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# get_next_fed_event
# ---------------------------------------------------------------------------

class TestGetNextFedEvent:
    def test_returns_dict_with_days_away(self):
        event = get_next_fed_event(date(2026, 4, 8))
        assert "date" in event
        assert "type" in event
        assert "days_away" in event

    def test_days_away_positive(self):
        event = get_next_fed_event(date(2026, 4, 8))
        assert event["days_away"] >= 0

    def test_next_event_is_warsh_hearing(self):
        # On 2026-04-08, next event should be Warsh hearing on 2026-04-16 (8 days)
        event = get_next_fed_event(date(2026, 4, 8))
        assert event["days_away"] == 8
        assert "Warsh" in event.get("member", "") or event["type"] in ("Senate hearing", "House hearing")

    def test_after_all_events_returns_none(self):
        event = get_next_fed_event(date(2027, 1, 1))
        assert event["date"] is None
