"""
tests/test_paper_trader.py — Paper trader unit tests (no I/O, mocked deps).
Run: pytest tests/test_paper_trader.py -v
"""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.trading.execution import (
    atomic_write_json,
    check_stops,
    execute_entry,
    execute_exit,
)


# ---------------------------------------------------------------------------
# atomic_write_json
# ---------------------------------------------------------------------------

def test_atomic_write_json(tmp_path):
    path = tmp_path / "state.json"
    data = {"has_position": False, "capital_usd": 10000.0}
    atomic_write_json(data, path)
    import json
    with open(path) as f:
        loaded = json.load(f)
    assert loaded["capital_usd"] == 10000.0
    assert not path.with_suffix(".tmp").exists()  # temp file cleaned up


# ---------------------------------------------------------------------------
# execute_entry
# ---------------------------------------------------------------------------

class TestExecuteEntry:
    def _portfolio(self):
        return {
            "has_position": False,
            "entry_price": None,
            "entry_time": None,
            "quantity": 0.0,
            "capital_usd": 10000.0,
            "trailing_high": None,
            "stop_loss_price": None,
            "take_profit_price": None,
            "last_updated": None,
        }

    def test_entry_sets_position(self, tmp_path):
        with patch("src.trading.execution.get_path") as mock_path:
            mock_path.return_value = tmp_path / "portfolio_state.json"
            portfolio = execute_entry(71000.0, self._portfolio())

        assert portfolio["has_position"] is True
        assert portfolio["entry_price"] == 71000.0
        assert portfolio["quantity"] > 0
        assert portfolio["stop_loss_price"] < 71000.0
        assert portfolio["take_profit_price"] > 71000.0

    def test_stop_loss_pct(self, tmp_path):
        with patch("src.trading.execution.get_path") as mock_path:
            mock_path.return_value = tmp_path / "portfolio_state.json"
            portfolio = execute_entry(70000.0, self._portfolio())

        # SL = 2% below entry
        assert portfolio["stop_loss_price"] == pytest.approx(70000.0 * 0.98, abs=1)

    def test_take_profit_pct(self, tmp_path):
        with patch("src.trading.execution.get_path") as mock_path:
            mock_path.return_value = tmp_path / "portfolio_state.json"
            portfolio = execute_entry(70000.0, self._portfolio())

        # TP = 1.5% above entry
        assert portfolio["take_profit_price"] == pytest.approx(70000.0 * 1.015, abs=1)


# ---------------------------------------------------------------------------
# execute_exit
# ---------------------------------------------------------------------------

class TestExecuteExit:
    def _portfolio_with_position(self, entry=70000.0):
        return {
            "has_position": True,
            "entry_price": entry,
            "entry_time": "2026-04-08T10:00:00+00:00",
            "quantity": 0.142857,  # 10000 / 70000
            "capital_usd": 10000.0,
            "trailing_high": entry,
            "stop_loss_price": entry * 0.98,
            "take_profit_price": entry * 1.015,
            "last_updated": None,
        }

    def test_exit_closes_position(self, tmp_path):
        with patch("src.trading.execution.get_path") as mock_path:
            mock_path.return_value = tmp_path / "portfolio_state.json"
            p = execute_exit(71050.0, self._portfolio_with_position(70000.0), "TAKE_PROFIT")

        assert p["has_position"] is False
        assert p["entry_price"] is None
        assert p["quantity"] == 0.0

    def test_exit_realises_pnl(self, tmp_path):
        with patch("src.trading.execution.get_path") as mock_path:
            mock_path.return_value = tmp_path / "portfolio_state.json"
            p = execute_exit(71050.0, self._portfolio_with_position(70000.0), "TAKE_PROFIT")

        # PnL = (71050 - 70000) * 0.142857 ≈ $150
        assert p["capital_usd"] > 10000.0

    def test_exit_loss_reduces_capital(self, tmp_path):
        with patch("src.trading.execution.get_path") as mock_path:
            mock_path.return_value = tmp_path / "portfolio_state.json"
            p = execute_exit(68600.0, self._portfolio_with_position(70000.0), "STOP_LOSS")

        assert p["capital_usd"] < 10000.0

    def test_exit_without_position_is_noop(self, tmp_path):
        no_pos = {
            "has_position": False, "entry_price": None, "quantity": 0.0,
            "capital_usd": 10000.0, "trailing_high": None,
            "stop_loss_price": None, "take_profit_price": None,
            "entry_time": None, "last_updated": None,
        }
        with patch("src.trading.execution.get_path") as mock_path:
            mock_path.return_value = tmp_path / "portfolio_state.json"
            p = execute_exit(70000.0, no_pos, "TEST")

        assert p["has_position"] is False


# ---------------------------------------------------------------------------
# check_stops
# ---------------------------------------------------------------------------

class TestCheckStops:
    def _portfolio(self, entry=70000.0):
        return {
            "has_position": True,
            "entry_price": entry,
            "quantity": 0.142857,
            "capital_usd": 10000.0,
            "trailing_high": entry,
            "stop_loss_price": round(entry * 0.98, 2),
            "take_profit_price": round(entry * 1.015, 2),
            "entry_time": "2026-04-08T10:00:00+00:00",
            "last_updated": None,
        }

    def test_take_profit_triggered(self, tmp_path):
        with patch("src.trading.execution.get_path") as mock_path:
            mock_path.return_value = tmp_path / "portfolio_state.json"
            triggered, reason = check_stops(71050.0, self._portfolio(70000.0))

        assert triggered is True
        assert reason == "TAKE_PROFIT"

    def test_stop_loss_triggered(self, tmp_path):
        with patch("src.trading.execution.get_path") as mock_path:
            mock_path.return_value = tmp_path / "portfolio_state.json"
            triggered, reason = check_stops(68600.0, self._portfolio(70000.0))

        assert triggered is True
        assert reason == "STOP_LOSS"

    def test_no_trigger_in_range(self, tmp_path):
        with patch("src.trading.execution.get_path") as mock_path:
            mock_path.return_value = tmp_path / "portfolio_state.json"
            triggered, reason = check_stops(70500.0, self._portfolio(70000.0))

        assert triggered is False

    def test_trailing_stop_moves_up(self, tmp_path):
        with patch("src.trading.execution.get_path") as mock_path:
            mock_path.return_value = tmp_path / "portfolio_state.json"
            p = self._portfolio(70000.0)
            original_sl = p["stop_loss_price"]
            check_stops(72000.0, p)  # new high → trailing stop moves up

        # After call, SL should have moved up (check the portfolio dict was mutated)
        assert p["stop_loss_price"] > original_sl

    def test_no_position_never_triggers(self, tmp_path):
        p = {"has_position": False, "entry_price": None}
        with patch("src.trading.execution.get_path") as mock_path:
            mock_path.return_value = tmp_path / "portfolio_state.json"
            triggered, reason = check_stops(68000.0, p)

        assert triggered is False
