"""
tests/test_paper_trader.py — Paper trader unit tests (no I/O, mocked deps).
Run: pytest tests/test_paper_trader.py -v
"""

import json
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.trading.execution import (
    atomic_write_json,
    check_stops,
    execute_entry,
    execute_exit,
)
from src.trading.paper_trader import (
    _build_trade_record,
    _init_trade_tracking,
    _update_excursions,
)


# ---------------------------------------------------------------------------
# atomic_write_json
# ---------------------------------------------------------------------------

def test_atomic_write_json(tmp_path):
    path = tmp_path / "state.json"
    data = {"has_position": False, "capital_usd": 10000.0}
    atomic_write_json(data, path)
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

        # SL = 3% below entry (parameters.yml: stop_loss_pct: 0.03)
        assert portfolio["stop_loss_price"] == pytest.approx(70000.0 * 0.97, abs=1)

    def test_take_profit_pct(self, tmp_path):
        with patch("src.trading.execution.get_path") as mock_path:
            mock_path.return_value = tmp_path / "portfolio_state.json"
            portfolio = execute_entry(70000.0, self._portfolio())

        # TP = 2% above entry (parameters.yml: take_profit_pct: 0.02)
        assert portfolio["take_profit_price"] == pytest.approx(70000.0 * 1.02, abs=1)


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
            "stop_loss_price": entry * 0.97,
            "take_profit_price": entry * 1.02,
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
            p = execute_exit(67900.0, self._portfolio_with_position(70000.0), "STOP_LOSS")

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
            "stop_loss_price": round(entry * 0.97, 2),
            "take_profit_price": round(entry * 1.02, 2),
            "entry_time": "2026-04-08T10:00:00+00:00",
            "last_updated": None,
        }

    def test_take_profit_triggered(self, tmp_path):
        with patch("src.trading.execution.get_path") as mock_path:
            mock_path.return_value = tmp_path / "portfolio_state.json"
            triggered, reason = check_stops(71400.0, self._portfolio(70000.0))

        assert triggered is True
        assert reason == "TAKE_PROFIT"

    def test_stop_loss_triggered(self, tmp_path):
        with patch("src.trading.execution.get_path") as mock_path:
            mock_path.return_value = tmp_path / "portfolio_state.json"
            triggered, reason = check_stops(67900.0, self._portfolio(70000.0))

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

        # After call, SL should have moved up
        assert p["stop_loss_price"] > original_sl

    def test_no_position_never_triggers(self, tmp_path):
        p = {"has_position": False, "entry_price": None}
        with patch("src.trading.execution.get_path") as mock_path:
            mock_path.return_value = tmp_path / "portfolio_state.json"
            triggered, reason = check_stops(68000.0, p)

        assert triggered is False


# ---------------------------------------------------------------------------
# MAE/MFE — _update_excursions
# ---------------------------------------------------------------------------

class TestUpdateExcursions:
    def _open_portfolio(self, entry=70000.0):
        return {
            "has_position": True,
            "entry_price": entry,
            "entry_time": "2026-04-16T10:00:00+00:00",
            "quantity": 0.142857,
            "capital_usd": 10000.0,
            "max_favorable": 0.0,
            "max_adverse": 0.0,
            "mfe_time": None,
            "mae_time": None,
            "price_path": [],
        }

    def test_favorable_excursion_recorded(self, tmp_path):
        p = self._open_portfolio(70000.0)
        ts = pd.Timestamp("2026-04-16T11:00:00", tz="UTC")
        with patch("src.trading.paper_trader.get_path") as mock_path:
            mock_path.return_value = tmp_path / "portfolio_state.json"
            _update_excursions(p, 71000.0, ts)  # +1.43%

        assert p["max_favorable"] == pytest.approx(1000 / 70000, abs=1e-6)
        assert p["mfe_time"] is not None
        assert p["max_adverse"] == 0.0  # no adverse move
        assert len(p["price_path"]) == 1
        assert p["price_path"][0]["return_pct"] == pytest.approx(1000 / 70000 * 100, abs=0.01)

    def test_adverse_excursion_recorded(self, tmp_path):
        p = self._open_portfolio(70000.0)
        ts = pd.Timestamp("2026-04-16T11:00:00", tz="UTC")
        with patch("src.trading.paper_trader.get_path") as mock_path:
            mock_path.return_value = tmp_path / "portfolio_state.json"
            _update_excursions(p, 68600.0, ts)  # -2%

        assert p["max_adverse"] == pytest.approx(-1400 / 70000, abs=1e-6)
        assert p["mae_time"] is not None
        assert p["max_favorable"] == 0.0

    def test_mfe_updates_only_on_new_high(self, tmp_path):
        p = self._open_portfolio(70000.0)
        ts1 = pd.Timestamp("2026-04-16T11:00:00", tz="UTC")
        ts2 = pd.Timestamp("2026-04-16T12:00:00", tz="UTC")
        with patch("src.trading.paper_trader.get_path") as mock_path:
            mock_path.return_value = tmp_path / "portfolio_state.json"
            _update_excursions(p, 71000.0, ts1)  # +1.43% new high
            first_mfe_time = p["mfe_time"]
            _update_excursions(p, 70500.0, ts2)  # +0.71% not a new high

        assert p["mfe_time"] == first_mfe_time  # mfe_time did not change
        assert p["max_favorable"] == pytest.approx(1000 / 70000, abs=1e-6)

    def test_price_path_accumulates(self, tmp_path):
        p = self._open_portfolio(70000.0)
        with patch("src.trading.paper_trader.get_path") as mock_path:
            mock_path.return_value = tmp_path / "portfolio_state.json"
            for i, price in enumerate([70500, 71000, 70800]):
                ts = pd.Timestamp(f"2026-04-16T{11+i}:00:00", tz="UTC")
                _update_excursions(p, price, ts)

        assert len(p["price_path"]) == 3
        assert p["price_path"][0]["price"] == 70500
        assert p["price_path"][2]["price"] == 70800

    def test_no_position_is_noop(self, tmp_path):
        p = {"entry_price": None, "has_position": False}
        ts = pd.Timestamp("2026-04-16T11:00:00", tz="UTC")
        with patch("src.trading.paper_trader.get_path") as mock_path:
            mock_path.return_value = tmp_path / "portfolio_state.json"
            _update_excursions(p, 71000.0, ts)

        assert "max_favorable" not in p  # unchanged


# ---------------------------------------------------------------------------
# MAE/MFE — _build_trade_record
# ---------------------------------------------------------------------------

class TestBuildTradeRecord:
    def _portfolio(self, entry=70000.0):
        return {
            "has_position": True,
            "entry_price": entry,
            "entry_time": "2026-04-16T10:00:00+00:00",
            "quantity": 0.142857,
            "capital_usd": 10000.0,
            "trade_id": "test-uuid-1234",
            "max_favorable": 0.02,    # +2%
            "max_adverse": -0.015,    # -1.5%
            "mfe_time": "2026-04-16T14:00:00+00:00",
            "mae_time": "2026-04-16T12:00:00+00:00",
            "price_path": [{"timestamp": "t", "price": 71400, "return_pct": 2.0, "hours_since_entry": 1.0}],
            "entry_score_raw": 3.8,
            "entry_score_adjusted": 3.8,
            "entry_regime": "Sideways",
            "entry_bb_pct": 0.25,
            "entry_rsi": 38.0,
            "entry_atr": None,
            "entry_oi_z": -0.5,
            "entry_fg_raw": -0.3,
            "entry_cluster_technical": 3.0,
            "entry_cluster_positioning": 0.8,
            "entry_cluster_macro": 0.5,
            "entry_cluster_liquidity": 1.2,
            "entry_cluster_sentiment": 0.3,
            "entry_cluster_news": 0.2,
            "entry_stop_gain_pct": 0.02,
            "entry_stop_loss_pct": 0.03,
            "entry_trailing_stop_pct": 0.015,
        }

    def test_record_has_required_fields(self):
        p = self._portfolio()
        rec = _build_trade_record(p, 71400.0, "TAKE_PROFIT")
        required = ["trade_id", "entry_time", "exit_time", "duration_hours",
                    "entry_price", "exit_price", "return_pct",
                    "mae_pct", "mfe_pct", "exit_reason"]
        for field in required:
            assert field in rec, f"Missing field: {field}"

    def test_return_pct_is_correct(self):
        p = self._portfolio(70000.0)
        rec = _build_trade_record(p, 71400.0, "TAKE_PROFIT")
        assert rec["return_pct"] == pytest.approx(2.0, abs=0.01)

    def test_mae_mfe_values(self):
        p = self._portfolio(70000.0)
        rec = _build_trade_record(p, 71400.0, "TAKE_PROFIT")
        assert rec["mfe_pct"] == pytest.approx(2.0, abs=0.01)
        assert rec["mae_pct"] == pytest.approx(-1.5, abs=0.01)

    def test_exit_reason_stored(self):
        p = self._portfolio()
        rec = _build_trade_record(p, 69000.0, "STOP_LOSS")
        assert rec["exit_reason"] == "STOP_LOSS"

    def test_price_path_included(self):
        p = self._portfolio()
        rec = _build_trade_record(p, 71400.0, "TAKE_PROFIT")
        assert "_price_path" in rec
        assert len(rec["_price_path"]) == 1
