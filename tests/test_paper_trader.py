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
    compute_dynamic_stops,
    execute_entry,
    execute_exit,
)
from src.trading.paper_trader import (
    _build_trade_record,
    _init_trade_tracking,
    _update_excursions,
    acquire_lock,
    check_reversal_filter,
    check_stops_only,
    release_lock,
    run_cycle,
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


# ---------------------------------------------------------------------------
# check_stops_only
# ---------------------------------------------------------------------------

class TestCheckStopsOnly:
    def _open_portfolio(self, entry=70000.0):
        return {
            "has_position": True,
            "entry_price": entry,
            "entry_time": "2026-04-16T10:00:00+00:00",
            "quantity": 0.142857,
            "capital_usd": 10000.0,
            "trailing_high": entry,
            "stop_loss_price": round(entry * 0.97, 2),
            "take_profit_price": round(entry * 1.02, 2),
            "last_updated": None,
            "trade_id": "test-uuid-stops",
            "max_favorable": 0.0,
            "max_adverse": 0.0,
            "mfe_time": None,
            "mae_time": None,
            "price_path": [],
            "entry_score_raw": 3.5,
            "entry_score_adjusted": 3.5,
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

    def _no_position_portfolio(self):
        return {
            "has_position": False,
            "entry_price": None,
            "capital_usd": 10000.0,
        }

    def test_check_stops_no_position(self):
        with patch("src.trading.paper_trader.load_portfolio", return_value=self._no_position_portfolio()):
            result = check_stops_only()
        assert result["action"] == "no_position"

    def test_check_stops_hold(self, tmp_path):
        portfolio = self._open_portfolio(70000.0)
        with (
            patch("src.trading.paper_trader.load_portfolio", return_value=portfolio),
            patch("src.features.technical.get_live_price", return_value=70500.0),
            patch("src.trading.paper_trader._update_excursions"),
            patch("src.trading.paper_trader.check_stops", return_value=(False, "")),
            patch("src.trading.execution.get_path", return_value=tmp_path / "portfolio_state.json"),
        ):
            result = check_stops_only()
        assert result["action"] == "hold"
        assert result["price"] == 70500.0

    def test_check_stops_stop_gain(self, tmp_path):
        portfolio = self._open_portfolio(70000.0)
        completed = {**portfolio, "_price_path": []}
        completed["entry_price"] = 70000.0
        with (
            patch("src.trading.paper_trader.load_portfolio", return_value=portfolio),
            patch("src.trading.paper_trader.get_latest_technical", return_value={"close": 71500.0}),
            patch("src.trading.paper_trader._update_excursions"),
            patch("src.trading.paper_trader.check_stops", return_value=(True, "TAKE_PROFIT")),
            patch("src.trading.paper_trader._build_trade_record", return_value={**completed}),
            patch("src.trading.paper_trader.execute_exit", return_value={**portfolio, "has_position": False}),
            patch("src.trading.paper_trader._save_completed_trade"),
        ):
            result = check_stops_only()
        assert result["action"] == "exit"
        assert result["reason"] == "TAKE_PROFIT"

    def test_check_stops_stop_loss(self, tmp_path):
        portfolio = self._open_portfolio(70000.0)
        completed = {**portfolio, "_price_path": [], "entry_price": 70000.0}
        with (
            patch("src.trading.paper_trader.load_portfolio", return_value=portfolio),
            patch("src.trading.paper_trader.get_latest_technical", return_value={"close": 67900.0}),
            patch("src.trading.paper_trader._update_excursions"),
            patch("src.trading.paper_trader.check_stops", return_value=(True, "STOP_LOSS")),
            patch("src.trading.paper_trader._build_trade_record", return_value={**completed}),
            patch("src.trading.paper_trader.execute_exit", return_value={**portfolio, "has_position": False}),
            patch("src.trading.paper_trader._save_completed_trade"),
        ):
            result = check_stops_only()
        assert result["action"] == "exit"
        assert result["reason"] == "STOP_LOSS"

    def test_check_stops_trailing(self, tmp_path):
        portfolio = self._open_portfolio(70000.0)
        portfolio["trailing_high"] = 72000.0
        completed = {**portfolio, "_price_path": [], "entry_price": 70000.0}
        with (
            patch("src.trading.paper_trader.load_portfolio", return_value=portfolio),
            patch("src.trading.paper_trader.get_latest_technical", return_value={"close": 70900.0}),
            patch("src.trading.paper_trader._update_excursions"),
            patch("src.trading.paper_trader.check_stops", return_value=(True, "STOP_LOSS")),
            patch("src.trading.paper_trader._build_trade_record", return_value={**completed}),
            patch("src.trading.paper_trader.execute_exit", return_value={**portfolio, "has_position": False}),
            patch("src.trading.paper_trader._save_completed_trade"),
        ):
            result = check_stops_only()
        assert result["action"] == "exit"
        assert result["reason"] == "STOP_LOSS"

    def test_check_stops_updates_trailing_high(self, tmp_path):
        portfolio = self._open_portfolio(70000.0)
        updated_portfolio = {**portfolio, "trailing_high": 71000.0}
        with (
            patch("src.trading.paper_trader.load_portfolio", return_value=portfolio),
            patch("src.trading.paper_trader.get_latest_technical", return_value={"close": 71000.0}),
            patch("src.trading.paper_trader._update_excursions"),
            patch("src.trading.paper_trader.check_stops", return_value=(False, "")) as mock_cs,
        ):
            mock_cs.side_effect = lambda price, p: (
                p.update({"trailing_high": 71000.0}) or (False, "")
            )
            result = check_stops_only()
        assert result["action"] == "hold"
        assert result["trailing_high"] == 71000.0

    def test_check_stops_updates_excursions(self, tmp_path):
        portfolio = self._open_portfolio(70000.0)
        mock_update = MagicMock()
        with (
            patch("src.trading.paper_trader.load_portfolio", return_value=portfolio),
            patch("src.features.technical.get_live_price", return_value=70500.0),
            patch("src.trading.paper_trader._update_excursions", mock_update),
            patch("src.trading.paper_trader.check_stops", return_value=(False, "")),
        ):
            check_stops_only()
        mock_update.assert_called_once()
        _, args, _ = mock_update.mock_calls[0]
        assert args[1] == 70500.0  # current_price passed correctly

    def test_check_stops_api_error(self):
        """Live API fails AND parquet raises → error returned."""
        portfolio = self._open_portfolio(70000.0)
        with (
            patch("src.trading.paper_trader.load_portfolio", return_value=portfolio),
            patch("src.features.technical.get_live_price", return_value=None),
            patch("src.trading.paper_trader.get_latest_technical", side_effect=Exception("timeout")),
        ):
            result = check_stops_only()
        assert result["action"] == "error"
        assert "timeout" in result["error"]

    def test_check_stops_close_none_is_error(self):
        """Live API fails AND parquet returns close=None → error returned."""
        portfolio = self._open_portfolio(70000.0)
        with (
            patch("src.trading.paper_trader.load_portfolio", return_value=portfolio),
            patch("src.features.technical.get_live_price", return_value=None),
            patch("src.trading.paper_trader.get_latest_technical", return_value={"close": None}),
        ):
            result = check_stops_only()
        assert result["action"] == "error"


# ---------------------------------------------------------------------------
# Lock mechanism
# ---------------------------------------------------------------------------

class TestLockMechanism:
    def test_acquire_release_lock(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src.trading.paper_trader._LOCK_FILE", str(tmp_path / "test.lock"))
        lock = acquire_lock()
        assert lock is not None
        release_lock(lock)

    def test_lock_conflict(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src.trading.paper_trader._LOCK_FILE", str(tmp_path / "test.lock"))
        lock1 = acquire_lock()
        assert lock1 is not None
        try:
            lock2 = acquire_lock()
            assert lock2 is None  # second acquire must fail (lock busy)
        finally:
            release_lock(lock1)


# ---------------------------------------------------------------------------
# Dynamic stops — compute_dynamic_stops
# ---------------------------------------------------------------------------

def _base_exec_params():
    return {
        "use_dynamic_stops": True,
        "atr_multiplier_sl": 2.0,
        "atr_multiplier_tp": 1.5,
        "atr_multiplier_trail": 1.0,
        "min_stop_loss_pct":   0.01,
        "max_stop_loss_pct":   0.06,
        "min_take_profit_pct": 0.01,
        "max_take_profit_pct": 0.06,
        "min_trailing_pct":    0.008,
        "max_trailing_pct":    0.04,
        "stop_loss_pct":    0.03,
        "take_profit_pct":  0.02,
        "trailing_stop_pct": 0.015,
        "position_size_pct": 1.0,
        "paper_capital_usd": 10000.0,
    }


class TestComputeDynamicStops:
    def test_dynamic_stops_normal(self):
        # BTC $75,000, ATR $900 → ATR% = 1.2%
        r = compute_dynamic_stops(75000.0, 900.0, _base_exec_params())
        assert r["stops_mode"] == "dynamic"
        assert r["atr_pct"] == pytest.approx(0.012, abs=1e-6)
        assert r["stop_loss_pct"] == pytest.approx(2.0 * 0.012, abs=1e-6)
        assert r["take_profit_pct"] == pytest.approx(1.5 * 0.012, abs=1e-6)
        assert r["trailing_stop_pct"] == pytest.approx(1.0 * 0.012, abs=1e-6)
        assert r["stop_loss_price"] == pytest.approx(75000.0 * (1 - 0.024), abs=1.0)
        assert r["take_profit_price"] == pytest.approx(75000.0 * (1 + 0.018), abs=1.0)

    def test_dynamic_stops_high_volatility(self):
        # ATR doubles → stops wider but clamped at max 6%
        r = compute_dynamic_stops(75000.0, 1800.0, _base_exec_params())
        assert r["stop_loss_pct"] == pytest.approx(2.0 * 0.024, abs=1e-5)  # 4.8%, within max
        assert r["stop_loss_pct"] <= 0.06

    def test_dynamic_stops_low_volatility(self):
        # Tiny ATR → stops clamped at min
        r = compute_dynamic_stops(75000.0, 50.0, _base_exec_params())  # 0.067% ATR
        assert r["stop_loss_pct"] == pytest.approx(0.01, abs=1e-6)   # clamped at min 1%
        assert r["take_profit_pct"] == pytest.approx(0.01, abs=1e-6)
        assert r["trailing_stop_pct"] == pytest.approx(0.008, abs=1e-6)

    def test_dynamic_stops_clamp_max(self):
        # ATR $4,500 → raw SL = 2.0 × 6% = 12% → clamped to 6%
        r = compute_dynamic_stops(75000.0, 4500.0, _base_exec_params())
        assert r["stop_loss_pct"] == pytest.approx(0.06, abs=1e-6)
        assert r["take_profit_pct"] == pytest.approx(0.06, abs=1e-6)
        assert r["trailing_stop_pct"] == pytest.approx(0.04, abs=1e-6)

    def test_dynamic_stops_clamp_min(self):
        # ATR $1 → all clamped to mins
        r = compute_dynamic_stops(75000.0, 1.0, _base_exec_params())
        assert r["stop_loss_pct"] == pytest.approx(0.01, abs=1e-6)
        assert r["trailing_stop_pct"] == pytest.approx(0.008, abs=1e-6)


class TestExecuteEntryDynamic:
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

    def test_entry_with_atr_dynamic(self, tmp_path):
        with (
            patch("src.trading.execution.get_path", return_value=tmp_path / "p.json"),
            patch("src.trading.execution.get_params", return_value={"execution": _base_exec_params()}),
        ):
            p = execute_entry(75000.0, self._portfolio(), atr_14=900.0)
        assert p["stops_mode"] == "dynamic"
        assert p["entry_atr_pct"] == pytest.approx(0.012, abs=1e-6)
        # TP = entry * (1 + 1.5 × 1.2%) = 75000 * 1.018 = 76350
        assert p["take_profit_price"] == pytest.approx(76350.0, abs=1.0)
        # SL = entry * (1 - 2.0 × 1.2%) = 75000 * 0.976 = 73200
        assert p["stop_loss_price"] == pytest.approx(73200.0, abs=1.0)

    def test_entry_without_atr_fallback(self, tmp_path):
        with (
            patch("src.trading.execution.get_path", return_value=tmp_path / "p.json"),
            patch("src.trading.execution.get_params", return_value={"execution": _base_exec_params()}),
        ):
            p = execute_entry(75000.0, self._portfolio(), atr_14=None)
        assert p["stops_mode"] == "fixed"
        assert p["entry_atr_pct"] is None
        assert p["stop_loss_price"] == pytest.approx(75000.0 * 0.97, abs=1.0)

    def test_entry_dynamic_disabled(self, tmp_path):
        params = {**_base_exec_params(), "use_dynamic_stops": False}
        with (
            patch("src.trading.execution.get_path", return_value=tmp_path / "p.json"),
            patch("src.trading.execution.get_params", return_value={"execution": params}),
        ):
            p = execute_entry(75000.0, self._portfolio(), atr_14=900.0)
        assert p["stops_mode"] == "fixed"
        assert p["stop_loss_price"] == pytest.approx(75000.0 * 0.97, abs=1.0)


class TestCheckStopsDynamic:
    def _portfolio(self, entry=70000.0, trailing_pct_actual=None):
        p = {
            "has_position": True,
            "entry_price": entry,
            "quantity": 0.142857,
            "capital_usd": 10000.0,
            "trailing_high": entry,
            "stop_loss_price": round(entry * 0.97, 2),
            "take_profit_price": round(entry * 1.02, 2),
            "entry_time": "2026-04-16T10:00:00+00:00",
            "last_updated": None,
        }
        if trailing_pct_actual is not None:
            p["trailing_stop_pct_actual"] = trailing_pct_actual
        return p

    def test_check_stops_uses_portfolio_trailing(self, tmp_path):
        # trailing_stop_pct_actual = 0.024 (ATR dynamic, wider than default 0.015)
        p = self._portfolio(70000.0, trailing_pct_actual=0.024)
        # Price = 71000 (just above entry, below TP 71400) → trailing_high moves to 71000
        # Trailing stop = 71000 * (1 - 0.024) = 69296 > original SL 67900 → moves up
        with patch("src.trading.execution.get_path", return_value=tmp_path / "p.json"):
            triggered, reason = check_stops(71000.0, p)
        assert triggered is False
        assert p["stop_loss_price"] == pytest.approx(71000.0 * (1 - 0.024), abs=1.0)

    def test_check_stops_trailing_fallback(self, tmp_path):
        # No trailing_stop_pct_actual in portfolio → falls back to global param (0.015)
        p = self._portfolio(70000.0)  # no trailing_pct_actual
        with (
            patch("src.trading.execution.get_path", return_value=tmp_path / "p.json"),
            patch("src.trading.execution.get_params", return_value={
                "execution": _base_exec_params()
            }),
        ):
            triggered, _ = check_stops(71000.0, p)
        assert triggered is False
        # Trailing high moved to 71000; new SL = 71000 * (1 - 0.015) = 69935
        assert p["stop_loss_price"] == pytest.approx(71000.0 * (1 - 0.015), abs=1.0)


class TestIntegrationDynamicStops:
    def test_full_cycle_dynamic_stops(self, tmp_path):
        """Entry with ATR → dynamic stops set → check_stops uses portfolio trailing → exit records mode."""
        portfolio = {
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
        params = _base_exec_params()

        # 1. Entry with ATR
        with (
            patch("src.trading.execution.get_path", return_value=tmp_path / "p.json"),
            patch("src.trading.execution.get_params", return_value={"execution": params}),
        ):
            portfolio = execute_entry(75000.0, portfolio, atr_14=900.0)

        assert portfolio["stops_mode"] == "dynamic"
        assert portfolio["trailing_stop_pct_actual"] == pytest.approx(0.012, abs=1e-6)

        # 2. check_stops uses portfolio trailing (0.012), not global (0.015)
        portfolio["entry_time"] = "2026-04-16T10:00:00+00:00"
        with patch("src.trading.execution.get_path", return_value=tmp_path / "p.json"):
            triggered, reason = check_stops(75800.0, portfolio)  # price up, trailing moves
        assert triggered is False
        # Trailing stop = 75800 * (1 - 0.012) = 74891.6
        assert portfolio["stop_loss_price"] == pytest.approx(75800.0 * (1 - 0.012), abs=1.0)

        # 3. TP hit → exit
        with (
            patch("src.trading.execution.get_path", return_value=tmp_path / "p.json"),
        ):
            triggered, reason = check_stops(portfolio["take_profit_price"] + 1, portfolio)
        assert triggered is True
        assert reason == "TAKE_PROFIT"


# ---------------------------------------------------------------------------
# TestCheckReversalFilter — unit tests for check_reversal_filter()
# ---------------------------------------------------------------------------

def _rf_params(enabled=True, rsi_max=35, ret_1d_min=-0.01, rsi_extreme_override=25):
    return {
        "reversal_filter": {
            "enabled": enabled,
            "rsi_max": rsi_max,
            "ret_1d_min": ret_1d_min,
            "rsi_extreme_override": rsi_extreme_override,
        }
    }


class TestCheckReversalFilter:
    def test_filter_disabled(self):
        r = check_reversal_filter({"rsi_14": 50, "ret_1d": -0.05}, _rf_params(enabled=False))
        assert r["passed"] is True
        assert r["reason"] == "filter_disabled"

    def test_filter_passes_all(self):
        r = check_reversal_filter({"rsi_14": 28.0, "ret_1d": -0.003}, _rf_params())
        assert r["passed"] is True
        assert r["reason"] == "reversal_confirmed"

    def test_filter_rsi_too_high(self):
        r = check_reversal_filter({"rsi_14": 42.0, "ret_1d": -0.003}, _rf_params())
        assert r["passed"] is False
        assert "RSI_TOO_HIGH" in r["reason"]

    def test_filter_falling_knife(self):
        r = check_reversal_filter({"rsi_14": 28.0, "ret_1d": -0.015}, _rf_params())
        assert r["passed"] is False
        assert "FALLING_KNIFE" in r["reason"]

    def test_filter_both_fail(self):
        r = check_reversal_filter({"rsi_14": 42.0, "ret_1d": -0.015}, _rf_params())
        assert r["passed"] is False
        assert "RSI_TOO_HIGH" in r["reason"]
        assert "FALLING_KNIFE" in r["reason"]
        assert " & " in r["reason"]

    def test_filter_rsi_exactly_35(self):
        # condition is rsi < rsi_max (strictly), so 35.0 should block
        r = check_reversal_filter({"rsi_14": 35.0, "ret_1d": -0.003}, _rf_params())
        assert r["passed"] is False
        assert "RSI_TOO_HIGH" in r["reason"]

    def test_filter_ret_1d_exactly_minus_1pct(self):
        # condition is ret_1d > ret_1d_min (strictly), so -0.01 should block
        r = check_reversal_filter({"rsi_14": 28.0, "ret_1d": -0.01}, _rf_params())
        assert r["passed"] is False
        assert "FALLING_KNIFE" in r["reason"]

    def test_filter_rsi_none(self):
        r = check_reversal_filter({"rsi_14": None, "ret_1d": -0.003}, _rf_params())
        assert r["passed"] is False
        assert "FILTER_DATA_MISSING" in r["reason"]

    def test_filter_ret_1d_none(self):
        r = check_reversal_filter({"rsi_14": 28.0, "ret_1d": None}, _rf_params())
        assert r["passed"] is False
        assert "FILTER_DATA_MISSING" in r["reason"]

    def test_filter_edge_case_rsi_34_9(self):
        r = check_reversal_filter({"rsi_14": 34.9, "ret_1d": -0.003}, _rf_params())
        assert r["passed"] is True

    def test_filter_edge_case_ret_minus_0_9pct(self):
        r = check_reversal_filter({"rsi_14": 28.0, "ret_1d": -0.009}, _rf_params())
        assert r["passed"] is True

    def test_filter_extreme_capitulation_override(self):
        # RSI=22 < rsi_extreme_override=25 → overrides ret_1d check
        r = check_reversal_filter({"rsi_14": 22.0, "ret_1d": -0.02}, _rf_params(rsi_extreme_override=25))
        assert r["passed"] is True

    def test_filter_extreme_override_disabled(self):
        # rsi_extreme_override=0 → override disabled, ret_1d still blocks
        r = check_reversal_filter({"rsi_14": 22.0, "ret_1d": -0.02}, _rf_params(rsi_extreme_override=0))
        assert r["passed"] is False
        assert "FALLING_KNIFE" in r["reason"]

    def test_filter_rsi_26_no_override(self):
        # RSI=26 >= rsi_extreme_override=25 → no override
        r = check_reversal_filter({"rsi_14": 26.0, "ret_1d": -0.02}, _rf_params(rsi_extreme_override=25))
        assert r["passed"] is False
        assert "FALLING_KNIFE" in r["reason"]


# ---------------------------------------------------------------------------
# TestRunCycleWithFilter — integration tests for run_cycle() with filter
# ---------------------------------------------------------------------------

def _base_portfolio():
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
        "last_signal": "HOLD",
        "last_score": 0.0,
        "last_threshold": 3.5,
        "last_regime": "Sideways",
        "updated_at": "2026-04-17T00:00:00",
    }


def _enter_result(score=3.5):
    return {
        "signal": "ENTER",
        "score": score,
        "score_raw": score,
        "threshold": 2.8,
        "block_reason": None,
        "gate_scores": {},
        "clusters": {},
        "proximity_adj": 0.0,
        "regime_multiplier": 1.0,
    }


def _hold_result():
    return {**_enter_result(), "signal": "HOLD"}


def _block_result():
    return {**_enter_result(), "signal": "BLOCK", "block_reason": "BLOCK_BEAR_REGIME"}


def _common_patches(tmp_path, portfolio, result, technical, params=None):
    """Build context manager patches for run_cycle integration tests."""
    if params is None:
        params = {
            "reversal_filter": {"enabled": True, "rsi_max": 35, "ret_1d_min": -0.01, "rsi_extreme_override": 25},
            "news": {"lookback_hours": 4},
            "execution": _base_exec_params(),
        }
    return [
        patch("src.trading.paper_trader.get_current_regime", return_value={"regime": "Sideways"}),
        patch("src.trading.paper_trader.get_latest_technical", return_value=technical),
        patch("src.trading.paper_trader.load_latest_zscores", return_value={}),
        patch("src.trading.paper_trader.compute_stale_days", return_value={}),
        patch("src.trading.paper_trader.load_news_crypto_score", return_value=0.0),
        patch("src.trading.paper_trader.get_fed_context", return_value={"fed_score": 0.0, "proximity_adjustment": 0.0, "is_blackout": False}),
        patch("src.trading.paper_trader.load_score_history", return_value=[]),
        patch("src.trading.paper_trader.run_scoring_pipeline", return_value=result),
        patch("src.trading.paper_trader.load_portfolio", return_value=portfolio),
        patch("src.trading.paper_trader.get_params", return_value=params),
        patch("src.trading.paper_trader.append_score_history"),
        patch("src.trading.paper_trader.log_cycle"),
        patch("src.trading.paper_trader.get_path", return_value=tmp_path / "portfolio_state.json"),
        patch("src.trading.execution.get_path", return_value=tmp_path / "portfolio_state.json"),
        patch("src.trading.execution.get_params", return_value={"execution": _base_exec_params()}),
        patch("src.trading.paper_trader.atomic_write_json"),
        patch("builtins.open", side_effect=lambda *a, **k: open(*a, **k) if str(a[0]).endswith(".parquet") else MagicMock()),
    ]


class TestRunCycleWithFilter:
    def _open_patches(self, tmp_path, portfolio, result, technical, params=None):
        patches = _common_patches(tmp_path, portfolio, result, technical, params)
        # Remove builtins open patch that breaks things — just patch the spot_df load
        return patches[:-1] + [patch("pandas.read_parquet", side_effect=Exception("no parquet"))]

    def test_entry_with_filter_pass(self, tmp_path):
        portfolio = _base_portfolio()
        result = _enter_result()
        tech = {"close": 75000.0, "rsi_14": 28.0, "ret_1d": -0.003, "bb_pct": 0.15, "atr_14": 900.0}
        patches = [
            patch("src.trading.paper_trader.get_current_regime", return_value={"regime": "Sideways"}),
            patch("src.trading.paper_trader.get_latest_technical", return_value=tech),
            patch("src.trading.paper_trader.load_latest_zscores", return_value={}),
            patch("src.trading.paper_trader.compute_stale_days", return_value={}),
            patch("src.trading.paper_trader.load_news_crypto_score", return_value=0.0),
            patch("src.trading.paper_trader.get_fed_context", return_value={"fed_score": 0.0, "proximity_adjustment": 0.0, "is_blackout": False}),
            patch("src.trading.paper_trader.load_score_history", return_value=[]),
            patch("src.trading.paper_trader.run_scoring_pipeline", return_value=result),
            patch("src.trading.paper_trader.load_portfolio", return_value=portfolio),
            patch("src.trading.paper_trader.get_params", return_value={
                "reversal_filter": {"enabled": True, "rsi_max": 35, "ret_1d_min": -0.01, "rsi_extreme_override": 25},
                "news": {"lookback_hours": 4},
                "execution": _base_exec_params(),
            }),
            patch("src.trading.paper_trader.append_score_history"),
            patch("src.trading.paper_trader.log_cycle"),
            patch("src.trading.paper_trader.get_path", return_value=tmp_path / "p.json"),
            patch("src.trading.execution.get_path", return_value=tmp_path / "p.json"),
            patch("src.trading.execution.get_params", return_value={"execution": _base_exec_params()}),
            patch("src.trading.paper_trader.atomic_write_json"),
            patch("pandas.read_parquet", side_effect=Exception("no parquet")),
        ]
        mock_entry = MagicMock(return_value={**portfolio, "has_position": True, "entry_price": 75000.0})
        patches.append(patch("src.trading.paper_trader.execute_entry", mock_entry))

        with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], patches[6], patches[7], patches[8], patches[9], patches[10], patches[11], patches[12], patches[13], patches[14], patches[15], patches[16], patches[17]:
            r = run_cycle()

        assert r["signal"] == "ENTER"
        mock_entry.assert_called_once()

    def test_entry_with_filter_blocked_rsi(self, tmp_path):
        portfolio = _base_portfolio()
        result = _enter_result()
        tech = {"close": 75000.0, "rsi_14": 42.0, "ret_1d": -0.003, "bb_pct": 0.15, "atr_14": 900.0}
        patches = [
            patch("src.trading.paper_trader.get_current_regime", return_value={"regime": "Sideways"}),
            patch("src.trading.paper_trader.get_latest_technical", return_value=tech),
            patch("src.trading.paper_trader.load_latest_zscores", return_value={}),
            patch("src.trading.paper_trader.compute_stale_days", return_value={}),
            patch("src.trading.paper_trader.load_news_crypto_score", return_value=0.0),
            patch("src.trading.paper_trader.get_fed_context", return_value={"fed_score": 0.0, "proximity_adjustment": 0.0, "is_blackout": False}),
            patch("src.trading.paper_trader.load_score_history", return_value=[]),
            patch("src.trading.paper_trader.run_scoring_pipeline", return_value=result),
            patch("src.trading.paper_trader.load_portfolio", return_value=portfolio),
            patch("src.trading.paper_trader.get_params", return_value={
                "reversal_filter": {"enabled": True, "rsi_max": 35, "ret_1d_min": -0.01, "rsi_extreme_override": 25},
                "news": {"lookback_hours": 4},
                "execution": _base_exec_params(),
            }),
            patch("src.trading.paper_trader.append_score_history"),
            patch("src.trading.paper_trader.log_cycle"),
            patch("src.trading.paper_trader.get_path", return_value=tmp_path / "p.json"),
            patch("src.trading.execution.get_path", return_value=tmp_path / "p.json"),
            patch("src.trading.execution.get_params", return_value={"execution": _base_exec_params()}),
            patch("src.trading.paper_trader.atomic_write_json"),
            patch("pandas.read_parquet", side_effect=Exception("no parquet")),
        ]
        with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], patches[6], patches[7], patches[8], patches[9], patches[10], patches[11], patches[12], patches[13], patches[14], patches[15], patches[16]:
            r = run_cycle()

        assert r["signal"] == "FILTERED"
        assert "RSI_TOO_HIGH" in r["filter_reason"]
        assert portfolio["has_position"] is False

    def test_entry_with_filter_blocked_ret(self, tmp_path):
        portfolio = _base_portfolio()
        result = _enter_result()
        tech = {"close": 75000.0, "rsi_14": 28.0, "ret_1d": -0.015, "bb_pct": 0.15, "atr_14": 900.0}
        patches = [
            patch("src.trading.paper_trader.get_current_regime", return_value={"regime": "Sideways"}),
            patch("src.trading.paper_trader.get_latest_technical", return_value=tech),
            patch("src.trading.paper_trader.load_latest_zscores", return_value={}),
            patch("src.trading.paper_trader.compute_stale_days", return_value={}),
            patch("src.trading.paper_trader.load_news_crypto_score", return_value=0.0),
            patch("src.trading.paper_trader.get_fed_context", return_value={"fed_score": 0.0, "proximity_adjustment": 0.0, "is_blackout": False}),
            patch("src.trading.paper_trader.load_score_history", return_value=[]),
            patch("src.trading.paper_trader.run_scoring_pipeline", return_value=result),
            patch("src.trading.paper_trader.load_portfolio", return_value=portfolio),
            patch("src.trading.paper_trader.get_params", return_value={
                "reversal_filter": {"enabled": True, "rsi_max": 35, "ret_1d_min": -0.01, "rsi_extreme_override": 25},
                "news": {"lookback_hours": 4},
                "execution": _base_exec_params(),
            }),
            patch("src.trading.paper_trader.append_score_history"),
            patch("src.trading.paper_trader.log_cycle"),
            patch("src.trading.paper_trader.get_path", return_value=tmp_path / "p.json"),
            patch("src.trading.execution.get_path", return_value=tmp_path / "p.json"),
            patch("src.trading.execution.get_params", return_value={"execution": _base_exec_params()}),
            patch("src.trading.paper_trader.atomic_write_json"),
            patch("pandas.read_parquet", side_effect=Exception("no parquet")),
        ]
        with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], patches[6], patches[7], patches[8], patches[9], patches[10], patches[11], patches[12], patches[13], patches[14], patches[15], patches[16]:
            r = run_cycle()

        assert r["signal"] == "FILTERED"
        assert "FALLING_KNIFE" in r["filter_reason"]

    def test_entry_with_filter_disabled(self, tmp_path):
        portfolio = _base_portfolio()
        result = _enter_result()
        tech = {"close": 75000.0, "rsi_14": 50.0, "ret_1d": -0.05, "bb_pct": 0.15, "atr_14": 900.0}
        mock_entry = MagicMock(return_value={**portfolio, "has_position": True, "entry_price": 75000.0})
        patches = [
            patch("src.trading.paper_trader.get_current_regime", return_value={"regime": "Sideways"}),
            patch("src.trading.paper_trader.get_latest_technical", return_value=tech),
            patch("src.trading.paper_trader.load_latest_zscores", return_value={}),
            patch("src.trading.paper_trader.compute_stale_days", return_value={}),
            patch("src.trading.paper_trader.load_news_crypto_score", return_value=0.0),
            patch("src.trading.paper_trader.get_fed_context", return_value={"fed_score": 0.0, "proximity_adjustment": 0.0, "is_blackout": False}),
            patch("src.trading.paper_trader.load_score_history", return_value=[]),
            patch("src.trading.paper_trader.run_scoring_pipeline", return_value=result),
            patch("src.trading.paper_trader.load_portfolio", return_value=portfolio),
            patch("src.trading.paper_trader.get_params", return_value={
                "reversal_filter": {"enabled": False, "rsi_max": 35, "ret_1d_min": -0.01, "rsi_extreme_override": 25},
                "news": {"lookback_hours": 4},
                "execution": _base_exec_params(),
            }),
            patch("src.trading.paper_trader.append_score_history"),
            patch("src.trading.paper_trader.log_cycle"),
            patch("src.trading.paper_trader.get_path", return_value=tmp_path / "p.json"),
            patch("src.trading.execution.get_path", return_value=tmp_path / "p.json"),
            patch("src.trading.execution.get_params", return_value={"execution": _base_exec_params()}),
            patch("src.trading.paper_trader.atomic_write_json"),
            patch("pandas.read_parquet", side_effect=Exception("no parquet")),
            patch("src.trading.paper_trader.execute_entry", mock_entry),
        ]
        with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], patches[6], patches[7], patches[8], patches[9], patches[10], patches[11], patches[12], patches[13], patches[14], patches[15], patches[16], patches[17]:
            r = run_cycle()

        assert r["signal"] == "ENTER"
        mock_entry.assert_called_once()

    def test_hold_signal_unaffected(self, tmp_path):
        portfolio = _base_portfolio()
        result = _hold_result()
        tech = {"close": 75000.0, "rsi_14": 28.0, "ret_1d": -0.003, "bb_pct": 0.15, "atr_14": 900.0}
        mock_entry = MagicMock()
        patches = [
            patch("src.trading.paper_trader.get_current_regime", return_value={"regime": "Sideways"}),
            patch("src.trading.paper_trader.get_latest_technical", return_value=tech),
            patch("src.trading.paper_trader.load_latest_zscores", return_value={}),
            patch("src.trading.paper_trader.compute_stale_days", return_value={}),
            patch("src.trading.paper_trader.load_news_crypto_score", return_value=0.0),
            patch("src.trading.paper_trader.get_fed_context", return_value={"fed_score": 0.0, "proximity_adjustment": 0.0, "is_blackout": False}),
            patch("src.trading.paper_trader.load_score_history", return_value=[]),
            patch("src.trading.paper_trader.run_scoring_pipeline", return_value=result),
            patch("src.trading.paper_trader.load_portfolio", return_value=portfolio),
            patch("src.trading.paper_trader.get_params", return_value={
                "reversal_filter": {"enabled": True, "rsi_max": 35, "ret_1d_min": -0.01, "rsi_extreme_override": 25},
                "news": {"lookback_hours": 4},
                "execution": _base_exec_params(),
            }),
            patch("src.trading.paper_trader.append_score_history"),
            patch("src.trading.paper_trader.log_cycle"),
            patch("src.trading.paper_trader.get_path", return_value=tmp_path / "p.json"),
            patch("src.trading.execution.get_path", return_value=tmp_path / "p.json"),
            patch("src.trading.execution.get_params", return_value={"execution": _base_exec_params()}),
            patch("src.trading.paper_trader.atomic_write_json"),
            patch("pandas.read_parquet", side_effect=Exception("no parquet")),
            patch("src.trading.paper_trader.execute_entry", mock_entry),
        ]
        with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], patches[6], patches[7], patches[8], patches[9], patches[10], patches[11], patches[12], patches[13], patches[14], patches[15], patches[16], patches[17]:
            r = run_cycle()

        assert r["signal"] == "HOLD"
        mock_entry.assert_not_called()

    def test_block_signal_unaffected(self, tmp_path):
        portfolio = _base_portfolio()
        result = _block_result()
        tech = {"close": 75000.0, "rsi_14": 28.0, "ret_1d": -0.003, "bb_pct": 0.15, "atr_14": 900.0}
        mock_entry = MagicMock()
        patches = [
            patch("src.trading.paper_trader.get_current_regime", return_value={"regime": "Bear"}),
            patch("src.trading.paper_trader.get_latest_technical", return_value=tech),
            patch("src.trading.paper_trader.load_latest_zscores", return_value={}),
            patch("src.trading.paper_trader.compute_stale_days", return_value={}),
            patch("src.trading.paper_trader.load_news_crypto_score", return_value=0.0),
            patch("src.trading.paper_trader.get_fed_context", return_value={"fed_score": 0.0, "proximity_adjustment": 0.0, "is_blackout": False}),
            patch("src.trading.paper_trader.load_score_history", return_value=[]),
            patch("src.trading.paper_trader.run_scoring_pipeline", return_value=result),
            patch("src.trading.paper_trader.load_portfolio", return_value=portfolio),
            patch("src.trading.paper_trader.get_params", return_value={
                "reversal_filter": {"enabled": True, "rsi_max": 35, "ret_1d_min": -0.01, "rsi_extreme_override": 25},
                "news": {"lookback_hours": 4},
                "execution": _base_exec_params(),
            }),
            patch("src.trading.paper_trader.append_score_history"),
            patch("src.trading.paper_trader.log_cycle"),
            patch("src.trading.paper_trader.get_path", return_value=tmp_path / "p.json"),
            patch("src.trading.execution.get_path", return_value=tmp_path / "p.json"),
            patch("src.trading.execution.get_params", return_value={"execution": _base_exec_params()}),
            patch("src.trading.paper_trader.atomic_write_json"),
            patch("pandas.read_parquet", side_effect=Exception("no parquet")),
            patch("src.trading.paper_trader.execute_entry", mock_entry),
        ]
        with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], patches[6], patches[7], patches[8], patches[9], patches[10], patches[11], patches[12], patches[13], patches[14], patches[15], patches[16], patches[17]:
            r = run_cycle()

        assert r["signal"] == "BLOCK"
        mock_entry.assert_not_called()

    def test_filter_context_saved(self):
        portfolio = _base_portfolio()
        result = _enter_result()
        tech = {"close": 75000.0, "rsi_14": 28.0, "ret_1d": -0.003, "bb_pct": 0.15, "atr_14": None}
        entered_portfolio = {**portfolio, "has_position": True, "entry_price": 75000.0, "quantity": 0.133}

        mock_entry = MagicMock(return_value=entered_portfolio)
        saved_portfolios = []
        mock_atomic = MagicMock(side_effect=lambda p, path: saved_portfolios.append(dict(p)))

        with (
            patch("src.trading.paper_trader.get_current_regime", return_value={"regime": "Sideways"}),
            patch("src.trading.paper_trader.get_latest_technical", return_value=tech),
            patch("src.trading.paper_trader.load_latest_zscores", return_value={}),
            patch("src.trading.paper_trader.compute_stale_days", return_value={}),
            patch("src.trading.paper_trader.load_news_crypto_score", return_value=0.0),
            patch("src.trading.paper_trader.get_fed_context", return_value={"fed_score": 0.0, "proximity_adjustment": 0.0, "is_blackout": False}),
            patch("src.trading.paper_trader.load_score_history", return_value=[]),
            patch("src.trading.paper_trader.run_scoring_pipeline", return_value=result),
            patch("src.trading.paper_trader.load_portfolio", return_value=portfolio),
            patch("src.trading.paper_trader.get_params", return_value={
                "reversal_filter": {"enabled": True, "rsi_max": 35, "ret_1d_min": -0.01, "rsi_extreme_override": 25},
                "news": {"lookback_hours": 4},
                "execution": _base_exec_params(),
            }),
            patch("src.trading.paper_trader.append_score_history"),
            patch("src.trading.paper_trader.log_cycle"),
            patch("src.trading.paper_trader.get_path", return_value="/tmp/p.json"),
            patch("src.trading.execution.get_path", return_value="/tmp/p.json"),
            patch("src.trading.execution.get_params", return_value={"execution": _base_exec_params()}),
            patch("src.trading.paper_trader.atomic_write_json", mock_atomic),
            patch("pandas.read_parquet", side_effect=Exception("no parquet")),
            patch("src.trading.paper_trader.execute_entry", mock_entry),
        ):
            run_cycle()

        assert any(p.get("entry_filter_passed") is True for p in saved_portfolios)
        assert any(p.get("entry_ret_1d") == pytest.approx(-0.003) for p in saved_portfolios)

    def test_trade_record_has_filter_fields(self):
        portfolio = {
            "has_position": True,
            "entry_price": 75000.0,
            "entry_time": "2026-04-17T00:00:00+00:00",
            "quantity": 0.133,
            "capital_usd": 10000.0,
            "trade_id": "test-uuid",
            "max_favorable": 0.01,
            "max_adverse": -0.005,
            "mfe_time": None, "mae_time": None,
            "price_path": [],
            "entry_score_raw": 3.5, "entry_score_adjusted": 3.5, "entry_regime": "Sideways",
            "entry_bb_pct": 0.15, "entry_rsi": 28.0, "entry_atr": None,
            "entry_ret_1d": -0.003,
            "entry_filter_passed": True,
            "entry_oi_z": None, "entry_fg_raw": None,
            "entry_cluster_technical": None, "entry_cluster_positioning": None,
            "entry_cluster_macro": None, "entry_cluster_liquidity": None,
            "entry_cluster_sentiment": None, "entry_cluster_news": None,
            "entry_stop_gain_pct": 0.02, "entry_stop_loss_pct": 0.03,
            "entry_trailing_stop_pct": 0.015,
        }
        rec = _build_trade_record(portfolio, 76000.0, "TAKE_PROFIT")
        assert "entry_ret_1d" in rec
        assert rec["entry_ret_1d"] == pytest.approx(-0.003)
        assert "entry_filter_passed" in rec
        assert rec["entry_filter_passed"] is True

    def test_filter_state_persisted_on_block(self, tmp_path):
        portfolio = _base_portfolio()
        result = _enter_result()
        tech = {"close": 75000.0, "rsi_14": 42.0, "ret_1d": -0.003, "bb_pct": 0.15, "atr_14": None}
        saved_portfolios = []
        mock_atomic = MagicMock(side_effect=lambda p, path: saved_portfolios.append(dict(p)))

        with (
            patch("src.trading.paper_trader.get_current_regime", return_value={"regime": "Sideways"}),
            patch("src.trading.paper_trader.get_latest_technical", return_value=tech),
            patch("src.trading.paper_trader.load_latest_zscores", return_value={}),
            patch("src.trading.paper_trader.compute_stale_days", return_value={}),
            patch("src.trading.paper_trader.load_news_crypto_score", return_value=0.0),
            patch("src.trading.paper_trader.get_fed_context", return_value={"fed_score": 0.0, "proximity_adjustment": 0.0, "is_blackout": False}),
            patch("src.trading.paper_trader.load_score_history", return_value=[]),
            patch("src.trading.paper_trader.run_scoring_pipeline", return_value=result),
            patch("src.trading.paper_trader.load_portfolio", return_value=portfolio),
            patch("src.trading.paper_trader.get_params", return_value={
                "reversal_filter": {"enabled": True, "rsi_max": 35, "ret_1d_min": -0.01, "rsi_extreme_override": 25},
                "news": {"lookback_hours": 4},
                "execution": _base_exec_params(),
            }),
            patch("src.trading.paper_trader.append_score_history"),
            patch("src.trading.paper_trader.log_cycle"),
            patch("src.trading.paper_trader.get_path", return_value="/tmp/p.json"),
            patch("src.trading.execution.get_path", return_value="/tmp/p.json"),
            patch("src.trading.execution.get_params", return_value={"execution": _base_exec_params()}),
            patch("src.trading.paper_trader.atomic_write_json", mock_atomic),
            patch("pandas.read_parquet", side_effect=Exception("no parquet")),
        ):
            run_cycle()

        last_saved = saved_portfolios[-1]
        assert last_saved["last_filter_passed"] is False
        assert "RSI_TOO_HIGH" in (last_saved.get("last_filter_reason") or "")

    def test_filter_state_persisted_on_hold(self, tmp_path):
        portfolio = _base_portfolio()
        result = _hold_result()
        tech = {"close": 75000.0, "rsi_14": 42.0, "ret_1d": -0.003, "bb_pct": 0.15, "atr_14": None}
        saved_portfolios = []
        mock_atomic = MagicMock(side_effect=lambda p, path: saved_portfolios.append(dict(p)))

        with (
            patch("src.trading.paper_trader.get_current_regime", return_value={"regime": "Sideways"}),
            patch("src.trading.paper_trader.get_latest_technical", return_value=tech),
            patch("src.trading.paper_trader.load_latest_zscores", return_value={}),
            patch("src.trading.paper_trader.compute_stale_days", return_value={}),
            patch("src.trading.paper_trader.load_news_crypto_score", return_value=0.0),
            patch("src.trading.paper_trader.get_fed_context", return_value={"fed_score": 0.0, "proximity_adjustment": 0.0, "is_blackout": False}),
            patch("src.trading.paper_trader.load_score_history", return_value=[]),
            patch("src.trading.paper_trader.run_scoring_pipeline", return_value=result),
            patch("src.trading.paper_trader.load_portfolio", return_value=portfolio),
            patch("src.trading.paper_trader.get_params", return_value={
                "reversal_filter": {"enabled": True, "rsi_max": 35, "ret_1d_min": -0.01, "rsi_extreme_override": 25},
                "news": {"lookback_hours": 4},
                "execution": _base_exec_params(),
            }),
            patch("src.trading.paper_trader.append_score_history"),
            patch("src.trading.paper_trader.log_cycle"),
            patch("src.trading.paper_trader.get_path", return_value="/tmp/p.json"),
            patch("src.trading.execution.get_path", return_value="/tmp/p.json"),
            patch("src.trading.execution.get_params", return_value={"execution": _base_exec_params()}),
            patch("src.trading.paper_trader.atomic_write_json", mock_atomic),
            patch("pandas.read_parquet", side_effect=Exception("no parquet")),
        ):
            run_cycle()

        last_saved = saved_portfolios[-1]
        assert last_saved["last_filter_rsi"] == pytest.approx(42.0)
        assert last_saved["last_filter_ret_1d"] == pytest.approx(-0.003)
