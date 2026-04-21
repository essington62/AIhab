"""Tests for MultiAssetManager (multi-asset capital aggregator)."""
import json
from pathlib import Path

import pytest
import yaml

from src.trading.multi_asset_manager import MultiAssetManager, BucketState


@pytest.fixture
def mock_config(tmp_path):
    """Config temporária para testes."""
    portfolio_btc = tmp_path / "portfolio.json"
    portfolio_eth = tmp_path / "portfolio_eth.json"
    state_path = tmp_path / "capital_manager.json"

    config = {
        "capital_manager": {
            "enabled": True,
            "buckets": {
                "btc": {
                    "asset": "BTCUSDT",
                    "initial_capital_usd": 10000,
                    "bots_allowed": ["bot_1", "bot_2"],
                    "legacy_portfolio_path": str(portfolio_btc),
                    "enabled": True,
                },
                "eth": {
                    "asset": "ETHUSDT",
                    "initial_capital_usd": 10000,
                    "bots_allowed": ["bot_3"],
                    "legacy_portfolio_path": str(portfolio_eth),
                    "enabled": True,
                },
            },
            "global_rules": {"enabled": False},
            "state_path": str(state_path),
        }
    }

    config_path = tmp_path / "capital_manager.yml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    return {
        "config_path": config_path,
        "portfolio_btc": portfolio_btc,
        "portfolio_eth": portfolio_eth,
        "state_path": state_path,
    }


# ===========================================================
# Init
# ===========================================================

class TestInit:
    def test_loads_buckets(self, mock_config):
        cm = MultiAssetManager(config_path=mock_config["config_path"])
        assert "btc" in cm.buckets
        assert "eth" in cm.buckets

    def test_starts_at_initial_capital(self, mock_config):
        cm = MultiAssetManager(config_path=mock_config["config_path"])
        assert cm.buckets["btc"].current_capital_usd == 10000
        assert cm.buckets["eth"].current_capital_usd == 10000

    def test_bucket_assets(self, mock_config):
        cm = MultiAssetManager(config_path=mock_config["config_path"])
        assert cm.buckets["btc"].asset == "BTCUSDT"
        assert cm.buckets["eth"].asset == "ETHUSDT"


# ===========================================================
# Sync flat portfolio (ETH style)
# ===========================================================

class TestSyncFlatPortfolio:
    def test_sync_no_position(self, mock_config):
        with open(mock_config["portfolio_eth"], "w") as f:
            json.dump({"capital_usd": 9800.0, "has_position": False}, f)

        cm = MultiAssetManager(config_path=mock_config["config_path"])
        cm.sync_from_legacy()

        eth = cm.get_bucket("eth")
        assert eth.current_capital_usd == 9800.0
        assert eth.has_position is False
        assert eth.last_sync is not None

    def test_sync_with_open_position(self, mock_config):
        with open(mock_config["portfolio_eth"], "w") as f:
            json.dump({
                "capital_usd": 10000,
                "has_position": True,
                "entry_price": 2500.0,
                "quantity": 4.0,
                "stop_loss_price": 2450.0,
                "take_profit_price": 2600.0,
                "trailing_high": 2520.0,
            }, f)

        cm = MultiAssetManager(config_path=mock_config["config_path"])
        cm.sync_from_legacy()

        eth = cm.get_bucket("eth")
        assert eth.has_position is True
        assert eth.entry_price == 2500.0
        assert eth.quantity == 4.0
        assert eth.entry_price_usd == pytest.approx(10000.0, rel=1e-3)
        assert eth.stop_loss_price == 2450.0
        assert eth.take_profit_price == 2600.0

    def test_sync_missing_portfolio_graceful(self, mock_config):
        # portfolio_eth not created → should not raise
        cm = MultiAssetManager(config_path=mock_config["config_path"])
        cm.sync_from_legacy()  # no error
        eth = cm.get_bucket("eth")
        assert eth.current_capital_usd == 10000  # unchanged


# ===========================================================
# Sync multi-bucket portfolio (BTC style)
# ===========================================================

class TestSyncMultiBucketPortfolio:
    def test_sync_btc_no_position(self, mock_config):
        with open(mock_config["portfolio_btc"], "w") as f:
            json.dump({
                "capital_usd": 10500.0,
                "has_position": False,
                "buckets": {
                    "btc_bot1": {"has_position": False, "current_capital": 5200.0},
                    "btc_bot2": {"has_position": False, "current_capital": 5300.0},
                },
            }, f)

        cm = MultiAssetManager(config_path=mock_config["config_path"])
        cm.sync_from_legacy()

        btc = cm.get_bucket("btc")
        assert btc.current_capital_usd == 10500.0
        assert btc.has_position is False

    def test_sync_btc_with_real_structure(self, mock_config):
        """Testa sync com estrutura real do portfolio_state.json BTC."""
        portfolio_data = {
            "has_position": True,
            "entry_price": 75199.97,
            "entry_time": "2026-04-20 10:05:35.671554+00:00",
            "quantity": 0.131108,
            "capital_usd": 9859.32,
            "trailing_high": 76415.99,
            "stop_loss_price": 75651.83,
            "take_profit_price": 76703.97,
            "entry_bot": "bot2",
            "buckets": {
                "btc_bot1": {"current_capital": 4929.66, "has_position": False},
                "btc_bot2": {"current_capital": 4929.66, "has_position": False},
            },
        }
        with open(mock_config["portfolio_btc"], "w") as f:
            json.dump(portfolio_data, f)

        cm = MultiAssetManager(config_path=mock_config["config_path"])
        cm.sync_from_legacy()

        btc = cm.get_bucket("btc")
        assert btc.has_position is True
        assert btc.entry_price == 75199.97
        assert btc.current_capital_usd == 9859.32
        assert btc.entry_timestamp == "2026-04-20 10:05:35.671554+00:00"
        assert btc.bot_origin == "bot2"
        assert btc.realized_pnl == pytest.approx(-140.68, abs=0.01)

    def test_sync_btc_active_sub_bucket(self, mock_config):
        # Estrutura real: campos de posição no root, sub-buckets só com current_capital
        with open(mock_config["portfolio_btc"], "w") as f:
            json.dump({
                "capital_usd": 10000.0,
                "has_position": True,
                "entry_price": 75000.0,
                "quantity": 0.066,
                "stop_loss_price": 73500.0,
                "take_profit_price": 76500.0,
                "trailing_high": 75200.0,
                "entry_time": "2026-04-20T10:00:00+00:00",
                "entry_bot": "bot2",
                "buckets": {
                    "btc_bot1": {"has_position": False, "current_capital": 5000.0},
                    "btc_bot2": {"has_position": True, "current_capital": 5000.0},
                },
            }, f)

        cm = MultiAssetManager(config_path=mock_config["config_path"])
        cm.sync_from_legacy()

        btc = cm.get_bucket("btc")
        assert btc.has_position is True
        assert btc.entry_price == 75000.0
        assert btc.quantity == 0.066
        assert btc.stop_loss_price == 73500.0
        assert btc.entry_price_usd == pytest.approx(75000.0 * 0.066, rel=1e-3)


# ===========================================================
# Summary
# ===========================================================

class TestSummary:
    def test_summary_initial(self, mock_config):
        cm = MultiAssetManager(config_path=mock_config["config_path"])
        s = cm.get_summary()
        assert s.total_initial_capital == 20000
        assert s.total_current_capital == 20000
        assert s.total_realized_pnl == 0.0
        assert s.total_pnl_pct == 0.0
        assert s.active_positions == 0
        assert s.n_buckets_enabled == 2

    def test_summary_after_sync(self, mock_config):
        with open(mock_config["portfolio_btc"], "w") as f:
            json.dump({"capital_usd": 10500, "has_position": False}, f)
        with open(mock_config["portfolio_eth"], "w") as f:
            json.dump({"capital_usd": 9800, "has_position": True,
                       "entry_price": 2500, "quantity": 4}, f)

        cm = MultiAssetManager(config_path=mock_config["config_path"])
        cm.sync_from_legacy()

        s = cm.get_summary()
        assert s.total_current_capital == pytest.approx(20300.0)
        assert s.total_realized_pnl == pytest.approx(300.0)
        assert s.total_pnl_pct == pytest.approx(0.015)
        assert s.active_positions == 1

    def test_get_total_pnl(self, mock_config):
        with open(mock_config["portfolio_btc"], "w") as f:
            json.dump({"capital_usd": 10200, "has_position": False}, f)
        with open(mock_config["portfolio_eth"], "w") as f:
            json.dump({"capital_usd": 9900, "has_position": False}, f)

        cm = MultiAssetManager(config_path=mock_config["config_path"])
        cm.sync_from_legacy()

        assert cm.get_total_pnl() == pytest.approx(100.0)
        assert cm.get_total_capital() == pytest.approx(20100.0)


# ===========================================================
# Persistence
# ===========================================================

class TestPersistence:
    def test_save_and_reload(self, mock_config):
        with open(mock_config["portfolio_btc"], "w") as f:
            json.dump({"capital_usd": 10750, "has_position": False}, f)
        with open(mock_config["portfolio_eth"], "w") as f:
            json.dump({"capital_usd": 10000, "has_position": False}, f)

        cm1 = MultiAssetManager(config_path=mock_config["config_path"])
        cm1.sync_from_legacy()
        cm1.save_state()

        # New instance loads from saved state
        cm2 = MultiAssetManager(config_path=mock_config["config_path"])
        assert cm2.buckets["btc"].current_capital_usd == 10750.0

    def test_state_file_created(self, mock_config):
        with open(mock_config["portfolio_btc"], "w") as f:
            json.dump({"capital_usd": 10000, "has_position": False}, f)

        cm = MultiAssetManager(config_path=mock_config["config_path"])
        cm.sync_from_legacy()
        cm.save_state()

        assert mock_config["state_path"].exists()
        with open(mock_config["state_path"]) as f:
            data = json.load(f)
        assert "buckets" in data
        assert "btc" in data["buckets"]


# ===========================================================
# Kill switch
# ===========================================================

class TestKillSwitch:
    def test_disabled_by_default(self, mock_config):
        cm = MultiAssetManager(config_path=mock_config["config_path"])
        assert cm.check_global_kill_switch() is False


# ===========================================================
# BucketState properties
# ===========================================================

class TestBucketStateProperties:
    def test_pnl_pct_positive(self):
        b = BucketState("test", "BTC", 10000, 11000)
        assert b.pnl_pct == pytest.approx(0.10)

    def test_pnl_pct_negative(self):
        b = BucketState("test", "BTC", 10000, 9500)
        assert b.pnl_pct == pytest.approx(-0.05)

    def test_pnl_pct_zero_initial(self):
        b = BucketState("test", "BTC", 0, 100)
        assert b.pnl_pct == 0.0

    def test_win_rate_no_trades(self):
        b = BucketState("test", "BTC", 10000, 10000)
        assert b.win_rate == 0.0

    def test_win_rate_with_trades(self):
        b = BucketState("test", "BTC", 10000, 10000)
        b.n_trades_total = 10
        b.n_wins = 6
        assert b.win_rate == pytest.approx(0.6)


# ===========================================================
# Alias
# ===========================================================

class TestAlias:
    def test_capital_manager_alias(self, mock_config):
        from src.trading.multi_asset_manager import CapitalManager
        cm = CapitalManager(config_path=mock_config["config_path"])
        assert isinstance(cm, MultiAssetManager)
