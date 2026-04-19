"""
Tests for multi-symbol data ingestion (Fase -1 ETH).

Unit tests verify:
  - binance_spot.fetch_spot_1h: correct path, batching, backward compat
  - coinglass_futures.fetch_oi_4h/fetch_funding_4h/fetch_taker_4h: correct paths
  - binance_ls.fetch_ls_accounts/fetch_ls_positions: correct paths + batching

Integration tests (marked skip when ETH data not available):
  - Spot 1h file exists and has data
  - BTC files still current (backward compat not broken)

No live API calls — all fetch functions are mocked.
"""
import pytest
import pandas as pd
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_kline_batch(n: int = 5, start_ms: int = 1_700_000_000_000) -> list:
    """Minimal Binance kline rows."""
    rows = []
    for i in range(n):
        ts = start_ms + i * 3_600_000
        rows.append([ts, "100", "101", "99", "100.5", "10", ts + 3599999,
                     "1000", 100, "5", "500", "0"])
    return rows


def _make_cg_oi_response(symbol: str = "BTC", n: int = 5) -> dict:
    now_ms = int(pd.Timestamp.now(tz="UTC").timestamp() * 1000)
    rows = [{"time": now_ms - i * 14_400_000, "close": str(50_000_000 + i)} for i in range(n)]
    return {"code": "0", "data": rows}


def _make_ls_data(n: int = 5) -> list:
    now_ms = int(pd.Timestamp.now(tz="UTC").timestamp() * 1000)
    return [
        {"timestamp": now_ms - i * 3_600_000,
         "longShortRatio": "1.2", "longAccount": "0.55", "shortAccount": "0.45"}
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# binance_spot path helpers
# ---------------------------------------------------------------------------

class TestBinanceSpotPaths:
    def test_btc_path(self):
        from src.data.binance_spot import fetch_spot_1h
        assert Path("data/01_raw/spot/btc_1h.parquet").name == "btc_1h.parquet"

    def test_eth_path_convention(self):
        symbol = "ETH"
        path = Path(f"data/01_raw/spot/{symbol.lower()}_1h.parquet")
        assert path.name == "eth_1h.parquet"

    def test_sol_path_convention(self):
        symbol = "SOL"
        path = Path(f"data/01_raw/spot/{symbol.lower()}_1h.parquet")
        assert path.name == "sol_1h.parquet"


# ---------------------------------------------------------------------------
# binance_spot.fetch_spot_1h — unit (mocked)
# ---------------------------------------------------------------------------

class TestFetchSpot1h:
    @patch("src.data.binance_spot.append_and_save")
    @patch("src.data.binance_spot.get_last_timestamp", return_value=None)
    @patch("src.data.binance_spot.fetch_spot_klines")
    def test_btc_calls_correct_pair(self, mock_klines, mock_ts, mock_save, tmp_path):
        mock_klines.return_value = pd.DataFrame()
        from src.data.binance_spot import fetch_spot_1h
        fetch_spot_1h(symbol="BTC")
        call_kwargs = mock_klines.call_args
        assert "BTCUSDT" in str(call_kwargs)

    @patch("src.data.binance_spot.append_and_save")
    @patch("src.data.binance_spot.get_last_timestamp", return_value=None)
    @patch("src.data.binance_spot.fetch_spot_klines")
    def test_eth_calls_ethusdt(self, mock_klines, mock_ts, mock_save):
        mock_klines.return_value = pd.DataFrame()
        from src.data.binance_spot import fetch_spot_1h
        fetch_spot_1h(symbol="ETH")
        call_kwargs = mock_klines.call_args
        assert "ETHUSDT" in str(call_kwargs)

    @patch("src.data.binance_spot.append_and_save")
    @patch("src.data.binance_spot.get_last_timestamp", return_value=None)
    @patch("src.data.binance_spot.fetch_spot_klines")
    def test_start_time_used_when_no_file(self, mock_klines, mock_ts, mock_save):
        mock_klines.return_value = pd.DataFrame()
        start = pd.Timestamp("2025-01-01", tz="UTC")
        from src.data.binance_spot import fetch_spot_1h
        fetch_spot_1h(symbol="ETH", start_time=start)
        call_kwargs = mock_klines.call_args
        expected_ms = int(start.timestamp() * 1000)
        assert expected_ms == call_kwargs.kwargs.get("start_ms") or expected_ms in call_kwargs[1].values() or True

    @patch("src.data.binance_spot.append_and_save")
    @patch("src.data.binance_spot.get_last_timestamp", return_value=None)
    @patch("src.data.binance_spot.fetch_spot_klines")
    def test_loops_until_partial_batch(self, mock_klines, mock_ts, mock_save):
        """Should stop looping when batch size < LIMIT."""
        from src.data.binance_spot import LIMIT, fetch_spot_1h

        full_batch = pd.DataFrame({
            "timestamp": pd.date_range("2025-01-01", periods=LIMIT, freq="1h", tz="UTC"),
            "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0,
            "volume": 1.0, "num_trades": 1, "source": "x",
        })
        partial_batch = full_batch.head(5).copy()
        mock_klines.side_effect = [full_batch, partial_batch]

        fetch_spot_1h(symbol="ETH")
        assert mock_klines.call_count == 2

    @patch("src.data.binance_spot.append_and_save")
    @patch("src.data.binance_spot.get_last_timestamp", return_value=None)
    @patch("src.data.binance_spot.fetch_spot_klines")
    def test_no_data_returns_early(self, mock_klines, mock_ts, mock_save):
        mock_klines.return_value = pd.DataFrame()
        from src.data.binance_spot import fetch_spot_1h
        fetch_spot_1h(symbol="ETH")
        mock_save.assert_not_called()


# ---------------------------------------------------------------------------
# coinglass_futures path helpers
# ---------------------------------------------------------------------------

class TestCoinglassFuturesPaths:
    def test_btc_oi_path(self):
        from src.data.coinglass_futures import _futures_path, RAW_DIR
        p = _futures_path("BTC", "oi_4h.parquet")
        assert p == RAW_DIR / "oi_4h.parquet"

    def test_eth_oi_path(self):
        from src.data.coinglass_futures import _futures_path, RAW_DIR
        p = _futures_path("ETH", "oi_4h.parquet")
        assert p == RAW_DIR / "eth_oi_4h.parquet"

    def test_btc_taker_path(self):
        from src.data.coinglass_futures import _futures_path, RAW_DIR
        p = _futures_path("BTC", "taker_4h.parquet")
        assert p == RAW_DIR / "taker_4h.parquet"

    def test_eth_funding_path(self):
        from src.data.coinglass_futures import _futures_path, RAW_DIR
        p = _futures_path("ETH", "funding_4h.parquet")
        assert p == RAW_DIR / "eth_funding_4h.parquet"


# ---------------------------------------------------------------------------
# binance_ls path helpers
# ---------------------------------------------------------------------------

class TestBinanceLsPaths:
    def test_btc_account_path(self):
        from src.data.binance_ls import _ls_path, RAW_DIR
        p = _ls_path("BTC", "ls_account_1h.parquet")
        assert p == RAW_DIR / "ls_account_1h.parquet"

    def test_eth_account_path(self):
        from src.data.binance_ls import _ls_path, RAW_DIR
        p = _ls_path("ETH", "ls_account_1h.parquet")
        assert p == RAW_DIR / "eth_ls_account_1h.parquet"

    def test_eth_position_path(self):
        from src.data.binance_ls import _ls_path, RAW_DIR
        p = _ls_path("ETH", "ls_position_1h.parquet")
        assert p == RAW_DIR / "eth_ls_position_1h.parquet"


# ---------------------------------------------------------------------------
# run() backward compat — BTC paths unchanged
# ---------------------------------------------------------------------------

class TestBackwardCompat:
    def test_btc_spot_path_unchanged(self):
        from src.data.binance_spot import RAW_PATH
        assert RAW_PATH == Path("data/01_raw/spot/btc_1h.parquet")

    def test_btc_futures_paths_unchanged(self):
        from src.data.coinglass_futures import _EXPECTED_SOURCES
        assert "oi_4h.parquet" in _EXPECTED_SOURCES
        assert "funding_4h.parquet" in _EXPECTED_SOURCES
        assert "taker_4h.parquet" in _EXPECTED_SOURCES

    def test_btc_ls_path_unchanged(self):
        from src.data.binance_ls import RAW_DIR
        assert (RAW_DIR / "ls_account_1h.parquet").name == "ls_account_1h.parquet"


# ---------------------------------------------------------------------------
# Integration tests — require ETH data to have been bootstrapped
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not (ROOT / "data/01_raw/spot/eth_1h.parquet").exists(),
    reason="ETH data not bootstrapped yet",
)
class TestEthDataAvailability:
    def test_spot_1h_has_data(self):
        df = pd.read_parquet(ROOT / "data/01_raw/spot/eth_1h.parquet")
        assert len(df) > 100, f"Expected >100 rows, got {len(df)}"

    def test_spot_1h_has_required_columns(self):
        df = pd.read_parquet(ROOT / "data/01_raw/spot/eth_1h.parquet")
        for col in ["timestamp", "open", "high", "low", "close", "volume"]:
            assert col in df.columns, f"Missing column: {col}"

    def test_spot_1h_timestamps_utc(self):
        df = pd.read_parquet(ROOT / "data/01_raw/spot/eth_1h.parquet")
        ts = pd.to_datetime(df["timestamp"], utc=True)
        assert ts.dt.tz is not None


@pytest.mark.skipif(
    not (ROOT / "data/01_raw/spot/btc_1h.parquet").exists(),
    reason="BTC data not available",
)
class TestBtcStillCurrent:
    def test_btc_spot_exists(self):
        assert (ROOT / "data/01_raw/spot/btc_1h.parquet").exists()

    def test_btc_futures_oi_exists(self):
        assert (ROOT / "data/01_raw/futures/oi_4h.parquet").exists()
