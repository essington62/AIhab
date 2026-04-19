"""
Tests for src/data/coinglass_ls.py and dedup_by_timestamp.
No live API calls — all network calls are mocked.
"""
import pytest
import pandas as pd
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.data.coinglass_ls import (
    _normalize_ls,
    _ls_raw_path,
    fetch_ls_accounts_cg,
    fetch_ls_positions_cg,
)
from src.data.utils import dedup_by_timestamp


# ---------------------------------------------------------------------------
# _normalize_ls
# ---------------------------------------------------------------------------

class TestNormalize:
    def test_basic(self):
        records = [
            {"time": 1700000000000, "longShortRatio": 1.5, "longAccount": 0.6, "shortAccount": 0.4},
            {"time": 1700003600000, "longShortRatio": 1.4, "longAccount": 0.58, "shortAccount": 0.42},
        ]
        df = _normalize_ls(records, "coinglass_ls_account")
        assert len(df) == 2
        assert set(df.columns) >= {"timestamp", "longShortRatio", "longAccount", "shortAccount", "source"}
        assert df["longShortRatio"].iloc[0] == 1.5
        assert df["source"].iloc[0] == "coinglass_ls_account"

    def test_empty_records(self):
        assert _normalize_ls([], "x").empty

    def test_missing_ts_skipped(self):
        records = [
            {"time": None, "longShortRatio": 1.5},
            {"time": 1700000000000, "longShortRatio": 1.0, "longAccount": 0.5, "shortAccount": 0.5},
        ]
        df = _normalize_ls(records, "x")
        assert len(df) == 1

    def test_sorted_ascending(self):
        records = [
            {"time": 1700003600000, "longShortRatio": 1.4},
            {"time": 1700000000000, "longShortRatio": 1.5},
        ]
        df = _normalize_ls(records, "x")
        assert df["timestamp"].iloc[0] < df["timestamp"].iloc[1]

    def test_alternative_field_names(self):
        """CoinGlass may use 'long'/'short' instead of 'longAccount'/'shortAccount'."""
        records = [
            {"time": 1700000000000, "longShortRatio": 1.5, "long": 0.6, "short": 0.4},
        ]
        df = _normalize_ls(records, "x")
        assert df["longAccount"].iloc[0] == 0.6
        assert df["shortAccount"].iloc[0] == 0.4


# ---------------------------------------------------------------------------
# _ls_raw_path — naming convention
# ---------------------------------------------------------------------------

class TestLsRawPath:
    def test_btc_no_prefix(self):
        from src.data.coinglass_ls import RAW_DIR
        assert _ls_raw_path("BTC", "ls_account_1h.parquet") == RAW_DIR / "ls_account_1h.parquet"

    def test_eth_with_prefix(self):
        from src.data.coinglass_ls import RAW_DIR
        assert _ls_raw_path("ETH", "ls_account_1h.parquet") == RAW_DIR / "eth_ls_account_1h.parquet"

    def test_eth_position_path(self):
        from src.data.coinglass_ls import RAW_DIR
        assert _ls_raw_path("ETH", "ls_position_1h.parquet") == RAW_DIR / "eth_ls_position_1h.parquet"

    def test_lowercase_input(self):
        from src.data.coinglass_ls import RAW_DIR
        assert _ls_raw_path("eth", "ls_account_1h.parquet") == RAW_DIR / "eth_ls_account_1h.parquet"


# ---------------------------------------------------------------------------
# dedup_by_timestamp
# ---------------------------------------------------------------------------

class TestDedupByTimestamp:
    def test_keeps_last_on_collision(self):
        """When CoinGlass (old) and Binance (new) overlap, Binance wins."""
        df_cg = pd.DataFrame({
            "timestamp": pd.to_datetime(["2026-04-19 10:00", "2026-04-19 11:00"], utc=True),
            "longShortRatio": [1.5, 1.4],
            "source": ["coinglass", "coinglass"],
        })
        df_bn = pd.DataFrame({
            "timestamp": pd.to_datetime(["2026-04-19 11:00", "2026-04-19 12:00"], utc=True),
            "longShortRatio": [1.42, 1.3],
            "source": ["binance", "binance"],
        })
        combined = pd.concat([df_cg, df_bn], ignore_index=True)
        result = dedup_by_timestamp(combined)
        assert len(result) == 3  # 10:00, 11:00, 12:00
        row_11 = result[result["timestamp"] == pd.Timestamp("2026-04-19 11:00", tz="UTC")]
        assert row_11["longShortRatio"].iloc[0] == pytest.approx(1.42)
        assert row_11["source"].iloc[0] == "binance"

    def test_no_duplicates_unchanged(self):
        df = pd.DataFrame({
            "timestamp": pd.to_datetime(["2026-04-19 10:00", "2026-04-19 11:00"], utc=True),
            "value": [1, 2],
        })
        result = dedup_by_timestamp(df)
        assert len(result) == 2

    def test_empty_df(self):
        assert dedup_by_timestamp(pd.DataFrame()).empty

    def test_no_timestamp_col(self):
        df = pd.DataFrame({"value": [1, 2]})
        result = dedup_by_timestamp(df)
        assert len(result) == 2

    def test_sorted_ascending(self):
        df = pd.DataFrame({
            "timestamp": pd.to_datetime(["2026-04-19 12:00", "2026-04-19 10:00"], utc=True),
            "value": [3, 1],
        })
        result = dedup_by_timestamp(df)
        assert result["timestamp"].iloc[0] < result["timestamp"].iloc[1]


# ---------------------------------------------------------------------------
# fetch_ls_accounts_cg / fetch_ls_positions_cg — mocked
# ---------------------------------------------------------------------------

_SAMPLE_RECORDS = [
    {"time": 1700000000000, "longShortRatio": 1.5, "longAccount": 0.6, "shortAccount": 0.4},
    {"time": 1700003600000, "longShortRatio": 1.4, "longAccount": 0.58, "shortAccount": 0.42},
]


class TestFetchLsAccountsCg:
    @patch("src.data.coinglass_ls._fetch_ls_paginated", return_value=_SAMPLE_RECORDS)
    @patch("src.data.coinglass_ls._load_api_key", return_value="test-key")
    def test_returns_dataframe(self, mock_key, mock_fetch):
        df = fetch_ls_accounts_cg("ETH")
        assert not df.empty
        assert "timestamp" in df.columns
        assert "longShortRatio" in df.columns

    @patch("src.data.coinglass_ls._fetch_ls_paginated", return_value=[])
    @patch("src.data.coinglass_ls._load_api_key", return_value="test-key")
    def test_empty_returns_empty_df(self, mock_key, mock_fetch):
        df = fetch_ls_accounts_cg("ETH")
        assert df.empty

    @patch("src.data.coinglass_ls._load_api_key", side_effect=ValueError("no key"))
    def test_missing_api_key_returns_empty(self, mock_key):
        df = fetch_ls_accounts_cg("ETH")
        assert df.empty


class TestFetchLsPositionsCg:
    @patch("src.data.coinglass_ls._fetch_ls_paginated", return_value=_SAMPLE_RECORDS)
    @patch("src.data.coinglass_ls._load_api_key", return_value="test-key")
    def test_uses_position_endpoint(self, mock_key, mock_fetch):
        from src.data.coinglass_ls import _ENDPOINT_POSITION
        fetch_ls_positions_cg("ETH")
        call_args = mock_fetch.call_args
        assert call_args[0][0] == _ENDPOINT_POSITION

    @patch("src.data.coinglass_ls._fetch_ls_paginated", return_value=[])
    @patch("src.data.coinglass_ls._load_api_key", return_value="test-key")
    def test_empty_returns_empty(self, mock_key, mock_fetch):
        df = fetch_ls_positions_cg("ETH")
        assert df.empty


# ---------------------------------------------------------------------------
# _fetch_ls_paginated — verify pagination logic
# ---------------------------------------------------------------------------

class TestPaginationLogic:
    @patch("src.data.coinglass_ls.fetch_with_retry")
    @patch("src.data.coinglass_ls._load_api_key", return_value="test-key")
    def test_stops_on_partial_batch(self, mock_key, mock_fetch):
        """Should stop after 1 page when < limit records returned."""
        from src.data.coinglass_ls import _fetch_ls_paginated
        mock_fetch.return_value = {
            "code": "0",
            "data": [{"time": 1700000000000, "longShortRatio": 1.5} for _ in range(10)],
        }
        records = _fetch_ls_paginated("/api/test", "ETH", "test-key")
        assert len(records) == 10
        assert mock_fetch.call_count == 1  # only one page needed

    @patch("src.data.coinglass_ls.fetch_with_retry")
    @patch("src.data.coinglass_ls._load_api_key", return_value="test-key")
    def test_stops_on_api_error(self, mock_key, mock_fetch):
        """Should stop and return empty on API error."""
        from src.data.coinglass_ls import _fetch_ls_paginated
        mock_fetch.return_value = {"code": "10001", "msg": "endpoint not available"}
        records = _fetch_ls_paginated("/api/test", "ETH", "test-key")
        assert records == []

    @patch("src.data.coinglass_ls.fetch_with_retry")
    @patch("src.data.coinglass_ls._load_api_key", return_value="test-key")
    def test_loops_on_full_batch(self, mock_key, mock_fetch):
        """Should request a second page when first page has exactly limit=500 records."""
        from src.data.coinglass_ls import _fetch_ls_paginated
        from datetime import datetime, timezone
        # Use a start_time before the test timestamps so pagination advances correctly
        start_time = datetime(2023, 11, 1, tzinfo=timezone.utc)
        full_page = [{"time": 1700000000000 + i * 3_600_000, "longShortRatio": 1.0} for i in range(500)]
        partial_page = full_page[:5]
        mock_fetch.side_effect = [
            {"code": "0", "data": full_page},
            {"code": "0", "data": partial_page},
        ]
        records = _fetch_ls_paginated("/api/test", "ETH", "test-key", start_time=start_time)
        assert len(records) == 505
        assert mock_fetch.call_count == 2
