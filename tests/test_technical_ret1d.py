"""
tests/test_technical_ret1d.py — Unit tests for ret_1d in get_latest_technical().
Run: pytest tests/test_technical_ret1d.py -v
"""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from unittest.mock import patch


def _make_spot_parquet(tmp_path: Path, n_rows: int = 50) -> Path:
    """Build a minimal clean_spot_1h parquet with n_rows of hourly candles."""
    base_close = 75000.0
    closes = [base_close + i * 10 for i in range(n_rows)]
    df = pd.DataFrame({
        "timestamp": pd.date_range("2026-04-01", periods=n_rows, freq="1h", tz="UTC"),
        "close": closes,
        "open": closes,
        "high": [c + 50 for c in closes],
        "low": [c - 50 for c in closes],
        "volume": [1.0] * n_rows,
        "bb_pct": [0.5] * n_rows,
        "bb_upper": [c + 500 for c in closes],
        "bb_middle": closes,
        "bb_lower": [c - 500 for c in closes],
        "rsi_14": [50.0] * n_rows,
        "atr_14": [300.0] * n_rows,
        "ma_7": closes,
        "ma_21": closes,
        "ma_50": closes,
        "ma_99": closes,
        "ma_200": closes,
        "high_7d": [c + 500 for c in closes],
        "low_7d": [c - 500 for c in closes],
        "high_30d": [c + 1000 for c in closes],
        "low_30d": [c - 1000 for c in closes],
    })
    path = tmp_path / "btc_1h_clean.parquet"
    df.to_parquet(path, index=False)
    return path


def _params():
    return {"technical": {"bb_window": 20, "bb_std": 2, "rsi_window": 14, "ma_windows": [7, 21, 50, 99, 200]}}


class TestGetLatestTechnicalRet1d:
    def test_ret_1d_calculated(self, tmp_path):
        path = _make_spot_parquet(tmp_path, n_rows=50)
        with (
            patch("src.features.technical.get_path", return_value=path),
            patch("src.features.technical.get_params", return_value=_params()),
        ):
            from src.features.technical import get_latest_technical
            result = get_latest_technical()

        assert result["ret_1d"] is not None
        assert isinstance(result["ret_1d"], float)

    def test_ret_1d_insufficient_data(self, tmp_path):
        path = _make_spot_parquet(tmp_path, n_rows=20)
        with (
            patch("src.features.technical.get_path", return_value=path),
            patch("src.features.technical.get_params", return_value=_params()),
        ):
            from src.features.technical import get_latest_technical
            result = get_latest_technical()

        assert result["ret_1d"] is None

    def test_ret_1d_matches_manual(self, tmp_path):
        path = _make_spot_parquet(tmp_path, n_rows=50)
        with (
            patch("src.features.technical.get_path", return_value=path),
            patch("src.features.technical.get_params", return_value=_params()),
        ):
            from src.features.technical import get_latest_technical
            result = get_latest_technical()

        # manual: close[-1]=75000+49*10=75490, close[-25]=75000+25*10=75250
        # ret_1d = (75490 - 75250) / 75250
        close_last = 75000.0 + 49 * 10
        close_24h_ago = 75000.0 + 25 * 10
        expected = (close_last - close_24h_ago) / close_24h_ago
        assert result["ret_1d"] == pytest.approx(round(expected, 6), abs=1e-7)
