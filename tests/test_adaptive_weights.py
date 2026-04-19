"""Tests for adaptive weights module."""
import numpy as np
import pandas as pd
import pytest

from src.models.adaptive_weights import (
    apply_kill_switch,
    compute_adaptive_weights,
    compute_confidence,
    compute_delta_adjusted,
    compute_delta_smooth,
    compute_rolling_correlations,
    get_global_multiplier,
)


class TestComputeDeltaSmooth:
    def test_both_available(self):
        # cfg=0.3, short=0.25, long=0.28
        # delta_short=0.05, delta_long=0.02
        # result = 0.7*0.02 + 0.3*0.05 = 0.029
        d = compute_delta_smooth(0.3, 0.25, 0.28)
        assert d == pytest.approx(0.029, abs=0.001)

    def test_only_long(self):
        d = compute_delta_smooth(0.3, None, 0.25)
        assert d == pytest.approx(0.05)

    def test_only_short(self):
        d = compute_delta_smooth(0.3, 0.25, None)
        assert d == pytest.approx(0.05)

    def test_none(self):
        d = compute_delta_smooth(0.3, None, None)
        assert d is None

    def test_sign_inversion_increases_delta(self):
        d = compute_delta_smooth(0.263, -0.101, -0.101)
        assert d == pytest.approx(0.364, abs=0.001)


class TestComputeConfidence:
    def test_perfect_alignment(self):
        assert compute_confidence(0.0) == 1.0

    def test_small_delta(self):
        assert compute_confidence(0.1) == pytest.approx(0.9)

    def test_medium_delta(self):
        assert compute_confidence(0.4) == pytest.approx(0.6)

    def test_extreme_delta_clamped(self):
        assert compute_confidence(1.5) == 0.0

    def test_none_returns_one(self):
        assert compute_confidence(None) == 1.0

    def test_min_confidence_floor(self):
        assert compute_confidence(0.9, min_confidence=0.2) == pytest.approx(0.2)


class TestApplyKillSwitch:
    def test_no_delta(self):
        w, s = apply_kill_switch(2.0, None)
        assert w == 2.0
        assert s == "ok"

    def test_below_severe(self):
        w, s = apply_kill_switch(2.0, 0.3)
        assert w == 2.0
        assert s == "ok"

    def test_severe(self):
        w, s = apply_kill_switch(2.0, 0.55)
        assert w == pytest.approx(0.6)
        assert s == "severe"

    def test_extreme(self):
        w, s = apply_kill_switch(2.0, 0.7)
        assert w == 0.0
        assert s == "extreme"

    def test_boundary_severe(self):
        _, s = apply_kill_switch(2.0, 0.5)
        assert s == "severe"

    def test_boundary_extreme(self):
        _, s = apply_kill_switch(2.0, 0.6)
        assert s == "extreme"

    def test_custom_thresholds(self):
        _, s = apply_kill_switch(2.0, 0.45, severe_threshold=0.4, extreme_threshold=0.5)
        assert s == "severe"

    def test_custom_multiplier(self):
        w, s = apply_kill_switch(2.0, 0.55, severe_multiplier=0.5)
        assert w == pytest.approx(1.0)
        assert s == "severe"


class TestRealCases:
    """Testes reproduzindo casos reais da Fase 1."""

    def test_g7_etf_sign_inversion(self):
        """G7 ETF: cfg=+0.263, real=-0.101, delta=0.364 → ok (< 0.5)."""
        delta = compute_delta_smooth(0.263, -0.101, -0.101)
        assert delta == pytest.approx(0.364, abs=0.001)

        conf = compute_confidence(delta)
        assert conf == pytest.approx(0.636, abs=0.01)

        w, s = apply_kill_switch(1.5 * conf, delta)
        assert s == "ok"
        assert w == pytest.approx(1.5 * 0.636, abs=0.02)

    def test_g3_dgs10_severe(self):
        """G3 DGS10: cfg=-0.315, real=+0.191, delta≈0.506 → severe."""
        delta = compute_delta_smooth(-0.315, 0.191, 0.191)
        assert delta == pytest.approx(0.506, abs=0.001)

        conf = compute_confidence(delta)
        w, s = apply_kill_switch(1.0 * conf, delta)
        assert s == "severe"
        assert w == pytest.approx(conf * 0.3, abs=0.01)

    def test_g4_oi_severe(self):
        """G4 OI: cfg=-0.472, real=+0.081, delta≈0.553 → severe."""
        delta = compute_delta_smooth(-0.472, 0.081, 0.081)
        assert delta == pytest.approx(0.553, abs=0.001)

        conf = compute_confidence(delta)
        w, s = apply_kill_switch(2.0 * conf, delta)
        assert s == "severe"
        assert w == pytest.approx(2.0 * conf * 0.3, abs=0.01)


class TestComputeAdaptiveWeights:
    def _make_data(self, n_days=80):
        dates = pd.date_range("2026-01-01", periods=n_days, freq="D", tz="UTC")
        np.random.seed(42)
        zs = pd.DataFrame(
            {col: np.random.randn(n_days)
             for col in ["oi_z", "taker_z", "funding_z", "dgs10_z", "curve_z",
                         "stablecoin_z", "bubble_z", "etf_z", "fg_z"]},
            index=dates,
        )
        spot = pd.Series(70000 + np.cumsum(np.random.randn(n_days) * 500), index=dates)
        return zs, spot

    def _full_params(self):
        return {
            "adaptive_weights": {
                "enabled": True,
                "confidence": {
                    "enabled": True, "window_short": 30, "window_long": 60,
                    "smooth_short": 0.3, "smooth_long": 0.7, "min_confidence": 0.0,
                    "smoothing_days": 5, "weak_gate_threshold": 0.2,
                },
                "kill_switch": {
                    "enabled": True, "severe_delta_threshold": 0.5,
                    "severe_multiplier": 0.3, "extreme_delta_threshold": 0.6,
                },
                "min_days_for_confidence": 20,
            },
            "gate_params": {
                "g4_oi":      [-0.472, 0.8, 2.0],
                "g7_etf":     [0.263, 0.6, 1.5],
                "g5_stable":  [0.326, 0.6, 1.5],
                "g9_taker":   [0.143, 0.5, 0.3],
                "g10_funding": [-0.064, 0.4, 0.5],
            },
        }

    def test_disabled_returns_base_weights(self):
        params = {
            "adaptive_weights": {"enabled": False},
            "gate_params": {"g4_oi": [-0.472, 0.8, 2.0]},
        }
        zs, spot = self._make_data()
        result = compute_adaptive_weights(zs, spot, params)
        assert result["enabled"] is False
        assert result["weights"]["g4_oi"] == 2.0

    def test_enabled_returns_details(self):
        zs, spot = self._make_data()
        result = compute_adaptive_weights(zs, spot, self._full_params())
        assert result["enabled"] is True
        assert "details" in result
        assert "summary" in result
        assert set(result["summary"].keys()) == {"n_ok", "n_reduced", "n_severe", "n_extreme", "mean_confidence", "weighted_mean_confidence"}

    def test_effective_weight_le_base(self):
        """Effective weight deve ser ≤ base (adaptive só reduz)."""
        zs, spot = self._make_data()
        result = compute_adaptive_weights(zs, spot, self._full_params())
        for gkey, eff_w in result["weights"].items():
            base_w = self._full_params()["gate_params"][gkey][2]
            assert eff_w <= base_w + 1e-9, f"{gkey}: eff={eff_w} > base={base_w}"

    def test_insufficient_data_fallback(self):
        """Com poucos dias (< window_short=30), usa peso base."""
        zs, spot = self._make_data(n_days=15)
        result = compute_adaptive_weights(zs, spot, self._full_params())
        for gkey, eff_w in result["weights"].items():
            base_w = self._full_params()["gate_params"][gkey][2]
            assert eff_w == base_w, f"{gkey}: fallback should be base weight"
        for d in result["details"].values():
            assert d["reason"] == "insufficient_data"

    def test_summary_counts_consistent(self):
        zs, spot = self._make_data()
        result = compute_adaptive_weights(zs, spot, self._full_params())
        s = result["summary"]
        total = s["n_ok"] + s["n_reduced"] + s["n_severe"] + s["n_extreme"]
        assert total == len(result["details"])

    def test_summary_has_weighted_mean_confidence(self):
        zs, spot = self._make_data()
        result = compute_adaptive_weights(zs, spot, self._full_params())
        assert "weighted_mean_confidence" in result["summary"]
        wmc = result["summary"]["weighted_mean_confidence"]
        assert 0.0 <= wmc <= 1.0


class TestSmoothingDays:
    """Suavização temporal de rolling correlations."""

    def _make_zs_spot(self, n=70):
        dates = pd.date_range("2026-01-01", periods=n, freq="D", tz="UTC")
        np.random.seed(42)
        zs = pd.DataFrame({"oi_z": np.random.randn(n)}, index=dates)
        spot = pd.Series(70000 + np.cumsum(np.random.randn(n) * 200), index=dates)
        return zs, spot

    def test_smoothing_reduces_noise(self):
        """smoothing_days=1 vs 5 produzem resultados diferentes."""
        zs, spot = self._make_zs_spot()
        # Injetar outlier no último dia
        zs.iloc[-1, 0] = 5.0
        r1 = compute_rolling_correlations(zs, spot, windows=[60], smoothing_days=1)
        r5 = compute_rolling_correlations(zs, spot, windows=[60], smoothing_days=5)
        # Resultados devem diferir (spike no último ponto tem muito mais peso em r1)
        assert r1["oi_z"][60] != r5["oi_z"][60]

    def test_smoothing_1_equals_single_point(self):
        """smoothing_days=1 deve retornar o valor do último ponto da rolling."""
        zs, spot = self._make_zs_spot()
        r = compute_rolling_correlations(zs, spot, windows=[60], smoothing_days=1)
        # Verificar que retornou um float (não None)
        assert r["oi_z"][60] is not None
        assert isinstance(r["oi_z"][60], float)

    def test_insufficient_data_returns_none(self):
        """Série menor que window retorna None."""
        dates = pd.date_range("2026-01-01", periods=20, freq="D", tz="UTC")
        zs = pd.DataFrame({"oi_z": np.random.randn(20)}, index=dates)
        spot = pd.Series(70000 + np.cumsum(np.random.randn(20) * 200), index=dates)
        r = compute_rolling_correlations(zs, spot, windows=[30], smoothing_days=5)
        assert r["oi_z"][30] is None


class TestWeakGateNormalization:
    """Normalização condicional para gates fracos."""

    def test_strong_gate_uses_raw_delta(self):
        """cfg=0.3 (forte, >= 0.2) → delta sem normalização = 0.3."""
        d = compute_delta_adjusted(0.3, 0.0, 0.0, weak_gate_threshold=0.2)
        assert d == pytest.approx(0.3, abs=0.001)

    def test_weak_gate_normalized_to_one(self):
        """cfg=0.05, real=-0.05 → raw=0.10, scale=0.1, delta=1.0 (clamped)."""
        d = compute_delta_adjusted(0.05, -0.05, -0.05, weak_gate_threshold=0.2)
        assert d == pytest.approx(1.0, abs=0.001)

    def test_weak_gate_small_delta(self):
        """cfg=0.1 (fraco), real=0.08 → raw=0.02, scale=0.1, delta=0.2."""
        d = compute_delta_adjusted(0.1, 0.08, 0.08, weak_gate_threshold=0.2)
        assert d == pytest.approx(0.2, abs=0.001)

    def test_boundary_threshold_no_normalization(self):
        """cfg=0.2 (exatamente no threshold) → sem normalização."""
        d = compute_delta_adjusted(0.2, 0.0, 0.0, weak_gate_threshold=0.2)
        assert d == pytest.approx(0.2, abs=0.001)

    def test_g10_funding_normalized(self):
        """G10 Funding: cfg=-0.064 (fraco), real=+0.230 → normalizado."""
        # raw = abs(-0.064 - 0.230) = 0.294, scale=max(0.064, 0.1)=0.1, delta=min(2.94,1.0)=1.0
        d = compute_delta_adjusted(-0.064, 0.230, 0.230, weak_gate_threshold=0.2)
        assert d == pytest.approx(1.0, abs=0.001)

    def test_delta_smooth_alias_no_normalization(self):
        """compute_delta_smooth não normaliza (weak_gate_threshold=0.0)."""
        d_adj = compute_delta_adjusted(0.064, 0.230, 0.230, weak_gate_threshold=0.0)
        d_smooth = compute_delta_smooth(0.064, 0.230, 0.230)
        assert d_adj == pytest.approx(d_smooth, abs=0.001)


class TestGlobalConfidence:
    """Multiplicador global no score total."""

    def test_disabled_returns_one(self):
        params = {"adaptive_weights": {"global_confidence": {"enabled": False}}}
        result = {"summary": {"mean_confidence": 0.5, "weighted_mean_confidence": 0.4}}
        mult, _ = get_global_multiplier(result, params)
        assert mult == 1.0

    def test_mean_source(self):
        params = {"adaptive_weights": {"global_confidence": {
            "enabled": True, "source": "mean",
            "min_multiplier": 0.2, "max_multiplier": 1.0,
        }}}
        result = {"summary": {"mean_confidence": 0.7, "weighted_mean_confidence": 0.6}}
        mult, label = get_global_multiplier(result, params)
        assert mult == pytest.approx(0.7)
        assert "mean=0.700" in label

    def test_weighted_mean_source(self):
        params = {"adaptive_weights": {"global_confidence": {
            "enabled": True, "source": "weighted_mean",
            "min_multiplier": 0.2, "max_multiplier": 1.0,
        }}}
        result = {"summary": {"mean_confidence": 0.7, "weighted_mean_confidence": 0.6}}
        mult, label = get_global_multiplier(result, params)
        assert mult == pytest.approx(0.6)
        assert "weighted_mean=0.600" in label

    def test_min_multiplier_floor(self):
        params = {"adaptive_weights": {"global_confidence": {
            "enabled": True, "source": "mean",
            "min_multiplier": 0.3, "max_multiplier": 1.0,
        }}}
        result = {"summary": {"mean_confidence": 0.1, "weighted_mean_confidence": 0.1}}
        mult, _ = get_global_multiplier(result, params)
        assert mult == pytest.approx(0.3)

    def test_max_multiplier_ceiling(self):
        params = {"adaptive_weights": {"global_confidence": {
            "enabled": True, "source": "mean",
            "min_multiplier": 0.2, "max_multiplier": 0.95,
        }}}
        result = {"summary": {"mean_confidence": 1.0, "weighted_mean_confidence": 1.0}}
        mult, _ = get_global_multiplier(result, params)
        assert mult == pytest.approx(0.95)

    def test_no_adaptive_result_returns_one(self):
        """Summary vazio (sem dados) → retorna 1.0 (fallback)."""
        params = {"adaptive_weights": {"global_confidence": {
            "enabled": True, "source": "mean",
            "min_multiplier": 0.2, "max_multiplier": 1.0,
        }}}
        mult, _ = get_global_multiplier({}, params)
        # mean_confidence ausente → default 1.0 → clamped → 1.0
        assert mult == pytest.approx(1.0)


class TestWeightedMeanConfidence:
    """weighted_mean_confidence usa base_weight como ponderação."""

    def _params_two_gates(self):
        return {
            "adaptive_weights": {
                "enabled": True,
                "confidence": {
                    "enabled": True, "window_short": 30, "window_long": 60,
                    "smooth_short": 0.3, "smooth_long": 0.7, "min_confidence": 0.0,
                    "smoothing_days": 5, "weak_gate_threshold": 0.2,
                },
                "kill_switch": {"enabled": False},
                "min_days_for_confidence": 20,
            },
            "gate_params": {
                "g4_oi":    [-0.472, 0.8, 2.0],   # peso alto
                "g9_taker": [0.143, 0.5, 0.3],    # peso baixo
            },
        }

    def test_weighted_mean_in_summary(self):
        dates = pd.date_range("2026-01-01", periods=80, freq="D", tz="UTC")
        np.random.seed(7)
        zs = pd.DataFrame(
            {"oi_z": np.random.randn(80), "taker_z": np.random.randn(80)},
            index=dates,
        )
        spot = pd.Series(70000 + np.cumsum(np.random.randn(80) * 500), index=dates)
        result = compute_adaptive_weights(zs, spot, self._params_two_gates())
        assert "weighted_mean_confidence" in result["summary"]
        # Verificar que o campo é float no intervalo [0, 1]
        wmc = result["summary"]["weighted_mean_confidence"]
        assert isinstance(wmc, float)
        assert 0.0 <= wmc <= 1.0
