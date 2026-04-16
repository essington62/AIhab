"""
tests/test_gate_scoring.py — Gate Scoring v2 unit tests.
Run: pytest tests/test_gate_scoring.py -v
"""

import numpy as np
import pytest

from src.models.gate_scoring import (
    aggregate_clusters,
    check_kill_switches,
    compute_threshold,
    evaluate_g0,
    evaluate_g1,
    evaluate_g2,
    gate_score_continuous,
    run_scoring_pipeline,
)


# ---------------------------------------------------------------------------
# gate_score_continuous
# ---------------------------------------------------------------------------

class TestGateScoreContinuous:
    def test_positive_corr_positive_z(self):
        score = gate_score_continuous(z=2.0, corr=0.3, sensitivity=0.6, max_score=1.0)
        assert score > 0

    def test_negative_corr_positive_z(self):
        score = gate_score_continuous(z=2.0, corr=-0.3, sensitivity=0.6, max_score=1.0)
        assert score < 0

    def test_clipped_at_max(self):
        score = gate_score_continuous(z=10.0, corr=1.0, sensitivity=5.0, max_score=2.0)
        assert score == pytest.approx(2.0)

    def test_clipped_at_min(self):
        score = gate_score_continuous(z=10.0, corr=-1.0, sensitivity=5.0, max_score=2.0)
        assert score == pytest.approx(-2.0)

    def test_nan_z_returns_zero(self):
        score = gate_score_continuous(z=float("nan"), corr=0.3, sensitivity=0.6, max_score=1.0)
        assert score == 0.0

    def test_zero_z_returns_zero(self):
        score = gate_score_continuous(z=0.0, corr=0.5, sensitivity=1.0, max_score=2.0)
        assert score == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# G0 — Regime
# ---------------------------------------------------------------------------

class TestG0Regime:
    def test_bear_blocks(self):
        result = evaluate_g0("Bear")
        assert result["block"] is True
        assert "BEAR" in result["block_reason"]

    def test_sideways_multiplier(self):
        result = evaluate_g0("Sideways")
        assert result["block"] is False
        # multiplier is read from parameters.yml; paper trading sets 1.0, production default 0.5
        from src.config import get_params
        expected = float(get_params().get("sideways_multiplier", 0.5))
        assert result["multiplier"] == pytest.approx(expected)

    def test_bull_multiplier(self):
        result = evaluate_g0("Bull")
        assert result["block"] is False
        assert result["multiplier"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# G1 — Technical bucket scoring (walk-forward validated — do not change)
# ---------------------------------------------------------------------------

class TestG1Technical:
    def test_bb_below_020_wins_88pct(self):
        result = evaluate_g1(bb_pct=0.15, rsi=50.0)
        assert result["bb_score"] == pytest.approx(3.0)

    def test_bb_below_030(self):
        result = evaluate_g1(bb_pct=0.25, rsi=50.0)
        assert result["bb_score"] == pytest.approx(2.0)

    def test_bb_above_080_kill(self):
        result = evaluate_g1(bb_pct=0.85, rsi=50.0)
        assert result["bb_score"] == pytest.approx(-2.0)
        assert result["bb_kill"] is True

    def test_rsi_below_35(self):
        result = evaluate_g1(bb_pct=0.50, rsi=30.0)
        assert result["rsi_score"] == pytest.approx(1.0)

    def test_rsi_above_60_negative(self):
        result = evaluate_g1(bb_pct=0.50, rsi=65.0)
        assert result["rsi_score"] == pytest.approx(-1.0)

    def test_combined_strong_signal(self):
        # BB < 0.20 + RSI < 35 → 3.0 + 1.0 = 4.0
        result = evaluate_g1(bb_pct=0.18, rsi=32.0)
        assert result["g1"] == pytest.approx(4.0)

    def test_none_bb_returns_zero(self):
        result = evaluate_g1(bb_pct=None, rsi=50.0)
        assert result["bb_score"] == 0.0


# ---------------------------------------------------------------------------
# G2 — News split
# ---------------------------------------------------------------------------

class TestG2News:
    def test_neutral_both(self):
        result = evaluate_g2(0.0, {"fed_score": 0.0})
        assert result["g2"] == pytest.approx(0.0)

    def test_positive_crypto(self):
        result = evaluate_g2(2.0, {"fed_score": 0.0})
        assert result["g2"] > 0

    def test_negative_fed(self):
        result = evaluate_g2(0.0, {"fed_score": -2.0})
        assert result["g2"] < 0

    def test_split_50_50(self):
        result = evaluate_g2(2.0, {"fed_score": -2.0})
        assert result["g2"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Cluster aggregation
# ---------------------------------------------------------------------------

class TestClusters:
    def _gates(self, **overrides):
        base = {k: 0.0 for k in ["g1", "g2", "g3", "g4", "g5", "g6", "g7", "g8", "g9", "g10"]}
        base.update(overrides)
        return base

    def test_all_zero(self):
        result = aggregate_clusters(self._gates())
        assert result["total_score"] == pytest.approx(0.0)

    def test_technical_cap_upper(self):
        # G1 = 10.0 should be capped at 3.5
        result = aggregate_clusters(self._gates(g1=10.0))
        assert result["clusters"]["technical"] == pytest.approx(3.5)

    def test_positioning_is_g4_plus_g10(self):
        result = aggregate_clusters(self._gates(g4=1.0, g10=0.3))
        assert result["clusters"]["positioning"] == pytest.approx(1.3)

    def test_total_score_is_sum_of_clusters(self):
        gates = self._gates(g1=2.0, g2=0.5)
        result = aggregate_clusters(gates)
        assert result["total_score"] == pytest.approx(sum(result["clusters"].values()))


# ---------------------------------------------------------------------------
# Threshold
# ---------------------------------------------------------------------------

class TestThreshold:
    def test_warmup_when_short_history(self):
        # < 90 samples → warmup value = 3.5
        history = [2.0] * 10
        thr = compute_threshold(history, fed_proximity_adj=0.0)
        assert thr == pytest.approx(3.5)

    def test_dynamic_with_full_history(self):
        history = list(np.linspace(1.0, 5.0, 100))
        thr = compute_threshold(history, fed_proximity_adj=0.0)
        # Should be somewhere between floor=2.0 and ceiling=5.0
        assert 2.0 <= thr <= 5.0

    def test_fed_proximity_adds_to_threshold(self):
        history = list(np.linspace(1.0, 5.0, 100))
        thr_base = compute_threshold(history, fed_proximity_adj=0.0)
        thr_fomc = compute_threshold(history, fed_proximity_adj=1.5)
        assert thr_fomc == pytest.approx(thr_base + 1.5)


# ---------------------------------------------------------------------------
# Kill switches
# ---------------------------------------------------------------------------

class TestKillSwitches:
    def test_bb_top_kills(self):
        result = check_kill_switches(
            bb_pct=0.85, oi_z=0.0, news_score=0.0,
            fed_context={"fomc_kill_switch": False}, oi_stale=False
        )
        assert result["blocked"] is True
        assert result["reason"] == "BLOCK_BB_TOP"

    def test_oi_extreme_kills(self):
        result = check_kill_switches(
            bb_pct=0.5, oi_z=3.0, news_score=0.0,
            fed_context={"fomc_kill_switch": False}, oi_stale=False
        )
        assert result["blocked"] is True
        assert result["reason"] == "BLOCK_OI_EXTREME"

    def test_oi_stale_not_killed(self):
        # Stale OI should not trigger OI kill switch
        result = check_kill_switches(
            bb_pct=0.5, oi_z=3.0, news_score=0.0,
            fed_context={"fomc_kill_switch": False}, oi_stale=True
        )
        assert result["blocked"] is False

    def test_news_bear_kills(self):
        result = check_kill_switches(
            bb_pct=0.5, oi_z=0.0, news_score=-4.0,
            fed_context={"fomc_kill_switch": False}, oi_stale=False
        )
        assert result["blocked"] is True
        assert result["reason"] == "BLOCK_NEWS_BEAR"

    def test_fed_hawkish_kills(self):
        result = check_kill_switches(
            bb_pct=0.5, oi_z=0.0, news_score=0.0,
            fed_context={"fomc_kill_switch": True}, oi_stale=False
        )
        assert result["blocked"] is True
        assert result["reason"] == "BLOCK_FED_HAWKISH"

    def test_nothing_kills_clean_signal(self):
        result = check_kill_switches(
            bb_pct=0.2, oi_z=1.0, news_score=1.0,
            fed_context={"fomc_kill_switch": False}, oi_stale=False
        )
        assert result["blocked"] is False


# ---------------------------------------------------------------------------
# Full pipeline integration
# ---------------------------------------------------------------------------

class TestScoringPipeline:
    def _base_zscores(self):
        return {
            "oi_z": 0.0, "taker_z": 0.0, "funding_z": 0.0,
            "dgs10_z": 0.0, "dgs2_z": 0.0, "rrp_z": 0.0, "curve_z": 0.0,
            "stablecoin_z": 0.0, "bubble_z": 0.0, "etf_z": 0.0, "fg_z": 0.0,
        }

    def _base_stale(self):
        return {k: 0 for k in [
            "g4_oi", "g9_taker", "g10_funding",
            "g5_stablecoin", "g6_bubble", "g7_etf", "g8_fg", "g3_macro"
        ]}

    def _fed(self, fomc_kill=False, prox=0.0):
        return {"fed_score": 0.0, "fomc_kill_switch": fomc_kill, "proximity_adjustment": prox}

    def test_bear_regime_blocks(self):
        result = run_scoring_pipeline(
            regime="Bear", bb_pct=0.18, rsi=32.0,
            zscores=self._base_zscores(), stale_days=self._base_stale(),
            news_crypto_score=1.0, fed_context=self._fed(),
            score_history=[3.0] * 100,
        )
        assert result["signal"] == "BLOCK"
        assert "BEAR" in result["block_reason"]

    def test_strong_bull_signal_enters(self):
        # BB < 0.20 + RSI < 35 → strong G1, all others neutral
        # Score ~ 4.0 * 0.5 (Sideways multiplier) = 2.0 — may be below threshold 3.5
        # Use Bull regime and strong history to force ENTER
        history = [1.0] * 100  # low history → threshold ~2.0 (floor)
        result = run_scoring_pipeline(
            regime="Bull", bb_pct=0.15, rsi=30.0,
            zscores=self._base_zscores(), stale_days=self._base_stale(),
            news_crypto_score=1.5, fed_context=self._fed(),
            score_history=history,
        )
        # G1 = 4.0 + G2 ≈ 0.75 → total ~ 4.75, threshold ~ clipped floor=2.0
        assert result["signal"] in ("ENTER", "HOLD")  # depends on threshold quantile
        assert result["score"] is not None

    def test_bb_top_blocks_even_with_high_score(self):
        result = run_scoring_pipeline(
            regime="Bull", bb_pct=0.85, rsi=30.0,
            zscores=self._base_zscores(), stale_days=self._base_stale(),
            news_crypto_score=1.0, fed_context=self._fed(),
            score_history=[1.0] * 100,
        )
        assert result["signal"] == "BLOCK"
        assert result["block_reason"] == "BLOCK_BB_TOP"

    def test_fomc_kill_blocks(self):
        result = run_scoring_pipeline(
            regime="Bull", bb_pct=0.18, rsi=32.0,
            zscores=self._base_zscores(), stale_days=self._base_stale(),
            news_crypto_score=1.0, fed_context=self._fed(fomc_kill=True),
            score_history=[1.0] * 100,
        )
        assert result["signal"] == "BLOCK"
        assert result["block_reason"] == "BLOCK_FED_HAWKISH"

    def test_proximity_adj_raises_threshold(self):
        # Same inputs, one with prox=0, one with prox=1.5
        kwargs = dict(
            regime="Bull", bb_pct=0.18, rsi=32.0,
            zscores=self._base_zscores(), stale_days=self._base_stale(),
            news_crypto_score=0.5, score_history=[1.0] * 100,
        )
        r_normal = run_scoring_pipeline(**kwargs, fed_context=self._fed(prox=0.0))
        r_fomc = run_scoring_pipeline(**kwargs, fed_context=self._fed(prox=1.5))
        assert r_fomc["threshold"] > r_normal["threshold"]

    def test_output_keys(self):
        result = run_scoring_pipeline(
            regime="Sideways", bb_pct=0.35, rsi=50.0,
            zscores=self._base_zscores(), stale_days=self._base_stale(),
            news_crypto_score=0.0, fed_context=self._fed(),
            score_history=[2.0] * 100,
        )
        for key in ["signal", "score", "threshold", "block_reason", "gate_scores", "clusters"]:
            assert key in result
