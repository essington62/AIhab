import pytest
from src.trading.dynamic_tp import get_dynamic_tp


def test_rule_1_volume_exhaustion():
    tp, reason = get_dynamic_tp(rsi=60, bb_pct=0.5, volume_z=1.5)
    assert tp == 0.010
    assert reason == "volume_exhaustion"


def test_rule_2_overbought():
    tp, reason = get_dynamic_tp(rsi=78, bb_pct=0.97, volume_z=0.5)
    assert tp == 0.015
    assert reason == "overbought"


def test_rule_2_only_rsi_not_enough():
    tp, reason = get_dynamic_tp(rsi=78, bb_pct=0.8, volume_z=0.5)
    assert tp == 0.020
    assert reason == "default"


def test_rule_2_only_bb_not_enough():
    tp, reason = get_dynamic_tp(rsi=60, bb_pct=0.97, volume_z=0.5)
    assert tp == 0.020
    assert reason == "default"


def test_rule_3_default():
    tp, reason = get_dynamic_tp(rsi=65, bb_pct=0.7, volume_z=0.2)
    assert tp == 0.020
    assert reason == "default"


def test_rule_1_dominates_rule_2():
    # volume_z > 1.0 wins even if RSI+BB are overbought
    tp, reason = get_dynamic_tp(rsi=80, bb_pct=0.98, volume_z=1.5)
    assert tp == 0.010
    assert reason == "volume_exhaustion"


def test_none_values():
    tp, reason = get_dynamic_tp(rsi=None, bb_pct=None, volume_z=None)
    assert tp == 0.020
    assert reason == "default"


def test_volume_z_exactly_threshold():
    # boundary: volume_z == 1.0 should NOT trigger rule 1 (needs > 1.0)
    tp, reason = get_dynamic_tp(rsi=60, bb_pct=0.5, volume_z=1.0)
    assert tp == 0.020
    assert reason == "default"


def test_rsi_exactly_threshold():
    # boundary: rsi == 75 should NOT trigger rule 2 (needs > 75)
    tp, reason = get_dynamic_tp(rsi=75, bb_pct=0.97, volume_z=0.5)
    assert tp == 0.020
    assert reason == "default"
