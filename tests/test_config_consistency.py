"""
tests/test_config_consistency.py — Detect drift between hardcoded fallbacks and parameters.yml.

If a fallback in app.py diverges from the YAML, this test fails before the bug
reaches production. Run: pytest tests/test_config_consistency.py -v
"""

import ast
import re
from pathlib import Path

import yaml

ROOT = Path(__file__).parent.parent
PARAMS_PATH = ROOT / "conf" / "parameters.yml"
APP_PATH = ROOT / "src" / "dashboard" / "app.py"


def load_yaml_params() -> dict:
    with open(PARAMS_PATH) as f:
        return yaml.safe_load(f)


def extract_dict_get_fallbacks(source: str, dict_var: str) -> dict[str, list]:
    """
    Extract all `dict_var.get("key", [v1, v2, v3])` calls from source.
    Returns {key: [v1, v2, v3]} for each match.
    """
    pattern = re.compile(
        rf'{re.escape(dict_var)}\.get\(\s*"([^"]+)"\s*,\s*\[([^\]]+)\]\s*\)'
    )
    found = {}
    for key, values_str in pattern.findall(source):
        try:
            values = [float(v.strip()) for v in values_str.split(",")]
            found[key] = values
        except ValueError:
            pass  # non-numeric fallback, skip
    return found


class TestGateParamFallbacksMatchYAML:
    """
    Every gp.get("key", [fallback]) in compute_clusters must equal parameters.yml gate_params[key].
    Catches the bug where max_score or sensitivity drifts (e.g. g9_taker 0.5 vs 0.3).
    """

    def test_gate_params_fallbacks(self):
        source = APP_PATH.read_text()
        fallbacks = extract_dict_get_fallbacks(source, "gp")

        assert fallbacks, "No gp.get(...) fallbacks found — regex may be broken"

        yaml_gate_params = load_yaml_params()["gate_params"]
        errors = []

        for key, fallback in fallbacks.items():
            if key not in yaml_gate_params:
                errors.append(f"  {key}: in code fallback but missing from parameters.yml")
                continue
            yaml_val = [float(v) for v in yaml_gate_params[key]]
            if fallback != yaml_val:
                errors.append(
                    f"  {key}: fallback={fallback} != yaml={yaml_val}"
                )

        assert not errors, (
            "Hardcoded fallbacks in app.py compute_clusters diverge from parameters.yml:\n"
            + "\n".join(errors)
        )

    def test_all_gate_params_have_fallback(self):
        """Every gate_params key in YAML should have a fallback in the code (no silent omissions)."""
        source = APP_PATH.read_text()
        fallbacks = extract_dict_get_fallbacks(source, "gp")
        yaml_gate_params = load_yaml_params()["gate_params"]

        missing = [k for k in yaml_gate_params if k not in fallbacks]
        assert not missing, (
            f"gate_params keys in YAML have no fallback in app.py compute_clusters: {missing}"
        )


class TestClusterCapsFallbacksMatchYAML:
    """
    Every caps.get("cluster", [fallback]) in compute_clusters must equal parameters.yml cluster_caps[cluster].
    """

    def test_cluster_caps_fallbacks(self):
        source = APP_PATH.read_text()
        fallbacks = extract_dict_get_fallbacks(source, "caps")

        assert fallbacks, "No caps.get(...) fallbacks found — regex may be broken"

        yaml_caps = load_yaml_params()["cluster_caps"]
        errors = []

        for key, fallback in fallbacks.items():
            if key not in yaml_caps:
                errors.append(f"  {key}: in code fallback but missing from parameters.yml")
                continue
            yaml_val = [float(v) for v in yaml_caps[key]]
            if fallback != yaml_val:
                errors.append(
                    f"  {key}: fallback={fallback} != yaml={yaml_val}"
                )

        assert not errors, (
            "Hardcoded cluster_caps fallbacks in app.py diverge from parameters.yml:\n"
            + "\n".join(errors)
        )

    def test_all_cluster_caps_have_fallback(self):
        """Every cluster_caps key in YAML should have a fallback in the code."""
        source = APP_PATH.read_text()
        fallbacks = extract_dict_get_fallbacks(source, "caps")
        yaml_caps = load_yaml_params()["cluster_caps"]

        missing = [k for k in yaml_caps if k not in fallbacks]
        assert not missing, (
            f"cluster_caps keys in YAML have no fallback in app.py: {missing}"
        )
