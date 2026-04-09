"""
src/config.py — Central config loader.
All modules call get_params() and get_path() — zero hardcoded paths or values.

Usage:
    from src.config import get_params, get_path, get_credentials

    params = get_params()
    path   = get_path("gate_zscores")        # → Path object
    key    = get_credentials("fred_api_key")
"""

from functools import lru_cache
from pathlib import Path

import yaml

# Project root = parent of src/
ROOT = Path(__file__).parent.parent

CATALOG_FILE = ROOT / "conf" / "catalog.yml"
PARAMS_FILE = ROOT / "conf" / "parameters.yml"
CREDS_FILE = ROOT / "conf" / "credentials.yml"
FED_CALENDAR_FILE = ROOT / "conf" / "fed_calendar.json"


@lru_cache(maxsize=1)
def get_params() -> dict:
    with open(PARAMS_FILE) as f:
        return yaml.safe_load(f)


@lru_cache(maxsize=1)
def _catalog() -> dict:
    with open(CATALOG_FILE) as f:
        return yaml.safe_load(f)


def get_path(name: str) -> Path:
    """Return absolute Path for a catalog entry."""
    catalog = _catalog()
    if name not in catalog:
        raise KeyError(f"'{name}' not found in catalog.yml. Available: {list(catalog)}")
    return ROOT / catalog[name]


@lru_cache(maxsize=1)
def get_credentials() -> dict:
    with open(CREDS_FILE) as f:
        return yaml.safe_load(f)


def get_credential(key: str) -> str:
    creds = get_credentials()
    if key not in creds:
        raise KeyError(f"'{key}' not found in credentials.yml")
    return creds[key]


@lru_cache(maxsize=1)
def get_fed_calendar() -> dict:
    import json
    with open(FED_CALENDAR_FILE) as f:
        return json.load(f)
