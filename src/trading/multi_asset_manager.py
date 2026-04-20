"""
src/trading/multi_asset_manager.py — Agregador multi-asset de topo.

Sincroniza com portfolios legados (portfolio.json, portfolio_eth.json)
mantendo visão agregada em capital_manager.json.

NÃO modifica os portfolios legados — zero breaking changes.
Os bots BTC e ETH continuam operando independentemente.

Uso:
    cm = MultiAssetManager()
    cm.sync_from_legacy()
    summary = cm.get_summary()
    bucket = cm.get_bucket("btc")
"""
import json
import logging
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import yaml

logger = logging.getLogger("multi_asset_manager")

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "conf/capital_manager.yml"


@dataclass
class BucketState:
    bucket_id: str
    asset: str
    initial_capital_usd: float
    current_capital_usd: float
    realized_pnl: float = 0.0
    has_position: bool = False
    entry_price: Optional[float] = None
    entry_price_usd: Optional[float] = None
    quantity: Optional[float] = None
    bot_origin: Optional[str] = None
    entry_timestamp: Optional[str] = None
    trailing_high: Optional[float] = None
    stop_loss_price: Optional[float] = None
    take_profit_price: Optional[float] = None
    n_trades_total: int = 0
    n_wins: int = 0
    n_losses: int = 0
    last_sync: Optional[str] = None
    bots_allowed: list = field(default_factory=list)
    legacy_portfolio_path: Optional[str] = None
    enabled: bool = True

    @property
    def pnl_pct(self) -> float:
        if self.initial_capital_usd == 0:
            return 0.0
        return (self.current_capital_usd - self.initial_capital_usd) / self.initial_capital_usd

    @property
    def win_rate(self) -> float:
        if self.n_trades_total == 0:
            return 0.0
        return self.n_wins / self.n_trades_total


@dataclass
class GlobalSummary:
    total_initial_capital: float
    total_current_capital: float
    total_realized_pnl: float
    total_pnl_pct: float
    active_positions: int
    n_buckets: int
    n_buckets_enabled: int
    timestamp: str


class MultiAssetManager:
    """
    Agregador central de capital multi-asset.

    Lê conf/capital_manager.yml e sincroniza com portfolios legados.
    Produz capital_manager.json como estado centralizado (read-only para os bots).
    """

    def __init__(self, config_path: Path = CONFIG_PATH):
        self.config_path = config_path
        self.config = self._load_config()
        self.state_path = ROOT / self.config["capital_manager"]["state_path"]
        self.buckets: dict[str, BucketState] = {}
        self._initialize_buckets()
        self._load_state()

    def _load_config(self) -> dict:
        with open(self.config_path) as f:
            return yaml.safe_load(f)

    def _initialize_buckets(self):
        buckets_cfg = self.config["capital_manager"]["buckets"]
        for bucket_id, cfg in buckets_cfg.items():
            self.buckets[bucket_id] = BucketState(
                bucket_id=bucket_id,
                asset=cfg["asset"],
                initial_capital_usd=cfg["initial_capital_usd"],
                current_capital_usd=cfg["initial_capital_usd"],
                bots_allowed=cfg.get("bots_allowed", []),
                legacy_portfolio_path=cfg.get("legacy_portfolio_path"),
                enabled=cfg.get("enabled", True),
            )

    def _load_state(self):
        """Carrega estado persistido se existir."""
        if not self.state_path.exists():
            return

        try:
            with open(self.state_path) as f:
                data = json.load(f)

            preserved = {"bucket_id", "asset", "initial_capital_usd",
                         "bots_allowed", "legacy_portfolio_path", "enabled"}

            for bucket_id, bucket_data in data.get("buckets", {}).items():
                if bucket_id not in self.buckets:
                    continue
                bucket = self.buckets[bucket_id]
                for k, v in bucket_data.items():
                    if k not in preserved and hasattr(bucket, k):
                        setattr(bucket, k, v)
        except Exception as e:
            logger.warning(f"Failed to load state from {self.state_path}: {e}")

    def sync_from_legacy(self):
        """
        Lê portfolios legados e atualiza buckets.

        Ponte entre portfolio.json / portfolio_eth.json e a visão centralizada.
        NÃO escreve de volta nos portfolios legados.
        """
        for bucket_id, bucket in self.buckets.items():
            if not bucket.enabled or not bucket.legacy_portfolio_path:
                continue

            legacy_path = ROOT / bucket.legacy_portfolio_path
            if not legacy_path.exists():
                logger.info(f"[{bucket_id}] Legacy portfolio not found: {legacy_path}")
                continue

            try:
                with open(legacy_path) as f:
                    legacy = json.load(f)

                bucket.current_capital_usd = legacy.get("capital_usd", bucket.initial_capital_usd)

                # BTC portfolio usa multi-bucket (buckets.btc_bot1 / btc_bot2)
                # Detecta se há posição em qualquer sub-bucket
                if "buckets" in legacy:
                    active = next(
                        (b for b in legacy["buckets"].values() if b.get("has_position")),
                        None,
                    )
                    bucket.has_position = active is not None
                    if active:
                        bucket.entry_price = active.get("entry_price")
                        bucket.quantity = active.get("quantity")
                        bucket.trailing_high = active.get("trailing_high")
                        bucket.stop_loss_price = active.get("stop_loss_price")
                        bucket.take_profit_price = active.get("take_profit_price")
                        bucket.entry_timestamp = active.get("entry_time")
                        bucket.bot_origin = active.get("entry_mode")
                    else:
                        bucket.entry_price = None
                        bucket.quantity = None
                        bucket.trailing_high = None
                        bucket.stop_loss_price = None
                        bucket.take_profit_price = None
                        bucket.entry_timestamp = None
                        bucket.bot_origin = None
                else:
                    # ETH portfolio (flat)
                    bucket.has_position = legacy.get("has_position", False)
                    bucket.entry_price = legacy.get("entry_price")
                    bucket.quantity = legacy.get("quantity")
                    bucket.trailing_high = legacy.get("trailing_high")
                    bucket.stop_loss_price = legacy.get("stop_loss_price")
                    bucket.take_profit_price = legacy.get("take_profit_price")
                    bucket.entry_timestamp = legacy.get("entry_timestamp")
                    bucket.bot_origin = legacy.get("entry_volume_z") and "bot_3_volume"

                # entry_price_usd
                if bucket.has_position and bucket.entry_price and bucket.quantity:
                    bucket.entry_price_usd = round(bucket.entry_price * bucket.quantity, 2)
                else:
                    bucket.entry_price_usd = None

                bucket.realized_pnl = bucket.current_capital_usd - bucket.initial_capital_usd
                bucket.last_sync = datetime.now(timezone.utc).isoformat()

                logger.debug(
                    f"[{bucket_id}] synced: capital=${bucket.current_capital_usd:,.2f} "
                    f"pnl={bucket.pnl_pct:+.2%} pos={bucket.has_position}"
                )

            except Exception as e:
                logger.warning(f"[{bucket_id}] Failed to sync from {legacy_path}: {e}")

    def save_state(self):
        """Persiste estado em capital_manager.json."""
        self.state_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "buckets": {bid: asdict(b) for bid, b in self.buckets.items()},
        }

        with open(self.state_path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def get_bucket(self, bucket_id: str) -> Optional[BucketState]:
        return self.buckets.get(bucket_id)

    def get_all_buckets(self) -> dict[str, BucketState]:
        return self.buckets

    def get_summary(self) -> GlobalSummary:
        enabled = [b for b in self.buckets.values() if b.enabled]
        total_initial = sum(b.initial_capital_usd for b in enabled)
        total_current = sum(b.current_capital_usd for b in enabled)
        total_pnl = total_current - total_initial
        total_pnl_pct = total_pnl / total_initial if total_initial else 0.0

        return GlobalSummary(
            total_initial_capital=total_initial,
            total_current_capital=total_current,
            total_realized_pnl=total_pnl,
            total_pnl_pct=total_pnl_pct,
            active_positions=sum(1 for b in enabled if b.has_position),
            n_buckets=len(self.buckets),
            n_buckets_enabled=len(enabled),
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    def get_total_pnl(self) -> float:
        return sum(b.realized_pnl for b in self.buckets.values() if b.enabled)

    def get_total_capital(self) -> float:
        return sum(b.current_capital_usd for b in self.buckets.values() if b.enabled)

    def check_global_kill_switch(self) -> bool:
        rules = self.config["capital_manager"].get("global_rules", {})
        if not rules.get("enabled", False):
            return False

        kill_switch = rules.get("global_kill_switch", {})
        if not kill_switch.get("enabled", False):
            return False

        summary = self.get_summary()
        max_dd_pct = kill_switch.get("max_drawdown_pct", -0.15)

        if summary.total_pnl_pct < max_dd_pct:
            logger.warning(
                f"[GLOBAL KILL SWITCH] Total DD {summary.total_pnl_pct:+.2%} "
                f"< threshold {max_dd_pct:+.2%}"
            )
            return True

        return False


# Alias para compatibilidade com o prompt
CapitalManager = MultiAssetManager


def get_capital_manager() -> MultiAssetManager:
    """Singleton-style factory para uso em scripts."""
    return MultiAssetManager()
