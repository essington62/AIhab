#!/usr/bin/env python3
"""
Capital Snapshot — sincroniza com portfolios legados e imprime estado.

Uso:
    python scripts/capital_snapshot.py
    python scripts/capital_snapshot.py --save    # persiste capital_manager.json
    python scripts/capital_snapshot.py --json    # output JSON
"""
import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.trading.multi_asset_manager import MultiAssetManager


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", action="store_true", help="Persist capital_manager.json")
    parser.add_argument("--json", action="store_true", help="Output JSON instead of text")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    cm = MultiAssetManager()
    cm.sync_from_legacy()

    if args.save:
        cm.save_state()
        print(f"State saved to {cm.state_path}")

    if args.json:
        import json
        from dataclasses import asdict
        data = {
            "summary": asdict(cm.get_summary()),
            "buckets": {bid: asdict(b) for bid, b in cm.get_all_buckets().items()},
        }
        print(json.dumps(data, indent=2, default=str))
        return 0

    summary = cm.get_summary()

    print("\n" + "=" * 60)
    print("CAPITAL MANAGER SNAPSHOT")
    print("=" * 60)
    print(f"Timestamp:           {summary.timestamp}")
    print(f"Total initial:       ${summary.total_initial_capital:>12,.2f}")
    print(f"Total current:       ${summary.total_current_capital:>12,.2f}")
    print(f"Total P&L:           ${summary.total_realized_pnl:>+12,.2f} ({summary.total_pnl_pct:+.2%})")
    print(f"Active positions:    {summary.active_positions}")
    print(f"Buckets enabled:     {summary.n_buckets_enabled}/{summary.n_buckets}")

    print("\n" + "-" * 60)
    print("BUCKETS")
    print("-" * 60)

    for bucket_id, bucket in cm.get_all_buckets().items():
        enabled = "OK" if bucket.enabled else "--"
        pos = "OPEN" if bucket.has_position else "CLOSED"
        print(f"\n[{enabled}] {bucket_id.upper()} ({bucket.asset}) — {pos}")
        print(f"  Initial:          ${bucket.initial_capital_usd:>12,.2f}")
        print(f"  Current:          ${bucket.current_capital_usd:>12,.2f}")
        print(f"  Realized P&L:     ${bucket.realized_pnl:>+12,.2f} ({bucket.pnl_pct:+.2%})")
        print(f"  Bots:             {', '.join(bucket.bots_allowed)}")

        if bucket.has_position:
            print(f"  Entry price:      ${bucket.entry_price:,.2f}")
            print(f"  Quantity:         {bucket.quantity:.6f}")
            print(f"  Position size:    ${bucket.entry_price_usd or 0:,.2f}")
            if bucket.stop_loss_price:
                print(f"  Stop loss:        ${bucket.stop_loss_price:,.2f}")
            if bucket.take_profit_price:
                print(f"  Take profit:      ${bucket.take_profit_price:,.2f}")
            if bucket.bot_origin:
                print(f"  Bot origin:       {bucket.bot_origin}")

        print(f"  Last sync:        {bucket.last_sync or 'never'}")

    if cm.check_global_kill_switch():
        print("\n GLOBAL KILL SWITCH TRIGGERED")

    print("\n" + "=" * 60 + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
