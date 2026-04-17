#!/usr/bin/env python3
"""Create a JSON state file with demo hash table data."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hash_table.demo_data import DEMO_BASE, DEMO_SIZE, save_demo_table  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Fill a hash table state file with demo data.")
    parser.add_argument(
        "--state",
        default=Path("demo_hash_table_state.json"),
        type=Path,
        help="Path to the JSON state file.",
    )
    parser.add_argument("--size", type=int, default=DEMO_SIZE, help="H: number of buckets.")
    parser.add_argument("--base", type=int, default=DEMO_BASE, help="B: hash base offset.")
    args = parser.parse_args()

    table = save_demo_table(args.state, size=args.size, base=args.base)
    print(f"Saved demo table to '{args.state}'.")
    print(f"H={table.size}, B={table.base}, count={len(table)}")
    print(f"Check it with: python scripts/hash_table_cli.py --state {args.state} list")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
