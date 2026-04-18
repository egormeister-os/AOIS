"""Command-line interface for manually checking hash table behavior."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

# Allows running this file directly as:
# python src/hash_table/cli.py ...
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hash_table.demo_data import DEMO_BASE, DEMO_SIZE, save_demo_table
from hash_table.exceptions import HashTableError
from hash_table.hash_table import HashTable
from hash_table.storage import JsonHashTableStorage

DEFAULT_STATE_FILE = Path("hash_table_state.json")


class CliError(Exception):
    """Raised for user-facing CLI errors."""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hash-table",
        description="Hash table CLI with linked-list collision chains.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  hash-table --state state.json init --size 5 --base 100\n"
            "  hash-table --state state.json create 11 Alice\n"
            "  hash-table --state state.json read 11\n"
            "  hash-table --state state.json update 11 Alicia\n"
            "  hash-table --state state.json delete 11\n"
            "  hash-table --state state.json hash 11\n"
            "  hash-table --state state.json list\n"
            "  hash-table --state state.json demo --force\n"
        ),
    )
    parser.add_argument(
        "--state",
        default=DEFAULT_STATE_FILE,
        type=Path,
        help=f"Path to JSON state file. Default: {DEFAULT_STATE_FILE}",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init", help="Create an empty table state file.")
    init_parser.add_argument("--size", type=int, required=True, help="H: number of buckets.")
    init_parser.add_argument("--base", type=int, default=0, help="B: hash base offset.")
    init_parser.add_argument("--force", action="store_true", help="Overwrite existing state file.")
    init_parser.set_defaults(handler=_handle_init)

    create_parser = subparsers.add_parser("create", help="Create a key-value pair.")
    create_parser.add_argument("key", type=int)
    create_parser.add_argument("value")
    create_parser.set_defaults(handler=_handle_create)

    read_parser = subparsers.add_parser("read", help="Read a value by key.")
    read_parser.add_argument("key", type=int)
    read_parser.set_defaults(handler=_handle_read)

    update_parser = subparsers.add_parser("update", help="Update an existing key.")
    update_parser.add_argument("key", type=int)
    update_parser.add_argument("value")
    update_parser.set_defaults(handler=_handle_update)

    delete_parser = subparsers.add_parser("delete", help="Delete a key.")
    delete_parser.add_argument("key", type=int)
    delete_parser.set_defaults(handler=_handle_delete)

    hash_parser = subparsers.add_parser("hash", help="Show h(V) and bucket index for a key.")
    hash_parser.add_argument("key", type=int)
    hash_parser.set_defaults(handler=_handle_hash)

    list_parser = subparsers.add_parser(
        "list",
        help="Print all buckets and chains in a cell-style table view.",
    )
    list_parser.set_defaults(handler=_handle_list)

    demo_parser = subparsers.add_parser("demo", help="Fill the state file with demo data.")
    demo_parser.add_argument("--size", type=int, default=DEMO_SIZE, help="H: number of buckets.")
    demo_parser.add_argument("--base", type=int, default=DEMO_BASE, help="B: hash base offset.")
    demo_parser.add_argument("--force", action="store_true", help="Overwrite existing state file.")
    demo_parser.set_defaults(handler=_handle_demo)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        message = args.handler(args)
    except (
        CliError,
        FileNotFoundError,
        HashTableError,
        json.JSONDecodeError,
        KeyError,
        TypeError,
        ValueError,
    ) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    print(message)
    return 0


def _handle_init(args: argparse.Namespace) -> str:
    storage = JsonHashTableStorage(args.state)
    if storage.exists() and not args.force:
        raise CliError(f"State file '{args.state}' already exists. Use --force to overwrite it.")

    table = HashTable[str](size=args.size, base=args.base)
    storage.save(table)
    return f"Created empty table: H={table.size}, B={table.base}, state='{args.state}'."


def _handle_create(args: argparse.Namespace) -> str:
    storage = JsonHashTableStorage(args.state)
    table = storage.load()
    table.create(args.key, args.value)
    storage.save(table)

    return (
        f"Created key={args.key}, value='{args.value}', "
        f"h(V)={table.hash(args.key)}, bucket={_bucket_index(table, args.key)}."
    )


def _handle_read(args: argparse.Namespace) -> str:
    table = JsonHashTableStorage(args.state).load()
    value = table.read(args.key)

    return (
        f"Read key={args.key}: value='{value}', "
        f"h(V)={table.hash(args.key)}, bucket={_bucket_index(table, args.key)}."
    )


def _handle_update(args: argparse.Namespace) -> str:
    storage = JsonHashTableStorage(args.state)
    table = storage.load()
    table.update(args.key, args.value)
    storage.save(table)

    return f"Updated key={args.key}: value='{args.value}'."


def _handle_delete(args: argparse.Namespace) -> str:
    storage = JsonHashTableStorage(args.state)
    table = storage.load()
    removed_value = table.delete(args.key)
    storage.save(table)

    return f"Deleted key={args.key}: removed value='{removed_value}'."


def _handle_hash(args: argparse.Namespace) -> str:
    table = JsonHashTableStorage(args.state).load()
    return f"key={args.key}: h(V)={table.hash(args.key)}, bucket={_bucket_index(table, args.key)}."


def _handle_list(args: argparse.Namespace) -> str:
    table = JsonHashTableStorage(args.state).load()
    lines = [
        f"Table H={table.size}, B={table.base}, count={len(table)}, load_factor={table.load_factor:.2f}"
    ]

    for index, bucket in enumerate(table.buckets()):
        chain = " -> ".join(f"{key}='{value}'" for key, value in bucket)
        lines.append("")
        lines.append(f"bucket {index} (h={table.base + index}): {chain or 'empty'}")
        lines.extend(_format_bucket_table(bucket))

    return "\n".join(lines)


def _handle_demo(args: argparse.Namespace) -> str:
    storage = JsonHashTableStorage(args.state)
    if storage.exists() and not args.force:
        raise CliError(f"State file '{args.state}' already exists. Use --force to overwrite it.")

    table = save_demo_table(args.state, size=args.size, base=args.base)
    return f"Saved demo table: H={table.size}, B={table.base}, count={len(table)}, state='{args.state}'."


def _format_bucket_table(bucket: tuple[tuple[int, str], ...]) -> list[str]:
    headers = ("ID", "C", "T", "Po", "Pi")

    if not bucket:
        return []

    rows = []
    collision_flag = "1" if len(bucket) > 1 else "0"
    last_index = len(bucket) - 1

    for row_index, (key, value) in enumerate(bucket):
        is_terminal = row_index == last_index
        rows.append(
            (
                str(key),
                collision_flag,
                "1" if is_terminal else "0",
                "-" if is_terminal else f"#{row_index + 2}",
                _format_cell_value(value),
            )
        )

    return _render_ascii_table(headers, rows)


def _render_ascii_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    widths = [len(header) for header in headers]

    for row in rows:
        widths = [max(width, len(cell)) for width, cell in zip(widths, row)]

    border = "+" + "+".join("-" * (width + 2) for width in widths) + "+"
    lines = [border, _render_table_row(headers, widths), border]
    lines.extend(_render_table_row(row, widths) for row in rows)
    lines.append(border)
    return lines


def _render_table_row(cells: Sequence[str], widths: Sequence[int]) -> str:
    padded_cells = [f" {cell:<{width}} " for cell, width in zip(cells, widths)]
    return "|" + "|".join(padded_cells) + "|"


def _format_cell_value(value: str) -> str:
    return value.replace("\\", "\\\\").replace("\n", "\\n").replace("|", "\\|")


def _bucket_index(table: HashTable[str], key: int) -> int:
    return table.hash(key) - table.base


if __name__ == "__main__":
    raise SystemExit(main())
