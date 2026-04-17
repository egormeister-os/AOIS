"""Demo data helpers for the hash table CLI."""

from __future__ import annotations

from pathlib import Path

from hash_table.hash_table import HashTable
from hash_table.storage import JsonHashTableStorage

DEMO_SIZE = 5
DEMO_BASE = 100
DEMO_ITEMS: tuple[tuple[int, str], ...] = (
    (1, "Alice"),
    (6, "Bob"),
    (11, "Charlie"),
    (2, "Diana"),
    (7, "Eve"),
)


def build_demo_table(size: int = DEMO_SIZE, base: int = DEMO_BASE) -> HashTable[str]:
    """Build a table with intentional collisions for demonstration."""

    table = HashTable[str](size=size, base=base)
    for key, value in DEMO_ITEMS:
        table.create(key, value)

    return table


def save_demo_table(path: str | Path, size: int = DEMO_SIZE, base: int = DEMO_BASE) -> HashTable[str]:
    """Save demo data to a JSON state file and return the created table."""

    table = build_demo_table(size=size, base=base)
    JsonHashTableStorage(path).save(table)
    return table
