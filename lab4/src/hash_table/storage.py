"""JSON persistence for the command-line demo application."""

from __future__ import annotations

import json
from pathlib import Path

from hash_table.hash_table import HashTable


class JsonHashTableStorage:
    """Store a hash table snapshot in a human-readable JSON file."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def exists(self) -> bool:
        return self.path.exists()

    def load(self) -> HashTable[str]:
        if not self.path.exists():
            raise FileNotFoundError(
                f"State file '{self.path}' does not exist. Run 'init' first "
                "or use scripts/fill_demo_data.py."
            )

        data = json.loads(self.path.read_text(encoding="utf-8"))
        table = HashTable[str](size=int(data["size"]), base=int(data.get("base", 0)))

        for item in data.get("items", []):
            table.create(int(item["key"]), str(item["value"]))

        return table

    def save(self, table: HashTable[str]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "size": table.size,
            "base": table.base,
            "items": [
                {"key": key, "value": value}
                for key, value in sorted(table.items(), key=lambda item: item[0])
            ],
        }
        self.path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
