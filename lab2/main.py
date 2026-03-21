from __future__ import annotations

import sys

from boollab.cli import build_report
from boollab.core import ExpressionError


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    expression = " ".join(args).strip() if args else input("Введите логическую функцию: ").strip()
    try:
        print(build_report(expression))
    except ExpressionError as error:
        print(f"Ошибка: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
