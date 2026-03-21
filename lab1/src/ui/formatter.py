from __future__ import annotations

from src.core.bit_array32 import BitArray32


class OutputFormatter:
    """Responsible only for rendering operation results."""

    def bits_and_decimal(self, bits: BitArray32, decimal_value: int | float) -> str:
        return f"Binary: {bits}\nDecimal: {decimal_value}"

    def binary_string_and_decimal(self, binary_value: str, decimal_value: int | float) -> str:
        return f"Binary: {binary_value}\nDecimal: {decimal_value}"

    def conversion_report(self, result: dict[str, object]) -> str:
        return (
            f"Decimal: {result['decimal']}\n"
            f"Sign-magnitude: {result['sign_magnitude']}\n"
            f"Ones' complement: {result['ones_complement']}\n"
            f"Two's complement: {result['twos_complement']}"
        )

    def excess3_digits(self, digits: list[list[int]]) -> str:
        return " ".join("".join(str(bit) for bit in digit) for digit in digits)
