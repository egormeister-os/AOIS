from src.core.bit_array32 import BitArray32
from src.ui.formatter import OutputFormatter


def test_output_formatter_renders_supported_views() -> None:
    formatter = OutputFormatter()
    bits = BitArray32()

    assert formatter.bits_and_decimal(bits, 0) == f"Binary: {bits}\nDecimal: 0"
    assert formatter.binary_string_and_decimal("101.1", 5.5) == "Binary: 101.1\nDecimal: 5.5"
    assert formatter.conversion_report(
        {
            "decimal": -5,
            "sign_magnitude": "1001",
            "ones_complement": "1110",
            "twos_complement": "1111",
        }
    ) == "Decimal: -5\nSign-magnitude: 1001\nOnes' complement: 1110\nTwo's complement: 1111"
    assert formatter.excess3_digits([[0, 0, 1, 1], [0, 1, 0, 0]]) == "0011 0100"
