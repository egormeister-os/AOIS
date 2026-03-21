import pytest

from src.converters.decimal_binary import DecimalBinaryConverter


def test_decimal_binary_converter_roundtrip() -> None:
    converter = DecimalBinaryConverter()

    bits = converter.unsigned_to_bits(42)

    assert converter.bits_to_unsigned(bits) == 42


def test_decimal_binary_converter_rejects_invalid_input() -> None:
    converter = DecimalBinaryConverter()

    with pytest.raises(ValueError):
        converter.unsigned_to_bits(-1)

    with pytest.raises(ValueError):
        converter.unsigned_to_bits(2 ** 32)
