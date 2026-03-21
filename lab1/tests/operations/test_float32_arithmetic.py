import math

from src.operations.float32_arithmetic import IEEE754Float32Arithmetic


def test_encode_decimal_produces_expected_bits_for_5_75() -> None:
    arithmetic = IEEE754Float32Arithmetic()

    bits = arithmetic.encode_decimal("5.75")

    assert str(bits) == "01000000101110000000000000000000"
    assert arithmetic.decode_to_decimal(bits) == 5.75


def test_float32_addition_matches_expected_result() -> None:
    arithmetic = IEEE754Float32Arithmetic()

    result = arithmetic.add(arithmetic.encode_decimal("1.5"), arithmetic.encode_decimal("2.25"))

    assert str(result) == "01000000011100000000000000000000"
    assert arithmetic.decode_to_decimal(result) == 3.75


def test_float32_subtraction_matches_expected_result() -> None:
    arithmetic = IEEE754Float32Arithmetic()

    result = arithmetic.subtract(arithmetic.encode_decimal("7.5"), arithmetic.encode_decimal("2.25"))

    assert str(result) == "01000000101010000000000000000000"
    assert arithmetic.decode_to_decimal(result) == 5.25


def test_float32_multiplication_matches_expected_result() -> None:
    arithmetic = IEEE754Float32Arithmetic()

    result = arithmetic.multiply(arithmetic.encode_decimal("-1.5"), arithmetic.encode_decimal("2.0"))

    assert str(result) == "11000000010000000000000000000000"
    assert arithmetic.decode_to_decimal(result) == -3.0


def test_float32_division_matches_expected_result() -> None:
    arithmetic = IEEE754Float32Arithmetic()

    result = arithmetic.divide(arithmetic.encode_decimal("7.0"), arithmetic.encode_decimal("2.0"))

    assert str(result) == "01000000011000000000000000000000"
    assert arithmetic.decode_to_decimal(result) == 3.5


def test_float32_division_by_zero_produces_infinity() -> None:
    arithmetic = IEEE754Float32Arithmetic()

    result = arithmetic.divide(arithmetic.encode_decimal("1.0"), arithmetic.encode_decimal("0.0"))

    assert str(result) == "01111111100000000000000000000000"
    assert math.isinf(arithmetic.decode_to_decimal(result))
