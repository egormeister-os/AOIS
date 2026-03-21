import math
from fractions import Fraction

import pytest

from src.core.bit_array32 import BitArray32
from src.operations.float32_arithmetic import DecodedFloat32, IEEE754Float32Arithmetic


def test_decode_to_decimal_handles_special_values_and_subnormals() -> None:
    arithmetic = IEEE754Float32Arithmetic()

    nan_bits = BitArray32([0] + ([1] * 8) + [1] + ([0] * 22))
    negative_infinity_bits = BitArray32([1] + ([1] * 8) + ([0] * 23))
    negative_zero_bits = BitArray32([1] + ([0] * 31))
    smallest_subnormal = BitArray32([0] + ([0] * 8) + ([0] * 22) + [1])

    assert math.isnan(arithmetic.decode_to_decimal(nan_bits))
    assert arithmetic.decode_to_decimal(negative_infinity_bits) == float("-inf")
    assert math.copysign(1.0, arithmetic.decode_to_decimal(negative_zero_bits)) == -1.0
    assert arithmetic.decode_to_decimal(smallest_subnormal) > 0.0
    assert arithmetic.decode_to_decimal(arithmetic.encode_decimal("16777216")) == 16777216.0


def test_float32_public_operations_cover_special_cases() -> None:
    arithmetic = IEEE754Float32Arithmetic()
    inf_bits = arithmetic.encode_decimal(float("inf"))
    neg_inf_bits = arithmetic.encode_decimal(float("-inf"))
    nan_bits = arithmetic.encode_decimal(float("nan"))
    zero_bits = arithmetic.encode_decimal("0.0")
    neg_zero_bits = arithmetic.encode_decimal("-0.0")
    one_bits = arithmetic.encode_decimal("1.0")
    two_bits = arithmetic.encode_decimal("2.0")

    assert math.isnan(arithmetic.decode_to_decimal(arithmetic.add(inf_bits, neg_inf_bits)))
    assert math.copysign(1.0, arithmetic.decode_to_decimal(arithmetic.add(neg_zero_bits, neg_zero_bits))) == -1.0
    assert arithmetic.decode_to_decimal(arithmetic.add(arithmetic.encode_decimal("-1.5"), arithmetic.encode_decimal("0.5"))) == -1.0

    assert math.isnan(arithmetic.decode_to_decimal(arithmetic.multiply(nan_bits, one_bits)))
    assert math.isnan(arithmetic.decode_to_decimal(arithmetic.multiply(inf_bits, zero_bits)))
    assert arithmetic.decode_to_decimal(arithmetic.multiply(neg_inf_bits, two_bits)) == float("-inf")

    assert math.isnan(arithmetic.decode_to_decimal(arithmetic.divide(nan_bits, one_bits)))
    assert math.isnan(arithmetic.decode_to_decimal(arithmetic.divide(inf_bits, neg_inf_bits)))
    assert arithmetic.decode_to_decimal(arithmetic.divide(neg_inf_bits, two_bits)) == float("-inf")
    assert arithmetic.decode_to_decimal(arithmetic.divide(two_bits, inf_bits)) == 0.0
    assert math.isnan(arithmetic.decode_to_decimal(arithmetic.divide(zero_bits, zero_bits)))
    assert arithmetic.decode_to_decimal(arithmetic.divide(zero_bits, two_bits)) == 0.0


def test_encode_decimal_accepts_multiple_input_forms() -> None:
    arithmetic = IEEE754Float32Arithmetic()

    assert arithmetic.decode_to_decimal(arithmetic.encode_decimal(Fraction(-3, 2))) == -1.5
    assert arithmetic.decode_to_decimal(arithmetic.encode_decimal(5)) == 5.0
    assert arithmetic.decode_to_decimal(arithmetic.encode_decimal(0.25)) == 0.25
    assert arithmetic.decode_to_decimal(arithmetic.encode_decimal("nan")) != arithmetic.decode_to_decimal(arithmetic.encode_decimal("nan"))
    assert arithmetic.decode_to_decimal(arithmetic.encode_decimal("+inf")) == float("inf")
    assert arithmetic.decode_to_decimal(arithmetic.encode_decimal("-infinity")) == float("-inf")
    assert arithmetic.decode_to_decimal(arithmetic.encode_decimal("+1.25e2")) == 125.0
    assert arithmetic.decode_to_decimal(arithmetic.encode_decimal("123")) == 123.0


def test_encode_decimal_rejects_invalid_inputs() -> None:
    arithmetic = IEEE754Float32Arithmetic()

    with pytest.raises(TypeError):
        arithmetic.encode_decimal(object())

    with pytest.raises(ValueError):
        arithmetic.encode_decimal("12a")

    with pytest.raises(ValueError):
        arithmetic.encode_decimal("1e+")

    with pytest.raises(ValueError):
        arithmetic.encode_decimal(".")


def test_float32_internal_helpers_cover_rounding_and_range_edges() -> None:
    arithmetic = IEEE754Float32Arithmetic()

    nan_value = DecodedFloat32("nan")
    assert arithmetic._negate(nan_value) is nan_value
    assert arithmetic._floor_log2(Fraction(1, 3)) == -2
    assert arithmetic._fraction_at_least_power_of_two(Fraction(3, 8), -1) is False
    assert arithmetic._scale_by_power_of_two(Fraction(3, 1), -1) == Fraction(3, 2)
    assert arithmetic._round_fraction_to_even(Fraction(14, 10)) == 1
    assert arithmetic._round_fraction_to_even(Fraction(16, 10)) == 2
    assert arithmetic._round_fraction_to_even(Fraction(5, 2)) == 2
    assert arithmetic._round_fraction_to_even(Fraction(7, 2)) == 4

    rounded_to_two = arithmetic._encode_decoded(
        DecodedFloat32("finite", 0, Fraction((1 << 25) - 1, 1 << 24))
    )
    assert str(rounded_to_two) == str(arithmetic.encode_decimal("2.0"))

    overflow_to_infinity = arithmetic._encode_decoded(
        DecodedFloat32("finite", 0, Fraction((1 << 25) - 1, 1 << 24) * (1 << 127))
    )
    assert overflow_to_infinity == arithmetic.encode_decimal(float("inf"))

    underflow_to_zero = arithmetic._encode_decoded(
        DecodedFloat32("finite", 0, Fraction(1, 1 << 200))
    )
    assert underflow_to_zero == BitArray32()

    rounded_to_smallest_normal = arithmetic._encode_decoded(
        DecodedFloat32("finite", 0, Fraction((1 << 24) - 1, 1 << 150))
    )
    assert str(rounded_to_smallest_normal) == "00000000100000000000000000000000"

    minimum_subnormal = arithmetic._encode_decoded(
        DecodedFloat32("finite", 0, Fraction(1, 1 << 149))
    )
    assert str(minimum_subnormal) == "00000000000000000000000000000001"
