from src.converters.sign_magnitude import SignMagnitudeCodec
from src.converters.twos_complement import TwosComplementCodec
from src.operations.integer_arithmetic import SignMagnitudeArithmetic, TwosComplementArithmetic


def test_twos_complement_addition_returns_expected_sum() -> None:
    codec = TwosComplementCodec()
    arithmetic = TwosComplementArithmetic()

    result = arithmetic.add(codec.encode(5), codec.encode(-3))

    assert codec.decode(result) == 2


def test_twos_complement_subtraction_uses_negation_without_mutating_operand() -> None:
    codec = TwosComplementCodec()
    arithmetic = TwosComplementArithmetic()
    left = codec.encode(5)
    right = codec.encode(7)
    right_before = right.copy()

    result = arithmetic.subtract(left, right)

    assert codec.decode(result) == -2
    assert right == right_before


def test_sign_magnitude_multiplication_returns_expected_product() -> None:
    codec = SignMagnitudeCodec()
    arithmetic = SignMagnitudeArithmetic()

    result = arithmetic.multiply(codec.encode(-6), codec.encode(5))

    assert codec.decode(result) == -30


def test_sign_magnitude_division_returns_bits_binary_and_decimal() -> None:
    codec = SignMagnitudeCodec()
    arithmetic = SignMagnitudeArithmetic()

    details = arithmetic.divide_with_details(codec.encode(13), codec.encode(2), precision=5)

    assert codec.decode(details["quotient_bits"]) == 6
    assert details["binary"] == "110.10000"
    assert details["decimal"] == 6.5
