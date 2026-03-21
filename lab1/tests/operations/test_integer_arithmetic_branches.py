import pytest

from src.converters.sign_magnitude import SignMagnitudeCodec
from src.operations.integer_arithmetic import SignMagnitudeArithmetic


def _magnitude(codec: SignMagnitudeCodec, value: int) -> list[int]:
    return codec.encode(value).bits[1:]


def test_sign_magnitude_multiplication_handles_zero_and_overflow() -> None:
    codec = SignMagnitudeCodec()
    arithmetic = SignMagnitudeArithmetic()

    zero_result = arithmetic.multiply(codec.encode(0), codec.encode(-7))
    assert codec.decode(zero_result) == 0
    assert zero_result[0] == 0

    with pytest.raises(OverflowError):
        arithmetic.multiply(codec.encode(1 << 30), codec.encode(4))


def test_sign_magnitude_division_validates_arguments_and_sign() -> None:
    codec = SignMagnitudeCodec()
    arithmetic = SignMagnitudeArithmetic()

    with pytest.raises(ValueError):
        arithmetic.divide_with_details(codec.encode(1), codec.encode(1), precision=-1)

    with pytest.raises(ZeroDivisionError):
        arithmetic.divide_with_details(codec.encode(1), codec.encode(0))

    zero_details = arithmetic.divide_with_details(codec.encode(0), codec.encode(-7), precision=5)
    assert zero_details["binary"] == "0.00000"
    assert codec.decode(zero_details["quotient_bits"]) == 0

    negative_details = arithmetic.divide_with_details(codec.encode(-1), codec.encode(2), precision=5)
    assert negative_details["binary"] == "-0.10000"
    assert negative_details["decimal"] == -0.5


def test_sign_magnitude_divide_method_and_helper_branches() -> None:
    codec = SignMagnitudeCodec()
    arithmetic = SignMagnitudeArithmetic()

    quotient_bits, decimal_value = arithmetic.divide(codec.encode(3), codec.encode(2), precision=0)
    assert codec.decode(quotient_bits) == 1
    assert decimal_value == 2.0

    result = arithmetic._subtract_magnitudes(_magnitude(codec, 4), _magnitude(codec, 1))
    assert result == _magnitude(codec, 3)

    shifted, overflow = arithmetic._shift_left_magnitude([1] + ([0] * 30), arithmetic.MAGNITUDE_SIZE)
    assert shifted == [0] * arithmetic.MAGNITUDE_SIZE
    assert overflow is True

    assert arithmetic._format_binary_quotient(_magnitude(codec, 5), "", 1) == "-101"
    assert arithmetic._decimal_division(_magnitude(codec, 3), _magnitude(codec, 2), 0, 0) == 2.0
    assert arithmetic._decimal_division(_magnitude(codec, 199996), _magnitude(codec, 100000), 0, 4) == 2.0
    assert arithmetic._decimal_division(_magnitude(codec, 1), _magnitude(codec, 2), 1, 1) == -0.5
