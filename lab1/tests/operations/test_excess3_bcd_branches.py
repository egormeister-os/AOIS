import pytest

from src.operations.bcd_excess3_arithmetic import Excess3BCDArithmetic


def test_excess3_rejects_negative_values_and_invalid_digits() -> None:
    arithmetic = Excess3BCDArithmetic()

    with pytest.raises(ValueError):
        arithmetic.encode_number(-1)

    with pytest.raises(ValueError):
        arithmetic.add(-1, 5)

    with pytest.raises(ValueError):
        arithmetic.decode_number([[0, 0, 0, 1]])

    with pytest.raises(ValueError):
        arithmetic._encode_digit(10)


def test_excess3_handles_zero_encoding() -> None:
    arithmetic = Excess3BCDArithmetic()

    assert arithmetic.encode_number(0) == [[0, 0, 1, 1]]
