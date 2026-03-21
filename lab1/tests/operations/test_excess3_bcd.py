from src.operations.bcd_excess3_arithmetic import Excess3BCDArithmetic


def test_encode_number_in_excess3() -> None:
    arithmetic = Excess3BCDArithmetic()

    encoded = arithmetic.encode_number(59)

    assert encoded == [[1, 0, 0, 0], [1, 1, 0, 0]]


def test_addition_in_excess3_without_extra_carry() -> None:
    arithmetic = Excess3BCDArithmetic()

    result_bits, result_decimal = arithmetic.add(25, 37)

    assert result_bits == [[1, 0, 0, 1], [0, 1, 0, 1]]
    assert result_decimal == 62


def test_addition_in_excess3_with_most_significant_carry() -> None:
    arithmetic = Excess3BCDArithmetic()

    result_bits, result_decimal = arithmetic.add(59, 74)

    assert result_bits == [[0, 1, 0, 0], [0, 1, 1, 0], [0, 1, 1, 0]]
    assert result_decimal == 133
