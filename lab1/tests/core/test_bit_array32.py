from src.core.bit_array32 import BitArray32
import pytest


def test_invert_returns_new_instance_without_mutating_original() -> None:
    original = BitArray32([0] * 31 + [1])

    inverted = original.invert()

    assert str(original) == ("0" * 31) + "1"
    assert str(inverted) == ("1" * 31) + "0"


def test_add_one_produces_new_bit_array() -> None:
    original = BitArray32([0] * 30 + [1, 1])

    incremented = original.add_one()

    assert str(original) == ("0" * 30) + "11"
    assert str(incremented) == ("0" * 29) + "100"


def test_bit_array32_validates_input_and_helpers() -> None:
    with pytest.raises(ValueError):
        BitArray32([0, 1])

    with pytest.raises(ValueError):
        BitArray32([0] * 31 + [2])

    bits = BitArray32()
    assert len(bits) == 32
    assert list(bits) == [0] * 32
    assert bits.is_zero() is True
    assert repr(bits) == "BitArray32(bits='00000000000000000000000000000000')"
    assert bits == BitArray32()
    assert (bits == object()) is False


def test_bit_array32_setitem_validates_value() -> None:
    bits = BitArray32()

    with pytest.raises(ValueError):
        bits[0] = 2
