import pytest

from src.converters.ones_complement import OnesComplementCodec
from src.converters.sign_magnitude import SignMagnitudeCodec
from src.converters.twos_complement import TwosComplementCodec
from src.core.bit_array32 import BitArray32


def test_sign_magnitude_codec_rejects_out_of_range_values() -> None:
    codec = SignMagnitudeCodec()

    with pytest.raises(ValueError):
        codec.encode(2 ** 31)


def test_ones_complement_codec_covers_positive_and_negative_zero_paths() -> None:
    codec = OnesComplementCodec()

    assert str(codec.encode(7)).endswith("0111")
    assert codec.decode(codec.encode(7)) == 7
    assert codec.decode(BitArray32([1] * 32)) == 0


def test_ones_complement_codec_rejects_out_of_range_values() -> None:
    codec = OnesComplementCodec()

    with pytest.raises(ValueError):
        codec.encode(-(2 ** 31))


def test_twos_complement_codec_rejects_out_of_range_values() -> None:
    codec = TwosComplementCodec()

    with pytest.raises(ValueError):
        codec.encode(-(2 ** 31) - 1)
