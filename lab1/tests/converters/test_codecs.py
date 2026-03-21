from src.converters.ones_complement import OnesComplementCodec
from src.converters.sign_magnitude import SignMagnitudeCodec
from src.converters.twos_complement import TwosComplementCodec
from src.core.bit_array32 import BitArray32


def test_sign_magnitude_roundtrip_for_negative_value() -> None:
    codec = SignMagnitudeCodec()

    encoded = codec.encode(-42)

    assert encoded[0] == 1
    assert codec.decode(encoded) == -42


def test_ones_complement_roundtrip_for_negative_value() -> None:
    codec = OnesComplementCodec()

    encoded = codec.encode(-13)

    assert encoded[0] == 1
    assert codec.decode(encoded) == -13


def test_twos_complement_roundtrip_for_negative_value() -> None:
    codec = TwosComplementCodec()

    encoded = codec.encode(-19)

    assert encoded[0] == 1
    assert codec.decode(encoded) == -19


def test_twos_complement_encodes_min_value() -> None:
    codec = TwosComplementCodec()

    encoded = codec.encode(-(2 ** 31))

    assert encoded == BitArray32([1] + ([0] * 31))
