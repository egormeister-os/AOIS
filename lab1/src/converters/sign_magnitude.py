from src.core.bit_array32 import BitArray32
from src.converters.decimal_binary import DecimalBinaryConverter
from src.core.interfaces import IntegerCodec

class SignMagnitudeCodec(IntegerCodec):
    """Sign-magnitude integer representation codec."""

    def encode(self, value: int) -> BitArray32:
        if abs(value) > (2**31 - 1):
            raise ValueError("Value does not fit sign-magnitude 32-bit range.")

        converter = DecimalBinaryConverter()
        bits = converter.unsigned_to_bits(abs(value))
        if value >= 0:
            bits[0] = 0

        else:
            bits[0] = 1

        return bits


    def decode(self, bits: BitArray32) -> int:
        value = 0

        for bit in bits.bits[1:]:
            value = (value * 2) + bit

        if bits.bits[0] == 1:
            return -value
        
        else:
            return value
