from src.core.bit_array32 import BitArray32
from src.converters.decimal_binary import DecimalBinaryConverter
from src.core.interfaces import IntegerCodec


class OnesComplementCodec(IntegerCodec):
    """Ones' complement integer representation codec."""

    MIN_VALUE = -(2 ** 31 - 1)
    MAX_VALUE = (2 ** 31) - 1

    def encode(self, value: int) -> BitArray32:
        if value < self.MIN_VALUE or value > self.MAX_VALUE:
            raise ValueError("Value does not fit ones' complement 32-bit range.")

        converter = DecimalBinaryConverter()
        if value >= 0:
            return converter.unsigned_to_bits(value)

        magnitude_bits = converter.unsigned_to_bits(abs(value))
        return magnitude_bits.invert()

    def decode(self, input_bits: BitArray32) -> int:
        bits = input_bits.copy()
        converter = DecimalBinaryConverter()

        if bits[0] == 0:
            return converter.bits_to_unsigned(bits)

        if all(bit == 1 for bit in bits):
            return 0

        magnitude_bits = bits.invert()
        return -converter.bits_to_unsigned(magnitude_bits)
