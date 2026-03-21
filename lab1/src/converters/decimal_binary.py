from src.core.bit_array32 import BitArray32


class DecimalBinaryConverter:
    """Manual decimal<->binary conversion utilities without built-in helpers."""

    def unsigned_to_bits(self, value: int) -> BitArray32:
        if value < 0:
            raise ValueError("unsigned_to_bits expects a non-negative integer.")
        if value >= 2 ** BitArray32.SIZE:
            raise ValueError("Value does not fit into 32 bits.")

        bits = BitArray32()

        i = BitArray32.SIZE - 1
        while value > 0:
            bits[i] = value % 2
            value //= 2
            i -= 1

        return bits

    def bits_to_unsigned(self, bits: BitArray32) -> int:
        value = 0

        for bit in bits.bits:
            value = (value * 2) + bit

        return value
