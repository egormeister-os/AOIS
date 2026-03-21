from __future__ import annotations

from fractions import Fraction

from src.converters.ones_complement import OnesComplementCodec
from src.converters.sign_magnitude import SignMagnitudeCodec
from src.converters.twos_complement import TwosComplementCodec
from src.core.bit_array32 import BitArray32
from src.operations.bcd_excess3_arithmetic import Excess3BCDArithmetic
from src.operations.float32_arithmetic import IEEE754Float32Arithmetic
from src.operations.integer_arithmetic import SignMagnitudeArithmetic, TwosComplementArithmetic


class LabService:
    """Facade for all lab tasks."""

    def __init__(self) -> None:
        self.sign_magnitude_codec = SignMagnitudeCodec()
        self.ones_codec = OnesComplementCodec()
        self.twos_codec = TwosComplementCodec()
        self.twos_ops = TwosComplementArithmetic()
        self.sign_ops = SignMagnitudeArithmetic()
        self.float_ops = IEEE754Float32Arithmetic()
        self.excess3_ops = Excess3BCDArithmetic()

    def convert_integer_representations(self, value: int) -> dict[str, object]:
        sign_magnitude = self.sign_magnitude_codec.encode(value)
        ones_complement = self.ones_codec.encode(value)
        twos_complement = self.twos_codec.encode(value)

        return {
            "decimal": value,
            "sign_magnitude": sign_magnitude,
            "ones_complement": ones_complement,
            "twos_complement": twos_complement,
            "decoded_sign_magnitude": self.sign_magnitude_codec.decode(sign_magnitude),
            "decoded_ones_complement": self.ones_codec.decode(ones_complement),
            "decoded_twos_complement": self.twos_codec.decode(twos_complement),
        }

    def add_twos_complement(self, left: int, right: int) -> dict[str, object]:
        left_bits = self.twos_codec.encode(left)
        right_bits = self.twos_codec.encode(right)
        result_bits = self.twos_ops.add(left_bits, right_bits)

        return {
            "left_bits": left_bits,
            "right_bits": right_bits,
            "result_bits": result_bits,
            "result_decimal": self.twos_codec.decode(result_bits),
        }

    def subtract_twos_complement(self, left: int, right: int) -> dict[str, object]:
        left_bits = self.twos_codec.encode(left)
        right_bits = self.twos_codec.encode(right)
        result_bits = self.twos_ops.subtract(left_bits, right_bits)

        return {
            "left_bits": left_bits,
            "right_bits": right_bits,
            "result_bits": result_bits,
            "result_decimal": self.twos_codec.decode(result_bits),
        }

    def multiply_sign_magnitude(self, left: int, right: int) -> dict[str, object]:
        left_bits = self.sign_magnitude_codec.encode(left)
        right_bits = self.sign_magnitude_codec.encode(right)
        result_bits = self.sign_ops.multiply(left_bits, right_bits)

        return {
            "left_bits": left_bits,
            "right_bits": right_bits,
            "result_bits": result_bits,
            "result_decimal": self.sign_magnitude_codec.decode(result_bits),
        }

    def divide_sign_magnitude(self, left: int, right: int, precision: int = 5) -> dict[str, object]:
        left_bits = self.sign_magnitude_codec.encode(left)
        right_bits = self.sign_magnitude_codec.encode(right)
        details = self.sign_ops.divide_with_details(left_bits, right_bits, precision)

        return {
            "left_bits": left_bits,
            "right_bits": right_bits,
            "result_bits": details["quotient_bits"],
            "result_binary": details["binary"],
            "result_decimal": details["decimal"],
        }

    def encode_float32(self, value: int | float | str | Fraction) -> BitArray32:
        return self.float_ops.encode_decimal(value)

    def decode_float32(self, bits: BitArray32) -> float:
        return self.float_ops.decode_to_decimal(bits)

    def add_float32(self, left: int | float | str | Fraction, right: int | float | str | Fraction) -> dict[str, object]:
        return self._float_operation(left, right, "add")

    def subtract_float32(self, left: int | float | str | Fraction, right: int | float | str | Fraction) -> dict[str, object]:
        return self._float_operation(left, right, "subtract")

    def multiply_float32(self, left: int | float | str | Fraction, right: int | float | str | Fraction) -> dict[str, object]:
        return self._float_operation(left, right, "multiply")

    def divide_float32(self, left: int | float | str | Fraction, right: int | float | str | Fraction) -> dict[str, object]:
        return self._float_operation(left, right, "divide")

    def add_excess3(self, left: int, right: int) -> dict[str, object]:
        result_bits, result_decimal = self.excess3_ops.add(left, right)

        return {
            "left_bits": self.excess3_ops.encode_number(left),
            "right_bits": self.excess3_ops.encode_number(right),
            "result_bits": result_bits,
            "result_decimal": result_decimal,
        }

    def _float_operation(self, left: int | float | str | Fraction, right: int | float | str | Fraction, operation: str) -> dict[str, object]:
        left_bits = self.float_ops.encode_decimal(left)
        right_bits = self.float_ops.encode_decimal(right)
        method = getattr(self.float_ops, operation)
        result_bits = method(left_bits, right_bits)

        return {
            "left_bits": left_bits,
            "right_bits": right_bits,
            "result_bits": result_bits,
            "result_decimal": self.float_ops.decode_to_decimal(result_bits),
        }
