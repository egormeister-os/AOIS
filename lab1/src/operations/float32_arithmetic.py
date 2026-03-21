from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

from src.core.bit_array32 import BitArray32


@dataclass(frozen=True)
class DecodedFloat32:
    kind: str
    sign: int = 0
    magnitude: Fraction = Fraction(0, 1)


class IEEE754Float32Arithmetic:
    """IEEE-754-2008 single-precision arithmetic over bit arrays."""

    EXPONENT_BIAS = 127
    FRACTION_BITS = 23
    EXPONENT_BITS = 8
    MAX_EXPONENT_FIELD = 255
    MAX_NORMAL_EXPONENT = 127
    MIN_NORMAL_EXPONENT = -126
    MIN_SUBNORMAL_EXPONENT = -149

    def add(self, left: BitArray32, right: BitArray32) -> BitArray32:
        left_value = self._decode_bits(left)
        right_value = self._decode_bits(right)
        return self._encode_decoded(self._add_decoded(left_value, right_value))

    def subtract(self, left: BitArray32, right: BitArray32) -> BitArray32:
        left_value = self._decode_bits(left)
        right_value = self._decode_bits(right)
        negated_right = self._negate(right_value)
        return self._encode_decoded(self._add_decoded(left_value, negated_right))

    def multiply(self, left: BitArray32, right: BitArray32) -> BitArray32:
        left_value = self._decode_bits(left)
        right_value = self._decode_bits(right)
        return self._encode_decoded(self._multiply_decoded(left_value, right_value))

    def divide(self, left: BitArray32, right: BitArray32) -> BitArray32:
        left_value = self._decode_bits(left)
        right_value = self._decode_bits(right)
        return self._encode_decoded(self._divide_decoded(left_value, right_value))

    def encode_decimal(self, value: int | float | str | Fraction) -> BitArray32:
        decoded = self._coerce_numeric_value(value)
        return self._encode_decoded(decoded)

    def decode_to_decimal(self, bits: BitArray32) -> float:
        decoded = self._decode_bits(bits)

        if decoded.kind == "nan":
            return float("nan")
        if decoded.kind == "inf":
            return float("-inf") if decoded.sign == 1 else float("inf")
        if decoded.magnitude == 0:
            return -0.0 if decoded.sign == 1 else 0.0

        value = decoded.magnitude
        if decoded.sign == 1:
            value = -value
        return value.numerator / value.denominator

    def _add_decoded(self, left: DecodedFloat32, right: DecodedFloat32) -> DecodedFloat32:
        if left.kind == "nan" or right.kind == "nan":
            return DecodedFloat32("nan")

        if left.kind == "inf" or right.kind == "inf":
            if left.kind == "inf" and right.kind == "inf" and left.sign != right.sign:
                return DecodedFloat32("nan")
            return left if left.kind == "inf" else right

        left_value = -left.magnitude if left.sign == 1 else left.magnitude
        right_value = -right.magnitude if right.sign == 1 else right.magnitude
        result = left_value + right_value

        if result == 0:
            return DecodedFloat32("finite", left.sign & right.sign, Fraction(0, 1))
        if result < 0:
            return DecodedFloat32("finite", 1, -result)
        return DecodedFloat32("finite", 0, result)

    def _multiply_decoded(self, left: DecodedFloat32, right: DecodedFloat32) -> DecodedFloat32:
        if left.kind == "nan" or right.kind == "nan":
            return DecodedFloat32("nan")

        sign = left.sign ^ right.sign

        if left.kind == "inf" or right.kind == "inf":
            if (left.kind == "finite" and left.magnitude == 0) or (right.kind == "finite" and right.magnitude == 0):
                return DecodedFloat32("nan")
            return DecodedFloat32("inf", sign)

        product = left.magnitude * right.magnitude
        return DecodedFloat32("finite", sign, product)

    def _divide_decoded(self, left: DecodedFloat32, right: DecodedFloat32) -> DecodedFloat32:
        if left.kind == "nan" or right.kind == "nan":
            return DecodedFloat32("nan")

        sign = left.sign ^ right.sign

        if left.kind == "inf" and right.kind == "inf":
            return DecodedFloat32("nan")
        if left.kind == "inf":
            return DecodedFloat32("inf", sign)
        if right.kind == "inf":
            return DecodedFloat32("finite", sign, Fraction(0, 1))

        if right.magnitude == 0:
            if left.magnitude == 0:
                return DecodedFloat32("nan")
            return DecodedFloat32("inf", sign)

        if left.magnitude == 0:
            return DecodedFloat32("finite", sign, Fraction(0, 1))

        return DecodedFloat32("finite", sign, left.magnitude / right.magnitude)

    def _negate(self, value: DecodedFloat32) -> DecodedFloat32:
        if value.kind == "nan":
            return value
        return DecodedFloat32(value.kind, 1 - value.sign, value.magnitude)

    def _coerce_numeric_value(self, value: int | float | str | Fraction) -> DecodedFloat32:
        if isinstance(value, Fraction):
            if value < 0:
                return DecodedFloat32("finite", 1, -value)
            return DecodedFloat32("finite", 0, value)

        if isinstance(value, int):
            if value < 0:
                return DecodedFloat32("finite", 1, Fraction(-value, 1))
            return DecodedFloat32("finite", 0, Fraction(value, 1))

        if isinstance(value, float):
            if value != value:
                return DecodedFloat32("nan")
            if value == float("inf"):
                return DecodedFloat32("inf", 0)
            if value == float("-inf"):
                return DecodedFloat32("inf", 1)
            return self._parse_decimal_string(repr(value))

        if isinstance(value, str):
            return self._parse_decimal_string(value)

        raise TypeError("Float32 values must be int, float, Fraction or decimal string.")

    def _parse_decimal_string(self, text: str) -> DecodedFloat32:
        normalized = text.strip().lower()
        if normalized in {"nan", "+nan", "-nan"}:
            return DecodedFloat32("nan")
        if normalized in {"inf", "+inf", "infinity", "+infinity"}:
            return DecodedFloat32("inf", 0)
        if normalized in {"-inf", "-infinity"}:
            return DecodedFloat32("inf", 1)

        sign = 0
        if normalized.startswith("-"):
            sign = 1
            normalized = normalized[1:]
        elif normalized.startswith("+"):
            normalized = normalized[1:]

        if "e" in normalized:
            base_part, exponent_part = normalized.split("e", 1)
            exponent10 = self._parse_signed_decimal_integer(exponent_part)
        else:
            base_part = normalized
            exponent10 = 0

        if "." in base_part:
            integer_part, fractional_part = base_part.split(".", 1)
        else:
            integer_part, fractional_part = base_part, ""

        if integer_part == "" and fractional_part == "":
            raise ValueError("Invalid decimal literal.")

        digits = integer_part + fractional_part
        if digits == "":
            digits = "0"
        if any(char < "0" or char > "9" for char in digits):
            raise ValueError("Invalid decimal literal.")

        numerator = self._parse_unsigned_decimal_integer(digits)
        scale = len(fractional_part) - exponent10

        if scale >= 0:
            magnitude = Fraction(numerator, 10 ** scale)
        else:
            magnitude = Fraction(numerator * (10 ** (-scale)), 1)

        return DecodedFloat32("finite", sign, magnitude)

    def _parse_unsigned_decimal_integer(self, digits: str) -> int:
        value = 0
        for char in digits:
            if char < "0" or char > "9":
                raise ValueError("Invalid decimal literal.")
            value = (value * 10) + (ord(char) - ord("0"))
        return value

    def _parse_signed_decimal_integer(self, digits: str) -> int:
        if digits == "":
            raise ValueError("Invalid decimal exponent.")

        sign = 1
        if digits.startswith("-"):
            sign = -1
            digits = digits[1:]
        elif digits.startswith("+"):
            digits = digits[1:]

        if digits == "":
            raise ValueError("Invalid decimal exponent.")

        return sign * self._parse_unsigned_decimal_integer(digits)

    def _decode_bits(self, bits: BitArray32) -> DecodedFloat32:
        sign = bits[0]
        exponent = self._bits_to_int(bits.bits[1:9])
        fraction = self._bits_to_int(bits.bits[9:])

        if exponent == self.MAX_EXPONENT_FIELD:
            if fraction == 0:
                return DecodedFloat32("inf", sign)
            return DecodedFloat32("nan", sign)

        if exponent == 0:
            if fraction == 0:
                return DecodedFloat32("finite", sign, Fraction(0, 1))
            return DecodedFloat32("finite", sign, Fraction(fraction, 1 << 149))

        significand = (1 << self.FRACTION_BITS) + fraction
        unbiased_exponent = exponent - self.EXPONENT_BIAS - self.FRACTION_BITS

        if unbiased_exponent >= 0:
            magnitude = Fraction(significand << unbiased_exponent, 1)
        else:
            magnitude = Fraction(significand, 1 << (-unbiased_exponent))

        return DecodedFloat32("finite", sign, magnitude)

    def _encode_decoded(self, value: DecodedFloat32) -> BitArray32:
        if value.kind == "nan":
            return BitArray32([0] + ([1] * self.EXPONENT_BITS) + [1] + ([0] * (self.FRACTION_BITS - 1)))

        if value.kind == "inf":
            return BitArray32([value.sign] + ([1] * self.EXPONENT_BITS) + ([0] * self.FRACTION_BITS))

        if value.magnitude == 0:
            return BitArray32([value.sign] + ([0] * (BitArray32.SIZE - 1)))

        exponent = self._floor_log2(value.magnitude)

        if exponent > self.MAX_NORMAL_EXPONENT:
            return BitArray32([value.sign] + ([1] * self.EXPONENT_BITS) + ([0] * self.FRACTION_BITS))

        if exponent >= self.MIN_NORMAL_EXPONENT:
            scaled_significand = self._scale_by_power_of_two(value.magnitude, self.FRACTION_BITS - exponent)
            rounded_significand = self._round_fraction_to_even(scaled_significand)

            if rounded_significand == (1 << (self.FRACTION_BITS + 1)):
                exponent += 1
                rounded_significand = 1 << self.FRACTION_BITS
                if exponent > self.MAX_NORMAL_EXPONENT:
                    return BitArray32([value.sign] + ([1] * self.EXPONENT_BITS) + ([0] * self.FRACTION_BITS))

            exponent_field = exponent + self.EXPONENT_BIAS
            fraction_field = rounded_significand - (1 << self.FRACTION_BITS)
            return BitArray32([value.sign] + self._int_to_bits(exponent_field, 8) + self._int_to_bits(fraction_field, 23))

        scaled_fraction = self._scale_by_power_of_two(value.magnitude, 149)
        fraction_field = self._round_fraction_to_even(scaled_fraction)

        if fraction_field == 0:
            return BitArray32([value.sign] + ([0] * (BitArray32.SIZE - 1)))

        if fraction_field >= (1 << self.FRACTION_BITS):
            return BitArray32([value.sign] + self._int_to_bits(1, 8) + ([0] * 23))

        return BitArray32([value.sign] + ([0] * 8) + self._int_to_bits(fraction_field, 23))

    def _floor_log2(self, value: Fraction) -> int:
        exponent = value.numerator.bit_length() - value.denominator.bit_length()
        if not self._fraction_at_least_power_of_two(value, exponent):
            exponent -= 1
        return exponent

    def _fraction_at_least_power_of_two(self, value: Fraction, exponent: int) -> bool:
        if exponent >= 0:
            return value.numerator >= (value.denominator << exponent)
        return (value.numerator << (-exponent)) >= value.denominator

    def _scale_by_power_of_two(self, value: Fraction, exponent: int) -> Fraction:
        if exponent >= 0:
            return value * (1 << exponent)
        return value / (1 << (-exponent))

    def _round_fraction_to_even(self, value: Fraction) -> int:
        integer_part = value.numerator // value.denominator
        remainder = value.numerator % value.denominator
        comparison = (remainder * 2) - value.denominator

        if comparison < 0:
            return integer_part
        if comparison > 0:
            return integer_part + 1
        if integer_part % 2 == 0:
            return integer_part
        return integer_part + 1

    def _bits_to_int(self, bits: list[int]) -> int:
        value = 0
        for bit in bits:
            value = (value * 2) + bit
        return value

    def _int_to_bits(self, value: int, width: int) -> list[int]:
        bits = [0] * width

        for idx in range(width - 1, -1, -1):
            bits[idx] = value % 2
            value //= 2

        return bits
