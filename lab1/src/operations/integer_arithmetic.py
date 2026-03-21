from __future__ import annotations

from src.core.bit_array32 import BitArray32


class TwosComplementArithmetic:
    """Addition and subtraction in two's complement."""

    def add(self, left: BitArray32, right: BitArray32) -> BitArray32:
        result_bits = BitArray32()
        carry = 0

        for bit_idx in range(left.SIZE - 1, -1, -1):
            bit_sum = left[bit_idx] + right[bit_idx] + carry
            result_bits[bit_idx] = bit_sum % 2
            carry = bit_sum // 2

        return result_bits

    def subtract(self, left: BitArray32, right: BitArray32) -> BitArray32:
        """Implement as left + (-right)."""
        negated_right = right.invert().add_one()
        return self.add(left, negated_right)


class SignMagnitudeArithmetic:
    """Multiplication and division in sign-magnitude representation."""

    MAGNITUDE_SIZE = 31

    def multiply(self, left: BitArray32, right: BitArray32) -> BitArray32:
        left_magnitude = self._extract_magnitude(left)
        right_magnitude = self._extract_magnitude(right)
        sign = left[0] ^ right[0]
        result = [0] * self.MAGNITUDE_SIZE

        for shift, multiplier_bit in enumerate(reversed(right_magnitude)):
            if multiplier_bit == 0:
                continue

            shifted_multiplicand, overflow = self._shift_left_magnitude(left_magnitude, shift)
            result, carry = self._add_magnitudes(result, shifted_multiplicand)

            if overflow or carry:
                raise OverflowError("Multiplication result does not fit sign-magnitude 32-bit range.")

        if self._is_zero_magnitude(result):
            sign = 0

        return self._combine_sign_and_magnitude(sign, result)

    def divide(self, left: BitArray32, right: BitArray32, precision: int = 5) -> tuple[BitArray32, float]:
        details = self.divide_with_details(left, right, precision)
        return details["quotient_bits"], details["decimal"]

    def divide_with_details(self, left: BitArray32, right: BitArray32, precision: int = 5) -> dict[str, BitArray32 | float | str]:
        if precision < 0:
            raise ValueError("Precision must be non-negative.")

        left_magnitude = self._extract_magnitude(left)
        right_magnitude = self._extract_magnitude(right)

        if self._is_zero_magnitude(right_magnitude):
            raise ZeroDivisionError("Division by zero is undefined.")

        sign = left[0] ^ right[0]
        quotient_magnitude, remainder = self._divide_magnitudes(left_magnitude, right_magnitude)
        fractional_bits = self._divide_fractional_bits(remainder, right_magnitude, precision)

        if self._is_zero_magnitude(quotient_magnitude) and all(bit == "0" for bit in fractional_bits):
            sign = 0

        return {
            "quotient_bits": self._combine_sign_and_magnitude(sign, quotient_magnitude),
            "binary": self._format_binary_quotient(quotient_magnitude, fractional_bits, sign),
            "decimal": self._decimal_division(left_magnitude, right_magnitude, sign, precision),
        }

    def _extract_magnitude(self, bits: BitArray32) -> list[int]:
        return bits.bits[1:]

    def _combine_sign_and_magnitude(self, sign: int, magnitude: list[int]) -> BitArray32:
        return BitArray32([sign] + magnitude)

    def _is_zero_magnitude(self, magnitude: list[int]) -> bool:
        return all(bit == 0 for bit in magnitude)

    def _add_magnitudes(self, left: list[int], right: list[int]) -> tuple[list[int], int]:
        result = [0] * self.MAGNITUDE_SIZE
        carry = 0

        for idx in range(self.MAGNITUDE_SIZE - 1, -1, -1):
            bit_sum = left[idx] + right[idx] + carry
            result[idx] = bit_sum % 2
            carry = bit_sum // 2

        return result, carry

    def _subtract_magnitudes(self, left: list[int], right: list[int]) -> list[int]:
        result = [0] * self.MAGNITUDE_SIZE
        borrow = 0

        for idx in range(self.MAGNITUDE_SIZE - 1, -1, -1):
            difference = left[idx] - right[idx] - borrow
            if difference >= 0:
                result[idx] = difference
                borrow = 0
            else:
                result[idx] = difference + 2
                borrow = 1

        return result

    def _shift_left_magnitude(self, magnitude: list[int], shift: int) -> tuple[list[int], bool]:
        if shift == 0:
            return magnitude[:], False
        if shift >= self.MAGNITUDE_SIZE:
            return [0] * self.MAGNITUDE_SIZE, not self._is_zero_magnitude(magnitude)

        overflow = any(bit == 1 for bit in magnitude[:shift])
        shifted = magnitude[shift:] + ([0] * shift)
        return shifted, overflow

    def _compare_magnitudes(self, left: list[int], right: list[int]) -> int:
        for left_bit, right_bit in zip(left, right):
            if left_bit > right_bit:
                return 1
            if left_bit < right_bit:
                return -1
        return 0

    def _shift_left_one(self, magnitude: list[int]) -> list[int]:
        return magnitude[1:] + [0]

    def _divide_magnitudes(self, dividend: list[int], divisor: list[int]) -> tuple[list[int], list[int]]:
        quotient = [0] * self.MAGNITUDE_SIZE
        remainder = [0] * self.MAGNITUDE_SIZE

        for idx, dividend_bit in enumerate(dividend):
            remainder = self._shift_left_one(remainder)
            remainder[-1] = dividend_bit

            if self._compare_magnitudes(remainder, divisor) >= 0:
                remainder = self._subtract_magnitudes(remainder, divisor)
                quotient[idx] = 1

        return quotient, remainder

    def _divide_fractional_bits(self, remainder: list[int], divisor: list[int], precision: int) -> str:
        bits: list[str] = []
        current_remainder = remainder[:]

        for _ in range(precision):
            current_remainder = self._shift_left_one(current_remainder)
            if self._compare_magnitudes(current_remainder, divisor) >= 0:
                current_remainder = self._subtract_magnitudes(current_remainder, divisor)
                bits.append("1")
            else:
                bits.append("0")

        return "".join(bits)

    def _magnitude_to_unsigned(self, magnitude: list[int]) -> int:
        value = 0
        for bit in magnitude:
            value = (value * 2) + bit
        return value

    def _magnitude_to_binary_string(self, magnitude: list[int]) -> str:
        binary = "".join(str(bit) for bit in magnitude).lstrip("0")
        return binary or "0"

    def _format_binary_quotient(self, quotient_magnitude: list[int], fractional_bits: str, sign: int) -> str:
        prefix = "-" if sign == 1 and (not self._is_zero_magnitude(quotient_magnitude) or "1" in fractional_bits) else ""
        integer_part = self._magnitude_to_binary_string(quotient_magnitude)

        if not fractional_bits:
            return f"{prefix}{integer_part}"

        return f"{prefix}{integer_part}.{fractional_bits}"

    def _decimal_division(self, dividend: list[int], divisor: list[int], sign: int, precision: int) -> float:
        dividend_value = self._magnitude_to_unsigned(dividend)
        divisor_value = self._magnitude_to_unsigned(divisor)

        integer_part = dividend_value // divisor_value
        remainder = dividend_value % divisor_value
        digits: list[int] = []

        for _ in range(precision + 1):
            remainder *= 10
            digits.append(remainder // divisor_value)
            remainder %= divisor_value

        if precision == 0:
            if digits and digits[0] >= 5:
                integer_part += 1
            result_text = str(integer_part)
        else:
            fractional_digits = digits[:precision]
            rounding_digit = digits[precision]

            if rounding_digit >= 5:
                carry = 1
                for idx in range(precision - 1, -1, -1):
                    if carry == 0:
                        break
                    updated = fractional_digits[idx] + carry
                    if updated < 10:
                        fractional_digits[idx] = updated
                        carry = 0
                    else:
                        fractional_digits[idx] = 0

                if carry == 1:
                    integer_part += 1

            fractional_text = "".join(str(digit) for digit in fractional_digits)
            result_text = f"{integer_part}.{fractional_text}"

        if sign == 1 and result_text.strip("0.") != "":
            result_text = f"-{result_text}"

        return float(result_text)
