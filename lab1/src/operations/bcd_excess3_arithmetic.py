class Excess3BCDArithmetic:
    """Variant D: Excess-3 BCD operations."""

    def encode_number(self, value: int) -> list[list[int]]:
        if value < 0:
            raise ValueError("Excess-3 BCD in this lab expects non-negative integers.")

        return [self._encode_digit(digit) for digit in self._split_digits(value)]

    def decode_number(self, encoded_digits: list[list[int]]) -> int:
        value = 0

        for digit_bits in encoded_digits:
            digit = self._tetrad_to_int(digit_bits) - 3
            if digit < 0 or digit > 9:
                raise ValueError("Invalid Excess-3 digit.")
            value = (value * 10) + digit

        return value

    def add(self, left: int, right: int) -> tuple[list[list[int]], int]:
        if left < 0 or right < 0:
            raise ValueError("Excess-3 addition in this lab expects non-negative integers.")

        left_digits = self.encode_number(left)
        right_digits = self.encode_number(right)
        max_length = max(len(left_digits), len(right_digits))

        left_digits = self._pad_left(left_digits, max_length)
        right_digits = self._pad_left(right_digits, max_length)

        carry = 0
        result: list[list[int]] = []

        for idx in range(max_length - 1, -1, -1):
            raw_sum, carry_out = self._add_tetrads(left_digits[idx], right_digits[idx], carry)
            correction = [0, 0, 1, 1] if carry_out == 1 else [1, 1, 0, 1]
            corrected, _ = self._add_tetrads(raw_sum, correction, 0)
            result.insert(0, corrected)
            carry = carry_out

        if carry == 1:
            result.insert(0, self._encode_digit(1))

        return result, self.decode_number(result)

    def _split_digits(self, value: int) -> list[int]:
        if value == 0:
            return [0]

        digits: list[int] = []
        current = value

        while current > 0:
            digits.append(current % 10)
            current //= 10

        return list(reversed(digits))

    def _encode_digit(self, digit: int) -> list[int]:
        if digit < 0 or digit > 9:
            raise ValueError("Excess-3 digit must be in range 0..9.")
        return self._int_to_tetrad(digit + 3)

    def _int_to_tetrad(self, value: int) -> list[int]:
        bits = [0, 0, 0, 0]

        for idx in range(3, -1, -1):
            bits[idx] = value % 2
            value //= 2

        return bits

    def _tetrad_to_int(self, bits: list[int]) -> int:
        value = 0
        for bit in bits:
            value = (value * 2) + bit
        return value

    def _add_tetrads(self, left: list[int], right: list[int], carry_in: int) -> tuple[list[int], int]:
        result = [0, 0, 0, 0]
        carry = carry_in

        for idx in range(3, -1, -1):
            bit_sum = left[idx] + right[idx] + carry
            result[idx] = bit_sum % 2
            carry = bit_sum // 2

        return result, carry

    def _pad_left(self, digits: list[list[int]], width: int) -> list[list[int]]:
        padding = [self._encode_digit(0) for _ in range(width - len(digits))]
        return padding + digits
