from src.services.lab_service import LabService
from src.ui.formatter import OutputFormatter


class CommandLineInterface:
    """CLI controller; later can be replaced by GUI without touching domain layer."""

    def __init__(self) -> None:
        self.service = LabService()
        self.formatter = OutputFormatter()

    def run(self) -> None:
        while True:
            print(self._menu())
            choice = input("Choose action: ").strip()

            if choice == "0":
                return

            try:
                self._handle_choice(choice)
            except Exception as exc:
                print(f"Error: {exc}\n")

    def _menu(self) -> str:
        return (
            "\nBit-level calculator\n"
            "1. Convert decimal integer to sign/ones/twos complement\n"
            "2. Add integers in two's complement\n"
            "3. Subtract integers in two's complement\n"
            "4. Multiply integers in sign-magnitude\n"
            "5. Divide integers in sign-magnitude\n"
            "6. IEEE-754 float32 operation\n"
            "7. Add numbers in Excess-3 BCD\n"
            "0. Exit\n"
        )

    def _handle_choice(self, choice: str) -> None:
        if choice == "1":
            value = self._read_int("Decimal integer: ")
            result = self.service.convert_integer_representations(value)
            print(self.formatter.conversion_report(result))
            print()
            return

        if choice == "2":
            left = self._read_int("Left operand: ")
            right = self._read_int("Right operand: ")
            result = self.service.add_twos_complement(left, right)
            print(self.formatter.bits_and_decimal(result["result_bits"], result["result_decimal"]))
            print()
            return

        if choice == "3":
            left = self._read_int("Minuend: ")
            right = self._read_int("Subtrahend: ")
            result = self.service.subtract_twos_complement(left, right)
            print(self.formatter.bits_and_decimal(result["result_bits"], result["result_decimal"]))
            print()
            return

        if choice == "4":
            left = self._read_int("Left operand: ")
            right = self._read_int("Right operand: ")
            result = self.service.multiply_sign_magnitude(left, right)
            print(self.formatter.bits_and_decimal(result["result_bits"], result["result_decimal"]))
            print()
            return

        if choice == "5":
            left = self._read_int("Dividend: ")
            right = self._read_int("Divisor: ")
            result = self.service.divide_sign_magnitude(left, right)
            print(self.formatter.binary_string_and_decimal(result["result_binary"], result["result_decimal"]))
            print()
            return

        if choice == "6":
            operation = input("Operation (+, -, *, /): ").strip()
            left = input("Left operand: ").strip()
            right = input("Right operand: ").strip()
            result = self._float_result(operation, left, right)
            print(self.formatter.bits_and_decimal(result["result_bits"], result["result_decimal"]))
            print()
            return

        if choice == "7":
            left = self._read_int("Left operand: ")
            right = self._read_int("Right operand: ")
            result = self.service.add_excess3(left, right)
            print(f"A: {self.formatter.excess3_digits(result['left_bits'])}")
            print(f"B: {self.formatter.excess3_digits(result['right_bits'])}")
            print(
                self.formatter.binary_string_and_decimal(
                    self.formatter.excess3_digits(result["result_bits"]),
                    result["result_decimal"],
                )
            )
            print()
            return

        raise ValueError("Unknown menu item.")

    def _float_result(self, operation: str, left: str, right: str) -> dict[str, object]:
        operations = {
            "+": self.service.add_float32,
            "-": self.service.subtract_float32,
            "*": self.service.multiply_float32,
            "/": self.service.divide_float32,
        }

        if operation not in operations:
            raise ValueError("Unsupported IEEE-754 operation.")

        return operations[operation](left, right)

    def _read_int(self, prompt: str) -> int:
        return int(input(prompt).strip())
