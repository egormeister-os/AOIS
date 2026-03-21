from fractions import Fraction

from src.services.lab_service import LabService


def test_service_covers_integer_operations() -> None:
    service = LabService()

    assert service.add_twos_complement(7, -2)["result_decimal"] == 5
    assert service.subtract_twos_complement(7, 2)["result_decimal"] == 5
    assert service.multiply_sign_magnitude(-3, 4)["result_decimal"] == -12


def test_service_covers_float_helpers_and_remaining_operations() -> None:
    service = LabService()

    encoded = service.encode_float32(Fraction(3, 2))
    assert service.decode_float32(encoded) == 1.5
    assert service.subtract_float32("7.5", "2.25")["result_decimal"] == 5.25
    assert service.multiply_float32("-1.5", "2.0")["result_decimal"] == -3.0
    assert service.divide_float32("7.0", "2.0")["result_decimal"] == 3.5
