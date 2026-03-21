from src.services.lab_service import LabService


def test_service_returns_all_integer_representations() -> None:
    service = LabService()

    result = service.convert_integer_representations(-5)

    assert result["decoded_sign_magnitude"] == -5
    assert result["decoded_ones_complement"] == -5
    assert result["decoded_twos_complement"] == -5


def test_service_divides_sign_magnitude_numbers() -> None:
    service = LabService()

    result = service.divide_sign_magnitude(13, 2)

    assert result["result_binary"] == "110.10000"
    assert result["result_decimal"] == 6.5


def test_service_adds_float32_numbers() -> None:
    service = LabService()

    result = service.add_float32("1.5", "2.25")

    assert str(result["result_bits"]) == "01000000011100000000000000000000"
    assert result["result_decimal"] == 3.75


def test_service_adds_excess3_numbers() -> None:
    service = LabService()

    result = service.add_excess3(25, 37)

    assert result["result_bits"] == [[1, 0, 0, 1], [0, 1, 0, 1]]
    assert result["result_decimal"] == 62
