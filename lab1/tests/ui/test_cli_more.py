import pytest

from src.ui.cli import CommandLineInterface


def test_cli_menu_and_read_int(monkeypatch) -> None:
    cli = CommandLineInterface()

    monkeypatch.setattr("builtins.input", lambda _prompt="": "42")

    assert "Bit-level calculator" in cli._menu()
    assert cli._read_int("Number: ") == 42


@pytest.mark.parametrize(
    ("choice", "values", "expected_text"),
    [
        ("2", [5, -3], "Decimal: 2"),
        ("3", [5, 7], "Decimal: -2"),
        ("4", [-6, 5], "Decimal: -30"),
        ("5", [13, 2], "Binary: 110.10000"),
        ("7", [25, 37], "Decimal: 62"),
    ],
)
def test_cli_handles_numeric_menu_choices(choice, values, expected_text, monkeypatch, capsys) -> None:
    cli = CommandLineInterface()
    answers = iter(values)

    monkeypatch.setattr(cli, "_read_int", lambda _prompt="": next(answers))

    cli._handle_choice(choice)

    assert expected_text in capsys.readouterr().out


def test_cli_handles_float_menu_choice(monkeypatch, capsys) -> None:
    cli = CommandLineInterface()
    answers = iter(["+", "1.5", "2.25"])

    monkeypatch.setattr("builtins.input", lambda _prompt="": next(answers))

    cli._handle_choice("6")

    assert "Decimal: 3.75" in capsys.readouterr().out


@pytest.mark.parametrize("operation", ["+", "-", "*", "/"])
def test_cli_float_result_dispatches_supported_operations(operation) -> None:
    cli = CommandLineInterface()

    result = cli._float_result(operation, "6.0", "2.0")

    assert "result_decimal" in result


def test_cli_rejects_invalid_operation_and_unknown_choice() -> None:
    cli = CommandLineInterface()

    with pytest.raises(ValueError):
        cli._float_result("%", "1", "1")

    with pytest.raises(ValueError):
        cli._handle_choice("8")


def test_cli_run_reports_errors_and_allows_exit(monkeypatch, capsys) -> None:
    cli = CommandLineInterface()
    answers = iter(["8", "0"])

    monkeypatch.setattr("builtins.input", lambda _prompt="": next(answers))

    cli.run()

    output = capsys.readouterr().out
    assert "Error: Unknown menu item." in output
