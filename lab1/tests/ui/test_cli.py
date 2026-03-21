from src.ui.cli import CommandLineInterface


def test_cli_can_show_integer_conversion(monkeypatch, capsys) -> None:
    cli = CommandLineInterface()
    answers = iter(["1", "-5", "0"])

    monkeypatch.setattr("builtins.input", lambda _prompt="": next(answers))

    cli.run()

    output = capsys.readouterr().out
    assert "Sign-magnitude" in output
    assert "Two's complement" in output
