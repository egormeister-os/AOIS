from __future__ import annotations

from boollab.cli import build_report
from main import main


def test_build_report_contains_expected_sections() -> None:
    report = build_report("a|b")

    assert "Truth table:" in report
    assert "SDNF: (!a&b) | (a&!b) | (a&b)" in report
    assert "Post classes: T0=yes, T1=yes, S=no, M=yes, L=no" in report
    assert "D_ab: 1" in report
    assert "Calculation method (DNF):" in report
    assert "Calculation method (KNF):" in report
    assert "Calculation-tabular method (DNF):" in report
    assert "Calculation-tabular method (KNF):" in report
    assert "Karnaugh map (DNF):" in report
    assert "Karnaugh map (KNF):" in report
    assert "Calculation-tabular method (DNF):\nStage 1" in report
    assert "K1: a; cells:" in report
    assert "Result: a | b" in report
    assert "Result: (a|b)" in report


def test_main_uses_arguments_and_returns_zero(capsys) -> None:
    exit_code = main(["a|b"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Expression: a|b" in captured.out
    assert captured.err == ""


def test_main_reads_input_when_arguments_are_missing(monkeypatch, capsys) -> None:
    monkeypatch.setattr("builtins.input", lambda prompt: "a&b")

    exit_code = main([])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Result: a | b" not in captured.out
    assert "Result: a&b" not in captured.out
    assert "Result: (a&b)" in captured.out


def test_main_returns_error_for_invalid_expression(capsys) -> None:
    exit_code = main(["a+"])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "Ошибка:" in captured.err
