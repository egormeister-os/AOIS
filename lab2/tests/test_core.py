from __future__ import annotations

import pytest

from boollab.core import (
    BooleanFunction,
    ExpressionError,
    bits_to_index,
    gray_code,
    index_to_bits,
)


def test_parser_supports_unicode_and_builds_truth_table() -> None:
    function = BooleanFunction.from_expression("!(!a→!b)∨c")

    assert function.variables == ("a", "b", "c")
    assert function.truth_vector == [0, 1, 1, 1, 0, 1, 0, 1]
    assert function.evaluate({"a": 0, "b": 0, "c": 1}) == 1
    assert function.evaluate({"a": 1, "b": 1, "c": 0}) == 0


@pytest.mark.parametrize(
    ("expression", "assignment", "expected"),
    [
        ("a->b|c", {"a": 1, "b": 0, "c": 1}, 1),
        ("a->b|c", {"a": 1, "b": 0, "c": 0}, 0),
        ("a~b", {"a": 1, "b": 1}, 1),
        ("a~b", {"a": 1, "b": 0}, 0),
    ],
)
def test_operator_precedence_and_equivalence(
    expression: str, assignment: dict[str, int], expected: int
) -> None:
    assert BooleanFunction.from_expression(expression).evaluate(assignment) == expected


@pytest.mark.parametrize(
    "expression",
    ["", "(a|b", "a+b"],
)
def test_invalid_expressions_raise(expression: str) -> None:
    with pytest.raises(ExpressionError):
        BooleanFunction.from_expression(expression)


def test_missing_operator_reports_clear_error() -> None:
    with pytest.raises(ExpressionError, match="Missing operator before '!' at position 8"):
        BooleanFunction.from_expression("((a&b&!c!d)|!e)|((a&b&!c&!d)|!e)")


def test_forms_post_classes_and_fictive_variables() -> None:
    function = BooleanFunction.from_expression("a|b")

    assert function.sdnf() == "(!a&b) | (a&!b) | (a&b)"
    assert function.sknf() == "(a|b)"
    assert function.numeric_forms() == {"sdnf": [1, 2, 3], "sknf": [0]}
    assert function.index_form() == {"binary": "0111", "decimal": 7}
    assert set(function.zhegalkin_polynomial().split(" ^ ")) == {"a", "b", "ab"}
    assert function.post_classes() == {
        "T0": True,
        "T1": True,
        "S": False,
        "M": True,
        "L": False,
    }

    fictive = BooleanFunction.from_expression("a|(b&!b)")
    assert fictive.fictive_variables() == ["b"]

    self_dual = BooleanFunction.from_expression("a")
    assert self_dual.post_classes()["S"] is True


def test_derivatives_and_truth_vector_building() -> None:
    function = BooleanFunction.from_expression("a|b")

    derivative_a = function.derivative(("a",))
    derivative_b = function.derivative(("b",))
    derivative_ab = function.derivative(("a", "b"))

    assert derivative_a.truth_vector == [1, 0, 1, 0]
    assert derivative_b.truth_vector == [1, 1, 0, 0]
    assert derivative_ab.truth_vector == [1, 1, 1, 1]
    assert set(function.all_derivatives(max_order=2)) == {
        ("a",),
        ("b",),
        ("a", "b"),
    }

    with pytest.raises(ValueError):
        BooleanFunction.from_truth_vector(("a", "b"), [0, 1, 1])


def test_helpers_cover_index_conversion_and_gray_code() -> None:
    bits = (1, 0, 1, 1)
    index = bits_to_index(bits)

    assert index == 11
    assert index_to_bits(index, 4) == bits
    assert gray_code(0) == [()]
    assert gray_code(2) == [(0, 0), (0, 1), (1, 1), (1, 0)]
