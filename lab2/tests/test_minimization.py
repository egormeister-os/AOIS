from __future__ import annotations

from boollab.core import BooleanFunction, index_to_bits
from boollab.minimization import (
    Implicant,
    MinimizationResult,
    _exact_cover,
    build_karnaugh_map,
    minimize_function,
)


def test_pdf_example_minimizes_to_a_or_bc() -> None:
    function = BooleanFunction.from_truth_vector(
        ("a", "b", "c"),
        [0, 0, 0, 1, 1, 1, 1, 1],
        source_expression="pdf_example",
    )

    result = minimize_function(function)

    assert result.expression == "a | (b&c)"
    assert len(result.gluing_stages) == 2
    assert {implicant.pattern for implicant in result.prime_implicants} == {"1--", "-11"}
    assert [row.coverage for row in result.chart_rows] == [
        (False, True, True, True, True),
        (True, False, False, False, True),
    ]

    kmap = build_karnaugh_map(function)

    assert kmap.expression == "a | (b&c)"
    assert {group.implicant.pattern for group in kmap.groups} == {"1--", "-11"}


def test_minimization_supports_dnf_and_knf() -> None:
    function = BooleanFunction.from_truth_vector(
        ("a", "b"),
        [0, 0, 1, 1],
        source_expression="a_with_helper_variable",
    )

    dnf_result = minimize_function(function, form="dnf")
    knf_result = minimize_function(function, form="knf")

    assert dnf_result.expression == "a"
    assert len(dnf_result.gluing_stages) == 1
    assert {implicant.pattern for implicant in dnf_result.selected_implicants} == {"1-"}
    assert dnf_result.chart_columns == (2, 3)

    assert knf_result.expression == "a"
    assert len(knf_result.gluing_stages) == 1
    assert {implicant.pattern for implicant in knf_result.selected_implicants} == {"0-"}
    assert knf_result.chart_columns == (0, 1)

    dnf_kmap = build_karnaugh_map(function, form="dnf")
    knf_kmap = build_karnaugh_map(function, form="knf")

    assert dnf_kmap.expression == "a"
    assert {group.implicant.pattern for group in dnf_kmap.groups} == {"1-"}
    assert knf_kmap.expression == "a"
    assert {group.implicant.pattern for group in knf_kmap.groups} == {"0-"}


def test_exact_cover_search_handles_non_essential_case() -> None:
    solution = _exact_cover(
        [1, 2, 3],
        [
            Implicant("1-", frozenset({1, 2})),
            Implicant("-1", frozenset({2, 3})),
            Implicant("0-", frozenset({1, 3})),
        ],
    )

    assert {implicant.pattern for implicant in solution} in (
        {"1-", "-1"},
        {"1-", "0-"},
        {"-1", "0-"},
    )
    assert len(solution) == 2


def test_constant_functions_and_karnaugh_layouts() -> None:
    zero = BooleanFunction.from_expression("0")
    zero_result = minimize_function(zero)
    zero_kmap = build_karnaugh_map(zero)
    zero_knf_result = minimize_function(zero, form="knf")
    zero_knf_kmap = build_karnaugh_map(zero, form="knf")

    assert zero_result.expression == "0"
    assert zero_result.chart_rows == ()
    assert zero_kmap.groups == ()
    assert zero_knf_result.expression == "0"
    assert len(zero_knf_kmap.groups) == 1

    one = BooleanFunction.from_expression("1")
    one_result = minimize_function(one)
    one_kmap = build_karnaugh_map(one)
    one_knf_result = minimize_function(one, form="knf")
    one_knf_kmap = build_karnaugh_map(one, form="knf")

    assert one_result.expression == "1"
    assert len(one_kmap.groups) == 1
    assert one_kmap.groups[0].cells == (("base", "-", "-"),)
    assert one_knf_result.expression == "1"
    assert one_knf_kmap.groups == ()


def test_five_variable_karnaugh_map_uses_two_layers() -> None:
    variables = ("a", "b", "c", "d", "e")
    values = [index_to_bits(index, 5)[-1] for index in range(32)]
    function = BooleanFunction.from_truth_vector(variables, values, source_expression="e")

    result = minimize_function(function)
    kmap = build_karnaugh_map(function)

    assert result.expression == "e"
    assert len(kmap.layers) == 2
    assert kmap.layers[0].row_variables == ("a", "b")
    assert kmap.layers[0].column_variables == ("c", "d")
    assert {layer.label for layer in kmap.layers} == {"e=0", "e=1"}


def test_karnaugh_map_does_not_depend_on_calculation_result() -> None:
    function = BooleanFunction.from_expression("a|b")
    bogus_result = MinimizationResult((), (), (), (), (), (), "0")

    kmap = build_karnaugh_map(function, bogus_result)

    assert kmap.expression == "a | b"
    assert {group.implicant.pattern for group in kmap.groups} == {"1-", "-1"}
