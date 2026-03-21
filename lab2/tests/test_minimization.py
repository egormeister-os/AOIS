from __future__ import annotations

from boollab.core import BooleanFunction, index_to_bits
from boollab.minimization import Implicant, _exact_cover, build_karnaugh_map, minimize_function


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
    zero_kmap = build_karnaugh_map(zero, zero_result)

    assert zero_result.expression == "0"
    assert zero_result.chart_rows == ()
    assert zero_kmap.groups == ()

    one = BooleanFunction.from_expression("1")
    one_result = minimize_function(one)
    one_kmap = build_karnaugh_map(one, one_result)

    assert one_result.expression == "1"
    assert len(one_kmap.groups) == 1
    assert one_kmap.groups[0].cells == (("base", "-", "-"),)


def test_five_variable_karnaugh_map_uses_two_layers() -> None:
    variables = ("a", "b", "c", "d", "e")
    values = [index_to_bits(index, 5)[-1] for index in range(32)]
    function = BooleanFunction.from_truth_vector(variables, values, source_expression="e")

    result = minimize_function(function)
    kmap = build_karnaugh_map(function, result)

    assert result.expression == "e"
    assert len(kmap.layers) == 2
    assert kmap.layers[0].row_variables == ("a", "b")
    assert kmap.layers[0].column_variables == ("c", "d")
    assert {layer.label for layer in kmap.layers} == {"e=0", "e=1"}
