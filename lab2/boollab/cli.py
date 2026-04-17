from __future__ import annotations

from .core import BooleanFunction
from .minimization import build_karnaugh_map, minimize_function


def _format_table(function: BooleanFunction) -> str:
    headers = list(function.variables) + ["f"]
    lines = [" | ".join(headers)]
    for row in function.truth_table:
        lines.append(" | ".join(str(bit) for bit in row.bits + (row.value,)))
    return "\n".join(lines)


def _format_post_classes(function: BooleanFunction) -> str:
    post = function.post_classes()
    return ", ".join(
        f"{name}={'yes' if belongs else 'no'}" for name, belongs in post.items()
    )


def _format_derivatives(function: BooleanFunction) -> str:
    lines = []
    for variables, derivative in function.all_derivatives().items():
        formula = minimize_function(derivative).expression
        title = "".join(variables)
        lines.append(f"D_{title}: {formula}")
    return "\n".join(lines) if lines else "No derivatives for constants."


def _format_gluing(result, variables: tuple[str, ...]) -> str:
    if not result.gluing_stages:
        return "No gluing stages."
    lines = []
    for stage in result.gluing_stages:
        lines.append(f"Stage {stage.number}")
        lines.append(
            "Input: "
            + ", ".join(implicant.term(variables, result.form) for implicant in stage.input_terms)
        )
        for pair in stage.combinations:
            lines.append(
                f"  {pair.left.term(variables, result.form)} + "
                f"{pair.right.term(variables, result.form)} -> "
                f"{pair.result.term(variables, result.form)}"
            )
        lines.append(
            "Result: "
            + ", ".join(implicant.term(variables, result.form) for implicant in stage.output_terms)
        )
    return "\n".join(lines)


def _format_prime_chart(result, variables: tuple[str, ...]) -> str:
    if not result.chart_rows:
        return "Prime implicant chart is empty."
    label = "Implicant" if result.form == "dnf" else "Clause"
    header = [label] + [str(column) for column in result.chart_columns]
    lines = [" | ".join(header)]
    for row in result.chart_rows:
        coverage = ["X" if covered else "." for covered in row.coverage]
        lines.append(" | ".join([row.implicant.term(variables, result.form), *coverage]))
    return "\n".join(lines)


def _format_karnaugh_map(kmap, variables: tuple[str, ...]) -> str:
    lines = []
    for layer in kmap.layers:
        row_header = "".join(layer.row_variables) or "-"
        column_header = "".join(layer.column_variables) or "-"
        lines.append(f"Layer: {layer.label}")
        lines.append(f"{row_header}\\{column_header}: " + " ".join(layer.column_labels))
        for row_label, row in zip(layer.row_labels, layer.grid):
            lines.append(f"{row_label}: " + " ".join(str(value) for value in row))
    if kmap.groups:
        lines.append("Groups:")
        for index, group in enumerate(kmap.groups, start=1):
            cells = ", ".join(f"{layer}/{row}/{column}" for layer, row, column in group.cells)
            term = group.implicant.term(variables, kmap.form)
            lines.append(f"  K{index}: {term}; cells: {cells}")
    else:
        lines.append("Groups: none")
    lines.append(f"Result: {kmap.expression}")
    return "\n".join(lines)


def _format_terms(implicants, variables: tuple[str, ...], form: str) -> str:
    if not implicants:
        return "none"
    return ", ".join(implicant.term(variables, form) for implicant in implicants)


def _prime_terms_label(form: str) -> str:
    return "Prime implicants" if form == "dnf" else "Prime clauses"


def _redundant_terms_label(form: str) -> str:
    return (
        "Removed as redundant implicants"
        if form == "dnf"
        else "Removed as redundant clauses"
    )


def build_report(expression: str) -> str:
    function = BooleanFunction.from_expression(expression)
    dnf_minimization = minimize_function(function, form="dnf")
    knf_minimization = minimize_function(function, form="knf")
    dnf_kmap = build_karnaugh_map(function, form="dnf")
    knf_kmap = build_karnaugh_map(function, form="knf")
    numeric = function.numeric_forms()
    index_form = function.index_form()
    fictive = function.fictive_variables()
    sections = [
        f"Expression: {function.source_expression}",
        f"Variables: {', '.join(function.variables) if function.variables else '(none)'}",
        "",
        "Truth table:",
        _format_table(function),
        "",
        f"SDNF: {function.sdnf()}",
        f"SKNF: {function.sknf()}",
        f"Numeric SDNF: S({', '.join(map(str, numeric['sdnf']))})",
        f"Numeric SKNF: P({', '.join(map(str, numeric['sknf']))})",
        f"Index form: {index_form['binary']} = {index_form['decimal']}",
        "",
        f"Post classes: {_format_post_classes(function)}",
        f"Zhegalkin polynomial: {function.zhegalkin_polynomial()}",
        f"Fictive variables: {', '.join(fictive) if fictive else 'none'}",
        "",
        "Boolean derivatives:",
        _format_derivatives(function),
        "",
        "Calculation method (DNF):",
        _format_gluing(dnf_minimization, function.variables),
        f"{_prime_terms_label('dnf')}: "
        + _format_terms(dnf_minimization.prime_implicants, function.variables, "dnf"),
        f"{_redundant_terms_label('dnf')}: "
        + _format_terms(dnf_minimization.redundant_implicants, function.variables, "dnf"),
        f"Result: {dnf_minimization.expression}",
        "",
        "Calculation method (KNF):",
        _format_gluing(knf_minimization, function.variables),
        f"{_prime_terms_label('knf')}: "
        + _format_terms(knf_minimization.prime_implicants, function.variables, "knf"),
        f"{_redundant_terms_label('knf')}: "
        + _format_terms(knf_minimization.redundant_implicants, function.variables, "knf"),
        f"Result: {knf_minimization.expression}",
        "",
        "Calculation-tabular method (DNF):",
        _format_gluing(dnf_minimization, function.variables),
        _format_prime_chart(dnf_minimization, function.variables),
        f"{_redundant_terms_label('dnf')}: "
        + _format_terms(dnf_minimization.redundant_implicants, function.variables, "dnf"),
        f"Result: {dnf_minimization.expression}",
        "",
        "Calculation-tabular method (KNF):",
        _format_gluing(knf_minimization, function.variables),
        _format_prime_chart(knf_minimization, function.variables),
        f"{_redundant_terms_label('knf')}: "
        + _format_terms(knf_minimization.redundant_implicants, function.variables, "knf"),
        f"Result: {knf_minimization.expression}",
        "",
        "Karnaugh map (DNF):",
        _format_karnaugh_map(dnf_kmap, function.variables),
        "",
        "Karnaugh map (KNF):",
        _format_karnaugh_map(knf_kmap, function.variables),
    ]
    return "\n".join(sections)
