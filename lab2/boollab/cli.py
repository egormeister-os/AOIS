from __future__ import annotations

from .core import BooleanFunction, format_implicant_pattern
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
            + ", ".join(implicant.term(variables) for implicant in stage.input_terms)
        )
        for pair in stage.combinations:
            lines.append(
                f"  {pair.left.term(variables)} + {pair.right.term(variables)} -> "
                f"{pair.result.term(variables)}"
            )
        lines.append(
            "Result: "
            + ", ".join(implicant.term(variables) for implicant in stage.output_terms)
        )
    return "\n".join(lines)


def _format_prime_chart(result, variables: tuple[str, ...]) -> str:
    if not result.chart_rows:
        return "Prime implicant chart is empty."
    header = ["Implicant"] + [str(column) for column in result.chart_columns]
    lines = [" | ".join(header)]
    for row in result.chart_rows:
        coverage = ["X" if covered else "." for covered in row.coverage]
        lines.append(" | ".join([row.implicant.term(variables), *coverage]))
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
            term = format_implicant_pattern(group.implicant.pattern, variables)
            lines.append(f"  K{index}: {term}; cells: {cells}")
    else:
        lines.append("Groups: none")
    lines.append(f"Result: {kmap.expression}")
    return "\n".join(lines)


def build_report(expression: str) -> str:
    function = BooleanFunction.from_expression(expression)
    minimization = minimize_function(function)
    kmap = build_karnaugh_map(function, minimization)
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
        "Calculation method:",
        _format_gluing(minimization, function.variables),
        "Prime implicants: "
        + ", ".join(implicant.term(function.variables) for implicant in minimization.prime_implicants),
        "Removed as redundant: "
        + (
            ", ".join(
                implicant.term(function.variables)
                for implicant in minimization.redundant_implicants
            )
            if minimization.redundant_implicants
            else "none"
        ),
        f"Result: {minimization.expression}",
        "",
        "Calculation-tabular method:",
        _format_gluing(minimization, function.variables),
        _format_prime_chart(minimization, function.variables),
        "Removed as redundant: "
        + (
            ", ".join(
                implicant.term(function.variables)
                for implicant in minimization.redundant_implicants
            )
            if minimization.redundant_implicants
            else "none"
        ),
        f"Result: {minimization.expression}",
        "",
        "Karnaugh map:",
        _format_karnaugh_map(kmap, function.variables),
    ]
    return "\n".join(sections)
