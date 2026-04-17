from __future__ import annotations

from dataclasses import dataclass
from itertools import product

from .core import (
    BooleanFunction,
    bits_to_index,
    format_implicant_pattern,
    gray_code,
)


@dataclass(frozen=True)
class Implicant:
    pattern: str
    minterms: frozenset[int]

    @property
    def literal_count(self) -> int:
        return sum(1 for bit in self.pattern if bit != "-")

    def covers(self, minterm: int) -> bool:
        return minterm in self.minterms

    def combine(self, other: "Implicant") -> "Implicant | None":
        differences = 0
        pattern = []
        for left, right in zip(self.pattern, other.pattern):
            if left == right:
                pattern.append(left)
                continue
            if left == "-" or right == "-":
                return None
            differences += 1
            pattern.append("-")
            if differences > 1:
                return None
        if differences != 1:
            return None
        return Implicant("".join(pattern), self.minterms | other.minterms)

    def term(self, variables: tuple[str, ...], form: str = "dnf") -> str:
        if form == "knf":
            return _format_clause_pattern(self.pattern, variables)
        return format_implicant_pattern(self.pattern, variables)


@dataclass(frozen=True)
class GluingPair:
    left: Implicant
    right: Implicant
    result: Implicant


@dataclass(frozen=True)
class GluingStage:
    number: int
    input_terms: tuple[Implicant, ...]
    combinations: tuple[GluingPair, ...]
    output_terms: tuple[Implicant, ...]


@dataclass(frozen=True)
class PrimeChartRow:
    implicant: Implicant
    coverage: tuple[bool, ...]


@dataclass(frozen=True)
class MinimizationResult:
    gluing_stages: tuple[GluingStage, ...]
    prime_implicants: tuple[Implicant, ...]
    selected_implicants: tuple[Implicant, ...]
    redundant_implicants: tuple[Implicant, ...]
    chart_columns: tuple[int, ...]
    chart_rows: tuple[PrimeChartRow, ...]
    expression: str
    form: str = "dnf"


@dataclass(frozen=True)
class KarnaughLayer:
    label: str
    row_variables: tuple[str, ...]
    column_variables: tuple[str, ...]
    row_labels: tuple[str, ...]
    column_labels: tuple[str, ...]
    grid: tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class KarnaughGroup:
    implicant: Implicant
    cells: tuple[tuple[str, str, str], ...]


@dataclass(frozen=True)
class KarnaughMap:
    layers: tuple[KarnaughLayer, ...]
    groups: tuple[KarnaughGroup, ...]
    expression: str
    form: str = "dnf"


def _normalize_form(form: str) -> str:
    normalized = form.lower()
    if normalized not in {"dnf", "knf"}:
        raise ValueError("form must be either 'dnf' or 'knf'")
    return normalized


def _target_value(form: str) -> int:
    return 1 if form == "dnf" else 0


def _target_indexes(function: BooleanFunction, form: str) -> list[int]:
    return function.minterm_indexes() if form == "dnf" else function.maxterm_indexes()


def _empty_expression(form: str) -> str:
    return "0" if form == "dnf" else "1"


def _universal_expression(form: str) -> str:
    return "1" if form == "dnf" else "0"


def _join_symbol(form: str) -> str:
    return " | " if form == "dnf" else " & "


def _format_clause_pattern(pattern: str, variables: tuple[str, ...]) -> str:
    literals = []
    for bit, variable in zip(pattern, variables):
        if bit == "-":
            continue
        literals.append(variable if bit == "0" else f"!{variable}")
    if not literals:
        return "0"
    if len(literals) == 1:
        return literals[0]
    return f"({'|'.join(literals)})"


def _sort_key(implicant: Implicant) -> tuple[int, str]:
    return (implicant.literal_count, implicant.pattern.replace("-", "2"))


def _merge_implicants(existing: Implicant | None, new: Implicant) -> Implicant:
    if existing is None:
        return new
    return Implicant(existing.pattern, existing.minterms | new.minterms)


def _expression_from_implicants(
    implicants: tuple[Implicant, ...], variables: tuple[str, ...], form: str
) -> str:
    if not implicants:
        return _empty_expression(form)
    if len(implicants) == 1 and implicants[0].pattern == "-" * len(variables):
        return _universal_expression(form)
    return _join_symbol(form).join(implicant.term(variables, form) for implicant in implicants)


def _exact_cover(minterms: list[int], prime_implicants: list[Implicant]) -> tuple[Implicant, ...]:
    if not minterms:
        return ()
    selected: list[Implicant] = []
    remaining_minterms = set(minterms)
    remaining_primes = list(prime_implicants)

    changed = True
    while changed:
        changed = False
        for minterm in list(remaining_minterms):
            covering = [implicant for implicant in remaining_primes if implicant.covers(minterm)]
            if len(covering) == 1:
                implicant = covering[0]
                if implicant not in selected:
                    selected.append(implicant)
                remaining_minterms -= implicant.minterms
                remaining_primes = [item for item in remaining_primes if item != implicant]
                changed = True

    if not remaining_minterms:
        return tuple(sorted(selected, key=_sort_key))

    best_solution: tuple[Implicant, ...] | None = None

    def solution_score(solution: tuple[Implicant, ...]) -> tuple[int, int, tuple[str, ...]]:
        return (
            len(solution),
            sum(implicant.literal_count for implicant in solution),
            tuple(implicant.pattern for implicant in solution),
        )

    def search(
        chosen: tuple[Implicant, ...],
        uncovered: frozenset[int],
        candidates: tuple[Implicant, ...],
    ) -> None:
        nonlocal best_solution
        partial = tuple(sorted(selected + list(chosen), key=_sort_key))
        if best_solution is not None and solution_score(partial) >= solution_score(best_solution):
            return
        if not uncovered:
            best_solution = partial
            return
        target = min(
            uncovered,
            key=lambda minterm: len([imp for imp in candidates if imp.covers(minterm)]),
        )
        options = sorted(
            [implicant for implicant in candidates if implicant.covers(target)],
            key=lambda implicant: (-len(implicant.minterms & uncovered), _sort_key(implicant)),
        )
        for implicant in options:
            new_uncovered = frozenset(uncovered - implicant.minterms)
            new_candidates = tuple(
                candidate
                for candidate in candidates
                if candidate != implicant and candidate.minterms & new_uncovered
            )
            search(chosen + (implicant,), new_uncovered, new_candidates)

    search((), frozenset(remaining_minterms), tuple(remaining_primes))
    if best_solution is None:
        raise RuntimeError("Unable to build an exact cover for the prime implicants")
    return best_solution


def minimize_function(function: BooleanFunction, form: str = "dnf") -> MinimizationResult:
    form = _normalize_form(form)
    variables = function.variables
    variable_count = len(variables)
    minterms = _target_indexes(function, form)
    if not minterms:
        return MinimizationResult((), (), (), (), (), (), _empty_expression(form), form=form)
    if variable_count == 0:
        universal = Implicant("", frozenset({0}))
        return MinimizationResult(
            (),
            (universal,),
            (universal,),
            (),
            (0,),
            (),
            _universal_expression(form),
            form=form,
        )

    current = {
        format(index, f"0{variable_count}b"): Implicant(
            format(index, f"0{variable_count}b"), frozenset({index})
        )
        for index in minterms
    }
    stages: list[GluingStage] = []
    prime_map: dict[str, Implicant] = {}

    if current:
        stage_number = 1
        while current:
            current_implicants = tuple(sorted(current.values(), key=_sort_key))
            used_patterns: set[str] = set()
            next_map: dict[str, Implicant] = {}
            pairs: list[GluingPair] = []
            current_list = list(current_implicants)
            for left_index, left in enumerate(current_list):
                for right in current_list[left_index + 1 :]:
                    combined = left.combine(right)
                    if combined is None:
                        continue
                    used_patterns.add(left.pattern)
                    used_patterns.add(right.pattern)
                    next_map[combined.pattern] = _merge_implicants(
                        next_map.get(combined.pattern), combined
                    )
                    pairs.append(GluingPair(left, right, next_map[combined.pattern]))
            for implicant in current_implicants:
                if implicant.pattern not in used_patterns:
                    prime_map[implicant.pattern] = _merge_implicants(
                        prime_map.get(implicant.pattern), implicant
                    )
            if not pairs:
                break
            output_terms = tuple(sorted(next_map.values(), key=_sort_key))
            stages.append(
                GluingStage(stage_number, current_implicants, tuple(pairs), output_terms)
            )
            current = {implicant.pattern: implicant for implicant in output_terms}
            stage_number += 1

    if len(minterms) == (1 << variable_count):
        universal = Implicant("-" * variable_count, frozenset(minterms))
        prime_implicants = (universal,)
        selected_implicants = (universal,)
    else:
        prime_implicants = tuple(sorted(prime_map.values(), key=_sort_key))
        selected_implicants = _exact_cover(minterms, list(prime_implicants))

    selected_patterns = {implicant.pattern for implicant in selected_implicants}
    redundant_implicants = tuple(
        implicant
        for implicant in prime_implicants
        if implicant.pattern not in selected_patterns
    )
    chart_columns = tuple(minterms)
    chart_rows = tuple(
        PrimeChartRow(
            implicant,
            tuple(implicant.covers(minterm) for minterm in chart_columns),
        )
        for implicant in prime_implicants
    )
    return MinimizationResult(
        gluing_stages=tuple(stages),
        prime_implicants=prime_implicants,
        selected_implicants=selected_implicants,
        redundant_implicants=redundant_implicants,
        chart_columns=chart_columns,
        chart_rows=chart_rows,
        expression=_expression_from_implicants(selected_implicants, variables, form),
        form=form,
    )


def _karnaugh_layout(
    variables: tuple[str, ...],
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    if len(variables) <= 1:
        return variables, (), ()
    if len(variables) == 2:
        return (variables[0],), (variables[1],), ()
    if len(variables) == 3:
        return (variables[0],), variables[1:], ()
    if len(variables) == 4:
        return variables[:2], variables[2:], ()
    return variables[:2], variables[2:4], variables[4:]


def _bits_label(bits: tuple[int, ...]) -> str:
    return "".join(str(bit) for bit in bits) or "-"


def _karnaugh_layers(
    function: BooleanFunction,
) -> tuple[
    tuple[KarnaughLayer, ...],
    dict[tuple[int, ...], tuple[str, str, str]],
    dict[tuple[int, int, int], tuple[int, ...]],
    tuple[int, int, int],
]:
    variables = function.variables
    row_variables, column_variables, layer_variables = _karnaugh_layout(variables)
    row_codes = gray_code(len(row_variables))
    column_codes = gray_code(len(column_variables))
    layer_codes = gray_code(len(layer_variables))
    variable_positions = {variable: index for index, variable in enumerate(variables)}
    layers: list[KarnaughLayer] = []
    cell_lookup: dict[tuple[int, ...], tuple[str, str, str]] = {}
    coordinate_lookup: dict[tuple[int, int, int], tuple[int, ...]] = {}

    for layer_index, layer_bits in enumerate(layer_codes):
        grid_rows = []
        layer_label = (
            ", ".join(f"{name}={bit}" for name, bit in zip(layer_variables, layer_bits))
            if layer_variables
            else "base"
        )
        for row_index, row_bits in enumerate(row_codes):
            values = []
            for column_index, column_bits in enumerate(column_codes):
                assignment = [0] * len(variables)
                for name, bit in zip(row_variables, row_bits):
                    assignment[variable_positions[name]] = bit
                for name, bit in zip(column_variables, column_bits):
                    assignment[variable_positions[name]] = bit
                for name, bit in zip(layer_variables, layer_bits):
                    assignment[variable_positions[name]] = bit
                bits = tuple(assignment)
                coordinate_lookup[(layer_index, row_index, column_index)] = bits
                cell_lookup[bits] = (
                    layer_label,
                    _bits_label(row_bits),
                    _bits_label(column_bits),
                )
                values.append(function.truth_vector[bits_to_index(bits)])
            grid_rows.append(tuple(values))
        layers.append(
            KarnaughLayer(
                label=layer_label,
                row_variables=row_variables,
                column_variables=column_variables,
                row_labels=tuple(_bits_label(bits) for bits in row_codes),
                column_labels=tuple(_bits_label(bits) for bits in column_codes),
                grid=tuple(grid_rows),
            )
        )
    return (
        tuple(layers),
        cell_lookup,
        coordinate_lookup,
        (len(layer_codes), len(row_codes), len(column_codes)),
    )


def _power_of_two_sizes(limit: int) -> tuple[int, ...]:
    sizes = [1]
    value = 2
    while value <= limit:
        sizes.append(value)
        value *= 2
    return tuple(sizes)


def _wrapped_indexes(size: int, length: int, start: int) -> tuple[int, ...]:
    return tuple((start + offset) % size for offset in range(length))


def _pattern_from_cells(cells: tuple[tuple[int, ...], ...]) -> str:
    if not cells:
        return ""
    width = len(cells[0])
    pattern = []
    for position in range(width):
        values = {bits[position] for bits in cells}
        pattern.append(str(values.pop()) if len(values) == 1 else "-")
    return "".join(pattern)


def build_karnaugh_map(
    function: BooleanFunction,
    result: MinimizationResult | None = None,
    form: str = "dnf",
) -> KarnaughMap:
    del result  # Retained for backward compatibility; the map minimizes independently.
    form = _normalize_form(form)
    layers, cell_lookup, coordinate_lookup, dimensions = _karnaugh_layers(function)
    minterms = _target_indexes(function, form)
    if not minterms:
        return KarnaughMap(layers, (), _empty_expression(form), form=form)

    candidates: dict[str, Implicant] = {}
    candidate_cells: dict[str, tuple[tuple[str, str, str], ...]] = {}
    layer_size, row_size, column_size = dimensions
    layer_lengths = _power_of_two_sizes(layer_size)
    row_lengths = _power_of_two_sizes(row_size)
    column_lengths = _power_of_two_sizes(column_size)

    for layer_length in layer_lengths:
        for row_length in row_lengths:
            for column_length in column_lengths:
                for layer_start in range(layer_size):
                    layer_indexes = _wrapped_indexes(layer_size, layer_length, layer_start)
                    for row_start in range(row_size):
                        row_indexes = _wrapped_indexes(row_size, row_length, row_start)
                        for column_start in range(column_size):
                            column_indexes = _wrapped_indexes(
                                column_size, column_length, column_start
                            )
                            bits_group = tuple(
                                coordinate_lookup[(layer_index, row_index, column_index)]
                                for layer_index in layer_indexes
                                for row_index in row_indexes
                                for column_index in column_indexes
                            )
                            if any(
                                function._value_by_bits[bits] != _target_value(form)
                                for bits in bits_group
                            ):
                                continue
                            pattern = _pattern_from_cells(bits_group)
                            candidates.setdefault(
                                pattern,
                                Implicant(
                                    pattern,
                                    frozenset(bits_to_index(bits) for bits in bits_group),
                                ),
                            )
                            candidate_cells.setdefault(
                                pattern,
                                tuple(sorted({cell_lookup[bits] for bits in bits_group})),
                            )

    prime_implicants = tuple(
        sorted(
            (
                implicant
                for implicant in candidates.values()
                if not any(
                    implicant.minterms < other.minterms for other in candidates.values()
                )
            ),
            key=_sort_key,
        )
    )
    selected_implicants = _exact_cover(minterms, list(prime_implicants))
    groups = tuple(
        KarnaughGroup(implicant, candidate_cells[implicant.pattern])
        for implicant in selected_implicants
    )
    return KarnaughMap(
        layers=layers,
        groups=groups,
        expression=_expression_from_implicants(selected_implicants, function.variables, form),
        form=form,
    )
