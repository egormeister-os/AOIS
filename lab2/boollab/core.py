from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from itertools import combinations, product
import re
from typing import Callable, Iterable


ALLOWED_VARIABLES = ("a", "b", "c", "d", "e")
TOKEN_REGEX = re.compile(r"\s+|->|[()!&|~01]|[a-eA-E]|.")


class ExpressionError(ValueError):
    """Raised when the boolean expression cannot be parsed."""


@dataclass(frozen=True)
class Token:
    kind: str
    value: str
    position: int


class Node:
    def evaluate(self, assignment: dict[str, int]) -> int:
        raise NotImplementedError

    def variables(self) -> set[str]:
        raise NotImplementedError


@dataclass(frozen=True)
class ConstNode(Node):
    value: int

    def evaluate(self, assignment: dict[str, int]) -> int:
        return self.value

    def variables(self) -> set[str]:
        return set()


@dataclass(frozen=True)
class VariableNode(Node):
    name: str

    def evaluate(self, assignment: dict[str, int]) -> int:
        return assignment[self.name]

    def variables(self) -> set[str]:
        return {self.name}


@dataclass(frozen=True)
class UnaryNode(Node):
    operand: Node

    def evaluate(self, assignment: dict[str, int]) -> int:
        return 1 - self.operand.evaluate(assignment)

    def variables(self) -> set[str]:
        return self.operand.variables()


@dataclass(frozen=True)
class BinaryNode(Node):
    operator: str
    left: Node
    right: Node

    def evaluate(self, assignment: dict[str, int]) -> int:
        left = self.left.evaluate(assignment)
        right = self.right.evaluate(assignment)
        if self.operator == "&":
            return left & right
        if self.operator == "|":
            return left | right
        if self.operator == "->":
            return int((not left) or right)
        if self.operator == "~":
            return int(left == right)
        raise ExpressionError(f"Unsupported operator: {self.operator}")

    def variables(self) -> set[str]:
        return self.left.variables() | self.right.variables()


def normalize_expression(expression: str) -> str:
    replacements = {
        "¬": "!",
        "∧": "&",
        "∨": "|",
        "→": "->",
        "↔": "~",
    }
    normalized = expression
    for source, target in replacements.items():
        normalized = normalized.replace(source, target)
    return normalized


def tokenize(expression: str) -> list[Token]:
    normalized = normalize_expression(expression)
    tokens: list[Token] = []
    for match in TOKEN_REGEX.finditer(normalized):
        value = match.group(0)
        position = match.start()
        if value.isspace():
            continue
        if value in {"(", ")", "!", "&", "|", "~", "->"}:
            tokens.append(Token("operator", value, position))
        elif value in {"0", "1"}:
            tokens.append(Token("const", value, position))
        elif value.lower() in ALLOWED_VARIABLES:
            tokens.append(Token("var", value.lower(), position))
        else:
            raise ExpressionError(f"Unexpected symbol '{value}' at position {position}")
    tokens.append(Token("eof", "", len(normalized)))
    return tokens


class ExpressionParser:
    def __init__(self, expression: str):
        self.expression = normalize_expression(expression)
        self.tokens = tokenize(expression)
        self.position = 0

    def current(self) -> Token:
        return self.tokens[self.position]

    def advance(self) -> Token:
        token = self.current()
        self.position += 1
        return token

    def parse(self) -> Node:
        if self.current().kind == "eof":
            raise ExpressionError("Expression is empty")
        node = self.parse_equivalence()
        if self.current().kind != "eof":
            token = self.current()
            raise ExpressionError(
                f"Unexpected token '{token.value}' at position {token.position}"
            )
        variables = node.variables()
        if len(variables) > 5:
            raise ExpressionError("A function may contain at most 5 variables")
        return node

    def parse_equivalence(self) -> Node:
        node = self.parse_implication()
        while self.current().value == "~":
            operator = self.advance().value
            node = BinaryNode(operator, node, self.parse_implication())
        return node

    def parse_implication(self) -> Node:
        node = self.parse_disjunction()
        if self.current().value == "->":
            operator = self.advance().value
            node = BinaryNode(operator, node, self.parse_implication())
        return node

    def parse_disjunction(self) -> Node:
        node = self.parse_conjunction()
        while self.current().value == "|":
            operator = self.advance().value
            node = BinaryNode(operator, node, self.parse_conjunction())
        return node

    def parse_conjunction(self) -> Node:
        node = self.parse_unary()
        while self.current().value == "&":
            operator = self.advance().value
            node = BinaryNode(operator, node, self.parse_unary())
        return node

    def parse_unary(self) -> Node:
        token = self.current()
        if token.value == "!":
            self.advance()
            return UnaryNode(self.parse_unary())
        if token.value == "(":
            self.advance()
            node = self.parse_equivalence()
            if self.current().value != ")":
                raise ExpressionError(f"Expected ')' at position {self.current().position}")
            self.advance()
            return node
        if token.kind == "var":
            self.advance()
            return VariableNode(token.value)
        if token.kind == "const":
            self.advance()
            return ConstNode(int(token.value))
        raise ExpressionError(f"Unexpected token '{token.value}' at position {token.position}")


@dataclass(frozen=True)
class TruthRow:
    index: int
    bits: tuple[int, ...]
    value: int


def bits_to_index(bits: Iterable[int]) -> int:
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def index_to_bits(index: int, size: int) -> tuple[int, ...]:
    if size == 0:
        return ()
    return tuple((index >> shift) & 1 for shift in range(size - 1, -1, -1))


def _wrap_term(content: str) -> str:
    if content in {"0", "1"}:
        return content
    if "&" not in content and "|" not in content:
        return content
    return f"({content})"


def format_minterm(bits: tuple[int, ...], variables: tuple[str, ...]) -> str:
    literals = [name if bit else f"!{name}" for name, bit in zip(variables, bits)]
    if not literals:
        return "1"
    return _wrap_term("&".join(literals))


def format_maxterm(bits: tuple[int, ...], variables: tuple[str, ...]) -> str:
    literals = [name if not bit else f"!{name}" for name, bit in zip(variables, bits)]
    if not literals:
        return "0"
    return _wrap_term("|".join(literals))


def format_implicant_pattern(pattern: str, variables: tuple[str, ...]) -> str:
    literals = []
    for bit, variable in zip(pattern, variables):
        if bit == "-":
            continue
        literals.append(variable if bit == "1" else f"!{variable}")
    if not literals:
        return "1"
    return _wrap_term("&".join(literals))


def gray_code(bits: int) -> list[tuple[int, ...]]:
    if bits == 0:
        return [()]
    previous = gray_code(bits - 1)
    return [(0,) + code for code in previous] + [(1,) + code for code in reversed(previous)]


@dataclass
class BooleanFunction:
    variables: tuple[str, ...]
    evaluator: Callable[[dict[str, int]], int]
    source_expression: str
    ast: Node | None = None

    @classmethod
    def from_expression(cls, expression: str) -> "BooleanFunction":
        parser = ExpressionParser(expression)
        ast = parser.parse()
        variables = tuple(sorted(ast.variables()))
        return cls(
            variables=variables,
            evaluator=ast.evaluate,
            source_expression=normalize_expression(expression),
            ast=ast,
        )

    @classmethod
    def from_truth_vector(
        cls,
        variables: Iterable[str],
        values: Iterable[int],
        source_expression: str = "<derived>",
    ) -> "BooleanFunction":
        variables_tuple = tuple(variables)
        values_list = [int(value) for value in values]
        expected = 1 << len(variables_tuple)
        if len(values_list) != expected:
            raise ValueError(
                f"Expected {expected} truth values for variables {variables_tuple}, "
                f"got {len(values_list)}"
            )
        lookup = {
            index_to_bits(index, len(variables_tuple)): value
            for index, value in enumerate(values_list)
        }

        def evaluator(assignment: dict[str, int]) -> int:
            bits = tuple(int(assignment[name]) for name in variables_tuple)
            return lookup[bits]

        return cls(variables_tuple, evaluator, source_expression)

    def evaluate(self, assignment: dict[str, int]) -> int:
        prepared = {name: int(assignment[name]) for name in self.variables}
        return int(self.evaluator(prepared))

    def evaluate_bits(self, bits: tuple[int, ...]) -> int:
        assignment = dict(zip(self.variables, bits))
        return self.evaluate(assignment)

    @cached_property
    def truth_table(self) -> list[TruthRow]:
        rows = []
        for bits in product((0, 1), repeat=len(self.variables)):
            rows.append(
                TruthRow(
                    index=bits_to_index(bits),
                    bits=tuple(int(bit) for bit in bits),
                    value=self.evaluate_bits(tuple(int(bit) for bit in bits)),
                )
            )
        return rows

    @cached_property
    def truth_vector(self) -> list[int]:
        return [row.value for row in self.truth_table]

    @cached_property
    def _value_by_bits(self) -> dict[tuple[int, ...], int]:
        return {row.bits: row.value for row in self.truth_table}

    def minterm_indexes(self) -> list[int]:
        return [row.index for row in self.truth_table if row.value == 1]

    def maxterm_indexes(self) -> list[int]:
        return [row.index for row in self.truth_table if row.value == 0]

    def sdnf(self) -> str:
        indexes = self.minterm_indexes()
        if not indexes:
            return "0"
        return " | ".join(
            format_minterm(index_to_bits(index, len(self.variables)), self.variables)
            for index in indexes
        )

    def sknf(self) -> str:
        indexes = self.maxterm_indexes()
        if not indexes:
            return "1"
        return " & ".join(
            format_maxterm(index_to_bits(index, len(self.variables)), self.variables)
            for index in indexes
        )

    def numeric_forms(self) -> dict[str, list[int]]:
        return {
            "sdnf": self.minterm_indexes(),
            "sknf": self.maxterm_indexes(),
        }

    def index_form(self) -> dict[str, int | str]:
        binary = "".join(str(value) for value in self.truth_vector)
        decimal = int(binary, 2) if binary else 0
        return {"binary": binary or "0", "decimal": decimal}

    def zhegalkin_coefficients(self) -> list[int]:
        coefficients = self.truth_vector[:]
        size = len(coefficients)
        bit = 1
        while bit < size:
            for index in range(size):
                if index & bit:
                    coefficients[index] ^= coefficients[index ^ bit]
            bit <<= 1
        return coefficients

    def zhegalkin_polynomial(self) -> str:
        coefficients = self.zhegalkin_coefficients()
        terms = []
        variable_count = len(self.variables)
        for mask, coefficient in enumerate(coefficients):
            if coefficient == 0:
                continue
            if mask == 0:
                terms.append("1")
                continue
            parts = []
            for position, variable in enumerate(self.variables):
                bit = 1 << (variable_count - position - 1)
                if mask & bit:
                    parts.append(variable)
            terms.append("".join(parts))
        return " ^ ".join(terms) if terms else "0"

    def post_classes(self) -> dict[str, bool]:
        all_zero_bits = tuple(0 for _ in self.variables)
        all_one_bits = tuple(1 for _ in self.variables)
        t0 = self.evaluate_bits(all_zero_bits) == 0
        t1 = self.evaluate_bits(all_one_bits) == 1
        self_dual = all(
            row.value != self._value_by_bits[tuple(1 - bit for bit in row.bits)]
            for row in self.truth_table
        )
        monotone = True
        for left in self.truth_table:
            for right in self.truth_table:
                if all(a <= b for a, b in zip(left.bits, right.bits)) and left.value > right.value:
                    monotone = False
                    break
            if not monotone:
                break
        linear = True
        for mask, coefficient in enumerate(self.zhegalkin_coefficients()):
            if coefficient and mask.bit_count() > 1:
                linear = False
                break
        return {
            "T0": t0,
            "T1": t1,
            "S": self_dual,
            "M": monotone,
            "L": linear,
        }

    def fictive_variables(self) -> list[str]:
        fictive = []
        size = len(self.variables)
        for position, variable in enumerate(self.variables):
            is_fictive = True
            for bits in product((0, 1), repeat=size):
                zero_bits = list(bits)
                one_bits = list(bits)
                zero_bits[position] = 0
                one_bits[position] = 1
                if self._value_by_bits[tuple(zero_bits)] != self._value_by_bits[tuple(one_bits)]:
                    is_fictive = False
                    break
            if is_fictive:
                fictive.append(variable)
        return fictive

    def derivative(self, variables: Iterable[str]) -> "BooleanFunction":
        selected = tuple(variables)
        indexes = [self.variables.index(variable) for variable in selected]
        values = []
        for bits in product((0, 1), repeat=len(self.variables)):
            value = 0
            for replacement in product((0, 1), repeat=len(indexes)):
                candidate = list(bits)
                for position, bit in zip(indexes, replacement):
                    candidate[position] = bit
                value ^= self._value_by_bits[tuple(candidate)]
            values.append(value)
        suffix = "".join(selected) if selected else "const"
        return BooleanFunction.from_truth_vector(
            self.variables,
            values,
            source_expression=f"D_{suffix}({self.source_expression})",
        )

    def all_derivatives(self, max_order: int = 4) -> dict[tuple[str, ...], "BooleanFunction"]:
        result: dict[tuple[str, ...], BooleanFunction] = {}
        upper_bound = min(max_order, len(self.variables))
        for order in range(1, upper_bound + 1):
            for variable_group in combinations(self.variables, order):
                result[variable_group] = self.derivative(variable_group)
        return result
