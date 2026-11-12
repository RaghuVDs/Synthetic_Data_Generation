"""The `derived` expression grammar (SPEC 8.4): our own tokenizer, parser and type check.

Expressions are parsed here, never by Python, so a model-written expression can only combine
column values with the operators and functions listed below.

    expr       := logic ["if" logic "else" expr]          (Python-style conditional)
    logic      := conjunct {"or" conjunct}
    conjunct   := negation {"and" negation}
    negation   := "not" negation | comparison
    comparison := sum [("<" | "<=" | ">" | ">=" | "==" | "!=") sum]
    sum        := product {("+" | "-") product}
    product    := unary {("*" | "/" | "%") unary}
    unary      := "-" unary | atom
    atom       := number | 'text' | true | false | name ["(" expr {"," expr} ")"] | "(" expr ")"

A name is a column, or `table.column` for a column of the driving parent row. Functions: min,
max, round, date_add(date, amount, unit), date_diff(later, earlier, unit) and concat.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum

from synthgen.domain.errors import ExpressionError

MAX_LENGTH = 1000
"""Longest expression accepted, in characters; keeps parsing and evaluation cheap."""
MAX_DEPTH = 40
"""Deepest nesting accepted; the parser is recursive."""
UNITS = {"seconds": 1, "minutes": 60, "hours": 3_600, "days": 86_400, "weeks": 604_800}
COMPARISONS = ("<", "<=", ">", ">=", "==", "!=")
ARITHMETIC = ("+", "-", "*", "/", "%")
_KEYWORDS = {"and", "or", "not", "if", "else", "true", "false"}
_TOKEN = re.compile(
    r"\s*(?:(?P<number>(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)"
    r"|(?P<text>'[^']*'|\"[^\"]*\")"
    r"|(?P<name>[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)?)"
    r"|(?P<op><=|>=|==|!=|[-+*/%<>(),]))"
)


class ExprType(StrEnum):
    """The value type of an expression or of a column it reads."""

    NUMBER = "number"
    TEXT = "text"
    BOOLEAN = "boolean"
    DATE = "date"
    DATETIME = "datetime"


TIMES = (ExprType.DATE, ExprType.DATETIME)


@dataclass(frozen=True)
class Literal:
    """A number, text or boolean written in the expression."""

    value: float | str | bool


@dataclass(frozen=True)
class Ref:
    """A column reference: `column`, or `table.column` for the driving parent."""

    name: str


@dataclass(frozen=True)
class Unary:
    """`-x` or `not x`."""

    op: str
    operand: Expr


@dataclass(frozen=True)
class Binary:
    """An arithmetic, comparison or logical operator."""

    op: str
    left: Expr
    right: Expr


@dataclass(frozen=True)
class Conditional:
    """`then if condition else other`."""

    condition: Expr
    then: Expr
    other: Expr


@dataclass(frozen=True)
class Call:
    """A whitelisted function call."""

    name: str
    args: tuple[Expr, ...]


Expr = Literal | Ref | Unary | Binary | Conditional | Call


def _tokens(text: str) -> list[tuple[str, str]]:
    found: list[tuple[str, str]] = []
    position = 0
    stripped = text.rstrip()
    while position < len(stripped):
        match = _TOKEN.match(stripped, position)
        if match is None or match.end() == position:
            raise ExpressionError(f"unexpected character at position {position}: {text!r}")
        kind = match.lastgroup or ""
        value = match.group(kind)
        found.append(("keyword" if kind == "name" and value in _KEYWORDS else kind, value))
        position = match.end()
    return found


class _Parser:
    """Recursive descent over the token list; one method per grammar rule."""

    def __init__(self, text: str) -> None:
        self._tokens = _tokens(text)
        self._at = 0
        self._depth = 0

    def parse(self) -> Expr:
        expr = self._expr()
        if self._at != len(self._tokens):
            raise ExpressionError(f"unexpected {self._tokens[self._at][1]!r}")
        return expr

    def _peek(self) -> str | None:
        return self._tokens[self._at][1] if self._at < len(self._tokens) else None

    def _take(self, *values: str) -> str | None:
        if self._peek() in values and self._tokens[self._at][0] in ("op", "keyword"):
            self._at += 1
            return self._tokens[self._at - 1][1]
        return None

    def _expect(self, value: str) -> None:
        if self._take(value) is None:
            raise ExpressionError(f"expected {value!r}, found {self._peek()!r}")

    def _expr(self) -> Expr:
        self._depth += 1
        if self._depth > MAX_DEPTH:
            raise ExpressionError(f"expression nests deeper than {MAX_DEPTH} levels")
        then = self._binary(0)
        if self._take("if"):
            condition = self._binary(0)
            self._expect("else")
            then = Conditional(condition=condition, then=then, other=self._expr())
        self._depth -= 1
        return then

    _LEVELS: tuple[tuple[str, ...], ...] = (("or",), ("and",), COMPARISONS, ("+", "-"))

    def _binary(self, level: int) -> Expr:
        """Left-associative operators by precedence level; `not` sits between and and compare."""
        if level == len(self._LEVELS):
            return self._product()
        if level == 2 and self._take("not"):  # noqa: PLR2004 - the comparison level
            return Unary(op="not", operand=self._binary(level))
        left = self._binary(level + 1)
        while op := self._take(*self._LEVELS[level]):
            left = Binary(op=op, left=left, right=self._binary(level + 1))
            if self._LEVELS[level] is COMPARISONS and self._peek() in COMPARISONS:
                raise ExpressionError("comparisons cannot be chained; join them with and")
        return left

    def _product(self) -> Expr:
        left = self._unary()
        while op := self._take("*", "/", "%"):
            left = Binary(op=op, left=left, right=self._unary())
        return left

    def _unary(self) -> Expr:
        if self._take("-"):
            return Unary(op="-", operand=self._unary())
        return self._atom()

    def _atom(self) -> Expr:
        if self._at >= len(self._tokens):
            raise ExpressionError("expression ends too early")
        kind, value = self._tokens[self._at]
        self._at += 1
        if kind == "number":
            return Literal(value=float(value))
        if kind == "text":
            return Literal(value=value[1:-1])
        if value in ("true", "false"):
            return Literal(value=value == "true")
        if value == "(":
            inner = self._expr()
            self._expect(")")
            return inner
        if kind == "name" and self._take("("):
            return self._call(value)
        if kind == "name":
            return Ref(name=value)
        raise ExpressionError(f"unexpected {value!r}")

    def _call(self, name: str) -> Expr:
        if name not in FUNCTIONS:
            raise ExpressionError(f"unknown function {name}; allowed: {', '.join(FUNCTIONS)}")
        args = [self._expr()]
        while self._take(","):
            args.append(self._expr())
        self._expect(")")
        return Call(name=name, args=tuple(args))


def parse(text: str) -> Expr:
    """The syntax tree of an expression; raises ExpressionError on anything outside the grammar."""
    if len(text) > MAX_LENGTH:
        raise ExpressionError(f"expression is longer than {MAX_LENGTH} characters")
    if not text.strip():
        raise ExpressionError("expression is empty")
    return _Parser(text).parse()


def references(expr: Expr) -> set[str]:
    """Every column name the expression reads."""
    if isinstance(expr, Ref):
        return {expr.name}
    if isinstance(expr, Unary):
        return references(expr.operand)
    if isinstance(expr, Binary):
        return references(expr.left) | references(expr.right)
    if isinstance(expr, Conditional):
        return references(expr.condition) | references(expr.then) | references(expr.other)
    if isinstance(expr, Call):
        return set().union(*(references(a) for a in expr.args))
    return set()


Types = Mapping[str, ExprType]


def type_of(expr: Expr, types: Types) -> ExprType:
    """The type the expression evaluates to; raises ExpressionError on unknown columns or misuse."""
    if isinstance(expr, Literal | Ref):
        return _leaf_type(expr, types)
    if isinstance(expr, Unary):
        want = ExprType.NUMBER if expr.op == "-" else ExprType.BOOLEAN
        _require(type_of(expr.operand, types), (want,), expr.op)
        return want
    if isinstance(expr, Binary):
        return _binary_type(expr, types)
    if isinstance(expr, Conditional):
        _require(type_of(expr.condition, types), (ExprType.BOOLEAN,), "if")
        return _common((type_of(expr.then, types), type_of(expr.other, types)), "if/else")
    return FUNCTIONS[expr.name](expr.args, types)


def _leaf_type(expr: Literal | Ref, types: Types) -> ExprType:
    if isinstance(expr, Ref):
        if expr.name not in types:
            raise ExpressionError(f"unknown column {expr.name}")
        return types[expr.name]
    if isinstance(expr.value, bool):
        return ExprType.BOOLEAN
    return ExprType.TEXT if isinstance(expr.value, str) else ExprType.NUMBER


def _require(found: ExprType, allowed: tuple[ExprType, ...], where: str) -> None:
    if found not in allowed:
        names = " or ".join(a.value for a in allowed)
        raise ExpressionError(f"{where} needs {names}, got {found.value}")


def _common(found: tuple[ExprType, ...], where: str) -> ExprType:
    """The shared type of several operands; a date mixed with a date-time is a date-time."""
    kinds = set(found)
    if len(kinds) == 1:
        return found[0]
    if kinds <= set(TIMES):
        return ExprType.DATETIME
    raise ExpressionError(f"{where} mixes {', '.join(sorted(k.value for k in kinds))}")


def _binary_type(expr: Binary, types: Types) -> ExprType:
    left, right = type_of(expr.left, types), type_of(expr.right, types)
    if expr.op in ("and", "or"):
        _require(left, (ExprType.BOOLEAN,), expr.op)
        _require(right, (ExprType.BOOLEAN,), expr.op)
        return ExprType.BOOLEAN
    if expr.op in ARITHMETIC:
        _require(left, (ExprType.NUMBER,), expr.op)
        _require(right, (ExprType.NUMBER,), expr.op)
        return ExprType.NUMBER
    shared = _common((left, right), expr.op)
    if shared is ExprType.BOOLEAN and expr.op not in ("==", "!="):
        raise ExpressionError(f"{expr.op} cannot order booleans")
    return ExprType.BOOLEAN


def _unit(arg: Expr, where: str) -> str:
    if not isinstance(arg, Literal) or arg.value not in UNITS:
        raise ExpressionError(f"{where} unit must be one of {', '.join(UNITS)} in quotes")
    return str(arg.value)


def _arity(args: tuple[Expr, ...], low: int, high: int | None, name: str) -> None:
    if len(args) < low or (high is not None and len(args) > high):
        raise ExpressionError(f"{name} takes {low}{'+' if high is None else f'-{high}'} arguments")


def _extreme(name: str) -> Callable[[tuple[Expr, ...], Types], ExprType]:
    def check(args: tuple[Expr, ...], types: Types) -> ExprType:
        _arity(args, 2, None, name)
        shared = _common(tuple(type_of(a, types) for a in args), name)
        _require(shared, (ExprType.NUMBER, *TIMES), name)
        return shared

    return check


def _round(args: tuple[Expr, ...], types: Types) -> ExprType:
    _arity(args, 1, 2, "round")
    _require(type_of(args[0], types), (ExprType.NUMBER,), "round")
    if len(args) == 2 and not (  # noqa: PLR2004 - round(x, digits)
        isinstance(args[1], Literal)
        and isinstance(args[1].value, float)
        and args[1].value.is_integer()
    ):
        raise ExpressionError("round digits must be a whole number literal")
    return ExprType.NUMBER


def _date_add(args: tuple[Expr, ...], types: Types) -> ExprType:
    _arity(args, 3, 3, "date_add")
    base = type_of(args[0], types)
    _require(base, TIMES, "date_add")
    _require(type_of(args[1], types), (ExprType.NUMBER,), "date_add amount")
    unit = _unit(args[2], "date_add")
    return (
        ExprType.DATE if base is ExprType.DATE and unit in ("days", "weeks") else ExprType.DATETIME
    )


def _date_diff(args: tuple[Expr, ...], types: Types) -> ExprType:
    _arity(args, 3, 3, "date_diff")
    _require(type_of(args[0], types), TIMES, "date_diff")
    _require(type_of(args[1], types), TIMES, "date_diff")
    _unit(args[2], "date_diff")
    return ExprType.NUMBER


def _concat(args: tuple[Expr, ...], types: Types) -> ExprType:
    _arity(args, 2, None, "concat")
    for arg in args:
        type_of(arg, types)
    return ExprType.TEXT


FUNCTIONS: dict[str, Callable[[tuple[Expr, ...], Types], ExprType]] = {
    "min": _extreme("min"),
    "max": _extreme("max"),
    "round": _round,
    "date_add": _date_add,
    "date_diff": _date_diff,
    "concat": _concat,
}
"""Whitelisted functions and their type rules."""
