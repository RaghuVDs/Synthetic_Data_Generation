from __future__ import annotations

import numpy as np
import pytest

from synthgen.domain.errors import ExpressionError
from synthgen.spec.evaluator import Column, evaluate
from synthgen.spec.expressions import (
    MAX_DEPTH,
    MAX_LENGTH,
    Binary,
    Conditional,
    ExprType,
    Literal,
    Ref,
    Unary,
    parse,
    references,
    type_of,
)

N = ExprType.NUMBER
COLUMNS: dict[str, Column] = {
    "a": (N, np.array([1.0, 2.0, 3.0, 4.0])),
    "b": (N, np.array([10, None, 30, 0], dtype=object)),
    "name": (ExprType.TEXT, np.array(["x", "y", None, "w"], dtype=object)),
    "flag": (ExprType.BOOLEAN, np.array([1, 0, 1, 0], dtype=object)),
    "day": (
        ExprType.DATE,
        np.array(["2024-01-01", "2024-02-29", None, "2024-12-31"], dtype=object),
    ),
    "at": (
        ExprType.DATETIME,
        np.array(
            ["2024-01-01 10:00:00", "2024-03-01 00:00:00", "2024-01-01 00:00:00", None],
            dtype=object,
        ),
    ),
    "parent.opened": (
        ExprType.DATE,
        np.array(["2023-12-31", "2024-01-01", "2024-01-01", "2024-01-01"], dtype=object),
    ),
}
TYPES = {name: kind for name, (kind, _) in COLUMNS.items()}


def run(text: str) -> list[object]:
    return list(evaluate(parse(text), COLUMNS, 4).tolist())


def test_precedence_and_associativity() -> None:
    assert parse("1 + 2 * 3") == Binary("+", Literal(1.0), Binary("*", Literal(2.0), Literal(3.0)))
    assert parse("8 - 2 - 1") == Binary("-", Binary("-", Literal(8.0), Literal(2.0)), Literal(1.0))
    assert parse("not a > 1 and true") == Binary(
        "and", Unary("not", Binary(">", Ref("a"), Literal(1.0))), Literal(True)
    )
    assert parse("-(a)") == Unary("-", Ref("a"))


def test_conditional_is_python_style_and_right_nested() -> None:
    tree = parse("1 if a > 2 else 2 if a > 1 else 3")
    assert isinstance(tree, Conditional)
    assert isinstance(tree.other, Conditional)
    assert run("1 if a > 2 else 2 if a > 1 else 3") == [3.0, 2.0, 1.0, 1.0]


def test_references_include_parent_columns() -> None:
    assert references(parse("date_diff(day, parent.opened, 'days') + a")) == {
        "day",
        "parent.opened",
        "a",
    }


def test_arithmetic_propagates_nulls_and_division_by_zero() -> None:
    assert run("a + b") == [11.0, None, 33.0, 4.0]
    assert run("a / b") == [0.1, None, 0.1, None]
    assert run("a % 3") == [1.0, 2.0, 0.0, 1.0]
    assert run("round(a / 3, 2)") == [0.33, 0.67, 1.0, 1.33]
    assert run("round(a * 1.6)") == [2.0, 3.0, 5.0, 6.0]


def test_comparisons_with_null_are_false() -> None:
    assert run("b > 5") == [1, 0, 1, 0]
    assert run("b != 10") == [0, 0, 1, 1]
    assert run("name == 'x' or name == 'w'") == [1, 0, 0, 1]
    assert run("flag == true") == [1, 0, 1, 0]


def test_min_max_on_numbers_and_dates() -> None:
    assert run("max(a, 2, 3.5)") == [3.5, 3.5, 3.5, 4.0]
    assert run("min(day, parent.opened)") == [
        "2023-12-31",
        "2024-01-01",
        None,
        "2024-01-01",
    ]


def test_date_add_and_diff() -> None:
    assert run("date_add(day, 1, 'days')") == ["2024-01-02", "2024-03-01", None, "2025-01-01"]
    assert run("date_add(day, 90, 'minutes')")[0] == "2024-01-01 01:30:00"
    assert run("date_add(at, -1, 'hours')")[:2] == ["2024-01-01 09:00:00", "2024-02-29 23:00:00"]
    assert run("date_diff(day, parent.opened, 'days')") == [1.0, 59.0, None, 365.0]
    assert type_of(parse("date_add(day, 2, 'weeks')"), TYPES) is ExprType.DATE
    assert type_of(parse("date_add(day, 2, 'hours')"), TYPES) is ExprType.DATETIME


def test_concat_formats_each_type() -> None:
    assert run("concat(name, '-', a, '-', flag == true)") == [
        "x-1-true",
        "y-2-false",
        None,
        "w-4-false",
    ]
    assert run("concat(day, ' ', 0.5)")[0] == "2024-01-01 0.5"


def test_dates_compare_with_date_times() -> None:
    assert run("at >= day") == [1, 1, 0, 0]


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("", "empty"),
        ("a +", "ends too early"),
        ("a b", "unexpected 'b'"),
        ("(a", r"expected '\)'"),
        ("1 < a < 3", "chained"),
        ("a @ 2", "unexpected character"),
        ("__import__('os')", "unknown function"),
        ("eval('1')", "unknown function"),
        ("a.b.c", "unexpected"),
        ("1 if a > 1", "expected 'else'"),
        ("a ** 2", "unexpected"),
        ("x[0]", "unexpected character"),
        ("lambda: 1", "unexpected character"),
    ],
)
def test_rejects_input_outside_the_grammar(text: str, message: str) -> None:
    with pytest.raises(ExpressionError, match=message):
        parse(text)


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("missing + 1", "unknown column missing"),
        ("name + 1", "needs number, got text"),
        ("a and flag", "needs boolean, got number"),
        ("flag < flag", "cannot order booleans"),
        ("a == name", "mixes"),
        ("1 if a else 2", "if needs boolean"),
        ("1 if a > 1 else 'x'", "mixes"),
        ("round(a, b)", "whole number literal"),
        ("round(a, 1.5)", "whole number literal"),
        ("date_add(day, 1, 'years')", "unit must be"),
        ("date_add(a, 1, 'days')", "needs date or datetime"),
        ("date_diff(day, a, 'days')", "needs date or datetime"),
        ("min(a)", "takes 2\\+ arguments"),
        ("min(name, name)", "min needs"),
        ("round()", "unexpected"),
        ("concat(a)", "takes 2\\+"),
        ("-name", "- needs number"),
        ("not a", "not needs boolean"),
    ],
)
def test_type_check_rejects_misuse(text: str, message: str) -> None:
    with pytest.raises(ExpressionError, match=message):
        type_of(parse(text), TYPES)


def test_size_limits() -> None:
    with pytest.raises(ExpressionError, match="longer than"):
        parse("1+" * MAX_LENGTH + "1")
    with pytest.raises(ExpressionError, match="deeper"):
        parse("(" * (MAX_DEPTH + 1) + "1" + ")" * (MAX_DEPTH + 1))


def test_large_vectors_evaluate_without_python_loops_on_numbers() -> None:
    n = 200_000
    columns: dict[str, Column] = {"x": (N, np.arange(n, dtype=np.float64))}
    out = evaluate(parse("x * 2 + 1 if x > 10 else 0"), columns, n)
    assert out.dtype == np.float64
    assert out[11] == 23.0
    assert out[3] == 0.0
