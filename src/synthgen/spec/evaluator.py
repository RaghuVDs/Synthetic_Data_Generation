"""Vectorised evaluation of parsed `derived` expressions. Pure: columns in, one column out.

Inside the evaluator numbers, dates and date-times are float64 (dates as seconds since 1970,
null as NaN), booleans are bool arrays and text is an object array with None for null.
Arithmetic and functions on a null give null; a comparison with a null is false.
"""

from __future__ import annotations

import operator
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
from numpy.typing import NDArray

from synthgen.domain.errors import ExpressionError
from synthgen.spec.expressions import (
    ARITHMETIC,
    TIMES,
    UNITS,
    Binary,
    Call,
    Conditional,
    Expr,
    ExprType,
    Literal,
    Ref,
    Unary,
    type_of,
)

Values = NDArray[Any]  # Any: float64, bool or object arrays, as the engine holds them
Column = tuple[ExprType, Values]
_COMPARE: dict[str, Callable[[Any, Any], Any]] = {  # Any: numpy arrays of any dtype
    "<": operator.lt,
    "<=": operator.le,
    ">": operator.gt,
    ">=": operator.ge,
    "==": operator.eq,
    "!=": operator.ne,
}
_ARITHMETIC: dict[str, Callable[[Any, Any], Any]] = {  # Any: float64 arrays
    "+": np.add,
    "-": np.subtract,
    "*": np.multiply,
    "/": np.divide,
    "%": np.mod,
}


def evaluate(expr: Expr, columns: Mapping[str, Column], n: int) -> Values:
    """The expression's value on each of n rows, as the engine stores that type.

    Numbers come back as float64 (object with None when any is null), booleans as 0/1,
    dates as ISO text.
    """
    types = {name: kind for name, (kind, _) in columns.items()}
    kind = type_of(expr, types)
    return to_output(_Evaluator(columns, types, n).run(expr), kind)


def to_internal(values: Values, kind: ExprType) -> Values:
    """Engine values to the evaluator's representation of `kind`."""
    if kind is ExprType.TEXT:
        return values.astype(object)
    if kind is ExprType.BOOLEAN:
        return np.array([bool(v) if v is not None else False for v in values], dtype=bool)
    if kind in TIMES:
        texts = [str(v).replace(" ", "T") if v is not None else "NaT" for v in values]
        stamps = _parse_times(texts)
        seconds = stamps.astype(np.int64).astype(np.float64)
        seconds[np.isnat(stamps)] = np.nan
        return seconds
    if values.dtype != object:
        return values.astype(np.float64)
    return np.array([np.nan if v is None else float(v) for v in values], dtype=np.float64)


def _parse_times(texts: list[str]) -> NDArray[np.datetime64]:
    """ISO texts to seconds; a text that is not a date (a deliberate defect) becomes missing."""
    try:
        return np.array(texts, dtype="datetime64[s]")
    except ValueError:
        return np.array([_parse_time(t) for t in texts], dtype="datetime64[s]")


def _parse_time(text: str) -> np.datetime64:
    try:
        return np.datetime64(text, "s")
    except ValueError:
        return np.datetime64("NaT", "s")


def to_output(values: Values, kind: ExprType) -> Values:
    """The evaluator's representation back to engine values."""
    if kind is ExprType.BOOLEAN:
        return values.astype(np.int64)
    if kind is ExprType.TEXT:
        return values
    missing = np.isnan(values)
    if kind is ExprType.NUMBER:
        if not missing.any():
            return values
        out = values.astype(object)
        out[missing] = None
        return out
    stamps = np.where(missing, 0, np.floor(values)).astype(np.int64).astype("datetime64[s]")
    if kind is ExprType.DATE:
        text = np.datetime_as_string(stamps.astype("datetime64[D]"), unit="D")
    else:
        text = np.char.replace(np.datetime_as_string(stamps, unit="s"), "T", " ")
    result = text.astype(object)
    result[missing] = None
    return result


class _Evaluator:
    """Walks the tree once; column conversions are cached per name."""

    def __init__(self, columns: Mapping[str, Column], types: Mapping[str, ExprType], n: int):
        self._columns = columns
        self._types = types
        self._n = n
        self._cache: dict[str, Values] = {}

    def run(self, expr: Expr) -> Values:
        if isinstance(expr, Literal):
            return self._literal(expr.value)
        if isinstance(expr, Ref):
            if expr.name not in self._cache:
                kind, values = self._columns[expr.name]
                self._cache[expr.name] = to_internal(values, kind)
            return self._cache[expr.name]
        if isinstance(expr, Unary):
            inner = self.run(expr.operand)
            return -inner if expr.op == "-" else ~inner
        if isinstance(expr, Binary):
            return self._binary(expr)
        if isinstance(expr, Conditional):
            return np.where(self.run(expr.condition), self.run(expr.then), self.run(expr.other))
        return self._call(expr)

    def _literal(self, value: float | str | bool) -> Values:
        if isinstance(value, bool):
            return np.full(self._n, value, dtype=bool)
        if isinstance(value, str):
            return np.full(self._n, value, dtype=object)
        return np.full(self._n, value, dtype=np.float64)

    def _binary(self, expr: Binary) -> Values:
        left, right = self.run(expr.left), self.run(expr.right)
        if expr.op == "and":
            return left & right
        if expr.op == "or":
            return left | right
        if expr.op in ARITHMETIC:
            with np.errstate(all="ignore"):
                out = _ARITHMETIC[expr.op](left, right)
            if expr.op in ("/", "%"):
                out[right == 0] = np.nan
            return np.asarray(out, dtype=np.float64)
        present = _present(left) & _present(right)
        result = np.zeros(self._n, dtype=bool)
        result[present] = _COMPARE[expr.op](left[present], right[present]).astype(bool)
        return result

    def _call(self, expr: Call) -> Values:
        args = expr.args
        if expr.name in ("min", "max"):
            stacked = np.stack([self.run(a) for a in args])
            extreme = stacked.min(axis=0) if expr.name == "min" else stacked.max(axis=0)
            return np.asarray(extreme, dtype=np.float64)
        if expr.name == "round":
            digits = int(_number(args[1])) if len(args) > 1 else 0
            return np.round(self.run(args[0]), digits)
        if expr.name == "date_add":
            seconds = UNITS[str(_text(args[2]))]
            return np.asarray(self.run(args[0]) + self.run(args[1]) * seconds, dtype=np.float64)
        if expr.name == "date_diff":
            return (self.run(args[0]) - self.run(args[1])) / UNITS[str(_text(args[2]))]
        return self._concat(args)

    def _concat(self, args: tuple[Expr, ...]) -> Values:
        parts = [_as_text(self.run(a), type_of(a, self._types)) for a in args]
        out = np.empty(self._n, dtype=object)
        for row in range(self._n):
            pieces = [p[row] for p in parts]
            out[row] = None if any(p is None for p in pieces) else "".join(pieces)
        return out


def _present(values: Values) -> NDArray[np.bool_]:
    if values.dtype == object:
        return np.array([v is not None for v in values], dtype=bool)
    if values.dtype == bool:
        return np.ones(values.size, dtype=bool)
    return ~np.isnan(values)


def _as_text(values: Values, kind: ExprType) -> Values:
    if kind is ExprType.NUMBER:
        return np.array(
            [
                None if np.isnan(v) else (str(int(v)) if v.is_integer() else repr(float(v)))
                for v in values
            ],
            dtype=object,
        )
    if kind is ExprType.BOOLEAN:
        return np.where(values, "true", "false").astype(object)
    return to_output(values, kind)


def _number(expr: Expr) -> float:
    if not isinstance(expr, Literal) or isinstance(expr.value, (str, bool)):
        raise ExpressionError("expected a number literal")
    return expr.value


def _text(expr: Expr) -> str:
    if not isinstance(expr, Literal) or not isinstance(expr.value, str):
        raise ExpressionError("expected a text literal")
    return expr.value
