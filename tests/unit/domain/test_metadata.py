from __future__ import annotations

import pytest
from pydantic import ValidationError

from synthgen.domain.metadata import (
    ColumnMeta,
    Relationship,
    RelationshipOrigin,
    Sensitivity,
    TableMeta,
)
from synthgen.domain.query import AggExpr, AggFunc, Condition, Filter, Operator


def _column(name: str, *, pk: bool = False) -> ColumnMeta:
    return ColumnMeta(name=name, declared_type="INTEGER", nullable=not pk, primary_key=pk)


def test_table_primary_key_and_column_lookup() -> None:
    table = TableMeta(name="t", columns=(_column("id", pk=True), _column("x")))
    assert table.primary_key == ("id",)
    assert table.column("x") is not None
    assert table.column("missing") is None


def test_table_needs_columns() -> None:
    with pytest.raises(ValidationError):
        TableMeta(name="t", columns=())


def test_sensitivity_parses_from_text() -> None:
    column = ColumnMeta.model_validate(
        {
            "name": "email",
            "declared_type": "TEXT",
            "nullable": False,
            "primary_key": False,
            "sensitivity": "sensitive",
        }
    )
    assert column.sensitivity is Sensitivity.SENSITIVE


def test_relationship_label_and_self_reference() -> None:
    rel = Relationship(
        child_table="employees",
        child_column="manager_id",
        parent_table="employees",
        parent_column="id",
        nullable=True,
        drives_cardinality=False,
        origin=RelationshipOrigin.PRAGMA,
    )
    assert rel.is_self_reference
    assert rel.label() == "employees.manager_id -> employees.id"


def test_count_star_needs_no_column_but_others_do() -> None:
    assert AggExpr(func=AggFunc.COUNT, column=None, alias="n").column is None
    with pytest.raises(ValidationError):
        AggExpr(func=AggFunc.SUM, column=None, alias="s")


def test_alias_must_be_identifier() -> None:
    with pytest.raises(ValidationError):
        AggExpr(func=AggFunc.COUNT, column=None, alias="n; DROP TABLE x")


@pytest.mark.parametrize(
    ("op", "value"),
    [
        (Operator.IS_NULL, 1),
        (Operator.IN, ()),
        (Operator.IN, 1),
        (Operator.EQ, None),
        (Operator.EQ, (1, 2)),
    ],
)
def test_condition_value_must_match_operator(op: Operator, value: object) -> None:
    with pytest.raises(ValidationError):
        Condition.model_validate({"column": "c", "op": op, "value": value})


def test_valid_conditions() -> None:
    where = Filter(
        conditions=(
            Condition(column="a", op=Operator.GE, value=3),
            Condition(column="b", op=Operator.IN, value=("x", "y")),
            Condition(column="c", op=Operator.NOT_NULL),
        )
    )
    assert len(where.conditions) == 3


def test_filter_needs_a_condition() -> None:
    with pytest.raises(ValidationError):
        Filter(conditions=())
