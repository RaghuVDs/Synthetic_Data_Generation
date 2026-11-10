from __future__ import annotations

import pytest

from synthgen.domain.errors import DependencyCycleError, UnknownIdentifierError
from synthgen.domain.metadata import Relationship, RelationshipOrigin
from synthgen.metadata.graph import build_table_set


def rel(child: str, parent: str, *, nullable: bool = False) -> Relationship:
    child_table, child_column = child.split(".")
    parent_table, parent_column = parent.split(".")
    return Relationship(
        child_table=child_table,
        child_column=child_column,
        parent_table=parent_table,
        parent_column=parent_column,
        nullable=nullable,
        drives_cardinality=False,
        origin=RelationshipOrigin.METADATA,
    )


FINANCE = [
    rel("accounts.customer_id", "customers.id"),
    rel("cards.account_id", "accounts.id"),
    rel("transactions.card_id", "cards.id"),
    rel("transactions.merchant_id", "merchants.id"),
]
TABLES = ["accounts", "cards", "customers", "merchants", "transactions"]


def test_parents_are_added_as_support_in_order() -> None:
    table_set = build_table_set(["transactions"], TABLES, FINANCE)
    assert table_set.selected == ("transactions",)
    assert table_set.order == ("customers", "accounts", "cards", "merchants", "transactions")
    assert table_set.support == ("customers", "accounts", "cards", "merchants")
    assert len(table_set.relationships) == 4
    assert table_set.deferred == ()


def test_children_are_not_added() -> None:
    table_set = build_table_set(["customers"], TABLES, FINANCE)
    assert table_set.order == ("customers",)
    assert table_set.relationships == ()


def test_order_is_independent_of_input_order() -> None:
    forward = build_table_set(["transactions", "customers"], TABLES, FINANCE)
    backward = build_table_set(["customers", "transactions"], TABLES, list(reversed(FINANCE)))
    assert forward.order == backward.order


def test_unknown_table_is_rejected() -> None:
    with pytest.raises(UnknownIdentifierError):
        build_table_set(["nope"], TABLES, FINANCE)


def test_self_reference_is_deferred() -> None:
    manager = rel("employees.manager_id", "employees.id", nullable=True)
    table_set = build_table_set(["employees"], ["employees"], [manager])
    assert table_set.order == ("employees",)
    assert table_set.deferred == (manager,)


def test_mutual_reference_breaks_at_nullable_key() -> None:
    head = rel("departments.head_id", "employees.id", nullable=True)
    member = rel("employees.department_id", "departments.id")
    table_set = build_table_set(["employees"], ["departments", "employees"], [head, member])
    assert table_set.order == ("departments", "employees")
    assert table_set.deferred == (head,)


def test_longer_cycle_breaks_at_nullable_key() -> None:
    edges = [rel("b.a_id", "a.id"), rel("c.b_id", "b.id"), rel("a.c_id", "c.id", nullable=True)]
    table_set = build_table_set(["a"], ["a", "b", "c"], edges)
    assert table_set.order == ("a", "b", "c")
    assert [r.label() for r in table_set.deferred] == ["a.c_id -> c.id"]


def test_cycle_without_nullable_key_fails() -> None:
    edges = [rel("b.a_id", "a.id"), rel("a.b_id", "b.id")]
    with pytest.raises(DependencyCycleError):
        build_table_set(["a"], ["a", "b"], edges)


def test_two_keys_to_the_same_parent() -> None:
    edges = [rel("transfers.from_id", "accounts.id"), rel("transfers.to_id", "accounts.id")]
    table_set = build_table_set(["transfers"], ["accounts", "transfers"], edges)
    assert table_set.order == ("accounts", "transfers")
