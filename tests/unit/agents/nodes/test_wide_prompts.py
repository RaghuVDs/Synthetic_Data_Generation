from __future__ import annotations

from synthgen.agents.meaning import check_semantic_map
from synthgen.agents.nodes.common import render_cards
from synthgen.agents.nodes.semantic_mapper import mapper_batches, merge_maps
from synthgen.domain.semantic import SemanticMap
from synthgen.profiling.card import CardColumn, LlmSafeProfile, card_text


def column(name: str) -> CardColumn:
    return CardColumn(
        name=name,
        description="",
        type="float",
        semantic_type=None,
        sensitivity="public",
        null_rate=0.0,
        distinct_count=100,
        five_number=(0.0, 1.0, 2.0, 3.0, 4.0),
        date_range=None,
        top_values=None,
        mask=None,
        length_range=None,
    )


def card(table: str, width: int) -> LlmSafeProfile:
    names = ["id", *(f"m{i}" for i in range(width - 1))]
    return LlmSafeProfile(
        table=table,
        description="",
        row_count=10,
        relationships=[],
        columns=[column(n) for n in names],
    )


def test_batches_respect_both_limits_and_cover_every_column() -> None:
    cards = [card("wide", 600), card("a", 5), card("b", 5)]
    batches = mapper_batches(cards, max_tables=15, max_columns=250)
    assert all(sum(len(c.columns) for c in b) <= 250 for b in batches)
    assert all(len(b) <= 15 for b in batches)
    covered = [(c.table, col.name) for b in batches for c in b for col in c.columns]
    expected = [(c.table, col.name) for c in cards for col in c.columns]
    assert sorted(covered) == sorted(expected)
    assert len(covered) == len(set(covered))
    assert [len(b) for b in batches] == [1, 1, 3]  # 250, 250, then 100 + 5 + 5


def test_table_limit_still_applies() -> None:
    batches = mapper_batches([card(f"t{i}", 3) for i in range(5)], max_tables=2, max_columns=250)
    assert [len(b) for b in batches] == [2, 2, 1]


def slice_map(columns: list[str], key: str | None) -> SemanticMap:
    return SemanticMap.model_validate(
        {
            "tables": [
                {
                    "table": "wide",
                    "grain": "event",
                    "entity_key": key,
                    "event_time": None,
                    "label_column": None,
                    "columns": [{"name": c, "role": "measure", "note": None} for c in columns],
                }
            ],
            "business_rules": [{"rule": "same rule", "columns": [], "confidence": 0.5}],
        }
    )


def test_slices_merge_into_one_table() -> None:
    merged = merge_maps([slice_map(["id", "m0"], None), slice_map(["m1"], "id")])
    [table] = merged.tables
    assert [c.name for c in table.columns] == ["id", "m0", "m1"]
    assert table.entity_key == "id"
    assert len(merged.business_rules) == 1


def test_slice_may_name_a_key_from_another_slice() -> None:
    answer = slice_map(["m1"], "id")
    assert (
        check_semantic_map(answer, {"wide": ["m1"]}, key_columns={"wide": ["id", "m0", "m1"]}) == []
    )
    assert check_semantic_map(answer, {"wide": ["m1"]}) == ["unknown key column wide.id"]


def roles_map(width: int) -> SemanticMap:
    columns = [{"name": "id", "role": "identifier", "note": None}]
    columns += [{"name": f"m{i}", "role": "measure", "note": None} for i in range(width - 1)]
    return SemanticMap.model_validate(
        {
            "tables": [
                {
                    "table": "wide",
                    "grain": "entity",
                    "entity_key": "id",
                    "event_time": None,
                    "label_column": None,
                    "columns": columns,
                }
            ],
            "business_rules": [],
        }
    )


def test_wide_table_gets_a_summary_card() -> None:
    wide = card("wide", 3_000)
    text = render_cards([wide], roles_map(3_000), limit=150, focus={("wide", "m2500")})
    assert "- id float [identifier]" in text
    assert "- m2500 float [measure]" in text
    assert "- m10 float" not in text
    assert "measure columns (2998): m0, m1," in text
    assert len(text) < len(card_text(wide)) / 4


def test_narrow_table_keeps_its_full_card() -> None:
    narrow = card("narrow", 20)
    assert render_cards([narrow], None, limit=150, focus=set()) == card_text(narrow, {})
