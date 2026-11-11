from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from synthgen.config import ProfilingSettings
from synthgen.connectors.sqlite import SqliteConnector
from synthgen.metadata.provider import SourceDoc, YamlMetadataProvider
from synthgen.profiling.categorical import zipf_exponent
from synthgen.profiling.profiler import profile_table
from synthgen.profiling.text import common_prefix, text_details, text_format, validity

PROFILE_NOW = datetime(2026, 1, 1, tzinfo=UTC)


def test_text_details() -> None:
    details = text_details(["Hello world", "abc", " ABC1 ", "Title Case", "12-34"])
    classes = dict(details.char_classes)
    assert classes["letter"] > classes["digit"] > 0
    assert classes["space"] > 0 and classes["punctuation"] > 0
    assert sum(classes.values()) == pytest.approx(1.0)
    styles = dict(details.case_styles)
    assert styles == {"mixed": 0.2, "lower": 0.2, "upper": 0.2, "title": 0.2, "none": 0.2}
    assert details.leading_space_rate == details.trailing_space_rate == 0.2
    assert details.word_count.max == 2
    assert dict(details.length_percentiles)[50.0] == 6


def test_validity_rates() -> None:
    rates = dict(validity(["a@b.com", "bad", "c@d.org", "4111111111111111"]))
    assert rates["email"] == 0.5
    assert rates["card_number_luhn"] == 0.25
    assert "iso_date" not in rates
    assert dict(validity(["2024-01-01", "2024-02-30", "x"]))["iso_date"] == pytest.approx(1 / 3)
    assert validity([]) == []


def test_text_format() -> None:
    fmt = text_format(["ELE-10001", "ELE-20002", "ELE-30003"])
    assert fmt.common_prefix == "ELE-"
    assert (3, "-") in fmt.fixed_positions
    assert (0, "E") in fmt.fixed_positions
    assert text_format(["a1", "b2"]).common_suffix == ""
    assert text_format([]).fixed_positions == []


def test_common_prefix() -> None:
    assert common_prefix(["abc", "abd", "ab"]) == "ab"
    assert common_prefix(["x", "y"]) == ""
    assert common_prefix([]) == ""


def test_zipf_exponent() -> None:
    counts = [round(10_000 / rank) for rank in range(1, 200)]
    exponent = zipf_exponent(counts)
    assert exponent is not None and exponent == pytest.approx(1.0, abs=0.05)
    assert zipf_exponent([5, 4]) is None


def test_profiler_keeps_formats_out_of_sensitive_columns(
    shop_db: Path, profiling_settings: ProfilingSettings
) -> None:
    connector = SqliteConnector(shop_db)
    try:
        provider = YamlMetadataProvider(SourceDoc(source="shop"), connector)
        profile = profile_table(
            "shop",
            connector,
            provider.table("customers"),
            [],
            settings=profiling_settings,
            seed=1,
            now=PROFILE_NOW,
        )
    finally:
        connector.close()
    email = profile.column("email")
    tier = profile.column("tier")
    assert email is not None and email.text is not None and email.text_format is None
    assert dict(email.text.details.value.validity)["email"] == 1.0
    assert tier is not None and tier.text_format is not None
    assert tier.categorical is not None and tier.categorical.zipf_exponent is not None
