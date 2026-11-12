"""Which kind of fake personal value suits a sensitive column, from its format or its name.

The format comes from the profile (email, phone, card number); the name is split into words
(`first_name`, `FirstName`, `dob`). Columns matching nothing are generated from their masks.
"""

from __future__ import annotations

import re

from synthgen.spec.generators import SemanticKind

K = SemanticKind
_BY_FORMAT = {"email": K.EMAIL, "phone": K.PHONE, "card_number": K.CARD_NUMBER}
_BY_WORDS: tuple[tuple[frozenset[str], SemanticKind], ...] = (
    (frozenset({"birth", "dob", "birthdate", "birthday"}), K.DATE_OF_BIRTH),
    (frozenset({"first", "given", "forename", "firstname"}), K.FIRST_NAME),
    (frozenset({"last", "family", "surname", "lastname"}), K.LAST_NAME),
    (frozenset({"email", "mail"}), K.EMAIL),
    (frozenset({"phone", "mobile", "telephone", "tel"}), K.PHONE),
    (frozenset({"address", "street"}), K.STREET_ADDRESS),
    (frozenset({"city", "town"}), K.CITY),
    (frozenset({"postcode", "zip", "postal", "zipcode"}), K.POSTCODE),
    (frozenset({"company", "employer"}), K.COMPANY),
    (frozenset({"name", "fullname"}), K.FULL_NAME),
)


def _words(name: str) -> set[str]:
    spaced = re.sub(r"([a-z])([A-Z])", r"\1 \2", name)
    return {w for w in re.split(r"[^a-z]+", spaced.lower()) if w}


def semantic_kind(column: str, semantic_type: str | None) -> SemanticKind | None:
    """The kind for a sensitive column, or None when neither format nor name tells."""
    if semantic_type in _BY_FORMAT:
        return _BY_FORMAT[semantic_type]
    words = _words(column)
    for markers, kind in _BY_WORDS:
        if words & markers:
            if kind is K.DATE_OF_BIRTH and semantic_type not in ("date", "datetime", None):
                continue
            return kind
    return None
