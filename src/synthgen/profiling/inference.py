"""Type, semantic type, sensitivity and mask inference from sampled values (SPEC 8.3 B, D)."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Callable, Sequence

from synthgen.domain.metadata import Sensitivity
from synthgen.profiling.models import InferredType, MaskShare

Value = str | int | float | bool | bytes

DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
DATETIME_RE = re.compile(r"^\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}(:\d{2}(\.\d+)?)?$")
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[A-Za-z]{2,}$")
PHONE_RE = re.compile(r"^\+?[\d\s().-]{7,}$")
CARD_RE = re.compile(r"^\d{13,19}$")
SEMANTIC_MATCH_SHARE = 0.9
"""Share of values that must match a format for the column to get that semantic type."""
SENSITIVE_SEMANTICS = frozenset({"email", "phone", "card_number"})
BOOLEAN_INTS = frozenset({0, 1})


def infer_type(values: Sequence[Value]) -> InferredType:
    """Real type of the non-null values of one column."""
    if not values:
        return InferredType.EMPTY
    if all(isinstance(v, bool) for v in values):
        return InferredType.BOOLEAN
    if all(isinstance(v, int) and not isinstance(v, bool) for v in values):
        return InferredType.BOOLEAN if set(values) == BOOLEAN_INTS else InferredType.INTEGER
    if all(isinstance(v, int | float) and not isinstance(v, bool) for v in values):
        return InferredType.FLOAT
    return _text_type(values)


def _text_type(values: Sequence[Value]) -> InferredType:
    """DATE or DATETIME when every value is ISO text of that form, otherwise TEXT."""
    if all(isinstance(v, str) for v in values):
        for pattern, kind in ((DATE_RE, InferredType.DATE), (DATETIME_RE, InferredType.DATETIME)):
            if all(pattern.match(str(v)) for v in values):
                return kind
    return InferredType.TEXT


def luhn_valid(number: str) -> bool:
    """True when a digit string passes the Luhn check."""
    total = 0
    for position, char in enumerate(reversed(number)):
        digit = int(char)
        if position % 2:
            digit = digit * 2 - 9 if digit > 4 else digit * 2  # noqa: PLR2004 - Luhn rule
        total += digit
    return total % 10 == 0


def semantic_type(values: Sequence[Value], inferred: InferredType) -> str | None:
    """A recognised format (email, phone, card_number, date) or None."""
    if inferred in (InferredType.DATE, InferredType.DATETIME):
        return inferred.value
    if inferred is not InferredType.TEXT or not values:
        return None
    texts = [str(v) for v in values]
    checks: dict[str, Callable[[str], bool]] = {
        "email": lambda t: bool(EMAIL_RE.match(t)),
        "card_number": lambda t: bool(CARD_RE.match(t)) and luhn_valid(t),
        "phone": lambda t: bool(PHONE_RE.match(t)) and sum(c.isdigit() for c in t) >= 7,  # noqa: PLR2004
    }
    for name, check in checks.items():
        if sum(check(t) for t in texts) >= SEMANTIC_MATCH_SHARE * len(texts):
            return name
    return None


def sensitivity_for(tag: Sensitivity | None, semantic: str | None) -> Sensitivity:
    """Metadata tags win; otherwise recognised personal formats are sensitive (SPEC 8.3)."""
    if tag is not None:
        return tag
    return Sensitivity.SENSITIVE if semantic in SENSITIVE_SEMANTICS else Sensitivity.PUBLIC


def mask(text: str) -> str:
    """Letters become `A`, digits `9`, everything else is kept."""
    return "".join("A" if c.isalpha() else "9" if c.isdigit() else c for c in text)


def mask_shares(texts: Sequence[str], limit: int) -> list[MaskShare]:
    """The most common masks and their shares of the given values."""
    if not texts:
        return []
    counts = Counter(mask(t) for t in texts)
    return [MaskShare(mask=m, share=n / len(texts)) for m, n in counts.most_common(limit)]
