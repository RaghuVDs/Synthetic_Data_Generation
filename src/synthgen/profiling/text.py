"""Text statistics (SPEC 8.3 D): lengths, masks, character classes, case, words, validity, format.

`text_shape` is value-free and is all a sensitive column keeps. `text_format` describes fixed
parts of the values and is only computed for non-sensitive columns.
"""

from __future__ import annotations

import re
import string
from collections import Counter
from collections.abc import Callable, Sequence
from datetime import datetime

import numpy as np

from synthgen.profiling.inference import CARD_RE, EMAIL_RE, PHONE_RE, luhn_valid, mask, mask_shares
from synthgen.profiling.models import Measured, StatSource, TextShape
from synthgen.profiling.shape_models import Spread, TextDetails, TextFormat

MASK_LIMIT = 20
LENGTH_PERCENTILES = (5.0, 50.0, 95.0)
FIXED_POSITION_SHARE = 0.99
MAX_FIXED_POSITIONS = 20
CHAR_CLASSES: dict[str, Callable[[str], bool]] = {
    "letter": str.isalpha,
    "digit": str.isdigit,
    "space": str.isspace,
    "punctuation": lambda c: c in string.punctuation,
}


def text_shape(texts: Sequence[str]) -> TextShape:
    """Lengths, masks and value-free details of non-null values."""
    lengths = [len(t) for t in texts] or [0]
    sample = StatSource.SAMPLE
    return TextShape(
        length_min=Measured[int](value=min(lengths), source=sample),
        length_max=Measured[int](value=max(lengths), source=sample),
        length_mean=Measured[float](value=sum(lengths) / len(lengths), source=sample),
        masks=Measured(value=mask_shares(texts, MASK_LIMIT), source=sample),
        details=Measured[TextDetails](value=text_details(texts), source=sample),
    )


def text_details(texts: Sequence[str]) -> TextDetails:
    """Character classes, case styles, padding, word counts and format validity."""
    lengths = np.array([len(t) for t in texts] or [0], dtype=np.float64)
    words = np.array([len(t.split()) for t in texts] or [0], dtype=np.float64)
    p50, p90, p99 = (float(v) for v in np.percentile(words, (50, 90, 99)))
    count = max(1, len(texts))
    return TextDetails(
        length_percentiles=[
            (p, float(v))
            for p, v in zip(
                LENGTH_PERCENTILES, np.percentile(lengths, LENGTH_PERCENTILES), strict=True
            )
        ],
        char_classes=_char_classes(texts),
        case_styles=[
            (style, n / count) for style, n in Counter(_case(t) for t in texts).most_common()
        ],
        leading_space_rate=sum(t[:1].isspace() for t in texts) / count,
        trailing_space_rate=sum(t[-1:].isspace() for t in texts) / count,
        word_count=Spread(p50=p50, p90=p90, p99=p99, max=float(np.max(words))),
        validity=validity(texts),
    )


def _char_classes(texts: Sequence[str]) -> list[tuple[str, float]]:
    counts: Counter[str] = Counter()
    for text in texts:
        for char in text:
            kind = next((name for name, test in CHAR_CLASSES.items() if test(char)), "other")
            counts[kind] += 1
    total = sum(counts.values()) or 1
    return [(kind, counts[kind] / total) for kind in (*CHAR_CLASSES, "other")]


def _case(text: str) -> str:
    if not any(c.isalpha() for c in text):
        return "none"
    if text.islower():
        return "lower"
    if text.isupper():
        return "upper"
    return "title" if text.istitle() else "mixed"


ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}([ T]\d{2}:\d{2}(:\d{2}(\.\d+)?)?)?$")


def _is_date(text: str) -> bool:
    """An ISO date or date-time that names a real calendar day."""
    if not ISO_DATE_RE.match(text):
        return False
    try:
        datetime.fromisoformat(text)
    except ValueError:
        return False
    return True


VALIDATORS: dict[str, Callable[[str], bool]] = {
    "email": lambda t: bool(EMAIL_RE.match(t)),
    "card_number_luhn": lambda t: bool(CARD_RE.match(t)) and luhn_valid(t),
    "phone": lambda t: bool(PHONE_RE.match(t)),
    "iso_date": _is_date,
}


def validity(texts: Sequence[str]) -> list[tuple[str, float]]:
    """Pass rate of each known format that at least some values follow."""
    if not texts:
        return []
    rates = [(name, sum(map(check, texts)) / len(texts)) for name, check in VALIDATORS.items()]
    return [(name, rate) for name, rate in rates if rate > 0]


def common_prefix(texts: Sequence[str]) -> str:
    """The longest prefix every string shares, compared character by character."""
    shortest = min(texts, key=len, default="")
    for i, char in enumerate(shortest):
        if any(t[i] != char for t in texts):
            return shortest[:i]
    return shortest


def text_format(texts: Sequence[str]) -> TextFormat:
    """Prefix and suffix shared by every value, and fixed positions of the dominant mask."""
    if not texts:
        return TextFormat(common_prefix="", common_suffix="", fixed_positions=[])
    prefix = common_prefix(texts)
    suffix = common_prefix([t[::-1] for t in texts])[::-1]
    dominant = Counter(mask(t) for t in texts).most_common(1)[0][0]
    same = [t for t in texts if mask(t) == dominant]
    fixed: list[tuple[int, str]] = []
    for position in range(len(dominant)):
        char, n = Counter(t[position] for t in same).most_common(1)[0]
        if n >= FIXED_POSITION_SHARE * len(same) and len(same) > 1:
            fixed.append((position, char))
    return TextFormat(
        common_prefix=prefix, common_suffix=suffix, fixed_positions=fixed[:MAX_FIXED_POSITIONS]
    )
