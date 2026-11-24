"""Tolerant comparison of two bottle identities read by the vision model.

Used by the same-bottle backstop (:class:`BatchProcessor`): after a group the
model called mixed is split in two, each half is read again, and the split is
kept only if the halves *disagree* on a field both of them read. The halves
usually show different sides of the label (front vs. back), so a field missing
on one side is a wildcard, not a difference:

- **name**: equal after normalising (case, accents, punctuation, spacing), or
  a token-set similarity of at least :data:`NAME_MATCH_RATIO`. Numbers in the
  name conflict only when *each* name has a number the other lacks, so
  "Stagg 23A" vs "Stagg 23C" and "Old Fitzgerald 8" vs "Old Fitzgerald 10"
  stay different, while "1792 Full Proof" vs "1792 Full Proof 125" (a number
  seen on one view only) match.
- **specific_release**: compared like the name.
- **batch_number**: equal after dropping punctuation and filler words
  ("Batch 23A" == "23a").
- **proof**: equal within :data:`PROOF_TOLERANCE`.
- **age**: same number of years (months converted).

Limitations (both lean towards "same bottle", because a front/back pair torn
in two was measured to hurt more than an occasional merge):

- Sibling bottles of one product that show no distinguishing field (e.g. two
  Elmer T. Lee bottles, Small Batch A/B without visible batch codes) compare
  equal and are kept together; the backstop adds a review hint for them.
- A name whose words are a subset of the other's matches (token-set ratio
  1.0), so "Stagg" vs "Stagg Jr" or "Weller" vs "Weller Special Reserve"
  count as one product unless another field read on both sides (proof,
  batch, age, release) differs. The same subset match is what lets a back
  label's short "Blanton's" match the front's "Blanton's Single Barrel".

Standard library only (``difflib``), no extra dependency.
"""

from __future__ import annotations

import re
import unicodedata
from difflib import SequenceMatcher
from typing import Any

#: Token-set similarity (0-1) at or above which two names match.
NAME_MATCH_RATIO = 0.9
#: Proofs this close (in proof points) match.
PROOF_TOLERANCE = 0.5

_FILLER_WORDS = frozenset({"batch", "lot", "no", "number", "num", "barrel", "bbl"})
_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")


def normalize_text(value: Any) -> str:
    """Casefolded, accent-free text with punctuation turned into single spaces.

    Apostrophes are dropped rather than split ("Blanton's" -> "blantons").
    """
    if value is None:
        return ""
    text = unicodedata.normalize("NFKD", str(value))
    text = "".join(ch for ch in text if not unicodedata.combining(ch)).casefold()
    text = re.sub(r"['\u2019`]", "", text).replace("&", " and ")
    return " ".join(re.sub(r"[^0-9a-z]+", " ", text).split())


def token_set_ratio(a: str, b: str) -> float:
    """Token-set similarity of two normalised strings, 0.0-1.0.

    Same idea as the well-known ``token_set_ratio``: shared tokens are
    compared against each side's full sorted token set, so word order and a
    word present on only one side matter little.
    """
    tokens_a, tokens_b = set(a.split()), set(b.split())
    if not tokens_a or not tokens_b:
        return 0.0
    common = " ".join(sorted(tokens_a & tokens_b))
    with_a = " ".join(filter(None, [common, " ".join(sorted(tokens_a - tokens_b))]))
    with_b = " ".join(filter(None, [common, " ".join(sorted(tokens_b - tokens_a))]))

    def ratio(x: str, y: str) -> float:
        return SequenceMatcher(None, x, y).ratio() if x and y else 0.0

    return max(ratio(common, with_a), ratio(common, with_b), ratio(with_a, with_b))


def _number_tokens(text: str) -> set[str]:
    """Tokens that contain a digit ('23a', '10', '2024')."""
    return {token for token in text.split() if any(ch.isdigit() for ch in token)}


def names_match(a: Any, b: Any) -> bool:
    """True if two names (or release names) denote the same product.

    Empty values match anything (wildcard).
    """
    na, nb = normalize_text(a), normalize_text(b)
    if not na or not nb or na == nb:
        return True
    numbers_a, numbers_b = _number_tokens(na), _number_tokens(nb)
    # Only a number on each side that the other lacks is a difference; an
    # extra number on one view (a proof, an age) is not.
    if numbers_a - numbers_b and numbers_b - numbers_a:
        return False
    return token_set_ratio(na, nb) >= NAME_MATCH_RATIO


def _batch_key(value: Any) -> str:
    """Batch code without punctuation and filler words ('Batch No. 23-A' -> '23a')."""
    return "".join(t for t in normalize_text(value).split() if t not in _FILLER_WORDS)


def _first_number(value: Any) -> float | None:
    """First number in a value ('114.2 Proof' -> 114.2), else None."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    match = _NUMBER_RE.search(str(value or ""))
    return float(match.group()) if match else None


def _age_years(value: Any) -> float | None:
    """Age statement in years ('8 Year Old' -> 8.0, '96 months' -> 8.0), else None."""
    number = _first_number(value)
    if number is None:
        return None
    return number / 12 if "month" in normalize_text(value) else number


def identity_conflicts(a: Any, b: Any) -> list[str]:
    """Fields that both identities state and that disagree.

    Args:
        a: First identity (an ExtractedMetadata or anything with its fields).
        b: Second identity.

    Returns:
        Names of the conflicting fields, empty if the two may be one bottle.
    """
    conflicts = [
        field
        for field in ("name", "specific_release")
        if not names_match(getattr(a, field, None), getattr(b, field, None))
    ]
    batch_a, batch_b = _batch_key(getattr(a, "batch_number", None)), _batch_key(getattr(b, "batch_number", None))
    if batch_a and batch_b and batch_a != batch_b:
        conflicts.append("batch_number")
    proof_a, proof_b = _first_number(getattr(a, "proof", None)), _first_number(getattr(b, "proof", None))
    if proof_a is not None and proof_b is not None and abs(proof_a - proof_b) > PROOF_TOLERANCE:
        conflicts.append("proof")
    age_a, age_b = _age_years(getattr(a, "age", None)), _age_years(getattr(b, "age", None))
    if age_a is not None and age_b is not None and abs(age_a - age_b) > 0.01:  # noqa: PLR2004
        conflicts.append("age")
    return conflicts
