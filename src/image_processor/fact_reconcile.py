"""Deterministic reconciliation, confidence and the publish gate (pure).

Given the label reading, OCR text, user facts and the claims that survived
``fact_verify``, decide each field's value and status, list conflicts for a
human (code never settles one), record every price with its kind, assign
confidence by fixed rules (a model's self-report is never used) and compute
``publish.ready`` with the blockers.
"""

from __future__ import annotations

import copy as _copy
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

from image_processor.fact_prompt import PROMPT_VERSION, EvidencePack, display_value
from image_processor.fact_verify import (
    Accepted,
    AcceptedPrice,
    age_value_months,
    batch_tokens,
    canonical_value,
    normalize,
    ocr_backing,
    significant_tokens,
)
from models.bottle import parse_money
from models.facts import (
    FACT_FIELDS,
    Candidate,
    Citation,
    Conflict,
    FactInput,
    FactSheet,
    FieldFact,
    PriceObservation,
    Source,
    UserFact,
    Verification,
)

#: Label-reading keys that map to fact fields (``label_price`` never does).
LABEL_FIELDS: tuple[str, ...] = tuple(f for f in FACT_FIELDS if f != "msrp")
#: Fields the vision prompt reads from the label.
VISION_READ_FIELDS: frozenset[str] = frozenset(
    {"name", "distillery", "specific_release", "batch_number", "proof", "abv", "age", "tasting_notes"}
)
#: Text fields compared by token sets.
TEXT_FIELDS: frozenset[str] = frozenset(
    {"name", "specific_release", "distillery", "bottler", "mash_bill", "tasting_notes"}
)
PROOF_TOLERANCE = Decimal("0.1")
ABV_TOLERANCE = Decimal("0.05")
PROOF_ABV_TOLERANCE = Decimal("0.2")
PROOF_ABV_REASON = "proof_abv_inconsistent"
_ORIGIN_ORDER = {"label": 0, "ocr": 1, "web": 2, "user": 3}
_LEVELS = ("low", "medium", "high")


# ---------------------------------------------------------------------------
# Agreement rules
# ---------------------------------------------------------------------------


def _dec(value: Any) -> Decimal | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return Decimal(repr(value)) if isinstance(value, float) else Decimal(str(value))
    except Exception:
        return None


def _text_agree(a: Any, b: Any) -> bool:
    ta, tb = set(significant_tokens(a)), set(significant_tokens(b))
    if not ta or not tb:
        return normalize(a) == normalize(b)
    if ta <= tb or tb <= ta:
        return True
    if not ta & tb:
        return False
    na = {t for t in ta if any(ch.isdigit() for ch in t)}
    nb = {t for t in tb if any(ch.isdigit() for ch in t)}
    return na == nb


def values_agree(field_name: str, a: Any, b: Any) -> bool:
    """Whether two candidate values of ``field_name`` agree (conflict rules)."""
    if field_name in ("proof", "abv", "msrp"):
        da, db = _dec(a), _dec(b)
        if da is None or db is None:
            return normalize(a) == normalize(b)
        diff = abs(da - db)
        if field_name == "proof":
            return diff <= PROOF_TOLERANCE
        if field_name == "abv":
            return diff <= ABV_TOLERANCE
        return diff <= max(Decimal(1), Decimal("0.02") * max(da, db))
    if field_name == "age":
        ma, mb = age_value_months(a), age_value_months(b)
        if ma is None or mb is None:
            return normalize(a) == normalize(b)
        return ma == mb
    if field_name == "release_year":
        return _dec(a) == _dec(b)
    if field_name == "batch_number":
        return batch_tokens(a) == batch_tokens(b)
    return _text_agree(a, b)


def corroborates(field_name: str, chosen: Any, other: Any) -> bool:
    """Whether ``other`` backs the chosen value (not merely fails to conflict).

    Text fields need equal or nested token sets: "Elijah Craig Small Batch"
    does not conflict with "Elijah Craig Barrel Proof" (shared tokens), but it
    is not evidence for it either, so it adds no citation or confidence.
    Other fields use :func:`values_agree`.
    """
    if field_name in TEXT_FIELDS:
        ta, tb = set(significant_tokens(chosen)), set(significant_tokens(other))
        if not ta or not tb:
            return normalize(chosen) == normalize(other)
        return ta <= tb or tb <= ta
    return values_agree(field_name, chosen, other)


# ---------------------------------------------------------------------------
# Context
# ---------------------------------------------------------------------------


@dataclass
class ReconcileContext:
    """What the confidence rules and Unknown reasons need to know."""

    vision_confidence: str | None = None
    tiers: dict[str, str | None] = field(default_factory=dict)
    domains: dict[str, str] = field(default_factory=dict)
    web_pages: int = 0
    ocr_sources: int = 0
    text_status: str = "verified"
    text_reason: str | None = None
    rejected: list[dict[str, Any]] = field(default_factory=list)
    label_keys: frozenset[str] = frozenset()
    label_price: Decimal | None = None
    retail_prices: list[PriceObservation] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Sources and candidates
# ---------------------------------------------------------------------------


def build_sources(inp: FactInput, pack: EvidencePack) -> list[Source]:
    """L1, O*, W* sources (every fetched page, also those over budget)."""
    sources = [Source(id="L1", kind="label", ref=", ".join(pack.images) or "(no photos)")]
    for sid in pack.ocr_ids:
        src = pack.sources[sid]
        sources.append(Source(id=sid, kind="ocr", ref=src.image or "", content_sha256=src.sha256))
    seen = {s.id for s in sources}
    for ev in inp.web:
        if ev.source_id in seen:
            continue
        seen.add(ev.source_id)
        sources.append(
            Source(
                id=ev.source_id,
                kind="web",
                ref=ev.url,
                url=ev.url,
                title=ev.title,
                tier=ev.tier,
                fetched_at=ev.fetched_at,
                content_sha256=ev.sha256,
                evidence_path=f"evidence/{ev.source_id}.txt",
            )
        )
    return sources


def _add(cands: dict[str, list[Candidate]], field_name: str, value: Any, citation: Citation, origin: str) -> None:
    """Add a citation to an identical (same origin, same value) candidate or a new one."""
    bucket = cands.setdefault(field_name, [])
    for cand in bucket:
        if cand.origin == origin and normalize(display_value(field_name, cand.value)) == normalize(
            display_value(field_name, value)
        ):
            if citation not in cand.citations:
                cand.citations.append(citation)
            return
    bucket.append(Candidate(value=value, citations=[citation], origin=origin))  # type: ignore[arg-type]


def label_candidates(inp: FactInput, pack: EvidencePack) -> dict[str, list[Candidate]]:
    """Vision readings (cited as ``L1`` readings) plus OCR backing lines."""
    cands: dict[str, list[Candidate]] = {}
    ocr = pack.ocr_full_texts()
    for name in LABEL_FIELDS:
        raw = inp.label.get(name)
        value = canonical_value(name, raw)
        if value is None:
            continue
        reading = re.sub(r"\s+", " ", str(raw)).strip()
        citations = [Citation("L1", reading, "reading")]
        backing = ocr_backing(raw, name, ocr)
        if backing is not None:
            citations.append(backing)
        cands[name] = [Candidate(value=value, citations=citations, origin="label")]
    return cands


def claim_candidates(accepted: Iterable[Accepted], pack: EvidencePack) -> dict[str, list[Candidate]]:
    """Accepted OCR and web claims."""
    cands: dict[str, list[Candidate]] = {}
    for acc in accepted:
        src = pack.get(acc.citation.source_id)
        origin = "web" if src is not None and src.kind == "web" else "ocr"
        _add(cands, acc.claim.field, acc.value, acc.citation, origin)
    return cands


def user_candidates(
    user_facts: Iterable[UserFact], start: int = 1
) -> tuple[dict[str, list[Candidate]], list[Source]]:
    """User facts as ``U<n>`` sources and candidates (citation quote = the ref)."""
    cands: dict[str, list[Candidate]] = {}
    sources: list[Source] = []
    n = start
    for uf in user_facts:
        value = canonical_value(uf.field, uf.value)
        if value is None:
            value = uf.value
        sid = f"U{n}"
        n += 1
        sources.append(Source(id=sid, kind="user", ref=uf.ref))
        cands.setdefault(uf.field, []).append(
            Candidate(value=value, citations=[Citation(sid, uf.ref, "reading")], origin="user")
        )
    return cands, sources


def price_observations(
    inp: FactInput, accepted_prices: Iterable[AcceptedPrice] = ()
) -> list[PriceObservation]:
    """Label price tag (retail), JSON-LD offers (retail) and accepted model prices."""
    out: list[PriceObservation] = []
    seen: set[tuple[str, Decimal, str]] = set()

    def add(obs: PriceObservation) -> None:
        key = (obs.kind, obs.amount, obs.citation.source_id)
        if key not in seen:
            seen.add(key)
            out.append(obs)

    label_price = parse_money(inp.label.get("label_price"))
    if label_price is not None and label_price > 0:
        add(PriceObservation("retail", label_price, "USD", Citation("L1", "price visible on label", "reading")))
    for ev in inp.web:
        for sp in ev.structured_prices:
            add(
                PriceObservation(
                    "retail",
                    sp.amount,
                    sp.currency,
                    Citation(ev.source_id, f"structured data: offers.price {sp.amount} {sp.currency}", "reading"),
                )
            )
    for ap in accepted_prices:
        add(PriceObservation(ap.kind, ap.price.amount.quantize(Decimal("0.01")), ap.price.currency, ap.citation))  # type: ignore[arg-type]
    return out


def msrp_candidates(prices: Iterable[PriceObservation], ctx: ReconcileContext) -> list[Candidate]:
    """MSRP observations as candidates of the ``msrp`` field."""
    cands: dict[str, list[Candidate]] = {}
    for obs in prices:
        if obs.kind != "msrp":
            continue
        origin = "web" if obs.citation.source_id in ctx.tiers else "ocr"
        _add(cands, "msrp", obs.amount, obs.citation, origin)
    return cands.get("msrp", [])


# ---------------------------------------------------------------------------
# Confidence
# ---------------------------------------------------------------------------


def _web_ids(cands: Iterable[Candidate], ctx: ReconcileContext) -> list[str]:
    ids: list[str] = []
    for cand in cands:
        if cand.origin != "web":
            continue
        for cit in cand.citations:
            if cit.source_id in ctx.tiers and cit.source_id not in ids:
                ids.append(cit.source_id)
    return ids


def _web_level(ids: Sequence[str], ctx: ReconcileContext) -> tuple[str, str]:
    domains = sorted({ctx.domains.get(i, i) for i in ids})
    official = {ctx.domains.get(i, i) for i in ids if ctx.tiers.get(i) == "official"}
    if len(domains) >= 2 and official:
        level = "high"
    elif official or len(domains) >= 2:
        level = "medium"
    else:
        level = "low"
    tiers = "+".join(f"{d}({'official' if d in official else 'other'})" for d in domains)
    return level, f"web[{tiers}]"


def confidence_for(field_name: str, cands: Sequence[Candidate], ctx: ReconcileContext) -> tuple[str, str]:
    """(confidence, basis) of agreeing non-user candidates, by the fixed rules."""
    label = next((c for c in cands if c.origin == "label"), None)
    web_ids = _web_ids(cands, ctx)
    ocr_claim = any(c.origin == "ocr" for c in cands)
    if field_name == "msrp":
        if not web_ids:
            return "low", "ocr" if ocr_claim else "none"
        level, basis = _web_level(web_ids, ctx)
        return level, f"msrp:{basis}"
    if label is not None:
        vc = ctx.vision_confidence or "unknown"
        backed = ocr_claim or any(c.source_id.startswith("O") for c in label.citations)
        if backed:
            return "high", f"label({vc})+ocr"
        if web_ids:
            _, basis = _web_level(web_ids, ctx)
            return "high", f"label({vc})+{basis}"
        return ("medium" if vc == "high" else "low"), f"label({vc})"
    if web_ids:
        level, basis = _web_level(web_ids, ctx)
        if ocr_claim and level == "low":
            return "medium", f"{basis}+ocr"
        return level, basis + ("+ocr" if ocr_claim else "")
    if ocr_claim:
        return "low", "ocr"
    return "low", "none"


# ---------------------------------------------------------------------------
# Field resolution
# ---------------------------------------------------------------------------


def _describe(cand: Candidate, field_name: str) -> str:
    ids = ",".join(dict.fromkeys(c.source_id for c in cand.citations)) or cand.origin
    return f"{ids} {display_value(field_name, cand.value)}"


def _sorted(cands: Iterable[Candidate]) -> list[Candidate]:
    return sorted(cands, key=lambda c: _ORIGIN_ORDER.get(c.origin, 9))


def _all_agree(field_name: str, cands: Sequence[Candidate]) -> bool:
    return all(
        values_agree(field_name, a.value, b.value) for i, a in enumerate(cands) for b in cands[i + 1 :]
    )


def _agreed_value(cands: Sequence[Candidate]) -> Any:
    label = next((c for c in cands if c.origin == "label"), None)
    if label is not None:
        return label.value
    best = max(cands, key=lambda c: (len(c.citations), -_ORIGIN_ORDER.get(c.origin, 9)))
    return best.value


def _citations(cands: Iterable[Candidate]) -> list[Citation]:
    out: list[Citation] = []
    for cand in cands:
        for cit in cand.citations:
            if cit not in out:
                out.append(cit)
    return out


def _rejected_summary(field_name: str, ctx: ReconcileContext) -> str:
    items = [
        r for r in ctx.rejected
        if r.get("kind") in ("claim", "price") and r.get("field") == field_name
    ]
    if not items:
        return ""
    shown = ", ".join(f"{r.get('source_id')} {r.get('reason')}" for r in items[:5])
    more = f" and {len(items) - 5} more" if len(items) > 5 else ""
    return f"{len(items)} claim(s) rejected ({shown}{more})"


def _text_note(ctx: ReconcileContext) -> str | None:
    if ctx.text_status in ("skipped", "failed"):
        why = f": {ctx.text_reason}" if ctx.text_reason else ""
        return f"OCR and web text not checked (text verification {ctx.text_status}{why})"
    return None


def unknown_reason(field_name: str, ctx: ReconcileContext) -> str:
    """Why a field is Unknown, built from what was tried."""
    label = field_name.replace("_", " ")
    if field_name == "msrp":
        parts = []
        if ctx.label_price is not None:
            parts.append(f"price on label (${ctx.label_price:.2f}) recorded as retail, not MSRP")
        else:
            parts.append("MSRP not printed on label")
        parts.append(f"{ctx.web_pages} page(s) fetched" if ctx.web_pages else "no web pages fetched")
        retail = [p for p in ctx.retail_prices if p.kind == "retail" and p.citation.source_id != "L1"]
        if retail:
            listed = ", ".join(f"{p.citation.source_id} ${p.amount:.2f}" for p in retail[:6])
            parts.append(f"{len(retail)} retail price(s) found ({listed})")
        rej = _rejected_summary("msrp", ctx)
        if rej:
            parts.append(rej.replace("claim(s)", "MSRP claim(s)"))
        note = _text_note(ctx)
        if note:
            parts.append(note)
        else:
            parts.append("no source stated an MSRP")
        return "; ".join(parts)
    parts = []
    if field_name in ctx.label_keys or field_name in VISION_READ_FIELDS:
        parts.append(f"{label.capitalize()} not read from label")
    else:
        parts.append(f"{label.capitalize()} not part of the label reading")
    note = _text_note(ctx)
    if note:
        parts.append(note)
    else:
        where = []
        if ctx.ocr_sources:
            where.append("OCR text")
        where.append(f"{ctx.web_pages} fetched page(s)" if ctx.web_pages else "no fetched pages")
        parts.append("not stated in " + " or ".join(where))
    rej = _rejected_summary(field_name, ctx)
    if rej:
        parts.append(rej)
    return "; ".join(parts)


def resolve_field(
    field_name: str,
    cands: Sequence[Candidate],
    ctx: ReconcileContext,
    *,
    conflict_no: int = 1,
    ignore: Sequence[Candidate] = (),
) -> tuple[FieldFact, list[Conflict]]:
    """Decide one field.

    Args:
        field_name: Field name.
        cands: Every candidate (all kept on the FieldFact).
        ctx: Context for confidence and Unknown reasons.
        conflict_no: Number for the conflict id.
        ignore: Candidates kept for review but left out of the decision
            (a label "distillery" that is really the bottler).

    Returns:
        The FieldFact and any conflict on it.
    """
    cands = _sorted(cands)
    users = [c for c in cands if c.origin == "user"]
    others = [c for c in cands if c.origin != "user" and c not in ignore]
    cid = f"{field_name}-{conflict_no}"
    if users:
        chosen = users[-1]
        disagree = [c for c in others if not values_agree(field_name, c.value, chosen.value)]
        conflicts = []
        if disagree:
            note = "; ".join(_describe(c, field_name) for c in disagree)
            conflicts.append(
                Conflict(
                    id=cid,
                    field=field_name,
                    kind="value_mismatch",
                    candidates=[*disagree, chosen],
                    note=f"{note} vs user {display_value(field_name, chosen.value)} (user value wins)",
                    resolved_by=chosen.citations[0],
                )
            )
        return (
            FieldFact(
                value=chosen.value,
                status="user",
                confidence="high",
                confidence_basis="user",
                citations=list(chosen.citations),
                candidates=list(cands),
                reason=None,
            ),
            conflicts,
        )
    if not others:
        return FieldFact.unknown(unknown_reason(field_name, ctx), list(cands)), []
    if not _all_agree(field_name, others):
        note = " vs ".join(_describe(c, field_name) for c in others)
        conflict = Conflict(id=cid, field=field_name, kind="value_mismatch", candidates=list(others), note=note)
        return (
            FieldFact(value=None, status="conflict", candidates=list(cands), reason=f"sources disagree: {note}"),
            [conflict],
        )
    value = _agreed_value(others)
    support = [c for c in others if corroborates(field_name, value, c.value)]
    level, basis = confidence_for(field_name, support, ctx)
    return (
        FieldFact(
            value=value,
            status="supported",
            confidence=level,  # type: ignore[arg-type]
            confidence_basis=basis,
            citations=_citations(support),
            candidates=list(cands),
            reason=None,
        ),
        [],
    )


def _role_ignored(dist_cands: Sequence[Candidate], bottler_cands: Sequence[Candidate]) -> list[Candidate]:
    """Label distillery candidates that are really the bottler.

    Vision mixes brand, bottler and distiller. When the label "distillery"
    disagrees with a verified distillery claim but agrees with a verified
    bottler claim, it is left out of the distillery decision.
    """
    label = [c for c in dist_cands if c.origin == "label"]
    claimed = [c for c in dist_cands if c.origin in ("web", "ocr")]
    bottlers = [c for c in bottler_cands if c.origin in ("web", "ocr", "user")]
    if not label or not claimed or not bottlers:
        return []
    out = []
    for cand in label:
        clashes = any(not values_agree("distillery", cand.value, c.value) for c in claimed)
        is_bottler = any(values_agree("bottler", cand.value, b.value) for b in bottlers)
        if clashes and is_bottler:
            out.append(cand)
    return out


def _check_proof_abv(fields: dict[str, FieldFact], conflicts: list[Conflict], ctx: ReconcileContext | None) -> None:
    """Flag ``|proof - 2*abv| > 0.2`` as ``proof_abv_inconsistent``.

    Non-user fields involved become ``conflict`` (value None). They keep
    their confidence and basis (what the agreed value would have) so a later
    re-reconcile can restore them once the inconsistency is gone.
    """
    conflicts[:] = [c for c in conflicts if c.kind != PROOF_ABV_REASON]
    proof, abv = fields.get("proof"), fields.get("abv")
    if proof is None or abv is None:
        return
    for fact in (proof, abv):
        if fact.status == "conflict" and (fact.reason or "").startswith(PROOF_ABV_REASON):
            agreed = [c for c in fact.candidates if c.origin != "user"]
            fact.value = _agreed_value(agreed) if agreed else None
            fact.status = "supported" if fact.value is not None else "unknown"
            fact.citations = _citations(agreed)
            fact.reason = None
    pv, av = _dec(proof.value), _dec(abv.value)
    if pv is None or av is None or abs(pv - 2 * av) <= PROOF_ABV_TOLERANCE:
        return
    pc = Candidate(value=proof.value, citations=list(proof.citations), origin=_origin(proof))
    ac = Candidate(value=abv.value, citations=list(abv.citations), origin=_origin(abv))
    note = f"proof {display_value('proof', proof.value)} vs abv {display_value('abv', abv.value)} (2 x abv = {2 * av})"
    conflicts.append(Conflict(id="proof_abv-1", field="proof/abv", kind=PROOF_ABV_REASON, candidates=[pc, ac], note=note))
    for fact in (proof, abv):
        if fact.status == "supported":
            fact.value = None
            fact.status = "conflict"
            fact.citations = []
            fact.reason = f"{PROOF_ABV_REASON}: {note}"


def _origin(fact: FieldFact) -> str:
    if fact.status == "user":
        return "user"
    srcs = {c.source_id[:1] for c in fact.citations}
    if "L" in srcs:
        return "label"
    if "W" in srcs:
        return "web"
    return "ocr"


def resolve_all(
    cands: Mapping[str, Sequence[Candidate]], ctx: ReconcileContext
) -> tuple[dict[str, FieldFact], list[Conflict]]:
    """Resolve every fact field, then the proof/ABV consistency check."""
    fields: dict[str, FieldFact] = {}
    conflicts: list[Conflict] = []
    ignore = _role_ignored(cands.get("distillery", []), cands.get("bottler", []))
    for name in FACT_FIELDS:
        fact, found = resolve_field(name, cands.get(name, []), ctx, ignore=ignore if name == "distillery" else ())
        fields[name] = fact
        conflicts.extend(found)
    _check_proof_abv(fields, conflicts, ctx)
    return fields, conflicts


# ---------------------------------------------------------------------------
# Publish gate
# ---------------------------------------------------------------------------

CORE_FIELD_GROUPS: tuple[tuple[str, ...], ...] = (("name",), ("distillery", "bottler"), ("proof", "abv"))


def _core_ok(fact: FieldFact | None) -> bool:
    return (
        fact is not None
        and fact.status in ("supported", "user")
        and fact.value is not None
        and fact.confidence in ("high", "medium")
    )


def compute_gate(sheet: FactSheet, group_warnings: Sequence[str] | None = None) -> dict[str, Any]:
    """``{ready, blockers, group_warnings}`` for the sheet.

    Args:
        sheet: The fact sheet.
        group_warnings: Split/mixed-group warnings; None keeps the ones the
            sheet's ``publish`` already records.
    """
    warnings = list(
        group_warnings if group_warnings is not None else sheet.publish.get("group_warnings") or []
    )
    blockers: list[str] = []
    v = sheet.verification
    if v.status != "verified":
        blockers.append(f"text verification {v.status}" + (f": {v.reason}" if v.reason else ""))
    for c in sheet.conflicts:
        if c.resolved_by is None:
            blockers.append(f"unresolved conflict {c.id} ({c.field}): {c.note}")
    for name in FACT_FIELDS:
        fact = sheet.fields.get(name)
        if fact is not None and fact.value is not None and fact.confidence == "low":
            blockers.append(f"low confidence: {name} ({fact.confidence_basis})")
    for group in CORE_FIELD_GROUPS:
        if not any(_core_ok(sheet.fields.get(n)) for n in group):
            blockers.append(f"core field {' or '.join(group)} not supported at medium confidence or better")
    for w in warnings:
        blockers.append(f"group warning: {w}")
    if sheet.copy is None:
        blockers.append("no verified description")
    elif sheet.copy.stale:
        blockers.append("description is stale" + (f": {sheet.copy.reason}" if sheet.copy.reason else ""))
    if not sheet.review.get("approved"):
        blockers.append("not approved by a reviewer")
    return {"ready": not blockers, "blockers": blockers, "group_warnings": warnings}


def apply_gate(sheet: FactSheet, group_warnings: Sequence[str] | None = None) -> FactSheet:
    """Set ``sheet.publish`` from :func:`compute_gate`; returns the sheet."""
    sheet.publish = compute_gate(sheet, group_warnings)
    return sheet


# ---------------------------------------------------------------------------
# Whole sheet
# ---------------------------------------------------------------------------


def make_context(
    inp: FactInput, pack: EvidencePack, verification: Verification, prices: Sequence[PriceObservation]
) -> ReconcileContext:
    """Context for :func:`reconcile`."""
    label_price = parse_money(inp.label.get("label_price"))
    return ReconcileContext(
        vision_confidence=inp.vision_confidence,
        tiers={ev.source_id: ev.tier for ev in inp.web},
        domains={ev.source_id: ev.domain for ev in inp.web},
        web_pages=len(inp.web),
        ocr_sources=len(pack.ocr_ids),
        text_status=verification.status,
        text_reason=verification.reason,
        rejected=list(verification.rejected),
        label_keys=frozenset(k for k in inp.label.keys() if k in FACT_FIELDS),
        label_price=label_price if label_price and label_price > 0 else None,
        retail_prices=[p for p in prices if p.kind != "msrp"],
    )


def reconcile(
    inp: FactInput,
    pack: EvidencePack,
    verification: Verification,
    *,
    accepted: Iterable[Accepted] = (),
    accepted_prices: Iterable[AcceptedPrice] = (),
) -> FactSheet:
    """Build a FactSheet (no copy yet) from verified inputs; the gate is applied.

    Web values reach a field only through ``accepted`` claims and prices;
    pass none (deterministic mode) and web pages are listed as sources only.
    """
    sources = build_sources(inp, pack)
    prices = price_observations(inp, accepted_prices)
    ctx = make_context(inp, pack, verification, prices)
    cands: dict[str, list[Candidate]] = {k: list(v) for k, v in label_candidates(inp, pack).items()}
    for name, items in claim_candidates(accepted, pack).items():
        for cand in items:
            for cit in cand.citations:
                _add(cands, name, cand.value, cit, cand.origin)
    msrp = msrp_candidates(prices, ctx)
    if msrp:
        cands.setdefault("msrp", []).extend(msrp)
    ucands, usources = user_candidates(inp.user_facts)
    sources.extend(usources)
    for name, items in ucands.items():
        cands.setdefault(name, []).extend(items)
    fields, conflicts = resolve_all(cands, ctx)
    msrp_fact = fields["msrp"]
    if msrp_fact.status == "supported":
        for obs in prices:
            if obs.kind == "msrp":
                obs.confidence = msrp_fact.confidence
    sheet = FactSheet(
        fields=fields,
        sources=sources,
        conflicts=conflicts,
        prices=prices,
        verification=verification,
        copy=None,
    )
    return apply_gate(sheet, list(inp.group_warnings))


def empty_verification(status: str, reason: str | None, model: str | None = None) -> Verification:
    """A Verification record with no model work."""
    return Verification(
        status=status,  # type: ignore[arg-type]
        reason=reason,
        text_model=model,
        think=None,
        prompt_version=PROMPT_VERSION,
    )


def rereconcile(sheet: FactSheet, user_facts: Iterable[UserFact]) -> FactSheet:
    """Add user facts to a saved sheet without any model call.

    User candidates win their fields (a disagreement stays as a resolved
    conflict); the proof/ABV check and the gate are recomputed, and the copy
    is marked stale if a field it cites changed value. Returns a new sheet.
    """
    new = _copy.deepcopy(sheet)
    before = new.flat_values()
    existing = [s for s in new.sources if s.kind == "user"]
    start = 1 + max((int(s.id[1:]) for s in existing if s.id[1:].isdigit()), default=0)
    fresh: list[UserFact] = []
    for uf in user_facts:
        fact = new.fields.get(uf.field)
        dup = fact is not None and any(
            c.origin == "user"
            and values_agree(uf.field, c.value, canonical_value(uf.field, uf.value) or uf.value)
            and any(cit.quote == uf.ref for cit in c.citations)
            for c in fact.candidates
        )
        if not dup:
            fresh.append(uf)
    ucands, usources = user_candidates(fresh, start)
    new.sources.extend(usources)
    ctx = ReconcileContext()
    for name, items in ucands.items():
        fact = new.fields.get(name) or FieldFact.unknown("not stated")
        cands = list(fact.candidates) + items
        new.conflicts = [c for c in new.conflicts if c.field != name]
        resolved, found = resolve_field(name, cands, ctx)
        new.fields[name] = resolved
        new.conflicts.extend(found)
    _check_proof_abv(new.fields, new.conflicts, None)
    after = new.flat_values()
    changed = sorted(f for f in FACT_FIELDS if before.get(f) != after.get(f))
    if new.copy is not None and changed:
        hit = sorted(set(changed) & new.copy.cited_fields())
        if hit:
            new.copy.stale = True
            new.copy.reason = f"field(s) {', '.join(hit)} changed after the description was written"
    return apply_gate(new)
