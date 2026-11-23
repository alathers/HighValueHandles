"""Enrichment phase of a batch: web research, then the fact check, with model swaps.

Runs on the main thread after the vision pass, the quantity count and the
hint overrides, and before interactive review. For each bottle dict it adds:

- ``web_research``: what the web layer did (``WebResearch.to_dict()``), with
  every consulted page's final outcome (``cited`` with the fields it
  supports, or why it was not cited);
- ``facts``: the :class:`~models.facts.FactSheet` (``to_dict()``);
- the flat bottle fields (``name``, ``distillery``, ``proof``, ``msrp``...)
  derived from the facts: a value only when the fact is ``supported`` or
  ``user``, else None. ``name`` keeps the label reading when its fact has no
  value, for folder naming only. ``msrp`` only ever comes from the facts, so
  a price seen on a label or a shop page never becomes an MSRP;
- ``vendor_image`` (``-v``): the accepted vendor reference image, if any.

Model residency: research (which may call the vision model to compare a
vendor image) runs first, then the vision model is unloaded, then the text
model checks each bottle serially, then the text model is unloaded. With
``keep_models_loaded`` nothing is unloaded.

Nothing here ever raises out of :meth:`Enricher.enrich_batch`: a failed web
lookup becomes ``web_research.status = 'failed'`` and a failed fact check a
deterministic sheet with ``verification.status = 'failed'`` (FR-035).

The helpers :func:`apply_fact_values`, :func:`user_facts_from_corrections`,
:func:`refresh_facts` and :func:`approve_facts` are shared with the CLI's
review, ``feedback`` and ``retry`` paths.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from image_processor.fact_checker import FactChecker
from image_processor.fact_prompt import PROMPT_VERSION, display_value, number_fields
from image_processor.fact_reconcile import apply_gate, rereconcile
from image_processor.fact_verify import canonical_value, render_description
from image_processor.preview_generator import PreviewGenerator
from models.evidence import PageRecord, WebResearch
from models.facts import FACT_FIELDS, FactInput, FactSheet, FieldFact, UserFact, Verification

logger = logging.getLogger(__name__)

#: Label-reading fields passed to the fact check (``ExtractedMetadata`` attributes).
LABEL_KEYS: tuple[str, ...] = (
    "name",
    "distillery",
    "specific_release",
    "batch_number",
    "proof",
    "abv",
    "age",
    "tasting_notes",
    "label_price",
)
#: Label fields that identify a bottle for the web search (hint values apply).
IDENTITY_KEYS: tuple[str, ...] = ("name", "distillery", "specific_release", "batch_number", "proof", "age")
#: Correction sources that are statements by a person (they become user facts).
USER_FACT_SOURCE_PREFIXES: tuple[str, ...] = ("hint:", "feedback:")
USER_FACT_SOURCE_EXACT = "interactive"
#: Metadata warnings that mean the photos may not all show one bottle.
_GROUP_WARNING_RE = re.compile(r"\bsplit\b|different products|\bmixed\b", re.IGNORECASE)
_SENT_AS_RE = re.compile(r"sent to fact check as (W\d+)$|^(W\d+): ")
#: Reason recorded when no web researcher was built.
WEB_DISABLED_REASON = "web lookups disabled"


def utc_now_iso() -> str:
    """Current UTC time, ISO-8601 with offset."""
    return datetime.now(UTC).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# Flat values
# ---------------------------------------------------------------------------


def format_flat_value(field_name: str, value: Any) -> str | None:  # noqa: ANN401
    """The bottle-dict form of a fact value (``proof 90.0 -> '90 Proof'``, ``abv -> '45%'``).

    Returns None for None. Numbers are rendered without trailing zeros;
    ``msrp`` is a plain amount (``'59.99'``) so ``parse_money`` reads it back.
    """
    if value is None:
        return None
    if field_name == "msrp":
        return display_value("msrp", value).lstrip("$")
    text = display_value(field_name, value)
    if field_name == "proof":
        return f"{text} Proof"
    if field_name == "abv":
        return f"{text}%"
    return text


def apply_fact_values(bottle_dict: dict, sheet: FactSheet) -> None:
    """Set the flat bottle fields from ``sheet`` (in place).

    Every fact field gets the fact's value when it is ``supported`` or
    ``user``, else None. ``name`` keeps its current value (the label reading)
    when the fact has none, so the output folder still gets a name; the
    facts record that the name is not supported. ``msrp`` comes only from
    ``facts.msrp``.
    """
    flat = sheet.flat_values()
    for name in FACT_FIELDS:
        value = format_flat_value(name, flat.get(name))
        if name == "name" and value is None:
            continue
        bottle_dict[name] = value


# ---------------------------------------------------------------------------
# User facts
# ---------------------------------------------------------------------------


def is_user_source(source: Any) -> bool:  # noqa: ANN401
    """Whether a correction source is a person's statement (hint, interactive, feedback)."""
    return isinstance(source, str) and (
        source == USER_FACT_SOURCE_EXACT or source.startswith(USER_FACT_SOURCE_PREFIXES)
    )


def user_facts_from_corrections(corrections: Iterable[Any]) -> list[UserFact]:
    """User facts from a bottle's recorded corrections (dicts or ``Correction`` objects).

    Only fact fields and person sources count. A null-like value
    (``'unknown'``, ``'n/a'``, empty) is not a statement of fact and is
    skipped with a warning, so it can never become a high-confidence value.
    """
    out: list[UserFact] = []
    for item in corrections or []:
        if isinstance(item, Mapping):
            field_name, value, source = item.get("field_name"), item.get("corrected_value"), item.get("source")
        else:
            field_name = getattr(item, "field_name", None)
            value = getattr(item, "corrected_value", None)
            source = getattr(item, "source", None)
        if field_name not in FACT_FIELDS or not is_user_source(source):
            continue
        if value is None or canonical_value(str(field_name), value) is None:
            logger.warning(
                "Correction of %s to %r (%s) is not a usable value; it is not used as a fact",
                field_name, value, source,
            )
            continue
        out.append(UserFact(field=str(field_name), value=str(value), ref=str(source)))
    return out


# ---------------------------------------------------------------------------
# Web research page outcomes
# ---------------------------------------------------------------------------


def _cited_fields(sheet: FactSheet) -> dict[str, list[str]]:
    """Web source id -> fields (in FACT_FIELDS order) whose supported/user value cites it."""
    out: dict[str, list[str]] = {}
    for name in FACT_FIELDS:
        fact = sheet.fields.get(name)
        if fact is None or fact.status not in ("supported", "user") or fact.value is None:
            continue
        for cit in fact.citations:
            if cit.source_id.startswith("W"):
                fields = out.setdefault(cit.source_id, [])
                if name not in fields:
                    fields.append(name)
    return out


def _not_cited_reason(source_id: str, sheet: FactSheet | None) -> str:
    """Why an evidence page sent to the fact check supports no field."""
    if sheet is None:
        return f"sent to fact check as {source_id}; no fact sheet was built"
    v = sheet.verification
    if v.status == "skipped":
        return (
            f"not used: the fact check did not run ({v.reason or v.status}); "
            "web text never fills a field without it"
        )
    if v.status == "failed":
        return f"not used: the fact check failed ({v.reason or 'no reason recorded'})"
    reasons: list[str] = []
    for rec in v.rejected:
        if rec.get("source_id") != source_id:
            continue
        if rec.get("stage") == "pack":
            return f"not shown to the text model: {rec.get('detail') or 'over the evidence budget'}"
        reason = str(rec.get("reason") or "rejected")
        if rec.get("kind") == "page_match" and str(rec.get("value")) == "different_product":
            reason = "not about this product"
        if reason not in reasons:
            reasons.append(reason)
    prices = sorted({p.kind for p in sheet.prices if p.citation.source_id == source_id and p.kind != "msrp"})
    parts = []
    if reasons:
        parts.append(f"no verifiable quote accepted ({', '.join(reasons)})")
    if prices:
        parts.append(f"only a {'/'.join(prices)} price observation")
    if not parts:
        parts.append("no verified fact cites it")
    return f"{source_id}: " + "; ".join(parts)


def update_page_outcomes(web_research: WebResearch, sheet: FactSheet | None) -> WebResearch:
    """Rewrite the evidence pages' outcomes after the fact check (returns a new record).

    A page whose excerpt a supported or user field cites becomes ``cited``
    with ``cited_for``; every other evidence page stays ``fetched_not_cited``
    with the reason. Pages that were never evidence keep their record.
    Idempotent: safe to call again after a re-reconcile.
    """
    by_url = {ev.url: ev.source_id for ev in web_research.evidence}
    cited = _cited_fields(sheet) if sheet is not None else {}
    pages: list[PageRecord] = []
    for page in web_research.pages:
        sid = None
        match = _SENT_AS_RE.search(page.reason or "")
        if match:
            sid = match.group(1) or match.group(2)
        elif page.url in by_url and page.outcome != "skipped":
            sid = by_url[page.url]
        if sid is None or sid not in {ev.source_id for ev in web_research.evidence}:
            pages.append(page)
            continue
        fields = cited.get(sid, [])
        if fields:
            new = PageRecord(
                url=page.url, title=page.title, domain=page.domain, source_class=page.source_class,
                retrieved_at=page.retrieved_at, outcome="cited", cited_for=list(fields),
                reason=None, from_cache=page.from_cache,
            )
        else:
            new = PageRecord(
                url=page.url, title=page.title, domain=page.domain, source_class=page.source_class,
                retrieved_at=page.retrieved_at, outcome="fetched_not_cited", cited_for=[],
                reason=_not_cited_reason(sid, sheet), from_cache=page.from_cache,
            )
        pages.append(new)
    return WebResearch(
        status=web_research.status,
        reason=web_research.reason,
        provider=web_research.provider,
        queries=list(web_research.queries),
        searched_at=web_research.searched_at,
        pages=pages,
        evidence=list(web_research.evidence),
        vendor_image=web_research.vendor_image,
    )


def log_page_outcomes(bottle_id: str, web_research: WebResearch) -> None:
    """Log every consulted URL with its final outcome (INFO; never a key: URLs come from results)."""
    for page in web_research.pages:
        detail = f"cited for {', '.join(page.cited_for)}" if page.outcome == "cited" else page.reason
        logger.info("Bottle %s web page %s: %s%s", bottle_id, page.url, page.outcome, f" ({detail})" if detail else "")


# ---------------------------------------------------------------------------
# Review helpers (interactive, feedback)
# ---------------------------------------------------------------------------


def approve_facts(sheet: FactSheet, by: str, at: str | None = None) -> FactSheet:
    """Record a reviewer's approval on ``sheet`` and recompute the publish gate."""
    sheet.review = {"approved": True, "by": by, "at": at or utc_now_iso()}
    return apply_gate(sheet)


def renumber_copy(sheet: FactSheet, numbers_before: Mapping[int, str], values_before: Mapping[str, Any]) -> None:
    """Keep the description's ``[n]`` markers pointing at the same fields (in place).

    The markers index :func:`fact_prompt.number_fields` (supported and user
    fields in ``FACT_FIELDS`` order). A re-reconcile can add or remove a
    field from that list (e.g. a settled proof conflict), which shifts the
    numbers, so ``[3]`` would silently point at another field. Each cited
    number is mapped back to its field and renumbered; if a cited field
    lost its value or the value changed, the copy is marked stale instead
    (it no longer says what the sources support). The gate is recomputed
    when anything changed.
    """
    copy = sheet.copy
    if copy is None or copy.stale:
        return
    numbers_after = {n.field: n.index for n in number_fields(sheet)}
    values_after = sheet.flat_values()
    sentences: list[dict[str, Any]] = []
    changed = False
    for sentence in copy.sentences:
        cites: list[int] = []
        for raw in sentence.get("citations") or []:
            try:
                index = int(raw)
            except (TypeError, ValueError):
                index = -1
            field_name = numbers_before.get(index)
            if (
                field_name is None
                or field_name not in numbers_after
                or values_after.get(field_name) != values_before.get(field_name)
            ):
                copy.stale = True
                copy.reason = f"citation [{raw}] no longer points at the value the description was written from"
                apply_gate(sheet)
                return
            cites.append(numbers_after[field_name])
            changed = changed or numbers_after[field_name] != index
        sentences.append({**sentence, "citations": cites})
    if changed:
        copy.sentences = sentences
        copy.description_md = render_description(sentences)
        apply_gate(sheet)


def refresh_facts(bottle_dict: dict) -> FactSheet | None:
    """Fold the bottle's user corrections into its saved facts (no model call).

    Re-reconciles ``bottle_dict['facts']`` with every user fact from
    ``bottle_dict['corrections']`` (already-recorded ones are not added
    twice), re-derives the flat fields and the web pages' outcomes. If a
    value changed after the bottle was approved, the approval is withdrawn:
    the reviewer approved different values. The copy is marked stale when a
    field it cites changed (see :func:`fact_reconcile.rereconcile`).

    Returns:
        The new sheet, or None when the bottle has no facts (nothing changes).
    """
    raw = bottle_dict.get("facts")
    if not raw:
        return None
    sheet = raw if isinstance(raw, FactSheet) else FactSheet.from_dict(raw)
    before = sheet.flat_values()
    numbers_before = {n.index: n.field for n in number_fields(sheet)}
    new = rereconcile(sheet, user_facts_from_corrections(bottle_dict.get("corrections") or []))
    renumber_copy(new, numbers_before, before)
    if new.review.get("approved") and new.flat_values() != before:
        logger.warning(
            "Bottle %s: values changed after it was approved; the approval is withdrawn",
            bottle_dict.get("bottle_id", "?"),
        )
        new.review = {"approved": False, "by": None, "at": None}
        new = apply_gate(new)
    bottle_dict["facts"] = new.to_dict()
    apply_fact_values(bottle_dict, new)
    wr = bottle_dict.get("web_research")
    if wr:
        try:
            research = wr if isinstance(wr, WebResearch) else WebResearch.from_dict(wr)
            bottle_dict["web_research"] = update_page_outcomes(research, new).to_dict()
        except (ValueError, TypeError, KeyError):
            logger.warning("Bottle %s: web research record could not be updated", bottle_dict.get("bottle_id", "?"))
    return new


# ---------------------------------------------------------------------------
# Enricher
# ---------------------------------------------------------------------------


def group_warnings_of(metadata: Any) -> list[str]:  # noqa: ANN401
    """Split or mixed-group warnings (each blocks publishing)."""
    return [str(w) for w in getattr(metadata, "warnings", None) or [] if _GROUP_WARNING_RE.search(str(w))]


def _label_from(bottle_dict: Mapping[str, Any], metadata: Any) -> dict[str, Any]:  # noqa: ANN401
    """The label reading: metadata fields (the vision answer), else the dict's."""
    label: dict[str, Any] = {}
    for key in LABEL_KEYS:
        value = getattr(metadata, key, None) if metadata is not None else bottle_dict.get(key)
        if value not in (None, ""):
            label[key] = value
    return label


def _hinted_identity(bottle_dict: Mapping[str, Any], metadata: Any) -> dict[str, Any]:  # noqa: ANN401
    """Search identity: label fields with hint values applied, plus vision confidence."""
    identity = {k: v for k, v in _label_from(bottle_dict, metadata).items() if k in IDENTITY_KEYS}
    for item in bottle_dict.get("corrections") or []:
        if isinstance(item, Mapping) and str(item.get("source") or "").startswith("hint:"):
            name = item.get("field_name")
            if name in IDENTITY_KEYS and item.get("corrected_value") not in (None, ""):
                identity[name] = item["corrected_value"]
    identity["vision_confidence"] = getattr(metadata, "vision_confidence", None) or bottle_dict.get(
        "vision_confidence"
    )
    return identity


def _fallback_sheet(reason: str, model: str | None, group_warnings: Sequence[str]) -> FactSheet:
    """A sheet with every field Unknown (used when even the deterministic check fails)."""
    sheet = FactSheet(
        fields={f: FieldFact.unknown("fact sheet could not be built") for f in FACT_FIELDS},
        sources=[],
        conflicts=[],
        prices=[],
        verification=Verification(
            status="failed", reason=reason, text_model=model, think=None, prompt_version=PROMPT_VERSION
        ),
        copy=None,
    )
    return apply_gate(sheet, list(group_warnings))


class Enricher:
    """Web research plus fact check for the bottles of one batch.

    Args:
        researcher: :class:`image_processor.web_research.WebResearcher`, or
            None (every bottle's web research is ``skipped``).
        fact_checker: The run's :class:`FactChecker` (deterministic when its
            ``llm`` is None).
        server_manager: The model server manager; its ``unload_model`` is
            used when ``unload`` is not given.
        vision_model: Vision model name (unloaded before the text phase).
        text_model: Text model name, or None when the fact check is off.
        keep_models_loaded: Skip both unloads.
        workers: Parallel web lookups (``--concurrency``).
        unload: ``unload(model_name)``; failures are logged and ignored.
        web_skip_reason: Recorded as the web research reason when there is
            no researcher.
    """

    def __init__(
        self,
        *,
        researcher: Any,  # noqa: ANN401
        fact_checker: FactChecker,
        server_manager: Any = None,  # noqa: ANN401
        vision_model: str,
        text_model: str | None,
        keep_models_loaded: bool,
        workers: int = 1,
        unload: Callable[[str], Any] | None = None,
        web_skip_reason: str | None = None,
    ) -> None:
        self.researcher = researcher
        self.fact_checker = fact_checker
        self.vision_model = vision_model
        self.text_model = text_model
        self.keep_models_loaded = bool(keep_models_loaded)
        self.workers = max(1, int(workers or 1))
        if unload is None:
            unload = getattr(server_manager, "unload_model", None) or (lambda name: None)
        self.unload = unload
        self.web_skip_reason = web_skip_reason or WEB_DISABLED_REASON

    @property
    def text_active(self) -> bool:
        """Whether the text model runs (a text model and an LLM client)."""
        return self.text_model is not None and getattr(self.fact_checker, "llm", None) is not None

    @property
    def provider_name(self) -> str | None:
        """The search provider's name, or None without a researcher."""
        provider = getattr(self.researcher, "provider", None)
        name = getattr(provider, "name", None)
        return name if isinstance(name, str) else None

    def _unload(self, model: str | None, phase: str) -> None:
        if not model:
            return
        try:
            self.unload(model)
            logger.info("Unloaded %s model %s", phase, model)
        except Exception as e:  # unloading is best-effort
            logger.warning("Could not unload %s model %s: %s", phase, model, e)

    def _research(self, items: list[tuple[dict[str, Any], Path | None]]) -> list[WebResearch]:
        if self.researcher is None:
            return [WebResearch(status="skipped", reason=self.web_skip_reason) for _ in items]
        try:
            results = list(self.researcher.research_many(items, workers=self.workers))
        except Exception as e:
            logger.exception("Web research failed for the batch")
            reason = f"web research error: {type(e).__name__}: {e}"[:300]
            return [WebResearch(status="failed", reason=reason) for _ in items]
        if len(results) != len(items):
            logger.error("Web research returned %d results for %d bottles", len(results), len(items))
            return [WebResearch(status="failed", reason="web research returned a wrong number of results")
                    for _ in items]
        return results

    def _fact_input(
        self, bottle_dict: dict, metadata: Any, bottle_id: str, research: WebResearch  # noqa: ANN401
    ) -> FactInput:
        images = [Path(str(p)).name for p in bottle_dict.get("images") or getattr(metadata, "source_images", []) or []]
        ocr = getattr(metadata, "ocr_by_image", None) or {}
        return FactInput(
            bottle_id=bottle_id,
            label=_label_from(bottle_dict, metadata),
            vision_confidence=getattr(metadata, "vision_confidence", None),
            images=images,
            ocr_by_image={Path(str(k)).name: str(v) for k, v in dict(ocr).items()},
            user_facts=tuple(user_facts_from_corrections(bottle_dict.get("corrections") or [])),
            web=tuple(research.evidence),
            group_warnings=tuple(group_warnings_of(metadata)),
        )

    def _check(self, inp: FactInput | None, bottle_id: str, metadata: Any) -> FactSheet:  # noqa: ANN401
        model = self.text_model
        if inp is None:
            return _fallback_sheet("fact-check input could not be built", model, group_warnings_of(metadata))
        try:
            return self.fact_checker.check(inp)
        except Exception as e:
            logger.exception("Fact check raised for bottle %s", bottle_id)
            reason = f"fact check error: {type(e).__name__}: {e}"[:300]
            try:
                return FactChecker.deterministic(inp, reason, status="failed", model=model)
            except Exception:
                logger.exception("Deterministic fact sheet failed for bottle %s", bottle_id)
                return _fallback_sheet(reason, model, inp.group_warnings)

    def enrich_batch(self, bottle_dicts: list[dict], metadata_list: list, *, run_id: str) -> None:
        """Research and fact-check every bottle (dicts are changed in place). Never raises."""
        if not bottle_dicts:
            return
        metas = [metadata_list[i] if i < len(metadata_list) else None for i in range(len(bottle_dicts))]
        ids: list[str] = []
        items: list[tuple[dict[str, Any], Path | None]] = []
        for bottle_dict, metadata in zip(bottle_dicts, metas, strict=True):
            bottle_dict["run_id"] = run_id
            try:
                bottle_id = PreviewGenerator.bottle_id_for(bottle_dict, metadata)
            except Exception:
                bottle_id = "unknown"
            ids.append(bottle_id)
            images = bottle_dict.get("images") or []
            try:
                items.append((_hinted_identity(bottle_dict, metadata), Path(str(images[0])) if images else None))
            except Exception:
                logger.exception("Could not build the search identity of bottle %s", bottle_id)
                items.append(({}, None))

        # 1. Web research (may call the vision model for vendor images).
        researches = self._research(items)

        # 2. Free the vision model's memory before the text model loads.
        if self.text_active and not self.keep_models_loaded:
            self._unload(self.vision_model, "vision")

        # 3. Fact check, one bottle at a time.
        for bottle_dict, metadata, bottle_id, fetched in zip(bottle_dicts, metas, ids, researches, strict=True):
            research = fetched
            try:
                inp = self._fact_input(bottle_dict, metadata, bottle_id, research)
            except Exception:
                logger.exception("Could not build the fact-check input of bottle %s", bottle_id)
                inp = None
            sheet = self._check(inp, bottle_id, metadata)
            try:
                research = update_page_outcomes(research, sheet)
            except Exception:
                logger.exception("Could not record the web page outcomes of bottle %s", bottle_id)
            bottle_dict["web_research"] = research.to_dict()
            bottle_dict["facts"] = sheet.to_dict()
            apply_fact_values(bottle_dict, sheet)
            if research.vendor_image is not None:
                bottle_dict["vendor_image"] = research.vendor_image.to_dict()
            log_page_outcomes(bottle_id, research)
            logger.info(
                "Bottle %s: web %s (%d pages consulted, %d cited); fact check %s; %s",
                bottle_id, research.status, research.pages_consulted, research.pages_cited,
                sheet.verification.status,
                "publish-ready" if sheet.publish.get("ready") else
                f"{len(sheet.publish.get('blockers') or [])} publish blocker(s)",
            )

        # 4. Free the text model.
        if self.text_active and not self.keep_models_loaded:
            self._unload(self.text_model, "text")
