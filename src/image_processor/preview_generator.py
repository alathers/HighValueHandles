"""Generate preview reports and output files for bourbon bottles.

Creates the bottle documents (Markdown, JSON, OCR text, images, web
evidence excerpts), batch previews, per-bottle failure reports, and the
interactive review.

The bottle Markdown (:func:`render_bottle_markdown`) is built from the
bottle's fact sheet: a publish-status banner, what needs review, a facts
table with confidence and sources, prices by kind, the verified description,
numbered sources, the verification record, references and the full web
research log. Every piece of untrusted text (page titles, quotes, model
output, user input) is escaped before it is written.
"""

import json
import logging
import os
import re
import shutil
import traceback
import unicodedata
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
from urllib.parse import urlsplit

from image_processor.image_organizer import OUTPUT_MARKER
from lib.error_handling import create_error_report
from lib.exceptions import FileSystemError
from models.bottle import Bottle, ExtractedMetadata, format_money, parse_money
from models.evidence import VENDOR_IMAGE_USAGE, WebResearch
from models.facts import FACT_FIELDS, FactSheet
from models.image import ImageCategory

logger = logging.getLogger(__name__)

#: Version of the per-bottle JSON, batch JSON and failure-report layouts.
#: 2: facts, web_research, vendor_image, run_id, reference_links always present.
OUTPUT_SCHEMA_VERSION = 2

#: Interactive review actions.
REVIEW_ACTIONS = ("A", "R", "E", "C", "N", "Q")
REVIEW_ACTIONS_LINE = (
    "[A]ccept (approve for publishing) | [R]eject | [E]dit | [C]onflicts | "
    "[N]ext (keep, not approved) | [Q]uit"
)
#: Characters escaped in untrusted text written to Markdown.
_MD_SPECIAL = re.compile(r"([\\`*\[\]|])")
_FIELD_LABELS = {
    "name": "Name",
    "distillery": "Distillery",
    "bottler": "Bottler",
    "specific_release": "Release",
    "batch_number": "Batch",
    "proof": "Proof",
    "abv": "ABV",
    "age": "Age",
    "release_year": "Release year",
    "mash_bill": "Mash bill",
    "tasting_notes": "Tasting notes",
    "msrp": "MSRP",
}

#: Highest numeric suffix tried before giving up on a unique output name.
_MAX_NAME_SUFFIX = 10_000

_PLACEHOLDER_NAMES = {"", "unknown", "unknown bottle", "n/a", "none"}


def slugify(text: Optional[str], fallback: str = "bottle", max_len: int = 60) -> str:
    """Turn text into a safe lowercase file/folder name.

    Only ``[a-z0-9_]`` survive, so the result can never contain a path
    separator, ``..`` or characters that some filesystems reject, and two
    names differing only in case map to the same slug (safe on
    case-insensitive filesystems).

    Examples: ``"Blanton's: Gold/Edition? <1>"`` -> ``'blantons_gold_edition_1'``,
    ``'../../etc'`` -> ``'etc'``.

    Args:
        text: Text to convert (None is treated as empty).
        fallback: Returned when nothing usable is left.
        max_len: Maximum length of the slug.

    Returns:
        The slug, or ``fallback``.
    """
    ascii_text = unicodedata.normalize("NFKD", text or "").encode("ascii", "ignore").decode("ascii")
    ascii_text = re.sub(r"['`\u2019]", "", ascii_text).lower()
    slug = re.sub(r"[^a-z0-9]+", "_", ascii_text).strip("_")
    slug = slug[:max_len].rstrip("_")
    return slug or fallback


def _is_placeholder_name(name: Optional[str]) -> bool:
    """True for empty names and 'Unknown ...' placeholders from the analyzer."""
    normalized = (name or "").strip().lower()
    return normalized in _PLACEHOLDER_NAMES or normalized.startswith("unknown")


def _unique_path(directory: Path, stem: str, suffix: str = "") -> Path:
    """First of ``stem``, ``stem_2``, ``stem_3``... (plus suffix) not in ``directory``.

    Raises:
        FileExistsError: If every candidate up to ``_MAX_NAME_SUFFIX`` exists.
    """
    for n in range(1, _MAX_NAME_SUFFIX + 1):
        candidate = directory / (f"{stem}{suffix}" if n == 1 else f"{stem}_{n}{suffix}")
        if not candidate.exists():
            return candidate
    raise FileExistsError(f"No free name for '{stem}{suffix}' in {directory}")


# ---------------------------------------------------------------------------
# Markdown helpers (every untrusted string goes through md_escape)
# ---------------------------------------------------------------------------


def md_escape(text: Any, limit: Optional[int] = None) -> str:  # noqa: ANN401
    """Untrusted text as safe inline Markdown.

    Whitespace (including newlines) collapses to single spaces; Markdown
    control characters (backslash, backtick, ``*``, ``[``, ``]``, ``|``)
    are backslash-escaped and ``<``/``>`` become entities, so a
    page title or quote can never break a table, open a link or inject HTML.
    ``limit`` cuts the text (before escaping) and marks the cut with ``[...]``.
    """
    value = re.sub(r"\s+", " ", "" if text is None else str(text)).strip()
    cut = limit is not None and len(value) > limit
    if cut:
        value = value[: max(0, int(limit or 0))].rstrip()
    value = _MD_SPECIAL.sub(r"\\\1", value).replace("<", "&lt;").replace(">", "&gt;")
    return value + (" \\[...\\]" if cut else "")


def md_quote(text: Any, limit: Optional[int] = None) -> str:  # noqa: ANN401
    """An escaped quote in double quotes."""
    return f'"{md_escape(text, limit)}"'


def _safe_url(url: Any) -> Optional[str]:  # noqa: ANN401
    """An http(s) URL safe inside ``(...)`` of a Markdown link, or None."""
    value = str(url or "").strip()
    if not value.lower().startswith(("http://", "https://")):
        return None
    for char, code in ((" ", "%20"), ("(", "%28"), (")", "%29"), ("<", "%3C"), (">", "%3E"), ("\n", ""), ("\r", "")):
        value = value.replace(char, code)
    return value


def md_link(text: Any, url: Any) -> str:  # noqa: ANN401
    """``[text](url)`` for an http(s) URL; just the escaped text otherwise."""
    safe = _safe_url(url)
    label = md_escape(text) or md_escape(url)
    return f"[{label}]({safe})" if safe else label


def _term(text: Any) -> str:  # noqa: ANN401
    """Untrusted text safe to print on a terminal: control characters (escape sequences) removed."""
    value = "" if text is None else str(text)
    return "".join(" " if ch in "\t\n\r" else ch for ch in value if ch in "\t\n\r" or unicodedata.category(ch)[0] != "C")


def _date(value: Any) -> str:  # noqa: ANN401
    """The date part of an ISO timestamp ('-' if none)."""
    text = str(value or "").strip()
    return md_escape(text[:10]) if text else "-"


def _host(url: Any) -> str:  # noqa: ANN401
    try:
        return (urlsplit(str(url)).hostname or "").lower()
    except ValueError:
        return ""


def _as_sheet(value: Any) -> Optional[FactSheet]:  # noqa: ANN401
    if value is None or isinstance(value, FactSheet):
        return value
    try:
        return FactSheet.from_dict(value)
    except (ValueError, TypeError, KeyError, AttributeError):
        logger.warning("Fact sheet could not be read; it is shown as missing", exc_info=True)
        return None


def _as_research(value: Any) -> Optional[WebResearch]:  # noqa: ANN401
    if value is None or isinstance(value, WebResearch):
        return value
    try:
        return WebResearch.from_dict(value)
    except (ValueError, TypeError, KeyError, AttributeError):
        logger.warning("Web research record could not be read; it is shown as missing", exc_info=True)
        return None


def _display(field_name: str, value: Any) -> str:  # noqa: ANN401
    from image_processor.fact_prompt import display_value

    return display_value(field_name, value)


def web_count_line(web_research: Any) -> str:  # noqa: ANN401
    """``web: N pages consulted, M cited`` (or why there was no research)."""
    wr = _as_research(web_research)
    if wr is None:
        return "web: not researched"
    line = f"web: {wr.pages_consulted} pages consulted, {wr.pages_cited} cited"
    if wr.status in ("skipped", "failed"):
        line += f" ({wr.status}: {wr.reason or 'no reason recorded'})"
    return line


def publish_banner(facts: Optional[FactSheet]) -> str:
    """``Publish-ready`` or ``NOT publish-ready: <blockers>``."""
    if facts is None:
        return "NOT publish-ready: not fact-checked (no fact sheet)"
    if facts.publish.get("ready"):
        return "Publish-ready"
    blockers = list(facts.publish.get("blockers") or []) or ["publish gate not computed"]
    return "NOT publish-ready: " + "; ".join(blockers)


def msrp_summary(facts: Optional[FactSheet], web_research: Optional[WebResearch] = None) -> str:
    """Short MSRP text with its source, for tables (``$59.99 (W1 example.com)``)."""
    if facts is None:
        return "Unknown (not fact-checked)"
    fact = facts.fields.get("msrp")
    if fact is None or fact.value is None or fact.status not in ("supported", "user"):
        return "Unknown"
    amount = _display("msrp", fact.value)
    if fact.status == "user":
        return f"{amount} (user)"
    ids = [c.source_id for c in fact.citations]
    hosts = []
    for sid in ids:
        src = facts.source(sid)
        if src is not None and src.url:
            hosts.append(f"{sid} {_host(src.url)}")
        else:
            hosts.append(sid)
    return f"{amount} ({', '.join(hosts) or 'source not recorded'})"


def _source_numbers(facts: FactSheet) -> Dict[str, int]:
    return {src.id: n for n, src in enumerate(facts.sources, 1)}


def _citation_cell(facts: FactSheet, citations: Sequence[Any], numbers: Dict[str, int], limit: int = 3) -> str:
    parts = []
    for cit in list(citations)[:limit]:
        n = numbers.get(cit.source_id)
        ref = f"[{n}]" if n is not None else md_escape(cit.source_id)
        src = facts.source(cit.source_id)
        kind = src.kind if src is not None else ""
        if kind == "label":
            parts.append(f"{ref} label {md_quote(cit.quote, 60)}")
        elif kind == "user":
            parts.append(f"{ref} user")
        else:
            parts.append(f"{ref} {md_escape(cit.source_id)} {md_quote(cit.quote, 60)}")
    if len(citations) > limit:
        parts.append(f"+{len(citations) - limit} more")
    return "; ".join(parts) or "-"


def _facts_table(facts: FactSheet) -> List[str]:
    numbers = _source_numbers(facts)
    lines = ["| Field | Value | Confidence | Sources |", "|---|---|---|---|"]
    for name in FACT_FIELDS:
        fact = facts.fields.get(name)
        label = _FIELD_LABELS.get(name, name)
        if fact is None:
            lines.append(f"| {label} | Unknown: not recorded | - | - |")
            continue
        if fact.status in ("supported", "user") and fact.value is not None:
            value = md_escape(_display(name, fact.value))
            if fact.status == "user":
                value += " (user-asserted)"
            sources = _citation_cell(facts, fact.citations, numbers)
        elif fact.status == "conflict":
            value = "CONFLICT: see Needs Review"
            sources = "-"
        else:
            value = f"Unknown: {md_escape(fact.reason or 'no source states it')}"
            sources = "-"
        conf = md_escape(fact.confidence) if fact.confidence and fact.value is not None else "-"
        lines.append(f"| {label} | {value} | {conf} | {sources} |")
    return lines


def _candidate_line(facts: FactSheet, field_name: str, cand: Any, numbers: Dict[str, int]) -> str:  # noqa: ANN401
    value = md_escape(_display(field_name, cand.value)) if field_name in FACT_FIELDS else md_escape(cand.value)
    quotes = []
    for cit in cand.citations:
        n = numbers.get(cit.source_id)
        ref = f"[{n}] {md_escape(cit.source_id)}" if n is not None else md_escape(cit.source_id)
        quotes.append(f"{ref} {md_quote(cit.quote)}")
    return f"{value} ({md_escape(cand.origin)}): " + ("; ".join(quotes) or "no citation")


def _needs_review(facts: FactSheet) -> List[str]:
    numbers = _source_numbers(facts)
    lines: List[str] = []
    open_conflicts = [c for c in facts.conflicts if not c.resolved]
    for conflict in open_conflicts:
        lines.append(f"- **{md_escape(conflict.field)}** ({md_escape(conflict.id)}): {md_escape(conflict.note)}")
        fields = conflict.field.split("/") if "/" in conflict.field else [conflict.field]
        for i, cand in enumerate(conflict.candidates, 1):
            fname = fields[min(i - 1, len(fields) - 1)] if len(fields) > 1 else fields[0]
            lines.append(f"  {i}. {_candidate_line(facts, fname, cand, numbers)}")
    for conflict in facts.conflicts:
        if conflict.resolved:
            lines.append(
                f"- Resolved: **{md_escape(conflict.field)}** ({md_escape(conflict.id)}): "
                f"{md_escape(conflict.note)}"
            )
    if facts.copy is not None and facts.copy.stale:
        lines.append(f"- Description is stale: {md_escape(facts.copy.reason or 'a cited field changed')}")
    return lines or ["Nothing needs review."]


def _price_source(facts: FactSheet, cit: Any, web: Optional[WebResearch]) -> str:  # noqa: ANN401
    src = facts.source(cit.source_id)
    if src is None:
        return f"{md_escape(cit.source_id)} {md_quote(cit.quote)}"
    if src.kind == "web":
        domain = next((e.domain for e in (web.evidence if web else []) if e.source_id == src.id), "") or _host(src.url)
        return f"{md_link(domain or src.url, src.url)}, {md_quote(cit.quote)}, retrieved {_date(src.fetched_at)}"
    if src.kind == "user":
        return f"user-asserted ({md_escape(src.ref)})"
    if src.kind == "label":
        return "label reading"
    return f"{md_escape(src.id)} OCR {md_quote(cit.quote)}"


def _prices(facts: Optional[FactSheet], bottle: Bottle, web: Optional[WebResearch]) -> List[str]:
    lines: List[str] = []
    if facts is None:
        lines.append(f"- **MSRP:** {format_money(bottle.msrp) if bottle.msrp is not None else 'Unknown: not fact-checked'}")
        if bottle.label_price is not None:
            lines.append(f"- **Retail:** {format_money(bottle.label_price)}: Price visible on label (retail, not MSRP)")
        return lines
    fact = facts.fields.get("msrp")
    if fact is not None and fact.status == "user" and fact.value is not None:
        src = facts.source(fact.citations[0].source_id) if fact.citations else None
        ref = md_escape(src.ref) if src is not None else "no reference"
        lines.append(f"- **MSRP:** {md_escape(_display('msrp', fact.value))}: user-asserted ({ref})")
    elif fact is not None and fact.status == "supported" and fact.value is not None:
        where = "; ".join(_price_source(facts, c, web) for c in fact.citations) or "source not recorded"
        lines.append(
            f"- **MSRP:** {md_escape(_display('msrp', fact.value))}: {where} "
            f"(confidence {md_escape(fact.confidence or '-')})"
        )
    else:
        reason = (fact.reason if fact is not None else None) or "no source stated an MSRP"
        lines.append(f"- **MSRP:** Unknown: {md_escape(reason)}")
    for obs in facts.prices:
        if obs.kind == "msrp":
            continue
        amount = md_escape(f"${obs.amount:.2f}" if obs.currency == "USD" else f"{obs.amount} {obs.currency}")
        label = "Retail" if obs.kind == "retail" else "Secondary"
        if obs.citation.source_id == "L1":
            lines.append(f"- **{label}:** {amount}: Price visible on label (retail, not MSRP)")
        else:
            lines.append(f"- **{label}:** {amount}: {_price_source(facts, obs.citation, web)}")
    return lines


def _description(facts: Optional[FactSheet]) -> List[str]:
    if facts is None:
        return ["_No description: the bottle was not fact-checked._"]
    copy = facts.copy
    if copy is None or not copy.description_md.strip():
        v = facts.verification
        if v.status == "partial" and v.reason:
            reason = v.reason
        elif v.status in ("skipped", "failed"):
            reason = f"text verification {v.status}" + (f": {v.reason}" if v.reason else "")
        else:
            reason = "a description needs a supported name and at least two supported fields"
        return [f"_No verified description: {md_escape(reason)}_"]
    from image_processor.fact_prompt import number_fields

    lines = []
    if copy.stale:
        lines.append(f"**Stale:** {md_escape(copy.reason or 'a cited field changed')}. Not publishable as is.")
        lines.append("")
    if copy.sentences:
        # Model-written text: escaped like any untrusted text; the markers are ours.
        parts = []
        for sentence in copy.sentences:
            marks = "".join(f"[{md_escape(c)}]" for c in sentence.get("citations") or [])
            parts.append(f"{md_escape(sentence.get('text'))} {marks}".strip())
        lines.append(" ".join(parts))
    else:
        lines.append(md_escape(copy.description_md))
    if not copy.stale:
        legend = [f"[{n.index}] {_FIELD_LABELS.get(n.field, n.field)}" for n in number_fields(facts)]
        if legend:
            lines.append("")
            lines.append(
                "_Markers number the facts the sentence relies on (not the Sources list): "
                + ", ".join(legend) + "._"
            )
    return lines


def _source_quotes(facts: FactSheet) -> Dict[str, List[str]]:
    quotes: Dict[str, List[str]] = {}

    def add(cit: Any) -> None:  # noqa: ANN401
        if cit.quote_kind != "verbatim":
            return
        bucket = quotes.setdefault(cit.source_id, [])
        if cit.quote not in bucket:
            bucket.append(cit.quote)

    for fact in facts.fields.values():
        for cand in fact.candidates:
            for cit in cand.citations:
                add(cit)
    for obs in facts.prices:
        add(obs.citation)
    return quotes


def _sources(facts: Optional[FactSheet], web: Optional[WebResearch]) -> List[str]:
    if facts is None:
        return ["No fact sheet: sources were not recorded."]
    if not facts.sources:
        return ["No sources were recorded."]
    quotes = _source_quotes(facts)
    lines = []
    for n, src in enumerate(facts.sources, 1):
        sid = md_escape(src.id)
        if src.kind == "label":
            text = f"label reading (vision model) of {md_escape(src.ref)}"
        elif src.kind == "ocr":
            text = f"OCR text of {md_escape(src.ref)}"
        elif src.kind == "web":
            domain = next((e.domain for e in (web.evidence if web else []) if e.source_id == src.id), "") or _host(src.url)
            title = src.title or domain or src.url
            text = (
                f"{md_link(title, src.url)} ({md_escape(src.tier or 'unclassified')}, {md_escape(domain)}, "
                f"retrieved {_date(src.fetched_at)})"
            )
            if src.evidence_path:
                text += f"; excerpt: `{md_escape(src.evidence_path).replace(chr(92), '')}`"
        else:
            text = f"user: {md_escape(src.ref)}"
        lines.append(f"{n}. **{sid}** {text}")
        for quote in quotes.get(src.id, []):
            lines.append(f"   - {md_quote(quote)}")
    return lines


def _verification(facts: Optional[FactSheet]) -> List[str]:
    if facts is None:
        return ["- Status: not run (no fact sheet)"]
    v = facts.verification
    accepted = set()
    for name, fact in facts.fields.items():
        if name == "msrp":  # MSRP observations are counted as prices
            continue
        for cand in fact.candidates:
            if cand.origin in ("ocr", "web"):
                for cit in cand.citations:
                    accepted.add((name, cit.source_id, cit.quote))
    model_prices = [p for p in facts.prices if p.citation.quote_kind == "verbatim"]
    think = md_escape(v.think or "-") + (" (lowered after a length cutoff)" if v.degraded_think else "")
    return [
        f"- Status: {md_escape(v.status)}" + (f" ({md_escape(v.reason)})" if v.reason else ""),
        f"- Model: {md_escape(v.text_model or 'none')}",
        f"- Think: extract {think}",
        f"- Prompt version: {md_escape(v.prompt_version)}",
        (
            f"- Claims accepted: {len(accepted)}; prices accepted: {len(model_prices)}; "
            f"proposals rejected: {len(v.rejected)} (listed in the JSON)"
        ),
        f"- Duration: {v.duration_s:.1f} s; model calls: {v.attempts}",
    ]


def _references(bottle: Bottle) -> List[str]:
    if not bottle.reference_links:
        return ["No web pages were used."]
    lines = []
    for link in bottle.reference_links:
        used = "cited" if link.used_as_evidence else "not cited"
        lines.append(
            f"- {md_link(link.label, link.url)} ({md_escape(link.source_class or 'unclassified')}, "
            f"retrieved {_date(link.retrieved_at)}, {used})"
        )
    return lines


def _web_log(web: Optional[WebResearch]) -> List[str]:
    if web is None:
        return ["Web research was not run for this bottle."]
    lines = [
        f"- Provider: {md_escape(web.provider or 'none')}",
        f"- Status: {md_escape(web.status)}" + (f" ({md_escape(web.reason)})" if web.reason else ""),
    ]
    if web.searched_at:
        lines.append(f"- Searched: {md_escape(web.searched_at)}")
    if web.queries:
        lines.append("- Queries: " + "; ".join(md_quote(q) for q in web.queries))
    else:
        lines.append("- Queries: none sent")
    lines.append(f"- {web_count_line(web)}")
    if not web.pages:
        lines.append("")
        lines.append("No pages were consulted.")
        return lines
    lines.append("")
    lines.append("| # | Site | Class | Retrieved | Outcome | Used for / Reason |")
    lines.append("|---|---|---|---|---|---|")
    for n, page in enumerate(web.pages, 1):
        site = md_escape(page.domain or _host(page.url) or page.url)
        if page.retrieved_at:  # only fetched URLs become links
            site = md_link(page.domain or page.url, page.url)
        if page.outcome == "cited":
            used = "cited for " + ", ".join(md_escape(f) for f in page.cited_for) if page.cited_for else (
                "cited (no field recorded; check)"
            )
        else:
            used = md_escape(page.reason or "-")
        lines.append(
            f"| {n} | {site} | {md_escape(page.source_class or '-')} | {_date(page.retrieved_at)} | "
            f"{md_escape(page.outcome)} | {used} |"
        )
    return lines


def _plain_facts(bottle: Bottle) -> List[str]:
    lines = ["| Field | Value | Confidence | Sources |", "|---|---|---|---|"]
    values = {
        "name": bottle.name, "distillery": bottle.distillery, "specific_release": bottle.specific_release,
        "batch_number": bottle.metadata.batch_number if bottle.metadata is not None else None,
        "proof": bottle.proof, "abv": bottle.abv, "age": bottle.age, "tasting_notes": bottle.tasting_notes,
        "msrp": format_money(bottle.msrp) if bottle.msrp is not None else None,
    }
    for name in FACT_FIELDS:
        if name not in values:
            continue
        value = values[name]
        shown = md_escape(value) if value not in (None, "") else "Unknown: not fact-checked"
        lines.append(f"| {_FIELD_LABELS.get(name, name)} | {shown} | - | not fact-checked |")
    return lines


def render_bottle_markdown(
    bottle: Bottle,
    image_links: Optional[List[str]] = None,
    model_used: Optional[str] = None,
    confidence: Optional[str] = None,
) -> str:
    """The bottle document in Markdown.

    Sections, in order: title, publish status banner, Needs Review, Facts,
    Prices, Description (draft), Sources, Verification, References, Web
    Research Log, Quantity, Images, Corrections, Processing Details.

    Args:
        bottle: The bottle (``facts`` and ``web_research`` may be None).
        image_links: Relative paths of the copied images (``images/IMG_1.jpg``),
            primary first; ``images/vendor_ref.*`` is marked as a vendor
            reference. Default: the names of ``bottle.metadata.source_images``.
        model_used: Vision model that read the label.
        confidence: Extraction confidence ('HIGH'/'MEDIUM'/'LOW').
    """
    facts = bottle.facts
    web = bottle.web_research
    metadata = bottle.metadata
    lines = [f"# {md_escape(bottle.name)}", "", f"**Status:** {md_escape(publish_banner(facts))}", ""]

    lines += ["## Needs Review", ""]
    lines += _needs_review(facts) if facts is not None else ["Not fact-checked: every value below is unverified."]
    lines += ["", "## Facts", ""]
    lines += _facts_table(facts) if facts is not None else _plain_facts(bottle)
    lines += ["", "## Prices", ""] + _prices(facts, bottle, web)
    lines += ["", "## Description (draft)", ""] + _description(facts)
    lines += ["", "## Sources", ""] + _sources(facts, web)
    lines += ["", "## Verification", ""] + _verification(facts)
    lines += ["", "## References", ""] + _references(bottle)
    lines += ["", "## Web Research Log", ""] + _web_log(web)

    lines += ["", "## Quantity", f"Total quantity: {bottle.quantity.count}"]
    total_value = bottle.total_value
    if bottle.quantity.count > 1 and total_value is not None:
        lines.append(f"Value per bottle: {format_money(total_value / bottle.quantity.count)}")
    lines.append(f"Total value: {format_money(total_value)}")

    lines += ["", "## Images"]
    if image_links is not None:
        links = list(image_links)
    else:
        links = [Path(p).name for p in (metadata.source_images if metadata is not None else [])]
        vendor_path = (bottle.vendor_image or {}).get("path")
        if vendor_path:
            links.append(str(vendor_path))
    photo_count = 0
    for i, link in enumerate(links, 1):
        name = Path(link).name
        if name.startswith("vendor_ref"):
            note = " (vendor reference, not for publication)"
        else:
            photo_count += 1
            note = " (primary)" if photo_count == 1 else ""
        shown = f"[{md_escape(name)}]({_safe_url_path(link)})" if image_links is not None else md_escape(name)
        lines.append(f"{i}. {shown}{note}")

    if bottle.corrections:
        lines += ["", "## Corrections"]
        for c in bottle.corrections:
            lines.append(
                f"- **{md_escape(c.field_name)}**: {md_quote(c.original_value)} -> {md_quote(c.corrected_value)} "
                f"({md_escape(c.source)})"
            )

    lines += ["", "## Processing Details", f"- **Images Processed**: {photo_count}"]
    lines.append(f"- **Model**: {md_escape(model_used or 'unknown')}")
    if confidence:
        lines.append(f"- **Extraction Confidence**: {md_escape(confidence)}")
    if bottle.run_id:
        lines.append(f"- **Run**: {md_escape(bottle.run_id)}")
    extracted_at = bottle.extracted_at
    if extracted_at.tzinfo is None:  # naive values are UTC (older files)
        extracted_at = extracted_at.replace(tzinfo=UTC)
    # Local time with its UTC offset, e.g. 2026-11-22T21:38:09-06:00.
    lines.append(f"- **Extracted**: {extracted_at.astimezone().isoformat(timespec='seconds')}")
    for warning in list(metadata.warnings) if metadata is not None else []:
        lines.append(f"- **Warning**: {md_escape(warning)}")
    return "\n".join(lines) + "\n"


def _safe_url_path(link: str) -> str:
    """A relative file link for Markdown (spaces and parentheses encoded)."""
    return str(link).replace(" ", "%20").replace("(", "%28").replace(")", "%29")


def _safe_relpath(path: str) -> Path:
    """A relative path inside a bottle folder (no absolute path, no ``..``).

    Raises:
        ValueError: If the path could leave the folder.
    """
    rel = Path(str(path))
    if rel.is_absolute() or not rel.parts or any(part in ("..", "") for part in rel.parts) or rel.drive:
        raise ValueError(f"extra file path must stay inside the bottle folder: {path!r}")
    return rel


class PreviewGenerator:
    """Generate preview reports and output files for bourbon bottles.

    Handles markdown generation, file saving, and error report creation.

    Attributes:
        output_dir: Base output directory for processed bottles.
    """

    def __init__(self, output_dir: Path, *, mark_output_dir: bool = True):
        """Initialize preview generator.

        Args:
            output_dir: Base output directory (required; created if missing).
            mark_output_dir: Write the output marker into ``output_dir`` so
                later scans skip it. Off when ``output_dir`` contains the
                photos being processed (each bottle folder is still marked).
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.mark_output_dir = mark_output_dir

    def generate_bottle_markdown(
        self,
        bottle: Bottle,
        image_links: Optional[List[str]] = None,
        model_used: Optional[str] = None,
        confidence: Optional[str] = None,
    ) -> str:
        """Generate the bottle document in Markdown (see :func:`render_bottle_markdown`).

        Args:
            bottle: Bottle model instance (metadata, facts and web research may be None).
            image_links: Relative paths of the copied images (e.g.
                ``'images/IMG_1.jpg'``); rendered as links, the first photo
                marked primary and ``vendor_ref.*`` marked as a vendor
                reference. Default: the names of ``bottle.metadata.source_images``.
            model_used: Vision model that extracted the data.
            confidence: Extraction confidence ('HIGH'/'MEDIUM'/'LOW').

        Returns:
            Markdown formatted string.
        """
        return render_bottle_markdown(bottle, image_links, model_used, confidence)

    @staticmethod
    def bottle_id_for(
        bottle_dict: Optional[Dict[str, Any]],
        metadata: Optional[ExtractedMetadata],
    ) -> str:
        """Identify a bottle: its run-assigned ``bottle_id``, else its primary image's stem.

        Matches the IDs ``BatchProcessor`` uses for failed bottles and that
        ``interactive_prompt`` shows, so rejects and reports line up. A
        ``process`` run assigns unique IDs (``bottle_dict['bottle_id']``), so
        photos with the same name in different folders never collide.

        Args:
            bottle_dict: Analyzer output (its ``bottle_id`` / ``images``), or None.
            metadata: Extracted metadata (its ``source_images``), or None.

        Returns:
            The bottle ID, or 'unknown'.
        """
        assigned = (bottle_dict or {}).get("bottle_id")
        if isinstance(assigned, str) and assigned:
            return assigned
        if metadata is not None and metadata.source_images:
            return Path(metadata.source_images[0]).stem
        images = (bottle_dict or {}).get("images") or []
        if images:
            return Path(images[0]).stem
        return "unknown"

    def save_bottle_files(
        self,
        bottle: Bottle,
        metadata: Optional[ExtractedMetadata] = None,
        prefix: Optional[str] = None,
        *,
        model_used: Optional[str] = None,
        confidence: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
        extra_files: Optional[List[Tuple[str, Union[bytes, str]]]] = None,
    ) -> Path:
        """Save a bottle's output folder atomically, never overwriting.

        Writes ``<output_dir>/<stem>/`` containing ``<stem>.md``,
        ``<stem>.json`` (``Bottle.to_dict()`` plus extra keys, schema
        version :data:`OUTPUT_SCHEMA_VERSION`), ``<stem>.txt`` (OCR text and
        the ``web: N pages consulted, M cited`` line), ``images/`` with a copy
        of every source image, ``evidence/W<n>.txt`` with each web excerpt
        exactly as the text model saw it (its sha256 matches the facts'
        ``content_sha256``), and ``images/vendor_ref.<ext>`` for an accepted
        vendor reference image (category VENDOR; not for publication). The
        stem is the slugified bottle name (``unknown_<primary image>`` for
        placeholder names); if that folder exists, ``_2``, ``_3``... is used.

        The folder is built in a hidden ``.<stem>.*.partial`` directory inside
        ``output_dir`` and renamed into place, so readers never see a
        half-written folder and a failure leaves nothing behind. Originals
        are only read. A missing source image is logged and listed in the
        JSON's ``missing_images``; it is not an error.

        Args:
            bottle: Bottle to save (build it with ``Bottle.from_analysis``).
            metadata: Extracted metadata; its images are added to the
                bottle's and its ``full_text`` is written to ``<stem>.txt``.
            prefix: Folder name to use instead of the bottle name (slugified).
            model_used: Vision model that extracted the data.
            confidence: Extraction confidence ('HIGH'/'MEDIUM'/'LOW').
            extra: Additional top-level keys for the JSON file.
            extra_files: More ``(relative path, content)`` files to write
                inside the folder (in the staging directory, so the save
                stays atomic). Paths must stay inside the folder.

        Returns:
            Path to the created output folder.

        Raises:
            FileSystemError: If the folder could not be written; nothing is
                left in ``output_dir`` in that case.
        """
        meta = bottle.metadata or metadata or ExtractedMetadata()
        sources: List[Path] = []
        seen: set = set()
        for image_path in [*meta.source_images, *(metadata.source_images if metadata else [])]:
            if str(image_path) not in seen:
                seen.add(str(image_path))
                sources.append(Path(image_path))
        ocr_text = (metadata or meta).full_text or ""
        bottle_id = Path(sources[0]).stem if sources else "unknown"

        if prefix:
            stem = slugify(prefix)
        elif _is_placeholder_name(bottle.name):
            stem = f"unknown_{slugify(bottle_id, fallback='bottle')}"
        else:
            stem = slugify(bottle.name)

        self.output_dir.mkdir(parents=True, exist_ok=True)
        if self.mark_output_dir:
            try:
                (self.output_dir / OUTPUT_MARKER).touch()
            except OSError:
                logger.debug("Could not write output marker in %s", self.output_dir, exc_info=True)

        staging: Optional[Path] = None
        try:
            staging = self._make_staging_dir(stem)
            # Marks the bottle folder itself as output, so it stays out of
            # later scans even when its parent is used as the scan root.
            (staging / OUTPUT_MARKER).touch()
            image_dir = staging / "images"
            image_dir.mkdir()

            copied: List[str] = []
            missing: List[str] = []
            used_names: set = set()
            for src in sources:
                if not src.is_file():
                    logger.warning("Source image not found, not copied: %s", src)
                    missing.append(str(src))
                    continue
                name, n = src.name, 2
                while name.lower() in used_names:
                    name = f"{src.stem}_{n}{src.suffix}"
                    n += 1
                used_names.add(name.lower())
                shutil.copy2(src, image_dir / name)
                copied.append(f"images/{name}")
                logger.debug("Copied image %s to %s", src, image_dir / name)

            vendor = self._copy_vendor_image(bottle, image_dir, used_names)
            files = self._evidence_files(bottle) + list(extra_files or [])
            self._write_extra_files(staging, files)

            final = _unique_path(self.output_dir, stem)
            facts = bottle.facts
            data = {
                **bottle.to_dict(),
                "schema_version": OUTPUT_SCHEMA_VERSION,
                "bottle_id": bottle_id,
                "batch_number": (
                    facts.flat_values().get("batch_number") if facts is not None else meta.batch_number
                ),
                "model_used": model_used,
                "confidence": confidence,
                "source_images": [str(p) for p in sources],
                "images": copied,
                "missing_images": missing,
                "total_value": str(bottle.total_value) if bottle.total_value is not None else None,
                "vendor_image": vendor,
                **(extra or {}),
            }
            links = copied + ([vendor["path"]] if vendor and vendor.get("path") else [])
            web_line = web_count_line(bottle.web_research)
            self._write_bottle_files(
                staging,
                final.name,
                markdown=self.generate_bottle_markdown(
                    bottle, image_links=links, model_used=model_used, confidence=confidence
                ),
                json_text=json.dumps(data, indent=2, ensure_ascii=False, default=str),
                ocr_text=(ocr_text.rstrip("\n") + "\n\n" if ocr_text.strip() else "") + web_line + "\n",
            )
            final = self._move_into_place(staging, final, stem)
            staging = None
        except Exception as e:
            if staging is not None:
                shutil.rmtree(staging, ignore_errors=True)
            raise FileSystemError(
                f"Failed to save output for bottle '{bottle.name}' in {self.output_dir}: "
                f"{type(e).__name__}: {e}"
            ) from e

        logger.info("Saved bottle '%s' to %s (%d image(s))", bottle.name, final, len(copied))
        return final

    def _make_staging_dir(self, stem: str) -> Path:
        """Create a hidden ``.<stem>.<random>.partial`` directory in ``output_dir``.

        Uses a plain ``mkdir`` (not ``tempfile.mkdtemp``, which forces mode
        0700) so the folder renamed into place gets the same umask-derived
        permissions as any other directory the user creates.
        """
        for _ in range(100):
            staging = self.output_dir / f".{stem}.{uuid.uuid4().hex}.partial"
            try:
                staging.mkdir()
            except FileExistsError:
                continue
            return staging
        raise FileExistsError(f"Could not create a staging directory in {self.output_dir}")

    @staticmethod
    def _evidence_files(bottle: Bottle) -> List[Tuple[str, Union[bytes, str]]]:
        """``evidence/W<n>.txt`` for every web excerpt shown to the text model."""
        research = bottle.web_research
        if research is None:
            return []
        files: List[Tuple[str, Union[bytes, str]]] = []
        for ev in research.evidence:
            if re.fullmatch(r"W\d+", ev.source_id):
                files.append((f"evidence/{ev.source_id}.txt", ev.text))
        return files

    @staticmethod
    def _write_extra_files(folder: Path, files: List[Tuple[str, Union[bytes, str]]]) -> None:
        """Write ``(relative path, content)`` files under ``folder``; str is UTF-8 with no newline added.

        Raises:
            ValueError: If a path would leave the folder or is written twice.
        """
        seen: set = set()
        for rel, content in files:
            path = _safe_relpath(rel)
            key = str(path).lower()
            if key in seen:
                raise ValueError(f"extra file written twice: {rel!r}")
            seen.add(key)
            target = folder / path
            target.parent.mkdir(parents=True, exist_ok=True)
            data = content.encode("utf-8") if isinstance(content, str) else bytes(content)
            with target.open("xb") as handle:  # never replaces a file
                handle.write(data)

    @staticmethod
    def _copy_vendor_image(bottle: Bottle, image_dir: Path, used_names: set) -> Optional[Dict[str, Any]]:
        """Copy an accepted vendor image to ``images/vendor_ref.<ext>``; its JSON record.

        The record keeps the image's provenance and verdict, and adds
        ``path``, ``category`` (VENDOR) and the ``usage`` note: the image is
        a reference for the reviewer, never licensed for publication. A
        missing cache file is recorded with ``path`` None (logged).
        """
        vendor = bottle.vendor_image
        if not vendor:
            return None
        record = {**vendor, "category": ImageCategory.VENDOR.name, "usage": VENDOR_IMAGE_USAGE, "path": None}
        cache = Path(str(vendor.get("cache_path") or ""))
        if not vendor.get("cache_path") or not cache.is_file():
            logger.warning("Vendor reference image not found in the cache, not copied: %s", cache)
            return record
        ext = re.sub(r"[^a-z0-9]", "", cache.suffix.lower())[:5] or "img"
        name, n = f"vendor_ref.{ext}", 2
        while name.lower() in used_names:
            name = f"vendor_ref_{n}.{ext}"
            n += 1
        used_names.add(name.lower())
        shutil.copy2(cache, image_dir / name)
        record["path"] = f"images/{name}"
        return record

    @staticmethod
    def _write_bottle_files(
        folder: Path,
        stem: str,
        *,
        markdown: str,
        json_text: str,
        ocr_text: str,
    ) -> None:
        """Write ``<stem>.md``, ``<stem>.json`` and ``<stem>.txt`` into ``folder``."""
        (folder / f"{stem}.md").write_text(markdown, encoding="utf-8")
        (folder / f"{stem}.json").write_text(json_text + "\n", encoding="utf-8")
        (folder / f"{stem}.txt").write_text(ocr_text, encoding="utf-8")

    @staticmethod
    def _move_into_place(staging: Path, final: Path, stem: str) -> Path:
        """Rename the staging folder to ``final`` or the next free ``stem_N``.

        If another process claimed ``final`` after it was chosen, the rename
        fails (the target is not empty) and the next free name is tried; the
        files inside are renamed to match. An existing folder is never
        replaced.

        Returns:
            The folder's final path.
        """
        for _ in range(_MAX_NAME_SUFFIX):
            try:
                if final.exists():
                    raise FileExistsError(final)
                staging.rename(final)
                return final
            except (FileExistsError, IsADirectoryError, NotADirectoryError):
                pass
            except OSError as e:
                if not final.exists():
                    raise
                logger.debug("Output name %s was taken: %s", final.name, e)
            new_final = _unique_path(final.parent, stem)
            for ext in (".md", ".json", ".txt"):
                (staging / f"{final.name}{ext}").rename(staging / f"{new_final.name}{ext}")
            final = new_final
        raise FileExistsError(f"No free output name for '{stem}' in {final.parent}")

    def generate_batch_preview_report(
        self,
        bottle_dicts: List[Dict],
        metadata_list: List[ExtractedMetadata],
        rejected_bottles: Optional[List[str]] = None,
        batch_id: Union[int, str] = 1,
    ) -> str:
        """Generate the batch preview: summary table, per-bottle details and feedback instructions.

        The summary table has, per bottle: name, quantity, images, the
        extraction confidence (fields filled), Ready (publish gate Y/N),
        unresolved Conflicts, Unknown fields, the Verification status and the
        MSRP with its source. Each bottle's details list its publish
        blockers and the ``web: N pages consulted, M cited`` line.

        Args:
            bottle_dicts: List of bottle dictionaries.
            metadata_list: List of ExtractedMetadata.
            rejected_bottles: List of rejected bottle identifiers.
            batch_id: Batch identifier for this report.

        Returns:
            Markdown formatted preview report.
        """
        lines = [f"# Batch Preview - Batch {md_escape(batch_id)}\n", "\n"]
        lines.append(f"Total bottles in batch: {len(bottle_dicts)}\n")

        if rejected_bottles:
            lines.append(f"Rejected bottles: {len(rejected_bottles)}\n")

        lines.append("\n## Grouping Summary\n")
        lines.append(
            "| Bottle Name | Quantity | Images | Confidence | Ready | Conflicts | Unknowns | Verification "
            "| MSRP (source) |\n"
        )
        lines.append("|---|---|---|---|---|---|---|---|---|\n")

        def filled(bottle_dict: Dict) -> int:
            return sum(1 for f in ["name", "distillery", "msrp", "proof", "age"] if bottle_dict.get(f))

        for bottle_dict, metadata in zip(bottle_dicts, metadata_list, strict=False):
            n = filled(bottle_dict)
            confidence = "HIGH" if n >= 4 else "MEDIUM" if n >= 2 else "LOW"
            facts = _as_sheet(bottle_dict.get("facts"))
            if facts is not None:
                ready = "Y" if facts.publish.get("ready") else "N"
                conflicts = str(sum(1 for c in facts.conflicts if not c.resolved))
                unknowns = str(sum(1 for f in facts.fields.values() if f.status == "unknown"))
                verification = facts.verification.status
            else:
                ready, conflicts, unknowns, verification = "N", "-", "-", "not run"
            lines.append(
                f"| {md_escape(bottle_dict.get('name', 'Unknown'))} | {bottle_dict.get('quantity', 1)} | "
                f"{len(metadata.source_images)} | {confidence} | {ready} | {conflicts} | {unknowns} | "
                f"{md_escape(verification)} | {md_escape(msrp_summary(facts))} |\n"
            )

        low_confidence_bottles = [
            i for i, (bottle_dict, _m) in enumerate(zip(bottle_dicts, metadata_list, strict=False), 1)
            if filled(bottle_dict) < 2
        ]
        if low_confidence_bottles or rejected_bottles:
            lines.append("\n## Low-Confidence or Rejected Items\n")
            if low_confidence_bottles:
                lines.append("**Low Confidence:**\n")
                for idx in low_confidence_bottles:
                    lines.append(f"- Bottle {idx} (check extracted metadata)\n")
            if rejected_bottles:
                lines.append("\n**Rejected:**\n")
                for bottle_id in rejected_bottles:
                    lines.append(f"- {md_escape(bottle_id)}\n")

        lines.append("\n## Extracted Metadata Details\n\n")

        for i, (bottle_dict, metadata) in enumerate(zip(bottle_dicts, metadata_list, strict=False), 1):
            facts = _as_sheet(bottle_dict.get("facts"))
            lines.append(f"### Bottle {i}\n")
            lines.append(f"**ID:** {md_escape(self.bottle_id_for(bottle_dict, metadata))}\n")
            lines.append(f"**Name:** {md_escape(bottle_dict.get('name', 'Unknown'))}\n")
            lines.append(f"**Distillery:** {md_escape(bottle_dict.get('distillery') or 'Unknown')}\n")
            lines.append(f"**MSRP:** {md_escape(msrp_summary(facts))}\n")
            label_price = parse_money(bottle_dict.get("label_price"))
            if label_price is not None and label_price > 0:
                lines.append(f"**Label price:** {format_money(label_price)} (retail, not MSRP)\n")
            for key, label in (("proof", "Proof"), ("age", "Age"), ("specific_release", "Release")):
                if bottle_dict.get(key):
                    lines.append(f"**{label}:** {md_escape(bottle_dict[key])}\n")
            lines.append(f"**Quantity:** {bottle_dict.get('quantity', 1)}\n")
            lines.append(f"**Status:** {md_escape(publish_banner(facts))}\n")
            lines.append(f"**{md_escape(web_count_line(bottle_dict.get('web_research')))}**\n")

            warnings = getattr(metadata, "warnings", None)
            lines.extend(f"**Review:** {md_escape(w)}\n" for w in (warnings if isinstance(warnings, list) else []))

            lines.append(f"**Images:** {len(metadata.source_images)}\n")
            for img_path in metadata.source_images:
                lines.append(f"- {md_escape(Path(img_path).name)}\n")

            if metadata.full_text:
                fence = "`" * 3
                lines.append("\n**OCR Preview:**\n")
                lines.append(f"{fence}\n")
                preview = metadata.full_text[:300].replace(fence, "'" * 3)
                if len(metadata.full_text) > 300:
                    preview += "..."
                lines.append(f"{preview}\n")
                lines.append(f"{fence}\n")

            lines.append("\n")

        lines.append("## Feedback Instructions\n")
        lines.append(
            "Review the batch above. Nothing is publish-ready until a person approves it. To give feedback:\n\n"
        )
        lines.append(f"1. **Interactive Mode** (`--interactive`): {REVIEW_ACTIONS_LINE}\n")
        lines.append("   - **A** - Approve the bottle for publishing (the other publish checks still apply)\n")
        lines.append("   - **R** - Reject this bottle for re-processing\n")
        lines.append("   - **E** - Edit fields (each edit is a user fact that wins over the sources)\n")
        lines.append("   - **C** - Settle conflicts: pick a candidate or type the value\n")
        lines.append("   - **N** - Keep the bottle as it is, not approved\n")
        lines.append("   - **Q** - Quit the review (remaining bottles are kept, not approved)\n\n")
        lines.append(
            "2. **Separate Feedback Command:** Use the `feedback` command with a JSON corrections "
            "file: a list of `{bottle_id, field_name, original_value, corrected_value}`, and/or "
            "`--approve <bottle_id>` to approve a bottle\n"
        )
        lines.append("   ```\n")
        lines.append(
            f"   bourbon-process feedback --batch {batch_id} --corrections corrections.json --approve <bottle_id>\n"
        )
        lines.append("   ```\n\n")
        lines.append("3. **Retry Failed Items:** Use the `retry` command with an error report file\n")
        lines.append("   ```\n")
        lines.append("   bourbon-process retry --report error_report_<bottle_id>.json\n")
        lines.append("   ```\n")

        return "\n".join(lines)

    def save_batch_preview(
        self,
        bottle_dicts: List[Dict],
        metadata_list: List[ExtractedMetadata],
        batch_id: Union[int, str] = 1,
        output_path: Optional[Path] = None,
        rejected_bottles: Optional[List[str]] = None,
    ) -> Path:
        """Save batch preview report to file with batch-specific naming.

        Writes the markdown preview to ``batch_{batch_id}_preview.md`` and the
        batch data (bottles, metadata, rejected IDs) to a sibling
        ``batch_{batch_id}.json``, which the ``feedback`` command loads.

        Args:
            bottle_dicts: List of bottle dictionaries.
            metadata_list: List of ExtractedMetadata.
            batch_id: Batch identifier for filename generation.
            output_path: Markdown file path, or a directory to write
                ``batch_{batch_id}_preview.md`` into (default: output_dir).
            rejected_bottles: IDs of bottles rejected or failed in this batch.

        Returns:
            Path to saved preview report.
        """
        if output_path is None:
            output_path = self.output_dir / f"batch_{batch_id}_preview.md"
        else:
            output_path = Path(output_path)
            if output_path.is_dir():
                output_path = output_path / f"batch_{batch_id}_preview.md"

        rejected = list(rejected_bottles or [])
        markdown = self.generate_batch_preview_report(
            bottle_dicts,
            metadata_list,
            rejected_bottles=rejected,
            batch_id=batch_id,
        )

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(markdown)
        logger.info("Saved batch preview to %s", output_path)

        if output_path.name.endswith("_preview.md"):
            json_path = output_path.with_name(output_path.name[: -len("_preview.md")] + ".json")
        else:
            json_path = output_path.with_suffix(".json")
        batch_data = {
            "schema_version": OUTPUT_SCHEMA_VERSION,
            "batch_id": str(batch_id),
            "generated_at": datetime.now(UTC).isoformat(),
            "bottles": bottle_dicts,
            "metadata": [m.to_dict() for m in metadata_list],
            "rejected": rejected,
        }
        with open(json_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(batch_data, indent=2, ensure_ascii=False, default=str))
        logger.info("Saved batch data to %s", json_path)

        return output_path

    def save_failure_report(
        self,
        bottle_id: str,
        images: List[str],
        error: Union[BaseException, str],
        stage: str,
        retry_count: int = 0,
    ) -> Path:
        """Write a per-bottle failure report that ``retry --report`` can read.

        The report is ``<output_dir>/error_report_<bottle_id>[_N].json`` (an
        existing report is never overwritten) and holds ``bottle_id``,
        ``images`` (absolute paths) and ``error_message``, plus error type,
        stage, hint, details and timestamp.

        Args:
            bottle_id: Bottle identifier (primary image stem).
            images: All image paths of the bottle.
            error: The exception, or a description of the failure.
            stage: Where it failed: 'analysis', 'save' or 'rejected'.
            retry_count: Retries already attempted.

        Returns:
            Path to the written report.
        """
        if isinstance(error, BaseException):
            error_message = f"{type(error).__name__}: {error}"
            info = create_error_report(error)
            error_type, hint = info.error_type.name, info.hint
            details = "".join(traceback.format_exception(type(error), error, error.__traceback__))
        else:
            error_message = str(error)
            error_type, hint, details = "UNKNOWN", None, None

        absolute_images = []
        for image in images:
            try:
                absolute_images.append(str(Path(image).absolute()))
            except (TypeError, ValueError):
                absolute_images.append(str(image))

        report = {
            "schema_version": OUTPUT_SCHEMA_VERSION,
            "bottle_id": bottle_id,
            "images": absolute_images,
            "error_message": error_message,
            "error_type": error_type,
            "stage": stage,
            "hint": hint,
            "details": details,
            "timestamp": datetime.now(UTC).isoformat(),
            "retry_count": retry_count,
            "suggested_fix": {},
        }

        self.output_dir.mkdir(parents=True, exist_ok=True)
        stem = f"error_report_{slugify(bottle_id, fallback='unknown')}"
        text = json.dumps(report, indent=2, ensure_ascii=False, default=str) + "\n"
        report_path = self._publish_new_file(stem, ".json", text)
        logger.info("Saved %s failure report for %s to %s", stage, bottle_id, report_path)
        return report_path

    def _publish_new_file(self, stem: str, suffix: str, text: str) -> Path:
        """Write ``text`` to the first free ``<stem>[_N]<suffix>`` in ``output_dir``.

        The content is written to a hidden temporary file first (created with
        ``open(..., 'x')`` so it gets umask-derived permissions) and then
        hard-linked to the final name. ``os.link`` fails instead of replacing
        an existing file, so a report written concurrently by another run is
        never overwritten: the next suffix is tried. On a filesystem without
        hard links the final name is created directly with mode ``'x'``.

        Returns:
            Path of the published file.
        """
        tmp: Optional[Path] = None
        for _ in range(100):
            candidate = self.output_dir / f".{stem}.{uuid.uuid4().hex}.tmp"
            try:
                with candidate.open("x", encoding="utf-8") as f:
                    f.write(text)
            except FileExistsError:
                continue
            tmp = candidate
            break
        if tmp is None:
            raise FileExistsError(f"Could not create a temporary file in {self.output_dir}")

        try:
            target = _unique_path(self.output_dir, stem, suffix)
            for _ in range(_MAX_NAME_SUFFIX):
                try:
                    os.link(tmp, target)
                    return target
                except FileExistsError:
                    target = _unique_path(self.output_dir, stem, suffix)
                except OSError:
                    # No hard links here (e.g. some network/FAT filesystems):
                    # create the final file exclusively instead.
                    logger.debug("Hard link not supported in %s", self.output_dir, exc_info=True)
                    break
            while True:
                try:
                    with target.open("x", encoding="utf-8") as f:
                        f.write(text)
                    return target
                except FileExistsError:
                    target = _unique_path(self.output_dir, stem, suffix)
        finally:
            tmp.unlink(missing_ok=True)

    def interactive_prompt(
        self,
        bottle_dicts: List[Dict],
        metadata_list: List[ExtractedMetadata],
    ) -> Tuple[List[Dict[str, str]], List[str], List[str]]:
        """Review each bottle interactively: facts, conflicts, blockers, then an action.

        Actions (case-insensitive; anything else asks again, so a stray key
        or a bare Enter never approves a bottle):

        - ``A`` approve the bottle for publishing (the other publish checks
          still apply) and go to the next bottle;
        - ``R`` reject it (not saved; a failure report is written);
        - ``E`` edit fields by name; each edit is a correction (source
          ``interactive``) carrying the bottle's ``bottle_id``;
        - ``C`` settle each unresolved conflict: pick candidate 1..n, type a
          value, or skip; each choice is a correction (source ``interactive``);
        - ``N`` keep the bottle as it is, not approved, and go on;
        - ``Q`` stop reviewing (remaining bottles are kept, not approved).

        After ``E`` or ``C`` the same bottle is offered again, so it can be
        approved with its edits. Nothing here changes the bottle dicts: apply
        the corrections with :func:`image_processor.corrections.apply_corrections`,
        re-reconcile the facts, then record the approvals (the CLI does this).

        Args:
            bottle_dicts: List of bottle dictionaries.
            metadata_list: List of ExtractedMetadata.

        Returns:
            ``(corrections, rejected_bottles, approvals)``: correction dicts
            (bottle_id, field_name, original_value, corrected_value,
            corrected_at), the IDs of rejected bottles, and the IDs of
            approved bottles.
        """
        corrections: List[Dict[str, str]] = []
        rejected_bottles: List[str] = []
        approvals: List[str] = []

        print("\n" + "=" * 70)
        print("BATCH PROCESSING - INTERACTIVE REVIEW")
        print("=" * 70)
        print(f"\nReviewing {len(bottle_dicts)} bottles in this batch")
        print(f"Actions: {REVIEW_ACTIONS_LINE}\n")

        quit_review = False
        for i, (bottle_dict, metadata) in enumerate(zip(bottle_dicts, metadata_list, strict=False), 1):
            bottle_id = self.bottle_id_for(bottle_dict, metadata)
            self._print_review(i, len(bottle_dicts), bottle_id, bottle_dict, metadata)
            pending: Dict[str, str] = {}
            while True:
                action = self._ask_action()
                if action == "Q":
                    print("\nQuitting interactive review (remaining bottles are kept, not approved)...")
                    quit_review = True
                    break
                if action == "A":
                    approvals.append(bottle_id)
                    print(f"✓ Approved bottle {i}: {bottle_id}")
                    break
                if action == "R":
                    rejected_bottles.append(bottle_id)
                    print(f"✗ Rejected bottle {i}: {bottle_id}")
                    break
                if action == "N":
                    print(f"→ Kept bottle {i} without approval: {bottle_id}")
                    break
                if action == "E":
                    corrections.extend(self._edit_fields(bottle_dict, bottle_id, pending))
                elif action == "C":
                    corrections.extend(self._settle_conflicts(bottle_dict, bottle_id, pending))
                print(f"\nBottle {i} ({bottle_id}): {REVIEW_ACTIONS_LINE}")
            if quit_review:
                break

        print("\n" + "=" * 70)
        print("INTERACTIVE REVIEW SUMMARY")
        print("=" * 70)
        print(f"Approved:     {len(approvals)}/{len(bottle_dicts)}")
        print(f"Rejected:     {len(rejected_bottles)}")
        print(f"Corrections:  {len(corrections)}")
        print("=" * 70)

        return corrections, rejected_bottles, approvals

    @staticmethod
    def _ask_action() -> str:
        print("\nAction [A/R/E/C/N/Q]? ", end="", flush=True)
        action = input().strip().upper()
        while action not in REVIEW_ACTIONS:
            print(f"⚠ Unknown action '{action}'; enter A, R, E, C, N or Q: ", end="", flush=True)
            action = input().strip().upper()
        return action

    def _print_review(
        self, index: int, total: int, bottle_id: str, bottle_dict: Dict, metadata: Optional[ExtractedMetadata]
    ) -> None:
        """Print one bottle: status, facts, conflicts, web line, quantity, images, confidence."""
        print("-" * 70)
        print(f"Bottle {index}/{total} - ID: {bottle_id}")
        print("-" * 70)
        facts = _as_sheet(bottle_dict.get("facts"))
        print(f"\nStatus:      {publish_banner(facts)}")
        if facts is None:
            print(f"Name:        {bottle_dict.get('name', 'Unknown')}")
            print(f"Distillery:  {bottle_dict.get('distillery', 'Unknown')}")
            print(f"MSRP:        {format_money(parse_money(bottle_dict.get('msrp')))}")
            for key, label in (("proof", "Proof"), ("age", "Age"), ("specific_release", "Release")):
                if bottle_dict.get(key):
                    print(f"{label + ':':13}{bottle_dict[key]}")
        else:
            print("\nFacts:")
            for name in FACT_FIELDS:
                fact = facts.fields.get(name)
                if fact is None:
                    continue
                if fact.status in ("supported", "user") and fact.value is not None:
                    value = _display(name, fact.value) + (" (user)" if fact.status == "user" else "")
                elif fact.status == "conflict":
                    value = "CONFLICT (see below)"
                else:
                    value = f"Unknown: {fact.reason or 'no source states it'}"
                conf = fact.confidence if fact.value is not None and fact.confidence else "-"
                print(f"  {_FIELD_LABELS.get(name, name):14} {conf:7} {_term(value)[:90]}")
            open_conflicts = [c for c in facts.conflicts if not c.resolved]
            if open_conflicts:
                print("\nConflicts (use [C] to settle):")
                for conflict in open_conflicts:
                    print(f"  {_term(conflict.id)} ({_term(conflict.field)}): {_term(conflict.note)[:200]}")
        label_price = parse_money(bottle_dict.get("label_price"))
        if label_price is not None and label_price > 0:
            print(f"Label price: {format_money(label_price)} (retail, not MSRP)")
        print(web_count_line(bottle_dict.get("web_research")))
        print(f"Quantity:    {bottle_dict.get('quantity', 1)}")
        print(f"Images:      {len(metadata.source_images) if metadata is not None else 0}")
        # The extraction confidence recorded for this bottle (not re-derived).
        print(f"Confidence:  {bottle_dict.get('confidence') or 'Unknown'}")

    @staticmethod
    def _correction(bottle_id: str, field_name: str, original: str, corrected: str) -> Dict[str, str]:
        return {
            "field_name": field_name,
            "original_value": original,
            "corrected_value": corrected,
            "bottle_id": bottle_id,
            "corrected_at": datetime.now(UTC).isoformat(),
        }

    @staticmethod
    def _editable_fields() -> List[str]:
        from image_processor.corrections import CORRECTABLE_FIELDS

        return [f for f in FACT_FIELDS if f in CORRECTABLE_FIELDS]

    def _edit_fields(self, bottle_dict: Dict, bottle_id: str, pending: Dict[str, str]) -> List[Dict[str, str]]:
        """Ask for field names and new values until a blank field name."""
        fields = self._editable_fields()
        out: List[Dict[str, str]] = []
        print(f"\nEditable fields: {', '.join(fields)}")
        while True:
            name = input("Field to edit (Enter when done): ").strip().lower()
            if not name:
                return out
            if name not in fields:
                print(f"  ⚠ '{name}' is not an editable field")
                continue
            raw = bottle_dict.get(name)
            current = pending.get(name, "" if raw is None else str(raw))
            new_value = input(f"{_FIELD_LABELS.get(name, name):14} [{current}]: ").strip()
            if new_value and new_value != current.strip():
                out.append(self._correction(bottle_id, name, current, new_value))
                pending[name] = new_value
                print(f"  Will update: {name} = {new_value}")

    def _settle_conflicts(self, bottle_dict: Dict, bottle_id: str, pending: Dict[str, str]) -> List[Dict[str, str]]:
        """For each unresolved conflict: pick a candidate, type a value, or skip."""
        from image_processor.enrichment import format_flat_value

        facts = _as_sheet(bottle_dict.get("facts"))
        open_conflicts = [c for c in facts.conflicts if not c.resolved] if facts is not None else []
        if not open_conflicts:
            print("No unresolved conflicts.")
            return []
        editable = set(self._editable_fields())
        out: List[Dict[str, str]] = []
        for conflict in open_conflicts:
            fields = conflict.field.split("/") if "/" in conflict.field else [conflict.field]
            print(f"\nConflict {_term(conflict.id)} ({_term(conflict.field)}): {_term(conflict.note)}")
            for fi, name in enumerate(fields):
                if name not in editable:
                    print(f"  ⚠ {name} cannot be corrected here; use feedback")
                    continue
                cands = list(conflict.candidates)
                if len(fields) > 1:  # proof/abv: candidate i belongs to field i
                    cands = cands[fi:fi + 1]
                for n, cand in enumerate(cands, 1):
                    quotes = "; ".join(f'{_term(c.source_id)} "{_term(c.quote)[:80]}"' for c in cand.citations) or "no citation"
                    print(f"  {n}. {_term(_display(name, cand.value))} ({_term(cand.origin)}): {quotes}")
                answer = input(
                    f"  {name}: 1-{len(cands)} picks a candidate, or type the value "
                    "('=' first for a number), Enter skips: "
                ).strip()
                if not answer:
                    continue
                if answer.startswith("="):
                    value = answer[1:].strip()
                elif answer.isdigit() and 1 <= int(answer) <= len(cands):
                    value = format_flat_value(name, cands[int(answer) - 1].value) or ""
                else:
                    value = answer
                if not value:
                    continue
                raw = bottle_dict.get(name)
                current = pending.get(name, "" if raw is None else str(raw))
                if value == current.strip():
                    continue
                out.append(self._correction(bottle_id, name, current, value))
                pending[name] = value
                print(f"  Will set: {name} = {value}")
        return out
