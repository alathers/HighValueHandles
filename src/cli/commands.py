"""CLI commands for bourbon image processor.

Every command resolves one immutable :class:`lib.settings.Settings` (defaults,
config files, ``BOURBON_*`` environment, flags) before doing anything; a
configuration error exits 1 with every problem listed. Commands then run with
a :class:`RunContext` holding the settings plus the run's live objects
(model server manager, model selector, hints).
"""

import argparse
import hashlib
import json
import logging
import os
import signal
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from image_processor.batch_processor import BatchProcessor, BottleLogFilter, describe_error
from image_processor.bottle_analyzer import BottleAnalyzer
from image_processor.corrections import CORRECTABLE_FIELDS, apply_corrections
from image_processor.enrichment import Enricher, approve_facts, refresh_facts
from image_processor.fact_checker import FactChecker
from image_processor.image_organizer import OUTPUT_MARKER, ImageOrganizer
from image_processor.preview_generator import OUTPUT_SCHEMA_VERSION, PreviewGenerator
from lib.constants import DEFAULT_VISION_MODEL
from lib.exceptions import FATAL_MODEL_ERRORS, ModelServerError, VisionAnalysisError
from lib.hints import HintBook, apply_hint_overrides
from lib.model_availability import (
    existing_ancestor,
    format_gb,
    is_local_url,
    memory_verdict,
    total_memory_bytes,
)
from lib.model_selector import ModelSelector, ModelStage, active_stages
from lib.model_server import ModelServerManager
from lib.settings import (
    THINK_LEVELS,
    WEB_PROVIDERS,
    Settings,
    SettingsError,
    describe,
    ignored_keys,
    render_template,
    resolve_settings,
    settings_to_dict,
)
from models.bottle import Correction
from models.facts import FactSheet
from models.image import group_bottle_id

logger = logging.getLogger(__name__)

#: Subcommands; an argument list naming none of them runs ``process``.
SUBCOMMANDS = ("process", "feedback", "retry", "config")
#: Top-level options that take a value (skipped when finding the subcommand).
_TOP_LEVEL_VALUE_OPTIONS = ("--config",)
#: Log line format; ``bottle`` is the bottle a worker thread is analysing ('-' otherwise).
LOG_FORMAT = "%(asctime)s %(levelname)s [%(bottle)s] %(name)s: %(message)s"


# ---------------------------------------------------------------------------
# Parser (every default is None: an absent flag never overrides config)
# ---------------------------------------------------------------------------


def _add_config_args(parser: argparse.ArgumentParser, *, suppress: bool) -> None:
    """``--debug``, ``--config`` and ``--no-config`` (top level, and again after the subcommand)."""
    default = argparse.SUPPRESS if suppress else None
    parser.add_argument(
        "--debug", action="store_const", const=True, default=default,
        help="Enable debug logging and verbose output",
    )
    parser.add_argument(
        "--config", type=Path, default=default, metavar="PATH",
        help="Read only this config file (default: ~/.config/bourbon-process/config.toml, then ./bourbon.toml)",
    )
    parser.add_argument(
        "--no-config", action="store_const", const=True, default=default,
        help="Read no config file (reproducible runs)",
    )


def _add_server_args(parser: argparse.ArgumentParser) -> None:
    """Model and model-server options shared by ``process`` and ``retry``."""
    parser.add_argument(
        "-m", "--model", type=str, default=None,
        help=f"Ollama vision model tag; must be vision-capable (default: {DEFAULT_VISION_MODEL})",
    )
    parser.add_argument("-u", "--ollama-url", type=str, default=None, help="Ollama service URL")
    parser.add_argument(
        "--server-start-timeout", type=float, default=None, metavar="SECONDS",
        help="Seconds to wait for a model server this run starts (default: 30)",
    )
    parser.add_argument(
        "--request-timeout", dest="request_timeout", type=float, default=None, metavar="SECONDS",
        help="Per-request vision timeout in seconds, covering a cold model load (default: 600)",
    )
    parser.add_argument(
        "--keep-server", action=argparse.BooleanOptionalAction, default=None,
        help="Leave a model server started by this run running afterwards",
    )
    parser.add_argument(
        "--server-autostart", action=argparse.BooleanOptionalAction, default=None,
        help="Start a local Ollama if none is running (--no-server-autostart: fail instead)",
    )
    parser.add_argument(
        "--keep-models-loaded", action=argparse.BooleanOptionalAction, default=None,
        help="Keep the vision and text models loaded together (skip the unloads between phases)",
    )


def _add_retry_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-r", "--allow-retry", dest="retry_enabled", action="store_const", const=True, default=None,
        help="Repeat model requests that fail in a way that may pass (default: on)",
    )
    parser.add_argument(
        "--no-retry", dest="retry_enabled", action="store_const", const=False, default=None,
        help="Make every model request once",
    )
    parser.add_argument(
        "--retry-attempts", type=int, default=None, metavar="N",
        help="Attempts per model request when retry is on, 1-5 (default: 3)",
    )


def _add_text_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--verify", action=argparse.BooleanOptionalAction, default=None,
        help="Fact-check each bottle with the text model (default: on)",
    )
    parser.add_argument(
        "--text-model", type=str, default=None, metavar="TAG",
        help="Ollama text model of the fact check, or 'none' (default: gpt-oss:120b)",
    )
    parser.add_argument(
        "--text-think", type=str, default=None, metavar="|".join(THINK_LEVELS),
        help="Reasoning effort of the fact-check extraction (default: high)",
    )
    parser.add_argument(
        "--text-timeout", type=float, default=None, metavar="SECONDS",
        help="Read timeout of one text-model call (default: 900)",
    )


def _add_web_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--web", action=argparse.BooleanOptionalAction, default=None,
        help="Look bottles up on the web and cite the pages used (default: on)",
    )
    parser.add_argument(
        "--web-provider", type=str, default=None, metavar="|".join(WEB_PROVIDERS),
        help="Search provider (default: auto = Brave key, Ollama key, SearXNG URL, else DuckDuckGo)",
    )
    parser.add_argument(
        "--web-cache-dir", type=Path, default=None, metavar="PATH",
        help="Web cache directory (default: ${XDG_CACHE_HOME:-~/.cache}/bourbon-process/web)",
    )
    parser.add_argument(
        "--refresh-web", action="store_const", const=True, default=None,
        help="Ignore cached web pages and search results for this run",
    )
    parser.add_argument(
        "-v", "--include-vendor-images", dest="include_vendor_images",
        action=argparse.BooleanOptionalAction, default=None,
        help="Fetch a vendor reference image per bottle (implies web lookups)",
    )


def _add_hints_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--hints", dest="hints_file", type=Path, default=None, metavar="PATH",
        help="Hints TOML: per-bottle reading notes and sourced field values (see examples/hints.example.toml)",
    )


def parse_args(argv: list | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Arguments that name no subcommand run ``process`` (``bourbon-process -f
    photos`` is ``bourbon-process process -f photos``). Every option defaults
    to None; :func:`lib.settings.resolve_settings` fills in config and defaults.
    """
    argv = list(sys.argv[1:] if argv is None else argv)
    if not _names_subcommand(argv):
        argv = ["process", *argv]
    return build_parser().parse_args(argv)


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser (top-level options and subcommands)."""
    parser = argparse.ArgumentParser(
        description="Process bourbon bottle images and extract metadata.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_config_args(parser, suppress=False)

    # Also accepted after the subcommand; SUPPRESS keeps a missing
    # subcommand-level flag from overwriting the top-level value.
    common = argparse.ArgumentParser(add_help=False)
    _add_config_args(common, suppress=True)

    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # process
    process_parser = subparsers.add_parser(
        "process", parents=[common], help="Process bourbon bottle images and extract metadata",
    )
    process_parser.add_argument(
        "-f", "--image-folder", type=Path, default=None,
        help="Folder containing images to process (default: current directory)",
    )
    process_parser.add_argument(
        "-o", "--output-dir", type=Path, default=None,
        help="Output directory for processed bottles (default: processed/)",
    )
    process_parser.add_argument(
        "-b", "--batch-size", type=str, default=None,
        help="Bottles per batch, or 'all' (default: 5)",
    )
    process_parser.add_argument(
        "--concurrency", type=int, default=None, metavar="N",
        help="Bottle groups analysed in parallel, 1-10; output is identical to a sequential run "
        "(default: 2; use 1 for evaluation runs)",
    )
    process_parser.add_argument(
        "--group-gap-seconds", type=float, default=None, metavar="SECONDS",
        help="Most seconds between consecutive photos of one bottle; photos are grouped by "
        "EXIF capture time (default: 12)",
    )
    process_parser.add_argument(
        "--group-max-size", type=int, default=None, metavar="N",
        help="Most photos per bottle; longer runs of photos are split at their largest time gap, "
        "0 = no limit (default: 3)",
    )
    process_parser.add_argument(
        "--split-budget-ratio", type=float, default=None, metavar="RATIO",
        help=(
            "When the vision model says a group's photos show different products, the group is "
            "split at its largest time gap and each half is read again (2 extra analyses). "
            "This caps the extra analyses of the whole run at RATIO x number of groups (at least 4); "
            "retries of a request do not count; 0 turns the split check off (default: 0.35, max 2)"
        ),
    )
    _add_server_args(process_parser)
    process_parser.add_argument(
        "--pull", action=argparse.BooleanOptionalAction, default=None,
        help="Download missing models after checking their size against free disk space "
        "(off by default: models are never downloaded implicitly)",
    )
    process_parser.add_argument(
        "--min-free-gb", type=float, default=None, metavar="GB",
        help="Free space (GiB) to keep on the Ollama model store after a --pull (default: 5)",
    )
    process_parser.add_argument(
        "-i", "--interactive", action=argparse.BooleanOptionalAction, default=None,
        help="Review every bottle interactively before saving",
    )
    _add_hints_arg(process_parser)
    _add_retry_args(process_parser)
    _add_text_args(process_parser)
    _add_web_args(process_parser)
    process_parser.add_argument(
        "--dry-run", action="store_const", const=True, default=None,
        help="Check inputs and probe the model server without starting it, pulling "
        "models or processing images",
    )

    # feedback
    feedback_parser = subparsers.add_parser(
        "feedback", parents=[common], help="Apply feedback and corrections to processed batch",
    )
    feedback_parser.add_argument(
        "-o", "--output-dir", type=Path, default=None,
        help="Output directory containing processed batches (default: processed/)",
    )
    feedback_parser.add_argument(
        "--batch", type=str, default=None,
        help=(
            "Batch ID to apply feedback to, e.g. '20261122_193000_001' for "
            "batch_20261122_193000_001_preview.md (default: most recent batch)"
        ),
    )
    feedback_parser.add_argument(
        "--corrections", type=Path, default=None, help="Path to JSON file containing corrections",
    )
    feedback_parser.add_argument(
        "-i", "--interactive", action=argparse.BooleanOptionalAction, default=None,
        help="Enable interactive feedback collection mode",
    )
    feedback_parser.add_argument(
        "--approve", action="append", default=None, metavar="BOTTLE_ID",
        help="Approve a bottle for publishing (repeatable; applied by the fact-check stage)",
    )

    # retry
    retry_parser = subparsers.add_parser(
        "retry", parents=[common], help="Retry processing failed bottles from error reports",
    )
    retry_parser.add_argument("--report", type=Path, required=True, help="Path to error_report.json file")
    retry_parser.add_argument(
        "-o", "--output-dir", type=Path, default=None,
        help="Output directory for processed bottles (default: processed/)",
    )
    _add_hints_arg(retry_parser)
    _add_server_args(retry_parser)
    _add_retry_args(retry_parser)
    _add_text_args(retry_parser)
    _add_web_args(retry_parser)

    # config
    config_parser = subparsers.add_parser(
        "config", parents=[common], help="Show the effective configuration or print a config template",
    )
    config_parser.add_argument(
        "config_action", choices=("show", "template"),
        help="show: files, values and their sources; template: a commented config file",
    )
    config_parser.add_argument(
        "--json", action="store_const", const=True, default=None, help="show: print JSON",
    )

    return parser


def _names_subcommand(argv: list[str]) -> bool:
    """True if ``argv`` names a subcommand, or asks for the top-level help.

    The first token that is neither an option nor the value of a top-level
    option that takes one (``--config PATH``) is the subcommand position.
    """
    skip = False
    for token in argv:
        if skip:
            skip = False
            continue
        if token in ("-h", "--help"):
            return True
        if token in _TOP_LEVEL_VALUE_OPTIONS:
            skip = True
            continue
        if not token.startswith("-"):
            return token in SUBCOMMANDS
    return False


def setup_logging(debug: bool = False) -> None:
    """Configure logging (stdout; each line names the bottle being analysed, or '-').

    Args:
        debug: Whether to enable DEBUG-level logging.
    """
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(level=level, format=LOG_FORMAT, stream=sys.stdout)
    for handler in logging.getLogger().handlers:
        if not any(isinstance(f, BottleLogFilter) for f in handler.filters):
            handler.addFilter(BottleLogFilter())
    # Third-party DEBUG output (PIL logs ~15 EXIF tags per image read) would
    # bury the pipeline's own messages.
    for noisy in ("PIL", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.INFO)


# ---------------------------------------------------------------------------
# Run context
# ---------------------------------------------------------------------------


@dataclass
class RunContext:
    """Settings plus the live objects of one command run."""

    settings: Settings
    server_manager: ModelServerManager | None = None
    model_selector: ModelSelector | None = None
    mark_output_dir: bool = True
    hints: HintBook = dataclass_field(default_factory=lambda: HintBook.EMPTY)
    run_id: str = ""


def _as_context(target: "Settings | RunContext | argparse.Namespace") -> RunContext:
    """A RunContext for a command entry point (parsed args are resolved to Settings)."""
    if isinstance(target, RunContext):
        return target
    if isinstance(target, Settings):
        return RunContext(target)
    return RunContext(resolve_settings(target))


def build_server_manager(settings: Settings) -> ModelServerManager:
    """Create (but do not start) the Ollama server manager for these settings.

    Raises:
        ValueError: If a server option is invalid.
    """
    server = settings.server
    return ModelServerManager(
        keep_running=server.keep_running,
        base_url=server.ollama_url,
        model=settings.model.vision_model,
        timeout=server.request_timeout,
        startup_timeout=server.start_timeout,
        auto_start=server.autostart,
    )


def _model_selector(settings: Settings) -> ModelSelector:
    """The run's per-stage model selector (vision model, plus the text model when active)."""
    config = SimpleNamespace(model=settings.model.vision_model)
    if settings.text.active:
        config.metadata_generation_model = settings.text.model
    return ModelSelector.from_config(config)


def _required_models(settings: Settings) -> list[tuple[str, str]]:
    """Models the server must provide before processing starts, with their capability.

    Only the stages this run calls are required: the vision model, plus the
    text model when the fact check is active. Inactive stages' models are
    never checked or pulled.
    """
    return _model_selector(settings).required_model_specs(active_stages(settings.text.active))


def _text_model_ram_verdict(manager: ModelServerManager, text_model: str) -> tuple[str, str, int | None]:
    """``(verdict, detail, size)`` of the text model against this machine's memory.

    verdict is ``ok``, ``warn``, ``error``, ``unknown`` or ``remote``.
    """
    base_url = getattr(getattr(manager, "server", None), "base_url", None)
    if isinstance(base_url, str) and not is_local_url(base_url):
        return "remote", f"Ollama at {base_url} is remote; its own memory is used", None
    size = manager.installed_model_size(text_model)
    if isinstance(size, bool) or not isinstance(size, (int, float)) or size <= 0:
        return "unknown", "model size unknown", None
    verdict, detail = memory_verdict(size, total_memory_bytes())
    return verdict, detail, size


def _text_model_memory_check(manager: ModelServerManager, text_model: str) -> int:
    """Advisory check of the text model's size against physical memory.

    The size is what the server reports for the installed model
    (``/api/tags``). Unknown sizes or memory are only logged at debug level.

    Returns:
        0 if the model fits (or this cannot be measured; a warning is
        logged above 75 % of memory), 2 if the model is larger than
        physical memory (an error is logged).
    """
    verdict, detail, size = _text_model_ram_verdict(manager, text_model)
    advice = "use a smaller model (--text-model gpt-oss:20b) or skip the fact check (--no-verify)"
    if verdict == "remote":
        logger.debug("%s; skipping the memory check of text model '%s'", detail, text_model)
        return 0
    if verdict == "error":
        logger.error("Text model '%s' does not fit in memory: %s; %s.", text_model, detail, advice)
        return 2
    if verdict == "warn":
        logger.warning(
            "Text model '%s' is large for this machine: %s; the vision model is unloaded "
            "before the fact check unless --keep-models-loaded. If it is too slow, %s.",
            text_model, detail, advice,
        )
    elif verdict == "unknown" or size is None:
        logger.debug("Text model '%s': %s", text_model, detail)
    else:
        logger.debug("Text model '%s' (%s): %s", text_model, format_gb(size), detail)
    return 0


# ---------------------------------------------------------------------------
# Enrichment (web research + fact check)
# ---------------------------------------------------------------------------


def _web_skip_reason(settings: Settings) -> str:
    """Why this run has no web researcher (recorded in every bottle's web research)."""
    if not settings.web.enabled:
        return "web lookups disabled (--no-web or web.enabled = false)"
    return settings.web.provider_reason or "no web search provider configured"


def _build_researcher(settings: Settings, client: Any) -> Any:  # noqa: ANN401
    """The run's web researcher, or None when web lookups are off or no provider resolves.

    One fetcher and one cache per run, so the per-host rate limits and the
    request budget are shared by every bottle. API keys are read from the
    environment by the provider and never logged.
    """
    web = settings.web
    if not web.enabled or (web.provider_resolved or "none") == "none":
        return None
    import requests

    from image_processor.vendor_image import make_vision_verifier
    from image_processor.web_research import WebOptions, WebResearcher
    from lib.search_providers import build_provider
    from lib.web_cache import WebCache, default_cache_dir
    from lib.web_fetch import HttpFetcher

    cache = WebCache(Path(web.cache_dir) if web.cache_dir else default_cache_dir(os.environ), refresh=web.refresh)
    provider = build_provider(
        web.provider_resolved,
        os.environ,
        requests.Session(),
        searxng_url=web.searxng_url or "",
        cache=cache,
        cache_ttl_hours=web.search_cache_ttl_hours,
    )
    if provider is None:
        return None
    fetcher = HttpFetcher(
        session=requests.Session(),
        cache=cache,
        timeout=(5, web.fetch_timeout_seconds),
        max_page_bytes=web.max_page_bytes,
        max_image_bytes=web.max_image_bytes,
        per_host_interval=web.per_host_interval_seconds,
        max_requests=web.max_requests_per_run,
        refresh=web.refresh,
        cache_ttl_hours=web.cache_ttl_hours,
    )
    options = WebOptions(
        max_queries=web.max_queries,
        max_results=web.max_results,
        max_pages=web.max_pages,
        max_llm_pages=web.max_llm_pages,
        deny_domains=tuple(web.deny_domains),
        allow_domains=tuple(web.allow_domains),
        vendor_images=web.vendor_images,
    )
    verifier = (
        make_vision_verifier(client, settings.model.vision_model, read_timeout=settings.server.request_timeout)
        if web.vendor_images
        else None
    )
    return WebResearcher(
        provider=provider,
        fetcher=fetcher,
        options=options,
        provider_reason=web.provider_reason,
        vendor_verifier=verifier,
    )


def _build_fact_checker(settings: Settings, client: Any, selector: Any) -> FactChecker:  # noqa: ANN401
    """The run's fact checker (deterministic, label/OCR/user facts only, when the text stage is off)."""
    text = settings.text

    def record(model: str, duration: float, success: bool, think: str | None) -> None:
        if selector is not None and hasattr(selector, "record_usage"):
            selector.record_usage(
                ModelStage.METADATA_GENERATION, model, success=success, duration=duration, think=think
            )

    return FactChecker(
        client if text.active else None,
        model=text.model,
        think_extract=text.think_extract,
        think_compose=text.think_compose,
        num_ctx=text.num_ctx,
        timeout_s=text.timeout_seconds,
        max_attempts=min(2, settings.retry.policy().max_attempts),
        keep_alive="30m" if settings.model.keep_models_loaded else "10m",
        usage_recorder=record,
    )


def build_enricher(ctx: RunContext, client: Any) -> Enricher:  # noqa: ANN401
    """The enrichment phase for this run (web research, unloads, fact check).

    A researcher that cannot be built is logged and the run continues with
    web research ``skipped`` (web lookups never fail a run).
    """
    settings = ctx.settings
    skip_reason = _web_skip_reason(settings)
    try:
        researcher = _build_researcher(settings, client)
    except Exception as e:
        logger.error("Web lookups are off for this run: the researcher could not be built: %s", e)
        researcher, skip_reason = None, f"web researcher could not be built: {type(e).__name__}"
    manager = ctx.server_manager
    return Enricher(
        researcher=researcher,
        fact_checker=_build_fact_checker(settings, client, ctx.model_selector),
        server_manager=manager,
        vision_model=settings.model.vision_model,
        text_model=settings.text.model if settings.text.active else None,
        keep_models_loaded=settings.model.keep_models_loaded,
        workers=settings.concurrency,
        unload=getattr(manager, "unload_model", None),
        web_skip_reason=skip_reason,
    )


def _same_directory(a: Any, b: Any) -> bool:  # noqa: ANN401
    """True if both paths name the same directory (False if unresolvable).

    Existing paths are compared by filesystem identity (``os.path.samefile``),
    which also catches names that differ only in letter case on
    case-insensitive filesystems such as the macOS default; otherwise the
    resolved paths are compared.
    """
    try:
        if Path(a).exists() and Path(b).exists():
            return os.path.samefile(a, b)
        return Path(a).resolve() == Path(b).resolve()
    except (TypeError, ValueError, OSError):
        return False


def _is_ancestor(ancestor: Any, path: Any) -> bool:  # noqa: ANN401
    """True if directory ``ancestor`` strictly contains ``path`` (False if unresolvable).

    Compared by filesystem identity, so letter-case differences on
    case-insensitive filesystems and symlinked paths still match.
    """
    try:
        if not Path(ancestor).exists():
            return False
        return any(
            os.path.samefile(parent, ancestor) for parent in Path(path).resolve().parents
        )
    except (TypeError, ValueError, OSError):
        return False


def _check_output_dir(ctx: RunContext) -> int:
    """Check the output directory is writable; see :func:`pre_flight_checks`.

    A dry run writes nothing: it checks that the nearest existing ancestor
    of the output directory is writable and searchable. A real run creates
    the directory, probes it with a uniquely named temporary file and marks
    it as output (unless it contains the photos).
    """
    output_dir = Path(ctx.settings.output_dir)
    if ctx.settings.dry_run:
        target = existing_ancestor(output_dir)
        if not target.is_dir() or not os.access(target, os.W_OK | os.X_OK):
            logger.error("Output directory is not writable: %s (checked %s)", output_dir, target)
            return 5
        logger.info("Dry run: output directory %s would be writable (checked %s)", output_dir, target)
        return 0
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=output_dir, prefix=".write_test_"):
            pass
        if ctx.mark_output_dir:
            (output_dir / OUTPUT_MARKER).touch()
        logger.info("Output directory is writable: %s", output_dir)
    except Exception:
        logger.exception("Output directory is not writable: %s", output_dir)
        return 5
    return 0


def _load_hints(ctx: RunContext) -> int:
    """Load ``settings.hints_file`` into ``ctx.hints``; 1 (logged) if the file is invalid."""
    path = ctx.settings.hints_file
    if path is None:
        ctx.hints = HintBook.EMPTY
        return 0
    try:
        ctx.hints = HintBook.load(path)
    except SettingsError as e:
        logger.error("Invalid hints file:\n%s", e)
        return 1
    return 0


def pre_flight_checks(ctx: RunContext) -> int:
    """Verify prerequisites before processing.

    Setting values were validated by :func:`lib.settings.resolve_settings`;
    this checks the filesystem, the hints file, Tesseract and the model
    server, cheap checks first. On success ``ctx.server_manager``,
    ``ctx.model_selector`` and ``ctx.hints`` are set. In a dry run the server
    is only probed: nothing is started, pulled or loaded.

    Returns:
        Exit code: 0 if checks pass, 1 (output directory equal to the input
        folder, invalid hints file, Tesseract missing), 2 (model server or
        model unavailable, text model larger than memory), 3 (no images),
        5 (output directory not writable).
    """
    settings = ctx.settings
    image_folder = Path(settings.image_folder)
    output_dir = Path(settings.output_dir)

    if not image_folder.exists():
        logger.error("Input folder does not exist: %s", image_folder)
        return 3
    if not image_folder.is_dir():
        logger.error("Input path is not a directory: %s", image_folder)
        return 3

    # Outputs written into the input folder itself would be re-processed.
    if _same_directory(output_dir, image_folder):
        logger.error(
            "--output-dir must not be the same directory as --image-folder "
            "(outputs would be re-processed): %s",
            output_dir,
        )
        return 1

    # Count exactly what the scan will process (earlier outputs, hidden
    # folders and the output directory are left out).
    image_files = ImageOrganizer().iter_image_paths(image_folder, recursive=True, exclude=[output_dir])
    if not image_files:
        logger.error("No supported images found in folder: %s", image_folder)
        logger.info("Supported formats: %s", ", ".join(sorted(ImageOrganizer.SUPPORTED_EXTENSIONS)))
        return 3

    logger.info("Found %s images in %s", len(image_files), image_folder)

    # A bad hints file stops the run before any model work.
    exit_code = _load_hints(ctx)
    if exit_code:
        return exit_code

    # An output directory that contains the photos must not be marked as
    # output: later scans of it (or of its parents) would skip the photos.
    ctx.mark_output_dir = not _is_ancestor(output_dir, image_folder)
    if not ctx.mark_output_dir:
        logger.warning(
            "--output-dir %s contains --image-folder %s; outputs are written next to your photos "
            "(each bottle folder is marked so later scans skip it). Prefer an output directory "
            "outside the photo folder.",
            output_dir, image_folder,
        )

    exit_code = _check_output_dir(ctx)
    if exit_code:
        return exit_code

    try:
        import pytesseract

        version = pytesseract.get_tesseract_version()
        logger.info("Tesseract OCR version: %s", version)
    except Exception:
        logger.exception("Tesseract OCR is not installed or not found in PATH")
        logger.info("Install Tesseract OCR: https://github.com/tesseract-ocr/tesseract")
        return 1

    # Model server last: it is the slow check and may start a process.
    return _check_model_server(ctx)


def _check_model_server(ctx: RunContext) -> int:
    """Pre-flight step for the model server; see :func:`pre_flight_checks`."""
    settings = ctx.settings
    try:
        manager = build_server_manager(settings)
    except Exception as e:
        logger.error("Failed to configure model server: %s", e)
        return 2

    ctx.server_manager = manager
    required = _required_models(settings)
    required_names = ", ".join(name for name, _ in required)
    stages = active_stages(settings.text.active)
    model_checks: dict[str, Any] = {
        "allow_pull": settings.model.pull,
        "min_free_bytes": settings.model.min_free_bytes,
        # Only the vision model is loaded at pre-flight: a text model loaded
        # now would sit beside it through the whole vision phase.
        "warm_models": [settings.model.vision_model],
    }

    def _fail(message: str, *params: object) -> int:
        logger.error(message, *params)
        manager.close()
        ctx.server_manager = None
        return 2

    if settings.dry_run:
        logger.info("Dry run: active stages: %s", ", ".join(sorted(s.name for s in stages)))
        logger.info(
            "Dry run: text model: %s (fact check %s)",
            settings.text.model, "on" if settings.text.active else "off",
        )
        logger.info("Dry run: web search provider: %s", settings.web.provider_resolved)
        state = manager.probe()
        if not state.is_running:
            allowed, why = manager.server.can_start()
            if not allowed:
                return _fail("Dry run: Ollama at %s is not running and %s", state.base_url, why)
            logger.warning(
                "Dry run: Ollama at %s is not running (would be started); skipping model checks",
                state.base_url,
            )
            if settings.text.active:
                logger.info("Dry run: RAM check of text model %s: not checked (server not running)",
                            settings.text.model)
        else:
            logger.info("Dry run: Ollama at %s is reachable", state.base_url)
            if not manager.ensure_models(required, dry_run=True, warm=False, **model_checks):
                return _fail("Required models unavailable: %s (see the errors above)", required_names)
            if settings.text.active:
                verdict, detail, _ = _text_model_ram_verdict(manager, settings.text.model)
                logger.info("Dry run: RAM check of text model %s: %s (%s)", settings.text.model, verdict, detail)
                if _text_model_memory_check(manager, settings.text.model):
                    return _fail("Text model %s does not fit in memory", settings.text.model)
    else:
        if not manager.ensure_running():
            return _fail("Model server unavailable: %s", manager.last_error or "health check failed")
        logger.info("Model server is reachable at %s", manager.server.base_url)
        if not manager.ensure_models(required, dry_run=False, warm=True, **model_checks):
            return _fail("Required models unavailable: %s (see the errors above)", required_names)
        if settings.text.active and _text_model_memory_check(manager, settings.text.model):
            return _fail("Text model %s does not fit in memory", settings.text.model)

    selector = _model_selector(settings)
    if settings.debug:
        selector.print_stage_report(stages)
    ctx.model_selector = selector
    return 0


# ---------------------------------------------------------------------------
# process
# ---------------------------------------------------------------------------


def process_command(target: "Settings | RunContext | argparse.Namespace") -> int:
    """Process bourbon bottle images from the input folder.

    The usage summary is always logged and the model server manager is always
    closed (stopping the server only if this run started it).

    Args:
        target: The run's Settings, or a prepared RunContext.

    Returns:
        Exit code: 0 (success), 1 (general error / some bottles failed),
                   2 (model server or model unavailable), 3 (no valid images found),
                   4 (all bottles failed), 5 (output directory creation failed).
    """
    ctx = _as_context(target)
    try:
        exit_code = pre_flight_checks(ctx)
        if exit_code:
            return exit_code
        if ctx.settings.hints_file is not None and ctx.hints.path is None:
            exit_code = _load_hints(ctx)  # pre-flight normally loaded it already
            if exit_code:
                return exit_code
        if ctx.server_manager is None:
            ctx.server_manager = build_server_manager(ctx.settings)
        if ctx.model_selector is None:
            ctx.model_selector = _model_selector(ctx.settings)
        return _run_pipeline(ctx)
    finally:
        _log_usage_summary(ctx.model_selector)
        if ctx.server_manager is not None:
            try:
                ctx.server_manager.close()
            except Exception:
                logger.warning("Error while shutting down the model server", exc_info=True)


def _tool_version() -> str:
    try:
        from importlib.metadata import version

        return version("bourbon-image-processor")
    except Exception:
        return "unknown"


def _write_run_manifest(ctx: RunContext) -> Path | None:
    """Write ``<output_dir>/run_<run_id>_settings.json`` (never overwritten; no API keys).

    A taken name gets a ``_N`` suffix, which also becomes part of ``ctx.run_id``.
    Returns the path, or None if it could not be written (logged).
    """
    settings = ctx.settings
    base_id = ctx.run_id
    try:
        Path(settings.output_dir).mkdir(parents=True, exist_ok=True)
    except OSError:
        logger.exception("Could not create the output directory %s", settings.output_dir)
        return None
    for n in range(1, 100):
        run_id = base_id if n == 1 else f"{base_id}_{n}"
        path = Path(settings.output_dir) / f"run_{run_id}_settings.json"
        manifest = {
            "run_id": run_id,
            "tool": "bourbon-image-processor",
            "tool_version": _tool_version(),
            "created_at": datetime.now().astimezone().isoformat(),
            "models": {
                "vision": settings.model.vision_model,
                "text": settings.text.model if settings.text.active else None,
            },
            **settings_to_dict(settings),
        }
        try:
            with path.open("x", encoding="utf-8") as handle:
                json.dump(manifest, handle, indent=2)
                handle.write("\n")
        except FileExistsError:
            continue
        except OSError:
            logger.exception("Could not write the run settings file %s", path)
            return None
        ctx.run_id = run_id
        logger.info("Run settings recorded in %s", path)
        return path
    logger.error("Could not find a free run settings file name for run %s", base_id)
    return None


def _run_pipeline(ctx: RunContext) -> int:
    """Scan, group, analyze and save bottles. Does not close the server manager.

    The scan skips the output directory and earlier outputs. Each bottle is
    saved to its own new folder (see ``PreviewGenerator.save_bottle_files``);
    a bottle that fails analysis or saving, or is rejected in interactive
    mode, gets an ``error_report_<bottle_id>.json`` that ``retry`` can read.

    Returns:
        Exit code: 0 (no failures), 1 (some failed), 2 (model server
        unavailable), 3 (no images), 4 (all failed).
    """
    settings = ctx.settings
    image_organizer = ImageOrganizer(
        gap_seconds=settings.grouping.gap_seconds,
        max_group_size=settings.grouping.max_group_size,
    )

    # Scan images, leaving out earlier outputs (the output dir may sit inside
    # the input folder, e.g. the default `-f . -o processed`).
    images = image_organizer.scan_images(settings.image_folder, recursive=True, exclude=[settings.output_dir])
    if not images:
        logger.error(
            "No supported images found in %s (earlier outputs and hidden folders are skipped)",
            settings.image_folder,
        )
        return 3

    logger.info("Processing %s images...", len(images))

    # Group by bottle (EXIF capture time; cheap, so a dry run shows it too).
    image_groups = image_organizer.group_by_bottle(images)
    _assign_bottle_ids(image_groups)
    logger.info("Found %s bottle groups", len(image_groups))

    if settings.dry_run:
        logger.info("Dry-run mode: would process images but not execute")
        for group in image_groups:
            logger.info("  %s: %s", group_bottle_id(group), [str(img.path) for img in group])
        return 0

    try:
        client = ctx.server_manager.get_client()
    except ModelServerError as e:
        logger.error("Model server unavailable: %s", e)
        return 2

    ctx.run_id = ctx.run_id or datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    _write_run_manifest(ctx)

    executor = None
    if settings.concurrency > 1:
        executor = ThreadPoolExecutor(max_workers=settings.concurrency, thread_name_prefix="bottle")
        logger.info(
            "Analysing up to %d bottle groups in parallel; parallel vision requests depend on the "
            "server's OLLAMA_NUM_PARALLEL (extra requests queue, and the wait counts toward "
            "--request-timeout)",
            settings.concurrency,
        )
    batch_processor = None
    try:
        bottle_analyzer = BottleAnalyzer(
            ollama_client=client,
            model_selector=ctx.model_selector,
            retry_policy=settings.retry.policy(),
        )
        batch_processor = BatchProcessor(
            batch_size=settings.batch_size,
            bottle_analyzer=bottle_analyzer,
            split_budget_ratio=settings.grouping.split_budget_ratio,
            max_group_size=settings.grouping.max_group_size,
            executor=executor,
            hints=ctx.hints,
        )
        enricher = build_enricher(ctx, client)
        exit_code = _process_batches(ctx, image_groups, bottle_analyzer, batch_processor, enricher)
    except BaseException:
        if batch_processor is not None:
            batch_processor.cancel.set()
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)
            executor = None
        raise
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
    return exit_code


def _process_batches(
    ctx: RunContext,
    image_groups: list,
    bottle_analyzer: BottleAnalyzer,
    batch_processor: BatchProcessor,
    enricher: Enricher | None = None,
) -> int:
    """Run every batch: analysis, quantity, hint overrides, enrichment, review, previews, saving."""
    settings = ctx.settings
    preview_generator = PreviewGenerator(output_dir=settings.output_dir, mark_output_dir=ctx.mark_output_dir)
    totals = _RunTotals()
    seen_ids: set[str] = {group_bottle_id(group) for group in image_groups if group}

    # One split-check budget for the whole run, shared by all batches.
    batch_processor.start_run(image_groups)
    # Groups are complete before batching (capture-time grouping), so each
    # batch is processed as sliced: no group is ever moved or seen twice.
    batches = batch_processor.slice_into_batches(image_groups)

    for batch_idx, batch in enumerate(batches, 1):
        logger.info("Processing batch %s/%s (%s image groups)", batch_idx, len(batches), len(batch))

        # Fatal model errors (model unusable, server gone) would fail every
        # remaining bottle the same way: abort the run.
        try:
            batch_result = batch_processor.process_batch(batch, vision_model=settings.model.vision_model)
        except FATAL_MODEL_ERRORS as e:
            logger.error("Aborting: %s", e)
            return 2

        # Quantity from each bottle's own image group (duplicate photos).
        path_to_image = {str(img.path): img for group in batch for img in group}
        for bottle_dict in batch_result.bottle_dicts:
            group = [
                path_to_image[p]
                for p in map(str, bottle_dict.get("images") or [])
                if p in path_to_image
            ]
            bottle_dict["quantity"] = max(1, bottle_analyzer.detect_quantity(group)) if group else 1

        # Hint field values: sourced human corrections, before review and saving.
        for bottle_dict, metadata in zip(batch_result.bottle_dicts, batch_result.metadata_list, strict=True):
            seen_ids.add(PreviewGenerator.bottle_id_for(bottle_dict, metadata))
            bottle_dict["run_id"] = ctx.run_id
        apply_hint_overrides(
            batch_result.bottle_dicts, batch_result.metadata_list, ctx.hints, settings.hints_file
        )

        # Web research and fact check (never fails a bottle).
        if enricher is not None and batch_result.bottle_dicts:
            try:
                enricher.enrich_batch(batch_result.bottle_dicts, batch_result.metadata_list, run_id=ctx.run_id)
            except Exception:
                logger.exception("Enrichment failed for batch %s; bottles are saved without facts", batch_idx)

        # Interactive mode: corrections, conflicts, approvals
        rejected: set[str] = set()
        if settings.interactive:
            rejected = _interactive_review(preview_generator, batch_result)

        # Batch preview report and JSON (feedback reads the JSON), after
        # review so they hold the corrected values and every rejected bottle.
        try:
            preview_path = preview_generator.save_batch_preview(
                batch_result.bottle_dicts,
                batch_result.metadata_list,
                batch_id=f"{ctx.run_id}_{batch_idx:03d}",
                rejected_bottles=list(batch_result.failed_bottles),
            )
            logger.info("Saved batch preview report to %s", preview_path)
        except Exception:
            logger.exception("Failed to generate batch preview report")

        _save_batch_bottles(batch_result, rejected, preview_generator, bottle_analyzer, totals)
        _report_failed_bottles(batch_result, batch, rejected, preview_generator, totals)

    for unused in ctx.hints.unused(seen_ids):
        logger.warning(
            "Hint for bottle %s matched no bottle in this run (check the bottle ID in %s)",
            unused, settings.hints_file,
        )

    # Completion reporting
    logger.info("=" * 60)
    logger.info("Processing Complete")
    logger.info("=" * 60)
    logger.info("Processed successfully: %s", totals.processed)
    logger.info("Failed: %s", totals.failed)
    logger.info("Error reports generated: %s", len(totals.error_reports))
    _log_enrichment_summary(settings, enricher, totals)
    logger.info("%s", batch_processor.split_summary())
    nudge = batch_processor.group_size_nudge()
    if nudge:
        logger.warning("%s", nudge)
    logger.info("Output directory: %s", settings.output_dir)
    for folder in totals.saved_folders:
        logger.info("  Saved: %s", folder)
    for report in totals.error_reports:
        logger.info("  Error report: %s", report)

    if totals.failed == 0:
        return 0
    if totals.processed == 0:
        logger.error("All bottles failed to process")
        return 4
    logger.warning("Some bottles failed (%s/%s)", totals.failed, totals.failed + totals.processed)
    return 1


def _interactive_review(preview_generator: PreviewGenerator, batch_result: Any) -> set[str]:  # noqa: ANN401
    """Interactive review of a batch: apply edits and conflict choices, re-reconcile, record approvals.

    Returns the rejected bottle IDs (also added to the batch's failed
    bottles). If the review itself fails (e.g. end of input), nothing from
    it is applied: no bottle is approved.
    """
    try:
        correction_dicts, rejected_ids, approvals = preview_generator.interactive_prompt(
            batch_result.bottle_dicts, batch_result.metadata_list
        )
    except Exception:
        logger.exception("Interactive review failed; bottles are kept without approval")
        return set()

    if correction_dicts:
        _apply_review_corrections(
            batch_result.bottle_dicts, batch_result.metadata_list, _corrections_from_prompt(correction_dicts),
            "interactive feedback",
        )
    _apply_approvals(batch_result.bottle_dicts, batch_result.metadata_list, approvals, "interactive")

    rejected = set(rejected_ids)
    batch_result.failed_bottles.extend(rejected_ids)
    if rejected_ids:
        logger.info("Rejected %d bottles from interactive feedback", len(rejected_ids))
    return rejected


def _apply_review_corrections(
    bottle_dicts: list[dict], metadata_list: list, corrections: list[Correction], what: str
) -> tuple[list[tuple[Correction, str]], set[int]]:
    """Apply corrections by bottle ID, then fold them into each touched bottle's facts.

    Returns:
        ``(unapplied, touched)``: corrections not applied (logged, with
        reasons) and the indexes of bottles that received a correction.
    """
    before = _correction_counts(bottle_dicts)
    _, unapplied = apply_corrections(bottle_dicts, metadata_list, corrections)
    _log_unapplied(unapplied)
    logger.info("Applied %d of %d corrections from %s", len(corrections) - len(unapplied), len(corrections), what)
    touched = {
        i for i, d in enumerate(bottle_dicts)
        if isinstance(d, dict) and len(d.get("corrections") or []) > (before[i] if i < len(before) else 0)
    }
    for index in sorted(touched):
        try:
            refresh_facts(bottle_dicts[index])
        except Exception:
            logger.exception("Could not re-reconcile the facts of bottle %s", bottle_dicts[index].get("bottle_id", "?"))
    return unapplied, touched


def _apply_approvals(
    bottle_dicts: list[dict], metadata_list: list, approvals: Any, by: str  # noqa: ANN401
) -> tuple[set[int], list[str]]:
    """Record reviewer approvals in the bottles' facts (gate recomputed).

    Returns:
        ``(approved, problems)``: indexes of approved bottles, and one message
        per approval that could not be recorded (unknown ID, no facts).
    """
    index_by_id: dict[str, int] = {}
    duplicate_ids: set[str] = set()
    for i, d in enumerate(bottle_dicts):
        key = PreviewGenerator.bottle_id_for(d, metadata_list[i] if i < len(metadata_list) else None)
        if key in index_by_id:
            duplicate_ids.add(key)
        index_by_id[key] = i
    approved: set[int] = set()
    problems: list[str] = []
    for bottle_id in approvals or []:
        if str(bottle_id) in duplicate_ids:
            # Same rule as apply_corrections: an ambiguous ID approves nothing.
            problems.append(f"bottle ID {bottle_id} is not unique in this batch")
            continue
        index = index_by_id.get(str(bottle_id))
        if index is None:
            problems.append(f"no bottle with ID {bottle_id}")
            continue
        raw = bottle_dicts[index].get("facts")
        if not raw:
            problems.append(f"bottle {bottle_id} has no fact sheet to approve")
            continue
        try:
            sheet = approve_facts(FactSheet.from_dict(raw), by)
        except Exception as e:
            problems.append(f"bottle {bottle_id}: fact sheet could not be read ({e})")
            continue
        bottle_dicts[index]["facts"] = sheet.to_dict()
        approved.add(index)
        state = "publish-ready" if sheet.publish.get("ready") else (
            "still blocked: " + "; ".join(sheet.publish.get("blockers") or [])
        )
        logger.info("Bottle %s approved by %s (%s)", bottle_id, by, state)
    for problem in problems:
        logger.error("Approval not recorded: %s", problem)
    return approved, problems


def _log_enrichment_summary(settings: Settings, enricher: Enricher | None, totals: "_RunTotals") -> None:
    """Run summary lines: provider, web and verification status counts, publish-ready count."""
    if enricher is not None and enricher.provider_name:
        provider = enricher.provider_name
    elif not settings.web.enabled:
        provider = "none (web lookups off)"
    else:
        provider = f"none ({_web_skip_reason(settings)})"
    logger.info("Web search provider: %s", provider)

    def counts(c: dict[str, int]) -> str:
        return ", ".join(f"{k}={v}" for k, v in sorted(c.items())) or "none"

    logger.info("Web research: %s", counts(totals.web_status))
    logger.info("Fact check (%s): %s", settings.text.model if settings.text.active else "off", counts(totals.verification))
    logger.info(
        "Publish-ready: %d of %d saved bottle(s)%s",
        totals.ready, totals.processed,
        "" if totals.ready == totals.processed else
        " (each needs a person's approval: --interactive [A]ccept or feedback --approve <bottle_id>)",
    )


@dataclass
class _RunTotals:
    """Outcome counters and written paths for one ``process`` run."""

    processed: int = 0
    failed: int = 0
    saved_folders: list[Path] = dataclass_field(default_factory=list)
    error_reports: list[Path] = dataclass_field(default_factory=list)
    web_status: dict[str, int] = dataclass_field(default_factory=dict)
    verification: dict[str, int] = dataclass_field(default_factory=dict)
    ready: int = 0

    def count_bottle(self, bottle: Any) -> None:  # noqa: ANN401
        """Count a saved bottle's web, verification and publish status."""
        web = getattr(bottle, "web_research", None)
        key = web.status if web is not None else "not run"
        self.web_status[key] = self.web_status.get(key, 0) + 1
        facts = getattr(bottle, "facts", None)
        key = facts.verification.status if facts is not None else "not run"
        self.verification[key] = self.verification.get(key, 0) + 1
        if facts is not None and facts.publish.get("ready"):
            self.ready += 1


def _assign_bottle_ids(groups: list) -> None:
    """Give every group a run-unique bottle ID (``Image.bottle_ref``).

    The ID is the first photo's file stem; stems shared by groups (same name
    in different folders) get a short hash of the photo's path appended, so
    failure reports and interactive rejections never mix bottles up.
    """
    stems = [Path(group[0].path).stem for group in groups if group]
    counts: dict[str, int] = {}
    for stem in stems:
        counts[stem] = counts.get(stem, 0) + 1
    for group in groups:
        if not group:
            continue
        first = Path(group[0].path)
        bottle_id = first.stem
        if counts[bottle_id] > 1:
            digest = hashlib.sha1(str(first.resolve()).encode("utf-8"), usedforsecurity=False).hexdigest()[:8]
            bottle_id = f"{first.stem}_{digest}"
        for img in group:
            img.bottle_ref = bottle_id


def _write_failure_report(
    preview_generator: PreviewGenerator,
    totals: _RunTotals,
    bottle_id: str,
    images: list,
    error: BaseException | str,
    stage: str,
    retry_count: int = 0,
) -> None:
    """Write a retry-able failure report; a failure to write is only logged."""
    try:
        totals.error_reports.append(
            preview_generator.save_failure_report(
                bottle_id, [str(p) for p in images], error, stage=stage, retry_count=retry_count
            )
        )
    except Exception:
        logger.exception("Could not write the %s failure report for %s", stage, bottle_id)


def _save_batch_bottles(
    batch_result: Any,  # noqa: ANN401
    rejected: set[str],
    preview_generator: PreviewGenerator,
    bottle_analyzer: BottleAnalyzer,
    totals: _RunTotals,
) -> None:
    """Save every analyzed, non-rejected bottle of a batch to its own folder.

    Rejected bottles get a 'rejected' failure report (they are counted as
    failed with the batch's failed bottles). A bottle that cannot be saved is
    counted as failed and gets a 'save' failure report.
    """
    from models.bottle import Bottle

    for bottle_dict, metadata in zip(batch_result.bottle_dicts, batch_result.metadata_list, strict=False):
        bottle_id = PreviewGenerator.bottle_id_for(bottle_dict, metadata)
        images = list(bottle_dict.get("images") or metadata.source_images)
        if bottle_id in rejected:
            _write_failure_report(
                preview_generator, totals, bottle_id, images, "Rejected during interactive review", "rejected"
            )
            continue
        try:
            bottle = Bottle.from_analysis(
                bottle_dict,
                metadata,
                quantity=bottle_dict.get("quantity"),
                corrections=bottle_dict.get("corrections") or [],
            )
            extra = (
                {"vision_raw_response": bottle_dict["vision_raw_response"]}
                if "vision_raw_response" in bottle_dict
                else None
            )
            folder = preview_generator.save_bottle_files(
                bottle,
                metadata,
                model_used=bottle_dict.get("model_used"),
                confidence=bottle_dict.get("confidence") or bottle_analyzer.calculate_confidence(metadata),
                extra=extra,
            )
            logger.info("✓ Successfully processed: %s → %s", bottle.name, folder)
            totals.saved_folders.append(folder)
            totals.processed += 1
            totals.count_bottle(bottle)
        except Exception as exc:
            totals.failed += 1
            logger.exception(
                "✗ Failed to save bottle %s (%s): %s: %s",
                bottle_dict.get("name", "Unknown"), bottle_id, type(exc).__name__, exc,
            )
            _write_failure_report(preview_generator, totals, bottle_id, images, exc, "save")


def _report_failed_bottles(
    batch_result: Any,  # noqa: ANN401
    batch: list,
    rejected: set[str],
    preview_generator: PreviewGenerator,
    totals: _RunTotals,
) -> None:
    """Count a batch's failed bottles and write 'analysis' failure reports.

    Rejected bottles already have a report from :func:`_save_batch_bottles`.
    """
    groups_by_id = {group_bottle_id(group): group for group in batch if group}
    for failed_bottle_id in batch_result.failed_bottles:
        totals.failed += 1
        if failed_bottle_id in rejected:
            continue
        error = batch_result.errors.get(failed_bottle_id)
        if error:
            logger.warning("Bottle failed to process: %s (%s)", failed_bottle_id, describe_error(error))
        else:
            logger.warning("Bottle failed to process: %s", failed_bottle_id)
        group = groups_by_id.get(failed_bottle_id, [])
        _write_failure_report(
            preview_generator,
            totals,
            failed_bottle_id,
            [img.path for img in group],
            error or "analysis failed; see the log",
            "analysis",
            retry_count=max(0, getattr(error, "attempts", 1) - 1),
        )


def _log_usage_summary(selector: Any) -> None:  # noqa: ANN401
    """Log the per-stage/per-model usage summary if any model calls were made.

    Args:
        selector: The run's ModelSelector (or None).
    """
    if selector is None or not hasattr(selector, "get_usage_summary"):
        return
    try:
        summary = selector.get_usage_summary()
    except Exception:
        logger.debug("Could not compute the model usage summary", exc_info=True)
        return
    if not isinstance(summary, dict) or summary.get("total_calls", 0) <= 0:
        return
    logger.info("=" * 60)
    logger.info("Model Usage Summary")
    logger.info("=" * 60)
    logger.info("  Total calls:      %s", summary.get("total_calls", 0))
    logger.info("  Successful calls: %s", summary.get("successful_calls", 0))
    logger.info("  Total duration:   %.2fs", summary.get("total_duration_seconds", 0))
    by_stage = summary.get("by_stage", {})
    if by_stage:
        logger.info("  By stage:")
        for stage_name, stats in by_stage.items():
            logger.info("    %s: %s calls, %s success",
                        stage_name, stats.get("total", 0), stats.get("success", 0))
    by_model = summary.get("by_model", {})
    if by_model:
        logger.info("  By model:")
        for model_name, stats in by_model.items():
            logger.info("    %s: %s calls, %s success",
                        model_name, stats.get("total", 0), stats.get("success", 0))


def _correction_counts(bottle_dicts: list[dict]) -> list[int]:
    """Number of recorded corrections per bottle dict (before applying new ones)."""
    return [len(d.get("corrections") or []) if isinstance(d, dict) else 0 for d in bottle_dicts]


def _resave_bottles(
    preview_generator: PreviewGenerator,
    bottle_dicts: list[dict],
    metadata_list: list,
    indexes: set[int],
    rejected: set[str],
) -> tuple[int, list[str]]:
    """Save each listed bottle (corrected or approved) as a new output folder.

    Bottles listed as rejected (in the batch JSON or during this review) are
    left alone; existing folders are never overwritten
    (``save_bottle_files`` picks a new ``_N`` name).

    Returns:
        ``(resaved, failed)``: the number of bottles resaved, and the IDs of
        bottles whose new folder could not be written (so the caller can
        report that the change reached no output).
    """
    from models.bottle import Bottle

    resaved = 0
    failed: list[str] = []
    for index in sorted(indexes):
        bottle_dict = bottle_dicts[index]
        # A batch JSON without (enough) metadata still gets its bottles saved.
        metadata = metadata_list[index] if index < len(metadata_list) else None
        bottle_id = PreviewGenerator.bottle_id_for(bottle_dict, metadata)
        if bottle_id in rejected:
            logger.warning("Bottle %s is rejected, so its changes were not saved to any output", bottle_id)
            continue
        try:
            bottle = Bottle.from_analysis(
                bottle_dict, metadata, quantity=bottle_dict.get("quantity"),
                corrections=bottle_dict.get("corrections") or [],
            )
            output_folder = preview_generator.save_bottle_files(
                bottle, metadata, model_used=bottle_dict.get("model_used"), confidence=bottle_dict.get("confidence"),
            )
            logger.info(
                "Resaved bottle %s to %s (%s)", bottle_id, output_folder,
                "publish-ready" if bottle.publish_ready else "not publish-ready",
            )
            resaved += 1
        except Exception:
            logger.exception("Failed to resave bottle %s", bottle_id)
            failed.append(bottle_id)
    return resaved, failed


def _write_json_atomically(path: Path, data: Any) -> None:  # noqa: ANN401
    """Replace ``path`` with ``data`` as JSON (temp file in the same folder, then rename)."""
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _corrections_from_prompt(correction_dicts: list[dict]) -> list[Correction]:
    """Turn ``interactive_prompt`` correction dicts into Corrections (source 'interactive')."""
    return [
        Correction(
            bottle_id=cd["bottle_id"],
            field_name=cd["field_name"],
            original_value=cd["original_value"],
            corrected_value=cd["corrected_value"],
            source="interactive",
        )
        for cd in correction_dicts
    ]


def _log_unapplied(unapplied: list[tuple[Correction, str]]) -> None:
    """Log every correction that was not applied, with its reason."""
    for correction, reason in unapplied:
        logger.error(
            "Correction not applied (%s): bottle %s, %s %r -> %r",
            reason, correction.bottle_id, correction.field_name,
            correction.original_value, correction.corrected_value,
        )


def _load_corrections_file(path: Path) -> list[Correction]:
    """Read a corrections file: a JSON list of ``{bottle_id, field_name, original_value, corrected_value}``.

    Raises:
        ValueError: If the file is not such a list or an entry lacks a key.
        OSError, json.JSONDecodeError: If the file cannot be read or parsed.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(
            "corrections file must be a JSON list of "
            "{bottle_id, field_name, original_value, corrected_value} objects"
        )
    corrections = []
    for index, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"entry {index} is not an object")
        missing = [
            key for key in ("bottle_id", "field_name", "original_value", "corrected_value")
            if key not in item
        ]
        if missing:
            raise ValueError(f"entry {index} lacks {', '.join(missing)}")
        if item["corrected_value"] is None:
            # str(None) would write the literal text "None" into the bottle.
            raise ValueError(f"entry {index} has a null corrected_value")
        corrections.append(
            Correction.from_dict(
                {
                    **item,
                    "original_value": "" if item["original_value"] is None else str(item["original_value"]),
                    "corrected_value": str(item["corrected_value"]),
                    "source": f"feedback:{path}",
                }
            )
        )
    return corrections


def _schema_problem(data: Any, what: str) -> str | None:  # noqa: ANN401
    """Why ``data`` (a batch JSON or error report) cannot be read by this version, or None."""
    if not isinstance(data, dict):
        return f"{what} is not a JSON object"
    version = data.get("schema_version")
    if version != OUTPUT_SCHEMA_VERSION:
        return (
            f"{what} has schema_version {version!r}; this version reads only {OUTPUT_SCHEMA_VERSION}. "
            "It was written by an older bourbon-process (no facts, no bottle IDs on corrections); "
            "re-run `bourbon-process process` on the photos"
        )
    return None


def feedback_command(target: "Settings | RunContext | argparse.Namespace") -> int:
    """Apply corrections and approvals to a processed batch.

    Uses ``settings.output_dir``, ``batch``, ``corrections``, ``interactive``
    and ``approve``; no model, server or web setting. Corrections (from
    ``--corrections`` or interactive review) are applied by bottle ID only
    (:func:`image_processor.corrections.apply_corrections`) and become user
    facts: each touched bottle's facts are re-reconciled without any model
    call (a description citing a changed field is marked stale). ``--approve
    ID`` records a reviewer approval (``by='feedback'``) and recomputes the
    publish gate. Every changed bottle is saved again in a new folder, and
    the batch JSON is updated so a later feedback round starts from the
    corrected values. Only schema-version-2 batch files are accepted.

    Returns:
        Exit code: 0 (success), 1 (general error, an older batch file, or a
        correction or approval that was not applied), 2 (batch not found).
    """
    from models.bottle import ExtractedMetadata

    settings = _as_context(target).settings
    output_dir = Path(settings.output_dir)

    # Determine batch to load
    if settings.batch:
        batch_preview_file = output_dir / f"batch_{settings.batch}_preview.md"
        batch_json_file = output_dir / f"batch_{settings.batch}.json"
        if not batch_json_file.exists() and not batch_preview_file.exists():
            logger.error("Batch %s not found in %s", settings.batch, output_dir)
            return 2
    else:
        # Find most recent batch preview
        batch_preview_files = sorted(output_dir.glob("batch_*_preview.md"))
        if not batch_preview_files:
            logger.error("No batch preview files found in %s", output_dir)
            return 2
        batch_preview_file = max(batch_preview_files, key=lambda p: (p.stat().st_mtime, p.name))
        batch_json_file = batch_preview_file.with_name(
            batch_preview_file.name.replace("_preview.md", ".json")
        )

    logger.info("Using batch file: %s", batch_preview_file)

    if not batch_json_file.exists():
        logger.error("Batch data file not found: %s (it holds the bottles to correct)", batch_json_file)
        return 2
    try:
        batch_data = json.loads(batch_json_file.read_text(encoding="utf-8"))
    except Exception:
        logger.exception("Failed to load batch JSON file %s", batch_json_file)
        return 1
    problem = _schema_problem(batch_data, f"Batch file {batch_json_file}")
    if problem:
        logger.error("%s", problem)
        return 1
    try:
        bottle_dicts: list[dict] = list(batch_data.get("bottles") or [])
        previously_rejected = {str(r) for r in batch_data.get("rejected") or []}
        metadata_list = [ExtractedMetadata.from_dict(m) for m in batch_data.get("metadata") or []]
    except Exception:
        logger.exception("Failed to load batch JSON file %s", batch_json_file)
        return 1
    logger.info("Loaded %d bottles from batch JSON", len(bottle_dicts))

    corrections: list[Correction] = []
    rejected = set(previously_rejected)
    interactive_approvals: list[str] = []
    if settings.corrections:
        corrections_file = Path(settings.corrections)
        if not corrections_file.exists():
            logger.error("Corrections file not found: %s", corrections_file)
            return 1
        try:
            corrections = _load_corrections_file(corrections_file)
        except Exception as e:
            logger.error("Failed to load corrections file %s: %s", corrections_file, e)
            return 1
        logger.info("Loaded %d corrections from %s", len(corrections), corrections_file)
    elif settings.interactive:
        try:
            correction_dicts, rejected_bottles, interactive_approvals = PreviewGenerator(
                output_dir=output_dir
            ).interactive_prompt(bottle_dicts, metadata_list)
        except Exception:
            logger.exception("Interactive feedback failed")
            return 1
        corrections = _corrections_from_prompt(correction_dicts)
        rejected |= {str(r) for r in rejected_bottles or []}

    if not corrections and not settings.approve and not interactive_approvals:
        logger.info("No corrections or approvals given")
        return 0

    changed: set[int] = set()
    unapplied: list[tuple[Correction, str]] = []
    if corrections:
        unapplied, changed = _apply_review_corrections(bottle_dicts, metadata_list, corrections, "feedback")

    approved_interactive, problems_i = _apply_approvals(bottle_dicts, metadata_list, interactive_approvals, "interactive")
    approved_cli, problems = _apply_approvals(bottle_dicts, metadata_list, settings.approve, "feedback")
    changed |= approved_interactive | approved_cli
    problems += problems_i
    for index in sorted(approved_interactive | approved_cli):
        metadata = metadata_list[index] if index < len(metadata_list) else None
        bottle_id = PreviewGenerator.bottle_id_for(bottle_dicts[index], metadata)
        if bottle_id in rejected:
            problems.append(f"bottle {bottle_id} is rejected; its approval reaches no output")

    _, failed = _resave_bottles(PreviewGenerator(output_dir=output_dir), bottle_dicts, metadata_list, changed, rejected)
    if changed:
        batch_data["bottles"] = bottle_dicts
        batch_data["rejected"] = sorted(rejected)
        batch_data["updated_at"] = datetime.now().astimezone().isoformat()
        try:
            _write_json_atomically(batch_json_file, batch_data)
            logger.info("Updated %s with the applied changes", batch_json_file)
        except OSError:
            logger.exception("Could not update the batch file %s", batch_json_file)
            failed.append(batch_json_file.name)
    if unapplied:
        logger.error("%d correction(s) were not applied (see above)", len(unapplied))
    if problems:
        logger.error("%d approval(s) were not recorded (see above)", len(problems))
    if failed:
        logger.error(
            "Changed bottle(s) could not be saved, so their changes reached no output: %s",
            ", ".join(failed),
        )
    if unapplied or failed or problems:
        return 1
    logger.info("Feedback processing complete")
    return 0


def retry_command(target: "Settings | RunContext | argparse.Namespace") -> int:
    """Retry processing a failed bottle from its error report.

    Uses ``settings.report``, the server, model, retry and hints settings.

    Returns:
        Exit code: 0 (success), 1 (general error, invalid hints file or
        model server unavailable), 2 (report not found).
    """
    from models.image import Image

    ctx = _as_context(target)
    settings = ctx.settings
    report_path = Path(settings.report) if settings.report is not None else None

    # Check error report exists
    if report_path is None or not report_path.exists():
        logger.error("Error report not found: %s", report_path)
        return 2

    # Load error report
    try:
        with report_path.open("r", encoding="utf-8") as f:
            error_report = json.load(f)
    except Exception:
        logger.exception("Failed to read error report")
        return 1

    logger.info("Loading error report: %s", report_path)
    problem = _schema_problem(error_report, f"Error report {report_path}")
    if problem:
        logger.error("%s", problem)
        return 1

    # Validate error report structure
    required_fields = ["bottle_id", "images", "error_message"]
    for field in required_fields:
        if field not in error_report:
            logger.error("Missing required field in error report: %s", field)
            return 1

    bottle_id = error_report["bottle_id"]
    image_paths = error_report["images"]
    error_message = error_report["error_message"]
    suggested_fix = error_report.get("suggested_fix") or {}
    if not isinstance(bottle_id, str) or not bottle_id.strip():
        logger.error("Error report has no usable bottle_id: %r", bottle_id)
        return 1
    problem = _suggested_fix_problem(suggested_fix)
    if problem:
        # Refuse before any model work: a person's fix that cannot be applied
        # must not be silently dropped from the reprocessed bottle.
        logger.error("Invalid suggested_fix in %s: %s", report_path, problem)
        return 1
    if _load_hints(ctx):
        return 1

    logger.info("Retrying bottle: %s", bottle_id)
    logger.info("Error was: %s", error_message)
    if suggested_fix:
        logger.info("Suggested fix: %s", suggested_fix)

    # Convert image paths to Image objects
    images = []
    for img_path in image_paths:
        path = Path(img_path)
        if path.exists():
            images.append(Image(path=str(path)))
        else:
            logger.warning("Image not found: %s", img_path)

    if not images:
        logger.error("No valid images found in error report")
        return 1

    # Start (or adopt) the model server; always release it afterwards.
    try:
        server_manager = build_server_manager(settings)
    except Exception as e:
        logger.error("Failed to configure model server: %s", e)
        return 1
    ctx.server_manager = server_manager
    try:
        if not server_manager.ensure_running():
            logger.error(
                "Model server unavailable: %s", server_manager.last_error or "health check failed"
            )
            return 1
        return _retry_bottle(ctx, report_path, bottle_id, images, suggested_fix)
    finally:
        server_manager.close()


#: ``suggested_fix`` keys that are report metadata, not bottle fields.
_REPORT_METADATA_KEYS = frozenset({"bottle_id", "images", "error_message"})


def _suggested_fix_problem(suggested_fix: Any) -> str | None:  # noqa: ANN401
    """Why an error report's ``suggested_fix`` cannot be applied, or None if it can."""
    if not isinstance(suggested_fix, dict):
        return f"must be an object of field -> value, got {type(suggested_fix).__name__}"
    for name, value in suggested_fix.items():
        if name in _REPORT_METADATA_KEYS:
            continue
        if name not in CORRECTABLE_FIELDS:
            return f"{name!r} is not a correctable field (allowed: {', '.join(sorted(CORRECTABLE_FIELDS))})"
        if value is None:
            return f"{name!r} has a null value"
    return None


def _retry_bottle(
    ctx: RunContext,
    report_path: Path,
    bottle_id: str,
    images: list,
    suggested_fix: dict,
) -> int:
    """Re-run analysis for one bottle from an error report (server already up).

    The bottle goes through the same steps as in ``process``: OCR and the
    vision request under the retry policy (with the hint note), then the
    hint field values, then the report's ``suggested_fix``, then the
    enrichment phase (web research and fact check), then saving.

    Returns:
        Exit code: 0 (success) or 1 (failure).
    """
    from models.bottle import Bottle

    settings = ctx.settings
    try:
        client = ctx.server_manager.get_client()
    except ModelServerError as e:
        logger.error("Model server unavailable: %s", e)
        return 1

    bottle_analyzer = BottleAnalyzer(ollama_client=client, retry_policy=settings.retry.policy())
    fix_fields = {
        name: value
        for name, value in (suggested_fix or {}).items()
        if name not in _REPORT_METADATA_KEYS
    }

    try:
        bottle_dict, metadata = bottle_analyzer.process_single_bottle(
            images,
            vision_model=settings.model.vision_model,
            reviewer_note=ctx.hints.note_for(bottle_id),
        )
        bottle_dict["bottle_id"] = bottle_id

        _, unapplied_hints = apply_hint_overrides([bottle_dict], [metadata], ctx.hints, settings.hints_file)
        if unapplied_hints:
            logger.error("Not saving bottle %s: its hint values could not be applied", bottle_id)
            return 1

        # Apply the report's suggested_fix: a person wrote these values for
        # this bottle, so each is aimed at it by ID against the fresh reading.
        if fix_fields:
            corrections = [
                Correction(
                    bottle_id=bottle_id,
                    field_name=field_name,
                    original_value="" if bottle_dict.get(field_name) is None else str(bottle_dict.get(field_name)),
                    corrected_value=str(value),
                    source=f"feedback:{report_path}",
                )
                for field_name, value in fix_fields.items()
            ]
            _, unapplied = apply_corrections([bottle_dict], [metadata], corrections)
            if unapplied:
                _log_unapplied(unapplied)
                logger.error("Not saving bottle %s: its suggested_fix could not be applied", bottle_id)
                return 1
            logger.info("Applied %d corrections from suggested_fix", len(corrections))

        try:
            enricher = build_enricher(ctx, client)
            enricher.enrich_batch(
                [bottle_dict], [metadata],
                run_id=ctx.run_id or "retry_" + datetime.now().astimezone().strftime("%Y%m%d_%H%M%S"),
            )
        except Exception:
            logger.exception("Enrichment failed for bottle %s; it is saved without facts", bottle_id)

        bottle = Bottle.from_analysis(
            bottle_dict,
            metadata,
            quantity=max(1, bottle_analyzer.detect_quantity(images)),
            corrections=bottle_dict.get("corrections") or [],
        )

        # Save output files (a new folder; earlier output is never replaced)
        preview_generator = PreviewGenerator(output_dir=settings.output_dir)
        output_folder = preview_generator.save_bottle_files(
            bottle,
            metadata,
            model_used=bottle_dict.get("model_used"),
            confidence=bottle_dict.get("confidence"),
        )
        logger.info("Successfully reprocessed bottle, saved to: %s", output_folder)

        return 0

    except (VisionAnalysisError, *FATAL_MODEL_ERRORS) as e:
        logger.error("Failed to reprocess bottle %s: %s", bottle_id, e)
        return 1
    except Exception:
        logger.exception("Failed to reprocess bottle")
        return 1


def config_command(target: "Settings | RunContext | argparse.Namespace") -> int:
    """``config show [--json]`` (effective values and their sources) or ``config template``."""
    settings = _as_context(target).settings
    if settings.config_action == "template":
        sys.stdout.write(render_template())
        return 0
    print(describe(settings, as_json=settings.json))
    return 0


def _exit_on_signal(signum: int, _frame: Any) -> None:  # noqa: ANN401
    """Signal handler: exit with the conventional 128 + signal status."""
    sys.exit(128 + signum)


#: Command name -> entry point.
_COMMANDS = {
    "process": "process_command",
    "feedback": "feedback_command",
    "retry": "retry_command",
    "config": "config_command",
}


def main(argv: list | None = None) -> int:
    """Main entry point for CLI.

    Args:
        argv: Optional list of arguments (default: sys.argv[1:]).

    Returns:
        Exit code (1 for any configuration error, listed on stderr).
    """
    if not (sys.argv[1:] if argv is None else list(argv)):
        # A bare `bourbon-process` would otherwise process the current folder.
        build_parser().print_help()
        return 1
    args = parse_args(argv)

    try:
        settings = resolve_settings(args)
    except SettingsError as e:
        print("bourbon-process: configuration error:", file=sys.stderr)
        for error in e.errors:
            print(f"  - {error}", file=sys.stderr)
        return 1

    setup_logging(debug=settings.debug)
    if settings.debug:
        ignored = ignored_keys(settings)
        if ignored:
            logger.debug("Settings not used by '%s' (ignored): %s", settings.command, ", ".join(ignored))
    if settings.command in ("process", "retry") and settings.web.enabled and settings.web.provider_resolved == "none":
        logger.warning("Web lookups are on but skipped: %s", settings.web.provider_reason)

    # Turn SIGTERM, SIGHUP (terminal closed, SSH session dropped) and SIGQUIT
    # (Ctrl-backslash) into SystemExit so `finally` blocks run and an owned model
    # server, which runs in its own session and never sees these signals, is
    # stopped instead of orphaned.
    for name in ("SIGTERM", "SIGHUP", "SIGQUIT"):
        sig = getattr(signal, name, None)
        if sig is not None:
            signal.signal(sig, _exit_on_signal)

    command = globals().get(_COMMANDS.get(settings.command, ""))
    if command is None:
        logger.error("Unknown command: %s", settings.command)
        return 1
    return command(settings)


if __name__ == "__main__":
    sys.exit(main())
