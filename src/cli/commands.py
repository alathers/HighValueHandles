"""CLI commands for bourbon image processor.

Provides command-line interface for processing bourbon bottle images
using argparse for argument parsing and structured command execution.
"""

import argparse
import hashlib
import logging
import math
import os
import signal
import sys
import tempfile
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from urllib.parse import urlparse

from image_processor.batch_processor import (
    DEFAULT_SPLIT_BUDGET_RATIO,
    MAX_SPLIT_BUDGET_RATIO,
    BatchProcessor,
)
from image_processor.bottle_analyzer import BottleAnalyzer
from image_processor.capture_grouping import DEFAULT_GAP_SECONDS, DEFAULT_MAX_GROUP_SIZE
from image_processor.image_organizer import OUTPUT_MARKER, ImageOrganizer
from image_processor.preview_generator import PreviewGenerator
from lib.config import DEFAULT_VISION_MODEL
from lib.exceptions import FATAL_MODEL_ERRORS, ModelServerError, VisionAnalysisError
from lib.model_availability import DEFAULT_MIN_FREE_BYTES, GiB, existing_ancestor
from lib.model_selector import ModelSelector
from lib.model_server import (
    DEFAULT_OLLAMA_URL,
    DEFAULT_REQUEST_TIMEOUT,
    ModelServerManager,
    ServerBackend,
)
from models.image import group_bottle_id

logger = logging.getLogger(__name__)

#: Default of ``--min-free-gb`` (GiB that must stay free on the model store
#: after a pull).
DEFAULT_MIN_FREE_GB = DEFAULT_MIN_FREE_BYTES / GiB

#: Subcommands; an argument list naming none of them runs ``process``.
SUBCOMMANDS = ("process", "feedback", "retry")
#: Upper bounds of the grouping options.
MAX_GROUP_GAP_SECONDS = 600.0
MAX_GROUP_SIZE_LIMIT = 20
#: Logged when the deprecated ``--lookahead`` option is given.
LOOKAHEAD_DEPRECATION_WARNING = (
    "--lookahead is deprecated and ignored: photos are grouped by capture time, "
    "so a bottle never spans two batches"
)


def _backend_arg(value: str) -> str:
    """argparse type for --server-backend: a supported backend name.

    Raises:
        argparse.ArgumentTypeError: For an unknown or removed backend (argparse
            then exits with status 2).
    """
    try:
        return ServerBackend.parse(value).value
    except ValueError as e:
        raise argparse.ArgumentTypeError(str(e)) from None


def _add_server_args(parser: argparse.ArgumentParser) -> None:
    """Add the model and model-server options shared by `process` and `retry`."""
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default=None,
        help=(
            "Ollama vision model tag for analysis; must be vision-capable "
            f"(default: $BOURBON_MODEL, else {DEFAULT_VISION_MODEL})"
        ),
    )
    parser.add_argument(
        "-u",
        "--ollama-url",
        type=str,
        default=DEFAULT_OLLAMA_URL,
        help=f"Ollama service URL (default: {DEFAULT_OLLAMA_URL})",
    )
    parser.add_argument(
        "--server-backend",
        type=_backend_arg,
        default=os.environ.get("BOURBON_SERVER_BACKEND", "ollama"),
        metavar="{" + ",".join(b.value for b in ServerBackend) + "}",
        help="Model serving backend (default: $BOURBON_SERVER_BACKEND, else ollama)",
    )
    parser.add_argument(
        "--server-start-timeout",
        type=float,
        default=None,
        help="Seconds to wait for a model server this run starts (default: 30)",
    )
    parser.add_argument(
        "--request-timeout",
        "--vision-timeout",
        dest="request_timeout",
        type=float,
        default=float(DEFAULT_REQUEST_TIMEOUT),
        help=f"Per-request model timeout in seconds (default: {DEFAULT_REQUEST_TIMEOUT})",
    )
    parser.add_argument(
        "--keep-server",
        action="store_true",
        help="Leave a model server started by this run running afterwards",
    )
    parser.add_argument(
        "--no-server-autostart",
        action="store_true",
        help="Never start a model server; fail if none is running",
    )


def parse_args(argv: list | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Arguments that name no subcommand run ``process`` (``bourbon-process -f
    photos`` is ``bourbon-process process -f photos``).

    Args:
        argv: Optional list of arguments (default: sys.argv[1:]).

    Returns:
        Parsed arguments namespace.
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

    # Global options
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging and verbose output",
    )

    # --debug is also accepted after the subcommand. SUPPRESS keeps a missing
    # subcommand-level flag from overwriting the top-level value.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--debug",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Enable debug logging and verbose output",
    )

    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Process command (default behavior)
    process_parser = subparsers.add_parser(
        "process",
        parents=[common],
        help="Process bourbon bottle images and extract metadata",
    )

    # Input/output options for process command
    process_parser.add_argument(
        "-f",
        "--image-folder",
        type=Path,
        default=Path(),
        help="Folder containing images to process (default: current directory)",
    )
    process_parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("processed"),
        help="Output directory for processed bottles (default: processed/)",
    )

    # Processing options
    process_parser.add_argument(
        "-b",
        "--batch-size",
        type=str,
        default="5",
        help="Number of bottles to process per batch, or 'all' to process all (default: 5)",
    )
    process_parser.add_argument(
        "--lookahead",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Deprecated and ignored (photos are grouped by capture time, so a bottle never "
            "spans two batches); still accepted (0-10) so existing scripts keep working"
        ),
    )
    process_parser.add_argument(
        "--concurrency",
        type=int,
        default=3,
        help="Maximum concurrent operations (default: 3)",
    )
    process_parser.add_argument(
        "--group-gap-seconds",
        type=float,
        default=DEFAULT_GAP_SECONDS,
        metavar="SECONDS",
        help=(
            "Most seconds between consecutive photos of one bottle; photos are grouped by "
            f"EXIF capture time (default: {DEFAULT_GAP_SECONDS:g})"
        ),
    )
    process_parser.add_argument(
        "--group-max-size",
        type=int,
        default=DEFAULT_MAX_GROUP_SIZE,
        metavar="N",
        help=(
            "Most photos per bottle; longer runs of photos are split at their largest time gap, "
            f"0 = no limit (default: {DEFAULT_MAX_GROUP_SIZE})"
        ),
    )
    process_parser.add_argument(
        "--split-budget-ratio",
        type=float,
        default=DEFAULT_SPLIT_BUDGET_RATIO,
        metavar="RATIO",
        help=(
            "When the vision model says a group's photos show different products, the group is "
            "split at its largest time gap and each half is read again (2 extra analyses). "
            "This caps the extra analyses of the whole run at RATIO x number of groups (at least 4); "
            "an analysis is one vision request (retried once if unusable) plus OCR; "
            f"0 turns the split check off (default: {DEFAULT_SPLIT_BUDGET_RATIO:g}, "
            f"max {MAX_SPLIT_BUDGET_RATIO:g})"
        ),
    )

    # Model and model-server options
    _add_server_args(process_parser)
    process_parser.add_argument(
        "--pull",
        "--pull-missing",
        dest="pull",
        action="store_true",
        help=(
            "Download the vision model if it is not installed, after checking its download "
            "size against free disk space (off by default: models are never downloaded "
            "implicitly)"
        ),
    )
    process_parser.add_argument(
        "--min-free-gb",
        type=float,
        default=DEFAULT_MIN_FREE_GB,
        metavar="GB",
        help=(
            "Free space (GiB) to keep on the Ollama model store after a --pull; a pull "
            f"also keeps 10%% of its size free (default: {DEFAULT_MIN_FREE_GB:g})"
        ),
    )

    # Behavior options
    process_parser.add_argument(
        "-i",
        "--interactive",
        action="store_true",
        help="Enable interactive preview and review mode",
    )
    process_parser.add_argument(
        "-v",
        "--include-vendor-images",
        action="store_true",
        help="Include vendor/supply chain images in processing",
    )
    process_parser.add_argument(
        "-r",
        "--allow-retry",
        action="store_true",
        default=True,
        help="Allow automatic retry on processing failures (default: True)",
    )
    process_parser.add_argument(
        "--no-retry",
        action="store_false",
        dest="allow_retry",
        help="Disable automatic retry on processing failures",
    )

    # Debug options for process command
    process_parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Check inputs and probe the model server without starting it, pulling "
            "models or processing images"
        ),
    )

    # Feedback command
    feedback_parser = subparsers.add_parser(
        "feedback",
        parents=[common],
        help="Apply feedback and corrections to processed batch",
    )

    feedback_parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("processed"),
        help="Output directory containing processed batches (default: processed/)",
    )
    feedback_parser.add_argument(
        "--batch",
        type=str,
        help=(
            "Batch ID to apply feedback to, e.g. '20261122_193000_001' for "
            "batch_20261122_193000_001_preview.md (default: most recent batch)"
        ),
    )
    feedback_parser.add_argument(
        "--corrections",
        type=Path,
        help="Path to JSON file containing corrections",
    )
    feedback_parser.add_argument(
        "-i",
        "--interactive",
        action="store_true",
        help="Enable interactive feedback collection mode",
    )

    # Retry command
    retry_parser = subparsers.add_parser(
        "retry",
        parents=[common],
        help="Retry processing failed bottles from error reports",
    )

    retry_parser.add_argument(
        "--report",
        type=Path,
        required=True,
        help="Path to error_report.json file",
    )
    retry_parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("processed"),
        help="Output directory for processed bottles (default: processed/)",
    )
    _add_server_args(retry_parser)

    return parser


def _names_subcommand(argv: list[str]) -> bool:
    """True if ``argv`` names a subcommand, or asks for the top-level help.

    Top-level options take no values, so the first token not starting with
    '-' is the subcommand position.
    """
    for token in argv:
        if token in ("-h", "--help"):
            return True
        if not token.startswith("-"):
            return token in SUBCOMMANDS
    return False


def setup_logging(debug: bool = False) -> None:
    """Configure logging based on debug flag.

    Args:
        debug: Whether to enable DEBUG-level logging.
    """
    level = logging.DEBUG if debug else logging.INFO
    format_str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    logging.basicConfig(
        level=level,
        format=format_str,
        stream=sys.stdout,
    )
    # Third-party DEBUG output (PIL logs ~15 EXIF tags per image read) would
    # bury the pipeline's own messages.
    for noisy in ("PIL", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.INFO)


def resolve_vision_model(args: argparse.Namespace) -> str:
    """Return the vision model requested for this run.

    Precedence: ``-m/--model``, then ``$BOURBON_MODEL``, then the default.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Model name (an Ollama tag), sent to the client unchanged.
    """
    return getattr(args, "model", None) or os.environ.get("BOURBON_MODEL") or DEFAULT_VISION_MODEL


def build_server_manager(args: argparse.Namespace) -> ModelServerManager:
    """Create (but do not start) the model server manager selected by the CLI flags.

    Args:
        args: Parsed command-line arguments (missing attributes use defaults).

    Returns:
        A ModelServerManager for the selected backend.

    Raises:
        ValueError: If the backend name or a backend option is invalid.
    """
    backend = ServerBackend.parse(getattr(args, "server_backend", None) or "ollama")
    kwargs: dict[str, Any] = {}

    timeout = getattr(args, "request_timeout", None)
    if timeout is not None:
        kwargs["timeout"] = timeout
    startup_timeout = getattr(args, "server_start_timeout", None)
    if startup_timeout is not None:
        kwargs["startup_timeout"] = startup_timeout
    if getattr(args, "no_server_autostart", False):
        kwargs["auto_start"] = False

    kwargs["base_url"] = getattr(args, "ollama_url", None) or DEFAULT_OLLAMA_URL
    kwargs["model"] = resolve_vision_model(args)

    return ModelServerManager(
        backend=backend,
        keep_running=bool(getattr(args, "keep_server", False)),
        **kwargs,
    )


def _required_models(args: argparse.Namespace) -> list[str]:
    """Models the server must provide before processing starts.

    Only the resolved vision model is required: it is the only model the
    pipeline calls (``lib.model_selector.IMPLEMENTED_MODEL_STAGES``).
    Delegates to ``ModelSelector.required_models`` so the two never drift.
    Fallback and unimplemented-stage models are never checked or pulled.

    Args:
        args: Parsed command-line arguments.

    Returns:
        The model names to check, in order.
    """
    return ModelSelector.from_config(SimpleNamespace(model=resolve_vision_model(args))).required_models()


def _min_free_bytes(args: argparse.Namespace) -> int:
    """``--min-free-gb`` in bytes (validated by :func:`pre_flight_checks`)."""
    return int(getattr(args, "min_free_gb", DEFAULT_MIN_FREE_GB) * GiB)


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


def _check_output_dir(args: argparse.Namespace) -> int:
    """Check the output directory is writable; see :func:`pre_flight_checks`.

    A dry run writes nothing: it checks that the nearest existing ancestor
    of the output directory is writable and searchable. A real run creates
    the directory, probes it with a uniquely named temporary file and marks
    it as output (unless it contains the photos).
    """
    output_dir = Path(args.output_dir)
    if getattr(args, "dry_run", False):
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
        if getattr(args, "_mark_output_dir", True):
            (output_dir / OUTPUT_MARKER).touch()
        logger.info("Output directory is writable: %s", output_dir)
    except Exception:
        logger.exception("Output directory is not writable: %s", output_dir)
        return 5
    return 0


def pre_flight_checks(args: argparse.Namespace) -> int:
    """Verify prerequisites before processing.

    Cheap local checks run first; the model server is checked last. On
    success ``args._server_manager``, ``args._model_selector`` and the
    resolved ``args.model`` are set. In a dry run the server is only probed:
    nothing is started, pulled or loaded.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Exit code: 0 if checks pass, 1 (invalid argument, output directory
        equal to the input folder, Tesseract missing),
        2 (model server or model unavailable, invalid --ollama-url port),
        3 (no images), 5 (output directory not writable).
    """
    # Validate argument values first
    if args.command == "process":
        # Validate batch_size: integer >= 1 OR literal 'all'
        batch_size = args.batch_size
        if batch_size.lower() != "all":
            try:
                batch_int = int(batch_size)
                if batch_int < 1:
                    logger.error("Invalid --batch-size: must be >= 1 or 'all', got '%s'", batch_size)
                    return 1
            except ValueError:
                logger.error("Invalid --batch-size: must be integer or 'all', got '%s'", batch_size)
                return 1

        # --lookahead is deprecated and ignored; its old range is still checked
        # so scripts that passed a bad value keep failing the same way.
        lookahead = getattr(args, "lookahead", None)
        if lookahead is not None:
            if lookahead < 0 or lookahead > 10:
                logger.error("Invalid --lookahead: must be between 0 and 10, got '%s'", lookahead)
                return 1
            logger.warning(LOOKAHEAD_DEPRECATION_WARNING)

        # Validate concurrency: integer 1-10
        if args.concurrency < 1 or args.concurrency > 10:
            logger.error("Invalid --concurrency: must be between 1 and 10, got '%s'", args.concurrency)
            return 1

        # Validate grouping options
        gap = getattr(args, "group_gap_seconds", DEFAULT_GAP_SECONDS)
        if not isinstance(gap, (int, float)) or not math.isfinite(gap) or not 0 < gap <= MAX_GROUP_GAP_SECONDS:
            logger.error(
                "Invalid --group-gap-seconds: must be > 0 and <= %g, got '%s'", MAX_GROUP_GAP_SECONDS, gap
            )
            return 1
        size = getattr(args, "group_max_size", DEFAULT_MAX_GROUP_SIZE)
        if not isinstance(size, int) or not 0 <= size <= MAX_GROUP_SIZE_LIMIT:
            logger.error(
                "Invalid --group-max-size: must be between 0 and %d, got '%s'", MAX_GROUP_SIZE_LIMIT, size
            )
            return 1
        ratio = getattr(args, "split_budget_ratio", DEFAULT_SPLIT_BUDGET_RATIO)
        if (
            not isinstance(ratio, (int, float))
            or not math.isfinite(ratio)
            or not 0 <= ratio <= MAX_SPLIT_BUDGET_RATIO
        ):
            logger.error(
                "Invalid --split-budget-ratio: must be between 0 and %g, got '%s'", MAX_SPLIT_BUDGET_RATIO, ratio
            )
            return 1

        # Validate timeouts
        for flag, dest in (
            ("--request-timeout", "request_timeout"),
            ("--server-start-timeout", "server_start_timeout"),
        ):
            value = getattr(args, dest, None)
            if value is not None and value <= 0:
                logger.error("Invalid %s: must be > 0, got '%s'", flag, value)
                return 1

        # Validate min_free_gb: a finite number >= 0
        min_free_gb = getattr(args, "min_free_gb", DEFAULT_MIN_FREE_GB)
        if not isinstance(min_free_gb, (int, float)) or not math.isfinite(min_free_gb) or min_free_gb < 0:
            logger.error("Invalid --min-free-gb: must be >= 0, got '%s'", min_free_gb)
            return 1

        # Validate ollama_url format
        try:
            parsed = urlparse(args.ollama_url)
            if not parsed.scheme or not parsed.netloc:
                logger.error("Invalid --ollama-url: must be valid URL, got '%s'", args.ollama_url)
                return 1
        except Exception as e:
            logger.error("Invalid --ollama-url: %s", e)
            return 1
        try:
            parsed.port  # noqa: B018 - raises ValueError for an out-of-range port
        except ValueError as e:
            logger.error("Invalid --ollama-url '%s': %s", args.ollama_url, e)
            return 2

    # Check input folder exists
    if not args.image_folder.exists():
        logger.error("Input folder does not exist: %s", args.image_folder)
        return 3

    if not args.image_folder.is_dir():
        logger.error("Input path is not a directory: %s", args.image_folder)
        return 3

    # Outputs written into the input folder itself would be re-processed.
    if _same_directory(args.output_dir, args.image_folder):
        logger.error(
            "--output-dir must not be the same directory as --image-folder "
            "(outputs would be re-processed): %s",
            args.output_dir,
        )
        return 1

    # Count exactly what the scan will process (earlier outputs, hidden
    # folders and the output directory are left out).
    image_files = ImageOrganizer().iter_image_paths(
        args.image_folder, recursive=True, exclude=[args.output_dir]
    )
    if not image_files:
        logger.error("No supported images found in folder: %s", args.image_folder)
        logger.info("Supported formats: %s", ", ".join(sorted(ImageOrganizer.SUPPORTED_EXTENSIONS)))
        return 3

    logger.info("Found %s images in %s", len(image_files), args.image_folder)

    # An output directory that contains the photos must not be marked as
    # output: later scans of it (or of its parents) would skip the photos.
    args._mark_output_dir = not _is_ancestor(args.output_dir, args.image_folder)
    if not args._mark_output_dir:
        logger.warning(
            "--output-dir %s contains --image-folder %s; outputs are written next to your photos "
            "(each bottle folder is marked so later scans skip it). Prefer an output directory "
            "outside the photo folder.",
            args.output_dir, args.image_folder,
        )

    exit_code = _check_output_dir(args)
    if exit_code:
        return exit_code

    # Check Tesseract OCR installed
    try:
        import pytesseract

        version = pytesseract.get_tesseract_version()
        logger.info("Tesseract OCR version: %s", version)
    except Exception:
        logger.exception("Tesseract OCR is not installed or not found in PATH")
        logger.info("Install Tesseract OCR: https://github.com/tesseract-ocr/tesseract")
        return 1

    # Model server last: it is the slow check and may start a process.
    return _check_model_server(args)


def _check_model_server(args: argparse.Namespace) -> int:
    """Pre-flight step for the model server; see :func:`pre_flight_checks`."""
    try:
        backend = ServerBackend.parse(getattr(args, "server_backend", None) or "ollama")
        manager = build_server_manager(args)
    except Exception as e:
        logger.error("Failed to configure model server: %s", e)
        return 2

    args._server_manager = manager
    args.model = resolve_vision_model(args)
    required = _required_models(args)
    model_checks: dict[str, Any] = {
        "allow_pull": bool(getattr(args, "pull", False)),
        "capability": "vision",
        "min_free_bytes": _min_free_bytes(args),
    }

    def _fail(message: str, *params: object) -> int:
        logger.error(message, *params)
        manager.close()
        args._server_manager = None
        return 2

    if getattr(args, "dry_run", False):
        state = manager.probe()
        if not state.is_running:
            allowed, why = manager.server.can_start()
            if not allowed:
                return _fail(
                    "Dry run: %s backend at %s is not running and %s",
                    backend.value, state.base_url, why,
                )
            logger.warning(
                "Dry run: %s backend at %s is not running (would be started); skipping model checks",
                backend.value, state.base_url,
            )
        else:
            logger.info("Dry run: %s backend at %s is reachable", backend.value, state.base_url)
            if not manager.ensure_models(required, dry_run=True, warm=False, **model_checks):
                return _fail(
                    "Required models unavailable: %s (see the errors above)",
                    ", ".join(required),
                )
    else:
        if not manager.ensure_running():
            return _fail("Model server unavailable: %s", manager.last_error or "health check failed")
        logger.info("Model server is reachable at %s", manager.server.base_url)
        if not manager.ensure_models(required, dry_run=False, warm=True, **model_checks):
            return _fail(
                "Required models unavailable: %s (see the errors above)",
                ", ".join(required),
            )

    selector = ModelSelector.from_config(SimpleNamespace(model=args.model))
    if getattr(args, "debug", False):
        selector.print_stage_report()
    args._model_selector = selector
    return 0


def process_command(args: argparse.Namespace) -> int:
    """Process bourbon bottle images from input folder.

    The usage summary is always logged and the model server manager is always
    closed (stopping the server only if this run started it).

    Args:
        args: Parsed command-line arguments.

    Returns:
        Exit code: 0 (success), 1 (general error / some bottles failed),
                   2 (model server or model unavailable), 3 (no valid images found),
                   4 (all bottles failed), 5 (output directory creation failed).
    """
    try:
        exit_code = pre_flight_checks(args)
        if exit_code:
            return exit_code
        if getattr(args, "_server_manager", None) is None:
            args._server_manager = build_server_manager(args)
        if getattr(args, "_model_selector", None) is None:
            args._model_selector = ModelSelector.from_config(
                SimpleNamespace(model=resolve_vision_model(args))
            )
        return _run_pipeline(args, args._server_manager, args._model_selector)
    finally:
        _log_usage_summary(getattr(args, "_model_selector", None))
        server_manager = getattr(args, "_server_manager", None)
        if server_manager is not None:
            try:
                server_manager.close()
            except Exception:
                logger.warning("Error while shutting down the model server", exc_info=True)


def _run_pipeline(
    args: argparse.Namespace,
    server_manager: ModelServerManager,
    model_selector: ModelSelector,
) -> int:
    """Scan, group, analyze and save bottles. Does not close the server manager.

    The scan skips the output directory and earlier outputs. Each bottle is
    saved to its own new folder (see ``PreviewGenerator.save_bottle_files``);
    a bottle that fails analysis or saving, or is rejected in interactive
    mode, gets an ``error_report_<bottle_id>.json`` that ``retry`` can read.

    Args:
        args: Parsed command-line arguments (pre-flight already passed).
        server_manager: Model server manager (started unless dry run).
        model_selector: Per-stage model selector (records usage).

    Returns:
        Exit code: 0 (no failures), 1 (some failed), 2 (model server
        unavailable), 4 (all failed).
    """
    image_organizer = ImageOrganizer(
        gap_seconds=getattr(args, "group_gap_seconds", DEFAULT_GAP_SECONDS),
        max_group_size=getattr(args, "group_max_size", DEFAULT_MAX_GROUP_SIZE),
    )

    # Scan images, leaving out earlier outputs (the output dir may sit inside
    # the input folder, e.g. the default `-f . -o processed`).
    images = image_organizer.scan_images(
        args.image_folder, recursive=True, exclude=[args.output_dir]
    )
    if not images:
        logger.error(
            "No supported images found in %s (earlier outputs and hidden folders are skipped)",
            args.image_folder,
        )
        return 3

    logger.info("Processing %s images...", len(images))

    # Group by bottle (EXIF capture time; cheap, so a dry run shows it too).
    image_groups = image_organizer.group_by_bottle(images)
    _assign_bottle_ids(image_groups)
    logger.info("Found %s bottle groups", len(image_groups))

    if args.dry_run:
        logger.info("Dry-run mode: would process images but not execute")
        for group in image_groups:
            logger.info("  %s: %s", group_bottle_id(group), [str(img.path) for img in group])
        return 0

    try:
        client = server_manager.get_client()
    except ModelServerError as e:
        logger.error("Model server unavailable: %s", e)
        return 2

    bottle_analyzer = BottleAnalyzer(
        ollama_client=client,
        image_organizer=image_organizer,
        model_selector=model_selector,
    )
    vision_model = resolve_vision_model(args)

    batch_processor = BatchProcessor(
        batch_size=args.batch_size,
        bottle_analyzer=bottle_analyzer,
        split_budget_ratio=getattr(args, "split_budget_ratio", DEFAULT_SPLIT_BUDGET_RATIO),
        max_group_size=getattr(args, "group_max_size", DEFAULT_MAX_GROUP_SIZE),
    )
    preview_generator = PreviewGenerator(
        output_dir=args.output_dir, mark_output_dir=getattr(args, "_mark_output_dir", True)
    )

    totals = _RunTotals()
    run_id = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")

    # One split-check budget for the whole run, shared by all batches.
    batch_processor.start_run(image_groups)
    # Groups are complete before batching (capture-time grouping), so each
    # batch is processed as sliced: no group is ever moved or seen twice.
    batches = batch_processor.slice_into_batches(image_groups)

    # Process each batch
    for batch_idx, batch in enumerate(batches, 1):
        logger.info("Processing batch %s/%s (%s image groups)", batch_idx, len(batches), len(batch))

        # Process the batch. Fatal model errors (model unusable, server gone)
        # would fail every remaining bottle the same way: abort the run.
        try:
            batch_result = batch_processor.process_batch(
                batch,
                vision_model=vision_model,
            )
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

        # Interactive mode: prompt user for corrections
        rejected: set[str] = set()
        if args.interactive:
            try:
                from models.bottle import Correction

                correction_dicts, rejected_bottles_from_prompt = preview_generator.interactive_prompt(
                    batch_result.bottle_dicts,
                    batch_result.metadata_list,
                )

                if correction_dicts:
                    # Convert dict corrections to Correction objects
                    corrections = [
                        Correction(
                            field_name=cd["field_name"],
                            original_value=cd["original_value"],
                            corrected_value=cd["corrected_value"],
                        )
                        for cd in correction_dicts
                    ]

                    # Apply corrections to bottle_dicts
                    batch_result.bottle_dicts = batch_processor.apply_corrections(
                        batch_result.bottle_dicts,
                        corrections,
                    )

                    # Learn from corrections for future batches
                    batch_processor.learn_from_feedback(corrections)

                    logger.info("Applied %d corrections from interactive feedback", len(corrections))

                # Rejected bottles are not saved and count as failed
                rejected = set(rejected_bottles_from_prompt)
                batch_result.failed_bottles.extend(rejected_bottles_from_prompt)
                if rejected_bottles_from_prompt:
                    logger.info("Rejected %d bottles from interactive feedback", len(rejected_bottles_from_prompt))

            except Exception:
                logger.exception("Interactive correction failed")

        # Batch preview report and JSON, after review so they hold the
        # corrected values and every rejected bottle (skipped when processing
        # all in a single batch).
        if str(args.batch_size).lower() != "all":
            try:
                preview_path = preview_generator.save_batch_preview(
                    batch_result.bottle_dicts,
                    batch_result.metadata_list,
                    batch_id=f"{run_id}_{batch_idx:03d}",
                    rejected_bottles=list(batch_result.failed_bottles),
                )
                logger.info("Saved batch preview report to %s", preview_path)
            except Exception:
                logger.exception("Failed to generate batch preview report")

        _save_batch_bottles(batch_result, rejected, preview_generator, bottle_analyzer, totals)
        _report_failed_bottles(batch_result, batch, rejected, preview_generator, totals)

    # Completion reporting
    logger.info("=" * 60)
    logger.info("Processing Complete")
    logger.info("=" * 60)
    logger.info("Processed successfully: %s", totals.processed)
    logger.info("Failed: %s", totals.failed)
    logger.info("Error reports generated: %s", len(totals.error_reports))
    logger.info("%s", batch_processor.split_summary())
    nudge = batch_processor.group_size_nudge()
    if nudge:
        logger.warning("%s", nudge)
    logger.info("Output directory: %s", args.output_dir)
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


@dataclass
class _RunTotals:
    """Outcome counters and written paths for one ``process`` run."""

    processed: int = 0
    failed: int = 0
    saved_folders: list[Path] = dataclass_field(default_factory=list)
    error_reports: list[Path] = dataclass_field(default_factory=list)


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
) -> None:
    """Write a retry-able failure report; a failure to write is only logged."""
    try:
        totals.error_reports.append(
            preview_generator.save_failure_report(bottle_id, [str(p) for p in images], error, stage=stage)
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

    for bottle_dict, metadata in zip(batch_result.bottle_dicts, batch_result.metadata_list):
        bottle_id = PreviewGenerator.bottle_id_for(bottle_dict, metadata)
        images = list(bottle_dict.get("images") or metadata.source_images)
        if bottle_id in rejected:
            _write_failure_report(
                preview_generator, totals, bottle_id, images, "Rejected during interactive review", "rejected"
            )
            continue
        try:
            bottle = Bottle.from_analysis(bottle_dict, metadata, quantity=bottle_dict.get("quantity"))
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
        reason = batch_result.errors.get(failed_bottle_id)
        if reason:
            logger.warning("Bottle failed to process: %s (%s)", failed_bottle_id, reason)
        else:
            logger.warning("Bottle failed to process: %s", failed_bottle_id)
        group = groups_by_id.get(failed_bottle_id, [])
        _write_failure_report(
            preview_generator,
            totals,
            failed_bottle_id,
            [img.path for img in group],
            reason or "analysis failed; see the log",
            "analysis",
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
    logger.info("  Fallback calls:   %s", summary.get("fallback_calls", 0))
    logger.info("  Total duration:   %.2fs", summary.get("total_duration_seconds", 0))
    by_stage = summary.get("by_stage", {})
    if by_stage:
        logger.info("  By stage:")
        for stage_name, stats in by_stage.items():
            logger.info("    %s: %s calls, %s success, %s fallback",
                        stage_name, stats.get("total", 0),
                        stats.get("success", 0), stats.get("fallback", 0))
    by_model = summary.get("by_model", {})
    if by_model:
        logger.info("  By model:")
        for model_name, stats in by_model.items():
            logger.info("    %s: %s calls, %s success",
                        model_name, stats.get("total", 0), stats.get("success", 0))


def _correction_counts(bottle_dicts: list[dict]) -> list[int]:
    """Number of recorded corrections per bottle dict (before applying new ones)."""
    return [len(d.get("corrections") or []) if isinstance(d, dict) else 0 for d in bottle_dicts]


def _resave_corrected_bottles(
    preview_generator: PreviewGenerator,
    bottle_dicts: list[dict],
    metadata_list: list,
    counts_before: list[int],
    rejected: set[str],
) -> int:
    """Save each bottle that just received a correction as a new output folder.

    ``BatchProcessor.apply_corrections`` appends to a bottle's
    ``corrections`` list, so a bottle is resaved only if that list grew.
    Bottles listed as rejected (in the batch JSON or during this review) and
    bottles no correction touched are left alone; existing folders are never
    overwritten (``save_bottle_files`` picks a new ``_N`` name).

    Returns:
        Number of bottles resaved.
    """
    from models.bottle import Bottle

    resaved = 0
    for index, (bottle_dict, metadata) in enumerate(zip(bottle_dicts, metadata_list)):  # noqa: B905
        bottle_id = PreviewGenerator.bottle_id_for(bottle_dict, metadata)
        if bottle_id in rejected:
            logger.info("Not resaving rejected bottle %s", bottle_id)
            continue
        recorded = bottle_dict.get("corrections") or []
        before = counts_before[index] if index < len(counts_before) else 0
        if len(recorded) <= before:
            continue
        try:
            bottle = Bottle.from_analysis(bottle_dict, metadata, corrections=recorded)
            output_folder = preview_generator.save_bottle_files(bottle, metadata)
            logger.info("Resaved bottle %s to %s", bottle_id, output_folder)
            resaved += 1
        except Exception:
            logger.exception("Failed to resave bottle %s", bottle_id)
    return resaved


def feedback_command(args: argparse.Namespace) -> int:
    """Apply feedback and corrections to processed batch.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Exit code: 0 (success), 1 (general error), 2 (batch not found).
    """
    from models.bottle import Correction

    # Determine batch to load
    if args.batch:
        batch_preview_file = args.output_dir / f"batch_{args.batch}_preview.md"
        batch_json_file = args.output_dir / f"batch_{args.batch}.json"
    else:
        # Find most recent batch preview
        batch_preview_files = sorted(args.output_dir.glob("batch_*_preview.md"))
        if not batch_preview_files:
            logger.error("No batch preview files found in %s", args.output_dir)
            return 2
        batch_preview_file = max(batch_preview_files, key=lambda p: (p.stat().st_mtime, p.name))
        batch_json_file = batch_preview_file.with_name(
            batch_preview_file.name.replace("_preview.md", ".json")
        )

    logger.info("Using batch file: %s", batch_preview_file)

    # Load batch data from JSON if available
    bottle_dicts = []
    metadata_list = []
    previously_rejected: set[str] = set()

    if batch_json_file.exists():
        try:
            import json

            with batch_json_file.open("r", encoding="utf-8") as f:
                batch_data = json.load(f)
                bottle_dicts = batch_data.get("bottles", [])
                previously_rejected = {str(r) for r in batch_data.get("rejected") or []}
                # Reconstruct metadata from loaded data
                from models.bottle import ExtractedMetadata

                metadata_list = [
                    ExtractedMetadata.from_dict(m) for m in batch_data.get("metadata", [])
                ]
            logger.info("Loaded %d bottles from batch JSON", len(bottle_dicts))
        except Exception:
            logger.exception("Failed to load batch JSON file")

    # Load corrections from JSON file if provided
    corrections: list[Correction] = []

    if args.corrections and args.corrections.exists():
        try:
            import json

            with args.corrections.open("r", encoding="utf-8") as f:
                corrections_data = json.load(f)
                corrections = [Correction.from_dict(c) for c in corrections_data]
            logger.info("Loaded %d corrections from %s", len(corrections), args.corrections)
        except Exception:
            logger.exception("Failed to load corrections file")
            return 1

    # Apply corrections
    if corrections and bottle_dicts:
        batch_processor = BatchProcessor()
        batch_processor.bottle_analyzer = BottleAnalyzer()

        before = _correction_counts(bottle_dicts)
        modified_bottles = batch_processor.apply_corrections(bottle_dicts, corrections)
        logger.info("Applied %d corrections to %d bottles", len(corrections), len(modified_bottles))

        _resave_corrected_bottles(
            PreviewGenerator(output_dir=args.output_dir),
            modified_bottles,
            metadata_list,
            before,
            previously_rejected,
        )
        logger.info("Feedback processing complete")
        return 0

    # Interactive mode
    if args.interactive:
        try:
            preview_generator = PreviewGenerator(output_dir=args.output_dir)
            batch_processor = BatchProcessor()

            correction_dicts, rejected_bottles = preview_generator.interactive_prompt(
                bottle_dicts,
                metadata_list,
            )

            if correction_dicts:
                # Convert dict corrections to Correction objects
                corrections = [
                    Correction(
                        field_name=cd["field_name"],
                        original_value=cd["original_value"],
                        corrected_value=cd["corrected_value"],
                    )
                    for cd in correction_dicts
                ]

                # Apply corrections to bottle_dicts
                before = _correction_counts(bottle_dicts)
                modified_bottles = batch_processor.apply_corrections(bottle_dicts, corrections)

                # Learn from corrections
                batch_processor.learn_from_feedback(corrections)

                logger.info("Applied %d corrections from interactive feedback", len(corrections))

                _resave_corrected_bottles(
                    preview_generator,
                    modified_bottles,
                    metadata_list,
                    before,
                    previously_rejected | {str(r) for r in rejected_bottles or []},
                )

            logger.info("Interactive feedback complete")
            return 0

        except Exception:
            logger.exception("Interactive feedback failed")
            return 1

    logger.info("No corrections applied")
    return 0


def retry_command(args: argparse.Namespace) -> int:
    """Retry processing failed bottles from error report.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Exit code: 0 (success), 1 (general error or model server unavailable),
        2 (report not found).
    """
    from models.image import Image

    # Check error report exists
    if not args.report.exists():
        logger.error("Error report not found: %s", args.report)
        return 2

    # Load error report
    import json

    try:
        with args.report.open("r", encoding="utf-8") as f:
            error_report = json.load(f)
    except Exception:
        logger.exception("Failed to read error report")
        return 1

    logger.info("Loading error report: %s", args.report)

    # Validate error report structure
    required_fields = ["bottle_id", "images", "error_message"]
    for field in required_fields:
        if field not in error_report:
            logger.error("Missing required field in error report: %s", field)
            return 1

    bottle_id = error_report["bottle_id"]
    image_paths = error_report["images"]
    error_message = error_report["error_message"]
    suggested_fix = error_report.get("suggested_fix", {})

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
        server_manager = build_server_manager(args)
    except Exception as e:
        logger.error("Failed to configure model server: %s", e)
        return 1
    try:
        if not server_manager.ensure_running():
            logger.error(
                "Model server unavailable: %s", server_manager.last_error or "health check failed"
            )
            return 1
        return _retry_bottle(args, server_manager, bottle_id, images, suggested_fix)
    finally:
        server_manager.close()


def _retry_bottle(
    args: argparse.Namespace,
    server_manager: ModelServerManager,
    bottle_id: str,
    images: list,
    suggested_fix: dict,
) -> int:
    """Re-run analysis for one bottle from an error report (server already up).

    Args:
        args: Parsed command-line arguments.
        server_manager: Running model server manager.
        bottle_id: Bottle identifier from the report.
        images: Image objects for the bottle.
        suggested_fix: Field corrections from the report.

    Returns:
        Exit code: 0 (success) or 1 (failure).
    """
    from models.bottle import Bottle

    try:
        client = server_manager.get_client()
    except ModelServerError as e:
        logger.error("Model server unavailable: %s", e)
        return 1
    vision_model = resolve_vision_model(args)

    image_organizer = ImageOrganizer()
    bottle_analyzer = BottleAnalyzer(ollama_client=client, image_organizer=image_organizer)

    # Apply suggested corrections if provided
    corrections: list = []
    if suggested_fix:
        from models.bottle import Correction

        for field_name, correction_value in suggested_fix.items():
            if field_name in ["bottle_id", "images", "error_message"]:
                continue  # Skip metadata fields
            corrections.append(
                Correction(
                    field_name=field_name,
                    original_value="",
                    corrected_value=str(correction_value),
                )
            )

    # Check hints.md for additional guidance
    hints_file = args.report.parent.parent / "hints.md"
    hints_text = ""

    if hints_file.exists():
        try:
            with hints_file.open("r", encoding="utf-8") as f:
                hints_text = f.read()
            # Extract hints relevant to this bottle
            import re

            bottle_hints_pattern = re.compile(
                rf".*{re.escape(bottle_id)}.*?:(.*?)(?=\n\n|\n[0-9]+\.|\Z)", re.MULTILINE | re.DOTALL
            )
            matches = bottle_hints_pattern.findall(hints_text)
            if matches:
                hints_text = "\n".join(matches).strip()
                logger.info("Found hints for bottle %s", bottle_id)
        except Exception:
            logger.warning("Failed to read hints.md")

    # Re-process the bottle
    try:
        # Apply hints as corrections if applicable
        if hints_text and not corrections:
            logger.info("Applying hints from hints.md")
            # Hints could be applied as context to the analyzer
            # For now, just log them as they may need manual interpretation

        bottle_dict, metadata = bottle_analyzer.process_single_bottle(
            images,
            vision_model=vision_model,
        )

        # Apply suggested corrections
        if corrections:
            batch_processor = BatchProcessor()
            bottle_dict = batch_processor.apply_corrections([bottle_dict], corrections)[0]
            logger.info("Applied %d corrections from suggested_fix", len(corrections))

        bottle = Bottle.from_analysis(
            bottle_dict,
            metadata,
            quantity=max(1, bottle_analyzer.detect_quantity(images)),
            corrections=corrections,
        )

        # Save output files (a new folder; earlier output is never replaced)
        preview_generator = PreviewGenerator(output_dir=args.output_dir)
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


def _exit_on_signal(signum: int, _frame: Any) -> None:  # noqa: ANN401
    """Signal handler: exit with the conventional 128 + signal status."""
    sys.exit(128 + signum)


def main(argv: list | None = None) -> int:
    """Main entry point for CLI.

    Args:
        argv: Optional list of arguments (default: sys.argv[1:]).

    Returns:
        Exit code.
    """
    if not (sys.argv[1:] if argv is None else list(argv)):
        # A bare `bourbon-process` would otherwise process the current folder.
        build_parser().print_help()
        return 1
    args = parse_args(argv)

    # Setup logging
    setup_logging(debug=args.debug)

    # Turn SIGTERM, SIGHUP (terminal closed, SSH session dropped) and SIGQUIT
    # (Ctrl-backslash) into SystemExit so `finally` blocks run and an owned model
    # server, which runs in its own session and never sees these signals, is
    # stopped instead of orphaned.
    for name in ("SIGTERM", "SIGHUP", "SIGQUIT"):
        sig = getattr(signal, name, None)
        if sig is not None:
            signal.signal(sig, _exit_on_signal)

    if args.command == "process":
        return process_command(args)
    elif args.command == "feedback":
        return feedback_command(args)
    elif args.command == "retry":
        return retry_command(args)
    else:
        logger.error("Unknown command: %s", args.command)
        return 1


if __name__ == "__main__":
    sys.exit(main())
