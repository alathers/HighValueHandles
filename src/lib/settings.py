"""Settings registry: defaults, config files, environment and CLI in one place.

Every setting the tool reads is declared once in :data:`REGISTRY`. A value is
resolved from these layers, lowest to highest precedence:

1. built-in defaults;
2. the user file ``${XDG_CONFIG_HOME:-~/.config}/bourbon-process/config.toml``;
3. the project file ``./bourbon.toml`` (current directory only);
4. ``--config PATH`` / ``$BOURBON_CONFIG`` (replaces 2 and 3; must exist);
5. ``--no-config`` / ``BOURBON_NO_CONFIG=1`` (skips every file);
6. ``BOURBON_*`` environment variables (an empty value counts as unset);
7. command-line flags that were given (argparse defaults are all None).

Config files are sectioned TOML read with :mod:`tomllib`. Unknown sections
and keys are errors, types are strict, and API keys are refused in files:
they come only from the ``BRAVE_API_KEY`` / ``BOURBON_BRAVE_API_KEY`` and
``OLLAMA_API_KEY`` environment variables, whose values are never stored,
logged or written (only "set"/"unset").

Every error is collected and reported together in one :class:`SettingsError`,
each naming the source of the bad value.

This module is imported only by ``src/cli/``; components take plain
keyword arguments.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import tomllib
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field, fields, is_dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal
from urllib.parse import urlparse

from image_processor.batch_processor import DEFAULT_SPLIT_BUDGET_RATIO, MAX_SPLIT_BUDGET_RATIO
from image_processor.capture_grouping import DEFAULT_GAP_SECONDS, DEFAULT_MAX_GROUP_SIZE
from lib.constants import DEFAULT_TEXT_MODEL, DEFAULT_VISION_MODEL
from lib.model_availability import DEFAULT_MIN_FREE_BYTES, GiB
from lib.model_server import DEFAULT_OLLAMA_URL, DEFAULT_REQUEST_TIMEOUT
from lib.retry import RetryPolicy

#: Commands.
PROCESS, FEEDBACK, RETRY, CONFIG = "process", "feedback", "retry", "config"
COMMANDS = (PROCESS, FEEDBACK, RETRY, CONFIG)

#: Upper bounds kept from the old pre-flight validation.
MAX_GROUP_GAP_SECONDS = 600.0
MAX_GROUP_SIZE_LIMIT = 20
MAX_CONCURRENCY = 10

#: Search providers (``web.provider``); ``auto`` resolves to one of the others.
WEB_PROVIDERS = ("auto", "duckduckgo", "brave", "ollama", "searxng", "none")
#: Think levels for the text (gpt-oss) stage.
THINK_LEVELS = ("low", "medium", "high")

#: Environment variables holding API keys (values are never read into Settings).
BRAVE_KEY_VARS = ("BRAVE_API_KEY", "BOURBON_BRAVE_API_KEY")
OLLAMA_KEY_VARS = ("OLLAMA_API_KEY",)
API_KEY_VARS = (*BRAVE_KEY_VARS, *OLLAMA_KEY_VARS)

#: Environment variables that select config files.
ENV_CONFIG = "BOURBON_CONFIG"
ENV_NO_CONFIG = "BOURBON_NO_CONFIG"

#: File names.
APP_DIR_NAME = "bourbon-process"
USER_CONFIG_NAME = "config.toml"
PROJECT_CONFIG_NAME = "bourbon.toml"

#: Key-name fragments that are never allowed in a config file.
_SECRET_KEY_PARTS = ("api_key", "apikey", "token", "secret", "password")
SECRET_KEY_MESSAGE = "put API keys in BRAVE_API_KEY / OLLAMA_API_KEY environment variables, never in a config file"

#: Reason recorded when web lookups are on but no search provider is selected.
NO_PROVIDER_REASON = (
    "no search provider configured (set BRAVE_API_KEY or OLLAMA_API_KEY or web.searxng_url, "
    "or web.provider = 'auto' for DuckDuckGo)"
)

_TRUE = frozenset({"1", "true", "yes", "on"})
_FALSE = frozenset({"0", "false", "no", "off"})


class SettingsError(ValueError):
    """One or more invalid settings; ``errors`` lists every problem with its source."""

    def __init__(self, errors: Iterable[str]):
        self.errors = list(errors)
        super().__init__("\n".join(f"  - {e}" for e in self.errors))


# ---------------------------------------------------------------------------
# Kinds and validators
# ---------------------------------------------------------------------------


class BatchSize:
    """Kind marker: an int >= 1 or the literal ``"all"``."""


class OptionalPath:
    """Kind marker: a path, or ``""`` for "not set"."""


class StrList:
    """Kind marker: a list of strings (comma-separated in the environment)."""


def _range(lo: float, hi: float | None = None, *, lo_open: bool = False) -> Callable[[Any], str | None]:
    """Validator for a number in [lo, hi] (or (lo, hi] with ``lo_open``); no upper bound if ``hi`` is None."""
    if hi is None:
        message = f"must be {'>' if lo_open else '>='} {lo:g}"
    elif lo_open:
        message = f"must be > {lo:g} and <= {hi:g}"
    else:
        message = f"must be between {lo:g} and {hi:g}"

    def check(value: Any) -> str | None:  # noqa: ANN401
        if isinstance(value, float) and not math.isfinite(value):
            return f"must be a finite number ({message[8:]})"
        above = lo < value if lo_open else lo <= value
        return None if above and (hi is None or value <= hi) else message

    return check


def _ok(_: Any) -> str | None:  # noqa: ANN401
    return None


def _non_empty(value: Any) -> str | None:  # noqa: ANN401
    return None if isinstance(value, str) and value.strip() else "must not be empty"


def _http_url(value: Any, *, allow_empty: bool = False) -> str | None:  # noqa: ANN401
    if allow_empty and value == "":
        return None
    try:
        parsed = urlparse(value)
        port = parsed.port  # raises ValueError for an invalid port
    except ValueError as e:
        return f"invalid URL: {e}"
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        return "must be an http:// or https:// URL with a host"
    if port is not None and not 0 < port < 65536:  # noqa: PLR2004
        return f"invalid port {port}"
    return None


def _batch_size(value: Any) -> str | None:  # noqa: ANN401
    if value == "all":
        return None
    return None if isinstance(value, int) and value >= 1 else "must be an integer >= 1 or 'all'"


def _domains(value: Any) -> str | None:  # noqa: ANN401
    bad = [d for d in value if not re.fullmatch(r"[A-Za-z0-9.-]+", d) or "." not in d]
    return f"not domain names: {', '.join(bad)}" if bad else None


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SettingSpec:
    """One setting.

    Attributes:
        key: ``"section.name"`` (the TOML section and key).
        kind: ``int``, ``float``, ``bool``, ``str``, ``Path``, :class:`BatchSize`,
            :class:`OptionalPath`, :class:`StrList`, or a tuple of allowed strings.
        default: Built-in default.
        env: Environment variable, or None.
        cli_dest: argparse ``dest``, or None.
        cli_flag: Flag spelling for messages, e.g. ``"--concurrency"``.
        validate: Returns an error text or None.
        commands: Commands that use it; the others ignore it.
        help: One-line description (template comment and ``--help``).
        file: Whether config files may set it (False for CLI-only keys).
    """

    key: str
    kind: Any
    default: Any
    env: str | None
    cli_dest: str | None
    cli_flag: str | None
    validate: Callable[[Any], str | None]
    commands: frozenset[str]
    help: str
    file: bool = True

    @property
    def section(self) -> str:
        return self.key.split(".", 1)[0]

    @property
    def name(self) -> str:
        return self.key.split(".", 1)[1]


_P = frozenset({PROCESS})
_PR = frozenset({PROCESS, RETRY})
_PF = frozenset({PROCESS, FEEDBACK})
_PFR = frozenset({PROCESS, FEEDBACK, RETRY})
_ALL = frozenset(COMMANDS)

DEFAULT_MIN_FREE_GB = DEFAULT_MIN_FREE_BYTES / GiB


def _spec(key, kind, default, env, dest, flag, validate, commands, help_text, *, file=True) -> SettingSpec:  # noqa: ANN001, PLR0913
    return SettingSpec(key, kind, default, env, dest, flag, validate, commands, help_text, file)


REGISTRY: tuple[SettingSpec, ...] = (
    # input / output
    _spec("input.image_folder", Path, ".", "BOURBON_IMAGE_FOLDER", "image_folder", "-f/--image-folder", _ok, _P,
          "Folder containing the photos to process."),
    _spec("output.output_dir", Path, "processed", "BOURBON_OUTPUT_DIR", "output_dir", "-o/--output-dir", _ok, _PFR,
          "Output directory for processed bottles."),
    # processing
    _spec("processing.batch_size", BatchSize, 5, "BOURBON_BATCH_SIZE", "batch_size", "-b/--batch-size", _batch_size,
          _P, "Bottles per batch (an integer >= 1, or \"all\")."),
    _spec("processing.concurrency", int, 2, "BOURBON_CONCURRENCY", "concurrency", "--concurrency",
          _range(1, MAX_CONCURRENCY), _P,
          "Bottle groups analysed in parallel (1-10); results are merged in input order. "
          "1 is strictly sequential (use it for evaluation runs)."),
    _spec("processing.interactive", bool, False, "BOURBON_INTERACTIVE", "interactive", "-i/--interactive", _ok, _PF,
          "Review every bottle interactively before saving."),
    _spec("processing.hints_file", OptionalPath, "", "BOURBON_HINTS_FILE", "hints_file", "--hints", _ok, _PR,
          "Hints TOML file with per-bottle reading notes and sourced field values (\"\" = none)."),
    # grouping
    _spec("grouping.gap_seconds", float, DEFAULT_GAP_SECONDS, "BOURBON_GROUP_GAP_SECONDS", "group_gap_seconds",
          "--group-gap-seconds", _range(0, MAX_GROUP_GAP_SECONDS, lo_open=True), _P,
          "Most seconds between consecutive photos of one bottle (EXIF capture time)."),
    _spec("grouping.max_group_size", int, DEFAULT_MAX_GROUP_SIZE, "BOURBON_GROUP_MAX_SIZE", "group_max_size",
          "--group-max-size", _range(0, MAX_GROUP_SIZE_LIMIT), _P,
          "Most photos per bottle; longer runs are split at their largest gap (0 = no limit)."),
    _spec("grouping.split_budget_ratio", float, DEFAULT_SPLIT_BUDGET_RATIO, "BOURBON_SPLIT_BUDGET_RATIO",
          "split_budget_ratio", "--split-budget-ratio", _range(0, MAX_SPLIT_BUDGET_RATIO), _P,
          "Extra split-check analyses per group over the run (0 turns the check off)."),
    # model
    _spec("model.vision_model", str, DEFAULT_VISION_MODEL, "BOURBON_MODEL", "model", "-m/--model", _non_empty, _PR,
          "Ollama vision model tag that reads the label (must be vision-capable)."),
    _spec("model.pull", bool, False, "BOURBON_PULL", "pull", "--pull/--no-pull", _ok, _P,
          "Download missing models after a disk-space check (never implicit)."),
    _spec("model.min_free_gb", float, DEFAULT_MIN_FREE_GB, "BOURBON_MIN_FREE_GB", "min_free_gb", "--min-free-gb",
          _range(0), _P, "GiB to keep free on the Ollama model store after a pull."),
    _spec("model.keep_models_loaded", bool, False, "BOURBON_KEEP_MODELS_LOADED", "keep_models_loaded",
          "--keep-models-loaded/--no-keep-models-loaded", _ok, _PR,
          "Keep the vision and text models loaded together instead of unloading between phases."),
    # text (gpt-oss fact check)
    _spec("text.enabled", bool, True, "BOURBON_VERIFY", "verify", "--verify/--no-verify", _ok, _PR,
          "Run the text-model fact check over OCR and web evidence."),
    _spec("text.model", str, DEFAULT_TEXT_MODEL, "BOURBON_TEXT_MODEL", "text_model", "--text-model", _non_empty, _PR,
          "Ollama text model for the fact check (\"none\" disables the stage)."),
    _spec("text.think_extract", THINK_LEVELS, "high", "BOURBON_TEXT_THINK", "text_think", "--text-think", _ok, _PR,
          "Reasoning effort of the fact-check extraction call (low, medium, high)."),
    _spec("text.think_compose", THINK_LEVELS, "medium", None, None, None, _ok, _PR,
          "Reasoning effort of the description-writing call (low, medium, high)."),
    _spec("text.timeout_seconds", float, 900.0, "BOURBON_TEXT_TIMEOUT", "text_timeout", "--text-timeout",
          _range(0, lo_open=True), _PR, "Read timeout of one text-model call, in seconds."),
    _spec("text.num_ctx", int, 32768, None, None, None, _range(4096, 131072), _PR,
          "Context window (tokens) of text-model calls (4096-131072)."),
    # server
    _spec("server.ollama_url", str, DEFAULT_OLLAMA_URL, "BOURBON_OLLAMA_URL", "ollama_url", "-u/--ollama-url",
          _http_url, _PR, "Ollama service URL."),
    _spec("server.autostart", bool, True, "BOURBON_SERVER_AUTOSTART", "server_autostart",
          "--server-autostart/--no-server-autostart", _ok, _PR,
          "Start a local Ollama server if none is running."),
    _spec("server.keep_running", bool, False, "BOURBON_KEEP_SERVER", "keep_server", "--keep-server/--no-keep-server",
          _ok, _PR, "Leave a server started by this run running afterwards."),
    _spec("server.start_timeout", float, 30.0, "BOURBON_SERVER_START_TIMEOUT", "server_start_timeout",
          "--server-start-timeout", _range(0, lo_open=True), _PR, "Seconds to wait for a server this run starts."),
    _spec("server.request_timeout", float, float(DEFAULT_REQUEST_TIMEOUT), "BOURBON_REQUEST_TIMEOUT",
          "request_timeout", "--request-timeout", _range(0, lo_open=True), _PR,
          "Per-request vision timeout in seconds (covers a cold model load)."),
    # retry
    _spec("retry.enabled", bool, True, "BOURBON_RETRY", "retry_enabled", "-r/--allow-retry, --no-retry", _ok, _PR,
          "Repeat model requests that fail in a way that may pass (unusable output, timeouts, 5xx)."),
    _spec("retry.max_attempts", int, 3, "BOURBON_RETRY_ATTEMPTS", "retry_attempts", "--retry-attempts",
          _range(1, 5), _PR, "Attempts per model request when retry is on (1-5)."),
    _spec("retry.initial_delay_seconds", float, 2.0, None, None, None, _range(0, 60), _PR,
          "Seconds before the second attempt (doubles each time, with jitter)."),
    _spec("retry.max_delay_seconds", float, 30.0, None, None, None, _range(0, 300), _PR,
          "Longest wait between attempts, in seconds."),
    # web
    _spec("web.enabled", bool, True, "BOURBON_WEB", "web", "--web/--no-web", _ok, _PR,
          "Look bottles up on the web (search, fetch, cite)."),
    _spec("web.provider", WEB_PROVIDERS, "auto", "BOURBON_WEB_PROVIDER", "web_provider", "--web-provider", _ok, _PR,
          "Search provider: auto (Brave key, Ollama key, SearXNG URL, else DuckDuckGo), duckduckgo, brave, "
          "ollama, searxng or none."),
    _spec("web.searxng_url", str, "", "BOURBON_SEARXNG_URL", None, None,
          lambda v: _http_url(v, allow_empty=True), _PR, "Base URL of a self-hosted SearXNG instance (\"\" = none)."),
    _spec("web.vendor_images", bool, False, "BOURBON_INCLUDE_VENDOR_IMAGES", "include_vendor_images",
          "-v/--include-vendor-images", _ok, _PR,
          "Fetch a vendor reference image per bottle (implies web lookups)."),
    _spec("web.max_queries", int, 2, None, None, None, _range(1, 10), _PR, "Search queries per bottle."),
    _spec("web.max_results", int, 8, None, None, None, _range(1, 20), _PR, "Search results kept per query."),
    _spec("web.max_pages", int, 5, None, None, None, _range(0, 20), _PR, "Pages fetched per bottle."),
    _spec("web.max_llm_pages", int, 4, None, None, None, _range(0, 10), _PR,
          "Fetched pages passed to the fact check per bottle."),
    _spec("web.fetch_timeout_seconds", float, 15.0, None, None, None, _range(0, 120, lo_open=True), _PR,
          "Timeout of one page fetch, in seconds."),
    _spec("web.max_page_bytes", int, 2_000_000, None, None, None, _range(10_000, 50_000_000), _PR,
          "Largest page body read, in bytes."),
    _spec("web.max_image_bytes", int, 5_000_000, None, None, None, _range(10_000, 50_000_000), _PR,
          "Largest vendor image read, in bytes."),
    _spec("web.per_host_interval_seconds", float, 2.0, None, None, None, _range(0, 60), _PR,
          "Least seconds between two requests to one host."),
    _spec("web.max_requests_per_run", int, 500, None, None, None, _range(0, 100_000), _PR,
          "Most web requests in one run (searches and fetches)."),
    _spec("web.cache_ttl_hours", int, 168, None, None, None, _range(0, 8760), _PR,
          "Hours a fetched page is reused from the cache (0 = never)."),
    _spec("web.search_cache_ttl_hours", int, 24, None, None, None, _range(0, 8760), _PR,
          "Hours a search result list is reused from the cache (0 = never)."),
    _spec("web.deny_domains", StrList, [], None, None, None, _domains, _PR,
          "Domains never fetched or cited (subdomains included)."),
    _spec("web.allow_domains", StrList, [], None, None, None, _domains, _PR,
          "If not empty, only these domains are fetched or cited (strict mode)."),
    _spec("web.cache_dir", OptionalPath, "", "BOURBON_WEB_CACHE_DIR", "web_cache_dir", "--web-cache-dir", _ok, _PR,
          "Web cache directory (\"\" = ${XDG_CACHE_HOME:-~/.cache}/bourbon-process/web)."),
    _spec("web.refresh", bool, False, None, "refresh_web", "--refresh-web", _ok, _PR,
          "Ignore cached pages and search results for this run.", file=False),
    # logging
    _spec("logging.debug", bool, False, "BOURBON_DEBUG", "debug", "--debug", _ok, _ALL, "Debug logging."),
)

SPECS: Mapping[str, SettingSpec] = MappingProxyType({s.key: s for s in REGISTRY})
SECTIONS: tuple[str, ...] = tuple(dict.fromkeys(s.section for s in REGISTRY))

#: Sections whose keys are top-level ``Settings`` attributes.
_TOP_LEVEL_SECTIONS = frozenset({"input", "output", "processing", "logging"})

#: Settings attributes whose name differs from the key's last part.
_ATTR_NAMES = {"logging.debug": "debug"}


def attr_path(key: str) -> str:
    """Dotted ``Settings`` attribute path of a registry key."""
    if key in _ATTR_NAMES:
        return _ATTR_NAMES[key]
    section, name = key.split(".", 1)
    return name if section in _TOP_LEVEL_SECTIONS else f"{section}.{name}"


# ---------------------------------------------------------------------------
# Data shapes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GroupingSettings:
    gap_seconds: float
    max_group_size: int
    split_budget_ratio: float


@dataclass(frozen=True)
class ModelSettings:
    vision_model: str
    pull: bool
    min_free_gb: float
    keep_models_loaded: bool

    @property
    def min_free_bytes(self) -> int:
        return int(self.min_free_gb * GiB)


@dataclass(frozen=True)
class TextSettings:
    enabled: bool
    model: str
    think_extract: str
    think_compose: str
    timeout_seconds: float
    num_ctx: int

    @property
    def active(self) -> bool:
        """Whether the text (fact-check) stage runs: enabled and a model other than "none"."""
        return self.enabled and self.model.strip().lower() != "none"


@dataclass(frozen=True)
class ServerSettings:
    ollama_url: str
    autostart: bool
    keep_running: bool
    start_timeout: float
    request_timeout: float


@dataclass(frozen=True)
class RetrySettings:
    enabled: bool
    max_attempts: int
    initial_delay_seconds: float
    max_delay_seconds: float

    def policy(self) -> RetryPolicy:
        """The run's retry policy (one attempt when retry is off)."""
        return RetryPolicy(
            max_attempts=self.max_attempts if self.enabled else 1,
            initial_delay=self.initial_delay_seconds,
            max_delay=self.max_delay_seconds,
        )


@dataclass(frozen=True)
class WebSettings:
    enabled: bool
    provider: str
    searxng_url: str
    vendor_images: bool
    max_queries: int
    max_results: int
    max_pages: int
    max_llm_pages: int
    fetch_timeout_seconds: float
    max_page_bytes: int
    max_image_bytes: int
    per_host_interval_seconds: float
    max_requests_per_run: int
    cache_ttl_hours: int
    search_cache_ttl_hours: int
    deny_domains: tuple[str, ...]
    allow_domains: tuple[str, ...]
    cache_dir: Path
    refresh: bool
    #: ``brave | ollama | searxng | duckduckgo | none``, resolved from ``provider``.
    provider_resolved: str = "none"
    #: Why that provider was chosen (never contains a key value).
    provider_reason: str = ""


@dataclass(frozen=True)
class Settings:
    """Effective settings of one command run (immutable)."""

    command: str
    image_folder: Path
    output_dir: Path
    batch_size: int | Literal["all"]
    concurrency: int
    interactive: bool
    dry_run: bool
    hints_file: Path | None
    debug: bool
    grouping: GroupingSettings
    model: ModelSettings
    text: TextSettings
    server: ServerSettings
    retry: RetrySettings
    web: WebSettings
    report: Path | None = None
    batch: str | None = None
    corrections: Path | None = None
    approve: tuple[str, ...] = ()
    config_action: str | None = None
    json: bool = False
    #: key -> "default" | "file:/abs/path" | "env:BOURBON_X" | "cli:--flag".
    sources: Mapping[str, str] = field(default_factory=dict)
    #: Config files read, in order.
    config_files: tuple[Path, ...] = ()
    #: Config files looked for (read or not).
    config_searched: tuple[Path, ...] = ()
    #: API-key variable name -> "set" | "unset" (never the value).
    api_keys: Mapping[str, str] = field(default_factory=dict)

    def value(self, key: str) -> Any:  # noqa: ANN401
        """The effective value of a registry key."""
        obj: Any = self
        for part in attr_path(key).split("."):
            obj = getattr(obj, part)
        return obj

    def is_explicit(self, key: str) -> bool:
        """Whether a key was set by a file, the environment or the CLI (not the default)."""
        return self.sources.get(key, "default") != "default"


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


def _type_name(kind: Any) -> str:  # noqa: ANN401
    if isinstance(kind, tuple):
        return "one of " + ", ".join(kind)
    return {
        int: "an integer", float: "a number", bool: "true or false", str: "a string", Path: "a path",
        BatchSize: "an integer or \"all\"", OptionalPath: "a path or \"\"", StrList: "a list of strings",
    }[kind]


def _from_file(spec: SettingSpec, raw: Any, base: Path) -> Any:  # noqa: ANN401
    """Strictly typed value from TOML; raises ValueError."""
    kind = spec.kind
    if kind is bool:
        if isinstance(raw, bool):
            return raw
    elif kind is int:
        if isinstance(raw, int) and not isinstance(raw, bool):
            return raw
    elif kind is float:
        if isinstance(raw, (int, float)) and not isinstance(raw, bool):
            return float(raw)
    elif kind is str:
        if isinstance(raw, str):
            return raw.strip()
    elif kind is BatchSize:
        if isinstance(raw, int) and not isinstance(raw, bool):
            return raw
        if isinstance(raw, str) and raw.strip().lower() == "all":
            return "all"
    elif kind in (Path, OptionalPath):
        if isinstance(raw, str):
            if not raw.strip():
                return None if kind is OptionalPath else _raise(f"must be {_type_name(kind)}")
            return _resolve_path(raw, base)
    elif kind is StrList:
        if isinstance(raw, list) and all(isinstance(v, str) for v in raw):
            return [v.strip() for v in raw if v.strip()]
    elif isinstance(kind, tuple):
        if isinstance(raw, str) and raw.strip().lower() in kind:
            return raw.strip().lower()
    raise ValueError(f"must be {_type_name(kind)}")


def _raise(message: str) -> Any:  # noqa: ANN401
    raise ValueError(message)


def _parse_bool(text: str) -> bool:
    lowered = text.strip().lower()
    if lowered in _TRUE:
        return True
    if lowered in _FALSE:
        return False
    raise ValueError("must be one of 1/0/true/false/yes/no/on/off")


def _from_text(spec: SettingSpec, text: str, base: Path) -> Any:  # noqa: ANN401
    """Value from an environment string; raises ValueError."""
    kind = spec.kind
    text = text.strip()
    if kind is bool:
        return _parse_bool(text)
    if kind is int:
        try:
            return int(text)
        except ValueError:
            raise ValueError(f"must be {_type_name(kind)}") from None
    if kind is float:
        try:
            return float(text)
        except ValueError:
            raise ValueError(f"must be {_type_name(kind)}") from None
    if kind is BatchSize:
        return _batch_size_text(text)
    if kind in (Path, OptionalPath):
        return _resolve_path(text, base)
    if kind is StrList:
        return [v.strip() for v in text.split(",") if v.strip()]
    if isinstance(kind, tuple):
        if text.lower() in kind:
            return text.lower()
        raise ValueError(f"must be {_type_name(kind)}")
    return text


def _batch_size_text(text: str) -> int | str:
    if text.lower() == "all":
        return "all"
    try:
        return int(text)
    except ValueError:
        raise ValueError("must be an integer >= 1 or 'all'") from None


def _from_cli(spec: SettingSpec, value: Any, base: Path) -> Any:  # noqa: ANN401
    """Value from argparse (already typed for numbers); raises ValueError."""
    if spec.kind is BatchSize:
        return _batch_size_text(str(value).strip())
    if spec.kind in (Path, OptionalPath):
        text = str(value)
        if not text.strip():
            if spec.kind is OptionalPath:
                return None
            raise ValueError("must not be empty")
        return _resolve_path(text, base)
    if isinstance(spec.kind, tuple):
        text = str(value).strip().lower()
        if text not in spec.kind:
            raise ValueError(f"must be {_type_name(spec.kind)}")
        return text
    if spec.kind is str:
        return str(value).strip()
    return value


def _resolve_path(text: str, base: Path) -> Path:
    path = Path(text).expanduser()
    return path if path.is_absolute() else base / path


def _default_value(spec: SettingSpec, cwd: Path) -> Any:  # noqa: ANN401
    if spec.kind in (Path, OptionalPath):
        return _resolve_path(spec.default, cwd) if spec.default else None
    if spec.kind is StrList:
        return list(spec.default)
    return spec.default


def _show(value: Any) -> str:  # noqa: ANN401
    if isinstance(value, str):
        return repr(value)
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return '""'
    return str(value)


def _describe_source(source: str, key: str) -> str:
    if source.startswith("file:"):
        return f"{source[5:]} [{key.split('.', 1)[0]}]"
    if source.startswith("env:"):
        return f"env {source[4:]}"
    if source.startswith("cli:"):
        return f"flag {source[4:]}"
    return source


# ---------------------------------------------------------------------------
# Config files
# ---------------------------------------------------------------------------


def user_config_path(environ: Mapping[str, str]) -> Path:
    """``${XDG_CONFIG_HOME:-~/.config}/bourbon-process/config.toml``."""
    base = environ.get("XDG_CONFIG_HOME") or ""
    root = Path(base).expanduser() if base.strip() else Path("~/.config").expanduser()
    return root / APP_DIR_NAME / USER_CONFIG_NAME


def default_cache_dir(environ: Mapping[str, str]) -> Path:
    """``${XDG_CACHE_HOME:-~/.cache}/bourbon-process/web``."""
    base = environ.get("XDG_CACHE_HOME") or ""
    root = Path(base).expanduser() if base.strip() else Path("~/.cache").expanduser()
    return root / APP_DIR_NAME / "web"


def config_search_paths(
    environ: Mapping[str, str], cwd: Path, explicit: Path | None = None
) -> list[Path]:
    """Config files consulted, lowest precedence first.

    With an explicit file (``--config`` or ``$BOURBON_CONFIG``) only that file;
    otherwise the user file, then ``./bourbon.toml``.
    """
    if explicit is not None:
        return [explicit]
    return [user_config_path(environ), cwd / PROJECT_CONFIG_NAME]


def _is_secret_key(name: str) -> bool:
    lowered = str(name).lower()
    return any(part in lowered for part in _SECRET_KEY_PARTS)


def _load_file(path: Path, errors: list[str]) -> dict[str, Any]:
    """Read one config file into ``{key: raw value}``; problems go to ``errors``."""
    try:
        with path.open("rb") as handle:
            data = tomllib.load(handle)
    except tomllib.TOMLDecodeError as e:
        errors.append(f"{path}: invalid TOML: {e}")
        return {}
    except OSError as e:
        errors.append(f"{path}: cannot read config file: {e}")
        return {}
    values: dict[str, Any] = {}
    for section, table in data.items():
        if _is_secret_key(section):
            errors.append(f"{path}: [{section}]: {SECRET_KEY_MESSAGE}")
            continue
        if section not in SECTIONS:
            errors.append(
                f"unknown section [{section}] in {path} (known sections: {', '.join(SECTIONS)})"
            )
            continue
        if not isinstance(table, dict):
            errors.append(f"{path}: '{section}' must be a [{section}] table")
            continue
        for name, raw in table.items():
            key = f"{section}.{name}"
            if _is_secret_key(name):
                errors.append(f"{path} [{section}] {name}: {SECRET_KEY_MESSAGE}")
                continue
            spec = SPECS.get(key)
            if spec is None or not spec.file:
                errors.append(f"unknown key '{name}' in {path} [{section}]")
                continue
            values[key] = raw
    return values


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------


def _explicit_config(args: argparse.Namespace, environ: Mapping[str, str], cwd: Path, errors: list[str]) -> tuple[
    Path | None, bool
]:
    """(explicit config file or None, skip all files)."""
    cli_config = getattr(args, "config", None)
    cli_no_config = bool(getattr(args, "no_config", None))
    env_config = (environ.get(ENV_CONFIG) or "").strip()
    env_no_config = False
    raw_no = (environ.get(ENV_NO_CONFIG) or "").strip()
    if raw_no:
        try:
            env_no_config = _parse_bool(raw_no)
        except ValueError as e:
            errors.append(f"{ENV_NO_CONFIG}={raw_no!r} (from env {ENV_NO_CONFIG}): {e}")
    explicit: Path | None = None
    if cli_config is not None and str(cli_config).strip():
        explicit = _resolve_path(str(cli_config), cwd)
    elif env_config:
        explicit = _resolve_path(env_config, cwd)
    no_config = cli_no_config or env_no_config
    if explicit is not None and no_config:
        errors.append(
            "--config/BOURBON_CONFIG and --no-config/BOURBON_NO_CONFIG cannot be used together"
        )
        return None, True
    if explicit is not None and not explicit.is_file():
        errors.append(f"config file not found: {explicit}")
    return explicit, no_config


def resolve_web_provider(web: WebSettings, environ: Mapping[str, str]) -> tuple[str, str]:
    """Pick the search provider for ``web.provider`` from the keys present.

    ``auto``: Brave if ``BRAVE_API_KEY``/``BOURBON_BRAVE_API_KEY`` is set, else
    Ollama web search if ``OLLAMA_API_KEY`` is set, else SearXNG if
    ``web.searxng_url`` is set, else DuckDuckGo (no key needed). Key values
    are only tested for presence.

    Returns:
        ``(name, reason)`` with name in ``duckduckgo|brave|ollama|searxng|none``.

    Raises:
        ValueError: An explicit provider whose key or URL is missing.
    """
    has_brave = any((environ.get(v) or "").strip() for v in BRAVE_KEY_VARS)
    has_ollama = any((environ.get(v) or "").strip() for v in OLLAMA_KEY_VARS)
    has_searxng = bool(web.searxng_url.strip())
    provider = web.provider
    if not web.enabled:
        return "none", "web lookups are off (web.enabled = false)"
    if provider == "auto":
        if has_brave:
            return "brave", "auto: BRAVE_API_KEY is set"
        if has_ollama:
            return "ollama", "auto: OLLAMA_API_KEY is set"
        if has_searxng:
            return "searxng", "auto: web.searxng_url is set"
        return "duckduckgo", "auto: no API key or SearXNG URL; using DuckDuckGo (no key needed)"
    if provider == "brave":
        if not has_brave:
            raise ValueError("web.provider 'brave' needs BRAVE_API_KEY (or BOURBON_BRAVE_API_KEY) in the environment")
        return "brave", "web.provider = brave"
    if provider == "ollama":
        if not has_ollama:
            raise ValueError("web.provider 'ollama' needs OLLAMA_API_KEY in the environment")
        return "ollama", "web.provider = ollama"
    if provider == "searxng":
        if not has_searxng:
            raise ValueError("web.provider 'searxng' needs web.searxng_url (or BOURBON_SEARXNG_URL)")
        return "searxng", "web.provider = searxng"
    if provider == "duckduckgo":
        return "duckduckgo", "web.provider = duckduckgo"
    return "none", NO_PROVIDER_REASON


def resolve_settings(
    args: argparse.Namespace,
    environ: Mapping[str, str] | None = None,
    *,
    cwd: Path | None = None,
) -> Settings:
    """Resolve the effective settings for one command.

    Args:
        args: Parsed CLI arguments (every flag defaults to None).
        environ: Environment (default ``os.environ``).
        cwd: Directory for relative env/CLI paths and ``./bourbon.toml``.

    Raises:
        SettingsError: Listing every invalid value with its source.
    """
    environ = os.environ if environ is None else environ
    cwd = Path.cwd() if cwd is None else Path(cwd)
    command = getattr(args, "command", None) or PROCESS
    errors: list[str] = []

    values: dict[str, Any] = {}
    sources: dict[str, str] = {}
    for spec in REGISTRY:
        values[spec.key] = _default_value(spec, cwd)
        sources[spec.key] = "default"

    # Files
    explicit, no_config = _explicit_config(args, environ, cwd, errors)
    searched: list[Path] = []
    loaded: list[Path] = []
    if not no_config:
        for path in config_search_paths(environ, cwd, explicit):
            searched.append(path)
            if not path.is_file():
                continue
            loaded.append(path)
            for key, raw in _load_file(path, errors).items():
                spec = SPECS[key]
                try:
                    values[key] = _from_file(spec, raw, path.parent)
                    sources[key] = f"file:{path}"
                except ValueError as e:
                    errors.append(f"{spec.name}={_show(raw)} (from {path} [{spec.section}]): {e}")

    # Environment
    for spec in REGISTRY:
        if not spec.env:
            continue
        raw = environ.get(spec.env)
        if raw is None or not raw.strip():
            continue
        try:
            values[spec.key] = _from_text(spec, raw, cwd)
            sources[spec.key] = f"env:{spec.env}"
        except ValueError as e:
            errors.append(f"{spec.name}={raw!r} (from env {spec.env}): {e}")

    # CLI
    for spec in REGISTRY:
        if not spec.cli_dest:
            continue
        raw = getattr(args, spec.cli_dest, None)
        if raw is None:
            continue
        flag = _cli_flag_used(spec, raw)
        try:
            values[spec.key] = _from_cli(spec, raw, cwd)
            sources[spec.key] = f"cli:{flag}"
        except ValueError as e:
            errors.append(f"{spec.name}={_show(raw)} (from flag {flag}): {e}")

    # Per-key validation
    for spec in REGISTRY:
        value = values[spec.key]
        if value is None:
            continue
        problem = spec.validate(value)
        if problem:
            source = _describe_source(sources[spec.key], spec.key)
            errors.append(f"{spec.name}={_show(value)} (from {source}): {problem}")

    # Cross-field rules
    if values["retry.max_delay_seconds"] < values["retry.initial_delay_seconds"]:
        errors.append(
            f"retry.max_delay_seconds={values['retry.max_delay_seconds']:g} "
            f"(from {_describe_source(sources['retry.max_delay_seconds'], 'retry.max_delay_seconds')}) "
            f"must be >= retry.initial_delay_seconds={values['retry.initial_delay_seconds']:g}"
        )
    if values["web.vendor_images"] and not values["web.enabled"]:
        errors.append(
            "vendor images (-v/--include-vendor-images, from "
            f"{_describe_source(sources['web.vendor_images'], 'web.vendor_images')}) need web lookups, "
            f"but web.enabled is false (from {_describe_source(sources['web.enabled'], 'web.enabled')})"
        )

    web_cache = values["web.cache_dir"] or default_cache_dir(environ)
    web = WebSettings(
        enabled=values["web.enabled"],
        provider=values["web.provider"],
        searxng_url=values["web.searxng_url"],
        vendor_images=values["web.vendor_images"],
        max_queries=values["web.max_queries"],
        max_results=values["web.max_results"],
        max_pages=values["web.max_pages"],
        max_llm_pages=values["web.max_llm_pages"],
        fetch_timeout_seconds=values["web.fetch_timeout_seconds"],
        max_page_bytes=values["web.max_page_bytes"],
        max_image_bytes=values["web.max_image_bytes"],
        per_host_interval_seconds=values["web.per_host_interval_seconds"],
        max_requests_per_run=values["web.max_requests_per_run"],
        cache_ttl_hours=values["web.cache_ttl_hours"],
        search_cache_ttl_hours=values["web.search_cache_ttl_hours"],
        deny_domains=tuple(values["web.deny_domains"]),
        allow_domains=tuple(values["web.allow_domains"]),
        cache_dir=web_cache,
        refresh=values["web.refresh"],
    )

    # Provider resolution (only for commands that use the web).
    if command in (PROCESS, RETRY, CONFIG):
        try:
            name, reason = resolve_web_provider(web, environ)
        except ValueError as e:
            name, reason = "none", str(e)
            errors.append(f"{e} (web.provider from {_describe_source(sources['web.provider'], 'web.provider')})")
        requested_by = next(
            (k for k in ("web.enabled", "web.vendor_images") if sources[k] != "default" and values[k]), None
        )
        if web.enabled and name == "none" and requested_by is not None:
            errors.append(
                f"web lookups were requested ({requested_by} from "
                f"{_describe_source(sources[requested_by], requested_by)}) but web.provider is 'none' "
                f"(from {_describe_source(sources['web.provider'], 'web.provider')})"
            )
    else:
        name, reason = "none", f"web lookups are not used by '{command}'"
    web = replace(web, provider_resolved=name, provider_reason=reason)

    if errors:
        raise SettingsError(errors)

    approve = getattr(args, "approve", None) or ()
    report = getattr(args, "report", None)
    corrections = getattr(args, "corrections", None)
    return Settings(
        command=command,
        image_folder=values["input.image_folder"],
        output_dir=values["output.output_dir"],
        batch_size=values["processing.batch_size"],
        concurrency=values["processing.concurrency"],
        interactive=values["processing.interactive"],
        dry_run=bool(getattr(args, "dry_run", None)),
        hints_file=values["processing.hints_file"],
        debug=values["logging.debug"],
        grouping=GroupingSettings(
            gap_seconds=values["grouping.gap_seconds"],
            max_group_size=values["grouping.max_group_size"],
            split_budget_ratio=values["grouping.split_budget_ratio"],
        ),
        model=ModelSettings(
            vision_model=values["model.vision_model"],
            pull=values["model.pull"],
            min_free_gb=values["model.min_free_gb"],
            keep_models_loaded=values["model.keep_models_loaded"],
        ),
        text=TextSettings(
            enabled=values["text.enabled"],
            model=values["text.model"],
            think_extract=values["text.think_extract"],
            think_compose=values["text.think_compose"],
            timeout_seconds=values["text.timeout_seconds"],
            num_ctx=values["text.num_ctx"],
        ),
        server=ServerSettings(
            ollama_url=values["server.ollama_url"].rstrip("/"),
            autostart=values["server.autostart"],
            keep_running=values["server.keep_running"],
            start_timeout=values["server.start_timeout"],
            request_timeout=values["server.request_timeout"],
        ),
        retry=RetrySettings(
            enabled=values["retry.enabled"],
            max_attempts=values["retry.max_attempts"],
            initial_delay_seconds=values["retry.initial_delay_seconds"],
            max_delay_seconds=values["retry.max_delay_seconds"],
        ),
        web=web,
        report=_resolve_path(str(report), cwd) if report is not None else None,
        batch=getattr(args, "batch", None),
        corrections=_resolve_path(str(corrections), cwd) if corrections is not None else None,
        approve=tuple(str(a) for a in approve),
        config_action=getattr(args, "config_action", None),
        json=bool(getattr(args, "json", None)),
        sources=MappingProxyType(dict(sources)),
        config_files=tuple(loaded),
        config_searched=tuple(searched),
        api_keys=MappingProxyType(
            {v: ("set" if (environ.get(v) or "").strip() else "unset") for v in API_KEY_VARS}
        ),
    )


def _cli_flag_used(spec: SettingSpec, value: Any) -> str:  # noqa: ANN401
    """The flag spelling to report for a CLI value (the --no- form for a False boolean pair)."""
    flag = spec.cli_flag or f"--{spec.cli_dest}"
    if spec.key == "retry.enabled":
        return "-r/--allow-retry" if value else "--no-retry"
    if spec.kind is not bool:
        return flag
    parts = flag.split("/")
    positive = "/".join(p for p in parts if not p.startswith("--no-"))
    if value:
        return positive
    negative = [p for p in parts if p.startswith("--no-")]
    if negative:
        return negative[0]
    longs = [p for p in parts if p.startswith("--")]
    return f"--no-{longs[-1][2:]}" if longs else positive


def ignored_keys(settings: Settings) -> list[str]:
    """Keys a file or the environment set that this command does not use."""
    return [
        spec.key
        for spec in REGISTRY
        if settings.command not in spec.commands
        and settings.sources.get(spec.key, "default").startswith(("file:", "env:"))
    ]


# ---------------------------------------------------------------------------
# Output: template, describe, manifest
# ---------------------------------------------------------------------------


def _toml_value(value: Any) -> str:  # noqa: ANN401
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        text = repr(value)
        return text if any(c in text for c in ".en") else f"{text}.0"
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, list):
        return "[" + ", ".join(_toml_value(v) for v in value) + "]"
    raise TypeError(f"cannot render {value!r} as TOML")


def _env_and_flag(spec: SettingSpec) -> str:
    parts = []
    if spec.env:
        parts.append(f"env {spec.env}")
    if spec.cli_flag:
        parts.append(f"flag {spec.cli_flag}")
    return "; ".join(parts) if parts else "config file only"


def render_template() -> str:
    """Commented, sectioned TOML with every file-settable key at its default.

    Parsing it with :mod:`tomllib` gives exactly the defaults. API keys are
    never set here: use ``BRAVE_API_KEY`` / ``OLLAMA_API_KEY``.
    """
    lines = [
        "# bourbon-process configuration (generated by `bourbon-process config template`).",
        "#",
        "# Files read, lowest precedence first:",
        f"#   ${{XDG_CONFIG_HOME:-~/.config}}/{APP_DIR_NAME}/{USER_CONFIG_NAME}",
        f"#   ./{PROJECT_CONFIG_NAME} (current directory only)",
        "# --config PATH (or $BOURBON_CONFIG) reads only that file; --no-config (or",
        "# BOURBON_NO_CONFIG=1) reads none. Environment variables override files and",
        "# command-line flags override both. Relative paths are relative to this file.",
        "# Unknown keys are errors. API keys never go here: set BRAVE_API_KEY or",
        "# OLLAMA_API_KEY in the environment.",
    ]
    for section in SECTIONS:
        specs = [s for s in REGISTRY if s.section == section and s.file]
        if not specs:
            continue
        lines += ["", f"[{section}]"]
        for spec in specs:
            lines.append(f"# {spec.help} ({_env_and_flag(spec)})")
            lines.append(f"{spec.name} = {_toml_value(spec.default)}")
    return "\n".join(lines) + "\n"


def template_defaults() -> dict[str, dict[str, Any]]:
    """The defaults as the nested dict ``tomllib`` returns for :func:`render_template`."""
    out: dict[str, dict[str, Any]] = {}
    for spec in REGISTRY:
        if spec.file:
            out.setdefault(spec.section, {})[spec.name] = spec.default
    return out


def _jsonable(value: Any) -> Any:  # noqa: ANN401
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if is_dataclass(value):
        return {f.name: _jsonable(getattr(value, f.name)) for f in fields(value)}
    return value


#: Keys holding URLs whose credentials (``user:pass@``, query string) are redacted on output.
_URL_KEYS = frozenset({"server.ollama_url", "web.searxng_url"})


def redact_url(url: str) -> str:
    """``url`` with any ``user:password@`` and query string replaced by ``***``.

    A URL may carry credentials (basic auth, ``?token=``); the run manifest
    and ``config show`` are written next to shareable output, so they never
    hold them.
    """
    if not isinstance(url, str) or not url:
        return url
    try:
        parsed = urlparse(url)
    except ValueError:
        return url
    netloc = parsed.netloc
    if "@" in netloc:
        netloc = "***@" + netloc.rsplit("@", 1)[1]
    query = "***" if parsed.query else ""
    return parsed._replace(netloc=netloc, query=query).geturl()


def settings_to_dict(settings: Settings) -> dict[str, Any]:
    """Effective settings as JSON-ready data (no key values, no URL credentials)."""
    return {
        "command": settings.command,
        "values": {
            spec.key: (
                redact_url(settings.value(spec.key)) if spec.key in _URL_KEYS
                else _jsonable(settings.value(spec.key))
            )
            for spec in REGISTRY
        },
        "sources": dict(settings.sources),
        "web_provider": {
            "resolved": settings.web.provider_resolved,
            "reason": settings.web.provider_reason,
        },
        "text_stage_active": settings.text.active,
        "config_files_searched": [str(p) for p in settings.config_searched],
        "config_files_loaded": [str(p) for p in settings.config_files],
    }


def describe(settings: Settings, as_json: bool = False) -> str:
    """``config show``: files searched and loaded, then each key, value and source.

    API keys appear only as ``set`` or ``unset``.
    """
    data = settings_to_dict(settings)
    data["api_keys"] = dict(settings.api_keys)
    if as_json:
        return json.dumps(data, indent=2, sort_keys=False)
    lines = ["Config files searched:"]
    loaded = set(data["config_files_loaded"])
    if not data["config_files_searched"]:
        lines.append("  (none: --no-config)")
    for path in data["config_files_searched"]:
        lines.append(f"  {path} ({'loaded' if path in loaded else 'not found'})")
    width = max(len(spec.key) for spec in REGISTRY)
    current = None
    for spec in REGISTRY:
        if spec.section != current:
            current = spec.section
            lines += ["", f"[{current}]"]
        value = data["values"][spec.key]
        shown = json.dumps(value) if not isinstance(value, str) else value
        lines.append(f"  {spec.key:<{width}} = {shown}  ({data['sources'][spec.key]})")
    lines += [
        "",
        f"Web search provider: {data['web_provider']['resolved']} ({data['web_provider']['reason']})",
        f"Text (fact-check) stage: {'on' if data['text_stage_active'] else 'off'}",
        "API keys: " + ", ".join(f"{k} {v}" for k, v in data["api_keys"].items()),
    ]
    return "\n".join(lines)
