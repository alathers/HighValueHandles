"""Shared test helpers.

``make_settings`` builds real :class:`lib.settings.Settings` (defaults, no
config files, an empty environment) and then applies overrides, so tests
never hand-build ``argparse.Namespace`` objects with stale attribute names.
"""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path
from typing import Any

from lib.settings import Settings, resolve_settings


def make_settings(command: str = "process", *, environ: dict[str, str] | None = None, **overrides: Any) -> Settings:
    """Default settings for ``command`` with ``overrides`` applied.

    Top-level fields are replaced directly (``concurrency=3``,
    ``image_folder=tmp_path``). Nested groups accept a dict of their fields
    (``web={"enabled": False}``, ``retry={"enabled": False}``) or a full
    replacement object. Path-like strings stay as given.

    Args:
        command: ``process``, ``feedback``, ``retry`` or ``config``.
        environ: Environment used for resolution (default: empty, so no API
            key or ``BOURBON_*`` variable leaks in).
        **overrides: Field values.
    """
    args = argparse.Namespace(command=command, no_config=True)
    settings = resolve_settings(args, environ or {}, cwd=Path.cwd())
    changes: dict[str, Any] = {}
    for name, value in overrides.items():
        current = getattr(settings, name)
        if isinstance(value, dict) and dataclasses.is_dataclass(current):
            changes[name] = dataclasses.replace(current, **value)
        else:
            changes[name] = value
    return dataclasses.replace(settings, **changes)
