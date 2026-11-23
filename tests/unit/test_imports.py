"""Regression tests for package import layout (src/ is a package root, not a package)."""

import re
import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src"
FORBIDDEN = re.compile(r"^\s*(from|import)\s+src(\.|\s|$)", re.MULTILINE)


def test_no_src_prefixed_imports():
    """Modules under src/ must import `lib.*`, `models.*`, ... never `src.*`.

    `src.` imports only resolve when the repo root is on sys.path, so they break
    the installed `bourbon-process` entry point and load modules twice.
    """
    offenders = [
        str(p.relative_to(SRC.parent))
        for p in SRC.rglob("*.py")
        if FORBIDDEN.search(p.read_text(encoding="utf-8"))
    ]
    assert offenders == []


def test_src_is_not_a_package():
    assert not (SRC / "__init__.py").exists()


def test_cli_imports_with_only_src_on_path():
    """Import the CLI the way the console script does: src/ on sys.path, not the repo root."""
    code = (
        "import sys; sys.path[:] = [p for p in sys.path if p not in ('', '.')];"
        f"sys.path.insert(0, {str(SRC)!r});"
        "import cli.commands, image_processor.preview_generator"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=SRC.parent.parent,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
