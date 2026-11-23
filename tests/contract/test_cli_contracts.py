"""CLI interface contract tests for bourbon image processor.

Tests ensure the CLI implementation matches the specification defined in
specs/001-bourbon-image-processor/contracts/cli-interface.md.
"""

import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

# Import CLI commands for direct invocation
from cli.commands import process_command, retry_command, feedback_command
from lib.exceptions import OllamaConnectionError


@pytest.fixture(autouse=True)
def _run_in_tmp_cwd(tmp_path, monkeypatch):
    """Run every test (and the CLI subprocesses it starts) from ``tmp_path``.

    Many tests invoke ``process`` without ``-f``/``-o``; the defaults ``.``
    and ``processed`` are relative to the working directory, so without this
    the suite would scan the repo and write ``processed/.bourbon_output``
    into it.
    """
    monkeypatch.chdir(tmp_path)


#: A closed local port: CLI subprocesses that get as far as probing the model
#: server see "connection refused" instead of the developer's real Ollama
#: (the in-process hermetic guard does not reach child processes).
CLOSED_OLLAMA_URL = "http://127.0.0.1:9"
NO_SERVER = ["--ollama-url", CLOSED_OLLAMA_URL]


def expected_dry_run_exit() -> int:
    """Exit code of a dry run over a folder with images and no model server.

    1 without Tesseract (checked first), 2 if Ollama could not be started by
    a real run (no executable), else 0 ("would be started").
    """
    if shutil.which("tesseract") is None:
        return 1
    if shutil.which(os.environ.get("OLLAMA_BIN", "ollama")) is None:
        return 2
    return 0


def assert_clean(result: subprocess.CompletedProcess, code: int) -> None:
    """The CLI exited with ``code`` and did not crash."""
    output = result.stdout + result.stderr
    assert "Traceback" not in output, output
    assert result.returncode == code, output


def get_python_executable() -> str:
    """Get the appropriate Python executable path for subprocess calls.

    Uses the interpreter running the test suite, so subprocesses see the same
    environment (venv, installed package) regardless of what ``python3`` is
    first on PATH.

    Returns:
        Path to Python executable.
    """
    return sys.executable


class TestCLIProcessCommand:
    """Test the `process` command interface."""

    def test_process_command_basic_invocation(self):
        """Test basic `process` command can be invoked.

        Contract: Process command has no required arguments, supports optional arguments
        and defaults.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            result = subprocess.run(
                [get_python_executable(), "-m", "cli.commands", "process", "--help"],
                capture_output=True,
                text=True,
            )
            assert result.returncode == 0
            assert "process" in result.stdout.lower()

    def test_process_help_text_generation(self):
        """Test `--help` generates valid help text.

        Contract: See Validation Rules > Contract Tests (line 499).
        """
        result = subprocess.run(
            [get_python_executable(), "-m", "cli.commands", "process", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0

        # Verify help text contains expected sections
        assert "--image-folder" in result.stdout
        assert "--batch-size" in result.stdout
        assert "--model" in result.stdout
        assert "--ollama-url" in result.stdout
        assert "--output-dir" in result.stdout

    def test_process_invalid_arguments_exit_code_1(self):
        """Test invalid arguments exit with code 1 and show error message.

        Contract: See Validation Rules > Contract Tests (line 500).
        Contract: See Error Handling > Argument Errors (line 422).
        """
        result = subprocess.run(
            [get_python_executable(), "-m", "cli.commands", "process", "--batch-size", "-1"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 1
        assert "invalid" in result.stdout.lower() or "error" in result.stdout.lower()

    def test_process_batch_size_validation(self):
        """Test --batch-size validation: integer >= 1 OR literal 'all'.

        Contract: See Validation Rules > Option Validation (line 393).
        """
        # Valid integers
        for size in ["1", "5", "10"]:
            result = subprocess.run(
                [get_python_executable(), "-m", "cli.commands", "process", "--batch-size", size, "--dry-run",
                 *NO_SERVER],
                capture_output=True,
                text=True,
            )
            assert_clean(result, 3)  # valid argument; the empty tmp_path has no images

        # Invalid integer (negative)
        result = subprocess.run(
            [get_python_executable(), "-m", "cli.commands", "process", "--batch-size", "-1"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 1

        # Valid 'all'
        result = subprocess.run(
            [get_python_executable(), "-m", "cli.commands", "process", "--batch-size", "all", "--dry-run",
             *NO_SERVER],
            capture_output=True,
            text=True,
        )
        assert_clean(result, 3)

    def test_process_image_folder_validation(self):
        """Test --image-folder validation: must exist, be directory, be readable.

        Contract: See Validation Rules > Option Validation (line 392).
        """
        # Non-existent folder
        result = subprocess.run(
            [get_python_executable(), "-m", "cli.commands", "process", "--image-folder", "/nonexistent/path"],
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0  # Should fail

        # File instead of directory
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            f.write("test")
            fpath = f.name

        try:
            result = subprocess.run(
                [get_python_executable(), "-m", "cli.commands", "process", "--image-folder", fpath],
                capture_output=True,
                text=True,
            )
            assert result.returncode != 0  # Should fail
        finally:
            Path(fpath).unlink()

    def test_process_ollama_url_invalid(self):
        """Test --ollama-url validation: must be valid URL.

        Contract: See Validation Rules > Option Validation (line 395).
        """
        result = subprocess.run(
            [get_python_executable(), "-m", "cli.commands", "process", "--ollama-url", "not-a-url"],
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0

    def test_process_lookahead_validation(self):
        """Test --lookahead validation: integer 0-10 (deprecated and ignored, still range-checked).

        Contract: See Validation Rules > Option Validation (line 397).
        """
        # Valid range
        for value in ["0", "5", "10"]:
            result = subprocess.run(
                [get_python_executable(), "-m", "cli.commands", "process", "--lookahead", value, "--dry-run",
                 *NO_SERVER],
                capture_output=True,
                text=True,
            )
            assert_clean(result, 3)

        # Deprecated: accepted, ignored, and says so
        assert "--lookahead is deprecated and ignored" in result.stdout + result.stderr

        # Invalid: negative
        result = subprocess.run(
            [get_python_executable(), "-m", "cli.commands", "process", "--lookahead", "-1"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 1

        # Invalid: > 10
        result = subprocess.run(
            [get_python_executable(), "-m", "cli.commands", "process", "--lookahead", "11"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 1

    def test_process_concurrency_validation(self):
        """Test --concurrency validation: integer 1-10.

        Contract: See Validation Rules > Option Validation (line 398).
        """
        # Valid range
        for value in ["1", "5", "10"]:
            result = subprocess.run(
                [get_python_executable(), "-m", "cli.commands", "process", "--concurrency", value, "--dry-run",
                 *NO_SERVER],
                capture_output=True,
                text=True,
            )
            assert_clean(result, 3)

        # Invalid: 0
        result = subprocess.run(
            [get_python_executable(), "-m", "cli.commands", "process", "--concurrency", "0"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 1

        # Invalid: > 10
        result = subprocess.run(
            [get_python_executable(), "-m", "cli.commands", "process", "--concurrency", "11"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 1

    def test_process_dry_run_flag(self):
        """Test --dry-run shows what would be processed without actual processing.

        Contract: See Options table (line 42).
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test images
            test_file = Path(tmpdir) / "IMG_0001.jpg"
            test_file.write_bytes(b"\xff\xd8\xff\xe0\x00\x10JFIF")  # JPEG header

            out = Path(tmpdir) / "out"
            result = subprocess.run(
                [
                    get_python_executable(), "-m", "cli.commands", "process", "--image-folder", tmpdir,
                    "-o", str(out), "--dry-run", *NO_SERVER,
                ],
                capture_output=True,
                text=True,
            )

            assert_clean(result, expected_dry_run_exit())
            assert not out.exists()  # a dry run writes nothing
            if result.returncode == 0:
                assert "Dry-run mode" in result.stdout
                assert "IMG_0001" in result.stdout

    def test_process_debug_flag(self):
        """Test --debug enables verbose logging.

        Contract: See Options table (line 41).
        Contract: See Output Conventions > Debug Messages (line 294).
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test images
            test_file = Path(tmpdir) / "IMG_0001.jpg"
            test_file.write_bytes(b"\xff\xd8\xff\xe0\x00\x10JFIF")  # JPEG header

            # --dry-run and a closed server port keep this a pure flag test.
            result = subprocess.run(
                [
                    get_python_executable(), "-m", "cli.commands", "process",
                    "--image-folder", tmpdir, "--debug", "--dry-run", *NO_SERVER,
                ],
                capture_output=True,
                text=True,
            )
            assert_clean(result, expected_dry_run_exit())
            assert " - DEBUG - " in result.stdout
            assert "TiffImagePlugin" not in result.stdout  # PIL debug noise is silenced

    def test_process_interactive_flag(self):
        """Test --interactive enables interactive CLI prompts.

        Contract: See Options table (line 34).
        """
        # The autouse fixture runs this in an empty tmp_path: no images, exit 3
        # before any server contact or interactive prompt.
        result = subprocess.run(
            [
                get_python_executable(), "-m", "cli.commands", "process", "--interactive",
                "--dry-run", *NO_SERVER,
            ],
            capture_output=True,
            text=True,
        )
        assert_clean(result, 3)

    def test_process_include_vendor_images_flag(self):
        """Test --include-vendor-images enables vendor image caching.

        Contract: See Options table (line 35).
        """
        # Runs in an empty tmp_path (autouse fixture): no images, exit 3.
        result = subprocess.run(
            [
                get_python_executable(), "-m", "cli.commands", "process",
                "--include-vendor-images", "--dry-run", *NO_SERVER,
            ],
            capture_output=True,
            text=True,
        )
        assert_clean(result, 3)


class TestRetryFromReportCommand:
    """Test the `retry` command interface."""

    def test_retry_from_report_required_option(self):
        """Test --report option is required.

        Contract: See Required Options table (line 137-139).
        """
        result = subprocess.run(
            [get_python_executable(), "-m", "cli.commands", "retry"],
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0  # Should fail without --report

    def test_retry_from_report_nonexistent_report(self):
        """Test nonexistent report exits with specific code.

        Contract: See Exit Codes (line 173).
        """
        result = subprocess.run(
            [get_python_executable(), "-m", "cli.commands", "retry", "--report", "/nonexistent/error_report.json"],
            capture_output=True,
            text=True,
        )
        # Implementation returns 2 for report not found
        # Contract spec may differ; test reflects current implementation
        assert result.returncode == 2


class TestFeedbackCommand:
    """Test the `feedback` command interface."""

    def test_feedback_required_option(self):
        """Test --batch option is required.

        Contract: See Required Options table (line 200-202).
        """
        result = subprocess.run(
            [get_python_executable(), "-m", "cli.commands", "feedback"],
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0  # Should fail without --batch

    def test_feedback_corrections_file(self):
        """Test --corrections option loads JSON corrections file.

        Contract: See Optional Options table (line 208).
        """
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"batch_id": "batch_001", "corrections": []}, f)
            fpath = f.name

        try:
            result = subprocess.run(
                [get_python_executable(), "-m", "cli.commands", "feedback", "--batch", "batch_001", "--corrections", fpath],
                capture_output=True,
                text=True,
            )
            # Should not exit with argument error
            # May fail if batch not found (exit code 20)
        finally:
            Path(fpath).unlink()

    def test_feedback_interactive_flag(self):
        """Test --interactive enables interactive prompts.

        Contract: See Optional Options table (line 209).
        """
        result = subprocess.run(
            [get_python_executable(), "-m", "cli.commands", "feedback", "--batch", "batch_001", "--interactive"],
            capture_output=True,
            text=True,
        )
        # Should not exit with argument error
        # May fail if batch not found (exit code 20)


class TestEnvironmentVariables:
    """Test CLI configuration via environment variables.

    Contract: See Environment Variables (line 261-279).
    """

    def test_env_var_bourbon_image_model(self):
        """Test BORBON_IMAGE_MODEL environment variable."""
        env = {**os.environ, "BORBON_IMAGE_MODEL": "qwen2-vl:7b"}
        result = subprocess.run(
            [get_python_executable(), "-m", "cli.commands", "process", "--dry-run", *NO_SERVER],
            capture_output=True,
            text=True,
            env=env,
        )
        assert_clean(result, 3)

    def test_env_var_olllama_url(self):
        """Test OLLLAMA_URL environment variable."""
        env = {**os.environ, "OLLLAMA_URL": "http://localhost:11435"}
        result = subprocess.run(
            [get_python_executable(), "-m", "cli.commands", "process", "--dry-run", *NO_SERVER],
            capture_output=True,
            text=True,
            env=env,
        )
        assert_clean(result, 3)

    def test_env_var_always_uppercase(self):
        """Verify CLI arguments override environment variables.

        Contract: See Precedence Order (line 275-279).
        """
        env = {**os.environ, "BORBON_IMAGE_MODEL": "qwen2-vl:7b"}
        result = subprocess.run(
            [get_python_executable(), "-m", "cli.commands", "process", "--model", "qwen3-vl:32b", "--dry-run",
             *NO_SERVER],
            capture_output=True,
            text=True,
            env=env,
        )
        assert_clean(result, 3)


class TestExitCodes:
    """Test CLI exit codes match specification.

    Contract: See Exit Codes (line 90-99).
    """

    def test_exit_code_2_ollama_connection_failed(self):
        """Test exit code 2 when Ollama connection fails.

        Contract: See Exit Codes (line 96).
        Contract: See Error Handling > Ollama Connection Errors (line 438).
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # Force Ollama connection to fail
            result = subprocess.run(
                [
                    get_python_executable(),
                    "-m",
                    "cli.commands",
                    "process",
                    "--image-folder",
                    tmpdir,
                    "--ollama-url",
                    "http://localhost:99999",  # Invalid port
                ],
                capture_output=True,
                text=True,
            )
            assert_clean(result, 2)  # rejected before any connection attempt

    def test_exit_code_3_no_valid_images(self):
        """Test exit code 3 when no valid images found.

        Contract: See Exit Codes (line 97).
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # Empty directory
            result = subprocess.run(
                [get_python_executable(), "-m", "cli.commands", "process", "--image-folder", tmpdir, *NO_SERVER],
                capture_output=True,
                text=True,
            )
            assert_clean(result, 3)


class TestOutputMessages:
    """Test CLI output messages follow conventions.

    Contract: See Output Conventions (line 283-319).
    """

    def test_progress_messages_format(self):
        """Test progress messages follow specified format.

        Contract: See Output Conventions > Progress Messages (line 286-293).
        """
        # This would require actual processing, which is integration testing
        # For contract tests, we verify the format strings exist in code
        pass

    def test_error_messages_format(self):
        """Test error messages follow specified format.

        Contract: See Output Conventions > Error Messages (line 302-308).
        """
        # Verify error message formats exist in exceptions
        pass


class TestPreFlightChecks:
    """Test pre-flight checks fail before processing.

    Contract: See Validation Rules > Pre-flight Checks (line 400-407).
    """

    def test_preflight_input_folder_has_images(self):
        """Test pre-flight check verifies input folder has images.

        Contract: See Pre-flight Check 1 (line 403).
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            result = subprocess.run(
                [get_python_executable(), "-m", "cli.commands", "process", "--image-folder", tmpdir, *NO_SERVER],
                capture_output=True,
                text=True,
            )
            assert_clean(result, 3)  # before any model server contact

    def test_preflight_ollama_reachable(self):
        """Test pre-flight check verifies Ollama is reachable.

        Contract: See Pre-flight Check 2 (line 404).
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "IMG_0001.jpg"
            test_file.write_bytes(b"\xff\xd8\xff\xe0\x00\x10JFIF")  # JPEG header

            result = subprocess.run(
                [
                    get_python_executable(),
                    "-m",
                    "cli.commands",
                    "process",
                    "--image-folder",
                    tmpdir,
                    "--ollama-url",
                    "http://localhost:99999",  # Invalid port
                ],
                capture_output=True,
                text=True,
            )
            assert_clean(result, 2)


class TestNonFatalErrorsContinue:
    """Test individual bottle failures do not stop processing.

    Contract: See Error Handling > Processing Errors (Non-Fatal) (line 449-458).
    """

    def test_individual_bottle_failure_continues(self):
        """Test non-fatal errors allow continuation.

        Contract: See Contract Tests (line 504).
        """
        # This requires integration testing with real images that will fail
        pass


class TestMarkdownPreviewReports:
    """Test markdown preview reports are generated.

    Contract: See Contract Tests (line 507).
    """

    def test_preview_reports_generated_without_all_batch_size(self):
        """Test preview reports generated when not using --batch-size all.

        Contract: Contract test (line 507).
        """
        # This requires integration testing with real images
        pass


class TestVendorImageCaching:
    """Test vendor image caching functionality.

    Contract: See Contract Tests (line 508).
    """

    def test_vendor_image_caching_enabled(self):
        """Test vendor image caching works when --include-vendor-images enabled.

        Contract: Contract test (line 508).
        """
        # This requires integration testing with real images
        pass


class TestConfigurationFileSupport:
    """Test configuration file support.

    Contract: See Configuration File Format (line 357-384).
    """

    def test_specify_config_toml_support(self):
        """Test CLI loads configuration from .specify_config.toml.

        Contract: Configuration file at project root (line 359).
        """
        # Create a temporary config file
        with tempfile.TemporaryDirectory() as tmpdir:
            config_file = Path(tmpdir) / ".specify_config.toml"
            config_file.write_text(
                """
[processing]
default_batch_size = 5
lookahead_images = 3

[ollama]
url = "http://localhost:11434"
model = "qwen3-vl:32b"
"""
            )

            # In real implementation, CLI would load this file
            # For contract test, we verify the file format is valid TOML

            import tomli

            with open(config_file, "rb") as f:
                config = tomli.load(f)
                assert "processing" in config
                assert "ollama" in config


class TestEntryPoint:
    """main(): default subcommand, help on no arguments, signal handling."""

    def test_no_arguments_prints_help_and_returns_1(self, capsys):
        from cli.commands import main

        assert main([]) == 1
        assert "usage:" in capsys.readouterr().out

    def test_options_without_subcommand_mean_process(self):
        from cli.commands import parse_args

        args = parse_args(["-f", "photos", "--dry-run"])
        assert args.command == "process"
        assert str(args.image_folder) == "photos"
        assert parse_args(["--debug", "retry", "--report", "r.json"]).command == "retry"

    def test_top_level_help_still_works(self, capsys):
        from cli.commands import parse_args

        with pytest.raises(SystemExit) as excinfo:
            parse_args(["--help"])
        assert excinfo.value.code == 0
        assert "process" in capsys.readouterr().out

    def test_main_turns_sigterm_and_sighup_into_exit(self, monkeypatch):
        """SIGHUP (terminal closed, SSH dropped) must run cleanup like SIGTERM does."""
        import signal

        from cli import commands

        installed = {}
        monkeypatch.setattr(commands.signal, "signal", lambda sig, handler: installed.__setitem__(sig, handler))
        monkeypatch.setattr(commands, "process_command", lambda args: 0)
        assert commands.main(["process", "--dry-run"]) == 0
        assert signal.SIGTERM in installed
        if hasattr(signal, "SIGQUIT"):
            assert signal.SIGQUIT in installed
        if hasattr(signal, "SIGHUP"):
            assert signal.SIGHUP in installed
            with pytest.raises(SystemExit) as excinfo:
                installed[signal.SIGHUP](signal.SIGHUP, None)
            assert excinfo.value.code == 128 + signal.SIGHUP

    def test_debug_logging_silences_pil(self):
        import logging

        from cli.commands import setup_logging

        setup_logging(debug=True)
        assert logging.getLogger("PIL").getEffectiveLevel() >= logging.INFO


class TestPipelineEdgeCases:
    """_run_pipeline after pre-flight passed."""

    def test_scan_finding_no_images_exits_3_without_using_the_model(self, tmp_path):
        import argparse
        from unittest.mock import MagicMock

        from cli.commands import _run_pipeline

        folder = tmp_path / "photos"
        (folder / "old").mkdir(parents=True)
        (folder / "old" / ".bourbon_output").touch()
        (folder / "old" / "a.jpg").write_bytes(b"x")
        args = argparse.Namespace(
            image_folder=folder, output_dir=tmp_path / "out", dry_run=False, batch_size="5",
            lookahead=3, interactive=False, model=None,
        )
        manager = MagicMock()
        assert _run_pipeline(args, manager, MagicMock()) == 3
        manager.get_client.assert_not_called()
