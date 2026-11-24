"""CLI interface contract tests for bourbon image processor.

Tests ensure the CLI implementation matches the specification defined in
specs/001-bourbon-image-processor/contracts/cli-interface.md.

Every test (and every CLI subprocess it starts) runs from its own
``tmp_path`` (see the autouse ``_isolate_environment`` fixture in
``tests/conftest.py``), so the ``process`` defaults ``-f .`` and
``-o processed`` never touch the repo.
"""

import json
import os
import shutil
import subprocess
import sys

import pytest
from PIL import Image as PILImage


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
        assert "configuration error" in result.stderr
        assert "batch_size=-1 (from flag -b/--batch-size)" in result.stderr

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

    def test_process_image_folder_validation(self, tmp_path):
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
        fpath = tmp_path / "not_a_dir.txt"
        fpath.write_text("test")
        result = subprocess.run(
            [get_python_executable(), "-m", "cli.commands", "process", "--image-folder", str(fpath)],
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0  # Should fail

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

    def test_process_dry_run_flag(self, tmp_path):
        """Test --dry-run shows what would be processed without actual processing.

        Contract: See Options table (line 42).
        """
        photos = tmp_path / "photos"
        photos.mkdir()
        (photos / "IMG_0001.jpg").write_bytes(b"\xff\xd8\xff\xe0\x00\x10JFIF")  # JPEG header

        out = tmp_path / "out"
        result = subprocess.run(
            [
                get_python_executable(), "-m", "cli.commands", "process", "--image-folder", str(photos),
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

    def test_process_debug_flag(self, tmp_path):
        """Test --debug enables verbose logging.

        Contract: See Options table (line 41).
        Contract: See Output Conventions > Debug Messages (line 294).
        """
        photos = tmp_path / "photos"
        photos.mkdir()
        (photos / "IMG_0001.jpg").write_bytes(b"\xff\xd8\xff\xe0\x00\x10JFIF")  # JPEG header

        # --dry-run and a closed server port keep this a pure flag test.
        result = subprocess.run(
            [
                get_python_executable(), "-m", "cli.commands", "process",
                "--image-folder", str(photos), "-o", str(tmp_path / "out"), "--debug", "--dry-run",
                *NO_SERVER,
            ],
            capture_output=True,
            text=True,
        )
        assert_clean(result, expected_dry_run_exit())
        assert " DEBUG [-] " in result.stdout
        assert "TiffImagePlugin" not in result.stdout  # PIL debug noise is silenced

    def test_process_interactive_flag(self):
        """Test --interactive enables interactive CLI prompts.

        Contract: See Options table (line 34).
        """
        # The conftest runs this in an empty tmp_path: no images, exit 3
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
        # Runs in an empty tmp_path (conftest): no images, exit 3.
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
    """Test the `feedback` command interface (run in an empty tmp_path)."""

    def test_feedback_without_any_batch_exits_2(self):
        """No --batch and no batch preview in the output directory: batch not found."""
        result = subprocess.run(
            [get_python_executable(), "-m", "cli.commands", "feedback"],
            capture_output=True,
            text=True,
        )
        assert_clean(result, 2)

    def test_feedback_corrections_file_batch_not_found_exits_2(self, tmp_path):
        """--corrections is accepted; a missing batch still exits 2 (checked first)."""
        fpath = tmp_path / "corrections.json"
        fpath.write_text(json.dumps([]))
        result = subprocess.run(
            [
                get_python_executable(), "-m", "cli.commands", "feedback",
                "--batch", "batch_001", "--corrections", str(fpath),
            ],
            capture_output=True,
            text=True,
        )
        assert_clean(result, 2)
        assert "batch_001 not found" in result.stdout

    def test_feedback_interactive_batch_not_found_exits_2(self):
        """--interactive is accepted; a missing batch exits 2 before any prompt."""
        result = subprocess.run(
            [get_python_executable(), "-m", "cli.commands", "feedback", "--batch", "batch_001", "--interactive"],
            capture_output=True,
            text=True,
            stdin=subprocess.DEVNULL,
        )
        assert_clean(result, 2)


class TestExitCodes:
    """Test CLI exit codes match specification.

    Contract: See Exit Codes (line 90-99).
    """

    def test_exit_code_1_invalid_ollama_port(self, tmp_path):
        """An --ollama-url whose port is out of range is a configuration error (exit 1)."""
        result = subprocess.run(
            [
                get_python_executable(), "-m", "cli.commands", "process",
                "--image-folder", str(tmp_path), "--ollama-url", "http://localhost:99999",
            ],
            capture_output=True,
            text=True,
        )
        assert_clean(result, 1)  # rejected before any connection attempt
        assert "ollama_url='http://localhost:99999' (from flag -u/--ollama-url)" in result.stderr

    def test_exit_code_3_no_valid_images(self, tmp_path):
        """Test exit code 3 when no valid images found.

        Contract: See Exit Codes (line 97).
        """
        result = subprocess.run(
            [get_python_executable(), "-m", "cli.commands", "process", "--image-folder", str(tmp_path), *NO_SERVER],
            capture_output=True,
            text=True,
        )
        assert_clean(result, 3)  # before any model server contact

    def test_unreachable_ollama_exits_2(self, tmp_path):
        """A valid --ollama-url with nothing listening, and no autostart: exit 2.

        Contract: See Pre-flight Check 2 (line 404) and Error Handling >
        Ollama Connection Errors (line 438). Exit 1 instead when Tesseract is
        missing (checked before the model server).
        """
        photos = tmp_path / "photos"
        photos.mkdir()
        PILImage.new("RGB", (32, 32), "white").save(photos / "IMG_0001.jpg", "JPEG")
        result = subprocess.run(
            [
                get_python_executable(), "-m", "cli.commands", "process",
                "--image-folder", str(photos), "-o", str(tmp_path / "out"),
                *NO_SERVER, "--no-server-autostart",
            ],
            capture_output=True,
            text=True,
        )
        assert_clean(result, 1 if shutil.which("tesseract") is None else 2)


def run_cli(*argv, env=None, cwd=None):
    """Run the CLI in a subprocess (the scrubbed test environment plus ``env``)."""
    return subprocess.run(
        [get_python_executable(), "-m", "cli.commands", *argv],
        capture_output=True,
        text=True,
        env={**os.environ, **(env or {})},
        cwd=cwd,
    )


class TestConfiguration:
    """Config files, environment and flags resolve to one validated Settings (exit 1 on error)."""

    def test_bad_project_file_value_exits_1_naming_file_and_key(self, tmp_path):
        (tmp_path / "bourbon.toml").write_text('[processing]\nbatch_size = "abc"\n')
        result = run_cli("process", "--dry-run", *NO_SERVER)
        assert_clean(result, 1)
        assert "configuration error" in result.stderr
        assert "batch_size='abc'" in result.stderr
        assert str(tmp_path / "bourbon.toml") in result.stderr

    def test_no_config_ignores_the_file(self, tmp_path):
        (tmp_path / "bourbon.toml").write_text('[processing]\nbatch_size = "abc"\n')
        result = run_cli("--no-config", "process", "--dry-run", *NO_SERVER)
        assert_clean(result, 3)  # past configuration; the empty folder has no images

    def test_unknown_key_in_file_exits_1(self, tmp_path):
        (tmp_path / "bourbon.toml").write_text("[processing]\nlookahead = 3\n")
        result = run_cli("process", "--dry-run", *NO_SERVER)
        assert_clean(result, 1)
        assert "unknown key 'lookahead'" in result.stderr

    def test_config_show_json_lists_sources(self, tmp_path):
        (tmp_path / "bourbon.toml").write_text("[processing]\nconcurrency = 4\n")
        result = run_cli("config", "show", "--json", env={"BOURBON_BATCH_SIZE": "all"})
        assert_clean(result, 0)
        data = json.loads(result.stdout)
        assert data["values"]["processing.concurrency"] == 4
        assert data["sources"]["processing.concurrency"] == f"file:{tmp_path / 'bourbon.toml'}"
        assert data["sources"]["processing.batch_size"] == "env:BOURBON_BATCH_SIZE"
        assert data["sources"]["model.vision_model"] == "default"
        assert data["web_provider"]["resolved"] == "duckduckgo"

    def test_config_show_never_prints_key_values(self):
        result = run_cli("config", "show", env={"BRAVE_API_KEY": "sekrit-xyz"})
        assert_clean(result, 0)
        assert "sekrit-xyz" not in result.stdout + result.stderr
        assert "BRAVE_API_KEY set" in result.stdout
        assert "Web search provider: brave" in result.stdout

    def test_config_template_parses_as_toml(self):
        import tomllib

        from lib.settings import template_defaults

        result = run_cli("config", "template")
        assert_clean(result, 0)
        assert tomllib.loads(result.stdout) == template_defaults()

    def test_env_concurrency_0_exits_1_naming_the_variable(self):
        result = run_cli("process", "--dry-run", *NO_SERVER, env={"BOURBON_CONCURRENCY": "0"})
        assert_clean(result, 1)
        assert "concurrency=0 (from env BOURBON_CONCURRENCY): must be between 1 and 10" in result.stderr

    def test_concurrency_11_exits_1(self):
        result = run_cli("process", "--concurrency", "11", "--dry-run", *NO_SERVER)
        assert_clean(result, 1)
        assert "concurrency=11 (from flag --concurrency)" in result.stderr

    def test_vendor_images_without_web_exits_1(self):
        result = run_cli("process", "-v", "--no-web", "--dry-run", *NO_SERVER)
        assert_clean(result, 1)
        assert "need web lookups" in result.stderr

    def test_brave_without_key_exits_1_naming_the_variable(self, tmp_path):
        photos = tmp_path / "photos"
        photos.mkdir()
        (photos / "IMG_0001.jpg").write_bytes(b"\xff\xd8\xff\xe0\x00\x10JFIF")
        result = run_cli("process", "-f", str(photos), "--web-provider", "brave", "--dry-run", *NO_SERVER)
        assert_clean(result, 1)
        assert "BRAVE_API_KEY" in result.stderr

    def test_all_errors_reported_together(self):
        result = run_cli(
            "process", "--concurrency", "0", "--retry-attempts", "9", "--dry-run", *NO_SERVER,
            env={"BOURBON_INTERACTIVE": "maybe"},
        )
        assert_clean(result, 1)
        assert result.stderr.count("\n  - ") == 3

    def test_dry_run_reports_stages_and_provider(self, tmp_path):
        photos = tmp_path / "photos"
        photos.mkdir()
        (photos / "IMG_0001.jpg").write_bytes(b"\xff\xd8\xff\xe0\x00\x10JFIF")
        result = run_cli("process", "-f", str(photos), "-o", str(tmp_path / "out"), "--dry-run", *NO_SERVER)
        assert_clean(result, expected_dry_run_exit())
        if result.returncode == 0:
            assert "Dry run: active stages: METADATA_GENERATION, VISION_ANALYSIS" in result.stdout
            assert "Dry run: text model: gpt-oss:120b (fact check on)" in result.stdout
            assert "Dry run: web search provider: duckduckgo" in result.stdout

    @pytest.mark.parametrize("flag", [["--lookahead", "3"], ["--server-backend", "ollama"], ["--pull-missing"]])
    def test_removed_flags_exit_2(self, flag):
        result = run_cli("process", *flag)
        assert result.returncode == 2
        assert "unrecognized arguments" in result.stderr


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
        from unittest.mock import MagicMock

        from cli.commands import RunContext, _run_pipeline
        from tests.helpers import make_settings

        folder = tmp_path / "photos"
        (folder / "old").mkdir(parents=True)
        (folder / "old" / ".bourbon_output").touch()
        (folder / "old" / "a.jpg").write_bytes(b"x")
        manager = MagicMock()
        ctx = RunContext(
            make_settings(image_folder=folder, output_dir=tmp_path / "out"),
            server_manager=manager,
            model_selector=MagicMock(),
        )
        assert _run_pipeline(ctx) == 3
        manager.get_client.assert_not_called()
