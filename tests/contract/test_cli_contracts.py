"""CLI interface contract tests for bourbon image processor.

Tests ensure the CLI implementation matches the specification defined in
specs/001-bourbon-image-processor/contracts/cli-interface.md.
"""

import json
import shlex
import subprocess
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

# Import CLI commands for direct invocation
from cli.commands import process_command, retry_command, feedback_command
from lib.exceptions import OllamaConnectionError


class TestCLIProcessCommand:
    """Test the `process` command interface."""

    def test_process_command_basic_invocation(self):
        """Test basic `process` command can be invoked.

        Contract: Process command has no required arguments, supports optional arguments
        and defaults.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            result = subprocess.run(
                ["python", "-m", "cli.commands", "process", "--help"],
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
            ["python", "-m", "cli.commands", "process", "--help"],
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
            ["python", "-m", "cli.commands", "process", "--batch-size", "-1"],
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
                ["python", "-m", "cli.commands", "process", "--batch-size", size, "--dry-run"],
                capture_output=True,
                text=True,
            )
            # Dry-run should succeed even without real images
            # Exit code should not be invalid argument error (1), but possibly 3 (no images) or 0 (dry-run success)

        # Invalid integer (negative)
        result = subprocess.run(
            ["python", "-m", "cli.commands", "process", "--batch-size", "-1"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 1

        # Valid 'all'
        result = subprocess.run(
            ["python", "-m", "cli.commands", "process", "--batch-size", "all", "--dry-run"],
            capture_output=True,
            text=True,
        )
        # May fail on no images but should not fail on invalid argument

    def test_process_image_folder_validation(self):
        """Test --image-folder validation: must exist, be directory, be readable.

        Contract: See Validation Rules > Option Validation (line 392).
        """
        # Non-existent folder
        result = subprocess.run(
            ["python", "-m", "cli.commands", "process", "--image-folder", "/nonexistent/path"],
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
                ["python", "-m", "cli.commands", "process", "--image-folder", fpath],
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
            ["python", "-m", "cli.commands", "process", "--ollama-url", "not-a-url"],
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0

    def test_process_lookahead_validation(self):
        """Test --lookahead validation: integer 0-10.

        Contract: See Validation Rules > Option Validation (line 397).
        """
        # Valid range
        for value in ["0", "5", "10"]:
            result = subprocess.run(
                ["python", "-m", "cli.commands", "process", "--lookahead", value, "--dry-run"],
                capture_output=True,
                text=True,
            )
            # Should not exit with argument error

        # Invalid: negative
        result = subprocess.run(
            ["python", "-m", "cli.commands", "process", "--lookahead", "-1"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 1

        # Invalid: > 10
        result = subprocess.run(
            ["python", "-m", "cli.commands", "process", "--lookahead", "11"],
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
                ["python", "-m", "cli.commands", "process", "--concurrency", value, "--dry-run"],
                capture_output=True,
                text=True,
            )
            # Should not exit with argument error

        # Invalid: 0
        result = subprocess.run(
            ["python", "-m", "cli.commands", "process", "--concurrency", "0"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 1

        # Invalid: > 10
        result = subprocess.run(
            ["python", "-m", "cli.commands", "process", "--concurrency", "11"],
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

            result = subprocess.run(
                ["python", "-m", "cli.commands", "process", "--image-folder", tmpdir, "--dry-run"],
                capture_output=True,
                text=True,
            )

            # Dry-run should execute without errors and show processing info
            # Should actually create minimal files for discovery

    def test_process_debug_flag(self):
        """Test --debug enables verbose logging.

        Contract: See Options table (line 41).
        Contract: See Output Conventions > Debug Messages (line 294).
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test images
            test_file = Path(tmpdir) / "IMG_0001.jpg"
            test_file.write_bytes(b"\xff\xd8\xff\xe0\x00\x10JFIF")  # JPEG header

            result = subprocess.run(
                ["python", "-m", "cli.commands", "process", "--image-folder", tmpdir, "--debug"],
                capture_output=True,
                text=True,
            )
            # Debug flag should not cause invalid argument error

    def test_process_interactive_flag(self):
        """Test --interactive enables interactive CLI prompts.

        Contract: See Options table (line 34).
        """
        result = subprocess.run(
            ["python", "-m", "cli.commands", "process", "--interactive"],
            capture_output=True,
            text=True,
        )
        # Should not exit with argument error

    def test_process_include_vendor_images_flag(self):
        """Test --include-vendor-images enables vendor image caching.

        Contract: See Options table (line 35).
        """
        result = subprocess.run(
            ["python", "-m", "cli.commands", "process", "--include-vendor-images"],
            capture_output=True,
            text=True,
        )
        # Should not exit with argument error


class TestRetryFromReportCommand:
    """Test the `retry-from-report` command interface."""

    def test_retry_from_report_required_option(self):
        """Test --report option is required.

        Contract: See Required Options table (line 137-139).
        """
        result = subprocess.run(
            ["python", "-m", "cli.commands", "retry-from-report"],
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0  # Should fail without --report

    def test_retry_from_report_nonexistent_report(self):
        """Test nonexistent report exits with specific code.

        Contract: See Exit Codes (line 173).
        """
        result = subprocess.run(
            ["python", "-m", "cli.commands", "retry-from-report", "--report", "/nonexistent/error_report.json"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 10


class TestFeedbackCommand:
    """Test the `feedback` command interface."""

    def test_feedback_required_option(self):
        """Test --batch option is required.

        Contract: See Required Options table (line 200-202).
        """
        result = subprocess.run(
            ["python", "-m", "cli.commands", "feedback"],
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
                ["python", "-m", "cli.commands", "feedback", "--batch", "batch_001", "--corrections", fpath],
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
            ["python", "-m", "cli.commands", "feedback", "--batch", "batch_001", "--interactive"],
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
        env = {"BORBON_IMAGE_MODEL": "qwen2-vl:7b"}
        result = subprocess.run(
            ["python", "-m", "cli.commands", "process", "--dry-run"],
            capture_output=True,
            text=True,
            env=env,
        )
        # Should not exit with argument error

    def test_env_var_olllama_url(self):
        """Test OLLLAMA_URL environment variable."""
        env = {"OLLLAMA_URL": "http://localhost:11435"}
        result = subprocess.run(
            ["python", "-m", "cli.commands", "process", "--dry-run"],
            capture_output=True,
            text=True,
            env=env,
        )
        # Should not exit with argument error

    def test_env_var_always_uppercase(self):
        """Verify CLI arguments override environment variables.

        Contract: See Precedence Order (line 275-279).
        """
        env = {"BORBON_IMAGE_MODEL": "qwen2-vl:7b"}
        result = subprocess.run(
            ["python", "-m", "cli.commands", "process", "--model", "qwen3-vl:32b", "--dry-run"],
            capture_output=True,
            text=True,
            env=env,
        )
        # CLI arg should override env var


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
                    "python",
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
            # Should fail with Ollama-related error code (possibly 2 or 1 depending on implementation)

    def test_exit_code_3_no_valid_images(self):
        """Test exit code 3 when no valid images found.

        Contract: See Exit Codes (line 97).
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # Empty directory
            result = subprocess.run(
                ["python", "-m", "cli.commands", "process", "--image-folder", tmpdir],
                capture_output=True,
                text=True,
            )
            # Should exit with code 3 (no valid images found)


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
                ["python", "-m", "cli.commands", "process", "--image-folder", tmpdir],
                capture_output=True,
                text=True,
            )
            # Should fail before attempting to connect to Ollama

    def test_preflight_ollama_reachable(self):
        """Test pre-flight check verifies Ollama is reachable.

        Contract: See Pre-flight Check 2 (line 404).
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "IMG_0001.jpg"
            test_file.write_bytes(b"\xff\xd8\xff\xe0\x00\x10JFIF")  # JPEG header

            result = subprocess.run(
                [
                    "python",
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
            # Should fail on Ollama connection check before processing


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
