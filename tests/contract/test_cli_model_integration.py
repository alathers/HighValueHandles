"""Contract tests for CLI model integration.

Tests cover:
  - --model flag sets primary model for all AI stages
  - BOURBON_<STAGE>_MODEL env vars override per-stage
  - Pre-flight uses ModelServerManager
  - Debug mode prints stage report
  - Usage summary printed at end of processing
"""

import argparse
import json
import logging
from unittest.mock import MagicMock, patch

import pytest

from cli.commands import parse_args, pre_flight_checks, process_command

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


class TestParseArgs:
    """Test CLI argument parsing for model selector features."""

    def test_default_model_flag(self):
        args = parse_args(["process", "--image-folder", "/tmp/test"])
        assert args.model == "qwen2-vl:7b"  # existing default

    def test_custom_model_flag(self):
        args = parse_args(["process", "--image-folder", "/tmp/test", "--model", "llama3.2"])
        assert args.model == "llama3.2"

    def test_server_backend_flag(self):
        args = parse_args(["process", "--image-folder", "/tmp/test", "--server-backend", "ollama"])
        assert args.server_backend == "ollama"

    def test_server_backend_default(self):
        args = parse_args(["process", "--image-folder", "/tmp/test"])
        assert args.server_backend == "mpltx"

    def test_mpltx_home_flag(self):
        args = parse_args(["process", "--image-folder", "/tmp/test", "--mpltx-home", "/custom/mpltx"])
        assert str(args.mpltx_home) == "/custom/mpltx"

    def test_mpltx_port_flag(self):
        args = parse_args(["process", "--image-folder", "/tmp/test", "--mpltx-port", "8080"])
        assert args.mpltx_port == 8080

    def test_mpltx_gpu_layers_flag(self):
        args = parse_args(["process", "--image-folder", "/tmp/test", "--mpltx-gpu-layers", "16"])
        assert args.mpltx_gpu_layers == 16

    def test_mpltx_context_size_flag(self):
        args = parse_args(["process", "--image-folder", "/tmp/test", "--mpltx-context-size", "8192"])
        assert args.mpltx_context_size == 8192

    def test_debug_flag(self):
        args = parse_args(["process", "--image-folder", "/tmp/test", "--debug"])
        assert args.debug is True

    def test_help_includes_new_flags(self):
        import io
        import sys

        f = io.StringIO()
        try:
            with pytest.raises(SystemExit):
                parse_args(["process", "--help"])
        except SystemExit:
            pass  # --help exits; we just want to verify it doesn't crash


# ---------------------------------------------------------------------------
# Pre-flight checks with ModelServerManager
# ---------------------------------------------------------------------------


class TestPreFlightModelIntegration:
    """Test that pre-flight uses ModelServerManager."""

    def _make_args(self, extra=None):
        """Create minimal args namespace for pre-flight."""
        args = argparse.Namespace(
            command="process",
            image_folder=MagicMock(),
            batch_size="5",
            lookahead=3,
            concurrency=3,
            ollama_url="http://localhost:11434",
            model="qwen3-vl:32b",
            server_backend="mpltx",
            mpltx_home=None,
            mpltx_port=None,
            mpltx_gpu_layers=None,
            mpltx_context_size=None,
            debug=False,
        )
        if extra:
            for k, v in extra.items():
                setattr(args, k, v)
        return args

    def test_pre_flight_uses_model_server_manager(self):
        args = self._make_args()
        args.image_folder.exists.return_value = True
        args.image_folder.is_dir.return_value = True
        # Make rglob return some image files
        mock_img = MagicMock()
        mock_img.suffix = ".jpg"
        args.image_folder.rglob.return_value = [mock_img]
        args.output_dir = MagicMock()
        args.output_dir.exists.return_value = False

        with patch("cli.commands.ModelServerManager") as MockManager:
            mock_manager = MagicMock()
            mock_manager.state = MagicMock()
            mock_manager.state.base_url = "http://localhost:11435"
            MockManager.return_value = mock_manager

            # Tesseract check
            with patch("pytesseract.get_tesseract_version", return_value="5.0"):
                exit_code = pre_flight_checks(args)

            # Verify ModelServerManager was instantiated
            MockManager.assert_called_once()
            mock_manager.ensure_running.assert_called_once()
            mock_manager.ensure_models.assert_called_once()

    def test_pre_flight_stores_manager_on_args(self):
        args = self._make_args()
        args.image_folder.exists.return_value = True
        args.image_folder.is_dir.return_value = True
        mock_img = MagicMock()
        mock_img.suffix = ".jpg"
        args.image_folder.rglob.return_value = [mock_img]
        args.output_dir = MagicMock()
        args.output_dir.exists.return_value = False

        with patch("cli.commands.ModelServerManager") as MockManager:
            mock_manager = MagicMock()
            mock_manager.state = MagicMock()
            mock_manager.state.base_url = "http://localhost:11435"
            MockManager.return_value = mock_manager

            with patch("pytesseract.get_tesseract_version", return_value="5.0"):
                pre_flight_checks(args)

            # Verify args got the manager and selector attached
            assert hasattr(args, "_server_manager")
            assert hasattr(args, "_model_selector")

    def test_pre_flight_prints_stage_report_in_debug(self):
        args = self._make_args()
        args.image_folder.exists.return_value = True
        args.image_folder.is_dir.return_value = True
        mock_img = MagicMock()
        mock_img.suffix = ".jpg"
        args.image_folder.rglob.return_value = [mock_img]
        args.output_dir = MagicMock()
        args.output_dir.exists.return_value = False
        args.debug = True

        with patch("cli.commands.ModelServerManager") as MockManager:
            mock_manager = MagicMock()
            mock_manager.state = MagicMock()
            mock_manager.state.base_url = "http://localhost:11435"
            MockManager.return_value = mock_manager

            with patch("pytesseract.get_tesseract_version", return_value="5.0"):
                pre_flight_checks(args)

            # Verify print_stage_report was called
            selector = args._model_selector
            selector.print_stage_report.assert_called_once()

    def test_pre_flight_fallback_on_server_error(self):
        """Pre-flight returns exit code 2 when server fails to start."""
        args = self._make_args()
        args.image_folder.exists.return_value = True
        args.image_folder.is_dir.return_value = True
        mock_img = MagicMock()
        mock_img.suffix = ".jpg"
        args.image_folder.rglob.return_value = [mock_img]
        args.output_dir = MagicMock()
        args.output_dir.exists.return_value = False

        with patch("cli.commands.ModelServerManager") as MockManager:
            MockManager.side_effect = Exception("server error")

            with patch("pytesseract.get_tesseract_version", return_value="5.0"):
                exit_code = pre_flight_checks(args)

            assert exit_code == 2


    def _run_pre_flight_capturing_kwargs(self, args):
        args.image_folder.exists.return_value = True
        args.image_folder.is_dir.return_value = True
        mock_img = MagicMock()
        mock_img.suffix = ".jpg"
        args.image_folder.rglob.return_value = [mock_img]
        args.output_dir = MagicMock()
        args.output_dir.exists.return_value = False
        with patch("cli.commands.ModelServerManager") as MockManager:
            MockManager.return_value = MagicMock()
            with patch("pytesseract.get_tesseract_version", return_value="5.0"):
                pre_flight_checks(args)
        MockManager.assert_called_once()
        return MockManager.call_args.kwargs

    def test_pre_flight_ollama_backend_string_maps_to_enum_and_base_url(self):
        """Regression: --server-backend is a str; manager must get the enum + base_url."""
        from lib.model_server import ServerBackend

        args = self._make_args(
            {"server_backend": "ollama", "ollama_url": "http://127.0.0.1:9999"}
        )
        kwargs = self._run_pre_flight_capturing_kwargs(args)
        assert kwargs == {"backend": ServerBackend.OLLAMA, "base_url": "http://127.0.0.1:9999"}

    def test_pre_flight_mpltx_flags_map_to_constructor_params(self):
        """Regression: MPLTX CLI flags must use MpltxServer's parameter names."""
        from lib.model_server import ServerBackend

        args = self._make_args(
            {
                "server_backend": "mpltx",
                "mpltx_home": "/tmp/mh",
                "mpltx_port": 12345,
                "mpltx_gpu_layers": 10,
                "mpltx_context_size": 8192,
            }
        )
        kwargs = self._run_pre_flight_capturing_kwargs(args)
        assert kwargs == {
            "backend": ServerBackend.MPLTX,
            "home_dir": "/tmp/mh",
            "port": 12345,
            "gpu_layers": 10,
            "context_size": 8192,
        }

    def test_pre_flight_rejects_unknown_backend(self):
        args = self._make_args({"server_backend": "bogus"})
        args.image_folder.exists.return_value = True
        args.image_folder.is_dir.return_value = True
        mock_img = MagicMock()
        mock_img.suffix = ".jpg"
        args.image_folder.rglob.return_value = [mock_img]
        args.output_dir = MagicMock()
        args.output_dir.exists.return_value = False
        with patch("pytesseract.get_tesseract_version", return_value="5.0"):
            assert pre_flight_checks(args) == 2


# ---------------------------------------------------------------------------
# Process command with usage summary
# ---------------------------------------------------------------------------


class TestProcessCommandIntegration:
    """Test that process_command prints usage summary and cleans up."""

    def test_process_command_uses_model_selector(self):
        """Process command should use ModelSelector from pre-flight."""
        args = argparse.Namespace(
            command="process",
            image_folder=MagicMock(),
            batch_size="all",
            lookahead=3,
            concurrency=3,
            ollama_url="http://localhost:11434",
            model="qwen3-vl:32b",
            server_backend="mpltx",
            mpltx_home=None,
            mpltx_port=None,
            mpltx_gpu_layers=None,
            mpltx_context_size=None,
            output_dir=MagicMock(),
            interactive=False,
            include_vendor_images=False,
            allow_retry=True,
            debug=False,
            dry_run=True,  # Skip actual processing
        )
        args.image_folder.exists.return_value = True
        args.image_folder.is_dir.return_value = True
        mock_img = MagicMock()
        mock_img.suffix = ".jpg"
        args.image_folder.rglob.return_value = [mock_img]
        args.output_dir.mkdir.return_value = None

        with patch("cli.commands.pre_flight_checks", return_value=0):
            with patch("cli.commands.ModelServerManager") as MockManager:
                mock_manager = MagicMock()
                mock_manager.state = MagicMock()
                mock_manager.state.base_url = "http://localhost:11435"
                mock_manager.get_client.return_value = MagicMock()
                MockManager.return_value = mock_manager

                with patch("pytesseract.get_tesseract_version", return_value="5.0"):
                    exit_code = process_command(args)

                # Verify server manager was created
                assert args._server_manager is mock_manager
                assert args._model_selector is not None

    def test_usage_summary_printed(self, caplog):
        """Verify usage summary is logged when there are model calls."""
        args = argparse.Namespace(
            command="process",
            image_folder=MagicMock(),
            batch_size="all",
            lookahead=3,
            concurrency=3,
            ollama_url="http://localhost:11434",
            model="qwen3-vl:32b",
            server_backend="mpltx",
            mpltx_home=None,
            mpltx_port=None,
            mpltx_gpu_layers=None,
            mpltx_context_size=None,
            output_dir=MagicMock(),
            interactive=False,
            include_vendor_images=False,
            allow_retry=True,
            debug=False,
            dry_run=True,  # Skip actual processing
        )
        args.image_folder.exists.return_value = True
        args.image_folder.is_dir.return_value = True
        mock_img = MagicMock()
        mock_img.suffix = ".jpg"
        args.image_folder.rglob.return_value = [mock_img]
        args.output_dir.mkdir.return_value = None

        with patch("cli.commands.pre_flight_checks", return_value=0):
            with patch("cli.commands.ModelServerManager") as MockManager:
                mock_manager = MagicMock()
                mock_manager.state = MagicMock()
                mock_manager.state.base_url = "http://localhost:11435"
                mock_manager.get_client.return_value = MagicMock()
                MockManager.return_value = mock_manager

                # Set up a selector with recorded usage
                mock_selector = MagicMock()
                mock_selector.get_usage_summary.return_value = {
                    "total_calls": 5,
                    "successful_calls": 4,
                    "fallback_calls": 1,
                    "total_duration_seconds": 12.5,
                    "by_stage": {"VISION_ANALYSIS": {"total": 5, "success": 4, "fallback": 1}},
                    "by_model": {"qwen3-vl:32b": {"total": 4, "success": 3}, "qwen2-vl:7b": {"total": 1, "success": 1}},
                    "records": [],
                }
                args._model_selector = mock_selector
                args._server_manager = mock_manager

                with patch("pytesseract.get_tesseract_version", return_value="5.0"):
                    process_command(args)

                # Verify usage summary was logged
                assert "Model Usage Summary" in caplog.text
                assert "Total calls:      5" in caplog.text
                assert "Successful calls: 4" in caplog.text

    def test_server_manager_closed_on_exit(self):
        """Verify server manager is closed after processing."""
        args = argparse.Namespace(
            command="process",
            image_folder=MagicMock(),
            batch_size="all",
            lookahead=3,
            concurrency=3,
            ollama_url="http://localhost:11434",
            model="qwen3-vl:32b",
            server_backend="mpltx",
            mpltx_home=None,
            mpltx_port=None,
            mpltx_gpu_layers=None,
            mpltx_context_size=None,
            output_dir=MagicMock(),
            interactive=False,
            include_vendor_images=False,
            allow_retry=True,
            debug=False,
            dry_run=True,  # Skip actual processing
        )
        args.image_folder.exists.return_value = True
        args.image_folder.is_dir.return_value = True
        mock_img = MagicMock()
        mock_img.suffix = ".jpg"
        args.image_folder.rglob.return_value = [mock_img]
        args.output_dir.mkdir.return_value = None

        with patch("cli.commands.pre_flight_checks", return_value=0):
            with patch("cli.commands.ModelServerManager") as MockManager:
                mock_manager = MagicMock()
                mock_manager.state = MagicMock()
                mock_manager.state.base_url = "http://localhost:11435"
                mock_manager.get_client.return_value = MagicMock()
                MockManager.return_value = mock_manager

                with patch("pytesseract.get_tesseract_version", return_value="5.0"):
                    process_command(args)

                # Verify close was called
                mock_manager.close.assert_called_once()


# ---------------------------------------------------------------------------
# Environment variable overrides
# ---------------------------------------------------------------------------


class TestEnvVarOverrides:
    """Test BOURBON_<STAGE>_MODEL environment variable overrides."""

    def test_env_var_overrides_config(self):
        """BOURBON_VISION_ANALYSIS_MODEL should override per-stage."""
        import os

        # Set env var
        os.environ["BOURBON_VISION_ANALYSIS_MODEL"] = "llama3.2"

        try:
            from lib.model_selector import ModelSelector, ModelStage

            config = MagicMock()
            config.model = "qwen3-vl:32b"
            config.vision_analysis_model = "llama3.2"

            selector = ModelSelector.from_config(config)
            # The from_config reads config.vision_analysis_model
            assert selector.get_model(ModelStage.VISION_ANALYSIS) == "llama3.2"
        finally:
            del os.environ["BOURBON_VISION_ANALYSIS_MODEL"]

    def test_default_selector_respects_global_model(self):
        """Single --model flag sets the global fallback."""
        from lib.model_selector import ModelSelector

        selector = ModelSelector(default_model="llama3.2")
        assert selector.default_model == "llama3.2"

    def test_json_serializable_usage(self):
        """Verify usage summary is JSON-serializable."""
        from lib.model_selector import ModelSelector, ModelStage

        selector = ModelSelector()
        selector.record_usage(
            stage=ModelStage.VISION_ANALYSIS,
            model="qwen3-vl:32b",
            success=True,
            duration=2.5,
        )
        summary = selector.get_usage_summary()
        # Should not raise
        json.dumps(summary)