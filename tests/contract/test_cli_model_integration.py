"""Contract tests for CLI model integration.

Tests cover:
  - --model flag sets primary model for all AI stages
  - BOURBON_<STAGE>_MODEL env vars override per-stage
  - --server-backend accepts only 'ollama'; unknown or removed backends exit 2
  - Pre-flight uses ModelServerManager (server checked last, failures exit 2,
    dry runs only probe, nothing is pulled by default)
  - Debug mode prints stage report
  - Usage summary printed at end of processing, manager always closed
"""

import argparse
import inspect
import json
import logging
import os
from unittest.mock import MagicMock, patch

import pytest

from cli.commands import (
    parse_args,
    pre_flight_checks,
    process_command,
    resolve_vision_model,
    retry_command,
)
from lib.config import DEFAULT_VISION_MODEL
from lib.model_availability import GiB
from lib.model_server import OllamaServer, ServerBackend

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


class TestParseArgs:
    """Test CLI argument parsing for model selector features."""

    def test_default_model_flag(self):
        args = parse_args(["process", "--image-folder", "/tmp/test"])
        assert args.model is None  # resolved later by cli.commands.resolve_vision_model

    def test_custom_model_flag(self):
        args = parse_args(["process", "--image-folder", "/tmp/test", "--model", "llama3.2"])
        assert args.model == "llama3.2"

    def test_server_backend_flag(self):
        args = parse_args(["process", "--image-folder", "/tmp/test", "--server-backend", "ollama"])
        assert args.server_backend == "ollama"

    def test_server_backend_default(self, monkeypatch):
        monkeypatch.delenv("BOURBON_SERVER_BACKEND", raising=False)
        args = parse_args(["process", "--image-folder", "/tmp/test"])
        assert args.server_backend == "ollama"

    def test_server_backend_case_insensitive(self):
        args = parse_args(["process", "--image-folder", "/tmp/test", "--server-backend", "OLLAMA"])
        assert args.server_backend == "ollama"

    def test_server_backend_rejects_unknown(self, capsys):
        with pytest.raises(SystemExit) as excinfo:
            parse_args(["process", "--server-backend", "bogus"])
        assert excinfo.value.code == 2
        assert "Unknown backend" in capsys.readouterr().err

    @pytest.mark.parametrize("value", ["mtplx", "mpltx"])
    def test_server_backend_rejects_removed_backend(self, value, capsys):
        with pytest.raises(SystemExit) as excinfo:
            parse_args(["process", "--server-backend", value])
        assert excinfo.value.code == 2
        err = capsys.readouterr().err
        assert "MTPLX support was removed" in err
        assert "Use 'ollama'" in err

    def test_server_backend_env_default_is_validated(self, monkeypatch):
        """$BOURBON_SERVER_BACKEND goes through the same validation as the flag."""
        monkeypatch.setenv("BOURBON_SERVER_BACKEND", "Ollama")
        assert parse_args(["process", "--image-folder", "/tmp/test"]).server_backend == "ollama"
        monkeypatch.setenv("BOURBON_SERVER_BACKEND", "bogus")
        with pytest.raises(SystemExit) as excinfo:
            parse_args(["process", "--image-folder", "/tmp/test"])
        assert excinfo.value.code == 2

    @pytest.mark.parametrize(
        "flag", ["--mtplx-model", "--mtplx-port", "--mpltx-port", "--mpltx-home", "--mpltx-gpu-layers"]
    )
    def test_removed_backend_flags_are_rejected(self, flag):
        with pytest.raises(SystemExit) as excinfo:
            parse_args(["process", "--image-folder", "/tmp/test", flag, "1"])
        assert excinfo.value.code == 2

    def test_pull_flag_and_alias(self):
        assert parse_args(["process"]).pull is False
        assert parse_args(["process", "--pull"]).pull is True
        assert parse_args(["process", "--pull-missing"]).pull is True

    def test_min_free_gb_flag(self):
        assert parse_args(["process"]).min_free_gb == 5.0
        assert parse_args(["process", "--pull", "--min-free-gb", "10"]).min_free_gb == 10.0
        assert parse_args(["process", "--min-free-gb", "0.5"]).min_free_gb == 0.5

    def test_resolve_vision_model_precedence(self, monkeypatch):
        monkeypatch.delenv("BOURBON_MODEL", raising=False)
        assert resolve_vision_model(parse_args(["process"])) == DEFAULT_VISION_MODEL == "qwen3.6:35b-a3b"
        monkeypatch.setenv("BOURBON_MODEL", "gemma4:26b")
        assert resolve_vision_model(parse_args(["process"])) == "gemma4:26b"
        assert resolve_vision_model(parse_args(["process", "-m", "qwen3.8:27b"])) == "qwen3.8:27b"

    def test_request_timeout_and_alias(self):
        assert parse_args(["process"]).request_timeout == 600.0
        assert parse_args(["process", "--request-timeout", "90"]).request_timeout == 90.0
        assert parse_args(["process", "--vision-timeout", "45"]).request_timeout == 45.0

    def test_server_lifecycle_flags(self):
        args = parse_args(
            ["process", "--keep-server", "--no-server-autostart", "--server-start-timeout", "12"]
        )
        assert args.keep_server is True
        assert args.no_server_autostart is True
        assert args.server_start_timeout == 12.0

    def test_retry_accepts_server_backend(self):
        args = parse_args(
            [
                "retry", "--report", "r.json", "--server-backend", "ollama",
                "--keep-server", "-m", "llama3.2", "-u", "http://127.0.0.1:1234",
            ]
        )
        assert args.server_backend == "ollama"
        assert args.keep_server is True
        assert args.model == "llama3.2"
        assert args.ollama_url == "http://127.0.0.1:1234"

    def test_retry_model_default_is_none(self):
        assert parse_args(["retry", "--report", "r.json"]).model is None

    def test_debug_flag(self):
        args = parse_args(["process", "--image-folder", "/tmp/test", "--debug"])
        assert args.debug is True

    def test_debug_flag_before_subcommand(self):
        args = parse_args(["--debug", "process", "--image-folder", "/tmp/test"])
        assert args.debug is True

    def test_debug_default_false(self):
        assert parse_args(["process", "--image-folder", "/tmp/test"]).debug is False
        assert parse_args(["retry", "--report", "r.json"]).debug is False

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

    @pytest.fixture(autouse=True)
    def _folders(self, tmp_path):
        """A real photo folder with one image and a not-yet-created output dir."""
        self.image_folder = tmp_path / "photos"
        self.image_folder.mkdir()
        (self.image_folder / "IMG_1.jpg").write_bytes(b"\xff\xd8\xff\xe0\x00\x10JFIF")
        self.output_dir = tmp_path / "out"

    def _make_args(self, extra=None):
        """Create minimal args namespace for pre-flight."""
        args = argparse.Namespace(
            command="process",
            image_folder=self.image_folder,
            output_dir=self.output_dir,
            batch_size="5",
            lookahead=3,
            concurrency=3,
            ollama_url="http://localhost:11434",
            model="qwen3-vl:32b",
            server_backend="ollama",
            debug=False,
        )
        if extra:
            for k, v in extra.items():
                setattr(args, k, v)
        return args

    def test_pre_flight_uses_model_server_manager(self):
        args = self._make_args()

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

        assert exit_code == 0
        call = mock_manager.ensure_models.call_args
        assert call.args[0] == ["qwen3-vl:32b"]  # only the -m vision model
        assert call.kwargs["allow_pull"] is False
        assert call.kwargs["dry_run"] is False
        assert call.kwargs["warm"] is True
        assert call.kwargs["capability"] == "vision"
        assert call.kwargs["min_free_bytes"] == 5 * GiB

    def test_pre_flight_stores_manager_on_args(self):
        args = self._make_args()

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
        args.debug = True

        with patch("cli.commands.ModelServerManager") as MockManager:
            mock_manager = MagicMock()
            mock_manager.state = MagicMock()
            mock_manager.state.base_url = "http://localhost:11435"
            MockManager.return_value = mock_manager

            with patch("cli.commands.ModelSelector") as MockSelector, \
                    patch("pytesseract.get_tesseract_version", return_value="5.0"):
                pre_flight_checks(args)

            # Verify print_stage_report was called
            assert args._model_selector is MockSelector.from_config.return_value
            MockSelector.from_config.return_value.print_stage_report.assert_called_once()

    def test_pre_flight_fallback_on_server_error(self):
        """Pre-flight returns exit code 2 when server fails to start."""
        args = self._make_args()

        with patch("cli.commands.ModelServerManager") as MockManager:
            MockManager.side_effect = Exception("server error")

            with patch("pytesseract.get_tesseract_version", return_value="5.0"):
                exit_code = pre_flight_checks(args)

            assert exit_code == 2


    def _run_pre_flight_capturing_kwargs(self, args):
        with patch("cli.commands.ModelServerManager") as MockManager:
            MockManager.return_value = MagicMock()
            with patch("pytesseract.get_tesseract_version", return_value="5.0"):
                pre_flight_checks(args)
        MockManager.assert_called_once()
        return MockManager.call_args.kwargs

    def test_pre_flight_ollama_backend_string_maps_to_enum_and_base_url(self):
        """Regression: --server-backend is a str; manager must get the enum + base_url."""
        args = self._make_args(
            {"server_backend": "ollama", "ollama_url": "http://127.0.0.1:9999"}
        )
        kwargs = self._run_pre_flight_capturing_kwargs(args)
        assert kwargs["backend"] is ServerBackend.OLLAMA
        assert kwargs["base_url"] == "http://127.0.0.1:9999"

    def test_pre_flight_flags_map_to_ollama_constructor_params(self):
        """Regression: CLI flags must use OllamaServer's parameter names."""
        args = self._make_args(
            {
                "model": "llama3.2-vision",
                "request_timeout": 123.0,
                "server_start_timeout": 45.0,
                "no_server_autostart": True,
                "keep_server": True,
            }
        )
        kwargs = self._run_pre_flight_capturing_kwargs(args)
        assert kwargs["backend"] is ServerBackend.OLLAMA
        assert kwargs["model"] == "llama3.2-vision"
        assert kwargs["timeout"] == 123.0
        assert kwargs["startup_timeout"] == 45.0
        assert kwargs["auto_start"] is False
        assert kwargs["keep_running"] is True
        # Every kwarg except the manager's own must be an OllamaServer parameter.
        accepted = set(inspect.signature(OllamaServer.__init__).parameters) - {"self"}
        assert set(kwargs) - {"backend", "keep_running"} <= accepted

    @pytest.mark.parametrize("backend", ["bogus", "mtplx"])
    def test_pre_flight_rejects_unknown_or_removed_backend(self, backend):
        """A backend that bypassed argparse (e.g. a Namespace) still exits 2."""
        args = self._make_args({"server_backend": backend})
        with patch("cli.commands.ModelServerManager") as MockManager, \
                patch("pytesseract.get_tesseract_version", return_value="5.0"):
            assert pre_flight_checks(args) == 2
        MockManager.assert_not_called()

    def _ready_args(self, **extra):
        args = self._make_args(extra)
        return args

    def test_pre_flight_returns_2_when_server_not_running(self):
        args = self._ready_args(server_backend="ollama")
        with patch("cli.commands.ModelServerManager") as MockManager, \
                patch("pytesseract.get_tesseract_version", return_value="5.0"):
            manager = MockManager.return_value
            manager.ensure_running.return_value = False
            manager.last_error = "Ollama is not reachable"
            assert pre_flight_checks(args) == 2
        manager.ensure_models.assert_not_called()
        manager.close.assert_called_once()
        assert args._server_manager is None

    def test_pre_flight_returns_2_when_models_missing(self):
        args = self._ready_args(server_backend="ollama")
        with patch("cli.commands.ModelServerManager") as MockManager, \
                patch("pytesseract.get_tesseract_version", return_value="5.0"):
            manager = MockManager.return_value
            manager.ensure_running.return_value = True
            manager.ensure_models.return_value = False
            assert pre_flight_checks(args) == 2
        manager.close.assert_called_once()
        assert args._server_manager is None

    def test_pre_flight_real_run_warms_the_vision_model(self):
        args = self._ready_args(server_backend="ollama", model="llama3.2-vision")
        with patch("cli.commands.ModelServerManager") as MockManager, \
                patch("pytesseract.get_tesseract_version", return_value="5.0"):
            manager = MockManager.return_value
            manager.ensure_running.return_value = True
            manager.ensure_models.return_value = True
            assert pre_flight_checks(args) == 0
        names = manager.ensure_models.call_args.args[0]
        kwargs = manager.ensure_models.call_args.kwargs
        assert names == ["llama3.2-vision"]
        assert kwargs["warm"] is True
        assert kwargs["dry_run"] is False

    def test_pre_flight_never_pulls_by_default(self):
        args = self._ready_args(server_backend="ollama")
        with patch("cli.commands.ModelServerManager") as MockManager, \
                patch("pytesseract.get_tesseract_version", return_value="5.0"):
            MockManager.return_value.ensure_running.return_value = True
            MockManager.return_value.ensure_models.return_value = True
            assert pre_flight_checks(args) == 0
        assert MockManager.return_value.ensure_models.call_args.kwargs["allow_pull"] is False

    def test_pre_flight_pull_flag_allows_pull(self):
        args = self._ready_args(server_backend="ollama", pull=True)
        with patch("cli.commands.ModelServerManager") as MockManager, \
                patch("pytesseract.get_tesseract_version", return_value="5.0"):
            MockManager.return_value.ensure_running.return_value = True
            MockManager.return_value.ensure_models.return_value = True
            assert pre_flight_checks(args) == 0
        assert MockManager.return_value.ensure_models.call_args.kwargs["allow_pull"] is True

    def test_pre_flight_dry_run_does_not_start(self):
        args = self._ready_args(server_backend="ollama", dry_run=True)
        with patch("cli.commands.ModelServerManager") as MockManager, \
                patch("pytesseract.get_tesseract_version", return_value="5.0"):
            manager = MockManager.return_value
            manager.probe.return_value.is_running = True
            manager.ensure_models.return_value = True
            assert pre_flight_checks(args) == 0
        manager.probe.assert_called_once()
        manager.ensure_running.assert_not_called()
        manager.ensure_models.assert_called_once()
        kwargs = manager.ensure_models.call_args.kwargs
        assert kwargs["dry_run"] is True
        assert kwargs["warm"] is False
        assert kwargs["allow_pull"] is False

    def test_pre_flight_dry_run_server_down_skips_model_checks(self):
        args = self._ready_args(server_backend="ollama", dry_run=True)
        with patch("cli.commands.ModelServerManager") as MockManager, \
                patch("pytesseract.get_tesseract_version", return_value="5.0"):
            manager = MockManager.return_value
            manager.probe.return_value.is_running = False
            manager.server.can_start.return_value = (True, "")
            assert pre_flight_checks(args) == 0
        manager.ensure_running.assert_not_called()
        manager.ensure_models.assert_not_called()

    def test_pre_flight_dry_run_server_down_and_not_startable_returns_2(self, caplog):
        """A real run could not start the server (e.g. --no-server-autostart): the dry run says so."""
        args = self._ready_args(server_backend="ollama", dry_run=True)
        with patch("cli.commands.ModelServerManager") as MockManager, \
                patch("pytesseract.get_tesseract_version", return_value="5.0"), \
                caplog.at_level(logging.ERROR, logger="cli.commands"):
            manager = MockManager.return_value
            manager.probe.return_value.is_running = False
            manager.server.can_start.return_value = (False, "auto-start is disabled (--no-server-autostart)")
            assert pre_flight_checks(args) == 2
        manager.ensure_running.assert_not_called()
        manager.close.assert_called_once()
        assert "auto-start is disabled" in caplog.text

    def test_dry_run_writes_nothing(self, tmp_path):
        """--dry-run checks writability without creating -o, probe files or the marker."""
        out = tmp_path / "new" / "out"
        args = parse_args(["process", "-f", str(self.image_folder), "-o", str(out), "--dry-run"])
        with patch("cli.commands.ModelServerManager") as MockManager, \
                patch("pytesseract.get_tesseract_version", return_value="5.0"):
            MockManager.return_value.probe.return_value.is_running = True
            MockManager.return_value.ensure_models.return_value = True
            assert pre_flight_checks(args) == 0
        assert not (tmp_path / "new").exists()

    def test_dry_run_unwritable_output_parent_returns_5(self, tmp_path):
        locked = tmp_path / "locked"
        locked.mkdir()
        locked.chmod(0o500)
        try:
            if os.access(locked, os.W_OK):
                pytest.skip("running with privileges that ignore permissions")
            args = parse_args(
                ["process", "-f", str(self.image_folder), "-o", str(locked / "out"), "--dry-run"]
            )
            with patch("cli.commands.ModelServerManager") as MockManager:
                assert pre_flight_checks(args) == 5
            MockManager.assert_not_called()
        finally:
            locked.chmod(0o700)

    def test_real_run_probe_never_touches_user_files(self):
        """The writability probe uses a unique temp name, never a fixed '.write_test'."""
        self.output_dir.mkdir()
        user_file = self.output_dir / ".write_test"
        user_file.write_text("mine")
        args = self._ready_args(server_backend="ollama")
        with patch("cli.commands.ModelServerManager") as MockManager, \
                patch("pytesseract.get_tesseract_version", return_value="5.0"):
            MockManager.return_value.ensure_running.return_value = True
            MockManager.return_value.ensure_models.return_value = True
            assert pre_flight_checks(args) == 0
        assert user_file.read_text() == "mine"
        assert (self.output_dir / ".bourbon_output").exists()
        assert sorted(p.name for p in self.output_dir.iterdir()) == [".bourbon_output", ".write_test"]

    def test_count_matches_scan_and_skips_earlier_outputs(self, tmp_path, caplog):
        """Pre-flight counts exactly what the scan processes (not rglob)."""
        folder = tmp_path / "shoot"
        for name in ("IMG_1.jpg", "IMG_2.webp", ".hidden.jpg"):
            (folder / name).parent.mkdir(parents=True, exist_ok=True)
            (folder / name).write_bytes(b"x")
        (folder / "old").mkdir()
        (folder / "old" / ".bourbon_output").touch()
        (folder / "old" / "a.jpg").write_bytes(b"x")
        (folder / "processed").mkdir()
        (folder / "processed" / "b.jpg").write_bytes(b"x")
        args = parse_args(["process", "-f", str(folder), "-o", str(folder / "processed"), "--dry-run"])
        with patch("cli.commands.ModelServerManager") as MockManager, \
                patch("pytesseract.get_tesseract_version", return_value="5.0"), \
                caplog.at_level(logging.INFO, logger="cli.commands"):
            MockManager.return_value.probe.return_value.is_running = True
            MockManager.return_value.ensure_models.return_value = True
            assert pre_flight_checks(args) == 0
        assert f"Found 2 images in {folder}" in caplog.text

    def test_only_earlier_outputs_exits_3(self, tmp_path):
        folder = tmp_path / "shoot"
        (folder / "old").mkdir(parents=True)
        (folder / "old" / ".bourbon_output").touch()
        (folder / "old" / "a.jpg").write_bytes(b"x")
        args = parse_args(["process", "-f", str(folder), "-o", str(tmp_path / "out"), "--dry-run"])
        with patch("cli.commands.ModelServerManager") as MockManager:
            assert pre_flight_checks(args) == 3
        MockManager.assert_not_called()

    def test_webp_only_folder_passes(self, tmp_path):
        folder = tmp_path / "shoot"
        folder.mkdir()
        (folder / "IMG_1.webp").write_bytes(b"x")
        args = parse_args(["process", "-f", str(folder), "-o", str(tmp_path / "out"), "--dry-run"])
        with patch("cli.commands.ModelServerManager") as MockManager, \
                patch("pytesseract.get_tesseract_version", return_value="5.0"):
            MockManager.return_value.probe.return_value.is_running = True
            MockManager.return_value.ensure_models.return_value = True
            assert pre_flight_checks(args) == 0

    def test_output_dir_containing_photos_is_not_marked(self, tmp_path, caplog):
        """-o an ancestor of -f: warn, and never hide the photo library behind the marker."""
        library = tmp_path / "library"
        shoot = library / "shoot"
        shoot.mkdir(parents=True)
        (shoot / "IMG_1.jpg").write_bytes(b"x")
        args = parse_args(["process", "-f", str(shoot), "-o", str(library)])
        with patch("cli.commands.ModelServerManager") as MockManager, \
                patch("pytesseract.get_tesseract_version", return_value="5.0"), \
                caplog.at_level(logging.WARNING, logger="cli.commands"):
            MockManager.return_value.ensure_running.return_value = True
            MockManager.return_value.ensure_models.return_value = True
            assert pre_flight_checks(args) == 0
        assert not (library / ".bourbon_output").exists()
        assert args._mark_output_dir is False
        assert "contains --image-folder" in caplog.text

    @pytest.mark.parametrize(
        ("flag", "value"),
        [("--group-gap-seconds", "0"), ("--group-gap-seconds", "-1"), ("--group-gap-seconds", "nan"),
         ("--group-gap-seconds", "601"), ("--group-max-size", "-1"), ("--group-max-size", "21"),
         ("--split-budget-ratio", "-0.1"), ("--split-budget-ratio", "2.5"), ("--split-budget-ratio", "nan"),
         ("--split-budget-ratio", "inf")],
    )
    def test_invalid_grouping_options_return_1(self, flag, value):
        args = parse_args(["process", "-f", str(self.image_folder), "-o", str(self.output_dir), flag, value])
        with patch("cli.commands.ModelServerManager") as MockManager:
            assert pre_flight_checks(args) == 1
        MockManager.assert_not_called()

    def test_grouping_option_defaults(self):
        args = parse_args(["process"])
        assert args.group_gap_seconds == 12.0
        assert args.group_max_size == 3
        assert args.split_budget_ratio == 0.35
        assert args.lookahead is None  # deprecated: absent unless given

    @pytest.mark.parametrize("value", ["0", "0.35", "2"])
    def test_valid_split_budget_ratio_passes_validation(self, value, caplog):
        args = parse_args(
            ["process", "-f", str(self.image_folder), "-o", str(self.output_dir), "--split-budget-ratio", value]
        )
        with patch("cli.commands.ModelServerManager"), caplog.at_level(logging.ERROR, logger="cli.commands"):
            pre_flight_checks(args)
        assert "--split-budget-ratio" not in caplog.text

    def test_lookahead_is_deprecated_but_accepted(self, caplog):
        args = parse_args(["process", "-f", str(self.image_folder), "-o", str(self.output_dir), "--lookahead", "5"])
        with patch("cli.commands.ModelServerManager"), caplog.at_level(logging.WARNING, logger="cli.commands"):
            pre_flight_checks(args)
        assert "--lookahead is deprecated and ignored" in caplog.text

    def test_lookahead_not_given_no_warning(self, caplog):
        args = parse_args(["process", "-f", str(self.image_folder), "-o", str(self.output_dir)])
        with patch("cli.commands.ModelServerManager"), caplog.at_level(logging.WARNING, logger="cli.commands"):
            pre_flight_checks(args)
        assert "--lookahead" not in caplog.text

    def test_pre_flight_dry_run_missing_model_returns_2(self):
        args = self._ready_args(server_backend="ollama", dry_run=True)
        with patch("cli.commands.ModelServerManager") as MockManager, \
                patch("pytesseract.get_tesseract_version", return_value="5.0"):
            manager = MockManager.return_value
            manager.probe.return_value.is_running = True
            manager.ensure_models.return_value = False
            assert pre_flight_checks(args) == 2
        manager.ensure_running.assert_not_called()

    def test_pre_flight_tesseract_missing_constructs_no_manager(self):
        args = self._ready_args()
        with patch("cli.commands.ModelServerManager") as MockManager, \
                patch("pytesseract.get_tesseract_version", side_effect=OSError("no tesseract")):
            assert pre_flight_checks(args) == 1
        MockManager.assert_not_called()

    def test_invalid_ollama_port_returns_2(self):
        args = self._ready_args(ollama_url="http://localhost:99999")
        with patch("cli.commands.ModelServerManager") as MockManager:
            assert pre_flight_checks(args) == 2
        MockManager.assert_not_called()

    def test_pre_flight_requires_only_default_vision_model(self, monkeypatch):
        monkeypatch.delenv("BOURBON_MODEL", raising=False)
        args = self._ready_args(server_backend="ollama", model=None)
        with patch("cli.commands.ModelServerManager") as MockManager, \
                patch("pytesseract.get_tesseract_version", return_value="5.0"):
            manager = MockManager.return_value
            manager.ensure_running.return_value = True
            manager.ensure_models.return_value = True
            assert pre_flight_checks(args) == 0
        call = manager.ensure_models.call_args
        assert call.args[0] == ["qwen3.6:35b-a3b"]  # no gpt-oss:120b, no fallbacks
        assert call.kwargs["warm"] is True
        assert call.kwargs["capability"] == "vision"

    def test_pre_flight_dry_run_passes_capability_and_disk_floor(self):
        args = self._ready_args(server_backend="ollama", dry_run=True, pull=True, min_free_gb=12.5)
        with patch("cli.commands.ModelServerManager") as MockManager, \
                patch("pytesseract.get_tesseract_version", return_value="5.0"):
            manager = MockManager.return_value
            manager.probe.return_value.is_running = True
            manager.ensure_models.return_value = True
            assert pre_flight_checks(args) == 0
        manager.ensure_running.assert_not_called()
        kwargs = manager.ensure_models.call_args.kwargs
        assert kwargs["dry_run"] is True
        assert kwargs["allow_pull"] is True
        assert kwargs["capability"] == "vision"
        assert kwargs["min_free_bytes"] == int(12.5 * GiB)

    def test_pre_flight_min_free_gb_passed_on_real_run(self):
        args = self._ready_args(server_backend="ollama", min_free_gb=0.0)
        with patch("cli.commands.ModelServerManager") as MockManager, \
                patch("pytesseract.get_tesseract_version", return_value="5.0"):
            MockManager.return_value.ensure_running.return_value = True
            MockManager.return_value.ensure_models.return_value = True
            assert pre_flight_checks(args) == 0
        assert MockManager.return_value.ensure_models.call_args.kwargs["min_free_bytes"] == 0

    @pytest.mark.parametrize("value", [-1.0, float("nan"), float("inf")])
    def test_pre_flight_invalid_min_free_gb_returns_1(self, value):
        args = self._ready_args(server_backend="ollama", min_free_gb=value)
        with patch("cli.commands.ModelServerManager") as MockManager:
            assert pre_flight_checks(args) == 1
        MockManager.assert_not_called()

    def test_pre_flight_resolves_model_from_env(self, monkeypatch):
        monkeypatch.setenv("BOURBON_MODEL", "env-vision:1b")
        args = self._ready_args(server_backend="ollama", model=None)
        with patch("cli.commands.ModelServerManager") as MockManager, \
                patch("pytesseract.get_tesseract_version", return_value="5.0"):
            MockManager.return_value.ensure_running.return_value = True
            MockManager.return_value.ensure_models.return_value = True
            assert pre_flight_checks(args) == 0
        assert args.model == "env-vision:1b"
        assert MockManager.return_value.ensure_models.call_args.args[0] == ["env-vision:1b"]

# ---------------------------------------------------------------------------
# Process command with usage summary
# ---------------------------------------------------------------------------


class TestProcessCommandIntegration:
    """Test that process_command prints usage summary and cleans up."""

    @pytest.fixture(autouse=True)
    def _folders(self, tmp_path):
        """A real photo folder with one image and a not-yet-created output dir."""
        self.image_folder = tmp_path / "photos"
        self.image_folder.mkdir()
        (self.image_folder / "IMG_1.jpg").write_bytes(b"\xff\xd8\xff\xe0\x00\x10JFIF")
        self.output_dir = tmp_path / "out"

    def test_process_command_uses_model_selector(self):
        """Process command should use ModelSelector from pre-flight."""
        args = argparse.Namespace(
            command="process",
            image_folder=self.image_folder,
            batch_size="all",
            lookahead=3,
            concurrency=3,
            ollama_url="http://localhost:11434",
            model="qwen3-vl:32b",
            server_backend="ollama",
            output_dir=self.output_dir,
            interactive=False,
            include_vendor_images=False,
            allow_retry=True,
            debug=False,
            dry_run=True,  # Skip actual processing
        )

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
            image_folder=self.image_folder,
            batch_size="all",
            lookahead=3,
            concurrency=3,
            ollama_url="http://localhost:11434",
            model="qwen3-vl:32b",
            server_backend="ollama",
            output_dir=self.output_dir,
            interactive=False,
            include_vendor_images=False,
            allow_retry=True,
            debug=False,
            dry_run=True,  # Skip actual processing
        )
        caplog.set_level(logging.INFO, logger="cli.commands")

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
            image_folder=self.image_folder,
            batch_size="all",
            lookahead=3,
            concurrency=3,
            ollama_url="http://localhost:11434",
            model="qwen3-vl:32b",
            server_backend="ollama",
            output_dir=self.output_dir,
            interactive=False,
            include_vendor_images=False,
            allow_retry=True,
            debug=False,
            dry_run=True,  # Skip actual processing
        )

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

    def _dry_run_args(self):
        args = argparse.Namespace(
            command="process",
            image_folder=self.image_folder,
            batch_size="all",
            lookahead=3,
            concurrency=3,
            ollama_url="http://localhost:11434",
            model="qwen3-vl:32b",
            server_backend="ollama",
            output_dir=self.output_dir,
            interactive=False,
            include_vendor_images=False,
            allow_retry=True,
            debug=False,
            dry_run=True,
        )
        return args

    def test_dry_run_does_not_call_get_client(self):
        args = self._dry_run_args()
        mock_manager = MagicMock()
        args._server_manager = mock_manager
        args._model_selector = MagicMock()
        with patch("cli.commands.pre_flight_checks", return_value=0):
            assert process_command(args) == 0
        mock_manager.get_client.assert_not_called()
        mock_manager.ensure_running.assert_not_called()
        mock_manager.close.assert_called_once()

    def test_manager_closed_when_pre_flight_fails_late(self):
        """A manager attached by pre-flight is closed even if processing never starts."""
        args = self._dry_run_args()
        mock_manager = MagicMock()

        def failing_pre_flight(a):
            a._server_manager = mock_manager
            return 2

        with patch("cli.commands.pre_flight_checks", side_effect=failing_pre_flight):
            assert process_command(args) == 2
        mock_manager.close.assert_called_once()

    def test_real_run_get_client_failure_returns_2(self):
        from lib.exceptions import ModelServerError

        args = self._dry_run_args()
        args.dry_run = False
        mock_manager = MagicMock()
        mock_manager.get_client.side_effect = ModelServerError("ollama model server is not running")
        args._server_manager = mock_manager
        args._model_selector = MagicMock()
        with patch("cli.commands.pre_flight_checks", return_value=0):
            assert process_command(args) == 2
        mock_manager.close.assert_called_once()


# ---------------------------------------------------------------------------
# Retry command server lifecycle
# ---------------------------------------------------------------------------


class TestRetryCommandServer:
    """retry builds the manager from CLI flags, checks it, and always closes it."""

    def _write_report(self, tmp_path):
        image = tmp_path / "IMG_0001.jpg"
        image.write_bytes(b"\xff\xd8\xff\xe0\x00\x10JFIF")
        report = tmp_path / "error_report.json"
        report.write_text(
            json.dumps({"bottle_id": "b1", "images": [str(image)], "error_message": "boom"})
        )
        return report

    def test_retry_returns_1_when_server_unavailable(self, tmp_path):
        args = parse_args(
            ["retry", "--report", str(self._write_report(tmp_path)), "-o", str(tmp_path / "out")]
        )
        with patch("cli.commands.ModelServerManager") as MockManager:
            manager = MockManager.return_value
            manager.ensure_running.return_value = False
            manager.last_error = "Ollama is not reachable"
            assert retry_command(args) == 1
        assert MockManager.call_args.kwargs["backend"] is ServerBackend.OLLAMA
        manager.get_client.assert_not_called()
        manager.close.assert_called_once()

    def test_retry_uses_resolved_model_and_closes(self, tmp_path):
        args = parse_args(
            [
                "retry", "--report", str(self._write_report(tmp_path)),
                "-o", str(tmp_path / "out"), "-m", "llama3.2-vision",
            ]
        )
        with patch("cli.commands.ModelServerManager") as MockManager, \
                patch("cli.commands.BottleAnalyzer") as MockAnalyzer:
            manager = MockManager.return_value
            manager.ensure_running.return_value = True
            MockAnalyzer.return_value.process_single_bottle.side_effect = RuntimeError("stop here")
            assert retry_command(args) == 1
        assert MockManager.call_args.kwargs["backend"] is ServerBackend.OLLAMA
        call = MockAnalyzer.return_value.process_single_bottle.call_args
        assert call.kwargs["vision_model"] == "llama3.2-vision"
        manager.close.assert_called_once()


# ---------------------------------------------------------------------------
# Vision failures surface as failures (B6)
# ---------------------------------------------------------------------------


class TestVisionFailureExitCodes:
    """Fatal model errors abort the run (exit 2); per-bottle failures are reported."""

    @pytest.fixture(autouse=True)
    def _folders(self, tmp_path):
        """A real photo folder with one image and a not-yet-created output dir."""
        self.image_folder = tmp_path / "photos"
        self.image_folder.mkdir()
        (self.image_folder / "IMG_1.jpg").write_bytes(b"\xff\xd8\xff\xe0\x00\x10JFIF")
        self.output_dir = tmp_path / "out"

    def _real_run_args(self):
        args = argparse.Namespace(
            command="process",
            image_folder=self.image_folder,
            batch_size="all",
            lookahead=3,
            concurrency=3,
            ollama_url="http://localhost:11434",
            model="qwen3.6:35b-a3b",
            server_backend="ollama",
            output_dir=self.output_dir,
            interactive=False,
            include_vendor_images=False,
            allow_retry=True,
            debug=False,
            dry_run=False,
        )
        args._server_manager = MagicMock()
        args._model_selector = MagicMock()
        return args

    def _run(self, args, process_batch):
        from models.image import Image

        group = [Image(path="/photos/IMG_1.jpg")]
        with patch("cli.commands.pre_flight_checks", return_value=0), \
                patch("cli.commands.ImageOrganizer") as MockOrganizer, \
                patch("cli.commands.BatchProcessor") as MockBatch, \
                patch("cli.commands.PreviewGenerator"):
            MockOrganizer.return_value.scan_images.return_value = group
            MockOrganizer.return_value.group_by_bottle.return_value = [group]
            MockBatch.return_value.slice_into_batches.return_value = [[group]]
            if isinstance(process_batch, BaseException):
                MockBatch.return_value.process_batch.side_effect = process_batch
            else:
                MockBatch.return_value.process_batch.return_value = process_batch
            return process_command(args)

    @pytest.mark.parametrize(
        "error_name", ["VisionModelUnavailableError", "OllamaConnectionError", "ModelServerError"]
    )
    def test_fatal_model_error_aborts_with_exit_2(self, error_name, caplog):
        import lib.exceptions as exceptions

        args = self._real_run_args()
        error = getattr(exceptions, error_name)("model 'x' is not usable")
        with caplog.at_level(logging.ERROR, logger="cli.commands"):
            assert self._run(args, error) == 2
        assert any("Aborting: model 'x' is not usable" in r.getMessage() for r in caplog.records)
        args._server_manager.close.assert_called_once()

    def test_failed_bottle_reason_logged_and_all_failed_exit_4(self, caplog):
        from image_processor.batch_processor import BatchResult

        result = BatchResult(batch_id=1)
        result.failed_bottles.append("IMG_1")
        result.errors["IMG_1"] = "VisionResponseError: returned no content"
        args = self._real_run_args()
        with caplog.at_level(logging.WARNING, logger="cli.commands"):
            assert self._run(args, result) == 4
        assert any(
            "IMG_1 (VisionResponseError: returned no content)" in r.getMessage() for r in caplog.records
        )

    def test_retry_vision_failure_returns_1(self, tmp_path, caplog):
        from lib.exceptions import VisionTimeoutError

        image = tmp_path / "IMG_0001.jpg"
        image.write_bytes(b"\xff\xd8\xff\xe0\x00\x10JFIF")
        report = tmp_path / "error_report.json"
        report.write_text(json.dumps({"bottle_id": "b1", "images": [str(image)], "error_message": "boom"}))
        args = parse_args(["retry", "--report", str(report), "-o", str(tmp_path / "out")])
        with patch("cli.commands.ModelServerManager") as MockManager, \
                patch("cli.commands.BottleAnalyzer") as MockAnalyzer, \
                caplog.at_level(logging.ERROR, logger="cli.commands"):
            manager = MockManager.return_value
            manager.ensure_running.return_value = True
            MockAnalyzer.return_value.process_single_bottle.side_effect = VisionTimeoutError("too slow")
            assert retry_command(args) == 1
        assert any("Failed to reprocess bottle b1: too slow" in r.getMessage() for r in caplog.records)
        manager.close.assert_called_once()


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