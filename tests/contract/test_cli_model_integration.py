"""Contract tests for CLI model integration.

Tests cover:
  - --model flag sets primary model for all AI stages
  - BOURBON_<STAGE>_MODEL env vars override per-stage
  - removed flags and aliases exit 2; a stale backend env var is ignored
  - Pre-flight uses ModelServerManager (server checked last, failures exit 2,
    dry runs only probe, nothing is pulled by default)
  - Debug mode prints stage report
  - Usage summary printed at end of processing, manager always closed
"""

import dataclasses
import inspect
import json
import logging
import os
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from cli.commands import (
    RunContext,
    main,
    parse_args,
    pre_flight_checks,
    process_command,
    retry_command,
)
from lib.constants import DEFAULT_VISION_MODEL
from lib.model_availability import GiB
from lib.model_server import OllamaServer
from lib.settings import SettingsError, resolve_settings
from tests.helpers import make_settings

logger = logging.getLogger(__name__)


def _settings(tmp_path, *, text=False, **overrides):
    """``process`` settings: photos in tmp_path/photos, output tmp_path/out, fact check off unless ``text``."""
    settings = make_settings(image_folder=tmp_path / "photos", output_dir=tmp_path / "out", **overrides)
    if not text:
        settings = dataclasses.replace(settings, text=dataclasses.replace(settings.text, enabled=False))
    return settings


def _ctx(tmp_path, *, text=False, **overrides) -> RunContext:
    return RunContext(_settings(tmp_path, text=text, **overrides))


def _ctx_from_argv(argv, *, text=False) -> RunContext:
    """A RunContext from real CLI arguments (fact check off unless ``text``)."""
    if not text:
        argv = [*argv, "--no-verify"]
    return RunContext(resolve_settings(parse_args(argv)))


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


class TestParseArgs:
    """Test CLI argument parsing for model selector features."""

    def test_default_model_flag(self):
        args = parse_args(["process", "--image-folder", "/tmp/test"])
        assert args.model is None  # resolved by lib.settings

    def test_custom_model_flag(self):
        args = parse_args(["process", "--image-folder", "/tmp/test", "--model", "llama3.2"])
        assert args.model == "llama3.2"

    @pytest.mark.parametrize(
        "argv",
        [
            ["--server-backend", "ollama"],
            ["--lookahead", "3"],
            ["--vision-timeout", "45"],
            ["--pull-missing"],
        ],
    )
    def test_removed_flags_exit_2(self, argv):
        with pytest.raises(SystemExit) as excinfo:
            parse_args(["process", *argv])
        assert excinfo.value.code == 2

    def test_retry_rejects_server_backend(self):
        with pytest.raises(SystemExit) as excinfo:
            parse_args(["retry", "--report", "r.json", "--server-backend", "ollama"])
        assert excinfo.value.code == 2

    def test_server_backend_env_var_is_ignored(self, monkeypatch):
        monkeypatch.setenv("BOURBON_SERVER_BACKEND", "mtplx")
        args = parse_args(["process", "--image-folder", "/tmp/test"])
        assert not hasattr(args, "server_backend")
        settings = resolve_settings(args)
        assert "server_backend" not in json.dumps(dict(settings.sources))

    def test_pull_flag(self):
        assert resolve_settings(parse_args(["process"])).model.pull is False
        assert resolve_settings(parse_args(["process", "--pull"])).model.pull is True

    def test_min_free_gb_flag(self):
        assert resolve_settings(parse_args(["process"])).model.min_free_gb == 5.0
        assert resolve_settings(parse_args(["process", "--pull", "--min-free-gb", "10"])).model.min_free_gb == 10.0
        assert resolve_settings(parse_args(["process", "--min-free-gb", "0.5"])).model.min_free_gb == 0.5

    def test_vision_model_precedence(self, monkeypatch):
        assert resolve_settings(parse_args(["process"])).model.vision_model == DEFAULT_VISION_MODEL == "qwen3.6:35b-a3b"
        monkeypatch.setenv("BOURBON_MODEL", "gemma4:26b")
        assert resolve_settings(parse_args(["process"])).model.vision_model == "gemma4:26b"
        assert resolve_settings(parse_args(["process", "-m", "qwen3.8:27b"])).model.vision_model == "qwen3.8:27b"

    def test_request_timeout(self):
        assert resolve_settings(parse_args(["process"])).server.request_timeout == 600.0
        assert resolve_settings(parse_args(["process", "--request-timeout", "90"])).server.request_timeout == 90.0

    def test_server_lifecycle_flags(self):
        server = resolve_settings(
            parse_args(["process", "--keep-server", "--no-server-autostart", "--server-start-timeout", "12"])
        ).server
        assert server.keep_running is True
        assert server.autostart is False
        assert server.start_timeout == 12.0

    def test_retry_server_flags(self):
        settings = resolve_settings(
            parse_args(
                [
                    "retry", "--report", "r.json",
                    "--keep-server", "-m", "llama3.2", "-u", "http://127.0.0.1:1234",
                ]
            )
        )
        assert settings.server.keep_running is True
        assert settings.model.vision_model == "llama3.2"
        assert settings.server.ollama_url == "http://127.0.0.1:1234"

    def test_retry_model_default_is_none(self):
        assert parse_args(["retry", "--report", "r.json"]).model is None

    def test_debug_flag(self):
        assert resolve_settings(parse_args(["process", "--image-folder", "/tmp/test", "--debug"])).debug is True

    def test_debug_flag_before_subcommand(self):
        assert resolve_settings(parse_args(["--debug", "process", "--image-folder", "/tmp/test"])).debug is True

    def test_debug_default_false(self):
        assert resolve_settings(parse_args(["process", "--image-folder", "/tmp/test"])).debug is False
        assert resolve_settings(parse_args(["retry", "--report", "r.json"])).debug is False

    def test_help_exits_0(self, capsys):
        with pytest.raises(SystemExit) as excinfo:
            parse_args(["process", "--help"])
        assert excinfo.value.code == 0
        assert "--request-timeout" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Pre-flight checks with ModelServerManager
# ---------------------------------------------------------------------------


class TestPreFlightModelIntegration:
    """Test that pre-flight uses ModelServerManager."""

    @pytest.fixture(autouse=True)
    def _folders(self, tmp_path):
        """A real photo folder with one image and a not-yet-created output dir."""
        self.tmp_path = tmp_path
        self.image_folder = tmp_path / "photos"
        self.image_folder.mkdir()
        (self.image_folder / "IMG_1.jpg").write_bytes(b"\xff\xd8\xff\xe0\x00\x10JFIF")
        self.output_dir = tmp_path / "out"

    def _make_args(self, extra=None, **kwargs):
        """Pre-flight context with ``-m qwen3-vl:32b`` (fact check off unless ``text=True``)."""
        extra = dict(extra or {})
        model = {"vision_model": extra.pop("model", "qwen3-vl:32b") or DEFAULT_VISION_MODEL}
        for key in ("pull", "min_free_gb"):
            if key in extra:
                model[key] = extra.pop(key)
        server = {}
        for old, new in (("ollama_url", "ollama_url"), ("request_timeout", "request_timeout"),
                         ("server_start_timeout", "start_timeout"), ("keep_server", "keep_running")):
            if old in extra:
                server[new] = extra.pop(old)
        if extra.pop("no_server_autostart", False):
            server["autostart"] = False
        return _ctx(self.tmp_path, model=model, server=server, **extra, **kwargs)

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
        assert call.args[0] == [("qwen3-vl:32b", "vision")]  # only the -m vision model
        assert call.kwargs["allow_pull"] is False
        assert call.kwargs["dry_run"] is False
        assert call.kwargs["warm"] is True
        assert call.kwargs["warm_models"] == ["qwen3-vl:32b"]
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

            # Verify the context got the manager and selector attached
            assert args.server_manager is mock_manager
            assert args.model_selector is not None

    def test_pre_flight_prints_stage_report_in_debug(self):
        args = self._make_args(debug=True)

        with patch("cli.commands.ModelServerManager") as MockManager:
            mock_manager = MagicMock()
            mock_manager.state = MagicMock()
            mock_manager.state.base_url = "http://localhost:11435"
            MockManager.return_value = mock_manager

            with patch("cli.commands.ModelSelector") as MockSelector, \
                    patch("pytesseract.get_tesseract_version", return_value="5.0"):
                pre_flight_checks(args)

            # Verify print_stage_report was called
            assert args.model_selector is MockSelector.from_config.return_value
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

    def test_pre_flight_passes_ollama_url_as_base_url(self):
        args = self._make_args({"ollama_url": "http://127.0.0.1:9999"})
        kwargs = self._run_pre_flight_capturing_kwargs(args)
        assert kwargs["base_url"] == "http://127.0.0.1:9999"
        assert "backend" not in kwargs

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
        assert kwargs["model"] == "llama3.2-vision"
        assert kwargs["timeout"] == 123.0
        assert kwargs["startup_timeout"] == 45.0
        assert kwargs["auto_start"] is False
        assert kwargs["keep_running"] is True
        # Every kwarg except the manager's own must be an OllamaServer parameter.
        accepted = set(inspect.signature(OllamaServer.__init__).parameters) - {"self"}
        assert set(kwargs) - {"keep_running"} <= accepted

    def _ready_args(self, **extra):
        dry_run = extra.pop("dry_run", False)
        return self._make_args(extra, dry_run=dry_run)

    def test_pre_flight_returns_2_when_server_not_running(self):
        args = self._ready_args()
        with patch("cli.commands.ModelServerManager") as MockManager, \
                patch("pytesseract.get_tesseract_version", return_value="5.0"):
            manager = MockManager.return_value
            manager.ensure_running.return_value = False
            manager.last_error = "Ollama is not reachable"
            assert pre_flight_checks(args) == 2
        manager.ensure_models.assert_not_called()
        manager.close.assert_called_once()
        assert args.server_manager is None

    def test_pre_flight_returns_2_when_models_missing(self):
        args = self._ready_args()
        with patch("cli.commands.ModelServerManager") as MockManager, \
                patch("pytesseract.get_tesseract_version", return_value="5.0"):
            manager = MockManager.return_value
            manager.ensure_running.return_value = True
            manager.ensure_models.return_value = False
            assert pre_flight_checks(args) == 2
        manager.close.assert_called_once()
        assert args.server_manager is None

    def test_pre_flight_real_run_warms_the_vision_model(self):
        args = self._ready_args(model="llama3.2-vision")
        with patch("cli.commands.ModelServerManager") as MockManager, \
                patch("pytesseract.get_tesseract_version", return_value="5.0"):
            manager = MockManager.return_value
            manager.ensure_running.return_value = True
            manager.ensure_models.return_value = True
            assert pre_flight_checks(args) == 0
        names = manager.ensure_models.call_args.args[0]
        kwargs = manager.ensure_models.call_args.kwargs
        assert names == [("llama3.2-vision", "vision")]
        assert kwargs["warm"] is True
        assert kwargs["warm_models"] == ["llama3.2-vision"]
        assert kwargs["dry_run"] is False

    def test_pre_flight_never_pulls_by_default(self):
        args = self._ready_args()
        with patch("cli.commands.ModelServerManager") as MockManager, \
                patch("pytesseract.get_tesseract_version", return_value="5.0"):
            MockManager.return_value.ensure_running.return_value = True
            MockManager.return_value.ensure_models.return_value = True
            assert pre_flight_checks(args) == 0
        assert MockManager.return_value.ensure_models.call_args.kwargs["allow_pull"] is False

    def test_pre_flight_pull_flag_allows_pull(self):
        args = self._ready_args(pull=True)
        with patch("cli.commands.ModelServerManager") as MockManager, \
                patch("pytesseract.get_tesseract_version", return_value="5.0"):
            MockManager.return_value.ensure_running.return_value = True
            MockManager.return_value.ensure_models.return_value = True
            assert pre_flight_checks(args) == 0
        assert MockManager.return_value.ensure_models.call_args.kwargs["allow_pull"] is True

    def test_pre_flight_dry_run_does_not_start(self):
        args = self._ready_args(dry_run=True)
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
        args = self._ready_args(dry_run=True)
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
        args = self._ready_args(dry_run=True)
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
        args = _ctx_from_argv(["process", "-f", str(self.image_folder), "-o", str(out), "--dry-run"])
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
            args = _ctx_from_argv(
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
        args = self._ready_args()
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
        args = _ctx_from_argv(["process", "-f", str(folder), "-o", str(folder / "processed"), "--dry-run"])
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
        args = _ctx_from_argv(["process", "-f", str(folder), "-o", str(tmp_path / "out"), "--dry-run"])
        with patch("cli.commands.ModelServerManager") as MockManager:
            assert pre_flight_checks(args) == 3
        MockManager.assert_not_called()

    def test_webp_only_folder_passes(self, tmp_path):
        folder = tmp_path / "shoot"
        folder.mkdir()
        (folder / "IMG_1.webp").write_bytes(b"x")
        args = _ctx_from_argv(["process", "-f", str(folder), "-o", str(tmp_path / "out"), "--dry-run"])
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
        args = _ctx_from_argv(["process", "-f", str(shoot), "-o", str(library)])
        with patch("cli.commands.ModelServerManager") as MockManager, \
                patch("pytesseract.get_tesseract_version", return_value="5.0"), \
                caplog.at_level(logging.WARNING, logger="cli.commands"):
            MockManager.return_value.ensure_running.return_value = True
            MockManager.return_value.ensure_models.return_value = True
            assert pre_flight_checks(args) == 0
        assert not (library / ".bourbon_output").exists()
        assert args.mark_output_dir is False
        assert "contains --image-folder" in caplog.text

    @pytest.mark.parametrize(
        ("flag", "value"),
        [("--group-gap-seconds", "0"), ("--group-gap-seconds", "-1"), ("--group-gap-seconds", "nan"),
         ("--group-gap-seconds", "601"), ("--group-max-size", "-1"), ("--group-max-size", "21"),
         ("--split-budget-ratio", "-0.1"), ("--split-budget-ratio", "2.5"), ("--split-budget-ratio", "nan"),
         ("--split-budget-ratio", "inf")],
    )
    def test_invalid_grouping_options_are_settings_errors(self, flag, value):
        with patch("cli.commands.ModelServerManager") as MockManager:
            assert main(["process", "-f", str(self.image_folder), "-o", str(self.output_dir), flag, value]) == 1
        MockManager.assert_not_called()

    def test_grouping_option_defaults(self):
        grouping = resolve_settings(parse_args(["process"])).grouping
        assert grouping.gap_seconds == 12.0
        assert grouping.max_group_size == 3
        assert grouping.split_budget_ratio == 0.35

    @pytest.mark.parametrize("value", ["0", "0.35", "2"])
    def test_valid_split_budget_ratio_passes_validation(self, value):
        settings = resolve_settings(
            parse_args(["process", "-f", str(self.image_folder), "-o", str(self.output_dir), "--split-budget-ratio", value])
        )
        assert settings.grouping.split_budget_ratio == float(value)

    def test_pre_flight_dry_run_missing_model_returns_2(self):
        args = self._ready_args(dry_run=True)
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

    def test_invalid_ollama_port_is_a_settings_error_exit_1(self, capsys):
        with patch("cli.commands.ModelServerManager") as MockManager:
            assert main(["process", "-f", str(self.image_folder), "-u", "http://localhost:99999"]) == 1
        MockManager.assert_not_called()
        assert "ollama_url='http://localhost:99999'" in capsys.readouterr().err

    def test_pre_flight_requires_only_default_vision_model_without_fact_check(self):
        args = self._ready_args(model=None)
        with patch("cli.commands.ModelServerManager") as MockManager, \
                patch("pytesseract.get_tesseract_version", return_value="5.0"):
            manager = MockManager.return_value
            manager.ensure_running.return_value = True
            manager.ensure_models.return_value = True
            assert pre_flight_checks(args) == 0
        call = manager.ensure_models.call_args
        assert call.args[0] == [("qwen3.6:35b-a3b", "vision")]  # no gpt-oss:120b, no fallbacks
        assert call.kwargs["warm"] is True
        assert call.kwargs["warm_models"] == ["qwen3.6:35b-a3b"]

    def test_pre_flight_dry_run_passes_capability_and_disk_floor(self):
        args = self._ready_args(dry_run=True, pull=True, min_free_gb=12.5)
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
        assert manager.ensure_models.call_args.args[0] == [("qwen3-vl:32b", "vision")]
        assert kwargs["min_free_bytes"] == int(12.5 * GiB)

    def test_pre_flight_min_free_gb_passed_on_real_run(self):
        args = self._ready_args(min_free_gb=0.0)
        with patch("cli.commands.ModelServerManager") as MockManager, \
                patch("pytesseract.get_tesseract_version", return_value="5.0"):
            MockManager.return_value.ensure_running.return_value = True
            MockManager.return_value.ensure_models.return_value = True
            assert pre_flight_checks(args) == 0
        assert MockManager.return_value.ensure_models.call_args.kwargs["min_free_bytes"] == 0

    @pytest.mark.parametrize("value", ["-1", "nan", "inf"])
    def test_invalid_min_free_gb_is_a_settings_error(self, value):
        with pytest.raises(SettingsError, match="min_free_gb"):
            resolve_settings(parse_args(["process", "--min-free-gb", value]))

    def test_pre_flight_resolves_model_from_env(self, monkeypatch):
        monkeypatch.setenv("BOURBON_MODEL", "env-vision:1b")
        args = _ctx_from_argv(["process", "-f", str(self.image_folder), "-o", str(self.output_dir)])
        with patch("cli.commands.ModelServerManager") as MockManager, \
                patch("pytesseract.get_tesseract_version", return_value="5.0"):
            MockManager.return_value.ensure_running.return_value = True
            MockManager.return_value.ensure_models.return_value = True
            assert pre_flight_checks(args) == 0
        assert args.settings.model.vision_model == "env-vision:1b"
        assert MockManager.return_value.ensure_models.call_args.args[0] == [("env-vision:1b", "vision")]

    def test_fact_check_requires_text_model_warms_only_vision_and_checks_ram(self):
        args = self._make_args(text=True)
        with patch("cli.commands.ModelServerManager") as MockManager, \
                patch("cli.commands._text_model_memory_check", return_value=0) as ram, \
                patch("pytesseract.get_tesseract_version", return_value="5.0"):
            manager = MockManager.return_value
            manager.ensure_running.return_value = True
            manager.ensure_models.return_value = True
            assert pre_flight_checks(args) == 0
        call = manager.ensure_models.call_args
        assert call.args[0] == [("qwen3-vl:32b", "vision"), ("gpt-oss:120b", "completion")]
        assert call.kwargs["warm_models"] == ["qwen3-vl:32b"]
        ram.assert_called_once_with(manager, "gpt-oss:120b")

    def test_text_model_too_big_fails_pre_flight(self):
        args = self._make_args(text=True)
        with patch("cli.commands.ModelServerManager") as MockManager, \
                patch("cli.commands._text_model_memory_check", return_value=2), \
                patch("pytesseract.get_tesseract_version", return_value="5.0"):
            MockManager.return_value.ensure_running.return_value = True
            MockManager.return_value.ensure_models.return_value = True
            assert pre_flight_checks(args) == 2
        assert args.server_manager is None

    def test_text_model_none_disables_the_stage(self):
        args = self._make_args(text=True)
        args.settings = dataclasses.replace(
            args.settings, text=dataclasses.replace(args.settings.text, model="none")
        )
        with patch("cli.commands.ModelServerManager") as MockManager, \
                patch("pytesseract.get_tesseract_version", return_value="5.0"):
            MockManager.return_value.ensure_running.return_value = True
            MockManager.return_value.ensure_models.return_value = True
            assert pre_flight_checks(args) == 0
        assert MockManager.return_value.ensure_models.call_args.args[0] == [("qwen3-vl:32b", "vision")]

    def test_dry_run_logs_stages_text_model_provider_and_ram(self, caplog, monkeypatch):
        from cli import commands

        monkeypatch.setattr(commands, "total_memory_bytes", lambda: 128 * GiB)
        args = self._make_args(text=True, dry_run=True)
        with patch("cli.commands.ModelServerManager") as MockManager, \
                patch("pytesseract.get_tesseract_version", return_value="5.0"), \
                caplog.at_level(logging.INFO, logger="cli.commands"):
            manager = MockManager.return_value
            manager.probe.return_value.is_running = True
            manager.ensure_models.return_value = True
            manager.server.base_url = "http://localhost:11434"
            manager.installed_model_size.return_value = 65 * GiB
            assert pre_flight_checks(args) == 0
        assert "Dry run: active stages: METADATA_GENERATION, VISION_ANALYSIS" in caplog.text
        assert "Dry run: text model: gpt-oss:120b (fact check on)" in caplog.text
        assert "Dry run: web search provider: duckduckgo" in caplog.text
        assert "Dry run: RAM check of text model gpt-oss:120b: ok" in caplog.text

    def test_bad_hints_file_exits_1_before_model_work(self, tmp_path):
        hints = tmp_path / "hints.toml"
        hints.write_text('[bottle.A]\n[bottle.A.fields]\nproof = "x"\n')
        args = self._make_args(hints_file=hints)
        with patch("cli.commands.ModelServerManager") as MockManager, \
                patch("pytesseract.get_tesseract_version", return_value="5.0"):
            assert pre_flight_checks(args) == 1
        MockManager.assert_not_called()

    def test_good_hints_file_loaded_into_context(self, tmp_path):
        hints = tmp_path / "hints.toml"
        hints.write_text('[bottle.IMG_1]\nnote = "neck label"\n')
        args = self._make_args(hints_file=hints)
        with patch("cli.commands.ModelServerManager") as MockManager, \
                patch("pytesseract.get_tesseract_version", return_value="5.0"):
            MockManager.return_value.ensure_running.return_value = True
            MockManager.return_value.ensure_models.return_value = True
            assert pre_flight_checks(args) == 0
        assert args.hints.note_for("IMG_1") == "neck label"

# ---------------------------------------------------------------------------
# Process command with usage summary
# ---------------------------------------------------------------------------


class TestProcessCommandIntegration:
    """Test that process_command prints usage summary and cleans up."""

    @pytest.fixture(autouse=True)
    def _folders(self, tmp_path):
        """A real photo folder with one image and a not-yet-created output dir."""
        self.tmp_path = tmp_path
        self.image_folder = tmp_path / "photos"
        self.image_folder.mkdir()
        (self.image_folder / "IMG_1.jpg").write_bytes(b"\xff\xd8\xff\xe0\x00\x10JFIF")
        self.output_dir = tmp_path / "out"

    def test_process_command_uses_model_selector(self):
        """Process command builds the manager and selector when pre-flight did not."""
        args = _ctx(self.tmp_path, batch_size="all", model={"vision_model": "qwen3-vl:32b"}, dry_run=True)

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
                assert exit_code == 0
                assert args.server_manager is mock_manager
                assert args.model_selector is not None

    def test_usage_summary_printed(self, caplog):
        """Verify usage summary is logged when there are model calls."""
        args = _ctx(self.tmp_path, batch_size="all", model={"vision_model": "qwen3-vl:32b"}, dry_run=True)
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
                    "total_duration_seconds": 12.5,
                    "by_stage": {"VISION_ANALYSIS": {"total": 5, "success": 4}},
                    "by_model": {"qwen3-vl:32b": {"total": 5, "success": 4}},
                    "records": [],
                }
                args.model_selector = mock_selector
                args.server_manager = mock_manager

                with patch("pytesseract.get_tesseract_version", return_value="5.0"):
                    process_command(args)

                # Verify usage summary was logged
                assert "Model Usage Summary" in caplog.text
                assert "Total calls:      5" in caplog.text
                assert "Successful calls: 4" in caplog.text
                assert "allback" not in caplog.text

    def test_server_manager_closed_on_exit(self):
        """Verify server manager is closed after processing."""
        args = _ctx(self.tmp_path, batch_size="all", model={"vision_model": "qwen3-vl:32b"}, dry_run=True)

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
        args = _ctx(self.tmp_path, batch_size="all", model={"vision_model": "qwen3-vl:32b"}, dry_run=True)
        return args

    def test_dry_run_does_not_call_get_client(self):
        args = self._dry_run_args()
        mock_manager = MagicMock()
        args.server_manager = mock_manager
        args.model_selector = MagicMock()
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
            a.server_manager = mock_manager
            return 2

        with patch("cli.commands.pre_flight_checks", side_effect=failing_pre_flight):
            assert process_command(args) == 2
        mock_manager.close.assert_called_once()

    def test_real_run_get_client_failure_returns_2(self):
        from lib.exceptions import ModelServerError

        args = self._dry_run_args()
        args.settings = dataclasses.replace(args.settings, dry_run=False)
        mock_manager = MagicMock()
        mock_manager.get_client.side_effect = ModelServerError("ollama model server is not running")
        args.server_manager = mock_manager
        args.model_selector = MagicMock()
        with patch("cli.commands.pre_flight_checks", return_value=0):
            assert process_command(args) == 2
        mock_manager.close.assert_called_once()


# ---------------------------------------------------------------------------
# Retry command server lifecycle
# ---------------------------------------------------------------------------


class TestRetryCommandServer:
    """retry builds the manager from CLI flags, checks it, and always closes it."""

    def _write_report(self, tmp_path, **extra):
        image = tmp_path / "IMG_0001.jpg"
        image.write_bytes(b"\xff\xd8\xff\xe0\x00\x10JFIF")
        report = tmp_path / "error_report.json"
        report.write_text(
            json.dumps({"bottle_id": "b1", "images": [str(image)], "error_message": "boom", **extra})
        )
        return report

    @pytest.mark.parametrize(
        "suggested_fix",
        [
            {"nmae": "Stagg"},  # typo: would otherwise "apply" to a key nothing reads
            {"name": None},  # would otherwise publish the text "None"
            ["name", "Stagg"],
        ],
    )
    def test_retry_refuses_unusable_suggested_fix_before_model_work(self, tmp_path, suggested_fix):
        args = parse_args(
            ["retry", "--report", str(self._write_report(tmp_path, suggested_fix=suggested_fix)),
             "-o", str(tmp_path / "out")]
        )
        with patch("cli.commands.ModelServerManager") as MockManager:
            assert retry_command(args) == 1
        MockManager.assert_not_called()

    def test_retry_applies_suggested_fix_to_the_fresh_reading(self, tmp_path):
        from models.bottle import ExtractedMetadata

        image = tmp_path / "IMG_0001.jpg"
        report = self._write_report(tmp_path, suggested_fix={"name": "Stagg Jr", "bottle_id": "b1"})
        out = tmp_path / "out"
        args = parse_args(["retry", "--report", str(report), "-o", str(out)])
        metadata = ExtractedMetadata(source_images=[str(image)])
        bottle_dict = {"name": "Stag", "distillery": "Buffalo Trace", "images": [str(image)]}
        with patch("cli.commands.ModelServerManager") as MockManager, \
                patch("cli.commands.BottleAnalyzer") as MockAnalyzer:
            MockManager.return_value.ensure_running.return_value = True
            MockAnalyzer.return_value.process_single_bottle.return_value = (bottle_dict, metadata)
            MockAnalyzer.return_value.detect_quantity.return_value = 1
            assert retry_command(args) == 0
        (saved_json,) = [p for p in out.rglob("*.json") if p.parent != out]
        saved = json.loads(saved_json.read_text())
        assert saved["name"] == "Stagg Jr"
        (correction,) = saved["corrections"]
        assert correction["bottle_id"] == "b1"
        assert correction["original_value"] == "Stag"
        assert correction["source"] == f"feedback:{report}"

    def test_retry_returns_1_when_server_unavailable(self, tmp_path):
        args = parse_args(
            ["retry", "--report", str(self._write_report(tmp_path)), "-o", str(tmp_path / "out")]
        )
        with patch("cli.commands.ModelServerManager") as MockManager:
            manager = MockManager.return_value
            manager.ensure_running.return_value = False
            manager.last_error = "Ollama is not reachable"
            assert retry_command(args) == 1
        assert "backend" not in MockManager.call_args.kwargs
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
        self.tmp_path = tmp_path
        self.image_folder = tmp_path / "photos"
        self.image_folder.mkdir()
        (self.image_folder / "IMG_1.jpg").write_bytes(b"\xff\xd8\xff\xe0\x00\x10JFIF")
        self.output_dir = tmp_path / "out"

    def _real_run_args(self):
        args = _ctx(self.tmp_path, batch_size="all", dry_run=False)
        args.server_manager = MagicMock()
        args.model_selector = MagicMock()
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
        args.server_manager.close.assert_called_once()

    def test_failed_bottle_reason_logged_and_all_failed_exit_4(self, caplog):
        from image_processor.batch_processor import BatchResult

        from lib.exceptions import VisionResponseError

        result = BatchResult(batch_id=1)
        result.failed_bottles.append("IMG_1")
        error = VisionResponseError("returned no content")
        error.attempts = 3
        result.errors["IMG_1"] = error
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

    def test_per_stage_override_wins_over_model(self):
        """A per-stage ``<stage>_model`` attribute overrides ``model`` for that stage."""
        from lib.model_selector import ModelSelector, ModelStage

        config = SimpleNamespace(model="qwen3-vl:32b", vision_analysis_model="llama3.2")
        selector = ModelSelector.from_config(config)
        assert selector.get_model(ModelStage.VISION_ANALYSIS) == "llama3.2"

    def test_global_model_sets_vision_model(self):
        """A single --model sets the vision model only."""
        from lib.model_selector import DEFAULT_TEXT_MODEL, ModelSelector, ModelStage

        selector = ModelSelector.from_config(SimpleNamespace(model="llama3.2"))
        assert selector.get_model(ModelStage.VISION_ANALYSIS) == "llama3.2"
        assert selector.get_model(ModelStage.METADATA_GENERATION) == DEFAULT_TEXT_MODEL

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

# ---------------------------------------------------------------------------
# Text-model memory check (advisory; called by the text stage's pre-flight)
# ---------------------------------------------------------------------------


class TestTextModelMemoryCheck:
    """``_text_model_memory_check``: 0 when it fits or is unknown, 2 when too big."""

    @staticmethod
    def _manager(size):
        manager = MagicMock()
        manager.installed_model_size.return_value = size
        return manager

    def test_fits(self, monkeypatch, caplog):
        from cli import commands

        monkeypatch.setattr(commands, "total_memory_bytes", lambda: 128 * GiB)
        with caplog.at_level(logging.WARNING, logger="cli.commands"):
            assert commands._text_model_memory_check(self._manager(65 * GiB), "gpt-oss:120b") == 0
        assert caplog.text == ""

    def test_large_model_warns(self, monkeypatch, caplog):
        from cli import commands

        monkeypatch.setattr(commands, "total_memory_bytes", lambda: 80 * GiB)
        with caplog.at_level(logging.WARNING, logger="cli.commands"):
            assert commands._text_model_memory_check(self._manager(65 * GiB), "gpt-oss:120b") == 0
        assert "large for this machine" in caplog.text
        assert "--text-model gpt-oss:20b" in caplog.text
        assert "--no-verify" in caplog.text

    def test_model_larger_than_memory_is_an_error(self, monkeypatch, caplog):
        from cli import commands

        monkeypatch.setattr(commands, "total_memory_bytes", lambda: 32 * GiB)
        with caplog.at_level(logging.ERROR, logger="cli.commands"):
            assert commands._text_model_memory_check(self._manager(65 * GiB), "gpt-oss:120b") == 2
        assert "does not fit in memory" in caplog.text
        assert "--text-model gpt-oss:20b" in caplog.text
        assert "--no-verify" in caplog.text

    def test_remote_server_is_skipped(self, monkeypatch):
        """A remote Ollama loads the model in its own memory, not this machine's."""
        from cli import commands

        monkeypatch.setattr(commands, "total_memory_bytes", lambda: 8 * GiB)
        manager = self._manager(65 * GiB)
        manager.server.base_url = "http://gpu-box.example:11434"
        assert commands._text_model_memory_check(manager, "gpt-oss:120b") == 0
        manager.installed_model_size.assert_not_called()

        manager.server.base_url = "http://127.0.0.1:11434"
        assert commands._text_model_memory_check(manager, "gpt-oss:120b") == 2

    def test_unknown_size_or_memory_is_ok(self, monkeypatch):
        from cli import commands

        monkeypatch.setattr(commands, "total_memory_bytes", lambda: None)
        assert commands._text_model_memory_check(self._manager(65 * GiB), "gpt-oss:120b") == 0
        assert commands._text_model_memory_check(self._manager(None), "gpt-oss:120b") == 0


# ---------------------------------------------------------------------------
# retry uses the run's retry policy and hints
# ---------------------------------------------------------------------------


class TestRetryCommandSettings:
    def _report(self, tmp_path, bottle_id="IMG_0001"):
        image = tmp_path / f"{bottle_id}.jpg"
        image.write_bytes(b"\xff\xd8\xff\xe0\x00\x10JFIF")
        report = tmp_path / "error_report.json"
        report.write_text(json.dumps({"bottle_id": bottle_id, "images": [str(image)], "error_message": "boom"}))
        return report, image

    def test_retry_passes_hint_note_policy_and_applies_hint_fields(self, tmp_path):
        from lib.retry import RetryPolicy
        from models.bottle import ExtractedMetadata

        report, image = self._report(tmp_path)
        hints = tmp_path / "hints.toml"
        hints.write_text(
            '[bottle.IMG_0001]\nnote = "neck tag"\nsource = "Owner, 2026-11-20"\n'
            '[bottle.IMG_0001.fields]\nbatch_number = "C923"\n'
        )
        out = tmp_path / "out"
        args = parse_args(["retry", "--report", str(report), "-o", str(out), "--hints", str(hints), "--no-retry"])
        metadata = ExtractedMetadata(source_images=[str(image)], batch_number="C922")
        bottle_dict = {"name": "Stagg", "distillery": "Buffalo Trace", "batch_number": "C922", "images": [str(image)]}
        with patch("cli.commands.ModelServerManager") as MockManager, \
                patch("cli.commands.BottleAnalyzer") as MockAnalyzer:
            MockManager.return_value.ensure_running.return_value = True
            MockAnalyzer.return_value.process_single_bottle.return_value = (bottle_dict, metadata)
            MockAnalyzer.return_value.detect_quantity.return_value = 1
            assert retry_command(args) == 0
        assert MockAnalyzer.call_args.kwargs["retry_policy"] == RetryPolicy(max_attempts=1)
        assert MockAnalyzer.return_value.process_single_bottle.call_args.kwargs["reviewer_note"] == "neck tag"
        (saved_json,) = [p for p in out.rglob("*.json") if p.parent != out]
        saved = json.loads(saved_json.read_text())
        (correction,) = saved["corrections"]
        assert correction["source"] == "hint:hints.toml#IMG_0001: Owner, 2026-11-20"
        assert "hint overrides label read for batch_number: label 'C922', hint 'C923'" in saved["metadata"]["warnings"]

    def test_retry_with_bad_hints_exits_1_before_model_work(self, tmp_path):
        report, _ = self._report(tmp_path)
        hints = tmp_path / "hints.toml"
        hints.write_text("[bottle.IMG_0001\n")
        args = parse_args(["retry", "--report", str(report), "--hints", str(hints)])
        with patch("cli.commands.ModelServerManager") as MockManager:
            assert retry_command(args) == 1
        MockManager.assert_not_called()


class TestFailureReportRetryCount:
    def test_failure_report_records_attempts_and_error_type(self, tmp_path):
        from cli.commands import _report_failed_bottles, _RunTotals
        from image_processor.batch_processor import BatchResult
        from image_processor.preview_generator import PreviewGenerator
        from lib.exceptions import VisionResponseError
        from models.image import Image

        error = VisionResponseError("no content")
        error.attempts = 3
        result = BatchResult(batch_id=1, failed_bottles=["IMG_1"], errors={"IMG_1": error})
        totals = _RunTotals()
        _report_failed_bottles(
            result, [[Image(path=str(tmp_path / "IMG_1.jpg"), bottle_ref="IMG_1")]], set(),
            PreviewGenerator(output_dir=tmp_path / "out"), totals,
        )
        (report,) = totals.error_reports
        data = json.loads(report.read_text())
        assert data["retry_count"] == 2
        assert data["error_type"] == "AI_MODEL_FAILURE"
        assert data["error_message"] == "VisionResponseError: no content"


class TestFeedbackApprove:
    def test_approve_is_recorded_and_logged_only(self, tmp_path, caplog):
        from cli.commands import feedback_command

        out = tmp_path / "out"
        out.mkdir()
        (out / "batch_b1.json").write_text(json.dumps({"bottles": [], "metadata": []}))
        args = parse_args(["feedback", "-o", str(out), "--batch", "b1", "--approve", "IMG_1", "--approve", "IMG_2"])
        with caplog.at_level(logging.INFO, logger="cli.commands"):
            assert feedback_command(args) == 0
        assert "--approve is applied by the fact-check stage" in caplog.text
        assert sorted(p.name for p in out.iterdir()) == ["batch_b1.json"]  # nothing else written
