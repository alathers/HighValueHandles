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
from tests.unit.test_enrichment import (
    COMPOSE_REPLY,
    EXTRACT_REPLY,
    LABEL,
    FakeTextLLM,
    web_research,
)

logger = logging.getLogger(__name__)


class _GenericText:
    """Text model stand-in for runs that are not about the fact check: finds nothing."""

    def chat_text(self, prompt, model=None, *, format=None, **_kwargs):
        from lib.ollama_client import ChatResult

        props = (format or {}).get("properties", {})
        body = {"page_match": [], "claims": [], "prices": []} if "claims" in props else {"sentences": []}
        return ChatResult(content=json.dumps(body), done_reason="stop", model=model or "")


@pytest.fixture(autouse=True)
def _hermetic_enrichment(monkeypatch):
    """No web researcher (default DuckDuckGo would be real network) and a fake text model.

    Tests about the enrichment wiring patch these again with scripted fakes.
    """
    import cli.commands as commands

    monkeypatch.setattr(commands, "_build_researcher", lambda settings, client: None)
    real = commands._build_fact_checker

    def build(settings, client, selector):
        checker = real(settings, client, selector)
        if checker.llm is not None:
            checker.llm = _GenericText()
        return checker

    monkeypatch.setattr(commands, "_build_fact_checker", build)


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
            json.dumps({"bottle_id": "b1", "images": [str(image)], "error_message": "boom", "schema_version": 2, **extra})
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
        report.write_text(json.dumps({"bottle_id": "b1", "images": [str(image)], "error_message": "boom", "schema_version": 2}))
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
        report.write_text(json.dumps({"bottle_id": bottle_id, "images": [str(image)], "error_message": "boom", "schema_version": 2}))
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
    def test_v1_batch_is_refused(self, tmp_path, caplog):
        from cli.commands import feedback_command

        out = tmp_path / "out"
        out.mkdir()
        (out / "batch_b1.json").write_text(json.dumps({"bottles": [], "metadata": []}))
        args = parse_args(["feedback", "-o", str(out), "--batch", "b1", "--approve", "IMG_1"])
        with caplog.at_level(logging.ERROR, logger="cli.commands"):
            assert feedback_command(args) == 1
        assert "schema_version None; this version reads only 2" in caplog.text
        assert sorted(p.name for p in out.iterdir()) == ["batch_b1.json"]  # nothing written

    def test_an_ambiguous_bottle_id_approves_nothing(self, caplog):
        """Two bottles share an ID: like corrections, the approval is refused for both."""
        from cli.commands import _apply_approvals
        from tests.unit.test_enrichment import checked_bottle

        a, ma, _ = checked_bottle()
        b, mb, _ = checked_bottle()
        with caplog.at_level(logging.ERROR, logger="cli.commands"):
            approved, problems = _apply_approvals([a, b], [ma, mb], ["IMG_1"], "feedback")
        assert approved == set()
        assert problems == ["bottle ID IMG_1 is not unique in this batch"]
        assert a["facts"]["review"]["approved"] is False and b["facts"]["review"]["approved"] is False

    def test_nothing_to_do_exits_0(self, tmp_path):
        from cli.commands import feedback_command

        out = tmp_path / "out"
        out.mkdir()
        (out / "batch_b1.json").write_text(json.dumps({"schema_version": 2, "bottles": [], "metadata": []}))
        assert feedback_command(parse_args(["feedback", "-o", str(out), "--batch", "b1"])) == 0


# ---------------------------------------------------------------------------
# Enrichment wiring (web research + fact check), in process
# ---------------------------------------------------------------------------


class _ScriptedResearcher:
    provider = type("Provider", (), {"name": "duckduckgo"})()

    def __init__(self):
        self.items = []

    def research_many(self, items, *, workers=1):
        self.items.extend(items)
        return [web_research() for _ in items]


def _label_vision(self, image_paths, ocr_text="", model=None, **_kwargs):
    from models.bottle import ExtractedMetadata

    return ExtractedMetadata(source_images=[str(p) for p in image_paths], vision_model=model,
                             vision_confidence="high", **LABEL)


class TestEnrichmentWiring:
    @pytest.fixture
    def run(self, tmp_path, monkeypatch):
        """Run ``process`` in process on one photo with a scripted researcher and text model."""
        from PIL import Image as PILImage

        import cli.commands as commands

        photos = tmp_path / "photos"
        photos.mkdir()
        PILImage.new("RGB", (32, 48), (120, 60, 20)).save(photos / "IMG_1.jpg", "JPEG")
        out = tmp_path / "out"
        researcher = _ScriptedResearcher()
        llm = FakeTextLLM(EXTRACT_REPLY, COMPOSE_REPLY)
        manager = MagicMock()
        manager.get_client.return_value.chat_text.side_effect = llm.chat_text
        monkeypatch.setattr(commands, "_build_researcher", lambda settings, client: researcher)
        monkeypatch.setattr(commands, "_build_fact_checker", _REAL_BUILD_FACT_CHECKER)

        def go(*extra):
            with patch.object(commands, "pre_flight_checks", return_value=0), \
                    patch.object(commands, "build_server_manager", return_value=manager), \
                    patch("image_processor.text_extractor.TextExtractor.extract_with_retry", lambda self, p, **k: ""), \
                    patch("image_processor.bottle_analyzer.BottleAnalyzer.analyze_with_vision_model", _label_vision):
                rc = commands.main(["process", "-f", str(photos), "-o", str(out), "-b", "5", *extra])
            return rc, out, manager, researcher, llm

        return go

    def test_bottle_json_has_facts_web_research_run_id_and_evidence(self, run, caplog):
        import hashlib

        with caplog.at_level(logging.INFO):
            rc, out, manager, researcher, llm = run()
        assert rc == 0
        (folder,) = [p for p in out.iterdir() if p.is_dir() and not p.name.startswith(".")]
        data = json.loads((folder / f"{folder.name}.json").read_text())
        assert data["schema_version"] == 2
        assert data["run_id"] and data["run_id"] in next(out.glob("run_*_settings.json")).name
        assert data["facts"]["verification"]["status"] == "verified"
        assert data["facts"]["fields"]["msrp"]["value"] == "59.99"
        assert data["msrp"] == "59.99"
        assert data["web_research"]["provider"] == "duckduckgo"
        assert [p["outcome"] for p in data["web_research"]["pages"]] == ["cited", "fetched_not_cited", "skipped"]
        w1 = next(e for e in data["web_research"]["evidence"] if e["source_id"] == "W1")
        assert hashlib.sha256((folder / "evidence" / "W1.txt").read_bytes()).hexdigest() == w1["sha256"]
        assert [link["used_as_evidence"] for link in data["reference_links"]] == [True, False]
        # Model swaps: the vision model is freed before the text model, which is freed after.
        assert [c.args[0] for c in manager.unload_model.call_args_list] == [
            commands_default_vision(), "gpt-oss:120b",
        ]
        assert len(llm.calls) == 2 and llm.calls[0]["keep_alive"] == "10m"
        identity, photo = researcher.items[0]
        assert identity["name"] == LABEL["name"] and photo.name == "IMG_1.jpg"
        # Run summary
        assert "Web search provider: duckduckgo" in caplog.text
        assert "Web research: ok=1" in caplog.text
        assert "Fact check (gpt-oss:120b): verified=1" in caplog.text
        assert "Publish-ready: 0 of 1 saved bottle(s)" in caplog.text
        assert f"web page {web_research().pages[0].url}: cited" in caplog.text

    def test_keep_models_loaded_skips_unloads(self, run):
        rc, _out, manager, _r, llm = run("--keep-models-loaded")
        assert rc == 0
        manager.unload_model.assert_not_called()
        assert llm.calls[0]["keep_alive"] == "30m"

    def test_feedback_approve_after_a_run(self, run):
        from cli.commands import feedback_command

        rc, out, *_ = run()
        assert rc == 0
        (batch_json,) = out.glob("batch_*.json")
        batch_id = batch_json.name[len("batch_"):-len(".json")]
        assert feedback_command(parse_args(["feedback", "-o", str(out), "--batch", batch_id,
                                            "--approve", "IMG_1"])) == 0
        resaved = json.loads(next(out.glob("*_2/*_2.json")).read_text())
        assert resaved["facts"]["review"]["approved"] is True
        assert resaved["facts"]["review"]["by"] == "feedback"
        # Still blocked by the proof conflict: approval never overrides the other checks.
        assert resaved["facts"]["publish"]["ready"] is False
        assert any("unresolved conflict proof-1" in b for b in resaved["facts"]["publish"]["blockers"])

    def test_no_verify_keeps_web_out_of_fields(self, run):
        rc, out, manager, _r, llm = run("--no-verify")
        assert rc == 0
        assert llm.calls == []
        manager.unload_model.assert_not_called()
        data = json.loads(next(p for p in out.rglob("*.json") if p.parent != out).read_text())
        assert data["facts"]["verification"]["status"] == "skipped"
        assert data["msrp"] is None
        assert data["facts"]["fields"]["age"]["status"] == "unknown"

    def test_retry_runs_enrichment(self, tmp_path, monkeypatch):
        import cli.commands as commands
        from models.bottle import ExtractedMetadata

        image = tmp_path / "IMG_1.jpg"
        image.write_bytes(b"\xff\xd8\xff\xe0\x00\x10JFIF")
        report = tmp_path / "error_report.json"
        report.write_text(json.dumps({"schema_version": 2, "bottle_id": "IMG_1", "images": [str(image)],
                                      "error_message": "boom"}))
        researcher = _ScriptedResearcher()
        llm = FakeTextLLM(EXTRACT_REPLY, COMPOSE_REPLY)
        monkeypatch.setattr(commands, "_build_researcher", lambda settings, client: researcher)
        monkeypatch.setattr(commands, "_build_fact_checker", _REAL_BUILD_FACT_CHECKER)
        out = tmp_path / "out"
        metadata = ExtractedMetadata(source_images=[str(image)], vision_confidence="high", **LABEL)
        bottle_dict = {"name": LABEL["name"], "distillery": LABEL["distillery"], "proof": LABEL["proof"],
                       "label_price": LABEL["label_price"], "msrp": None, "images": [str(image)]}
        with patch("cli.commands.ModelServerManager") as MockManager, \
                patch("cli.commands.BottleAnalyzer") as MockAnalyzer:
            MockManager.return_value.ensure_running.return_value = True
            MockManager.return_value.get_client.return_value.chat_text.side_effect = llm.chat_text
            MockAnalyzer.return_value.process_single_bottle.return_value = (bottle_dict, metadata)
            MockAnalyzer.return_value.detect_quantity.return_value = 1
            assert retry_command(parse_args(["retry", "--report", str(report), "-o", str(out)])) == 0
            unloads = [c.args[0] for c in MockManager.return_value.unload_model.call_args_list]
        (saved_json,) = [p for p in out.rglob("*.json") if p.parent != out]
        saved = json.loads(saved_json.read_text())
        assert saved["facts"]["verification"]["status"] == "verified"
        assert saved["web_research"]["status"] == "ok"
        assert saved["run_id"].startswith("retry_")
        assert saved["msrp"] == "59.99"
        assert (saved_json.parent / "evidence" / "W1.txt").is_file()
        assert unloads == [commands_default_vision(), "gpt-oss:120b"]
        assert len(researcher.items) == 1

    def test_retry_refuses_v1_report(self, tmp_path, caplog):
        image = tmp_path / "IMG_1.jpg"
        image.write_bytes(b"x")
        report = tmp_path / "error_report.json"
        report.write_text(json.dumps({"bottle_id": "IMG_1", "images": [str(image)], "error_message": "boom"}))
        with patch("cli.commands.ModelServerManager") as MockManager, caplog.at_level(logging.ERROR):
            assert retry_command(parse_args(["retry", "--report", str(report)])) == 1
        MockManager.assert_not_called()
        assert "schema_version None" in caplog.text


class TestEnricherBuilders:
    def test_researcher_none_when_web_off_or_provider_none(self):
        from cli.commands import _web_skip_reason

        off = make_settings(web={"enabled": False})
        assert _REAL_BUILD_RESEARCHER(off, MagicMock()) is None
        assert "disabled" in _web_skip_reason(off)
        none = make_settings(web={"provider_resolved": "none", "provider_reason": "web.provider = 'none'"})
        assert _REAL_BUILD_RESEARCHER(none, MagicMock()) is None
        assert _web_skip_reason(none) == "web.provider = 'none'"

    def test_duckduckgo_researcher_built_without_network(self, tmp_path):
        settings = make_settings(web={"provider_resolved": "duckduckgo", "cache_dir": tmp_path / "cache",
                                      "vendor_images": True, "max_pages": 3})
        researcher = _REAL_BUILD_RESEARCHER(settings, MagicMock())
        assert researcher.provider.name == "duckduckgo"
        assert researcher.options.max_pages == 3
        assert researcher.options.vendor_images is True
        assert researcher.vendor_verifier is not None
        assert researcher.fetcher.cache.root == tmp_path / "cache"

    def test_fact_checker_settings(self):
        from lib.model_selector import ModelStage

        selector = MagicMock()
        settings = make_settings(retry={"enabled": False}, model={"keep_models_loaded": True})
        client = object()
        checker = _REAL_BUILD_FACT_CHECKER(settings, client, selector)
        assert checker.llm is client
        assert checker.max_attempts == 1
        assert checker.keep_alive == "30m"
        assert checker.model == settings.text.model
        checker.usage_recorder("gpt-oss:120b", 1.5, True, "high")
        selector.record_usage.assert_called_once_with(
            ModelStage.METADATA_GENERATION, "gpt-oss:120b", success=True, duration=1.5, think="high"
        )
        off = make_settings(text={"enabled": False})
        assert _REAL_BUILD_FACT_CHECKER(off, client, selector).llm is None
        assert _REAL_BUILD_FACT_CHECKER(make_settings(), client, selector).max_attempts == 2

    def test_researcher_build_failure_skips_web(self, monkeypatch):
        import cli.commands as commands

        def boom(settings, client):
            raise ValueError("bad searxng url")

        monkeypatch.setattr(commands, "_build_researcher", boom)
        enricher = commands.build_enricher(RunContext(make_settings()), MagicMock())
        assert enricher.researcher is None
        assert "could not be built" in enricher.web_skip_reason


def commands_default_vision():
    return DEFAULT_VISION_MODEL


import cli.commands as _commands_module  # noqa: E402

_REAL_BUILD_RESEARCHER = _commands_module._build_researcher
_REAL_BUILD_FACT_CHECKER = _commands_module._build_fact_checker
