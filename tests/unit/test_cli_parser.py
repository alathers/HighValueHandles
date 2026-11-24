"""Unit tests for the argparse surface: every default is None, so config is never clobbered."""

import pytest

from cli.commands import build_parser, parse_args
from lib.settings import resolve_settings

#: argparse bookkeeping dests that are not setting values.
_NON_SETTING_DESTS = {"command", "help", "report", "config_action"}


def _dests(subcommand: str) -> set[str]:
    parser = build_parser()
    sub = next(a for a in parser._subparsers._group_actions)  # noqa: SLF001 - test introspection
    return {a.dest for a in sub.choices[subcommand]._actions} - _NON_SETTING_DESTS  # noqa: SLF001


class TestDefaultsAreNone:
    @pytest.mark.parametrize(
        ("argv", "subcommand"),
        [
            (["process"], "process"),
            (["feedback"], "feedback"),
            (["retry", "--report", "r.json"], "retry"),
            (["config", "show"], "config"),
        ],
    )
    def test_every_dest_defaults_to_none(self, argv, subcommand):
        args = vars(parse_args(argv))
        dests = _dests(subcommand)
        assert dests  # the introspection found the options
        not_none = {d: args[d] for d in dests if args.get(d) is not None}
        assert not_none == {}

    def test_top_level_defaults_none(self):
        args = parse_args(["process"])
        assert (args.debug, args.config, args.no_config) == (None, None, None)


class TestAbsentFlagDoesNotClobberConfig:
    def test_file_value_kept_unless_flag_given(self, tmp_path):
        (tmp_path / "bourbon.toml").write_text("[processing]\nconcurrency = 4\n")
        env = {"XDG_CONFIG_HOME": str(tmp_path / "xdg")}
        assert resolve_settings(parse_args(["process"]), env, cwd=tmp_path).concurrency == 4
        assert resolve_settings(parse_args(["process", "--concurrency", "1"]), env, cwd=tmp_path).concurrency == 1

    def test_boolean_pair_absent_keeps_file_value(self, tmp_path):
        (tmp_path / "bourbon.toml").write_text("[server]\nkeep_running = true\n[web]\nenabled = false\n")
        env = {"XDG_CONFIG_HOME": str(tmp_path / "xdg")}
        s = resolve_settings(parse_args(["process"]), env, cwd=tmp_path)
        assert s.server.keep_running is True and s.web.enabled is False
        s = resolve_settings(parse_args(["process", "--no-keep-server", "--web"]), env, cwd=tmp_path)
        assert s.server.keep_running is False and s.web.enabled is True


class TestFlags:
    def test_retry_flags(self):
        assert parse_args(["process", "-r"]).retry_enabled is True
        assert parse_args(["process", "--allow-retry"]).retry_enabled is True
        assert parse_args(["process", "--no-retry"]).retry_enabled is False
        assert parse_args(["process"]).retry_enabled is None
        assert parse_args(["retry", "--report", "r", "--no-retry", "--retry-attempts", "2"]).retry_attempts == 2

    def test_vendor_images(self):
        assert parse_args(["process", "-v"]).include_vendor_images is True
        assert parse_args(["process", "--include-vendor-images"]).include_vendor_images is True
        assert parse_args(["process", "--no-include-vendor-images"]).include_vendor_images is False

    def test_verify(self):
        assert parse_args(["process", "--verify"]).verify is True
        assert parse_args(["process", "--no-verify"]).verify is False
        assert parse_args(["retry", "--report", "r", "--no-verify"]).verify is False

    def test_approve_repeatable(self):
        args = parse_args(["feedback", "--approve", "IMG_1", "--approve", "IMG_2"])
        assert args.approve == ["IMG_1", "IMG_2"]

    def test_interactive(self):
        assert parse_args(["process", "-i"]).interactive is True
        assert parse_args(["process", "--no-interactive"]).interactive is False
        assert parse_args(["feedback", "-i"]).interactive is True

    @pytest.mark.parametrize("subcommand", [["process"], ["retry", "--report", "r"]])
    def test_web_and_text_flags_on_process_and_retry(self, subcommand):
        args = parse_args(
            [*subcommand, "--web", "--web-provider", "brave", "--web-cache-dir", "c", "--refresh-web",
             "--text-model", "gpt-oss:20b", "--text-think", "low", "--text-timeout", "60",
             "--keep-models-loaded", "--hints", "h.toml", "-v"]
        )
        assert (args.web, args.web_provider, args.refresh_web, args.include_vendor_images) == (True, "brave", True, True)
        assert (args.text_model, args.text_think, args.text_timeout) == ("gpt-oss:20b", "low", 60.0)
        assert args.keep_models_loaded is True
        assert str(args.hints_file) == "h.toml"

    def test_config_flags_before_and_after_subcommand(self):
        assert str(parse_args(["--config", "a.toml", "process"]).config) == "a.toml"
        assert str(parse_args(["process", "--config", "b.toml"]).config) == "b.toml"
        assert parse_args(["--no-config", "feedback"]).no_config is True
        # --config's value is not mistaken for the subcommand position.
        assert parse_args(["--config", "a.toml", "-f", "photos"]).command == "process"

    def test_config_subcommand(self):
        args = parse_args(["config", "show", "--json"])
        assert (args.command, args.config_action, args.json) == ("config", "show", True)
        assert parse_args(["config", "template"]).config_action == "template"
        with pytest.raises(SystemExit) as excinfo:
            parse_args(["config", "edit"])
        assert excinfo.value.code == 2

    @pytest.mark.parametrize(
        "argv",
        [
            ["process", "--lookahead", "3"],
            ["process", "--server-backend", "ollama"],
            ["process", "--vision-timeout", "45"],
            ["process", "--pull-missing"],
            ["process", "--no-server-autostartx"],
            ["retry", "--report", "r", "--lookahead", "2"],
            ["feedback", "--concurrency", "2"],
        ],
    )
    def test_removed_or_foreign_flags_exit_2(self, argv):
        with pytest.raises(SystemExit) as excinfo:
            parse_args(argv)
        assert excinfo.value.code == 2

    def test_argparse_type_errors_exit_2(self):
        with pytest.raises(SystemExit) as excinfo:
            parse_args(["process", "--concurrency", "abc"])
        assert excinfo.value.code == 2
