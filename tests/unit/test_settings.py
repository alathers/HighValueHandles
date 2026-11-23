"""Unit tests for lib.settings: registry, precedence, files, env, validation, provider, output."""

import argparse
import json
import tomllib
from pathlib import Path

import pytest

from cli.commands import parse_args
from lib.retry import RetryPolicy
from lib.settings import (
    NO_PROVIDER_REASON,
    REGISTRY,
    SPECS,
    SettingsError,
    WebSettings,
    describe,
    render_template,
    resolve_settings,
    resolve_web_provider,
    template_defaults,
    user_config_path,
)
from tests.helpers import make_settings

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def env(tmp_path):
    """A minimal environment with XDG dirs under tmp_path (no keys, no BOURBON_*)."""
    return {"XDG_CONFIG_HOME": str(tmp_path / "xdg"), "XDG_CACHE_HOME": str(tmp_path / "cache")}


def _resolve(argv, environ, cwd):
    return resolve_settings(parse_args(argv), environ, cwd=cwd)


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _user_file(env, text):
    return _write(Path(env["XDG_CONFIG_HOME"]) / "bourbon-process" / "config.toml", text)


def _errors(excinfo) -> str:
    return "\n".join(excinfo.value.errors)


class TestDefaults:
    def test_defaults(self, env, tmp_path):
        s = _resolve(["process"], env, tmp_path)
        assert s.command == "process"
        assert s.image_folder == tmp_path
        assert s.output_dir == tmp_path / "processed"
        assert (s.batch_size, s.concurrency, s.interactive, s.dry_run) == (5, 2, False, False)
        assert s.hints_file is None
        assert s.model.vision_model == "qwen3.6:35b-a3b"
        assert s.text.model == "gpt-oss:120b" and s.text.active
        assert (s.text.think_extract, s.text.think_compose, s.text.num_ctx) == ("high", "medium", 32768)
        assert s.server.ollama_url == "http://localhost:11434"
        assert s.retry.policy() == RetryPolicy(max_attempts=3, initial_delay=2.0, max_delay=30.0)
        assert s.web.enabled and s.web.provider == "auto"
        assert s.web.cache_dir == tmp_path / "cache" / "bourbon-process" / "web"
        assert all(source == "default" for source in s.sources.values())
        assert set(s.sources) == set(SPECS)

    def test_text_active_property(self):
        assert not make_settings(text={"enabled": False}).text.active
        assert not make_settings(text={"model": "None"}).text.active
        assert make_settings().text.active

    def test_retry_disabled_policy_has_one_attempt(self):
        assert make_settings(retry={"enabled": False}).retry.policy().max_attempts == 1
        assert make_settings(retry={"max_attempts": 5}).retry.policy().max_attempts == 5


class TestPrecedence:
    def test_default_user_project_env_cli(self, env, tmp_path):
        assert _resolve(["process"], env, tmp_path).concurrency == 2
        _user_file(env, "[processing]\nconcurrency = 3\nbatch_size = 7\n")
        s = _resolve(["process"], env, tmp_path)
        assert (s.concurrency, s.batch_size) == (3, 7)
        assert s.sources["processing.concurrency"].startswith("file:") and "xdg" in s.sources["processing.concurrency"]

        project = _write(tmp_path / "bourbon.toml", "[processing]\nconcurrency = 4\n")
        s = _resolve(["process"], env, tmp_path)
        assert (s.concurrency, s.batch_size) == (4, 7)  # project over user, user still fills the rest
        assert s.sources["processing.concurrency"] == f"file:{project}"
        assert s.config_files == (_user_file(env, "[processing]\nconcurrency = 3\nbatch_size = 7\n"), project)

        s = _resolve(["process"], {**env, "BOURBON_CONCURRENCY": "5"}, tmp_path)
        assert s.concurrency == 5
        assert s.sources["processing.concurrency"] == "env:BOURBON_CONCURRENCY"

        s = _resolve(["process", "--concurrency", "6"], {**env, "BOURBON_CONCURRENCY": "5"}, tmp_path)
        assert s.concurrency == 6
        assert s.sources["processing.concurrency"] == "cli:--concurrency"

    def test_config_flag_replaces_discovery(self, env, tmp_path):
        _user_file(env, "[processing]\nconcurrency = 3\n")
        _write(tmp_path / "bourbon.toml", "[processing]\nconcurrency = 4\n")
        other = _write(tmp_path / "other" / "mine.toml", "[processing]\nbatch_size = 9\n")
        s = _resolve(["--config", str(other), "process"], env, tmp_path)
        assert (s.concurrency, s.batch_size) == (2, 9)
        assert s.config_files == (other,)
        assert s.config_searched == (other,)
        # After the subcommand too
        assert _resolve(["process", "--config", str(other)], env, tmp_path).batch_size == 9

    def test_missing_config_file_is_an_error(self, env, tmp_path):
        with pytest.raises(SettingsError, match="config file not found"):
            _resolve(["process", "--config", str(tmp_path / "nope.toml")], env, tmp_path)

    def test_bourbon_config_env(self, env, tmp_path):
        other = _write(tmp_path / "env.toml", "[processing]\nbatch_size = \"all\"\n")
        assert _resolve(["process"], {**env, "BOURBON_CONFIG": str(other)}, tmp_path).batch_size == "all"
        with pytest.raises(SettingsError, match="config file not found"):
            _resolve(["process"], {**env, "BOURBON_CONFIG": str(tmp_path / "gone.toml")}, tmp_path)

    def test_no_config_skips_every_file(self, env, tmp_path):
        _user_file(env, "[processing]\nconcurrency = 3\n")
        _write(tmp_path / "bourbon.toml", "[processing]\nbatch_size = \"abc\"\n")
        s = _resolve(["process", "--no-config"], env, tmp_path)
        assert (s.concurrency, s.batch_size, s.config_files, s.config_searched) == (2, 5, (), ())
        assert _resolve(["process"], {**env, "BOURBON_NO_CONFIG": "1"}, tmp_path).config_files == ()

    def test_config_and_no_config_together_is_an_error(self, env, tmp_path):
        other = _write(tmp_path / "x.toml", "")
        with pytest.raises(SettingsError, match="cannot be used together"):
            _resolve(["process", "--config", str(other), "--no-config"], env, tmp_path)

    def test_xdg_config_home_honoured_with_home_fallback(self, tmp_path, monkeypatch):
        assert user_config_path({"XDG_CONFIG_HOME": "/x/y"}) == Path("/x/y/bourbon-process/config.toml")
        monkeypatch.setenv("HOME", str(tmp_path / "home"))
        assert user_config_path({}) == tmp_path / "home" / ".config" / "bourbon-process" / "config.toml"
        assert user_config_path({"XDG_CONFIG_HOME": ""}) == user_config_path({})


class TestFileRules:
    @pytest.mark.parametrize(
        ("text", "needle"),
        [
            ("[processing]\nlookahead = 3\n", "unknown key 'lookahead'"),
            ("[server]\nserver_backend = \"mtplx\"\n", "unknown key 'server_backend'"),
            ("[processsing]\nconcurrency = 2\n", "unknown section [processsing]"),
            ("[web]\nrefresh = true\n", "unknown key 'refresh'"),  # CLI-only
        ],
    )
    def test_unknown_keys_and_sections(self, env, tmp_path, text, needle):
        _write(tmp_path / "bourbon.toml", text)
        with pytest.raises(SettingsError) as excinfo:
            _resolve(["process"], env, tmp_path)
        assert needle in _errors(excinfo)
        assert str(tmp_path / "bourbon.toml") in _errors(excinfo)

    def test_strict_types(self, env, tmp_path):
        _write(
            tmp_path / "bourbon.toml",
            '[processing]\nconcurrency = "5"\ninteractive = "yes"\n[server]\nrequest_timeout = 90\n',
        )
        with pytest.raises(SettingsError) as excinfo:
            _resolve(["process"], env, tmp_path)
        text = _errors(excinfo)
        assert "concurrency='5'" in text and "must be an integer" in text
        assert "interactive='yes'" in text
        assert "request_timeout" not in text  # a float field accepts an int

    @pytest.mark.parametrize(("value", "expected"), [("3", 3), ('"all"', "all"), ('"ALL"', "all")])
    def test_batch_size_int_or_all(self, env, tmp_path, value, expected):
        _write(tmp_path / "bourbon.toml", f"[processing]\nbatch_size = {value}\n")
        assert _resolve(["process"], env, tmp_path).batch_size == expected

    def test_batch_size_from_cli_and_env(self, env, tmp_path):
        assert _resolve(["process", "-b", "all"], env, tmp_path).batch_size == "all"
        assert _resolve(["process"], {**env, "BOURBON_BATCH_SIZE": "12"}, tmp_path).batch_size == 12
        with pytest.raises(SettingsError, match="batch_size=-1 .from flag -b/--batch-size.: must be an integer >= 1"):
            _resolve(["process", "-b", "-1"], env, tmp_path)

    @pytest.mark.parametrize("name", ["api_key", "brave_api_key", "token", "client_secret"])
    def test_secrets_in_files_rejected(self, env, tmp_path, name):
        _write(tmp_path / "bourbon.toml", f'[web]\n{name} = "sk-123"\n')
        with pytest.raises(SettingsError) as excinfo:
            _resolve(["process"], env, tmp_path)
        assert "BRAVE_API_KEY / OLLAMA_API_KEY environment variables" in _errors(excinfo)
        assert "sk-123" not in _errors(excinfo)

    def test_bad_toml(self, env, tmp_path):
        _write(tmp_path / "bourbon.toml", "[processing\n")
        with pytest.raises(SettingsError, match="invalid TOML"):
            _resolve(["process"], env, tmp_path)

    def test_relative_paths_resolve_against_the_file_and_tilde_expands(self, env, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path / "home"))
        conf = _write(
            tmp_path / "conf" / "c.toml",
            '[input]\nimage_folder = "photos"\n[output]\noutput_dir = "~/out"\n'
            '[processing]\nhints_file = "hints.toml"\n',
        )
        s = _resolve(["process", "--config", str(conf)], env, tmp_path / "elsewhere")
        assert s.image_folder == tmp_path / "conf" / "photos"
        assert s.output_dir == tmp_path / "home" / "out"
        assert s.hints_file == tmp_path / "conf" / "hints.toml"

    def test_env_and_cli_paths_resolve_against_cwd(self, env, tmp_path):
        s = _resolve(["process", "-f", "shoot"], {**env, "BOURBON_OUTPUT_DIR": "out"}, tmp_path)
        assert s.image_folder == tmp_path / "shoot"
        assert s.output_dir == tmp_path / "out"

    def test_empty_optional_path_means_unset(self, env, tmp_path):
        _write(tmp_path / "bourbon.toml", '[processing]\nhints_file = ""\n[web]\ncache_dir = ""\n')
        s = _resolve(["process"], env, tmp_path)
        assert s.hints_file is None
        assert s.web.cache_dir == tmp_path / "cache" / "bourbon-process" / "web"


class TestEnvironment:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [("1", True), ("true", True), ("YES", True), ("on", True), ("0", False), ("False", False),
         ("no", False), ("OFF", False)],
    )
    def test_strict_env_bools(self, env, tmp_path, raw, expected):
        assert _resolve(["process"], {**env, "BOURBON_INTERACTIVE": raw}, tmp_path).interactive is expected

    def test_invalid_env_bool_is_an_error(self, env, tmp_path):
        with pytest.raises(SettingsError, match="interactive='maybe' .from env BOURBON_INTERACTIVE."):
            _resolve(["process"], {**env, "BOURBON_INTERACTIVE": "maybe"}, tmp_path)

    def test_empty_env_value_is_unset(self, env, tmp_path):
        s = _resolve(["process"], {**env, "BOURBON_CONCURRENCY": "", "BOURBON_MODEL": "  "}, tmp_path)
        assert s.concurrency == 2
        assert s.sources["processing.concurrency"] == "default"
        assert s.model.vision_model == "qwen3.6:35b-a3b"

    def test_env_names(self, env, tmp_path):
        s = _resolve(
            ["process"],
            {**env, "BOURBON_MODEL": "gemma4:26b", "BOURBON_TEXT_MODEL": "gpt-oss:20b", "BOURBON_VERIFY": "0",
             "BOURBON_RETRY": "off", "BOURBON_WEB_PROVIDER": "duckduckgo"},
            tmp_path,
        )
        assert s.model.vision_model == "gemma4:26b"
        assert s.text.model == "gpt-oss:20b" and not s.text.enabled
        assert not s.retry.enabled
        assert s.web.provider_resolved == "duckduckgo"


class TestValidation:
    def test_all_errors_collected_with_sources(self, env, tmp_path):
        _write(tmp_path / "bourbon.toml", "[grouping]\nmax_group_size = 50\n")
        with pytest.raises(SettingsError) as excinfo:
            _resolve(
                ["process", "--split-budget-ratio", "3", "-u", "ftp://x"],
                {**env, "BOURBON_CONCURRENCY": "0"},
                tmp_path,
            )
        errors = excinfo.value.errors
        assert len(errors) == 4
        text = _errors(excinfo)
        assert "concurrency=0 (from env BOURBON_CONCURRENCY): must be between 1 and 10" in text
        assert f"max_group_size=50 (from {tmp_path / 'bourbon.toml'} [grouping])" in text
        assert "split_budget_ratio=3.0 (from flag --split-budget-ratio)" in text
        assert "ollama_url='ftp://x' (from flag -u/--ollama-url)" in text

    @pytest.mark.parametrize(
        "argv",
        [
            ["--group-gap-seconds", "0"], ["--group-gap-seconds", "nan"], ["--group-gap-seconds", "601"],
            ["--group-max-size", "-1"], ["--group-max-size", "21"], ["--split-budget-ratio", "inf"],
            ["--min-free-gb", "-1"], ["--request-timeout", "0"], ["--server-start-timeout", "-5"],
            ["--retry-attempts", "6"], ["--text-timeout", "0"], ["--text-think", "max"],
            ["--web-provider", "google"], ["-u", "http://localhost:99999"], ["-u", "not-a-url"],
            ["-m", " "],
        ],
    )
    def test_invalid_values(self, env, tmp_path, argv):
        with pytest.raises(SettingsError):
            _resolve(["process", *argv], env, tmp_path)

    def test_max_delay_must_cover_initial_delay(self, env, tmp_path):
        _write(tmp_path / "bourbon.toml", "[retry]\ninitial_delay_seconds = 10.0\nmax_delay_seconds = 5.0\n")
        with pytest.raises(SettingsError, match="must be >= retry.initial_delay_seconds"):
            _resolve(["process"], env, tmp_path)

    def test_vendor_images_need_web(self, env, tmp_path):
        with pytest.raises(SettingsError, match="need web lookups"):
            _resolve(["process", "-v", "--no-web"], env, tmp_path)
        s = _resolve(["process", "-v"], env, tmp_path)
        assert s.web.vendor_images and s.web.enabled


class TestWebProvider:
    @staticmethod
    def _web(provider="auto", searxng_url="", enabled=True):
        base = make_settings().web
        return WebSettings(**{**{f: getattr(base, f) for f in base.__dataclass_fields__},
                              "provider": provider, "searxng_url": searxng_url, "enabled": enabled})

    def test_auto_order(self):
        url = "http://searx.local:8080"
        all_keys = {"BRAVE_API_KEY": "b", "OLLAMA_API_KEY": "o"}
        assert resolve_web_provider(self._web(searxng_url=url), all_keys)[0] == "brave"
        assert resolve_web_provider(self._web(searxng_url=url), {"BOURBON_BRAVE_API_KEY": "b"})[0] == "brave"
        assert resolve_web_provider(self._web(searxng_url=url), {"OLLAMA_API_KEY": "o"})[0] == "ollama"
        assert resolve_web_provider(self._web(searxng_url=url), {})[0] == "searxng"
        name, reason = resolve_web_provider(self._web(), {})
        assert name == "duckduckgo"  # keyless default
        assert "DuckDuckGo" in reason

    def test_explicit_provider_without_key_is_an_error(self, env, tmp_path):
        with pytest.raises(SettingsError) as excinfo:
            _resolve(["process", "--web-provider", "brave"], env, tmp_path)
        assert "BRAVE_API_KEY" in _errors(excinfo)
        with pytest.raises(SettingsError, match="OLLAMA_API_KEY"):
            _resolve(["process", "--web-provider", "ollama"], env, tmp_path)
        with pytest.raises(SettingsError, match="searxng_url"):
            _resolve(["process", "--web-provider", "searxng"], env, tmp_path)

    def test_explicit_provider_with_key(self, env, tmp_path):
        s = _resolve(["process", "--web-provider", "brave"], {**env, "BRAVE_API_KEY": "sekrit-123"}, tmp_path)
        assert s.web.provider_resolved == "brave"
        assert _resolve(["process", "--web-provider", "duckduckgo"], env, tmp_path).web.provider_resolved == "duckduckgo"

    def test_web_explicitly_on_with_provider_none_is_an_error(self, env, tmp_path):
        with pytest.raises(SettingsError, match="web lookups were requested"):
            _resolve(["process", "--web", "--web-provider", "none"], env, tmp_path)
        with pytest.raises(SettingsError, match="web lookups were requested"):
            _resolve(["process", "-v", "--web-provider", "none"], env, tmp_path)

    def test_web_on_by_default_with_provider_none_warns(self, env, tmp_path):
        _write(tmp_path / "bourbon.toml", '[web]\nprovider = "none"\n')
        s = _resolve(["process"], env, tmp_path)
        assert s.web.enabled
        assert s.web.provider_resolved == "none"
        assert s.web.provider_reason == NO_PROVIDER_REASON

    def test_web_off_resolves_none_without_errors(self, env, tmp_path):
        s = _resolve(["process", "--no-web", "--web-provider", "brave"], env, tmp_path)
        assert s.web.provider_resolved == "none"

    def test_feedback_never_resolves_a_provider(self, env, tmp_path):
        s = _resolve(["feedback"], {**env, "BOURBON_WEB_PROVIDER": "brave"}, tmp_path)
        assert s.web.provider_resolved == "none"

    def test_key_values_never_stored_or_shown(self, env, tmp_path):
        secret = "sekrit-brave-123"
        s = _resolve(["process"], {**env, "BRAVE_API_KEY": secret, "OLLAMA_API_KEY": "sekrit-ollama"}, tmp_path)
        assert s.api_keys["BRAVE_API_KEY"] == "set"
        for text in (repr(s), describe(s), describe(s, as_json=True)):
            assert secret not in text
            assert "sekrit-ollama" not in text
        assert "BRAVE_API_KEY set" in describe(s)
        assert json.loads(describe(s, as_json=True))["api_keys"]["OLLAMA_API_KEY"] == "set"

    def test_url_credentials_never_shown(self, env, tmp_path):
        """A password or token inside a URL is redacted in config show and the run manifest data."""
        from lib.settings import redact_url, settings_to_dict

        s = _resolve(
            ["process", "-u", "http://alice:pw-sekrit@localhost:11434"],
            {**env, "BOURBON_SEARXNG_URL": "https://search.example.org/?token=tok-sekrit"},
            tmp_path,
        )
        assert s.server.ollama_url == "http://alice:pw-sekrit@localhost:11434"  # still used as given
        for text in (describe(s), describe(s, as_json=True), json.dumps(settings_to_dict(s))):
            assert "pw-sekrit" not in text and "tok-sekrit" not in text and "alice" not in text
        values = settings_to_dict(s)["values"]
        assert values["server.ollama_url"] == "http://***@localhost:11434"
        assert values["web.searxng_url"] == "https://search.example.org/?***"
        assert redact_url("http://localhost:11434") == "http://localhost:11434"
        assert redact_url("") == ""


class TestOutput:
    def test_template_parses_and_equals_defaults(self):
        data = tomllib.loads(render_template())
        assert data == template_defaults()
        file_keys = {spec.key for spec in REGISTRY if spec.file}
        assert {f"{section}.{key}" for section, table in data.items() for key in table} == file_keys

    def test_template_has_a_comment_per_key(self):
        lines = render_template().splitlines()
        for i, line in enumerate(lines):
            if " = " in line and not line.startswith("#"):
                assert lines[i - 1].startswith("# "), line

    def test_template_loads_back_as_a_config_file(self, env, tmp_path):
        conf = _write(tmp_path / "t.toml", render_template())
        s = _resolve(["process", "--config", str(conf)], env, tmp_path)
        assert s.concurrency == 2 and s.web.provider_resolved == "duckduckgo"

    def test_example_file_is_the_template(self):
        assert (REPO_ROOT / "bourbon.example.toml").read_text(encoding="utf-8") == render_template()

    def test_describe_lists_files_values_and_sources(self, env, tmp_path):
        _write(tmp_path / "bourbon.toml", "[processing]\nconcurrency = 4\n")
        s = _resolve(["config", "show"], {**env, "BOURBON_DEBUG": "1"}, tmp_path)
        text = describe(s)
        assert f"{tmp_path / 'bourbon.toml'} (loaded)" in text
        assert "(not found)" in text  # the user file
        assert "processing.concurrency" in text and f"(file:{tmp_path / 'bourbon.toml'})" in text
        data = json.loads(describe(s, as_json=True))
        assert data["values"]["processing.concurrency"] == 4
        assert data["sources"]["logging.debug"] == "env:BOURBON_DEBUG"
        assert data["config_files_loaded"] == [str(tmp_path / "bourbon.toml")]


class TestCliOnlyFields:
    def test_feedback_fields(self, env, tmp_path):
        s = _resolve(
            ["feedback", "--batch", "b1", "--corrections", "c.json", "--approve", "IMG_1", "--approve", "IMG_2"],
            env, tmp_path,
        )
        assert (s.command, s.batch, s.corrections, s.approve) == ("feedback", "b1", tmp_path / "c.json", ("IMG_1", "IMG_2"))

    def test_retry_and_config_fields(self, env, tmp_path):
        assert _resolve(["retry", "--report", "r.json"], env, tmp_path).report == tmp_path / "r.json"
        s = _resolve(["config", "show", "--json"], env, tmp_path)
        assert (s.config_action, s.json) == ("show", True)

    def test_namespace_without_attributes_resolves_defaults(self, env, tmp_path):
        s = resolve_settings(argparse.Namespace(command="process"), env, cwd=tmp_path)
        assert s.batch_size == 5
