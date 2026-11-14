"""Unit tests for Config class.

Tests configuration loading, priority order, and validation.
"""

import os
import tempfile
from pathlib import Path

import pytest

from lib.config import Config, get_config


class TestConfigInitialization:
    """Test Config class initialization and defaults."""

    def test_config_default_values(self):
        """Test Config has correct default values."""
        config = Config()

        assert config.image_folder == "."
        assert config.batch_size == "5"
        assert config.lookahead == 3
        assert config.concurrency == 3
        assert config.model == "qwen3-vl:32b"
        assert config.interactive is False
        assert config.include_vendor_images is False
        assert config.ollama_url == "http://localhost:11434"
        assert config.output_dir == "processed/"
        assert config.allow_retry is True
        assert config.debug is False

    def test_config_with_cli_args_override(self):
        """Test CLI args override defaults."""
        config = Config()
        config.set_cli_arg("model", "qwen2-vl:7b")
        config.set_cli_arg("debug", True)

        assert config.model == "qwen2-vl:7b"
        assert config.debug is True
        assert config.batch_size == "5"  # Other defaults unchanged


class TestConfigFileLoading:
    """Test configuration loading from .specify_config.toml file."""

    def test_config_load_from_toml_file(self):
        """Test config loads values from TOML file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create temp TOML config file (flat structure, as expected by Config)
            config_path = Path(tmpdir) / ".specify_config.toml"
            config_path.write_text(
                """
model = "qwen2-vl:7b"
batch_size = "10"
lookahead = 5
concurrency = 4
ollama_url = "http://localhost:11435"
output_dir = "output_test/"
debug = true
"""
            )

            # Change to temp dir and create config
            old_cwd = Path.cwd()
            os.chdir(tmpdir)
            try:
                config = Config()
                assert config.model == "qwen2-vl:7b"
                assert config.batch_size == "10"
                assert config.lookahead == 5
                assert config.concurrency == 4
                assert config.ollama_url == "http://localhost:11435"
                assert config.output_dir == "output_test/"
                assert config.debug is True
            finally:
                os.chdir(old_cwd)

    def test_config_file_not_found_uses_defaults(self):
        """Test missing config file uses defaults."""
        with tempfile.TemporaryDirectory() as tmpdir:
            old_cwd = Path.cwd()
            os.chdir(tmpdir)
            try:
                config = Config()
                assert config.model == "qwen3-vl:32b"  # Default
            finally:
                os.chdir(old_cwd)


class TestConfigEnvironmentVariables:
    """Test configuration loading from environment variables."""

    def test_env_var_model_override(self):
        """Test BOURBON_MODEL environment variable overrides config."""
        os.environ["BOURBON_MODEL"] = "test-model:latest"
        try:
            config = Config()
            assert config.model == "test-model:latest"
        finally:
            del os.environ["BOURBON_MODEL"]

    def test_env_var_batch_size_override(self):
        """Test BOURBON_BATCH_SIZE environment variable."""
        os.environ["BOURBON_BATCH_SIZE"] = "7"
        try:
            config = Config()
            assert config.batch_size == "7"
        finally:
            del os.environ["BOURBON_BATCH_SIZE"]

    def test_env_var_bool_parsing(self):
        """Test boolean env var parsing."""
        test_cases = [
            ("true", True),
            ("TRUE", True),
            ("1", True),
            ("yes", True),
            ("on", True),
            ("false", False),
            ("FALSE", False),
            ("0", False),
            ("no", False),
            ("off", False),
        ]

        for env_value, expected in test_cases:
            os.environ["BOURBON_DEBUG"] = env_value
            try:
                config = Config()
                assert config.debug is expected, f"Failed for {env_value}"
            finally:
                del os.environ["BOURBON_DEBUG"]

    def test_env_var_invalid_bool_uses_false(self):
        """Test invalid boolean env var doesn't crash."""
        os.environ["BOURBON_DEBUG"] = "invalid"
        try:
            config = Config()
            # Should log warning and not crash
            assert hasattr(config, "debug")
        finally:
            del os.environ["BOURBON_DEBUG"]


class TestConfigPriority:
    """Test configuration priority order: CLI > ENV > FILE > DEFAULT."""

    def test_cli_args_override_env_vars(self):
        """Test CLI args have highest priority."""
        os.environ["BOURBON_MODEL"] = "env-model"
        os.environ["BOURBON_DEBUG"] = "true"
        try:
            config = Config()
            config.set_cli_arg("model", "cli-model")

            assert config.model == "cli-model"  # CLI wins
            assert config.debug is True  # Env still applies
        finally:
            del os.environ["BOURBON_MODEL"]
            del os.environ["BOURBON_DEBUG"]

    def test_env_vars_override_file_config(self):
        """Test env vars override file config."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / ".specify_config.toml"
            config_path.write_text('model = "file-model"')

            os.environ["BOURBON_MODEL"] = "env-model"
            old_cwd = Path.cwd()
            os.chdir(tmpdir)
            try:
                config = Config()
                assert config.model == "env-model"  # Env wins over file
            finally:
                os.chdir(old_cwd)
                del os.environ["BOURBON_MODEL"]

    def test_file_config_overrides_defaults(self):
        """Test file config overrides defaults."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / ".specify_config.toml"
            config_path.write_text('model = "file-model"')

            old_cwd = Path.cwd()
            os.chdir(tmpdir)
            try:
                config = Config()
                assert config.model == "file-model"  # File wins over default
            finally:
                os.chdir(old_cwd)


class TestConfigHelperMethods:
    """Test Config type-safe accessor methods."""

    def test_get_bool(self):
        """Test get_bool method."""
        config = Config()
        assert config.get_bool("debug") is False
        assert config.get_bool("allow_retry") is True
        assert config.get_bool("interactive") is False
        # Test default parameter when key doesn't exist
        assert config.get_bool("nonexistent_key", True) is True

    def test_get_bool_invalid_key_raises(self):
        """Test get_bool with invalid key raises AttributeError."""
        config = Config()
        with pytest.raises(AttributeError):
            config.get_bool("nonexistent_key")

    def test_get_int(self):
        """Test get_int method."""
        config = Config()
        assert config.get_int("lookahead") == 3
        assert config.get_int("concurrency") == 3

    def test_get_int_string_parses(self):
        """Test get_int parses string to int."""
        config = Config()
        assert config.get_int("batch_size") == 5  # "5" -> 5

    def test_get_int_invalid_key_raises(self):
        """Test get_int with invalid key raises AttributeError."""
        config = Config()
        with pytest.raises(AttributeError):
            config.get_int("nonexistent_key")

    def test_get_str(self):
        """Test get_str method."""
        config = Config()
        assert config.get_str("model") == "qwen3-vl:32b"
        assert config.get_str("image_folder") == "."

    def test_get_str_int_to_string(self):
        """Test get_str converts int to string."""
        config = Config()
        result = config.get_str("lookahead")
        assert isinstance(result, str)
        assert result == "3"

    def test_get_str_invalid_key_raises(self):
        """Test get_str with invalid key raises AttributeError."""
        config = Config()
        with pytest.raises(AttributeError):
            config.get_str("nonexistent_key")

    def test_get_url_valid(self):
        """Test get_url with valid URL."""
        config = Config()
        assert config.get_url() == "http://localhost:11434"
        assert config.get_url("ollama_url") == "http://localhost:11434"

    def test_get_url_invalid_raises(self):
        """Test get_url with invalid URL raises ValueError."""
        config = Config()
        config.ollama_url = "not-a-url"
        with pytest.raises(ValueError, match="Invalid URL"):
            config.get_url()

    def test_get_url_https_allowed(self):
        """Test get_url accepts HTTPS URLs."""
        config = Config()
        config.ollama_url = "https://example.com"
        assert config.get_url() == "https://example.com"


class TestConfigSetProxyMethods:
    """Test Config CLI argument setter methods."""

    def test_set_cli_arg_single(self):
        """Test set_cli_arg sets single value."""
        config = Config()
        config.set_cli_arg("debug", True)

        assert config.debug is True
        assert config._cli_args == {"debug": True}

    def test_set_cli_args_multiple(self):
        """Test set_cli_args sets multiple values."""
        config = Config()
        config.set_cli_args({
            "debug": True,
            "model": "test-model",
            "lookahead": 5,
        })

        assert config.debug is True
        assert config.model == "test-model"
        assert config.lookahead == 5
        assert set(config._cli_args.keys()) == {"debug", "model", "lookahead"}

    def test_cli_args_apply_immediately(self):
        """Test CLI args apply immediately when set."""
        config = Config()
        original_debug = config.debug
        original_model = config.model

        config.set_cli_arg("debug", True)

        assert config._cli_args["debug"] is True
        assert config.debug is True  # Applied immediately
        assert config.model == original_model  # Other values unchanged


class TestConfigValidation:
    """Test Config validation rules."""

    def test_validate_default_config(self):
        """Test default config validates successfully."""
        config = Config()
        errors = config.validate()
        assert errors == []

    def test_validate_batch_size_valid_integer(self):
        """Test valid batch_size integer."""
        config = Config()
        config.batch_size = "10"
        errors = config.validate()
        assert errors == []

    def test_validate_batch_size_valid_all(self):
        """Test batch_size='all' is valid."""
        config = Config()
        config.batch_size = "all"
        errors = config.validate()
        assert errors == []

    def test_validate_batch_size_negative_fails(self):
        """Test negative batch_size fails validation."""
        config = Config()
        config.batch_size = "-1"
        with pytest.raises(ValueError, match="batch_size must be >= 1"):
            config.validate()

    def test_validate_batch_size_invalid_string_fails(self):
        """Test invalid batch_size string fails validation."""
        config = Config()
        config.batch_size = "invalid"
        with pytest.raises(ValueError, match="batch_size must be 'all' or a positive integer"):
            config.validate()

    def test_validate_lookahead_valid_range(self):
        """Test valid lookahead range (0-10)."""
        config = Config()
        for value in [0, 5, 10]:
            config.lookahead = value
            errors = config.validate()
            assert errors == []

    def test_validate_lookahead_negative_fails(self):
        """Test negative lookahead fails validation."""
        config = Config()
        config.lookahead = -1
        with pytest.raises(ValueError, match="lookahead must be between 0 and 10"):
            config.validate()

    def test_validate_lookahead_above_10_fails(self):
        """Test lookahead > 10 fails validation."""
        config = Config()
        config.lookahead = 11
        with pytest.raises(ValueError, match="lookahead must be between 0 and 10"):
            config.validate()

    def test_validate_concurrency_valid_range(self):
        """Test valid concurrency range (1-10)."""
        config = Config()
        for value in [1, 5, 10]:
            config.concurrency = value
            errors = config.validate()
            assert errors == []

    def test_validate_concurrency_zero_fails(self):
        """Test concurrency=0 fails validation."""
        config = Config()
        config.concurrency = 0
        with pytest.raises(ValueError, match="concurrency must be between 1 and 10"):
            config.validate()

    def test_validate_concurrency_above_10_fails(self):
        """Test concurrency > 10 fails validation."""
        config = Config()
        config.concurrency = 11
        with pytest.raises(ValueError, match="concurrency must be between 1 and 10"):
            config.validate()

    def test_validate_ollama_url_invalid_fails(self):
        """Test invalid Ollama URL fails validation."""
        config = Config()
        config.ollama_url = "not-a-valid-url"
        with pytest.raises(ValueError, match="Invalid URL"):
            config.validate()

    def test_validate_multiple_errors(self):
        """Test validation returns multiple errors."""
        config = Config()
        config.batch_size = "-1"
        config.lookahead = 15
        config.concurrency = 0

        with pytest.raises(ValueError) as exc_info:
            config.validate()

        error_msg = str(exc_info.value)
        assert "batch_size must be >= 1" in error_msg
        assert "lookahead must be between 0 and 10" in error_msg
        assert "concurrency must be between 1 and 10" in error_msg


class TestGetConfigConvenience:
    """Test get_config() convenience function."""

    def test_get_config_creates_valid_config(self):
        """Test get_config returns validated Config."""
        config = get_config()
        assert isinstance(config, Config)
        # Should not raise validation exception

    def test_get_config_with_cli_args(self):
        """Test get_config with CLI argument overrides."""
        config = get_config(model="test-model", debug=True)

        assert config.model == "test-model"
        assert config.debug is True

    def test_get_config_invalid_raises(self):
        """Test get_config with invalid config raises ValueError."""
        with pytest.raises(ValueError):
            get_config(batch_size="-1")
