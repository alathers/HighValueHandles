"""Configuration management for Bourbon Image Processor.

Implements configuration priority: CLI args → environment variables →
.specify_config.toml → defaults.
"""

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

try:
    import tomllib  # Python 3.11+
except ImportError:
    import tomli as tomllib  # Fallback for Python < 3.11


@dataclass
class Config:
    """Configuration class for bourbon image processor.

    Loads configuration from multiple sources in priority order:
    1. Command-line arguments (set via properties)
    2. Environment variables
    3. .specify_config.toml file
    4. Default values
    """

    # Image processing settings
    image_folder: str = "."
    batch_size: str = "5"
    lookahead: int = 3
    concurrency: int = 3

    # AI/model settings
    model: str = "qwen3-vl:32b"
    interactive: bool = False
    include_vendor_images: bool = False

    # Ollama connection
    ollama_url: str = "http://localhost:11434"

    # Output settings
    output_dir: str = "processed/"
    allow_retry: bool = True

    # Debugging
    debug: bool = False

    # Private fields for CLI args override
    _cli_args: dict = field(default_factory=dict, repr=False)

    def __post_init__(self):
        """Load configuration from file and environment variables."""
        self._load_from_file()
        self._load_from_env()
        self._apply_cli_args()
        self._setup_logging()

    def _load_from_file(self):
        """Load configuration from .specify_config.toml if exists."""
        config_path = Path(".specify_config.toml")
        if config_path.exists():
            try:
                with open(config_path, "rb") as f:
                    config_data = tomllib.load(f)
                for key, value in config_data.items():
                    if hasattr(self, key):
                        # Don't override defaults with None from file
                        if value is not None:
                            object.__setattr__(self, key, value)
            except Exception as e:
                logging.warning(f"Failed to load config file: {e}")

    def _load_from_env(self):
        """Load configuration from environment variables.

        Environment variable names are uppercase versions of field names
        with BOURBON_ prefix (e.g., BOURBON_MODEL, BOURBON_BATCH_SIZE).
        """
        env_prefix = "BOURBON_"
        for field_name in self.__dataclass_fields__:
            if field_name.startswith("_"):
                continue
            env_name = f"{env_prefix}{field_name.upper()}"
            env_value = os.getenv(env_name)
            if env_value is not None:
                current_type = type(getattr(self, field_name))
                try:
                    if current_type == bool:
                        parsed_value = env_value.lower() in ("true", "1", "yes", "on")
                    else:
                        parsed_value = current_type(env_value)
                    object.__setattr__(self, field_name, parsed_value)
                except (ValueError, TypeError):
                    logging.warning(
                        f"Failed to parse env var {env_name}={env_value} as {current_type}"
                    )

    def _apply_cli_args(self):
        """Apply CLI argument overrides."""
        for key, value in self._cli_args.items():
            if hasattr(self, key):
                object.__setattr__(self, key, value)

    def _setup_logging(self):
        """Configure logging based on debug flag."""
        logging.basicConfig(
            level=logging.DEBUG if self.debug else logging.INFO,
            format="%(levelname)s: %(message)s",
        )

    # Helper methods for type-safe config access

    def get_bool(self, key: str, default: Optional[bool] = None) -> bool:
        """Get boolean configuration value.

        Args:
            key: Configuration key name.
            default: Default value if key not found.

        Returns:
            Boolean value.

        Raises:
            AttributeError: If key doesn't exist and no default provided.
        """
        value = getattr(self, key, default)
        if value is None:
            raise AttributeError(f"Configuration key '{key}' not found and no default provided")
        if not isinstance(value, bool):
            return bool(value)
        return value

    def get_int(self, key: str, default: Optional[int] = None) -> int:
        """Get integer configuration value.

        Args:
            key: Configuration key name.
            default: Default value if key not found.

        Returns:
            Integer value.

        Raises:
            AttributeError: If key doesn't exist and no default provided.
            ValueError: If value cannot be converted to integer.
        """
        value = getattr(self, key, default)
        if value is None:
            raise AttributeError(f"Configuration key '{key}' not found and no default provided")
        return int(value)

    def get_str(self, key: str, default: Optional[str] = None) -> str:
        """Get string configuration value.

        Args:
            key: Configuration key name.
            default: Default value if key not found.

        Returns:
            String value.

        Raises:
            AttributeError: If key doesn't exist and no default provided.
        """
        value = getattr(self, key, default)
        if value is None:
            raise AttributeError(f"Configuration key '{key}' not found and no default provided")
        return str(value)

    def get_url(self, key: str = "ollama_url") -> str:
        """Get URL configuration value with validation.

        Args:
            key: Configuration key name (default: ollama_url).

        Returns:
            URL string.

        Raises:
            ValueError: If URL is not valid HTTP/HTTPS URL.
            AttributeError: If key doesn't exist.
        """
        url = self.get_str(key)
        if not url.startswith(("http://", "https://")):
            raise ValueError(f"Invalid URL: {url}. Must start with http:// or https://")
        return url

    # CLI argument setters

    def set_cli_arg(self, key: str, value: Any) -> None:
        """Set CLI argument override.

        Args:
            key: Configuration key name.
            value: Value to set.
        """
        self._cli_args[key] = value
        self._apply_cli_args()

    def set_cli_args(self, args: dict) -> None:
        """Set multiple CLI argument overrides.

        Args:
            args: Dictionary of config key to value mappings.
        """
        self._cli_args.update(args)
        self._apply_cli_args()

    # Validation

    def validate(self) -> list[str]:
        """Validate configuration values.

        Returns:
            List of validation error messages (empty if valid).

        Raises:
            ValueError: If any validation rule fails.
        """
        errors = []

        # Validate batch_size
        if self.batch_size != "all":
            try:
                batch_int = int(self.batch_size)
                if batch_int < 1:
                    errors.append("batch_size must be >= 1")
            except ValueError:
                errors.append("batch_size must be 'all' or a positive integer")

        # Validate lookahead
        if not 0 <= self.lookahead <= 10:
            errors.append("lookahead must be between 0 and 10")

        # Validate concurrency
        if not 1 <= self.concurrency <= 10:
            errors.append("concurrency must be between 1 and 10")

        # Validate Ollama URL
        try:
            self.get_url("ollama_url")
        except ValueError as e:
            errors.append(str(e))

        if errors:
            raise ValueError("Configuration validation failed:\n" + "\n".join(f"  - {e}" for e in errors))

        return []


def get_config(**cli_args) -> Config:
    """Create and validate configuration object.

    Args:
        **cli_args: CLI argument overrides.

    Returns:
        Validated Config object.

    Raises:
        ValueError: If configuration validation fails.
    """
    config = Config()
    if cli_args:
        config.set_cli_args(cli_args)
    config.validate()
    return config
