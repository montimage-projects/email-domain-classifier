"""LLM configuration loading from .env file."""

import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv


class LLMProvider(str, Enum):
    """Supported LLM providers."""

    GOOGLE = "google"
    MISTRAL = "mistral"
    OLLAMA = "ollama"
    GROQ = "groq"
    OPENROUTER = "openrouter"
    TYPESAFE = "typesafe"


# Default models per provider
DEFAULT_MODELS: dict[LLMProvider, str] = {
    LLMProvider.GOOGLE: "gemini-2.0-flash",
    LLMProvider.MISTRAL: "mistral-large-latest",
    LLMProvider.OLLAMA: "llama3.2",
    LLMProvider.GROQ: "llama-3.3-70b-versatile",
    LLMProvider.OPENROUTER: "",  # Must be specified explicitly
    LLMProvider.TYPESAFE: "jev-latest",
}

# Required environment variables per provider
PROVIDER_API_KEYS: dict[LLMProvider, str] = {
    LLMProvider.GOOGLE: "GOOGLE_API_KEY",
    LLMProvider.MISTRAL: "MISTRAL_API_KEY",
    LLMProvider.GROQ: "GROQ_API_KEY",
    LLMProvider.OPENROUTER: "OPENROUTER_API_KEY",
    LLMProvider.TYPESAFE: "TYPESAFE_API_KEY",
    # Ollama doesn't require an API key
}

# Package requirements per provider
PROVIDER_PACKAGES: dict[LLMProvider, str] = {
    LLMProvider.GOOGLE: "langchain-google-genai",
    LLMProvider.MISTRAL: "langchain-mistralai",
    LLMProvider.OLLAMA: "langchain-ollama",
    LLMProvider.GROQ: "langchain-groq",
    LLMProvider.OPENROUTER: "langchain-openai",
    LLMProvider.TYPESAFE: "typesafe-sdk",
}


# Minimum Method 3 confidence the hybrid workflow accepts when the classic
# methods disagree (issue #18). An answer below it is not used: the email goes
# to the classic weighted fallback instead.
#
# PROVISIONAL: this value was not measured. Issue #19 chooses the final cutoff
# from an evaluation on labelled data. Until then, override it with the
# LLM_CONFIDENCE_CUTOFF environment variable. 0.0 accepts every answer that is
# not a failed call.
DEFAULT_LLM_CONFIDENCE_CUTOFF = 0.5

# Environment variable that sets LLMConfig.llm_confidence_cutoff.
LLM_CONFIDENCE_CUTOFF_ENV = "LLM_CONFIDENCE_CUTOFF"


class LLMConfigError(Exception):
    """Error in LLM configuration."""

    pass


@dataclass
class LLMConfig:
    """Configuration for LLM-based classification."""

    # Provider settings
    provider: LLMProvider
    model: str
    api_key: Optional[str] = field(default=None, repr=False)

    # Generation settings
    temperature: float = 0.0
    max_tokens: int = 1024

    # Reliability settings
    timeout: int = 30
    retry_count: int = 2

    # Ollama-specific
    ollama_base_url: str = "http://localhost:11434"

    # Weight settings for combining methods
    llm_weight: float = 0.40
    keyword_weight: float = 0.35
    structural_weight: float = 0.25

    # Hybrid workflow gate: minimum Method 3 confidence accepted (0.0 to 1.0).
    # Provisional default, see DEFAULT_LLM_CONFIDENCE_CUTOFF.
    llm_confidence_cutoff: float = DEFAULT_LLM_CONFIDENCE_CUTOFF

    def __post_init__(self) -> None:
        """Validate and normalize configuration after initialization."""
        self._normalize_weights()
        self._validate()

    def _normalize_weights(self) -> None:
        """Normalize weights to sum to 1.0."""
        total = self.llm_weight + self.keyword_weight + self.structural_weight
        if total > 0 and abs(total - 1.0) > 0.001:
            self.llm_weight = self.llm_weight / total
            self.keyword_weight = self.keyword_weight / total
            self.structural_weight = self.structural_weight / total

    def _validate(self) -> None:
        """Validate configuration."""
        # Check API key for cloud providers
        if self.provider != LLMProvider.OLLAMA:
            if not self.api_key:
                key_name = PROVIDER_API_KEYS.get(self.provider, "API_KEY")
                raise LLMConfigError(
                    f"Missing API key for {self.provider.value}. "
                    f"Set {key_name} in your .env file."
                )

        # Check model for OpenRouter
        if self.provider == LLMProvider.OPENROUTER and not self.model:
            raise LLMConfigError(
                "OpenRouter requires an explicit model name. "
                "Set LLM_MODEL in your .env file."
            )

        # Validate numeric ranges
        if not 0.0 <= self.temperature <= 2.0:
            raise LLMConfigError(
                f"Invalid temperature: {self.temperature}. Must be between 0.0 and 2.0."
            )

        if self.max_tokens < 1:
            raise LLMConfigError(
                f"Invalid max_tokens: {self.max_tokens}. Must be positive."
            )

        if self.timeout < 1:
            raise LLMConfigError(f"Invalid timeout: {self.timeout}. Must be positive.")

        if self.retry_count < 0:
            raise LLMConfigError(
                f"Invalid retry_count: {self.retry_count}. Must be non-negative."
            )

        validate_confidence_cutoff(self.llm_confidence_cutoff)

    @classmethod
    def from_env(cls, env_file: Optional[Path] = None) -> "LLMConfig":
        """Load configuration from .env file.

        Args:
            env_file: Optional path to .env file. If not provided,
                     searches for .env in current directory and parents.

        Returns:
            LLMConfig instance with loaded settings.

        Raises:
            LLMConfigError: If required configuration is missing or invalid.
        """
        # Load .env file
        if env_file:
            if not env_file.exists():
                raise LLMConfigError(f".env file not found: {env_file}")
            load_dotenv(env_file)
        else:
            load_dotenv()

        # Parse provider
        provider_str = os.getenv("LLM_PROVIDER", "ollama").lower()
        try:
            provider = LLMProvider(provider_str)
        except ValueError:
            valid = ", ".join(p.value for p in LLMProvider)
            raise LLMConfigError(
                f"Invalid LLM_PROVIDER: {provider_str}. Must be one of: {valid}"
            )

        # Get model (use default if not specified)
        model = os.getenv("LLM_MODEL", "").strip()
        if not model:
            model = DEFAULT_MODELS.get(provider, "")

        # Get API key for the provider
        api_key: Optional[str] = None
        if provider in PROVIDER_API_KEYS:
            key_name = PROVIDER_API_KEYS[provider]
            api_key = os.getenv(key_name, "").strip() or None

        # Parse numeric settings with defaults
        temperature = _parse_float(os.getenv("LLM_TEMPERATURE", ""), 0.0)
        max_tokens = _parse_int(os.getenv("LLM_MAX_TOKENS", ""), 1024)
        timeout = _parse_int(os.getenv("LLM_TIMEOUT", ""), 30)
        retry_count = _parse_int(os.getenv("LLM_RETRY_COUNT", ""), 2)

        # Parse Ollama settings
        ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").strip()

        # Parse weight settings
        llm_weight = _parse_float(os.getenv("LLM_WEIGHT", ""), 0.40)
        keyword_weight = _parse_float(os.getenv("KEYWORD_WEIGHT", ""), 0.35)
        structural_weight = _parse_float(os.getenv("STRUCTURAL_WEIGHT", ""), 0.25)

        # Parse the hybrid confidence gate
        llm_confidence_cutoff = parse_confidence_cutoff(
            os.getenv(LLM_CONFIDENCE_CUTOFF_ENV, "")
        )

        return cls(
            provider=provider,
            model=model,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            retry_count=retry_count,
            ollama_base_url=ollama_base_url,
            llm_weight=llm_weight,
            keyword_weight=keyword_weight,
            structural_weight=structural_weight,
            llm_confidence_cutoff=llm_confidence_cutoff,
        )

    def get_package_name(self) -> str:
        """Get the package name required for this provider."""
        return PROVIDER_PACKAGES.get(self.provider, "langchain")

    def get_install_command(self) -> str:
        """Get pip install command for this provider."""
        extra = self.provider.value
        return f"pip install email-domain-classifier[{extra}]"


def validate_confidence_cutoff(value: float) -> float:
    """Check that a confidence cutoff is a number from 0.0 to 1.0.

    Args:
        value: Cutoff to check.

    Returns:
        The cutoff, unchanged.

    Raises:
        LLMConfigError: If the cutoff is outside 0.0 to 1.0 or is NaN.
    """
    # Written as a chained comparison so NaN fails it.
    if not 0.0 <= value <= 1.0:
        raise LLMConfigError(
            f"Invalid llm_confidence_cutoff: {value}. Must be between 0.0 and 1.0."
        )
    return value


def parse_confidence_cutoff(value: str) -> float:
    """Parse an LLM confidence cutoff from text.

    Unlike the other numeric settings, an invalid value is an error instead of
    being replaced by the default, because a silently ignored cutoff would
    change which LLM answers the hybrid workflow accepts.

    Args:
        value: Text such as "0.7". Empty or blank text gives the default.

    Returns:
        The cutoff, from 0.0 to 1.0.

    Raises:
        LLMConfigError: If the text is not a number from 0.0 to 1.0.
    """
    value = value.strip()
    if not value:
        return DEFAULT_LLM_CONFIDENCE_CUTOFF
    try:
        cutoff = float(value)
    except ValueError:
        raise LLMConfigError(
            f"Invalid {LLM_CONFIDENCE_CUTOFF_ENV}: {value!r}. "
            "Must be a number between 0.0 and 1.0."
        ) from None
    return validate_confidence_cutoff(cutoff)


def _parse_float(value: str, default: float) -> float:
    """Parse float value with default."""
    if not value:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _parse_int(value: str, default: int) -> int:
    """Parse int value with default."""
    if not value:
        return default
    try:
        return int(value)
    except ValueError:
        return default
