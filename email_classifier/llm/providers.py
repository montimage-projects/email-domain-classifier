"""LLM provider factory for creating provider-specific LLM instances."""

import sys
from typing import Any, Optional

from .config import LLMConfig, LLMConfigError, LLMProvider


class ProviderNotInstalledError(Exception):
    """Raised when a provider's package is not installed."""

    def __init__(self, provider: LLMProvider, package: str, install_cmd: str) -> None:
        self.provider = provider
        self.package = package
        self.install_cmd = install_cmd
        super().__init__(
            f"Provider '{provider.value}' requires package '{package}'. "
            f"Install with: {install_cmd}"
        )


def create_llm(config: LLMConfig) -> Any:
    """Create an LLM instance based on the configuration.

    Args:
        config: LLM configuration specifying provider and settings.

    Returns:
        A LangChain-compatible LLM instance.

    Raises:
        ProviderNotInstalledError: If the provider's package is not installed.
        LLMConfigError: If configuration is invalid.
    """
    if config.provider == LLMProvider.GOOGLE:
        return _create_google_llm(config)
    elif config.provider == LLMProvider.MISTRAL:
        return _create_mistral_llm(config)
    elif config.provider == LLMProvider.OLLAMA:
        return _create_ollama_llm(config)
    elif config.provider == LLMProvider.GROQ:
        return _create_groq_llm(config)
    elif config.provider == LLMProvider.OPENROUTER:
        return _create_openrouter_llm(config)
    elif config.provider == LLMProvider.TYPESAFE:
        raise LLMConfigError(
            "TypeSafe is not a LangChain provider. "
            "Use email_classifier.llm.create_classifier() instead."
        )
    else:
        raise LLMConfigError(f"Unknown provider: {config.provider}")


def _create_google_llm(config: LLMConfig) -> Any:
    """Create Google Gemini LLM instance."""
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
    except ImportError:
        raise ProviderNotInstalledError(
            provider=LLMProvider.GOOGLE,
            package="langchain-google-genai",
            install_cmd=config.get_install_command(),
        )

    return ChatGoogleGenerativeAI(
        model=config.model,
        google_api_key=config.api_key,
        temperature=config.temperature,
        max_output_tokens=config.max_tokens,
        timeout=config.timeout,
    )


def _create_mistral_llm(config: LLMConfig) -> Any:
    """Create Mistral AI LLM instance."""
    try:
        from langchain_mistralai import ChatMistralAI
    except ImportError:
        raise ProviderNotInstalledError(
            provider=LLMProvider.MISTRAL,
            package="langchain-mistralai",
            install_cmd=config.get_install_command(),
        )

    return ChatMistralAI(
        model=config.model,
        api_key=config.api_key,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        timeout=config.timeout,
    )


def _create_ollama_llm(config: LLMConfig) -> Any:
    """Create Ollama LLM instance."""
    try:
        from langchain_ollama import ChatOllama
    except ImportError:
        raise ProviderNotInstalledError(
            provider=LLMProvider.OLLAMA,
            package="langchain-ollama",
            install_cmd=config.get_install_command(),
        )

    return ChatOllama(
        model=config.model,
        base_url=config.ollama_base_url,
        temperature=config.temperature,
        num_predict=config.max_tokens,
    )


def _create_groq_llm(config: LLMConfig) -> Any:
    """Create Groq LLM instance."""
    try:
        from langchain_groq import ChatGroq
    except ImportError:
        raise ProviderNotInstalledError(
            provider=LLMProvider.GROQ,
            package="langchain-groq",
            install_cmd=config.get_install_command(),
        )

    return ChatGroq(
        model=config.model,
        api_key=config.api_key,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        timeout=config.timeout,
    )


def _create_openrouter_llm(config: LLMConfig) -> Any:
    """Create OpenRouter LLM instance via OpenAI-compatible API."""
    try:
        from langchain_openai import ChatOpenAI
    except ImportError:
        raise ProviderNotInstalledError(
            provider=LLMProvider.OPENROUTER,
            package="langchain-openai",
            install_cmd=config.get_install_command(),
        )

    return ChatOpenAI(
        model=config.model,
        api_key=config.api_key,
        base_url="https://openrouter.ai/api/v1",
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        timeout=config.timeout,
    )


def check_provider_available(provider: LLMProvider) -> tuple[bool, Optional[str]]:
    """Check if a provider's package is installed.

    Args:
        provider: The provider to check.

    Returns:
        Tuple of (is_available, error_message).
        If available, error_message is None.
    """
    package_map = {
        LLMProvider.GOOGLE: "langchain_google_genai",
        LLMProvider.MISTRAL: "langchain_mistralai",
        LLMProvider.OLLAMA: "langchain_ollama",
        LLMProvider.GROQ: "langchain_groq",
        LLMProvider.OPENROUTER: "langchain_openai",
        LLMProvider.TYPESAFE: "typesafe_sdk",
    }

    package = package_map.get(provider)
    if not package:
        return False, f"Unknown provider: {provider}"

    if package in sys.modules:
        return True, None

    try:
        __import__(package)
        return True, None
    except ImportError:
        from .config import PROVIDER_PACKAGES

        pip_package = PROVIDER_PACKAGES.get(provider, package)
        return False, f"Package '{pip_package}' not installed"
