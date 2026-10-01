"""Factory that picks the Method 3 classifier for an LLM configuration."""

from typing import Union

from .agent import LLMClassifier
from .config import LLMConfig, LLMProvider
from .typesafe_classifier import TypeSafeClassifier

Method3Classifier = Union[LLMClassifier, TypeSafeClassifier]


def create_classifier(config: LLMConfig) -> Method3Classifier:
    """Create the Method 3 classifier for a configuration.

    Args:
        config: LLM configuration.

    Returns:
        ``TypeSafeClassifier`` when ``config.provider`` is ``typesafe``,
        otherwise the LangChain-based ``LLMClassifier``.
    """
    if config.provider == LLMProvider.TYPESAFE:
        return TypeSafeClassifier(config)
    return LLMClassifier(config)
