"""LLM-based email classification module.

This module provides optional LLM classification as Method 3,
complementing the existing keyword taxonomy and structural template methods.

Usage:
    from email_classifier.llm import LLMClassifier, LLMConfig

    config = LLMConfig.from_env()
    classifier = LLMClassifier(config)
    result = classifier.classify(email_data)

Set ``LLM_PROVIDER=typesafe`` (with ``TYPESAFE_API_KEY``) to use the TypeSafe
classifier instead; ``create_classifier(config)`` returns the right class.
"""

from .agent import LLMClassifier
from .config import LLMConfig
from .factory import Method3Classifier, create_classifier
from .schemas import DomainClassification, LLMClassificationResult
from .typesafe_classifier import TypeSafeClassifier

__all__ = [
    "LLMClassifier",
    "LLMConfig",
    "Method3Classifier",
    "TypeSafeClassifier",
    "create_classifier",
    "DomainClassification",
    "LLMClassificationResult",
]
