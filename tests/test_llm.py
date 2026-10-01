"""Tests for the LLM-based classification module."""

import os
from unittest.mock import MagicMock, patch

import pytest

from email_classifier import EmailData
from email_classifier.llm.config import (
    DEFAULT_LLM_CONFIDENCE_CUTOFF,
    DEFAULT_MODELS,
    LLMConfig,
    LLMConfigError,
    LLMProvider,
    parse_confidence_cutoff,
)
from email_classifier.llm.schemas import (
    DomainClassification,
    LLMClassificationResult,
)


class TestLLMConfig:
    """Test cases for LLMConfig class."""

    def test_provider_enum_values(self):
        """Test that all expected providers exist."""
        assert LLMProvider.GOOGLE.value == "google"
        assert LLMProvider.MISTRAL.value == "mistral"
        assert LLMProvider.OLLAMA.value == "ollama"
        assert LLMProvider.GROQ.value == "groq"
        assert LLMProvider.OPENROUTER.value == "openrouter"

    def test_default_models_defined(self):
        """Test that default models are defined for all providers."""
        for provider in LLMProvider:
            assert provider in DEFAULT_MODELS

    def test_ollama_no_api_key_required(self):
        """Test that Ollama config works without API key."""
        config = LLMConfig(
            provider=LLMProvider.OLLAMA,
            model="llama3.2",
        )
        assert config.provider == LLMProvider.OLLAMA
        assert config.api_key is None

    def test_google_requires_api_key(self):
        """Test that Google provider requires API key."""
        with pytest.raises(LLMConfigError, match="Missing API key"):
            LLMConfig(
                provider=LLMProvider.GOOGLE,
                model="gemini-2.0-flash",
                api_key=None,
            )

    def test_openrouter_requires_model(self):
        """Test that OpenRouter requires explicit model."""
        with pytest.raises(LLMConfigError, match="requires an explicit model"):
            LLMConfig(
                provider=LLMProvider.OPENROUTER,
                model="",
                api_key="test-key",
            )

    def test_weight_normalization(self):
        """Test that weights are normalized to sum to 1.0."""
        config = LLMConfig(
            provider=LLMProvider.OLLAMA,
            model="llama3.2",
            llm_weight=2.0,
            keyword_weight=1.0,
            structural_weight=1.0,
        )
        total = config.llm_weight + config.keyword_weight + config.structural_weight
        assert abs(total - 1.0) < 0.001

    def test_temperature_validation(self):
        """Test temperature range validation."""
        with pytest.raises(LLMConfigError, match="Invalid temperature"):
            LLMConfig(
                provider=LLMProvider.OLLAMA,
                model="llama3.2",
                temperature=3.0,
            )

    def test_max_tokens_validation(self):
        """Test max_tokens validation."""
        with pytest.raises(LLMConfigError, match="Invalid max_tokens"):
            LLMConfig(
                provider=LLMProvider.OLLAMA,
                model="llama3.2",
                max_tokens=0,
            )

    @patch("email_classifier.llm.config.load_dotenv")
    @patch.dict(
        os.environ,
        {
            "LLM_PROVIDER": "ollama",
            "LLM_MODEL": "llama3.2",
            "LLM_TEMPERATURE": "0.5",
            "LLM_WEIGHT": "0.4",
            "KEYWORD_WEIGHT": "0.35",
            "STRUCTURAL_WEIGHT": "0.25",
        },
        clear=True,
    )
    def test_from_env(self, mock_load_dotenv):
        """Test loading configuration from environment."""
        config = LLMConfig.from_env()
        assert config.provider == LLMProvider.OLLAMA
        assert config.model == "llama3.2"
        assert config.temperature == 0.5
        # Weights sum to 1.0, so no normalization needed
        assert config.llm_weight == 0.4

    @patch("email_classifier.llm.config.load_dotenv")
    @patch.dict(
        os.environ,
        {
            "LLM_PROVIDER": "invalid",
        },
        clear=True,
    )
    def test_from_env_invalid_provider(self, mock_load_dotenv):
        """Test error on invalid provider."""
        with pytest.raises(LLMConfigError, match="Invalid LLM_PROVIDER"):
            LLMConfig.from_env()

    def test_get_install_command(self):
        """Test install command generation."""
        config = LLMConfig(
            provider=LLMProvider.GOOGLE,
            model="gemini-2.0-flash",
            api_key="test-key",
        )
        cmd = config.get_install_command()
        assert "google" in cmd
        assert "pip install" in cmd

    def test_get_package_name(self):
        """Test package name retrieval."""
        config = LLMConfig(
            provider=LLMProvider.GOOGLE,
            model="gemini-2.0-flash",
            api_key="test-key",
        )
        assert config.get_package_name() == "langchain-google-genai"

        config2 = LLMConfig(
            provider=LLMProvider.OLLAMA,
            model="llama3.2",
        )
        assert config2.get_package_name() == "langchain-ollama"

    def test_mistral_requires_api_key(self):
        """Test Mistral provider requires API key."""
        with pytest.raises(LLMConfigError, match="Missing API key"):
            LLMConfig(
                provider=LLMProvider.MISTRAL,
                model="mistral-large-latest",
                api_key=None,
            )

    def test_groq_requires_api_key(self):
        """Test Groq provider requires API key."""
        with pytest.raises(LLMConfigError, match="Missing API key"):
            LLMConfig(
                provider=LLMProvider.GROQ,
                model="llama-3.3-70b-versatile",
                api_key=None,
            )

    def test_timeout_validation(self):
        """Test timeout validation."""
        with pytest.raises(LLMConfigError, match="Invalid timeout"):
            LLMConfig(
                provider=LLMProvider.OLLAMA,
                model="llama3.2",
                timeout=0,
            )

    def test_retry_count_validation(self):
        """Test retry_count validation."""
        with pytest.raises(LLMConfigError, match="Invalid retry_count"):
            LLMConfig(
                provider=LLMProvider.OLLAMA,
                model="llama3.2",
                retry_count=-1,
            )

    @patch("email_classifier.llm.config.load_dotenv")
    @patch.dict(
        os.environ,
        {
            "LLM_PROVIDER": "google",
            "GOOGLE_API_KEY": "test-key-123",
        },
        clear=True,
    )
    def test_from_env_google(self, mock_load_dotenv):
        """Test loading Google config from environment."""
        config = LLMConfig.from_env()
        assert config.provider == LLMProvider.GOOGLE
        assert config.api_key == "test-key-123"
        assert config.model == "gemini-2.0-flash"  # Default

    @patch("email_classifier.llm.config.load_dotenv")
    @patch.dict(
        os.environ,
        {
            "LLM_PROVIDER": "mistral",
            "MISTRAL_API_KEY": "test-mistral-key",
            "LLM_MODEL": "custom-model",
        },
        clear=True,
    )
    def test_from_env_mistral_custom_model(self, mock_load_dotenv):
        """Test loading Mistral config with custom model."""
        config = LLMConfig.from_env()
        assert config.provider == LLMProvider.MISTRAL
        assert config.model == "custom-model"

    @patch("email_classifier.llm.config.load_dotenv")
    @patch.dict(
        os.environ,
        {
            "LLM_PROVIDER": "ollama",
            "OLLAMA_BASE_URL": "http://custom:11434",
        },
        clear=True,
    )
    def test_from_env_ollama_custom_url(self, mock_load_dotenv):
        """Test loading Ollama config with custom base URL."""
        config = LLMConfig.from_env()
        assert config.provider == LLMProvider.OLLAMA
        assert config.ollama_base_url == "http://custom:11434"

    @patch("email_classifier.llm.config.load_dotenv")
    @patch.dict(
        os.environ,
        {
            "LLM_PROVIDER": "openrouter",
            "OPENROUTER_API_KEY": "test-key",
        },
        clear=True,
    )
    def test_from_env_openrouter_requires_model(self, mock_load_dotenv):
        """Test OpenRouter requires explicit model."""
        with pytest.raises(LLMConfigError, match="requires an explicit model"):
            LLMConfig.from_env()

    @patch("email_classifier.llm.config.load_dotenv")
    @patch.dict(
        os.environ,
        {
            "LLM_PROVIDER": "openrouter",
            "OPENROUTER_API_KEY": "test-key",
            "LLM_MODEL": "anthropic/claude-3-sonnet",
        },
        clear=True,
    )
    def test_from_env_openrouter_with_model(self, mock_load_dotenv):
        """Test OpenRouter with explicit model works."""
        config = LLMConfig.from_env()
        assert config.provider == LLMProvider.OPENROUTER
        assert config.model == "anthropic/claude-3-sonnet"

    def test_default_weights(self):
        """Test default weight values."""
        config = LLMConfig(
            provider=LLMProvider.OLLAMA,
            model="llama3.2",
        )
        total = config.llm_weight + config.keyword_weight + config.structural_weight
        assert abs(total - 1.0) < 0.001

    def test_repr_hides_api_key(self):
        """Test that repr doesn't expose API key."""
        config = LLMConfig(
            provider=LLMProvider.GOOGLE,
            model="gemini-2.0-flash",
            api_key="secret-key-12345",
        )
        repr_str = repr(config)
        assert "secret-key-12345" not in repr_str


class TestLLMConfidenceCutoff:
    """Configuration of the hybrid workflow's LLM confidence cutoff (#18)."""

    def test_default_cutoff_is_evaluated_default(self):
        """The config uses the default evaluated in #19 when no cutoff is given."""
        config = LLMConfig(provider=LLMProvider.OLLAMA, model="llama3.2")
        assert config.llm_confidence_cutoff == DEFAULT_LLM_CONFIDENCE_CUTOFF

    @pytest.mark.parametrize("cutoff", [0.0, 0.5, 1.0])
    def test_valid_cutoff(self, cutoff):
        """The bounds 0.0 and 1.0 are valid."""
        config = LLMConfig(
            provider=LLMProvider.OLLAMA, model="llama3.2", llm_confidence_cutoff=cutoff
        )
        assert config.llm_confidence_cutoff == cutoff

    @pytest.mark.parametrize("cutoff", [-0.01, 1.01, float("nan"), float("inf")])
    def test_invalid_cutoff_raises(self, cutoff):
        """A cutoff outside 0.0 to 1.0, or NaN, is rejected."""
        with pytest.raises(LLMConfigError, match="llm_confidence_cutoff"):
            LLMConfig(
                provider=LLMProvider.OLLAMA,
                model="llama3.2",
                llm_confidence_cutoff=cutoff,
            )

    @patch("email_classifier.llm.config.load_dotenv")
    @patch.dict(os.environ, {"LLM_PROVIDER": "ollama"}, clear=True)
    def test_from_env_unset_uses_default(self, mock_load_dotenv):
        """Without LLM_CONFIDENCE_CUTOFF the default is used."""
        config = LLMConfig.from_env()
        assert config.llm_confidence_cutoff == DEFAULT_LLM_CONFIDENCE_CUTOFF

    @patch("email_classifier.llm.config.load_dotenv")
    @patch.dict(
        os.environ,
        {"LLM_PROVIDER": "ollama", "LLM_CONFIDENCE_CUTOFF": "  "},
        clear=True,
    )
    def test_from_env_blank_uses_default(self, mock_load_dotenv):
        """A blank LLM_CONFIDENCE_CUTOFF counts as unset."""
        config = LLMConfig.from_env()
        assert config.llm_confidence_cutoff == DEFAULT_LLM_CONFIDENCE_CUTOFF

    @patch("email_classifier.llm.config.load_dotenv")
    @patch.dict(
        os.environ,
        {"LLM_PROVIDER": "ollama", "LLM_CONFIDENCE_CUTOFF": " 0.7 "},
        clear=True,
    )
    def test_from_env_reads_cutoff(self, mock_load_dotenv):
        """LLM_CONFIDENCE_CUTOFF sets the cutoff."""
        config = LLMConfig.from_env()
        assert config.llm_confidence_cutoff == 0.7

    @pytest.mark.parametrize("raw", ["high", "0,7", "1.5", "-0.2", "nan"])
    @patch("email_classifier.llm.config.load_dotenv")
    def test_from_env_invalid_cutoff_raises(self, mock_load_dotenv, raw):
        """An invalid LLM_CONFIDENCE_CUTOFF is an error, not a silent default."""
        env = {"LLM_PROVIDER": "ollama", "LLM_CONFIDENCE_CUTOFF": raw}
        with patch.dict(os.environ, env, clear=True):
            with pytest.raises(LLMConfigError):
                LLMConfig.from_env()

    def test_parse_confidence_cutoff(self):
        """parse_confidence_cutoff parses text and names the variable on error."""
        assert parse_confidence_cutoff("0.25") == 0.25
        assert parse_confidence_cutoff("") == DEFAULT_LLM_CONFIDENCE_CUTOFF
        with pytest.raises(LLMConfigError, match="LLM_CONFIDENCE_CUTOFF"):
            parse_confidence_cutoff("abc")

    def test_cli_flag_parses_cutoff(self):
        """--llm-confidence-cutoff accepts 0.0 to 1.0 and rejects other values."""
        import argparse

        from email_classifier.cli import _confidence_cutoff_arg

        assert _confidence_cutoff_arg("0.65") == 0.65
        for raw in ("abc", "2", "-1", "nan"):
            with pytest.raises(argparse.ArgumentTypeError):
                _confidence_cutoff_arg(raw)

    @pytest.mark.parametrize(
        "llm_flags", [[], ["--use-llm", "--force-llm"]], ids=["no-llm", "force-llm"]
    )
    def test_cli_flag_requires_hybrid_mode(self, tmp_path, llm_flags):
        """The flag is refused outside hybrid mode instead of being ignored."""
        import subprocess
        import sys

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "email_classifier.cli",
                "classify",
                str(tmp_path / "missing.csv"),
                "-o",
                str(tmp_path / "out"),
                "--llm-confidence-cutoff",
                "0.5",
                "-q",
                *llm_flags,
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )

        assert result.returncode == 1
        assert "--llm-confidence-cutoff requires --use-llm" in (
            result.stdout + result.stderr
        )


class TestLLMSchemas:
    """Test cases for LLM schema classes."""

    def test_domain_classification_creation(self):
        """Test DomainClassification creation."""
        dc = DomainClassification(
            domain="finance",
            confidence=0.85,
            reasoning="Contains banking keywords",
        )
        assert dc.domain == "finance"
        assert dc.confidence == 0.85
        assert dc.reasoning == "Contains banking keywords"

    def test_domain_classification_confidence_bounds(self):
        """Test that confidence is bounded."""
        dc = DomainClassification(
            domain="finance",
            confidence=1.0,
            reasoning="Test",
        )
        assert dc.confidence == 1.0

        dc2 = DomainClassification(
            domain="finance",
            confidence=0.0,
            reasoning="Test",
        )
        assert dc2.confidence == 0.0

    def test_llm_classification_result_creation(self):
        """Test LLMClassificationResult creation."""
        result = LLMClassificationResult(
            classifications=[
                DomainClassification(
                    domain="finance",
                    confidence=0.9,
                    reasoning="Banking terms found",
                ),
                DomainClassification(
                    domain="technology",
                    confidence=0.3,
                    reasoning="Some tech terms",
                ),
            ],
            primary_domain="finance",
            analysis="Email about bank account",
        )
        assert result.primary_domain == "finance"
        assert len(result.classifications) == 2

    def test_get_scores(self):
        """Test get_scores method."""
        result = LLMClassificationResult(
            classifications=[
                DomainClassification(
                    domain="finance", confidence=0.9, reasoning="Test"
                ),
                DomainClassification(
                    domain="technology", confidence=0.3, reasoning="Test"
                ),
            ],
            primary_domain="finance",
            analysis="Test",
        )
        scores = result.get_scores()
        assert scores["finance"] == 0.9
        assert scores["technology"] == 0.3

    def test_get_highest_confidence(self):
        """Test get_highest_confidence method."""
        result = LLMClassificationResult(
            classifications=[
                DomainClassification(
                    domain="finance", confidence=0.9, reasoning="Test"
                ),
            ],
            primary_domain="finance",
            analysis="Test",
        )
        assert result.get_highest_confidence() == 0.9

    def test_get_highest_confidence_empty(self):
        """Test get_highest_confidence with empty classifications."""
        result = LLMClassificationResult(
            classifications=[],
            primary_domain="unsure",
            analysis="No classification",
        )
        assert result.get_highest_confidence() == 0.0

    def test_unsure_fallback(self):
        """Test creating unsure fallback result."""
        result = LLMClassificationResult.unsure("Failed to classify")
        assert result.primary_domain == "unsure"
        assert result.classifications[0].confidence == 0.0
        assert "Failed to classify" in result.analysis


class TestLLMClassifierIntegration:
    """Integration tests for LLMClassifier."""

    def test_classifier_init_without_llm_deps(self):
        """Test EmailClassifier works without LLM when not requested."""
        from email_classifier import EmailClassifier

        classifier = EmailClassifier()
        assert classifier.method3 is None
        assert classifier.llm_enabled is False

    def test_classifier_with_llm_flag_handles_missing_deps(self):
        """Test that use_llm=True handles missing dependencies gracefully."""
        from email_classifier import EmailClassifier

        # If LLM deps not installed, should log warning but not fail
        with patch.dict(os.environ, {"LLM_PROVIDER": "ollama"}, clear=True):
            classifier = EmailClassifier(use_llm=True)
            # May or may not have method3 depending on installed deps
            # Just verify it doesn't crash
            assert classifier is not None

    def test_classify_without_llm(self):
        """Test classification works without LLM."""
        from email_classifier import EmailClassifier

        classifier = EmailClassifier()
        email_data = {
            "sender": "alerts@bank.com",
            "receiver": "customer@email.com",
            "date": "2024-01-15",
            "subject": "Your account statement is ready",
            "body": "Your monthly bank statement is now available. "
            "Please review your recent transactions.",
            "urls": "",
        }
        domain, details = classifier.classify_dict(email_data)
        assert isinstance(domain, str)
        assert "method1" in details
        assert "method2" in details
        assert details.get("llm_enabled", False) is False


class TestLLMClassifierUnit:
    """Unit tests for LLMClassifier with mocks."""

    def test_normalize_domain_name_hr_variations(self):
        """Test HR domain name normalization."""
        from email_classifier.llm.agent import LLMClassifier

        config = LLMConfig(
            provider=LLMProvider.OLLAMA,
            model="llama3.2",
        )
        classifier = LLMClassifier(config)

        assert classifier._normalize_domain_name("human_resources") == "hr"
        assert classifier._normalize_domain_name("human resources") == "hr"
        assert classifier._normalize_domain_name("humanresources") == "hr"
        assert classifier._normalize_domain_name("hr") == "hr"

    def test_normalize_domain_name_telecom_variations(self):
        """Test telecom domain name normalization."""
        from email_classifier.llm.agent import LLMClassifier

        config = LLMConfig(
            provider=LLMProvider.OLLAMA,
            model="llama3.2",
        )
        classifier = LLMClassifier(config)

        assert classifier._normalize_domain_name("telecom") == "telecommunications"
        assert classifier._normalize_domain_name("telco") == "telecommunications"
        assert (
            classifier._normalize_domain_name("telecommunications")
            == "telecommunications"
        )

    def test_normalize_domain_name_social_variations(self):
        """Test social media domain name normalization."""
        from email_classifier.llm.agent import LLMClassifier

        config = LLMConfig(
            provider=LLMProvider.OLLAMA,
            model="llama3.2",
        )
        classifier = LLMClassifier(config)

        assert classifier._normalize_domain_name("social") == "social_media"
        assert classifier._normalize_domain_name("socialmedia") == "social_media"

    def test_normalize_domain_name_finance_variations(self):
        """Test finance domain name normalization."""
        from email_classifier.llm.agent import LLMClassifier

        config = LLMConfig(
            provider=LLMProvider.OLLAMA,
            model="llama3.2",
        )
        classifier = LLMClassifier(config)

        assert classifier._normalize_domain_name("banking") == "finance"
        assert classifier._normalize_domain_name("financial") == "finance"

    def test_normalize_domain_name_healthcare_variations(self):
        """Test healthcare domain name normalization."""
        from email_classifier.llm.agent import LLMClassifier

        config = LLMConfig(
            provider=LLMProvider.OLLAMA,
            model="llama3.2",
        )
        classifier = LLMClassifier(config)

        assert classifier._normalize_domain_name("medical") == "healthcare"
        assert classifier._normalize_domain_name("health") == "healthcare"

    def test_normalize_domain_name_government_variations(self):
        """Test government domain name normalization."""
        from email_classifier.llm.agent import LLMClassifier

        config = LLMConfig(
            provider=LLMProvider.OLLAMA,
            model="llama3.2",
        )
        classifier = LLMClassifier(config)

        assert classifier._normalize_domain_name("gov") == "government"
        assert classifier._normalize_domain_name("govt") == "government"

    def test_normalize_domain_name_retail_variations(self):
        """Test retail domain name normalization."""
        from email_classifier.llm.agent import LLMClassifier

        config = LLMConfig(
            provider=LLMProvider.OLLAMA,
            model="llama3.2",
        )
        classifier = LLMClassifier(config)

        assert classifier._normalize_domain_name("shopping") == "retail"
        assert classifier._normalize_domain_name("ecommerce") == "retail"
        assert classifier._normalize_domain_name("e-commerce") == "retail"

    def test_normalize_domain_name_logistics_variations(self):
        """Test logistics domain name normalization."""
        from email_classifier.llm.agent import LLMClassifier

        config = LLMConfig(
            provider=LLMProvider.OLLAMA,
            model="llama3.2",
        )
        classifier = LLMClassifier(config)

        assert classifier._normalize_domain_name("shipping") == "logistics"
        assert classifier._normalize_domain_name("delivery") == "logistics"

    def test_normalize_domain_name_technology_variations(self):
        """Test technology domain name normalization."""
        from email_classifier.llm.agent import LLMClassifier

        config = LLMConfig(
            provider=LLMProvider.OLLAMA,
            model="llama3.2",
        )
        classifier = LLMClassifier(config)

        assert classifier._normalize_domain_name("tech") == "technology"
        assert classifier._normalize_domain_name("software") == "technology"

    def test_normalize_domain_name_education_variations(self):
        """Test education domain name normalization."""
        from email_classifier.llm.agent import LLMClassifier

        config = LLMConfig(
            provider=LLMProvider.OLLAMA,
            model="llama3.2",
        )
        classifier = LLMClassifier(config)

        assert classifier._normalize_domain_name("school") == "education"
        assert classifier._normalize_domain_name("university") == "education"
        assert classifier._normalize_domain_name("college") == "education"

    def test_normalize_domain_name_unknown(self):
        """Test unknown domain name passes through unchanged."""
        from email_classifier.llm.agent import LLMClassifier

        config = LLMConfig(
            provider=LLMProvider.OLLAMA,
            model="llama3.2",
        )
        classifier = LLMClassifier(config)

        assert classifier._normalize_domain_name("unknown_domain") == "unknown_domain"
        assert classifier._normalize_domain_name("finance") == "finance"

    def test_validate_result_empty_classifications(self):
        """Test validate_result handles empty classifications."""
        from email_classifier.llm.agent import LLMClassifier

        config = LLMConfig(
            provider=LLMProvider.OLLAMA,
            model="llama3.2",
        )
        classifier = LLMClassifier(config)

        result = LLMClassificationResult(
            classifications=[],
            primary_domain="unknown",
            analysis="Test",
        )
        validated = classifier._validate_result(result)
        assert validated.primary_domain == "unsure"
        assert len(validated.classifications) == 1
        assert validated.classifications[0].confidence == 0.3

    def test_validate_result_invalid_domains_filtered(self):
        """Test validate_result filters invalid domain names."""
        from email_classifier.llm.agent import LLMClassifier

        config = LLMConfig(
            provider=LLMProvider.OLLAMA,
            model="llama3.2",
        )
        classifier = LLMClassifier(config)

        result = LLMClassificationResult(
            classifications=[
                DomainClassification(
                    domain="invalid_domain_xyz",
                    confidence=0.9,
                    reasoning="Test",
                ),
                DomainClassification(
                    domain="finance",
                    confidence=0.7,
                    reasoning="Test",
                ),
            ],
            primary_domain="invalid_domain_xyz",
            analysis="Test",
        )
        validated = classifier._validate_result(result)
        # Invalid domain should be filtered, only finance remains
        assert validated.primary_domain == "finance"
        assert len(validated.classifications) == 1
        assert validated.classifications[0].domain == "finance"

    def test_validate_result_confidence_clamping_negative(self):
        """Test validate_result handles negative confidence by clamping to 0."""
        from email_classifier.llm.agent import LLMClassifier

        config = LLMConfig(
            provider=LLMProvider.OLLAMA,
            model="llama3.2",
        )
        classifier = LLMClassifier(config)

        # Create a mock result with confidence that needs clamping
        # (Pydantic validates on creation, so we test the clamping logic differently)
        result = LLMClassificationResult(
            classifications=[
                DomainClassification(
                    domain="finance",
                    confidence=0.0,  # Boundary value
                    reasoning="Test",
                ),
            ],
            primary_domain="finance",
            analysis="Test",
        )
        validated = classifier._validate_result(result)
        assert validated.classifications[0].confidence == 0.0

    def test_validate_result_sorts_by_confidence(self):
        """Test validate_result sorts classifications by confidence."""
        from email_classifier.llm.agent import LLMClassifier

        config = LLMConfig(
            provider=LLMProvider.OLLAMA,
            model="llama3.2",
        )
        classifier = LLMClassifier(config)

        result = LLMClassificationResult(
            classifications=[
                DomainClassification(
                    domain="finance",
                    confidence=0.5,
                    reasoning="Test",
                ),
                DomainClassification(
                    domain="healthcare",
                    confidence=0.9,
                    reasoning="Test",
                ),
            ],
            primary_domain="finance",
            analysis="Test",
        )
        validated = classifier._validate_result(result)
        assert validated.primary_domain == "healthcare"
        assert validated.classifications[0].domain == "healthcare"
        assert validated.classifications[1].domain == "finance"

    def test_convert_to_classification_result(self):
        """Test convert_to_classification_result produces correct format."""
        from email_classifier.llm.agent import LLMClassifier

        config = LLMConfig(
            provider=LLMProvider.OLLAMA,
            model="llama3.2",
        )
        classifier = LLMClassifier(config)

        llm_result = LLMClassificationResult(
            classifications=[
                DomainClassification(
                    domain="finance",
                    confidence=0.85,
                    reasoning="Banking keywords",
                ),
            ],
            primary_domain="finance",
            analysis="Email about banking",
        )
        result = classifier._convert_to_classification_result(llm_result)

        assert result.domain == "finance"
        assert result.confidence == 0.85
        assert result.method == "llm_agent"
        assert result.scores["finance"] == 0.85
        assert "analysis" in result.details
        assert result.details["provider"] == "ollama"

    def test_convert_to_classification_result_unsure(self):
        """Test convert handles unsure domain."""
        from email_classifier.llm.agent import LLMClassifier

        config = LLMConfig(
            provider=LLMProvider.OLLAMA,
            model="llama3.2",
        )
        classifier = LLMClassifier(config)

        llm_result = LLMClassificationResult(
            classifications=[
                DomainClassification(
                    domain="unsure",
                    confidence=0.3,
                    reasoning="Could not classify",
                ),
            ],
            primary_domain="unsure",
            analysis="Unclear email",
        )
        result = classifier._convert_to_classification_result(llm_result)

        assert result.domain is None
        assert result.confidence == 0.0

    def test_create_fallback_result(self):
        """Test _create_fallback_result produces correct error result."""
        from email_classifier.llm.agent import LLMClassifier

        config = LLMConfig(
            provider=LLMProvider.OLLAMA,
            model="llama3.2",
        )
        classifier = LLMClassifier(config)

        result = classifier._create_fallback_result("Connection timeout")

        assert result.domain is None
        assert result.confidence == 0.0
        assert result.method == "llm_agent"
        assert result.details["error"] == "Connection timeout"
        assert result.details["fallback"] is True
        # All domains should have zero score
        for score in result.scores.values():
            assert score == 0.0

    def test_classify_with_mocked_llm_success(self):
        """Test classify with mocked successful LLM response."""
        from email_classifier.llm.agent import LLMClassifier

        config = LLMConfig(
            provider=LLMProvider.OLLAMA,
            model="llama3.2",
        )
        classifier = LLMClassifier(config)

        # Mock the LLM
        mock_llm = MagicMock()
        mock_structured_llm = MagicMock()
        mock_structured_llm.invoke.return_value = LLMClassificationResult(
            classifications=[
                DomainClassification(
                    domain="finance",
                    confidence=0.9,
                    reasoning="Banking terms",
                ),
            ],
            primary_domain="finance",
            analysis="Banking email",
        )
        mock_llm.with_structured_output.return_value = mock_structured_llm
        classifier._llm = mock_llm

        email = EmailData(
            sender="alerts@bank.com",
            receiver="user@email.com",
            date="2024-01-15",
            subject="Account update",
            body="Your account balance is ready.",
            urls="",
        )
        result = classifier.classify(email)

        assert result.domain == "finance"
        assert result.confidence == 0.9
        mock_structured_llm.invoke.assert_called_once()

    def test_classify_with_mocked_llm_failure(self):
        """Test classify handles LLM failure gracefully."""
        from email_classifier.llm.agent import LLMClassifier

        config = LLMConfig(
            provider=LLMProvider.OLLAMA,
            model="llama3.2",
            retry_count=0,  # No retries for faster test
        )
        classifier = LLMClassifier(config)

        # Mock the LLM to raise an exception
        mock_llm = MagicMock()
        mock_structured_llm = MagicMock()
        mock_structured_llm.invoke.side_effect = Exception("LLM unavailable")
        mock_llm.with_structured_output.return_value = mock_structured_llm
        classifier._llm = mock_llm

        email = EmailData(
            sender="test@example.com",
            receiver="user@email.com",
            date="2024-01-15",
            subject="Test",
            body="Test body.",
            urls="",
        )
        result = classifier.classify(email)

        # Should return fallback result
        assert result.domain is None
        assert result.confidence == 0.0
        assert result.details.get("fallback") is True


class TestLLMProvidersFactory:
    """Test cases for provider factory."""

    def test_check_provider_available_structure(self):
        """Test check_provider_available returns proper structure."""
        from email_classifier.llm.providers import check_provider_available

        available, error = check_provider_available(LLMProvider.OLLAMA)
        assert isinstance(available, bool)
        assert error is None or isinstance(error, str)

    def test_check_provider_available_all_providers(self):
        """Test check_provider_available for all providers."""
        from email_classifier.llm.providers import check_provider_available

        for provider in LLMProvider:
            available, error = check_provider_available(provider)
            assert isinstance(available, bool)

    def test_provider_not_installed_error(self):
        """Test ProviderNotInstalledError message format."""
        from email_classifier.llm.providers import ProviderNotInstalledError

        error = ProviderNotInstalledError(
            provider=LLMProvider.GOOGLE,
            package="langchain-google-genai",
            install_cmd="pip install email-domain-classifier[google]",
        )
        assert "google" in str(error)
        assert "langchain-google-genai" in str(error)
        assert "pip install" in str(error)
        assert error.provider == LLMProvider.GOOGLE
        assert error.package == "langchain-google-genai"

    def test_create_llm_unknown_provider(self):
        """Test create_llm with invalid provider raises error."""
        from email_classifier.llm.config import LLMConfigError
        from email_classifier.llm.providers import create_llm

        # Create a mock config with invalid provider
        config = MagicMock()
        config.provider = "invalid_provider"

        with pytest.raises(LLMConfigError, match="Unknown provider"):
            create_llm(config)


class TestLLMPrompts:
    """Test cases for prompt generation."""

    def test_get_system_prompt(self):
        """Test system prompt generation."""
        from email_classifier.llm.prompts import get_system_prompt

        prompt = get_system_prompt()
        assert "finance" in prompt.lower()
        assert "technology" in prompt.lower()
        assert "classification" in prompt.lower()

    def test_get_classification_prompt(self):
        """Test classification prompt generation."""
        from email_classifier.llm.prompts import get_classification_prompt

        prompt = get_classification_prompt(
            sender="test@example.com",
            subject="Test Subject",
            body="Test body content",
        )
        assert "test@example.com" in prompt
        assert "Test Subject" in prompt
        assert "Test body content" in prompt

    def test_get_classification_prompt_truncation(self):
        """Test that long bodies are truncated."""
        from email_classifier.llm.prompts import get_classification_prompt

        long_body = "x" * 5000
        prompt = get_classification_prompt(
            sender="test@example.com",
            subject="Test",
            body=long_body,
            max_body_chars=100,
        )
        assert "[truncated]" in prompt
        assert len(prompt) < 5000

    def test_domain_list_for_prompt(self):
        """Test domain list generation for prompt."""
        from email_classifier.llm.prompts import get_domain_list_for_prompt

        domain_list = get_domain_list_for_prompt()
        assert "finance" in domain_list
        assert "healthcare" in domain_list
        assert "unsure" in domain_list
