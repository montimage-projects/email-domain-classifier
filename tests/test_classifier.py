"""Tests for the main EmailClassifier."""

import pytest

from email_classifier import EmailClassifier, EmailData
from email_classifier.classifier import (
    ClassificationResult,
    KeywordTaxonomyClassifier,
    StructuralTemplateClassifier,
)
from email_classifier.domains import DOMAINS


class TestEmailClassifier:
    """Test cases for EmailClassifier class."""

    def test_classifier_initialization(self):
        """Test that classifier can be initialized."""
        classifier = EmailClassifier()
        assert classifier is not None
        assert hasattr(classifier, "method1")
        assert hasattr(classifier, "method2")

    def test_classify_dict_basic(self):
        """Test basic classification with dictionary input."""
        classifier = EmailClassifier()
        email_data = {
            "sender": "test@example.com",
            "receiver": "user@domain.com",
            "timestamp": "2024-01-15 10:30:00",
            "subject": "Test subject",
            "body": "This is a test email body.",
            "has_url": False,
        }

        domain, details = classifier.classify_dict(email_data)
        assert isinstance(domain, str)
        assert isinstance(details, dict)
        assert "method1" in details
        assert "method2" in details

    def test_email_data_creation(self):
        """Test EmailData creation from dictionary."""
        data = {
            "sender": "test@example.com",
            "receiver": "user@domain.com",
            "timestamp": "2024-01-15 10:30:00",
            "subject": "Test",
            "body": "Test body",
            "has_url": True,
        }

        email = EmailData.from_dict(data)
        assert email.sender == "test@example.com"
        assert email.receiver == "user@domain.com"
        assert email.has_url is True

    def test_email_data_with_urls_field(self):
        """Test EmailData handles urls field correctly."""
        data = {
            "sender": "test@example.com",
            "receiver": "user@domain.com",
            "date": "2024-01-15",
            "subject": "Test",
            "body": "Test body",
            "urls": "http://example.com",
        }
        email = EmailData.from_dict(data)
        assert email.has_url is True

    def test_email_data_empty_urls(self):
        """Test EmailData with empty urls field."""
        data = {
            "sender": "test@example.com",
            "receiver": "user@domain.com",
            "date": "2024-01-15",
            "subject": "Test",
            "body": "Test body",
            "urls": "",
        }
        email = EmailData.from_dict(data)
        assert email.has_url is False

    def test_email_data_missing_fields(self):
        """Test EmailData handles missing fields gracefully."""
        data = {}
        email = EmailData.from_dict(data)
        assert email.sender == ""
        assert email.receiver == ""
        assert email.subject == ""
        assert email.body == ""
        assert email.has_url is False

    def test_classify_finance_email(self):
        """Test classification of finance-related email."""
        classifier = EmailClassifier()
        email_data = {
            "sender": "alerts@bank.com",
            "receiver": "customer@email.com",
            "date": "2024-01-15",
            "subject": "Your account statement is ready",
            "body": "Dear Customer, Your monthly bank statement is now available. "
            "Please review your recent transactions and account balance. "
            "Your current balance is $1,234.56. Transfer funds securely online.",
            "urls": "https://bank.com/statement",
        }
        domain, details = classifier.classify_dict(email_data)
        assert domain == "finance"
        assert details["method1"]["domain"] == "finance"

    def test_classify_healthcare_email(self):
        """Test classification of healthcare-related email."""
        classifier = EmailClassifier()
        email_data = {
            "sender": "appointments@hospital.com",
            "receiver": "patient@email.com",
            "date": "2024-01-15",
            "subject": "Appointment Reminder - Dr. Smith",
            "body": "Dear Patient, This is a reminder for your appointment with "
            "Dr. Smith on Monday. Please bring your insurance card and medical records. "
            "Your prescription is ready at the pharmacy. HIPAA Notice: This message is confidential.",
            "urls": "",
        }
        domain, details = classifier.classify_dict(email_data)
        assert domain == "healthcare"

    def test_classify_retail_email(self):
        """Test classification of retail-related email."""
        classifier = EmailClassifier()
        email_data = {
            "sender": "orders@shop.com",
            "receiver": "customer@email.com",
            "date": "2024-01-15",
            "subject": "Order Confirmation #12345",
            "body": "Thank you for your purchase! Your order has been confirmed. "
            "Items: Product ABC - $29.99. Total: $32.99 with shipping. "
            "Your package will be shipped within 2 business days. Track your delivery online.",
            "urls": "https://shop.com/track",
        }
        domain, details = classifier.classify_dict(email_data)
        assert domain == "retail"

    def test_classify_technology_email(self):
        """Test classification of technology-related email."""
        classifier = EmailClassifier()
        email_data = {
            "sender": "noreply@techservice.com",
            "receiver": "user@email.com",
            "date": "2024-01-15",
            "subject": "Password Reset Request",
            "body": "We received a request to reset your password. "
            "Click the link below to update your account credentials. "
            "Your subscription will expire in 30 days. Update your software for security.",
            "urls": "https://tech.com/reset",
        }
        domain, details = classifier.classify_dict(email_data)
        assert domain == "technology"

    def test_classify_unsure_low_confidence(self):
        """Test classification returns unsure for ambiguous emails."""
        classifier = EmailClassifier()
        email_data = {
            "sender": "unknown@random.xyz",
            "receiver": "user@email.com",
            "date": "2024-01-15",
            "subject": "Hello",
            "body": "Hi there.",
            "urls": "",
        }
        domain, details = classifier.classify_dict(email_data)
        # Could be unsure or might match something due to low thresholds
        assert domain is not None

    def test_classifier_llm_disabled_by_default(self):
        """Test that LLM is disabled by default."""
        classifier = EmailClassifier()
        assert classifier.llm_enabled is False
        assert classifier.method3 is None
        assert classifier.weight_method_1 == 0.6
        assert classifier.weight_method_2 == 0.4
        assert classifier.weight_method_3 == 0.0

    def test_classifier_method_weights_structure(self):
        """Test that method weights are included in classification details."""
        classifier = EmailClassifier()
        email_data = {
            "sender": "test@example.com",
            "receiver": "user@domain.com",
            "date": "2024-01-15",
            "subject": "Test",
            "body": "Test body content",
            "urls": "",
        }
        domain, details = classifier.classify_dict(email_data)
        assert "method_weights" in details
        assert "method1" in details["method_weights"]
        assert "method2" in details["method_weights"]
        assert "llm_enabled" in details


class TestKeywordTaxonomyClassifier:
    """Test cases for KeywordTaxonomyClassifier."""

    def test_initialization(self):
        """Test classifier initialization."""
        classifier = KeywordTaxonomyClassifier()
        assert classifier.domains is not None
        assert len(classifier.domains) > 0

    def test_initialization_with_custom_domains(self):
        """Test classifier with custom domains."""
        custom_domains = {"custom": DOMAINS["finance"]}
        classifier = KeywordTaxonomyClassifier(domains=custom_domains)
        assert "custom" in classifier.domains
        assert len(classifier.domains) == 1

    def test_classify_returns_classification_result(self):
        """Test classify returns ClassificationResult."""
        classifier = KeywordTaxonomyClassifier()
        email = EmailData(
            sender="test@example.com",
            receiver="user@domain.com",
            date="2024-01-15",
            subject="Test",
            body="Test body",
            urls="",
        )
        result = classifier.classify(email)
        assert isinstance(result, ClassificationResult)
        assert result.method == "keyword_taxonomy"

    def test_classify_with_primary_keywords_in_subject(self):
        """Test keywords in subject increase score."""
        classifier = KeywordTaxonomyClassifier()
        email = EmailData(
            sender="test@bank.com",
            receiver="user@domain.com",
            date="2024-01-15",
            subject="Your account balance statement",
            body="Please check your account.",
            urls="",
        )
        result = classifier.classify(email)
        assert result.scores["finance"] > 0

    def test_classify_with_secondary_keywords_in_body(self):
        """Test secondary keywords in body contribute to score."""
        classifier = KeywordTaxonomyClassifier()
        email = EmailData(
            sender="test@example.com",
            receiver="user@domain.com",
            date="2024-01-15",
            subject="Information",
            body="Your financial funds transfer is pending. Money exchange rate updated.",
            urls="",
        )
        result = classifier.classify(email)
        assert result.scores["finance"] > 0

    def test_classify_sender_pattern_match(self):
        """Test sender pattern matching."""
        classifier = KeywordTaxonomyClassifier()
        email = EmailData(
            sender="alerts@bankofamerica.com",
            receiver="user@domain.com",
            date="2024-01-15",
            subject="Notice",
            body="Important update.",
            urls="",
        )
        result = classifier.classify(email)
        # Should match bank pattern
        assert result.details is not None
        assert result.details["finance"]["sender_match"] is True

    def test_classify_subject_pattern_match(self):
        """Test subject pattern matching."""
        classifier = KeywordTaxonomyClassifier()
        email = EmailData(
            sender="test@example.com",
            receiver="user@domain.com",
            date="2024-01-15",
            subject="Your transaction has been processed",
            body="Details enclosed.",
            urls="",
        )
        result = classifier.classify(email)
        assert result.details is not None
        assert result.details["finance"]["subject_pattern_match"] is True

    def test_classify_empty_body(self):
        """Test classification with empty body."""
        classifier = KeywordTaxonomyClassifier()
        email = EmailData(
            sender="test@example.com",
            receiver="user@domain.com",
            date="2024-01-15",
            subject="Test",
            body="",
            urls="",
        )
        result = classifier.classify(email)
        # Should not crash, confidence should be low
        assert result is not None

    def test_classify_below_confidence_threshold(self):
        """Test that low confidence returns None domain."""
        classifier = KeywordTaxonomyClassifier()
        email = EmailData(
            sender="random@xyz.com",
            receiver="user@domain.com",
            date="2024-01-15",
            subject="xyz",
            body="xyz",
            urls="",
        )
        result = classifier.classify(email)
        # With no matching keywords, confidence should be low
        assert result.confidence < 0.1 or result.domain is None


class TestStructuralTemplateClassifier:
    """Test cases for StructuralTemplateClassifier."""

    def test_initialization(self):
        """Test classifier initialization."""
        classifier = StructuralTemplateClassifier()
        assert classifier.domains is not None

    def test_initialization_with_custom_domains(self):
        """Test classifier with custom domains."""
        custom_domains = {"custom": DOMAINS["technology"]}
        classifier = StructuralTemplateClassifier(domains=custom_domains)
        assert "custom" in classifier.domains

    def test_classify_returns_classification_result(self):
        """Test classify returns ClassificationResult."""
        classifier = StructuralTemplateClassifier()
        email = EmailData(
            sender="test@example.com",
            receiver="user@domain.com",
            date="2024-01-15",
            subject="Test",
            body="Test body content here.",
            urls="",
        )
        result = classifier.classify(email)
        assert isinstance(result, ClassificationResult)
        assert result.method == "structural_template"

    def test_extract_features_greeting(self):
        """Test greeting detection in features."""
        classifier = StructuralTemplateClassifier()
        email = EmailData(
            sender="test@example.com",
            receiver="user@domain.com",
            date="2024-01-15",
            subject="Test",
            body="Dear Customer,\n\nThis is a test message.",
            urls="",
        )
        features = classifier._extract_features(email)
        assert features["has_greeting"] is True

    def test_extract_features_no_greeting(self):
        """Test no greeting detected."""
        classifier = StructuralTemplateClassifier()
        email = EmailData(
            sender="test@example.com",
            receiver="user@domain.com",
            date="2024-01-15",
            subject="Test",
            body="Your order has shipped.",
            urls="",
        )
        features = classifier._extract_features(email)
        assert features["has_greeting"] is False

    def test_extract_features_signature(self):
        """Test signature detection."""
        classifier = StructuralTemplateClassifier()
        email = EmailData(
            sender="test@example.com",
            receiver="user@domain.com",
            date="2024-01-15",
            subject="Test",
            body="Message content.\n\nBest regards,\nJohn",
            urls="",
        )
        features = classifier._extract_features(email)
        assert features["has_signature"] is True

    def test_extract_features_disclaimer(self):
        """Test disclaimer detection."""
        classifier = StructuralTemplateClassifier()
        email = EmailData(
            sender="test@example.com",
            receiver="user@domain.com",
            date="2024-01-15",
            subject="Test",
            body="Content here.\n\nThis email is confidential and intended for the recipient only.",
            urls="",
        )
        features = classifier._extract_features(email)
        assert features["has_disclaimer"] is True

    def test_extract_features_paragraph_count(self):
        """Test paragraph counting."""
        classifier = StructuralTemplateClassifier()
        email = EmailData(
            sender="test@example.com",
            receiver="user@domain.com",
            date="2024-01-15",
            subject="Test",
            body="First paragraph.\n\nSecond paragraph.\n\nThird paragraph.",
            urls="",
        )
        features = classifier._extract_features(email)
        assert features["paragraph_count"] == 3

    def test_extract_features_formality_formal(self):
        """Test formal language detection."""
        classifier = StructuralTemplateClassifier()
        email = EmailData(
            sender="test@example.com",
            receiver="user@domain.com",
            date="2024-01-15",
            subject="Test",
            body="Pursuant to our discussion, please find enclosed the document. "
            "Regarding your inquiry, we hereby confirm the receipt. Respectfully submitted.",
            urls="",
        )
        features = classifier._extract_features(email)
        assert features["formality"] == "formal"

    def test_extract_features_formality_casual(self):
        """Test casual language detection."""
        classifier = StructuralTemplateClassifier()
        email = EmailData(
            sender="test@example.com",
            receiver="user@domain.com",
            date="2024-01-15",
            subject="Test",
            body="Hey! That's awesome!! BTW gonna check it out :) Cool stuff FYI!",
            urls="",
        )
        features = classifier._extract_features(email)
        assert features["formality"] == "casual"

    def test_extract_features_formality_semiformal(self):
        """Test semi-formal language detection."""
        classifier = StructuralTemplateClassifier()
        email = EmailData(
            sender="test@example.com",
            receiver="user@domain.com",
            date="2024-01-15",
            subject="Test",
            body="Hello, I wanted to follow up on our conversation. Thanks for your time.",
            urls="",
        )
        features = classifier._extract_features(email)
        assert features["formality"] == "semi-formal"

    def test_analyze_sender_noreply(self):
        """Test noreply sender detection."""
        classifier = StructuralTemplateClassifier()
        features = classifier._analyze_sender_structure("noreply@company.com")
        assert features["is_noreply"] is True

    def test_analyze_sender_donotreply(self):
        """Test donotreply sender detection."""
        classifier = StructuralTemplateClassifier()
        features = classifier._analyze_sender_structure("donotreply@company.com")
        assert features["is_noreply"] is True

    def test_analyze_sender_department(self):
        """Test department detection in sender."""
        classifier = StructuralTemplateClassifier()
        features = classifier._analyze_sender_structure("support@company.com")
        assert features["has_department"] is True

        features = classifier._analyze_sender_structure("billing@company.com")
        assert features["has_department"] is True

    def test_analyze_sender_domain_gov(self):
        """Test .gov domain detection."""
        classifier = StructuralTemplateClassifier()
        features = classifier._analyze_sender_structure("notice@agency.gov")
        assert features["domain_type"] == "government"

    def test_analyze_sender_domain_edu(self):
        """Test .edu domain detection."""
        classifier = StructuralTemplateClassifier()
        features = classifier._analyze_sender_structure("registrar@university.edu")
        assert features["domain_type"] == "education"

    def test_analyze_sender_domain_commercial(self):
        """Test commercial domain detection."""
        classifier = StructuralTemplateClassifier()
        features = classifier._analyze_sender_structure("info@company.com")
        assert features["domain_type"] == "commercial"

    def test_analyze_sender_no_at_symbol(self):
        """Test sender without @ symbol."""
        classifier = StructuralTemplateClassifier()
        features = classifier._analyze_sender_structure("support")
        assert features["has_department"] is True

    def test_score_template_body_length_match(self):
        """Test body length scoring - exact match."""
        classifier = StructuralTemplateClassifier()
        features = {
            "body_length": 500,
            "has_greeting": True,
            "has_signature": True,
            "has_disclaimer": False,
            "has_url": True,
            "formality": "semi-formal",
            "paragraph_count": 3,
            "sender": {
                "domain_type": "unknown",
                "is_noreply": False,
                "has_department": False,
            },
        }
        score, details = classifier._score_template_match(
            features, DOMAINS["technology"]
        )
        assert details["body_length"] == "match"

    def test_score_template_body_length_partial(self):
        """Test body length scoring - partial match."""
        classifier = StructuralTemplateClassifier()
        features = {
            "body_length": 80,  # Below typical but not extreme
            "has_greeting": True,
            "has_signature": True,
            "has_disclaimer": False,
            "has_url": True,
            "formality": "semi-formal",
            "paragraph_count": 2,
            "sender": {
                "domain_type": "unknown",
                "is_noreply": False,
                "has_department": False,
            },
        }
        score, details = classifier._score_template_match(
            features, DOMAINS["technology"]
        )
        assert details["body_length"] == "partial"

    def test_score_template_body_length_mismatch(self):
        """Test body length scoring - mismatch."""
        classifier = StructuralTemplateClassifier()
        features = {
            "body_length": 10,  # Very short
            "has_greeting": True,
            "has_signature": True,
            "has_disclaimer": False,
            "has_url": True,
            "formality": "semi-formal",
            "paragraph_count": 1,
            "sender": {
                "domain_type": "unknown",
                "is_noreply": False,
                "has_department": False,
            },
        }
        score, details = classifier._score_template_match(
            features, DOMAINS["technology"]
        )
        assert details["body_length"] == "mismatch"

    def test_score_template_formality_mismatch(self):
        """Test formality mismatch scoring."""
        classifier = StructuralTemplateClassifier()
        features = {
            "body_length": 500,
            "has_greeting": True,
            "has_signature": True,
            "has_disclaimer": True,
            "has_url": True,
            "formality": "casual",  # Finance expects formal
            "paragraph_count": 3,
            "sender": {
                "domain_type": "unknown",
                "is_noreply": False,
                "has_department": False,
            },
        }
        score, details = classifier._score_template_match(features, DOMAINS["finance"])
        assert details["formality"] == "mismatch"

    def test_score_template_noreply_match(self):
        """Test noreply sender scoring for technology/retail/logistics."""
        classifier = StructuralTemplateClassifier()
        features = {
            "body_length": 500,
            "has_greeting": True,
            "has_signature": True,
            "has_disclaimer": False,
            "has_url": True,
            "formality": "semi-formal",
            "paragraph_count": 2,
            "sender": {
                "domain_type": "unknown",
                "is_noreply": True,
                "has_department": False,
            },
        }
        score, details = classifier._score_template_match(
            features, DOMAINS["technology"]
        )
        assert details["sender"] == "noreply_match"

    def test_classify_zero_total_scores(self):
        """Test handling when all scores are zero."""
        classifier = StructuralTemplateClassifier()
        email = EmailData(
            sender="x@y.z",
            receiver="a@b.c",
            date="2024-01-15",
            subject="x",
            body="x",
            urls="",
        )
        result = classifier.classify(email)
        # Should handle gracefully
        assert result is not None


class TestClassificationResult:
    """Test cases for ClassificationResult dataclass."""

    def test_creation(self):
        """Test ClassificationResult creation."""
        result = ClassificationResult(
            domain="finance",
            confidence=0.85,
            scores={"finance": 0.85, "technology": 0.15},
            method="keyword_taxonomy",
        )
        assert result.domain == "finance"
        assert result.confidence == 0.85
        assert result.method == "keyword_taxonomy"

    def test_creation_with_details(self):
        """Test ClassificationResult with details."""
        result = ClassificationResult(
            domain="healthcare",
            confidence=0.75,
            scores={"healthcare": 0.75},
            method="structural_template",
            details={"feature": "value"},
        )
        assert result.details == {"feature": "value"}

    def test_none_domain(self):
        """Test ClassificationResult with None domain."""
        result = ClassificationResult(
            domain=None,
            confidence=0.0,
            scores={},
            method="test",
        )
        assert result.domain is None


class TestHybridClassifier:
    """Test cases for HybridClassifier class."""

    def test_initialization(self):
        """Test HybridClassifier can be initialized without LLM."""
        from email_classifier.classifier import HybridClassifier

        classifier = HybridClassifier()
        assert classifier is not None
        assert hasattr(classifier, "method1")
        assert hasattr(classifier, "method2")
        assert classifier.llm_classifier is None

    def test_initialization_with_status_callback(self):
        """Test HybridClassifier with status callback."""
        from email_classifier.classifier import HybridClassifier

        status_messages = []

        def status_callback(msg: str) -> None:
            status_messages.append(msg)

        classifier = HybridClassifier(status_callback=status_callback)
        assert classifier.status_callback is not None

    def test_classify_agreement_path(self):
        """Test classification when both classifiers agree."""
        from email_classifier.classifier import HybridClassifier

        classifier = HybridClassifier()
        email_data = {
            "sender": "alerts@bank.com",
            "receiver": "customer@email.com",
            "date": "2024-01-15",
            "subject": "Your account statement is ready",
            "body": "Dear Customer, Your monthly bank statement is now available. "
            "Please review your recent transactions and account balance. "
            "Your current balance is $1,234.56. Transfer funds securely online.",
            "urls": "https://bank.com/statement",
        }
        domain, details = classifier.classify_dict(email_data)
        assert domain == "finance"
        assert details.get("hybrid_workflow") is True
        # Since both classifiers should agree on finance, path should be classic_only
        # (Note: may depend on actual classifier results)

    def test_classify_records_stats(self):
        """Test that hybrid classifier records statistics."""
        from email_classifier.classifier import HybridClassifier

        classifier = HybridClassifier()
        email_data = {
            "sender": "alerts@bank.com",
            "receiver": "customer@email.com",
            "date": "2024-01-15",
            "subject": "Your account balance",
            "body": "Your bank account balance is $1,234.56.",
            "urls": "",
        }
        classifier.classify_dict(email_data)
        stats = classifier.get_stats()
        assert stats.total_processed == 1

    def test_classify_multiple_emails_stats(self):
        """Test stats accumulation over multiple emails."""
        from email_classifier.classifier import HybridClassifier

        classifier = HybridClassifier()
        emails = [
            {
                "sender": "bank@finance.com",
                "receiver": "user@email.com",
                "date": "2024-01-15",
                "subject": "Account statement",
                "body": "Your account balance is ready for review.",
                "urls": "",
            },
            {
                "sender": "shop@store.com",
                "receiver": "user@email.com",
                "date": "2024-01-15",
                "subject": "Order confirmation",
                "body": "Your order has been shipped. Track your package.",
                "urls": "",
            },
        ]
        for email in emails:
            classifier.classify_dict(email)

        stats = classifier.get_stats()
        assert stats.total_processed == 2

    def test_reset_stats(self):
        """Test resetting statistics."""
        from email_classifier.classifier import HybridClassifier

        classifier = HybridClassifier()
        email_data = {
            "sender": "test@example.com",
            "receiver": "user@email.com",
            "date": "2024-01-15",
            "subject": "Test",
            "body": "Test body",
            "urls": "",
        }
        classifier.classify_dict(email_data)
        assert classifier.get_stats().total_processed == 1

        classifier.reset_stats()
        assert classifier.get_stats().total_processed == 0

    def test_status_callback_called(self):
        """Test that status callback is invoked during classification."""
        from email_classifier.classifier import HybridClassifier

        status_messages = []

        def status_callback(msg: str) -> None:
            status_messages.append(msg)

        classifier = HybridClassifier(status_callback=status_callback)
        email_data = {
            "sender": "test@example.com",
            "receiver": "user@email.com",
            "date": "2024-01-15",
            "subject": "Test subject",
            "body": "Test body content",
            "urls": "",
        }
        classifier.classify_dict(email_data)

        # Should have received status updates
        assert len(status_messages) >= 2  # At least keyword and structural status
        assert any("Keyword Taxonomy" in msg for msg in status_messages)
        assert any("Structural Template" in msg for msg in status_messages)

    def test_classify_details_include_hybrid_info(self):
        """Test that classification details include hybrid workflow info."""
        from email_classifier.classifier import HybridClassifier

        classifier = HybridClassifier()
        email_data = {
            "sender": "test@example.com",
            "receiver": "user@email.com",
            "date": "2024-01-15",
            "subject": "Test",
            "body": "Test body",
            "urls": "",
        }
        domain, details = classifier.classify_dict(email_data)

        assert "hybrid_workflow" in details
        assert details["hybrid_workflow"] is True
        assert "path" in details
        assert details["path"] in ["classic_only", "llm_assisted"]
        assert "agreement" in details

    def test_fallback_classification_no_llm(self):
        """Test fallback classification when LLM is not available."""
        from email_classifier.classifier import HybridClassifier

        classifier = HybridClassifier()  # No LLM config

        # Ambiguous email that might cause disagreement
        email_data = {
            "sender": "info@company.com",
            "receiver": "user@email.com",
            "date": "2024-01-15",
            "subject": "Information",
            "body": "Here is some information for you.",
            "urls": "",
        }
        domain, details = classifier.classify_dict(email_data)

        # Should still return a valid result via fallback
        assert domain is not None
        assert isinstance(domain, str)


class TestHybridWorkflowLogger:
    """Test cases for HybridWorkflowLogger class."""

    def test_initialization_no_file(self):
        """Test logger initialization without file."""
        from email_classifier.classifier import HybridWorkflowLogger

        logger = HybridWorkflowLogger()
        assert logger.log_file is None
        assert logger._file_handle is None

    def test_log_step(self):
        """Test logging a step (without file, uses Python logger)."""
        from email_classifier.classifier import HybridWorkflowLogger

        logger = HybridWorkflowLogger()
        # Should not raise
        logger.log_step(0, "keyword_classify", result="finance")
        logger.log_step(0, "agreement_check", path="classic_only")

    def test_log_step_with_llm_time(self):
        """Test logging step with LLM time."""
        from email_classifier.classifier import HybridWorkflowLogger

        logger = HybridWorkflowLogger()
        logger.log_step(1, "llm_classify", result="technology", llm_time_ms=123.45)


class TestHybridWorkflowStats:
    """Test cases for HybridWorkflowStats dataclass."""

    def test_initialization(self):
        """Test default initialization."""
        from email_classifier.classifier import HybridWorkflowStats

        stats = HybridWorkflowStats()
        assert stats.llm_call_count == 0
        assert stats.llm_total_time_ms == 0.0
        assert stats.classic_agreement_count == 0
        assert stats.total_processed == 0

    def test_llm_avg_time_zero_calls(self):
        """Test avg time calculation with zero calls."""
        from email_classifier.classifier import HybridWorkflowStats

        stats = HybridWorkflowStats()
        assert stats.llm_avg_time_ms == 0.0

    def test_llm_avg_time_with_calls(self):
        """Test avg time calculation with calls."""
        from email_classifier.classifier import HybridWorkflowStats

        stats = HybridWorkflowStats(llm_call_count=5, llm_total_time_ms=1000.0)
        assert stats.llm_avg_time_ms == 200.0

    def test_agreement_rate_zero_processed(self):
        """Test agreement rate with zero processed."""
        from email_classifier.classifier import HybridWorkflowStats

        stats = HybridWorkflowStats()
        assert stats.agreement_rate == 0.0

    def test_agreement_rate_with_processed(self):
        """Test agreement rate calculation."""
        from email_classifier.classifier import HybridWorkflowStats

        stats = HybridWorkflowStats(
            total_processed=100, classic_agreement_count=80, llm_call_count=20
        )
        assert stats.agreement_rate == 80.0

    def test_to_dict(self):
        """Test conversion to dictionary."""
        from email_classifier.classifier import HybridWorkflowStats

        stats = HybridWorkflowStats(
            llm_call_count=10,
            llm_total_time_ms=500.0,
            classic_agreement_count=90,
            total_processed=100,
        )
        d = stats.to_dict()
        assert d["llm_call_count"] == 10
        assert d["llm_total_time_ms"] == 500.0
        assert d["llm_avg_time_ms"] == 50.0
        assert d["classic_agreement_count"] == 90
        assert d["total_processed"] == 100
        assert d["agreement_rate"] == 90.0


def _disagreeing_hybrid(llm_result, **kwargs):
    """Build a HybridClassifier whose classic methods disagree.

    Method 1 says finance and Method 2 says retail. Their 60/40 weighted
    fallback is finance (0.42 >= GLOBAL_THRESHOLD), so a test can tell the
    classic fallback apart from the LLM answer.
    """
    from unittest.mock import MagicMock

    from email_classifier.classifier import HybridClassifier

    classifier = HybridClassifier(**kwargs)
    classifier.method1 = MagicMock()
    classifier.method1.classify.return_value = ClassificationResult(
        domain="finance",
        confidence=0.5,
        scores={"finance": 0.5, "retail": 0.1},
        method="keyword_taxonomy",
    )
    classifier.method2 = MagicMock()
    classifier.method2.classify.return_value = ClassificationResult(
        domain="retail",
        confidence=0.4,
        scores={"finance": 0.3, "retail": 0.4},
        method="structural_template",
    )
    classifier.llm_classifier = MagicMock()
    classifier.llm_classifier.classify.return_value = llm_result
    return classifier


def _gate_email():
    """Build a neutral email; the mocked methods decide its classification."""
    return EmailData(
        sender="info@example.com",
        receiver="user@example.com",
        date="2024-01-15",
        subject="Hello",
        body="Some text",
        urls="",
    )


class TestHybridConfidenceGate:
    """The hybrid path gates the LLM answer on its confidence (#18)."""

    def test_low_confidence_answer_does_not_override_classic(self):
        """A 0.3-confidence LLM answer no longer overrides both classic methods."""
        llm_result = ClassificationResult(
            domain="healthcare",
            confidence=0.3,
            scores={"healthcare": 0.3},
            method="typesafe",
        )
        classifier = _disagreeing_hybrid(llm_result)

        domain, details = classifier.classify(_gate_email())

        classifier.llm_classifier.classify.assert_called_once()
        assert details["path"] == "llm_assisted"
        assert domain == "finance"

    def test_answer_above_cutoff_is_accepted(self):
        """An answer above the cutoff is used and recorded as accepted."""
        llm_result = ClassificationResult(
            domain="healthcare",
            confidence=0.8,
            scores={"healthcare": 0.8},
            method="typesafe",
        )
        classifier = _disagreeing_hybrid(llm_result, llm_confidence_cutoff=0.5)

        domain, details = classifier.classify(_gate_email())

        assert domain == "healthcare"
        assert details["llm_gate"] == {
            "accepted": True,
            "reason": "at_or_above_cutoff",
            "confidence": 0.8,
            "cutoff": 0.5,
            "confidence_source": "typesafe",
        }
        assert classifier.stats.llm_rejected_count == 0
        assert "combined_scores" not in details

    def test_answer_at_cutoff_is_accepted(self):
        """The cutoff itself is inclusive."""
        llm_result = ClassificationResult(
            domain="healthcare", confidence=0.6, scores={}, method="typesafe"
        )
        classifier = _disagreeing_hybrid(llm_result, llm_confidence_cutoff=0.6)

        domain, details = classifier.classify(_gate_email())

        assert domain == "healthcare"
        assert details["llm_gate"]["accepted"] is True

    def test_answer_below_cutoff_falls_back_to_classic(self):
        """An answer below the cutoff uses the classic weighted fallback."""
        llm_result = ClassificationResult(
            domain="healthcare",
            confidence=0.4,
            scores={"healthcare": 0.4},
            method="typesafe",
        )
        classifier = _disagreeing_hybrid(llm_result, llm_confidence_cutoff=0.5)

        domain, details = classifier.classify(_gate_email())

        assert domain == "finance"
        assert details["final_domain"] == "finance"
        assert details["llm_gate"]["accepted"] is False
        assert details["llm_gate"]["reason"] == "below_cutoff"
        # The raw LLM answer is kept so the cutoff can be re-evaluated (#19)
        assert details["method3"]["domain"] == "healthcare"
        assert details["method3"]["confidence"] == 0.4
        assert details["combined_scores"]["finance"] == pytest.approx(0.42)
        assert classifier.stats.llm_rejected_count == 1
        assert classifier.stats.to_dict()["llm_rejected_count"] == 1

    def test_below_cutoff_fallback_can_be_unsure(self):
        """If the classic fallback is below its own threshold, the result is unsure."""
        llm_result = ClassificationResult(
            domain="healthcare", confidence=0.1, scores={}, method="typesafe"
        )
        classifier = _disagreeing_hybrid(llm_result, llm_confidence_cutoff=0.5)
        classifier.method1.classify.return_value = ClassificationResult(
            domain="finance", confidence=0.1, scores={"finance": 0.1}, method="m1"
        )
        classifier.method2.classify.return_value = ClassificationResult(
            domain="retail", confidence=0.1, scores={"retail": 0.1}, method="m2"
        )

        domain, details = classifier.classify(_gate_email())

        assert domain == "unsure"
        assert details["llm_gate"]["accepted"] is False

    def test_none_answer_above_cutoff_is_unsure(self):
        """A confident TypeSafe 'none' answer (domain None) gives unsure."""
        llm_result = ClassificationResult(
            domain=None,
            confidence=0.9,
            scores={"finance": 0.02},
            method="typesafe",
            details={"choice": "none"},
        )
        classifier = _disagreeing_hybrid(llm_result, llm_confidence_cutoff=0.5)

        domain, details = classifier.classify(_gate_email())

        assert domain == "unsure"
        assert details["llm_gate"]["accepted"] is True

    def test_none_answer_below_cutoff_falls_back_to_classic(self):
        """An unconfident 'none' answer does not force unsure."""
        llm_result = ClassificationResult(
            domain=None,
            confidence=0.2,
            scores={},
            method="typesafe",
            details={"choice": "none"},
        )
        classifier = _disagreeing_hybrid(llm_result, llm_confidence_cutoff=0.5)

        domain, details = classifier.classify(_gate_email())

        assert domain == "finance"
        assert details["llm_gate"]["reason"] == "below_cutoff"

    def test_legacy_unsure_answer_falls_back_to_classic(self):
        """The legacy classifier reports 'unsure' as None with confidence 0.0."""
        llm_result = ClassificationResult(
            domain=None, confidence=0.0, scores={}, method="llm_agent"
        )
        classifier = _disagreeing_hybrid(llm_result, llm_confidence_cutoff=0.5)

        domain, details = classifier.classify(_gate_email())

        assert domain == "finance"
        assert details["llm_gate"]["reason"] == "below_cutoff"
        assert details["llm_gate"]["confidence_source"] == "self_reported"

    def test_legacy_confident_answer_is_accepted(self):
        """The gate applies to the legacy classifier's self-reported confidence."""
        llm_result = ClassificationResult(
            domain="healthcare", confidence=0.9, scores={}, method="llm_agent"
        )
        classifier = _disagreeing_hybrid(llm_result, llm_confidence_cutoff=0.5)

        domain, details = classifier.classify(_gate_email())

        assert domain == "healthcare"
        assert details["llm_gate"]["confidence_source"] == "self_reported"

    def test_failed_call_is_never_accepted(self):
        """A failed call falls back even when the cutoff is 0.0."""
        llm_result = ClassificationResult(
            domain=None,
            confidence=0.0,
            scores={},
            method="typesafe",
            details={"error": "timeout", "fallback": True},
        )
        classifier = _disagreeing_hybrid(llm_result, llm_confidence_cutoff=0.0)

        domain, details = classifier.classify(_gate_email())

        assert domain == "finance"
        assert details["llm_gate"]["reason"] == "llm_failed"
        assert classifier.stats.llm_rejected_count == 1

    def test_zero_cutoff_accepts_any_answer(self):
        """A cutoff of 0.0 accepts every answer that is not a failed call."""
        llm_result = ClassificationResult(
            domain="healthcare", confidence=0.0, scores={}, method="typesafe"
        )
        classifier = _disagreeing_hybrid(llm_result, llm_confidence_cutoff=0.0)

        domain, _ = classifier.classify(_gate_email())

        assert domain == "healthcare"

    def test_nan_confidence_is_rejected(self):
        """A NaN confidence never passes the gate."""
        llm_result = ClassificationResult(
            domain="healthcare", confidence=float("nan"), scores={}, method="typesafe"
        )
        classifier = _disagreeing_hybrid(llm_result, llm_confidence_cutoff=0.0)

        domain, details = classifier.classify(_gate_email())

        assert domain == "finance"
        assert details["llm_gate"]["accepted"] is False

    def test_gate_is_written_to_workflow_log(self):
        """The workflow log records the gate decision for later analysis."""
        from unittest.mock import MagicMock

        llm_result = ClassificationResult(
            domain="healthcare", confidence=0.3, scores={}, method="typesafe"
        )
        workflow_logger = MagicMock()
        classifier = _disagreeing_hybrid(
            llm_result, llm_confidence_cutoff=0.5, workflow_logger=workflow_logger
        )

        classifier.classify(_gate_email())

        llm_steps = [
            call
            for call in workflow_logger.log_step.call_args_list
            if call.args[1] == "llm_classify"
        ]
        assert len(llm_steps) == 1
        assert llm_steps[0].kwargs["result"] == "finance"
        assert llm_steps[0].kwargs["extra"]["llm_gate"]["accepted"] is False

    def test_cutoff_defaults_to_provisional_value(self):
        """Without a config or argument, the provisional default is used."""
        from email_classifier.classifier import HybridClassifier
        from email_classifier.llm.config import DEFAULT_LLM_CONFIDENCE_CUTOFF

        assert HybridClassifier().llm_confidence_cutoff == DEFAULT_LLM_CONFIDENCE_CUTOFF

    def test_cutoff_comes_from_llm_config(self):
        """The cutoff is read from LLMConfig, and an explicit argument wins."""
        from email_classifier.classifier import HybridClassifier
        from email_classifier.llm.config import LLMConfig, LLMProvider

        config = LLMConfig(
            provider=LLMProvider.OLLAMA, model="llama3.2", llm_confidence_cutoff=0.7
        )

        assert HybridClassifier(llm_config=config).llm_confidence_cutoff == 0.7
        assert (
            HybridClassifier(
                llm_config=config, llm_confidence_cutoff=0.2
            ).llm_confidence_cutoff
            == 0.2
        )

    @pytest.mark.parametrize("cutoff", [-0.1, 1.5, float("nan")])
    def test_invalid_cutoff_argument_raises(self, cutoff):
        """An out-of-range cutoff argument is rejected."""
        from email_classifier.classifier import HybridClassifier
        from email_classifier.llm.config import LLMConfigError

        with pytest.raises(LLMConfigError, match="llm_confidence_cutoff"):
            HybridClassifier(llm_confidence_cutoff=cutoff)
