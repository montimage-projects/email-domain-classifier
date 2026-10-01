"""
Email classifier implementing multiple classification methods:
- Method 1: Keyword Taxonomy Matching
- Method 2: Structural Template Matching
- Method 3: LLM-based Classification (optional)
"""

import json
import logging
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import IO, TYPE_CHECKING, Any, Dict, List, Optional, Protocol, Tuple

from .domains import DOMAINS, DomainProfile, get_domain_names
from .parsing import parse_url_flag

if TYPE_CHECKING:
    from .llm import LLMConfig, Method3Classifier

logger = logging.getLogger(__name__)


class StatusCallback(Protocol):
    """Protocol for status update callbacks."""

    def __call__(self, message: str) -> None:
        """Update status with message."""
        ...


class HybridWorkflowLogger:
    """Structured JSON logger for hybrid workflow steps."""

    def __init__(self, log_file: str | None = None) -> None:
        """Initialize the hybrid workflow logger.

        Args:
            log_file: Optional path to log file. If None, uses Python logging.
        """
        self.log_file = log_file
        self._file_handle: IO[str] | None = None
        if log_file:
            self._file_handle = open(log_file, "a", encoding="utf-8")

    def log_step(
        self,
        email_idx: int,
        step: str,
        result: str | None = None,
        path: str | None = None,
        llm_time_ms: float | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        """Log a workflow step as JSON.

        Args:
            email_idx: Index of the email being processed.
            step: Step name (keyword_classify, structural_classify, agreement_check,
                  llm_classify, final_result).
            result: Classification result domain.
            path: Path taken (classic_only or llm_assisted).
            llm_time_ms: LLM response time in milliseconds.
            extra: Additional data to include.
        """
        entry: dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
            "email_idx": email_idx,
            "step": step,
        }
        if result is not None:
            entry["result"] = result
        if path is not None:
            entry["path"] = path
        if llm_time_ms is not None:
            entry["llm_time_ms"] = round(llm_time_ms, 2)
        if extra:
            entry.update(extra)

        self.emit_entry(entry)

    def emit_entry(self, entry: dict[str, Any]) -> None:
        """Emit a complete entry (also used for ordered coordinator replay)."""
        json_line = json.dumps(entry)

        if self._file_handle:
            self._file_handle.write(json_line + "\n")
            self._file_handle.flush()
        else:
            logger.info(f"[HybridWorkflow] {json_line}")

    def close(self) -> None:
        """Close the log file if open."""
        if self._file_handle:
            self._file_handle.close()
            self._file_handle = None


@dataclass
class ClassificationResult:
    """Result of a single classification method."""

    domain: str | None
    confidence: float
    scores: dict[str, float]
    method: str
    details: dict[str, Any] | None = field(default=None)


@dataclass
class EmailData:
    """Parsed email data structure."""

    sender: str
    receiver: str
    date: str
    subject: str
    body: str
    urls: str

    @property
    def has_url(self) -> bool:
        """Compatibility property for classification logic.

        Interprets flag spellings through :func:`parse_url_flag`, so raw
        ``'0'``/``'1'`` values (as in the CEAS_08 ``urls`` column) and
        ``'false'``/``'no'``/``'off'``/blank all read as false.
        """
        return parse_url_flag(self.urls)

    @classmethod
    def from_dict(cls, data: dict) -> "EmailData":
        """Create EmailData from dictionary."""
        # Handle both 'urls' and 'has_url' fields
        urls_value = data.get("urls", "")
        if not urls_value and "has_url" in data:
            # Convert has_url flag to the urls field's true-spelling
            urls_value = "true" if parse_url_flag(data.get("has_url")) else ""

        return cls(
            sender=str(data.get("sender", "")).lower().strip(),
            receiver=str(data.get("receiver", "")).lower().strip(),
            date=str(data.get("date", "")).strip(),
            subject=str(data.get("subject", "")).strip(),
            body=str(data.get("body", "")).strip(),
            urls=str(urls_value).strip(),
        )


class KeywordTaxonomyClassifier:
    """
    Method 1: Keyword Taxonomy Matching

    Scores emails based on keyword frequency and pattern matching
    against domain-specific taxonomies.
    """

    # Weights for different scoring components
    WEIGHTS = {
        "primary_keyword": 3.0,
        "secondary_keyword": 1.5,
        "sender_pattern": 4.0,
        "subject_pattern": 3.5,
        "subject_keyword": 2.5,
        "body_keyword_density": 2.0,
    }

    def __init__(self, domains: dict[str, DomainProfile] | None = None) -> None:
        self.domains = domains or DOMAINS
        self._compile_patterns()

    def _compile_patterns(self) -> None:
        """Pre-compile regex patterns for efficiency."""
        self._sender_patterns = {}
        self._subject_patterns = {}

        for name, profile in self.domains.items():
            self._sender_patterns[name] = [
                re.compile(p, re.IGNORECASE) for p in profile.sender_patterns
            ]
            self._subject_patterns[name] = [
                re.compile(p, re.IGNORECASE) for p in profile.subject_patterns
            ]

    def classify(self, email: EmailData) -> ClassificationResult:
        """Classify email using keyword taxonomy method."""
        scores = {}
        details = {}

        for domain_name, profile in self.domains.items():
            score, domain_details = self._score_domain(email, domain_name, profile)
            scores[domain_name] = score
            details[domain_name] = domain_details

        # Normalize scores
        total = sum(scores.values())
        if total > 0:
            normalized = {k: v / total for k, v in scores.items()}
        else:
            normalized = scores

        # Find best match
        best_domain: str | None = None
        confidence: float = 0.0
        if scores:
            best_domain = max(scores, key=lambda k: scores[k])
            best_score = scores[best_domain]
            confidence = normalized.get(best_domain, 0)

            # Require minimum confidence threshold
            if confidence < 0.05 or best_score < 1.5:
                best_domain = None
                confidence = 0.0

        return ClassificationResult(
            domain=best_domain,
            confidence=confidence,
            scores=normalized,
            method="keyword_taxonomy",
            details=details,
        )

    def _score_domain(
        self, email: EmailData, domain_name: str, profile: DomainProfile
    ) -> tuple[float, dict[str, Any]]:
        """Calculate score for a specific domain."""
        score = 0.0
        details: dict[str, Any] = {
            "primary_matches": [],
            "secondary_matches": [],
            "sender_match": False,
            "subject_pattern_match": False,
        }

        # Check sender patterns
        for pattern in self._sender_patterns[domain_name]:
            if pattern.match(email.sender):
                score += self.WEIGHTS["sender_pattern"]
                details["sender_match"] = True
                break

        # Check subject patterns
        for pattern in self._subject_patterns[domain_name]:
            if pattern.search(email.subject):
                score += self.WEIGHTS["subject_pattern"]
                details["subject_pattern_match"] = True
                break

        # Check subject keywords
        subject_lower = email.subject.lower()
        for keyword in profile.primary_keywords:
            if keyword in subject_lower:
                score += self.WEIGHTS["subject_keyword"]
                details["primary_matches"].append(f"subject:{keyword}")

        # Check body keywords
        body_lower = email.body.lower()
        body_words = len(body_lower.split())

        primary_count = 0
        for keyword in profile.primary_keywords:
            if keyword in body_lower:
                primary_count += body_lower.count(keyword)
                details["primary_matches"].append(f"body:{keyword}")

        secondary_count = 0
        for keyword in profile.secondary_keywords:
            if keyword in body_lower:
                secondary_count += body_lower.count(keyword)
                details["secondary_matches"].append(keyword)

        # Calculate keyword density score
        if body_words > 0:
            primary_density = (primary_count / body_words) * 100
            secondary_density = (secondary_count / body_words) * 100

            score += min(primary_density * self.WEIGHTS["primary_keyword"], 15)
            score += min(secondary_density * self.WEIGHTS["secondary_keyword"], 8)

        return score, details


class StructuralTemplateClassifier:
    """
    Method 2: Structural Template Matching

    Analyzes email structure and format characteristics
    to match against domain templates.
    """

    # Structural feature weights
    WEIGHTS = {
        "body_length": 2.0,
        "greeting": 1.5,
        "signature": 1.5,
        "disclaimer": 2.0,
        "url_match": 2.5,
        "formality": 2.0,
        "paragraph_count": 1.5,
        "sender_structure": 2.5,
    }

    # Greeting patterns
    GREETING_PATTERNS = [
        re.compile(
            r"^(dear|hello|hi|greetings|good\s+(morning|afternoon|evening))",
            re.IGNORECASE | re.MULTILINE,
        ),
    ]

    # Signature patterns
    SIGNATURE_PATTERNS = [
        re.compile(
            r"(sincerely|regards|best|thank you|thanks|cheers),?\s*\n", re.IGNORECASE
        ),
        re.compile(r"\n[-–—]{2,}\s*\n", re.MULTILINE),
        re.compile(r"(sent from my|get outlook)", re.IGNORECASE),
    ]

    # Disclaimer patterns
    DISCLAIMER_PATTERNS = [
        re.compile(
            r"(confidential|disclaimer|privileged|intended recipient)", re.IGNORECASE
        ),
        re.compile(r"(this (email|message|communication) (is|may be))", re.IGNORECASE),
        re.compile(r"(do not (distribute|forward|share))", re.IGNORECASE),
    ]

    # Formality indicators
    FORMAL_INDICATORS = [
        "pursuant",
        "hereby",
        "aforementioned",
        "regarding",
        "enclosed",
        "please find",
        "kindly",
        "respectfully",
        "we regret",
        "be advised",
    ]

    CASUAL_INDICATORS = [
        "hey",
        "thanks!",
        "awesome",
        "cool",
        "btw",
        "fyi",
        "asap",
        "gonna",
        "wanna",
        "gotta",
        ":)",
        "!!",
    ]

    def __init__(self, domains: dict[str, DomainProfile] | None = None) -> None:
        self.domains = domains or DOMAINS

    def classify(self, email: EmailData) -> ClassificationResult:
        """Classify email using structural template matching."""
        # Extract structural features
        features = self._extract_features(email)

        scores = {}
        details = {}

        for domain_name, profile in self.domains.items():
            score, domain_details = self._score_template_match(features, profile)
            scores[domain_name] = score
            details[domain_name] = domain_details

        # Normalize scores
        total = sum(scores.values())
        if total > 0:
            normalized = {k: v / total for k, v in scores.items()}
        else:
            normalized = scores

        # Find best match
        best_domain: str | None = None
        confidence: float = 0.0
        if scores:
            best_domain = max(scores, key=lambda k: scores[k])
            best_score = scores[best_domain]
            confidence = normalized.get(best_domain, 0)

            # Require minimum threshold
            if confidence < 0.04 or best_score < 2.0:
                best_domain = None
                confidence = 0.0

        return ClassificationResult(
            domain=best_domain,
            confidence=confidence,
            scores=normalized,
            method="structural_template",
            details={"features": features, "domain_scores": details},
        )

    def _extract_features(self, email: EmailData) -> dict[str, Any]:
        """Extract structural features from email."""
        body = email.body

        # Body length
        body_length = len(body)

        # Check for greeting
        has_greeting = any(p.search(body) for p in self.GREETING_PATTERNS)

        # Check for signature
        has_signature = any(p.search(body) for p in self.SIGNATURE_PATTERNS)

        # Check for disclaimer
        has_disclaimer = any(p.search(body) for p in self.DISCLAIMER_PATTERNS)

        # Count paragraphs (separated by blank lines)
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
        paragraph_count = len(paragraphs)

        # Assess formality
        body_lower = body.lower()
        formal_count = sum(1 for ind in self.FORMAL_INDICATORS if ind in body_lower)
        casual_count = sum(1 for ind in self.CASUAL_INDICATORS if ind in body_lower)

        if formal_count > casual_count + 1:
            formality = "formal"
        elif casual_count > formal_count + 1:
            formality = "casual"
        else:
            formality = "semi-formal"

        # Analyze sender structure
        sender_features = self._analyze_sender_structure(email.sender)

        return {
            "body_length": body_length,
            "has_greeting": has_greeting,
            "has_signature": has_signature,
            "has_disclaimer": has_disclaimer,
            "paragraph_count": paragraph_count,
            "formality": formality,
            "has_url": email.has_url,
            "sender": sender_features,
        }

    def _analyze_sender_structure(self, sender: str) -> dict[str, Any]:
        """Analyze sender address structure."""
        features: dict[str, Any] = {
            "is_noreply": False,
            "has_department": False,
            "domain_type": "unknown",
        }

        # Check for noreply pattern
        if re.search(r"no.?reply|donotreply", sender, re.IGNORECASE):
            features["is_noreply"] = True

        # Check for department indicators
        dept_patterns = [
            "support",
            "billing",
            "sales",
            "info",
            "contact",
            "admin",
            "help",
            "service",
            "team",
            "notifications",
        ]
        local_part = sender.split("@")[0] if "@" in sender else sender
        features["has_department"] = any(d in local_part for d in dept_patterns)

        # Determine domain type
        if ".gov" in sender:
            features["domain_type"] = "government"
        elif ".edu" in sender:
            features["domain_type"] = "education"
        elif any(x in sender for x in [".org", ".net", ".com"]):
            features["domain_type"] = "commercial"

        return features

    def _score_template_match(
        self, features: dict[str, Any], profile: DomainProfile
    ) -> tuple[float, dict[str, Any]]:
        """Score how well features match a domain template."""
        score = 0.0
        details: dict[str, Any] = {}

        # Body length match
        min_len, max_len = profile.typical_body_length
        body_len = features["body_length"]

        if min_len <= body_len <= max_len:
            score += self.WEIGHTS["body_length"]
            details["body_length"] = "match"
        elif body_len < min_len * 0.5 or body_len > max_len * 2:
            details["body_length"] = "mismatch"
        else:
            score += self.WEIGHTS["body_length"] * 0.5
            details["body_length"] = "partial"

        # Greeting match
        if features["has_greeting"] == profile.has_greeting:
            score += self.WEIGHTS["greeting"]
            details["greeting"] = "match"
        else:
            details["greeting"] = "mismatch"

        # Signature match
        if features["has_signature"] == profile.has_signature:
            score += self.WEIGHTS["signature"]
            details["signature"] = "match"
        else:
            details["signature"] = "mismatch"

        # Disclaimer match
        if features["has_disclaimer"] == profile.has_disclaimer:
            score += self.WEIGHTS["disclaimer"]
            details["disclaimer"] = "match"
        elif profile.has_disclaimer and features["has_disclaimer"]:
            score += self.WEIGHTS["disclaimer"]
            details["disclaimer"] = "match"
        else:
            details["disclaimer"] = "mismatch"

        # URL expectation match
        if features["has_url"] == profile.url_expected:
            score += self.WEIGHTS["url_match"]
            details["url"] = "match"
        else:
            score += self.WEIGHTS["url_match"] * 0.3
            details["url"] = "partial"

        # Formality match
        if features["formality"] == profile.formality_level:
            score += self.WEIGHTS["formality"]
            details["formality"] = "match"
        elif (
            features["formality"] == "semi-formal"
            or profile.formality_level == "semi-formal"
        ):
            score += self.WEIGHTS["formality"] * 0.5
            details["formality"] = "partial"
        else:
            details["formality"] = "mismatch"

        # Paragraph count match
        min_para, max_para = profile.typical_paragraph_count
        para_count = features["paragraph_count"]

        if min_para <= para_count <= max_para:
            score += self.WEIGHTS["paragraph_count"]
            details["paragraphs"] = "match"
        elif para_count < min_para * 0.5 or para_count > max_para * 2:
            details["paragraphs"] = "mismatch"
        else:
            score += self.WEIGHTS["paragraph_count"] * 0.5
            details["paragraphs"] = "partial"

        # Sender structure bonuses
        sender_features = features["sender"]
        if sender_features["domain_type"] == profile.name:
            score += self.WEIGHTS["sender_structure"]
            details["sender"] = "domain_match"
        elif sender_features["is_noreply"] and profile.name in [
            "technology",
            "retail",
            "logistics",
        ]:
            score += self.WEIGHTS["sender_structure"] * 0.5
            details["sender"] = "noreply_match"
        else:
            details["sender"] = "no_match"

        return score, details


class EmailClassifier:
    """
    Main classifier combining multiple methods.

    Classification logic:
    - Run Method 1 (Keyword Taxonomy)
    - Run Method 2 (Structural Template)
    - Run Method 3 (LLM) if enabled
    - Calculate weighted combined score
    - Assign domain with highest score if above threshold
    """

    # Default weights (without LLM)
    DEFAULT_WEIGHT_METHOD_1 = 0.6  # Keywords
    DEFAULT_WEIGHT_METHOD_2 = 0.4  # Structure

    # Weights with LLM enabled
    LLM_WEIGHT_METHOD_1 = 0.35  # Keywords
    LLM_WEIGHT_METHOD_2 = 0.25  # Structure
    LLM_WEIGHT_METHOD_3 = 0.40  # LLM

    GLOBAL_THRESHOLD = 0.15

    def __init__(
        self,
        domains: dict[str, DomainProfile] | None = None,
        llm_config: Optional["LLMConfig"] = None,
        use_llm: bool = False,
    ) -> None:
        """Initialize the classifier.

        Args:
            domains: Optional custom domain profiles. Uses defaults if None.
            llm_config: Configuration for LLM classifier. If provided, enables LLM.
            use_llm: Enable LLM classification. Requires llm_config or will
                    load from environment.
        """
        self.domains = domains or DOMAINS
        self.method1 = KeywordTaxonomyClassifier(self.domains)
        self.method2 = StructuralTemplateClassifier(self.domains)

        # LLM classifier (optional)
        self.method3: Optional["Method3Classifier"] = None
        self._llm_config: Optional["LLMConfig"] = llm_config

        # Set weights based on LLM availability
        if use_llm or llm_config is not None:
            self._init_llm_classifier(llm_config)

        self._update_weights()
        # Identity guard: parallel cloning must not discard injected methods.
        self._worker_methods = (self.method1, self.method2, self.method3)

    def _init_llm_classifier(self, config: Optional["LLMConfig"] = None) -> None:
        """Initialize the LLM classifier.

        Args:
            config: LLM configuration. If None, loads from environment.
        """
        try:
            from .llm import LLMConfig, create_classifier

            if config is None:
                config = LLMConfig.from_env()

            self._llm_config = config
            self.method3 = create_classifier(config)
            logger.info(
                f"LLM classifier initialized with {config.provider.value}/{config.model}"
            )
        except ImportError as e:
            logger.warning(f"LLM dependencies not installed: {e}")
            self.method3 = None
        except Exception as e:
            logger.warning(f"Failed to initialize LLM classifier: {e}")
            self.method3 = None

    def _update_weights(self) -> None:
        """Update method weights based on LLM availability."""
        if self.method3 is not None and self._llm_config is not None:
            # Use configurable weights from LLM config
            self.weight_method_1 = self._llm_config.keyword_weight
            self.weight_method_2 = self._llm_config.structural_weight
            self.weight_method_3 = self._llm_config.llm_weight
        else:
            # Default dual-method weights
            self.weight_method_1 = self.DEFAULT_WEIGHT_METHOD_1
            self.weight_method_2 = self.DEFAULT_WEIGHT_METHOD_2
            self.weight_method_3 = 0.0

    @property
    def llm_enabled(self) -> bool:
        """Check if LLM classification is enabled."""
        return self.method3 is not None

    def classify(self, email: EmailData) -> tuple[str, dict[str, Any]]:
        """
        Classify email using multi-method validation with weighted scoring.

        When LLM is enabled, combines three methods:
        - Method 1: Keyword Taxonomy (default 35%)
        - Method 2: Structural Template (default 25%)
        - Method 3: LLM Agent (default 40%)

        When LLM is disabled, uses dual-method classification:
        - Method 1: Keyword Taxonomy (60%)
        - Method 2: Structural Template (40%)

        Returns:
            Tuple of (domain_name or 'unsure', classification_details)
        """
        result1 = self.method1.classify(email)
        result2 = self.method2.classify(email)

        # Initialize details
        details: dict[str, Any] = {
            "method1": {
                "domain": result1.domain,
                "confidence": result1.confidence,
                "scores": result1.scores,
            },
            "method2": {
                "domain": result2.domain,
                "confidence": result2.confidence,
                "scores": result2.scores,
            },
            "method_weights": {
                "method1": self.weight_method_1,
                "method2": self.weight_method_2,
            },
            "llm_enabled": self.llm_enabled,
        }

        # Run LLM classification if enabled
        result3: ClassificationResult | None = None
        if self.method3 is not None:
            try:
                result3 = self.method3.classify(email)
                details["method3"] = {
                    "domain": result3.domain,
                    "confidence": result3.confidence,
                    "scores": result3.scores,
                    "details": result3.details,
                }
                details["method_weights"]["method3"] = self.weight_method_3
            except Exception as e:
                logger.warning(f"LLM classification failed, falling back: {e}")
                details["method3"] = {
                    "error": str(e),
                    "fallback": True,
                }

        # Calculate combined scores. Iterate domains in sorted order so the
        # result does not depend on the process's string-hash seed: on a tie,
        # max() returns the alphabetically first domain, and combined_scores
        # is serialized in that same deterministic order.
        combined_scores: dict[str, float] = {}
        all_domains = set(result1.scores.keys()) | set(result2.scores.keys())
        if result3 is not None:
            all_domains |= set(result3.scores.keys())

        for domain in sorted(all_domains):
            score1 = result1.scores.get(domain, 0.0)
            score2 = result2.scores.get(domain, 0.0)

            combined_score = (score1 * self.weight_method_1) + (
                score2 * self.weight_method_2
            )

            # Add LLM score if available
            if result3 is not None:
                score3 = result3.scores.get(domain, 0.0)
                combined_score += score3 * self.weight_method_3

            combined_scores[domain] = combined_score

        details["combined_scores"] = combined_scores

        # Find best match
        if combined_scores:
            best_domain = max(combined_scores, key=lambda k: combined_scores[k])
            best_score = combined_scores[best_domain]

            details["final_confidence"] = best_score

            if best_score >= self.GLOBAL_THRESHOLD:
                final_domain = best_domain
            else:
                final_domain = "unsure"
                details["reason"] = "below_threshold"
        else:
            final_domain = "unsure"
            details["reason"] = "no_scores"

        return final_domain, details

    def classify_dict(self, email_dict: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        """Classify email from dictionary input."""
        email = EmailData.from_dict(email_dict)
        return self.classify(email)


@dataclass
class HybridWorkflowStats:
    """Statistics for hybrid classification workflow."""

    llm_call_count: int = 0
    llm_total_time_ms: float = 0.0
    classic_agreement_count: int = 0
    total_processed: int = 0
    total_processing_time_ms: float = 0.0  # Total time for all emails
    # LLM answers the confidence gate did not accept (issue #18)
    llm_rejected_count: int = 0

    @property
    def llm_avg_time_ms(self) -> float:
        """Calculate average LLM response time."""
        if self.llm_call_count == 0:
            return 0.0
        return self.llm_total_time_ms / self.llm_call_count

    @property
    def avg_time_per_email_ms(self) -> float:
        """Calculate average processing time per email."""
        if self.total_processed == 0:
            return 0.0
        return self.total_processing_time_ms / self.total_processed

    @property
    def agreement_rate(self) -> float:
        """Calculate classifier agreement rate (percentage)."""
        if self.total_processed == 0:
            return 0.0
        return (self.classic_agreement_count / self.total_processed) * 100

    def estimate_remaining_time_ms(self, remaining_emails: int) -> float:
        """Estimate time to process remaining emails."""
        return self.avg_time_per_email_ms * remaining_emails

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "llm_call_count": self.llm_call_count,
            "llm_total_time_ms": round(self.llm_total_time_ms, 2),
            "llm_avg_time_ms": round(self.llm_avg_time_ms, 2),
            "classic_agreement_count": self.classic_agreement_count,
            "total_processed": self.total_processed,
            "agreement_rate": round(self.agreement_rate, 2),
            "llm_rejected_count": self.llm_rejected_count,
            "avg_time_per_email_ms": round(self.avg_time_per_email_ms, 2),
            "total_processing_time_ms": round(self.total_processing_time_ms, 2),
        }


class HybridClassifier:
    """
    Hybrid classifier that runs classic classifiers first and only
    invokes LLM when they disagree.

    Workflow:
    1. Run Keyword Taxonomy classifier
    2. Run Structural Template classifier
    3. If both agree on domain, accept that result (skip LLM)
    4. If they disagree, invoke LLM for tie-breaking
    5. Accept the LLM answer only if its confidence is at least
       ``llm_confidence_cutoff``; otherwise use the classic weighted fallback
    """

    GLOBAL_THRESHOLD = 0.15

    # ClassificationResult.method of the TypeSafe classifier, whose confidence
    # comes from its Choice distribution. Every other Method 3 result carries a
    # confidence the model reported about itself.
    TYPESAFE_METHOD = "typesafe"

    def __init__(
        self,
        llm_config: Optional["LLMConfig"] = None,
        domains: dict[str, DomainProfile] | None = None,
        status_callback: Callable[[str], None] | None = None,
        workflow_logger: HybridWorkflowLogger | None = None,
        llm_confidence_cutoff: float | None = None,
    ) -> None:
        """Initialize the hybrid classifier.

        Args:
            llm_config: Configuration for LLM classifier.
            domains: Optional custom domain profiles.
            status_callback: Optional callback for status updates.
            workflow_logger: Optional logger for structured workflow logging.
            llm_confidence_cutoff: Minimum LLM confidence accepted when the
                classic methods disagree, from 0.0 to 1.0. Defaults to
                ``llm_config.llm_confidence_cutoff``, or to
                ``DEFAULT_LLM_CONFIDENCE_CUTOFF`` (0.5, evaluated in #19) without
                a config.

        Raises:
            LLMConfigError: If ``llm_confidence_cutoff`` is outside 0.0 to 1.0.
        """
        from .llm.config import (
            DEFAULT_LLM_CONFIDENCE_CUTOFF,
            validate_confidence_cutoff,
        )

        self.domains = domains or DOMAINS
        self.method1 = KeywordTaxonomyClassifier(self.domains)
        self.method2 = StructuralTemplateClassifier(self.domains)
        self.status_callback = status_callback
        self.workflow_logger = workflow_logger
        self.stats = HybridWorkflowStats()

        if llm_confidence_cutoff is None:
            llm_confidence_cutoff = (
                llm_config.llm_confidence_cutoff
                if llm_config is not None
                else DEFAULT_LLM_CONFIDENCE_CUTOFF
            )
        self.llm_confidence_cutoff = validate_confidence_cutoff(llm_confidence_cutoff)

        # Initialize LLM classifier
        self.llm_classifier: Optional["Method3Classifier"] = None
        self._llm_config = llm_config
        if llm_config is not None:
            self._init_llm_classifier(llm_config)
        # Identity guard: custom methods require an explicit worker factory.
        self._worker_methods = (self.method1, self.method2, self.llm_classifier)

    def _init_llm_classifier(self, config: "LLMConfig") -> None:
        """Initialize the LLM classifier."""
        try:
            from .llm import create_classifier

            self.llm_classifier = create_classifier(config)
            logger.info(
                f"Hybrid classifier LLM initialized: {config.provider.value}/{config.model}"
            )
        except ImportError as e:
            logger.warning(f"LLM dependencies not installed: {e}")
            self.llm_classifier = None
        except Exception as e:
            logger.warning(f"Failed to initialize LLM classifier: {e}")
            self.llm_classifier = None

    def _format_time(self, ms: float) -> str:
        """Format milliseconds to human-readable time."""
        if ms < 1000:
            return f"{ms:.0f}ms"
        seconds = ms / 1000
        if seconds < 60:
            return f"{seconds:.1f}s"
        minutes = seconds / 60
        if minutes < 60:
            return f"{minutes:.1f}m"
        hours = minutes / 60
        return f"{hours:.1f}h"

    def _update_status(
        self, message: str, email_idx: int = 0, total_emails: int = 0
    ) -> None:
        """Send status update if callback is set."""
        if self.status_callback:
            if total_emails > 0:
                # Build rich status with stats
                # Note: stats are from PREVIOUS emails (current one not yet finished)
                remaining = total_emails - email_idx  # Current email not done yet
                parts = [f"Email {email_idx + 1}/{total_emails}"]

                # Add average time per email (need at least 1 completed email)
                # Use total_processed - 1 because current email incremented it but isn't done
                completed_count = self.stats.total_processed - 1
                if completed_count > 0 and self.stats.total_processing_time_ms > 0:
                    avg_ms = self.stats.total_processing_time_ms / completed_count
                    avg_time = self._format_time(avg_ms)
                    parts.append(f"avg:{avg_time}")

                    # Add ETA based on completed emails' average
                    if remaining > 0:
                        eta_ms = avg_ms * remaining
                        eta = self._format_time(eta_ms)
                        parts.append(f"ETA:{eta}")

                # Add LLM call count (only if > 0)
                if self.stats.llm_call_count > 0:
                    parts.append(f"LLM:{self.stats.llm_call_count}")

                # Combine: "Email 5/100 | avg:1.2s | ETA:2.5m | LLM:3 - waiting..."
                status_prefix = " | ".join(parts)
                self.status_callback(f"{status_prefix} - {message}")
            else:
                self.status_callback(message)

    def classify(
        self, email: EmailData, email_idx: int = 0, total_emails: int = 0
    ) -> tuple[str, dict[str, Any]]:
        """
        Classify email using hybrid workflow.

        Args:
            email: Email data to classify.
            email_idx: Index of email for logging.
            total_emails: Total number of emails for progress display.

        Returns:
            Tuple of (domain_name or 'unsure', classification_details)
        """
        # Start timing for this email
        email_start_time = time.perf_counter()

        self.stats.total_processed += 1

        # Step 1: Run Keyword Taxonomy classifier
        self._update_status(
            "Classifying with Keyword Taxonomy...", email_idx, total_emails
        )
        result1 = self.method1.classify(email)
        if self.workflow_logger:
            self.workflow_logger.log_step(
                email_idx, "keyword_classify", result=result1.domain
            )

        # Step 2: Run Structural Template classifier
        self._update_status(
            "Classifying with Structural Template...", email_idx, total_emails
        )
        result2 = self.method2.classify(email)
        if self.workflow_logger:
            self.workflow_logger.log_step(
                email_idx, "structural_classify", result=result2.domain
            )

        # Initialize details
        details: dict[str, Any] = {
            "method1": {
                "domain": result1.domain,
                "confidence": result1.confidence,
                "scores": result1.scores,
            },
            "method2": {
                "domain": result2.domain,
                "confidence": result2.confidence,
                "scores": result2.scores,
            },
            "hybrid_workflow": True,
        }

        # Step 3: Check agreement
        classifiers_agree = (
            result1.domain is not None
            and result2.domain is not None
            and result1.domain == result2.domain
        )

        if classifiers_agree:
            # Both classifiers agree - skip LLM
            final_domain = result1.domain
            self.stats.classic_agreement_count += 1
            details["path"] = "classic_only"
            details["agreement"] = True

            self._update_status(
                f"Classifiers agree - '{final_domain}'", email_idx, total_emails
            )
            if self.workflow_logger:
                self.workflow_logger.log_step(
                    email_idx,
                    "agreement_check",
                    result=final_domain,
                    path="classic_only",
                )
        else:
            # Classifiers disagree - invoke LLM
            details["path"] = "llm_assisted"
            details["agreement"] = False

            if self.llm_classifier is None:
                # No LLM available, fall back to weighted combination
                final_domain = self._fallback_classification(result1, result2, details)
                if self.workflow_logger:
                    self.workflow_logger.log_step(
                        email_idx,
                        "agreement_check",
                        result=final_domain,
                        path="llm_assisted",
                        extra={"llm_fallback": True, "reason": "no_llm_available"},
                    )
            else:
                self._update_status(
                    "Classifiers disagree - invoking LLM...", email_idx, total_emails
                )
                if self.workflow_logger:
                    self.workflow_logger.log_step(
                        email_idx, "agreement_check", path="llm_assisted"
                    )

                # Invoke LLM with timing
                start_time = time.perf_counter()
                try:
                    self._update_status("waiting for LLM...", email_idx, total_emails)
                    result3 = self.llm_classifier.classify(email)
                    elapsed_ms = (time.perf_counter() - start_time) * 1000

                    self.stats.llm_call_count += 1
                    self.stats.llm_total_time_ms += elapsed_ms

                    details["method3"] = {
                        "domain": result3.domain,
                        "confidence": result3.confidence,
                        "scores": result3.scores,
                        "response_time_ms": round(elapsed_ms, 2),
                    }

                    gate = self._gate_llm_result(result3)
                    details["llm_gate"] = gate

                    if gate["accepted"]:
                        # A None domain means no listed domain fits the email.
                        final_domain = result3.domain or "unsure"
                        status = (
                            f"LLM responded ({elapsed_ms:.0f}ms) - '{final_domain}'"
                        )
                    else:
                        self.stats.llm_rejected_count += 1
                        final_domain = self._fallback_classification(
                            result1, result2, details
                        )
                        status = (
                            f"LLM answer rejected ({gate['reason']}) - "
                            f"fallback '{final_domain}'"
                        )

                    self._update_status(status, email_idx, total_emails)
                    if self.workflow_logger:
                        self.workflow_logger.log_step(
                            email_idx,
                            "llm_classify",
                            result=final_domain,
                            llm_time_ms=elapsed_ms,
                            extra={"llm_gate": gate},
                        )

                except Exception as e:
                    elapsed_ms = (time.perf_counter() - start_time) * 1000
                    logger.warning(f"LLM classification failed: {e}")
                    details["method3"] = {
                        "error": str(e),
                        "response_time_ms": round(elapsed_ms, 2),
                    }
                    # Fall back to weighted combination
                    final_domain = self._fallback_classification(
                        result1, result2, details
                    )
                    if self.workflow_logger:
                        self.workflow_logger.log_step(
                            email_idx,
                            "llm_classify",
                            result=final_domain,
                            llm_time_ms=elapsed_ms,
                            extra={"error": str(e)},
                        )

        # Log final result
        if self.workflow_logger:
            self.workflow_logger.log_step(
                email_idx, "final_result", result=final_domain, path=details["path"]
            )

        # Record total processing time for this email
        email_elapsed_ms = (time.perf_counter() - email_start_time) * 1000
        self.stats.total_processing_time_ms += email_elapsed_ms

        # Ensure we always return a valid domain string
        final_result = final_domain if final_domain is not None else "unsure"
        details["final_domain"] = final_result
        details["processing_time_ms"] = round(email_elapsed_ms, 2)
        return final_result, details

    def _gate_llm_result(self, result: ClassificationResult) -> dict[str, Any]:
        """Decide whether the hybrid workflow accepts a Method 3 answer.

        An answer is accepted when its confidence is at least
        ``llm_confidence_cutoff``. A failed call (``details["fallback"]``) is
        never accepted, whatever the cutoff.

        The rule is the same for both Method 3 classifiers. TypeSafe's confidence
        comes from its Choice distribution. The legacy ``LLMClassifier`` uses the
        confidence the model reports about itself, which is not calibrated, and
        reports an "unsure" answer as confidence 0.0. ``confidence_source`` in the
        returned record tells the two apart so they can be analysed separately.

        Args:
            result: Method 3 classification result.

        Returns:
            Gate record with ``accepted``, ``reason`` (``at_or_above_cutoff``,
            ``below_cutoff`` or ``llm_failed``), ``confidence``, ``cutoff`` and
            ``confidence_source`` (``typesafe`` or ``self_reported``).
        """
        if result.details and result.details.get("fallback"):
            accepted, reason = False, "llm_failed"
        elif result.confidence >= self.llm_confidence_cutoff:
            accepted, reason = True, "at_or_above_cutoff"
        else:
            # Also covers a NaN confidence, which compares False.
            accepted, reason = False, "below_cutoff"

        return {
            "accepted": accepted,
            "reason": reason,
            "confidence": result.confidence,
            "cutoff": self.llm_confidence_cutoff,
            "confidence_source": (
                "typesafe" if result.method == self.TYPESAFE_METHOD else "self_reported"
            ),
        }

    def _fallback_classification(
        self,
        result1: ClassificationResult,
        result2: ClassificationResult,
        details: dict[str, Any],
    ) -> str:
        """Fall back to weighted combination when the LLM answer is not used.

        Used when no LLM is available, the LLM call raised, or the confidence
        gate rejected the LLM answer.
        """
        # Use 60/40 weighting like dual-method. Iterate domains in sorted
        # order so the result does not depend on the process's string-hash
        # seed: on a tie, max() returns the alphabetically first domain, and
        # combined_scores is serialized in that same deterministic order.
        combined_scores: dict[str, float] = {}
        all_domains = set(result1.scores.keys()) | set(result2.scores.keys())

        for domain in sorted(all_domains):
            score1 = result1.scores.get(domain, 0.0)
            score2 = result2.scores.get(domain, 0.0)
            combined_scores[domain] = (score1 * 0.6) + (score2 * 0.4)

        details["combined_scores"] = combined_scores

        if combined_scores:
            best_domain = max(combined_scores, key=lambda k: combined_scores[k])
            best_score = combined_scores[best_domain]
            details["final_confidence"] = best_score

            if best_score >= self.GLOBAL_THRESHOLD:
                return best_domain
            else:
                details["reason"] = "below_threshold"
                return "unsure"
        else:
            details["reason"] = "no_scores"
            return "unsure"

    def classify_dict(
        self, email_dict: dict[str, Any], email_idx: int = 0, total_emails: int = 0
    ) -> tuple[str, dict[str, Any]]:
        """Classify email from dictionary input."""
        email = EmailData.from_dict(email_dict)
        return self.classify(email, email_idx, total_emails)

    def get_stats(self) -> HybridWorkflowStats:
        """Get current workflow statistics."""
        return self.stats

    def reset_stats(self) -> None:
        """Reset workflow statistics."""
        self.stats = HybridWorkflowStats()
