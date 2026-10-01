"""TypeSafe-based email classifier (Method 3 alternative to LLMClassifier).

Asks TypeSafe one Choice question over the ten domains plus a ``none`` option.
The answer can only be one of those options, so no name-variant normalization
is needed, and the confidence comes from TypeSafe's probability distribution
instead of a self-reported number.

The question and option text is the "Text for the TypeSafe Choice question" block
in ``docs/design/domain-profiles.md`` (Domain definition, issue #15). A test keeps
the two in sync.
"""

import json
import logging
import math
from collections.abc import Mapping
from typing import Any, Optional
from urllib.request import Request, urlopen

from ..classifier import ClassificationResult, EmailData
from ..domains import get_domain_names
from .config import LLMConfig, LLMProvider
from .providers import ProviderNotInstalledError

logger = logging.getLogger(__name__)

# Question id used in the TypeSafe request. Ids are for code only and are not
# sent to the model.
QUESTION_ID = "domain"

# Option meaning "no listed sector fits".
NONE_OPTION = "none"

# Maximum number of body characters sent as state.
MAX_BODY_CHARS = 2000

# Classification method name reported in ClassificationResult.method.
METHOD_NAME = "typesafe"

DOMAIN_QUESTION = (
    "Which business sector does this email claim to come from? Judge only what the "
    "email claims about itself: the organization it presents itself as, or, if it "
    "names no organization, what it sells or asks the reader to do. Do not judge "
    "whether the email is genuine, spam or phishing: a fake bank email is finance, "
    "exactly like a real one. Use the subject and body first. Use the sender's name "
    "or address only if the subject and body give no sector. Ignore random or "
    "unrelated sender addresses, link hostnames and random filler text. A seller is "
    "classified by what it sells. A job offer, job posting or an employer's message "
    "to its own staff about employment is hr, whatever the employer's industry. "
    "Answer none only if no other option fits."
)

DOMAIN_OPTIONS: dict[str, str] = {
    "finance": (
        "Covers banks, credit cards, payment services such as PayPal, loans, "
        "mortgages, debt relief, investments, stock tips, non-health insurance and "
        "fund-transfer offers that are not job offers. Excludes health insurance, "
        "tax authorities, lotteries, casinos and job offers."
    ),
    "technology": (
        "Covers software companies, webmail and online-account providers that are "
        "not phone or internet carriers, cloud and hosting, antivirus, sellers of "
        "software or software licences, and software projects' development mailing "
        "lists and bug trackers. Excludes job offers from software companies, job "
        "lists run by software projects, phone carriers, internet providers, social "
        "networks and electronics sold by a shop."
    ),
    "retail": (
        "Covers shops and marketplaces such as Amazon and eBay selling physical "
        "consumer goods: orders, receipts, returns, promotions, replica watches, "
        "clothing, jewellery, electronics and cosmetics. Excludes shops selling "
        "medicines, pills, software, loans, degrees or phone plans, and parcel "
        "tracking sent by a carrier."
    ),
    "logistics": (
        "Covers carriers and couriers such as UPS, FedEx, DHL and postal services: "
        "tracking, failed delivery, redelivery, freight and courier customs fees. "
        "Excludes order or shipping confirmations sent by a shop."
    ),
    "healthcare": (
        "Covers hospitals, doctors, clinics, pharmacies including online pharmacies "
        "and chemists, and any medicine, pill, supplement or formula taken for "
        "health, sexual performance, enlargement or weight loss, lab results and "
        "health insurance. Excludes sexual or dating content that names no pill, "
        "medicine or other product to take, fitness equipment and cosmetics sold by "
        "a shop, and job postings and a hospital's or clinic's messages to its own "
        "staff about employment."
    ),
    "government": (
        "Covers tax authorities, courts, police, customs agencies, licences, "
        "permits and benefits agencies. Excludes political campaigns and parties, "
        "and customs fees charged by a courier."
    ),
    "hr": (
        "Covers an employer's messages to its own staff (payroll, benefits, leave, "
        "reviews, onboarding, policies) and recruiting: job boards and job lists "
        "(including those run by software projects), job postings and job offers in "
        "any industry (including from banks, software companies, universities and "
        "hospitals), recruiter messages and work-from-home job offers. Excludes "
        "training courses and money-making schemes that are not a job."
    ),
    "telecommunications": (
        "Covers mobile carriers and phone, internet and cable providers: bills, "
        "plans, SIM cards, devices sold by the carrier, calling cards, and webmail "
        "and account messages from a phone or internet carrier. Excludes webmail "
        "and online accounts from providers that are not phone or internet carriers."
    ),
    "social_media": (
        "Covers notifications from a social network about the reader's account on "
        "it: friend or follow requests, messages, comments, tags, and profile or "
        "security alerts from sites such as Facebook, MySpace, Twitter and "
        "LinkedIn. Excludes mailing lists, discussion lists, news digests, "
        "newsletters, web forums, bulletin boards, dating and adult sites, e-cards "
        "and personal email."
    ),
    "education": (
        "Covers universities, schools, online courses, degree and diploma offers, "
        "academic conferences, calls for papers, journals and research mailing "
        "lists. Excludes job postings and a university's messages to its own staff "
        "about employment."
    ),
    NONE_OPTION: (
        "Covers every email that no option above fits: news outlets and "
        "newsletters, entertainment, gambling, casinos, lotteries, adult content, "
        "dating, religion, politics, charities, market-research surveys, personal "
        "conversation, and emails with no readable content that shows a sector."
    ),
}


def build_domain_question() -> dict[str, Any]:
    """Build the TypeSafe Choice question over the ten domains plus ``none``.

    Returns:
        A Choice question as a plain dictionary, which the TypeSafe SDK accepts
        in place of a ``Choice`` object.
    """
    return {
        "type": "choice",
        "instructions": DOMAIN_QUESTION,
        "criteria": dict(DOMAIN_OPTIONS),
    }


def trim_body(body: str, max_chars: int = MAX_BODY_CHARS) -> str:
    """Strip surrounding whitespace and cut the body to ``max_chars`` characters.

    Args:
        body: Raw email body.
        max_chars: Maximum number of characters to keep.

    Returns:
        The trimmed body.
    """
    body = body.strip()
    if len(body) > max_chars:
        body = body[:max_chars].rstrip() + "... [truncated]"
    return body


def build_state(email: EmailData) -> dict[str, str]:
    """Build the TypeSafe state for an email as named fields.

    Args:
        email: Email to classify.

    Returns:
        State with ``sender``, ``subject`` and a trimmed ``body``.
    """
    return {
        "sender": email.sender,
        "subject": email.subject,
        "body": trim_body(email.body),
    }


def _response_field(response: Any, name: str) -> Any:
    """Read a response field from either an SDK object or JSON dictionary."""
    if isinstance(response, Mapping):
        return response[name]
    return getattr(response, name)


def _optional_response_field(response: Any, name: str, default: Any = None) -> Any:
    """Read an optional response field from SDK objects or JSON dictionaries."""
    if response is None:
        return default
    if isinstance(response, Mapping):
        return response.get(name, default)
    return getattr(response, name, default)


def _validate_custom_answer(response: Any, answers: Any, answer: Any) -> None:
    """Reject malformed custom-endpoint decisions before they can be trusted."""
    if not isinstance(response, Mapping) or not isinstance(answers, Mapping):
        raise ValueError("TypeSafe endpoint response has an invalid answers object")
    if not isinstance(answer, Mapping):
        raise ValueError("TypeSafe endpoint answer must be an object")
    if answer.get("type") != "choice":
        raise ValueError("TypeSafe endpoint answer type must be 'choice'")

    choice = answer.get("choice")
    if not isinstance(choice, str) or choice not in DOMAIN_OPTIONS:
        raise ValueError("TypeSafe endpoint returned an unknown choice")

    confidence = answer.get("confidence")
    if not _is_probability(confidence):
        raise ValueError("TypeSafe endpoint confidence must be finite and in [0, 1]")

    probabilities = answer.get("probabilities")
    if not isinstance(probabilities, Mapping) or set(probabilities) != set(
        DOMAIN_OPTIONS
    ):
        raise ValueError("TypeSafe endpoint probabilities do not match the choices")
    if any(not _is_probability(value) for value in probabilities.values()):
        raise ValueError("TypeSafe endpoint probabilities must be finite and in [0, 1]")


def _is_probability(value: Any) -> bool:
    """Whether value is a real, non-boolean probability in the inclusive range."""
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and 0.0 <= value <= 1.0
    )


class _TypeSafeHTTPClient:
    """Minimal standard-library client for a configured System One endpoint."""

    def __init__(self, config: LLMConfig) -> None:
        self.config = config

    def system_one(
        self, *, state: dict[str, str], questions: dict[str, Any]
    ) -> dict[str, Any]:
        """POST one System One request and return its JSON response."""
        base_url = self.config.typesafe_base_url
        if base_url is None:
            raise ValueError("TYPESAFE_BASE_URL is not configured")

        request_body = json.dumps(
            {"model": self.config.model, "state": state, "questions": questions}
        ).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        request = Request(
            f"{base_url.rstrip('/')}/systemone",
            data=request_body,
            headers=headers,
            method="POST",
        )

        with urlopen(request, timeout=self.config.timeout) as response:
            status = response.getcode()
            if status != 200:
                raise RuntimeError(f"TypeSafe endpoint returned HTTP {status}")
            raw_response = response.read()

        try:
            decoded = json.loads(raw_response.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("TypeSafe endpoint returned malformed JSON") from error
        if not isinstance(decoded, dict):
            raise ValueError("TypeSafe endpoint response must be a JSON object")
        return decoded


class TypeSafeClassifier:
    """Email classifier backed by one TypeSafe Choice question (Method 3).

    Returns the same ``ClassificationResult`` as ``LLMClassifier``:

    - ``domain`` is the chosen option, or ``None`` when TypeSafe answers ``none``.
    - ``scores`` holds the Choice probabilities of the ten domains. The ``none``
      probability is left out so it never becomes a candidate domain when scores
      are combined; it is kept in ``details["probabilities"]``.
    - ``confidence`` is TypeSafe's own Choice confidence, also for ``none``. A
      failed call returns ``confidence=0.0`` with ``details["fallback"] = True``.
    """

    def __init__(self, config: LLMConfig, client: Optional[Any] = None) -> None:
        """Initialize the TypeSafe classifier.

        Args:
            config: LLM configuration with ``provider=typesafe``. ``api_key``,
                ``model`` and ``timeout`` are passed to the TypeSafe transport.
            client: Optional ready-made TypeSafe client (used by tests). When
                omitted, a configured custom URL uses the standard-library HTTP
                adapter; the hosted path creates a ``typesafe_sdk.TypeSafeClient``.
        """
        self.config = config
        self._client: Optional[Any] = client
        self._question = build_domain_question()
        self._domain_names = get_domain_names()

    def _get_client(self) -> Any:
        """Get or create the TypeSafe client (lazy initialization).

        Raises:
            ProviderNotInstalledError: If ``typesafe-sdk`` is not installed.
        """
        if self._client is None:
            if self.config.typesafe_base_url:
                self._client = _TypeSafeHTTPClient(self.config)
            else:
                try:
                    from typesafe_sdk import TypeSafeClient
                except ImportError:
                    raise ProviderNotInstalledError(
                        provider=LLMProvider.TYPESAFE,
                        package="typesafe-sdk",
                        install_cmd=self.config.get_install_command(),
                    )
                # The SDK retries rate-limit and overload errors itself.
                self._client = TypeSafeClient(
                    api_key=self.config.api_key,
                    model=self.config.model or None,
                    timeout=float(self.config.timeout),
                )
        return self._client

    def classify(self, email: EmailData) -> ClassificationResult:
        """Classify an email with the TypeSafe Choice question.

        Args:
            email: Email data to classify.

        Returns:
            ClassificationResult with domain, confidence and scores.
        """
        try:
            client = self._get_client()
            response = client.system_one(
                state=build_state(email),
                questions={QUESTION_ID: self._question},
            )
            return self._convert_response(
                response, validate_custom=bool(self.config.typesafe_base_url)
            )
        except Exception as e:
            logger.warning(f"TypeSafe classification failed: {e}")
            return self._create_fallback_result(str(e))

    def _convert_response(
        self, response: Any, *, validate_custom: bool = False
    ) -> ClassificationResult:
        """Map a TypeSafe response to a ClassificationResult.

        Args:
            response: ``SystemOneResponse`` from the TypeSafe SDK.

        Returns:
            Standard ClassificationResult compatible with other methods.
        """
        answers = _response_field(response, "answers")
        answer = _response_field(answers, QUESTION_ID)
        if validate_custom:
            _validate_custom_answer(response, answers, answer)

        choice_value = _response_field(answer, "choice")
        choice = choice_value if validate_custom else str(choice_value)
        probability_values = _response_field(answer, "probabilities")
        probabilities = {
            option: float(probability)
            for option, probability in probability_values.items()
        }
        confidence = float(_response_field(answer, "confidence"))

        scores = {name: probabilities.get(name, 0.0) for name in self._domain_names}
        domain: Optional[str] = choice if choice in scores else None

        usage = _optional_response_field(response, "usage")
        return ClassificationResult(
            domain=domain,
            confidence=confidence,
            scores=scores,
            method=METHOD_NAME,
            details={
                "choice": choice,
                "probabilities": probabilities,
                "provider": LLMProvider.TYPESAFE.value,
                "model": _optional_response_field(response, "model", self.config.model),
                "usage": {
                    "input_tokens": _optional_response_field(usage, "input_tokens"),
                    "output_tokens": _optional_response_field(usage, "output_tokens"),
                },
            },
        )

    def _create_fallback_result(self, error_message: str) -> ClassificationResult:
        """Create a fallback result when the TypeSafe call fails.

        Args:
            error_message: Description of the failure.

        Returns:
            ClassificationResult indicating failure.
        """
        return ClassificationResult(
            domain=None,
            confidence=0.0,
            scores={name: 0.0 for name in self._domain_names},
            method=METHOD_NAME,
            details={
                "error": error_message,
                "fallback": True,
                "provider": LLMProvider.TYPESAFE.value,
            },
        )
