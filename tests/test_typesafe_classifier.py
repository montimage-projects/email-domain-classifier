"""Tests for the TypeSafe-based classifier (issue #17).

The TypeSafe client is always mocked: no test makes a network call.
"""

import json
import os
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional
from unittest.mock import MagicMock, patch

import pytest

from email_classifier import EmailData
from email_classifier.classifier import ClassificationResult
from email_classifier.domains import get_domain_names
from email_classifier.llm import (
    LLMClassifier,
    TypeSafeClassifier,
    create_classifier,
)
from email_classifier.llm.config import (
    DEFAULT_MODELS,
    PROVIDER_API_KEYS,
    LLMConfig,
    LLMConfigError,
    LLMProvider,
)
from email_classifier.llm.providers import create_llm
from email_classifier.llm.typesafe_classifier import (
    DOMAIN_OPTIONS,
    DOMAIN_QUESTION,
    MAX_BODY_CHARS,
    NONE_OPTION,
    QUESTION_ID,
    build_domain_question,
    build_state,
    trim_body,
)

DOMAIN_DOC = Path(__file__).parent.parent / "docs" / "design" / "domain-profiles.md"
TEST_KEY = "ts-test-key-not-real"


def make_config(**overrides: Any) -> LLMConfig:
    """Build a TypeSafe LLMConfig for tests."""
    values: dict[str, Any] = {
        "provider": LLMProvider.TYPESAFE,
        "model": "jev-latest",
        "api_key": TEST_KEY,
    }
    values.update(overrides)
    return LLMConfig(**values)


def make_email(body: str = "Your statement is ready.") -> EmailData:
    """Build an email for tests."""
    return EmailData(
        sender="alerts@bank.example",
        receiver="me@example.com",
        date="2024-01-01",
        subject="Account statement",
        body=body,
        urls="",
    )


def make_response(
    choice: str,
    probabilities: dict[str, float],
    confidence: float,
    input_tokens: Optional[int] = 512,
    output_tokens: Optional[int] = 20,
) -> SimpleNamespace:
    """Build an object shaped like typesafe_sdk.SystemOneResponse."""
    answer = SimpleNamespace(
        type="choice",
        choice=choice,
        probabilities=probabilities,
        confidence=confidence,
    )
    return SimpleNamespace(
        model="jev-1.13.0",
        answers={QUESTION_ID: answer},
        usage=SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens),
    )


def make_client(response: Any) -> MagicMock:
    """Build a mock TypeSafe client returning ``response``."""
    client = MagicMock()
    client.system_one.return_value = response
    return client


def custom_response_data(
    choice: str = "finance",
    probabilities: Optional[dict[str, Any]] = None,
    confidence: Any = 0.8,
    answer_type: str = "choice",
) -> dict[str, Any]:
    """Build the JSON shape returned by the custom System One transport."""
    answer_probabilities: dict[str, Any] = (
        dict(probabilities) if probabilities is not None else finance_probabilities()
    )
    return {
        "model": "kev-latest",
        "answers": {
            QUESTION_ID: {
                "type": answer_type,
                "choice": choice,
                "probabilities": answer_probabilities,
                "confidence": confidence,
            }
        },
        "usage": {"input_tokens": 12, "output_tokens": 3},
    }


class MockHTTPResponse:
    """Context manager matching the urllib response interface."""

    def __init__(self, body: bytes, status: int = 200) -> None:
        self.body = body
        self.status = status

    def __enter__(self) -> "MockHTTPResponse":
        return self

    def __exit__(self, *args: Any) -> None:
        return None

    def getcode(self) -> int:
        return self.status

    def read(self) -> bytes:
        return self.body


def finance_probabilities() -> dict[str, float]:
    """Probabilities over all 11 options, mostly finance."""
    probabilities = {name: 0.0 for name in DOMAIN_OPTIONS}
    probabilities.update({"finance": 0.85, "retail": 0.1, NONE_OPTION: 0.05})
    return probabilities


class TestDomainQuestion:
    """The Choice question sent to TypeSafe."""

    def test_one_choice_question_with_ten_domains_plus_none(self):
        question = build_domain_question()
        assert question["type"] == "choice"
        assert question["instructions"] == DOMAIN_QUESTION
        assert set(question["criteria"]) == set(get_domain_names()) | {NONE_OPTION}
        assert len(question["criteria"]) == 11

    def test_each_domain_option_says_what_it_covers_and_excludes(self):
        for name, text in DOMAIN_OPTIONS.items():
            assert text.startswith("Covers "), name
            if name != NONE_OPTION:
                assert " Excludes " in text, name

    def test_social_media_excludes_lists_digests_and_forums(self):
        excludes = DOMAIN_OPTIONS["social_media"].split(" Excludes ", 1)[1]
        for term in ("mailing lists", "news digests", "web forums"):
            assert term in excludes

    def test_question_matches_domain_definition_doc(self):
        """The question text must stay in sync with the #15 domain definition."""
        doc = DOMAIN_DOC.read_text(encoding="utf-8")
        section = doc.split("### Text for the TypeSafe Choice question", 1)[1]
        block = re.search(r"```text\n(.*?)```", section, re.DOTALL)
        assert block is not None
        text = block.group(1)

        question_part, options_part = text.split("\nOptions:\n", 1)
        doc_question = " ".join(question_part.replace("Question:", "", 1).split())
        assert doc_question == DOMAIN_QUESTION

        doc_options = {}
        for paragraph in options_part.strip().split("\n\n"):
            name, description = paragraph.split(":", 1)
            doc_options[name.strip()] = " ".join(description.split())
        assert doc_options == DOMAIN_OPTIONS


class TestState:
    """The state sent to TypeSafe."""

    def test_state_is_named_fields(self):
        email = make_email(body="  Hello there.  ")
        assert build_state(email) == {
            "sender": "alerts@bank.example",
            "subject": "Account statement",
            "body": "Hello there.",
        }

    def test_long_body_is_trimmed(self):
        body = "x" * (MAX_BODY_CHARS + 500)
        trimmed = build_state(make_email(body=body))["body"]
        assert trimmed == "x" * MAX_BODY_CHARS + "... [truncated]"

    def test_short_body_is_not_truncated(self):
        assert trim_body("short body") == "short body"


class TestTypeSafeClassifier:
    """Mapping a TypeSafe Choice answer to ClassificationResult."""

    def test_sends_state_and_question(self):
        response = make_response("finance", finance_probabilities(), 0.8)
        client = make_client(response)
        email = make_email()

        TypeSafeClassifier(make_config(), client=client).classify(email)

        client.system_one.assert_called_once()
        kwargs = client.system_one.call_args.kwargs
        assert kwargs["state"] == build_state(email)
        assert kwargs["questions"] == {QUESTION_ID: build_domain_question()}

    def test_choice_maps_to_domain_and_probabilities_to_scores(self):
        probabilities = finance_probabilities()
        response = make_response("finance", probabilities, 0.81)
        classifier = TypeSafeClassifier(make_config(), client=make_client(response))

        result = classifier.classify(make_email())

        assert isinstance(result, ClassificationResult)
        assert result.domain == "finance"
        assert result.confidence == pytest.approx(0.81)
        assert result.method == "typesafe"
        assert set(result.scores) == set(get_domain_names())
        for name in get_domain_names():
            assert result.scores[name] == pytest.approx(probabilities[name])

    def test_confidence_is_typesafe_confidence_not_max_probability(self):
        response = make_response("finance", finance_probabilities(), 0.42)
        classifier = TypeSafeClassifier(make_config(), client=make_client(response))
        result = classifier.classify(make_email())
        assert result.confidence == pytest.approx(0.42)
        assert result.scores["finance"] == pytest.approx(0.85)

    def test_none_maps_to_no_domain_and_stays_out_of_scores(self):
        probabilities = {name: 0.01 for name in get_domain_names()}
        probabilities[NONE_OPTION] = 0.9
        response = make_response(NONE_OPTION, probabilities, 0.77)
        classifier = TypeSafeClassifier(make_config(), client=make_client(response))

        result = classifier.classify(make_email())

        assert result.domain is None
        assert result.confidence == pytest.approx(0.77)
        assert NONE_OPTION not in result.scores
        assert result.details is not None
        assert result.details["choice"] == NONE_OPTION
        assert result.details["probabilities"][NONE_OPTION] == pytest.approx(0.9)
        assert "fallback" not in result.details

    def test_details_carry_model_and_usage(self):
        response = make_response("finance", finance_probabilities(), 0.8)
        classifier = TypeSafeClassifier(make_config(), client=make_client(response))
        result = classifier.classify(make_email())
        assert result.details is not None
        assert result.details["provider"] == "typesafe"
        assert result.details["model"] == "jev-1.13.0"
        assert result.details["usage"] == {"input_tokens": 512, "output_tokens": 20}

    def test_client_error_returns_fallback_result(self):
        client = MagicMock()
        client.system_one.side_effect = RuntimeError("service unavailable")
        classifier = TypeSafeClassifier(make_config(), client=client)

        result = classifier.classify(make_email())

        assert result.domain is None
        assert result.confidence == 0.0
        assert result.scores == {name: 0.0 for name in get_domain_names()}
        assert result.details is not None
        assert result.details["fallback"] is True
        assert "service unavailable" in result.details["error"]

    def test_missing_sdk_returns_fallback_with_install_hint(self):
        classifier = TypeSafeClassifier(make_config())
        with patch.dict("sys.modules", {"typesafe_sdk": None}):
            result = classifier.classify(make_email())
        assert result.domain is None
        assert result.details is not None
        assert result.details["fallback"] is True
        assert "typesafe-sdk" in result.details["error"]
        assert "email-domain-classifier[typesafe]" in result.details["error"]
        assert TEST_KEY not in result.details["error"]

    def test_client_is_created_once_with_config(self):
        fake_sdk = MagicMock()
        fake_sdk.TypeSafeClient.return_value = make_client(
            make_response("finance", finance_probabilities(), 0.8)
        )
        classifier = TypeSafeClassifier(make_config(timeout=12))
        with patch.dict("sys.modules", {"typesafe_sdk": fake_sdk}):
            classifier.classify(make_email())
            classifier.classify(make_email())
        fake_sdk.TypeSafeClient.assert_called_once_with(
            api_key=TEST_KEY, model="jev-latest", timeout=12.0
        )

    def test_does_not_use_name_variant_normalization(self):
        assert not hasattr(TypeSafeClassifier, "_normalize_domain_name")


class TestCustomHTTPTransport:
    """Custom System One requests use urllib and preserve classifier behavior."""

    def _config(self, **overrides: Any) -> LLMConfig:
        values: dict[str, Any] = {
            "provider": LLMProvider.TYPESAFE,
            "model": "kev-latest",
            "typesafe_base_url": "http://systemone.example/v1///",
        }
        values.update(overrides)
        return LLMConfig(**values)

    def test_keyless_request_has_exact_url_model_state_and_question(self):
        email = make_email()
        response = MockHTTPResponse(json.dumps(custom_response_data()).encode())
        classifier = TypeSafeClassifier(self._config(timeout=17))

        with (
            patch(
                "email_classifier.llm.typesafe_classifier.urlopen",
                return_value=response,
            ) as mock_urlopen,
            patch.dict("sys.modules", {"typesafe_sdk": None}),
        ):
            result = classifier.classify(email)

        request = mock_urlopen.call_args.args[0]
        assert mock_urlopen.call_args.kwargs == {"timeout": 17}
        assert request.full_url == "http://systemone.example/v1/systemone"
        assert request.method == "POST"
        assert request.get_header("Content-type") == "application/json"
        assert request.get_header("Authorization") is None
        body = json.loads(request.data.decode("utf-8"))
        assert body == {
            "model": "kev-latest",
            "state": build_state(email),
            "questions": {QUESTION_ID: build_domain_question()},
        }
        assert result.domain == "finance"
        assert result.details is not None
        assert result.details["model"] == "kev-latest"

    def test_optional_api_key_is_sent_as_bearer(self):
        response = MockHTTPResponse(json.dumps(custom_response_data()).encode())
        classifier = TypeSafeClassifier(self._config(api_key=TEST_KEY))

        with patch(
            "email_classifier.llm.typesafe_classifier.urlopen",
            return_value=response,
        ) as mock_urlopen:
            result = classifier.classify(make_email())

        request = mock_urlopen.call_args.args[0]
        assert request.get_header("Authorization") == f"Bearer {TEST_KEY}"
        assert result.domain == "finance"

    @pytest.mark.parametrize("failure", ["http", "timeout", "malformed_json"])
    def test_transport_failures_return_fallback(self, failure: str):
        if failure == "http":
            transport = MockHTTPResponse(b"server failure", status=503)
        elif failure == "timeout":
            transport = TimeoutError("request timed out")
        else:
            transport = MockHTTPResponse(b"not json")

        with patch(
            "email_classifier.llm.typesafe_classifier.urlopen",
            side_effect=transport if isinstance(transport, Exception) else None,
            return_value=None if isinstance(transport, Exception) else transport,
        ):
            result = TypeSafeClassifier(self._config()).classify(make_email())

        assert result.domain is None
        assert result.confidence == 0.0
        assert result.details is not None
        assert result.details["fallback"] is True

    @pytest.mark.parametrize(
        "response",
        [
            custom_response_data(answer_type="score"),
            custom_response_data(choice="unrecognized"),
            custom_response_data(confidence=float("nan")),
            custom_response_data(confidence=1.01),
            custom_response_data(
                probabilities={**finance_probabilities(), "finance": float("inf")}
            ),
            custom_response_data(
                probabilities={**finance_probabilities(), "finance": -0.1}
            ),
        ],
    )
    def test_corrupt_custom_decisions_return_fallback(self, response: dict[str, Any]):
        http_response = MockHTTPResponse(json.dumps(response).encode())
        with patch(
            "email_classifier.llm.typesafe_classifier.urlopen",
            return_value=http_response,
        ):
            result = TypeSafeClassifier(self._config()).classify(make_email())

        assert result.domain is None
        assert result.confidence == 0.0
        assert result.details is not None
        assert result.details["fallback"] is True


class TestTypeSafeConfig:
    """Selecting TypeSafe through LLMConfig."""

    def test_positional_temperature_argument_keeps_legacy_position(self):
        config = LLMConfig(LLMProvider.OLLAMA, "llama3.2", None, 0.7)
        assert config.temperature == pytest.approx(0.7)
        assert config.typesafe_base_url is None

    def test_provider_defaults(self):
        assert LLMProvider.TYPESAFE.value == "typesafe"
        assert DEFAULT_MODELS[LLMProvider.TYPESAFE] == "jev-latest"
        assert PROVIDER_API_KEYS[LLMProvider.TYPESAFE] == "TYPESAFE_API_KEY"

    def test_requires_api_key(self):
        with pytest.raises(LLMConfigError, match="TYPESAFE_API_KEY"):
            LLMConfig(provider=LLMProvider.TYPESAFE, model="jev-latest")

    def test_custom_http_endpoint_allows_optional_api_key(self):
        config = LLMConfig(
            provider=LLMProvider.TYPESAFE,
            model="kev-latest",
            typesafe_base_url="http://systemone.example/v1",
        )
        assert config.api_key is None
        assert config.typesafe_base_url == "http://systemone.example/v1"

    @pytest.mark.parametrize(
        "url",
        ["ftp://systemone.example/v1", "http:///v1", "https://example.com:bad/v1"],
    )
    def test_custom_endpoint_requires_http_url(self, url: str):
        with pytest.raises(LLMConfigError, match="TYPESAFE_BASE_URL"):
            LLMConfig(
                provider=LLMProvider.TYPESAFE,
                model="kev-latest",
                api_key=TEST_KEY,
                typesafe_base_url=url,
            )

    @patch("email_classifier.llm.config.load_dotenv")
    def test_from_env_reads_typesafe_key(self, _mock_load_dotenv):
        env = {"LLM_PROVIDER": "typesafe", "TYPESAFE_API_KEY": TEST_KEY}
        with patch.dict(os.environ, env, clear=True):
            config = LLMConfig.from_env()
        assert config.provider == LLMProvider.TYPESAFE
        assert config.model == "jev-latest"
        assert config.api_key == TEST_KEY
        assert TEST_KEY not in repr(config)

    @patch("email_classifier.llm.config.load_dotenv")
    def test_from_env_reads_keyless_custom_endpoint(self, _mock_load_dotenv):
        env = {
            "LLM_PROVIDER": "typesafe",
            "LLM_MODEL": "kev-latest",
            "TYPESAFE_BASE_URL": "http://192.168.0.124:8009/v1",
            "LLM_TIMEOUT": "30",
        }
        with patch.dict(os.environ, env, clear=True):
            config = LLMConfig.from_env()
        assert config.provider == LLMProvider.TYPESAFE
        assert config.model == "kev-latest"
        assert config.typesafe_base_url == "http://192.168.0.124:8009/v1"
        assert config.timeout == 30
        assert config.api_key is None

    def test_hosted_path_still_requires_key_without_custom_url(self):
        with pytest.raises(LLMConfigError, match="TYPESAFE_API_KEY"):
            LLMConfig(
                provider=LLMProvider.TYPESAFE,
                model="jev-latest",
                typesafe_base_url=None,
            )

    def test_install_command(self):
        assert make_config().get_install_command() == (
            "pip install email-domain-classifier[typesafe]"
        )

    def test_create_llm_rejects_typesafe(self):
        with pytest.raises(LLMConfigError, match="not a LangChain provider"):
            create_llm(make_config())


class TestClassifierSelection:
    """create_classifier and the classifiers that use it."""

    def test_factory_returns_typesafe_classifier(self):
        assert isinstance(create_classifier(make_config()), TypeSafeClassifier)

    def test_factory_keeps_llm_classifier_as_default(self):
        config = LLMConfig(provider=LLMProvider.OLLAMA, model="llama3.2")
        assert isinstance(create_classifier(config), LLMClassifier)

    def test_email_classifier_uses_typesafe(self):
        from email_classifier import EmailClassifier

        classifier = EmailClassifier(llm_config=make_config())
        assert isinstance(classifier.method3, TypeSafeClassifier)
        assert classifier.llm_enabled is True

    def test_email_classifier_three_method_scores_have_no_none(self):
        from email_classifier import EmailClassifier

        classifier = EmailClassifier(llm_config=make_config())
        response = make_response("finance", finance_probabilities(), 0.8)
        assert isinstance(classifier.method3, TypeSafeClassifier)
        classifier.method3._client = make_client(response)

        domain, details = classifier.classify(make_email())

        assert details["method3"]["domain"] == "finance"
        assert NONE_OPTION not in details["combined_scores"]
        assert domain == "finance"

    def test_hybrid_classifier_uses_typesafe_on_disagreement(self):
        from email_classifier.classifier import HybridClassifier

        hybrid = HybridClassifier(llm_config=make_config())
        assert isinstance(hybrid.llm_classifier, TypeSafeClassifier)
        hybrid.llm_classifier._client = make_client(
            make_response("finance", finance_probabilities(), 0.8)
        )
        scores = {name: 0.0 for name in get_domain_names()}
        hybrid.method1 = MagicMock()
        hybrid.method1.classify.return_value = ClassificationResult(
            "retail", 0.5, scores, "keyword_taxonomy"
        )
        hybrid.method2 = MagicMock()
        hybrid.method2.classify.return_value = ClassificationResult(
            "technology", 0.5, scores, "structural_template"
        )

        domain, details = hybrid.classify(make_email())

        assert details["path"] == "llm_assisted"
        assert details["method3"]["confidence"] == pytest.approx(0.8)
        assert domain == "finance"


class TestVerifyPrerequisitesTypeSafe:
    """The CLI prerequisite check on the TypeSafe path."""

    def _run(
        self, tmp_path: Path, available: bool, env: dict[str, str]
    ) -> tuple[bool, list[str], Optional[LLMConfig], MagicMock, MagicMock]:
        from email_classifier.cli import verify_prerequisites

        input_path = tmp_path / "emails.csv"
        input_path.write_text("sender,subject,body\n", encoding="utf-8")
        ui = MagicMock()
        with (
            patch.dict(os.environ, env, clear=True),
            patch("email_classifier.cli.load_dotenv"),
            patch("email_classifier.llm.config.load_dotenv"),
            patch(
                "email_classifier.llm.providers.check_provider_available",
                return_value=(available, None if available else "missing"),
            ) as check_provider,
        ):
            success, errors, config = verify_prerequisites(
                input_path, tmp_path / "out", use_llm=True, ui=ui
            )
        return success, errors, config, ui, check_provider

    def test_ok_when_key_and_sdk_present(self, tmp_path):
        env = {"LLM_PROVIDER": "typesafe", "TYPESAFE_API_KEY": TEST_KEY}
        success, errors, config, ui, check_provider = self._run(tmp_path, True, env)
        assert success is True
        assert errors == []
        assert config is not None
        assert config.provider == LLMProvider.TYPESAFE
        printed = json.dumps([str(c) for c in ui.mock_calls])
        assert TEST_KEY not in printed
        check_provider.assert_called_once_with(LLMProvider.TYPESAFE)

    def test_custom_endpoint_skips_sdk_prerequisite(self, tmp_path):
        env = {
            "LLM_PROVIDER": "typesafe",
            "LLM_MODEL": "kev-latest",
            "TYPESAFE_BASE_URL": "http://systemone.example/v1",
        }
        success, errors, config, _, check_provider = self._run(tmp_path, False, env)
        assert success is True
        assert errors == []
        assert config is not None
        assert config.api_key is None
        check_provider.assert_not_called()

    def test_error_when_sdk_missing(self, tmp_path):
        env = {"LLM_PROVIDER": "typesafe", "TYPESAFE_API_KEY": TEST_KEY}
        success, errors, _, _, _ = self._run(tmp_path, False, env)
        assert success is False
        assert any("typesafe-sdk" in e for e in errors)
        assert any("email-domain-classifier[typesafe]" in e for e in errors)

    def test_error_when_key_missing(self, tmp_path):
        success, errors, _, _, check_provider = self._run(
            tmp_path, True, {"LLM_PROVIDER": "typesafe"}
        )
        assert success is False
        assert any("TYPESAFE_API_KEY" in e for e in errors)
        check_provider.assert_not_called()


class TestRealSdkContract:
    """Check the request and response shapes against the installed SDK.

    Runs only where ``typesafe-sdk`` is installed. Uses a mock HTTP transport,
    so no request leaves the machine.
    """

    def test_request_and_response_with_real_sdk(self):
        typesafe_sdk = pytest.importorskip("typesafe_sdk")
        httpx2 = pytest.importorskip("httpx2")
        captured: dict[str, Any] = {}
        probabilities = finance_probabilities()

        def handler(request: Any) -> Any:
            captured["path"] = request.url.path
            captured["body"] = json.loads(request.content)
            return httpx2.Response(
                200,
                json={
                    "model": "jev-1.13.0",
                    "answers": {
                        QUESTION_ID: {
                            "type": "choice",
                            "choice": "finance",
                            "probabilities": probabilities,
                            "confidence": 0.79,
                        }
                    },
                    "usage": {"input_tokens": 640, "output_tokens": 30},
                },
            )

        client = typesafe_sdk.TypeSafeClient(
            api_key=TEST_KEY, transport=httpx2.MockTransport(handler)
        )
        classifier = TypeSafeClassifier(make_config(), client=client)
        email = make_email()

        result = classifier.classify(email)

        assert captured["path"] == "/v1/systemone"
        assert captured["body"]["state"] == build_state(email)
        question = captured["body"]["questions"][QUESTION_ID]
        assert question["type"] == "choice"
        assert set(question["criteria"]) == set(DOMAIN_OPTIONS)
        assert result.details is not None
        assert "fallback" not in result.details
        assert result.domain == "finance"
        assert result.confidence == pytest.approx(0.79)
        assert result.scores["finance"] == pytest.approx(0.85)
        assert result.details["usage"] == {"input_tokens": 640, "output_tokens": 30}
        assert result.details["model"] == "jev-1.13.0"
