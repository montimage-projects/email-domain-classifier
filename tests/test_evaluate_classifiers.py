"""Tests for the classifier evaluation harness (#19).

No test makes a network call, needs the TypeSafe SDK or reads the Git LFS
CEAS_08 data: TypeSafe clients are fakes and every input file is built in
``tmp_path``.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, Optional

import pytest

from email_classifier.classifier import EmailData, HybridClassifier
from email_classifier.domains import get_domain_names
from email_classifier.llm.config import LLMConfig, LLMProvider
from email_classifier.llm.typesafe_classifier import TypeSafeClassifier

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = REPO_ROOT / "scripts" / "evaluate_classifiers.py"
FAKE_KEY = "ts-test-key-not-real"


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("evaluate_classifiers", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ev = _load_script()

# Method 1 says finance, Method 2 says social_media: the hybrid gate fires.
DISAGREEING = {
    "sender": "service@paypal.com",
    "receiver": "me@example.com",
    "date": "2024-01-01",
    "subject": "Your account statement",
    "body": "Your bank account balance and credit card payment are ready. "
    "Verify your account.",
    "urls": "",
    "label": "1",
}
# Both methods say social_media: the hybrid never calls the LLM.
AGREEING = {
    "sender": "no-reply@facebook.com",
    "receiver": "me@example.com",
    "date": "2024-01-02",
    "subject": "Friend request",
    "body": "Someone sent you a friend request and tagged you in a photo. "
    "View profile, comment, follow.",
    "urls": "http://x",
    "label": "0",
}


def make_email(row: dict[str, str]) -> EmailData:
    """Build an EmailData the way the pipeline does."""
    return ev.build_email(row, ev.StreamingProcessor())


def make_response(choice: str, confidence: float) -> SimpleNamespace:
    """Build an object shaped like typesafe_sdk.SystemOneResponse."""
    options = [*get_domain_names(), "none"]
    rest = (1.0 - confidence) / (len(options) - 1)
    probabilities = {o: confidence if o == choice else rest for o in options}
    answer = SimpleNamespace(
        choice=choice, probabilities=probabilities, confidence=confidence
    )
    return SimpleNamespace(
        answers={"domain": answer},
        usage=SimpleNamespace(input_tokens=700, output_tokens=12),
        model="jev-test",
    )


class StaticClient:
    """TypeSafe client stand-in that returns one response or raises."""

    def __init__(
        self, response: Optional[SimpleNamespace], error: Optional[Exception] = None
    ) -> None:
        self.response = response
        self.error = error

    def system_one(self, **kwargs: Any) -> Any:
        if self.error is not None:
            raise self.error
        return self.response


def make_classifier(client: Any) -> TypeSafeClassifier:
    config = LLMConfig(
        provider=LLMProvider.TYPESAFE, model="jev-test", api_key=FAKE_KEY
    )
    return TypeSafeClassifier(config, client=client)


def make_record(
    eid: str,
    choice: Optional[str] = "finance",
    confidence: float = 0.9,
    fallback: bool = False,
    set_name: str = "labeled",
) -> dict[str, Any]:
    """Build a cache record through the real classifier and build_record."""
    error = RuntimeError("boom") if fallback else None
    response = None if fallback else make_response(str(choice), confidence)
    result = make_classifier(StaticClient(response, error)).classify(
        make_email(DISAGREEING)
    )
    return ev.build_record(
        eid=eid,
        set_name=set_name,
        result=result,
        latency_ms=120.0,
        attempts=1,
        http_statuses=[500 if fallback else 200],
        error_type="RuntimeError" if fallback else None,
        model="jev-test",
        timestamp="2026-10-01T00:00:00+00:00",
    )


def make_labels(ids: list[str], domain: str = "finance", label: str = "1") -> list[Any]:
    return [
        ev.LabeledEmail(
            email_id=eid,
            raw_row=i,
            source_file="email_finance.csv",
            label=label,
            domain=domain,
            ambiguous=False,
            verified=False,
        )
        for i, eid in enumerate(ids)
    ]


def eid_of(n: int) -> str:
    return f"{n:016x}"


class TestStatistics:
    def test_wilson_known_value(self) -> None:
        low, high = ev.wilson_interval(5, 10)
        assert low == pytest.approx(0.2366, abs=1e-4)
        assert high == pytest.approx(0.7634, abs=1e-4)

    def test_wilson_edges(self) -> None:
        assert ev.wilson_interval(0, 0) is None
        low, high = ev.wilson_interval(10, 10)
        assert high == pytest.approx(1.0)
        assert low == pytest.approx(0.7225, abs=1e-4)
        assert ev.wilson_interval(0, 10)[0] == pytest.approx(0.0)

    @pytest.mark.parametrize(
        ("b", "c", "expected"),
        [(0, 6, 0.03125), (6, 0, 0.03125), (3, 3, 1.0), (0, 0, 1.0), (3, 1, 0.625)],
    )
    def test_mcnemar_exact(self, b: int, c: int, expected: float) -> None:
        assert ev.mcnemar_exact(b, c) == pytest.approx(expected)

    def test_percentile_linear(self) -> None:
        assert ev.percentile([4, 1, 3, 2], 50) == pytest.approx(2.5)
        assert ev.percentile([1, 2, 3, 4], 95) == pytest.approx(3.85)
        assert ev.percentile([7], 95) == 7
        assert ev.percentile([], 50) is None

    def test_stratum_weighted_accuracy(self) -> None:
        items = [("a", True), ("a", False), ("b", True)]
        assert ev.stratum_weighted_accuracy(items, {"a": 100, "b": 300}) == (
            pytest.approx(350 / 400)
        )
        assert ev.stratum_weighted_accuracy(items, {"a": 100}) is None
        assert ev.stratum_weighted_accuracy([], {"a": 1}) is None


class TestMetrics:
    @pytest.mark.parametrize(
        ("prediction", "expected"),
        [(None, "none"), ("unsure", "none"), ("", "none"), ("finance", "finance")],
    )
    def test_normalize_prediction(
        self, prediction: Optional[str], expected: str
    ) -> None:
        assert ev.normalize_prediction(prediction) == expected

    def test_classification_metrics_hand_built(self) -> None:
        pairs = (
            [("tech", "tech")] * 4
            + [("tech", "none")]
            + [("none", "none")] * 3
            + [("none", "tech")] * 2
            + [("health", "health"), ("health", "none")]
        )
        m = ev.classification_metrics(pairs, min_support=5)
        assert m["strict_accuracy"]["k"] == 8 and m["strict_accuracy"]["n"] == 12
        assert m["answer_rate"]["k"] == 7
        assert m["answered_accuracy"]["k"] == 5 and m["answered_accuracy"]["n"] == 7
        assert m["none_precision"] == pytest.approx(0.6)
        assert m["none_recall"] == pytest.approx(0.6)
        assert m["macro_f1_classes"] == ["none", "tech"]
        assert m["macro_f1"] == pytest.approx((8 / 11 + 0.6) / 2)
        assert m["per_class"]["health"]["insufficient_support"] is True
        assert m["confusion"]["none"]["tech"] == 2

    def test_unpredicted_class_has_no_precision(self) -> None:
        m = ev.per_class_metrics([("a", "b")])
        assert m["a"]["precision"] is None and m["a"]["f1"] == 0.0
        assert m["b"]["recall"] is None


class TestHybridReplayParity:
    """Replaying cached answers gives the same answer as the live hybrid."""

    @pytest.mark.parametrize(
        ("choice", "confidence", "cutoff", "fallback", "expected"),
        [
            ("technology", 0.8, 0.5, False, "technology"),  # accepted
            ("technology", 0.8, 0.9, False, "finance"),  # below cutoff
            ("none", 0.8, 0.5, False, "none"),  # accepted None -> unsure
            (None, 0.0, 0.0, True, "finance"),  # failed call never accepted
        ],
    )
    def test_replay_matches_live_hybrid(
        self,
        choice: Optional[str],
        confidence: float,
        cutoff: float,
        fallback: bool,
        expected: str,
    ) -> None:
        email = make_email(DISAGREEING)
        error = RuntimeError("boom") if fallback else None
        response = None if fallback else make_response(str(choice), confidence)
        live = HybridClassifier(llm_confidence_cutoff=cutoff)
        live.llm_classifier = make_classifier(StaticClient(response, error))
        live_domain, details = live.classify(email)
        assert details["path"] == "llm_assisted"

        eid = eid_of(1)
        record = make_record(eid, choice, confidence, fallback)
        (outcome,) = ev.replay_hybrid(cutoff, [(eid, email)], {eid: record})
        assert outcome.prediction == ev.normalize_prediction(live_domain) == expected
        assert outcome.accepted is details["llm_gate"]["accepted"]

    def test_no_llm_replay_uses_classic_fallback(self) -> None:
        (outcome,) = ev.replay_hybrid(None, [("x", make_email(DISAGREEING))], None)
        assert outcome.prediction == "finance" and outcome.accepted is None


class TestSweepAndCutoff:
    def _rows_and_records(self) -> tuple[list[Any], dict[str, Any], list[Any]]:
        confidences = [0.1, 0.3, 0.55, 0.9]
        ids = [eid_of(i) for i in range(6)]
        records = [
            make_record(eid, "technology", c) for eid, c in zip(ids, confidences)
        ]
        records.append(make_record(ids[4], fallback=True))
        emails = {eid: make_email(DISAGREEING) for eid in ids[:5]}
        emails[ids[5]] = make_email(AGREEING)
        labels = make_labels(ids[:5], domain="technology") + make_labels(
            [ids[5]], domain="social_media"
        )
        return labels, emails, records

    def test_acceptance_rate_never_increases_with_cutoff(self) -> None:
        labels, emails, records = self._rows_and_records()
        latest = ev.latest_records(records)
        rows = ev.build_rows(labels, emails, latest)
        sweep, _ = ev.run_sweep(rows, emails, latest)
        rates = [row["acceptance_rate"] for row in sweep]
        assert rates == sorted(rates, reverse=True)
        by_cutoff = {row["cutoff"]: row for row in sweep}
        assert by_cutoff[0.0]["n_gate_rows"] == 5
        assert by_cutoff[0.0]["n_accepted"] == 4  # the failed call is rejected
        assert by_cutoff[0.5]["n_accepted"] == 2
        assert by_cutoff[0.95]["n_accepted"] == 0
        assert by_cutoff[0.95]["rejected_llm_counterfactual_accuracy"]["k"] == 4
        assert ev.CUTOFF_SWEEP[0] == 0.0 and ev.CUTOFF_SWEEP[-1] == 0.95
        assert len(ev.CUTOFF_SWEEP) == 20

    @staticmethod
    def _sweep(correct: dict[float, list[bool]]) -> list[dict[str, Any]]:
        return [{"cutoff": c, "correct": v} for c, v in correct.items()]

    def test_adopts_significantly_better_candidate(self) -> None:
        sweep = self._sweep({0.3: [True] * 10, 0.5: [False] * 7 + [True] * 3})
        choice = ev.choose_cutoff(sweep, 0.5)
        assert choice["candidate"] == 0.3 and choice["chosen"] == 0.3
        assert choice["b_candidate_only"] == 7 and choice["c_incumbent_only"] == 0
        assert choice["p_value"] == pytest.approx(2 / 128)

    def test_ties_go_to_the_value_closest_to_incumbent(self) -> None:
        best = [True] * 10
        sweep = self._sweep({0.3: best, 0.5: [False] + [True] * 9, 0.6: best})
        assert ev.choose_cutoff(sweep, 0.5)["candidate"] == 0.6
        worse = [False] + [True] * 9
        for low, high in [(0.4, 0.6), (0.3, 0.7), (0.05, 0.95)]:
            sweep = self._sweep({low: best, 0.5: worse, high: best})
            assert ev.choose_cutoff(sweep, 0.5)["candidate"] == low

    def test_keeps_incumbent_when_not_significant(self) -> None:
        incumbent = [True, True, True, True, False, False, False, True]
        candidate = [True, True, True, False, True, True, True, True]
        choice = ev.choose_cutoff(self._sweep({0.2: candidate, 0.5: incumbent}))
        assert choice["candidate"] == 0.2 and choice["chosen"] == 0.5
        assert choice["p_value"] == pytest.approx(0.625)
        assert "cannot distinguish" in choice["rationale"]

    def test_incumbent_already_best(self) -> None:
        choice = ev.choose_cutoff(self._sweep({0.2: [False], 0.5: [True]}))
        assert choice["chosen"] == 0.5 and choice["p_value"] == 1.0


class TestReport:
    def test_failed_call_is_an_error_not_none(self) -> None:
        ids = [eid_of(i) for i in range(3)]
        emails = {eid: make_email(DISAGREEING) for eid in ids}
        records = [
            make_record(ids[0], "finance"),
            make_record(ids[1], "none"),
            make_record(ids[2], fallback=True),
        ]
        labels = make_labels(ids[:2]) + make_labels([ids[2]], domain="none")
        report = ev.build_report(labels, emails, records, [], None)
        ts = report["systems"]["typesafe"]
        assert ts["n_errors"] == 1 and ts["n_scored"] == 2
        assert ts["strict_accuracy"]["n"] == 2 and ts["strict_accuracy"]["k"] == 1
        assert report["per_row"][ids[2]]["typesafe_status"] == "error"
        assert report["cutoff_choice"] is not None

    def test_agreement_rows_compare_classic_and_typesafe(self) -> None:
        ids = [eid_of(i) for i in range(2)]
        emails = {ids[0]: make_email(AGREEING), ids[1]: make_email(DISAGREEING)}
        records = [make_record(ids[0], "none"), make_record(ids[1], "finance")]
        labels = make_labels([ids[0]], domain="social_media") + make_labels([ids[1]])
        agreement = ev.build_report(labels, emails, records, [], None)["agreement"]
        assert agreement["n_rows"] == 1
        assert agreement["agreed_answer_correct"] == 1
        assert agreement["typesafe_scored"] == 1
        assert agreement["typesafe_correct"] == 0

    def test_report_without_cache_degrades_to_classic_only(self) -> None:
        ids = [eid_of(i) for i in range(2)]
        emails = {eid: make_email(DISAGREEING) for eid in ids}
        report = ev.build_report(make_labels(ids), emails, [], [], None)
        assert set(report["systems"]) == {"method1", "method2", "classic"}
        assert report["cutoff_sweep"] == [] and report["cutoff_choice"] is None
        assert report["typesafe_available"] is False
        # make_labels says the pipeline wrote these rows to email_finance.csv.
        assert report["classic_reproduces_source_file"] == 2
        assert "typesafe | missing" in ev.render_markdown(ev.round_floats(report))


class TestRuntime:
    RUNS = [
        {"set": "labeled", "rps": 10.0, "workers": 4, "model": "jev-latest"}
        | {"n_emails": 100, "n_calls_attempted": 100, "n_429": 0}
        | {"attempts_per_second": 10.0, "wall_seconds": 10.0},
        {"set": "sample", "rps": 20.0, "workers": 8, "model": "jev-latest"}
        | {"n_emails": 100, "n_calls_attempted": 100, "n_429": 0}
        | {"attempts_per_second": 20.0, "wall_seconds": 5.0},
    ]

    def test_concurrency_needed(self) -> None:
        assert ev.concurrency_needed(40, 0.288) == 12
        assert ev.concurrency_needed(40, 0.25) == 10
        assert ev.concurrency_needed(40, None) is None

    def test_extrapolation_is_per_run_not_pooled(self) -> None:
        records = [make_record(eid_of(i)) for i in range(4)]
        rt = ev.runtime_stats(records, self.RUNS, 0.5)
        full = rt["extrapolation"]["full_typesafe"]["hours_per_run_rate"]
        assert list(full.values()) == pytest.approx(
            [35_000 / 10 / 3600, 35_000 / 20 / 3600]
        )
        hybrid = rt["extrapolation"]["hybrid_labeled_disagreement"]
        assert hybrid["hours_at_cap"] == pytest.approx(17_500 / 40 / 3600)
        assert rt["max_rps_exercised"] == 20.0 and rt["n_429_total"] == 0
        assert rt["concurrency_needed_at_cap"]["p95"] == 5  # 120 ms latency

    def test_collection_info(self) -> None:
        info = ev.collection_info([make_record(eid_of(1))], self.RUNS)
        assert info["requested_models"] == ["jev-latest"]
        assert info["resolved_models"] == {"jev-test": 1}
        assert info["collection_dates"] == ["2026-10-01"]


class TestCache:
    @pytest.mark.parametrize("key", ["subject", "body", "sender", "receiver", "urls"])
    def test_allow_list_rejects_text_keys(self, tmp_path: Path, key: str) -> None:
        record = make_record(eid_of(1))
        record[key] = "text"
        cache = tmp_path / "cache.jsonl"
        with pytest.raises(ValueError, match="allow-list"):
            ev.append_record(cache, record)
        assert not cache.exists()

    def test_free_text_values_are_rejected(self) -> None:
        record = make_record(eid_of(1))
        record["error_type"] = "Some message: with spaces"
        with pytest.raises(ValueError):
            ev.validate_record(record)
        record = make_record(eid_of(1))
        record["probabilities"] = {"a subject line": 1.0}
        with pytest.raises(ValueError):
            ev.validate_record(record)

    def test_resume_prefers_successful_records(self, tmp_path: Path) -> None:
        cache = tmp_path / "cache.jsonl"
        ev.append_record(cache, make_record(eid_of(1), fallback=True))
        ev.append_record(cache, make_record(eid_of(2)))
        ev.append_record(cache, make_record(eid_of(1)))
        ev.append_record(cache, make_record(eid_of(2), fallback=True))
        records = ev.load_cache(cache)
        assert ev.completed_ids(records) == {eid_of(1), eid_of(2)}
        latest = ev.latest_records(records)
        assert not latest[eid_of(1)]["fallback"]
        assert not latest[eid_of(2)]["fallback"]


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def clock(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


class TestLimits:
    @pytest.mark.parametrize("rate", [4, 5, 10, 40])
    def test_rate_limiter_never_exceeds_rate(self, rate: int) -> None:
        fake = FakeClock()
        limiter = ev.RateLimiter(rate, clock=fake.clock, sleep=fake.sleep)
        starts = [limiter.acquire() for _ in range(5 * rate + 3)]
        for t in starts:
            assert sum(1 for s in starts if t <= s < t + 1) <= rate
        assert min(b - a for a, b in zip(starts, starts[1:])) >= 1 / rate
        assert starts[-1] == pytest.approx((5 * rate + 2) / rate)

    def test_recorder_measures_rate_limit_wait(self) -> None:
        recorder = ev.AttemptRecorder(ev.RateLimiter(20), ev.CallBudget(5))
        recorder.before_attempt()
        recorder.before_attempt()  # waits about 1/20 s for its slot
        assert recorder.wait_seconds >= 0.04
        recorder.reset()
        assert recorder.wait_seconds == 0.0

    def test_rate_limiter_rejects_non_positive_rate(self) -> None:
        with pytest.raises(ValueError):
            ev.RateLimiter(0)

    def test_budget_stops_at_max(self) -> None:
        budget = ev.CallBudget(3)
        assert [budget.try_acquire() for _ in range(4)] == [True, True, True, False]
        assert budget.used == 3 and budget.exhausted

    def test_recorder_raises_when_budget_is_spent(self) -> None:
        fake = FakeClock()
        limiter = ev.RateLimiter(10, clock=fake.clock, sleep=fake.sleep)
        recorder = ev.AttemptRecorder(limiter, ev.CallBudget(1))
        recorder.before_attempt()
        recorder.after_attempt(429)
        with pytest.raises(ev.BudgetExhausted):
            recorder.before_attempt()
        assert recorder.attempts == 1 and recorder.http_statuses == [429]


def test_counting_transport_counts_sdk_retries() -> None:
    """Every HTTP attempt, retries included, is charged and recorded."""
    httpx2 = pytest.importorskip("httpx2")
    sdk = pytest.importorskip("typesafe_sdk")
    seen: list[int] = []

    def handler(request: Any) -> Any:
        seen.append(1)
        return httpx2.Response(429, json={"error": {"message": "slow down"}})

    fake = FakeClock()
    budget = ev.CallBudget(10)
    recorder = ev.AttemptRecorder(
        ev.RateLimiter(40, clock=fake.clock, sleep=fake.sleep), budget
    )
    client = sdk.TypeSafeClient(
        api_key=FAKE_KEY,
        model="jev-test",
        retry=sdk.RetryPolicy(max_retries=2, backoff_initial=0, backoff_max=0),
        transport=ev.make_counting_transport(recorder, httpx2.MockTransport(handler)),
    )
    classifier = make_classifier(ev.ErrorCapturingClient(client, recorder))
    result = classifier.classify(make_email(DISAGREEING))
    assert result.details is not None and result.details["fallback"] is True
    assert len(seen) == recorder.attempts == budget.used == 3
    assert recorder.http_statuses == [429, 429, 429]
    assert budget.status_counts[429] == 3
    assert ev.derive_error_type(recorder) == "TypeSafeRateLimitError"


class FakeTypeSafeClient:
    """Fake client that reports one attempt per call to the recorder."""

    calls = 0

    def __init__(self, recorder: Any, error: Optional[Exception] = None) -> None:
        self.recorder = recorder
        self.error = error

    def system_one(self, **kwargs: Any) -> Any:
        FakeTypeSafeClient.calls += 1
        self.recorder.before_attempt()
        if self.error is not None:
            self.recorder.after_attempt(401)
            raise self.error
        self.recorder.after_attempt(200)
        return make_response("finance", 0.9)


class FakeAuthError(Exception):
    pass


class TestCollect:
    @pytest.fixture
    def inputs(self, tmp_path: Path) -> dict[str, Path]:
        rows = []
        for i in range(3):
            row = dict(DISAGREEING)
            row["subject"] = f"Statement {i}"
            rows.append(row)
        raw = tmp_path / "raw.csv"
        columns = ["sender", "receiver", "date", "subject", "body", "label", "urls"]
        with open(raw, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
        labels = tmp_path / "labels.csv"
        with open(labels, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(
                [
                    "email_id",
                    "raw_row",
                    "source_file",
                    "source_row",
                    "label",
                    "domain",
                    "confidence",
                    "ambiguous",
                    "rationale",
                    "labeler",
                    "verified",
                    "definition_ref",
                ]
            )
            for i, row in enumerate(rows):
                eid = ev.email_id(row["sender"], row["date"], row["subject"])
                writer.writerow(
                    [eid, i, "email_finance.csv", i, "1", "finance", "high"]
                    + ["false", "test", "agent", "false", "v1"]
                )
        return {
            "raw": raw,
            "labels": labels,
            "cache": tmp_path / "out" / "cache.jsonl",
            "runs": tmp_path / "out" / "runs.json",
        }

    @staticmethod
    def _argv(inputs: dict[str, Path], *extra: str) -> list[str]:
        argv = ["collect", "--set", "labeled", "--rps", "40", "--workers", "2"]
        for name, path in inputs.items():
            argv += [f"--{name}", str(path)]
        return argv + list(extra)

    @staticmethod
    def _patch_client(
        monkeypatch: pytest.MonkeyPatch, error: Optional[Exception] = None
    ) -> None:
        FakeTypeSafeClient.calls = 0
        monkeypatch.setenv("TYPESAFE_API_KEY", FAKE_KEY)
        monkeypatch.setattr(
            ev,
            "create_typesafe_client",
            lambda api_key, model, timeout, recorder: FakeTypeSafeClient(
                recorder, error
            ),
        )

    def test_missing_key_exits_with_clear_message(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        tmp_path: Path,
    ) -> None:
        monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
        code = ev.main(["collect", "--cache", str(tmp_path / "c.jsonl")])
        assert code == ev.EXIT_USAGE
        assert "TYPESAFE_API_KEY is not set" in capsys.readouterr().err
        assert not (tmp_path / "c.jsonl").exists()

    def test_rps_above_cap_is_refused(
        self, monkeypatch: pytest.MonkeyPatch, inputs: dict[str, Path]
    ) -> None:
        self._patch_client(monkeypatch)
        assert ev.main(self._argv(inputs, "--rps", "41")) == ev.EXIT_USAGE
        assert FakeTypeSafeClient.calls == 0

    def test_writes_allow_listed_records(
        self, monkeypatch: pytest.MonkeyPatch, inputs: dict[str, Path]
    ) -> None:
        self._patch_client(monkeypatch)
        assert ev.main(self._argv(inputs)) == ev.EXIT_OK
        text = inputs["cache"].read_text(encoding="utf-8")
        records = [json.loads(line) for line in text.splitlines()]
        assert len(records) == 3
        assert all(set(r) == ev.CACHE_KEYS for r in records)
        assert all(r["choice"] == "finance" and r["attempts"] == 1 for r in records)
        for field in ("Statement", "paypal", "bank account"):
            assert field not in text
        (run,) = json.loads(inputs["runs"].read_text(encoding="utf-8"))
        assert run["status"] == "complete" and run["n_calls_attempted"] == 3

        # A second run finds everything cached and makes no call.
        FakeTypeSafeClient.calls = 0
        assert ev.main(self._argv(inputs)) == ev.EXIT_OK
        assert FakeTypeSafeClient.calls == 0

    def test_failed_first_call_aborts(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        inputs: dict[str, Path],
    ) -> None:
        self._patch_client(monkeypatch, FakeAuthError("invalid key"))
        assert ev.main(self._argv(inputs)) == ev.EXIT_FIRST_CALL_FAILED
        err = capsys.readouterr().err
        assert "FakeAuthError" in err and "invalid key" not in err
        assert FAKE_KEY not in err
        assert FakeTypeSafeClient.calls == 1
        assert not inputs["cache"].exists()

    def test_budget_guard_stops_and_resume_finishes(
        self, monkeypatch: pytest.MonkeyPatch, inputs: dict[str, Path]
    ) -> None:
        self._patch_client(monkeypatch)
        code = ev.main(self._argv(inputs, "--max-calls", "2", "--workers", "1"))
        assert code == ev.EXIT_BUDGET_EXHAUSTED
        assert len(ev.load_cache(inputs["cache"])) == 2
        (run,) = json.loads(inputs["runs"].read_text(encoding="utf-8"))
        assert run["status"] == "budget_exhausted" and run["n_calls_attempted"] == 2

        FakeTypeSafeClient.calls = 0
        assert ev.main(self._argv(inputs, "--max-calls", "5")) == ev.EXIT_OK
        assert FakeTypeSafeClient.calls == 1
        assert len(ev.completed_ids(ev.load_cache(inputs["cache"]))) == 3
