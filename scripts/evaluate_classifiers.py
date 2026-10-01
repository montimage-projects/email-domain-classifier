#!/usr/bin/env python3
"""Evaluate the classic classifiers and TypeSafeClassifier on CEAS_08 (#19).

The script has two subcommands.

``collect`` (live, needs ``TYPESAFE_API_KEY``) runs the shipped
``TypeSafeClassifier.classify()`` on a set of CEAS_08 emails and appends one
record per email to a JSONL cache. A record holds identifiers, the model's
answer and call statistics only; it never holds email text. Every HTTP attempt,
SDK retries included, is counted, rate limited and charged to a call budget.

``labeled`` is the 180-row ground-truth set in
``data/ground-truth/ceas_08_domain_labels.csv``; ``sample`` is a seeded random
sample of the other CEAS_08 rows, used for token and runtime statistics only.

``report`` (offline, no key, no network) re-runs Method 1, Method 2 and the
classic combined classifier, loads the cached TypeSafe answers, replays the real
``HybridClassifier`` against them over a sweep of confidence cutoffs, and writes
the metrics as JSON plus a markdown summary.

Usage::

    TYPESAFE_API_KEY=... python scripts/evaluate_classifiers.py collect --set labeled
    TYPESAFE_API_KEY=... python scripts/evaluate_classifiers.py collect \\
        --set sample --sample-size 100
    python scripts/evaluate_classifiers.py report

``collect`` exit codes: 0 done, 1 call budget (``--max-calls``) exhausted,
2 usage or input error (including a missing key), 3 the first call failed.

Scoring follows ``docs/design/domain-profiles.md``: a ``None`` or ``unsure``
prediction counts as predicting ``none``. A failed TypeSafe call is an error,
never a ``none`` prediction.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import logging
import math
import os
import random
import re
import statistics
import sys
import threading
import time
from collections import Counter
from collections.abc import Callable, Hashable, Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any, Optional, cast

from email_classifier.classifier import (
    ClassificationResult,
    EmailClassifier,
    EmailData,
    HybridClassifier,
    KeywordTaxonomyClassifier,
    StructuralTemplateClassifier,
)
from email_classifier.domains import get_domain_names
from email_classifier.llm.config import (
    DEFAULT_LLM_CONFIDENCE_CUTOFF,
    DEFAULT_MODELS,
    LLMConfig,
    LLMProvider,
)
from email_classifier.llm.typesafe_classifier import (
    METHOD_NAME,
    NONE_OPTION,
    TypeSafeClassifier,
)
from email_classifier.processor import StreamingProcessor

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LABELS_PATH = REPO_ROOT / "data" / "ground-truth" / "ceas_08_domain_labels.csv"
DEFAULT_RAW_PATH = REPO_ROOT / "raw-data" / "CEAS_08.csv"
DEFAULT_CLASSIFIED_DIR = REPO_ROOT / "classified-data" / "ceas_08"
EVALUATION_DIR = REPO_ROOT / "data" / "evaluation"
DEFAULT_CACHE_PATH = EVALUATION_DIR / "typesafe_outputs.jsonl"
DEFAULT_RUNS_PATH = EVALUATION_DIR / "typesafe_runs.json"
DEFAULT_JSON_OUT = EVALUATION_DIR / "typesafe_evaluation_results.json"

API_KEY_ENV = "TYPESAFE_API_KEY"
DEFAULT_MODEL = DEFAULT_MODELS[LLMProvider.TYPESAFE]
DEFAULT_TIMEOUT_SECONDS = 30
DEFAULT_WORKERS = 4
DEFAULT_RPS = 10.0
# Hard cap on HTTP attempts per second; --rps above it is refused.
MAX_RPS = 40.0
DEFAULT_MAX_CALLS = 600
DEFAULT_SAMPLE_SIZE = 100
DEFAULT_SAMPLE_SEED = 19
# Same value as the SDK's RetryPolicy default; passed explicitly so the
# measured retry behavior is pinned.
SDK_MAX_RETRIES = 2

SET_LABELED = "labeled"
SET_SAMPLE = "sample"
SETS: tuple[str, ...] = (SET_LABELED, SET_SAMPLE)

EXIT_OK = 0
EXIT_BUDGET_EXHAUSTED = 1
EXIT_USAGE = 2
EXIT_FIRST_CALL_FAILED = 3

NONE_LABEL = "none"
UNSURE_LABEL = "unsure"
INCUMBENT_CUTOFF = DEFAULT_LLM_CONFIDENCE_CUTOFF
CUTOFF_SWEEP: tuple[float, ...] = tuple(round(i * 0.05, 2) for i in range(20))
MIN_SUPPORT = 5
SIGNIFICANCE_LEVEL = 0.05
EXTRAPOLATION_EMAILS = 35_000
# Method agreement rate the reporter measured on the full CEAS_08 corpus
# (classified-data/ceas_08/classification_report.txt).
FULL_CORPUS_AGREEMENT_RATE = 0.2162
FLOAT_DIGITS = 4
PROBABILITY_DIGITS = 6

# Status recorded for an attempt that got no HTTP response (connection error,
# timeout).
NO_RESPONSE_STATUS = 0

STATUS_OK = "ok"
STATUS_ERROR = "error"
STATUS_MISSING = "missing"

# The only keys a cache record may hold. None of them can carry email text.
CACHE_KEYS: frozenset[str] = frozenset(
    {
        "email_id",
        "set",
        "choice",
        "probabilities",
        "confidence",
        "domain",
        "fallback",
        "error_type",
        "usage",
        "latency_ms",
        "attempts",
        "http_statuses",
        "model",
        "timestamp",
    }
)
USAGE_KEYS: frozenset[str] = frozenset({"input_tokens", "output_tokens"})

_EMAIL_ID_RE = re.compile(r"^[0-9a-f]{16}$")
_ERROR_TYPE_RE = re.compile(r"^[A-Za-z0-9_.]{1,64}$")
_MODEL_RE = re.compile(r"^[A-Za-z0-9_.:/@-]{1,128}$")
_TIMESTAMP_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?\+00:00$")

EmailKey = tuple[str, str, str]
CachedRecord = dict[str, Any]


def _load_sampler() -> ModuleType:
    """Load ``scripts/sample_ground_truth.py``, which is not a package module."""
    name = "sample_ground_truth"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(
        name, Path(__file__).with_name("sample_ground_truth.py")
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_sampler = _load_sampler()


class InputError(Exception):
    """An input file or option is missing or invalid."""


class BudgetExhausted(Exception):
    """Raised before an HTTP attempt that would exceed ``--max-calls``.

    A plain ``Exception``: the TypeSafe SDK neither wraps nor retries it.
    """


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------


def email_id(sender: str, date: str, subject: str) -> str:
    """Return the stable 16-hex-digit id the ground-truth set uses."""
    return str(_sampler.email_id(sender, date, subject))


def is_lfs_pointer(path: Path) -> bool:
    """Return True when ``path`` is a Git LFS pointer, not the real file."""
    return bool(_sampler.is_lfs_pointer(path))


def read_csv_rows(path: Path) -> Iterator[dict[str, str]]:
    """Yield the data rows of a CSV file that may have very large fields."""
    rows: Iterator[dict[str, str]] = _sampler._read_rows(path)
    yield from rows


def require_real_file(path: Path) -> None:
    """Raise InputError unless ``path`` exists and is not an LFS pointer."""
    if not path.exists() or is_lfs_pointer(path):
        raise InputError(f"{path} is missing or an LFS pointer; run `git lfs pull`")


@dataclass(frozen=True)
class LabeledEmail:
    """One row of the ground-truth set (identifiers and labels only)."""

    email_id: str
    raw_row: int
    source_file: str
    label: str
    domain: str
    ambiguous: bool
    verified: bool


def load_labels(path: Path) -> list[LabeledEmail]:
    """Load the ground-truth set.

    Args:
        path: Path to ``ceas_08_domain_labels.csv``.

    Returns:
        One LabeledEmail per row, in file order.
    """
    if not path.exists():
        raise InputError(f"{path} not found")
    with open(path, newline="", encoding="utf-8") as fh:
        return [
            LabeledEmail(
                email_id=row["email_id"],
                raw_row=int(row["raw_row"]),
                source_file=row["source_file"],
                label=row["label"],
                domain=row["domain"],
                ambiguous=row["ambiguous"] == "true",
                verified=row["verified"] == "true",
            )
            for row in csv.DictReader(fh)
        ]


def _row_key(row: Mapping[str, str]) -> EmailKey:
    return (row["sender"], row["date"], row["subject"])


def build_email(row: Mapping[str, str], processor: StreamingProcessor) -> EmailData:
    """Build an EmailData exactly as the pipeline does for a CSV row."""
    return EmailData.from_dict(processor._normalize_row(dict(row)))


def resolve_labeled_emails(
    labels: Sequence[LabeledEmail], raw_path: Path
) -> dict[str, EmailData]:
    """Look up each labeled email in the raw CSV and check its id.

    Args:
        labels: Ground-truth rows.
        raw_path: Path to ``raw-data/CEAS_08.csv``.

    Returns:
        EmailData keyed by email_id.

    Raises:
        InputError: If a row is missing or its email_id does not match.
    """
    wanted = {label.raw_row for label in labels}
    rows: dict[int, dict[str, str]] = {}
    for i, row in enumerate(read_csv_rows(raw_path)):
        if i in wanted:
            rows[i] = row
    processor = StreamingProcessor()
    emails: dict[str, EmailData] = {}
    for label in labels:
        found = rows.get(label.raw_row)
        if found is None:
            raise InputError(f"raw_row {label.raw_row} not found in {raw_path}")
        if email_id(*_row_key(found)) != label.email_id:
            raise InputError(
                f"email_id mismatch at raw_row {label.raw_row} "
                f"(expected {label.email_id})"
            )
        emails[label.email_id] = build_email(found, processor)
    return emails


def select_sample_indices(
    keys: Sequence[EmailKey], size: int, seed: int, exclude: set[int]
) -> list[int]:
    """Pick a seeded random sample of row indices.

    Rows in ``exclude`` and rows whose (sender, date, subject) key is not unique
    are never picked, so every picked row has a unique email_id.

    Args:
        keys: The (sender, date, subject) key of every raw row, in file order.
        size: Number of rows wanted; fewer come back if fewer are eligible.
        seed: Random seed.
        exclude: Row indices that must not be picked.

    Returns:
        The picked row indices, sorted.
    """
    counts = Counter(keys)
    eligible = [
        i for i, key in enumerate(keys) if i not in exclude and counts[key] == 1
    ]
    rng = random.Random(f"{seed}:{SET_SAMPLE}")
    return sorted(rng.sample(eligible, min(size, len(eligible))))


def draw_sample_emails(
    raw_path: Path, size: int, seed: int, exclude: set[int]
) -> list[tuple[str, EmailData]]:
    """Draw the ``sample`` set: (email_id, EmailData) pairs in raw-row order."""
    rows = list(read_csv_rows(raw_path))
    picked = select_sample_indices([_row_key(r) for r in rows], size, seed, exclude)
    processor = StreamingProcessor()
    return [
        (email_id(*_row_key(rows[i])), build_email(rows[i], processor)) for i in picked
    ]


# --------------------------------------------------------------------------
# Cache records
# --------------------------------------------------------------------------


def _answer_options() -> set[str]:
    return set(get_domain_names()) | {NONE_OPTION}


def validate_record(record: Mapping[str, Any]) -> None:
    """Check that a cache record holds only allow-listed, text-free values.

    Args:
        record: Cache record.

    Raises:
        ValueError: If a key is outside the allow-list or missing, or a value
            has a shape that could carry free text.
    """
    unknown = set(record) - CACHE_KEYS
    if unknown:
        raise ValueError(
            f"cache record has keys outside the allow-list: {sorted(unknown)}"
        )
    missing = CACHE_KEYS - set(record)
    if missing:
        raise ValueError(f"cache record is missing keys: {sorted(missing)}")

    options = _answer_options()
    if not _EMAIL_ID_RE.match(str(record["email_id"])):
        raise ValueError("cache record email_id is not a 16-hex-digit id")
    if record["set"] not in SETS:
        raise ValueError(f"cache record set must be one of {SETS}")
    if record["choice"] is not None and record["choice"] not in options:
        raise ValueError("cache record choice is not an answer option")
    if record["domain"] is not None and record["domain"] not in get_domain_names():
        raise ValueError("cache record domain is not a domain name")
    probabilities = record["probabilities"]
    if not isinstance(probabilities, dict) or not set(probabilities) <= options:
        raise ValueError("cache record probabilities must map answer options")
    if not all(isinstance(v, (int, float)) for v in probabilities.values()):
        raise ValueError("cache record probabilities must be numbers")
    if not isinstance(record["fallback"], bool):
        raise ValueError("cache record fallback must be a bool")
    error_type = record["error_type"]
    if error_type is not None and not _ERROR_TYPE_RE.match(str(error_type)):
        raise ValueError("cache record error_type must be a short identifier")
    usage = record["usage"]
    if usage is not None:
        if not isinstance(usage, dict) or not set(usage) <= USAGE_KEYS:
            raise ValueError("cache record usage may hold token counts only")
        if not all(v is None or isinstance(v, int) for v in usage.values()):
            raise ValueError("cache record token counts must be integers")
    statuses = record["http_statuses"]
    if not isinstance(statuses, list) or not all(isinstance(s, int) for s in statuses):
        raise ValueError("cache record http_statuses must be a list of integers")
    if not isinstance(record["attempts"], int):
        raise ValueError("cache record attempts must be an integer")
    for name in ("confidence", "latency_ms"):
        if not isinstance(record[name], (int, float)):
            raise ValueError(f"cache record {name} must be a number")
    if record["model"] is not None and not _MODEL_RE.match(str(record["model"])):
        raise ValueError("cache record model must be a model identifier")
    if not _TIMESTAMP_RE.match(str(record["timestamp"])):
        raise ValueError("cache record timestamp must be an ISO 8601 UTC time")


def append_record(path: Path, record: Mapping[str, Any]) -> None:
    """Validate a record and append it to the JSONL cache, flushed at once."""
    validate_record(record)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, sort_keys=True) + "\n")
        fh.flush()


def load_cache(path: Path) -> list[CachedRecord]:
    """Load and validate every record of the JSONL cache (empty if absent)."""
    if not path.exists():
        return []
    records: list[CachedRecord] = []
    with open(path, encoding="utf-8") as fh:
        for n, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            try:
                validate_record(record)
            except ValueError as error:
                raise InputError(f"{path}:{n}: {error}") from None
            records.append(record)
    return records


def completed_ids(records: Sequence[Mapping[str, Any]]) -> set[str]:
    """Return the ids that already have a successful (non-fallback) record."""
    return {str(r["email_id"]) for r in records if not r["fallback"]}


def latest_records(records: Sequence[CachedRecord]) -> dict[str, CachedRecord]:
    """Pick one record per email_id: the last success, else the last failure."""
    chosen: dict[str, CachedRecord] = {}
    for record in records:
        eid = str(record["email_id"])
        previous = chosen.get(eid)
        if previous is None or not record["fallback"] or previous["fallback"]:
            chosen[eid] = record
    return chosen


def utc_now() -> str:
    """Return the current UTC time in ISO 8601 with seconds."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def build_record(
    *,
    eid: str,
    set_name: str,
    result: ClassificationResult,
    latency_ms: float,
    attempts: int,
    http_statuses: Sequence[int],
    error_type: Optional[str],
    model: str,
    timestamp: str,
) -> CachedRecord:
    """Turn a TypeSafeClassifier result into an allow-listed cache record.

    The error message in ``result.details`` is dropped: only ``error_type``
    (an exception class name or HTTP status code) is kept.
    """
    details = result.details or {}
    fallback = bool(details.get("fallback"))
    if fallback:
        choice: Optional[str] = None
        probabilities: dict[str, float] = {}
        usage: Optional[dict[str, Any]] = None
        recorded_model = model
    else:
        choice = str(details.get("choice"))
        probabilities = {
            option: round(float(value), PROBABILITY_DIGITS)
            for option, value in sorted(details.get("probabilities", {}).items())
        }
        raw_usage = details.get("usage") or {}
        usage = {key: raw_usage.get(key) for key in sorted(USAGE_KEYS)}
        recorded_model = str(details.get("model") or model)
    return {
        "email_id": eid,
        "set": set_name,
        "choice": choice,
        "probabilities": probabilities,
        "confidence": float(result.confidence),
        "domain": result.domain,
        "fallback": fallback,
        "error_type": error_type if fallback else None,
        "usage": usage,
        "latency_ms": round(latency_ms, 1),
        "attempts": attempts,
        "http_statuses": list(http_statuses),
        "model": recorded_model,
        "timestamp": timestamp,
    }


def append_run_summary(path: Path, summary: Mapping[str, Any]) -> None:
    """Append one run summary to the JSON list in ``path``."""
    runs: list[Any] = []
    if path.exists():
        runs = json.loads(path.read_text(encoding="utf-8"))
    runs.append(dict(summary))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(runs, indent=2) + "\n", encoding="utf-8")


def load_runs(path: Path) -> list[dict[str, Any]]:
    """Load the run summaries (empty if the file is absent)."""
    if not path.exists():
        return []
    runs: list[dict[str, Any]] = json.loads(path.read_text(encoding="utf-8"))
    return runs


# --------------------------------------------------------------------------
# Rate limit, budget and attempt counting
# --------------------------------------------------------------------------


class RateLimiter:
    """Thread-safe limiter that spaces calls at least ``1 / rate`` apart.

    Any half-open window of one second therefore holds at most ``rate`` calls.
    """

    def __init__(
        self,
        rate: float,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        """Create a limiter.

        Args:
            rate: Maximum calls per second; must be positive.
            clock: Monotonic clock in seconds (injectable for tests).
            sleep: Sleep function (injectable for tests).
        """
        if not rate > 0:
            raise ValueError("rate must be positive")
        self.interval = 1.0 / rate
        self._clock = clock
        self._sleep = sleep
        self._lock = threading.Lock()
        self._next: Optional[float] = None

    def acquire(self) -> float:
        """Wait for the next free slot and return its start time."""
        with self._lock:
            now = self._clock()
            slot = now if self._next is None else max(now, self._next)
            self._next = slot + self.interval
        wait = slot - now
        if wait > 0:
            self._sleep(wait)
        return slot


class CallBudget:
    """Thread-safe count of HTTP attempts against a maximum."""

    def __init__(self, max_calls: int) -> None:
        """Create a budget of ``max_calls`` attempts."""
        self.max_calls = max_calls
        self._used = 0
        self._lock = threading.Lock()
        self.exhausted = False
        self.status_counts: Counter[int] = Counter()

    @property
    def used(self) -> int:
        """Attempts charged so far."""
        return self._used

    def try_acquire(self) -> bool:
        """Charge one attempt; return False (and mark exhausted) if none is left."""
        with self._lock:
            if self._used >= self.max_calls:
                self.exhausted = True
                return False
            self._used += 1
            return True

    def record_status(self, status: int) -> None:
        """Count the HTTP status of one attempt."""
        with self._lock:
            self.status_counts[status] += 1


class AttemptRecorder:
    """Per-thread record of the HTTP attempts made for the current email.

    The counting transport calls ``before_attempt`` and ``after_attempt``
    around every HTTP attempt, SDK retries included.
    """

    def __init__(self, limiter: RateLimiter, budget: CallBudget) -> None:
        """Create a recorder sharing the run's limiter and budget."""
        self.limiter = limiter
        self.budget = budget
        self.attempts = 0
        self.http_statuses: list[int] = []
        self.error_type: Optional[str] = None

    def reset(self) -> None:
        """Start recording a new email."""
        self.attempts = 0
        self.http_statuses = []
        self.error_type = None

    def before_attempt(self) -> None:
        """Charge the budget, then wait for a rate-limit slot.

        Raises:
            BudgetExhausted: If the budget has no attempt left.
        """
        if not self.budget.try_acquire():
            raise BudgetExhausted("call budget exhausted")
        self.limiter.acquire()
        self.attempts += 1

    def after_attempt(self, status: Optional[int]) -> None:
        """Record an attempt's HTTP status (None: no response)."""
        code = NO_RESPONSE_STATUS if status is None else int(status)
        self.http_statuses.append(code)
        self.budget.record_status(code)


def make_counting_transport(recorder: AttemptRecorder, inner: Any = None) -> Any:
    """Build an httpx2 transport that reports every attempt to ``recorder``.

    ``httpx2`` ships with ``typesafe-sdk`` and is imported here, not at module
    level, so the offline ``report`` works without the SDK.

    Args:
        recorder: Charged before, and told the status after, every attempt.
        inner: Transport that sends the request (default: a new
            ``httpx2.HTTPTransport``; tests pass a mock).
    """
    import httpx2

    class CountingTransport(httpx2.BaseTransport):
        def __init__(self) -> None:
            self._inner = inner if inner is not None else httpx2.HTTPTransport()

        def handle_request(self, request: Any) -> Any:
            recorder.before_attempt()
            try:
                response = self._inner.handle_request(request)
            except Exception:
                recorder.after_attempt(None)
                raise
            recorder.after_attempt(response.status_code)
            return response

        def close(self) -> None:
            self._inner.close()

    return CountingTransport()


def create_typesafe_client(
    api_key: str, model: str, timeout: float, recorder: AttemptRecorder
) -> Any:
    """Create the TypeSafe client used by ``collect`` (tests replace this).

    Args:
        api_key: TypeSafe API key.
        model: Model name.
        timeout: Per-request timeout in seconds.
        recorder: Receives every HTTP attempt.

    Returns:
        A ``typesafe_sdk.TypeSafeClient`` with the SDK's default retry policy
        and a counting, rate-limited transport.
    """
    from typesafe_sdk import RetryPolicy, TypeSafeClient

    return TypeSafeClient(
        api_key=api_key,
        model=model,
        retry=RetryPolicy(max_retries=SDK_MAX_RETRIES),
        timeout=timeout,
        transport=make_counting_transport(recorder),
    )


class ErrorCapturingClient:
    """Client proxy that records the class name of any error it raises.

    TypeSafeClassifier keeps only the error message, which may echo request
    content; the class name is safe to store.
    """

    def __init__(self, inner: Any, recorder: AttemptRecorder) -> None:
        """Wrap ``inner`` and report errors to ``recorder``."""
        self._inner = inner
        self._recorder = recorder

    def system_one(self, **kwargs: Any) -> Any:
        """Forward to the wrapped client's ``system_one``."""
        try:
            return self._inner.system_one(**kwargs)
        except Exception as error:
            self._recorder.error_type = type(error).__name__
            raise


def derive_error_type(recorder: AttemptRecorder) -> str:
    """Name a failed call without using its message text."""
    if recorder.error_type:
        return recorder.error_type
    if recorder.http_statuses and recorder.http_statuses[-1] >= 400:
        return f"http_{recorder.http_statuses[-1]}"
    return "ResponseConversionError"


class Collector:
    """Runs TypeSafeClassifier on emails, one client per worker thread."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        timeout: int,
        limiter: RateLimiter,
        budget: CallBudget,
        set_name: str,
    ) -> None:
        """Store the run settings; clients are created lazily per thread."""
        self._api_key = api_key
        self.model = model
        self.timeout = timeout
        self.limiter = limiter
        self.budget = budget
        self.set_name = set_name
        self._local = threading.local()

    def _worker(self) -> tuple[TypeSafeClassifier, AttemptRecorder]:
        worker = getattr(self._local, "worker", None)
        if worker is None:
            recorder = AttemptRecorder(self.limiter, self.budget)
            client = ErrorCapturingClient(
                create_typesafe_client(
                    self._api_key, self.model, float(self.timeout), recorder
                ),
                recorder,
            )
            config = LLMConfig(
                provider=LLMProvider.TYPESAFE,
                model=self.model,
                api_key=self._api_key,
                timeout=self.timeout,
            )
            worker = (TypeSafeClassifier(config, client=client), recorder)
            self._local.worker = worker
        return cast(tuple[TypeSafeClassifier, AttemptRecorder], worker)

    def classify(self, eid: str, email: EmailData) -> Optional[CachedRecord]:
        """Classify one email and build its record.

        Returns:
            The cache record, or None when the call budget refused an attempt
            (the email is left for a later run).
        """
        classifier, recorder = self._worker()
        recorder.reset()
        start = time.perf_counter()
        result = classifier.classify(email)
        latency_ms = (time.perf_counter() - start) * 1000
        fallback = bool((result.details or {}).get("fallback"))
        if fallback and recorder.error_type == BudgetExhausted.__name__:
            return None
        return build_record(
            eid=eid,
            set_name=self.set_name,
            result=result,
            latency_ms=latency_ms,
            attempts=recorder.attempts,
            http_statuses=recorder.http_statuses,
            error_type=derive_error_type(recorder) if fallback else None,
            model=self.model,
            timestamp=utc_now(),
        )


def _quiet_classifier_logs() -> None:
    """Keep failure messages (which may echo request content) off the console."""
    for name in ("email_classifier.llm.typesafe_classifier", "typesafe_sdk", "httpx2"):
        logging.getLogger(name).setLevel(logging.ERROR)


def _collect_emails(args: argparse.Namespace) -> list[tuple[str, EmailData]]:
    require_real_file(args.raw)
    labels = load_labels(args.labels)
    if args.set == SET_LABELED:
        emails = resolve_labeled_emails(labels, args.raw)
        return [(label.email_id, emails[label.email_id]) for label in labels]
    exclude = {label.raw_row for label in labels}
    return draw_sample_emails(args.raw, args.sample_size, args.seed, exclude)


def run_collect(args: argparse.Namespace) -> int:
    """Run the ``collect`` subcommand; see the module docstring for exit codes."""
    api_key = os.environ.get(API_KEY_ENV, "").strip()
    if not api_key:
        print(
            f"error: {API_KEY_ENV} is not set. Export it in the environment; "
            "collect reads the key from nowhere else.",
            file=sys.stderr,
        )
        return EXIT_USAGE
    if not 0 < args.rps <= MAX_RPS:
        print(f"error: --rps must be above 0 and at most {MAX_RPS:g}", file=sys.stderr)
        return EXIT_USAGE
    if args.workers < 1 or args.max_calls < 1 or args.sample_size < 1:
        print(
            "error: --workers, --max-calls and --sample-size must be >= 1",
            file=sys.stderr,
        )
        return EXIT_USAGE

    _quiet_classifier_logs()
    emails = _collect_emails(args)
    done = completed_ids(load_cache(args.cache))
    pending = [(eid, email) for eid, email in emails if eid not in done]
    print(
        f"{args.set}: {len(emails)} emails, {len(emails) - len(pending)} cached, "
        f"{len(pending)} to classify",
        file=sys.stderr,
    )
    if not pending:
        return EXIT_OK

    budget = CallBudget(args.max_calls)
    limiter = RateLimiter(args.rps)
    collector = Collector(
        api_key=api_key,
        model=args.model,
        timeout=args.timeout,
        limiter=limiter,
        budget=budget,
        set_name=args.set,
    )
    started = utc_now()
    t0 = time.perf_counter()
    n_written = 0
    n_errors = 0

    # Fail fast: the first email alone, before any concurrency.
    first_id, first_email = pending[0]
    first = collector.classify(first_id, first_email)
    if first is not None and first["fallback"]:
        print(
            f"error: the first call failed ({first['error_type']}); aborting "
            "before the full pass. Check the key, model and network.",
            file=sys.stderr,
        )
        return EXIT_FIRST_CALL_FAILED
    if first is not None:
        append_record(args.cache, first)
        n_written += 1

    stop = threading.Event()

    def task(eid: str, email: EmailData) -> Optional[CachedRecord]:
        if stop.is_set():
            return None
        record = collector.classify(eid, email)
        if record is None:
            stop.set()
        return record

    rest = pending[1:] if first is not None else []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(task, eid, email) for eid, email in rest]
        for future in as_completed(futures):
            record = future.result()
            if record is None:
                continue
            append_record(args.cache, record)
            n_written += 1
            n_errors += int(record["fallback"])
            if n_written % 20 == 0:
                print(f"  {n_written}/{len(pending)} written", file=sys.stderr)

    wall = time.perf_counter() - t0
    status = "budget_exhausted" if budget.exhausted else "complete"
    append_run_summary(
        args.runs,
        {
            "set": args.set,
            "model": args.model,
            "status": status,
            "started_utc": started,
            "finished_utc": utc_now(),
            "wall_seconds": round(wall, 3),
            "n_emails": n_written,
            "n_pending": len(pending),
            "n_calls_attempted": budget.used,
            "n_429": budget.status_counts.get(429, 0),
            "n_errors": n_errors,
            "rps": args.rps,
            "workers": args.workers,
            "max_calls": args.max_calls,
            "emails_per_second": round(n_written / wall, 4) if wall > 0 else None,
            "attempts_per_second": round(budget.used / wall, 4) if wall > 0 else None,
        },
    )
    print(
        f"wrote {n_written} records ({n_errors} errors), "
        f"{budget.used} HTTP attempts in {wall:.1f}s",
        file=sys.stderr,
    )
    if budget.exhausted:
        print(
            f"error: --max-calls {args.max_calls} reached; "
            f"{len(pending) - n_written} emails left for a later run",
            file=sys.stderr,
        )
        return EXIT_BUDGET_EXHAUSTED
    return EXIT_OK


# --------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------


def wilson_interval(k: int, n: int, z: float = 1.96) -> Optional[tuple[float, float]]:
    """Wilson score interval for a binomial proportion.

    Args:
        k: Successes.
        n: Trials.
        z: Normal quantile (1.96 gives a 95% interval).

    Returns:
        (low, high), or None when ``n`` is 0 (no data, no interval).
    """
    if n == 0:
        return None
    p = k / n
    denominator = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return (max(0.0, centre - half), min(1.0, centre + half))


def proportion(k: int, n: int) -> dict[str, Any]:
    """Return k, n, the rate and its Wilson 95% interval (None when n = 0)."""
    interval = wilson_interval(k, n)
    return {
        "k": k,
        "n": n,
        "rate": k / n if n else None,
        "ci_low": interval[0] if interval else None,
        "ci_high": interval[1] if interval else None,
    }


def mcnemar_exact(b: int, c: int) -> float:
    """Exact two-sided McNemar test on the discordant pair counts.

    Args:
        b: Pairs where only the first system is correct.
        c: Pairs where only the second system is correct.

    Returns:
        The two-sided p-value (1.0 when there are no discordant pairs).
    """
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1)) / (1 << n)
    return min(1.0, 2 * tail)


def percentile(values: Sequence[float], q: float) -> Optional[float]:
    """Percentile with linear interpolation between closest ranks.

    Same as numpy's default ("linear", Hyndman-Fan type 7).

    Args:
        values: Data.
        q: Percentile from 0 to 100.

    Returns:
        The percentile, or None for empty data.
    """
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * q / 100
    low = math.floor(position)
    high = math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def normalize_prediction(prediction: Optional[str]) -> str:
    """Map a prediction to a scored label: None, '' and 'unsure' become 'none'."""
    if prediction is None or prediction in ("", UNSURE_LABEL):
        return NONE_LABEL
    return prediction


def per_class_metrics(
    pairs: Sequence[tuple[str, str]], min_support: int = MIN_SUPPORT
) -> dict[str, dict[str, Any]]:
    """Precision, recall and F1 per class over (truth, prediction) pairs.

    Precision is None for a class that is never predicted, and recall is None
    for a class with no support; F1 then treats the missing value as 0.

    Returns:
        Metrics keyed by class, sorted; ``insufficient_support`` is True when
        the class has fewer than ``min_support`` true rows.
    """
    classes = sorted({t for t, _ in pairs} | {p for _, p in pairs})
    metrics: dict[str, dict[str, Any]] = {}
    for cls in classes:
        tp = sum(1 for t, p in pairs if t == cls and p == cls)
        predicted = sum(1 for _, p in pairs if p == cls)
        support = sum(1 for t, _ in pairs if t == cls)
        precision = tp / predicted if predicted else None
        recall = tp / support if support else None
        p, r = precision or 0.0, recall or 0.0
        metrics[cls] = {
            "precision": precision,
            "recall": recall,
            "f1": 2 * p * r / (p + r) if p + r else 0.0,
            "support": support,
            "predicted": predicted,
            "insufficient_support": support < min_support,
        }
    return metrics


def macro_f1(
    per_class: Mapping[str, Mapping[str, Any]], min_support: int = MIN_SUPPORT
) -> tuple[Optional[float], list[str]]:
    """Mean F1 over the classes with at least ``min_support`` true rows.

    Returns:
        (macro-F1 or None if no class qualifies, the classes used).
    """
    used = sorted(c for c, m in per_class.items() if m["support"] >= min_support)
    if not used:
        return None, []
    return sum(float(per_class[c]["f1"]) for c in used) / len(used), used


def classification_metrics(
    pairs: Sequence[tuple[str, str]], min_support: int = MIN_SUPPORT
) -> dict[str, Any]:
    """All metrics for one system over scored (truth, prediction) pairs.

    The strict view counts an abstention as a ``none`` prediction. The coverage
    view looks only at rows where the system answered a domain.
    """
    n = len(pairs)
    correct = sum(1 for t, p in pairs if t == p)
    answered = [(t, p) for t, p in pairs if p != NONE_LABEL]
    answered_correct = sum(1 for t, p in answered if t == p)
    per_class = per_class_metrics(pairs, min_support)
    none_metrics = per_class.get(NONE_LABEL, {})
    macro, macro_classes = macro_f1(per_class, min_support)
    confusion: dict[str, dict[str, int]] = {}
    for t, p in sorted(pairs):
        confusion.setdefault(t, {}).setdefault(p, 0)
        confusion[t][p] += 1
    return {
        "strict_accuracy": proportion(correct, n),
        "answer_rate": proportion(len(answered), n),
        "answered_accuracy": proportion(answered_correct, len(answered)),
        "none_precision": none_metrics.get("precision"),
        "none_recall": none_metrics.get("recall"),
        "macro_f1": macro,
        "macro_f1_classes": macro_classes,
        "per_class": per_class,
        "confusion": confusion,
    }


def stratum_weighted_accuracy(
    items: Sequence[tuple[Hashable, bool]], stratum_sizes: Mapping[Hashable, int]
) -> Optional[float]:
    """Population accuracy estimate from a stratified sample.

    Each row is weighted by (stratum population size / stratum sample size).

    Args:
        items: (stratum, correct) per scored row.
        stratum_sizes: Population size of every stratum.

    Returns:
        The weighted accuracy, or None if there are no rows or a stratum's
        population size is unknown.
    """
    sample_sizes = Counter(stratum for stratum, _ in items)
    if not sample_sizes or any(s not in stratum_sizes for s in sample_sizes):
        return None
    total_weight = 0.0
    correct_weight = 0.0
    for stratum, correct in items:
        weight = stratum_sizes[stratum] / sample_sizes[stratum]
        total_weight += weight
        correct_weight += weight if correct else 0.0
    return correct_weight / total_weight if total_weight else None


def default_stratum(row: LabeledEmail) -> tuple[str, str]:
    """Sampling stratum of a ground-truth row: (output file, CEAS_08 label)."""
    return (row.source_file, row.label)


def count_strata(
    classified_dir: Path, files: Sequence[str]
) -> Optional[dict[Hashable, int]]:
    """Count the rows of each (output file, label) stratum.

    Returns:
        Population sizes, or None if any file is missing or an LFS pointer.
    """
    paths = [classified_dir / name for name in files]
    if not all(p.exists() and not is_lfs_pointer(p) for p in paths):
        return None
    sizes: dict[Hashable, int] = {}
    for name, path in zip(files, paths):
        for row in read_csv_rows(path):
            key = (name, row.get("label", ""))
            sizes[key] = sizes.get(key, 0) + 1
    return sizes


# --------------------------------------------------------------------------
# Hybrid replay and cutoff sweep
# --------------------------------------------------------------------------


def result_from_record(record: Optional[Mapping[str, Any]]) -> ClassificationResult:
    """Rebuild TypeSafeClassifier's ClassificationResult from a cache record.

    A missing record becomes a failed call, which the hybrid gate rejects.
    """
    names = get_domain_names()
    if record is None or record["fallback"]:
        return ClassificationResult(
            domain=None,
            confidence=0.0,
            scores={name: 0.0 for name in names},
            method=METHOD_NAME,
            details={
                "fallback": True,
                "error": STATUS_MISSING if record is None else record["error_type"],
            },
        )
    probabilities = {k: float(v) for k, v in record["probabilities"].items()}
    choice = record["choice"]
    return ClassificationResult(
        domain=choice if choice in names else None,
        confidence=float(record["confidence"]),
        scores={name: probabilities.get(name, 0.0) for name in names},
        method=METHOD_NAME,
        details={"choice": choice, "probabilities": probabilities, "fallback": False},
    )


class CachedLLMClassifier:
    """Method 3 stand-in that answers from cached TypeSafe outputs.

    The replay loop sets ``current_id`` before each ``classify`` call.
    """

    def __init__(self, records: Mapping[str, Mapping[str, Any]]) -> None:
        """Answer from ``records`` keyed by email_id."""
        self._records = records
        self.current_id: Optional[str] = None
        self.calls = 0

    def classify(self, email: EmailData) -> ClassificationResult:
        """Return the cached answer for ``current_id``."""
        self.calls += 1
        if self.current_id is None:
            return result_from_record(None)
        return result_from_record(self._records.get(self.current_id))


HybridFactory = Callable[[float], HybridClassifier]


def default_hybrid_factory(cutoff: float) -> HybridClassifier:
    """Build the real HybridClassifier with no LLM of its own."""
    return HybridClassifier(llm_confidence_cutoff=cutoff)


@dataclass(frozen=True)
class ReplayOutcome:
    """What the real HybridClassifier did with one email."""

    email_id: str
    prediction: str
    path: str
    accepted: Optional[bool]


def replay_hybrid(
    cutoff: Optional[float],
    items: Sequence[tuple[str, EmailData]],
    records: Optional[Mapping[str, Mapping[str, Any]]],
    hybrid_factory: HybridFactory = default_hybrid_factory,
) -> list[ReplayOutcome]:
    """Run the real HybridClassifier over ``items`` with cached LLM answers.

    Args:
        cutoff: Confidence cutoff; ignored when ``records`` is None.
        items: (email_id, EmailData) pairs.
        records: Cached TypeSafe records by email_id, or None to run without an
            LLM (every disagreement goes to the classic weighted fallback).
        hybrid_factory: Builds a HybridClassifier for a cutoff.

    Returns:
        One ReplayOutcome per item, in order; ``prediction`` is normalized.
    """
    hybrid = hybrid_factory(INCUMBENT_CUTOFF if cutoff is None else cutoff)
    stub: Optional[CachedLLMClassifier] = None
    if records is not None:
        stub = CachedLLMClassifier(records)
    hybrid.llm_classifier = cast(Any, stub)
    outcomes: list[ReplayOutcome] = []
    for eid, email in items:
        if stub is not None:
            stub.current_id = eid
        domain, details = hybrid.classify(email)
        gate = details.get("llm_gate")
        outcomes.append(
            ReplayOutcome(
                email_id=eid,
                prediction=normalize_prediction(domain),
                path=str(details["path"]),
                accepted=bool(gate["accepted"]) if gate else None,
            )
        )
    return outcomes


@dataclass(frozen=True)
class EvalRow:
    """One labeled email with every system's normalized prediction."""

    email_id: str
    truth: str
    label: str
    ambiguous: bool
    verified: bool
    stratum: Hashable
    method1: str
    method2: str
    classic: str
    classic_fallback: str
    path: str
    typesafe_status: str
    typesafe: Optional[str]
    typesafe_confidence: Optional[float]


def summarize_cutoff(
    cutoff: float, rows: Sequence[EvalRow], outcomes: Sequence[ReplayOutcome]
) -> dict[str, Any]:
    """Sweep statistics of the hybrid at one cutoff.

    ``correct`` (per-row correctness over all rows, in ``rows`` order) is kept
    for ``choose_cutoff``.
    """
    by_id = {o.email_id: o for o in outcomes}
    gate_rows = [r for r in rows if by_id[r.email_id].path == "llm_assisted"]
    accepted = [r for r in gate_rows if by_id[r.email_id].accepted]
    rejected = [r for r in gate_rows if not by_id[r.email_id].accepted]
    answered_rejected = [r for r in rejected if r.typesafe_status == STATUS_OK]
    correct = [by_id[r.email_id].prediction == r.truth for r in rows]

    def accuracy(subset: Sequence[EvalRow]) -> dict[str, Any]:
        return proportion(
            sum(1 for r in subset if by_id[r.email_id].prediction == r.truth),
            len(subset),
        )

    return {
        "cutoff": cutoff,
        "n_gate_rows": len(gate_rows),
        "n_accepted": len(accepted),
        "acceptance_rate": len(accepted) / len(gate_rows) if gate_rows else None,
        "n_gate_errors": sum(1 for r in gate_rows if r.typesafe_status == STATUS_ERROR),
        "n_gate_missing": sum(
            1 for r in gate_rows if r.typesafe_status == STATUS_MISSING
        ),
        "accepted_llm_accuracy": accuracy(accepted),
        "rejected_fallback_accuracy": accuracy(rejected),
        "rejected_llm_counterfactual_accuracy": proportion(
            sum(1 for r in answered_rejected if r.typesafe == r.truth),
            len(answered_rejected),
        ),
        "overall_accuracy": proportion(sum(correct), len(rows)),
        "accuracy_spam": accuracy([r for r in rows if r.label == "1"]),
        "accuracy_legit": accuracy([r for r in rows if r.label == "0"]),
        "correct": correct,
    }


def run_sweep(
    rows: Sequence[EvalRow],
    emails: Mapping[str, EmailData],
    records: Mapping[str, Mapping[str, Any]],
    cutoffs: Sequence[float] = CUTOFF_SWEEP,
    hybrid_factory: HybridFactory = default_hybrid_factory,
) -> tuple[list[dict[str, Any]], dict[float, list[ReplayOutcome]]]:
    """Replay the hybrid at every cutoff and summarize each one."""
    items = [(r.email_id, emails[r.email_id]) for r in rows]
    sweep: list[dict[str, Any]] = []
    outcomes: dict[float, list[ReplayOutcome]] = {}
    for cutoff in cutoffs:
        outcomes[cutoff] = replay_hybrid(cutoff, items, records, hybrid_factory)
        sweep.append(summarize_cutoff(cutoff, rows, outcomes[cutoff]))
    return sweep, outcomes


def choose_cutoff(
    sweep_rows: Sequence[Mapping[str, Any]], incumbent: float = INCUMBENT_CUTOFF
) -> dict[str, Any]:
    """Apply the pre-registered cutoff selection rule.

    1. The primary metric is overall hybrid strict accuracy.
    2. The candidate is, among the cutoffs with the most correct rows, the one
       closest to the incumbent (the lower one on a tie).
    3. The candidate replaces the incumbent only if an exact two-sided McNemar
       test on the paired per-row correctness gives p < 0.05.

    Args:
        sweep_rows: Rows with ``cutoff`` and ``correct`` (per-row booleans in
            the same row order for every cutoff).
        incumbent: Current cutoff; must be one of the swept cutoffs.

    Returns:
        candidate, chosen, p_value, b (candidate right, incumbent wrong),
        c (the reverse), the correct counts and a rationale.
    """
    by_cutoff = {round(float(r["cutoff"]), 2): list(r["correct"]) for r in sweep_rows}
    incumbent = round(incumbent, 2)
    if incumbent not in by_cutoff:
        raise ValueError(f"incumbent cutoff {incumbent} is not in the sweep")
    best = max(sum(v) for v in by_cutoff.values())
    tied = sorted(c for c, v in by_cutoff.items() if sum(v) == best)
    candidate = min(tied, key=lambda c: (abs(c - incumbent), c))
    cand, inc = by_cutoff[candidate], by_cutoff[incumbent]
    b = sum(1 for x, y in zip(cand, inc) if x and not y)
    c = sum(1 for x, y in zip(cand, inc) if y and not x)
    p_value = mcnemar_exact(b, c)
    n = len(inc)
    if candidate == incumbent:
        chosen = incumbent
        rationale = (
            f"The incumbent {incumbent:.2f} already has the most correct rows "
            f"({best}/{n}); it is kept."
        )
    elif p_value < SIGNIFICANCE_LEVEL:
        chosen = candidate
        rationale = (
            f"Cutoff {candidate:.2f} has the most correct rows ({best}/{n}) versus "
            f"{sum(inc)}/{n} at the incumbent {incumbent:.2f}; exact McNemar "
            f"b={b}, c={c}, p={p_value:.4f} < {SIGNIFICANCE_LEVEL}, so it is adopted."
        )
    else:
        chosen = incumbent
        rationale = (
            f"Cutoff {candidate:.2f} has the most correct rows ({best}/{n}) versus "
            f"{sum(inc)}/{n} at the incumbent {incumbent:.2f}, but exact McNemar "
            f"b={b}, c={c}, p={p_value:.4f} >= {SIGNIFICANCE_LEVEL}: the data cannot "
            f"distinguish them, so the incumbent is kept."
        )
    return {
        "candidate": candidate,
        "chosen": chosen,
        "incumbent": incumbent,
        "tied_best_cutoffs": tied,
        "candidate_correct": sum(cand),
        "incumbent_correct": sum(inc),
        "n": n,
        "b_candidate_only": b,
        "c_incumbent_only": c,
        "p_value": p_value,
        "rationale": rationale,
    }


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------


def system_report(
    rows: Sequence[EvalRow],
    predictions: Mapping[str, Optional[str]],
    stratum_sizes: Optional[Mapping[Hashable, int]],
    excluded: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Metrics for one system.

    Args:
        rows: Labeled rows.
        predictions: Normalized prediction by email_id.
        stratum_sizes: Population size per stratum, or None to skip weighting.
        excluded: email_id -> reason (``error`` or ``missing``) for rows that
            are not scored at all.
    """
    excluded = excluded or {}
    scored = [r for r in rows if r.email_id not in excluded]

    def pairs(subset: Sequence[EvalRow]) -> list[tuple[str, str]]:
        return [(r.truth, str(predictions[r.email_id])) for r in subset]

    def accuracy(subset: Sequence[EvalRow]) -> dict[str, Any]:
        return proportion(sum(1 for t, p in pairs(subset) if t == p), len(subset))

    report = classification_metrics(pairs(scored))
    report.update(
        {
            "n_rows": len(rows),
            "n_scored": len(scored),
            "n_errors": sum(1 for v in excluded.values() if v == STATUS_ERROR),
            "n_missing": sum(1 for v in excluded.values() if v == STATUS_MISSING),
            "by_label": {
                "spam_1": accuracy([r for r in scored if r.label == "1"]),
                "legit_0": accuracy([r for r in scored if r.label == "0"]),
            },
            "by_ambiguous": {
                "true": accuracy([r for r in scored if r.ambiguous]),
                "false": accuracy([r for r in scored if not r.ambiguous]),
            },
            "by_verified": {
                "true": accuracy([r for r in scored if r.verified]),
                "false": accuracy([r for r in scored if not r.verified]),
            },
            "stratum_weighted_accuracy": (
                stratum_weighted_accuracy(
                    [(r.stratum, predictions[r.email_id] == r.truth) for r in scored],
                    stratum_sizes,
                )
                if stratum_sizes is not None
                else None
            ),
        }
    )
    return report


def build_rows(
    labels: Sequence[LabeledEmail],
    emails: Mapping[str, EmailData],
    records: Mapping[str, Mapping[str, Any]],
    stratum: Callable[[LabeledEmail], Hashable] = default_stratum,
) -> list[EvalRow]:
    """Run the classic methods on every labeled email and attach TypeSafe."""
    method1 = KeywordTaxonomyClassifier()
    method2 = StructuralTemplateClassifier()
    classic = EmailClassifier()
    items = [(label.email_id, emails[label.email_id]) for label in labels]
    no_llm = {o.email_id: o for o in replay_hybrid(None, items, None)}
    rows: list[EvalRow] = []
    for label in labels:
        email = emails[label.email_id]
        record = records.get(label.email_id)
        if record is None:
            status, prediction, confidence = STATUS_MISSING, None, None
        elif record["fallback"]:
            status, prediction, confidence = STATUS_ERROR, None, None
        else:
            status = STATUS_OK
            prediction = normalize_prediction(result_from_record(record).domain)
            confidence = float(record["confidence"])
        rows.append(
            EvalRow(
                email_id=label.email_id,
                truth=label.domain,
                label=label.label,
                ambiguous=label.ambiguous,
                verified=label.verified,
                stratum=stratum(label),
                method1=normalize_prediction(method1.classify(email).domain),
                method2=normalize_prediction(method2.classify(email).domain),
                classic=normalize_prediction(classic.classify(email)[0]),
                classic_fallback=no_llm[label.email_id].prediction,
                path=no_llm[label.email_id].path,
                typesafe_status=status,
                typesafe=prediction,
                typesafe_confidence=confidence,
            )
        )
    return rows


def usage_stats(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Token, latency and attempt statistics over successful records."""
    ok = [r for r in records if not r["fallback"]]
    usage = [r["usage"] or {} for r in ok]
    inputs = [
        float(u["input_tokens"]) for u in usage if u.get("input_tokens") is not None
    ]
    outputs = [
        float(u["output_tokens"]) for u in usage if u.get("output_tokens") is not None
    ]
    latencies = [float(r["latency_ms"]) for r in ok]
    attempts = [float(r["attempts"]) for r in records]
    return {
        "n_records": len(records),
        "n_success": len(ok),
        "n_errors": len(records) - len(ok),
        "input_tokens": {
            "mean": statistics.fmean(inputs) if inputs else None,
            "median": percentile(inputs, 50),
            "p95": percentile(inputs, 95),
            "min": min(inputs) if inputs else None,
            "max": max(inputs) if inputs else None,
        },
        "output_tokens_mean": statistics.fmean(outputs) if outputs else None,
        "latency_ms_p50": percentile(latencies, 50),
        "latency_ms_p95": percentile(latencies, 95),
        "attempts_per_email": statistics.fmean(attempts) if attempts else None,
        "http_429": sum(s == 429 for r in records for s in r["http_statuses"]),
    }


def runtime_stats(
    records: Sequence[Mapping[str, Any]],
    runs: Sequence[Mapping[str, Any]],
    disagreement_fraction: Optional[float],
) -> dict[str, Any]:
    """Token and runtime statistics plus the 35,000-email extrapolation."""
    by_set = {
        name: usage_stats([r for r in records if r["set"] == name]) for name in SETS
    }
    overall = usage_stats(records)
    wall = sum(float(r.get("wall_seconds") or 0) for r in runs)
    attempts = sum(int(r.get("n_calls_attempted") or 0) for r in runs)
    emails = sum(int(r.get("n_emails") or 0) for r in runs)
    attempts_per_s = attempts / wall if wall else None
    attempts_per_email = attempts / emails if emails else overall["attempts_per_email"]
    mean_in = overall["input_tokens"]["mean"]
    mean_out = overall["output_tokens_mean"]

    def scenario(n_calls: float) -> dict[str, Any]:
        n_attempts = n_calls * attempts_per_email if attempts_per_email else None
        return {
            "n_llm_emails": n_calls,
            "hours_at_measured_rate": (
                n_attempts / attempts_per_s / 3600
                if n_attempts is not None and attempts_per_s
                else None
            ),
            "hours_at_cap": n_attempts / MAX_RPS / 3600 if n_attempts else None,
            "input_tokens_total": n_calls * mean_in if mean_in is not None else None,
            "output_tokens_total": n_calls * mean_out if mean_out is not None else None,
        }

    return {
        "by_set": by_set,
        "overall": overall,
        "runs": [dict(r) for r in runs],
        "aggregate_attempts_per_second": attempts_per_s,
        "aggregate_emails_per_second": emails / wall if wall else None,
        "attempts_per_email": attempts_per_email,
        "extrapolation": {
            "n_emails": EXTRAPOLATION_EMAILS,
            "rps_cap": MAX_RPS,
            "full_typesafe": scenario(EXTRAPOLATION_EMAILS),
            "hybrid_labeled_disagreement": (
                scenario(EXTRAPOLATION_EMAILS * disagreement_fraction)
                if disagreement_fraction is not None
                else None
            ),
            "labeled_disagreement_fraction": disagreement_fraction,
            "hybrid_full_corpus_agreement": scenario(
                EXTRAPOLATION_EMAILS * (1 - FULL_CORPUS_AGREEMENT_RATE)
            ),
            "full_corpus_agreement_rate": FULL_CORPUS_AGREEMENT_RATE,
        },
    }


def round_floats(value: Any, digits: int = FLOAT_DIGITS) -> Any:
    """Round every float inside nested dicts and lists."""
    if isinstance(value, float):
        return round(value, digits)
    if isinstance(value, dict):
        return {k: round_floats(v, digits) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [round_floats(v, digits) for v in value]
    return value


def _stratum_name(stratum: Hashable) -> str:
    if isinstance(stratum, tuple):
        return "|".join(str(s) for s in stratum)
    return str(stratum)


def build_report(
    labels: Sequence[LabeledEmail],
    emails: Mapping[str, EmailData],
    cache_records: Sequence[CachedRecord],
    runs: Sequence[Mapping[str, Any]],
    stratum_sizes: Optional[Mapping[Hashable, int]],
    incumbent: float = INCUMBENT_CUTOFF,
    hybrid_factory: HybridFactory = default_hybrid_factory,
) -> dict[str, Any]:
    """Compute every number of the evaluation (ids only, no email text)."""
    label_ids = {label.email_id for label in labels}
    records = {
        eid: r for eid, r in latest_records(cache_records).items() if eid in label_ids
    }
    rows = build_rows(labels, emails, records)
    gate_rows = [r for r in rows if r.path == "llm_assisted"]
    have_typesafe = bool(records)

    systems: dict[str, Any] = {}
    for name in ("method1", "method2", "classic"):
        systems[name] = system_report(
            rows, {r.email_id: getattr(r, name) for r in rows}, stratum_sizes
        )
    excluded = {
        r.email_id: r.typesafe_status for r in rows if r.typesafe_status != STATUS_OK
    }
    if have_typesafe:
        systems["typesafe"] = system_report(
            rows, {r.email_id: r.typesafe for r in rows}, stratum_sizes, excluded
        )

    disagreement: dict[str, Any] = {
        "n_rows": len(gate_rows),
        "fraction": len(gate_rows) / len(rows) if rows else None,
        "classic_fallback_accuracy": proportion(
            sum(1 for r in gate_rows if r.classic_fallback == r.truth), len(gate_rows)
        ),
    }
    sweep: list[dict[str, Any]] = []
    choice: Optional[dict[str, Any]] = None
    hybrid_predictions: dict[str, dict[str, str]] = {}
    notes: list[str] = []
    if have_typesafe:
        answered_gate = [r for r in gate_rows if r.typesafe_status == STATUS_OK]
        disagreement["typesafe_accuracy"] = proportion(
            sum(1 for r in answered_gate if r.typesafe == r.truth), len(answered_gate)
        )
        disagreement["classic_fallback_accuracy_same_rows"] = proportion(
            sum(1 for r in answered_gate if r.classic_fallback == r.truth),
            len(answered_gate),
        )
        disagreement["n_typesafe_errors"] = sum(
            1 for r in gate_rows if r.typesafe_status == STATUS_ERROR
        )
        disagreement["n_typesafe_missing"] = sum(
            1 for r in gate_rows if r.typesafe_status == STATUS_MISSING
        )
        sweep, outcomes = run_sweep(rows, emails, records, CUTOFF_SWEEP, hybrid_factory)
        if disagreement["n_typesafe_missing"]:
            notes.append(
                "Some disagreement rows have no cached TypeSafe output; the sweep "
                "treats them as failed calls and no cutoff is chosen."
            )
        else:
            choice = choose_cutoff(sweep, incumbent)
        shown = sorted(
            {round(incumbent, 2)} | ({choice["chosen"]} if choice else set())
        )
        for cutoff in shown:
            key = f"hybrid@{cutoff:.2f}"
            preds = {o.email_id: o.prediction for o in outcomes[cutoff]}
            systems[key] = system_report(rows, preds, stratum_sizes)
            hybrid_predictions[key] = preds
    else:
        notes.append(
            "No cached TypeSafe outputs: TypeSafe and the hybrid sweep are missing. "
            "Run `collect --set labeled` first."
        )
    if stratum_sizes is None:
        notes.append(
            "classified-data output files are missing or LFS pointers: "
            "stratum-weighted accuracy is skipped."
        )

    per_row = {
        r.email_id: {
            "truth": r.truth,
            "label": r.label,
            "ambiguous": r.ambiguous,
            "stratum": _stratum_name(r.stratum),
            "path": r.path,
            "method1": r.method1,
            "method2": r.method2,
            "classic": r.classic,
            "classic_fallback": r.classic_fallback,
            "typesafe_status": r.typesafe_status,
            "typesafe": r.typesafe,
            "typesafe_confidence": r.typesafe_confidence,
            **{key: preds[r.email_id] for key, preds in hybrid_predictions.items()},
        }
        for r in sorted(rows, key=lambda r: r.email_id)
    }
    return {
        "n_labeled": len(rows),
        "typesafe_available": have_typesafe,
        "incumbent_cutoff": incumbent,
        "min_support": MIN_SUPPORT,
        "scoring_rule": "None and 'unsure' count as a 'none' prediction; failed "
        "TypeSafe calls are errors and are not scored.",
        "stratum_definition": "(source_file, CEAS_08 label)",
        "stratum_sizes": (
            {_stratum_name(k): v for k, v in sorted(stratum_sizes.items(), key=str)}
            if stratum_sizes is not None
            else None
        ),
        "systems": systems,
        "disagreement": disagreement,
        "cutoff_sweep": [
            {k: v for k, v in row.items() if k != "correct"} for row in sweep
        ],
        "cutoff_choice": choice,
        "runtime": runtime_stats(
            cache_records, runs, disagreement["fraction"] if rows else None
        ),
        "notes": notes,
        "per_row": per_row,
    }


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _fmt_prop(p: Mapping[str, Any]) -> str:
    if not p["n"]:
        return "— (n=0)"
    return (
        f"{p['rate']:.3f} [{p['ci_low']:.3f}, {p['ci_high']:.3f}] "
        f"({p['k']}/{p['n']})"
    )


def render_markdown(report: Mapping[str, Any]) -> str:
    """Render the report's tables as markdown."""
    lines: list[str] = [
        "# Classifier evaluation on the CEAS_08 ground truth (#19)",
        "",
        f"Labeled rows: {report['n_labeled']}. {report['scoring_rule']}",
        "Intervals are Wilson 95%.",
        "",
    ]
    for note in report["notes"]:
        lines.append(f"> {note}")
    if report["notes"]:
        lines.append("")
    lines += [
        "## Systems",
        "",
        "| System | Scored | Errors | Missing | Strict accuracy | Answer rate "
        "| Accuracy when answering | none P / R | Macro-F1 | Weighted acc. |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    systems = report["systems"]
    for name, s in systems.items():
        lines.append(
            f"| {name} | {s['n_scored']} | {s['n_errors']} | {s['n_missing']} "
            f"| {_fmt_prop(s['strict_accuracy'])} | {_fmt_prop(s['answer_rate'])} "
            f"| {_fmt_prop(s['answered_accuracy'])} "
            f"| {_fmt(s['none_precision'], 3)} / {_fmt(s['none_recall'], 3)} "
            f"| {_fmt(s['macro_f1'], 3)} | {_fmt(s['stratum_weighted_accuracy'], 3)} |"
        )
    if "typesafe" not in systems:
        lines.append("| typesafe | missing | | | | | | | | |")
    first = next(iter(systems.values()))
    lines += [
        "",
        f"Macro-F1 averages classes with support >= {report['min_support']}: "
        f"{', '.join(first['macro_f1_classes'])}.",
        "",
        "## By CEAS_08 label and ambiguity (strict accuracy)",
        "",
        "| System | Spam (1) | Legit (0) | Ambiguous | Not ambiguous |",
        "|---|---|---|---|---|",
    ]
    for name, s in systems.items():
        lines.append(
            f"| {name} | {_fmt_prop(s['by_label']['spam_1'])} "
            f"| {_fmt_prop(s['by_label']['legit_0'])} "
            f"| {_fmt_prop(s['by_ambiguous']['true'])} "
            f"| {_fmt_prop(s['by_ambiguous']['false'])} |"
        )
    lines += ["", "## Per class", ""]
    for name, s in systems.items():
        lines += [
            f"### {name}",
            "",
            "| Class | Precision | Recall | F1 | Support | Predicted | Note |",
            "|---|---|---|---|---|---|---|",
        ]
        for cls, m in s["per_class"].items():
            note = "insufficient support" if m["insufficient_support"] else ""
            lines.append(
                f"| {cls} | {_fmt(m['precision'], 3)} | {_fmt(m['recall'], 3)} "
                f"| {_fmt(m['f1'], 3)} | {m['support']} | {m['predicted']} | {note} |"
            )
        lines.append("")
    d = report["disagreement"]
    lines += [
        "## Disagreement rows (where the hybrid gate fires)",
        "",
        f"Rows where Method 1 and Method 2 disagree or either abstains: "
        f"{d['n_rows']} ({_fmt(d['fraction'], 3)} of labeled rows).",
        "",
        f"- Classic weighted fallback accuracy: "
        f"{_fmt_prop(d['classic_fallback_accuracy'])}",
    ]
    if "typesafe_accuracy" in d:
        lines += [
            f"- TypeSafe accuracy (answered rows): {_fmt_prop(d['typesafe_accuracy'])}",
            f"- Classic fallback on the same answered rows: "
            f"{_fmt_prop(d['classic_fallback_accuracy_same_rows'])}",
            f"- TypeSafe errors: {d['n_typesafe_errors']}; missing: "
            f"{d['n_typesafe_missing']}",
        ]
    if report["cutoff_sweep"]:
        lines += [
            "",
            "## Cutoff sweep",
            "",
            "| Cutoff | Gate rows | Accepted | Acceptance | Accepted LLM acc. "
            "| Fallback acc. (rejected) | LLM acc. if used (rejected) "
            "| Hybrid overall | Spam | Legit |",
            "|---|---|---|---|---|---|---|---|---|---|",
        ]
        for row in report["cutoff_sweep"]:
            lines.append(
                f"| {row['cutoff']:.2f} | {row['n_gate_rows']} | {row['n_accepted']} "
                f"| {_fmt(row['acceptance_rate'], 3)} "
                f"| {_fmt_prop(row['accepted_llm_accuracy'])} "
                f"| {_fmt_prop(row['rejected_fallback_accuracy'])} "
                f"| {_fmt_prop(row['rejected_llm_counterfactual_accuracy'])} "
                f"| {_fmt_prop(row['overall_accuracy'])} "
                f"| {_fmt_prop(row['accuracy_spam'])} "
                f"| {_fmt_prop(row['accuracy_legit'])} |"
            )
    if report["cutoff_choice"]:
        c = report["cutoff_choice"]
        lines += [
            "",
            "## Cutoff choice (pre-registered rule)",
            "",
            f"Candidate {c['candidate']:.2f}, chosen {c['chosen']:.2f}. "
            f"{c['rationale']}",
        ]
    rt = report["runtime"]
    lines += [
        "",
        "## Tokens and runtime",
        "",
        "| Set | Records | Errors | Input tokens mean / median / p95 / min / max "
        "| Output tokens mean | Latency p50 / p95 ms | Attempts per email | 429s |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, u in [*rt["by_set"].items(), ("overall", rt["overall"])]:
        t = u["input_tokens"]
        lines.append(
            f"| {name} | {u['n_records']} | {u['n_errors']} "
            f"| {_fmt(t['mean'], 1)} / {_fmt(t['median'], 1)} / {_fmt(t['p95'], 1)} "
            f"/ {_fmt(t['min'], 0)} / {_fmt(t['max'], 0)} "
            f"| {_fmt(u['output_tokens_mean'], 1)} "
            f"| {_fmt(u['latency_ms_p50'], 0)} / {_fmt(u['latency_ms_p95'], 0)} "
            f"| {_fmt(u['attempts_per_email'], 3)} | {u['http_429']} |"
        )
    if rt["runs"]:
        lines += [
            "",
            "| Run set | Status | Emails | Attempts | 429s | Errors | rps | Workers "
            "| Emails/s | Attempts/s |",
            "|---|---|---|---|---|---|---|---|---|---|",
        ]
        for r in rt["runs"]:
            lines.append(
                f"| {r.get('set')} | {r.get('status')} | {r.get('n_emails')} "
                f"| {r.get('n_calls_attempted')} | {r.get('n_429')} "
                f"| {r.get('n_errors')} | {r.get('rps')} | {r.get('workers')} "
                f"| {_fmt(r.get('emails_per_second'), 2)} "
                f"| {_fmt(r.get('attempts_per_second'), 2)} |"
            )
    ex = rt["extrapolation"]
    lines += [
        "",
        f"Extrapolation to {ex['n_emails']:,} emails (attempts per email "
        f"{_fmt(rt['attempts_per_email'], 3)}, measured attempts/s "
        f"{_fmt(rt['aggregate_attempts_per_second'], 2)}, cap {ex['rps_cap']:g}/s):",
        "",
        "| Scenario | LLM emails | Hours at measured rate | Hours at cap "
        "| Input tokens | Output tokens |",
        "|---|---|---|---|---|---|",
    ]
    scenarios = [
        ("TypeSafe on every email", ex["full_typesafe"]),
        (
            "Hybrid, labeled-set disagreement "
            f"({_fmt(ex['labeled_disagreement_fraction'], 3)})",
            ex["hybrid_labeled_disagreement"],
        ),
        (
            "Hybrid, full-corpus agreement " f"{ex['full_corpus_agreement_rate']:.2%}",
            ex["hybrid_full_corpus_agreement"],
        ),
    ]
    for title, s in scenarios:
        if s is None:
            continue
        lines.append(
            f"| {title} | {s['n_llm_emails']:,.0f} "
            f"| {_fmt(s['hours_at_measured_rate'], 2)} | {_fmt(s['hours_at_cap'], 2)} "
            f"| {_fmt(s['input_tokens_total'], 0)} "
            f"| {_fmt(s['output_tokens_total'], 0)} |"
        )
    lines += [
        "",
        "The labeled set is a stratified sample, so its disagreement fraction is not "
        "a population estimate; the full-corpus agreement rate comes from the "
        "reporter's run over all of CEAS_08.",
    ]
    return "\n".join(lines) + "\n"


def run_report(args: argparse.Namespace) -> int:
    """Run the offline ``report`` subcommand."""
    if round(args.incumbent, 2) not in CUTOFF_SWEEP:
        raise InputError("--incumbent must be one of the swept cutoffs (0.00..0.95)")
    require_real_file(args.raw)
    labels = load_labels(args.labels)
    emails = resolve_labeled_emails(labels, args.raw)
    stratum_sizes = count_strata(args.classified_dir, _sampler.OUTPUT_FILES)
    report = build_report(
        labels,
        emails,
        load_cache(args.cache),
        load_runs(args.runs),
        stratum_sizes,
        incumbent=args.incumbent,
    )
    report = round_floats(report)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    markdown = render_markdown(report)
    if args.md_out:
        args.md_out.parent.mkdir(parents=True, exist_ok=True)
        args.md_out.write_text(markdown, encoding="utf-8")
    sys.stdout.write(markdown)
    return EXIT_OK


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--labels", type=Path, default=DEFAULT_LABELS_PATH)
        p.add_argument("--raw", type=Path, default=DEFAULT_RAW_PATH)
        p.add_argument("--cache", type=Path, default=DEFAULT_CACHE_PATH)
        p.add_argument("--runs", type=Path, default=DEFAULT_RUNS_PATH)

    collect = sub.add_parser(
        "collect", help=f"classify emails with TypeSafe (reads {API_KEY_ENV})"
    )
    add_common(collect)
    collect.add_argument("--set", choices=SETS, default=SET_LABELED)
    collect.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE)
    collect.add_argument("--seed", type=int, default=DEFAULT_SAMPLE_SEED)
    collect.add_argument("--model", default=DEFAULT_MODEL)
    collect.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    collect.add_argument(
        "--rps",
        type=float,
        default=DEFAULT_RPS,
        help=f"HTTP attempts per second, retries included (max {MAX_RPS:g})",
    )
    collect.add_argument(
        "--max-calls",
        type=int,
        default=DEFAULT_MAX_CALLS,
        help="maximum HTTP attempts in this run, retries included",
    )
    collect.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT_SECONDS,
        help="per-request timeout in seconds",
    )

    report = sub.add_parser("report", help="compute metrics offline")
    add_common(report)
    report.add_argument("--classified-dir", type=Path, default=DEFAULT_CLASSIFIED_DIR)
    report.add_argument("--json-out", type=Path, default=DEFAULT_JSON_OUT)
    report.add_argument("--md-out", type=Path)
    report.add_argument("--incumbent", type=float, default=INCUMBENT_CUTOFF)
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    """Run the evaluation from the command line."""
    args = build_parser().parse_args(argv)
    try:
        if args.command == "collect":
            return run_collect(args)
        return run_report(args)
    except InputError as error:
        print(f"error: {error}", file=sys.stderr)
        return EXIT_USAGE


if __name__ == "__main__":
    sys.exit(main())
