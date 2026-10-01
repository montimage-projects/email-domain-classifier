"""
Streaming processor for large CSV email datasets.

Processes emails in chunks to minimize memory usage while
maintaining progress tracking and logging.
"""

import csv
import logging
import os
import time
from collections import defaultdict, deque
from collections.abc import Callable, Generator
from concurrent.futures import Future, ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from threading import local
from typing import IO, Any, Dict, List, Optional

from .classifier import (
    EmailClassifier,
    EmailData,
    HybridClassifier,
    HybridWorkflowLogger,
    HybridWorkflowStats,
)
from .domains import get_domain_names
from .parsing import parse_url_flag
from .validator import (
    EmailValidator,
    InvalidEmailWriter,
    SkippedEmailWriter,
    SkippedStats,
    ValidationStats,
)


@dataclass
class HybridWorkflowProcessingStats:
    """Statistics for hybrid classification workflow during processing."""

    llm_call_count: int = 0
    llm_total_time_ms: float = 0.0
    classic_agreement_count: int = 0
    total_hybrid_processed: int = 0
    # LLM answers the confidence gate did not accept (issue #18)
    llm_rejected_count: int = 0

    @property
    def llm_avg_time_ms(self) -> float:
        """Calculate average LLM response time."""
        if self.llm_call_count == 0:
            return 0.0
        return self.llm_total_time_ms / self.llm_call_count

    @property
    def agreement_rate(self) -> float:
        """Calculate classifier agreement rate (percentage)."""
        if self.total_hybrid_processed == 0:
            return 0.0
        return (self.classic_agreement_count / self.total_hybrid_processed) * 100

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "llm_call_count": self.llm_call_count,
            "llm_total_time_ms": round(self.llm_total_time_ms, 2),
            "llm_avg_time_ms": round(self.llm_avg_time_ms, 2),
            "classic_agreement_count": self.classic_agreement_count,
            "total_hybrid_processed": self.total_hybrid_processed,
            "agreement_rate": round(self.agreement_rate, 2),
            "llm_rejected_count": self.llm_rejected_count,
        }


@dataclass
class ProcessingStats:
    """Statistics collected during processing.

    Enhanced statistics include:
    - label_distributions: Original label values by classified domain
    - url_distributions: URL presence (True/False) by domain
    - cross_tabulation: Label x URL relationship matrix by domain
    - hybrid_workflow: Statistics from hybrid classification workflow
    """

    total_processed: int = 0
    total_classified: int = 0
    total_unsure: int = 0
    # Classified emails where method1 and method2 named the same domain
    method_agreement_count: int = 0
    domain_counts: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    errors: int = 0
    start_time: datetime | None = None
    end_time: datetime | None = None
    label_distributions: dict[str, dict[str, int]] = field(
        default_factory=lambda: defaultdict(lambda: defaultdict(int))
    )
    url_distributions: dict[str, dict[bool, int]] = field(
        default_factory=lambda: defaultdict(lambda: defaultdict(int))
    )
    cross_tabulation: dict[str, dict[str, dict[bool, int]]] = field(
        default_factory=lambda: defaultdict(
            lambda: defaultdict(lambda: defaultdict(int))
        )
    )
    # Validation statistics
    validation_stats: ValidationStats = field(default_factory=ValidationStats)
    # Skipped email statistics
    skipped_stats: SkippedStats = field(default_factory=SkippedStats)
    # Hybrid workflow statistics
    hybrid_workflow: HybridWorkflowProcessingStats = field(
        default_factory=HybridWorkflowProcessingStats
    )

    def to_dict(self) -> dict:
        """Convert stats to dictionary."""
        return {
            "total_processed": self.total_processed,
            "total_classified": self.total_classified,
            "total_unsure": self.total_unsure,
            "method_agreement_count": self.method_agreement_count,
            "domain_counts": dict(self.domain_counts),
            "errors": self.errors,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_seconds": (
                (self.end_time - self.start_time).total_seconds()
                if self.start_time and self.end_time
                else None
            ),
            "label_distributions": {
                k: dict(v) for k, v in self.label_distributions.items()
            },
            "url_distributions": {
                k: dict(v) for k, v in self.url_distributions.items()
            },
            "cross_tabulation": {
                k: {l: dict(v) for l, v in val.items()}
                for k, val in self.cross_tabulation.items()
            },
            "validation": self.validation_stats.to_dict(),
            "skipped": self.skipped_stats.to_dict(),
            "hybrid_workflow": self.hybrid_workflow.to_dict(),
        }


class OutputManager:
    """Manages output CSV files for each domain."""

    # Standard columns in required order
    STANDARD_COLUMNS = [
        "sender",
        "receiver",
        "date",
        "subject",
        "body",
        "urls",
        "label",
    ]

    # Classification columns (always after standard)
    CLASSIFICATION_COLUMNS = ["classified_domain", "method1_domain", "method2_domain"]

    # Optional detail columns
    DETAIL_COLUMNS = ["method1_confidence", "method2_confidence", "agreement"]

    def __init__(
        self, output_dir: Path, fieldnames: list[str], include_details: bool = False
    ) -> None:
        self.output_dir = output_dir
        self.include_details = include_details
        self.files: dict[str, IO[str]] = {}
        self.writers: dict[str, csv.DictWriter[str]] = {}

        # Build ordered fieldnames
        self.fieldnames = self._build_fieldnames(fieldnames, include_details)

        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _build_fieldnames(
        self, input_fieldnames: list[str], include_details: bool
    ) -> list[str]:
        """Build ordered fieldnames list with standard columns first."""
        fieldnames = self.STANDARD_COLUMNS.copy()
        fieldnames.extend(self.CLASSIFICATION_COLUMNS)
        if include_details:
            fieldnames.extend(self.DETAIL_COLUMNS)
        # Add any extra columns from input that are not already included
        for col in input_fieldnames:
            if col not in fieldnames and col not in ["timestamp", "has_url"]:
                fieldnames.append(col)
        return fieldnames

    def _get_writer(self, domain: str) -> "csv.DictWriter[str]":
        """Get or create CSV writer for domain."""
        if domain not in self.writers:
            filename = f"email_{domain}.csv"
            filepath = self.output_dir / filename

            # Open file in append mode with newline handling
            file = open(filepath, "w", newline="", encoding="utf-8")
            self.files[domain] = file

            writer = csv.DictWriter(file, fieldnames=self.fieldnames)
            writer.writeheader()
            self.writers[domain] = writer

        return self.writers[domain]

    def write_email(self, domain: str, email_data: dict[str, Any]) -> None:
        """Write email to appropriate domain file."""
        writer = self._get_writer(domain)
        writer.writerow(email_data)
        self.files[domain].flush()

    def close_all(self) -> None:
        """Close all open files."""
        for file in self.files.values():
            file.close()
        self.files.clear()
        self.writers.clear()

    def get_output_files(self) -> dict[str, Path]:
        """Return mapping of domains to their output file paths."""
        return {
            domain: self.output_dir / f"email_{domain}.csv"
            for domain in self.writers.keys()
        }


class _BufferedWorkflowLogger(HybridWorkflowLogger):
    """Worker-owned sink; never touches a file or the UI."""

    def __init__(self) -> None:
        super().__init__()
        self.entries: list[dict[str, Any]] = []

    def emit_entry(self, entry: dict[str, Any]) -> None:
        self.entries.append(entry)


class _WorkerLogFilter(logging.Filter):
    """Keep application diagnostics off worker console/file handlers."""

    def __init__(self, state: local) -> None:
        super().__init__()
        self.state = state

    def filter(self, record: logging.LogRecord) -> bool:
        records = getattr(self.state, "log_records", None)
        if records is None:
            return True
        records.append(record)
        return False


@dataclass
class _TaskOutcome:
    result: tuple[str, dict[str, Any]] | None = None
    error: BaseException | None = None
    events: list[dict[str, Any]] = field(default_factory=list)
    stats: HybridWorkflowStats | None = None
    log_records: list[logging.LogRecord] = field(default_factory=list)


class StreamingProcessor:
    """
    Stream processes large CSV files for email classification.

    Features:
    - Memory-efficient chunk-based processing
    - Real-time progress tracking
    - Comprehensive logging
    - Statistics collection
    - Email validation with invalid email logging
    - Standardized output column structure
    """

    # Default expected CSV columns
    EXPECTED_COLUMNS = [
        "sender",
        "receiver",
        "date",
        "subject",
        "body",
        "label",
        "urls",
    ]

    # Column mappings from input to standard output
    COLUMN_MAPPINGS = {
        "timestamp": "date",
        "has_url": "urls",
    }

    def __init__(
        self,
        classifier: EmailClassifier | HybridClassifier | None = None,
        chunk_size: int = 1000,
        logger: logging.Logger | None = None,
        allow_large_fields: bool = True,
        strict_validation: bool = False,
        max_body_length: int | None = None,
        use_hybrid: bool = False,
        classifier_factory: (
            Callable[[], EmailClassifier | HybridClassifier] | None
        ) = None,
        workers: int = 1,
    ) -> None:
        if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
            raise ValueError("workers must be an integer >= 1")
        self.workers = workers
        self.classifier: EmailClassifier | HybridClassifier = (
            classifier or EmailClassifier()
        )
        self.chunk_size = chunk_size
        self.logger = logger or logging.getLogger(__name__)
        self.allow_large_fields = allow_large_fields
        self.strict_validation = strict_validation
        self.max_body_length = max_body_length
        self.use_hybrid = use_hybrid
        self.validator = EmailValidator()
        self.stats = ProcessingStats()
        self._classifier_factory = classifier_factory

    def _worker_factory(self) -> Callable[[], EmailClassifier | HybridClassifier]:
        """Snapshot constructor settings, not live clients or callbacks.

        Subclasses and post-construction method injections need a factory that
        returns a fresh independent classifier. In-place customizations also
        require a factory; automatic cloning supports constructor settings only.
        """
        if self._classifier_factory is not None:
            return self._classifier_factory
        template = self.classifier
        if type(template) not in (EmailClassifier, HybridClassifier):
            raise ValueError(
                "custom classifiers require classifier_factory with workers > 1"
            )
        hybrid = isinstance(template, HybridClassifier)
        methods = (
            template.method1,
            template.method2,
            template.llm_classifier if hybrid else template.method3,
        )
        if (
            any(
                current is not original
                for current, original in zip(methods, template._worker_methods)
            )
            or any(
                "classify" in method.__dict__
                for method in methods
                if method is not None
            )
            or any(
                method.__dict__.get(name) is not None
                for method in methods
                if method is not None
                for name in ("_client", "_llm", "_structured_llm")
            )
            or "classify_dict" in template.__dict__
            or "classify" in template.__dict__
        ):
            raise ValueError(
                "injected methods or initialized clients require "
                "classifier_factory with workers > 1"
            )

        domains = deepcopy(template.domains)
        config = deepcopy(template._llm_config)
        # Respect disabled/unavailable LLMs; never reload environment settings.
        if methods[2] is None:
            config = None
        cutoff = template.llm_confidence_cutoff if hybrid else None
        threshold = template.GLOBAL_THRESHOLD
        weights = (
            None
            if hybrid
            else (
                template.weight_method_1,
                template.weight_method_2,
                template.weight_method_3,
            )
        )

        def create() -> EmailClassifier | HybridClassifier:
            if hybrid:
                worker = HybridClassifier(
                    llm_config=deepcopy(config),
                    domains=deepcopy(domains),
                    llm_confidence_cutoff=cutoff,
                )
            else:
                worker = EmailClassifier(
                    domains=deepcopy(domains), llm_config=deepcopy(config)
                )
                (
                    worker.weight_method_1,
                    worker.weight_method_2,
                    worker.weight_method_3,
                ) = weights
            worker.GLOBAL_THRESHOLD = threshold
            return worker

        return create

    def _serial_rows(
        self, input_path: Path
    ) -> Generator[tuple[int, dict, None], None, None]:
        """Adapt the original serial stream without prefetching or cloning."""
        source = self._stream_emails(input_path)
        try:
            for idx, row in enumerate(source):
                yield idx, row, None
        finally:
            source.close()

    def _parallel_rows(
        self,
        input_path: Path,
        total_rows: int,
        factory: Callable[[], EmailClassifier | HybridClassifier],
    ) -> Generator[tuple[int, dict, Future[_TaskOutcome] | None], None, None]:
        """FIFO window of at most workers source rows, including finished tasks.

        The head stays in the window while the coordinator commits it. On early
        exit queued tasks are cancelled; running calls finish within their
        configured provider timeouts before shutdown returns.
        """
        source = self._stream_emails(input_path)
        state = local()
        pending = deque()
        executor = ThreadPoolExecutor(
            max_workers=self.workers, thread_name_prefix="email"
        )
        # Filters must sit on emitting loggers, not only their package parent:
        # parent logger filters do not see propagated child records.
        diagnostic_loggers = [
            logging.getLogger(name)
            for name in (
                "email_classifier",
                "email_classifier.classifier",
                "email_classifier.processor",
                "email_classifier.llm.agent",
                "email_classifier.llm.typesafe_classifier",
            )
        ]
        log_filter = _WorkerLogFilter(state)
        for diagnostic_logger in diagnostic_loggers:
            diagnostic_logger.addFilter(log_filter)

        def classify(idx: int, row: dict) -> _TaskOutcome:
            outcome = _TaskOutcome()
            worker = None
            sink = _BufferedWorkflowLogger()
            state.log_records = outcome.log_records
            try:
                if not hasattr(state, "classifier"):
                    state.classifier = factory()
                worker = state.classifier
                if isinstance(worker, HybridClassifier):
                    worker.status_callback = None
                    worker.workflow_logger = sink
                    worker.reset_stats()
                normalized = self._normalize_row(row)
                if self.use_hybrid and isinstance(worker, HybridClassifier):
                    outcome.result = worker.classify_dict(
                        normalized, email_idx=idx, total_emails=total_rows
                    )
                else:
                    outcome.result = worker.classify_dict(normalized)
            except BaseException as exc:
                # Transport interruptions as well as ordinary row failures.
                outcome.error = exc
            finally:
                state.log_records = None
                outcome.events = sink.entries
                if isinstance(worker, HybridClassifier):
                    outcome.stats = deepcopy(worker.get_stats())
            return outcome

        exhausted = False
        next_idx = 0
        try:
            while pending or not exhausted:
                while len(pending) < self.workers and not exhausted:
                    try:
                        row = next(source)
                    except StopIteration:
                        exhausted = True
                        break
                    except Exception as exc:
                        # Ordinary source failures must not leapfrog buffered rows.
                        # Interruptions instead reach cleanup immediately.
                        pending.append((next_idx, None, exc))
                        exhausted = True
                        break
                    future = None
                    try:
                        eligible = self.validator.validate(row).is_valid and (
                            self.max_body_length is None
                            or len(str(row.get("body", ""))) <= self.max_body_length
                        )
                    except Exception:
                        # The authoritative validation below handles this row.
                        eligible = False
                    if eligible:
                        future = executor.submit(classify, next_idx, row)
                    pending.append((next_idx, row, future))
                    next_idx += 1
                if pending:
                    idx, row, task = pending[0]
                    if row is None:
                        raise task
                    yield idx, row, task
                    pending.popleft()
        finally:
            for _, _, task in pending:
                if isinstance(task, Future):
                    task.cancel()
            try:
                executor.shutdown(wait=True, cancel_futures=True)
            finally:
                try:
                    source.close()
                finally:
                    for diagnostic_logger in diagnostic_loggers:
                        diagnostic_logger.removeFilter(log_filter)

    def _consume_outcome(
        self, task: Future[_TaskOutcome]
    ) -> tuple[str, dict[str, Any]]:
        """Replay task effects exactly once on the coordinator, then raise/return."""
        outcome = task.result()
        if isinstance(self.classifier, HybridClassifier):
            if outcome.stats is not None:
                for name in HybridWorkflowStats.__dataclass_fields__:
                    setattr(
                        self.classifier.stats,
                        name,
                        getattr(self.classifier.stats, name)
                        + getattr(outcome.stats, name),
                    )
        for record in outcome.log_records:
            logging.getLogger(record.name).handle(record)
        if isinstance(self.classifier, HybridClassifier):
            if self.classifier.workflow_logger is not None:
                for entry in outcome.events:
                    self.classifier.workflow_logger.emit_entry(entry)
        if outcome.error is not None:
            raise outcome.error
        assert outcome.result is not None
        return outcome.result

    def _normalize_row(self, row: dict) -> dict:
        """
        Apply column mappings and ensure standard structure.

        Maps alternative column names to standard names and handles
        special conversions like has_url -> urls.
        """
        normalized = {}

        for standard_col in OutputManager.STANDARD_COLUMNS:
            # Check if there's a mapped column in the input
            input_col = standard_col
            for mapped_from, mapped_to in self.COLUMN_MAPPINGS.items():
                if mapped_to == standard_col and mapped_from in row:
                    input_col = mapped_from
                    break

            value = row.get(input_col, "")

            # Special handling for has_url -> urls conversion
            if input_col == "has_url":
                value = "true" if parse_url_flag(value) else ""

            normalized[standard_col] = value

        # Copy any extra columns that are not in standard columns or mappings
        for col, val in row.items():
            if col not in normalized and col not in self.COLUMN_MAPPINGS:
                normalized[col] = val

        return normalized

    def count_rows(self, input_path: Path) -> int:
        """Count total rows in CSV file for progress tracking."""
        count = 0
        try:
            # Set field limit if needed
            current_limit = csv.field_size_limit()
            if self.allow_large_fields:
                csv.field_size_limit(2**31 - 1)

            with open(input_path, encoding="utf-8", errors="replace") as f:
                reader = csv.reader(f)
                try:
                    # Skip header
                    next(reader, None)
                    for _ in reader:
                        count += 1
                except csv.Error:
                    # Fallback to line counting if CSV parsing fails during counting
                    # This is just an estimate anyway
                    f.seek(0)
                    next(f, None)
                    for _ in f:
                        count += 1
        finally:
            if self.allow_large_fields:
                csv.field_size_limit(current_limit)

        return count

    def _stream_emails(self, input_path: Path) -> Generator[dict, None, None]:
        """Stream emails from CSV file one at a time."""
        current_limit = None
        try:
            # Configure CSV reader to handle large fields
            if self.allow_large_fields:
                # Temporarily increase field size limit for this file
                current_limit = csv.field_size_limit()
                csv.field_size_limit(2**31 - 1)  # Set to a very large limit (about 2GB)
                self.logger.info(f"Field size limit set to: {current_limit}")
            else:
                current_limit = None
                self.logger.info(f"Field size limit used: default (131,072)")

            with open(input_path, encoding="utf-8", errors="replace") as f:
                reader = csv.DictReader(f)
                if reader.fieldnames:
                    missing = set(self.EXPECTED_COLUMNS) - set(reader.fieldnames)
                    if missing:
                        # Allow label to be missing in input if it's going to be added in output
                        # but warning if other core columns are missing
                        critical_missing = missing - {"label"}
                        if critical_missing:
                            self.logger.warning(
                                f"Missing expected columns: {critical_missing}"
                            )

                yield from reader
        except csv.Error as e:
            error_msg = str(e).lower()
            if (
                "field larger than field limit" in error_msg
                or "fieldsize limit" in error_msg
            ):
                if self.allow_large_fields:
                    self.logger.warning(
                        f"Large CSV field detected. Processing continues with unlimited field size."
                    )
                else:
                    self.logger.error(
                        f"✖ Error: Processing failed: field larger than field limit (131,072 characters)\n"
                        f"Solutions:\n"
                        f"1. Use --allow-large-fields flag to process files with large fields\n"
                        f"2. Preprocess your CSV to truncate or split large fields\n"
                        f"3. Use smaller email samples for testing\n"
                        f"Error details: {e}"
                    )
            else:
                self.logger.error(f"CSV parsing error: {e}")
            raise
        except OSError as e:
            self.logger.error(f"File I/O error: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error reading CSV: {e}")
            raise
        finally:
            if current_limit is not None:
                csv.field_size_limit(current_limit)

    def process(
        self,
        input_path: Path,
        output_dir: Path,
        progress_callback: Callable[[int, int, str], None] | None = None,
        include_details: bool = False,
    ) -> ProcessingStats:
        """
        Process input CSV and write classified emails to domain-specific files.

        Args:
            input_path: Path to input CSV file
            output_dir: Directory for output files
            progress_callback: Optional callback(current, total, status) for progress updates
            include_details: Include classification details in output

        Returns:
            ProcessingStats with classification results
        """
        self.stats = ProcessingStats()
        self.stats.start_time = datetime.now()
        wall_start = time.perf_counter()
        factory = self._worker_factory() if self.workers > 1 else None

        input_path = Path(input_path)
        output_dir = Path(output_dir)

        self.logger.info(f"Starting processing of {input_path}")

        # Count total rows for progress
        if progress_callback:
            progress_callback(0, 0, "Counting rows...")

        total_rows = self.count_rows(input_path)
        self.logger.info(f"Total emails to process: {total_rows}")

        # Determine input fieldnames for invalid email writer
        current_limit = csv.field_size_limit()
        input_fieldnames = []

        try:
            if self.allow_large_fields:
                csv.field_size_limit(2**31 - 1)

            with open(input_path, encoding="utf-8", errors="replace") as f:
                reader = csv.DictReader(f)
                if reader.fieldnames:
                    input_fieldnames = list(reader.fieldnames)
        finally:
            csv.field_size_limit(current_limit)

        # Initialize output manager with standardized columns
        output_manager = OutputManager(output_dir, input_fieldnames, include_details)

        # Initialize invalid email writer
        invalid_writer = InvalidEmailWriter(output_dir, input_fieldnames)

        # Initialize skipped email writer (only if max_body_length is set)
        skipped_writer = None
        if self.max_body_length is not None:
            skipped_writer = SkippedEmailWriter(output_dir, input_fieldnames)

        rows = (
            self._parallel_rows(input_path, total_rows, factory)
            if factory is not None
            else self._serial_rows(input_path)
        )
        try:
            for idx, email_dict, task in rows:
                try:
                    # Validate email before processing
                    validation_result = self.validator.validate(email_dict)

                    if not validation_result.is_valid:
                        # Write to invalid emails file
                        invalid_writer.write(email_dict, validation_result.errors)

                        # Update validation stats
                        for error in validation_result.errors:
                            if error == "invalid_sender_format":
                                self.stats.validation_stats.invalid_sender_format += 1
                            elif error == "invalid_receiver_format":
                                self.stats.validation_stats.invalid_receiver_format += 1
                            elif error == "empty_sender":
                                self.stats.validation_stats.invalid_empty_sender += 1
                            elif error == "empty_receiver":
                                self.stats.validation_stats.invalid_empty_receiver += 1
                            elif error == "empty_subject":
                                self.stats.validation_stats.invalid_empty_subject += 1
                            elif error == "empty_body":
                                self.stats.validation_stats.invalid_empty_body += 1

                        self.stats.validation_stats.total_invalid += 1

                        # In strict mode, raise an error
                        if self.strict_validation:
                            raise ValueError(
                                f"Validation failed for email {idx + 1}: {validation_result.errors}"
                            )

                        # Skip to next email
                        continue

                    # Check body length filter (if max_body_length is set)
                    if self.max_body_length is not None:
                        body = str(email_dict.get("body", ""))
                        body_length = len(body)
                        if body_length > self.max_body_length:
                            # Write to skipped emails file
                            if skipped_writer is not None:
                                skipped_writer.write(email_dict, "body_too_long")

                            # Update skipped stats
                            self.stats.skipped_stats.total_skipped += 1
                            self.stats.skipped_stats.skipped_body_too_long += 1

                            # Log the skip
                            self.logger.debug(
                                f"Skipped email {idx + 1}: body length {body_length} "
                                f"exceeds limit {self.max_body_length}"
                            )

                            # Skip to next email
                            continue

                    # Normalize row to standard column structure
                    normalized_row = self._normalize_row(email_dict)

                    # Classify email (use different call for hybrid classifier)
                    if task is not None:
                        domain, details = self._consume_outcome(task)
                    elif self.use_hybrid and isinstance(
                        self.classifier, HybridClassifier
                    ):
                        domain, details = self.classifier.classify_dict(
                            normalized_row, email_idx=idx, total_emails=total_rows
                        )
                    else:
                        domain, details = self.classifier.classify_dict(normalized_row)

                    # Prepare output row with standard columns
                    # Preserve original label from input (do not overwrite with domain)
                    output_row = normalized_row.copy()
                    output_row["classified_domain"] = domain
                    output_row["method1_domain"] = (
                        details["method1"]["domain"] or "none"
                    )
                    output_row["method2_domain"] = (
                        details["method2"]["domain"] or "none"
                    )

                    if include_details:
                        output_row["method1_confidence"] = (
                            f"{details['method1']['confidence']:.4f}"
                        )
                        output_row["method2_confidence"] = (
                            f"{details['method2']['confidence']:.4f}"
                        )
                        output_row["agreement"] = details.get("agreement", False)

                    # Write to appropriate file
                    output_manager.write_email(domain, output_row)

                    # Update stats
                    self.stats.total_processed += 1
                    self.stats.domain_counts[domain] += 1

                    if domain != "unsure":
                        self.stats.total_classified += 1
                        method1_domain = details["method1"]["domain"]
                        if (
                            method1_domain is not None
                            and method1_domain == details["method2"]["domain"]
                        ):
                            self.stats.method_agreement_count += 1
                    else:
                        self.stats.total_unsure += 1

                    # Enhanced statistics collection
                    original_label = email_dict.get("label", "unknown")
                    self.stats.label_distributions[domain][original_label] += 1

                    # Parse has_url value (handle various formats)
                    has_url = parse_url_flag(
                        email_dict.get("has_url", email_dict.get("urls", "false"))
                    )

                    self.stats.url_distributions[domain][has_url] += 1
                    self.stats.cross_tabulation[domain][original_label][has_url] += 1

                    # Parallel UI updates describe committed wall-clock throughput,
                    # not the sum of overlapping worker classification durations.
                    if self.workers > 1 and isinstance(
                        self.classifier, HybridClassifier
                    ):
                        if self.classifier.status_callback is not None:
                            elapsed = time.perf_counter() - wall_start
                            rate = (
                                self.stats.total_processed / elapsed if elapsed else 0.0
                            )
                            try:
                                self.classifier.status_callback(
                                    f"Committed email {idx + 1}/{total_rows} | {rate:.1f} emails/s"
                                )
                            except Exception as exc:
                                # Notification failure cannot undo a committed row.
                                self.logger.warning(
                                    "Status callback failed after committing email %s: %s",
                                    idx + 1,
                                    exc,
                                )

                    # Log and progress callback based on chunk_size
                    if (idx + 1) % self.chunk_size == 0:
                        self.logger.info(
                            f"Processed {idx + 1}/{total_rows} emails "
                            f"({(idx + 1) / total_rows * 100:.1f}%)"
                        )

                        # Progress callback respects chunk_size
                        if progress_callback:
                            progress_callback(
                                idx + 1,
                                total_rows,
                                f"Processing email {idx + 1}/{total_rows}",
                            )

                except ValueError as e:
                    # Re-raise ValueError (used for strict validation mode)
                    if self.strict_validation:
                        raise
                    self.stats.errors += 1
                    self.logger.error(f"Error processing email {idx + 1}: {e}")

                except Exception as e:
                    self.stats.errors += 1
                    self.logger.error(f"Error processing email {idx + 1}: {e}")

                    # Still write to unsure if there's an error
                    # Preserve original label from input (do not overwrite)
                    try:
                        normalized_row = self._normalize_row(email_dict)
                        output_row = normalized_row.copy()
                        output_row["classified_domain"] = "unsure"
                        output_row["method1_domain"] = "error"
                        output_row["method2_domain"] = "error"
                        if include_details:
                            output_row["method1_confidence"] = "0"
                            output_row["method2_confidence"] = "0"
                            output_row["agreement"] = False
                        output_manager.write_email("unsure", output_row)
                        self.stats.total_unsure += 1
                    except:
                        pass

            # Final progress update
            if progress_callback:
                progress_callback(total_rows, total_rows, "Processing complete")

        finally:
            try:
                rows.close()
            finally:
                output_manager.close_all()
                invalid_writer.close()
                if skipped_writer is not None:
                    skipped_writer.close()

        self.stats.end_time = datetime.now()

        # Collect hybrid workflow stats if using hybrid classifier
        if self.use_hybrid and isinstance(self.classifier, HybridClassifier):
            hybrid_stats = self.classifier.get_stats()
            self.stats.hybrid_workflow.llm_call_count = hybrid_stats.llm_call_count
            self.stats.hybrid_workflow.llm_total_time_ms = (
                hybrid_stats.llm_total_time_ms
            )
            self.stats.hybrid_workflow.classic_agreement_count = (
                hybrid_stats.classic_agreement_count
            )
            self.stats.hybrid_workflow.total_hybrid_processed = (
                hybrid_stats.total_processed
            )
            self.stats.hybrid_workflow.llm_rejected_count = (
                hybrid_stats.llm_rejected_count
            )

        # Log summary
        duration = (self.stats.end_time - self.stats.start_time).total_seconds()
        self.logger.info(f"Processing complete in {duration:.2f} seconds")
        self.logger.info(f"Total processed: {self.stats.total_processed}")
        self.logger.info(f"Total classified: {self.stats.total_classified}")
        self.logger.info(f"Total unsure: {self.stats.total_unsure}")
        self.logger.info(
            f"Total invalid (skipped): {self.stats.validation_stats.total_invalid}"
        )
        if self.stats.skipped_stats.total_skipped > 0:
            self.logger.info(
                f"Total skipped (body too long): {self.stats.skipped_stats.skipped_body_too_long}"
            )
        self.logger.info(f"Errors: {self.stats.errors}")

        # Log hybrid workflow stats
        if self.use_hybrid:
            self.logger.info(
                f"Hybrid workflow - LLM calls: {self.stats.hybrid_workflow.llm_call_count}"
            )
            self.logger.info(
                f"Hybrid workflow - Agreement rate: "
                f"{self.stats.hybrid_workflow.agreement_rate:.1f}%"
            )
            if self.stats.hybrid_workflow.llm_call_count > 0:
                self.logger.info(
                    f"Hybrid workflow - Avg LLM time: "
                    f"{self.stats.hybrid_workflow.llm_avg_time_ms:.0f}ms"
                )
                self.logger.info(
                    f"Hybrid workflow - LLM answers rejected by confidence gate: "
                    f"{self.stats.hybrid_workflow.llm_rejected_count}"
                )

        return self.stats

    def get_output_summary(self, output_dir: Path) -> dict[str, int]:
        """Get summary of output files and their row counts."""
        output_dir = Path(output_dir)
        summary = {}

        # Set field limit if needed
        current_limit = csv.field_size_limit()
        if self.allow_large_fields:
            csv.field_size_limit(2**31 - 1)

        try:
            for csv_file in output_dir.glob("email_*.csv"):
                domain = csv_file.stem.replace("email_", "")

                try:
                    with open(csv_file, encoding="utf-8") as f:
                        reader = csv.reader(f)
                        # Count rows (excluding header)
                        count = sum(1 for _ in reader) - 1

                    if count > 0:
                        summary[domain] = count
                except Exception as e:
                    self.logger.warning(f"Error counting rows for {csv_file}: {e}")

        finally:
            if self.allow_large_fields:
                csv.field_size_limit(current_limit)

        return summary
