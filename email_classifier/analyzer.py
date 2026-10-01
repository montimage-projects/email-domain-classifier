"""
Dataset analyzer for email CSV files.

Provides comprehensive statistics and analysis without performing classification.
Uses streaming processing for memory efficiency with large files.
"""

import csv
import os
import re
import statistics
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from .parsing import parse_url_flag


@dataclass
class AnalysisResult:
    """Complete analysis result for a dataset."""

    # File metadata
    file_path: str = ""
    file_size_bytes: int = 0
    total_rows: int = 0

    # Label distribution
    label_counts: dict[str, int] = field(default_factory=dict)

    # Body length statistics
    body_length_min: int = 0
    body_length_max: int = 0
    body_length_mean: float = 0.0
    body_length_median: float = 0.0
    body_length_buckets: dict[str, int] = field(default_factory=dict)

    # Sender domain analysis
    sender_domain_counts: dict[str, int] = field(default_factory=dict)
    total_unique_domains: int = 0

    # Subject statistics
    subject_length_mean: float = 0.0
    subject_empty_count: int = 0

    # URL presence
    url_count: int = 0
    url_percentage: float = 0.0

    # Data quality
    empty_sender_count: int = 0
    empty_receiver_count: int = 0
    empty_subject_count: int = 0
    empty_body_count: int = 0
    invalid_sender_format_count: int = 0
    invalid_receiver_format_count: int = 0

    # Column info
    columns: list[str] = field(default_factory=list)
    has_label_column: bool = False
    has_url_column: bool = False

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON output."""
        return {
            "file": {
                "path": self.file_path,
                "size_bytes": self.file_size_bytes,
                "size_human": self._format_size(self.file_size_bytes),
                "total_rows": self.total_rows,
                "columns": self.columns,
            },
            "labels": {
                "distribution": self.label_counts,
                "has_label_column": self.has_label_column,
            },
            "body_length": {
                "min": self.body_length_min,
                "max": self.body_length_max,
                "mean": round(self.body_length_mean, 1),
                "median": round(self.body_length_median, 1),
                "buckets": self.body_length_buckets,
            },
            "sender_domains": {
                "top_domains": dict(
                    sorted(
                        self.sender_domain_counts.items(),
                        key=lambda x: -x[1],
                    )[:10]
                ),
                "total_unique": self.total_unique_domains,
            },
            "subjects": {
                "mean_length": round(self.subject_length_mean, 1),
                "empty_count": self.subject_empty_count,
            },
            "urls": {
                "count": self.url_count,
                "percentage": round(self.url_percentage, 1),
                "has_url_column": self.has_url_column,
            },
            "data_quality": {
                "empty_sender": self.empty_sender_count,
                "empty_receiver": self.empty_receiver_count,
                "empty_subject": self.empty_subject_count,
                "empty_body": self.empty_body_count,
                "invalid_sender_format": self.invalid_sender_format_count,
                "invalid_receiver_format": self.invalid_receiver_format_count,
            },
        }

    @staticmethod
    def _format_size(size_bytes: int) -> str:
        """Format bytes to human readable size."""
        size_float: float = float(size_bytes)
        for unit in ["B", "KB", "MB", "GB"]:
            if size_float < 1024:
                return f"{size_float:.1f} {unit}"
            size_float /= 1024
        return f"{size_float:.1f} TB"


class DatasetAnalyzer:
    """
    Analyzes email CSV datasets with streaming processing.

    Features:
    - Memory-efficient streaming for large files
    - Comprehensive statistics collection
    - Progress callback support
    """

    # Email regex pattern for validation
    EMAIL_PATTERN = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

    # Body length bucket boundaries
    BODY_BUCKETS = [
        (0, 100, "0-100"),
        (100, 500, "100-500"),
        (500, 1000, "500-1000"),
        (1000, 2000, "1000-2000"),
        (2000, float("inf"), "2000+"),
    ]

    def __init__(self, allow_large_fields: bool = True):
        """Initialize analyzer.

        Args:
            allow_large_fields: Allow CSV fields larger than default limit (default: True).
        """
        self.allow_large_fields = allow_large_fields

    def analyze(
        self,
        file_path: Path,
        progress_callback: Callable[[int, int, str], None] | None = None,
    ) -> AnalysisResult:
        """
        Analyze a CSV dataset.

        Args:
            file_path: Path to CSV file.
            progress_callback: Optional callback(current, total, status).

        Returns:
            AnalysisResult with comprehensive statistics.
        """
        file_path = Path(file_path)
        result = AnalysisResult()

        # File metadata
        result.file_path = str(file_path)
        result.file_size_bytes = os.path.getsize(file_path)

        # Count total rows first for progress
        if progress_callback:
            progress_callback(0, 0, "Counting rows...")

        total_rows = self._count_rows(file_path)

        # Collect data in single pass
        body_lengths: list[int] = []
        subject_lengths: list[int] = []
        label_counts: dict[str, int] = defaultdict(int)
        domain_counts: dict[str, int] = defaultdict(int)
        bucket_counts: dict[str, int] = {b[2]: 0 for b in self.BODY_BUCKETS}

        url_count = 0
        empty_sender = 0
        empty_receiver = 0
        empty_subject = 0
        empty_body = 0
        invalid_sender = 0
        invalid_receiver = 0

        # Configure CSV field limit
        current_limit = csv.field_size_limit()
        if self.allow_large_fields:
            csv.field_size_limit(2**31 - 1)

        try:
            with open(file_path, encoding="utf-8", errors="replace") as f:
                reader = csv.DictReader(f)
                result.columns = list(reader.fieldnames) if reader.fieldnames else []
                result.has_label_column = "label" in result.columns
                result.has_url_column = (
                    "urls" in result.columns or "has_url" in result.columns
                )

                for idx, row in enumerate(reader):
                    # Progress update
                    if progress_callback and idx % 1000 == 0:
                        progress_callback(idx, total_rows, f"Analyzing row {idx:,}")

                    # Label distribution
                    label = row.get("label", "").strip()
                    if label:
                        label_counts[label] += 1
                    else:
                        label_counts["(unlabeled)"] += 1

                    # Body analysis
                    body = row.get("body", "")
                    body_len = len(body)
                    body_lengths.append(body_len)

                    # Bucket the body length
                    for min_val, max_val, bucket_name in self.BODY_BUCKETS:
                        if min_val <= body_len < max_val:
                            bucket_counts[bucket_name] += 1
                            break

                    if not body.strip():
                        empty_body += 1

                    # Subject analysis
                    subject = row.get("subject", "")
                    subject_lengths.append(len(subject))
                    if not subject.strip():
                        empty_subject += 1

                    # Sender analysis
                    sender = row.get("sender", "").strip().lower()
                    if not sender:
                        empty_sender += 1
                    elif not self.EMAIL_PATTERN.match(sender):
                        invalid_sender += 1
                    else:
                        # Extract domain
                        domain = sender.split("@")[-1] if "@" in sender else ""
                        if domain:
                            domain_counts[domain] += 1

                    # Receiver analysis
                    receiver = row.get("receiver", "").strip().lower()
                    if not receiver:
                        empty_receiver += 1
                    elif not self.EMAIL_PATTERN.match(receiver):
                        invalid_receiver += 1

                    # URL presence
                    if parse_url_flag(row.get("has_url", row.get("urls", ""))):
                        url_count += 1

                result.total_rows = idx + 1 if "idx" in dir() else 0

        finally:
            csv.field_size_limit(current_limit)

        if progress_callback:
            progress_callback(total_rows, total_rows, "Calculating statistics...")

        # Populate result
        result.total_rows = len(body_lengths)
        result.label_counts = dict(label_counts)

        # Body statistics
        if body_lengths:
            result.body_length_min = min(body_lengths)
            result.body_length_max = max(body_lengths)
            result.body_length_mean = statistics.mean(body_lengths)
            result.body_length_median = statistics.median(body_lengths)
        result.body_length_buckets = bucket_counts

        # Sender domains
        result.sender_domain_counts = dict(domain_counts)
        result.total_unique_domains = len(domain_counts)

        # Subject statistics
        if subject_lengths:
            result.subject_length_mean = statistics.mean(subject_lengths)
        result.subject_empty_count = empty_subject

        # URL presence
        result.url_count = url_count
        result.url_percentage = (
            (url_count / result.total_rows * 100) if result.total_rows > 0 else 0
        )

        # Data quality
        result.empty_sender_count = empty_sender
        result.empty_receiver_count = empty_receiver
        result.empty_subject_count = empty_subject
        result.empty_body_count = empty_body
        result.invalid_sender_format_count = invalid_sender
        result.invalid_receiver_format_count = invalid_receiver

        if progress_callback:
            progress_callback(total_rows, total_rows, "Analysis complete")

        return result

    def _count_rows(self, file_path: Path) -> int:
        """Count total rows in CSV file."""
        count = 0
        current_limit = csv.field_size_limit()

        try:
            if self.allow_large_fields:
                csv.field_size_limit(2**31 - 1)

            with open(file_path, encoding="utf-8", errors="replace") as f:
                reader = csv.reader(f)
                next(reader, None)  # Skip header
                for _ in reader:
                    count += 1
        finally:
            csv.field_size_limit(current_limit)

        return count
