"""Tests for enhanced statistics collection and reporting."""

import csv
import tempfile
from collections import defaultdict
from pathlib import Path

import pytest

from email_classifier.classifier import EmailClassifier
from email_classifier.processor import ProcessingStats, StreamingProcessor
from email_classifier.reporter import ClassificationReporter


class TestEnhancedStatistics:
    """Test cases for enhanced statistics functionality."""

    def test_processing_stats_extended_fields(self):
        """Test that ProcessingStats has new distribution fields."""
        stats = ProcessingStats()

        # Test new fields exist and are properly initialized
        assert hasattr(stats, "label_distributions")
        assert hasattr(stats, "url_distributions")
        assert hasattr(stats, "cross_tabulation")

        # Test they are defaultdicts
        assert isinstance(stats.label_distributions, defaultdict)
        assert isinstance(stats.url_distributions, defaultdict)
        assert isinstance(stats.cross_tabulation, defaultdict)

    def test_label_distribution_collection(self):
        """Test collection of label distribution data."""
        stats = ProcessingStats()

        # Simulate data collection
        stats.label_distributions["finance"]["banking"] += 5
        stats.label_distributions["finance"]["investment"] += 3
        stats.label_distributions["tech"]["software"] += 4

        assert stats.label_distributions["finance"]["banking"] == 5
        assert stats.label_distributions["finance"]["investment"] == 3
        assert stats.label_distributions["tech"]["software"] == 4

    def test_url_distribution_collection(self):
        """Test collection of URL distribution data."""
        stats = ProcessingStats()

        # Simulate URL data collection
        stats.url_distributions["finance"][True] += 7
        stats.url_distributions["finance"][False] += 2
        stats.url_distributions["retail"][True] += 10

        assert stats.url_distributions["finance"][True] == 7
        assert stats.url_distributions["finance"][False] == 2
        assert stats.url_distributions["retail"][True] == 10

    def test_cross_tabulation_collection(self):
        """Test collection of cross-tabulation data."""
        stats = ProcessingStats()

        # Simulate cross-tabulation data collection
        stats.cross_tabulation["finance"]["banking"][True] += 3
        stats.cross_tabulation["finance"]["banking"][False] += 2
        stats.cross_tabulation["finance"]["investment"][True] += 4

        assert stats.cross_tabulation["finance"]["banking"][True] == 3
        assert stats.cross_tabulation["finance"]["banking"][False] == 2
        assert stats.cross_tabulation["finance"]["investment"][True] == 4

    def test_to_dict_extended_fields(self):
        """Test that to_dict includes new fields."""
        stats = ProcessingStats()

        # Add some test data
        stats.label_distributions["finance"]["banking"] = 5
        stats.url_distributions["finance"][True] = 3
        stats.cross_tabulation["finance"]["banking"][True] = 2

        result = stats.to_dict()

        # Test new fields are in the dictionary
        assert "label_distributions" in result
        assert "url_distributions" in result
        assert "cross_tabulation" in result

        # Test data is properly converted
        assert result["label_distributions"]["finance"]["banking"] == 5
        assert result["url_distributions"]["finance"][True] == 3
        assert result["cross_tabulation"]["finance"]["banking"][True] == 2

    def test_enhanced_reporter_analysis_methods(self):
        """Test new analysis methods in ClassificationReporter."""
        reporter = ClassificationReporter()
        stats = ProcessingStats()

        # Add test data
        stats.label_distributions["finance"]["banking"] = 8
        stats.label_distributions["finance"]["investment"] = 2
        stats.url_distributions["finance"][True] = 7
        stats.url_distributions["finance"][False] = 3
        stats.cross_tabulation["finance"]["banking"][True] = 5
        stats.cross_tabulation["finance"]["banking"][False] = 3
        stats.cross_tabulation["finance"]["investment"][True] = 2
        stats.cross_tabulation["finance"]["investment"][False] = 0

        # Test new analysis methods
        label_analysis = reporter._generate_label_distribution_analysis(stats)
        url_analysis = reporter._generate_url_distribution_analysis(stats)
        cross_analysis = reporter._generate_cross_tabulation_analysis(stats)

        # Validate label analysis
        assert "finance" in label_analysis
        assert label_analysis["finance"]["total_emails"] == 10
        assert label_analysis["finance"]["unique_labels"] == 2
        assert "banking" in label_analysis["finance"]["distribution"]

        # Validate URL analysis
        assert "finance" in url_analysis
        assert url_analysis["finance"]["with_urls"]["count"] == 7
        assert url_analysis["finance"]["without_urls"]["count"] == 3

        # Validate cross-tabulation analysis
        assert "finance" in cross_analysis
        assert cross_analysis["finance"]["total_emails"] == 10
        assert "banking" in cross_analysis["finance"]["labels"]
        assert cross_analysis["finance"]["labels"]["banking"]["with_urls"] == 5

    def test_enhanced_report_generation(self):
        """Test that enhanced report includes new sections."""
        reporter = ClassificationReporter()
        stats = ProcessingStats()

        # Add minimal test data
        stats.total_processed = 10
        stats.total_classified = 8
        stats.total_unsure = 2
        stats.domain_counts["finance"] = 8
        stats.domain_counts["unsure"] = 2

        # Add enhanced statistics
        stats.label_distributions["finance"]["banking"] = 5
        stats.label_distributions["finance"]["investment"] = 3
        stats.url_distributions["finance"][True] = 6
        stats.url_distributions["finance"][False] = 2

        # Generate report
        report = reporter.generate_report(stats, Path("/tmp"))

        # Test new sections are present
        assert "label_distribution_analysis" in report
        assert "url_distribution_analysis" in report
        assert "cross_tabulation_analysis" in report

        # Test data integrity
        assert report["label_distribution_analysis"]["finance"]["total_emails"] == 8
        assert report["url_distribution_analysis"]["finance"]["with_urls"]["count"] == 6

    def test_streaming_processor_enhanced_collection(self):
        """Test that StreamingProcessor collects enhanced statistics."""
        # Create temporary CSV file with test data
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "sender",
                    "receiver",
                    "timestamp",
                    "subject",
                    "body",
                    "label",
                    "has_url",
                ],
            )
            writer.writeheader()

            # Write test emails
            writer.writerow(
                {
                    "sender": "bank@finance.com",
                    "receiver": "user@test.com",
                    "timestamp": "2024-01-15 10:00:00",
                    "subject": "Account Update",
                    "body": "Your account balance is $1000",
                    "label": "finance",
                    "has_url": "true",
                }
            )
            writer.writerow(
                {
                    "sender": "tech@software.com",
                    "receiver": "user@test.com",
                    "timestamp": "2024-01-15 11:00:00",
                    "subject": "Software Update",
                    "body": "Please update your software",
                    "label": "tech",
                    "has_url": "false",
                }
            )

            temp_path = f.name

        try:
            # Test processing
            processor = StreamingProcessor()
            with tempfile.TemporaryDirectory() as temp_dir:
                stats = processor.process(Path(temp_path), Path(temp_dir))

                # Validate enhanced statistics were collected
                assert len(stats.label_distributions) > 0
                assert len(stats.url_distributions) > 0
                assert len(stats.cross_tabulation) > 0

                # Test specific data was collected
                total_label_count = sum(
                    count
                    for domain_dict in stats.label_distributions.values()
                    for count in domain_dict.values()
                )
                assert total_label_count == 2  # We processed 2 emails

        finally:
            Path(temp_path).unlink()

    def test_backward_compatibility(self):
        """Test that existing functionality remains unchanged."""
        # Test with empty enhanced statistics (backward compatibility)
        stats = ProcessingStats()
        stats.total_processed = 5
        stats.total_classified = 4
        stats.total_unsure = 1
        stats.domain_counts["finance"] = 4
        stats.domain_counts["unsure"] = 1

        reporter = ClassificationReporter()
        report = reporter.generate_report(stats, Path("/tmp"))

        # Test existing sections still work
        assert "summary" in report
        assert "domain_breakdown" in report
        assert "timing" in report
        assert "quality_metrics" in report

        # Test new sections are empty but present
        assert "label_distribution_analysis" in report
        assert "url_distribution_analysis" in report
        assert "cross_tabulation_analysis" in report


class _ScriptedClassifier:
    """Test double returning scripted (domain, method1, method2) per subject."""

    def __init__(self, script: dict[str, tuple[str, str | None, str | None]]):
        self.script = script

    def classify_dict(self, row: dict) -> tuple[str, dict]:
        domain, method1, method2 = self.script[row["subject"]]
        details = {
            "method1": {"domain": method1, "confidence": 0.5, "scores": {}},
            "method2": {"domain": method2, "confidence": 0.5, "scores": {}},
        }
        return domain, details


class TestMethodAgreementRate:
    """Regression tests for issue #14: real method1/method2 agreement rate."""

    def test_rate_uses_agreement_count_not_classification_count(self):
        """Agreement rate is agreed / classified, not classified / processed."""
        stats = ProcessingStats()
        stats.total_processed = 100
        stats.total_classified = 90
        stats.total_unsure = 10
        stats.method_agreement_count = 18
        stats.domain_counts["finance"] = 90
        stats.domain_counts["unsure"] = 10

        report = ClassificationReporter().generate_report(stats, Path("/tmp"))

        # 18 / 90 = 20%, while the classification rate would be 90%
        assert report["quality_metrics"]["method_agreement_rate"] == 20.0

    def test_rate_matches_ceas_08_counts(self):
        """CEAS_08 figures (7,143 agreed of 33,036 classified) give ~21.6%."""
        stats = ProcessingStats()
        stats.total_processed = 34858
        stats.total_classified = 33036
        stats.total_unsure = 1822
        stats.method_agreement_count = 7143
        stats.domain_counts["finance"] = 33036
        stats.domain_counts["unsure"] = 1822

        report = ClassificationReporter().generate_report(stats, Path("/tmp"))

        assert report["quality_metrics"]["method_agreement_rate"] == 21.62

    def test_rate_is_zero_when_nothing_classified(self):
        """No classified emails yields a 0% rate instead of a division error."""
        stats = ProcessingStats()
        stats.total_processed = 3
        stats.total_unsure = 3
        stats.domain_counts["unsure"] = 3

        report = ClassificationReporter().generate_report(stats, Path("/tmp"))

        assert report["quality_metrics"]["method_agreement_rate"] == 0

    def test_to_dict_includes_agreement_count(self):
        """The agreement counter is serialized with the other stats."""
        stats = ProcessingStats()
        stats.method_agreement_count = 7

        assert stats.to_dict()["method_agreement_count"] == 7

    def test_processor_counts_only_classified_agreeing_emails(self):
        """Count rows where both methods name the same domain and it is kept."""
        script = {
            "agree-finance": ("finance", "finance", "finance"),
            "agree-tech": ("technology", "technology", "technology"),
            "disagree": ("finance", "finance", "technology"),
            "both-none": ("unsure", None, None),
            "agree-below-threshold": ("unsure", "retail", "retail"),
            "one-none": ("hr", "hr", None),
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "input.csv"
            with open(input_path, "w", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=["sender", "receiver", "date", "subject", "body"],
                )
                writer.writeheader()
                for subject in script:
                    writer.writerow(
                        {
                            "sender": "alice@example.com",
                            "receiver": "bob@example.com",
                            "date": "2024-01-15 10:00:00",
                            "subject": subject,
                            "body": f"Body for {subject}",
                        }
                    )

            processor = StreamingProcessor(
                classifier=_ScriptedClassifier(script)  # type: ignore[arg-type]
            )
            stats = processor.process(input_path, Path(temp_dir) / "out")

        assert stats.total_processed == 6
        assert stats.total_classified == 4
        assert stats.method_agreement_count == 2

        report = ClassificationReporter().generate_report(stats, Path("/tmp"))
        assert report["quality_metrics"]["method_agreement_rate"] == 50.0
