"""
Report generation for email classification results.

Generates comprehensive reports with statistics, metrics,
and visualizations (ASCII-based for terminal).
"""

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .domains import DOMAINS
from .processor import ProcessingStats


@dataclass
class ReportConfig:
    """Configuration for report generation."""

    include_domain_breakdown: bool = True
    include_timing_metrics: bool = True
    include_confidence_analysis: bool = True
    include_recommendations: bool = True


class ClassificationReporter:
    """
    Generates comprehensive classification reports.

    Output formats:
    - JSON (machine-readable)
    - Text (human-readable with ASCII visualization)
    - Markdown (for documentation)
    """

    def __init__(self, config: ReportConfig | None = None) -> None:
        self.config = config or ReportConfig()

    def generate_report(
        self, stats: ProcessingStats, output_dir: Path, input_file: str | None = None
    ) -> dict[str, Any]:
        """Generate full report from processing statistics."""
        report: dict[str, Any] = {
            "meta": {
                "generated_at": datetime.now().isoformat(),
                "input_file": input_file,
                "output_directory": str(output_dir),
            },
            "summary": self._generate_summary(stats),
            "validation": self._generate_validation_summary(stats),
            "skipped": self._generate_skipped_summary(stats),
            "domain_breakdown": self._generate_domain_breakdown(stats),
            "label_distribution_analysis": self._generate_label_distribution_analysis(
                stats
            ),
            "url_distribution_analysis": self._generate_url_distribution_analysis(
                stats
            ),
            "cross_tabulation_analysis": self._generate_cross_tabulation_analysis(
                stats
            ),
            "timing": self._generate_timing_metrics(stats),
            "quality_metrics": self._generate_quality_metrics(stats),
        }

        # Add hybrid workflow stats if available
        if stats.hybrid_workflow.total_hybrid_processed > 0:
            report["hybrid_workflow"] = self._generate_hybrid_workflow_stats(stats)

        if self.config.include_recommendations:
            report["recommendations"] = self._generate_recommendations(stats)

        return report

    def _generate_summary(self, stats: "ProcessingStats") -> dict:
        """Generate summary statistics."""
        classification_rate = (
            stats.total_classified / stats.total_processed * 100
            if stats.total_processed > 0
            else 0
        )

        error_rate = (
            stats.errors / stats.total_processed * 100
            if stats.total_processed > 0
            else 0
        )

        return {
            "total_emails": stats.total_processed,
            "classified": stats.total_classified,
            "unsure": stats.total_unsure,
            "errors": stats.errors,
            "classification_rate_percent": round(classification_rate, 2),
            "error_rate_percent": round(error_rate, 2),
            "unique_domains_found": len(
                [k for k, v in stats.domain_counts.items() if k != "unsure" and v > 0]
            ),
        }

    def _generate_validation_summary(self, stats: ProcessingStats) -> dict:
        """Generate validation statistics summary."""
        validation = stats.validation_stats
        total_attempted = stats.total_processed + validation.total_invalid

        return {
            "total_invalid": validation.total_invalid,
            "invalid_percentage": (
                round(validation.total_invalid / total_attempted * 100, 2)
                if total_attempted > 0
                else 0
            ),
            "breakdown": {
                "invalid_sender_format": validation.invalid_sender_format,
                "invalid_receiver_format": validation.invalid_receiver_format,
                "empty_sender": validation.invalid_empty_sender,
                "empty_receiver": validation.invalid_empty_receiver,
                "empty_subject": validation.invalid_empty_subject,
                "empty_body": validation.invalid_empty_body,
            },
        }

    def _generate_skipped_summary(self, stats: ProcessingStats) -> dict:
        """Generate skipped email statistics summary."""
        skipped = stats.skipped_stats
        total_attempted = (
            stats.total_processed
            + stats.validation_stats.total_invalid
            + skipped.total_skipped
        )

        return {
            "total_skipped": skipped.total_skipped,
            "skipped_percentage": (
                round(skipped.total_skipped / total_attempted * 100, 2)
                if total_attempted > 0
                else 0
            ),
            "breakdown": {
                "body_too_long": skipped.skipped_body_too_long,
            },
        }

    def _generate_domain_breakdown(self, stats: ProcessingStats) -> dict:
        """Generate per-domain statistics."""
        total = stats.total_processed
        breakdown = {}

        for domain, count in sorted(stats.domain_counts.items(), key=lambda x: -x[1]):
            percentage = count / total * 100 if total > 0 else 0

            domain_info = DOMAINS.get(domain)
            display_name = domain_info.display_name if domain_info else domain.title()

            breakdown[domain] = {
                "count": count,
                "percentage": round(percentage, 2),
                "display_name": display_name,
            }

        return breakdown

    def _generate_timing_metrics(self, stats: ProcessingStats) -> dict:
        """Generate timing and performance metrics."""
        if stats.start_time and stats.end_time:
            duration = (stats.end_time - stats.start_time).total_seconds()
            emails_per_second = stats.total_processed / duration if duration > 0 else 0
        else:
            duration = 0
            emails_per_second = 0

        return {
            "start_time": stats.start_time.isoformat() if stats.start_time else None,
            "end_time": stats.end_time.isoformat() if stats.end_time else None,
            "duration_seconds": round(duration, 2),
            "emails_per_second": round(emails_per_second, 2),
        }

    def _generate_quality_metrics(self, stats: ProcessingStats) -> dict:
        """Generate quality and confidence metrics."""
        # Share of classified emails where method1 and method2 chose the same
        # domain (not the share of emails that received any domain)
        classified = stats.total_classified
        agreement_rate = (
            stats.method_agreement_count / classified * 100 if classified > 0 else 0
        )

        # Domain distribution evenness (entropy-based)
        domain_counts = [v for k, v in stats.domain_counts.items() if k != "unsure"]
        if domain_counts:
            total_classified = sum(domain_counts)
            if total_classified > 0:
                import math

                probabilities = [c / total_classified for c in domain_counts]
                entropy = -sum(p * math.log2(p) for p in probabilities if p > 0)
                max_entropy = (
                    math.log2(len(domain_counts)) if len(domain_counts) > 1 else 1
                )
                evenness = entropy / max_entropy if max_entropy > 0 else 0
            else:
                evenness = 0
        else:
            evenness = 0

        return {
            "method_agreement_rate": round(agreement_rate, 2),
            "domain_distribution_evenness": round(evenness, 4),
            "unsure_rate": (
                round(stats.total_unsure / stats.total_processed * 100, 2)
                if stats.total_processed > 0
                else 0
            ),
        }

    def _generate_hybrid_workflow_stats(self, stats: ProcessingStats) -> dict:
        """Generate hybrid workflow statistics."""
        hybrid = stats.hybrid_workflow
        return {
            "total_processed": hybrid.total_hybrid_processed,
            "llm_call_count": hybrid.llm_call_count,
            "classic_agreement_count": hybrid.classic_agreement_count,
            "agreement_rate": round(hybrid.agreement_rate, 2),
            "llm_total_time_ms": round(hybrid.llm_total_time_ms, 2),
            "llm_avg_time_ms": round(hybrid.llm_avg_time_ms, 2),
            "llm_savings_percent": (
                round(
                    (1 - hybrid.llm_call_count / hybrid.total_hybrid_processed) * 100, 2
                )
                if hybrid.total_hybrid_processed > 0
                else 0
            ),
        }

    def _generate_recommendations(self, stats: ProcessingStats) -> list[str]:
        """Generate actionable recommendations based on results."""
        recommendations = []

        unsure_rate = (
            stats.total_unsure / stats.total_processed * 100
            if stats.total_processed > 0
            else 0
        )

        if unsure_rate > 30:
            recommendations.append(
                "High unsure rate detected (>30%). Consider expanding keyword taxonomies "
                "or adjusting classification thresholds."
            )

        if unsure_rate > 50:
            recommendations.append(
                "Very high unsure rate (>50%). The email dataset may contain domains "
                "not currently defined in the taxonomy. Review unsure emails for new domain patterns."
            )

        error_rate = (
            stats.errors / stats.total_processed * 100
            if stats.total_processed > 0
            else 0
        )
        if error_rate > 5:
            recommendations.append(
                f"Error rate of {error_rate:.1f}% detected. Check input data quality "
                "and encoding issues."
            )

        # Check for dominant domains
        if stats.domain_counts:
            max_domain = max(stats.domain_counts.items(), key=lambda x: x[1])
            max_percentage = (
                max_domain[1] / stats.total_processed * 100
                if stats.total_processed > 0
                else 0
            )

            if max_percentage > 70 and max_domain[0] != "unsure":
                recommendations.append(
                    f"Domain '{max_domain[0]}' dominates with {max_percentage:.1f}% of emails. "
                    "Consider if the dataset is representative or if classification is over-fitting."
                )

        if not recommendations:
            recommendations.append(
                "Classification completed successfully with balanced results."
            )

        return recommendations

    def _generate_label_distribution_analysis(self, stats: ProcessingStats) -> dict:
        """Generate label distribution analysis by domain.

        Shows what original label values from the input data map to each
        classified domain, helping users understand the classification patterns
        and validate domain assignments.
        """
        analysis = {}

        for domain, label_counts in stats.label_distributions.items():
            total_in_domain = sum(label_counts.values())
            if total_in_domain == 0:
                continue

            distribution = {}
            for label, count in sorted(label_counts.items(), key=lambda x: -x[1]):
                percentage = count / total_in_domain * 100
                distribution[label] = {
                    "count": count,
                    "percentage": round(percentage, 2),
                }

            analysis[domain] = {
                "total_emails": total_in_domain,
                "unique_labels": len(label_counts),
                "distribution": distribution,
            }

        return analysis

    def _generate_url_distribution_analysis(self, stats: ProcessingStats) -> dict:
        """Generate URL presence analysis by domain.

        Shows the distribution of emails with and without URLs for each
        classified domain, helping identify URL patterns across business domains.
        """
        analysis = {}

        for domain, url_counts in stats.url_distributions.items():
            total_in_domain = sum(url_counts.values())
            if total_in_domain == 0:
                continue

            with_urls = url_counts.get(True, 0)
            without_urls = url_counts.get(False, 0)

            analysis[domain] = {
                "total_emails": total_in_domain,
                "with_urls": {
                    "count": with_urls,
                    "percentage": round(with_urls / total_in_domain * 100, 2),
                },
                "without_urls": {
                    "count": without_urls,
                    "percentage": round(without_urls / total_in_domain * 100, 2),
                },
            }

        return analysis

    def _generate_cross_tabulation_analysis(self, stats: ProcessingStats) -> dict:
        """Generate cross-tabulation analysis of labels vs URL presence by domain.

        Creates a matrix showing for each domain: how many emails had each
        original label value with URLs present vs. URLs absent, enabling
        deeper analysis of classification patterns.
        """
        analysis: dict[str, Any] = {}

        for domain, label_data in stats.cross_tabulation.items():
            domain_analysis: dict[str, dict[str, int | float]] = {}
            total_in_domain = 0

            # Calculate totals first
            for label, url_data in label_data.items():
                domain_total = sum(url_data.values())
                domain_analysis[label] = {
                    "with_urls": url_data.get(True, 0),
                    "without_urls": url_data.get(False, 0),
                    "total": domain_total,
                }
                total_in_domain += domain_total

            # Add percentages
            if total_in_domain > 0:
                for label in domain_analysis:
                    data = domain_analysis[label]
                    data["with_urls_percentage"] = (
                        round(data["with_urls"] / data["total"] * 100, 2)
                        if data["total"] > 0
                        else 0.0
                    )
                    data["without_urls_percentage"] = (
                        round(data["without_urls"] / data["total"] * 100, 2)
                        if data["total"] > 0
                        else 0.0
                    )
                    data["percentage_of_domain"] = round(
                        data["total"] / total_in_domain * 100, 2
                    )

            analysis[domain] = {
                "total_emails": total_in_domain,
                "labels": domain_analysis,
            }

        return analysis

    def save_json_report(self, report: dict, output_path: Path) -> None:
        """Save report as JSON file."""
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

    def save_text_report(self, report: dict, output_path: Path) -> None:
        """Save report as formatted text with ASCII visualization."""
        lines = []

        # Header
        lines.append("═" * 80)
        lines.append("                    EMAIL CLASSIFICATION REPORT")
        lines.append("═" * 80)
        lines.append("")

        # Meta info
        lines.append(f"Generated: {report['meta']['generated_at']}")
        if report["meta"]["input_file"]:
            lines.append(f"Input: {report['meta']['input_file']}")
        lines.append(f"Output: {report['meta']['output_directory']}")
        lines.append("")

        # Summary section
        lines.append("─" * 80)
        lines.append("  SUMMARY")
        lines.append("─" * 80)
        summary = report["summary"]
        lines.append(f"  Total Emails Processed:  {summary['total_emails']:,}")
        lines.append(
            f"  Successfully Classified: {summary['classified']:,} ({summary['classification_rate_percent']}%)"
        )
        lines.append(f"  Unsure/Unclassified:     {summary['unsure']:,}")
        lines.append(
            f"  Errors:                  {summary['errors']:,} ({summary['error_rate_percent']}%)"
        )
        lines.append(f"  Unique Domains Found:    {summary['unique_domains_found']}")
        lines.append("")

        # Validation section
        if report.get("validation") and report["validation"]["total_invalid"] > 0:
            lines.append("─" * 80)
            lines.append("  DATA VALIDATION")
            lines.append("─" * 80)
            validation = report["validation"]
            lines.append(
                f"  Invalid Emails Skipped:  {validation['total_invalid']:,} ({validation['invalid_percentage']}%)"
            )
            lines.append("  Validation Errors Breakdown:")
            breakdown = validation["breakdown"]
            if breakdown["invalid_sender_format"] > 0:
                lines.append(
                    f"    - Invalid sender format:   {breakdown['invalid_sender_format']:,}"
                )
            if breakdown["invalid_receiver_format"] > 0:
                lines.append(
                    f"    - Invalid receiver format: {breakdown['invalid_receiver_format']:,}"
                )
            if breakdown["empty_sender"] > 0:
                lines.append(
                    f"    - Empty sender:            {breakdown['empty_sender']:,}"
                )
            if breakdown["empty_receiver"] > 0:
                lines.append(
                    f"    - Empty receiver:          {breakdown['empty_receiver']:,}"
                )
            if breakdown["empty_subject"] > 0:
                lines.append(
                    f"    - Empty subject:           {breakdown['empty_subject']:,}"
                )
            if breakdown["empty_body"] > 0:
                lines.append(
                    f"    - Empty body:              {breakdown['empty_body']:,}"
                )
            lines.append("  (See invalid_emails.csv for details)")
            lines.append("")

        # Skipped emails section
        if report.get("skipped") and report["skipped"]["total_skipped"] > 0:
            lines.append("─" * 80)
            lines.append("  SKIPPED EMAILS")
            lines.append("─" * 80)
            skipped = report["skipped"]
            lines.append(
                f"  Total Skipped:           {skipped['total_skipped']:,} ({skipped['skipped_percentage']}%)"
            )
            lines.append("  Skip Reasons Breakdown:")
            breakdown = skipped["breakdown"]
            if breakdown["body_too_long"] > 0:
                lines.append(
                    f"    - Body too long:           {breakdown['body_too_long']:,}"
                )
            lines.append("  (See skipped_emails.csv for details)")
            lines.append("")

        # Hybrid workflow section
        if report.get("hybrid_workflow"):
            lines.append("─" * 80)
            lines.append("  HYBRID WORKFLOW STATISTICS")
            lines.append("─" * 80)
            hybrid = report["hybrid_workflow"]
            lines.append(f"  Total Processed (Hybrid): {hybrid['total_processed']:,}")
            lines.append(
                f"  Classic Agreement Count:  {hybrid['classic_agreement_count']:,}"
            )
            lines.append(f"  Agreement Rate:           {hybrid['agreement_rate']}%")
            lines.append(f"  LLM Calls Made:           {hybrid['llm_call_count']:,}")
            lines.append(
                f"  LLM Savings:              {hybrid['llm_savings_percent']}%"
            )
            if hybrid["llm_call_count"] > 0:
                lines.append(
                    f"  LLM Total Time:           {hybrid['llm_total_time_ms']:.0f}ms"
                )
                lines.append(
                    f"  LLM Avg Response Time:    {hybrid['llm_avg_time_ms']:.0f}ms"
                )
            lines.append("")

        # Domain breakdown with bar chart
        lines.append("─" * 80)
        lines.append("  DOMAIN BREAKDOWN")
        lines.append("─" * 80)

        breakdown = report["domain_breakdown"]
        max_count = max(d["count"] for d in breakdown.values()) if breakdown else 1

        for domain, data in breakdown.items():
            bar_length = int(data["count"] / max_count * 30) if max_count > 0 else 0
            bar = "█" * bar_length + "░" * (30 - bar_length)
            lines.append(
                f"  {data['display_name']:<20} {bar} {data['count']:>8,} ({data['percentage']:>5.1f}%)"
            )
        lines.append("")

        # Label distribution analysis
        if report.get("label_distribution_analysis"):
            lines.append("─" * 80)
            lines.append("  LABEL DISTRIBUTION ANALYSIS")
            lines.append("─" * 80)

            label_analysis = report["label_distribution_analysis"]
            for domain, data in label_analysis.items():
                domain_info = DOMAINS.get(domain)
                display_name = (
                    domain_info.display_name if domain_info else domain.title()
                )

                lines.append(
                    f"  {display_name} ({data['total_emails']} emails, {data['unique_labels']} labels):"
                )

                # Show top 5 labels for this domain
                sorted_labels = sorted(
                    data["distribution"].items(), key=lambda x: -x[1]["percentage"]
                )[:5]
                for label, label_data in sorted_labels:
                    bar_length = int(label_data["percentage"] / 100 * 20)
                    bar = "█" * bar_length + "░" * (20 - bar_length)
                    label_str = str(label) if label is not None else "(none)"
                    lines.append(
                        f"    {label_str:<15} {bar} {label_data['count']:>6,} ({label_data['percentage']:>5.1f}%)"
                    )

                if len(data["distribution"]) > 5:
                    lines.append(
                        f"    ... and {len(data['distribution']) - 5} more labels"
                    )
                lines.append("")

        # URL distribution analysis
        if report.get("url_distribution_analysis"):
            lines.append("─" * 80)
            lines.append("  URL PRESENCE ANALYSIS")
            lines.append("─" * 80)

            url_analysis = report["url_distribution_analysis"]
            for domain, data in url_analysis.items():
                domain_info = DOMAINS.get(domain)
                display_name = (
                    domain_info.display_name if domain_info else domain.title()
                )

                with_urls = data["with_urls"]
                without_urls = data["without_urls"]

                lines.append(f"  {display_name}:")
                lines.append(
                    f"    With URLs:    {with_urls['count']:>6,} ({with_urls['percentage']:>5.1f}%)"
                )
                lines.append(
                    f"    Without URLs: {without_urls['count']:>6,} ({without_urls['percentage']:>5.1f}%)"
                )
                lines.append("")

        # Cross-tabulation analysis (compact format)
        if report.get("cross_tabulation_analysis"):
            lines.append("─" * 80)
            lines.append("  CROSS-TABULATION INSIGHTS")
            lines.append("─" * 80)

            cross_analysis = report["cross_tabulation_analysis"]
            for domain, data in cross_analysis.items():
                domain_info = DOMAINS.get(domain)
                display_name = (
                    domain_info.display_name if domain_info else domain.title()
                )

                lines.append(f"  {display_name} ({data['total_emails']} emails):")

                # Show top 3 labels by domain percentage
                sorted_labels = sorted(
                    data["labels"].items(), key=lambda x: -x[1]["percentage_of_domain"]
                )[:3]
                for label, label_data in sorted_labels:
                    with_pct = label_data["with_urls_percentage"]
                    without_pct = label_data["without_urls_percentage"]
                    label_str = str(label) if label is not None else "(none)"
                    lines.append(
                        f"    {label_str:<12}: {label_data['total']:>4,} ({label_data['percentage_of_domain']:>5.1f}%) [URLs: {with_pct:>4.1f}%]"
                    )

                if len(data["labels"]) > 3:
                    lines.append(
                        f"    ... and {len(data['labels']) - 3} more label patterns"
                    )
                lines.append("")

        # Timing metrics
        lines.append("─" * 80)
        lines.append("  PERFORMANCE METRICS")
        lines.append("─" * 80)
        timing = report["timing"]
        lines.append(f"  Duration:            {timing['duration_seconds']:.2f} seconds")
        lines.append(
            f"  Processing Speed:    {timing['emails_per_second']:.2f} emails/second"
        )
        lines.append("")

        # Quality metrics
        lines.append("─" * 80)
        lines.append("  QUALITY METRICS")
        lines.append("─" * 80)
        quality = report["quality_metrics"]
        lines.append(
            f"  Method Agreement Rate:      {quality['method_agreement_rate']}%"
        )
        lines.append(f"  Unsure Rate:                {quality['unsure_rate']}%")
        lines.append(
            f"  Domain Distribution Score:  {quality['domain_distribution_evenness']:.4f}"
        )
        lines.append("")

        # Recommendations
        if "recommendations" in report:
            lines.append("─" * 80)
            lines.append("  RECOMMENDATIONS")
            lines.append("─" * 80)
            for i, rec in enumerate(report["recommendations"], 1):
                # Word wrap recommendations
                words = rec.split()
                current_line = f"  {i}. "
                for word in words:
                    if len(current_line) + len(word) > 76:
                        lines.append(current_line)
                        current_line = "     " + word + " "
                    else:
                        current_line += word + " "
                if current_line.strip():
                    lines.append(current_line)
            lines.append("")

        # Footer
        lines.append("═" * 80)
        lines.append("                         END OF REPORT")
        lines.append("═" * 80)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

    def format_terminal_summary(self, report: dict) -> str:
        """Format a brief summary for terminal output."""
        summary = report["summary"]
        timing = report["timing"]

        lines = [
            "",
            "┌─────────────────────────────────────────────────────────┐",
            "│              CLASSIFICATION COMPLETE                    │",
            "├─────────────────────────────────────────────────────────┤",
            f"│  Processed: {summary['total_emails']:>10,} emails                       │",
            f"│  Classified: {summary['classified']:>9,} ({summary['classification_rate_percent']:>5.1f}%)                     │",
            f"│  Unsure: {summary['unsure']:>13,}                                │",
            f"│  Duration: {timing['duration_seconds']:>10.2f}s @ {timing['emails_per_second']:.0f} emails/sec       │",
            "└─────────────────────────────────────────────────────────┘",
            "",
        ]

        return "\n".join(lines)
