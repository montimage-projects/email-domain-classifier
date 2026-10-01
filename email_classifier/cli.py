#!/usr/bin/env python3
"""Email Domain Classifier CLI.

A command-line tool for classifying emails by domain using
dual-method validation (keyword taxonomy + structural templates)
with optional LLM-based classification.

Usage:
    email-cli input.csv -o output_dir/
    email-cli classify input.csv -o output_dir/ --verbose
    email-cli classify input.csv -o output_dir/ --use-llm
    email-cli info input.csv
    email-cli info input.csv --json
"""

import argparse
import json
import logging
import sys
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from dotenv import load_dotenv

from .analyzer import DatasetAnalyzer
from .classifier import EmailClassifier, HybridClassifier, HybridWorkflowLogger
from .domains import get_domain_names
from .processor import StreamingProcessor
from .reporter import ClassificationReporter
from .ui import RICH_AVAILABLE, StatusBar, get_ui

if TYPE_CHECKING:
    from .llm import LLMConfig
    from .ui import SimpleUI, TerminalUI


def verify_prerequisites(
    input_path: Path,
    output_dir: Path,
    use_llm: bool,
    ui: "TerminalUI | SimpleUI",
    quiet: bool = False,
) -> tuple[bool, list[str], Optional["LLMConfig"]]:
    """Verify all prerequisites before starting analysis.

    Checks:
    1. Input file exists and is readable
    2. Output directory can be created/written to
    3. LLM model is available and properly loaded (if --use-llm)
    4. Required dependencies are installed

    Args:
        input_path: Path to input CSV file.
        output_dir: Path to output directory.
        use_llm: Whether LLM classification is enabled.
        ui: Terminal UI instance for output.
        quiet: Suppress output if True.

    Returns:
        Tuple of (success, error_messages, llm_config).
        If success is False, error_messages contains the issues found.
        llm_config is returned if LLM is enabled and verified.
    """
    errors: list[str] = []
    warnings: list[str] = []
    llm_config = None

    if not quiet:
        ui.print_info("Verifying prerequisites...")

    # -------------------------------------------------------------------------
    # 1. Verify input file
    # -------------------------------------------------------------------------
    if not quiet:
        ui.print_info("  [1/4] Checking input file...")

    if not input_path.exists():
        errors.append(f"Input file not found: {input_path}")
    elif not input_path.is_file():
        errors.append(f"Input path is not a file: {input_path}")
    elif input_path.suffix.lower() != ".csv":
        errors.append(f"Input file must be a CSV file: {input_path}")
    else:
        # Try to read first few bytes to verify access
        try:
            with open(input_path, "rb") as f:
                f.read(1024)
            if not quiet:
                ui.print_success("  [1/4] Input file: OK")
        except PermissionError:
            errors.append(f"Permission denied reading input file: {input_path}")
        except Exception as e:
            errors.append(f"Cannot read input file: {e}")

    # -------------------------------------------------------------------------
    # 2. Verify output directory
    # -------------------------------------------------------------------------
    if not quiet:
        ui.print_info("  [2/4] Checking output directory...")

    try:
        # Try to create the directory
        output_dir.mkdir(parents=True, exist_ok=True)

        # Try to create a test file to verify write permissions
        test_file = output_dir / ".write_test"
        try:
            test_file.write_text("test")
            test_file.unlink()
            if not quiet:
                ui.print_success("  [2/4] Output directory: OK")
        except PermissionError:
            errors.append(
                f"Permission denied writing to output directory: {output_dir}"
            )
        except Exception as e:
            errors.append(f"Cannot write to output directory: {e}")
    except PermissionError:
        errors.append(f"Permission denied creating output directory: {output_dir}")
    except Exception as e:
        errors.append(f"Cannot create output directory: {e}")

    # -------------------------------------------------------------------------
    # 3. Verify LLM configuration (if enabled)
    # -------------------------------------------------------------------------
    if use_llm:
        if not quiet:
            ui.print_info("  [3/4] Checking LLM configuration...")

        # Load .env file
        load_dotenv()

        try:
            from .llm import LLMConfig
            from .llm.config import LLMConfigError, LLMProvider
            from .llm.providers import (
                ProviderNotInstalledError,
                check_provider_available,
                create_llm,
            )

            # Load and validate configuration
            llm_config = LLMConfig.from_env()

            if not quiet:
                ui.print_info(f"        Provider: {llm_config.provider.value}")
                ui.print_info(f"        Model: {llm_config.model}")

            if llm_config.provider == LLMProvider.TYPESAFE:
                if llm_config.typesafe_base_url:
                    # The custom System One transport uses only the standard library.
                    if not quiet:
                        ui.print_success("  [3/4] LLM configuration: OK")
                else:
                    # Hosted TypeSafe uses the optional SDK and requires its API key.
                    available, _ = check_provider_available(LLMProvider.TYPESAFE)
                    if not available:
                        errors.append(
                            "Provider 'typesafe' requires package 'typesafe-sdk'. "
                            f"Install with: {llm_config.get_install_command()}"
                        )
                    elif not quiet:
                        ui.print_success("  [3/4] LLM configuration: OK")
            else:
                # Try to create the LLM instance to verify provider is installed
                try:
                    llm_instance = create_llm(llm_config)

                    # For Ollama, verify the model is available
                    if llm_config.provider.value == "ollama":
                        if not quiet:
                            ui.print_info("        Verifying Ollama connection...")
                        try:
                            import httpx

                            # Check if Ollama is running
                            response = httpx.get(
                                f"{llm_config.ollama_base_url}/api/tags",
                                timeout=5.0,
                            )
                            if response.status_code != 200:
                                errors.append(
                                    f"Ollama server not responding at {llm_config.ollama_base_url}. "
                                    "Is Ollama running? Start with: ollama serve"
                                )
                            else:
                                # Check if the model is available
                                models_data = response.json()
                                available_models = [
                                    m.get("name", "").split(":")[0]
                                    for m in models_data.get("models", [])
                                ]
                                model_name = llm_config.model.split(":")[0]
                                if model_name not in available_models:
                                    errors.append(
                                        f"Ollama model '{llm_config.model}' not found. "
                                        f"Available models: {', '.join(available_models) or 'none'}. "
                                        f"Pull with: ollama pull {llm_config.model}"
                                    )
                                else:
                                    if not quiet:
                                        ui.print_success(
                                            "  [3/4] LLM configuration: OK"
                                        )
                        except httpx.ConnectError:
                            errors.append(
                                f"Cannot connect to Ollama at {llm_config.ollama_base_url}. "
                                "Is Ollama running? Start with: ollama serve"
                            )
                        except Exception as e:
                            errors.append(f"Error connecting to Ollama: {e}")
                    else:
                        # For cloud providers, we trust the config is valid
                        # (actual API validation happens on first request)
                        if not quiet:
                            ui.print_success("  [3/4] LLM configuration: OK")

                except ProviderNotInstalledError as e:
                    errors.append(str(e))

        except ImportError as e:
            errors.append(
                "LLM dependencies not installed. "
                "Install with: pip install email-domain-classifier[llm]\n"
                f"Details: {e}"
            )
        except LLMConfigError as e:
            errors.append(f"LLM configuration error: {e}")
        except Exception as e:
            errors.append(f"LLM initialization error: {e}")
    else:
        if not quiet:
            ui.print_info("  [3/4] LLM configuration: Skipped (not enabled)")

    # -------------------------------------------------------------------------
    # 4. Verify other dependencies
    # -------------------------------------------------------------------------
    if not quiet:
        ui.print_info("  [4/4] Checking dependencies...")

    # Check for Rich (optional but recommended)
    try:
        import rich

        if not quiet:
            ui.print_success("  [4/4] Dependencies: OK")
    except ImportError:
        warnings.append(
            "Rich library not installed. Terminal output will be basic. "
            "Install with: pip install rich"
        )
        if not quiet:
            ui.print_warning(
                "  [4/4] Dependencies: OK (Rich not installed, basic output)"
            )

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    if errors:
        if not quiet:
            ui.print_error("\nPrerequisite check failed:")
            for error in errors:
                ui.print_error(f"  • {error}")
        return False, errors, None

    if warnings and not quiet:
        ui.print_warning("\nWarnings:")
        for warning in warnings:
            ui.print_warning(f"  • {warning}")

    if not quiet:
        ui.print_success("\nAll prerequisites verified successfully!\n")

    return True, [], llm_config


def setup_logging(log_file: Path, verbose: bool = False) -> logging.Logger:
    """Configure logging to file and optionally console."""
    logger = logging.getLogger("email_classifier")
    logger.setLevel(logging.DEBUG)

    # File handler - always detailed
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(file_format)
    logger.addHandler(file_handler)

    # Console handler - only if verbose
    if verbose:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_format = logging.Formatter("%(levelname)s: %(message)s")
        console_handler.setFormatter(console_format)
        logger.addHandler(console_handler)

    return logger


def validate_input(input_path: Path) -> bool:
    """Validate input file exists and is readable."""
    if not input_path.exists():
        return False
    if not input_path.is_file():
        return False
    if input_path.suffix.lower() != ".csv":
        return False
    return True


def cmd_info(args: argparse.Namespace) -> int:
    """Execute the info command to analyze a dataset."""
    ui = get_ui(quiet=args.quiet)

    # Validate input file
    input_path = Path(args.input).resolve()
    if not input_path.exists():
        if args.json:
            print(json.dumps({"error": f"File not found: {args.input}"}))
        else:
            ui.print_error(f"File not found: {args.input}")
        return 1

    if input_path.suffix.lower() != ".csv":
        if args.json:
            print(json.dumps({"error": "Invalid file type. Expected .csv file"}))
        else:
            ui.print_error("Invalid file type. Expected .csv file")
        return 1

    # Initialize analyzer
    analyzer = DatasetAnalyzer(allow_large_fields=args.allow_large_fields)

    # Progress callback for terminal
    def progress_callback(current: int, total: int, status: str) -> None:
        if not args.quiet and not args.json and RICH_AVAILABLE:
            # Progress is handled by the UI
            pass

    try:
        # Run analysis
        if not args.quiet and not args.json:
            ui.print_info(f"Analyzing {input_path.name}...")

        if RICH_AVAILABLE and not args.quiet and not args.json:
            progress = ui.create_progress()
            task_id = None

            def rich_progress(current: int, total: int, status: str) -> None:
                nonlocal task_id
                if progress is not None:
                    if task_id is None and total > 0:
                        task_id = progress.add_task(
                            "[cyan]Analyzing dataset...", total=total
                        )
                    if task_id is not None:
                        progress.update(
                            task_id, completed=current, description=f"[cyan]{status}"
                        )

            if progress is not None:
                with progress:
                    result = analyzer.analyze(
                        input_path, progress_callback=rich_progress
                    )
            else:
                result = analyzer.analyze(input_path, progress_callback=rich_progress)
        else:
            result = analyzer.analyze(input_path, progress_callback=progress_callback)

        # Output results
        if args.json:
            print(json.dumps(result.to_dict(), indent=2))
        else:
            ui.print_analysis_report(result)

        return 0

    except Exception as e:
        if args.json:
            print(json.dumps({"error": str(e)}))
        else:
            ui.print_error(f"Analysis failed: {e}")
        return 1


def _positive_int_arg(value: str) -> int:
    """Parse a positive worker count for argparse."""
    try:
        parsed = int(value)
        if parsed >= 1:
            return parsed
    except ValueError:
        pass
    raise argparse.ArgumentTypeError("must be an integer >= 1")


def _confidence_cutoff_arg(value: str) -> float:
    """Parse --llm-confidence-cutoff for argparse."""
    from .llm.config import LLMConfigError, parse_confidence_cutoff

    try:
        return parse_confidence_cutoff(value)
    except LLMConfigError:
        raise argparse.ArgumentTypeError(
            f"invalid value {value!r}: must be a number between 0.0 and 1.0"
        ) from None


def cmd_classify(args: argparse.Namespace) -> int:
    """Execute the classify command."""
    # Initialize UI
    ui = get_ui(quiet=args.quiet)

    # Display banner
    if not args.quiet:
        ui.print_banner()

    # Validate --force-llm requires --use-llm
    force_llm = getattr(args, "force_llm", False)
    if force_llm and not args.use_llm:
        ui.print_error("--force-llm requires --use-llm")
        return 1

    # The confidence cutoff only gates the hybrid workflow
    llm_confidence_cutoff = getattr(args, "llm_confidence_cutoff", None)
    if llm_confidence_cutoff is not None and (not args.use_llm or force_llm):
        ui.print_error(
            "--llm-confidence-cutoff requires --use-llm in hybrid mode "
            "(not with --force-llm)"
        )
        return 1

    # Resolve paths
    input_path = Path(args.input).resolve()
    output_dir = Path(args.output).resolve()

    # =========================================================================
    # PREREQUISITE VERIFICATION
    # =========================================================================
    success, errors, llm_config = verify_prerequisites(
        input_path=input_path,
        output_dir=output_dir,
        use_llm=args.use_llm,
        ui=ui,
        quiet=args.quiet,
    )

    if not success:
        # Errors already printed by verify_prerequisites
        return 1

    # Setup logging (output_dir was already created by verify_prerequisites)
    log_file = (
        Path(args.log_file) if args.log_file else output_dir / "classification.log"
    )
    logger = setup_logging(log_file, verbose=args.verbose)

    # Determine workflow mode
    use_hybrid = args.use_llm and not force_llm

    logger.info("=" * 60)
    logger.info("EMAIL DOMAIN CLASSIFIER - Started")
    logger.info("=" * 60)
    logger.info(f"Input file: {input_path}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Chunk size: {args.chunk_size}")
    logger.info(f"Workers: {getattr(args, 'workers', 1)}")
    logger.info(f"Include details: {args.include_details}")
    logger.info(f"Strict validation: {args.strict_validation}")
    logger.info(f"LLM enabled: {args.use_llm}")
    logger.info(f"Hybrid workflow: {use_hybrid}")
    logger.info(f"Force LLM: {force_llm}")
    if args.max_body_length:
        logger.info(f"Max body length: {args.max_body_length}")
    if llm_config:
        logger.info(f"LLM provider: {llm_config.provider.value}")
        logger.info(f"LLM model: {llm_config.model}")

    # Display configuration
    if not args.quiet:
        config_opts = {
            "Chunk Size": args.chunk_size,
            "Workers": getattr(args, "workers", 1),
            "Include Details": args.include_details,
            "Strict Validation": args.strict_validation,
            "Log File": str(log_file),
        }
        if args.use_llm and llm_config:
            if use_hybrid:
                config_opts["LLM Classification"] = (
                    f"Hybrid ({llm_config.provider.value}/{llm_config.model})"
                )
            else:
                config_opts["LLM Classification"] = (
                    f"Force ({llm_config.provider.value}/{llm_config.model})"
                )
        if args.max_body_length:
            config_opts["Max Body Length"] = f"{args.max_body_length:,} chars"
        ui.print_config(
            str(input_path),
            str(output_dir),
            config_opts,
        )

    # Initialize workflow logger for hybrid mode
    workflow_logger: Optional[HybridWorkflowLogger] = None
    if use_hybrid:
        workflow_log_path = output_dir / "hybrid_workflow.jsonl"
        workflow_logger = HybridWorkflowLogger(str(workflow_log_path))
        logger.info(f"Hybrid workflow log: {workflow_log_path}")

    # Create appropriate classifier based on mode
    classifier: EmailClassifier | HybridClassifier
    # Status callback will be set after progress bar is created (for hybrid mode)
    hybrid_status_callback: Optional[Callable[[str], None]] = None

    if use_hybrid:
        # Hybrid mode: LLM only when classifiers disagree
        # Status callback will be set below after progress bar is created
        classifier = HybridClassifier(
            llm_config=llm_config,
            workflow_logger=workflow_logger,
            llm_confidence_cutoff=llm_confidence_cutoff,
        )
        logger.info(f"LLM confidence cutoff: {classifier.llm_confidence_cutoff}")
    else:
        # Standard mode: dual-method or three-method (with --force-llm)
        classifier = EmailClassifier(llm_config=llm_config, use_llm=args.use_llm)

    processor = StreamingProcessor(
        classifier=classifier,
        chunk_size=args.chunk_size,
        logger=logger,
        allow_large_fields=args.allow_large_fields,
        strict_validation=args.strict_validation,
        max_body_length=args.max_body_length,
        use_hybrid=use_hybrid,
        workers=getattr(args, "workers", 1),
    )

    # Process emails with progress tracking
    try:
        if RICH_AVAILABLE and not args.quiet:
            # Use Rich progress bar
            progress = ui.create_progress()
            task_id = None
            current_hybrid_status = ""

            def progress_callback(current: int, total: int, status: str) -> None:
                nonlocal task_id, current_hybrid_status
                if progress is not None:
                    if task_id is None and total > 0:
                        task_id = progress.add_task(
                            "[cyan]Classifying emails...", total=total
                        )
                    if task_id is not None:
                        # Include hybrid status if available
                        display_status = status
                        if current_hybrid_status:
                            display_status = f"{status} | {current_hybrid_status}"
                        progress.update(
                            task_id,
                            completed=current,
                            description=f"[cyan]{display_status}",
                        )

            def hybrid_status_update(message: str) -> None:
                nonlocal task_id, current_hybrid_status
                current_hybrid_status = message
                # Update progress bar description immediately
                if progress is not None and task_id is not None:
                    progress.update(task_id, description=f"[cyan]{message}")

            # Set the status callback on the hybrid classifier
            if use_hybrid and isinstance(classifier, HybridClassifier):
                classifier.status_callback = hybrid_status_update

            if progress is not None:
                with progress:
                    stats = processor.process(
                        input_path=input_path,
                        output_dir=output_dir,
                        progress_callback=progress_callback,
                        include_details=args.include_details,
                    )
            else:
                stats = processor.process(
                    input_path=input_path,
                    output_dir=output_dir,
                    progress_callback=progress_callback,
                    include_details=args.include_details,
                )
        else:
            # Simple progress for non-Rich environments
            def simple_progress(current: int, total: int, status: str) -> None:
                if hasattr(ui, "print_progress"):
                    ui.print_progress(current, total, status)

            stats = processor.process(
                input_path=input_path,
                output_dir=output_dir,
                progress_callback=simple_progress if not args.quiet else None,
                include_details=args.include_details,
            )

        logger.info("Processing completed successfully")

    except KeyboardInterrupt:
        ui.print_warning("Processing interrupted by user")
        logger.warning("Processing interrupted by user")
        return 130
    except Exception as e:
        ui.print_error(f"Processing failed: {e}")
        logger.exception("Processing failed with exception")
        return 1

    # Generate reports
    if not args.no_report:
        reporter = ClassificationReporter()
        report = reporter.generate_report(
            stats=stats, output_dir=output_dir, input_file=str(input_path)
        )

        # Save JSON report
        json_report_path = output_dir / "classification_report.json"
        reporter.save_json_report(report, json_report_path)
        logger.info(f"JSON report saved: {json_report_path}")

        # Save text report
        if not args.json_only:
            text_report_path = output_dir / "classification_report.txt"
            reporter.save_text_report(report, text_report_path)
            logger.info(f"Text report saved: {text_report_path}")

        # Display results in terminal
        if not args.quiet:
            ui.print_domain_stats(
                dict(stats.domain_counts),
                stats.total_processed,
                report,
                input_file=input_path.name,
            )
            ui.print_summary_panel(report)

            # Show output files
            file_counts = processor.get_output_summary(output_dir)
            ui.print_output_files(output_dir, file_counts)

            # Show recommendations
            if "recommendations" in report:
                ui.print_recommendations(report["recommendations"])

    # Close workflow logger if used
    if workflow_logger:
        workflow_logger.close()

    # Final success message
    if not args.quiet:
        ui.print_success("Classification complete!")
        ui.print_info(f"Results saved to: {output_dir}")

    logger.info("=" * 60)
    logger.info("EMAIL DOMAIN CLASSIFIER - Finished")
    logger.info("=" * 60)

    return 0


def main() -> int:
    """Execute main CLI entry point with subcommand support."""
    # Create main parser
    parser = argparse.ArgumentParser(
        prog="email-cli",
        description="Classify emails by domain using dual-method validation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
  classify    Classify emails in a CSV file by domain (default)
  info        Analyze a dataset and display statistics

Examples:
  email-cli emails.csv -o classified/
  email-cli classify data/input.csv -o results/ --verbose
  email-cli info emails.csv
  email-cli info emails.csv --json

For command-specific help:
  email-cli classify --help
  email-cli info --help
        """,
    )

    parser.add_argument("--version", action="version", version="%(prog)s 1.0.0")

    # Handle --list-domains early (before subparsers)
    if "--list-domains" in sys.argv:
        print("\nSupported Domain Categories:")
        print("-" * 40)
        for domain in get_domain_names():
            print(f"  • {domain}")
        print()
        return 0

    # Create subparsers
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # =========================================================================
    # INFO subcommand
    # =========================================================================
    info_parser = subparsers.add_parser(
        "info",
        help="Analyze a dataset and display statistics",
        description="Analyze a CSV dataset and display comprehensive statistics.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  email-cli info emails.csv
  email-cli info emails.csv --json
  email-cli info dataset.csv --no-large-fields  # Restrict to small fields only
        """,
    )

    info_parser.add_argument(
        "input", type=str, help="Path to input CSV file to analyze"
    )

    info_parser.add_argument(
        "--json",
        action="store_true",
        help="Output analysis results in JSON format",
    )

    info_parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Suppress progress output (still outputs JSON if --json is set)",
    )

    info_parser.add_argument(
        "--no-large-fields",
        dest="allow_large_fields",
        action="store_false",
        default=True,
        help="Disable processing of CSV fields larger than default limit (131KB)",
    )

    # =========================================================================
    # CLASSIFY subcommand
    # =========================================================================
    classify_parser = subparsers.add_parser(
        "classify",
        help="Classify emails in a CSV file by domain",
        description="Classify emails by domain using dual-method validation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  email-cli classify emails.csv -o classified/
  email-cli classify data/input.csv --output results/ --verbose
  email-cli classify emails.csv -o output/ --include-details --chunk-size 500

Output:
  Creates email_[domain].csv files for each detected domain,
  plus email_unsure.csv for unclassified emails.
  Also generates classification_report.txt and classification_report.json.
        """,
    )

    # Required arguments
    classify_parser.add_argument(
        "input", type=str, help="Path to input CSV file containing email dataset"
    )

    # Output options
    classify_parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
        help="Output directory for classified email files",
    )

    # Processing options
    classify_parser.add_argument(
        "--chunk-size",
        type=int,
        default=1,
        help="Number of emails to process before updating progress (default: 1)",
    )

    classify_parser.add_argument(
        "--workers",
        type=_positive_int_arg,
        default=1,
        help="Concurrent classification workers (default: 1, serial). "
        "Output remains in input order; provider concurrency limits apply.",
    )

    classify_parser.add_argument(
        "--include-details",
        action="store_true",
        help="Include detailed classification scores in output files",
    )

    # Logging options
    classify_parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose console output"
    )

    classify_parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Suppress all console output except errors",
    )

    classify_parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Custom log file path (default: output_dir/classification.log)",
    )

    # Report options
    classify_parser.add_argument(
        "--no-report", action="store_true", help="Skip generating summary reports"
    )

    classify_parser.add_argument(
        "--json-only",
        action="store_true",
        help="Generate only JSON report (no text report)",
    )

    # Misc options
    classify_parser.add_argument(
        "--list-domains",
        action="store_true",
        help="List all supported domain categories and exit",
    )

    classify_parser.add_argument(
        "--no-large-fields",
        dest="allow_large_fields",
        action="store_false",
        default=True,
        help="Disable processing of CSV fields larger than default limit (131KB)",
    )

    # Validation options
    classify_parser.add_argument(
        "--strict-validation",
        action="store_true",
        help="Fail processing if any invalid emails are found",
    )

    # Filtering options
    classify_parser.add_argument(
        "--max-body-length",
        type=int,
        default=None,
        metavar="CHARS",
        help="Skip emails with body length exceeding this limit (in characters). "
        "Skipped emails are logged to skipped_emails.csv.",
    )

    # LLM classification options
    classify_parser.add_argument(
        "--use-llm",
        action="store_true",
        help="Enable LLM-based classification (hybrid mode by default: "
        "LLM only when classic classifiers disagree). "
        "Configure LLM settings via .env file (see .env.example). "
        "Requires: pip install email-domain-classifier[llm]",
    )

    classify_parser.add_argument(
        "--force-llm",
        action="store_true",
        help="Force LLM classification for every email (requires --use-llm). "
        "Overrides default hybrid mode to use three-method weighted scoring.",
    )

    classify_parser.add_argument(
        "--llm-confidence-cutoff",
        type=_confidence_cutoff_arg,
        default=None,
        metavar="CUTOFF",
        help="Hybrid mode: minimum LLM confidence (0.0-1.0) accepted when the "
        "classic classifiers disagree; lower-confidence answers fall back to "
        "the classic weighted result. Overrides LLM_CONFIDENCE_CUTOFF "
        "(default 0.5, evaluated for TypeSafe in issue #19; see "
        "docs/evaluation/typesafe-evaluation.md).",
    )

    # =========================================================================
    # Parse arguments with backward compatibility
    # =========================================================================

    # Check if first arg looks like a file (backward compatibility)
    # If no subcommand is given but a .csv file is provided, treat as classify
    args = None
    if len(sys.argv) > 1:
        first_arg = sys.argv[1]
        # If first arg is not a known command and looks like a file path
        if first_arg not in (
            "info",
            "classify",
            "-h",
            "--help",
            "--version",
            "--list-domains",
        ):
            if first_arg.endswith(".csv") or (
                not first_arg.startswith("-") and "-o" in sys.argv
            ):
                # Insert 'classify' as the command for backward compatibility
                sys.argv.insert(1, "classify")

    args = parser.parse_args()

    # Route to appropriate command
    if args.command == "info":
        return cmd_info(args)
    elif args.command == "classify":
        return cmd_classify(args)
    else:
        # No command specified, show help
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())
