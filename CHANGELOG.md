# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Standard Python project structure for GitHub publication
- Comprehensive documentation and development workflow
- GitHub Actions CI/CD pipelines
- Security scanning and dependency management
- Community contribution guidelines

### Changed
- Enhanced package configuration with complete metadata
- Improved README structure for better discoverability

### Fixed
- `method_agreement_rate` in the report now measures how often the keyword
  and structural methods chose the same domain, over classified emails. It
  previously reported the classification rate (94.77% on CEAS_08; the real
  agreement is 21.62%) (#14)

## [1.0.0] - 2024-12-22

### Added
- Initial release of Email Domain Classifier
- Dual-method classification (keyword taxonomy + structural template)
- Streaming CSV processing for large datasets
- Rich terminal UI with progress bars and tables
- Command-line interface with comprehensive options
- 10 business domain categories (Finance, Technology, Retail, etc.)
- Comprehensive logging and reporting
- JSON and text report generation
- Python package with proper CLI entry point

### Features
- Memory-efficient streaming processing
- Configurable chunk sizes for large files
- Detailed classification scores and confidence levels
- Cross-platform compatibility (Windows, macOS, Linux)
- Type hints and comprehensive documentation
- Automated testing and code quality checks

[Unreleased]: https://github.com/luongnv89/email-classifier/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/luongnv89/email-classifier/releases/tag/v1.0.0
