# Installation Guide

This guide will help you install the Email Domain Classifier on your system.

## AI-Agent Automated Setup

If you're using an AI coding assistant (Claude Code, Cursor, etc.), you can request automated setup. The AI-Agent will:

1. Detect and validate system prerequisites (Python 3.10+, Git, pip)
2. Create and activate a virtual environment
3. Install the package and dependencies
4. Configure LLM provider (Ollama with llama3.1:8b by default)
5. Generate the `.env` configuration file
6. Present a summary for you to verify

**Default configuration:**
- Setup type: Basic usage
- LLM: Enabled with Ollama (local, no API key needed)
- Model: llama3.1:8b

Simply ask: *"Set up this project from scratch"* or *"Install and configure email-classifier"*

---

## Prerequisites

- **Python 3.10 or higher** - Check with `python --version`
- **pip** - Package installer (comes with Python)
- **git** - Version control (for cloning the repository)

## Method 1: Install from PyPI (Recommended)

```bash
# Basic installation
pip install email-domain-classifier

# With development tools
pip install email-domain-classifier[dev]
```

## Method 2: Install from Source

### Clone the Repository

```bash
# Clone the repository
git clone https://github.com/montimage/email-domain-classifier.git

# Navigate into the project directory
cd email-domain-classifier
```

### Set Up Virtual Environment (Recommended)

```bash
# Create a virtual environment
python -m venv .venv

# Activate the virtual environment
# On macOS/Linux:
source .venv/bin/activate
# On Windows:
.venv\Scripts\activate
```

### Install the Package

```bash
# Install in development mode
pip install -e .

# Or for development with extra tools:
pip install -e ".[dev]"
```

## Verify Installation

### Check CLI Installation

```bash
# Check version
email-cli --version

# List supported domains
email-cli --list-domains

# Show help
email-cli --help
```

### Check Python Installation

```bash
# Test Python import
python -c "from email_classifier import EmailClassifier; print('✅ Import successful')"
```

## Method 3: Development Installation

For contributors who want to work on the codebase:

```bash
# Clone and navigate to project
git clone https://github.com/montimage/email-domain-classifier.git
cd email-domain-classifier

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install in development mode with all tools
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

## Dependencies

### Required Dependencies

- **rich >= 13.0.0** - Terminal UI components for progress bars and tables

### Development Dependencies

- **pytest >= 7.0.0** - Testing framework
- **pytest-cov >= 4.0.0** - Test coverage reporting
- **black >= 23.0.0** - Code formatting
- **isort >= 5.0.0** - Import sorting
- **mypy >= 1.0.0** - Static type checking

## Platform Support

Email Domain Classifier supports:

- ✅ **Linux** (Ubuntu, CentOS, Debian, etc.)
- ✅ **macOS** (Intel and Apple Silicon)
- ✅ **Windows** (Windows 10 and 11)

## Troubleshooting

### Common Installation Issues

#### "Python not found" or Version Issues

```bash
# Check Python version
python --version
python3 --version

# On macOS, install Python 3.10+:
brew install python@3.11

# On Ubuntu/Debian:
sudo apt update
sudo apt install python3.11 python3.11-venv

# On Windows: Download from python.org
```

#### "pip command not found"

```bash
# Ensure pip is installed and up to date
python -m pip install --upgrade pip

# Use python -m pip if pip is not in PATH
python -m pip install -e .
```

#### "Permission denied" during installation

```bash
# Use user directory installation
pip install --user -e .

# Or fix permissions (Unix/Linux)
sudo chown -R $USER:.local
```

#### Import errors after installation

```bash
# Reinstall in development mode
pip uninstall email-domain-classifier
pip install -e .

# Verify package location
pip show email-domain-classifier
```

### Virtual Environment Issues

#### Activation fails on Windows

```cmd
# Try the full path
C:\path\to\project\.venv\Scripts\activate

# If using PowerShell
.\.venv\Scripts\Activate.ps1
```

#### Python not found after activation

```bash
# Ensure venv was created with correct Python
python3.11 -m venv .venv  # Use specific version
```

### Dependency Issues

#### Rich installation fails

```bash
# Update pip and setuptools
python -m pip install --upgrade pip setuptools

# Try installing rich separately
pip install rich>=13.0.0
pip install -e .
```

#### Development tools installation issues

```bash
# Install development dependencies manually
pip install pytest pytest-cov black isort mypy

# Or install from requirements.txt if available
pip install -r requirements-dev.txt
```

## Next Steps

After successful installation:

1. **Try the basic example**: `email-cli sample_emails.csv -o output/`
2. **Read the [User Guide](user-guide/index)** for comprehensive usage
3. **Explore the [API Documentation](api/index)** for Python integration
4. **Check the [Examples](user-guide/examples)** for practical use cases

## Getting Help

If you encounter issues not covered here:

1. **Check the [Troubleshooting Guide](user-guide/troubleshooting)**
2. **Search existing [GitHub Issues](https://github.com/montimage/email-domain-classifier/issues)**
3. **Create a new issue** with detailed information
4. **Ask in [GitHub Discussions](https://github.com/montimage/email-domain-classifier/discussions)**

## Uninstallation

If you need to uninstall the package:

```bash
# Uninstall the package
pip uninstall email-domain-classifier

# Remove virtual environment (if created)
rm -rf .venv
```
