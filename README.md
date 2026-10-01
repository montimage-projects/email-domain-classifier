# Email Domain Classifier

[![CI Status](https://github.com/montimage/email-domain-classifier/workflows/CI/badge.svg)](https://github.com/montimage/email-domain-classifier/actions)
[![codecov](https://codecov.io/gh/montimage/email-domain-classifier/graph/badge.svg?token=YOUR_TOKEN)](https://codecov.io/gh/montimage/email-domain-classifier)
[![PyPI version](https://badge.fury.io/py/email-domain-classifier.svg)](https://badge.fury.io/py/email-domain-classifier)
[![Python versions](https://img.shields.io/pypi/pyversions/email-domain-classifier.svg)](https://pypi.org/project/email-domain-classifier/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

A Python library for classifying emails by domain using dual-method validation with optional LLM-based semantic classification. Designed for processing large datasets efficiently with streaming processing.

## Screenshots

### Dataset Analysis
Analyze your dataset before classification to understand label distribution, body lengths, and data quality:

![Dataset Analysis](screenshots/email-cli-info.png)

### Classification in Progress
Real-time progress tracking with configuration display:

![Classification Progress](screenshots/classification-in-progress.png)

### Classification Results
Domain distribution table showing classification breakdown:

![Classification Table](screenshots/classification-table.png)

### Processing Summary
Detailed results with validation stats, performance metrics, and output files:

![Results Overview](screenshots/overview-result.png)

## Quick Start

```bash
# Install
git clone git@github.com:montimage/email-domain-classifier.git && cd email-domain-classifier
python -m venv .venv && source .venv/bin/activate
pip install -e .

# Analyze dataset
email-cli info sample_emails.csv

# Classify emails
email-cli sample_emails.csv -o output/

# Classify with LLM (requires additional setup - see LLM Classification section)
email-cli sample_emails.csv -o output/ --use-llm
```

## Supported Domains

| Domain | Description |
|--------|-------------|
| Finance | Banking, payments, financial services |
| Technology | Software, hardware, IT services |
| Retail | E-commerce, shopping, consumer goods |
| Logistics | Shipping, supply chain, transportation |
| Healthcare | Medical services, health insurance |
| Government | Public sector, regulatory agencies |
| HR | Human resources, recruitment |
| Telecommunications | Phone, internet, communication services |
| Social Media | Social platforms, networking services |
| Education | Schools, universities, learning platforms |

## Module Structure

| Module | Description |
|--------|-------------|
| `email_classifier/analyzer.py` | Dataset analysis (info command) |
| `email_classifier/classifier.py` | Core classification logic |
| `email_classifier/cli.py` | Command-line interface |
| `email_classifier/domains.py` | Domain definitions and profiles |
| `email_classifier/llm/` | LLM-based classification (optional) |
| `email_classifier/processor.py` | CSV streaming processor |
| `email_classifier/reporter.py` | Report generation |
| `email_classifier/ui.py` | Terminal UI components |
| `email_classifier/validator.py` | Email validation |
| `tests/` | Test suite |
| `docs/` | Documentation |
| `raw-data/` | Sample input datasets (Git LFS) |
| `classified-data/` | Example classification outputs (Git LFS) |

## Example Data

The repository includes sample datasets for testing and reference:

**Input** (`raw-data/`):
- `CEAS_08.csv` - CEAS 2008 email dataset (~39K emails, 68MB)
- `sample_emails.csv` - Small sample for quick testing (100 emails)

**Output** (`classified-data/ceas_08/`):
- `email_[domain].csv` - Emails classified by domain (finance, technology, retail, etc.)
- `email_unsure.csv` - Emails that couldn't be confidently classified
- `invalid_emails.csv` - Emails that failed validation
- `skipped_emails.csv` - Emails filtered by body length
- `classification_report.json` - Detailed statistics
- `classification_report.txt` - Human-readable summary

```bash
# Try with sample data
email-cli info raw-data/sample_emails.csv
email-cli raw-data/sample_emails.csv -o output/

# Or use the full CEAS dataset
email-cli raw-data/CEAS_08.csv -o classified-data/my_output/
```

> **Note**: Large CSV files are stored with Git LFS. Run `git lfs pull` after cloning to download them.

## LLM Classification (Optional)

The classifier supports an optional LLM-based Method 3 that uses semantic analysis for improved classification accuracy. This method complements the existing keyword taxonomy and structural template methods.

### Installation

Install with LLM support for your preferred provider:

```bash
# For Ollama (local, free, no API key required)
pip install -e ".[ollama]"

# For Google Gemini
pip install -e ".[google]"

# For Mistral AI
pip install -e ".[mistral]"

# For Groq (fast inference)
pip install -e ".[groq]"

# For OpenRouter (access to multiple models)
pip install -e ".[openrouter]"

# For TypeSafe (one Choice question over the domains; no LangChain needed)
pip install -e ".[typesafe]"

# Install all providers
pip install -e ".[all-llm]"
```

### Configuration

1. Copy the example configuration:
   ```bash
   cp .env.example .env
   ```

2. Edit `.env` with your settings:
   ```bash
   # For Ollama (local)
   LLM_PROVIDER=ollama
   LLM_MODEL=llama3.2

   # For Google Gemini
   LLM_PROVIDER=google
   GOOGLE_API_KEY=your-api-key

   # For TypeSafe (model defaults to jev-latest)
   LLM_PROVIDER=typesafe
   TYPESAFE_API_KEY=your-api-key

   # For other providers, set the appropriate API key
   ```

With `LLM_PROVIDER=typesafe`, Method 3 is the `TypeSafeClassifier`: it asks
TypeSafe one Choice question over the ten domains plus `none`, using the option
text from the [domain definition](docs/design/domain-profiles.md#domain-definition).
The chosen option becomes the domain (`none` gives no domain), the Choice
probabilities of the ten domains become the scores, and TypeSafe's own
confidence becomes the confidence. Every other provider keeps the LangChain
`LLMClassifier`.

### Usage

```bash
# Enable LLM classification (hybrid mode - default)
email-cli sample_emails.csv -o output/ --use-llm

# Force LLM for every email (original three-method behavior)
email-cli sample_emails.csv -o output/ --use-llm --force-llm
```

### Hybrid Workflow (Default with `--use-llm`)

When you enable LLM classification with `--use-llm`, the classifier uses a **hybrid workflow** by default:

1. **Run both classic classifiers** on each email:
   - Method 1: Keyword Taxonomy Matching
   - Method 2: Structural Template Matching

2. **Check for agreement**:
   - If both classifiers agree on the domain → accept that result and **skip the LLM** (saves API costs/time)
   - If they disagree → invoke the LLM to determine the final classification

3. **Gate the LLM answer on its confidence**:
   - If the LLM's confidence is at least the cutoff → accept its domain (a confident "no listed domain fits" answer gives `unsure`)
   - If it is below the cutoff, or the LLM call failed → ignore the LLM answer and use the classic weighted fallback (60% keyword, 40% structural; `unsure` if that is below its own threshold)

   Set the cutoff (0.0 to 1.0) with `LLM_CONFIDENCE_CUTOFF` in `.env` or `--llm-confidence-cutoff`. **The default is 0.5.** Issue #19 evaluated it on the 180-email labelled CEAS_08 set ([evaluation](docs/evaluation/typesafe-evaluation.md)) and kept it: no swept cutoff was significantly better (exact McNemar p=0.125 for the best one, 0.30). It should be re-evaluated once humans have verified the labels. Only TypeSafe's confidence was evaluated; the self-reported confidence of the LangChain providers was not. `0.0` accepts every answer except a failed call. The gate applies to every provider: with `typesafe` the confidence comes from TypeSafe's answer probabilities; with the LangChain providers it is the confidence the model reports about itself, which is not calibrated. Each gated email's details record the decision under `llm_gate`, and `hybrid_workflow.jsonl` records it in the `llm_classify` step.

This hybrid approach significantly reduces LLM API calls while maintaining classification accuracy. In typical datasets, 60-80% of emails can be classified without LLM involvement.

#### Workflow Modes

| Command | Mode | Description |
|---------|------|-------------|
| `email-cli data.csv -o out/` | Dual-method | Keyword + Structural only (no LLM) |
| `email-cli data.csv -o out/ --use-llm` | **Hybrid** | LLM only when classifiers disagree |
| `email-cli data.csv -o out/ --use-llm --force-llm` | Force LLM | LLM for every email (three-method) |

#### Hybrid Workflow Output

When using hybrid mode, additional output is generated:

- **`hybrid_workflow.jsonl`** - Structured JSON Lines log of every workflow step:
  ```json
  {"timestamp": "2024-01-15T10:30:00", "email_idx": 0, "step": "keyword_classify", "result": "finance"}
  {"timestamp": "2024-01-15T10:30:00", "email_idx": 0, "step": "structural_classify", "result": "finance"}
  {"timestamp": "2024-01-15T10:30:00", "email_idx": 0, "step": "agreement_check", "path": "classic_only", "result": "finance"}
  ```

- **Hybrid statistics in reports**:
  ```
  HYBRID WORKFLOW STATISTICS
  ────────────────────────────────────────
  Total Processed (Hybrid): 1,000
  Classic Agreement Count:  750
  Agreement Rate:           75.0%
  LLM Calls Made:           250
  LLM Savings:              75.0%
  LLM Avg Response Time:    1,234ms
  ```

#### Real-time Status Bar

During hybrid processing, a real-time status bar shows the current step:
```
⠋ Classifying with Keyword Taxonomy...
⠋ Classifying with Structural Template...
⠋ Classifiers agree - accepting 'finance'
⠋ Classifiers disagree - invoking LLM...
⠋ LLM responded in 1234ms - classified as 'technology'
```

### Supported Providers

| Provider | Package | Default Model | API Key Required |
|----------|---------|---------------|------------------|
| Ollama | `langchain-ollama` | `llama3.2` | No (local) |
| Google | `langchain-google-genai` | `gemini-2.0-flash` | Yes |
| Mistral | `langchain-mistralai` | `mistral-large-latest` | Yes |
| Groq | `langchain-groq` | `llama-3.3-70b-versatile` | Yes |
| OpenRouter | `langchain-openai` | (specify) | Yes |

### Method Weights

When LLM is enabled, the classification weights are:
- **Method 1** (Keyword Taxonomy): 35%
- **Method 2** (Structural Template): 25%
- **Method 3** (LLM): 40%

Weights can be customized in the `.env` file:
```bash
LLM_WEIGHT=0.40
KEYWORD_WEIGHT=0.35
STRUCTURAL_WEIGHT=0.25
```

In hybrid mode the minimum LLM confidence accepted is set with:
```bash
LLM_CONFIDENCE_CUTOFF=0.5   # default, evaluated in #19 (docs/evaluation/typesafe-evaluation.md)
```

### Migration Note

**Breaking change for `--use-llm` users**: Previously, `--use-llm` invoked the LLM for every email. Now it uses hybrid mode by default (LLM only on classifier disagreement). To restore the previous behavior, add `--force-llm`:

```bash
# Old behavior (LLM for every email)
email-cli data.csv -o output/ --use-llm --force-llm

# New default (hybrid mode)
email-cli data.csv -o output/ --use-llm
```

## Documentation

Full documentation is available in [`docs/`](docs/index.md):

- [Installation Guide](docs/integration/installation.md)
- [User Guide](docs/user-guide/)
- [API Reference](docs/api/)
- [Architecture](docs/architecture/)
- [Development Playbook](docs/playbooks/development-playbook.md)
- [Deployment Playbook](docs/playbooks/deployment-playbook.md)
- [Troubleshooting](docs/troubleshooting/)

## License

Apache License 2.0 - see [LICENSE](LICENSE) file for details.

## Contact

Built by [Montimage Security Research](https://www.montimage.eu/)

- **GitHub**: [montimage/email-domain-classifier](https://github.com/montimage/email-domain-classifier)
- **Issues**: [Issue Tracker](https://github.com/montimage/email-domain-classifier/issues)
- **Email**: developer@montimage.com
