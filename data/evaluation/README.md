# Classifier evaluation outputs (#19)

`scripts/evaluate_classifiers.py` writes these files. They hold email ids, model
answers, call statistics and metrics only. They never hold email text: no sender,
receiver, subject, body or URLs. The cache writer enforces this with an allow-list
of record keys and value shapes.

| File | Written by | Contents |
|------|------------|----------|
| `typesafe_outputs.jsonl` | `collect` | One record per TypeSafe call: `email_id`, `set` (`labeled` or `sample`), `choice`, `probabilities`, `confidence`, `domain`, `fallback`, `error_type` (an exception class name or HTTP code, never a message), token `usage`, `latency_ms`, `attempts`, `http_statuses`, `model`, `timestamp` |
| `typesafe_runs.json` | `collect` | One summary per run: set, start and end times, wall time, emails, HTTP attempts (retries included), 429 responses, errors, `--rps`, `--workers`, achieved emails/s and attempts/s |
| `typesafe_evaluation_results.json` | `report` | Every metric, the cutoff sweep, the cutoff choice, the token and runtime statistics, and per-row predictions keyed by `email_id` |

`email_id` is the id used in `data/ground-truth/ceas_08_domain_labels.csv`: the
first 16 hex digits of SHA-256 over sender, date and subject.

## Regenerate

Both commands need the Git LFS data (`git lfs pull`).

```bash
# Live: needs the TypeSafe SDK (pip install -e ".[typesafe]") and a key.
# The key is read from the environment only.
export TYPESAFE_API_KEY=...
python scripts/evaluate_classifiers.py collect --set labeled
python scripts/evaluate_classifiers.py collect --set sample --sample-size 100

# Offline: no key, no network.
python scripts/evaluate_classifiers.py report --md-out /tmp/evaluation.md
```

`collect` resumes: it skips every email that already has a successful record in
the cache. `--rps` (default 10, at most 40) limits HTTP attempts per second and
`--max-calls` (default 600) caps the attempts of one run, retries included.
