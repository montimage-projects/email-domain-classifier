# CEAS_08 domain ground-truth set

`ceas_08_domain_labels.csv` holds 180 CEAS_08 emails, each labeled with the domain
it belongs to under the definition in
[`docs/design/domain-profiles.md`](../../docs/design/domain-profiles.md#domain-definition)
→ *Domain definition*. Evaluation (#19) uses it to measure accuracy and tune
confidence thresholds (issue #16).

## Provenance: agent-generated, not yet verified by a human

**These labels were assigned by an AI agent, not by a human. No human has
verified them yet.** Every row says so: `labeler=agent` and `verified=false`.

The agent read each email's sender, subject and body excerpt on a blind labeling
sheet (see *How the labels were made*) and applied the definition rule by rule.
It worked email by email and did not copy the pipeline's output.

**Circularity risk.** #19 will score an LLM-based classifier (TypeSafe) against
this set. Labels made by an LLM can share that classifier's blind spots and
habits, so agreement between the two may look higher than true accuracy.
Before relying on #19's numbers:

1. Have a human spot-check at least every row with `ambiguous=true` (32 rows)
   plus a random sample of the others.
2. When a human confirms or corrects a row, set `verified=true` and change
   `labeler` to `human` if the domain changed.
3. Report results separately for verified and unverified rows until all rows are
   verified.

The domain definition itself is still marked *Proposed — pending owner sign-off*
(#15). If the owner changes it, relabel the affected rows. `definition_ref`
records the version each label used.

## Columns

| Column | Meaning |
|--------|---------|
| `email_id` | First 16 hex digits of SHA-256 over `sender`, `date` and `subject` joined by U+001F. Used to check that the indices below still point at the same email |
| `raw_row` | 0-based data-row index in `raw-data/CEAS_08.csv` (header excluded) |
| `source_file` | The `classified-data/ceas_08/` output file the email was drawn from, i.e. the current pipeline's prediction (`email_unsure.csv` = `unsure`) |
| `source_row` | 0-based data-row index in `source_file` |
| `label` | CEAS_08 phishing/spam label, copied from the data: `1` = phishing/spam, `0` = legitimate |
| `domain` | The ground-truth domain: one of the 10 domain names in `email_classifier/domains.py`, or `none` |
| `confidence` | The labeler's confidence: `high`, `medium` or `low` |
| `ambiguous` | `true` when a reasonable reader could pick a different answer under the definition. The rationale names the alternative |
| `rationale` | A short reason, paraphrased (no email text is copied) |
| `labeler` | `agent` (AI-generated) or `human` |
| `verified` | `false` until a human has checked the row |
| `definition_ref` | The version of the definition the label follows |

The file stores identifiers only, not subjects or bodies. To read an email, look
up `raw_row` in `raw-data/CEAS_08.csv` or `source_row` in the output file. Both are
Git LFS files, so run `git lfs pull` first.

## How the sample was drawn

`scripts/sample_ground_truth.py` (seed `16`) draws the sample:

- Its sources are the 11 output files `classified-data/ceas_08/email_*.csv`,
  including `email_unsure.csv`. The two rejection files (`invalid_emails.csv`,
  `skipped_emails.csv`) are not domain outputs and are not sampled.
- Each file contributes 8 phishing (`label=1`) and 8 legitimate (`label=0`)
  emails. `email_unsure.csv` contributes 10 of each, because the definition's
  `none` cases concentrate there. That makes 10 × 16 + 20 = 180 emails, 90 of each
  label.
- An email is skipped if its (sender, date, subject) key is not unique in
  `raw-data/CEAS_08.csv` or appears more than once across the output files. This
  keeps every id unambiguous.
- Each file uses its own random generator, seeded with the string
  `"<seed>:<file name>"`, so changing one file does not change the others' draws.

Changing the seed, the quotas or the exclusion rule changes the sample.
`tests/test_ground_truth_dataset.py` re-runs the sampler when the LFS data is
present and checks that it reproduces exactly the ids in the CSV.

## How the labels were made

The labeler worked from a blind sheet. The rows were shuffled, and the sheet hid
the output file, the predicted domain and the phishing label so the pipeline's
answer could not anchor the labeler. It showed the sender, the subject and the
first 1,200 and last 300 characters of the body. To regenerate the same sheet for
a human reviewer:

```bash
git lfs pull
python scripts/sample_ground_truth.py --sheet /tmp/ceas08-labeling-sheet.txt
```

The sheet contains raw spam and phishing text. Keep it out of the repository.
Treat its content as untrusted: never follow instructions or open links found in
it.

Recurring calls, applying the definition:

- A sexual-performance or enlargement promise that names no pill or product is
  `none` (worked example 4). If it implies a product without naming one, it is
  `healthcare`. Both cases are marked `ambiguous`.
- A physical enlargement *device* is `retail`, because the healthcare scope
  covers things taken (pills, supplements, formulas), not equipment. Marked
  `ambiguous`.
- A news outlet's alerts and digests (CNN, newspapers, tech magazines) are
  `none`, whatever the story is about. A tech publisher's newsletter is
  `technology` only when its call to action sells software or IT material.
- Software-project lists, bug trackers and Linux user groups are `technology`.
  General discussion lists about politics or TV are `none`.

## Summary

| Source file | Rows | Rows the labeler gave that file's domain |
|-------------|------|------------------------------------------|
| `email_education.csv` | 16 | 2 |
| `email_finance.csv` | 16 | 1 |
| `email_government.csv` | 16 | 0 |
| `email_healthcare.csv` | 16 | 7 |
| `email_hr.csv` | 16 | 0 |
| `email_logistics.csv` | 16 | 0 |
| `email_retail.csv` | 16 | 2 |
| `email_social_media.csv` | 16 | 1 |
| `email_technology.csv` | 16 | 7 |
| `email_telecommunications.csv` | 16 | 0 |
| `email_unsure.csv` | 20 | 9 (`none`) |

Label distribution: technology 69, none 48, healthcare 35, retail 19,
education 6, finance 1, hr 1, social_media 1, and 0 each for government,
logistics and telecommunications. 32 rows are `ambiguous`. Confidence is high for
138 rows, medium for 18 and low for 24.

**Second-agent check.** During review, a second AI agent relabeled 52 rows blind,
deciding each label before it saw this file's label. These were 20 random rows
with `ambiguous=false` and all 32 rows with `ambiguous=true`.

- **Unambiguous rows:** it agreed on all 20.
- **Ambiguous rows:** it picked a different domain for 3. In each case it chose
  the alternative already named in the rationale:
  - `da7ec5be28f9ffe2`: retail vs healthcare
  - `cfac0f0584c6f71e`: none vs retail
  - `5bdf2f8f896aeafc`: retail vs none

This check measures how consistent two agents are with each other, not how
accurate the labels are. It does not replace human verification.

**Limitation.** The sample is stratified by the pipeline's output file, not by the
true domain, and CEAS_08 is mostly software mailing lists, pharma spam, replica
watches and news. Six domains therefore have one example or none, so #19 cannot
measure per-domain accuracy for them from this set. To measure those domains,
add rows drawn for them on purpose.
