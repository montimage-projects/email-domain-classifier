# TypeSafe classifier evaluation (#19)

This page reports how `TypeSafeClassifier` (#17) and the hybrid workflow's
confidence gate (#18) perform on the 180-email CEAS_08 ground-truth set (#16). It
compares them with the classic methods and explains how the cutoff was chosen.
Every number on this page comes from
[`data/evaluation/typesafe_evaluation_results.json`](../../data/evaluation/typesafe_evaluation_results.json).
`scripts/evaluate_classifiers.py report` produced that file offline. The tables
below are copied from its markdown output.

## Summary

- **TypeSafe alone:** 136/180 correct, strict accuracy 0.756 [0.688, 0.813].
- **Classic combined classifier:** 29/180 (0.161 [0.115, 0.222]).
- **Method 1 (keywords):** 25/180.
- **Method 2 (structure):** 38/180.
- **Hybrid at the current cutoff 0.50:** 126/180 (0.700 [0.629, 0.762]).
- **Best hybrid result in the sweep:** 131/180, at every cutoff from 0.00 to 0.30.
- **Decision:** the cutoff stays at **0.50**. Under the pre-registered rule the
  candidate was 0.30, with 131 correct rows against 126 at 0.50. The exact McNemar
  test gave b=6, c=1, p=0.125 ≥ 0.05, so this data cannot tell the two cutoffs
  apart.
- **The hybrid scores below TypeSafe alone at every cutoff:** at best 131 against
  136. See [Why the hybrid scores below TypeSafe alone](#why-the-hybrid-scores-below-typesafe-alone).
- **TypeSafe's main error:** it answers `none` for emails labeled `technology`. This
  happens on 27 of its 44 errors, and all 27 are legitimate (label 0) emails:
  every one of the 69 `technology` rows in the labeled set is legitimate.
- **Live run:** 380 TypeSafe calls (180 labeled + 200 sample emails) on
  2026-10-01, with no errors, no retries and no HTTP 429 responses.

> **Limitations — read before using these numbers**
>
> 1. **The labels were made by an AI agent and no human has checked them.** All 180
>    rows have `labeler=agent` and `verified=false`. An LLM scored against labels
>    an LLM made can agree more often than it is right (circularity), so TypeSafe's
>    accuracy may be inflated. See
>    [`data/ground-truth/README.md`](../../data/ground-truth/README.md).
> 2. **Six domains have one labeled row or none,** so this page makes no claim
>    about them: government, logistics and telecommunications have 0 rows;
>    finance, hr and social_media have 1. Per-class rows below 5 are marked
>    "insufficient support".
> 3. **The set is small (n=180).** Every rate has a Wilson 95% interval. Paired
>    comparisons between cutoffs use the exact McNemar test.
> 4. **Raw accuracy is not a population estimate.** The sample is stratified by
>    the classic pipeline's own predicted output file × CEAS_08 label. The
>    "Weighted acc." column reweights each (file, label) stratum to its size in
>    `classified-data/ceas_08/`. Rows the sampler could never draw (non-unique
>    keys) still count in those sizes.
> 5. **Re-running the classic classifier does not exactly reproduce the
>    pipeline's output file.** It matches `source_file` for 173/180 rows. That
>    count changes with Python's string-hash seed (173–180 over seeds 0–11),
>    because the classic methods break score ties in set order. The headline
>    accuracies did not change over those seeds. The report pins
>    `PYTHONHASHSEED=0`.
> 6. **The model may change.** `jev-latest` resolved to `jev-1.13.0` on
>    2026-10-01, and collecting again later may give different answers. The
>    reproducible artifact is the committed cache,
>    [`data/evaluation/typesafe_outputs.jsonl`](../../data/evaluation/typesafe_outputs.jsonl).
> 7. **Spam:** TypeSafe scored 0.844 on spam rows and 0.667 on legitimate rows.
>    This shows no degradation on CEAS bulk spam. It does **not** show robustness
>    to adversarial content: these are bulk spam emails, not targeted prompt
>    injection.
> 8. **Known bug, out of scope here:** the raw CEAS `urls` column holds the
>    string `'0'` or `'1'`, and `EmailData.has_url` treats `'0'` as true. Method 2's
>    URL feature is therefore true on every raw row. Fixing it needs a follow-up
>    issue, and it would change the classic numbers.

## Method

**Data.**
- `data/ground-truth/ceas_08_domain_labels.csv`: 180 rows, 90 spam/phishing
  (label 1) and 90 legitimate (label 0); 32 are marked ambiguous.
- Each row is looked up in `raw-data/CEAS_08.csv` by `raw_row`, and its
  `email_id` is checked.
- A second set of 200 rows was drawn at random from the rest of CEAS_08 (seed 19).
  It is used for token and runtime statistics only.

**Emails.** Each email is built exactly as the pipeline builds it:
`EmailData.from_dict(StreamingProcessor()._normalize_row(row))`.

**Systems.**
- Method 1 (`KeywordTaxonomyClassifier`), Method 2
  (`StructuralTemplateClassifier`) and the classic combined `EmailClassifier`
  are re-run offline.
- TypeSafe answers come from the shipped `TypeSafeClassifier.classify()`,
  including its prompt, its 2,000-character body trim and its state. They are
  cached once in `typesafe_outputs.jsonl`.
- The hybrid is the real `HybridClassifier`. Its LLM is replaced by a stub that
  returns the cached answer as a `ClassificationResult`, so the replay runs the
  real agreement check, confidence gate and weighted fallback.

**Scoring.**
- Following `docs/design/domain-profiles.md`, a `None` or `unsure` prediction
  counts as `none`.
- A failed TypeSafe call is an error, is not scored, and is counted separately.
  There were no failed calls.
- "Answer rate" and "Accuracy when answering" consider only the rows where a
  system predicted a domain other than `none`.
- Macro-F1 averages the classes with at least 5 labeled rows.

**Cutoff sweep.**
- The cutoff takes the values 0.00, 0.05, … 0.95, and the full hybrid is
  replayed at each one.
- The gate fires on *disagreement rows*: rows where Methods 1 and 2 disagree or
  either one abstains.

**Pre-registered cutoff rule**, fixed before the live run:
1. The primary metric is overall hybrid strict accuracy.
2. Among the cutoffs with the most correct rows, take the one closest to the
   incumbent 0.50 (the lower one on a tie).
3. Adopt it only if an exact two-sided McNemar test on paired per-row
   correctness against the incumbent gives p < 0.05. Otherwise keep the
   incumbent.

**Live collection.**
- The API key was read from `TYPESAFE_API_KEY` only.
- Each HTTP attempt, retries included, was counted, rate limited and charged to a
  call budget:
  - labeled set: `--rps 10 --workers 4 --max-calls 260`;
  - sample set: `--rps 20 --workers 8 --max-calls 250`.
- Result: 380 HTTP attempts, all answered with status 200. There were 0 errors,
  0 retries and 0 HTTP 429 responses.

## Results

Labeled rows: 180. None and 'unsure' count as a 'none' prediction; failed TypeSafe calls are errors and are not scored.
Intervals are Wilson 95%.

Re-running the classic classifier reproduces the pipeline's output file (`source_file`) for 173/180 rows (PYTHONHASHSEED=0).

TypeSafe model requested: jev-latest; resolved (records): jev-1.13.0 (380); collected on 2026-10-01.

### Systems

| System | Scored | Errors | Missing | Strict accuracy | Answer rate | Accuracy when answering | none P / R | Macro-F1 | Weighted acc. |
|---|---|---|---|---|---|---|---|---|---|
| method1 | 180 | 0 | 0 | 0.139 [0.096, 0.197] (25/180) | 0.917 [0.867, 0.949] (165/180) | 0.115 [0.075, 0.173] (19/165) | 0.400 / 0.125 | 0.187 | 0.184 |
| method2 | 180 | 0 | 0 | 0.211 [0.158, 0.276] (38/180) | 1.000 [0.979, 1.000] (180/180) | 0.211 [0.158, 0.276] (38/180) | — / 0.000 | 0.105 | 0.215 |
| classic | 180 | 0 | 0 | 0.161 [0.115, 0.222] (29/180) | 0.889 [0.835, 0.927] (160/180) | 0.125 [0.082, 0.185] (20/160) | 0.450 / 0.188 | 0.200 | 0.199 |
| typesafe | 180 | 0 | 0 | 0.756 [0.688, 0.813] (136/180) | 0.617 [0.544, 0.685] (111/180) | 0.874 [0.799, 0.923] (97/111) | 0.565 / 0.812 | 0.793 | 0.775 |
| hybrid@0.50 | 180 | 0 | 0 | 0.700 [0.629, 0.762] (126/180) | 0.672 [0.601, 0.737] (121/180) | 0.752 [0.668, 0.821] (91/121) | 0.593 / 0.729 | 0.763 | 0.675 |

Macro-F1 averages classes with support >= 5: education, healthcare, none, retail, technology.

### By CEAS_08 label and ambiguity

| System | Spam (1) | Legit (0) | Ambiguous | Not ambiguous |
|---|---|---|---|---|
| method1 | 0.144 [0.086, 0.232] (13/90) | 0.133 [0.078, 0.219] (12/90) | 0.156 [0.069, 0.318] (5/32) | 0.135 [0.089, 0.200] (20/148) |
| method2 | 0.011 [0.002, 0.060] (1/90) | 0.411 [0.315, 0.514] (37/90) | 0.031 [0.005, 0.157] (1/32) | 0.250 [0.187, 0.326] (37/148) |
| classic | 0.144 [0.086, 0.232] (13/90) | 0.178 [0.113, 0.269] (16/90) | 0.188 [0.089, 0.353] (6/32) | 0.155 [0.106, 0.222] (23/148) |
| typesafe | 0.844 [0.756, 0.905] (76/90) | 0.667 [0.564, 0.755] (60/90) | 0.500 [0.336, 0.664] (16/32) | 0.811 [0.740, 0.866] (120/148) |
| hybrid@0.50 | 0.800 [0.706, 0.870] (72/90) | 0.600 [0.497, 0.695] (54/90) | 0.500 [0.336, 0.664] (16/32) | 0.743 [0.667, 0.807] (110/148) |

### Per class

The other systems' per-class tables are in the results JSON.

#### typesafe

| Class | Precision | Recall | F1 | Support | Predicted | Note |
|---|---|---|---|---|---|---|
| education | 0.833 | 0.833 | 0.833 | 6 | 6 |  |
| finance | 0.500 | 1.000 | 0.667 | 1 | 2 | insufficient support |
| healthcare | 0.762 | 0.914 | 0.831 | 35 | 42 |  |
| hr | 0.500 | 1.000 | 0.667 | 1 | 2 | insufficient support |
| none | 0.565 | 0.812 | 0.667 | 48 | 69 |  |
| retail | 0.941 | 0.842 | 0.889 | 19 | 17 |  |
| social_media | 1.000 | 1.000 | 1.000 | 1 | 1 | insufficient support |
| technology | 1.000 | 0.594 | 0.746 | 69 | 41 |  |

#### classic

| Class | Precision | Recall | F1 | Support | Predicted | Note |
|---|---|---|---|---|---|---|
| education | 0.125 | 0.333 | 0.182 | 6 | 16 |  |
| finance | 0.059 | 1.000 | 0.111 | 1 | 17 | insufficient support |
| government | 0.000 | — | 0.000 | 0 | 15 | insufficient support |
| healthcare | 0.438 | 0.200 | 0.275 | 35 | 16 |  |
| hr | 0.000 | 0.000 | 0.000 | 1 | 16 | insufficient support |
| logistics | 0.000 | — | 0.000 | 0 | 20 | insufficient support |
| none | 0.450 | 0.188 | 0.265 | 48 | 20 |  |
| retail | 0.118 | 0.105 | 0.111 | 19 | 17 |  |
| social_media | 0.062 | 1.000 | 0.118 | 1 | 16 | insufficient support |
| technology | 0.467 | 0.101 | 0.167 | 69 | 15 |  |
| telecommunications | 0.000 | — | 0.000 | 0 | 12 | insufficient support |

#### hybrid@0.50

| Class | Precision | Recall | F1 | Support | Predicted | Note |
|---|---|---|---|---|---|---|
| education | 0.833 | 0.833 | 0.833 | 6 | 6 |  |
| finance | 0.333 | 1.000 | 0.500 | 1 | 3 | insufficient support |
| government | 0.000 | — | 0.000 | 0 | 1 | insufficient support |
| healthcare | 0.744 | 0.914 | 0.821 | 35 | 43 |  |
| hr | 0.250 | 1.000 | 0.400 | 1 | 4 | insufficient support |
| logistics | 0.000 | — | 0.000 | 0 | 2 | insufficient support |
| none | 0.593 | 0.729 | 0.654 | 48 | 59 |  |
| retail | 0.842 | 0.842 | 0.842 | 19 | 19 |  |
| social_media | 0.143 | 1.000 | 0.250 | 1 | 7 | insufficient support |
| technology | 0.972 | 0.507 | 0.667 | 69 | 36 |  |

### Disagreement and agreement rows

**Disagreement rows (where the hybrid gate fires).** Rows where Method 1 and Method 2 disagree or either abstains: 168 (0.933 of labeled rows).

- Classic weighted fallback accuracy: 0.149 [0.103, 0.210] (25/168)
- TypeSafe accuracy (scored, non-error rows): 0.756 [0.686, 0.815] (127/168)
- Classic fallback on the same scored (non-error) rows: 0.149 [0.103, 0.210] (25/168)
- TypeSafe errors: 0; missing: 0

**Agreement rows (Method 1 = Method 2; the hybrid skips the LLM).** Rows: 12. The agreed answer (what the hybrid returns) is correct on 4/12; the classic combined classifier on 4/12; TypeSafe on 9/12 scored rows.

### Cutoff sweep

"Accepted LLM acc." is accuracy on the rows whose TypeSafe answer the gate
accepted. "Fallback acc. (rejected)" is the classic fallback's accuracy on the
rows the gate rejected. "LLM acc. if used (rejected)" is how the rejected
TypeSafe answers would have scored if they had been used.

| Cutoff | Gate rows | Accepted | Acceptance | Accepted LLM acc. | Fallback acc. (rejected) | LLM acc. if used (rejected) | Hybrid overall | Spam | Legit |
|---|---|---|---|---|---|---|---|---|---|
| 0.00 | 168 | 168 | 1.000 | 0.756 [0.686, 0.815] (127/168) | — (n=0) | — (n=0) | 0.728 [0.658, 0.787] (131/180) | 0.800 [0.706, 0.870] (72/90) | 0.656 [0.553, 0.746] (59/90) |
| 0.05 | 168 | 168 | 1.000 | 0.756 [0.686, 0.815] (127/168) | — (n=0) | — (n=0) | 0.728 [0.658, 0.787] (131/180) | 0.800 [0.706, 0.870] (72/90) | 0.656 [0.553, 0.746] (59/90) |
| 0.10 | 168 | 168 | 1.000 | 0.756 [0.686, 0.815] (127/168) | — (n=0) | — (n=0) | 0.728 [0.658, 0.787] (131/180) | 0.800 [0.706, 0.870] (72/90) | 0.656 [0.553, 0.746] (59/90) |
| 0.15 | 168 | 168 | 1.000 | 0.756 [0.686, 0.815] (127/168) | — (n=0) | — (n=0) | 0.728 [0.658, 0.787] (131/180) | 0.800 [0.706, 0.870] (72/90) | 0.656 [0.553, 0.746] (59/90) |
| 0.20 | 168 | 168 | 1.000 | 0.756 [0.686, 0.815] (127/168) | — (n=0) | — (n=0) | 0.728 [0.658, 0.787] (131/180) | 0.800 [0.706, 0.870] (72/90) | 0.656 [0.553, 0.746] (59/90) |
| 0.25 | 168 | 168 | 1.000 | 0.756 [0.686, 0.815] (127/168) | — (n=0) | — (n=0) | 0.728 [0.658, 0.787] (131/180) | 0.800 [0.706, 0.870] (72/90) | 0.656 [0.553, 0.746] (59/90) |
| 0.30 | 168 | 168 | 1.000 | 0.756 [0.686, 0.815] (127/168) | — (n=0) | — (n=0) | 0.728 [0.658, 0.787] (131/180) | 0.800 [0.706, 0.870] (72/90) | 0.656 [0.553, 0.746] (59/90) |
| 0.35 | 168 | 167 | 0.994 | 0.754 [0.684, 0.814] (126/167) | 0.000 [0.000, 0.793] (0/1) | 1.000 [0.206, 1.000] (1/1) | 0.722 [0.653, 0.782] (130/180) | 0.789 [0.694, 0.861] (71/90) | 0.656 [0.553, 0.746] (59/90) |
| 0.40 | 168 | 165 | 0.982 | 0.758 [0.687, 0.817] (125/165) | 0.000 [0.000, 0.561] (0/3) | 0.667 [0.208, 0.939] (2/3) | 0.717 [0.647, 0.777] (129/180) | 0.789 [0.694, 0.861] (71/90) | 0.644 [0.541, 0.736] (58/90) |
| 0.45 | 168 | 163 | 0.970 | 0.761 [0.690, 0.820] (124/163) | 0.200 [0.036, 0.625] (1/5) | 0.600 [0.231, 0.882] (3/5) | 0.717 [0.647, 0.777] (129/180) | 0.800 [0.706, 0.870] (72/90) | 0.633 [0.530, 0.726] (57/90) |
| 0.50 | 168 | 158 | 0.941 | 0.766 [0.694, 0.825] (121/158) | 0.100 [0.018, 0.404] (1/10) | 0.600 [0.313, 0.832] (6/10) | 0.700 [0.629, 0.762] (126/180) | 0.800 [0.706, 0.870] (72/90) | 0.600 [0.497, 0.695] (54/90) |
| 0.55 | 168 | 147 | 0.875 | 0.782 [0.709, 0.841] (115/147) | 0.095 [0.026, 0.289] (2/21) | 0.571 [0.365, 0.755] (12/21) | 0.672 [0.601, 0.737] (121/180) | 0.789 [0.694, 0.861] (71/90) | 0.556 [0.453, 0.654] (50/90) |
| 0.60 | 168 | 141 | 0.839 | 0.794 [0.720, 0.853] (112/141) | 0.111 [0.038, 0.281] (3/27) | 0.556 [0.373, 0.724] (15/27) | 0.661 [0.589, 0.726] (119/180) | 0.789 [0.694, 0.861] (71/90) | 0.533 [0.431, 0.633] (48/90) |
| 0.65 | 168 | 127 | 0.756 | 0.811 [0.734, 0.870] (103/127) | 0.098 [0.039, 0.226] (4/41) | 0.585 [0.434, 0.722] (24/41) | 0.617 [0.544, 0.685] (111/180) | 0.744 [0.646, 0.823] (67/90) | 0.489 [0.388, 0.591] (44/90) |
| 0.70 | 168 | 114 | 0.679 | 0.833 [0.754, 0.891] (95/114) | 0.111 [0.052, 0.222] (6/54) | 0.593 [0.460, 0.713] (32/54) | 0.583 [0.510, 0.653] (105/180) | 0.711 [0.610, 0.795] (64/90) | 0.456 [0.357, 0.558] (41/90) |
| 0.75 | 168 | 100 | 0.595 | 0.880 [0.802, 0.930] (88/100) | 0.088 [0.041, 0.179] (6/68) | 0.574 [0.455, 0.684] (39/68) | 0.544 [0.471, 0.616] (98/180) | 0.689 [0.587, 0.775] (62/90) | 0.400 [0.305, 0.503] (36/90) |
| 0.80 | 168 | 94 | 0.559 | 0.894 [0.815, 0.941] (84/94) | 0.108 [0.056, 0.199] (8/74) | 0.581 [0.467, 0.687] (43/74) | 0.533 [0.461, 0.605] (96/180) | 0.678 [0.576, 0.765] (61/90) | 0.389 [0.295, 0.492] (35/90) |
| 0.85 | 168 | 89 | 0.530 | 0.899 [0.819, 0.946] (80/89) | 0.101 [0.052, 0.187] (8/79) | 0.595 [0.485, 0.696] (47/79) | 0.511 [0.439, 0.583] (92/180) | 0.678 [0.576, 0.765] (61/90) | 0.344 [0.255, 0.447] (31/90) |
| 0.90 | 168 | 79 | 0.470 | 0.911 [0.828, 0.956] (72/79) | 0.090 [0.046, 0.168] (8/89) | 0.618 [0.514, 0.712] (55/89) | 0.467 [0.395, 0.539] (84/180) | 0.633 [0.530, 0.726] (57/90) | 0.300 [0.215, 0.401] (27/90) |
| 0.95 | 168 | 70 | 0.417 | 0.900 [0.808, 0.951] (63/70) | 0.092 [0.049, 0.165] (9/98) | 0.653 [0.555, 0.740] (64/98) | 0.422 [0.352, 0.495] (76/180) | 0.611 [0.508, 0.705] (55/90) | 0.233 [0.158, 0.331] (21/90) |

## Cutoff decision

Candidate 0.30, chosen 0.50. Cutoff 0.30 has the most correct rows (131/180) versus 126/180 at the incumbent 0.50, but exact McNemar b=6, c=1, p=0.1250 >= 0.05: the data cannot distinguish them, so the incumbent is kept.

**The cutoff stays at 0.50.** Under the pre-registered rule, the 5-row advantage
of 0.30 is not significant on 180 rows.

The data does point toward lower cutoffs:
- At every cutoff where the gate rejects anything, the classic fallback scores
  far worse than the TypeSafe answer it replaces. At 0.50 the fallback is right
  on 1 of the 10 rejected rows, and the TypeSafe answers on those rows would
  have been right on 6.
- Overall hybrid accuracy never rises as the cutoff rises.

Re-evaluate the cutoff once humans have verified the labels (see limitation 1).
This issue does not change the design.

### Why the hybrid scores below TypeSafe alone

The hybrid scores below TypeSafe alone at every cutoff: 131/180 at best, against
136/180. The gap comes from the hybrid design, not from the cutoff.

- **Agreement shortcut:** on the 12 rows where Methods 1 and 2 agree, the hybrid
  returns their shared answer. That answer is right on 4 of them, and TypeSafe
  would be right on 9. This accounts for the whole 5-row gap at cutoffs 0.00 to
  0.30. On the gate rows the hybrid then uses every TypeSafe answer: 127/168.
- **Weighted fallback:** at 0.50, the 10 rejected rows lose 5 more correct
  answers (126/180).

Follow-up for epic #13: decide whether the agreement shortcut and the classic
fallback should stay in front of TypeSafe. This issue does not change that design.

### TypeSafe's main error

The confusion counts in the results JSON show that TypeSafe answers `none` for
emails labeled `technology` 27 times, out of 44 errors. All 27 are legitimate
(label 0) rows; in fact all 69 `technology` rows in the labeled set are label 0.
As a result, `technology` has precision 1.000 but recall 0.594, and TypeSafe
predicts `none` 69 times against a support of 48. This
one confusion explains most of the gap between spam accuracy (0.844) and
legitimate accuracy (0.667). Before changing the option text, check these rows
against the definition's rule for software projects' mailing lists.

## Tokens and runtime

| Set | Records | Errors | Input tokens mean / median / p95 / min / max | Output tokens mean | Latency p50 / p95 ms | Attempts per email | 429s |
|---|---|---|---|---|---|---|---|
| labeled | 180 | 0 | 1523.8 / 1435.0 / 2043.6 / 1300 / 2924 | 103.6 | 226 / 284 | 1.000 | 0 |
| sample | 200 | 0 | 1543.7 / 1423.5 / 2062.1 / 1295 / 2077 | 103.6 | 229 / 289 | 1.000 | 0 |
| overall | 380 | 0 | 1534.2 / 1430.0 / 2061.1 / 1295 / 2924 | 103.6 | 227 / 288 | 1.000 | 0 |

| Run set | Status | Emails | Attempts | 429s | Errors | rps | Workers | Emails/s | Attempts/s |
|---|---|---|---|---|---|---|---|---|---|
| labeled | complete | 180 | 180 | 0 | 0 | 10.0 | 4 | 9.75 | 9.75 |
| sample | complete | 200 | 200 | 0 | 0 | 20.0 | 8 | 19.12 | 19.12 |

Extrapolation to 35,000 emails at 1.000 attempts per email, at each run's measured rate (labeled @ 10 rps, 4 workers: 9.75 attempts/s; sample @ 20 rps, 8 workers: 19.12 attempts/s) and at the 40 rps ceiling. Sustaining 40 rps needs about 10 concurrent requests at the p50 latency and 12 at the p95 latency.

Measured throughput tracked the --rps throttle, with 0 HTTP 429 responses, up to 20 rps. 40 rps itself was not exercised, so the ceiling row is a bound, not a measurement.

| Scenario | LLM emails | Hours at labeled @ 10 rps, 4 workers | Hours at sample @ 20 rps, 8 workers | Hours at 40 rps | Input tokens | Output tokens |
|---|---|---|---|---|---|---|
| TypeSafe on every email | 35,000 | 1.00 | 0.51 | 0.24 | 53,698,658 | 3,625,079 |
| Hybrid, labeled-set disagreement (0.933) | 32,667 | 0.93 | 0.47 | 0.23 | 50,118,747 | 3,383,407 |
| Hybrid, full-corpus agreement 21.62% | 27,433 | 0.78 | 0.40 | 0.19 | 42,089,008 | 2,841,337 |

The labeled set is a stratified sample, so its disagreement fraction is not a population estimate; the full-corpus agreement rate comes from the reporter's run over all of CEAS_08.

## Reproduce

The live step needs the TypeSafe SDK (`pip install -e ".[typesafe]"`) and a key
in the environment. Never put the key on the command line or in a file in the
repository. The report step is offline and needs only the Git LFS data
(`git lfs pull`).

```bash
export TYPESAFE_API_KEY=...   # from your secret store
python scripts/evaluate_classifiers.py collect --set labeled --rps 10 --workers 4 --max-calls 260
python scripts/evaluate_classifiers.py collect --set sample --sample-size 200 --rps 20 --workers 8 --max-calls 250
python scripts/evaluate_classifiers.py report --md-out /tmp/typesafe-evaluation.md
```

- `collect` skips every email that is already cached. With the committed cache it
  therefore makes no calls.
- `report` re-runs itself with `PYTHONHASHSEED=0` and writes
  `data/evaluation/typesafe_evaluation_results.json`.
- These numbers come from the script at commit `c13b9bd`. The answers were
  collected on 2026-10-01.
