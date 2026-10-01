#!/usr/bin/env python3
"""Draw the reproducible CEAS_08 sample for the domain ground-truth set (#16).

The sample is stratified over the 11 ``email_*.csv`` output files that the
pipeline wrote to ``classified-data/ceas_08/`` and, inside each file, over the
CEAS_08 ``label`` column (1 = phishing/spam, 0 = legitimate). Every draw uses a
fixed seed, so the same inputs always give the same sample.

Each sampled email is identified by:

* ``raw_row``: its 0-based data-row index in ``raw-data/CEAS_08.csv``;
* ``source_file`` / ``source_row``: the output file it was drawn from and its
  0-based data-row index there;
* ``email_id``: the first 16 hex digits of SHA-256 over sender, date and
  subject, which lets anyone check that an index still points at the same email.

Rows whose (sender, date, subject) key is not unique, either in the raw file or
across the output files, are never drawn, so every id resolves to exactly one
email.

Usage::

    # write the id manifest (no email text)
    python scripts/sample_ground_truth.py --out manifest.csv

    # also write a blind labeling sheet (contains email text; keep it out of Git)
    python scripts/sample_ground_truth.py --sheet /tmp/labeling-sheet.txt

The labeling sheet hides the output file, the predicted domain and the
phishing label so that the labeler is not anchored on the pipeline's answer.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Iterator

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CLASSIFIED_DIR = REPO_ROOT / "classified-data" / "ceas_08"
DEFAULT_RAW_PATH = REPO_ROOT / "raw-data" / "CEAS_08.csv"

DEFAULT_SEED = 16

OUTPUT_FILES: tuple[str, ...] = (
    "email_education.csv",
    "email_finance.csv",
    "email_government.csv",
    "email_healthcare.csv",
    "email_hr.csv",
    "email_logistics.csv",
    "email_retail.csv",
    "email_social_media.csv",
    "email_technology.csv",
    "email_telecommunications.csv",
    "email_unsure.csv",
)

# Emails drawn per output file and per label value. email_unsure.csv gets a
# larger quota because it is where the definition's `none` cases concentrate.
DEFAULT_PER_LABEL_QUOTA = 8
PER_LABEL_QUOTA_OVERRIDES: dict[str, int] = {"email_unsure.csv": 10}

LABEL_VALUES: tuple[str, ...] = ("1", "0")

MANIFEST_COLUMNS: tuple[str, ...] = (
    "email_id",
    "raw_row",
    "source_file",
    "source_row",
    "label",
)

LFS_POINTER_PREFIX = b"version https://git-lfs.github.com/spec/v1"

EmailKey = tuple[str, str, str]


def email_id(sender: str, date: str, subject: str) -> str:
    """Return the stable 16-hex-digit id of an email."""
    joined = "\x1f".join((sender, date, subject))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:16]


def is_lfs_pointer(path: Path) -> bool:
    """Return True when ``path`` is a Git LFS pointer, not the real file."""
    with open(path, "rb") as fh:
        return fh.read(len(LFS_POINTER_PREFIX)) == LFS_POINTER_PREFIX


def _allow_large_fields() -> None:
    """Raise the csv module's field size limit; email bodies can be large."""
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


def _read_rows(path: Path) -> Iterator[dict[str, str]]:
    _allow_large_fields()
    with open(path, newline="", encoding="utf-8") as fh:
        yield from csv.DictReader(fh)


def _key(row: dict[str, str]) -> EmailKey:
    return (row["sender"], row["date"], row["subject"])


def index_raw(raw_path: Path) -> dict[EmailKey, list[int]]:
    """Map each (sender, date, subject) key to its raw data-row indices."""
    index: dict[EmailKey, list[int]] = {}
    for i, row in enumerate(_read_rows(raw_path)):
        index.setdefault(_key(row), []).append(i)
    return index


def per_label_quota(file_name: str) -> int:
    """Return how many emails of each label to draw from ``file_name``."""
    return PER_LABEL_QUOTA_OVERRIDES.get(file_name, DEFAULT_PER_LABEL_QUOTA)


def draw_sample(
    classified_dir: Path = DEFAULT_CLASSIFIED_DIR,
    raw_path: Path = DEFAULT_RAW_PATH,
    seed: int = DEFAULT_SEED,
    output_files: tuple[str, ...] = OUTPUT_FILES,
) -> list[dict[str, str]]:
    """Draw the stratified sample and return one manifest row per email.

    Rows come back ordered by output file, then by row index in that file.
    """
    raw_index = index_raw(raw_path)

    output_key_counts: Counter[EmailKey] = Counter()
    for name in output_files:
        for row in _read_rows(classified_dir / name):
            output_key_counts[_key(row)] += 1

    sample: list[dict[str, str]] = []
    for name in output_files:
        candidates: dict[str, list[tuple[int, dict[str, str]]]] = {
            label: [] for label in LABEL_VALUES
        }
        for i, row in enumerate(_read_rows(classified_dir / name)):
            key = _key(row)
            if len(raw_index.get(key, [])) != 1 or output_key_counts[key] != 1:
                continue
            if row["label"] in candidates:
                candidates[row["label"]].append((i, row))

        rng = random.Random(f"{seed}:{name}")
        quota = per_label_quota(name)
        drawn: list[tuple[int, dict[str, str]]] = []
        for label in LABEL_VALUES:
            pool = candidates[label]
            if len(pool) < quota:
                raise ValueError(
                    f"{name}: only {len(pool)} eligible rows with label={label}, "
                    f"need {quota}"
                )
            drawn.extend(rng.sample(pool, quota))

        for i, row in sorted(drawn, key=lambda item: item[0]):
            key = _key(row)
            sample.append(
                {
                    "email_id": email_id(*key),
                    "raw_row": str(raw_index[key][0]),
                    "source_file": name,
                    "source_row": str(i),
                    "label": row["label"],
                }
            )
    return sample


def _excerpt(text: str, head: int = 1200, tail: int = 300) -> str:
    flat = re.sub(r"\s+", " ", text).strip()
    if len(flat) <= head + tail + 20:
        return flat
    return f"{flat[:head]} [...] {flat[-tail:]}"


def write_sheet(
    sample: list[dict[str, str]],
    sheet_path: Path,
    raw_path: Path = DEFAULT_RAW_PATH,
    seed: int = DEFAULT_SEED,
) -> None:
    """Write a blind labeling sheet: shuffled, without file, prediction or label.

    The sheet contains raw email text from CEAS_08 (spam and phishing). Keep it
    out of the repository.
    """
    wanted = {int(row["raw_row"]): row["email_id"] for row in sample}
    texts: dict[str, dict[str, str]] = {}
    for i, row in enumerate(_read_rows(raw_path)):
        if i in wanted:
            texts[wanted[i]] = row

    order = [row["email_id"] for row in sample]
    random.Random(f"{seed}:sheet").shuffle(order)

    with open(sheet_path, "w", encoding="utf-8") as fh:
        for n, eid in enumerate(order, start=1):
            row = texts[eid]
            fh.write(f"=== {n} | email_id={eid}\n")
            fh.write(f"From: {_excerpt(row['sender'], 200, 0)}\n")
            fh.write(f"Subject: {_excerpt(row['subject'], 300, 0)}\n")
            fh.write(f"Body: {_excerpt(row['body'])}\n\n")


def write_manifest(sample: list[dict[str, str]], out_path: Path) -> None:
    """Write the id-only manifest (no email text)."""
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(sample)


def main(argv: list[str] | None = None) -> int:
    """Run the sampler from the command line."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--classified-dir", type=Path, default=DEFAULT_CLASSIFIED_DIR)
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW_PATH)
    parser.add_argument("--out", type=Path, help="write the id manifest here")
    parser.add_argument(
        "--sheet", type=Path, help="write a blind labeling sheet (email text) here"
    )
    args = parser.parse_args(argv)

    for path in [args.raw, *(args.classified_dir / n for n in OUTPUT_FILES)]:
        if not path.exists() or is_lfs_pointer(path):
            parser.error(f"{path} is missing or an LFS pointer; run `git lfs pull`")

    sample = draw_sample(args.classified_dir, args.raw, args.seed)
    if args.out:
        write_manifest(sample, args.out)
    if args.sheet:
        write_sheet(sample, args.sheet, args.raw, args.seed)
    if not args.out and not args.sheet:
        writer = csv.DictWriter(sys.stdout, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(sample)
    print(f"sampled {len(sample)} emails", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
