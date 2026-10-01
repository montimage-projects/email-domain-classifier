"""Validate the CEAS_08 domain ground-truth set and its sampler (#16)."""

from __future__ import annotations

import csv
import importlib.util
import sys
from collections import Counter
from functools import lru_cache
from pathlib import Path
from types import ModuleType

import pytest

from email_classifier.domains import get_domain_names

REPO_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = REPO_ROOT / "data" / "ground-truth" / "ceas_08_domain_labels.csv"
SAMPLER_PATH = REPO_ROOT / "scripts" / "sample_ground_truth.py"

REQUIRED_COLUMNS = [
    "email_id",
    "raw_row",
    "source_file",
    "source_row",
    "label",
    "domain",
    "confidence",
    "ambiguous",
    "rationale",
    "labeler",
    "verified",
    "definition_ref",
]
EXPECTED_FILES = {
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
}


def _load_sampler() -> ModuleType:
    spec = importlib.util.spec_from_file_location("sample_ground_truth", SAMPLER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


sampler = _load_sampler()


@lru_cache(maxsize=1)
def _dataset() -> tuple[tuple[str, ...], tuple[dict[str, str], ...]]:
    with open(DATASET_PATH, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        rows = tuple(reader)
        return tuple(reader.fieldnames or ()), rows


def _rows() -> tuple[dict[str, str], ...]:
    return _dataset()[1]


def _real_data_available() -> bool:
    paths = [sampler.DEFAULT_RAW_PATH] + [
        sampler.DEFAULT_CLASSIFIED_DIR / name for name in sampler.OUTPUT_FILES
    ]
    return all(p.exists() and not sampler.is_lfs_pointer(p) for p in paths)


needs_lfs_data = pytest.mark.skipif(
    not _real_data_available(),
    reason="CEAS_08 CSVs are Git LFS pointers here; run `git lfs pull`",
)


class TestDatasetFile:
    """Shape and content checks that need only the committed CSV."""

    def test_has_exactly_the_required_columns(self) -> None:
        """The header is fixed and carries no email text."""
        header = list(_dataset()[0])
        assert header == REQUIRED_COLUMNS
        assert "body" not in header and "subject" not in header

    def test_row_count_is_150_to_200(self) -> None:
        """The set holds 150-200 emails, as the issue requires."""
        assert 150 <= len(_rows()) <= 200

    def test_ids_are_unique(self) -> None:
        """Each email appears once and every id is a 16-hex-digit hash."""
        rows = _rows()
        assert len({r["email_id"] for r in rows}) == len(rows)
        assert len({r["raw_row"] for r in rows}) == len(rows)
        assert all(
            len(r["email_id"]) == 16
            and all(c in "0123456789abcdef" for c in r["email_id"])
            for r in rows
        )

    def test_domain_values_are_the_ten_domains_or_none(self) -> None:
        """Labels use the 10 domain names or `none`, never `unsure`."""
        valid = set(get_domain_names()) | {"none"}
        assert len(valid) == 11
        domains = {r["domain"] for r in _rows()}
        assert domains <= valid
        assert "unsure" not in domains

    def test_every_output_file_is_represented(self) -> None:
        """All 11 output files contribute, including email_unsure.csv."""
        counts = Counter(r["source_file"] for r in _rows())
        assert set(counts) == EXPECTED_FILES
        assert counts["email_unsure.csv"] >= 10

    def test_both_phishing_labels_are_present(self) -> None:
        """Phishing and legitimate emails are both well represented."""
        counts = Counter(r["label"] for r in _rows())
        assert set(counts) == {"0", "1"}
        assert min(counts.values()) >= 50

    def test_metadata_values(self) -> None:
        """Metadata columns hold only their allowed values."""
        for r in _rows():
            assert r["confidence"] in {"high", "medium", "low"}
            assert r["ambiguous"] in {"true", "false"}
            assert r["labeler"] in {"agent", "human"}
            assert r["verified"] in {"true", "false"}
            assert r["definition_ref"].startswith("docs/design/domain-profiles.md@")
            assert r["raw_row"].isdigit() and r["source_row"].isdigit()

    def test_unverified_agent_labels_are_flagged(self) -> None:
        """Agent labels stay marked unverified until a human checks them."""
        for r in _rows():
            if r["labeler"] == "agent":
                assert r["verified"] == "false"

    def test_ambiguous_rows_have_a_rationale(self) -> None:
        """Every ambiguous row explains the alternative reading."""
        assert all(r["rationale"].strip() for r in _rows() if r["ambiguous"] == "true")


def _write_csv(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _make_synthetic_data(tmp_path: Path) -> tuple[Path, Path, tuple[str, ...]]:
    files = ("email_a.csv", "email_b.csv")
    fields = ["sender", "date", "subject", "body", "label"]
    raw_rows: list[dict[str, str]] = []
    classified = tmp_path / "classified"
    classified.mkdir()
    for name in files:
        out_rows = []
        for i in range(30):
            row = {
                "sender": f"s{i}@{name}",
                "date": f"d{i}",
                "subject": f"subj {i}",
                "body": "b",
                "label": str(i % 2),
            }
            out_rows.append(row)
            raw_rows.append(row)
        _write_csv(classified / name, out_rows, fields)
    # A key duplicated in the raw file must never be drawn.
    raw_rows.append(dict(raw_rows[0]))
    raw = tmp_path / "raw.csv"
    _write_csv(raw, raw_rows, fields)
    return classified, raw, files


class TestSampler:
    """Sampler logic on a small synthetic dataset (no LFS data needed)."""

    def test_quota_determinism_and_exclusion(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Quotas hold per file and label; same seed, same sample; dups skipped."""
        classified, raw, files = _make_synthetic_data(tmp_path)
        monkeypatch.setattr(sampler, "DEFAULT_PER_LABEL_QUOTA", 4)
        monkeypatch.setattr(sampler, "PER_LABEL_QUOTA_OVERRIDES", {"email_b.csv": 5})

        first = sampler.draw_sample(classified, raw, 7, files)
        second = sampler.draw_sample(classified, raw, 7, files)
        other_seed = sampler.draw_sample(classified, raw, 8, files)

        assert first == second
        assert first != other_seed
        per_file = Counter((r["source_file"], r["label"]) for r in first)
        assert per_file == {
            ("email_a.csv", "0"): 4,
            ("email_a.csv", "1"): 4,
            ("email_b.csv", "0"): 5,
            ("email_b.csv", "1"): 5,
        }
        dup_id = sampler.email_id("s0@email_a.csv", "d0", "subj 0")
        assert dup_id not in {r["email_id"] for r in first}

    def test_too_few_rows_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A quota larger than the eligible pool fails loudly."""
        classified, raw, files = _make_synthetic_data(tmp_path)
        monkeypatch.setattr(sampler, "DEFAULT_PER_LABEL_QUOTA", 16)
        monkeypatch.setattr(sampler, "PER_LABEL_QUOTA_OVERRIDES", {})
        with pytest.raises(ValueError, match="eligible rows"):
            sampler.draw_sample(classified, raw, 7, files)

    def test_lfs_pointer_detection(self, tmp_path: Path) -> None:
        """LFS pointer files are told apart from real CSVs."""
        pointer = tmp_path / "pointer.csv"
        pointer.write_text(
            "version https://git-lfs.github.com/spec/v1\noid sha256:abc\nsize 1\n"
        )
        real = tmp_path / "real.csv"
        real.write_text("sender,date\n")
        assert sampler.is_lfs_pointer(pointer)
        assert not sampler.is_lfs_pointer(real)


@needs_lfs_data
class TestAgainstSourceData:
    """Checks that resolve the ids against the real CEAS_08 files."""

    def test_ids_resolve_in_raw_and_output_files(self) -> None:
        """raw_row and source_file/source_row point at the hashed email."""
        rows = _rows()
        wanted_raw = {int(r["raw_row"]): r for r in rows}
        seen = 0
        for i, raw in enumerate(sampler._read_rows(sampler.DEFAULT_RAW_PATH)):
            if i in wanted_raw:
                r = wanted_raw[i]
                eid = sampler.email_id(raw["sender"], raw["date"], raw["subject"])
                assert eid == r["email_id"]
                assert raw["label"] == r["label"]
                seen += 1
        assert seen == len(rows)

        by_file: dict[str, dict[int, dict[str, str]]] = {}
        for r in rows:
            by_file.setdefault(r["source_file"], {})[int(r["source_row"])] = r
        for name, wanted in by_file.items():
            found = 0
            path = sampler.DEFAULT_CLASSIFIED_DIR / name
            for i, out in enumerate(sampler._read_rows(path)):
                if i in wanted:
                    eid = sampler.email_id(out["sender"], out["date"], out["subject"])
                    assert eid == wanted[i]["email_id"]
                    found += 1
            assert found == len(wanted)

    def test_sampler_reproduces_the_dataset(self) -> None:
        """Re-running the seeded sampler yields exactly the committed ids."""
        key_cols = list(sampler.MANIFEST_COLUMNS)
        committed = [{k: r[k] for k in key_cols} for r in _rows()]
        assert committed == sampler.draw_sample()
