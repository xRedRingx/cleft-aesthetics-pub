"""The two pre-flight diagnostics.

They exist to be run before the expensive build, so what matters is that they
report rather than assert, work with no hashes declared, and leak nothing.
"""

from __future__ import annotations

import json

import pytest

from cleft.data import diagnostics as D
from cleft.run import main

from fixtures import builders, cohort as C


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    root = tmp_path_factory.mktemp("diag")
    return C.build_cohort(root / "cohort", write_images=True)


# --------------------------------------------------------------------------
# folder scan
# --------------------------------------------------------------------------


def test_folder_scan_counts_the_real_shape(built):
    payload, rendered = D.scan_folders_report(built.images_dir)
    assert payload["n_directories"] == 238
    assert payload["n_with_images"] == 237
    assert payload["total_images"] == 473
    assert payload["first_folder"] == 1
    assert payload["last_folder"] == 238


def test_folder_scan_reports_52_as_empty_not_missing(built):
    """The corrected fact: no number is missing; folder 52 holds nothing."""
    payload, rendered = D.scan_folders_report(built.images_dir)
    assert payload["empty_folders"] == [52]
    assert payload["gaps"] == [], "every number 1..238 is present"
    assert "present but EMPTY" in rendered
    assert "missing numbers      none" in rendered


def test_folder_scan_finds_the_single_image_folder(built):
    payload, _ = D.scan_folders_report(built.images_dir)
    assert payload["single_image_folders"] == [238]
    assert payload["images_per_folder"] == {"0": 1, "1": 1, "2": 236}


def test_folder_scan_reports_the_clutter_it_skipped(built):
    """479 skipped files is a fact worth seeing, not a silent drop."""
    payload, rendered = D.scan_folders_report(built.images_dir)
    assert payload["skipped_clutter"] == 473 + 3 + 3
    assert payload["skipped_by_name"]["._*"] == 476
    assert payload["skipped_by_name"]["Thumbs.db"] == 3
    assert "clutter skipped" in rendered


def test_folder_scan_prints_what_was_expected(built):
    _, rendered = D.scan_folders_report(built.images_dir)
    assert "EXPECTED" in rendered
    assert "237" in rendered and "473" in rendered
    assert "SHAREABLE" in rendered


def test_folder_scan_reports_no_image_content(built):
    """It reads a directory listing; it never opens an image."""
    payload, _ = D.scan_folders_report(built.images_dir)
    # Filenames appear only in the clutter tally, as the "._*" pattern.
    assert "1001.jpg" not in json.dumps(payload)


def test_folder_scan_surfaces_a_structural_surprise(tmp_path):
    """The point of running it first: a clear error, not a half-built artifact."""
    from cleft.data import folderscan

    root = tmp_path / "photos"
    (root / "1").mkdir(parents=True)
    (root / "1" / "1.jpg").write_bytes(b"x")
    (root / "1" / "2.jpg").write_bytes(b"x")
    (root / "1" / "3.jpg").write_bytes(b"x")

    with pytest.raises(folderscan.FolderScanError, match="folder 1"):
        D.scan_folders_report(root)


# --------------------------------------------------------------------------
# score sheet
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def sheet_path(built):
    return C.write_scoresheet(built, built.scoresheet_integer, form="integer")


def test_scoresheet_report_counts(sheet_path):
    payload, _ = D.inspect_scoresheet_report(sheet_path)
    assert payload["n_rows"] == 251
    assert payload["n_raters"] == 5
    assert payload["n_cells"] == 1255
    assert payload["n_missing"] == 0


def test_scoresheet_report_shows_normalised_rater_names(sheet_path):
    """The most useful line: if header matching failed, it failed here."""
    payload, rendered = D.inspect_scoresheet_report(sheet_path)
    assert payload["raters_normalised"] == list(C.RATERS)
    for rater in C.RATERS:
        assert rater in rendered


def test_scoresheet_histogram_sums_to_the_cell_count(sheet_path):
    payload, _ = D.inspect_scoresheet_report(sheet_path)
    assert sum(payload["grade_histogram"].values()) == payload["n_cells"]
    for counts in payload["per_rater_histogram"].values():
        assert sum(counts.values()) == payload["n_rows"]


def test_scoresheet_report_names_the_grade_words(sheet_path):
    _, rendered = D.inspect_scoresheet_report(sheet_path)
    for word in ("Excellent", "Good", "Fair", "Poor", "Very Poor"):
        assert word in rendered


def test_scoresheet_report_contains_no_per_patient_value(sheet_path):
    """Counts only. No photo id may appear beside a grade."""
    payload, rendered = D.inspect_scoresheet_report(sheet_path)
    assert "rows" not in payload and "grades" not in payload
    # The id range is a bound, not a list.
    assert isinstance(payload["photo_id_min"], int)
    assert "per_rater_histogram" in payload
    assert rendered.count("\n") < 60, "a per-patient dump would be far longer"


# --------------------------------------------------------------------------
# they run as tasks, with no declared inputs and therefore no hashes
# --------------------------------------------------------------------------


def test_the_scan_task_reads_through_a_declared_input(
    tmp_path, built, clean_repo, monkeypatch
):
    """[UPDATED 2026-09-05] Was "runnable before the hash round trip",
    with ``inputs=[]`` -- the property that let the diagnostics read
    cohort data guard 3 never verified. It now runs through a DECLARED
    input, and the declaration is hash-checked before the task opens
    anything."""
    from cleft.provenance import hash_dir

    monkeypatch.setenv("CLEFT_REPO_ROOT", str(clean_repo))
    config = builders.write_config(
        tmp_path / "scan.yaml",
        tier="dev",
        phase="p1",
        inputs=[{
            "name": "patient_folders",
            "path": str(built.images_dir),
            "rollup_sha256": hash_dir(built.images_dir)["rollup"],
        }],
        task={
            "kind": "scan_folders",
            "patient_folders": "patient_folders",
        },
    )
    run_dir = main(["--config", str(config), "--out", str(tmp_path / "runs")])
    payload = json.loads((run_dir / "folder_scan.json").read_text(encoding="utf-8"))
    assert payload["n_directories"] == 238
    assert payload["n_with_images"] == 237


def test_the_inspect_task_reads_through_a_declared_input(
    tmp_path, sheet_path, clean_repo, monkeypatch
):
    """[UPDATED 2026-09-05] See the scan task above: the workbook is
    declared and hash-verified rather than opened on a path."""
    from cleft.provenance import hash_path

    monkeypatch.setenv("CLEFT_REPO_ROOT", str(clean_repo))
    config = builders.write_config(
        tmp_path / "inspect.yaml",
        tier="dev",
        phase="p1",
        inputs=[{
            "name": "scoresheet",
            "path": str(sheet_path),
            "rollup_sha256": hash_path(sheet_path)["rollup"],
        }],
        task={"kind": "inspect_scoresheet", "scoresheet": "scoresheet"},
    )
    run_dir = main(["--config", str(config), "--out", str(tmp_path / "runs")])
    payload = json.loads((run_dir / "scoresheet_report.json").read_text(encoding="utf-8"))
    assert payload["n_rows"] == 251


def test_the_shipped_diagnostic_configs_are_valid(repo_root):
    from cleft.config import load_config

    for name in ("p1_scan_folders.yaml", "p1_inspect_scoresheet.yaml"):
        cfg = load_config(repo_root / "configs" / name)
        # [CORRECTED 2026-09-05] This asserted `inputs == []`, which is
        # the property that let these two read the cohort score sheet
        # with nothing verifying it. The invariant is the opposite one.
        assert cfg["inputs"], (
            f"{name} reads a cohort artifact and must DECLARE it, so "
            "guard 3 verifies the bytes before the task opens them"
        )
        for entry in cfg["inputs"]:
            assert len(entry["rollup_sha256"]) == 64, entry["name"]
            assert set(entry["rollup_sha256"]) != {"0"}, (
                f"{name}: {entry['name']} still carries a placeholder; "
                "the same artifact is declared in p1_build_manifest.yaml"
            )
        # ...and the task names those inputs rather than carrying paths.
        for key, value in cfg["task"].items():
            if key == "kind" or not isinstance(value, str):
                continue
            assert "/" not in value and not value.startswith("${"), (
                f"{name}: task.{key} carries a path, not an input name"
            )
        assert cfg["tier"] == "dev", f"{name} is a look, not a citable result"
