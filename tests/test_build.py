"""The Phase 1 build, end to end on the synthetic cohort."""

from __future__ import annotations

import csv
import json

import pytest

from cleft.data import build as B
from cleft.data import manifest as M

from fixtures import cohort as C

EXPECTED = M.Counts(patients=237, frontal=237, basal=236, photoless=14)


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    root = tmp_path_factory.mktemp("build")
    cohort = C.build_cohort(root / "cohort", write_images=True)
    C.write_scoresheet(cohort, cohort.scoresheet_integer, form="integer")
    C.write_scoresheet(cohort, cohort.scoresheet_text, form="text")
    return cohort


def run(cohort, artifact_dir, **overrides):
    kwargs = dict(
        primary_path=cohort.scoresheet_integer,
        patient_folders_path=cohort.images_dir,
        artifact_dir=artifact_dir,
        orthodontist_rater="Rater 8 - Orthodontist",
        n_folds=5,
        seed=1337,
        expected=EXPECTED,
        expected_scored_rows=251,
        log=lambda *_: None,
    )
    kwargs.update(overrides)
    return B.build(**kwargs)


# --------------------------------------------------------------------------
# the happy path
# --------------------------------------------------------------------------


def test_the_build_produces_the_expected_shape(built, tmp_path):
    result = run(built, tmp_path / "cleft_v1")
    counts = result.summary.counts
    assert (counts.patients, counts.frontal, counts.basal) == (237, 237, 236)
    assert counts.photoless == 14
    assert counts.images == 473
    assert counts.scored_rows == 251


def test_the_rule_cross_check_names_exactly_143_and_238(built, tmp_path):
    result = run(built, tmp_path / "cleft_v1")
    assert result.summary.rule_disagreements == (143, 238)
    assert result.summary.rule_matches_expected is True


def test_the_equivalence_check_passes_on_both_forms(built, tmp_path):
    result = run(
        built, tmp_path / "cleft_v1", equivalence_path=built.scoresheet_text
    )
    assert result.summary.counts.patients == 237


def test_the_artifact_contains_every_expected_file(built, tmp_path):
    result = run(built, tmp_path / "cleft_v1")
    for name in ("manifest.csv", "folds.json", "photoless_ids.json", "MANIFEST.json"):
        assert (result.artifact_dir / name).is_file(), f"missing {name}"


def test_the_manifest_csv_is_marked_cluster_only(built, tmp_path):
    result = run(built, tmp_path / "cleft_v1")
    text = (result.artifact_dir / "manifest.csv").read_text(encoding="utf-8")
    assert text.splitlines()[0].startswith("# CLUSTER-ONLY")


def test_the_manifest_csv_has_one_row_per_patient(built, tmp_path):
    result = run(built, tmp_path / "cleft_v1")
    rows = M.load_manifest(result.artifact_dir / "manifest.csv")
    assert len(rows) == 237
    assert rows[0].keys() >= {
        "patient_id", "frontal_id", "basal_id", "mean", "median", "mode",
        "weighted_mean", "orthodontist", "soft_1", "soft_5", "class3", "fold",
    }


def test_a_naive_csv_reader_would_misread_the_header(built, tmp_path):
    """Why load_manifest exists, asserted rather than left as folklore.

    The tier marker is the first line, so csv.DictReader on the raw file takes it
    as the header and every column name comes out wrong. The marker is a safety
    feature and stays; this documents the consequence and pins the workaround to
    one function instead of every caller rediscovering it.
    """
    result = run(built, tmp_path / "cleft_v1")
    path = result.artifact_dir / "manifest.csv"

    with path.open(encoding="utf-8", newline="") as handle:
        naive = list(csv.DictReader(handle))
    assert "fold" not in naive[0], "the trap this test documents has gone away"

    assert "fold" in M.load_manifest(path)[0]


def test_the_artifact_documents_its_own_csv_layout(built, tmp_path):
    """A consumer reads the schema from MANIFEST.json, not from experiment."""
    result = run(built, tmp_path / "cleft_v1")
    payload = json.loads((result.artifact_dir / "MANIFEST.json").read_text(encoding="utf-8"))
    schema = payload["manifest_csv"]

    assert schema["fold_column"] == "fold"
    assert schema["comment_lines_before_header"] == 1
    assert schema["tier"] == "CLUSTER-ONLY"
    assert "load_manifest" in schema["note"]

    documented = [c["name"] for c in schema["columns"]]
    actual = list(M.load_manifest(result.artifact_dir / "manifest.csv")[0])
    assert documented == actual, "the documented layout has drifted from the file"
    assert all(c["description"] for c in schema["columns"])


def test_the_artifact_records_learnability_under_a_findable_key(built, tmp_path):
    """It was under `learnability_237`, so a consumer found nothing."""
    result = run(built, tmp_path / "cleft_v1")
    payload = json.loads((result.artifact_dir / "MANIFEST.json").read_text(encoding="utf-8"))
    block = payload["summary"]["learnability"]

    assert block["population"] == 237
    assert set(block["scores"]) == set(__import__(
        "cleft.data.labels", fromlist=["TARGETS"]
    ).TARGETS)
    assert block["ordering"]
    assert "mean-of-four" in block["caveat"], (
        "the asymmetry caveat must travel with the numbers into the artifact"
    )


def test_patient_238_has_no_basal_in_the_csv(built, tmp_path):
    result = run(built, tmp_path / "cleft_v1")
    lines = (result.artifact_dir / "manifest.csv").read_text(encoding="utf-8").splitlines()
    row = next(r for r in csv.DictReader(lines[1:]) if r["patient_id"] == "238")
    assert row["frontal_id"] == "581"
    assert row["basal_id"] == ""


def test_folder_143_records_524_as_frontal(built, tmp_path):
    """The whole reason this phase is built by lookup."""
    result = run(built, tmp_path / "cleft_v1")
    lines = (result.artifact_dir / "manifest.csv").read_text(encoding="utf-8").splitlines()
    row = next(r for r in csv.DictReader(lines[1:]) if r["patient_id"] == "143")
    assert row["frontal_id"] == "524"
    assert row["basal_id"] == "523"


def test_the_manifest_json_records_its_sources_and_rollup(built, tmp_path):
    result = run(built, tmp_path / "cleft_v1")
    payload = json.loads((result.artifact_dir / "MANIFEST.json").read_text(encoding="utf-8"))
    assert len(payload["payload_rollup"]) == 64
    assert payload["sources"]["primary_scoresheet"].endswith(".xlsx")
    assert payload["empty_folders"] == [52]
    assert payload["rule_disagreements"] == [143, 238]


def test_the_soft_label_crosscheck_passes_against_our_own_output(built, tmp_path):
    """Round trip: write our soft labels out, feed them back as the reference."""
    first = run(built, tmp_path / "cleft_v1")
    lines = (first.artifact_dir / "manifest.csv").read_text(encoding="utf-8").splitlines()

    reference = tmp_path / "per_image_labels.csv"
    with reference.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["RanaPhotoID", "soft_1", "soft_2", "soft_3", "soft_4", "soft_5"])
        for row in csv.DictReader(lines[1:]):
            writer.writerow([row["frontal_id"], *[row[f"soft_{g}"] for g in range(1, 6)]])

    run(built, tmp_path / "cleft_v2", soft_reference_path=reference)


# --------------------------------------------------------------------------
# nothing is written unless everything passes
# --------------------------------------------------------------------------


def test_an_existing_artifact_directory_is_refused(built, tmp_path):
    target = tmp_path / "cleft_v1"
    target.mkdir(parents=True)
    with pytest.raises(B.BuildError, match="immutable|already exists"):
        run(built, target)


def test_a_failed_build_leaves_no_artifact(built, tmp_path):
    """Validate everything, then write. A half-built artifact is worse than none."""
    target = tmp_path / "cleft_v1"
    with pytest.raises(Exception):
        run(built, target, expected_scored_rows=999)
    assert not target.exists(), (
        "a failed build created an artifact directory; a later run could mistake "
        "it for a finished one"
    )


def test_a_wrong_expected_count_stops_the_build(built, tmp_path):
    with pytest.raises(M.ManifestError, match="cohort shape"):
        run(built, tmp_path / "cleft_v1", expected=M.Counts(1, 1, 1, 1))
    assert not (tmp_path / "cleft_v1").exists()


def test_an_unknown_rater_name_stops_the_build(built, tmp_path):
    with pytest.raises(B.BuildError, match="not a column"):
        run(built, tmp_path / "cleft_v1", orthodontist_rater="Rater 99 - Nobody")
    assert not (tmp_path / "cleft_v1").exists()


def test_the_rater_is_found_by_name_not_position(built, tmp_path):
    """A column reorder must not silently change which rater is the target."""
    result = run(built, tmp_path / "cleft_v1")
    assert "Rater 8 - Orthodontist" in result.summary.per_rater_r


def test_a_workbook_disagreement_stops_the_build(built, tmp_path, monkeypatch):
    from cleft.data import scoresheet as S

    real = S.load

    def fake(path, **kwargs):
        sheet = real(path, **kwargs)
        if path == built.scoresheet_text:
            first = sheet.photo_ids()[0]
            sheet.rows[first] = S.ScoreRow(first, (5, 5, 5, 5, 5))
        return sheet

    monkeypatch.setattr(S, "load", fake)
    with pytest.raises(B.BuildError, match="disagree"):
        run(built, tmp_path / "cleft_v1", equivalence_path=built.scoresheet_text)
    assert not (tmp_path / "cleft_v1").exists()


# --------------------------------------------------------------------------
# the summary
# --------------------------------------------------------------------------


def test_no_image_id_reaches_the_summary(built, tmp_path):
    """Image ids are checked here; patient ids cannot be, and are checked elsewhere.

    Patient ids run 1..238, and so do the cohort counts: 237 patients, 236 basal
    images, 238 folders. Searching the block for "236" matches the basal count,
    which is an aggregate the block is *required* to contain. There is no way to
    tell a leak from a legitimate count by looking at the text, so this test does
    not pretend to.

    The decisive check lives in ``test_report.py``, which routes patient ids of
    900001+ through the same builder, plus a structural test asserting
    ``build_summary`` cannot receive patient-level data at all.

    Image ids are distinctive here (1001+, and 523/524/581), so they are checked
    in full.
    """
    import re

    result = run(built, tmp_path / "cleft_v1")
    rendered = result.summary.render()

    for row in result.manifest.rows:
        assert not re.search(rf"\b{row.frontal_id}\b", rendered), (
            f"image id {row.frontal_id} leaked into the SHAREABLE block"
        )
        if row.basal_id is not None:
            assert not re.search(rf"\b{row.basal_id}\b", rendered), (
                f"image id {row.basal_id} leaked into the SHAREABLE block"
            )


def test_the_summary_covers_the_exit_criteria(built, tmp_path):
    result = run(built, tmp_path / "cleft_v1")
    rendered = result.summary.render()
    for marker in ("criterion 6", "criterion 7", "criterion 8", "criterion 9", "criterion 10"):
        assert marker in rendered
