"""Building the contact sheet end to end on synthetic crops."""

from __future__ import annotations

import json

import numpy as np
import pytest

from cleft.geometry import contact as C
from cleft.run import main

from fixtures import builders


def write_cohort(root, n_patients=12, size=(222, 300)):
    """A folder tree of real PNG files plus manifest rows describing them."""
    from PIL import Image

    root.mkdir(parents=True, exist_ok=True)
    rows = []
    for index in range(1, n_patients + 1):
        folder = root / str(index)
        folder.mkdir(exist_ok=True)
        # Vary aspect ratio so the per-image mapping is genuinely exercised.
        width = size[0] + (index % 4) * 25
        array = np.full((size[1], width, 3), 100 + index, dtype=np.uint8)
        image_id = 1000 + index
        Image.fromarray(array).save(folder / f"{image_id}.png")
        # An AppleDouble sidecar, as on the real tree.
        (folder / f"._{image_id}.png").write_bytes(b"\x00" * 4096)
        rows.append(
            {
                "patient_id": str(index),
                "frontal_id": str(image_id),
                "class3": str(index % 3),
            }
        )
    return rows


@pytest.fixture(scope="module")
def cohort(tmp_path_factory):
    root = tmp_path_factory.mktemp("contact")
    rows = write_cohort(root / "photos")
    return root / "photos", rows


# --------------------------------------------------------------------------
# selection
# --------------------------------------------------------------------------


def test_selection_is_stratified_across_classes(cohort):
    _, rows = cohort
    chosen = C.select_patients(rows, 9)
    classes = [r["class3"] for r in chosen]
    assert len(set(classes)) == 3
    assert max(classes.count(c) for c in set(classes)) <= 4


def test_selection_is_deterministic(cohort):
    """A sheet that showed a different set each run would make the judgement
    unrepeatable, and the point is something you can come back to."""
    _, rows = cohort
    assert C.select_patients(rows, 6) == C.select_patients(rows, 6)


def test_selection_includes_the_worst_grades(cohort):
    """Grade 4-5 anatomy is where a band boundary is most likely to land badly."""
    _, rows = cohort
    chosen = C.select_patients(rows, 6)
    assert any(r["class3"] == "2" for r in chosen)


def test_selection_caps_at_the_cohort_size(cohort):
    _, rows = cohort
    assert len(C.select_patients(rows, 500)) == len(rows)


def test_selection_rejects_a_bad_count(cohort):
    _, rows = cohort
    with pytest.raises(C.ContactError, match="n_patients"):
        C.select_patients(rows, 0)


# --------------------------------------------------------------------------
# locating images
# --------------------------------------------------------------------------


def test_the_image_finder_skips_appledouble_sidecars(cohort):
    root, _ = cohort
    found = C.find_image(root, 1, 1001)
    assert not found.name.startswith("._")
    assert "1001" in found.stem


def test_a_missing_image_is_reported_with_an_inventory(cohort):
    root, _ = cohort
    with pytest.raises(C.ContactError, match="Present:"):
        C.find_image(root, 1, 99999)


def test_a_missing_folder_is_reported(cohort):
    root, _ = cohort
    with pytest.raises(C.ContactError, match="no folder"):
        C.find_image(root, 9999, 1001)


# --------------------------------------------------------------------------
# the sheets
# --------------------------------------------------------------------------


@pytest.mark.parametrize("geometry", ["g1", "g2"])
def test_a_sheet_renders_at_both_geometries(cohort, geometry):
    root, rows = cohort
    result = C.build_sheet(geometry, C.select_patients(rows, 6), root, columns=3)
    assert result.geometry == geometry
    assert len(result.labels) == 6
    assert result.sheet.ndim == 3 and result.sheet.shape[2] == 3


def test_the_two_geometries_produce_different_sheets(cohort):
    root, rows = cohort
    selected = C.select_patients(rows, 4)
    g1 = C.build_sheet("g1", selected, root, columns=2)
    g2 = C.build_sheet("g2", selected, root, columns=2)
    assert not np.array_equal(g1.sheet, g2.sheet)


def test_g1_panels_show_white_in_the_outer_patches(cohort):
    """One of the four questions the sheet answers."""
    root, rows = cohort
    result = C.build_sheet("g1", C.select_patients(rows, 4), root, columns=2)
    assert all(r["n_patches_with_white"] > 0 for r in result.reports)


def test_g2_panels_have_no_white_patches(cohort):
    """Under G2 the mask is the full square, so nothing is excluded."""
    root, rows = cohort
    result = C.build_sheet("g2", C.select_patients(rows, 4), root, columns=2)
    assert all(r["n_patches_with_white"] == 0 for r in result.reports)


def test_labels_carry_the_grade_and_aspect_ratio(cohort):
    """Because "does this band land right" differs for a grade 1 and a grade 5."""
    root, rows = cohort
    result = C.build_sheet("g1", C.select_patients(rows, 3), root, columns=3)
    for label in result.labels:
        assert " c" in label and "ar=" in label


@pytest.mark.parametrize("generator", ["grid", "anatomy", "random"])
def test_every_generator_renders(cohort, generator):
    root, rows = cohort
    result = C.build_sheet(
        "g1", C.select_patients(rows, 3), root, generator=generator, columns=3
    )
    assert result.reports[0]["n_patches"] == 27


def test_an_unknown_geometry_is_rejected(cohort):
    root, rows = cohort
    with pytest.raises(C.ContactError, match="unknown geometry"):
        C.build_sheet("g3", C.select_patients(rows, 2), root)


def test_the_aggregate_drops_patient_ids(cohort):
    """The sheet keeps them; the SHAREABLE summary must not."""
    root, rows = cohort
    results = [C.build_sheet(g, C.select_patients(rows, 4), root, columns=2)
               for g in ("g1", "g2")]
    summary = C.aggregate(results)

    assert set(summary) == {"g1", "g2"}
    serialised = json.dumps(summary)
    for row in rows:
        assert row["frontal_id"] not in serialised
    assert summary["g1"]["worst_patch_coverage"] < 1.0
    assert summary["g2"]["worst_patch_coverage"] == pytest.approx(1.0, abs=1e-6)


# --------------------------------------------------------------------------
# as a task
# --------------------------------------------------------------------------


def test_the_task_writes_a_cluster_only_png_and_a_shareable_report(
    cohort, tmp_path, clean_repo, monkeypatch
):
    root, rows = cohort
    monkeypatch.setenv("CLEFT_REPO_ROOT", str(clean_repo))

    artifact = tmp_path / "cleft_v1"
    artifact.mkdir()
    header = "patient_id,frontal_id,basal_id,mean,median,mode,weighted_mean,orthodontist,soft_1,soft_2,soft_3,soft_4,soft_5,class3,fold"
    lines = ["# CLUSTER-ONLY: patient-keyed", header]
    for row in rows:
        lines.append(
            f"{row['patient_id']},{row['frontal_id']},,2.0,2.0,2.0,2.0,2.0,"
            f"0.2,0.2,0.2,0.2,0.2,{row['class3']},0"
        )
    (artifact / "manifest.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")

    from cleft.provenance import hash_dir

    config = builders.write_config(
        tmp_path / "sheet.yaml",
        tier="dev",
        phase="p2",
        inputs=[
            {"name": "m", "path": str(artifact), "rollup_sha256": hash_dir(artifact)["rollup"]},
            {"name": "p", "path": str(root), "rollup_sha256": hash_dir(root)["rollup"]},
        ],
        task={
            "kind": "contact_sheet",
            "manifest_artifact": "m",
            "patient_folders": "p",
            "generator": "grid",
            "geometries": ["g1", "g2"],
            "n_patients": 6,
            "columns": 3,
        },
    )

    run_dir = main(["--config", str(config), "--out", str(tmp_path / "runs")])

    outputs = json.loads((run_dir / "outputs.json").read_text(encoding="utf-8"))
    tiers = {entry["name"]: entry["tier"] for entry in outputs["outputs"]}
    assert tiers["contact_sheet_g1.png"] == "CLUSTER-ONLY"
    assert tiers["contact_sheet_g2.png"] == "CLUSTER-ONLY"
    assert tiers["metrics.json"] == "SHAREABLE"
    assert (run_dir / "contact_sheet_g1.png").stat().st_size > 0


def test_the_shipped_config_is_valid(repo_root):
    from cleft.config import load_config

    cfg = load_config(repo_root / "configs" / "p2_contact_sheet.yaml")
    assert cfg["task"]["kind"] == "contact_sheet"
    assert cfg["task"]["geometries"] == ["g1", "g2"]
    assert cfg["tier"] == "dev", "a visual check is a look, not a citable result"
