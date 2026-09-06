"""The Phase 2 cluster build: exit criteria 7, 8 and 10."""

from __future__ import annotations

import json

import numpy as np
import pytest

from cleft.geometry import stage_build as SB

from fixtures import builders


def cohort(root, n=8):
    """Patients spanning the measured aspect-ratio range."""
    from PIL import Image

    root.mkdir(parents=True, exist_ok=True)
    ratios = np.linspace(0.56, 1.09, n)
    rows = []
    for index, ratio in enumerate(ratios, start=1):
        folder = root / str(index)
        folder.mkdir(exist_ok=True)
        image_id = 1000 + index
        width = int(round(300 * ratio))
        Image.fromarray(
            np.full((300, width, 3), 100 + index, dtype=np.uint8)
        ).save(folder / f"{image_id}.png")
        rows.append((index, image_id, index % 3))
    return rows


def manifest(root, rows):
    root.mkdir(parents=True, exist_ok=True)
    header = (
        "patient_id,frontal_id,basal_id,mean,median,mode,weighted_mean,"
        "orthodontist,soft_1,soft_2,soft_3,soft_4,soft_5,class3,fold"
    )
    lines = ["# CLUSTER-ONLY: patient-keyed", header]
    for patient_id, image_id, cls in rows:
        lines.append(
            f"{patient_id},{image_id},,2.0,2.0,2.0,2.0,2.0,"
            f"0.2,0.2,0.2,0.2,0.2,{cls},0"
        )
    (root / "manifest.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return root


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    base = tmp_path_factory.mktemp("p2build")
    rows = cohort(base / "photos")
    manifest_dir = manifest(base / "cleft_v1", rows)
    result = SB.build(
        manifest_dir=manifest_dir,
        patient_folders=base / "photos",
        artifact_dir=base / "staged_v1",
        log=lambda *_: None,
    )
    return result, base, rows


# --------------------------------------------------------------------------
# criterion 7 -- staging over the cohort
# --------------------------------------------------------------------------


def test_every_patient_is_staged(built):
    result, _, rows = built
    assert result.summary["n_patients"] == len(rows)


def test_the_aspect_ratio_distribution_is_reported_and_checked(built):
    result, _, _ = built
    ratios = result.summary["aspect_ratio"]
    assert ratios["expected_min"] == 0.553 and ratios["expected_max"] == 1.099
    assert ratios["min"] >= 0.553 - 0.02
    assert ratios["max"] <= 1.099 + 0.02


def test_an_aspect_ratio_outside_the_measured_range_stops_the_build(tmp_path):
    """Either these are not the crops it was measured on, or one is malformed."""
    from PIL import Image

    root = tmp_path / "photos"
    rows = cohort(root, n=3)
    # A crop far wider than anything measured.
    Image.fromarray(np.full((100, 400, 3), 50, dtype=np.uint8)).save(
        root / "1" / "1001.png"
    )
    with pytest.raises(SB.StageBuildError, match="outside the measured"):
        SB.build(
            manifest_dir=manifest(tmp_path / "m", rows),
            patient_folders=root,
            artifact_dir=tmp_path / "out",
            log=lambda *_: None,
        )


def test_staged_tensors_have_the_expected_shape(built):
    result, _, rows = built
    for geometry in ("g1", "g2"):
        array = np.load(result.artifact_dir / f"staged_patient_{geometry}.npy")
        assert array.shape == (len(rows), 224, 224, 3)
        assert array.dtype == np.uint8


def test_g2_staging_removes_the_white_corners(built):
    result, _, _ = built
    g2 = np.load(result.artifact_dir / "staged_patient_g2.npy")
    corner = g2[:, 0, 0]
    assert not (corner == 255).all(), "G2 must leave no white corner"


# --------------------------------------------------------------------------
# criterion 8 -- patches at both geometries
# --------------------------------------------------------------------------


def test_patches_are_generated_for_every_combination(built):
    result, _, _ = built
    for generator in ("grid", "anatomy", "random"):
        for geometry in ("g1", "g2"):
            entry = result.summary["patches"][f"{generator}_{geometry}"]
            assert entry["n_patches"] == 27


def test_the_g1_white_fraction_is_reported_per_band(built):
    """Exit criterion 8 asks for the distribution, and it differs by band."""
    result, _, _ = built
    per_band = result.summary["patches"]["grid_g1"]["white_fraction_per_band"]
    assert set(per_band) == {"top", "middle", "bottom"}
    assert per_band["top"]["max"] > per_band["bottom"]["max"]
    for entry in per_band.values():
        assert 0.0 <= entry["mean"] <= 1.0


def test_g2_carries_no_white_for_any_generator(built):
    result, _, _ = built
    for generator in ("grid", "anatomy", "random"):
        entry = result.summary["patches"][f"{generator}_g2"]
        assert entry["white_fraction_max"] == pytest.approx(0.0, abs=1e-6)


def test_the_settled_anatomy_parameters_travel_with_the_artifact(built):
    result, _, _ = built
    params = result.summary["anatomy_parameters"]
    assert params == {
        "v_offset": 0.04, "scale": 1.80, "scale_x": 1.60, "box_scale_x": 1.0
    }


def test_the_mapping_safeguard_runs_over_the_real_cohort(built):
    """Two patients of different AR must get different boxes -- checked here on
    the actual crops, not only on two synthetic images."""
    result, _, _ = built
    assert result.summary["pad_fraction"]["min"] < result.summary["pad_fraction"]["max"]


# --------------------------------------------------------------------------
# criterion 10 -- the artifact
# --------------------------------------------------------------------------


def test_the_artifact_contains_everything_phase_three_needs(built):
    result, _, _ = built
    for name in (
        "staged_patient_g1.npy", "staged_patient_g2.npy",
        "geometry.csv", "patches.json", "MANIFEST.json",
    ):
        assert (result.artifact_dir / name).is_file(), f"missing {name}"


def test_the_artifact_is_hashed(built):
    result, _, _ = built
    payload = json.loads((result.artifact_dir / "MANIFEST.json").read_text(encoding="utf-8"))
    assert len(payload["payload_rollup"]) == 64
    assert payload["phase"] == "p2"
    assert payload["staged_tier"] == "CLUSTER-ONLY"


def test_the_manifest_explains_the_tensor_layout(built):
    """A consumer must not have to guess the row order."""
    result, _, _ = built
    payload = json.loads((result.artifact_dir / "MANIFEST.json").read_text(encoding="utf-8"))
    assert "geometry.csv" in payload["note"]
    assert "content_box_to_pixels" in payload["note"]


def test_geometry_csv_is_marked_cluster_only(built):
    result, _, _ = built
    text = (result.artifact_dir / "geometry.csv").read_text(encoding="utf-8")
    assert text.splitlines()[0].startswith("# CLUSTER-ONLY")


def test_geometry_rows_match_the_tensor_order(built):
    import csv

    result, _, rows = built
    lines = (result.artifact_dir / "geometry.csv").read_text(encoding="utf-8").splitlines()
    parsed = list(csv.DictReader(lines[1:]))
    assert [int(r["patient_id"]) for r in parsed] == [r[0] for r in rows]


def test_patches_json_carries_the_mirror_relation(built):
    result, _, _ = built
    payload = json.loads((result.artifact_dir / "patches.json").read_text(encoding="utf-8"))
    for key, produced in payload.items():
        assert len(produced) == 27, key
        ids = {p["id"] for p in produced}
        assert all(p["mirror_id"] in ids for p in produced), key


def test_an_existing_artifact_is_refused(built):
    result, base, rows = built
    with pytest.raises(SB.StageBuildError, match="immutable|already exists"):
        SB.build(
            manifest_dir=base / "cleft_v1",
            patient_folders=base / "photos",
            artifact_dir=result.artifact_dir,
            log=lambda *_: None,
        )


def test_the_shipped_config_is_valid(repo_root):
    from cleft.config import load_config

    cfg = load_config(repo_root / "configs" / "p2_stage_and_patch.yaml")
    assert cfg["tier"] == "keeper", "this artifact is cited by Phase 3"
    assert cfg["task"]["generators"] == ["grid", "anatomy", "random"]
    assert cfg["task"]["geometries"] == ["g1", "g2"]
