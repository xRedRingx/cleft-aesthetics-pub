"""The mirror-difference symmetry index (Phase 4 §3.1).

Laptop exit criterion 3: features computable on synthetic data, **with a known
asymmetry recovered**. Recovering it is the part that matters -- a feature set
that runs and returns numbers proves only that it runs.

Every fixture here is synthetic. No real data, at any point.
"""

from __future__ import annotations

import numpy as np
import pytest

from cleft.geometry import mirror
from cleft.geometry.patches import BandSpec, PatchConfig

SIZE = 224


def blank(value: int = 128) -> np.ndarray:
    return np.full((SIZE, SIZE, 3), value, dtype=np.uint8)


def with_blob(
    centre_y: float, centre_x: float, radius: int = 10, value: int = 255
) -> np.ndarray:
    """A bright square at a normalised position. One side only, so it is asymmetric."""
    image = blank()
    row = int(centre_y * SIZE)
    column = int(centre_x * SIZE)
    image[
        max(row - radius, 0) : row + radius,
        max(column - radius, 0) : column + radius,
    ] = value
    return image


# --------------------------------------------------------------------------
# the geometry requirement -- mechanical, not remembered
# --------------------------------------------------------------------------


def test_g1_is_refused_with_the_reason():
    with pytest.raises(mirror.MirrorError, match="white and symmetric"):
        mirror.require_g2("g1")


def test_the_feature_matrix_refuses_g1():
    with pytest.raises(mirror.MirrorError):
        mirror.feature_matrix(np.stack([blank()]), geometry="g1")


def test_g2_is_accepted():
    mirror.require_g2("g2")


# --------------------------------------------------------------------------
# the residual
# --------------------------------------------------------------------------


def test_a_symmetric_image_has_no_residual():
    """The floor. Anything above zero here is an artefact of the instrument."""
    image = blank()
    assert mirror.residual_map(image).max() == 0.0
    assert mirror.mirror_features(image)[mirror.WHOLE] == 0.0


def test_a_mirrored_pair_of_blobs_is_symmetric():
    image = with_blob(0.5, 0.3)
    image[:, ::-1] = np.maximum(image[:, ::-1], image)
    assert mirror.residual_map(image).max() == pytest.approx(0.0, abs=1e-6)


def test_a_one_sided_blob_produces_a_residual():
    assert mirror.residual_map(with_blob(0.5, 0.3)).max() > 0.4


def test_the_residual_map_is_itself_mirror_symmetric():
    """The property the pair-collapsing rests on. If this broke, half the
    anatomy features would be silently discarding real signal."""
    for image in (blank(), with_blob(0.5, 0.3), with_blob(0.2, 0.15, radius=25)):
        assert mirror.mirror_symmetry_error(image) == pytest.approx(0.0, abs=1e-6)


def test_uint8_and_float_agree():
    image = with_blob(0.5, 0.3)
    assert mirror.residual_map(image) == pytest.approx(
        mirror.residual_map(image.astype(np.float32) / 255.0), abs=1e-6
    )


def test_a_greyscale_image_is_accepted():
    image = with_blob(0.5, 0.3)[:, :, 0]
    assert mirror.residual_map(image).shape == (SIZE, SIZE)


# --------------------------------------------------------------------------
# EXIT CRITERION 3 -- a known asymmetry is recovered
# --------------------------------------------------------------------------


def test_a_known_asymmetry_lands_in_the_band_that_contains_it():
    """Blob in the bottom third -> the bottom band carries the difference."""
    features = mirror.mirror_features(with_blob(0.85, 0.25))
    bottom = features[f"{mirror.BAND_PREFIX}bottom"]
    assert bottom > features[f"{mirror.BAND_PREFIX}top"]
    assert bottom > features[f"{mirror.BAND_PREFIX}middle"]
    assert features[f"{mirror.BAND_PREFIX}top"] == pytest.approx(0.0, abs=1e-6)


def test_a_known_asymmetry_lands_in_the_named_structure_that_contains_it():
    """The interpretable claim: an asymmetric alar base reads as *alar_base*.

    The blob is placed at the frozen alar_base box's own centre, so the test
    cannot drift away from the region table -- if the regions move, the blob
    moves with them and the assertion still means what it says.
    """
    boxes = mirror.anatomy_boxes()
    x, y, w, h = boxes["alar_base"]
    features = mirror.mirror_features(with_blob(y + h / 2, x + w / 2, radius=8))

    regions = {
        name: value
        for name, value in features.items()
        if name.startswith(mirror.REGION_PREFIX)
    }
    assert max(regions, key=regions.get) == f"{mirror.REGION_PREFIX}alar_base"


def test_a_larger_asymmetry_reads_as_a_larger_index():
    """Monotonicity. Without it the index orders patients by nothing in particular."""
    small = mirror.mirror_features(with_blob(0.5, 0.25, radius=6))[mirror.WHOLE]
    large = mirror.mirror_features(with_blob(0.5, 0.25, radius=20))[mirror.WHOLE]
    assert large > small > 0.0


def test_a_fainter_asymmetry_reads_as_a_smaller_index():
    faint = mirror.mirror_features(with_blob(0.5, 0.25, value=160))[mirror.WHOLE]
    stark = mirror.mirror_features(with_blob(0.5, 0.25, value=255))[mirror.WHOLE]
    assert stark > faint > 0.0


def test_the_side_the_asymmetry_falls_on_does_not_change_the_index():
    """Laterality is not recorded anywhere in this cohort (PLAN §4.1), so an
    index that scored left-sided and right-sided asymmetry differently would be
    measuring something the labels cannot contain."""
    left = mirror.mirror_features(with_blob(0.5, 0.25))
    right = mirror.mirror_features(with_blob(0.5, 0.75))
    for name, value in left.items():
        assert right[name] == pytest.approx(value, abs=1e-6)


# --------------------------------------------------------------------------
# the feature set
# --------------------------------------------------------------------------


def test_mirrored_pairs_contribute_one_feature_not_two():
    """9 midline + 9 bilateral pairs = 18 named regions, not 27 boxes."""
    boxes = mirror.anatomy_boxes()
    assert len(boxes) == 18
    assert len(set(boxes)) == len(boxes)


def test_the_feature_set_is_whole_plus_bands_plus_regions():
    names = mirror.feature_names()
    assert names[0] == mirror.WHOLE
    assert sum(n.startswith(mirror.BAND_PREFIX) for n in names) == 3
    assert sum(n.startswith(mirror.REGION_PREFIX) for n in names) == 18
    assert len(names) == 22


def test_the_feature_order_is_stable():
    assert mirror.feature_names() == mirror.feature_names()


def test_the_names_match_what_is_computed():
    """A hand-maintained name list would be a second place for the region set to
    live, and the two would drift."""
    assert sorted(mirror.mirror_features(blank())) == sorted(mirror.feature_names())


def test_the_feature_matrix_has_one_row_per_image():
    images = np.stack([blank(), with_blob(0.5, 0.25), with_blob(0.8, 0.3)])
    rows, names = mirror.feature_matrix(images)
    assert rows.shape == (3, len(names))
    assert rows[0].max() == 0.0, "the symmetric image scores zero on every feature"
    assert rows[1].max() > 0.0


def test_the_feature_matrix_is_deterministic():
    images = np.stack([with_blob(0.5, 0.25), with_blob(0.8, 0.3)])
    first, _ = mirror.feature_matrix(images)
    second, _ = mirror.feature_matrix(images)
    assert np.array_equal(first, second)


def test_a_custom_band_set_flows_through():
    """Bands are configuration, not constants -- the same rule the patch scheme
    follows, so moving them moves both instruments together."""
    config = PatchConfig(bands=(BandSpec("upper", 0.0, 0.5, 3), BandSpec("lower", 0.5, 1.0, 3)))
    names = mirror.feature_names(config)
    assert f"{mirror.BAND_PREFIX}upper" in names
    assert f"{mirror.BAND_PREFIX}top" not in names


# --------------------------------------------------------------------------
# the record
# --------------------------------------------------------------------------


def test_describe_records_what_was_computed():
    described = mirror.describe()
    assert described["geometry"] == "g2"
    assert described["statistic"] == "mean_absolute_difference"
    assert described["n_features"] == 22
    assert described["mirror_pairs_collapsed"] is True
    assert "not tuned against the label" in described["note"]


# --------------------------------------------------------------------------
# the whole arm, end to end
# --------------------------------------------------------------------------


def planted_cohort(root, n=40):
    """Staged G2 images whose asymmetry grows with the label.

    The signal is planted in the geometry, so an arm that recovers it has
    recovered the thing the index claims to measure -- not an artefact of the
    plumbing.
    """
    from cleft.geometry import mirror as M

    manifest_dir = root / "cleft_v1"
    staged_dir = root / "staged_v1"
    manifest_dir.mkdir(parents=True)
    staged_dir.mkdir(parents=True)

    rng = np.random.default_rng(0)
    truth = np.clip(rng.normal(2.6, 0.8, size=n), 1.0, 5.0)

    header = (
        "patient_id,frontal_id,basal_id,mean,median,mode,weighted_mean,"
        "orthodontist,soft_1,soft_2,soft_3,soft_4,soft_5,class3,fold"
    )
    lines = ["# CLUSTER-ONLY: patient-keyed", header]
    for index in range(n):
        pid = index + 1
        value = truth[index]
        lines.append(
            f"{pid},{1000 + pid},,{value:.4f},{value:.4f},{value:.4f},{value:.4f},"
            f"{value:.4f},0.2,0.2,0.2,0.2,0.2,{index % 3},{index % 5}"
        )
    (manifest_dir / "manifest.csv").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )

    # A one-sided patch in the alar_base region, brighter for worse grades.
    x, y, w, h = M.anatomy_boxes()["alar_base"]
    images = np.zeros((n, SIZE, SIZE, 3), dtype=np.uint8)
    for index, value in enumerate(truth):
        image = blank()
        row, column = int((y + h / 2) * SIZE), int((x + w / 2) * SIZE)
        image[row - 8 : row + 8, column - 8 : column + 8] = int(
            np.clip(128 + value * 25, 0, 255)
        )
        images[index] = image

    np.save(staged_dir / "staged_patient_g2.npy", images)
    np.save(staged_dir / "staged_patient_g1.npy", images)

    geometry = ["# CLUSTER-ONLY: patient-keyed geometry", "patient_id,aspect_ratio"]
    geometry += [f"{pid},0.74" for pid in range(1, n + 1)]
    (staged_dir / "geometry.csv").write_text(
        "\n".join(geometry) + "\n", encoding="utf-8"
    )
    return manifest_dir, staged_dir


def test_the_mirror_arm_runs_end_to_end_and_recovers_a_planted_signal(tmp_path):
    """Laptop exit criterion: the arm reaches the harness through the same door
    the probe does, and the index recovers an asymmetry planted in the geometry."""
    from cleft.train import phase3

    manifest_dir, staged_dir = planted_cohort(tmp_path)
    result = phase3.run(
        manifest_dir=manifest_dir,
        staged_dir=staged_dir,
        geometry="g2",
        backbone="ridge",
        log=lambda *_: None,
    )

    assert result.summary["features"]["features"] == "mirror_difference_index"
    assert result.summary["features"]["n_features"] == 22
    assert result.summary["oof"]["pcc"] > 0.5, "the planted signal is recoverable"
    assert result.summary["sanity"]["pcc_positive"] is True


def test_the_mirror_arm_reports_its_own_parameter_count(tmp_path):
    """23 parameters -- 22 weights and a bias -- not a fraction of a backbone it
    does not have."""
    from cleft.train import phase3

    manifest_dir, staged_dir = planted_cohort(tmp_path)
    result = phase3.run(
        manifest_dir=manifest_dir,
        staged_dir=staged_dir,
        geometry="g2",
        backbone="ridge",
        log=lambda *_: None,
    )
    parameters = result.summary["parameters"]
    assert parameters["trainable_parameters"] == 23
    assert parameters["backbone"] == "ridge"
    assert parameters["components"]["frozen_feature_extractor"]["parameters"] == 0


def test_the_arm_refuses_to_run_at_g1(tmp_path):
    """The geometry requirement reaches all the way out to the run."""
    from cleft.train import phase3

    manifest_dir, staged_dir = planted_cohort(tmp_path)
    with pytest.raises(mirror.MirrorError):
        phase3.run(
            manifest_dir=manifest_dir,
            staged_dir=staged_dir,
            geometry="g1",
            backbone="ridge",
            log=lambda *_: None,
        )


def test_describe_carries_no_patient_data():
    """SHAREABLE. Region names and counts only."""
    described = mirror.describe()
    assert set(described) == {
        "index", "geometry", "statistic", "n_features",
        "bands", "regions", "mirror_pairs_collapsed", "note",
    }
