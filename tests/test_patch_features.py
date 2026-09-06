"""Patch extraction for the Phase 4 frozen probes (§3.3, §3.4).

The backbone runs only on the cluster; what is testable here is everything up to
it -- that the right patches are generated, that each one is placed through the
patient's own content box rather than a fixed one, that pooling does what it
says, and that the arm cannot silently be built at the wrong geometry.

Synthetic fixtures throughout.
"""

from __future__ import annotations

import numpy as np
import pytest

from cleft.geometry import patch_features as PF
from cleft.geometry.patches import PatchConfig
from cleft.geometry.trapezium import DEFAULT as G1_TRAPEZIUM

SIZE = 224


def image(value: int = 128) -> np.ndarray:
    return np.full((SIZE, SIZE, 3), value, dtype=np.uint8)


def row(aspect_ratio: float = 0.74) -> dict:
    """One geometry.csv row. Narrow crops get a tall, narrow content box."""
    if aspect_ratio <= 1.0:
        width = int(round(SIZE * aspect_ratio))
        height = SIZE
    else:
        width = SIZE
        height = int(round(SIZE / aspect_ratio))
    return {
        "patient_id": "1",
        "aspect_ratio": str(aspect_ratio),
        "content_x": str((SIZE - width) // 2),
        "content_y": str((SIZE - height) // 2),
        "content_w": str(width),
        "content_h": str(height),
    }


# --------------------------------------------------------------------------
# geometry and scheme selection
# --------------------------------------------------------------------------


def test_g2_is_the_full_square():
    """The mask IS the square at G2, which is why the scheme comparison runs
    there -- every scheme then sees zero background."""
    assert PF.G2_TRAPEZIUM.top_half_width == 0.5
    assert PF.G2_TRAPEZIUM.bot_half_width == 0.5


def test_g2_matches_the_contact_sheet_definition():
    """Two modules construct this; they must agree or the probe and the sheet
    would be describing different geometries."""
    from cleft.geometry import contact

    assert PF.G2_TRAPEZIUM == contact.G2_TRAPEZIUM


def test_g1_is_the_phase_2_trapezium():
    assert PF.trapezium_for("g1") == G1_TRAPEZIUM


def test_an_unknown_geometry_is_rejected():
    with pytest.raises(PF.PatchFeatureError, match="unknown geometry"):
        PF.trapezium_for("g3")


@pytest.mark.parametrize("scheme", ("grid", "anatomy", "random"))
def test_every_scheme_generates_patches_at_both_geometries(scheme):
    for geometry in PF.GEOMETRIES:
        patches = PF.patches_for(scheme, geometry)
        assert len(patches) == 27, f"{scheme} at {geometry}"


def test_the_whole_scheme_has_no_patches_and_says_why():
    with pytest.raises(PF.PatchFeatureError, match="no patches"):
        PF.patches_for("whole", "g2")


def test_an_unknown_scheme_is_rejected():
    with pytest.raises(PF.PatchFeatureError, match="unknown scheme"):
        PF.patches_for("saliency", "g2")


def test_the_schemes_come_from_the_shared_registry():
    """Not a private copy -- the probe must use the set Phase 2 measured,
    rendered and froze."""
    from cleft.geometry import patches as patch_module

    for scheme in ("grid", "anatomy", "random"):
        assert patch_module.get(scheme) is not None


def test_the_anatomy_scheme_is_the_frozen_one():
    """Pinned through the probe as well, so a probe run cannot quietly use a
    differently-parameterised anatomy set from the ablation it feeds."""
    from cleft.geometry.generators import ACCEPTED_ANATOMY

    patches = PF.patches_for("anatomy", "g2")
    assert sum(1 for p in patches if p.clamped) == ACCEPTED_ANATOMY["n_clamped"]


# --------------------------------------------------------------------------
# the per-image mapping -- the load-bearing part
# --------------------------------------------------------------------------


def test_the_content_box_is_read_from_the_geometry_row():
    staged = PF.staged_from_row(image(), row(0.74))
    assert staged.content_box == (29, 0, 166, 224)


def test_a_geometry_row_without_a_content_box_is_refused_with_the_reason():
    with pytest.raises(PF.PatchFeatureError, match="content box per patient"):
        PF.staged_from_row(image(), {"patient_id": "1", "aspect_ratio": "0.74"})


def test_patch_pixels_vary_with_aspect_ratio():
    """The Phase 2 defect, re-asserted on the Phase 4 path.

    Two patients with different aspect ratios must receive different pixel boxes
    for the same normalised patch. If they did not, the same anatomy would sit at
    a different place in each and every arm would still produce a number.
    """
    patches = PF.patches_for("grid", "g2")
    narrow = PF.extract_for_patient(image(), row(0.55), patches, output_size=16)
    wide = PF.extract_for_patient(image(), row(1.10), patches, output_size=16)
    assert narrow.shape == wide.shape

    from cleft.geometry.mapping import map_patches

    narrow_boxes = [m.pixels for m in map_patches(PF.staged_from_row(image(), row(0.55)), patches)]
    wide_boxes = [m.pixels for m in map_patches(PF.staged_from_row(image(), row(1.10)), patches)]
    assert narrow_boxes != wide_boxes


def test_extraction_returns_one_patch_per_scheme_member():
    patches = PF.patches_for("grid", "g2")
    stack = PF.extract_for_patient(image(), row(), patches, output_size=32)
    assert stack.shape == (27, 32, 32, 3)


def test_extraction_is_deterministic():
    patches = PF.patches_for("random", "g2")
    first = PF.extract_for_patient(image(), row(), patches, output_size=16)
    second = PF.extract_for_patient(image(), row(), patches, output_size=16)
    assert np.array_equal(first, second)


def test_the_generator_yields_one_stack_per_patient():
    images = np.stack([image(100), image(150), image(200)])
    rows = [row(0.6), row(0.8), row(1.0)]
    patches = PF.patches_for("grid", "g2")
    stacks = list(PF.iter_patient_patches(images, rows, patches, output_size=8))
    assert len(stacks) == 3
    assert all(s.shape == (27, 8, 8, 3) for s in stacks)


def test_a_length_mismatch_is_refused():
    images = np.stack([image(), image()])
    with pytest.raises(PF.PatchFeatureError, match="row for row"):
        list(PF.iter_patient_patches(images, [row()], PF.patches_for("grid", "g2")))


def test_patches_carry_the_pixels_they_cover():
    """A patch of a uniform image is uniform; a patch of a striped one is not.
    Cheap, but it is the difference between extracting pixels and extracting
    zeros."""
    striped = image()
    striped[:112] = 20
    patches = PF.patches_for("grid", "g2")
    stack = PF.extract_for_patient(striped, row(1.0), patches, output_size=16)
    means = [float(p.mean()) for p in stack]
    assert max(means) - min(means) > 50


# --------------------------------------------------------------------------
# pooling
# --------------------------------------------------------------------------


def test_mean_pooling_averages_over_patches():
    embeddings = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    assert PF.pool(embeddings, "mean").tolist() == [2.0, 3.0]


def test_max_pooling_takes_the_maximum():
    embeddings = np.array([[1.0, 9.0], [3.0, 4.0]], dtype=np.float32)
    assert PF.pool(embeddings, "max").tolist() == [3.0, 9.0]


def test_pooling_collapses_the_patch_axis():
    embeddings = np.zeros((27, 768), dtype=np.float32)
    assert PF.pool(embeddings).shape == (768,)


def test_concat_is_refused_and_says_why():
    """Not an option to try later: 27x768 features for 237 patients would need a
    different claim procedure, not a different config value."""
    with pytest.raises(PF.PatchFeatureError, match="excluded by design"):
        PF.pool(np.zeros((27, 768), dtype=np.float32), "concat")


def test_pooling_is_order_independent_for_mean():
    """Mean pooling discards patch identity entirely -- which is the limitation
    these arms carry, demonstrated rather than described."""
    embeddings = np.random.default_rng(0).normal(size=(27, 16)).astype(np.float32)
    shuffled = embeddings[::-1].copy()
    assert PF.pool(embeddings) == pytest.approx(PF.pool(shuffled), abs=1e-6)


# --------------------------------------------------------------------------
# the record
# --------------------------------------------------------------------------


def test_describe_records_the_scheme_and_the_geometry():
    patches = PF.patches_for("anatomy", "g2")
    described = PF.describe("anatomy", "g2", patches)
    assert described["scheme"] == "anatomy"
    assert described["geometry"] == "g2"
    assert described["n_patches"] == 27
    assert described["pooling"] == "mean"


def test_describe_carries_the_pooling_limitation():
    """A reader comparing a patch arm to the whole-image bar must not conclude
    "region structure does not help" from "mean-pooled patches did not win"."""
    note = PF.describe("grid", "g2", PF.patches_for("grid", "g2"))["pooling_note"]
    assert "SR-GNN" in note
    assert "NOT that region structure does not help" in note


def test_the_limitation_does_not_extend_to_the_scheme_comparison():
    """Pooling is identical across grid, anatomy and random, so a difference
    between them IS a difference in placement. Q4's placement half is answered
    cleanly here; only its relational half defers to Phase 7. A reader who
    discounts the scheme comparison along with the absolute numbers has thrown
    away the result these three arms exist to produce."""
    note = PF.describe("grid", "g2", PF.patches_for("grid", "g2"))["pooling_note"]
    assert "identical across all three" in note
    assert "PLACEMENT" in note

    # And the property the claim rests on: the three arms really do pool alike.
    pooling = {
        PF.describe(scheme, "g2", PF.patches_for(scheme, "g2"))["pooling"]
        for scheme in ("grid", "anatomy", "random")
    }
    assert len(pooling) == 1


def test_describe_carries_no_patient_data():
    described = PF.describe("grid", "g2", PF.patches_for("grid", "g2"))
    assert "patient_id" not in repr(described)


def test_coverage_is_reported_so_a_g1_arm_shows_its_background():
    """At G1 the outer patches contain white. The report says how much rather
    than leaving a scheme comparison to be read as if it were clean."""
    g1 = PF.describe("grid", "g1", PF.patches_for("grid", "g1"))
    g2 = PF.describe("grid", "g2", PF.patches_for("grid", "g2"))
    assert g1["min_coverage"] < 1.0
    assert g2["min_coverage"] == 1.0


def test_the_backbone_input_size_is_the_backbones_not_the_contact_sheets():
    """PatchConfig.output_size (64) sizes the Phase 2 sheet. Feeding 64px crops
    to a 224px ViT would just make the model interpolate them, less well."""
    assert PF.BACKBONE_INPUT_SIZE == 224
    assert PatchConfig().output_size != PF.BACKBONE_INPUT_SIZE
