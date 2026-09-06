"""Masked SCUT variants and the parity measurement (Phase 5 §2).

Synthetic faces and synthetic landmark sets. The real faces are checked on the
contact sheet, which is the only thing that can answer whether the result looks
like a cleft crop.
"""

from __future__ import annotations

import numpy as np
import pytest

from cleft.geometry.staging import CLEFT_STAGED_GEOMETRY, OUTPUT_SIZE, PAD_VALUE
from cleft.geometry.trapezium import EXPECTED_AREA_FRACTION
from cleft.scut import masked, placement
from test_scut_placement import a_face

SOURCE = 350


def a_source_image() -> np.ndarray:
    image = np.zeros((SOURCE, SOURCE, 3), dtype=np.uint8)
    image[:, :] = (40, 60, 80)
    image[60:300, 60:300] = (210, 170, 150)
    return image


# --------------------------------------------------------------------------
# aspect-ratio sampling
# --------------------------------------------------------------------------


def test_sampling_stays_inside_the_measured_cleft_range():
    """**The frozen record's bounds are 4-decimal ROUNDINGS of the raw ratios**,
    so the raw minimum sits just below the recorded one: 0.553067 against 0.5531.

    Under the interpolating sampler the bound held exactly, because it drew from
    the rounded grid itself. The bootstrap draws the raw values, so an exact
    comparison against a rounded constant now fails by 3.3e-5 -- a difference in
    how the two records are stored, not in the data. The tolerance is the
    rounding, and it is named rather than widened until green.
    """
    ratios = CLEFT_STAGED_GEOMETRY["aspect_ratio"]
    rounding = 5e-5
    drawn = masked.sample_aspect_ratios(400, seed=1337)
    assert drawn.min() >= ratios["min"] - rounding
    assert drawn.max() <= ratios["max"] + rounding

    # Against the unrounded source, the bound is exact.
    from cleft.scut.ar_distribution import AR_OBSERVED

    assert drawn.min() >= min(AR_OBSERVED)
    assert drawn.max() <= max(AR_OBSERVED)


def test_sampling_is_reproducible_from_the_seed():
    first = masked.sample_aspect_ratios(50, seed=7)
    assert np.array_equal(first, masked.sample_aspect_ratios(50, seed=7))
    assert not np.array_equal(first, masked.sample_aspect_ratios(50, seed=8))


def test_the_fixed_mode_is_the_special_case():
    """The fixed-median variant is `sampled` with no spread, which is why both
    exist: the question becomes answerable rather than assumed."""
    drawn = masked.sample_aspect_ratios(20, seed=1, mode="fixed_median")
    assert set(np.unique(drawn)) == {CLEFT_STAGED_GEOMETRY["aspect_ratio"]["median"]}


def test_an_unknown_sampling_mode_is_refused():
    with pytest.raises(masked.MaskedError, match="unknown ar sampling"):
        masked.sample_aspect_ratios(4, seed=1, mode="gaussian")


# --------------------------------------------------------------------------
# the variant itself
# --------------------------------------------------------------------------


def test_both_geometries_produce_a_staged_square():
    for geometry in masked.GEOMETRIES:
        variant, _ = masked.build_one(a_source_image(), a_face(), 0.74, geometry)
        assert variant.shape == (OUTPUT_SIZE, OUTPUT_SIZE, 3)


def test_the_caption_grid_is_the_grid_tile_laid_out():
    """**[CORRECTED 2026-08-09, 768 sheet review] Labels landed on
    neighbouring faces.** ``save_sheet`` drew captions at positions computed
    from ``panels[0]``'s dimensions while ``tile`` laid out at the max panel
    size -- two computations of one quantity, and they disagreed the moment
    panels[0] stopped being the largest. ``cell_for`` is the single source
    both use now."""
    from cleft.geometry.render import Panel, RenderError, cell_for

    mixed = [
        Panel("placement", np.zeros((350, 350, 3), dtype=np.uint8)),
        Panel("masked", np.zeros((768, 768, 3), dtype=np.uint8)),
    ]
    assert cell_for(mixed) == (768, 768), (
        "the cell is the MAX, which panels[0] is not"
    )
    assert cell_for(mixed[:1]) == (350, 350)
    with pytest.raises(RenderError, match="no panels"):
        cell_for([])

    # And no call site builds the caption grid from panels[0] any more.
    import inspect

    from cleft import run as run_module

    assert "panels[0].image.shape" not in inspect.getsource(run_module)


def test_the_placement_panel_fills_its_cell_at_the_build_size():
    """**[CORRECTED 2026-08-09, 768 sheet review]** The 350px annotated
    source floated in a 768 cell of empty grey. It now renders at the build
    size via the frozen nearest resize -- blocky-sharp, so the masked
    panels' smoothing beside it stays the visible interpolation."""
    native = masked.placement_panel(a_source_image(), a_face(), 0.74)
    assert native.shape == (OUTPUT_SIZE, OUTPUT_SIZE, 3)
    big = masked.placement_panel(a_source_image(), a_face(), 0.74, size=768)
    assert big.shape == (768, 768, 3)


def test_size_threads_the_same_frozen_composition_to_a_new_resolution():
    """**[2026-08-09] Road B's resolution axis through masked SCUT.** The
    default stays byte-identical to the composition the bit-for-bit test
    pins; a declared size runs the SAME path at the new resolution -- stage
    already took the argument, the Phase 2 reuse pattern."""
    for geometry in masked.GEOMETRIES:
        default, _ = masked.build_one(a_source_image(), a_face(), 0.74, geometry)
        explicit, _ = masked.build_one(
            a_source_image(), a_face(), 0.74, geometry, size=OUTPUT_SIZE
        )
        assert np.array_equal(default, explicit), (
            "an explicit 224 must be the default, byte for byte"
        )
        resized, record = masked.build_one(
            a_source_image(), a_face(), 0.74, geometry, size=512
        )
        assert resized.shape == (512, 512, 3), geometry
        assert record["geometry"] == geometry
    # G2 still unwarps the staged square at the new size: the corner block
    # that G1 keeps white is reclaimed, exactly as at 224.
    g1, _ = masked.build_one(a_source_image(), a_face(), 1.0, "g1", size=512)
    g2, _ = masked.build_one(a_source_image(), a_face(), 1.0, "g2", size=512)
    corner_g1 = np.all(g1[:40, :40] == PAD_VALUE, axis=2)
    assert corner_g1.all(), "G1 keeps its corner white at 512"
    corner_g2 = np.all(g2[:40, :40] == PAD_VALUE, axis=2)
    assert corner_g2.mean() < 0.5, "G2 reclaims the corner at 512"


def test_g1_leaves_white_corners_and_g2_reclaims_them():
    """The cleft crops ARRIVED with white corners baked into the CONTENT; SCUT
    faces do not, so ``apply_arrival_mask`` bakes them before staging. G2
    unwarps the staged square and the corner PIXELS become content.

    Tested at AR 1.0 deliberately. At other ratios ``stage`` letterboxes the
    content, and that white padding sits INSIDE the trapezium and is stretched
    along with everything else. **That is matched behaviour, and it is now
    measured rather than argued**: the frozen cleft path itself stages then
    unwarps (stage_build.py), so a cleft crop at the same ratio pads
    identically and its pad reaches G2 identically -- the zero-white
    "invariant" was a patch-geometry constant misread as a pixel fact. See
    masked.G2_WHITE_DEFECT. A square crop removes the pad so this test
    measures the corners alone.
    """
    g1, _ = masked.build_one(a_source_image(), a_face(), 1.0, "g1")
    g2, _ = masked.build_one(a_source_image(), a_face(), 1.0, "g2")

    corner_g1 = np.all(g1[:20, :20] == PAD_VALUE, axis=2)
    corner_g2 = np.all(g2[:20, :20] == PAD_VALUE, axis=2)
    assert corner_g1.all(), "G1 keeps its top-left corner block white"
    # Not a single-pixel assertion: unwarp can leave a few boundary pixels
    # white where the baked edge rasterises through resize_nearest -- the
    # frozen path does the identical thing to a cleft crop of the same size
    # (measured: 3 sliver pixels on this fixture). The semantics live at
    # neighbourhood scale: the block flips from all-white to mostly content.
    assert corner_g2.mean() < 0.5, (
        "G2 stretches rows to full width, so the corner block is mostly content"
    )


def test_the_mask_covers_the_recorded_trapezium_fraction():
    """Taken from the frozen trapezium, not restated here -- a second copy of
    the mask is where the two domains would diverge."""
    from cleft.geometry import trapezium

    inside = trapezium.mask(OUTPUT_SIZE)
    assert inside.mean() == pytest.approx(EXPECTED_AREA_FRACTION, abs=0.005)


def test_the_arrival_mask_is_the_frozen_mask_on_a_square():
    """``apply_arrival_mask`` rasterises the frozen trapezium in CONTENT
    coordinates. On a square content it must reproduce ``trapezium.mask``
    pixel-for-pixel -- the pin that stops the content-frame rasterisation
    drifting from the frozen square one."""
    from cleft.geometry import trapezium

    for size in (64, 97, OUTPUT_SIZE):
        content = np.full((size, size, 3), 33, dtype=np.uint8)
        arrived = masked.apply_arrival_mask(content)
        white = np.all(arrived == PAD_VALUE, axis=2)
        assert np.array_equal(white, ~trapezium.mask(size))


def test_an_unknown_geometry_is_refused():
    with pytest.raises(masked.MaskedError, match="unknown geometry"):
        masked.build_one(a_source_image(), a_face(), 0.74, "g3")


def test_a_box_that_lands_outside_the_image_is_refused_not_silently_empty():
    points = a_face(midline=-500.0)
    with pytest.raises(masked.MaskedError, match="leaves nothing inside"):
        masked.build_one(a_source_image(), points, 0.74, "g1")


def test_the_record_carries_both_the_requested_and_the_realised_ratio():
    """**They agree while the box fits and diverge the moment it is clipped.**
    Parity measured against the requested value would be measuring an intention
    rather than an output."""
    _, record = masked.build_one(a_source_image(), a_face(), 0.74, "g1")
    assert record["aspect_ratio_requested"] == pytest.approx(0.74)
    assert record["aspect_ratio"] == pytest.approx(0.74, abs=0.02)


def test_a_clipped_box_shows_the_two_ratios_diverging():
    """The case the pair exists for. A box running off the edge is clipped, so
    what was produced is not what was asked for -- and the record says so."""
    points = a_face(midline=40.0)
    _, record = masked.build_one(a_source_image(), points, 1.09, "g1")
    assert record["box_inside_frame"] is False
    assert record["aspect_ratio"] != record["aspect_ratio_requested"]


# --------------------------------------------------------------------------
# parity, measured rather than claimed
# --------------------------------------------------------------------------


def some_records(n: int = 6, geometry: str = "g1") -> list[dict]:
    image = a_source_image()
    ratios = masked.sample_aspect_ratios(n, seed=1337)
    return [
        masked.build_one(image, a_face(), float(ratio), geometry)[1]
        for ratio in ratios
    ]


def test_the_parity_report_compares_against_the_recorded_cleft_figures():
    report = masked.parity_report(some_records())
    assert report["cleft_reference"] == CLEFT_STAGED_GEOMETRY
    assert set(report["gaps"]) == {"aspect_ratio", "pad_fraction"}
    assert report["gaps"]["aspect_ratio"]["cleft"]["median"] == 0.7404


def test_the_parity_report_gives_gaps_not_a_verdict():
    """No pass/fail: what counts as close enough on a domain gap is a judgement
    about pretraining transfer, not a threshold justifiable here."""
    report = masked.parity_report(some_records())
    assert "pass" not in report
    assert "verdict" not in report
    assert "NO PASS/FAIL" in report["note"]
    assert report["gaps"]["aspect_ratio"]["median_gap"] is not None


def test_the_report_states_the_expected_white_fractions():
    """Derived from the recorded cleft constants, so the produced numbers have
    something correct to be read against. G1 is pad plus the baked trapezium's
    complement of the content -- NOT 0.198, which was the square-relative
    complement, the same wrong frame the build itself had. G2 is the pad
    fraction -- NOT zero, which was a patch-geometry constant misread as a
    pixel fact."""
    report = masked.parity_report(some_records())
    expected = report["expected_white_fraction"]
    pad_mean = CLEFT_STAGED_GEOMETRY["pad_fraction"]["mean"]
    area = CLEFT_STAGED_GEOMETRY["trapezium"]["area_of_content"]
    assert expected["g1_at_cleft_mean_pad"] == pytest.approx(
        1.0 - area * (1.0 - pad_mean), abs=1e-6
    )
    assert expected["g2_at_cleft_mean_pad"] == pytest.approx(pad_mean, abs=1e-6)
    assert expected["g2_at_cleft_mean_pad"] > 0.0


def test_g2_has_less_white_than_g1():
    g1 = masked.parity_report(some_records(geometry="g1"))
    g2 = masked.parity_report(some_records(geometry="g2"))
    assert (
        g2["produced"]["white_fraction"]["median"]
        < g1["produced"]["white_fraction"]["median"]
    )


def test_the_report_counts_clipped_boxes():
    """Clipping puts padding INSIDE the content box, which no cleft crop has."""
    report = masked.parity_report(some_records())
    assert report["n_box_outside_frame"] == 0


def test_an_empty_record_list_is_refused():
    """A parity report over nothing would be a green with no evidence."""
    with pytest.raises(masked.MaskedError, match="no records"):
        masked.parity_report([])


# --------------------------------------------------------------------------
# the sheet
# --------------------------------------------------------------------------


def test_the_placement_annotation_marks_the_face_without_destroying_it():
    image = a_source_image()
    annotated = masked.annotate_placement(image, a_face(), 0.74)
    assert annotated.shape == image.shape
    assert not np.array_equal(annotated, image)


def test_the_cleft_reference_panel_is_drawn_from_the_record():
    """So "does this look like a cleft crop" is a visual comparison rather than
    a memory test. No cleft image is used or needed."""
    panel = masked.cleft_reference_panel()
    assert panel.shape == (OUTPUT_SIZE, OUTPUT_SIZE, 3)
    assert panel.dtype == np.uint8
    # It carries the trapezium: some pixels are neither pad nor content fill.
    assert len(np.unique(panel.reshape(-1, 3), axis=0)) >= 3


def test_levelling_is_off_in_the_summary_path():
    """Recorded with the run, because the cleft crops are not levelled and this
    matched property is a decision someone should see in metrics.json."""
    assert placement.SCUT_HEAD_TILT_LEVELLING == "off"


# --------------------------------------------------------------------------
# THE PARITY INVARIANT: build_one IS the frozen cleft composition
# --------------------------------------------------------------------------
#
# The earlier invariant here -- "G2 has exactly zero white" -- was a strict
# xfail waiting on a fix. It came off because the invariant itself was wrong:
# p2-stage-1's white_fraction_max is 1 - min(patch coverage) against the
# full-square G2 trapezium, 0.0 BY CONSTRUCTION, a patch-geometry quantity with
# no pixels in it. The frozen cleft path (stage, then unwarp the padded square)
# was replayed and measured: G2 white ~= the pad fraction, 0.259 at the median
# AR, never zero. The executable invariant is therefore IDENTITY with that
# composition, which is strictly stronger than any white-fraction assertion:
# it pins the amount, the pattern, and the left/right structure all at once.
# See masked.G2_WHITE_DEFECT for the full record.


def frozen_composition(content: np.ndarray, geometry: str) -> np.ndarray:
    """The cleft path, restated in this file from FROZEN parts only.

    Bake the content-relative trapezium (the state a cleft crop arrives in),
    ``stage``, and for G2 ``unwarp`` the staged square -- ``stage_build.py``'s
    composition. Deliberately independent of ``masked.apply_arrival_mask``:
    the scalar ``half_width_at`` is used here so the module and the test only
    agree if both match the frozen arithmetic.
    """
    from cleft.geometry import trapezium
    from cleft.geometry.staging import stage

    height, width = content.shape[:2]
    ys = (np.arange(height) + 0.5) / height
    xs = (np.arange(width) + 0.5) / width
    half = np.array([trapezium.DEFAULT.half_width_at(float(y)) for y in ys])
    arrived = content.copy()
    arrived[np.abs(xs[None, :] - trapezium.MIDLINE) > half[:, None]] = PAD_VALUE
    staged = stage(arrived)
    return staged.image if geometry == "g1" else trapezium.unwarp(staged.image)


def test_the_g2_white_record_is_resolved_and_names_the_misread_quantity():
    """The record must carry the rediagnosis: the zero-white invariant was a
    patch-geometry constant misread as a pixel fact, the actual defect was the
    mask's coordinate frame, and the retracted "matched padding" conclusion is
    un-retracted -- the probe, not the code, was the defect (fifth instance)."""
    defect = masked.G2_WHITE_DEFECT
    assert defect["status"] == "RESOLVED"
    assert "BY CONSTRUCTION" in defect["invariant_source"]
    assert "wrong quantity" in defect["invariant_source"]
    assert "content-relative" in defect["actual_defect"]
    assert "order was already matched" in defect["actual_defect"]
    assert "reinstated" in defect["unretracted"]
    assert "MATCHED" in defect["second_defect_resolution"]


def test_build_one_is_the_frozen_cleft_composition_exactly():
    """**The parity invariant, executable.** Bit-identical, both geometries,
    portrait and square -- not a white-fraction tolerance. If any operation is
    added, removed or reordered relative to the frozen cleft path, this fails."""
    image = a_source_image()
    points = a_face()
    for ratio in (0.62, 0.74, 1.0):
        box = placement.crop_box(points, ratio)
        content = masked.crop_content(image, box)
        for geometry in masked.GEOMETRIES:
            variant, _ = masked.build_one(image, points, ratio, geometry)
            assert np.array_equal(variant, frozen_composition(content, geometry)), (
                f"AR {ratio} {geometry}: build_one diverged from the frozen "
                "cleft composition"
            )


def test_g2_white_is_the_cleft_paths_own_not_zero():
    """**[MEASURED 2026-07-30]** The frozen cleft path stretches the pad into
    G2, so a portrait G2 variant's white sits near its pad fraction -- 0.259 at
    the cohort median AR -- and zero was never the cleft behaviour. Asserting
    zero here would re-create the domain gap this module exists to close."""
    _, record = masked.build_one(a_source_image(), a_face(), 0.74, "g2")
    assert record["white_fraction"] > 0.0
    assert record["white_fraction"] == pytest.approx(
        record["pad_fraction"], abs=0.02
    ), "G2 white tracks the pad fraction (plus edge rasterisation slivers)"


def test_white_is_reported_per_quadrant_so_asymmetry_is_visible():
    """An aggregate white fraction hides a left/right difference completely.
    The gap is matched behaviour (the frozen path produces it too -- see below),
    and per-quadrant reporting is what lets the sheet review confirm the
    magnitude instead of discovering it."""
    _, record = masked.build_one(a_source_image(), a_face(), 0.74, "g1")
    assert set(record["white_by_quadrant"]) == {
        "top_left", "top_right", "bottom_left", "bottom_right",
    }
    assert record["white_left_right_gap"] >= 0.0


def test_the_left_right_gap_is_the_frozen_paths_own():
    """**[MEASURED 2026-07-30] The gap is MATCHED behaviour, and the odd-pad
    hypothesis was wrong.**

    The frozen replay shows the same left/right white gap on 17 of 23 cohort
    ARs (max 0.0089 G1 / 0.0114 G2) -- including at even 29/29 pads and at
    AR 1.0 with no pad at all, which the odd-pad-split explanation cannot
    produce. The source is ``resize_nearest``'s floor-based index mapping,
    left-biased by half a source pixel, plus edge rasterisation.

    Since build_one IS the frozen composition (asserted bit-identical above),
    its gap is the cleft path's gap at the same content geometry. Asserted
    here through the independent quadrant arithmetic so the record's numbers,
    not just the pixels, are pinned. Real cleft arrays' per-patient figures
    remain cluster-only; the canonical-geometry replay is the local ground
    truth, and hand-made per-patient baked masks vary around it.
    """
    image = a_source_image()
    points = a_face()
    _, record = masked.build_one(image, points, 0.74, "g1")

    replay = frozen_composition(
        masked.crop_content(image, placement.crop_box(points, 0.74)), "g1"
    )
    white = np.all(replay == PAD_VALUE, axis=2)
    height, width = white.shape
    top, left = height // 2, width // 2
    left_white = (white[:top, :left].mean() + white[top:, :left].mean())
    right_white = (white[:top, left:].mean() + white[top:, left:].mean())
    replay_gap = abs(left_white - right_white) / 2.0

    assert record["white_left_right_gap"] == pytest.approx(replay_gap, abs=1e-6)


def test_white_fraction_counts_content_white_too_and_says_so():
    """**[MEASURED 2026-07-30] The metric is named for pad-and-mask and computes
    ALL pure-white pixels**, so a face photographed on a white background counts
    as mask. Demonstrated directly: the same crop geometry with a white patch in
    the content reports more white, and the extra is measured separately."""
    points = a_face()
    plain = a_source_image()
    white_backed = a_source_image()

    # Inside the crop box and on ONE side of it, derived rather than guessed --
    # a patch placed by eye landed outside the box and the test passed nothing.
    x, y, w, h = placement.crop_box(points, 0.74)
    x0, y0 = int(round(x)), int(round(y))
    white_backed[y0 : y0 + int(h * 0.4), x0 : x0 + int(w * 0.3)] = PAD_VALUE

    _, plain_record = masked.build_one(plain, points, 0.74, "g1")
    _, white_record = masked.build_one(white_backed, points, 0.74, "g1")

    assert white_record["white_fraction"] > plain_record["white_fraction"]
    assert plain_record["white_content_fraction"] == 0.0
    assert white_record["white_content_fraction"] > 0.0, (
        "content white must be measurable separately, or the two whites stay "
        "conflated inside one number"
    )
    # One-sided content white moves the gap: that is the tail mechanism.
    assert (
        white_record["white_left_right_gap"] > plain_record["white_left_right_gap"]
    )


def test_the_content_white_finding_is_recorded_with_its_correlation():
    """The tail of the left/right gap is image content, not geometry: the
    geometry-only gap maxes at 0.0089 (matching the frozen replay) while the
    build reported 0.046, and content white explains the excess at r=0.977."""
    finding = masked.WHITE_IS_ANY_WHITE_PIXEL
    assert finding["geometry_only_gap_max"]["g1"] == pytest.approx(0.00893)
    assert finding["frozen_replay_gap_max"]["g1"] == pytest.approx(0.0089, abs=1e-4)
    assert finding["build_reported_gap_max"]["g1"] == pytest.approx(0.046)
    assert finding["correlation_content_white_to_excess_gap"] > 0.97
    assert "TAIL only" in finding["affects"]
    assert "named for pad and mask" in finding["quantity_confusion"]


def test_the_bimodal_gap_distribution_is_recorded_as_bimodal_not_skewed():
    """**Mean below median is BIMODALITY here, not left skew.** 44% of faces sit
    below 0.0004 and 43% above 0.0086. Reading mean-minus-median as a skew
    statistic would describe a shape this distribution does not have -- the same
    error as any summary answering the wrong question about its data."""
    finding = masked.GAP_DISTRIBUTION_IS_BIMODAL
    assert finding["n_below_0_0004"] + finding["n_above_0_0086"] > 4700
    assert finding["mean"] < finding["median"], "the pair this explains"
    assert "not left skew" in finding["why_mean_below_median"]
    assert "resize_nearest" in finding["mechanism"]


def test_the_parity_report_separates_content_white_from_geometry_white():
    report = masked.parity_report(some_records())
    assert "white_content_fraction" in report
    assert "n_faces_with_white_content" in report
    assert report["white_is_any_white_pixel"]["correlation_content_white_to_excess_gap"]
    assert report["gap_distribution_is_bimodal"]["mean"] < (
        report["gap_distribution_is_bimodal"]["median"]
    )


def test_the_parity_report_surfaces_the_resolution_and_the_asymmetry_count():
    report = masked.parity_report(some_records())
    assert report["g2_white_defect"]["status"] == "RESOLVED"
    assert "n_left_right_asymmetric" in report
    assert "white_left_right_gap" in report


def test_the_parity_report_compares_ar_against_the_cohort_shape():
    """Not just the range: the empirical sampler exists because uniform gives
    mean 0.826 against the cohort's 0.7475."""
    report = masked.parity_report(some_records(n=40))
    assert report["ar_vs_cohort"]["cleft"]["mean"] == 0.7475
    assert "mean_gap" in report["ar_vs_cohort"]


def test_the_default_sampler_is_the_bootstrap_over_observed_ratios():
    """**[2026-07-30] Moved from `empirical` once the 237 raw ratios landed.**

    The interpolating sampler realised 2.75% landscape against the cohort's 0.42%
    over the full build; the bootstrap reproduces the cohort exactly. Both older
    modes stay available so earlier runs can be re-derived -- a superseded sampler
    is not a deleted one, or the runs it produced become unreproducible.
    """
    assert masked.DEFAULT_AR_SAMPLING == "observed"
    for superseded in ("empirical", "uniform_range", "fixed_median"):
        assert superseded in masked.AR_SAMPLING, (
            f"{superseded} is kept so the runs made with it can be re-derived"
        )


def test_every_shipped_scut_config_uses_the_corrected_sampler(repo_root):
    """The constant, the default and the CONFIGS must move together. A config
    left on `empirical` would rebuild the artifact with the known-defective tail
    while everything else said the fix was in."""
    import yaml

    for name in ("p5_masked_scut.yaml", "p5_asymmetry_synthesis.yaml"):
        payload = yaml.safe_load(
            (repo_root / "configs" / name).read_text(encoding="utf-8")
        )
        assert payload["task"]["ar_sampling"] == "observed", (
            f"{name} still samples with the interpolating sampler"
        )


# --------------------------------------------------------------------------
# the task: the artifact is WRITTEN, resumable, and the silent mode refuses
# [2026-07-31] All three p5_masked_scut runs produced only the sheet and
# metrics.json -- parity measured, sheet approved, and data/scut/masked_v1
# never existed on any machine, found when the p6 pretraining configs declared
# it as an input. These tests run the TASK, not the geometry.
# --------------------------------------------------------------------------


def a_scut_root(tmp_path, stems=("AF1", "AF2", "AM1", "AM2", "CF1", "CF2")):
    """A tiny on-disk SCUT root the task can consume end to end."""
    from PIL import Image

    from cleft.scut.dataset import IMAGE_DIR, LANDMARK_DIR
    from test_scut import pts_bytes

    root = tmp_path / "scut"
    (root / IMAGE_DIR).mkdir(parents=True)
    (root / LANDMARK_DIR).mkdir(parents=True)
    for stem in stems:
        Image.fromarray(a_source_image()).save(root / IMAGE_DIR / f"{stem}.jpg")
        (root / LANDMARK_DIR / f"{stem}.pts").write_bytes(pts_bytes(a_face()))
    return root


def masked_task(**overrides) -> dict:
    task = {
        "kind": "masked_scut",
        "scut_root": "scut_root",
        "geometries": ["g1", "g2"],
        "n_faces": 0,
        "n_sheet": 2,
        "ar_sampling": "observed",
        "write_artifact": True,
        "out_version": "masked_test_v1",
        "checkpoint_every": 2,
    }
    task.update(overrides)
    return task


def run_masked_task(config_path, out_root, repo_root):
    from cleft.provenance import RunContext
    from cleft.run import TASKS

    with RunContext(config_path, out_root, repo_root=repo_root) as ctx:
        TASKS["masked_scut"](ctx)
        return ctx.run_dir


def declare_root(root):
    from cleft.provenance import hash_dir

    return [{
        "name": "scut_root",
        "path": str(root),
        "rollup_sha256": hash_dir(root)["rollup"],
    }]


def test_the_full_build_writes_a_versioned_hashed_artifact(
    tmp_path, clean_repo, out_root
):
    """Exit criterion 10, finally: arrays, index, MANIFEST.json whose hashes
    verify, and metrics.json carrying the rollup a consuming config declares."""
    import json

    from cleft.provenance import hash_dir
    from cleft.provenance.hashing import hash_file
    from fixtures import builders

    root = a_scut_root(tmp_path)
    config = builders.write_config(
        tmp_path / "cfg.yaml", phase="p5",
        inputs=declare_root(root), task=masked_task(),
    )
    run_dir = run_masked_task(config, out_root, clean_repo)

    artifact = clean_repo / "data" / "scut" / "masked_test_v1"
    assert artifact.is_dir(), "the artifact was not published"
    assert not artifact.with_name("masked_test_v1.inprogress").exists()

    faces = json.loads((artifact / "faces.json").read_text(encoding="utf-8"))
    assert [face["stem"] for face in faces] == sorted(
        ("AF1", "AF2", "AM1", "AM2", "CF1", "CF2")
    )
    for geometry in ("g1", "g2"):
        array = np.load(artifact / f"masked_{geometry}.npy")
        assert array.shape == (6, OUTPUT_SIZE, OUTPUT_SIZE, 3)
        assert array.dtype == np.uint8

    # The journal and the resume checkpoint are not part of the artifact.
    assert not (artifact / "faces.jsonl").exists()
    assert not (artifact / "checkpoint.npz").exists()

    # MANIFEST.json describes the payload without describing itself, and every
    # hash in it must verify against the file beside it.
    manifest = json.loads((artifact / "MANIFEST.json").read_text(encoding="utf-8"))
    assert manifest["artifact"] == "scut/masked_test_v1"
    assert manifest["n_faces"] == 6
    for name, digest in manifest["payload_files"].items():
        assert hash_file(artifact / name) == digest, name
    assert "MANIFEST.json" not in manifest["payload_files"]

    # metrics.json carries the whole-directory rollup -- the value the p6
    # masked pretraining configs must declare, and what guard 3 verifies.
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["artifact"]["rollup_sha256_for_configs"] == (
        hash_dir(artifact)["rollup"]
    )
    assert metrics["artifact"]["payload_rollup"] == manifest["payload_rollup"]


def test_a_sized_build_carries_its_size_and_note_in_the_artifact(
    tmp_path, clean_repo, out_root
):
    """**[2026-08-09, review question settled by rebuild rather than
    reader-side]** A consumer opening masked_512_v1 could not tell the
    staging size, and the write-up guard was absent from the thing a reader
    opens. Unlike the v1 staging manifests -- whose rollups were already
    recorded in a passed verdict -- nothing declares these artifacts yet, so
    the fields go IN the MANIFEST and the artifacts rebuild."""
    import json

    from fixtures import builders

    root = a_scut_root(tmp_path)
    config = builders.write_config(
        tmp_path / "cfg.yaml", phase="roadb_p5",
        inputs=declare_root(root),
        task=masked_task(size=512, out_version="masked_size_test_v1"),
    )
    run_dir = run_masked_task(config, out_root, clean_repo)

    artifact = clean_repo / "data" / "scut" / "masked_size_test_v1"
    for geometry in ("g1", "g2"):
        array = np.load(artifact / f"masked_{geometry}.npy")
        assert array.shape == (6, 512, 512, 3), (
            "the arrays are self-describing; the MANIFEST field is the "
            "human-facing record"
        )
    manifest = json.loads((artifact / "MANIFEST.json").read_text(encoding="utf-8"))
    assert manifest["size"] == 512
    assert "350x350" in manifest["interpolation_note"]
    assert "no texture information beyond 350px" in manifest["interpolation_note"]
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["size"] == 512
    assert metrics["interpolation_note"] == manifest["interpolation_note"]


def test_a_224_build_carries_its_size_and_no_interpolation_note(
    tmp_path, clean_repo, out_root
):
    """224 downsamples the 350px sources, so the note would be false there --
    its absence is load-bearing, not an omission."""
    import json

    from fixtures import builders

    root = a_scut_root(tmp_path)
    config = builders.write_config(
        tmp_path / "cfg.yaml", phase="p5",
        inputs=declare_root(root),
        task=masked_task(out_version="masked_224_note_test_v1"),
    )
    run_masked_task(config, out_root, clean_repo)

    manifest = json.loads(
        (clean_repo / "data" / "scut" / "masked_224_note_test_v1" /
         "MANIFEST.json").read_text(encoding="utf-8")
    )
    assert manifest["size"] == 224
    assert "interpolation_note" not in manifest


def test_the_artifact_feeds_the_pretraining_loader(tmp_path, clean_repo, out_root):
    """The artifact this task writes is the artifact the pretrain loader reads:
    written by one task and consumed by the other, aligned by stem. This is the
    integration the placeholder path never had."""
    from cleft.train.pretrain import load_masked_features
    from fixtures import builders

    root = a_scut_root(tmp_path)
    config = builders.write_config(
        tmp_path / "cfg.yaml", phase="p5",
        inputs=declare_root(root), task=masked_task(),
    )
    run_masked_task(config, out_root, clean_repo)

    artifact = clean_repo / "data" / "scut" / "masked_test_v1"
    rows = load_masked_features(artifact, "g2", ["CF1", "AF2"])
    assert rows.shape == (2, OUTPUT_SIZE, OUTPUT_SIZE, 3)


def test_a_second_build_of_the_same_version_is_refused(
    tmp_path, clean_repo, out_root
):
    from fixtures import builders

    root = a_scut_root(tmp_path)
    config = builders.write_config(
        tmp_path / "cfg.yaml", phase="p5",
        inputs=declare_root(root), task=masked_task(),
    )
    run_masked_task(config, out_root, clean_repo)
    with pytest.raises(ValueError, match="immutable"):
        run_masked_task(
            builders.write_config(
                tmp_path / "cfg2.yaml", phase="p5",
                inputs=declare_root(root), task=masked_task(),
            ),
            tmp_path / "runs2",
            clean_repo,
        )


def test_a_full_build_that_persists_nothing_is_refused(
    tmp_path, clean_repo, out_root
):
    """THE guard. n_faces: 0 with write_artifact: false is the mode all three
    Phase 5 runs took: full compute, a PNG, and no artifact."""
    from fixtures import builders

    root = a_scut_root(tmp_path)
    config = builders.write_config(
        tmp_path / "cfg.yaml", phase="p5",
        inputs=declare_root(root),
        task=masked_task(write_artifact=False),
    )
    with pytest.raises(ValueError, match="nothing kept beyond the sheet"):
        run_masked_task(config, out_root, clean_repo)


def test_the_synthesis_task_refuses_the_same_silent_mode(
    tmp_path, clean_repo, out_root
):
    """Same class, other artifact-declaring kind -- and there the silent mode
    would also be the known pathologically slow cluster build."""
    from cleft.provenance import RunContext
    from cleft.run import TASKS
    from fixtures import builders

    root = a_scut_root(tmp_path)
    config = builders.write_config(
        tmp_path / "cfg.yaml", phase="p5",
        inputs=declare_root(root),
        task={
            "kind": "asymmetry_synthesis",
            "scut_root": "scut_root",
            "n_faces": 0,
            "write_artifact": False,
        },
    )
    with pytest.raises(ValueError, match="nothing kept beyond the sheet"):
        with RunContext(config, out_root, repo_root=clean_repo) as ctx:
            TASKS["asymmetry_synthesis"](ctx)


def test_a_sheet_only_review_pass_still_works(tmp_path, clean_repo, out_root):
    """Small n_faces without the artifact is the legitimate review mode and
    must keep working -- the guard refuses the FULL silent build, not review."""
    from fixtures import builders

    root = a_scut_root(tmp_path)
    config = builders.write_config(
        tmp_path / "cfg.yaml", phase="p5",
        inputs=declare_root(root),
        task=masked_task(write_artifact=False, n_faces=3),
    )
    run_dir = run_masked_task(config, out_root, clean_repo)
    assert (run_dir / "masked_scut_sheet.png").exists()
    assert not (clean_repo / "data" / "scut" / "masked_test_v1").exists()


def test_a_killed_build_resumes_and_matches_an_uninterrupted_one(
    tmp_path, clean_repo, out_root, monkeypatch
):
    """The synthesis contract, on this task: killed mid-build after a
    checkpoint, resumed under the same job id, and the published payload is
    byte-identical to a never-interrupted build's. MANIFEST.json differs by the
    generating run's identity, so the comparison is per payload file plus the
    payload rollup -- the quantities a consumer's declared hash covers."""
    import json

    from cleft.scut import masked as masked_module
    from fixtures import builders

    root = a_scut_root(tmp_path)

    # The uninterrupted reference, its own version.
    reference_config = builders.write_config(
        tmp_path / "ref.yaml", phase="p5",
        inputs=declare_root(root),
        task=masked_task(out_version="masked_ref_v1"),
    )
    run_masked_task(reference_config, tmp_path / "runs_ref", clean_repo)
    reference = clean_repo / "data" / "scut" / "masked_ref_v1"

    # The interrupted build: die while building face 5 of 6 (checkpoint_every
    # 2 -> checkpoints at faces 2 and 4, journal rolled back to 4 on resume).
    config = builders.write_config(
        tmp_path / "cfg.yaml", phase="p5",
        inputs=declare_root(root), task=masked_task(),
    )
    monkeypatch.setenv("CLEFT_JOB_ID", "masked-resume-test")

    real_build_one = masked_module.build_one
    calls = {"n": 0}

    def dying_build_one(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] > 8:  # 4 faces x 2 geometries survive
            raise RuntimeError("simulated pod kill")
        return real_build_one(*args, **kwargs)

    monkeypatch.setattr(masked_module, "build_one", dying_build_one)
    with pytest.raises(RuntimeError, match="simulated pod kill"):
        run_masked_task(config, out_root, clean_repo)

    working = clean_repo / "data" / "scut" / "masked_test_v1.inprogress"
    assert working.is_dir(), "the kill should leave the in-progress build"
    assert (working / "checkpoint.npz").exists()

    monkeypatch.setattr(masked_module, "build_one", real_build_one)
    run_masked_task(config, out_root, clean_repo)

    artifact = clean_repo / "data" / "scut" / "masked_test_v1"
    assert artifact.is_dir() and not working.exists()

    for name in ("masked_g1.npy", "masked_g2.npy", "faces.json"):
        assert (artifact / name).read_bytes() == (reference / name).read_bytes(), (
            f"{name} differs between the resumed and uninterrupted builds"
        )
    resumed_manifest = json.loads(
        (artifact / "MANIFEST.json").read_text(encoding="utf-8")
    )
    reference_manifest = json.loads(
        (reference / "MANIFEST.json").read_text(encoding="utf-8")
    )
    assert resumed_manifest["payload_rollup"] == reference_manifest["payload_rollup"]


def a_textured_scut_root(tmp_path):
    """a_scut_root with textured images. The shared fixture image is
    two FLAT regions, so a warp changes ~no pixel above
    pixel_change_extent's threshold and the summary takes a short-dict
    path real faces never hit; texture makes the fixture walk the same
    summary branch the cluster runs do."""
    from PIL import Image

    from cleft.scut.dataset import IMAGE_DIR

    root = a_scut_root(tmp_path)
    rng = np.random.default_rng(5)
    for jpg in sorted((root / IMAGE_DIR).glob("*.jpg")):
        textured = np.clip(
            a_source_image().astype(int)
            + rng.integers(-60, 60, a_source_image().shape), 0, 255
        ).astype(np.uint8)
        Image.fromarray(textured).save(jpg)
    return root



@pytest.mark.parametrize("warp_family", ["tps", "piecewise_affine"])
def test_both_warp_families_survive_the_full_task_including_the_summary(
    tmp_path, clean_repo, out_root, warp_family
):
    """[2026-08-30] The test the PWA crash named. Run p17-synth-pwa
    completed all 5,499 faces, wrote and renamed the artifact, then died
    at KeyError: 'identity' in the post-artifact summary -- the suite
    covered PWA's warp MATH, but nothing drove the piecewise family
    through the task's TAIL: the summary consumes the TPS record
    contract (identity, deformation.max_unintended_motion_frac_width,
    deformation.pixel_change.{y_min,outside_box}) and the piecewise
    m>0 records carried none of it.

    Family-parameterised, small n, write_artifact True so the run
    reaches the exact code that crashed. **Fails pre-fix on
    piecewise_affine (KeyError: 'identity') and passes on tps** --
    which is precisely the asymmetry the cluster showed.
    """
    from cleft.provenance import RunContext
    from cleft.run import TASKS
    from fixtures import builders

    root = a_textured_scut_root(tmp_path)
    config = builders.write_config(
        tmp_path / f"cfg_{warp_family}.yaml", phase="p17",
        inputs=declare_root(root),
        task={
            "kind": "asymmetry_synthesis",
            "scut_root": "scut_root",
            "geometries": ["g1"],
            "n_faces": 3,
            "n_sheet": 2,
            "magnitudes": [0.0, 0.025],
            "warp_family": warp_family,
            "checkpoint_every": 2,
            "write_artifact": True,
            "out_version": f"synth_smoke_{warp_family}_v1",
        },
    )
    with RunContext(config, out_root, repo_root=clean_repo) as ctx:
        TASKS["asymmetry_synthesis"](ctx)
    artifact = clean_repo / "data" / "scut" / f"synth_smoke_{warp_family}_v1"
    assert artifact.exists()
    # The summary path ran to completion: metrics.json exists and the
    # deformation aggregates it crashes without are present.
    import json

    metrics = json.loads(
        (ctx.run_dir / "metrics.json").read_text(encoding="utf-8")
    )
    assert metrics["deformation_is_local"]["n_deformations_measured"] > 0


def test_the_arm_readers_load_what_the_synthesis_writer_actually_wrote(
    tmp_path, clean_repo, out_root
):
    """[2026-08-30] The test the arm-A crash named: the family smoke
    drove the WRITER end-to-end and this file already asserted the
    artifact has no faces.jsonl -- but nothing ever CHAINED
    writer->reader, so both arm tasks shipped reading a journal name
    the finalizer deletes. Synthesis writes on the fixture root, then
    the arm-side load path (run._synth_index + the magnitude arrays it
    names) reads the published artifact.

    **Pre-fix this failed with FileNotFoundError: faces.jsonl --
    reproducing the cluster crash at run.py:4718 -- and it guards the
    FORMAT too**: faces.json is ONE JSON document (a list), not
    JSON-lines, so a reader that kept the line-by-line parse on the
    right name would also fail here.
    """
    import numpy as np

    from cleft.provenance import RunContext
    from cleft.run import TASKS, _synth_index
    from fixtures import builders

    root = a_textured_scut_root(tmp_path)
    magnitudes = [0.0, 0.025]
    config = builders.write_config(
        tmp_path / "cfg.yaml", phase="p17",
        inputs=declare_root(root),
        task={
            "kind": "asymmetry_synthesis",
            "scut_root": "scut_root",
            "geometries": ["g1"],
            "n_faces": 3,
            "n_sheet": 2,
            "magnitudes": magnitudes,
            "warp_family": "tps",
            "checkpoint_every": 2,
            "write_artifact": True,
            "out_version": "synth_chain_v1",
        },
    )
    with RunContext(config, out_root, repo_root=clean_repo) as ctx:
        TASKS["asymmetry_synthesis"](ctx)
    artifact = clean_repo / "data" / "scut" / "synth_chain_v1"

    # The cluster's exact crash, reproduced against the published
    # artifact: the old readers opened the resume journal's name.
    with pytest.raises(FileNotFoundError):
        (artifact / "faces.jsonl").read_text(encoding="utf-8")

    # The arm-side load path, against the PUBLISHED artifact.
    per_face, parsed = _synth_index(artifact)
    assert parsed == magnitudes
    assert len(per_face) == 3
    assert all("stem" in face and "magnitudes" in face for face in per_face)
    # And the arrays the readers open under the names the index implies.
    for magnitude in parsed:
        stack = np.load(
            artifact / f"synth_g1_m{magnitude:.3f}.npy", mmap_mode="r"
        )
        assert stack.shape[0] == 3
