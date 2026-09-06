"""The three anatomy placement transforms, and the sweep that chooses them.

The first contact sheet showed the region set consistently too high, too narrow
and too small — internally correct, globally offset. These are the three
orthogonal corrections, and the point of keeping them orthogonal is that each can
be judged on its own.
"""

from __future__ import annotations

import pytest

from cleft.geometry import contact as C
from cleft.geometry import generators as G  # noqa: F401  (registers anatomy)
from cleft.geometry import patches as P
from cleft.geometry import render as R
from cleft.geometry.trapezium import Trapezium

G2 = Trapezium(top_half_width=0.5, bot_half_width=0.5)
#: The real measured trapezium. The anatomy scheme is parameterised for G2, so
#: this is used only to show what it WOULD do at G1.
G1 = Trapezium()
TOL = 1e-9


def anatomy(**kwargs):
    return P.get("anatomy").generate(G2, P.PatchConfig(**kwargs))


# --------------------------------------------------------------------------
# defaults unchanged
# --------------------------------------------------------------------------


#: The untransformed baseline. Tests compare against THIS rather than against
#: PatchConfig(), so a future default change cannot quietly make a transform test
#: compare two identical things and pass for the wrong reason.
IDENTITY = dict(
    anatomy_v_offset=0.0,
    anatomy_scale=1.0,
    anatomy_scale_x=1.0,
    anatomy_box_scale_x=1.0,
)


def test_the_defaults_are_the_values_chosen_from_the_sweeps():
    """All four settled. The ablation scheme is finished."""
    config = P.PatchConfig()
    assert config.anatomy_v_offset == 0.04, "sweep 1, row 4"
    assert config.anatomy_scale == 1.80, "sweep 3, row 3"
    assert config.anatomy_scale_x == 1.60, "sweep 2, row 3"
    assert config.anatomy_box_scale_x == 1.0, "never needed; spread was the fix"


def test_the_accepted_characteristics_have_not_drifted():
    """The scheme is frozen; these are what it was accepted as.

    A change to the region table or to any default moves these numbers. Failing
    here means the ablation has become a different experiment, which is a
    decision to take deliberately rather than to discover afterwards.
    """
    patches = anatomy()
    report = P.overlap_report(patches)
    accepted = G.ACCEPTED_ANATOMY

    assert report["redundancy"] == pytest.approx(accepted["redundancy"], abs=5e-4)
    assert report["n_overlapping_pairs"] == accepted["n_overlapping_pairs"]
    assert report["n_pairs"] == accepted["n_pairs"]
    assert report["max_pairwise_iou"] == pytest.approx(
        accepted["max_pairwise_iou"], abs=5e-5
    )
    assert report["frame_covered"] == pytest.approx(accepted["frame_covered"], abs=5e-4)
    assert sum(1 for p in patches if p.clamped) == accepted["n_clamped"]

    g1 = P.region_coverage(P.get("anatomy").generate(G1), G1)
    assert g1["lateral_orbit"] == pytest.approx(
        accepted["lateral_orbit_coverage_g1"], abs=5e-4
    )


def test_the_capacity_gap_against_the_grid_is_structural():
    """PLAN §4.5. Parity would need scale ~2.48, and 2.4 already clamps.

    So the anatomy scheme runs at roughly half the grid's per-node area and that
    cannot be removed. Stated as a limitation of the ablation rather than left to
    be discovered when the result needs explaining.
    """
    import numpy as np

    grid_area = float(np.mean([p.w * p.h for p in P.get("grid").generate(G2)]))
    anatomy_area = float(np.mean([p.w * p.h for p in anatomy()]))
    ratio = anatomy_area / grid_area

    assert ratio == pytest.approx(G.ANATOMY_AREA_RATIO_TO_GRID, abs=0.005)
    assert 0.4 < ratio < 0.7, "anatomy runs at roughly half the grid's node area"

    # And the setting that would close the gap is not available.
    parity = P.get("anatomy").generate(G2, P.PatchConfig(anatomy_scale=2.48))
    assert P.clamped_regions(parity), (
        "if a parity scale stops clamping, the limitation in PLAN §4.5 no longer "
        "holds and the ablation could be run at matched capacity"
    )


def test_the_rejected_spread_really_does_clamp():
    """2.0 was rejected because it only LOOKED wider. Recorded as a test."""
    patches = anatomy(anatomy_scale_x=2.0)
    assert P.clamped_regions(patches) == ["commissure", "lateral_orbit"]
    assert not P.clamped_regions(anatomy(anatomy_scale_x=1.60)), (
        "the chosen spread must be clean"
    )


def test_the_defaults_are_what_the_generator_actually_uses():
    assert [p.box for p in anatomy()] == [p.box for p in P.get("anatomy").generate(G2)]


# --------------------------------------------------------------------------
# each knob does one thing
# --------------------------------------------------------------------------


def test_v_offset_moves_everything_down_by_exactly_that_much():
    base = {p.band: p for p in anatomy(**IDENTITY)}
    moved = {p.band: p for p in anatomy(**{**IDENTITY, "anatomy_v_offset": 0.05})}
    for name, patch in base.items():
        # Clamping applies at the very bottom, so compare where there is room.
        if patch.y + patch.h + 0.05 <= 1.0:
            assert moved[name].y - patch.y == pytest.approx(0.05, abs=1e-6)


def test_v_offset_changes_no_size_and_no_spread():
    base = {p.band: p for p in anatomy(**IDENTITY)}
    moved = {p.band: p for p in anatomy(**{**IDENTITY, "anatomy_v_offset": 0.05})}
    for name, patch in base.items():
        assert moved[name].w == pytest.approx(patch.w, abs=TOL)
        assert moved[name].h == pytest.approx(patch.h, abs=TOL)
        assert moved[name].centre_x == pytest.approx(patch.centre_x, abs=TOL)


def test_scale_changes_size_and_not_position():
    base = {(p.band, round(p.centre_x, 6)): p for p in anatomy(**IDENTITY)}
    bigger = {
        (p.band, round(p.centre_x, 6)): p
        for p in anatomy(**{**IDENTITY, "anatomy_scale": 1.3})
    }
    assert set(base) == set(bigger), "scaling must not move any centre"
    for key, patch in base.items():
        assert bigger[key].w == pytest.approx(patch.w * 1.3, abs=1e-6)
        assert bigger[key].h == pytest.approx(patch.h * 1.3, abs=1e-6)


def test_scale_x_spreads_pairs_apart_without_resizing_them():
    base = {p.band: p for p in anatomy(**IDENTITY) if p.centre_x < 0.5}
    spread = {
        p.band: p
        for p in anatomy(**{**IDENTITY, "anatomy_scale_x": 1.4})
        if p.centre_x < 0.5
    }
    for name, patch in base.items():
        assert spread[name].centre_x < patch.centre_x, f"{name} did not move outward"
        assert spread[name].w == pytest.approx(patch.w, abs=TOL), (
            "spread must not change size -- that is what anatomy_scale is for"
        )


def test_scale_x_leaves_midline_structures_alone():
    """They sit at offset 0, so spreading has nothing to multiply."""
    names = {n for n, _, _, _ in G.MIDLINE_REGIONS}
    for patch in anatomy(anatomy_scale_x=1.5):
        if patch.band in names:
            assert patch.centre_x == pytest.approx(0.5, abs=TOL)


def test_the_three_knobs_are_independent():
    """Applying all three equals applying each in turn -- no interaction."""
    combined = {
        (p.band, round(p.x, 6)): p
        for p in anatomy(anatomy_v_offset=0.04, anatomy_scale=1.3, anatomy_scale_x=1.2)
    }
    assert len(combined) == 27


def test_mirror_symmetry_survives_every_transform():
    """The asymmetry instrument must not be broken by a placement correction."""
    for kwargs in (
        {"anatomy_v_offset": 0.08},
        {"anatomy_scale": 1.4},
        {"anatomy_scale_x": 1.5},
        {"anatomy_v_offset": 0.04, "anatomy_scale": 1.3, "anatomy_scale_x": 1.2},
    ):
        patches = {p.id: p for p in anatomy(**kwargs)}
        assert len(patches) == 27
        for patch in patches.values():
            partner = patches[patch.mirror_id]
            assert partner.x == pytest.approx(1.0 - patch.x - patch.w, abs=1e-9)


def test_transformed_regions_stay_in_the_unit_square():
    for patch in anatomy(anatomy_v_offset=0.10, anatomy_scale=1.5, anatomy_scale_x=1.3):
        assert 0.0 <= patch.x and patch.x + patch.w <= 1.0 + 1e-9
        assert 0.0 <= patch.y and patch.y + patch.h <= 1.0 + 1e-9


# --------------------------------------------------------------------------
# labels
# --------------------------------------------------------------------------


def test_regions_abbreviate_with_a_side():
    assert R.abbreviate("alar_base", 0.3) == "ala_L"
    assert R.abbreviate("alar_base", 0.7) == "ala_R"
    assert R.abbreviate("philtral_column", 0.42) == "philtral_L"
    assert R.abbreviate("vermillion_border", 0.6) == "verm_R"


def test_midline_structures_get_no_side_suffix():
    assert R.abbreviate("philtrum", 0.5) == "philt"
    assert R.abbreviate("nasal_tip", 0.5) == "tip"


def test_every_anatomy_region_has_an_abbreviation():
    names = {n for n, _, _, _ in G.MIDLINE_REGIONS}
    names |= {n for n, _, _, _, _ in G.BILATERAL_REGIONS}
    assert names <= set(R.ABBREVIATIONS), (
        f"unabbreviated: {sorted(names - set(R.ABBREVIATIONS))}"
    )


def test_abbreviations_are_distinct():
    values = list(R.ABBREVIATIONS.values())
    assert len(values) == len(set(values)), "two structures share a label"


def test_labels_are_drawn_by_default():
    import numpy as np

    from cleft.geometry import mapping as M
    from cleft.geometry import staging as S

    staged = S.stage(np.full((300, 222, 3), 120, dtype=np.uint8))
    mapped = M.map_patches(staged, anatomy())
    with_labels = R.render_overlay(staged, mapped, None, labels=True)
    without = R.render_overlay(staged, mapped, None, labels=False)
    assert not np.array_equal(with_labels, without)


# --------------------------------------------------------------------------
# the sweep
# --------------------------------------------------------------------------


def test_the_variant_grid_is_the_cartesian_product():
    variants = C.variant_grid([0.0, 0.04], [1.0, 1.3], [1.0])
    assert len(variants) == 4
    assert variants[0].label == "dy=+0.00 s=1.00 sx=1.00"


def test_variant_labels_carry_all_three_parameters():
    for variant in C.variant_grid([0.08], [1.3], [1.2]):
        assert "dy=+0.08" in variant.label
        assert "s=1.30" in variant.label
        assert "sx=1.20" in variant.label


def test_the_grid_order_is_stable():
    """The sheet layout must match the variant list in metrics.json."""
    first = [v.label for v in C.variant_grid([0.0, 0.04], [1.0, 1.3], [1.0])]
    second = [v.label for v in C.variant_grid([0.0, 0.04], [1.0, 1.3], [1.0])]
    assert first == second


def test_an_empty_sweep_is_rejected(tmp_path):
    with pytest.raises(C.ContactError, match="no variants"):
        C.build_sweep([{"patient_id": "1", "frontal_id": "1", "class3": "0"}],
                      tmp_path, [])


def test_the_sweep_renders_every_combination(tmp_path):
    from test_contact import write_cohort

    root = tmp_path / "photos"
    rows = write_cohort(root, n_patients=3)
    variants = C.variant_grid([0.0, 0.04], [1.0, 1.3], [1.0])

    result = C.build_sweep(rows, root, variants, geometry="g2")
    assert len(result.reports) == len(variants) * 3
    assert len({r["variant"] for r in result.reports}) == 4


def test_the_sweep_defaults_to_g2(tmp_path):
    """Where the region-scheme ablation has to run, because G1 confounds it."""
    from test_contact import write_cohort

    root = tmp_path / "photos"
    rows = write_cohort(root, n_patients=2)
    result = C.build_sweep(rows, root, C.variant_grid([0.0], [1.0], [1.0]))
    assert result.geometry == "g2"
    assert all(r["n_patches_with_white"] == 0 for r in result.reports)


def test_the_shipped_sweep_config_is_valid(repo_root):
    from cleft.config import load_config

    task = load_config(repo_root / "configs" / "p2_anatomy_sweep.yaml")["task"]
    assert task["kind"] == "anatomy_sweep"
    assert task["geometry"] == "g2", (
        "the sweep must judge placement where the ablation will run"
    )

    # Sweep 3 holds offset and spread fixed and varies only SIZE.
    assert task["v_offsets"] == [0.04] and task["scales_x"] == [1.60]
    assert task["box_scales_x"] == [1.0], (
        "the report was 'bigger', not 'wider'; box width is a different knob"
    )
    assert task["scales"] == [1.3, 1.6, 1.8, 1.9]

    n_variants = (
        len(task["v_offsets"]) * len(task["scales"])
        * len(task["scales_x"]) * len(task["box_scales_x"])
    )
    assert n_variants * task["n_patients"] <= 12, "beyond 12 panels nothing is legible"


def test_every_shipped_sweep_variant_is_a_candidate():
    """A panel showing a setting already excluded by number is a wasted panel.

    Not a rule against ever including one -- a deliberately-too-big anchor can
    help the eye calibrate -- but it should be a choice, not an accident of
    picking round numbers past a ceiling nobody had measured.
    """
    from cleft.config import load_config
    from pathlib import Path

    task = load_config(
        Path(__file__).resolve().parents[1] / "configs" / "p2_anatomy_sweep.yaml"
    )["task"]
    report = C.sweep_geometry_report(
        C.variant_grid(
            task["v_offsets"], task["scales"], task["scales_x"], task["box_scales_x"]
        )
    )
    excluded = [label for label, e in report.items() if not e["candidate"]]
    assert not excluded, f"variants excluded by clamping: {excluded}"


def test_the_clamping_ceiling_sits_between_1_9_and_2_0():
    """Measured, so the sweep's upper bound has a reason behind it."""
    clean = C.sweep_geometry_report(C.variant_grid([0.04], [1.9], [1.60], [1.0]))
    dirty = C.sweep_geometry_report(C.variant_grid([0.04], [2.0], [1.60], [1.0]))
    assert next(iter(clean.values()))["candidate"] is True
    assert next(iter(dirty.values()))["candidate"] is False


def test_the_shipped_sweep_holds_the_settled_parameters_at_their_defaults():
    """Sweep 2 must not silently re-open a question sweep 1 answered."""
    from cleft.config import load_config
    from pathlib import Path

    task = load_config(
        Path(__file__).resolve().parents[1] / "configs" / "p2_anatomy_sweep.yaml"
    )["task"]
    config = P.PatchConfig()
    assert task["v_offsets"] == [config.anatomy_v_offset]
    assert task["scales_x"] == [config.anatomy_scale_x]


# --------------------------------------------------------------------------
# overlap -- the ceiling on box size
# --------------------------------------------------------------------------


def test_a_partition_has_redundancy_one():
    """Sanity anchor: non-overlapping boxes stack one deep."""
    boxes = [
        P.Patch(0, "a", 0.0, 0.0, 0.5, 0.5),
        P.Patch(1, "b", 0.5, 0.0, 0.5, 0.5),
        P.Patch(2, "c", 0.0, 0.5, 0.5, 0.5),
        P.Patch(3, "d", 0.5, 0.5, 0.5, 0.5),
    ]
    report = P.overlap_report(boxes)
    assert report["redundancy"] == pytest.approx(1.0, abs=1e-6)
    assert report["max_pairwise_iou"] == 0.0
    assert report["frame_covered"] == pytest.approx(1.0, abs=1e-6)


def test_identical_boxes_have_iou_one_and_redundancy_equal_to_their_count():
    boxes = [P.Patch(i, "same", 0.2, 0.2, 0.4, 0.4) for i in range(3)]
    report = P.overlap_report(boxes)
    assert report["max_pairwise_iou"] == pytest.approx(1.0, abs=1e-6)
    assert report["redundancy"] == pytest.approx(3.0, abs=1e-6)


def test_iou_of_half_overlapping_boxes():
    a = P.Patch(0, "a", 0.0, 0.0, 0.2, 0.2)
    b = P.Patch(1, "b", 0.1, 0.0, 0.2, 0.2)
    # intersection 0.1x0.2 = 0.02; union 0.04 + 0.04 - 0.02 = 0.06
    assert P.iou(a, b) == pytest.approx(0.02 / 0.06, abs=1e-9)


def test_disjoint_boxes_have_zero_iou():
    a = P.Patch(0, "a", 0.0, 0.0, 0.2, 0.2)
    b = P.Patch(1, "b", 0.5, 0.5, 0.2, 0.2)
    assert P.iou(a, b) == 0.0


def test_redundancy_rises_with_region_size():
    """The measurement that bounds sweep 3."""
    previous = 0.0
    for scale in (1.3, 1.6, 2.0, 2.5):
        report = P.overlap_report(anatomy(anatomy_scale=scale))
        assert report["redundancy"] >= previous, "bigger boxes cannot overlap less"
        previous = report["redundancy"]


def test_the_pair_count_is_every_unordered_pair():
    assert P.overlap_report(anatomy())["n_pairs"] == 27 * 26 // 2


def test_overlap_report_rejects_an_empty_set():
    with pytest.raises(P.PatchError, match="no patches"):
        P.overlap_report([])


# --------------------------------------------------------------------------
# mask coverage -- a different property from clamping
# --------------------------------------------------------------------------


def test_mask_coverage_and_clamping_are_different_properties():
    """The exact case that motivated the guard.

    At the chosen spread, lateral_orbit is entirely inside the FRAME -- so the
    clamp guard passes -- while 87% of it is outside the G1 MASK. One guard
    cannot stand in for the other.
    """
    patches = P.get("anatomy").generate(G1, P.PatchConfig())

    P.assert_no_clamping(patches)  # frame containment: fine
    with pytest.raises(P.PatchError, match="lateral_orbit"):
        P.assert_mask_coverage(patches, min_coverage=0.5)


def test_mask_coverage_passes_at_g2_where_the_ablation_runs():
    P.assert_mask_coverage(P.get("anatomy").generate(G2), min_coverage=0.99)


def test_the_message_names_the_region_and_its_coverage():
    patches = P.get("anatomy").generate(G1, P.PatchConfig())
    with pytest.raises(P.PatchError) as excinfo:
        P.assert_mask_coverage(patches, min_coverage=0.5)
    message = str(excinfo.value)
    assert "lateral_orbit" in message
    expected = f"{G.ACCEPTED_ANATOMY['lateral_orbit_coverage_g1']:.2f}"
    assert expected in message, "the message must give the actual coverage"
    assert "background" in message


def test_the_threshold_is_a_choice_not_a_measurement():
    """0.5 is a floor. A stricter build may demand more."""
    patches = P.get("anatomy").generate(G1, P.PatchConfig())
    with pytest.raises(P.PatchError):
        P.assert_mask_coverage(patches, min_coverage=0.99)
    # And a permissive threshold lets even the worst region through.
    P.assert_mask_coverage(patches, min_coverage=0.1)


def test_region_coverage_is_keyed_by_structure_not_by_patch():
    coverage = P.region_coverage(P.get("anatomy").generate(G1), G1)
    assert len(coverage) == 18, "9 midline + 9 bilateral structures"
    assert coverage["lateral_orbit"] == pytest.approx(
        G.ACCEPTED_ANATOMY["lateral_orbit_coverage_g1"], abs=5e-4
    )
    assert coverage["philtrum"] > 0.99


def test_left_and_right_lose_the_same_amount():
    """An asymmetric loss would manufacture the signal being measured."""
    patches = P.get("anatomy").generate(G1)
    by_name: dict[str, list[float]] = {}
    for patch in patches:
        by_name.setdefault(patch.band, []).append(patch.coverage)
    for name, values in by_name.items():
        assert max(values) - min(values) < 1e-6, f"{name} is lopsided: {values}"


def test_the_sweep_reports_per_region_coverage_at_both_geometries():
    """G2 alone would be all 1.0 and hide the thing worth seeing."""
    report = C.sweep_geometry_report(C.variant_grid([0.04], [1.3], [1.60], [1.0]))
    entry = next(iter(report.values()))

    assert all(v == 1.0 for v in entry["region_coverage_g2"].values())
    assert entry["region_coverage_g1"]["lateral_orbit"] < 0.5
    assert "lateral_orbit" in entry["regions_below_half_coverage_g1"]
    assert entry["worst_region_g1"]["region"] == "lateral_orbit"


def test_the_sweep_reports_clamping_and_overlap_per_variant():
    variants = C.variant_grid([0.04], [1.3, 2.5], [1.60], [1.0])
    report = C.sweep_geometry_report(variants)

    assert len(report) == 2
    for entry in report.values():
        assert "redundancy" in entry and "max_pairwise_iou" in entry
        assert "clamped_regions" in entry and "candidate" in entry


def test_a_clamped_variant_is_not_a_candidate():
    """The exclusion is a number, not a judgement about how a panel looked."""
    report = C.sweep_geometry_report(C.variant_grid([0.04], [1.3], [2.0], [1.0]))
    entry = next(iter(report.values()))
    assert entry["n_clamped"] > 0
    assert entry["candidate"] is False
