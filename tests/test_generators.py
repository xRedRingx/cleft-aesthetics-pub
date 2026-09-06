"""The random control and the anatomy ablation.

Exit criterion 3: all three generators satisfy the same interface and produce
comparable node counts. The comparison is supposed to be about PLACEMENT, so
anything else that differs between them is a confound.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

from cleft.geometry import generators as G  # noqa: F401  (registers both)
from cleft.geometry import patches as P
from cleft.geometry.trapezium import DEFAULT as DEFAULT_TRAPEZIUM
from cleft.geometry.trapezium import Trapezium

G1 = Trapezium()
G2 = Trapezium(top_half_width=0.5, bot_half_width=0.5)

ALL = ("grid", "random", "anatomy")


# --------------------------------------------------------------------------
# one interface, comparable counts
# --------------------------------------------------------------------------


@pytest.mark.parametrize("name", ALL)
def test_every_generator_is_registered(name):
    assert P.get(name).name == name


@pytest.mark.parametrize("name", ALL)
def test_every_generator_returns_27_patches(name):
    """Exit criterion 3. Matched node count, or placement is confounded with capacity."""
    assert len(P.get(name).generate(G2)) == 27


@pytest.mark.parametrize("name", ALL)
def test_every_generator_takes_the_same_arguments(name):
    generator = P.get(name)
    assert generator.generate(G2) is not None
    assert generator.generate(G2, P.PatchConfig()) is not None


@pytest.mark.parametrize("name", ALL)
def test_every_generator_produces_mirror_pairs(name):
    """Asymmetry is a comparison between paired locations, in all three schemes."""
    patches = {p.id: p for p in P.get(name).generate(G2)}
    for patch in patches.values():
        assert patch.mirror_id is not None
        assert patches[patch.mirror_id].mirror_id == patch.id


@pytest.mark.parametrize("name", ALL)
def test_every_generator_has_self_mirrored_midline_patches(name):
    patches = P.get(name).generate(G2)
    centred = [p for p in patches if p.mirror_id == p.id]
    assert centred, "no patch straddles the midline"
    for patch in centred:
        assert patch.centre_x == pytest.approx(0.5, abs=1e-9)


@pytest.mark.parametrize("name", ALL)
def test_every_generator_stays_inside_the_mask_under_g1(name):
    for patch in P.get(name).generate(G1, P.PatchConfig(boundary_rule="clip")):
        assert patch.coverage > 0.9


# --------------------------------------------------------------------------
# random -- the control
# --------------------------------------------------------------------------


def test_random_matches_the_grid_size_distribution():
    """Matched in count AND size, so only placement differs."""
    grid = sorted((p.w, p.h) for p in P.get("grid").generate(G2))
    random = sorted((p.w, p.h) for p in P.get("random").generate(G2))
    assert grid == random


def test_random_places_patches_differently_from_the_grid():
    """The control must actually be a control."""
    grid = sorted(p.box for p in P.get("grid").generate(G2))
    random = sorted(p.box for p in P.get("random").generate(G2))
    assert grid != random


def test_random_is_reproducible_from_the_seed():
    config = P.PatchConfig(patch_seed=1337)
    first = [p.box for p in P.get("random").generate(G2, config)]
    second = [p.box for p in P.get("random").generate(G2, config)]
    assert first == second


def test_a_different_seed_gives_different_placement():
    a = [p.box for p in P.get("random").generate(G2, P.PatchConfig(patch_seed=1))]
    b = [p.box for p in P.get("random").generate(G2, P.PatchConfig(patch_seed=2))]
    assert a != b


def test_the_seed_lives_in_the_config_not_the_generator():
    """So a run is reproducible from its config file alone."""
    assert "patch_seed" in P.PatchConfig().__dataclass_fields__


def test_random_patches_lie_inside_the_mask_under_g1():
    for patch in P.get("random").generate(G1):
        assert patch.coverage == pytest.approx(1.0, abs=0.02), (
            "random placement must respect the mask it samples inside"
        )


def test_random_pairs_are_genuinely_reflected():
    patches = {p.id: p for p in P.get("random").generate(G2)}
    for patch in patches.values():
        partner = patches[patch.mirror_id]
        assert partner.x == pytest.approx(1.0 - patch.x - patch.w, abs=1e-9)
        assert partner.y == pytest.approx(patch.y, abs=1e-9)


def test_random_is_mirrored_by_default_and_that_is_a_decision():
    """Unconstrained random would confound placement with having pairs at all."""
    assert G.RandomGenerator().mirrored is True


def test_random_can_be_unconstrained_for_a_separate_question():
    """Whether the pairing itself carries signal is a different investigation."""
    loose = G.RandomGenerator(mirrored=False)
    assert len(loose.generate(G2)) == 27


def test_random_refuses_an_impossible_placement():
    """A patch wider than half the mask cannot sit on one side of the midline."""
    config = P.PatchConfig(bands=(P.BandSpec("wide", 0.0, 1.0, 1),))
    with pytest.raises(P.PatchError, match="could not place"):
        P.get("random").generate(Trapezium(top_half_width=0.2, bot_half_width=0.2), config)


# --------------------------------------------------------------------------
# anatomy -- the ablation
# --------------------------------------------------------------------------


def test_anatomy_is_nine_midline_plus_nine_pairs():
    assert len(G.MIDLINE_REGIONS) == 9
    assert len(G.BILATERAL_REGIONS) == 9
    assert 9 + 2 * 9 == 27


def test_anatomy_regions_are_named_for_structures():
    patches = P.get("anatomy").generate(G2)
    names = {p.band for p in patches}
    for expected in ("alar_base", "philtral_column", "vermillion_border", "commissure"):
        assert expected in names


def test_each_bilateral_structure_appears_exactly_twice():
    patches = P.get("anatomy").generate(G2)
    counts: dict[str, int] = {}
    for patch in patches:
        counts[patch.band] = counts.get(patch.band, 0) + 1
    for name, _, _, _, _ in G.BILATERAL_REGIONS:
        assert counts[name] == 2, f"{name} should be a left/right pair"
    for name, _, _, _ in G.MIDLINE_REGIONS:
        assert counts[name] == 1, f"{name} is a midline structure"


def test_a_bilateral_pair_is_the_mirror_of_itself():
    patches = {p.id: p for p in P.get("anatomy").generate(G2)}
    for patch in patches.values():
        partner = patches[patch.mirror_id]
        assert partner.band == patch.band, (
            "a structure's mirror must be the same structure on the other side"
        )


def test_midline_structures_straddle_the_midline():
    patches = P.get("anatomy").generate(G2)
    midline_names = {name for name, _, _, _ in G.MIDLINE_REGIONS}
    for patch in patches:
        if patch.band in midline_names:
            assert patch.centre_x == pytest.approx(0.5, abs=1e-9)
            assert patch.mirror_id == patch.id


def test_anatomy_runs_top_to_bottom_in_the_expected_order():
    """Eyes above nose above lips -- a sanity check on the y placements."""
    by_name = {p.band: p for p in P.get("anatomy").generate(G2)}
    assert by_name["glabella"].y < by_name["nasal_tip"].y
    assert by_name["nasal_tip"].y < by_name["philtrum"].y
    assert by_name["philtrum"].y < by_name["labial_tubercle"].y
    assert by_name["lateral_orbit"].y < by_name["alar_base"].y
    assert by_name["alar_base"].y < by_name["commissure"].y


def test_anatomy_regions_fit_inside_the_unit_square():
    for patch in P.get("anatomy").generate(G2):
        assert 0.0 <= patch.x and patch.x + patch.w <= 1.0 + 1e-9
        assert 0.0 <= patch.y and patch.y + patch.h <= 1.0 + 1e-9


def test_anatomy_regions_fit_the_g2_frame_where_the_ablation_runs():
    """At G2 the mask is the whole square, so every region is fully inside."""
    for patch in P.get("anatomy").generate(G2):
        assert patch.coverage == pytest.approx(1.0, abs=1e-6)


def test_the_lateral_regions_fall_off_the_g1_mask_at_the_chosen_spread():
    """MEASURED consequence of anatomy_scale_x = 1.60, recorded not hidden.

    The spread chosen from sweep 2 puts ``lateral_orbit`` at 0.384 from the
    midline, where the G1 trapezium half-width is only 0.315. The region is
    inside the FRAME -- so no clamp fires -- but 87% of it is outside the MASK.

    This is not a defect in the spread: 1.60 was chosen on G2 panels, and G2 is
    where the region ablation runs (PLAN §4.4). It is a third instance of the
    geometry x patch-scheme interaction, and a hard reason not to run the anatomy
    scheme at G1: those nodes would be mostly background.
    """
    by_name = {}
    for patch in P.get("anatomy").generate(G1):
        by_name.setdefault(patch.band, []).append(patch.coverage)

    # Compared against the RECORDED accepted value rather than a literal, so a
    # future parameter change moves one constant instead of silently drifting
    # past a threshold nobody revisits.
    assert min(by_name["lateral_orbit"]) == pytest.approx(
        G.ACCEPTED_ANATOMY["lateral_orbit_coverage_g1"], abs=5e-4
    )
    assert min(by_name["lateral_orbit"]) < 0.5, (
        "if this region now covers half the mask at G1, the spread or the "
        "trapezium has changed and the G1-only warning may no longer apply"
    )
    # Left and right must lose the SAME amount, or the loss itself is a bias.
    assert by_name["lateral_orbit"][0] == pytest.approx(
        by_name["lateral_orbit"][1], abs=1e-6
    )
    # The midline structures are unaffected -- the problem is lateral only.
    assert min(by_name["philtrum"]) > 0.99


def test_anatomy_scale_is_configurable():
    """The knob that exists in case area difference confounds placement."""
    small = P.get("anatomy").generate(G2, P.PatchConfig(anatomy_scale=1.0))
    large = P.get("anatomy").generate(G2, P.PatchConfig(anatomy_scale=1.4))
    assert sum(p.w * p.h for p in large) > sum(p.w * p.h for p in small)
    assert len(large) == len(small) == 27


# --------------------------------------------------------------------------
# the comparison
# --------------------------------------------------------------------------


def test_compare_reports_all_three():
    table = G.compare(G2)
    assert set(table) == set(ALL)
    assert all(entry["n_patches"] == 27 for entry in table.values())


def test_the_area_difference_is_reported_rather_than_hidden():
    """Anatomy regions ARE smaller than grid patches. That is part of the
    hypothesis, and also a potential confound, so it must be visible."""
    table = G.compare(G2)
    assert table["anatomy"]["mean_area"] < table["grid"]["mean_area"]
    assert table["random"]["mean_area"] == pytest.approx(
        table["grid"]["mean_area"], abs=1e-9
    ), "random is size-matched to the grid, so only anatomy should differ"


# --------------------------------------------------------------------------
# the grid scheme, pinned -- the asymmetry against ACCEPTED_ANATOMY, closed
# --------------------------------------------------------------------------
#
# ACCEPTED_ANATOMY pins the anatomy scheme because its parameters were tuned
# across three visual sweeps and could drift. The grid was pinned by NOTHING,
# even though Phase 7's stage E compares the two directly -- so a silent change
# to one side of that comparison would have gone unnoticed.
#
# The grid is arithmetic, not tuning: 3/5/7 columns at 50% overlap gives 2n-1
# positions per band and exactly 27 boxes, mirror-symmetric by construction.
# That makes it cheap to pin and there was never a reason not to.

#: [MEASURED] The grid scheme at its defaults. Arithmetic, so these are
#: consequences of (3, 5, 7) columns at 50% overlap rather than chosen values.
ACCEPTED_GRID = {
    "n_patches": 27,
    "per_band": {"top": 5, "middle": 9, "bottom": 13},
    "band_widths": {"top": 1 / 3, "middle": 1 / 5, "bottom": 1 / 7},
    "self_mirrored_ids": (2, 9, 20),
}


def test_the_grid_produces_exactly_27_boxes():
    """3x3 crossed with shape variants gives 27 -- the supervision material's construction, and the
    arithmetic that produces it (PLAN §4.5.1)."""
    patches = P.GridGenerator().generate(DEFAULT_TRAPEZIUM, P.PatchConfig())
    assert len(patches) == ACCEPTED_GRID["n_patches"]


def test_the_grid_band_counts_are_the_overlap_arithmetic():
    """50% overlap makes n columns give 2n-1 positions, which is odd, which is
    what guarantees a centre patch straddling the midline in every band."""
    patches = P.GridGenerator().generate(DEFAULT_TRAPEZIUM, P.PatchConfig())
    counts = Counter(patch.band for patch in patches)
    assert dict(counts) == ACCEPTED_GRID["per_band"]
    for band, expected in ACCEPTED_GRID["per_band"].items():
        assert expected % 2 == 1, f"{band} has no centre patch"


def test_the_grid_widths_follow_the_column_counts():
    patches = P.GridGenerator().generate(DEFAULT_TRAPEZIUM, P.PatchConfig())
    for band, width in ACCEPTED_GRID["band_widths"].items():
        widths = {round(p.w, 6) for p in patches if p.band == band}
        assert widths == {round(width, 6)}


def test_the_grid_is_mirror_symmetric_by_construction():
    """The asymmetry instrument. Every patch has a mirror, and reflecting a
    patch's box lands on its partner's -- checked, not trusted, because if it
    broke silently every downstream comparison would still run."""
    patches = P.GridGenerator().generate(DEFAULT_TRAPEZIUM, P.PatchConfig())
    by_id = {patch.id: patch for patch in patches}

    assert all(patch.mirror_id is not None for patch in patches)
    for patch in patches:
        partner = by_id[patch.mirror_id]
        assert partner.mirror_id == patch.id, "mirroring must be reciprocal"
        assert partner.box == pytest.approx(patch.mirrored_box(), abs=1e-9)
        assert partner.band == patch.band


def test_the_grid_centre_patches_are_self_mirrored():
    """One per band, straddling the midline. Their ids are pinned so a change to
    the band construction cannot quietly move them."""
    patches = P.GridGenerator().generate(DEFAULT_TRAPEZIUM, P.PatchConfig())
    self_mirrored = tuple(sorted(p.id for p in patches if p.mirror_id == p.id))
    assert self_mirrored == ACCEPTED_GRID["self_mirrored_ids"]
    assert len(self_mirrored) == len(ACCEPTED_GRID["per_band"])
