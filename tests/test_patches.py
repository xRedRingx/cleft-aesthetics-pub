"""The patch interface and the grid generator.

Exit criterion 2: exactly 27 patches at defaults, mirror symmetry asserted, every
patch inside the mask under G2.
"""

from __future__ import annotations

import numpy as np
import pytest

from cleft.geometry import patches as P
from cleft.geometry.trapezium import Trapezium

TOL = 1e-9

#: G2's mask is the full square, so nothing crosses an edge.
G2 = Trapezium(top_half_width=0.5, bot_half_width=0.5)
G1 = Trapezium()


@pytest.fixture
def grid():
    return P.get("grid")


# --------------------------------------------------------------------------
# the count
# --------------------------------------------------------------------------


def test_defaults_give_exactly_27_patches(grid):
    """Exit criterion 2. 5 + 9 + 13 from 3 + 5 + 7 columns."""
    assert len(grid.generate(G2)) == 27


def test_the_band_arithmetic_is_two_n_minus_one():
    for columns, expected in ((3, 5), (5, 9), (7, 13)):
        assert P.BandSpec("b", 0.0, 1.0, columns).n_patches == expected
    assert P.PatchConfig().expected_patches == 27


def test_patches_per_band(grid):
    counts = P.summarise(grid.generate(G2))["per_band"]
    assert counts == {"bottom": 13, "middle": 9, "top": 5}


def test_the_bands_are_named_for_their_anatomy():
    assert [b.name for b in P.DEFAULT_BANDS] == ["top", "middle", "bottom"]
    assert [b.columns for b in P.DEFAULT_BANDS] == [3, 5, 7]


def test_finer_bands_lower_down():
    """Coarse top, normal middle, finest bottom."""
    columns = [b.columns for b in P.DEFAULT_BANDS]
    assert columns == sorted(columns), "bands must get finer toward the lip"


# --------------------------------------------------------------------------
# 50% overlap
# --------------------------------------------------------------------------


def test_stride_is_half_a_patch_width(grid):
    patches = sorted(
        [p for p in grid.generate(G2) if p.band == "middle"], key=lambda p: p.x
    )
    width = patches[0].w
    assert width == pytest.approx(1 / 5, abs=TOL)
    for earlier, later in zip(patches, patches[1:]):
        assert later.x - earlier.x == pytest.approx(width / 2, abs=1e-6)


def test_a_feature_narrower_than_half_a_patch_is_always_fully_contained(grid):
    """The reason for overlapping at all, checked rather than asserted in prose."""
    patches = [p for p in grid.generate(G2) if p.band == "middle"]
    width = patches[0].w
    feature = width / 2 * 0.99

    for centre in np.linspace(feature / 2, 1 - feature / 2, 200):
        left, right = centre - feature / 2, centre + feature / 2
        assert any(
            p.x - 1e-9 <= left and right <= p.x + p.w + 1e-9 for p in patches
        ), f"a feature at {centre:.3f} lives in no single patch"


def test_the_philtral_columns_are_not_orphaned_by_a_seam(grid):
    """At five columns the seams fall at 0.4/0.6 and the philtral columns near
    0.42/0.58 — the specific case that motivated the overlap."""
    patches = [p for p in grid.generate(G2) if p.band == "middle"]
    for x in (0.42, 0.58):
        containing = [p for p in patches if p.x <= x <= p.x + p.w]
        assert len(containing) >= 2, f"{x} is covered by only {len(containing)} patch"


def test_bands_span_the_full_height():
    total = sum(b.y1 - b.y0 for b in P.DEFAULT_BANDS)
    assert total == pytest.approx(1.0, abs=TOL)


def test_patches_span_the_full_width(grid):
    for band in ("top", "middle", "bottom"):
        members = [p for p in grid.generate(G2) if p.band == band]
        assert min(p.x for p in members) == pytest.approx(0.0, abs=1e-9)
        assert max(p.x + p.w for p in members) == pytest.approx(1.0, abs=1e-9)


# --------------------------------------------------------------------------
# mirror symmetry -- the instrument
# --------------------------------------------------------------------------


def test_every_patch_has_a_mirror_partner(grid):
    for patch in grid.generate(G2):
        assert patch.mirror_id is not None


def test_patch_i_mirrors_patch_2n_minus_2_minus_i(grid):
    patches = {p.id: p for p in grid.generate(G2)}
    for band in P.DEFAULT_BANDS:
        members = sorted(
            [p for p in patches.values() if p.band == band.name], key=lambda p: p.x
        )
        n = len(members)
        for index, patch in enumerate(members):
            expected = members[n - 1 - index]
            assert patch.mirror_id == expected.id


def test_the_mirror_relation_is_involutive(grid):
    patches = {p.id: p for p in grid.generate(G2)}
    for patch in patches.values():
        assert patches[patch.mirror_id].mirror_id == patch.id


def test_mirrored_boxes_actually_reflect(grid):
    patches = {p.id: p for p in grid.generate(G2)}
    for patch in patches.values():
        partner = patches[patch.mirror_id]
        assert partner.x == pytest.approx(1.0 - patch.x - patch.w, abs=1e-9)
        assert partner.w == pytest.approx(patch.w, abs=1e-9)


def test_each_band_has_exactly_one_centre_patch(grid):
    patches = grid.generate(G2)
    for band in P.DEFAULT_BANDS:
        centred = [
            p for p in patches
            if p.band == band.name and abs(p.centre_x - 0.5) <= 1e-9
        ]
        assert len(centred) == 1


def test_the_centre_patch_is_its_own_mirror(grid):
    patches = {p.id: p for p in grid.generate(G2)}
    for patch in patches.values():
        if abs(patch.centre_x - 0.5) <= 1e-9:
            assert patch.mirror_id == patch.id


def test_a_broken_mirror_set_is_rejected():
    """The guard has teeth: nudge one patch and pairing must fail."""
    broken = [
        P.Patch(0, "top", 0.0, 0.0, 0.3, 0.3),
        P.Patch(1, "top", 0.71, 0.0, 0.3, 0.3),  # should be at 0.70
    ]
    with pytest.raises(P.PatchError, match="no mirror partner"):
        P.pair_mirrors(broken)


def test_an_even_column_count_is_rejected_for_lacking_a_centre():
    """2n-1 is odd by construction; a scheme without a centre patch is caught."""
    config = P.PatchConfig(bands=(P.BandSpec("only", 0.0, 1.0, 2),))
    patches = P.get("grid").generate(G2, config)
    assert len(patches) == 3, "2 columns still give 2n-1 = 3 positions"
    assert any(abs(p.centre_x - 0.5) <= 1e-9 for p in patches)


# --------------------------------------------------------------------------
# the trapezium boundary (brief §4.4)
# --------------------------------------------------------------------------


def test_under_g2_every_patch_is_fully_inside_the_mask(grid):
    """Exit criterion 2. G2's mask is the whole square, so nothing is clipped."""
    for patch in grid.generate(G2):
        assert patch.coverage == pytest.approx(1.0, abs=1e-6)


def test_under_g1_the_outer_top_patches_lose_area(grid):
    """The trapezium is narrowest at the brow, so that is where white appears."""
    patches = grid.generate(G1)
    top = sorted([p for p in patches if p.band == "top"], key=lambda p: p.x)
    assert top[0].coverage < 1.0
    assert top[-1].coverage < 1.0
    assert top[len(top) // 2].coverage == pytest.approx(1.0, abs=1e-6)


def test_under_g1_coverage_loss_is_left_right_symmetric(grid):
    """An asymmetric loss would manufacture the very signal being measured."""
    patches = {p.id: p for p in grid.generate(G1)}
    for patch in patches.values():
        assert patch.coverage == pytest.approx(
            patches[patch.mirror_id].coverage, abs=1e-6
        )


def test_under_g1_every_band_loses_area_not_only_the_top(grid):
    """Corrected: the trapezium is full width only at y=1.0, not across the band.

    The bottom band starts at y=2/3, where the half-width is still 0.434 — so its
    outermost patch is 23% white. It is tempting to assume "full width at the lip"
    means the whole lip band is clean; it does not, and the G1 boundary rule has
    to cope with losses in all three bands rather than just the brow.
    """
    patches = grid.generate(G1)
    outermost = {}
    for band in ("top", "middle", "bottom"):
        members = [p for p in patches if p.band == band]
        outermost[band] = min(p.coverage for p in members)

    assert all(c < 1.0 for c in outermost.values()), (
        "every band's outer column crosses the mask edge under G1"
    )
    # Loss shrinks downward as the trapezium widens.
    assert outermost["top"] < outermost["middle"] < outermost["bottom"]
    assert outermost["bottom"] == pytest.approx(0.77, abs=0.02)


def test_only_the_very_bottom_row_is_full_width():
    """Where the "full width at the lip" intuition comes from, stated precisely."""
    assert G1.half_width_at(1.0) == pytest.approx(0.5, abs=TOL)
    assert G1.half_width_at(2 / 3) == pytest.approx(0.434, abs=0.002)


def test_drop_removes_low_coverage_patches_and_changes_the_count(grid):
    """The brief requires reporting the count, because dropping changes it."""
    config = P.PatchConfig(boundary_rule="drop", min_coverage=0.9)
    dropped = grid.generate(G1, config)
    assert len(dropped) < 27
    assert all(p.coverage >= 0.9 for p in dropped)


def test_drop_preserves_mirror_symmetry(grid):
    """Dropping must remove pairs, never one half of one."""
    config = P.PatchConfig(boundary_rule="drop", min_coverage=0.9)
    patches = {p.id: p for p in grid.generate(G1, config)}
    for patch in patches.values():
        assert patch.mirror_id in patches


def test_clip_keeps_the_count_and_pulls_patches_inside(grid):
    config = P.PatchConfig(boundary_rule="clip")
    clipped = grid.generate(G1, config)
    assert len(clipped) == 27
    for patch in clipped:
        assert patch.coverage == pytest.approx(1.0, abs=0.02)


def test_clip_stays_mirror_symmetric(grid):
    patches = {p.id: p for p in grid.generate(G1, P.PatchConfig(boundary_rule="clip"))}
    for patch in patches.values():
        partner = patches[patch.mirror_id]
        assert partner.x == pytest.approx(1.0 - patch.x - patch.w, abs=1e-9)


def test_an_unknown_boundary_rule_is_rejected():
    with pytest.raises(P.PatchError, match="unknown boundary_rule"):
        P.PatchConfig(boundary_rule="shrink")


def test_a_rule_that_removes_everything_is_rejected(grid):
    config = P.PatchConfig(boundary_rule="drop", min_coverage=1.5)
    with pytest.raises(P.PatchError, match="removed every patch"):
        grid.generate(G1, config)


# --------------------------------------------------------------------------
# configurability (supervision left the count to the student)
# --------------------------------------------------------------------------


def test_columns_per_band_are_configurable():
    config = P.PatchConfig(
        bands=(
            P.BandSpec("top", 0.0, 0.4, 2),
            P.BandSpec("bottom", 0.4, 1.0, 4),
        )
    )
    patches = P.get("grid").generate(G2, config)
    assert len(patches) == 3 + 7 == config.expected_patches


def test_band_boundaries_are_configurable_away_from_thirds():
    """If they land badly on the contact sheet, they get moved — not tuned."""
    config = P.PatchConfig(
        bands=(
            P.BandSpec("top", 0.0, 0.30, 3),
            P.BandSpec("middle", 0.30, 0.62, 5),
            P.BandSpec("bottom", 0.62, 1.0, 7),
        )
    )
    patches = P.get("grid").generate(G2, config)
    assert len(patches) == 27
    top = next(p for p in patches if p.band == "top")
    assert top.h == pytest.approx(0.30, abs=TOL)


def test_output_size_is_one_value_for_every_patch():
    """Resolution by coverage: do NOT vary output size per band."""
    config = P.PatchConfig()
    assert isinstance(config.output_size, int)
    assert not hasattr(P.BandSpec("b", 0.0, 1.0, 3), "output_size")


def test_an_invalid_overlap_is_rejected():
    for bad in (0.0, 1.0, 1.5, -0.2):
        with pytest.raises(P.PatchError, match="overlap"):
            P.PatchConfig(overlap=bad)


def test_an_invalid_band_extent_is_rejected():
    with pytest.raises(P.PatchError, match="invalid extent"):
        P.PatchConfig(bands=(P.BandSpec("bad", 0.6, 0.4, 3),))


# --------------------------------------------------------------------------
# the shared interface
# --------------------------------------------------------------------------


def test_the_grid_generator_is_registered():
    assert "grid" in P.GENERATORS
    assert P.get("grid").name == "grid"


def test_an_unknown_generator_is_rejected():
    with pytest.raises(P.PatchError, match="unknown generator"):
        P.get("saliency")


def test_generation_is_deterministic(grid):
    assert [p.box for p in grid.generate(G2)] == [p.box for p in grid.generate(G2)]


def test_the_summary_is_aggregate_only(grid):
    summary = P.summarise(grid.generate(G1))
    assert summary["n_patches"] == 27
    assert summary["per_band"] == {"bottom": 13, "middle": 9, "top": 5}
    assert 0.0 < summary["coverage_min"] <= summary["coverage_mean"] <= 1.0
    assert summary["white_fraction_max"] == pytest.approx(
        1.0 - summary["coverage_min"], abs=TOL
    )
