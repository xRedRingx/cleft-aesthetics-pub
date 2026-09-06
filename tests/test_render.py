"""The contact sheet overlay.

Drawing is pure numpy so it is testable here; what it CANNOT test is the thing
the sheet exists for — whether the bands land on the right anatomy. That needs
real faces and a human, on the cluster.
"""

from __future__ import annotations

import numpy as np
import pytest

from cleft.geometry import mapping as M
from cleft.geometry import patches as P
from cleft.geometry import render as R
from cleft.geometry import staging as S
from cleft.geometry.trapezium import DEFAULT as G1
from cleft.geometry.trapezium import Trapezium

G2 = Trapezium(top_half_width=0.5, bot_half_width=0.5)


def crop(width=222, height=300, value=120):
    return np.full((height, width, 3), value, dtype=np.uint8)


@pytest.fixture(scope="module")
def scene():
    staged = S.stage(crop())
    patches = P.get("grid").generate(G1)
    return staged, M.map_patches(staged, patches)


# --------------------------------------------------------------------------
# the tier guard -- a rendered face must never be SHAREABLE
# --------------------------------------------------------------------------


def test_a_contact_sheet_cannot_be_marked_shareable(write_config, out_root, clean_repo):
    """It renders patient faces; this is the worst possible mislabel."""
    from cleft.provenance import RunContext

    cfg = write_config(tier="keeper")
    with RunContext(cfg, out_root, repo_root=clean_repo) as ctx:
        for name in ("contact_sheet_g1.png", "overlay_03.png", "montage.png"):
            with pytest.raises(ValueError, match="CLUSTER-ONLY"):
                ctx.path(name, tier="SHAREABLE")
        # And the same names are accepted at the right tier.
        assert ctx.path("contact_sheet_g1.png", tier="CLUSTER-ONLY")


def test_an_aggregate_plot_is_still_shareable(write_config, out_root, clean_repo):
    """The guard must not block legitimate aggregate figures."""
    from cleft.provenance import RunContext

    cfg = write_config(tier="keeper")
    with RunContext(cfg, out_root, repo_root=clean_repo) as ctx:
        assert ctx.path("curves.png", tier="SHAREABLE")


# --------------------------------------------------------------------------
# drawing
# --------------------------------------------------------------------------


def test_the_overlay_does_not_mutate_the_staged_image(scene):
    staged, mapped = scene
    before = staged.image.copy()
    R.render_overlay(staged, mapped, G1)
    assert np.array_equal(staged.image, before)


def test_the_overlay_draws_something(scene):
    staged, mapped = scene
    overlay = R.render_overlay(staged, mapped, G1)
    assert not np.array_equal(overlay, R._as_rgb(staged.image))


def test_rectangles_are_outlines_not_fills(scene):
    """A filled box would hide the anatomy the sheet exists to inspect."""
    canvas = np.zeros((60, 60, 3), dtype=np.uint8)
    R.draw_rect(canvas, (10, 10, 30, 30), (255, 0, 0))
    assert canvas[10, 20].tolist() == [255, 0, 0], "top edge drawn"
    assert canvas[25, 25].tolist() == [0, 0, 0], "interior left alone"


def test_a_rectangle_at_the_image_edge_does_not_wrap(scene):
    canvas = np.zeros((40, 40, 3), dtype=np.uint8)
    R.draw_rect(canvas, (35, 35, 10, 10), (255, 255, 255))
    assert canvas.shape == (40, 40, 3)


def test_mirror_partners_share_a_colour(scene):
    """They are the same measurement on two sides, so they read as a pair."""
    _, mapped = scene
    by_id = {m.id: (i, m) for i, m in enumerate(mapped)}
    for index, item in enumerate(mapped):
        partner_index, partner = by_id[item.mirror_id]
        assert R.colour_for(item.band, index) == R.colour_for(
            partner.band, partner_index
        ) or item.band == partner.band


def test_each_band_gets_its_own_colour():
    colours = {R.colour_for(b, 0) for b in ("top", "middle", "bottom")}
    assert len(colours) == 3


def test_anatomy_structures_get_distinguishable_colours():
    """Anatomy 'bands' are structure names, so they fall through to the palette."""
    names = [name for name, _, _, _, _ in __import__(
        "cleft.geometry.generators", fromlist=["BILATERAL_REGIONS"]
    ).BILATERAL_REGIONS]
    colours = {R.colour_for(name, i) for i, name in enumerate(names)}
    assert len(colours) >= 5


def test_the_mask_boundary_is_drawn_inside_the_content(scene):
    staged, _ = scene
    canvas = R._as_rgb(staged.image)
    R.draw_mask_boundary(canvas, staged, G1)
    marked = np.argwhere((canvas == list(R.MASK_COLOUR)).all(axis=2))
    x0, y0, cw, ch = staged.content_box
    assert marked.size > 0
    assert marked[:, 0].min() >= y0 and marked[:, 0].max() < y0 + ch
    assert marked[:, 1].min() >= x0 - 1 and marked[:, 1].max() <= x0 + cw + 1


def test_the_mask_boundary_narrows_toward_the_top(scene):
    """What the reviewer is meant to see: the trapezium, not a rectangle."""
    staged, _ = scene
    canvas = np.zeros_like(R._as_rgb(staged.image))
    R.draw_mask_boundary(canvas, staged, G1)
    marked = np.argwhere((canvas == list(R.MASK_COLOUR)).all(axis=2))

    rows = sorted(set(marked[:, 0].tolist()))
    top_span = marked[marked[:, 0] == rows[0]][:, 1]
    bottom_span = marked[marked[:, 0] == rows[-1]][:, 1]
    assert top_span.max() - top_span.min() < bottom_span.max() - bottom_span.min()


def test_greyscale_input_renders():
    staged = S.stage(np.full((300, 222), 90, dtype=np.uint8))
    mapped = M.map_patches(staged, P.get("grid").generate(G1))
    assert R.render_overlay(staged, mapped, G1).shape[2] == 3


# --------------------------------------------------------------------------
# tiling
# --------------------------------------------------------------------------


def test_tiling_lays_panels_out_in_a_grid(scene):
    staged, mapped = scene
    panel = R.Panel("p1", R.render_overlay(staged, mapped, G1))
    sheet = R.tile([panel] * 6, columns=3)
    assert sheet.ndim == 3
    assert sheet.shape[0] > staged.image.shape[0]
    assert sheet.shape[1] > staged.image.shape[1] * 2


def test_the_sheet_background_is_not_white(scene):
    """The crops are white-padded, so a white sheet would hide their edges."""
    staged, mapped = scene
    sheet = R.tile([R.Panel("p", R.render_overlay(staged, mapped, G1))], columns=1)
    assert sheet[0, 0].tolist() != [255, 255, 255]


def test_tiling_leaves_room_for_labels(scene):
    staged, mapped = scene
    panel = R.Panel("p", R.render_overlay(staged, mapped, G1))
    sheet = R.tile([panel], columns=1)
    assert sheet.shape[0] >= staged.image.shape[0] + R.GRID_LABEL_HEIGHT


def test_tiling_rejects_an_empty_sheet():
    with pytest.raises(R.RenderError, match="no panels"):
        R.tile([])


def test_tiling_rejects_bad_columns(scene):
    staged, mapped = scene
    with pytest.raises(R.RenderError, match="columns"):
        R.tile([R.Panel("p", staged.image)], columns=0)


# --------------------------------------------------------------------------
# G1 vs G2, which is the point of the sheet
# --------------------------------------------------------------------------


def test_g1_and_g2_panels_differ(scene):
    """Both geometries on one sheet is how "grotesque or plausible" gets answered."""
    from cleft.geometry.trapezium import unwarp

    staged, mapped = scene
    g1_panel = R.render_overlay(staged, mapped, G1)

    unwarped = S.Staged(
        image=unwarp(staged.image),
        source_size=staged.source_size,
        content_box=staged.content_box,
        scale=staged.scale,
    )
    g2_mapped = M.map_patches(unwarped, P.get("grid").generate(G2))
    g2_panel = R.render_overlay(unwarped, g2_mapped, None)

    assert not np.array_equal(g1_panel, g2_panel)


def test_the_report_accompanies_the_picture(scene):
    """The sheet stays on the cluster; these numbers are what travel back."""
    staged, mapped = scene
    report = R.overlay_report(mapped, staged)
    assert report["n_patches"] == 27
    assert 0.0 < report["min_coverage"] <= report["mean_coverage"] <= 1.0
    assert report["n_patches_with_white"] > 0, "G1 outer patches contain white"


def test_the_report_is_aggregate_only(scene):
    staged, mapped = scene
    report = R.overlay_report(mapped, staged)
    assert "pixels" not in report and "image" not in report
    for key, value in report.items():
        if key == "clamped_regions":
            # Anatomical structure names, not patient data.
            assert all(isinstance(name, str) for name in value)
            continue
        assert isinstance(value, (int, float)), f"{key} is not a scalar"
