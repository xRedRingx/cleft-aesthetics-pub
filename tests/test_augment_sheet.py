"""The Phase 7C contact sheet: the check that decides what tests cannot.

Every assertion in the suite passes on a protect-list covering the wrong
pixels. What these tests can check is that the sheet would SHOW the wrong
thing if it were wrong -- that the overlay follows the falloff rather than
binarising it, that the numbers beside the picture measure the property the
policy claims, and that the panels are the pixels the arm will actually see.
"""

from __future__ import annotations

import numpy as np
import pytest

from cleft import phase7c
from cleft.train import augment, augment_sheet


def test_the_overlay_follows_the_falloff_rather_than_binarising_it():
    """**A binary tint would hide the property the soft blend was chosen
    for.** A reviewer could not tell a hard mask from a soft one on the
    sheet, which is the thing they are being asked to confirm."""
    strength = augment.protection_mask(
        64, 64, [(20, 20, 24, 24)],
        inside_strength=phase7c.REGION_AWARE["inside_strength"],
        sigma_fraction=phase7c.BLEND["sigma_fraction_of_crop_width"],
    )
    crop = np.full((64, 64, 3), 120, dtype=np.uint8)
    overlay = augment_sheet.protection_overlay(
        crop, strength, inside_strength=phase7c.REGION_AWARE["inside_strength"]
    )

    assert overlay.shape == crop.shape and overlay.dtype == np.uint8
    # Distinct tint levels, not two.
    levels = np.unique(overlay[..., 2])
    assert len(levels) > 5, (
        f"the overlay takes {len(levels)} values; it has binarised the mask "
        "and the falloff is invisible on the sheet"
    )
    # Protected centre tinted, far corner untouched.
    assert overlay[32, 32, 2] > crop[32, 32, 2]
    assert tuple(overlay[0, 0]) == tuple(crop[0, 0])


def test_the_tint_colour_is_not_the_boundary_colour():
    """Magenta already means trapezium boundary on every Phase 2 sheet. Two
    meanings for one colour, on sheets a reviewer sees side by side, is how a
    boundary gets read as a protected region."""
    from cleft.geometry.render import MASK_COLOUR

    assert augment_sheet.PROTECTED_COLOUR != MASK_COLOUR


def test_an_unprotected_policy_tints_nothing():
    """``inside_strength`` of 1.0 is "protect nothing", and the overlay must
    say so rather than tinting the whole frame."""
    crop = np.full((16, 16, 3), 90, dtype=np.uint8)
    overlay = augment_sheet.protection_overlay(
        crop, np.ones((16, 16)), inside_strength=1.0
    )
    assert np.array_equal(overlay, crop)


def test_the_report_measures_the_property_the_policy_claims():
    """``moved_inside`` against ``moved_outside`` is the numeric form of the
    visual question. On fixtures it is a unit test; here it is the same
    quantity on real crops, where the boxes are placed by each patient's own
    content box -- which is exactly what could be wrong."""
    strength = augment.protection_mask(
        64, 64, [(16, 16, 32, 32)], inside_strength=0.25, sigma_fraction=0.02
    )
    crop = np.full((64, 64, 3), 100.0)
    # An "augmented" image that moved unprotected pixels more, as the policy
    # intends.
    protection = np.clip((1.0 - strength) / 0.75, 0.0, 1.0)
    augmented = crop + 40.0 * (1.0 - protection)[..., None]

    entry = augment_sheet.per_patient_report(
        crop, augmented, strength, patient_id=7, inside_strength=0.25
    )
    assert entry["patient_id"] == 7
    assert 0.0 < entry["protected_fraction_of_frame"] < 1.0
    assert entry["moved_inside"] < entry["moved_outside"]
    assert entry["min_strength"] == pytest.approx(0.25, abs=0.02)
    assert entry["max_strength"] == pytest.approx(1.0, abs=1e-6)


def test_the_white_pad_dilutes_the_whole_frame_comparison():
    """**[MEASURED 2026-08-03] The artefact, reproduced.**

    A staged crop is pad-square-then-resize with a WHITE pad -- 0.2534 of the
    frame at the median aspect ratio -- and flat white barely moves under
    photometric jitter. Averaging the outside over the whole frame therefore
    mixes skin with pixels that cannot move, and reported inversion on twelve
    correct masks.

    Built here so the fix is demonstrated rather than asserted: the same
    numbers must invert on the whole frame and not on the content.
    """
    size, pad = 64, 8  # 25% pad, the median aspect ratio's 0.2534
    content = np.zeros((size, size), dtype=bool)
    content[:, pad : size - pad] = True
    assert content.mean() == pytest.approx(0.75)

    strength = np.ones((size, size))
    strength[28:36, 28:36] = 0.25  # a protected box, inside the content

    crop = np.full((size, size, 3), 120.0)
    crop[:, :pad] = 255.0
    crop[:, size - pad :] = 255.0

    # Content moves 10 outside the protected box and 8 inside it -- so the
    # policy IS working, at the weak damping the real sheet showed (47.3
    # against 47.5) -- and the white pad moves not at all.
    augmented = crop.copy()
    moved = np.zeros((size, size))
    moved[content] = 10.0
    moved[28:36, 28:36] = 8.0
    augmented += moved[..., None]

    entry = per_patient(crop, augmented, strength, content=content)
    whole = per_patient(crop, augmented, strength, content=None)

    # Corrected: the policy shows through.
    assert entry["moved_inside"] < entry["moved_outside"]
    # Diluted: the same data reads as inverted, because the pad is in the
    # outside average and contributes zero.
    assert whole["moved_inside"] >= whole["moved_outside"], (
        "the fixture no longer reproduces the dilution, so this test is not "
        "demonstrating the artefact it was written for"
    )
    # And both are reported, so the size of it is visible.
    assert entry["moved_outside_whole_frame"] < entry["moved_outside"]
    assert entry["content_fraction_of_frame"] == pytest.approx(
        (size - 2 * pad) / size
    )


def per_patient(crop, augmented, strength, *, content):
    return augment_sheet.per_patient_report(
        crop, augmented, strength,
        patient_id=1, inside_strength=0.25, content=content,
    )


def test_the_content_mask_excludes_the_pad_and_the_trapezium_corners():
    """Both exclusions are white by construction, so both dilute. The mask is
    built from the frozen ``Trapezium`` edges across the content box, because
    the trapezium is defined relative to the crop content and the content box
    is not square."""
    size = 224
    # A tall crop: pad down the sides, content in the middle.
    mask = augment_sheet.content_mask(size, (20, 0, size - 40, size), "g1")

    assert mask.shape == (size, size)
    assert not mask[:, :20].any(), "the pad is inside the content mask"
    assert not mask[:, size - 20 :].any()
    assert mask.any(), "the content mask is empty"

    # G1 keeps the trapezium corners in the pixels and out of the computation,
    # so the top row must be narrower than the middle.
    top = mask[2].sum()
    middle = mask[size // 2].sum()
    assert top < middle, (
        "the mask is a plain rectangle; the trapezium corners are still in it "
        "and they are white"
    )

    # A degenerate box is empty rather than an exception -- a patient with no
    # content is a data question, not a crash here.
    assert not augment_sheet.content_mask(32, (0, 0, 0, 0), "g1").any()


def test_inverted_protection_is_reported_not_raised():
    """If augmentation moves protected pixels MORE on some patient, that
    belongs on the summary where a reviewer sees it -- beside the panel that
    would explain it -- rather than stopping the render before anyone can
    look."""
    report = [
        {"patient_id": 1, "protected_fraction_of_frame": 0.3,
         "moved_inside": 1.0, "moved_outside": 4.0,
         "moved_inside_whole_frame": 1.0, "moved_outside_whole_frame": 0.5},
        {"patient_id": 2, "protected_fraction_of_frame": 0.4,
         "moved_inside": 5.0, "moved_outside": 2.0,
         "moved_inside_whole_frame": 5.0, "moved_outside_whole_frame": 1.0},
    ]
    summary = augment_sheet.summarise(report)
    assert summary["patients_where_protection_inverted"] == [2]
    assert summary["n_inverted"] == 1
    assert summary["n_patients"] == 2
    assert summary["decided_by"] == "human review of the contact sheet"

    # The whole-frame count is reported beside it, so the size of the pad
    # artefact is visible on every sheet rather than described once.
    assert summary["n_inverted_whole_frame"] == 2
    assert "pad dilution, not policy" in summary["dilution_note"]


def test_the_summary_records_that_the_sheet_decides():
    """The segmentation gate's lesson, carried: the criteria narrow what is
    worth looking at and the sheet decides. Recording ``role: review`` is what
    stops the write-up describing this as an automated check."""
    summary = augment_sheet.summarise([
        {"patient_id": 1, "protected_fraction_of_frame": 0.3,
         "moved_inside": 1.0, "moved_outside": 2.0},
    ])
    assert summary["role"] == "review"
    assert "checked by eye or not at all" in summary["note"]


def test_the_sheet_has_three_panels_per_patient_in_row_order():
    """One patient per row, so the eye compares across rather than hunting
    for the matching panel."""
    n, size = 4, 24
    rng = np.random.default_rng(0)
    images = rng.integers(0, 255, (n, size, size, 3)).astype(np.uint8)
    geometry_rows = [
        {"patient_id": str(i + 1), "content_x": "0", "content_y": "0",
         "content_w": str(size), "content_h": str(size)}
        for i in range(n)
    ]

    sheet, report = augment_sheet.build_sheet(
        images, geometry_rows, [1, 2, 3, 4],
        rows_to_render=[0, 2],
        geometry="g1",
        policy=augment.Policy(photometric=True, geometric=True,
                              region_aware=True),
        protected_regions=phase7c.PROTECTED_REGIONS,
        photometric_settings=phase7c.PHOTOMETRIC,
        geometric_settings=phase7c.GEOMETRIC,
        inside_strength=phase7c.REGION_AWARE["inside_strength"],
        sigma_fraction=phase7c.BLEND["sigma_fraction_of_crop_width"],
        seed=1337,
    )

    assert sheet.ndim == 3 and sheet.shape[2] == 3
    assert [entry["patient_id"] for entry in report] == [1, 3]
    for entry in report:
        assert entry["moved_inside"] is not None
        assert entry["moved_outside"] is not None


def test_the_real_anatomy_regions_resolve_and_land_in_the_lower_face():
    """**The mask path end to end, unmocked** -- the anatomy generator, the
    per-patient mapping, and the falloff, on a real content box.

    This is as close as a test gets to the sheet's question. It cannot say the
    boxes are on the right anatomy, which is what the review is for, but it
    can say the protected mass sits in the lower two-thirds of the frame: the
    protect-list is nose and upper lip, and a mask centred on the forehead
    would be wrong in a way arithmetic CAN see.
    """
    size = 224
    image = np.full((1, size, size, 3), 128, dtype=np.uint8)
    rows = [{
        "patient_id": "1", "content_x": "10", "content_y": "0",
        "content_w": str(size - 20), "content_h": str(size),
        "aspect_ratio": "0.74",
    }]

    masks = augment.protection_masks_for(
        image, rows, phase7c.PROTECTED_REGIONS,
        geometry="g1",
        inside_strength=phase7c.REGION_AWARE["inside_strength"],
        sigma_fraction=phase7c.BLEND["sigma_fraction_of_crop_width"],
    )
    assert masks.shape == (1, size, size)

    protection = 1.0 - masks[0]
    assert protection.max() > 0.5, "nothing was protected at all"

    rows_weight = protection.sum(axis=1)
    centroid_y = float((rows_weight * np.arange(size)).sum() / rows_weight.sum())
    assert centroid_y > size * 0.45, (
        f"the protected mass is centred at y={centroid_y:.0f} of {size} -- "
        "the protect-list is nose and upper lip, so it should sit in the "
        "lower face, not the forehead"
    )

    # The forehead is left augmentable, which is what glabella being excluded
    # means in pixels.
    assert protection[: size // 8].mean() < protection[size // 2 :].mean()


def test_an_empty_protect_list_is_refused_rather_than_protecting_nothing():
    """A mask that protects nothing would make arms 4 and 5 identical to 1 and
    5's comparison vacuous, and the run would complete."""
    image = np.full((1, 64, 64, 3), 128, dtype=np.uint8)
    rows = [{"patient_id": "1", "content_x": "0", "content_y": "0",
             "content_w": "64", "content_h": "64", "aspect_ratio": "1.0"}]

    with pytest.raises(augment.AugmentError, match="would protect nothing"):
        augment.protection_masks_for(
            image, rows, ("no_such_region",), geometry="g1",
            inside_strength=0.25, sigma_fraction=0.02,
        )


def test_the_sheet_config_renders_the_full_policy_at_the_baseline_geometry(
    repo_root, monkeypatch
):
    """The review is of the protection mask and of what FULL-strength
    augmentation does to it. A sheet rendered at some weaker policy would be
    evidence about neither."""
    from cleft.config import load_config

    monkeypatch.setenv("CLEFT_SCUT_ROOT", "/nonexistent/sentinel")
    cfg = load_config(repo_root / "configs" / "p7c_contact_sheet.yaml")
    task = cfg["task"]

    assert task["kind"] == "augmentation_contact_sheet"
    # The arms run at G1 -- the baseline cell.
    assert task["geometry"] == phase7c.BASELINE["geometry"] == "g1"
    assert task["n_patients"] >= 9

    # It reads the same artifacts the arms do, so the sheet cannot show
    # pixels from a parallel staging path.
    names = {entry["name"] for entry in cfg["inputs"]}
    assert {"manifest_v1", "staged_v1"} <= names

    # And the arm whose policy it renders is the full one.
    full = next(a for a in phase7c.arms() if a["name"] == "p7c_5_region_full")
    assert full["photometric"] and full["geometric"] and full["region_aware"]
