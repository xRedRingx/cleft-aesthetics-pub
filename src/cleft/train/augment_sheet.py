"""Phase 7C exit criterion 5: what is protected, and what augmentation does.

**The one check no test substitutes for.** A protect-list covering the wrong
pixels passes every assertion in the suite -- the names resolve, the mask is
soft, the boxes are inside the frame, the arm trains and reports a plausible
number -- and is wrong in the only way that matters. Region names are a
specification; where they land on a face is a fact about the mapping, and the
mapping is checked by eye or not at all.

That is the same finding the segmentation gate reached five times over: the
pre-declared criteria narrowed what was worth looking at and **the contact
sheet decided**, including once where screen and sheet disagreed outright
(PLAN Part 5). This sheet is the operative decision procedure for the
protect-list, and the write-up should say so.

----------------------------------------------------------------------------
IT RENDERS THE PIXELS THE ARM WILL SEE
----------------------------------------------------------------------------
Built from ``staged_v1`` and ``geometry.csv`` -- the same artifact and the
same per-patient content boxes the arms consume, through the same
``protection_masks_for`` -- rather than re-staged from the folders. A sheet
built on a parallel staging path could look right while the arm saw something
else, which is the failure it exists to catch.

Three panels per patient, because the review has two questions and they need
different evidence:

* **the crop** -- what the arm starts from;
* **the strength overlay** -- *is the right area protected?* The tint follows
  the falloff, so the softness is visible rather than asserted;
* **one augmented sample at full policy** -- *does the output still look like
  a plausible clinical photograph?* A policy can protect exactly the right
  anatomy and still produce something no camera would.

CLUSTER-ONLY: it renders patient faces. The report beside it carries counts
and fractions only and is SHAREABLE.
"""

from __future__ import annotations

import numpy as np

from ..geometry.render import Panel, tile

#: The protected area is tinted cyan. Deliberately NOT ``render.MASK_COLOUR``
#: (magenta), which already means "trapezium boundary" on every Phase 2 sheet
#: -- two meanings for one colour on sheets a reviewer sees side by side is
#: how a boundary gets read as a protected region.
PROTECTED_COLOUR = (0, 255, 255)

#: How strongly the tint shows at full protection. Enough to read at panel
#: scale, light enough that the anatomy under it stays judgeable -- the
#: reviewer is being asked whether the tint covers the right structures, which
#: needs both visible.
TINT_ALPHA = 0.45


def protection_overlay(
    image: np.ndarray, strength: np.ndarray, *, inside_strength: float
) -> np.ndarray:
    """The crop tinted where augmentation is damped.

    Alpha is the protection FRACTION -- ``(1 - strength) / (1 - inside)`` --
    so a fully protected pixel tints at ``TINT_ALPHA`` and an unprotected one
    not at all, with the falloff showing as a gradient. A binary tint would
    hide exactly the property the soft blend was chosen for, and a reviewer
    could not tell a hard mask from a soft one on the sheet.
    """
    array = np.asarray(image, dtype=np.float64)
    if inside_strength >= 1.0:
        alpha = np.zeros(array.shape[:2])
    else:
        alpha = np.clip(
            (1.0 - np.asarray(strength, dtype=np.float64)) / (1.0 - inside_strength),
            0.0, 1.0,
        )
    tint = np.asarray(PROTECTED_COLOUR, dtype=np.float64)
    blended = array + (TINT_ALPHA * alpha)[..., None] * (tint - array)
    return np.clip(blended, 0, 255).astype(np.uint8)


#: **[MEASURED 2026-08-03] ``moved_outside`` reported inversion on a correct
#: mask, and the cause was the white pad.**
#:
#: The first version averaged the pixel change over every unprotected pixel.
#: A staged crop is pad-square-then-resize with a **white** pad -- 0.2534 of
#: the frame at the median aspect ratio (PLAN §4.4) -- and flat white barely
#: moves under photometric jitter while skin moves a lot. So the outside
#: average was diluted by pixels that cannot move, and the comparison read
#: "inside moves more" on twelve panels whose masks were correct: **7 of 12
#: flagged inverted, and none of them was.**
#:
#: **The tell was patient 5** -- 47.3 inside against 47.5 outside, nearly
#: equal, and the darker-skinned patient from the Phase 4 fairness work whose
#: crop carries the least white margin relative to face. Least pad, least
#: dilution, no apparent inversion.
#:
#: **R2, in the form the tally tracks:** ``moved_outside`` is named for the
#: POLICY'S EFFECT and computed the average change over a REGION DEFINED BY
#: GEOMETRY. Those are different quantities, and the name is not evidence
#: about which one it is.
#:
#: The failure mode was a false alarm rather than a silent pass, which is the
#: safer direction for a diagnostic -- but it still cost a review round, and
#: a check that cries wolf is a check people learn to overrule.
#:
#: **Both figures are reported now, not just the corrected one.** Removing
#: the diluted number would hide the artefact; reporting them side by side
#: makes the size of it visible per patient, which is the same treatment
#: ``separability`` got when it turned out to be measuring the wrong contrast.
MOVED_OUTSIDE_WAS_DILUTED_BY_THE_PAD = {
    "measured": "2026-08-03",
    "symptom": "7 of 12 patients flagged as protection-inverted on a correct mask",
    "cause": (
        "moved_outside averaged over every unprotected pixel, including the "
        "white pad -- 0.2534 of the frame at the median aspect ratio (PLAN "
        "§4.4) -- and flat white barely moves under photometric jitter"
    ),
    "tell": (
        "patient 5, 47.3 inside against 47.5 outside: the darker-skinned "
        "patient from the Phase 4 fairness work, whose crop carries the least "
        "white margin relative to face. Least pad, least dilution"
    ),
    "r2": (
        "named for the policy's effect, computed as the average change over a "
        "region defined by geometry. Different quantities; the name is not "
        "evidence about which"
    ),
    "fix": "restrict both figures to content pixels -- inside the trapezium, "
           "excluding the pad",
    "both_reported": (
        "the diluted figure is kept beside the corrected one so the artefact "
        "is visible rather than removed"
    ),
    "direction": (
        "a false alarm rather than a silent pass -- the safer failure for a "
        "diagnostic, and still a review round"
    ),
}


def content_mask(size: int, content_box, geometry: str) -> np.ndarray:
    """Pixels that are image content: inside the trapezium, outside the pad.

    Two exclusions, and both are white by construction so both dilute:

    * **the pad** -- ``content_box`` is where the original pixels landed in
      the padded square (``staging.Staged``), so everything outside it is fill;
    * **the trapezium corners** -- at G1 they stay in the pixels and are
      excluded from the computation (PLAN §4.4). White there too.

    Built from the frozen ``Trapezium`` edge arrays, evaluated across the
    content box rather than the square, because the trapezium is defined
    relative to the crop content and the content box is not square.
    """
    from ..geometry.patch_features import trapezium_for

    x0, y0, width, height = (int(v) for v in content_box)
    mask = np.zeros((size, size), dtype=bool)
    if width <= 0 or height <= 0:
        return mask

    trapezium = trapezium_for(geometry)
    ys = (np.arange(height) + 0.5) / height
    xs = (np.arange(width) + 0.5) / width
    left = trapezium.left_edge_array(ys)
    right = trapezium.right_edge_array(ys)
    inside = (xs[None, :] >= left[:, None]) & (xs[None, :] <= right[:, None])

    x1, y1 = min(x0 + width, size), min(y0 + height, size)
    mask[y0:y1, x0:x1] = inside[: y1 - y0, : x1 - x0]
    return mask


def per_patient_report(
    image: np.ndarray,
    augmented: np.ndarray,
    strength: np.ndarray,
    *,
    patient_id: int,
    inside_strength: float,
    content: np.ndarray | None = None,
) -> dict:
    """The numbers beside the picture, so the review is not eye alone.

    ``moved_inside`` against ``moved_outside`` is the numeric form of the
    visual question: augmentation must move protected pixels LESS.

    **Both are restricted to content pixels**, and the whole-frame figures are
    reported beside them -- see ``MOVED_OUTSIDE_WAS_DILUTED_BY_THE_PAD``. The
    comparison is only like-with-like inside the content: the pad is a quarter
    of the frame, it is white, and white does not move.
    """
    protection = (
        np.zeros(strength.shape)
        if inside_strength >= 1.0
        else np.clip((1.0 - strength) / (1.0 - inside_strength), 0.0, 1.0)
    )
    moved = np.abs(
        np.asarray(augmented, dtype=np.float64)
        - np.asarray(image, dtype=np.float64)
    ).mean(axis=2)

    protected = protection > 0.5
    exposed = protection < 0.05
    if content is None:
        content = np.ones(moved.shape, dtype=bool)

    def mean_over(selection):
        return float(moved[selection].mean()) if selection.any() else None

    return {
        "patient_id": int(patient_id),
        "protected_fraction_of_frame": float(protection.mean()),
        "protected_fraction_of_content": (
            float(protection[content].mean()) if content.any() else None
        ),
        "fully_protected_fraction": float(protected.mean()),
        "content_fraction_of_frame": float(content.mean()),
        "min_strength": float(strength.min()),
        "max_strength": float(strength.max()),
        #: The corrected figures: content pixels only, so both sides are made
        #: of things that can move.
        "moved_inside": mean_over(protected & content),
        "moved_outside": mean_over(exposed & content),
        #: The diluted figures, kept so the artefact stays visible rather than
        #: being quietly removed.
        "moved_inside_whole_frame": mean_over(protected),
        "moved_outside_whole_frame": mean_over(exposed),
    }


def build_sheet(
    images: np.ndarray,
    geometry_rows: list,
    patient_ids: list,
    *,
    rows_to_render: list,
    geometry: str,
    policy,
    protected_regions,
    photometric_settings: dict,
    geometric_settings: dict,
    inside_strength: float,
    sigma_fraction: float,
    seed: int,
) -> tuple:
    """Three panels per selected patient, plus the SHAREABLE report.

    ``rows_to_render`` indexes into ``images``. The augmented sample is drawn
    at ``(seed, fold=0, epoch=0)`` so the sheet is reproducible: a review that
    showed a different sample each run could not be returned to, and the
    per-patient rows would not correspond to what anyone approved.
    """
    from ..geometry.patch_features import staged_from_row
    from .augment import augment_batch, protection_masks_for

    masks = protection_masks_for(
        images, geometry_rows, protected_regions,
        geometry=geometry,
        inside_strength=inside_strength,
        sigma_fraction=sigma_fraction,
    )
    augmented = augment_batch(
        images,
        policy=policy,
        seed=seed, fold=0, epoch=0,
        photometric_settings=photometric_settings,
        geometric_settings=geometric_settings,
        strength_per_image=masks if policy.region_aware else None,
    )

    panels, report = [], []
    for row in rows_to_render:
        patient_id = int(patient_ids[row])
        crop = np.asarray(images[row], dtype=np.uint8)
        panels.extend([
            Panel(f"{patient_id} crop", crop),
            Panel(
                f"{patient_id} protected",
                protection_overlay(
                    crop, masks[row], inside_strength=inside_strength
                ),
            ),
            Panel(
                f"{patient_id} augmented",
                np.clip(augmented[row], 0, 255).astype(np.uint8),
            ),
        ])
        staged = staged_from_row(crop, geometry_rows[row])
        report.append(
            per_patient_report(
                crop, augmented[row], masks[row],
                patient_id=patient_id, inside_strength=inside_strength,
                content=content_mask(crop.shape[0], staged.content_box, geometry),
            )
        )

    # Three columns, so each patient is one row and the eye compares across
    # rather than hunting for the matching panel.
    return tile(panels, columns=3), report


def summarise(report: list[dict]) -> dict:
    """What the reviewer is being asked to confirm, as numbers.

    ``moved_inside_exceeds_outside`` is reported as a COUNT rather than
    asserted: if it ever holds, the region policy is doing the opposite of
    what it says on that patient, and that belongs on the sheet's summary
    where a reviewer sees it -- not as an exception that stops the render
    before anyone can look at the panel that would explain it.
    """
    def flagged(inside_key, outside_key):
        # `.get`, because a caller may report only the corrected pair -- a
        # missing figure is "not compared", not "not inverted by zero".
        return [
            row["patient_id"] for row in report
            if row.get(inside_key) is not None and row.get(outside_key) is not None
            and row[inside_key] >= row[outside_key]
        ]

    protected = [row["protected_fraction_of_frame"] for row in report]
    inverted = flagged("moved_inside", "moved_outside")
    #: What the SAME check said before the pad was excluded. Reported so the
    #: size of the artefact is visible per sheet rather than described once in
    #: a docstring -- see MOVED_OUTSIDE_WAS_DILUTED_BY_THE_PAD.
    diluted = flagged("moved_inside_whole_frame", "moved_outside_whole_frame")
    return {
        "n_patients": len(report),
        "protected_fraction_mean": float(np.mean(protected)) if protected else None,
        "protected_fraction_min": float(np.min(protected)) if protected else None,
        "protected_fraction_max": float(np.max(protected)) if protected else None,
        "patients_where_protection_inverted": inverted,
        "n_inverted": len(inverted),
        "n_inverted_whole_frame": len(diluted),
        "dilution_note": (
            "n_inverted_whole_frame is the SAME check over the whole frame, "
            "including the white pad. Where it exceeds n_inverted the "
            "difference is pad dilution, not policy"
        ),
        "role": "review",
        "decided_by": "human review of the contact sheet",
        "note": (
            "a protect-list covering the wrong pixels passes every test in "
            "the suite. Region names are a specification; where they land on "
            "a face is a fact about the mapping, and the mapping is checked "
            "by eye or not at all"
        ),
    }
