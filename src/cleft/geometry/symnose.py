"""A mirrored upper-lip index. **NOT SymNose** -- see ``FAITHFULNESS``.

**RETIRED 2026-07-28. Phase 4 §3.2 is closed and SymNose is OUT, on a scope
boundary rather than on any result produced here.** This module is kept as the
record of what was built and what it measured; it is not an arm, and it must not
be described as a SymNose reimplementation in the write-up.

**What this computes:** an auto-segmented upper lip mirrored about the fixed
image midline. No rotation correction, no registration, no nose, one axis.

**What SymNose does:** levels the image on the intercanthal line first, outlines
the bottom of the nose *and* the upper lip, and produces four face-view
assessments each with a different anatomically-derived fold axis -- one
registering the extreme lateral points to maximise overlap, one superimposing
alar bases and mouth corners, one on a philtrum roundel, one on the mid-columella
line -- plus lip dehiscence, three user-outlined scar types and three base-view
measures.

Those are not the same instrument. ``FAITHFULNESS`` lists the differences and
``WHY_OUT`` records why no faithful version is being attempted.

**[LITERATURE] The validation design that was planned, and why it no longer
applies.** Strange's design was to score against SymNose-style output *and*
against the human ratings, and the protocol records excellent correlation between
human and SymNose scores *when ranked*. Comparing this index's numbers against
that finding would have been doubly wrong: PCC against a rank result is a
quantity confusion (R2), and more fundamentally **this is not the instrument that
produced the rank result**. ``SPEARMAN_IS_COMPARABLE`` is kept for the first
point; the second is why the comparison is not made at all.

**Two midlines, and the difference matters.** At G2 the mask is the full square
and the image midline is exact, so that is the primary axis -- the same choice
and the same reason as the mirror-difference index (§3.1). But a crop that is not
centred on the face turns miscentring into apparent asymmetry. Mirroring about
the mask's own centroid removes that, and removes genuine lateral displacement
with it, which is real asymmetry. **Neither axis is right on its own**, so both
are computed and ``centroid_offset`` reports how far apart they are. If that is
small the choice does not matter; if it is large, it is a confound to declare
rather than a parameter to pick.
"""

from __future__ import annotations

import numpy as np

from . import segmentation

#: **[MEASURED from the SymNose app bundle's help files, 2026-07-28] What the
#: instrument actually does, against what this module does.**
#:
#: The differences are not refinements to add later. Each of the first three
#: removes a confound that this index carries wholesale, and the fourth is a
#: different construct: SymNose outlines the bottom of the nose *and* the upper
#: lip, so its assessments are nasolabial rather than labial.
FAITHFULNESS = {
    "faithful": False,
    "symnose_does": (
        "levels the image on the intercanthal line before anything else; "
        "outlines the bottom of the nose AND the upper lip; produces four "
        "face-view assessments each on a different anatomically-derived fold "
        "axis -- registering the extreme lateral points to maximise overlap, "
        "superimposing alar bases and mouth corners, a philtrum roundel, and the "
        "mid-columella line; plus lip dehiscence, three user-outlined scar types "
        "and three base-view measures"
    ),
    "this_module_does": (
        "mirrors an auto-segmented upper lip about the FIXED IMAGE MIDLINE. No "
        "rotation correction. No registration. No nose. One axis."
    ),
    "confounds_carried": (
        "head tilt (no intercanthal levelling), crop centring (fixed midline "
        "rather than a registered axis), and lateral displacement (no "
        "registration step). Each of these plausibly DOMINATES any clinical "
        "asymmetry in the measurement, and none is separable from it afterwards."
    ),
}

#: **Why SymNose is out, and it is not because of anything measured here.**
#:
#: Every SymNose face measure depends on **user-placed roundels at named
#: anatomical points** -- canthi, alar bases, mouth corners, philtrum centre,
#: nose tip, mid-columella. Strange's proposal was to automate the *tracing*; the
#: roundels are separate manual work and equally essential to every assessment.
#:
#: So automating SymNose requires **landmark detection**. That method class is
#: outside this project's remit by supervisory decision: the supervision material rejected Bakaki's
#: landmark work because it is not deep learning, and the project is directed at
#: deep-learning methods.
#:
#: **This is a scope boundary, not a failure**, and the null result below is NOT
#: the evidence for it. The two are independent and must not be presented as
#: though one supported the other.
WHY_OUT = {
    "decision": "SymNose is OUT of Phase 4",
    "reason": "scope boundary",
    "argument": (
        "Every SymNose face measure depends on user-placed roundels at named "
        "anatomical points -- canthi, alar bases, mouth corners, philtrum centre, "
        "nose tip, mid-columella. Strange proposed automating the TRACING; the "
        "roundels are separate manual work and equally essential. Automating "
        "SymNose therefore requires LANDMARK DETECTION, which is outside this "
        "project's remit by supervisory decision -- the supervision material rejected Bakaki's "
        "landmark work because it is not deep learning and the project is "
        "directed at deep-learning methods."
    ),
    "not_evidenced_by": (
        "the null result in MEASURED_RESULT. That came from a non-faithful index "
        "and says nothing about SymNose. The scope argument stands on its own and "
        "the null must not be offered in support of it."
    ),
    "faithful_reimplementation": "not attempted, by decision",
}

#: The protocol's comparison is about rank order, so Spearman is the statistic
#: that can be set beside it. PCC answers a different question.
SPEARMAN_IS_COMPARABLE = (
    "The protocol reports excellent correlation between human scores and SymNose "
    "scores WHEN RANKED best to worst. That is a rank correlation, so Spearman "
    "is the comparable statistic. Quoting PCC against it would compare different "
    "quantities (PLAN §4.3). Both are reported; say which one a number is."
)

#: G2, for the same reason the mirror index requires it: the mask is the full
#: square, so the midline is exact and no per-image mapping stands between a
#: normalised axis and a pixel column.
REQUIRED_GEOMETRY = "g2"

AXES = ("image_midline", "mask_centroid")
DEFAULT_AXIS = "image_midline"


class SymNoseError(RuntimeError):
    """The SymNose index cannot be computed as asked."""


# --------------------------------------------------------------------------
# mirroring and superimposition
# --------------------------------------------------------------------------


def mirror_about(mask: np.ndarray, axis_x: float) -> np.ndarray:
    """Reflect a mask about a vertical line at ``axis_x`` (pixel coordinates)."""
    selected = np.asarray(mask, dtype=bool)
    width = selected.shape[1]
    source = np.rint(2.0 * axis_x - np.arange(width)).astype(int)
    valid = (source >= 0) & (source < width)

    mirrored = np.zeros_like(selected)
    mirrored[:, valid] = selected[:, source[valid]]
    return mirrored


def axis_for(mask: np.ndarray, axis: str = DEFAULT_AXIS) -> float:
    """The vertical line to mirror about, in pixel coordinates."""
    selected = np.asarray(mask, dtype=bool)
    if axis == "image_midline":
        return (selected.shape[1] - 1) / 2.0
    if axis == "mask_centroid":
        if not selected.any():
            return (selected.shape[1] - 1) / 2.0
        return float(np.nonzero(selected)[1].mean())
    raise SymNoseError(f"unknown axis {axis!r}; expected one of {AXES}")


def boundary_of(mask: np.ndarray) -> np.ndarray:
    """The traced outline: mask pixels with a non-mask neighbour.

    SymNose superimposes **boundaries**, not areas, which is why the primary
    index below is a boundary distance and the overlap scores are companions.
    """
    from scipy.ndimage import binary_erosion

    selected = np.asarray(mask, dtype=bool)
    if not selected.any():
        return selected.copy()
    return selected & ~binary_erosion(selected, border_value=0)


# --------------------------------------------------------------------------
# the measures
# --------------------------------------------------------------------------


def overlap_scores(mask: np.ndarray, mirrored: np.ndarray) -> dict:
    """Area agreement between a mask and its reflection."""
    a = np.asarray(mask, dtype=bool)
    b = np.asarray(mirrored, dtype=bool)
    intersection = int((a & b).sum())
    union = int((a | b).sum())
    total = int(a.sum()) + int(b.sum())

    return {
        "dice": round(2.0 * intersection / total, 5) if total else 0.0,
        "iou": round(intersection / union, 5) if union else 0.0,
        # Unmatched area as a fraction of the lip itself: "this much of the lip
        # has no counterpart on the other side".
        "unmatched_fraction": (
            round(float((a ^ b).sum() / a.sum()), 5) if a.any() else 0.0
        ),
    }


def boundary_distances(mask: np.ndarray, mirrored: np.ndarray) -> dict:
    """How far the traced outline sits from its reflection, in pixels.

    Distances are measured from every boundary pixel of the mask to the nearest
    boundary pixel of the mirrored mask. The 95th percentile is reported instead
    of the maximum: a single stray pixel should not define the index, and an
    unbounded maximum is what makes Hausdorff distance fragile on segmentations.
    """
    from scipy.ndimage import distance_transform_edt

    outline = boundary_of(mask)
    reflected = boundary_of(mirrored)
    if not outline.any() or not reflected.any():
        return {"mean_px": 0.0, "median_px": 0.0, "p95_px": 0.0, "n_boundary": 0}

    distance = distance_transform_edt(~reflected)
    values = distance[outline]
    return {
        "mean_px": round(float(values.mean()), 4),
        "median_px": round(float(np.median(values)), 4),
        "p95_px": round(float(np.percentile(values, 95)), 4),
        "n_boundary": int(outline.sum()),
    }


def half_measures(mask: np.ndarray, axis_x: float) -> dict:
    """Left against right, about the mirroring axis.

    The interpretable half of the index: "the left vermillion is 12% larger than
    the right" is a sentence a clinician can check on the photograph, which a
    boundary distance is not.
    """
    selected = np.asarray(mask, dtype=bool)
    if not selected.any():
        return {
            "left_area": 0, "right_area": 0,
            "area_difference_fraction": 0.0, "height_difference_px": 0,
        }

    columns = np.arange(selected.shape[1])
    left = selected[:, columns < axis_x]
    right = selected[:, columns > axis_x]
    left_area, right_area = int(left.sum()), int(right.sum())
    total = left_area + right_area

    def height(half: np.ndarray) -> int:
        rows = np.nonzero(half.any(axis=1))[0]
        return int(rows.max() - rows.min() + 1) if rows.size else 0

    return {
        "left_area": left_area,
        "right_area": right_area,
        # Signed would encode laterality, which this cohort does not record
        # (PLAN §4.1) -- so the magnitude, deliberately.
        "area_difference_fraction": (
            round(abs(left_area - right_area) / total, 5) if total else 0.0
        ),
        "height_difference_px": abs(height(left) - height(right)),
    }


def measures_about(mask: np.ndarray, axis: str) -> dict:
    """Every measure about one axis. Higher means more asymmetric."""
    selected = np.asarray(mask, dtype=bool)
    axis_x = axis_for(selected, axis)
    mirrored = mirror_about(selected, axis_x)

    overlap = overlap_scores(selected, mirrored)
    boundary = boundary_distances(selected, mirrored)
    halves = half_measures(selected, axis_x)

    area = int(selected.sum())
    # Normalised by the lip's own scale, so the index does not simply track how
    # big the crop was. sqrt(area) is the natural length for a 2-D region.
    scale = float(np.sqrt(area)) if area else 0.0

    return {
        # PRIMARY. SymNose superimposes traced boundaries, so the boundary gap is
        # the faithful quantity; 1 - dice is the area-based companion.
        "index": round(boundary["mean_px"] / scale, 6) if scale else 0.0,
        "index_definition": "mean boundary-to-mirrored-boundary distance / sqrt(area)",
        "asymmetry_1_minus_dice": round(1.0 - overlap["dice"], 5),
        "axis": axis,
        "axis_x": round(float(axis_x), 3),
        # How far the lip's own centre sits from the image midline.
        "centroid_offset_px": round(
            float(abs(axis_for(selected, "mask_centroid") - axis_for(selected, "image_midline"))),
            3,
        ),
        "area_px": area,
        **{f"overlap_{k}": v for k, v in overlap.items()},
        **{f"boundary_{k}": v for k, v in boundary.items()},
        **halves,
    }


def symnose_measures(mask: np.ndarray, axis: str = DEFAULT_AXIS) -> dict:
    """Both indices, with ``axis`` selecting which is primary.

    **The image midline is primary and the reason is anatomical, not stylistic.**
    It is consistent with §3.1, exact at G2, and it **preserves lateral
    displacement** -- which in a unilateral cleft is genuine asymmetry and often
    the largest component of it. Mirroring about the mask's own centroid scores a
    purely translated lip at exactly zero, which is precisely why it cannot be
    the primary index: it would discard the thing being measured.

    **Both are reported anyway, with ``centroid_offset_px`` beside them.** If the
    offsets are a few pixels, miscentring is a theoretical worry and the two
    indices agree. If they are large, the gap between them is the
    **lateral-displacement component** -- interesting in its own right rather
    than a nuisance, because it separates *the lip is shifted* from *the lip is
    misshapen*, and those are different clinical findings.

    **``centroid_offset_px`` is not purely a miscentring measure**, and that
    limits how far it can be read. An asymmetric lip moves its own centroid: a
    perfectly centred crop with 12px of extra vermillion on one side produces a
    6px offset entirely from the asymmetry. So a large offset means *miscentring
    or genuine asymmetry* and cannot distinguish them on its own.
    """
    primary_axis = axis
    other_axis = "mask_centroid" if axis == "image_midline" else "image_midline"

    primary = measures_about(mask, primary_axis)
    other = measures_about(mask, other_axis)

    midline = primary if primary_axis == "image_midline" else other
    centroid = primary if primary_axis == "mask_centroid" else other

    return {
        **primary,
        "index_image_midline": midline["index"],
        "index_mask_centroid": centroid["index"],
        # Signed: index about the midline minus index about the centroid. Usually
        # positive, because mirroring about the mask's own centre generally fits
        # better -- but it is not guaranteed to be, so the sign is reported
        # rather than assumed away by taking an absolute value.
        "lateral_component": round(
            midline["index"] - centroid["index"], 6
        ),
        "primary_axis": primary_axis,
    }


# --------------------------------------------------------------------------
# per-patient, from the staged image
# --------------------------------------------------------------------------

#: The feature vector handed to the regressor. Deliberately short: these are
#: interpretable clinical quantities, not a learned representation, and the arm
#: exists to show what the traced boundary alone can do.
FEATURE_NAMES = (
    "index",
    # The centroid-mirrored index and the gap between them. The pair carries a
    # decomposition either one alone loses: shape asymmetry against lateral
    # displacement. Eight features for 237 patients is still a probe.
    "index_mask_centroid",
    "lateral_component",
    "centroid_offset_px",
    "asymmetry_1_minus_dice",
    "overlap_unmatched_fraction",
    "boundary_p95_px",
    "area_difference_fraction",
    "height_difference_px",
)

#: **Say this before the number lands, not after.**
#:
#: SymNose measures **upper-lip asymmetry**. The label is an Asher-McDade
#: composite covering **nasal form, nasal symmetry, nasolabial profile and
#: vermillion border** -- four components, of which upper-lip symmetry is one and
#: contributes partially to a second. Only part of the target is the thing this
#: index measures.
#:
#: **So a modest correlation is the expected outcome, not a disappointing one.**
#: An index that tracked the whole composite would be the surprise, and would
#: warrant asking what else it had picked up. Stating the expectation in advance
#: is what stops a modest number being read as a failure of the method, or a
#: large one being accepted without asking why.
#: [MEASURED 2026-07-28] What this index scored. **A null result for a
#: non-faithful reimplementation, and nothing more.**
#:
#: PCC -0.066, Spearman -0.003. It is tempting to read that as a finding about
#: upper-lip asymmetry, and it is not one. The index carries head tilt, crop
#: centring and lateral displacement (see ``FAITHFULNESS``), any of which
#: plausibly dominates the clinical asymmetry it was meant to measure. **A zero
#: from an instrument confounded like that says nothing about the construct**,
#: and it says nothing about SymNose, which this is not.
#:
#: It is also **not evidence for the scope decision** in ``WHY_OUT``. That
#: argument stands on the manual instrument's dependence on landmark placement
#: and would stand identically had this index scored 0.4.
#:
#: The prediction made before the run -- a modest positive -- is kept beside the
#: measurement, but the pairing no longer means what it was set up to mean: the
#: prediction was about a faithful instrument's construct overlap, and this did
#: not test it.
MEASURED_RESULT = {
    "pcc_mean": -0.066,
    "pcc_sd": 0.051,
    "spearman_mean": -0.003,
    "n_seeds": 5,
    "shrinkage_range": (0.20, 0.30),
    "predicted": "a modest positive, stated before the run",
    "measured": "zero",
    "measured_on": "2026-07-28",
    "faithful": False,
    "interpretation": (
        "NULL RESULT FOR A NON-FAITHFUL REIMPLEMENTATION. Not evidence about "
        "upper-lip asymmetry, not evidence about SymNose, and not evidence for "
        "the scope decision that retired it. The index carries head tilt, crop "
        "centring and lateral displacement, any of which plausibly dominates the "
        "asymmetry it was meant to measure."
    ),
}

CONSTRUCT_NOTE = (
    "SymNose measures UPPER-LIP asymmetry. The label is an Asher-McDade "
    "composite over nasal form, nasal symmetry, nasolabial profile and "
    "vermillion border -- only part of the target is upper-lip symmetry. A "
    "MODEST CORRELATION IS THE EXPECTED OUTCOME, not a disappointing one, and "
    "this was stated before the run rather than after it. An index tracking the "
    "whole composite would be the surprising result."
)


def measures_for_image(
    image: np.ndarray,
    axis: str = DEFAULT_AXIS,
    spatial_prior: str = segmentation.DEFAULT_SPATIAL_PRIOR,
    relaxation: float = segmentation.CONTOUR_RELAXATION,
    normalisation: str = segmentation.DEFAULT_NORMALISATION,
    split_rule: str = segmentation.DEFAULT_SPLIT_RULE,
) -> dict:
    """Segment the upper lip, then measure it. One patient."""
    region = segmentation.analysis_region(image, spatial_prior)
    mask = segmentation.segment(
        image,
        segmentation.INSTRUMENT,
        region,
        relaxation=relaxation,
        normalisation=normalisation,
        split_rule=split_rule,
    )
    measures = symnose_measures(mask, axis)
    measures["segmented"] = bool(mask.any())
    return measures


def feature_matrix(
    images: np.ndarray,
    *,
    geometry: str = REQUIRED_GEOMETRY,
    axis: str = DEFAULT_AXIS,
    **kwargs,
) -> tuple[np.ndarray, list[str]]:
    """(N, D) SymNose features, plus the column names.

    Same shape the other arms return, so this reaches the harness through the
    same door and the comparison is about the index rather than the plumbing.
    """
    if geometry != REQUIRED_GEOMETRY:
        raise SymNoseError(
            f"SymNose is defined at {REQUIRED_GEOMETRY!r}, not {geometry!r}. At G1 "
            "the trapezium corners are white and the midline the mask is mirrored "
            "about is not the image midline, so the reflection would be taken "
            "about the wrong line."
        )

    names = list(FEATURE_NAMES)
    rows = np.zeros((len(images), len(names)), dtype=np.float32)
    for index, image in enumerate(images):
        measures = measures_for_image(image, axis, **kwargs)
        rows[index] = [float(measures[name]) for name in names]
    return rows, names


# --------------------------------------------------------------------------
# the audit -- does the segmentation hold at 237, or only at 9?
# --------------------------------------------------------------------------

#: A feature whose coefficient of variation is below this is near-constant.
#: **If the index barely moves across patients, the segmentation is producing
#: similar masks regardless of input**, and that is its own diagnosis: the arm
#: measured nothing because there was nothing varying to measure.
NEAR_CONSTANT_CV = 0.05


def distribution(values: np.ndarray) -> dict:
    """Quantiles and spread for one feature. No patient ids, so SHAREABLE."""
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {"n": 0}

    mean = float(array.mean())
    sd = float(array.std(ddof=1)) if array.size > 1 else 0.0
    return {
        "n": int(array.size),
        "min": round(float(array.min()), 6),
        "p05": round(float(np.percentile(array, 5)), 6),
        "q1": round(float(np.percentile(array, 25)), 6),
        "median": round(float(np.median(array)), 6),
        "q3": round(float(np.percentile(array, 75)), 6),
        "p95": round(float(np.percentile(array, 95)), 6),
        "max": round(float(array.max()), 6),
        "mean": round(mean, 6),
        "sd": round(sd, 6),
        # sd / |mean|. Scale-free, so it is comparable across features that live
        # on different ranges.
        "cv": round(sd / abs(mean), 4) if abs(mean) > 1e-12 else None,
        "n_distinct": int(np.unique(np.round(array, 6)).size),
        "near_constant": bool(abs(mean) > 1e-12 and sd / abs(mean) < NEAR_CONSTANT_CV),
    }


#: Quartiles of the cohort's own brightness distribution, darkest first.
BRIGHTNESS_QUARTILES = ("q1_darkest", "q2", "q3", "q4_lightest")


def pass_rate_by_brightness(per_patient: list[dict]) -> dict:
    """Segmentation pass rate split by the skin-tone proxy. **n=237, not n=1.**

    **This is the point of running the audit.** "Unsupervised colour segmentation
    is implicitly calibrated on lighter skin" currently rests on a single patient
    out of nine -- a real observation with a mechanism behind it, and an
    indefensible sample. The per-patient records already exist across the whole
    cohort; splitting the pass rate by brightness quartile turns the strongest
    ethical finding in the project into a cohort-scale measurement.

    **What a positive result looks like:** a monotone gradient, with the darkest
    quartile failing more than the lightest. The per-criterion breakdown says
    *how* it fails -- patient 5's signature was over-selection, so an excess of
    ``area_max`` in the darkest quartile is the specific pattern to look for.

    **What it is not.** ``median_value_in_region`` is a **proxy** for skin tone
    and it also carries lighting, exposure and how much of the frame the face
    fills. A gradient is evidence of a **brightness-linked failure pattern**,
    which is what the mechanism predicts; calling it a skin-tone gradient
    requires the ethnicity data this cohort does not record. Say the first, not
    the second.

    **A flat result is also a finding** and must be reported as one: it would
    mean the n=1 observation did not generalise, and the fairness claim would
    have to be narrowed to that patient rather than quietly kept.
    """
    values = [r.get("median_value_in_region", 0.0) for r in per_patient]
    if len(values) < 4:
        return {"n_patients": len(values), "quartiles": {}, "insufficient": True}

    array = np.asarray(values, dtype=float)
    cuts = np.percentile(array, [25, 50, 75])
    # Right-open bins, so the lightest quartile keeps the maximum.
    index_of = np.searchsorted(cuts, array, side="right")

    quartiles = {}
    for position, name in enumerate(BRIGHTNESS_QUARTILES):
        members = [r for r, q in zip(per_patient, index_of) if q == position]
        if not members:
            quartiles[name] = {"n": 0}
            continue
        brightness = [r.get("median_value_in_region", 0.0) for r in members]
        n_pass = sum(1 for r in members if r["passes"])
        quartiles[name] = {
            "n": len(members),
            "n_pass": n_pass,
            "pass_rate": round(n_pass / len(members), 4),
            "brightness_min": round(min(brightness), 4),
            "brightness_max": round(max(brightness), 4),
            # How it fails, not just that it does. Over-selection on darker skin
            # was patient 5's signature, so area_max is the one to watch.
            "failures_by_criterion": {
                name_: sum(1 for r in members if name_ in r["failed_criteria"])
                for name_ in segmentation.CRITERION_NAMES
            },
        }

    rates = [
        quartiles[name]["pass_rate"]
        for name in BRIGHTNESS_QUARTILES
        if quartiles[name].get("n")
    ]
    return {
        "n_patients": len(values),
        "proxy": "median_value_in_region (lower is darker)",
        "quartiles": quartiles,
        "darkest_minus_lightest": (
            round(rates[0] - rates[-1], 4) if len(rates) >= 2 else None
        ),
        "monotone_increasing": bool(rates == sorted(rates)) if rates else None,
        "note": (
            "THE FAIRNESS FINDING AT COHORT SCALE. It previously rested on one "
            "patient in nine. A monotone gradient with the darkest quartile "
            "failing most is what the mechanism predicts -- every method here is "
            "intensity- or redness-based, so all are implicitly calibrated on "
            "lighter skin, and on darker skin the lip-versus-skin contrast "
            "compresses in both hue and value. "
            "PROXY, NOT SKIN TONE: median_value_in_region also carries lighting, "
            "exposure and how much of the frame the face fills. A gradient is "
            "evidence of a BRIGHTNESS-LINKED FAILURE PATTERN; calling it a "
            "skin-tone gradient needs ethnicity data this cohort does not record. "
            "A FLAT RESULT IS ALSO A FINDING and must be reported as one -- it "
            "would mean the n=1 observation did not generalise and the claim has "
            "to be narrowed rather than quietly kept."
        ),
    }


def audit(
    images: np.ndarray,
    *,
    spatial_prior: str = segmentation.DEFAULT_SPATIAL_PRIOR,
    relaxation: float = segmentation.CONTOUR_RELAXATION,
    normalisation: str = segmentation.DEFAULT_NORMALISATION,
    split_rule: str = segmentation.DEFAULT_SPLIT_RULE,
) -> tuple[dict, list[dict]]:
    """Run the screen and the index over the whole cohort. Diagnostic only.

    **Why this has to exist before the SymNose result can be interpreted.** The
    gate validated the segmentation on **9 patients -- 3.8% of the cohort**. If it
    fails at scale, the index is noise for the patients it fails on, and a
    correlation of zero says nothing whatever about the construct. Two very
    different findings produce the same number:

    * the segmentation holds and upper-lip symmetry genuinely does not predict an
      Asher-McDade composite;
    * the segmentation collapses on most of the cohort and the arm measured
      nothing at all.

    **Nothing here is tuned.** No relaxation sweep, no split change, no feature
    selection. This reports what the existing configuration does at 237, so the
    zero can be attributed before anything is adjusted -- adjusting first would
    make the attribution impossible.

    Returns ``(aggregate, per_patient)``. The aggregate is SHAREABLE; the
    per-patient list is patient-keyed and belongs in a CLUSTER-ONLY file.
    """
    per_patient: list[dict] = []

    for index, image in enumerate(images):
        region = segmentation.analysis_region(image, spatial_prior)
        mask = segmentation.segment(
            image,
            segmentation.INSTRUMENT,
            region,
            relaxation=relaxation,
            normalisation=normalisation,
            split_rule=split_rule,
        )
        unconstrained = segmentation.chan_vese(
            image, region, relaxation=relaxation, normalisation=normalisation
        )
        division = segmentation.split_row(image, unconstrained, split_rule)
        described = segmentation.describe_mask(mask, region, division)
        failures = segmentation.failed_criteria(described)
        measures = symnose_measures(mask)

        # The skin-tone proxy, per patient. This is what turns the fairness
        # finding from n=1 into n=237.
        selected = np.asarray(region, dtype=bool)
        brightness = (
            float(np.median(segmentation.to_hsv(image)[..., 2][selected]))
            if selected.any()
            else 0.0
        )

        per_patient.append(
            {
                "row": index,
                "passes": not failures,
                "failed_criteria": failures,
                "median_value_in_region": round(brightness, 4),
                "area": described["area"],
                "area_within_region": described["area_within_region"],
                "centroid_x": described["centroid_x"],
                "centroid_y": described["centroid_y"],
                "largest_component": described["largest_component"],
                "fraction_above_split": described["fraction_above_split"],
                "empty_mask": not bool(mask.any()),
                **{name: measures[name] for name in FEATURE_NAMES},
            }
        )

    n = len(per_patient)
    n_pass = sum(1 for r in per_patient if r["passes"])

    by_criterion = {
        name: sum(1 for r in per_patient if name in r["failed_criteria"])
        for name in segmentation.CRITERION_NAMES
    }
    # How many failed on this criterion ALONE. A criterion that is always the
    # sole cause is the one to look at; one that only ever co-occurs is a symptom.
    only_criterion = {
        name: sum(1 for r in per_patient if r["failed_criteria"] == [name])
        for name in segmentation.CRITERION_NAMES
    }

    features = {
        name: distribution(np.array([r[name] for r in per_patient]))
        for name in FEATURE_NAMES
    }
    fairness = pass_rate_by_brightness(per_patient)
    near_constant = sorted(
        name for name, d in features.items() if d.get("near_constant")
    )

    aggregate = {
        "n_patients": n,
        "n_pass": n_pass,
        "pass_rate": round(n_pass / n, 4) if n else 0.0,
        "n_empty_masks": sum(1 for r in per_patient if r["empty_mask"]),
        "failures_by_criterion": by_criterion,
        "failures_by_sole_criterion": only_criterion,
        "n_failing_multiple": sum(
            1 for r in per_patient if len(r["failed_criteria"]) > 1
        ),
        "feature_distributions": features,
        "near_constant_features": near_constant,
        # THE REASON THIS RUNS. See pass_rate_by_brightness.
        "fairness_by_brightness": fairness,
        "near_constant_threshold_cv": NEAR_CONSTANT_CV,
        "gate_sample_fraction": round(9 / n, 4) if n else None,
        "note": (
            "DIAGNOSTIC ONLY -- nothing is tuned here. The gate validated the "
            "segmentation on 9 patients, which is gate_sample_fraction of the "
            "cohort. This says whether it holds at scale. "
            "A LOW pass_rate means the index is noise for the failing patients "
            "and a correlation of zero says nothing about the construct. A HIGH "
            "pass_rate means the segmentation held and the zero is about the "
            "construct. "
            "near_constant_features is the other diagnosis: if the index barely "
            "moves across patients then the segmentation is producing similar "
            "masks whatever the input, and the arm measured nothing because "
            "nothing varied. "
            "failures_by_sole_criterion separates a cause from a symptom -- a "
            "criterion that is only ever one of several is not the reason."
        ),
    }
    return aggregate, per_patient


def describe(axis: str = DEFAULT_AXIS) -> dict:
    """What this arm computed. SHAREABLE -- definitions and counts only."""
    return {
        "arm": "mirrored_upper_lip_index",
        "retired": "2026-07-28",
        "is_symnose": False,
        "faithfulness": dict(FAITHFULNESS),
        "why_out": dict(WHY_OUT),
        "geometry": REQUIRED_GEOMETRY,
        "axis": axis,
        "n_features": len(FEATURE_NAMES),
        "feature_names": list(FEATURE_NAMES),
        "primary_index": "mean boundary-to-mirrored-boundary distance / sqrt(area)",
        "primary_axis": "image_midline",
        "axis_note": (
            "The image midline is primary because it PRESERVES LATERAL "
            "DISPLACEMENT, which in a unilateral cleft is genuine asymmetry and "
            "often its largest component. Mirroring about the mask's own centroid "
            "scores a purely translated lip at exactly zero, which is why it "
            "cannot be primary. Both are reported with centroid_offset_px: small "
            "offsets mean the two indices agree and miscentring is a theoretical "
            "worry; large ones mean lateral_component is the displacement part of "
            "the asymmetry, which is a finding rather than a nuisance. "
            "CAVEAT: centroid_offset_px is NOT purely a miscentring measure. A "
            "lip with more vermillion on one side moves its own centroid, so a "
            "large offset can mean a miscentred crop OR genuine asymmetry, and "
            "the offset alone CANNOT SEPARATE them. Read it against the sheet."
        ),
        "segmentation_instrument": segmentation.INSTRUMENT,
        "spearman_note": SPEARMAN_IS_COMPARABLE,
        "construct_note": CONSTRUCT_NOTE,
        "note": (
            "RETIRED. This is a mirrored upper-lip index, NOT a SymNose "
            "reimplementation -- see faithfulness. SymNose is out on the scope "
            "boundary in why_out, which is independent of anything measured here. "
            "Beyond the missing levelling and registration, the index also "
            "inherits every limitation of the segmentation feeding it: the oral "
            "fissure is unrecoverable in 7 of 9 sampled patients so the "
            "upper/lower division is geometric rather than anatomical; the split "
            "always fires and halves a mask whose crop excludes the lower lip (1 "
            "of 9); and the discriminant is implicitly calibrated on lighter "
            "skin. Those are properties of THIS index, not caveats about the "
            "segmentation alone."
        ),
    }
