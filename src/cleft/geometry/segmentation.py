"""Phase 4 §2 — the segmentation feasibility gate. A diagnostic, not a pipeline.

**PARKED 2026-07-28. Not deleted, and the tests stay green.**

Its only consumer was the mirrored upper-lip index (``geometry.symnose``), which
was retired when SymNose went out on a scope boundary. Nothing in the current arm
list segments anything.

**It is kept because it is the evidence behind a write-up chapter**, and deleting
it would leave that chapter describing work with no artefact. Five gate runs
produced four results that stand on their own regardless of what consumed them:

* **§2's feasibility answer** — lips *can* be segmented from compressed JPEG
  crops, against a protocol that said they could not without raw camera files;
* **the fissure floor** — in 7 of 9 patients the oral fissure is unrecoverable,
  with the depth distribution and threshold sweep behind it;
* **the geometric split**, validated against the two detectable fissures at max
  4px, 3.1% of mask height;
* **the fairness finding** — every method here is intensity- or redness-based and
  so implicitly calibrated on lighter skin.

**The tests keep running so the code does not rot.** A parked module whose suite
is skipped is a module that quietly stops working, and then the chapter's
artefact cannot be re-derived when someone asks.

**Do not extend it.** If a future phase needs segmentation, that is a decision to
take deliberately, with a consumer named first.

---

**[LITERATURE] The protocol says this may be impossible on this data:**

    "The Computer subtraction imaging technology to extract just the lip from the
    surrounding skin is not possible on compressed low resolution JPEG type files
    and needs either the original DNG raw camera negative file or its most
    uncompressed server stored equivalent in TIFF."

Those originals do not exist. So before any segmentation pipeline is built, this
answers whether lips can be separated from skin **on these crops at all**:

* **if yes** — the SymNose reimplementation (§3.2) proceeds;
* **if no** — SymNose is out, recorded in limitations with the protocol as the
  citation, and the mirror-difference index (§3.1) carries the symmetry question
  instead.

This is the cheap-diagnostic-first pattern that found the AppleDouble sidecars and
the band boundaries: spend an hour finding out, rather than a week building on an
assumption.

**Deliberately not a pipeline.** Three standard methods, run once, described with
numbers and rendered for a human to look at. Nothing here is imported by an arm,
nothing is tuned, and nothing survives the gate: if the answer is yes, SymNose
gets a real implementation, and this module's job was to say so.

**The three methods** (the ones §2 names), all on the *staged* crop:

1. ``colour_separation`` — lips differ from skin in hue and saturation. A
   discriminant channel plus a fixed cut.
2. ``otsu_threshold`` — the same discriminant, with the cut chosen by Otsu
   instead of fixed, so a bad fixed threshold cannot be mistaken for a bad
   discriminant.
3. ``chan_vese`` — a region-based active contour. Region-based rather than
   edge-based **on purpose**: the protocol's complaint is that the lip-skin
   boundary is not resolvable in compressed JPEG, and an edge-seeking snake
   assumes the edge it is looking for. Chan-Vese asks whether two regions exist,
   which is the question actually being posed.

scipy only -- no scikit-image, no opencv. Both are heavier than a gate needs, and
opencv is not in the test extra, so this stays laptop-testable.

**On the verdict.** The plausibility criteria below are declared from anatomy
*before* any of them is run, and they are not adjusted afterwards. Loosening a
criterion until a method passes would turn a gate into a formality -- and this is
a gate whose "no" is a perfectly good answer with a citation behind it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

METHODS = (
    "colour_separation",
    "otsu_threshold",
    "chan_vese",
    "chan_vese_relaxed",
    "upper_lip",
)

#: [DECIDED 2026-07-28, run 2] **SymNose is IN, and it is built on Chan-Vese.**
#:
#: Otsu and Chan-Vese trace the lips near-perfectly on nearly every patient and
#: find the same region -- Otsu is kept as a comparator precisely because they
#: agree, and agreement between an independent method is worth more than either
#: one alone. But the instrument is the contour, for a reason specific to what
#: SymNose does:
#:
#: **SymNose mirrors a traced boundary and superimposes it, so boundary noise
#: enters the asymmetry measurement directly as spurious asymmetry.** Otsu
#: thresholds per pixel, so its boundary is pixelated; Chan-Vese is a boundary
#: method with a curvature term, so its boundary is smooth. A jagged edge would
#: manufacture exactly the signal being measured -- the noise would not average
#: out, it would read as asymmetry.
#:
#: ``colour_separation`` stays OUT: unchanged from run 1 apart from being
#: confined to the band. It is kept in the table as the record of what a fixed
#: cut does, not as a candidate.
#: [DECIDED 2026-07-28, run 3] The instrument is the relaxed contour **with the
#: upper-lip constraint applied**. See ``fissure_row``.
INSTRUMENT = "upper_lip"
COMPARATOR = "otsu_threshold"

#: The contour the instrument is derived from. Reported alongside, because once
#: the constraint is applied ``fraction_above_fissure`` is 1.0 by construction
#: and the number that carries information is the unconstrained one.
UNCONSTRAINED = "chan_vese_relaxed"

#: **G1, and unlike the mirror index this is deliberate in the other direction.**
#: The question here is whether the lip-skin distinction survives JPEG
#: compression in the pixels as they exist. G2 resamples every row to full width,
#: which smooths exactly the local colour structure being tested -- it could only
#: flatter the methods. G1 is unmodified staging, so a "no" at G1 is a real no.
DEFAULT_GEOMETRY = "g1"


class SegmentationError(RuntimeError):
    """The feasibility diagnostic could not be run."""


def load_staged(folders_root, row: dict, geometry: str = DEFAULT_GEOMETRY):
    """One patient's frontal image, staged, ready for the methods.

    Lives here rather than in ``contact`` so the Phase 2 module is not touched
    for a Phase 4 diagnostic.
    """
    from . import contact, render
    from .staging import Staged, stage
    from .trapezium import unwarp

    path = contact.find_image(
        folders_root, int(row["patient_id"]), int(row["frontal_id"])
    )
    staged = stage(render.load_image(path))
    if geometry == "g1":
        return staged.image
    if geometry == "g2":
        return unwarp(staged.image)
    raise SegmentationError(
        f"unknown geometry {geometry!r}; expected 'g1' or 'g2'"
    )


# --------------------------------------------------------------------------
# plausibility criteria -- declared before the methods are run
# --------------------------------------------------------------------------

#: What a plausible lip segmentation looks like on a nasolabial crop, from
#: anatomy. Every bound is a shape or position fact, not a tuned number.
#:
#: ``area_min`` -- **against the frame.**
#:     Below this the method found a speckle. A lip is a structure of a certain
#:     physical size on the crop, and that does not change because the search was
#:     narrowed, so this bound does not scale with the search area.
#: ``area_max`` -- **against the search region.** See ``CRITERIA_PROVENANCE``.
#:     Above this the method took essentially everything it was allowed to look
#:     at, which is not a segmentation.
#: ``centroid_y``
#:     Lips sit low. The bottom band starts at 2/3; a centroid above the middle
#:     of the frame means the method found the eyes or the nose instead, which is
#:     the failure mode that looks most convincing on a thumbnail. **Note this
#:     bound is satisfied by construction once a bottom-band prior is used**, so
#:     under a prior it stops being evidence.
#: ``centroid_x``
#:     Roughly on the midline. A cleft lip is asymmetric, so this is loose --
#:     but a centroid at 0.15 is a cheek, not a lip.
#: ``largest_component``
#:     A lip region is one connected blob. Speckle noise that happens to cover
#:     the right area would pass every other check.
PLAUSIBLE = {
    "area_min": 0.02,
    "area_max": 0.35,
    "centroid_y_min": 0.50,
    "centroid_x_min": 0.30,
    "centroid_x_max": 0.70,
    "largest_component_min": 0.60,
    #: [PRE-DECLARED 2026-07-28, before run 3.] The mask must lie predominantly
    #: above the division between the lips -- it is an UPPER-lip instrument.
    #:
    #: Expected to FAIL the run-2 relaxed output on 9 of 9 and the run-2
    #: unrelaxed output on 3 of 9. That is the check working, and it is asserted
    #: as a regression rather than hoped for: a new criterion that passes
    #: everything on arrival has not been shown to test anything.
    #:
    #: **The value is unchanged; the LINE it is measured against changed** at run
    #: 5, from the detected oral fissure to the geometric split, because no
    #: viable detection threshold exists. The invariant -- *the mask lies
    #: predominantly above the division between the lips* -- did not change; its
    #: instantiation did, exactly as with ``area_max``. Renamed from
    #: ``fraction_above_fissure_min`` so the reported field and the criterion say
    #: the same thing. See ``CRITERIA_PROVENANCE``.
    #:
    #: **Satisfied by construction for the constrained instrument**, exactly as
    #: centroid_y is under a bottom-band prior. The number that carries
    #: information is `fraction_above_split_unconstrained`, reported beside it.
    "fraction_above_split_min": 0.80,
}

#: Otsu separability below this means the discriminant is not bimodal **within
#: the face**, and no choice of threshold on it can work.
#:
#: **[MEASURED 2026-07-28] This number was measuring the wrong contrast in run 1,
#: and that is a correction to this module's design, not to the data.** Computed
#: over the whole frame it came out at 0.888, which reads as "strongly bimodal,
#: the protocol's compression worry is unfounded". It was nothing of the kind: the
#: staged crop is padded with white, so the dominant contrast in the frame is
#: **face against background**, and 0.888 measured that. Otsu and Chan-Vese
#: returned whole-face masks on all nine patients for the same reason -- they
#: found the strongest split available, and it was the pad.
#:
#: So run 1's separability says nothing about lip-versus-skin in either
#: direction, and cannot be cited for or against the protocol. It is now computed
#: **inside the face mask**, which is the contrast that was always meant.
WEAK_SEPARABILITY = 0.30

#: What counts as background rather than face. ``staging`` pads with 255, so the
#: padding is exactly white by construction; the bounds are loosened from that to
#: tolerate JPEG ringing at the pad edge. Derived from how staging works, not
#: tuned against any outcome.
BACKGROUND_VALUE_MIN = 0.90
BACKGROUND_SATURATION_MAX = 0.10

#: Where a lip may be looked for. ``"none"`` reproduces run 1; a band name
#: restricts the search to that band of ``PatchConfig.bands``.
#:
#: **[MEASURED 2026-07-28] why a prior is needed.** Colour separation picked lips
#: *and nose* together on the patients where it worked. That is the discriminant
#: behaving correctly -- both are red and saturated in a nasolabial crop -- so no
#: amount of thresholding separates them. Position does. The bottom band is
#: already defined in Phase 2 and was confirmed on the Phase 2 contact sheet to
#: contain the commissures, so this adds a constraint rather than a constant.
SPATIAL_PRIORS = ("none", "top", "middle", "bottom")
DEFAULT_SPATIAL_PRIOR = "bottom"

#: [MEASURED 2026-07-28, run 2] How far to move the contour outward, as a
#: fraction of the half-gap between the two region means.
#:
#: **Why any relaxation.** The run-2 masks sit just *inside* the vermillion
#: border and want a little more on the upper lip. That is not a defect in the
#: method: Chan-Vese converges where the two region means are equidistant, which
#: is where between-class separation is maximised, and that is not the anatomical
#: boundary. The anatomical edge is slightly outside it.
#:
#: **Why expressed against the class gap rather than in pixels.** A fixed pixel
#: dilation would be a different relaxation on every patient, because contrast
#: and crop scale vary. A fraction of ``(inner_mean - outer_mean)/2`` is ONE
#: rule with ONE parameter applied identically to everyone, and it lands in the
#: same place relative to each image's own separation. That is what makes it a
#: fixed relaxation rather than a per-patient tune.
#:
#: **Why it is small, and what to watch.** SymNose measures the UPPER lip. Over-
#: relaxing pulls the mask down across the oral fissure into the lower lip and
#: silently changes what is being measured -- the arm would keep running and keep
#: producing numbers. ``chan_vese`` (unrelaxed) is rendered beside the relaxed
#: mask on every sheet and ``relaxation_growth`` reports the area increase per
#: patient, so the magnitude is judged on the contact sheet rather than assumed
#: here. **Check the growth figures and the sheet before trusting this value.**
CONTOUR_RELAXATION = 0.15

#: Growth beyond this is flagged. Not a verdict -- a prompt to look at that
#: patient on the sheet, because a mask that grew by half has probably crossed
#: into the lower lip.
RELAXATION_GROWTH_LIMIT = 1.5

#: A mask covering essentially all of the region it was allowed to search has not
#: segmented anything. Kept as a separate flag even though ``area_max`` now
#: catches the same case, because the two can disagree at the margin and a mask
#: at 0.88 of its region is worth seeing whatever the criteria say.
SATURATION_LIMIT = 0.90

#: **How the area criterion came to be read against two different denominators.**
#: Recorded because it is a POST-HOC reading and the record should say so.
#:
#: Nobody wrote "relative to the search region" before run 1. What was declared
#: was a set of shape and position facts -- *a plausible lip is a modest fraction
#: of what was searched, sits low, near the midline, and is one blob* -- and
#: under a whole-frame search ``area <= 0.35 of frame`` expressed that faithfully.
#: The number was only ever the **instantiation** of the invariant for the search
#: area in use.
#:
#: Adding the spatial prior changed the search area and broke the instantiation:
#: a method selecting the entire bottom band scored 0.335 against a ceiling of
#: 0.35 and passed everything. Keeping 0.35-of-frame would have kept the number
#: and abandoned the intent.
#:
#: **[MEASURED 2026-07-28] Checking the other end showed one denominator is not
#: enough.** With ``area_min`` also read against the region, a speckle of 0.4% of
#: the frame scores 0.023 against a floor of 0.02 and passes -- and centroid_y,
#: centroid_x and largest_component all pass a single small midline blob too, so
#: nothing catches it. Meanwhile frame-relative ``area_max`` cannot catch
#: band-filling either: the face-masked bottom band is only 0.168 of the frame,
#: comfortably under 0.35.
#:
#: So the two ends carry different invariants and take different denominators:
#:
#: * ``area_max`` -- "a modest fraction of **what was searched**". Scales with the
#:   search area. **Region-relative.**
#: * ``area_min`` -- "not a speckle; a lip is a real structure". Does **not**
#:   scale with the search area. **Frame-relative.**
#:
#: Both quantities are reported on every mask whichever bound applies, so run 1
#: and run 2 stay directly comparable on the original frame-relative number.
CRITERIA_PROVENANCE = {
    "pre_declared": "2026-07-28, before run 1: area against the frame, both ends",
    "amended": "2026-07-28, after run 1: area_max against the search region",
    "amendment_is_post_hoc": True,
    "invariant": (
        "a plausible lip is a modest fraction of what was searched, sits low, "
        "near the midline, and is one connected blob"
    ),
    "why": (
        "The invariant did not change. Its expression did, because the spatial "
        "prior changed the search area and 0.35-of-frame was only the "
        "instantiation of 'modest fraction of what was searched' when the frame "
        "WAS the search area. area_min stays frame-relative because 'not a "
        "speckle' does not scale with the search area -- measured: region-relative "
        "area_min admits a 0.4%-of-frame blob that nothing else rejects."
    ),
    "numbers_unchanged": True,
    #: Added before the run that tests it, not after -- and expected to FAIL on
    #: arrival, which is what distinguishes a criterion from a formality.
    "added": (
        "2026-07-28, before run 3: fraction_above_fissure_min, because the "
        "contour crosses onto the lower lip and SymNose measures the upper one"
    ),
    "added_criterion_expected_to_fail": (
        "run-2 relaxed output on 9 of 9, run-2 unrelaxed on 3 of 9"
    ),
    #: [MEASURED run 5] `upper_lip` mean area is 0.0230 against an area_min of
    #: 0.02. The floor was calibrated when the instrument segmented BOTH lips, so
    #: a correct upper-lip mask sits on it and about half fall through.
    #:
    #: **It is deliberately not corrected.** area_max was re-expressed after run
    #: 1, fraction_above_fissure added before run 3 and renamed at run 5, and now
    #: area_min is mis-calibrated: three amendments across six runs is the shape
    #: of a criterion set being fitted to the data. A fourth would tighten the fit
    #: and add no meaning.
    #:
    #: What it means is that **the screen never decided anything** -- human review
    #: of the contact sheet was the operative decision procedure at every gate
    #: run, including run 5 where screen and sheet disagreed outright. That is
    #: recorded as stated rather than repaired.
    "known_miscalibrated": (
        "area_min (0.02) against an upper-lip mean area of 0.0230. Calibrated for "
        "a both-lips target and left uncorrected on purpose -- see the write-up "
        "note on the criterion set."
    ),
    "renamed": (
        "2026-07-28, run 5: fraction_above_fissure_min -> "
        "fraction_above_split_min. VALUE UNCHANGED at 0.80; the LINE it is "
        "measured against changed from the detected oral fissure to the "
        "geometric split, because the run-4 sweep showed no viable detection "
        "threshold exists. The invariant -- the mask lies predominantly above "
        "the division between the lips -- did not change; its instantiation did, "
        "exactly as with area_max. Renamed so the field and the criterion say "
        "the same thing."
    ),
}


# --------------------------------------------------------------------------
# colour
# --------------------------------------------------------------------------


def to_hsv(image: np.ndarray) -> np.ndarray:
    """RGB uint8 to HSV floats on [0, 1]. Vectorised, no dependencies.

    Written out rather than imported so the diagnostic runs in the test extra,
    which has neither opencv nor scikit-image.
    """
    array = np.asarray(image, dtype=np.float32)
    if array.ndim != 3 or array.shape[2] != 3:
        raise SegmentationError(f"expected an RGB image, got shape {array.shape}")
    array = array / 255.0 if array.max() > 1.0 else array

    red, green, blue = array[..., 0], array[..., 1], array[..., 2]
    high = array.max(axis=2)
    low = array.min(axis=2)
    span = high - low

    hue = np.zeros_like(high)
    safe = span > 1e-12
    with np.errstate(invalid="ignore", divide="ignore"):
        red_max = safe & (high == red)
        green_max = safe & (high == green) & ~red_max
        blue_max = safe & (high == blue) & ~red_max & ~green_max
        hue[red_max] = ((green - blue)[red_max] / span[red_max]) % 6.0
        hue[green_max] = (blue - red)[green_max] / span[green_max] + 2.0
        hue[blue_max] = (red - green)[blue_max] / span[blue_max] + 4.0
    hue = hue / 6.0

    saturation = np.zeros_like(high)
    saturation[high > 1e-12] = span[high > 1e-12] / high[high > 1e-12]
    return np.stack([hue, saturation, high], axis=2)


#: Per-image normalisation of the discriminant, applied inside the analysis
#: region before any thresholding.
#:
#: **[MEASURED 2026-07-28, run 3] why this exists.** Patient 5 is of Indian
#: ethnicity; the other eight in the sample are white British. Every method here
#: is intensity- or redness-based, so all are **implicitly calibrated on lighter
#: skin**: on darker skin the lip-versus-skin contrast compresses in both hue and
#: value. The signature is unmistakable -- median saturation 0.824 against
#: 0.331-0.492 for the others, and all four methods taking 44-99% of the band,
#: with the mask escaping onto the chin and cheeks well outside the vermillion
#: lines.
#:
#: ``rank`` replaces each value by its percentile *within the region*, which
#: forces a full-range spread regardless of how compressed the original was.
#: ``zscore`` standardises instead, which preserves shape but not range.
#:
#: **This is a hypothesis, not a fix.** It is untested on the real crops, and it
#: must be evaluated as a rule applying identically to everyone -- never adjusted
#: until patient 5 works. A parameter fitted to one image would hide the fairness
#: problem rather than solve it, and would leave the next darker-skinned patient
#: exactly where this one is.
DISCRIMINANT_NORMALISATIONS = ("none", "rank", "zscore")
DEFAULT_NORMALISATION = "none"


def normalise_within(
    values: np.ndarray, region: np.ndarray, method: str = DEFAULT_NORMALISATION
) -> np.ndarray:
    """Renormalise a discriminant using only the pixels inside the region."""
    if method == "none":
        return values
    if method not in DISCRIMINANT_NORMALISATIONS:
        raise SegmentationError(
            f"unknown normalisation {method!r}; expected one of "
            f"{DISCRIMINANT_NORMALISATIONS}"
        )

    selected = np.asarray(region, dtype=bool)
    if selected.sum() < 2:
        return values

    inside = values[selected]
    out = np.zeros_like(values)

    if method == "zscore":
        spread = float(inside.std())
        if spread <= 1e-12:
            return values
        out[selected] = (inside - float(inside.mean())) / spread
        return out

    # rank: percentile within the region, so the spread is full-range whatever
    # the original compression was.
    order = np.argsort(inside, kind="stable")
    ranks = np.empty(inside.size, dtype=np.float64)
    ranks[order] = np.arange(inside.size, dtype=np.float64)
    out[selected] = ranks / max(inside.size - 1, 1)
    return out


def discriminant_for(
    image: np.ndarray,
    region: np.ndarray | None = None,
    normalisation: str = DEFAULT_NORMALISATION,
) -> np.ndarray:
    """The discriminant every method thresholds, normalised if asked."""
    values = lip_discriminant(image)
    if normalisation == "none" or region is None:
        return values
    return normalise_within(values, region, normalisation)


def lip_discriminant(image: np.ndarray) -> np.ndarray:
    """One channel on [0, 1] where lips should score high and skin low.

    Lips are redder and more saturated than the skin around them. "Redder" has to
    be expressed carefully: hue is circular and skin sits near the same red-orange
    arc, so the useful quantity is **how close the hue is to pure red**, weighted
    by saturation. Multiplying rather than adding means a strongly red but pale
    pixel and a saturated but orange pixel both score low, which is what
    distinguishes vermillion from cheek.
    """
    hsv = to_hsv(image)
    hue, saturation = hsv[..., 0], hsv[..., 1]
    # Circular distance to red (hue 0), normalised so 0.5 of the wheel away = 0.
    redness = 1.0 - 2.0 * np.minimum(hue, 1.0 - hue)
    return np.clip(redness * saturation, 0.0, 1.0)


# --------------------------------------------------------------------------
# the three methods
# --------------------------------------------------------------------------


def otsu(values: np.ndarray, bins: int = 256) -> tuple[float, float]:
    """Otsu's threshold and its separability ``eta``, on flattened values.

    ``eta`` is between-class variance over total variance: **1 means two clean
    modes, 0 means one blob**. It is the number that answers the protocol's
    question directly, because it measures whether two classes exist at all
    rather than how well a particular method found them.
    """
    flat = np.asarray(values, dtype=np.float64).ravel()
    if flat.size == 0:
        raise SegmentationError("cannot threshold an empty array")

    total_variance = float(flat.var())
    if total_variance <= 1e-12:
        return float(flat[0]), 0.0

    counts, edges = np.histogram(flat, bins=bins, range=(flat.min(), flat.max()))
    centres = (edges[:-1] + edges[1:]) / 2.0
    weights = counts / counts.sum()

    cumulative = np.cumsum(weights)
    means = np.cumsum(weights * centres)
    grand_mean = means[-1]

    # Between-class variance for every candidate split, in one pass.
    with np.errstate(invalid="ignore", divide="ignore"):
        between = (grand_mean * cumulative - means) ** 2 / (
            cumulative * (1.0 - cumulative)
        )
    between = np.nan_to_num(between, nan=0.0, posinf=0.0, neginf=0.0)

    index = int(np.argmax(between))
    return float(centres[index]), float(between[index] / total_variance)


# --------------------------------------------------------------------------
# where a lip may be looked for
# --------------------------------------------------------------------------


def face_mask(image: np.ndarray) -> np.ndarray:
    """Face pixels, excluding the white padding staging writes.

    **The fix for run 1's central failure.** Every method searched the full frame,
    where the strongest available split is face against white pad -- so that is
    what they found, on all nine patients. Restricting the search to the face
    makes the next-strongest split the one being asked about.

    Filled and reduced to the largest component, because the interior of a face
    contains genuinely unsaturated pixels (a specular highlight on the nose tip)
    that are not background and would otherwise punch holes in the region.
    """
    from scipy.ndimage import binary_fill_holes, label

    hsv = to_hsv(image)
    background = (hsv[..., 2] >= BACKGROUND_VALUE_MIN) & (
        hsv[..., 1] <= BACKGROUND_SATURATION_MAX
    )
    face = ~background
    if not face.any():
        # Nothing but padding. Return it as-is rather than inventing a region:
        # the descriptors will report an empty search area, which is the truth.
        return face

    face = binary_fill_holes(face)
    labelled, count = label(face)
    if count > 1:
        sizes = np.bincount(labelled.ravel())[1:]
        face = labelled == (int(np.argmax(sizes)) + 1)
    return np.asarray(face, dtype=bool)


def prior_mask(shape, spatial_prior: str = DEFAULT_SPATIAL_PRIOR, config=None) -> np.ndarray:
    """The band a lip is allowed to be found in, as a boolean mask.

    Bands come from ``PatchConfig``, so this cannot drift away from the patch
    scheme -- moving a band boundary moves both instruments together.
    """
    from .patches import PatchConfig

    height, width = shape[:2]
    if spatial_prior == "none":
        return np.ones((height, width), dtype=bool)

    bands = {band.name: band for band in (config or PatchConfig()).bands}
    if spatial_prior not in bands:
        raise SegmentationError(
            f"unknown spatial_prior {spatial_prior!r}; expected 'none' or one of "
            f"{sorted(bands)}"
        )

    band = bands[spatial_prior]
    mask = np.zeros((height, width), dtype=bool)
    mask[int(band.y0 * height) : int(round(band.y1 * height)), :] = True
    return mask


def analysis_region(
    image: np.ndarray,
    spatial_prior: str = DEFAULT_SPATIAL_PRIOR,
    config=None,
    use_face_mask: bool = True,
) -> np.ndarray:
    """Face, intersected with the spatial prior. Where the methods may look.

    ``use_face_mask=False`` with ``spatial_prior="none"`` reproduces run 1
    exactly -- the whole frame, padding included. Kept so the run that produced
    the whole-face masks can be re-derived rather than only described.
    """
    region = prior_mask(image.shape, spatial_prior, config)
    return (face_mask(image) & region) if use_face_mask else region


def _region_or_all(image: np.ndarray, region: np.ndarray | None) -> np.ndarray:
    if region is None:
        return np.ones(image.shape[:2], dtype=bool)
    return np.asarray(region, dtype=bool)


# --------------------------------------------------------------------------
# the three methods
# --------------------------------------------------------------------------


def colour_separation(
    image: np.ndarray, region: np.ndarray | None = None, cut: float = 0.35
) -> np.ndarray:
    """Fixed cut on the discriminant. The simplest thing that could work.

    Kept unchanged as the **control**, so the adaptive method has something to be
    compared against. Its measured failure mode is exactly what a fixed constant
    does: on 3 of 9 patients the overall skin tone sat near 0.35 and everything
    passed.
    """
    return (lip_discriminant(image) >= cut) & _region_or_all(image, region)


def otsu_threshold(
    image: np.ndarray,
    region: np.ndarray | None = None,
    normalisation: str = DEFAULT_NORMALISATION,
) -> np.ndarray:
    """The same discriminant with a **per-image** cut, chosen inside the region.

    This is the adaptive threshold. Computing Otsu inside the region rather than
    over the frame is what makes it a lip-versus-skin cut instead of a
    face-versus-background one -- the same correction as ``WEAK_SEPARABILITY``,
    and the reason this method returned whole-face masks in run 1.
    """
    selected = _region_or_all(image, region)
    if selected.sum() < 2:
        return np.zeros(image.shape[:2], dtype=bool)
    discriminant = discriminant_for(image, selected, normalisation)
    threshold, _ = otsu(discriminant[selected])
    return (discriminant >= threshold) & selected


def chan_vese(
    image: np.ndarray,
    region: np.ndarray | None = None,
    iterations: int = 40,
    smoothing: float = 1.0,
    relaxation: float = 0.0,
    normalisation: str = DEFAULT_NORMALISATION,
) -> np.ndarray:
    """Region-based active contour ("active contours without edges").

    Each step reassigns every pixel to whichever region mean it is closer to, then
    smooths the indicator -- which is the curvature term, doing what it does in
    the variational form without needing a level set.

    Region-based deliberately: an edge-seeking snake assumes a lip-skin edge
    exists to be found, which is exactly the assumption under test.

    The two means are taken **inside the search region only**. Taken over the
    frame they were skin-mean and pad-mean, which is why this returned the face.

    ``relaxation`` moves the decision boundary outward by that fraction of the
    half-gap between the two means -- see ``CONTOUR_RELAXATION``. At 0 the
    boundary sits where the two means are equidistant, which is where
    between-class separation is maximised and slightly inside the anatomical
    edge.
    """
    from scipy.ndimage import gaussian_filter

    selected = _region_or_all(image, region)
    if selected.sum() < 2:
        return np.zeros(image.shape[:2], dtype=bool)

    discriminant = discriminant_for(image, selected, normalisation).astype(np.float64)
    inside = (discriminant >= discriminant[selected].mean()) & selected

    for _ in range(iterations):
        if not inside.any() or not (selected & ~inside).any():
            break
        inner_mean = discriminant[inside].mean()
        outer_mean = discriminant[selected & ~inside].mean()

        # The midpoint is the unrelaxed boundary; shifting it toward the OUTER
        # mean admits slightly less lip-like pixels, which is outward.
        midpoint = (inner_mean + outer_mean) / 2.0
        threshold = midpoint - relaxation * (inner_mean - outer_mean) / 2.0

        indicator = (discriminant >= threshold).astype(np.float64)
        updated = (gaussian_filter(indicator, sigma=smoothing) >= 0.5) & selected
        if np.array_equal(updated, inside):
            break
        inside = updated
    return inside


# --------------------------------------------------------------------------
# the oral fissure, and the upper-lip constraint
# --------------------------------------------------------------------------

#: How much of the mask's vertical extent is excluded from the fissure search at
#: each end. The fissure lies BETWEEN two lips, so it is interior by anatomy --
#: without this the darkest row is often the mask's own bottom edge, where the
#: contour is thinning into shadow.
FISSURE_INTERIOR_MARGIN = 0.25

#: A candidate fissure row must be at least this wide relative to the mask's
#: widest row. A two-pixel row at the corner of the mask can be arbitrarily dark
#: by noise alone and would win a plain argmin.
FISSURE_MIN_WIDTH_FRACTION = 0.5

#: How much darker the fissure row must be than the mask's typical row, in value
#: units on [0, 1]. Below this there is no dark line and the mask is one lip.
#:
#: **Without this the detector fires on every mask.** A uniform upper lip has a
#: darkest row like anything else, and cutting there would halve a correct
#: segmentation -- turning the fix into a new failure mode that looks like
#: success on the screen, because the result would satisfy
#: ``fraction_above_fissure`` by construction.
FISSURE_MIN_DEPTH = 0.08


#: Candidate dip thresholds, swept on the same nine patients so the choice is
#: made from a distribution rather than from one fixture. [MEASURED run 4] 0.08
#: fired on 2 of 9: it separated the synthetic fixtures cleanly (0.0000 against
#: 0.1373) and real fissures are shallower than that.
FISSURE_THRESHOLD_SWEEP = (0.02, 0.03, 0.05, 0.08)

#: A dip shallower than this many standard deviations of the row profile is not
#: distinguishable from the profile's own noise. **This is what defines the floor
#: below which no threshold recovers anything** -- lowering the depth threshold
#: past a dip like that buys detections that are arbitrary dark rows, which is
#: strictly worse than not firing, because a false fissure cuts a correct mask in
#: half and the halved mask then satisfies the criterion by construction.
FISSURE_MIN_SHARPNESS = 1.0


def fissure_scan(image: np.ndarray, mask: np.ndarray) -> dict:
    """Measure the dip, whatever its size. Detection is a separate decision.

    Returns the candidate row and **how deep and how sharp** the dip is, so a
    threshold can be chosen from a distribution instead of from found/not-found.
    ``depth`` is in value units; ``sharpness`` is that depth in standard
    deviations of the row profile, which is what separates *a shallow dark line*
    from *the darkest row of something uniform*. Depth alone cannot: at 0.03 both
    look identical.
    """
    empty = {
        "row": None,
        "depth": 0.0,
        "sharpness": 0.0,
        "row_relative": None,
        "n_candidates": 0,
        "profile_sd": 0.0,
    }

    selected = np.asarray(mask, dtype=bool)
    if not selected.any():
        return empty

    rows = np.nonzero(selected.any(axis=1))[0]
    top, bottom = int(rows.min()), int(rows.max())
    height = bottom - top + 1
    if height < 5:
        return empty

    widths = selected.sum(axis=1).astype(float)
    widest = widths[top : bottom + 1].max()
    if widest <= 0:
        return empty

    value = to_hsv(image)[..., 2]
    margin = max(1, int(round(FISSURE_INTERIOR_MARGIN * height)))
    candidates = np.array(
        [
            row
            for row in range(top + margin, bottom - margin + 1)
            if widths[row] >= FISSURE_MIN_WIDTH_FRACTION * widest
        ],
        dtype=int,
    )
    if candidates.size == 0:
        return empty

    profile = np.array(
        [float(value[row][selected[row]].mean()) for row in candidates]
    )
    if profile.size >= 3:
        # Edge-replicated, NOT zero-padded. ``mode="same"`` pads with zeros, so
        # the first and last entries come out at two thirds of their true value
        # and the darkest row is always an end of the profile -- a fissure
        # manufactured by the smoothing, on every mask including correct
        # upper-lip-only ones.
        kernel = np.ones(3) / 3.0
        profile = np.convolve(np.pad(profile, 1, mode="edge"), kernel, mode="valid")

    index = int(np.argmin(profile))
    depth = float(np.median(profile) - profile.min())
    spread = float(profile.std())
    return {
        "row": int(candidates[index]),
        "depth": round(depth, 5),
        "sharpness": round(depth / spread, 4) if spread > 1e-9 else 0.0,
        # Where in the mask the candidate sits, 0 at the top and 1 at the bottom.
        # A fissure between two lips should be near the middle; a candidate at
        # 0.9 is the mask thinning into shadow.
        "row_relative": round(float((candidates[index] - top) / max(height - 1, 1)), 4),
        "n_candidates": int(candidates.size),
        "profile_sd": round(spread, 5),
    }


def fissure_row(
    image: np.ndarray, mask: np.ndarray, min_depth: float | None = None
) -> int | None:
    """The darkest horizontal line inside a lip mask: the oral fissure.

    **[MEASURED 2026-07-28, run 3] Why an explicit anatomical rule is needed.**
    Chan-Vese segments regions of similar intensity, and the two lips *are*
    similar in intensity. The oral fissure is a thin dark line between two
    lookalike regions, so an intensity-based method has no reason to stop there;
    whether it crosses depends only on how pronounced the fissure happens to be
    in that image. Run 2 already crossed on 3 of 9 at relaxation 0, and the
    relaxation made a pre-existing failure universal rather than creating it --
    so **reducing the relaxation returns to 3 of 9, not to 0**.

    **This is why the constraint is the fix and not an addition to a smaller
    relaxation.** Any intensity-based segmenter needs an explicit anatomical rule
    to stop at the fissure, because intensity does not encode one.

    **Why it matters that it is the upper lip.** SymNose measures the upper lip.
    A both-lips mask is a different instrument: its mirrored asymmetry includes
    lower-lip variation, which is not part of the construct, and the resulting
    index would not be comparable with SymNose's.

    Returns the row index, or ``None`` if the mask is too small or too thin for
    the question to be meaningful, or if the dip is not a dark **line** -- in
    which case nothing is constrained and the caller reports that rather than
    silently returning the mask unchanged.

    Two requirements, and both matter. **Depth** rejects a uniform lip whose
    darkest row is merely darkest. **Sharpness** rejects a dip that is shallow
    *and* indistinguishable from the profile's own noise, which is what a lower
    depth threshold would otherwise start admitting. A false fissure is worse
    than none: it halves a correct mask, and the halved mask then satisfies
    ``fraction_above_fissure`` by construction.
    """
    scan = fissure_scan(image, mask)
    if scan["row"] is None:
        return None

    limit = FISSURE_MIN_DEPTH if min_depth is None else min_depth
    if scan["depth"] < limit:
        return None
    if scan["sharpness"] < FISSURE_MIN_SHARPNESS:
        return None
    return scan["row"]


def sweep_fissure_thresholds(
    scans: list[dict], thresholds=FISSURE_THRESHOLD_SWEEP
) -> dict:
    """What each candidate depth threshold would have detected, on one sweep.

    The candidate row does not move with the threshold -- ``argmin`` is
    threshold-independent -- so this is purely about accept/reject and needs no
    re-run. **Detection rate is not the outcome on its own.** A threshold low
    enough to fire on 9 of 9 while landing on arbitrary dark rows is *worse* than
    one that fires on 2, so ``n_sharp`` counts only the detections that are also
    distinguishable from profile noise, and ``n_noise`` counts what the threshold
    would buy that is not.
    """
    depths = [s["depth"] for s in scans if s.get("row") is not None]
    sharp = [s["sharpness"] for s in scans if s.get("row") is not None]

    table = {}
    for threshold in thresholds:
        accepted = [
            (depth, sharpness)
            for depth, sharpness in zip(depths, sharp)
            if depth >= threshold
        ]
        table[str(threshold)] = {
            "n_found": len(accepted),
            "n_sharp": sum(1 for _, s in accepted if s >= FISSURE_MIN_SHARPNESS),
            "n_noise": sum(1 for _, s in accepted if s < FISSURE_MIN_SHARPNESS),
        }
    return table


# --------------------------------------------------------------------------
# the geometric split -- what the fissure detection was proxying for
# --------------------------------------------------------------------------

#: How the lip mask is divided into upper and lower.
#:
#: **[MEASURED run 4 sweep] Why the fissure route was abandoned.** There is no
#: viable depth threshold. 0.03 buys two more sharp detections and one noise
#: detection -- and a noise detection is exactly the false positive that halves a
#: correct mask while satisfying the criterion by construction. The depths say
#: why: seven of nine sit below 0.05, and the only real gap in the distribution
#: is between 0.047 and 0.114. There is no cut that separates them.
#:
#: So the mask is divided at a position derived from **its own shape**. No dark
#: line is needed, it applies uniformly to every patient, and it is what the
#: detection was proxying for in the first place.
#:
#: ``centroid``  the row of the mask's centroid -- shape-derived, and it moves
#:               with a bottom-heavy or top-heavy mask.
#: ``fraction``  a fixed fraction of the mask's vertical extent.
#: ``fissure``   the detected dark line. Kept so run 4's behaviour can be
#:               re-derived and so the two can be compared, NOT as the operating
#:               rule.
SPLIT_RULES = ("centroid", "fraction", "fissure")
DEFAULT_SPLIT_RULE = "centroid"

#: A priori, not fitted. In a closed mouth the vermillion of the upper lip and
#: the lower lip are roughly comparable in height, so the division sits near the
#: middle of the combined mask. **This value is declared, not tuned** -- see
#: ``validate_split`` for why tuning it would be worthless.
SPLIT_FRACTION = 0.5

#: Sharpness a detection needs to enter the **validation set**. Deliberately
#: looser than the operating depth threshold, and that is legitimate because the
#: two answer different questions:
#:
#: * the operating rule asks *what do I run on every patient in production*, and
#:   must not fire on a mask with no division;
#: * the validation set asks *where do I trust the ground truth*, and a shallow
#:   but sharp detection is trustworthy ground truth even where it would be too
#:   risky to act on automatically.
#:
#: **This is not the operating criterion loosened.** Depth >= 0.08 gives two
#: patients; sharpness >= 1.4 gives more, and several of those detections are
#: real but shallow. Using them to check the geometric split is what they are
#: for.
VALIDATION_MIN_SHARPNESS = 1.4


def centroid_row(mask: np.ndarray) -> int | None:
    """The row of the mask's centroid."""
    selected = np.asarray(mask, dtype=bool)
    if not selected.any():
        return None
    rows, _ = np.nonzero(selected)
    return int(round(float(rows.mean())))


def fraction_row(mask: np.ndarray, fraction: float = SPLIT_FRACTION) -> int | None:
    """A fixed fraction down the mask's vertical extent."""
    selected = np.asarray(mask, dtype=bool)
    if not selected.any():
        return None
    rows = np.nonzero(selected.any(axis=1))[0]
    top, bottom = int(rows.min()), int(rows.max())
    return int(round(top + fraction * (bottom - top)))


def split_row(
    image: np.ndarray,
    mask: np.ndarray,
    rule: str = DEFAULT_SPLIT_RULE,
    fraction: float = SPLIT_FRACTION,
    min_depth: float | None = None,
) -> int | None:
    """Where to divide this lip mask into upper and lower."""
    if rule == "centroid":
        return centroid_row(mask)
    if rule == "fraction":
        return fraction_row(mask, fraction)
    if rule == "fissure":
        return fissure_row(image, mask, min_depth)
    raise SegmentationError(
        f"unknown split rule {rule!r}; expected one of {SPLIT_RULES}"
    )


def validate_split(records: list[dict]) -> dict:
    """Do the geometric splits land where the detected fissures are?

    **The two detected fissures are the whole point of run 4.** Where a fissure
    was found on evidence, it is ground truth for where the division actually
    is; if a shape-derived line lands in the same place, the shape predicts the
    anatomy and the split is a defensible substitute on the patients where no
    fissure could be found. If it does not, **the geometry does not predict the
    anatomy and this route fails too -- which is a result, not a setback.**

    **The disagreement is reported, not minimised.** If the split lands 8 pixels
    below the fissure on average, that is the measurement. Tuning
    ``SPLIT_FRACTION`` until the error is smallest would fit it to the two or
    four patients that have a detected fissure and say nothing whatever about the
    other five -- which are the patients the split exists to serve.
    """
    eligible = [r for r in records if r.get("validation_eligible")]
    report = {
        "n_eligible": len(eligible),
        "n_records": len(records),
        "selected_on": f"sharpness >= {VALIDATION_MIN_SHARPNESS}",
        "selection_note": (
            "The validation set is selected on SHARPNESS, which is looser than "
            "the operating depth threshold. That is deliberate and it is not the "
            "criterion loosened: 'where do I trust the ground truth' and 'what do "
            "I run on every patient' are different questions. A shallow but sharp "
            "detection is trustworthy evidence of where the division is, even "
            "where firing on it automatically would be too risky."
        ),
        "rules": {},
        "note": (
            "Errors are SIGNED, in pixels, split minus fissure -- positive means "
            "the split sits BELOW the detected fissure, so the upper-lip mask "
            "would keep some lower lip. THE DISAGREEMENT IS THE MEASUREMENT. Do "
            "not tune SPLIT_FRACTION against it: that fits the line to the "
            "handful of patients with a detected fissure and says nothing about "
            "the ones the split exists to serve. And these numbers say how close "
            "the split is on the validated cases only -- ONLY THE SHEET says "
            "whether it lands on the upper lip for the rest."
        ),
    }

    for rule in ("centroid", "fraction"):
        errors = [
            r[f"split_row_{rule}"] - r["fissure_row_detected"]
            for r in eligible
            if r.get(f"split_row_{rule}") is not None
            and r.get("fissure_row_detected") is not None
        ]
        relative = [
            (r[f"split_row_{rule}"] - r["fissure_row_detected"])
            / max(r.get("split_mask_height") or 1, 1)
            for r in eligible
            if r.get(f"split_row_{rule}") is not None
            and r.get("fissure_row_detected") is not None
        ]
        report["rules"][rule] = {
            "n": len(errors),
            "errors_px": errors,
            "mean_error_px": round(float(np.mean(errors)), 2) if errors else None,
            "median_error_px": round(float(np.median(errors)), 2) if errors else None,
            "max_abs_error_px": int(max(abs(e) for e in errors)) if errors else None,
            "mean_error_as_mask_height": (
                round(float(np.mean(relative)), 4) if relative else None
            ),
        }
    return report


def above_fissure(mask: np.ndarray, row: int | None) -> np.ndarray:
    """The part of a mask above the fissure. The upper lip."""
    selected = np.asarray(mask, dtype=bool)
    if row is None:
        return selected.copy()
    constrained = np.zeros_like(selected)
    constrained[:row] = selected[:row]
    return constrained


def fraction_above_split(mask: np.ndarray, row: int | None) -> float:
    """How much of this mask is upper lip. 1.0 when no fissure was found."""
    selected = np.asarray(mask, dtype=bool)
    total = int(selected.sum())
    if total == 0:
        return 0.0
    if row is None:
        return 1.0
    return float(selected[:row].sum() / total)


def segment(
    image: np.ndarray,
    method: str,
    region: np.ndarray | None = None,
    relaxation: float = CONTOUR_RELAXATION,
    normalisation: str = DEFAULT_NORMALISATION,
    min_depth: float | None = None,
    split_rule: str = DEFAULT_SPLIT_RULE,
) -> np.ndarray:
    if method == "colour_separation":
        # NOT normalised, deliberately. This method is the run-1 record of what a
        # fixed cut does; normalising would turn its fixed cut into a quantile
        # cut and it would no longer be that record.
        return colour_separation(image, region)
    if method == "otsu_threshold":
        return otsu_threshold(image, region, normalisation)
    if method == "chan_vese":
        # The unrelaxed contour, kept so the relaxation's effect is visible on
        # the sheet rather than being a number nobody can see the size of.
        return chan_vese(image, region, normalisation=normalisation)
    if method == "chan_vese_relaxed":
        return chan_vese(
            image, region, relaxation=relaxation, normalisation=normalisation
        )
    if method == "upper_lip":
        contour = chan_vese(
            image, region, relaxation=relaxation, normalisation=normalisation
        )
        return above_fissure(
            contour,
            split_row(image, contour, split_rule, min_depth=min_depth),
        )
    raise SegmentationError(f"unknown method {method!r}; expected one of {METHODS}")


def iou(a: np.ndarray, b: np.ndarray) -> float:
    """Intersection over union of two masks. 0 when both are empty."""
    first, second = np.asarray(a, dtype=bool), np.asarray(b, dtype=bool)
    union = int((first | second).sum())
    return float((first & second).sum() / union) if union else 0.0


# --------------------------------------------------------------------------
# describing a mask
# --------------------------------------------------------------------------


def largest_component_share(mask: np.ndarray) -> float:
    """Fraction of the mask in its biggest connected blob. 0 for an empty mask."""
    from scipy.ndimage import label

    if not mask.any():
        return 0.0
    labelled, count = label(mask)
    if count == 0:
        return 0.0
    sizes = np.bincount(labelled.ravel())[1:]
    return float(sizes.max() / sizes.sum())


def describe_mask(
    mask: np.ndarray,
    region: np.ndarray | None = None,
    fissure: int | None = None,
) -> dict:
    """Shape and position of a candidate segmentation. No pixels, so SHAREABLE.

    ``area`` stays **frame-relative, exactly as declared**, and it is what
    ``is_plausible`` judges. ``area_within_region`` and ``saturates_region`` are
    additional descriptors, reported and deliberately **not** part of the verdict.

    **Read the warning in ``is_plausible`` before trusting a verdict taken under a
    spatial prior.** These two fields are what that warning is about.
    """
    mask = np.asarray(mask, dtype=bool)
    height, width = mask.shape[:2]
    selected = _region_or_all(mask, region)
    region_size = int(selected.sum())

    base = {
        "area": round(float(mask.mean()), 5),
        "region_fraction_of_frame": round(region_size / (height * width), 5),
        "area_within_region": 0.0,
        "saturates_region": False,
        "n_pixels": int(mask.sum()),
        "fissure_row": fissure,
        "fraction_above_split": round(fraction_above_split(mask, fissure), 4),
    }

    if not mask.any():
        return {**base, "centroid_x": 0.0, "centroid_y": 0.0, "largest_component": 0.0}

    within = float(mask.sum() / region_size) if region_size else 0.0
    rows, columns = np.nonzero(mask)
    return {
        **base,
        # Same precision as `area` deliberately: with no search region the two
        # ARE the same quantity, and reporting them at different precision would
        # make identical numbers look different.
        "area_within_region": round(within, 5),
        "saturates_region": bool(within >= SATURATION_LIMIT),
        "centroid_x": round(float(columns.mean() / width), 4),
        "centroid_y": round(float(rows.mean() / height), 4),
        "largest_component": round(largest_component_share(mask), 4),
    }


def is_plausible(description: dict) -> tuple[bool, list[str]]:
    """Does this look like a lip region? Returns the verdict and every reason.

    Every failing criterion is reported, not just the first. "It failed on area"
    and "it failed on area, position and connectivity" are different findings, and
    the second is much stronger evidence that the method found nothing at all.

    **This is a screen, not a verdict.** It flags candidates worth a human
    looking at, and nothing here has ever decided anything on its own -- the
    segmentation question was settled twice by reading the contact sheet. Worth
    having and worth declaring in advance, so that what counts as a candidate is
    fixed before the candidates are seen; but a method passing this screen has
    not passed a gate, and the write-up should not imply it did.

    **The numbers in ``PLAUSIBLE`` are unchanged and must stay that way.** What
    changed is which denominator ``area_max`` is read against, and the reasoning
    -- including that it is post-hoc -- is in ``CRITERIA_PROVENANCE``.

    ``centroid_y`` still deserves the caveat: once a bottom-band prior is used it
    is satisfied **by construction**, so it can no longer fail and has stopped
    being evidence. It is left in place because it is exactly the criterion that
    matters when the prior is off.
    """
    failures = _failures(description)
    return (not failures), [message for _, message in failures]


#: The criteria a mask can fail, as keys rather than prose. A breakdown by
#: criterion has to group on something stable, and grouping on message text would
#: break the moment a message was reworded.
CRITERION_NAMES = (
    "area_min",
    "area_max",
    "centroid_y",
    "centroid_x",
    "largest_component",
    "fraction_above_split",
)


def failed_criteria(description: dict) -> list[str]:
    """Which criteria this mask fails, by name. Empty means it passes."""
    return [name for name, _ in _failures(description)]


def _failures(description: dict) -> list[tuple[str, str]]:
    reasons: list[tuple[str, str]] = []

    # Two ends, two denominators. See CRITERIA_PROVENANCE for why, and for the
    # measurement showing that one denominator leaves a hole at whichever end it
    # is not suited to. With no search region the two coincide exactly, so this
    # reduces to the pre-declared form.
    area_frame = description["area"]
    area_region = description.get("area_within_region", area_frame)

    if area_frame < PLAUSIBLE["area_min"]:
        reasons.append((
            "area_min",
            f"area {area_frame:.3f} of the frame is below {PLAUSIBLE['area_min']} "
            "-- a speckle, not a structure",
        ))
    if area_region > PLAUSIBLE["area_max"]:
        reasons.append((
            "area_max",
            f"area {area_region:.3f} of the SEARCH REGION is above "
            f"{PLAUSIBLE['area_max']} -- it took essentially everything it was "
            "allowed to look at",
        ))
    if description["centroid_y"] < PLAUSIBLE["centroid_y_min"]:
        reasons.append((
            "centroid_y",
            f"centroid_y {description['centroid_y']:.3f} is above "
            f"{PLAUSIBLE['centroid_y_min']} -- too high on the face for lips",
        ))
    if not (
        PLAUSIBLE["centroid_x_min"]
        <= description["centroid_x"]
        <= PLAUSIBLE["centroid_x_max"]
    ):
        reasons.append((
            "centroid_x",
            f"centroid_x {description['centroid_x']:.3f} is off the midline",
        ))
    if description["largest_component"] < PLAUSIBLE["largest_component_min"]:
        reasons.append((
            "largest_component",
            f"largest component is {description['largest_component']:.3f} of the "
            "mask -- fragmented rather than one region",
        ))

    # Absent when no fissure was located, in which case the question was not
    # asked rather than answered yes -- `fissure_row: None` says which.
    above = description.get("fraction_above_split")
    if above is not None and above < PLAUSIBLE["fraction_above_split_min"]:
        reasons.append((
            "fraction_above_split",
            f"only {above:.3f} of the mask is above the split, below "
            f"{PLAUSIBLE['fraction_above_split_min']} -- it has crossed onto "
            "the lower lip, and SymNose measures the UPPER lip",
        ))
    return reasons


# --------------------------------------------------------------------------
# the diagnostic
# --------------------------------------------------------------------------


@dataclass
class MethodResult:
    method: str
    mask: np.ndarray = field(repr=False)
    description: dict = field(default_factory=dict)
    plausible: bool = False
    reasons: list[str] = field(default_factory=list)


def run_methods(
    image: np.ndarray,
    spatial_prior: str = DEFAULT_SPATIAL_PRIOR,
    config=None,
    use_face_mask: bool = True,
    relaxation: float = CONTOUR_RELAXATION,
    normalisation: str = DEFAULT_NORMALISATION,
    min_depth: float | None = None,
    split_rule: str = DEFAULT_SPLIT_RULE,
) -> list[MethodResult]:
    """Every method on one staged image, inside the analysis region.

    The split is located **once**, from the unconstrained relaxed contour -- the
    most complete lip mask available -- and every method is then scored against
    that same line. Locating it per method would let a method that found only the
    upper lip declare its own top edge the division and pass trivially.
    """
    region = analysis_region(image, spatial_prior, config, use_face_mask)
    contour = chan_vese(
        image, region, relaxation=relaxation, normalisation=normalisation
    )
    division = split_row(image, contour, split_rule, min_depth=min_depth)

    results = []
    for method in METHODS:
        mask = segment(
            image, method, region, relaxation, normalisation, min_depth, split_rule
        )
        description = describe_mask(mask, region, division)
        plausible, reasons = is_plausible(description)
        results.append(
            MethodResult(
                method=method,
                mask=mask,
                description=description,
                plausible=plausible,
                reasons=reasons,
            )
        )
    return results


def attribute(
    results: list[MethodResult],
    image: np.ndarray,
    region: np.ndarray,
    split_rule: str = DEFAULT_SPLIT_RULE,
) -> dict:
    """Why did this patient come out as it did? Method breakdown, or the image?

    **A failure that is genuine difficulty is a limitation; a failure that is
    method breakdown is a defect.** They are recorded differently in the
    write-up, so the diagnostic should say which rather than leaving it to be
    inferred from a thumbnail.

    The discriminator is **agreement between two independent methods**:

    * ``instrument_comparator_iou`` high but both implausible -> the two methods
      found the same thing and it is not lip-shaped. That is the image: a poor
      repair with no clear vermillion border, an open mouth putting teeth where
      the lip should be, or lighting that flattens the colour difference.
      **Genuine difficulty.**
    * ``instrument_comparator_iou`` low -> the methods disagree about what is
      there, so at least one is breaking down. **Method breakdown**, and it needs
      fixing rather than recording.

    The lighting and exposure figures are here to separate the sub-cases of the
    first, not to gate anything.
    """
    by_method = {result.method: result for result in results}
    instrument = by_method[INSTRUMENT]
    comparator = by_method[COMPARATOR]
    unrelaxed = by_method["chan_vese"]
    unconstrained = by_method[UNCONSTRAINED]

    hsv = to_hsv(image)
    selected = np.asarray(region, dtype=bool)
    agreement = iou(instrument.mask, comparator.mask)
    any_plausible = any(r.plausible for r in results)

    scan = fissure_scan(image, unconstrained.mask)
    rows = np.nonzero(np.asarray(unconstrained.mask, dtype=bool).any(axis=1))[0]
    mask_height = int(rows.max() - rows.min() + 1) if rows.size else 0

    growth = (
        float(unconstrained.mask.sum() / unrelaxed.mask.sum())
        if unrelaxed.mask.any()
        else 0.0
    )

    return {
        "instrument": INSTRUMENT,
        "comparator": COMPARATOR,
        # Satisfied by construction for the instrument, so the number that
        # carries information is the contour it was cut from.
        "fraction_above_split_unconstrained": unconstrained.description.get(
            "fraction_above_split"
        ),
        "split_row": unconstrained.description.get("fissure_row"),
        "split_found": unconstrained.description.get("fissure_row") is not None,
        # The fissure MEASUREMENT, reported whether or not it cleared the
        # threshold. It no longer drives the instrument -- it is the validation
        # evidence the geometric split is checked against.
        "fissure_scan": scan,
        "fissure_row_detected": (
            scan["row"] if scan["sharpness"] >= VALIDATION_MIN_SHARPNESS else None
        ),
        "validation_eligible": bool(
            scan["row"] is not None
            and scan["sharpness"] >= VALIDATION_MIN_SHARPNESS
        ),
        # Both shape-derived candidates, so the sheet and the validation can
        # compare them without a second run.
        "split_rule": split_rule,
        "split_row_centroid": centroid_row(unconstrained.mask),
        "split_row_fraction": fraction_row(unconstrained.mask),
        "split_mask_height": mask_height,
        # How much the split removed. ~0.5 for a both-lips mask -- and ALSO ~0.5
        # for a mask that was already upper-lip only, which is exactly why this
        # number cannot distinguish the two. See the `split` block's limitation.
        "split_removed_fraction": (
            round(
                1.0 - float(instrument.mask.sum() / unconstrained.mask.sum()), 4
            )
            if unconstrained.mask.any()
            else 0.0
        ),
        "crossed_the_split": bool(
            (unconstrained.description.get("fraction_above_split") or 1.0)
            < PLAUSIBLE["fraction_above_split_min"]
        ),
        "instrument_comparator_iou": round(agreement, 4),
        "instrument_plausible": instrument.plausible,
        "comparator_plausible": comparator.plausible,
        "any_method_plausible": any_plausible,
        # How much the relaxation added on THIS patient. Watch for the mask
        # crossing into the lower lip -- see CONTOUR_RELAXATION.
        "relaxation_growth": round(growth, 4),
        "relaxation_growth_high": bool(growth > RELAXATION_GROWTH_LIMIT),
        # Skin-tone proxy. Median brightness inside the search region: lower is
        # darker. A PROXY, not skin tone -- lighting moves it too -- but it is
        # what the pixels can say, and the fairness signature showed up in
        # exactly these two channels.
        "median_value_in_region": round(
            float(np.median(hsv[..., 2][selected])) if selected.any() else 0.0, 4
        ),
        "median_saturation_in_region": round(
            float(np.median(hsv[..., 1][selected])) if selected.any() else 0.0, 4
        ),
        "separability": round(separability_of(image, region), 4),
        "attribution": (
            "ok"
            if any_plausible
            else ("genuine_difficulty" if agreement >= 0.5 else "method_breakdown")
        ),
        "attribution_note": (
            "ok: at least one method produced a plausible region. "
            "genuine_difficulty: the instrument and the comparator agree (IoU >= "
            "0.5) and neither is lip-shaped -- the two methods found the same "
            "thing and it is not a lip. Record as a LIMITATION. "
            "method_breakdown: the two disagree, so at least one is failing -- "
            "that is a DEFECT to fix, not a limitation to record. "
            "**THESE TWO DO NOT EXHAUST THE CAUSES, AND THE LABELS ARE ABOUT "
            "AGREEMENT, NOT ABOUT CAUSE.** [MEASURED 2026-07-28] On run 3 "
            "'genuine_difficulty' reached the right verdict for the wrong "
            "reason: it reads as 'hard image', and the actual cause was 'every "
            "method here assumes a skin tone this patient does not have'. That "
            "is a FAIRNESS limitation, not an image-quality one, and the two are "
            "recorded differently. Check median_value_in_region and "
            "median_saturation_in_region against the rest of the sample before "
            "writing down a cause. "
            "Either way, LOOK AT THE SHEET: this attribution is a prompt, not a "
            "diagnosis."
        ),
    }


def separability_of(
    image: np.ndarray, region: np.ndarray | None = None
) -> float:
    """Otsu's eta on the discriminant, **inside the region**.

    Over the whole frame this measures face against white padding and scored
    0.888 on run 1 while saying nothing about lips. Inside the face -- and inside
    the spatial prior, if one is used -- it measures the contrast the gate is
    actually about. See ``WEAK_SEPARABILITY``.

    Passing ``region=None`` reproduces the run-1 quantity. It is kept only so the
    two can be reported side by side and the correction is visible rather than
    asserted.
    """
    selected = _region_or_all(image, region)
    if selected.sum() < 2:
        return 0.0
    return otsu(lip_discriminant(image)[selected])[1]


def overlay(image: np.ndarray, mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Tint the segmented region green, for the contact sheet.

    Green because the crops are red-dominated and white-padded: a red or white
    overlay would be invisible on exactly the pixels being judged.
    """
    canvas = np.asarray(image, dtype=np.float32).copy()
    if canvas.ndim == 2:
        canvas = np.stack([canvas] * 3, axis=2)
    tint = np.array([0.0, 255.0, 0.0], dtype=np.float32)
    selected = np.asarray(mask, dtype=bool)
    canvas[selected] = (1 - alpha) * canvas[selected] + alpha * tint
    return np.clip(canvas, 0, 255).astype(np.uint8)


def verdict(reports: list[dict]) -> dict:
    """The aggregate screen output. SHAREABLE.

    A method is **usable** if it is plausible on a majority of the patients
    looked at. One good result out of nine is a coincidence, not a technique.

    **This is a screen, not a gate, and the name of the field says so.** The
    segmentation question has been decided twice, both times  reading
    the contact sheet -- run 1 came back "undecided, not out" from exactly that
    reading, against a screen that had said "proceed". These numbers narrow what
    is worth looking at and fix what counts as a candidate before the candidates
    are seen. They do not decide, and the write-up should not describe them as
    having decided.
    """
    if not reports:
        raise SegmentationError("no per-patient reports to aggregate")

    n = len(reports)
    separability = [r["separability"] for r in reports]
    by_method = {}
    for method in METHODS:
        entries = [r["methods"][method] for r in reports]
        plausible = [e["plausible"] for e in entries]
        saturating = [e.get("saturates_region", False) for e in entries]
        by_method[method] = {
            "n_plausible": int(sum(plausible)),
            "n_patients": n,
            "fraction_plausible": round(sum(plausible) / n, 4),
            "usable": sum(plausible) > n / 2,
            # Plausible AND not merely selecting everything it was allowed to.
            # Reported beside `n_plausible` so the difference is visible without
            # changing what `usable` means between runs.
            "n_plausible_not_saturating": int(
                sum(p and not s for p, s in zip(plausible, saturating))
            ),
            "n_saturating_region": int(sum(saturating)),
            "mean_area": round(float(np.mean([e["area"] for e in entries])), 5),
            "mean_area_within_region": round(
                float(np.mean([e.get("area_within_region", 0.0) for e in entries])), 4
            ),
            "mean_centroid_y": round(
                float(np.mean([e["centroid_y"] for e in entries])), 4
            ),
        }

    # Which patients no method handled, and why. Counts and attributions only --
    # the ids are patient-keyed and stay in the CLUSTER-ONLY per-patient file.
    attributions = [r.get("attribution", {}) for r in reports]
    failures = [a for a in attributions if a.get("attribution") not in (None, "ok")]
    failure_summary = {
        "n_failures": len(failures),
        "by_attribution": {
            kind: sum(1 for a in failures if a.get("attribution") == kind)
            for kind in ("genuine_difficulty", "method_breakdown")
        },
        "note": (
            "genuine_difficulty is a LIMITATION -- record it with the reason. "
            "method_breakdown is a DEFECT -- fix it. The failing patients are "
            "named in segmentation_per_patient.json, which is CLUSTER-ONLY."
        ),
    }

    growths = [
        a["relaxation_growth"] for a in attributions if "relaxation_growth" in a
    ]
    relaxation_report = {
        "relaxation": CONTOUR_RELAXATION,
        "mean_growth": round(float(np.mean(growths)), 4) if growths else 0.0,
        "max_growth": round(float(np.max(growths)), 4) if growths else 0.0,
        "n_growth_high": sum(
            1 for a in attributions if a.get("relaxation_growth_high")
        ),
        "growth_limit": RELAXATION_GROWTH_LIMIT,
        "note": (
            "One fixed relaxation, applied identically to every patient as a "
            "fraction of that image's own class gap -- not a per-patient tune. "
            "SymNose measures the UPPER lip, so over-relaxing crosses the oral "
            "fissure into the lower lip and silently changes what is measured. "
            "The unrelaxed contour is rendered beside the relaxed one on the "
            "sheet; judge the magnitude there, not from this number."
        ),
    }

    scans = [a["fissure_scan"] for a in attributions if a.get("fissure_scan")]
    depths = sorted(s["depth"] for s in scans)
    sharpness = [s["sharpness"] for s in scans]

    # [MEASURED run 4] Is there a floor? A dip that is shallow AND within the
    # profile's own noise is not a faint fissure -- there is nothing there to
    # find, and no threshold recovers it. That is a property of the image, and a
    # real limitation of intensity-based upper-lip segmentation on compressed
    # JPEG: in a closed relaxed mouth the fissure is one or two pixels of shadow.
    below_noise = [
        s for s in scans if s["sharpness"] < FISSURE_MIN_SHARPNESS
    ]
    floor = {
        "n_below_noise": len(below_noise),
        "n_patients": n,
        "sharpness_criterion": FISSURE_MIN_SHARPNESS,
        "max_depth_below_noise": (
            round(max((s["depth"] for s in below_noise), default=0.0), 5)
        ),
        "note": (
            "A dip within the row profile's own noise is not a faint fissure -- "
            "there is nothing there, and NO threshold recovers it. If these "
            "patients cluster just under the chosen depth the fix is to lower it; "
            "if they are at the noise floor the fissure is genuinely absent from "
            "the image. In a closed relaxed mouth it is one or two pixels of "
            "shadow in a compressed JPEG. THAT IS A LIMITATION OF INTENSITY-BASED "
            "UPPER-LIP SEGMENTATION ON THIS DATA, not a defect to fix, and it "
            "belongs in the write-up beside the fairness point -- both are "
            "constraints the method inherits."
        ),
    }

    fissure_report = {
        "n_crossing": sum(1 for a in attributions if a.get("crossed_the_split")),
        "n_split_found": sum(1 for a in attributions if a.get("split_found")),
        "split_rule": next(
            (a["split_rule"] for a in attributions if a.get("split_rule")), None
        ),
        # The two detected fissures are the validation set, not the instrument.
        "validation": validate_split(attributions),
        "removed_fraction_sorted": sorted(
            a.get("split_removed_fraction", 0.0) for a in attributions
        ),
        "limitation": (
            "THE GEOMETRIC SPLIT ALWAYS FIRES. It has no not-found case, so it "
            "halves a contour that was ALREADY upper-lip only -- which happens "
            "when a patient's lower lip is outside the crop. "
            "split_removed_fraction is ~0.5 for a both-lips mask and ~0.5 for an "
            "upper-lip-only mask, so it cannot tell them apart, and neither can "
            "any other shape statistic: the two masks look the same. "
            "This is the exact TRADE against the fissure rule, which was correct "
            "wherever it fired and inert on 7 of 9. The geometric rule is right "
            "for the majority whose contour spans both lips and wrong for the "
            "minority whose crop excludes the lower lip. ONLY THE SHEET can say "
            "which patient is which."
        ),
        "split_rows": {
            rule: [a.get(f"split_row_{rule}") for a in attributions]
            for rule in ("centroid", "fraction")
        },
        "n_patients": n,
        "criterion": PLAUSIBLE["fraction_above_split_min"],
        "measured_on": UNCONSTRAINED,
        "n_patients_": n,
        # The distribution, so "the threshold is too strict" is a measurement
        # rather than an inference. Depths only -- no patient ids, so SHAREABLE.
        "depths_sorted": depths,
        "median_depth": round(float(np.median(depths)), 5) if depths else 0.0,
        "sharpness_sorted": sorted(round(v, 4) for v in sharpness),
        "threshold_in_use": FISSURE_MIN_DEPTH,
        "sweep": sweep_fissure_thresholds(scans),
        "floor": floor,
        "note": (
            "Counted on the UNCONSTRAINED contour, because the instrument "
            "satisfies this by construction. Chan-Vese segments regions of "
            "similar intensity and the two lips are similar in intensity, so an "
            "intensity-based method has no reason to stop at the oral fissure -- "
            "whether it crosses depends only on how pronounced the fissure is in "
            "that image. Run 2 crossed on 3 of 9 at relaxation 0 and on 9 of 9 "
            "relaxed, so the relaxation made a pre-existing failure universal "
            "rather than creating it, and REDUCING IT RETURNS TO 3 OF 9, NOT 0. "
            "The explicit anatomical constraint is the fix; the relaxation is "
            "then re-tuned against the upper lip alone. "
            "[RESOLVED run 5] THE FISSURE ROUTE IS ABANDONED. The run-4 sweep "
            "showed no viable threshold: 0.03 bought two more sharp detections "
            "and one NOISE detection, which is the false positive that halves a "
            "correct mask while satisfying the criterion by construction. Seven "
            "of nine depths sit below 0.05 and the only real gap is between "
            "0.047 and 0.114 -- there is no cut that separates them. The split is "
            "now derived from the mask's own SHAPE (see split_rule), which needs "
            "no dark line and applies uniformly. `sweep` and `depths_sorted` are "
            "kept as the evidence for that decision. The detected fissures are "
            "now the VALIDATION set -- see `validation`."
        ),
    }

    # Fairness. The proxy is median brightness inside the search region: lower is
    # darker. It is a proxy, not skin tone -- but the run-3 failure showed up in
    # exactly this channel and the screen has to be expressible in what pixels
    # can say.
    tones = [
        (a.get("median_value_in_region", 1.0), a)
        for a in attributions
        if a.get("median_value_in_region") is not None
    ]
    darkest = min(tones, key=lambda pair: pair[0])[1] if tones else {}
    fairness = {
        "criterion": "the instrument must succeed on the darkest-skinned patient",
        "declared": "2026-07-28, before run 3",
        "proxy": "median_value_in_region (lower is darker)",
        "darkest_median_value": round(min(t for t, _ in tones), 4) if tones else None,
        "lightest_median_value": round(max(t for t, _ in tones), 4) if tones else None,
        "darkest_instrument_plausible": bool(
            darkest.get("instrument_plausible", False)
        ),
        "passes": bool(darkest.get("instrument_plausible", False)),
        "note": (
            "[MEASURED 2026-07-28, run 3] Every method here is intensity- or "
            "redness-based, so all are implicitly calibrated on lighter skin: on "
            "darker skin the lip-versus-skin contrast compresses in both hue and "
            "value. One patient of the nine is not white British, and that "
            "patient is the only failure -- median saturation 0.824 against "
            "0.331-0.492, all methods taking 44-99% of the band, and the mask "
            "escaping onto the chin and cheeks. "
            "THE SAMPLE IS NOT REPRESENTATIVE ON THIS AXIS: n=1 non-white, so "
            "any success rate quoted from it carries that caveat and this "
            "criterion is a very weak instrument. "
            "SymNose was developed and validated on a British cleft cohort, so "
            "an automated reimplementation inherits its POPULATION ASSUMPTIONS "
            "along with its algorithm -- that belongs in the write-up whatever "
            "this screen says. "
            "Any fix must be a rule applying identically to everyone. A "
            "parameter adjusted until this patient works would fit one image and "
            "HIDE the fairness problem rather than solve it."
        ),
    }

    usable = sorted(m for m, v in by_method.items() if v["usable"])
    mean_separability = float(np.mean(separability))
    weak = mean_separability < WEAK_SEPARABILITY
    compromised = sorted(
        m for m, v in by_method.items() if v["usable"] and v["n_saturating_region"]
    )

    return {
        "n_patients": n,
        "methods": by_method,
        "usable_methods": usable,
        "usable_but_saturating": compromised,
        "mean_separability": round(mean_separability, 4),
        "min_separability": round(float(np.min(separability)), 4),
        "separability_is_weak": weak,
        "weak_separability_threshold": WEAK_SEPARABILITY,
        "separability_measures": "lip_vs_skin_within_the_analysis_region",
        "plausibility_criteria": dict(PLAUSIBLE),
        "criteria_provenance": dict(CRITERIA_PROVENANCE),
        "saturation_limit": SATURATION_LIMIT,
        "role": "screen",
        "decided_by": "human review of the contact sheet",
        "symnose_screen": "proceed" if usable else "out",
        "instrument": INSTRUMENT,
        "comparator": COMPARATOR,
        "unconstrained": UNCONSTRAINED,
        "failures": failure_summary,
        "relaxation": relaxation_report,
        "fissure": fissure_report,
        "fairness": fairness,
        "note": (
            "THIS IS A SCREEN, NOT A GATE. It narrows what is worth looking at "
            "and fixes what counts as a candidate before the candidates are seen. "
            "The decision is taken by reading the contact sheet, and has been "
            "both times -- run 1's screen said proceed and the sheet said "
            "undecided. Do not describe this output as having decided anything. "
            "PROCEED means at least one method was plausible on a majority of the "
            "patients looked at, so §3.2 is worth building. OUT means none was, "
            "and the protocol's claim that lip extraction needs raw camera files "
            "rather than compressed JPEG is supported on this cohort -- record it "
            "in limitations with that citation and let the mirror-difference "
            "index carry the symmetry question. "
            "area is screened at TWO denominators: area_min against the frame "
            "('not a speckle', which does not scale with the search area) and "
            "area_max against the search region ('a modest fraction of what was "
            "searched', which does). Both are reported on every mask, so runs "
            "with and without a spatial prior stay comparable on the original "
            "frame-relative number. See criteria_provenance -- the region-relative "
            "reading is POST-HOC. "
            "centroid_y is satisfied by construction under a bottom-band prior "
            "and stops being evidence there. "
            "separability now measures lip-versus-skin INSIDE the face; over the "
            "whole frame it measured face-versus-background, and on synthetic "
            "fixtures a face with NO lips scored higher than one with lips -- so "
            "the run-1 value was anti-informative, not merely uninformative."
        ),
    }
