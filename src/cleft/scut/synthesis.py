"""Unilateral cleft-like asymmetry, synthesised by thin-plate spline.

----------------------------------------------------------------------------
PARKED 2026-07-30: BUILT AND VERIFIED, NOT RUN. See ``PARKED`` below.
Stage F is struck from the arm list. Nothing in Phase 6 depends on this module.
Its tests keep running deliberately; do not extend it.
----------------------------------------------------------------------------

PLAN §4.11 Q-b, **[DECIDED 2026-07-28]: landmark-driven thin-plate spline,
deformation only.** One rule, one magnitude parameter, unilateral, side recorded
and balanced across the set.

**What it does.** Four landmarks on one side are displaced -- alar base, philtral
column, cupid's bow peak, commissure -- a thin-plate spline is fitted to those
displacements against an unmoved anchor ring, and the image is warped by it.
Nothing else moves: no scar, no colour change, no texture synthesis.

----------------------------------------------------------------------------
TWO LIMITATIONS, RECORDED BEFORE THE ARM RUNS
----------------------------------------------------------------------------
These are in ``LIMITATIONS`` and carried into every ``metrics.json``, in the
pattern PLAN §3.2 established for the SymNose construct-note: **properties of the
instrument, not disclaimers about a component of it.**

**1. No scar.** A repaired unilateral cleft lip carries a surgical scar, and on
the Phase 4 segmentation runs Otsu picked the scar up on several patients -- "on
a post-repair cleft, much of what the aesthetic grade is about" (PLAN §4.9 notes).
A deformation-only synthesis produces asymmetric *shape* with intact *texture*,
so a model pretrained on it sees no scar at all. Adding one would mean inventing
appearance, and this project does not have the data to say what a scar looks like
on a SCUT face. **So the gap is stated, not closed.**

**2. The magnitude-to-grade mapping is an assumption.** ``magnitude`` is a
displacement in units of crop width. Nothing here establishes that a given
displacement corresponds to any particular Asher-McDade grade -- no clinical
measurement in this project relates the two, and the cleft cohort records no
landmark displacements to calibrate against. Magnitudes are therefore **an
ordered series, not a graded one**: larger means more deformed, and that is all
it means. Any downstream use as a label is a further assumption to declare.

----------------------------------------------------------------------------
DESIGN CHOICES, AND WHY
----------------------------------------------------------------------------
**Unilateral, with the side recorded and balanced.** Unilateral cleft is the
clinical case, and laterality is **not recorded anywhere in the cleft cohort**
(PLAN §4.1) -- so a set deformed on one side only would hand the model a
consistent direction the clinical data cannot confirm exists. ``assign_sides``
balances left and right deterministically from the seed, and the side is recorded
per face so any arm can check the balance rather than trust it.

**The anchor ring is what makes the deformation local.** A TPS fitted to moved
points alone warps the whole plane, so the brow and the frame would drift and the
result would be a differently-framed face rather than an asymmetric one. Anchors
on the crop boundary and on the contralateral landmarks hold everything else
still. ``deformation_report`` measures the residual motion at the anchors and at
the untouched side, so "local" is a number rather than an intention.

**The midline stays anchored, and the cost is stated [DECIDED 2026-07-30].** The
mirror-difference instrument measures asymmetry about a **fixed axis**. A
synthetic face with a deviated philtrum moves that axis relative to the anatomy,
so the instrument would measure **displacement plus shape** instead of shape --
the confound that made the mirrored upper-lip index uninterpretable. The cost is
that the deformation cannot touch the philtrum, the top-ranked region in the
relevance diagnostic, so **this arm is not predicted to succeed**; see
``LIMITATIONS["the_arm_is_not_predicted_to_succeed"]``. The unanchored variant is
built only if TSTR underperforms and the cause needs narrowing.

**Displacement directions come from the anatomy, not from a rotation.** In a
unilateral cleft the alar base is displaced laterally and the philtrum and
cupid's bow peak are pulled up and toward the cleft side, with the commissure
comparatively stable -- so the four targets get four different direction vectors,
scaled by the one magnitude parameter. That is the "one rule" of the decision:
one magnitude, one fixed direction pattern.

**Applied to the CROP CONTENT, before staging** -- the same frame the arrival
mask uses (``masked.apply_arrival_mask``). Warping after staging would deform the
white padding along with the face and put the deformation in a different frame
than the mask, which is the class of defect ``masked.G2_WHITE_DEFECT`` records.

**Asymmetry must survive G2**, and it is measured two ways because the obvious
one has a floor. ``trapezium.unwarp``'s row stretch is left-right symmetric, so
it should survive -- but PLAN §4.4 flags "should" as the word that has been wrong
every previous time here, and ``asymmetry_is_preserved`` gates only the stretch,
on arbitrary offsets, not this deformation.

* ``landmark_asymmetry_survives`` -- **primary, and exact.** Maps the actual
  synthesised displacements through ``unwarp_x`` analytically. No resampling, so
  no floor.
* ``synthesised_asymmetry_survives`` -- the pixel companion. **[MEASURED
  2026-07-30] Its floor is larger than a small deformation's signal**: the frozen
  path's own nearest-neighbour resampling leaves a mirror difference of ~0.0042
  on an exactly symmetric input, while a magnitude-0.06 deformation contributes
  ~0.0011. The floor cancels in the before-minus-after difference, so the number
  is still informative, but it must be read with ``baseline_dominates`` and never
  as an absolute. On a *real* face, whose texture is mirror-asymmetric anyway,
  the pixel index is ~0.1 and the deformation moves it by ~1e-5 -- it cannot see
  the warp at all. Hence the landmark measurement is the primary one.
"""

from __future__ import annotations

import numpy as np

from . import landmarks as L

#: Displacement targets, per PLAN §4.11 Q-b. The philtral column has no landmark
#: of its own in the 86-point set (``L.PHILTRAL_COLUMN_ENDPOINTS``); it is
#: represented by the nostril sill, which is the end of the column the cleft
#: displaces, with the cupid's bow peak as the other end and already a target.
TARGETS = ("alar_base", "nostril_sill", "cupids_bow_peak", "commissure")

#: **[DECIDED 2026-07-30] The direction pattern**, as (dx, dy) unit vectors in
#: crop-content coordinates, for a deformation on the IMAGE-LEFT side. ``dx`` is
#: positive toward image-right, ``dy`` positive downward. Mirrored for the right
#: side by negating ``dx``.
#:
#: **[LITERATURE]** The pattern follows the described morphology of unilateral
#: cleft lip and palate: the alar base on the cleft side is displaced **laterally
#: and inferiorly** (flattened, splayed nostril); the philtral column and
#: cupid's-bow peak on that side are pulled **superiorly and laterally** toward
#: the cleft, which is what shortens the lip and rotates the bow; the commissure
#: is comparatively **stable**, so it takes a small share and exists mainly to
#: keep the deformation from terminating abruptly mid-lip.
#:
#: **[REASONED] The relative weights are not measured.** They are a stated shape
#: for the one-rule deformation, and no clinical measurement here fixes them --
#: which is the same standing as the magnitude mapping in ``LIMITATIONS``. They
#: are constants rather than parameters deliberately: a per-target sweep would
#: turn one rule into four and make the arm a search rather than a test.
DIRECTIONS = {
    "alar_base": (-1.0, 0.35),
    "nostril_sill": (-0.5, -0.45),
    "cupids_bow_peak": (-0.6, -1.0),
    "commissure": (-0.25, -0.15),
}

#: Per-target share of ``magnitude``. Alar base and bow peak carry the
#: deformation; the sill and commissure taper it. [REASONED], see DIRECTIONS.
WEIGHTS = {
    "alar_base": 1.0,
    "nostril_sill": 0.7,
    "cupids_bow_peak": 0.8,
    "commissure": 0.35,
}

#: How many anchor points around the crop boundary. Enough that the TPS is
#: pinned along each edge rather than only at the corners.
N_BOUNDARY_ANCHORS = 4

#: **[MEASURED 2026-07-30] Where the deformation stops being a deformation.**
#:
#: A TPS pinned at nearby points cannot move a target far without creasing. The
#: ratio ``displacement / distance to nearest anchor`` scales linearly at about
#: 13.2x the magnitude, and the binding target is always the **cupid's bow
#: peak**: its neighbours are the midline dip about 0.09 crop widths away and the
#: nostril sill above it, both anchored.
#:
#: | magnitude | max ratio |
#: |---|---|
#: | 0.03 | 0.40 |
#: | 0.04 | 0.53 |
#: | 0.06 | 0.79 |
#: | 0.09 | **1.19** |
#:
#: At 0.09 the peak is displaced **further than its nearest anchor is away**, and
#: the vermillion folds into a chevron -- clearly visible on the sheet, and
#: confirmed to be the FIELD rather than the sampling because the shape is
#: identical under bilinear interpolation. 0.5 is the working ceiling, giving a
#: usable magnitude range up to about 0.038, and the default series in the schema
#: stays inside it.
#:
#: **The bound is a property of the ANCHOR SET, not of the method**, and that
#: makes it a design question rather than a constant to live with: the midline
#: landmarks (subnasale, bow dip) are anchored here so the mirror-difference
#: measurement keeps a stable midline, and anchoring the dip 0.09 from the peak
#: is exactly what limits the peak's travel. In a real unilateral cleft the
#: philtrum IS deviated toward the cleft. **Unanchoring the midline would widen
#: the usable range and change what the instrument measures**, so it is recorded
#: as an open decision rather than taken here.
MAX_DISPLACEMENT_TO_ANCHOR_RATIO = 0.5

#: TPS regularisation. Zero would interpolate the displacements exactly; a small
#: positive value keeps the system well-conditioned when two targets land close
#: together on a narrow face, at the cost of a fraction of a pixel at the
#: targets. ``deformation_report`` measures the realised displacement so the cost
#: is visible rather than assumed.
SMOOTHING = 1e-8

LIMITATIONS = {
    "no_scar": (
        "DEFORMATION ONLY: this synthesis changes shape and leaves texture "
        "intact, so it produces no surgical scar. A repaired unilateral cleft "
        "carries one, and the Phase 4 segmentation runs found Otsu picking it up "
        "on several patients -- on a post-repair cleft much of what the "
        "aesthetic grade is about. A model pretrained on this sees asymmetric "
        "shape and no scar. The gap is STATED, not closed: adding a scar would "
        "mean inventing appearance this project has no data to specify."
    ),
    "magnitude_to_grade_is_an_assumption": (
        "magnitude is a displacement in units of crop width. NOTHING HERE "
        "ESTABLISHES that a given displacement corresponds to any particular "
        "Asher-McDade grade -- no measurement in this project relates the two, "
        "and the cleft cohort records no landmark displacements to calibrate "
        "against. Magnitudes are an ORDERED series, not a graded one: larger "
        "means more deformed and that is all it means. Using magnitude as a "
        "label is a further assumption to declare where it is used."
    ),
    "direction_weights_are_reasoned": (
        "the per-target direction vectors and weights (DIRECTIONS, WEIGHTS) "
        "follow the described morphology of unilateral cleft but are NOT "
        "measured from data. They are constants, not swept parameters, so the "
        "arm tests one stated rule rather than searching for a flattering one."
    ),
    "laterality_is_unrecorded_in_the_cleft_cohort": (
        "cleft laterality is not recorded anywhere (PLAN 4.1), so sides are "
        "BALANCED rather than matched to a clinical distribution, and the side "
        "is recorded per face so balance is checkable rather than assumed."
    ),
    "midline_is_anchored": (
        "[DECIDED 2026-07-30] **The subnasale and the cupid's bow dip are TPS "
        "anchors, so the synthesis cannot deviate the philtrum** -- and a real "
        "unilateral cleft does deviate it. This is a deliberate trade, not an "
        "oversight. The mirror-difference instrument measures asymmetry about a "
        "FIXED AXIS; a synthetic face with a moved philtrum moves that axis "
        "relative to the anatomy, so the index would measure DISPLACEMENT PLUS "
        "SHAPE rather than shape alone -- the confound that made the mirrored "
        "upper-lip index uninterpretable (PLAN Part 5 3.2). "
        "It has two measured consequences, recorded separately because they are "
        "separate: it caps the usable magnitude at ~0.038 "
        "(magnitude_range_is_bounded_by_anchor_spacing), and it excludes the "
        "highest-ranked region (the_arm_is_not_predicted_to_succeed). "
        "The unanchored variant is built ONLY if the first TSTR arm "
        "underperforms and the cause needs narrowing."
    ),
    "the_arm_is_not_predicted_to_succeed": (
        "[DECIDED 2026-07-30] **THIS ARM IS NOT PREDICTED TO SUCCEED, and the "
        "reason is specific rather than general: it cannot deform the philtrum, "
        "which the univariate relevance diagnostic ranks FIRST of 22 at "
        "|Spearman| 0.151.** The philtrum is anchored because the "
        "mirror-difference instrument measures asymmetry about a FIXED AXIS -- so "
        "if synthetic faces carry a deviated philtrum, the axis moves relative to "
        "the anatomy and the instrument measures DISPLACEMENT PLUS SHAPE rather "
        "than shape alone. That is the same confound that made the mirrored "
        "upper-lip index uninterpretable (PLAN Part 5 3.2): head tilt, crop "
        "centring and lateral displacement entered the index and none was "
        "separable from it afterwards. "
        "Unanchoring would buy realism at the cost of a measurement whose "
        "components cannot be separated, so the midline STAYS ANCHORED for the "
        "first TSTR arm; the unanchored variant is built only if TSTR "
        "underperforms and the cause needs narrowing. "
        "Written down BEFORE the arm runs, in the pattern PLAN Part 5 3.2 "
        "established: a modest or null result is the expected outcome, and "
        "recording that in advance is what stops it being read afterwards as a "
        "failure of the synthesis method."
    ),
    # [RECKONED 2026-08-30 -- the entry above is preserved byte-for-byte;
    # this note sits beside it, per the standing correction pattern. The
    # message naming it called it ARM_PREDICTION; its actual symbol is
    # this LIMITATIONS entry.]
    "prediction_reckoned_2026_08_30": (
        "**the arm ran 2026-08-30 and the outcome contradicts the "
        "prediction in substance**: the philtrum-anchored, scar-free "
        "synthesis transferred to PCC 0.2334 on the real 237 -- ~93% of "
        "the probe's 0.2520, statistically indistinguishable (0/5 "
        "intervals exclude zero; ledger p17-a-vs-probe). Worded per "
        "discipline: PARITY, NOT SUCCESS -- the point estimate runs "
        "slightly against A and the cohort cannot resolve it. One "
        "honesty note: 'not predicted to succeed' was never given a "
        "numeric definition, so the contradiction is SUBSTANTIVE rather "
        "than formal -- and the prediction's own pre-commitment is why "
        "this reckoning exists: 'a positive would have contradicted a "
        "prediction this module recorded in advance... knowable before "
        "the run' (PARKED). **The ruling on record: the old "
        "prediction was wrong.** What made it wrong is not measured: "
        "the magnitude-to-grade assumption is the named candidate for "
        "why A works, [REASONED] only (phase17.PHASE_17_CLOSING)"
    ),
    "targets_are_misaligned_with_measured_regional_relevance": (
        "[MEASURED 2026-07-30, the univariate relevance diagnostic] THE "
        "SYNTHESIS ANCHORS THE MOST RELEVANT REGION AND DEFORMS TWO OF THE "
        "LEAST. Of the 22 mirror-difference features against the mean label, "
        "`philtrum` ranks 1st (|Spearman| 0.151) and `labial_tubercle` 2nd "
        "(0.138) -- and the philtrum is ANCHORED here, because the midline "
        "landmarks are pinned so the mirror-difference measurement keeps a "
        "stable midline. So the deformation cannot touch the area that "
        "correlates best with the grade. Of its own four targets, "
        "`philtral_column` ranks 3rd (0.122) but `alar_base` ranks 19th and "
        "`commissure` 20th, both essentially zero. "
        "NOT DECISIVE, and the reason is R2: these are DIFFERENT QUANTITIES. "
        "The TPS displaces a LANDMARK; the index measures PIXEL ASYMMETRY IN A "
        "BOX. A displaced landmark need not raise the mirror difference inside a "
        "box that travels with the anatomy fractions, and a region whose pixel "
        "asymmetry does not track the grade may still be one whose displacement "
        "does. Nothing here measures the second. "
        "ALSO NOT TO BE OVERREAD IN THE OTHER DIRECTION: nothing in the "
        "diagnostic clears the Bonferroni bar (0.199), so the ranking informs a "
        "design decision and is not a set of claims. "
        "A DECISION IS NEEDED BEFORE PHASE 7 and the deformation is NOT being "
        "changed now -- changing targets to chase a ranking that clears no "
        "corrected threshold would be fitting the instrument to a diagnostic. "
        "It is recorded beside the other limitations so the arm is read with it."
    ),
    "magnitude_range_is_bounded_by_anchor_spacing": (
        "[MEASURED 2026-07-30] The usable magnitude range has a measured upper "
        "bound of about 0.038, and it comes from the ANCHOR SET rather than from "
        "the method. displacement/nearest-anchor scales at 13.2x the magnitude "
        "and the cupid's bow peak binds first: at 0.09 it is displaced further "
        "than its nearest anchor is away and the vermillion folds into a chevron "
        "(identical under bilinear sampling, so it is the field, not the "
        "resampling). The midline landmarks are anchored so the "
        "mirror-difference measurement keeps a stable midline, and anchoring the "
        "bow dip ~0.09 crop widths from the peak is precisely what limits the "
        "peak's travel -- while in a real unilateral cleft the philtrum IS "
        "deviated toward the cleft. Unanchoring the midline would widen the "
        "range and CHANGE WHAT THE INSTRUMENT MEASURES, so it is recorded as an "
        "OPEN DECISION and not taken."
    ),
}


#: **[DECIDED 2026-07-30] PARKED: built and verified, never run at scale.**
#:
#: Stage F is struck from the arm list (PLAN Part 6). This module works — the
#: deformation is local, unilateral, survives G2, and the resume is
#: byte-identical — and it is **not going to produce a result**, because the four
#: entries in ``LIMITATIONS`` meant **no interpretable outcome was available in
#: either direction**:
#:
#: * a **null** would have been consistent with all four at once and would not
#:   have distinguished between them;
#: * a **positive** would have contradicted a prediction this module recorded in
#:   advance, and would have needed explaining before it could be believed.
#:
#: **That was knowable before the run, which is the whole reason the prediction
#: was written down first.** Writing a null down in advance is normally about
#: protecting the reading of a result; here it paid off one step earlier, as a
#: decision not to spend the time at all.
#:
#: **Kept, not deleted, and its tests keep running** -- the same posture as the
#: parked segmentation module (PLAN Part 5 §3.2). A parked module whose suite is
#: skipped quietly stops working, and then the chapter it evidences cannot be
#: re-derived when someone asks. **Do not extend it**; reviving Stage F is a
#: deliberate decision with a consumer named first, and it means resolving
#: ``CLUSTER_PERFORMANCE_UNRESOLVED`` before anything else.
PARKED = {
    "status": "BUILT AND VERIFIED, NOT RUN",
    "decided": "2026-07-30",
    "arm": "Stage F, TSTR vs CV-on-237 (Rosero replication)",
    # The line above is preserved as written; the word it uses is
    # withdrawn below.
    "arm_amended_2026_08_29": (
        "**Amended 2026-08-29: literature verification against the "
        "primary source showed the parked design is NOT Rosero's** -- "
        "Rosero warps by piecewise affine over landmark triangulation "
        "and trains on binary symmetric/asymmetric pairs with a "
        "contrastive loss, mapping distance to grades only at readout; "
        "this arm warps by TPS and trains on magnitude-ordered labels. "
        "**The word 'replication' is withdrawn**; the arm is a TSTR "
        "design in the Rosero family that TESTS the magnitude-to-grade "
        "assumption Rosero's construction avoids. Note: that assumption "
        "was already flagged at original adoption -- "
        "LIMITATIONS['magnitude_to_grade_is_an_assumption'], whose "
        "recorded wording is that 'magnitudes are an ORDERED series, "
        "not a graded one: larger means more deformed and that is all "
        "it means. Using magnitude as a label is a further assumption "
        "to declare where it is used' -- so this verification CONFIRMS "
        "a recorded worry rather than discovering a new one. (The "
        "adoption-time record makes no comparison to Rosero; the "
        "'stronger than Rosero' framing is the 2026-08-29 "
        "verification's addition, dated here.) Phase 17's rulings "
        "consume this amendment: phase17.PHASE_17_RULINGS"
    ),
    "reason": (
        "no interpretable outcome was available in either direction. A null "
        "would have been consistent with all four recorded limitations and could "
        "not have distinguished between them; a positive would have contradicted "
        "a prediction recorded in advance. An arm whose two outcomes are both "
        "uninterpretable does not earn cluster time."
    ),
    "the_four_limitations": (
        "no_scar; magnitude_to_grade_is_an_assumption; midline_is_anchored; "
        "the_arm_is_not_predicted_to_succeed (cannot deform the philtrum, ranked "
        "first of 22 at |Spearman| 0.151)"
    ),
    "what_is_verified": (
        "deformation local and unilateral (measured in the pixels), asymmetry "
        "survives G2 at 1.11-1.15, sides exactly balanced, resume "
        "byte-identical over 232.9 MiB across two interruptions"
    ),
    "tests_keep_running": (
        "deliberately. A parked module whose suite is skipped quietly stops "
        "working, and the write-up chapter it evidences then cannot be "
        "re-derived."
    ),
    "to_revive": (
        "resolve CLUSTER_PERFORMANCE_UNRESOLVED first, then name a consumer. "
        "Nothing in Phase 6 depends on this module."
    ),
}

#: **[KNOWN UNRESOLVED 2026-07-30] The full build is pathologically slow on the
#: cluster and the cause is not established.** Recorded as an open performance
#: issue rather than a blocker: the only arm that consumes the output is parked.
#:
#: ``p5-synth-1`` managed fewer than 250 faces in 233 minutes while sustaining
#: ~25 cores -- about **1,400 CPU-seconds per face against 0.30 s/face on
#: identical code locally**, roughly 185x wall and 4,600x CPU. Memory flat at
#: 789 MB, so nothing accumulates.
#:
#: **What profiling settled.** The build is **flat in n**: at 250 faces locally
#: the last block of 25 costs 0.96x the first, so nothing scales with faces
#: already processed and the cluster's throughput is equally consistent with a
#: constant per-face cost. The hot path is **single-threaded elementwise numpy
#: over a 33 MB intermediate** in ``apply_tps`` -- broadcast 28%, reduce 34%,
#: kernel 35% -- and NOT linear algebra: ``np.linalg.solve`` is 7% of the build
#: and the GEMMs are 2.9% of ``apply_tps``. Memory-bandwidth-bound work is
#: exactly what degrades on a node at load 406.
#:
#: **What it did not settle.** 99.98% of the cluster's CPU is overhead, which is
#: the signature of spin-waiting rather than slow work, and 12 uncontended cores
#: could not reproduce it. Leading hypothesis: BLAS thread-pool spin, ~33 BLAS
#: calls per face waking a 64-thread pool. Untried decisive tests, both cheap and
#: neither needing SYS_PTRACE: run with ``OPENBLAS_NUM_THREADS=1``, and read
#: ``utime`` against ``stime`` in ``/proc/<pid>/stat``, which separates userspace
#: spinning from kernel time.
#:
#: **A correction that goes with it.** An earlier local figure of 0.975 s/face
#: was measured while the test suite ran in the background; clean re-measurement
#: gives **0.301 s/face**, and every SCUT image is 350x350 so there is no
#: per-face size variation either. The "~90 minute build" projection derived from
#: the contaminated number and should have read ~28 minutes. It made the cluster
#: gap look smaller than it is.
CLUSTER_PERFORMANCE_UNRESOLVED = {
    # [CLOSED 2026-08-29, as NOT-GATING -- everything below is
    # preserved as the original observation's history.]
    "closed_2026_08_29": (
        "closed as not-gating: the pathology was NOT reproduced at "
        "re-measurement (0.2-1.2 s/face vs the 56 s/face recorded "
        "below, same pinned image, clean tree at 377d769; "
        "phase17.COMPUTE_GATE_MEASUREMENT['cluster_performance_closed']"
        "). The futex hypothesis was never tested -- the disease did "
        "not show -- and the candidate explanations (node, contention "
        "at the time, code path differences) are unadjudicated"
    ),
    "status": "OPEN, not a blocker",
    "measured": "2026-07-30, p5-synth-1",
    "cluster": {
        "faces": "<250 in 233 min",
        "cpu_seconds_per_face": 1400,
        "cores_sustained": 25,
        "node_load": 406,
        "node_cores": 64,
        "rss_mb": 789,
    },
    "local": {"seconds_per_face": 0.301, "flat_in_n": True,
              "last_block_over_first": 0.96},
    "hot_path": (
        "elementwise numpy over a 33 MB intermediate in apply_tps "
        "(broadcast 28%, reduce 34%, kernel 35%). np.linalg.solve is 7% of the "
        "build; the GEMMs are 2.9% of apply_tps. Memory-bandwidth bound, not "
        "compute bound."
    ),
    "not_explained": (
        "99.98% of cluster CPU is overhead -- the signature of spin-waiting, not "
        "slow work -- and it did not reproduce on 12 uncontended cores."
    ),
    "leading_hypothesis": "BLAS thread-pool spin: ~33 BLAS calls/face x 64 threads",
    "untried_decisive_tests": (
        "OPENBLAS_NUM_THREADS=1 for a 250-face run; and utime vs stime from "
        "/proc/<pid>/stat, which separates userspace spinning from kernel time"
    ),
    "why_not_a_blocker": "the only arm consuming this output is PARKED",
}


class SynthesisError(RuntimeError):
    """The deformation could not be built."""


# --------------------------------------------------------------------------
# sides
# --------------------------------------------------------------------------


def assign_sides(n: int, seed: int) -> np.ndarray:
    """One side per face, balanced and reproducible from the seed.

    Exactly ``n // 2`` faces get "right" (and the odd one out goes left), then
    the assignment is permuted -- so the balance is exact by construction rather
    than approximate in expectation, which a per-face coin flip would be.
    """
    if n < 0:
        raise SynthesisError(f"n must be non-negative, got {n}")
    sides = np.array(["left"] * (n - n // 2) + ["right"] * (n // 2), dtype=object)
    np.random.default_rng(seed).shuffle(sides)
    return sides


def side_balance(sides) -> dict:
    """The realised balance. Reported so it is checkable, never assumed."""
    array = np.asarray(list(sides), dtype=object)
    left = int((array == "left").sum())
    right = int((array == "right").sum())
    return {
        "n": int(array.size),
        "n_left": left,
        "n_right": right,
        "difference": abs(left - right),
        "note": (
            "cleft laterality is unrecorded (PLAN 4.1), so the set is balanced "
            "rather than matched to a clinical distribution"
        ),
    }


# --------------------------------------------------------------------------
# the thin-plate spline
# --------------------------------------------------------------------------


def _kernel(r2: np.ndarray) -> np.ndarray:
    """The 2-D TPS radial basis, ``r^2 log r``, written on ``r^2``.

    ``r^2 log r == 0.5 * r2 * log(r2)``, and the ``r2 == 0`` diagonal is set to
    zero explicitly: the limit is 0 but ``log(0)`` is not, and letting numpy
    produce a nan there would give a singular system with no clear error.
    """
    out = np.zeros_like(r2)
    positive = r2 > 0
    out[positive] = 0.5 * r2[positive] * np.log(r2[positive])
    return out


def fit_tps(source: np.ndarray, target: np.ndarray, smoothing: float = SMOOTHING):
    """Fit a TPS mapping ``source`` control points onto ``target``.

    Returns the coefficient arrays; ``apply_tps`` consumes them. Solved once and
    reused for every pixel, which is why the two are separate.
    """
    source = np.asarray(source, dtype=float)
    target = np.asarray(target, dtype=float)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 2:
        raise SynthesisError(
            f"source {source.shape} and target {target.shape} must both be (N, 2)"
        )
    n = source.shape[0]
    if n < 3:
        raise SynthesisError(f"a TPS needs at least 3 control points, got {n}")

    diff = source[:, None, :] - source[None, :, :]
    kernel = _kernel((diff**2).sum(axis=2))
    kernel[np.diag_indices(n)] += smoothing * n

    affine = np.column_stack([np.ones(n), source])
    system = np.zeros((n + 3, n + 3))
    system[:n, :n] = kernel
    system[:n, n:] = affine
    system[n:, :n] = affine.T

    rhs = np.zeros((n + 3, 2))
    rhs[:n] = target
    try:
        solution = np.linalg.solve(system, rhs)
    except np.linalg.LinAlgError as exc:
        raise SynthesisError(
            f"the TPS system is singular: {exc}. Control points are probably "
            "collinear or duplicated."
        ) from None
    return source, solution[:n], solution[n:]


def apply_tps(fitted, points: np.ndarray) -> np.ndarray:
    """Map ``points`` (N, 2) through a fitted TPS."""
    control, weights, affine = fitted
    points = np.asarray(points, dtype=float)
    diff = points[:, None, :] - control[None, :, :]
    kernel = _kernel((diff**2).sum(axis=2))
    homogeneous = np.column_stack([np.ones(points.shape[0]), points])
    return homogeneous @ affine + kernel @ weights


# --------------------------------------------------------------------------
# control points
# --------------------------------------------------------------------------


def target_indices(side: str) -> dict[str, int]:
    """Which landmark index each target names, on the given side."""
    if side not in ("left", "right"):
        raise SynthesisError(f"unknown side {side!r}; expected 'left' or 'right'")
    position = 0 if side == "left" else 1
    return {
        "alar_base": L.ALAR_BASE[position],
        "nostril_sill": L.NOSTRIL_SILL[position],
        "cupids_bow_peak": L.CUPIDS_BOW_PEAK[position],
        "commissure": L.MOUTH_CORNERS[position],
    }


def control_points(
    points: np.ndarray,
    box: tuple[float, float, float, float],
    side: str,
    magnitude: float,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """(source, target, detail) control points in SOURCE IMAGE pixels.

    Displacements are expressed in crop widths and converted to pixels here, so
    the same ``magnitude`` is the same deformation relative to the face on every
    image whatever its resolution -- the same discipline as the SymNose
    relaxation being a fraction of each image's own class gap rather than a fixed
    pixel count (PLAN §3.2).
    """
    if magnitude < 0:
        raise SynthesisError(f"magnitude must be non-negative, got {magnitude}")
    array = np.asarray(points, dtype=float)
    x, y, width, height = box
    flip = 1.0 if side == "left" else -1.0

    indices = target_indices(side)
    moved_source, moved_target, detail = [], [], {}
    for name, index in indices.items():
        dx, dy = DIRECTIONS[name]
        shift = np.array([flip * dx, dy]) * WEIGHTS[name] * magnitude * width
        moved_source.append(array[index])
        moved_target.append(array[index] + shift)
        detail[name] = {
            "index": int(index),
            "shift_px": [round(float(v), 4) for v in shift],
            "shift_frac_width": [
                round(float(flip * dx * WEIGHTS[name] * magnitude), 6),
                round(float(dy * WEIGHTS[name] * magnitude), 6),
            ],
        }

    # ---- anchors: EVERY landmark that is not a target, plus the boundary ----
    #
    # **[MEASURED 2026-07-30] A sparse anchor set does not confine a TPS, and
    # checking only the anchors cannot reveal that.** An earlier version anchored
    # the crop boundary, the four contralateral targets and the two midline
    # landmarks -- ten points -- and reported "unintended motion 1e-10", which was
    # true and meaningless: every point it measured was an anchor, so it verified
    # that anchors are anchored, which the fit guarantees by construction. The
    # thin-plate kernel is globally supported and decays slowly, so everything
    # NOT anchored moved: measured **48 of 86 landmarks displaced, including both
    # eyes, three brow points and eleven face-outline points**, and pixel change
    # spanning the whole 400x400 source rather than the 126x170 crop.
    #
    # That is failure mode 2 in PLAN R7's tally -- a criterion satisfiable by
    # construction -- and the fix is structural, not a tighter tolerance: anchor
    # every landmark except the four being displaced. The deformation is then
    # confined to their neighbourhood because the surrounding anatomy pins the
    # field, and `deformation_report` measures locality in the PIXELS, where a
    # by-construction pass is not available.
    moved_indices = set(indices.values())
    anchor_indices = [
        i for i in range(array.shape[0]) if i not in moved_indices
    ]
    anchors = [array[i].tolist() for i in anchor_indices]

    for fraction in np.linspace(0.0, 1.0, N_BOUNDARY_ANCHORS):
        anchors += [
            [x + fraction * width, y],
            [x + fraction * width, y + height],
            [x, y + fraction * height],
            [x + width, y + fraction * height],
        ]

    anchor_array = np.asarray(anchors, dtype=float)
    source = np.vstack([np.asarray(moved_source, dtype=float), anchor_array])
    target = np.vstack([np.asarray(moved_target, dtype=float), anchor_array])
    detail["n_anchors"] = int(anchor_array.shape[0])
    detail["n_landmark_anchors"] = len(anchor_indices)
    detail["moved_indices"] = sorted(int(i) for i in moved_indices)
    return source, target, detail


# --------------------------------------------------------------------------
# warping
# --------------------------------------------------------------------------


def warp_content(
    image: np.ndarray,
    points: np.ndarray,
    box: tuple[float, float, float, float],
    side: str,
    magnitude: float,
) -> tuple[np.ndarray, dict]:
    """Deform the SOURCE IMAGE inside the crop box. Returns (warped, record).

    Takes the **full source image** in source pixel coordinates -- the same frame
    the landmarks and ``box`` are in -- and rewrites only the pixels inside the
    box, copying the rest. Two reasons, both learned the hard way:

    * **One coordinate frame throughout.** An earlier version was named for the
      cropped content and offset its sampling grid by the box origin, while every
      call site passed the full image: the deformation was applied about
      ``(x0, y0)`` pixels away from the anatomy it was fitted to. The pixel change
      centroid sat 31 x 47 pixels off the landmark centroid, which is the sort of
      geometric slip that produces a plausible-looking sheet.
    * **Locality in the pixels by construction of the loop bound**, not by hoping
      the spline decays. Outside the box nothing is read downstream anyway, since
      staging crops to the box.

    Sampling is nearest-neighbour by **inverse** mapping -- for each output pixel,
    find where it came from -- because forward mapping leaves holes. Nearest
    neighbour matches ``staging.resize_nearest``: dependency-free, exactly
    reproducible, no library version to disagree about between laptop and cluster.
    """
    array = np.asarray(image)
    if array.ndim not in (2, 3):
        raise SynthesisError(f"expected a 2-D or 3-D image, got {array.shape}")
    source, target, detail = control_points(points, box, side, magnitude)

    if magnitude == 0:
        # An explicit identity rather than a spline that should be one. The
        # zero-magnitude control is the arm's own reference, and "should be
        # identical" is not a thing to leave to floating point.
        return array.copy(), {
            "side": side,
            "magnitude": 0.0,
            "identity": True,
            "targets": detail,
        }

    image_h, image_w = array.shape[:2]
    x, y, box_w, box_h = box
    left = max(int(np.floor(x)), 0)
    top = max(int(np.floor(y)), 0)
    right = min(int(np.ceil(x + box_w)), image_w)
    bottom = min(int(np.ceil(y + box_h)), image_h)
    if right - left < 2 or bottom - top < 2:
        raise SynthesisError(
            f"the crop box {box} leaves nothing inside a {image_w}x{image_h} image"
        )

    # Inverse: fit target -> source, then every output pixel reads its origin.
    # Source pixel coordinates on both sides -- pixel (row, col) has centre
    # (col + 0.5, row + 0.5), the same convention trapezium.mask uses.
    inverse = fit_tps(target, source)
    rows, columns = np.mgrid[top:bottom, left:right]
    grid = np.column_stack([columns.ravel() + 0.5, rows.ravel() + 0.5])
    origins = apply_tps(inverse, grid)
    shape = (bottom - top, right - left)
    sample_x = np.clip(
        np.floor(origins[:, 0]).astype(int), 0, image_w - 1
    ).reshape(shape)
    sample_y = np.clip(
        np.floor(origins[:, 1]).astype(int), 0, image_h - 1
    ).reshape(shape)

    warped = array.copy()
    warped[top:bottom, left:right] = array[sample_y, sample_x]

    forward = fit_tps(source, target)
    report = deformation_report(forward, points, box, side, magnitude)
    report["pixel_change"] = pixel_change_extent(array, warped, box)
    return warped, {
        "side": side,
        "magnitude": round(float(magnitude), 6),
        "identity": False,
        "targets": detail,
        "deformation": report,
    }


def pixel_change_extent(
    plain: np.ndarray, warped: np.ndarray, box, threshold: int = 10
) -> dict:
    """Where the pixels actually changed, in crop-box coordinates.

    **The locality check that cannot pass by construction.** Every landmark
    except the four targets is an anchor, so a landmark-based locality check now
    only re-measures anchors -- true by the fit, evidence of nothing. This looks
    at the image: which pixels moved, expressed as a bounding box normalised to
    the crop, so "the deformation stays in the nasolabial region and does not
    reach the eyes" is a number the sheet review can be checked against.

    ``y_min`` is the one to read. The eyes sit near 0.23 of the crop and the alar
    bases near 0.60, so a deformation of the nose and lip should not produce
    change much above ~0.35.
    """
    a = np.asarray(plain, dtype=np.int32)
    b = np.asarray(warped, dtype=np.int32)
    difference = np.abs(a - b)
    if difference.ndim == 3:
        difference = difference.sum(axis=2)
    changed = difference > threshold

    x, y, width, height = box
    n_changed = int(changed.sum())
    if not n_changed:
        return {
            "n_changed_pixels": 0,
            "note": "no pixel changed; the deformation had no visible effect",
        }

    ys, xs = np.nonzero(changed)
    return {
        "n_changed_pixels": n_changed,
        "fraction_of_image": round(n_changed / changed.size, 8),
        # Normalised to the crop box: 0 is its left/top edge, 1 its right/bottom.
        "x_min": round(float((xs.min() - x) / width), 6),
        "x_max": round(float((xs.max() - x) / width), 6),
        "y_min": round(float((ys.min() - y) / height), 6),
        "y_max": round(float((ys.max() - y) / height), 6),
        "centroid_x": round(float((xs.mean() - x) / width), 6),
        "centroid_y": round(float((ys.mean() - y) / height), 6),
        "outside_box": bool(
            xs.min() < x - 1 or xs.max() > x + width + 1
            or ys.min() < y - 1 or ys.max() > y + height + 1
        ),
        "note": (
            "Bounding box of changed pixels, normalised to the CROP BOX. Read "
            "y_min: the eyes sit near 0.23 of the crop and the alar bases near "
            "0.60, so a nasolabial deformation should not change pixels much "
            "above ~0.35. This is measured on the image rather than on "
            "landmarks, because every landmark except the four targets is an "
            "anchor and would report zero motion by construction."
        ),
    }


def warp_landmarks(
    points: np.ndarray,
    box: tuple[float, float, float, float],
    side: str,
    magnitude: float,
) -> np.ndarray:
    """Where the landmarks land after the same deformation.

    The arm needs these: a synthesised face whose landmarks were not moved with
    it would be measured against the geometry of the face it used to be.
    """
    source, target, _ = control_points(points, box, side, magnitude)
    if magnitude == 0:
        return np.asarray(points, dtype=float).copy()
    return apply_tps(fit_tps(source, target), np.asarray(points, dtype=float))


# --------------------------------------------------------------------------
# is it local, unilateral, and does it survive G2?
# --------------------------------------------------------------------------


def deformation_report(
    fitted, points: np.ndarray, box, side: str, magnitude: float
) -> dict:
    """Measure what moved and what did not. **"Local" as a number.**

    Three things a deformation intended to be local and unilateral must show,
    each of which fails silently otherwise:

    * the **targets** actually moved by the requested amount -- the smoothing
      term means they move by slightly less, and this says how much less;
    * the **anchors** did not move -- if they did, the whole frame drifted and
      the result is a reframed face;
    * the **contralateral landmarks** did not move -- if they did, the
      deformation is not unilateral, and a mirror-difference index computed on it
      would be measuring a bilateral change.
    """
    array = np.asarray(points, dtype=float)
    width = box[2]
    indices = target_indices(side)
    moved = apply_tps(fitted, array)

    requested = {}
    realised = {}
    for name, index in indices.items():
        dx, dy = DIRECTIONS[name]
        want = np.array([(1.0 if side == "left" else -1.0) * dx, dy])
        want = want * WEIGHTS[name] * magnitude * width
        got = moved[index] - array[index]
        requested[name] = float(np.hypot(*want))
        realised[name] = float(np.hypot(*got))

    other = 1 if side == "left" else 0
    contralateral = [
        L.ALAR_BASE[other], L.NOSTRIL_SILL[other],
        L.CUPIDS_BOW_PEAK[other], L.MOUTH_CORNERS[other],
    ]
    contra_motion = [
        float(np.hypot(*(moved[i] - array[i]))) / width for i in contralateral
    ]
    midline_motion = [
        float(np.hypot(*(moved[i] - array[i]))) / width for i in L.MIDLINE_LANDMARKS
    ]

    x, y, w, h = box
    corners = np.array([[x, y], [x + w, y], [x, y + h], [x + w, y + h]])
    corner_motion = [
        float(np.hypot(*(m - c))) / width for m, c in zip(apply_tps(fitted, corners), corners)
    ]

    # Every landmark that is NOT one of the four targets. These are all anchors
    # now, so this is expected to be ~0 -- and that is exactly why it is reported
    # with the count and NOT treated as evidence of locality: it is true by
    # construction of the fit. `pixel_change` is the locality measurement that
    # can fail. This is kept only as a consistency check that the anchor set is
    # what it is meant to be: if a landmark ever appears here with real motion,
    # the anchor set lost a member.
    non_target = [i for i in range(array.shape[0]) if i not in set(indices.values())]
    non_target_motion = [
        float(np.hypot(*(moved[i] - array[i]))) / width for i in non_target
    ]

    # **Displacement against anchor spacing -- the quantity that bounds the
    # usable magnitude range [MEASURED 2026-07-30].** A TPS pinned at nearby
    # points cannot move a target far without creasing: if a target is displaced
    # by more than the distance to its nearest anchor, the field folds and the
    # result is a spike rather than anatomy. It is visible on the sheet at
    # magnitude 0.09, where the cupid's bow peak rises about 8.6px while the
    # anchored midline dip sits about 11px away, pulling the vermillion into a
    # chevron. Confirmed as the FIELD and not the resampling: the shape is
    # identical under bilinear sampling.
    spacing = {}
    for name, index in indices.items():
        others = [i for i in range(array.shape[0]) if i != index]
        nearest = float(np.min(np.hypot(*(array[others] - array[index]).T)))
        displacement = float(np.hypot(*(moved[index] - array[index])))
        spacing[name] = {
            "nearest_anchor_px": round(nearest, 4),
            "displacement_px": round(displacement, 4),
            "ratio": round(displacement / nearest, 6) if nearest > 0 else None,
        }

    return {
        "displacement_to_anchor_spacing": spacing,
        "max_displacement_to_anchor_ratio": max(
            v["ratio"] for v in spacing.values() if v["ratio"] is not None
        ),
        "displacement_to_anchor_note": (
            "A TPS pinned at nearby points cannot move a target far without "
            "creasing. Ratio above ~0.5 folds the field into a visible spike "
            "rather than deforming smoothly -- MEASURED on the sheet at "
            "magnitude 0.09, where the cupid's bow peak rises ~8.6px against an "
            "~11px gap to the anchored midline dip and the vermillion becomes a "
            "chevron. Confirmed to be the FIELD, not the resampling: identical "
            "under bilinear sampling. This bounds the usable magnitude range "
            "from above, and the bound is a property of the anchor set."
        ),
        "n_non_target_landmarks": len(non_target),
        "max_non_target_motion_frac_width": max(non_target_motion),
        "non_target_motion_is_zero_by_construction": (
            "every landmark except the four targets is a TPS anchor, so this is "
            "guaranteed small by the fit and is NOT evidence that the "
            "deformation is local. An earlier version measured only anchors and "
            "reported 1e-10 while 48 of 86 landmarks were in fact moving, "
            "because the anchor set was ten points. Read pixel_change instead."
        ),
        "requested_px": {k: round(v, 4) for k, v in requested.items()},
        "realised_px": {k: round(v, 4) for k, v in realised.items()},
        "realised_fraction_of_requested": {
            k: round(realised[k] / requested[k], 6) if requested[k] > 0 else 1.0
            for k in requested
        },
        # These three should be ~0. Reported rather than asserted here so the
        # magnitudes are visible in metrics.json; the tests assert the bounds.
        #
        # **Kept at full precision deliberately.** Rounded to 8 places these all
        # print as a flat 0.0, and an exact zero is indistinguishable from a
        # quantity that was never computed -- the shape of failure mode 5 in
        # PLAN R7's tally. At full precision a genuine zero reads as ~1e-13 and a
        # missing measurement still reads as 0.0, so the two stay tellable apart.
        # ``realised_px`` above is the companion evidence that the same machinery
        # reports real motion where motion was asked for.
        "contralateral_motion_frac_width": contra_motion,
        "midline_motion_frac_width": midline_motion,
        "corner_motion_frac_width": corner_motion,
        "max_unintended_motion_frac_width": max(
            contra_motion + midline_motion + corner_motion
        ),
    }


def assert_index_describes_arrays(
    per_face: list[dict], survival: list[dict], n_faces: int, magnitudes
) -> dict:
    """**The artifact's index must describe the arrays it sits beside.**

    Returns the counts it checked; raises ``SynthesisError`` naming the mismatch.

    **[DEFECT 2026-07-30, found by the cluster gate and not by any test here.]**
    A 200-face build published ``faces.json`` with **211 rows and 11 duplicate
    stems** while all eight arrays were byte-correct. A resume had rolled the
    arrays back but appended to the face journal instead of replacing it, so the
    reprocessed faces overwrote the right array positions and added a second row
    each.

    That is the worst shape a data defect can take here: **the pixels are right
    and the index beside them is wrong**, so nothing downstream fails loudly.
    Arrays are addressed by face position, so a longer index silently misaligns
    every consumer -- enumerating the set, indexing into the arrays, checking side
    balance all read from a record claiming 211 faces where 200 exist.

    Cheap, total, and true of every build, so it runs before publishing rather
    than being discovered by a byte comparison afterwards.
    """
    n_deformed = sum(1 for m in magnitudes if m > 0)
    stems = [entry["stem"] for entry in per_face]
    duplicates = sorted({stem for stem in stems if stems.count(stem) > 1})
    expected_survival = n_faces * n_deformed

    problems = []
    if len(per_face) != n_faces:
        problems.append(f"faces.json has {len(per_face)} rows, expected {n_faces}")
    if duplicates:
        problems.append(
            f"{len(duplicates)} duplicate stem(s): {duplicates[:5]}"
            + (" ..." if len(duplicates) > 5 else "")
        )
    if len(survival) != expected_survival:
        problems.append(
            f"survival has {len(survival)} rows, expected {expected_survival} "
            f"({n_faces} faces x {n_deformed} deformed magnitudes)"
        )

    if problems:
        raise SynthesisError(
            "the artifact index does not describe the arrays beside it:\n  "
            + "\n  ".join(problems)
            + "\nA resume that appended rather than rolled back its journal "
            "produces exactly this, and the arrays can be perfectly correct "
            "while it happens. Not publishing."
        )

    return {
        "n_faces": len(per_face),
        "n_unique_stems": len(set(stems)),
        "n_survival_rows": len(survival),
        "expected_survival_rows": expected_survival,
    }


def asymmetry_index(image: np.ndarray) -> float:
    """Mean absolute difference between an image and its mirror, normalised.

    The mirror-difference quantity Phase 4's primary geometric baseline is built
    on, reduced to one number over the whole frame -- enough to answer "did the
    asymmetry survive this operation", which is all it is used for here. It is
    **not** the Phase 4 index and must not be quoted as it: that one is 22
    per-region features (PLAN Part 5).

    **It measures a DIFFERENCE, so it is only meaningful against a baseline it
    dominates.** On a face whose own texture is mirror-asymmetric -- any real face
    -- this number is mostly texture, and a local deformation moves it by a
    rounding error. That is why the G2 survival check runs on a symmetric base
    (PLAN §4.4: *synthesise a known asymmetry*), and why
    ``synthesised_asymmetry_survives`` reports ``baseline_dominates`` rather than
    leaving a meaningless ratio to be read as a result.
    """
    array = np.asarray(image, dtype=float)
    if array.ndim == 3:
        array = array.mean(axis=2)
    return float(np.abs(array - array[:, ::-1]).mean() / 255.0)


def landmark_asymmetry_survives(
    points: np.ndarray,
    box: tuple[float, float, float, float],
    side: str,
    magnitude: float,
) -> dict:
    """**The exact survival measurement: does G2 preserve the displacement?**

    Primary over the pixel version below, because it has **no resampling floor**.
    Landmark positions are mapped analytically through ``trapezium.unwarp_x``, so
    what comes out is the deformation's own survival rather than the deformation
    plus nearest-neighbour jitter.

    Why it is not redundant with ``trapezium.asymmetry_is_preserved``: the
    analytic gate displaces arbitrary offsets at chosen heights and checks the row
    stretch is symmetric. This runs **the actual synthesised displacement
    pattern** -- four targets at their measured anatomical heights, one side only
    -- and checks the mirror-pair asymmetry it creates survives. Same operation,
    different input, and the input is the one the arm will use.

    Reported per target as the mirror-pair asymmetry it induces, before and after
    unwarping, normalised to the frame so the two are comparable.
    """
    from ..geometry.staging import Staged, content_to_output, stage
    from ..geometry.trapezium import unwarp_x

    array = np.asarray(points, dtype=float)
    x, y, width, height = box
    if width <= 0 or height <= 0:
        raise SynthesisError(f"box has a non-positive dimension: {box}")

    moved = warp_landmarks(array, box, side, magnitude)
    # A Staged carrying this crop's content box, so points map into the padded
    # square exactly as the real path places them. No image is needed.
    probe = stage(np.zeros((max(int(round(height)), 2), max(int(round(width)), 2)), dtype=np.uint8))
    staged = Staged(
        image=probe.image,
        source_size=probe.source_size,
        content_box=probe.content_box,
        scale=probe.scale,
    )
    size = staged.image.shape[0]

    def to_square(point) -> tuple[float, float]:
        fx = float(np.clip((point[0] - x) / width, 0.0, 1.0))
        fy = float(np.clip((point[1] - y) / height, 0.0, 1.0))
        px, py = content_to_output(staged, fx, fy)
        return px / size, py / size

    indices = target_indices(side)
    other = 1 if side == "left" else 0
    partner = {
        "alar_base": L.ALAR_BASE[other],
        "nostril_sill": L.NOSTRIL_SILL[other],
        "cupids_bow_peak": L.CUPIDS_BOW_PEAK[other],
        "commissure": L.MOUTH_CORNERS[other],
    }

    per_target = {}
    for name, index in indices.items():
        results = {}
        for label, source_points in (("plain", array), ("deformed", moved)):
            a = to_square(source_points[index])
            b = to_square(source_points[partner[name]])
            # **SIGNED, not absolute, and that is the whole correctness of this
            # measurement [MEASURED 2026-07-30].** With ``abs`` around the pair
            # difference, a displacement acting AGAINST the face's existing
            # asymmetry first cancels it and only then adds -- so ``introduced``
            # became a difference of near-equal quantities and the ratio of two
            # such differences was unstable: the commissure measured 0.34
            # deforming left against 1.15 deforming right, from the same rule at
            # the same magnitude, which reads as "G2 destroyed two thirds of the
            # asymmetry" and is an artefact of the absolute value. Signed
            # differences add linearly, so the displacement contributes the same
            # amount whichever way the face was already leaning.
            g1 = abs(a[0] - 0.5) - abs(b[0] - 0.5)
            # After: each row is stretched by its own factor, so both points are
            # mapped at their OWN heights -- which is precisely the property that
            # could fail and the reason a single-height check would not see it.
            ua = unwarp_x(min(max(a[0], 0.0), 1.0), a[1])
            ub = unwarp_x(min(max(b[0], 0.0), 1.0), b[1])
            g2 = abs(ua - 0.5) - abs(ub - 0.5)
            results[label] = {"g1": g1, "g2": g2}
        introduced_g1 = results["deformed"]["g1"] - results["plain"]["g1"]
        introduced_g2 = results["deformed"]["g2"] - results["plain"]["g2"]

        per_target[name] = {
            "signed_asymmetry_g1": round(results["deformed"]["g1"], 8),
            "signed_asymmetry_g2": round(results["deformed"]["g2"], 8),
            "plain_signed_asymmetry_g1": round(results["plain"]["g1"], 8),
            "introduced_g1": round(introduced_g1, 8),
            "introduced_g2": round(introduced_g2, 8),
            "survival_ratio": (
                round(introduced_g2 / introduced_g1, 6)
                if abs(introduced_g1) > 1e-12
                else None
            ),
        }

    ratios = [
        v["survival_ratio"]
        for v in per_target.values()
        if v["survival_ratio"] is not None
    ]
    return {
        "side": side,
        "magnitude": round(float(magnitude), 6),
        "per_target": per_target,
        "min_survival_ratio": round(min(ratios), 6) if ratios else None,
        "max_survival_ratio": round(max(ratios), 6) if ratios else None,
        "n_targets_measured": len(ratios),
        "note": (
            "EXACT: landmark coordinates mapped through trapezium.unwarp_x, so "
            "there is no resampling floor -- unlike the pixel measurement, whose "
            "floor exceeds a small deformation's contribution in absolute terms. "
            "Asymmetry is the SIGNED pair difference: with an absolute value a "
            "displacement opposing the face's existing asymmetry cancels it "
            "before adding, and the resulting ratio was unstable across sides "
            "(0.34 vs 1.15 at the commissure) for a purely arithmetic reason. "
            "survival_ratio > 1 is expected and correct, not an error: G2 "
            "MAGNIFIES horizontal distances (each row is stretched by "
            "1/(2*half_width), which is 1.66 at the brow and 1.0 at the lip), so "
            "a displacement high in the frame is amplified. What would matter is "
            "a ratio near ZERO, meaning the asymmetry was destroyed."
        ),
    }


def synthesised_asymmetry_survives(
    plain: np.ndarray, deformed: np.ndarray, unwarp=None
) -> dict:
    """Does the synthesised asymmetry survive G2's unwarp? **Measured on pixels.**

    PLAN §4.4 marks unwarp's asymmetry preservation [REASONED] and
    ``trapezium.asymmetry_is_preserved`` gates it analytically on displaced point
    pairs. This is the empirical companion on real pixels and on the deformation
    this module actually produces -- because the analytic gate tests the row
    stretch, not the composition of the stretch with a spline warp and a
    nearest-neighbour resample, and those are different quantities (R2).

    Reports the index before and after at both geometries, and the ratio. A ratio
    near 1 means the operation preserved the asymmetry it was handed; a collapse
    toward 0 would mean G2 is not usable for this arm.
    """
    if unwarp is None:
        from ..geometry.trapezium import unwarp as unwarp_fn
    else:
        unwarp_fn = unwarp

    g1_plain = asymmetry_index(plain)
    g1_deformed = asymmetry_index(deformed)
    g2_plain = asymmetry_index(unwarp_fn(plain))
    g2_deformed = asymmetry_index(unwarp_fn(deformed))

    introduced_g1 = g1_deformed - g1_plain
    introduced_g2 = g2_deformed - g2_plain

    # **The guard that keeps this from reporting a meaningless ratio.** If the
    # base image's own mirror asymmetry is comparable to what the deformation
    # introduced, the ratio is a measurement of the base, not of the warp -- and
    # it would still print a plausible-looking number. Measured: on a randomly
    # textured face the introduced quantity is ~1e-5 against a baseline of ~0.1.
    baseline_dominates = bool(abs(introduced_g1) < 0.10 * g1_plain)

    return {
        "g1": {"plain": round(g1_plain, 8), "deformed": round(g1_deformed, 8)},
        "g2": {"plain": round(g2_plain, 8), "deformed": round(g2_deformed, 8)},
        "introduced_g1": round(introduced_g1, 8),
        "introduced_g2": round(introduced_g2, 8),
        "survival_ratio": (
            round(introduced_g2 / introduced_g1, 6) if introduced_g1 > 0 else None
        ),
        "baseline_dominates": baseline_dominates,
        "note": (
            "The asymmetry INTRODUCED by the deformation, before and after "
            "unwarping, on pixels. The analytic gate "
            "(trapezium.asymmetry_is_preserved) tests the row stretch on point "
            "pairs; this tests the stretch composed with the spline warp and a "
            "nearest-neighbour resample, which is a different quantity. "
            "survival_ratio near 1 means G2 preserved it. "
            "READ baseline_dominates FIRST: when true, the base image's own "
            "mirror asymmetry swamps what the deformation introduced and the "
            "ratio describes the base rather than the warp. Run this on a "
            "mirror-symmetric base (PLAN 4.4: synthesise a KNOWN asymmetry); a "
            "real face's texture is asymmetric enough to hide the deformation "
            "entirely."
        ),
    }
