"""Phase 7C: augmentation, PRE-REGISTERED AS DATA.

**This module is the pre-registration**, on the same terms as ``phase7b``: the
arms enumerate here, the strengths are constants here, and the comparisons are
a function here, so the suite can check that what ran is what was declared.

----------------------------------------------------------------------------
AUGMENTATION WAS A SUPERVISOR INSTRUCTION AND HAS NEVER BEEN DONE
----------------------------------------------------------------------------
No cleft arm in this project has applied any -- not Phase 3, not Phase 4, not
the ladder's **47 lattice cells**, not Phase 7B's 24 trials.

It was not decided against. It was **foreclosed by an architectural choice
made for an unrelated reason**: Phase 3 measured full fine-tuning of ViT-B/16
on 237 images at PCC -0.013, worse than a constant (PLAN §4.7), so arms became
frozen-backbone with embeddings extracted ONCE per image. A stored artifact
has exactly one view per patient, and there is no point in that path where an
augmented view could enter. ``phase7b.OUT_OF_BUDGET`` records the same fact
from the other side: augmentation was *structurally* unavailable to a
stored-artifact arm, not merely unbudgeted.

That design made 47 cells affordable and silently removed a technique the
project was told to use. **The stored-artifact path is an optimisation, not a
constraint on the science**, and it is what this phase replaces.

----------------------------------------------------------------------------
THIS IS NOT A SEARCH, AND THAT IS THE PROTOCOL DIFFERENCE FROM 7B
----------------------------------------------------------------------------
[DECIDED 2026-08-03] Seven arms, a fixed list of hypotheses, **every one
reported**. No budget, no selection rule, no exhaustion criterion -- see
``NOT_A_SEARCH``.

Phase 7B needed those because it *searched*: it ranked 24 configurations and
reported the winner, so selecting on the test folds would have biased the
result by roughly the spread of what was tried. **A fixed comparison set has
no selection step, so there is nothing to bias.** Each arm is measured against
the same baseline with a paired BCa and reported whether it wins or loses.

What carries over unchanged is the *reporting* discipline: every arm gets its
own seed band (PLAN §4.12.1, inherited never) and a per-seed paired BCa
against 0.2520 through ``phase7b.paired_comparison``.

**Phase 7B is why that matters.** Its winner led inner-validation by +0.022
and came in 0.054 BELOW the baseline out-of-fold. The protocol's value was
exposing that rather than hiding it.
"""

from __future__ import annotations

from . import phase7b
from .ladder import SEED_POOL

#: The arm being tuned is the arm Phase 7B tuned -- ViT-B/16, ImageNet, G1,
#: whole image, mean label, PCC 0.2520, sd 0.0148 over five seeds.
#:
#: **Imported rather than copied.** Two records of one baseline is two places
#: for it to drift, and this project has already corrected a five-seed mean
#: paired with a ten-seed SD once (``ladder.CLEAN_GEOMETRY_MEASUREMENTS``).
BASELINE = phase7b.BASELINE

#: Five seeds, the transformer regime's count, taken from the ladder's spine
#: so every Phase 7C arm is comparable seed-for-seed with the baseline it is
#: measured against -- the same seeds mean the same inner-val splits.
SEEDS = list(SEED_POOL[:5])


class Phase7CError(RuntimeError):
    """The pre-registered protocol was not followed."""


#: **[MEASURED 2026-08-03] ONE EPOCH IS NOT AUGMENTATION, and the first run
#: measured that instead of the thing it was for.**
#:
#: Every arm reported ``selected_epochs`` of mostly 1s. **The model saw each
#: training image once, under a single random transform.** Augmentation works
#: by accumulating many transformed views so the model learns what is
#: invariant across them; one view is not a sample of a distribution, and a
#: single distorted view is strictly worse than the undistorted one.
#:
#: The ordering the arms produced was **plausible and monotone** -- identity
#: best, mild photometric next, heavy geometric worst -- which is exactly what
#: "how much does one random distortion degrade a one-epoch fit" looks like.
#: Nothing in the numbers said so.
#:
#: **Same signature as Phase 6's scheme runs before the gate-4 amendment**: a
#: 769-parameter head over frozen embeddings converges immediately, inner-val
#: never improves again, and patience terminates at epoch 1.
#:
#: ----------------------------------------------------------------------
#: THE FIX NEEDS NO HARNESS CHANGE
#: ----------------------------------------------------------------------
#: ``harness.run_fold`` ALREADY does best-checkpoint selection -- it keeps
#: ``best_predictions`` from the epoch with the best monitored score, and
#: ``patience`` only controls whether the loop breaks early. So **a patience
#: at least equal to ``max_epochs`` cannot fire**, and what remains is a fixed
#: budget with best-checkpoint selection: precisely gate 4's amendment, out of
#: the frozen harness, through config alone.
#:
#: (``pretrain.FIXED_BUDGET_POLICY`` implements the same policy separately
#: because the pretraining loop is not this one. Nothing here needs to.)
#:
#: **Identical across all seven arms**, so convergence speed cannot confound
#: the comparison -- an arm that ran longer because it kept improving would
#: differ from its neighbour in epochs as well as in policy.
#:
#: ----------------------------------------------------------------------
#: WHY THE MONITOR STAYS ``inner_val_mse``
#: ----------------------------------------------------------------------
#: The amendment specifies selection on ``inner_val_pcc``, and its REASON was
#: that patience was firing on noise in a flat region -- anatomy's epoch 5 and
#: epoch 7 differed by 2.3e-05, so run length tracked luck. **That reason does
#: not apply when patience cannot fire.** The monitor here only picks which
#: epoch's predictions are kept.
#:
#: Keeping ``inner_val_mse`` buys the thing this phase most needs: **arm 0
#: still reproduces 0.2520 exactly.** With no augmentation the features are
#: identical every epoch, so the trajectory is deterministic and the best
#: epoch is the same whether the loop stops at 6 or runs to 30 -- the budget
#: is inert for arm 0 and load-bearing for the other six. Changing the monitor
#: would move the selected epoch and cost the only end-to-end check this phase
#: has that the path is sound.
#:
#: Stated as a deliberate deviation from the letter of the amendment rather
#: than left implicit, because it is one.
EPOCH_POLICY = {
    "decided": "2026-08-03",
    "max_epochs": 30,
    #: Equal to the budget, so ``since_improvement`` peaks at ``max_epochs -
    #: 1`` and the break is unreachable. Fixed budget by arithmetic rather
    #: than by a flag the harness does not have.
    "patience": 30,
    "monitor": "inner_val_mse",
    "early_stopping_can_fire": False,
    "selection": "best checkpoint on the monitored quantity, by harness.run_fold",
    "identical_across_arms": True,
    "why": (
        "one epoch is one view per image, which is not a sample of a "
        "transform distribution. Augmentation works by accumulating views"
    ),
    "monitor_deviates_from_the_amendment": (
        "gate 4 specifies inner_val_pcc, for a reason -- patience firing on "
        "noise -- that cannot apply when patience cannot fire. inner_val_mse "
        "keeps arm 0 reproducing 0.2520 exactly, which is this phase's only "
        "end-to-end check that the path is sound"
    ),
}

#: **[MEASURED 2026-08-03] The tenth instance, and the first invisible in the
#: arms' own numbers.**
#:
#: The seven arms formed a plausible monotone ordering -- identity best, mild
#: photometric next, heavy geometric worst -- and every one of them was
#: measuring how much a single random distortion degrades a one-epoch fit. **No
#: quantity in the results was anomalous.** The defect surfaced only when
#: ``selected_epochs`` was read, which is a diagnostic field nobody compares
#: between arms because it is not the answer to anything.
#:
#: That distinguishes it from the nine before: those were checks reporting
#: success on their own failure mode, or a number attached to the wrong
#: question. This was **a result that looked exactly like the finding it was
#: supposed to produce**, and the only tell was in the run's bookkeeping.
#:
#: **Phase 7B was immune, by accident.** Its fits are closed form -- one solve,
#: no iteration, no epoch to stop early from -- so its configs carry no
#: ``max_epochs`` or ``patience`` at all and its winner block has no
#: ``selected_epochs``. That closed-form choice was made for the out-of-fold
#: discipline (``SEARCH_HEAD_PROCEDURE``: ``run_fold`` computes test
#: predictions as part of its job, so using it per trial would mean computing
#: 24 OOF vectors and declining to read them). It incidentally immunised 7B
#: against the defect that voided 7C. Worth recording because it is the
#: opposite of the usual pattern: a decision taken for one reason that turned
#: out to protect against an unrelated one.
ONE_EPOCH_IS_NOT_AUGMENTATION = {
    "measured": "2026-08-03",
    "symptom": "selected_epochs mostly 1 in all seven arms",
    "what_was_measured_instead": (
        "how much one random distortion degrades a one-epoch fit"
    ),
    "why_it_looked_right": (
        "identity best, mild photometric next, heavy geometric worst -- a "
        "plausible monotone ordering, and no anomalous quantity anywhere"
    ),
    "found_by": "reading selected_epochs, which is bookkeeping and not an answer",
    "precedent": (
        "Phase 6's scheme runs before the gate-4 amendment: a 769-parameter "
        "head over frozen embeddings converges immediately, inner-val never "
        "improves, patience terminates at epoch 1"
    ),
    "phase_7b_immune_by_accident": (
        "7B's fits are closed form -- no iteration, no epoch to stop early "
        "from -- because of a decision taken for the OOF discipline. It "
        "protected against an unrelated defect"
    ),
    "arms_are_void": "not superseded -- none of them ran the policy its config named",
}


#: ---------------------------------------------------------------------------
#: WHERE AUGMENTATION ENTERS. No new training loop.
#: ---------------------------------------------------------------------------
#:
#: **[DECIDED 2026-08-03] The frozen harness already supports this**, and
#: building a parallel loop would put a second implementation of the
#: measurement apparatus in the one place it must not go: ``harness.py`` is
#: frozen precisely because it IS the measurement, and gate 1's determinism
#: guarantee is about that loop.
#:
#: The path that exists:
#:
#: * ``phase3.prepare_features`` returns **raw images** when
#:   ``trainable == "full"`` -- it already does, for the full fine-tuning arm.
#: * ``harness.run_fold`` calls ``backbone.train_epoch(features, labels)``
#:   **once per epoch**, and ``backbone.predict(features)`` for inner-val and
#:   test.
#:
#: So an augmenting backbone receives images, and inside ``train_epoch``
#: augments the fold's training pixels, forwards them through the frozen
#: ViT-B/16, and fits the head. ``predict`` forwards **without augmenting**.
#:
#: **That makes the never-augment-evaluation rule STRUCTURAL rather than
#: remembered.** Evaluation reaches the model only through ``predict``;
#: ``predict`` contains no augmentation; therefore validation and test images
#: cannot be augmented. Brief §8's first trap stops being a discipline someone
#: has to observe and becomes a property of the call graph.
LIVE_PATH = {
    "decided": "2026-08-03",
    "no_new_loop": (
        "harness.run_fold is frozen and calls train_epoch once per epoch; an "
        "augmenting Backbone slots into the protocol that already exists"
    ),
    "features_are_images": (
        "phase3.prepare_features returns raw images at trainable == 'full', "
        "which is the existing route for a model that does its own forward"
    ),
    "augment_in": "train_epoch",
    "never_in": "predict",
    "structural_guarantee": (
        "evaluation reaches the model only through predict, and predict does "
        "not augment -- so validation and test images CANNOT be augmented. A "
        "property of the call graph, not a rule to observe"
    ),
    "gate_1_intact": (
        "the training loop is unchanged, so gate 1 still tests the harness it "
        "was measured on"
    ),
}

#: **Arm 0's scope, recorded narrowly so a settled question is not
#: re-litigated.**
#:
#: The brief framed arm 0 as establishing that the live and stored paths are
#: the same experiment. **That is already measured at this exact cell**:
#: ``p3_train_cv`` (live) gives 0.25206 sd 0.01483 and the Stage D1 artifact
#: arm gives 0.2520 sd 0.0148 -- agreement to four decimals on the mean AND
#: the SD, recorded in ``ladder.CLEAN_GEOMETRY_MEASUREMENTS`` and
#: ``ladder.STAGE_D1_AT_G1``.
#:
#: **Arm 0 tests the NEW augmenting backbone at identity policy, and nothing
#: more.** It is still a gate -- if the new code at identity does not
#: reproduce the known value, nothing downstream is interpretable -- but it is
#: a gate on new code, not on a question this project answered twice.
ARM_0_SCOPE = {
    "tests": "the new augmenting backbone at identity policy reproduces 0.2520",
    "does_not_test": (
        "whether live and stored extraction agree -- settled at this cell, "
        "0.25206 (live) against 0.2520 (artifact), four decimals on mean and SD"
    ),
    "settled_by": [
        "ladder.CLEAN_GEOMETRY_MEASUREMENTS['imagenet']['g1'] -- p3_train_cv, live",
        "ladder.STAGE_D1_AT_G1 -- p7_d1_vit_b16_imagenet_g1, artifact",
    ],
    "is_a_gate": True,
    "on_failure": "stop; nothing downstream is interpretable",
}


#: ---------------------------------------------------------------------------
#: THE THREE FAMILIES, at ONE pre-registered setting each
#: ---------------------------------------------------------------------------

#: **Photometric [DECIDED 2026-08-03].** One setting, not a sweep: this phase
#: is a set of comparisons and a swept strength would make it a search again.
#:
#: **[MEASURED] a fairness rationale beyond variance reduction.** Phase 4's
#: segmentation work found a **brightness-linked failure pattern** -- the
#: instrument's failures were not distributed evenly across the brightness
#: quartiles, and every method in that family is intensity-based and so
#: implicitly calibrated on lighter skin (PLAN Part 5, the fairness
#: limitation). Brightness and contrast augmentation therefore has a fairness
#: motivation here, not only a regularisation one, and if this arm helps it is
#: worth reporting per brightness quartile.
#:
#: > **[STRUCK 2026-08-03] Specific figures were removed as unsourced.** The
#: > brief quoted "7 of 8 over-selection failures in the darkest brightness
#: > quartile, a 3.6x enrichment". **Nothing in the repo carries those
#: > numbers**; PLAN records the pattern qualitatively and
#: > ``fairness_by_brightness`` splits the pass rate by quartile without those
#: > values appearing anywhere. Recorded as struck rather than silently
#: > dropped: it is the same shape as the fabricated ORIGINAL-folder count,
#: > and a measured-sounding figure with no source is exactly what survives
#: > into a deck unchallenged. If the numbers exist in a cluster run's
#: > metrics.json they can be cited; until then the motivation stands
#: > qualitatively, which is all it needs to.
PHOTOMETRIC = {
    "decided": "2026-08-03",
    "brightness": 0.2,
    "contrast": 0.2,
    "saturation": 0.2,
    "hue": 0.05,
    "single_setting": True,
    "why_not_swept": (
        "a swept strength would reintroduce a selection step, and this phase "
        "is a fixed comparison set precisely so there is nothing to select on"
    ),
    "fairness_rationale": (
        "Phase 4's segmentation work found a brightness-linked failure "
        "pattern, and every method in that family is intensity-based and so "
        "implicitly calibrated on lighter skin. Report per brightness "
        "quartile if this arm helps"
    ),
    "struck": (
        "the brief's '7 of 8 ... 3.6x enrichment' -- unsourced, same shape as "
        "the fabricated ORIGINAL-folder count. The motivation is qualitative"
    ),
}

#: **Geometric [DECIDED -- parameters given at supervision].** Rotation is carried
#: separately because it is an ARM, not a default: see ``ROTATION_IS_AN_ARM``.
GEOMETRIC = {
    "rotation_degrees": 5.0,
    "translate_fraction": 0.10,
    "scale": (0.9, 1.1),
    "lockstep": (
        "the trapezium mask and every region box move WITH the image. "
        "geometry/mapping.py supplies each patient's boxes in their own "
        "pixels and is FROZEN -- composing a rotation onto them is new code, "
        "additive, outside the frozen set. A rotated image with a stationary "
        "mask is a different experiment from the one intended"
    ),
}

#: **[MEASURED] Horizontal flip is OUT, by the supervision own condition.**
#:
#: The rule given was "only if labels are symmetric". They are not:
#: **laterality is not recorded anywhere in this cohort** (PLAN §4.1), and
#: Asher-McDade grades *nasal symmetry* as one of its four components.
#: Flipping a left-sided cleft produces a right-sided one carrying the same
#: grade, teaching the model that side is irrelevant to a label that is partly
#: about a lateral deviation.
#:
#: Recorded as **excluded by the stated condition on a measured fact**, not by
#: preference -- the distinction matters, because a preference invites
#: revisiting and a failed condition does not. PLAN §4.7 reached the same
#: conclusion independently.
HORIZONTAL_FLIP = {
    "included": False,
    "condition_given": "only if labels are symmetric",
    "condition_fails_because": (
        "laterality is not recorded anywhere in this cohort (PLAN §4.1), and "
        "Asher-McDade grades nasal symmetry as one of four components"
    ),
    "excluded_by": "the stated condition, on a measured fact -- not preference",
}

#: **[REASONED] Rotation is an ARM, not a default.**
#:
#: The cleft crops are **not levelled** -- measured on the Phase 2 contact
#: sheet -- and the tilt is worst at the nose tips and least at the lips. That
#: gradient is not head roll, which rotates every structure equally; it is
#: consistent with **nasal deviation**, which is clinical signal and one of
#: the four things the grade assesses.
#:
#: So +/-5 degrees may either simulate real photographic variation or augment
#: away part of the signal, and one extra configuration answers it: arms 2 and
#: 3 differ in rotation alone.
ROTATION_IS_AN_ARM = {
    "provenance": "REASONED",
    "why": (
        "the crops are not levelled and the tilt gradient is worst at the "
        "nose tips, least at the lips -- consistent with nasal deviation, "
        "which is clinical signal, rather than with head roll"
    ),
    "resolved_by": ("arm_2_geometric", "arm_3_geometric_rotation"),
    "determinism": (
        "rotation interpolation is a kernel gate 1 has never covered. If it "
        "raises under use_deterministic_algorithms(True) that is a FINDING to "
        "report, not a reason to disable the flag -- the AG-Net roi_align "
        "rule (ladder._arm's `deterministic` field): an opt-out is per-arm, "
        "declared, and refused at load and runtime if undeclared. Never "
        "global, never silent"
    ),
}

#: ---------------------------------------------------------------------------
#: REGION-AWARE AUGMENTATION [DECIDED -- supervision's design]
#: ---------------------------------------------------------------------------
#:
#: Stronger augmentation outside the key regions, lighter inside. The
#: protection mask comes from the **Phase 2 anatomy regions**
#: (``geometry/generators.py``, frozen), mapped per patient into that image's
#: own pixels at the frozen parameters ``v_offset 0.04``, ``scale 1.80``,
#: ``scale_x 1.60``.
#:
#: **Note what this means: the anatomy scheme, which LOST as a patch layout,
#: has a second use as a spatial prior for augmentation.** The same design
#: serving a different purpose, and worth recording as such -- Phase 4 and
#: Stage E measured it as a place to *look*, and this uses it as a place to
#: *preserve*, which the scheme comparison says nothing about either way.
#:
#: **The names are the real ones.** The brief's protect-list named
#: ``cupids_bow_peak``, which does not exist -- ``generators.py`` defines 9
#: midline and 9 bilateral regions (27 boxes), and the cupid's bow appears
#: only in a comment, covered by ``philtrum`` and ``philtral_column``. Every
#: name below is asserted against the generator at test time, which is the
#: check that would have caught it.
#:
#: ----------------------------------------------------------------------
#: **[CORRECTED 2026-08-03] R9 AGAIN: THE PARENTHESES WERE EXAMPLES**
#: ----------------------------------------------------------------------
#: The list was seven names. It is fifteen.
#:
#: The instruction was *"nose (alar base asymmetry), upper lip (philtral
#: column, Cupid's bow), vermillion border"*, and the seven-name list read the
#: parentheses as an **enumeration** when they are **examples of what each
#: region is about**. The categories are *nose*, *upper lip* and *vermillion
#: border*; ``alar_base`` and ``philtral_column`` illustrate them.
#:
#: **This is the second instance of PLAN R9** -- "separate the procedure from
#: the examples" -- and the first was the same supervision "27 regions",
#: where the anatomical terms listed at supervision were taken as the
#: specification and
#: the grid construction as loose (PLAN §4.5). Same reader, same shape, same
#: correction: the named structures illustrate the category rather than
#: bounding it.
#:
#: So: **every nose and upper-lip region is protected.** What remains
#: strongly augmented is glabella (forehead), the canthi and the orbits
#: (eyes, cheeks) -- plus everything the 27 boxes do not cover at all, which
#: is ~30% of the frame (the scheme covers 70.2%, PLAN §4.5.3). That is
#: "cheeks, forehead, background and periphery", which is what was specified.
PROTECTED_REGIONS = (
    # midline -- 8 of the 9; only `glabella` (forehead) is left augmentable
    "nasal_root",
    "nasal_dorsum_upper",
    "nasal_dorsum_lower",
    "nasal_tip",
    "columella",
    "subnasale",
    "philtrum",
    "labial_tubercle",
    # bilateral -- 7 of the 9, each a mirrored pair, so 2 boxes apiece.
    # `medial_canthus` and `lateral_orbit` are the eyes and cheeks.
    "nasal_sidewall",
    "alar_rim",
    "alar_base",
    "nostril_sill",
    "philtral_column",
    "vermillion_border",
    "commissure",
)

#: The record of the correction, kept as data because R9 has now cost this
#: project twice and the shape is what carries.
PROTECT_LIST_IS_CATEGORIES_NOT_EXAMPLES = {
    "corrected": "2026-08-03",
    "was": 7,
    "now": len(PROTECTED_REGIONS),
    "instruction": (
        "nose (alar base asymmetry), upper lip (philtral column, Cupid's "
        "bow), vermillion border"
    ),
    "misreading": "the parentheses read as an enumeration",
    "correct_reading": (
        "the parentheses are examples of what each region is ABOUT; the "
        "categories are nose, upper lip and vermillion border"
    ),
    "rule": "PLAN R9 -- separate the procedure from the examples",
    "first_instance": (
        "the same supervision '27 regions', where the anatomical terms were "
        "taken as the specification and the grid construction as loose "
        "(PLAN §4.5). Same shape, same correction"
    ),
    "left_augmentable": ("glabella", "medial_canthus", "lateral_orbit"),
}

#: Augmentation strength inside a protected region, as a multiplier on the
#: declared family strength. Outside is full strength.
REGION_AWARE = {
    "decided": "2026-08-03",
    "outside_strength": 1.0,
    "inside_strength": 0.25,
    "protected": PROTECTED_REGIONS,
    "source": "geometry/generators.py anatomy regions, frozen, mapped per patient",
    "frozen_parameters": {"v_offset": 0.04, "scale": 1.80, "scale_x": 1.60},
    "second_use_of_a_losing_scheme": (
        "the anatomy layout lost as a PATCH scheme (Phase 4, Stage E). This "
        "uses it as a spatial prior for what to PRESERVE, which the scheme "
        "comparison speaks to neither way"
    ),
}

#: **[DECIDED 2026-08-03] The boundary is a SOFT FALLOFF, not a hard mask.**
#:
#: A hard mask leaves a seam where augmentation stops -- and that seam sits at
#: a **fixed anatomical offset**, on the very regions the grade is about. A
#: model can key on it, which is the shortcut-learning concern in its most
#: literal form: an edge that correlates with the protected anatomy and
#: appears in every augmented training image but no evaluation image.
#:
#: The falloff is Gaussian, with sigma a fraction of the crop width so it
#: scales with the image like every other placement in this project rather
#: than being a fixed pixel count on crops whose aspect ratios span a factor
#: of two (PLAN §4.4).
BLEND = {
    "decided": "2026-08-03",
    "kind": "soft_falloff",
    "profile": "gaussian",
    "sigma_fraction_of_crop_width": 0.02,
    "why_not_hard": (
        "a hard seam sits at a fixed anatomical offset, on the regions the "
        "grade is about, and appears in every augmented training image and no "
        "evaluation image -- a feature a model can key on"
    ),
    "why_relative": (
        "crop aspect ratios span a factor of two, so a fixed pixel width "
        "would be a different blend on each patient"
    ),
}


#: ---------------------------------------------------------------------------
#: THE SEVEN ARMS
#: ---------------------------------------------------------------------------
#:
#: Each is (photometric, geometric, rotation, region_aware). Written as
#: factors rather than as names so the one-factor property of each declared
#: comparison can be ASSERTED over the derived set, exactly as
#: ``ladder.comparisons()`` does -- a pair differing in two factors measures
#: neither, and the failure is invisible in the output.
_ARM_FACTORS = (
    ("identity", False, False, False, False),
    ("photometric", True, False, False, False),
    ("geometric", False, True, False, False),
    ("geometric_rotation", False, True, True, False),
    ("region_photometric", True, False, False, True),
    ("region_full", True, True, False, True),
    ("whole_image_full", True, True, False, False),
)

#: **[DECIDED 2026-08-03] Arms 5 and 6 carry NO rotation, and that is
#: deliberate.**
#:
#: They are the pair that answers the supervision design question -- whether
#: protecting the key regions beats augmenting everywhere -- so they must
#: differ in region-awareness ALONE. Including rotation would be legal for
#: that (both sides would carry it) but wrong for a different reason:
#: **rotation is unresolved until arms 2 and 3 report**, and baking an
#: unresolved and possibly harmful transform into both sides asks the region
#: question at a degraded operating point.
#:
#: That is the Stage G1 lesson -- a label comparison on ViT's weakest cell
#: could only ever have returned "unresolved" (``ladder.LABEL_CONTROL_INIT``).
#: An instrument is chosen for its power, not for its completeness.
ARMS_5_AND_6_EXCLUDE_ROTATION = {
    "decided": "2026-08-03",
    "why": (
        "rotation is unresolved until arms 2 and 3 report; baking a possibly "
        "harmful transform into both sides asks the region question at a "
        "degraded operating point"
    ),
    "precedent": (
        "Stage G1 -- the label triple went on imagenet rather than the weakest "
        "cell, because an instrument is chosen for its power"
    ),
    "note": (
        "'all' in arm 6 means both FAMILIES, not every transform: rotation is "
        "an arm of its own and is not folded into a comparison that cannot "
        "interpret it"
    ),
}


def arms() -> list[dict]:
    """The seven arms, in a fixed order, as data."""
    out = []
    for index, (name, photometric, geometric, rotation, region_aware) in enumerate(
        _ARM_FACTORS
    ):
        if rotation and not geometric:
            raise Phase7CError(
                f"arm {name!r} declares rotation without geometric; rotation "
                "is a geometric transform and cannot vary on its own"
            )
        out.append({
            "index": index,
            "name": f"p7c_{index}_{name}",
            "photometric": photometric,
            "geometric": geometric,
            "rotation": rotation,
            "region_aware": region_aware,
            "seeds": list(SEEDS),
            "backbone": BASELINE["backbone"],
            "init": BASELINE["init"],
            "geometry": BASELINE["geometry"],
            "label": BASELINE["label"],
        })
    return out


#: The fields a comparison may vary. ``rotation`` is separate from
#: ``geometric`` because arms 2 and 3 vary it while both carry the family.
COMPARED_FIELDS = ("photometric", "geometric", "rotation", "region_aware")


def comparisons() -> list[dict]:
    """Every delta this phase claims, with the ONE field it varies.

    Each arm is additionally measured against the 0.2520 baseline with a
    per-seed paired BCa; those are not listed here because they are not
    between-arm comparisons and share no factor structure.
    """
    by_name = {arm["name"]: arm for arm in arms()}

    def pair(question, varies, a, b):
        entry = {
            "question": question, "varies": varies,
            "a": by_name[a], "b": by_name[b],
        }
        differing = {
            field for field in COMPARED_FIELDS
            if entry["a"][field] != entry["b"][field]
        }
        if differing != {varies}:
            raise Phase7CError(
                f"{question} claims to vary {varies!r} but {a} and {b} differ "
                f"in {sorted(differing)}"
            )
        return entry

    return [
        pair(
            "rotation", "rotation",
            "p7c_2_geometric", "p7c_3_geometric_rotation",
        ),
        pair(
            "region_awareness", "region_aware",
            "p7c_6_whole_image_full", "p7c_5_region_full",
        ),
        pair(
            "region_awareness_photometric", "region_aware",
            "p7c_1_photometric", "p7c_4_region_photometric",
        ),
    ]


#: **[DECIDED 2026-08-03] There is no budget, no selection rule and no
#: exhaustion criterion, and their absence is the design.**
#:
#: Phase 7B needed all three because it SEARCHED: it ranked 24 configurations
#: on inner-validation and reported the best, so selecting on the test folds
#: would have biased the result by roughly the spread of what was tried, and
#: invalidated the seed bands and the claim criterion together.
#:
#: **Phase 7C ranks nothing.** Seven arms, each a hypothesis, each reported
#: with its own band and its own paired BCa whether it wins or loses. With no
#: selection step there is nothing to bias -- the protection Phase 7B needed
#: is protection against a procedure this phase does not run.
#:
#: The reporting discipline carries over unchanged and is where the value is:
#: **every arm reported**, each with its OWN seed SD (PLAN §4.12.1), each
#: against 0.2520 per seed through ``phase7b.paired_comparison``.
NOT_A_SEARCH = {
    "decided": "2026-08-03",
    "budget": None,
    "selection_rule": None,
    "exhaustion_criterion": None,
    "why": (
        "Phase 7B was the search and needed those because it ranked 24 "
        "configurations and reported the best. This is a fixed set of "
        "comparisons with no selection step, so there is nothing to bias"
    ),
    "carried_over": [
        "every arm reported, winners and losers alike",
        "each arm's OWN seed SD -- PLAN §4.12.1, inherited never",
        "per-seed paired BCa against 0.2520, phase7b.paired_comparison",
    ],
    "why_the_discipline_still_matters": (
        "Phase 7B's winner led inner-validation by +0.022 and came in 0.054 "
        "BELOW the baseline out-of-fold. Reporting every arm is what exposed "
        "that rather than hiding it"
    ),
}


#: **[MEASURED 2026-08-03, run v3] AUGMENTATION DOES NOT HELP. Every arm is
#: claimably below identity, and both region tests are null.**
#:
#: Thirty epochs, the supervision parameters and the region design they
#: specify. Arm 0 held
#: at 0.2520 sd 0.0148 -- the gate survived the budget change, as predicted:
#: with no augmentation the features are identical every epoch, so
#: best-checkpoint selection picks the same epoch whether the loop stops at 6
#: or runs to 30.
#:
#: Every delta re-derived from the two arms' own SDs. **All six augmented arms
#: are claimably worse**, and two clear ``single_run_95`` as well.
#:
#: **Rotation is directionally worse and not claimable** (-0.0114 against a
#: 0.0555 threshold). Its SD is 0.0586, the widest in the phase, which is most
#: of why: the threshold it must clear is more than twice any other arm's.
#: Recorded as unresolved rather than as "no effect" -- the arm cannot
#: separate them, which is a different statement from their being equal.
#:
#: **Region-awareness is null on BOTH independent tests** -- arm 5 against
#: arm 6 gives -0.0195 (threshold 0.0478) and arm 1 against arm 4 gives
#: +0.0065 (threshold 0.0313). They disagree in sign and neither is claimable,
#: which is what two underpowered nulls look like rather than a contradiction.
#: The the design question raised at supervision gets no answer from this data, in either
#: direction.
#:
#: **The mechanism is in the curves, and it is not premature convergence.**
#: Arm 1, seed 1337, fold 0: train loss 0.195 -> 0.068 by epoch 3 while
#: inner-val MSE RISES 0.491 -> 0.559. Fold 4 by epoch 30: train loss 0.0087,
#: inner-val MSE 0.70, PCC -0.114. **The head overfits 152 samples within
#: about three epochs and augmentation does not prevent it** -- the model
#: memorises perturbed views instead of clean ones. Epoch 1 is selected
#: because it is the only epoch before divergence.
#:
#: That is the finding underneath the numbers, and it is confound-free: **if
#: augmentation were regularising, the augmented arms' inner-val optimum would
#: arrive LATER than identity's.** It does not. Most folds still select epoch
#: 1-3, the same as arm 0.
#: **[MEASURED 2026-08-03] TWO ROUNDS, and the correction changed a verdict.**
#:
#: ``matched3`` is the PRIMARY reported round. ``selected30`` is not history:
#: it carries the only evidence for the mechanism, which ``matched3``
#: structurally cannot -- see ``MECHANISM_NEEDS_THE_LONG_BUDGET``.
#:
#: **What the correction changed.** Photometric moved 0.2069 -> 0.2364 and its
#: delta against identity went from claimably worse (-0.0451, threshold
#: 0.0220) to **not claimable** (-0.0156, threshold 0.0217). One verdict
#: crossed; five did not.
#:
#: **The number of deltas surviving ``single_run_95`` DOUBLED**, from two to
#: four. Removing draw luck tightened every arm; it only moved one across a
#: threshold.
#:
#: **Arm 0 is identical in both rounds** -- 0.2520, SD 0.0148, epoch 1 in
#: every fold. That is what makes the two rounds comparable, and it is the
#: budget being inert for a deterministic arm, as predicted.
#:
#: > **[RETRACTED before it was quoted] "The confound bit hardest where it was
#: > predicted to."** The first draft of this record said photometric flipped
#: > because it is the mildest augmentation and so had the most to gain from a
#: > lucky late draw. **Three things in the arithmetic refuse it**, and all
#: > three were found by checking rather than by review:
#: >
#: > * ``4_region_photometric`` is MILDER than ``1_photometric`` -- it is the
#: >   same photometric jitter damped to 0.25x inside the protected regions --
#: >   so "mildest" named the wrong arm;
#: > * ``4_region_photometric`` was also the arm CLOSEST to its threshold at
#: >   the long budget (margin +0.0097 against photometric's +0.0231), so the
#: >   arm most likely to flip on any perturbation did not flip;
#: > * ``6_whole_image_full`` had the MOST late-selecting folds (three past
#: >   epoch 3, including 29 and 19) and moved in the OPPOSITE direction,
#: >   -0.0171.
#: >
#: > So the between-round move is **not predictable from how late an arm
#: > selected**. The confound is real -- ``SELECTION_CONFLATES_DRAW_QUALITY``
#: > stands on its mechanism, and the round was worth running because it
#: > changed a verdict -- but its per-arm effect is not established by seven
#: > points, and a story that fits one arm and contradicts two is not a
#: > finding. Recorded rather than deleted because it is the shape of
#: > explanation this project keeps having to withdraw: plausible, directional,
#: > and refuted by the arithmetic it was built to explain.
STAGE_7C_RESULTS = {
    "measured": "2026-08-03",
    #: **[CORRECTED 2026-08-03] The PRE-REGISTERED round is primary.**
    #: ``matched3`` was going to be nominated primary, on the grounds that it
    #: removes a confound. It is not a second round --
    #: ``ROUND_2_IS_ROUND_1_TRUNCATED`` -- and its window was chosen from
    #: round 1's own curves after round 1's verdicts were known.
    "primary_round": "selected30",
    "why_primary": (
        "its budget was fixed in EPOCH_POLICY BEFORE it ran. matched3's was "
        "chosen from selected30's curves after selected30's verdicts were "
        "known, and matched3 re-selects over the same trajectories rather "
        "than measuring new ones"
    ),
    "matched3_role": "post-hoc sensitivity analysis, not a replication",
    "rounds": {
        "selected30": {
            "epoch_policy": "fixed 30-epoch budget, best-checkpoint selection",
            "arms": {
                "0_identity": {"pcc": 0.2520, "sd": 0.0148, "epochs": [1, 1, 1, 1, 1]},
                "1_photometric": {"pcc": 0.2069, "sd": 0.0202,
                                  "epochs": [1, 3, 30, 11, 1]},
                "2_geometric": {"pcc": 0.1629, "sd": 0.0239,
                                "epochs": [14, 1, 8, 3, 2]},
                "3_geometric_rotation": {"pcc": 0.1515, "sd": 0.0586,
                                         "epochs": [1, 1, 2, 16, 1]},
                "4_region_photometric": {"pcc": 0.2134, "sd": 0.0295,
                                         "epochs": [1, 3, 16, 4, 1]},
                "5_region_full": {"pcc": 0.1636, "sd": 0.0357,
                                  "epochs": [1, 14, 1, 1, 21]},
                "6_whole_image_full": {"pcc": 0.1831, "sd": 0.0412,
                                       "epochs": [1, 5, 29, 19, 2]},
            },
            "against_identity": {
                "1_photometric": {"delta": -0.0451, "threshold": 0.0220,
                                  "claimable": True},
                "2_geometric": {"delta": -0.0891, "threshold": 0.0246,
                                "claimable": True, "survives_single_run_95": True},
                "3_geometric_rotation": {"delta": -0.1005, "threshold": 0.0530,
                                         "claimable": True},
                "4_region_photometric": {"delta": -0.0386, "threshold": 0.0289,
                                         "claimable": True},
                "5_region_full": {"delta": -0.0884, "threshold": 0.0339,
                                  "claimable": True, "survives_single_run_95": True},
                "6_whole_image_full": {"delta": -0.0689, "threshold": 0.0384,
                                       "claimable": True},
            },
            "comparisons": {
                "rotation": {"delta": -0.0114, "threshold": 0.0555,
                             "claimable": False},
                "region_awareness": {"delta": -0.0195, "threshold": 0.0478,
                                     "claimable": False},
                "region_awareness_photometric": {"delta": 0.0065,
                                                 "threshold": 0.0313,
                                                 "claimable": False},
            },
        },
        "matched3": {
            "epoch_policy": "fixed 3-epoch budget, best-checkpoint selection",
            "arms": {
                "0_identity": {"pcc": 0.2520, "sd": 0.0148, "epochs": [1, 1, 1, 1, 1]},
                "1_photometric": {"pcc": 0.2364, "sd": 0.0199,
                                  "epochs": [1, 3, 1, 1, 1]},
                "2_geometric": {"pcc": 0.1757, "sd": 0.0283,
                                "epochs": [3, 1, 3, 3, 2]},
                "3_geometric_rotation": {"pcc": 0.1542, "sd": 0.0378,
                                         "epochs": [1, 1, 2, 3, 1]},
                "4_region_photometric": {"pcc": 0.2157, "sd": 0.0263,
                                         "epochs": [1, 3, 1, 1, 1]},
                "5_region_full": {"pcc": 0.1659, "sd": 0.0360,
                                  "epochs": [1, 2, 1, 1, 1]},
                "6_whole_image_full": {"pcc": 0.1660, "sd": 0.0254,
                                       "epochs": [1, 2, 1, 1, 2]},
            },
            "against_identity": {
                #: **The verdict the correction changed.**
                "1_photometric": {"delta": -0.0156, "threshold": 0.0217,
                                  "claimable": False},
                "2_geometric": {"delta": -0.0763, "threshold": 0.0280,
                                "claimable": True, "survives_single_run_95": True},
                "3_geometric_rotation": {"delta": -0.0978, "threshold": 0.0356,
                                         "claimable": True,
                                         "survives_single_run_95": True},
                "4_region_photometric": {"delta": -0.0363, "threshold": 0.0265,
                                         "claimable": True},
                "5_region_full": {"delta": -0.0861, "threshold": 0.0341,
                                  "claimable": True, "survives_single_run_95": True},
                "6_whole_image_full": {"delta": -0.0860, "threshold": 0.0258,
                                       "claimable": True,
                                       "survives_single_run_95": True},
            },
            "comparisons": {
                "rotation": {"delta": -0.0215, "threshold": 0.0414,
                             "claimable": False},
                "region_awareness": {"delta": -0.0001, "threshold": 0.0386,
                                     "claimable": False},
                "region_awareness_photometric": {"delta": -0.0207,
                                                 "threshold": 0.0289,
                                                 "claimable": False},
            },
        },
    },
    "changed_verdict": {
        "arm": "1_photometric",
        "was": "claimably worse than identity (-0.0451, threshold 0.0220)",
        "now": "not claimable (-0.0156, threshold 0.0217)",
        "moved_most": True,
        "between_round_move": 0.0295,
        "why_here": (
            "NOT ESTABLISHED. It moved the most (+0.0295), and that is all "
            "seven points support -- see the retraction above. The arm "
            "closest to its threshold (4_region_photometric, margin +0.0097) "
            "did not flip, and the arm with the most late-selecting folds "
            "(6_whole_image_full) moved the other way"
        ),
    },
    #: The per-arm moves, so the retraction above can be checked rather than
    #: taken. Positive means the shorter budget scored higher.
    "between_round_moves": {
        "1_photometric": 0.0295,
        "2_geometric": 0.0128,
        "3_geometric_rotation": 0.0027,
        "4_region_photometric": 0.0023,
        "5_region_full": 0.0023,
        "6_whole_image_full": -0.0171,
    },
    "single_run_95_survivors": {"selected30": 2, "matched3": 4},
    #: **[CORRECTED 2026-08-03] Photometric is UNRESOLVED, not neutral.**
    #:
    #: It is claimably worse in the pre-registered round and not claimable
    #: under a post-hoc narrower selection window. ``MATCHED_EPOCH_POLICY``
    #: pre-committed to exactly this case -- *"if they disagree, the
    #: disagreement is the result and the selection procedure is the factor"*
    #: -- and they disagreed. Calling it neutral takes the answer from the
    #: window that gave the preferred one.
    # **[SUPERSEDED 2026-08-04] Every verdict below was formed from PLAN
    # §4.3's condition 2 alone. Condition 1 was then computed and withdrew all
    # nine. See PAIRED_BCA_WITHDREW_EVERY_VERDICT -- that record, not this
    # field, is the phase's outcome.**
    #
    # Kept rather than overwritten because the SEQUENCE is the finding: these
    # are the verdicts the phase held, in the words it held them in, before
    # the criterion it had pre-registered was applied to them.
    "verdict": (
        "SUPERSEDED -- see PAIRED_BCA_WITHDREW_EVERY_VERDICT. Nothing in "
        "this phase is claimable. The verdicts below were condition-2-only "
        "and did not survive condition 1"
    ),
    "verdict_before_condition_1": (
        "GEOMETRIC augmentation claimably hurts, under both selection "
        "windows and conservatively under the unpaired-threshold error -- "
        "this is the phase's one robust finding. PHOTOMETRIC is UNRESOLVED. "
        "REGION-AWARE protection gets NO ANSWER in either direction"
    ),
    "per_family": {
        "geometric": {
            "verdict": "WITHDRAWN -- was 'claimably worse than identity'",
            "robust": "arms 2, 3, 5 and 6 in BOTH windows",
            "withdrawn_by": "condition 1: no arm has five intervals excluding zero",
        },
        "photometric": {
            "verdict": "UNRESOLVED",
            "why": (
                "claimable at the pre-registered window, not claimable at a "
                "post-hoc narrower one. The phase pre-registered that this "
                "makes the selection procedure the factor"
            ),
            "and_now": (
                "moot -- it fails condition 1 in both windows, so the "
                "disagreement between windows was never load-bearing"
            ),
        },
        "region_aware": {
            #: NOT "makes no difference" -- see
            #: REGION_TESTS_ARE_NOT_INDEPENDENT. The two tests are the same
            #: factor in two dilutions, unequally powered, and 1-vs-4 flips
            #: to claimable at ten seeds.
            "verdict": "NO ANSWER, in either direction",
            "why": (
                "both tests null at n=5, but they are not independent, the "
                "intervention is small by construction, the unprotected side "
                "is diluted by the white pad, and 1-vs-4 flips at ten seeds"
            ),
            "and_now": (
                "the wording was already right and is now the wording for the "
                "WHOLE phase, not just this family"
            ),
        },
        "rotation": {
            "verdict": "UNRESOLVED",
            "why": "directionally worse, not claimable in either window",
        },
    },
}

#: **[MEASURED 2026-08-03] ROUND 2 IS ROUND 1 TRUNCATED. It is a re-selection,
#: not a replication, and it cost the phase its cleanest-sounding claim.**
#:
#: The augmentation view stream is ``rng_for(seed, fold, epoch)`` -- derived
#: from the coordinates alone, deliberately, so a resumed run reproduces its
#: views with no RNG state to save. The head resets on ``torch.manual_seed``
#: at a constant learning rate with no schedule keyed to ``max_epochs``, and
#: ``run_fold`` uses ``patience`` only to break. **So epoch n's fit does not
#: depend on the budget**, and round 2's epochs 1-3 are the same fits as round
#: 1's epochs 1-3.
#:
#: Round 2 therefore changes exactly one thing: the candidate set the
#: best-checkpoint argmin runs over, from {1..30} to {1..3}. It could have
#: been computed from round 1's stored curves without running anything.
#:
#: **23 of the 35 fits are the same fit in both rounds** -- every fold where
#: round 1 already selected an epoch <= 3. Every epoch pattern is consistent
#: with this and none contradicts it. So "the two rounds agree" is two-thirds
#: arithmetic identity; there is one direction of evidence, not two.
#:
#: **And the window was chosen from the data it was applied to.** The three
#: came from *"the curves put divergence at about epoch 3"* -- round 1's
#: curves -- after round 1's verdicts were known. That is a data-dependent
#: restriction of a model-selection candidate set, decided post hoc, and it
#: moved a verdict in the direction the analysis was hoping for.
#:
#: **The phase pre-registered what to do here and the first draft contradicted
#: it.** ``MATCHED_EPOCH_POLICY``: *"If they agree the finding is settled from
#: two directions; if they disagree, the disagreement is the result and the
#: selection procedure is the factor."* They disagreed on photometric. The
#: pre-registered output is therefore that photometric is **unresolved**, not
#: that the narrower window wins.
#:
#: **Round 2 is still worth having.** It is a legitimate sensitivity analysis
#: and it produced a real finding: photometric's verdict is not robust to the
#: selection window, and geometric's is. What it is not is a second
#: measurement.
ROUND_2_IS_ROUND_1_TRUNCATED = {
    "measured": "2026-08-03",
    "why": (
        "rng_for(seed, fold, epoch) derives views from the coordinates alone; "
        "the head resets on a fixed seed at a constant LR with no schedule "
        "keyed to max_epochs; run_fold uses patience only to break. So "
        "epoch n's fit is independent of the budget"
    ),
    "consequence": "round 2 re-selects over round 1's own trajectories",
    "identical_fits": 23,
    "total_fits": 35,
    "window_chosen_post_hoc": (
        "three came from round 1's curves, after round 1's verdicts were "
        "known -- a data-dependent restriction of the candidate set"
    ),
    "pre_registered_rule": (
        "MATCHED_EPOCH_POLICY: if they disagree, the disagreement is the "
        "result and the selection procedure is the factor"
    ),
    "outcome": "photometric is UNRESOLVED, not neutral",
    "still_worth_having": (
        "a legitimate sensitivity analysis, and it found that photometric's "
        "verdict is not robust to the window while geometric's is"
    ),
}

#: **[MEASURED 2026-08-03] The mechanism can only be evidenced by the LONG
#: budget, and the primary round cannot speak to it at all.**
#:
#: The mechanism claim is: *the head overfits 152 samples within about three
#: epochs and augmentation does not prevent it -- if augmentation were
#: regularising, the augmented arms' inner-val optimum would arrive LATER than
#: identity's.*
#:
#: In ``matched3`` the budget is **3**, so "every optimum is at epoch 1-3" is
#: **true by construction**. It is not a measurement, and quoting it as one
#: would be reading a constant as a result -- the same error as PLAN §4.4's
#: ``white_fraction_max: 0.0``, which was a patch-geometry constant read as a
#: pixel fact, and as ``centroid_y`` being satisfied by construction under a
#: bottom-band prior.
#:
#: **Only ``selected30`` can evidence it**, because only there was a late
#: optimum reachable. It was reachable and mostly did not happen: 18 of 30
#: augmented folds selected epoch <= 3, against identity's 5 of 5 at epoch 1.
#: The twelve that went later did so on draw luck, not on delayed
#: overfitting -- which is the same finding from the other side.
#:
#: So ``selected30`` is **not kept merely as the record that the correction
#: mattered.** It carries the sole evidence for the mechanism underneath the
#: primary round's numbers, and dropping it would leave the verdict standing
#: on the deltas alone.
MECHANISM_NEEDS_THE_LONG_BUDGET = {
    "measured": "2026-08-03",
    "claim": (
        "the head overfits 152 samples within about three epochs and "
        "augmentation does not prevent it. If augmentation were regularising, "
        "the augmented arms' inner-val optimum would arrive LATER than "
        "identity's"
    ),
    "status": "WITHDRAWN -- see MECHANISM_CLAIM_WITHDRAWN",
    "matched3_cannot": (
        "its budget IS 3, so 'every optimum is at epoch 1-3' is true by "
        "construction. A constant read as a result -- the same error as "
        "PLAN §4.4's white_fraction_max. That part stands"
    ),
}

#: **[WITHDRAWN 2026-08-03] The mechanism claim is refuted by its own
#: criterion, and it was asserted in a passing test.**
#:
#: The claim: *"if augmentation were regularising, the augmented arms'
#: inner-val optimum would arrive LATER than identity's, and it does not."*
#: It was called confound-free and made the verdict's mechanism.
#:
#: **It does arrive later.** Mean selected epoch, long budget: identity
#: **1.00**; augmented **7.13**. Per arm 9.2, 5.6, 4.2, 5.0, 7.6, 11.2 --
#: every augmented arm between 4x and 11x identity. 18 of 30 augmented folds
#: arrive after epoch 1 and none arrives before, because epoch 1 is the floor.
#:
#: The supporting test asserted *"most folds select epoch <= 3"* -- 18 of 30,
#: true -- and read that as "not later". **A majority being early is not the
#: same quantity as the distribution not shifting**, and the second is what
#: the claim needed. R2 again, in a test I wrote to guard the claim.
#:
#: ----------------------------------------------------------------------
#: BOTH BRANCHES KILL IT
#: ----------------------------------------------------------------------
#: * If the late selections are **signal**, the augmented optimum arrives
#:   later and the claim is simply false.
#: * If they are **draw luck** -- which ``SELECTION_CONFLATES_DRAW_QUALITY``
#:   asserts -- then ``selected_epoch`` carries no information about when the
#:   true optimum arrives, and calling the epoch distribution a "confound-free
#:   tell" is false.
#:
#: There is no reading of the epoch distribution that supports the claim, and
#: the phase asserted both premises at once.
#:
#: ----------------------------------------------------------------------
#: WHAT SURVIVES, AND IT IS LESS
#: ----------------------------------------------------------------------
#: The training curves show overfitting under augmentation directly, without
#: any selection argument: arm 1, seed 1337, fold 0 -- train loss 0.195 ->
#: 0.068 by epoch 3 while inner-val MSE RISES 0.491 -> 0.559; fold 4 by epoch
#: 30 -- train loss 0.0087, inner-val MSE 0.70, PCC -0.114.
#:
#: **That shows the augmented arm overfits. It does not show augmentation
#: failed to DELAY overfitting**, which needs identity's curve at the same
#: epochs to compare against and the phase did not record one. And it is two
#: folds of one arm.
#:
#: So the honest position: **augmented arms score below identity (measured)
#: and augmented arms overfit (measured, thinly). The causal link between
#: them is not established by this phase.** The verdict on the deltas stands;
#: the story underneath it does not.
MECHANISM_CLAIM_WITHDRAWN = {
    "withdrawn": "2026-08-03",
    "claim": (
        "if augmentation were regularising the augmented optimum would arrive "
        "later than identity's, and it does not"
    ),
    "refuted_by": {
        "identity_mean_selected_epoch": 1.00,
        "augmented_mean_selected_epoch": 7.13,
        "per_arm_means": [9.2, 5.6, 4.2, 5.0, 7.6, 11.2],
        "folds_later_than_epoch_1": "18/30",
        "folds_earlier": "0/30 -- epoch 1 is the floor",
    },
    "how_the_test_missed_it": (
        "it asserted 'most folds select epoch <= 3' (18 of 30, true) and read "
        "that as 'not later'. A majority being early is a different quantity "
        "from the distribution not shifting"
    ),
    "both_branches_kill_it": (
        "if the late selections are signal the claim is false; if they are "
        "draw luck then selected_epoch says nothing about the true optimum "
        "and the tell is not confound-free. The phase asserted both premises"
    ),
    "what_survives": (
        "the curves show the augmented arm overfits (arm 1 fold 0: train loss "
        "0.195 -> 0.068 by epoch 3 while inner-val MSE rises 0.491 -> 0.559). "
        "They do NOT show augmentation failed to delay it -- that needs "
        "identity's curve at the same epochs, which was not recorded -- and "
        "it is two folds of one arm"
    ),
    "honest_position": (
        "augmented arms score below identity, and augmented arms overfit. The "
        "causal link between them is not established by this phase"
    ),
}

#: **[MEASURED 2026-08-03] Two of the phase's nulls are n=5 artefacts, and the
#: paired half of the claim criterion was never run.**
#:
#: **The seed count.** PLAN §4.12 says the seed count follows from what the
#: arm needs to claim. Recomputed at ten seeds, two nulls become claimable:
#:
#:     photometric vs identity (matched3)  -0.0156 : n5 0.0217 -> n10 0.0154
#:     region_photometric 1->4 (matched3)  -0.0207 : n5 0.0289 -> n10 0.0204
#:
#: Both are the verdicts the phase leans on -- "photometric is unresolved"
#: and "region protection makes no difference". At ten seeds the first becomes
#: *claimably worse* and the second becomes *claimably against protection*.
#: **Neither null is a property of the data; both are properties of n=5.**
#:
#: **The paired BCa was promised and not run.** PLAN §4.3's claim criterion
#: has TWO conditions, and this phase recorded only the second (the combined
#: seed threshold). ``NOT_A_SEARCH["carried_over"]`` promises *"per-seed
#: paired BCa against 0.2520, phase7b.paired_comparison"*; exit criterion 6
#: requires it; no interval appears anywhere in the phase.
#:
#: The direction matters: an unpaired threshold on a paired design is
#: **conservative for the "claimably worse" verdicts** -- they would survive a
#: paired test -- and **anti-conservative for every null**. So the phase's
#: negative results are safe and its null results are not reportable until it
#: lands.
OUTSTANDING_BEFORE_THE_NULLS_ARE_REPORTABLE = {
    "measured": "2026-08-03",
    "nulls_that_flip_at_ten_seeds": {
        "photometric_vs_identity": {"delta": -0.0156, "n5": 0.0217, "n10": 0.0154,
                                    "round": "matched3"},
        "region_awareness_photometric": {"delta": -0.0207, "n5": 0.0289,
                                         "n10": 0.0204, "round": "matched3"},
    },
    # **[MEASURED 2026-08-03] BOTH flips are in the SENSITIVITY round, and
    # nothing in the primary round flips at all.** Recomputed across all nine
    # comparisons in both rounds: in `selected30` the six against identity
    # already claim at five seeds and the three nulls stay null at ten. So
    # ten seeds cannot change a single reported verdict -- every verdict this
    # phase reports comes from the primary round.
    #
    # That is what makes the seed count a question about the ROUND
    # DISAGREEMENT rather than about any one arm. Photometric is unresolved
    # because the rounds disagree (-0.0451 claimable against -0.0156 null),
    # and the region contrast disagrees in SIGN (+0.0065 against -0.0207).
    # A disagreement cannot be settled by measuring one side better, so
    # raising only the round whose nulls flip would be promoting the
    # sensitivity analysis by a second route -- exactly what
    # ROUND_2_IS_ROUND_1_TRUNCATED was written about.
    "primary_round_flips_nothing": (
        "in selected30 no verdict changes at ten seeds: the six against "
        "identity claim at five, the three nulls stay null at ten. Both "
        "flips are matched3's, so the seed count is a question about the "
        "disagreement BETWEEN rounds, not about an arm"
    ),
    "projection_assumes_the_sd_holds": (
        "n10 thresholds are the n5 SDs divided by root two. Ten seeds "
        "re-estimates the SD, so a flip is a prediction to test, not a "
        "result to bank"
    ),
    "paired_bca": {
        "promised_by": "NOT_A_SEARCH['carried_over'] and exit criterion 6",
        # **[CORRECTED 2026-08-04] It landed, and both predictions below were
        # wrong -- the second harmlessly, the first completely.**
        "status": "RUN 2026-08-04 -- see PAIRED_BCA_WITHDREW_EVERY_VERDICT",
        "effect_on_negatives": (
            "PREDICTED 'conservative; they survive'. WRONG: all six were "
            "withdrawn. The reasoning -- that an unpaired threshold overstates "
            "variance -- is correct about condition 2, which was never the "
            "binding constraint on them. Condition 1 was, and it was not "
            "computed"
        ),
        "effect_on_nulls": (
            "PREDICTED 'anti-conservative; not reportable until it lands'. It "
            "landed and they stayed null, so the caution cost nothing -- but "
            "for the wrong reason: they fail condition 1 as well"
        ),
        "the_error_shape": (
            "reasoning confidently about the direction of a test's effect "
            "instead of running a test that needed no new fits"
        ),
    },
    "consequence": (
        "the claimably-worse verdicts stand. Every NULL in this phase is "
        "provisional on both the paired test and a seed count chosen from "
        "what it needs to claim"
    ),
}


#: **[DECIDED 2026-08-03] Ten seeds for arms 0, 1 and 4, in BOTH rounds.**
#:
#: **The arms.** Only three arms appear in the two contrasts whose verdicts
#: turn on the seed count: identity and photometric (arms 0 and 1) carry
#: ``photometric_vs_identity``, and photometric and region-photometric (arms 1
#: and 4) carry ``region_awareness_photometric``. The geometric arms are not
#: raised: their deltas clear the five-seed threshold by three to four times,
#: so ten seeds cannot change them and PLAN §4.12's rule -- the seed count
#: follows from what the arm needs to claim -- says do not spend the fits.
#:
#: **Both rounds, and this is the part that is not obvious.** The flips are
#: *matched3's*; the primary round flips nothing
#: (``OUTSTANDING_BEFORE_THE_NULLS_ARE_REPORTABLE``). The temptation is
#: therefore to raise only ``matched3``, which is precisely the wrong move:
#: photometric is UNRESOLVED because the two rounds DISAGREE, and comparing a
#: ten-seed estimate against a five-seed one does not settle a disagreement,
#: it just re-weights it. Raising only the round whose nulls flip would
#: promote the sensitivity analysis by a second route --
#: ``ROUND_2_IS_ROUND_1_TRUNCATED`` again, arrived at from the other side.
#:
#: **The primary round is run first regardless**, because it is the round
#: every reported verdict comes from.
#:
#: **The first five seeds are re-run, not reused.** They are the same fits --
#: ``rng_for(seed, fold, epoch)`` and a fixed head seed make them
#: reproducible -- so their prediction files must come back BIT-IDENTICAL to
#: the five already on disk. That is a determinism re-verification obtained
#: as a side effect, and it is the reason to run ten in one config rather
#: than five more in a second: one run directory, one metrics.json, one SD
#: computed over ten seeds in the place that reports it, and a free gate-1
#: check on the way. Merging two directories would have none of those and
#: would invite pairing a v3 vector with a v4 one.
SEEDS_10 = list(SEED_POOL[:10])

#: Arms 0, 1 and 4 -- the only arms in the two contrasts that turn on n.
TEN_SEED_ARMS = (0, 1, 4)

TEN_SEED_ROUND = {
    "decided": "2026-08-03",
    # **[SUPERSEDED 2026-08-04] Designed, generated, and never run.** The
    # paired BCa withdrew every verdict, and condition 1 is anti-monotone in
    # the seed set, so ten seeds fails by construction.
    # See TEN_SEED_ROUND_NOT_RUN. The design is kept because the reasoning
    # that produced it -- raise both rounds, not just the one whose nulls
    # flip -- is still right about seed counts generally.
    "status": "SUPERSEDED -- see TEN_SEED_ROUND_NOT_RUN. The configs exist "
              "and must not be launched",
    "arms": list(TEN_SEED_ARMS),
    "seeds": list(SEEDS_10),
    "rounds": ["selected30", "matched3"],
    "run_order": "selected30 first -- it is the round every verdict comes from",
    "why_these_arms": (
        "arms 0/1 carry photometric_vs_identity and arms 1/4 carry "
        "region_awareness_photometric; those are the only two contrasts whose "
        "verdict turns on the seed count. The geometric arms clear the "
        "five-seed threshold by 3-4x and are not raised"
    ),
    "why_both_rounds": (
        "the flips are matched3's and the primary round flips nothing, so the "
        "question is the DISAGREEMENT between rounds -- which a ten-seed "
        "estimate against a five-seed one cannot settle. Raising only "
        "matched3 would promote the sensitivity analysis by a second route"
    ),
    "first_five_must_reproduce": (
        "bit-identical prediction files against the five already on disk; the "
        "fits are budget- and count-independent, so this is a determinism "
        "re-verification obtained for free"
    ),
    "not_two_directories": (
        "ten seeds in ONE config, not five more merged with the existing "
        "five: one SD over ten seeds computed where it is reported, and no "
        "chance of pairing a v3 vector with a v4 one"
    ),
}


def config_name(arm: dict, prefix: str) -> str:
    """``p7c_1_photometric``, ``p7c_m3_1_photometric``, ``p7c_s10_1_...``.

    The prefix goes after ``p7c_`` rather than at the end, so the rounds of
    one arm do not sort apart in a directory listing.
    """
    if not prefix:
        return arm["name"]
    return arm["name"].replace("p7c_", f"p7c_{prefix}", 1)


def config_stems() -> set[str]:
    """Every arm config stem this phase ships, across all four rounds."""
    return {
        config_name(arm, round_["prefix"])
        for round_ in CONFIG_ROUNDS
        for arm in arms_at(round_["n_seeds"], only=round_["only"])
    }


def arms_at(n_seeds: int, *, only: tuple = ()) -> list[dict]:
    """The arm list at a different seed count, optionally filtered by index.

    Derived from ``arms()`` so the ten-seed configs cannot drift from the
    five-seed ones in any factor -- only the seed list differs.
    """
    if n_seeds > len(SEED_POOL):
        raise Phase7CError(
            f"{n_seeds} seeds requested and the pool holds {len(SEED_POOL)}"
        )
    seeds = list(SEED_POOL[:n_seeds])
    out = []
    for arm in arms():
        if only and arm["index"] not in only:
            continue
        entry = dict(arm)
        entry["seeds"] = list(seeds)
        out.append(entry)
    if only and len(out) != len(only):
        raise Phase7CError(
            f"asked for arms {sorted(only)} and matched {len(out)}"
        )
    return out

#: **[CORRECTED 2026-08-03] The two region tests AGREE in direction at the
#: matched budget. They disagreed at the long one.**
#:
#: Under the pre-registration's own convention -- ``comparisons()`` orders
#: each pair ``(a, b)`` and the delta is ``b - a``, so for both region tests
#: the delta is **protected minus unprotected** -- the signs are:
#:
#:     selected30:  5v6 -0.0195   1v4 +0.0065   -> OPPOSITE
#:     matched3:    5v6 -0.0001   1v4 -0.0207   -> SAME (both against protection)
#:
#: So the framing "the two region tests still disagree in sign" describes
#: round 1, not round 2. At the matched budget both point weakly against
#: protection, one of them at essentially zero (-0.0001, four decimal places
#: from identical).
#:
#: **Neither is claimable in either round**, so this changes the framing and
#: not the verdict: region-aware protection makes no difference. Recorded
#: because a sign flip between rounds is the kind of thing that gets quoted
#: from the wrong round, and because reading the delta in the opposite order
#: for one test and not the other is exactly how it happens.
REGION_TESTS_AGREE_AT_THE_MATCHED_BUDGET = {
    "corrected": "2026-08-03",
    "convention": (
        "comparisons() orders each pair (a, b) and the delta is b - a, which "
        "for both region tests is PROTECTED minus UNPROTECTED"
    ),
    "selected30": {"region_awareness": -0.0195,
                   "region_awareness_photometric": 0.0065,
                   "direction": "opposite"},
    "matched3": {"region_awareness": -0.0001,
                 "region_awareness_photometric": -0.0207,
                 "direction": "same -- both weakly against protection"},
    "verdict_unchanged": (
        "neither is claimable in either round; region-aware protection makes "
        "no difference. This changes the framing, not the finding"
    ),
}

#: **[MEASURED 2026-08-03] Under augmentation, best-checkpoint selection
#: partly selects the luckiest DRAW rather than the best-converged model.**
#:
#: Each epoch is a different random perturbation, so a fold's inner-val score
#: at epoch *n* is a function of two things -- how good the model is, and how
#: kind that epoch's transform draw was to those 38 patients. Taking the
#: minimum over 30 epochs therefore selects on both. **"Best epoch" conflates
#: model quality with draw quality.**
#:
#: Same class as gate 4's noise-selection, in a different guise: there,
#: patience fired on whichever side of a 2.3e-05 difference a value landed;
#: here, selection lands on whichever draw flattered the inner-val fold.
#:
#: **The direction is not benign, and that is why it matters.** A late epoch
#: wins only if its draw made inner-val look good -- and by epoch 30 the model
#: has train loss 0.0087 and inner-val PCC -0.114, so the luck is
#: inner-val-specific and does not transfer to the test fold. Selecting late
#: therefore reports a heavily overfit model's predictions: **the confound
#: penalises the augmented arms**, which are the only ones that can select
#: late. Arm 0 is deterministic and always selects epoch 1.
#:
#: So it cannot be waved through as "would not change the direction". It could
#: be manufacturing part of a gap that is being reported as a negative result.
#:
#: **What it cannot be engineered away.** Arm 0 has NO draw variance at all,
#: so a deterministic arm and a stochastic one never have matched selection
#: procedures. That is inherent to the comparison and belongs in the write-up
#: as a stated limitation rather than being designed around.
SELECTION_CONFLATES_DRAW_QUALITY = {
    "measured": "2026-08-03",
    "what": (
        "each epoch is a different perturbation, so best-epoch selection "
        "picks on model quality AND draw quality together"
    ),
    "same_class_as": "gate 4's noise-selection, in a different guise",
    "direction_predicted": (
        "AGAINST the augmented arms: a late epoch wins on an inner-val-"
        "specific lucky draw, and the model there is heavily overfit, so the "
        "reported OOF is degraded. Only augmented arms can select late"
    ),
    #: **[MEASURED 2026-08-03] The predicted direction is not supported.**
    #:
    #: If late selection systematically degraded the reported OOF, truncating
    #: the window would RAISE every augmented arm, and most where selection
    #: was latest. Neither holds. The mean move is **+0.0054** -- an order of
    #: magnitude below the deltas being adjudicated -- and
    #: ``6_whole_image_full``, the arm with the MOST late-selecting folds
    #: (three past epoch 3, including 29 and 19), moved **-0.0171**: for those
    #: folds the late epochs gave BETTER out-of-fold predictions than the
    #: pre-divergence ones.
    #:
    #: So the mechanism is coherent and its magnitude and sign are not
    #: established. The confound exists; whether it materially biased round 1
    #: is unmeasured, and the one arm that tests it hardest points the other
    #: way. **That removes the justification for treating the truncated round
    #: as a correction rather than as a sensitivity analysis.**
    "direction_measured": {
        "mean_move": 0.0054,
        "most_late_selecting_arm": "6_whole_image_full",
        "its_move": -0.0171,
        "verdict": (
            "NOT SUPPORTED. Truncation should have raised every augmented arm "
            "and most where selection was latest; the arm with the most late "
            "folds moved down"
        ),
    },
    "evidence": "arm 1 selected epochs [1, 3, 30, 11, 1]; arm 6 [1, 5, 29, 19, 2]",
    "arm_0_is_immune": "deterministic features, so no draw variance, always epoch 1",
    "not_engineerable_away": (
        "a deterministic arm and a stochastic one cannot have matched "
        "selection procedures. A stated limitation, not a design problem"
    ),
    "addressed_by": "MATCHED_EPOCH_POLICY -- best of three rather than best of thirty",
}

#: **[DECIDED 2026-08-03] A second round at a matched, pre-divergence epoch.**
#:
#: Not because the direction is in doubt -- the mechanism above is
#: confound-free -- but because ``SELECTION_CONFLATES_DRAW_QUALITY`` runs
#: AGAINST the augmented arms, so it could be inflating a negative result that
#: is load-bearing for the exhaustion argument. A finding that survives two
#: selection procedures cannot be attacked on the selection.
#:
#: **Three epochs, not thirty.** The curves put divergence at about epoch 3,
#: so every arm is compared before it overfits, and selection shrinks from
#: best-of-thirty to **best-of-three** -- an order of magnitude less draw
#: luck, with every arm drawing the same number of times.
#:
#: **It is a reduction, not an elimination**, and the record says so: the
#: frozen ``run_fold`` always keeps the best epoch, so "no selection at all"
#: is not reachable through config, and arm 0 has no draws to select among
#: regardless. Describing this as removing the confound would be overclaiming.
#:
#: **Both rounds are reported.** If they agree the finding is settled from two
#: directions; if they disagree, the disagreement is the result and the
#: selection procedure is the factor.
MATCHED_EPOCH_POLICY = {
    "decided": "2026-08-03",
    "max_epochs": 3,
    "patience": 3,
    "monitor": "inner_val_mse",
    "why_three": "the curves put divergence at about epoch 3",
    "reduces_not_removes": (
        "best-of-three rather than best-of-thirty. run_fold always keeps the "
        "best epoch, so no-selection is not reachable through config"
    ),
    "both_rounds_reported": True,
}


#: **[MEASURED 2026-08-03] 12 of 30 augmented folds report a model that saw
#: one view per image -- and that is a RESULT here, where it was a DEFECT in
#: v1. The distinction is not obvious and a careful reader missed it.**
#:
#: An adversarial review put it this way: the phase declared v1 void because
#: *"an arm that stops at epoch 1 has not run the procedure its config names"*
#: (PLAN R7's tenth instance), and the primary round still reports epoch-1
#: fits for 40% of its augmented folds -- so the primary round is largely made
#: of the configuration the phase itself voided.
#:
#: **The fits are the same; their evidential status is not.**
#:
#: * **v1**: ``max_epochs 40, patience 5`` -- the loop STOPPED at epoch 1. The
#:   remaining 39 epochs never happened, so nothing was chosen and nothing was
#:   compared. The procedure did not run.
#: * **selected30**: thirty epochs ran, and inner-validation SELECTED epoch 1
#:   as the best of thirty. The procedure ran and its answer was "epoch 1" --
#:   which is a measurement that more augmented views made that fold worse.
#:
#: So the epoch-1 folds are not void here. They are the finding, at fold
#: resolution: on 40% of augmented folds, additional views did not help.
#:
#: **The counts, because the fraction is the thing that reads badly:**
#: ``selected30`` 12/30 (40%), ``matched3`` 19/30 (63%). That the sensitivity
#: round raises it to nearly two-thirds is a further reason it is not the
#: primary -- its shorter window has fewer epochs to prefer over epoch 1.
#:
#: **[UNRESOLVED] A rationale tension the review also found, and it is real.**
#: ``test_early_stopping_cannot_fire_in_any_arm`` asserts ``max_epochs >= 30``
#: with the message *"fewer than 30 epochs is a thin sample of the transform
#: distribution"*. Taken at face value that says the 3-epoch round does not
#: sample the transform distribution either -- so the sensitivity analysis is
#: not a weaker test of the same thing, it is a test of something else. Either
#: the 30 is a real requirement and ``matched3`` cannot speak to augmentation,
#: or it is not and round 1's config rationale overstates. **The phase asserts
#: both and does not resolve it**; recorded rather than papered over, because
#: it bears directly on how much weight the sensitivity round can carry.
EPOCH_1_SELECTED_IS_NOT_EPOCH_1_FORCED = {
    "measured": "2026-08-03",
    "counts": {"selected30": "12/30 (40%)", "matched3": "19/30 (63%)"},
    "v1_was_void_because": (
        "max_epochs 40, patience 5 -- the loop STOPPED at 1. The remaining "
        "epochs never happened, so nothing was chosen or compared"
    ),
    "selected30_is_not_void_because": (
        "thirty epochs ran and inner-val SELECTED epoch 1 as best of thirty. "
        "That is a measurement: on those folds, more views made it worse"
    ),
    "why_it_reads_badly": (
        "the reported fit is byte-identical to the voided one; only the "
        "mechanism that chose it differs, and nothing in the number says so"
    ),
    "unresolved_rationale_tension": (
        "the suite asserts max_epochs >= 30 because 'fewer than 30 epochs is "
        "a thin sample of the transform distribution', while the sensitivity "
        "round uses 3. Either 30 is required and matched3 cannot speak to "
        "augmentation, or it is not and round 1's rationale overstates. The "
        "phase asserts both"
    ),
}

#: **[MEASURED 2026-08-03] The two region tests are not independent, not
#: equally powered, and test a small intervention in a diluted contrast. "Null
#: on both tests" reads as corroboration and is not.**
#:
#: Three things, and the second is the sharpest because this phase found it
#: itself and did not carry it through:
#:
#: 1. **The intervention is small by construction.**
#:    ``REGION_AWARE["inside_strength"] = 0.25`` applied to a jitter of
#:    0.2/0.2/0.2/0.05 -- a fourfold reduction of an already mild
#:    perturbation, over 22 of the 27 boxes. Protection changes little because
#:    there was little to protect against.
#: 2. **The unprotected side is diluted by the white pad** -- the same object
#:    that produced R2's eighth instance in this very phase.
#:    ``augment_sheet.MOVED_OUTSIDE_WAS_DILUTED_BY_THE_PAD`` measured that the
#:    strongly-augmented region is 25.34% white pad at the median aspect
#:    ratio and that flat white barely moves under photometric jitter. The
#:    unprotected set is glabella, canthi, orbits and everything the boxes
#:    miss -- much of which is that pad. **The phase found the dilution in its
#:    diagnostic and did not carry the implication into the arm design.**
#: 3. **Geometry is not modulated at all** (``augment.py``: a global affine
#:    cannot be spatially modulated). So arm 5 against arm 6 asks whether
#:    modulating the WEAKER component matters while the stronger, unmodulated
#:    one runs identically on both sides -- and pays those arms' larger SDs
#:    for the privilege.
#:
#: So the 5-vs-6 and 1-vs-4 tests are not two independent looks at one
#: question. They are the same factor in two dilutions, unequally powered, and
#: 1-vs-4 flips to claimable at ten seeds
#: (``OUTSTANDING_BEFORE_THE_NULLS_ARE_REPORTABLE``).
#:
#: **The reportable wording is the one round 1 already had**: *the
#: the design question raised at supervision gets no answer from this data, in either
#: direction.* Telling a room their design makes no difference, on an
#: underpowered null in a diluted contrast, is both wrong and costly.
REGION_TESTS_ARE_NOT_INDEPENDENT = {
    "measured": "2026-08-03",
    "intervention_is_small": (
        "inside_strength 0.25 on a jitter of 0.2/0.2/0.2/0.05 -- a fourfold "
        "reduction of an already mild perturbation, over 22 of 27 boxes"
    ),
    "contrast_is_diluted": (
        "the unprotected side is largely white pad (25.34% of frame at the "
        "median AR) which barely moves under photometric jitter -- the "
        "dilution this phase found in its own contact sheet and did not carry "
        "into the arm design"
    ),
    "geometry_is_not_modulated": (
        "a global affine cannot be spatially modulated, so 5-vs-6 modulates "
        "the weaker component while the stronger runs on both sides"
    ),
    "consequence": (
        "not two independent tests. The same factor in two dilutions, "
        "unequally powered, and 1-vs-4 flips to claimable at ten seeds"
    ),
    "reportable_wording": (
        "the supervision design question gets no answer from this data, in "
        "either direction -- NOT 'region protection makes no difference'"
    ),
}


# --------------------------------------------------------------------------
# Exit criterion 6 -- the per-seed paired BCa
# --------------------------------------------------------------------------

#: **[DECIDED 2026-08-03] This phase evaluated ONE of PLAN §4.3's two
#: conditions and reported its verdicts as though it had evaluated both.**
#:
#: §4.3 claims a delta only if BOTH hold: (1) the paired BCa CI over patients
#: excludes zero, and (2) the delta exceeds combined seed uncertainty. Every
#: 7C verdict -- the six against identity and the three between arms -- came
#: from condition 2 alone. ``NOT_A_SEARCH["carried_over"]`` promised the paired
#: BCa in writing, and exit criterion 6 required it. It was never run.
#:
#: **The two conditions fail in opposite directions, which is why the gap is
#: not symmetric:**
#:
#: * The conditions are ANDed, so adding condition 1 can only ever REMOVE
#:   claims. The six "claimably worse" verdicts are the ones genuinely at
#:   risk: each must now show five intervals excluding zero, all one way.
#: * The three nulls cannot be rescued by it. They failed condition 2, and no
#:   result from condition 1 changes an AND that is already false.
#:
#: **So the nulls' binding constraint is condition 2 -- and its threshold is
#: computed UNPAIRED.** ``combined_claimable_delta`` adds two independent seed
#: bands in quadrature, but the arms share patients, folds and inner-val splits
#: seed for seed; a paired contrast has strictly less variance than that sum.
#: The threshold is therefore too LARGE, which is conservative for a claim and
#: **anti-conservative for a null** -- it makes "no difference" the easy
#: answer, and 7C returned that answer three times.
#:
#: **What the paired intervals are for, then, is not rescuing the nulls but
#: diagnosing them.** If a null's five intervals consistently exclude zero
#: while condition 2 fails, the null is a POWER failure, not an absence -- and
#: that is exactly the distinction ``REGION_TESTS_ARE_NOT_INDEPENDENT`` and
#: ``OUTSTANDING_BEFORE_THE_NULLS_ARE_REPORTABLE`` already insisted on in
#: words. This supplies the evidence for it.
#:
#: **Not a new threshold.** Replacing condition 2's unpaired combination with
#: a paired one after seeing which way the nulls went is exactly the move
#: ``ROUND_2_IS_ROUND_1_TRUNCATED`` was written about. §4.3's rule stands as
#: written; what changes is that both of its conditions get computed and
#: reported, and a null is labelled underpowered rather than empty.
#: **Every round of arm configs this phase ships, as data.**
#:
#: Kept here rather than in the generator script because which rounds exist is
#: a pre-registration fact -- the suite checks the shipped directory against
#: this, so a stray ``p7c_*`` config that is neither an arm nor a declared
#: non-arm fails on the laptop instead of sitting there looking launchable.
#:
#: Declared here, below ``MATCHED_EPOCH_POLICY``, because it references it.
CONFIG_ROUNDS = (
    {"prefix": "", "policy": EPOCH_POLICY, "n_seeds": 5, "only": ()},
    {"prefix": "m3_", "policy": MATCHED_EPOCH_POLICY, "n_seeds": 5, "only": ()},
    {"prefix": "s10_", "policy": EPOCH_POLICY, "n_seeds": 10,
     "only": TEN_SEED_ARMS},
    {"prefix": "m3s10_", "policy": MATCHED_EPOCH_POLICY, "n_seeds": 10,
     "only": TEN_SEED_ARMS},
)


PAIRED_BCA_IS_THE_MISSING_CONDITION = {
    "decided": "2026-08-03",
    "gap": (
        "every 7C verdict came from PLAN §4.3 condition 2 alone; condition 1, "
        "the per-seed paired BCa, was promised in NOT_A_SEARCH['carried_over'] "
        "and required by exit criterion 6, and was never run"
    ),
    "needs_no_new_fits": (
        "the per-seed out-of-fold vectors are on disk, one CSV per arm per "
        "seed, written by phase3.write_outputs"
    ),
    "and_can_only_remove": (
        "the conditions are ANDed, so condition 1 can only withdraw claims -- "
        "the six against-identity verdicts are at risk, the three nulls cannot "
        "be rescued"
    ),
    "nulls_are_bound_by_condition_2": (
        "combined_claimable_delta adds two seed bands in quadrature as though "
        "independent, but the arms share patients, folds and inner-val splits "
        "seed for seed; the paired contrast has less variance, so the "
        "threshold is too large -- conservative for a claim, ANTI-conservative "
        "for a null"
    ),
    "what_the_intervals_decide": (
        "not whether a null is claimable, but whether it is a POWER failure or "
        "an absence: five intervals excluding zero under a failed condition 2 "
        "is an underpowered contrast, not 'no difference'"
    ),
    "not_a_new_threshold": (
        "§4.3's rule stands as written. Swapping condition 2's combination for "
        "a paired one after seeing which way the nulls went is the move "
        "ROUND_2_IS_ROUND_1_TRUNCATED was written about"
    ),
}

#: **[PASTED 2026-08-03] Which runs on the cluster ARE the primary round.**
#:
#: Three rounds of these arms exist under the same config stems: v1 (the
#: augmenter never ran, ``backbone_config`` dropped the key) and v2 (every arm
#: stopped at epoch 1, ``ONE_EPOCH_IS_NOT_AUGMENTATION``) are both VOID and
#: both left ``seed_*__predictions.csv`` behind. ``declare_ladder_inputs.py``
#: refused all 35 globs on that ambiguity rather than choosing, which is the
#: only reason this is a decision being recorded and not a silent pairing
#: against a run that never applied its policy.
#:
#: **Pinned by JOB ID.** Not by stem -- all three rounds share it, which is
#: what made the globs ambiguous in the first place. And not by SHA: the job
#: id names the ROUND, while the SHA names the CODE, so pinning the SHA would
#: need editing whenever a future round ran at a different commit, for a fact
#: that has nothing to do with which round is meant. The SHA stays globbed.
#:
#: **Written out rather than derived.** Six of the seven follow
#: ``p7c-arm<N>-<descriptive-name>-v3``; arm 0 is bare ``p7c-arm0-v3`` with no
#: descriptive part. A derivation rule would carry a special case for arm 0
#: that a later arm could silently break, and being wrong here does not fail
#: loudly -- it pairs against a void run and returns plausible intervals.
#:
#: **The matched-epoch round will need the same treatment.** Its ``p7c_m3_*``
#: stems are unique today, so its globs would resolve on a single match and
#: look fine; they stop being unique the moment any m3 arm is re-run. A
#: ``matched3`` paired config must pin its job ids rather than inherit that
#: accident.
V3_JOB_IDS = {
    0: "p7c-arm0-v3",
    1: "p7c-arm1-photometric-v3",
    2: "p7c-arm2-geometric-v3",
    3: "p7c-arm3-geometric-rotation-v3",
    4: "p7c-arm4-region-photometric-v3",
    5: "p7c-arm5-region-full-v3",
    6: "p7c-arm6-whole-image-full-v3",
}

#: The arm every other arm is measured against, and the ``STAGE_7C_RESULTS``
#: key for it. Identity IS the 0.2520 baseline reproduced through the
#: augmenting backbone (``ARM_0_SCOPE``), so "against identity" and "against
#: the baseline" are the same comparison here.
IDENTITY_ARM = "0_identity"

#: **Each arm's five vectors are declared as five FILE inputs**, named
#: ``oof_<arm>_seed_<seed>`` -- 35 in all. The convention, and the reason it
#: is per file rather than per run directory, is ``phase7b``'s verbatim: a run
#: directory carries ``code/``, a git worktree, so its rollup would cover the
#: source tree and move whenever anything in it was touched.
OOF_INPUT_PREFIX = "oof_"
OOF_SEED_SEPARATOR = "_seed_"


def result_key(name: str) -> str:
    """``p7c_1_photometric`` -> ``1_photometric``, the results key."""
    prefix = "p7c_"
    return name[len(prefix):] if name.startswith(prefix) else name


def oof_paths_from_inputs(declared: dict) -> dict:
    """``{arm_key: {seed: path}}`` for every declared OOF vector."""
    paths: dict[str, dict[int, str]] = {}
    for name, entry in declared.items():
        if not name.startswith(OOF_INPUT_PREFIX):
            continue
        arm, separator, seed = name[len(OOF_INPUT_PREFIX):].rpartition(
            OOF_SEED_SEPARATOR
        )
        if not separator or not arm or not seed.isdigit():
            raise Phase7CError(
                f"input {name!r} does not name an arm and a seed; the "
                f"convention is {OOF_INPUT_PREFIX}<arm>{OOF_SEED_SEPARATOR}<seed>"
            )
        paths.setdefault(arm, {})[int(seed)] = (
            entry["path"] if isinstance(entry, dict) else entry
        )
    return paths


def load_oof_vectors(paths_by_arm: dict, *, seeds=None) -> dict:
    """Every arm's per-seed OOF vector, realigned and cross-checked.

    Rows are keyed by patient id and re-sorted, never taken in file order --
    ``load_baseline_predictions`` learned that once and this reads 35 files
    instead of five.

    **The TRUTH column is checked across all 35, not just read from one.**
    Every file carries it, so a disagreement is detectable, and a
    disagreement would mean the arms were fitted against different labels --
    at which point the pairing is meaningless and every interval below would
    still come out plausible. ``STAGE_G_LABEL_FORMULATION`` is why that is a
    live possibility rather than a paranoid one: this project runs the same
    cell under ``mean`` and under other label formulations.
    """
    from pathlib import Path

    from .cluster_csv import PREDICTIONS_COLUMNS, read_cluster_csv
    from .eval.metrics import pcc
    from .train.phase3 import seed_variance

    wanted = sorted(int(seed) for seed in (SEEDS if seeds is None else seeds))
    patient_ids: list | None = None
    truth: list | None = None
    loaded: dict[str, dict] = {}

    for arm in sorted(paths_by_arm):
        available = paths_by_arm[arm]
        absent = [seed for seed in wanted if seed not in available]
        if absent:
            raise Phase7CError(
                f"arm {arm!r} has no vector for seeds {absent}; declared "
                f"{sorted(available)}. The comparison is per seed and needs "
                "every seed on both sides"
            )
        by_seed: dict[int, list] = {}
        for seed in wanted:
            path = Path(available[seed])
            if not path.is_file():
                raise Phase7CError(
                    f"arm {arm!r} seed {seed}: predictions missing at {path}"
                )
            rows = list(read_cluster_csv(path, expect=PREDICTIONS_COLUMNS))
            table = {
                int(row["patient_id"]): (
                    float(row["truth"]), float(row["prediction"])
                )
                for row in rows
            }
            if len(table) != len(rows):
                raise Phase7CError(
                    f"{path.name} lists {len(rows)} rows for {len(table)} "
                    "patients -- a repeated id would silently overwrite"
                )
            if patient_ids is None:
                patient_ids = sorted(table)
                truth = [table[pid][0] for pid in patient_ids]
            elif set(table) != set(patient_ids):
                only_here = sorted(set(table) - set(patient_ids))[:5]
                only_there = sorted(set(patient_ids) - set(table))[:5]
                raise Phase7CError(
                    f"arm {arm!r} seed {seed} covers a different patient set: "
                    f"extra {only_here}, missing {only_there}"
                )
            drift = max(
                abs(table[pid][0] - value)
                for pid, value in zip(patient_ids, truth)
            )
            if drift > 1e-9:
                raise Phase7CError(
                    f"arm {arm!r} seed {seed} disagrees with the first vector "
                    f"on the TRUTH column by up to {drift:.3e}. The arms were "
                    "not fitted against the same labels, so nothing here may "
                    "be paired -- resolve the label formulation first"
                )
            by_seed[seed] = [table[pid][1] for pid in patient_ids]

        pccs = [float(pcc(truth, by_seed[seed])) for seed in wanted]
        band = seed_variance(pccs)
        loaded[arm] = {
            "by_seed": by_seed,
            # **The seed order is recorded, not left to position.** ``wanted``
            # is sorted, which is not the order the configs declare seeds in
            # (1337, 2024, 7, ...), so a reader pairing ``pccs`` with the
            # config's seed list by index would mis-attribute four of five.
            "seeds": list(wanted),
            "pccs": pccs,
            "mean": band["mean"],
            "sd": band["sd"],
            "n_seeds": band["n_seeds"],
        }

    if patient_ids is None:
        raise Phase7CError(
            f"no {OOF_INPUT_PREFIX}<arm>{OOF_SEED_SEPARATOR}<seed> inputs "
            "declared; exit criterion 6 cannot be met by this run"
        )
    return {"patient_ids": patient_ids, "truth": truth, "arms": loaded}


#: Four decimals is what ``STAGE_7C_RESULTS`` records, so a reproduction can
#: differ by half a unit in the last place for rounding alone. Anything the
#: wrong run directory would cause is orders of magnitude larger.
RECORD_TOLERANCE = 2e-4


def verify_against_record(loaded: dict, *, round_label: str,
                          tolerance: float = RECORD_TOLERANCE) -> dict:
    """Refuse to pair vectors that are not the ones the phase reported.

    Every arm's mean AND its SD must reproduce ``STAGE_7C_RESULTS``. Phase 7B
    checked its baseline this way for one arm; the failure it guards against
    is worse here, because a wrong run directory among 35 files would still
    load, still pair, still produce intervals, and be about different fits.

    Checking the SD as well as the mean is the part that matters: two rounds
    of this phase share 23 of 35 fits (``ROUND_2_IS_ROUND_1_TRUNCATED``), so
    means alone are a weak discriminator between them and the spreads are not.
    """
    recorded = STAGE_7C_RESULTS["rounds"].get(round_label)
    if recorded is None:
        raise Phase7CError(
            f"no round {round_label!r} in STAGE_7C_RESULTS; known rounds are "
            f"{sorted(STAGE_7C_RESULTS['rounds'])}"
        )
    arms_recorded = recorded["arms"]
    checked, mismatched = [], []
    for arm in sorted(loaded["arms"]):
        if arm not in arms_recorded:
            raise Phase7CError(
                f"arm {arm!r} is not recorded in round {round_label!r}; "
                f"recorded arms are {sorted(arms_recorded)}"
            )
        observed = loaded["arms"][arm]
        expected = arms_recorded[arm]
        delta_mean = abs(observed["mean"] - expected["pcc"])
        delta_sd = abs(observed["sd"] - expected["sd"])
        entry = {
            "arm": arm,
            "observed_pcc": observed["mean"], "recorded_pcc": expected["pcc"],
            "observed_sd": observed["sd"], "recorded_sd": expected["sd"],
            "delta_pcc": round(delta_mean, 6), "delta_sd": round(delta_sd, 6),
        }
        checked.append(entry)
        if delta_mean > tolerance or delta_sd > tolerance:
            mismatched.append(entry)
    if mismatched:
        detail = "; ".join(
            f"{e['arm']}: pcc {e['observed_pcc']:.4f} vs {e['recorded_pcc']}, "
            f"sd {e['observed_sd']:.4f} vs {e['recorded_sd']}"
            for e in mismatched
        )
        raise Phase7CError(
            f"{len(mismatched)} of {len(checked)} arms do not reproduce round "
            f"{round_label!r} within {tolerance}: {detail}. Either the wrong "
            "run directories are declared, or STAGE_7C_RESULTS is stale -- "
            "resolve it before pairing anything"
        )
    return {"round": round_label, "tolerance": tolerance, "arms": checked}


def paired_matrix(loaded: dict, *, round_label: str, n_boot: int = 10000) -> dict:
    """All nine comparisons, each as a per-seed paired BCa.

    Six against identity and the three declared between-arm pairs. Every entry
    names which arm is subtracted from which, in words, because the sign
    convention on the region pairs has already been quoted backwards once:
    ``comparisons()`` puts the PROTECTED arm in ``b`` for both region tests,
    so both deltas are protected-minus-unprotected and a sign disagreement
    between them is a real disagreement rather than a bookkeeping one.
    """
    truth = loaded["truth"]
    arms_loaded = loaded["arms"]
    recorded = STAGE_7C_RESULTS["rounds"][round_label]

    if IDENTITY_ARM not in arms_loaded:
        raise Phase7CError(
            f"{IDENTITY_ARM!r} is not among the loaded arms "
            f"{sorted(arms_loaded)}; every comparison is against it"
        )

    def compare(a: str, b: str, was: dict | None) -> dict:
        for arm in (a, b):
            if arm not in arms_loaded:
                raise Phase7CError(f"arm {arm!r} was not declared")
        paired = phase7b.paired_comparison(
            truth=truth,
            winner_by_seed=arms_loaded[b]["by_seed"],
            baseline_by_seed=arms_loaded[a]["by_seed"],
            winner_sd=arms_loaded[b]["sd"],
            n_boot=n_boot,
        )
        condition_1 = bool(
            paired["n_excluding_zero"] == paired["n_seeds"]
            and paired["same_direction"]
        )
        entry = {
            "a": a, "b": b, "delta_is": f"{b} minus {a}",
            "condition_1_paired_bca": condition_1,
            "condition_2_exceeds_threshold": bool(paired["exceeds_threshold"]),
            "claimable": bool(paired["claimable"]),
            "paired": paired,
        }
        if was is not None:
            entry["reported_on_condition_2_alone"] = bool(was["claimable"])
            entry["verdict_changed"] = bool(was["claimable"] != paired["claimable"])
            entry["recorded_delta"] = was["delta"]
        return entry

    against_identity = {}
    for arm in sorted(arms_loaded):
        if arm == IDENTITY_ARM:
            continue
        against_identity[arm] = compare(
            IDENTITY_ARM, arm, recorded["against_identity"].get(arm)
        )

    between_arms = {}
    for pair in comparisons():
        a, b = result_key(pair["a"]["name"]), result_key(pair["b"]["name"])
        between_arms[pair["question"]] = compare(
            a, b, recorded["comparisons"].get(pair["question"])
        )
        between_arms[pair["question"]]["varies"] = pair["varies"]

    changed = [
        name for group in (against_identity, between_arms)
        for name, entry in group.items() if entry.get("verdict_changed")
    ]
    return {
        "round": round_label,
        "rule": phase7b.COMPARISON_RULE,
        "gap_this_closes": PAIRED_BCA_IS_THE_MISSING_CONDITION["gap"],
        "against_identity": against_identity,
        "comparisons": between_arms,
        "n_comparisons": len(against_identity) + len(between_arms),
        "verdicts_changed": sorted(changed),
        "underpowered_nulls": sorted(
            name for group in (against_identity, between_arms)
            for name, entry in group.items()
            if entry["condition_1_paired_bca"]
            and not entry["condition_2_exceeds_threshold"]
        ),
    }


#: **[MEASURED 2026-08-04] The paired BCa withdrew all nine verdicts. Phase
#: 7C supports nothing.**
#:
#: Verification passed first: all seven arms reproduce ``selected30`` within
#: 0.0002 on mean AND SD, and the truth column agrees across all 35 vectors.
#: These are the phase's own fits, and the result is about them.
#:
#: **Not one interval, in any of the 45, excludes zero.** Arm 1 against
#: identity, per seed: [-0.094, +0.050], [-0.137, +0.034], [-0.151, +0.033],
#: [-0.146, +0.002], [-0.130, +0.047]. Condition 1 fails, the conditions are
#: ANDed, and a mean delta of -0.045 clearing its threshold no longer
#: suffices. The same holds for all six against identity and all three
#: between arms.
#:
#: **The reportable statement.** Augmentation was implemented to
#: specification, applied at the full thirty-epoch budget with the
#: parameters given at supervision and the region design they specify, and
#: *this cohort cannot
#: resolve whether it helps or hurts*. Not "augmentation hurts", not
#: "geometric hurts", not "region protection is null". The shape is Phase 4's
#: unresolvable scheme ordering, reached by a different route.
#:
#: **Why the mean delta was misleading.** The per-seed interval is a paired
#: bootstrap over the 237 PATIENTS within one seed, so its width is set by how
#: idiosyncratically the two prediction vectors disagree patient by patient --
#: not by how far apart their means are. Augmentation widens that disagreement
#: while moving the mean, so a -0.10 mean delta can sit inside intervals half
#: a point wide. Arm 1's are ~0.165 wide, half-width ~0.082: a per-seed delta
#: needs to be roughly that large in EVERY seed, and -0.045 is not.
#:
#: **The sequence is the finding as much as the result is.** The criterion was
#: pre-registered in PLAN §4.3, promised again in
#: ``NOT_A_SEARCH["carried_over"]``, required by exit criterion 6 -- and not
#: computed until after every verdict had been formed, written up, defended
#: through two rounds of correction, and recorded. When it was finally
#: computed it withdrew all of them. Nothing about the data changed; the phase
#: simply had not applied its own rule.
#:
#: **And the phase predicted the opposite.**
#: ``OUTSTANDING_BEFORE_THE_NULLS_ARE_REPORTABLE`` said in writing that the
#: paired test was *"conservative; they survive"* for the negatives and
#: dangerous only for the nulls. Exactly backwards: the nulls were already
#: null and stayed null, and every negative fell. The reasoning was that an
#: unpaired threshold overstates variance -- true, and true about condition 2,
#: which was never the binding constraint on those six.
PAIRED_BCA_WITHDREW_EVERY_VERDICT = {
    "measured": "2026-08-04",
    "run": "p7c_paired_selected30, round selected30, n_boot 10000",
    "verification": (
        "passed before pairing: seven arms reproduce the recorded round "
        "within 0.0002 on mean and SD; truth agrees across all 35 vectors"
    ),
    "result": "all nine comparisons claimable: false; underpowered_nulls: []",
    "intervals_excluding_zero": "0 of 45",
    "arm_1_per_seed": [
        [-0.094, 0.050], [-0.137, 0.034], [-0.151, 0.033],
        [-0.146, 0.002], [-0.130, 0.047],
    ],
    "reportable": (
        "augmentation was implemented to specification and applied at the "
        "full budget with the supervision parameters and region design, and "
        "this cohort cannot resolve whether it helps or hurts -- NOT "
        "'augmentation hurts', NOT 'region protection is null'"
    ),
    "why_the_mean_delta_misled": (
        "the per-seed interval is a paired bootstrap over PATIENTS, so its "
        "width is set by patient-level disagreement between the two vectors, "
        "not by the gap between their means. Augmentation widens the first "
        "while moving the second"
    ),
    "the_sequence_is_the_finding": (
        "pre-registered in PLAN §4.3, promised in NOT_A_SEARCH, required by "
        "exit criterion 6, computed only after every verdict was formed and "
        "twice corrected -- and it withdrew all of them. The data did not "
        "change; the phase had not applied its own rule"
    ),
    "the_phase_predicted_the_opposite": (
        "OUTSTANDING_BEFORE_THE_NULLS_ARE_REPORTABLE recorded the paired test "
        "as 'conservative; they survive' for the negatives and dangerous only "
        "for the nulls. Backwards: the nulls stayed null and every negative "
        "fell. Condition 2 was never the binding constraint on those six"
    ),
    "precedent": "Phase 4's unresolvable scheme ordering, reached by another route",
}


#: **[DECIDED 2026-08-04] Ten seeds cannot change any of this, and the six
#: configs must not be run.**
#:
#: **Condition 1 is a universal quantifier over the seeds present**, not a
#: proportion and not an aggregate. ``COMPARISON_RULE["claimable_requires"]``
#: says *"every seed's BCa interval excludes zero, all in the same
#: direction"*, and ``paired_comparison`` implements exactly that:
#:
#:     all_exclude = len(excluding) == len(per_seed) and len(directions) == 1
#:
#: So the predicate is ANTI-MONOTONE in the seed set: if S ⊂ S', then
#: condition 1 on S' implies condition 1 on S. Contrapositive -- failing at
#: five seeds forces failure at ten.
#:
#: **And the five failing intervals are not re-drawn, they REAPPEAR.** A
#: seed's fit is budget- and count-independent (``rng_for(seed, fold,
#: epoch)``, fixed head seed), and ``paired_delta_bca`` seeds its bootstrap
#: from the seed itself. So seeds 1337, 2024, 7, 99 and 12345 produce the
#: identical vectors and the identical intervals inside a ten-seed run --
#: which is the same fact the ten-seed round was going to exploit as a free
#: determinism check.
#:
#: **Ten seeds therefore fails condition 1 BY CONSTRUCTION, not by
#: probability.** It does not depend on how the five new seeds behave. Sixty
#: fits to reconfirm a verdict that cannot move.
#:
#: **The temptation to note and refuse.** A proportion rule ("most intervals
#: exclude zero"), or pooling the seeds into one vector before the bootstrap,
#: would both let some of these verdicts back. Both are available, both are
#: defensible in the abstract, and choosing either NOW -- after seeing that
#: the pre-registered rule returned an unwelcome answer -- is the move this
#: phase has already corrected twice. ``COMPARISON_RULE["why_not"]`` had
#: independently ruled out the pooling variant before any of this ran, on the
#: grounds that averaging prediction vectors is an ensemble and moves the bar.
TEN_SEED_ROUND_NOT_RUN = {
    "decided": "2026-08-04",
    "status": "NOT RUN -- the six configs stay unlaunched",
    "condition_1_is_universal": (
        "every seed's interval must exclude zero, all one direction; "
        "paired_comparison tests len(excluding) == len(per_seed)"
    ),
    "anti_monotone": (
        "condition 1 on a superset implies it on the subset, so failing at "
        "five forces failure at ten"
    ),
    "the_five_reappear": (
        "a seed's fit is budget- and count-independent and the bootstrap is "
        "seeded from the seed, so the five failing intervals are identical "
        "inside a ten-seed run rather than re-drawn"
    ),
    "conclusion": (
        "fails BY CONSTRUCTION, independent of the five new seeds. 60 fits to "
        "reconfirm a verdict that cannot move"
    ),
    "refused_alternatives": (
        "a proportion rule, or pooling the seeds before the bootstrap, would "
        "each readmit some verdicts -- and choosing either after seeing the "
        "pre-registered rule return an unwelcome answer is the move this "
        "phase corrected twice. COMPARISON_RULE['why_not'] had already ruled "
        "out pooling, before any of this ran"
    ),
    "what_would_change_it": (
        "not more seeds. A cohort large enough that a per-seed paired "
        "interval over patients is narrower than the effect -- arm 1's are "
        "~0.165 wide against a 0.045 delta"
    ),
}


#: **[OPEN 2026-08-04] Phase 7B's headline carries the same gap, and the
#: check costs nothing.**
#:
#: ``phase7b.SEARCH_AXIS_VERDICTS["outcome"]["against_baseline"]`` records
#: ``delta``, ``threshold``, ``claimable`` and ``single_run_95`` -- all
#: condition-2 quantities. No interval appears in it. So 7B's *"the tuned arm
#: is claimably WORSE than the untuned one"* is, in the record, a
#: condition-2-only verdict of exactly the kind 7C just lost all nine of.
#:
#: 7C's arms failed condition 1 at mean deltas up to -0.10. 7B's claim rests
#: on -0.0542 with sd 0.0215 over the same 237 patients. That is not a
#: prediction -- interval width is set by patient-level disagreement, not by
#: the mean gap, and 7B's arm is not augmented so its vectors may well agree
#: with the baseline far more tightly. It is a reason to look.
#:
#: **7B's headline does not depend on it.** *"Nothing beat 0.2520"* needs the
#: winner not to be claimably BETTER, which no reading threatens. What is at
#: risk is the sharper sentence -- "claimably worse" -- and the narrative that
#: the protocol caught a real degradation rather than an unresolvable one.
#:
#: **No fits, and probably no run.** ``task_phase7b_search`` computes
#: ``comparison["paired"]`` whenever the ``baseline_oof_seed_<seed>`` inputs
#: are declared, and they are declared in the shipped config. If the
#: p7b-search-3 run had them, its ``metrics.json`` already holds the five
#: intervals and this is a read.
PHASE_7B_MAY_CARRY_THE_SAME_GAP = {
    "opened": "2026-08-04",
    "observation": (
        "phase7b.SEARCH_AXIS_VERDICTS records delta, threshold, claimable and "
        "single_run_95 for the winner-vs-baseline comparison and no interval; "
        "those are all condition-2 quantities"
    ),
    "at_risk": (
        "the sentence 'the tuned arm is claimably WORSE', not the headline "
        "'nothing beat 0.2520' -- the latter needs only that the winner is "
        "not claimably better"
    ),
    "not_a_prediction": (
        "7C failed at deltas up to -0.10, but interval width is set by "
        "patient-level disagreement, not the mean gap, and 7B's winner is not "
        "augmented. This is a reason to look, not an expectation"
    ),
    "how_to_check": (
        "read comparison.paired.per_seed from the p7b-search-3 metrics.json; "
        "task_phase7b_search computes it whenever the baseline_oof_seed_<seed> "
        "inputs are declared, and the shipped config declares all five. No "
        "fits, and probably no run"
    ),
}


def summary() -> dict:
    """The pre-registration in one object, for the run record."""
    return {
        "arms": [arm["name"] for arm in arms()],
        "n_arms": len(arms()),
        "seeds_per_arm": len(SEEDS),
        "total_fits": len(arms()) * len(SEEDS),
        "comparisons": [
            {"question": c["question"], "varies": c["varies"],
             "a": c["a"]["name"], "b": c["b"]["name"]}
            for c in comparisons()
        ],
        "baseline": {"arm": BASELINE["arm"], "pcc": BASELINE["pcc"],
                     "sd": BASELINE["sd"], "seeds": BASELINE["seeds"]},
        "is_a_search": False,
        "photometric": PHOTOMETRIC,
        "geometric": GEOMETRIC,
        "region_aware": REGION_AWARE,
        "blend": BLEND,
        "horizontal_flip": HORIZONTAL_FLIP["included"],
        "live_path": LIVE_PATH,
    }
