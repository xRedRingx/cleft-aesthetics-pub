"""Phase 14: Label Distribution Learning -- CONCEDED-COVERED at the
reckoning, before any code.

The shortest phase in the project, and correctly so: its first
obligation (``phase12.PHASE_SEQUENCE_RENUMBERED_2``, the accepted
reckoning) was to distinguish its mechanism from the ladder's closed LDL
negative or concede coverage -- and reading the module answered it. The
concession is the phase doing its job, not the phase failing.
"""

from __future__ import annotations

#: **[CLOSED 2026-08-24: CONCEDED-COVERED] PHASE 14
#: OPENS AND CLOSES AT THE RECKONING.** No arm was built; the closed
#: verdict covers the proposal line by line.
PHASE_14_CONCEDED_COVERED = {
    "closed": (
        "2026-08-24, the ruling on the restatement: CONCEDED-"
        "COVERED. Nothing was built; criterion 1 of the proposed exit "
        "criteria -- the reckoning's answer -- closed the phase"
    ),
    "the_prior_measurement": (
        "ladder.STAGE_G_LABEL_FORMULATION, measured 2026-08-02: ldl "
        "0.2336 (sd 0.0305) against the mean's 0.2520 (sd 0.0148) at "
        "imagenet/G1, delta -0.0184 inside its 0.0297 threshold; "
        "0.1583 against 0.2001 at masked/G2, CLAIMABLY HARMFUL. "
        "Verdict as pinned: 'the distribution carries no information "
        "the mean discards that this head can use'"
    ),
    "the_mechanism_identity": (
        "the reckoning asked whether Phase 14's proposal differs from "
        "what the 0.2336 arm trained. VERIFIED AGAINST THE MODULE "
        "(train/ldl.py, LDLHeadBackbone) and the shipped config "
        "(p7_g1_vit_b16_imagenet_g1_ldl.yaml), not memory: the closed "
        "arm was a FIVE-OUTPUT SOFTMAX HEAD over the frozen 768-d "
        "embeddings, KL LOSS (target||predicted, batchmean), targets "
        "the SOFT_1..SOFT_5 distribution, readout the EXPECTATION over "
        "grades 1-5 for PCC against the mean column, on the standard "
        "recipe with the label formulation as the single factor. "
        "Phase 14 proposed: a distribution head (5 outputs), a "
        "distribution loss (KL or similar), targets soft_1..soft_5, "
        "predicted mean read out for PCC. HEAD, LOSS, TARGET, READOUT, "
        "RECIPE: identical on every axis. This is identity, checkable "
        "line by line -- not a judgement call"
    ),
    "supporting_evidence_as_cited": {
        "rater_screen": (
            "phase10.CLEFTGNN_COMPARATOR_TABLES: rater-trained models "
            "score -0.273, 0.323, -0.092, -0.121, 0.598 on the "
            "benchmark -- 4 OF 5 BELOW the panel-mean probe. The "
            "distribution's components are individually HARDER targets "
            "than their mean; what soft_1..5 adds over the mean is "
            "mostly rater noise"
        ),
        "the_closed_arms_own_sd": (
            "the LDL arm's five-seed sd was 0.0305 against the mean "
            "arm's 0.0148 -- LOOSER, not tighter, so even the Phase-12 "
            "co-primary reading (a tighter arm matters at equal mean) "
            "ran the wrong way in the measurement that exists"
        ),
        "coarse_features_bar": (
            "phase13's mechanism chapter: the features are coarse -- "
            "lips-dominant, extremes-separable, the mild-moderate "
            "middle (199 of 237) collapsed. A distribution head could "
            "only add value by resolving WITHIN-grade structure the "
            "scalar head discards, and nothing measured predicts the "
            "embedding carries any"
        ),
    },
    "variants_named_not_pursued": (
        "**EXPLICITLY UNREGISTERED -- no fishing.** Two variants were "
        "named in the restatement as what a genuinely different "
        "mechanism would have to look like: LDL-style loss as "
        "PRETRAINING for a scalar head, and distribution supervision "
        "at a DIFFERENT REPRESENTATION. Neither is registered, neither "
        "has a record-based reason to expect a different outcome, and "
        "naming them here is what prevents a later turn from "
        "presenting either as a fresh idea that escapes this record"
    ),
    "the_corrected_reckoning": (
        "the reckoning field's clause 'through the standard head' "
        "mis-described the prior measurement; corrected dated in place "
        "(phase12.PHASE_SEQUENCE_RENUMBERED_2"
        "['ldl_mechanism_description_corrected']). The pinned figures "
        "stand; only the mechanism description was wrong. The "
        "mis-description originated in the phase-sequencing discussion "
        "and was caught by READING THE MODULE rather than trusting the "
        "description -- which is the whole reckoning discipline "
        "working as designed"
    ),
    "what_would_reopen": (
        "only a NEW measured reason -- a mechanism differing from the "
        "closed arm's on a named axis, with a record-based prediction "
        "of why the difference matters. Scheduling was not reopening; "
        "closing is not forbidding"
    ),
    "sequence_advances": (
        "Phase 15 (the second beauty dataset, Road B Branch 2 "
        "unparked) is next, per PHASE_SEQUENCE_RENUMBERED_2 -- its "
        "exit criteria MUST include the comparative verdict against "
        # [2026-08-24] "Phase 16" here means TSTR, which the third
        # sequence amendment moved to Phase 17
        # (phase15.PHASE_SEQUENCE_RENUMBERED_3), where the fourth left
        # it (phase15.PHASE_SEQUENCE_RENUMBERED_4). Preserved as
        # written.
        "SCUT, as registered at scheduling, because Phase 16 consumes "
        "'the best' as a measured answer"
    ),
    "ledger": "no entry -- nothing was measured; the phase is a ruling",
}


def summary() -> dict:
    """The phase's records, importable as one object."""
    return {"conceded_covered": PHASE_14_CONCEDED_COVERED}
