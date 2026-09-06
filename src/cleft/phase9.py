"""Phase 9: prototypes, the results ledger, and the external reference.

Opened 2026-08-16 under PLAN_AMENDMENT_2026-08-13 section 4, with the
write-up split out to Phase 13 and the convergence deliverable DROPPED by
the framing decision below. The registrations live here as data with
tests, the ledger lives in ``results_ledger``, and nothing computes a
number before its choices are registered -- the t-SNE discipline, now the
phase's own.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

#: **[DECIDED 2026-08-16, the framing decision -- recorded verbatim
#: as structured] Road A is the main project; Road B was never a parallel
#: or co-equal road.** Road B is the ANNEX of additional and alternative
#: ideas -- things deliberately kept out of the main line so the main
#: project stayed clean, tracked separately so it is known what the main
#: project did versus what else was explored.
#:
#: **The consequences:**
#:
#: (a) The amendment's "both roads converging" deliverable is DROPPED --
#: there is no convergence because there was never a fork, just a main
#: line and its annex.
#:
#: (b) All Road B measurements stand unchanged AS MEASUREMENTS -- paired
#: claims, margin-table points, voids; nothing is demoted evidentially.
#: Their framing in records and eventually the write-up is "additional
#: ideas explored," extensions of the one investigation.
#:
#: (c) Run directories, config names, and record names keep their
#: ``roadb_`` prefixes -- renaming history would damage provenance. The
#: framing lives in the records' prose, not in file paths.
#:
#: **The parity prose this supersedes, quoted where it stands**: the
#: ``roadb`` module docstring's "Road A is not superseded. It is the
#: control arm... Both roads converge at Phase 9" (the "control arm"
#: framing INVERTED the actual hierarchy -- a dated note now sits beside
#: it); ``ROAD_B_BRIEF.md``'s "Both roads converge at Phase 9" (lines 16,
#: 192), "Neither supersedes the other" (line 29), "It is the control"
#: (line 255); the amendment's section-4 title and "Phase 9 delivers the
#: prototypes and the convergence record". The amendment's dependency
#: line "9 needs both roads closed" dissolves under this framing: there
#: is no road-closure precondition, only measurements the ledger
#: consumes, and parked Branch 2 blocks nothing.
ROAD_B_IS_THE_ANNEX = {
    "decided": "2026-08-16",
    "framing": (
        "Road A is the main project; Road B is the annex of additional "
        "and alternative ideas, kept out of the main line deliberately "
        "and tracked separately"
    ),
    "consequences": {
        "convergence_deliverable": "DROPPED -- no fork, no convergence",
        "evidential_status": (
            "every Road B measurement stands unchanged; only the framing "
            "becomes 'additional ideas explored'"
        ),
        "naming": (
            "roadb_ prefixes stay in run dirs, configs, and record names; "
            "the framing lives in prose, never in paths"
        ),
    },
    "ledger_field": (
        "the main-project/additional-ideas distinction is a framing FIELD "
        "on ledger entries (main | additional), never two ledgers"
    ),
}


#: **[REGISTERED 2026-08-16, before any computation] The prototypes: all
#: six formerly-blocked choices resolved by the maintainer, plus the
#: build's necessary defaults, stated here at commitment time.**
#:
#: **(i) The space**: arm A's pooled set
#: (``embeddings_g1_ladder_v1/vit_b16__imagenet__g1``), Euclidean
#: distance -- the same subject and metric as the t-SNE, and the measured
#: prior travels WITH the registration: the t-SNE companion read 0.409
#: against a majority baseline of 0.502, so the space's neighbourhoods
#: predict class3 WORSE than majority guessing. **Both readings committed
#: now**: unstable medoids would CORROBORATE that weak neighbourhood
#: structure, not surprise; stable medoids would be the surprising
#: outcome and would sharpen the question of what stability rests on.
#:
#: **(ii) The groups**: three medoids at class3, the PRIMARY; the
#: five-grade set as DESCRIPTIVE SECONDARY -- computed and shown, never
#: validated, never quoted as if it were. Necessary default, stated
#: blind: the five-grade grouping column is ``median`` (the plan's own
#: ordinal secondary); a five-grade group with no members is reported
#: ABSENT, and a single-member group's medoid is that member, flagged.
#:
#: **(iii) Leave-one-out** validates MEDOID IDENTITY STABILITY on the
#: primary, with nearest-medoid accuracy beside it as companion and the
#: majority baseline printed -- the t-SNE display rule's shape. Honesty
#: floor, stated now: removing the medoid itself necessarily changes the
#: identity, so per-class stability has ceiling (n-1)/n, and removals
#: outside a class cannot change its medoid, so each class's statistic is
#: over its OWN members' removals.
#:
#: **(iv) Bootstrap** = identity persistence frequency: 2,000 patient
#: resamples, seed 1337; per class, the fraction of resamples in which
#: the full-data medoid is again the medoid of the resampled members
#: (weighted by multiplicity). Registered before any number.
#:
#: **(v) The 25-image set is IN** -- as the external reference
#: (``DEALL_REFERENCE_REGISTERED``), its own registration.
#:
#: **(vi) Tier**: CLUSTER-ONLY. Medoids are patient faces; identities and
#: rendered faces never leave EHU infrastructure. The SHAREABLE metrics
#: carry stability and accuracy numbers only, no patient ids.
#:
#: Remaining necessary default, stated blind: a distance tie between two
#: candidate medoids is broken toward the lower patient id -- committed
#: with the metric, never revisited after a number exists.
PROTOTYPES_REGISTERED = {
    "registered": "2026-08-16, before any computation",
    "space": {
        "embeddings": "embeddings_g1_ladder_v1/vit_b16__imagenet__g1",
        "set_name": "vit_b16__imagenet__g1",
        "distance": "euclidean",
        "tie_rule": "toward the lower patient id, committed blind",
    },
    "measured_prior": {
        "tsne_companion": 0.409,
        "majority": 0.502,
        "committed_both_ways": (
            "unstable medoids corroborate the weak neighbourhood "
            "structure; stable medoids would be the surprise"
        ),
    },
    "primary": {"column": "class3", "n_groups": 3},
    "secondary": {
        "column": "median",
        "status": "DESCRIPTIVE -- computed and shown, never validated",
        "empty_group": "reported absent",
        "single_member_group": "the member, flagged",
    },
    "loo": {
        "validates": "medoid identity stability, primary only",
        "companion": "nearest-medoid accuracy, majority baseline printed",
        "ceiling": "(n-1)/n per class -- the medoid's own removal counts",
        "scope": "each class's statistic is over its own members' removals",
    },
    "bootstrap": {
        "statistic": "identity persistence frequency",
        "n_boot": 2000,
        "seed": 1337,
    },
    "tier": "CLUSTER-ONLY; shareable metrics carry numbers only, no ids",
    # [RELABELLED 2026-08-16, from supervision directly via the maintainer] "Prototypes"
    # means the 25-set images as reference ANCHORS, not cohort medoids --
    # the plan v3 sentence this registration was built on was never the supervision material's
    # meaning. Everything registered here stands under its honest name,
    # COHORT MEDOIDS; "the Phase 9 prototypes" now means the anchor
    # experiment (PROTOTYPE_CLASSIFIER_REGISTERED).
    "relabelled": (
        "2026-08-16: this is the COHORT-MEDOID registration; 'prototypes' "
        "is the anchor experiment (PROTOTYPE_CLASSIFIER_REGISTERED)"
    ),
}


#: **[REGISTERED 2026-08-16; BUILD BLOCKED, and the block is itself the
#: recorded state] The 25-image set as EXTERNAL REFERENCE -- what is
#: fixed, what the trap beside it is, and the two reads the build waits
#: on.**
#:
#: **Fixed by a filesystem verification on the cluster**: 25 identities
#: exactly at the cohort data directory (the 26th stem is ``Thumbs.db``,
#: excluded by the standing basename rules); three views per identity
#: (composite, -lips, -nose); per Deall et al. 2016 (ref [16] of the
#: CleftGNN manuscript) -- the same 25 images as that manuscript's
#: Benchmark Test Set. Labels: ``bch/cl_images_test_details.csv``, keyed
#: by ``PhotoID``, a ``Score`` column and four ``Para`` columns.
#:
#: **[2026-09-02] Fleiss kappa is UNWEIGHTED** -- it scores 1-vs-2 as
#: identically wrong to 1-vs-5, which is the wrong loss for an ordinal
#: scale. The distance-aware figures on the same matrix are QWK 0.4276
#: and mean inter-rater r 0.4696. Nothing below is edited:
#: ``record_audit.THE_KAPPA_LIMITATION``.
#: **THE TRAP, recorded so nobody re-falls into it**: the file beside it,
#: the derived per-image labels CSV, is the 251-row COHORT sheet
#: (RanaPhotoIDs, Fleiss 0.1605) -- it is NOT the 25-set's labels, and
#: reading it as such would score the external reference against the
#: wrong study's numbers without an error raised.
#:
#: **Registered before any number exists**: scoring the 25 is
#: generalisation to a DIFFERENT label (surgeon consensus, not the
#: cohort's panel mean); it is reported as EXTERNAL REFERENCE ONLY and
#: never pooled with cohort results; and it lands on the same 25 images
#: as CleftGNN Table 5, making it the first directly-placed number
#: beside the group's published benchmark. The task shape when unblocked:
#: stage the COMPOSITE view through the standard pipeline, extract with
#: the frozen best arm (vit_b16 imagenet, the live-path extractor), score
#: with arm A's frozen heads -- NO refitting on the 25; all 25 heads
#: (5 seeds x 5 folds) score every image, mean and spread reported, since
#: external images belong to no fold. PCC and Spearman against ``Score``
#: at n=25. The -lips/-nose views: recorded PRESENT and AVAILABLE, not
#: staged now.
#:
#: **BLOCKED ON TWO READS, both cluster-side** (the exit criterion's
#: "blocking issue recorded" path -- this record is that recording):
#:
#: 1. ``cl_images_test_details.csv`` read in full -- what ``Score`` is
#:    (the 27-surgeon consensus per the provenance, or otherwise) and
#:    what ``Para1``-``Para4`` are (plausibly Deall's digital-symmetry
#:    parameters -- READ, never assumed). The file lives beside the
#:    cohort scoresheets in the cohort data directory, and this machine
#:    has no cluster access; the standing convention applies -- the file
#:    arrives in the handoff directory outside the repository and is
#:    read directly.
#: 2. The composite view's NATURE against the G1 pipeline's calibration.
#:    ``geometry.staging.stage`` itself is aspect-agnostic (pad to
#:    square, content box recorded) -- the load-bearing assumption is
#:    WHAT FILLS THE CONTENT: the cohort pipeline is calibrated to
#:    single nose-lip clinical crops, and if the composite is a montage
#:    of views rather than such a crop, staging needs its own decision.
#:    A dimensions-and-description listing of the 25 composite files
#:    suffices to resolve this.
#:
#: The task and config are NOT built until both reads land -- building
#: against an unread label file would be the exact move the trap above
#: punishes.
DEALL_REFERENCE_REGISTERED = {
    "registered": "2026-08-16",
    "set": {
        "identities": 25,
        "excluded": "Thumbs.db, by the standing basename rules",
        "views": ("composite", "-lips", "-nose"),
        "provenance": (
            "Deall et al. 2016, ref [16] of the CleftGNN manuscript -- "
            "the manuscript's Benchmark Test Set, same 25 images as its "
            "Table 5"
        ),
        "labels": "bch/cl_images_test_details.csv, keyed by PhotoID",
        "label_columns": ("PhotoID", "Score", "Para1", "Para2", "Para3", "Para4"),
    },
    # **[2026-09-02] Fleiss kappa is UNWEIGHTED** -- 1-vs-2 scores as
    # identically wrong to 1-vs-5. Distance-aware beside it: QWK 0.4276,
    # mean inter-rater r 0.4696. record_audit.THE_KAPPA_LIMITATION.
    "trap": (
        "the derived per-image labels CSV beside it is the 251-row COHORT "
        "sheet (RanaPhotoIDs, Fleiss 0.1605), NOT the 25-set's labels"
    ),
    "reading_registered": (
        "generalisation to a DIFFERENT label (surgeon consensus, not "
        "panel mean); external reference only, never pooled with cohort "
        "results; the first directly-placed number beside CleftGNN "
        "Table 5"
    ),
    "task_shape": (
        "stage the composite view through the standard pipeline; extract "
        "with the frozen best arm; score with arm A's frozen heads, no "
        "refitting; all 25 heads score every image (no fold owns an "
        "external image), mean and spread reported; PCC and Spearman "
        "against Score at n=25"
    ),
    "other_views": "-lips/-nose recorded present and available, not staged",
    "blocked_on": (
        "the full read of cl_images_test_details.csv (Score provenance, "
        "Para1-4 semantics) -- delivered as a file per the standing "
        "convention",
        "the composite view's nature vs the G1 single-crop calibration "
        "-- a dimensions-and-description listing of the 25 files",
    ),
    "status": "REGISTERED; task and config not built until both reads land",
    # [2026-08-16, same day] Both reads answered from the files --
    # DEALL_REFERENCE_READS. The task and config are built; the launch
    # waits on the eyeballed composite (the criterion is in the READS).
    "unblocked": "2026-08-16 -- see DEALL_REFERENCE_READS",
    # [EXTENDED 2026-08-16] The set now carries TWO roles: (1) external
    # reference test set -- DONE: run p9_deall_reference__a78e20da__
    # p9-deall-reference-3 scored PCC 0.2512 at n=25, and the staged
    # sheet was reviewed and PASSED  (the eyeball criterion,
    # answered on the run's own document); (2) ANCHOR SET for the
    # prototype classifier (PROTOTYPE_CLASSIFIER_REGISTERED) -- the supervision material's
    # actual reading of "prototypes".
    "roles": {
        "external_reference": (
            "DONE -- p9_deall_reference__a78e20da__p9-deall-reference-3, "
            "PCC 0.2512 at n=25; staged sheet reviewed and PASSED "
            "(2026-08-16)"
        ),
        "anchor_set": (
            "the prototype classifier's trusted references "
            "(PROTOTYPE_CLASSIFIER_REGISTERED)"
        ),
    },
}


#: **[REGISTERED 2026-08-16, before the phase's work] Phase 9's exit
#: criteria -- written down because none existed and the amendment's
#: implicit trio (prototypes + convergence record + ledger) went stale
#: when the framing decision struck convergence.**
PHASE_9_EXIT_CRITERIA = {
    "registered": "2026-08-16",
    "criteria": {
        "1_choices_first": (
            "every prototype choice registered before computation "
            "(PROTOTYPES_REGISTERED)"
        ),
        "2_prototypes": (
            "medoids computed and validated per the registration -- LOO "
            "identity stability with its companion, bootstrap persistence"
        ),
        "3_external_reference": (
            "the 25-set staged, scored, and recorded as external "
            "reference -- or a blocking issue recorded "
            "(DEALL_REFERENCE_REGISTERED.blocked_on is that record)"
        ),
        "4_ledger": (
            "results_ledger registered, born populated, append-only "
            "enforced by the rolling-checksum test"
        ),
        "5_suite": "green",
    },
    "explicitly_not_a_criterion": (
        "any convergence deliverable -- dropped by ROAD_B_IS_THE_ANNEX"
    ),
    # [AMENDED 2026-08-16, with the relabel] Criterion 2 was written when
    # "prototypes" meant the plan sentence's cohort medoids. Under the supervision material's
    # actual reading it is met three ways at once: "prototypes" resolved
    # to the anchor reading; the medoid work stands under its corrected
    # label as descriptive (validated exactly as registered, weak, the
    # committed reading fired); and the anchor classifier is registered
    # and measured (PROTOTYPE_CLASSIFIER_OBSERVED).
    "criterion_2_amended": (
        "2026-08-16: 'prototypes' resolved to the supervision anchor reading; the "
        "medoid work stands as descriptive under its corrected label; "
        "the anchor classifier registered and measured"
    ),
}


# --------------------------------------------------------------------------
# the medoid machinery (pure, laptop-testable)
# --------------------------------------------------------------------------


def medoid_index(features, member_ids=None, weights=None) -> int:
    """The medoid's positional index: the member minimising the summed
    Euclidean distance to the others, ties toward the lower patient id
    (``PROTOTYPES_REGISTERED`` -- committed blind).

    ``member_ids`` orders the tie-break; positional order stands in when
    ids are absent. ``weights`` (bootstrap multiplicities) weight each
    column's contribution to the sums."""
    features = np.asarray(features, dtype=np.float64)
    if features.ndim != 2 or not len(features):
        raise ValueError(f"expected a non-empty (n, d) matrix, got {features.shape}")
    n = len(features)
    ids = np.arange(n) if member_ids is None else np.asarray(member_ids)
    if len(ids) != n:
        raise ValueError(f"{len(ids)} ids do not align with {n} rows")
    squares = np.sum(features * features, axis=1)
    distances = np.sqrt(np.maximum(
        squares[:, None] + squares[None, :] - 2.0 * (features @ features.T), 0.0
    ))
    column_weights = (
        np.ones(n) if weights is None else np.asarray(weights, dtype=np.float64)
    )
    sums = distances @ column_weights
    best = np.flatnonzero(np.isclose(sums, sums.min(), rtol=0.0, atol=1e-12))
    return int(best[np.argmin(ids[best])])


def loo_identity_stability(features, member_ids) -> dict:
    """Per-class LOO, one class at a time: the fraction of member removals
    that leave the medoid unchanged. Ceiling (n-1)/n by construction --
    the medoid's own removal always changes the identity, and the record
    says so rather than letting the number look like a defect."""
    features = np.asarray(features, dtype=np.float64)
    member_ids = np.asarray(member_ids)
    n = len(features)
    if n < 2:
        raise ValueError(f"stability needs at least 2 members, got {n}")
    full = medoid_index(features, member_ids)
    unchanged = 0
    for leave in range(n):
        keep = [i for i in range(n) if i != leave]
        reduced = medoid_index(features[keep], member_ids[keep])
        if member_ids[keep][reduced] == member_ids[full]:
            unchanged += 1
    return {
        "medoid_id": int(member_ids[full]),
        "stability": unchanged / n,
        "ceiling": (n - 1) / n,
        "n": n,
    }


def nearest_medoid_accuracy(features, classes, patient_ids) -> dict:
    """The LOO companion: each patient classified by the nearest medoid,
    their OWN class's medoid recomputed without them (the other classes'
    medoids cannot depend on them). Majority and chance ride along --
    the accuracy is unreadable without its bar (the t-SNE lesson)."""
    features = np.asarray(features, dtype=np.float64)
    classes = np.asarray(classes)
    patient_ids = np.asarray(patient_ids)
    values, counts = np.unique(classes, return_counts=True)
    rows_of = {value: np.flatnonzero(classes == value) for value in values}
    full_medoid_row = {
        value: rows_of[value][medoid_index(
            features[rows_of[value]], patient_ids[rows_of[value]]
        )]
        for value in values
    }
    correct = 0
    for i in range(len(classes)):
        medoid_rows = dict(full_medoid_row)
        own = classes[i]
        keep = rows_of[own][rows_of[own] != i]
        if not len(keep):
            raise ValueError(
                f"class {own} has a single member; leave-one-out cannot "
                "classify them against their own class's medoid"
            )
        medoid_rows[own] = keep[medoid_index(features[keep], patient_ids[keep])]
        nearest = min(
            values,
            key=lambda value: float(
                np.linalg.norm(features[i] - features[medoid_rows[value]])
            ),
        )
        if nearest == own:
            correct += 1
    return {
        "accuracy": correct / len(classes),
        "majority": float(counts.max()) / float(counts.sum()),
        "chance": 1.0 / len(values),
        "n": int(len(classes)),
    }


def bootstrap_persistence(features, member_ids, *, n_boot: int, seed: int) -> dict:
    """Identity persistence frequency (``PROTOTYPES_REGISTERED``): the
    fraction of patient resamples in which the full-data medoid is again
    the medoid of the resampled members, multiplicities as weights."""
    features = np.asarray(features, dtype=np.float64)
    member_ids = np.asarray(member_ids)
    n = len(features)
    full_id = int(member_ids[medoid_index(features, member_ids)])
    rng = np.random.default_rng(seed)
    persisted = 0
    for _ in range(int(n_boot)):
        draw = rng.integers(0, n, size=n)
        present, multiplicity = np.unique(draw, return_counts=True)
        resample = medoid_index(
            features[present], member_ids[present], weights=multiplicity
        )
        if int(member_ids[present][resample]) == full_id:
            persisted += 1
    return {
        "medoid_id": full_id,
        "persistence": persisted / int(n_boot),
        "n_boot": int(n_boot),
        "seed": int(seed),
    }



#: **[OBSERVED 2026-08-16, against the registration] The prototypes ran,
#: and the COMMITTED WEAK-STABILITY READING FIRED.**
#:
#: The figures: class3=0 medoid patient 183 (n=88), LOO 0.534 (ceiling
#: 0.989), bootstrap persistence 0.394; class3=1 medoid 147 (n=119), LOO
#: 0.992 (ceiling 0.992), persistence 0.427; class3=2 medoid 148 (n=30),
#: LOO 0.967 (ceiling 0.967), persistence 0.414. The companion: accuracy
#: 0.257 against majority 0.502 and chance 0.333 -- BELOW CHANCE.
#:
#: **The two at-ceiling LOO figures are an ARTIFACT and may never be
#: quoted as stability.** 0.992 and 0.967 are the single-deletion-
#: robustness ceiling (n-1)/n reached exactly: removing any one patient
#: other than the medoid does not move a medoid in a weakly-structured
#: space, which measures the statistic's insensitivity at these group
#: sizes, not coherence. The bootstrap (persistence ~0.4 on all three)
#: and the below-chance companion are the informative figures, and they
#: agree.
#:
#: **The verdict**: the medoids STAND AS DESCRIPTIVE ARTIFACTS -- three
#: real faces, one per grade, usable as figures with their caveats.
#: Representativeness claims are DEAD. This corroborates the t-SNE
#: companion (0.409 vs majority 0.502) exactly as pre-committed in
#: ``PROTOTYPES_REGISTERED.measured_prior`` -- the reading was written
#: before the run and it fired.
PROTOTYPES_OBSERVED = {
    "observed": "2026-08-16",
    "figures": {
        "class3_0": {"medoid": 183, "n": 88, "loo": 0.534,
                     "ceiling": 0.989, "persistence": 0.394},
        "class3_1": {"medoid": 147, "n": 119, "loo": 0.992,
                     "ceiling": 0.992, "persistence": 0.427},
        "class3_2": {"medoid": 148, "n": 30, "loo": 0.967,
                     "ceiling": 0.967, "persistence": 0.414},
    },
    "companion": {"accuracy": 0.257, "majority": 0.502, "chance": 0.333},
    "reading_fired": (
        "the committed weak-stability reading: companion below chance, "
        "persistence ~0.4 on all three"
    ),
    "artifact_rule": (
        "the at-ceiling LOO figures (0.992, 0.967) are the "
        "single-deletion-robustness artifact and may NEVER be quoted as "
        "stability"
    ),
    "verdict": (
        "medoids stand as descriptive artifacts (three real faces per "
        "grade); representativeness claims are dead; corroborates the "
        "t-SNE companion as pre-committed"
    ),
    # [RELABELLED 2026-08-16] Everything measured above stands unchanged
    # under its honest name -- cohort medoids, descriptive, not stable.
    # "The Phase 9 prototypes" means the anchor experiment
    # (PROTOTYPE_CLASSIFIER_REGISTERED); the ledger entry's claim sentence
    # already says "medoids" and needed no correcting entry.
    "relabelled": (
        "2026-08-16: cohort medoids, not 'the prototypes' -- the supervision material's "
        "meaning is the anchor experiment"
    ),
}


#: The 25-set's label file, inside the images folder -- one declared
#: input covers folder and labels together.
DEALL_LABELS_FILENAME = "cl_images_test_details.csv"

#: **[READ 2026-08-16, answered from the files; recorded before the task
#: was built] The two blocked reads, resolved -- and the one thing that
#: cannot be asserted from disk, stated as such.**
#:
#: **Score**: the five-grade assignment from the survey-design key in
#: "Image codes TingLi Pid No" (the survey-design lineage; the document
#: reconciles the survey's 25 presentations = 22 unique + 3 repeats,
#: AOFA tutorial-only). Every overlapping stem cross-checks exactly; the
#: CSV additionally grades FNGA=2 and FPIA=3, giving all 25 stems, with
#: spread 3/7/6/6/3 over grades 1-5. **What cannot be asserted from
#: disk**: whether this key equals the manuscript's "27-surgeon
#: highest-agreement consensus". Both sources are stated; NO
#: reconciliation is forced -- the write-up quotes the provenance it can
#: prove.
#:
#: **Para1-4**: angle-like (clusters near 0, +/-180, and 170-222),
#: plausibly the Deall/SymNose digital-symmetry parameters -- recorded as
#: UNIDENTIFIED-BUT-CHARACTERIZED. They play no role in the task.
#:
#: **The trap, extended -- TWO cohort files masquerade beside the
#: 25-set**: the derived per-image labels CSV AND ``APScores.txt`` are
#: both the 251-row COHORT sheet (processed and raw forms; row 241
#: cross-checks). Neither is the 25-set's labels.
#:
#: **The staging answer, from the pipeline's own code**: G1's
#: whole-image extraction input is the PLAIN staged square --
#: ``stage_build.build`` takes ``base.image`` for g1 and unwarps only
#: for g2; the trapezium (half-width 0.301 top to 0.500 bottom, midline
#: 0.5, 80.2% coverage) defines PATCH AND REGION geometry and never
#: touches the whole-image path. So the pipeline is mechanically safe on
#: any RGB image and the content question is NOT decidable from its
#: requirements: the pipeline is content-blind. **What one eyeballed
#: composite must show** (the images are inspected before launch): a SINGLE
#: nasolabial view -- one nose above one mouth, no panel montage, no
#: fuller composition with eyes or brow -- framed like the cohort's
#: clinical crops with the facial midline near the vertical centre. The
#: composites' near-square aspect ratios (0.86-1.17, 246-712 px, RGB)
#: sit outside the cohort's portrait crop family, which is weak evidence
#: of a fuller composition -- the eyeball decides. A fuller composition
#: is not automatically a blocker: it becomes a RECORDED CAVEAT on the
#: external reference, at a maintainer decision.
#:
#: **Provenance note for this machine**: ``p9_deall_labels.csv`` was not
#: found locally (the repo tree and the handoff directory outside it
#: were both searched); the facts above are the
#: maintainer's file-verified statements, and the task RE-ASSERTS the
#: checkable ones (25 composites, columns, the 3/7/6/6/3 spread) against
#: the cluster original at runtime.
DEALL_REFERENCE_READS = {
    "read": "2026-08-16",
    "score": {
        "source": (
            "the survey-design key in 'Image codes TingLi Pid No' "
            "(the survey-design lineage): 25 presentations = 22 unique + 3 "
            "repeats, AOFA tutorial-only; FNGA=2 and FPIA=3 graded in "
            "the CSV, completing all 25"
        ),
        "expected_spread": (3, 7, 6, 6, 3),
        "cannot_assert_from_disk": (
            "whether this key equals the manuscript's 27-surgeon "
            "highest-agreement consensus; both sources stated, no "
            "reconciliation forced"
        ),
    },
    "para_columns": (
        "angle-like (clusters near 0, +/-180, 170-222), plausibly the "
        "Deall/SymNose symmetry parameters; unidentified-but-"
        "characterized; no role in the task"
    ),
    "trap_extended": (
        "the derived per-image labels CSV AND APScores.txt are both the "
        "251-row cohort sheet (processed and raw; row 241 cross-checks)"
    ),
    "composites": "25 files, RGB, aspect ratio 0.86-1.17, 246-712 px",
    "staging_answer": (
        "G1's whole-image input is the PLAIN staged square (stage_build "
        "takes base.image for g1); the trapezium is patch/region "
        "geometry only; the pipeline is content-blind, so the content "
        "question needs one eyeballed composite"
    ),
    "eyeball_criterion": (
        "a single nasolabial view -- one nose above one mouth, no "
        "montage, no eyes/brow -- framed like the cohort's crops, facial "
        "midline near the vertical centre; a fuller composition becomes "
        "a recorded caveat, at a maintainer decision"
    ),
    "declaration": (
        "the labels CSV lives inside the images folder, so ONE declared "
        "input covers folder+CSV; Thumbs.db is excluded by the standing "
        "suffix rules (folderscan.IMAGE_SUFFIXES)"
    ),
    # [FILLED 2026-08-16, from the listing this session]
    "folder": (
        "${CLEFT_ANCHOR_SET} -- verified by "
        "listing: 75 image files + Thumbs.db + cl_images_test_details.csv "
        "inside (25 identities x 3 views, the CSV under the same rollup)"
    ),
    # [CORRECTED 2026-08-16, a dated maintainer error] The
    # "CSV lives inside the images folder" note described THE SUPERVISOR'S ORIGINAL
    # SHARE layout (the shared tree.txt), not the cluster copy, where the CSV
    # sat at bch ROOT -- and the pinned-filename labels read refused on
    # exactly that difference: the first launch died in four instant
    # attempts (p9_deall_reference__d67248c9__p9-deall-reference-2, VOID,
    # ledger void-deall-first-launch; nothing scored). Fixed by copying
    # the CSV into the folder ON THE CLUSTER, restoring the canonical
    # layout. The re-declared rollup covers 76 files (Thumbs.db
    # excluded) -- resolving the earlier 75-vs-76 arithmetic exactly as
    # flagged: the missing hashable file WAS the CSV.
    "corrected": (
        "2026-08-16: the inside-the-folder note was the shared material layout, "
        "not the cluster copy (CSV at bch root); first launch VOID on "
        "the difference; canonical layout restored, rollup re-declared "
        "at 76 files"
    ),
}


#: **[FLAGGED 2026-08-16 -- a suggestion to CONSIDER, deliberately not
#: built now]** The keeper dirty-tree refusal that preceded the void
#: launch fired BEFORE RunContext existed, so it left nothing findable
#: outside the pod log -- no run dir, no log.txt, no FAILED line -- and
#: the difference between that and a ten-second diagnosis was one
#: debugging cycle. The suggestion: a one-line refusal note written
#: somewhere reachable (beside the out-root, or a well-known refusals
#: file) whenever a guard refuses before the run directory exists.
#: Recorded so the consideration has a date; nothing is built.
PRE_CONTEXT_REFUSALS_LEAVE_NO_TRACE = {
    "flagged": "2026-08-16",
    "observed": (
        "the keeper dirty-tree refusal fired before RunContext; nothing "
        "findable outside the pod log; one debugging cycle spent"
    ),
    "suggestion": (
        "a one-line refusal note somewhere reachable when a guard "
        "refuses before the run directory exists -- a ten-second "
        "diagnosis instead"
    ),
    "status": "TO CONSIDER; deliberately not built now",
}


def deall_partition_stems(names) -> dict:
    """Partition the 25-set folder's files by view, images only.

    The standing suffix rules apply (``folderscan.IMAGE_SUFFIXES`` --
    Thumbs.db and the labels CSV fall out naturally); ``-lips``/``-nose``
    suffixed stems are the other views; everything else is a composite.
    Returns full filenames keyed by view, stems sorted."""
    from .data.folderscan import IMAGE_SUFFIXES

    views: dict = {"composite": [], "lips": [], "nose": []}
    for name in sorted(names):
        path = Path(name)
        if path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        stem = path.stem
        if stem.endswith("-lips"):
            views["lips"].append(name)
        elif stem.endswith("-nose"):
            views["nose"].append(name)
        else:
            views["composite"].append(name)
    return views


def deall_read_labels(csv_path) -> dict:
    """``{PhotoID: Score}`` from the 25-set's own labels file -- and ONLY
    that file: the two cohort sheets beside it are the recorded trap
    (``DEALL_REFERENCE_READS.trap_extended``), so the filename is pinned
    and the shape is asserted against the read facts (25 rows, the
    3/7/6/6/3 spread) rather than trusted."""
    import csv

    path = Path(csv_path)
    if path.name != DEALL_LABELS_FILENAME:
        raise ValueError(
            f"{path.name!r} is not {DEALL_LABELS_FILENAME!r}. The files "
            "beside it are the 251-row cohort sheet in two forms "
            "(DEALL_REFERENCE_READS.trap_extended); only the pinned "
            "filename is the 25-set's labels."
        )
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or "PhotoID" not in rows[0] or "Score" not in rows[0]:
        raise ValueError(
            f"{path.name} does not carry PhotoID and Score columns; got "
            f"{sorted(rows[0]) if rows else 'no rows'}"
        )
    if len(rows) != 25:
        raise ValueError(
            f"{path.name} has {len(rows)} rows, not 25 -- the 251-row "
            "cohort sheet wearing the right name would fail here"
        )
    labels = {row["PhotoID"].strip(): int(float(row["Score"])) for row in rows}
    if len(labels) != 25:
        raise ValueError("duplicate PhotoID in the labels file")
    values, counts = np.unique(sorted(labels.values()), return_counts=True)
    spread = tuple(
        int(counts[list(values).index(grade)]) if grade in values else 0
        for grade in (1, 2, 3, 4, 5)
    )
    expected = DEALL_REFERENCE_READS["score"]["expected_spread"]
    if spread != expected:
        raise ValueError(
            f"grade spread {spread} does not match the read {expected} "
            "(DEALL_REFERENCE_READS): the labels are not the file that "
            "was verified"
        )
    return labels



#: **[REGISTERED 2026-08-16, before any number] THE PROTOTYPE CLASSIFIER
#: -- the actual reading of "prototypes", near-verbatim from the run outputs:
#: the 25 graded images are TRUSTED REFERENCES; extract their features;
#: classify each cohort patient by comparing their features to the
#: anchors' -- NO TRAINING ANYWHERE.**
#:
#: **The caveat travels with the anchors**: their grades are trusted
#: SINGLE grades from the survey lineage (the Image-codes key + the CSV),
#: not verified-unanimous consensus -- the distinction is recorded, and
#: supervision can confirm unanimity later.
#:
#: **The cells, fixed by arithmetic and consistency, not preference**:
#: k in {1, 3} x {plain vote, distance-weighted} = 4 cells per metric,
#: ALL reported, none selected after the fact. **The admissibility
#: arithmetic, corrected at registration** (the fifth-arm precedent --
#: the stated reasoning was checked before it shipped): anchors per grade
#: are 3/7/6/6/3, and under strict majority (more than k/2) the extreme
#: grades fail OUTRIGHT at k >= 6 (four votes needed, three anchors
#: exist); at k = 4 and k = 5 they remain reachable only when EVERY
#: extreme-grade anchor is simultaneously among the k nearest -- a
#: near-degenerate condition no classifier should hinge on -- and even k
#: invites ties besides. k = 3 is the largest k with slack (2 of 3
#: suffice), so {1, 3} is the registered set, standing on the corrected
#: arithmetic. k=1's two vote rules coincide by construction and are
#: still reported as declared.
#:
#: **Distance**: Euclidean PRIMARY -- consistency with every Phase 9
#: measurement in this space (t-SNE, medoids, LOO, bootstrap); cosine as
#: the registered SECONDARY (the one defensible alternative for ViT
#: embeddings), same 4 cells, reported beside. Necessary defaults,
#: committed blind: a tied vote is broken by the nearest neighbour among
#: the tied grades (the standing tie-rule family); an exact-duplicate
#: anchor (zero distance) decides its patient outright in the weighted
#: cells; a zero-norm embedding under cosine is a fault, refused.
#:
#: **Evaluation, all standing conventions, n = 237**: predicted grade per
#: patient -> PCC and Spearman against the panel MEAN; 3-class accuracy
#: against class3 -- the predicted integer grade collapsed at the FIXED
#: thresholds 2.5/3.5 (1,2 -> low; 3 -> mid; 4,5 -> high), never tuned --
#: with majority (0.502) and chance (0.333) printed beside. **Anchor
#: self-consistency alongside**: leave-one-out over the 25 anchors --
#: does each anchor's nearest fellow anchor share its grade? -- the
#: measurement any prototype method presupposes, per metric.
#:
#: **The pre-committed prediction, both readings registered before any
#: number.** Three convergent priors -- the t-SNE companion 0.409 (below
#: majority), the medoid companion 0.257 (below chance), and the trained
#: head needing 769 fitted parameters to reach 0.2520 -- all predict the
#: raw-distance classifier lands near or below chance. If it does: the
#: THIRD convergent measurement of the phase's central finding, by the supervision material's
#: own suggested method. If it beats the priors: genuinely surprising --
#: the feature space carries grade structure that the trained head's
#: neighbourhood statistics missed -- and that would matter more.
#:
#: **Inputs**: the cohort's cached arm-A embeddings and the labels CSV
#: via the verified ``deall_set`` declaration -- and one correction to
#: the all-existing premise, owned at registration: the -3 external-
#: reference run persisted its SCORES and SHEET, not the 25 features (no
#: embeddings npz was ever written -- the builder's own choice), so this
#: task RE-EXTRACTS the 25 live through the same recorded path
#: (stage -> FrozenExtractor) with the live-path parity check repeated.
#: A small GPU task, as the task spec must therefore say.
PROTOTYPE_CLASSIFIER_REGISTERED = {
    "registered": "2026-08-16, before any number",
    "intent": (
        "the 25 graded images are trusted references; extract their "
        "features; classify each cohort patient by comparing their "
        "features to the anchors' -- no training anywhere (the supervision material's "
        "reading, near-verbatim via the maintainer)"
    ),
    "anchor_grade_caveat": (
        "trusted SINGLE grades from the survey lineage (Image-codes key "
        "+ CSV), not verified-unanimous; supervision can confirm unanimity later"
    ),
    "cells": {
        "k": (1, 3),
        "votes": ("plain", "weighted"),
        "all_reported": "none selected after the fact",
        "admissibility": (
            "3/7/6/6/3 anchors per grade: strict majority fails outright "
            "for extremes at k>=6; k=4,5 reachable only with every "
            "extreme anchor simultaneously nearest (near-degenerate), "
            "even k invites ties; k=3 is the largest k with slack -- "
            "{1,3} stands on this corrected arithmetic"
        ),
        "k1_note": "k=1's two vote rules coincide by construction",
    },
    "metrics": {
        "primary": "euclidean -- consistency with every Phase 9 measurement",
        "secondary": "cosine -- the one defensible alternative, beside",
    },
    "defaults_committed_blind": (
        "tie -> nearest neighbour among tied grades; zero-distance "
        "duplicate decides outright in weighted cells; zero-norm vector "
        "under cosine refused",
    ),
    "evaluation": {
        "n": 237,
        "against_mean": "PCC and Spearman",
        "against_class3": (
            "3-class accuracy, predicted grade collapsed at fixed "
            "2.5/3.5; majority 0.502 and chance 0.333 printed beside"
        ),
        "self_consistency": (
            "LOO over the 25 anchors: nearest fellow anchor shares the "
            "grade? -- per metric"
        ),
    },
    "prediction_committed_both_ways": {
        "priors": (
            "t-SNE companion 0.409 (below majority); medoid companion "
            "0.257 (below chance); 769 fitted parameters needed for "
            "0.2520"
        ),
        "if_near_or_below_chance": (
            "the third convergent measurement of the phase's central "
            "finding, by the supervision material's own suggested method"
        ),
        "if_it_beats_the_priors": (
            "genuinely surprising: the space carries grade structure the "
            "trained head's neighbourhood statistics missed -- and that "
            "would matter more"
        ),
    },
    "inputs_correction": (
        "the -3 run persisted scores and sheet, NOT the 25 features; "
        "this task re-extracts the 25 live through the same recorded "
        "path, parity check repeated -- a small GPU task"
    ),
}


def grade_to_class3(grade: float) -> int:
    """The manifest's own collapse at the FIXED thresholds 2.5/3.5."""
    if grade < 2.5:
        return 0
    if grade > 3.5:
        return 2
    return 1


def _anchor_distances(anchor_features, features, metric: str):
    anchors = np.asarray(anchor_features, dtype=np.float64)
    rows = np.asarray(features, dtype=np.float64)
    if metric == "euclidean":
        deltas = rows[:, None, :] - anchors[None, :, :]
        return np.sqrt(np.sum(deltas * deltas, axis=2))
    if metric == "cosine":
        anchor_norms = np.linalg.norm(anchors, axis=1)
        row_norms = np.linalg.norm(rows, axis=1)
        if np.any(anchor_norms == 0) or np.any(row_norms == 0):
            raise ValueError(
                "a zero-norm embedding has no direction; cosine distance "
                "is undefined and the vector is a fault, not a datum"
            )
        similarity = (rows @ anchors.T) / np.outer(row_norms, anchor_norms)
        return 1.0 - similarity
    raise ValueError(f"unknown metric {metric!r}")


def anchor_knn_grades(anchor_features, anchor_grades, features, *,
                      k: int, weighted: bool, metric: str):
    """The prototype classifier's one operation: each row's grade from
    its k nearest anchors (``PROTOTYPE_CLASSIFIER_REGISTERED`` -- no
    training anywhere). Ties break to the nearest neighbour among the
    tied grades; a zero-distance duplicate decides outright in the
    weighted cells."""
    anchor_grades = np.asarray(anchor_grades)
    distances = _anchor_distances(anchor_features, features, metric)
    order = np.argsort(distances, axis=1, kind="stable")
    out = []
    for row in range(distances.shape[0]):
        nearest = order[row, :k]
        if k == 1:
            out.append(int(anchor_grades[nearest[0]]))
            continue
        if weighted and np.any(distances[row, nearest] == 0.0):
            duplicate = nearest[distances[row, nearest] == 0.0][0]
            out.append(int(anchor_grades[duplicate]))
            continue
        tallies: dict = {}
        for anchor in nearest:
            grade = int(anchor_grades[anchor])
            weight = (
                1.0 / float(distances[row, anchor]) if weighted else 1.0
            )
            tallies[grade] = tallies.get(grade, 0.0) + weight
        best = max(tallies.values())
        tied = {g for g, tally in tallies.items() if np.isclose(tally, best)}
        if len(tied) == 1:
            out.append(tied.pop())
        else:
            out.append(next(
                int(anchor_grades[a]) for a in nearest
                if int(anchor_grades[a]) in tied
            ))
    return np.array(out)


def anchor_self_consistency(anchor_features, anchor_grades, *, metric: str) -> dict:
    """LOO over the anchors: does each anchor's nearest FELLOW anchor
    share its grade? The measurement any prototype method presupposes."""
    anchor_grades = np.asarray(anchor_grades)
    distances = _anchor_distances(anchor_features, anchor_features, metric)
    np.fill_diagonal(distances, np.inf)
    nearest = np.argmin(distances, axis=1)
    agreement = (anchor_grades[nearest] == anchor_grades)
    return {
        "fraction": float(agreement.mean()),
        "agreeing": int(agreement.sum()),
        "n": int(len(anchor_grades)),
    }



#: **[OBSERVED 2026-08-16, against the registration] THE PROTOTYPE
#: CLASSIFIER RAN -- one attempt, parity 1.53e-05 -- AND THE REGISTERED
#: NEAR-CHANCE READING FIRED.**
#:
#: The cells (PCC vs mean / 3-class accuracy; chance 0.3333, majority
#: 0.502): Euclidean k1 0.1823/0.3376, k3 0.1494/0.3333; cosine k1
#: 0.1531/0.3502, k3 0.1573/0.3629. **Stated precisely**: every cell sits
#: within 0.03 of chance and far below majority -- the registered
#: "near or below chance" fired; the two cosine cells sit MARGINALLY
#: ABOVE chance, so "at or below chance" would overstate and is not the
#: record's phrasing. PCC 0.15-0.18 everywhere, all under the trained
#: head's 0.2520. The k=1 vote cells coincide BY CONSTRUCTION (as
#: registered); the k=3 plain and weighted cells returned identical
#: numbers IN THIS RUN -- an observed property of this data, NOT an
#: arithmetic identity: a singleton nearest anchor can outweigh a
#: same-grade pair under inverse-distance weighting, and the suite's own
#: tests construct exactly that flip.
#:
#: **The sharpest finding: the anchors themselves do not neighbour by
#: grade.** Self-consistency 4/25 (Euclidean) and 3/25 (cosine) against
#: a chance expectation of ~4.75/25 -- the arithmetic, recorded beside
#: the observed: with grade counts 3/7/6/6/3, a randomly drawn fellow
#: anchor shares the grade with probability sum(n_g*(n_g-1))/(25*24) =
#: (6+42+30+30+6)/600 = 114/600 = 0.19, i.e. 4.75 of 25. So the method's
#: premise FAILS AMONG THE TRUSTED REFERENCES before any cohort patient
#: is scored.
#:
#: **The registered corroboration reading applies verbatim**: this is
#: the FOURTH convergent measurement that arm A's feature-space
#: neighbourhoods do not carry clinical grade -- the t-SNE companion
#: (0.409, below majority), the medoid companion (0.257, below chance),
#: the bootstrap persistence (~0.4 on all three medoids), and now the
#: anchor classifier by the supervision material's own suggested method -- the last being the
#: most persuasive form for supervision conversation. The anchor-grade
#: provenance caveat travels with every quotation: trusted single grades
#: from the survey lineage, not verified-unanimous.
PROTOTYPE_CLASSIFIER_OBSERVED = {
    "observed": "2026-08-16",
    "run": "p9_prototype_classifier__18d0d9fb__p9-prototype-classifier",
    "attempts": 1,
    "live_path_parity_max_abs": 1.53e-05,
    "cells": {
        "euclidean": {
            "k1": {"pcc": 0.1823, "acc3": 0.3376},
            "k3": {"pcc": 0.1494, "acc3": 0.3333},
        },
        "cosine": {
            "k1": {"pcc": 0.1531, "acc3": 0.3502},
            "k3": {"pcc": 0.1573, "acc3": 0.3629},
        },
        "baselines": {"chance": 0.3333, "majority": 0.502},
        "stated_precisely": (
            "every cell within 0.03 of chance, far below majority; the "
            "two cosine cells sit marginally ABOVE chance -- 'at or "
            "below' would overstate"
        ),
        "vote_coincidences": (
            "k=1 by construction (registered); k=3 observed in this "
            "data, not an arithmetic identity -- the suite constructs "
            "the counterexample flip"
        ),
    },
    "pcc_vs_trained_head": "0.15-0.18 everywhere, under 0.2520",
    "anchor_self_consistency": {
        "euclidean": "4/25",
        "cosine": "3/25",
        "chance_expectation": (
            "~4.75/25: sum(n_g*(n_g-1))/(25*24) = 114/600 = 0.19 from "
            "the 3/7/6/6/3 grade counts"
        ),
        "reading": (
            "the method's premise fails among the trusted references "
            "before any cohort patient is scored"
        ),
    },
    "convergence": (
        "the FOURTH convergent measurement that arm A's feature-space "
        "neighbourhoods do not carry clinical grade: t-SNE companion "
        "0.409; medoid companion 0.257; bootstrap persistence ~0.4; the "
        "anchor classifier by the supervision material's own suggested method -- the most "
        "persuasive form for supervision conversation"
    ),
    "caveat": (
        "anchor grades are trusted single grades from the survey "
        "lineage, not verified-unanimous; travels with every quotation"
    ),
    # [REFINED 2026-08-29, from Phase 16 -- a refinement, NOT a
    # retraction. Everything above is preserved as written.]
    "readout_refined_2026_08_29": (
        "**part of pass zero's failure was the READOUT, not the "
        "feature space.** Phase 16's identity baseline -- the SAME "
        "space, the SAME 25 anchors, a softmax-expectation readout "
        "instead of k-NN voting, W = I so nothing is trained -- scores "
        "0.2151 (deterministic; sd ~3e-17) where the k-NN best cell "
        "above is 0.1823. This does NOT overturn the "
        "four-convergent-nulls conclusion: 0.2151 remains well below "
        "the trained probe's 0.2520, and the fifth convergent "
        "measurement (Phase 16's ledgered UNRESOLVED) points the same "
        "way. But it softens one leg of an argument quoted repeatedly "
        "-- 'the anchor classifier lands near chance' is partly a "
        "fact about voting. **Both readouts are deterministic single "
        "values with no seed variance, so the criterion cannot apply: "
        "DESCRIPTIVE only.** Pass zero's numbers stand; the reading "
        "of its null changes slightly "
        "(phase16.PHASE_16_CLOSING['criterion_6_identity_baseline'], "
        "phase16.IDENTITY_BASELINE)"
    ),
}



#: **[CLOSED 2026-08-16, on confirmation] PHASE 9 IS CLOSED.**
#: Everything below is already recorded; this record only cites it, and
#: NO NEW CLAIM is made in closing.
#:
#: **The five exit criteria, all MET** (``PHASE_9_EXIT_CRITERIA``):
#:
#: 1. Choices before computation -- every registration preceded its
#:    numbers, through all five rounds of the phase.
#: 2. Prototypes -- MET under the dated amendment
#:    (``criterion_2_amended``): "prototypes" resolved to the supervision anchor
#:    reading; the medoid work stands under its corrected label as
#:    DESCRIPTIVE (``PROTOTYPES_OBSERVED``, its committed weak-stability
#:    reading fired); the anchor classifier registered and measured.
#: 3. External reference -- the 25-set staged and scored: **PCC 0.2512
#:    at n=25, the first directly-placed number beside CleftGNN
#:    Table 5**, sheet reviewed and PASSED, never pooled with cohort
#:    results; the first launch's blocking issue recorded and resolved
#:    (ledger ``void-deall-first-launch``).
#: 4. The ledger -- born 2026-08-16 with 17 entries, append-only
#:    enforced by the pinned checksum chain, FOUR PINS DEEP (17/18/19/20)
#:    and exercised by three real appends plus one caught mid-tuple
#:    violation by its own author.
#: 5. Suite green.
#:
#: **What the phase decided and measured**: the framing decision
#: (``ROAD_B_IS_THE_ANNEX`` -- main line and annex, convergence dropped,
#: nothing demoted evidentially); the cohort medoids as descriptive
#: artifacts under their corrected label; the external reference beside
#: the group's published benchmark; and the anchor classifier's
#: near-chance result as the FOURTH convergent measurement that arm A's
#: feature-space neighbourhoods do not carry clinical grade -- by the supervision material's
#: own suggested method, the most persuasive form for supervision
#: conversation (``PROTOTYPE_CLASSIFIER_OBSERVED``, ledger
#: ``p9-anchor-classifier-convergence``).
#:
#: **Carried forward OPEN, non-blocking, by name**:
#:
#: 1. the unanimity confirmation on the anchor grades (the provenance
#:    caveat on every classifier quotation until then).
#: 2. ``phase8.RETRY_LIMIT_IS_NOT_HOLDING`` -- the standing
#:    infrastructure item, now eleven attempts of evidence.
#: 3. ``PRE_CONTEXT_REFUSALS_LEAVE_NO_TRACE`` -- to consider,
#:    deliberately not built.
#: 4. The 25-set's ``-lips``/``-nose`` views -- present, available,
#:    never staged.
PHASE_9_CLOSING = {
    "closed": "2026-08-16, on confirmation",
    "criteria": {
        "1_choices_first": "MET -- every registration preceded its numbers",
        "2_prototypes": (
            "MET under the dated amendment: the supervision anchor reading; "
            "medoids descriptive under the corrected label; the anchor "
            "classifier measured"
        ),
        "3_external_reference": (
            "MET -- PCC 0.2512 at n=25 beside CleftGNN Table 5, sheet "
            "PASSED, never pooled; the first launch's block recorded "
            "and resolved"
        ),
        "4_ledger": (
            "MET -- born with 17, four pins deep (17/18/19/20), "
            "append-only with teeth"
        ),
        "5_suite": "MET -- green",
    },
    "decided_and_measured": {
        "framing": "ROAD_B_IS_THE_ANNEX",
        "medoids": (
            "descriptive artifacts under the corrected label; "
            "representativeness dead"
        ),
        "external_reference": "0.2512 beside CleftGNN Table 5",
        "anchor_classifier": (
            "near chance in all eight cells; the anchors do not "
            "neighbour by grade; the FOURTH convergent measurement, by "
            "the supervision material's own method"
        ),
    },
    "carried_forward_open": (
        "the unanimity confirmation on the anchor grades",
        "phase8.RETRY_LIMIT_IS_NOT_HOLDING",
        "phase9.PRE_CONTEXT_REFUSALS_LEAVE_NO_TRACE",
        "the -lips/-nose views, present and never staged",
    ),
    "no_new_claims": (
        "everything cited above was recorded when it happened; the "
        "closing adds nothing"
    ),
}



#: **[CONCEDED 2026-09-02 -- a dated addendum to a closed phase, on the
#: phase18 precedent] EXTERNAL VALIDATION OF THE PROBE AGAINST THE DEALL
#: 25 IS NOT AVAILABLE, AND THE IDEA MUST NOT ARRIVE FRESH AGAIN.**
#:
#: **How it arrived.** it was proposed it in conversation **without
#: knowing Phase 9 had already done it** -- which is exactly the failure
#: this record exists to prevent, and is why the concession is written
#: beside the set rather than in whatever module happens to be current.
#: A reader who has the idea will look here.
#:
#: **Nothing below is new measurement.** Every ground cites a record
#: that already existed when the proposal was made.
EXTERNAL_VALIDATION_ON_THE_25_CONCEDED = {
    "conceded": "2026-09-02",
    "the_proposal": (
        "score the probe on the Deall 25 as EXTERNAL VALIDATION against "
        "a cleaner label -- a unanimous surgeon consensus -- and read a "
        "higher number as generalisation"
    ),
    "how_it_arrived": (
        "**it was proposed it without knowing Phase 9 had already run "
        "it.** The proposal was not checked against "
        "DEALL_REFERENCE_REGISTERED before it was made. **That is the "
        "provenance, and it is recorded rather than smoothed**: the "
        "idea did not survive verification, and it arrived only because "
        "verification came second"
    ),

    # ---- ground 1 ----------------------------------------------------
    "1_already_measured": (
        "**PROTOTYPE_CLASSIFIER_OBSERVED, 2026-08-16**, tested "
        "cohort-trained arm A against these 25: 3-class accuracy "
        "**0.3376 / 0.3333 (Euclidean k=1, k=3) and 0.3502 / 0.3629 "
        "(cosine)** against chance 0.3333 and majority 0.502 -- 'every "
        "cell within 0.03 of chance'. PCC **0.15-0.18 everywhere, under "
        "the trained head's 0.2520**. And the external reference itself "
        "ran: **PCC 0.2512 at n=25** "
        "(PHASE_9_CLOSING criterion 3). **The measurement the proposal "
        "asked for exists**"
    ),
    "1a_the_self_consistency_and_the_records_OWN_reading": (
        "anchor self-consistency **4/25 Euclidean, 3/25 cosine against "
        "a chance expectation of ~4.75/25** "
        "(sum(n_g(n_g-1))/(25*24) = 114/600 from the 3/7/6/6/3 counts). "
        "**[The record's reading is WITHIN-SET, and it is quoted rather "
        "than paraphrased:** 'the method's premise fails among the "
        "trusted references before any cohort patient is scored'. "
        "**The proposal's framing -- 'the arms measured a CROSS-SET GAP "
        "rather than prototype coherence' -- is NOT in the record**, "
        "and the distinction matters: a cross-set gap would leave "
        "external validation open (the 25 merely out of distribution), "
        "whereas the banked reading is that **grade structure is absent "
        "among the 25 THEMSELVES**, which is the stronger ground.]**"
    ),
    "1b_the_refinement_that_travels": (
        "``readout_refined_2026_08_29`` is preserved and travels: part "
        "of pass zero's failure was the READOUT, not the space -- the "
        "identity baseline scores 0.2151 against k-NN's 0.1823. **It "
        "does not overturn the conclusion** (0.2151 is still below "
        "0.2520) and it is cited here so the concession does not rest "
        "on a leg the record itself softened"
    ),

    # ---- ground 2 ----------------------------------------------------
    "2_the_label_is_not_comparable": (
        "**a different quantity on BOTH scale and construction.** The "
        "237's target is the panel MEAN of five integer grades, so it "
        "is continuous on rater-steps of **0.2** (data.softlabels: "
        "'soft labels are fractions of five raters, so every value is a "
        "multiple of 0.2'). The Deall ``Score`` is an **INTEGER 1-5**. "
        "A correlation against one is not a correlation against the "
        "other, and the -3 run said so at registration: 'generalisation "
        "to a DIFFERENT label ... **external reference only, never "
        "pooled with cohort results**'"
    ),

    # ---- ground 3 ----------------------------------------------------
    "3_the_reliability_is_not_computable": (
        "**and the guard refuses rather than returning a number.** A "
        "unanimous consensus is ONE column, and every reliability route "
        "in the repository needs a rater MATRIX: "
        "``mean_inter_rater_r``, ``fleiss_kappa``, ``cronbach_alpha`` "
        "and ``mean_pairwise_qwk`` all raise "
        "``ReliabilityError('need at least 2 raters, got 1')`` through "
        "``reliability._check``. **Zero within-set rater variance means "
        "there is nothing to average**"
    ),
    "3a_and_assuming_1_0_would_be_the_hazard_not_the_fix": (
        "**a hypothetical unanimous FIVE-column panel gives mean_r = "
        "1.0, Fleiss 1.0, reliability 1.0 and a ceiling of exactly "
        "1.0** -- checked. But **those per-rater grades do not exist in "
        "our record**; only the single Score column does. So 1.0 would "
        "be inferred FROM THE WORD 'unanimous', not measured. **The "
        "comparison the proposal wanted -- 0.2512 against 0.2520 -- "
        "would then be a model scored against a ceiling of 1.0 placed "
        "beside one scored against 0.9032**, with the difference "
        "attributed to the model. ``phase24.A_CEILING_IS_NOT_A_SCORE`` "
        "and ``data/reliability``'s opening ('confusing them has "
        "already cost this project once') both bite here"
    ),

    # ---- ground 4 ----------------------------------------------------
    "4_unanimity_is_REPORTED_not_banked": (
        "**[REPORTED], and it answers a STANDING ASK rather than "
        "restating something banked.** The record says the opposite "
        "twice, deliberately: DEALL_REFERENCE_READS -- 'whether this "
        "key equals the manuscript's 27-surgeon highest-agreement "
        "consensus; both sources stated, **no reconciliation forced**' "
        "-- and PROTOTYPE_CLASSIFIER_REGISTERED -- 'trusted SINGLE "
        "grades from the survey lineage ..., **not verified-unanimous**; "
        "supervision can confirm unanimity later'. It is carried OPEN by name in "
        "PHASE_9_CLOSING and is **supervision ask 7** (phase11), still inherited "
        "unchanged by phase16's closing. **A label construction stated "
        "by the maintainer is [REPORTED] until it reaches the record with "
        "a source**"
    ),
    "4a_and_the_two_phrases_are_not_the_same_claim": (
        "**'unanimous consensus' is not what the manuscript claims.** "
        "Its claim (phase10.CLEFTGNN_COMPARATOR_TABLES) is 'the 25-of-76 "
        "**highest-agreement** images from a separate 27-surgeon study' "
        "-- **a SELECTION CRITERION over 76 images, not unanimity among "
        "27 raters on these 25**. The two license different arithmetic "
        "in ground 3a, so they are kept apart"
    ),

    # ---- what the concession does and does not do --------------------
    "what_this_does_NOT_forbid": (
        "**the 25 remain in use and nothing about them is withdrawn.** "
        "The external reference 0.2512 stands as recorded; the anchor "
        "set ``anchor_deall_v1`` stands; Phase 16's anchor loop stands. "
        "**What is conceded is the VALIDATION FRAMING** -- reading a "
        "number on the 25 as evidence that the probe generalises"
    ),
    "what_would_reopen_it": (
        "**a banked reliability figure for the 25's label**, with its "
        "source, letting the ceiling ratio be stated BEFORE the "
        "comparison; or per-rater grades for the 25, which would make "
        "the reliability computable rather than assumed. the supervision material's "
        "unanimity confirmation ALONE does not reopen it -- it would "
        "close ask 7 and still leave ground 2 and the missing ceiling "
        "arithmetic standing"
    ),

    # ---- two record defects found by the verification -----------------
    "found_the_0_2512_IS_NOT_LEDGERED": (
        "**phase10.CLEFTGNN_COMPARATOR_TABLES cites it as '(Phase 9 "
        "external reference, ledger)' and its docstring as '(Phase 9, "
        "ledger; never pooled)'. NO LEDGER ROW CARRIES 0.2512.** The "
        "ledger's only Deall rows are 18 (``void-deall-first-launch``, "
        "VOID) and 19 (``p9-anchor-classifier-convergence``). The figure "
        "is recorded here and in PHASE_9_CLOSING, and **misattributed to "
        "the ledger in two places**. Flagged, not corrected -- the fix "
        "is a maintainer decision"
    ),
    "found_the_SPEARMAN_was_computed_and_never_banked": (
        "the registration promised 'PCC **and Spearman** against Score "
        "at n=25' and ``run.task_deall_reference`` computes both "
        "(``metrics.pcc`` and ``metrics.spearman``, logged together). "
        "**Only the PCC reached the record.** The rank correlation is "
        "recoverable from the -3 run's own metrics.json **without "
        "relaunching anything**"
    ),
    "and_the_n_25_interval_nobody_stated": (
        "**PCC 0.2512 at n=25 has a Fisher 95% CI of [-0.1598, "
        "0.5880] -- it SPANS ZERO**, and the |r| a correlation must "
        "reach at n=25 is **0.4179** "
        "(``relevance.significance_threshold(25)``, against 0.1281 at "
        "n=237). Computed at concession time by the same arithmetic "
        "phase10 used for its own n=25 cell (arctanh, se = 1/sqrt(22)), "
        "which reproduces its banked [0.2656, 0.8033] exactly. **The "
        "existing external figure is not distinguishable from no "
        "correlation, and no record said so**"
    ),
    "tag": "[CONCEDED] -- no measurement, no ledger row",
}


def summary() -> dict:
    """The phase's records, importable as one object."""
    return {
        "framing": ROAD_B_IS_THE_ANNEX,
        "prototypes": PROTOTYPES_REGISTERED,
        "external_reference": DEALL_REFERENCE_REGISTERED,
        "external_validation_conceded": EXTERNAL_VALIDATION_ON_THE_25_CONCEDED,
        "exit_criteria": PHASE_9_EXIT_CRITERIA,
    }
