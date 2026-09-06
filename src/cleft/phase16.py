"""Phase 16 -- the anchor loop.

Scheduled by the fourth sequence amendment
(``phase15.PHASE_SEQUENCE_RENUMBERED_4``): the 25-image design, promoted
from a noted item to a phase in its own right. **Its registration is
``phase15.ANCHOR_LOOP_REGISTERED`` and it stands unchanged** -- this
module does not restate it, it extends it: the restate rulings, the
pre-run readings, the primary contrast, and the measured compute shape.

The module exists because the registration material outgrew phase15,
which is a CLOSED phase's record file; the promotion gave the loop its
own number, so it gets its own module, with pointers both ways.

**[CLOSED 2026-08-29]** (``PHASE_16_CLOSING``): built, run
(p16_anchor_loop__f342fed9), ledgered UNRESOLVED
(``p16-anchor-loop-unresolved``), and closed with
``REPAIR_WITHOUT_TRANSFER`` as the phase's finding.
"""

from __future__ import annotations

from pathlib import Path

#: **[RULED 2026-08-29] THE THREE RESTATE DECISIONS.**
ANCHOR_LOOP_RULINGS = {
    "ruled": "2026-08-29, the maintainer, on the three open restate decisions",
    "registration": (
        "phase15.ANCHOR_LOOP_REGISTERED, standing unchanged -- these "
        "rulings bind its open parameters; they amend nothing in it"
    ),

    # ---- decision 1: the training target -----------------------------
    "target_is_continuous": (
        "**the pull loss and the readout train against the same "
        "CONTINUOUS panel mean as the 0.2520 probe** -- the maintainer's "
        "ruling, overriding the lean toward the 3-class variant"
    ),
    "the_rationale": (
        "a 3-class grouping of the 25 anchors into 10/6/9 discards "
        "exactly the SUB-GRADE structure registration item 1 exists "
        "for -- each anchor its own prototype BECAUSE a grade-3 face "
        "can be grade-3 for different reasons -- and the continuous "
        "target makes the primary contrast LIKE-FOR-LIKE across target "
        "types: loop vs probe differ in mechanism, not in what they "
        "are asked to predict"
    ),

    # ---- explicitly unregistered, Phase-14 style ----------------------
    "unregistered_not_deferred": (
        "**three variants are UNREGISTERED, by name, with reasons -- "
        "the Phase-14 pattern**: naming them now is what stops any of "
        "them being presented later as a fresh idea after the "
        "registered arm's number is known"
    ),
    "unregistered_a_3class_pull_loss": (
        "(i) the 3-CLASS PULL-LOSS VARIANT (option a): grouping "
        "anchors 10/6/9 by class3 before pulling. Unregistered because "
        "it discards the sub-grade structure the anchors exist to "
        "provide, and because a second target type would make the "
        "primary contrast a two-variable comparison"
    ),
    "unregistered_b_diagonal_w": (
        "(ii) DIAGONAL ``W``: a per-dimension rescale. Unregistered "
        "because it cannot rotate the space -- it can only reweight "
        "axes the frozen backbone happened to choose, and the four "
        "convergent nulls already say grade is not axis-aligned there"
    ),
    "unregistered_c_lowrank_w": (
        "(iii) LOW-RANK ``W``: a bottlenecked correction. Unregistered "
        "because rank is a tunable capacity knob with no registered "
        "value, and a rank chosen after seeing numbers is the "
        "threshold-moved-after-the-data shape in a new coat"
    ),

    # ---- the registered default correction ----------------------------
    "registered_correction": (
        "**full linear ``W``, 768x768, initialised at IDENTITY, weight "
        "decay TOWARD IDENTITY** (not toward zero -- zero decay would "
        "pull the metric toward collapse; identity decay pulls it "
        "toward the uncorrected space, so the null hypothesis is the "
        "resting state and any departure is paid for by the loss)"
    ),
    "weight_decay_is_a_scientific_setting": (
        "**the decay strength is declared in the YAML before the first "
        "run and NEVER tuned across runs; movement requires a dated "
        "amendment with a reason.** The config machinery already "
        "enforces the detectable half: the config will be generated, "
        "and the generator's --check turns any silent edit into DRIFT; "
        "the schema will carry the key as required-no-default so it "
        "cannot arrive implicitly. What machinery cannot enforce -- "
        "that a MOTIVATED retune arrives dressed as an amendment -- is "
        "what this clause is for, and the clause is the record"
    ),
    "epoch_budget": (
        "**FIXED epoch budget with the best inner-validation checkpoint "
        "kept; NO patience.** Patience is a stopping rule tuned by the "
        "data that triggers it; a fixed budget with checkpoint "
        "selection spends the same compute in every fold x seed and "
        "selects the same way the probe's recipe does"
    ),
    "the_forbidden_variant_restated": (
        "the no-folds full-cohort assignment loop remains named "
        "MEMORISATION-AND-FORBIDDEN "
        "(phase15.ANCHOR_LOOP_REGISTERED['the_forbidden_version_named'])"
        " -- restated at ruling time so no ruling can be read as "
        "softening it"
    ),
}


#: **[REGISTERED 2026-08-29, PRE-RUN] THE SELF-CONSISTENCY READINGS.**
SELF_CONSISTENCY_READINGS = {
    "registered": "2026-08-29, before any corrected metric exists",
    "the_measurement": (
        "the 25-anchor leave-one-out self-consistency, recomputed under "
        "EACH fold x seed corrected metric, reported PER FOLD x SEED "
        "against the banked uncorrected baseline: **4/25 euclidean, "
        "3/25 cosine, with ~4.75/25 the chance expectation** from the "
        "3/7/6/6/3 grade counts "
        "(phase9.PROTOTYPE_CLASSIFIER_OBSERVED['anchor_self_consistency']"
        "). The chance line is printed beside every value"
    ),
    "reading_if_repaired": (
        "**if the corrected metric makes the anchors neighbour by "
        "grade** -- self-consistency claimably above the ~4.75/25 "
        "chance line -- **that is the phase's CLEANEST POSITIVE, "
        "whatever the cohort-side number does**: the premise that "
        "failed before any cohort patient was scored is repaired by "
        "the learned metric, on the 25 faces the loss never trained on "
        "individually"
    ),
    "reading_if_not_repaired": (
        "**if self-consistency stays at its uncorrected level, the "
        "premise fails UNDER CORRECTION TOO** -- and any cohort-side "
        "result becomes harder to attribute to the anchors: a loop "
        "that scores patients well while its own references do not "
        "neighbour by grade is doing something, but not the thing the "
        "design says it does"
    ),
    "why_pre_run": (
        "both readings committed before a single corrected metric "
        "exists, so neither can be written to fit the number -- the "
        "same both-ways discipline the registration's cohort-side "
        "readings carry"
    ),
}


#: **[REGISTERED 2026-08-29] THE PRIMARY CONTRAST. No runner exists.**
PRIMARY_CONTRAST_REGISTERED = {
    "registered": "2026-08-29, registration entry only -- no runner",
    "the_contrast": (
        "anchor-loop OOF predictions vs the 0.2520 probe "
        "(p7_d1_vit_b16_imagenet_g1), **paired BCa with 10,000 "
        "resamples, five seeds, BOTH criterion conditions** -- the "
        "standing machinery, nothing invented for this phase"
    ),
    "pass_zero_beside_it": (
        "pass zero's best cell is reported DESCRIPTIVELY beside the "
        "contrast (euclidean k=1, PCC 0.1823 -- "
        "phase9.PROTOTYPE_CLASSIFIER_OBSERVED), never inside it: the "
        "un-looped baseline a reader needs to see what the loop bought"
    ),
    "the_prediction": (
        "**COHORT_CANNOT_RESOLVE predicts UNRESOLVED** -- at n=237 the "
        "paired interval is expected to span zero, as it did for 29 of "
        "30 comparisons on the main line. Success is "
        "PARITY-WITH-EXPLANATIONS "
        "(phase15.ANCHOR_LOOP_REGISTERED"
        "['success_is_defined_before_the_run']), not victory"
    ),
}


#: **[MEASURED 2026-08-29, ON THE LAPTOP] THE COMPUTE SHAPE -- the
#: [REASONED] tag discharged by ls and source, not by memory.**
COMPUTE_SHAPE_VERIFIED = {
    "verified": (
        "2026-08-29, from the shipped configs, the task source, and ls "
        "-- with the standing bound that data artifacts live on the "
        "cluster, so local existence was CHECKED (absent, as tiering "
        "predicts: local data/ holds only scut/ and smoke/) and "
        "cluster existence is attested by declared verified hashes and "
        "the banked runs that read them"
    ),

    "cohort_embeddings": (
        "**EXIST as a cached artifact.** "
        "data/embeddings/embeddings_g1_ladder_v1/vit_b16__imagenet__g1 "
        "(cluster), rollup ba4b12535f60aca4cc679246a60ecbcdc880c5ca"
        "66319937126e4f85c632d7ab, produced by p7_extract_g1_ladder "
        "(out_version embeddings_g1_ladder_v1). Shape (237, 768) "
        "float32 values.npy + metadata.json -- the ladder set format. "
        "**Declared at the IDENTICAL path and hash in both the probe "
        "config and the pass-zero config**, so the two consumers "
        "verifiably read the same artifact and the loop would be the "
        "third"
    ),
    "anchor_embeddings": (
        "**DO NOT EXIST as an artifact -- verified in source, not "
        "assumed.** The p9-anchor-classifier-convergence lineage "
        "re-extracted the 25 LIVE (stage -> FrozenExtractor) from the "
        "declared deall_set (${CLEFT_ANCHOR_SET}, 'All 25 images as "
        "JPG', rollup 7312c11290c4652a6caf9d306f662c50a0fba0257"
        "45e6339c03e6173ad72b293) and persisted only "
        "prototype_classifier_grades.csv and metrics.json: "
        "task_prototype_classifier contains zero np.save/savez calls. "
        "The registration said why -- the -3 external-reference run "
        "persisted scores, not features, by its builder's own choice"
    ),

    "conclusion": (
        "**(b), for the anchors only.** The cohort side is (a): cached "
        "768-d features, CPU arithmetic, pod-eligible. The 25 anchors "
        "require RE-EXTRACTION FROM PIXELS -- Phase 16 therefore "
        "inherits pass zero's exact compute shape: one small "
        "pinned-image job that stages the 25 and runs them through the "
        "frozen ViT-B/16 with the live-path parity check repeated "
        "(banked parity 1.53e-05; FrozenExtractor defaults to cuda "
        "with a cpu fallback, so the 25-image forward pass is minutes "
        "on CPU if no GPU is free), after which everything -- the W "
        "training over cached features, the self-consistency "
        "recomputation, the contrast -- is CPU arithmetic"
    ),
    "a_build_option_flagged_not_decided": (
        "the extraction could be persisted ONCE as a new 25-row "
        "artifact in its own namespace, making every later run pure "
        "pod arithmetic -- or re-run live inside the task as pass zero "
        "did, keeping the parity check fresh per run. A build "
        "decision for the build turn, flagged so it is not made by "
        "default"
    ),
    "BOTH_ENTRIES_ABOVE_ARE_SUPERSEDED_2026_09_06": (
        "**the backward pointer this record was missing, added when "
        "Phase 27 read it and was told the artifact does not "
        "exist.** Two entries are the state of an afternoon rather "
        "than the state now, and both are preserved as written. "
        "(1) ``anchor_embeddings`` says they DO NOT EXIST as an "
        "artifact. **They exist.** ``ANCHOR_ARTIFACT_PERSISTED``, "
        "dated the SAME day, decided to persist once, and "
        "``data/embeddings/anchor_deall_v1`` is declared at a "
        "verified rollup in ``configs/p16_anchor_loop.yaml``. "
        "(2) ``a_build_option_flagged_not_decided`` says the "
        "persist-or-re-extract choice is undecided. **It was "
        "decided**, and that record's own ``decided`` field says so "
        "by name: 'the build option COMPUTE_SHAPE_VERIFIED flagged, "
        "now decided: persist once'"
    ),
    "why_the_pointer_was_missing_and_why_it_matters": (
        "**the forward pointer existed and the backward one did "
        "not.** ``ANCHOR_ARTIFACT_PERSISTED`` names this record; "
        "this record named nothing. A reader arriving here first, "
        "which is what a phase reckoning does when it asks what "
        "already exists, is told the artifact is absent and has no "
        "way from here to the record that made it. **A one-way "
        "cross-reference between a superseded entry and its "
        "successor is a cross-reference that only works if you "
        "already knew the answer**"
    ),

    "fold_honesty_of_reuse": (
        "**no entanglement -- stated because item 4 asked.** The "
        "cohort set was extracted ONCE from the frozen backbone before "
        "any fold or seed exists (folds live in the manifest, seeds "
        "only in trained heads), and the anchor extraction is "
        "deterministic with no training and no seed. Reusing both "
        "across fold x seed cells is clean. **The only fold- and "
        "seed-entangled object in Phase 16 is the learned W itself**, "
        "which is trained per fold x seed by design -- that is the "
        "build plan working, not a compromise of it"
    ),
}


class AnchorLoopError(ValueError):
    """A refusal specific to the anchor loop."""


#: **[LOCKED 2026-08-29] THE EXIT CRITERIA. Nothing is added after this
#: record.**
EXIT_CRITERIA = {
    "locked": (
        "2026-08-29 -- **nothing is added after this record**; a "
        "criterion discovered missing later is a limitation of the "
        "lock, recorded as such, never a retro-fitted entry"
    ),
    "criteria": (
        # 1
        "FOLD-HONESTY IN CODE: W trained on training folds only; "
        "anchors belong to no fold; the no-folds full-cohort path "
        "structurally absent -- asserted by tests, not prose",
        # 2
        "5 folds x 5 seeds, OOF reconstruction, REGISTERED METRICS "
        "ONLY: PCC and Spearman vs the mean; 3-class accuracy vs "
        "class3 at 2.5/3.5 with chance 0.333 and majority 0.502 "
        "printed beside",
        # 3
        "corrected 25-anchor self-consistency per fold x seed, against "
        "banked 4/25 euclidean / 3/25 cosine / ~4.75/25 chance",
        # 4
        "the primary contrast (vs p7_d1_vit_b16_imagenet_g1, 0.2520) "
        "under the FULL criterion; outcome recorded "
        "claimable/withdrawn/unresolved with both pre-committed "
        "readings attached",
        # 5
        "mismatch records for all 237, produced regardless of outcome",
        # 6
        "the Identity baseline reported descriptively alongside "
        "(IDENTITY_BASELINE)",
        # 7
        "LOCKED as of 2026-08-29, this record's date",
    ),
}


#: **[RETIRED 2026-08-29, DELIBERATELY] the negative-space test.**
NEGATIVE_SPACE_RETIRED = {
    "retired": "2026-08-29, converted -- not deleted",
    "what_it_was": (
        "test_nothing_is_built_and_the_pointers_hold: no anchor_loop "
        "task, no schema kind, no configs/p16_*"
    ),
    "what_it_guarded": (
        "the gap between the rulings and the exit-criteria lock -- "
        "nothing could be built until the compute verification is "
        "reviewed, so no build could rest on an unreviewed assumption"
    ),
    "what_ended_it": (
        "the maintainer reviewed the verification and locked EXIT_CRITERIA "
        "(2026-08-29). The condition the test held open is discharged"
    ),
    "the_successor": (
        "its positive-space successor asserts the same objects PRESENT "
        "and shaped as the rulings require -- the guard inverted, not "
        "dropped"
    ),
}


#: **[RECORDED 2026-08-29] WHY THE ANCHOR EMBEDDINGS ARE PERSISTED ONCE.**
ANCHOR_ARTIFACT_PERSISTED = {
    "decided": (
        "2026-08-29 -- the build option COMPUTE_SHAPE_VERIFIED flagged, "
        "now decided: persist once"
    ),
    "reason_1_declared_input_purity": (
        "every downstream run declares the SAME hashed artifact instead "
        "of re-deriving its inputs -- the two-pass declare rhythm "
        "covers the anchors exactly as it covers every other input"
    ),
    "reason_2_zero_drift_surface": (
        "no per-run pixel dependency: one extraction instead of five "
        "(or more), so there is ONE drift surface instead of one per "
        "run -- a change in staging or the extractor cannot silently "
        "shift the anchors between seeds"
    ),
    "reason_3_tiering": (
        "CLUSTER-ONLY, like all patient-derived artifacts -- the 25 "
        "Deall images are clinical photographs and their features "
        "follow the images' tier"
    ),
    # [OBSERVED 2026-08-29, at the declare fill] confirmed the
    # extraction log's printed hash prefix-matches the declare output.
    "the_log_printed_the_declarable_hash_deliberately": (
        "**the writer behaving correctly under the two-digest labelling "
        "rule, on both counts.** First: the task recomputes hash_dir "
        "AFTER the writer returns and labels the printed value 'DECLARE "
        "THIS' -- the rule's required shape. Second, structurally: "
        "save_anchor_set writes NO MANIFEST.json (2 files, values.npy + "
        "metadata.json), so here the payload digest and the declarable "
        "digest COINCIDE -- the two-digest trap needs a manifest file "
        "to open the gap, and this writer has none. The declare output "
        "remains the SOLE fill source regardless; the log line is a "
        "cross-check, never a source (49905fe9... prefix confirmed "
        "against declare_inputs.py, 2 files / 77,903 bytes)"
    ),
}


#: **[DECLARED 2026-08-29, NO PRECEDENT TO MIRROR] tau.**
TAU_DECLARED = {
    "why_declared_not_mirrored": (
        "tau has NO precedent in the repo and cannot be mirrored -- "
        "the ruling. Every other setting inherits a value from "
        "the probe lineage; this one is declared with its reasoning "
        "instead"
    ),
    "the_rule": (
        "**tau = tau_scale x the mean squared anchor-anchor distance, "
        "measured ONCE in the UNCORRECTED space and frozen** -- never "
        "recomputed under W (a tau that tracked the corrected space "
        "would co-adapt with the thing being trained), never "
        "recomputed per fold or seed. The softmax argument -d^2/tau "
        "must be dimensionless; the anchor cloud's own scale is the "
        "only unit the design provides, and using it is a units "
        "choice, not a tuned constant. tau_scale = 1.0 in the YAML"
    ),
    "the_degenerate_mode_registered": (
        "**registered IN ADVANCE so the outcome is attributable, not "
        "mysterious**: as tau grows the softmax flattens toward a "
        "constant predictor at the anchor-grade mean (74/25 = 2.96 "
        "from the 3/7/6/6/3 spread), which would present as near-zero "
        "PCC with 3-class accuracy near the 0.502 majority floor -- "
        "the constant lands in the mid class. If the run shows that "
        "signature, the first suspect is tau, and this record says so "
        "before any number exists"
    ),
    "never_tuned": (
        "tau_scale is a scientific setting under the same clause as "
        "lambda: declared before the first run, NEVER tuned across "
        "runs, movement is a dated amendment with a reason"
    ),
}


#: **[RECORDED 2026-08-29] Every setting's provenance, per value.**
SETTINGS_PROVENANCE = {
    "rule": (
        "mirror the closest existing probe-training precedent; report "
        "which config each value came from. The precedent is the "
        "primary contrast's own baseline, so the loop and the probe "
        "differ in mechanism and nothing else"
    ),
    "max_epochs": "40 -- p7_d1_vit_b16_imagenet_g1.yaml max_epochs",
    "learning_rate": "0.001 -- p7_d1_vit_b16_imagenet_g1.yaml learning_rate",
    "batch_size": "32 -- p7_d1_vit_b16_imagenet_g1.yaml batch_size",
    "inner_val_frac": "0.2 -- p7_d1_vit_b16_imagenet_g1.yaml inner_val_frac",
    "seeds": (
        "[1337, 2024, 7, 99, 12345] -- p7_d1_vit_b16_imagenet_g1.yaml "
        "seeds, so the paired contrast pairs by seed"
    ),
    "monitor": "inner_val_mse -- p7_d1_vit_b16_imagenet_g1.yaml monitor",
    "optimiser": (
        "AdamW -- the repo's sole training optimiser (torch_backbone, "
        "pretrain, ldl, graph_cleft all use it), and specifically what "
        "trains p7_d1_vit_b16_imagenet_g1's head"
    ),
    "lambda_identity": (
        "0.01 -- p7_d1_vit_b16_imagenet_g1.yaml weight_decay, playing "
        "the same regularisation role. **Implemented as the explicit "
        "loss term lambda * ||W - I||^2_F with the optimiser's own "
        "weight_decay=0**, because AdamW's built-in decay pulls toward "
        "ZERO and the ruling decays toward IDENTITY -- reusing the "
        "value, not the mechanism"
    ),
    "tau_scale": "1.0 -- NO PRECEDENT; declared with reasoning (TAU_DECLARED)",
    "patience": (
        "ABSENT BY RULING (fixed budget, best checkpoint) -- and absent "
        "from the schema, so a config that writes it is REFUSED as an "
        "unknown key rather than silently ignored"
    ),
    "n_boot": "10000 -- the registered primary contrast; pinned by choices",
}


#: **[RECORDED 2026-08-29] The identity baseline.**
IDENTITY_BASELINE = {
    "added_by": "the maintainer, at the build ruling",
    "what_it_is": (
        "the IDENTICAL softmax-expectation readout with **W = I, "
        "untrained**, same folds and seeds -- the loop's own untrained "
        "score"
    ),
    "distinct_from_pass_zero": (
        "**recorded as DISTINCT from pass zero**: pass zero used k-NN "
        "voting over anchors, a different readout. Its 0.1823 best "
        "cell is not this baseline, and neither may be quoted as the "
        "other"
    ),
    "what_it_buys": (
        "the correction's movement is measured WITHIN ONE READOUT -- "
        "trained W vs identity W through the same softmax expectation "
        "-- rather than inferred across two different readouts"
    ),
    "descriptive_only": (
        "DESCRIPTIVE, no ledger row: the baseline exists to make the "
        "trained number readable, not to be a result"
    ),
}


#: **[RECORDED 2026-08-29] The mismatch record, field by field.**
MISMATCH_RECORD_SPEC = {
    "per_patient": (
        "id, fold, seed, nearest anchor id and grade, distance to it, "
        "ALL 25 distances retained, patient class3, match flag, margin "
        "-- one row per patient per seed, OOF, all 237 regardless of "
        "outcome (EXIT_CRITERIA criterion 5)"
    ),
    "match_flag": (
        "(nearest anchor's grade collapsed at 2.5/3.5) == patient "
        "class3"
    ),
    "margin_semantics": (
        "**margin = d(nearest) - min d over anchors of the patient's "
        "own class, which is 0 by construction whenever the nearest "
        "anchor is already own-class -- so the field is informative on "
        "mismatches only.** Stated here so a column of zeros cannot "
        "read as a finding: on a match row, zero is the definition, "
        "not a measurement"
    ),
    "class3_is_descriptive_only": (
        "the class3 mapping is descriptive readout only, per "
        "classification.THREE_NOT_FIVE -- the training target remains "
        "continuous (ANCHOR_LOOP_RULINGS['target_is_continuous'])"
    ),
}


# --------------------------------------------------------------------------
# the machinery -- pure numpy where testable, torch only inside train_w
# --------------------------------------------------------------------------


def training_indices(folds, held_out: int):
    """Indices of every row NOT in the held-out fold.

    **The no-folds full-cohort path is structurally absent here**: the
    only way to ask for training rows is to name a held-out fold that
    exists, so every training set is a proper subset of the cohort.
    """
    import numpy as np

    folds = np.asarray(folds, dtype=int)
    if held_out not in set(folds.tolist()):
        raise AnchorLoopError(
            f"{held_out} is not a fold in this manifest; the full-cohort "
            "no-holdout call cannot be expressed"
        )
    return np.flatnonzero(folds != held_out)


def parity_reference_row(embeddings_dir, patient_ids):
    """The cached row the live-path parity check compares against.

    **[FIXED 2026-08-29, after run e2b1eaf2 -- four identical load-time
    crashes.]** ``embeddings.load(directory, manifest_ids)`` takes the
    COMPLETE expected id list: it is a verification contract
    (``assert_row_order``), not a row selector. The first version of
    the extraction task passed ``[patient_ids[0]]`` -- one id -- and
    the loader correctly refused: 0 missing, 236 unexpected. The fix is
    pass zero's working shape: load against the FULL manifest list,
    then select the row. Row 0 is the first manifest patient because
    ``assert_row_order`` has just guaranteed artifact order == manifest
    order -- the selection is safe BECAUSE the full-list contract ran.

    The guard below stops the misuse being reintroduced through this
    helper: a truncated list is refused here with the mechanism named,
    before the loader's less specific message.
    """
    from .embeddings import load as load_embeddings

    identifiers = [int(p) for p in patient_ids]
    if len(identifiers) < 2:
        raise AnchorLoopError(
            "parity_reference_row needs the full manifest id list -- "
            "embeddings.load verifies the WHOLE artifact against it "
            "(a verification contract, not a row selector); a "
            f"{len(identifiers)}-element list is the exact misuse that "
            "crashed run e2b1eaf2"
        )
    cached, _ = load_embeddings(Path(embeddings_dir), identifiers)
    return cached[0]


def anchor_tau(anchor_features, tau_scale: float) -> float:
    """TAU_DECLARED's rule: tau_scale x mean squared anchor-anchor
    distance, in the UNCORRECTED space, frozen thereafter."""
    import numpy as np

    anchors = np.asarray(anchor_features, dtype=np.float64)
    deltas = anchors[:, None, :] - anchors[None, :, :]
    squared = np.sum(deltas * deltas, axis=2)
    off_diagonal = squared[~np.eye(len(anchors), dtype=bool)]
    tau = float(tau_scale) * float(off_diagonal.mean())
    if tau <= 0:
        raise AnchorLoopError("tau must be positive; the anchors coincide")
    return tau


def corrected_distances(features, anchor_features, transform):
    """d(x, a) = ||Wx - Wa|| for every (patient, anchor) pair."""
    import numpy as np

    rows = np.asarray(features, dtype=np.float64) @ np.asarray(
        transform, dtype=np.float64
    ).T
    anchors = np.asarray(anchor_features, dtype=np.float64) @ np.asarray(
        transform, dtype=np.float64
    ).T
    deltas = rows[:, None, :] - anchors[None, :, :]
    return np.sqrt(np.sum(deltas * deltas, axis=2))


def expected_grade(features, anchor_features, anchor_grades, transform, *,
                   tau: float):
    """THE readout -- softmax over -d^2/tau across all 25 anchors,
    prediction = expectation of anchor grades. **One implementation**:
    the trained arm and the identity baseline both call this, which is
    what lets IDENTITY_BASELINE say 'within one readout'."""
    import numpy as np

    distances = corrected_distances(features, anchor_features, transform)
    logits = -(distances ** 2) / float(tau)
    logits = logits - logits.max(axis=1, keepdims=True)
    weights = np.exp(logits)
    weights = weights / weights.sum(axis=1, keepdims=True)
    return weights @ np.asarray(anchor_grades, dtype=np.float64)


def anchor_self_consistency(anchor_features, anchor_grades, transform) -> int:
    """Leave-one-out: does each anchor's nearest fellow anchor share its
    grade, under the corrected metric? Returns the count of 25."""
    import numpy as np

    distances = corrected_distances(anchor_features, anchor_features, transform)
    np.fill_diagonal(distances, np.inf)
    nearest = distances.argmin(axis=1)
    grades = np.asarray(anchor_grades)
    return int(np.sum(grades[nearest] == grades))


def mismatch_margin(distances, anchor_class3, patient_class3: int) -> float:
    """MISMATCH_RECORD_SPEC's margin: d(nearest) - min d over anchors of
    the patient's own class. Zero by construction on a match row."""
    import numpy as np

    distances = np.asarray(distances, dtype=float)
    anchor_class3 = np.asarray(anchor_class3, dtype=int)
    own = distances[anchor_class3 == int(patient_class3)]
    if not len(own):
        raise AnchorLoopError(
            f"no anchor maps to class {patient_class3}; the 3/7/6/6/3 "
            "spread covers every class, so this is a wiring defect"
        )
    return float(distances.min() - own.min())


#: The fixed grade spread the writer refuses to deviate from.
ANCHOR_GRADE_SPREAD = (3, 7, 6, 6, 3)


def save_anchor_set(directory, values, *, stems, grades, parity: float,
                    run_name: str) -> dict:
    """Write the 25-anchor embedding artifact -- once
    (ANCHOR_ARTIFACT_PERSISTED). Own namespace, CLUSTER-ONLY tier at
    the task layer; this writer enforces the shape and the spread."""
    import json

    import numpy as np

    from .provenance import atomic_write_text
    from .provenance.hashing import hash_dir

    directory = Path(directory)
    if directory.exists():
        raise AnchorLoopError(
            f"{directory} already exists. Data artifacts are immutable: "
            "create a new version, never overwrite"
        )
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] != 25:
        raise AnchorLoopError(
            f"values {values.shape}: the registration fixes 25 anchors"
        )
    if len(stems) != 25 or len(set(stems)) != 25:
        raise AnchorLoopError("25 distinct anchor stems required")
    grades = [int(g) for g in grades]
    spread = tuple(grades.count(g) for g in (1, 2, 3, 4, 5))
    if spread != ANCHOR_GRADE_SPREAD:
        raise AnchorLoopError(
            f"grade spread {spread} is not the registered 3/7/6/6/3 -- "
            "these are not the 25 Deall anchors"
        )
    directory.mkdir(parents=True)
    np.save(directory / "values.npy", values)
    atomic_write_text(
        directory / "metadata.json",
        json.dumps({
            "namespace": "anchor_deall",
            "n_anchors": 25,
            "feature_dim": int(values.shape[1]),
            "stems": list(stems),
            "grades": grades,
            "grade_spread": list(ANCHOR_GRADE_SPREAD),
            "live_path_parity_max_abs": float(parity),
            "banked_parity_reference": 1.53e-05,
            "generating_run": run_name,
            "anchors_belong_to_no_fold": (
                "the 25 are references, not cohort patients; no fold "
                "column exists here BY CONSTRUCTION (EXIT_CRITERIA "
                "criterion 1)"
            ),
        }, indent=2, sort_keys=True) + "\n",
    )
    return hash_dir(directory)


def load_anchor_set(directory):
    """Read what save_anchor_set wrote, re-checking the invariants."""
    import json

    import numpy as np

    directory = Path(directory)
    values = np.load(directory / "values.npy")
    metadata = json.loads(
        (directory / "metadata.json").read_text(encoding="utf-8")
    )
    if metadata.get("namespace") != "anchor_deall":
        raise AnchorLoopError(
            f"{directory} is not an anchor set: namespace "
            f"{metadata.get('namespace')!r}"
        )
    if values.shape[0] != 25 or len(metadata["grades"]) != 25:
        raise AnchorLoopError(f"anchor set corrupted: {values.shape}")
    spread = tuple(
        metadata["grades"].count(g) for g in (1, 2, 3, 4, 5)
    )
    if spread != ANCHOR_GRADE_SPREAD:
        raise AnchorLoopError(
            f"stored spread {spread} is not the registered 3/7/6/6/3"
        )
    return values, metadata


def train_w(train_features, train_targets, anchor_features, anchor_grades, *,
            tau: float, lambda_identity: float, learning_rate: float,
            batch_size: int, max_epochs: int, inner_val_frac: float,
            seed: int) -> tuple:
    """The registered correction: full linear W, identity init, MSE on
    the continuous target + lambda ||W - I||^2_F, AdamW with the
    optimiser's own weight_decay at 0 (SETTINGS_PROVENANCE), fixed
    epoch budget, best inner-val checkpoint, no patience
    (ANCHOR_LOOP_RULINGS). Returns (best W as numpy, curve rows)."""
    import numpy as np
    import torch

    rng = np.random.default_rng(seed)
    n = len(train_features)
    order = rng.permutation(n)
    n_val = max(1, int(round(n * inner_val_frac)))
    val_idx, fit_idx = order[:n_val], order[n_val:]
    if not len(fit_idx):
        raise AnchorLoopError("inner-val split consumed every training row")

    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_fit = torch.tensor(
        np.asarray(train_features, dtype=np.float32)[fit_idx], device=device
    )
    y_fit = torch.tensor(
        np.asarray(train_targets, dtype=np.float32)[fit_idx], device=device
    )
    anchors = torch.tensor(
        np.asarray(anchor_features, dtype=np.float32), device=device
    )
    grades = torch.tensor(
        np.asarray(anchor_grades, dtype=np.float32), device=device
    )
    dim = x_fit.shape[1]
    transform = torch.eye(dim, device=device, requires_grad=True)
    identity = torch.eye(dim, device=device)
    optimizer = torch.optim.AdamW(
        [transform], lr=learning_rate, weight_decay=0.0
    )

    def readout(block):
        projected = block @ transform.T
        projected_anchors = anchors @ transform.T
        squared = (
            (projected[:, None, :] - projected_anchors[None, :, :]) ** 2
        ).sum(dim=2)
        weights = torch.softmax(-squared / tau, dim=1)
        return weights @ grades

    def inner_val_stats():
        with torch.no_grad():
            x_val = torch.tensor(
                np.asarray(train_features, dtype=np.float32)[val_idx],
                device=device,
            )
            y_val = np.asarray(train_targets, dtype=np.float64)[val_idx]
            predicted = readout(x_val).cpu().numpy().astype(np.float64)
            mse = float(np.mean((predicted - y_val) ** 2))
            if len(y_val) > 1 and np.std(predicted) > 0 and np.std(y_val) > 0:
                pcc = float(np.corrcoef(y_val, predicted)[0, 1])
            else:
                pcc = 0.0
        return mse, pcc

    best = {"epoch": -1, "mse": float("inf"), "w": None}
    curve = []
    epoch_rng = np.random.default_rng(seed + 1)
    for epoch in range(max_epochs):
        batch_order = epoch_rng.permutation(len(fit_idx))
        epoch_loss = 0.0
        for start in range(0, len(batch_order), batch_size):
            batch = torch.tensor(
                batch_order[start:start + batch_size], device=device
            )
            optimizer.zero_grad()
            predicted = readout(x_fit[batch])
            mse = torch.mean((predicted - y_fit[batch]) ** 2)
            regulariser = lambda_identity * torch.sum(
                (transform - identity) ** 2
            )
            loss = mse + regulariser
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.detach()) * len(batch)
        val_mse, val_pcc = inner_val_stats()
        curve.append({
            "epoch": epoch,
            "train_loss": epoch_loss / len(fit_idx),
            "inner_val_mse": val_mse,
            "inner_val_pcc": val_pcc,
        })
        if val_mse < best["mse"]:
            best = {
                "epoch": epoch, "mse": val_mse,
                "w": transform.detach().cpu().numpy().copy(),
            }
    if best["w"] is None:
        raise AnchorLoopError("no epoch produced a finite inner-val MSE")
    return best["w"], best["epoch"], curve


#: **[ATTACHED 2026-08-29, DESCRIPTIVE] Self-consistency, measured --
#: and one limitation of the lock.**
SELF_CONSISTENCY_ATTACHED = {
    "attached": "2026-08-29, from run f342fed9's metrics.json",
    "the_numbers": (
        "banked uncorrected 4/25 euclidean / 3/25 cosine -> corrected "
        "mean **6.76/25 across the 25 fold x seed cells, 22 of 25 "
        "above the ~4.75/25 chance line** (per-cell table in the run's "
        "metrics.json). The repaired reading attaches DESCRIPTIVELY"
    ),
    "the_absolute_reading_beside_it": (
        "**stated plainly: 6.76/25 is 27%** -- roughly three anchors "
        "in four still sit nearest a different-grade anchor. Repaired "
        "RELATIVE TO ITS BASELINE, weak in absolute terms. Both "
        "sentences travel together"
    ),
    "limitation_of_the_lock": (
        "**the readings record's phrase 'claimably above' had no "
        "registered claimability machinery for this measurement** -- "
        "the criterion's machinery is built for paired per-seed "
        "cohort statistics, not for a count over 25 references -- and "
        "inventing one now would be post-hoc. Recorded in the lock "
        "clause's own language: a criterion discovered missing later "
        "is a limitation of the lock, recorded as such, never a "
        "retro-fitted entry (EXIT_CRITERIA['locked']). The attachment "
        "is therefore DESCRIPTIVE, and 'claimably' goes unexercised"
    ),
}


#: **[RECORDED 2026-08-29: its own record] REPAIR
#: WITHOUT TRANSFER -- the phase's most interesting sentence.**
REPAIR_WITHOUT_TRANSFER = {
    "recorded": "2026-08-29, named so the write-up can find it",
    "the_finding": (
        "**W was trained on MSE against the continuous panel mean; "
        "anchor self-consistency was never in the objective. The "
        "self-consistency improvement is therefore EMERGENT, not the "
        "objective -- and it runs OPPOSITE to the cohort result**: the "
        "correction clusters the anchors better (4/25 -> 6.76/25 mean) "
        "while being mildly harmful to cohort PCC (identity 0.2151 -> "
        "loop 0.2040, 2 of 5 seeds above identity). **The loop learned "
        "something real about the anchor space that does not transfer "
        "to predicting cohort grades. That pairing is the finding**"
    ),
    "why_the_pairing_matters": (
        "either half alone is unremarkable -- a small PCC drop is "
        "noise-shaped, a self-consistency bump on 25 references is a "
        "curiosity. TOGETHER they dissociate two things the design "
        "treated as one: metric structure AROUND THE ANCHORS and "
        "grade signal ACROSS THE COHORT. The correction found the "
        "first without the second, so the anchors' neighbourhood "
        "geometry and the cohort's grade geometry are not the same "
        "thing in this space"
    ),
    "bounds": (
        "descriptive throughout: the cohort-side contrast is the "
        "ledger's UNRESOLVED row (p16-anchor-loop-unresolved), the "
        "self-consistency side has no claimability machinery "
        "(SELF_CONSISTENCY_ATTACHED['limitation_of_the_lock']), and "
        "one backbone bounds everything"
    ),
}


#: **[RECORDED 2026-08-29, WITH ITS OWN LIMIT] The epoch-0 cross-check.**
EPOCH_ZERO_CROSS_CHECK = {
    "recorded": "2026-08-29, the ruling: record with its limit",
    "the_observation": (
        "within seed 1337, the three folds whose selected checkpoint "
        "was EPOCH 0 give self-consistency 3/3/4 -- the uncorrected "
        "level -- and the two trained folds (epochs 15, 12) give 8/8. "
        "Coherent: where W stayed at identity the measurement "
        "reproduces the baseline, where W trained it moves. **Good "
        "evidence the measurement behaves**"
    ),
    "the_limit_beside_it": (
        "**those folds differ in TRAINING DATA as well as training "
        "amount, so this is suggestive, not a paired comparison** -- "
        "an epoch-0 fold saw different patients than a trained fold, "
        "and nothing here separates the two differences. Stated in "
        "the record so the 3/3/4-vs-8/8 pattern is quoted with its "
        "confound attached"
    ),
}


#: **[CLOSED 2026-08-29] PHASE 16. THE ANCHOR LOOP.**
PHASE_16_CLOSING = {
    "closed": (
        "2026-08-29 -- run p16_anchor_loop__f342fed9__p16-anchor-loop, "
        "finalized, single attempt; all seven locked criteria walked"
    ),
    "criterion_1_fold_honesty": (
        "MET IN CODE: training_indices cannot express the full-cohort "
        "path (tested), the task's single train_w call site receives "
        "only its output (AST-tested), anchors carry no fold key by "
        "writer construction (tested)"
    ),
    "criterion_2_registered_metrics": (
        "MET: 5 folds x 5 seeds OOF; PCC, Spearman, and 3-class "
        "accuracy at 2.5/3.5 with chance 0.333 and majority 0.502 "
        "beside, in the run's metrics.json -- nothing beyond the "
        "registered set"
    ),
    "criterion_3_self_consistency": (
        "MET: the per-cell table (25 fold x seed rows) in metrics.json "
        "against banked 4/25 / 3/25 / ~4.75/25 "
        "(SELF_CONSISTENCY_ATTACHED, with the limitation of the lock "
        "recorded)"
    ),
    "criterion_4_primary_contrast": (
        "MET: outcome UNRESOLVED, ledgered as "
        "p16-anchor-loop-unresolved with both pre-committed readings "
        "attached and the condition-2-passes/condition-1-fails flag -- "
        "the ledger's first such case"
    ),
    # [CORRECTED 2026-08-31] "the ledger's first such case" above is
    # WRONG AS WRITTEN and is preserved as written. Derived from the
    # ledger's own condition fields, EIGHT rows have condition 1 FALSE
    # and condition 2 TRUE; four of them (indices 25, 26, 27, 28)
    # precede this one, which is the FIFTH. What IS true, and was
    # measured: this is the first row where ALL per-seed intervals span
    # zero AND directions are mixed. the ruling was the descriptive
    # reading of the row's em-dash clause, so the bolded sentence is the
    # claim and it does not carry that qualifier. No figure changes.
    # Ledger entry ledger-condition-split-count-corrected (index 37).
    "first_such_case_corrected_2026_08_31": (
        "results_ledger.ENTRIES[37] "
        "'ledger-condition-split-count-corrected' -- this is the FIFTH "
        "condition-2-pass/condition-1-fail row, not the first; it IS "
        "the first with all intervals spanning zero and directions "
        "mixed. No measurement changes"
    ),
    "criterion_5_mismatch_records": (
        "MET: five seed_<n>__mismatch_records.csv files at 237 rows "
        "each, confirmed in the run directory -- produced regardless "
        "of outcome, as locked"
    ),
    "criterion_6_identity_baseline": (
        "MET: identity 0.2151 reported descriptively beside the loop's "
        "0.2040, same readout, same folds and seeds "
        "(IDENTITY_BASELINE; and the Phase 9 refinement it licenses is "
        "on the Phase 9 record, "
        "phase9.PROTOTYPE_CLASSIFIER_OBSERVED['readout_refined_2026_"
        "08_29'])"
    ),
    "the_pointer_in_criterion_6_CORRECTED_2026_09_06": (
        "**the entry above cited PASS_ZERO_READOUT_REFINED, a name "
        "that exists nowhere in phase9 or anywhere else in this "
        "repository.** The refinement it means is "
        "``phase9.PROTOTYPE_CLASSIFIER_OBSERVED['readout_refined_"
        "2026_08_29']``, and the pointer is corrected to it. **No "
        "figure changes and no reading changes**: the identity "
        "0.2151, the loop 0.2040 and the descriptive disposition all "
        "stand. What was broken was the cross-reference, and it "
        "survived because its pin accepted the string 'Phase 9' as "
        "an alternative to the name, so the name was never checked. "
        "**The pin is tightened in the same pass**"
    ),
    "criterion_7_locked": (
        "the lock held: nothing was added after 2026-08-29, and the "
        "one gap it produced is recorded AS a limitation "
        "(SELF_CONSISTENCY_ATTACHED['limitation_of_the_lock'])"
    ),
    "tau_degenerate_mode_not_observed": (
        "**the named alternative is discharged**: the registered "
        "degenerate signature (constant ~2.96, near-zero PCC, acc3 at "
        "the 0.502 floor) did NOT appear -- the loop's PCC is 0.2040 "
        "and predictions vary; tau_scale 1.0 stands untouched, never "
        "tuned"
    ),
    "the_findings": (
        "REPAIR_WITHOUT_TRANSFER (the phase's sentence); the fifth "
        "convergent null, ledgered UNRESOLVED with the negative point "
        "estimate stated; parity-with-explanations met on its own "
        "terms -- the mismatch records exist for all 237; and the "
        "Phase 9 readout refinement, applied as a dated note on the "
        "Phase 9 record"
    ),
    "caveats_inherited": (
        "from the registration, unchanged: anchor grades are trusted "
        "single grades from the survey lineage, not verified-unanimous "
        "-- still supervision question 5's neighbour; one backbone, frozen "
        "space; the ImageNet anchor rides in every reading"
    ),
    # [2026-08-31] The two arm means this closing states IN PROSE now
    # have a structured home: ARM_MEANS. The prose above is unedited --
    # it is what the phase closed on -- and the constant carries the
    # same figures in a form a checker can read.
    "arm_means_structured_2026_08_31": "ARM_MEANS",
}


#: **[BANKED 2026-08-31] THE PHASE 16 ARM MEANS, STRUCTURED.**
#:
#: These two figures closed Phase 16 and have lived only inside prose
#: ever since -- ``PHASE_16_CLOSING["criterion_6_identity_baseline"]``
#: ("identity 0.2151 reported descriptively beside the loop's 0.2040")
#: and ``REPAIR_WITHOUT_TRANSFER["the_finding"]`` ("identity 0.2151 ->
#: loop 0.2040"). The standing record-artifact check
#: (``record_audit``) could not see them, so two of the 68 locked arms
#: were compared by nothing. **The prose is unedited; this is a second
#: expression of the same numbers, not a correction of them.**
#:
#: **Provenance and how each figure was checked:**
#:
#: * **PCC** -- taken from the closing above, where it appears TWICE in
#:   two independently-written records that agree (``criterion_6`` and
#:   ``REPAIR_WITHOUT_TRANSFER``). Not regex-parsed out of either.
#: * **The loop's sd 0.0345** -- from the ledger row
#:   ``p16-anchor-loop-unresolved``, and **verified by arithmetic**:
#:   ``phase3.combined_claimable_delta(0.0345, 5, 0.0148, 5)`` returns
#:   ``arm_means_95 = 0.0329``, which is the threshold that row records.
#:   A wrong sd would not reproduce the banked threshold.
#: * **The identity baseline's sd** -- the readout is deterministic
#:   (``IDENTITY_BASELINE``: W = I, untrained), so its five seeds agree
#:   to floating point. Phase 9 records "~3e-17", a float zero rather
#:   than a measured spread, and it is banked as that with the
#:   deterministic flag rather than as a number to compare.
#:
#: The point estimate runs AGAINST the loop (0.2040 < 0.2151) and that
#: is the phase's finding, not an embarrassment to be smoothed:
#: ``REPAIR_WITHOUT_TRANSFER`` is built on exactly this pairing.
ARM_MEANS = {
    "banked": "2026-08-31",
    "provenance": (
        "PCC from PHASE_16_CLOSING['criterion_6_identity_baseline'] and "
        "REPAIR_WITHOUT_TRANSFER['the_finding'], which agree; the "
        "loop's sd from the ledger row p16-anchor-loop-unresolved. "
        "Transcribed from those records, NOT regex-parsed, and the sd "
        "is verified by arithmetic (see sd_verified_by)"
    ),
    "arms": {
        "p16_anchor_loop": {"pcc": 0.2040, "sd": 0.0345, "n_seeds": 5},
        "p16_identity_baseline": {
            "pcc": 0.2151, "sd": None, "n_seeds": 5,
            "deterministic": (
                "W = I, untrained: the five seeds agree to floating "
                "point. Phase 9 records ~3e-17, a float zero rather "
                "than a measured spread -- banked as None with this "
                "flag rather than as a number to compare"
            ),
        },
    },
    "sd_verified_by": (
        "phase3.combined_claimable_delta(0.0345, 5, 0.0148, 5) returns "
        "arm_means_95 = 0.0329, the threshold the ledger row records. A "
        "wrong sd would not reproduce a banked threshold"
    ),
    "the_prose_is_unedited": (
        "PHASE_16_CLOSING and REPAIR_WITHOUT_TRANSFER are untouched -- "
        "this is a second expression of the same numbers, not a "
        "correction, and the closing carries a dated pointer here"
    ),
    "direction_is_the_finding": (
        "0.2040 < 0.2151: the point estimate runs AGAINST the loop, "
        "which is what REPAIR_WITHOUT_TRANSFER is built on"
    ),
}



def summary() -> dict:
    """The phase's records, importable as one object."""
    return {
        "rulings": ANCHOR_LOOP_RULINGS,
        "self_consistency": SELF_CONSISTENCY_READINGS,
        "primary_contrast": PRIMARY_CONTRAST_REGISTERED,
        "compute_shape": COMPUTE_SHAPE_VERIFIED,
        "exit_criteria": EXIT_CRITERIA,
        "negative_space_retired": NEGATIVE_SPACE_RETIRED,
        "anchor_artifact": ANCHOR_ARTIFACT_PERSISTED,
        "tau": TAU_DECLARED,
        "settings_provenance": SETTINGS_PROVENANCE,
        "identity_baseline": IDENTITY_BASELINE,
        "mismatch_spec": MISMATCH_RECORD_SPEC,
        "self_consistency_attached": SELF_CONSISTENCY_ATTACHED,
        "repair_without_transfer": REPAIR_WITHOUT_TRANSFER,
        "epoch_zero_cross_check": EPOCH_ZERO_CROSS_CHECK,
        "closing": PHASE_16_CLOSING,
    }
