"""Phase 16 -- the anchor loop's rulings, readings, and compute shape.

Tests were written BEFORE the code they guard (the build ruling's
item 5); the negative-space test was retired deliberately on the
exit-criteria lock and its successor asserts the positive space
(phase16.NEGATIVE_SPACE_RETIRED).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from cleft import phase9, phase15, phase16

REPO = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------
# the rulings
# --------------------------------------------------------------------------


def test_the_target_ruling_is_recorded_with_its_rationale():
    record = phase16.ANCHOR_LOOP_RULINGS
    assert "2026-08-29" in record["ruled"]
    # The registration stands unchanged; the rulings bind, not amend.
    assert "standing unchanged" in record["registration"]
    assert "amend nothing" in record["registration"]

    target = record["target_is_continuous"]
    assert "CONTINUOUS panel mean" in target
    assert "0.2520 probe" in target
    assert "overriding the lean" in target

    rationale = record["the_rationale"]
    assert "10/6/9" in rationale
    assert "SUB-GRADE structure" in rationale
    assert "grade-3 for different reasons" in rationale
    assert "LIKE-FOR-LIKE" in rationale
    # The 10/6/9 arithmetic is the 3/7/6/6/3 spread collapsed at
    # class3: (3+7, 6, 6+3).
    spread = (3, 7, 6, 6, 3)
    assert (spread[0] + spread[1], spread[2], spread[3] + spread[4]) == (
        10, 6, 9,
    )
    assert "3/7/6/6/3" in str(phase15.ANCHOR_LOOP_REGISTERED["anchors"])


def test_all_three_variants_are_named_unregistered():
    """The Phase-14 pattern: named now so none can be presented as a
    fresh idea after the registered arm's number is known."""
    record = phase16.ANCHOR_LOOP_RULINGS
    assert "UNREGISTERED, by name, with reasons" in record[
        "unregistered_not_deferred"
    ]
    assert "presented later as a fresh idea" in record[
        "unregistered_not_deferred"
    ]

    assert "3-CLASS PULL-LOSS VARIANT" in record["unregistered_a_3class_pull_loss"]
    assert "10/6/9" in record["unregistered_a_3class_pull_loss"]
    assert "two-variable comparison" in record["unregistered_a_3class_pull_loss"]

    assert "DIAGONAL" in record["unregistered_b_diagonal_w"]
    assert "cannot rotate" in record["unregistered_b_diagonal_w"]
    assert "not axis-aligned" in record["unregistered_b_diagonal_w"]

    assert "LOW-RANK" in record["unregistered_c_lowrank_w"]
    assert "capacity knob" in record["unregistered_c_lowrank_w"]
    assert "threshold-moved-after-the-data" in record["unregistered_c_lowrank_w"]

    # All three carry the word, so a later grep for "unregistered"
    # finds each by name.
    for key in ("unregistered_a_3class_pull_loss", "unregistered_b_diagonal_w",
                "unregistered_c_lowrank_w"):
        assert "Unregistered because" in record[key], key


def test_the_registered_correction_and_its_scientific_setting():
    record = phase16.ANCHOR_LOOP_RULINGS
    correction = record["registered_correction"]
    assert "768x768" in correction
    assert "IDENTITY" in correction
    assert "TOWARD IDENTITY" in correction
    # The reason identity-decay rather than zero-decay: the null
    # hypothesis is the resting state.
    assert "null hypothesis is the resting state" in correction

    decay = record["weight_decay_is_a_scientific_setting"]
    assert "declared in the YAML before the first" in decay
    assert "NEVER tuned across runs" in decay
    assert "dated amendment with a reason" in decay
    # The enforcement split is honest: drift-detectable half named to
    # the machinery, the undetectable half named to the record.
    assert "--check turns any silent edit into DRIFT" in decay
    assert "required-no-default" in decay
    assert "cannot enforce" in decay

    budget = record["epoch_budget"]
    assert "FIXED epoch budget" in budget
    assert "NO patience" in budget
    assert "same compute in every fold" in budget

    forbidden = record["the_forbidden_variant_restated"]
    assert "MEMORISATION-AND-FORBIDDEN" in forbidden
    assert "the_forbidden_version_named" in forbidden
    assert "MEMORISATION" in phase15.ANCHOR_LOOP_REGISTERED[
        "the_forbidden_version_named"
    ]


# --------------------------------------------------------------------------
# the pre-run readings
# --------------------------------------------------------------------------


def test_both_self_consistency_readings_are_registered_with_the_baseline():
    record = phase16.SELF_CONSISTENCY_READINGS
    assert "before any corrected metric exists" in record["registered"]

    measurement = record["the_measurement"]
    assert "PER FOLD x SEED" in measurement
    assert "4/25 euclidean" in measurement
    assert "3/25 cosine" in measurement
    assert "4.75/25" in measurement
    assert "3/7/6/6/3" in measurement

    # The baseline numbers are the banked ones, not retyped ones.
    banked = phase9.PROTOTYPE_CLASSIFIER_OBSERVED["anchor_self_consistency"]
    assert banked["euclidean"] == "4/25"
    assert banked["cosine"] == "3/25"
    assert "4.75/25" in banked["chance_expectation"]

    repaired = record["reading_if_repaired"]
    assert "CLEANEST POSITIVE" in repaired
    assert "whatever the cohort-side number does" in repaired

    unrepaired = record["reading_if_not_repaired"]
    assert "UNDER CORRECTION TOO" in unrepaired
    assert "harder to attribute to the anchors" in unrepaired

    assert "neither can be written to fit the number" in record["why_pre_run"]


def test_the_primary_contrast_is_registered_and_has_no_runner():
    record = phase16.PRIMARY_CONTRAST_REGISTERED
    assert "no runner" in record["registered"]

    contrast = record["the_contrast"]
    assert "p7_d1_vit_b16_imagenet_g1" in contrast
    assert "10,000" in contrast
    assert "five seeds" in contrast
    assert "BOTH criterion conditions" in contrast
    assert "nothing invented for this phase" in contrast

    beside = record["pass_zero_beside_it"]
    assert "DESCRIPTIVELY" in beside
    assert "0.1823" in beside
    # And 0.1823 really is pass zero's best cell.
    cells = phase9.PROTOTYPE_CLASSIFIER_OBSERVED["cells"]
    best = max(
        cell["pcc"]
        for metric in ("euclidean", "cosine")
        for cell in cells[metric].values()
    )
    assert best == 0.1823

    prediction = record["the_prediction"]
    assert "COHORT_CANNOT_RESOLVE predicts UNRESOLVED" in prediction
    assert "29 of" in prediction
    assert "PARITY-WITH-EXPLANATIONS" in prediction
    assert "parity-with-explanations, not victory" in phase15.ANCHOR_LOOP_REGISTERED[
        "success_is_defined_before_the_run"
    ]


# --------------------------------------------------------------------------
# the compute shape, verified against the repo rather than the record
# --------------------------------------------------------------------------


def test_the_cohort_artifact_is_shared_by_probe_and_pass_zero():
    """The identical-path-identical-hash fact the record rests on,
    re-read from the shipped configs."""
    record = phase16.COMPUTE_SHAPE_VERIFIED

    def embeddings_entry(name):
        payload = yaml.safe_load(
            (REPO / "configs" / name).read_text(encoding="utf-8")
        )
        return next(e for e in payload["inputs"] if e["name"] == "embeddings")

    probe = embeddings_entry("p7_d1_vit_b16_imagenet_g1.yaml")
    pass_zero = embeddings_entry("p9_prototype_classifier.yaml")
    assert probe == pass_zero
    assert probe["path"].endswith(
        "data/embeddings/embeddings_g1_ladder_v1/vit_b16__imagenet__g1"
    )
    assert probe["rollup_sha256"] in record["cohort_embeddings"]
    assert "IDENTICAL path and hash" in record["cohort_embeddings"]

    # The deall_set the anchors would be re-extracted from, likewise.
    payload = yaml.safe_load(
        (REPO / "configs" / "p9_prototype_classifier.yaml").read_text(
            encoding="utf-8"
        )
    )
    deall = next(e for e in payload["inputs"] if e["name"] == "deall_set")
    assert deall["rollup_sha256"] in record["anchor_embeddings"]


def test_no_anchor_feature_artifact_exists_and_the_source_proves_it():
    """The record's central negative -- verified in source here exactly
    as it was verified when written."""
    import inspect

    from cleft import run as run_module

    source = inspect.getsource(run_module.task_prototype_classifier)
    assert "FrozenExtractor" in source
    assert "np.save" not in source
    assert "savez" not in source
    record = phase16.COMPUTE_SHAPE_VERIFIED["anchor_embeddings"]
    assert "DO NOT EXIST" in record
    assert "zero np.save/savez calls" in record

    conclusion = phase16.COMPUTE_SHAPE_VERIFIED["conclusion"]
    assert "(b), for the anchors only" in conclusion
    assert "RE-EXTRACTION FROM PIXELS" in conclusion
    assert "pass zero's exact compute shape" in conclusion
    assert "1.53e-05" in conclusion
    assert phase9.PROTOTYPE_CLASSIFIER_OBSERVED[
        "live_path_parity_max_abs"
    ] == 1.53e-05

    # The build option is flagged, not decided.
    flagged = phase16.COMPUTE_SHAPE_VERIFIED["a_build_option_flagged_not_decided"]
    assert "not made by default" in flagged


def test_fold_honesty_of_reuse_is_stated():
    record = phase16.COMPUTE_SHAPE_VERIFIED["fold_honesty_of_reuse"]
    assert "no entanglement" in record
    assert "before any fold or seed exists" in record
    assert "deterministic" in record
    assert "learned W itself" in record
    assert "not a compromise" in record


# --------------------------------------------------------------------------
# [RETIRED 2026-08-29] the negative space -- converted, not deleted.
# --------------------------------------------------------------------------


def test_the_negative_space_test_was_retired_deliberately():
    """What test_nothing_is_built_and_the_pointers_hold guarded, and why
    it ended: it held the build shut between the rulings and the
    exit-criteria lock, so nothing could be built on unreviewed compute
    assumptions. The lock (EXIT_CRITERIA) is the dated event that ends
    it; its successor below asserts the POSITIVE space instead --
    exactly the objects whose absence the old test asserted, now
    present and shaped as the rulings require."""
    record = phase16.NEGATIVE_SPACE_RETIRED
    assert "2026-08-29" in record["retired"]
    assert "test_nothing_is_built_and_the_pointers_hold" in record["what_it_was"]
    assert "compute verification is reviewed" in record["what_it_guarded"]
    assert "EXIT_CRITERIA" in record["what_ended_it"]
    assert "positive-space successor" in record["the_successor"]

    # The pointers it also guarded still hold.
    assert phase15.ANCHOR_LOOP_REGISTERED["restate_rulings"] == (
        "phase16.ANCHOR_LOOP_RULINGS"
    )
    assert "phase15.ANCHOR_LOOP_REGISTERED" in phase16.ANCHOR_LOOP_RULINGS[
        "registration"
    ]


def test_the_positive_space_successor():
    """The objects the retired test forbade, now present by name."""
    from cleft.config.schema import TASK_SPECS
    from cleft.run import TASKS

    for registry in (TASK_SPECS, TASKS):
        assert "anchor_loop" in registry
        assert "extract_anchor_embeddings" in registry
    assert (REPO / "configs" / "p16_anchor_loop.yaml").is_file()
    assert (REPO / "configs" / "p16_extract_anchor_embeddings.yaml").is_file()


# --------------------------------------------------------------------------
# the exit criteria, locked
# --------------------------------------------------------------------------


def test_the_exit_criteria_are_locked_and_complete():
    record = phase16.EXIT_CRITERIA
    criteria = record["criteria"]
    assert len(criteria) == 7

    assert "trained on training folds only" in criteria[0]
    assert "anchors belong to no fold" in criteria[0]
    assert "asserted by tests, not prose" in criteria[0]

    assert "5 folds x 5 seeds" in criteria[1]
    assert "PCC and Spearman" in criteria[1]
    assert "2.5/3.5" in criteria[1]
    assert "0.333" in criteria[1] and "0.502" in criteria[1]

    assert "4/25" in criteria[2] and "3/25" in criteria[2]
    assert "4.75/25" in criteria[2]

    assert "p7_d1_vit_b16_imagenet_g1" in criteria[3]
    assert "0.2520" in criteria[3]
    assert "claimable/withdrawn/unresolved" in criteria[3]
    assert "both pre-committed readings" in criteria[3]

    assert "all 237" in criteria[4]
    assert "regardless of outcome" in criteria[4]

    assert "Identity baseline" in criteria[5]
    assert "descriptively" in criteria[5]

    assert "LOCKED" in criteria[6]
    assert "2026-08-29" in criteria[6]

    # The lock clause: nothing added after this record.
    assert "nothing is added after" in record["locked"]
    assert "2026-08-29" in record["locked"]


# --------------------------------------------------------------------------
# fold-honesty, in code
# --------------------------------------------------------------------------


def test_fold_honesty_is_structural():
    """The training split helper refuses anything but a proper subset of
    folds, and anchors have no fold representation at all."""
    import numpy as np

    folds = np.array([0, 1, 2, 3, 4] * 4)
    # Training indices for held-out fold 0: exactly the non-0 rows.
    train = phase16.training_indices(folds, held_out=0)
    assert set(folds[train]) == {1, 2, 3, 4}
    assert len(train) == 16
    # The full-cohort path cannot be expressed: every fold value is a
    # held-out fold, so no call leaves all rows in training.
    for held_out in range(5):
        train = phase16.training_indices(folds, held_out=held_out)
        assert len(train) < len(folds)
    # A held-out fold that does not exist is refused, not ignored.
    with pytest.raises(phase16.AnchorLoopError, match="not a fold"):
        phase16.training_indices(folds, held_out=9)


def test_the_config_surface_cannot_express_the_forbidden_variants():
    """The unregistered-variants guard over the NEW surface: the schema
    carries no key through which the 3-class pull loss, a diagonal or
    low-rank W, a no-folds run, or patience could arrive."""
    from cleft.config.schema import TASK_SPECS

    spec = TASK_SPECS["anchor_loop"]
    keys = set(spec)
    assert keys == {
        "kind", "arm", "probe_run", "anchor_embeddings",
        "embeddings_artifact", "manifest_artifact", "seeds", "max_epochs",
        "learning_rate", "batch_size", "inner_val_frac", "monitor",
        "lambda_identity", "tau_scale", "n_boot",
    }
    # By name: nothing that could switch target type, W shape, folds,
    # or early stopping.
    for forbidden in ("target", "label", "w_shape", "rank", "diagonal",
                      "patience", "folds", "no_folds", "use_folds",
                      "full_cohort", "n_classes"):
        assert forbidden not in keys, forbidden
    # Every scientific setting is required with no default.
    for key in ("lambda_identity", "tau_scale", "max_epochs",
                "learning_rate", "batch_size", "inner_val_frac", "seeds"):
        field = spec[key]
        assert field.required and field.default is None, key
    # n_boot is pinned to the registered 10,000.
    assert spec["n_boot"].choices == (10000,)
    assert spec["monitor"].choices == ("inner_val_mse",)


def test_patience_is_refused_by_the_loader():
    import copy

    from cleft.config.schema import TASK_SPECS, _validate_mapping
    from cleft.config import ConfigError

    task = yaml.safe_load(
        (REPO / "configs" / "p16_anchor_loop.yaml").read_text(
            encoding="utf-8"
        )
    )["task"]
    _validate_mapping(task, TASK_SPECS["anchor_loop"], "task")  # loads clean
    with_patience = copy.deepcopy(task)
    with_patience["patience"] = 5
    with pytest.raises(ConfigError, match="patience"):
        _validate_mapping(with_patience, TASK_SPECS["anchor_loop"], "task")


# --------------------------------------------------------------------------
# the settings, their provenance, and tau
# --------------------------------------------------------------------------


def test_every_mirrored_setting_matches_its_named_precedent():
    provenance = phase16.SETTINGS_PROVENANCE
    probe = yaml.safe_load(
        (REPO / "configs" / "p7_d1_vit_b16_imagenet_g1.yaml").read_text(
            encoding="utf-8"
        )
    )["task"]
    loop = yaml.safe_load(
        (REPO / "configs" / "p16_anchor_loop.yaml").read_text(
            encoding="utf-8"
        )
    )["task"]

    assert loop["max_epochs"] == probe["max_epochs"]
    assert loop["learning_rate"] == probe["learning_rate"]
    assert loop["batch_size"] == probe["batch_size"]
    assert loop["inner_val_frac"] == probe["inner_val_frac"]
    assert loop["seeds"] == probe["seeds"]
    assert loop["lambda_identity"] == probe["weight_decay"]
    assert "patience" not in loop

    for key in ("max_epochs", "learning_rate", "batch_size",
                "inner_val_frac", "seeds", "lambda_identity", "optimiser"):
        assert "p7_d1_vit_b16_imagenet_g1" in provenance[key], key
    # lambda's provenance says WHY AdamW's own decay must be zero.
    assert "toward ZERO" in provenance["lambda_identity"]
    assert "weight_decay=0" in provenance["lambda_identity"]
    # tau explicitly has NO precedent.
    assert "no precedent" in provenance["tau_scale"].lower()


def test_tau_is_declared_with_its_reasoning_and_degenerate_mode():
    record = phase16.TAU_DECLARED
    assert "cannot be mirrored" in record["why_declared_not_mirrored"]
    assert "the ruling" in record["why_declared_not_mirrored"]
    # The rule: dimensionless argument, anchor-cloud scale, frozen in
    # the UNCORRECTED space.
    rule = record["the_rule"]
    assert "mean squared anchor-anchor distance" in rule
    assert "UNCORRECTED" in rule
    assert "frozen" in rule.lower()
    degenerate = record["the_degenerate_mode_registered"]
    assert "flattens toward a constant predictor" in degenerate
    assert "2.96" in degenerate
    assert "near-zero PCC" in degenerate
    assert "0.502" in degenerate
    assert "attributable, not mysterious" in degenerate
    # The flat-softmax constant really is the anchor-grade mean.
    spread = {1: 3, 2: 7, 3: 6, 4: 6, 5: 3}
    mean_grade = sum(g * n for g, n in spread.items()) / 25
    assert round(mean_grade, 2) == 2.96
    # Never tuned: the scientific-settings clause covers it by name.
    assert "tau_scale" in phase16.ANCHOR_LOOP_RULINGS[
        "weight_decay_is_a_scientific_setting"
    ] or "NEVER tuned" in record["never_tuned"]


# --------------------------------------------------------------------------
# the readout, the baseline, the mismatch records
# --------------------------------------------------------------------------


def test_the_readout_is_shared_and_the_identity_baseline_is_descriptive():
    import numpy as np

    anchors = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]])
    grades = np.array([1.0, 3.0, 5.0])
    x = np.array([[0.1, 0.0]])
    # Near-anchor-0 point, sharp tau: prediction ~ grade 1.
    sharp = phase16.expected_grade(x, anchors, grades, np.eye(2), tau=0.5)
    assert abs(sharp[0] - 1.0) < 1e-6
    # Huge tau: the registered degenerate mode -- the anchor-grade mean.
    flat = phase16.expected_grade(x, anchors, grades, np.eye(2), tau=1e12)
    assert abs(flat[0] - 3.0) < 1e-6

    # The identity baseline record: same readout, W = I, distinct from
    # pass zero, descriptive only.
    record = phase16.IDENTITY_BASELINE
    assert "W = I" in record["what_it_is"]
    assert "IDENTICAL softmax-expectation readout" in record["what_it_is"]
    assert "k-NN voting" in record["distinct_from_pass_zero"]
    assert "different readout" in record["distinct_from_pass_zero"]
    assert "WITHIN ONE READOUT" in record["what_it_buys"]
    assert "DESCRIPTIVE" in record["descriptive_only"]
    assert "no ledger row" in record["descriptive_only"].lower()


def test_margin_semantics_are_recorded_and_hold_in_code():
    import numpy as np

    record = phase16.MISMATCH_RECORD_SPEC
    semantics = record["margin_semantics"]
    assert "0 by construction" in semantics
    assert "informative on mismatches only" in semantics
    assert "column of zeros cannot read as a finding" in semantics
    assert "descriptive readout only" in record["class3_is_descriptive_only"]
    assert "THREE_NOT_FIVE" in record["class3_is_descriptive_only"]
    assert "remains continuous" in record["class3_is_descriptive_only"]

    # And the code agrees: nearest own-class -> margin exactly 0.
    distances = np.array([1.0, 2.0, 3.0])
    anchor_class3 = np.array([1, 1, 2])
    margin = phase16.mismatch_margin(distances, anchor_class3, patient_class3=1)
    assert margin == 0.0
    # Nearest is other-class -> margin NEGATIVE by the definition
    # (d(nearest) < min own-class d), quantifying how much nearer the
    # wrong-class anchor sits. First written expecting +2.0 -- reading
    # the definition, not the sign convention -- and corrected against
    # the ruling's own formula before any implementation moved.
    margin = phase16.mismatch_margin(distances, anchor_class3, patient_class3=2)
    assert margin == 1.0 - 3.0


def test_the_anchor_set_writer_refuses_a_wrong_spread():
    import tempfile

    import numpy as np

    ids = [f"a{i}" for i in range(25)]
    grades = [1] * 3 + [2] * 7 + [3] * 6 + [4] * 6 + [5] * 3
    d = Path(tempfile.mkdtemp()) / "anchor_set"
    payload = phase16.save_anchor_set(
        d, np.zeros((25, 768), dtype="float32"), stems=ids, grades=grades,
        parity=1.5e-5, run_name="r",
    )
    assert "rollup" in payload
    values, metadata = phase16.load_anchor_set(d)
    assert values.shape == (25, 768)
    assert metadata["grade_spread"] == [3, 7, 6, 6, 3]

    # A 24-anchor set, or a drifted spread, is refused.
    with pytest.raises(phase16.AnchorLoopError, match="25"):
        phase16.save_anchor_set(
            Path(tempfile.mkdtemp()) / "x", np.zeros((24, 768)),
            stems=ids[:24], grades=grades[:24], parity=0.0, run_name="r",
        )
    bad = [1] * 4 + [2] * 6 + [3] * 6 + [4] * 6 + [5] * 3
    with pytest.raises(phase16.AnchorLoopError, match="3/7/6/6/3"):
        phase16.save_anchor_set(
            Path(tempfile.mkdtemp()) / "y", np.zeros((25, 768)),
            stems=ids, grades=bad, parity=0.0, run_name="r",
        )


def test_the_two_configs_carry_their_declared_inputs():
    extract = yaml.safe_load(
        (REPO / "configs" / "p16_extract_anchor_embeddings.yaml").read_text(
            encoding="utf-8"
        )
    )
    names = [e["name"] for e in extract["inputs"]]
    assert names == ["deall_set", "manifest_v1", "staged_v1", "embeddings"]
    # Every extraction input is already resolved -- carried from the
    # pass-zero config, so the run needs NO new declaration.
    for entry in extract["inputs"]:
        assert set(entry["rollup_sha256"]) != {"0"}, entry["name"]
    deall = next(e for e in extract["inputs"] if e["name"] == "deall_set")
    assert deall["rollup_sha256"].startswith("7312c112")

    loop = yaml.safe_load(
        (REPO / "configs" / "p16_anchor_loop.yaml").read_text(
            encoding="utf-8"
        )
    )
    names = [e["name"] for e in loop["inputs"]]
    assert names == [
        "anchor_embeddings", "probe_run", "manifest_v1", "embeddings",
    ]
    anchor = next(e for e in loop["inputs"] if e["name"] == "anchor_embeddings")
    # **UPDATED 2026-08-29, at the declare fill** -- this assertion read
    # "still a placeholder" and FAILED when the hash landed, which is
    # what pinning declaration state is for. The declared value is the
    # measured one from declare_inputs.py (2 files, 77,903 bytes).
    assert anchor["rollup_sha256"] == (
        "49905fe94c2065e56bcc22887377bb77b3b25ec61dbbf2a2f3aa0790265917c6"
    )
    # And the coincidence the log relied on is STRUCTURAL: the anchor
    # writer emits no MANIFEST.json, so payload and declarable digests
    # are the same object (ANCHOR_ARTIFACT_PERSISTED's observation).
    import inspect

    writer_source = inspect.getsource(phase16.save_anchor_set)
    assert "MANIFEST.json" not in writer_source
    probe_run = next(e for e in loop["inputs"] if e["name"] == "probe_run")
    assert probe_run["rollup_sha256"].startswith("4573e984")
    # Everything is resolved; the header says so and no placeholder
    # remains anywhere in either config.
    for name in ("p16_anchor_loop.yaml", "p16_extract_anchor_embeddings.yaml"):
        text = (REPO / "configs" / name).read_text(encoding="utf-8")
        assert "**RESOLVED.**" in text, name
        assert "PLACEHOLDER" not in text, name
    assert probe_run["path"].endswith(
        "p7_d1_vit_b16_imagenet_g1__3f71a6a9__p7-d1-vit-imagenet"
    )


def test_anchors_appear_in_no_fold_structure_and_the_task_uses_the_gate():
    """EXIT_CRITERIA criterion 1's remaining clauses, in code."""
    import inspect
    import tempfile

    import numpy as np

    from cleft import run as run_module

    source = inspect.getsource(run_module.task_anchor_loop)
    # The training loop receives indices ONLY through the gate that
    # cannot express the full-cohort path.
    assert "phase16.training_indices(folds, held_out=held_out)" in source
    assert source.count("train_w(") == 1
    assert "features[train], truth_mean[train]" in source
    # Folds come from the manifest column and nowhere else.
    assert 'int(r["fold"])' in source
    # Anchors are loaded from their own artifact, which carries NO fold
    # key -- by writer construction.
    ids = [f"a{i}" for i in range(25)]
    grades = [1] * 3 + [2] * 7 + [3] * 6 + [4] * 6 + [5] * 3
    d = Path(tempfile.mkdtemp()) / "set"
    phase16.save_anchor_set(
        d, np.zeros((25, 8), dtype="float32"), stems=ids, grades=grades,
        parity=0.0, run_name="r",
    )
    _, metadata = phase16.load_anchor_set(d)
    assert "fold" not in metadata
    assert "no fold" in metadata["anchors_belong_to_no_fold"]


def test_train_w_smoke_identity_init_and_shared_readout():
    """A tiny end-to-end: at identity init the trained arm's readout IS
    the identity baseline's (one implementation), and training returns
    a finite W of the right shape with a valid checkpoint epoch."""
    import numpy as np
    import pytest as _pytest

    _pytest.importorskip("torch")

    rng = np.random.default_rng(0)
    dim = 8
    anchors = rng.normal(size=(25, dim))
    grades = np.array(
        [1] * 3 + [2] * 7 + [3] * 6 + [4] * 6 + [5] * 3, dtype=float
    )
    x = rng.normal(size=(40, dim))
    tau = phase16.anchor_tau(anchors, 1.0)
    y = phase16.expected_grade(x, anchors, grades, np.eye(dim), tau=tau)

    w, best_epoch, curve = phase16.train_w(
        x, y, anchors, grades, tau=tau, lambda_identity=0.01,
        learning_rate=0.001, batch_size=16, max_epochs=3,
        inner_val_frac=0.2, seed=1337,
    )
    assert w.shape == (dim, dim)
    assert np.all(np.isfinite(w))
    assert 0 <= best_epoch < 3
    assert len(curve) == 3  # the FIXED budget: every epoch runs
    # The trained arm and the identity baseline flow through the SAME
    # readout function -- the identity call reproduces the target
    # exactly, because the target was built with it.
    identity_readout = phase16.expected_grade(
        x, anchors, grades, np.eye(dim), tau=tau
    )
    assert np.allclose(identity_readout, y)


def test_the_parity_reference_load_uses_the_full_manifest_list():
    """[2026-08-29] The test that would have caught the extraction's
    load-time crash (run p16_extract_anchor_embeddings__e2b1eaf2, four
    identical attempts).

    ``embeddings.load(directory, manifest_ids)`` treats its second
    argument as a VERIFICATION CONTRACT -- the complete expected id
    list, asserted against the artifact by ``assert_row_order`` -- and
    never as a row selector. The task passed the single parity id,
    ``[patient_ids[0]]``, so the loader compared 237 stored ids against
    a 1-element expectation: 0 missing, 236 unexpected. This test
    exercises the load path against a multi-row fixture cache so the
    misuse fails in the suite, not on the cluster.

    **Pre-fix it fails** (the helper the fixed task routes through did
    not exist; the misuse lived inline); post-fix it passes and pins
    both the working shape and the refusal of the broken one.
    """
    import tempfile

    import numpy as np

    from cleft import embeddings
    from cleft.embeddings import EmbeddingError

    ids = [1, 2, 3, 4]
    values = np.arange(4 * 8, dtype="float32").reshape(4, 8)
    directory = Path(tempfile.mkdtemp()) / "vit_b16__imagenet__g1"
    embeddings.save(
        directory, values, backbone="vit_b16", backbone_kind="transformer",
        init="imagenet", geometry="g1", variant=None, checkpoint_sha256=None,
        patient_ids=ids, manifest_ids=ids,
    )

    # The fixed path: full manifest list in, the parity row out.
    row = phase16.parity_reference_row(directory, ids)
    assert np.allclose(row, values[0])

    # The broken call shape -- the run's exact crash, reproduced: a
    # single-element expectation against a multi-row cache.
    with pytest.raises(EmbeddingError, match="unexpected"):
        embeddings.load(directory, [ids[0]])
    # And the helper refuses to be handed a truncated list either, so
    # the misuse cannot be reintroduced THROUGH it.
    with pytest.raises(phase16.AnchorLoopError, match="full manifest"):
        phase16.parity_reference_row(directory, [ids[0]])


def test_every_cohort_cache_load_in_the_new_tasks_passes_the_full_list():
    """Item 3 of the diagnosis: the same misuse must not exist anywhere
    else in the phase's surface. Checked by AST call shape, not by
    word-matching."""
    import ast
    import inspect
    import textwrap

    from cleft import run as run_module

    def load_calls(function):
        tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
        return [
            node for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "load"
            and isinstance(node.func.value, ast.Name)
            and "embeddings" in node.func.value.id
        ]

    # The extraction task no longer calls the loader directly at all --
    # its one cohort-cache read routes through the helper this defect
    # produced.
    assert load_calls(run_module.task_extract_anchor_embeddings) == []
    source = inspect.getsource(run_module.task_extract_anchor_embeddings)
    assert "phase16.parity_reference_row(" in source
    assert "[int(p) for p in patient_ids]" in source

    # The loop task's load passes the full patient_ids name, and no
    # call anywhere in either task subscripts a single id into a list.
    calls = load_calls(run_module.task_anchor_loop)
    assert len(calls) == 1
    argument = calls[0].args[1]
    assert isinstance(argument, ast.Name) and argument.id == "patient_ids"
    for function in (run_module.task_extract_anchor_embeddings,
                     run_module.task_anchor_loop):
        tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
        for node in ast.walk(tree):
            if isinstance(node, ast.List) and len(node.elts) == 1:
                element = ast.unparse(node.elts[0])
                assert "patient_ids[0]" not in element, (
                    "a single-element id list is the crash's exact shape"
                )


# --------------------------------------------------------------------------
# 2026-08-29, the phase closed
# --------------------------------------------------------------------------


def test_the_threshold_and_ratio_rederive_from_the_banked_sds():
    """The ledger row's arithmetic, recomputed from the standing
    machinery rather than trusted as prose."""
    from cleft.train.phase3 import combined_claimable_delta

    threshold = combined_claimable_delta(0.0345, 5, 0.0148, 5)
    assert round(threshold["arm_means_95"], 4) == 0.0329
    assert round(abs(-0.0481) / threshold["arm_means_95"], 2) == 1.46
    # Condition 2 passes on the mean-of-means band; the single-run band
    # would NOT be cleared -- which is condition 2's registered scope,
    # not a loophole.
    assert abs(-0.0481) < threshold["single_run_95"]


def test_self_consistency_attaches_with_both_sentences_and_the_lock_limit():
    record = phase16.SELF_CONSISTENCY_ATTACHED
    numbers = record["the_numbers"]
    assert "6.76/25" in numbers
    assert "22 of 25" in numbers
    assert "4.75/25" in numbers
    assert "DESCRIPTIVELY" in numbers

    absolute = record["the_absolute_reading_beside_it"]
    assert "27%" in absolute
    assert "three anchors" in absolute
    assert "weak in absolute terms" in absolute
    # The percentage is the fraction, not a typo: 6.76/25 = 27.04%.
    assert round(6.76 / 25 * 100) == 27

    limitation = record["limitation_of_the_lock"]
    assert "no " + "registered claimability machinery" in limitation
    assert "claimably above" in limitation
    assert "post-hoc" in limitation
    # In the lock clause's OWN language, and the clause really says it.
    assert "limitation of the lock" in limitation
    assert "never a" in phase16.EXIT_CRITERIA["locked"]
    assert "retro-fitted entry" in phase16.EXIT_CRITERIA["locked"]
    assert "'claimably' goes unexercised" in limitation


def test_repair_without_transfer_carries_both_numbers_and_the_pairing():
    record = phase16.REPAIR_WITHOUT_TRANSFER
    finding = record["the_finding"]
    assert "EMERGENT, not the" in finding
    assert "never in the objective" in finding
    assert "4/25 -> 6.76/25" in finding
    assert "0.2151" in finding and "0.2040" in finding
    assert "2 of 5 seeds" in finding
    assert "does not transfer" in finding
    assert "That pairing is the finding" in finding
    assert "not the same" in record["why_the_pairing_matters"]
    # Bounded: unresolved row named, no claimability, one backbone.
    bounds = record["bounds"]
    assert "p16-anchor-loop-unresolved" in bounds
    assert "limitation_of_the_lock" in bounds


def test_the_epoch_zero_cross_check_travels_with_its_limit():
    record = phase16.EPOCH_ZERO_CROSS_CHECK
    assert "3/3/4" in record["the_observation"]
    assert "8/8" in record["the_observation"]
    assert "epochs 15, 12" in record["the_observation"]
    assert "measurement behaves" in record["the_observation"]
    limit = record["the_limit_beside_it"]
    assert "TRAINING DATA as well as training" in limit
    assert "suggestive, not a paired comparison" in limit


def test_the_phase9_refinement_is_dated_preserves_and_points_both_ways():
    observed = phase9.PROTOTYPE_CLASSIFIER_OBSERVED
    # The original wording is preserved, byte for byte where pinned.
    assert observed["anchor_self_consistency"]["euclidean"] == "4/25"
    assert observed["pcc_vs_trained_head"] == (
        "0.15-0.18 everywhere, under 0.2520"
    )
    assert "FOURTH convergent" in observed["convergence"]

    refinement = observed["readout_refined_2026_08_29"]
    assert "READOUT, not the" in refinement
    assert "0.2151" in refinement and "0.1823" in refinement
    assert "sd ~3e-17" in refinement
    assert "does NOT overturn" in refinement
    assert "DESCRIPTIVE only" in refinement
    assert "criterion cannot apply" in refinement
    # Pointers both ways: phase9 -> phase16, phase16 -> phase9's record.
    assert "phase16.IDENTITY_BASELINE" in refinement

    # [TIGHTENED 2026-09-06] This assertion used to accept either the
    # name PASS_ZERO_READOUT_REFINED or the bare string "Phase 9". The
    # disjunct meant the name was never checked, and the name did not
    # exist: phase9 has no such constant. The pointer is corrected and
    # the pin now RESOLVES it rather than matching a string.
    back = phase16.PHASE_16_CLOSING["criterion_6_identity_baseline"]
    assert "readout_refined_2026_08_29" in back
    assert "PASS_ZERO_READOUT_REFINED" not in back
    assert not hasattr(phase9, "PASS_ZERO_READOUT_REFINED")
    # And the name it now cites resolves to a real key.
    assert "readout_refined_2026_08_29" in phase9.PROTOTYPE_CLASSIFIER_OBSERVED

    # The correction is recorded rather than made silently.
    corrected = " ".join(
        phase16.PHASE_16_CLOSING["the_pointer_in_criterion_6_CORRECTED_2026_09_06"]
        .split()
    )
    assert "a name that exists nowhere in phase9" in corrected
    assert "No figure changes and no reading changes" in corrected
    assert "its pin accepted the string 'Phase 9'" in corrected
    # And the refinement's identity figure matches the closing's.
    assert "0.2151" in phase16.PHASE_16_CLOSING["criterion_6_identity_baseline"]


def test_the_compute_shape_record_points_forward_to_what_superseded_it():
    """[ADDED 2026-09-06] Two entries in COMPUTE_SHAPE_VERIFIED were the
    state of an afternoon: the anchor embeddings 'DO NOT EXIST' and the
    persist-or-re-extract choice is 'not decided'. Both were overtaken
    the same day. The originals stand and the backward pointer is now
    there, because a reader who lands here first was being told the
    artifact is absent."""
    record = phase16.COMPUTE_SHAPE_VERIFIED

    # The originals are preserved, word for word on the load-bearing part.
    assert "DO NOT EXIST as an artifact" in record["anchor_embeddings"]
    assert "not decided" in " ".join(
        record["a_build_option_flagged_not_decided"].split()
    ) or "flagged so it is not made by default" in record[
        "a_build_option_flagged_not_decided"
    ]

    superseded = " ".join(record["BOTH_ENTRIES_ABOVE_ARE_SUPERSEDED_2026_09_06"].split())
    assert "They exist" in superseded
    assert "It was decided" in superseded
    assert "anchor_deall_v1" in superseded
    assert "the SAME day" in superseded

    # The successor really does name this record, which is the asymmetry
    # the fix removes.
    forward = " ".join(phase16.ANCHOR_ARTIFACT_PERSISTED["decided"].split())
    assert "COMPUTE_SHAPE_VERIFIED flagged, now decided: persist once" in forward

    why = " ".join(record["why_the_pointer_was_missing_and_why_it_matters"].split())
    assert "the forward pointer existed and the backward one did not" in why
    assert "only works if you already knew the answer" in why


def test_the_artifact_the_stale_entry_denies_is_declared_at_a_real_hash():
    """The correction says the artifact exists. Checked against the
    config that declares it rather than against the prose."""
    import yaml

    config = yaml.safe_load(
        (REPO / "configs" / "p16_anchor_loop.yaml").read_text(encoding="utf-8")
    )
    declared = {entry["name"]: entry for entry in config["inputs"]}
    assert "anchor_embeddings" in declared
    assert declared["anchor_embeddings"]["path"].endswith("anchor_deall_v1")
    rollup = declared["anchor_embeddings"]["rollup_sha256"]
    assert len(rollup) == 64 and set(rollup) <= set("0123456789abcdef")

def test_the_closing_walks_all_seven_criteria():
    closing = phase16.PHASE_16_CLOSING
    assert "f342fed9" in closing["closed"]
    assert "single attempt" in closing["closed"]

    walked = [k for k in closing if k.startswith("criterion_")]
    assert len(walked) == 7
    for key in walked:
        assert "MET" in closing[key] or "the lock held" in closing[key], key

    assert "in the run directory" in closing["criterion_5_mismatch_records"]
    assert "237 rows" in closing["criterion_5_mismatch_records"]
    assert "p16-anchor-loop-unresolved" in closing["criterion_4_primary_contrast"]
    # [2026-08-31] "the ledger's first such case" is WRONG AS WRITTEN --
    # this is the FIFTH condition-2-pass/condition-1-fail row. The
    # assertion STAYS because the closing's wording is preserved, not
    # rewritten; the dated correction sits beside it in the same record
    # and is asserted here so the two cannot drift apart.
    assert "first such case" in closing["criterion_4_primary_contrast"]
    correction = closing["first_such_case_corrected_2026_08_31"]
    assert "ledger-condition-split-count-corrected" in correction
    assert "FIFTH" in correction
    assert "No measurement changes" in correction

    # The tau degenerate mode: named alternative, discharged.
    tau = closing["tau_degenerate_mode_not_observed"]
    assert "did NOT appear" in tau
    assert "2.96" in tau and "0.502" in tau
    assert "never tuned" in tau.replace("\n", " ")

    caveats = closing["caveats_inherited"]
    assert "not verified-unanimous" in caveats
    assert "question 5's neighbour" in caveats
    assert "one backbone" in caveats
    assert "ImageNet anchor rides" in caveats

    # The ledger row it cites exists with the status it says.
    from cleft import results_ledger

    entry = next(
        e for e in results_ledger.ENTRIES
        if e["id"] == "p16-anchor-loop-unresolved"
    )
    assert entry["status"] == "UNRESOLVED-WITHDRAWN"
