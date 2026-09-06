"""Phase 17 -- the TSTR rulings, the amendment, the compute gate.

No arm exists yet; the restate waits on the measurement. These tests pin
the amendment (original preserved, adoption flag quoted not
paraphrased), the claim-wording bound in the rulings, and the
outcome readings committed before the measurement's numbers exist.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from cleft import phase11, phase17
from cleft.scut import synthesis

REPO = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------
# the amendment
# --------------------------------------------------------------------------


def test_the_amendment_preserves_the_original_and_withdraws_the_word():
    # The original line, byte-preserved.
    assert synthesis.PARKED["arm"] == (
        "Stage F, TSTR vs CV-on-237 (Rosero replication)"
    )
    amendment = synthesis.PARKED["arm_amended_2026_08_29"]
    assert "NOT Rosero's" in amendment
    assert "piecewise affine over landmark triangulation" in amendment
    assert "binary symmetric/asymmetric pairs" in amendment
    assert "only at readout" in amendment
    assert "TPS and trains on magnitude-ordered labels" in amendment
    assert "'replication' is withdrawn" in amendment
    assert "Rosero family" in amendment
    assert "assumption Rosero's construction avoids" in amendment


def test_the_adoption_flag_is_cited_by_name_and_quoted_not_paraphrased():
    """the instruction: the adoption-time flag's wording differs
    from 'stronger assumption', so the amendment must quote what it
    actually says."""
    amendment = synthesis.PARKED["arm_amended_2026_08_29"]
    assert "magnitude_to_grade_is_an_assumption" in amendment
    # The quoted words are really the flag's own words.
    flag = synthesis.LIMITATIONS["magnitude_to_grade_is_an_assumption"]
    for quoted in ("ORDERED series, not a graded one",
                   "larger means more deformed and that is all"):
        assert quoted in amendment, quoted
        assert quoted in flag, quoted
    # And the amendment is honest that "stronger than Rosero" is the
    # verification's own dated addition, absent from the adoption record.
    assert "makes no comparison to Rosero" in amendment
    assert "Rosero" not in flag
    assert "confirms" in amendment.lower()
    # Pointer forward to the phase that consumes it.
    assert "phase17.PHASE_17_RULINGS" in amendment


def test_the_four_adjacent_wordings_carry_dated_pointers_unrenamed():
    from cleft import phase12, phase15

    # The dict values are preserved as written...
    assert "Rosero-design arm" in phase12.PHASE_SEQUENCE_RENUMBERED_2[
        "becomes"
    ]["16"]
    assert "Rosero-design arm" in phase15.PHASE_SEQUENCE_RENUMBERED_3[
        "becomes"
    ]["17"]
    # ...and every module that says "Rosero" carries the pointer.
    for module in (phase12, phase15):
        source = Path(module.__file__).read_text(encoding="utf-8")
        assert "Rosero" in source
        assert "arm_amended_2026_08_29" in source, module.__name__


# --------------------------------------------------------------------------
# the rulings
# --------------------------------------------------------------------------


def test_the_three_arms_and_the_bound_claim_wording():
    record = phase17.PHASE_17_RULINGS
    assert "as parked" in record["arm_a"]
    assert "stronger-assumption cousin" in record["arm_a"]
    assert "not predicted to\n        succeed".replace("\n        ", " ") in (
        record["arm_a"].replace("\n", " ")
    ) or "not predicted to" in record["arm_a"]

    arm_b = record["arm_b"]
    assert "method replication with two declared adaptations" in arm_b
    assert "harder label" in arm_b
    assert "SHIPPED" in arm_b and "no detector anywhere" in arm_b
    assert "DECLARED, not discovered" in arm_b
    assert "shipped landmarks in place of detection" in arm_b
    assert "fixed" in arm_b and "midline in place of a landmark-localised" in arm_b
    assert "never 'we replicated Rosero'" in arm_b

    assert "TPS synthesis" in record["arm_c"]
    assert "B's" in record["arm_c"]


def test_frozen_branches_with_the_scoping_note_and_the_unexercised_path():
    record = phase17.PHASE_17_RULINGS
    frozen = record["frozen_branches"]
    assert "frozen ViT-B/16 branches" in frozen
    assert "trained linear projection" in frozen
    assert "'transfer' in Phase" in frozen and "17 means THIS" in frozen
    assert "NEVER-OPERATIONALISED" in frozen

    unexercised = record["trainable_branches_unexercised"]
    assert "UNEXERCISED ALTERNATIVE" in unexercised
    assert "THREE NOVELTIES AT ONCE" in unexercised
    assert "could\n        not be attributed".replace("\n        ", " ") in (
        unexercised.replace("\n", " ")
    ) or "not be attributed" in unexercised


def test_the_landmark_question_is_on_the_supervisor_list_with_its_basis():
    questions = phase11.SUPERVISOR_CONSOLIDATED_ASK["questions"]
    assert [q["n"] for q in questions] == [2, 3, 4, 5, 6, 7, 8, 9]
    nine = questions[-1]
    assert nine["n"] == 9
    assert nine["blocking"] is False
    assert "shipped landmark files" in nine["ask"]
    assert "no detector anywhere" in nine["ask"]
    assert "crosses the line you drew" in nine["ask"]
    # The basis, quoted from where it lives -- and it really lives there.
    assert "rejected the one from its own" in nine["why"]
    mirror_source = " ".join(
        (REPO / "src" / "cleft" / "geometry" / "mirror.py")
        .read_text(encoding="utf-8").split()
    )
    assert "the supervision material rejected the one from its own group" in mirror_source
    assert "Phase 5" in nine["why"] and "Phase 15" in nine["why"]
    assert nine["records"] == "phase17.PHASE_17_RULINGS"
    assert "question 9" in phase17.PHASE_17_RULINGS["landmark_boundary_asked"]


def test_the_five_contrast_family_is_registered_with_its_skeleton():
    record = phase17.PHASE_17_RULINGS
    family = record["the_five_contrast_family"]
    assert "count = FIVE" in family
    assert "nothing added after" in family
    assert "Three PRIMARIES" in family
    assert "Two SECONDARIES" in family
    assert "A-C" in family and "SHARED synthesis" in family
    assert "B-C" in family and "SHARED training" in family

    skeleton = record["combination_readings_skeleton"]
    assert "SKELETON, to be completed at the restate" in skeleton
    assert "non-independent" in skeleton
    assert "before the numbers" in skeleton
    for cell in ("(i)", "(ii)", "(iii)", "(iv)", "(v)"):
        assert cell in skeleton, cell
    assert "none is filled tonight" in skeleton

    assert "five seeds per arm" in str(record["five_seeds_per_arm"]) or (
        "five" in record["five_seeds_per_arm"]
    )
    assert "paired_comparison" in record["five_seeds_per_arm"]
    assert "all 237" in record["five_seeds_per_arm"]


# --------------------------------------------------------------------------
# the compute gate
# --------------------------------------------------------------------------


def test_the_timing_observability_claim_is_true_of_the_source():
    """The record says sheet-only mode logs nothing per face; verified
    against the task source, not trusted."""
    import inspect

    from cleft import run as run_module

    source = inspect.getsource(run_module.task_asymmetry_synthesis)
    # The only per-face log is the checkpoint line, and it is guarded
    # by write_artifact.
    checkpoint_at = source.index("checkpoint at face")
    guard_at = source.rindex("if write_artifact:", 0, checkpoint_at)
    assert guard_at != -1
    # No other ctx.log call sits inside the per-face loop body between
    # the loop head and the checkpoint guard.
    loop_at = source.index("for index in range(start_index, n_faces):")
    body = source[loop_at:guard_at]
    # [UPDATED 2026-08-29, same day] This line read `"ctx.log" not in
    # body` -- true when the observability claim was recorded, and
    # FALSE the moment the reported timing line was applied on El
    # the maintainer's go. The pin fired on the change it described; updated
    # dated: the ONLY log in the pre-checkpoint body is now the timing
    # line itself.
    assert body.count("ctx.log(") == 1
    assert "face_started" in body

    record = phase17.COMPUTE_GATE_MEASUREMENT
    assert "logs NOTHING per face" in record["timing_observability"]
    assert "one-second resolution" in record["timing_observability"]
    change = record["smallest_honest_change_reported_not_applied"]
    assert "not applied this turn" in change
    assert "perf_counter" in change
    assert "resolution-independent" in change


def test_both_outcome_readings_are_committed_before_the_numbers():
    record = phase17.COMPUTE_GATE_MEASUREMENT
    collapse = record["reading_if_the_gap_collapses"]
    assert "ENVIRONMENTAL" in collapse
    assert "futex" in collapse
    assert "CONFIRMED by measurement" in collapse
    assert "discharged by measurement" in collapse
    assert "unconstrained" in collapse

    persists = record["reading_if_the_gap_persists"]
    assert "UNRESOLVED-AND-NOT-GATING" in persists
    assert "SIZED TO THE MEASURED COST" in persists

    assert "GPUs are idle throughout" in record["gpus_idle_is_expected"]
    assert "THREAD" in record["gpus_idle_is_expected"]
    assert "not hardware" in record["gpus_idle_is_expected"]

    sizing = record["no_size_until_both_numbers"]
    assert "no synthesis set size is declared until both" in sizing
    assert "measure-don't-reason" in sizing

    # The diagnosis it tests is the one on record.
    docstring = (
        REPO / "src" / "cleft" / "scut" / "synthesis.py"
    ).read_text(encoding="utf-8")
    assert "BLAS thread-pool spin" in docstring
    assert "99.98%" in docstring


def test_no_arm_is_built():
    """[RETIRED 2026-08-29, converted] The negative space held from the
    rulings to the exit-criteria lock; the measurement landed (third
    outcome -- no pathology), the lock followed, and the successor
    asserts the positive space."""
    from cleft.config.schema import TASK_SPECS
    from cleft.run import TASKS

    for registry in (TASK_SPECS, TASKS):
        assert "siamese_contrastive" in registry
        assert "tstr_regression" in registry
        assert "tstr_family_analysis" in registry
    assert (REPO / "configs" / "p17_arm_a.yaml").is_file()


def test_the_timing_line_is_applied_in_the_right_task_and_only_there():
    """[2026-08-29] The reported change, applied on the maintainer's go --
    and scoped: the first application landed in task_masked_scut, whose
    loop tail is near-identical, so this test pins WHICH function
    carries the line."""
    import inspect

    from cleft import run as run_module

    source = inspect.getsource(run_module.task_asymmetry_synthesis)
    assert "face_started = time.perf_counter()" in source
    assert 'f"face {index + 1}/{n_faces} in "' in source
    # Unconditional: the log sits before the write_artifact block, so it
    # fires in sheet-only mode and a checkpoint flush cannot spike one
    # face's number.
    log_at = source.index("face {index + 1}/{n_faces} in")
    artifact_at = source.index(
        "if write_artifact:\n            with faces_path"
    )
    assert log_at < artifact_at
    # And ONLY there -- the near-identical loop it first landed in is
    # untouched.
    other = inspect.getsource(run_module.task_masked_scut)
    assert "face_started" not in other
    record = phase17.COMPUTE_GATE_MEASUREMENT
    assert record["timing_line_applied"] == "2026-08-29, exactly as reported"


def test_the_job_forms_use_the_pinned_image_and_differ_only_in_blas_pins():
    forms = phase17.COMPUTE_GATE_MEASUREMENT["job_forms"]
    image = forms["image_both_runs"]
    # The image is the models' own MEASURED_IN_IMAGE value, whole --
    # never a partial digest (the workflow-hygiene rule).
    from cleft.models import agnet

    assert phase17.PINNED_IMAGE == agnet.MEASURED_IN_IMAGE
    assert phase17.PINNED_IMAGE in image
    assert "pathology was observed" in image
    # The corrected image is the one PLAN pins and the models measured in.
    plan = (REPO / "docs" / "PLAN.md").read_text(encoding="utf-8")
    assert phase17.PINNED_IMAGE in plan

    assert "entrypoint.sh" in forms["command_both_runs"]
    assert "p5_asymmetry_synthesis.yaml" in forms["command_both_runs"]
    env = forms["env_both_runs"]
    for var in ("CLEFT_REPO", "CLEFT_OUT", "CLEFT_IMAGE_DIGEST",
                "CLEFT_JOB_ID", "CLEFT_SCUT_ROOT"):
        assert var in env, var
    assert "1041ceed" in env
    # The declared root really is the config's one env-resolved input.
    config_text = (
        REPO / "configs" / "p5_asymmetry_synthesis.yaml"
    ).read_text(encoding="utf-8")
    assert "${CLEFT_SCUT_ROOT}" in config_text
    assert "1041ceed" in config_text
    run2 = forms["run_2_adds_only"]
    for var in ("OMP_NUM_THREADS=1", "MKL_NUM_THREADS=1",
                "OPENBLAS_NUM_THREADS=1"):
        assert var in run2, var
    assert "NOTHING else differs" in run2


# --------------------------------------------------------------------------
# 2026-08-29, the gate closed, the lock, the readings
# --------------------------------------------------------------------------


def test_the_compute_gate_closed_on_the_third_outcome():
    record = phase17.COMPUTE_GATE_MEASUREMENT
    outcome = record["outcome_2026_08_29"]
    assert "THIRD outcome" in outcome
    assert "neither committed reading anticipated" in outcome
    for figure in ("0.55-1.17", "0.21-0.59", "0.23-0.53"):
        assert figure in outcome, figure
    assert "no pathology present to diagnose" in outcome
    assert "warm baseline" in outcome

    closed = record["cluster_performance_closed"]
    assert "not-gating" in closed
    assert "NOT reproduced" in closed
    assert "56 s/face" in closed and "0.2-1.2 s/face" in closed
    assert "never tested because the disease did not show" in closed
    assert "UNADJUDICATED" in closed
    assert "preserved as history" in closed
    assert "full SCUT" in record["set_size"]
    # The synthesis module's record carries the dated closure, original
    # docstring preserved.
    assert "closed_2026_08_29" in synthesis.CLUSTER_PERFORMANCE_UNRESOLVED
    assert "not-gating" in synthesis.CLUSTER_PERFORMANCE_UNRESOLVED[
        "closed_2026_08_29"
    ]
    # The tier mechanism, one sentence, confirmed against the config.
    assert "tier" in record["dev_tier_mechanism"]
    config_text = (
        REPO / "configs" / "p5_asymmetry_synthesis.yaml"
    ).read_text(encoding="utf-8")
    assert "tier: dev" in config_text


def test_the_exit_criteria_are_locked_with_six_and_the_clause():
    record = phase17.EXIT_CRITERIA
    criteria = record["criteria"]
    assert len(criteria) == 6
    assert "full-SCUT" in criteria[0] and "G1" in criteria[0]
    assert "G2 recorded as unregistered" in criteria[0]
    assert "(0.0, 0.015, 0.025, 0.035)" in criteria[0]
    assert "zero real patient images" in criteria[1]
    assert "all 237 as pure test" in criteria[1]
    assert "nothing beyond it" in criteria[2]
    assert "claimable/withdrawn/unresolved" in criteria[2]
    assert "0.333" in criteria[3] and "0.502" in criteria[3]
    assert "before any synthesis-set number exists" in criteria[4]
    assert "2026-08-29" in criteria[5]
    assert "nothing is added after" in record["locked"]


def test_the_readings_are_committed_both_ways_with_all_five_cells():
    record = phase17.READINGS_COMMITTED
    assert "before any synthesis-set number exists" in record["committed"]
    # A carries the parked prediction.
    assert "not predicted to succeed" in record["arm_a_carried"]
    assert "philtrum" in record["arm_a_carried"]
    # B, both ways, with the unattributable-adaptations honesty.
    b_null = record["arm_b_if_null"]
    assert "0.31" in b_null and "CARS" in b_null
    assert "more than symmetry" in b_null
    assert "unattributable between them" in b_null
    b_pos = record["arm_b_if_positive"]
    assert "zero-real-image" in b_pos
    assert "no scar" in b_pos and "CARS-vs-composite" in b_pos
    # C, both ways.
    assert "parked prediction extending to the training scheme" in record[
        "arm_c_if_null"
    ]
    assert "rescues what magnitude-labels could not" in record[
        "arm_c_if_positive"
    ]
    assert "identical images" in record["arm_c_if_positive"]
    # The five cells, now committed sentences (the skeleton filled).
    cells = record["combination_cells"]
    assert len(cells) == 5
    assert "SIXTH convergent null" in cells["all_null"]
    assert "regardless of scheme or transformation" in cells["all_null"]
    assert "most informative outcome" in cells["b_above_a_c_near_b"]
    assert "magnitude-to-grade assumption" in cells["b_above_a_c_near_b"]
    assert "transformation family matters" in cells["b_above_c"]
    assert "anchored philtrum" in cells["b_above_c"]
    assert "never worded as beating the probe" in cells["any_positive"]
    assert "cannot attribute" in cells["discordant_secondaries"]
    assert "unresolved" in cells["discordant_secondaries"]


def test_the_declared_settings_carry_their_reasoning_and_values():
    record = phase17.DECLARED_SETTINGS_17
    # The A-arm magnitude-to-grade map, tau-style.
    mapping = record["magnitude_to_grade_map"]
    assert "1 + 4" in mapping and "0.035" in mapping
    assert "map IS the assumption" in mapping
    assert "linear" in mapping.lower()
    import numpy as np

    grades = phase17.magnitude_to_grade(
        np.array([0.0, 0.015, 0.025, 0.035])
    )
    assert grades[0] == 1.0 and grades[-1] == 5.0
    assert abs(grades[1] - (1 + 4 * 0.015 / 0.035)) < 1e-12
    # Monotone in magnitude -- ordered stays ordered.
    assert list(grades) == sorted(grades)

    margin = record["margin"]
    assert "1.0" in margin
    assert "units choice" in margin
    readout = record["readout_normalization"]
    assert "min(d/margin" in readout.replace(" ", "") or "d/margin" in readout
    assert "rounded" in readout.lower()
    # The readout function: distance 0 -> grade 1; >= margin -> grade 5.
    assert phase17.distance_to_grade(0.0, margin=1.0) == 1
    assert phase17.distance_to_grade(1.0, margin=1.0) == 5
    assert phase17.distance_to_grade(2.5, margin=1.0) == 5
    assert phase17.distance_to_grade(0.5, margin=1.0) == 3
    projection = record["projection_dim"]
    assert "768" in projection
    assert "rank knob" in projection or "low-rank" in projection

    provenance = phase17.SETTINGS_PROVENANCE_17
    for key in ("max_epochs", "learning_rate", "batch_size",
                "inner_val_frac", "seeds", "monitor"):
        assert "p7_d1_vit_b16_imagenet_g1" in provenance[key], key
    assert "ABSENT BY RULING" in provenance["patience"]
    for key in ("margin", "readout_normalization", "magnitude_to_grade_map"):
        assert "no precedent" in provenance[key].lower(), key


def test_left_right_views_split_at_fixed_midline_with_white_fill():
    import numpy as np

    image = np.zeros((8, 8, 3), dtype=np.uint8)
    image[:, :4] = 10   # left half
    image[:, 4:] = 200  # right half
    left, right = phase17.left_right_views(image)
    assert left.shape == image.shape and right.shape == image.shape
    # Left view keeps the left half, fills the right with staging white.
    assert (left[:, :4] == 10).all()
    assert (left[:, 4:] == 255).all()
    # Right view is MIRRORED so anatomy aligns across the pair, then
    # filled the same way: its left columns carry the mirrored right
    # half.
    assert (right[:, :4] == 200).all()
    assert (right[:, 4:] == 255).all()
    # A symmetric image yields identical views -- the property the
    # contrastive loss trains against.
    symmetric = np.zeros((8, 8, 3), dtype=np.uint8)
    symmetric[:, :] = 7
    a, b = phase17.left_right_views(symmetric)
    assert (a == b).all()


def test_the_piecewise_warp_mirrors_the_tps_suites_guarantees():
    """Locality, exact target displacement, anchor stillness, mirrored
    sides -- the TPS suite's pattern, on the new warp family."""
    import numpy as np

    from cleft.scut import piecewise, synthesis as synth

    rng = np.random.default_rng(3)
    image = (rng.uniform(0, 255, (224, 224, 3))).astype(np.uint8)
    # A synthetic landmark set: reuse the TPS suite's approach of real
    # geometry -- control points from the module's own rule.
    source, target = piecewise.displaced_control_points_for_tests()
    warped = piecewise.warp_points(image, source, target)
    assert warped.shape == image.shape
    # Vertices map exactly: a pixel at an ANCHOR (zero displacement)
    # is untouched.
    still = piecewise.warp_points(image, source, source)
    assert (still == image).all()

    # The module records its two declared adaptations.
    record = piecewise.DECLARED_ADAPTATIONS
    assert "shipped landmarks in place of detection" in record["adaptation_1"]
    assert "fixed midline in place of a landmark-localised" in record[
        "adaptation_2"
    ]
    assert "DECLARED, not discovered" in record["stated"]
    # No skimage, no detector -- nothing beyond the pinned image.
    # Checked by AST over the module's IMPORTS, not by word-matching
    # (the docstring legitimately names skimage to say it is absent).
    import ast

    tree = ast.parse(
        (REPO / "src" / "cleft" / "scut" / "piecewise.py").read_text(
            encoding="utf-8"
        )
    )
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    for forbidden in ("skimage", "dlib", "mediapipe", "face_recognition"):
        assert forbidden not in imported, forbidden
    assert "scipy" in imported  # Delaunay from the pinned image


def test_the_siamese_schema_refuses_trainable_with_the_reason():
    import copy

    from cleft.config.schema import TASK_SPECS, _validate_mapping
    from cleft.config import ConfigError

    spec = TASK_SPECS["siamese_contrastive"]
    # Required-no-default scientific settings.
    for key in ("branch_trainability", "margin", "pair_rule",
                "readout_normalization", "seeds", "max_epochs",
                "learning_rate", "batch_size", "inner_val_frac"):
        assert key in spec, key
        assert spec[key].required and spec[key].default is None, key
    assert "patience" not in spec
    assert spec["branch_trainability"].choices == ("frozen",)

    task = yaml.safe_load(
        (REPO / "configs" / "p17_arm_b.yaml").read_text(encoding="utf-8")
    )["task"]
    _validate_mapping(task, spec, "task")  # loads clean
    crossed = copy.deepcopy(task)
    crossed["branch_trainability"] = "trainable"
    try:
        _validate_mapping(crossed, spec, "task")
        raise AssertionError("trainable branches were not refused")
    except ConfigError as error:
        assert "frozen" in str(error)

    # train_cv's vocabulary stays closed.
    # [UPDATED 2026-09-02, THIS PIN FIRED AS DESIGNED] Phase 25's ruling
    # added two FACTORY-PRETRAINED inits -- DINOv2 on LVD-142M and DINO on
    # unlabelled ImageNet-1k. They are named apart from "imagenet" because
    # neither is ImageNet-supervised, and the set directory carries the
    # init in its name (phase25.THE_WEIGHTS_ARE_FACTORY_PRETRAINED_LIKE_
    # THE_PROBES). The three SCUT/ImageNet values are unchanged.
    assert TASK_SPECS["train_cv"]["init"].choices == (
        "imagenet", "scut_original", "scut_masked",
        "dinov2_lvd142m", "dino_in1k",
    )


def test_no_real_patient_data_reaches_any_training_path():
    """The TSTR premise, structurally: in every arm task, the manifest
    and staged/embeddings patient inputs are read only AFTER training
    completes, and the patient label column never feeds a fit."""
    import inspect

    from cleft import run as run_module

    for task_name in ("task_tstr_regression", "task_siamese_contrastive"):
        source = inspect.getsource(getattr(run_module, task_name))
        train_at = source.index("TRAINING ON SYNTHETIC ONLY")
        evaluate_at = source.index("EVALUATION ON REAL, AFTER TRAINING")
        assert train_at < evaluate_at
        training = source[:evaluate_at]
        # No patient-side artifact is touched in the training section.
        for name in ("manifest_dir", "staged_dir", "load_manifest",
                     "patient", "truth"):
            assert name not in training.split(
                "TRAINING ON SYNTHETIC ONLY"
            )[1], (task_name, name)


def test_the_six_configs_and_their_declare_rhythm():
    expectations = {
        "p17_synth_tps_g1.yaml": {"scut_root": True},
        "p17_synth_pwa_g1.yaml": {"scut_root": True},
        # [2026-08-30] A and C flipped True when the TPS set's hash
        # (855c13fa...) landed; B flipped True the same day when El
        # the maintainer ruled on the PWA artifact and declared it (878626ee...)
        # -- each fill fired this pin, which is what pinning the
        # declare rhythm is for. All three arms are now fully resolved;
        # only the family analysis pends, on the three arm run dirs.
        "p17_arm_a.yaml": {
            "synth_set": True, "manifest_v1": True, "embeddings": True,
        },
        "p17_arm_b.yaml": {
            "synth_set": True, "manifest_v1": True, "staged_v1": True,
        },
        "p17_arm_c.yaml": {
            "synth_set": True, "manifest_v1": True, "staged_v1": True,
        },
        # [2026-08-30, pass 2] The three arm hashes landed (cfab87ac /
        # dc734d82 / 7ecbfd08) and this pin fired on the state change,
        # as it did on every fill before it. **Every Phase-17 config is
        # now fully resolved** -- the family analysis is launchable and
        # nothing in the phase pends on a declaration.
        "p17_family_analysis.yaml": {
            "arm_a_run": True, "arm_b_run": True, "arm_c_run": True,
            "probe_run": True, "manifest_v1": True,
        },
    }
    for name, inputs in expectations.items():
        payload = yaml.safe_load(
            (REPO / "configs" / name).read_text(encoding="utf-8")
        )
        declared = {
            e["name"]: set(e["rollup_sha256"]) != {"0"}
            for e in payload["inputs"]
        }
        assert declared == inputs, (name, declared)
    # The two synthesis configs are launchable NOW; every pending hash
    # is downstream of them.
    for name in ("p17_synth_tps_g1.yaml", "p17_synth_pwa_g1.yaml"):
        payload = yaml.safe_load(
            (REPO / "configs" / name).read_text(encoding="utf-8")
        )
        task = payload["task"]
        assert task["geometries"] == ["g1"]
        assert task["n_faces"] == 0
        assert task["write_artifact"] is True
        assert task["magnitudes"] == [0.0, 0.015, 0.025, 0.035]
    tps = yaml.safe_load(
        (REPO / "configs" / "p17_synth_tps_g1.yaml").read_text(
            encoding="utf-8"
        )
    )["task"]
    pwa = yaml.safe_load(
        (REPO / "configs" / "p17_synth_pwa_g1.yaml").read_text(
            encoding="utf-8"
        )
    )["task"]
    assert tps.get("warp_family", "tps") == "tps"
    assert pwa["warp_family"] == "piecewise_affine"


# --------------------------------------------------------------------------
# 2026-08-30, the phase closed
# --------------------------------------------------------------------------


def test_the_unpredicted_pattern_preserves_the_cells_unfired():
    record = phase17.UNPREDICTED_PATTERN
    assert "amend nothing" in record["observed"]
    assert "none of the five committed combination cells" in record[
        "no_cell_fired"
    ]
    assert "PRESERVED AND" in record["no_cell_fired"]

    pattern = record["the_observed_pattern"]
    assert "OPPOSITE to expectation" in pattern
    assert "0.2334" in pattern and "-0.0044" in pattern
    assert "-0.0185" in pattern
    assert "expecting B" in pattern
    assert "built ON" in pattern

    honesty = record["what_this_is_and_is_not"]
    assert "OBSERVATION, not a registered reading" in honesty
    assert "attribution is not claimed" in honesty
    assert "NOT amended" in honesty
    assert "not a reading" in honesty

    # The five cells really are unchanged: every committed sentence
    # still present, none rewritten to fit.
    cells = phase17.READINGS_COMMITTED["combination_cells"]
    assert len(cells) == 5
    assert "SIXTH convergent null" in cells["all_null"]
    assert "most informative outcome" in cells["b_above_a_c_near_b"]
    assert "transformation family matters" in cells["b_above_c"]
    assert "never worded as beating the probe" in cells["any_positive"]
    assert "recorded unresolved" in cells["discordant_secondaries"]


def test_the_prediction_reckoning_preserves_the_original():
    limitations = synthesis.LIMITATIONS
    original = limitations["the_arm_is_not_predicted_to_succeed"]
    assert "THIS ARM IS NOT PREDICTED TO SUCCEED" in original
    assert "|Spearman| 0.151" in original
    assert "Written down BEFORE the arm runs" in original

    reckoning = " ".join(
        limitations["prediction_reckoned_2026_08_30"].split()
    )
    assert "contradicts the prediction in substance" in reckoning
    assert "0.2334" in reckoning and "93%" in reckoning
    assert "PARITY, NOT SUCCESS" in reckoning
    assert "cohort cannot resolve it" in reckoning
    assert "never given a numeric definition" in reckoning
    assert "SUBSTANTIVE rather than formal" in reckoning
    quoted = "a positive would have contradicted a prediction this module"
    assert quoted in reckoning
    # The quoted pre-commitment really is the parked record's.
    parked_text = " ".join(str(synthesis.PARKED).split())
    assert "a positive would have contradicted" in parked_text
    assert "old prediction was wrong" in reckoning
    assert "[REASONED]" in reckoning
    assert "p17-a-vs-probe" in reckoning


def test_the_closing_walks_all_six_criteria():
    closing = phase17.PHASE_17_CLOSING
    assert "3d0ca56c" in closing["closed"]
    assert "single attempt" in closing["closed"]
    walked = [k for k in closing if k.startswith("criterion_")]
    assert len(walked) == 6
    for key in walked:
        assert "MET" in closing[key] or "the lock held" in closing[key], key

    one = closing["criterion_1_synthesis_artifacts"]
    assert "855c13fa" in one and "878626ee" in one
    assert "DELETE-AND-RELAUNCH" in one
    assert "the ruling was" in one

    two = closing["criterion_2_three_arms_zero_real"]
    for figure in ("0.2334", "0.0044", "-0.0044", "0.0317",
                   "-0.0185", "0.0586"):
        assert figure in two, figure
    # The deltas the rows quote are consistent with these means.
    assert round(0.2334 - 0.2520, 4) == -0.0186
    assert round(0.2334 - (-0.0185), 4) == 0.2519

    three = closing["criterion_3_five_contrasts"]
    for row_id in ("p17-a-vs-probe", "p17-b-vs-probe", "p17-c-vs-probe",
                   "p17-a-vs-c", "p17-b-vs-c"):
        assert row_id in three, row_id
    assert "CLAIMABLE" in three

    assert "0.333/0.502" in closing["criterion_4_registered_metrics"]
    assert "UNPREDICTED_PATTERN" in closing["criterion_5_readings_committed"]
    assert "five" in closing["criterion_6_locked"]

    defects = closing["build_cycle_defects"]
    assert "record-contract crash" in defects
    assert "faces.jsonl" in defects
    assert "_synth_index" in defects
    assert "Three instrument defects, zero data defects" in defects

    headline = closing["the_headline"]
    assert "93%" in headline
    assert "NOBODY predicted would succeed" in headline
    assert "Parity, not" in headline

    caveats = closing["caveats_carried_forward"]
    assert "measured irony" in caveats
    assert "NAMED CANDIDATE" in caveats
    assert "[REASONED] and not measured" in caveats
    assert "question 9" in caveats

    from cleft import results_ledger

    claimable = next(
        e for e in results_ledger.ENTRIES if e["id"] == "p17-b-vs-probe"
    )
    assert claimable["status"] == "CLAIMABLE"
