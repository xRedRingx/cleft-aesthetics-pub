"""The Phase 7C pre-registration, checked rather than trusted.

Same terms as ``test_phase7b.py``: the arms enumerate here, the strengths are
constants, and the comparisons are a function -- so the suite can check that
what runs is what was declared, and a markdown pre-registration cannot drift
into disagreeing with the code.

The sharpest check is the region-name one. The brief's protect-list named
``cupids_bow_peak``, which ``generators.py`` does not define; validating every
name against the generator is what would have caught it before a run failed
at lookup.
"""

from __future__ import annotations

import pytest

from cleft import phase7c


def test_every_protected_region_is_a_real_anatomy_region():
    """**The check that would have caught ``cupids_bow_peak``.**

    ``generators.py`` defines 9 midline and 9 bilateral regions -- 27 boxes.
    A protect-list naming something else fails at mask-build time, on the
    cluster, after the images are loaded. Validated here against the frozen
    generator instead.
    """
    from cleft.geometry.generators import BILATERAL_REGIONS, MIDLINE_REGIONS

    midline = {entry[0] for entry in MIDLINE_REGIONS}
    bilateral = {entry[0] for entry in BILATERAL_REGIONS}
    available = midline | bilateral

    assert len(midline) == 9 and len(bilateral) == 9
    # 9 midline + 9 mirrored pairs = the 27 the scheme is built on.
    assert len(midline) + 2 * len(bilateral) == 27

    unknown = [
        name for name in phase7c.PROTECTED_REGIONS if name not in available
    ]
    assert not unknown, (
        f"protected regions that do not exist: {unknown}. Available: "
        f"{sorted(available)}"
    )
    assert "cupids_bow_peak" not in available, (
        "the generator now defines cupids_bow_peak, so the brief's original "
        "name was right after all and this record needs revisiting"
    )

    # A strict, non-empty subset: protecting everything is not region-aware
    # augmentation, and protecting nothing makes arms 4 and 5 identical to 1
    # and 6.
    assert 0 < len(phase7c.PROTECTED_REGIONS) < len(available)


def test_the_protect_list_is_categories_not_the_examples_given():
    """**[R9, second instance] The instruction's parentheses were examples.**

    "nose (alar base asymmetry), upper lip (philtral column, Cupid's bow),
    vermillion border" names three CATEGORIES with illustrations, and reading
    the illustrations as an enumeration is the same error as the "27 regions"
    misreading (PLAN §4.5). So every nose and upper-lip region is protected,
    and the check is that the categories are covered EXHAUSTIVELY rather than
    that the named examples appear.
    """
    from cleft.geometry.generators import BILATERAL_REGIONS, MIDLINE_REGIONS

    available = {entry[0] for entry in MIDLINE_REGIONS} | {
        entry[0] for entry in BILATERAL_REGIONS
    }
    protected = set(phase7c.PROTECTED_REGIONS)

    # Every region whose name is nasal or labial anatomy, not just the ones
    # the instruction happened to name.
    nose = {n for n in available if n.startswith(("nasal_", "alar_", "nostril_"))}
    nose |= {"columella", "subnasale"}
    lip = {"philtrum", "philtral_column", "labial_tubercle", "vermillion_border",
           "commissure"}

    assert nose <= protected, f"unprotected nose regions: {sorted(nose - protected)}"
    assert lip <= protected, f"unprotected lip regions: {sorted(lip - protected)}"

    # And what is left augmentable is the forehead, the eyes and the cheeks --
    # which is what "cheeks, forehead, background and periphery" means.
    assert available - protected == {"glabella", "medial_canthus", "lateral_orbit"}

    record = phase7c.PROTECT_LIST_IS_CATEGORIES_NOT_EXAMPLES
    assert record["now"] == len(protected) > record["was"]
    assert record["rule"].startswith("PLAN R9")


def test_the_protected_box_count_is_visible_arithmetic():
    """9 midline + 9 mirrored pairs = 27 boxes; the protected set is a
    specific count of them, derived rather than stated, so a name moving
    between the midline and bilateral lists shows up here."""
    from cleft.geometry.generators import BILATERAL_REGIONS, MIDLINE_REGIONS

    midline = {entry[0] for entry in MIDLINE_REGIONS}
    bilateral = {entry[0] for entry in BILATERAL_REGIONS}
    protected = set(phase7c.PROTECTED_REGIONS)

    boxes = len(protected & midline) + 2 * len(protected & bilateral)
    total = len(midline) + 2 * len(bilateral)
    assert total == 27
    assert boxes == 8 + 2 * 7 == 22, (
        f"{boxes} of {total} boxes protected; the split between midline and "
        "bilateral has changed"
    )


def test_the_seven_arms_are_distinct_and_ordered():
    arms = phase7c.arms()
    assert len(arms) == 7
    assert [arm["index"] for arm in arms] == list(range(7))
    assert len({arm["name"] for arm in arms}) == 7

    factors = {
        tuple(arm[field] for field in phase7c.COMPARED_FIELDS) for arm in arms
    }
    assert len(factors) == 7, "two arms declare the same policy"

    # Arm 0 is the identity gate: no family on at all.
    identity = arms[0]
    assert not any(identity[field] for field in phase7c.COMPARED_FIELDS)
    assert identity["name"] == "p7c_0_identity"

    # Every arm sits on the baseline's cell and its seeds.
    for arm in arms:
        for field in ("backbone", "init", "geometry", "label"):
            assert arm[field] == phase7c.BASELINE[field]
        assert arm["seeds"] == phase7c.SEEDS
        assert len(arm["seeds"]) == phase7c.BASELINE["seeds"] == 5


def test_every_comparison_varies_exactly_one_field():
    """**The property the phase rests on**, asserted over the derived arms
    rather than trusted to the list having been written carefully -- a pair
    differing in two fields measures neither, and nothing downstream says so.

    ``comparisons()`` raises on construction, so this also checks the raise
    is reachable rather than decorative.
    """
    checks = phase7c.comparisons()
    assert checks, "the phase declares no comparisons, so this checks nothing"

    for check in checks:
        differing = {
            field for field in phase7c.COMPARED_FIELDS
            if check["a"][field] != check["b"][field]
        }
        assert differing == {check["varies"]}
        assert check["a"][check["varies"]] != check["b"][check["varies"]]

    by_question = {check["question"] for check in checks}
    assert "rotation" in by_question
    assert "region_awareness" in by_question


def test_the_one_factor_check_has_teeth(monkeypatch):
    """It is silent on a correct arm list, so the raise is exercised against
    a pair that genuinely differs in two."""
    broken = list(phase7c._ARM_FACTORS)
    # Give arm 2 photometric as well, so arms 2 and 3 differ in two factors.
    broken[2] = ("geometric", True, True, False, False)
    monkeypatch.setattr(phase7c, "_ARM_FACTORS", tuple(broken))

    with pytest.raises(phase7c.Phase7CError, match="claims to vary"):
        phase7c.comparisons()


def test_rotation_cannot_be_declared_without_the_family(monkeypatch):
    """Rotation is a geometric transform. An arm carrying it without the
    family would be declaring a policy the augmenter cannot build."""
    broken = list(phase7c._ARM_FACTORS)
    broken[2] = ("geometric", False, False, True, False)
    monkeypatch.setattr(phase7c, "_ARM_FACTORS", tuple(broken))

    with pytest.raises(phase7c.Phase7CError, match="rotation without geometric"):
        phase7c.arms()


def test_the_region_pair_holds_rotation_off_on_both_sides():
    """**Arms 5 and 6 answer the supervision design question**, so they must
    differ in region-awareness alone -- and both carry NO rotation, because
    rotation is unresolved until arms 2 and 3 report and baking a possibly
    harmful transform into both sides asks the region question at a degraded
    operating point. The Stage G1 lesson: an instrument is chosen for its
    power."""
    by_name = {arm["name"]: arm for arm in phase7c.arms()}
    five = by_name["p7c_5_region_full"]
    six = by_name["p7c_6_whole_image_full"]

    assert five["rotation"] is False and six["rotation"] is False
    assert five["photometric"] and five["geometric"]
    assert six["photometric"] and six["geometric"]
    assert five["region_aware"] and not six["region_aware"]
    assert phase7c.ARMS_5_AND_6_EXCLUDE_ROTATION["precedent"]


def test_this_phase_declares_that_it_is_not_a_search():
    """**The protocol difference from 7B, and the reason it is legitimate.**

    7B needed a budget, a selection rule and an exhaustion criterion because
    it ranked 24 configurations and reported the best. This ranks nothing, so
    there is no selection step to bias -- and the absence must be DECLARED
    rather than merely true, or a reader cannot tell it from an omission.
    """
    record = phase7c.NOT_A_SEARCH
    assert record["budget"] is None
    assert record["selection_rule"] is None
    assert record["exhaustion_criterion"] is None
    assert phase7c.summary()["is_a_search"] is False

    # The module must not grow the machinery it says it does not have.
    for absent in ("select", "is_exhausted", "SEARCH_BUDGET", "configurations"):
        assert not hasattr(phase7c, absent), (
            f"phase7c gained {absent!r}; if it now selects, NOT_A_SEARCH is "
            "no longer true and the protocol needs rewriting"
        )

    # And the reporting discipline that DOES carry over is named.
    assert any("every arm reported" in item for item in record["carried_over"])
    assert any("§4.12.1" in item for item in record["carried_over"])


def test_the_baseline_is_the_one_phase_7b_tuned_not_a_copy():
    """Two records of one baseline is two places for it to drift -- this
    project already corrected a five-seed mean paired with a ten-seed SD
    once."""
    from cleft import ladder, phase7b

    assert phase7c.BASELINE is phase7b.BASELINE

    record = ladder.STAGE_D1_AT_G1
    backbone, init = phase7c.BASELINE["backbone"], phase7c.BASELINE["init"]
    assert phase7c.BASELINE["pcc"] == pytest.approx(
        record["cells"][backbone][record["inits"].index(init)], abs=5e-5
    )
    assert phase7c.BASELINE["sd"] == pytest.approx(
        record["sd"][backbone][init], abs=5e-5
    )


def test_arm_0_is_scoped_to_the_new_code_not_a_settled_question():
    """**Live-versus-artifact is settled at this exact cell** -- 0.25206 live
    against 0.2520 artifact, four decimals on mean and SD. Arm 0 tests the new
    augmenting backbone at identity policy and nothing more, and the record
    says so, so nobody re-litigates it."""
    from cleft import ladder

    scope = phase7c.ARM_0_SCOPE
    assert scope["is_a_gate"] is True
    assert "does_not_test" in scope and "settled" in scope["does_not_test"]

    live = ladder.CLEAN_GEOMETRY_MEASUREMENTS["imagenet"]["g1"]
    artifact_pcc = ladder.STAGE_D1_AT_G1["cells"]["vit_b16"][0]
    assert live["mean"] == pytest.approx(artifact_pcc, abs=1e-4), (
        "the live and artifact numbers no longer agree, so arm 0's scope note "
        "is describing a question that is no longer settled"
    )
    assert live["sd"] == pytest.approx(
        ladder.STAGE_D1_AT_G1["sd"]["vit_b16"]["imagenet"], abs=1e-4
    )


def test_augmentation_enters_through_train_epoch_and_never_predict():
    """**The never-augment-evaluation trap becomes structural.**

    Evaluation reaches the model only through ``predict``; ``predict`` does
    not augment; therefore validation and test images cannot be augmented. A
    property of the call graph rather than a rule to observe -- and the
    harness that guarantees it is the frozen one, so gate 1 still tests the
    loop it was measured on.
    """
    from cleft.train.harness import Backbone

    assert phase7c.LIVE_PATH["augment_in"] == "train_epoch"
    assert phase7c.LIVE_PATH["never_in"] == "predict"
    assert "CANNOT" in phase7c.LIVE_PATH["structural_guarantee"]

    # The protocol the augmenting backbone must satisfy really has those two
    # methods, or the plan is describing an interface that does not exist.
    for method in ("reset", "train_epoch", "predict"):
        assert hasattr(Backbone, method)

    # And the route that hands a model raw images is really there.
    source = __import__("inspect").getsource(
        __import__("cleft.train.phase3", fromlist=["prepare_features"]).prepare_features
    )
    assert 'trainable == "full"' in source and "raw_images" in source, (
        "prepare_features no longer returns raw images for a model that does "
        "its own forward; the live path needs rerouting"
    )


def test_horizontal_flip_is_excluded_by_a_failed_condition_not_a_preference():
    """The distinction matters: a preference invites revisiting and a failed
    condition does not. The condition was "only if labels are symmetric", and
    laterality is not recorded anywhere in this cohort."""
    record = phase7c.HORIZONTAL_FLIP
    assert record["included"] is False
    assert "symmetric" in record["condition_given"]
    assert "laterality is not recorded" in record["condition_fails_because"]
    assert "not preference" in record["excluded_by"]

    for arm in phase7c.arms():
        assert "flip" not in arm["name"]


def test_the_strengths_are_single_pre_registered_settings():
    """One setting each, not a sweep -- a swept strength would reintroduce a
    selection step, which is the thing this phase's protocol rests on not
    having."""
    assert phase7c.PHOTOMETRIC["single_setting"] is True
    for channel in ("brightness", "contrast", "saturation"):
        assert phase7c.PHOTOMETRIC[channel] == 0.2
    assert phase7c.PHOTOMETRIC["hue"] == 0.05

    assert phase7c.GEOMETRIC["rotation_degrees"] == 5.0
    assert phase7c.GEOMETRIC["translate_fraction"] == 0.10
    assert phase7c.GEOMETRIC["scale"] == (0.9, 1.1)

    assert phase7c.REGION_AWARE["outside_strength"] == 1.0
    assert phase7c.REGION_AWARE["inside_strength"] == 0.25
    assert 0 < phase7c.REGION_AWARE["inside_strength"] < 1


def test_the_boundary_is_a_soft_falloff_scaled_to_the_crop():
    """A hard seam sits at a fixed anatomical offset, on the regions the grade
    is about, and appears in every augmented training image and no evaluation
    image -- a feature a model can key on. And the width is relative because
    crop aspect ratios span a factor of two."""
    blend = phase7c.BLEND
    assert blend["kind"] == "soft_falloff"
    assert "key on" in blend["why_not_hard"]
    assert 0 < blend["sigma_fraction_of_crop_width"] < 0.1
    # The width must be expressed RELATIVE to the crop, not in pixels -- the
    # key name carries that, so it is asserted rather than assumed.
    assert any("fraction_of_crop" in key for key in blend), (
        "the falloff width is no longer crop-relative, so it is a different "
        "blend on each patient"
    )
    assert not any(key.endswith("_px") or key.endswith("_pixels") for key in blend)


def test_the_unsourced_fairness_figures_are_recorded_as_struck():
    """Same shape as the fabricated ORIGINAL-folder count: a measured-sounding
    figure with no source, in a section arguing a motivation. Struck rather
    than silently dropped, and the motivation restated qualitatively."""
    record = phase7c.PHOTOMETRIC
    assert "unsourced" in record["struck"]
    assert "3.6" in record["struck"], "the struck figure must be named to be findable"
    assert "brightness-linked failure pattern" in record["fairness_rationale"]
    assert "3.6" not in record["fairness_rationale"]


def test_the_determinism_rule_reports_rather_than_disables():
    """Rotation interpolation is a kernel gate 1 has never covered. If it
    raises under ``use_deterministic_algorithms(True)`` that is a finding --
    the AG-Net roi_align rule: an opt-out is per-arm, declared, and refused if
    undeclared. Never global, never silent."""
    rule = phase7c.ROTATION_IS_AN_ARM["determinism"]
    assert "FINDING" in rule
    assert "not a reason to disable" in rule
    assert "per-arm" in rule and "declared" in rule
    assert "Never" in rule and "silent" in rule


def test_the_shipped_configs_match_the_derived_arms(repo_root):
    """**The pre-registration is the source; the configs are output.** A hand
    edit would break the one-factor property invisibly, because the
    comparisons read the arm list and not the files."""
    import os
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable,
         str(repo_root / "scripts" / "generate_phase7c_configs.py"), "--check"],
        capture_output=True, text=True, cwd=str(repo_root),
        env={**os.environ, "PYTHONPATH": str(repo_root / "src")},
    )
    assert result.returncode == 0, (
        f"the shipped Phase 7C configs differ from the derived arms:\n"
        f"{result.stdout}{result.stderr}"
    )


#: Phase 7C configs that are NOT arms. Listed explicitly so adding one is a
#: deliberate act: a stray ``p7c_*`` config that is neither an arm nor a
#: declared gate would otherwise sit in the directory looking like part of the
#: phase, and could be launched as one. Same reason ``NON_ARM_P7_CONFIGS``
#: exists for the ladder.
NON_ARM_P7C_CONFIGS = {
    # Exit criterion 5, and the gate arm 0 waits on.
    "p7c_contact_sheet",
    # Exit criterion 6, and it fits nothing: the per-seed paired BCa over
    # vectors the arms already wrote. phase7c.PAIRED_BCA_IS_THE_MISSING_CONDITION.
    "p7c_paired_selected30",
}


def test_every_arm_has_a_config_and_nothing_else_does(repo_root):
    """**Both directions, across all four rounds.**

    [CORRECTED 2026-08-03] This filtered ``p7c_m3_`` out by prefix and compared
    what was left against the five-seed arm list. That worked while there were
    two rounds and silently stopped covering the ten-seed ones the moment there
    were four -- a stray ``p7c_s10_2_geometric`` would have sat in the
    directory looking launchable. The expected set is now derived from
    ``phase7c.CONFIG_ROUNDS``, so adding a round extends the check instead of
    escaping it.
    """
    shipped = {
        p.stem for p in (repo_root / "configs").glob("p7c_*.yaml")
    } - NON_ARM_P7C_CONFIGS
    derived = phase7c.config_stems()
    assert shipped == derived, (
        f"shipped-but-not-derived: {sorted(shipped - derived)}; "
        f"derived-but-not-shipped: {sorted(derived - shipped)}"
    )
    # Every declared non-arm must exist, or the exemption is excusing a config
    # that is not there.
    for stem in NON_ARM_P7C_CONFIGS:
        assert (repo_root / "configs" / f"{stem}.yaml").is_file()


def test_each_config_declares_its_arms_policy_and_no_strengths(
    repo_root, monkeypatch
):
    """The flags come from the arm; the strengths must NOT be in the file --
    a config that could set them would make the pre-registration editable."""
    from cleft.config import load_config

    monkeypatch.setenv("CLEFT_SCUT_ROOT", "/nonexistent/sentinel")
    for arm in phase7c.arms():
        cfg = load_config(repo_root / "configs" / f"{arm['name']}.yaml")
        task = cfg["task"]

        assert task["augmentation"] == {
            field: arm[field] for field in phase7c.COMPARED_FIELDS
        }
        assert task["seeds"] == arm["seeds"]
        assert task["geometry"] == arm["geometry"]
        assert task["backbone"] == arm["backbone"]
        # Frozen backbone, 769-parameter head -- unchanged by augmenting.
        assert task["trainable"] == "head"
        # Features are recomputed from pixels, so there is no stored artifact.
        assert not task.get("embeddings_artifact")

        text = (repo_root / "configs" / f"{arm['name']}.yaml").read_text(
            encoding="utf-8"
        )
        for strength in ("0.25", "brightness", "sigma", "rotation_degrees"):
            assert f"{strength}:" not in text, (
                f"{arm['name']} sets {strength} -- strengths are "
                "pre-registered in cleft.phase7c, not per config"
            )


def test_a_config_cannot_declare_a_policy_the_augmenter_cannot_build():
    """Refused at LOAD time, so it costs nothing on the cluster: an unknown
    flag would otherwise leave the arm unaugmented while its record said
    otherwise, and rotation alone declares something with no meaning."""
    from cleft.config.schema import ConfigError, validate

    def build(augmentation, **overrides):
        task = {
            "kind": "train_cv", "manifest_artifact": "m", "staged_artifact": "s",
            "geometry": "g1", "label": "mean", "backbone": "vit_b16",
            "max_epochs": 40, "patience": 5, "inner_val_frac": 0.2,
            "monitor": "inner_val_mse", "augmentation": augmentation,
        }
        task.update(overrides)
        # [2026-09-01] The scaffold declares the placeholder inputs its
        # task names -- schema.validate now checks the reference, not
        # just the field's type.
        return {"schema_version": 1, "phase": "p7c", "tier": "dev",
                "seed": 1337,
                "inputs": [
                    {"name": n, "path": f"/tmp/{n}",
                     "rollup_sha256": "ab" * 32} for n in ("m", "s")
                ], "task": task}

    with pytest.raises(ConfigError, match="unknown key"):
        validate(build({"photmetric": True}))
    with pytest.raises(ConfigError, match="rotation without geometric"):
        validate(build({"rotation": True}))
    with pytest.raises(ConfigError, match="booleans"):
        validate(build({"photometric": 0.2}))
    with pytest.raises(ConfigError, match="trainable"):
        validate(build({"photometric": True}, trainable="full"))

    # Every shipped policy validates.
    for arm in phase7c.arms():
        policy = {field: arm[field] for field in phase7c.COMPARED_FIELDS}
        assert validate(build(policy))["task"]["augmentation"] == policy


def test_early_stopping_cannot_fire_in_any_arm(repo_root, monkeypatch):
    """**[MEASURED 2026-08-03] The first run stopped at epoch 1 everywhere**,
    so the model saw each image once under one random transform -- which is
    not augmentation, and produced a plausible monotone ordering measuring
    something else.

    The fix is arithmetic, not a flag: ``harness.run_fold`` breaks when
    ``since_improvement >= patience``, and ``since_improvement`` can reach at
    most ``max_epochs - 1``. So ``patience >= max_epochs`` makes the break
    unreachable, and what remains is a fixed budget with the best-checkpoint
    selection the harness already does.
    """
    from cleft.config import load_config

    monkeypatch.setenv("CLEFT_SCUT_ROOT", "/nonexistent/sentinel")
    policy = phase7c.EPOCH_POLICY
    assert policy["patience"] >= policy["max_epochs"], (
        "patience is below the budget, so early stopping can fire and the "
        "arms may run for different numbers of epochs"
    )
    assert policy["early_stopping_can_fire"] is False
    assert policy["max_epochs"] >= 30, (
        "fewer than 30 epochs is a thin sample of the transform distribution"
    )

    seen = set()
    for arm in phase7c.arms():
        task = load_config(repo_root / "configs" / f"{arm['name']}.yaml")["task"]
        assert task["patience"] >= task["max_epochs"]
        assert task["max_epochs"] == policy["max_epochs"]
        assert task["monitor"] == policy["monitor"]
        seen.add((task["max_epochs"], task["patience"], task["monitor"]))

    # Identical across all seven, so convergence speed cannot confound the
    # comparison -- an arm running longer because it kept improving would
    # differ from its neighbour in epochs as well as in policy.
    assert len(seen) == 1, f"the arms do not share an epoch policy: {seen}"


def test_the_one_epoch_defect_is_recorded_with_what_made_it_invisible():
    """The tenth instance, and the first where the arms' own numbers said
    nothing: a plausible monotone ordering, no anomalous quantity, and the
    only tell in a bookkeeping field nobody compares between arms."""
    record = phase7c.ONE_EPOCH_IS_NOT_AUGMENTATION
    assert "selected_epochs" in record["symptom"]
    assert "monotone" in record["why_it_looked_right"]
    assert "bookkeeping" in record["found_by"]
    assert record["arms_are_void"].startswith("not superseded")

    # And the 7B immunity, which is the opposite of the usual pattern.
    assert "closed form" in record["phase_7b_immune_by_accident"]

    from cleft import phase7b

    assert "closed form" in phase7b.SEARCH_HEAD_PROCEDURE["fit"], (
        "7B is no longer closed form, so it is no longer immune and its "
        "epoch counts need checking too"
    )


def test_the_monitor_deviation_from_the_amendment_is_declared():
    """Gate 4 specifies ``inner_val_pcc``. Its REASON -- patience firing on
    noise in a flat region -- cannot apply when patience cannot fire, and
    keeping ``inner_val_mse`` is what lets arm 0 still reproduce 0.2520. A
    deliberate deviation, so it is stated rather than left to be noticed."""
    policy = phase7c.EPOCH_POLICY
    assert policy["monitor"] == "inner_val_mse"
    assert "cannot apply when patience cannot fire" in (
        policy["monitor_deviates_from_the_amendment"]
    )
    assert "0.2520" in policy["monitor_deviates_from_the_amendment"]

    # The gate survives the budget change, which is the reason for the choice.
    assert phase7c.ARM_0_SCOPE["is_a_gate"] is True


PAIRS = {
    "rotation": ("2_geometric", "3_geometric_rotation"),
    "region_awareness": ("6_whole_image_full", "5_region_full"),
    "region_awareness_photometric": ("1_photometric", "4_region_photometric"),
}


@pytest.mark.parametrize("round_name", ["selected30", "matched3"])
def test_every_delta_is_recomputed_from_the_two_arms_own_sds(round_name):
    """**Every verdict in BOTH rounds re-derived**, as everywhere else in this
    project: the threshold wants the two arms' own spreads, and "claimable" is
    a comparison rather than a property of a number.

    Parameterised rather than written twice, so a round added later cannot be
    recorded without its verdicts being checked.
    """
    from cleft.train.phase3 import combined_claimable_delta

    entry = phase7c.STAGE_7C_RESULTS["rounds"][round_name]
    arms = entry["arms"]
    seeds = len(phase7c.SEEDS)
    identity = arms["0_identity"]

    assert set(arms) == {a["name"].replace("p7c_", "") for a in phase7c.arms()}
    # The gate: arm 0 IS the baseline, so "against identity" and "against
    # 0.2520" are one comparison -- and it must hold in BOTH rounds, which is
    # what makes them comparable at all.
    assert identity["pcc"] == pytest.approx(phase7c.BASELINE["pcc"], abs=5e-4)
    assert identity["sd"] == pytest.approx(phase7c.BASELINE["sd"], abs=5e-4)
    assert identity["epochs"] == [1] * seeds

    for name, verdict in entry["against_identity"].items():
        arm = arms[name]
        delta = arm["pcc"] - identity["pcc"]
        assert verdict["delta"] == pytest.approx(delta, abs=5e-5), name
        threshold = combined_claimable_delta(
            identity["sd"], seeds, arm["sd"], seeds
        )
        assert verdict["threshold"] == pytest.approx(
            threshold["arm_means_95"], abs=5e-5
        ), name
        assert verdict["claimable"] == (abs(delta) > threshold["arm_means_95"])
        assert delta < 0, f"{name} is not below identity; the verdict is stale"
        if "survives_single_run_95" in verdict:
            assert verdict["survives_single_run_95"] == (
                abs(delta) > threshold["single_run_95"]
            )
    assert len(entry["against_identity"]) == len(phase7c.arms()) - 1

    # And the recorded survivor COUNT, which is the quantity that doubled.
    survivors = sum(
        1 for v in entry["against_identity"].values()
        if v.get("survives_single_run_95")
    )
    assert survivors == phase7c.STAGE_7C_RESULTS["single_run_95_survivors"][
        round_name
    ]


@pytest.mark.parametrize("round_name", ["selected30", "matched3"])
def test_the_declared_comparisons_are_recomputed_and_null_in_both_rounds(
    round_name,
):
    """Rotation and both region tests. None is claimable in either round, and
    the record must say so from its own numbers."""
    from cleft.train.phase3 import combined_claimable_delta

    entry = phase7c.STAGE_7C_RESULTS["rounds"][round_name]
    arms = entry["arms"]
    seeds = len(phase7c.SEEDS)

    # The pairs must be the comparisons the pre-registration declares, in the
    # order it declares them -- the delta's SIGN depends on that ordering.
    assert set(PAIRS) == {c["question"] for c in phase7c.comparisons()}
    for check in phase7c.comparisons():
        a, b = PAIRS[check["question"]]
        assert check["a"]["name"] == f"p7c_{a}"
        assert check["b"]["name"] == f"p7c_{b}"

    for question, (a, b) in PAIRS.items():
        verdict = entry["comparisons"][question]
        delta = arms[b]["pcc"] - arms[a]["pcc"]
        assert verdict["delta"] == pytest.approx(delta, abs=5e-5), question
        threshold = combined_claimable_delta(
            arms[a]["sd"], seeds, arms[b]["sd"], seeds
        )
        assert verdict["threshold"] == pytest.approx(
            threshold["arm_means_95"], abs=5e-5
        ), question
        assert verdict["claimable"] == (abs(delta) > threshold["arm_means_95"])
        assert not verdict["claimable"], f"{question} is now claimable"


def test_the_region_tests_agree_at_the_matched_budget_and_not_at_the_long_one():
    """**[CORRECTED] The sign flip is between ROUNDS, and it is easy to quote
    from the wrong one.**

    Under the pre-registration's own convention both region deltas are
    PROTECTED minus UNPROTECTED. At the long budget they point in opposite
    directions; at the matched budget both point weakly against protection,
    one of them four decimal places from identical. Neither is claimable
    anywhere, so this changes the framing and not the verdict.
    """
    record = phase7c.REGION_TESTS_AGREE_AT_THE_MATCHED_BUDGET
    rounds = phase7c.STAGE_7C_RESULTS["rounds"]

    for round_name in ("selected30", "matched3"):
        comparisons = rounds[round_name]["comparisons"]
        for question in ("region_awareness", "region_awareness_photometric"):
            assert record[round_name][question] == pytest.approx(
                comparisons[question]["delta"], abs=5e-5
            )

    def same_direction(entry):
        return (entry["region_awareness"] <= 0) == (
            entry["region_awareness_photometric"] <= 0
        )

    assert not same_direction(record["selected30"])
    assert same_direction(record["matched3"])
    assert record["selected30"]["direction"] == "opposite"
    assert record["matched3"]["direction"].startswith("same")
    assert "not the finding" in record["verdict_unchanged"]


def test_round_2_is_round_1_truncated_not_a_replication():
    """**[MEASURED] The second round re-selects over the first round's own
    trajectories.**

    ``rng_for(seed, fold, epoch)`` derives views from the coordinates alone,
    so epoch n's fit does not depend on ``max_epochs``. Round 2 changes only
    the candidate set the best-checkpoint argmin runs over. Every fold where
    round 1 already selected epoch <= 3 is therefore the SAME fit in both
    rounds -- which is asserted here from the epoch patterns, because it is
    the difference between "settled from two directions" and "one direction,
    counted twice".
    """
    from cleft.train.augment import rng_for

    record = phase7c.ROUND_2_IS_ROUND_1_TRUNCATED
    rounds = phase7c.STAGE_7C_RESULTS["rounds"]

    # The property the whole finding rests on: the stream is a function of
    # its coordinates, so it cannot know the budget.
    assert (rng_for(1337, 0, 0).random(4) == rng_for(1337, 0, 0).random(4)).all()

    identical = total = 0
    for name, arm in rounds["selected30"]["arms"].items():
        long_epochs = arm["epochs"]
        short_epochs = rounds["matched3"]["arms"][name]["epochs"]
        total += len(long_epochs)
        for i, epoch in enumerate(long_epochs):
            if epoch <= phase7c.MATCHED_EPOCH_POLICY["max_epochs"]:
                identical += 1
                assert short_epochs[i] == epoch, (
                    f"{name} fold {i}: round 1 selected epoch {epoch}, which "
                    f"is inside round 2's window, so round 2 must select it "
                    f"too -- it selected {short_epochs[i]}. Either the "
                    "trajectories differ between rounds or a number is wrong"
                )
            else:
                assert short_epochs[i] <= phase7c.MATCHED_EPOCH_POLICY[
                    "max_epochs"
                ]

    assert identical == record["identical_fits"] == 23
    assert total == record["total_fits"] == 35
    assert identical > total / 2, (
        "fewer than half the fits are shared, so the rounds are closer to "
        "independent than this record claims"
    )
    assert "after round 1's verdicts were known" in record["window_chosen_post_hoc"]


def test_the_pre_registered_disagreement_rule_decides_photometric():
    """**The pre-registration pre-committed to this exact case.**

    ``MATCHED_EPOCH_POLICY``: *if they disagree, the disagreement is the
    result and the selection procedure is the factor*. They disagreed on
    photometric, so photometric is UNRESOLVED -- taking the answer from the
    window that gave the preferred one is what the rule exists to prevent.
    """
    record = phase7c.STAGE_7C_RESULTS
    families = record["per_family"]

    # **[UPDATED 2026-08-04] The verdict this test was written about is now
    # `verdict_before_condition_1`.** The paired BCa withdrew all nine, so the
    # top-level field points at that outcome instead. The rule below is still
    # worth asserting: it decided photometric correctly on the evidence
    # available at the time, and would decide the same way again.
    assert families["photometric"]["verdict"] == "UNRESOLVED"
    assert "SUPERSEDED" in record["verdict"]
    before = record["verdict_before_condition_1"]
    assert "UNRESOLVED" in before
    assert "NEUTRAL" not in before.upper().replace("UNRESOLVED", "")

    # The rule the verdict follows must still be in the policy it came from.
    rule = phase7c.MATCHED_EPOCH_POLICY
    assert rule["both_rounds_reported"] is True
    assert "disagreement is the result" in (
        phase7c.ROUND_2_IS_ROUND_1_TRUNCATED["pre_registered_rule"]
    )

    # And the disagreement is real: claimable in one window, not the other.
    long_v = record["rounds"]["selected30"]["against_identity"]["1_photometric"]
    short_v = record["rounds"]["matched3"]["against_identity"]["1_photometric"]
    assert long_v["claimable"] != short_v["claimable"]

    # The families that AGREE across windows keep their verdicts.
    for arm in ("2_geometric", "3_geometric_rotation", "5_region_full",
                "6_whole_image_full"):
        assert (
            record["rounds"]["selected30"]["against_identity"][arm]["claimable"]
            is record["rounds"]["matched3"]["against_identity"][arm]["claimable"]
            is True
        )
    # [UPDATED 2026-08-04] Both rounds agree on the geometric arms under
    # condition 2, which is what this loop checks and what it should keep
    # checking. The FAMILY verdict is withdrawn -- condition 1 fails for every
    # one of them -- so the recorded verdict names the withdrawal and the
    # condition-2 agreement it superseded.
    geometric = families["geometric"]
    assert geometric["verdict"].startswith("WITHDRAWN")
    assert "claimably worse than identity" in geometric["verdict"]
    assert "condition 1" in geometric["withdrawn_by"]


def test_the_confounds_predicted_direction_is_recorded_as_unsupported():
    """**The justification for the second round, measured and refused.**

    If late selection systematically degraded the reported OOF, truncating
    would raise every augmented arm and most where selection was latest.
    Neither holds -- so the confound is coherent as a mechanism and
    unestablished as a magnitude, which is why the truncated round is a
    sensitivity analysis rather than a correction.
    """
    record = phase7c.SELECTION_CONFLATES_DRAW_QUALITY["direction_measured"]
    moves = phase7c.STAGE_7C_RESULTS["between_round_moves"]
    rounds = phase7c.STAGE_7C_RESULTS["rounds"]

    assert record["mean_move"] == pytest.approx(
        sum(moves.values()) / len(moves), abs=5e-4
    )
    assert "NOT SUPPORTED" in record["verdict"]

    def late(name):
        return sum(1 for e in rounds["selected30"]["arms"][name]["epochs"] if e > 3)

    latest = max(moves, key=late)
    assert latest == record["most_late_selecting_arm"]
    assert moves[latest] == pytest.approx(record["its_move"], abs=5e-5)
    assert moves[latest] < 0, (
        "the most late-selecting arm no longer moves against the prediction, "
        "so the confound's direction may be supported after all"
    )


def test_the_mechanism_claim_stays_withdrawn():
    """**[WITHDRAWN] The augmented optimum DOES arrive later than identity's,
    which is the opposite of what the claim asserted** -- and the test that
    guarded it asserted a different quantity and passed.

    It checked "most folds select epoch <= 3" (18 of 30, true) and read that
    as "the distribution has not shifted". A majority being early and the mean
    not moving are different quantities; the mean moved from 1.0 to 7.13.

    This asserts the refutation from the recorded epochs, so the claim cannot
    be revived without the data changing.
    """
    import numpy as np

    record = phase7c.MECHANISM_CLAIM_WITHDRAWN
    arms = phase7c.STAGE_7C_RESULTS["rounds"]["selected30"]["arms"]

    identity = np.mean(arms["0_identity"]["epochs"])
    augmented = [
        e for name, arm in arms.items() if name != "0_identity"
        for e in arm["epochs"]
    ]
    assert identity == pytest.approx(record["refuted_by"][
        "identity_mean_selected_epoch"
    ])
    assert np.mean(augmented) == pytest.approx(
        record["refuted_by"]["augmented_mean_selected_epoch"], abs=5e-3
    )
    assert np.mean(augmented) > identity, (
        "the augmented optimum no longer arrives later than identity's, so "
        "the withdrawn mechanism claim may be revivable -- re-read it"
    )
    # Every augmented arm, not just the pooled mean.
    for name, arm in arms.items():
        if name == "0_identity":
            continue
        assert np.mean(arm["epochs"]) > identity, name

    assert phase7c.MECHANISM_NEEDS_THE_LONG_BUDGET["status"].startswith("WITHDRAWN")
    assert "not established by this phase" in record["honest_position"]

    # The by-construction half still stands: the short round's budget forbids
    # the observation that would settle it either way.
    short = phase7c.STAGE_7C_RESULTS["rounds"]["matched3"]["arms"]
    assert max(
        e for arm in short.values() for e in arm["epochs"]
    ) <= phase7c.MATCHED_EPOCH_POLICY["max_epochs"]
    assert "by construction" in (
        phase7c.MECHANISM_NEEDS_THE_LONG_BUDGET["matched3_cannot"]
    )


def test_a_selected_epoch_1_is_distinguished_from_a_forced_one():
    """**The same fit, byte for byte, with opposite evidential status.**

    v1 was voided because the loop STOPPED at epoch 1 -- the procedure never
    ran. In the primary round thirty epochs ran and inner-val SELECTED epoch 1
    as best of thirty, which is a measurement that more views made that fold
    worse. Nothing in the number says which, and a careful reviewer read the
    primary round as "largely the configuration the phase voided".
    """
    record = phase7c.EPOCH_1_SELECTED_IS_NOT_EPOCH_1_FORCED
    rounds = phase7c.STAGE_7C_RESULTS["rounds"]

    for round_name, expected in (("selected30", 12), ("matched3", 19)):
        arms = rounds[round_name]["arms"]
        ones = sum(
            1 for name, arm in arms.items() if name != "0_identity"
            for e in arm["epochs"] if e == 1
        )
        assert ones == expected, f"{round_name}: {ones} epoch-1 folds"
        assert str(expected) in record["counts"][round_name]

    # The sensitivity round has MORE of them, which is a further reason it is
    # not the primary: a shorter window has fewer epochs to prefer over 1.
    assert record["counts"]["matched3"] > record["counts"]["selected30"]

    # And the primary round genuinely had the choice -- its budget is long
    # enough that epoch 1 is a selection rather than a stopping point.
    assert phase7c.EPOCH_POLICY["max_epochs"] >= 30
    assert phase7c.EPOCH_POLICY["early_stopping_can_fire"] is False
    assert "never happened" in record["v1_was_void_because"]
    assert "SELECTED" in record["selected30_is_not_void_because"]

    # The rationale tension is recorded as unresolved, not resolved by fiat.
    assert "The phase asserts both" in record["unresolved_rationale_tension"]


def test_the_region_finding_is_no_answer_rather_than_no_difference():
    """**Two tests that are not independent, on a small intervention in a
    diluted contrast.** "Null on both" reads as corroboration and is not --
    and one of them flips to claimable at ten seeds.

    Telling a room their design makes no difference, on that evidence, is both
    wrong and costly. The reportable wording is round 1's own.
    """
    record = phase7c.REGION_TESTS_ARE_NOT_INDEPENDENT
    families = phase7c.STAGE_7C_RESULTS["per_family"]

    assert families["region_aware"]["verdict"] == "NO ANSWER, in either direction"
    # **[UPDATED 2026-08-04] "No answer" is now the wording for the WHOLE
    # phase**, not just this family -- the paired BCa withdrew all nine. The
    # prohibition holds on both the current verdict and the superseded one:
    # "makes no difference" was never sayable and still is not.
    results = phase7c.STAGE_7C_RESULTS
    assert "NO ANSWER" in results["verdict_before_condition_1"]
    for text in (results["verdict"], results["verdict_before_condition_1"]):
        assert "no difference" not in text
    assert "Nothing in this phase is claimable" in results["verdict"]

    # The three reasons must each be recorded, since each alone would be
    # enough to stop "makes no difference" being said.
    assert "fourfold reduction" in record["intervention_is_small"]
    assert "white pad" in record["contrast_is_diluted"]
    assert "cannot be spatially modulated" in record["geometry_is_not_modulated"]
    assert "NOT 'region protection makes no difference'" in (
        record["reportable_wording"]
    )

    # The dilution is the phase's OWN earlier finding, so the cross-reference
    # must still point at something real.
    from cleft.train import augment_sheet

    assert "0.2534" in augment_sheet.MOVED_OUTSIDE_WAS_DILUTED_BY_THE_PAD["cause"]

    # And the intervention really is a fourfold reduction of a mild jitter.
    assert phase7c.REGION_AWARE["inside_strength"] == 0.25
    assert max(
        phase7c.PHOTOMETRIC[k] for k in ("brightness", "contrast", "saturation")
    ) == 0.2


def test_the_nulls_are_recorded_as_provisional_on_seeds_and_the_paired_test():
    """**[MEASURED] Two of the phase's nulls are properties of n=5, not of the
    data**, and the paired half of PLAN §4.3's criterion was promised and
    never run.

    **[UPDATED 2026-08-04] The direction argument this test used to assert was
    wrong, and the record now says so.** It read: an unpaired threshold on a
    paired design is conservative for the "claimably worse" verdicts and
    anti-conservative for every null, so the negatives are safe. Backwards.
    The paired test ran and withdrew all six negatives; the nulls stayed null.
    The reasoning was correct about condition 2 -- which was never what bound
    those six.

    The n=5-versus-n=10 arithmetic below is still exactly right and still
    worth pinning: it is a true statement about condition 2's threshold. It
    simply never decided anything, because condition 1 fails for every one of
    these comparisons. ``TEN_SEED_ROUND_NOT_RUN``.
    """
    from cleft.train.phase3 import combined_claimable_delta

    record = phase7c.OUTSTANDING_BEFORE_THE_NULLS_ARE_REPORTABLE
    arms = phase7c.STAGE_7C_RESULTS["rounds"]["matched3"]["arms"]

    for key, pair in (
        ("photometric_vs_identity", ("0_identity", "1_photometric")),
        ("region_awareness_photometric", ("1_photometric", "4_region_photometric")),
    ):
        a, b = pair
        entry = record["nulls_that_flip_at_ten_seeds"][key]
        delta = arms[b]["pcc"] - arms[a]["pcc"]
        assert entry["delta"] == pytest.approx(delta, abs=5e-5), key

        for n, field in ((5, "n5"), (10, "n10")):
            threshold = combined_claimable_delta(
                arms[a]["sd"], n, arms[b]["sd"], n
            )["arm_means_95"]
            assert entry[field] == pytest.approx(threshold, abs=5e-5), key

        # Null at five, claimable at ten -- which is the whole point.
        assert abs(delta) < entry["n5"]
        assert abs(delta) > entry["n10"], (
            f"{key} no longer flips at ten seeds; the fragility this records "
            "has gone and the note should be re-read"
        )

    # And the promised paired test is recorded as outstanding, not as done.
    # It ran on 2026-08-04, and both directional predictions are recorded as
    # what they were -- predictions, one of them wrong -- rather than edited
    # into agreement with the outcome.
    paired = record["paired_bca"]
    assert paired["status"].startswith("RUN")
    assert "WRONG" in paired["effect_on_negatives"]
    assert "PREDICTED" in paired["effect_on_nulls"]
    assert phase7c.PAIRED_BCA_WITHDREW_EVERY_VERDICT["intervals_excluding_zero"] == (
        "0 of 45"
    )
    assert any(
        "paired BCa" in item
        for item in phase7c.NOT_A_SEARCH["carried_over"]
    ), "the phase no longer promises the paired test, so this note is stale"


def test_the_correction_changed_exactly_one_verdict():
    """**The vindication of running the second round, and its limit.**

    One arm crossed a threshold; five did not. Both readings belong -- "the
    correction mattered" and "the rounds broadly agree" are each true, and
    quoting only one of them would misdescribe the result.
    """
    record = phase7c.STAGE_7C_RESULTS
    long_round = record["rounds"]["selected30"]["against_identity"]
    short_round = record["rounds"]["matched3"]["against_identity"]

    changed = {
        name for name in long_round
        if long_round[name]["claimable"] != short_round[name]["claimable"]
    }
    assert changed == {record["changed_verdict"]["arm"]} == {"1_photometric"}

    # It went claimable -> null, not the other way.
    assert long_round["1_photometric"]["claimable"] is True
    assert short_round["1_photometric"]["claimable"] is False

    # The recorded moves must be the two rounds' own difference.
    rounds = record["rounds"]
    for name, move in record["between_round_moves"].items():
        actual = (
            rounds["matched3"]["arms"][name]["pcc"]
            - rounds["selected30"]["arms"][name]["pcc"]
        )
        assert move == pytest.approx(actual, abs=5e-5), name
    assert max(
        record["between_round_moves"], key=lambda n: abs(
            record["between_round_moves"][n]
        )
    ) == "1_photometric"
    assert record["changed_verdict"]["moved_most"] is True


def test_the_lateness_story_stays_retracted():
    """**A plausible explanation the arithmetic refuses**, kept refuted so it
    cannot be re-told.

    The draft claim was that photometric flipped because it is the mildest
    augmentation and so had the most to gain from a lucky late draw. Three
    things in the data refuse it, and each is asserted here rather than
    described -- if a future round made any of them false, the story would be
    live again and this test would say so.
    """
    record = phase7c.STAGE_7C_RESULTS
    long_round = record["rounds"]["selected30"]
    against = long_round["against_identity"]

    # 1. The arm closest to its threshold at the long budget did NOT flip.
    def margin(name):
        return abs(against[name]["delta"]) - against[name]["threshold"]

    closest = min(against, key=margin)
    assert closest == "4_region_photometric" != record["changed_verdict"]["arm"]

    # 2. The arm with the most late-selecting folds moved the OTHER way.
    def late_folds(name):
        return sum(1 for e in long_round["arms"][name]["epochs"] if e > 3)

    latest = max(
        (n for n in against), key=lambda n: (late_folds(n), abs(
            record["between_round_moves"][n]
        ))
    )
    assert latest == "6_whole_image_full"
    assert record["between_round_moves"][latest] < 0, (
        "the arm that selected latest no longer moves against the trend, so "
        "the retracted lateness story may deserve re-examining"
    )

    # 3. And the record says the explanation is not established.
    assert "NOT ESTABLISHED" in record["changed_verdict"]["why_here"]


def test_the_matched_epoch_round_is_declared_as_a_reduction_not_a_removal():
    """**Overclaiming here would be the easy mistake.** ``run_fold`` always
    keeps the best epoch, so no-selection is not reachable through config, and
    arm 0 has no draws to select among regardless. Best-of-three rather than
    best-of-thirty is a real reduction and is not the same as removing the
    confound."""
    matched = phase7c.MATCHED_EPOCH_POLICY
    conflate = phase7c.SELECTION_CONFLATES_DRAW_QUALITY

    assert matched["max_epochs"] == 3 < phase7c.EPOCH_POLICY["max_epochs"]
    assert matched["patience"] >= matched["max_epochs"]
    assert matched["monitor"] == phase7c.EPOCH_POLICY["monitor"], (
        "the two rounds must differ in the epoch budget ALONE, or the "
        "comparison between them varies two factors"
    )
    assert "best-of-three rather than best-of-thirty" in matched["reduces_not_removes"]
    assert "not reachable" in matched["reduces_not_removes"]
    assert matched["both_rounds_reported"] is True

    # The confound's direction was the reason the round was run, and it is
    # recorded as PREDICTED rather than as established -- the measurement
    # refused it (see test_the_confounds_predicted_direction_is_...).
    assert "AGAINST the augmented arms" in conflate["direction_predicted"]
    assert "NOT SUPPORTED" in conflate["direction_measured"]["verdict"]
    assert "deterministic" in conflate["arm_0_is_immune"]
    assert "not a design problem" in conflate["not_engineerable_away"]


def test_both_rounds_ship_and_differ_only_in_the_budget(repo_root, monkeypatch):
    from cleft.config import load_config

    monkeypatch.setenv("CLEFT_SCUT_ROOT", "/nonexistent/sentinel")
    for arm in phase7c.arms():
        base = load_config(
            repo_root / "configs" / f"{arm['name']}.yaml"
        )["task"]
        matched = load_config(
            repo_root / "configs" / f"{arm['name'].replace('p7c_', 'p7c_m3_', 1)}.yaml"
        )["task"]

        differing = {
            key for key in set(base) | set(matched)
            if base.get(key) != matched.get(key)
        }
        assert differing == {"max_epochs", "patience"}, (
            f"{arm['name']}: the two rounds differ in {sorted(differing)}; "
            "they must differ in the epoch budget alone"
        )
        assert matched["patience"] >= matched["max_epochs"]


def test_the_summary_arithmetic_is_visible():
    summary = phase7c.summary()
    assert summary["n_arms"] == 7
    assert summary["seeds_per_arm"] == 5
    assert summary["total_fits"] == 35
    assert len(summary["comparisons"]) == 3
    assert summary["horizontal_flip"] is False
