"""Phase 8's pre-registration, checked rather than read.

**A pre-registration in a markdown file is a document; one the suite checks is
a constraint.** Every wording here that has to survive contact with an
unwelcome result is asserted, and the assertions are one-directional where the
direction matters: they permit a finding getting weaker and refuse it getting
stronger without evidence.

The tests that carry weight are the ones about ORDER. §0's descriptive wording
was pre-registered before the A-vs-B comparison ran, and the identical
sentence adopted afterwards would be a retreat. Only the record distinguishes
them, so the record is what these check.
"""

from __future__ import annotations

import pytest

from cleft import ladder, phase8


# --------------------------------------------------------------------------
# the arms
# --------------------------------------------------------------------------


def test_every_arm_exists_in_the_ladder_at_the_geometry_claimed():
    """A stem that does not exist, or exists at another geometry, would send
    every downstream artifact to the wrong run."""
    by_name = {arm["name"]: arm for arm in ladder.arms()}
    for key, spec in phase8.ARMS.items():
        arm = by_name.get(spec["stem"])
        assert arm is not None, f"arm {key}: {spec['stem']} is not in the ladder"
        assert arm["geometry"] == spec["geometry"], key


def test_arm_b_is_srgnns_best_cell_and_shares_arm_as_geometry():
    """**The brief named the masked-G2 cell at 0.1507 while its own framing
    sentence quoted 0.1719.** Arm B is the cell the sentence cites, it keeps
    the native scheme, and it is at G1 -- which is what removes the
    cross-geometry region mapping from §5 rather than requiring it be built.
    """
    b = phase8.ARMS["B"]
    assert b["pcc"] == max(ladder.STAGE_D1_AT_G1["cells"]["srgnn"])
    assert b["pcc"] > ladder.STAGE_D_AT_G2["cells"]["srgnn"][2], (
        "the masked-G2 cell now scores higher; re-read the arm-B decision"
    )
    assert b["geometry"] == phase8.ARMS["A"]["geometry"] == "g1"

    arm = next(a for a in ladder.arms() if a["name"] == b["stem"])
    assert arm["region_scheme"] in (None, "native"), arm["region_scheme"]


def test_arm_c_stays_at_g2_and_the_cost_of_moving_it_is_recorded():
    """AG-Net's G1 cells are 0.0613 / 0.0209 / 0.0634 against 0.1300 at
    masked-G2. Moving it to G1 for consistency would cost more than half its
    performance, and explaining a model at 0.06 is explaining noise."""
    c = phase8.ARMS["C"]
    assert c["geometry"] == "g2"
    best_g1 = max(ladder.STAGE_D1_AT_G1["cells"]["agnet"])
    assert best_g1 < c["pcc"] / 2, (
        "AG-Net's G1 cells are no longer less than half its G2 cell; the "
        "reason arm C stays at G2 has changed"
    )
    assert "0.0613" in c["why"]


def test_only_a_vs_b_gets_an_interval():
    """A and C are at different geometries, so their region boxes differ. The
    brief's own exit criterion 5 already scopes the interval'd comparison to A
    and B; this makes that explicit rather than leaving it to be inferred."""
    assert phase8.INTERVALLED_COMPARISONS == ("A_vs_B",)
    for name, reason in phase8.DESCRIPTIVE_ONLY.items():
        assert "cross-geometry" in reason, name
    assert "A_vs_B" not in phase8.DESCRIPTIVE_ONLY


# --------------------------------------------------------------------------
# §0 -- and the order in which it was written
# --------------------------------------------------------------------------


def test_the_finding_is_descriptive_and_was_pre_registered_before_the_run():
    """**The ordering is the whole point.** A wording picked in advance is a
    pre-registration; the identical wording picked after the comparison failed
    is a retreat. ``ladder.TRADE_OFF_PAIR["if_unresolvable"]`` carries the same
    sentence and was written before the run, which is what makes this the
    default rather than a fallback.
    """
    statement = phase8.STATEMENT_OF_THE_FINDING
    assert "description of the ordering" in statement["reportable"]
    assert "best-SCORING" in statement["reportable"]

    # The same sentence, in the record written before the comparison ran.
    assert "best-SCORING" in ladder.TRADE_OFF_PAIR["if_unresolvable"]
    assert "description" in ladder.TRADE_OFF_PAIR["if_unresolvable"]


def test_the_trade_off_may_not_be_stated_as_a_claim():
    statement = phase8.STATEMENT_OF_THE_FINDING
    assert "0 of 5" in statement["forbidden"]
    assert statement["supporting"]["n_excluding_zero"] == 0
    assert ladder.TRADE_OFF_PAIR["result"]["claimable"] is False


def test_the_supporting_numbers_match_the_measured_result():
    """One arithmetic, two records -- so a later correction cannot land in
    only one of them."""
    supporting = phase8.STATEMENT_OF_THE_FINDING["supporting"]
    result = ladder.TRADE_OFF_PAIR["result"]
    assert supporting["delta"] == result["delta"]
    assert supporting["margin"] == result["margin"]
    assert supporting["n_excluding_zero"] == result["n_excluding_zero"]
    assert supporting["n_seeds"] == result["n_seeds"]
    assert supporting["sd_ratio"] == ladder.SEED_STABILITY_ASYMMETRY["ratio"]


def test_both_of_srgnns_means_are_recorded_with_the_reason():
    """**A reader meeting 0.1719 in the ladder tables and +0.0637 here will
    try to reconcile them.** 0.1884 is the same arm over the five seeds ViT
    also ran, which is the subset pairing requires."""
    b = phase8.ARMS["B"]
    assert b["pcc"] == 0.1719 and b["seeds"] == 10
    assert b["pcc_paired_five_seeds"] == 0.1884
    assert b["pcc_paired_five_seeds"] == (
        ladder.TRADE_OFF_PAIR["result"]["srgnn_paired_mean"]
    )
    why = ladder.TRADE_OFF_PAIR["why_two_means"]
    assert "ten-seed" in why and "five seeds" in why

    # The delta follows the PAIRED means, not the recorded ones. Tolerance is
    # 1.5e-4 because the run computes the delta at full precision while the
    # means are recorded to four decimals, so differencing two of them admits
    # up to 1e-4 of rounding: 0.2520 - 0.1884 gives 0.0636 against a reported
    # 0.0637. Tightening this would be asserting that the record's rounding
    # matches the run's arithmetic, which it cannot.
    result = ladder.TRADE_OFF_PAIR["result"]
    assert result["delta"] == pytest.approx(
        result["vit_paired_mean"] - result["srgnn_paired_mean"], abs=1.5e-4
    )
    assert result["delta"] != ladder.TRADE_OFF_PAIR["delta_if_recorded_means_hold"]


def test_the_seed_stability_asymmetry_is_reported_with_the_means():
    """0.0520 against 0.0148 on the same five seeds and the same folds -- a
    fact about the architectures, not about the comparison."""
    record = ladder.SEED_STABILITY_ASYMMETRY
    assert record["ratio"] == pytest.approx(
        record["srgnn_sd"] / record["vit_sd"], abs=0.01
    )
    assert "architecture, not evaluation" in record["same_seeds_same_folds"]
    # It is what drives the margin down despite a LARGER delta than the
    # ladder headline's.
    assert ladder.TRADE_OFF_PAIR["result"]["delta"] > (
        ladder.BEST_ARM["nearest_challenger"]["delta"]
    )
    assert ladder.TRADE_OFF_PAIR["result"]["margin"] > (
        float(ladder.BEST_ARM["nearest_challenger"]["margin"].rstrip("x"))
    )


# --------------------------------------------------------------------------
# Grad-CAM
# --------------------------------------------------------------------------


def test_guided_grad_cam_is_not_used():
    assert phase8.GUIDED_GRAD_CAM["used"] is False


def test_no_maps_are_published_before_the_randomisation_test():
    test = phase8.RANDOMISATION_TEST
    assert test["required"] is True
    assert test["publish_before_it_passes"] is False
    assert "R7 instance" in test["if_maps_survive"]


def test_the_frozen_backbone_caveat_lives_in_the_artifact():
    """A caveat in prose travels separately from the figure, and the figure is
    what gets reused."""
    caveat = phase8.FROZEN_BACKBONE_CAVEAT
    assert caveat["in_the_artifact"] is True
    assert "NOT what the model learned about clefts" in caveat["text"]
    assert caveat["forbidden_wording"] not in caveat["text"].replace(
        "NOT what the model learned about clefts", ""
    )


def test_the_sample_is_stratified_not_cherry_picked():
    assert phase8.SAMPLING["cherry_picking"] is False
    assert "grade range" in phase8.SAMPLING["stratified_by"]


# --------------------------------------------------------------------------
# node voting and the comparison
# --------------------------------------------------------------------------


def test_the_node_counts_match_the_models():
    assert phase8.NODES["B"]["count"] == 27
    assert phase8.NODES["C"]["count"] == 37
    # The difference in what a "region" means is recorded, not flattened.
    assert phase8.NODES["B"]["named"] is True
    assert phase8.NODES["C"]["named"] is False
    assert "kappa=8" in phase8.NODES["C"]["source"]


def test_node_weights_are_reported_with_a_spread():
    """A 27-way ranking on 237 patients produces an apparent ordering whether
    or not one exists. Phase 4's relevance diagnostic is the precedent."""
    reporting = phase8.NODE_WEIGHT_REPORTING
    assert "across patients" in reporting["spread"]
    assert "Bonferroni" in reporting["precedent"]
    assert set(reporting["clinical_prior"]) == {
        "nasal form", "nasal symmetry", "nasolabial profile", "vermillion border",
    }
    assert "equally worth reporting" in reporting["both_outcomes_reportable"]


def test_the_pairing_unit_is_the_patient_and_the_alternative_is_recorded():
    """**Decided in the pre-registration, not during the build.** Pooling to
    one 27-vector per arm and bootstrapping over the regions would be a spread
    over a quantity with no replicates: 27 regions are one partition of one
    face, not 27 independent draws."""
    unit = phase8.PAIRING_UNIT
    assert unit["unit"] == "patient"
    assert "237" in unit["spread"]
    assert "no replicates" in unit["rejected"]["why"]
    assert "DISTRIBUTION" in unit["consequence"]
    # All three outcomes are reportable, including the null.
    assert set(unit["outcomes"]) == {"agree", "disagree", "unresolvable"}


def test_a_comparative_statement_needs_a_number():
    rule = phase8.COMPARATIVE_CLAIMS_NEED_A_NUMBER
    assert "descriptive" in rule["rule"]
    assert "29 of 30" in rule["descriptive_is_allowed"]
    assert "t-SNE" in rule["applies_to"]


# --------------------------------------------------------------------------
# t-SNE
# --------------------------------------------------------------------------


def test_tsne_is_a_figure_never_evidence():
    assert phase8.TSNE["is_evidence"] is False
    assert len(phase8.TSNE["perplexities"]) >= 2
    assert phase8.TSNE["tuned_on_the_result"] is False
    assert phase8.TSNE["companion_needs_uncertainty"] is True
    assert phase8.TSNE["figure_without_companion"] is False


def test_the_summary_carries_the_decisions_a_run_record_needs():
    summary = phase8.summary()
    assert summary["finding"] == phase8.STATEMENT_OF_THE_FINDING["reportable"]
    assert summary["intervalled"] == ["A_vs_B"]
    assert summary["geometries"]["A"] == summary["geometries"]["B"] == "g1"
    assert summary["randomisation_test_required"] is True
    assert summary["guided_grad_cam"] is False
    assert "UNRESOLVED" in summary["trade_off_verdict"]


# --------------------------------------------------------------------------
# the target layer -- measured, not assumed
# --------------------------------------------------------------------------


def test_the_grad_cam_target_is_not_the_final_blocks_output():
    """**[MEASURED] Block 12's patch-token gradient is identically zero.**

    ``extract`` reads the transformer as ``forward_head(forward_features(x),
    pre_logits=True)``, and timm's default token pooling makes that ``x[:, 0]``
    after a per-token LayerNorm -- so token 0's output depends only on token
    0's input, and the patch tokens at block 12's output feed nothing.

    A zero gradient does not give a blank map: Grad-CAM normalises by the
    maximum, so float noise becomes a full-range picture. That is why the
    target is pinned here rather than chosen when maps exist.
    """
    target = phase8.GRAD_CAM_TARGET
    # [CORRECTED 2026-08-04] This asserted index == 11, which IS the final
    # block on a twelve-block model -- the test agreed with the defect because
    # it restated the number instead of deriving it. It now derives it.
    assert target["index"] == target["n_blocks"] - 2
    assert target["measurement"]["blocks_11_final"]["patch_grad_max"] == 0.0
    assert target["measurement"]["blocks_10_target"]["patch_grad_max"] > 0.0
    assert "identically zero" in target["why_not_block_12"]
    assert target["fixed_before_any_map_exists"] is True


def test_the_depth_choice_cites_the_measurement_that_makes_it_consequential():
    """Phase 7B measured the pooling axis as monotone in depth, so a shallower
    target is a different representation rather than a smoother view of the
    same one."""
    target = phase8.GRAD_CAM_TARGET
    assert "7B" in target["why_not_shallower"] or "phase 7B" in (
        target["why_not_shallower"]
    )
    assert "-0.08" in target["why_not_shallower"]
    # Every alternative is named with its refusal, including the sweep.
    assert len(target["alternatives_considered"]) == 3
    assert all("REFUSED" in entry for entry in target["alternatives_considered"])
    assert any("sweep" in entry for entry in target["alternatives_considered"])


def test_the_resolution_floor_is_recorded_before_anyone_reads_a_map():
    """14x14 upsampled to 224 is 16x, so nothing finer than a 16-pixel block
    is real -- and a smooth field invites exactly the finer reading."""
    floor = phase8.RESOLUTION_FLOOR
    grid_w = floor["token_grid"][0]
    assert floor["image"][0] // grid_w == floor["upsample_factor"] == 16
    assert floor["floor_px"] == floor["upsample_factor"]
    assert "philtral column" in floor["forbidden_reading"]
    assert "do not assume" in floor["consequence_for_region_comparison"]
    # The token grid must match what the target layer actually produces.
    assert phase8.GRAD_CAM_TARGET["grid"] == floor["token_grid"]


def test_the_sample_rule_is_fixed_and_forbids_substitution():
    """**Replacement-on-rejection is cherry-picking with an audit trail that
    looks principled**, and it is the failure mode a "stratified sample"
    invites."""
    rule = phase8.STRATIFIED_SAMPLE
    assert rule["n_strata"] * rule["per_stratum"] == rule["n_total"] == 15
    assert rule["manual_substitution"] is False
    assert "do NOT replace" in rule["on_failure"]
    assert isinstance(rule["seed"], int)
    assert phase8.SAMPLING["rule"] is rule


def test_the_live_path_is_reused_for_the_model_but_not_the_forward():
    """``FrozenExtractor`` owns the frozen boundary, and its forward runs
    under ``no_grad`` -- so the gradient pass is a method on it rather than a
    second class defining that boundary again."""
    reuse = phase8.LIVE_PATH_REUSE
    assert "model construction" in reuse["reuse"]
    assert "no_grad" in reuse["cannot_reuse"]
    assert "METHOD on FrozenExtractor" in reuse["resolution"]

    source = (
        __import__("pathlib").Path(phase8.__file__).parent
        / "train" / "torch_backbone.py"
    ).read_text(encoding="utf-8")
    assert "class FrozenExtractor" in source
    assert "no_grad" in source, (
        "FrozenExtractor no longer runs under no_grad; the reason the "
        "gradient pass could not reuse it has changed"
    )


# --------------------------------------------------------------------------
# the config
# --------------------------------------------------------------------------


def test_the_config_copies_arm_as_hyperparameters_rather_than_restating_them(
    repo_root,
):
    """**The refit gates on reproducing 0.2520**, so a mistyped learning rate
    would fail on the cluster after loading every embedding, for a reason the
    log states but nobody expects. Copying makes "same arm" true by
    construction."""
    import yaml

    arm = yaml.safe_load(
        (repo_root / "configs" / "p7_d1_vit_b16_imagenet_g1.yaml").read_text("utf-8")
    )
    p8 = yaml.safe_load(
        (repo_root / "configs" / "p8_grad_cam.yaml").read_text("utf-8")
    )
    for field in ("learning_rate", "weight_decay", "seeds", "inner_val_frac",
                  "max_epochs", "patience", "monitor", "batch_size",
                  "geometry", "label", "backbone"):
        assert p8["task"][field] == arm["task"][field], field
    # And the same data, hash for hash.
    assert p8["inputs"] == arm["inputs"]


def test_the_pre_registration_is_not_a_config_field(repo_root):
    """The layer is the field most likely to be nudged because a map looked
    better, so a config must not be able to set it."""
    import yaml

    from cleft.config.schema import TASK_SPECS

    p8 = yaml.safe_load(
        (repo_root / "configs" / "p8_grad_cam.yaml").read_text("utf-8")
    )
    allowed = set(TASK_SPECS["grad_cam"])
    for forbidden in ("block", "layer", "target_layer", "perplexity",
                      "n_strata", "per_stratum", "baseline", "threshold"):
        assert forbidden not in allowed, forbidden
        assert forbidden not in p8["task"], forbidden


def test_the_shipped_phase8_config_matches_the_derived_one(repo_root):
    import os
    import subprocess
    import sys as _sys

    result = subprocess.run(
        [_sys.executable,
         str(repo_root / "scripts" / "generate_phase8_configs.py"), "--check"],
        capture_output=True, text=True, cwd=str(repo_root),
        env={**os.environ, "PYTHONPATH": str(repo_root / "src")},
    )
    assert result.returncode == 0, f"{result.stdout}{result.stderr}"


def test_the_target_index_is_the_second_to_last_block():
    """**[MEASURED 2026-08-04] This is the check that was missing.**

    The record carried ``index: 11`` on a twelve-block model, so the task
    hooked ``blocks[11]`` -- the final block, the one the record exists to
    avoid -- and got the identically-zero gradient, refused downstream with a
    message that read as a property of the layer rather than as the wrong
    layer having been asked for.

    **No GPU was needed to catch it.** The target must be the second-to-last
    block: the last one's patch tokens feed nothing, and anything shallower
    changes the representation (7B). That is arithmetic over two integers.
    """
    target = phase8.GRAD_CAM_TARGET
    assert target["index"] == target["n_blocks"] - 2 == 10
    assert target["depth_1indexed"] == target["index"] + 1


def test_the_layer_label_names_blocks_that_exist():
    """**The label said ``blocks[12]`` and no twelve-block model has one.**

    It contradicted itself in writing and was read twice without the
    contradiction registering, because 1-based prose and 0-based indices were
    mixed in the same sentence.
    """
    import re

    target = phase8.GRAD_CAM_TARGET
    named = [int(n) for n in re.findall(r"blocks\[(\d+)\]", target["layer"])]
    assert named == [target["index"], target["index"] + 1], named
    assert max(named) <= target["n_blocks"] - 1, (
        f"the label names blocks[{max(named)}] and there are only "
        f"{target['n_blocks']} blocks (0..{target['n_blocks'] - 1})"
    )


def test_the_measurement_keys_are_zero_based_like_the_index():
    """1-based key names are how the off-by-one survived being written down
    twice, so the measurement is keyed the way ``model.blocks`` is."""
    measurement = phase8.GRAD_CAM_TARGET["measurement"]
    assert measurement["blocks_11_final"]["patch_grad_max"] == 0.0
    assert measurement["blocks_10_target"]["patch_grad_max"] > 0.0
    through = measurement["through_the_real_function"]
    assert through[f"blocks_{phase8.GRAD_CAM_TARGET['index']}"] > 0.0
    assert through[f"blocks_{phase8.GRAD_CAM_TARGET['n_blocks'] - 1}"] == 0.0


# --------------------------------------------------------------------------
# the randomisation result, and the criterion's own defect
# --------------------------------------------------------------------------


def test_parameter_dependence_is_established_with_an_interval():
    """**The phase's finding.** Randomising the weights degrades the maps, and
    the statement carries a spread because §2's rule applies to Phase 8's own
    results as much as to its comparisons."""
    record = phase8.PARAMETER_DEPENDENCE_ESTABLISHED
    ci = record["separability"]["ci95"]
    assert ci[0] < ci[1] < 0, ci
    assert record["separability"]["excludes_zero"] is True
    assert record["separability"]["resampling_unit"] == "patient"


def test_the_criterions_own_null_corroborates_the_finding():
    """A median cut implies a 50% survival rate under exchangeability, so
    fifteen patients predict 7.5 survivors. Two were observed."""
    from math import comb

    corroboration = phase8.PARAMETER_DEPENDENCE_ESTABLISHED["corroboration"]
    n = 15
    assert corroboration["expected_survivors"] == n * 0.5
    exact = sum(comb(n, k) for k in range(corroboration["observed_survivors"] + 1))
    assert corroboration["p_binomial"] == pytest.approx(exact / 2**n, abs=5e-5)


def test_the_two_survivors_are_distinguished_by_percentile():
    """**Treating them as one class reports an artefact of the threshold as a
    property of two patients.** 77 sits at the 91st percentile of the
    baseline; 190 at the 62nd."""
    record = phase8.THE_TWO_SURVIVORS
    assert record["77"]["baseline_percentile"] > 0.9
    assert 0.5 < record["190"]["baseline_percentile"] < 0.75
    # 77's account is that both statistics measure one weakness; 190's is not.
    assert "one weakness" in record["77"]["reading"]
    assert "does not fit" in record["190"]["reading"]
    assert "artefact of the threshold" in record["do_not_merge"]


def test_the_median_cut_defect_is_recorded_separately_from_the_run():
    """**The rule's design and the run's verdict must not be confusable.** The
    verdict stands; the criterion is still criticised."""
    record = phase8.MEDIAN_CUT_WAS_NOT_DEFENSIBLE
    assert "NOT this run" in record["scope"]
    assert "50% cut" in record["defect"]
    assert "phase7c" in record["not_changed_here"]
    # And the rule really is unchanged.
    assert phase8.RANDOMISATION_TEST["criterion"]["baseline"].startswith("median")


def test_a_per_patient_gate_is_not_estimable_at_this_sample_size():
    """**Arithmetic, not preference.** The family-wise-corrected per-patient
    level is below the finest percentile 105 pairs can estimate."""
    record = phase8.DEFENSIBLE_CRITERION_FOR_FUTURE_USE
    numbers = record["per_patient_is_not_estimable"]
    assert numbers["bonferroni_alpha"] == pytest.approx(0.05 / 15, abs=1e-5)
    assert numbers["finest_percentile_from_105_pairs"] == pytest.approx(
        1 / 105, abs=1e-5
    )
    assert numbers["bonferroni_alpha"] < numbers["finest_percentile_from_105_pairs"]
    assert numbers["q05_family_wise_failure_rate"] == pytest.approx(
        1 - 0.95**15, abs=1e-3
    )
    assert record["applies_to"].endswith("NOT this run")
    assert "separability interval is the criterion" in record["recommended"]


# --------------------------------------------------------------------------
# the closing findings
# --------------------------------------------------------------------------


def test_the_framing_finding_is_refuted_not_weakened():
    """**[RE-MEASURED 2026-08-08] The maps do not attend to framing more than
    chance.**

    Scored against each patient's OWN expectation: mass median 0.4051 against
    an expected median of 0.4386, so measured mass is BELOW the typical
    patient's expectation. Ratio median 1.0652 -- over 1.0 only because the
    median of a ratio is not the ratio of medians -- and 10 of 15 above their
    own, where coin flips give 8.

    The 1.60x was entirely the omitted trapezium corners.
    """
    record = phase8.FRAMING_NOT_ANATOMY
    assert record["status"].startswith("REFUTED")
    corrected = record["corrected_result"]
    assert corrected["mass_median"] < corrected["expected_median"], (
        "the measured mass is below the expectation; that is the refutation"
    )
    assert corrected["above_own_expectation"] == 10
    assert corrected["undefined"] == 0
    assert "not weakened, refuted" in record["status"]
    # The reading came from a picture, and that is worth saying.
    assert "edge being large" in record["the_picture_was_not_lying_the_reading_was"]

    # The superseded numbers are kept so the correction stays legible, and
    # their own arithmetic must still be self-consistent.
    pad, ring = record["out_of_content"], record["outer_ring"]
    assert pad["ratio"] > ring["ratio"]
    assert pad["ratio"] == pytest.approx(pad["median"] / pad["uniform"], abs=0.01)
    assert ring["ratio"] == pytest.approx(ring["median"] / ring["uniform"], abs=0.01)
    # The uniform expectations are the ones the sheet computes.
    from cleft import gradcam_sheet

    assert ring["uniform"] == pytest.approx(gradcam_sheet.OUTER_RING_UNIFORM, abs=5e-4)
    # [CORRECTED 2026-08-08] The pad expectation is no longer a constant --
    # see FRAMING_NOT_ANATOMY's superseded block.
    assert not hasattr(gradcam_sheet, "PAD_FRACTION_AT_MEDIAN_AR")
    assert "not ViT's border tokens" in record["verdict"]


def test_the_phase_closes_with_one_standing_finding():
    """Parameter dependence stands; framing is refuted. One finding, and the
    record must not read as two."""
    assert phase8.PARAMETER_DEPENDENCE_ESTABLISHED[
        "separability"]["excludes_zero"] is True
    assert phase8.PHASE_8_CLOSING["findings_standing"] == 1
    assert len(phase8.PHASE_8_CLOSING["findings"]) == 1, (
        "the refuted finding should no longer be listed as a finding"
    )
    # Road B's Branch 3 rests on the withdrawn figure and must be rewritten.
    assert "Branch 3" in phase8.PHASE_8_CLOSING["refuted_by_remeasurement"]
    assert "no effect" in phase8.PHASE_8_CLOSING["refuted_by_remeasurement"]


def test_the_convergence_nulls_are_recorded_with_their_widths():
    """**The widths are the finding.** "rho -0.321, not significant" reads as
    weak evidence against; the interval shows it is no evidence at all."""
    record = phase8.CONVERGENCE_UNTESTED
    for name in ("ring_vs_randomisation_final", "ring_vs_seed_agreement"):
        entry = record[name]
        lo, hi = entry["ci95"]
        assert lo < entry["spearman"] < hi
        assert entry["width"] == pytest.approx(hi - lo, abs=0.01)
        assert entry["width"] > 1.0, "a narrow interval would change the reading"
    assert record["status"].startswith("UNTESTED")
    assert "not refuted" in record["status"]


def test_the_region_comparison_constraint_is_recorded_before_it_is_built():
    """~40% of A's mass falls outside the 27 boxes, so §5 would compare a
    renormalised 60% of A against 100% of B. Stated now, not discovered in
    the result."""
    note = phase8.FRAMING_NOT_ANATOMY["constrains_the_region_comparison"]
    assert "renormalised" in note and "no outside" in note
    assert phase8.PHASE_8_CLOSING["exit_criteria"][
        "5_region_comparison_a_vs_b"] == "NOT BUILT"


def test_the_criterion_adoption_carries_both_guards():
    """**A better rule adopted, and the two facts that stop it reading as
    a failed gate being swapped out.**"""
    record = phase8.RANDOMISATION_CRITERION_ADOPTED
    assert record["criterion"] == "DEFENSIBLE_CRITERION_FOR_FUTURE_USE"

    # Guard 1: registered BEFORE the outcome it judges.
    registered = phase8.DEFENSIBLE_CRITERION_FOR_FUTURE_USE["pre_registered"]
    assert registered == "2026-08-04"
    assert registered < record["adopted"], (
        "the criterion must predate its adoption, or it is a forking path"
    )
    assert "forking path" in record["guard_registered_before_the_outcome"]

    # Guard 2: the old run keeps its verdict, and the record it came from
    # still says so.
    assert "does not retroactively pass" in record[
        "guard_the_old_run_keeps_its_verdict"
    ]
    assert "verdict stands" in phase8.MEDIAN_CUT_WAS_NOT_DEFENSIBLE["scope"]
    assert phase8.PHASE_8_CLOSING["gate"].startswith("stands")

    # Adoption computes nothing: the phase-level test already ran and its
    # own record already says criterion 2 is met at that level.
    established = phase8.PARAMETER_DEPENDENCE_ESTABLISHED
    assert established["separability"]["excludes_zero"] is True
    assert "met at the level it tests" in established["status"]
    assert "already ran" in record["costs_no_computation"]

    # And the licence is narrow, in both directions.
    assert "AS A SET" in record["licenses"]
    assert "any individual map" in record["does_not_license"]


def test_what_arm_a_randomised_matches_the_code_that_did_it(repo_root):
    """**Read from the implementation, and pinned to it.** If the stage
    generator ever touches the head as well, this record is stale and the
    A<->B analogue argument built on it has to be re-read."""
    import ast

    record = phase8.ARM_A_RANDOMISED_THE_FROZEN_BACKBONE
    assert "model.blocks" in record["touched"]
    assert "head" in record["never_touched"]

    source = (repo_root / "src" / "cleft" / "run.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(source)
    fn = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and node.name == "phase8_randomisation_stages"
    )
    body = ast.get_source_segment(source, fn) or ""
    # It randomises blocks, top-down, and touches nothing else.
    assert "extractor.model.blocks" in body
    assert "range(len(blocks) - 1, -1, -1)" in body
    assert "head" not in body, (
        "the head is untouched -- if that changes, the record above and "
        "the whole component-role argument need re-reading"
    )
    assert "normal_" in body and "parameter.std()" in body


def test_the_two_arms_randomisations_are_recorded_as_incomparable():
    """**A finding about the design, not a detail.** Three descriptions
    coincide for arm A and come apart for arm B, so the phase's stated
    contribution rests on structurally different components."""
    record = phase8.ARM_A_AND_B_RANDOMISATIONS_ARE_NOT_EQUIVALENT

    # Arm A's coincidence is arithmetic, and the numbers agree with the
    # record of what was randomised.
    split = phase8.ARM_A_RANDOMISED_THE_FROZEN_BACKBONE["arm_a_parameters"]
    total = split["backbone_frozen"] + split["head_trained"]
    assert split["backbone_frozen"] / total > 0.9999
    assert f"{split['backbone_frozen']:,}" in record[
        "for_arm_a_three_descriptions_coincide"
    ]
    assert f"{total:,}" in record["for_arm_a_three_descriptions_coincide"]

    # Both tests are run; only one feeds the comparison.
    both = record["decided_run_both"]
    assert "randomise xception" in both["component_role"]
    assert "randomise the graph layers" in both["explanation_source"]
    assert both["comparison_uses"] == "the component-role match only"
    assert "not the same test" in both["explanation_source"]
    # The weaker-pass argument is stated, not glossed.
    assert "structured weights over noise features" in both["component_role"]

    # And the finding is framed as a limitation of the design.
    assert "WEAKER than the brief assumed" in record["the_finding"]
    assert "structurally different components" in record["the_finding"]
    assert record["goes_in_the_write_up_as"].startswith("a limitation")


def test_the_node_weight_statistic_predicts_its_own_failure_mode():
    """**Said before the gate is registered**, as asked: the shorter
    vector may not resolve, and the fix is patients rather than a
    statistic chosen after the result."""
    import math

    record = phase8.NODE_WEIGHT_STATISTIC_DECIDED

    # The noise ratio is arithmetic, re-derived here.
    def spearman_se(k):
        return 1 / math.sqrt(k - 3)

    assert abs(
        spearman_se(196) - record["precision_does_not"]["spearman_se_196_tokens"]
    ) < 5e-4
    assert abs(
        spearman_se(27) - record["precision_does_not"]["spearman_se_27_regions"]
    ) < 5e-4
    assert abs(
        spearman_se(27) / spearman_se(196)
        - record["precision_does_not"]["ratio"]
    ) < 0.05

    # Why it matters: arm A's interval was already close to the boundary.
    ci = phase8.PARAMETER_DEPENDENCE_ESTABLISHED["separability"]["ci95"]
    assert abs(ci[1]) < 0.01, "the upper end really is near zero"
    assert "0.1013" in record["why_it_matters"]

    # The fix, and the asymmetry it creates.
    assert "forward pass" in record["the_fix_is_patients_not_a_statistic"]
    assert "n >= 41" in record["arm_b_could_support_a_per_patient_gate"]
    assert "PHASE-LEVEL criterion for both" in record["decision"]
    assert "Spearman primary" in record["decision"]

    # The pre-commitment that stops a post-hoc statistic switch.
    assert "UNRESOLVED" in record["if_it_does_not_resolve"]
    assert "chosen on the result" in record["if_it_does_not_resolve"]


def test_node_weights_have_a_gate_registered_before_they_exist():
    """**Registered first, deliberately** -- and the two things that do
    not transfer from Grad-CAM's version are named as measurements rather
    than assumptions."""
    record = phase8.NODE_WEIGHT_RANDOMISATION_REGISTERED
    assert "before any weights exist" in record["registered"]
    assert "about EXPLANATIONS" in record["why_they_need_one"]
    assert "asymmetry" in record["why_they_need_one"]
    assert "paid for repeatedly" in record["why_registered_first"]

    # Inherited: the per-patient finding transfers because it is about
    # sample size, not about Grad-CAM.
    assert "percentiles" in record["inherited"]
    assert "below the resolution" in phase8.\
        DEFENSIBLE_CRITERION_FOR_FUTURE_USE[
            "per_patient_is_not_estimable"
        ]["conclusion"]

    # Not inherited: both flagged for measurement.
    differs = record["must_be_measured_not_assumed"]
    assert "196 tokens" in differs["shorter_vector"]
    assert "WIDER baseline" in differs["shorter_vector"]
    assert "frozen" in differs["what_randomise_means"]
    assert "rather than assuming" in differs["what_randomise_means"]
    # The arm it applies to is the one whose explanation is node voting.
    assert phase8.ARMS["B"]["explanation"] == "node_voting"


def test_the_interpretable_arm_is_capped_at_its_own_best():
    """**The pairing is the finding**: arm B is the 0.1719 arm, and Road B
    reproduced that value from a totally different representation."""
    from cleft import roadb

    record = phase8.THE_INTERPRETABLE_ARM_IS_CAPPED
    assert record["arm_b"]["pcc"] == phase8.ARMS["B"]["pcc"] == 0.1719
    assert record["arm_b"]["stem"] == phase8.ARMS["B"]["stem"]
    assert record["against_arm_a"]["pcc"] == phase8.ARMS["A"]["pcc"] == 0.2520

    # Road B's number, from Road B's own record.
    ceiling = roadb.REGION_CROP_ARMS_OBSERVED["srgnn_on_identical_pixels"]
    assert record["road_b_reproduced_it"]["pcc"] == roadb.\
        REGION_CROP_ARMS_OBSERVED["values"]["anatomy_concat_srgnn"]["pcc"]
    assert ceiling["srgnn_best_prior"]["pcc"] == record["arm_b"]["pcc"]
    assert record["road_b_reproduced_it"]["difference"] == 0.0002

    # Framed as the finding, and held at observation strength.
    assert "not a disappointment" in record["the_finding"]
    assert "DESCRIPTIVE form" in record["sharpens_without_claiming"]
    assert "not a proof of a cap" in record["stated_at_its_real_strength"]
    # The statement it sharpens is still the pre-registered descriptive one.
    assert "not a claim about a trade-off" in phase8.\
        STATEMENT_OF_THE_FINDING["reportable"]


def test_the_per_face_cost_is_derived_not_asserted():
    """**n >= 41**, re-derived here so the number cannot drift from the
    arithmetic that produced it -- and necessary, not sufficient, for AR."""
    record = phase8.PER_FACE_CLAIM_COSTS_N_41

    def finest(n):
        return 2 / (n * (n - 1))

    def needed(n):
        return 0.05 / n

    assert finest(15) > needed(15), "not estimable at the current sample"
    assert abs(finest(15) - record["at_n_15"]["finest_percentile"]) < 1e-6
    assert abs(needed(15) - record["at_n_15"]["needed_alpha"]) < 1e-6
    # 41 is the smallest n that resolves it.
    assert finest(41) <= needed(41)
    assert finest(40) > needed(40)
    assert abs(finest(41) - record["at_n_41"]["finest_percentile"]) < 1e-6
    assert "n >= 41" in record["closed_form"]

    # The part that matters more than the arithmetic.
    assert "IN THE SAMPLE" in record["necessary_not_sufficient_for_ar"]
    assert "in no sample at all" in record["necessary_not_sufficient_for_ar"]
    assert "not arithmetic" in record["necessary_not_sufficient_for_ar"]
    assert "demonstration" in record["honest_ar"]
    assert "may not be presented" in record["honest_ar"]
    assert "produces no finding" in record["ar_is_a_deployment_artifact"]


def test_the_closing_record_states_the_criteria_honestly():
    """**Three of seven met, one partial, three unbuilt.** The phase must not
    be written up as the both-families comparison the brief was structured
    around."""
    closing = phase8.PHASE_8_CLOSING
    criteria = closing["exit_criteria"]
    assert len(criteria) == 7
    met = sum(1 for v in criteria.values() if v.startswith("MET"))
    partial = sum(1 for v in criteria.values() if v.startswith("PARTIAL"))
    unbuilt = sum(1 for v in criteria.values() if v == "NOT BUILT")
    assert (met, partial, unbuilt) == (3, 1, 3)
    assert "PARTIALLY MET" in closing["status"]
    assert "not the both-families comparison" in closing["do_not_write_up_as_complete"]


def test_the_gate_is_recorded_as_standing_after_the_criterion_was_criticised():
    """The criterion was shown unsound and left unchanged. Both facts, in one
    record, so neither can be quoted without the other."""
    gate = phase8.PHASE_8_CLOSING["gate"]
    assert "nothing published" in gate and "criterion unchanged" in gate
    assert "median cut" in gate
    assert phase8.PUBLICATION_SCOPE["npz"]["written"] is False


def test_both_graph_models_now_expose_their_explanation(repo_root):
    """**Two of two, one cause** -- and AG-Net's was fixed at a grep's
    cost rather than at arm C's."""
    record = phase8.THE_ARTIFACT_PATH_DISCARDS_THE_EXPLANATION
    assert set(record["instances"]) == {"srgnn", "agnet"}
    assert "region_w" in record["instances"]["srgnn"]
    assert "importance" in record["instances"]["agnet"]
    assert "rode along unused" in record["one_cause"]
    assert "arm C will not rediscover" in record["fixed_in_the_same_pass"]
    assert "DROPPED at a boundary" in record["distinct_from_r11"]

    models = repo_root / "src" / "cleft" / "models"
    for name in ("srgnn", "agnet"):
        text = (models / f"{name}.py").read_text(encoding="utf-8")
        assert "def forward_from_features_with_weights" in text, name
        # The old scalar-returning method is still there, unchanged, so
        # its callers keep working.
        assert "def forward_from_features(" in text, name
        assert "return self._from_features(feat, boxes)[0]" in text, name


def test_the_fold_is_derived_on_both_paths_and_counted():
    """**Derived, counted, and honest about what it does not prove.**"""
    import ast
    from pathlib import Path

    from cleft.train import graph_cleft

    record = phase8.FOLD_IS_DERIVED_ON_BOTH_PATHS
    assert "only against itself" in record["why_derive_rather_than_pass_in"]
    assert "row INDEX" in record["stronger_check_unavailable"]
    assert "pack_indexed" in record["stronger_check_unavailable"]
    assert "not a proof" in record["what_remains_an_assumption"]

    source = Path(graph_cleft.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    run_fn = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "run"
    )
    body = ast.get_source_segment(source, run_fn) or ""
    # The derivation is no longer inside the per-fold branch...
    assert "this_fold = fold_order[fold_calls[\"n\"]]" in body
    unguarded = body.split("this_fold = fold_order")[0]
    assert unguarded.rstrip().endswith(")") or "maps_by_fold is not None" not in (
        unguarded.split("def make_backbone")[-1]
    ), "the fold derivation must not sit inside the per-fold branch"
    # ...and the count check exists and names what it protects.
    assert "backbones were built for" in body
    assert "attributed to the wrong patients" in body


# --------------------------------------------------------------------------
# arm B's extraction, and the three things measuring it turned up
# --------------------------------------------------------------------------


def test_the_component_role_no_op_is_recorded_with_its_consequence():
    """**A no-op that fails in the direction of a finding.** Similarity 1.0
    for every patient reads as "the explanation survives randomisation" --
    the worst available verdict, from a test that changed nothing."""
    record = phase8.COMPONENT_ROLE_IS_A_NO_OP
    assert record["backbone_parameters"] == 20_806_952
    assert record["region_weights_identical"] is True
    assert record["logits_identical"] is True
    assert record["max_abs_difference"] == 0.0
    assert "BELOW the backbone" in record["mechanism"]
    assert "INPUT here" in record["mechanism"]
    # The same structural fact the BatchNorm scoping already recorded.
    assert "_batch_norms" in record["same_structural_fact_as"]
    # And the cost is stated rather than absorbed: this is the mode the one
    # intervalled comparison was designed around.
    assert phase8.INTERVALLED_COMPARISONS == ("A_vs_B",)
    assert "re-extract" in record["what_would_restore_it"]
    assert "not taken here" in record["what_would_restore_it"]


def test_the_stage_curve_is_flat_by_architecture_and_says_so():
    """Four of six stages cannot move region_w. The GATE is unaffected --
    it reads the final stage -- but a flat curve is exactly what gets read
    as robustness after the fact."""
    record = phase8.WHICH_STAGES_CAN_MOVE_THE_EXPLANATION
    assert set(record["can_move"]) == {"self_attn", "weighted_attn"}
    assert set(record["cannot_move"]) == {
        "gnn_mlp1", "gnn_mlp2", "gnn_out", "classifier"
    }
    assert not set(record["can_move"]) & set(record["cannot_move"])
    assert "FINAL stage" in record["gate_unaffected"]
    assert "INDEPENDENT of the message passing" in record["sharpens"]


def test_arm_as_dirty_restart_is_recorded_and_deliberately_not_repaired():
    """Nothing published is wrong: the verdict reads the final stage and the
    per-stage curve is never serialised. Recorded so the next reader does not
    rediscover it, and left alone so arm A's run still matches its record."""
    record = phase8.ARM_A_RANDOMISATION_RESTARTS_DIRTY
    assert "PER PATIENT" in record["what"]
    assert "final" in record["verdict_unaffected"]
    assert "never serialised" in record["nothing_published_is_wrong"]
    assert "cost without a finding" in record["not_repaired"]
    assert "clean snapshot" in record["what_it_earns"]


def test_arm_bs_config_is_generated_from_arm_bs_own_config(repo_root):
    """The refit gate is 0.1719, so the hyperparameters must BE arm B's --
    a retyped learning rate fails the gate after ten seeds of fitting."""
    import yaml

    generated = repo_root / "configs" / "p8_node_weights.yaml"
    arm = yaml.safe_load(
        (repo_root / "configs" / "p7_d1_srgnn_imagenet_g1_native.yaml")
        .read_text(encoding="utf-8")
    )
    payload = yaml.safe_load(generated.read_text(encoding="utf-8"))
    assert payload["task"]["kind"] == "node_weights"
    for field in (
        "geometry", "label", "backbone", "seeds", "learning_rate",
        "weight_decay", "batch_size", "init", "region_scheme", "trainable",
        "max_epochs", "patience", "inner_val_frac", "monitor",
    ):
        assert payload["task"][field] == arm["task"][field], field
    # Same inputs, same verified hashes: the weights must come off the arm's
    # own representation, not a set that merely looks like it. The control
    # set is the one addition. [FILLED 2026-08-14] Its rollup is the declare
    # pass's figure, carried by the generator -- so the assertion is that it
    # is a REAL hash, not the placeholder, and that a regeneration cannot
    # silently regress it to zeros.
    assert payload["inputs"][: len(arm["inputs"])] == arm["inputs"]
    control = payload["inputs"][-1]
    assert control["name"] == payload["task"]["randomised_embeddings_artifact"]
    assert control["path"].endswith("__randomised")
    rollup = control["rollup_sha256"]
    assert len(rollup) == 64 and set(rollup) <= set("0123456789abcdef")
    assert set(rollup) != {"0"}, (
        "the control's hash regressed to the placeholder; the generator must "
        "CARRY the declare pass's figure while the path is unchanged"
    )
    # The seed count is the gate's, stated in one place.
    gate = phase8.SRGNN_NODE_WEIGHTS_R10_READ["refit_gate"]
    assert len(payload["task"]["seeds"]) == gate["seeds"]

    header = generated.read_text(encoding="utf-8").split("schema_version")[0]
    assert "COMPONENT_ROLE_IS_A_NO_OP" in header
    assert "BITWISE identical" in header
    assert str(gate["expected_pcc"]) in header


def test_the_task_reads_weights_through_the_backbone_that_owns_the_packing(
    repo_root,
):
    """R10: ``_tensors`` is what turns a harness row into (maps, boxes) on
    the right device. An extraction that re-derived it would be a second
    definition of what the model is fed."""
    source = (repo_root / "src" / "cleft" / "train" / "graph_cleft.py").read_text(
        encoding="utf-8"
    )
    assert "def region_weights" in source
    assert "forward_from_features_with_weights" in source
    # It goes through _tensors rather than unpacking a second time.
    assert source.count("def region_weights") == 1
    region = source.split("def region_weights")[1].split("\n    def ")[0]
    assert "self._tensors(" in region
    assert "unpack(" not in region
    # Eval mode and no_grad, as predict does -- SR-GNN has two Dropouts.
    assert "self._model.eval()" in region and "torch.no_grad()" in region
    # And a model without the accessor is refused, not silently skipped.
    assert "has no forward_from_features_with_weights" in region


def test_the_fold_models_are_handed_over_only_after_training(repo_root):
    """Called after ``run_cv`` returns and after the count check: before
    that the instances exist but are untrained, and an explanation read off
    an untrained model would be an explanation of nothing."""
    source = (repo_root / "src" / "cleft" / "train" / "graph_cleft.py").read_text(
        encoding="utf-8"
    )
    assert "on_fold_models" in source
    after_cv = source.split("cv = run_cv(")[1]
    assert "if len(trained) != len(fold_order):" in after_cv
    # The count check comes FIRST, so the zip handed over is total.
    assert after_cv.index("len(trained) != len(fold_order)") < after_cv.index(
        "on_fold_models("
    )


def test_the_built_control_is_recorded_apart_from_the_probes_statistic():
    """**Two statistics on two inputs, not a prediction and its failure.**
    The probe's 5.96e-08 was a max difference over four synthetic images;
    the cluster's 1.59e-04 is a between-patient SD over 237 real faces.
    Both land on the same reading, by different routes."""
    record = phase8.THE_RANDOMISED_BACKBONE_COLLAPSES
    built = record["measured_on_the_built_control"]
    assert built["shape"] == (237, 2048, 7, 7)
    assert built["n_parameters_randomised"] == 20_806_952
    assert built["n_borrowed_group_std"] == 40
    assert built["buffers_untouched"] is True
    assert built["between_patient_sd"] == 1.59e-04
    assert "Different statistic, different inputs" in built[
        "not_the_probes_statistic"
    ]
    assert "apply as written" in built["reading"]
    # The probe figure is still there, unrevised -- the two coexist.
    assert record["srgnn"]["randomised_between_patient_max_difference"] == 5.96e-08


def test_n_41_now_states_what_it_governs_and_what_it_does_not():
    """**The record was not wrong; the scope was.** Amendment 8c displays
    phase-licensed artifacts for cohort patients, which is not a per-patient
    claim -- and honest_ar stands untouched, because AR remains the case the
    derivation constrains."""
    record = phase8.PER_FACE_CLAIM_COSTS_N_41
    assert "PASS/FAIL" in record["scope"]
    assert "amendment 8c" in record["scope"]
    assert "display is not a claim" in record["scope"]
    # honest_ar is exactly as first written -- the scope line must not have
    # softened it.
    assert "PHASE-LEVEL licence" in record["honest_ar"]
    assert "how the model sees THIS" in record["honest_ar"]
    # And the arithmetic itself is untouched.
    assert record["closed_form"].endswith("n >= 41")


# --------------------------------------------------------------------------
# the maintainer's three checks, after arm B's run
# --------------------------------------------------------------------------


def test_check_1_the_map_is_not_flat_and_the_weights_are_still_arithmetic():
    """**The map carries a fixed spatial pattern and the region information
    dies downstream** -- so the tautology is not 'blank image in' but
    'indistinguishable regions in', which the symmetry test then makes
    parameter-independent."""
    record = phase8.CHECK_1_UNIFORM_CONTROL_IS_ARITHMETIC
    # [2026-08-15] The map figures are the CLUSTER artifact's, replacing the
    # local reconstruction's; the local 4.28e-03 was a mean-over-channels
    # aggregation of the same quantity and is reconciled in its own field.
    assert record["map"]["per_channel_spatial_sd_median"] == 2.917e-04
    assert record["map"]["between_image_sd_of_image_means"] == 3.826e-09
    assert "NOT spatially flat" in record["map"]["reading"]
    reconciled = record["local_figure_reconciled"]
    assert reconciled["reported_locally"] == 4.28e-03
    assert "MEAN over channels" in reconciled["was"]
    assert "2.917e-04" in reconciled["measured_reconciliation"]
    assert "different aggregation" in reconciled["measured_reconciliation"]

    chain = record["where_region_information_dies"]
    # Each stage loses orders of magnitude, and the real map stays above
    # the randomised one at every stage.
    assert chain["descriptor_spread"]["randomised"] < 1e-6
    assert chain["descriptor_spread"]["real"] > 1e-4
    assert chain["weight_sd_across_26"]["randomised"] == 0.0
    assert "EVERY parameter value" in record["symmetry"]["why_it_must"]
    assert "UNINFORMATIVE BY CONSTRUCTION" in record["supported_reading"]
    # The correction is on the record, and on the reading future runs print.
    assert "too generous" in record["corrects"].lower()
    from cleft import node_weights as nw

    reading = nw._DEGENERACY_READINGS["all"]
    assert "CORRECTED 2026-08-15" in reading
    assert "UNINFORMATIVE" in reading
    # The corrected phrase survives only as a QUOTATION inside the dated
    # correction -- visible as what was corrected, no longer asserted.
    phrase = "strong form of parameter dependence"
    assert reading.count(phrase) == 1
    assert reading.index("CORRECTED 2026-08-15") < reading.index(phrase)
    assert f"'a {phrase}'" in reading
    # Not passed, not unresolved -- and not rescued.
    assert "not rescued" in record["a_vs_b_consequence"]


def test_check_2_the_agreement_statistic_matches_the_task_construction():
    """The record must describe what the task COMPUTES, so the construction
    is re-derived from the task source rather than trusted."""
    from pathlib import Path

    record = phase8.CHECK_2_SEED_AGREEMENT_CONSTRUCTION
    assert "held the patient out" in record["is"]
    assert "whole-image node excluded" in record["is"]
    # 10 seeds -> 45 pairs; arm A's 5 -> 10. The record's pair counts are
    # the binomial ones, not typed guesses.
    assert record["differs_from_arm_a"]["pairs"] == (45, 10)
    assert 45 == 10 * 9 // 2 and 10 == 5 * 4 // 2
    assert record["differs_from_arm_a"]["elements"] == (26, 196)

    source = (
        Path(__file__).resolve().parents[1] / "src" / "cleft" / "run.py"
    ).read_text(encoding="utf-8")
    task = source.split("def task_node_weights")[1].split("\ndef ")[0]
    # The construction: split first (26 only), pairs of seeds, similarity,
    # median per patient, median of medians.
    assert "split_whole_image" in task
    assert "gradcam.similarity(vectors[i], vectors[j])" in task
    assert "median_of_medians" in task
    assert "0.018" not in task  # the value lives in the record, not the code


def test_check_3_seed_fresh_attention_and_both_fourth_check_readings():
    """**0.018 is indistinguishable from the untrained baseline**, so the
    fourth check's npz read decides what it even is -- and both readings
    exist before that number does."""
    record = phase8.CHECK_3_ATTENTION_IS_SEED_FRESH
    assert record["init_policy"]["arm_b_checkpoint_arrays"] is None
    assert record["init_policy"]["same_seed_bitwise_identical"] is True
    assert "6 of 7" in record["init_policy"][
        "attention_tensors_differing_across_seeds"
    ]
    assert "read past its scope" in record["init_policy"][
        "warm_start_sentence_scope"
    ]
    assert "indistinguishable from the untrained baseline" in record[
        "untrained_baseline"
    ]["verdict"]

    fourth = record["fourth_check"]
    assert "before the npz figure is read" in fourth["registered"]
    assert "UNDERDETERMINED" in fourth["reading_if_spread_above_dust"]
    assert "never a ranking" in fourth["reading_if_spread_at_dust"]
    # The contrast survives WITH its confound, and the clinical draft is
    # blocked until the fourth number exists.
    assert "769" in record["what_survives"] and "6.4M" in record["what_survives"]
    assert "BLOCKED" in record["clinical_material"]


# --------------------------------------------------------------------------
# after the fourth check fired
# --------------------------------------------------------------------------


def test_the_fourth_check_fired_dust_and_the_records_agree():
    """The npz read landed inside the untrained band, uniformity includes
    the whole-image node at exactly 1/27, and the two artifacts cross-check."""
    read = phase8.CHECK_3_ATTENTION_IS_SEED_FRESH["fourth_check_read"]
    sd = read["across_region_sd"]
    assert sd["median"] == 4.407e-05
    low, high = read["untrained_band"]
    assert low <= sd["median"] <= high, "the median left the untrained band"
    # The whole distribution is orders of magnitude below a ranking, and
    # the quantiles are ordered -- a pasted figure out of order would mean
    # a transcription error, not a distribution.
    assert sd["min"] <= sd["p5"] <= sd["p25"] <= sd["median"] <= sd["p75"] \
        <= sd["p95"] <= sd["max"]
    assert sd["max"] < 1e-3
    assert abs(1 / 27 - 0.037037) < 1e-6
    assert read["uniformity_is_total"]["whole_image_median"] == 0.037036
    assert "VOID, not weakened" in read["framing"]
    assert "0.0181" in read["cross_check"]
    # The block is lifted for the comparison statement only.
    lifted = phase8.CHECK_3_ATTENTION_IS_SEED_FRESH["clinical_material"]
    assert "LIFTED 2026-08-15" in lifted and "8b and 8c stay untouched" in lifted


def test_the_comparison_statement_is_ranking_versus_no_ranking():
    """**Not stable-versus-unstable.** And the caveat is carried by
    REFERENCE to the one definition, so it cannot drift from the npz's."""
    statement = phase8.A_VS_B_COMPARISON_STATEMENT
    assert "never produced a ranking" in statement["reportable"]
    assert "RANKING versus NO-RANKING" in statement["reportable"]
    assert "not stable versus unstable" in statement["reportable"]
    assert statement["caveat_verbatim"] is phase8.FROZEN_BACKBONE_CAVEAT["text"]
    assert "769" in statement["confound"] and "6.4M" in statement["confound"]
    assert "not SR-GNN's capacity in general" in statement["scope"]
    for phrase in (
        "node voting is seed-unstable",
        "SR-GNN cannot rank regions",
        "intervalled comparison",
    ):
        assert phrase in statement["forbidden"], phrase
    assert statement["numbers"]["arm_b_across_region_sd_median"] == 4.407e-05


def test_the_convergence_candidate_is_labelled_post_hoc():
    """**The maintainer's framing had it pre-registered; it is not, and the
    record says so.** The ceiling half predates the run; the joining of the
    two is after both numbers existed -- a candidate, not a finding."""
    record = phase8.GRAPH_HEAD_CONTRIBUTION_CANDIDATE
    assert "post hoc" in record["registered"]
    assert "before arm B ran" in record["converging_measurements"][
        "branch_3_ceiling"
    ]
    assert "post hoc" in record["ordering_stated"]
    assert "not a finding" in record["ordering_stated"]
    # The bearing evidence is quoted with its own non-claimability intact.
    bears = record["existing_evidence_that_bears_without_settling"]
    assert "0.1340" in bears
    assert "not claimable" in bears
    assert "does not establish it" in bears
    assert record["caveat_verbatim"] is phase8.FROZEN_BACKBONE_CAVEAT["text"]
    # And the ceiling's own record still exists at the figures cited.
    ceiling = phase8.THE_INTERPRETABLE_ARM_IS_CAPPED
    assert ceiling["road_b_reproduced_it"]["difference"] == 0.0002


# --------------------------------------------------------------------------
# 8b: the Grad-CAM variant -- a second method, never a modification
# --------------------------------------------------------------------------


def test_the_variant_differs_from_the_original_in_exactly_the_two_deltas():
    """**Negatives dropped, softmax weighting -- and nothing else.** On
    inputs where no channel importance is negative and the importances are
    equal, the two deltas are both no-ops up to weight scale, so the two
    methods must RANK the tokens identically; on inputs with a dominant
    negative channel, only the variant ignores it."""
    import numpy as np

    from cleft import gradcam

    rng = np.random.default_rng(0)
    acts = rng.random((6, 4))

    # Equal positive importances: softmax(constant) is uniform, mean weight
    # is constant -- both methods aggregate the same channels with flat
    # weights, so the token RANKING agrees exactly.
    grads_flat = np.full((6, 4), 0.5)
    original = gradcam.cam(acts, grads_flat)
    variant = gradcam.cam_softmax(acts, grads_flat)
    assert np.array_equal(np.argsort(original), np.argsort(variant))

    # A strongly negative channel: the original SUBTRACTS its activations;
    # the variant drops it before aggregation. Make that channel dominant
    # for one token and the two methods must disagree about that token.
    grads_neg = np.full((6, 4), 0.5)
    grads_neg[:, 0] = -5.0
    acts_dominant = acts.copy()
    acts_dominant[0, 0] = 50.0
    original = gradcam.cam(acts_dominant, grads_neg)
    variant = gradcam.cam_softmax(acts_dominant, grads_neg)
    assert original[0] == 0.0, "the original's ReLU floors the subtraction"
    assert variant[0] > 0.0, "the variant dropped the negative channel"

    # The variant's weights are a DISTRIBUTION over surviving channels:
    # scaling all gradients by a constant changes cam's scale but not
    # cam_softmax's relative weighting shape.
    v1 = gradcam.cam_softmax(acts, grads_flat)
    v2 = gradcam.cam_softmax(acts, grads_flat * 3.0)
    assert np.allclose(v1, v2), "softmax over equal importances is scale-free"


def test_the_variant_meets_the_same_guards_and_refuses_all_negative():
    import numpy as np
    import pytest as _pytest

    from cleft import gradcam

    acts = np.ones((4, 3))
    # The zero-gradient guard is the SAME code path, same message.
    with _pytest.raises(gradcam.GradCamError, match="channel weights vanish"):
        gradcam.cam_softmax(acts, np.zeros((4, 3)))
    with _pytest.raises(gradcam.GradCamError, match="channel weights vanish"):
        gradcam.cam(acts, np.zeros((4, 3)))
    # All-negative importances: nothing survives the registered drop, and
    # the refusal names what the original would have produced.
    with _pytest.raises(gradcam.GradCamError, match="nothing to aggregate"):
        gradcam.cam_softmax(acts, -np.ones((4, 3)))


def test_the_variant_map_names_its_method_and_shares_the_provenance():
    import numpy as np

    from cleft import gradcam

    rng = np.random.default_rng(1)
    tokens = 196
    acts, grads = rng.random((tokens, 8)), rng.random((tokens, 8))
    original = gradcam.map_for(acts, grads)
    variant = gradcam.map_for_softmax(acts, grads)
    assert variant["method"] == "grad_cam_softmax"
    assert "method" not in original, (
        "the original's return shape is untouched -- the variant is the "
        "one that must name itself"
    )
    for key in ("layer", "resolution_floor_px", "caveat"):
        assert variant[key] == original[key], key
    assert variant["grid"].shape == original["grid"].shape


def test_the_8b_registration_fixes_the_gate_and_commits_both_readings():
    """**Registered before any variant map exists**, gate form taken from
    the adopted criterion rather than the original's inestimable per-patient
    form, both concentration readings committed, and the governing sentence
    verbatim."""
    record = phase8.GRAD_CAM_SOFTMAX_REGISTERED
    assert "before any variant map exists" in record["registered"]
    assert "never the weighting" in record["second_method_because"]
    assert "corrected" in record["second_method_because"]
    assert record["deltas"] == (
        "negatives dropped before aggregation",
        "softmax over the surviving channel importances",
    )
    assert "final ReLU" in record["identical_otherwise"]
    assert record["name_everywhere"] == "grad_cam_softmax"

    gate = record["gate"]
    assert "RANDOMISATION_CRITERION_ADOPTED" in gate["form_fixed_by"]
    assert "NOT" in gate["form_fixed_by"] and "inherited" in gate["form_fixed_by"]
    assert "0.2520" in gate["gate_1"]
    assert "OWN in-run between-patient baseline" in gate["gate_2"]
    assert "UNRESOLVED" in gate["if_unresolved"]
    assert "not rescued" in gate["if_unresolved"]

    readings = record["concentration_readings"]
    assert "may not be read as 'better'" in readings["if_concentrates"]
    assert "never the weighting" in readings["if_diffuse"]
    assert readings["descriptive_never_a_gate"] == "n=15"

    artifacts = record["artifacts"]
    assert "grad_cam_methods.npz" in artifacts["display_scope_always"]
    assert "grad_cam_softmax_maps.npz" in artifacts["claim_scope_gated"]
    assert "untouched" in artifacts["original_maps_recomputed"]
    assert "not\nextraction-shaped" in record["runs_fresh"] or (
        "not extraction-shaped" in record["runs_fresh"].replace("\n", " ")
    )
    assert "clean snapshot" in record["dirty_restart_avoided"]


def test_the_variant_config_is_the_originals_inputs_under_a_new_kind(repo_root):
    """Everything verified in the original config carries; only the kind
    differs -- so the variant cannot silently run on different data."""
    import yaml

    original = yaml.safe_load(
        (repo_root / "configs" / "p8_grad_cam.yaml").read_text(encoding="utf-8")
    )
    variant = yaml.safe_load(
        (repo_root / "configs" / "p8_grad_cam_softmax.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert variant["task"]["kind"] == "grad_cam_softmax"
    assert variant["inputs"] == original["inputs"]
    assert {k: v for k, v in variant["task"].items() if k != "kind"} == (
        {k: v for k, v in original["task"].items() if k != "kind"}
    )
    header = (repo_root / "configs" / "p8_grad_cam_softmax.yaml").read_text(
        encoding="utf-8"
    ).split("schema_version")[0]
    assert "never the weighting" in header
    assert "SECOND METHOD" in header
    assert "OWN" in header and "never inherited" in header


def test_the_refusal_handling_is_registered_with_the_pod_numbers():
    """**Per-cell recording, operative figures from the real cohort, both
    boundary rules registered though unreached** -- and the both-numbers
    rule binding every report."""
    record = phase8.GRAD_CAM_SOFTMAX_REFUSAL_HANDLING
    pod = record["operative_numbers_pod_probe"]
    assert pod["cells_refusing"] == "19 of 75"
    assert sum(pod["per_patient"]) == 19
    assert len(pod["per_patient"]) == 15
    assert pod["patients_losing_all_five"] == 0
    assert pod["gate_2_n"].startswith("15 of 15")
    assert pod["survivor_top10"]["median"] == 1.000
    assert pod["original_top10"]["median"] == 0.195
    assert pod["distribution_overlap"] == "none"
    assert "patient 120" in pod["crash_point"]
    # The three single-cell survivors are named as UNDEFINED-agreement.
    assert pod["per_patient"].count(4) == 3
    assert "UNDEFINED" in pod["single_cell_survivors"]

    assert "randomisation finals" in record["handling"]
    unreached = record["registered_though_unreached"]
    assert "n(n-1)/2 >= 20" in unreached["unestimable_boundary"]
    assert "by another road" in unreached["unestimable_boundary"]
    # [2026-08-15, the rerun] The note gained its dated correction: the
    # real-map side stayed unreached, the FINALS side was reached.
    assert unreached["note"].startswith("neither rule was needed at 15/15")
    assert "CORRECTED 2026-08-15" in unreached["note"]
    assert "all 15 randomised finals" in unreached["note"]
    assert "never\nreported apart" in record["both_numbers_rule"] or (
        "never reported apart" in record["both_numbers_rule"].replace("\n", " ")
    )
    assert "concentration-to-destruction" in record["both_numbers_rule"]
    assert "cam_softmax is untouched" in record["deltas_unmoved"]


def test_the_flatness_finding_and_its_specification_boundary():
    """**Softmax max weight 0.003 over a median 388 channels against a
    uniform 0.0026** -- the registered variant erases the weighting, the
    synthetic study is marked synthetic, and a tempered softmax is a NEW
    method by registration."""
    record = phase8.SOFTMAX_FLATTENS_THE_WEIGHTING
    assert record["max_softmax_weight"] == 0.003
    assert record["median_surviving_channels"] == 388
    assert record["uniform_reference"] == 0.0026
    assert "uniform to three decimals" in record["so"]
    assert "2 of 75" in record["sharpening_account_refuted"]
    assert "SYNTHETIC" in record["synthetic_study"]["marked"]
    assert record["synthetic_study"]["cells_refusing"] == "44 of 75"
    assert "THIRD delta" in record["specification_boundary"]
    assert "never a fix" in record["specification_boundary"]
    assert "flatness included" in record["specification_boundary"]


def test_the_task_implements_the_registered_handling(repo_root):
    """The handling is CODE, not narration: the per-cell catch, the
    surviving-patient gate, the finals rule, and the derived estimability
    floor all exist in the task source."""
    source = (repo_root / "src" / "cleft" / "run.py").read_text(encoding="utf-8")
    task = source.split("def task_grad_cam_softmax")[1].split("\ndef ")[0]
    # The original computes FIRST, so the shared wrong-layer guard stays
    # fatal on the original's path; the variant's refusal is the datum.
    assert task.index("gradcam.map_for(acts[0], grads[0])") < task.index(
        "gradcam.map_for_softmax(acts[0], grads[0])"
    )
    assert '"all_zero": True' in task
    assert "except gradcam.GradCamError" in task
    # Patient map over surviving cells; agreement undefined below two.
    assert "fewer than two surviving cells" in task
    # The finals apply the same rule -- excluded, never imputed.
    assert "final_refusals" in task and "never imputed" in task
    # The derived floor, both places it can bite, with the sentence.
    assert task.count("UNESTIMABLE_BELOW = 7") == 1
    assert task.count("by another road") >= 1
    # The both-numbers sentence is the log line and the artifact field.
    assert "both_numbers_sentence" in task
    # The surface stamps refusals beside the grids.
    assert "refused_cells_per_patient" in task


def test_the_unestimable_verdict_is_a_completion_path_not_a_crash(repo_root):
    """**The registration says declared and stamped; a raise is neither.**
    The floor branch sets a verdict and the run finalizes -- the raise
    survives ONLY on the UNRESOLVED path, which is a gate that ran."""
    source = (repo_root / "src" / "cleft" / "run.py").read_text(encoding="utf-8")
    task = source.split("def task_grad_cam_softmax")[1].split("\ndef ")[0]
    # The two floor checks assign a reason rather than raising.
    floor = task.split("UNESTIMABLE_BELOW = 7")[1]
    assert "unestimable = (" in task
    assert 'raise ValueError(\n            f"only' not in task, (
        "a floor branch still raises; the registered verdict must complete"
    )
    # Three-state verdict, the reason stamped in metrics and on the surface.
    assert 'gate_verdict = "UNESTIMABLE"' in task
    assert '"verdict": gate_verdict' in task
    assert 'f"UNESTIMABLE -- {unestimable}"' in task
    # The unestimable completion logs and does NOT raise; UNRESOLVED still
    # raises last, the arm-A precedent for a gate that ran and failed.
    unest_branch = task.split("elif unestimable is not None:")[1].split(
        "    else:"
    )[0]
    assert "raise" not in unest_branch
    assert "the run \\n" not in unest_branch  # formatting guard, cheap
    unresolved_branch = task.split("elif unestimable is not None:")[1]
    assert "raise ValueError(" in unresolved_branch
    # With no finals, separability is not computed on empty data.
    assert "separable, resolves, percentiles = None, False, []" in task


def test_the_outcome_and_the_existence_finding_are_recorded():
    """The boundary fired as registered; the delivery defect is recorded
    with its fix; the existence contrast is a finding and NOT a pass."""
    outcome = phase8.GRAD_CAM_SOFTMAX_REFUSAL_HANDLING["outcome"]
    assert "19/75" in outcome["trained_cells"]
    assert "0.5305" in outcome["trained_cells"]
    assert "+0.7532" in outcome["trained_cells"]
    assert outcome["randomised_finals"] == "ALL 15 refused -- 0 defined, floor 7"
    assert "by another road" in outcome["verdict"]
    assert "exactly as pre-registered" in outcome["verdict"]
    assert "Same altitude class as the first crash" in outcome["delivery_defect"]
    assert "completion path" in outcome["delivery_defect"]
    assert "correctly never exists" in outcome["delivery_defect"]

    finding = phase8.MAP_EXISTENCE_DEPENDS_ON_TRAINED_WEIGHTS
    assert "all 15 finals refuse" in finding["at_randomised_weights"]
    assert "SOFTMAX_FLATTENS_THE_WEIGHTING" in finding["mechanism"]
    assert "existence is prior to rank similarity" in finding["is"]
    # The temptation is closed by name: the contrast is NOT a pass, and an
    # existence criterion is a FUTURE registration, not this run's.
    assert finding["is_not"].startswith("a pass")
    assert "corrected-twice move" in finding["is_not"]
    assert "FUTURE run" in finding["future_use"]


def test_phase_8b_closes_on_the_three_findings_with_the_contrast_intact():
    """**Unpublished by gate; the record IS the findings** -- each of the
    three exists as the record the closing names, and the headline contrast
    carries all three positions unchanged."""
    closing = phase8.PHASE_8B_CLOSING
    assert "2cce9625" in closing["closed"]
    assert "one attempt" in closing["closed"]
    assert "UNPUBLISHED BY GATE" in closing["verdict"]
    figures = closing["run_figures"]
    assert figures["refusal_ledger"].startswith("19/75")
    assert "0.5305" in figures["variant_baseline"]
    assert "105 pairs" in figures["variant_baseline"]
    assert "+0.7532" in figures["concentration"]
    assert "both stamps" in figures["methods_npz"]
    assert "correctly never written" in figures["claim_npz"]

    # The three named findings all exist as records; no dangling name.
    assert phase8.SOFTMAX_FLATTENS_THE_WEIGHTING["max_softmax_weight"] == 0.003
    assert "concentration-to-destruction" in closing["the_three_findings"][1]
    assert phase8.MAP_EXISTENCE_DEPENDS_ON_TRAINED_WEIGHTS["is_not"].startswith(
        "a pass"
    )
    # The contrast: three positions, none moved by 8b.
    contrast = closing["headline_contrast_unchanged"]
    assert "0.659" in contrast
    assert "never ranked" in contrast
    assert "cannot be assessed by" in contrast
    # The meeting line: two respects, scale interaction, new-method door.
    assert "exactly its two" in closing["for_the_meeting"]
    assert "SCALE INTERACTION" in closing["for_the_meeting"]
    assert "NEW" in closing["for_the_meeting"]
    assert closing["still_untouched"] == ("8c", "t-SNE")
    assert "nothing_claimable" in closing and "claimable" not in set(closing)


def test_tsne_blocked_and_scut_audit_records():
    """2026-08-15: the two end-of-phase audits. The t-SNE build is blocked
    on exactly the two choices the registration never fixed; the SCUT
    animation audit resolved frames-from-reruns and the SHAREABLE tier;
    the two maintainer flags are record-only."""
    blocked = phase8.TSNE_BUILD_BLOCKED_ON_TWO_CHOICES
    assert len(blocked["not_fixed"]) == 2
    assert "which embeddings" in blocked["not_fixed"][0]
    assert "class3" in blocked["not_fixed"][1]
    assert "(5, 30)" in blocked["fixed"] and "1337" in blocked["fixed"]
    # A proposal awaiting sign-off is not an adoption.
    assert "proposed_for_sign_off" in blocked and "adopted" not in set(blocked)

    audit = phase8.SCUT_ANIMATION_AUDIT
    assert audit["a_frames_from"].startswith("RERUNS")
    assert "no per-epoch archive" in audit["a_frames_from"]
    assert audit["cells"] == (
        "p6_pretrain_vit_b16_original",
        "p6_pretrain_vit_b16_masked_g1",
    )
    assert "SAME source faces" in audit["b_faces"]
    assert "31 honest frames" in audit["c_axis"]
    assert audit["governance"].startswith("SHAREABLE, not cluster-bound")
    assert audit["status"] == "AUDITED; build not started (maintainer's stop)"

    flags = phase8.FLAGGED_FOR_LATER
    assert "no action" in flags["second_beauty_dataset"]
    assert flags["road_b_branch_2"] == "stays parked, unchanged"


def test_tsne_resolution_and_companion_spec():
    """2026-08-15: the maintainer resolved the two blocked choices; the spec
    is fully committed BEFORE any number exists, baselines included."""
    assert phase8.TSNE_BUILD_BLOCKED_ON_TWO_CHOICES["resolved"].startswith(
        "2026-08-15"
    )
    spec = phase8.TSNE_COMPANION
    assert spec["k"] == 5
    assert spec["expected_class_counts"] == (88, 119, 30)
    assert spec["embeddings_set_name"] == "vit_b16__imagenet__g1"
    assert spec["classes"] == "class3"
    assert spec["n_boot"] == 10000 and spec["bootstrap_seed"] == 1337
    assert "refuses a missing companion" in spec["display_rule"]
    assert "chance" in spec["baselines"] and "majority" in spec["baselines"]


def test_knn_companion_separates_and_breaks_ties_as_committed():
    """Two clean clusters give accuracy 1.0; a constructed 2-2 tie at k=5
    resolves to the class owning the nearest neighbour -- the committed
    rule, not an accident of argsort."""
    import numpy as np

    rng = np.random.default_rng(0)
    a = rng.normal(0, 0.05, size=(20, 4)) + np.array([10.0, 0, 0, 0])
    b = rng.normal(0, 0.05, size=(20, 4)) - np.array([10.0, 0, 0, 0])
    features = np.vstack([a, b])
    classes = np.array([1] * 20 + [2] * 20)
    accuracy, correct, _ = phase8.knn_loo_accuracy(features, classes, 5)
    assert accuracy == 1.0 and correct.sum() == 40

    # From point 0 the 5 nearest others vote 1,1,2,2,3: the {1,2} tie is
    # broken by the nearest neighbour, whose class is 1.
    line = np.array([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]])
    line_classes = np.array([9, 1, 1, 2, 2, 3])
    _, _, voted = phase8.knn_loo_accuracy(line, line_classes, 5)
    assert voted[0] == 1

    with pytest.raises(ValueError):
        phase8.knn_loo_accuracy(line, line_classes, 6)


def test_bootstrap_interval_is_deterministic_and_brackets_the_mean():
    import numpy as np

    correct = np.array([1.0] * 70 + [0.0] * 30)
    one = phase8.bootstrap_interval(correct, n_boot=2000, seed=1337)
    two = phase8.bootstrap_interval(correct, n_boot=2000, seed=1337)
    assert one == two
    assert one[0] < correct.mean() < one[1]
    assert 0.0 <= one[0] and one[1] <= 1.0


def test_tsne_embed_is_deterministic_with_pinned_parameters():
    import numpy as np

    rng = np.random.default_rng(1)
    features = rng.normal(size=(40, 8))
    one = phase8.tsne_embed(features, perplexity=5, seed=1337)
    two = phase8.tsne_embed(features, perplexity=5, seed=1337)
    assert one.shape == (40, 2)
    assert np.array_equal(one, two)


def test_scut_face_sample_is_seeded_stratified_and_order_free():
    """The registered rule reads only (stems, scores, seed): insertion
    order cannot move it, ten distinct faces come out, and each score
    quintile contributes exactly two -- the paired-by-construction
    property both variants rely on."""
    import numpy as np

    from cleft.run import phase8_scut_face_sample

    rng = np.random.default_rng(3)
    stems = [f"F{i:03d}" for i in range(50)]
    scores = rng.uniform(1.0, 5.0, size=50)
    labels = dict(zip(stems, scores))
    one = phase8_scut_face_sample(labels)
    two = phase8_scut_face_sample(
        {stem: labels[stem] for stem in reversed(stems)}
    )
    assert one == two
    assert len(one["stems"]) == 10 and len(set(one["stems"])) == 10
    ordered = np.sort(scores)
    strata = np.searchsorted(ordered, np.array(one["scores"])) // 10
    assert sorted(strata.tolist()) == [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]


def test_scut_animation_built_record_corrects_its_own_audit():
    """The build's two corrections are dated in place on the audit AND
    carried in the built record: the byte gate is dead by the project's
    own measurement, and epoch 0 has no map."""
    assert phase8.SCUT_ANIMATION_AUDIT["corrected_at_build"].startswith(
        "2026-08-15"
    )
    built = phase8.SCUT_ANIMATION_BUILT
    assert built["self_gate"]["band"] == 0.06
    assert "0.7893/0.8120/0.8194" in built["corrections"]["byte_gate"]
    assert "photograph" in built["corrections"]["epoch_zero"]
    assert built["self_gate"]["byte_identity"].startswith("reported")
    assert built["self_gate"]["out_of_band"].startswith("raises LAST")
    assert built["tier"] == "SHAREABLE throughout"
    assert built["configs"] == (
        "p8_scut_anim_original.yaml", "p8_scut_anim_masked_g1.yaml",
    )
    faces = phase8.SCUT_ANIMATION_FACES
    assert (faces["n_strata"], faces["per_stratum"], faces["seed"]) == (5, 2, 1337)


def test_the_three_new_configs_are_fully_declared_and_copied():
    """The t-SNE config points at the registered subject; each SCUT config
    copies its cell's training block verbatim and carries verified hashes
    for the init and the shipped deliverable -- launch-ready, no declare
    pass owed."""
    from pathlib import Path

    import yaml

    repo = Path(__file__).resolve().parents[1]

    tsne = yaml.safe_load(
        (repo / "configs" / "p8_tsne.yaml").read_text(encoding="utf-8")
    )
    assert tsne["task"]["kind"] == "tsne"
    by_name = {e["name"]: e for e in tsne["inputs"]}
    assert by_name["embeddings"]["path"].endswith(
        phase8.TSNE_COMPANION["embeddings_set_name"]
    )
    assert all(set(e["rollup_sha256"]) != {"0"} for e in tsne["inputs"])

    for variant in ("original", "masked_g1"):
        config = yaml.safe_load((
            repo / "configs" / f"p8_scut_anim_{variant}.yaml"
        ).read_text(encoding="utf-8"))
        cell = yaml.safe_load((
            repo / "configs" / f"p6_pretrain_vit_b16_{variant}.yaml"
        ).read_text(encoding="utf-8"))
        for key in ("epochs", "learning_rate", "weight_decay", "batch_size",
                    "expect_train", "expect_test", "monitor", "deterministic",
                    "inner_val_frac", "backbone", "source", "region_scheme"):
            assert config["task"][key] == cell["task"][key], (variant, key)
        names = [e["name"] for e in config["inputs"]]
        assert ("masked_scut" in names) == (variant != "original")
        entries = {e["name"]: e for e in config["inputs"]}
        assert entries["shipped_pretrained"]["path"].endswith("pretrained.npz")
        assert set(entries["shipped_pretrained"]["rollup_sha256"]) != {"0"}
        assert set(entries["pretrained_init"]["rollup_sha256"]) != {"0"}


def test_scut_animation_caption_rule_is_registered_before_launch():
    """2026-08-15, the maintainer's addition on accepting the corrections:
    the animation says what it is, the endpoint comparison shows its seam,
    and a difference is shown with its number -- never smoothed."""
    rule = phase8.SCUT_ANIMATION_CAPTION_RULE
    assert rule["registered"].startswith("2026-08-15, before launch")
    assert "not the run that produced the shipped weights" in (
        rule["recipe_sentence"]
    )
    assert "never smoothed or hidden" in rule["comparison_sentence"]
    assert "caption-less" in rule["burned"]
    assert "measured largest per-cell map difference" in rule["beside"]
    assert rule["never"] == "smoothed, hidden, or averaged into one map"
    assert phase8.SCUT_ANIMATION_BUILT["caption_rule"].startswith(
        "SCUT_ANIMATION_CAPTION_RULE"
    )


def test_scut_capture_cell_treats_the_refusal_as_a_datum():
    """The capture path walked through a refusing cell (the first launch's
    crash, 2026-08-16): the method's own GradCamError becomes a photograph
    frame with the measured-refusal caption and status all_zero; a healthy
    cell composites; a zero head is the registered init state, not a
    refusal; and anything that is not the method refusing propagates."""
    import numpy as np

    from cleft import gradcam, phase8c
    from cleft.run import phase8_scut_capture_cell

    rng = np.random.default_rng(7)
    face = rng.integers(0, 255, size=(224, 224, 3)).astype(np.uint8)
    weights = np.ones(8)
    healthy_grid = rng.uniform(0, 1, size=(14, 14))

    grid, image, status = phase8_scut_capture_cell(
        face, weights, lambda: healthy_grid, "epoch 7 of 30"
    )
    assert status is None and grid is healthy_grid
    assert np.array_equal(
        image, phase8c.frame(face, healthy_grid, axis_label="epoch 7 of 30")
    )

    def refuse():
        raise gradcam.GradCamError("all values are zero after the ReLU")

    grid, image, status = phase8_scut_capture_cell(
        face, weights, refuse, "epoch 2 of 30"
    )
    assert status == "all_zero" and grid is None
    assert np.array_equal(image, phase8c.frame(
        face, None,
        axis_label="epoch 2 of 30 -- no map: every cell zero after the "
        "ReLU (a measured refusal, recorded as a datum)",
    ))

    grid, image, status = phase8_scut_capture_cell(
        face, np.zeros(8), None, "epoch 0 of 30"
    )
    assert status == "zero_head" and grid is None
    assert np.array_equal(
        image, phase8c.frame(face, None, axis_label="epoch 0 of 30")
    )

    def fault():
        raise ValueError("not the method refusing")

    with pytest.raises(ValueError, match="not the method refusing"):
        phase8_scut_capture_cell(face, weights, fault, "epoch 3 of 30")


def test_scut_first_launch_records_and_claim_once_pattern():
    """Both launch defects recorded with their salvage verdicts, and the
    fixed claim pattern pinned at the source: pretrained.npz is claimed
    exactly once and read back through the variable; the assembly reads
    frames through ctx.run_dir, never a second claim."""
    import inspect

    from cleft import run as run_module

    handling = phase8.SCUT_ANIMATION_REFUSAL_HANDLING
    assert handling["registered"].startswith("2026-08-16")
    assert "EARLY" in handling["mechanism"]
    assert "photograph" in handling["handling"]
    assert "stay fatal" in handling["catch_scope"]
    assert "raise LAST" in handling["boundary"]

    launch = phase8.SCUT_ANIMATION_FIRST_LAUNCH
    assert launch["salvage"]["original"].startswith("RESUMES")
    assert launch["salvage"]["masked_g1"].startswith("RERUNS CLEAN")
    assert "duplicate-claim" not in launch["defect_1"]
    assert "claim-once-reuse" in launch["defect_2"]
    assert phase8.SCUT_ANIMATION_BUILT["refusal_handling"].startswith(
        "SCUT_ANIMATION_REFUSAL_HANDLING"
    )

    source = inspect.getsource(run_module.task_scut_animation)
    assert source.count("ctx.path(pretrain.PRETRAINED_NAME") == 1
    assert "pretrain.ckpt.load(pretrained_path)" in source
    assert 'ctx.path(f"scut_anim_{stem}_e' not in source
    assert 'ctx.run_dir / f"scut_anim_{stem}_e' in source


def test_phase_8_closing_pins_every_item_and_every_figure():
    """2026-08-16: the phase closes. The citable SCUT runs, both
    self-gate figures, the void first launch, the t-SNE companion's
    below-majority reading, the two flagged follow-ups, and the five
    items of the phase-wide closing -- all pinned."""
    scut = phase8.SCUT_ANIMATION_CLOSING
    assert scut["closed"].startswith("2026-08-16")
    assert "PASSED" in scut["closed"]
    citable = scut["citable_runs"]
    assert citable["sha"] == "a2f7c8c3"
    gates = citable["self_gates"]
    assert gates["original"] == {
        "shipped": 0.8914, "rerun": 0.8937, "delta": 0.0023,
        "verdict": "WITHIN_RECORDED_BAND",
    }
    assert gates["masked_g1"]["shipped"] == 0.7893
    assert gates["masked_g1"]["rerun"] == 0.7639
    assert gates["masked_g1"]["delta"] == 0.0254
    assert gates["masked_g1"]["verdict"] == "WITHIN_RECORDED_BAND"
    assert gates["band"] == phase8.SCUT_ANIMATION_BUILT["self_gate"]["band"]
    assert citable["frames_per_face"] == 32
    assert citable["refusals"].startswith("zero")
    assert citable["byte_identical"].startswith("False")
    assert scut["void_runs"]["sha"] == "e40f09de"
    assert "superseded" in scut["void_runs"]["status"]
    assert "nothing_claimable" in scut

    retry = phase8.RETRY_LIMIT_IS_NOT_HOLDING
    assert retry["status"].startswith("OPEN")
    assert "4-7 attempts" in retry["observed"]
    assert "scut_anim_<stem>.png" in (
        scut["flagged_not_fixed"]["filename_collision"]
    )

    tsne = phase8.TSNE_CLOSING
    assert tsne["companion"]["accuracy"] == 0.409
    assert tsne["companion"]["interval_95"] == (0.346, 0.473)
    assert tsne["companion"]["majority"] == 0.502
    assert "worse" in tsne["companion"]["reading"]
    assert tsne["is_evidence"] is False

    closing = phase8.PHASE_8_COMPLETE
    assert closing["complete"] is True
    # The 2026-08-04 partial closing stands unchanged, with only the dated
    # completion pointer added -- never silently rewritten.
    assert phase8.PHASE_8_CLOSING["completed"].startswith("2026-08-16")
    assert phase8.PHASE_8_CLOSING["status"].startswith("PARTIALLY MET")
    assert "never ran" in closing["exit_criteria_resolved"]["4_node_weights_b_and_c"]
    assert set(closing["items"]) == {
        "node_weights", "8b", "8c", "tsne", "scut_animation",
    }
    assert "DUST" in closing["items"]["node_weights"]
    assert "0.659" in closing["items"]["node_weights"]
    assert "UNESTIMABLE" in closing["items"]["8b"]
    assert "clinical HTML" in closing["items"]["8c"]
    assert "below the majority" in closing["items"]["tsne"]
    assert "RETRY_LIMIT_IS_NOT_HOLDING" in closing["flagged_follow_ups"]
