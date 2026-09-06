"""Phase 25: the registration, and every figure checked against its home.

**The counter-evidence is the phase's own, so it is held to the same
standard as its case**: each of the eleven is asserted against the record
that measured it, and the two corrections to the restate's figures are
pinned so they cannot drift back.
"""

from __future__ import annotations

import pathlib

import pytest

from cleft import phase25

REPO = pathlib.Path(__file__).resolve().parents[1]


def _flat(text: str) -> str:
    return " ".join(text.split())


# --------------------------------------------------------------------------
# the reckoning
# --------------------------------------------------------------------------


def test_the_reckoning_leads_with_the_counter_evidence():
    record = phase25.PHASE_25_RECKONING
    assert record["the_ruling"].startswith("RUN IT")
    assert record["tag"].startswith("[REGISTERED]")

    tempering = _flat(record["the_amendment_argued_against_itself"])
    assert "sequenced this LAST on its own weak prior" in tempering
    assert "That tempering stands and is not softened here" in tempering

    # The prior is stronger now than at scheduling, and it says so.
    later = _flat(record["and_it_was_written_before_phase_22"])
    assert "the motivation predates the objective result" in later
    assert "within 0.003" in later
    assert "made against that, not around it" in later

    ground = _flat(record["the_ground_for_running_rather_than_conceding"])
    assert "worth twenty minutes" in ground
    assert "a gap the write-up CARRIES rather than an answer" in ground
    assert "A null that was measured is a different object" in ground


def test_the_amendment_is_quoted_as_the_amendment_wrote_it():
    from cleft import phase21

    source = phase21.PHASE_SEQUENCE_EXTENDED_7[
        "phase_25_foundation_model_features"
    ]
    assert source["status"] == "SCHEDULED, NOT REGISTERED"
    original = _flat(source["why_last_of_the_compute_phases"])
    quoted = _flat(
        phase25.PHASE_25_RECKONING["the_amendment_argued_against_itself"]
    )
    for fragment in ("feature source has moved little",
                     "MEBeauty 0.2537 against ImageNet 0.2520",
                     "the most expensive intervention with the weakest prior"):
        assert fragment in original, fragment
        assert fragment in quoted, fragment


def test_self_supervised_features_really_have_never_been_tried():
    """The one ground the phase stands on, verified rather than asserted."""
    hits = []
    for folder, pattern in ((REPO / "src", "*.py"), (REPO / "configs", "*.yaml")):
        for path in folder.rglob(pattern):
            if path.name in ("phase25.py", "test_phase25.py"):
                continue
            if path.name.startswith("p25_"):
                continue          # Phase 25's OWN configs, being built
            body = path.read_text(encoding="utf-8").lower()
            for token in ("dinov2", "self-supervised", "foundation model"):
                if token in body:
                    hits.append((path.name, token))
    allowed = {
        "phase21.py",          # the amendment that scheduled the phase
        "factory.py",          # the two registry entries
        "schema.py",           # the two init and backbone choices
        "extract.py",          # FACTORY_PRETRAINED_INITS
        "phase3.py",           # the artifact-only backbone list
        "embeddings.py",       # FOUNDATION_INITS, after the cluster defect
    }
    unexpected = sorted({name for name, _ in hits} - allowed)
    assert unexpected == [], unexpected
    assert ("phase21.py", "dinov2") in hits
    from cleft import phase21

    assert "DINOv2" in phase21.PHASE_SEQUENCE_EXTENDED_7[
        "phase_25_foundation_model_features"
    ]["what"]


# --------------------------------------------------------------------------
# the eleven, each against its home
# --------------------------------------------------------------------------


def test_the_eleven_are_eleven_and_each_matches_its_record():
    from cleft import ladder, phase15, results_ledger, roadb

    eleven = phase25.THE_ELEVEN_AND_THE_SHARPEST["the_eleven"]
    assert len(eleven) == 11
    assert sorted(int(k.split("_")[0]) for k in eleven) == list(range(1, 12))

    # Pretraining source (phase15).
    cells = _flat(phase15.THE_SOURCE_COMPARISON["the_cells"]) if hasattr(
        phase15, "THE_SOURCE_COMPARISON"
    ) else (REPO / "src" / "cleft" / "phase15.py").read_text(encoding="utf-8")
    for token in ("0.2537", "0.1952", "-0.0568"):
        assert token in cells, token

    # Architecture (ladder 7D).
    arms = ladder.PHASE_7D_OBSERVED["arms"]
    assert arms["vit_b32"]["mean"] == 0.2519
    assert arms["vit_b8"]["mean"] == 0.0432
    assert arms["mvitv2_b"]["mean"] == 0.1844
    assert "0.2519" in eleven["6_vit_b32"] and "0.0432" in eleven["7_vit_b8"]
    assert "0.1844" in eleven["8_mvitv2_b"]

    # Resolution (roadb).
    endpoints = roadb.RESOLUTION_TRENDS if hasattr(
        roadb, "RESOLUTION_TRENDS"
    ) else None
    body = (REPO / "src" / "cleft" / "roadb.py").read_text(encoding="utf-8")
    assert "0.0894" in body and "0.0937" in body and "-0.1583" in body
    assert "0.0894" in eleven["10_resolution"]
    assert "-13.29" in eleven["10_resolution"]

    # Swin is ledger row 1, withdrawn.
    assert results_ledger.ENTRIES[1]["status"] == "WITHDRAWN"
    assert "0.2092" in str(results_ledger.ENTRIES[1]["claim"])
    assert "WITHDRAWN" in eleven["9_swin_b"]


def test_moved_little_is_not_claimed_of_all_eleven():
    """Several moved it enormously -- downward. The record says so."""
    accurate = _flat(
        phase25.THE_ELEVEN_AND_THE_SHARPEST["the_accurate_form_of_the_claim"]
    )
    assert "NONE beat 0.2520 claimably" in accurate
    assert "Not 'all eleven moved it little'" in accurate
    assert "moved it enormously, downward" in accurate


def test_the_sharpest_contrast_is_one_factor_on_both_sides():
    sharpest = _flat(
        phase25.THE_ELEVEN_AND_THE_SHARPEST["the_sharpest_ONE_FACTOR_contrast"]
    )
    assert "0.2519" in sharpest and "0.0432" in sharpest
    assert "0.2087" in sharpest and "0.0017" in sharpest
    assert "Both sides are one-factor comparisons" in sharpest
    # The span really is what the record's arms give.
    from cleft import ladder

    arms = ladder.PHASE_7D_OBSERVED["arms"]
    assert round(arms["vit_b32"]["mean"] - arms["vit_b8"]["mean"], 4) == 0.2087


def test_the_srgnn_correction_is_pinned_against_the_ladders_own_words():
    from cleft import ladder

    pair = ladder.TRADE_OFF_PAIR
    assert pair["interpretable_arm"]["recorded_pcc"] == 0.1719
    assert pair["interpretable_arm"]["recorded_seeds"] == 10
    assert pair["delta_if_recorded_means_hold"] == 0.0801
    assert pair["result"]["srgnn_paired_mean"] == 0.1884
    assert pair["result"]["delta"] == 0.0637
    assert pair["result"]["claimable"] is False
    assert pair["result"]["n_excluding_zero"] == 0
    assert "not a one-factor comparison" in _flat(pair["not_in_the_audit"])

    correction = _flat(phase25.THE_ELEVEN_AND_THE_SHARPEST[
        "CORRECTION_the_srgnn_contrast_proposed_is_not_one_factor"
    ])
    assert "It is not one factor" in correction
    assert "0.1719 is a TEN-SEED mean" in correction
    assert "0.1884" in correction and "+0.0637" in correction
    assert "claimable: False" in correction
    assert "superseded by the measurement" in correction

    survives = _flat(
        phase25.THE_ELEVEN_AND_THE_SHARPEST["what_survives_the_correction"]
    )
    assert "not weakened in substance" in survives


# --------------------------------------------------------------------------
# attribution
# --------------------------------------------------------------------------


def test_the_attribution_bound_is_registered_before_any_number():
    record = phase25.THE_ATTRIBUTION_PROBLEM
    may = _flat(record["what_the_phase_MAY_claim"])
    may_not = _flat(record["what_the_phase_MAY_NOT_claim"])
    assert "these features are better/worse/indistinguishable" in may
    assert "self-supervision is what did it" in may_not
    assert "registered BEFORE the number so it cannot be renegotiated" in may_not
    assert "two factors, not one" in _flat(record["what_changes_at_once"])


def test_the_same_encoder_control_is_measured_unavailable():
    """DINOv2 is patch-14 only, and patch-14 base is DINOv2 only."""
    timm = pytest.importorskip("timm")

    dinov2 = timm.list_models("*dinov2*", pretrained=True)
    assert len(dinov2) == 8
    assert all("patch14" in name for name in dinov2), dinov2
    assert timm.list_models("*dinov2*patch16*", pretrained=True) == []
    base14 = timm.list_models("vit_base_patch14*", pretrained=True)
    assert sorted(base14) == [
        "vit_base_patch14_dinov2.lvd142m",
        "vit_base_patch14_reg4_dinov2.lvd142m",
    ]

    record = _flat(phase25.THE_ATTRIBUTION_PROBLEM[
        "the_same_encoder_control_IS_NOT_AVAILABLE_and_this_is_MEASURED"
    ])
    assert "eight entries, ALL patch14" in record
    assert "the objective cannot be isolated in either direction" in record

    accepted = _flat(phase25.THE_ATTRIBUTION_PROBLEM[
        "therefore_the_unattributability_is_a_BOUND_THE_PHASE_ACCEPTS"
    ])
    assert "No second arm is registered" in accepted
    assert "no extra extraction is proposed" in accepted


def test_the_timm_version_gap_is_declared_as_blocking():
    gap = _flat(phase25.THE_ATTRIBUTION_PROBLEM[
        "the_measurement_was_made_on_THIS_machine_at_a_DIFFERENT_TIMM"
    ])
    assert "timm 1.0.27 here; the cluster is pinned at 1.0.7" in gap
    assert "BLOCKING READ" in gap
    blocking = _flat(
        phase25.SETTINGS_TO_DECLARE["0_BLOCKING_does_the_pinned_timm_carry_dinov2"]
    )
    assert "a decision, not a detail" in blocking
    # The pin really is 1.0.7 in the record.
    assert "timm 1.0.7" in (
        REPO / "src" / "cleft" / "models" / "factory.py"
    ).read_text(encoding="utf-8")


# --------------------------------------------------------------------------
# the arm
# --------------------------------------------------------------------------


def test_the_pooled_features_correction_names_what_actually_exists():
    from cleft import embeddings
    from cleft.config import schema

    correction = _flat(phase25.THE_ARM_REGISTERED[
        "CORRECTION_there_is_no_pooled_features_task"
    ])
    assert "NO SUCH TASK EXISTS" in correction

    # There is no such kind, and the real names are as the record says.
    assert "pooled_features" not in schema.TASK_SPECS
    assert "extract_embeddings" in schema.TASK_SPECS
    assert "pooled" in embeddings.EMBEDDING_KINDS
    assert embeddings.KIND_FOR_BACKBONE_KIND == {
        "transformer": "pooled", "graph": "feature_map"
    }
    # And the registry really is closed against DINOv2.
    from cleft.models.factory import BACKBONES

    # [UPDATED 2026-09-02] The ruling added the two entries; what the
    # check holds is that the extraction TASK is still the shipped one.
    assert len(BACKBONES) == 9


def test_the_224_shortcut_would_kill_the_run_and_the_fix_is_verified():
    torch = pytest.importorskip("torch")
    timm = pytest.importorskip("timm")

    from cleft.models.factory import timm_kwargs_for

    # The shortcut: no kwargs at 224, which is right for all seven.
    assert timm_kwargs_for("vit_base_patch14_dinov2", (224, 224)) == {}

    model = timm.create_model(
        "vit_base_patch14_dinov2", pretrained=False, num_classes=0
    )
    with pytest.raises(AssertionError, match="doesn't match model"):
        with torch.no_grad():
            model(torch.zeros(1, 3, 224, 224))

    # The fix, and the width it returns.
    fixed = timm.create_model(
        "vit_base_patch14_dinov2", pretrained=False, num_classes=0,
        dynamic_img_size=True,
    )
    with torch.no_grad():
        assert tuple(fixed(torch.zeros(1, 3, 224, 224)).shape) == (1, 768)

    hazard = _flat(
        phase25.THE_ARM_REGISTERED["THE_224_SHORTCUT_WOULD_KILL_THE_RUN_AT_LAUNCH"]
    )
    assert "518-native" in hazard
    assert "die at launch, not at config time" in hazard
    caveat = _flat(phase25.THE_ARM_REGISTERED["the_fix_and_its_caveat"])
    assert "37x37 to 16x16" in caveat
    assert "first DOWNWARD interpolation" in caveat


def test_the_resolution_ladder_is_unreachable_for_patch_14():
    from cleft import roadb
    from cleft.models.factory import FactoryError, timm_kwargs_for

    assert roadb.RESOLUTIONS == (224, 512, 768)
    for size in (512, 768):
        assert size % 14, size                 # divides by neither
        with pytest.raises(FactoryError, match="divisible by 14"):
            timm_kwargs_for("vit_base_patch14_dinov2", (size, size))
    # 224 and the other multiples of 112 are fine.
    for size in (224, 336, 448):
        assert size % 14 == 0 and size % 16 == 0

    recorded = _flat(phase25.THE_ARM_REGISTERED[
        "the_resolution_ladder_is_UNREACHABLE_for_this_backbone"
    ])
    assert "multiples of 112" in recorded


def test_the_head_does_not_change_because_the_width_is_768():
    pytest.importorskip("torch")
    timm = pytest.importorskip("timm")

    model = timm.create_model(
        "vit_base_patch14_dinov2", pretrained=False, num_classes=0
    )
    assert model.num_features == 768

    from cleft.models.factory import BACKBONES

    assert BACKBONES["vit_b16"]["embedding_dim"] == 768
    recorded = _flat(phase25.THE_ARM_REGISTERED["the_head_does_NOT_change"])
    assert "768-dim" in recorded and "769 fitted parameters" in recorded
    assert "not by assuming ViT-B means 768" in recorded


# --------------------------------------------------------------------------
# readings, bounds, criteria, negative space
# --------------------------------------------------------------------------


def test_the_bounds_include_the_interpolation():
    record = phase25.WHAT_THIS_DOES_NOT_TEST
    assert "ONE self-supervised checkpoint, not self-supervision" in _flat(
        record["one_checkpoint_not_self_supervision"]
    )
    assert "HELD at the probe's values" in _flat(
        record["not_resolution_geometry_or_head"]
    )
    rides = _flat(record["the_downward_interpolation_rides_along"])
    assert "could be the checkpoint or could be that interpolation" in rides


def test_the_criteria_are_a_draft_and_the_settings_are_open():
    draft = phase25.EXIT_CRITERIA_DRAFT
    assert draft["status"] == "DRAFT, NOT LOCKED"
    numbered = [k for k in draft if k[0].isdigit()]
    assert len(numbered) == 9
    assert sorted(int(k.split("_")[0]) for k in numbered) == list(range(1, 10))
    assert "not an addition to this phase once a result is visible" in _flat(
        draft["8_no_second_arm_is_added_after_numbers_exist"]
    )

    settings = phase25.SETTINGS_TO_DECLARE
    assert len(settings) == 6
    # Two are already MEASURED rather than open, and say so.
    assert "MEASURED: 768" in _flat(settings["3_output_dimensionality"])
    norm = _flat(settings["5_normalization"])
    assert "MEASURED: ImageNet statistics" in norm
    assert "Nothing to rule" in norm
    # The guard fired on the first draft and the record says so.
    assert "The guard was right" in norm
    assert "0.485" not in norm and "0.229" not in norm


def test_the_summary_carries_every_record():
    assert sorted(phase25.summary()) == [
        "arm",
        "arm_dict_key_defect",
        "attribution_problem",
        "axis_closes",
        "checkpoint_declaration",
        "checkpoints_resolved",
        "closing",
        "condition_split_recounted",
        "contrast_launch_void",
        "contrasts",
        "contrasts_config_emitted",
        "counter_evidence",
        "delta_inside_the_band",
        "does_not_test",
        "exemption_is_noisy",
        "exit_criteria",
        "exit_criteria_draft",
        "fabricated_figures_withdrawn",
        "factory_pretrained",
        "family",
        "init_vocabulary_defect",
        "limits_as_literals",
        "neither_cell_fired",
        "readings",
        "reckoning",
        "seed_instability",
        "settings_ruled",
        "settings_to_declare",
        "the_unexpected_cell",
        "three_verdicts",
        "transcription_slip",
        "two_arms_ruled",
        "what_the_arms_actually_show",
        "withdrawal_swept",
    ]


# --------------------------------------------------------------------------
# [2026-09-02] The ruling, the lock and the four configs
# --------------------------------------------------------------------------

import yaml                                                    # noqa: E402

ARMS = ("p25_arm_d2", "p25_arm_d1")
EXTRACTS = ("p25_extract_d2_dinov2", "p25_extract_d1_dino")


def _config(stem):
    return yaml.safe_load(
        (REPO / "configs" / f"{stem}.yaml").read_text(encoding="utf-8")
    )


def test_the_ruling_records_the_pairs_logic_before_any_number():
    record = phase25.THE_TWO_ARMS_RULED
    ground = _flat(record["the_ground"])
    assert "attributable to neither" in ground
    assert "0.0017" in ground

    logic = _flat(record["the_pairs_logic_stated_before_the_numbers"])
    assert "if BOTH move together, the OBJECTIVE is doing the work" in logic
    assert "If only D2 moves, it is the encoder, the patch size or the" in logic
    assert "cannot separate those three" in logic

    assert "not adding a chance to win" in _flat(record["why_this_is_not_two_bites"])


def test_d2_is_named_as_not_a_control_and_d1_as_dino_not_dinov2():
    record = phase25.THE_TWO_ARMS_RULED
    d2, d1 = record["arm_D2"], record["arm_D1"]

    assert d2["checkpoint"] == "vit_base_patch14_dinov2.lvd142m"
    assert "objective, patch size (14 vs 16) and native resolution" in _flat(
        d2["what_it_varies"]
    )
    assert "must never be described as one" in _flat(
        d2["it_is_NOT_a_clean_control"]
    )

    assert d1["checkpoint"] == "vit_base_patch16_224.dino"
    assert "SAME-ENCODER control" in _flat(d1["what_it_is"])
    limit = _flat(d1["it_is_DINO_not_DINOv2_and_that_limit_travels"])
    assert "a DIFFERENT self-supervised method" in limit
    assert "isolates THE OBJECTIVE, not DINOv2 specifically" in limit
    assert "the natural misreading is that D1 is 'DINOv2 at patch 16'" in limit


def test_the_family_is_three_and_fixed_before_any_number():
    family = phase25.THE_FAMILY_OF_THREE
    assert family["count"] == 3
    assert "D2 vs the 0.2520 probe" in _flat(family["primaries"])
    assert "D2 vs D1" in _flat(family["secondary"])
    correction = _flat(family["no_alpha_correction"])
    assert "FIXED BEFORE ANY NUMBER" in correction
    assert "No contrast is added, dropped or selected" in correction
    assert "the three results are correlated" in _flat(
        family["the_arms_are_NOT_independent"]
    )


def test_the_four_combination_cells_are_committed():
    readings = phase25.READINGS_COMMITTED
    for key in ("cell_BOTH_above", "cell_D2_above_D1_not",
                "cell_D1_above_D2_not", "cell_NEITHER"):
        assert key in readings, key

    assert "the OBJECTIVE is doing the work" in _flat(readings["cell_BOTH_above"])
    assert "WHICH IT CANNOT SEPARATE" in _flat(readings["cell_D2_above_D1_not"])
    odd = _flat(readings["cell_D1_above_D2_not"])
    assert "recorded as an OBSERVATION" in odd
    assert "No mechanism is to be invented for this cell" in odd
    neither = _flat(readings["cell_NEITHER"])
    assert "EXPECTED outcome" in neither
    assert "twelfth and thirteenth" in neither
    assert "CLOSES BY MEASUREMENT" in neither

    assert "the record predicts UNRESOLVED, on all three" in _flat(
        readings["the_expectation_registered_explicitly"]
    )
    assert "not hoping for any of them" in _flat(
        readings["no_cell_is_ranked_as_the_good_one"]
    )


def test_the_limits_are_tested_literals():
    literals = phase25.LIMITS_AS_LITERALS
    assert len(literals) == 3
    assert literals[0].startswith(
        "THIS PHASE TESTS TWO CHECKPOINTS, NOT SELF-SUPERVISION."
    )
    assert literals[1].startswith(
        "ARM D2 IS NOT A CLEAN CONTROL AND MUST NEVER BE DESCRIBED AS ONE."
    )
    assert literals[2].startswith(
        "RESOLUTION, GEOMETRY AND THE HEAD ARE HELD AT THE PROBE'S VALUES"
    )
    assert "multiples of 112" in literals[2]
    assert "D1 is the control; D2 is the arm." in literals[1]


def test_the_criteria_are_locked_with_the_nothing_added_clause():
    criteria = phase25.EXIT_CRITERIA
    assert criteria["status"] == "LOCKED"
    numbered = [k for k in criteria if k[0].isdigit()]
    assert len(numbered) == 9
    assert sorted(int(k.split("_")[0]) for k in numbered) == list(range(1, 10))
    added = _flat(criteria["nothing_added_after"])
    assert "NOTHING IS ADDED ONCE NUMBERS EXIST" in added
    assert "separating them later is visibly a new phase" in added
    assert "A third arm to separate D2's three candidate causes is a NEW" in _flat(
        criteria["8_the_family_is_three_and_no_arm_is_added_after_numbers_exist"]
    )


# ---- the code the arms need -------------------------------------------


def test_the_two_backbones_are_registered_with_the_measured_width():
    from cleft.models.factory import BACKBONES, NATIVE_INPUT_SIZE

    assert len(BACKBONES) == 9
    for name, timm_name in (("vit_b14_dinov2", "vit_base_patch14_dinov2.lvd142m"),
                            ("vit_b16_dino", "vit_base_patch16_224.dino")):
        spec = BACKBONES[name]
        assert spec["timm_name"] == timm_name
        assert spec["kind"] == "transformer"
        assert spec["embedding_dim"] == 768        # the head stays 769 params
    # Only DINOv2 is non-224-native, and only it is listed as such.
    assert NATIVE_INPUT_SIZE == {"vit_base_patch14_dinov2.lvd142m": 518}


def test_the_518_native_case_routes_before_the_224_shortcut():
    from cleft.models.factory import FactoryError, timm_kwargs_for

    dinov2 = "vit_base_patch14_dinov2.lvd142m"
    # At 224 it needs dynamic sizing; the shortcut would have returned {}.
    assert timm_kwargs_for(dinov2, (224, 224)) == {"dynamic_img_size": True}
    # At its OWN native size it needs nothing, exactly as 224 is for the rest.
    assert timm_kwargs_for(dinov2, (518, 518)) == {}
    # D1 is 224-native and takes the historical path untouched.
    assert timm_kwargs_for("vit_base_patch16_224.dino", (224, 224)) == {}
    # Road B's sizes are unreachable for patch 14, and it says why.
    for size in (512, 768):
        with pytest.raises(FactoryError, match="divisible by 14"):
            timm_kwargs_for(dinov2, (size, size))
    # And the probe's own routing is unchanged.
    assert timm_kwargs_for("vit_base_patch16_224", (224, 224)) == {}
    assert timm_kwargs_for("vit_base_patch16_224", (768, 768)) == {
        "dynamic_img_size": True
    }


def test_the_new_inits_are_factory_pretrained_from_one_definition():
    from cleft.train.extract import FACTORY_PRETRAINED_INITS

    assert FACTORY_PRETRAINED_INITS == (
        "imagenet", "dinov2_lvd142m", "dino_in1k"
    )
    # The schema IMPORTS the list rather than repeating it.
    body = (REPO / "src" / "cleft" / "config" / "schema.py").read_text(
        encoding="utf-8"
    )
    assert "from ..train.extract import FACTORY_PRETRAINED_INITS" in body
    assert "init not in FACTORY_PRETRAINED_INITS and not checkpoint" in body
    assert 'if init != "imagenet" and not checkpoint' not in body

    record = _flat(
        phase25.THE_WEIGHTS_ARE_FACTORY_PRETRAINED_LIKE_THE_PROBES[
            "why_they_are_NOT_called_imagenet"
        ]
    )
    assert "DINO's is ImageNet-1k **without labels**" in record
    assert "the R2 shape" in record


# ---- the four configs -------------------------------------------------


def test_all_four_configs_validate_and_name_the_ruled_checkpoints():
    from cleft.config import schema

    for stem in EXTRACTS + ARMS:
        schema.validate(_config(stem))

    d2 = _config("p25_extract_d2_dinov2")["task"]["sets"][0]
    assert d2 == {"backbone": "vit_b14_dinov2", "init": "dinov2_lvd142m",
                  "geometry": "g1"}
    d1 = _config("p25_extract_d1_dino")["task"]["sets"][0]
    assert d1 == {"backbone": "vit_b16_dino", "init": "dino_in1k",
                  "geometry": "g1"}


def test_the_arms_hold_every_probe_field_and_vary_only_the_backbone():
    """One factor, checked against the probe's OWN config."""
    probe = _config("p7_d1_vit_b16_imagenet_g1")["task"]
    for stem in ARMS:
        task = _config(stem)["task"]
        for field in ("kind", "geometry", "label", "max_epochs", "patience",
                      "inner_val_frac", "monitor", "seeds", "trainable",
                      "learning_rate"):
            assert task[field] == probe[field], (stem, field)
        # The two that differ are the factor and the artifact it needs.
        assert task["backbone"] != probe["backbone"]
        assert task["init"] != probe["init"]
        assert task["seeds"] == [1337, 2024, 7, 99, 12345]


ARM_ROLLUPS = {
    "p25_arm_d2": (
        "dinov2_g1_v1/vit_b14_dinov2__dinov2_lvd142m__g1",
        "b354326677cbc78f39286e43389be1a6eebc0aef300373cb32bff294c20b0edd",
    ),
    "p25_arm_d1": (
        "dino_g1_v1/vit_b16_dino__dino_in1k__g1",
        "5290095906192da10f7e2313a9f47e8f51a83a264d86634306e7fff599c1d23d",
    ),
}


def test_each_arm_carries_ITS_OWN_artifact_and_no_placeholder():
    """[UPDATED 2026-09-02, THIS PIN FIRED AS DESIGNED] It held each arm at
    one placeholder, true until the extractions ran. **The crossed-fill
    hazard is what the pin holds now**: the two sets are 731,250 and
    731,243 bytes and the same shape, so giving each arm the other's hash
    would run each on the other's features and **nothing downstream would
    notice** -- check_pairing compares the config against the artifact's
    own metadata, and both would look entirely ordinary."""
    for stem in ARMS:
        config = _config(stem)
        placeholders = [
            e["name"] for e in config["inputs"]
            if set(e["rollup_sha256"]) == {"0"}
        ]
        assert placeholders == [], stem

        directory, rollup = ARM_ROLLUPS[stem]
        embeddings = next(
            e for e in config["inputs"] if e["name"] == "embeddings"
        )
        assert embeddings["path"].endswith(directory), stem
        assert embeddings["rollup_sha256"] == rollup, stem
        assert len(rollup) == 64
        assert set(rollup) <= set("0123456789abcdef")

        text = (REPO / "configs" / f"{stem}.yaml").read_text(encoding="utf-8")
        assert "NO PLACEHOLDER" in text
        assert "crossed fill" in text

    # **Not crossed**, asserted as its own fact rather than implied.
    hashes = {stem: _config(stem)["inputs"][-1]["rollup_sha256"] for stem in ARMS}
    assert len(set(hashes.values())) == 2, hashes
    assert hashes["p25_arm_d2"] == ARM_ROLLUPS["p25_arm_d2"][1]
    assert hashes["p25_arm_d1"] == ARM_ROLLUPS["p25_arm_d1"][1]


def test_the_generator_refuses_a_crossed_fill():
    """The guard is in the generator, not only in this test."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "gen_p25", REPO / "scripts" / "generate_phase25_configs.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # The rollups are keyed by SET DIRECTORY, which is what they are of.
    assert set(module.ROLLUPS) == {directory for directory, _ in ARM_ROLLUPS.values()}
    for stem, (directory, rollup) in ARM_ROLLUPS.items():
        assert module.ROLLUPS[directory] == rollup, stem

    # Two arms sharing a hash is refused rather than emitted.
    original = dict(module.ROLLUPS)
    try:
        for key in module.ROLLUPS:
            module.ROLLUPS[key] = original[
                "dinov2_g1_v1/vit_b14_dinov2__dinov2_lvd142m__g1"
            ]
        with pytest.raises(module.DriftError, match="reading the other's"):
            module.build()
    finally:
        module.ROLLUPS.clear()
        module.ROLLUPS.update(original)
    texts, placeholders = module.build()
    assert len(texts) == 5
    assert placeholders == []          # both arm rollups are filled


def test_the_arms_copy_the_probes_recipe_through_the_generator():
    """One factor by CONSTRUCTION: the generator reads the probe's config
    rather than repeating its fields, so a moved recipe is drift, not a
    silent second axis."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "gen_p25b", REPO / "scripts" / "generate_phase25_configs.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    probe = _config(module.PROBE)["task"]
    assert "backbone" not in module.COPIED and "init" not in module.COPIED
    for stem in ARMS:
        task = _config(stem)["task"]
        for field in module.COPIED:
            assert task[field] == probe[field], (stem, field)


def test_every_config_header_carries_the_unclosed_pin_read():
    for stem in EXTRACTS + ARMS:
        text = (REPO / "configs" / f"{stem}.yaml").read_text(encoding="utf-8")
        assert "pinned at timm 1.0.7" in text
        assert "CANNOT run" in text
        # The sentence wraps across two comment lines.
        assert "the fix is a" in text and "decision about the pin" in text


def test_the_checkpoints_resolution_names_where_it_was_measured():
    record = phase25.THE_CHECKPOINTS_RESOLVED
    where = _flat(record["WHERE_that_was_measured"])
    assert "timm 1.0.27 / torch 2.13" in where
    assert "NOT the pinned image" in where
    assert "SKIP there" in where
    blocking = _flat(record["THE_BLOCKING_READ_IS_NOT_CLOSED"])
    assert "Resolving a tag in 1.0.27 is not resolving it in 1.0.7" in blocking
    assert "neither config can run" in blocking
    # The DINOv3 observation is recorded and explicitly not proposed.
    assert "Recorded as an observation, not proposed" in _flat(
        record["an_observation_that_changes_nothing_here"]
    )


def test_the_settings_table_carries_per_value_provenance():
    settings = phase25.THE_SETTINGS_RULED
    for key, entry in settings.items():
        if key == "ruled":
            continue
        assert set(entry) == {"value", "provenance"}, key
        assert entry["provenance"], key
    assert settings["embedding_width"]["value"] == 768
    assert "MEASURED" in settings["embedding_width"]["provenance"]
    assert "CHOSEN HERE" in _flat(settings["init_names"]["provenance"])
    assert "BLOCKING" in _flat(settings["the_pin"]["provenance"])
    # The normalization row names the guard rather than the numbers.
    norm = _flat(settings["normalization"]["provenance"])
    assert "The guard was right" in norm
    assert "0.485" not in norm


def test_the_ARMS_still_need_no_new_task():
    """[UPDATED 2026-09-02, THIS PIN FIRED AS DESIGNED] It held that Phase
    25 added no task at all -- true of the arms, which reuse
    ``extract_embeddings`` and ``train_cv``, and that is the whole claim
    that they are one-factor contrasts against the probe rather than a new
    machine. **The CONTRASTS legitimately need one**, as Phase 22's did.
    The pin now holds the part that was the claim."""
    import yaml as _yaml

    from cleft.config import schema
    from cleft import results_ledger

    for stem in EXTRACTS + ARMS:
        kind = _yaml.safe_load(
            (REPO / "configs" / f"{stem}.yaml").read_text(encoding="utf-8")
        )["task"]["kind"]
        assert kind in ("extract_embeddings", "train_cv"), (stem, kind)

    # Exactly one p25 task, and it is the contrast reader.
    assert [k for k in schema.TASK_SPECS if k.startswith("p25")] == [
        "p25_contrasts"
    ]
    assert "pooled_features" not in schema.TASK_SPECS
    # [2026-09-06] Was `assert len(results_ledger.ENTRIES) == 38`, a
    # global count standing in for Phase 25 banked no ledger row. It went stale when Phase 27
    # appended entry 39. Rewritten as the relation, which does not.
    assert not [
        e for e in results_ledger.ENTRIES
        if e["phase"] == "p25" or "p25-" in e["id"]
    ]
    results_ledger.validate()


def test_the_init_vocabulary_defect_is_recorded_with_its_class():
    record = phase25.THE_INIT_WAS_OUTSIDE_THE_VOCABULARY
    assert record["tag"].startswith("[FIXED]")

    what_for = _flat(record["what_INITS_is_actually_for"])
    assert "the Phase 7 LADDER'S LATTICE" in what_for
    assert "not a variant of this project's own pretraining" in what_for

    why = _flat(record["why_the_fix_is_NOT_a_fourth_member_of_INITS"])
    assert "24 embedding sets into 40" in why
    assert "LADDER_BACKBONES" in why

    caught = _flat(record["the_guard_existed_for_ONE_FIELD_and_not_this_one"])
    assert "one field over" in caught
    assert "FOUND 2026-08-14" in caught

    sweep = _flat(record["the_settings_sweep_would_NOT_have_caught_it"])
    assert "Consumed and refused is a different failure" in sweep

    arms = _flat(record["the_arm_configs_were_checked_and_are_CLEAN"])
    assert "vit_b14_dinov2__dinov2_lvd142m__g1" in arms
    assert "No second wall on the arm side" in arms

    # The path the record says the arm configs declare is the path they do.
    import yaml

    from cleft import embedding_plan

    for stem, backbone, init in (("p25_arm_d2", "vit_b14_dinov2", "dinov2_lvd142m"),
                                 ("p25_arm_d1", "vit_b16_dino", "dino_in1k")):
        config = yaml.safe_load(
            (REPO / "configs" / f"{stem}.yaml").read_text(encoding="utf-8")
        )
        entry = next(e for e in config["inputs"] if e["name"] == "embeddings")
        expected = embedding_plan.set_name(
            {"backbone": backbone, "init": init, "geometry": "g1",
             "pretrain_scheme": None}
        )
        assert entry["path"].endswith(expected), (stem, entry["path"])


# --------------------------------------------------------------------------
# [2026-09-02] The three contrasts
# --------------------------------------------------------------------------


def test_the_family_is_three_and_cannot_grow_through_a_config():
    family = phase25.contrast_family()
    assert [c["key"] for c in family] == [
        "d2-vs-probe", "d1-vs-probe", "d2-vs-d1"
    ]
    assert [c["kind"] for c in family] == ["primary", "primary", "secondary"]
    # Exactly the ruled shape: each arm against the probe, plus D2 vs D1.
    assert {c["a"] for c in family} == {"p25_arm_d2", "p25_arm_d1"}
    assert [c for c in family if c["b"] == "probe"] == family[:2]

    # The pair list is NOT a config field: the SPEC has no such field.
    from cleft.config.schema import TASK_SPECS

    assert "contrasts" not in TASK_SPECS["p25_contrasts"]
    assert "expect_contrasts" in TASK_SPECS["p25_contrasts"]
    body = (REPO / "src" / "cleft" / "run.py").read_text(encoding="utf-8")
    task = body.split("def task_p25_contrasts(")[1].split("\ndef ")[0]
    assert "phase25.contrast_family()" in task


def test_the_contrast_task_uses_paired_comparison_unchanged():
    import inspect

    from cleft import phase7b
    from cleft import run as run_module

    task = inspect.getsource(run_module.task_p25_contrasts)
    assert "from .phase7b import paired_comparison" in task
    assert "paired_comparison(" in task
    # No second implementation of the criterion anywhere in the task.
    for forbidden in ("paired_delta_bca", "combined_claimable_delta",
                      "def _claimable"):
        assert forbidden not in task, forbidden
    assert callable(phase7b.paired_comparison)


def test_both_conditions_are_recorded_and_a_null_refuses():
    """The Phase 22 null-condition defect must not recur."""
    import inspect

    from cleft import run as run_module

    task = inspect.getsource(run_module.task_p25_contrasts)
    assert 'condition_1 = result["all_seeds_exclude_zero_one_direction"]' in task
    assert 'condition_2 = result["exceeds_threshold"]' in task
    assert "if condition_1 is None or condition_2 is None:" in task
    assert "may not be recorded without both" in task
    # Both reach the written record, with n and the threshold beside them.
    row = task.split("verdicts[contrast[\"key\"]] = {")[1].split("}")[0]
    for field in ('"condition_1"', '"condition_2"', '"n_seeds"',
                  '"n_patients"', '"threshold"', '"verdict"'):
        assert field in row, field


def test_the_winner_sd_is_computed_from_the_artifacts_not_declared():
    """A declared sd is a number that can disagree with its artifact."""
    import inspect

    from cleft.config.schema import TASK_SPECS
    from cleft import run as run_module

    assert "winner_sd" not in TASK_SPECS["p25_contrasts"]
    task = inspect.getsource(run_module.task_p25_contrasts)
    assert 'winner_sd=arm_stats[contrast["a"]]["sd"]' in task
    assert 'float(np.std(scores, ddof=1))' in task
    # And the arm means are measured, not read from a config field.
    assert 'pcc(truth, by_seed[seed][arm])' in task


def test_the_owed_state_ENDED_when_the_names_arrived():
    """**The guard was right.** A first attempt emitted PENDING_SHA8
    markers for the two run directories;
    ``test_no_declared_run_directory_contradicts_itself`` refused them,
    because a declared run directory must LOOK like one. The config is
    not shipped until the names exist, and the generator reports what it
    waits on rather than guessing -- Phase 22 invented four run-directory
    names and all four were wrong."""
    import importlib.util

    # [UPDATED 2026-09-02, THIS PIN FIRED AS DESIGNED] It held the config
    # OWED while the two run-directory names were unknown, and refused to
    # let them be guessed. **The names arrived and the config shipped**,
    # so the pin now holds the end of that state -- and that the guard
    # which forbade a fake path still stands.
    assert (REPO / "configs" / "p25_contrasts.yaml").exists()

    spec = importlib.util.spec_from_file_location(
        "gen_p25c", REPO / "scripts" / "generate_phase25_configs.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert set(module.RUN_DIRECTORIES) == {"p25_arm_d2", "p25_arm_d1"}
    assert module.owed() == []          # nothing is owed any more
    texts, _ = module.build()
    assert "p25_contrasts" in texts
    # The refusal that made the wait necessary is unchanged.
    source = (
        REPO / "scripts" / "generate_phase25_configs.py"
    ).read_text(encoding="utf-8")
    assert "Phase 22 invented four run" in source
    assert "directory names and all four were wrong" in source


def _unused_contrasts_config_shape():
    from cleft.config import schema

    config = _config("p25_contrasts")
    schema.validate(config)
    assert [a["name"] for a in config["task"]["arms"]] == [
        "probe", "p25_arm_d2", "p25_arm_d1"
    ]
    assert config["task"]["expect_patients"] == 237

    pending = [
        e["name"] for e in config["inputs"]
        if set(e["rollup_sha256"]) == {"0"}
    ]
    assert pending == ["p25_arm_d2", "p25_arm_d1"]
    # **The PATHS are pending too**, and visibly so: Phase 22 invented four
    # run-directory names and all four were wrong.
    for entry in config["inputs"]:
        if entry["name"] in pending:
            assert "PENDING_SHA8__PENDING_JOB_ID" in entry["path"], entry

    # The probe's run directory is the one Phase 22 already declared.
    donor = yaml.safe_load(
        (REPO / "configs" / "p22_contrasts.yaml").read_text(encoding="utf-8")
    )
    probe = next(
        e for e in donor["inputs"] if e["name"] == "p7_d1_vit_b16_imagenet_g1"
    )
    ours = next(
        e for e in config["inputs"]
        if e["name"] == "p7_d1_vit_b16_imagenet_g1"
    )
    assert ours == probe


# ---- the two registrations, before the numbers ---------------------------


def test_the_delta_record_is_withdrawn_with_its_original_preserved():
    """[UPDATED 2026-09-02, THIS PIN FIRED AS DESIGNED] It held a record
    that reasons about a +0.0656 delta. **The figures it rests on are not
    in the runs** -- D1 is 0.2393 against the probe's 0.2520, so the delta
    is negative. The record is WITHDRAWN and preserved, and the pin now
    holds both facts."""
    record = phase25.THE_DELTA_IS_INSIDE_THE_UNRESOLVED_BAND
    withdrawn = _flat(record["WITHDRAWN_2026_09_02"])
    assert "the +0.0656 does not exist" in withdrawn
    assert "0.2393" in withdrawn and "0.2520" in withdrawn
    assert "Nothing below is edited" in withdrawn

    # The original body is preserved, unedited.
    delta = _flat(record["the_delta"])
    assert "+0.0656" in delta
    assert "largest positive delta this project has produced" in delta

    band = _flat(record["the_band_it_sits_in"])
    assert "0.04-0.10" in band
    assert "0.1386" in band
    assert "may therefore still return UNRESOLVED" in band

    why = _flat(record["why_that_is_not_a_contradiction"])
    assert "Five arm means above five arm means is neither of those" in why

    finding = _flat(record["the_combination_is_ITSELF_the_finding_if_it_happens"])
    assert "still does not pass" in finding
    assert "a statement about the COHORT'S RESOLUTION" in finding
    assert "not argued around" in finding
    # Symmetric: the other outcome is registered too.
    assert "first feature source in twelve measurements" in _flat(
        record["and_if_it_DOES_pass"]
    )


def test_the_unexpected_cell_and_its_mechanism_are_withdrawn_entirely():
    """[UPDATED 2026-09-02, THIS PIN FIRED AS DESIGNED] It held an
    observation of D1-above-D2-not and a [REASONED] token-density
    hypothesis for it. **The pattern did not occur**: D1 0.2393 and D2
    0.1326 both sit at or below the probe's 0.2520, so the NEITHER cell
    fired -- the registered expectation. The hypothesis explained a
    difference that does not exist and is withdrawn ENTIRELY, not
    qualified."""
    record = phase25.THE_UNEXPECTED_CELL
    withdrawn = _flat(record["WITHDRAWN_2026_09_02"])
    assert "the pattern did not occur" in withdrawn
    assert "explains nothing" in withdrawn
    assert "the NEITHER cell fired" in withdrawn

    # Withdrawn ENTIRELY, and the module says why a [REASONED] tag did
    # not make it safe.
    module = (REPO / "src" / "cleft" / "phase25.py").read_text(encoding="utf-8")
    assert "WITHDRAWN ENTIRELY 2026-09-02" in module
    assert "It explains nothing." in module
    assert "a claim about nothing" in module
    # Phase 7D's own figures are NOT withdrawn -- only their use here.
    from cleft import ladder

    assert "196 tokens 0.2520, 256 tokens 0.2280" in (
        ladder.PHASE_7D_OBSERVED["grid_density"]["ordering"]
    )

    # And the original body is preserved, unedited.
    which = _flat(record["which_cell"])
    assert "D1 above, D2 not" in which
    assert "the registered expectation was cell neither" in which.lower()

    beside = _flat(record["it_is_an_OBSERVATION_beside_the_unfired_cells"])
    assert "Phase 17 style" in beside
    assert "PRESERVED AND UNFIRED" in beside
    assert "no reading is amended after the fact" in beside.lower()

    mechanism = _flat(record["the_candidate_mechanism_REASONED_not_concluded"])
    assert "[REASONED]" in mechanism
    assert "196 tokens" in mechanism and "256 tokens" in mechanism
    assert "may be giving back what its objective gains" in mechanism

    # The 7D figures are quoted, and they are the ladder's own.
    from cleft import ladder

    ordering = ladder.PHASE_7D_OBSERVED["grid_density"]["ordering"]
    assert "196 tokens 0.2520, 256 tokens 0.2280" in ordering
    assert "196 tokens 0.2520" in mechanism.replace("**", "")

    # It is suggestive, not a decomposition, and the phase cannot resolve it.
    assert "is not 0.0679 and this record does not pretend" in _flat(
        record["the_arithmetic_is_SUGGESTIVE_and_NOT_a_decomposition"]
    )
    cannot = _flat(record["the_phase_did_NOT_test_it_and_CANNOT_resolve_it"])
    assert "two arms cannot separate three factors" in cannot
    assert "unfalsifiable within Phase 25" in cannot
    assert "NOT REGISTERED" in _flat(record["what_would_test_it"])


def test_the_four_registered_cells_are_unamended():
    """The observation sits beside them; it does not edit them."""
    readings = phase25.READINGS_COMMITTED
    for key in ("cell_BOTH_above", "cell_D2_above_D1_not",
                "cell_D1_above_D2_not", "cell_NEITHER"):
        assert key in readings, key
    # The cell that fired still says what it said before it fired.
    fired = _flat(readings["cell_D1_above_D2_not"])
    assert "recorded as an OBSERVATION" in fired
    assert "No mechanism is to be invented for this cell" in fired
    assert "EXPECTED outcome" in _flat(readings["cell_NEITHER"])


def test_no_ledger_row_is_added_by_building_the_contrasts():
    from cleft import results_ledger

    # [2026-09-06] Was `assert len(results_ledger.ENTRIES) == 38`, a
    # global count standing in for Phase 25 banked no ledger row. It went stale when Phase 27
    # appended entry 39. Rewritten as the relation, which does not.
    assert not [
        e for e in results_ledger.ENTRIES
        if e["phase"] == "p25" or "p25-" in e["id"]
    ]
    results_ledger.validate()


def test_run_directories_are_matched_by_stem_not_by_line_order():
    """The names arrive as a file; a mis-ordered file must not swap the
    two arms, and a half-named one must not half-build the family."""
    import importlib.util
    import tempfile

    spec = importlib.util.spec_from_file_location(
        "gen_p25d", REPO / "scripts" / "generate_phase25_configs.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def _write(text):
        handle = tempfile.NamedTemporaryFile(
            "w", suffix=".txt", delete=False, encoding="utf-8"
        )
        handle.write(text)
        handle.close()
        return handle.name

    # D1 first, D2 second -- the OPPOSITE of the family's order.
    path = _write(
        "runs/keeper/p25/p25_arm_d1__aaaaaaaa__p25-arm-d1-3/\n"
        "runs/keeper/p25/p25_arm_d2__bbbbbbbb__p25-arm-d2-3/\n"
    )
    resolved = module.load_run_directories(path)
    assert resolved["p25_arm_d1"].endswith("p25_arm_d1__aaaaaaaa__p25-arm-d1-3")
    assert resolved["p25_arm_d2"].endswith("p25_arm_d2__bbbbbbbb__p25-arm-d2-3")
    assert all(v.startswith("/home/user/codex/") for v in resolved.values())

    # A missing arm is refused rather than half-read.
    path = _write("runs/keeper/p25/p25_arm_d1__aaaaaaaa__p25-arm-d1-3/\n")
    with pytest.raises(module.DriftError, match="All three contrasts run"):
        module.load_run_directories(path)

    # A duplicated arm is refused.
    path = _write(
        "runs/keeper/p25/p25_arm_d1__aaaaaaaa__p25-arm-d1-3/\n"
        "runs/keeper/p25/p25_arm_d1__cccccccc__p25-arm-d1-4/\n"
    )
    with pytest.raises(module.DriftError, match="appears twice"):
        module.load_run_directories(path)

    # A stranger is refused.
    path = _write("runs/keeper/p7/p7_d1_vit_b16_imagenet_g1__3f71a6a9__x/\n")
    with pytest.raises(module.DriftError, match="not a Phase 25 arm"):
        module.load_run_directories(path)


def test_an_unknown_run_directory_gets_no_hash():
    """[UPDATED 2026-09-02, THIS PIN FIRED AS DESIGNED] It held that
    supplying names emits the config with both hashes pending -- true
    while none was filled. **The rollups are keyed BY RUN DIRECTORY**, so
    what the pin holds now is the consequence: a directory the generator
    has no hash for gets a placeholder, and cannot inherit another
    directory's."""
    import importlib.util
    import tempfile

    import yaml as _yaml

    spec = importlib.util.spec_from_file_location(
        "gen_p25e", REPO / "scripts" / "generate_phase25_configs.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    handle = tempfile.NamedTemporaryFile(
        "w", suffix=".txt", delete=False, encoding="utf-8"
    )
    handle.write(
        "runs/keeper/p25/p25_arm_d2__aaaaaaaa__p25-arm-d2-3/\n"
        "runs/keeper/p25/p25_arm_d1__bbbbbbbb__p25-arm-d1-3/\n"
    )
    handle.close()
    for arm, path in module.load_run_directories(handle.name).items():
        module.RUN_DIRECTORIES[arm] = path

    texts, placeholders = module.build()
    assert "p25_contrasts" in texts
    # These are NOT the baked directories, so neither has a rollup.
    assert sorted(placeholders) == [
        "p25_contrasts:p25_arm_d1", "p25_contrasts:p25_arm_d2"
    ]

    config = _yaml.safe_load(texts["p25_contrasts"])
    assert config["task"]["kind"] == "p25_contrasts"
    assert config["task"]["expect_contrasts"] == 3
    assert [a["name"] for a in config["task"]["arms"]] == [
        "probe", "p25_arm_d2", "p25_arm_d1"
    ]
    pending = [
        e["name"] for e in config["inputs"]
        if set(e["rollup_sha256"]) == {"0"}
    ]
    assert sorted(pending) == ["p25_arm_d1", "p25_arm_d2"]
    # The probe's entry is the one Phase 22 already declared, unchanged.
    donor = _yaml.safe_load(
        (REPO / "configs" / "p22_contrasts.yaml").read_text(encoding="utf-8")
    )
    probe = next(
        e for e in donor["inputs"] if e["name"] == "p7_d1_vit_b16_imagenet_g1"
    )
    assert probe in config["inputs"]

    from cleft.config import schema

    schema.validate(config)


def test_what_the_arms_actually_show_is_measured_from_the_artifacts():
    record = phase25.WHAT_THE_ARMS_ACTUALLY_SHOW
    numbers = _flat(record["the_three_numbers"])
    assert "D1 (DINO-B/16) 0.2393" in numbers
    assert "D2 (DINOv2-B/14) 0.1326" in numbers
    assert "Probe 0.2520" in numbers
    assert "at or below" in numbers

    fired = _flat(record["the_cell_that_fired_is_NEITHER"])
    assert "the registered expectation" in fired
    assert "predicted its own null and got it" in fired
    # The quoted reading really is the registered one, word for word.
    registered = _flat(phase25.READINGS_COMMITTED["cell_NEITHER"])
    assert "twelfth and thirteenth measurements agree with the eleven" in registered
    assert "CLOSES BY MEASUREMENT rather than by assumption" in registered

    spread = _flat(record["the_spread_is_WIDE_not_tight"])
    assert "0.1721 to 0.2949" in spread
    assert "wider than the probe's, not narrower" in spread
    assert "0.0055" in spread            # the withdrawn sd, named as withdrawn

    assert "not the project's highest single-arm PCC" in _flat(
        record["no_claim_of_a_highest_arm"]
    )
    assert "twelfth and thirteenth" in _flat(record["the_axis_closes_across_THIRTEEN"])
    # Arm means, not verdicts.
    assert "these are ARM MEANS, not verdicts" in _flat(
        record["the_contrasts_have_NOT_run"]
    )
    assert record["tag"].startswith("[MEASURED]")


def test_the_withdrawal_names_its_provenance_and_spares_the_neighbours():
    record = phase25.THE_FABRICATED_FIGURES_WITHDRAWN
    what = _flat(record["what_was_withdrawn"])
    for token in ("0.3176", "0.3080", "0.3210", "0.3223", "0.3168",
                  "0.3200", "0.0055", "0.2497"):
        assert token in what, token

    grep = _flat(record["the_grep_that_settles_it"])
    assert "0.308062" in grep
    assert "seed_7__curves.csv" in grep
    assert "there was no source" in grep

    provenance = _flat(record["the_provenance"])
    assert "TRANSCRIPTION SLIP" in provenance
    assert "banked them into three records and one shipped log line" in provenance
    assert "is not an excuse for it" in provenance

    rule = _flat(record["the_rule_that_should_have_stopped_it"])
    assert "never quote a hash from a chat log" in rule
    assert "its principle is not about hashes" in rule

    # **The neighbours were checked before anything was edited.**
    from cleft import phase21, phase22

    assert phase21.CROSS_ARM_SHRINKAGE_MEASURED["min"] == 0.0656 if hasattr(
        phase21, "CROSS_ARM_SHRINKAGE_MEASURED"
    ) else True
    body21 = (REPO / "src" / "cleft" / "phase21.py").read_text(encoding="utf-8")
    body22 = (REPO / "src" / "cleft" / "phase22.py").read_text(encoding="utf-8")
    assert "0.0656" in body21 and "0.0656" in body22
    assert "0.2497" in body22
    assert "DIFFERENT quantity" in _flat(record["what_is_NOT_affected"])


def test_no_fabricated_figure_survives_outside_the_withdrawal():
    """The five per-seed values never reached the repo; the two derived
    from them did, and both are gone from every live claim."""
    fabricated = ("0.3176", "0.3080", "0.3210", "0.3223", "0.3168", "0.3200")
    for path in list((REPO / "src").rglob("*.py")) + \
            list((REPO / "tests").glob("*.py")) + \
            list((REPO / "configs").glob("*.yaml")):
        body = path.read_text(encoding="utf-8")
        for token in fabricated:
            if token in body:
                # Only the withdrawal record may name them.
                assert path.name in ("phase25.py", "test_phase25.py"), (
                    path, token
                )

    # The derived delta is gone from the shipped task except as a dated
    # correction naming it.
    run_body = (REPO / "src" / "cleft" / "run.py").read_text(encoding="utf-8")
    hits = [
        line for line in run_body.splitlines()
        if "0.0656" in line or "0.0679" in line
    ]
    assert len(hits) == 1, hits
    assert hits[0].lstrip().startswith("#")
    assert "does not" in hits[0]


def test_the_transcription_slip_is_filed_beside_its_predecessors():
    record = phase25.THE_TRANSCRIPTION_SLIP
    family = _flat(record["the_family"])
    # The predecessor is named by its RECORD, not by the phrase --
    # test_the_wrong_figure_appears_in_no_record forbids the words, and
    # it fired on the first draft of this very withdrawal.
    assert "ladder.DETECTION_FLOOR_PROHIBITION" in family
    assert "0.068" in family
    assert "This one had no source at all" in family
    from cleft import ladder

    assert hasattr(ladder, "DETECTION_FLOOR_PROHIBITION")

    worse = _flat(record["why_that_is_worse_not_milder"])
    assert "A figure with no source is invisible to every check" in worse
    assert "found by reading the artifacts" in worse

    shape = _flat(record["the_shape_that_made_it_dangerous"])
    assert "PLAUSIBLE and it was FLATTERING" in shape
    assert "a figure that rewards the phase gets elaborated" in shape

    correction = _flat(record["the_correction_to_practice"])
    assert "[REPORTED] until an artifact says otherwise" in correction
    assert "may not reason from it" in correction
    assert "not fixable from here" in _flat(
        record["what_would_have_caught_it_earlier"]
    )


def test_the_shipped_task_no_longer_prints_a_fabricated_delta():
    import inspect

    from cleft import run as run_module

    task = inspect.getsource(run_module.task_p25_contrasts)
    logged = task.split('ctx.log(\n        "REGISTERED BEFORE')[1]
    assert "+0.0656" not in logged
    assert "0.1386" in logged                  # the band argument stands
    assert "both PLAN 4.3 conditions decide" in logged


def test_the_emitted_config_names_the_2d54e77a_pair():
    """The live pair was identified by MEASUREMENT: the 0bf86712 pair has
    no log, the 2d54e77a pair carries five seeds each and its
    seed_variance.json gives the banked means."""
    import importlib.util

    import yaml as _yaml

    config = _yaml.safe_load(
        (REPO / "configs" / "p25_contrasts.yaml").read_text(encoding="utf-8")
    )
    by_name = {e["name"]: e for e in config["inputs"]}
    assert by_name["p25_arm_d2"]["path"].endswith(
        "p25_arm_d2__2d54e77a__p25-arm-d2-2"
    )
    assert by_name["p25_arm_d1"]["path"].endswith(
        "p25_arm_d1__2d54e77a__p25-arm-d1-2"
    )
    # The dead pair appears nowhere.
    text = (REPO / "configs" / "p25_contrasts.yaml").read_text(encoding="utf-8")
    assert "0bf86712" not in text

    # [UPDATED 2026-09-02, THIS PIN FIRED AS DESIGNED] It held the two
    # arm rollups PENDING, true until the declare. **The
    # crossed-fill hazard is what it holds now**: the two runs hold 645
    # files each and differ by 136 bytes, so a swap would load one arm's
    # predictions under the other's name and compute three entirely
    # ordinary contrasts for the wrong arms.
    ARM_RUN_ROLLUPS = {
        "p25_arm_d2": (
            "p25_arm_d2__2d54e77a__p25-arm-d2-2",
            "eed3b646040c9f14a5d056b3d34a3106fddf8f50e143b214edfe592996b7dec8",
        ),
        "p25_arm_d1": (
            "p25_arm_d1__2d54e77a__p25-arm-d1-2",
            "9b30f39203264af154f7130ff9d1f7ba589d1d2ee26f2a06c8a38478a4f4277c",
        ),
    }
    assert [
        e["name"] for e in config["inputs"]
        if set(e["rollup_sha256"]) == {"0"}
    ] == []
    for name, (directory, rollup) in ARM_RUN_ROLLUPS.items():
        entry = by_name[name]
        assert entry["path"].endswith(directory), name
        assert entry["rollup_sha256"] == rollup, name
        assert len(rollup) == 64
        assert set(rollup) <= set("0123456789abcdef")
    # **Not crossed**, asserted as its own fact.
    assert (by_name["p25_arm_d2"]["rollup_sha256"]
            != by_name["p25_arm_d1"]["rollup_sha256"])
    text = (REPO / "configs" / "p25_contrasts.yaml").read_text(encoding="utf-8")
    assert "NO PLACEHOLDERS" in text
    assert "crossed fill" in text

    # The two donors are untouched.
    import yaml as _y

    donor = {
        e["name"]: e for e in _y.safe_load(
            (REPO / "configs" / "p22_contrasts.yaml").read_text(encoding="utf-8")
        )["inputs"]
    }
    assert by_name["p7_d1_vit_b16_imagenet_g1"] == donor[
        "p7_d1_vit_b16_imagenet_g1"
    ]

    # The names are BAKED IN, so a bare --check covers all five.
    spec = importlib.util.spec_from_file_location(
        "gen_p25f", REPO / "scripts" / "generate_phase25_configs.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.LIVE_RUN_SHA8 == "2d54e77a"
    assert set(module.RUN_DIRECTORIES) == {"p25_arm_d2", "p25_arm_d1"}
    texts, placeholders = module.build()
    assert "p25_contrasts" in texts
    assert len(texts) == 5
    assert placeholders == []          # both arm rollups are filled


def test_a_supplied_file_may_confirm_the_baked_names_but_not_replace_them():
    import importlib.util
    import tempfile

    spec = importlib.util.spec_from_file_location(
        "gen_p25g", REPO / "scripts" / "generate_phase25_configs.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    handle = tempfile.NamedTemporaryFile(
        "w", suffix=".txt", delete=False, encoding="utf-8"
    )
    handle.write("runs/keeper/p25/p25_arm_d2__0bf86712__p25-arm-d2/\n"
                 "runs/keeper/p25/p25_arm_d1__0bf86712__p25-arm-d1/\n")
    handle.close()
    supplied = module.load_run_directories(handle.name)
    # The file parses; what refuses is CONTRADICTING a baked name.
    assert set(supplied) == {"p25_arm_d2", "p25_arm_d1"}
    for arm, path in supplied.items():
        assert module.RUN_DIRECTORIES[arm] != path
    source = (
        REPO / "scripts" / "generate_phase25_configs.py"
    ).read_text(encoding="utf-8")
    assert "confirm the baked name, never silently replace it" in source


def test_the_owed_registry_warns_and_the_ruling_is_recorded():
    """The mechanism is exercised with a stub, because Phase 25's own
    entry is satisfied and would warn about nothing."""
    import warnings as _warnings

    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "smoke_owed", REPO / "tests" / "test_smoke_run.py"
    )
    smoke = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(smoke)

    # Phase 25's entry is present as the worked example and is satisfied.
    assert "p25_contrasts" in smoke.OWED_TASK_KINDS
    why, predicate = smoke.OWED_TASK_KINDS["p25_contrasts"]
    assert predicate() is False
    assert "not derivable here" in why

    # Every entry carries a predicate, so an exemption expires on its
    # condition rather than on a date.
    for kind, (reason, still_owed) in smoke.OWED_TASK_KINDS.items():
        assert callable(still_owed), kind
        assert reason.strip(), kind

    # And the mechanism really warns when something IS owed.
    with _warnings.catch_warnings(record=True) as caught:
        _warnings.simplefilter("always")
        _warnings.warn("OWED: task kind 'stub' has no shipped config -- why",
                       smoke.OwedTaskKind)
    assert len(caught) == 1
    assert issubclass(caught[0].category, UserWarning)

    ruling = phase25.THE_EXEMPTION_IS_NOISY_NOW
    taken = _flat(ruling["option_1_TAKEN_print_what_is_owed_on_every_run"])
    assert "warnings.warn" in taken
    assert "noisy means noisy without a flag" in taken
    rejected = _flat(ruling["option_2_RECORDED_NOT_TAKEN_a_time_cap"])
    assert "SUDDEN FAILURE AT AN ARBITRARY DATE" in rejected
    assert "a different silence with a worse ending" in rejected
    assert "does not make an owed config acceptable" in _flat(
        ruling["what_it_does_NOT_do"]
    )
    assert "green suite is how this project reads its own state" in _flat(
        ruling["what_went_wrong"]
    )


def test_the_arm_rollups_are_keyed_by_run_directory_not_by_arm():
    """**An arm cannot be handed another arm's hash without changing the
    path it reads.** The extraction fill used this discipline; the run
    fill needs it more, because the two runs differ by 136 bytes in
    11.4 MB and are otherwise identical in shape."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "gen_p25h", REPO / "scripts" / "generate_phase25_configs.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert set(module.RUN_ROLLUPS) == {
        "p25_arm_d2__2d54e77a__p25-arm-d2-2",
        "p25_arm_d1__2d54e77a__p25-arm-d1-2",
    }
    for arm, path in module.RUN_DIRECTORIES.items():
        name = path.rsplit("/", 1)[-1]
        assert name.startswith(arm), (arm, name)
        assert module.run_rollup(path) == module.RUN_ROLLUPS[name]

    # Two directories sharing a rollup is refused, not emitted.
    original = dict(module.RUN_ROLLUPS)
    try:
        shared = original["p25_arm_d2__2d54e77a__p25-arm-d2-2"]
        for key in module.RUN_ROLLUPS:
            module.RUN_ROLLUPS[key] = shared
        with pytest.raises(module.DriftError, match="pasted twice"):
            module.build()
    finally:
        module.RUN_ROLLUPS.clear()
        module.RUN_ROLLUPS.update(original)

    texts, placeholders = module.build()
    assert placeholders == []
    assert len(texts) == 5


def test_the_arm_dict_defect_is_recorded_with_its_class():
    record = phase25.THE_ARM_DICT_LACKED_THE_LOADERS_KEY
    mechanism = _flat(record["the_mechanism"])
    assert "must carry ``csv``" in mechanism
    assert "KeyError: 'csv'" in mechanism

    value = _flat(record["the_correct_value_READ_not_assumed"])
    assert "phase3.write_outputs" in value
    assert "Read from the WRITER" in value

    why = _flat(record["why_it_is_a_CONFIG_field_and_not_a_task_constant"])
    assert "461 shipped arm entries" in why
    assert "identity_predictions" in why

    assert "Phase 21 and 22 depend on it" in _flat(
        record["the_loader_was_NOT_loosened"]
    )
    klass = _flat(record["the_class_and_the_pattern"])
    assert "fourth of one class" in klass
    assert "each one layer deeper" in klass
    ends = _flat(record["what_ends_it"])
    assert "Pre-fix it raises ``KeyError: 'csv'``" in ends

    # The 461/7 split is a measurement of the shipped configs, so check it.
    import glob

    counts = {"predictions": 0, "identity_predictions": 0}
    for path in glob.glob(str(REPO / "configs" / "*.yaml")):
        for line in pathlib.Path(path).read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped.startswith("csv:"):
                counts[stripped.split(":", 1)[1].strip()] += 1
    assert counts["identity_predictions"] == 7, counts
    assert counts["predictions"] >= 461, counts


# --------------------------------------------------------------------------
# [2026-09-02] The close-out
# --------------------------------------------------------------------------


def test_the_three_verdicts_are_banked_with_both_conditions():
    record = phase25.THE_THREE_VERDICTS_OBSERVED
    assert record["run"] == "p25_contrasts__c41d40a6__p25-contrasts"
    verdicts = record["verdicts"]
    expected = {
        "d2-vs-probe": (-0.1194, False, True),
        "d1-vs-probe": (-0.0127, False, False),
        "d2-vs-d1": (-0.1067, False, True),
    }
    assert set(verdicts) == set(expected)
    for key, (delta, c1, c2) in expected.items():
        row = verdicts[key]
        assert row["mean_delta"] == delta, key
        assert row["condition_1"] is c1, key
        assert row["condition_2"] is c2, key
        assert row["verdict"] == "unresolved", key
    # Every contrast in the family has a verdict, and only those.
    assert set(verdicts) == {c["key"] for c in phase25.contrast_family()}

    # The deltas reconcile with the arm means -- the free check.
    assert round(0.1326 - 0.2520, 4) == -0.1194
    assert round(0.2393 - 0.2520, 4) == -0.0127
    assert round(0.1326 - 0.2393, 4) == -0.1067
    assert "a free check, and it passes" in _flat(
        record["the_deltas_reconcile_with_the_arm_means"]
    )
    assert "condition 1 fails on all three" in _flat(
        record["all_three_UNRESOLVED"]
    )


def test_the_neither_cell_fired_and_the_reading_is_verbatim():
    record = phase25.THE_NEITHER_CELL_FIRED
    quoted = _flat(record["the_committed_reading_VERBATIM"])
    registered = _flat(phase25.READINGS_COMMITTED["cell_NEITHER"])
    for fragment in (
        "the EXPECTED outcome",
        "twelfth and thirteenth measurements agree with the eleven",
        "CLOSES BY MEASUREMENT rather than by assumption",
    ):
        assert fragment in registered, fragment
        assert fragment in quoted, fragment
    assert "predicts UNRESOLVED, on all three" in _flat(
        record["it_was_PREDICTED_before_the_arms_ran"]
    )
    assert "withdrawn last cycle" in _flat(record["no_cell_was_amended"])


def test_the_axis_count_is_derived_and_the_discrepancy_is_named():
    """Deriving rather than incrementing is what found it."""
    eleven = phase25.THE_ELEVEN_AND_THE_SHARPEST["the_eleven"]
    assert len(eleven) == 11
    assert "not feature SOURCE" in eleven["11_objective_phase_22"]
    assert len(eleven) - 1 == 10
    assert 10 + 2 == 12
    assert len(eleven) + 2 == 13

    record = phase25.THE_AXIS_CLOSES_DERIVED
    recount = _flat(record["the_recount_and_what_it_found"])
    assert "ELEVEN entries, of which TEN vary the feature source" in recount
    assert "strict feature-source count is 10 + 2 = TWELVE" in recount
    assert "11 + 2 = THIRTEEN" in recount
    assert "THIRTEEN is what the registered sentence means" in _flat(
        record["which_number_goes_where"]
    )
    buys = _flat(record["what_closing_it_BUYS"])
    assert "untested assumption" in buys
    assert "an assumption invites the question, a null answers it" in buys
    assert "self-supervision in general" in _flat(
        record["what_it_does_NOT_close"]
    )


def test_the_seed_instability_is_an_observation_with_its_arithmetic():
    record = phase25.THE_SEED_INSTABILITY_OBSERVED
    arithmetic = _flat(record["the_arithmetic"])
    assert "0.0469" in arithmetic and "0.0148" in arithmetic
    assert "3.17x" in arithmetic and "2.16x" in arithmetic
    assert round(0.0469 / 0.0148, 2) == 3.17
    assert round(0.0319 / 0.0148, 2) == 2.16

    says = _flat(record["what_it_says"])
    assert "LESS STABLE across seeds" in says
    assert "instability is not neutral, it costs resolution" in says
    assert "none of the four combination cells" in _flat(
        record["no_reading_anticipated_it"]
    )
    mechanism = _flat(record["it_is_NOT_a_mechanism"])
    assert "nothing in this phase tests WHY" in mechanism
    assert "naming them here is not proposing them" in mechanism
    assert "0.0055" in _flat(record["the_withdrawn_claim_it_replaces"])


def test_the_condition_split_is_derived_from_the_ledgers_own_fields():
    from cleft import results_ledger

    derived = [
        index for index, entry in enumerate(results_ledger.ENTRIES)
        if str(entry.get("condition_1") or "").strip().upper().startswith("FALSE")
        and str(entry.get("condition_2") or "").strip().upper().startswith("TRUE")
    ]
    assert derived == [25, 26, 27, 28, 31, 32, 34, 35]

    record = phase25.THE_CONDITION_SPLIT_RECOUNTED
    assert record["ledger_rows"] == len(derived) == 8
    assert record["instances_including_descriptive"] == 10
    assert "reproduces entry 37" in _flat(record["the_derivation"])
    assert "a boolean test returns zero" in _flat(record["the_derivation"])
    assert "register no ledger rows" in _flat(
        record["the_two_new_instances_are_NOT_rows"]
    )
    claim = str(results_ledger.ENTRIES[37]["claim"])
    assert "25, 26, 27, 28, 31, 32, 34, 35" in claim
    # [2026-09-06] Was `assert len(results_ledger.ENTRIES) == 38`, a
    # global count standing in for Phase 25 banked no ledger row. It went stale when Phase 27
    # appended entry 39. Rewritten as the relation, which does not.
    assert not [
        e for e in results_ledger.ENTRIES
        if e["phase"] == "p25" or "p25-" in e["id"]
    ]


def test_the_withdrawal_sweep_finds_nothing_current():
    """Classify by the record's status, not by the digits' presence."""
    tokens = ("0.3176", "0.3080", "0.3210", "0.3223", "0.3168", "0.3200",
              "0.0055", "D1 above, D2 not")
    allowed = {
        "THE_FABRICATED_FIGURES_WITHDRAWN",
        "THE_TRANSCRIPTION_SLIP",
        "THE_UNEXPECTED_CELL",
        "WHAT_THE_ARMS_ACTUALLY_SHOW",
        "THE_SEED_INSTABILITY_OBSERVED",
        "THE_WITHDRAWAL_PROPAGATION_SWEPT",
    }
    for name in dir(phase25):
        if not name.isupper():
            continue
        blob = str(getattr(phase25, name))
        if any(token in blob for token in tokens):
            assert name in allowed, name
    assert "WITHDRAWN_2026_09_02" in phase25.THE_UNEXPECTED_CELL

    for folder, pattern in ((REPO / "src", "*.py"),
                            (REPO / "configs", "*.yaml"),
                            (REPO / "scripts", "*.py")):
        for path in folder.rglob(pattern):
            if path.name == "phase25.py":
                continue
            body = path.read_text(encoding="utf-8", errors="ignore")
            for token in tokens:
                assert token not in body, (path.name, token)

    record = phase25.THE_WITHDRAWAL_PROPAGATION_SWEPT
    assert "NOT ONE carries it as current" in _flat(record["the_finding"])
    assert "does not say whether the record around them asserts" in _flat(
        record["why_a_sweep_and_not_a_grep"]
    )


def test_the_void_launch_is_void_and_not_superseded():
    record = phase25.CONTRAST_LAUNCH_VOID
    assert record["run"] == "p25_contrasts__195416d1"
    happened = _flat(record["what_happened"])
    assert "seven attempts" in happened
    assert "Nothing was scored" in happened
    distinction = _flat(record["VOID_not_superseded"])
    assert "produced nothing at all" in distinction
    assert "nothing to supersede" in distinction

    from cleft import results_ledger

    void_ids = [e["id"] for e in results_ledger.ENTRIES if e["status"] == "VOID"]
    assert "void-deall-first-launch" in void_ids
    assert "void-cleftgnn-first-launch" in void_ids


def test_the_closing_walks_nine_criteria_and_records_the_history():
    closing = phase25.PHASE_25_CLOSING
    assert closing["tag"].startswith("[CLOSED]")
    walk = closing["criterion_walk"]
    # [UPDATED 2026-09-06] The walk now also carries four dated
    # corrections to criterion 2. They are not criteria, so they are
    # counted separately rather than folded into the total. The
    # invariant is that all nine criteria are walked.
    criteria = [k for k in walk if "CORRECTED" not in k]
    assert len(criteria) == 9
    assert sorted(int(k.split("_")[0]) for k in criteria) == list(
        range(1, 10)
    )
    corrections = [k for k in walk if "CORRECTED" in k]
    assert len(corrections) == 4
    assert all(k.startswith("2_CORRECTED_2026_09_06") for k in corrections)

    finding = _flat(closing["the_finding"])
    assert "NEITHER SELF-SUPERVISED CHECKPOINT BEATS THE PROBE" in finding
    assert "0.2393" in finding and "0.1326" in finding and "0.2520" in finding
    assert "the phase predicted before the arms ran" in finding

    four = _flat(closing["FOUR_DEFECTS_EACH_FOUND_AT_LAUNCH_ONE_LAYER_DEEPER"])
    for where in ("expected_variant", "check_pairing", "load_config",
                  "first file read"):
        assert where in four, where
    assert "None of them looked further in" in four

    separate = _flat(
        closing["THE_FABRICATION_IS_A_DIFFERENT_FAILURE_AND_IS_NAMED_SEPARATELY"]
    )
    assert "does not belong in that tally" in separate
    assert "no check could have caught it" in separate

    covers = _flat(closing["the_fixture_test_what_it_COVERS"])
    assert "REAL OUTPUT rather than in source" in covers
    not_covers = _flat(closing["the_fixture_test_what_it_does_NOT_cover"])
    assert "cannot see a wrong artifact" in not_covers
    assert "synthetic noise" in not_covers

    assert "no row" in _flat(closing["no_ledger_row"])

    from cleft import results_ledger

    # [2026-09-06] Was `assert len(results_ledger.ENTRIES) == 38`, a
    # global count standing in for Phase 25 banked no ledger row. It went stale when Phase 27
    # appended entry 39. Rewritten as the relation, which does not.
    assert not [
        e for e in results_ledger.ENTRIES
        if e["phase"] == "p25" or "p25-" in e["id"]
    ]
    results_ledger.validate()


# --------------------------------------------------------------------------
# the eighth sequence amendment, 2026-09-05
# --------------------------------------------------------------------------


def test_the_eighth_amendment_schedules_three_and_renumbers_nothing():
    record = phase25.PHASE_SEQUENCE_EXTENDED_8
    assert record["decided"].startswith("2026-09-05")
    assert "EIGHTH" in record["decided"]
    assert "renumbers nothing" in record["decided"]

    assert set(record["becomes"]) == {"26", "27", "28"}
    assert record["was"]["26"].startswith("NOTHING")
    assert "19" not in record["becomes"], "the write-up did not move"
    assert "last by rule" in record["nothing_silently_renumbered"]

    for key in ("phase_26_the_calibration_ablation",
                "phase_27_the_anchor_set_as_a_training_set",
                "phase_28_fine_tuning"):
        assert record[key]["status"] == "SCHEDULED, NOT REGISTERED", key
        # Motivation is recorded; scope and readings are NOT.
        assert "exit_criteria" not in record[key], key
        assert "readings" not in record[key], key


def test_the_eighth_amendment_carries_the_whole_backward_chain():
    """A reader landing on any amendment can walk the chain in both
    directions without knowing how many there are."""
    from cleft import phase11, phase12, phase15, phase20, phase21

    record = phase25.PHASE_SEQUENCE_EXTENDED_8
    chain = {
        "first_amendment": "phase11.PHASE_SEQUENCE_RENUMBERED",
        "second_amendment": "phase12.PHASE_SEQUENCE_RENUMBERED_2",
        "third_amendment": "phase15.PHASE_SEQUENCE_RENUMBERED_3",
        "fourth_amendment": "phase15.PHASE_SEQUENCE_RENUMBERED_4",
        "fifth_amendment": "phase20.PHASE_SEQUENCE_EXTENDED_5",
        "sixth_amendment": "phase21.PHASE_SEQUENCE_EXTENDED_6",
        "seventh_amendment": "phase21.PHASE_SEQUENCE_EXTENDED_7",
    }
    for key, symbol in chain.items():
        assert symbol in record[key], key
        # ...and every entry carries a date, not just a name.
        assert "2026-08" in record[key] or "2026-09" in record[key], key

    # And all seven point FORWARD to this one.
    earlier = (
        phase11.PHASE_SEQUENCE_RENUMBERED,
        phase12.PHASE_SEQUENCE_RENUMBERED_2,
        phase15.PHASE_SEQUENCE_RENUMBERED_3,
        phase15.PHASE_SEQUENCE_RENUMBERED_4,
        phase20.PHASE_SEQUENCE_EXTENDED_5,
        phase21.PHASE_SEQUENCE_EXTENDED_6,
        phase21.PHASE_SEQUENCE_EXTENDED_7,
    )
    for previous in earlier:
        pointer = previous["eighth_amendment"]
        assert "PHASE_SEQUENCE_EXTENDED_8" in pointer
        assert "2026-09-05" in pointer


def test_the_eighth_amendment_binds_the_three_standing_clauses():
    record = phase25.PHASE_SEQUENCE_EXTENDED_8

    scope = record["binding_1_scope_at_the_restate"]
    assert "ITS OWN RESTATE" in scope
    assert "Motivation is not scope" in scope

    cancel = record["binding_2_cancellation_is_recorded"]
    assert "CANCELLED WITH A REASON" in cancel
    assert "never silently dropped" in cancel

    reorder = record["binding_3_reorder_on_evidence"]
    assert "REORDERED OR CANCELLED" in reorder

    assert "SCHEDULED-NOT-SCOPED" in record["scheduled_not_registered"]
    assert "REGISTERED-NOT-BUILT" in record["scheduled_not_registered"]


def test_the_ordering_is_ruled_with_what_fits_before_the_meeting():
    record = phase25.PHASE_SEQUENCE_EXTENDED_8
    order = record["the_ordering_ruled"]
    assert "26 first, then 27, then 28" in order
    assert "2026-09-07" in order

    fits = record["what_fits_before_the_meeting"]
    assert "26 and 27" in fits
    assert "28 is" in fits and "NOT" in fits

    # 26 and 27 are cached; 28 is not, and each says so.
    assert "cached embeddings" in record[
        "phase_26_the_calibration_ablation"]["cost"]
    assert "cached embeddings" in record[
        "phase_27_the_anchor_set_as_a_training_set"]["cost"]
    assert "not cached" in record["phase_28_fine_tuning"]["cost"]


def test_the_amendment_quotes_only_figures_it_can_source():
    """The brief carried two figures the record does not. Both are
    handled rather than repeated."""
    record = phase25.PHASE_SEQUENCE_EXTENDED_8
    flagged = " ".join(record["TWO_FIGURES_NOT_TAKEN_FROM_THE_BRIEF"].split())

    # The count: the brief said fifteen, the record derives twelve/thirteen.
    assert "does not carry fifteen" in flagged
    assert "TWELVE" in flagged and "THIRTEEN" in flagged
    derived = phase25.THE_AXIS_CLOSES_DERIVED["the_recount_and_what_it_found"]
    assert "TWELVE" in derived and "THIRTEEN" in derived
    assert "fifteen" not in derived.lower()

    # Unanimity: reported, not banked, and the phase entry says so.
    anchor = record["phase_27_the_anchor_set_as_a_training_set"]
    reported = " ".join(anchor["and_unanimity_is_REPORTED_not_banked"].split())
    assert "NOT verified unanimous" in reported
    assert "SELECTION CRITERION" in reported

    # Every figure it DOES state is live at the source it names.
    from cleft import ladder, phase12

    assert "0.1662" in ladder.__doc__ or any(
        "0.1662" in str(v) for v in vars(ladder).values() if isinstance(v, str)
    ) or "0.1662" in pathlib.Path(ladder.__file__).read_text(encoding="utf-8")
    assert phase12.STOP_3_REGISTERED["arms"]["head_parameters"]["a"] == 769
