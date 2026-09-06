"""The Phase 7 lattice: one factor per comparison, and the seed rule.

Every claim the ladder makes is a delta between two arms. If a pair differs in
two fields the delta measures neither, and nothing downstream would say so --
the run completes, the BCa interval is computed, and the number goes in a
table. So the one-factor property is asserted over the DERIVED arm set rather
than trusted to the list having been written carefully.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from cleft import ladder
# BACKBONES is the CONSTRUCTABLE registry (7 entries since 7D);
# LADDER_BACKBONES is the closed Phase-7 set the lattice derives from.
from cleft.models.factory import BACKBONES, LADDER_BACKBONES


COMPARED_FIELDS = (
    "backbone", "init", "geometry", "region_scheme", "label", "trainable",
)


def test_every_comparison_varies_exactly_one_field():
    """**The property the whole ladder rests on.**

    Not "the arms look right" -- for each declared comparison, the two arms
    must agree on every compared field except the one it names. A pair
    differing in two is not a measurement of either, and the failure is
    invisible in the output.
    """
    checks = ladder.comparisons()
    assert checks, "the lattice declares no comparisons, so this checks nothing"

    for check in checks:
        a, b = check["a"], check["b"]
        differing = {
            field for field in COMPARED_FIELDS if a[field] != b[field]
        }
        assert differing == {check["varies"]}, (
            f"{check['question']} ({check['stage']}) claims to vary "
            f"{check['varies']!r} but {a['name']} and {b['name']} differ in "
            f"{sorted(differing)}"
        )
        # And they must genuinely differ in it -- an arm compared with itself
        # would satisfy "differs in at most one field" trivially.
        assert a[check["varies"]] != b[check["varies"]]


def test_the_comparisons_cover_every_question_the_brief_asks():
    """A lattice that silently dropped Q2 would pass the one-factor check
    perfectly."""
    by_question: dict[str, int] = {}
    for check in ladder.comparisons():
        by_question[check["question"]] = by_question.get(check["question"], 0) + 1

    # Q1 and Q2 for each of the four backbones -- the central question.
    assert by_question["Q1"] == len(LADDER_BACKBONES) == 4
    assert by_question["Q2"] == len(LADDER_BACKBONES) == 4
    # Both transformers at both geometries: the mechanism test, not a pick.
    assert by_question["geometry"] == 2
    # Three schemes against native.
    assert by_question["scheme"] == 3
    # Median and LDL against mean, at all three operating points.
    assert by_question["label"] == 2
    assert by_question["label_at_imagenet_g1"] == 2
    assert by_question["label_at_imagenet_g2"] == 2


def test_seeds_are_keyed_on_the_regime_not_on_named_arms():
    """[DECIDED 2026-08-01] Graph arms get ten seeds, transformers five,
    because the BAND differs by regime: 0.0251 gives a five-seed threshold of
    0.031, so a graph delta under that is unresolvable at five.

    Keyed on the kind so a new graph arm inherits it -- writing "10" beside
    four named arms is how the next one silently gets five.
    """
    assert ladder.SEEDS_BY_KIND == {"transformer": 5, "graph": 10}
    for arm in ladder.arms():
        assert arm["seeds"] == ladder.SEEDS_BY_KIND[arm["backbone_kind"]]
        assert arm["backbone_kind"] == BACKBONES[arm["backbone"]]["kind"]

    # The rule has teeth only if both regimes are actually present.
    kinds = {arm["backbone_kind"] for arm in ladder.arms()}
    assert kinds == {"transformer", "graph"}


def test_stage_e_uses_the_trained_graph_layer_regime():
    """Brief §5, a named trap: the probe cannot test whether message passing
    benefits from placement, so Stage E must not run on it."""
    stage_e = [arm for arm in ladder.arms() if arm["stage"] == "E"]
    assert len(stage_e) == 4
    for arm in stage_e:
        assert arm["trainable"] == "graph_layers", (
            "Stage E on a probe would compare node layouts through a model "
            "that does no message passing"
        )
        assert arm["task"] == "train_graph_cv"
    assert {arm["region_scheme"] for arm in stage_e} == set(ladder.SCHEMES)


def test_transformers_carry_no_scheme_and_graph_arms_always_do():
    """A region scheme is a graph backbone's node structure. A transformer
    declaring one would be refused by embeddings.check_pairing at run time;
    catching it here costs nothing."""
    for arm in ladder.arms():
        if arm["backbone_kind"] == "graph":
            assert arm["region_scheme"] in ladder.SCHEMES
        else:
            assert arm["region_scheme"] is None


def test_reused_cells_are_identical_arms_not_similar_ones():
    """Stage C's G2 cells, Stage E's native cell and Stage G's mean cell are
    Stage D arms. Reuse is only legitimate if they are the SAME arm on every
    compared field -- otherwise the table reports one run's number under two
    descriptions."""
    by_name = {arm["name"]: arm for arm in ladder.arms() if "reuses" not in arm}
    reused = [arm for arm in ladder.arms() if "reuses" in arm]
    assert reused, "no cell is reused, so this check has nothing to verify"

    for cell in reused:
        original = by_name[cell["reuses"]]
        assert ladder.identity(cell) == ladder.identity(original)
        assert cell["seeds"] == original["seeds"], (
            "a reused cell must inherit the run's seed count, or the table "
            "quotes a precision the run does not have"
        )


def test_cells_and_runs_are_counted_separately():
    """Confusing the two is how an arm list promises more than it ran."""
    summary = ladder.summary()
    assert summary["cells"] > summary["distinct_runs"]
    assert summary["cells"] == len(ladder.arms())
    assert summary["distinct_runs"] == len(ladder.distinct_runs())
    assert summary["by_stage"]["D"]["cells"] == 12 == summary["by_stage"]["D"]["runs"]
    assert summary["by_stage"]["C"] == {"cells": 4, "runs": 2}

    # Total fits is what the cluster spends: seeds summed over RUNS, not cells.
    assert summary["total_fits"] == sum(
        arm["seeds"] for arm in ladder.distinct_runs()
    )


def test_arm_names_are_unique_and_describe_the_arm():
    names = [arm["name"] for arm in ladder.distinct_runs()]
    assert len(names) == len(set(names)), "two runs would write one config"
    for arm in ladder.distinct_runs():
        assert arm["name"].startswith(f"p7_{arm['stage'].lower()}_")
        assert arm["backbone"] in arm["name"]
        assert arm["init"] in arm["name"]
        assert arm["geometry"] in arm["name"]


def test_stage_g1_asks_the_label_question_on_the_strongest_available_arm():
    """**The init choice is an instrument decision, so it is recomputed.**

    Stage G runs the label triple at G2 on ``scut_masked``; Stage G1 runs it
    at G1 on ``imagenet`` instead of holding init constant. The reason is
    power: ``scut_masked`` at G1 is ViT's weakest cell AND has the widest SD
    of the candidates, so a label triple there could claim nothing smaller
    than about 37% of the arm's own value.

    Asserted from the measured cells rather than from the recorded decision,
    so if a re-measurement ever made ``scut_masked`` the better instrument
    this would say so.
    """
    from cleft.train.phase3 import combined_claimable_delta

    g1 = ladder.STAGE_D1_AT_G1
    inits = g1["inits"]
    seeds = ladder.SEEDS_BY_KIND["transformer"]

    def sensitivity(init):
        base = g1["cells"][ladder.LABEL_BACKBONE][inits.index(init)]
        sd = g1["sd"][ladder.LABEL_BACKBONE][init]
        threshold = combined_claimable_delta(sd, seeds, sd, seeds)["arm_means_95"]
        return threshold / base

    chosen = sensitivity(ladder.LABEL_CONTROL_INIT)
    rejected = sensitivity(ladder.LABEL_INIT)
    assert chosen < rejected / 3, (
        f"the chosen init resolves {chosen:.1%} of its arm and the rejected "
        f"one {rejected:.1%}; the decision rests on that gap being large"
    )
    # And it must be the better-MATCHED instrument to the existing G2 triple,
    # which is the part that answers "but you lost comparability".
    g2 = ladder.STAGE_D_AT_G2
    base_g2 = g2["cells"][ladder.LABEL_BACKBONE][g2["inits"].index(ladder.LABEL_INIT)]
    sd_g2 = g2["sd"][ladder.LABEL_BACKBONE][ladder.LABEL_INIT]
    existing = combined_claimable_delta(
        sd_g2, seeds, sd_g2, seeds
    )["arm_means_95"] / base_g2
    assert abs(chosen - existing) < abs(rejected - existing), (
        "the chosen init must be closer in sensitivity to the G2 triple than "
        "the rejected one, or holding init constant would have been better"
    )


def test_stage_g1_is_three_cells_two_runs_and_reuses_the_init_ladder_arm():
    """Its ``mean`` cell IS ``p7_d1_vit_b16_imagenet_g1``, so Stage G1 costs
    two runs and no extraction. The reuse must point that way round -- an
    init-ladder arm recorded as reusing a label arm would invert which stage
    owns the run, and the ordering in ``arms()`` is what decides it."""
    summary = ladder.summary()
    assert summary["by_stage"]["G1"] == {"cells": 3, "runs": 2}

    cells = [arm for arm in ladder.arms() if arm["stage"] == "G1"]
    assert {arm["label"] for arm in cells} == set(ladder.LABELS)
    for arm in cells:
        assert arm["backbone"] == ladder.LABEL_BACKBONE
        assert arm["init"] == ladder.LABEL_CONTROL_INIT
        assert arm["geometry"] == ladder.LABEL_CONTROL_GEOMETRY

    mean_cell = next(arm for arm in cells if arm["label"] == "mean")
    assert mean_cell["reuses"] == "p7_d1_vit_b16_imagenet_g1", (
        f"the mean cell reuses {mean_cell.get('reuses')!r}; it must be the "
        "Stage D1 arm, or the label stage has claimed an init-ladder run"
    )
    # No new extraction: every Stage G1 run reads a set already declared.
    for arm in cells:
        assert _expected_set(arm) in ladder.DECLARED_EMBEDDING_HASHES


def test_every_label_delta_is_recomputed_from_the_two_arms_own_sds():
    """**Both triples, all four deltas, re-derived** -- the same discipline
    every other stage gets, and the reason is that "claimable" is a comparison
    rather than a property of a number.

    Also asserts the headline does not overreach: every measured label delta
    is negative, so "nothing beats the mean" is a statement about the recorded
    cells and not a summary someone wrote.
    """
    from cleft.train.phase3 import combined_claimable_delta

    record = ladder.STAGE_G_LABEL_FORMULATION
    seeds = record["seeds"]

    for name, triple in record["triples"].items():
        base = triple["arms"]["mean"]
        assert set(triple["arms"]) == set(ladder.LABELS)
        assert set(triple["rederived"]) == set(ladder.LABELS) - {"mean"}

        for label, verdict in triple["rederived"].items():
            arm = triple["arms"][label]
            delta = arm["pcc"] - base["pcc"]
            assert verdict["delta"] == pytest.approx(delta, abs=5e-5), (
                f"{name} {label}: recorded delta is not the two arms' difference"
            )
            threshold = combined_claimable_delta(
                base["sd"], seeds, arm["sd"], seeds
            )
            assert verdict["threshold"] == pytest.approx(
                threshold["arm_means_95"], abs=5e-5
            ), f"{name} {label}: the threshold is not its arms'"
            assert verdict["single_run_95"] == pytest.approx(
                threshold["single_run_95"], abs=5e-5
            )
            assert verdict["claimable"] == (
                abs(delta) > threshold["arm_means_95"]
            )
            assert verdict["survives_single_run_95"] == (
                abs(delta) > threshold["single_run_95"]
            )
            # The headline rests on this and nothing else.
            assert delta < 0, (
                f"{name} {label} is POSITIVE -- 'nothing beats the mean' no "
                "longer follows from the cells"
            )


def test_stage_g0_completes_three_corners_and_can_detect_what_it_attributes():
    """**A null is only informative if the arm could have seen the effect.**

    Stage G0 is bought to attribute a disagreement between two triples whose
    measured effects are 0.0595 and 0.0418. If its own thresholds were above
    those, a null would mean "underpowered" and the arm would settle nothing.
    Recomputed here from the base arm's SD and each label arm's own SD as
    measured at masked/g2, which is the best estimate available before it runs.
    """
    from cleft.train.phase3 import combined_claimable_delta

    record = ladder.STAGE_G_LABEL_FORMULATION
    g2 = ladder.STAGE_D_AT_G2
    seeds = record["seeds"]

    base_sd = g2["sd"][ladder.LABEL_BACKBONE][ladder.LABEL_CONTROL_INIT]
    masked = record["triples"]["scut_masked__g2"]

    for label in set(ladder.LABELS) - {"mean"}:
        detectable = combined_claimable_delta(
            base_sd, seeds, masked["arms"][label]["sd"], seeds
        )["arm_means_95"]
        effect = abs(masked["rederived"][label]["delta"])
        assert effect > detectable, (
            f"Stage G0 could not detect {label}'s {effect:.4f} effect at its "
            f"own threshold {detectable:.4f}; a null there would be "
            "uninformative and the arm is not worth running"
        )

    # Three corners, and the fourth deliberately absent: masked/g1 is ViT's
    # weakest cell, where a label triple can only report "unresolved".
    corners = {
        (arm["init"], arm["geometry"])
        for arm in ladder.arms()
        if arm["stage"] in ("G", "G1", "G0")
    }
    assert corners == {
        (ladder.LABEL_INIT, ladder.LADDER_GEOMETRY),
        (ladder.LABEL_CONTROL_INIT, ladder.LABEL_CONTROL_GEOMETRY),
        (ladder.LABEL_CONTROL_INIT, ladder.LABEL_GEOMETRY_CONTROL_GEOMETRY),
    }
    assert (ladder.LABEL_INIT, "g1") not in corners, (
        "the fourth corner is not run on purpose -- it can only be unresolved"
    )


def test_stage_g0_is_three_cells_two_runs_and_reuses_the_init_ladder_arm():
    """Its ``mean`` cell IS ``p7_d_vit_b16_imagenet_g2``, the 0.1347 bar."""
    summary = ladder.summary()
    assert summary["by_stage"]["G0"] == {"cells": 3, "runs": 2}

    cells = [arm for arm in ladder.arms() if arm["stage"] == "G0"]
    mean_cell = next(arm for arm in cells if arm["label"] == "mean")
    assert mean_cell["reuses"] == "p7_d_vit_b16_imagenet_g2"
    for arm in cells:
        assert arm["geometry"] == ladder.LABEL_GEOMETRY_CONTROL_GEOMETRY
        assert arm["init"] == ladder.LABEL_CONTROL_INIT
        # No new extraction.
        assert _expected_set(arm) in ladder.DECLARED_EMBEDDING_HASHES


def test_stage_b_is_recorded_as_a_decision_not_as_a_gap():
    """**An arm list that promises what the project does not deliver is the
    inconsistency that survives into a write-up** -- PLAN Part 6 says so, and
    had to correct it once already when Stage A kept counting two struck arms.

    Stage B is not run and must not appear as one. What it leaves untested
    must stay [REASONED] and stay visible: the claim that the missing basal
    view explains the r~0.3 plateau is accepted as an assumption, not
    measured, and a record that quietly upgraded it would be asserting
    something this project has no measurement for.
    """
    record = ladder.STAGE_B_NOT_RUN
    assert record["status"].startswith("NOT RUN")
    assert record["claim_left_untested"]["provenance"] == "REASONED"
    assert "not measured" in record["claim_left_untested"]["status"]
    assert record["blocker"] and record["write_up"]
    # [2026-08-23] The item's PREMISE turned out to be unsupported -- the
    # correction is dated in place, and the original text stays as what
    # was believed (BASAL_RATIONALE_UNSUPPORTED).
    corrected = record["claim_left_untested"]["corrected"]
    assert corrected.startswith("2026-08-23")
    assert "UNSUPPORTED, not refuted" in corrected
    assert "cannot" in corrected and "be relied on" in corrected

    # And it must genuinely be absent from the derived ladder, or the record
    # is describing a stage the arm list still promises.
    assert not [arm for arm in ladder.arms() if arm["stage"] == "B"]
    assert "B" not in ladder.summary()["by_stage"]


def test_the_basal_rationale_is_downgraded_precisely_and_not_overstated():
    """[2026-08-23] The claim PLAN 4.6 carried as [MEASURED] was never
    measured. Downgraded to UNSUPPORTED -- and deliberately NOT to
    REFUTED, because the sheet cannot distinguish shown-both from
    frontal-only. The record must not overstate in either direction."""
    record = ladder.BASAL_RATIONALE_UNSUPPORTED
    assert record["corrected"] == "2026-08-23"
    # The original text is preserved as what was believed.
    original = record["original_claim_preserved"]
    assert "Basal is core, not optional" in original
    assert "[MEASURED]" in original
    # The downgrade stops at UNSUPPORTED, with the reason.
    assert "UNSUPPORTED" in record["downgrade"]
    assert "NOT refuted" in record["downgrade"]
    assert "must not overstate" in record["downgrade"]

    # The three evidence lines, each tagged as what it is.
    measured = record["measured"]
    assert "523 ABSENT" in measured and "524 present" in measured
    assert "2 adjacent-ID pairs" in measured
    assert "no basal ID receives a score" in measured
    # The contradiction sat in the PLAN itself.
    assert "basal (unscored)" in measured
    literature = record["literature"]
    assert "LATERAL, not basal" in literature
    assert "CAPTURE protocol" in literature
    assert "Frontal-Eye-View" in literature
    unresolved = record["unresolved"]
    assert "nonetheless SHOWN" in unresolved
    assert "could distinguish" in unresolved

    # The linked REASONED item is corrected in place, not silently.
    assert "cannot be relied on" in record["linked_item_corrected"]
    linked = ladder.STAGE_B_NOT_RUN["claim_left_untested"]
    assert linked["corrected"].startswith("2026-08-23")
    # And its original provenance and text still stand, visibly.
    assert linked["provenance"] == "REASONED"
    assert "missing basal view explains" in linked["text"]

    # The error's class: the sixth of its kind, predecessors named, and
    # every named predecessor record actually exists.
    error = record["error_class"]
    assert "SIXTH" in error["instance"]
    assert "tagged with a provenance it did not have" in error["class"]
    assert len(error["predecessors"]) == 5
    from cleft import phase10, phase11

    for reference in error["predecessors"]:
        name = reference.rsplit("(", 1)[1].rstrip(")").split(",")[0]
        if name == "same record":
            continue  # the MODE reading shares the consensus-column record
        module_name, _, attribute = name.partition(".")
        module = {"phase10": phase10, "phase11": phase11}[module_name]
        assert getattr(module, attribute), name

    # The lesson, in the maintainer's terms.
    lesson = record["lesson"]
    assert "A RATIONALE IS NOT A MEASUREMENT" in lesson
    assert "UNFALSIFIABLE IN PRACTICE" in lesson
    assert "nobody audits" in lesson

    # And the question that decides Phase 12 is on the record.
    assert "frontal photograph only" in record["for_supervisor"]
    assert "decides which experiment Phase 12 is" in record["for_supervisor"]


def test_the_phase_sequence_is_renumbered_by_pointer_not_by_rewrite():
    """[2026-08-23] the maintainer's re-sequencing: the ablation takes Phase 12,
    the decoder moves to 13, the write-up to 14. Nothing already written
    is renumbered -- the pointers are dated and the originals visible."""
    from cleft import phase11

    record = phase11.PHASE_SEQUENCE_RENUMBERED
    assert record["decided"].startswith("2026-08-23")
    assert record["was"] == {"12": "decoder/reconstruction", "13": "write-up"}
    assert "VIEW ABLATION" in record["becomes"]["12"]
    assert record["becomes"]["13"] == "decoder/reconstruction"
    assert record["becomes"]["14"] == "write-up"
    assert "inputs already exist" in record["reason"]
    assert "original sequence remains visible" in (
        record["nothing_silently_renumbered"]
    )

    # The closing's original 'next' line survives, with the pointer beside.
    closing = phase11.PHASE_11_CLOSING
    assert "decoder/reconstruction" in closing["next"]
    assert closing["renumbered"].startswith("2026-08-23")
    assert "PHASE_SEQUENCE_RENUMBERED" in closing["renumbered"]

    # The interaction with the same-day correction is stated, not hidden:
    # the renumbering fixes the ablation's place, the correction reopens
    # its question.
    interaction = record["interaction_with_the_correction"]
    assert "BASAL_RATIONALE_UNSUPPORTED" in interaction
    assert "fixes the ablation's PLACE" in interaction
    assert "reopens its QUESTION" in interaction
    assert "decides which" in interaction


def test_an_unknown_backbone_or_init_is_refused_by_name():
    with pytest.raises(ladder.LadderError, match="unknown backbone"):
        ladder._arm("D", "resnet50", "imagenet")
    with pytest.raises(ladder.LadderError, match="unknown init"):
        ladder._arm("D", "vit_b16", "places365")


# --------------------------------------------------------------------------
# the generated configs
# --------------------------------------------------------------------------


def test_the_shipped_configs_match_the_derived_arms(repo_root):
    """**The generator is the source; the configs are output.**

    A hand edit to a generated config would break the one-factor property the
    lattice asserts -- and it would break it invisibly, because the lattice
    test reads the arm list, not the files. `--check` re-derives every config
    and diffs, so the two cannot drift apart silently.
    """
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, str(repo_root / "scripts" / "generate_ladder_configs.py"),
         "--check"],
        capture_output=True, text=True, cwd=str(repo_root),
        env={**__import__("os").environ, "PYTHONPATH": str(repo_root / "src")},
    )
    assert result.returncode == 0, (
        f"the shipped ladder configs differ from the derived arms:\n"
        f"{result.stdout}{result.stderr}"
    )


#: Phase 7 configs that are NOT ladder arms. Listed explicitly so adding one
#: is a deliberate act: a stray p7_* config that is neither an arm nor a
#: declared diagnostic would otherwise sit in the directory looking like part
#: of the ladder, and could be launched as one.
NON_ARM_P7_CONFIGS = {
    # The one-factor variant isolating geometry from feature source. Live
    # extraction, so it is deliberately outside the arm list -- every ladder
    # arm reads a stored artifact.
    "p7_repro_g2_live",
    # The extraction Stage C0 needs: vit_b16__scut_original__g1 is not in
    # embeddings_v1, and a published artifact cannot gain a set.
    "p7_extract_g1_control",
    # The nine Stage D1 needs. A THIRD version, because
    # embeddings_g1_control_v1 already exists with its one set and
    # task_extract_embeddings refuses an out_version that does.
    "p7_extract_g1_ladder",
    # PLAN §4.3 condition 1 over vectors that already exist. Neither fits
    # anything; `headline` is the 1.05x claim alone, `ladder` is all 26.
    # ladder.CLAIMS_REST_ON_HALF_THE_CRITERION.
    "p7_paired_headline",
    "p7_paired_ladder",
    # Phase 8's §0, tested before the phase is designed around it.
    "p7_paired_tradeoff",
    # [2026-08-24] A post-closing ADDENDUM, not an arm: it re-reads the
    # d1 imagenet g1 arm's existing predictions under a second metric
    # family and trains nothing
    # (classification.CLASSIFICATION_METRICS_SECONDARY). It carries the
    # arm's name because it describes that arm's outputs and a reader
    # should find it beside them -- which is exactly why this list has
    # to name it, so it cannot be launched as an arm.
    "p7_d1_classification_metrics",
}


def test_every_distinct_run_has_a_config_and_nothing_else_does(repo_root):
    shipped = {
        p.stem for p in (repo_root / "configs").glob("p7_*.yaml")
    } - NON_ARM_P7_CONFIGS
    derived = {arm["name"] for arm in ladder.distinct_runs()}
    assert shipped == derived, (
        f"shipped-but-not-derived: {sorted(shipped - derived)}; "
        f"derived-but-not-shipped: {sorted(derived - shipped)}"
    )
    # And no config exists for a REUSED cell -- that would be the duplicate
    # run the reuse map exists to prevent.
    reused = {arm["name"] for arm in ladder.arms() if "reuses" in arm}
    assert not (shipped & reused), (
        f"a reused cell has its own config: {sorted(shipped & reused)}"
    )


def test_each_config_declares_the_embedding_set_its_arm_needs(
    repo_root, monkeypatch
):
    """The pairing check refuses a mismatch at run time; this catches it before
    anything is launched, and covers the scheme-matching rule that makes a
    graph arm's set specific to its own scheme."""
    from cleft.config import load_config

    monkeypatch.setenv("CLEFT_SCUT_ROOT", "/nonexistent/sentinel")
    for arm in ladder.distinct_runs():
        cfg = load_config(repo_root / "configs" / f"{arm['name']}.yaml")
        by_name = {e["name"]: e for e in cfg["inputs"]}
        task = cfg["task"]

        embeddings = by_name[task["embeddings_artifact"]]["path"]
        assert embeddings.endswith(f"/{_expected_set(arm)}"), (
            f"{arm['name']} declares {embeddings.rsplit('/', 1)[-1]}, expected "
            f"{_expected_set(arm)}"
        )
        # A pretrained init must pair with a checkpoint; imagenet must not.
        # The schema fills absent optionals with their default, so an
        # undeclared checkpoint is present as None rather than missing.
        if arm["init"] == "imagenet":
            assert task.get("checkpoint") is None, (
                "imagenet has no pretraining checkpoint; declaring one would "
                "hash-verify an input the run never reads"
            )
        else:
            assert task["checkpoint"] in by_name

        assert task["seeds"] == arm["seed_list"]
        assert task["geometry"] == arm["geometry"]
        assert task["label"] == arm["label"]
        assert task["kind"] == arm["task"]


def _expected_set(arm: dict) -> str:
    # The module's own implementation, deliberately. A test that recomputes
    # the rule can only catch a change to it, and this rule is not the one
    # under test here -- what is under test is whether the CONFIG declares the
    # set the arm list says it consumes.
    return ladder.embedding_set_name(arm)


def _shipped_extractions(repo_root, monkeypatch) -> dict:
    """``(out_version, set name) -> config`` over every shipped extraction.

    Keyed on the PAIR: two extraction configs may legitimately produce the
    same set name into different versions -- ``p6_extract_srgnn_adabn.yaml``
    re-extracts ``srgnn__scut_masked__g2__native`` under per-fold BN
    adaptation, which is a different representation of the same arm and lives
    in its own artifact.
    """
    from cleft.config import load_config

    from conftest import sentinel_every_env_root

    sentinel_every_env_root(repo_root, monkeypatch, "extraction-sweep")
    produced: dict[tuple, str] = {}
    for path in sorted((repo_root / "configs").glob("*.yaml")):
        cfg = load_config(path)
        if cfg["task"]["kind"] != "extract_embeddings":
            continue
        for entry in cfg["task"]["sets"]:
            name = _expected_set({
                "backbone": entry["backbone"],
                "init": entry["init"],
                "geometry": entry["geometry"],
                "region_scheme": entry["pretrain_scheme"],
            })
            produced[(cfg["task"]["out_version"], name)] = path.name
    return produced


def test_every_set_an_arm_reads_is_produced_by_a_shipped_extraction(
    repo_root, monkeypatch
):
    """**Reachability, applied to the embedding artifacts** -- the rule that
    already covers checkpoints (``tests/test_extract.py``), and the one whose
    absence let Stage D1's nine arms point at a version that could not hold
    them.

    The version and the set are read from the arm's own config, so this does
    not consult ``ladder.embeddings_version_for``: it was that routing which
    was wrong, and a check that asked the same function would have agreed with
    it. Both halves must match some shipped extraction's ``out_version`` and
    set list.

    **What it catches, concretely.** Nine G1 sets were routed to
    ``embeddings_g1_control_v1``, whose extraction config produces exactly one
    set. Nothing on the laptop objected. The failure would have been an
    aborted cluster run -- ``task_extract_embeddings`` refuses an out_version
    that exists (PLAN §2.6) -- after the arms had been launched and refused by
    guard 3 for a hash that no extraction was ever going to produce.
    """
    from cleft.config import load_config

    produced = _shipped_extractions(repo_root, monkeypatch)
    assert produced, "no shipped extraction config found; this checks nothing"

    unreachable = []
    for arm in ladder.distinct_runs():
        cfg = load_config(repo_root / "configs" / f"{arm['name']}.yaml")
        by_name = {e["name"]: e for e in cfg["inputs"]}
        declared = by_name[cfg["task"]["embeddings_artifact"]]["path"]
        version, _, set_name = declared.rpartition("/")
        version = version.rsplit("/", 1)[-1]
        if (version, set_name) not in produced:
            unreachable.append(f"{arm['name']} -> {version}/{set_name}")

    assert not unreachable, (
        "these arms read an embedding set no shipped extraction config "
        f"produces at that version: {unreachable}. Either the extraction is "
        "missing the set, or the arm names the wrong artifact version -- and "
        "both fail on the cluster rather than here."
    )


def test_the_three_embedding_versions_are_produced_by_three_extractions(
    repo_root, monkeypatch
):
    """**A version is a batch of extraction, and each batch runs once.**

    ``embeddings_v1`` holds the plan's seventeen, ``embeddings_g1_control_v1``
    Stage C0's one, ``embeddings_g1_ladder_v1`` Stage D1's nine. Two configs
    writing one version is the state that cannot exist: the second aborts on
    the immutability guard, and which one aborts depends on launch order.
    """
    from cleft.config import load_config

    from conftest import sentinel_every_env_root

    sentinel_every_env_root(repo_root, monkeypatch, "version-sweep")
    writers: dict[str, list[str]] = {}
    for path in sorted((repo_root / "configs").glob("*.yaml")):
        cfg = load_config(path)
        if cfg["task"]["kind"] != "extract_embeddings":
            continue
        writers.setdefault(cfg["task"]["out_version"], []).append(path.name)

    collisions = {v: names for v, names in writers.items() if len(names) > 1}
    assert not collisions, (
        f"one artifact version, two extraction configs: {collisions}. The "
        "second to run aborts on PLAN §2.6, so which of them exists depends "
        "on the order they were launched in"
    )

    for version in (
        ladder.DEFAULT_EMBEDDINGS_VERSION,
        ladder.G1_CONTROL_EMBEDDINGS_VERSION,
        ladder.G1_LADDER_EMBEDDINGS_VERSION,
    ):
        assert version in writers, f"nothing shipped produces {version}"


def test_the_stage_d1_extraction_equals_the_derivation(repo_root, monkeypatch):
    """The config is the provenance record, so it enumerates its sets -- and
    this keeps that list equal to the derivation, exactly as
    ``tests/test_extract.py`` does for the Phase 6 batch.

    Nine, not twelve: three of Stage D1's cells reuse runs that already exist
    (Stage C's two masked-G1 transformers and Stage C0's), so their sets are
    already extracted. ``g1_ladder_sets`` walks the DISTINCT runs, where that
    reuse is already resolved.
    """
    from cleft.config import load_config
    from cleft.train.extract import checkpoint_input_name

    from conftest import sentinel_every_env_root

    sentinel_every_env_root(repo_root, monkeypatch, "producer-sweep")
    cfg = load_config(repo_root / "configs" / "p7_extract_g1_ladder.yaml")

    fields = ("backbone", "init", "geometry", "pretrain_scheme")
    declared = [{f: entry[f] for f in fields} for entry in cfg["task"]["sets"]]
    derived = [{f: entry[f] for f in fields} for entry in ladder.g1_ladder_sets()]
    assert declared == derived, "the shipped set list drifted from the ladder"
    # **[DECIDED 2026-08-02] Nine, not eight.** Citing the live 0.2521 for the
    # ViT ImageNet G1 cell instead of running it would leave that one arm
    # differing from the other eight in FEATURE SOURCE while the lattice
    # recorded only geometry and init -- the declared-fields-said-one-factor
    # problem that MASKED_G1_ARTEFACT is a record of. One extra run to keep
    # the lattice uniform is cheap; an undeclared second factor is not.
    assert len(declared) == 9

    assert cfg["task"]["out_version"] == ladder.G1_LADDER_EMBEDDINGS_VERSION
    # Stage D at G2 was extracted without per-fold BN adaptation. Turning it
    # on for the G1 batch alone would make every D-against-D1 delta vary the
    # representation as well as the geometry.
    assert cfg["task"]["per_fold_bn_reestimation"] is False

    # Every pretrained set's checkpoint is declared by its conventional name,
    # and nothing else is -- an undeclared one aborts the extraction partway,
    # after some sets have been written into a version that then cannot be
    # rewritten.
    from cleft.embeddings import expected_variant

    needed = {
        checkpoint_input_name({
            "backbone": entry["backbone"],
            "variant": expected_variant(entry["init"], entry["geometry"]),
            "pretrain_scheme": entry["pretrain_scheme"],
        })
        for entry in ladder.g1_ladder_sets()
        if expected_variant(entry["init"], entry["geometry"]) is not None
    }
    input_names = {entry["name"] for entry in cfg["inputs"]}
    assert needed <= input_names, sorted(needed - input_names)
    assert input_names - needed == {"manifest_v1", "staged_v1"}


def test_a_swin_arm_cannot_run_on_live_extraction(repo_root, monkeypatch):
    """**Live extraction builds ViT-B/16 and nothing else.** A Swin arm without
    an embeddings_artifact would train on ViT features while metrics.json said
    swin_b -- every shape right, the number attributed to a model that never
    ran. Refused at load time."""
    from cleft.config.schema import ConfigError, validate

    base = {
        "schema_version": 1, "phase": "p7", "tier": "dev", "seed": 1337,
        # [2026-09-01] schema.validate now refuses a task that names an
        # input the config does not declare, so the scaffold declares
        # its placeholders ("e" is referenced by the second case below).
        "inputs": [{"name": n, "path": f"/tmp/{n}", "rollup_sha256": "ab" * 32}
                   for n in ("m", "s", "e")],
        "task": {
            "kind": "train_cv", "manifest_artifact": "m", "staged_artifact": "s",
            "geometry": "g2", "label": "mean", "backbone": "swin_b",
            "max_epochs": 10, "patience": 3, "inner_val_frac": 0.2,
            "monitor": "inner_val_mse",
        },
    }
    with pytest.raises(ConfigError, match="Live extraction builds ViT-B/16 only"):
        validate(base)

    # With an artifact and its checkpoint it validates.
    base["task"].update(
        embeddings_artifact="e", init="scut_masked", checkpoint="c"
    )
    assert validate(base)["task"]["backbone"] == "swin_b"


def test_the_declared_hash_registry_is_well_formed_and_covers_the_ladder():
    """**One hash per path, recorded once.**

    The registry exists so the cross-config invariant is structural rather
    than remembered: `vit_b16__scut_masked__g2` is declared by three configs,
    and pasting into each by hand is how the one observed drift happened.
    """
    registry = ladder.DECLARED_EMBEDDING_HASHES
    provenance = ladder.DECLARED_EMBEDDING_PROVENANCE

    # 26 across THREE artifact versions -- 16 for the original ladder, the G1
    # control set Stage C0 needed, and Stage D1's nine. The per-batch counts
    # must sum to the total, so a set added without recording where it came
    # from fails here.
    assert len(registry) == provenance["n_sets"] == 26
    assert sum(b["n_sets"] for b in provenance["batches"]) == len(registry)
    # Both the count and the RELATION: two batches naming one artifact version
    # cannot exist, because the second to run aborts on PLAN §2.6. The
    # relation is the part that does not go stale as batches land.
    assert len({b["artifact"] for b in provenance["batches"]}) == len(
        provenance["batches"]
    ) == 3
    for name, rollup in registry.items():
        assert len(rollup) == 64 and set(rollup) <= set("0123456789abcdef"), (
            f"{name} is not a sha256 digest"
        )
        assert rollup != "0" * 64, f"{name} is still a placeholder"
    assert len(set(registry.values())) == len(registry), (
        "two sets share a rollup, which would mean two artifacts are "
        "byte-identical -- possible, but worth knowing rather than assuming"
    )

    # It must cover every set the ladder needs, EXCEPT the one deliberately
    # left to its existing declaration.
    needed = set()
    for arm in ladder.distinct_runs():
        from cleft import embedding_plan

        needed.add(embedding_plan.set_name({
            "backbone": arm["backbone"], "init": arm["init"],
            "geometry": arm["geometry"],
            "pretrain_scheme": (
                arm["region_scheme"]
                if arm["backbone_kind"] == "graph" and arm["init"] != "imagenet"
                else None
            ),
        }))
    absent = provenance["absent_because_already_declared"]
    # **The G1 control extraction has RUN [2026-08-01], so its set is in the
    # registry like every other.** This briefly excluded sets awaiting
    # extraction; that allowance was correct for exactly as long as one was
    # pending, which is the cycle this project keeps meeting. Reverted rather
    # than kept: an allowance that outlives its cause silently permits the
    # next unfilled hash.
    off_version = {
        _expected_set(arm) for arm in ladder.distinct_runs()
        if arm["embeddings_version"] != ladder.DEFAULT_EMBEDDINGS_VERSION
    }
    # Sets outside embeddings_v1 are extracted in batches; those already
    # extracted are in the registry, the rest are pending. Derived, so the
    # allowance shrinks by itself as batches land.
    pending = off_version - set(registry)
    needed = needed - pending
    assert set(registry) == needed - {absent}, (
        f"registry covers {sorted(set(registry) - needed)} it does not need, "
        f"and misses {sorted(needed - set(registry) - {absent})}"
    )
    assert absent in needed, (
        "the set excused from the registry must still be one the ladder uses, "
        "or the exemption is describing nothing"
    )


def test_an_unresolvable_path_never_carries_a_real_hash(repo_root, monkeypatch):
    """**A hash for a path nobody can resolve was not read from a file.**

    That is the invented-provenance failure the ``UNRESOLVED_GLOB`` marker
    exists to make impossible, and it is the half of this check that can never
    legitimately flip: no state of the repo makes it correct to write a digest
    for a file this machine cannot name.

    ----------------------------------------------------------------------
    THE OTHER HALF WAS REMOVED RATHER THAN SWUNG A FOURTH TIME
    ----------------------------------------------------------------------
    This test also used to assert **no placeholder on a resolvable path**, and
    that clause alternated four times in two days: zero-placeholders, a Stage
    D1 allowance, zero-placeholders again when D1's rollups landed, a Phase 7B
    allowance keyed on the pooling artifact version -- and then the baseline
    predictions arrived with **resolved paths and pending hashes**, a state
    none of those four forms had a place for.

    That is not four mistakes. It is a state assertion tracking a state that
    legitimately changes, which is the pattern PLAN R7's tally is a list of.
    A hash is obtainable only on the machine holding the data; **every**
    cluster input passes through "path known, hash pending" on its way in, so
    a check that forbids that state forbids the normal way work arrives here.

    What survives is durable and lives in ``tests/test_smoke_run.py``: a path
    declares the same hash in **every** config (true while unbuilt, true once
    built, false only during the drift worth catching), and a placeholder is
    **documented** (one-directional, vacuous once filled). Between them a
    forgotten sibling and an unexplained placeholder are both caught, in both
    states, with nothing to maintain.
    """
    from cleft.config import load_config

    from conftest import sentinel_every_env_root

    sentinel_every_env_root(repo_root, monkeypatch, "unresolvable-sweep")
    marker = "UNRESOLVED_GLOB"

    offenders, unresolvable, inspected = [], 0, 0
    for path in sorted((repo_root / "configs").glob("*.yaml")):
        for entry in load_config(path)["inputs"]:
            inspected += 1
            if marker not in entry["path"]:
                continue
            unresolvable += 1
            if entry["rollup_sha256"] != "0" * 64:
                offenders.append(f"{path.name}:{entry['name']}")

    # A sweep that inspected nothing is green for the wrong reason -- lesson 5
    # in R7's tally, and it arrived from a shrunk fixture rather than a wrong
    # assertion, so the guard is on the input rather than the conclusion.
    assert inspected > 100, (
        f"only {inspected} declared inputs found; the sweep is not covering "
        "the shipped configs"
    )
    assert not offenders, (
        f"unresolvable paths carrying a real hash: {offenders}. The path "
        "contains {marker}, so nobody could have read that file to hash it."
    )


def test_the_unresolvable_path_check_has_teeth():
    """It is vacuous whenever no config carries the marker, which is most of
    the time -- so the predicate is exercised directly rather than left to
    look green. Failure mode 5 in R7's tally, built against on purpose."""
    marker = "UNRESOLVED_GLOB"
    rows = [
        ("a.yaml", "ckpt", f"/runs/{marker}/x__*/f.npz", "0" * 64),
        ("b.yaml", "ckpt", f"/runs/{marker}/y__*/f.npz", "ab" + "0" * 62),
        ("c.yaml", "data", "/data/real/v1", "cd" + "0" * 62),
    ]
    offenders = [
        f"{config}:{name}"
        for config, name, path, rollup in rows
        if marker in path and rollup != "0" * 64
    ]
    assert offenders == ["b.yaml:ckpt"], (
        "the predicate must flag a real hash on an unresolvable path, and "
        "only that"
    )


def test_every_set_the_ladder_reads_is_declared_in_the_registry(repo_root):
    """The complement of the sweep above, from the other direction: no arm may
    read a set whose hash nobody recorded.

    Between them the two say the repo is fully declared -- every config input
    carries a real hash, and every set an arm consumes has a registry entry
    that a person obtained deliberately on the cluster.
    """
    undeclared = sorted(
        _expected_set(arm) for arm in ladder.distinct_runs()
        if _expected_set(arm) not in ladder.DECLARED_EMBEDDING_HASHES
        and _expected_set(arm)
        != ladder.DECLARED_EMBEDDING_PROVENANCE["absent_because_already_declared"]
    )
    assert not undeclared, (
        f"arms read sets with no registry entry: {undeclared}"
    )


# --------------------------------------------------------------------------
# the reproduction diagnostics
# --------------------------------------------------------------------------


def test_the_g2_live_variant_differs_from_p3_in_geometry_alone(
    repo_root, monkeypatch
):
    """**The only way to attribute the ladder's depression.**

    Fifteen arms came back low against a bar that is G1-live while every arm
    is G2-artifact -- two factors at once. This isolates geometry: p3_train_cv
    with `geometry: g2` and nothing else moved, still LIVE extraction. If it
    differed in a second field the result would attribute nothing.

    The seed truncation is not a second factor: five is a PREFIX of
    p3_train_cv's ten, so the two are comparable seed-for-seed.
    """
    from cleft.config import load_config

    monkeypatch.setenv("CLEFT_SCUT_ROOT", "/nonexistent/sentinel")
    base = load_config(repo_root / "configs" / "p3_train_cv.yaml")["task"]
    variant = load_config(repo_root / "configs" / "p7_repro_g2_live.yaml")["task"]

    differing = {
        key for key in set(base) | set(variant)
        if base.get(key) != variant.get(key)
    }
    assert differing == {"geometry", "seeds"}, (
        f"the variant differs in {sorted(differing)}; it must isolate geometry"
    )
    assert base["geometry"] == "g1" and variant["geometry"] == "g2"
    assert variant["seeds"] == base["seeds"][: len(variant["seeds"])], (
        "the variant's seeds must be a PREFIX of p3_train_cv's, or the two "
        "cannot be compared seed-for-seed"
    )

    # It must stay on LIVE extraction -- an artifact would reintroduce the
    # very factor it exists to hold constant.
    assert not variant.get("embeddings_artifact")
    assert variant.get("init", "imagenet") == "imagenet"
    assert variant["backbone"] == "vit_b16"

    # And it is NOT a ladder arm: the generator must not claim it, and it must
    # be declared as a diagnostic rather than sitting in configs/ looking like
    # an arm that could be launched as one.
    assert "p7_repro_g2_live" not in {
        arm["name"] for arm in ladder.distinct_runs()
    }
    assert "p7_repro_g2_live" in NON_ARM_P7_CONFIGS

    # Every declared non-arm must actually exist, or the exemption is
    # excusing a config that is not there.
    for stem in NON_ARM_P7_CONFIGS:
        assert (repo_root / "configs" / f"{stem}.yaml").is_file()


def test_the_comparison_script_reports_numbers_not_a_bare_verdict(repo_root):
    """A 0.999 correlation at a scale factor of 3 is a different problem from
    a 0.2 correlation, and collapsing both into 'differs' discards the part
    that says which. The script must surface the distinguishing quantities."""
    source = (
        repo_root / "scripts" / "compare_live_and_stored_features.py"
    ).read_text(encoding="utf-8")
    for quantity in (
        "norm_ratio", "mean_per_row_correlation", "max_abs_diff",
        "stored_dtype", "permutation_check",
    ):
        assert quantity in source, f"the comparison must report {quantity}"

    # It must compare against BOTH geometries -- an artifact named __g2 built
    # from G1 pixels passes every hash, shape and pairing check downstream, so
    # matching the wrong one is the finding.
    assert 'for geometry in ("g1", "g2")' in source


def test_every_resolved_difference_is_declared_not_just_the_named_field():
    """**The blind spot the geometry finding exposed.**

    ``test_every_comparison_varies_exactly_one_field`` compares the declared
    CONFIG FIELDS, and Stage C passed it cleanly -- both cells declare
    ``init: scut_masked``. But masked inits are GEOMETRY-BOUND, so the two
    cells resolve to DIFFERENT checkpoints (masked_g1 at 0.7893 pretraining,
    masked_g2 at 0.8306). Stage C therefore varies geometry AND weights while
    reporting one factor, and the field-level check could not see it.

    That matters for the reading, not just for tidiness: the ImageNet pair
    gives geometry costing 0.1174 with G1 better, and the masked pair gives
    0.1171 with G2 better. Those look like a contradiction and are not -- they
    are different quantities (R2), and only this check makes the difference
    visible in the lattice rather than in a person's head.
    """
    for check in ladder.comparisons():
        ra = ladder.resolved_inputs(check["a"])
        rb = ladder.resolved_inputs(check["b"])
        differing = {k for k in ra if ra[k] != rb[k]}
        entailed = ladder.ENTAILED[check["varies"]]
        # The embedding set encodes every factor, so it always differs and is
        # never an extra factor in its own right.
        surprising = differing - entailed - {"embedding_set"}

        assert surprising == set(check["also_varies"]), (
            f"{check['question']} ({check['stage']}) resolves to differing "
            f"{sorted(surprising)} but declares also_varies="
            f"{check['also_varies']}"
        )
        # Every extra factor must carry a recorded licence -- a second factor
        # is either a deliberate design decision with a reason, or a defect.
        for extra in check["also_varies"]:
            assert check["licences"][extra], (
                f"{check['question']} varies {extra} beyond {check['varies']} "
                "with no recorded licence"
            )


def test_stage_c_declares_that_it_varies_the_checkpoint_too():
    """Stage C is a matched-PIPELINE comparison, not a geometry measurement,
    and the licence must say so -- otherwise its number gets read as "the
    geometry effect" and contradicts the ImageNet pair, which IS one."""
    geometry_checks = [
        c for c in ladder.comparisons() if c["question"] == "geometry"
    ]
    assert geometry_checks
    for check in geometry_checks:
        assert check["also_varies"] == ["checkpoint"]
        licence = check["licences"]["checkpoint"]
        assert "GEOMETRY-BOUND" in licence
        assert "NOT a measurement of geometry alone" in licence
        assert "imagenet" in licence, (
            "the licence must point at the pair that IS a pure-geometry "
            "measurement, or a reader has no alternative to misreading this one"
        )


def test_stage_e_declares_its_checkpoint_variation_and_that_it_is_unlicensed():
    """**Inverted 2026-08-01, and this one is a correction rather than a
    refresh.** This asserted that SCHEME_AXIS_AT_PRETRAINING licensed Stage
    E's checkpoint variation. Stage E0 measured that inference false: the four
    checkpoints, statistically equal on 2,199 SCUT faces, spread 0.095 at
    cleft time while placement contributes 0.016.

    So the licence text must now REFUSE rather than grant, and point at the
    controlled comparison. A stale licence here would keep Stage E's ordering
    readable as a placement result in the one place a reader checks."""
    scheme_checks = [c for c in ladder.comparisons() if c["question"] == "scheme"]
    assert len(scheme_checks) == 3
    for check in scheme_checks:
        assert check["also_varies"] == ["checkpoint"]
        licence = check["licences"]["checkpoint"]
        assert "NOT LICENSED" in licence
        assert "MEASURED FALSE" in licence
        assert "Stage E0 is the controlled comparison" in licence

    # And the controlled stage exists, with the checkpoint genuinely held.
    e0 = [c for c in ladder.comparisons() if c["question"] == "scheme_controlled"]
    assert len(e0) == 3
    for check in e0:
        assert check["also_varies"] == [], (
            "Stage E0 exists precisely to vary placement alone"
        )


# --------------------------------------------------------------------------
# Stage D at G2
# --------------------------------------------------------------------------


def test_the_stage_d_deltas_are_derivable_from_its_cells():
    """**A recorded delta and the cells it came from are two chances to make
    one typo.**

    Q1 is ``original - imagenet`` and Q2 is ``masked - original``, so both are
    functions of the twelve cells. Recomputing them here means the record
    cannot carry a delta that its own table does not produce -- which is the
    failure that puts a wrong number in a summary table and a right one in the
    run directory it claims to summarise.
    """
    record = ladder.STAGE_D_AT_G2
    cells = record["cells"]
    assert set(cells) == set(LADDER_BACKBONES), "the record must cover every backbone"
    assert record["inits"] == ladder.INITS

    for backbone, (imagenet, original, masked) in cells.items():
        assert record["q1"]["delta"][backbone] == pytest.approx(
            original - imagenet, abs=5e-5
        ), f"{backbone}'s recorded Q1 is not original - imagenet"
        assert record["q2"]["delta"][backbone] == pytest.approx(
            masked - original, abs=5e-5
        ), f"{backbone}'s recorded Q2 is not masked - original"

    # Every re-derived verdict must name a backbone that exists, and the two
    # lists must partition the backbones -- a verdict quoted with no SD and
    # not declared pending is one nobody can check.
    for question in ("q1", "q2"):
        entry = record[question]
        assert set(entry["rederived"]) | set(entry["pending_sd"]) == set(LADDER_BACKBONES)
        assert not set(entry["rederived"]) & set(entry["pending_sd"])
        for backbone, verdict in entry["rederived"].items():
            delta = abs(entry["delta"][backbone])
            assert verdict["claimable"] == (delta > verdict["threshold"]), (
                f"{backbone} {question}: the recorded verdict disagrees with "
                "its own delta and threshold"
            )


def test_the_additive_residual_ranking_is_recomputed_not_asserted():
    """**The residual arithmetic is recomputed; what it MEANS is recorded
    separately, because those are different things.**

    Residuals against an additive backbone+init model say where additivity
    breaks. The record keeps the ranking, and this re-derives it. What the
    record no longer does is call the largest one an outlier: with every arm
    well-behaved the same number is a backbone x init interaction, which is a
    finding. That distinction is asserted below as text, so a future edit
    cannot quietly restore the reading without the assertion failing.
    """
    import numpy as np

    record = ladder.STAGE_D_AT_G2
    conflict = ladder.STAGE_D_BACKBONE_CONFLICT

    backbones = list(record["cells"])
    matrix = np.array([record["cells"][b] for b in backbones])
    residual = (
        matrix
        - matrix.mean()
        - (matrix.mean(axis=1) - matrix.mean())[:, None]
        - (matrix.mean(axis=0) - matrix.mean())[None, :]
    )

    ranked = sorted(
        (
            (abs(residual[i, j]), f"{b} {record['inits'][j]}", residual[i, j])
            for i, b in enumerate(backbones)
            for j in range(len(record["inits"]))
        ),
        reverse=True,
    )
    largest = conflict["largest_additive_residual"]
    assert ranked[0][1] == largest["cell"], (
        f"the record names {largest['cell']} but the largest residual is "
        f"{ranked[0][1]}"
    )
    assert ranked[0][2] == pytest.approx(largest["value"], abs=5e-4)
    assert ranked[1][1] == conflict["next_largest"]["cell"]
    assert ranked[1][2] == pytest.approx(
        conflict["next_largest"]["residual"], abs=5e-4
    )

    # The shared-term argument needs no residuals, and it only applies while
    # Swin's two deltas have opposite signs.
    assert largest["cell"].endswith("scut_original")
    assert record["q1"]["delta"]["swin_b"] > 0 > record["q2"]["delta"]["swin_b"], (
        "the argument is that ONE high original cell would inflate Q1 and "
        "deflate Q2; if those signs ever agree, it does not apply"
    )
    assert "INTERACTION" in conflict["residual_measures"]
    assert "does not establish" in conflict["shared_term_argument"]


def test_every_stage_d_threshold_is_recomputed_from_the_two_arms_own_sds():
    """**No threshold in Stage D may be inherited** (PLAN §4.12.1), so each is
    recomputed from the two arms it belongs to.

    This is the check that would have caught the ImageNet-G1 pairing -- a
    five-seed mean quoted with a ten-seed SD -- which survived review and was
    found by re-deriving. Every arm now has its own SD, so there is nothing
    left to excuse: ``pending_sd`` must be empty in both questions.
    """
    from cleft.train.phase3 import combined_claimable_delta

    record = ladder.STAGE_D_AT_G2
    seeds = {b: ladder.SEEDS_BY_KIND[BACKBONES[b]["kind"]] for b in LADDER_BACKBONES}

    for question, (first, second) in (("q1", (0, 1)), ("q2", (1, 2))):
        entry = record[question]
        assert entry["pending_sd"] == [], (
            f"{question} still quotes a verdict with no SD to check it against"
        )
        assert set(entry["rederived"]) == set(LADDER_BACKBONES)

        for backbone, verdict in entry["rederived"].items():
            sd = record["sd"][backbone]
            n = seeds[backbone]
            threshold = combined_claimable_delta(
                sd[record["inits"][first]], n, sd[record["inits"][second]], n
            )
            assert verdict["threshold"] == pytest.approx(
                threshold["arm_means_95"], abs=5e-5
            ), f"{backbone} {question}: the recorded threshold is not its arms'"
            delta = abs(entry["delta"][backbone])
            assert verdict["claimable"] == (delta > threshold["arm_means_95"])
            # Both readings are carried, and both must be the arms' own.
            assert verdict["single_run_95"] == pytest.approx(
                threshold["single_run_95"], abs=5e-5
            ), f"{backbone} {question}: the conservative figure is not its arms'"
            assert verdict["survives_single_run_95"] == (
                delta > threshold["single_run_95"]
            ), f"{backbone} {question}: the conservative reading disagrees"


def test_exactly_one_stage_d_delta_survives_without_averaging():
    """**PLAN §4.3 says to quote ``single_run_95`` when a difference survives
    without averaging, "which is a stronger statement worth making when
    true".** The same sentence obliges saying when it is not.

    Across Stage D that is true exactly once, and it is Swin's Q1 -- the delta
    the ladder is least sure of, in the backbone held contested. Every Q2
    delta clears the claim threshold and none clears this one, so Q2 is a
    claim about ARM MEANS and must not be written as though a single masked
    run beats a single original one.
    """
    record = ladder.STAGE_D_AT_G2
    surviving = {
        (question, backbone)
        for question in ("q1", "q2")
        for backbone, verdict in record[question]["rederived"].items()
        if verdict["survives_single_run_95"]
    }
    assert surviving == {("q1", "swin_b")}, (
        f"expected Swin's Q1 alone to survive without averaging, got {surviving}"
    )
    # And every Q2 delta is claimable, so the contrast is genuinely between
    # the two readings rather than between strong and weak deltas.
    assert all(v["claimable"] for v in record["q2"]["rederived"].values())


def test_stage_d1_deltas_and_thresholds_are_recomputed_from_its_own_arms():
    """Stage D1 gets the same treatment Stage D got: every delta derived from
    the cells, every threshold from the two arms' own SDs.

    Two verdicts here were first read against the wrong bar -- AG-Net's Q2
    against its G2 threshold rather than its G1 one, which is tighter. That is
    the inherited-band error inside a single stage, and re-deriving is what
    catches it.
    """
    from cleft.train.phase3 import combined_claimable_delta

    record = ladder.STAGE_D1_AT_G1
    seeds = {b: ladder.SEEDS_BY_KIND[BACKBONES[b]["kind"]] for b in LADDER_BACKBONES}
    assert set(record["cells"]) == set(LADDER_BACKBONES)
    assert record["inits"] == ladder.INITS

    for backbone, (imagenet, original, masked) in record["cells"].items():
        assert record["q1"]["delta"][backbone] == pytest.approx(
            original - imagenet, abs=5e-5
        )
        assert record["q2"]["delta"][backbone] == pytest.approx(
            masked - original, abs=5e-5
        )

    for question, (first, second) in (("q1", (0, 1)), ("q2", (1, 2))):
        entry = record[question]
        assert entry["pending_sd"] == []
        assert set(entry["rederived"]) == set(LADDER_BACKBONES)
        for backbone, verdict in entry["rederived"].items():
            sd, n = record["sd"][backbone], seeds[backbone]
            threshold = combined_claimable_delta(
                sd[record["inits"][first]], n, sd[record["inits"][second]], n
            )
            assert verdict["threshold"] == pytest.approx(
                threshold["arm_means_95"], abs=5e-5
            ), f"{backbone} {question}: threshold is not from its own G1 arms"
            delta = abs(entry["delta"][backbone])
            assert verdict["claimable"] == (delta > threshold["arm_means_95"])
            assert verdict["single_run_95"] == pytest.approx(
                threshold["single_run_95"], abs=5e-5
            )
            assert verdict["survives_single_run_95"] == (
                delta > threshold["single_run_95"]
            )

    # The headline: Q1 is claimably negative in three backbones and positive
    # in none. A future edit that flips a sign has to face this.
    q1 = record["q1"]
    negative = {
        b for b, v in q1["rederived"].items()
        if v["claimable"] and q1["delta"][b] < 0
    }
    positive = {
        b for b, v in q1["rederived"].items()
        if v["claimable"] and q1["delta"][b] > 0
    }
    assert len(negative) == 3 and not positive, (
        f"Q1 at G1: expected three claimable negatives and no positives, got "
        f"negative={sorted(negative)} positive={sorted(positive)}"
    )


def test_the_swin_g2_cell_is_predicted_by_the_geometry_penalty():
    """**The hypothesis is arithmetic, so it is recomputed.**

    The claim is that ``swin_b imagenet g2`` needs no special mechanism: it is
    a modest G1 value minus a geometry penalty every backbone shows. Both
    halves are checkable -- Swin's drop must be unremarkable among the four,
    and the mean drop subtracted from its G1 value must land near the observed
    cell.
    """
    import numpy as np

    record = ladder.SWIN_G2_IMAGENET_CELL
    g1 = ladder.STAGE_D1_AT_G1["cells"]
    g2 = ladder.STAGE_D_AT_G2["cells"]

    drops = np.array([g1[b][0] - g2[b][0] for b in LADDER_BACKBONES])
    assert (drops > 0).all(), (
        "the account needs every backbone to lose from G1 to G2 at ImageNet; "
        "if one gains, Swin's drop is not the ordinary case"
    )

    swin_drop = g1["swin_b"][0] - g2["swin_b"][0]
    z = {b: (g1[b][0] - g2[b][0] - drops.mean()) / drops.std(ddof=1)
         for b in LADDER_BACKBONES}
    assert abs(z["swin_b"]) < abs(z["vit_b16"]), (
        "the point is that Swin's drop is LESS remarkable than ViT's; if that "
        "reverses, the cell is not the ordinary case this record claims"
    )

    predicted = g1["swin_b"][0] - drops.mean()
    observed = g2["swin_b"][0]
    assert observed == record["value"]
    assert abs(predicted - observed) < ladder.STAGE_D_AT_G2["sd"]["swin_b"][
        "imagenet"
    ], "the residual must sit inside the arm's own SD for the account to hold"

    # And the reason it read as an effect: the same comparison collapses when
    # the reference is an ordinary arm.
    assert ladder.STAGE_D_AT_G2["q1"]["rederived"]["swin_b"]["claimable"]
    assert not ladder.STAGE_D1_AT_G1["q1"]["rederived"]["swin_b"]["claimable"]

    # The contested record must actually be resolved, and by this set.
    conflict = ladder.STAGE_D_BACKBONE_CONFLICT
    assert conflict["status"].startswith("RESOLVED")
    assert conflict["resolved_by"]["set"] == "swin_b__imagenet__g1"
    assert conflict["resolved_by"]["value"] == g1["swin_b"][0]


def test_the_best_arm_is_the_best_cell_of_all_twenty_four():
    """**"Nothing beats it" is a claim over the whole ladder**, so it is
    checked over the whole ladder -- both geometries, all twelve cells each --
    and the thin margin over the nearest challenger is asserted as thin.
    """
    from cleft.train.phase3 import combined_claimable_delta

    best = ladder.BEST_ARM
    seeds = {b: ladder.SEEDS_BY_KIND[BACKBONES[b]["kind"]] for b in LADDER_BACKBONES}
    cells = []
    for record, geometry in (
        (ladder.STAGE_D_AT_G2, "g2"), (ladder.STAGE_D1_AT_G1, "g1"),
    ):
        for backbone, values in record["cells"].items():
            for init, mean in zip(record["inits"], values):
                cells.append((
                    mean, record["sd"][backbone][init], seeds[backbone],
                    f"{backbone} {init} {geometry}",
                ))
    assert len(cells) == 24
    cells.sort(reverse=True)

    top, runner_up = cells[0], cells[1]
    assert top[0] == best["pcc"], (
        f"the record names {best['pcc']} but the best cell is {top[3]} at "
        f"{top[0]}"
    )
    assert "vit_b16 imagenet g1" == top[3]
    assert runner_up[3].replace("scut_", "").startswith(
        best["nearest_challenger"]["arm"].replace("scut_", "").split()[0]
    )

    delta = top[0] - runner_up[0]
    threshold = combined_claimable_delta(top[1], top[2], runner_up[1], runner_up[2])
    assert best["nearest_challenger"]["delta"] == pytest.approx(delta, abs=5e-5)
    assert best["nearest_challenger"]["threshold"] == pytest.approx(
        threshold["arm_means_95"], abs=5e-5
    )
    assert delta > threshold["arm_means_95"], "the headline claim fails"
    # Thin, and recorded as thin -- under 1.2x, and not surviving the
    # conservative reading.
    assert delta / threshold["arm_means_95"] < 1.2
    assert not best["nearest_challenger"]["survives_single_run_95"]
    assert delta < threshold["single_run_95"]


def test_which_arms_span_zero_is_derived_from_the_cells_and_sds():
    """An arm whose mean is inside its own standard error has not been shown
    to have learned anything, and a Q1 measured from it is a difference from a
    baseline that carries no signal. Recomputed, so the list cannot name an
    arm the numbers do not support -- or miss one they do."""
    import numpy as np

    record = ladder.STAGE_D_AT_G2
    seeds = {b: ladder.SEEDS_BY_KIND[BACKBONES[b]["kind"]] for b in LADDER_BACKBONES}

    spanning = set()
    for backbone, cells in record["cells"].items():
        for init, mean in zip(record["inits"], cells):
            sd = record["sd"][backbone][init]
            if abs(mean / (sd / np.sqrt(seeds[backbone]))) < 1.96:
                spanning.add(f"{backbone}__{init}__g2")

    assert spanning == set(record["arms_spanning_zero"]), (
        f"recomputed {sorted(spanning)}, recorded "
        f"{sorted(record['arms_spanning_zero'])}"
    )
    # The reading that matters for Q1: the one claimable Q1 is in a backbone
    # whose ImageNet reference arm is one of these.
    claimable_q1 = {
        b for b, v in record["q1"]["rederived"].items() if v["claimable"]
    }
    assert claimable_q1 == {"swin_b"}
    assert "swin_b__imagenet__g2" in spanning


def test_the_warm_start_prediction_is_recorded_as_refuted_and_recomputed():
    """**A prediction with a mechanism that named the arm testing it, and the
    arm refuted it.**

    Recomputed rather than believed, in both directions: the refutation must
    follow from SR-GNN's own bands, and the weaker surviving claim must hold
    everywhere it says it does. A record that kept the mechanism after the
    test failed would be the more expensive error -- it was quoted once as an
    explanation already.
    """
    record = ladder.STAGE_D_AT_G2
    bands = ladder.STAGE_D_SEED_BANDS
    inits = record["inits"]

    # Refuted: the widest band of the tested backbone is not its imagenet arm.
    tested = bands["refuted"]["tested_on"]
    sd = record["sd"][tested]
    widest = max(inits, key=lambda init: sd[init])
    assert widest != "imagenet", (
        f"{tested}'s widest band IS imagenet, so the prediction is not refuted "
        "and this record is wrong"
    )
    assert widest in bands["refuted"]["outcome"]

    # ...while the backbone it was predicted FROM does show the ordering, or
    # there was never a prediction to refute.
    predicted_from = record["sd"][bands["refuted"]["predicted_from"]]
    assert max(inits, key=lambda init: predicted_from[init]) == "imagenet"

    # Surviving: masked narrowest, strictly where claimed and tied where not.
    survives = bands["survives"]
    assert set(survives["strict_in"]) | set(survives["tied_in"]) == set(LADDER_BACKBONES)
    for backbone in LADDER_BACKBONES:
        sd = record["sd"][backbone]
        others = (sd["imagenet"], sd["scut_original"])
        if backbone in survives["strict_in"]:
            assert sd["scut_masked"] < min(others), backbone
        else:
            assert sd["scut_masked"] == min(others), backbone

    # And no mechanism is attached, because the surviving pattern holds in the
    # transformers, where the refuted one could not apply.
    assert survives["mechanism"] is None
    assert any(
        BACKBONES[b]["kind"] == "transformer"
        for b in survives["strict_in"] + survives["tied_in"]
    )


def test_the_g2_conflict_framing_survives_its_own_resolution():
    """**Q2 at G2 must not read as a consensus with an exception**, and that
    stays true after Stage D1 explained where the conflict came from.

    Three backbones claim masking helps at G2 and Swin claims it hurts, all
    four against their own arms' SDs. An "except Swin" framing reports an
    agreement the data does not contain. **The G2 table is still the G2
    table** -- what D1 added is a second operating point to read it against,
    not a reason to restate the first.

    [AMENDED 2026-08-02] This previously asserted the record was held
    CONTESTED, and failed when ``swin_b__imagenet__g1`` was read and the
    status became RESOLVED -- which is the assertion doing its job rather than
    breaking. The resolution itself is checked in
    ``test_the_swin_g2_cell_is_predicted_by_the_geometry_penalty``; what
    remains here is the part that does not depend on it.
    """
    record = ladder.STAGE_D_AT_G2
    conflict = ladder.STAGE_D_BACKBONE_CONFLICT

    assert "CONFLICT" in record["q2"]["verdict"]
    assert "Not a consensus with an exception" in record["q2"]["verdict"]

    # The conflict is only a conflict if BOTH directions are claimable, so the
    # sign split and the verdicts must agree with each other.
    q2 = record["q2"]
    positive = {b for b, d in q2["delta"].items() if d > 0}
    negative = {b for b, d in q2["delta"].items() if d < 0}
    assert negative == {"swin_b"} and len(positive) == 3
    for backbone in ("swin_b", "agnet"):
        assert q2["rederived"][backbone]["claimable"], (
            "the conflict framing needs claimable deltas on both sides; "
            f"{backbone}'s is not recorded as one"
        )

    # Neither Swin arm may be written off as collapsed. That reading is what
    # shrinkage refuted, and the resolution did NOT reinstate it: the G2 cell
    # is ordinary arithmetic, not a failed run.
    assert record["shrinkage"]["swin_b"]["imagenet"] == max(
        record["shrinkage"]["swin_b"].values()
    ), "the refutation rests on the ImageNet arm's shrinkage being the highest"
    assert "not_defective" in ladder.SWIN_G2_IMAGENET_CELL
    assert conflict["collapse_refuted_by"]["shrinkage"] == record["shrinkage"][
        "swin_b"
    ]["imagenet"]


def test_q1_and_q2_do_not_count_the_checkpoint_as_a_surprise():
    """Changing the init obviously changes the weights -- that IS the factor.
    Flagging it would make the check noisy and train people to ignore it."""
    for check in ladder.comparisons():
        if check["question"] in ("Q1", "Q2"):
            assert check["also_varies"] == []
            assert "checkpoint" in ladder.ENTAILED["init"]


# --------------------------------------------------------------------------
# PLAN §4.3 condition 1 -- the half of the criterion Phase 7 never computed
# --------------------------------------------------------------------------


CLAIM_INTERVAL_FIELDS = {
    "lo", "hi", "ci", "bca", "interval", "per_seed", "excludes_zero", "paired",
    # [ADDED 2026-08-04] Phase 7B's withdrawal records its condition 1 under
    # this key. Omitting it would have counted a resolved claim as unevidenced
    # -- the audit under-reporting its own progress.
    "condition_1", "per_seed_deltas",
}


def _claim_evidence(record: dict) -> list:
    """Which fields carry ACTUAL condition-1 data, not a marker.

    ``condition_1: "UNCOMPUTED -- see ..."`` is honest documentation and not
    evidence; counting it would let the audit report progress it has not made.
    Only a dict under that key is data.
    """
    fields = []
    for name in sorted(set(record) & CLAIM_INTERVAL_FIELDS):
        if name == "condition_1" and not isinstance(record[name], dict):
            continue
        fields.append(name)
    return fields


def _claim_records():
    """Every ``claimable`` verdict in the Phase 7 family, with its evidence."""
    # phase8 is included so the audit does not silently stop covering a new
    # phase -- the shape of the comparisons() gap, from the other direction.
    from cleft import phase7b, phase7c, phase8

    def walk(obj, path):
        if isinstance(obj, dict):
            if "claimable" in obj:
                yield path, obj, _claim_evidence(obj)
            for key, value in obj.items():
                yield from walk(value, f"{path}.{key}")
        elif isinstance(obj, (list, tuple)):
            for index, value in enumerate(obj):
                yield from walk(value, f"{path}[{index}]")

    out = []
    for module in (ladder, phase7b, phase7c, phase8):
        short = module.__name__.rsplit(".", 1)[-1]
        for name in dir(module):
            if not name.isupper():
                continue
            value = getattr(module, name)
            if isinstance(value, (dict, list, tuple)):
                out.extend(walk(value, f"{short}.{name}"))
    return out


def test_the_claim_audit_has_not_drifted():
    """**[MEASURED 2026-08-04] 43 claimable verdicts, 0 carrying an interval.**

    PLAN §4.3 claims a delta only if BOTH conditions hold. Phase 7 reported
    condition 2 as though it were the criterion, and when Phase 7C finally
    computed condition 1 it withdrew all nine of that phase's verdicts.

    **One-directional and deliberately brittle.** If the counts move, someone
    has added or resolved a claim and must re-read
    ``ladder.CLAIMS_REST_ON_HALF_THE_CRITERION`` rather than update a number.
    A new claim arriving without an interval is exactly what this exists to
    surface.
    """
    records = _claim_records()
    with_interval = sorted(path for path, _, evidence in records if evidence)
    audit = ladder.CLAIMS_REST_ON_HALF_THE_CRITERION

    # [UPDATED 2026-08-15] 47 -> 48: PHASE_7D_CLOSING joined the audit as
    # the first record BORN with condition 1 attached. The guard fired, the
    # audit record was re-read and updated, and the count follows it.
    assert len(records) == audit["records_with_a_claimable_field"] == 48, (
        f"the claim count moved to {len(records)}; re-read the audit record "
        "rather than editing the number"
    )
    # **[UPDATED 2026-08-04] One now carries condition 1: Phase 7B's, which
    # withdrew "claimably worse".** The count moved because that resolution
    # added a record, and this guard is what forced it to be re-read rather
    # than the number quietly edited.
    assert with_interval == audit["carrying_condition_1"] == [
        "ladder.BEST_ARM.nearest_challenger",
        "ladder.BEST_ARM.nearest_challenger.condition_1",
        "ladder.PHASE_7D_CLOSING",
        "ladder.STAGE_D_AT_G2.q1.rederived.swin_b",
        "phase7b.SEARCH_AXIS_VERDICTS.outcome.against_baseline",
    ], (
        f"condition-1 evidence now appears at {with_interval}. That is "
        "progress -- update the audit record and say which verdicts survived"
    )


def test_the_condition_2_margin_is_not_treated_as_a_safety_indicator():
    """**7C failed condition 1 at 3.62x, so no margin in this project is
    reassurance.** Nothing claimed exceeds 5.51x."""
    from cleft import phase7c

    margins = [
        abs(record["delta"]) / record["threshold"]
        for _, record, _ in _claim_records()
        if record.get("claimable") is True
        and isinstance(record.get("delta"), (int, float))
        and isinstance(record.get("threshold"), (int, float))
        and record["threshold"]
    ]
    assert margins and max(margins) < 6.0

    # The calibration point: an arm that cleared condition 2 comfortably and
    # failed condition 1 anyway.
    arm_2 = phase7c.STAGE_7C_RESULTS["rounds"]["selected30"]["against_identity"][
        "2_geometric"
    ]
    assert abs(arm_2["delta"]) / arm_2["threshold"] > 3.5
    assert phase7c.PAIRED_BCA_WITHDREW_EVERY_VERDICT[
        "intervals_excluding_zero"
    ] == "0 of 45"
    assert "does not predict" in (
        ladder.CLAIMS_REST_ON_HALF_THE_CRITERION["margin_does_not_predict_condition_1"]
        or ""
    ) or "no information" in (
        ladder.CLAIMS_REST_ON_HALF_THE_CRITERION["margin_does_not_predict_condition_1"]
    )


def test_the_headline_comparison_is_marked_uncomputed():
    """The thinnest claim in the project, at 1.05x, is the one a write-up
    leans on hardest."""
    challenger = ladder.BEST_ARM["nearest_challenger"]
    assert challenger["margin"] == "1.05x"
    # [UPDATED 2026-08-04] Computed, and withdrawn: 0 of 5 exclude zero.
    assert challenger["condition_1"]["claimable"] is False
    assert ladder.CLAIMS_REST_ON_HALF_THE_CRITERION["thinnest"]["record"] == (
        "ladder.BEST_ARM.nearest_challenger"
    )


# --------------------------------------------------------------------------
# the generalised paired-claim task
# --------------------------------------------------------------------------


def test_run_stem_follows_reuses_not_the_arm_name():
    """``p7_g1_vit_b16_imagenet_g1`` IS ``p7_d1_vit_b16_imagenet_g1``'s fits.

    Pairing on the arm name would look for a run directory that was never
    created, and the declare script would report it MISSING rather than the
    task explaining why.
    """
    by_name = {arm["name"]: arm for arm in ladder.arms()}
    reusing = [a for a in ladder.arms() if a.get("reuses")]
    assert reusing, "no arm reuses another; this test has stopped covering it"
    for arm in reusing:
        assert ladder.run_stem(arm) == arm["reuses"]
        assert arm["reuses"] in by_name, arm["name"]
    for arm in ladder.arms():
        if not arm.get("reuses"):
            assert ladder.run_stem(arm) == arm["name"]


def test_the_label_question_is_excluded_because_it_has_no_common_truth():
    """``label`` selects a manifest COLUMN, so median and mean arms are scored
    against different truth vectors. Their delta compares two task
    difficulties, not two models on one task -- condition 1 is UNDEFINED for
    them, not uncomputed."""
    keys = {pair["question"] for pair in ladder.paired_claim_pairs("ladder")}
    assert not any(key.startswith("label") for key in keys), sorted(keys)
    assert any(
        c["varies"] == "label" for c in ladder.comparisons()
    ), "the ladder no longer asks the label question; the exclusion is stale"
    assert "different truth vectors" in (
        ladder.PAIRED_CLAIM_COVERAGE["excluded_label_formulation"]
    )


def test_a_g0_arm_that_reuses_a_run_is_not_excluded():
    """**[CORRECTED] The stage label is not the test.**

    ``p7_g0_vit_b16_imagenet_g2`` reuses ``p7_d_vit_b16_imagenet_g2``, which
    ran. Filtering on "G0 appears in the pair" dropped a Q1 comparison that is
    perfectly computable -- caught because Q1 came back with three entries
    instead of four.
    """
    questions = [p["question"] for p in ladder.paired_claim_pairs("ladder")]
    assert questions.count("Q1") == 4, (
        "one of the four Q1 comparisons has been dropped again"
    )
    assert questions.count("Q2") == 4
    stems = ladder.paired_claim_stems("ladder")
    assert "p7_d_vit_b16_imagenet_g2" in stems
    # And nothing unrun-and-unreused sneaks in.
    unrun = {
        arm["name"] for arm in ladder.arms()
        if arm["stage"] in ladder.UNRUN_STAGES and not arm.get("reuses")
    }
    assert unrun and not (unrun & set(stems)), sorted(unrun & set(stems))


def test_the_d1_at_g1_claims_are_covered_although_comparisons_omits_them():
    """``STAGE_D1_AT_G1`` carries eight rederived verdicts and
    ``comparisons()`` has no entry for any of them. Same shape as the discovery
    glob that reported "nothing to paste" over twelve placeholders."""
    pairs = ladder.paired_claim_pairs("ladder")
    at_g1 = [p for p in pairs if p["question"].endswith("_at_g1")]
    assert len(at_g1) == 8, [p["key"] for p in at_g1]
    assert not any(
        c["question"].endswith("_at_g1") for c in ladder.comparisons()
    ), "comparisons() now covers these; _g1_init_pairs would duplicate them"
    recorded = set(ladder.STAGE_D1_AT_G1["q1"]["rederived"])
    assert recorded == {"vit_b16", "swin_b", "srgnn", "agnet"}
    covered = {p["key"].split("__")[1] for p in at_g1}
    assert covered == recorded


def test_seed_bands_are_derived_per_stem_not_assumed():
    """Transformer arms ran at five seeds, graph arms at ten. Declaring a
    ten-seed arm with five files would compare it on half its band."""
    groups = ladder.paired_claim_seed_groups("ladder")
    assert len(groups) == 2, sorted(groups)
    assert {len(seeds) for seeds in groups} == {5, 10}

    vectors = ladder.paired_claim_vectors("ladder")
    per_stem = {}
    for stem, seed in vectors:
        per_stem.setdefault(stem, set()).add(seed)
    assert {len(v) for v in per_stem.values()} == {5, 10}
    assert len(vectors) == len(set(vectors)), "a (stem, seed) is duplicated"


def test_the_headline_scope_is_the_thinnest_claim_and_two_directories():
    pairs = ladder.paired_claim_pairs("headline")
    assert len(pairs) == 1
    assert pairs[0]["recorded"]["margin"] == "1.05x"
    assert pairs[0]["b"] == ladder.BEST_ARM["arm"]
    assert len(ladder.paired_claim_vectors("headline")) == 10
    # The headline pair must also appear in the full scope, or answering one
    # would not answer the other.
    assert pairs[0]["key"] in {p["key"] for p in ladder.paired_claim_pairs("ladder")}


def test_an_unknown_scope_is_refused():
    with pytest.raises(ladder.LadderError, match="unknown paired-claim scope"):
        ladder.paired_claim_pairs("everything")


def test_the_shipped_paired_claim_configs_match_the_derived_scopes(repo_root):
    import os
    import subprocess

    result = subprocess.run(
        [sys.executable,
         str(repo_root / "scripts" / "generate_paired_claim_configs.py"),
         "--check"],
        capture_output=True, text=True, cwd=str(repo_root),
        env={**os.environ, "PYTHONPATH": str(repo_root / "src")},
    )
    assert result.returncode == 0, f"{result.stdout}{result.stderr}"


def test_a_stem_with_siblings_is_pinned_by_job_id(repo_root):
    """**[MEASURED 2026-08-04] One stem of thirty has siblings.**

    ``p7_e_srgnn_scut_masked_g2_random`` matches three runs, one of them a
    4-seed partial. The declare script refused all ten of its vectors rather
    than choosing; the job id is now pinned to the 10-seed relaunch.
    """
    import yaml

    payload = yaml.safe_load(
        (repo_root / "configs" / "p7_paired_ladder.yaml").read_text(
            encoding="utf-8"
        )
    )
    audit = ladder.SIBLING_RUNS_AUDIT
    assert audit["hashed"] + audit["ambiguous"] == len(payload["inputs"]) == 240
    assert audit["missing"] == 0, (
        "a MISSING vector means a stem resolves only to a partial run, which "
        "the ambiguity guard does not catch"
    )

    # **[UPDATED] The config resolves incrementally**, so this must hold in
    # both states: an unresolved entry carries the pinned job id (or `*` where
    # no sibling is known), and a resolved one carries a real job id that
    # still matches the pin. Asserting only the glob shape would stop covering
    # the file the moment the cluster paste landed.
    for entry in payload["inputs"]:
        run_dir = entry["path"].split("/")[-2]
        stem, sha, job = run_dir.split("__")
        pinned = ladder.LADDER_JOB_IDS.get(stem)
        if "*" in entry["path"]:
            assert sha == "*", entry["name"]
            assert job == (pinned or "*"), entry["name"]
        else:
            assert len(sha) == 8 and all(c in "0123456789abcdef" for c in sha)
            if pinned:
                assert job == pinned, (
                    f"{entry['name']} resolved to {job!r}, not the pinned "
                    f"{pinned!r} -- a sibling run was declared"
                )


def test_a_short_seed_band_is_refused_rather_than_computed():
    """**Condition 1 is a universal over the seeds PRESENT, so fewer
    intervals is an EASIER bar.**

    The 4-seed partial is a run whose numbers are right and whose band is
    short -- unlike every earlier ambiguity, where a verification against the
    recorded numbers would have caught it. Four intervals instead of ten
    returns entirely ordinary output.
    """
    assert "easier bar" in (
        ladder.SIBLING_RUNS_AUDIT["the_partial_is_a_new_shape"].lower()
    )
    source = (
        Path(ladder.__file__).parent / "run.py"
    ).read_text(encoding="utf-8")
    assert "loaded {observed} seeds where the arm list says" in source, (
        "task_paired_claims no longer refuses a short band"
    )


def test_the_unplanned_determinism_check_is_recorded():
    """Two complete runs at different SHAs agreeing to four decimals on mean
    AND sd is the opposite of the void ladder's 0.068 drift, and it was
    obtained for free."""
    note = ladder.SIBLING_RUNS_AUDIT["unplanned_determinism_check"]
    assert "different SHAs" in note and "0.068" in note


def test_same_direction_is_not_vacuously_true_on_an_empty_set():
    """**[CORRECTED 2026-08-04] R7 instance 14.**

    ``same_direction`` was ``len(directions) <= 1`` over the EXCLUDING subset,
    so with nothing excluding zero it reported True on an empty set. The
    ladder headline came back ``n_excluding_zero: 0, same_direction: true`` --
    reading as five seeds agreeing when the deltas ran -0.0027 to +0.1040 and
    the sign flipped.
    """
    import numpy as np

    from cleft import phase7b

    rng = np.random.default_rng(4)
    truth = rng.normal(size=60)
    good = truth + rng.normal(scale=0.5, size=60)
    seeds = [1, 2]
    result = phase7b.paired_comparison(
        truth=truth,
        # Near-identical vectors: no interval will exclude zero.
        winner_by_seed={s: good + rng.normal(scale=0.01, size=60) for s in seeds},
        baseline_by_seed={s: good for s in seeds},
        winner_sd=1e-4, n_boot=200,
    )
    assert result["n_excluding_zero"] == 0
    assert result["same_direction"] is None, (
        "no seed excluded zero, so 'the excluding seeds agreed' has no "
        "answer -- None, not True"
    )
    assert isinstance(result["same_direction_all_seeds"], bool)
    assert result["claimable"] is False


def test_the_verdict_was_never_wrong_only_the_reported_field():
    """``all_exclude`` requires ``len(directions) == 1``, which an empty set
    fails -- so no claim was ever made on the vacuous True."""
    # Comments quote the old expression verbatim, so compare CODE only.
    code = "\n".join(
        line for line in
        (Path(ladder.__file__).parent / "phase7b.py").read_text("utf-8").splitlines()
        if not line.lstrip().startswith("#")
    )
    assert "len(directions) == 1" in code
    assert "len(directions) <= 1" not in code


def test_the_headline_is_withdrawn_and_the_number_is_kept():
    challenger = ladder.BEST_ARM["nearest_challenger"]
    assert challenger["condition_1"]["n_excluding_zero"] == 0
    assert challenger["condition_1"]["sign_flips"] is True
    assert challenger["claimable_condition_2_only"] is True

    assert "HIGHEST-SCORING" in ladder.BEST_ARM["verdict"]
    assert "NOT distinguishable" in ladder.BEST_ARM["verdict"]
    assert "strongest model" in ladder.BEST_ARM["verdict_before_condition_1"]
    # The PCC itself is untouched by any of this.
    assert ladder.BEST_ARM["pcc"] == 0.2520


def test_the_cohort_finding_is_recorded_as_a_finding_not_a_caveat():
    """Four for four, across three unrelated interventions. Recorded in its
    own right rather than as a limitation attached to each withdrawal."""
    record = ladder.COHORT_CANNOT_RESOLVE
    assert len(record["tested"]) == 4
    margins = [entry["margin"] for entry in record["tested"]]
    assert min(margins) == 1.05 and max(margins) == 3.62
    # Every entry failed: none reports all seeds excluding zero.
    for entry in record["tested"]:
        count, _, total = entry["excluding"].split()
        assert int(count) < int(total), entry
    assert "scored higher" in record["forbids"]


def test_the_survivor_carries_its_caveats_beside_the_claim():
    """**The one surviving claim in Phase 7, and neither caveat is optional.**

    Its +0.2027 is large because the baseline is 0.0065, and the same
    comparison at the other geometry gives +0.0114 at 0.44x with 0 of 5. Both
    live in the claim's own record rather than one entry away, because a
    reader who quotes the delta will read that dict and not necessarily
    anything else.
    """
    claim = ladder.STAGE_D_AT_G2["q1"]["rederived"]["swin_b"]
    assert claim["condition_1"]["n_excluding_zero"] == 5
    assert claim["condition_1"]["claimable"] is True
    assert claim["rests_on"] == "SWIN_G2_IMAGENET_CELL, a baseline of 0.0065"

    # The cell it rests on, and the recovery that makes it G2-specific.
    cell = ladder.SWIN_G2_IMAGENET_CELL
    assert cell["value"] == 0.0065
    assert cell["same_weights_at_g1"]["value"] == 0.1076

    # The same comparison at G1 fails, and the record says so with numbers.
    other = claim["does_not_reproduce_at_g1"]
    assert other["n_excluding_zero"] == 0 and other["margin"] < 1.0
    assert other["delta"] == ladder.STAGE_D1_AT_G1["q1"]["delta"]["swin_b"]


def test_the_margin_does_not_predict_condition_1_at_scale():
    """Measured across 26 pairs: the two highest margins both fail and the
    survivor ranks third. Ordering claims by margin orders them by something
    that does not govern whether they hold."""
    record = ladder.LADDER_PAIRED_AUDIT["margin_does_not_predict_condition_1"]
    assert record["measured_across"] == 26
    survivor_margin = ladder.LADDER_PAIRED_AUDIT["survivor"]["margin"]
    for entry in record["two_highest_margins_both_fail"]:
        count, _, total = entry["excluding"].split()
        assert int(count) < int(total), entry
        assert entry["margin"] > survivor_margin, (
            "a failing margin no longer exceeds the survivor's, so this "
            "record's point has changed"
        )
    assert record["survivor_rank_by_margin"] == 3


def test_the_audit_totals_agree_across_the_records():
    """One arithmetic, three places -- so a later correction cannot land in
    only one of them."""
    audit = ladder.LADDER_PAIRED_AUDIT
    assert audit["survived"] + audit["withdrawn"] == 26
    at_scale = ladder.COHORT_CANNOT_RESOLVE["at_scale"]
    assert at_scale["survived"] + at_scale["withdrawn"] == at_scale["tested"] == 30
    # 26 ladder pairs + 4 already tested = 30, of which the ladder's one is
    # the only survivor anywhere.
    assert at_scale["survived"] == audit["survived"] == 1
    assert "29" in ladder.CLAIMS_REST_ON_HALF_THE_CRITERION["resolved_so_far"] or (
        "Twenty-nine" in ladder.CLAIMS_REST_ON_HALF_THE_CRITERION["resolved_so_far"]
    )


def test_the_tradeoff_scope_is_the_phase_8_claim(repo_root):
    """**Phase 8's §0 is a comparative claim of exactly the size §2 of the
    same brief warns is unresolvable here.** It is tested before the phase is
    designed around it, because that is what the audit was for.
    """
    pairs = ladder.paired_claim_pairs("tradeoff")
    assert len(pairs) == 1
    pair = pairs[0]
    assert pair["b"] == ladder.BEST_ARM["arm"]
    assert pair["a"] == ladder.TRADE_OFF_PAIR["interpretable_arm"]["stem"]

    # It is NOT one of the audited 26 -- three factors differ at once.
    assert pair["key"] not in {p["key"] for p in ladder.paired_claim_pairs("ladder")}


def test_the_tradeoff_arm_is_srgnns_best_cell_and_at_g1():
    """The brief named the masked-G2 cell at 0.1507 while its framing sentence
    quoted 0.1719. This uses the cell the sentence cites -- same native
    scheme, and at G1 so the region comparison needs no cross-geometry map."""
    stem = ladder.TRADE_OFF_PAIR["interpretable_arm"]["stem"]
    arm = next(a for a in ladder.arms() if a["name"] == stem)
    assert arm["geometry"] == "g1"
    assert arm["init"] == "imagenet"
    assert arm["region_scheme"] in (None, "native"), arm["region_scheme"]

    best_g1 = max(ladder.STAGE_D1_AT_G1["cells"]["srgnn"])
    assert ladder.TRADE_OFF_PAIR["interpretable_arm"]["recorded_pcc"] == best_g1
    # And it beats the brief's arm B, the masked-G2 cell.
    assert best_g1 > ladder.STAGE_D_AT_G2["cells"]["srgnn"][2]

    # Both arms at the same geometry as the best arm.
    best = next(a for a in ladder.arms() if a["name"] == ladder.BEST_ARM["arm"])
    assert best["geometry"] == arm["geometry"] == "g1"


def test_the_tradeoff_pairs_only_the_seeds_the_two_arms_share():
    """SR-GNN ran at ten seeds and ViT at five. Pairing needs the same
    held-out patients on both sides, so it uses the five they share -- and the
    run recomputes each arm's own band over those five rather than quoting
    SR-GNN's ten-seed number."""
    by_name = {a["name"]: a for a in ladder.arms()}
    pair = ladder.paired_claim_pairs("tradeoff")[0]
    a_seeds = set(by_name[pair["a"]]["seed_list"])
    b_seeds = set(by_name[pair["b"]]["seed_list"])
    assert len(a_seeds) == 10 and len(b_seeds) == 5
    assert set(pair["seeds"]) == a_seeds & b_seeds == b_seeds
    # The record must say plainly that SR-GNN's quoted number comes from a
    # different seed count than the comparison uses, or the 0.1719 will be
    # read as the paired arm's mean.
    assert ladder.TRADE_OFF_PAIR["seeds_used"] == 5
    assert ladder.TRADE_OFF_PAIR["interpretable_arm"]["recorded_seeds"] == 10
    assert ladder.TRADE_OFF_PAIR["best_arm"]["recorded_seeds"] == 5
    assert "five seeds the two arms share" in (
        ladder.paired_claim_pairs("tradeoff")[0]["recorded"]["note"]
    )


def test_the_tradeoff_config_needed_no_cluster_round_trip(repo_root):
    """Both stems were already declared elsewhere, and a path's contents are
    immutable -- so one hash serves every config declaring it. Reusing them
    satisfies the cross-config invariant by construction rather than by
    remembering to update the sibling."""
    import yaml

    payload = yaml.safe_load(
        (repo_root / "configs" / "p7_paired_tradeoff.yaml").read_text("utf-8")
    )
    assert len(payload["inputs"]) == 10
    for entry in payload["inputs"]:
        assert entry["rollup_sha256"] != "0" * 64, entry["name"]
        assert "*" not in entry["path"], entry["name"]


def test_the_defensive_wording_is_recorded_before_the_result():
    """If A-vs-B is unresolvable the contribution becomes a description, and
    that wording is licensed either way -- so it is the default rather than
    the fallback."""
    assert "description" in ladder.TRADE_OFF_PAIR["if_unresolvable"]
    assert "best-SCORING" in ladder.TRADE_OFF_PAIR["if_unresolvable"]


def test_phase_7d_registration_predates_the_build(repo_root):
    """**[REWRITTEN 2026-08-15] The no-artifacts form of this test did its
    job and retired**: it held the ground until the build, and the build now
    exists. What the priority claim needs from here on is that the
    REGISTRATION's own date never moves and the build record says it came
    second -- a re-dated registration would be a post-hoc registration
    wearing a pre-registration's timestamp."""
    record = ladder.PHASE_7D_PATCH_AXIS_REGISTERED
    built = ladder.PHASE_7D_BUILT
    assert record["registered"] == (
        "2026-08-14, before any 7D snapshot, extraction or arm"
    ), "the registration's date moved; that is the one edit it must never take"
    assert built["built"] == "2026-08-15"
    assert built["registration_predates_build"] is True
    assert "2026-08-14" in built["existed_before"]
    # And the build is real: twelve configs -- arm 5's two joined on
    # 2026-08-15, AFTER the maintainer's staging decision, never before it.
    names = {p.name for p in (repo_root / "configs").glob("p7d_*.yaml")}
    assert len(names) == 13, sorted(names)
    assert "p7d_extract_vit_b32_512.yaml" in names
    assert "p7d_arm_vit_b32_512.yaml" in names
    assert "p7d_paired.yaml" in names


def test_phase_7d_is_not_road_b_and_the_control_is_the_existing_arm():
    record = ladder.PHASE_7D_PATCH_AXIS_REGISTERED
    assert "PATCH SIZE" in record["not_road_b"]
    assert "0.2319 -> 0.0857" in record["not_road_b"]
    control = record["arms"]["vit_b16_224"]
    assert "0.2520" in control["control"]
    assert "p7_d1_vit_b16_imagenet_g1" in control["control"]
    # The control's figures agree with the ladder's own record of the arm.
    assert ladder.BEST_ARM["pcc"] == 0.2520
    # And the concat arm carries 7B's measured prior, in 7B's own numbers.
    from cleft import phase7b

    prior = record["arms"]["concat_8_16_32"]["prior_against"]
    assert "+0.0221" in prior and "-0.0542" in prior
    outcome = phase7b.SEARCH_AXIS_VERDICTS["outcome"]
    assert outcome["inner_val_lead"] == 0.0221
    assert outcome["against_baseline"]["delta"] == -0.0542


def test_the_fifth_arms_arithmetic_is_the_measured_one():
    """**The amendment's rationale contradicted its own arms table**, and
    the registration records the measured grids: both 512 configurations
    interpolate at ratio 2.286, so 'much milder' is dead and the arm
    discriminates mechanisms instead."""
    record = ladder.PHASE_7D_PATCH_AXIS_REGISTERED
    corrected = record["amendment_arithmetic_corrected"]
    assert "pretrained grid is 7x7" in corrected["measured"]
    assert "IDENTICAL" in corrected["measured"]
    assert "2.286" in corrected["measured"]
    assert "false as stated" in corrected["consequence"]

    # The grids in the arms table are internally consistent: tokens are the
    # square of the grid side, and 224/patch reproduces each side.
    for name, arm in record["arms"].items():
        if "tokens" not in arm:
            continue
        side = int(arm["grid"].split("x")[0])
        assert arm["tokens"] == side * side, name
        patch = int(name.split("_b")[1].split("_")[0])
        assert side == 224 // patch, name

    readings = record["fifth_arm_readings_committed"]
    assert set(readings) == {
        "cliffs_like_patch16_at_512", "does_not_cliff", "either_way",
    }
    assert "ratio form" in readings["cliffs_like_patch16_at_512"]
    assert "stated revision" in readings["does_not_cliff"]
    assert "Neither outcome licenses tuning" in readings["either_way"]
    assert "post-hoc" in record["why_registered_now"]


def test_phase_7d_build_record_and_the_substitution(repo_root):
    """**Arm 6 runs MViTv2, not the paper's MViTv1, and the record names
    both the reason and the architectural deltas** -- measured against the
    pinned timm, where v1 does not exist."""
    built = ladder.PHASE_7D_BUILT
    sub = built["arm_6"]["substitution"]
    assert sub["paper"] == "MViTv1"
    assert "mvitv2_base" in sub["ran"] and "fb_in1k" in sub["ran"]
    assert "does not exist in the pinned timm" in sub["why"]
    assert "decomposed relative position embeddings" in sub["v2_adds"]
    assert "residual pooling" in sub["v2_adds"]
    # The head config is deliberate: dim and pooling are measured facts.
    assert built["arm_6"]["embedding_dim"] == 768
    assert "NO cls token" in built["arm_6"]["pooling"]
    # Capacity class: nearest in1k variant, stated against the comparators.
    assert built["arm_6"]["capacity"]["parameters"] == 50_703_744
    # Both committed readings exist and neither is "MViT should win".
    framing = built["arm_6"]["framing"]
    assert framing["reading_if_near_swin"] == "class property"
    assert "mechanisms differ" in framing["reading_if_different"]
    assert "different question" in framing["neither_is_mvit_should_win"]
    # The registry entry matches the record.
    assert BACKBONES["mvitv2_b"]["timm_name"] == "mvitv2_base"
    assert BACKBONES["mvitv2_b"]["embedding_dim"] == 768
    # And the attribution is in every generated header.
    for path in sorted((repo_root / "configs").glob("p7d_*.yaml")):
        text = path.read_text(encoding="utf-8")
        assert "ATTRIBUTION" in text, path.name
        assert "Fan et al." in text, path.name


def test_phase_7d_arm_5_decision_and_its_configs(repo_root):
    """[DECIDED 2026-08-15] The hold's record is kept as the state it was;
    the decision sits beside it -- SQUARE, the comparator's own artifact --
    and the two configs exist, reference it, carry its VERIFIED hash from
    the comparator's shipped config, and share the 224 snapshot."""
    import yaml as _yaml

    held = ladder.PHASE_7D_BUILT["arm_5_held"]
    assert "does not specify square or nonsquare" in held["why"]
    assert "roadb_512_square_g1_v1" in held["decision_relevant_fact"]
    decided = held["decided"]
    assert decided["staging"].startswith("SQUARE")
    assert "one-factor match" in decided["why"]

    configs = repo_root / "configs"
    comparator = _yaml.safe_load(
        (configs / "roadb_p7_arm_vit_b16_imagenet_512.yaml").read_text(
            encoding="utf-8"
        )
    )
    staged = next(
        e for e in comparator["inputs"] if e["name"] == "roadb_512_square_g1_v1"
    )
    for name in ("p7d_extract_vit_b32_512.yaml", "p7d_arm_vit_b32_512.yaml"):
        text = (configs / name).read_text(encoding="utf-8")
        # The committed reading, verbatim commitments in the header.
        assert "2.286x" in text, name
        assert "CANNOT 'beat the\n# cliff by milder interpolation'" in text, name
        assert "DISCRIMINATE" in text, name
        payload = _yaml.safe_load(text)
        entry = next(
            e for e in payload["inputs"]
            if e["name"] == "roadb_512_square_g1_v1"
        )
        # The staged input IS the comparator's, hash and path -- one factor
        # by construction, not by intention.
        assert entry["path"] == staged["path"], name
        assert entry["rollup_sha256"] == staged["rollup_sha256"], name
        assert payload["task"]["staged_artifact"] == "roadb_512_square_g1_v1"
    # One snapshot serves both extractions: the 512 extraction declares the
    # SAME init artifact as the 224 one.
    extract_224 = _yaml.safe_load(
        (configs / "p7d_extract_vit_b32.yaml").read_text(encoding="utf-8")
    )
    extract_512 = _yaml.safe_load(
        (configs / "p7d_extract_vit_b32_512.yaml").read_text(encoding="utf-8")
    )
    init_224 = next(
        e for e in extract_224["inputs"] if e["name"] == "pretrained_init"
    )
    init_512 = next(
        e for e in extract_512["inputs"] if e["name"] == "pretrained_init"
    )
    assert init_224["path"] == init_512["path"]
    # And the two vit_b32 extractions write DIFFERENT versions -- the set
    # names inside are identical, so the version is what tells them apart.
    assert extract_224["task"]["out_version"] != extract_512["task"]["out_version"]
    assert "512" in extract_512["task"]["out_version"]


def test_phase_7d_configs_carry_their_readings_and_declared_hashes(repo_root):
    """The pre-registered readings ride in the headers verbatim, and every
    hash is the declare pass's own figure.

    [UPDATED 2026-08-15, twice] This test pinned the placeholder state at
    build time; the extraction declares filled the init hashes, and the arm
    declares filled the embeddings hashes. What it pins NOW is the identity
    rule the maintainer set for the fill: the concat arm's constituents must
    be BYTE-IDENTICAL to the singles' hashes at the same artifact paths --
    same artifacts, so any divergence is an error to stop on, never to fill
    through."""
    import yaml as _yaml

    def _hex_filled(rollup: str) -> bool:
        return (
            len(rollup) == 64
            and set(rollup) <= set("0123456789abcdef")
            and set(rollup) != {"0"}
        )

    configs = repo_root / "configs"
    concat = (configs / "p7d_arm_concat_multiscale.yaml").read_text(
        encoding="utf-8"
    )
    assert "+0.0221" in concat and "-0.0542" in concat
    assert "prior AGAINST this arm" in concat
    payload = _yaml.safe_load(concat)
    order = [
        entry["backbone"]
        for entry in payload["task"]["concat_embeddings_artifacts"]
    ]
    assert order == ["vit_b8", "vit_b16", "vit_b32"], "8+16+32, as named"
    by_name = {e["name"]: e for e in payload["inputs"]}
    assert by_name["emb_vit_b16"]["path"].endswith("vit_b16__imagenet__g1")

    # Every single arm is filled, and the concat's b8/b32 constituents are
    # the SAME hash at the SAME path as the singles'.
    singles = {}
    for backbone in ("vit_b32", "vit_b8", "mvitv2_b"):
        text = (configs / f"p7d_arm_{backbone}.yaml").read_text(encoding="utf-8")
        arm = _yaml.safe_load(text)
        entry = next(e for e in arm["inputs"] if e["name"] == "embeddings")
        assert _hex_filled(entry["rollup_sha256"]), backbone
        singles[backbone] = entry
        assert "declare pass's own figure" in text, backbone
    for backbone, input_name in (
        ("vit_b8", "emb_vit_b8"), ("vit_b32", "emb_vit_b32"),
    ):
        assert by_name[input_name]["path"] == singles[backbone]["path"]
        assert by_name[input_name]["rollup_sha256"] == (
            singles[backbone]["rollup_sha256"]
        ), (
            f"the concat arm's {input_name} hash diverged from the "
            f"{backbone} single's for the same artifact -- an error to stop "
            "on, not to fill through"
        )
    # The two vit_b32 sets (224 and 512) are different artifacts and must
    # NOT share a hash -- the version directory is what tells them apart.
    arm_512 = _yaml.safe_load(
        (configs / "p7d_arm_vit_b32_512.yaml").read_text(encoding="utf-8")
    )
    emb_512 = next(e for e in arm_512["inputs"] if e["name"] == "embeddings")
    assert _hex_filled(emb_512["rollup_sha256"])
    assert emb_512["rollup_sha256"] != singles["vit_b32"]["rollup_sha256"]

    for backbone in ("vit_b32", "vit_b8"):
        text = (configs / f"p7d_arm_{backbone}.yaml").read_text(encoding="utf-8")
        assert "cliff mechanism is absent by construction" in text, backbone
    mvit = (configs / "p7d_arm_mvitv2_b.yaml").read_text(encoding="utf-8")
    assert "Swin-specific or a property of the" in mvit
    assert "NO cls token" in mvit
    # The b8 extraction carries the measured memory decision.
    b8x = (configs / "p7d_extract_vit_b8.yaml").read_text(encoding="utf-8")
    assert "29.6 MB" in b8x and "batch_size: 8" in b8x


def test_the_concat_arm_is_load_time_guarded():
    """A concat arm without constituents, or constituents on another arm,
    would verify inputs that fed nothing."""
    from cleft.config.schema import ConfigError, _check_train_cv_init

    good = {
        "backbone": "concat", "init": "imagenet",
        "concat_embeddings_artifacts": [
            {"artifact": "a", "backbone": "vit_b8"},
            {"artifact": "b", "backbone": "vit_b16"},
        ],
    }
    _check_train_cv_init(good)  # does not raise

    with pytest.raises(ConfigError, match="fewer than two"):
        _check_train_cv_init({
            "backbone": "concat", "init": "imagenet",
            "concat_embeddings_artifacts": [
                {"artifact": "a", "backbone": "vit_b8"}
            ],
        })
    with pytest.raises(ConfigError, match="ambiguous which"):
        _check_train_cv_init({**good, "embeddings_artifact": "x"})
    with pytest.raises(ConfigError, match="imagenet-only"):
        _check_train_cv_init({**good, "init": "scut_masked"})
    with pytest.raises(ConfigError, match="Only the 'concat' arm"):
        _check_train_cv_init({
            "backbone": "vit_b16", "init": "imagenet",
            "concat_embeddings_artifacts": good["concat_embeddings_artifacts"],
        })
    # And the new single-backbone arms are artifact-only, like swin_b.
    for backbone in ("vit_b32", "vit_b8", "mvitv2_b"):
        with pytest.raises(ConfigError, match="Live extraction builds ViT-B/16"):
            _check_train_cv_init({"backbone": backbone, "init": "imagenet"})


def test_phase_7d_results_apply_the_registered_readings():
    """**Each reading was committed before these numbers existed** -- the
    record APPLIES them; a reading composed after the fact would have no
    ordering to stand on."""
    observed = ladder.PHASE_7D_OBSERVED
    registered = ladder.PHASE_7D_PATCH_AXIS_REGISTERED

    # (1) Arm 5 fired does_not_cliff -- a key that must exist in the
    # registration, worded as registered.
    assert observed["arm_5_discrimination"]["fired"] == "does_not_cliff"
    assert "does_not_cliff" in registered["fifth_arm_readings_committed"]
    assert "0.2280" in observed["arm_5_discrimination"]["figures"]
    assert "0.0857" in observed["arm_5_discrimination"]["figures"]
    assert "2.286x" in observed["arm_5_discrimination"]["figures"]
    assert "ZERO interpolation" in observed["arm_5_discrimination"][
        "arm_3_corroborates"
    ]
    # The grid-density refinement says the two candidates were one variable,
    # and says it is a refinement rather than a new registration.
    grid = observed["grid_density"]
    assert "deterministically" in grid["link"]
    assert grid["implicated"] == "grid density"
    assert "not a new registration" in grid["status"]

    # (2) The null is the recorded delta, re-derived.
    arms = observed["arms"]
    assert observed["arm_1"]["delta"] == round(
        arms["vit_b32"]["mean"] - arms["vit_b16"]["mean"], 4
    ) == -0.0001

    # (3) Arm 4: direction against the prior, explicitly unclaimable.
    assert observed["arm_4"]["delta"] == round(
        arms["concat"]["mean"] - arms["vit_b16"]["mean"], 4
    ) == 0.0074
    assert "-0.0542" in observed["arm_4"]["against_prior"]
    assert "UNCLAIMABLE PENDING" in observed["arm_4"]["status"]

    # (4) Arm 6 against the MATCHED Swin cell, reading_if_different, with
    # the registered sentence verbatim.
    arm6 = observed["arm_6"]
    assert arm6["swin_b_matched_cell"]["value"] == (
        ladder.STAGE_D1_AT_G1["cells"]["swin_b"][0]
    ) == 0.1076
    assert arm6["difference"] == round(0.1844 - 0.1076, 4)
    assert arm6["reading_applied"] == "reading_if_different"
    framing = ladder.PHASE_7D_BUILT["arm_6"]["framing"]
    assert framing["reading_if_different"] == arm6["sentence"]
    assert arm6["descriptive_pending_bca"] is True


def test_the_p7d_paired_scope_derives_its_six_contrasts(repo_root):
    """One arithmetic, two records: the pairs' recorded deltas re-derive
    from PHASE_7D_OBSERVED, the seeds are the shared five everywhere, and
    the cross-phase comparator's vectors resolve at Road B's own runs
    directory with hashes carried from its paired config."""
    import yaml as _yaml

    pairs = ladder.paired_claim_pairs("p7d")
    assert len(pairs) == 6
    assert {p["key"] for p in pairs} == {
        key for key, *_ in ladder.P7D_PAIRED_CONTRASTS
    }
    for pair in pairs:
        assert pair["seeds"] == list(ladder.SEED_POOL[:5]), pair["key"]
    by_key = {p["key"]: p for p in pairs}
    cross = by_key["p7d__b32_512_vs_patch16_512"]
    assert cross["a"] == "roadb_p7_arm_vit_b16_imagenet_512"
    assert cross["recorded"]["delta_of_means"] == round(0.2280 - 0.0857, 4)
    swin = by_key["p7d__mvitv2_vs_swin"]
    assert swin["recorded"]["delta_of_means"] == round(0.1844 - 0.1076, 4)

    payload = _yaml.safe_load(
        (repo_root / "configs" / "p7d_paired.yaml").read_text(encoding="utf-8")
    )
    assert payload["task"] == {
        "kind": "paired_claims", "scope": "p7d", "n_boot": 10000,
    }
    assert len(payload["inputs"]) == 40  # 8 run dirs x 5 seeds
    comparator = [
        e for e in payload["inputs"]
        if "roadb_p7_arm_vit_b16_imagenet_512" in e["name"]
    ]
    assert len(comparator) == 5, (
        "the ten-seed comparator must be declared at the five SHARED seeds "
        "only, or the short-band check reads five of ten as a partial"
    )
    for entry in comparator:
        assert "/runs/keeper/roadb_p7/" in entry["path"]
        assert set(entry["rollup_sha256"]) != {"0"}, (
            "the comparator's vectors were declared for the Road B paired "
            "run; the hash must carry from that config, not wait for a "
            "second declare"
        )
    # [FILLED 2026-08-15] Every vector is declared: real run directory and
    # the declare pass's hash, no placeholders, no pending paths. The five
    # 7D run dirs share one SHA (one launch commit), each with a job id the
    # mistyped-launch guard accepted before the paths were filled.
    for entry in payload["inputs"]:
        assert set(entry["rollup_sha256"]) != {"0"}, entry["name"]
        assert "PENDING_" not in entry["path"], entry["name"]
    p7d_shas = {
        e["path"].split("/")[-2].split("__")[1]
        for e in payload["inputs"] if "/runs/keeper/p7d/" in e["path"]
    }
    assert p7d_shas == {"c3a8dabd"}


def test_phase_7d_closes_with_two_claims_and_the_ledger_consistent():
    """**Closed: two grid-density claims, four withdrawals, headline
    unchanged** -- and every number in the closing agrees with the records
    it cites, so a later correction cannot land in one place only."""
    closing = ladder.PHASE_7D_CLOSING
    verdicts = closing["verdicts"]
    assert len(verdicts) == 6
    assert {p["key"] for p in ladder.paired_claim_pairs("p7d")} == set(verdicts)

    claimed = {k for k, v in verdicts.items() if v["verdict"] == "CLAIMABLE"}
    assert claimed == {"p7d__patch8_vs_b16", "p7d__b32_512_vs_patch16_512"}
    for key in claimed:
        assert verdicts[key]["n_excluding_zero"] == "5/5"
    # The claims' margins are the ones the margin table gained.
    from cleft import roadb

    assert verdicts["p7d__patch8_vs_b16"]["margin"] == 6.42
    assert verdicts["p7d__b32_512_vs_patch16_512"]["margin"] == 5.32
    assert 6.42 in roadb.CONDITION_1_MARGIN_STRUCTURE["survivors"]
    assert 5.32 in roadb.CONDITION_1_MARGIN_STRUCTURE["survivors"]
    assert 1.77 in roadb.CONDITION_1_MARGIN_STRUCTURE["withdrawals"]
    assert 1.89 in roadb.CONDITION_1_MARGIN_STRUCTURE["withdrawals"]

    # (2) The claim retires the interpolation mechanism jointly, and the
    # deterministic linkage travels with it.
    claim = closing["claims"]
    assert "ZERO interpolation" in claim["interpolation_mechanism_retired"]
    assert "IDENTICAL 2.286x" in claim["interpolation_mechanism_retired"]
    assert "grid density" in claim["one_variable"]
    assert "not two separable candidates" in claim["one_variable"]

    # (3) Arm 6's downgrade is in place, dated, with the forbidden claim.
    downgraded = ladder.PHASE_7D_OBSERVED["arm_6"]["downgraded"]
    assert downgraded["date"] == "2026-08-15"
    assert downgraded["bca"]["vs_vit_b16"]["margin"] == 1.77
    assert downgraded["bca"]["vs_swin_b"]["margin"] == 1.89
    assert "NEITHER" in downgraded["stands"]
    assert "FORBIDDEN" not in downgraded["forbidden"]  # the key names it
    assert "as a\nCLAIM" in downgraded["forbidden"] or "as a CLAIM" in (
        downgraded["forbidden"].replace("\n", " ")
    )
    # The original reading's fields are KEPT, not rewritten -- the
    # downgrade sits beside them.
    arm6 = ladder.PHASE_7D_OBSERVED["arm_6"]
    assert arm6["reading_applied"] == "reading_if_different"
    assert arm6["descriptive_pending_bca"] is True

    # (4) Concat: sign disagreement, neither confirmed nor overturned.
    assert "disagree in SIGN" in closing["concat_closes"]
    assert "-0.0542" in closing["concat_closes"]
    assert "neither confirmed nor overturned" in closing["concat_closes"]

    # (1) The expectation resolution names the date order.
    assert closing["expectation_resolution"]["resolved_as_registered"] is True
    assert "committed before the paired run" in closing[
        "expectation_resolution"
    ]["date_order"]

    # (6) Headline unchanged, the b32 match recorded as descriptive.
    assert "0.2520" in closing["headline"]
    assert "quarter of the tokens" in closing["headline"]
    # And the closing deltas agree with the pair enumeration's recorded
    # means to within BCa-vs-recorded rounding (1.5e-4, the TRADE_OFF rule).
    by_key = {p["key"]: p for p in ladder.paired_claim_pairs("p7d")}
    for key, verdict in verdicts.items():
        recorded = by_key[key]["recorded"]["delta_of_means"]
        assert abs(verdict["delta"] - recorded) < 4e-3, key

    assert closing["still_untouched"] == ("8b", "8c", "t-SNE")
