"""Phase 10's opening records: the comparator's two tables, the measured
ROI check, the IEM constants carried for Phase 11, and the silences."""

from __future__ import annotations

import numpy as np
import pytest

from cleft import phase10
from cleft.models import srgnn


def test_the_comparator_carries_both_tables_and_the_598_analysis():
    record = phase10.CLEFTGNN_COMPARATOR_TABLES
    assert record["recorded"].startswith("2026-08-16")
    assert record["study_test_set"]["table"] == "Table 1"
    assert record["study_test_set"]["n"] == 28
    assert record["benchmark_test_set"]["table"] == "Table 5"
    assert record["benchmark_test_set"]["n"] == 25
    assert record["benchmark_test_set"]["per_rater_model"] == (
        -0.273, 0.323, -0.092, -0.121, 0.598,
    )
    cell = record["the_598_cell"]
    assert cell["fisher_ci_95_recomputed"] == (0.2656, 0.8033)
    assert "15 cells" in cell["selection"]
    assert "0.4696" in cell["plausibility"]
    assert "0.2512" in record["our_number_beside_it"]
    assert "never pooled" in record["our_number_beside_it"]
    assert "without the other" in record["rule"]


def test_the_fisher_ci_reproduces_from_the_published_cell():
    """The recorded interval is recomputable, not carried arithmetic."""
    r, n = 0.598, 25
    z = np.arctanh(r)
    se = 1.0 / np.sqrt(n - 3)
    low, high = np.tanh(z - 1.959963985 * se), np.tanh(z + 1.959963985 * se)
    recorded = phase10.CLEFTGNN_COMPARATOR_TABLES["the_598_cell"][
        "fisher_ci_95_recomputed"
    ]
    assert abs(low - recorded[0]) < 5e-5
    assert abs(high - recorded[1]) < 5e-5


def test_the_roi_check_identity_holds_against_the_live_code():
    """The record's claim re-verified against grid_rois itself: the
    sliding-window enumeration (sizes {1..3}x{1..3} minus 1x1, stride one
    cell, whole map separate) equals the code's set exactly."""
    boxes = srgnn.grid_rois()
    assert len(boxes) == 26
    step = srgnn.ROI_RESOLUTION / srgnn.GRID_SIZE
    code_set = {tuple(int(v) for v in box) for box in boxes}
    recon = set()
    for w in (1, 2, 3):
        for h in (1, 2, 3):
            if (w, h) in ((1, 1), (3, 3)):
                continue
            for x in range(4 - w):
                for y in range(4 - h):
                    recon.add((
                        int(x * step), int(y * step),
                        int(w * step), int(h * step),
                    ))
    assert recon == code_set

    record = phase10.ROI_CHECK_MEASURED
    assert record["verdict"].startswith("no set contradiction")
    assert "UNDERDETERMINES" in record["verdict"]
    assert record["replication_builds"].startswith("srgnn.grid_rois")
    composition = record["code_set"]["composition_by_cells"]
    assert sum(composition.values()) == 26
    assert "one sentence" in record["for_supervisor"]


def test_the_iem_constants_are_the_published_equation_verbatim():
    record = phase10.IEM_CARRIED_FOR_PHASE_11
    assert record["recorded"].endswith("record only, no action")
    assert record["optimistic"] == {
        "condition": "y_hat - G < 0", "form": "1.2 * |d|**1.12",
    }
    assert record["pessimistic"] == {
        "condition": "y_hat - G >= 0", "form": "0.8 * |d|**0.87",
    }
    assert record["bounds"] == (0, 4)
    assert "before any loss build" in record["pre_step"]
    assert "max(0, 4-|i-G|)" in record["also_available"]


def test_the_scope_restatement_lists_seven_silences_unfilled():
    record = phase10.PHASE_10_SCOPE_AND_SILENCES
    assert record["restated"].endswith("section 5")
    assert "verified SR-GNN port" in record["commits"]["from_existing"]
    assert "Spatial Attention Branch" in record["commits"]["new_code"]
    assert "DAYS" in record["commits"]["training"]
    assert "both\\nconditions" not in record["commits"]["criterion"]
    assert "paired BCa" in record["commits"]["criterion"]
    assert len(record["silences"]) == 7
    assert any("SABM" in s for s in record["silences"])
    assert any("exit criteria" in s for s in record["silences"])
    assert any("0.2520" in s for s in record["silences"])


# --------------------------------------------------------------------------
# 2026-08-16, the registration turn: the corrected ROI finding, the five
# decisions, and the blocked build
# --------------------------------------------------------------------------


def test_the_notebook_roi_set_refutes_the_27_and_reenumerates_exactly():
    """The notebook's own generator, re-enumerated float-exact: 37 raw,
    36 unique, nine 1x1 cells IN, the appended whole deduplicating
    cleanly -- and the dated correction sits on the earlier check."""
    image, C = 224, 3
    cell = image / C
    rois = []
    for r in range(C):
        for c in range(C):
            for sh in range(1, C - r + 1):
                for sw in range(1, C - c + 1):
                    rois.append((c * cell, r * cell, sw * cell, sh * cell))
    rois.append((0.0, 0.0, float(image), float(image)))
    unique = set(rois)
    assert len(rois) == 37 and len(unique) == 36
    assert 3 * cell == 224.0
    singles = [b for b in unique if b[2] == cell and b[3] == cell]
    assert len(singles) == 9

    record = phase10.NOTEBOOK_ROI_SET_MEASURED
    assert record["notebook"]["regions"] == 36
    assert record["repo_port"]["regions"] == 27
    assert "unimplementable" in record["manuscript"]
    assert "class default" in record["amendment_claim_refuted"]
    assert "only the group can answer" in record["thursday_agenda"]
    assert phase10.ROI_CHECK_MEASURED["corrected"].startswith("2026-08-16")


def test_the_registration_carries_the_five_decisions_and_the_sabm_form():
    spec = phase10.PHASE_10_REGISTERED
    assert "premise_refuted_same_turn" in spec["region_scheme"]
    assert "the builder picks neither" in (
        spec["region_scheme"]["premise_refuted_same_turn"]
    )
    assert spec["output_readings"]["primary"].startswith("softmax-expected")
    assert "eq. 13" in spec["output_readings"]["beside"]
    assert "neither selected after the fact" in spec["output_readings"]["rule"]
    assert spec["label"]["train"].startswith("cross-entropy")
    assert "every report of the number" in spec["label"]["caveat"]
    assert "MODE" in spec["label"]["consensus_column_interpretation"]
    assert "awaiting confirmation" in (
        spec["label"]["consensus_column_interpretation"]
    )
    assert "SGD lr 0.01" in spec["regime"]["recipe"]
    assert "zero mentions" in spec["regime"]["epochs"]
    assert "epochs 2-11" in spec["regime"]["epochs"]
    assert spec["backbone"].startswith("resnet50")
    sabm = spec["sabm_published_form"]
    assert "VerticalCenter" in sabm["eq7"]
    assert sabm["eq8"] == "v_a = sum_i(F_att,i * m_i)"
    assert "additive" in sabm["eq9"] and "OLD design" in sabm["eq9"]
    assert len(spec["exit_criteria"]) == 6
    assert spec["estimate"].startswith("days")


def test_the_build_is_blocked_on_four_choices_with_no_config_shipped():
    from pathlib import Path

    blocked = phase10.PHASE_10_BUILD_BLOCKED_ON
    assert len(blocked["choices"]) == 4
    assert any("notebook-36" in c and "port-27" in c for c in blocked["choices"])
    assert any("proportional to" in c for c in blocked["choices"])
    assert any("momentum" in c for c in blocked["choices"])
    assert any("MODE" in c for c in blocked["choices"])
    assert "GRAPH_NODE_PATH_IS_NOT_WIRED" in blocked["consequence"]
    # [2026-08-16, later the same day] The block RESOLVED (the maintainer's
    # four answers, PHASE_10_UNBLOCKED) and the config now exists -- the
    # no-config assertion inverted with the state, dated.
    assert blocked["resolved"].startswith("2026-08-16")
    repo = Path(__file__).resolve().parents[1]
    # [2026-08-17] Two more cells joined: the faithful arm and the rater
    # screen. The list is pinned, not merely non-empty, so a fourth config
    # arriving unregistered fails here. [2026-08-17, later] And a fourth
    # DID arrive -- the notebook-recipe cell, registered at
    # NOTEBOOK_RECIPE_CELL_REGISTERED. The pin caught it, which is the pin
    # working rather than the pin being wrong. [2026-08-17, closing] And
    # a fifth: the paired BCa, the phase's last outstanding run
    # (PHASE_10_PAIRED_REGISTERED).
    assert [p.name for p in sorted((repo / "configs").glob("p10_*.yaml"))] == [
        "p10_cleftgnn.yaml",
        "p10_cleftgnn_faithful.yaml",
        "p10_cleftgnn_notebook.yaml",
        "p10_paired.yaml",
        "p10_rater_screen.yaml",
    ]


# --------------------------------------------------------------------------
# 2026-08-16, the build turn: unblocked, model, config
# --------------------------------------------------------------------------


def test_the_unblock_carries_the_four_answers_and_the_cc89d27_note():
    record = phase10.PHASE_10_UNBLOCKED
    assert record["unblocked"] == "2026-08-16"
    assert "overstates" in record["cc89d27_note"]
    answers = record["answers"]
    assert answers["region_set"].startswith("notebook-36")
    assert answers["m_i"] == "y_center/224, committed blind"
    assert "momentum 0" in answers["recipe"]
    assert "batch 16" in answers["recipe"]
    assert "the sheet wins" in answers["consensus"]


def test_the_model_regions_and_mask_match_the_registration():
    from cleft.models import cleftgnn

    rois = cleftgnn.cleftgnn_rois()
    assert rois.shape == (36, 4)
    # The array is float32; compare tolerantly, never by float64 equality.
    cell = np.float32(224 / 3)
    is_cell = lambda v: np.isclose(v, cell, rtol=1e-6)
    singles = [b for b in rois if is_cell(b[2]) and is_cell(b[3])]
    assert len(singles) == 9
    wholes = [b for b in rois if b[2] == 224.0 and b[3] == 224.0]
    assert len(wholes) == 1

    mask = cleftgnn.vertical_mask(rois)
    assert mask.shape == (36,)
    whole_row = int(np.flatnonzero(
        (rois[:, 2] == 224.0) & (rois[:, 3] == 224.0)
    )[0])
    assert mask[whole_row] == 0.5
    # A bottom-row cell outweighs a top-row cell; everything is in (0, 1].
    top = [i for i, b in enumerate(rois) if b[1] == 0 and is_cell(b[3])]
    bottom = [
        i for i, b in enumerate(rois)
        if np.isclose(b[1], 2 * cell, rtol=1e-6) and is_cell(b[3])
    ]
    assert mask[bottom[0]] > mask[top[0]]
    assert np.all(mask > 0) and np.all(mask <= 1)


def test_expected_grade_reads_the_softmax_as_registered():
    from cleft.models import cleftgnn

    uniform = np.full((1, 5), 0.2)
    assert cleftgnn.expected_grade(uniform)[0] == 3.0
    hard = np.zeros((1, 5))
    hard[0, 4] = 1.0
    assert cleftgnn.expected_grade(hard)[0] == 5.0


def test_the_built_record_and_config_carry_the_registered_recipe():
    import yaml
    from pathlib import Path

    built = phase10.CLEFTGNN_BUILT
    assert built["regions"] == 36
    assert "calibrated CE init" in built["recipe"]
    assert "monitor inner_val_mse" in built["early_stopping"]
    assert "0.2520" in built["readings"]["primary"]
    assert len(built["caveats_travel"]) == 3

    repo = Path(__file__).resolve().parents[1]
    config = yaml.safe_load(
        (repo / "configs" / "p10_cleftgnn.yaml").read_text(encoding="utf-8")
    )
    arm = yaml.safe_load((
        repo / "configs" / "p7_d1_vit_b16_imagenet_g1.yaml"
    ).read_text(encoding="utf-8"))
    task = config["task"]
    assert task["kind"] == "cleftgnn_cv"
    assert task["seeds"] == arm["task"]["seeds"]
    assert (task["learning_rate"], task["momentum"], task["batch_size"]) == (
        0.01, 0.0, 16,
    )
    assert (task["inner_val_frac"], task["max_epochs"], task["patience"]) == (
        0.2, 40, 5,
    )
    by_name = {e["name"]: e for e in config["inputs"]}
    assert set(by_name) == {"manifest_v1", "staged_v1", "scoresheet_primary"}
    assert all(set(e["rollup_sha256"]) != {"0"} for e in config["inputs"])


# [RETIRED 2026-08-17] ``test_sheet_consensus_reader_refuses_ambiguity_and
# _reads_by_photo`` lived here. It pinned ``run.phase10_sheet_consensus``,
# an ad-hoc reader that searched for a "consensus" header the sheet does
# not carry -- the premise, not the code, was wrong
# (phase10.CONSENSUS_LABEL_IS_THE_MEDIAN). The reader is deleted and its
# job now belongs to ``scoresheet.load_median``, whose contract is pinned
# by ``test_load_median_verifies_the_column_against_the_rater_cells``
# below -- including the same zero-or-multiple-column refusal this test
# was written for.


# --------------------------------------------------------------------------
# 2026-08-17: the label is the sheet's Median; the mode reading is refuted
# --------------------------------------------------------------------------


def test_the_median_label_record_refutes_the_mode_and_cites_the_237_figure():
    from cleft.data import labels

    record = phase10.CONSENSUS_LABEL_IS_THE_MEDIAN
    assert record["measured"].startswith("2026-08-17")
    assert record["label"] == "the sheet's Median column"
    assert "NO consensus column exists" in record["columns"]
    assert "refuted by measurement" in record["mode_refuted"]
    assert "241" in record["derived_files"] and "245" in record["derived_files"]
    # The learnability figure is the repo's own 237 measurement, not the
    # 251-row one -- the population trap labels.py exists to prevent.
    assert record["learnability_237"] == labels.LEARNABILITY_237["median"]
    assert record["learnability_237"] != labels.LEARNABILITY_251["median"]
    assert "0.5561" in record["learnability_population_trap"]
    # The design-fidelity consequence carries the measured gap.
    assert "0.6022" in record["design_fidelity_consequence"]
    assert labels.LEARNABILITY_237["mean"] == 0.6022
    # The round-half-up statement: searched, and it was in no record.
    assert "no record" in record["round_half_up_statement"]
    assert "half-to-EVEN" in record["round_half_up_statement"]

    # The superseded readings stand as history, each with its dated note.
    registered = phase10.PHASE_10_REGISTERED["label"]
    assert "MODE" in registered["consensus_column_interpretation"]
    assert registered["consensus_column_corrected"].startswith("2026-08-17")
    unblocked = phase10.PHASE_10_UNBLOCKED["answers"]
    assert "MODE ties-lower" in unblocked["consensus"]
    assert unblocked["consensus_corrected"].startswith("2026-08-17")


def test_the_void_first_launch_and_the_retry_item_collect_it():
    from cleft import phase8

    void = phase10.CLEFTGNN_FIRST_LAUNCH_VOID
    assert void["run"] == "p10_cleftgnn__9a28f8a8__p10-cleftgnn"
    assert void["attempts"] == 4
    assert "consensus" in void["cause"]
    assert void["citable"].startswith("nothing")
    assert phase8.RETRY_LIMIT_IS_NOT_HOLDING["updated_2"].startswith("2026-08-17")
    assert "+3 retries" in phase8.RETRY_LIMIT_IS_NOT_HOLDING["updated_2"]


def test_load_median_verifies_the_column_against_the_rater_cells(tmp_path):
    """The reader's contract: the sheet's Median is READ but never TRUSTED
    -- a row whose column disagrees with its own five cells raises, and a
    sheet without exactly one Median column raises."""
    openpyxl = pytest.importorskip("openpyxl")
    from cleft.data import scoresheet

    def write(path, rows, median_header="Median"):
        book = openpyxl.Workbook()
        sheet = book.active
        sheet.append([scoresheet.ID_COLUMN, *scoresheet.RATERS, "Average",
                      median_header])
        for photo_id, grades, median in rows:
            sheet.append([photo_id, *grades, sum(grades) / 5, median])
        book.save(path)
        return path

    # The maintainer's verified rows: 241 -> 2, 243 -> 3, 245 -> 3.
    good = write(tmp_path / "good.xlsx", [
        (241, (2, 3, 2, 1, 3), 2),
        (243, (3, 3, 3, 3, 4), 3),
        (245, (3, 3, 2, 4, 4), 3),
    ])
    assert scoresheet.load_median(good) == {241: 2, 243: 3, 245: 3}
    assert scoresheet.load_median(good, expected_rows=3) == {241: 2, 243: 3, 245: 3}

    # A drifted column -- the exact failure the module docstring names.
    drifted = write(tmp_path / "drifted.xlsx", [(241, (2, 3, 2, 1, 3), 4)])
    with pytest.raises(scoresheet.ScoreSheetError, match="two stories"):
        scoresheet.load_median(drifted)

    # No Median column at all (the shape my own reader assumed).
    missing = write(tmp_path / "missing.xlsx", [(241, (2, 3, 2, 1, 3), 2)],
                    median_header="Consensus")
    with pytest.raises(scoresheet.ScoreSheetError, match="exactly one"):
        scoresheet.load_median(missing)

    # load() is unchanged: it still ignores the derived columns.
    assert scoresheet.load(good).rows[241].grades == (2, 3, 2, 1, 3)


def test_the_model_refuses_non_integer_labels_instead_of_rounding():
    from cleft.models import cleftgnn

    assert cleftgnn._as_grades([1, 3, 5]).tolist() == [1, 3, 5]
    with pytest.raises(cleftgnn.CleftGNNError, match="integer grades"):
        cleftgnn._as_grades([2.5, 3.0])
    with pytest.raises(cleftgnn.CleftGNNError, match="outside"):
        cleftgnn._as_grades([0, 3])


def test_the_nan_measurement_refutes_log_zero_and_names_the_real_source():
    """2026-08-17: the hypothesis was measured, not adopted. The bias is
    smoothed and finite; the source is first-step divergence from summed
    36-region features; both apparatus findings are recorded."""
    import numpy as np

    from cleft.models import cleftgnn

    record = phase10.CLEFTGNN_NAN_MEASURED
    assert "REFUTED" in record["hypothesis_refuted"]
    assert "-23.65" in record["hypothesis_refuted"]
    assert "first" in record["actual_source"].lower()
    assert "29,519" in record["actual_source"]
    assert record["scale_at_init"]["sum"] == 21.5
    assert "FREEZES" in record["the_combination_no_source_prescribed"]
    assert "substitute" in record["caveats"][0]
    findings = record["apparatus_findings"]
    assert "False" in findings["gate_3_misses_non_finite"]
    assert "FROZEN" in findings["gate_3_misses_non_finite"]
    assert "no CPU kernel" in findings["not_laptop_testable"]
    assert "NOT the crash cause" in record["thin_class_artifact_still_open"]

    # The refutation is reproducible from the shipped arithmetic: an absent
    # class gives a finite, large-negative bias -- never -inf.
    counts = np.array([4, 71, 88, 24, 0], dtype=np.float64)
    frequencies = (counts + 1e-8) / (counts.sum() + cleftgnn.NUM_CLASSES * 1e-8)
    bias = np.log(frequencies)
    assert np.isfinite(bias).all()
    assert -24.0 < bias[-1] < -23.0

    void = phase10.CLEFTGNN_SECOND_LAUNCH_VOID
    assert void["run"] == "p10_cleftgnn__32519178__p10-cleftgnn-2"
    assert void["attempts"] == 4
    assert "1:5" in void["label_worked"]
    assert void["fix"].startswith("NOT BUILT")


# --------------------------------------------------------------------------
# 2026-08-17: freeze, smoothing, the faithful arm, the rater screen
# --------------------------------------------------------------------------


def test_the_freeze_is_recorded_as_a_fidelity_correction():
    record = phase10.BACKBONE_IS_FROZEN
    assert record["corrected"].endswith("a fidelity correction, not a fix")
    assert "inference" in record["was"]
    assert "eval()" in record["now"]
    assert "notebook" in record["why"]
    assert "like-for-like" in record["benefit"]
    assert "void ladder" in record["batchnorm"]
    assert "clipping" in record["alternatives_rejected"]


def test_the_backbone_is_actually_frozen_and_the_optimiser_agrees():
    """Not just recorded -- asserted on a constructed model: zero
    trainable backbone parameters, everything else trainable, and the
    optimiser handed only what trains."""
    pytest.importorskip("torch")
    import torch

    from cleft.models import cleftgnn

    torch.manual_seed(0)
    backbone = cleftgnn.CleftGNNBackbone(pretrained=False, device="cpu")
    backbone.reset(np.array([1, 2, 2, 3, 3, 3, 4], dtype=float))

    model = backbone._model
    assert not any(p.requires_grad for p in model.backbone.parameters())
    assert all(p.requires_grad for p in model.classifier.parameters())
    report = backbone.parameter_report
    assert report["frozen_backbone_parameters"] > 0
    assert report["trainable_parameters"] == (
        report["total_parameters"] - report["frozen_backbone_parameters"]
    )
    optimised = sum(
        p.numel() for group in backbone._optimizer.param_groups
        for p in group["params"]
    )
    assert optimised == report["trainable_parameters"]


def test_the_laplace_smoother_makes_an_absent_grade_reachable():
    pytest.importorskip("torch")
    import torch

    from cleft.models import cleftgnn

    torch.manual_seed(0)
    backbone = cleftgnn.CleftGNNBackbone(pretrained=False, device="cpu")
    # No grade 5 anywhere in the training labels.
    backbone.reset(np.array([1, 2, 2, 3, 3, 3, 4], dtype=float))
    bias = backbone._model.classifier.bias.detach().numpy()
    assert np.isfinite(bias).all()
    # The old epsilon gave -23.65; Laplace keeps the absent grade reachable.
    assert bias[-1] > -8.0
    probabilities = np.exp(bias) / np.exp(bias).sum()
    assert probabilities[-1] > 1e-3

    record = phase10.THIN_CLASS_SMOOTHING
    assert "(count + 1) / (N + K)" in record["now"]
    assert "-23.65" in record["was"]
    assert "Thursday-flagged" in record["manuscript_silence"]


def test_the_deviations_are_enumerated_in_one_place():
    """[2026-08-17, extended twice] Two became four when the divergence
    was measured, then FIVE when route 3 was chosen on the stage table.
    Every entry must carry a fidelity cost, and lr/sum must be recorded
    as untouched."""
    record = phase10.REGISTERED_DEVIATIONS
    assert len(record["deviations"]) == record["expected_count"] == 5
    assert any("GNN branch" in d for d in record["deviations"])
    assert any("early stopping" in d for d in record["deviations"])
    assert any("Laplace" in d for d in record["deviations"])
    assert any("LayerNorm" in d for d in record["deviations"])
    assert any("nn.Linear's own init" in d for d in record["deviations"])
    # Every deviation states a fidelity cost -- the point of the list.
    assert all("fidelity cost" in d for d in record["deviations"])
    # The freeze is explicitly NOT a deviation.
    assert any("fidelity correction" in n for n in record["not_deviations"])
    # The two things the fix deliberately did not touch.
    untouched = record["untouched_deliberately"]
    assert any("lr 0.01" in u for u in untouched)
    assert any("sum" in u for u in untouched)
    assert any("clipping" in u for u in untouched)


def test_the_faithful_arm_registration_states_its_purpose_and_the_598():
    record = phase10.FAITHFUL_ARM_REGISTERED
    assert "ONLY in protocol" in record["purpose"]
    protocol = record["protocol"]
    assert "five models" in protocol["models"]
    assert "85:15" in protocol["split"]
    assert "NO intervals" in protocol["runs"]
    assert "25-image benchmark" in protocol["test_shapes"]
    cell = record["their_598_cell"]
    assert cell["one_of"] == 15 and cell["n"] == 25
    assert cell["fisher_95_recomputed"] == (0.2656, 0.8033)
    assert cell["same_model_on_their_study_set"] == 0.283
    assert "sampling noise" in cell["reading"]


def test_the_rater_screen_prior_is_committed_both_ways():
    from cleft.data import labels

    record = phase10.RATER_SCREEN_REGISTERED
    prior = record["prior"]
    assert prior["panel_mean_learnability"] == labels.LEARNABILITY_237["mean"]
    assert prior["orthodontist_learnability"] == (
        labels.LEARNABILITY_237["orthodontist"]
    )
    assert "BELOW 0.2520" in prior["prediction"]
    assert "WORSE" in record["readings"]["all_below"]
    assert "refuted" in record["readings"]["any_meaningfully_above"]
    assert "0.0148" in record["readings"]["any_meaningfully_above"]

    ladder = phase10.RATER_LADDER_CONDITIONAL
    assert ladder["runs"] == 400
    assert 5 * 4 * 2 * 2 * 5 == ladder["runs"]
    assert "not built" in ladder["registered"]


def test_top1_macro_prf_excludes_absent_classes_and_scores_perfectly():
    truth = np.array([1, 1, 2, 2, 3])
    assert phase10.top1_macro_prf(truth, truth.copy())["f1_macro"] == 1.0

    report = phase10.top1_macro_prf(truth, truth.copy())
    assert report["classes_present"] == (1, 2, 3)
    assert report["classes_absent_excluded"] == (4, 5)

    # A class predicted but never true does not enter the macro average;
    # a class never predicted scores zero precision, not a division error.
    predicted = np.array([1, 1, 2, 2, 5])
    report = phase10.top1_macro_prf(truth, predicted)
    assert report["accuracy"] == 0.8
    assert report["recall_macro"] == pytest.approx((1.0 + 1.0 + 0.0) / 3)


def test_the_three_p10_configs_are_declared_and_copy_the_arm():
    import yaml
    from pathlib import Path

    repo = Path(__file__).resolve().parents[1]
    arm = yaml.safe_load((
        repo / "configs" / "p7_d1_vit_b16_imagenet_g1.yaml"
    ).read_text(encoding="utf-8"))

    faithful = yaml.safe_load((
        repo / "configs" / "p10_cleftgnn_faithful.yaml"
    ).read_text(encoding="utf-8"))
    assert faithful["task"]["kind"] == "cleftgnn_faithful"
    assert {e["name"] for e in faithful["inputs"]} == {
        "manifest_v1", "staged_v1", "scoresheet_primary", "deall_set",
    }
    assert all(set(e["rollup_sha256"]) != {"0"} for e in faithful["inputs"])

    screen = yaml.safe_load((
        repo / "configs" / "p10_rater_screen.yaml"
    ).read_text(encoding="utf-8"))
    assert screen["task"]["kind"] == "rater_screen"
    assert screen["task"]["seeds"] == arm["task"]["seeds"]
    # The refit block is COPIED so "the same arm" is true by construction.
    for key in ("learning_rate", "weight_decay", "inner_val_frac",
                "max_epochs", "patience", "monitor", "geometry", "label"):
        assert screen["task"][key] == arm["task"][key], key
    assert all(set(e["rollup_sha256"]) != {"0"} for e in screen["inputs"])


# --------------------------------------------------------------------------
# 2026-08-17: the screen's result, the faithful arm's diagnosis, the scale
# measurement that refuted the backbone-gradient hypothesis
# --------------------------------------------------------------------------


def test_the_screen_result_fired_the_registered_mixed_reading():
    from cleft.data import labels

    record = phase10.RATER_SCREEN_OBSERVED
    per_rater = record["per_rater"]
    assert per_rater["plastic surgeon"]["pcc"] == 0.2703
    assert per_rater["cleft patient"]["pcc"] == -0.0098
    assert record["verdict"].startswith("MIXED")
    # The one cell above the arm is inside the registered band: 1.24 sd,
    # against a two-sd refutation threshold. Recomputed, not asserted.
    excess = per_rater["plastic surgeon"]["pcc"] - record["arm"]
    assert excess / 0.0148 < 2.0
    assert "1.24" in record["inside_the_band"]
    # Four of five below the arm.
    below = [r for r, c in per_rater.items() if c["pcc"] < record["arm"]]
    assert len(below) == 4
    # The Phase 1 prior, confirmed by a second route.
    assert "0.6022" in record["prior_confirmed_independently"]
    assert labels.LEARNABILITY_237["mean"] == 0.6022
    assert "fourth of five" in record["orthodontist_premise_fails_again"]


def test_the_faithful_arm_diagnosis_states_the_confound_and_its_limits():
    record = phase10.FAITHFUL_ARM_OBSERVED
    assert "CONSTANT" in record["mechanism"]
    assert "RETURNS nan" in record["mechanism"] and "RAISES" in record["mechanism"]
    # The F1 arithmetic for a constant predictor, recomputed here.
    for counts, expected in (([3, 7, 6, 6, 3], 0.088), ):
        n = sum(counts)
        prevalence = max(counts) / n
        macro = (2 * prevalence / (prevalence + 1)) / len(counts)
        assert abs(macro - expected) < 0.005
    assert "0.088" in record["f1_arithmetic_confirms"]
    # The confound governs what may be said.
    assert "our divergence explains the collapse" in record["confound"]
    assert len(record["does_not_show"]) == 2
    assert any("does NOT refute their published" in s
               for s in record["does_not_show"])
    assert any("does NOT yet show the regime's fragility" in s
               for s in record["does_not_show"])
    assert "REPORTING regime" in record["does_show"]
    assert record["regime_question"].startswith("OPEN")


def test_the_scale_measurement_refutes_the_backbone_gradient_diagnosis():
    record = phase10.CLEFTGNN_SCALE_MEASURED
    assert record["backbone_gradient_diagnosis"].startswith("REFUTED")
    assert record["parameters"] == {"frozen": 23508032, "trainable": 5778949}
    assert "sd 0.000000" in record["epoch_0_predictions_constant_by_construction"]
    divergence = record["divergence_inside_epoch_1"]
    assert divergence["first_batch_loss"] == 1.2975
    assert divergence["epoch_1_mean_loss"] == 15.0116
    assert "21.5" in record["surviving_hypothesis"]
    assert "substitute" in record["caveats"]

    probe = phase10.P10_SCALE_PROBE
    assert probe["script"] == "scripts/p10_scale_probe.py"
    assert len(probe["prints"]) == 5
    assert any("zero sd is collapse" in p for p in probe["prints"])

    void = phase10.CLEFTGNN_THIRD_LAUNCH_VOID
    assert void["attempts"] == 4
    assert "+6" in void["retry_item"]


def test_the_pod_probe_script_exists_and_compiles():
    import py_compile
    from pathlib import Path

    script = Path(__file__).resolve().parents[1] / "scripts" / "p10_scale_probe.py"
    assert script.is_file()
    py_compile.compile(str(script), doraise=True)
    text = script.read_text(encoding="utf-8")
    # A probe, not a run: it neither imports nor constructs a RunContext
    # and writes no artifact. (The name appears in its docstring, saying
    # exactly that -- so the check is on USE, not mention.)
    assert "import RunContext" not in text
    assert "RunContext(" not in text
    assert "ctx.path" not in text and "ctx.atomic" not in text


# --------------------------------------------------------------------------
# 2026-08-17: the shared mechanism settled, sd-zero's two causes, the fix
# --------------------------------------------------------------------------


def test_the_shared_mechanism_is_settled_by_the_pod_figures():
    record = phase10.ARMS_SHARE_THE_MECHANISM
    assert record["f_t_real_features"] == {"protocol": 21.1, "faithful": 21.1}
    assert record["offline_estimate_validated"] == 21.5
    assert record["logits_after_one_step"]["protocol"] == (3.89, 12.75)
    assert record["logits_after_one_step"]["faithful"] == (3.94, 18.21)
    assert record["epoch_1_mean_loss"]["faithful"] == 1.3e16
    assert "demonstrates nothing about their protocol" in record["conclusion"]
    assert record["regime_question"].startswith("OPEN, unledgered")
    # The regime finding stayed OUT of the ledger, as registered.
    from cleft import results_ledger

    assert not any(
        "regime" in entry["claim"] and entry["status"] != "VOID"
        for entry in results_ledger.ENTRIES
    )


def test_sd_zero_is_recorded_as_two_different_findings():
    record = phase10.SD_ZERO_HAS_TWO_CAUSES
    assert "BY CONSTRUCTION" in record["cause_1_epoch_0"]
    assert "Fix (b) removes it" in record["cause_1_epoch_0"]
    assert "SATURATION" in record["cause_2_epoch_1"]
    assert "not trained yet" in record["why_it_matters"]
    assert "ruined" in record["why_it_matters"]


def test_the_fix_is_recorded_with_its_measurement_and_its_caveat():
    record = phase10.FUSED_NORM_AND_STANDARD_INIT
    assert "torch.sum intact" in record["fix_a"]
    assert "mine to give up" in record["fix_b"]
    assert any("lr 0.01" in u for u in record["untouched"])
    measured = record["measured_under_the_fix"]
    assert measured["protocol"]["logits_after_step_1"] == (3.65, 4.40)
    assert measured["protocol"]["epoch_1_mean_loss"] == 3.868
    assert measured["faithful"]["epoch_1_mean_loss"] == 3.480
    assert "4,581" in measured["against_pre_fix_pod"]
    assert record["verdict"].startswith("the divergence is gone")
    # The caveat is not optional: the losses sit above ln(5) on noise.
    assert "ABOVE ln(5)" in record["caveat"]
    assert "NOT that the arm trains well" in record["caveat"]
    assert record["parameters"]["added_by_layernorm"] == 2048


def test_the_model_carries_the_fused_norm_and_a_non_constant_epoch_zero():
    """The fix is in the model, not only in the record: a LayerNorm sits
    before the classifier, the classifier's weight is no longer zero, and
    the Laplace bias survives."""
    pytest.importorskip("torch")
    import torch

    from cleft.models import cleftgnn

    torch.manual_seed(0)
    backbone = cleftgnn.CleftGNNBackbone(pretrained=False, device="cpu")
    backbone.reset(np.array([1, 2, 2, 3, 3, 3, 4], dtype=float))
    model = backbone._model

    assert isinstance(model.fused_norm, torch.nn.LayerNorm)
    assert model.fused_norm.normalized_shape == (cleftgnn.GNN_OUT_DIM,)
    # Fix (b): the weight is the framework's own small-random init.
    weight = model.classifier.weight.detach().numpy()
    assert weight.any(), "the zero-weight init should be gone"
    assert np.abs(weight).max() < 0.1
    # The Laplace bias survives, and an absent grade stays reachable.
    bias = model.classifier.bias.detach().numpy()
    assert np.isfinite(bias).all() and bias[-1] > -8.0
    # The LayerNorm's parameters are trainable and counted.
    assert all(p.requires_grad for p in model.fused_norm.parameters())


# --------------------------------------------------------------------------
# 2026-08-17: both arms complete but collapsed; the collapse diagnosed
# --------------------------------------------------------------------------


def test_the_collapsed_arms_are_recorded_with_the_forbidden_reading():
    record = phase10.ARMS_COMPLETE_BUT_COLLAPSED
    protocol = record["protocol_arm"]
    assert protocol["pcc_by_seed"] == (0.0071, 0.0637, -0.0269, -0.0327, 0.0041)
    assert protocol["label_sd"] == 0.6587
    # The spread is the point: the widest seed is 13x the narrowest.
    spreads = protocol["prediction_sd_by_seed"]
    assert max(spreads) / min(spreads) > 12
    assert all(s < protocol["label_sd"] for s in spreads)
    assert "13x" in protocol["seed_sd_ratio"]
    assert "may NOT be said" in record["forbidden"]
    assert "scores zero on this cohort" in record["forbidden"]
    assert "not interpretable" in record["honest_sentence"]
    assert "divergence fix holds" in record["what_it_does_establish"]
    assert record["ledger_status"].startswith("HELD UNLEDGERED")


def test_the_ledger_hold_is_reasoned_and_the_ledger_stays_clean():
    from cleft import results_ledger

    record = phase10.LEDGER_HOLD_REASONS
    assert len(record["reasons"]) == 3
    assert any("citable object" in r for r in record["reasons"])
    assert any("not VOID" in r for r in record["reasons"])
    assert "span the label range" in record["what_would_be_ledgered"]
    # The hold is real: no ledger entry carries the collapsed arms' run.
    assert not any(
        "p10-cleftgnn-4" in str(entry["run_dirs"])
        for entry in results_ledger.ENTRIES
    )


def test_the_collapse_diagnosis_locates_the_stage_and_answers_the_control():
    record = phase10.COLLAPSE_DIAGNOSED
    control = record["d_linear_head_control"]
    assert control["verdict"].startswith("DECISIVE")
    # The linear head's loss falls; the CleftGNN head's does not.
    linear = control["linear"]["loss_1_to_4"]
    assert linear[0] > linear[-1]
    cleftgnn_loss = control["cleftgnn"]["loss_1_to_4"]
    assert cleftgnn_loss[-1] > cleftgnn_loss[0]
    assert "all five" in control["linear"]["classes_used"]
    assert "ONE at a time" in control["cleftgnn"]["classes_used"]
    assert "input-independent" in control["on_pure_noise"]
    assert "WITHIN-PROBE" in control["why_the_caveats_cancel"]

    # (a) refuted as the driver by a balanced target.
    assert "perfectly balanced" in record["a_class_imbalance_not_the_driver"]
    assert "2.7342" in record["a_class_imbalance_not_the_driver"]

    # The stage table locates the collapse at the sigmoid, and shows the
    # SABM path surviving it.
    stages = record["stage_variance_ratio"]
    assert stages["appnp"] / stages["sigmoid_f_hat"] > 30
    assert stages["sabm_v_a"] > 0.5
    assert stages["gated_pooling_f_t"] < 0.05
    assert stages["logits"] < 0.1
    assert "K==1 sigmoid" in record["where_it_dies"]
    assert "notebook's own line" in record["the_sigmoid_is_theirs"]
    assert "Thursday-flagged" in record["the_sigmoid_is_theirs"]


def test_the_routes_are_proposed_with_costs_and_nothing_is_picked():
    record = phase10.COLLAPSE_ROUTES_PROPOSED
    assert record["proposed"].endswith("not picked")
    assert len(record["routes"]) == 4
    # The first route is to measure, not to change.
    assert record["routes"][0].startswith("confirm the stage table")
    assert "cost NONE" in record["routes"][0]
    assert any("HIGH" in r for r in record["routes"])
    assert any("MEDIUM" in r for r in record["routes"])
    assert any("LOW" in r for r in record["routes"])
    assert "lr 0.01 and clipping" in record["still_out"]


def test_the_pod_probe_prints_the_stage_table():
    from pathlib import Path

    script = (
        Path(__file__).resolve().parents[1] / "scripts" / "p10_scale_probe.py"
    )
    text = script.read_text(encoding="utf-8")
    assert "--stages" in text
    assert "def stage_variance(" in text
    assert "SUSPECT" in text  # the sigmoid row is flagged in the output


# --------------------------------------------------------------------------
# 2026-08-17: the stage table confirmed on real features; route 3 registered
# --------------------------------------------------------------------------


def test_the_stage_table_is_confirmed_and_names_the_second_finding():
    record = phase10.STAGE_TABLE_CONFIRMED
    ratios = record["ratios"]
    # The sigmoid kills 87x on real features -- sharper than offline's 39x.
    assert ratios["appnp"] / ratios["sigmoid_f_hat"] > 80
    assert ratios["sabm_v_a"] > 0.7
    assert ratios["fused"] < 0.15 and ratios["logits"] < 0.03
    magnitudes = record["magnitudes"]
    # Finding 2: the branch imbalance, recomputed from the magnitudes.
    assert magnitudes["f_t_mean"] / magnitudes["v_a_mean"] > 5.5
    assert "5.8x LARGER" in record["finding_2_imbalance"]
    assert "the sum, not at the sigmoid" in record["why_it_directs_the_fix"]
    # The sigmoid sits at its midpoint because APPNP's magnitude is tiny.
    assert magnitudes["appnp_mean"] < 0.05
    assert abs(magnitudes["f_hat_mean"] - 0.5) < 0.01


def test_route_3_is_registered_with_its_form_reasons_and_criteria():
    record = phase10.GNN_BRANCH_NORM
    assert "elementwise_affine=False" in record["form"]
    assert len(record["why_this_form"]) == 4
    assert any("void ladders" in r for r in record["why_this_form"])
    assert any("no learnable affine" in r for r in record["why_this_form"])
    assert any("v_a is NOT normalised" in r for r in record["why_this_form"])
    # Honest about scope: it fixes dilution, not compression.
    assert "does not fix the sigmoid" in record["what_it_does_not_do"]
    assert record["fidelity_cost"].startswith("MEDIUM")
    assert "untouched" in record["fidelity_cost"]

    # The offline measurement: magnitude equalised, ratio unchanged --
    # the signature of a dilution fix.
    offline = record["measured_offline"]
    before, after = offline["gnn_branch_magnitude"]
    assert before / after > 5
    assert offline["gnn_branch_ratio"][0] == offline["gnn_branch_ratio"][1]
    assert offline["fused_ratio"][1] > 2 * offline["fused_ratio"][0]
    assert offline["logits_ratio"][1] > 2 * offline["logits_ratio"][0]

    # Success is defined BEFORE the run, and falsifiably.
    assert len(record["success_criteria"]) == 2
    assert "0.6587" in record["success_criteria"][1]
    assert "0.4-0.7" in record["pre_registered_expectation"]
    assert "the sigmoid is" in record["pre_registered_expectation"]


def test_both_readings_of_the_normalised_arm_are_committed_now():
    record = phase10.NORMALISED_ARM_READINGS
    assert record["committed"].endswith("before the run")
    assert "ARCHITECTURE MEASUREMENT" in record["if_spans_and_pcc_near_zero"]
    assert "LEDGERABLE" in record["if_spans_and_pcc_near_zero"]
    assert "cannot produce an interpretable predictor" in (
        record["if_still_collapses"]
    )
    assert "never a performance figure" in record["if_still_collapses"]
    assert "stay unledgered" in record["either_way"]


def test_the_model_normalises_the_gnn_branch_before_the_sum():
    """Route 3 is in the model: an affine-free LayerNorm on f_t, v_a
    untouched, and the notebook's sum still a sum."""
    pytest.importorskip("torch")
    import inspect

    import torch

    from cleft.models import cleftgnn

    torch.manual_seed(0)
    backbone = cleftgnn.CleftGNNBackbone(pretrained=False, device="cpu")
    backbone.reset(np.array([1, 2, 3, 3, 4], dtype=float))
    model = backbone._model

    assert isinstance(model.gnn_norm, torch.nn.LayerNorm)
    assert model.gnn_norm.elementwise_affine is False
    assert model.gnn_norm.normalized_shape == (cleftgnn.GNN_OUT_DIM,)

    source = inspect.getsource(cleftgnn.build)
    # The GNN branch is normalised; the SABM branch is not.
    assert "self.gnn_norm(f_t) + v_a" in source
    assert "gnn_norm(v_a)" not in source
    # The notebook's sum and sigmoid are still there.
    assert ".sum(dim=1)" in source
    assert "torch.sigmoid(propagated)" in source

    # It equalises magnitude without inventing variation: a constant
    # batch stays constant after the norm.
    constant = torch.ones(4, cleftgnn.GNN_OUT_DIM) * 7.0
    normed = model.gnn_norm(constant)
    assert float(normed.std(dim=0).max()) == 0.0


# --------------------------------------------------------------------------
# 2026-08-17: the probe rebuilt the pipeline and fell behind the model
# --------------------------------------------------------------------------


def test_the_probe_defect_is_recorded_with_its_verification():
    record = phase10.PROBE_RECONSTRUCTED_THE_PIPELINE
    assert "never called gnn_norm" in record["defect"]
    assert "four decimals" in record["defect"]
    assert "two implementations of one computation" in record["defect_class"]
    assert "REMOVED, not synced" in record["fix"]
    verification = record["verification"]
    # Nine stages bitwise identical -- the diagnosis's own stages.
    assert len(verification["bitwise_identical"]) == 9
    for stage in ("appnp", "sigmoid_f_hat", "gated_pooling_f_t", "sabm_v_a"):
        assert stage in verification["bitwise_identical"]
    # Exactly three diverged, all downstream of the missing call.
    assert set(verification["diverged"]) == {
        "fused", "after_layernorm", "logits"
    }
    assert "BITWISE exactly" in record["diagnosis_stands"]
    assert record["cost"].startswith("one pod run")


def test_the_model_reports_its_own_stages_and_they_are_what_it_returns():
    """The instrument reads the model, so the model's report must BE its
    computation -- the logits it records are the logits it returns, and
    the fused row is the normalised branch plus the untouched one."""
    pytest.importorskip("torch")
    import torch

    from cleft.models import cleftgnn

    torch.manual_seed(0)
    backbone = cleftgnn.CleftGNNBackbone(pretrained=False, device="cpu")
    backbone.reset(np.array([1, 2, 3, 4, 5], dtype=float))
    model = backbone._model

    # Passing no dict changes nothing: the signature is observation-only.
    import inspect

    signature = inspect.signature(model.forward)
    assert signature.parameters["stages"].default is None

    stages = {}
    with torch.no_grad():
        for batch, _ in backbone._batches(
            np.zeros((2, 224, 224, 3), dtype=np.uint8), None, False
        ):
            returned = model(batch, stages=stages)
            break

    for name in (
        "input", "backbone_map", "region_vectors", "region_proposer",
        "gnn_mlp", "appnp", "sigmoid_f_hat", "gated_pooling_f_t",
        "sabm_v_a", "gnn_norm_f_t", "fused", "after_layernorm", "logits",
    ):
        assert name in stages, name
    # What it recorded IS what it returned.
    assert torch.equal(stages["logits"], returned)
    # And the fused row is the normalised branch plus the untouched one.
    assert torch.allclose(
        stages["fused"], stages["gnn_norm_f_t"] + stages["sabm_v_a"], atol=1e-6
    )
    # gnn_norm really shrinks the branch it normalises.
    assert (
        float(stages["gnn_norm_f_t"].abs().mean())
        < float(stages["gated_pooling_f_t"].abs().mean())
    )


def test_the_probe_no_longer_rebuilds_the_pipeline():
    from pathlib import Path

    script = (
        Path(__file__).resolve().parents[1] / "scripts" / "p10_scale_probe.py"
    )
    text = script.read_text(encoding="utf-8")
    # [2026-08-17] Bounded at gate3_decomposition, which joined between
    # stage_variance and main. It reads the model's stages too, but it
    # takes an L2 norm of one of them, so it legitimately contains
    # ``.sum(dim=1)`` and is checked on its own terms below.
    body = text[
        text.index("def stage_variance("):text.index("def gate3_decomposition(")
    ]
    # The instrument reads the model; it does not recompute the pipeline.
    assert "stages=stages" in body
    for rebuilt in ("roi_align(", "torch.sigmoid(", ".sum(dim=1)",
                    "region_proposer(", "gnn_mlp("):
        assert rebuilt not in body, f"stage_variance recomputes {rebuilt}"
    # And probe_split reads the model's stages for its scale figures too.
    split = text[text.index("def probe_split("):text.index("def stage_variance(")]
    assert "stages=stages" in split
    assert "roi_align(" not in split
    # The gate-3 decomposition reads the model as well: it takes the
    # classifier's input from the recorded stages rather than re-deriving
    # it, and never re-runs a pipeline step.
    gate3 = text[
        text.index("def gate3_decomposition("):text.index("def main(")
    ]
    assert "stages=stages" in gate3
    assert "CLASSIFIER_INPUT_STAGE[backbone.recipe]" in gate3
    for rebuilt in ("roi_align(", "region_proposer(", "gnn_mlp(",
                    "torch.sigmoid("):
        assert rebuilt not in gate3, f"gate3_decomposition recomputes {rebuilt}"


def test_the_expected_post_fix_table_is_stated_with_a_falsifier():
    record = phase10.EXPECTED_POST_FIX_TABLE
    # Upstream rows are predicted UNCHANGED -- that is the instrument check.
    upstream = record["unchanged_upstream"]
    assert upstream["sigmoid_f_hat"] == 0.0058
    assert upstream["sabm_v_a"] == 0.7794
    assert upstream["gated_pooling_f_t"] == 0.0068
    low, high = record["predicted"]["fused"]
    assert 0.4 <= low < high <= 0.7
    # The band is derived from the pod's own magnitudes, not guessed.
    assert "0.7794 x 0.7323" in record["how_the_band_was_computed"]
    assert "measured-and-insufficient" in record["falsifier"]
    assert "within noise of run 4" in record["also_a_check_on_the_instrument"]


# --------------------------------------------------------------------------
# 2026-08-17: route 3's arms -- neither registered reading applies
# --------------------------------------------------------------------------


def test_route_3_observed_applies_neither_reading_and_says_so():
    record = phase10.ROUTE_3_OBSERVED
    assert "neither spanning nor collapsed" in record["neither_reading_applies"]
    # The PCC is indistinguishable from zero: mean inside one seed sd.
    assert abs(record["pcc_mean"]) < record["pcc_seed_sd"]
    pcc = np.array(record["pcc_by_seed"])
    assert abs(float(pcc.mean()) - record["pcc_mean"]) < 5e-4
    assert abs(float(pcc.std(ddof=1)) - record["pcc_seed_sd"]) < 5e-4
    # Criterion (ii) fails, arithmetically.
    spreads = np.array(record["prediction_sd_by_seed"])
    assert spreads.max() / 0.6587 < 0.4
    assert "33.7%" in record["criterion_ii_fails"]
    # Criterion (i) is honestly marked unreported, not assumed.
    assert "UNKNOWN" in record["criterion_i_unreported"]
    assert "--stages" in record["criterion_i_unreported"]
    # Route 3 reduced seed spread but did NOT buy range -- stated as such.
    did = record["what_route_3_did"]
    assert did["mean_prediction_sd"][1] < did["mean_prediction_sd"][0]
    assert did["seed_spread_ratio"][1] < did["seed_spread_ratio"][0] / 2
    assert "not to be reported as success" in did["reading"]


def test_the_third_reading_is_labelled_post_hoc_and_held_unledgered():
    record = phase10.THIRD_READING_PROPOSED
    assert "post hoc" in record["proposed"]
    assert record["ledgered"] is False
    assert len(record["reasons"]) == 3
    assert any("post-hoc selection" in r for r in record["reasons"])
    assert any("known-incomplete" in r for r in record["reasons"])
    assert any("MISNAMED" in r for r in record["reasons"])
    assert any("SABM head, not CleftGNN" in r for r in record["reasons"])
    assert "IS ledgerable as it stands" in (
        record["what_would_make_it_ledgerable"][1]
    )


def test_the_psychologist_cell_is_recorded_with_both_its_limits():
    record = phase10.PSYCHOLOGIST_CELL_IS_THE_INSTABILITY
    figures = record["figures"]
    swing = figures["pcc_benchmark_n25"] - figures["pcc_study_n36"]
    assert abs(swing - figures["swing"]) < 1e-6
    their = record["their_rater_e"]
    assert abs((their["benchmark"] - their["study"]) - their["swing"]) < 1e-6
    # Same direction, but explicitly not claimed as a pattern.
    assert "anecdote" in record["same_direction"]
    assert "0.2-0.3 when the test set changes" in record["establishes"]
    assert "nothing about their cohort" in record["does_not_establish"]


def test_the_seed_diagnostic_needs_no_new_run_and_commits_both_readings():
    record = phase10.SEED_INSTABILITY_NEXT_DIAGNOSTIC
    assert "READ of artifacts" in record["needs_no_new_run"]
    assert "selected_epochs" in record["needs_no_new_run"]
    assert "DEVIATION is implicated" in record["if_narrow_seeds_stopped_early"]
    assert "IS the finding" in record["if_all_seeds_ran_similar"]
    assert "epoch 33" in record["existing_hint"].lower()
    # Neither branch reaches for lr or the sigmoid.
    for value in record.values():
        assert "route 2" not in str(value)


def test_the_epochs_refute_reading_one_and_support_neither():
    """The narrowest seed holds the longest fold; stopping varies far
    more within seeds than between them, so it cannot explain a
    seed-level spread."""
    record = phase10.EPOCHS_ANSWER_NEITHER_READING
    epochs = record["epochs_by_seed"]
    # Recomputed, not asserted: seed 99 is narrowest AND holds the max.
    assert max(epochs[99]) == max(max(f) for f in epochs.values()) == 26
    assert "REFUTED" not in record["reading_2_not_supported"]
    assert "the LONGEST fold" in record["reading_1_refuted"]
    structure = record["the_actual_structure"]
    assert structure["within_seed_variance"] > 10 * structure[
        "between_seed_variance_of_means"
    ]
    assert "not coupled" in structure["consequence"]
    # The count, recomputed.
    flat = [v for folds in epochs.values() for v in folds]
    assert len(flat) == 25
    assert sum(1 for v in flat if v <= 4) == 15
    assert "not 14" in record["count_corrected"]
    assert "EPOCH 33" in record["monitor_question_still_live"]


def test_the_curve_proposal_names_the_discarded_data_and_its_cost():
    record = phase10.CURVE_WRITING_PROPOSED
    assert record["proposed"].endswith("not run")
    assert "harness.py:309" in record["the_data_already_exists"]
    assert "computed 25 times, kept zero times" in (
        record["the_data_already_exists"]
    )
    assert "NO new computation" in record["cost"]
    assert "one arm run" in record["cost"]
    assert len(record["worth_it"]) == 3
    assert any("supervision" in w for w in record["worth_it"])
    assert any("regardless" in w for w in record["worth_it"])
    assert len(record["readings"]) == 2
    # It reaches for neither forbidden lever: route 2 and lr appear only
    # in the negation that rules them out, never as the remedy.
    assert "no touch to lr, the sigmoid, or route 2" in record["cost"]
    remedies = " ".join(record["readings"])
    assert "monitor" in remedies and "registered-deviation" in remedies
    assert "route 2" not in remedies and "learning rate" not in remedies


def test_criterion_one_is_recorded_as_an_open_check_not_an_assumption():
    record = phase10.CRITERION_I_UNREPORTED
    assert "never requested" in record["gap"] or "never taken" in record["gap"]
    assert "must not be assumed" in str(
        phase10.ROUTE_3_OBSERVED["criterion_i_unreported"]
    ) or "UNKNOWN" in phase10.ROUTE_3_OBSERVED["criterion_i_unreported"]
    assert "one batch of one fold at initialisation" in (
        record["why_it_does_not_transfer"]
    )
    assert record["status"].startswith("OPEN CHECK")


# --------------------------------------------------------------------------
# 2026-08-17: the arm writes its curves and its stage table
# --------------------------------------------------------------------------


def test_the_backbone_records_stages_by_true_epoch_and_costs_nothing():
    """The capture rides a forward that already happens, indexes by the
    true epoch, does not overwrite within an epoch, and is off unless
    asked -- the properties the selected-epoch lookup depends on."""
    pytest.importorskip("torch")
    import torch

    from cleft.models import cleftgnn

    torch.manual_seed(0)
    backbone = cleftgnn.CleftGNNBackbone(
        pretrained=False, device="cpu", batch_size=4, record_stages=True
    )
    grades = np.array([1, 2, 3, 4], dtype=float)
    faces = np.zeros((4, 224, 224, 3), dtype=np.uint8)
    backbone.reset(grades)

    # The harness's shape: an epoch-0 predict, then train/predict pairs.
    backbone.predict(faces)
    assert sorted(backbone.stage_ratios_by_epoch) == [0]
    backbone.train_epoch(faces, grades)
    backbone.predict(faces)
    assert sorted(backbone.stage_ratios_by_epoch) == [0, 1]

    # A second predict inside the same epoch must not overwrite.
    first = backbone.stage_ratios_by_epoch[1]["fused"]
    backbone.predict(faces)
    assert backbone.stage_ratios_by_epoch[1]["fused"] == first

    # Every stage, and the ratio triple's own arithmetic.
    table = backbone.stage_ratios_by_epoch[1]
    assert set(table) == set(cleftgnn.STAGE_ORDER)
    magnitude, across, ratio = table["fused"]
    assert abs(ratio - across / magnitude) < 1e-9

    # Off unless asked, and reset clears it.
    plain = cleftgnn.CleftGNNBackbone(pretrained=False, device="cpu")
    assert plain.record_stages is False
    backbone.reset(grades)
    assert backbone.stage_ratios_by_epoch == {}


def test_stage_ratios_live_in_the_model_and_the_probe_calls_them():
    """One implementation of the stage table -- the probe defect's lesson
    applied before it can be repeated."""
    from pathlib import Path

    from cleft.models import cleftgnn

    # [2026-08-17] 14, not 13: the notebook cell's ``acm_v`` joined
    # (NOTEBOOK_RECIPE_CELL_REGISTERED). One order still covers both
    # paths -- stage_ratios skips what a recipe did not record, so
    # neither cell can report the other's stages.
    assert len(cleftgnn.STAGE_ORDER) == 14
    assert callable(cleftgnn.stage_ratios)

    script = (
        Path(__file__).resolve().parents[1] / "scripts" / "p10_scale_probe.py"
    )
    text = script.read_text(encoding="utf-8")
    # [2026-08-17] The import narrowed to RECIPE_STAGES when the probe
    # became recipe-aware: checking against all fourteen names would have
    # made it unusable on the notebook cell, and the per-recipe list lives
    # in the model for the same reason the arithmetic does.
    assert "from cleft.models.cleftgnn import RECIPE_STAGES, stage_ratios" in (
        text
    )
    # No second copy of either list, or of the arithmetic, in the probe.
    assert 'STAGE_ORDER = (' not in text
    assert "RECIPE_STAGES = {" not in text
    assert "flat.std(dim=0).mean()" not in text


def test_the_arm_writes_curves_and_the_selected_epoch_stage_table():
    import inspect

    from cleft import run as run_module

    source = inspect.getsource(run_module.task_cleftgnn_cv)
    # The curves the harness computed are written, not discarded.
    assert 'seed_{seed}__curves.csv' in source
    assert "fold,epoch,train_loss,inner_val_mse,inner_val_pcc" in source
    assert "fold_run.curve" in source
    # The stage table comes from the fold's SELECTED epoch.
    assert 'seed_{seed}__stage_ratios.csv' in source
    assert "stage_ratios_by_epoch.get(" in source
    assert "fold_run.selected_epoch" in source
    assert "NOT_RECORDED" in source
    assert "record_stages=True" in source
    assert "fused_ratio_at_selected_epoch" in source


def test_the_curves_build_is_recorded_with_config_untouched():
    record = phase10.CURVES_AND_STAGES_WRITTEN
    assert "does not restore best weights" in record["stage_table_at_selection"]
    assert "train_epoch exactly once per epoch" in (
        record["stage_table_at_selection"]
    )
    assert "rides the first batch" in record["costs_no_extra_forward"]
    assert "does not overwrite" in record["costs_no_extra_forward"]
    assert "OFF by default" in record["verified_offline"]
    assert len(record["artifacts"]) == 3
    assert any("NOT_RECORDED" in a for a in record["artifacts"])
    assert "config_unchanged" in record
    # The proposal and the open check both point forward, dated.
    assert phase10.CURVE_WRITING_PROPOSED["built"].startswith("2026-08-17")
    assert phase10.CRITERION_I_UNREPORTED["answered_by"].startswith(
        "CURVES_AND_STAGES_WRITTEN"
    )


# --------------------------------------------------------------------------
# 2026-08-17: criterion (i) fails on the arm; the monitor is implicated
# --------------------------------------------------------------------------


def test_criterion_one_failed_and_names_sabm_not_the_sigmoid():
    record = phase10.CRITERION_I_FAILED_ON_THE_ARM
    assert record["verdict"].startswith("FAILS")
    assert record["registered_threshold"] == 0.4
    assert "did not transfer" in record["verdict"]
    # The GNN branch did not improve; SABM fell. That is the finding.
    assert "UNCHANGED and still dead" in record["gnn_branch"]
    assert "0.7794 at init" in record["sabm_branch"]
    assert "SABM DEGRADATION, not the sigmoid" in record["reading"]
    # And it is explicitly not a licence for route 2.
    assert "NOT a licence for" in record["disposition"]
    assert "measured-and-insufficient" in record["disposition"]


def test_the_monitor_finding_registers_its_threshold_before_the_numbers():
    record = phase10.MONITOR_SELECTS_AGAINST_PCC
    fold = record["fold_0_seed_1337"]
    # The correlation is recomputed here, not taken on trust.
    mse = np.array(fold["inner_val_mse"])
    pcc = np.array(fold["inner_val_pcc"])
    assert abs(float(np.corrcoef(mse, pcc)[0, 1]) - fold["r_mse_pcc"]) < 5e-4
    assert int(np.argmin(mse)) + 1 == fold["mse_selected_epoch"]
    assert int(np.argmax(pcc)) + 1 == fold["pcc_best_epoch"]
    # A positive r means minimising MSE picks poor PCC.
    assert fold["r_mse_pcc"] > 0
    assert "not varying" in record["mechanism"]
    assert "18 of 25" in record["registered_threshold"]
    assert record["status"].startswith("PENDING")


def test_the_monitor_checker_reproduces_the_known_fold():
    """The tool's arithmetic is pinned against the one fold measured by
    hand, so a later refactor cannot quietly change what it reports."""
    import importlib.util
    from pathlib import Path

    script = (
        Path(__file__).resolve().parents[1] / "scripts" / "p10_monitor_check.py"
    )
    spec = importlib.util.spec_from_file_location("p10_monitor_check", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    rows = list(zip(
        range(1, 8),
        [0.683, 0.591, 0.532, 0.594, 0.820, 0.415, 0.790],
        [-0.198, -0.209, -0.193, -0.170, -0.082, -0.181, -0.066],
    ))
    result = module.analyse_fold(rows)
    assert result["mse_selected_epoch"] == 6
    assert result["pcc_best_epoch"] == 7
    assert result["differ"] is True
    assert result["pcc_rank_of_selected"] == 4
    assert abs(result["r_mse_pcc"] - 0.7559) < 5e-4
    assert module.MAJORITY_OF_25 == 18


def test_the_synthesis_is_labelled_a_hypothesis_with_its_test():
    record = phase10.THE_TWO_FINDINGS_MAY_BE_ONE
    assert record["hypothesis"].startswith("2026-08-17")
    assert "0.436 at epoch 14" in record["the_alignment"]
    assert "not_monotonic" in record
    assert "stops the model in the trough" in record["the_synthesis"]
    assert "EVERY epoch's stage table" in record["the_test"]
    assert "no extra computation" in record["the_test"]
    assert "my choice and it" in record["my_design_error"]
    assert record["not_built"].startswith("proposed")


def test_the_monitor_arm_is_registrable_under_five_conditions():
    record = phase10.MONITOR_ARM_REGISTRABILITY
    assert record["registrable"] is True
    assert len(record["conditions"]) == 5
    assert any("SECOND cell, never a replacement" in c for c in record["conditions"])
    assert any("evaluation criterion untouched" in c for c in record["conditions"])
    assert any("NO third monitor" in c for c in record["conditions"])
    # The licensing distinction is stated explicitly.
    assert "REPORTING statistic" in record["why_it_is_not_the_forbidden_move"]
    assert "TRAINING protocol" in record["why_it_is_not_the_forbidden_move"]
    assert "CLAIMABLY_WORSE_WITHDRAWN" in record["why_it_is_not_the_forbidden_move"]
    # The conclusion is bounded, and the ledger correction waits.
    assert "scores zero on this cohort" in record["conclusion_cannot_be"]
    assert "our stopping rule" in record["conclusion_cannot_be"]
    assert "do not span the label range" in record["defensible_sentence"]
    assert "Not before the check runs" in record["ledger_consequence"]
    assert "later phase" in record["otherwise"]


# --------------------------------------------------------------------------
# 2026-08-17: the monitor is exonerated; the conclusion loses its qualifier
# --------------------------------------------------------------------------


def test_the_monitor_is_exonerated_and_the_story_is_withdrawn():
    record = phase10.MONITOR_EXONERATED
    sign = record["sign_condition"]
    assert sign["binomial_p"] == 1.000
    assert "13 of 25" in sign["positive_r"]
    assert sign["median_r"] == 0.0927
    assert sign["verdict"].startswith("indistinguishable")
    # The mismatch leg, read against its own null, points the other way.
    mismatch = record["mismatch_condition"]
    assert "EXCEEDS chance" in mismatch["reading"]
    assert "AGAINST anti-selection" in mismatch["reading"]
    assert "1/n" in mismatch["its_own_null"]
    # 24% observed against 1/n for n in 6..31 -- recomputed here.
    assert 6 / 25 > 1 / 6 or abs(6 / 25 - 1 / 6) < 0.08
    for n in (10, 15, 20, 31):
        assert 6 / 25 > 1 / n
    # The error is owned and the story explicitly withdrawn.
    assert "one fold as a mechanism" in record["my_error"]
    assert "before the numbers caught it" in record["my_error"]
    assert "no support at 25 folds" in record["withdrawn_as_unsupported"]
    assert "not to be repeated" in record["withdrawn_as_unsupported"]
    # The superseded records carry dated pointers, not edits.
    assert phase10.MONITOR_SELECTS_AGAINST_PCC["withdrawn"].startswith("2026-08-17")
    assert phase10.MONITOR_ARM_REGISTRABILITY["outcome"].startswith("NO CELL")
    assert phase10.THE_TWO_FINDINGS_MAY_BE_ONE["half_withdrawn"].startswith(
        "2026-08-17"
    )


def test_the_conclusion_is_unqualified_and_scoped():
    record = phase10.PHASE_10_CONCLUSION_UNQUALIFIED
    assert "do not span the label range" in record["sentence"]
    assert "indistinguishable from zero" in record["sentence"]
    # No stopping-rule qualifier survives.
    assert "exonerated" in record["no_stopping_caveat"]
    assert any("K==1 sigmoid" in m for m in record["mechanism"])
    assert any("0.78 to ~0.25" in m for m in record["mechanism"])
    # The scope is stated, so the sentence cannot travel further than it should.
    assert "NOT a claim about CleftGNN in" in record["scope"]


def test_what_remains_names_the_one_outstanding_criterion_and_its_prediction():
    record = phase10.WHAT_REMAINS_BEFORE_CLOSING
    status = record["criteria_status"]
    assert status["4_paired_bca"].startswith("NOT DONE")
    # The five-deviations-against-one expectation is not quietly passed.
    assert "FIVE" in status["2_deviations_enumerated"]
    assert "not a quiet pass" in status["2_deviations_enumerated"]

    bca = record["paired_bca"]
    # The margin prediction is recomputed from the project's own formula.
    threshold = 1.96 * np.sqrt(0.0148 ** 2 / 5 + 0.0676 ** 2 / 5)
    assert abs(threshold - bca["predicted_threshold"]) < 5e-4
    assert abs(0.2279 / threshold - bca["predicted_margin"]) < 0.02
    # It lands in the margin table's mixed region.
    assert 3.6 < bca["predicted_margin"] < 4.7
    assert "mixed region" in bca["where_that_lands"]
    assert "condition 1 decides" in bca["where_that_lands"]
    # Both outcomes are committed, and neither overclaims.
    assert "NOT architectural superiority" in bca["if_claimable"]
    assert "COHORT_CANNOT_RESOLVE" in bca["if_not_claimable"]
    assert "four of five raters" in record["faithful_arm_disposition"]


def test_the_u_shape_is_deferred_but_the_recording_change_is_not():
    record = phase10.U_SHAPE_BELONGS_LATER
    assert "never depended on the monitor" in record["observation_survives"]
    assert len(record["not_in_phase_10"]) == 3
    assert any("cannot change the conclusion" in r for r in record["not_in_phase_10"])
    assert any("does not otherwise need" in r for r in record["not_in_phase_10"])
    assert any("registered before that run" in r for r in record["not_in_phase_10"])
    # The cheap half stays in this phase.
    assert "RECORDING change" in record["what_belongs_here"]
    assert "no computation" in record["what_belongs_here"]
    assert "hit twice" in record["what_belongs_here"]
    assert record["proposed_not_built"].startswith("a five-line change")


# --------------------------------------------------------------------------
# 2026-08-17: the notebook and the deck, verified at source
# --------------------------------------------------------------------------


def test_the_cifar_conflation_is_confirmed_from_both_artifacts():
    record = phase10.CIFAR_CONFLATION_CONFIRMED_AT_SOURCE
    assert "90.94%" in record["deck_slide_3"]
    assert "88.40%" in record["deck_slide_3"]
    assert "CIFAR10" in record["deck_slide_3"]
    assert "NUM_CLASSES = 10" in record["notebook_backing"]
    assert "EPOCHS = 5" in record["notebook_backing"]
    assert "INFERENCE" in record["status_change"]
    assert record["thursday"] is True
    # The deck is a fourth voice on the region count.
    assert "27 crops" in record["fourth_voice_on_regions"]
    assert "belief-vs-execution" in record["fourth_voice_on_regions"]
    assert phase10.NOTEBOOK_ROI_SET_MEASURED["deck_also_says_27"].startswith(
        "2026-08-17"
    )


def test_the_notebook_recipe_is_adam_and_my_batch_claim_is_corrected():
    record = phase10.NOTEBOOK_RECIPE_IS_ADAM
    assert "optim.Adam" in record["recipe"]
    assert "LR = 0.001" in record["recipe"]
    assert "BATCH_SIZE = 16" in record["recipe"]
    assert "SGD at 0.01" in record["manuscript_says"]
    # The error is owned, with its cause.
    assert "states none" in record["my_record_was_wrong"]
    assert "right by coincidence" in record["my_record_was_wrong"]
    # The Adam explanation is a hypothesis, labelled.
    assert "NOT measured" in record["adam_hypothesis"]
    assert "29,519" in record["adam_hypothesis"]
    # The whole-read findings the first pass missed.
    found = record["whole_read_also_found"]
    assert len(found) == 6
    assert any("EPOCHS = 5" in f and "NO validation split" in f for f in found)
    assert any("CIFAR-10 normalisation" in f for f in found)
    assert any("REGION_COUNT = 27" in f for f in found)
    # And the deviations record carries the dated correction.
    # [2026-08-17] Now a tuple: a second dated correction joined it when
    # the notebook's EPOCHS = 5 raised the early-stopping deviation's
    # fidelity cost (NOTEBOOK_BUDGET_AND_NORMALISATION).
    corrections = phase10.REGISTERED_DEVIATIONS["corrected"]
    assert isinstance(corrections, tuple)
    assert all(c.startswith("2026-08-17") for c in corrections)
    assert any("DOES state BATCH_SIZE = 16" in c for c in corrections)


def test_the_multiplicative_fusion_is_recorded_with_its_stake():
    record = phase10.NOTEBOOK_FUSION_IS_MULTIPLICATIVE
    assert "f_t * (1 + v)" in record["notebook"]
    assert "torch.sigmoid" in record["v_is_bounded"]
    assert "additive" in record["manuscript"]
    assert "0.1315" in record["why_it_matters"]
    assert "0.7794" in record["why_it_matters"]
    # The estimate is labelled an estimate, and does not overclaim.
    assert "NOT a restoration" in record["estimate_not_measurement"]
    assert "not a run" in record["estimate_not_measurement"]
    assert "rehabilitate route 3" in record["does_not"]


def test_the_frozen_vit_and_parallel_acm_are_confirmed_not_corrected():
    record = phase10.NOTEBOOK_BACKBONE_AND_ACM_CONFIRMED
    assert "requires_grad = False" in record["frozen_twice"]
    assert "torch.no_grad()" in record["frozen_twice"]
    assert "ViT-B/16" in record["identity_still_differs"]
    assert "ResNet-50" in record["identity_still_differs"]
    assert "CONFIRMED, not corrected" in record["acm_is_parallel"]
    # Our model really does read the pre-GNN features, as claimed.
    import inspect

    from cleft.models import cleftgnn

    source = inspect.getsource(cleftgnn.build)
    assert "self.acm_q(regions)" in source
    assert "self.acm_q(f_hat)" not in source


def test_three_descriptions_of_the_attention_are_recorded_as_distinct():
    record = phase10.THREE_DESCRIPTIONS_OF_THE_ATTENTION
    assert "WEIGHTING" in record["manuscript"]
    assert "SCOPING" in record["deck"]
    assert "NO SABM" in record["notebook"]
    assert "5 (bottom half), 10 (bottom third)" in record["deck"]
    assert "not three phrasings" in str(record["not_three_phrasings"]).lower() or (
        "weighting keeps every region" in record["not_three_phrasings"]
    )
    assert "question for supervision" in record["not_derivable"]
    # Slide 9's cleft comparison is flagged as obtainable and wanted.
    assert "CLEFT DATASET" in record["slide_9_has_cleft_results"]
    assert "Worth asking for" in record["slide_9_has_cleft_results"]


def test_the_iem_direction_ambiguity_is_flagged_before_phase_11_builds():
    record = phase10.IEM_IS_THEIRS_DIRECTION_AMBIGUOUS
    assert "all three methods" in record["confirms"]
    assert "UNDERprediction" in record["the_ambiguity"]
    assert "HIGHER score" in record["the_ambiguity"]
    assert "y_hat - G < 0" in record["the_manuscript_is_unambiguous"]
    assert "backwards" in record["why_it_must_be_settled_first"]
    # The Phase 11 carry record still holds the equation itself.
    assert phase10.IEM_CARRIED_FOR_PHASE_11["bounds"] == (0, 4)


def test_the_notebook_cell_is_proposed_with_readings_and_scope():
    record = phase10.NOTEBOOK_RECIPE_CELL_PROPOSED
    assert record["proposed"].endswith("not picked")
    assert "Adam lr 0.001" in record["the_cell"]
    assert "multiplicative fusion" in record["the_cell"]
    assert len(record["for"]) == 4
    assert len(record["against"]) == 2
    # The shape-comparison to the failed monitor arm is made explicitly,
    # and offered for testing rather than assertion.
    assert "MONITOR_ARM_REGISTRABILITY" in record["against"][1]
    assert "fidelity question, not a tuning question" in (
        record["why_the_shape_differs"]
    )
    assert "rather than accept it" in record["why_the_shape_differs"]
    # Three readings, committed before any run.
    assert len(record["readings_committed"]) == 3
    assert any("RECIPE-INDEPENDENT" in r for r in record["readings_committed"])
    assert any("LEDGERABLE" in r for r in record["readings_committed"])
    # Conditions mirror the monitor arm's, and the scope caveat travels.
    assert len(record["conditions"]) == 4
    assert any("SECOND cell" in c for c in record["conditions"])
    assert "CIFAR-10, not cleft" in record["scope_caveat"]
    assert "never seen" in record["scope_caveat"]


# --------------------------------------------------------------------------
# 2026-08-17, the notebook-recipe cell: approved, registered, built
# --------------------------------------------------------------------------


def test_the_notebook_cell_registers_four_outcomes_including_divergence():
    """The approved registration. The fourth outcome is the one that
    matters procedurally: divergence is named BEFORE the run and bound to
    'recorded, not repaired', so the monitor move is closed off in
    advance rather than argued about afterwards."""
    record = phase10.NOTEBOOK_RECIPE_CELL_REGISTERED
    assert record["registered"].startswith("2026-08-17")
    assert "NOT launched" in record["registered"]
    assert record["config"] == "configs/p10_cleftgnn_notebook.yaml"
    assert "p10_cleftgnn.yaml" in record["manuscript_cell_untouched"]
    assert "stand exactly as they were" in record["manuscript_cell_untouched"]

    # Exactly four differences, each naming the notebook AND what the
    # manuscript says instead.
    differences = record["four_differences"]
    assert len(differences) == 4
    assert any("Adam lr 0.001" in d and "SGD 0.01" in d for d in differences)
    assert any("f_t + f_t*v" in d and "additive" in d for d in differences)
    assert any("ViT-B/16" in d and "ResNet-50" in d for d in differences)
    assert any("acm_w_beta" in d and "m_vertical" in d for d in differences)
    assert "both artifacts state it" in record["batch_16_is_not_a_difference"]

    # Neither LayerNorm, and the reason is readability of the result.
    assert "normalises nowhere" in record["neither_layernorm"]
    assert "indistinguishable" in record["neither_layernorm"]
    # The cell is harder, and says so.
    assert "LARGER" in record["the_cell_is_harder_not_easier"]

    outcomes = record["outcomes_committed"]
    assert len(outcomes) == 4
    assert any("RECIPE-INDEPENDENT" in o for o in outcomes)
    assert any("LEDGERABLE" in o for o in outcomes)
    diverges = [o for o in outcomes if o.startswith("DIVERGES")]
    assert len(diverges) == 1
    assert "RECORDED, NOT REPAIRED" in diverges[0]
    assert "MONITOR_ARM_REGISTRABILITY" in diverges[0]

    assert len(record["conditions"]) == 5
    assert any("NO third recipe" in c for c in record["conditions"])
    assert "CIFAR-10" in record["scope_caveat"]
    assert "never seen" in record["scope_caveat"]
    assert "believing it might collapse" in record["why_not_the_monitor_move"]

    # The proposal record stays a proposal, with a dated pointer forward.
    assert "NOTEBOOK_RECIPE_CELL_REGISTERED" in (
        phase10.NOTEBOOK_RECIPE_CELL_PROPOSED["approved"]
    )


def test_the_notebook_cell_enumerates_its_own_three_deviations():
    """Its deviations are against the NOTEBOOK, not the manuscript -- and
    the one it shares with the manuscript cell is named as shared,
    because that is what keeps the two comparable."""
    record = phase10.NOTEBOOK_CELL_DEVIATIONS
    assert record["expected_count"] == 3
    assert len(record["deviations"]) == 3
    early, normalisation, init = record["deviations"]

    assert "EPOCHS = 5" in early
    assert "NO validation split" in early
    assert "50,000 CIFAR-10 images" in early
    assert "THE ONE DEVIATION THIS CELL ALSO CARRIES" in early
    # The fidelity cost against the notebook is stated as real, not low.
    assert "REAL" in early

    assert "CIFAR-10 statistics" in normalisation
    assert "DEFECT in the notebook rather than a recipe choice" in normalisation
    assert "Fidelity cost: NONE" in normalisation

    # The init is declared as a deviation rather than defended as neutral.
    assert "held CONSTANT across both" in init
    assert "not fidelity to the notebook" in init
    assert "it does change the trajectory" in init

    # The criterion itself is not a deviation -- it is the point.
    assert "the point of the comparison" in record["not_deviations"]
    # And this cell has FEWER against its own artifact than five.
    assert "fewer" in record["against_the_manuscript_cells_five"].lower()
    assert (
        record["expected_count"]
        < phase10.REGISTERED_DEVIATIONS["expected_count"]
    )


def test_the_region_count_is_reframed_as_belief_versus_execution():
    """Four voices, three of them the group's, and the reframing is what
    makes it a finding rather than a typo: no result in the paper was
    produced by the architecture the paper reports."""
    record = phase10.REGION_COUNT_BELIEF_VS_EXECUTION
    assert record["finding"] == "2026-08-17"
    voices = record["four_voices"]
    assert len(voices) == 4
    assert sum("27" in v for v in voices) == 3
    assert any("36" in v and "self-adjusts" in v for v in voices)

    assert "not paper-vs-code" in record["reframing"]
    assert "BELIEF-vs-EXECUTION" in record["reframing"]
    assert "has never been run" in record["reframing"]

    # Phrased for authors revising a manuscript: two options, and only
    # they can choose.
    revision = record["for_the_revision"]
    assert "a correction" in revision
    assert "a different paper" in revision
    assert "Only the authors can say which was intended" in revision
    assert record["thursday"] is True


def test_the_iem_direction_is_recorded_as_blocking_phase_eleven():
    """A block, not a caveat -- and the reason is that the wrong choice
    leaves every number plausible, which is the failure mode with no
    internal detector."""
    record = phase10.PHASE_11_BLOCKED_ON_IEM_DIRECTION
    assert record["blocking"] == "2026-08-17"
    assert "penalise-optimism metric" in record["blocks"]
    why = record["why_blocking_and_not_a_caveat"]
    assert "a block prevents the number existing" in why
    assert "EVERY number plausible" in why
    assert "no internal detector" in why
    # It is unblocked by one sentence, and the sentence is spelled out.
    assert "one sentence from supervision" in record["what_unblocks_it"]
    assert "worked example" in record["what_unblocks_it"]
    # The block is narrow.
    assert "everything else in Phase 11" in record["not_blocked"]
    # And it points at the full reading rather than restating it.
    assert record["detail"].startswith("IEM_IS_THEIRS_DIRECTION_AMBIGUOUS")


def test_slide_nine_numbers_are_requested_and_nothing_is_claimed_from_them():
    record = phase10.SLIDE_9_CLEFT_NUMBERS_REQUESTED
    assert record["request"] == "2026-08-17"
    assert "CLEFT DATASET" in record["what_the_slide_shows"]
    assert "tables are images" in record["we_do_not_have_them"]
    # The argument for asking is comparative, and cites the alternatives.
    why = record["why_worth_asking"]
    assert "0.598" in why and "n=25" in why and "0.283" in why
    assert "like-for-like comparator" in why
    # A precise ask, including the crop sets that are not derivable.
    assert len(record["ask_precisely"]) == 5
    assert any("interval" in a for a in record["ask_precisely"])
    assert any("coordinates" in a for a in record["ask_precisely"])
    assert record["not_a_finding"].startswith("nothing is claimed")


def test_the_notebook_budget_and_normalisation_are_artifact_facts():
    """Both recorded as what the artifact says. The consequence recorded
    is the one that makes OUR position worse, and the deviations record
    is corrected in place rather than left standing."""
    record = phase10.NOTEBOOK_BUDGET_AND_NORMALISATION
    assert "EPOCHS = 5" in record["budget"]
    assert "NO validation split" in record["budget"]
    assert "last epoch IS its model" in record["budget"]

    consequence = record["consequence_for_our_record"]
    assert "MANUSCRIPT" in consequence and "remains exactly true" in consequence
    assert "higher than REGISTERED_DEVIATIONS recorded" in consequence

    assert "0.4914" in record["normalisation"]
    assert "ImageNet-pretrained" in record["normalisation"]
    # Calling it a defect is a judgement, and it is owned as one.
    judgement = record["why_we_call_it_a_defect"]
    assert "A judgement we own" in judgement
    assert "frozen weights cannot adapt" in judgement

    # The correction actually landed on the deviations record.
    corrected = phase10.REGISTERED_DEVIATIONS["corrected"]
    assert len(corrected) == 2
    assert any("fidelity cost is HIGHER" in c for c in corrected)


def test_the_closing_list_gains_the_notebook_cell_openly():
    """A closing criterion that grows silently is how a phase closes on
    what it happened to finish."""
    amendment = phase10.WHAT_REMAINS_BEFORE_CLOSING["amended_2026_08_17"]
    assert "NOTEBOOK_RECIPE_CELL_REGISTERED" in amendment["new_item"]
    assert "not launched" in amendment["new_item"]
    assert "ARCHITECTURE" in amendment["why_it_belongs_in_this_phase"]
    # The manuscript cell's conclusion and its outstanding criterion are
    # explicitly unaffected.
    assert "unqualified conclusion" in amendment["does_not_change"]
    assert "MANUSCRIPT cell's against 0.2520" in amendment["does_not_change"]


def test_the_notebook_config_is_a_second_cell_beside_an_untouched_first():
    """The config differs from the manuscript cell's in exactly the keys
    the registration names, carries the same three verified hashes, and
    leaves p10_cleftgnn.yaml alone."""
    from pathlib import Path

    import yaml

    repo = Path(__file__).resolve().parents[1]
    manuscript = yaml.safe_load(
        (repo / "configs" / "p10_cleftgnn.yaml").read_text(encoding="utf-8")
    )
    notebook_path = repo / "configs" / "p10_cleftgnn_notebook.yaml"
    notebook = yaml.safe_load(notebook_path.read_text(encoding="utf-8"))

    # The first cell is untouched: still SGD, still lr 0.01, still no
    # recipe key at all.
    assert manuscript["task"]["learning_rate"] == 0.01
    assert manuscript["task"]["momentum"] == 0.0
    assert "recipe" not in manuscript["task"]
    assert "optimizer" not in manuscript["task"]

    # The second cell is the notebook's recipe.
    task = notebook["task"]
    assert task["kind"] == "cleftgnn_cv"
    assert task["recipe"] == "notebook"
    assert task["optimizer"] == "adam"
    assert task["learning_rate"] == 0.001
    assert task["batch_size"] == 16
    # Adam has no momentum, so the config does not claim one.
    assert "momentum" not in task

    # Same inputs, same hashes -- nothing new was declared.
    assert notebook["inputs"] == manuscript["inputs"]
    for entry in notebook["inputs"]:
        rollup = entry["rollup_sha256"]
        assert len(rollup) == 64 and set(rollup) != {"0"}

    # The evaluation criterion is identical key for key.
    for key in (
        "geometry", "label", "seeds", "inner_val_frac", "max_epochs",
        "patience", "monitor", "manifest_artifact", "staged_artifact",
        "scoresheet_artifact",
    ):
        assert task[key] == manuscript["task"][key], key

    # And the header carries the scope caveat and the divergence rule.
    header = notebook_path.read_text(encoding="utf-8")
    assert "IS UNTOUCHED" in header
    assert "RECORDED, NOT REPAIRED" in header
    assert "CIFAR-10" in header


def test_the_generator_writes_the_notebook_config_and_does_not_drift():
    import importlib.util
    from pathlib import Path

    repo = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "generate_phase10_configs",
        repo / "scripts" / "generate_phase10_configs.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert "p10_cleftgnn_notebook.yaml" in module.CONFIGS
    for name, builder in module.CONFIGS.items():
        shipped = (repo / "configs" / name).read_text(encoding="utf-8")
        assert builder() == shipped, name


def test_the_task_defaults_to_the_manuscript_recipe():
    """The notebook cell arrives without touching the arm that ran: the
    task reads both new keys with the manuscript's values as defaults, so
    p10_cleftgnn.yaml needs no edit and its results still describe it."""
    import inspect

    from cleft import run

    source = inspect.getsource(run.task_cleftgnn_cv)
    assert 'task.get("recipe", "manuscript")' in source
    assert 'task.get("optimizer", "sgd")' in source
    assert 'task.get("momentum", 0.0)' in source
    # The recipe reaches the model, and an unknown one is refused early.
    assert "recipe=recipe" in source
    assert "optimizer=optimizer" in source
    assert "cleftgnn.RECIPES" in source
    # Everything after the model is shared: the CSV writers, the readings
    # and the label resolution are not duplicated per recipe.
    assert source.count("write_predictions(") == 2
    # [2026-08-17] The label resolution moved to run.median_by_patient
    # when Phase 11's pre-step needed the same grade; the point of the
    # assertion is unchanged -- ONE resolution, called once.
    assert source.count("median_by_patient(") == 1
    assert source.count("load_median(") == 0


def test_the_two_recipes_are_one_implementation_of_what_they_share():
    """The probe lesson, applied to the second cell before it can be
    relearned: the notebook path is a branch inside the model, not a copy
    of it, so the shared stages cannot drift apart."""
    import inspect

    from cleft.models import cleftgnn

    assert cleftgnn.RECIPES == ("manuscript", "notebook")
    source = inspect.getsource(cleftgnn.build)

    # The notebook's literal fusion, kept literal.
    assert "f_t + f_t * v" in source
    # And the manuscript's additive eq (9) still there, unchanged.
    assert "self.gnn_norm(f_t)) + v_a" in source

    # Shared machinery appears ONCE, not once per recipe.
    for shared in (
        "roi_align(", "self.region_proposer(vectors)",
        "self.gnn_mlp(regions)", "torch.sigmoid(propagated)",
        "self.acm_q(regions)", "queries @ keys.transpose(1, 2)",
    ):
        assert source.count(shared) == 1, shared

    # The notebook's attention pools over regions and closes with sigmoid.
    assert "self.acm_w_beta(f_att)" in source
    assert "torch.sigmoid((f_att * w_r).sum(dim=1))" in source
    # The notebook's ViT path is the notebook's, including its omission
    # of the final norm.
    assert "self.model.patch_embed(x)" in source
    assert "self.model.cls_token.expand" in source
    assert "self.model.pos_embed" in source
    assert "self.model.blocks(embeddings)" in source
    assert "self.model.norm(" not in source
    # Stage order covers both paths; neither claims the other's stages.
    assert "acm_v" in cleftgnn.STAGE_ORDER
    assert "sabm_v_a" in cleftgnn.STAGE_ORDER


def test_the_notebook_model_is_theirs_and_the_manuscript_model_is_unchanged():
    """Structural, on real modules: the notebook path has their attention
    head and NEITHER LayerNorm; the manuscript path is exactly what it
    was."""
    pytest.importorskip("torch")
    import torch

    from cleft.models import cleftgnn

    torch.manual_seed(0)
    manuscript = cleftgnn.build(pretrained=False, recipe="manuscript")
    assert hasattr(manuscript, "m_vertical")
    assert isinstance(manuscript.gnn_norm, torch.nn.LayerNorm)
    assert isinstance(manuscript.fused_norm, torch.nn.LayerNorm)
    assert not hasattr(manuscript, "acm_w_beta")

    torch.manual_seed(0)
    notebook = cleftgnn.build(pretrained=False, recipe="notebook")
    # Their attention head, and no vertical mask anywhere on this path.
    assert isinstance(notebook.acm_w_beta, torch.nn.Linear)
    assert notebook.acm_w_beta.out_features == 1
    assert not hasattr(notebook, "m_vertical")
    # NEITHER LayerNorm -- the notebook normalises nowhere.
    assert not hasattr(notebook, "gnn_norm")
    assert not hasattr(notebook, "fused_norm")
    # A frozen ViT-B/16, and the data config resolves from the timm model
    # itself rather than from the wrapper.
    assert notebook.backbone.output_channels == 768
    assert notebook.backbone.patch_resolution == 14
    assert all(not p.requires_grad for p in notebook.backbone.parameters())
    assert type(notebook.timm_backbone()).__name__ == "VisionTransformer"
    assert notebook.timm_backbone() is notebook.backbone.model
    assert manuscript.timm_backbone() is manuscript.backbone

    # A third recipe is refused: it would need a third artifact.
    with pytest.raises(cleftgnn.CleftGNNError, match="third documented"):
        cleftgnn.build(pretrained=False, recipe="srgnn")


def test_adam_refuses_a_momentum_it_could_not_honour():
    """A config that declares Adam AND a momentum is claiming something
    the run cannot deliver; silently ignoring it is how a recipe drifts
    from its record."""
    pytest.importorskip("torch")
    import torch

    from cleft.models import cleftgnn

    labels = np.array([1, 2, 3, 4, 5], dtype=float)
    with pytest.raises(cleftgnn.CleftGNNError, match="no momentum knob"):
        cleftgnn.CleftGNNBackbone(
            pretrained=False, device="cpu", optimizer="adam", momentum=0.9,
        ).reset(labels)
    with pytest.raises(cleftgnn.CleftGNNError, match="neither the manuscript"):
        cleftgnn.CleftGNNBackbone(
            pretrained=False, device="cpu", optimizer="rmsprop",
        ).reset(labels)

    # And the honest pair builds the optimiser the artifact names.
    notebook = cleftgnn.CleftGNNBackbone(
        pretrained=False, device="cpu", recipe="notebook",
        optimizer="adam", learning_rate=0.001,
    )
    notebook.reset(labels)
    assert isinstance(notebook._optimizer, torch.optim.Adam)
    assert notebook._optimizer.param_groups[0]["lr"] == 0.001
    assert notebook.parameter_report["recipe"] == "notebook"
    assert notebook.parameter_report["optimizer"] == "adam"
    # The ViT is frozen and only the head trains.
    assert notebook.parameter_report["frozen_backbone_parameters"] > 80_000_000
    assert notebook.parameter_report["trainable_parameters"] < 10_000_000


# --------------------------------------------------------------------------
# 2026-08-17, gate 3 refuses the notebook cell at fold 0
# --------------------------------------------------------------------------


def test_the_gate_3_finding_confirms_the_cause_and_corrects_the_character():
    """The maintainer's reading was right about the mechanism and wrong
    about its character, and the record says both."""
    record = phase10.GATE_3_FAILED_ON_THE_NOTEBOOK_CELL
    assert record["finding"] == "2026-08-17"
    assert "2.0030" in record["failure"] and "2.7566" in record["failure"]
    assert "0.95 SD" in record["failure"]
    assert "nothing citable" in record["failure"]

    # Confirmed: the departure is W . f and nothing else.
    confirmed = record["reading_confirmed"]
    assert "(N*mean + 15)/(N + 5)" in confirmed
    assert "2.7371" in confirmed and "0.009 SD" in confirmed
    assert "0.000000" in confirmed

    # Corrected: sign-random across seeds, not a drag.
    corrected = record["reading_corrected"]
    assert corrected.startswith("NOT a systematic drag")
    assert "both directions" in corrected
    assert "one draw, not a tendency" in corrected

    measured = record["magnitudes_measured"]
    # The manuscript path's rms is exact, not measured-and-rounded.
    assert measured["manuscript"]["rms"] == 1.0
    assert "fused_norm" in measured["manuscript_rms_is_exact"]
    # And the notebook path is an order larger, with the ratio named.
    assert min(measured["notebook"]["norm2"]) > 7 * measured[
        "manuscript"
    ]["norm2"]
    assert "EIGHTFOLD" in measured["analytic"]
    assert "0.01804" in measured["analytic"]

    # The four registered outcomes are not quietly re-used for this.
    stands = record["the_four_outcomes_stand"]
    assert "NOT one of them" in stands
    assert "never reached epoch 1" in stands
    assert "reading a gate failure as a result" in stands

    # The offline provenance still travels with THIS table's numbers --
    # and [2026-08-17] it now ends in the pod confirmation rather than in
    # an invitation to run one.
    assert "OFFLINE" in record["provenance"]
    assert "stand-in for roi_align" in record["provenance"]
    assert "CONFIRMED on the pod" in record["provenance"]


def test_the_gate_3_failure_is_a_per_seed_lottery_not_a_per_fold_accident():
    """The decisive measurement: within a seed the epoch-0 mean barely
    moves between folds, and between seeds it spans the grade range."""
    record = phase10.GATE_3_FAILED_ON_THE_NOTEBOOK_CELL[
        "fixed_per_seed_not_per_fold"
    ]
    assert "SAME W" in record["why"]
    low, high = record["spread_across_five_folds_of_a_seed"]
    assert high < 0.06

    per_seed = record["per_seed"]
    assert len(per_seed) == 5
    # Both directions, and across nearly the whole 1..5 range.
    assert "3.737" in per_seed[1337] and "FAIL on all five" in per_seed[1337]
    assert "1.015" in per_seed[99] and "FAIL on all five" in per_seed[99]
    assert "pass on all five" in per_seed[2024]
    # A seed does not fail one fold; it fails all of them.
    assert sum("all five" in v for v in per_seed.values()) == 4

    reading = record["reading"]
    assert "function of the random SEED" in reading
    assert "not" in reading and "training fold's labels" in reading
    assert "five draws rather than twenty-five" in reading


def test_the_gate_3_coverage_limit_is_recorded_and_the_gate_is_untouched():
    """A limitation of the guard is recorded as a limitation. The guard
    itself keeps its tolerances -- route (d) is closed."""
    record = phase10.GATE_3_FAILED_ON_THE_NOTEBOOK_CELL
    limit = record["gate_3_coverage_limit"]
    assert "99.94%" in limit and "2.9995" in limit
    assert "PASSES both checks" in limit
    assert "NOT AS A REASON TO WIDEN THE TOLERANCE" in limit
    # The gate was built against a different defect, and that is said.
    assert "BatchNorm running statistics" in limit

    # The frozen apparatus still carries the tolerances it always had.
    from cleft.train import harness

    config = harness.TrainConfig(seed=1337)
    assert config.epoch0_mean_tol == 0.5
    assert config.epoch0_mse_tol == 0.6


def test_the_gate_3_routes_pick_nothing_and_keep_d_closed():
    record = phase10.GATE_3_ROUTES_PROPOSED
    assert record["proposed"].endswith("nothing picked")
    routes = record["routes"]

    # (a) and (b) each carry a for AND an against.
    assert set(routes["a_record_as_outcome"]) == {"what", "for", "against"}
    assert "answers none of the four" in routes["a_record_as_outcome"][
        "against"
    ]
    # (b)'s cost is measured, not hypothesised.
    against_b = routes["b_zero_the_classifier_weight"]["against_measured"]
    assert "REINTRODUCES the artifact FIX (b) removed" in against_b
    assert "sd 0.000000" in against_b
    assert "SD_ZERO_HAS_TWO_CAUSES" in against_b
    assert "asymmetric" in against_b

    # The additions the maintainer did not name.
    assert "not make the gate-3 failure untrue" in routes["c_i_do_a_regardless"]
    assert "NON-constant" in routes["c_ii_scale_matched_init"]["for"]
    assert "more OURS than (b)" in routes["c_ii_scale_matched_init"]["against"]

    # Two options are named in order to be rejected, with reasons.
    assert "hides exactly what the gate exists to detect" in routes[
        "c_iii_rejected_zero_for_the_gate_only"
    ]
    assert "ruled out by the cell's OWN registration" in routes[
        "c_iv_rejected_give_it_fused_norm"
    ]

    # (d) is closed, and recorded as closed rather than omitted.
    assert routes["d_closed_relax_gate_3"].startswith("CLOSED by the maintainer")
    assert "every time" in routes["d_closed_relax_gate_3"]

    assert "four registered outcomes stand" in record["not_in_doubt"]


def test_the_zero_weight_control_is_a_closed_form_of_the_folds_own_labels():
    """Computed, not quoted: with W = 0 the head predicts softmax(bias),
    the bias IS the Laplace frequency vector, and the expected grade is
    (N*mean + 15)/(N + 5). That is why it lands on the fold mean and why
    it is the control that isolates W . f."""
    # The measured Median distribution on the 237, taken to a ~190 fold.
    counts = np.array([4, 71, 88, 24, 2], dtype=float)
    labels = np.concatenate(
        [np.full(int(n), g + 1, dtype=float) for g, n in enumerate(counts)]
    )
    total = counts.sum()

    laplace = (counts + 1.0) / (total + 5.0)
    assert np.isclose(laplace.sum(), 1.0)  # softmax(log p) == p already
    predicted = float(laplace @ np.arange(1, 6))
    closed_form = (total * labels.mean() + 15.0) / (total + 5.0)
    assert np.isclose(predicted, closed_form)
    assert np.isclose(predicted, 2.7371, atol=5e-5)

    # And it clears gate 3's head-position limit with room to spare.
    offset = abs(predicted - float(labels.mean())) / float(labels.std())
    assert offset < 0.01
    assert offset <= 0.5


def test_the_recipe_stage_sets_live_in_the_model_and_the_probe_reads_them():
    """A recipe-aware instrument must not rebuild the answer to 'which
    stages does this path have' -- that is the probe defect in another
    hat."""
    from pathlib import Path

    from cleft.models import cleftgnn

    manuscript = cleftgnn.RECIPE_STAGES["manuscript"]
    notebook = cleftgnn.RECIPE_STAGES["notebook"]
    # Each is a subset of the one order, in that order.
    for stages in (manuscript, notebook):
        assert set(stages) <= set(cleftgnn.STAGE_ORDER)
        assert list(stages) == [
            name for name in cleftgnn.STAGE_ORDER if name in stages
        ]
    # The exclusives are exactly the four the recipes disagree about.
    assert set(manuscript) - set(notebook) == {
        "sabm_v_a", "gnn_norm_f_t", "after_layernorm",
    }
    assert set(notebook) - set(manuscript) == {"acm_v"}
    # What the classifier receives is named per recipe, and is a stage
    # that recipe actually records.
    for recipe, stage in cleftgnn.CLASSIFIER_INPUT_STAGE.items():
        assert stage in cleftgnn.RECIPE_STAGES[recipe]

    probe = (
        Path(__file__).resolve().parents[1] / "scripts" / "p10_scale_probe.py"
    )
    text = probe.read_text(encoding="utf-8")
    # The probe reads the model's lists rather than carrying its own.
    assert "RECIPE_STAGES[backbone.recipe]" in text
    assert "CLASSIFIER_INPUT_STAGE" in text
    # It measures whichever cell the config names.
    assert 'task.get("recipe", "manuscript")' in text
    assert "recipe=recipe" in text
    # And the gate-3 decomposition includes the zeroed-weight control.
    assert "--gate3" in text
    assert "classifier.weight.zero_()" in text


def test_the_notebook_cell_yields_a_finding_and_never_a_registered_outcome():
    """Route (c-i), taken unconditionally: the finding is banked with its
    mechanism, its closed-form control and its provenance -- and it is
    fenced off from the four outcomes it did not reach."""
    record = phase10.NOTEBOOK_CELL_CANNOT_START_CALIBRATED
    assert record["finding"].startswith("2026-08-17")
    assert "route (c-i)" in record["finding"]
    assert "FINDING, not a number" in record["statement"]

    # The mechanism carries the measured magnitudes and the comparison.
    mechanism = record["mechanism"]
    assert "7.72-8.40" in mechanism and "exactly 1.0" in mechanism
    assert "2.43-5.39" in mechanism and "~3.3" in mechanism
    assert "The bias loses" in mechanism

    # Epoch 0 tracks the seed, not the fold. [2026-08-17] Both passes are
    # kept: the offline figures as what was predicted before the pod ran,
    # and the real-feature ones that confirmed it.
    seeded = record["epoch_0_is_a_seed_function"]
    assert seeded["across_seeds_offline"] == (1.026, 3.774)
    assert seeded["across_seeds_real_features"] == (2.0015, 4.0099)
    assert seeded["within_a_seeds_five_folds"][1] < 0.06
    assert "function of the SEED" in seeded["reading"]
    assert "did not survive real features" in (
        seeded["and_on_real_features_none_passes"]
    )

    # The control is a closed form, stated with why it holds.
    control = record["closed_form_control"]
    assert control["identity"] == "(N * mean + 15) / (N + 5)"
    assert "Laplace frequency vector" in control["why"]
    assert control["value"] == 2.7371
    assert control["sd_out"] < 0.01
    assert "0.000000" in control["verified"]

    # [2026-08-17] The provenance line now LEADS with the pod
    # confirmation: real pretrained features kept every structural claim
    # and made the magnitudes larger.
    provenance = record["provenance"]
    assert provenance.startswith("CONFIRMED ON REAL")
    assert "all five seeds failing at 0.94-1.67 SD" in provenance
    assert "2.7333 on every seed" in provenance
    assert "larger, not smaller" in provenance

    # And the fence around the four registered outcomes.
    fence = record["none_of_the_four_outcomes"]
    assert fence.startswith("FORBIDDEN")
    assert "before a single optimiser step" in fence
    assert "masquerade as a measurement" in fence
    # The four outcomes themselves are untouched by any of this.
    outcomes = phase10.NOTEBOOK_RECIPE_CELL_REGISTERED["outcomes_committed"]
    assert len(outcomes) == 4
    assert any(o.startswith("DIVERGES") for o in outcomes)


def test_the_notebook_cells_disposition_is_closed_against_preference():
    """Registered as a closure so a later turn cannot reopen it by
    wanting a different answer."""
    record = phase10.NOTEBOOK_CELL_DISPOSITION_CLOSED
    assert record["closed"].endswith("maintainer decision")
    assert "FINDING, not a number" in record["disposition"]

    # Every number-producing route is answered, each on a stated ground.
    answered = record["routes_answered_no"]
    assert len(answered) == 5
    assert "sd 0.000000" in answered["b_zero_the_classifier_weight"]
    assert "construction-nan" in answered["b_zero_the_classifier_weight"]
    assert "more OURS than (b)" in answered["c_ii_scale_matched_init"]
    assert "hides what the gate exists to detect" in (
        answered["c_iii_zero_for_the_gate_only"]
    )
    assert "own registration" in answered["c_iv_give_it_fused_norm"]
    assert "closed before" in answered["d_relax_gate_3"]

    # The bar for reopening is measured, not felt.
    bar = record["bar_for_reopening"]
    assert "NEW MEASURED REASON" in bar
    assert "not a reason" in bar
    assert "new documented artifact" in bar


def test_gate_3s_coverage_limit_is_standing_and_the_gate_is_unchanged():
    """A limitation of a guard, recorded for whoever next leans on it --
    and the guard's tolerances are deliberately left alone, because the
    failure passes inside the band."""
    record = phase10.GATE_3_COVERAGE_LIMIT
    assert record["standing_finding"] == "2026-08-17"
    assert "calibrated start" in record["for"]

    # [2026-08-17] Renamed when the pod pass showed this instance does not
    # occur on real features; the row is relabelled, not deleted.
    measurement = record["measurement_offline"]
    assert "99.94%" in measurement and "2.9995" in measurement
    assert "PASSES both checks" in measurement
    assert "BECAUSE the predictions are constant" in measurement

    # The gate is defended, not blamed. [2026-08-17] "three seeds in
    # five" was the offline pass; on real features it caught every seed,
    # which is a stronger defence of the gate and a weaker exemplar of
    # its limit.
    assert "void ladders" in record["the_gate_is_not_wrong"]
    assert "EVERY real-feature seed" in record["the_gate_is_not_wrong"]

    # The operational statement: position is not dependence.
    meaning = record["what_it_means"]
    assert "POSITIONED near the fold mean" in meaning
    assert "NOT evidence that the head DEPENDS on its input" in meaning
    assert "grade 3" in meaning

    # Untouched, and the reason is that widening would not help.
    untouched = record["tolerances_untouched"]
    assert "passes INSIDE the band" in untouched
    assert "RECORDED, NOT FIXED" in untouched

    from cleft.train import harness

    config = harness.TrainConfig(seed=1337)
    assert (config.epoch0_mean_tol, config.epoch0_mse_tol) == (0.5, 0.6)


def test_the_exit_walk_covers_six_criteria_and_names_the_one_run_left():
    record = phase10.PHASE_10_EXIT_WALK
    assert record["walked"] == "2026-08-17"
    criteria = record["criteria"]
    assert len(criteria) == 6
    # [2026-08-17, same day] Five when the walk was made, six once
    # criterion 4 ran. The walk is the snapshot; PHASE_10_CLOSING is the
    # settled statement.
    assert sum(v.startswith("DONE") for v in criteria.values()) == 6
    # [2026-08-17, same day] Criterion 4 ran and withdrew; the walk's
    # wording at the time it was made is preserved beside the resolution
    # rather than overwritten.
    assert criteria["4_paired_bca"].startswith("DONE 2026-08-17")
    assert "WITHDRAWN" in criteria["4_paired_bca"]
    assert "NOT DONE -- the only outstanding RUN" in criteria["4_paired_bca"]
    # The deviations criterion does not pass quietly.
    assert "anticipated ONE" in criteria["2_deviations_enumerated"]
    assert "FIVE" in criteria["2_deviations_enumerated"]
    assert "THREE" in criteria["2_deviations_enumerated"]
    # The finding is explicitly not ledgered, with the reason.
    assert "NOT a ledger entry" in criteria["5_ledgered"]
    assert "no interval" in criteria["5_ledgered"]

    assert "DISPOSED" in record["amendment_item_notebook_cell"]
    assert record["paired_bca_is_the_only_outstanding_run"] is True
    assert "no training" in record["and_it_needs_no_training"] or (
        "OOF CSVs exist" in record["and_it_needs_no_training"]
    )
    # What remains is writing, and it is enumerated rather than implied.
    assert len(record["remaining_is_writing_not_running"]) == 3
    assert any(
        "faithful arm" in item
        for item in record["remaining_is_writing_not_running"]
    )
    # The optional run is named as optional, with what it would add.
    optional = record["one_optional_run"]
    assert "NOT required" in optional
    assert "1.026 to 3.774" in optional
    assert "stands either way" in optional


# --------------------------------------------------------------------------
# 2026-08-17, the pod confirms gate 3, and Phase 10 closes
# --------------------------------------------------------------------------


def test_the_pod_confirmed_gate_3_and_the_offline_caveat_is_lifted():
    """Real pretrained features kept every structural claim and made the
    magnitudes larger. The caveat comes off the one clause it covered and
    stays on the rest."""
    record = phase10.GATE_3_FAILED_ON_THE_NOTEBOOK_CELL
    pod = record["confirmed_on_real_features"]
    assert "--gate3" in pod["probe"]
    assert "n 190" in pod["fold"] and "0.3835" in pod["fold"]
    # Larger, not smaller, than offline.
    assert min(pod["rms"]) > 7.9 and max(pod["sd_w_dot_f"]) > 8.0
    assert max(pod["p_max"]) > 0.999
    assert len(pod["predicted_means"]) == 5
    assert "ALL FIVE FAIL" in pod["verdict"]
    # The closed form verified on real features.
    assert "2.7333" in pod["zero_weight_control"]
    assert "(190*2.7263 + 15)/195" in pod["zero_weight_control"]
    assert "W.f alone" in pod["zero_weight_control"]
    # The saturation signature is named as the mechanism's visible form.
    assert "PICKED A CLASS" in pod["saturation_signature"]
    assert "before any training step" in pod["saturation_signature"]
    # The caveat is lifted from one clause and kept on the others.
    assert "2.00-4.01" in pod["caveat_lifted"]
    assert "stays on everything else" in pod["caveat_lifted"]
    assert "none passes" in pod["verdict"]

    # The diagnostic record's own provenance stops saying "confirmable"
    # and starts saying confirmed.
    assert "CONFIRMED on the pod" in record["provenance"]
    assert "confirmed_on_real_features" in record["provenance"]

    # And the CITABLE record -- the finding -- carries the confirmed
    # spread beside the offline one it was registered from.
    finding = phase10.NOTEBOOK_CELL_CANNOT_START_CALIBRATED
    seeded = finding["epoch_0_is_a_seed_function"]
    assert seeded["across_seeds_real_features"] == (2.0015, 4.0099)
    assert seeded["across_seeds_offline"] == (1.026, 3.774)
    assert finding["provenance"].startswith("CONFIRMED ON REAL")


def test_the_coverage_limit_loses_its_exemplar_and_gains_a_proof():
    """The offline seed-2024 instance does not occur on real features.
    The limitation stands because it never needed an instance -- it
    follows from gate 3's own arithmetic."""
    record = phase10.GATE_3_COVERAGE_LIMIT
    corrected = record["corrected"]
    assert corrected.startswith("2026-08-17")
    assert "'three of five' was the OFFLINE pass" in corrected
    assert "ALL FIVE seeds FAIL" in corrected
    assert "does NOT exemplify" in corrected
    # And why it did not: the classes drawn were 2 and 4, not 3.
    assert "grade 3" in corrected and "WOULD have passed" in corrected

    # The offline measurement is relabelled rather than deleted.
    assert "OFFLINE ONLY" in record["measurement_offline"]
    assert "measurement" not in record

    # The limitation is now proved from the gate's own arithmetic.
    structural = record["limitation_is_structural"]
    assert "var(inner) + (mean(inner) - pred_mean)^2" in structural
    assert "passes BY IDENTITY" in structural
    assert "arithmetic rather than by luck" in structural
    assert "does not depend on it" in structural
    # The gate's claim is upgraded, not softened.
    assert "EVERY real-feature seed" in record["the_gate_is_not_wrong"]


def test_the_paired_scope_derives_its_prediction_and_covers_one_pair():
    """One pair, derived rather than typed, and from the baseline figures
    that describe the vectors actually declared."""
    from cleft import ladder

    pairs = phase10.paired_claim_pairs("p10")
    assert len(pairs) == 1
    pair = pairs[0]
    assert pair["a"] == phase10.PAIRED_ARM_STEM == "p10_cleftgnn"
    assert pair["b"] == phase10.PAIRED_BASELINE_STEM
    assert pair["seeds"] == list(ladder.SEED_POOL[:5])

    # The registered prediction, and it comes out of the derivation.
    recorded = pair["recorded"]
    assert recorded["delta_of_means"] == 0.2279
    assert recorded["threshold"] == 0.0607
    assert recorded["margin"] == 3.76
    assert "DESCRIPTIVE, not claimable" in recorded["source"]
    spec = phase10.PHASE_10_PAIRED_REGISTERED
    assert spec["prediction"]["delta"] == recorded["delta_of_means"]
    assert spec["prediction"]["margin"] == recorded["margin"]

    # It is the D1 artifact figures, and the record says why that matters.
    assert "STAGE_D1_AT_G1" in recorded["source"]
    assert "0.2280" in spec["prediction"]["which_baseline_figure"]
    assert "the declared vectors are the D1 run's" in (
        spec["prediction"]["which_baseline_figure"]
    )
    assert ladder.STAGE_D1_AT_G1["cells"]["vit_b16"][0] == 0.2520

    # The derivations in `ladder` accept these pairs unchanged.
    vectors = ladder.paired_claim_vectors("p10", pairs=pairs)
    stems = ladder.paired_claim_stems("p10", pairs=pairs)
    assert len(vectors) == 10 and len(stems) == 2
    groups = ladder.paired_claim_seed_groups("p10", pairs=pairs)
    assert len(groups) == 1

    # An unknown scope is refused rather than silently treated as p10.
    with pytest.raises(ValueError, match="unknown Phase 10"):
        phase10.paired_claim_pairs("p11")

    # The coverage record says what is excluded and why, not just what is in.
    coverage = phase10.PAIRED_CLAIM_COVERAGE
    assert set(coverage["excluded"]) == {
        "notebook_cell", "faithful_arm", "rater_screen",
    }
    assert "no vectors" in coverage["excluded"]["notebook_cell"]
    assert "DIFFERENT target" in coverage["excluded"]["rater_screen"]


def test_the_paired_config_is_two_pass_and_declares_ten_vectors():
    from pathlib import Path

    import yaml

    repo = Path(__file__).resolve().parents[1]
    path = repo / "configs" / "p10_paired.yaml"
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert payload["task"] == {
        "kind": "paired_claims", "scope": "p10", "n_boot": 10000,
    }
    inputs = payload["inputs"]
    assert len(inputs) == 10

    placeholder = "0" * 64
    # [2026-08-17, BOTH PASSES DONE] The run directory was pasted and
    # declare_inputs.py verified the files, so the config is fully
    # resolved: no PENDING path, no placeholder hash, nothing left for
    # guard 3 to refuse.
    assert all("/PENDING_" not in e["path"] for e in inputs)
    replication = [e for e in inputs if "p10_cleftgnn" in e["name"]]
    baseline = [e for e in inputs if "p7_d1_vit_b16" in e["name"]]
    assert len(replication) == 5 and len(baseline) == 5
    assert all("p10-cleftgnn-6" in e["path"] for e in replication)

    from cleft.run_names import check_run_dir

    for entry in inputs:
        # Every declared path is a real run directory obeying the contract.
        check_run_dir(entry["path"].split("/")[-2])
        assert entry["rollup_sha256"] != placeholder
        assert len(entry["rollup_sha256"]) == 64
        assert set(entry["rollup_sha256"]) <= set("0123456789abcdef")
    # Ten vectors, ten distinct files -- a repeated hash would mean two
    # seeds wrote the same predictions, which is not a thing that happens.
    assert len({e["rollup_sha256"] for e in inputs}) == 10

    header = path.read_text(encoding="utf-8")
    assert "**RESOLVED.**" in header
    assert "PLACEHOLDER" not in header
    assert "PENDING" not in header
    assert "FITS NOTHING" in header
    # And the header names the baseline figures the prediction came from.
    assert "STAGE_D1_AT_G1" in header
    assert "CLEAN_GEOMETRY_MEASUREMENTS" not in header
    assert "3.76x" in header and "0.2279" in header
    assert "AN UNSTABLE ARM IS HARD TO BEAT CLAIMABLY" in header
    # Both readings are in the config, not only in the record.
    assert "UNDER THIS PROTOCOL" in header
    assert "COHORT_CANNOT_RESOLVE" in header


def test_the_paired_scope_reaches_the_task_and_the_schema():
    import inspect

    from cleft import run as run_module
    from cleft.config.schema import TASK_SPECS

    assert "p10" in TASK_SPECS["paired_claims"]["scope"].choices
    source = inspect.getsource(run_module.task_paired_claims)
    # The dispatcher gains one branch; the comparison itself is not copied.
    assert 'else phase10 if scope == "p10"' in source
    assert source.count("paired_claim_pairs(scope)") == 1
    assert source.count("phase7b.paired_comparison") <= 1


def test_all_epochs_stage_tables_are_written_and_the_old_reading_survives():
    import inspect

    from cleft import run as run_module

    record = phase10.ALL_EPOCHS_RECORDED
    assert record["built"].endswith("no run attached")
    assert "epoch column" in record["change"]
    assert "one filter away" in record["change"]
    assert "already populated for every epoch" in record["costs_nothing"]
    assert "discarded computed data twice" in record["why_now"]

    source = inspect.getsource(run_module.task_cleftgnn_cv)
    # Every epoch's rows, with the selected one still identifiable.
    assert "for epoch in sorted(by_epoch)" in source
    assert "fold,epoch,selected,selected_epoch,stage" in source
    assert "chosen = int(epoch == fold_run.selected_epoch)" in source
    # And no second forward was added to pay for it.
    assert "record_stages=True" in source
    assert source.count("harness.run_cv(") == 1


def test_the_faithful_arm_closes_in_its_own_words():
    record = phase10.FAITHFUL_ARM_CLOSING
    assert record["closed"] == "2026-08-17"
    sentence = record["sentence"]
    assert "four of five raters" in sentence
    assert "0.077-0.154" in sentence
    assert "epoch 33" in sentence
    assert "-0.1287" in sentence and "+0.0663" in sentence
    assert "claimable" in sentence

    # The two things absorption would have hidden.
    reasons = record["why_its_own_sentence"]
    assert len(reasons) == 2
    assert "THE GAP IS NOT PROTOCOL" in reasons[0]
    assert "evaluated differently" in reasons[0]
    assert "OPPOSITE SIGNS" in reasons[1]
    assert "0.598/0.283" in reasons[1]

    assert "not ledgered" in record["disposition"]
    assert "the CONTRAST" in record["disposition"]


def test_the_closing_carries_criterion_2s_arithmetic_and_the_eight_items():
    record = phase10.PHASE_10_CLOSING
    assert record["closed"] == "2026-08-17"
    assert record["answer"].startswith("NO, unqualified")

    # Criterion 2: eight against one, and the closing says so.
    arithmetic = record["criterion_2_arithmetic"]
    assert arithmetic["anticipated"] == 1
    assert arithmetic["manuscript_cell"] == 5
    assert arithmetic["notebook_cell"] == 3
    assert arithmetic["total"] == 8
    assert arithmetic["total"] == (
        arithmetic["manuscript_cell"] + arithmetic["notebook_cell"]
    )
    assert "true and misleading" in arithmetic["statement"]
    assert "CORRECTED UPWARD" in arithmetic["statement"]
    assert "PERMANENTLY UNEXERCISED" in arithmetic["notebook_cells_three"]
    # The counts agree with the deviation records themselves.
    assert arithmetic["manuscript_cell"] == phase10.REGISTERED_DEVIATIONS[
        "expected_count"
    ]
    assert arithmetic["notebook_cell"] == phase10.NOTEBOOK_CELL_DEVIATIONS[
        "expected_count"
    ]

    # Criterion 1: seven discrepancies plus the blocking ambiguity.
    items = record["criterion_1_discrepancies"]
    assert len(items) == 8
    assert sum("BLOCKING" in item for item in items) == 1
    assert any(item.startswith("region count") for item in items)
    assert any("CIFAR-10" in item for item in items)
    assert "has never been run" in record["sharpest"]

    # Four cells, each with its own disposition.
    cells = record["four_cells"]
    assert set(cells) == {
        "protocol_arm", "faithful_arm", "rater_screen", "notebook_cell",
    }
    assert "NOT protocol" in cells["faithful_arm"]
    assert "FINDING, not a number" in cells["notebook_cell"]

    # The convergence, with the mechanism rather than an assertion.
    assert "FOUR independent routes" in record["what_no_single_arm_shows"]
    assert "0.0068" in record["what_no_single_arm_shows"]
    assert "SABM DEGRADES" in record["what_no_single_arm_shows"]

    # [2026-08-17, same day] The one run left ran, and it withdrew.
    assert record["still_open"].startswith("NOTHING")
    assert "WITHDRAWN" in record["still_open"]
    assert "ledger entry 26" in record["still_open"]
    assert "applied verbatim" in record["still_open"]


# --------------------------------------------------------------------------
# 2026-08-17, the paired BCa withdraws and Phase 10 closes
# --------------------------------------------------------------------------


def test_the_paired_verdict_is_withdrawn_and_the_prediction_held():
    """Both conditions recorded with their figures, and the prediction
    checked against the registration rather than remembered."""
    record = phase10.PAIRED_BCA_OBSERVED
    assert record["observed"].startswith("2026-08-17")
    assert "p10_paired__27b895b6__p10-paired" in record["observed"]
    assert record["key"] == "p10__replication_vs_0p2520"
    assert record["verdict"] == "WITHDRAWN"
    # PLAN 4.3 needs BOTH; one true and one false withdraws.
    assert record["condition_1"] is False
    assert record["condition_2"] is True
    assert record["condition_2_margin"] == 3.76
    assert record["seeds_excluding_zero"] == "3 of 5"

    arms = record["arms_as_declared"]
    assert arms["p10_cleftgnn"] == {"mean": 0.0242, "sd": 0.0676}
    assert arms["p7_d1_vit_b16_imagenet_g1"] == {"mean": 0.2520, "sd": 0.0148}
    # The threshold is driven by the noisier arm: 4.6x.
    assert round(
        arms["p10_cleftgnn"]["sd"] / arms["p7_d1_vit_b16_imagenet_g1"]["sd"], 1
    ) == 4.6

    # The prediction held -- and it is checked against the registration.
    held = record["prediction_held"]
    predicted, observed = held["delta"]
    spec = phase10.PHASE_10_PAIRED_REGISTERED["prediction"]
    assert predicted == spec["delta"] == observed == record["d"]
    assert held["threshold_and_margin"] == (spec["threshold"], spec["margin"])
    assert "condition 1" in held["decider"]
    # And it is NOT oversold: the arithmetic was derived, the verdict was
    # what was actually at risk.
    assert held["honest_reading"].startswith("NOT foresight")
    assert "at risk was the VERDICT" in held["honest_reading"]

    # The 0.0001 is recorded rather than smoothed.
    assert "0.0242" in record["the_0_0001"] and "0.0241" in record["the_0_0001"]
    assert "0.2278" in record["the_0_0001"]
    assert "goes unnoticed" in record["the_0_0001"]

    # The registered reading is applied verbatim, not reworded.
    applied = record["registered_reading_applied"]
    assert applied.startswith("NOT CLAIMABLE, verbatim")
    assert "FOURTH independent arrival at COHORT_CANNOT_RESOLVE" in applied
    registered = phase10.PHASE_10_PAIRED_REGISTERED["readings_committed"]
    assert "COHORT_CANNOT_RESOLVE" in registered["not_claimable"]

    # What the withdrawal does and does not say.
    meaning = record["what_that_means"]
    assert "0.024" in meaning and "NOT CLAIMABLE" in meaning
    assert "instability sets the threshold" in meaning
    assert "not about the two architectures" in meaning


def test_the_withdrawal_is_ledgered_and_the_earlier_pins_still_verify():
    """Entry 26, append-only: the born prefix and entry 25's pin are
    untouched, so nothing earlier was rewritten to make room."""
    from cleft import results_ledger

    results_ledger.validate()
    # [2026-08-17] `>=`, not `==`: the ledger is append-only and later
    # phases add rows. The PREFIX CHECKSUMS below are the guard -- they
    # break if anything at or before entry 26 is rewritten, which is the
    # property this test exists for.
    assert len(results_ledger.ENTRIES) >= 26
    # Appending leaves every earlier prefix checksum exactly as it was.
    assert results_ledger.cumulative_checksum(25) == (
        "1b1f3167715f724cf802bc88bd23497b6c8832b9d502000196d1e6354aa83f38"
    )
    assert results_ledger.cumulative_checksum(26) == (
        "f310ef8f284ada62525ab6625fcb1561de9914602d3ea12f8a4adf30a33920df"
    )

    entry = results_ledger.ENTRIES[25]
    assert entry["id"] == "p10-replication-vs-best-arm-withdrawn"
    assert entry["status"] == "WITHDRAWN"
    assert entry["framing"] == "main"
    assert entry["phase"] == "p10"
    assert entry["corrects"] is None

    # Both conditions carry their figures, not just their verdicts.
    assert "FALSE" in entry["condition_1"] and "3 of 5" in entry["condition_1"]
    assert "3.76x" in entry["condition_2"] and "0.0607" in entry["condition_2"]
    assert "condition 1 decides" in entry["condition_2"]
    assert "0.2279" in entry["claim"] and "0.0242" in entry["claim"]
    assert "COHORT_CANNOT_RESOLVE" in entry["claim"]

    # The cause is attached to the claim rather than footnoted.
    assert "THE CAUSE IS THE LOSER'S INSTABILITY" in entry["caveats"][0]
    assert "4.6x" in entry["caveats"][0]
    # And the withdrawal is fenced from the reading it is not.
    assert any(
        "does NOT say the two architectures are comparable" in c
        for c in entry["caveats"]
    )
    assert any("FITS NOTHING" in c for c in entry["caveats"])


def test_phase_10_closes_on_six_met_criteria_and_carries_six_items_by_name():
    record = phase10.PHASE_10_CLOSING

    criteria = record["exit_criteria"]
    assert len(criteria) == 6
    assert all(v.startswith("MET") for v in criteria.values())
    assert "WITHDRAWN" in criteria["4_paired_bca"]
    assert "EIGHT" in criteria["2_deviations_enumerated"]
    assert "entry 26" in criteria["5_ledgered"]
    assert "nothing earlier was rewritten" in criteria["5_ledgered"]

    # The closing rests only on what was recorded when it happened.
    cites = record["cites_only_what_is_recorded"]
    assert "87x" in cites["manuscript_cell"]
    assert "0.7794" in cites["manuscript_cell"]
    assert "0.230-0.267" in cites["manuscript_cell"]
    assert "function of the SEED" in cites["notebook_cell"]
    assert "real pretrained features" in cites["notebook_cell"]
    assert "0.94-1.67 SD" in cites["notebook_cell"]
    assert "GAP IS NOT PROTOCOL" in cites["faithful_arm"]
    assert "PROVED from gate 3's own arithmetic" in (
        cites["gate_3_coverage_limit"]
    )
    assert "Tolerances untouched" in cites["gate_3_coverage_limit"]
    assert "entry 25" in cites["ledger"] and "entry 26" in cites["ledger"]
    assert "adds nothing" in record["no_new_claims"]

    # Six open items, each carried BY NAME so none is lost to a summary.
    carried = record["carried_forward_open"]
    assert len(carried) == 6
    for name in (
        "PHASE_11_BLOCKED_ON_IEM_DIRECTION",
        "SLIDE_9_CLEFT_NUMBERS_REQUESTED",
        "U_SHAPE_BELONGS_LATER",
        "phase8.RETRY_LIMIT_IS_NOT_HOLDING",
        "phase9.PRE_CONTEXT_REFUSALS_LEAVE_NO_TRACE",
    ):
        assert any(name in item for item in carried), name
    assert any("unanimity confirmation" in item for item in carried)
    # The blocking one is still labelled blocking as it travels.
    assert any("BLOCKING for Phase 11" in item for item in carried)

    # Every named record actually exists, so a carried name cannot rot.
    from cleft import phase8, phase9

    assert phase10.PHASE_11_BLOCKED_ON_IEM_DIRECTION["blocking"]
    assert phase10.SLIDE_9_CLEFT_NUMBERS_REQUESTED["request"]
    assert phase10.U_SHAPE_BELONGS_LATER["deferred"]
    assert phase8.RETRY_LIMIT_IS_NOT_HOLDING["status"].startswith("OPEN")
    assert phase9.PRE_CONTEXT_REFUSALS_LEAVE_NO_TRACE
