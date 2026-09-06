"""Phase 8c: the rendering's honesty, checked where the laptop can check it."""

from __future__ import annotations

import numpy as np
import pytest

from cleft import phase8c


def test_block_upsample_introduces_no_value_the_grid_did_not_have():
    """**The honesty primitive.** Bilinear smoothing draws the kernel;
    nearest blocks draw the model. Every output pixel must equal exactly
    one input cell, at exactly 16x."""
    rng = np.random.default_rng(0)
    grid = rng.random((14, 14))
    out = phase8c.block_upsample(grid, 16)
    assert out.shape == (224, 224)
    assert set(np.round(out.ravel(), 12)) == set(np.round(grid.ravel(), 12))
    # And each 16x16 block is constant -- no gradient inside a block.
    assert float(out[:16, :16].std()) == 0.0
    with pytest.raises(phase8c.Phase8cError, match="2-D"):
        phase8c.block_upsample(grid.ravel())


def test_rater_multiset_recovers_five_scores_and_refuses_non_fifths():
    """soft_k are grade FRACTIONS; five raters means fifths, asserted."""
    row = {"soft_1": "0.0", "soft_2": "0.4", "soft_3": "0.4",
           "soft_4": "0.2", "soft_5": "0.0"}
    assert phase8c.rater_multiset(row) == [2, 2, 3, 3, 4]
    with pytest.raises(phase8c.Phase8cError, match="not a fifth"):
        phase8c.rater_multiset({**row, "soft_2": "0.33"})
    bad_sum = {"soft_1": "0.2", "soft_2": "0.2", "soft_3": "0.2",
               "soft_4": "0.2", "soft_5": "0.4"}
    with pytest.raises(phase8c.Phase8cError, match="sum"):
        phase8c.rater_multiset(bad_sum)


def test_the_flatness_panel_has_a_fixed_axis_and_never_sorts():
    """**The flatness IS the picture.** Uniform weights must render as
    equal bars at the 1/27 line -- an auto-scaled axis would fabricate the
    ranking the dust verdict refuted."""
    uniform = np.full(26, 1 / 27)
    panel = phase8c.flatness_panel(uniform)
    # Every bar column has the same top row (equal heights).
    bar_width = panel.shape[1] // 26
    tops = []
    for index in range(26):
        column = panel[:, index * bar_width]
        filled = np.flatnonzero((column == (70, 90, 160)).all(axis=1))
        tops.append(filled[0] if len(filled) else -1)
    assert len(set(tops)) == 1, "uniform weights rendered at unequal heights"
    # A shuffled input renders bars in INPUT order (nothing sorts): a
    # distinctive value stays at its own column.
    values = np.full(26, 1 / 27)
    values[7] = 0.09
    panel = phase8c.flatness_panel(values)
    heights = []
    for index in range(26):
        column = panel[:, index * bar_width]
        filled = np.flatnonzero((column == (70, 90, 160)).all(axis=1))
        heights.append(len(filled))
    assert np.argmax(heights) == 7
    with pytest.raises(phase8c.Phase8cError, match="26"):
        phase8c.flatness_panel(np.ones(27))


def test_frames_composite_the_face_and_apng_assembles(tmp_path):
    """**A Grad-CAM frame without its image is not a Grad-CAM frame** --
    the unviolable rule from the eye review. Every composited frame's
    pixels must differ from the pure heat grid, the no-map frame IS the
    face, and the composite depends on WHICH face it covers."""
    from PIL import Image

    rng = np.random.default_rng(1)
    grid = rng.random((14, 14))
    face_a = (rng.random((224, 224, 3)) * 255).astype(np.uint8)
    face_b = np.zeros((224, 224, 3), dtype=np.uint8)
    face_b[40:180, 40:180] = 200

    drawn = phase8c.frame(face_a, grid, axis_label="training checkpoint e1s3")
    pure_heat = phase8c.heat_panel(grid)
    body = drawn[: pure_heat.shape[0], : pure_heat.shape[1]]
    assert not np.array_equal(body, pure_heat), (
        "the frame is heat-on-black; the face is missing"
    )
    # The composite depends on which face it covers.
    drawn_b = phase8c.frame(face_b, grid, axis_label="training checkpoint e1s3")
    assert not np.array_equal(drawn, drawn_b)
    # The before-state frame IS the photograph -- never a void.
    before = phase8c.frame(face_a, None, axis_label="before training")
    assert np.array_equal(before[:224, :224], face_a)
    # Zero heat leaves the photograph untouched under the blend.
    untouched = phase8c.composite_heat(face_a, np.zeros((14, 14)))
    assert np.array_equal(untouched, face_a)

    path = tmp_path / "axis.png"
    phase8c.write_animation([drawn, before, drawn_b], path)
    with Image.open(path) as animation:
        assert getattr(animation, "n_frames", 1) == 3
    with pytest.raises(phase8c.Phase8cError, match="no frames"):
        phase8c.write_animation([], tmp_path / "empty.png")


def test_the_records_carry_the_decisions_verbatim():
    naming = phase8c.PANEL_1_NAMING
    assert naming["label"] == (
        "22 cleft-anatomy regions (of the 27-patch frozen scheme)"
    )
    assert "Road B Branch 3 module" in naming["naming_note"]
    assert "strict subset" in naming["naming_note"]

    registered = phase8c.ANIMATION_REGISTERED
    assert registered["a_checkpoints"]["epoch_1_steps"] == (1, 2, 3, 5, 10, 20, 50)
    assert "BARRED" in registered["a_checkpoints"]["head"]
    assert "unordered" in registered["axis_c_seeds"]
    assert "failed-gate stamp" in registered["b_walk"]
    assert "block 11 structurally excluded" in registered["d_depth"]["blocks"]
    assert "'better'" in registered["d_depth"]["caveat"]
    assert "byte-identical by construction" in registered["endpoint_resolution"]
    assert "recorded, not smoothed" in registered["endpoint_resolution"]
    assert "no interpolated frames" in registered["frame_rules"]

    design = phase8c.SHEET_DESIGN
    assert "no map on this sheet is a published claim" in design["banner"]
    assert "not per-rater columns" in design["rater_columns"]
    assert "no ranking fabricated from dust" in design["panels"][2]


def test_the_tasks_read_stamps_from_artifacts_and_never_recompose(repo_root):
    source = (repo_root / "src" / "cleft" / "run.py").read_text(encoding="utf-8")
    sheet = source.split("def task_phase8c_sheet")[1].split("\ndef ")[0]
    animation = source.split("def task_phase8c_animation")[1].split("\ndef ")[0]
    # Stamps and the sentence come FROM the npz, and land on the APPENDIX.
    for fragment in ('methods["original_verdict"]', 'methods["caveat"]',
                     'methods["both_numbers_sentence"]'):
        assert fragment in sheet, fragment
    assert "appendix_page(" in sheet and "clinical_page(" in sheet
    assert "index_page(" in sheet
    # The animation run is a DECLARED input the pages embed.
    assert 'task["animation_artifact"]' in sheet
    assert "file_data_uri" in sheet
    # The sheet uses the 22-set and refuses a drifted count.
    assert "protected_patches()" in sheet
    assert "not the 22" in sheet
    # Curves rows are tagged per source so no cross-fold/seed joins draw.
    assert '"series": name' in sheet
    # The animation's frames composite the FACE, its endpoint frames are
    # READ from the npz, the walk restores a clean snapshot, depth stops at
    # block 10, and the fixed extension has no .apng.png anywhere.
    assert 'methods["original_grids"][position]' in animation
    assert "load_state_dict(snapshot)" in animation
    assert "range(0, 11)" in animation
    assert "phase8c.frame(\n            face" in animation.replace("    ", "    ") or (
        "frame(\n                face" in animation
    )
    assert ".apng.png" not in source
    assert "on_head=capturing_head" in animation


def test_the_rebuild_record_and_the_clinical_pages():
    """The rebuild's decisions, and the pages that carry them: plain
    sentences present, provenance on the appendix, everything linked."""
    record = phase8c.CLINICAL_DELIVERY_REBUILT
    assert record["verdict"] == "measured data sound; delivery failed its audience"
    assert "no face under the heat" in record["defects"][0]
    assert ".apng.png" in record["defects"][2]
    assert "frame_amendment" in record and "axis value only" in record["frame_amendment"]
    assert record["clinical_caveat"] == phase8c.CLINICAL_CAVEAT
    assert "max |difference| 1.0" in record["input_dependence_measured"]
    assert "differ from the pure heat grid" in record["unviolable"]

    page = phase8c.clinical_page(
        patient=55, hero_uri="data:x", train_anim_uri="data:t",
        walk_anim_uri="data:w", depth_anim_uri="data:d",
        seed_strip_uri="data:s", regions_uri="data:r",
        prediction_uri="data:p",
        prediction_sentence="Five raters scored this outcome.",
        appendix_href="8c_appendix.html", index_href="8c_index.html",
    )
    assert "each frame is one training step" in page.lower()
    assert phase8c.CLINICAL_CAVEAT in page
    assert "8c_appendix.html" in page and "8c_index.html" in page
    assert "<details>" in page  # walk/depth collapsed, secondary
    # No verdict stamps on the clinical page -- they live on the appendix.
    assert "FAILED its own gate" not in page
    assert "UNESTIMABLE" not in page

    appendix = phase8c.appendix_page(
        caveat_verbatim="the backbone is frozen: X",
        original_stamp="FAILED its own gate; maps unpublished",
        variant_stamp="UNESTIMABLE -- 0 of 15",
        both_numbers_sentence="19 of 75 cells refuse; ...",
        refusal_summary="counts...", flatness_uri="data:f",
        curves_uri="data:c", scatter_uri="data:s",
        index_href="8c_index.html",
    )
    assert "no map on any page" in appendix.lower()
    assert "FAILED its own gate" in appendix
    assert "UNESTIMABLE" in appendix
    assert "the backbone is frozen: X" in appendix

    index = phase8c.index_page(
        patients=[55, 120], scatter_uri="data:s",
        appendix_href="8c_appendix.html",
    )
    assert '8c_patient_55.html' in index and '8c_patient_120.html' in index
    assert "jittered for visibility" in index


def test_phase_8c_closes_on_the_review_gate_with_the_property_pinned():
    """**The review is the gate, and it fired once**: round 1 failed on
    audience, round 2 passed -- and the clinician-cold rule is now a tested
    property, which this suite itself enforces."""
    closing = phase8c.PHASE_8C_CLOSING
    assert "PASSED" in closing["closed"]
    assert closing["closing_artifacts"]["animation_run"] == (
        "p8c_animation__8d8fe58f__p8c-animation-2"
    )
    assert "one entry point" in closing["closing_artifacts"]["deliverable"]
    gate = closing["gate"]
    assert "FAILED on audience" in gate["round_1"]
    assert "PASSED" in gate["round_2"]
    assert "not \nan intention" in gate["now_a_property"] or (
        "not an intention" in gate["now_a_property"].replace("\n", " ")
    )
    assert "CLINICAL_DELIVERY_REBUILT" in gate["now_a_property"]
    assert "byte 36" in closing["input_dependence"]
    assert "measured checkpoints" in closing["request_answered"]
    for artifact in ("grad_cam_methods.npz", "node_weights.npz"):
        assert artifact in closing["consumed"], artifact
    assert "t-SNE" in closing["still_untouched"][0]


def test_tsne_figure_cannot_exist_without_its_companion():
    """The registered display rule, structural: the only function that
    composes the separation figure refuses a companion missing a field,
    and the composed figure carries the companion strip beneath the
    panels rather than beside the conversation."""
    rng = np.random.default_rng(2)
    points = {5: rng.normal(size=(30, 2)), 30: rng.normal(size=(30, 2))}
    classes = np.array([1] * 10 + [2] * 15 + [3] * 5)

    with pytest.raises(phase8c.Phase8cError, match="companion"):
        phase8c.tsne_figure(points, classes, {"k": 5})

    companion = {
        "k": 5, "accuracy": 0.7, "interval": (0.6, 0.8),
        "chance": 1.0 / 3.0, "majority": 0.5,
        "class_counts": (10, 15, 5), "n_boot": 100,
    }
    figure = phase8c.tsne_figure(points, classes, companion)
    bare = phase8c.side_by_side([
        phase8c.tsne_scatter(points[5], classes, title="x"),
        phase8c.tsne_scatter(points[30], classes, title="x"),
    ])
    assert figure.shape[0] > bare.shape[0]
    assert figure.ndim == 3 and figure.shape[2] == 3


def test_tsne_scatter_refuses_misaligned_points():
    rng = np.random.default_rng(4)
    with pytest.raises(phase8c.Phase8cError, match="aligned"):
        phase8c.tsne_scatter(
            rng.normal(size=(10, 2)), np.array([1] * 9), title="x"
        )


def test_captioned_frames_carry_the_sentence_and_keep_the_count():
    """The recipe sentence is burned beneath every frame at assembly:
    same frame count, every output taller than its input by the same
    strip, and an empty list refuses."""
    rng = np.random.default_rng(5)
    frames = [
        rng.integers(0, 255, size=(60, 80, 3)).astype(np.uint8)
        for _ in range(4)
    ]
    out = phase8c.captioned_frames(frames, "a sentence")
    assert len(out) == 4
    growth = {o.shape[0] - f.shape[0] for o, f in zip(out, frames)}
    assert len(growth) == 1 and growth.pop() > 0
    assert all(o.shape[1] == 80 for o in out)
    with pytest.raises(phase8c.Phase8cError, match="caption"):
        phase8c.captioned_frames([], "a sentence")


def test_endpoint_comparison_panel_puts_the_frames_side_by_side():
    rng = np.random.default_rng(6)
    final = rng.integers(0, 255, size=(60, 80, 3)).astype(np.uint8)
    shipped = rng.integers(0, 255, size=(60, 80, 3)).astype(np.uint8)
    panel = phase8c.endpoint_comparison_panel(
        final, shipped, ["the sentence", "difference: 0.123"]
    )
    assert panel.shape[1] >= final.shape[1] + shipped.shape[1]
    assert panel.shape[0] > final.shape[0]


# --------------------------------------------------------------------------
# the panel was misdescribed on the pages, corrected 2026-09-05
# --------------------------------------------------------------------------


def test_the_panel_correction_is_recorded_with_what_it_does_not_touch():
    """The pages said "five clinicians". One of the five is a cleft
    patient. The correction is dated, and it is explicit that banked
    sheets still carry the old sentence."""
    record = phase8c.THE_PANEL_WAS_MISDESCRIBED_ON_THE_PAGES
    assert record["corrected"] == "2026-09-05"

    wrong = " ".join(record["what_was_wrong"].split())
    assert "five clinicians" in wrong
    assert "One of the five is not a clinician" in wrong

    why = " ".join(record["why_it_is_not_a_wording_quibble"].split())
    assert "THE_ASSUMPTIONS_DECLARED" in why
    assert "interchangeable draws" in why
    assert "0.654" in why and "0.628" in why

    banked = " ".join(
        record["SHEETS_ALREADY_GENERATED_CARRY_THE_OLD_WORDING"].split()
    )
    assert "before 2026-09-05" in banked
    assert "NOT edited by this correction" in banked

    later = " ".join(record["regenerating_them_is_a_SEPARATE_decision"].split())
    assert "not taken here" in later
    assert "the generator is the\n        one that is right".replace(
        "\n        ", " "
    ) in later


def test_the_pages_name_the_panel_rather_than_calling_it_clinical():
    """Asserted on the generated HTML, not on the record about it."""
    index = phase8c.index_page(
        patients=[1], scatter_uri="data:x", appendix_href="a.html"
    )
    # The page says "not five clinicians" ON PURPOSE; what must not
    # appear is the phrase standing as a DESCRIPTION of the panel.
    assert "five clinicians" not in index.replace(
        "not five clinicians", ""
    )
    for role in ("cleft patient", "orthodontist",
                 "speech and language therapist", "plastic surgeon",
                 "psychologist"):
        assert role in index, role

    # And every role named on the page is a real score-sheet column.
    from cleft.data import scoresheet

    columns = " ".join(scoresheet.RATERS).lower()
    for role in ("cleft patient", "orthodontist",
                 "speech and language therapist", "plastic surgeon",
                 "psychologist"):
        assert role in columns, role
