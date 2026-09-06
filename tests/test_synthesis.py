"""Unilateral asymmetry synthesis by TPS (Phase 5, PLAN §4.11 Q-b).

The properties that fail silently if wrong: the deformation is local (anchors and
frame do not drift), unilateral (the untouched side does not move), monotone in
magnitude, deterministic, side-balanced, and it survives G2's unwarp. Each is
asserted with a counterfactual where one exists, because a check that passes on
its own failure mode is this project's recurring defect (PLAN R7).
"""

from __future__ import annotations

import numpy as np
import pytest

from cleft.scut import landmarks as L
from cleft.scut import placement, synthesis
from test_scut_placement import a_realistic_face

SOURCE = 400


def a_textured_face() -> np.ndarray:
    """A face-ish image with fine texture, so a warp is visible in the pixels.

    Flat colour would make any warp a no-op in the pixels while the landmarks
    moved -- the deformation would look successful and change nothing.
    """
    rng = np.random.default_rng(4)
    image = np.zeros((SOURCE, SOURCE, 3), dtype=np.uint8)
    image[:, :] = (40, 60, 80)
    image[40:360, 60:340] = (205, 170, 150)
    noise = rng.integers(0, 40, size=(320, 280, 3), dtype=np.int16)
    image[40:360, 60:340] = np.clip(
        image[40:360, 60:340].astype(np.int16) + noise, 0, 255
    ).astype(np.uint8)
    return image


def a_symmetric_textured_face() -> np.ndarray:
    """Textured AND exactly mirror-symmetric about the facial midline (x=200).

    **The fixture the asymmetry-survival check needs, and why the noisy one will
    not do.** PLAN §4.4 prescribes the construction: *synthesise a known
    asymmetry, unwarp, confirm it survives.* A randomly textured face is already
    maximally mirror-asymmetric, so the whole-frame mirror difference is
    dominated by the texture and a local warp moves it by ~1e-5 -- the instrument
    reads a floor rather than the deformation, which is R2 in miniature. From a
    symmetric base the introduced asymmetry IS the whole signal.

    Built by generating the left half and reflecting it, so symmetry is exact by
    construction rather than approximate.
    """
    rng = np.random.default_rng(11)
    image = np.zeros((SOURCE, SOURCE, 3), dtype=np.uint8)
    image[:, :] = (40, 60, 80)
    half = SOURCE // 2
    left = np.full((SOURCE, half, 3), (205, 170, 150), dtype=np.int16)
    # Coarse blobs rather than per-pixel noise: nearest-neighbour resampling of
    # per-pixel noise is itself a large perturbation, which would put a second
    # source of difference into the measurement.
    for _ in range(60):
        cy, cx = rng.integers(40, 360), rng.integers(10, half - 10)
        size = int(rng.integers(6, 16))
        left[
            max(cy - size, 0):cy + size, max(cx - size, 0):cx + size
        ] += int(rng.integers(-45, 45))
    left = np.clip(left, 0, 255).astype(np.uint8)
    image[:, :half] = left
    image[:, half:] = left[:, ::-1]
    image[:40] = (40, 60, 80)
    image[360:] = (40, 60, 80)
    return image


def a_case(magnitude: float = 0.06, side: str = "left"):
    points = a_realistic_face(midline=200.0, brow_y=110.0, span=170.0)
    box = placement.crop_box(points, 0.74)
    return a_textured_face(), points, box, side, magnitude


# --------------------------------------------------------------------------
# the two limitations, recorded BEFORE the arm runs
# --------------------------------------------------------------------------


def test_the_two_limitations_are_recorded_in_the_module():
    """PLAN §4.11 Q-b: **two limitations must be recorded in the artifact before
    the arm runs**, in the SymNose construct-note pattern -- properties of the
    instrument, not disclaimers about a component of it."""
    limitations = synthesis.LIMITATIONS
    assert "DEFORMATION ONLY" in limitations["no_scar"]
    assert "no surgical scar" in limitations["no_scar"]
    assert "STATED, not closed" in limitations["no_scar"]

    mapping = limitations["magnitude_to_grade_is_an_assumption"]
    assert "NOTHING HERE ESTABLISHES" in mapping
    assert "ORDERED series, not a graded one" in mapping


# --------------------------------------------------------------------------
# the index must describe the arrays beside it
# --------------------------------------------------------------------------


def an_index(n_faces: int, magnitudes=(0.0, 0.015, 0.025, 0.035)):
    """A well-formed (per_face, survival) pair for ``n_faces``."""
    per_face = [{"stem": f"AF{i}"} for i in range(n_faces)]
    n_deformed = sum(1 for m in magnitudes if m > 0)
    survival = [{"stem": f"AF{i}"} for i in range(n_faces) for _ in range(n_deformed)]
    return per_face, survival


def test_a_well_formed_index_passes_and_reports_what_it_checked():
    per_face, survival = an_index(200)
    counts = synthesis.assert_index_describes_arrays(
        per_face, survival, 200, (0.0, 0.015, 0.025, 0.035)
    )
    assert counts["n_faces"] == 200
    assert counts["n_unique_stems"] == 200
    assert counts["n_survival_rows"] == counts["expected_survival_rows"] == 600


def test_the_exact_shape_the_cluster_gate_produced_is_refused():
    """**[DEFECT 2026-07-30] Eight of nine files byte-identical; `faces.json` was
    not.** 211 rows for a 200-face build, 11 duplicate stems, survival at 634
    against 600 — a resume that rolled the arrays back but appended to the
    journal. Every array was correct, so nothing else would have noticed.

    Reconstructed here exactly, because a regression test built from a
    hypothetical shape would not prove this one is caught.
    """
    per_face, survival = an_index(200)
    # 11 faces reprocessed: their rows appended a second time.
    per_face += [{"stem": f"AF{i}"} for i in range(189, 200)]
    survival += [{"stem": f"AF{i}"} for i in range(189, 200) for _ in range(3)]
    survival.append({"stem": "AF199"})  # the in-flight face's partial row

    with pytest.raises(synthesis.SynthesisError) as excinfo:
        synthesis.assert_index_describes_arrays(
            per_face, survival, 200, (0.0, 0.015, 0.025, 0.035)
        )

    message = str(excinfo.value)
    assert "211 rows, expected 200" in message
    assert "11 duplicate stem(s)" in message
    assert "634 rows, expected 600" in message
    assert "Not publishing" in message


def test_each_failure_mode_is_caught_on_its_own():
    """A composite check that only fires when everything is wrong at once would
    miss the single-fault cases, which are the likelier ones."""
    magnitudes = (0.0, 0.015, 0.025, 0.035)

    # Too few faces -- a journal short of its checkpoint.
    per_face, survival = an_index(200)
    with pytest.raises(synthesis.SynthesisError, match="199 rows, expected 200"):
        synthesis.assert_index_describes_arrays(
            per_face[:-1], survival, 200, magnitudes
        )

    # A duplicate with the RIGHT total, which a length check alone would pass.
    per_face, survival = an_index(200)
    per_face[7] = {"stem": per_face[6]["stem"]}
    with pytest.raises(synthesis.SynthesisError, match="duplicate stem"):
        synthesis.assert_index_describes_arrays(per_face, survival, 200, magnitudes)

    # Survival adrift while faces.json is perfect.
    per_face, survival = an_index(200)
    with pytest.raises(synthesis.SynthesisError, match="survival has 601 rows"):
        synthesis.assert_index_describes_arrays(
            per_face, survival + [{"stem": "AF0"}], 200, magnitudes
        )


def test_the_survival_expectation_follows_the_magnitude_count():
    """Derived from the magnitudes, not hardcoded -- a build with a different
    series must not need this check edited to keep working."""
    per_face, survival = an_index(50, magnitudes=(0.0, 0.02))
    counts = synthesis.assert_index_describes_arrays(
        per_face, survival, 50, (0.0, 0.02)
    )
    assert counts["expected_survival_rows"] == 50


def test_the_two_gate_configs_differ_only_in_out_version(repo_root):
    """**The gate compares two artifacts for byte-identity, so the two configs
    that produce them must differ in exactly one field.**

    If they differed in anything else, a byte difference would have a second
    possible cause and the gate would not isolate the pause. Asserted rather than
    maintained by hand, in the same pattern as Phase 4's arms — each differing
    from its neighbour in exactly one field, checked by test.
    """
    import yaml

    a = yaml.safe_load(
        (repo_root / "configs" / "p5_synthesis_gate_a.yaml").read_text(
            encoding="utf-8"
        )
    )
    b = yaml.safe_load(
        (repo_root / "configs" / "p5_synthesis_gate_b.yaml").read_text(
            encoding="utf-8"
        )
    )

    assert a["task"]["out_version"] == "synth_gate_a"
    assert b["task"]["out_version"] == "synth_gate_b"

    a["task"] = {k: v for k, v in a["task"].items() if k != "out_version"}
    b["task"] = {k: v for k, v in b["task"].items() if k != "out_version"}
    assert a == b, (
        "the gate configs differ beyond out_version, so a byte difference "
        "between their artifacts would not isolate the pause"
    )


def test_the_gate_configs_produce_enough_checkpoints_to_exercise_the_resume(
    repo_root,
):
    """**A single checkpoint would barely exercise the journal truncation**, which
    is the part of the resume that can actually be wrong: rows appended after the
    last checkpoint have to be rolled back, and with one checkpoint there is
    almost nothing to roll back.

    Four is the floor here -- 200 faces at an interval of 50 -- so a pause between
    any two of them tests the same path the 5,499-face build uses.
    """
    import yaml

    gate = yaml.safe_load(
        (repo_root / "configs" / "p5_synthesis_gate_a.yaml").read_text(
            encoding="utf-8"
        )
    )["task"]
    checkpoints = gate["n_faces"] // gate["checkpoint_every"]
    assert checkpoints >= 4, (
        f"{checkpoints} checkpoint(s) over {gate['n_faces']} faces: too few to "
        "exercise the journal rollback the resume depends on"
    )

    # And the real build differs from the gate by CONFIGURATION, not by code:
    # same task kind, same magnitudes, same sampler, same geometries.
    real = yaml.safe_load(
        (repo_root / "configs" / "p5_asymmetry_synthesis.yaml").read_text(
            encoding="utf-8"
        )
    )["task"]
    for shared in ("kind", "magnitudes", "ar_sampling", "geometries"):
        assert gate[shared] == real[shared], (
            f"the gate and the artifact build differ in {shared!r}, so the gate "
            "does not verify the path the artifact build takes"
        )


# --------------------------------------------------------------------------
# parked: built and verified, not run
# --------------------------------------------------------------------------


def test_the_parked_status_records_why_no_result_was_available():
    """**[DECIDED 2026-07-30]** Stage F is struck. The reason is not "it looked
    unpromising" — it is that **neither outcome would have been interpretable**:
    a null was consistent with all four limitations at once, and a positive would
    have contradicted a prediction recorded in advance.

    That is the payoff of writing the prediction down first, one step earlier
    than usual — it bought a decision not to spend the time, rather than a
    defensible reading of a number afterwards.
    """
    parked = synthesis.PARKED
    assert parked["status"] == "BUILT AND VERIFIED, NOT RUN"
    assert "either direction" in parked["reason"]
    assert "does not earn cluster time" in parked["reason"]
    assert "Stage F" in parked["arm"]

    # The four, named in the parked record and not only in LIMITATIONS, so a
    # reader landing here does not have to go looking for them.
    for limitation in (
        "no_scar",
        "magnitude_to_grade_is_an_assumption",
        "midline_is_anchored",
        "the_arm_is_not_predicted_to_succeed",
    ):
        assert limitation in parked["the_four_limitations"]
        assert limitation in synthesis.LIMITATIONS


def test_the_parked_module_keeps_its_tests_running():
    """**The trap this avoids, learned from the parked segmentation module:** a
    parked module whose suite is skipped quietly stops working, and then the
    write-up chapter it evidences cannot be re-derived when someone asks.

    This test exists to make that policy explicit rather than implicit in the
    fact that nobody added a skipif — so removing the suite means deleting a test
    that says why it is there.
    """
    assert "quietly stops working" in synthesis.PARKED["tests_keep_running"]
    assert "do not extend it" in synthesis.__doc__.lower()
    # And the module is genuinely still exercised: this file is not skipped.
    assert synthesis.assert_index_describes_arrays is not None


def test_the_performance_issue_is_recorded_as_open_not_as_solved():
    """**A hypothesis recorded as a finding would be the R2 error in a new
    place.** Profiling settled that the build is flat in n and that the hot path
    is elementwise numpy rather than linear algebra; it did NOT settle why the
    cluster burns 99.98% of its CPU on overhead. The record has to keep those
    apart, and name the tests that would decide it.
    """
    issue = synthesis.CLUSTER_PERFORMANCE_UNRESOLVED
    assert issue["status"] == "OPEN, not a blocker"
    assert issue["local"]["flat_in_n"] is True
    assert "not compute bound" in issue["hot_path"]
    assert "did not reproduce" in issue["not_explained"]
    assert "hypothesis" in issue["leading_hypothesis"] or issue["leading_hypothesis"]
    assert "OPENBLAS_NUM_THREADS=1" in issue["untried_decisive_tests"]
    assert "utime" in issue["untried_decisive_tests"]
    assert "PARKED" in issue["why_not_a_blocker"]


def test_the_four_required_limitations_are_each_individually_findable():
    """**All four go in the artifact before the arm runs**, and each as its own
    entry rather than folded into a neighbour: no scar; the magnitude-to-grade
    mapping is an assumption not a measurement; the midline is anchored; and the
    deformation cannot reach the philtrum, ranked first at |Spearman| 0.151.

    ``midline_is_anchored`` was previously only implicit -- the reasoning lived
    inside ``the_arm_is_not_predicted_to_succeed``. A limitation that can only be
    found by reading another limitation is one a reader can miss.
    """
    limitations = synthesis.LIMITATIONS

    assert "DEFORMATION ONLY" in limitations["no_scar"]
    assert "no surgical scar" in limitations["no_scar"]

    mapping = limitations["magnitude_to_grade_is_an_assumption"]
    assert "NOTHING HERE ESTABLISHES" in mapping

    anchored = limitations["midline_is_anchored"]
    assert "TPS anchors" in anchored
    assert "FIXED AXIS" in anchored
    assert "cannot deviate the philtrum" in anchored

    philtrum = limitations["the_arm_is_not_predicted_to_succeed"]
    assert "cannot deform the philtrum" in philtrum
    assert "0.151" in philtrum
    assert "ranks FIRST of 22" in philtrum


def test_the_prediction_is_recorded_as_made_in_advance():
    """A null recorded before the run is a result; the same null recorded after
    is indistinguishable from explaining away a poor one. The record has to say
    which it is."""
    philtrum = synthesis.LIMITATIONS["the_arm_is_not_predicted_to_succeed"]
    assert "NOT PREDICTED TO SUCCEED" in philtrum
    assert "BEFORE the arm runs" in philtrum


def test_the_reasoned_direction_weights_are_declared_as_reasoned():
    """[REASONED] must not gate anything silently (PLAN provenance tagging)."""
    assert "NOT measured" in synthesis.LIMITATIONS["direction_weights_are_reasoned"]
    laterality = synthesis.LIMITATIONS["laterality_is_unrecorded_in_the_cleft_cohort"]
    assert "not recorded anywhere" in laterality
    assert "BALANCED" in laterality


def test_the_targets_are_the_four_from_the_decision():
    """Alar base, philtral column, cupid's bow peak, commissure. The philtral
    column has no landmark of its own, so it enters as the nostril sill -- the
    end of the column the cleft displaces -- and that substitution is stated."""
    assert set(synthesis.TARGETS) == {
        "alar_base", "nostril_sill", "cupids_bow_peak", "commissure",
    }
    assert set(synthesis.DIRECTIONS) == set(synthesis.TARGETS)
    assert set(synthesis.WEIGHTS) == set(synthesis.TARGETS)
    assert "no landmark of its own" in L.__doc__ or L.PHILTRAL_COLUMN_ENDPOINTS


# --------------------------------------------------------------------------
# unilateral, and the side is balanced and recorded
# --------------------------------------------------------------------------


def test_sides_are_exactly_balanced_not_balanced_in_expectation():
    """A per-face coin flip is balanced in expectation and not in fact. At n=25
    a fair flip lands 12/13 only about 31% of the time."""
    for n in (2, 24, 25, 100):
        balance = synthesis.side_balance(synthesis.assign_sides(n, seed=1337))
        assert balance["n"] == n
        assert balance["difference"] <= 1


def test_side_assignment_is_reproducible_from_the_seed():
    first = synthesis.assign_sides(50, seed=7)
    assert np.array_equal(first, synthesis.assign_sides(50, seed=7))
    assert not np.array_equal(first, synthesis.assign_sides(50, seed=8))


def test_the_two_sides_produce_mirrored_displacements():
    points = a_realistic_face()
    box = placement.crop_box(points, 0.74)
    _, _, left = synthesis.control_points(points, box, "left", 0.06)
    _, _, right = synthesis.control_points(points, box, "right", 0.06)
    for name in synthesis.TARGETS:
        lx, ly = left[name]["shift_frac_width"]
        rx, ry = right[name]["shift_frac_width"]
        assert lx == pytest.approx(-rx)
        assert ly == pytest.approx(ry), "only the horizontal component mirrors"


def test_an_unknown_side_is_refused():
    with pytest.raises(synthesis.SynthesisError, match="unknown side"):
        synthesis.target_indices("port")


def test_the_targets_are_the_measured_landmark_indices():
    """From the measured constants, not restated here -- a second copy of the
    index table is where the anatomy would silently diverge."""
    left = synthesis.target_indices("left")
    assert left["alar_base"] == L.ALAR_BASE[0]
    assert left["cupids_bow_peak"] == L.CUPIDS_BOW_PEAK[0]
    assert left["commissure"] == L.MOUTH_CORNERS[0]
    right = synthesis.target_indices("right")
    assert right["alar_base"] == L.ALAR_BASE[1]
    assert right["commissure"] == L.MOUTH_CORNERS[1]


# --------------------------------------------------------------------------
# the TPS itself
# --------------------------------------------------------------------------


def test_the_spline_interpolates_its_control_points():
    source = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0], [10.0, 10.0], [5.0, 5.0]])
    target = source.copy()
    target[4] += (1.5, -0.5)
    fitted = synthesis.fit_tps(source, target, smoothing=0.0)
    assert np.allclose(synthesis.apply_tps(fitted, source), target, atol=1e-8)


def test_an_identity_deformation_is_the_identity_map():
    source = np.array([[0.0, 0.0], [8.0, 1.0], [2.0, 9.0], [7.0, 8.0]])
    fitted = synthesis.fit_tps(source, source, smoothing=0.0)
    probe = np.array([[3.0, 4.0], [6.5, 2.5]])
    assert np.allclose(synthesis.apply_tps(fitted, probe), probe, atol=1e-8)


def test_too_few_or_mismatched_control_points_are_refused():
    with pytest.raises(synthesis.SynthesisError, match="at least 3"):
        synthesis.fit_tps(np.zeros((2, 2)), np.zeros((2, 2)))
    with pytest.raises(synthesis.SynthesisError, match="must both be"):
        synthesis.fit_tps(np.zeros((4, 2)), np.zeros((3, 2)))


def test_collinear_control_points_are_refused_not_silently_wrong():
    line = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    with pytest.raises(synthesis.SynthesisError):
        synthesis.fit_tps(line, line, smoothing=0.0)


def test_the_kernel_is_zero_at_zero_radius():
    """r^2 log r -> 0 as r -> 0, but log(0) is not 0; a nan on the diagonal would
    give a singular system with no clear error."""
    fitted = synthesis.fit_tps(
        np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
        np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
        smoothing=0.0,
    )
    assert np.isfinite(synthesis.apply_tps(fitted, np.array([[0.0, 0.0]]))).all()


# --------------------------------------------------------------------------
# LOCAL and UNILATERAL, as numbers
# --------------------------------------------------------------------------


def test_the_deformation_moves_the_targets_by_what_was_asked():
    image, points, box, side, magnitude = a_case()
    _, record = synthesis.warp_content(image, points, box, side, magnitude)
    fractions = record["deformation"]["realised_fraction_of_requested"]
    for name, value in fractions.items():
        assert value == pytest.approx(1.0, abs=0.02), (
            f"{name} moved {value:.3f} of the requested displacement"
        )


def test_the_frame_and_the_anchors_do_not_drift():
    """A TPS fitted to moved points alone warps the whole plane -- the result
    would be a reframed face rather than an asymmetric one."""
    image, points, box, side, magnitude = a_case()
    _, record = synthesis.warp_content(image, points, box, side, magnitude)
    corners = record["deformation"]["corner_motion_frac_width"]
    assert max(corners) < 1e-6, f"the crop corners drifted: {corners}"


def test_the_untouched_side_and_the_midline_do_not_move():
    """If they move, the deformation is not unilateral, and a mirror-difference
    index computed on it would be measuring a bilateral change."""
    image, points, box, side, magnitude = a_case()
    _, record = synthesis.warp_content(image, points, box, side, magnitude)
    deformation = record["deformation"]
    assert max(deformation["contralateral_motion_frac_width"]) < 1e-6
    assert max(deformation["midline_motion_frac_width"]) < 1e-6
    assert deformation["max_unintended_motion_frac_width"] < 1e-6


def test_the_motion_report_distinguishes_a_real_zero_from_no_measurement():
    """**A zero must be tellable from a quantity nobody computed** -- failure
    mode 5 in PLAN R7's tally, where two tests iterated an empty dict and passed
    asserting nothing.

    Two guards: the unintended-motion values are kept at full precision, so a
    genuine zero reads as ~1e-13 rather than a flat 0.0; and the same machinery
    reports real, large motion at the targets, which is the evidence that it is
    live rather than vacuous.
    """
    image, points, box, side, magnitude = a_case()
    _, record = synthesis.warp_content(image, points, box, side, magnitude)
    deformation = record["deformation"]

    # Live: the same report shows substantial motion where motion was requested.
    assert min(deformation["realised_px"].values()) > 0.1
    assert len(deformation["contralateral_motion_frac_width"]) == 4
    assert len(deformation["midline_motion_frac_width"]) == 2
    assert len(deformation["corner_motion_frac_width"]) == 4

    # The aggregate is the max OF THOSE LISTS, not a separately computed number
    # that could be a default. An aggregate over an empty set is how failure
    # mode 5 passed while asserting nothing.
    values = (
        deformation["contralateral_motion_frac_width"]
        + deformation["midline_motion_frac_width"]
        + deformation["corner_motion_frac_width"]
    )
    assert len(values) == 10
    assert deformation["max_unintended_motion_frac_width"] == max(values)


def test_the_locality_check_can_fail():
    """**The counterfactual.** Drop the anchors and the same displacements warp
    the whole plane -- so the assertions above are testing something."""
    points = a_realistic_face(midline=200.0, brow_y=110.0, span=170.0)
    box = placement.crop_box(points, 0.74)
    source, target, _ = synthesis.control_points(points, box, "left", 0.06)
    # The four moved points only: the first four rows, before the anchors.
    unanchored = synthesis.fit_tps(source[:4], target[:4])

    x, y, w, h = box
    corners = np.array([[x, y], [x + w, y], [x, y + h], [x + w, y + h]])
    drift = np.abs(synthesis.apply_tps(unanchored, corners) - corners).max() / w
    assert drift > 0.01, (
        "without anchors the frame must drift, or the anchor assertions above "
        "would pass for the wrong reason"
    )


def test_no_landmark_outside_the_four_targets_moves():
    """**THE regression test for a deformation that was not local [MEASURED
    2026-07-30].**

    The first anchor set was ten points -- the crop boundary, the four
    contralateral targets, the two midline landmarks -- and it reported
    "unintended motion 1e-10" while **48 of 86 landmarks were in fact moving,
    including both eyes, three brow points and eleven face-outline points.** The
    thin-plate kernel is globally supported and decays slowly; ten points do not
    confine it. And the check could not see it, because every point it measured
    was an anchor: failure mode 2 in PLAN R7's tally, a criterion satisfiable by
    construction.

    This asserts over ALL 86 landmarks, so the set being measured is not the set
    being pinned by assumption.
    """
    points = a_realistic_face(midline=200.0, brow_y=110.0, span=170.0)
    box = placement.crop_box(points, 0.74)
    moved = synthesis.warp_landmarks(points, box, "left", 0.09)

    targets = set(synthesis.target_indices("left").values())
    displaced = {
        index
        for index in range(len(points))
        if float(np.hypot(*(moved[index] - points[index]))) > 0.5
    }
    assert displaced == targets, (
        f"landmarks outside the four targets moved: "
        f"{sorted(displaced - targets)}"
    )

    # Named explicitly, because these are the ones that moved before and the
    # ones whose motion would be least defensible: asymmetric eyes are not cleft
    # morphology and would hand the model a cue nothing clinical supports.
    for index in list(placement.EYE_LEFT) + list(placement.EYE_RIGHT) + list(
        L.LANDMARK_GROUPS["brows"]
    ):
        assert float(np.hypot(*(moved[index] - points[index]))) < 1e-6, (
            f"landmark {index} (an eye or brow point) moved"
        )


def test_the_deformation_stays_below_the_eyes_in_the_pixels():
    """**Locality measured on the IMAGE, which is the measurement that can
    fail.** Every landmark except the four targets is now an anchor, so a
    landmark-based locality check re-measures anchors and passes by construction.

    The eyes sit near 0.23 of the crop and the alar bases near 0.60, so a
    nasolabial deformation must not change pixels much above ~0.35.
    """
    image, points, box, side, magnitude = a_case(magnitude=0.09)
    _, record = synthesis.warp_content(image, points, box, side, magnitude)
    change = record["deformation"]["pixel_change"]

    assert change["n_changed_pixels"] > 0, "the deformation changed nothing"
    assert change["outside_box"] is False, (
        "pixels changed outside the crop box; the warp is not confined to it"
    )
    assert change["y_min"] > 0.35, (
        f"pixel change reaches y={change['y_min']:.3f} of the crop, above the "
        "nasolabial region -- the eyes sit near 0.23"
    )
    assert change["y_max"] <= 1.0


def test_the_pixel_warp_lands_where_the_landmarks_moved():
    """**The regression test for the coordinate-frame slip [MEASURED
    2026-07-30].**

    ``warp_content`` was named for the cropped content and offset its sampling
    grid by the box origin, while every call site passed the full source image.
    The deformation was therefore applied about ``(x0, y0)`` away from the anatomy
    it was fitted to: measured, the pixel-change centroid sat **31 x 47 pixels**
    off the moved-landmark centroid, and the change spanned the whole 400x400
    source instead of the 126x170 crop.

    Both frames are source pixel coordinates now, so the two centroids must
    agree to within a fraction of the crop.
    """
    image, points, box, side, magnitude = a_case(magnitude=0.09)
    warped, record = synthesis.warp_content(image, points, box, side, magnitude)
    x, y, width, height = box

    difference = np.abs(image.astype(int) - warped.astype(int)).sum(axis=2)
    ys, xs = np.nonzero(difference > 10)
    pixel_centroid = ((xs.mean() - x) / width, (ys.mean() - y) / height)

    targets = list(synthesis.target_indices(side).values())
    landmark_centroid = (
        float((points[targets, 0].mean() - x) / width),
        float((points[targets, 1].mean() - y) / height),
    )

    assert abs(pixel_centroid[0] - landmark_centroid[0]) < 0.20, (
        f"pixel change centred at x={pixel_centroid[0]:.3f} of the crop but the "
        f"targets are at x={landmark_centroid[0]:.3f}"
    )
    assert abs(pixel_centroid[1] - landmark_centroid[1]) < 0.20, (
        f"pixel change centred at y={pixel_centroid[1]:.3f} of the crop but the "
        f"targets are at y={landmark_centroid[1]:.3f}"
    )
    # And it agrees with what the record reports, so metrics.json is not a
    # separately computed number that could drift from the pixels.
    change = record["deformation"]["pixel_change"]
    assert change["centroid_x"] == pytest.approx(pixel_centroid[0], abs=1e-6)
    assert change["centroid_y"] == pytest.approx(pixel_centroid[1], abs=1e-6)


def test_pixels_outside_the_crop_box_are_untouched():
    """The warp rewrites only inside the box, so locality in the pixels is a
    property of the loop bound rather than of the spline decaying."""
    image, points, box, side, magnitude = a_case(magnitude=0.09)
    warped, _ = synthesis.warp_content(image, points, box, side, magnitude)
    x, y, width, height = box

    left = max(int(np.floor(x)), 0)
    top = max(int(np.floor(y)), 0)
    right = min(int(np.ceil(x + width)), image.shape[1])
    bottom = min(int(np.ceil(y + height)), image.shape[0])

    outside = np.ones(image.shape[:2], dtype=bool)
    outside[top:bottom, left:right] = False
    assert np.array_equal(image[outside], warped[outside])
    assert not np.array_equal(
        image[top:bottom, left:right], warped[top:bottom, left:right]
    )


def test_the_non_target_motion_field_is_labelled_as_by_construction():
    """It is reported for consistency, never as evidence of locality -- and the
    record must say so, because that is exactly how the first version misled."""
    image, points, box, side, magnitude = a_case()
    _, record = synthesis.warp_content(image, points, box, side, magnitude)
    deformation = record["deformation"]

    assert deformation["n_non_target_landmarks"] == len(points) - len(
        synthesis.TARGETS
    )
    assert deformation["max_non_target_motion_frac_width"] < 1e-6
    explanation = deformation["non_target_motion_is_zero_by_construction"]
    assert "NOT evidence" in explanation
    assert "48 of 86" in explanation


def test_a_larger_magnitude_deforms_more_and_zero_deforms_not_at_all():
    image, points, box, side, _ = a_case()
    plain, plain_record = synthesis.warp_content(image, points, box, side, 0.0)
    small, _ = synthesis.warp_content(image, points, box, side, 0.03)
    large, _ = synthesis.warp_content(image, points, box, side, 0.09)

    assert plain_record["identity"] is True
    assert np.array_equal(plain, image), "magnitude 0 is the reference, exactly"

    small_change = float(np.abs(small.astype(int) - image.astype(int)).mean())
    large_change = float(np.abs(large.astype(int) - image.astype(int)).mean())
    assert 0 < small_change < large_change


def test_the_usable_magnitude_range_is_bounded_by_anchor_spacing():
    """**[MEASURED 2026-07-30] The upper end of the series is a measurement, not
    a preference.**

    A TPS pinned at nearby points cannot move a target far without creasing.
    ``displacement / nearest anchor`` scales at about 13.2x the magnitude and the
    cupid's bow peak binds first -- its neighbours are the anchored midline dip
    ~0.09 crop widths away and the nostril sill. **On real faces the ratio
    reaches 1.19 at magnitude 0.09** (worst: AF1006): the peak travels further
    than its nearest anchor is away, and the vermillion folds into a chevron.
    **Confirmed as the field and not the resampling: the shape is identical under
    bilinear sampling.**

    This fixture's landmark spacing is not the cohort's, so it reaches 0.90 at
    0.09 rather than 1.19. What is asserted here is the shape of the relationship
    and that it clears the ceiling; the 1.19 figure belongs to the real faces and
    is recorded where it was measured.
    """
    image, points, box, side, _ = a_case()
    ratios = {}
    for magnitude in (0.03, 0.06, 0.09):
        _, record = synthesis.warp_content(image, points, box, side, magnitude)
        ratios[magnitude] = record["deformation"][
            "max_displacement_to_anchor_ratio"
        ]

    # Monotone in magnitude, and it crosses the ceiling within the swept range.
    assert ratios[0.03] < ratios[0.06] < ratios[0.09]
    assert ratios[0.03] < synthesis.MAX_DISPLACEMENT_TO_ANCHOR_RATIO
    assert ratios[0.09] > synthesis.MAX_DISPLACEMENT_TO_ANCHOR_RATIO, (
        "magnitude 0.09 must clear the ceiling, or the mechanism behind the "
        "chevron on the sheet is not what was measured"
    )
    # Linear in magnitude: the ratio is a displacement over a fixed distance.
    assert ratios[0.09] / ratios[0.03] == pytest.approx(3.0, rel=0.05)

    _, record = synthesis.warp_content(image, points, box, side, 0.09)
    binding = max(
        record["deformation"]["displacement_to_anchor_spacing"].items(),
        key=lambda kv: kv[1]["ratio"],
    )[0]
    assert binding == "cupids_bow_peak", (
        f"the binding target is {binding}, not the cupid's bow peak -- the "
        "recorded mechanism names the peak and its midline-dip neighbour"
    )


def test_the_default_magnitude_series_stays_inside_the_measured_bound():
    """The shipped default must not produce the artefact. Read from the SCHEMA,
    so the config and the bound cannot drift apart silently."""
    from cleft.config.schema import TASK_SPECS

    default = TASK_SPECS["asymmetry_synthesis"]["magnitudes"].default
    assert 0.0 in default, "the undeformed reference must be present"

    image, points, box, side, _ = a_case()
    for magnitude in default:
        if magnitude == 0.0:
            continue
        _, record = synthesis.warp_content(image, points, box, side, magnitude)
        ratio = record["deformation"]["max_displacement_to_anchor_ratio"]
        assert ratio < synthesis.MAX_DISPLACEMENT_TO_ANCHOR_RATIO, (
            f"default magnitude {magnitude} gives ratio {ratio:.3f}, at or over "
            f"the measured ceiling {synthesis.MAX_DISPLACEMENT_TO_ANCHOR_RATIO}"
        )


def test_the_anchor_spacing_bound_is_recorded_as_an_open_decision():
    """The bound comes from the anchor set, so it is a design question -- and in
    a real unilateral cleft the philtrum IS deviated. Recorded, not resolved."""
    limitation = synthesis.LIMITATIONS[
        "magnitude_range_is_bounded_by_anchor_spacing"
    ]
    assert "ANCHOR SET" in limitation
    assert "OPEN DECISION" in limitation
    assert "CHANGE WHAT THE INSTRUMENT MEASURES" in limitation


def test_a_negative_magnitude_is_refused():
    image, points, box, side, _ = a_case()
    with pytest.raises(synthesis.SynthesisError, match="non-negative"):
        synthesis.warp_content(image, points, box, side, -0.05)


def test_the_warp_is_deterministic():
    image, points, box, side, magnitude = a_case()
    first, _ = synthesis.warp_content(image, points, box, side, magnitude)
    second, _ = synthesis.warp_content(image, points, box, side, magnitude)
    assert np.array_equal(first, second)


def test_the_output_keeps_the_shape_and_dtype_of_its_input():
    image, points, box, side, magnitude = a_case()
    warped, _ = synthesis.warp_content(image, points, box, side, magnitude)
    assert warped.shape == image.shape
    assert warped.dtype == image.dtype


# --------------------------------------------------------------------------
# the landmarks move with the image
# --------------------------------------------------------------------------


def test_the_landmarks_move_with_the_deformation():
    """A synthesised face whose landmarks were not moved with it would be
    measured against the geometry of the face it used to be."""
    points = a_realistic_face(midline=200.0, brow_y=110.0, span=170.0)
    box = placement.crop_box(points, 0.74)
    moved = synthesis.warp_landmarks(points, box, "left", 0.06)

    indices = synthesis.target_indices("left")
    for name, index in indices.items():
        shift = float(np.hypot(*(moved[index] - points[index])))
        assert shift > 0.5, f"{name} did not move"

    other = synthesis.target_indices("right")
    for name, index in other.items():
        shift = float(np.hypot(*(moved[index] - points[index])))
        assert shift < 1e-6, f"the contralateral {name} moved by {shift}"


def test_zero_magnitude_leaves_the_landmarks_exactly_alone():
    points = a_realistic_face()
    box = placement.crop_box(points, 0.74)
    assert np.array_equal(synthesis.warp_landmarks(points, box, "left", 0.0), points)


def test_the_deformation_displaces_the_measured_midline_of_the_face():
    """The synthesis must produce something the asymmetry instruments can see:
    after a unilateral deformation the mirror-pair asymmetry must RISE, since the
    pairs on one side moved and their partners did not."""
    points = a_realistic_face(midline=200.0, brow_y=110.0, span=170.0)
    box = placement.crop_box(points, 0.74)
    before = placement.mirror_pair_asymmetry(points)["mean_asymmetry_frac_width"]
    moved = synthesis.warp_landmarks(points, box, "left", 0.06)
    after = placement.mirror_pair_asymmetry(moved)["mean_asymmetry_frac_width"]
    assert after > before * 2


# --------------------------------------------------------------------------
# does the asymmetry survive G2? measured on pixels
# --------------------------------------------------------------------------


def test_the_asymmetry_index_is_zero_on_a_mirror_symmetric_image():
    symmetric = np.zeros((40, 40, 3), dtype=np.uint8)
    symmetric[10:30, 5:15] = 200
    symmetric[10:30, 25:35] = 200
    assert synthesis.asymmetry_index(symmetric) == pytest.approx(0.0, abs=1e-12)

    lopsided = symmetric.copy()
    lopsided[10:30, 25:35] = 40
    assert synthesis.asymmetry_index(lopsided) > 0.05


def test_the_synthesised_asymmetry_survives_the_g2_unwarp():
    """**[MEASURED] The empirical companion to the analytic gate.**

    ``trapezium.asymmetry_is_preserved`` tests the row stretch on displaced point
    pairs. This tests the stretch composed with a spline warp and a
    nearest-neighbour resample -- a different quantity (R2). If the introduced
    asymmetry collapsed here, G2 would not be usable for this arm.
    """
    from cleft.geometry.staging import stage
    from cleft.scut import masked

    points, box = a_case()[1], a_case()[2]
    image = a_symmetric_textured_face()
    content = masked.crop_content(image, box)
    warped, _ = synthesis.warp_content(image, points, box, "left", 0.06)
    warped_content = masked.crop_content(warped, box)

    plain = stage(masked.apply_arrival_mask(content)).image
    deformed = stage(masked.apply_arrival_mask(warped_content)).image

    report = synthesis.synthesised_asymmetry_survives(plain, deformed)
    assert report["introduced_g1"] > 0, "the deformation introduced no asymmetry"
    assert report["introduced_g2"] > 0, "G2 destroyed the synthesised asymmetry"
    assert report["survival_ratio"] > 0.5, (
        f"only {report['survival_ratio']:.2f} of the introduced asymmetry "
        "survived unwarping"
    )
    # **The floor is recorded, not glossed [MEASURED 2026-07-30].** Even on an
    # exactly mirror-symmetric input the frozen path's nearest-neighbour
    # resampling leaves ~0.0042 of mirror difference, against ~0.0011 introduced
    # here -- so this ratio rests on the floor cancelling in the subtraction, and
    # `baseline_dominates` says so. `landmark_asymmetry_survives` is the primary
    # measurement precisely because it has no floor.
    assert report["baseline_dominates"] is False, (
        "on a symmetric base the introduced asymmetry (~0.0011) is 27% of the "
        "resampling floor (~0.0042) -- above the flag's threshold, and the ratio "
        "comes out sensible at ~0.93"
    )
    assert report["g1"]["plain"] > report["introduced_g1"], (
        "if this ever reverses, the resampling floor has changed and the note in "
        "synthesis.py about which measurement is primary should be revisited"
    )


def test_the_survival_check_is_useless_on_a_randomly_textured_face():
    """**The negative control for the fixture, not for the code.**

    On a randomly textured face the whole-frame mirror difference is dominated by
    the texture, so the introduced asymmetry is a rounding error against it and
    ``survival_ratio`` means nothing. Asserted so the next person to reach for the
    convenient fixture finds this instead of a mysterious ratio -- and so
    ``baseline_dominates`` is exercised in the state it exists to flag.
    """
    from cleft.geometry.staging import stage
    from cleft.scut import masked

    points, box = a_case()[1], a_case()[2]
    image = a_textured_face()
    plain = stage(masked.apply_arrival_mask(masked.crop_content(image, box))).image
    warped, _ = synthesis.warp_content(image, points, box, "left", 0.06)
    deformed = stage(
        masked.apply_arrival_mask(masked.crop_content(warped, box))
    ).image

    report = synthesis.synthesised_asymmetry_survives(plain, deformed)
    assert report["baseline_dominates"] is True
    assert abs(report["introduced_g1"]) < 0.05 * report["g1"]["plain"]


def test_the_landmark_survival_measurement_is_exact_and_has_no_floor():
    """**The PRIMARY survival measurement.** Analytic, so a symmetric face gives
    exactly zero introduced asymmetry and any nonzero value is the deformation.

    ``asymmetry_is_preserved`` gates the row stretch on arbitrary offsets; this
    runs the actual synthesised displacement pattern through it -- four targets at
    their measured anatomical heights, one side only.
    """
    points = a_realistic_face(midline=200.0, brow_y=110.0, span=170.0)
    box = placement.crop_box(points, 0.74)

    plain = synthesis.landmark_asymmetry_survives(points, box, "left", 0.0)
    for name, values in plain["per_target"].items():
        assert values["introduced_g1"] == pytest.approx(0.0, abs=1e-12), (
            f"a zero-magnitude deformation introduced asymmetry at {name}"
        )

    report = synthesis.landmark_asymmetry_survives(points, box, "left", 0.06)
    for name, values in report["per_target"].items():
        assert values["introduced_g1"] > 0, f"{name} introduced no asymmetry"
        assert values["introduced_g2"] > 0, f"G2 destroyed {name}'s asymmetry"
    # All four targets survive, and G2 magnifies rather than damps: 1.14-1.27.
    assert report["n_targets_measured"] == len(synthesis.TARGETS)
    assert report["min_survival_ratio"] > 1.0


def test_the_signed_asymmetry_is_stable_across_sides_where_absolute_was_not():
    """**[MEASURED 2026-07-30] Why the pair difference is SIGNED.**

    With an absolute value, a displacement acting against the face's existing
    asymmetry cancels it before adding, so ``introduced`` becomes a difference of
    near-equal quantities. The commissure -- the deliberately smallest of the four
    displacements, about a tenth of the alar base's -- then measured **0.34
    deforming left against 1.15 deforming right, from the same rule at the same
    magnitude**. Read literally that says G2 destroys two thirds of the
    asymmetry, and it is an artefact of the absolute value, not a property of the
    geometry.

    Signed differences add linearly, so the same displacement contributes the
    same amount whichever way the face was already leaning. Asserted at the
    commissure, where the artefact was largest.
    """
    points = a_realistic_face(midline=200.0, brow_y=110.0, span=170.0)
    box = placement.crop_box(points, 0.74)
    left = synthesis.landmark_asymmetry_survives(points, box, "left", 0.06)
    right = synthesis.landmark_asymmetry_survives(points, box, "right", 0.06)

    for name in synthesis.TARGETS:
        assert left["per_target"][name]["introduced_g1"] == pytest.approx(
            right["per_target"][name]["introduced_g1"], rel=0.05
        ), f"{name}'s introduced asymmetry still depends on which side deformed"
        assert left["per_target"][name]["survival_ratio"] == pytest.approx(
            right["per_target"][name]["survival_ratio"], rel=0.10
        ), f"{name}'s survival ratio still depends on which side deformed"

    assert left["min_survival_ratio"] > 1.0
    assert "SIGNED" in left["note"]


def test_the_absolute_formulation_would_have_been_unstable():
    """**The counterfactual for the test above.** Recomputing the commissure with
    ``abs`` around the pair difference must reproduce the instability, or the
    signed formulation is fixing nothing and the note in ``synthesis.py`` is
    describing a problem that does not exist.
    """
    points = a_realistic_face(midline=200.0, brow_y=110.0, span=170.0)
    box = placement.crop_box(points, 0.74)

    ratios = {}
    for side in ("left", "right"):
        report = synthesis.landmark_asymmetry_survives(points, box, side, 0.06)
        commissure = report["per_target"]["commissure"]
        plain = commissure["plain_signed_asymmetry_g1"]
        # The absolute-value version, reconstructed from the signed quantities.
        absolute_plain_g1 = abs(plain)
        absolute_deformed_g1 = abs(commissure["signed_asymmetry_g1"])
        introduced_abs = absolute_deformed_g1 - absolute_plain_g1
        ratios[side] = introduced_abs

    # Under abs the two sides disagree substantially; under signed they do not.
    assert ratios["left"] != pytest.approx(ratios["right"], rel=0.10), (
        "the absolute formulation is expected to differ across sides at the "
        "commissure -- if it no longer does, the fixture changed"
    )


def test_g2_magnifies_rather_than_destroys_the_displacement():
    """**The direction of the effect, stated so a ratio above 1 is not read as a
    defect.** Each row is stretched by ``1/(2*half_width)`` -- 1.66 at the brow,
    1.0 at the lip -- so unwarping AMPLIFIES a horizontal displacement. Every
    target must land in that band, above 1 and below the brow's factor.

    **Not asserted: that a higher target survives by more.** The obvious version
    of this test ordered the targets by height and failed, for a reason worth
    recording. The ratio is not the magnification at the target's original row:
    the deformation moves each target VERTICALLY as well, which changes the
    factor its offset is magnified by, and the two effects partly cancel. The
    alar base moves down (its factor falls) while the commissure moves up (its
    rises), so the two come out within 0.0002 of each other despite sitting 0.21
    of the frame apart. Height alone does not order the ratios.
    """
    from cleft.geometry.trapezium import DEFAULT as TRAPEZIUM

    points = a_realistic_face(midline=200.0, brow_y=110.0, span=170.0)
    box = placement.crop_box(points, 0.74)
    report = synthesis.landmark_asymmetry_survives(points, box, "left", 0.06)

    brow_factor = 1.0 / (2.0 * TRAPEZIUM.half_width_at(0.0))
    for name, values in report["per_target"].items():
        assert 1.0 < values["survival_ratio"] <= brow_factor, (
            f"{name} survived at {values['survival_ratio']}, outside the row "
            f"stretch band (1.0, {brow_factor:.3f}]"
        )
    assert "MAGNIFIES" in report["note"]


def test_the_two_sides_are_deformed_by_equal_amounts():
    """A left and a right deformation at the same magnitude must **displace by
    the same distance** -- otherwise 'balanced sides' would not balance anything.

    Measured on the realised displacement, not on the induced pair asymmetry: the
    latter interacts with whatever asymmetry the face already had, which is a
    property of the face rather than of the rule (the commissure test above is
    that interaction in action).
    """
    image, points, box, _, magnitude = a_case()
    _, left = synthesis.warp_content(image, points, box, "left", magnitude)
    _, right = synthesis.warp_content(image, points, box, "right", magnitude)

    for name in synthesis.TARGETS:
        assert left["deformation"]["realised_px"][name] == pytest.approx(
            right["deformation"]["realised_px"][name], rel=0.02
        ), f"{name} is displaced by different distances on the two sides"


def test_a_degenerate_box_is_refused():
    points = a_realistic_face()
    with pytest.raises(synthesis.SynthesisError, match="non-positive"):
        synthesis.landmark_asymmetry_survives(points, (0.0, 0.0, 0.0, 10.0), "left", 0.05)


def test_the_survival_check_reports_both_geometries_and_names_the_quantity():
    plain = np.zeros((32, 32, 3), dtype=np.uint8)
    plain[8:24, 4:28] = 180
    deformed = plain.copy()
    deformed[8:24, 4:12] = 60

    report = synthesis.synthesised_asymmetry_survives(plain, deformed)
    assert set(report) >= {"g1", "g2", "introduced_g1", "introduced_g2",
                          "survival_ratio", "note"}
    assert "different quantity" in report["note"]


def test_the_survival_ratio_is_none_when_nothing_was_introduced():
    """A ratio computed from a zero denominator would be a number with no
    meaning; it is reported as absent instead."""
    plain = np.zeros((24, 24, 3), dtype=np.uint8)
    plain[6:18, 4:20] = 150
    report = synthesis.synthesised_asymmetry_survives(plain, plain)
    assert report["introduced_g1"] == 0.0
    assert report["survival_ratio"] is None
