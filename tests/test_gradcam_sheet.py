"""The Grad-CAM contact sheet.

**The sheet itself is the check no test substitutes for** -- whether the maps
land on plausible anatomy is decided by eye or not at all. What IS testable is
that the sheet shows the things that make that judgement possible: the grid
that makes the resolution floor visible, and the concentration figures that
survive normalisation.
"""

from __future__ import annotations

import numpy as np
import pytest

from cleft import gradcam_sheet, phase8

GRID = phase8.GRAD_CAM_TARGET["grid"]
N_TOKENS = GRID[0] * GRID[1]
SIZE = tuple(phase8.RESOLUTION_FLOOR["image"])


def _crop(seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(*SIZE, 3), dtype=np.uint8)


def _maps(concentrated: bool, seed=1):
    rng = np.random.default_rng(seed)
    grid = rng.random(N_TOKENS) * 0.01
    if concentrated:
        # Peak deliberately NOT 1.0: a fixture that is already normalised
        # cannot exercise the before-versus-after distinction below.
        grid[:5] = 7.0
    else:
        grid[:] = 0.5 + rng.random(N_TOKENS) * 0.01
    grid = grid.reshape(GRID)
    from cleft import gradcam

    return grid, gradcam.upsample(gradcam.normalise(grid), SIZE)


# --------------------------------------------------------------------------
# the scale -- what normalisation destroys
# --------------------------------------------------------------------------


def test_concentration_separates_a_peaked_map_from_a_diffuse_one():
    """**Both arrive stretched to full range and look equally decisive.**
    That is what the figures beside the panel are for."""
    peaked, _ = _maps(concentrated=True)
    diffuse, _ = _maps(concentrated=False)
    top = f"top_{gradcam_sheet.CONCENTRATION_AT[0]}_share"
    assert gradcam_sheet.concentration(peaked)[top] > (
        gradcam_sheet.concentration(diffuse)[top]
    )


def test_concentration_is_computed_before_normalisation():
    """Dividing by the maximum destroys exactly the quantity being reported,
    so the same map normalised must give the same shares -- ratios are
    scale-free -- while the PEAK must not be forced to 1."""
    from cleft import gradcam

    peaked, _ = _maps(concentrated=True)
    raw = gradcam_sheet.concentration(peaked)
    scaled = gradcam_sheet.concentration(gradcam.normalise(peaked))
    top = f"top_{gradcam_sheet.CONCENTRATION_AT[0]}_share"
    assert raw[top] == pytest.approx(scaled[top])
    assert raw["peak"] != pytest.approx(1.0), (
        "the fixture is already normalised, so this test cannot see the "
        "difference it exists to check"
    )


def test_two_depths_are_reported_because_one_cannot_separate_the_cases():
    """A map with 60% in its top 10 is concentrated; one with 60% in its top
    50 is not. Both look identical once normalised."""
    assert len(gradcam_sheet.CONCENTRATION_AT) == 2
    peaked, _ = _maps(concentrated=True)
    report = gradcam_sheet.concentration(peaked)
    shallow, deep = gradcam_sheet.CONCENTRATION_AT
    assert report[f"top_{shallow}_share"] <= report[f"top_{deep}_share"]


def test_a_map_with_no_mass_is_refused():
    with pytest.raises(ValueError, match="no positive mass"):
        gradcam_sheet.concentration(np.zeros(GRID))


def test_excess_over_uniform_is_zero_for_a_flat_map():
    flat = np.full(GRID, 0.3)
    assert gradcam_sheet.concentration(flat)["excess_over_uniform"] == (
        pytest.approx(0.0, abs=1e-12)
    )


# --------------------------------------------------------------------------
# the grid -- the resolution floor, drawn
# --------------------------------------------------------------------------


def test_the_token_grid_is_drawn_on_the_overlay():
    """**Not decoration.** A reviewer studying a warm patch over the philtral
    column needs the floor in view while looking, not in a caption read
    earlier."""
    crop = np.zeros((*SIZE, 3), dtype=np.uint8)
    drawn = gradcam_sheet.draw_token_grid(crop)
    rows, cols = GRID
    # One line per internal boundary, in both directions.
    assert (drawn[int(round(SIZE[0] / rows)), :, 0] == gradcam_sheet.GRID_VALUE).all()
    assert (drawn[:, int(round(SIZE[1] / cols)), 0] == gradcam_sheet.GRID_VALUE).all()
    assert (drawn[1, 1] == 0).all(), "the grid should not fill the image"


def test_the_grid_matches_the_layer_the_maps_come_from():
    """A grid drawn at a different resolution than the tokens would misstate
    the floor in the one place a reviewer can see it."""
    crop = np.zeros((*SIZE, 3), dtype=np.uint8)
    drawn = gradcam_sheet.draw_token_grid(crop)
    lines = sum(
        1 for r in range(SIZE[0])
        if (drawn[r, :, 0] == gradcam_sheet.GRID_VALUE).all()
    )
    assert lines == GRID[0] - 1
    assert GRID == tuple(phase8.RESOLUTION_FLOOR["token_grid"])


def test_a_non_rgb_image_is_refused():
    with pytest.raises(ValueError, match="RGB"):
        gradcam_sheet.draw_token_grid(np.zeros(SIZE, dtype=np.uint8))


# --------------------------------------------------------------------------
# the overlay and the sheet
# --------------------------------------------------------------------------


def test_the_overlay_lifts_red_where_the_map_is_warm():
    crop = np.full((*SIZE, 3), 40, dtype=np.uint8)
    hot = np.ones(SIZE)
    cold = np.zeros(SIZE)
    assert gradcam_sheet.overlay(crop, hot)[..., 0].mean() > 40
    assert gradcam_sheet.overlay(crop, cold)[..., 0].mean() == pytest.approx(40)


def test_a_mismatched_map_is_refused():
    with pytest.raises(ValueError, match="does not match crop"):
        gradcam_sheet.overlay(np.zeros((*SIZE, 3), np.uint8), np.zeros((8, 8)))


def test_each_patient_gets_crop_overlay_and_scale():
    grid, image = _maps(concentrated=True)
    entry = gradcam_sheet.per_patient(42, _crop(), grid, image,
                                      seed_agreement=[0.8, 0.7, 0.9])
    assert len(entry["panels"]) == 3
    labels = [panel.label for panel in entry["panels"]]
    assert "crop" in labels[0] and "map" in labels[1] and "scale" in labels[2]
    # The concentration is IN the label, so it is read beside the picture.
    assert "%" in labels[1]
    assert entry["report"]["seed_agreement_median"] == pytest.approx(0.8)
    assert entry["report"]["patient_id"] == 42


def test_the_sheet_tiles_one_patient_per_row():
    entries = [
        gradcam_sheet.per_patient(i, _crop(i), *_maps(concentrated=True, seed=i))
        for i in range(3)
    ]
    sheet = gradcam_sheet.build_sheet(entries, columns=3)
    assert sheet.ndim == 3 and sheet.shape[2] == 3
    assert sheet.shape[0] > SIZE[0] * 2, "three rows should be taller than two"


def test_an_empty_sheet_is_refused():
    with pytest.raises(ValueError, match="no patients"):
        gradcam_sheet.build_sheet([])


# --------------------------------------------------------------------------
# the summary
# --------------------------------------------------------------------------


def test_the_summary_flags_maps_that_are_not_localising_anything():
    """**A map at or below the uniform share is not localising**, however
    decisive it looks once normalised."""
    entries = [
        gradcam_sheet.per_patient(0, _crop(), *_maps(concentrated=False)),
        gradcam_sheet.per_patient(1, _crop(1), *_maps(concentrated=True, seed=2)),
    ]
    summary = gradcam_sheet.summarise(entries)
    assert summary["n_patients"] == 2
    assert summary["uniform_top_k_share"] == pytest.approx(
        gradcam_sheet.CONCENTRATION_AT[0] / N_TOKENS
    )
    assert summary["n_at_or_below_uniform"] >= 0
    assert summary["resolution_floor_px"] == 16


def test_the_review_questions_are_written_down():
    """"Does it look right" is not a reviewable question, and every previous
    sheet in this project needed the same discipline."""
    questions = gradcam_sheet.REVIEW_QUESTIONS
    assert len(questions) == 4
    joined = " ".join(questions)
    assert "white pad" in joined
    assert "NOT the picture" in joined
    assert "interpolation kernel" in joined
    assert "not the mean map" in joined


def test_the_summary_carries_the_frozen_backbone_caveat():
    entries = [gradcam_sheet.per_patient(0, _crop(), *_maps(concentrated=True))]
    summary = gradcam_sheet.summarise(entries)
    assert "NOT what the model learned about clefts" in summary["caveat"]


# --------------------------------------------------------------------------
# what "publish" scopes over
# --------------------------------------------------------------------------


def test_the_banner_carries_the_numbers_not_just_the_word_failed():
    """**"FAILED" alone invites discounting all fifteen** when thirteen sit
    strongly below the baseline, and the numbers are what distinguish a
    survivor at the 91st percentile from one at the 62nd."""
    verdict = {
        "passes": False, "survivors": [0, 1], "n_patients": 15,
        "baseline": 0.0868,
        "baseline_distribution": {"sd": 0.1586, "min": -0.395, "max": 0.438},
        "separability": {"observed_difference": -0.1061,
                         "ci95": [-0.2047, -0.0020]},
    }
    banner = gradcam_sheet.verdict_banner(verdict)
    assert "FAILED" in banner and "NOT PUBLISHED" in banner
    assert "2 of 15" in banner
    assert "0.1586" in banner and "-0.1061" in banner
    assert "-0.2047" in banner


def test_the_banner_reports_a_pass_too():
    banner = gradcam_sheet.verdict_banner(
        {"passes": True, "survivors": [], "n_patients": 15, "baseline": 0.05}
    )
    assert "PASSED" in banner
    assert "NOT PUBLISHED" not in banner


def test_a_survivor_is_marked_on_its_own_panels():
    """**A reviewer studying patient 77 should know it is a survivor while
    looking at it**, not from a header read earlier -- the same reason the
    token grid is drawn on the map rather than described in a caption."""
    grid, image = _maps(concentrated=True)
    marked = gradcam_sheet.per_patient(77, _crop(), grid, image, survivor=0.91)
    plain = gradcam_sheet.per_patient(190, _crop(1), grid, image)
    assert all("SURVIVED RANDOMISATION" in p.label for p in marked["panels"][:2])
    assert marked["report"]["survived_randomisation"] is True
    assert marked["report"]["baseline_percentile"] == pytest.approx(0.91)
    assert not any("SURVIVED" in p.label for p in plain["panels"])
    assert "survived_randomisation" not in plain["report"]


def test_the_percentile_is_in_the_mark_so_survivors_are_not_one_class():
    """77 at the 91st and 190 at the 62nd are different findings."""
    grid, image = _maps(concentrated=True)
    high = gradcam_sheet.per_patient(77, _crop(), grid, image, survivor=0.91)
    low = gradcam_sheet.per_patient(190, _crop(1), grid, image, survivor=0.62)
    assert "91%" in high["panels"][0].label
    assert "62%" in low["panels"][0].label


# --------------------------------------------------------------------------
# the edge statistics -- kept apart on purpose
# --------------------------------------------------------------------------


def test_the_outer_ring_uniform_expectation_is_the_cell_arithmetic():
    """52 of 196 cells. A wrong constant here would make an ordinary map look
    edge-dominated or the reverse."""
    ring = gradcam_sheet.outer_ring()
    assert ring.sum() == 52
    assert gradcam_sheet.OUTER_RING_UNIFORM == pytest.approx(52 / 196)
    assert gradcam_sheet.OUTER_RING_UNIFORM == pytest.approx(0.265, abs=5e-4)


def test_a_uniform_map_gives_the_uniform_ring_share():
    flat = np.ones(GRID)
    assert gradcam_sheet.outer_ring_mass(flat) == pytest.approx(
        gradcam_sheet.OUTER_RING_UNIFORM
    )


def test_an_edge_dominated_map_is_detected():
    """The pattern the review saw by eye and the numbers did not."""
    edged = np.zeros(GRID)
    edged[gradcam_sheet.outer_ring()] = 1.0
    assert gradcam_sheet.outer_ring_mass(edged) == pytest.approx(1.0)
    centred = np.zeros(GRID)
    centred[5:9, 5:9] = 1.0
    assert gradcam_sheet.outer_ring_mass(centred) == 0.0


def test_the_content_mask_is_reduced_to_cell_resolution():
    """**The map carries nothing finer than a cell**, so the mask comes down
    to the map rather than the map being inflated to pixels."""
    mask = np.zeros(SIZE, dtype=bool)
    mask[:112, :] = True
    coverage = gradcam_sheet.cell_coverage(mask)
    assert coverage.shape == GRID
    assert coverage[0, 0] == pytest.approx(1.0)
    assert coverage[-1, -1] == pytest.approx(0.0)
    assert coverage.mean() == pytest.approx(0.5)


def test_a_mask_that_does_not_divide_into_the_grid_is_refused():
    with pytest.raises(ValueError, match="does not divide"):
        gradcam_sheet.cell_coverage(np.zeros((225, 225), dtype=bool))


def test_the_expectation_is_derived_per_patient_from_their_own_mask():
    """**[CORRECTED 2026-08-08] The constant is gone, not fixed.**

    It was the cohort MEAN pad fraction applied to every patient, and it also
    omitted the trapezium corners -- ``content_mask`` excludes those too, and
    at the median aspect ratio they add another 0.148. The expectation is now
    ``1 - mask.mean()``, from the same object the mass is measured against.
    """
    assert not hasattr(gradcam_sheet, "PAD_FRACTION_AT_MEDIAN_AR"), (
        "the constant is back; a better constant repeats the defect"
    )
    half = np.zeros(SIZE, dtype=bool)
    half[:, :112] = True
    report = gradcam_sheet.out_of_content(np.ones(GRID), half)
    assert report["expected"] == pytest.approx(0.5)
    assert report["mass"] == pytest.approx(0.5)
    assert report["ratio"] == pytest.approx(1.0)
    assert report["above"] is False, "a uniform map is not above chance"


def test_the_expectation_includes_the_trapezium_corners_not_only_the_pad():
    """**The defect that mattered most.** ``pad_fraction`` covers the pad;
    ``content_mask`` also excludes the trapezium corners. Scoring against the
    pad alone understates the expectation and inflates every ratio."""
    from cleft.train.augment_sheet import content_mask

    size, pad = 224, 0.2534
    width = int(round(size * (1 - pad)))
    mask = content_mask(size, ((size - width) // 2, 0, width, size), "g1")
    derived = 1.0 - mask.mean()
    assert derived > pad + 0.10, (
        f"the corners add only {derived - pad:.4f}; if that has become small "
        "the correction's magnitude has changed and the record needs re-reading"
    )


def test_the_statistic_is_undefined_when_there_is_no_pad():
    """**Non-square: no pad, full-square trapezium, so no out-of-content
    region.** The statistic does not exist rather than evaluating to zero, and
    a ratio against zero is not infinite."""
    report = gradcam_sheet.out_of_content(
        np.ones(GRID), np.ones(SIZE, dtype=bool)
    )
    assert report["expected"] == 0.0
    assert report["ratio"] is None and report["above"] is None
    assert "does not exist" in report["undefined_because"]


def test_a_map_entirely_outside_the_content_is_measured_not_refused():
    report = gradcam_sheet.out_of_content(
        np.ones(GRID), np.zeros(SIZE, dtype=bool)
    )
    assert report["mass"] == pytest.approx(1.0)
    assert report["expected"] == pytest.approx(1.0)


def test_the_two_edge_statistics_are_reported_separately():
    """**Their causes differ**: the pad boundary is a property of these crops
    and vanishes at G2; the border-token behaviour is architectural and does
    not. One figure would attribute either to the other."""
    grid, image = _maps(concentrated=True)
    mask = np.zeros(SIZE, dtype=bool)
    mask[:, 40:184] = True          # a real pad, so the statistic is defined
    entry = gradcam_sheet.per_patient(1, _crop(), grid, image, content=mask)
    report = entry["report"]
    assert "outer_ring_mass" in report and "out_of_content" in report
    assert report["outer_ring_uniform"] != report["out_of_content"]["expected"]
    # Both appear on the panel, beside the top-10 share.
    assert "ring" in entry["panels"][1].label and "pad" in entry["panels"][1].label


def test_out_of_content_is_omitted_when_no_mask_is_supplied():
    """It is a property of the crop, so without the crop's geometry there is
    nothing to report -- and reporting a default would invent one."""
    grid, image = _maps(concentrated=True)
    report = gradcam_sheet.per_patient(1, _crop(), grid, image)["report"]
    assert "outer_ring_mass" in report
    assert "out_of_content" not in report


def test_edge_convergence_is_descriptive_unless_it_excludes_zero():
    """**Three statistics converging would be one finding, not three** -- but
    at n=15 the interval will be wide, and a wide one is the answer."""
    rng = np.random.default_rng(5)
    entries = []
    for i in range(8):
        grid, image = _maps(concentrated=True, seed=i)
        entries.append(gradcam_sheet.per_patient(
            i, _crop(i), grid, image, seed_agreement=list(rng.random(3))
        ))
    result = gradcam_sheet.edge_convergence(
        entries, finals=list(rng.normal(size=8)), n_boot=200
    )
    assert result["n_patients"] == 8
    for name in ("ring_vs_seed_agreement", "ring_vs_randomisation_final"):
        entry = result[name]
        assert entry["ci95"][0] <= entry["spearman"] <= entry["ci95"][1]
        if not entry["excludes_zero"]:
            assert "DESCRIPTIVE" in entry["reading"]


# --------------------------------------------------------------------------
# the geometry reader -- R10-adjacent, and this one WAS laptop-reachable
# --------------------------------------------------------------------------


GEOMETRY_COLUMNS = (
    "patient_id,frontal_id,source_w,source_h,aspect_ratio,"
    "content_x,content_y,content_w,content_h,pad_fraction"
)


def _staged_with_geometry(tmp_path, size=224):
    """A real geometry.csv: tier marker, all ten columns, varying content."""
    staged = tmp_path / "staged"
    staged.mkdir()
    rows = ["# CLUSTER-ONLY: patient-keyed geometry", GEOMETRY_COLUMNS]
    for pid in (1, 2, 3):
        width = size - 40 * pid
        rows.append(
            f"{pid},{1000 + pid},800,1000,0.8,"
            f"{(size - width) // 2},0,{width},{size},0.25"
        )
    (staged / "geometry.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    return staged


def test_the_content_masks_come_from_the_geometry_reader(tmp_path):
    """**[MEASURED 2026-08-04] This path called ``load_manifest`` and was
    refused** -- that validates against the manifest schema and geometry.csv
    carries ``content_x`` and the rest.

    ``phase3.load_geometry_rows`` is the reader for this file, already
    migrated to ``cluster_csv.read_cluster_csv``. It is the fourth site to
    read it and must not become a fifth.
    """
    from cleft.run import phase8_content_masks

    staged = _staged_with_geometry(tmp_path)
    masks = phase8_content_masks(staged, "g1", 224)

    assert sorted(masks) == [1, 2, 3]
    assert all(m.shape == (224, 224) for m in masks.values())
    # The per-patient content box is honoured: narrower boxes mask more out.
    covered = [masks[pid].mean() for pid in (1, 2, 3)]
    assert covered[0] > covered[1] > covered[2], covered
    assert 0.0 < covered[-1] < 1.0, "a mask that is all or nothing is not a box"


def test_the_manifest_loader_refuses_geometry_csv(tmp_path):
    """The regression, pinned: if someone swaps the reader back, this says
    why rather than failing at the first missing column."""
    from cleft.data.manifest import load_manifest

    staged = _staged_with_geometry(tmp_path)
    with pytest.raises(Exception) as caught:
        load_manifest(staged / "geometry.csv")
    message = str(caught.value)
    # The error paid for itself: it printed both column lists and named the
    # tier-marker cause, so the diagnosis took one read. Keep that shape.
    assert "content_x" in message or "column" in message.lower()


def test_the_geometry_reader_skips_the_tier_marker(tmp_path):
    """The marker has cost this project three rounds. The fourth reader gets
    it right by being the same reader."""
    from cleft.train.phase3 import load_geometry_rows

    staged = _staged_with_geometry(tmp_path)
    rows = load_geometry_rows(staged)
    assert len(rows) == 3
    assert all(row["patient_id"].isdigit() for row in rows)
    assert "content_x" in rows[0]


# --------------------------------------------------------------------------
# serialisation -- the boundary, with the types the real path produces
# --------------------------------------------------------------------------


def test_as_builtin_converts_every_numpy_scalar_kind():
    """**[MEASURED 2026-08-04] ``default=`` would not have been enough.**

        np.bool_    TypeError
        np.int64    TypeError
        np.float32  TypeError
        np.float64  SERIALISES SILENTLY -- it subclasses float

    So a handler catches the loud cases and leaves np.float64 in the output:
    a value that serialises but is not a Python type, which survives a round
    trip looking fine.
    """
    import json

    from cleft.run import as_builtin

    payload = {
        "b": np.bool_(True), "i": np.int64(3),
        "f32": np.float32(1.5), "f64": np.float64(2.5),
        "nested": {"arr": np.arange(3), "list": [np.bool_(False)]},
        "tuple": (np.float64(1.0), 2),
    }
    converted = as_builtin(payload)
    assert type(converted["b"]) is bool
    assert type(converted["i"]) is int
    assert type(converted["f32"]) is float
    assert type(converted["f64"]) is float, (
        "np.float64 serialises silently, so a default= handler never sees it"
    )
    assert converted["nested"]["arr"] == [0, 1, 2]
    assert type(converted["nested"]["list"][0]) is bool
    assert converted["tuple"] == [1.0, 2]
    json.dumps(converted)  # the point of all of it


def test_as_builtin_leaves_plain_values_alone():
    from cleft.run import as_builtin

    payload = {"s": "text", "n": None, "i": 1, "f": 2.0, "b": True}
    assert as_builtin(payload) == payload


def test_the_real_report_serialises_with_the_types_it_actually_produces():
    """**The gap was the fixture, not the cast.** A report built from Python
    floats serialises whatever the code does; this builds one through the real
    functions, which is what produces the numpy types.
    """
    import json

    from cleft.run import as_builtin

    rng = np.random.default_rng(9)
    mask = np.zeros(SIZE, dtype=bool)
    mask[:, 40:184] = True
    entries = []
    for i in range(6):
        grid, image = _maps(concentrated=True, seed=i)
        entries.append(gradcam_sheet.per_patient(
            i, _crop(i), grid, image, content=mask,
            seed_agreement=list(rng.random(3)),
            survivor=float(rng.random()) if i == 0 else None,
        ))
    report = gradcam_sheet.summarise(entries)
    report["edge_convergence"] = gradcam_sheet.edge_convergence(
        entries, finals=list(rng.normal(size=6)), n_boot=100
    )
    report["verdict_banner"] = gradcam_sheet.verdict_banner(
        {"passes": False, "survivors": [0], "n_patients": 6, "baseline": 0.09}
    )

    # This is the crash: np.bool_ from a comparison reaching json.dumps.
    json.dumps(as_builtin(report))

    # And the boundary is doing the work, not luck about which types appeared.
    def has_numpy(value):
        if isinstance(value, dict):
            return any(has_numpy(v) for v in value.values())
        if isinstance(value, (list, tuple)):
            return any(has_numpy(v) for v in value)
        return isinstance(value, (np.generic, np.ndarray))

    assert not has_numpy(as_builtin(report))


def test_edge_convergence_excludes_zero_is_a_python_bool():
    """`bool(draws) and (...)` returns the SECOND operand when the first is
    truthy, so this was np.bool_ and crashed the report write. Fixed at source
    as well as at the boundary: a wrong type in the record is a defect even
    where json tolerates it."""
    rng = np.random.default_rng(11)
    entries = [
        gradcam_sheet.per_patient(
            i, _crop(i), *_maps(concentrated=True, seed=i),
            seed_agreement=list(rng.random(3)),
        )
        for i in range(6)
    ]
    result = gradcam_sheet.edge_convergence(
        entries, finals=list(rng.normal(size=6)), n_boot=100
    )
    for name, entry in result.items():
        if isinstance(entry, dict) and "excludes_zero" in entry:
            assert type(entry["excludes_zero"]) is bool, name
