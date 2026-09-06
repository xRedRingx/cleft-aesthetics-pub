"""Grad-CAM's arithmetic and its refusals, on the laptop.

Torch stays out of the test extra, so the split in ``gradcam.py`` is what
makes this file possible: everything except the model forward is numpy, and
everything that can silently produce a plausible picture is here.

**The tests that matter are the refusals.** A zero gradient does not give a
blank map -- it gives whatever the normalisation step manufactures from float
noise -- so the module refuses instead, and these check it refuses for the
right reason and does not refuse a real map.
"""

from __future__ import annotations

import numpy as np
import pytest

from cleft import gradcam, phase8

GRID = phase8.GRAD_CAM_TARGET["grid"]
N_TOKENS = GRID[0] * GRID[1]


def _sample(seed=0, channels=8, tokens=N_TOKENS):
    rng = np.random.default_rng(seed)
    return (
        rng.normal(size=(tokens, channels)),
        rng.normal(size=(tokens, channels)),
    )


# --------------------------------------------------------------------------
# the CAM
# --------------------------------------------------------------------------


def test_the_cam_is_relu_of_the_weighted_activation_sum():
    acts, grads = _sample()
    expected = np.maximum((acts * grads.mean(axis=0)).sum(axis=1), 0.0)
    assert gradcam.cam(acts, grads) == pytest.approx(expected)
    assert (gradcam.cam(acts, grads) >= 0).all()


def test_a_zero_gradient_is_refused_rather_than_normalised():
    """**[MEASURED] This is block 12's patch tokens.** The head reads the CLS
    token after a per-token LayerNorm, so they feed nothing and their gradient
    is identically zero. Normalising that map divides noise by noise.
    """
    acts, _ = _sample()
    with pytest.raises(gradcam.GradCamError, match="structurally absent"):
        gradcam.cam(acts, np.zeros_like(acts))


def test_gradients_that_cancel_under_the_spatial_mean_are_refused():
    """A pathology an absolute floor misses: gradients that are large but
    anti-symmetric, so the channel weights carry no signal even though the raw
    gradients look healthy."""
    acts, _ = _sample()
    cancelling = np.tile(np.array([[1.0], [-1.0]]), (acts.shape[0] // 2, acts.shape[1]))
    assert np.abs(cancelling).max() == 1.0
    with pytest.raises(gradcam.GradCamError, match="vanish relative"):
        gradcam.cam(acts, cancelling)


@pytest.mark.parametrize("scale", [1e6, 1.0, 1e-6, 1e-12])
def test_the_refusal_is_scale_free(scale):
    """**[MEASURED 2026-08-04] An absolute floor was safe only by accident.**

    With arm A's linear head scaled by 1e-4, block 11's channel weights fall
    to 3.18e-07 -- 2.7x above the old float32-epsilon floor, so one more
    factor of ten would have refused a valid map. Gradient magnitude depends
    on the head weights, the loss scale and the input normalisation, none of
    which is fixed by anything.
    """
    acts, grads = _sample(seed=21)
    accepted = gradcam.cam(acts, grads * scale)
    assert accepted.max() > 0
    # Zero stays refused at every scale, because zero has no scale.
    with pytest.raises(gradcam.GradCamError, match="vanish relative"):
        gradcam.cam(acts, np.zeros_like(acts) * scale)


def test_the_refusal_names_the_layer_that_causes_it():
    """A reader hitting this error should not have to work out why."""
    acts, _ = _sample()
    with pytest.raises(gradcam.GradCamError) as caught:
        gradcam.cam(acts, np.zeros_like(acts))
    message = str(caught.value)
    assert "block 12" in message and "CLS" in message
    assert "GRAD_CAM_TARGET" in message


def test_a_real_gradient_is_not_refused():
    """The refusal must not be so eager that it rejects working maps."""
    acts, grads = _sample(seed=3)
    assert gradcam.cam(acts, grads).max() > 0


def test_mismatched_shapes_are_refused():
    acts, grads = _sample()
    with pytest.raises(gradcam.GradCamError, match="differ"):
        gradcam.cam(acts, grads[:-1])
    with pytest.raises(gradcam.GradCamError, match="tokens, channels"):
        gradcam.cam(acts[:, 0], grads[:, 0])


def test_an_all_negative_map_is_refused_by_normalise():
    """After the ReLU it is zero everywhere, and every convention for that --
    epsilon, zeros, NaN -- produces something a reader will look at."""
    with pytest.raises(gradcam.GradCamError, match="zero everywhere"):
        gradcam.normalise(np.zeros(N_TOKENS))


def test_normalise_scales_to_one():
    values = np.array([0.0, 1.0, 4.0])
    assert gradcam.normalise(values).max() == pytest.approx(1.0)
    assert gradcam.normalise(values)[2] == pytest.approx(1.0)


# --------------------------------------------------------------------------
# the grid and the upsample
# --------------------------------------------------------------------------


def test_the_token_count_must_fill_the_pre_registered_grid():
    assert gradcam.to_grid(np.arange(N_TOKENS)).shape == GRID
    with pytest.raises(gradcam.GradCamError, match="do not fill"):
        gradcam.to_grid(np.arange(N_TOKENS + 1))


def test_the_upsample_is_the_pre_registered_factor():
    """14x14 to 224 is 16x, and the record says nothing finer than a 16-pixel
    block is real."""
    grid = np.arange(N_TOKENS, dtype=float).reshape(GRID)
    image = gradcam.upsample(grid)
    assert image.shape == tuple(phase8.RESOLUTION_FLOOR["image"])
    assert image.shape[0] // grid.shape[0] == phase8.RESOLUTION_FLOOR[
        "upsample_factor"
    ]


def test_the_upsample_preserves_the_corners_and_the_range():
    grid = np.random.default_rng(1).random(GRID)
    image = gradcam.upsample(grid)
    assert image[0, 0] == pytest.approx(grid[0, 0])
    assert image[-1, -1] == pytest.approx(grid[-1, -1])
    # Bilinear interpolation cannot overshoot its inputs.
    assert image.min() >= grid.min() - 1e-9
    assert image.max() <= grid.max() + 1e-9


def test_the_map_carries_its_caveat_in_the_object():
    """``FROZEN_BACKBONE_CAVEAT`` puts it in the artifact, because a caveat in
    prose travels separately from the figure and the figure gets reused."""
    acts, grads = _sample(seed=5)
    result = gradcam.map_for(acts, grads)
    assert result["grid"].shape == GRID
    assert result["image"].shape == tuple(phase8.RESOLUTION_FLOOR["image"])
    assert "NOT what the model learned about clefts" in result["caveat"]
    assert result["layer"] == phase8.GRAD_CAM_TARGET["layer"]
    assert result["resolution_floor_px"] == 16


# --------------------------------------------------------------------------
# the randomisation test
# --------------------------------------------------------------------------


def test_similarity_is_rank_based_so_amplitude_does_not_count():
    """A randomised model whose map has the same shape at a different scale
    has not degraded."""
    rng = np.random.default_rng(2)
    first = rng.random(N_TOKENS)
    assert gradcam.similarity(first, first * 7.0) == pytest.approx(1.0)
    assert gradcam.similarity(first, -first) == pytest.approx(-1.0)


def test_similarity_handles_ties_without_ordering_by_position():
    flat = np.zeros(10)
    assert gradcam.similarity(flat, np.arange(10.0)) == 0.0


def test_the_baseline_is_measured_between_patients_not_chosen():
    """**The pass criterion is empirical.** A chosen threshold invites 'why
    that number', and the honest answer would be that it looked right."""
    rng = np.random.default_rng(7)
    maps = [rng.random(N_TOKENS) for _ in range(5)]
    baseline = gradcam.between_patient_baseline(maps)
    assert -1.0 <= baseline <= 1.0
    with pytest.raises(gradcam.GradCamError, match="at least two"):
        gradcam.between_patient_baseline(maps[:1])
    assert "measured" in phase8.RANDOMISATION_TEST["criterion"]["why_not_a_constant"] \
        or "looked right" in phase8.RANDOMISATION_TEST["criterion"]["why_not_a_constant"]


def test_a_map_that_survives_randomisation_fails_the_test():
    """An edge detector wearing an explanation's name. Publishing it would be
    R7's fifteenth instance."""
    rng = np.random.default_rng(11)
    original = rng.random(N_TOKENS)
    survives = gradcam.randomisation_curve(
        original, {"block12": original * 3.0}  # identical ranking
    )
    verdict = gradcam.randomisation_verdict([survives], baseline=0.1)
    assert verdict["passes"] is False
    assert verdict["survivors"] == [0]
    assert "NOTHING MAY BE PUBLISHED" in verdict["reading"]


def test_a_map_that_degrades_passes():
    rng = np.random.default_rng(13)
    original = rng.random(N_TOKENS)
    curve = gradcam.randomisation_curve(original, {"block12": rng.random(N_TOKENS)})
    verdict = gradcam.randomisation_verdict([curve], baseline=0.9)
    assert verdict["passes"] is True
    assert verdict["survivors"] == []


def test_one_surviving_map_is_not_averaged_away_by_the_others():
    """The verdict is per patient. Fifteen maps of which one survives is a
    failure, not a 93% pass."""
    rng = np.random.default_rng(17)
    curves, originals = [], []
    for index in range(5):
        original = rng.random(N_TOKENS)
        originals.append(original)
        randomised = original * 2.0 if index == 4 else rng.random(N_TOKENS)
        curves.append(gradcam.randomisation_curve(original, {"s": randomised}))
    verdict = gradcam.randomisation_verdict(curves, baseline=0.5)
    assert verdict["survivors"] == [4]
    assert verdict["passes"] is False
    assert verdict["n_patients"] == 5


def test_a_curve_with_no_stages_is_refused():
    curve = gradcam.randomisation_curve(np.arange(4.0), {})
    with pytest.raises(gradcam.GradCamError, match="nothing was randomised"):
        gradcam.randomisation_verdict([curve], baseline=0.5)


# --------------------------------------------------------------------------
# the one function no test can execute
# --------------------------------------------------------------------------


def test_every_extractor_attribute_gradcam_uses_actually_exists():
    """**[MEASURED 2026-08-04] It did not, and nothing here could have run it.**

    ``token_activations_and_gradients`` needs torch and a model, so no laptop
    test executes it -- the same position ``task_phase7b_search`` was in when
    it died on the cluster calling ``ctx.write_metrics``, a method that did not
    exist. The first draft of this module called ``extractor.model`` and
    ``extractor.preprocess``, and ``FrozenExtractor`` had neither.

    So the attribute names are checked statically instead, exactly as
    ``test_workflow_hygiene`` checks every ``ctx.<name>`` in every task.
    """
    import ast
    import pathlib

    from cleft.train.torch_backbone import FrozenExtractor

    source = pathlib.Path(gradcam.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)

    used = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "extractor"
        ):
            used.add(node.attr)

    assert used, (
        "no extractor attribute is used any more; this guard has stopped "
        "covering anything and should be re-read rather than deleted"
    )
    missing = sorted(name for name in used if not hasattr(FrozenExtractor, name))
    assert not missing, (
        f"gradcam.py uses extractor.{missing} and FrozenExtractor has no such "
        "attribute. No test can execute that path, so this is the only place "
        "it fails before the cluster"
    )


def test_preprocess_is_shared_with_the_embedding_path():
    """Grad-CAM cannot run under ``no_grad``, so it cannot reuse ``__call__``.
    Preprocessing is factored out rather than copied -- two definitions of what
    the model is fed is the shape of defect this project keeps finding."""
    import inspect

    from cleft.train.torch_backbone import FrozenExtractor

    call_source = inspect.getsource(FrozenExtractor.__call__)
    assert "self.preprocess(" in call_source, (
        "__call__ no longer routes through preprocess, so the embedding path "
        "and the Grad-CAM path can now disagree about normalisation"
    )
    assert "no_grad" in call_source

    # Structural, not textual: `preprocess`'s own docstring says the words
    # "no_grad", so a substring check would pass for the wrong reason. What
    # matters is that it opens no such block -- Grad-CAM needs the graph.
    import ast
    import textwrap

    body = ast.parse(
        textwrap.dedent(inspect.getsource(FrozenExtractor.preprocess))
    )
    contexts = [
        node for node in ast.walk(body)
        if isinstance(node, ast.With)
        for item in node.items
        if "no_grad" in ast.dump(item.context_expr)
    ]
    assert not contexts, "preprocess opens a no_grad block; Grad-CAM needs the graph"


def test_every_api_the_phase8_task_calls_actually_exists():
    """**[MEASURED 2026-08-04] Third instance of the same shape, and I wrote
    two of them.**

    The first draft of ``phase8_refit_heads`` called
    ``phase3.run_cv(..., keep_heads=True)`` -- wrong module, wrong signature,
    and a return field that does not exist. Before it, ``gradcam`` called
    ``extractor.preprocess``; before that, ``task_phase7b_search`` called
    ``ctx.write_metrics`` and died on the cluster.

    Every one of them sits on a torch-only path no laptop test can execute, so
    the names are checked statically. This is the generalisation
    ``test_workflow_hygiene`` made for ``ctx.<name>``, applied to the two
    objects Phase 8's task drives.
    """
    import ast
    import pathlib

    from cleft.train import harness
    from cleft.train.torch_backbone import EmbeddingHeadBackbone

    source = pathlib.Path(
        pathlib.Path(gradcam.__file__).parent / "run.py"
    ).read_text(encoding="utf-8")
    tree = ast.parse(source)

    functions = {
        node.name: node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name.startswith("phase8_")
    }
    assert functions, "the phase8 helpers are gone; this guard covers nothing"

    used_on = {"harness": set(), "backbone": set()}
    for node in ast.walk(functions["phase8_refit_heads"]):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            if node.value.id == "harness":
                used_on["harness"].add(node.attr)
            elif node.value.id == "backbone":
                used_on["backbone"].add(node.attr)

    assert used_on["harness"], "no harness attribute is used; re-read this guard"
    missing = sorted(n for n in used_on["harness"] if not hasattr(harness, n))
    assert not missing, f"train.harness has no {missing}"
    missing = sorted(
        n for n in used_on["backbone"] if not hasattr(EmbeddingHeadBackbone, n)
    )
    assert not missing, f"EmbeddingHeadBackbone has no {missing}"


def test_the_train_config_fields_the_task_sets_are_real():
    """A TrainConfig keyword that does not exist would raise on the cluster,
    after the embeddings had been loaded."""
    import dataclasses

    from cleft.train import harness

    fields = {f.name for f in dataclasses.fields(harness.TrainConfig)}
    for name in ("seed", "inner_val_frac", "max_epochs", "patience", "monitor"):
        assert name in fields, name


def test_the_sheet_call_sites_in_the_task_are_real():
    """R10: the sheet is drawn on a torch-only path no laptop test executes,
    so the names it calls are checked statically like every other one."""
    import ast
    import pathlib

    from cleft import gradcam_sheet
    from cleft.geometry import render

    tree = ast.parse(
        (pathlib.Path(gradcam.__file__).parent / "run.py").read_text("utf-8")
    )
    task = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "task_grad_cam"
    )
    used = {"gradcam_sheet": set(), "gradcam": set()}
    for node in ast.walk(task):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            if node.value.id in used:
                used[node.value.id].add(node.attr)

    assert used["gradcam_sheet"], "the task no longer draws a sheet"
    for module, names in (
        (gradcam_sheet, used["gradcam_sheet"]), (gradcam, used["gradcam"]),
    ):
        missing = sorted(n for n in names if not hasattr(module, n))
        assert not missing, f"{module.__name__} has no {missing}"
    assert hasattr(render, "save_sheet")


def test_the_sheet_is_written_cluster_only():
    """It renders patient faces. The tier guard enforces by filename, so the
    task must claim the path at the right tier."""
    import pathlib
    import re

    source = (pathlib.Path(gradcam.__file__).parent / "run.py").read_text("utf-8")
    body = source.split("def task_grad_cam", 1)[1].split("\ndef ", 1)[0]
    for name in ("grad_cam_sheet.png", "grad_cam_maps.npz"):
        claim = re.search(
            rf'ctx\.path\(\s*"{re.escape(name)}"\s*,\s*tier="([A-Z-]+)"', body
        )
        assert claim, f"{name} is not claimed through ctx.path"
        assert claim.group(1) == "CLUSTER-ONLY", f"{name}: {claim.group(1)}"


# --------------------------------------------------------------------------
# the baseline's own spread -- Phase 7's question applied to Phase 8's gate
# --------------------------------------------------------------------------


def test_the_baseline_is_reported_as_a_distribution_not_only_a_median():
    """**A threshold without a spread is condition 2 without condition 1.**

    0.081 passing while 0.146 fails is only meaningful if the baseline has
    less spread than the gap between them, and one number cannot say.
    """
    rng = np.random.default_rng(31)
    maps = [rng.random(N_TOKENS) for _ in range(6)]
    spread = gradcam.between_patient_distribution(maps)
    assert spread["n_pairs"] == 6 * 5 // 2
    assert spread["median"] == pytest.approx(
        gradcam.between_patient_baseline(maps)
    ), "the criterion's number must be the same one the distribution reports"
    for key in ("mean", "sd", "min", "max", "q05", "q25", "q75", "q95"):
        assert key in spread
    assert spread["min"] <= spread["median"] <= spread["max"]
    assert len(spread["pairs"]) == spread["n_pairs"]


def test_separability_resamples_patients_not_pairs():
    """**Fifteen maps make 105 correlated pairs**, each map appearing in
    fourteen. Bootstrapping over pairs would treat one patient's oddity as
    fourteen pieces of evidence."""
    rng = np.random.default_rng(37)
    maps = [rng.random(N_TOKENS) for _ in range(8)]
    finals = list(rng.normal(scale=0.1, size=8))
    result = gradcam.separability(finals, maps, n_boot=200)
    assert result["resampling_unit"] == "patient"
    assert result["ci95"][0] <= result["observed_difference"] <= result["ci95"][1]
    assert isinstance(result["excludes_zero"], bool)


def test_separability_refuses_a_mismatched_sample():
    rng = np.random.default_rng(41)
    maps = [rng.random(N_TOKENS) for _ in range(5)]
    with pytest.raises(gradcam.GradCamError, match="must correspond"):
        gradcam.separability([0.1, 0.2], maps, n_boot=50)


def test_separability_detects_a_real_separation():
    """The instrument must be able to say yes, or a no means nothing."""
    rng = np.random.default_rng(43)
    maps = [rng.random(N_TOKENS) for _ in range(10)]
    far_below = list(np.full(10, -0.9))
    result = gradcam.separability(far_below, maps, n_boot=300)
    assert result["excludes_zero"] is True
    assert "are separable" in result["reading"]


def test_the_verdict_carries_the_spread_without_using_it():
    """**Reported beside the verdict, never folded into it.** Changing the
    rule in response is the move phase7c corrected twice."""
    rng = np.random.default_rng(47)
    maps = [rng.random(N_TOKENS) for _ in range(6)]
    curves = [
        gradcam.randomisation_curve(m, {"s": rng.random(N_TOKENS)}) for m in maps
    ]
    spread = gradcam.between_patient_distribution(maps)
    plain = gradcam.randomisation_verdict(curves, spread["median"])
    rich = gradcam.randomisation_verdict(
        curves, spread["median"], distribution=spread,
        separable=gradcam.separability(
            [c["final"] for c in curves], maps, n_boot=100
        ),
    )
    # The verdict itself is identical; only the reporting differs.
    for key in ("passes", "survivors", "baseline", "finals"):
        assert plain[key] == rich[key], key
    assert "baseline_distribution" in rich and "separability" in rich
    assert len(rich["final_percentiles"]) == len(curves)


def test_the_record_says_reporting_does_not_change_the_rule():
    criterion = phase8.RANDOMISATION_TEST["criterion"]
    assert "stands as written" in criterion["reporting_does_not_change_the_rule"]
    assert len(criterion["reported_alongside"]) == 3


def test_the_sheet_is_rendered_but_the_npz_is_withheld_on_failure():
    """**[DECIDED] The npz is what 'published' scopes over; the sheet is not.**

    The review asks whether the METHOD puts maps on plausible anatomy, which
    is a different question from the gate's and is worth answering whichever
    way the gate went. Enforced structurally rather than by a stamp, because
    ``FROZEN_BACKBONE_CAVEAT`` already records that a caveat in prose travels
    separately from the figure.
    """
    import ast
    import pathlib

    from cleft import phase8

    source = (pathlib.Path(gradcam.__file__).parent / "run.py").read_text("utf-8")
    tree = ast.parse(source)
    task = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "task_grad_cam"
    )
    body = ast.get_source_segment(source, task) or ""

    # The npz write is guarded by the verdict; the sheet write is not.
    npz_line = body.index("grad_cam_maps.npz")
    sheet_line = body.index("grad_cam_sheet.png")
    guard = body.rindex('if verdict["passes"]:', 0, npz_line)
    assert guard < npz_line, "the npz write is not guarded by the verdict"
    assert sheet_line > npz_line, "the sheet should be written after the branch"
    assert 'if verdict["passes"]:' not in body[npz_line:sheet_line], (
        "the sheet write must not sit inside the pass branch"
    )

    # And the raise is last, so both artifacts exist before it fires.
    raise_at = body.index("survive model-parameter randomisation")
    assert raise_at > sheet_line, (
        "the task raises before the sheet is written, so a failed gate would "
        "decide whether the method can be reviewed at all"
    )
    assert phase8.PUBLICATION_SCOPE["npz"]["written"] is False
    assert phase8.PUBLICATION_SCOPE["sheet"]["rendered"] is True
