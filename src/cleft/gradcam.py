"""Grad-CAM for the frozen ViT arm, and the randomisation test that gates it.

**The arithmetic is pure numpy and the torch is confined to one function.**
Torch stays out of the test extra (PLAN, standing constraint), so everything
that can fail on a laptop is tested on a laptop: the CAM itself, the
zero-gradient refusal, the upsampling, the similarity statistic and the pass
criterion. Only ``token_activations_and_gradients`` needs a GPU and a model.

**The zero-gradient refusal is the point of this module, not a detail.**
[MEASURED 2026-08-04, ``phase8.GRAD_CAM_TARGET``] the head reads ``x[:, 0]``
after a per-token LayerNorm, so at block 12's output the patch tokens feed
nothing and their gradient is *identically zero*. A CAM built from zero
gradients is a zero map, and the conventional final step -- divide by the
maximum -- turns float noise into a full-range picture. So a map that would
have looked like an explanation is refused here instead, by construction.
"""

from __future__ import annotations

import numpy as np

from . import phase8


class GradCamError(RuntimeError):
    """A map was asked for that cannot be computed, or must not be trusted."""


#: **The refusal is RELATIVE to the gradient's own scale, not an absolute
#: floor, and the difference matters.**
#:
#: [MEASURED 2026-08-04] An absolute float32-epsilon floor (1.19e-07) happened
#: to separate the two cases -- block 11 reaches 3.19e-03 under arm A's linear
#: head and block 12 is exactly 0.0 -- but only by accident of the current
#: scale. Gradient magnitude depends on the head's weights, the loss scale and
#: the input normalisation, none of which is fixed by anything; halve the head
#: and an absolute threshold starts refusing valid maps.
#:
#: So the test is whether the channel weights are vanishing **relative to the
#: raw gradients they were averaged from**. That is scale-free, it still
#: refuses the exactly-zero case (0 <= eps * 0), and it additionally catches a
#: pathology an absolute floor misses: gradients that are large but cancel
#: under the spatial mean, leaving weights that carry no signal.
GRADIENT_FLOOR = float(np.finfo(np.float32).eps)


def cam(activations, gradients) -> np.ndarray:
    """The Grad-CAM map for one sample, on the token grid.

    ``activations`` and ``gradients`` are ``(tokens, channels)`` for the patch
    tokens only -- the CLS token is not part of the spatial grid and including
    it would put a non-spatial quantity in a spatial average.

    Returns the map BEFORE normalisation, so a caller can see its scale. It is
    ``ReLU(sum_k alpha_k A_k)`` with ``alpha_k`` the gradient averaged over
    tokens, which is Grad-CAM's definition with the spatial mean taken over a
    1-D token axis instead of a 2-D map.
    """
    acts = np.asarray(activations, dtype=np.float64)
    grads = np.asarray(gradients, dtype=np.float64)
    if acts.shape != grads.shape:
        raise GradCamError(
            f"activations {acts.shape} and gradients {grads.shape} differ"
        )
    if acts.ndim != 2:
        raise GradCamError(
            f"expected (tokens, channels), got {acts.shape}. The CLS token is "
            "excluded by the caller, not here"
        )

    return np.maximum((acts * _guarded_channel_weights(acts, grads)).sum(axis=1), 0.0)


def _guarded_channel_weights(acts: np.ndarray, grads: np.ndarray) -> np.ndarray:
    """``alpha_k``: the gradient averaged over tokens, refused if it vanishes.

    Extracted from ``cam`` unchanged (message included) so the softmax
    variant meets the SAME zero-gradient guard rather than a second copy of
    it -- the block-12 refusal must fire identically for both methods.
    """
    weights = grads.mean(axis=0)
    peak = float(np.abs(weights).max()) if weights.size else 0.0
    scale = float(np.abs(grads).max()) if grads.size else 0.0
    if peak <= GRADIENT_FLOOR * scale:
        raise GradCamError(
            f"the channel weights vanish relative to the gradients they came "
            f"from: max |alpha| = {peak:.3e} against max |grad| = "
            f"{scale:.3e}. Either the gradient is structurally absent -- which "
            "is what block 12's patch tokens give, since the head reads the "
            "CLS token after a per-token LayerNorm so they feed nothing -- or "
            "it cancels under the spatial mean. Normalising the resulting map "
            "would turn float noise into a full-range picture. See "
            "phase8.GRAD_CAM_TARGET"
        )
    return weights


def cam_softmax(activations, gradients) -> np.ndarray:
    """**The 8b variant** (phase8.GRAD_CAM_SOFTMAX_REGISTERED): the SECOND
    method, beside ``cam`` and never a modification of it.

    Identical to ``cam`` in everything except the two registered deltas,
    applied to the same ``alpha_k`` the same guard released:

    * **negatives dropped BEFORE aggregation** -- channels whose importance
      is not positive contribute nothing, rather than subtracting;
    * **softmax over the surviving importances** -- the weighting becomes a
      distribution over channels instead of raw gradient means.

    The final ReLU stays: it is part of "everything else identical", and
    without it the two methods would differ in a third, unregistered way
    (activations can be negative even under positive weights).
    """
    acts = np.asarray(activations, dtype=np.float64)
    grads = np.asarray(gradients, dtype=np.float64)
    if acts.shape != grads.shape:
        raise GradCamError(
            f"activations {acts.shape} and gradients {grads.shape} differ"
        )
    if acts.ndim != 2:
        raise GradCamError(
            f"expected (tokens, channels), got {acts.shape}. The CLS token is "
            "excluded by the caller, not here"
        )
    weights = _guarded_channel_weights(acts, grads)
    positive = weights > 0.0
    if not positive.any():
        raise GradCamError(
            "every channel importance is non-positive, so after the "
            "registered negative-drop there is nothing to aggregate. The "
            "original method's map would have been all-zero here too (its "
            "ReLU), and gradcam.normalise refuses that map for the same "
            "reason this refuses: no picture beats a fabricated one"
        )
    kept = weights[positive]
    exp = np.exp(kept - kept.max())
    softmax_weights = exp / exp.sum()
    return np.maximum(
        (acts[:, positive] * softmax_weights).sum(axis=1), 0.0
    )


def map_for_softmax(activations, gradients, *, grid=None, size=None) -> dict:
    """One patient's VARIANT map, provenance attached, method named.

    Mirrors ``map_for`` and additionally carries ``method`` so no artifact
    or record downstream can be confused with the original's -- the naming
    rule the 8b registration sets for every surface.
    """
    raw = cam_softmax(activations, gradients)
    grid_map = to_grid(normalise(raw), grid)
    return {
        "method": "grad_cam_softmax",
        "grid": grid_map,
        "image": upsample(grid_map, size),
        "layer": phase8.GRAD_CAM_TARGET["layer"],
        "resolution_floor_px": phase8.RESOLUTION_FLOOR["floor_px"],
        "caveat": phase8.FROZEN_BACKBONE_CAVEAT["text"],
    }


def normalise(values) -> np.ndarray:
    """Scale a map to [0, 1]. Refuses a map with no positive part.

    A CAM that is zero everywhere after the ReLU has no maximum to divide by,
    and every convention for that case -- add an epsilon, return zeros, return
    NaN -- produces something a reader will look at.
    """
    array = np.asarray(values, dtype=np.float64)
    peak = float(array.max()) if array.size else 0.0
    if peak <= 0.0:
        raise GradCamError(
            "the map is zero everywhere after the ReLU, so it has no maximum "
            "to normalise by. Refusing rather than returning a picture"
        )
    return array / peak


def to_grid(tokens, grid=None) -> np.ndarray:
    """Reshape a token vector to the 2-D grid it came from."""
    grid = tuple(grid or phase8.GRAD_CAM_TARGET["grid"])
    array = np.asarray(tokens, dtype=np.float64)
    if array.size != grid[0] * grid[1]:
        raise GradCamError(
            f"{array.size} tokens do not fill a {grid[0]}x{grid[1]} grid. The "
            "CLS token is excluded before this point"
        )
    return array.reshape(grid)


def upsample(grid_map, size=None) -> np.ndarray:
    """Bilinear upsample to image resolution.

    **The result carries no information finer than
    ``phase8.RESOLUTION_FLOOR``.** A 14x14 grid at 224 is a 16x interpolation,
    so structure below a 16-pixel block is the kernel, not the model. This
    function cannot enforce that -- it is a statement about how the output is
    read, and it lives in the record and in the artifact.
    """
    source = np.asarray(grid_map, dtype=np.float64)
    if source.ndim != 2:
        raise GradCamError(f"expected a 2-D grid, got {source.shape}")
    height, width = tuple(size or phase8.RESOLUTION_FLOOR["image"])

    rows = np.linspace(0, source.shape[0] - 1, height)
    cols = np.linspace(0, source.shape[1] - 1, width)
    r0 = np.clip(np.floor(rows).astype(int), 0, source.shape[0] - 1)
    r1 = np.clip(r0 + 1, 0, source.shape[0] - 1)
    c0 = np.clip(np.floor(cols).astype(int), 0, source.shape[1] - 1)
    c1 = np.clip(c0 + 1, 0, source.shape[1] - 1)
    dr = (rows - r0)[:, None]
    dc = (cols - c0)[None, :]

    top = source[np.ix_(r0, c0)] * (1 - dc) + source[np.ix_(r0, c1)] * dc
    bottom = source[np.ix_(r1, c0)] * (1 - dc) + source[np.ix_(r1, c1)] * dc
    return top * (1 - dr) + bottom * dr


def map_for(activations, gradients, *, grid=None, size=None) -> dict:
    """One patient's map, from tokens to image, with its provenance attached.

    The caveat travels IN the returned object, because
    ``phase8.FROZEN_BACKBONE_CAVEAT`` puts it in the artifact rather than the
    prose: a caveat in prose travels separately from the figure, and the
    figure is what gets reused.
    """
    raw = cam(activations, gradients)
    grid_map = to_grid(normalise(raw), grid)
    return {
        "grid": grid_map,
        "image": upsample(grid_map, size),
        "layer": phase8.GRAD_CAM_TARGET["layer"],
        "resolution_floor_px": phase8.RESOLUTION_FLOOR["floor_px"],
        "caveat": phase8.FROZEN_BACKBONE_CAVEAT["text"],
    }


# --------------------------------------------------------------------------
# the randomisation test
# --------------------------------------------------------------------------


def similarity(first, second) -> float:
    """Spearman rank correlation between two maps, over their tokens.

    Rank-based so it measures whether the maps agree about WHERE, not about
    scale -- a randomised model whose map has the same shape at a different
    amplitude has not degraded.
    """
    a = np.asarray(first, dtype=np.float64).ravel()
    b = np.asarray(second, dtype=np.float64).ravel()
    if a.shape != b.shape:
        raise GradCamError(f"maps differ in shape: {a.shape} against {b.shape}")
    if a.size < 2:
        raise GradCamError("a rank correlation needs at least two points")

    ranked = [_ranks(v) for v in (a, b)]
    if any(np.std(r) == 0 for r in ranked):
        return 0.0
    return float(np.corrcoef(ranked[0], ranked[1])[0, 1])


def _ranks(values: np.ndarray) -> np.ndarray:
    """Average ranks, so ties do not order themselves by array position."""
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    ranks[order] = np.arange(values.size, dtype=np.float64)
    unique, inverse, counts = np.unique(values, return_inverse=True,
                                        return_counts=True)
    sums = np.zeros(unique.size, dtype=np.float64)
    np.add.at(sums, inverse, ranks)
    return (sums / counts)[inverse]


def between_patient_baseline(maps) -> float:
    """The similarity that carries no patient-specific information.

    **The pass criterion is measured, not chosen**
    (``phase8.RANDOMISATION_TEST["criterion"]``). Two different patients'
    unrandomised maps agree to whatever extent the model attends to the same
    anatomy in everyone; a randomised model that stays above that level has
    retained something patient-specific, which is what the test is looking
    for.
    """
    maps = [np.asarray(m, dtype=np.float64).ravel() for m in maps]
    if len(maps) < 2:
        raise GradCamError(
            f"the baseline needs at least two patients' maps, got {len(maps)}"
        )
    pairs = [
        similarity(maps[i], maps[j])
        for i in range(len(maps)) for j in range(i + 1, len(maps))
    ]
    return float(np.median(pairs))


def between_patient_distribution(maps) -> dict:
    """The baseline as a DISTRIBUTION, not a point.

    **[DECIDED 2026-08-04] A threshold without a spread is PLAN §4.3's
    condition 2 without condition 1**, and Phase 7 spent itself learning that.
    ``between_patient_baseline`` returns one number and the criterion uses it
    unchanged; this reports what that number was computed from, so a reader
    can see whether a final at 0.081 passing and one at 0.146 failing is a
    distinction inside the baseline's own noise.

    Reporting it changes no verdict. It changes what the verdict is worth.
    """
    arrays = [np.asarray(m, dtype=np.float64).ravel() for m in maps]
    if len(arrays) < 2:
        raise GradCamError(
            f"the baseline needs at least two patients' maps, got {len(arrays)}"
        )
    pairs = np.array([
        similarity(arrays[i], arrays[j])
        for i in range(len(arrays)) for j in range(i + 1, len(arrays))
    ], dtype=float)
    quantiles = np.percentile(pairs, [5, 25, 50, 75, 95])
    return {
        "n_patients": len(arrays),
        "n_pairs": int(pairs.size),
        "median": float(np.median(pairs)),
        "mean": float(pairs.mean()),
        "sd": float(pairs.std(ddof=1)),
        "min": float(pairs.min()),
        "max": float(pairs.max()),
        "q05": float(quantiles[0]), "q25": float(quantiles[1]),
        "q50": float(quantiles[2]), "q75": float(quantiles[3]),
        "q95": float(quantiles[4]),
        "pairs": [float(v) for v in pairs],
    }


def separability(finals, maps, *, n_boot: int = 2000, seed: int = 1337) -> dict:
    """Are the randomised finals separable from the between-patient baseline?

    **The resampling unit is the PATIENT, not the pair.** Fifteen maps make
    105 pairwise similarities, and those 105 are not 105 independent
    observations -- each map appears in fourteen of them. Bootstrapping over
    pairs would treat one patient's oddity as fourteen pieces of evidence and
    understate the uncertainty, which is the same unit question §5's pairing
    decision turned on.

    So each draw resamples patients with replacement and recomputes both
    quantities. **Pairs between two draws of the SAME patient are excluded**:
    their similarity is 1.0 by construction and would inflate the baseline
    toward agreement.

    A percentile bootstrap rather than BCa: these are two samples over the
    same patients rather than one statistic over paired vectors, and BCa's
    jackknife acceleration is not defined for the difference of two medians
    computed on a resampled set.
    """
    arrays = [np.asarray(m, dtype=np.float64).ravel() for m in maps]
    values = np.asarray(finals, dtype=float)
    if len(arrays) != values.size:
        raise GradCamError(
            f"{values.size} finals against {len(arrays)} maps; the bootstrap "
            "resamples patients, so they must correspond"
        )
    if values.size < 3:
        raise GradCamError("too few patients to resample")

    rng = np.random.default_rng(seed)
    n = values.size
    differences = []
    for _ in range(n_boot):
        draw = rng.integers(0, n, size=n)
        pairs = [
            similarity(arrays[draw[i]], arrays[draw[j]])
            for i in range(n) for j in range(i + 1, n)
            if draw[i] != draw[j]
        ]
        if not pairs:
            continue
        differences.append(float(np.median(values[draw]) - np.median(pairs)))

    if len(differences) < n_boot // 2:
        raise GradCamError(
            f"only {len(differences)}/{n_boot} bootstrap draws yielded any "
            "distinct patient pair; the sample is too small to resample"
        )
    lo, hi = np.percentile(differences, [2.5, 97.5])
    observed = float(
        np.median(values) - between_patient_distribution(arrays)["median"]
    )
    return {
        "resampling_unit": "patient",
        "n_boot": int(n_boot),
        "n_effective_draws": len(differences),
        "observed_difference": observed,
        "ci95": [float(lo), float(hi)],
        "excludes_zero": bool(lo > 0 or hi < 0),
        "reading": (
            "the randomised maps are separable from the between-patient "
            "baseline"
            if lo > 0 or hi < 0 else
            "NOT separable: this cohort cannot establish that the maps depend "
            "on the model's parameters. The per-patient verdict stands as the "
            "pre-registered criterion, but it is being read off a threshold "
            "whose own uncertainty covers the differences it is drawing"
        ),
    }


def randomisation_curve(original, randomised_by_stage: dict) -> dict:
    """Similarity to the original at each randomisation stage, top-down."""
    stages = {
        str(stage): similarity(original, value)
        for stage, value in randomised_by_stage.items()
    }
    return {"stages": stages, "final": stages[list(stages)[-1]] if stages else None}


def randomisation_verdict(curves: list, baseline: float, *,
                          distribution: dict | None = None,
                          separable: dict | None = None) -> dict:
    """Does the map depend on the model's parameters?

    **Pass** if every patient's fully-randomised map is at or below the
    between-patient baseline. A map that survives randomisation is an edge
    detector wearing an explanation's name, and publishing it would be R7's
    fifteenth instance.

    The verdict is per patient and reported per patient: one map surviving is
    not averaged away by fourteen that did not.
    """
    finals = [curve["final"] for curve in curves]
    if not finals or any(value is None for value in finals):
        raise GradCamError("a curve has no final stage; nothing was randomised")
    survivors = [i for i, value in enumerate(finals) if value > baseline]
    verdict = {
        "baseline": float(baseline),
        "finals": [float(value) for value in finals],
        "n_patients": len(finals),
        "survivors": survivors,
        "passes": not survivors,
        "criterion": phase8.RANDOMISATION_TEST["criterion"]["pass_if"],
        "reading": (
            "the maps depend on the model's parameters"
            if not survivors else
            f"{len(survivors)} of {len(finals)} maps survive randomisation -- "
            "they are edge detectors, not explanations. NOTHING MAY BE "
            "PUBLISHED (phase8.RANDOMISATION_TEST)"
        ),
    }
    # **Reported beside the verdict, never folded into it.** The criterion is
    # pre-registered and stays exactly as written; what these add is whether
    # the threshold it uses has enough resolution for the distinctions it is
    # drawing. Changing the rule in response to them is the move Phase 7C
    # corrected twice.
    if distribution is not None:
        verdict["baseline_distribution"] = distribution
        pairs = np.asarray(distribution["pairs"], dtype=float)
        verdict["final_percentiles"] = [
            float((pairs < value).mean()) for value in finals
        ]
    if separable is not None:
        verdict["separability"] = separable
    return verdict


def token_activations_and_gradients(extractor, images, head_weights, *,
                                    block: int | None = None):
    """Patch-token activations and their gradients at the pre-registered layer.

    **Torch only, and deliberately the one function here that needs it.** It
    is a thin adapter: the model, its normalisation and the frozen boundary
    all come from ``extractor`` (``phase8.LIVE_PATH_REUSE``), because a second
    definition of that boundary is exactly the duplication to avoid.
    """
    import torch

    index = phase8.GRAD_CAM_TARGET["index"] if block is None else int(block)
    model = extractor.model

    # **[MEASURED 2026-08-04] The final block is refused HERE, not downstream.**
    # The pre-registration carried ``index: 11`` on a twelve-block model, so
    # the task hooked ``blocks[11]`` -- the final block -- and got the
    # identically-zero gradient this module exists to catch. ``cam`` refused
    # it correctly, but the message read as a property of the layer rather
    # than as the wrong layer having been asked for, which cost a cluster run
    # to diagnose. Refusing at the hook names the actual fault.
    last = len(model.blocks) - 1
    if index == last:
        raise GradCamError(
            f"blocks[{index}] is the FINAL block of {len(model.blocks)} "
            "(0-based), and its patch tokens feed nothing: the head reads "
            "x[:, 0] after a per-token LayerNorm, so their gradient is "
            f"identically zero. The target is blocks[{last - 1}], whose output "
            f"is blocks[{last}]'s attention input. See phase8.GRAD_CAM_TARGET"
        )
    if not 0 <= index < len(model.blocks):
        raise GradCamError(
            f"blocks[{index}] is out of range for {len(model.blocks)} blocks"
        )

    captured: dict = {}

    def hook(_module, _inputs, output):
        tensor = output[0] if isinstance(output, tuple) else output
        tensor.retain_grad()
        captured["tokens"] = tensor

    handle = model.blocks[index].register_forward_hook(hook)
    try:
        batch = extractor.preprocess(images)
        weights = torch.as_tensor(
            np.asarray(head_weights, dtype=np.float32), device=batch.device
        )
        pooled = model.forward_head(model.forward_features(batch),
                                    pre_logits=True)
        (pooled @ weights).sum().backward()
    finally:
        handle.remove()

    tokens = captured["tokens"]
    if tokens.grad is None:
        raise GradCamError(
            f"no gradient reached blocks[{index}]; the graph was not built. "
            "Check that no no_grad context wraps this call -- "
            "FrozenExtractor.__call__ has one and preprocess deliberately "
            "does not (phase8.LIVE_PATH_REUSE)"
        )
    # Patch tokens only -- the CLS token is not part of the spatial grid.
    return (
        tokens[:, 1:, :].detach().cpu().numpy(),
        tokens.grad[:, 1:, :].detach().cpu().numpy(),
    )
