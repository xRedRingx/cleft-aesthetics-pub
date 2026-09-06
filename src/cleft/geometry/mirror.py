"""The mirror-difference symmetry index: the landmark-free geometric baseline.

Phase 4 §3.1. Mirror the staged image about the midline, subtract, and quantify
what is left. No landmarks, no segmentation, no detector -- which is the point,
because no landmark detector is available and the supervision material rejected the one from its own
group.

**Why this is the right primary geometric baseline.** Asymmetry is the clinical
construct, this measures it directly, and it is the cheapest possible instrument.
If a frozen ViT embedding cannot beat a mirror difference, that is a substantial
finding about what the embedding captures -- which makes this a baseline worth
losing to, not a formality.

**G2 only, and mechanically so.** Two reasons, and neither is a preference:

* At G2 the mask *is* the full square, so the midline is exact and every pixel is
  data. At G1 the trapezium corners are white **and symmetric**, so they
  contribute exactly zero difference while still counting in the denominator --
  the index would be diluted by an amount that varies with each patient's aspect
  ratio.
* Patch and region coordinates are normalised to the **trapezium bounding box**.
  At G2 that box is the whole image, so a normalised box maps to pixels directly.
  At G1 it does not, and every region would need the per-image mapping applied
  first.

``require_g2`` therefore refuses rather than warning.

**Do not tune these features against the outcome** (Phase 4 §6). They are defined
from the geometry -- the bands the patch scheme already uses, the anatomy regions
already frozen in ``generators`` -- and measured once. Mean absolute difference is
the only statistic taken, deliberately: adding percentiles or thresholds until
something correlates is fitting the feature set to the labels with extra steps.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from .generators import AnatomyGenerator
from .patches import DEFAULT_BANDS, Patch, PatchConfig
from .trapezium import DEFAULT as DEFAULT_TRAPEZIUM

#: The geometry this index is defined at. See the module docstring.
REQUIRED_GEOMETRY = "g2"

#: [MEASURED 2026-07-28] What this arm scored. Five seeds, G2, mean label, ridge
#: over the 22 features, 5-fold CV over the Phase 1 folds.
#:
#: **It loses to the frozen ViT probe, and the loss is real**: delta -0.095
#: against 0.2529, clearing the arm-means threshold of 0.019 by 5.1x and even the
#: conservative single-run band of 0.046 by 2.1x. A learned representation beats
#: direct geometric asymmetry measurement on this cohort, and that is now a
#: measurement rather than an assumption.
#:
#: **Its SD is split variance**, not initialisation-plus-split: the solver is
#: closed-form, so the only thing the seed moves is which patients are held out
#: (PLAN §4.12.1). Not the same quantity as the probe's 0.0137 even though both
#: are "seed SD".
#:
#: **What makes it worth reporting rather than discarding.** 22 hand-defined
#: features and 23 trained parameters reach **62%** of what an 86M-parameter
#: ImageNet representation achieves, with no learned representation at all. For
#: scale, Bakaki's automated plane-of-symmetry method reached r~0.236 and human
#: raters r~0.457 on related data.
MEASURED_BASELINE = {
    "n_seeds": 5,
    "pccs": (0.140, 0.142, 0.150, 0.175, 0.183),
    "mean": 0.158,
    "sd": 0.019,
    "measures": "inner_val_split_only",
    "shrinkage_range": (0.39, 0.46),
    "delta_vs_probe": -0.095,
    "n_features": 22,
    "n_trained_parameters": 23,
    "fraction_of_probe": 0.62,
    "arm": "mirror-difference index + ridge, g2, label mean, 5-fold CV",
    "measured": "2026-07-28",
}


class MirrorError(RuntimeError):
    """The mirror index cannot be computed as asked."""


def require_g2(geometry: str) -> None:
    """Refuse any geometry but G2, with the reason attached."""
    if geometry != REQUIRED_GEOMETRY:
        raise MirrorError(
            f"the mirror-difference index is defined at {REQUIRED_GEOMETRY!r}, not "
            f"{geometry!r}. At G1 the trapezium corners are white and symmetric, so "
            "they contribute zero difference while still counting in the mean -- the "
            "index would be diluted by an amount that varies with each patient's "
            "aspect ratio. Region coordinates are normalised to the trapezium "
            "bounding box, which is the full square only at G2."
        )


# --------------------------------------------------------------------------
# the residual
# --------------------------------------------------------------------------


def as_float(image: np.ndarray) -> np.ndarray:
    """(H, W) or (H, W, C) on [0, 1]. uint8 input is scaled, float is trusted."""
    array = np.asarray(image)
    if array.ndim not in (2, 3):
        raise MirrorError(f"expected a 2-D or 3-D image, got shape {array.shape}")
    if array.dtype == np.uint8:
        return array.astype(np.float32) / 255.0
    return array.astype(np.float32)


def residual_map(image: np.ndarray) -> np.ndarray:
    """``|image - mirror(image)|``, averaged over channels, as (H, W) on [0, 1].

    **This map is itself mirror-symmetric**, and that is not an incidental
    property -- it is the reason the feature set below keeps only one member of
    each mirrored region pair. ``|a - b| == |b - a|``, so a region and its mirror
    measure the same number. Reporting both would double the feature count
    without adding one bit of information, and would make a ridge fit look like
    it had twice the evidence it has.

    ``mirror_symmetry_error`` asserts the property rather than trusting it.
    """
    array = as_float(image)
    difference = np.abs(array - array[:, ::-1])
    return difference.mean(axis=2) if difference.ndim == 3 else difference


def mirror_symmetry_error(image: np.ndarray) -> float:
    """How far the residual map departs from being mirror-symmetric. Must be ~0.

    A runtime check on the claim the deduplication rests on (R3). If this is ever
    non-zero the feature set is silently discarding real signal.
    """
    residual = residual_map(image)
    return float(np.abs(residual - residual[:, ::-1]).max())


# --------------------------------------------------------------------------
# where the residual is measured
# --------------------------------------------------------------------------


def band_boxes(config: PatchConfig | None = None) -> dict[str, tuple[float, float, float, float]]:
    """Full-width horizontal bands, as ``(x, y, w, h)`` normalised boxes.

    The same bands the patch scheme divides on, so "the asymmetry is in the
    bottom third" means the same thing in both instruments.
    """
    bands = (config or PatchConfig()).bands or DEFAULT_BANDS
    return {band.name: (0.0, band.y0, 1.0, band.y1 - band.y0) for band in bands}


@lru_cache(maxsize=8)
def _anatomy_boxes(config: PatchConfig) -> tuple[tuple[str, tuple], ...]:
    """The anatomy boxes for one config, generated once.

    Cached because the boxes are a pure function of the config and generating
    them is not free -- it runs the whole anatomy generator including coverage
    sampling. Uncached, ``feature_matrix`` rebuilt the entire scheme **once per
    patient**: 237 identical generations per arm, per seed. ``PatchConfig`` is a
    frozen dataclass, so it is hashable and this is safe.

    Returns a tuple rather than a dict so a caller cannot mutate the cached
    value out from under the next one.
    """
    patches = AnatomyGenerator().generate(DEFAULT_TRAPEZIUM, config)
    kept: dict[str, Patch] = {}
    for patch in sorted(patches, key=lambda p: p.id):
        kept.setdefault(patch.band, patch)
    return tuple((name, patch.box) for name, patch in kept.items())


def anatomy_boxes(
    config: PatchConfig | None = None,
) -> dict[str, tuple[float, float, float, float]]:
    """One box per anatomical structure, mirrored pairs collapsed to one.

    Gives a **named** asymmetry per structure -- "the alar base is the asymmetric
    one" -- which is the interpretable part of this baseline and the thing a
    frozen embedding cannot report.

    Pairs collapse because the residual map is mirror-symmetric (see
    ``residual_map``), so the left and right member of a pair are numerically
    identical. The surviving box is the left member, chosen by lower ``id`` so
    the choice is deterministic rather than dict-order.
    """
    return dict(_anatomy_boxes(config or PatchConfig()))


def feature_boxes(
    config: PatchConfig | None = None,
) -> tuple[tuple[str, tuple | None], ...]:
    """Every feature's name and box, in order. ``None`` means the whole map.

    One place the feature set is assembled, used by ``feature_names`` and
    ``mirror_features`` alike, so the order and the contents cannot disagree.
    """
    config = config or PatchConfig()
    boxes: list[tuple[str, tuple | None]] = [(WHOLE, None)]
    boxes += [
        (f"{BAND_PREFIX}{name}", box) for name, box in band_boxes(config).items()
    ]
    boxes += [
        (f"{REGION_PREFIX}{name}", box) for name, box in _anatomy_boxes(config)
    ]
    return tuple(boxes)


def _slice(box: tuple[float, float, float, float], height: int, width: int):
    """A normalised box to array slices, clipped to the image and never empty."""
    x, y, w, h = box
    x0 = int(np.clip(round(x * width), 0, width - 1))
    x1 = int(np.clip(round((x + w) * width), x0 + 1, width))
    y0 = int(np.clip(round(y * height), 0, height - 1))
    y1 = int(np.clip(round((y + h) * height), y0 + 1, height))
    return slice(y0, y1), slice(x0, x1)


def mean_absolute_difference(
    residual: np.ndarray, box: tuple[float, float, float, float] | None = None
) -> float:
    """Mean of the residual over a box, or over the whole map if none is given."""
    if box is None:
        return float(residual.mean())
    rows, columns = _slice(box, residual.shape[0], residual.shape[1])
    return float(residual[rows, columns].mean())


# --------------------------------------------------------------------------
# the feature set
# --------------------------------------------------------------------------

WHOLE = "mad_whole"
BAND_PREFIX = "mad_band_"
REGION_PREFIX = "mad_region_"


def feature_names(config: PatchConfig | None = None) -> list[str]:
    """The feature order, fixed and derived from the geometry.

    Computed from the same source the features are, so the two cannot disagree
    -- a hand-written list would be a second place for the region set to live.
    """
    return [name for name, _ in feature_boxes(config)]


def mirror_features(
    image: np.ndarray,
    config: PatchConfig | None = None,
    boxes: tuple[tuple[str, tuple | None], ...] | None = None,
) -> dict[str, float]:
    """The three granularities of §3.1: whole mask, band, anatomical structure.

    ``boxes`` lets a caller hoist ``feature_boxes`` out of a loop over patients.
    It defaults to computing them, so a single-image call needs nothing extra.
    """
    residual = residual_map(image)
    return {
        name: mean_absolute_difference(residual, box)
        for name, box in (boxes if boxes is not None else feature_boxes(config))
    }


def feature_matrix(
    images: np.ndarray,
    *,
    geometry: str = REQUIRED_GEOMETRY,
    config: PatchConfig | None = None,
) -> tuple[np.ndarray, list[str]]:
    """(N, D) features for a staged tensor, plus the column names.

    The same shape ``phase3.prepare_features`` returns for a frozen backbone, so
    this arm reaches the harness through the same door as the probe and the
    comparison is about the features rather than about the plumbing.
    """
    require_g2(geometry)
    config = config or PatchConfig()
    # Hoisted out of the loop: the boxes are a pure function of the config, and
    # rebuilding them per patient meant generating the anatomy scheme 237 times
    # for one arm.
    boxes = feature_boxes(config)
    names = [name for name, _ in boxes]

    rows = np.zeros((len(images), len(names)), dtype=np.float32)
    for index, image in enumerate(images):
        features = mirror_features(image, config, boxes)
        rows[index] = [features[name] for name in names]
    return rows, names


def describe(config: PatchConfig | None = None) -> dict:
    """What this arm computed, for ``metrics.json``. SHAREABLE -- no patient data."""
    config = config or PatchConfig()
    return {
        "index": "mirror_difference",
        "geometry": REQUIRED_GEOMETRY,
        "statistic": "mean_absolute_difference",
        "n_features": len(feature_names(config)),
        "bands": sorted(band_boxes(config)),
        "regions": sorted(anatomy_boxes(config)),
        "mirror_pairs_collapsed": True,
        "note": (
            "The residual map is mirror-symmetric, so each mirrored region pair "
            "contributes one feature, not two. Features are defined from the "
            "geometry and measured once; they are not tuned against the label."
        ),
    }
