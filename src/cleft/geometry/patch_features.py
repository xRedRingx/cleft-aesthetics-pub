"""Patch extraction for the Phase 4 frozen probes (§3.3, §3.4).

Turns a staged tensor into per-patient patch stacks, using **Phase 2's mapping
machinery unchanged**. Nothing here re-derives a box: ``patches.get`` generates
the scheme, ``mapping.map_patches`` places it in each image's own pixel
coordinates through that image's ``content_box``, and ``mapping.extract`` crops
and resizes. Those three are already asserted by the Phase 2 suite, and a second
implementation here would be a second thing to keep in step.

**Why the per-image mapping still matters at G2.** It is tempting to skip it:
under G2 the trapezium bounding box is the whole square, so a normalised box maps
to pixels by multiplying by 224. That is true for the *mask*, and false for the
*content*. Cleft aspect ratios span a factor of two, so after pad-to-square the
same anatomy sits at a different height for different patients whatever the
geometry -- which is the defect ``assert_varies_with_aspect_ratio`` exists to
catch. The mapping is applied at both geometries for the same reason it was
applied in Phase 2.

**What this deliberately does not do: relate the patches to each other.** The
probes mean-pool the patch embeddings, which discards every relationship between
regions -- and relationships between regions are the entire hypothesis the patch
scheme exists to test. Pooling is exactly what SR-GNN replaces with message
passing in Phase 7.

So these arms are a **lower bound on what patches can do**, not a test of the
patch hypothesis. If a pooled patch arm loses to the whole-image probe, the
finding is "mean-pooled patch embeddings do not beat whole-image embeddings" and
**not** "region structure does not help". The write-up must not collapse those
two, and §3.3's framing as "the first clean test of Q4" should be read with that
limit attached.
"""

from __future__ import annotations

import numpy as np

from . import patches as patch_module
from .mapping import extract, map_patches
from .patches import Patch, PatchConfig
from .staging import Staged
from .trapezium import DEFAULT as G1_TRAPEZIUM
from .trapezium import Trapezium

#: Under G2 the mask IS the full square, so nothing is excluded. Same value
#: ``contact`` uses; imported rather than redefined would be circular, so it is
#: constructed identically and asserted equal in the tests.
G2_TRAPEZIUM = Trapezium(top_half_width=0.5, bot_half_width=0.5)

GEOMETRIES = ("g1", "g2")

#: Which schemes the probes compare. "whole" is the Phase 3 bar and takes no
#: patches at all -- named here so an arm list can be written uniformly.
SCHEMES = ("whole", "grid", "anatomy", "random")

#: How the per-patch embeddings become one vector per patient.
#:
#: ``concat`` is deliberately absent. 27 patches x 768 dimensions is 20,736
#: features for 237 patients -- roughly 136 features per training sample -- which
#: makes the linear head a memorisation exercise rather than a probe. It is not
#: an option to be tried later either: it would need a different claim procedure,
#: not a different config value.
POOLINGS = ("mean", "max")

DEFAULT_POOLING = "mean"

#: Patches are resized to this before the backbone sees them. It is the
#: backbone's own input size rather than ``PatchConfig.output_size`` (64), which
#: sizes the Phase 2 contact sheet. Feeding 64px crops to a 224px ViT would mean
#: the model interpolating them anyway, less well.
BACKBONE_INPUT_SIZE = 224


class PatchFeatureError(RuntimeError):
    """Patch features could not be built."""


def trapezium_for(geometry: str) -> Trapezium:
    if geometry == "g1":
        return G1_TRAPEZIUM
    if geometry == "g2":
        return G2_TRAPEZIUM
    raise PatchFeatureError(
        f"unknown geometry {geometry!r}; expected one of {GEOMETRIES}"
    )


def patches_for(
    scheme: str, geometry: str, config: PatchConfig | None = None
) -> list[Patch]:
    """The patch set for one scheme at one geometry.

    Generated from the shared registry, so the probe cannot use a different set
    from the one Phase 2 measured, rendered and froze.
    """
    if scheme == "whole":
        raise PatchFeatureError(
            "the 'whole' scheme has no patches -- it is the Phase 3 whole-image "
            "arm and reaches the harness through prepare_features directly"
        )
    if scheme not in SCHEMES:
        raise PatchFeatureError(
            f"unknown scheme {scheme!r}; expected one of {SCHEMES}"
        )
    generator = patch_module.get(scheme)
    return generator.generate(trapezium_for(geometry), config or PatchConfig())


def staged_from_row(image: np.ndarray, row: dict) -> Staged:
    """Rebuild the ``Staged`` for one patient from ``geometry.csv``.

    The artifact records ``content_x/y/w/h`` per patient precisely so this is
    possible without re-staging from the original photograph -- which the laptop
    could not do and the cluster should not need to.
    """
    try:
        content_box = (
            int(row["content_x"]),
            int(row["content_y"]),
            int(row["content_w"]),
            int(row["content_h"]),
        )
    except KeyError as exc:
        raise PatchFeatureError(
            f"geometry.csv is missing {exc.args[0]!r}. The staged artifact must "
            "record the content box per patient; without it the per-image mapping "
            "cannot be applied and every patient would get the same pixel boxes."
        ) from exc

    aspect_ratio = float(row.get("aspect_ratio", 1.0))
    # source_size is used only for reporting the aspect ratio back; the mapping
    # itself reads content_box alone.
    return Staged(
        image=image,
        source_size=(int(round(aspect_ratio * 1000)), 1000),
        content_box=content_box,
        scale=1.0,
    )


def extract_for_patient(
    image: np.ndarray,
    row: dict,
    patches: list[Patch],
    output_size: int = BACKBONE_INPUT_SIZE,
) -> np.ndarray:
    """(P, S, S, 3) -- every patch of one patient, cropped and resized."""
    staged = staged_from_row(image, row)
    return extract(staged, map_patches(staged, patches), output_size)


def iter_patient_patches(
    images: np.ndarray,
    geometry_rows: list[dict],
    patches: list[Patch],
    output_size: int = BACKBONE_INPUT_SIZE,
):
    """Yield one patient's patch stack at a time.

    A generator rather than one array: 237 patients x 27 patches at 224x224x3 is
    about 960 MB materialised, and the consumer only ever needs one patient's
    worth before handing it to the backbone.
    """
    if len(images) != len(geometry_rows):
        raise PatchFeatureError(
            f"{len(images)} staged images against {len(geometry_rows)} geometry "
            "rows -- the staged tensor and geometry.csv must agree row for row"
        )
    for image, row in zip(images, geometry_rows):
        yield extract_for_patient(image, row, patches, output_size)


def pool(embeddings: np.ndarray, method: str = DEFAULT_POOLING) -> np.ndarray:
    """(P, D) per-patch embeddings -> (D,) for one patient."""
    if method == "mean":
        return np.asarray(embeddings, dtype=np.float32).mean(axis=0)
    if method == "max":
        return np.asarray(embeddings, dtype=np.float32).max(axis=0)
    raise PatchFeatureError(
        f"unknown pooling {method!r}; expected one of {POOLINGS}. 'concat' is "
        "excluded by design -- see POOLINGS."
    )


def describe(
    scheme: str,
    geometry: str,
    patches: list[Patch],
    pooling: str = DEFAULT_POOLING,
    output_size: int = BACKBONE_INPUT_SIZE,
) -> dict:
    """What this arm's features are. SHAREABLE -- geometry and counts only."""
    return {
        "scheme": scheme,
        "geometry": geometry,
        "n_patches": len(patches),
        "pooling": pooling,
        "patch_output_size": output_size,
        "n_clamped": sum(1 for p in patches if p.clamped),
        "mean_coverage": round(
            float(np.mean([p.coverage for p in patches])), 4
        ),
        "min_coverage": round(float(min(p.coverage for p in patches)), 4),
        "bands": sorted({p.band for p in patches}),
        "pooling_note": (
            "Mean pooling discards every relationship between regions, which is "
            "half of what the patch scheme exists to test -- SR-GNN replaces "
            "pooling with message passing in Phase 7. So a pooled patch arm "
            "losing to the whole-image probe shows that MEAN-POOLED patch "
            "embeddings do not beat whole-image embeddings, NOT that region "
            "structure does not help. That limit applies to the ABSOLUTE numbers. "
            "It does NOT weaken the grid-vs-anatomy-vs-random comparison: pooling "
            "is identical across all three, so a difference between them is a "
            "difference in PLACEMENT. Q4 has a placement half and a relational "
            "half; these arms answer the placement half cleanly and defer the "
            "relational half to Phase 7. Do not discount the scheme comparison "
            "along with the absolute numbers."
        ),
    }
