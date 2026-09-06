"""The other two patch generators: random (control) and anatomy (ablation).

Brief §4.2 and §4.3. All three share the interface in ``patches.py``, so the
comparison between them is about **placement** and nothing else.

----------------------------------------------------------------------------
Why random is mandatory
----------------------------------------------------------------------------
Charm (CVPR 2025) found random patch selection **beat** saliency, entropy,
frequency and gradient selection on small aesthetic datasets. So a random control
is not a formality: if it matches anatomy, prior clinical knowledge is buying
nothing, and that is a finding rather than a disappointment.

----------------------------------------------------------------------------
Why random is mirrored [DECIDED]
----------------------------------------------------------------------------
Random patches are placed as **mirrored pairs**, matching the grid's count, its
size multiset, and its pairing structure. Unconstrained random placement would
make the comparison confound two things at once: placement, and whether the
scheme has mirror pairs at all. Asymmetry is a comparison between paired
locations, so a control without pairs is not a control for placement -- it is a
different instrument.

``mirrored=False`` is available for investigating whether the pairing itself
carries the signal, which is a separate and legitimate question.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .patches import (
    Patch,
    PatchConfig,
    PatchError,
    apply_boundary_rule,
    get,
    pair_mirrors,
    register,
)
from .trapezium import DEFAULT as DEFAULT_TRAPEZIUM
from .trapezium import Trapezium

MAX_PLACEMENT_ATTEMPTS = 500


# --------------------------------------------------------------------------
# random -- the control
# --------------------------------------------------------------------------


def _fits(x: float, y: float, w: float, h: float, mask: Trapezium) -> bool:
    """Is the whole box inside the mask?

    The trapezium narrows upward, so the binding constraint is the box's TOP
    edge -- check there and the rest follows.
    """
    if x < 0.0 or y < 0.0 or x + w > 1.0 or y + h > 1.0:
        return False
    top = float(np.clip(y, 0.0, 1.0))
    return mask.left_edge_at(top) <= x and x + w <= mask.right_edge_at(top)


@dataclass
class RandomGenerator:
    """Random placement, matched to the grid in count and size distribution."""

    name: str = "random"
    mirrored: bool = True

    def generate(
        self, mask: Trapezium = DEFAULT_TRAPEZIUM, config: PatchConfig | None = None
    ) -> list[Patch]:
        config = config or PatchConfig()
        template = get("grid").generate(mask, replace(config, boundary_rule="keep"))
        rng = np.random.default_rng(config.patch_seed)

        singles, pairs = _split_by_mirror(template)
        patches: list[Patch] = []
        next_id = 0

        # Self-mirrored patches stay straddling the midline; only their height
        # moves. A centre patch placed off-centre would not be self-mirrored.
        for item in singles:
            y = self._sample_y(item, mask, rng, x=0.5 - item.w / 2)
            patches.append(
                Patch(next_id, "random", 0.5 - item.w / 2, y, item.w, item.h)
            )
            next_id += 1

        for item in pairs:
            x, y = self._sample_pair(item, mask, rng)
            patches.append(Patch(next_id, "random", x, y, item.w, item.h))
            patches.append(
                Patch(next_id + 1, "random", 1.0 - x - item.w, y, item.w, item.h)
            )
            next_id += 2

        patches = apply_boundary_rule(patches, mask, config)
        return pair_mirrors(patches) if self.mirrored else patches

    def _sample_y(self, item: Patch, mask: Trapezium, rng, x: float) -> float:
        for _ in range(MAX_PLACEMENT_ATTEMPTS):
            y = float(rng.uniform(0.0, 1.0 - item.h))
            if _fits(x, y, item.w, item.h, mask):
                return y
        raise PatchError(
            f"could not place a {item.w:.3f}x{item.h:.3f} centre patch inside the "
            f"mask in {MAX_PLACEMENT_ATTEMPTS} attempts"
        )

    def _sample_pair(self, item: Patch, mask: Trapezium, rng) -> tuple[float, float]:
        """Place the LEFT member; the right is its reflection."""
        for _ in range(MAX_PLACEMENT_ATTEMPTS):
            y = float(rng.uniform(0.0, 1.0 - item.h))
            left_limit = mask.left_edge_at(float(y))
            right_limit = 0.5 - item.w
            if right_limit < left_limit:
                continue
            x = float(rng.uniform(left_limit, right_limit))
            if _fits(x, y, item.w, item.h, mask):
                return x, y
        raise PatchError(
            f"could not place a {item.w:.3f}x{item.h:.3f} patch pair inside the "
            f"mask in {MAX_PLACEMENT_ATTEMPTS} attempts. A patch wider than half "
            "the mask cannot sit entirely on one side of the midline."
        )


def _split_by_mirror(template: list[Patch]) -> tuple[list[Patch], list[Patch]]:
    """Split a patch set into self-mirrored singles and one member per pair."""
    singles, pairs, seen = [], [], set()
    for patch in template:
        if patch.id in seen:
            continue
        if patch.mirror_id == patch.id:
            singles.append(patch)
            seen.add(patch.id)
        else:
            pairs.append(patch)
            seen.update({patch.id, patch.mirror_id})
    return singles, pairs


register(RandomGenerator())


# --------------------------------------------------------------------------
# anatomy -- the ablation
# --------------------------------------------------------------------------

#: 27 anatomy-anchored regions: **9 midline + 9 mirrored pairs**.
#:
#: [DECIDED, and to be checked on the contact sheet.] These placements are a
#: design choice made from the anatomy of a nasolabial crop, not a measurement,
#: and I am not a clinician. Section 7 of the brief exists partly to check them:
#: do the alar bases sit off the eyes/nose seam, does the centre of the bottom
#: band cover the philtrum and cupid's bow. If they land badly the fix is to move
#: them here, from anatomy, NOT to tune them against a metric.
#:
#: Coordinates are (y_centre, half_offset_from_midline, width, height) normalised
#: to the trapezium bbox. Offset 0.0 means a midline structure.
MIDLINE_REGIONS: tuple[tuple[str, float, float, float], ...] = (
    ("glabella", 0.06, 0.16, 0.09),
    ("nasal_root", 0.16, 0.14, 0.09),
    ("nasal_dorsum_upper", 0.28, 0.13, 0.10),
    ("nasal_dorsum_lower", 0.40, 0.14, 0.10),
    ("nasal_tip", 0.51, 0.16, 0.10),
    ("columella", 0.60, 0.10, 0.08),
    ("subnasale", 0.67, 0.12, 0.07),
    ("philtrum", 0.75, 0.13, 0.09),
    ("labial_tubercle", 0.86, 0.15, 0.11),
)

BILATERAL_REGIONS: tuple[tuple[str, float, float, float, float], ...] = (
    # (name, y_centre, offset, width, height)
    ("medial_canthus", 0.08, 0.13, 0.11, 0.09),
    ("lateral_orbit", 0.09, 0.24, 0.12, 0.09),
    ("nasal_sidewall", 0.30, 0.14, 0.11, 0.11),
    ("alar_rim", 0.48, 0.16, 0.12, 0.10),
    ("alar_base", 0.58, 0.17, 0.12, 0.09),
    ("nostril_sill", 0.63, 0.09, 0.09, 0.07),
    ("philtral_column", 0.75, 0.07, 0.08, 0.10),
    ("vermillion_border", 0.81, 0.15, 0.12, 0.07),
    ("commissure", 0.84, 0.24, 0.11, 0.08),
)


#: The accepted characteristics of the anatomy scheme at its settled parameters,
#: measured 2026-07-28 after three contact-sheet sweeps. **The scheme is
#: finished**: v_offset 0.04, scale 1.80, scale_x 1.60, box_scale_x 1.00.
#:
#: Recorded so they cannot drift unnoticed. Any change to the region table or to
#: a default moves these numbers, and the regression test that checks them will
#: say so rather than letting the ablation quietly become a different experiment.
ACCEPTED_ANATOMY: dict[str, float | int] = {
    "redundancy": 1.392,
    "n_overlapping_pairs": 53,
    "n_pairs": 351,
    "max_pairwise_iou": 0.28236,
    "frame_covered": 0.7018,
    "n_clamped": 0,
    # G1 only, and the reason this is a G2-only scheme (PLAN §4.4).
    "lateral_orbit_coverage_g1": 0.2356,
}

#: Mean patch area as a fraction of the grid's. See PLAN §4.5: full parity would
#: need scale ~2.48, and 2.4 already clamps, so this gap is structural.
ANATOMY_AREA_RATIO_TO_GRID = 0.527


@dataclass
class AnatomyGenerator:
    """Clinically predefined regions -- a different hypothesis, not a variant.

    The grid asks whether data-driven multi-scale patches let graph learning
    discover which regional relationships matter. This asks whether injecting
    prior anatomical knowledge helps. Node count is matched at 27 so the
    comparison is about placement rather than capacity.
    """

    name: str = "anatomy"

    def generate(
        self, mask: Trapezium = DEFAULT_TRAPEZIUM, config: PatchConfig | None = None
    ) -> list[Patch]:
        config = config or PatchConfig()
        scale = config.anatomy_scale
        spread = config.anatomy_scale_x
        box_x = config.anatomy_box_scale_x
        drop = config.anatomy_v_offset
        patches: list[Patch] = []
        next_id = 0

        for name, y_centre, width, height in MIDLINE_REGIONS:
            w, h = width * scale * box_x, height * scale
            x, clamped = _clamp_x(0.5 - w / 2, w)
            patches.append(
                Patch(next_id, name, x, _top(y_centre + drop, h), w, h, clamped=clamped)
            )
            next_id += 1

        for name, y_centre, offset, width, height in BILATERAL_REGIONS:
            # Three separate multipliers, each doing one thing: size, box width,
            # spread. Folding any two together would make them interact, which is
            # exactly what they exist to avoid.
            w, h = width * scale * box_x, height * scale
            y = _top(y_centre + drop, h)
            left_x, clamped = _clamp_x(0.5 - offset * spread - w / 2, w)
            # Mirror the CLAMPED position, so the pair stays a pair. Mirroring the
            # unclamped one would leave the two members at different distances
            # from the midline -- a left-right bias in the asymmetry instrument.
            patches.append(Patch(next_id, name, left_x, y, w, h, clamped=clamped))
            patches.append(
                Patch(next_id + 1, name, 1.0 - left_x - w, y, w, h, clamped=clamped)
            )
            next_id += 2

        patches = apply_boundary_rule(patches, mask, config)
        return pair_mirrors(patches)


def _top(y_centre: float, height: float) -> float:
    """Convert a centre height to a top edge, clamped into the unit square."""
    return float(np.clip(y_centre - height / 2.0, 0.0, 1.0 - height))


def _clamp_x(x: float, width: float) -> tuple[float, bool]:
    """Pull a box back inside the frame, reporting whether it had to move.

    At anatomy_scale_x = 2.0 the outermost pairs -- the commissures at offset
    0.24 -- run past the edge: 0.5 - 0.48 - w/2 is negative. Left alone that is a
    region sitting half outside the image, which at contact-sheet panel scale
    looks like a slightly small box rather than a broken one.
    """
    if width > 1.0:
        return 0.0, True
    clamped = float(np.clip(x, 0.0, 1.0 - width))
    return clamped, abs(clamped - x) > 1e-12


register(AnatomyGenerator())


# --------------------------------------------------------------------------
# comparability
# --------------------------------------------------------------------------


def compare(mask: Trapezium = DEFAULT_TRAPEZIUM, config: PatchConfig | None = None) -> dict:
    """Side-by-side aggregate description of all three generators.

    The node counts must match. The AREA distributions do not, and that is worth
    seeing rather than assuming: anatomy regions are genuinely smaller than grid
    patches, which is part of the hypothesis but also a potential confound.
    ``PatchConfig.anatomy_scale`` exists to equalise it if that matters.
    """
    config = config or PatchConfig()
    out = {}
    for name in ("grid", "random", "anatomy"):
        patches = get(name).generate(mask, config)
        areas = [p.w * p.h for p in patches]
        out[name] = {
            "n_patches": len(patches),
            "mean_area": round(float(np.mean(areas)), 5),
            "min_area": round(float(np.min(areas)), 5),
            "max_area": round(float(np.max(areas)), 5),
            "mean_coverage": round(float(np.mean([p.coverage for p in patches])), 4),
        }
    return out
