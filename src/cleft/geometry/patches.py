"""Patch generation: one interface, three generators.

Brief §4. The three generators — grid (primary), anatomy (ablation), random
(control) — must share **one interface**, because the comparison between them is
supposed to be about *placement*, and any difference in how they are called or
counted would confound that.

    generate(mask, config) -> list[Patch]

Patch coordinates are normalised to the **trapezium bounding box**, never to the
padded square. Cleft aspect ratios vary by a factor of two, so a box fixed in
padded-square coordinates would land on one patient's chin and another's nose.
``cleft.geometry.staging.content_box_to_pixels`` does the per-image mapping.

The grid construction (brief §4.1, [DECIDED] and fully configurable, because supervision
explicitly left the count to the student):

    band    anatomy   columns   patches
    top     eyes         3         5
    middle  nose         5         9
    bottom  lips         7        13
                                 ---
                                  27

With 50% overlap the stride is half a patch width, so *n* columns give **2n−1**
positions. That is odd, so every band has a centre patch straddling the midline
and patch *i* mirrors patch *(2n−2−i)*. The generator asserts that rather than
trusting it: mirror pairs are the instrument the asymmetry signal is measured
with, and if they broke silently every downstream comparison would still run.

**Why overlap at all:** with stride *w*/2 any feature narrower than *w*/2 is
fully contained in at least one patch. Without overlap a feature straddling a
seam lives in no single patch, and on this crop the seams fall badly — at five
columns the philtral columns sit near 0.42 and 0.58, almost exactly on the 0.4
and 0.6 boundaries.

**Resolution by coverage:** every patch is resized to the same output size, so a
bottom patch covering 1/7 of the width carries more detail than a top patch
covering 1/3. Output size is deliberately NOT varied per band.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Callable, Protocol

import numpy as np

from .trapezium import DEFAULT as DEFAULT_TRAPEZIUM
from .trapezium import Trapezium

#: How a patch that crosses the trapezium edge is handled (brief §4.4).
#:
#: Under G1 the outer columns of each band cross the mask edge and contain white.
#: Under G2 the mask IS the full square and the problem vanishes — which is why
#: geometry and patch scheme interact and cannot be treated as independent
#: factors in the arm list.
BOUNDARY_RULES = ("keep", "drop", "clip")

#: Sampling resolution for the coverage estimate. Fixed, so coverage is
#: reproducible rather than dependent on an incidental default.
#:
#: Held at 64. A reduction to 32 was tried on 2026-07-28 to save suite time and
#: reverted: it bought almost nothing (coverage was not the bottleneck) while
#: shifting every recorded coverage figure in the fourth decimal, including the
#: ACCEPTED_ANATOMY constants. Changing an estimator's resolution changes the
#: measurements that were accepted under it, so it is not a free optimisation.
COVERAGE_SAMPLES = 64


class PatchError(ValueError):
    """The patch scheme is not valid."""


@dataclass(frozen=True)
class BandSpec:
    """One horizontal band: its extent, and how finely it is divided."""

    name: str
    y0: float
    y1: float
    columns: int

    @property
    def n_patches(self) -> int:
        """50% overlap makes this 2n-1, which is odd, which gives the centre patch."""
        return 2 * self.columns - 1


#: Defaults. Band boundaries are exact thirds; the brief is explicit that if they
#: land badly on the contact sheet the fix is to MOVE them, not to tune them
#: against a metric — which is why they are configuration and not constants.
DEFAULT_BANDS: tuple[BandSpec, ...] = (
    BandSpec("top", 0.0, 1.0 / 3.0, 3),
    BandSpec("middle", 1.0 / 3.0, 2.0 / 3.0, 5),
    BandSpec("bottom", 2.0 / 3.0, 1.0, 7),
)


@dataclass(frozen=True)
class PatchConfig:
    bands: tuple[BandSpec, ...] = DEFAULT_BANDS
    #: 0.5 = 50% overlap = stride of half a patch width.
    overlap: float = 0.5
    #: One size for every patch, whatever its coverage. See "resolution by
    #: coverage" above; do not make this per-band.
    output_size: int = 64
    boundary_rule: str = "keep"
    #: Used only by the "drop" rule.
    min_coverage: float = 0.5
    #: Used only by the random generator. It lives in the config rather than on
    #: the generator so that a run is reproducible from its config file alone --
    #: the same reason nothing scientific is a CLI flag.
    patch_seed: int = 1337
    # ---- anatomy set placement -------------------------------------------
    # Three ORTHOGONAL controls, so "too high", "too narrow" and "too small" can
    # be corrected independently rather than by trading one against another.
    # The first contact sheet showed the set was wrong in all three directions at
    # once while being internally consistent -- regions on the right structures
    # relative to each other, identical across faces. That is a systematic offset
    # in the coordinates, not a mapping defect, so it wants global transforms
    # rather than 27 individual edits.
    #
    # All default to the current values: nothing changes unless asked.

    #: Shift every region DOWN. **0.04 chosen from the 2026-07-28 sweep** (row 4).
    anatomy_v_offset: float = 0.04
    #: Multiply region SIZE, both dimensions.
    #: **1.80 chosen from the 2026-07-28 sweep 3** (row 3). 2.0 and above clamp.
    anatomy_scale: float = 1.80
    #: Multiply the horizontal DISTANCE FROM THE MIDLINE -- how far apart the two
    #: members of a pair sit. SPREAD, not size.
    #: **1.60 chosen from the 2026-07-28 sweep 2** (row 3). 2.0 was rejected: it
    #: clamped the commissures and lateral orbits against the frame, so it only
    #: looked wider.
    anatomy_scale_x: float = 1.60
    #: Multiply region WIDTH only, on top of anatomy_scale. BOX WIDTH, not spread.
    #:
    #: Separate from anatomy_scale_x because "the set is too narrow" has two
    #: possible causes that look nearly identical on a small panel: the pairs sit
    #: too close together, or each box is too thin. Settling on one when the real
    #: problem was the other would leave the regions still wrong, and the sheet
    #: would not show which.
    anatomy_box_scale_x: float = 1.0

    def __post_init__(self) -> None:
        if self.boundary_rule not in BOUNDARY_RULES:
            raise PatchError(
                f"unknown boundary_rule {self.boundary_rule!r}; expected one of "
                f"{BOUNDARY_RULES}"
            )
        if not 0.0 < self.overlap < 1.0:
            raise PatchError(f"overlap must be in (0, 1), got {self.overlap}")
        if not self.bands:
            raise PatchError("at least one band is required")
        for band in self.bands:
            if band.columns < 1:
                raise PatchError(f"band {band.name!r} has {band.columns} columns")
            if not 0.0 <= band.y0 < band.y1 <= 1.0:
                raise PatchError(
                    f"band {band.name!r} has an invalid extent [{band.y0}, {band.y1}]"
                )

    @property
    def expected_patches(self) -> int:
        return sum(band.n_patches for band in self.bands)


@dataclass(frozen=True)
class Patch:
    """A box normalised to the trapezium bounding box."""

    id: int
    band: str
    x: float
    y: float
    w: float
    h: float
    coverage: float = 1.0
    mirror_id: int | None = None
    #: True if the box had to be pulled back inside the frame. At high spread the
    #: outermost pairs run off the edge, and a commissure sliding off is invisible
    #: on a contact sheet at panel scale -- so it is recorded, labelled and
    #: reported rather than left to be noticed.
    clamped: bool = False

    @property
    def centre_x(self) -> float:
        return self.x + self.w / 2.0

    @property
    def box(self) -> tuple[float, float, float, float]:
        return (self.x, self.y, self.w, self.h)

    def mirrored_box(self) -> tuple[float, float, float, float]:
        """This patch reflected about the midline."""
        return (1.0 - self.x - self.w, self.y, self.w, self.h)


# --------------------------------------------------------------------------
# coverage and the trapezium boundary
# --------------------------------------------------------------------------


def coverage_of(patch: Patch, mask: Trapezium, samples: int = COVERAGE_SAMPLES) -> float:
    """Fraction of the patch's area lying inside the mask.

    Under G2 this is 1.0 everywhere. Under G1 the outer columns of the top band
    are the ones that lose area, because the trapezium is narrowest at the brow.
    """
    ys = patch.y + (np.arange(samples) + 0.5) / samples * patch.h
    xs = patch.x + (np.arange(samples) + 0.5) / samples * patch.w

    # One broadcast rather than a loop over rows. The loop version made `samples`
    # numpy calls per patch and was the largest single cost in the test suite.
    left = mask.left_edge_array(ys)[:, None]
    right = mask.right_edge_array(ys)[:, None]
    inside = (xs[None, :] >= left) & (xs[None, :] <= right)
    return float(inside.mean())


def _clip_to_mask(patch: Patch, mask: Trapezium) -> Patch | None:
    """Shrink a patch to the mask's narrowest extent over its own height.

    The trapezium narrows upward, so the binding constraint is the patch's TOP
    edge. Clipping there keeps the patch rectangular and entirely inside the
    mask. It stays mirror-symmetric because the clip is symmetric.
    """
    top = float(np.clip(patch.y, 0.0, 1.0))
    left = mask.left_edge_at(top)
    right = mask.right_edge_at(top)

    new_x = max(patch.x, left)
    new_right = min(patch.x + patch.w, right)
    if new_right <= new_x:
        return None
    return replace(patch, x=new_x, w=new_right - new_x)


# --------------------------------------------------------------------------
# mirror symmetry -- asserted, never assumed
# --------------------------------------------------------------------------


def pair_mirrors(patches: list[Patch], tolerance: float = 1e-9) -> list[Patch]:
    """Attach each patch's mirror id, raising if any patch has no partner.

    Mirror pairs are the instrument that makes asymmetry measurable. If they
    broke, nothing would error and every downstream comparison would still
    produce numbers — so this is a hard failure.
    """
    out: list[Patch] = []
    by_band: dict[str, list[Patch]] = {}
    for patch in patches:
        by_band.setdefault(patch.band, []).append(patch)

    for band, members in by_band.items():
        for patch in members:
            target = patch.mirrored_box()
            match = next(
                (
                    other
                    for other in members
                    if abs(other.x - target[0]) <= tolerance
                    and abs(other.w - target[2]) <= tolerance
                    and abs(other.y - patch.y) <= tolerance
                ),
                None,
            )
            if match is None:
                raise PatchError(
                    f"patch {patch.id} in band {band!r} at x={patch.x:.6f} "
                    f"w={patch.w:.6f} has no mirror partner. Mirror pairs are how "
                    "asymmetry is measured; a band without them silently stops "
                    "measuring the thing being modelled."
                )
            out.append(replace(patch, mirror_id=match.id))
    return sorted(out, key=lambda p: p.id)


def assert_centre_patch(patches: list[Patch], config: PatchConfig) -> None:
    """Every band must have exactly one patch straddling the midline."""
    for band in config.bands:
        members = [p for p in patches if p.band == band.name]
        if not members:
            continue
        centred = [p for p in members if abs(p.centre_x - 0.5) <= 1e-9]
        if len(centred) != 1:
            raise PatchError(
                f"band {band.name!r} has {len(centred)} centre patches, expected 1. "
                f"With {band.columns} columns and 50% overlap there are "
                f"{band.n_patches} positions, which is odd, so exactly one must "
                "straddle the midline."
            )


# --------------------------------------------------------------------------
# the generator interface
# --------------------------------------------------------------------------


class PatchGenerator(Protocol):
    """All three generators satisfy this, so the ablation compares placement."""

    name: str

    def generate(self, mask: Trapezium, config: PatchConfig) -> list[Patch]:
        ...


GENERATORS: dict[str, PatchGenerator] = {}


def register(generator: PatchGenerator) -> PatchGenerator:
    GENERATORS[generator.name] = generator
    return generator


def get(name: str) -> PatchGenerator:
    if name not in GENERATORS:
        raise PatchError(f"unknown generator {name!r}; registered: {sorted(GENERATORS)}")
    return GENERATORS[name]


def apply_boundary_rule(
    patches: list[Patch], mask: Trapezium, config: PatchConfig
) -> list[Patch]:
    """Handle patches crossing the mask edge, then re-pair mirrors."""
    scored = [replace(p, coverage=coverage_of(p, mask)) for p in patches]

    if config.boundary_rule == "keep":
        kept = scored
    elif config.boundary_rule == "drop":
        kept = [p for p in scored if p.coverage >= config.min_coverage]
    else:  # clip
        clipped = [(p, _clip_to_mask(p, mask)) for p in scored]
        kept = [c for _, c in clipped if c is not None]
        kept = [replace(p, coverage=coverage_of(p, mask)) for p in kept]

    if not kept:
        raise PatchError(
            f"the {config.boundary_rule!r} boundary rule removed every patch "
            f"(min_coverage={config.min_coverage}). Nothing would be fed to the model."
        )
    return kept


# --------------------------------------------------------------------------
# the grid generator -- the primary
# --------------------------------------------------------------------------


@dataclass
class GridGenerator:
    """the specification given at supervision: mask first, patches inside the mask, grid positions."""

    name: str = "grid"

    def generate(
        self, mask: Trapezium = DEFAULT_TRAPEZIUM, config: PatchConfig | None = None
    ) -> list[Patch]:
        config = config or PatchConfig()
        patches: list[Patch] = []
        next_id = 0

        for band in config.bands:
            width = 1.0 / band.columns
            stride = width * config.overlap
            for position in range(band.n_patches):
                x = position * stride
                # Floating-point drift would put the last patch a hair past 1.0
                # and break the mirror match; pin it instead of widening the
                # tolerance, which would hide a real asymmetry.
                if position == band.n_patches - 1:
                    x = 1.0 - width
                patches.append(
                    Patch(
                        id=next_id,
                        band=band.name,
                        x=x,
                        y=band.y0,
                        w=width,
                        h=band.y1 - band.y0,
                    )
                )
                next_id += 1

        patches = apply_boundary_rule(patches, mask, config)
        assert_centre_patch(patches, config)
        return pair_mirrors(patches)


register(GridGenerator())


def iou(a: Patch, b: Patch) -> float:
    """Intersection over union of two patch boxes."""
    ix = max(0.0, min(a.x + a.w, b.x + b.w) - max(a.x, b.x))
    iy = max(0.0, min(a.y + a.h, b.y + b.h) - max(a.y, b.y))
    intersection = ix * iy
    union = a.w * a.h + b.w * b.h - intersection
    return intersection / union if union > 0 else 0.0


def overlap_report(patches: list[Patch], resolution: int = 256) -> dict:
    """How distinct the regions still are, as they grow.

    Enlarging the anatomy set is not free: at some size 27 regions stop being 27
    structures and become 27 views of the same area. The scheme's entire claim is
    that clinically predefined *locations* carry information, so a set that has
    collapsed into mutual overlap has quietly stopped testing that claim while
    still producing 27 nodes and a plausible number.

    ``redundancy`` is the clearest single figure: the mean number of regions
    stacked on each covered pixel. 1.0 is a partition; 3.0 means the average
    covered point is inside three regions.
    """
    if not patches:
        raise PatchError("no patches to report on")

    pairs = [
        iou(a, b)
        for index, a in enumerate(patches)
        for b in patches[index + 1 :]
    ]
    overlapping = [v for v in pairs if v > 0.0]

    grid = np.zeros((resolution, resolution), dtype=np.int32)
    for patch in patches:
        x0 = int(np.floor(patch.x * resolution))
        y0 = int(np.floor(patch.y * resolution))
        x1 = int(np.ceil((patch.x + patch.w) * resolution))
        y1 = int(np.ceil((patch.y + patch.h) * resolution))
        grid[max(0, y0) : min(resolution, y1), max(0, x0) : min(resolution, x1)] += 1

    covered = int((grid > 0).sum())
    stacked = int(grid.sum())

    return {
        "n_pairs": len(pairs),
        "mean_pairwise_iou": round(float(np.mean(pairs)), 5),
        "max_pairwise_iou": round(float(np.max(pairs)), 5),
        "n_overlapping_pairs": len(overlapping),
        "mean_iou_where_overlapping": (
            round(float(np.mean(overlapping)), 5) if overlapping else 0.0
        ),
        # Mean stacking depth over the covered area. 1.0 = a partition.
        "redundancy": round(stacked / covered, 3) if covered else 0.0,
        "frame_covered": round(covered / (resolution * resolution), 4),
    }


def clamped_regions(patches: list[Patch]) -> list[str]:
    """Names of regions pulled back inside the frame, deduplicated and sorted."""
    return sorted({p.band for p in patches if p.clamped})


def region_coverage(
    patches: list[Patch], mask: Trapezium, samples: int = COVERAGE_SAMPLES
) -> dict[str, float]:
    """Worst coverage per region NAME, across that region's instances.

    Keyed by name rather than by patch id so a bilateral pair reports as one
    structure. Left and right should agree exactly; if they ever do not, the
    mask or the placement is lopsided.
    """
    worst: dict[str, float] = {}
    for patch in patches:
        value = coverage_of(patch, mask, samples)
        worst[patch.band] = min(worst.get(patch.band, 1.0), value)
    return {name: round(value, 4) for name, value in sorted(worst.items())}


def assert_mask_coverage(
    patches: list[Patch], min_coverage: float = 0.5, mask: Trapezium | None = None
) -> None:
    """Refuse a patch set with a node that is mostly background.

    **A sibling to** ``assert_no_clamping``, **and a different property.**
    Clamping is FRAME containment: did the box run off the image. This is MASK
    containment: how much of the box is inside the trapezium. A region can pass
    the first and fail the second badly -- ``lateral_orbit`` at
    ``anatomy_scale_x = 1.60`` sits entirely within the frame while 87% of it
    lies outside the G1 mask, so nothing caught it and it was found by a test.

    A node fed almost entirely background still produces a feature vector, still
    trains, and still contributes to a number. Nothing downstream would notice.

    ``min_coverage`` defaults to 0.5 -- more than half the node is real image --
    which is a floor, not a recommendation. Phase 3 should pick deliberately.
    """
    if mask is not None:
        offenders = {
            name: value
            for name, value in region_coverage(patches, mask).items()
            if value < min_coverage
        }
    else:
        worst: dict[str, float] = {}
        for patch in patches:
            worst[patch.band] = min(worst.get(patch.band, 1.0), patch.coverage)
        offenders = {n: round(v, 4) for n, v in worst.items() if v < min_coverage}

    if offenders:
        listed = ", ".join(f"{n} {v:.2f}" for n, v in sorted(offenders.items()))
        raise PatchError(
            f"{len(offenders)} region(s) fall below {min_coverage:.0%} coverage of "
            f"the mask: {listed}. Those nodes are mostly background, and a node "
            "fed background still trains and still contributes to a number. "
            "Reduce anatomy_scale_x / anatomy_scale, or run this scheme at the "
            "geometry it was parameterised for."
        )


def assert_no_clamping(patches: list[Patch]) -> None:
    """Refuse a patch set whose geometry was silently altered to fit.

    The sweep clamps and reports, because aborting would hide the other variants.
    A real build should call this: a clamped region is not the region that was
    specified, and training on one while believing the other is the kind of
    mismatch that produces a plausible number nobody can explain later.
    """
    offenders = clamped_regions(patches)
    if offenders:
        raise PatchError(
            f"{len(offenders)} region(s) had to be clamped into the frame: "
            f"{offenders}. Reduce anatomy_scale_x or anatomy_box_scale_x until "
            "they fit, rather than training on geometry that is not what the "
            "config describes."
        )


def summarise(patches: list[Patch]) -> dict:
    """Aggregate description of a patch set. SHAREABLE — no patient data here."""
    by_band: dict[str, int] = {}
    band_coverage: dict[str, list[float]] = {}
    for patch in patches:
        by_band[patch.band] = by_band.get(patch.band, 0) + 1
        band_coverage.setdefault(patch.band, []).append(patch.coverage)

    coverages = [p.coverage for p in patches]
    return {
        "n_patches": len(patches),
        "per_band": dict(sorted(by_band.items())),
        "coverage_min": min(coverages),
        "coverage_mean": float(np.mean(coverages)),
        "coverage_max": max(coverages),
        "n_below_full_coverage": sum(1 for c in coverages if c < 1.0 - 1e-9),
        "white_fraction_max": 1.0 - min(coverages),
        "clamped_regions": clamped_regions(patches),
        "n_clamped": sum(1 for p in patches if p.clamped),
        # Per band, because under G1 the loss is NOT confined to the top: the
        # trapezium reaches full width only at y=1.0, so the bottom band's outer
        # column is still ~23% white. Exit criterion 8 wants this distribution.
        "white_fraction_per_band": {
            band: {
                "max": 1.0 - min(values),
                "mean": float(np.mean([1.0 - v for v in values])),
                "n_affected": sum(1 for v in values if v < 1.0 - 1e-9),
            }
            for band, values in sorted(band_coverage.items())
        },
    }
