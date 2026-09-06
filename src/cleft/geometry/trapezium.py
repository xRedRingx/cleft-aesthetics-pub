"""The cleft trapezium, and the unwarp that G2 depends on.

Measured geometry (brief §2), normalised within the crop content:

    top_y = 0.000   bot_y = 1.000   top_hw = 0.301   bot_hw = 0.500

so the mask is a symmetric trapezium, narrow at the brow and full-width at the
lip, covering 80.2% of the content.

**G1** leaves the pixels alone and exposes ``mask()`` so consumers can exclude
the white corners from the computation.

**G2** resamples to a canonical trapezium and then stretches each row to full
width, so the corners cease to exist. ``unwarp`` is that operation.

----------------------------------------------------------------------------
G2 IS GATED. Do not use it until ``asymmetry_is_preserved`` passes.
----------------------------------------------------------------------------

Each row scales by a different factor, so a horizontal distance at the brow is
magnified more than the same distance at the lip. The stretch is left-right
symmetric, so asymmetry *should* survive -- but "should" is exactly the word that
has been wrong every previous time in this project. If asymmetry is not
preserved, G2 is dead and G1 is the only geometry, because the entire signal
being modelled is left-right difference.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: Measured trapezium, normalised to the crop content.
TOP_Y, BOT_Y = 0.0, 1.0
TOP_HALF_WIDTH = 0.301
BOT_HALF_WIDTH = 0.500

#: Fraction of the content the trapezium covers. Recorded so a change is visible.
EXPECTED_AREA_FRACTION = 0.802

#: The midline. Asymmetry is measured against this and nothing else.
MIDLINE = 0.5


class TrapeziumError(ValueError):
    """The geometry is not usable."""


@dataclass(frozen=True)
class Trapezium:
    top_half_width: float = TOP_HALF_WIDTH
    bot_half_width: float = BOT_HALF_WIDTH
    midline: float = MIDLINE

    #: Lopsidedness: the left edge sits ``(1 - skew)`` of a half-width from the
    #: midline and the right edge ``(1 + skew)``. Zero for the measured mask.
    #:
    #: It exists so the gate can be shown to have teeth. A trapezium whose
    #: MIDLINE is displaced is still symmetric about that midline, so it
    #: preserves asymmetry perfectly -- perturbing the midline does not perturb
    #: the property under test. Skew is the perturbation that actually breaks it:
    #: with skew != 0 the stretch is no longer left-right symmetric, an offset of
    #: +d and one of -d are magnified differently, and asymmetry is destroyed.
    #:
    #: If a future measurement finds the real mask lopsided, this is also how the
    #: gate would report it rather than passing silently.
    skew: float = 0.0

    def half_width_at(self, y: float) -> float:
        """Half-width at normalised height ``y``, linear between top and bottom."""
        if not 0.0 <= y <= 1.0:
            raise TrapeziumError(f"y={y} is outside [0, 1]")
        return self.top_half_width + y * (self.bot_half_width - self.top_half_width)

    def left_edge_at(self, y: float) -> float:
        return self.midline - self.half_width_at(y) * (1.0 - self.skew)

    def right_edge_at(self, y: float) -> float:
        return self.midline + self.half_width_at(y) * (1.0 + self.skew)

    # -- vectorised forms ---------------------------------------------------
    # Same arithmetic over an array of heights. These exist because coverage was
    # computed with a Python loop over rows, one numpy call per row, per patch,
    # on every generate() -- which turned out to be the suite's largest single
    # cost. ``test_scalar_and_array_edges_agree`` pins them to the scalar forms so
    # the duplication cannot drift.

    def half_width_array(self, ys: np.ndarray) -> np.ndarray:
        clipped = np.clip(np.asarray(ys, dtype=float), 0.0, 1.0)
        return self.top_half_width + clipped * (
            self.bot_half_width - self.top_half_width
        )

    def left_edge_array(self, ys: np.ndarray) -> np.ndarray:
        return self.midline - self.half_width_array(ys) * (1.0 - self.skew)

    def right_edge_array(self, ys: np.ndarray) -> np.ndarray:
        return self.midline + self.half_width_array(ys) * (1.0 + self.skew)

    def contains(self, x: float, y: float) -> bool:
        return self.left_edge_at(y) <= x <= self.right_edge_at(y)

    def area_fraction(self, samples: int = 2048) -> float:
        """Mean half-width x 2 -- the trapezium's share of the unit square."""
        ys = (np.arange(samples) + 0.5) / samples
        widths = np.array([2.0 * self.half_width_at(float(y)) for y in ys])
        return float(np.clip(widths, 0.0, 1.0).mean())


DEFAULT = Trapezium()


def mask(size: int, trapezium: Trapezium = DEFAULT) -> np.ndarray:
    """Boolean mask of trapezium-interior pixels, for G1.

    G1's whole approach: the white corners stay in the pixels and are excluded
    from the *computation*. Removing them at pixel level is impossible -- alpha
    is dropped at load, so any removal is just choosing a different fill.
    """
    ys = (np.arange(size) + 0.5) / size
    xs = (np.arange(size) + 0.5) / size
    half = np.array([trapezium.half_width_at(float(y)) for y in ys])
    return np.abs(xs[None, :] - trapezium.midline) <= half[:, None]


def unwarp(image: np.ndarray, trapezium: Trapezium = DEFAULT) -> np.ndarray:
    """G2: stretch each row from the trapezium out to full width.

    The image is assumed already resampled to the canonical trapezium, which is
    what makes this identical for every patient. Unwarping each patient against
    their own crop would give each a different amount of distortion -- a
    patient-specific confound, which is precisely what staging exists to avoid.
    """
    array = np.asarray(image)
    if array.ndim not in (2, 3):
        raise TrapeziumError(f"expected a 2-D or 3-D image, got {array.shape}")

    height, width = array.shape[:2]
    out = np.empty_like(array)

    for row in range(height):
        y = (row + 0.5) / height

        # Sample only pixels whose CENTRE lies inside the mask. Computing the
        # span in continuous coordinates and flooring let the outermost sample
        # land one pixel outside on the narrow rows, which put white corner
        # pixels back into a supposedly corner-free G2 image -- 20 of them at
        # size 128. Deriving the span from the same pixel-centre rule the mask
        # uses makes that impossible rather than unlikely.
        first = int(np.ceil(trapezium.left_edge_at(y) * width - 0.5))
        last = int(np.floor(trapezium.right_edge_at(y) * width - 0.5))
        first = max(first, 0)
        last = min(last, width - 1)
        if last < first:
            raise TrapeziumError(
                f"row {row} (y={y:.4f}) has no pixel inside the mask; the "
                "trapezium is degenerate at this height"
            )

        # Affine in x and, when skew is zero, symmetric about the midline. That
        # symmetry is the reason asymmetry survives -- see the gate below.
        if last > first:
            columns = first + np.round(
                np.arange(width) * (last - first) / (width - 1)
            ).astype(int)
        else:
            columns = np.full(width, first, dtype=int)
        out[row] = array[row, columns]

    return out


def unwarp_x(x: float, y: float, trapezium: Trapezium = DEFAULT) -> float:
    """Where a normalised x at height y lands after unwarping."""
    left = trapezium.left_edge_at(y)
    right = trapezium.right_edge_at(y)
    if right <= left:
        raise TrapeziumError(f"degenerate row at y={y}: [{left}, {right}]")
    return (x - left) / (right - left)


# --------------------------------------------------------------------------
# THE GATE
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class AsymmetryCheck:
    height: float
    offset_in: float
    offset_out: float
    scale_factor: float
    recovered: float
    centre_out: float = 0.5

    @property
    def error(self) -> float:
        return abs(self.recovered - self.offset_in)

    @property
    def centre_error(self) -> float:
        """How far the mirrored pair's midpoint drifted from the output centre.

        Non-zero means the stretch was not symmetric: the pair no longer
        straddles the midline, so "left minus right" no longer means what it did.
        """
        return abs(self.centre_out - 0.5)


def asymmetry_at(
    y: float, offset: float, trapezium: Trapezium = DEFAULT
) -> AsymmetryCheck:
    """Displace a feature ``offset`` from the midline at height ``y``, unwarp, recover.

    The row's scale factor is ``1 / (2 * half_width)``: a row spanning 0.602 of
    the width is stretched to 1.0. If asymmetry survives, dividing the output
    offset by that factor returns the input offset exactly.
    """
    half = trapezium.half_width_at(y)
    if offset > half:
        raise TrapeziumError(
            f"offset {offset} lies outside the trapezium at y={y} (half-width {half})"
        )

    # Measured as a MIRRORED PAIR about the midline, not as a single displaced
    # point. A single point cannot distinguish "asymmetry preserved" from
    # "everything shifted", and a skewed stretch magnifies +d and -d differently
    # -- which only shows up when both are measured.
    left = unwarp_x(trapezium.midline - offset, y, trapezium)
    right = unwarp_x(trapezium.midline + offset, y, trapezium)
    offset_out = (right - left) / 2.0

    span = trapezium.right_edge_at(y) - trapezium.left_edge_at(y)
    scale = 1.0 / span
    return AsymmetryCheck(
        height=y,
        offset_in=offset,
        offset_out=offset_out,
        scale_factor=scale,
        recovered=offset_out / scale,
        centre_out=(left + right) / 2.0,
    )


def asymmetry_is_preserved(
    trapezium: Trapezium = DEFAULT,
    heights: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0),
    offsets: tuple[float, ...] = (0.02, 0.05, 0.10, 0.15),
    tolerance: float = 1e-9,
) -> tuple[bool, list[AsymmetryCheck]]:
    """The gate. Returns (passed, every check performed).

    Several heights, because the scale factor varies with row and a check at one
    height would prove nothing about the others.
    """
    checks = []
    for y in heights:
        half = trapezium.half_width_at(y)
        for offset in offsets:
            if offset > half:
                continue
            checks.append(asymmetry_at(y, offset, trapezium))
    if not checks:
        raise TrapeziumError("no offsets fit inside the trapezium; nothing was checked")
    passed = all(
        c.error <= tolerance and c.centre_error <= tolerance for c in checks
    )
    return passed, checks
