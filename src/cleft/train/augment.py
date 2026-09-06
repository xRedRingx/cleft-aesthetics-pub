"""Phase 7C: the augmenting path. Pixels in, augmented pixels out.

Pure numpy, no torch. That is not incidental -- see ``DETERMINISM`` below.

----------------------------------------------------------------------------
WHERE THIS SITS
----------------------------------------------------------------------------
``phase7c.LIVE_PATH``: ``harness.run_fold`` calls ``train_epoch`` once per
epoch and ``predict`` for inner-val and test. An augmenting backbone applies
this module inside ``train_epoch`` and does not call it in ``predict``, so
**validation and test images cannot be augmented** -- a property of the call
graph rather than a rule to observe. The training loop itself is untouched,
so gate 1 still tests the harness it was measured on.

----------------------------------------------------------------------------
THE ORDER, AND WHY REGION-AWARENESS APPLIES TO PHOTOMETRIC ONLY
----------------------------------------------------------------------------
PLAN §4.7 [REASONED] gives the order: background exclusion -> photometric ->
geometric with boxes transformed in lockstep. That is what ``augment_batch``
does, and it has a consequence the brief did not resolve.

**[DECIDED 2026-08-03] A global affine cannot be spatially modulated.** The
declared geometric transforms are a single rotation, translation and scale of
the whole frame; there is no coherent way to apply them "more strongly outside
the nose" without tearing the image. Region-awareness therefore modulates
**photometric strength only**, and geometric transforms are global.

That does not weaken the arm 5 / arm 6 comparison -- both carry the same
global geometry, and they still differ in exactly one factor: whether
photometric strength is spatially modulated. It does mean "region-aware,
photometric + geometric" is precisely *region-aware photometric plus global
geometric*, and the record says so rather than leaving a reader to assume the
rotation was somehow masked.

The mask is built and consumed in the ORIGINAL frame, before any warp, which
is why the order matters: after a rotation the boxes no longer describe the
pixels they were mapped to.

----------------------------------------------------------------------------
DETERMINISM
----------------------------------------------------------------------------
Every draw comes from ``rng_for(seed, fold, epoch)`` -- a fresh generator from
a spawn key, not a shared stream -- so a resumed run reproduces the same views
without any RNG state needing to travel in a checkpoint. Gate 1's guarantee
survives by construction rather than by saving and restoring.

**And the warp is numpy, not torch.** ``phase7c.ROTATION_IS_AN_ARM`` records
that a rotation interpolation kernel is something gate 1 has never covered and
that a raise under ``use_deterministic_algorithms(True)`` would be a finding
rather than a reason to disable the flag. Doing the interpolation here, in
float64 numpy with a fixed evaluation order, means **no torch kernel is
involved in it at all** -- the tensor that reaches the backbone is already
augmented. The flag's exposure is unchanged from the arms that ran before.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: What fills pixels a warp pulls in from outside the source frame.
#:
#: **White, because the crops are white-padded** (PLAN §4.4: staging is
#: pad-square-then-resize, white pad). A rotation of a G1 crop brings the
#: padded corners inward; filling with white keeps that continuous with what
#: the pad already is. Black would introduce a frame no real image has, and a
#: model can key on a border that appears only in training.
FILL_VALUE = 255.0

#: ``hue`` is declared as a fraction of the full hue circle, applied as a
#: luma-preserving rotation in YIQ. Written down because "hue 0.05" alone does
#: not say of what -- degrees, radians or turns -- and the three differ by two
#: orders of magnitude.
HUE_UNITS = "fraction of a full turn; 0.05 -> +/-18 degrees"

#: RGB <-> YIQ, for the hue rotation. Luma (Y) is untouched by the rotation,
#: so hue jitter does not silently double as brightness jitter -- which would
#: confound the photometric family with itself.
_RGB_TO_YIQ = np.array([
    [0.299, 0.587, 0.114],
    [0.596, -0.274, -0.322],
    [0.211, -0.523, 0.312],
])
_YIQ_TO_RGB = np.linalg.inv(_RGB_TO_YIQ)


class AugmentError(RuntimeError):
    """The augmentation policy cannot be applied as declared."""


@dataclass(frozen=True)
class Policy:
    """One arm's augmentation, as flags plus the pre-registered strengths."""

    photometric: bool = False
    geometric: bool = False
    rotation: bool = False
    region_aware: bool = False

    @classmethod
    def from_arm(cls, arm: dict) -> "Policy":
        return cls(
            photometric=bool(arm["photometric"]),
            geometric=bool(arm["geometric"]),
            rotation=bool(arm["rotation"]),
            region_aware=bool(arm["region_aware"]),
        )

    def __post_init__(self) -> None:
        if self.rotation and not self.geometric:
            raise AugmentError(
                "rotation without geometric: rotation is a geometric "
                "transform and cannot be applied on its own"
            )

    @property
    def is_identity(self) -> bool:
        return not (self.photometric or self.geometric)


def rng_for(seed: int, fold: int, epoch: int) -> np.random.Generator:
    """A generator determined by (seed, fold, epoch) and nothing else.

    **Not a shared stream.** A single generator advanced across folds and
    epochs would make every view depend on how many draws happened before it,
    so a resumed run -- which restarts the epoch loop -- would produce
    different pixels. PLAN §2.7's measured hazard is exactly that: run
    identity survives a pause and training does not.

    Deriving the generator from the coordinates instead means a view depends
    only on where it is in the schedule, so a resume reproduces it with no RNG
    state to save.
    """
    return np.random.default_rng([int(seed), int(fold), int(epoch)])


# --------------------------------------------------------------------------
# the protection mask
# --------------------------------------------------------------------------


def protection_mask(
    height: int,
    width: int,
    boxes,
    *,
    inside_strength: float,
    sigma_fraction: float,
) -> np.ndarray:
    """Augmentation strength per pixel: 1.0 outside, ``inside_strength``
    inside a protected box, blended by a Gaussian falloff.

    **Soft, not hard** (``phase7c.BLEND``): a hard seam sits at a fixed
    anatomical offset, on the regions the grade is about, and appears in every
    augmented training image and no evaluation image -- a feature a model can
    key on.

    ``sigma_fraction`` is a fraction of the image width rather than a pixel
    count, because crop aspect ratios span a factor of two (PLAN §4.4) and a
    fixed width would be a different blend on each patient.

    Boxes are ``(x, y, w, h)`` in pixels -- ``mapping.MappedPatch.pixels``.
    """
    if not 0.0 <= inside_strength <= 1.0:
        raise AugmentError(f"inside_strength {inside_strength} outside [0, 1]")

    mask = np.ones((height, width), dtype=np.float64)
    for x, y, w, h in boxes:
        x0, y0 = max(int(x), 0), max(int(y), 0)
        x1, y1 = min(int(x) + int(w), width), min(int(y) + int(h), height)
        if x1 > x0 and y1 > y0:
            mask[y0:y1, x0:x1] = inside_strength

    sigma = float(sigma_fraction) * float(width)
    if sigma > 0:
        mask = _gaussian_blur(mask, sigma)
    return mask


def _gaussian_blur(image: np.ndarray, sigma: float) -> np.ndarray:
    """Separable Gaussian, reflect-padded.

    Reflect rather than zero padding: zeros at the edge would pull the mask
    toward "protect" along the frame border, which is the opposite of what the
    policy says about the periphery. Phase 1's row-profile defect was a zero
    pad doing exactly this kind of thing at the ends of a signal.
    """
    radius = max(int(round(3.0 * sigma)), 1)
    offsets = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-(offsets ** 2) / (2.0 * sigma * sigma))
    kernel /= kernel.sum()

    padded = np.pad(image, radius, mode="reflect")
    rows = np.apply_along_axis(
        lambda line: np.convolve(line, kernel, mode="valid"), 1, padded
    )
    return np.apply_along_axis(
        lambda line: np.convolve(line, kernel, mode="valid"), 0, rows
    )


def protection_masks_for(
    images: np.ndarray,
    geometry_rows: list,
    protected: tuple,
    *,
    geometry: str,
    inside_strength: float,
    sigma_fraction: float,
) -> np.ndarray:
    """One strength mask per patient, from the frozen anatomy regions.

    **The boxes are per patient**: ``geometry/mapping.py`` places each region
    in that image's own pixels using its content box, and crop aspect ratios
    span a factor of two. A single shared mask would sit on different anatomy
    in every image.

    Built here, once, before training -- and then carried to the backbone as a
    fourth channel on the image array (``phase3.prepare_features``), because
    the harness slices ``features[train_rows]`` and a mask held separately
    could not be sliced with it. A mask that travelled beside the rows rather
    than with them would need the row indices recovered inside the backbone,
    which means recomputing ``inner_val_split`` outside the frozen harness --
    a second implementation of the one rule that must not have two.
    """
    from ..geometry import patch_features
    from ..geometry.mapping import map_patches

    unknown = [name for name in protected if not isinstance(name, str)]
    if unknown:
        raise AugmentError(f"protected region names must be strings: {unknown}")

    patches = [
        patch for patch in patch_features.patches_for("anatomy", geometry)
        if patch.band in set(protected)
    ]
    if not patches:
        raise AugmentError(
            f"none of {sorted(set(protected))} matched an anatomy region at "
            f"geometry {geometry!r}; the mask would protect nothing"
        )

    stack = np.asarray(images)
    if len(stack) != len(geometry_rows):
        raise AugmentError(
            f"{len(stack)} images against {len(geometry_rows)} geometry rows"
        )

    size = stack.shape[1]
    masks = np.empty((len(stack), size, size), dtype=np.float64)
    for index, row in enumerate(geometry_rows):
        staged = patch_features.staged_from_row(stack[index], row)
        boxes = [item.pixels for item in map_patches(staged, patches)]
        masks[index] = protection_mask(
            size, size, boxes,
            inside_strength=inside_strength, sigma_fraction=sigma_fraction,
        )
    return masks


# --------------------------------------------------------------------------
# photometric
# --------------------------------------------------------------------------


def apply_photometric(
    image: np.ndarray,
    rng: np.random.Generator,
    settings: dict,
    strength: np.ndarray | None = None,
) -> np.ndarray:
    """Brightness, contrast, saturation and hue, at the declared magnitudes.

    ``strength`` is the per-pixel map from ``protection_mask``. It is applied
    by blending the fully-augmented image back toward the original --
    ``out = image + strength * (augmented - image)`` -- which is what makes
    the soft falloff a falloff rather than a second hard threshold: a pixel at
    strength 0.6 receives 60% of the same jitter its neighbours receive, not a
    different jitter.
    """
    original = np.asarray(image, dtype=np.float64)
    out = original.copy()

    brightness = float(settings["brightness"])
    contrast = float(settings["contrast"])
    saturation = float(settings["saturation"])
    hue = float(settings["hue"])

    if brightness:
        out *= 1.0 + rng.uniform(-brightness, brightness)
    if contrast:
        factor = 1.0 + rng.uniform(-contrast, contrast)
        grey = out.mean()
        out = grey + factor * (out - grey)
    if saturation:
        factor = 1.0 + rng.uniform(-saturation, saturation)
        luma = out @ _RGB_TO_YIQ[0]
        out = luma[..., None] + factor * (out - luma[..., None])
    if hue:
        turns = rng.uniform(-hue, hue)
        out = _rotate_hue(out, turns)

    out = np.clip(out, 0.0, 255.0)
    if strength is not None:
        out = original + strength[..., None] * (out - original)
    return out


def _rotate_hue(image: np.ndarray, turns: float) -> np.ndarray:
    """Rotate hue by ``turns`` of the full circle, preserving luma."""
    angle = 2.0 * np.pi * float(turns)
    cos, sin = np.cos(angle), np.sin(angle)
    rotation = np.array([
        [1.0, 0.0, 0.0],
        [0.0, cos, -sin],
        [0.0, sin, cos],
    ])
    matrix = _YIQ_TO_RGB @ rotation @ _RGB_TO_YIQ
    return image @ matrix.T


# --------------------------------------------------------------------------
# geometric -- image, mask and boxes in lockstep
# --------------------------------------------------------------------------


def sample_affine(
    rng: np.random.Generator, policy: Policy, settings: dict, size: int
) -> dict:
    """Draw one rotation / translation / scale, as declared parameters.

    Returned as the parameters rather than as a matrix so a run record can say
    what was applied to an image, not just that something was.
    """
    degrees = (
        rng.uniform(-settings["rotation_degrees"], settings["rotation_degrees"])
        if policy.rotation
        else 0.0
    )
    fraction = float(settings["translate_fraction"])
    low, high = settings["scale"]
    return {
        "degrees": float(degrees),
        "translate_x": float(rng.uniform(-fraction, fraction) * size),
        "translate_y": float(rng.uniform(-fraction, fraction) * size),
        "scale": float(rng.uniform(low, high)),
    }


def affine_matrices(params: dict, size: int) -> tuple[np.ndarray, np.ndarray]:
    """``(forward, inverse)`` 3x3 matrices about the image centre.

    ``forward`` maps a source coordinate to where it lands -- what boxes need.
    ``inverse`` maps an output pixel back to where to sample -- what the warp
    needs. Both are returned from one construction so they cannot disagree,
    which is the failure that puts the mask somewhere the image is not.
    """
    centre = (size - 1) / 2.0
    angle = np.deg2rad(params["degrees"])
    cos, sin = np.cos(angle), np.sin(angle)
    scale = params["scale"]

    linear = scale * np.array([[cos, -sin], [sin, cos]])
    offset = (
        np.array([centre, centre])
        - linear @ np.array([centre, centre])
        + np.array([params["translate_x"], params["translate_y"]])
    )

    forward = np.eye(3)
    forward[:2, :2] = linear
    forward[:2, 2] = offset
    return forward, np.linalg.inv(forward)


def warp_image(
    image: np.ndarray, inverse: np.ndarray, *, fill: float = FILL_VALUE
) -> np.ndarray:
    """Bilinear inverse warp. Deterministic: float64, fixed evaluation order."""
    array = np.asarray(image, dtype=np.float64)
    height, width = array.shape[:2]

    ys, xs = np.meshgrid(
        np.arange(height, dtype=np.float64),
        np.arange(width, dtype=np.float64),
        indexing="ij",
    )
    flat = np.stack([xs.ravel(), ys.ravel(), np.ones(xs.size)])
    source = inverse @ flat
    sx = source[0].reshape(height, width)
    sy = source[1].reshape(height, width)

    x0 = np.floor(sx)
    y0 = np.floor(sy)
    wx = sx - x0
    wy = sy - y0
    x0 = x0.astype(np.int64)
    y0 = y0.astype(np.int64)

    def gather(xi, yi):
        inside = (xi >= 0) & (xi < width) & (yi >= 0) & (yi < height)
        clipped_x = np.clip(xi, 0, width - 1)
        clipped_y = np.clip(yi, 0, height - 1)
        values = array[clipped_y, clipped_x]
        return np.where(inside[..., None], values, fill)

    top = gather(x0, y0) * (1 - wx)[..., None] + gather(x0 + 1, y0) * wx[..., None]
    bottom = (
        gather(x0, y0 + 1) * (1 - wx)[..., None]
        + gather(x0 + 1, y0 + 1) * wx[..., None]
    )
    return top * (1 - wy)[..., None] + bottom * wy[..., None]


def warp_boxes(boxes, forward: np.ndarray):
    """Move ``(x, y, w, h)`` boxes with the image.

    A rotated rectangle is not a rectangle, so each box becomes the
    **axis-aligned envelope** of its four transformed corners. That is a
    slight enlargement at non-zero rotation, recorded rather than hidden: the
    boxes are a spatial prior for what to protect, and an envelope protects a
    little more than the region rather than a little less, which is the safe
    direction for this use.
    """
    out = []
    for x, y, w, h in boxes:
        corners = np.array([
            [x, y, 1.0], [x + w, y, 1.0], [x, y + h, 1.0], [x + w, y + h, 1.0],
        ]).T
        moved = (forward @ corners)[:2]
        x0, y0 = moved[0].min(), moved[1].min()
        x1, y1 = moved[0].max(), moved[1].max()
        out.append((float(x0), float(y0), float(x1 - x0), float(y1 - y0)))
    return out


# --------------------------------------------------------------------------
# the whole policy, applied
# --------------------------------------------------------------------------


def augment_image(
    image: np.ndarray,
    *,
    policy: Policy,
    rng: np.random.Generator,
    photometric_settings: dict,
    geometric_settings: dict,
    strength: np.ndarray | None = None,
) -> dict:
    """One image through one policy. Returns the pixels and what was applied.

    ``strength`` is the per-pixel map from ``protection_mask``, precomputed by
    the caller -- ``protection_masks_for`` builds one per patient before
    training and they travel to the backbone as a fourth image channel.

    Order is PLAN §4.7's: photometric in the ORIGINAL frame, then the
    geometric warp. **That ordering is what satisfies the lockstep
    requirement**, and it satisfies it more simply than warping the mask
    would: the mask is consumed before the warp exists, so it cannot stand
    still while the image rotates. ``warp_boxes`` is here and tested for any
    consumer that needs post-warp geometry; this path does not, because a mask
    built after a warp would be protecting the wrong pixels anyway.
    """
    array = np.asarray(image, dtype=np.float64)
    if array.ndim != 3 or array.shape[2] != 3:
        raise AugmentError(
            f"expected (H, W, 3) pixels; got shape {array.shape}"
        )
    if array.shape[0] != array.shape[1]:
        raise AugmentError(
            f"expected a square staged crop; got {array.shape[:2]}. The "
            "affine is built about a single centre and a rectangular frame "
            "would rotate about the wrong point"
        )
    if strength is not None and strength.shape != array.shape[:2]:
        raise AugmentError(
            f"strength map {strength.shape} does not match the image "
            f"{array.shape[:2]}"
        )

    size = array.shape[0]
    applied: dict = {"policy": policy, "photometric": None, "affine": None}
    out = array

    if policy.photometric:
        used = strength if policy.region_aware else None
        out = apply_photometric(out, rng, photometric_settings, used)
        applied["photometric"] = {
            "region_aware": used is not None,
            "min_strength": None if used is None else float(used.min()),
        }

    if policy.geometric:
        params = sample_affine(rng, policy, geometric_settings, size)
        _, inverse = affine_matrices(params, size)
        out = warp_image(out, inverse)
        applied["affine"] = params

    return {"image": np.clip(out, 0.0, 255.0), "applied": applied}


def augment_batch(
    images: np.ndarray,
    *,
    policy: Policy,
    seed: int,
    fold: int,
    epoch: int,
    photometric_settings: dict,
    geometric_settings: dict,
    strength_per_image=None,
) -> np.ndarray:
    """A fold's training images for one epoch.

    **Identity is exactly identity.** With no family enabled this returns the
    input unchanged rather than round-tripping it through float64 and back --
    arm 0 is a gate on reproducing 0.2520, and a policy that quietly requantised
    every pixel would be testing something slightly different from the arm it
    is meant to reproduce.
    """
    if policy.is_identity:
        return np.asarray(images)

    rng = rng_for(seed, fold, epoch)
    stack = np.asarray(images)
    if strength_per_image is not None and len(strength_per_image) != len(stack):
        raise AugmentError(
            f"{len(stack)} images against {len(strength_per_image)} strength "
            "maps; the masks are per patient and must be sliced with the rows"
        )

    out = np.empty(stack.shape, dtype=np.float64)
    for index in range(len(stack)):
        out[index] = augment_image(
            stack[index],
            policy=policy,
            rng=rng,
            photometric_settings=photometric_settings,
            geometric_settings=geometric_settings,
            strength=(
                None if strength_per_image is None else strength_per_image[index]
            ),
        )["image"]

    if stack.dtype == np.float64:
        return out
    if np.issubdtype(stack.dtype, np.integer):
        # **Round, do not truncate.** `astype` truncates toward zero, which on
        # non-negative pixels is a systematic downward bias of about half a
        # level on every augmented pixel -- a brightness shift applied to the
        # training set and not the evaluation set, which is exactly the kind
        # of difference that would read as an augmentation effect.
        return np.rint(out).astype(stack.dtype)
    return out.astype(stack.dtype)
