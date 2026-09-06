"""Branch 3: anatomy region crops, taken per patient and magnified.

Brief ``ROADB_REGION_CROP_BRIEF.md`` (outside the repository) §2. The backbone never sees a whole
face -- 22 protected anatomy regions are cropped from the 768 non-square G2
staging and each is resized to the backbone's 224 input, so a structure that
occupied ~96 px of the frame gets the full frame to itself.

----------------------------------------------------------------------------
WHY THE FROZEN CONTAINMENT CHECK CANNOT BE USED HERE [MEASURED 2026-08-14]
----------------------------------------------------------------------------
``mapping.map_patches`` calls ``assert_inside_content``, which takes
``size = staged.image.shape[0]`` and bounds **both** axes by it. That is
correct for every image the frozen apparatus has ever seen, because Road A
staged square. This branch stages NON-SQUARE, and on a landscape image the
check rejects boxes that are genuinely inside::

    stage_non_square(2424x2173, 768) -> (688, 768) array
    mapping.map_patches(...) -> MappingError: patch 12 at (596, 34, 166, 111)
                                falls outside the 688x688 image

x + w = 762 <= 768, the real width. The box is fine; the check is square.

**The cohort contains exactly one landscape crop** (aspect 1.0986,
``scut.ar_distribution.COHORT``: 236 of 237 portrait). So calling the frozen
path would have processed 236 patients and died on the last -- the worst
shape a failure can take, and one that invites "just skip that patient",
which would silently make the cohort 236.

``mapping.py`` is frozen, so this module does what ``roadb_staging`` did for
the same reason: it reuses the frozen ARITHMETIC (``content_box_to_pixels``,
which is aspect-correct because it goes through the content box) and
supplies its own containment check, bounding x by the width and y by the
height. The mapping is not reimplemented -- only the assertion that was
written for squares.
"""

from __future__ import annotations

import numpy as np

from .geometry.patches import PatchConfig, coverage_of, get
from .geometry.staging import Staged, content_box_to_pixels, resize_nearest
from .geometry.trapezium import DEFAULT as DEFAULT_TRAPEZIUM, Trapezium


class RegionCropError(ValueError):
    """A region could not be cropped from this image."""


#: The backbone input every crop is resized to. NOT a resolution knob: Road
#: B measured what leaving 224 costs a frozen transformer, and the whole
#: point of this branch is that the magnification happens in the CROP.
CROP_OUTPUT_SIZE = 224

#: Placement seed for the random control. Fixed so the random arm is a
#: reproducible artifact rather than a different set of boxes per run --
#: the same discipline every other artifact on this road follows.
RANDOM_PLACEMENT_SEED = 20260814


def protected_patches(config: PatchConfig | None = None) -> list:
    """The 22 anatomy patches this branch crops, derived not listed.

    ``phase7c.PROTECTED_REGIONS`` names fifteen regions; the anatomy
    generator expands the seven bilateral ones into mirrored pairs, giving
    22 of its 27 boxes. The three it leaves out -- glabella, medial_canthus,
    lateral_orbit -- are exactly the cheeks-forehead-periphery the branch
    means to drop (brief §2.1).
    """
    from .phase7c import PROTECTED_REGIONS

    generated = get("anatomy").generate(
        DEFAULT_TRAPEZIUM, config or PatchConfig()
    )
    selected = [p for p in generated if p.band in PROTECTED_REGIONS]
    if len(selected) != 22:
        raise RegionCropError(
            f"expected 22 protected boxes, got {len(selected)} -- the "
            "anatomy generator or the protect list changed, and the "
            "branch's region count is not a free parameter"
        )
    return selected


def region_boxes(staged: Staged, patches: list) -> list[tuple]:
    """Each patch in THIS image's pixels, checked against its real shape.

    Uses the frozen ``content_box_to_pixels`` for the arithmetic -- that is
    the mapping, and it is aspect-correct -- then applies the containment
    rule the frozen assertion gets wrong on non-square images.
    """
    height, width = staged.image.shape[:2]
    x0, y0, content_w, content_h = staged.content_box
    boxes = []
    for patch in patches:
        px, py, pw, ph = content_box_to_pixels(staged, patch.box)
        if pw < 1 or ph < 1:
            raise RegionCropError(
                f"region {patch.band!r} collapsed to {pw}x{ph} px"
            )
        # **Bounded per axis** -- this is the whole reason the module exists.
        if px < 0 or py < 0 or px + pw > width or py + ph > height:
            raise RegionCropError(
                f"region {patch.band!r} at {(px, py, pw, ph)} falls outside "
                f"the {width}x{height} image"
            )
        # And never into the pad, which at G2 non-square does not exist
        # (content fraction 1.0000) but is checked rather than assumed.
        if (
            px + 1 < x0 or py + 1 < y0
            or px + pw > x0 + content_w + 1
            or py + ph > y0 + content_h + 1
        ):
            raise RegionCropError(
                f"region {patch.band!r} at {(px, py, pw, ph)} extends into "
                f"the padding (content box {staged.content_box})"
            )
        boxes.append((px, py, pw, ph))
    return boxes


def crop_region(
    staged: Staged, box: tuple[int, int, int, int], size: int = CROP_OUTPUT_SIZE
) -> np.ndarray:
    """One region, cut and resized to the backbone's input.

    ``resize_nearest`` is the frozen resampler every other staging step
    uses. Nearest-neighbour UPSAMPLES a ~96 px region to 224 blockily, and
    that is the honest behaviour: it magnifies without inventing detail. The
    contact sheet shows crops at NATIVE resolution precisely so the reviewer
    judges the information present, not the resampler.
    """
    px, py, pw, ph = box
    window = staged.image[py:py + ph, px:px + pw]
    if window.shape[0] != ph or window.shape[1] != pw:
        raise RegionCropError(
            f"crop {box} came back {window.shape[:2]}, expected {(ph, pw)}"
        )
    return resize_nearest(window, size, size)


def crops_for(
    staged: Staged, patches: list, size: int = CROP_OUTPUT_SIZE
) -> np.ndarray:
    """All 22 regions as one ``(n, size, size, C)`` batch.

    Batched so the backbone call count per patient stays at ONE, matching a
    whole-frame arm (brief §2.2) -- and extracted ONCE, not per epoch: this
    branch is frozen-backbone, head-only, like every other Road B arm.
    """
    boxes = region_boxes(staged, patches)
    return np.stack([crop_region(staged, box, size) for box in boxes])


def random_patches(
    template: list, seed: int = RANDOM_PLACEMENT_SEED,
    trapezium: Trapezium = DEFAULT_TRAPEZIUM,
    config: PatchConfig | None = None,
) -> list:
    """The random control: the SAME boxes, re-placed inside the mask.

    Brief §3 requires random crops matching the anatomy crops "in count,
    size distribution and total area, sampled inside the mask, so the only
    difference is placement". Re-placing the anatomy boxes themselves makes
    all three exact BY CONSTRUCTION rather than approximately: same count,
    same widths and heights, same total area, different location.

    This is the arm that decides what a positive finding is ABOUT -- if
    random placement matches anatomy, the result is about cropping, not
    anatomy (Charm 2025, and this project's own scheme axis null four
    times). Its placement is seeded so the control is a reproducible
    artifact.
    """
    from dataclasses import replace

    settings = config or PatchConfig()
    rng = np.random.default_rng(seed)
    placed = []
    for patch in template:
        for _ in range(2000):
            x = float(rng.uniform(0.0, 1.0 - patch.w))
            y = float(rng.uniform(0.0, 1.0 - patch.h))
            candidate = replace(patch, x=x, y=y, mirror_id=None)
            if coverage_of(candidate, trapezium) >= settings.min_coverage:
                placed.append(candidate)
                break
        else:
            raise RegionCropError(
                f"could not place a {patch.w:.3f}x{patch.h:.3f} box inside "
                f"the mask in 2000 attempts (region {patch.band!r})"
            )
    if len(placed) != len(template):
        raise RegionCropError("random placement lost a box")
    return placed


def whole_frame_patch(config: PatchConfig | None = None):
    """The control's single "region": the entire content box.

    **[DECIDED 2026-08-14] The whole-frame control goes through the CROP
    path, not through the extraction task.** ``phase3.load_inputs`` wants
    ``staged_patient_<geometry>.npy`` -- the STACKED tensor -- and a
    non-square artifact can never have one, because the shapes differ per
    patient (``roadb_staging.SHAPE_IS_FIXED_AT_STAGING``; stacking them
    was one of the five original stage-task defects). And even given a
    per-patient loader, non-square frames cannot batch and would give
    every patient a different ViT token count.

    Routing the control through here dissolves both: the whole content box
    is cut and resized to 224 exactly as each crop is, so crop-vs-whole
    feeds the backbone 224x224 either way and differs in **content
    alone** -- one whole face against 22 magnified regions -- rather than
    in content plus batching plus token count.

    It also means the control inherits the same square stretch the crops
    get (``CROPS_ARE_SQUARE_STRETCHED``), which is what makes it a valid
    comparator rather than a confound.
    """
    from dataclasses import replace

    template = get("anatomy").generate(
        DEFAULT_TRAPEZIUM, config or PatchConfig()
    )[0]
    return replace(
        template, id=0, band="whole_frame", x=0.0, y=0.0, w=1.0, h=1.0,
        mirror_id=None, clamped=False,
    )


def pool_feature_maps(values: np.ndarray) -> np.ndarray:
    """``(n, channels, h, w)`` -> ``(n, channels)`` by spatial mean.

    **[DECIDED 2026-08-14, option B] The graph backbone's crops are pooled
    to vectors, and this is NOT SR-GNN as published.**

    ``extract_features`` returns feature maps for graph backbones by a
    recorded decision (``embeddings.KIND_FOR_BACKBONE_KIND``): the frozen
    boundary for both graph models is the backbone's final map. So a crop
    set from SR-GNN arrives as ``(n_crops, 2048, 7, 7)``.

    Storing those whole -- 22 x 100,352 dims per patient -- would feed
    ``SeqSelfAttention`` exactly the descriptor it was built for, but it
    leaves the matched concat partner with a 2.2M-dimensional head at
    n=237, which is not an arm. Pooling keeps both SR-GNN arms on
    identical nodes, so the combination contrast varies combination alone
    -- the question brief §2.3 actually asks.

    **What is given up, recorded so nobody reads this arm as a CleftGNN
    replication**: pooling changes ``SeqSelfAttention`` into a smaller
    layer (2048-wide input rather than 100,352). The arm tests whether
    message passing over anatomy regions adds anything concatenation does
    not. It does not test SR-GNN's published architecture.

    Mean rather than max: it is the standard global average pool and it
    matches what "pooled" already means for the transformer sets, so the
    two backbones' region vectors are the same KIND of quantity.
    """
    array = np.asarray(values)
    if array.ndim != 4:
        raise RegionCropError(
            f"expected (n, channels, h, w) feature maps, got {array.shape}"
        )
    return array.mean(axis=(2, 3))


#: Sheet layout. The caption strip must fit "philtral_column_L 132x88".
SHEET_CAPTION_HEIGHT = 14
SHEET_GAP = 4
SHEET_BACKGROUND = 32


def crop_sheet(
    staged: Staged, patches: list, columns: int = 6, as_fed: bool = False,
    layout: str = "anatomy",
) -> np.ndarray:
    """One patient: the whole face with its boxes, beside every crop at
    NATIVE resolution.

    **Two questions, one panel, and the sheet is the gate for both.**

    * *Is this the right anatomy?* -- answered by the left panel, where
      each box is drawn and named on the face it was cut from. Whether a
      box labelled ``philtral_column`` contains a philtral column is
      checked by eye or not at all.
    * *Does 96 px of philtral column actually show anything?* -- answered
      only by NATIVE resolution. Resizing the crops to a uniform cell
      would show the reviewer the resampler, not the information, and the
      magnification premise the whole branch rests on would go unchecked.
      So each crop is pasted at its own pixel size and captioned with it.

    **``as_fed=True`` answers a THIRD question the native sheet cannot**:
    what the backbone actually receives. ``crop_region`` resizes every
    window to ``size x size``, so a 149x124 box is stretched about 1.2:1
    before it reaches the model. The native sheet deliberately hides that
    -- it was built to show information, not transformation -- so a review
    passing on it has confirmed anatomy and magnification and nothing
    about distortion. Captions name the source size and the stretch
    factor, because the number is the finding
    (``roadb.CROPS_ARE_SQUARE_STRETCHED``).
    """
    from PIL import Image, ImageDraw

    from .geometry.render import colour_for, draw_rect

    # **Captions use the LAYOUT's names.** The random sheets previously
    # captioned anatomy names onto randomly placed boxes -- a reviewer
    # reading "philt_R" on a box that is not the philtrum is being told
    # something false by the artifact meant to be checked.
    names = region_names(patches, layout)

    # Refuse an empty layout here, with the reason. A zero-panel sheet
    # renders as a blank rectangle that a reviewer would sign off, which
    # is the one failure a review gate must never have.
    if not patches:
        raise RegionCropError(
            "no regions to draw: the crop sheet was asked to render an "
            "empty layout, which would produce a blank sheet for a "
            "reviewer to pass"
        )
    boxes = region_boxes(staged, patches)
    crops = [
        crop_region(staged, box) if as_fed
        else staged.image[box[1]:box[1] + box[3], box[0]:box[0] + box[2]]
        for box in boxes
    ]

    face = np.asarray(staged.image).astype(np.uint8).copy()
    if face.ndim == 2:
        face = np.stack([face] * 3, axis=-1)
    for index, (patch, box) in enumerate(zip(patches, boxes)):
        draw_rect(face, box, colour_for(patch.band, index), thickness=2)

    cell_w = max(c.shape[1] for c in crops)
    cell_h = max(c.shape[0] for c in crops) + SHEET_CAPTION_HEIGHT
    rows = (len(crops) + columns - 1) // columns
    grid_w = columns * cell_w + (columns + 1) * SHEET_GAP
    grid_h = rows * cell_h + (rows + 1) * SHEET_GAP

    height = max(face.shape[0], grid_h)
    canvas = np.full(
        (height, face.shape[1] + SHEET_GAP + grid_w, 3),
        SHEET_BACKGROUND, dtype=np.uint8,
    )
    canvas[:face.shape[0], :face.shape[1]] = face

    image = Image.fromarray(canvas)
    draw = ImageDraw.Draw(image)
    left = face.shape[1] + SHEET_GAP
    for index, (patch, crop, box) in enumerate(zip(patches, crops, boxes)):
        row, column = divmod(index, columns)
        x = left + SHEET_GAP + column * (cell_w + SHEET_GAP)
        y = SHEET_GAP + row * (cell_h + SHEET_GAP)
        panel = Image.fromarray(
            crop.astype(np.uint8) if crop.ndim == 3
            else np.stack([crop] * 3, axis=-1).astype(np.uint8)
        )
        image.paste(panel, (x, y))
        colour = colour_for(patch.band, index)
        draw.rectangle(
            [x, y, x + crop.shape[1] - 1, y + crop.shape[0] - 1],
            outline=colour,
        )
        # The pixel size IS the finding under review -- caption it. On the
        # as-fed sheet the STRETCH is the finding, so name it too: a
        # reviewer must be able to see 149x124 became 224x224 at 1.20:1.
        name = names[index]
        caption = f"{name} {box[2]}x{box[3]}"
        if as_fed:
            stretch = (box[2] / box[3]) / 1.0
            caption = (
                f"{name} {box[2]}x{box[3]}>{crop.shape[1]} "
                f"{stretch:.2f}:1"
            )
        draw.text((x, y + crop.shape[0] + 1), caption, fill=colour)
    return np.asarray(image)


def native_crop_sizes(staged: Staged, patches: list) -> dict:
    """What the sheet is asking the reviewer to judge, as numbers.

    Reported beside the sheet so the review has the magnification premise
    in front of it: the brief predicted "~96 px wide" at 768 staging.
    """
    boxes = region_boxes(staged, patches)
    widths = [box[2] for box in boxes]
    heights = [box[3] for box in boxes]
    return {
        "n_regions": len(boxes),
        "width_px": {"min": min(widths), "max": max(widths),
                     "median": int(np.median(widths))},
        "height_px": {"min": min(heights), "max": max(heights),
                      "median": int(np.median(heights))},
        "staged_shape": list(staged.image.shape[:2]),
        "brief_predicted_width_px": 96,
        "upsample_to_backbone": {
            "min": round(CROP_OUTPUT_SIZE / max(widths), 3),
            "max": round(CROP_OUTPUT_SIZE / min(widths), 3),
        },
    }


def region_names(patches: list, layout: str) -> list[str]:
    """The ordered axis labels for a crop set, by layout.

    **[FIXED 2026-08-14] Laterality is an anatomical fact, not a
    positional one, and it stops being computable once placement is
    randomised.** ``render.abbreviate`` derives ``_L``/``_R`` from a box's
    centre, which is right for the anatomy layout -- its 8 midline boxes
    sit at exactly x=0.5 and take no suffix, its 7 bilateral pairs take
    the correct sides. Applied to randomly placed boxes it produced
    ``nroot_R`` (a midline structure with a side) and ``rim_L`` twice (a
    pair that happened to land on the same side), which collided and
    stopped the run.

    **The boxes were never wrong.** ``random_patches`` re-places the 22
    anatomy boxes and preserves band, width, height and total area
    exactly (``area_matches``) -- the matching the control depends on was
    intact throughout. Only the labels were derived from a quantity that
    no longer meant anything.

    So the random layout is named positionally and neutrally. Naming its
    boxes after anatomy would be worse than colliding: a consumer asking
    which column is ``philtral_column`` would get a box that is not on
    the philtral column, and the whole point of the arm is that its
    placement is not anatomical.
    """
    from .geometry.render import abbreviate

    if layout == "anatomy":
        names = [
            abbreviate(p.band, p.x + p.w / 2, p.clamped) for p in patches
        ]
    elif layout == "random":
        names = [f"random_{index:02d}" for index in range(len(patches))]
    elif layout == "whole":
        names = ["whole_frame"]
    else:
        raise RegionCropError(f"unknown layout {layout!r}")

    if len(names) != len(patches):
        raise RegionCropError(
            f"{len(names)} names for {len(patches)} boxes in layout {layout!r}"
        )
    if len(set(names)) != len(names):
        duplicated = sorted({n for n in names if names.count(n) > 1})
        raise RegionCropError(
            f"region names are not unique in layout {layout!r}: "
            f"{duplicated}. An axis a consumer cannot index by name is an "
            "axis it must trust positionally, which is what naming exists "
            "to prevent"
        )
    return names


def size_provenance(template: list, placed: list, layout: str) -> dict:
    """Which anatomy box each random box inherited its size from.

    The random names are deliberately anatomy-free, so the size matching
    that makes the control a control is recorded HERE rather than implied
    by a label. Auditable without reading the placement code.
    """
    if layout != "random":
        return {}
    return {
        name: {"inherits_size_from": source.band,
               "w": round(source.w, 6), "h": round(source.h, 6)}
        for name, source in zip(region_names(placed, "random"), template)
    }


def area_matches(template: list, placed: list, tolerance: float = 1e-9) -> bool:
    """Same count, same size multiset, same total area -- by construction,
    asserted anyway because "by construction" is a claim about code."""
    if len(template) != len(placed):
        return False
    sizes_a = sorted((round(p.w, 12), round(p.h, 12)) for p in template)
    sizes_b = sorted((round(p.w, 12), round(p.h, 12)) for p in placed)
    if sizes_a != sizes_b:
        return False
    total_a = sum(p.w * p.h for p in template)
    total_b = sum(p.w * p.h for p in placed)
    return abs(total_a - total_b) <= tolerance
