"""The Grad-CAM contact sheet. CLUSTER-ONLY -- it renders patient faces.

**This is the check no test substitutes for.** Everything in ``gradcam.py``
verifies that the arithmetic is right and that a map which cannot be computed
is refused. None of it can say whether the maps land on plausible anatomy,
and that has been true at every phase where it mattered.

Three things are drawn per patient, and each answers a question the others
cannot:

**The crop**, so the reviewer knows what was looked at.

**The map over the crop, with the 14x14 token grid drawn on it.** The grid is
not decoration. A 14x14 field upsampled to 224 is a 16x interpolation, so
nothing finer than one cell is real -- and a reviewer studying a warm patch
over the philtral column needs that in view *while looking*, not in a caption
they read earlier. ``phase8.RESOLUTION_FLOOR``.

**The map's own scale.** Normalisation divides every map by its own maximum,
so a diffuse map and a concentrated one both arrive stretched to full range
and look equally decisive. The concentration figures beside each panel are
what distinguish them, and they are computed BEFORE normalisation.
"""

from __future__ import annotations

import numpy as np

from . import phase8
from .geometry.render import Panel, tile

#: Drawn at 40% grey. Visible against both a warm map and a pale cheek, and
#: light enough not to be mistaken for anatomy.
GRID_VALUE = 102

#: How much of the map's mass sits in its strongest tokens. Reported at two
#: depths because one number cannot separate "one hot spot" from "a broad
#: warm region": a map with 60% in its top 10 is concentrated, one with 60%
#: in its top 50 is not, and both can look identical once normalised.
CONCENTRATION_AT = (10, 50)


def concentration(grid_map) -> dict:
    """What a normalised picture cannot show: how peaked the map actually is.

    **Computed on the map before normalisation**, because dividing by the
    maximum destroys exactly the quantity being reported. Everything here is a
    ratio, so it is comparable between patients even though the raw scales are
    not.
    """
    values = np.asarray(grid_map, dtype=np.float64).ravel()
    total = float(values.sum())
    if total <= 0:
        raise ValueError(
            "the map has no positive mass, so it has no concentration to "
            "report -- gradcam.normalise would have refused it too"
        )
    ordered = np.sort(values)[::-1]
    report = {
        "peak": float(ordered[0]),
        "total_mass": total,
        "n_tokens": int(values.size),
    }
    for k in CONCENTRATION_AT:
        report[f"top_{k}_share"] = float(ordered[:k].sum() / total)
    # A single number for the sheet's label, bounded and orientation-fixed:
    # 0 is uniform, 1 is all mass in one token.
    uniform = 1.0 / values.size
    report["excess_over_uniform"] = float(
        (ordered[: CONCENTRATION_AT[0]].sum() / total)
        - CONCENTRATION_AT[0] * uniform
    )
    return report


#: **[REVIEW 2026-08-04] The two edge statistics, and they are reported
#: SEPARATELY because their causes differ.**
#:
#: The reviewed sheet showed several maps putting most of their mass at the
#: frame border -- four bright corner spots on one patient, bright vertical
#: columns down both edges on two others, a bright top edge on a fourth. One
#: patient was the exception, a broad warm region over the central face, so it
#: is not universal.
#:
#: **Out-of-content mass** is the fraction outside the trapezium content box.
#: Its uniform expectation is the pad fraction, 0.2534 at the median aspect
#: ratio. **This is a property of these CROPS**: pad-square-then-resize leaves
#: white corners at G1, and they vanish at G2 where the mask is the full
#: square.
#:
#: **Outer-ring mass** is the fraction in the outermost ring of the token
#: grid, 52 of 196 cells, uniform expectation 0.2653. **This is a property of
#: the ARCHITECTURE**: a ViT's border patches have fewer neighbours to attend
#: to, and the behaviour would appear on a full-frame image with no pad at
#: all. It survives at G2; out-of-content does not.
#:
#: Collapsing them into one "edge mass" figure would attribute an
#: architectural artefact to a staging choice, or the reverse, and the two
#: have different consequences for whether the maps mean anything.
#: **[CORRECTED 2026-08-08] The out-of-content expectation is derived PER
#: PATIENT from that patient's own mask. The constant is gone, not fixed.**
#:
#: It was ``PAD_FRACTION_AT_MEDIAN_AR = 0.2534`` -- the cohort MEAN pad
#: fraction, applied to every patient as though it were theirs.
#: ``geometry/staging.py`` already records the real distribution as min
#: 0.0179, mean 0.2534, max 0.4464, so the constant was both a second copy of
#: measured data and a threshold from the wrong population: the same class of
#: defect as an inherited seed band.
#:
#: **And it was wrong in a second, larger way.** ``content_mask`` excludes the
#: pad AND the trapezium corners; ``pad_fraction`` covers only the pad. At the
#: median aspect ratio the corners add another 0.148, so the true expectation
#: is 0.4028 against the 0.2534 that was used. Replacing 0.2534 with a better
#: constant would have repeated the defect in a third form.
#:
#: So the expectation is ``1 - mask.mean()``: the fraction of THIS patient's
#: frame that is not content, computed from the same object the mass is
#: measured against. One source of truth, automatically right at any geometry.
#:
#: **At non-square there is no pad and the trapezium is the full square, so
#: the expectation is 0 and the statistic is UNDEFINED** -- there is no
#: out-of-content region for mass to fall into. Reported as ``None`` with a
#: reason, the way ``ladder.PAIRED_CLAIM_COVERAGE`` records the label
#: comparison as undefined rather than uncomputed. A ratio against zero is not
#: infinite; it does not exist.
OUTER_RING_UNIFORM = 52 / 196


def outer_ring(grid=None) -> np.ndarray:
    """Boolean mask of the token grid's outermost ring."""
    rows, cols = tuple(grid or phase8.GRAD_CAM_TARGET["grid"])
    mask = np.zeros((rows, cols), dtype=bool)
    mask[0, :] = mask[-1, :] = True
    mask[:, 0] = mask[:, -1] = True
    return mask


def outer_ring_mass(grid_map) -> float:
    """Share of the map's mass in the border cells.

    Uniform expectation is ``OUTER_RING_UNIFORM``. At or above it, the map is
    no more concentrated on the interior than chance.
    """
    values = np.asarray(grid_map, dtype=np.float64)
    total = float(values.sum())
    if total <= 0:
        raise ValueError("the map has no positive mass")
    return float(values[outer_ring(values.shape)].sum() / total)


def cell_coverage(mask, grid=None) -> np.ndarray:
    """What fraction of each token cell is content, not pad.

    **Reduced to CELL resolution rather than the map inflated to pixels.** The
    map carries no information finer than one cell
    (``phase8.RESOLUTION_FLOOR``), so upsampling it to compare against a
    pixel mask would weight interpolated values that are not measurements.
    """
    array = np.asarray(mask, dtype=np.float64)
    rows, cols = tuple(grid or phase8.GRAD_CAM_TARGET["grid"])
    if array.shape[0] % rows or array.shape[1] % cols:
        raise ValueError(
            f"a {array.shape} mask does not divide into a {rows}x{cols} grid"
        )
    block = (array.shape[0] // rows, array.shape[1] // cols)
    return array.reshape(rows, block[0], cols, block[1]).mean(axis=(1, 3))


def out_of_content(grid_map, mask) -> dict:
    """Mass outside the content region, against THIS patient's own expectation.

    Returns ``mass``, the ``uniform`` expectation derived from the mask, their
    ``ratio``, and whether the map is ``above`` chance. **``ratio`` and
    ``above`` are ``None`` when the expectation is zero** -- with no pad and a
    full-square trapezium there is no out-of-content region, so the statistic
    does not exist rather than evaluating to something.
    """
    values = np.asarray(grid_map, dtype=np.float64)
    total = float(values.sum())
    if total <= 0:
        raise ValueError("the map has no positive mass")
    coverage = cell_coverage(mask, values.shape)
    mass = float((values * (1.0 - coverage)).sum() / total)
    # **Named ``expected``, not ``uniform``.** It was one cohort constant and
    # is now this patient's own non-content fraction, so "uniform" describes
    # what it used to be. The name was reached for as ``expected`` by the
    # first person to query the artifact, which is the test that matters.
    expected = float(1.0 - np.asarray(mask, dtype=np.float64).mean())
    if expected <= 0.0:
        return {
            "mass": mass, "expected": expected, "ratio": None, "above": None,
            "undefined_because": (
                "the frame is entirely content -- no pad and a full-square "
                "trapezium -- so there is no out-of-content region. The "
                "statistic does not exist here rather than being uncomputed"
            ),
        }
    return {
        "mass": mass, "expected": expected, "ratio": mass / expected,
        "above": bool(mass > expected),
    }


def draw_token_grid(image, grid=None, value: int = GRID_VALUE) -> np.ndarray:
    """Draw the token lattice, so the resolution floor is visible while looking.

    One line per token boundary. A reviewer who can see that the whole
    philtral column sits inside a single cell cannot read the map as
    localising to it.
    """
    canvas = np.asarray(image, dtype=np.uint8).copy()
    if canvas.ndim != 3 or canvas.shape[2] != 3:
        raise ValueError(f"expected an RGB image, got {canvas.shape}")
    rows, cols = tuple(grid or phase8.GRAD_CAM_TARGET["grid"])
    height, width = canvas.shape[:2]
    for r in range(1, rows):
        canvas[int(round(r * height / rows)), :, :] = value
    for c in range(1, cols):
        canvas[:, int(round(c * width / cols)), :] = value
    return canvas


def overlay(crop, image_map, *, alpha: float = 0.55) -> np.ndarray:
    """The map over the crop, as a red channel lift.

    Deliberately not a rainbow colourmap: a jet or turbo ramp invents
    boundaries where the data is smooth, which on a 16x-interpolated field is
    the last thing this sheet should add.
    """
    base = np.asarray(crop, dtype=np.float64)
    if base.ndim == 2:
        base = np.repeat(base[:, :, None], 3, axis=2)
    weight = np.asarray(image_map, dtype=np.float64)
    if weight.shape[:2] != base.shape[:2]:
        raise ValueError(
            f"map {weight.shape[:2]} does not match crop {base.shape[:2]}"
        )
    lifted = base.copy()
    lifted[:, :, 0] = base[:, :, 0] * (1 - alpha * weight) + 255.0 * alpha * weight
    lifted[:, :, 1] = base[:, :, 1] * (1 - alpha * weight)
    lifted[:, :, 2] = base[:, :, 2] * (1 - alpha * weight)
    return np.clip(lifted, 0, 255).astype(np.uint8)


def per_patient(patient: int, crop, grid_map, image_map, *,
                seed_agreement=None, survivor=None, content=None) -> dict:
    """One patient's three panels and the numbers that go beside them.

    ``survivor`` marks a map that survived model-parameter randomisation, with
    its percentile in the between-patient baseline. **It goes on the patient's
    OWN panel** (``phase8.PUBLICATION_SCOPE``): a reviewer studying this face
    should know the map has the weakest evidence of parameter dependence while
    looking at it, not from a header they read earlier.
    """
    report = concentration(grid_map)
    report["patient_id"] = int(patient)
    report["outer_ring_mass"] = outer_ring_mass(grid_map)
    report["outer_ring_uniform"] = OUTER_RING_UNIFORM
    if content is not None:
        report["out_of_content"] = out_of_content(grid_map, content)
    if seed_agreement is not None:
        values = np.asarray(seed_agreement, dtype=float)
        report["seed_agreement_median"] = float(np.median(values))
        report["seed_agreement_min"] = float(values.min())
    mark = ""
    if survivor is not None:
        report["survived_randomisation"] = True
        report["baseline_percentile"] = float(survivor)
        mark = f"  ** SURVIVED RANDOMISATION, {survivor:.0%} pct **"
    panels = [
        Panel(label=f"{patient} crop{mark}",
              image=np.asarray(crop, dtype=np.uint8)),
        Panel(
            label=(
                f"{patient} map  top{CONCENTRATION_AT[0]} "
                f"{report[f'top_{CONCENTRATION_AT[0]}_share']:.0%}  ring "
                f"{report['outer_ring_mass']:.0%}"
                + (
                    "  pad {mass:.0%}/{expected:.0%}".format(**report["out_of_content"])
                    if "out_of_content" in report else ""
                )
                + mark
            ),
            image=draw_token_grid(overlay(crop, image_map)),
        ),
        Panel(
            label=(
                f"{patient} scale  peak {report['peak']:.3g}  "
                f"top{CONCENTRATION_AT[1]} "
                f"{report[f'top_{CONCENTRATION_AT[1]}_share']:.0%}"
            ),
            image=draw_token_grid(
                np.repeat(
                    (np.asarray(image_map, dtype=np.float64) * 255)
                    .clip(0, 255).astype(np.uint8)[:, :, None],
                    3, axis=2,
                )
            ),
        ),
    ]
    return {"report": report, "panels": panels}


def verdict_banner(verdict: dict) -> str:
    """The gate's outcome AND its numbers, for the top of the sheet.

    **Not the word FAILED alone.** That invites discounting all fifteen maps
    when thirteen sit strongly below the baseline, and the numbers are what
    distinguish a survivor at the 91st percentile from one at the 62nd.
    ``phase8.PUBLICATION_SCOPE``.
    """
    survivors = verdict.get("survivors", [])
    head = (
        "RANDOMISATION TEST PASSED -- maps depend on the model's parameters"
        if verdict.get("passes") else
        f"RANDOMISATION TEST FAILED: {len(survivors)} of "
        f"{verdict.get('n_patients', '?')} maps survive. NOT PUBLISHED"
    )
    parts = [head, f"baseline {verdict.get('baseline', float('nan')):.4f}"]
    spread = verdict.get("baseline_distribution")
    if spread:
        parts.append(
            f"baseline sd {spread['sd']:.4f}, range "
            f"[{spread['min']:.3f}, {spread['max']:.3f}]"
        )
    separable = verdict.get("separability")
    if separable:
        parts.append(
            f"separability {separable['observed_difference']:+.4f} "
            f"CI [{separable['ci95'][0]:+.4f}, {separable['ci95'][1]:+.4f}]"
        )
    return " | ".join(parts)


def build_sheet(entries: list[dict], columns: int = 3) -> np.ndarray:
    """Three panels per patient, one patient per row."""
    if not entries:
        raise ValueError("no patients to render")
    panels = [panel for entry in entries for panel in entry["panels"]]
    return tile(panels, columns=columns)


#: **[DECIDED 2026-08-04] What the reviewer is being asked, in the order the
#: sheet answers it.** Written down because "does it look right" is not a
#: reviewable question and every previous sheet in this project needed the
#: same discipline.
REVIEW_QUESTIONS = (
    "does the warm region sit on the nasolabial area, or on hair, background "
    "or the white pad",
    "is the map concentrated or diffuse -- read the top-10 share, NOT the "
    "picture, because normalisation stretches every map to full range",
    "does any apparent localisation depend on structure finer than one grid "
    "cell; if so it is the interpolation kernel and not the model",
    "do the five seeds agree -- read the agreement figure, not the mean map",
)


def edge_convergence(entries: list[dict], *, finals=None, n_boot: int = 2000,
                     seed: int = 1337) -> dict:
    """Do the edge-dominated maps coincide with the weak ones?

    **Three statistics converging on the same patients would be one finding,
    not three.** If the maps with the most border mass are also the ones that
    survived randomisation and the ones whose seeds disagree, then all three
    are measuring "this map carries nothing" -- which is a stronger and
    simpler reading than three separate weaknesses.

    Spearman, with a bootstrap CI over patients. **At n=15 the interval will
    be wide, and a wide one is the answer**: per
    ``phase8.COMPARATIVE_CLAIMS_NEED_A_NUMBER`` this is reported as
    descriptive unless it excludes zero.
    """
    from .gradcam import similarity

    reports = [entry["report"] for entry in entries]
    ring = np.array([r["outer_ring_mass"] for r in reports], dtype=float)
    pairs = {}
    if any("seed_agreement_median" in r for r in reports):
        pairs["ring_vs_seed_agreement"] = np.array(
            [r.get("seed_agreement_median", np.nan) for r in reports], float
        )
    if finals is not None:
        pairs["ring_vs_randomisation_final"] = np.asarray(finals, dtype=float)

    rng = np.random.default_rng(seed)
    out = {"n_patients": len(reports), "outer_ring_mass": ring.tolist()}
    for name, other in pairs.items():
        keep = ~np.isnan(other)
        a, b = ring[keep], other[keep]
        if a.size < 3:
            continue
        observed = similarity(a, b)
        draws = []
        for _ in range(n_boot):
            idx = rng.integers(0, a.size, size=a.size)
            if np.std(a[idx]) == 0 or np.std(b[idx]) == 0:
                continue
            draws.append(similarity(a[idx], b[idx]))
        lo, hi = np.percentile(draws, [2.5, 97.5]) if draws else (np.nan, np.nan)
        # `bool(draws) and (...)` returns the SECOND operand when the
        # first is truthy, so this was np.bool_ and crashed the report
        # write. Cast here as well as at the boundary: a wrong type in
        # the record is a defect even where json tolerates it.
        excludes = bool(draws and (lo > 0 or hi < 0))
        out[name] = {
            "spearman": float(observed),
            "ci95": [float(lo), float(hi)],
            "excludes_zero": excludes,
            "reading": (
                "the edge-dominated maps are the weak ones -- one finding, "
                "not three"
                if excludes else
                "DESCRIPTIVE: the interval covers zero, so the convergence is "
                "not established at this sample size"
            ),
        }
    return out


def summarise(entries: list[dict]) -> dict:
    """The counts a reader needs beside the sheet, not instead of it."""
    reports = [entry["report"] for entry in entries]
    top_k = f"top_{CONCENTRATION_AT[0]}_share"
    shares = np.array([r[top_k] for r in reports], dtype=float)
    uniform = float(CONCENTRATION_AT[0] / reports[0]["n_tokens"])
    summary = {
        "n_patients": len(reports),
        "grid": list(phase8.GRAD_CAM_TARGET["grid"]),
        "resolution_floor_px": phase8.RESOLUTION_FLOOR["floor_px"],
        "uniform_top_k_share": uniform,
        f"{top_k}_median": float(np.median(shares)),
        f"{top_k}_min": float(shares.min()),
        # **A map at or below the uniform share is not localising anything**,
        # however decisive it looks once normalised.
        "n_at_or_below_uniform": int((shares <= uniform).sum()),
        "review_questions": list(REVIEW_QUESTIONS),
        "caveat": phase8.FROZEN_BACKBONE_CAVEAT["text"],
    }
    # **The two edge statistics, kept apart.** Out-of-content is a property
    # of these crops and vanishes at G2; outer-ring is a property of the
    # architecture and does not.
    ring = np.array([r["outer_ring_mass"] for r in reports], dtype=float)
    summary["outer_ring_mass_median"] = float(np.median(ring))
    summary["outer_ring_uniform"] = OUTER_RING_UNIFORM
    summary["n_ring_at_or_above_uniform"] = int((ring >= OUTER_RING_UNIFORM).sum())
    # **Each patient against their OWN expectation.** The ratios are what
    # aggregate; the masses do not, because every patient's chance level is
    # different. Undefined entries (no pad, full-square trapezium) are counted
    # and excluded rather than treated as zero.
    pad = [r["out_of_content"] for r in reports if "out_of_content" in r]
    defined = [entry for entry in pad if entry["ratio"] is not None]
    if pad:
        summary["n_out_of_content_undefined"] = len(pad) - len(defined)
    if defined:
        ratios = np.array([entry["ratio"] for entry in defined], dtype=float)
        summary["out_of_content_mass_median"] = float(
            np.median([entry["mass"] for entry in defined])
        )
        summary["out_of_content_expected_median"] = float(
            np.median([entry["expected"] for entry in defined])
        )
        summary["out_of_content_ratio_median"] = float(np.median(ratios))
        summary["n_out_of_content_above_own_uniform"] = int(
            sum(1 for entry in defined if entry["above"])
        )
        summary["n_out_of_content_defined"] = len(defined)

    agreements = [
        r["seed_agreement_median"] for r in reports
        if "seed_agreement_median" in r
    ]
    if agreements:
        summary["seed_agreement_median"] = float(np.median(agreements))
        summary["seed_agreement_min"] = float(np.min(agreements))
    return summary
