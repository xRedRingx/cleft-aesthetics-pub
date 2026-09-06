"""Road B Phase 2's two contact sheets. CLUSTER-ONLY -- patient faces.

**Two sheets, because one cannot do both jobs**
(``roadb.SHEET_DENSITY``), and this reasoning is recorded so nobody later
"simplifies" it into one:

**The axis sheet** puts a few patients against all ten settings on one page,
downscaled to a common display size. How the settings differ IS the review, so
the axis has to be visible at a glance; one sheet per setting would put it
across ten files and turn comparison into flipping between images.

**But downscaling to fit ten columns destroys exactly the sharpness
difference a resolution axis exists to show.** Everything rendered at a common
size looks equally sharp. So the axis sheet cannot answer the interpolation
question and does not pretend to.

**The detail strip** answers it: the flagged upsampled patients at NATIVE
resolution. They are where interpolation is real, and few enough to render
without downscaling. **[CORRECTED 2026-08-09]** the first draft quoted 7 at
512 and 81 at 768 -- those are the 473-source ceilings; the staged frontal
counts measured **1 at 512 and 41 at 768** on the first staging runs.
"""

from __future__ import annotations

import numpy as np

from .geometry.render import Panel, tile

#: The axis sheet's per-panel display size. Small enough that ten columns
#: fit on a screen, which is the point -- a sheet legible only at full
#: resolution is one nobody reviews.
DISPLAY_SIZE = 160

#: How many patients the axis sheet shows. Few, because ten columns each is
#: already 10N panels and the axis is what is being read, not the cohort.
AXIS_PATIENTS = 4


def downscale(image, size: int = DISPLAY_SIZE) -> np.ndarray:
    """Nearest-neighbour, deliberately.

    A smoothing resize would make a 512 crop and a 768 crop look identical at
    display size, which is the confusion this sheet already cannot resolve.
    Nearest keeps aliasing visible as a hint that something was resampled,
    without pretending the sheet answers the sharpness question.
    """
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"expected an RGB image, got {array.shape}")
    rows = (np.arange(size) * array.shape[0] // size).clip(0, array.shape[0] - 1)
    cols = (np.arange(size) * array.shape[1] // size).clip(0, array.shape[1] - 1)
    return array[np.ix_(rows, cols)].astype(np.uint8)


def axis_sheet(crops_by_setting: dict, patients: list, *,
               size: int = DISPLAY_SIZE) -> dict:
    """Patients down, all ten settings across, one page.

    ``crops_by_setting`` is ``{setting name: {patient: image}}``. Settings are
    laid out in a fixed order so a column means the same thing on every row.
    """
    order = [s["name"] for s in _settings() if s["name"] in crops_by_setting]
    missing = sorted(set(crops_by_setting) - set(order))
    if missing:
        raise ValueError(f"unknown settings: {missing}")
    if not order:
        raise ValueError("no settings to render")

    panels, absent = [], []
    for patient in patients:
        for name in order:
            image = crops_by_setting[name].get(patient)
            if image is None:
                absent.append((patient, name))
                continue
            panels.append(Panel(
                label=f"{patient} {_short(name)}",
                image=downscale(image, size),
            ))
    if absent:
        raise ValueError(f"{len(absent)} patient/setting crops missing, e.g. {absent[:3]}")
    return {
        "sheet": tile(panels, columns=len(order)),
        "columns": order,
        "patients": list(patients),
        "display_size": size,
        # For save_sheet's label pass: same order as the panels, and the cell
        # is uniform because every panel is downscaled to `size`.
        "labels": [panel.label for panel in panels],
        "cell": (size, size),
        "answers": "is the framing right, and how does it differ across the axis",
        "does_not_answer": (
            "sharpness -- every panel is downscaled to a common size, so 512 "
            "and 768 render identically here. That is the detail strip's job"
        ),
    }


def detail_strip(crops_by_setting: dict, upsampled_by_setting: dict, *,
                 max_patients: int = 6, columns: int | None = None) -> dict:
    """The flagged upsampled patients, at NATIVE resolution, one setting per row.

    Only settings that actually upsample appear: at 224 nothing does, so a row
    there would be an empty claim. ``max_patients`` caps how many of a
    setting's flagged patients render -- and the cap is REPORTED, because a
    silently truncated sheet reads as complete.

    **[CORRECTED 2026-08-09] ``columns`` is the row width, separate from the
    cap.** The first draft tiled at ``columns=max_patients``, which coupled
    the sheet's width to the cap and let a short setting's row run into the
    next setting's panels. A setting now wraps across as many rows as it
    needs and always starts its own row -- a row that mixes settings makes
    the rows stop meaning one thing (``roadb.SHEET_DENSITY``). Measured
    frontal counts are 1 at 512 and 41 at 768, so at the default width a 768
    setting occupies seven rows rather than truncating to six patients.
    """
    columns = columns if columns is not None else min(max_patients, 6)
    blank = Panel(label="", image=np.full((1, 1, 3), 128, dtype=np.uint8))

    rows, shown = [], {}
    for name, patients in sorted(upsampled_by_setting.items()):
        if not patients:
            continue
        taken = list(patients)[:max_patients]
        shown[name] = {"n_flagged": len(patients), "n_shown": len(taken)}
        for patient in taken:
            image = crops_by_setting[name][patient]
            rows.append(Panel(
                label=f"{patient} {_short(name)} NATIVE {image.shape[0]}px",
                image=np.asarray(image, dtype=np.uint8),
            ))
        # Pad the setting out to a whole number of rows, so the next setting
        # starts its own.
        while len(rows) % columns:
            rows.append(blank)
    if not rows:
        return {
            "sheet": None, "shown": {},
            "why_empty": "no setting upsamples any patient; there is nothing to inspect",
        }
    cell = (
        max(panel.image.shape[1] for panel in rows),
        max(panel.image.shape[0] for panel in rows),
    )
    return {
        "sheet": tile(rows, columns=columns),
        "shown": shown,
        "native": True,
        "columns": columns,
        "labels": [panel.label for panel in rows],
        "cell": cell,
        "answers": "do the upsampled patients look interpolated",
        "truncated": {
            name: entry["n_flagged"] - entry["n_shown"]
            for name, entry in shown.items()
            if entry["n_flagged"] > entry["n_shown"]
        },
    }


def _settings():
    from . import roadb

    return roadb.staging_settings()


def _short(name: str) -> str:
    """``roadB_p2_512_nonsquare_g2`` -> ``512 ns g2``, so a label fits."""
    parts = name.replace("roadB_p2_", "").split("_")
    if len(parts) != 3:
        return name
    resolution, aspect, geometry = parts
    return f"{resolution} {'ns' if aspect == 'nonsquare' else 'sq'} {geometry}"


#: What the reviewer is being asked, in the order the sheets answer it.
#: Written down because "does it look right" is not a reviewable question.
REVIEW_QUESTIONS = (
    "AXIS SHEET: is the crop the same face at every setting -- does any "
    "column show a different region, a shifted centre or a lost chin",
    "AXIS SHEET: is the pad present at square and absent at non-square, and "
    "are the trapezium corners present at G1 and absent at G2",
    "AXIS SHEET: does any column look systematically different in a way the "
    "setting does not explain",
    "DETAIL STRIP: do the flagged patients look soft or blocky at native "
    "resolution -- interpolation should be visible, and if it is not, the "
    "upsampling flag may be measuring the wrong thing",
)
