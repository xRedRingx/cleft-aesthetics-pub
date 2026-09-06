"""Phase 8c: "show how the model trained and predicted" -- a RENDERING.

Nothing here is measured. Every number on every panel is read from an
artifact that already exists, and the sheet-review rule governs: the sheet
must show what the consumer receives. The scope line of 2026-08-13 applies
throughout -- cohort patients, never live inference on an arbitrary face --
and every input is CLUSTER-ONLY, so every output is too.

Two tasks consume this module (``run.task_phase8c_sheet``,
``run.task_phase8c_animation``): the sheet is a CPU artifact-read job, the
animation is a GPU refit-plus-forwards job, and they are split so the sheet
can re-render after review without refitting anything.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

#: **[DECIDED 2026-08-15, maintainer -- the panel-1 naming decision.]**
#: Panel 1 shows the **22 cleft-anatomy regions**, labelled
#: "22 cleft-anatomy regions (of the 27-patch frozen scheme)".
#:
#: The dated naming note: the amendment's phrase "the 22 anatomy regions"
#: RESOLVES to ``roadb_regioncrop.protected_patches()`` -- a Road B Branch 3
#: module -- not to the frozen anatomy SCHEME, which is 27 patches
#: (``patches_for("anatomy", geometry)``). Measured 2026-08-15: the 22 are a
#: strict subset of the 27 by name identity; the five scheme-only patches
#: are the periorbital periphery (glabella, both medial canthi, both lateral
#: orbits). Both sets derive from the same frozen Phase-2 patch definitions,
#: and the placement machinery (``staged_from_row`` + ``map_patches``) is
#: shared, so the choice changes WHICH boxes draw, not how.
PANEL_1_NAMING = {
    "decided": "2026-08-15, maintainer",
    "shows": "the 22 cleft-anatomy subset (roadb_regioncrop.protected_patches)",
    "label": "22 cleft-anatomy regions (of the 27-patch frozen scheme)",
    "naming_note": (
        "the amendment's '22 anatomy regions' resolves to the Road B "
        "Branch 3 module, not the 27-patch frozen anatomy scheme; 22 is a "
        "strict subset of 27 by name identity, verified 2026-08-15"
    ),
    "scheme_only_patches": (
        "glabella, medial_canthus x2, lateral_orbit x2 -- the periorbital "
        "periphery",
    ),
}

#: **[DESIGN 2026-08-15] The static sheet: form, panels, licences.**
#:
#: One PNG page per patient for the FIFTEEN stratified patients (the map
#: panels exist only for them) plus one cohort page whose panel-4 covers
#: all 237 -- model error against rater disagreement as a scatter with the
#: fifteen marked. CLUSTER-ONLY, landing in the task's own run directory.
#:
#: On every page banner: both verdict stamps and the sentence "no map on
#: this sheet is a published claim". On every map-shaped panel: the
#: frozen-backbone caveat VERBATIM from the artifact it came from, and
#: 16-px nearest-neighbour rendering -- ``block_upsample`` introduces no
#: value that was not in the 14x14 grid, which is the honesty the bilinear
#: ``gradcam.upsample`` cannot give a clinical page.
SHEET_DESIGN = {
    "decided": "2026-08-15",
    "form": "PNG per patient (15) + one cohort page; CLUSTER-ONLY, run dir",
    "panels": (
        "1 face + 22 cleft-anatomy regions",
        "2 Grad-CAM original beside variant, stamps + both_numbers_sentence "
        "read from grad_cam_methods.npz, 16-px nearest blocks",
        "3 node-weight FLATNESS against the 1/27 line -- the ranking-vs-no-"
        "ranking contrast made visual; no ranking fabricated from dust",
        "4 prediction vs the five raters' score multiset, five-seed spread",
        "5 training: per-fold loss/early-stop curves from curves.csv + the "
        "frozen-plus-probe structure",
    ),
    "cohort_page": (
        "panel 4 for all 237: |model error| against rater disagreement, "
        "the fifteen sampled patients marked"
    ),
    "banner": (
        "both verdict stamps; 'no map on this sheet is a published claim'"
    ),
    "rater_columns": (
        "soft_1..soft_5 are grade FRACTIONS, not per-rater columns; the "
        "five scores are recovered as a multiset (5 x soft_k, asserted "
        "integral at runtime) and are not attributable to named raters"
    ),
}

#: **[REGISTERED 2026-08-15, before any frame exists] The animation task:
#: axes, frame rules, and the two resolutions the audit forced.**
#:
#: **Axes built**: (a) the training trajectory -- the registered checkpoint
#: subset below; (b) the randomisation walk -- 13 frames, ORIGINAL method,
#: its recorded failed-gate stamp on every frame, clean-snapshot-per-patient
#: loop; (d) depth, blocks 0..10. **Axis (c) -- seeds -- is a 5-frame
#: STATIC strip on the sheet, labelled unordered, never an animation**:
#: seeds are categorical and an animation implies an order they do not have.
#:
#: **(a)'s registered checkpoints**: init; steps 1, 2, 3, 5, 10, 20, 50 of
#: epoch 1 (the head trains by 50 full-batch steps per epoch, so every step
#: is a measured parameter state); then each epoch's end to early stop.
#: Head = the patient's held-out fold head at ``seeds[0]``, the walk
#: precedent. **Per-seed-mean INTERIOR frames are BARRED** -- an ensemble
#: artifact, not a training state.
#:
#: **The endpoint resolution, recorded because the two rules collide.**
#: Interior frames are single-seed states; the static sheet map is the
#: FIVE-SEED MEAN. "Endpoint frames byte-identical to the static maps"
#: therefore cannot be satisfied by the last axis frame itself. Resolution:
#: the final frame of every animation is READ from ``grad_cam_methods.npz``
#: (byte-identity by construction) and labelled "endpoint -- the sheet's
#: map (5-seed mean)", visually separated from the single-seed axis frames.
#: The tension and this resolution are recorded rather than smoothed.
#:
#: **The zero-init frame**: the head is lazily zero-initialised, so at
#: epoch 0 no gradient direction exists and ``gradcam.cam``'s guard would
#: refuse -- correctly. The init frame is therefore an ANNOTATED NO-MAP
#: frame ("epoch 0: zero head, no gradient direction, no map") -- a
#: measured refusal drawn as a datum, the 8b principle.
#:
#: **The depth caveat, registered as written in the audit**: the sweep
#: shows maps at NON-registered layers; block 10 is the assessed endpoint
#: (``phase8.GRAD_CAM_TARGET`` is not a config field precisely because a
#: layer is the thing most likely to be nudged for a better-looking map);
#: no depth frame may be quoted as an alternative "better" map. Block 11 is
#: structurally excluded -- its patch tokens feed nothing.
#:
#: **Frame rules, all axes**: no interpolated frames -- only measured
#: checkpoints are drawn; 16-px nearest-neighbour blocks; the caveat and
#: the method's verdict stamp on every frame; the axis value (epoch/step,
#: stage, block) burned into every frame. Format: APNG assembled by Pillow
#: with the individual PNG frames written beside it -- the frames are the
#: measured objects, the animation is assembly; GIF only on viewer refusal.
ANIMATION_REGISTERED = {
    "registered": "2026-08-15, before any frame exists",
    "axes_built": ("training_trajectory", "randomisation_walk", "depth"),
    "axis_c_seeds": (
        "5-frame STATIC strip on the sheet, labelled unordered -- seeds "
        "are categorical; an animation implies an order they do not have"
    ),
    "a_checkpoints": {
        "epoch_1_steps": (1, 2, 3, 5, 10, 20, 50),
        "then": "each epoch's end to early stop",
        "init": "annotated NO-MAP frame (zero head, no gradient direction)",
        "head": "held-out fold head at seeds[0]; per-seed-mean interior "
                "frames are BARRED (an ensemble artifact)",
    },
    "b_walk": (
        "13 frames: trained endpoint + 12 cumulative stages, ORIGINAL "
        "method, failed-gate stamp on every frame, clean snapshot restored "
        "per patient"
    ),
    "d_depth": {
        "blocks": "0..10 (block 11 structurally excluded: patch tokens "
                  "feed nothing)",
        "caveat": (
            "non-registered layers on display; block 10 is the assessed "
            "endpoint; no frame may be quoted as an alternative 'better' "
            "map"
        ),
    },
    "endpoint_resolution": (
        "interior frames are single-seed states and the sheet map is the "
        "5-seed mean, so the final frame is READ from grad_cam_methods.npz "
        "-- byte-identical by construction -- and labelled 'endpoint: the "
        "sheet's map (5-seed mean)'. The collision of the two rules is "
        "recorded, not smoothed"
    ),
    "frame_rules": (
        "no interpolated frames; 16-px nearest blocks; caveat + verdict "
        "stamp + axis value on every frame"
    ),
    "format": "APNG + individual PNG frames beside; GIF only on viewer refusal",
}


#: **[REBUILT 2026-08-15, after the eye review -- CLINICAL_DELIVERY_REBUILT]**
#: The clinical rephrasing of the frozen-backbone caveat. The VERBATIM
#: caveat still travels -- on the appendix page, from the artifact -- and
#: this sentence carries its meaning to the audience the page is for.
CLINICAL_CAVEAT = (
    "the model's visual features come from general image training, not "
    "from cleft photographs; only the final scoring layer was trained on "
    "this cohort"
)


#: **[REBUILT 2026-08-15, after the maintainer's eye review of both runs]
#: THE MEASURED DATA IS SOUND; THE DELIVERY FAILED ITS AUDIENCE -- rebuilt
#: as one self-contained HTML page per patient, clinical-first.**
#:
#: The review's defects, in its order: (1) no face under the heat anywhere
#: -- heat blocks on black beside the face, a black void for the init
#: frame; (2) insider furniture as the visible text -- stamps, refusal
#: counts, both-numbers sentences -- with no axis labels, no legends, no
#: grade scale, truncated captions, cross-fold sawtooth in the curves, an
#: unreadable cohort page; (3) 45 loose run-dir files with a ``.apng.png``
#: double extension that most viewers render as a static frame zero.
#:
#: **The rebuild**: heat ON the face (``composite_heat``, blocks
#: preserved); the training animation inline in the page (an ``<img>``-
#: embedded APNG animates in any browser); frame zero = the face alone,
#: captioned as the before-state; walk and depth beneath, collapsed,
#: secondary; the seed strip static as registered. Every panel carries one
#: plain-language sentence written for a reader who has never seen this
#: project. The caveat is REPHRASED for the audience (``CLINICAL_CAVEAT``)
#: while the verbatim caveat, both verdict stamps, the refusal ledger, the
#: both-numbers sentence, the flatness panel, the training curves and the
#: UNESTIMABLE banner all move to a LINKED TECHNICAL APPENDIX -- they must
#: exist (the registered rules require the stamps to travel with the maps)
#: and they travel BEHIND the clinical page, not on it. A cohort index
#: page links the fifteen patient pages and carries the fixed scatter.
#:
#: **The frame-rule amendment, dated**: frames now carry the axis value
#: only; the stamps and verbatim caveat ride the appendix the page links.
#: This amends ANIMATION_REGISTERED's frame_rules for DELIVERY -- the
#: measured frames, axes, checkpoints and endpoint resolution are
#: unchanged. The composite is made unviolable by test: a frame's pixels
#: must differ from the pure heat grid, because a Grad-CAM frame without
#: its image is not a Grad-CAM frame.
#:
#: **Input dependence, measured before the rebuild shipped**: same head,
#: two different inputs, through the shipped render path -- maps differ at
#: max |difference| 1.0 (full range). The cluster-side check is one line:
#: compare two patients' same-step frame PNGs for byte-inequality.
CLINICAL_DELIVERY_REBUILT = {
    "rebuilt": "2026-08-15, after the maintainer's eye review",
    "verdict": "measured data sound; delivery failed its audience",
    "defects": (
        "no face under the heat (init frame a black void)",
        "insider furniture as the visible text; no labels, legends or "
        "grade scale; truncation; cross-fold sawtooth; unreadable cohort "
        "page",
        "45 loose files, .apng.png double extension, format fragility",
    ),
    "form": (
        "one self-contained HTML page per patient (images embedded as data "
        "URIs), a linked technical appendix, a cohort index page"
    ),
    "frame_amendment": (
        "frames carry the axis value only; verdict stamps and the verbatim "
        "caveat move to the appendix the page links. Delivery only -- the "
        "measured frames, axes, checkpoints and endpoint resolution are "
        "unchanged"
    ),
    "clinical_caveat": CLINICAL_CAVEAT,
    "appendix_carries": (
        "verbatim caveat, both verdict stamps, refusal ledger, "
        "both-numbers sentence, flatness panel, fold-broken training "
        "curves, UNESTIMABLE banner"
    ),
    "input_dependence_measured": (
        "same head, two inputs, shipped render path: maps differ at max "
        "|difference| 1.0. Cluster check: byte-compare two patients' "
        "same-step frames"
    ),
    "unviolable": (
        "a composited frame's pixels must differ from the pure heat grid "
        "-- pinned by test"
    ),
}



#: **[CLOSED 2026-08-15, review round two PASSED] PHASE 8c IS CLOSED: the
#: amendment's request -- "show how the model trained and predicted" -- is
#: answered in full, on the cohort's own patients, delivered as one entry
#: point.**
#:
#: **The closing artifacts**: the animation run
#: ``p8c_animation__8d8fe58f__p8c-animation-2`` (frames over the three
#: measured axes, endpoint frames read from the methods npz) and the
#: second sheet run (stem ``p8c_sheet``, the rebuilt clinical delivery;
#: its directory identity is the run listing's) -- fifteen self-contained
#: patient pages, the technical appendix, and ``8c_index.html``.
#:
#: **The eye review is the phase's GATE, and it fired once**: the first
#: delivery FAILED on audience -- measured data sound, heat on black,
#: insider furniture as the visible text -- and the rebuild PASSED round
#: two. "Readable by a clinician cold" is now a REGISTERED, TESTED
#: property rather than an intention: the suite pins that a composited
#: frame differs from the pure heat grid, that the before-frame is the
#: photograph, that the clinical page carries no verdict stamp, and that
#: the plain-language sentences are present
#: (``CLINICAL_DELIVERY_REBUILT``).
#:
#: **Input dependence closed both ways**: locally on the render path
#: (same head, two inputs, max |difference| 1.0) and cluster-side on the
#: real frames (p55 vs p120 same-step frames differ, cmp at byte 36).
#:
#: **What 8c consumed -- the phase's artifacts rendered for their final
#: audience**: ``grad_cam_methods.npz`` (both verdict stamps travelling on
#: the appendix), ``node_weights.npz`` (the flatness panel: the ranking-
#: vs-no-ranking contrast a clinician can see), the 0.2520 arm's OOF
#: predictions (the score inside the raters' own disagreement -- the panel
#: nothing had shown before this phase) and its training curves.
PHASE_8C_CLOSING = {
    "closed": "2026-08-15, review round two PASSED",
    "closing_artifacts": {
        "animation_run": "p8c_animation__8d8fe58f__p8c-animation-2",
        "sheet_run": (
            "the second p8c_sheet run -- the rebuilt clinical delivery; "
            "directory identity per the run listing"
        ),
        "deliverable": (
            "fifteen self-contained patient pages + technical appendix + "
            "8c_index.html, one entry point"
        ),
    },
    "gate": {
        "form": "the maintainer's eye review, per delivery round",
        "round_1": (
            "FAILED on audience: measured data sound, heat on black, "
            "insider furniture as the visible text"
        ),
        "round_2": "PASSED -- the HTML delivery is what the request meant",
        "now_a_property": (
            "'readable by a clinician cold' is registered and TESTED, not "
            "an intention: composite != pure heat, before-frame == the "
            "photograph, no stamps on the clinical page, plain sentences "
            "present (CLINICAL_DELIVERY_REBUILT)"
        ),
    },
    "input_dependence": (
        "closed both ways: render path locally (max |difference| 1.0) and "
        "cluster-side cmp on the real frames (p55 vs p120 differ, byte 36)"
    ),
    "request_answered": (
        "training and prediction shown on the cohort's own patients, "
        "animated over measured checkpoints, one entry point -- the "
        "amendment's 8c in full"
    ),
    "consumed": (
        "grad_cam_methods.npz (both stamps on the appendix), "
        "node_weights.npz (the flatness panel), the 0.2520 arm's OOF "
        "predictions and training curves -- the phase's artifacts rendered "
        "for their final audience"
    ),
    "still_untouched": ("t-SNE with its companion -- the last Phase 8 item",),
}


#: **[CORRECTED 2026-09-05] THE PAGES SAID "FIVE CLINICIANS". THE PANEL
#: IS NOT FIVE CLINICIANS.**
THE_PANEL_WAS_MISDESCRIBED_ON_THE_PAGES = {
    "corrected": "2026-09-05",
    "what_was_wrong": (
        "**the index page and every per-patient page described the "
        "graders as 'five clinicians'.** ``scoresheet.RATERS`` is a "
        "CLEFT PATIENT, an orthodontist, a speech and language "
        "therapist, a plastic surgeon and a psychologist. **One of the "
        "five is not a clinician**, and describing the panel as though "
        "all five were is wrong on the face of it"
    ),
    "why_it_is_not_a_wording_quibble": (
        "**it restores an assumption the record spent a section "
        "refuting.** ``phase24.THE_ASSUMPTIONS_DECLARED`` assumption 1 "
        "is that Spearman-Brown treats raters as interchangeable draws "
        "of equal quality, **and that ours are not** -- five different "
        "backgrounds, measured unequal on two instruments (item-total "
        "0.654 against 0.628; trained-model -0.0098 to 0.2703). "
        "'Five clinicians' is exactly the flattening that assumption "
        "exists to deny"
    ),
    "where_it_appeared": (
        "**the worst place available**: the sentence sat beside a "
        "patient's own photographs on the clinical page, and on the "
        "index a reader meets first. A record's internal prose is read "
        "by whoever audits it; an artifact page is read by whoever is "
        "shown the result"
    ),
    "how_it_arose": (
        "**an earlier project document said 'five surgeons'; the record "
        "corrected that to the multidisciplinary panel; the page text "
        "drifted back part of the way to 'five clinicians'.** A "
        "correction that reaches the record but not the artifacts it "
        "generates is not finished -- the same shape as the two-view "
        "mis-tagging, where the correction never reached the governing "
        "document"
    ),
    "SHEETS_ALREADY_GENERATED_CARRY_THE_OLD_WORDING": (
        "**stated plainly rather than left to be discovered.** Every "
        "sheet produced before 2026-09-05 says 'five clinicians' on its "
        "index and on all fifteen patient pages. Those files are on the "
        "cluster and are NOT edited by this correction"
    ),
    "regenerating_them_is_a_SEPARATE_decision": (
        "**and it is not taken here.** Regeneration needs a cluster run "
        "-- the cohort images are not reachable from an authoring "
        "machine -- so it is a launch, and launches are ruled "
        "separately. Until one happens, a banked sheet and this "
        "generator disagree on one sentence, and **the generator is the "
        "one that is right**"
    ),
    "tag": "[CORRECTED] -- generator fixed, banked artifacts untouched",
}


class Phase8cError(RuntimeError):
    """A panel's input is not what the design audited."""


# --------------------------------------------------------------------------
# honesty primitives
# --------------------------------------------------------------------------


def block_upsample(grid: np.ndarray, factor: int = 16) -> np.ndarray:
    """Nearest-neighbour block upsample: the HONEST map rendering.

    ``gradcam.upsample`` is bilinear and therefore draws structure finer
    than the 16-px resolution floor -- the kernel, not the model. This
    introduces NO value that was not in the grid: every output pixel equals
    exactly one input cell.
    """
    array = np.asarray(grid, dtype=np.float64)
    if array.ndim != 2:
        raise Phase8cError(f"expected a 2-D grid, got {array.shape}")
    return np.kron(array, np.ones((factor, factor)))


def heat_panel(grid: np.ndarray, factor: int = 16) -> np.ndarray:
    """A map as red-scale blocks, (H, W, 3) uint8. No smoothing anywhere."""
    blocks = block_upsample(grid, factor)
    blocks = np.clip(blocks, 0.0, 1.0)
    out = np.zeros((*blocks.shape, 3), dtype=np.uint8)
    out[..., 0] = (blocks * 255).astype(np.uint8)
    out[..., 1] = (blocks * 64).astype(np.uint8)
    out[..., 2] = (blocks * 64).astype(np.uint8)
    return out


def rater_multiset(row: dict) -> list[int]:
    """The five raters' scores as a MULTISET, from the grade fractions.

    ``soft_k`` is the fraction of raters at grade ``k``
    (``ldl.SOFT_COLUMNS``); five raters means every ``5 * soft_k`` is an
    integer, and that is ASSERTED rather than assumed -- a manifest with a
    different rater count must refuse here, not render five invented dots.
    """
    counts = []
    for k in range(1, 6):
        value = 5.0 * float(row[f"soft_{k}"])
        if abs(value - round(value)) > 1e-6:
            raise Phase8cError(
                f"soft_{k} = {row[f'soft_{k}']} is not a fifth: 5*soft_k = "
                f"{value:.4f} is not an integer, so these are not five "
                "raters' grade fractions and the multiset cannot be "
                "recovered. The panel refuses rather than inventing raters."
            )
        counts.append(int(round(value)))
    if sum(counts) != 5:
        raise Phase8cError(
            f"the grade counts {counts} sum to {sum(counts)}, not 5; the "
            "soft columns do not describe five raters"
        )
    scores = [grade for grade, count in enumerate(counts, start=1)
              for _ in range(count)]
    return scores


def label_strip(width: int, lines: list[str], *, height_per_line: int = 14
                ) -> np.ndarray:
    """Text as pixels via Pillow's built-in bitmap font. (H, W, 3) uint8."""
    from PIL import Image, ImageDraw

    height = height_per_line * max(1, len(lines)) + 6
    image = Image.new("RGB", (width, height), (20, 20, 20))
    draw = ImageDraw.Draw(image)
    for index, line in enumerate(lines):
        draw.text((4, 3 + index * height_per_line), line, fill=(230, 230, 230))
    return np.asarray(image, dtype=np.uint8)


def stack_panels(panels: list[np.ndarray], gap: int = 4) -> np.ndarray:
    """Vertical stack, width-padded, uint8."""
    width = max(p.shape[1] for p in panels)
    rows = []
    for panel in panels:
        if panel.shape[1] < width:
            pad = np.zeros(
                (panel.shape[0], width - panel.shape[1], 3), dtype=np.uint8
            )
            panel = np.concatenate([panel, pad], axis=1)
        rows.append(panel)
        rows.append(np.zeros((gap, width, 3), dtype=np.uint8))
    return np.concatenate(rows[:-1], axis=0)


def side_by_side(panels: list[np.ndarray], gap: int = 4) -> np.ndarray:
    """Horizontal join, height-padded, uint8."""
    height = max(p.shape[0] for p in panels)
    columns = []
    for panel in panels:
        if panel.shape[0] < height:
            pad = np.zeros(
                (height - panel.shape[0], panel.shape[1], 3), dtype=np.uint8
            )
            panel = np.concatenate([panel, pad], axis=0)
        columns.append(panel)
        columns.append(np.zeros((height, gap, 3), dtype=np.uint8))
    return np.concatenate(columns[:-1], axis=1)


# --------------------------------------------------------------------------
# drawn panels (numpy/PIL only -- matplotlib is docker-only and unnecessary)
# --------------------------------------------------------------------------


def flatness_panel(weights_26: np.ndarray, *, width: int = 448,
                   height: int = 120, axis_max: float = 0.10) -> np.ndarray:
    """Panel 3: the 26 node weights as bars against the 1/27 line.

    **The flatness IS the picture.** The axis is FIXED at [0, axis_max] so
    uniform bars render visibly flat -- an auto-scaled axis would zoom into
    float dust and fabricate the ranking the dust verdict refuted. Nothing
    here sorts.
    """
    values = np.asarray(weights_26, dtype=np.float64)
    if values.shape != (26,):
        raise Phase8cError(f"expected 26 node weights, got {values.shape}")
    canvas = np.full((height, width, 3), 245, dtype=np.uint8)
    bar_width = width // 26
    for index, value in enumerate(values):
        top = height - int(np.clip(value / axis_max, 0, 1) * (height - 2))
        x0 = index * bar_width
        canvas[top:height, x0:x0 + bar_width - 2] = (70, 90, 160)
    uniform_y = height - int(np.clip((1 / 27) / axis_max, 0, 1) * (height - 2))
    canvas[max(0, uniform_y - 1):uniform_y + 1, :] = (200, 60, 60)
    return canvas


def prediction_panel(scores: list[int], predictions: list[float], *,
                     width: int = 448, height: int = 90) -> np.ndarray:
    """Panel 4: five rater dots (grade multiset), their mean, and the
    model's five-seed predictions on the 1..5 axis."""
    canvas = np.full((height, width, 3), 245, dtype=np.uint8)

    def x_of(grade: float) -> int:
        return int((float(grade) - 1.0) / 4.0 * (width - 20)) + 10

    axis_y = height // 2
    canvas[axis_y:axis_y + 1, 10:width - 10] = (120, 120, 120)
    for grade in range(1, 6):
        gx = x_of(grade)
        canvas[axis_y - 4:axis_y + 5, gx:gx + 1] = (120, 120, 120)
    # Raters above the axis; stacked when several share a grade.
    seen: dict[int, int] = {}
    for score in scores:
        offset = seen.get(score, 0)
        seen[score] = offset + 1
        gx, gy = x_of(score), axis_y - 10 - offset * 8
        canvas[gy - 3:gy + 3, gx - 3:gx + 3] = (40, 120, 60)
    mean_x = x_of(float(np.mean(scores)))
    canvas[axis_y - 26:axis_y + 27, mean_x:mean_x + 1] = (40, 120, 60)
    # Model below: five seeds as small marks, their mean as a tall mark.
    for prediction in predictions:
        px = x_of(np.clip(prediction, 1.0, 5.0))
        canvas[axis_y + 8:axis_y + 14, px - 1:px + 1] = (160, 70, 40)
    model_x = x_of(np.clip(float(np.mean(predictions)), 1.0, 5.0))
    canvas[axis_y + 6:axis_y + 30, model_x:model_x + 1] = (160, 70, 40)
    # [2026-08-15, the eye review] The grade scale, numbered 1..5.
    return axis_ticks(
        canvas, [x_of(g) for g in range(1, 6)],
        [str(g) for g in range(1, 6)],
    )


def curves_panel(curve_rows: list[dict], *, width: int = 448,
                 height: int = 130) -> np.ndarray:
    """Panel 5: inner-val MSE per epoch per fold, selected epochs implicit
    in where each polyline stops. Reads curves.csv rows; measures nothing."""
    canvas = np.full((height, width, 3), 245, dtype=np.uint8)
    if not curve_rows:
        raise Phase8cError("no curve rows; curves.csv was empty")
    # [FIXED 2026-08-15, the eye review's sawtooth] Grouped by (SERIES,
    # fold), not fold alone: five seed files share fold numbers, and
    # joining seed A's last epoch to seed B's first drew a connecting line
    # no training run ever took. Rows carry a "series" tag from the reader.
    by_fold: dict[tuple, list[dict]] = {}
    for row in curve_rows:
        by_fold.setdefault(
            (str(row.get("series", "")), int(row["fold"])), []
        ).append(row)
    max_epoch = max(int(r["epoch"]) for r in curve_rows)
    values = [float(r["inner_val_mse"]) for r in curve_rows]
    lo, hi = min(values), max(values)
    span = (hi - lo) or 1.0
    colours = [(70, 90, 160), (40, 120, 60), (160, 70, 40),
               (120, 60, 140), (60, 60, 60)]
    for (series, fold), rows in sorted(by_fold.items()):
        colour = colours[fold % len(colours)]
        points = sorted(rows, key=lambda r: int(r["epoch"]))
        previous = None
        for row in points:
            x = int((int(row["epoch"]) - 1) / max(1, max_epoch - 1)
                    * (width - 20)) + 10
            y = height - 10 - int(
                (float(row["inner_val_mse"]) - lo) / span * (height - 20)
            )
            if previous is not None:
                x0, y0 = previous
                steps = max(abs(x - x0), abs(y - y0), 1)
                for t in range(steps + 1):
                    xi = x0 + (x - x0) * t // steps
                    yi = y0 + (y - y0) * t // steps
                    canvas[
                        max(0, yi - 1):yi + 1, max(0, xi - 1):xi + 1
                    ] = colour
            previous = (x, y)
    return canvas


def cohort_scatter(model_errors: np.ndarray, rater_spreads: np.ndarray,
                   marked: np.ndarray, *, width: int = 640,
                   height: int = 420) -> np.ndarray:
    """The cohort page: |model error| (y) against rater spread (x), all 237,
    the fifteen sampled patients marked."""
    errors = np.asarray(model_errors, dtype=np.float64)
    spreads = np.asarray(rater_spreads, dtype=np.float64)
    flags = np.asarray(marked, dtype=bool)
    if not (len(errors) == len(spreads) == len(flags)):
        raise Phase8cError("scatter inputs disagree on length")
    canvas = np.full((height, width, 3), 245, dtype=np.uint8)
    x_hi = max(float(spreads.max()), 1e-6)
    y_hi = max(float(errors.max()), 1e-6)
    # [2026-08-15, the eye review] Jitter on the DISCRETE x (display only,
    # deterministic), rings for the fifteen, numerals on both axes.
    rng = np.random.default_rng(0)
    jitter = rng.uniform(-0.18, 0.18, size=len(spreads))
    for error, spread, offset, flag in zip(errors, spreads, jitter, flags):
        x = int((spread + offset) / x_hi * (width - 60)) + 40
        y = height - 30 - int(error / y_hi * (height - 60))
        if flag:
            canvas[y - 5:y + 5, x - 5:x + 5] = (160, 70, 40)
            canvas[y - 3:y + 3, x - 3:x + 3] = (245, 245, 245)
        else:
            canvas[y - 2:y + 2, x - 2:x + 2] = (90, 90, 150)
    tick_x, tick_labels = [], []
    for spread_value in sorted(set(int(s) for s in spreads)):
        tick_x.append(int(spread_value / x_hi * (width - 60)) + 40)
        tick_labels.append(str(spread_value))
    return axis_ticks(canvas, tick_x, tick_labels)


# --------------------------------------------------------------------------
# animation assembly
# --------------------------------------------------------------------------

#: Axis (a)'s registered epoch-1 checkpoints, by optimizer step.
A_EPOCH1_STEPS = (1, 2, 3, 5, 10, 20, 50)

def composite_heat(face: np.ndarray, grid: np.ndarray, *,
                   alpha: float = 0.55, factor: int = 16) -> np.ndarray:
    """The heat ON the face: alpha-blended, 16-px blocks preserved.

    **A Grad-CAM frame without its image is not a Grad-CAM frame** -- the
    eye review's first defect. The blend keeps the blocks (nearest
    upsample, no smoothing) and keeps the face visible under them: each
    pixel moves toward heat-red in proportion to alpha times its block's
    heat, so zero-heat regions ARE the photograph.
    """
    face = np.asarray(face, dtype=np.float64)
    if face.ndim != 3 or face.shape[2] != 3:
        raise Phase8cError(f"expected a (H, W, 3) face, got {face.shape}")
    heat = np.clip(block_upsample(grid, factor), 0.0, 1.0)
    if heat.shape != face.shape[:2]:
        raise Phase8cError(
            f"heat {heat.shape} does not cover the face {face.shape[:2]}; "
            "the grid and the photograph must share a frame"
        )
    weight = (alpha * heat)[..., None]
    red = np.array([255.0, 40.0, 40.0])
    return np.clip(
        face * (1.0 - weight) + red * weight, 0, 255
    ).astype(np.uint8)


def frame(face: np.ndarray, map_grid: np.ndarray | None, *,
          axis_label: str) -> np.ndarray:
    """One animation frame: THE FACE, with heat composited on it when a map
    exists, and the axis value beneath.

    ``map_grid=None`` is the before-state -- the photograph alone, which is
    the honest init frame (the model has not yet learned where to look) --
    never a black void. [REBUILT 2026-08-15: the first delivery drew heat
    on black beside the face and a void for init; the eye review rejected
    it. Verdict stamps and the verbatim caveat moved to the appendix page
    per CLINICAL_DELIVERY_REBUILT; the frame keeps the axis value.]
    """
    face = np.asarray(face, dtype=np.uint8)
    body = face.copy() if map_grid is None else composite_heat(face, map_grid)
    footer = label_strip(body.shape[1], [axis_label])
    return stack_panels([body, footer], gap=2)


def write_animation(frames: list[np.ndarray], path: Path, *,
                    duration_ms: int = 600) -> None:
    """APNG via Pillow, the frames as given -- assembly only, no resampling,
    no added frames. The caller writes the per-frame PNGs beside it."""
    from PIL import Image

    if not frames:
        raise Phase8cError("no frames; an empty animation is a broken axis")
    images = [Image.fromarray(f) for f in frames]
    images[0].save(
        path, format="PNG", save_all=True, append_images=images[1:],
        duration=duration_ms, loop=0, default_image=False,
    )



# --------------------------------------------------------------------------
# the SCUT animation captions (phase8.SCUT_ANIMATION_CAPTION_RULE)
# --------------------------------------------------------------------------


def captioned_frames(frames: list[np.ndarray], sentence: str) -> list[np.ndarray]:
    """Every frame with the registered recipe sentence beneath it.

    Applied at ASSEMBLY: the statement of what the animation is travels
    inside the animation file, while the per-frame PNGs on disk stay the
    measured objects, caption-less (the rule's own split)."""
    if not frames:
        raise Phase8cError("no frames; nothing to caption")
    out = []
    for image in frames:
        image = np.asarray(image, dtype=np.uint8)
        strip = label_strip(image.shape[1], [sentence])
        out.append(stack_panels([image, strip], gap=2))
    return out


def endpoint_comparison_panel(final_frame: np.ndarray,
                              shipped_frame: np.ndarray,
                              lines: list[str]) -> np.ndarray:
    """The trajectory's own final frame BESIDE the shipped-weights endpoint,
    caption beneath -- the seam shown, never smoothed into one map."""
    row = side_by_side([
        np.asarray(final_frame, dtype=np.uint8),
        np.asarray(shipped_frame, dtype=np.uint8),
    ])
    return stack_panels([row, label_strip(row.shape[1], list(lines))], gap=4)


# --------------------------------------------------------------------------
# the t-SNE figure (phase8.TSNE, phase8.TSNE_COMPANION)
# --------------------------------------------------------------------------

#: One fixed colour per class3 value, in sorted class order: low / mid /
#: high grade bins. Fixed here so two renders of the same data cannot
#: disagree about who is which.
TSNE_CLASS_COLOURS = (
    (60, 120, 216),
    (130, 130, 130),
    (216, 130, 40),
)


def tsne_scatter(points: np.ndarray, classes: np.ndarray, *,
                 title: str, side: int = 420) -> np.ndarray:
    """One t-SNE panel: the 2-D points coloured by class, title above.

    Coordinates are normalised into the panel with a fixed margin; the
    axes carry NO ticks deliberately -- t-SNE distances have no unit, and
    an axis with numbers on it would invite reading one."""
    points = np.asarray(points, dtype=np.float64)
    classes = np.asarray(classes)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) != len(classes):
        raise Phase8cError(
            f"expected (n, 2) points aligned with classes, got "
            f"{points.shape} vs {len(classes)}"
        )
    panel = np.full((side, side, 3), 255, dtype=np.uint8)
    margin = int(side * 0.06)
    spans = points.max(axis=0) - points.min(axis=0)
    spans[spans == 0] = 1.0
    scaled = (points - points.min(axis=0)) / spans
    pixels = margin + (scaled * (side - 2 * margin - 1)).astype(int)
    order = sorted(set(int(c) for c in classes))
    colour_of = {
        value: TSNE_CLASS_COLOURS[i % len(TSNE_CLASS_COLOURS)]
        for i, value in enumerate(order)
    }
    half = 2
    for (x, y), value in zip(pixels, classes):
        row, column = side - 1 - int(y), int(x)
        panel[
            max(row - half, 0):row + half + 1,
            max(column - half, 0):column + half + 1,
        ] = colour_of[int(value)]
    header = label_strip(side, [title])
    return stack_panels([header, panel], gap=2)


def tsne_figure(embedded_by_perplexity: dict, classes: np.ndarray,
                companion: dict) -> np.ndarray:
    """The registered t-SNE figure: every perplexity side by side, the
    companion PRINTED BENEATH -- and refused without it.

    The display rule is structural here, not procedural: this is the only
    function that composes the separation figure, and it raises when the
    companion is missing a field, so a figure without its number beside it
    cannot exist (phase8.TSNE_COMPANION's display_rule)."""
    required = (
        "k", "accuracy", "interval", "chance", "majority",
        "class_counts", "n_boot",
    )
    missing = [key for key in required if key not in companion]
    if missing:
        raise Phase8cError(
            f"the separation figure may not exist without its companion; "
            f"missing {missing}. The display rule is registered "
            "(phase8.TSNE_COMPANION) and this refusal is it."
        )
    if not embedded_by_perplexity:
        raise Phase8cError("no embeddings; nothing to draw")
    panels = [
        tsne_scatter(
            points, classes,
            title=f"t-SNE, perplexity {perplexity} -- a figure, never evidence",
        )
        for perplexity, points in sorted(embedded_by_perplexity.items())
    ]
    row = side_by_side(panels)
    low, high = companion["interval"]
    counts = tuple(int(c) for c in companion["class_counts"])
    order = sorted(set(int(c) for c in classes))
    legend = ", ".join(
        f"class {value} n={count}"
        for value, count in zip(order, counts)
    )
    footer = label_strip(row.shape[1], [
        (
            f"companion: leave-one-out k-NN (k={companion['k']}) accuracy "
            f"{companion['accuracy']:.3f} in the EMBEDDING space, 95% "
            f"bootstrap interval [{low:.3f}, {high:.3f}] over patients "
            f"({companion['n_boot']:,} draws)"
        ),
        (
            f"baselines: chance {companion['chance']:.3f} (uniform over "
            f"{len(counts)} classes), majority class "
            f"{companion['majority']:.3f} -- the accuracy is unreadable "
            "without this bar"
        ),
        f"colours by class3 ({legend}); axes carry no units by design",
    ])
    return stack_panels([row, footer], gap=4)


# --------------------------------------------------------------------------
# the clinical pages (CLINICAL_DELIVERY_REBUILT)
# --------------------------------------------------------------------------


def png_data_uri(image: np.ndarray) -> str:
    """A PNG as a data URI, so every page is self-contained."""
    import base64
    import io

    from PIL import Image

    buffer = io.BytesIO()
    Image.fromarray(np.asarray(image, dtype=np.uint8)).save(buffer, "PNG")
    return "data:image/png;base64," + base64.b64encode(
        buffer.getvalue()
    ).decode("ascii")


def file_data_uri(path: Path) -> str:
    """An existing PNG/APNG file embedded as-is -- an <img>-embedded APNG
    animates in any browser page, which is the format decision's point."""
    import base64

    return "data:image/png;base64," + base64.b64encode(
        Path(path).read_bytes()
    ).decode("ascii")


def axis_ticks(canvas: np.ndarray, x_positions: list[int],
               labels: list[str]) -> np.ndarray:
    """Tick numerals under a drawn axis, via the PIL bitmap font."""
    from PIL import Image, ImageDraw

    image = Image.fromarray(canvas)
    draw = ImageDraw.Draw(image)
    for x, text in zip(x_positions, labels):
        draw.text((x - 3, canvas.shape[0] - 14), text, fill=(60, 60, 60))
    return np.asarray(image, dtype=np.uint8)


_PAGE_STYLE = (
    "body { font-family: system-ui, sans-serif; max-width: 860px; "
    "margin: 2em auto; color: #222; line-height: 1.5; padding: 0 1em; }\n"
    "img { max-width: 100%; height: auto; border: 1px solid #ddd; }\n"
    "figure { margin: 1.5em 0; }\n"
    "figcaption, p.plain { font-size: 1.0em; color: #333; }\n"
    "details { margin: 1em 0; } summary { cursor: pointer; color: #555; }\n"
    ".caveat { background: #f6f3e8; padding: 0.8em 1em; "
    "border-left: 4px solid #c9b458; margin: 1.5em 0; }\n"
    "blockquote { background: #f4f4f4; padding: 0.8em 1em; "
    "border-left: 4px solid #888; font-family: monospace; "
    "font-size: 0.9em; overflow-wrap: anywhere; }\n"
    "nav { margin-bottom: 1.5em; } nav a { margin-right: 1.5em; }\n"
    "li { margin: 0.3em 0; }\n"
)


def _page(title: str, body: str) -> str:
    return (
        "<!DOCTYPE html>\n"
        '<html lang="en"><head><meta charset="utf-8">\n'
        f"<title>{title}</title>\n"
        f"<style>\n{_PAGE_STYLE}</style></head><body>\n"
        f"{body}\n</body></html>\n"
    )


def clinical_page(*, patient: int, hero_uri: str, train_anim_uri: str,
                  walk_anim_uri: str, depth_anim_uri: str,
                  seed_strip_uri: str, regions_uri: str,
                  prediction_uri: str, prediction_sentence: str,
                  appendix_href: str, index_href: str) -> str:
    """One self-contained clinical page. Plain language first; every piece
    of provenance machinery lives on the linked appendix."""
    body = f"""<nav><a href="{index_href}">&larr; all patients</a>
<a href="{appendix_href}">technical appendix</a></nav>
<h1>Patient {patient}</h1>

<figure>
<img src="{train_anim_uri}" alt="training animation: heat over the face">
<figcaption><strong>Where the model learned to look.</strong>
Each frame is one training step; the model is learning where to look.
The first frame is the photograph before any training; the heat that
follows shows which parts of the face influence the model's score at
that step.</figcaption>
</figure>

<figure>
<img src="{hero_uri}" alt="final attention over the face">
<figcaption><strong>Where the trained model looks.</strong> The red
regions influenced this patient's score the most. The blocks are the
model's true resolution&mdash;finer detail would be decoration, not
information.</figcaption>
</figure>

<figure>
<img src="{regions_uri}" alt="the 22 cleft-anatomy regions">
<figcaption><strong>The 22 cleft-anatomy regions</strong> (of the
27-patch frozen scheme) that this project's region-based analyses
use, drawn on this patient's photograph.</figcaption>
</figure>

<figure>
<img src="{prediction_uri}" alt="model prediction beside the five raters">
<figcaption><strong>The score, in context.</strong>
{prediction_sentence}</figcaption>
</figure>

<div class="caveat"><strong>What this model is:</strong> {CLINICAL_CAVEAT}.
</div>

<details><summary>What happens when the model is deliberately broken
(robustness check)</summary>
<img src="{walk_anim_uri}" alt="randomisation walk animation">
<p class="plain">Each frame progressively scrambles the model's internal
wiring. The heat pattern should fall apart&mdash;if it did not, the
pattern would not really depend on what the model learned.</p></details>

<details><summary>The same face at different depths of the network</summary>
<img src="{depth_anim_uri}" alt="depth sweep animation">
<p class="plain">Earlier layers see edges and textures; the assessed
layer is the last frame. These are shown for completeness, not as
alternative answers.</p></details>

<details><summary>Five independent training runs (unordered)</summary>
<img src="{seed_strip_uri}" alt="seed strip">
<p class="plain">The model was trained five times from different random
starting points; each panel is one run's final attention.</p></details>

<p><a href="{appendix_href}">Technical appendix</a>: verification
verdicts, refusal counts, and the full provenance for every figure on
this page.</p>"""
    return _page(f"Patient {patient} — how the model looked and scored", body)


def appendix_page(*, caveat_verbatim: str, original_stamp: str,
                  variant_stamp: str, both_numbers_sentence: str,
                  refusal_summary: str, flatness_uri: str,
                  curves_uri: str, scatter_uri: str,
                  index_href: str) -> str:
    """The technical appendix: every registered stamp and ledger, verbatim,
    travelling BEHIND the clinical pages -- they must exist, and they must
    not be the visible text a clinician meets first."""
    body = f"""<nav><a href="{index_href}">&larr; index</a></nav>
<h1>Technical appendix</h1>
<p><strong>No map on any page of this deliverable is a published
claim.</strong> The verdict stamps below are the registered gates'
outcomes and travel with every map they govern.</p>
<h2>Verdict stamps</h2>
<blockquote>original Grad-CAM: {original_stamp}</blockquote>
<blockquote>softmax variant: {variant_stamp}</blockquote>
<h2>The frozen-backbone caveat (verbatim, from the artifact)</h2>
<blockquote>{caveat_verbatim}</blockquote>
<h2>The variant's refusal ledger</h2>
<blockquote>{both_numbers_sentence}</blockquote>
<p>{refusal_summary}</p>
<h2>Node weights: the flatness finding</h2>
<img src="{flatness_uri}" alt="node weight flatness panel">
<p>SR-GNN's 26 grid-region weights (not the anatomy set) are uniform at
1/27 for every patient (median across-region sd 4.4e-05); the red line
is exact uniformity. The flatness is the finding: the graph model's
attention never produced a ranking, where Grad-CAM did.</p>
<h2>Training curves (per seed, per fold — no cross-fold joins)</h2>
<img src="{curves_uri}" alt="training curves">
<h2>Cohort: model error vs rater disagreement</h2>
<img src="{scatter_uri}" alt="cohort scatter">"""
    return _page("Technical appendix — Phase 8c", body)


def index_page(*, patients: list[int], scatter_uri: str,
               appendix_href: str) -> str:
    """The single entry point: fifteen patient pages, the appendix, and the
    cohort scatter with its axes labelled."""
    links = "\n".join(
        f'<li><a href="8c_patient_{p}.html">Patient {p}</a></li>'
        for p in patients
    )
    body = f"""<h1>How the model trained and predicted</h1>
<p>Fifteen patients from the study cohort, one page each: where the
model looked, how that changed as it trained, and how its score sits
beside the five raters' scores. The panel is a cleft patient, an
orthodontist, a speech and language therapist, a plastic surgeon and a
psychologist -- five different backgrounds, not five clinicians. All
faces and figures stay on this system.</p>
<ul>{links}</ul>
<p><a href="{appendix_href}">Technical appendix</a></p>
<h2>The whole cohort at a glance</h2>
<img src="{scatter_uri}" alt="cohort scatter">
<p>Each dot is one of the 237 patients: how far the model's score was
from the raters' mean (vertical) against how much the five raters
disagreed with each other (horizontal, jittered for visibility). The
fifteen patients with full pages are ringed in red.</p>"""
    return _page("How the model trained and predicted — cohort index", body)
