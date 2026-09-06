"""Road B Phase 2's two tasks: stage one setting, and gate the ordering.

Kept beside ``roadb_staging`` rather than in ``run.py`` because they are one
phase's wiring and ``run.py`` already carries every other phase's. ``run.py``
imports them into ``TASKS``, so the registry stays the single place a task
kind is resolved.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import roadb, roadb_staging


def stage(ctx) -> None:
    """Stage ONE setting into its OWN artifact.

    **One run per setting** (``roadb.ARTIFACT_PER_SETTING``), so a partial
    re-run is one job and moves only that hash.

    **The square cells reuse frozen code at a new argument, which is cheap
    without being free of risk.** ``staging.stage`` takes ``size`` and has only
    ever run at 224; ``resize_nearest`` at 512 and 768 is an untested regime,
    and the pixel residual measures exactly that.
    ``roadb_staging.FROZEN_CODE_NEW_REGIME``.

    **The artifact carries its OWN residual and content fraction**, not a
    shared summary -- a shared one goes stale on a partial re-run, which is
    the coupling the artifact split removed.

    **[CORRECTED 2026-08-09] Four defects sat here, each reachable only by a
    run, each now pinned by an end-to-end test:**

    * **G2 was never unwarped** -- a g2 artifact's pixels were its g1 twin's,
      so five settings were aliases of the other five. G2 is baked at staging
      exactly as Road A's frozen composition does it (``stage_build``: stage,
      then unwarp), at the new size.
    * **Non-square arrays were stacked**, and they cannot be
      (``roadb_staging.SHAPE_IS_FIXED_AT_STAGING``): shapes differ per
      patient, so ``np.stack`` raised on 6 of the 10 settings. Storage is per
      patient, as decided; square cells keep Road A's stacked layout so
      frozen readers consume them unchanged.
    * **The content fraction averaged over a square canvas the non-square
      artifact does not contain**, and it read patient 0 only. It is now per
      patient over the frame that ships, with the median recorded -- and for
      non-square settings asserted against the axis's own expectation.
    * **The residual was measured at the square column count for every
      setting.** It tracks column count (``roadb.RESIDUAL_FAMILIES``), so the
      non-square residual is measured at the artifact's own median staged
      width and the columns are recorded beside it.

    **And a fifth, found by the sheet REVIEW rather than by a test** -- the
    upsampling flag compared the source's shorter side where staging scales
    by the longer, over-counting interpolation; the flagged patients looked
    sharp and the review question fired exactly as written.
    ``roadb.UPSAMPLING_FLAG_MEASURED_THE_WRONG_SIDE``.
    """
    from .data.manifest import load_manifest
    from .geometry import staging
    from .geometry.trapezium import unwarp
    from .run import as_builtin

    task = ctx.config["task"]
    declared = {entry["name"]: Path(entry["path"]) for entry in ctx.inputs}
    settings = {s["name"]: s for s in roadb.staging_settings()}
    setting = settings.get(task["setting"])
    if setting is None:
        raise ValueError(
            f"{task['setting']!r} is not a Road B staging setting; the axis "
            f"is {sorted(settings)}"
        )
    if task["out_version"] != setting["artifact"]:
        raise ValueError(
            f"out_version {task['out_version']!r} is not this setting's "
            f"artifact {setting['artifact']!r}; they must agree or a partial "
            "re-run moves the wrong hash"
        )

    artifact = ctx.repo_root / "data" / "staged" / task["out_version"]
    if artifact.exists():
        raise ValueError(
            f"{artifact} already exists. Data artifacts are immutable: bump "
            "the version rather than modifying one results may cite."
        )

    size = setting["resolution"]
    geometry = setting["geometry"]
    rows = load_manifest(declared[task["manifest_artifact"]] / "manifest.csv")
    folders = declared[task["patient_folders"]]
    ctx.log(
        f"SETTING {setting['name']}: {size}px {setting['aspect']} "
        f"{geometry}, {len(rows)} patients -> {task['out_version']}"
    )

    images, geometry_rows, longer_sides, fractions = [], [], [], []
    for row in rows:
        source = source_image(folders, row)
        # The LONGER side, because that is the side staging scales by --
        # scale = target / max(w, h) on both paths -- so it is the side
        # interpolation follows. The shorter side was the first version's
        # mistake (roadb.UPSAMPLING_FLAG_MEASURED_THE_WRONG_SIDE).
        longer_sides.append(int(max(source.shape[0], source.shape[1])))
        staged = (
            staging.stage(source, size=size) if setting["aspect"] == "square"
            else roadb_staging.stage_non_square(source, size)
        )
        # G2 is baked at staging, as the frozen composition does it
        # (stage_build: stage, then unwarp) -- at the new size and, for
        # non-square, on the padless frame, where the content box is the
        # frame and unwarp therefore reaches zero non-content.
        image = staged.image if geometry == "g1" else unwarp(staged.image)
        images.append(image)
        # Per patient, over the FRAME the backbone sees. Square fractions
        # vary with each patient's aspect ratio; non-square ones are the
        # geometry's own constant, which is what makes them assertable.
        fractions.append(roadb_staging.content_fraction(
            image.shape[:2], geometry, staged.content_box
        ))
        x, y, w, h = staged.content_box
        geometry_rows.append({
            "patient_id": int(row["patient_id"]),
            "frontal_id": row.get("frontal_id", ""),
            "content_x": int(x), "content_y": int(y),
            "content_w": int(w), "content_h": int(h),
            "pad_fraction": round(float(staged.pad_fraction), 6),
        })

    flags = roadb_staging.upsampling_flags(longer_sides, size)
    roadb_staging.assert_upsampling_counts(flags)

    # The residual tracks COLUMN COUNT (roadb.RESIDUAL_FAMILIES), so it is
    # measured at this artifact's own median staged width -- for square cells
    # that is the nominal size, for non-square the rounded content width.
    columns = int(np.median([image.shape[1] for image in images]))
    residual = roadb_staging.pixel_asymmetry_residual(columns)

    fraction = float(np.median(fractions))
    if setting["expected_non_content"] is not None:
        expected = 1.0 - setting["expected_non_content"]
        if abs(fraction - expected) > 0.02:
            raise ValueError(
                f"content fraction {fraction:.4f} against the axis's "
                f"{expected:.4f} -- beyond edge quantisation, so the frame "
                "or the mask is not what the setting says"
            )
    ctx.log(
        f"  residual {residual:.3e} at {columns} columns  "
        f"content fraction {fraction:.4f}  "
        f"upsampled {flags['n_upsampled']}/{flags['n']}"
    )

    manifest = {
        "setting": setting,
        "pixel_residual": residual,
        "residual_columns": columns,
        "content_fraction": fraction,
        "content_fraction_range": [
            float(min(fractions)), float(max(fractions)),
        ],
        "upsampling": flags,
        "n_patients": len(rows),
        "storage": (
            "stacked: staged_patient_<geometry>.npy, (N, H, W, 3), row "
            "order = manifest order -- Road A's layout, so frozen readers "
            "consume it at a new size unchanged"
            if setting["aspect"] == "square" else
            "per patient: staged_patient_<geometry>_p<patient_id>.npy -- "
            "shapes differ per patient and cannot stack "
            "(roadb_staging.SHAPE_IS_FIXED_AT_STAGING)"
        ),
        "g2_pixels": (
            None if geometry == "g1" else
            "unwarp(staged) at this size -- Road A's frozen composition "
            "(stage_build); geometry.csv describes the PRE-unwarp staging, "
            "as Road A's does"
        ),
        "clean_pad_pair": clean_pad_pair(setting),
        "gate_note": roadb_staging.FROZEN_CODE_NEW_REGIME["not_free_of_risk"],
    }

    artifact.mkdir(parents=True)
    if setting["aspect"] == "square":
        np.save(
            artifact / f"staged_patient_{geometry}.npy", np.stack(images)
        )
    else:
        for entry, image in zip(geometry_rows, images):
            np.save(
                artifact
                / f"staged_patient_{geometry}_p{entry['patient_id']:03d}.npy",
                image,
            )
    write_geometry_csv(artifact / "geometry.csv", geometry_rows)
    (artifact / "MANIFEST.json").write_text(
        json.dumps(as_builtin(manifest), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with ctx.atomic("metrics.json", tier="SHAREABLE") as tmp:
        tmp.write_text(
            json.dumps(as_builtin(manifest), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def clean_pad_pair(setting: dict) -> dict | None:
    """**The one-factor pad comparison, recorded so a later phase finds it.**

    Road A's ``staged_v1`` against a 224 non-square cell differs only by the
    pad: ``stage`` pads and THEN resizes, so the content pixel box is
    identical at the same target. Every 512 comparison confounds pad removal
    with resolution; this pair does not.

    It is a Phase 4 or 7 arm rather than a staging output. This records that
    it exists and what it isolates.
    """
    if setting["resolution"] != roadb.CONTROL_RESOLUTION:
        return None
    return {
        "against": "staged_v1 (Road A, 224 square)",
        "differs_by": "the pad, and nothing else",
        "isolates": (
            "Branch 1's pad claim at constant content resolution -- stage "
            "pads THEN resizes, so the content pixel box is identical"
        ),
        "not_a_staging_output": "a Phase 4 or 7 arm consumes this pair",
    }


def source_image(folders: Path, row: dict):
    """The source image for one manifest row.

    **[CORRECTED 2026-08-09] This built its own path and built it wrong.** The
    layout is ``<root>/<patient_id>/<frontal_id>.jpg`` and it joined the root
    straight to ``frontal_id``, skipping the patient folder. Both inputs
    resolved and matched, so guard 3 was satisfied and only the join was
    wrong.

    **It is now a delegation, because the resolver already existed.**
    ``contact.find_image`` is what ``stage_build`` uses, and a fourth path
    builder would have had to rediscover everything it already carries:

    * AppleDouble sidecars ``._<name>.jpg``, excluded by basename rather than
      extension -- and ``Thumbs.db`` / ``._Thumbs.db``, which regenerate on
      Windows browsing so they must stay excluded from input hash walks too;
    * folder 143's exception, where 524 is the frontal rather than 523;
    * folder 238's single image.

    It matches on ``str(image_id) in path.stem`` rather than an exact name,
    which is what tolerates all three without a special case per patient.
    """
    from .geometry.contact import find_image
    from .geometry.render import load_image

    return load_image(
        find_image(folders, int(row["patient_id"]), int(row["frontal_id"]))
    )


def write_geometry_csv(path: Path, rows: list[dict]) -> None:
    """``geometry.csv`` with the tier marker Road A's readers expect.

    The marker is what ``cluster_csv.read_cluster_csv`` skips, and it has cost
    this project three rounds. Written by the same convention so
    ``phase3.load_geometry_rows`` reads a Road B artifact unchanged.
    """
    columns = list(rows[0])
    lines = ["# CLUSTER-ONLY: patient-keyed geometry", ",".join(columns)]
    lines += [",".join(str(row[column]) for column in columns) for row in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def residual_gate(ctx) -> None:
    """The per-family residual gate, over artifacts it does NOT own.

    **A separate task because no single staging run sees all three
    resolutions.** It reads each artifact's recorded residual plus Road A's
    224 from ``roadb.ROAD_A_224_PIXEL_RESIDUAL``, and evaluates the ordering
    **within each (aspect, geometry) family** -- the residual tracks column
    count, so a pooled ordering would call non-square a defect for having
    fewer (``roadb.RESIDUAL_FAMILIES``).

    **It records the rollups it read.** It reads artifacts it does not own, so
    a re-run after the verdict leaves the verdict silently wrong. Recording
    them lets a later reader tell whether the verdict still describes what is
    on disk -- the same reason run directories carry input hashes.

    **[CORRECTED 2026-08-09] It read ``entry["rollup_sha256"]`` and died on
    the cluster with a KeyError -- that is the CONFIG's field name.**
    ``RunContext._verify_inputs`` resolves each declared input into
    ``declared_rollup`` (what the config said) and ``rollup`` (what the
    directory actually hashed to). Guard 3 had just verified all ten
    artifacts; the run failed reading back the hashes it had checked. It
    records ``rollup`` -- the measured one -- and the input-entry key sweep
    in ``test_workflow_hygiene`` now closes this class for every ctx handler.
    """
    from .run import as_builtin

    declared = {entry["name"]: entry for entry in ctx.inputs}
    settings = {s["name"]: s for s in roadb.staging_settings()}
    by_artifact = {s["artifact"]: name for name, s in settings.items()}

    observed, rollups = {}, {}
    for name, entry in sorted(declared.items()):
        setting_name = by_artifact.get(name)
        if setting_name is None:
            continue
        manifest = json.loads(
            (Path(entry["path"]) / "MANIFEST.json").read_text(encoding="utf-8")
        )
        observed[setting_name] = float(manifest["pixel_residual"])
        # "rollup" is what the directory hashed to, verified by guard 3;
        # "rollup_sha256" is the config field and does not exist here.
        rollups[name] = entry["rollup"]
        ctx.log(f"  {setting_name}: residual {observed[setting_name]:.3e}")

    absent = sorted(set(settings) - set(observed))
    if absent:
        raise ValueError(
            f"{len(absent)} settings not declared: {absent}. The gate is an "
            "ordering within a family, so a missing point silently shortens "
            "a family rather than failing"
        )

    report = roadb_staging.assert_residual_monotone_by_family(observed)
    ctx.log(
        f"GATE PASSED: {report['n_families']} families monotone; "
        f"{len(report['three_point_families'])} have three points"
    )

    summary = {
        **report,
        "artifacts_read": rollups,
        "road_a_224": roadb.ROAD_A_224_PIXEL_RESIDUAL,
        # A reader of this verdict meets four pairs of g1/g2 residuals that
        # are identical to full precision -- the aliasing defect's numeric
        # signature -- so the reason they are EXPECTED to be identical
        # travels with the verdict.
        "residual_geometry_note": roadb.RESIDUAL_IS_GEOMETRY_BLIND,
        "verdict_describes": (
            "the artifacts whose rollups are recorded above. A re-run of any "
            "of them makes this verdict stale, and comparing the rollups is "
            "how a reader can tell"
        ),
    }
    with ctx.atomic("metrics.json", tier="SHAREABLE") as tmp:
        tmp.write_text(
            json.dumps(as_builtin(summary), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def swin_build_gate(ctx) -> None:
    """Road B Phase 6, gate 1: swin built AT 512 and 768, on the pinned image.

    **The one gate that could refute the four-backbone conclusion, so it
    runs first, and it is short** (``roadb.PHASE_6_STRUCTURE`` build order).
    Measured at timm 1.0.27: building at ``img_size`` carries the 224
    checkpoint's parameters unchanged (``roadb.SWIN_AT_RESOLUTION``). The
    pinned image has timm 1.0.7; this run is that confirmation.

    **The criterion is over PARAMETERS, not the full state dict.** Some timm
    versions register resolution-dependent BUFFERS on swin (shifted-window
    attention masks); buffers carry no learning and may legitimately differ
    across sizes and versions, so they are reported rather than gated --
    gating on them would refuse a pass over a quantity the claim is not
    about (R2).
    """
    import timm
    import torch

    from .models.factory import BACKBONES
    from .run import as_builtin

    task = ctx.config["task"]
    name = BACKBONES["swin_b"]["timm_name"]
    pretrained = bool(task.get("pretrained", True))
    sizes = [int(s) for s in task.get("sizes", (512, 768))]

    reference = timm.create_model(name, pretrained=pretrained, num_classes=0)
    reference_parameters = dict(reference.named_parameters())
    reference_buffers = dict(reference.named_buffers())
    ctx.log(
        f"GATE: {name} at {sizes}, pretrained={pretrained}, "
        f"timm {timm.__version__}"
    )

    report = {
        "timm_version": timm.__version__,
        "pretrained": pretrained,
        "criterion": (
            "parameter names and shapes identical to the 224 build, tensors "
            "equal when pretrained, forward produces the 1024-dim "
            "embedding. Buffers are reported, not gated"
        ),
        "sizes": {},
    }
    failures = []
    for size in sizes:
        model = timm.create_model(
            name, pretrained=pretrained, num_classes=0, img_size=size
        )
        parameters = dict(model.named_parameters())
        missing = sorted(set(reference_parameters) - set(parameters))
        extra = sorted(set(parameters) - set(reference_parameters))
        shape_mismatches = sorted(
            key for key in reference_parameters
            if key in parameters
            and reference_parameters[key].shape != parameters[key].shape
        )
        tensors_equal = None
        if pretrained and not (missing or extra or shape_mismatches):
            tensors_equal = all(
                torch.equal(reference_parameters[key], parameters[key])
                for key in reference_parameters
            )

        buffers = dict(model.named_buffers())
        buffer_name_diffs = sorted(set(buffers) ^ set(reference_buffers))
        buffer_shape_diffs = sorted(
            key for key in set(buffers) & set(reference_buffers)
            if buffers[key].shape != reference_buffers[key].shape
        )

        model.eval()
        with torch.no_grad():
            out = model(torch.zeros(1, 3, size, size))

        entry = {
            "parameter_names_identical": not missing and not extra,
            "n_shape_mismatches": len(shape_mismatches),
            "shape_mismatches": shape_mismatches[:8],
            "pretrained_tensors_equal": tensors_equal,
            "buffer_name_diffs": buffer_name_diffs[:8],
            "buffer_shape_diffs": buffer_shape_diffs[:8],
            "forward_shape": [int(v) for v in out.shape],
        }
        report["sizes"][size] = entry
        ok = (
            entry["parameter_names_identical"]
            and not shape_mismatches
            and tuple(out.shape) == (1, 1024)
            and tensors_equal in (None, True)
        )
        if not ok:
            failures.append(size)
        ctx.log(f"  {size}: {'OK' if ok else 'REFUSED'} {entry}")

    if failures:
        raise RuntimeError(
            f"the swin build gate REFUSED at {failures} on timm "
            f"{timm.__version__}: the 224 checkpoint does not carry "
            "unchanged, and the four-backbone conclusion "
            "(roadb.SWIN_AT_RESOLUTION) does not hold on the pinned image. "
            "The resolution axis loses swin at these sizes unless a "
            "different route is designed -- record that before building "
            "anything that assumes four backbones"
        )
    with ctx.atomic("metrics.json", tier="SHAREABLE") as tmp:
        tmp.write_text(
            json.dumps(as_builtin(report), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def graph_geometry_gate(ctx) -> None:
    """Road B Phase 6, gate 3: the graph backbones' region geometry at
    512 and 768. Measured, not reasoned about -- the same defect class the
    non-square probe found was fractions preserved, sampled regions not.

    **The two backbones have different geometry contracts, and the gate
    gates each on ITS OWN contract rather than one rule (R2):**

    * **SR-GNN**: 26 fixed-pixel boxes in a feature map upsampled to a
      FIXED 42x42 (``ROI_RESOLUTION``) plus the whole image -- region count
      and placement are input-size-invariant BY CONSTRUCTION, and the gate
      asserts the construction: 27 regions at every size, the box buffer
      untouched by input size. What input size changes is how interpolated
      that 42x42 map is: 6.0x from 224's 7x7, 1.75x from 768's 24x24 --
      the CleftGNN parallel, where upsampling the map compensates for
      coarseness, COMPOSES with feeding more pixels.
    * **AG-Net**: regions are IMAGE-SPACE -- SIFT keypoints, seeded GMM,
      normalised (x1, y1, x2, y2) boxes -- so their placement legitimately
      varies with resolution (detection is resolution-dependent). Gating on
      placement identity would refuse a property as a defect; the contract
      is the count law (``region_count``), normalised bounds in [0, 1], and
      the full forward producing an output at every size. Placement drift
      across sizes is REPORTED so the property is recorded, not gated.
    """
    import numpy as np_  # noqa: F401 -- torch path below
    import torch

    from .models import agnet, srgnn
    from .run import as_builtin

    task = ctx.config["task"]
    sizes = [int(s) for s in task.get("sizes", (224, 512, 768))]
    backbones = list(task.get("backbones", ("srgnn", "agnet")))
    batch = 2
    # CUDA when available, like extract_embeddings: production runs on GPU,
    # and torchvision wheels exist whose roi_align ships no CPU kernel --
    # measured on the laptop's cu126 build, where the CPU path raises
    # NotImplementedError while the CUDA path runs.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # One piece of content at every size, so AG-Net's cross-size box drift
    # measures RESOLUTION, not content. Deterministic; noise carries plenty
    # of SIFT keypoints.
    rng = np.random.default_rng(31337)
    base = rng.integers(0, 256, (768, 768, 3), dtype=np.uint8)

    def content_at(size: int) -> np.ndarray:
        from .geometry.staging import resize_nearest

        return base if size == 768 else resize_nearest(base, size, size)

    report = {"sizes": sizes, "backbones": {}, "device": str(device)}
    failures = []

    if "srgnn" in backbones:
        model = srgnn.build(pretrained=False, num_outputs=1)
        model.to(device).eval()
        boxes_before = model.default_boxes.clone()
        entry = {
            "contract": (
                "27 regions, fixed pixels in the 42x42 upsampled map, "
                "input-size-invariant by construction"
            ),
            "expected_regions": int(srgnn.N_REGIONS),
            "per_size": {},
        }
        for size in sizes:
            x = torch.zeros(batch, 3, size, size, device=device)
            with torch.no_grad():
                feat = model.backbone(x)[-1]
                out, region_w = model.forward_with_maps(x)
            per = {
                "feature_map": [int(v) for v in feat.shape[-2:]],
                "upsampled_to": int(srgnn.ROI_RESOLUTION),
                "map_interpolation_factor": round(
                    srgnn.ROI_RESOLUTION / int(feat.shape[-1]), 2
                ),
                "n_regions": int(region_w.shape[1]),
                "output_shape": [int(v) for v in out.shape],
            }
            entry["per_size"][size] = per
            if per["n_regions"] != srgnn.N_REGIONS:
                failures.append(f"srgnn@{size}: {per['n_regions']} regions")
            ctx.log(f"  srgnn {size}: map {per['feature_map']} -> 42x42 "
                    f"({per['map_interpolation_factor']}x), "
                    f"{per['n_regions']} regions")
        entry["boxes_input_size_invariant"] = bool(
            torch.equal(model.default_boxes, boxes_before)
        )
        if not entry["boxes_input_size_invariant"]:
            failures.append("srgnn: the box buffer moved with input size")
        report["backbones"]["srgnn"] = entry

    if "agnet" in backbones:
        # TWO quantities, named apart -- comparing them under one name cost
        # a review round (R2): generate_srs returns the SIFT+GMM boxes
        # WITHOUT the whole image, which pooling appends. On full-keypoint
        # content: 36 srs boxes -> 37 model regions, the law exactly. The
        # law is CONDITIONAL -- keypoint-starved content falls back
        # (measured: flat images give 1 srs box -> 2 regions) -- which is
        # why the deterministic noise here is asserted at full kappa.
        full_kappa_boxes = agnet.KAPPA * (agnet.KAPPA + 1) // 2
        entry = {
            "contract": (
                "image-space regions (SIFT + seeded GMM), normalised boxes "
                "in [0, 1]; srs boxes + the appended whole image = "
                "region_count on full-keypoint content; placement varies "
                "with resolution BY DESIGN and is reported, not gated"
            ),
            "srs_boxes_full_kappa": int(full_kappa_boxes),
            "model_regions_full_kappa": int(agnet.region_count()),
            "count_law_conditional_on": (
                "enough SIFT keypoints; starved content falls back to "
                "fewer regions, which the batch padding absorbs"
            ),
            "per_size": {},
        }
        boxes_by_size = {}
        for size in sizes:
            boxes = agnet.generate_srs(content_at(size)[:, :, ::-1])
            boxes_by_size[size] = boxes
            per = {
                "srs_boxes": int(boxes.shape[0]),
                "model_regions": int(boxes.shape[0]) + 1,
                "boxes_normalised": bool(
                    (boxes >= 0.0).all() and (boxes <= 1.0).all()
                ),
            }
            if not per["boxes_normalised"]:
                failures.append(f"agnet@{size}: boxes outside [0, 1]")
            # The gate's content is deterministic full-keypoint noise, so
            # the law applies EXACTLY here -- 36 and 37, not a bound.
            if per["srs_boxes"] != full_kappa_boxes:
                failures.append(
                    f"agnet@{size}: {per['srs_boxes']} srs boxes against "
                    f"the full-kappa {full_kappa_boxes} on full-keypoint "
                    "content"
                )
            entry["per_size"][size] = per
            ctx.log(f"  agnet {size}: {per['srs_boxes']} srs boxes -> "
                    f"{per['model_regions']} model regions, "
                    f"normalised={per['boxes_normalised']}")
        # The measured property: same content, different resolutions,
        # different boxes. Recorded so nobody later reads the drift as a bug.
        counts = {s: int(b.shape[0]) for s, b in boxes_by_size.items()}
        entry["placement_varies_with_resolution"] = {
            "counts_by_size": counts,
            "identical_across_sizes": bool(
                len({b.shape[0] for b in boxes_by_size.values()}) == 1
                and all(
                    np.allclose(boxes_by_size[sizes[0]], b)
                    for b in boxes_by_size.values()
                )
            ),
            "note": (
                "SIFT detection is resolution-dependent, so image-space "
                "regions move with input size -- a property of the design, "
                "not a defect. The count law and normalised bounds are the "
                "contract"
            ),
        }
        model = agnet.build(pretrained=False, num_outputs=1)
        model.to(device).eval()
        for size in sizes:
            image = torch.as_tensor(
                content_at(size).astype(np.float32) / 255.0
            ).permute(2, 0, 1).unsqueeze(0).repeat(batch, 1, 1, 1).to(device)
            with torch.no_grad():
                out = model(image)
            entry["per_size"][size]["forward_shape"] = [int(v) for v in out.shape]
            if out.shape != (batch, 1):
                failures.append(f"agnet@{size}: forward {tuple(out.shape)}")
            ctx.log(f"  agnet {size}: forward -> {tuple(out.shape)}")
        report["backbones"]["agnet"] = entry

    if failures:
        raise RuntimeError(
            f"the graph geometry gate REFUSED: {failures}. The resolution "
            "axis loses the failing backbone at the failing sizes "
            "(roadb.PHASE_6_STRUCTURE gate 3) -- record that before building "
            "anything that assumes four backbones"
        )
    with ctx.atomic("metrics.json", tier="SHAREABLE") as tmp:
        tmp.write_text(
            json.dumps(as_builtin(report), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def pretrain_build_gate(ctx) -> None:
    """Road B Phase 6, gate 2: the pretrain path builds every backbone at
    every size, and the optimiser sees the parameters.

    **What this gate covers, and what it deliberately does not -- a pass
    here must not be read as more than it is:**

    * **COVERED (laptop-reachable, confirmed on the pinned image by this
      config's run)**: construction of all four backbones THROUGH the
      pretrain path's own builder -- ``TorchPretrainModel.reset``, which
      derives ``input_size`` from the training features' own shape -- plus
      the optimiser registering every model parameter, and the trainable
      parameter count being IDENTICAL across sizes per backbone: resizing
      must add or drop nothing learned (ViT interpolates at forward, not
      at build; swin and the graphs carry no size-dependent parameters).
    * **NOT COVERED: whether training is STABLE at 512/768 with
      interpolated position grids.** That is Phase 6's first pretraining
      run, an experiment with its own record -- not a gate, and not
      inferable from construction succeeding.
    """
    import torch

    from .run import as_builtin
    from .train.pretrain import TorchPretrainModel

    task = ctx.config["task"]
    sizes = [int(s) for s in task.get("sizes", (224, 512, 768))]
    backbones = list(
        task.get("backbones", ("vit_b16", "swin_b", "srgnn", "agnet"))
    )
    pretrained = bool(task.get("pretrained", True))
    labels = np.full(4, 2.5, dtype=np.float32)

    report = {
        "pretrained": pretrained,
        "scope": (
            "construction + optimiser registration only; training "
            "stability at interpolated positions is Phase 6's first run"
        ),
        "backbones": {},
    }
    failures = []
    for backbone in backbones:
        per_size = {}
        counts = {}
        for size in sizes:
            features = np.zeros((4, size, size, 3), dtype=np.uint8)
            model = TorchPretrainModel(
                name=backbone, pretrained=pretrained, deterministic=False,
            )
            model.reset(features, labels)
            n_model = sum(p.numel() for p in model._model.parameters())
            n_optimizer = sum(
                p.numel()
                for group in model._optimizer.param_groups
                for p in group["params"]
            )
            counts[size] = n_model
            per_size[size] = {
                "parameters": int(n_model),
                "optimizer_parameters": int(n_optimizer),
                "optimizer_sees_all": bool(n_optimizer == n_model),
            }
            if n_optimizer != n_model or n_model == 0:
                failures.append(
                    f"{backbone}@{size}: optimiser sees {n_optimizer} of "
                    f"{n_model} parameters"
                )
            ctx.log(f"  {backbone} {size}: {n_model:,} params, "
                    f"optimiser sees all: {per_size[size]['optimizer_sees_all']}")
        if len(set(counts.values())) != 1:
            failures.append(
                f"{backbone}: parameter count varies with size {counts} -- "
                "resizing added or dropped something learned"
            )
        report["backbones"][backbone] = {
            "per_size": per_size,
            "parameters_size_invariant": len(set(counts.values())) == 1,
        }

    if failures:
        raise RuntimeError(
            f"the pretrain build gate REFUSED: {failures}. The Phase 6 "
            "configs must not be written against a builder that cannot "
            "construct their cells (roadb.PHASE_6_STRUCTURE gate 2)"
        )
    with ctx.atomic("metrics.json", tier="SHAREABLE") as tmp:
        tmp.write_text(
            json.dumps(as_builtin(report), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


#: Where each backbone's pretrained init actually comes from -- the upstream
#: identity the snapshot captures. The graph backbones snapshot their BODY
#: weights (the pretrained part); their assemblies' own heads are
#: seed-initialised at reset either way.
INIT_SOURCES = {
    "vit_b16": ("timm", "vit_base_patch16_224"),
    "swin_b": ("timm", "swin_base_patch4_window7_224"),
    "srgnn": ("timm", "legacy_xception"),
    "agnet": ("torchvision", "resnet50/IMAGENET1K_V2"),
    # Phase 7D [MEASURED 2026-08-15 against the pinned timm 1.0.7]: all
    # three resolve with pretrained weights. The default tags the snapshot
    # will capture: patch32 augreg_in21k_ft_in1k, patch8
    # augreg2_in21k_ft_in1k, mvitv2_base fb_in1k -- each recorded in the
    # artifact's MANIFEST.json by the task, so which recipe fed which arm is
    # never reconstructed from memory.
    "vit_b32": ("timm", "vit_base_patch32_224"),
    "vit_b8": ("timm", "vit_base_patch8_224"),
    "mvitv2_b": ("timm", "mvitv2_base"),
}


def snapshot_pretrained_init(ctx) -> None:
    """Freeze one backbone's pretrained init as a versioned, hashed artifact.

    **[2026-08-10] Until this task, the ImageNet init was the one input
    guard 3 never verified**: ``pretrained=True`` re-downloads at run time
    (HOME is ephemeral, PLAN 2.7), the bytes were never hashed, and when two
    ViT-512 runs at one SHA produced different checkpoints, NO recorded field
    could say whether their inits differed -- the unarbitrable comparison
    that demonstrated the hole
    (``roadb.PRETRAINED_INIT_IS_NOW_DECLARABLE``). Chain 3's shape: the
    weights that produced a result were not recoverable from the config.

    This run performs the LAST undeclared download, once, and captures:
    the weights (``init.npz``), the upstream identity (timm's
    ``pretrained_cfg`` or the torchvision weights enum), and the per-file
    hashes + rollup every consuming config declares. Road A's thirty
    pretraining runs predate it and share the hole -- recorded, not
    repaired.
    """
    import timm
    import torch

    from .provenance.hashing import hash_dir, hash_file
    from .run import as_builtin

    task = ctx.config["task"]
    backbone = task["backbone"]
    kind, name = INIT_SOURCES[backbone]

    artifact = ctx.repo_root / "data" / "inits" / f"init_{backbone}_v1"
    if artifact.exists():
        raise ValueError(
            f"{artifact} already exists. Data artifacts are immutable "
            "(PLAN 2.6): create a new version, never overwrite."
        )

    if kind == "timm":
        model = timm.create_model(name, pretrained=True, num_classes=0)
        upstream = {
            "loader": f"timm {timm.__version__}",
            "model": name,
            "pretrained_cfg": {
                key: str(value)
                for key, value in (model.pretrained_cfg.__dict__ or {}).items()
            } if hasattr(model.pretrained_cfg, "__dict__") else str(
                model.pretrained_cfg
            ),
        }
        state = model.state_dict()
    else:
        import torchvision
        from torchvision.models import ResNet50_Weights, resnet50

        weights = ResNet50_Weights.IMAGENET1K_V2
        model = resnet50(weights=weights)
        upstream = {
            "loader": f"torchvision {torchvision.__version__}",
            "model": "resnet50",
            "weights_enum": str(weights),
            "url": str(weights.url),
        }
        state = model.state_dict()

    arrays = {
        key: value.detach().cpu().numpy() for key, value in state.items()
    }
    artifact.mkdir(parents=True)
    np.savez(artifact / "init.npz", **arrays)

    manifest = {
        "artifact": f"inits/init_{backbone}_v1",
        "backbone": backbone,
        "upstream": upstream,
        "n_tensors": len(arrays),
        "generated_by": "cleft.run task snapshot_pretrained_init",
        "generating_run": ctx.run_dir.name,
        "generator_git_sha8": ctx.git.sha8,
        "torch_version": torch.__version__,
        "payload_files": {
            "init.npz": hash_file(artifact / "init.npz"),
        },
        "purpose": (
            "the pretrained init as a DECLARED input: consuming configs "
            "declare this artifact's rollup, guard 3 verifies it, and "
            "inputs.json records what the run started from"
        ),
    }
    (artifact / "MANIFEST.json").write_text(
        json.dumps(as_builtin(manifest), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    rollup = hash_dir(artifact)
    summary = {
        "backbone": backbone,
        "upstream": upstream,
        "n_tensors": len(arrays),
        "artifact": {
            "path": str(artifact),
            "rollup_sha256_for_configs": rollup["rollup"],
            "total_bytes": rollup["total_bytes"],
        },
    }
    ctx.log(
        f"init snapshot {backbone}: {len(arrays)} tensors, rollup "
        f"{rollup['rollup'][:16]}..."
    )
    with ctx.atomic("metrics.json", tier="SHAREABLE") as tmp:
        tmp.write_text(
            json.dumps(as_builtin(summary), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def load_crop(root, setting: dict, rows: list, patient_id: int):
    """One patient's staged array, from either storage layout.

    Square artifacts stack in manifest row order (Road A's layout); non-square
    store per patient because the shapes cannot stack
    (``roadb_staging.SHAPE_IS_FIXED_AT_STAGING``).
    """
    geometry = setting["geometry"]
    if setting["aspect"] == "square":
        stack = np.load(
            root / f"staged_patient_{geometry}.npy", mmap_mode="r"
        )
        for index, row in enumerate(rows):
            if int(row["patient_id"]) == patient_id:
                return np.asarray(stack[index])
        raise ValueError(
            f"patient {patient_id} is not in {root.name}'s geometry.csv"
        )
    return np.load(root / f"staged_patient_{geometry}_p{patient_id:03d}.npy")


def region_crop_extract(ctx) -> None:
    """Branch 3 criterion 2: 22 region crops per patient, embedded once.

    **No new torch code, deliberately.** The crops are flattened into ONE
    stack of ``n_patients * 22`` images and handed to
    ``extract.extract_features`` -- the same path every Road B set used, so
    model loading, eval/no_grad, the declared-init route, batching and the
    provenance report are reused rather than reimplemented. The result is
    reshaped to ``(n_patients, 22, feature_dim)`` and saved as
    ``region_vectors``.

    **Precompute-once.** Frozen backbone, head-only fits downstream, so this
    runs once per crop set and every arm reads it -- the brief's §2.2 cost
    sentence described the augmentation regime's live path, corrected there.

    **The region names travel with the array** (``regions=``), ordered, so a
    consumer asks which column is ``philtral_column`` instead of trusting
    position.
    """
    import json

    import numpy as np

    from . import embeddings, roadb, roadb_regioncrop
    from .geometry.staging import Staged
    from .models.factory import BACKBONES
    from .run import as_builtin
    from .train import extract as extract_module
    from .train.phase3 import load_geometry_rows

    task = ctx.config["task"]
    declared = {entry["name"]: entry for entry in ctx.inputs}
    root = Path(declared[task["staged_artifact"]]["path"])
    setting = [
        s for s in roadb.staging_settings()
        if s["artifact"] == task["staged_artifact"]
    ]
    if len(setting) != 1 or setting[0]["aspect"] != "nonsquare" or (
        setting[0]["geometry"] != "g2"
    ):
        raise ValueError(
            f"{task['staged_artifact']} is not the branch's non-square G2 "
            "staging (brief §2.1)"
        )
    setting = setting[0]

    layout = task["layout"]
    anatomy = roadb_regioncrop.protected_patches()
    if layout == "anatomy":
        patches = anatomy
    elif layout == "random":
        patches = roadb_regioncrop.random_patches(anatomy)
    else:
        # The whole-frame control, through this path rather than the
        # extraction task: non-square staging has no stacked tensor for
        # phase3.load_inputs to read, and could not be batched anyway.
        # One "region" that is the whole content box, cut and resized to
        # 224 exactly as every crop is, so crop-vs-whole differs in
        # CONTENT alone (roadb_regioncrop.whole_frame_patch).
        patches = [roadb_regioncrop.whole_frame_patch()]
    # Ordered names, BY LAYOUT. Anatomy disambiguates mirrored pairs by
    # side; random is named neutrally, because laterality is anatomical
    # and a randomly placed box has none (roadb_regioncrop.region_names).
    region_names = roadb_regioncrop.region_names(patches, layout)

    rows = load_geometry_rows(root)
    patient_ids = [int(row["patient_id"]) for row in rows]

    stacks = []
    for patient_id in patient_ids:
        array = np.asarray(load_crop(root, setting, rows, patient_id))
        staged = Staged(
            image=array,
            source_size=(int(array.shape[1]), int(array.shape[0])),
            content_box=(0, 0, int(array.shape[1]), int(array.shape[0])),
            scale=1.0,
        )
        stacks.append(roadb_regioncrop.crops_for(staged, patches))
    crops = np.concatenate(stacks, axis=0)
    n_regions = len(patches)
    if crops.shape[0] != len(patient_ids) * n_regions:
        raise ValueError(
            f"{crops.shape[0]} crops for {len(patient_ids)} patients x "
            f"{n_regions} regions"
        )
    ctx.log(
        f"  {len(patient_ids)} patients x {n_regions} {layout} regions = "
        f"{crops.shape[0]} crops at {crops.shape[1]}x{crops.shape[2]}"
    )

    init_dir = (
        Path(declared["pretrained_init"]["path"])
        if "pretrained_init" in declared else None
    )
    values, report = extract_module.extract_features(
        task["backbone"], task["init"], crops,
        checkpoint_path=None,
        batch_size=int(task["batch_size"]),
        init_dir=init_dir,
    )
    values = np.asarray(values)
    # **Graph backbones return feature maps by decision, not by accident**
    # (embeddings.KIND_FOR_BACKBONE_KIND: the frozen boundary for a graph
    # model is its final map). Option B: pool each crop's map to a vector
    # so both SR-GNN arms sit on identical nodes and the combination
    # contrast varies combination alone. NOT SR-GNN as published --
    # roadb_regioncrop.pool_feature_maps carries what that gives up.
    pooled_from_maps = values.ndim == 4
    if pooled_from_maps:
        values = roadb_regioncrop.pool_feature_maps(values)
    if values.ndim != 2 or values.shape[0] != crops.shape[0]:
        raise ValueError(
            f"expected one vector per crop, got {values.shape} for "
            f"{crops.shape[0]} crops"
        )
    if layout == "whole":
        # The control is an ordinary pooled set: (n_patients, dim), the
        # shape train_cv reads. Its single region is not a region.
        values = values.reshape(len(patient_ids), values.shape[1])
    else:
        values = values.reshape(len(patient_ids), n_regions, values.shape[1])

    directory = ctx.repo_root / "data" / "embeddings" / task["out_version"]
    whole = layout == "whole"
    saved = embeddings.save(
        directory / f"{task['backbone']}__{task['init']}__{layout}",
        values,
        backbone=task["backbone"],
        backbone_kind=BACKBONES[task["backbone"]]["kind"],
        init=task["init"],
        geometry="g2",
        variant=None,
        checkpoint_sha256=None,
        patient_ids=patient_ids,
        manifest_ids=patient_ids,
        regions=None if whole else region_names,
        kind=None if whole else "region_vectors",
    )
    summary = {
        "layout": layout,
        "n_patients": len(patient_ids),
        "n_regions": None if whole else n_regions,
        "feature_dim": int(values.shape[-1]),
        "regions": None if whole else region_names,
        # The random arm's names carry no anatomy, so the size matching
        # that makes it a control is recorded rather than implied.
        "size_provenance": roadb_regioncrop.size_provenance(
            anatomy, patches, layout
        ) or None,
        "pooled_from_feature_maps": pooled_from_maps,
        "not_srgnn_as_published": (
            "pooling changes SeqSelfAttention into a smaller layer; this "
            "arm tests message passing over anatomy regions, not SR-GNN's "
            "published architecture"
        ) if pooled_from_maps else None,
        "crop_output_size": roadb_regioncrop.CROP_OUTPUT_SIZE,
        "extraction": report,
        "saved": saved,
        "precompute_once": (
            "frozen backbone, head-only fits downstream -- every arm reads "
            "this set; nothing re-extracts per epoch"
        ),
    }
    with ctx.atomic("metrics.json", tier="SHAREABLE") as tmp:
        tmp.write_text(
            json.dumps(as_builtin(summary), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    ctx.log(
        f"  saved {values.shape} region_vectors, dim {values.shape[2]}"
    )


def region_crop_sheet(ctx) -> None:
    """Branch 3's gate: 22 anatomy crops per patient, reviewed by eye.

    **Nothing in this branch runs until this sheet is reviewed** (brief §5,
    criterion 3). Whether a box labelled ``philtral_column`` contains a
    philtral column is not something arithmetic can answer, and every phase
    that skipped its sheet produced a defect no test caught.

    **Two questions, one panel** (``roadb_regioncrop.crop_sheet``): the face
    with its boxes drawn answers *is this the right anatomy*; the crops at
    NATIVE resolution answer *does ~96 px of philtral column show anything*
    -- the magnification premise the whole branch rests on. Resizing the
    crops to a uniform cell would show the reviewer the resampler instead.

    **The random control is rendered too.** It is the arm that decides what
    a positive finding is ABOUT, so its placement gets the same eye.

    CLUSTER-ONLY PNGs -- patient faces. Patient ids go to the CLUSTER-ONLY
    selection record, never to the SHAREABLE metrics.
    """
    import json

    import numpy as np

    from . import roadb, roadb_regioncrop
    from .geometry.render import save_sheet
    from .geometry.staging import Staged
    from .run import as_builtin
    from .train.phase3 import load_geometry_rows

    task = ctx.config["task"]
    declared = {entry["name"]: entry for entry in ctx.inputs}
    artifact = declared[task["staged_artifact"]]
    root = Path(artifact["path"])
    setting = [
        s for s in roadb.staging_settings()
        if s["artifact"] == task["staged_artifact"]
    ]
    if len(setting) != 1:
        raise ValueError(
            f"{task['staged_artifact']} is not one of the ten staged settings"
        )
    setting = setting[0]
    if setting["aspect"] != "nonsquare" or setting["geometry"] != "g2":
        raise ValueError(
            f"the branch crops from non-square G2 (brief §2.1); "
            f"{setting['name']} is {setting['aspect']} {setting['geometry']}"
        )

    rows = load_geometry_rows(root)
    patients = [int(row["patient_id"]) for row in rows]
    step = max(1, len(patients) // int(task["n_patients"]))
    sample = patients[::step][: int(task["n_patients"])]
    # **A sheet with nothing on it is a review gate that cannot fail
    # visibly**, so emptiness refuses HERE and names what produced it,
    # rather than reaching the renderer as a shape error.
    if not patients:
        raise ValueError(
            f"{root.name}'s geometry.csv lists no patients, so no sheet can "
            "be built -- the artifact is empty or its rows did not parse"
        )
    if not sample:
        raise ValueError(
            f"patient selection produced nothing from {len(patients)} rows "
            f"at n_patients={task['n_patients']} (step {step})"
        )

    # **[ADDED 2026-08-14] Two modes, because they answer different
    # questions.** `native` shows the crops AS CUT -- the information
    # present, which is the magnification premise. `as_fed` shows them AS
    # THE BACKBONE RECEIVES THEM, after resize_nearest squares every
    # window: a 149x124 box arrives stretched. The first sheet passed
    # review without anyone seeing that, which is why the second exists
    # (roadb.SHEETS_MUST_SHOW_WHAT_THE_CONSUMER_RECEIVES).
    mode = task.get("mode", "native")
    anatomy = roadb_regioncrop.protected_patches()
    layouts = {"anatomy": anatomy, "random": roadb_regioncrop.random_patches(anatomy)}

    sizes, rendered = {}, 0
    for label, patches in layouts.items():
        for patient_id in sample:
            array = load_crop(root, setting, rows, patient_id)
            staged = Staged(
                image=np.asarray(array),
                source_size=(int(array.shape[1]), int(array.shape[0])),
                # Non-square G2 has no pad: content fraction is 1.0000,
                # measured at staging. The box IS the image.
                content_box=(0, 0, int(array.shape[1]), int(array.shape[0])),
                scale=1.0,
            )
            sheet = roadb_regioncrop.crop_sheet(
                staged, patches, columns=int(task["columns"]),
                as_fed=mode == "as_fed", layout=label,
            )
            # save_sheet(SHEET, PATH) -- the array first. Reversing these
            # is what crashed the first cluster run: np.asarray(Path) is a
            # 0-d object array, so the renderer reported "got ()" for what
            # was a two-argument swap at this call site.
            save_sheet(
                sheet,
                ctx.path(
                    f"crops_{mode}_{label}_p{patient_id:03d}.png",
                    tier="CLUSTER-ONLY",
                ),
            )
            rendered += 1
            if label == "anatomy":
                sizes[str(patient_id)] = roadb_regioncrop.native_crop_sizes(
                    staged, patches
                )
        ctx.log(f"  {label}: {len(sample)} sheets")

    widths = [entry["width_px"]["median"] for entry in sizes.values()]
    summary = {
        "artifact": task["staged_artifact"],
        "mode": mode,
        "shows": (
            "the crops AS THE BACKBONE RECEIVES THEM, squared by "
            "resize_nearest -- the distortion the native sheet hides"
            if mode == "as_fed" else
            "the crops AS CUT, at native resolution -- the information "
            "present, which does NOT show the square resize"
        ),
        "rollup": artifact.get("rollup", artifact.get("rollup_sha256")),
        "n_regions": 22,
        "n_patients_rendered": len(sample),
        "sheets_written": rendered,
        "layouts": sorted(layouts),
        # The magnification premise, as numbers beside the sheet the
        # reviewer is judging it on.
        "median_crop_width_px": {
            "min": int(min(widths)), "max": int(max(widths)),
        },
        "brief_predicted_width_px": 96,
        "every_crop_upsamples_to_224": bool(
            all(
                entry["width_px"]["max"] <= roadb_regioncrop.CROP_OUTPUT_SIZE
                for entry in sizes.values()
            )
        ),
        "gate": (
            "no Branch 3 arm runs until this sheet is reviewed -- brief §5 "
            "criterion 3"
        ),
    }
    with ctx.atomic("metrics.json", tier="SHAREABLE") as tmp:
        tmp.write_text(
            json.dumps(as_builtin(summary), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    with ctx.atomic("crop_sizes.json", tier="CLUSTER-ONLY") as tmp:
        tmp.write_text(
            json.dumps(as_builtin(sizes), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    ctx.log(
        f"  {rendered} sheets over {len(sample)} patients; median crop width "
        f"{min(widths)}-{max(widths)} px against the brief's 96"
    )


def sheets(ctx) -> None:
    """Both Phase 2 sheets, over artifacts the run does not own. CLUSTER-ONLY.

    **The review these feed gates Phase 3** (brief §6): every phase that
    skipped the sheet produced a defect no test caught. Until 2026-08-09
    ``roadb_sheet`` was built and tested but nothing called it -- the same
    built-not-wired gap the stage task's defects hid in.

    **Axis patients are measured, not hand-picked**: evenly spaced across the
    aspect ratios recorded in the 224 non-square G1 artifact's geometry.csv,
    extremes included, because the pad and the corners differ most at the
    extremes and that is where framing defects live.

    **The detail strip shows every flagged patient** -- measured frontal
    counts are 1 at 512 and 41 at 768 -- wrapping each setting across rows at
    the configured width rather than truncating.

    **It records the rollups it read**, like the gate: the sheets and the
    verdict must describe the same bytes, and a re-run of any artifact makes
    a reviewed sheet stale. Patient ids are patient-keyed, so they go to the
    CLUSTER-ONLY selection record, never to the SHAREABLE metrics.
    """
    from . import roadb_sheet
    from .geometry.render import save_sheet
    from .run import as_builtin
    from .train.phase3 import load_geometry_rows

    task = ctx.config["task"]
    declared = {entry["name"]: entry for entry in ctx.inputs}
    by_artifact = {
        s["artifact"]: s["name"] for s in roadb.staging_settings()
    }
    settings = {s["name"]: s for s in roadb.staging_settings()}

    absent = sorted(set(by_artifact) - set(declared))
    if absent:
        raise ValueError(
            f"{len(absent)} settings not declared: {absent}. A sheet missing "
            "a column would review nine settings as ten"
        )

    roots, rows_by_setting, manifests, rollups = {}, {}, {}, {}
    for name, entry in sorted(declared.items()):
        setting_name = by_artifact.get(name)
        if setting_name is None:
            continue
        root = Path(entry["path"])
        roots[setting_name] = root
        rows_by_setting[setting_name] = load_geometry_rows(root)
        manifests[setting_name] = json.loads(
            (root / "MANIFEST.json").read_text(encoding="utf-8")
        )
        rollups[name] = entry["rollup"]

    # Axis patients from the measured aspect range. The 224 non-square G1
    # artifact records each patient's actual staged ratio, and every setting
    # shares the manifest population, so one setting's geometry serves all.
    reference = rows_by_setting["roadB_p2_224_nonsquare_g1"]
    by_aspect = sorted(
        reference,
        key=lambda row: int(row["content_w"]) / int(row["content_h"]),
    )
    count = min(roadb_sheet.AXIS_PATIENTS, len(by_aspect))
    if count == 0:
        raise ValueError("the reference geometry has no rows; nothing to review")
    positions = sorted({
        round(index * (len(by_aspect) - 1) / max(count - 1, 1))
        for index in range(count)
    })
    axis_patients = [int(by_aspect[p]["patient_id"]) for p in positions]

    axis_crops = {
        name: {
            patient: load_crop(roots[name], settings[name],
                               rows_by_setting[name], patient)
            for patient in axis_patients
        }
        for name in settings
    }
    axis = roadb_sheet.axis_sheet(axis_crops, axis_patients)
    axis_target = ctx.path("contact_sheet_axis.png", tier="CLUSTER-ONLY")
    save_sheet(
        axis["sheet"], axis_target, labels=axis["labels"],
        columns=len(axis["columns"]), cell=axis["cell"],
    )
    ctx.log(
        f"AXIS SHEET: {len(axis_patients)} patients x {len(axis['columns'])} "
        f"settings -> {axis_target.name} {axis['sheet'].shape}"
    )

    # The flagged patients come from each artifact's OWN manifest -- the
    # per-patient flags in manifest row order, which is geometry.csv's order.
    flagged = {}
    for name, manifest in manifests.items():
        flags = manifest["upsampling"]["per_patient"]
        rows = rows_by_setting[name]
        if len(flags) != len(rows):
            raise ValueError(
                f"{name}: {len(flags)} upsampling flags against "
                f"{len(rows)} geometry rows; the artifact is inconsistent"
            )
        flagged[name] = [
            int(row["patient_id"])
            for row, is_upsampled in zip(rows, flags) if is_upsampled
        ]

    strip_crops = {
        name: {
            patient: load_crop(roots[name], settings[name],
                               rows_by_setting[name], patient)
            for patient in patients
        }
        for name, patients in flagged.items() if patients
    }
    strip = roadb_sheet.detail_strip(
        strip_crops, flagged,
        max_patients=int(task.get("max_patients", 41)),
        columns=int(task.get("columns", 6)),
    )
    if strip["sheet"] is not None:
        strip_target = ctx.path("contact_sheet_detail.png", tier="CLUSTER-ONLY")
        save_sheet(
            strip["sheet"], strip_target, labels=strip["labels"],
            columns=strip["columns"], cell=strip["cell"],
        )
        ctx.log(
            f"DETAIL STRIP: {sum(e['n_shown'] for e in strip['shown'].values())} "
            f"native panels -> {strip_target.name} {strip['sheet'].shape}"
        )
    else:
        ctx.log(f"DETAIL STRIP: empty -- {strip['why_empty']}")

    # Patient-keyed, so CLUSTER-ONLY -- the same rule as predictions.csv.
    ctx.path("sheets_selection.json", tier="CLUSTER-ONLY").write_text(
        json.dumps(as_builtin({
            "axis_patients": axis_patients,
            "axis_aspects": {
                int(row["patient_id"]):
                    round(int(row["content_w"]) / int(row["content_h"]), 4)
                for row in reference
                if int(row["patient_id"]) in axis_patients
            },
            "selection_rule": (
                "evenly spaced across the aspect ratios of "
                "roadB_p2_224_nonsquare_g1's geometry rows, extremes included"
            ),
            "flagged": flagged,
        }), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    metrics = {
        "axis": {
            "columns": axis["columns"],
            "n_patients": len(axis_patients),
            "display_size": axis["display_size"],
            "sheet_shape": list(axis["sheet"].shape),
            "selection_rule": (
                "evenly spaced across the measured aspect range, extremes "
                "included -- framing defects live at the extremes"
            ),
            "answers": axis["answers"],
            "does_not_answer": axis["does_not_answer"],
        },
        "detail": {
            "shown": strip.get("shown", {}),
            "truncated": strip.get("truncated", {}),
            "sheet_shape": (
                None if strip["sheet"] is None else list(strip["sheet"].shape)
            ),
            "why_empty": strip.get("why_empty"),
        },
        "flagged_counts": {name: len(ids) for name, ids in flagged.items()},
        # Which quantity each artifact's flags are. v1 manifests carry
        # shorter-side flags -- deprecated, an overcount, but a SUPERSET of
        # the interpolated set, so a strip drawn from them misses nothing.
        # The corrected longer-side counts ride along so nobody takes
        # n_upsampled for the current definition.
        "flags_measure": {
            name: roadb_staging.manifest_upsampling_quantity(manifest)["measures"]
            for name, manifest in manifests.items()
        },
        **({
            "corrected_interpolation_counts": roadb.MEASURED_INTERPOLATION_COUNTS,
        } if any(
            roadb_staging.manifest_upsampling_quantity(m)["deprecated"]
            for m in manifests.values()
        ) else {}),
        "artifacts_read": rollups,
        "cluster_only": [
            "contact_sheet_axis.png", "contact_sheet_detail.png",
            "sheets_selection.json",
        ],
        "review_questions": list(roadb_sheet.REVIEW_QUESTIONS),
        "review_gates_phase_3": (
            "brief §6: reviewed before anything downstream consumes the "
            "artifacts"
        ),
    }
    with ctx.atomic("metrics.json", tier="SHAREABLE") as tmp:
        tmp.write_text(
            json.dumps(as_builtin(metrics), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
