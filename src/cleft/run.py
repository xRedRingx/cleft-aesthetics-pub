"""The entry point.

    python -m cleft.run --config configs/smoke.yaml

Exactly two command-line arguments exist and always will: ``--config`` and
``--out``. Nothing scientific is ever a flag (Part 2.3). Adding ``--lr`` "just
for convenience" is how the previous pipeline lost the ability to say what
produced a result; ``tests/test_config.py`` asserts the parser stays this shape.
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np

from .eval import metrics
from .provenance import RunContext, atomic_write_text

#: The smoke task does no real work, so its resample count is a speed choice
#: rather than a scientific one. It is written into metrics.json regardless:
#: a number used but not recorded is a provenance gap.
SMOKE_N_BOOT = 2000


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m cleft.run",
        description="Run one config. The config is the provenance record.",
    )
    parser.add_argument(
        "--config", required=True, type=Path, help="path to the config YAML"
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("runs"),
        help="run root; the tier and phase subdirectories are added from the config",
    )
    return parser


# --------------------------------------------------------------------------
# tasks
# --------------------------------------------------------------------------


def _read_values(path: Path) -> tuple[list[str], np.ndarray, np.ndarray]:
    with (path / "values.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    ids = [row["patient_id"] for row in rows]
    truth = np.array([float(row["truth"]) for row in rows])
    pred = np.array([float(row["pred"]) for row in rows])
    return ids, truth, pred


def task_smoke(ctx: RunContext) -> None:
    """Exercise the whole path on a synthetic input without doing real work.

    Declared input, hash check, metrics, both output tiers, one curve row. This
    is the template every real config follows.
    """
    declared = {entry["name"]: entry for entry in ctx.inputs}
    if "values" not in declared:
        raise ValueError("the smoke task requires an input named 'values'")

    source = Path(declared["values"]["path"])
    ids, truth, pred = _read_values(source)
    ctx.log(f"read {len(ids)} rows from {source}")

    # Runtime invariant (R2): the config's claim about its input is checked, not
    # trusted. A manifest that has drifted from its config must stop the run.
    expected = ctx.config["task"]["n_samples"]
    if len(ids) != expected:
        raise ValueError(
            f"config declares n_samples={expected} but the input has {len(ids)} rows"
        )

    point = metrics.pcc(truth, pred)
    lo, hi = metrics.bca_ci(
        metrics.pcc, truth, pred, n_boot=SMOKE_N_BOOT, seed=ctx.config["seed"]
    )

    summary = {
        "n": len(ids),
        "pcc": point,
        "pcc_ci95": [lo, hi],
        "pcc_ci95_n_boot": SMOKE_N_BOOT,
        "spearman": metrics.spearman(truth, pred),
        "qwk_3cat": metrics.qwk_3cat(truth, pred),
        "mae": metrics.mae(truth, pred),
        "rmse": metrics.rmse(truth, pred),
        "seed": ctx.config["seed"],
    }

    # Every output goes through ctx.atomic(): written to a temporary file and
    # renamed into place only on clean exit. Run:AI can pause a workload
    # mid-write, and a half-written file that still parses is worse than one
    # that fails outright. Checkpoints in Phase 3 use the same mechanism.
    import json

    with ctx.atomic("metrics.json", tier="SHAREABLE") as tmp:
        tmp.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    # One row, holding the metrics actually computed. A smoke run has one
    # "epoch"; inventing a curve shape would be fabricating data.
    with ctx.atomic("curves.csv", tier="SHAREABLE") as tmp:
        with tmp.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["epoch", "pcc", "mae", "rmse"])
            writer.writerow(
                [0, f"{point:.6f}", f"{summary['mae']:.6f}", f"{summary['rmse']:.6f}"]
            )

    truth_class = metrics.to_3class(truth)
    pred_class = metrics.to_3class(pred, clip=True)
    with ctx.atomic("predictions.csv", tier="CLUSTER-ONLY") as tmp:
        with tmp.open("w", encoding="utf-8", newline="") as handle:
            handle.write("# CLUSTER-ONLY: patient-keyed, never leaves EHU infrastructure\n")
            writer = csv.writer(handle)
            writer.writerow(["patient_id", "truth", "pred", "truth_3class", "pred_3class"])
            for pid, t, p, tc, pc in zip(ids, truth, pred, truth_class, pred_class):
                writer.writerow([pid, f"{t:.6f}", f"{p:.6f}", int(tc), int(pc)])

    ctx.log(f"pcc={point:.4f} ci95=({lo:.4f}, {hi:.4f})")


def declared_input(ctx: "RunContext", key: str) -> Path:
    """The path of the input a task field NAMES, not a path a task field
    carries.

    **[ADDED 2026-09-05] The distinction is guard 3.** An input declared
    under ``inputs:`` is hashed before the run starts and the run aborts
    if the bytes moved; a path inlined into ``task:`` is opened on trust.
    Two Phase 1 diagnostics read the cohort score sheet the second way,
    which meant *the workbook every label in this project derives from
    was never hash-verified by the runs that read it*.

    ``task_build_manifest`` already had this logic inline and 63 configs
    already used the naming pattern (``scut_root``, ``mebeauty_root``);
    what was missing was applying it here. Raises rather than falling
    back, because a silent fallback is how the hole stayed open.
    """
    task = ctx.config["task"]
    name = task.get(key)
    if not name:
        raise KeyError(
            f"task.{key} is required and names a declared input. "
            "[CHANGED 2026-09-05] This field used to carry a PATH; it now "
            "carries the NAME of an entry in inputs:, so the artifact is "
            "hash-verified before the task opens it."
        )
    declared = {entry["name"]: Path(entry["path"]) for entry in ctx.inputs}
    if name not in declared:
        raise KeyError(
            f"task.{key} names input {name!r}, which is not declared in "
            f"inputs: {sorted(declared)}"
        )
    return declared[name]


def task_scan_folders(ctx: RunContext) -> None:
    """Read-only inventory of the patient folder tree. Writes no artifact."""
    from .data import diagnostics

    root = declared_input(ctx, "patient_folders")
    payload, rendered = diagnostics.scan_folders_report(root)

    import json

    ctx.path("folder_scan.json", tier="SHAREABLE").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(rendered)
    for line in rendered.splitlines():
        ctx.log(line)


def task_inspect_scoresheet(ctx: RunContext) -> None:
    """Load a workbook and describe it. Builds nothing."""
    from .data import diagnostics

    task = ctx.config["task"]
    compare_with = (
        declared_input(ctx, "compare_with") if task.get("compare_with")
        else None
    )
    payload, rendered = diagnostics.inspect_scoresheet_report(
        declared_input(ctx, "scoresheet"), compare_with=compare_with
    )

    import json

    ctx.path("scoresheet_report.json", tier="SHAREABLE").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(rendered)
    for line in rendered.splitlines():
        ctx.log(line)


def task_build_manifest(ctx: RunContext) -> None:
    """Phase 1: manifest, labels, reliability, folds, and the SHAREABLE block."""
    import json

    from .data import build as build_module
    from .data import manifest as manifest_module

    task = ctx.config["task"]
    declared = {entry["name"]: Path(entry["path"]) for entry in ctx.inputs}

    def resolve(key: str) -> Path | None:
        name = task.get(key)
        if not name:
            return None
        if name not in declared:
            raise ValueError(
                f"task.{key} names input {name!r}, which is not declared in inputs: "
                f"{sorted(declared)}"
            )
        return declared[name]

    artifact_dir = ctx.repo_root / "data" / "manifests" / task["out_version"]

    result = build_module.build(
        primary_path=resolve("primary_scoresheet"),
        patient_folders_path=resolve("patient_folders"),
        equivalence_path=resolve("equivalence_scoresheet"),
        soft_reference_path=resolve("soft_label_crosscheck"),
        artifact_dir=artifact_dir,
        orthodontist_rater=task["orthodontist_rater"],
        n_folds=task["n_folds"],
        seed=ctx.config["seed"],
        expected=manifest_module.Counts(
            patients=task["expect_patients"],
            frontal=task["expect_frontal"],
            basal=task["expect_basal"],
            photoless=task["expect_photoless"],
        ),
        expected_scored_rows=task["expect_scored_rows"],
        log=ctx.log,
    )

    ctx.path("metrics.json", tier="SHAREABLE").write_text(
        json.dumps(result.summary.as_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    rendered = result.summary.render()
    print(rendered)
    for line in rendered.splitlines():
        ctx.log(line)


def task_build_views_manifest(ctx: RunContext) -> None:
    """Phase 12, stop 1: derive cleft_v1_views from cleft_v1
    (phase12.STOP_1_MANIFEST). 237 rows in, 236 out, folder 238 dropped,
    labels and FOLDS carried verbatim, the two known exceptions asserted.

    The pairing is CARRIED from the artifact that measured it, never
    re-derived -- data/views.py holds the invariants and refuses a source
    that has drifted from the Phase 1 measurement.
    """
    import json

    from . import phase12
    from .data import views
    from .data.manifest import MANIFEST_COLUMNS, load_manifest
    from .provenance.hashing import hash_dir

    task = ctx.config["task"]
    declared = {entry["name"]: Path(entry["path"]) for entry in ctx.inputs}
    source_dir = declared[task["manifest_artifact"]]
    rows = load_manifest(source_dir / "manifest.csv")

    kept = views.derive(rows)
    # Configs-not-flags: the expected shape is declared in the config,
    # then required to agree with the derivation's own contract -- a
    # disagreement stops the run rather than electing a winner.
    expected = {
        "patients": int(task["expect_patients"]),
        "frontal": int(task["expect_frontal"]),
        "basal": int(task["expect_basal"]),
    }
    if expected != views.VIEWS_COHORT:
        raise ValueError(
            f"config expects {expected}, the derivation is defined for "
            f"{views.VIEWS_COHORT} -- the config and the code disagree "
            "about the cohort and neither may be silently preferred"
        )

    artifact_dir = ctx.repo_root / "data" / "manifests" / task["out_version"]
    if artifact_dir.exists():
        raise ValueError(
            f"{artifact_dir} already exists. NEVER overwrite an artifact "
            "version: derive into a new version instead"
        )
    artifact_dir.mkdir(parents=True)

    # Same columns, same order, same tier marker as cleft_v1, so every
    # existing loader reads the derived artifact unchanged.
    header = [name for name, _ in MANIFEST_COLUMNS]
    lines = [
        "# CLUSTER-ONLY: patient-keyed, never leaves EHU infrastructure",
        ",".join(header),
    ]
    for row in kept:
        lines.append(",".join(str(row[name]) for name in header))
    atomic_write_text(artifact_dir / "manifest.csv", "\n".join(lines) + "\n")

    summary = views.pairing_summary(kept)
    payload = hash_dir(artifact_dir)
    atomic_write_text(
        artifact_dir / "MANIFEST.json",
        json.dumps(
            {
                "artifact": artifact_dir.name,
                "derived_from": str(source_dir),
                "payload_rollup": payload["rollup"],
                "payload_files": payload["files"],
                "payload_total_bytes": payload["total_bytes"],
                "manifest_csv": views.views_manifest_schema(),
                "pairing": summary,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )

    with ctx.atomic("metrics.json", tier="SHAREABLE") as tmp:
        tmp.write_text(
            json.dumps(as_builtin({
                "registration": phase12.PHASE_12_REGISTERED,
                "stop_1": phase12.STOP_1_MANIFEST,
                "pairing": summary,
                "artifact": str(artifact_dir),
            }), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    ctx.log(
        "cleft_v1_views: {} patients, 2 views each; excluded {}; "
        "exception pairing {}; folds carried verbatim".format(
            summary["patients"], summary["excluded"]["patient"],
            summary["exception_pairing"],
        )
    )


def task_stage_basal_views(ctx: RunContext) -> None:
    """Phase 12, stop 2: stage all 236 basal views at G1 and render the
    review sheets (phase12.STOP_2_STAGING). GATE FIRST: refuses before
    touching any pixel unless the maintainer has set basal_use_acknowledged
    (phase12.BASAL_USE_GATE).

    The FROZEN staging.stage() runs unchanged -- no trapezium parameter
    is consumed at whole-image G1 (phase12.STOP_2_STAGING_DECISION), and
    the boundary is asserted here: no G2 tensor, no patch definitions.
    NO EMBEDDING is extracted; the sheets go in front of the eye first.
    """
    import json

    from . import phase12
    from .data.manifest import load_manifest
    from .geometry import basal, contact, render
    from .geometry.staging import stage
    from .provenance.hashing import hash_dir

    task = ctx.config["task"]

    # ---- the ethics gate, before any pixel --------------------------------
    if task.get("basal_use_acknowledged") is not True:
        raise ValueError(
            "basal_use_acknowledged is not set. The REC approvals' crop "
            "conditions were written around the RATED frontal view; using "
            "the captured-but-unrated submental view is a DIFFERENT USE "
            "of the same images, and that judgement belongs to the maintainer, "
            "not to code. Edit configs/p12_stage_basal.yaml to "
            "basal_use_acknowledged: true to confirm "
            "(phase12.BASAL_USE_GATE); the generator carries the edit"
        )
    ctx.log("basal_use_acknowledged: true -- confirmation on record")

    declared = {entry["name"]: Path(entry["path"]) for entry in ctx.inputs}
    manifest_dir = declared[task["manifest_artifact"]]
    folders = declared[task["patient_folders"]]
    rows = load_manifest(manifest_dir / "manifest.csv")
    if len(rows) != int(task["expect_patients"]):
        raise ValueError(
            f"{len(rows)} rows in {manifest_dir}, expected "
            f"{task['expect_patients']} -- this task is defined against "
            "cleft_v1_views and no other manifest"
        )

    artifact_dir = ctx.repo_root / "data" / "staged" / task["out_version"]
    if artifact_dir.exists():
        raise ValueError(
            f"{artifact_dir} already exists. NEVER overwrite an artifact "
            "version: stage into a new version instead"
        )

    # ---- stage every basal, measuring as we go ----------------------------
    staged_rows = []
    tensors = []
    for row in rows:
        patient_id = int(row["patient_id"])
        basal_id = int(row["basal_id"])
        path = contact.find_image(folders, patient_id, basal_id)
        image = render.load_image(path)
        white = basal.corner_white_fraction(image)
        staged = stage(image)
        tensors.append(staged.image)
        staged_rows.append(
            {
                "patient_id": patient_id,
                "basal_id": basal_id,
                "source_w": staged.source_size[0],
                "source_h": staged.source_size[1],
                "aspect_ratio": round(staged.aspect_ratio, 6),
                "content_x": staged.content_box[0],
                "content_y": staged.content_box[1],
                "content_w": staged.content_box[2],
                "content_h": staged.content_box[3],
                "pad_fraction": round(staged.pad_fraction, 6),
                "corner_white_fraction": round(white, 6),
            }
        )

    ratios = [r["aspect_ratio"] for r in staged_rows]
    verdict = basal.whiteness_verdict(
        [r["corner_white_fraction"] for r in staged_rows]
    )
    ctx.log(
        "staged {} basal views; AR {:.3f}..{:.3f} (frontal range "
        "{}..{}); corner-white median {:.4f}, {} flagged".format(
            len(staged_rows), min(ratios), max(ratios),
            basal.FRONTAL_AR_RANGE[0], basal.FRONTAL_AR_RANGE[1],
            verdict["median"], verdict["flagged_below_0_90"],
        )
    )
    ctx.log(verdict["verdict"])

    # ---- the artifact: staged_v1's G1 layout, G1 ONLY ---------------------
    # The boundary asserted (phase12.STOP_2_STAGING_DECISION): no G2
    # tensor, no patch definitions -- those consume FRONTAL trapezium
    # parameters that were never characterised for the submental view.
    artifact_dir.mkdir(parents=True)
    np.save(artifact_dir / "staged_patient_g1.npy", np.stack(tensors))
    header = list(staged_rows[0])
    lines = ["# CLUSTER-ONLY: patient-keyed geometry", ",".join(header)]
    lines += [",".join(str(r[k]) for k in header) for r in staged_rows]
    atomic_write_text(artifact_dir / "geometry.csv", "\n".join(lines) + "\n")

    payload = hash_dir(artifact_dir)
    atomic_write_text(
        artifact_dir / "MANIFEST.json",
        json.dumps(
            {
                "artifact": artifact_dir.name,
                "view": "basal (submental)",
                "derived_from": str(manifest_dir),
                "payload_rollup": payload["rollup"],
                "payload_files": payload["files"],
                "payload_total_bytes": payload["total_bytes"],
                "staged_tier": "CLUSTER-ONLY",
                "staging_decision": phase12.STOP_2_STAGING_DECISION,
                "whiteness": verdict,
                "note": (
                    "staged_patient_g1.npy is (236, 224, 224, 3) uint8, "
                    "row order matching geometry.csv. G1 ONLY: no G2 "
                    "tensor and no patch definitions, by the registered "
                    "boundary."
                ),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )

    # ---- the review sheets: ALL 236, most suspicious first ----------------
    order = sorted(
        range(len(staged_rows)),
        key=lambda i: staged_rows[i]["corner_white_fraction"],
    )
    cell = (224, 224)
    n_sheets = 0
    for start in range(0, len(order), basal.PANELS_PER_SHEET):
        chunk = order[start:start + basal.PANELS_PER_SHEET]
        labels = []
        for i in chunk:
            r = staged_rows[i]
            flag = " FLAG" if (
                r["corner_white_fraction"] < basal.FLAG_BELOW
            ) else ""
            labels.append(
                "p{} b{} ar{:.2f} w{:.2f}{}".format(
                    r["patient_id"], r["basal_id"],
                    r["aspect_ratio"], r["corner_white_fraction"], flag,
                )
            )
        panels = [
            render.Panel(label=label, image=tensors[i])
            for label, i in zip(labels, chunk)
        ]
        sheet = render.tile(panels, columns=basal.SHEET_COLUMNS)
        n_sheets += 1
        target = ctx.path(
            f"basal_review_sheet_{n_sheets:02d}.png", tier="CLUSTER-ONLY"
        )
        render.save_sheet(
            sheet, target, labels=labels,
            columns=basal.SHEET_COLUMNS, cell=cell,
        )
    ctx.log(
        f"{n_sheets} review sheets, ALL {len(staged_rows)} crops, ordered "
        "ascending corner-white (most suspicious on sheet 1). NO "
        "EMBEDDING before the visual check pass"
    )

    summary = {
        "registration": phase12.STOP_2_STAGING,
        "staging_decision": phase12.STOP_2_STAGING_DECISION,
        "gate": phase12.BASAL_USE_GATE,
        "n_staged": len(staged_rows),
        "aspect_ratio": {
            "min": round(min(ratios), 4),
            "median": round(float(np.median(ratios)), 4),
            "max": round(max(ratios), 4),
            "frontal_range_for_comparison": basal.FRONTAL_AR_RANGE,
        },
        "pad_fraction": {
            "min": round(min(r["pad_fraction"] for r in staged_rows), 4),
            "mean": round(
                float(np.mean([r["pad_fraction"] for r in staged_rows])), 4
            ),
            "max": round(max(r["pad_fraction"] for r in staged_rows), 4),
        },
        "whiteness": verdict,
        "sheets": n_sheets,
        "review": (
            "PENDING -- the visual check on the cluster; no embedding is "
            "extracted before that pass"
        ),
    }
    with ctx.atomic("metrics.json", tier="SHAREABLE") as tmp:
        tmp.write_text(
            json.dumps(as_builtin(summary), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def task_extract_basal_embeddings(ctx: RunContext) -> None:
    """Phase 12, stop 3: the basal embedding set, plus the registered
    staging-geometry confound report (phase12.STOP_3_REGISTERED).

    The EXISTING machinery end to end: extract.extract_features (vit_b16,
    imagenet, checkpoint None -- deterministic) and embeddings.save with
    the views manifest ids, so the row order is checked at write time.
    """
    import json

    from . import embeddings as embeddings_module
    from . import phase12
    from .data.manifest import load_manifest
    from .eval.metrics import pcc
    from .provenance.hashing import hash_dir
    from .train import extract

    task = ctx.config["task"]
    declared = {entry["name"]: Path(entry["path"]) for entry in ctx.inputs}
    manifest_dir = declared[task["manifest_artifact"]]
    staged_dir = declared[task["staged_artifact"]]

    rows = load_manifest(manifest_dir / "manifest.csv")
    manifest_ids = [int(r["patient_id"]) for r in rows]
    if len(manifest_ids) != int(task["expect_patients"]):
        raise ValueError(
            f"{len(manifest_ids)} manifest rows, expected "
            f"{task['expect_patients']}"
        )

    images = np.load(staged_dir / "staged_patient_g1.npy")
    from .cluster_csv import read_cluster_csv

    geometry_rows = read_cluster_csv(
        staged_dir / "geometry.csv", expect=("patient_id",)
    )
    staged_ids = [int(r["patient_id"]) for r in geometry_rows]
    if staged_ids != manifest_ids:
        raise ValueError(
            "staged_basal_v1's row order does not match the views "
            "manifest -- the staging was built FROM this manifest, so a "
            "mismatch means one of the two artifacts changed"
        )
    if len(images) != len(manifest_ids):
        raise ValueError(
            f"{len(images)} staged images against {len(manifest_ids)} rows"
        )

    values, provenance = extract.extract_features(
        task["backbone"], task["init"], images,
        checkpoint_path=None, batch_size=int(task.get("batch_size", 32)),
    )
    out_dir = (
        ctx.repo_root / "data" / "embeddings" / task["out_version"]
        / f"{task['backbone']}__{task['init']}__{task['geometry']}"
    )
    # **[FIXED 2026-08-23, phase12.EXTRACT_GUARD_AFTER_MKDIR]** The task
    # does NOT create or pre-check this path: embeddings.save owns its
    # own existence guard and directory creation, atomically. The first
    # version pre-created the directory and save then refused it -- the
    # task's own scaffolding tripping the artifact guard, seven attempts
    # dead against an EMPTY directory. The guard is not taught to accept
    # empty directories either: an empty-dir exemption would also accept
    # a genuinely interrupted extraction's husk, which is exactly what
    # the guard exists to refuse.
    embeddings_module.save(
        out_dir, values,
        backbone=task["backbone"],
        backbone_kind=extract.backbone_kind_for(task["backbone"]),
        init=task["init"],
        geometry=task["geometry"],
        variant=None,
        checkpoint_sha256=None,
        patient_ids=manifest_ids,
        manifest_ids=manifest_ids,
    )
    payload = hash_dir(out_dir)
    ctx.log(
        f"basal set: {values.shape} written to {out_dir.name}; rollup "
        f"{payload['rollup'][:12]}..."
    )

    # ---- the registered confound report, computed where the two meet ----
    mean_label = np.array([float(r["mean"]) for r in rows])
    white = np.array(
        [float(r["corner_white_fraction"]) for r in geometry_rows]
    )
    ratio = np.array([float(r["aspect_ratio"]) for r in geometry_rows])
    spec = phase12.STOP_3_REGISTERED["confound_report"]
    r_white = float(pcc(mean_label, white))
    r_ar = float(pcc(mean_label, ratio))
    threshold = 0.13
    attaches = bool(abs(r_white) >= threshold or abs(r_ar) >= threshold)
    confound = {
        "r_corner_white_vs_mean": r_white,
        "r_aspect_ratio_vs_mean": r_ar,
        "threshold": threshold,
        "caveat_attaches_to_b_and_c": attaches,
        "reading": (
            "the staging-geometry confound caveat ATTACHES to arm B and "
            "C results" if attaches else
            "reported as null -- below the registered 0.13"
        ),
        "registered": spec["reading_registered"],
    }
    ctx.log(
        f"confound report: r(white, mean) {r_white:+.4f}, r(AR, mean) "
        f"{r_ar:+.4f} -> {confound['reading']}"
    )

    summary = {
        "registration": phase12.STOP_3_REGISTERED,
        "pad_decision": phase12.PAD_DECISION_TAKEN,
        "set": {
            "shape": list(values.shape),
            "artifact": str(out_dir),
            "rollup": payload["rollup"],
            "provenance": provenance,
        },
        "confound_report": confound,
    }
    with ctx.atomic("metrics.json", tier="SHAREABLE") as tmp:
        tmp.write_text(
            json.dumps(as_builtin(summary), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def _view_channel(name: str, declared_dirs: dict, task: dict,
                  manifest_ids: list) -> "np.ndarray":
    """One channel's (236, d) block, aligned to the views manifest.

    ``basal`` loads STRICT -- its artifact was written against these very
    ids. ``frontal`` is the 0.2520 arm's 237-patient set reused unchanged
    (phase12.STOP_3_REGISTERED), so it is loaded whole and aligned BY ID,
    with the dropped set asserted to be exactly {238}: alignment by key
    lookup, never by position.
    """
    from . import embeddings as embeddings_module
    from .data.views import EXCLUDED_PATIENT

    directory = declared_dirs[task[f"{name}_embeddings"]]
    if name == "basal":
        values, _ = embeddings_module.load(directory, manifest_ids)
        return values
    values, metadata = embeddings_module.load(directory, None)
    index_of = {int(p): i for i, p in enumerate(metadata["patient_ids"])}
    missing = [p for p in manifest_ids if p not in index_of]
    if missing:
        raise ValueError(
            f"frontal set lacks {len(missing)} views patients, e.g. "
            f"{missing[:5]} -- it is not the 237-cohort set this "
            "registration reuses"
        )
    dropped = sorted(set(index_of) - set(manifest_ids))
    if dropped != [EXCLUDED_PATIENT]:
        raise ValueError(
            f"reusing the frontal set drops {dropped}, expected exactly "
            f"[{EXCLUDED_PATIENT}] -- the subset is not the views cohort"
        )
    return values[[index_of[p] for p in manifest_ids]]


def task_view_arm(ctx: RunContext) -> None:
    """Phase 12, stop 3: one view-ablation arm
    (phase12.STOP_3_REGISTERED). ONE task for A, B, C and D -- the
    channel list is the only thing that differs between the four configs,
    which is what makes the contrasts one-factor by construction.
    """
    import json

    from . import phase12
    from .data.manifest import load_manifest
    from .eval import metrics
    from .train import harness, phase3
    from .train.torch_backbone import EmbeddingHeadBackbone

    task = ctx.config["task"]
    entries = {entry["name"]: entry for entry in ctx.inputs}
    declared_dirs = {name: Path(e["path"]) for name, e in entries.items()}
    manifest_dir = declared_dirs[task["manifest_artifact"]]

    rows = load_manifest(manifest_dir / "manifest.csv")
    manifest_ids = [int(r["patient_id"]) for r in rows]
    if len(manifest_ids) != int(task["expect_patients"]):
        raise ValueError(
            f"{len(manifest_ids)} manifest rows, expected "
            f"{task['expect_patients']} -- every arm of this phase runs "
            "on the 236 both-views cohort, the frontal-only arm included"
        )
    labels = np.array([float(r[task["label"]]) for r in rows])
    assignments = {int(r["patient_id"]): int(r["fold"]) for r in rows}

    channels = [str(v) for v in task["views"]]
    blocks = [
        _view_channel(name, declared_dirs, task, manifest_ids)
        for name in channels
    ]
    features = np.concatenate(blocks, axis=1)
    expected_parameters = int(task["expect_head_parameters"])
    if features.shape[1] + 1 != expected_parameters:
        raise ValueError(
            f"{features.shape[1]}-d features give a "
            f"{features.shape[1] + 1}-parameter head; the config expects "
            f"{expected_parameters} -- the channel list and the "
            "registration disagree"
        )
    ctx.log(
        f"arm {task['arm']}: channels {channels} -> {features.shape[1]}-d, "
        f"{expected_parameters}-parameter head, {len(manifest_ids)} patients"
    )

    from .cluster_csv import write_predictions

    seeds = [int(s) for s in task["seeds"]]
    pooled = []
    per_seed = {}
    for seed in seeds:
        result = harness.run_cv(
            features=features, labels=labels, patient_ids=manifest_ids,
            assignments=assignments,
            make_backbone=lambda _seed=seed: EmbeddingHeadBackbone(
                learning_rate=float(task["learning_rate"]),
                weight_decay=float(task["weight_decay"]),
                seed=_seed,
            ),
            config=harness.TrainConfig(
                seed=seed,
                inner_val_frac=float(task["inner_val_frac"]),
                max_epochs=int(task["max_epochs"]),
                patience=int(task["patience"]),
                monitor=task["monitor"],
            ),
        )
        seed_pcc = float(result.metrics()["pcc"])
        pooled.append(seed_pcc)
        fold_of = {int(p): assignments[int(p)] for p in result.oof_ids}
        write_predictions(
            ctx.path(f"seed_{seed}__predictions.csv", tier="CLUSTER-ONLY"),
            zip(result.oof_ids, result.oof_truth, result.oof_predictions,
                (fold_of[int(p)] for p in result.oof_ids)),
        )
        per_seed[seed] = {
            "pcc": seed_pcc,
            "selected_epochs": [f.selected_epoch for f in result.folds],
        }
        ctx.log(f"  seed {seed}: PCC {seed_pcc:.4f}")

    band = phase3.seed_variance(pooled)
    summary = {
        "arm": task["arm"],
        "channels": channels,
        "registration": phase12.STOP_3_REGISTERED,
        "readings": phase12.PHASE_12_READINGS,
        "sentences": phase12.INTERPRETATION_SENTENCES_PREWRITTEN,
        "pad_decision": phase12.PAD_DECISION_TAKEN,
        "residual_novelty": phase12.PAD_DECISION_TAKEN["residual_novelty"],
        "per_seed": per_seed,
        "pooled": {"mean": band["mean"], "sd": band["sd"]},
        "next": (
            "the paired scope B/C/D vs A consumes the per-seed CSVs -- "
            "stop 4"
        ),
    }
    with ctx.atomic("metrics.json", tier="SHAREABLE") as tmp:
        tmp.write_text(
            json.dumps(as_builtin(summary), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def task_extract_scut_embeddings(ctx: RunContext) -> None:
    """Phase 13, stop 1: frozen imagenet ViT-B/16 embeddings over the
    masked SCUT artifact, G1 224 (phase13.SCUT_SHAKEDOWN_PLAN).

    Reuses pretrain.load_masked_features (stem-aligned, the artifact's
    own index) and extract.extract_features (deterministic, eval,
    no_grad). The artifact is decoder.save_scut_embeddings' own format,
    deliberately NOT the cleft embeddings-set format. The save layer owns
    existence and creation (the guard-after-mkdir lesson): this task
    neither pre-creates nor pre-checks the path.
    """
    import json

    from . import decoder as decoder_module
    from .provenance.hashing import hash_dir
    from .scut import labels as scut_labels
    from .train import extract, pretrain

    task = ctx.config["task"]
    declared = {entry["name"]: Path(entry["path"]) for entry in ctx.inputs}
    masked_dir = declared[task["masked_scut"]]
    scut_root = declared[task["scut_root"]]

    sides = {}
    for side, expect_key in (("train", "expect_train"), ("test", "expect_test")):
        labelled = scut_labels.read_labelled_split(scut_root, side)
        stems = sorted(labelled)
        if len(stems) != int(task[expect_key]):
            raise ValueError(
                f"{side} side has {len(stems)} stems, expected "
                f"{task[expect_key]} -- the official split is READ, and a "
                "different count means a different dataset"
            )
        sides[side] = stems
    stems = sides["train"] + sides["test"]

    images = pretrain.load_masked_features(masked_dir, task["geometry"], stems)
    ctx.log(
        f"masked SCUT loaded: {images.shape} "
        f"({len(sides['train'])} train + {len(sides['test'])} test)"
    )
    values, provenance = extract.extract_features(
        task["backbone"], task["init"], images,
        checkpoint_path=None, batch_size=int(task.get("batch_size", 32)),
    )
    out_dir = ctx.repo_root / "data" / "embeddings" / task["out_version"]
    payload = decoder_module.save_scut_embeddings(
        out_dir, values, stems,
        {
            "backbone": task["backbone"], "init": task["init"],
            "geometry": task["geometry"],
            "split": {side: len(sides[side]) for side in sides},
            "provenance": provenance,
        },
    )
    # [2026-08-24] The SAME two-digest trap as the MEBeauty extraction:
    # `decoder.save_scut_embeddings` writes MANIFEST.json AFTER hashing
    # the payload and returns the payload digest, so this line was
    # printing a value no config can declare. Found while labelling the
    # MEBeauty line; MEASURED, not assumed -- the ladder's own
    # `embeddings.save` writes no manifest, so the tasks built on it
    # print a digest that IS declarable and are left alone.
    declare = hash_dir(out_dir)
    ctx.log(
        f"scut set: {values.shape} -> {out_dir.name}; "
        f"rollup_sha256_for_configs {declare['rollup'][:12]} (DECLARE THIS) "
        f"/ payload_rollup {payload['rollup'][:12]} (in MANIFEST.json)"
    )

    with ctx.atomic("metrics.json", tier="SHAREABLE") as tmp:
        tmp.write_text(
            json.dumps(as_builtin({
                "registration": __import__(
                    "cleft.phase13", fromlist=["summary"]
                ).SCUT_SHAKEDOWN_PLAN,
                "shape": list(values.shape),
                "split": {side: len(sides[side]) for side in sides},
                "artifact": str(out_dir),
                "payload_rollup": payload["rollup"],
                "rollup_sha256_for_configs": declare["rollup"],
                "two_digests": (
                    "payload_rollup covers values.npy + metadata.json and is "
                    "what MANIFEST.json records; rollup_sha256_for_configs "
                    "covers the directory AS IT STANDS, MANIFEST.json "
                    "included -- what declare_inputs.py measures and guard "
                    "3 verifies. **They cannot coincide**: the manifest "
                    "contains the payload digest, so hashing the "
                    "directory again necessarily gives a different value. "
                    "DECLARE THE SECOND ONE."
                ),
            }), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def task_train_scut_decoder(ctx: RunContext) -> None:
    """Phase 13, stop 1: train the bottleneck decoder on the SCUT fit
    side, reconstruct the held-out side, write the SCUT band baseline and
    the comparability table, render the review sheets
    (phase13.STOP_1_BUILT).

    NO COHORT PIXEL IS GENERATED HERE: the cleft staged artifact is read
    only for the comparability table's CONTENT STATISTICS (band
    content-pixel fraction and variance); no cleft embedding passes
    through the decoder in this task. The pretraining discipline applies:
    a FIXED epoch budget, best-checkpoint selection by inner L2, no early
    stopping, every epoch's checkpoint and curve row written.
    """
    import json

    from . import decoder as decoder_module
    from . import phase13
    from .data.manifest import load_manifest  # noqa: F401  (row order note)
    from .geometry import render
    from .scut import labels as scut_labels
    from .train import pretrain

    task = ctx.config["task"]
    declared = {entry["name"]: Path(entry["path"]) for entry in ctx.inputs}
    embeddings_dir = declared[task["scut_embeddings"]]
    masked_dir = declared[task["masked_scut"]]
    scut_root = declared[task["scut_root"]]
    staged_dir = declared[task["cleft_staged"]]

    values, stems, metadata = decoder_module.load_scut_embeddings(
        embeddings_dir
    )
    index_of = {stem: i for i, stem in enumerate(stems)}
    train_stems = sorted(scut_labels.read_labelled_split(scut_root, "train"))
    test_stems = sorted(scut_labels.read_labelled_split(scut_root, "test"))
    if len(train_stems) != int(task["expect_train"]) or (
        len(test_stems) != int(task["expect_test"])
    ):
        raise ValueError(
            f"split {len(train_stems)}/{len(test_stems)}, expected "
            f"{task['expect_train']}/{task['expect_test']}"
        )
    missing = [s for s in train_stems + test_stems if s not in index_of]
    if missing:
        raise ValueError(
            f"{len(missing)} split stems missing from the embedding "
            f"artifact, e.g. {missing[:3]}"
        )

    # ---- fit / inner-val, seeded, the pretraining shape -----------------
    rng = np.random.default_rng(int(ctx.config["seed"]))
    order = rng.permutation(len(train_stems))
    n_inner = max(1, int(round(len(train_stems) * float(task["inner_val_frac"]))))
    inner_stems = [train_stems[i] for i in sorted(order[:n_inner])]
    fit_stems = [s for s in train_stems if s not in set(inner_stems)]
    ctx.log(
        f"fit {len(fit_stems)}, inner-val {len(inner_stems)}, held-out "
        f"{len(test_stems)}; fixed {task['epochs']}-epoch budget, best by "
        "inner L2, no early stopping (the pretraining discipline)"
    )

    images = pretrain.load_masked_features(
        masked_dir, task["geometry"], fit_stems + inner_stems + test_stems
    )
    row_of = {
        stem: i for i, stem in enumerate(fit_stems + inner_stems + test_stems)
    }

    import torch

    torch.manual_seed(int(ctx.config["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = decoder_module.build(seed=int(ctx.config["seed"])).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(task["learning_rate"])
    )
    n_parameters = sum(p.numel() for p in model.parameters())
    ctx.log(
        f"decoder: {n_parameters:,} parameters (registered exactly "
        f"{decoder_module.REGISTERED_PARAMETERS:,})"
    )

    def embedding_rows(subset):
        return torch.as_tensor(
            np.stack([values[index_of[s]] for s in subset]), device=device
        )

    def target_rows(subset):
        rows = np.stack(
            [images[row_of[s]] for s in subset]
        ).astype(np.float32) / 255.0
        return torch.as_tensor(rows, device=device).permute(0, 3, 1, 2)

    def reconstruct(subset, batch: int):
        out = []
        model.eval()
        with torch.no_grad():
            for start in range(0, len(subset), batch):
                chunk = subset[start:start + batch]
                pred = model(embedding_rows(chunk))
                out.append(
                    (pred.permute(0, 2, 3, 1).cpu().numpy() * 255.0)
                )
        return np.concatenate(out)

    batch = int(task["batch_size"])
    ssim_sample = inner_stems[: int(task["ssim_inner_sample"])]
    curve = []
    best = None
    checkpoints_dir = ctx.path("checkpoints", tier="SHAREABLE")
    checkpoints_dir.mkdir()
    for epoch in range(1, int(task["epochs"]) + 1):
        model.train()
        epoch_order = np.random.default_rng(
            int(ctx.config["seed"]) + epoch
        ).permutation(len(fit_stems))
        total, seen = 0.0, 0
        for start in range(0, len(fit_stems), batch):
            chunk = [fit_stems[i] for i in epoch_order[start:start + batch]]
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.mse_loss(
                model(embedding_rows(chunk)), target_rows(chunk)
            )
            loss.backward()
            optimizer.step()
            total += float(loss.item()) * len(chunk)
            seen += len(chunk)
        train_l2 = total / max(seen, 1)

        inner_recon = reconstruct(inner_stems, batch)
        inner_reference = np.stack(
            [images[row_of[s]] for s in inner_stems]
        ).astype(np.float64)
        inner_l2 = float(
            np.mean(((inner_recon - inner_reference) / 255.0) ** 2)
        )
        inner_ssim = float(np.mean([
            decoder_module.ssim(
                images[row_of[s]],
                inner_recon[inner_stems.index(s)],
            )
            for s in ssim_sample
        ]))
        curve.append({
            "epoch": epoch, "train_l2": train_l2,
            "inner_val_l2": inner_l2, "inner_val_ssim": inner_ssim,
        })
        # Every epoch's checkpoint and curve row is WRITTEN, not
        # discarded -- the computed-then-discarded lesson, criterion 7.
        torch.save(
            model.state_dict(),
            checkpoints_dir / decoder_module.checkpoint_name(epoch),
        )
        if best is None or inner_l2 < best[1]:
            best = (epoch, inner_l2)
        ctx.log(
            f"  epoch {epoch:2d}: train L2 {train_l2:.5f}, inner L2 "
            f"{inner_l2:.5f}, inner SSIM {inner_ssim:.4f}"
        )
    selected_epoch = best[0]
    model.load_state_dict(
        torch.load(
            checkpoints_dir / decoder_module.checkpoint_name(selected_epoch),
            map_location=device,
        )
    )
    ctx.log(f"selected epoch {selected_epoch} by inner L2 {best[1]:.5f}")

    with ctx.atomic("curves.csv", tier="SHAREABLE") as tmp:
        lines = ["epoch,train_l2,inner_val_l2,inner_val_ssim"]
        lines += [
            f"{r['epoch']},{r['train_l2']!r},{r['inner_val_l2']!r},"
            f"{r['inner_val_ssim']!r}"
            for r in curve
        ]
        tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # ---- held-out reconstruction, metrics over ALL ----------------------
    held_recon = reconstruct(test_stems, batch)
    scut_box = (0, 0, decoder_module.OUTPUT_SIZE, decoder_module.OUTPUT_SIZE)
    per_face = []
    scut_band_rows = {band: [] for band in decoder_module.BANDS}
    scut_band_stats = {band: {"fraction": [], "variance": []}
                       for band in decoder_module.BANDS}
    for position, stem in enumerate(test_stems):
        reference = images[row_of[stem]]
        reconstruction = held_recon[position]
        l2 = float(np.mean(((reconstruction - reference) / 255.0) ** 2))
        bands = decoder_module.band_errors(reference, reconstruction, scut_box)
        for band in decoder_module.BANDS:
            scut_band_rows[band].append(bands[band]["mse"])
            scut_band_stats[band]["fraction"].append(
                bands[band]["content_fraction"]
            )
            scut_band_stats[band]["variance"].append(
                bands[band]["content_variance"]
            )
        per_face.append({"stem": stem, "l2": l2})
    ssim_all = [
        decoder_module.ssim(images[row_of[s]], held_recon[i])
        for i, s in enumerate(test_stems)
    ]
    held_l2 = [f["l2"] for f in per_face]
    scut_baseline = {
        band: float(np.nanmean(scut_band_rows[band]))
        for band in decoder_module.BANDS
    }
    with ctx.atomic("scut_band_baseline.json", tier="SHAREABLE") as tmp:
        tmp.write_text(
            json.dumps({
                "note": (
                    "the double difference's denominator "
                    "(phase13.DOUBLE_DIFFERENCE_RULE): per-band mean MSE "
                    "over CONTENT pixels, held-out SCUT, the selected "
                    "epoch's decoder"
                ),
                "selected_epoch": selected_epoch,
                "baseline_mse_by_band": scut_baseline,
                "n_faces": len(test_stems),
            }, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    # ---- the comparability table, both families, no reconstruction ------
    cleft_images = np.load(staged_dir / "staged_patient_g1.npy")
    from .cluster_csv import read_cluster_csv

    geometry_rows = read_cluster_csv(
        staged_dir / "geometry.csv", expect=("patient_id",)
    )
    cleft_band_stats = {band: {"fraction": [], "variance": []}
                        for band in decoder_module.BANDS}
    for row, image in zip(geometry_rows, cleft_images):
        box = (
            int(row["content_x"]), int(row["content_y"]),
            int(row["content_w"]), int(row["content_h"]),
        )
        bands = decoder_module.band_errors(image, image, box)
        for band in decoder_module.BANDS:
            cleft_band_stats[band]["fraction"].append(
                bands[band]["content_fraction"]
            )
            cleft_band_stats[band]["variance"].append(
                bands[band]["content_variance"]
            )
    comparability = {}
    for band in decoder_module.BANDS:
        cleft_variance = float(np.nanmean(cleft_band_stats[band]["variance"]))
        scut_variance = float(np.nanmean(scut_band_stats[band]["variance"]))
        ratio = (
            max(cleft_variance, scut_variance)
            / max(min(cleft_variance, scut_variance), 1e-12)
        )
        comparability[band] = {
            "cleft_content_fraction": float(
                np.nanmean(cleft_band_stats[band]["fraction"])
            ),
            "scut_content_fraction": float(
                np.nanmean(scut_band_stats[band]["fraction"])
            ),
            "cleft_content_variance": cleft_variance,
            "scut_content_variance": scut_variance,
            "variance_ratio": float(ratio),
            "caveat_attaches": bool(ratio > 2.0),
        }
    flagged = [b for b, c in comparability.items() if c["caveat_attaches"]]
    ctx.log(
        "comparability: "
        + ", ".join(
            f"{b} ratio {comparability[b]['variance_ratio']:.2f}"
            f"{' CAVEAT' if comparability[b]['caveat_attaches'] else ''}"
            for b in decoder_module.BANDS
        )
    )

    # ---- the review sheets: worst first, then a seeded sample -----------
    worst = sorted(range(len(per_face)), key=lambda i: -per_face[i]["l2"])
    n_worst = int(task["sheet_worst"])
    n_sample = int(task["sheet_sample"])
    remaining = [i for i in range(len(per_face)) if i not in set(worst[:n_worst])]
    sampled = list(
        np.random.default_rng(int(ctx.config["seed"])).choice(
            remaining, size=min(n_sample, len(remaining)), replace=False
        )
    )
    chosen = worst[:n_worst] + sorted(int(i) for i in sampled)
    pairs_per_sheet = 12
    n_sheets = 0
    for start in range(0, len(chosen), pairs_per_sheet):
        chunk = chosen[start:start + pairs_per_sheet]
        panels, labels = [], []
        for i in chunk:
            stem = per_face[i]["stem"]
            side_by_side = np.concatenate(
                [
                    images[row_of[stem]].astype(np.uint8),
                    np.clip(held_recon[i], 0, 255).astype(np.uint8),
                ],
                axis=1,
            )
            panels.append(render.Panel(
                label=f"{stem} l2 {per_face[i]['l2']:.4f} "
                      f"ssim {ssim_all[i]:.3f}",
                image=side_by_side,
            ))
            labels.append(panels[-1].label)
        sheet = render.tile(panels, columns=3)
        n_sheets += 1
        target = ctx.path(
            f"scut_reconstruction_sheet_{n_sheets:02d}.png", tier="SHAREABLE"
        )
        render.save_sheet(sheet, target, labels=labels, columns=3,
                          cell=(448, 224))
    ctx.log(
        f"{n_sheets} sheets: worst {n_worst} by L2 first, then a seeded "
        f"sample of {len(sampled)} -- SHAREABLE, SCUT is public. NO "
        "COHORT PIXEL was generated in this run"
    )

    summary = {
        "registration": phase13.STOP_1_BUILT,
        "rule": phase13.DOUBLE_DIFFERENCE_RULE,
        "n_parameters": n_parameters,
        "selected_epoch": selected_epoch,
        "curve": curve,
        "held_out": {
            "n": len(test_stems),
            "l2_mean": float(np.mean(held_l2)),
            "l2_worst": float(np.max(held_l2)),
            "ssim_mean": float(np.mean(ssim_all)),
        },
        "scut_band_baseline": scut_baseline,
        "comparability": comparability,
        "comparability_caveats": flagged or "none -- all ratios within 2x",
        "review": (
            "PENDING -- the visual check on the sheets before any cohort "
            "pixel is generated (exit criterion 1)"
        ),
    }
    with ctx.atomic("metrics.json", tier="SHAREABLE") as tmp:
        tmp.write_text(
            json.dumps(as_builtin(summary), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def task_reconstruct_cohort(ctx: RunContext) -> None:
    """Phase 13, stop 2: reconstruct the 237-patient cohort from the
    frozen cleft embeddings through the shakedown's picked checkpoint,
    bank the per-patient band figures, and render CLUSTER-ONLY review
    sheets for the visual check on the cluster (phase13.STOP_2_BUILT).

    GATE FIRST: refuses before any read while the maintainer has not set
    cleft_reconstruction_acknowledged (phase13.CLEFT_RECONSTRUCTION_GATE).
    Every reconstruction is CLUSTER-ONLY without exception, and the raw
    reconstruction tensor is DELIBERATELY NOT PERSISTED -- generated
    cohort pixels exist in the sheets alone. NO double difference is
    computed here: the regional comparison is stop 3's, composed from
    this run's band figures and the shakedown's scut_band_baseline.json,
    with the bound readings applied where they fire and
    phase13.SCUT_NORMAL_PRIOR travelling by name.
    """
    import json

    from . import decoder as decoder_module
    from . import embeddings as cleft_embeddings_module
    from . import phase13
    from .cluster_csv import read_cluster_csv
    from .data.manifest import load_manifest
    from .geometry import render

    task = ctx.config["task"]

    # ---- the governance gate, before any read ---------------------------
    if task.get("cleft_reconstruction_acknowledged") is not True:
        raise ValueError(
            "cleft_reconstruction_acknowledged is not set. A "
            "reconstructed cleft face is a GENERATED PATIENT-DERIVED "
            "image -- a new kind of output the REC approvals' crop "
            "conditions never contemplated, and that judgement belongs "
            "to the maintainer, not to code. Edit "
            "configs/p13_cohort_reconstruct.yaml to "
            "cleft_reconstruction_acknowledged: true to confirm "
            "(phase13.CLEFT_RECONSTRUCTION_GATE); the generator carries "
            "the edit"
        )
    ctx.log(
        "cleft_reconstruction_acknowledged: true -- the maintainer's "
        "confirmation on record. Every reconstruction is CLUSTER-ONLY "
        "without exception"
    )

    declared = {entry["name"]: Path(entry["path"]) for entry in ctx.inputs}
    run_dir = declared[task["decoder_run"]]
    embeddings_dir = declared[task["cleft_embeddings"]]
    manifest_dir = declared[task["manifest_artifact"]]
    staged_dir = declared[task["staged_artifact"]]

    rows = load_manifest(manifest_dir / "manifest.csv")
    if len(rows) != int(task["expect_patients"]):
        raise ValueError(
            f"{len(rows)} rows in {manifest_dir}, expected "
            f"{task['expect_patients']} -- this task is defined against "
            "cleft_v1 and no other manifest"
        )
    manifest_ids = [int(r["patient_id"]) for r in rows]

    values, metadata = cleft_embeddings_module.load(
        embeddings_dir, manifest_ids
    )
    if metadata.get("init") != "imagenet" or (
        metadata.get("geometry") != task["geometry"]
    ):
        raise ValueError(
            f"embedding set is {metadata.get('init')}/"
            f"{metadata.get('geometry')}; the decoder is an instrument "
            "for its OWN representation -- imagenet ViT-B/16 at "
            f"{task['geometry']} -- and reads no other"
        )
    if values.ndim != 2 or values.shape[1] != decoder_module.EMBED_DIM:
        raise ValueError(
            f"values {values.shape}: the decoder consumes the pooled "
            f"{decoder_module.EMBED_DIM}-d set, not a regions set"
        )

    cleft_images = np.load(staged_dir / "staged_patient_g1.npy")
    geometry_rows = read_cluster_csv(
        staged_dir / "geometry.csv", expect=("patient_id",)
    )
    position_of = {
        int(row["patient_id"]): i for i, row in enumerate(geometry_rows)
    }
    absent = [p for p in manifest_ids if p not in position_of]
    if absent:
        raise ValueError(
            f"{len(absent)} manifest patients missing from the staged "
            f"artifact, e.g. {absent[:3]}"
        )

    # ---- the picked checkpoint, the shakedown's own selection beside ----
    epoch = int(task["checkpoint_epoch"])
    checkpoint = (
        run_dir / "checkpoints" / decoder_module.checkpoint_name(epoch)
    )
    if not checkpoint.is_file():
        raise ValueError(
            f"{checkpoint} does not exist -- the declared decoder run "
            f"has no epoch-{epoch} checkpoint"
        )
    shakedown = json.loads(
        (run_dir / "metrics.json").read_text(encoding="utf-8")
    )
    selected_epoch = shakedown.get("selected_epoch")
    ctx.log(
        f"checkpoint epoch {epoch} (the pick, "
        f"phase13.STOP_1_REVIEWED); the shakedown's own inner-L2 "
        f"selection was epoch {selected_epoch}"
        + ("" if selected_epoch == epoch
           else " -- REPORTED beside the pick, never gated")
    )

    import torch

    torch.manual_seed(int(ctx.config["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = decoder_module.build(seed=int(ctx.config["seed"])).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()

    batch = int(task.get("batch_size", 32))
    chunks = []
    with torch.no_grad():
        for start in range(0, len(manifest_ids), batch):
            block = torch.as_tensor(
                values[start:start + batch], device=device
            )
            chunks.append(
                model(block).permute(0, 2, 3, 1).cpu().numpy() * 255.0
            )
    reconstructions = np.concatenate(chunks)

    # ---- per-patient figures: the double difference's NUMERATOR side ----
    per_patient = []
    band_columns = {band: [] for band in decoder_module.BANDS}
    ssim_all = []
    for i, patient in enumerate(manifest_ids):
        reference = cleft_images[position_of[patient]]
        row = geometry_rows[position_of[patient]]
        box = (
            int(row["content_x"]), int(row["content_y"]),
            int(row["content_w"]), int(row["content_h"]),
        )
        reconstruction = reconstructions[i]
        l2 = float(np.mean(((reconstruction - reference) / 255.0) ** 2))
        score = decoder_module.ssim(reference, reconstruction)
        bands = decoder_module.band_errors(reference, reconstruction, box)
        for band in decoder_module.BANDS:
            band_columns[band].append(bands[band]["mse"])
        ssim_all.append(score)
        per_patient.append({
            "patient_id": patient, "l2": l2, "ssim": score,
            "band_mse": {
                b: bands[b]["mse"] for b in decoder_module.BANDS
            },
            "band_content_fraction": {
                b: bands[b]["content_fraction"]
                for b in decoder_module.BANDS
            },
        })
    cohort_band_mean = {
        band: float(np.nanmean(band_columns[band]))
        for band in decoder_module.BANDS
    }
    with ctx.atomic("cleft_band_errors.json", tier="SHAREABLE") as tmp:
        tmp.write_text(
            json.dumps(as_builtin({
                "note": (
                    "the double difference's NUMERATOR side "
                    "(phase13.DOUBLE_DIFFERENCE_RULE): per-band MSE "
                    "over CONTENT pixels, all cohort patients, El "
                    "the maintainer's picked checkpoint. Stop 3 composes this "
                    "with the shakedown's scut_band_baseline.json; "
                    "phase13.SCUT_NORMAL_PRIOR travels with these "
                    "figures by name"
                ),
                "checkpoint_epoch": epoch,
                "cohort_band_mean_mse": cohort_band_mean,
                "n_patients": len(manifest_ids),
                "per_patient": per_patient,
            }), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    # ---- the review sheets: ALL patients, worst L2 first ----------------
    order = sorted(
        range(len(per_patient)), key=lambda i: -per_patient[i]["l2"]
    )
    pairs_per_sheet = 12
    n_sheets = 0
    for start in range(0, len(order), pairs_per_sheet):
        chunk = order[start:start + pairs_per_sheet]
        panels, labels = [], []
        for i in chunk:
            patient = per_patient[i]["patient_id"]
            reference = cleft_images[position_of[patient]]
            side_by_side = np.concatenate(
                [
                    reference.astype(np.uint8),
                    np.clip(reconstructions[i], 0, 255).astype(np.uint8),
                ],
                axis=1,
            )
            panels.append(render.Panel(
                label=f"{patient} l2 {per_patient[i]['l2']:.4f} "
                      f"ssim {per_patient[i]['ssim']:.3f}",
                image=side_by_side,
            ))
            labels.append(panels[-1].label)
        sheet = render.tile(panels, columns=3)
        n_sheets += 1
        target = ctx.path(
            f"cohort_reconstruction_sheet_{n_sheets:02d}.png",
            tier="CLUSTER-ONLY",
        )
        render.save_sheet(sheet, target, labels=labels, columns=3,
                          cell=(448, 224))
    ctx.log(
        f"{n_sheets} sheets: ALL {len(per_patient)} patients, worst L2 "
        "first -- CLUSTER-ONLY without exception, for the visual check on "
        "the cluster. The raw reconstruction tensor is NOT persisted"
    )

    summary = {
        "registration": phase13.STOP_2_BUILT,
        "prior_observation": phase13.SCUT_NORMAL_PRIOR,
        "checkpoint_epoch": epoch,
        "shakedown_selected_epoch": selected_epoch,
        "n_patients": len(manifest_ids),
        "l2_mean": float(np.mean([f["l2"] for f in per_patient])),
        "l2_worst": float(np.max([f["l2"] for f in per_patient])),
        "ssim_mean": float(np.mean(ssim_all)),
        "cohort_band_mean_mse": cohort_band_mean,
        "double_difference": (
            "NOT COMPUTED HERE -- stop 3's, with the bound readings "
            "applied where they fire"
        ),
        "review": (
            "PENDING -- the visual check on the CLUSTER-ONLY sheets, on "
            "the cluster, before stop 3"
        ),
    }
    with ctx.atomic("metrics.json", tier="SHAREABLE") as tmp:
        tmp.write_text(
            json.dumps(as_builtin(summary), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def _registered_reading(ctx, *, reading: str, joined: int, expected: int,
                        what: str) -> dict:
    """Apply a registered reading ONLY when its computation covered the
    rows it was registered against (phase13.READING_COUNT_GUARD).

    **[2026-08-24, the general fix for the auto-applied-sentence
    defect.]** Stop 3's (b) computed r over a join that produced ZERO
    rows and then printed the registered "the eye's impression was
    SAMPLING" sentence -- a reading about 237 patients applied to a
    computation over nobody. The sentence was not wrong about its own
    subject; it was applied to the wrong subject, silently, because
    nothing checked the count first.

    A reading's applies-when is part of the reading. So: joined must
    equal expected, or the sentence is REFUSED and the mismatch is
    reported in its place. Refusal is not failure -- the run continues
    and the numbers are still written; what is withheld is the
    INTERPRETATION.
    """
    if joined == expected:
        ctx.log(f"registered reading ({what}): {reading}")
        return {
            "applied": True, "joined": joined, "expected": expected,
            "reading": reading,
        }
    refusal = (
        f"READING REFUSED ({what}): the registered sentence applies to "
        f"{expected} rows and this computation joined {joined}. The "
        "count mismatch is the finding here, not the statistic -- fix "
        "the join and rerun before any reading is applied "
        "(phase13.READING_COUNT_GUARD)"
    )
    ctx.log(refusal)
    return {
        "applied": False, "joined": joined, "expected": expected,
        "reading_withheld": reading, "refusal": refusal,
    }


def _grades_for(ctx, declared, task, patient_ids) -> dict:
    """The median grade per patient, through the SHIPPED resolver.

    **[2026-08-24, the shared root of defects (b) and (c-ii)]**
    ``scoresheet.load_median`` is keyed by the sheet's FRONTAL PHOTO id,
    not the patient id, so joining it on patient ids produced zero rows
    in both tasks. ``median_by_patient`` already resolves patient ->
    frontal photo -> median through the manifest, and its own docstring
    says why it was extracted: "a second copy of a label resolution is
    how two runs quietly measure against different labels." This turn
    wrote a third. Both stop-3 tasks now go through the shipped one.
    """
    from .data.manifest import load_manifest

    rows = load_manifest(declared[task["manifest_artifact"]] / "manifest.csv")
    by_id = {int(row["patient_id"]): row for row in rows}
    sheet = (
        declared[task["scoresheet_artifact"]] / task["scoresheet_file"]
        if task.get("scoresheet_file")
        else declared[task["scoresheet_artifact"]]
    )
    ids = [int(p) for p in patient_ids]
    values = median_by_patient(by_id, sheet, ids)
    grades = {pid: int(value) for pid, value in zip(ids, values)}
    ctx.log(
        f"grades resolved for {len(grades)} of {len(ids)} patients via "
        "median_by_patient (patient -> frontal photo -> Median)"
    )
    return grades


def _band_error_by_grade(per_patient: list, grades: dict, bands) -> dict:
    """Per-grade means and the correlation of nose+lips band error with
    the median grade -- stop 3 check (b), shared so (b) and (c) group by
    grade identically (phase13.STOP_3_REGISTERED)."""
    paired = [
        (grades[f["patient_id"]],
         float(np.mean([f["band_mse"][b] for b in bands])))
        for f in per_patient if f["patient_id"] in grades
    ]
    by_grade: dict = {}
    for grade, value in paired:
        by_grade.setdefault(int(grade), []).append(value)
    xs = np.array([p[0] for p in paired], dtype=np.float64)
    ys = np.array([p[1] for p in paired], dtype=np.float64)
    correlation = (
        float(np.corrcoef(xs, ys)[0, 1])
        if len(paired) > 1 and xs.std() > 0 and ys.std() > 0
        else float("nan")
    )
    return {
        "n": len(paired),
        "bands": list(bands),
        "r_band_error_vs_grade": correlation,
        "mean_by_grade": {
            str(g): float(np.mean(v)) for g, v in sorted(by_grade.items())
        },
        "n_by_grade": {
            str(g): len(v) for g, v in sorted(by_grade.items())
        },
    }


def task_regional_comparison(ctx: RunContext) -> None:
    """Phase 13, stop 3 checks (a) and (b): the registered double
    difference, and band error against grade
    (phase13.STOP_3_REGISTERED).

    PURE ARITHMETIC on banked artifacts -- two JSON files written by runs
    that already happened, plus the score sheet. NO pixels, no model, no
    reconstruction: this task cannot regenerate a cohort image and
    therefore needs no gate.
    """
    import json

    from . import decoder as decoder_module
    from . import phase13

    task = ctx.config["task"]
    declared = {entry["name"]: Path(entry["path"]) for entry in ctx.inputs}
    cohort_run = declared[task["cohort_run"]]
    decoder_run = declared[task["decoder_run"]]

    cleft = json.loads(
        (cohort_run / "cleft_band_errors.json").read_text(encoding="utf-8")
    )
    scut = json.loads(
        (decoder_run / "scut_band_baseline.json").read_text(encoding="utf-8")
    )
    shakedown = json.loads(
        (decoder_run / "metrics.json").read_text(encoding="utf-8")
    )
    if int(cleft["checkpoint_epoch"]) != int(task["checkpoint_epoch"]):
        raise ValueError(
            f"the cohort run used epoch {cleft['checkpoint_epoch']}, this "
            f"comparison declares {task['checkpoint_epoch']} -- the "
            "numerator and the declaration must name one instrument"
        )

    # ---- (a) the double difference, as registered -----------------------
    result = decoder_module.double_difference(
        cleft["cohort_band_mean_mse"], scut["baseline_mse_by_band"]
    )
    comparability = shakedown.get("comparability") or {}
    caveats = [
        band for band, entry in comparability.items()
        if entry.get("caveat_attaches")
    ]
    ctx.log(
        "double difference "
        f"{result['double_difference']:+.4f} (relevant "
        f"{result['grader_relevant_mean']:.4f} vs irrelevant "
        f"{result['grade_irrelevant_mean']:.4f})"
    )
    ctx.log(
        "comparability caveats: "
        + (", ".join(caveats) if caveats else "none -- all ratios within 2x")
    )
    fires = result["double_difference"] > 0
    reading = (
        phase13.REGIONAL_COMPARISON_READINGS["reading_if_relevant_worse"]
        if fires else
        phase13.REGIONAL_COMPARISON_READINGS[
            "reading_if_no_difference_or_better"
        ]
    )

    # ---- (a) the EVERYWHERE gap: the larger fact, stated first ----------
    # [2026-08-24, phase13.EVERYWHERE_GAP_IS_THE_LARGER_FACT] Both band
    # ratios sit near 3.2: the cohort reconstructs ~3x worse than SCUT in
    # EVERY band, and the grader-relevant deficit rides on top of that as
    # a small relative difference. Reporting the double difference alone
    # would foreground a ~2% effect while silently omitting the ~200% one.
    relevant = result["grader_relevant_mean"]
    irrelevant = result["grade_irrelevant_mean"]
    everywhere = float(np.mean(list(result["normalised_ratio_by_band"].values())))
    relative_gap = float(
        (relevant - irrelevant) / irrelevant if irrelevant else float("nan")
    )
    ctx.log(
        f"THE LARGER FACT FIRST: every band reconstructs ~{everywhere:.2f}x "
        "worse than the SCUT baseline. The grader-relevant deficit is "
        f"{relative_gap * 100:+.1f}% RELATIVE to the irrelevant band, "
        "riding on top of that everywhere-gap"
    )

    # ---- (a) dispersion: the +0.0775 is a mean of 237 patient values ----
    per_patient_dd = []
    for entry in cleft["per_patient"]:
        try:
            one = decoder_module.double_difference(
                entry["band_mse"], scut["baseline_mse_by_band"]
            )
        except Exception:  # a band with no content pair -- reported, not fatal
            continue
        if np.isfinite(one["double_difference"]):
            per_patient_dd.append(one["double_difference"])
    dd_values = np.array(per_patient_dd, dtype=np.float64)
    rng = np.random.default_rng(int(ctx.config["seed"]))
    boots = (
        [
            float(np.mean(rng.choice(dd_values, size=len(dd_values),
                                     replace=True)))
            for _ in range(int(task.get("n_boot", 10000)))
        ]
        if len(dd_values) > 1 else []
    )
    dispersion = {
        "n_patients": int(len(dd_values)),
        "per_patient_mean": float(np.mean(dd_values)) if len(dd_values) else (
            float("nan")
        ),
        "per_patient_sd": float(np.std(dd_values, ddof=1)) if (
            len(dd_values) > 1
        ) else float("nan"),
        "per_patient_min": float(np.min(dd_values)) if len(dd_values) else (
            float("nan")
        ),
        "per_patient_max": float(np.max(dd_values)) if len(dd_values) else (
            float("nan")
        ),
        "fraction_positive": (
            float(np.mean(dd_values > 0)) if len(dd_values) else float("nan")
        ),
        "bootstrap_ci95": (
            [float(np.percentile(boots, 2.5)),
             float(np.percentile(boots, 97.5))] if boots else None
        ),
        "n_boot": int(task.get("n_boot", 10000)) if boots else 0,
        "status": (
            "DESCRIPTIVE dispersion, REPORT-NEVER-GATE: this interval "
            "enters no PLAN 4.3 condition and decides nothing"
        ),
    }
    ctx.log(
        f"dispersion: per-patient double difference mean "
        f"{dispersion['per_patient_mean']:+.4f}, sd "
        f"{dispersion['per_patient_sd']:.4f}, range "
        f"[{dispersion['per_patient_min']:+.4f}, "
        f"{dispersion['per_patient_max']:+.4f}], "
        f"{dispersion['fraction_positive'] * 100:.1f}% positive"
        + (
            f", bootstrap 95% CI [{dispersion['bootstrap_ci95'][0]:+.4f}, "
            f"{dispersion['bootstrap_ci95'][1]:+.4f}]"
            if dispersion["bootstrap_ci95"] else ""
        )
    )
    dd_reading = _registered_reading(
        ctx, reading=reading, joined=dispersion["n_patients"],
        expected=int(task["expect_patients"]), what="a, double difference",
    )
    ctx.log(
        "SCUT-normal prior, by name: "
        + phase13.SCUT_NORMAL_PRIOR["mitigation"]
    )

    # ---- (b) band error vs grade ----------------------------------------
    grades = _grades_for(
        ctx, declared, task,
        [entry["patient_id"] for entry in cleft["per_patient"]],
    )
    by_grade = _band_error_by_grade(
        cleft["per_patient"], grades, decoder_module.GRADER_RELEVANT
    )
    ctx.log(
        f"band error vs grade: r {by_grade['r_band_error_vs_grade']:+.4f} "
        f"over {by_grade['n']} patients; means "
        + ", ".join(
            f"{g}:{v:.5f}" for g, v in by_grade["mean_by_grade"].items()
        )
    )
    rises = by_grade["r_band_error_vs_grade"] > 0
    # The reading is applied ONLY if the join covered the cohort -- the
    # guard that the first run needed and did not have.
    band_reading = _registered_reading(
        ctx,
        reading=phase13.STOP_3_REGISTERED["b_band_error_vs_grade"][
            "reading_if_rises" if rises else "reading_if_flat"
        ],
        joined=by_grade["n"], expected=int(task["expect_patients"]),
        what="b, band error vs grade",
    )
    ctx.log(
        "registered limit (b): "
        + phase13.STOP_3_REGISTERED["b_band_error_vs_grade"][
            "registered_limit"
        ]
    )

    summary = {
        "registration": phase13.STOP_3_REGISTERED,
        "hypothesis_under_test": phase13.COARSE_SEVERITY_HYPOTHESIS,
        "checkpoint_epoch": int(task["checkpoint_epoch"]),
        "a_double_difference": result,
        "a_everywhere_gap": {
            "mean_ratio_all_bands": everywhere,
            "grader_relevant_mean": relevant,
            "grade_irrelevant_mean": irrelevant,
            "relevant_deficit_relative": relative_gap,
            "the_larger_fact": phase13.EVERYWHERE_GAP_IS_THE_LARGER_FACT[
                "must_be_stated_first"
            ],
        },
        "a_dispersion": dispersion,
        "a_comparability_caveats": caveats or "none -- all ratios within 2x",
        "a_reading": dd_reading,
        "a_scut_normal_prior": phase13.SCUT_NORMAL_PRIOR["mitigation"],
        "b_band_error_vs_grade": by_grade,
        "b_reading": band_reading,
        "b_registered_limit": (
            phase13.STOP_3_REGISTERED["b_band_error_vs_grade"][
                "registered_limit"
            ]
        ),
        "b_retraction_note": phase13.READING_APPLIED_TO_NOBODY["withdrawn"],
        "status": "DESCRIPTIVE throughout -- no ledger row, no paired claim",
    }
    with ctx.atomic("metrics.json", tier="SHAREABLE") as tmp:
        tmp.write_text(
            json.dumps(as_builtin(summary), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def _cohort_reconstructions(ctx, task, declared, decoder_module):
    """Regenerate the cohort reconstructions IN MEMORY, behind the gate.

    Shared by stop 3's (c) and (d) so the gate check, the instrument
    assertions and the no-persist boundary have ONE implementation
    (phase13.STOP_3_REGISTERED). Returns (reconstructions, patient_ids,
    references, geometry_rows, position_of) and writes NOTHING.
    """
    from . import embeddings as cleft_embeddings_module
    from .cluster_csv import read_cluster_csv
    from .data.manifest import load_manifest

    # ---- the governance gate, before any read ---------------------------
    if task.get("cleft_reconstruction_acknowledged") is not True:
        raise ValueError(
            "cleft_reconstruction_acknowledged is not set. This task "
            "REGENERATES cohort reconstructions in memory -- generated "
            "patient-derived images -- so it carries the same gate as "
            "the run that first wrote them, and that judgement belongs "
            "to the maintainer, not to code "
            "(phase13.CLEFT_RECONSTRUCTION_GATE)"
        )
    ctx.log(
        "cleft_reconstruction_acknowledged: true -- the maintainer's "
        "confirmation on record. Reconstructions are regenerated IN "
        "MEMORY and never written to disk"
    )

    manifest_dir = declared[task["manifest_artifact"]]
    staged_dir = declared[task["staged_artifact"]]
    rows = load_manifest(manifest_dir / "manifest.csv")
    if len(rows) != int(task["expect_patients"]):
        raise ValueError(
            f"{len(rows)} rows in {manifest_dir}, expected "
            f"{task['expect_patients']}"
        )
    patient_ids = [int(r["patient_id"]) for r in rows]

    values, metadata = cleft_embeddings_module.load(
        declared[task["cleft_embeddings"]], patient_ids
    )
    if metadata.get("init") != "imagenet" or (
        metadata.get("geometry") != task["geometry"]
    ):
        raise ValueError(
            f"embedding set is {metadata.get('init')}/"
            f"{metadata.get('geometry')}; the decoder reads its OWN "
            "representation and no other"
        )
    if values.ndim != 2 or values.shape[1] != decoder_module.EMBED_DIM:
        raise ValueError(
            f"values {values.shape}: expected the pooled "
            f"{decoder_module.EMBED_DIM}-d set"
        )

    references = np.load(staged_dir / "staged_patient_g1.npy")
    geometry_rows = read_cluster_csv(
        staged_dir / "geometry.csv", expect=("patient_id",)
    )
    position_of = {
        int(row["patient_id"]): i for i, row in enumerate(geometry_rows)
    }

    epoch = int(task["checkpoint_epoch"])
    checkpoint = (
        declared[task["decoder_run"]] / "checkpoints"
        / decoder_module.checkpoint_name(epoch)
    )
    if not checkpoint.is_file():
        raise ValueError(f"{checkpoint} does not exist")

    import torch

    torch.manual_seed(int(ctx.config["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = decoder_module.build(seed=int(ctx.config["seed"])).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()
    batch = int(task.get("batch_size", 32))
    chunks = []
    with torch.no_grad():
        for start in range(0, len(patient_ids), batch):
            block = torch.as_tensor(values[start:start + batch], device=device)
            chunks.append(
                model(block).permute(0, 2, 3, 1).cpu().numpy() * 255.0
            )
    ctx.log(
        f"regenerated {len(patient_ids)} reconstructions in memory from "
        f"epoch {epoch} -- nothing written"
    )
    return (
        np.concatenate(chunks), patient_ids, references, geometry_rows,
        position_of,
    )


def task_asymmetry_retention(ctx: RunContext) -> None:
    """Phase 13, stop 3 check (c): does the decoder retain asymmetry, and
    does retention separate severe from mild (phase13.STOP_3_REGISTERED)?

    The DIRECT test of the eye's claim (phase13.STOP_2_REVIEWED, whose
    figures are EYE-IMPRESSIONS). Reconstructions are regenerated in
    memory behind the gate and never written. The SCUT held-out
    reconstructions' own asymmetry is computed IDENTICALLY -- same
    function, same mask rule -- as the fully-normalised floor, because
    blur shrinks |left-right| everywhere and only the RELATIVE comparison
    survives.
    """
    import json

    from . import decoder as decoder_module
    from . import phase13
    from .geometry import mirror
    from .scut import labels as scut_labels
    from .train import pretrain

    task = ctx.config["task"]
    declared = {entry["name"]: Path(entry["path"]) for entry in ctx.inputs}
    bands = decoder_module.GRADER_RELEVANT

    (
        reconstructions, patient_ids, references, geometry_rows, position_of,
    ) = _cohort_reconstructions(ctx, task, declared, decoder_module)

    grades = _grades_for(ctx, declared, task, patient_ids)

    def band_mean(image, box, mask_source=None):
        # The mask comes from the REFERENCE for both members of a pair
        # (decoder.band_asymmetry's mask_source): measuring the two over
        # different pixel sets is defect (c-i), and a retention ratio
        # between different regions is not a ratio.
        values = [
            decoder_module.band_asymmetry(image, box, band, mask_source)
            for band in bands
        ]
        return float(np.nanmean(values))

    per_patient = []
    for i, patient in enumerate(patient_ids):
        row = geometry_rows[position_of[patient]]
        box = (
            int(row["content_x"]), int(row["content_y"]),
            int(row["content_w"]), int(row["content_h"]),
        )
        original = references[position_of[patient]]
        asym_orig = band_mean(original, box, original)
        asym_recon = band_mean(reconstructions[i], box, original)
        per_patient.append({
            "patient_id": patient,
            "asym_orig": asym_orig,
            "asym_recon": asym_recon,
            "retention": (
                float(asym_recon / asym_orig)
                if asym_orig and np.isfinite(asym_orig) and asym_orig > 0
                else float("nan")
            ),
            "grade": int(grades[patient]) if patient in grades else None,
        })
    # The mirror-symmetry property the whole measure rests on, asserted
    # rather than trusted -- mirror's own standing check (R3).
    symmetry_error = max(
        mirror.mirror_symmetry_error(references[position_of[p]])
        for p in patient_ids[:8]
    )

    originals = np.array([f["asym_orig"] for f in per_patient])
    recons = np.array([f["asym_recon"] for f in per_patient])
    usable = np.isfinite(originals) & np.isfinite(recons)

    # [2026-08-24, defect (c-i)] A correlation of exactly +0.0000 over 237
    # is a signature, not a measurement: the first run reported one and
    # nothing in the task could say whether an operand was constant, all
    # zeros, or nan-coerced. Both vectors are now DESCRIBED before they
    # are correlated, and a degenerate operand is named rather than
    # silently returned as a number.
    def describe(name, values):
        finite = values[np.isfinite(values)]
        return {
            "name": name, "n_finite": int(finite.size),
            "n_nan": int(values.size - finite.size),
            "mean": float(finite.mean()) if finite.size else float("nan"),
            "sd": float(finite.std(ddof=1)) if finite.size > 1 else (
                float("nan")
            ),
            "min": float(finite.min()) if finite.size else float("nan"),
            "max": float(finite.max()) if finite.size else float("nan"),
            "n_zero": int((finite == 0).sum()),
            "is_constant": bool(finite.size > 1 and finite.std() == 0),
        }

    vectors = [
        describe("asym_orig", originals), describe("asym_recon", recons),
    ]
    for entry in vectors:
        ctx.log(
            f"  {entry['name']}: n_finite {entry['n_finite']}, nan "
            f"{entry['n_nan']}, mean {entry['mean']:.6f}, sd "
            f"{entry['sd']:.6f}, range [{entry['min']:.6f}, "
            f"{entry['max']:.6f}], zeros {entry['n_zero']}"
            + (" -- CONSTANT" if entry["is_constant"] else "")
        )
    degenerate = [e["name"] for e in vectors if e["is_constant"] or (
        e["n_finite"] < 2
    )]
    r_orig_recon = (
        float(np.corrcoef(originals[usable], recons[usable])[0, 1])
        if usable.sum() > 1 and not degenerate else float("nan")
    )
    if degenerate:
        ctx.log(
            f"DEGENERATE OPERAND(S) {degenerate}: no correlation is "
            "reported, because a constant or empty vector has none. The "
            "vector description above is the finding"
        )
    graded = [f for f in per_patient if f["grade"] is not None]
    grade_values = np.array([f["grade"] for f in graded], dtype=np.float64)
    orig_graded = np.array([f["asym_orig"] for f in graded])
    anchor = (
        float(np.corrcoef(grade_values, orig_graded)[0, 1])
        if len(graded) > 1 and grade_values.std() > 0 else float("nan")
    )
    retention_by_grade: dict = {}
    for entry in graded:
        if np.isfinite(entry["retention"]):
            retention_by_grade.setdefault(entry["grade"], []).append(
                entry["retention"]
            )

    # ---- the SCUT floor: the same function on the shakedown side --------
    # [2026-08-24, defect (c-iii)] The first run computed this floor over
    # 237 faces -- the COHORT's count, crossed in from the wrong variable
    # via a `scut_floor_sample: 237` the generator wrote, and taken as
    # `sorted(...)[:237]`, which is an ALPHABETICAL subsample, not a
    # random one. The floor is now the WHOLE held-out side, asserted
    # against the declared count, so neither the size nor the selection
    # can drift again.
    scut_stems = sorted(
        scut_labels.read_labelled_split(declared[task["scut_root"]], "test")
    )
    if len(scut_stems) != int(task["expect_scut_test"]):
        raise ValueError(
            f"{len(scut_stems)} held-out SCUT faces, expected "
            f"{task['expect_scut_test']} -- the floor is defined over the "
            "WHOLE held-out side"
        )
    scut_images = pretrain.load_masked_features(
        declared[task["masked_scut"]], task["geometry"], scut_stems
    )
    scut_values, scut_all_stems, _ = decoder_module.load_scut_embeddings(
        declared[task["scut_embeddings"]]
    )
    scut_index = {stem: i for i, stem in enumerate(scut_all_stems)}
    epoch = int(task["checkpoint_epoch"])
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = decoder_module.build(seed=int(ctx.config["seed"])).to(device)
    model.load_state_dict(torch.load(
        declared[task["decoder_run"]] / "checkpoints"
        / decoder_module.checkpoint_name(epoch),
        map_location=device,
    ))
    model.eval()
    with torch.no_grad():
        scut_recon = model(torch.as_tensor(
            np.stack([scut_values[scut_index[s]] for s in scut_stems]),
            device=device,
        )).permute(0, 2, 3, 1).cpu().numpy() * 255.0
    scut_box = (
        0, 0, decoder_module.OUTPUT_SIZE, decoder_module.OUTPUT_SIZE
    )
    # Paired mask on this side too: the ORIGINAL's content pixels for
    # both members, identically to the cohort side.
    scut_orig = [band_mean(img, scut_box, img) for img in scut_images]
    scut_rec = [
        band_mean(rec, scut_box, orig)
        for rec, orig in zip(scut_recon, scut_images)
    ]
    scut_floor = float(np.nanmean([
        r / o for o, r in zip(scut_orig, scut_rec) if o and o > 0
    ]))

    by_grade = {
        str(g): float(np.mean(v))
        for g, v in sorted(retention_by_grade.items())
    }
    ctx.log(
        f"r(asym_orig, asym_recon) {r_orig_recon:+.4f} over "
        f"{int(usable.sum())}; anchor r(asym_orig, grade) {anchor:+.4f} "
        f"(Phase 4's index sits at {mirror.MEASURED_BASELINE['mean']} -- "
        "asymmetry predicts grade WEAKLY, so nobody over-reads these)"
    )
    ctx.log(
        "retention by grade: "
        + (", ".join(f"{g}:{v:.4f}" for g, v in by_grade.items())
           or "EMPTY -- no graded patient had a finite retention")
        + f"; SCUT fully-normalised floor {scut_floor:.4f} over ALL "
        f"{len(scut_stems)} held-out faces"
    )
    retention_reading = _registered_reading(
        ctx,
        reading=(
            "retention by grade is reported against the SCUT floor; "
            "severe-vs-mild is the surviving comparison "
            "(STOP_3_REGISTERED['c_asymmetry_retention'])"
        ),
        joined=sum(len(v) for v in retention_by_grade.values()),
        expected=int(task["expect_patients"]),
        what="c, retention by grade",
    )
    ctx.log(
        "blur caveat, registered: "
        + phase13.STOP_3_REGISTERED["c_asymmetry_retention"][
            "blur_caveat_and_its_answer"
        ]
    )

    summary = {
        "registration": phase13.STOP_3_REGISTERED["c_asymmetry_retention"],
        "hypothesis_under_test": phase13.COARSE_SEVERITY_HYPOTHESIS,
        "eye_impression_under_test": phase13.STOP_2_REVIEWED[
            "clinical_detail_EYE_IMPRESSION"
        ],
        "checkpoint_epoch": epoch,
        "bands": list(bands),
        "r_asym_orig_vs_recon": r_orig_recon,
        "vector_descriptions": vectors,
        "degenerate_operands": degenerate or "none",
        "retention_mean_by_grade": by_grade,
        "retention_reading": retention_reading,
        "n_by_grade": {
            str(g): len(v) for g, v in sorted(retention_by_grade.items())
        },
        "scut_reconstruction_floor": scut_floor,
        "scut_floor_n": len(scut_stems),
        "scut_floor_is_whole_held_out": True,
        "anchor_r_asym_orig_vs_grade": anchor,
        "anchor_reference": mirror.MEASURED_BASELINE["mean"],
        "mirror_symmetry_error_max": symmetry_error,
        "blur_caveat": phase13.STOP_3_REGISTERED["c_asymmetry_retention"][
            "blur_caveat_and_its_answer"
        ],
        "per_patient": per_patient,
        "status": "DESCRIPTIVE, report-never-gate",
    }
    with ctx.atomic("metrics.json", tier="SHAREABLE") as tmp:
        tmp.write_text(
            json.dumps(as_builtin(summary), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def task_extract_reconstruction_embeddings(ctx: RunContext) -> None:
    """Phase 13, stop 3 check (d), part 1: re-embed the 237
    reconstructions through the SAME frozen ViT-B/16
    (phase13.STOP_3_REGISTERED).

    The set is PATIENT-DERIVED -- reconstructions of patient images --
    and is CLUSTER-ONLY like everything else. The gate rides with the
    regeneration: this task rebuilds cohort pixels in memory, so it
    carries the same acknowledged check as the run that first wrote them.
    The probe itself is a SEPARATE run of the standard train_cv recipe on
    the set this writes, byte-identical to the 0.2520 arm's.
    """
    from . import decoder as decoder_module
    from . import embeddings as embeddings_module
    from . import phase13
    from .train.extract import extract_features

    task = ctx.config["task"]
    declared = {entry["name"]: Path(entry["path"]) for entry in ctx.inputs}

    reconstructions, patient_ids, _, _, _ = _cohort_reconstructions(
        ctx, task, declared, decoder_module
    )
    values, metadata = extract_features(
        task["backbone"], task["init"],
        np.clip(reconstructions, 0, 255).astype(np.uint8),
        checkpoint_path=None,
        batch_size=int(task.get("extract_batch_size", 32)),
    )
    out_dir = ctx.repo_root / "data" / "embeddings" / task["out_version"]
    # [2026-08-24, defect (d)] `embeddings.save` returns the artifact's
    # METADATA -- kind/backbone/patient_ids/shape -- and has no "rollup"
    # key; `decoder.save_scut_embeddings` returns a hash_dir payload,
    # which does. Reading one writer's shape off the other raised
    # KeyError AFTER the artifact was written, so the run died between
    # the save and its own bookkeeping (phase13.WRITER_DIED_AFTER_SAVE).
    # The rollup is now computed the way every other caller computes it.
    metadata_written = embeddings_module.save(
        out_dir, values,
        backbone=task["backbone"],
        backbone_kind="transformer",
        init=task["init"],
        geometry=task["geometry"],
        variant=None,
        checkpoint_sha256=None,
        patient_ids=patient_ids,
        manifest_ids=patient_ids,
    )
    from .provenance.hashing import hash_dir

    payload = hash_dir(out_dir)
    ctx.log(
        f"{task['out_version']}: {values.shape} from RECONSTRUCTIONS, "
        f"rollup {payload['rollup'][:8]} -- CLUSTER-ONLY, patient-derived"
    )
    import json

    with ctx.atomic("metrics.json", tier="SHAREABLE") as tmp:
        tmp.write_text(
            json.dumps(as_builtin({
                "registration": phase13.STOP_3_REGISTERED[
                    "d_grade_decodability"
                ],
                "artifact": task["out_version"],
                "rollup": payload["rollup"],
                "n_patients": len(patient_ids),
                "shape": list(values.shape),
                "kind_written": metadata_written["kind"],
                "extractor": metadata.get("backbone"),
                "tier": (
                    "CLUSTER-ONLY -- embeddings of generated "
                    "patient-derived images"
                ),
                "next": (
                    "the probe is a SEPARATE train_cv run on this set, "
                    "the 0.2520 arm's recipe byte-identical"
                ),
            }), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def _decoder_statistic_names() -> tuple:
    """The five non-anatomical statistic names, from their one home."""
    from . import decoder as decoder_module

    return tuple(decoder_module.GLOBAL_STATISTIC_NAMES)


def _confound_statistics(ctx, declared: dict, task: dict, patient_ids: list):
    """The five NON-ANATOMICAL global statistics, one row per patient.

    **[EXTRACTED 2026-08-31]** It was inline in
    ``task_confound_ceiling``. Phase 20's Arm C scores the SAME five
    against a different target (``phase20.ARM_C_REGISTERED``), and this
    project's own lesson -- ``median_by_patient``'s docstring, *"a second
    copy of a label resolution is how two runs quietly measure against
    different labels"* -- applies just as exactly to a second copy of a
    FEATURE construction. Two ceilings computed from two copies of these
    five would be uncomparable in a way nothing would flag.

    Returns ``(features, names)``; the row order is ``patient_ids``'.
    """
    from . import decoder as decoder_module
    from .cluster_csv import read_cluster_csv

    staged_dir = declared[task["staged_artifact"]]
    images = np.load(staged_dir / "staged_patient_g1.npy")
    geometry_rows = read_cluster_csv(
        staged_dir / "geometry.csv", expect=("patient_id",)
    )
    position_of = {
        int(row["patient_id"]): i for i, row in enumerate(geometry_rows)
    }
    names = list(decoder_module.GLOBAL_STATISTIC_NAMES)
    features = np.zeros((len(patient_ids), len(names)), dtype=np.float64)
    for i, patient in enumerate(patient_ids):
        row = geometry_rows[position_of[patient]]
        box = (
            int(row["content_x"]), int(row["content_y"]),
            int(row["content_w"]), int(row["content_h"]),
        )
        stats = decoder_module.global_statistics(images[position_of[patient]], box)
        features[i] = [stats[name] for name in names]
    ctx.log(
        f"{len(names)} non-anatomical statistics over {len(patient_ids)} "
        f"patients: {', '.join(names)} -- none of them can see a nose"
    )
    return features, names


def _confound_ceiling_oof(
    features, labels, patient_ids, fold_of, *,
    seeds, inner_val_frac, alpha, log,
) -> dict:
    """Per-seed out-of-fold PCC of a closed-form ridge over ``features``.

    **[EXTRACTED 2026-08-31]**, with ``_confound_statistics`` and for the
    same reason: Phase 20's Arm C is this recipe with one vector
    swapped, and a ceiling is only comparable to another ceiling when
    both came from the same recipe.

    The recipe's SHAPE follows the mirror arm's (5-fold on the manifest's
    fold column, inner-val split per seed, ridge), so the seed variance
    has the same source: which patients are held out, not initialisation.
    """
    from .eval import metrics
    from .train.ridge import RidgeBackbone

    folds = sorted({fold_of[p] for p in patient_ids})
    by_seed = {}
    for seed in seeds:
        predictions = np.zeros(len(patient_ids), dtype=float)
        rng = np.random.default_rng(seed)
        for fold in folds:
            test = [i for i, p in enumerate(patient_ids) if fold_of[p] == fold]
            train = [
                i for i, p in enumerate(patient_ids) if fold_of[p] != fold
            ]
            # The inner-val split the standard recipe holds out. The
            # solver is closed-form, so this does not select anything --
            # it is what makes the SEED move the fit at all, exactly as
            # in the mirror arm (PLAN 4.12.1: split variance).
            order = rng.permutation(len(train))
            n_inner = max(1, int(round(len(train) * float(inner_val_frac))))
            fit = [train[i] for i in order[n_inner:]]
            model = RidgeBackbone(alpha=float(alpha), seed=seed)
            model.reset(labels[fit])
            model.train_epoch(features[fit], labels[fit])
            predictions[test] = model.predict(features[test])
        by_seed[seed] = float(metrics.pcc(labels, predictions))
        log(f"  seed {seed}: OOF PCC {by_seed[seed]:+.4f}")
    return by_seed


def task_confound_ceiling(ctx: RunContext) -> None:
    """Phase 13 addendum P1: how much of the grade is predictable from
    NON-ANATOMICAL global image statistics alone
    (phase13.CLOSING_ADDENDUM_REGISTERED).

    POD ARITHMETIC: staged pixels and the geometry rows, five scalars per
    patient, a closed-form ridge over cleft_v1's own folds. No backbone,
    no GPU, no reconstruction -- and no gate, because nothing here
    generates or modifies a cohort pixel.

    **[2026-08-31] Two targets, one recipe.** Absent ``task.label`` this
    is Phase 13's measurement unchanged, against the MEDIAN grade. With
    ``task.label: mean`` it is Phase 20's **Arm C**, the like-for-like
    ceiling on the PANEL MEAN (``phase20.ARM_C_REGISTERED``) -- the same
    five statistics and the same ridge, so the two are comparable by
    construction rather than by assertion. Arm C is a NEW ARM, not a
    correction: Phase 13's +0.1000 is correct for its own quantity.
    """
    import json

    from . import phase13
    from .data.manifest import load_manifest

    task = ctx.config["task"]
    declared = {entry["name"]: Path(entry["path"]) for entry in ctx.inputs}

    rows = load_manifest(declared[task["manifest_artifact"]] / "manifest.csv")
    if len(rows) != int(task["expect_patients"]):
        raise ValueError(
            f"{len(rows)} manifest rows, expected {task['expect_patients']}"
        )
    patient_ids = [int(row["patient_id"]) for row in rows]
    fold_of = {int(row["patient_id"]): int(row["fold"]) for row in rows}

    # **[2026-08-31] The target is DECLARED, and its default does not
    # move.** Absent `label`, this is Phase 13's measurement exactly:
    # the MEDIAN grade through the shipped resolver. With `label`, it is
    # the named manifest column -- which is how Phase 20's Arm C scores
    # the same five statistics against the PANEL MEAN
    # (phase20.ARM_C_REGISTERED). One task, one recipe, two targets, and
    # neither can be mistaken for the other in a metrics.json.
    label_column = task.get("label")
    if label_column:
        if label_column not in rows[0]:
            raise ValueError(
                f"task.label {label_column!r} is not a manifest column; "
                f"available: {sorted(rows[0])}"
            )
        labels = np.array([float(row[label_column]) for row in rows], dtype=float)
        target = f"the manifest's {label_column!r} column (the panel mean)"
    else:
        grades = _grades_for(ctx, declared, task, patient_ids)
        labels = np.array([grades[p] for p in patient_ids], dtype=float)
        target = "the MEDIAN grade (run._grades_for)"
    ctx.log(f"target: {target}")

    features, names = _confound_statistics(ctx, declared, task, patient_ids)
    by_seed = _confound_ceiling_oof(
        features, labels, patient_ids, fold_of,
        seeds=[int(s) for s in task["seeds"]],
        inner_val_frac=float(task["inner_val_frac"]),
        alpha=float(task["alpha"]),
        log=ctx.log,
    )
    seeds = list(by_seed)

    values = np.array(list(by_seed.values()))
    mean_pcc = float(values.mean())
    sd_pcc = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    threshold = float(task["substantial_pcc"])
    substantial = mean_pcc >= threshold
    ctx.log(
        f"confound ceiling: mean OOF PCC {mean_pcc:+.4f} (sd {sd_pcc:.4f}) "
        f"over {len(seeds)} seeds, threshold {threshold}"
    )
    if label_column:
        # **Arm C applies PHASE 20's readings, not Phase 13's.** Phase
        # 13's two cells turn on whether a ceiling exists at all; Arm C's
        # turn on whether the TARGET DIFFERENCE MATTERS, which is a
        # different question and was committed before this number
        # (phase20.ARM_C_REGISTERED["readings"]). Firing Phase 13's cell
        # here would apply a registered reading to a quantity it was not
        # registered against -- the R2 shape, in the very task that
        # exists because of one.
        from . import phase20

        # The banked figure comes from a STRUCTURED constant, never
        # parsed out of the prose beside it: a comma or a re-wording in
        # a record must not be able to change a number a run reads.
        banked = float(phase13.P1_CEILING_FIGURES["pcc_mean"])
        moved = abs(mean_pcc - banked) >= float(task["materially_different"])
        registration = phase20.ARM_C_REGISTERED
        reading = _registered_reading(
            ctx,
            reading=registration["readings"][
                "materially_different" if moved else "near_0_1000"
            ],
            joined=len(patient_ids), expected=int(task["expect_patients"]),
            what="Phase 20 Arm C, the like-for-like ceiling",
        )
        ctx.log(
            f"Arm C {mean_pcc:+.4f} against Phase 13's median-target "
            f"{banked:+.4f}: difference {mean_pcc - banked:+.4f}, "
            f"threshold {task['materially_different']} -- "
            f"{'MATERIALLY DIFFERENT' if moved else 'near'}"
        )
        ctx.log("not a correction: " + registration["not_a_correction_to_phase_13"])
    else:
        reading = _registered_reading(
            ctx,
            reading=phase13.CLOSING_ADDENDUM_REGISTERED["p1_confound_ceiling"][
                "reading_if_substantial" if substantial
                else "reading_if_near_zero"
            ],
            joined=len(patient_ids), expected=int(task["expect_patients"]),
            what="P1, confound ceiling",
        )
        registration = phase13.CLOSING_ADDENDUM_REGISTERED["p1_confound_ceiling"]
        ctx.log("composes with: " + registration["composes_with"])

    summary = {
        "registration": registration,
        "statistics": names,
        "target": target,
        "label": label_column,
        "n_patients": len(patient_ids),
        "pcc_by_seed": {str(s): v for s, v in by_seed.items()},
        "pcc_mean": mean_pcc,
        "pcc_sd": sd_pcc,
        "substantial_threshold": threshold,
        "reading": reading,
        "status": (
            "DESCRIPTIVE, REPORT-NEVER-GATE: no ladder entry, no ledger "
            "row, no PLAN 4.3 machinery"
        ),
    }
    with ctx.atomic("metrics.json", tier="SHAREABLE") as tmp:
        tmp.write_text(
            json.dumps(as_builtin(summary), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def task_extract_occluded_embeddings(ctx: RunContext) -> None:
    """Phase 13 addendum P2: re-embed the cohort with ONE positional
    third blanked to white (phase13.CLOSING_ADDENDUM_REGISTERED).

    The occluded crops exist IN MEMORY only; what lands is the embedding
    set, which is patient-derived and CLUSTER-ONLY like every other. The
    probe on it is a SEPARATE run of A's recipe, byte-identical.

    **The gate is applied by INHERITANCE and caution, not because its
    registered trigger fires**: occlusion MODIFIES an existing patient
    crop rather than GENERATING a new image, so
    CLEFT_RECONSTRUCTION_GATE's own words ("the first GENERATED cohort
    pixel") do not strictly reach it. the instruction was that
    the gate rides wherever cohort pixels are touched, and the flag is
    already true and carried -- so the conservative reading costs
    nothing and the distinction is recorded rather than blurred.
    """
    import json

    from . import decoder as decoder_module
    from . import embeddings as embeddings_module
    from . import phase13
    from .cluster_csv import read_cluster_csv
    from .data.manifest import load_manifest
    from .provenance.hashing import hash_dir
    from .train.extract import extract_features

    task = ctx.config["task"]
    if task.get("cleft_reconstruction_acknowledged") is not True:
        raise ValueError(
            "cleft_reconstruction_acknowledged is not set. This task "
            "modifies and re-embeds cohort pixels in memory; the gate "
            "rides wherever cohort pixels are touched, and that "
            "judgement belongs to the maintainer, not to code "
            "(phase13.CLEFT_RECONSTRUCTION_GATE)"
        )
    ctx.log(
        "cleft_reconstruction_acknowledged: true -- the maintainer's "
        "confirmation on record. Occluded crops stay IN MEMORY; only "
        "the embedding set lands, CLUSTER-ONLY"
    )

    declared = {entry["name"]: Path(entry["path"]) for entry in ctx.inputs}
    staged_dir = declared[task["staged_artifact"]]
    band = task["occluded_band"]
    if band not in decoder_module.BANDS:
        raise ValueError(
            f"occluded_band {band!r} is not one of {decoder_module.BANDS}"
        )

    rows = load_manifest(declared[task["manifest_artifact"]] / "manifest.csv")
    if len(rows) != int(task["expect_patients"]):
        raise ValueError(
            f"{len(rows)} manifest rows, expected {task['expect_patients']}"
        )
    patient_ids = [int(row["patient_id"]) for row in rows]

    images = np.load(staged_dir / "staged_patient_g1.npy")
    geometry_rows = read_cluster_csv(
        staged_dir / "geometry.csv", expect=("patient_id",)
    )
    position_of = {
        int(row["patient_id"]): i for i, row in enumerate(geometry_rows)
    }
    occluded = np.zeros(
        (len(patient_ids),) + images.shape[1:], dtype=np.uint8
    )
    for i, patient in enumerate(patient_ids):
        row = geometry_rows[position_of[patient]]
        box = (
            int(row["content_x"]), int(row["content_y"]),
            int(row["content_w"]), int(row["content_h"]),
        )
        occluded[i] = decoder_module.occlude_band(
            images[position_of[patient]], box, band
        ).astype(np.uint8)
    ctx.log(
        f"occluded the {band!r} third (positional, band_rows) on "
        f"{len(patient_ids)} crops -- blanked to WHITE, the value the "
        "staging pad already uses"
    )

    values, metadata = extract_features(
        task["backbone"], task["init"], occluded,
        checkpoint_path=None,
        batch_size=int(task.get("extract_batch_size", 32)),
    )
    out_dir = ctx.repo_root / "data" / "embeddings" / task["out_version"]
    metadata_written = embeddings_module.save(
        out_dir, values,
        backbone=task["backbone"],
        backbone_kind="transformer",
        init=task["init"],
        geometry=task["geometry"],
        variant=None,
        checkpoint_sha256=None,
        patient_ids=patient_ids,
        manifest_ids=patient_ids,
    )
    payload = hash_dir(out_dir)
    ctx.log(
        f"{task['out_version']}: {values.shape}, rollup "
        f"{payload['rollup'][:8]} -- CLUSTER-ONLY, patient-derived"
    )
    with ctx.atomic("metrics.json", tier="SHAREABLE") as tmp:
        tmp.write_text(
            json.dumps(as_builtin({
                "registration": phase13.CLOSING_ADDENDUM_REGISTERED[
                    "p2_band_occlusion"
                ],
                "occluded_band": band,
                "artifact": task["out_version"],
                "rollup": payload["rollup"],
                "n_patients": len(patient_ids),
                "shape": list(values.shape),
                "kind_written": metadata_written["kind"],
                "extractor": metadata.get("backbone"),
                "tier": (
                    "CLUSTER-ONLY -- embeddings of modified patient images"
                ),
                "next": (
                    "the probe is a SEPARATE train_cv run on this set, "
                    "A's recipe byte-identical"
                ),
            }), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def task_extract_occluded_recon_embeddings(ctx: RunContext) -> None:
    """Phase 13 addendum P3: re-embed the RECONSTRUCTIONS with one
    positional third blanked to white (phase13.P3_TRIGGERED).

    ONE composition of two shipped pieces: _cohort_reconstructions (the
    gate-first regeneration, in memory, nothing persisted) and
    decoder.occlude_band (the same thirds P2 used on originals). The
    occluded reconstructions never land; what lands is the embedding
    set, patient-derived and CLUSTER-ONLY. The probe is a SEPARATE
    train_cv run, A's recipe byte-identical.

    The band-smearing caveat (CLOSING_ADDENDUM_REGISTERED
    ['p3_conditional_not_built']) rides in the metrics as registered.
    """
    import json

    from . import decoder as decoder_module
    from . import embeddings as embeddings_module
    from . import phase13
    from .provenance.hashing import hash_dir
    from .train.extract import extract_features

    task = ctx.config["task"]
    declared = {entry["name"]: Path(entry["path"]) for entry in ctx.inputs}
    band = task["occluded_band"]
    if band not in decoder_module.BANDS:
        raise ValueError(
            f"occluded_band {band!r} is not one of {decoder_module.BANDS}"
        )

    # Gate FIRST, inside the shared helper -- the same implementation
    # (c), (d) and stop 2 used; this task adds no second gate copy.
    (
        reconstructions, patient_ids, _references, geometry_rows,
        position_of,
    ) = _cohort_reconstructions(ctx, task, declared, decoder_module)

    occluded = np.zeros(
        (len(patient_ids),) + reconstructions.shape[1:], dtype=np.uint8
    )
    for i, patient in enumerate(patient_ids):
        row = geometry_rows[position_of[patient]]
        box = (
            int(row["content_x"]), int(row["content_y"]),
            int(row["content_w"]), int(row["content_h"]),
        )
        occluded[i] = decoder_module.occlude_band(
            np.clip(reconstructions[i], 0, 255).astype(np.uint8), box, band
        )
    ctx.log(
        f"occluded the {band!r} third on {len(patient_ids)} regenerated "
        "reconstructions -- in memory, nothing persisted but the set"
    )

    values, metadata = extract_features(
        task["backbone"], task["init"], occluded,
        checkpoint_path=None,
        batch_size=int(task.get("extract_batch_size", 32)),
    )
    out_dir = ctx.repo_root / "data" / "embeddings" / task["out_version"]
    metadata_written = embeddings_module.save(
        out_dir, values,
        backbone=task["backbone"],
        backbone_kind="transformer",
        init=task["init"],
        geometry=task["geometry"],
        variant=None,
        checkpoint_sha256=None,
        patient_ids=patient_ids,
        manifest_ids=patient_ids,
    )
    payload = hash_dir(out_dir)
    ctx.log(
        f"{task['out_version']}: {values.shape}, rollup "
        f"{payload['rollup'][:8]} -- CLUSTER-ONLY, patient-derived"
    )
    with ctx.atomic("metrics.json", tier="SHAREABLE") as tmp:
        tmp.write_text(
            json.dumps(as_builtin({
                "registration": phase13.P3_TRIGGERED,
                "band_smearing_caveat": (
                    phase13.CLOSING_ADDENDUM_REGISTERED[
                        "p3_conditional_not_built"
                    ]["pre_registered_caveat"]
                ),
                "occluded_band": band,
                "checkpoint_epoch": int(task["checkpoint_epoch"]),
                "artifact": task["out_version"],
                "rollup": payload["rollup"],
                "n_patients": len(patient_ids),
                "shape": list(values.shape),
                "kind_written": metadata_written["kind"],
                "extractor": metadata.get("backbone"),
                "tier": (
                    "CLUSTER-ONLY -- embeddings of occluded generated "
                    "patient-derived images"
                ),
                "next": (
                    "the probe is a SEPARATE train_cv run on this set, "
                    "A's recipe byte-identical"
                ),
            }), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def task_asymmetry_encoding_probe(ctx: RunContext) -> None:
    """Phase 13 post-closing check: does the EMBEDDING carry the
    mirror-scalar (phase13.ASYMMETRY_ENCODING_PROBE_REGISTERED)?

    A feature-space probe the decoder never touches -- which is the
    point: (c)'s pixel-space r +0.0082 cannot distinguish 'never
    encoded' from 'encoded but not rendered', and every pixel statistic
    shares that blindness. POD ARITHMETIC: the cached 768-d set, the
    BANKED asym_orig scalars (consumed from the stop-3 run's metrics,
    never recomputed), a closed-form ridge on cleft_v1's folds. No
    pixels, no model, NO GATE.
    """
    import json

    from . import embeddings as cleft_embeddings_module
    from . import phase13
    from .data.manifest import load_manifest
    from .eval import metrics
    from .train.ridge import RidgeBackbone

    task = ctx.config["task"]
    declared = {entry["name"]: Path(entry["path"]) for entry in ctx.inputs}

    rows = load_manifest(declared[task["manifest_artifact"]] / "manifest.csv")
    if len(rows) != int(task["expect_patients"]):
        raise ValueError(
            f"{len(rows)} manifest rows, expected {task['expect_patients']}"
        )
    patient_ids = [int(row["patient_id"]) for row in rows]
    fold_of = {int(row["patient_id"]): int(row["fold"]) for row in rows}

    banked = json.loads(
        (declared[task["asymmetry_run"]] / "metrics.json").read_text(
            encoding="utf-8"
        )
    )
    asym_of = {
        int(entry["patient_id"]): float(entry["asym_orig"])
        for entry in banked["per_patient"]
        if entry.get("asym_orig") is not None
        and np.isfinite(entry["asym_orig"])
    }
    absent = [p for p in patient_ids if p not in asym_of]
    if absent:
        raise ValueError(
            f"{len(absent)} patients missing a banked asym_orig, e.g. "
            f"{absent[:3]} -- the probe consumes the stop-3 run's own "
            "scalars and computes none"
        )
    targets = np.array([asym_of[p] for p in patient_ids], dtype=np.float64)

    values, metadata = cleft_embeddings_module.load(
        declared[task["cleft_embeddings"]], patient_ids
    )
    ctx.log(
        f"probe: {values.shape[1]}-d embeddings -> banked asym_orig "
        f"(mean {targets.mean():.6f}, sd {targets.std():.6f}) over "
        f"{len(patient_ids)} patients -- the decoder is nowhere in "
        "this path"
    )

    folds = sorted(set(fold_of.values()))
    seeds = [int(s) for s in task["seeds"]]
    by_seed = {}
    for seed in seeds:
        predictions = np.zeros(len(patient_ids), dtype=float)
        rng = np.random.default_rng(seed)
        for fold in folds:
            test = [i for i, p in enumerate(patient_ids) if fold_of[p] == fold]
            train = [
                i for i, p in enumerate(patient_ids) if fold_of[p] != fold
            ]
            order = rng.permutation(len(train))
            n_inner = max(
                1, int(round(len(train) * float(task["inner_val_frac"])))
            )
            fit = [train[i] for i in order[n_inner:]]
            model = RidgeBackbone(alpha=float(task["alpha"]), seed=seed)
            model.reset(targets[fit])
            model.train_epoch(values[fit], targets[fit])
            predictions[test] = model.predict(values[test])
        by_seed[seed] = float(metrics.pcc(targets, predictions))
        ctx.log(f"  seed {seed}: OOF PCC {by_seed[seed]:+.4f}")

    seed_values = np.array(list(by_seed.values()))
    mean_pcc = float(seed_values.mean())
    sd_pcc = float(seed_values.std(ddof=1)) if len(seed_values) > 1 else 0.0
    threshold = float(task["substantial_pcc"])
    substantial = mean_pcc >= threshold
    ctx.log(
        f"encoding probe: mean OOF PCC {mean_pcc:+.4f} (sd {sd_pcc:.4f}) "
        f"over {len(seeds)} seeds, declared threshold {threshold}"
        + (" -- AT THE BOUNDARY, say so with the spread (the P1 "
           "precedent)" if abs(mean_pcc - threshold) <= sd_pcc else "")
    )
    reading = _registered_reading(
        ctx,
        reading=phase13.ASYMMETRY_ENCODING_PROBE_REGISTERED[
            "reading_if_substantial" if substantial
            else "reading_if_near_zero"
        ],
        joined=len(patient_ids), expected=int(task["expect_patients"]),
        what="asymmetry encoding probe",
    )
    ctx.log(
        "limitation, riding: "
        + phase13.ASYMMETRY_ENCODING_PROBE_REGISTERED[
            "limitation_rides_either_way"
        ]
    )

    summary = {
        "registration": phase13.ASYMMETRY_ENCODING_PROBE_REGISTERED,
        "n_patients": len(patient_ids),
        "target": "banked asym_orig (stop-3 run), consumed not recomputed",
        "pcc_by_seed": {str(s): v for s, v in by_seed.items()},
        "pcc_mean": mean_pcc,
        "pcc_sd": sd_pcc,
        "substantial_threshold": threshold,
        "reading": reading,
        "limitation": phase13.ASYMMETRY_ENCODING_PROBE_REGISTERED[
            "limitation_rides_either_way"
        ],
        "status": (
            "DESCRIPTIVE, report-never-gate; the verdict amends "
            "PHASE_13_CLOSING dated, in place"
        ),
    }
    with ctx.atomic("metrics.json", tier="SHAREABLE") as tmp:
        tmp.write_text(
            json.dumps(as_builtin(summary), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def _mebeauty_landmarks(path):
    """Parse landmarks.csv in its MEASURED format, refusing surprises.

    **[2026-08-24, measured from the source after the first run's
    refusal]** The header is ``['', 'image', 'score', 'landmarks']`` --
    an unnamed index, the AUTHORS' absolute image path, a 1-10 score
    (a THIRD score surface, returned for the consistency report), and
    the coordinates PACKED in one quoted cell: 136 comma-separated
    integers = 68 (x, y) pairs, dlib-68 convention, with a trailing
    comma inside the cell. The first build assumed spread numeric
    columns and refused with the measured header -- as designed
    (phase15.STOP_1_FIRST_RUN).

    The discipline is unchanged: a header that is not the measured one,
    or a cell that does not parse to exactly 136 numerics after the
    trailing-empty strip, REFUSES with the evidence (the header, or the
    cell's first 60 characters) in the message.

    Returns ``({stem: (68, 2) points}, {stem: score})``. **[2026-08-24]
    Widened from x-only to full (x, y) when stop 2a needed the anchors'
    heights**: one parser, because two would be two places for a format
    change to be handled differently.
    """
    import csv

    expected_header = ["", "image", "score", "landmarks"]
    expected_count = 136  # 68 (x, y) pairs, measured from the source
    with open(path, encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        if header != expected_header:
            raise ValueError(
                f"{path}: header {header!r} is not the measured "
                f"{expected_header!r} -- the format moved; measure "
                "again before parsing (phase15.STOP_1_FIRST_RUN)"
            )
        xs_by_stem, score_by_stem = {}, {}
        for line_number, row in enumerate(reader, start=2):
            if len(row) != 4 or not row[1].strip():
                continue
            stem = Path(row[1].strip()).name
            cells = [c for c in row[3].split(",") if c.strip() != ""]
            try:
                values = [float(c) for c in cells]
            except ValueError:
                raise ValueError(
                    f"{path} row {line_number}: non-numeric landmark "
                    f"cell, first 60 chars: {row[3][:60]!r}"
                ) from None
            if len(values) != expected_count:
                raise ValueError(
                    f"{path} row {line_number}: {len(values)} numerics, "
                    f"expected {expected_count} (68 pairs). First 60 "
                    f"chars: {row[3][:60]!r}"
                )
            xs_by_stem[stem] = np.array(
                values, dtype=np.float64
            ).reshape(-1, 2)
            try:
                score_by_stem[stem] = float(row[2])
            except ValueError:
                score_by_stem[stem] = float("nan")
    if not xs_by_stem:
        raise ValueError(f"{path}: no landmark rows parsed")
    return xs_by_stem, score_by_stem


def _resize_nearest(image, size: int):
    """Nearest-neighbour resize to (size, size, 3) -- sheets only, no
    interpolation library, deterministic."""
    array = np.asarray(image)
    if array.ndim == 2:
        array = np.stack([array] * 3, axis=2)
    rows = (np.arange(size) * array.shape[0] // size).clip(
        0, array.shape[0] - 1
    )
    cols = (np.arange(size) * array.shape[1] // size).clip(
        0, array.shape[1] - 1
    )
    return array[rows][:, cols, :3]


def task_survey_mebeauty(ctx: RunContext) -> None:
    """Phase 15, stop 1: the MEBeauty survey (phase15.STOP_1_BUILT).

    PUBLIC DATA, SHAREABLE throughout, NO GATE -- no patient pixel is
    anywhere near this task. Inventory against the banked clone listing,
    the scores/ directory inventoried file by file with the means-carrier
    STATED, the frontal screen by the shipped landmarks (threshold
    declared in the YAML), the usable count banked with the
    scale-asymmetry restatement, and accept/reject sheets for the maintainer's
    eye. cropped_images/, geometric_features.csv and FaceNet features
    are inventoried and NOT consumed (exit criterion 3's boundary).
    """
    import csv
    import json

    from . import phase15
    from .geometry import render

    task = ctx.config["task"]
    declared = {entry["name"]: Path(entry["path"]) for entry in ctx.inputs}
    root = declared[task["mebeauty_root"]]

    # ---- inventory against the banked clone listing ---------------------
    expected_entries = (
        "original_images", "cropped_images", "landmarks.csv", "scores",
        "geometric_features.csv", "README.md",
    )
    missing = [e for e in expected_entries if not (root / e).exists()]
    if missing:
        raise ValueError(
            f"{root} is missing {missing} -- not the clone the survey "
            "was registered against (phase15.CLONE_CONTENTS)"
        )
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    originals = {
        path.name: path
        for path in (root / "original_images").rglob("*")
        if path.suffix.lower() in image_extensions
    }
    ctx.log(
        f"inventory: {len(originals)} images under original_images/ "
        f"(published {task['published_images']}; measured is the number "
        "that counts -- report, never gate)"
    )

    # ---- scores/: inventoried file by file, means-carrier STATED --------
    scores_report = []
    means_candidates = []
    for path in sorted((root / "scores").iterdir()):
        entry = {"file": path.name, "bytes": path.stat().st_size}
        if path.suffix.lower() in (".csv", ".txt"):
            with open(path, encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.reader(handle))
            entry["rows"] = max(0, len(rows) - 1)
            entry["columns"] = rows[0] if rows else []
            numeric_in_range = []
            if len(rows) > 1:
                for index in range(len(rows[0])):
                    values = []
                    for row in rows[1:200]:
                        if index < len(row):
                            try:
                                values.append(float(row[index]))
                            except ValueError:
                                break
                    if values and all(1.0 <= v <= 10.0 for v in values):
                        numeric_in_range.append(rows[0][index])
            entry["columns_in_1_10"] = numeric_in_range
            # Many in-range numeric columns = per-rater or distributional
            # data; exactly one = a mean carrier (CLONE_CONTENTS' first
            # survey item -- STATED, consumed by nothing yet).
            if len(numeric_in_range) == 1:
                means_candidates.append(path.name)
            entry["shape_reading"] = (
                "MEAN CARRIER (one 1-10 column)"
                if len(numeric_in_range) == 1 else
                "PER-RATER OR DISTRIBUTIONAL (multiple 1-10 columns)"
                if len(numeric_in_range) > 1 else
                "no 1-10 column detected"
            )
        scores_report.append(entry)
        ctx.log(f"  scores/{entry['file']}: {entry.get('shape_reading', 'binary')}")

    # ---- the frontal screen, threshold declared in the YAML -------------
    landmarks, landmark_scores = _mebeauty_landmarks(
        root / "landmarks.csv"
    )
    threshold = float(task["frontal_max_offset"])
    screen = []
    for name, points in landmarks.items():
        xs = points[:, 0]
        width = float(xs.max() - xs.min())
        offset = (
            abs(float(xs.mean()) - float((xs.max() + xs.min()) / 2.0))
            / width if width > 0 else float("nan")
        )
        stem = Path(name).name
        screen.append({
            "image": stem,
            "offset": offset,
            "accepted": bool(np.isfinite(offset) and offset <= threshold),
            "exists": stem in originals,
        })
    n_landmarked = len(screen)
    accepted = [s for s in screen if s["accepted"]]
    usable = [s for s in accepted if s["exists"]]
    frontal_fraction = len(accepted) / n_landmarked if n_landmarked else 0.0
    ctx.log(
        f"frontal screen: {len(accepted)}/{n_landmarked} pass the "
        f"declared centroid-offset threshold {threshold} "
        f"({frontal_fraction:.1%}); USABLE (accepted with an image "
        f"present) {len(usable)} -- the phase's first banked number"
    )
    scut_total = 5500
    ctx.log(
        "scale asymmetry, restated with the SCREENED count: usable "
        f"{len(usable)} vs SCUT {scut_total} ({len(usable)/scut_total:.2f}x)"
        f"; headline was 2,550 vs 5,500 (0.46x). Mechanism ii's "
        "inversion is binding either way (CANDIDATE_SURVEY_REGISTERED)"
    )

    # ---- accept/reject sheets for the visual check ------------------------
    rng = np.random.default_rng(int(ctx.config["seed"]))
    rejected = [s for s in screen if not s["accepted"] and s["exists"]]
    n_sheets = 0
    for kind, pool, wanted in (
        ("accept", usable, int(task["sheet_accept"])),
        ("reject", rejected, int(task["sheet_reject"])),
    ):
        chosen = list(rng.choice(
            len(pool), size=min(wanted, len(pool)), replace=False,
        )) if pool else []
        panels, labels = [], []
        for index in chosen:
            entry = pool[int(index)]
            image = render.load_image(originals[entry["image"]])
            panels.append(render.Panel(
                label=f"{entry['image']} offset {entry['offset']:.3f}",
                image=_resize_nearest(image, 224).astype(np.uint8),
            ))
            labels.append(panels[-1].label)
        if not panels:
            continue
        sheet = render.tile(panels, columns=4)
        n_sheets += 1
        target = ctx.path(
            f"mebeauty_{kind}_sheet.png", tier="SHAREABLE"
        )
        render.save_sheet(sheet, target, labels=labels, columns=4,
                          cell=(224, 224))
    ctx.log(
        f"{n_sheets} sheets (accept/reject samples) -- SHAREABLE, "
        "public data, for the visual check"
    )

    # ---- survey sub-findings (phase15.STOP_1_FIRST_RUN) -----------------
    # (a) universal-vs-crop: the README is SILENT, so the families are
    # distinguished by MEASUREMENT -- which image directory each file's
    # keys resolve against. (b) landmarks.csv's own score column is a
    # THIRD score surface; its consistency with each carrier is checked
    # and REPORTED, never gated.
    cropped_names = {
        path.name
        for path in (root / "cropped_images").rglob("*")
        if path.suffix.lower() in image_extensions
    }
    family_reports = []
    for entry in scores_report:
        if len(entry.get("columns_in_1_10", [])) != 1:
            continue
        import csv as _csv

        with open(root / "scores" / entry["file"], encoding="utf-8-sig",
                  newline="") as handle:
            rows_ = list(_csv.reader(handle))
        header_ = rows_[0]
        score_index = header_.index(entry["columns_in_1_10"][0])
        name_index = next(
            (i for i in range(len(header_)) if i != score_index
             and any(Path(r[i]).suffix.lower() in image_extensions
                     for r in rows_[1:20] if i < len(r))),
            None,
        )
        if name_index is None:
            family_reports.append({
                "file": entry["file"],
                "note": "no filename-like column found -- not joinable",
            })
            continue
        stems, diffs = 0, []
        in_original, in_cropped = 0, 0
        for row_ in rows_[1:]:
            if len(row_) <= max(score_index, name_index):
                continue
            stem = Path(row_[name_index].strip()).name
            in_original += stem in originals
            in_cropped += stem in cropped_names
            if stem in landmark_scores and np.isfinite(
                landmark_scores[stem]
            ):
                try:
                    diffs.append(
                        abs(float(row_[score_index])
                            - landmark_scores[stem])
                    )
                    stems += 1
                except ValueError:
                    pass
        family_reports.append({
            "file": entry["file"],
            "keys_in_original_images": in_original,
            "keys_in_cropped_images": in_cropped,
            "third_surface_joined": stems,
            "third_surface_mean_abs_diff": (
                float(np.mean(diffs)) if diffs else None
            ),
            "third_surface_max_abs_diff": (
                float(np.max(diffs)) if diffs else None
            ),
        })
        ctx.log(
            f"  family {entry['file']}: keys resolve original="
            f"{in_original} cropped={in_cropped}; third-surface join "
            f"{stems}, mean|diff| "
            + (f"{np.mean(diffs):.4f}" if diffs else "n/a")
            + " -- REPORT, NEVER GATE"
        )

    summary = {
        "registration": phase15.STOP_1_BUILT,
        "survey": phase15.CANDIDATE_SURVEY_REGISTERED["mechanisms"],
        "inventory": {
            "images_measured": len(originals),
            "images_published": int(task["published_images"]),
            "landmark_rows": n_landmarked,
            "not_consumed": (
                "cropped_images/, geometric_features.csv, "
                "FaceNet_512_features/ -- inventoried, not consumed "
                "(phase15.CLONE_CONTENTS)"
            ),
        },
        "scores_inventory": scores_report,
        "carrier_families": family_reports,
        "readme_silent_on_families": (
            "the README does not distinguish universal from crop score "
            "files; the families are distinguished by the MEASURED "
            "key-resolution above (phase15.STOP_1_FIRST_RUN)"
        ),
        "means_carrier_stated": (
            means_candidates or "NONE UNAMBIGUOUS -- see scores_inventory"
        ),
        "frontal_screen": {
            "threshold_declared": threshold,
            "statistic": "normalised centroid offset (STOP_1_BUILT)",
            "n_landmarked": n_landmarked,
            "n_accepted": len(accepted),
            "frontal_fraction": frontal_fraction,
            "usable_count": len(usable),
        },
        "scale_asymmetry": {
            "usable_vs_scut": (
                f"{len(usable)} vs {scut_total}"
            ),
            "headline_vs_scut": "2550 vs 5500",
            "binding_note": (
                "mechanism ii is DEAD AND INVERTED; every reading "
                "carries the win-strong / loss-ambiguous sentence"
            ),
        },
        "review": "PENDING -- the visual check on the sheets before stop 2",
    }
    with ctx.atomic("metrics.json", tier="SHAREABLE") as tmp:
        tmp.write_text(
            json.dumps(as_builtin(summary), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def task_verify_mebeauty_mapping(ctx: RunContext) -> None:
    """Phase 15, stop 2a: verify the dlib-68 mapping, then render sheets
    (phase15.STOP_2A_BUILT).

    The mapping is [LITERATURE] (phase15.STOP_2_RULINGS); these are
    SCUT's own three checks re-run on this dataset. **If a check fails
    the MAPPING reopens -- never the check**: the thresholds arrive from
    the config, this task compares against them, and nothing here tunes
    anything.

    The verdict reads LANDMARKS ONLY. Pixels are touched solely to
    render the review sheets, so a rendering problem cannot corrupt a
    measurement. PUBLIC DATA, SHAREABLE, no gate.
    """
    import csv
    import json

    from . import mebeauty
    from . import phase15
    from .geometry import render

    task = ctx.config["task"]
    declared = {entry["name"]: Path(entry["path"]) for entry in ctx.inputs}
    root = declared[task["mebeauty_root"]]

    # ---- the usable set, reproduced by the SAME screen as stop 1 --------
    points_by_stem, landmark_scores = _mebeauty_landmarks(
        root / "landmarks.csv"
    )
    threshold = float(task["frontal_max_offset"])
    usable = {}
    for stem, points in points_by_stem.items():
        xs = points[:, 0]
        width = float(xs.max() - xs.min())
        if width <= 0:
            continue
        offset = abs(float(xs.mean()) - float((xs.max() + xs.min()) / 2)) / width
        if offset <= threshold:
            usable[stem] = points
    if len(usable) != int(task["expect_usable"]):
        raise ValueError(
            f"{len(usable)} usable faces at threshold {threshold}, "
            f"expected {task['expect_usable']} -- stop 1's banked count "
            "(phase15.STOP_1_BANKED). The screen must reproduce exactly, "
            "or the two stops are measuring different sets"
        )
    ctx.log(
        f"usable set reproduced: {len(usable)} faces at the declared "
        f"{threshold} -- identical to stop 1's banked count"
    )

    # ---- the three checks, landmarks only -------------------------------
    report = mebeauty.verify_mapping(
        usable,
        midline_tolerance=float(task["midline_tolerance"]),
        eye_symmetry_tolerance=float(task["eye_symmetry_tolerance"]),
        corner_margin=float(task["corner_margin"]),
    )
    for index, stats in report["check_1_midline"].items():
        ctx.log(
            f"  check 1, landmark {index}: median |x| offset "
            f"{stats['median']:.5f} (p95 {stats['p95']:.5f}) against "
            f"{task['midline_tolerance']}"
        )
    ctx.log(
        f"  check 2: median eye imbalance "
        f"{report['check_2_eye_symmetry']['median']:.5f} against "
        f"{task['eye_symmetry_tolerance']}; image-left fraction "
        f"{report['check_2_image_left_fraction']:.4f} (the "
        "subject-right trap: must be ~1.0)"
    )
    ctx.log(
        f"  check 3: {report['check_3_corner_widest_fraction']:.4f} of "
        "faces have 48/54 as the widest symmetric pair"
    )
    ctx.log(
        f"MAPPING VERDICT: {'ALL CHECKS PASS' if report['all_pass'] else 'FAILED'}"
        + ("" if report["all_pass"] else
           " -- the MAPPING reopens; the check is not adjusted "
           "(phase15.STOP_2_RULINGS)")
    )

    # ---- the split's per-half usable counts, for 2b's config ------------
    split_counts = {}
    for half in ("train", "test"):
        path = root / "scores" / f"{half}_universal_scores.csv"
        with open(path, encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.reader(handle))
        header = rows[0]
        name_index = next(
            i for i in range(len(header))
            if any(
                Path(row[i]).suffix.lower() in {".jpg", ".jpeg", ".png"}
                for row in rows[1:20] if i < len(row)
            )
        )
        stems = {
            Path(row[name_index].strip()).name
            for row in rows[1:] if len(row) > name_index
        }
        split_counts[half] = {
            "rows": len(rows) - 1,
            "usable_in_half": sum(1 for s in usable if s in stems),
        }
        ctx.log(
            f"split {half}: {split_counts[half]['rows']} rows, "
            f"{split_counts[half]['usable_in_half']} usable -- declare "
            "this into 2b's config"
        )

    # ---- the mapping sheets: pixels, and only for the eye ---------------
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    originals = {
        path.name: path
        for path in (root / "original_images").rglob("*")
        if path.suffix.lower() in image_extensions
    }
    rng = np.random.default_rng(int(ctx.config["seed"]))
    stems = sorted(s for s in usable if s in originals)
    chosen = [
        stems[int(i)] for i in rng.choice(
            len(stems), size=min(int(task["sheet_faces"]), len(stems)),
            replace=False,
        )
    ] if stems else []
    panels, labels = [], []
    for stem in chosen:
        points = usable[stem]
        anchors = points[list(
            list(mebeauty.BROW_INDICES) + list(mebeauty.MOUTH_OUTER_INDICES)
            + list(mebeauty.MIDLINE_LANDMARKS) + list(mebeauty.MOUTH_CORNERS)
        )]
        panels.append(render.Panel(
            label=stem,
            image=_overlay_panel(
                render.load_image(originals[stem]), anchors, 224,
                midline=mebeauty.mirror_pair_midline(points),
            ),
        ))
        labels.append(stem)
    n_sheets = 0
    for start in range(0, len(panels), 16):
        chunk = panels[start:start + 16]
        sheet = render.tile(chunk, columns=4)
        n_sheets += 1
        target = ctx.path(
            f"mebeauty_mapping_sheet_{n_sheets:02d}.png", tier="SHAREABLE"
        )
        render.save_sheet(
            sheet, target, labels=labels[start:start + 16], columns=4,
            cell=(224, 224),
        )
        _write_label_sidecar(
            ctx, f"mebeauty_mapping_sheet_{n_sheets:02d}_labels.csv",
            labels[start:start + 16],
        )
    ctx.log(
        f"{n_sheets} mapping sheets: midline in red, proposed anchors in "
        "green, overlay drawn in DISPLAY space -- SHAREABLE, with a "
        "full-label sidecar beside each sheet"
    )

    summary = {
        "registration": phase15.STOP_2A_BUILT,
        "rulings": phase15.STOP_2_RULINGS,
        "usable_reproduced": len(usable),
        "mapping_report": report,
        "split_counts": split_counts,
        "declare_into_2b": (
            "split_counts above are the per-half usable counts 2b's "
            "config must declare"
        ),
        "review": (
            "PENDING -- the visual check on the mapping sheets before any "
            "pixel is staged"
        ),
    }
    with ctx.atomic("metrics.json", tier="SHAREABLE") as tmp:
        tmp.write_text(
            json.dumps(as_builtin(summary), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def _overlay_panel(image, points, size: int, midline=None):
    """A review panel with its overlay drawn in DISPLAY space.

    **[2026-08-24, phase15.SHEET_OVERLAY_DEFECT -- measured]** The first
    version drew marks at SOURCE resolution and then decimated the panel
    with nearest-neighbour sampling, so the overlay's survival depended
    on the source image's size. At 4000x6000 (119.jpg) the stride is
    17.9 x 26.8 pixels per sample: of ~1,539 marked pixels, FOUR
    survived, and the 3-pixel midline column survived only if a sample
    column happened to land on it -- 16.8% of positions, and 119.jpg's
    (~2134) fell between the sampled 2125 and 2142, so it vanished
    entirely. At a typical 500x600 the same code left 281 pixels and
    looked fine, which is why the failure read as a DATA defect on
    exactly the largest images.

    Resizing first makes the mark a constant fraction of the panel at
    any source resolution, which is what a review overlay must be.
    """
    array = np.asarray(image)
    if array.ndim == 2:
        array = np.stack([array] * 3, axis=2)
    array = array[:, :, :3]
    height, width = array.shape[:2]
    panel = np.array(_resize_nearest(array, size), copy=True).astype(np.uint8)
    if width <= 0 or height <= 0:
        return panel
    if midline is not None and np.isfinite(midline):
        column = int(round(float(midline) * size / width))
        if 0 <= column < size:
            panel[:, max(column - 1, 0):column + 2] = (255, 0, 0)
    for x, y in np.asarray(points, dtype=float):
        column = int(round(x * size / width))
        row = int(round(y * size / height))
        if 0 <= row < size and 0 <= column < size:
            panel[
                max(row - 2, 0):row + 3, max(column - 2, 0):column + 3
            ] = (0, 255, 0)
    return panel


def _write_label_sidecar(ctx, sheet_name: str, labels: list) -> None:
    """The FULL labels beside the sheet, so a name in a panel can always
    be looked up (phase15.SHEET_LABEL_DEFECT). The drawn label is
    trimmed to its cell; this is not."""
    import csv as _csv
    import io as _io

    buffer = _io.StringIO()
    writer = _csv.writer(buffer, lineterminator="\n")
    writer.writerow(["panel_index", "label"])
    for index, label in enumerate(labels):
        writer.writerow([index, label])
    with ctx.atomic(sheet_name, tier="SHAREABLE") as tmp:
        tmp.write_text(buffer.getvalue(), encoding="utf-8")


def _image_size(path):
    """(width, height) from the image HEADER -- no pixel is decoded."""
    from PIL import Image

    with Image.open(path) as handle:
        return handle.size


def task_screen_mebeauty_landmarks(ctx: RunContext) -> None:
    """Phase 15, stop 2a-ii: the LANDMARK-QUALITY screen
    (phase15.STOP_2A_II_BUILT).

    The pose screen is blind to detector failure -- a misplaced but
    SYMMETRIC cloud passes a centroid-offset test happily -- so faces
    with drifted, collapsed or foreign-frame landmark sets reached the
    accepted 1,519 and were caught only by eye on the mapping sheets.
    These five checks test what the eye tested, with every threshold
    declared in the config before the run.

    The verdict reads landmark geometry and image DIMENSIONS only (PIL
    headers, no decode). Pixels are decoded solely for the sheets.
    """
    import json

    from . import mebeauty
    from . import phase15
    from .geometry import render

    task = ctx.config["task"]
    declared = {entry["name"]: Path(entry["path"]) for entry in ctx.inputs}
    root = declared[task["mebeauty_root"]]

    points_by_stem, _scores = _mebeauty_landmarks(root / "landmarks.csv")
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    originals = {
        path.name: path
        for path in (root / "original_images").rglob("*")
        if path.suffix.lower() in image_extensions
    }

    # ---- reproduce stop 1's usable set, exactly ------------------------
    threshold = float(task["frontal_max_offset"])
    pose_offsets = {}
    usable = {}
    for stem, points in points_by_stem.items():
        xs = points[:, 0]
        width = float(xs.max() - xs.min())
        if width <= 0:
            pose_offsets[stem] = float("nan")
            continue
        offset = abs(float(xs.mean()) - float((xs.max() + xs.min()) / 2)) / width
        pose_offsets[stem] = offset
        if offset <= threshold:
            usable[stem] = points
    if len(usable) != int(task["expect_usable"]):
        raise ValueError(
            f"{len(usable)} usable faces, expected "
            f"{task['expect_usable']} -- stop 1's banked count. The "
            "screen must reproduce exactly or the stops are measuring "
            "different sets"
        )
    ctx.log(f"usable set reproduced: {len(usable)} (stop 1's banked count)")

    # ---- the quality screen, thresholds from the config ---------------
    span_band = (float(task["span_min"]), float(task["span_max"]))
    interocular_band = (
        float(task["interocular_min"]), float(task["interocular_max"])
    )
    bounds_margin = float(task["bounds_margin"])
    max_points_outside = int(task.get("bounds_max_points_outside") or 0)
    survivors, rejected, missing_image = {}, [], []
    failure_counts = {name: 0 for name in mebeauty.QUALITY_CHECKS}
    for stem, points in usable.items():
        if stem not in originals:
            missing_image.append(stem)
            continue
        report = mebeauty.quality_screen(
            points, _image_size(originals[stem]),
            span_band=span_band, interocular_band=interocular_band,
            bounds_margin=bounds_margin,
            bounds_max_points_outside=max_points_outside,
        )
        if report["passes"]:
            survivors[stem] = points
        else:
            rejected.append({
                "image": stem,
                "failed": report["failed"],
                "violation": report["checks"]["bounds"]["violation"],
            })
            for name in report["failed"]:
                failure_counts[name] += 1
    ctx.log(
        f"quality screen: {len(survivors)} survive of {len(usable)} "
        f"({len(survivors) / max(len(usable), 1):.1%}); "
        f"{len(rejected)} rejected, {len(missing_image)} had no image file"
    )
    for name, count in failure_counts.items():
        ctx.log(f"  failed {name}: {count}")

    # ---- the bounds violation, AS A DISTRIBUTION ----------------------
    # [2026-08-24, phase15.BOUNDS_VIOLATION_READINGS] The eye says most
    # of the bounds-rejects do not deserve rejection. Before any
    # tolerance is chosen, the VIOLATION ITSELF is measured: how many
    # points fall outside and by how much, absolutely and as a fraction
    # of the image. Reported as a histogram and quantiles, so the two
    # candidate populations -- a jaw grazing the border, and a foreign
    # coordinate frame -- are visible as SHAPES rather than collapsed
    # into one pass/fail count.
    bounds_rejects = [
        entry for entry in rejected if "bounds" in entry["failed"]
    ]
    violation_distribution = {"n_bounds_rejects": len(bounds_rejects)}
    if bounds_rejects:
        counts = np.array(
            [e["violation"]["n_points_outside"] for e in bounds_rejects],
            dtype=float,
        )
        px_max = np.array(
            [e["violation"]["overshoot_px_max"] for e in bounds_rejects],
            dtype=float,
        )
        frac_max = np.array(
            [
                e["violation"]["overshoot_fraction_max"]
                for e in bounds_rejects
            ],
            dtype=float,
        )
        frac_median = np.array(
            [
                e["violation"]["overshoot_fraction_median"]
                for e in bounds_rejects
            ],
            dtype=float,
        )

        def _quantiles(values):
            return {
                "min": float(values.min()),
                "p25": float(np.percentile(values, 25)),
                "median": float(np.median(values)),
                "p75": float(np.percentile(values, 75)),
                "p90": float(np.percentile(values, 90)),
                "max": float(values.max()),
            }

        # The points-outside histogram, exact for small counts and
        # bucketed above -- the shape the reading turns on.
        histogram = {}
        for bucket in ("1", "2", "3", "4", "5", "6-10", "11-20", "21-68"):
            if "-" in bucket:
                low, high = (int(v) for v in bucket.split("-"))
                histogram[bucket] = int(
                    ((counts >= low) & (counts <= high)).sum()
                )
            else:
                histogram[bucket] = int((counts == int(bucket)).sum())
        violation_distribution.update({
            "points_outside_histogram": histogram,
            "points_outside": _quantiles(counts),
            "overshoot_px_max": _quantiles(px_max),
            "overshoot_fraction_max": _quantiles(frac_max),
            "overshoot_fraction_median": _quantiles(frac_median),
            "reading_registered": (
                phase15.BOUNDS_VIOLATION_READINGS["reading_if_marginal"]
                + " || "
                + phase15.BOUNDS_VIOLATION_READINGS["reading_if_split"]
            ),
        })
        ctx.log(
            "bounds violation, AS A DISTRIBUTION over "
            f"{len(bounds_rejects)} rejects:"
        )
        ctx.log(
            "  points outside per face -- "
            + ", ".join(f"{k}:{v}" for k, v in histogram.items())
        )
        ctx.log(
            f"  points outside: median {np.median(counts):.0f}, p90 "
            f"{np.percentile(counts, 90):.0f}, max {counts.max():.0f}"
        )
        ctx.log(
            f"  max overshoot (px): median {np.median(px_max):.1f}, p90 "
            f"{np.percentile(px_max, 90):.1f}, max {px_max.max():.1f}"
        )
        ctx.log(
            "  max overshoot (fraction of image): median "
            f"{np.median(frac_max):.5f}, p90 "
            f"{np.percentile(frac_max, 90):.5f}, max {frac_max.max():.5f}"
        )
        ctx.log(
            "  NO TOLERANCE IS CHOSEN BY THIS RUN -- the readings are "
            "registered and the value is the maintainer's, declared in the "
            "YAML before any re-screen "
            "(phase15.BOUNDS_VIOLATION_READINGS)"
        )

    # ---- the named diagnosis: images that must explain themselves -----
    # 119.jpg passed a screen it should never have reached; whatever let
    # it through may not be unique to it, so the diagnosis is STRUCTURAL
    # (a declared list) rather than a one-off print.
    diagnoses = {}
    diagnosis_panels = []
    for name in task.get("diagnose_images") or []:
        requested = Path(name).name
        # **[2026-08-24, phase15.SHEET_LABEL_DEFECT] Match by SUBSTRING,
        # not equality.** Two of the three first-pass diagnoses came back
        # "no such file" because the names were read off sheet labels
        # that had been visually truncated -- so an exact-match diagnosis
        # answers a question about a filename that never existed. A
        # requested name now resolves to every stem containing it, and
        # the matches are reported, so a truncated label still reaches
        # the face it names.
        exact = requested if requested in points_by_stem else None
        needle = Path(requested).stem
        matches = (
            [exact] if exact else
            sorted(s for s in points_by_stem if needle and needle in s)
        )
        if not matches:
            matches = sorted(s for s in originals if needle and needle in s)
        diagnoses[requested] = {
            "requested": requested,
            "matched_exactly": bool(exact),
            "matches": matches,
            "n_matches": len(matches),
        }
        if not matches:
            ctx.log(
                f"DIAGNOSIS {requested}: NO MATCH -- neither a landmark "
                "row nor an image file contains this name"
            )
            continue
        diagnoses[requested]["per_match"] = {}
    for requested, record in diagnoses.items():
        for stem in record.get("matches", []):
            entry = {
                "landmark_row_present": stem in points_by_stem,
                "image_file_present": stem in originals,
                "pose_offset": pose_offsets.get(stem),
                "reached_usable_set": stem in usable,
            }
            if stem in points_by_stem:
                points = points_by_stem[stem]
                entry["coordinate_ranges"] = {
                    "x_min": float(points[:, 0].min()),
                    "x_max": float(points[:, 0].max()),
                    "y_min": float(points[:, 1].min()),
                    "y_max": float(points[:, 1].max()),
                }
                entry["n_points"] = int(points.shape[0])
                entry["all_zero"] = bool(not points.any())
                entry["n_distinct_points"] = int(
                    len(np.unique(points, axis=0))
                )
            if stem in originals:
                size = _image_size(originals[stem])
                entry["image_size"] = list(size)
                if stem in points_by_stem:
                    entry["quality"] = mebeauty.quality_screen(
                        points_by_stem[stem], size,
                        span_band=span_band,
                        interocular_band=interocular_band,
                        bounds_margin=bounds_margin,
                    )
                    entry["in_bounds_rejects"] = bool(
                        "bounds" in entry["quality"]["failed"]
                    )
                # The panel itself, as the renderer fix's EVIDENCE: a
                # named face that rendered blank must come back visible
                # (phase15.SHEET_OVERLAY_DEFECT).
                diagnosis_panels.append((stem, points_by_stem.get(stem)))
            record["per_match"][stem] = entry
            ctx.log(
                f"DIAGNOSIS {requested} -> {stem}: "
                f"row={entry['landmark_row_present']} "
                f"image={entry['image_file_present']} "
                f"pose_offset={entry.get('pose_offset')} "
                f"reached_usable={entry['reached_usable_set']} "
                + (
                    f"ranges x[{entry['coordinate_ranges']['x_min']:.0f},"
                    f"{entry['coordinate_ranges']['x_max']:.0f}] "
                    f"y[{entry['coordinate_ranges']['y_min']:.0f},"
                    f"{entry['coordinate_ranges']['y_max']:.0f}] "
                    if "coordinate_ranges" in entry else ""
                )
                + (
                    f"image_size={entry['image_size']} "
                    if "image_size" in entry else ""
                )
                + (
                    f"quality_failed={entry['quality']['failed']}"
                    if "quality" in entry else ""
                )
            )

    # ---- the scale asymmetry, restated at the surviving count ---------
    scut_total = 5500
    # [2026-08-24] The price-of-quality sentence is REGISTERED for a
    # loss, so it fires only where there is one: p15-screen-landmarks-4
    # printed "the falling count..." at a run where 1,519 of 1,519
    # survived and nothing fell. A registered sentence that prints
    # where its condition does not hold is the auto-applied-reading
    # defect in miniature (phase13.READING_COUNT_GUARD).
    lost = len(usable) - len(survivors)
    ctx.log(
        f"scale asymmetry restated: {len(survivors)} vs SCUT {scut_total} "
        f"= {len(survivors) / scut_total:.2f}x (was 0.28x at 1,519, 0.46x "
        "at the 2,550 headline)"
        + (
            f". The count fell by {lost} -- THE PRICE OF QUALITY, "
            "recorded as the 0.08 band's ~5% was"
            if lost > 0 else
            ". NOTHING FELL at this tolerance, so the price-of-quality "
            "sentence does not apply and is not printed"
        )
    )

    # ---- sheets: survivors and the newly rejected ---------------------
    rng = np.random.default_rng(int(ctx.config["seed"]))
    n_sheets = 0
    for kind, pool, wanted in (
        ("survivor", sorted(survivors), int(task["sheet_survivors"])),
        (
            "rejected", [entry["image"] for entry in rejected],
            int(task["sheet_rejected"]),
        ),
    ):
        if not pool:
            continue
        chosen = [
            pool[int(i)] for i in rng.choice(
                len(pool), size=min(wanted, len(pool)), replace=False,
            )
        ]
        panels, labels = [], []
        for stem in chosen:
            panel = _overlay_panel(
                render.load_image(originals[stem]), points_by_stem[stem], 224,
            )
            label = stem
            if kind == "rejected":
                failed = next(
                    e["failed"] for e in rejected if e["image"] == stem
                )
                label = f"{stem} [{'+'.join(failed)}]"
            panels.append(render.Panel(label=label, image=panel))
            labels.append(label)
        for start in range(0, len(panels), 16):
            chunk = panels[start:start + 16]
            sheet = render.tile(chunk, columns=4)
            n_sheets += 1
            target = ctx.path(
                f"mebeauty_quality_{kind}_sheet_{n_sheets:02d}.png",
                tier="SHAREABLE",
            )
            render.save_sheet(
                sheet, target, labels=labels[start:start + 16], columns=4,
                cell=(224, 224),
            )
            _write_label_sidecar(
                ctx,
                f"mebeauty_quality_{kind}_sheet_{n_sheets:02d}_labels.csv",
                labels[start:start + 16],
            )
    ctx.log(
        f"{n_sheets} sheets: survivors and newly-rejected (rejection "
        "reasons in the labels) -- SHAREABLE, for the second eye pass"
    )

    if diagnosis_panels:
        panels = [
            render.Panel(
                label=stem,
                image=_overlay_panel(
                    render.load_image(originals[stem]),
                    points if points is not None else np.zeros((0, 2)),
                    224,
                    midline=(
                        mebeauty.mirror_pair_midline(points)
                        if points is not None else None
                    ),
                ),
            )
            for stem, points in diagnosis_panels
        ]
        target = ctx.path(
            "mebeauty_diagnosis_sheet.png", tier="SHAREABLE"
        )
        render.save_sheet(
            render.tile(panels, columns=4),
            target, labels=[p.label for p in panels], columns=4,
            cell=(224, 224),
        )
        _write_label_sidecar(
            ctx, "mebeauty_diagnosis_sheet_labels.csv",
            [p.label for p in panels],
        )
        ctx.log(
            f"diagnosis sheet: {len(panels)} named faces, overlaid in "
            "display space -- the renderer fix's own evidence"
        )

    summary = {
        "registration": phase15.STOP_2A_II_BUILT,
        "usable_reproduced": len(usable),
        "survivors": len(survivors),
        "rejected": len(rejected),
        "rejected_detail": rejected,
        "no_image_file": missing_image,
        "failure_counts": failure_counts,
        "bounds_violation_distribution": violation_distribution,
        "thresholds_declared": {
            "span_band": list(span_band),
            "interocular_band": list(interocular_band),
            "bounds_margin": bounds_margin,
            "bounds_max_points_outside": max_points_outside,
            "frontal_max_offset": threshold,
        },
        "named_diagnoses": diagnoses,
        "scale_asymmetry": {
            "survivors_vs_scut": f"{len(survivors)} vs {scut_total}",
            "ratio": len(survivors) / scut_total,
            "was_at_1519": 0.28,
            "was_at_headline": 0.46,
            "the_price": (
                "the falling count is THE PRICE OF QUALITY, recorded as "
                "the 0.08 band's ~5% was -- not a defect to be argued "
                "down"
            ),
        },
        "review": (
            "PENDING -- the second eye pass on the survivor and "
            "rejected sheets before any staging"
        ),
    }
    with ctx.atomic("metrics.json", tier="SHAREABLE") as tmp:
        tmp.write_text(
            json.dumps(as_builtin(summary), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def task_stage_mebeauty(ctx: RunContext) -> None:
    """Phase 15, stop 2b: stage the usable MEBeauty faces at G1 and G2
    (phase15.STOP_2B_BUILT).

    **The composition is SCUT's, verbatim**: ``masked.build_one`` does
    the crop, the arrival mask, the staging and the G2 unwarp. The only
    MEBeauty-specific step is the content box, whose ANCHORS come from
    the dlib-68 convention and whose FORMULA is the shared
    ``placement.box_from_anchors``. Levelling stays OFF and the AR
    parity sampling is CLEFT_AR's, both INHERITED rather than restated.

    STAGES ALL survivors; the split column records which half each face
    belongs to so pretraining can select the split-covered subset
    without a second staging pass. PUBLIC DATA, SHAREABLE, no gate.

    Nothing is embedded here: the staged sheets go to the eye first.
    """
    import csv
    import io as _io
    import json
    import shutil

    from . import mebeauty
    from . import phase15
    from .geometry import render
    from .provenance import atomic_write_text
    from .provenance.hashing import hash_dir
    from .scut import masked as masked_module

    task = ctx.config["task"]
    declared = {entry["name"]: Path(entry["path"]) for entry in ctx.inputs}
    root = declared[task["mebeauty_root"]]

    points_by_stem, _scores = _mebeauty_landmarks(root / "landmarks.csv")
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    originals = {
        path.name: path
        for path in (root / "original_images").rglob("*")
        if path.suffix.lower() in image_extensions
    }

    # ---- the surviving set, reproduced by BOTH declared screens -------
    threshold = float(task["frontal_max_offset"])
    span_band = (float(task["span_min"]), float(task["span_max"]))
    interocular_band = (
        float(task["interocular_min"]), float(task["interocular_max"])
    )
    bounds_margin = float(task["bounds_margin"])
    max_points_outside = int(task["bounds_max_points_outside"])
    # **[2026-08-24] EVERY config read happens HERE, before anything is
    # created or destroyed** (phase15.KEY_MISMATCH_DIAGNOSED). The v2
    # run died on `task["geometries"]` -- a key the config had renamed
    # to `variants` -- AFTER the incomplete-directory removal. Reading
    # the config first makes a key mismatch fail before any side
    # effect: validate, then act. It is hygiene rather than damage
    # control (no artifact was ever at risk: the removal targets this
    # task's own <version>.inprogress), and hygiene is still worth
    # having.
    size = int(task.get("size", masked_module.OUTPUT_SIZE))
    variants = tuple(task["variants"])
    out_version = task["out_version"]
    sheet_faces = int(task["sheet_faces"])
    survivors = []
    for stem in sorted(points_by_stem):
        if stem not in originals:
            continue
        points = points_by_stem[stem]
        xs = points[:, 0]
        width = float(xs.max() - xs.min())
        if width <= 0:
            continue
        offset = abs(float(xs.mean()) - float((xs.max() + xs.min()) / 2)) / width
        if offset > threshold:
            continue
        report = mebeauty.quality_screen(
            points, _image_size(originals[stem]),
            span_band=span_band, interocular_band=interocular_band,
            bounds_margin=bounds_margin,
            bounds_max_points_outside=max_points_outside,
        )
        if report["passes"]:
            survivors.append(stem)
    if len(survivors) != int(task["expect_survivors"]):
        raise ValueError(
            f"{len(survivors)} survivors, expected "
            f"{task['expect_survivors']} -- the declared screens must "
            "reproduce the banked count or this stages a different set"
        )
    ctx.log(
        f"survivors reproduced: {len(survivors)} at the declared "
        f"tolerance ({max_points_outside} points, margin {bounds_margin})"
    )

    # ---- the SHIPPED split, restricted to the survivors ---------------
    split_of = {}
    for half in ("train", "test"):
        path = root / "scores" / f"{half}_universal_scores.csv"
        with open(path, encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.reader(handle))
        header = rows[0]
        name_index = next(
            i for i in range(len(header))
            if any(
                Path(row[i]).suffix.lower() in image_extensions
                for row in rows[1:20] if i < len(row)
            )
        )
        for row in rows[1:]:
            if len(row) > name_index:
                split_of[Path(row[name_index].strip()).name] = half
    counts = {
        "train": sum(1 for s in survivors if split_of.get(s) == "train"),
        "test": sum(1 for s in survivors if split_of.get(s) == "test"),
    }
    counts["uncovered"] = len(survivors) - counts["train"] - counts["test"]
    for half in ("train", "test"):
        if counts[half] != int(task[f"expect_{half}"]):
            raise ValueError(
                f"split {half}: {counts[half]} usable, expected "
                f"{task[f'expect_{half}']} -- the shipped split must "
                "cover exactly the measured faces"
            )
    ctx.log(
        f"shipped split over survivors: train {counts['train']}, test "
        f"{counts['test']}, UNCOVERED {counts['uncovered']} -- the "
        "uncovered faces are staged and carry split='uncovered' so "
        "pretraining selects without a second staging pass "
        "(phase15.STOP_2A_BANKED's count gap)"
    )

    # ---- stage, one face resident, into <version>.inprogress ----------
    version = out_version
    final_dir = ctx.repo_root / "data" / "mebeauty_staged" / version
    if final_dir.exists():
        raise ValueError(
            f"{final_dir} already exists. Data artifacts are immutable: "
            "create a new version, never overwrite"
        )
    staging_dir = final_dir.with_suffix(".inprogress")
    if staging_dir.exists():
        # [2026-08-24] LOUD, because it deletes hundreds of megabytes.
        # An .inprogress directory is by definition incomplete: it holds
        # tensors with no ids, no split column and no MANIFEST, so it
        # CANNOT be resumed -- nothing in it says which row is which
        # face. Restaging costs ~20s, so the honest move is to remove
        # and rebuild rather than invent a resume for a cheap step
        # (phase15.PARITY_NONE_DIAGNOSED['inprogress_disposition']).
        ctx.log(
            f"removing an incomplete {staging_dir.name}: it holds only "
            "tensors with no ids and no MANIFEST, so it cannot be "
            "resumed -- restaging instead"
        )
        shutil.rmtree(staging_dir)
    staging_dir.mkdir(parents=True)

    ratios = masked_module.sample_aspect_ratios(
        len(survivors), int(ctx.config["seed"]),
        mode=task.get("ar_sampling", masked_module.DEFAULT_AR_SAMPLING),
    )
    # **[2026-08-24] The artifact takes the SCUT masked-artifact FORMAT**
    # -- faces.json plus masked_<variant>.npy -- so that
    # ``pretrain.load_masked_features`` reads it UNCHANGED. That loader
    # exists because "a positional read would train every face against
    # another face's label and still produce a plausible number"; a
    # second loader for MEBeauty would be a second place for exactly
    # that bug to live (phase15.ARTIFACT_FORMAT_IS_SCUTS).
    #
    # ``original`` is a VARIANT alongside g1/g2: the whole image through
    # the FROZEN stage(), no landmarks, no trapezium, no content box --
    # byte-identically what SCUT's original arm stages
    # (pretrain.load_original_features). It is stored as an artifact
    # rather than read live because MEBeauty's images are nested, mixed
    # -extension and non-square, none of which SCUT's path expects.
    arrays = {
        variant: np.lib.format.open_memmap(
            staging_dir / f"masked_{variant}.npy", mode="w+",
            dtype=np.uint8, shape=(len(survivors), size, size, 3),
        )
        for variant in variants
    }
    from .geometry.staging import stage as frozen_stage

    records, rows, original_pads = [], [], []
    for index, stem in enumerate(survivors):
        image = render.load_image(originals[stem])
        points = points_by_stem[stem]
        box = mebeauty.crop_box(points, float(ratios[index]))
        masked_variants = [v for v in variants if v in ("g1", "g2")]
        for geometry in masked_variants:
            staged, record = masked_module.build_one(
                image, points, float(ratios[index]), geometry,
                size=size, box=box,
            )
            arrays[geometry][index] = staged
            if geometry == masked_variants[0]:
                record["stem"] = stem
                record["split"] = split_of.get(stem, "uncovered")
                records.append(record)
                rows.append({
                    "stem": stem,
                    "split": record["split"],
                    "aspect_ratio": record.get("aspect_ratio"),
                    "aspect_ratio_requested": record.get(
                        "aspect_ratio_requested", float(ratios[index])
                    ),
                    "pad_fraction": record.get("pad_fraction"),
                    "white_fraction": record.get("white_fraction"),
                    "box_inside_frame": record.get("box_inside_frame"),
                })
        if "original" in variants:
            # The WHOLE image through the frozen stage -- SCUT's original
            # arm, byte-identically. **MEASURED, not assumed**: SCUT's
            # sources are 350x350 so stage() is a pure resize there and
            # no pad appears; MEBeauty's are in-the-wild and non-square,
            # so stage() PADS to square first. The pad fraction is
            # measured per face and reported, because it is a difference
            # between mebeauty_original and scut_original that the
            # "no padding" reading did not anticipate
            # (phase15.ORIGINAL_ARM_PADS_TOO).
            staged_original = frozen_stage(image, size=size)
            arrays["original"][index] = staged_original.image
            original_pads.append(float(staged_original.pad_fraction))
        if (index + 1) % 200 == 0:
            ctx.log(f"  staged {index + 1}/{len(survivors)}")
    for array in arrays.values():
        array.flush()
    ctx.log(
        f"STAGING COMPLETE: {len(survivors)} faces at "
        f"{len(variants)} variants. Everything after this line is "
        "REPORTING -- if the run fails below, the pixels were made and "
        "the report was not"
    )

    # ---- parity, re-run on the NEW artifact ---------------------------
    # [2026-08-24, phase15.PARITY_NONE_DIAGNOSED] Report every DEFINED
    # gap and NAME the undefined ones. The first version formatted
    # median_gap for every key; pad_fraction's is None by design (the
    # cleft reference records mean for it, median for aspect_ratio), so
    # the whole line raised -- and because the join is evaluated before
    # ctx.log, NOTHING printed and the reader saw only a traceback.
    try:
        parity = masked_module.parity_report(records)
    except Exception as error:
        ctx.log(
            "WORK COMPLETED, REPORT FAILED: the staging finished and "
            f"the parity report raised {type(error).__name__}: {error}. "
            f"The incomplete {staging_dir.name} survives and is NOT "
            "reusable (tensors only, no ids, no MANIFEST); the next run "
            "removes it and restages, which costs ~20s"
        )
        raise
    for key, value in parity.get("gaps", {}).items():
        defined = [
            f"{statistic} gap {value[statistic]:+.4f}"
            for statistic in ("median_gap", "mean_gap")
            if value.get(statistic) is not None
        ]
        ctx.log(
            f"parity {key}: "
            + (", ".join(defined) if defined else "NO GAP IS DEFINED")
            + " -- gaps against CLEFT_STAGED_GEOMETRY, no verdict"
        )
        for statistic, reason in (value.get("undefined") or {}).items():
            ctx.log(f"  {statistic} UNDEFINED: {reason}")
    inside = sum(1 for r in rows if r["box_inside_frame"])
    ctx.log(
        f"content box inside the source frame for {inside}/{len(rows)} "
        "faces -- measured and reported, never silently padded"
    )

    with _io.StringIO() as buffer:
        writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        atomic_write_text(staging_dir / "faces.csv", buffer.getvalue())
    # faces.json is the INDEX the frozen loader reads -- same name, same
    # shape, same stem-keyed alignment as the SCUT masked artifact.
    atomic_write_text(
        staging_dir / "faces.json",
        json.dumps(as_builtin(records), indent=2, sort_keys=True) + "\n",
    )
    if original_pads:
        padded = sum(1 for value in original_pads if value > 1e-9)
        ctx.log(
            f"original variant: pad fraction mean "
            f"{float(np.mean(original_pads)):.4f}, median "
            f"{float(np.median(original_pads)):.4f}, max "
            f"{float(np.max(original_pads)):.4f}; {padded}/"
            f"{len(original_pads)} faces carry ANY pad -- SCUT's "
            "originals are 350x350 square and carry none, so this is a "
            "difference between the two original arms, measured "
            "(phase15.ORIGINAL_ARM_PADS_TOO)"
        )
    atomic_write_text(
        staging_dir / "metadata.json",
        json.dumps({
            "artifact": version,
            "n_faces": len(survivors),
            "variants": list(variants),
            "size": size,
            "original_pad_fraction": (
                {
                    "mean": float(np.mean(original_pads)),
                    "median": float(np.median(original_pads)),
                    "max": float(np.max(original_pads)),
                    "n_with_any_pad": sum(
                        1 for v in original_pads if v > 1e-9
                    ),
                }
                if original_pads else None
            ),
            "split_counts": counts,
            "levelling": masked_module.placement.SCUT_HEAD_TILT_LEVELLING,
            "ar_sampling": task.get(
                "ar_sampling", masked_module.DEFAULT_AR_SAMPLING
            ),
            "landmark_convention": "dlib-68 (mebeauty, verified 2a)",
            "note": (
                "MEBeauty staged through the SCUT composition "
                "(masked.build_one); the content box uses MEBeauty's "
                "anchors and the shared placement.box_from_anchors "
                "formula. Levelling OFF and CLEFT_AR parity inherited"
            ),
        }, indent=2, sort_keys=True) + "\n",
    )
    payload = hash_dir(staging_dir)
    atomic_write_text(
        staging_dir / "MANIFEST.json",
        json.dumps({
            "artifact": version,
            "payload_rollup": payload["rollup"],
            "payload_files": payload["files"],
            "generating_run": ctx.run_dir.name,
        }, indent=2, sort_keys=True) + "\n",
    )
    staging_dir.rename(final_dir)
    rollup = hash_dir(final_dir)["rollup"]
    ctx.log(f"{version}: {len(survivors)} faces, rollup {rollup[:8]}")

    # ---- the staged sheets: the eye gate before any embedding ---------
    rng = np.random.default_rng(int(ctx.config["seed"]))
    chosen = sorted(
        int(i) for i in rng.choice(
            len(survivors), size=min(sheet_faces, len(survivors)),
            replace=False,
        )
    )
    n_sheets = 0
    for geometry in variants:
        panels = [
            render.Panel(
                label=f"{survivors[i]} [{rows[i]['split']}]",
                image=np.asarray(arrays[geometry][i]),
            )
            for i in chosen
        ]
        for start in range(0, len(panels), 16):
            chunk = panels[start:start + 16]
            n_sheets += 1
            target = ctx.path(
                f"mebeauty_staged_{geometry}_sheet_{n_sheets:02d}.png",
                tier="SHAREABLE",
            )
            render.save_sheet(
                render.tile(chunk, columns=4), target,
                labels=[p.label for p in chunk], columns=4,
                cell=(size, size),
            )
            _write_label_sidecar(
                ctx,
                f"mebeauty_staged_{geometry}_sheet_{n_sheets:02d}_labels.csv",
                [p.label for p in chunk],
            )
    ctx.log(
        f"{n_sheets} staged sheets across {len(variants)} variants -- "
        "SHAREABLE, for the visual check BEFORE any embedding "
        "(phase15.ANATOMICALLY_BLIND_LIMITATION's catch-net)"
    )

    summary = {
        "registration": phase15.STOP_2B_BUILT,
        "artifact": version,
        "rollup": rollup,
        "n_faces": len(survivors),
        "split_counts": counts,
        "variants": list(variants),
        "parity": parity,
        "original_pad_fraction": (
            {
                "mean": float(np.mean(original_pads)),
                "median": float(np.median(original_pads)),
                "max": float(np.max(original_pads)),
                "n_with_any_pad": sum(1 for v in original_pads if v > 1e-9),
            }
            if original_pads else None
        ),
        "box_inside_frame": {"inside": inside, "of": len(rows)},
        "levelling": masked_module.placement.SCUT_HEAD_TILT_LEVELLING,
        "review": (
            "PENDING -- the staged-sheet eye gate before any embedding; "
            "if the anatomically-blind class appears at any noticeable "
            "rate the limitation is REQUANTIFIED before pretraining "
            "(phase15.ANATOMICALLY_BLIND_LIMITATION)"
        ),
    }
    with ctx.atomic("metrics.json", tier="SHAREABLE") as tmp:
        tmp.write_text(
            json.dumps(as_builtin(summary), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def task_pretrain_mebeauty(ctx: RunContext) -> None:
    """Phase 15, stop 3: pretrain on MEBeauty through the PHASE 6 RECIPE
    (phase15.STOP_3_AMENDED).

    **Only the data differs.** The loop, the fixed 30-epoch budget, the
    best-checkpoint selection, the optimiser and every knob come from
    ``pretrain.run_pretraining`` and a ``PretrainConfig`` built from the
    same fields the shipped Phase 6 configs carry. MEBeauty reaches it
    through two default-preserving seams: the artifact is in the SCUT
    masked-artifact FORMAT (so ``load_masked_features`` reads it
    unchanged, row order guaranteed by the same stem-keyed index), and
    ``labels_by_stem`` supplies MEBeauty's SHIPPED universal split.

    One checkpoint per source, and the source is the variant: a masked
    init is GEOMETRY-BOUND (VARIANT_FOR_INIT), so g1 and g2 are separate
    runs and separate checkpoints, never one crossed between them.

    PUBLIC DATA, SHAREABLE, no gate.
    """
    import csv
    import json

    from . import phase15
    from .train import pretrain

    task = ctx.config["task"]
    declared = {entry["name"]: Path(entry["path"]) for entry in ctx.inputs}
    artifact = declared[task["mebeauty_staged"]]
    source = task["source"]

    # ---- the split: MEBeauty's shipped universal halves, as staged ----
    faces = json.loads(
        (artifact / "faces.json").read_text(encoding="utf-8")
    )
    split_of = {entry["stem"]: entry.get("split") for entry in faces}
    scores = {}
    root = declared[task["mebeauty_root"]]
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    for half in ("train", "test"):
        path = root / "scores" / f"{half}_universal_scores.csv"
        with open(path, encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.reader(handle))
        header = rows[0]
        name_index = next(
            i for i in range(len(header))
            if any(
                Path(row[i]).suffix.lower() in image_extensions
                for row in rows[1:20] if i < len(row)
            )
        )
        score_index = next(
            i for i in range(len(header)) if i != name_index and all(
                _is_float(row[i]) for row in rows[1:20] if i < len(row)
            )
        )
        for row in rows[1:]:
            if len(row) > max(name_index, score_index):
                stem = Path(row[name_index].strip()).name
                if split_of.get(stem) == half:
                    scores[stem] = (half, float(row[score_index]))
    train_labels = {
        stem: value for stem, (half, value) in scores.items()
        if half == "train"
    }
    test_labels = {
        stem: value for stem, (half, value) in scores.items()
        if half == "test"
    }
    ctx.log(
        f"MEBeauty split as staged: {len(train_labels)} train / "
        f"{len(test_labels)} test -- the SHIPPED universal split, "
        "restricted to the staged survivors, never a home-made cut"
    )

    config = pretrain.PretrainConfig(
        epochs=int(task["epochs"]),
        inner_val_frac=float(task["inner_val_frac"]),
        seed=int(ctx.config["seed"]),
        deterministic=bool(task["deterministic"]),
        monitor=task["monitor"],
        learning_rate=float(task["learning_rate"]),
        weight_decay=float(task["weight_decay"]),
        batch_size=int(task["batch_size"]),
        checkpoint_every=int(task["checkpoint_every"]),
    )
    ctx.log(
        f"recipe: {config.epochs} fixed epochs, monitor {config.monitor}, "
        f"lr {config.learning_rate}, wd {config.weight_decay}, batch "
        f"{config.batch_size} -- the Phase 6 values, byte-identical"
    )

    result = pretrain.run_pretraining(
        scut_root=root,
        source=source,
        backbone=task["backbone"],
        region_scheme=task["region_scheme"],
        masked_dir=artifact,
        expect_train=int(task["expect_train"]),
        expect_test=int(task["expect_test"]),
        run_dir=ctx.run_dir,
        curves_path=ctx.path("curves.csv", tier="SHAREABLE"),
        pretrained_path=ctx.path("pretrained.pt", tier="SHAREABLE"),
        config=config,
        log=ctx.log,
        labels_by_stem=(train_labels, test_labels),
    )
    selected = result.summary.get("selected_epoch")
    test_pcc = (result.summary.get("test") or {}).get("pcc")
    ctx.log(
        f"pretraining complete: selected epoch {selected}, source-side "
        + (f"test PCC {test_pcc:.4f}" if test_pcc is not None else "test PCC unavailable")
        + " -- and SCUT test PCC turned out NOT to predict cleft "
        "transfer at all (ladder), so this number is PROVENANCE, not "
        "the verdict"
    )

    summary = {
        "registration": phase15.STOP_3_AMENDED,
        "source": source,
        "variant": source.removeprefix("masked_"),
        "n_train": len(train_labels),
        "n_test": len(test_labels),
        "selected_epoch": selected,
        "source_side_test_pcc": test_pcc,
        "pretrain_summary": result.summary,
        "recipe": "Phase 6, byte-identical -- only the data differs",
        "the_verdict_is_not_here": (
            "the cleft-side probe decides; source-side PCC does not "
            "predict cleft transfer (ladder's own finding)"
        ),
        "caveats_bound": phase15.caveats_for(source),
    }
    with ctx.atomic("metrics.json", tier="SHAREABLE") as tmp:
        tmp.write_text(
            json.dumps(as_builtin(summary), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def _is_float(text) -> bool:
    try:
        float(text)
    except (TypeError, ValueError):
        return False
    return True


def task_extract_anchor_embeddings(ctx: RunContext) -> None:
    """Phase 16: persist the 25 Deall anchor embeddings ONCE
    (phase16.ANCHOR_ARTIFACT_PERSISTED).

    The staging-and-extraction path is pass zero's, line for line where
    it matters: deall_partition_stems, deall_read_labels, stage ->
    FrozenExtractor, and the live-path parity check against a cached
    cohort row -- banked reference 1.53e-05, refusal threshold 1e-2,
    both pass zero's. What is NEW is only the persistence.

    CLUSTER-ONLY -- the Deall images are clinical photographs and the
    features follow the images' tier.
    """
    import json

    from . import phase9, phase16
    from .data.manifest import load_manifest
    from .provenance.hashing import hash_dir
    from .train import phase3
    from .train.torch_backbone import FrozenExtractor
    from .geometry import render, staging

    task = ctx.config["task"]
    declared = {entry["name"]: entry for entry in ctx.inputs}
    deall_dir = Path(declared[task["deall_artifact"]]["path"])
    manifest_dir = Path(declared[task["manifest_artifact"]]["path"])
    staged_dir = Path(declared[task["staged_artifact"]]["path"])
    embeddings_dir = Path(declared[task["embeddings_artifact"]]["path"])

    # ---- the anchors: pass zero's partition and label path ------------
    views = phase9.deall_partition_stems(
        [child.name for child in deall_dir.iterdir()]
    )
    if len(views["composite"]) != int(task["expect_anchors"]):
        raise ValueError(
            f"{len(views['composite'])} composite images where the "
            f"registration fixes {task['expect_anchors']}"
        )
    labels_by_id = phase9.deall_read_labels(
        deall_dir / phase9.DEALL_LABELS_FILENAME
    )
    stems = [Path(name).stem for name in views["composite"]]
    mismatch = sorted(set(stems) ^ set(labels_by_id))
    if mismatch:
        raise ValueError(f"composites and labels disagree on {mismatch}")
    grades = [int(labels_by_id[stem]) for stem in stems]

    staged_faces = np.stack([
        staging.stage(render.load_image(deall_dir / name)).image
        for name in views["composite"]
    ])
    extractor = FrozenExtractor(
        task["backbone"], batch_size=int(task["batch_size"])
    )
    values = np.asarray(extractor(staged_faces), dtype=np.float32)

    # ---- the live-path parity check, pass zero's ----------------------
    images, _, patient_ids, _ = phase3.load_inputs(
        manifest_dir, staged_dir, task["geometry"], "mean"
    )
    # [2026-08-29, run e2b1eaf2] embeddings.load takes the FULL manifest
    # id list -- a verification contract, not a row selector. The first
    # version passed the single parity id and crashed at load time on
    # all four attempts (phase16.parity_reference_row has the story);
    # this is pass zero's working shape: full-list load, then select.
    reference = phase16.parity_reference_row(
        embeddings_dir, [int(p) for p in patient_ids]
    )
    parity = float(np.max(np.abs(
        np.asarray(extractor(np.asarray(images[:1]))[0], dtype=np.float64)
        - np.asarray(reference, dtype=np.float64)
    )))
    ctx.log(
        f"live-path parity (patient {int(patient_ids[0])}): max|delta| "
        f"{parity:.2e} (banked reference 1.53e-05, refusal at 1e-2)"
    )
    if parity > 1e-2:
        raise ValueError(
            f"live-path parity {parity:.2e} exceeds 1e-2: the anchors "
            "would be extracted through a different path than the "
            "cached cohort features"
        )

    # ---- persist once, in the anchor namespace ------------------------
    out_dir = ctx.repo_root / "data" / "embeddings" / task["out_version"]
    payload = phase16.save_anchor_set(
        out_dir, values, stems=stems, grades=grades, parity=parity,
        run_name=ctx.run_dir.name,
    )
    declare = hash_dir(out_dir)
    ctx.log(
        f"{task['out_version']}: {values.shape}; "
        f"rollup_sha256_for_configs {declare['rollup'][:8]} (DECLARE THIS)"
        " -- CLUSTER-ONLY, anchor namespace"
    )

    with ctx.atomic("metrics.json", tier="SHAREABLE") as tmp:
        tmp.write_text(
            json.dumps(as_builtin({
                "registration": phase16.ANCHOR_ARTIFACT_PERSISTED,
                "artifact": task["out_version"],
                "rollup_sha256_for_configs": declare["rollup"],
                "shape": list(values.shape),
                "grade_spread": list(phase16.ANCHOR_GRADE_SPREAD),
                "live_path_parity_max_abs": parity,
                "banked_parity_reference": 1.53e-05,
            }), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def task_anchor_loop(ctx: RunContext) -> None:
    """Phase 16: the anchor loop (phase16.EXIT_CRITERIA, locked
    2026-08-29; registration phase15.ANCHOR_LOOP_REGISTERED).

    Full linear W, identity init, identity decay; softmax-expectation
    readout shared byte-for-byte with the untrained identity baseline;
    continuous target; folds from the manifest ONLY -- the training
    loop receives training-fold indices from phase16.training_indices,
    through which the no-folds full-cohort path cannot be expressed.
    """
    import json

    from . import phase7b, phase15, phase16
    from . import embeddings as embeddings_module
    from .cluster_csv import (
        PREDICTIONS_COLUMNS, read_cluster_csv, write_predictions,
    )
    from .data.manifest import load_manifest
    from .eval import metrics as frozen_metrics

    task = ctx.config["task"]
    declared = {entry["name"]: entry for entry in ctx.inputs}
    probe_run = Path(declared[task["probe_run"]]["path"])
    anchor_dir = Path(declared[task["anchor_embeddings"]]["path"])
    embeddings_dir = Path(declared[task["embeddings_artifact"]]["path"])
    manifest_dir = Path(declared[task["manifest_artifact"]]["path"])
    seeds = [int(seed) for seed in task["seeds"]]

    rows = load_manifest(manifest_dir / "manifest.csv")
    patient_ids = [int(r["patient_id"]) for r in rows]
    folds = np.array([int(r["fold"]) for r in rows])
    truth_mean = np.array([float(r["mean"]) for r in rows])
    truth_class3 = np.array([int(r["class3"]) for r in rows])
    features, _ = embeddings_module.load(embeddings_dir, patient_ids)
    features = np.asarray(features, dtype=np.float64)

    anchors, anchor_meta = phase16.load_anchor_set(anchor_dir)
    anchors = np.asarray(anchors, dtype=np.float64)
    anchor_grades = np.array(anchor_meta["grades"], dtype=float)
    anchor_class3 = frozen_metrics.to_3class(anchor_grades)

    # tau: frozen in the UNCORRECTED space (phase16.TAU_DECLARED).
    tau = phase16.anchor_tau(anchors, float(task["tau_scale"]))
    ctx.log(f"tau = {tau:.4f} (tau_scale {task['tau_scale']}, frozen)")

    values, counts = np.unique(truth_class3, return_counts=True)
    majority = float(counts.max()) / float(counts.sum())
    chance = 1.0 / len(values)
    identity = np.eye(features.shape[1])

    loop_by_seed, baseline_by_seed = {}, {}
    per_seed, consistency_rows = {}, []
    for seed in seeds:
        oof_loop = np.zeros(len(rows))
        oof_identity = np.zeros(len(rows))
        selected_epochs, mismatch_lines = [], []
        for held_out in sorted(set(folds.tolist())):
            train = phase16.training_indices(folds, held_out=held_out)
            test = np.flatnonzero(folds == held_out)
            transform, best_epoch, _curve = phase16.train_w(
                features[train], truth_mean[train], anchors, anchor_grades,
                tau=tau,
                lambda_identity=float(task["lambda_identity"]),
                learning_rate=float(task["learning_rate"]),
                batch_size=int(task["batch_size"]),
                max_epochs=int(task["max_epochs"]),
                inner_val_frac=float(task["inner_val_frac"]),
                seed=seed,
            )
            selected_epochs.append(best_epoch)
            oof_loop[test] = phase16.expected_grade(
                features[test], anchors, anchor_grades, transform, tau=tau
            )
            oof_identity[test] = phase16.expected_grade(
                features[test], anchors, anchor_grades, identity, tau=tau
            )
            consistency_rows.append({
                "seed": seed, "fold": int(held_out),
                "corrected_self_consistency": phase16.anchor_self_consistency(
                    anchors, anchor_grades, transform
                ),
                "of": 25,
            })
            # ---- mismatch records (phase16.MISMATCH_RECORD_SPEC) ------
            distances = phase16.corrected_distances(
                features[test], anchors, transform
            )
            for row_index, patient_index in enumerate(test):
                d_row = distances[row_index]
                nearest = int(d_row.argmin())
                nearest_class = int(anchor_class3[nearest])
                patient_class = int(truth_class3[patient_index])
                mismatch_lines.append(",".join([
                    str(patient_ids[patient_index]), str(int(held_out)),
                    str(seed), anchor_meta["stems"][nearest],
                    str(int(anchor_grades[nearest])),
                    f"{d_row[nearest]:.6f}",
                    str(patient_class),
                    str(int(nearest_class == patient_class)),
                    f"{phase16.mismatch_margin(d_row, anchor_class3, patient_class3=patient_class):.6f}",
                    ";".join(f"{value:.6f}" for value in d_row),
                ]))

        loop_by_seed[seed] = oof_loop
        baseline_by_seed[seed] = oof_identity
        loop_pcc = frozen_metrics.pcc(truth_mean, oof_loop)
        per_seed[seed] = {
            "loop_pcc": loop_pcc,
            "loop_spearman": frozen_metrics.spearman(truth_mean, oof_loop),
            "loop_acc3": float(np.mean(
                frozen_metrics.to_3class(oof_loop, clip=True) == truth_class3
            )),
            "identity_pcc": frozen_metrics.pcc(truth_mean, oof_identity),
            "identity_spearman": frozen_metrics.spearman(
                truth_mean, oof_identity
            ),
            "identity_acc3": float(np.mean(
                frozen_metrics.to_3class(oof_identity, clip=True)
                == truth_class3
            )),
            "selected_epochs": selected_epochs,
        }
        fold_of = {patient_ids[i]: int(folds[i]) for i in range(len(rows))}
        write_predictions(
            ctx.path(f"seed_{seed}__predictions.csv", tier="CLUSTER-ONLY"),
            zip(patient_ids, truth_mean, oof_loop,
                (fold_of[p] for p in patient_ids)),
        )
        write_predictions(
            ctx.path(
                f"seed_{seed}__identity_predictions.csv", tier="CLUSTER-ONLY"
            ),
            zip(patient_ids, truth_mean, oof_identity,
                (fold_of[p] for p in patient_ids)),
        )
        header = (
            "patient_id,fold,seed,nearest_anchor,nearest_grade,"
            "nearest_distance,class3,match,margin,all_25_distances"
        )
        with ctx.atomic(
            f"seed_{seed}__mismatch_records.csv", tier="CLUSTER-ONLY"
        ) as tmp:
            tmp.write_text(
                "# CLUSTER-ONLY: patient-keyed\n" + header + "\n"
                + "\n".join(mismatch_lines) + "\n",
                encoding="utf-8",
            )
        ctx.log(
            f"  seed {seed}: loop PCC {loop_pcc:.4f} / identity "
            f"{per_seed[seed]['identity_pcc']:.4f}; acc3 "
            f"{per_seed[seed]['loop_acc3']:.4f} (chance {chance:.3f}, "
            f"majority {majority:.3f})"
        )

    # ---- the primary contrast: standing machinery only ----------------
    probe_by_seed = {}
    for seed in seeds:
        records = read_cluster_csv(
            probe_run / f"seed_{seed}__predictions.csv",
            expect=PREDICTIONS_COLUMNS,
        )
        by_id = {
            int(r["patient_id"]): float(r["prediction"]) for r in records
        }
        probe_by_seed[seed] = np.array([by_id[p] for p in patient_ids])
    loop_pccs = [per_seed[seed]["loop_pcc"] for seed in seeds]
    contrast = phase7b.paired_comparison(
        truth=truth_mean,
        winner_by_seed=loop_by_seed,
        baseline_by_seed=probe_by_seed,
        winner_sd=float(np.std(loop_pccs, ddof=1)),
        n_boot=int(task["n_boot"]),
    )

    summary = {
        "arm": task["arm"],
        "registration": phase15.ANCHOR_LOOP_REGISTERED,
        "rulings": phase16.ANCHOR_LOOP_RULINGS,
        "exit_criteria": phase16.EXIT_CRITERIA,
        "tau": {"value": tau, "rule": phase16.TAU_DECLARED["the_rule"],
                "degenerate_mode": phase16.TAU_DECLARED[
                    "the_degenerate_mode_registered"]},
        "per_seed": {str(seed): per_seed[seed] for seed in seeds},
        "pooled": {
            "loop_pcc": {
                "mean": float(np.mean(loop_pccs)),
                "sd": float(np.std(loop_pccs, ddof=1)),
            },
            "identity_pcc": {
                "mean": float(np.mean(
                    [per_seed[seed]["identity_pcc"] for seed in seeds]
                )),
                "sd": float(np.std(
                    [per_seed[seed]["identity_pcc"] for seed in seeds],
                    ddof=1,
                )),
            },
        },
        "acc3_baselines": {"chance": chance, "majority": majority},
        "identity_baseline_is": phase16.IDENTITY_BASELINE,
        "self_consistency": {
            "per_fold_seed": consistency_rows,
            "banked_uncorrected": {"euclidean": "4/25", "cosine": "3/25",
                                   "chance_expectation": "~4.75/25"},
            "readings": phase16.SELF_CONSISTENCY_READINGS,
        },
        "primary_contrast": contrast,
        "contrast_registration": phase16.PRIMARY_CONTRAST_REGISTERED,
        "pass_zero_descriptive": {
            "best_cell": "euclidean k=1, PCC 0.1823",
            "not_the_identity_baseline": phase16.IDENTITY_BASELINE[
                "distinct_from_pass_zero"],
        },
        "mismatch_spec": phase16.MISMATCH_RECORD_SPEC,
        "no_ledger_row_without_the_paired_bar": True,
    }
    with ctx.atomic("metrics.json", tier="SHAREABLE") as tmp:
        tmp.write_text(
            json.dumps(as_builtin(summary), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def _synth_index(synth_dir):
    """The synthesis artifact's index, parsed as the writer writes it.

    **[2026-08-30, arm-A crash]** The published artifact carries
    ``faces.json`` -- ONE JSON document, a list of per-face record
    dicts -- written at finalization; ``faces.jsonl`` is the RESUME
    JOURNAL, deleted before the rename (the writer's own comment says
    so, and test_scut_masked asserts the artifact has no .jsonl). The
    first readers looked for the journal name with a line-by-line
    parse: FileNotFoundError on a correct artifact, all attempts
    identical. One reader, shared, so the name and the format cannot
    drift apart across arms again.

    Returns (per_face, magnitudes): the record list, and the sorted
    magnitude values from the first face's per-magnitude dict keys.
    """
    import json

    per_face = json.loads(
        (Path(synth_dir) / "faces.json").read_text(encoding="utf-8")
    )
    if not isinstance(per_face, list) or not per_face:
        raise ValueError(
            f"{synth_dir}/faces.json: expected a non-empty list of "
            "per-face records"
        )
    magnitudes = sorted(float(m) for m in per_face[0]["magnitudes"])
    return per_face, magnitudes


def _tstr_extract(extractor, images, batch_size: int):
    """Frozen features of a uint8 image stack, batched."""
    chunks = []
    for start in range(0, len(images), batch_size):
        chunks.append(np.asarray(
            extractor(np.asarray(images[start:start + batch_size]))
        ))
    return np.concatenate(chunks, axis=0).astype(np.float64)


def task_tstr_regression(ctx: RunContext) -> None:
    """Phase 17 arm A: frozen ViT-B/16 embeddings of the TPS synthesis
    set, a linear head on MAGNITUDE-MAPPED labels
    (phase17.DECLARED_SETTINGS_17['magnitude_to_grade_map']), evaluated
    on all 237 real patients as pure test.

    Zero real patient images or labels in training -- the TSTR premise,
    and the ordering below is structural: everything before the
    evaluation marker touches only the synthetic artifact.
    """
    import json

    from . import phase17
    from . import embeddings as embeddings_module
    from .cluster_csv import write_predictions
    from .data.manifest import load_manifest
    from .eval import metrics as frozen_metrics
    from .train.torch_backbone import FrozenExtractor

    import torch

    task = ctx.config["task"]
    declared = {entry["name"]: entry for entry in ctx.inputs}
    synth_dir = Path(declared[task["synth_set"]]["path"])
    seeds = [int(seed) for seed in task["seeds"]]

    # ---- TRAINING ON SYNTHETIC ONLY -----------------------------------
    _, magnitudes = _synth_index(synth_dir)
    extractor = FrozenExtractor(
        task["backbone"], batch_size=int(task.get("extract_batch_size", 32))
    )
    features_blocks, label_blocks = [], []
    for magnitude in magnitudes:
        stack = np.load(
            synth_dir / f"synth_g1_m{magnitude:.3f}.npy", mmap_mode="r"
        )
        features_blocks.append(
            _tstr_extract(extractor, stack, int(task.get(
                "extract_batch_size", 32
            )))
        )
        label_blocks.append(np.full(
            len(stack), phase17.magnitude_to_grade([magnitude])[0]
        ))
        ctx.log(
            f"extracted m={magnitude:.3f}: {len(stack)} faces "
            f"-> grade {label_blocks[-1][0]:.3f}"
        )
    train_x = np.concatenate(features_blocks).astype(np.float32)
    train_y = np.concatenate(label_blocks).astype(np.float32)

    per_seed, predictions_by_seed = {}, {}
    weights_by_seed = {}
    for seed in seeds:
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(train_x))
        n_val = max(1, int(round(len(order) * float(task["inner_val_frac"]))))
        val_idx, fit_idx = order[:n_val], order[n_val:]
        torch.manual_seed(seed)
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        x_fit = torch.tensor(train_x[fit_idx], device=device)
        y_fit = torch.tensor(train_y[fit_idx], device=device)
        x_val = torch.tensor(train_x[val_idx], device=device)
        y_val = torch.tensor(train_y[val_idx], device=device)
        head = torch.nn.Linear(train_x.shape[1], 1).to(device)
        optimizer = torch.optim.AdamW(
            head.parameters(), lr=float(task["learning_rate"]),
            weight_decay=0.0,
        )
        best = {"mse": float("inf"), "state": None, "epoch": -1}
        batch = int(task["batch_size"])
        epoch_rng = np.random.default_rng(seed + 1)
        for epoch in range(int(task["max_epochs"])):
            perm = epoch_rng.permutation(len(fit_idx))
            for start in range(0, len(perm), batch):
                rows = torch.tensor(perm[start:start + batch], device=device)
                optimizer.zero_grad()
                loss = torch.mean(
                    (head(x_fit[rows]).squeeze(-1) - y_fit[rows]) ** 2
                )
                loss.backward()
                optimizer.step()
            with torch.no_grad():
                val_mse = float(torch.mean(
                    (head(x_val).squeeze(-1) - y_val) ** 2
                ))
            if val_mse < best["mse"]:
                best = {
                    "mse": val_mse, "epoch": epoch,
                    "state": {
                        k: v.detach().cpu().clone()
                        for k, v in head.state_dict().items()
                    },
                }
        head.load_state_dict(best["state"])
        weights_by_seed[seed] = (head, best["epoch"])
        ctx.log(f"  seed {seed}: best inner-val MSE {best['mse']:.4f} "
                f"at epoch {best['epoch']}")

    # ---- EVALUATION ON REAL, AFTER TRAINING ---------------------------
    manifest_dir = Path(declared[task["manifest_artifact"]]["path"])
    embeddings_dir = Path(declared[task["embeddings_artifact"]]["path"])
    rows = load_manifest(manifest_dir / "manifest.csv")
    patient_ids = [int(r["patient_id"]) for r in rows]
    folds = {int(r["patient_id"]): int(r["fold"]) for r in rows}
    truth_mean = np.array([float(r["mean"]) for r in rows])
    truth_class3 = np.array([int(r["class3"]) for r in rows])
    cohort, _ = embeddings_module.load(embeddings_dir, patient_ids)
    cohort_t = torch.tensor(np.asarray(cohort, dtype=np.float32))

    values, counts = np.unique(truth_class3, return_counts=True)
    chance, majority = 1.0 / len(values), float(counts.max() / counts.sum())
    for seed in seeds:
        head, _epoch = weights_by_seed[seed]
        with torch.no_grad():
            predicted = head.cpu()(cohort_t).squeeze(-1).numpy().astype(float)
        predictions_by_seed[seed] = predicted
        per_seed[seed] = {
            "pcc": frozen_metrics.pcc(truth_mean, predicted),
            "spearman": frozen_metrics.spearman(truth_mean, predicted),
            "acc3": float(np.mean(
                frozen_metrics.to_3class(predicted, clip=True) == truth_class3
            )),
            "selected_epoch": weights_by_seed[seed][1],
        }
        write_predictions(
            ctx.path(f"seed_{seed}__predictions.csv", tier="CLUSTER-ONLY"),
            zip(patient_ids, truth_mean, predicted,
                (folds[p] for p in patient_ids)),
        )
        ctx.log(f"  seed {seed}: PCC {per_seed[seed]['pcc']:.4f} "
                f"(chance {chance:.3f} / majority {majority:.3f} beside acc3 "
                f"{per_seed[seed]['acc3']:.4f})")

    pccs = [per_seed[s]["pcc"] for s in seeds]
    with ctx.atomic("metrics.json", tier="SHAREABLE") as tmp:
        tmp.write_text(json.dumps(as_builtin({
            "arm": task["arm"],
            "readings": phase17.READINGS_COMMITTED["arm_a_carried"],
            "magnitude_to_grade_map": phase17.DECLARED_SETTINGS_17[
                "magnitude_to_grade_map"],
            "per_seed": {str(s): per_seed[s] for s in seeds},
            "pooled": {"pcc_mean": float(np.mean(pccs)),
                       "pcc_sd": float(np.std(pccs, ddof=1))},
            "acc3_baselines": {"chance": chance, "majority": majority},
            "zero_real_images_in_training": True,
            "exit_criteria": phase17.EXIT_CRITERIA,
        }), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def task_siamese_contrastive(ctx: RunContext) -> None:
    """Phase 17 arms B/C: frozen branches over fixed-midline left/right
    views, a trained 768x768 linear projection, contrastive loss on
    binary symmetric/asymmetric pairs, distance readout
    normalized-and-rounded (phase17.DECLARED_SETTINGS_17).

    Which arm this is comes from the DECLARED synthesis artifact: B's
    config declares the piecewise-affine set, C's the TPS set -- the
    training and readout are byte-identical, which is what makes B-C
    isolate the transformation family.
    """
    import json

    from . import phase17
    from .cluster_csv import write_predictions
    from .data.manifest import load_manifest
    from .eval import metrics as frozen_metrics
    from .train import phase3
    from .train.torch_backbone import FrozenExtractor

    import torch

    task = ctx.config["task"]
    declared = {entry["name"]: entry for entry in ctx.inputs}
    synth_dir = Path(declared[task["synth_set"]]["path"])
    seeds = [int(seed) for seed in task["seeds"]]
    margin = float(task["margin"])

    # ---- TRAINING ON SYNTHETIC ONLY -----------------------------------
    _, magnitudes = _synth_index(synth_dir)
    extract_batch = int(task.get("extract_batch_size", 32))
    extractor = FrozenExtractor(task["backbone"], batch_size=extract_batch)

    left_blocks, right_blocks, symmetric_blocks = [], [], []
    for magnitude in magnitudes:
        stack = np.load(
            synth_dir / f"synth_g1_m{magnitude:.3f}.npy", mmap_mode="r"
        )
        lefts, rights = [], []
        for image in stack:
            left, right = phase17.left_right_views(np.asarray(image))
            lefts.append(left)
            rights.append(right)
        left_blocks.append(_tstr_extract(extractor, lefts, extract_batch))
        right_blocks.append(_tstr_extract(extractor, rights, extract_batch))
        # The pair rule (schema: symmetric_if_zero_magnitude): the
        # binary label is symmetric iff the face is undeformed.
        symmetric_blocks.append(
            np.full(len(stack), 1.0 if magnitude == 0 else 0.0)
        )
        ctx.log(f"views m={magnitude:.3f}: {len(stack)} pairs "
                f"({'symmetric' if magnitude == 0 else 'asymmetric'})")
    left_x = np.concatenate(left_blocks).astype(np.float32)
    right_x = np.concatenate(right_blocks).astype(np.float32)
    labels = np.concatenate(symmetric_blocks).astype(np.float32)

    projections_by_seed = {}
    for seed in seeds:
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(labels))
        n_val = max(1, int(round(len(order) * float(task["inner_val_frac"]))))
        val_idx, fit_idx = order[:n_val], order[n_val:]
        torch.manual_seed(seed)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dim = left_x.shape[1]
        assert int(task["projection_dim"]) == dim
        projection = torch.eye(dim, device=device, requires_grad=True)
        optimizer = torch.optim.AdamW(
            [projection], lr=float(task["learning_rate"]), weight_decay=0.0
        )
        l_fit = torch.tensor(left_x[fit_idx], device=device)
        r_fit = torch.tensor(right_x[fit_idx], device=device)
        y_fit = torch.tensor(labels[fit_idx], device=device)
        l_val = torch.tensor(left_x[val_idx], device=device)
        r_val = torch.tensor(right_x[val_idx], device=device)
        y_val = torch.tensor(labels[val_idx], device=device)

        def contrastive(distance, symmetric):
            pull = symmetric * distance ** 2
            push = (1 - symmetric) * torch.clamp(
                margin - distance, min=0.0
            ) ** 2
            return torch.mean(pull + push)

        best = {"loss": float("inf"), "w": None, "epoch": -1}
        batch = int(task["batch_size"])
        epoch_rng = np.random.default_rng(seed + 1)
        for epoch in range(int(task["max_epochs"])):
            perm = epoch_rng.permutation(len(fit_idx))
            for start in range(0, len(perm), batch):
                rows = torch.tensor(perm[start:start + batch], device=device)
                optimizer.zero_grad()
                distance = torch.norm(
                    l_fit[rows] @ projection.T - r_fit[rows] @ projection.T,
                    dim=1,
                )
                loss = contrastive(distance, y_fit[rows])
                loss.backward()
                optimizer.step()
            with torch.no_grad():
                val_distance = torch.norm(
                    l_val @ projection.T - r_val @ projection.T, dim=1
                )
                val_loss = float(contrastive(val_distance, y_val))
            if val_loss < best["loss"]:
                best = {"loss": val_loss, "epoch": epoch,
                        "w": projection.detach().cpu().numpy().copy()}
        projections_by_seed[seed] = best
        ctx.log(f"  seed {seed}: best inner-val loss {best['loss']:.4f} "
                f"at epoch {best['epoch']}")

    # ---- EVALUATION ON REAL, AFTER TRAINING ---------------------------
    manifest_dir = Path(declared[task["manifest_artifact"]]["path"])
    staged_dir = Path(declared[task["staged_artifact"]]["path"])
    rows = load_manifest(manifest_dir / "manifest.csv")
    patient_ids = [int(r["patient_id"]) for r in rows]
    folds = {int(r["patient_id"]): int(r["fold"]) for r in rows}
    truth_mean = np.array([float(r["mean"]) for r in rows])
    truth_class3 = np.array([int(r["class3"]) for r in rows])
    images, _, loaded_ids, _ = phase3.load_inputs(
        manifest_dir, staged_dir, task["geometry"], "mean"
    )
    assert [int(p) for p in loaded_ids] == patient_ids
    patient_lefts, patient_rights = [], []
    for image in images:
        left, right = phase17.left_right_views(np.asarray(image))
        patient_lefts.append(left)
        patient_rights.append(right)
    left_p = _tstr_extract(extractor, patient_lefts, extract_batch)
    right_p = _tstr_extract(extractor, patient_rights, extract_batch)

    values, counts = np.unique(truth_class3, return_counts=True)
    chance, majority = 1.0 / len(values), float(counts.max() / counts.sum())
    per_seed = {}
    for seed in seeds:
        w = projections_by_seed[seed]["w"]
        distance = np.linalg.norm(
            left_p @ w.T - right_p @ w.T, axis=1
        )
        predicted = np.array([
            phase17.distance_to_grade(d, margin=margin) for d in distance
        ], dtype=float)
        per_seed[seed] = {
            "pcc": frozen_metrics.pcc(truth_mean, predicted),
            "spearman": frozen_metrics.spearman(truth_mean, predicted),
            "acc3": float(np.mean(
                frozen_metrics.to_3class(predicted, clip=True) == truth_class3
            )),
            "selected_epoch": projections_by_seed[seed]["epoch"],
        }
        write_predictions(
            ctx.path(f"seed_{seed}__predictions.csv", tier="CLUSTER-ONLY"),
            zip(patient_ids, truth_mean, predicted,
                (folds[p] for p in patient_ids)),
        )
        ctx.log(f"  seed {seed}: PCC {per_seed[seed]['pcc']:.4f}")

    pccs = [per_seed[s]["pcc"] for s in seeds]
    with ctx.atomic("metrics.json", tier="SHAREABLE") as tmp:
        tmp.write_text(json.dumps(as_builtin({
            "arm": task["arm"],
            "warp_family_of_declared_set": task["arm"],
            "readings": {
                "if_null": phase17.READINGS_COMMITTED[
                    f"arm_{task['arm'][-1].lower()}_if_null"],
                "if_positive": phase17.READINGS_COMMITTED[
                    f"arm_{task['arm'][-1].lower()}_if_positive"],
            },
            "declared_settings": {
                "margin": phase17.DECLARED_SETTINGS_17["margin"],
                "readout": phase17.DECLARED_SETTINGS_17[
                    "readout_normalization"],
            },
            "per_seed": {str(s): per_seed[s] for s in seeds},
            "pooled": {"pcc_mean": float(np.mean(pccs)),
                       "pcc_sd": float(np.std(pccs, ddof=1))},
            "acc3_baselines": {"chance": chance, "majority": majority},
            "zero_real_images_in_training": True,
            "exit_criteria": phase17.EXIT_CRITERIA,
        }), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def task_tstr_family_analysis(ctx: RunContext) -> None:
    """Phase 17: the five-contrast family, and nothing beyond it
    (phase17.EXIT_CRITERIA criterion 3). Pod arithmetic over the three
    arm runs' prediction CSVs and the probe's OOF CSVs."""
    import json

    from . import phase7b, phase17
    from .cluster_csv import PREDICTIONS_COLUMNS, read_cluster_csv
    from .data.manifest import load_manifest

    task = ctx.config["task"]
    declared = {entry["name"]: entry for entry in ctx.inputs}
    manifest_dir = Path(declared[task["manifest_artifact"]]["path"])
    rows = load_manifest(manifest_dir / "manifest.csv")
    patient_ids = [int(r["patient_id"]) for r in rows]
    truth_mean = np.array([float(r["mean"]) for r in rows])
    seeds = [int(seed) for seed in task["seeds"]]

    def read_arm(input_name):
        run_dir = Path(declared[input_name]["path"])
        by_seed = {}
        for seed in seeds:
            records = read_cluster_csv(
                run_dir / f"seed_{seed}__predictions.csv",
                expect=PREDICTIONS_COLUMNS,
            )
            by_id = {
                int(r["patient_id"]): float(r["prediction"]) for r in records
            }
            by_seed[seed] = np.array([by_id[p] for p in patient_ids])
        return by_seed

    arms = {name: read_arm(task[f"{name}_run"])
            for name in ("arm_a", "arm_b", "arm_c")}
    probe = read_arm("probe_run")

    from .eval.metrics import pcc as pcc_metric

    def arm_sd(by_seed):
        scores = [pcc_metric(truth_mean, by_seed[s]) for s in seeds]
        return float(np.std(scores, ddof=1))

    contrasts = {}
    family = (
        ("primary_a_vs_probe", arms["arm_a"], probe),
        ("primary_b_vs_probe", arms["arm_b"], probe),
        ("primary_c_vs_probe", arms["arm_c"], probe),
        ("secondary_a_vs_c", arms["arm_a"], arms["arm_c"]),
        ("secondary_b_vs_c", arms["arm_b"], arms["arm_c"]),
    )
    for name, winner, baseline in family:
        result = phase7b.paired_comparison(
            truth=truth_mean, winner_by_seed=winner,
            baseline_by_seed=baseline, winner_sd=arm_sd(winner),
            n_boot=int(task["n_boot"]),
        )
        # paired_comparison's own vocabulary. [UPDATED 2026-09-02:
        # condition 1 is now RETURNED by paired_comparison rather than
        # rebuilt here from three of its fields. Same value -- a test
        # pins the equivalence -- with one copy of the rule instead of
        # two.]
        contrasts[name] = {
            "result": result,
            "condition_1": result["all_seeds_exclude_zero_one_direction"],
            "condition_2": result["exceeds_threshold"],
            "verdict": (
                "claimable" if result["claimable"] else "unresolved"
            ),
        }
        ctx.log(
            f"{name}: mean delta {result['mean_delta']:+.4f} -> "
            f"{contrasts[name]['verdict']} ({result['reading']})"
        )

    with ctx.atomic("metrics.json", tier="SHAREABLE") as tmp:
        tmp.write_text(json.dumps(as_builtin({
            "family": "the five contrasts and nothing beyond them",
            "contrasts": contrasts,
            "combination_cells": phase17.READINGS_COMMITTED[
                "combination_cells"],
            "exit_criteria": phase17.EXIT_CRITERIA,
        }), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def task_record_artifact_check(ctx: RunContext) -> None:
    """[2026-08-31] The standing record-versus-artifact check
    (``record_audit.STANDING_CHECK_DESIGN``).

    For every locked arm with a machine-readable banked PCC, recompute
    that PCC from the arm's OWN per-seed prediction CSVs and compare.
    **Reads artifacts; asserts nothing about them beyond agreement with
    the records.**

    It is a job rather than a suite test because the CSVs are
    CLUSTER-ONLY and reach no authoring machine. It reuses
    ``cluster_csv.read_cluster_csv`` and the frozen ``eval.metrics.pcc``
    -- the same reader and metric ``task_metric_space_analysis`` uses --
    and does not touch that task, whose run 5 is citable and whose exit
    criteria are locked.

    A mismatch is REPORTED, not raised: the point is a table of deltas
    someone can rule on, and a check that dies on the first disagreement
    hides the other 62.
    """
    import json

    from . import record_audit
    from .cluster_csv import PREDICTIONS_COLUMNS, read_cluster_csv
    from .data.manifest import load_manifest
    from .eval import metrics as frozen_metrics

    task = ctx.config["task"]
    declared = {entry["name"]: entry for entry in ctx.inputs}
    manifest_dir = Path(declared[task["manifest_artifact"]]["path"])

    rows = load_manifest(manifest_dir / "manifest.csv")
    patient_ids = [int(r["patient_id"]) for r in rows]
    truth_mean = np.array([float(r["mean"]) for r in rows])
    index_of = {p: i for i, p in enumerate(patient_ids)}

    banked = record_audit.banked_arm_pcc()
    threshold = float(task["threshold"])
    measured, per_arm = {}, {}
    for arm in task["arms"]:
        name = arm["name"]
        if banked.get(name, {}).get("pcc") is None:
            continue  # a named gap; reported below, never silently skipped
        run_dir = Path(declared[arm["input"]]["path"])
        expected_rows = int(arm["n_patients"])
        arm_ids, per_seed = None, []
        for seed in [int(s) for s in arm["seeds"]]:
            records = read_cluster_csv(
                run_dir / f"seed_{seed}__{arm['csv']}.csv",
                expect=PREDICTIONS_COLUMNS,
            )
            if len(records) != expected_rows:
                raise ValueError(
                    f"{name} seed {seed}: {len(records)} rows where the "
                    f"arm's documented shape is {expected_rows}"
                )
            seed_ids = sorted(int(r["patient_id"]) for r in records)
            if arm_ids is None:
                arm_ids = seed_ids
                unknown = set(arm_ids) - set(patient_ids)
                if unknown:
                    raise ValueError(
                        f"{name}: patient ids {sorted(unknown)[:5]} are "
                        "not in the manifest"
                    )
                arm_truth = truth_mean[[index_of[p] for p in arm_ids]]
            elif seed_ids != arm_ids:
                raise ValueError(
                    f"{name} seed {seed}: patient set differs from this "
                    "arm's other seeds"
                )
            by_id = {
                int(r["patient_id"]): float(r["prediction"]) for r in records
            }
            predicted = np.array([by_id[p] for p in arm_ids])
            per_seed.append(frozen_metrics.pcc(arm_truth, predicted))
        # The banked figure is the MEAN of per-seed pooled-OOF PCCs --
        # the same pooling task_metric_space_analysis reports.
        measured[name] = float(np.mean(per_seed))
        per_arm[name] = {
            "measured": measured[name],
            "per_seed": [float(v) for v in per_seed],
            "n_seeds": len(per_seed),
        }
        ctx.log(
            f"{name}: measured {measured[name]:+.4f} vs banked "
            f"{banked[name]['pcc']:+.4f} "
            f"(delta {measured[name] - banked[name]['pcc']:+.4f})"
        )

    verdict = record_audit.compare_to_artifact(banked, measured, threshold)
    ctx.log(
        f"{verdict['n_compared']} arms compared at threshold {threshold}: "
        + ("AGREE" if verdict["agree"]
           else f"{len(verdict['mismatches'])} MISMATCH(ES)")
    )

    summary = {
        "what": (
            "record-versus-artifact agreement: every locked arm's banked "
            "PCC against the same PCC recomputed from its own prediction "
            "CSVs (record_audit.STANDING_CHECK_DESIGN)"
        ),
        "threshold": threshold,
        "agree": verdict["agree"],
        "n_compared": verdict["n_compared"],
        "mismatches": verdict["mismatches"],
        "largest_delta": verdict["largest_delta"],
        "deltas": verdict["deltas"],
        "per_arm": per_arm,
        "banked_sources": {
            name: cell["source"] for name, cell in sorted(banked.items())
            if cell["pcc"] is not None
        },
        "not_covered": {
            "arms": list(record_audit.BANKED_VALUE_GAPS),
            "why": (
                "no machine-readable banked PCC exists for these -- their "
                "values live only in prose "
                "(record_audit.BANKED_VALUE_GAP_REASONS). Reported, never "
                "silently dropped"
            ),
        },
        "banked_without_artifact": verdict["banked_without_artifact"],
        "artifact_without_banked": verdict["artifact_without_banked"],
        "a_mismatch_is_reported_not_raised": (
            "the deliverable is a table of deltas to rule on; a check "
            "that died on the first disagreement would hide the rest"
        ),
    }
    with ctx.atomic("metrics.json", tier="SHAREABLE") as tmp:
        tmp.write_text(
            json.dumps(as_builtin(summary), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def _p21_load_arm_predictions(
    ctx, task: dict, declared: dict, patient_ids: list, with_truth: bool = False,
):
    """Every declared arm's per-patient OOF predictions, per seed.

    **[2026-09-01] ``with_truth`` returns the TRUTH column too**, added
    for the cross-arm shrinkage measurement, which needs
    ``sd(prediction) / sd(truth)`` per arm. The truth is already in every
    CSV (``PREDICTIONS_COLUMNS``), so this reads a column that was being
    parsed and discarded -- it is not a second reader, and the default
    leaves both existing callers byte-identical.

    **The one reader for both Phase 21 arms.** No glob: every arm reaches
    this through the config's literal list, which the generator derives
    from ``phase18.ARM_LIST_LOCKED`` filtered by
    ``phase21.ARM_SET_RULED``. Cached CSVs only -- no pixels, no
    artifact, no hash.

    Returns ``{seed: {arm_name: array aligned to patient_ids}}``, or
    ``(predictions, truth)`` in the same shape when ``with_truth``.
    """
    from .cluster_csv import PREDICTIONS_COLUMNS, read_cluster_csv

    index_of = {p: i for i, p in enumerate(patient_ids)}
    seeds = [int(s) for s in task["seeds"]]
    by_seed = {seed: {} for seed in seeds}
    truth_by_seed = {seed: {} for seed in seeds}
    for arm in task["arms"]:
        # The ruled set is 237-cohort only; a 236-row arm here would mean
        # the config drifted from ARM_SET_RULED, and averaging it in
        # would silently misalign patients.
        if int(arm["n_patients"]) != len(patient_ids):
            raise ValueError(
                f"{arm['name']}: declared n_patients {arm['n_patients']} "
                f"against the manifest's {len(patient_ids)}. The ruled arm "
                "set is the 237 cohort only (phase21.ARM_SET_RULED)"
            )
        if [int(s) for s in arm["seeds"]] != seeds:
            raise ValueError(
                f"{arm['name']}: seeds {arm['seeds']} against the declared "
                f"basis {seeds}. Every arm contributes the SHARED FIVE "
                "(phase21.SEED_BASIS_RULED)"
            )
        run_dir = Path(declared[arm["input"]]["path"])
        for seed in seeds:
            records = read_cluster_csv(
                run_dir / f"seed_{seed}__{arm['csv']}.csv",
                expect=PREDICTIONS_COLUMNS,
            )
            if len(records) != len(patient_ids):
                raise ValueError(
                    f"{arm['name']} seed {seed}: {len(records)} rows against "
                    f"{len(patient_ids)} manifest patients"
                )
            vector = np.full(len(patient_ids), np.nan)
            truth_vector = np.full(len(patient_ids), np.nan)
            for row in records:
                pid = int(row["patient_id"])
                if pid not in index_of:
                    raise ValueError(
                        f"{arm['name']} seed {seed}: patient {pid} is not in "
                        "the manifest"
                    )
                vector[index_of[pid]] = float(row["prediction"])
                truth_vector[index_of[pid]] = float(row["truth"])
            if np.isnan(vector).any():
                raise ValueError(
                    f"{arm['name']} seed {seed}: {int(np.isnan(vector).sum())} "
                    "manifest patients have no prediction"
                )
            by_seed[seed][arm["name"]] = vector
            truth_by_seed[seed][arm["name"]] = truth_vector
    ctx.log(
        f"loaded {len(task['arms'])} arms x {len(seeds)} seeds of cached OOF "
        f"predictions over {len(patient_ids)} patients -- no pixels read"
    )
    if with_truth:
        return by_seed, truth_by_seed
    return by_seed


def task_cohort_pair_separation(ctx: RunContext) -> None:
    """How often do two patients' panel means differ by less than the
    panel's own noise? (``phase21.COHORT_PAIR_SEPARATION_DESIGNED``.)

    **A reckoning input for Phase 22, not Phase 22 machinery.** Phase
    22's cohort ranking arm would train on patient pairs; a pair
    separated by less than the panel's noise is ordered by chance, and a
    ranking loss cannot tell a coin flip from a signal.

    **The noise scale is DERIVED, never chosen**: classical test theory
    on ``reliability.RELIABILITY_237``. ``sd_obs`` is computed from the
    manifest here -- the docstring figure 0.628 is NOT used, and note
    that ``reliability.item_total``'s docstring carries the same digits
    as an ITEM-TOTAL CORRELATION, a different quantity.

    POD ARITHMETIC: one manifest column and 27,966 differences. No
    backbone, no GPU, no new artifact, and no gate.
    """
    import json
    import math

    from . import phase21
    from .data import reliability
    from .data.manifest import load_manifest

    task = ctx.config["task"]
    declared = {entry["name"]: entry for entry in ctx.inputs}
    rows = load_manifest(
        Path(declared[task["manifest_artifact"]]["path"]) / "manifest.csv"
    )
    if len(rows) != int(task["expect_patients"]):
        raise ValueError(
            f"{len(rows)} manifest rows, expected {task['expect_patients']}"
        )
    means = np.array([float(r[task["label"]]) for r in rows], dtype=float)

    # ---- the derivation, from banked figures only ---------------------
    # sd with ddof=0, matching phase3.sanity_report's np.std(truth) --
    # the project's own convention for this cohort's label spread.
    sd_obs = float(np.std(means))
    reliability_237 = float(reliability.RELIABILITY_237)
    error_fraction = 1.0 - reliability_237
    k_single = math.sqrt(error_fraction)
    k_diff = math.sqrt(2.0) * k_single
    se_single = sd_obs * k_single
    se_diff = sd_obs * k_diff
    derivation = (
        f"MEAN_R_237 {reliability.MEAN_R_237} -> RELIABILITY_237 "
        f"{reliability_237} (Spearman-Brown, k=5); error fraction "
        f"1 - {reliability_237} = {error_fraction:.4f}; "
        f"SE_single = sd_obs * sqrt({error_fraction:.4f}) = sd_obs * "
        f"{k_single:.4f}; SE_diff = sqrt(2) * that = sd_obs * "
        f"{k_diff:.4f}. sd_obs COMPUTED from the manifest here "
        f"(ddof=0) = {sd_obs:.6f}, never quoted"
    )
    ctx.log("derivation: " + derivation)
    ctx.log(
        f"  SE_single = {se_single:.6f}   SE_diff = {se_diff:.6f}   "
        f"(the pair ORDERING turns on the difference, so SE_diff is the "
        f"proper scale; both are reported)"
    )

    # ---- all C(n,2) separations --------------------------------------
    diffs = np.abs(means[:, None] - means[None, :])[
        np.triu_indices(len(means), 1)
    ]
    n_pairs = int(len(diffs))
    ctx.log(f"{n_pairs} pairs from {len(means)} patients")

    scales = {"se_single": se_single, "se_diff": se_diff}
    multiples = [int(m) for m in task["multiples"]]
    fractions = {}
    for name, scale in scales.items():
        for m in multiples:
            key = f"{name}_x{m}"
            fractions[key] = float(np.mean(diffs < m * scale))
            ctx.log(
                f"  below {m}x {name} ({m * scale:.6f}): "
                f"{fractions[key]:.4f}"
            )

    # ---- the registered prior, beside what landed ---------------------
    prior = phase21.COHORT_PAIR_SEPARATION_DESIGNED[
        "the_prior_registered_before_the_number"
    ]
    ctx.log("registered prior (an EXPECTATION, not a result): " + prior)
    distinct = int(len(np.unique(np.round(diffs, 9))))
    ctx.log(
        f"discreteness: the panel mean falls on multiples of 0.2, and the "
        f"{n_pairs} separations take {distinct} distinct values -- if the "
        f"fractions sit far from the prior, this is why"
    )

    # ---- the cell, on the DECLARED scale and multiple ------------------
    primary = f"{task['primary_scale']}_x{int(task['primary_multiple'])}"
    below = fractions[primary]
    cell = ("reading_most_pairs_do_not_clear_it" if below >= 0.5
            else "reading_most_pairs_clear_the_noise")
    ctx.log(
        f"primary scale {primary} (DECLARED in the config): {below:.4f} of "
        f"pairs below it -> {cell}"
    )
    reading = _registered_reading(
        ctx,
        reading=phase21.COHORT_PAIR_SEPARATION_DESIGNED[cell],
        joined=len(means), expected=int(task["expect_patients"]),
        what="the cohort-pair separation measurement",
    )
    ctx.log(
        "bears on the COHORT arm only: "
        + phase21.COHORT_PAIR_SEPARATION_DESIGNED[
            "it_bears_on_the_cohort_arm_only"
        ]
    )

    summary = {
        "design": phase21.COHORT_PAIR_SEPARATION_DESIGNED,
        "status": (
            "DESCRIPTIVE reckoning input for Phase 22; no ledger row, no "
            "lock touched"
        ),
        "n_patients": len(means),
        "n_pairs": n_pairs,
        "sd_obs": sd_obs,
        "sd_ddof": 0,
        "reliability_237": reliability_237,
        "error_fraction": error_fraction,
        "se_single": se_single,
        "se_diff": se_diff,
        "multipliers": {"k_single": k_single, "k_diff": k_diff},
        "derivation": derivation,
        "fraction_below": fractions,
        "separation_quantiles": {
            str(q): float(np.quantile(diffs, q / 100))
            for q in (0, 5, 25, 50, 75, 95, 100)
        },
        "separation_mean": float(diffs.mean()),
        "n_distinct_separations": distinct,
        "registered_prior": prior,
        "primary_scale_declared": primary,
        "cell_fired": cell,
        "reading": reading,
    }
    with ctx.atomic("metrics.json", tier="SHAREABLE") as tmp:
        tmp.write_text(
            json.dumps(as_builtin(summary), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def task_train_rank_cv(ctx: RunContext) -> None:
    """Phase 22's six ranking arms: pairwise logistic over patient pairs.

    **One new task kind** (``phase22.AXIS_RULED``) -- ``train_cv``'s
    closed vocabulary and the shipped ``siamese_contrastive`` are both
    untouched. The objective travels in ``backbone_config``, so
    ``make_factory`` returns the ranking head and every other part of
    the path is the probe's own.

    **The settings are RULED, and this task refuses to default any of
    them** (``EXIT_CRITERIA`` criterion 13): an unruled value reaches
    here as the sentinel -1 and stops the run rather than becoming a
    plausible number nobody chose.

    Fold honesty is structural: pairs are formed inside ``train_epoch``
    from that fold's training rows alone.
    """
    import json

    from . import phase22
    from .data.manifest import load_manifest
    from .train import phase3
    from .train.harness import TrainConfig
    from .train.pooled import PooledSource
    from .train.ranking import pair_census as ranking_pair_census

    task = ctx.config["task"]
    entries = {entry["name"]: entry for entry in ctx.inputs}
    declared = {name: Path(entry["path"]) for name, entry in entries.items()}

    # ---- criterion 13: no setting may be silently defaulted -----------
    for name in ("pretrain_epochs", "finetune_trainable_code"):
        value = task.get(name)
        if value is not None and int(value) < 0:
            raise ValueError(
                f"{name} is the UNRULED sentinel ({value}). "
                f"phase22.DECLARED_SETTINGS_22 lists it as still open and "
                "EXIT_CRITERIA criterion 13 forbids starting with it "
                "defaulted. the ruling is it before this arm runs."
            )

    bounded = bool(task["bounded"])
    ctx.log(
        f"arm {task['arm']}: pair_source={task['pair_source']}, "
        f"head={'bounded' if bounded else 'unbounded'}, "
        f"monitor={task['monitor']}"
    )
    if not bounded:
        ctx.log(
            "DECLARED LIMITATION (phase22.MONITOR_BIND_RESOLVED): this arm "
            "monitors inner_val_pcc, which harness.py's own comment calls "
            "'selecting on the thing being claimed'. Reported with every "
            "result this arm produces."
        )
        ctx.log(
            "prediction scale: the CSVs carry RAW SCORES, not label-scale "
            "values (phase22.DECLARED_SETTINGS_22 setting 5). Any mapping "
            "to 1-5 is presentation and is never a second arm."
        )

    # ---- pair_source -> min_separation, the ONLY thing separating -----
    # R-all from R-clear. [ADDED 2026-09-01: pair_source previously
    # reached only the log line, so the two arms trained on identical
    # pair sets and produced identical results -- four runs at 4472be00
    # measured one arm twice. phase22.PAIR_SOURCE_WAS_NOT_CONSUMED.]
    pair_source = task["pair_source"]
    if pair_source == "cohort_se_diff":
        declared_threshold = task.get("se_diff_threshold")
        if declared_threshold is None:
            raise ValueError(
                "pair_source 'cohort_se_diff' needs se_diff_threshold: it IS "
                "the restriction, and without it R-clear is R-all."
            )
        # EXIT_CRITERIA criterion 4: RECOMPUTED from sd_obs and
        # RELIABILITY_237 at build time and CHECKED against the declared
        # value -- never pinned as a literal and trusted.
        import math

        from .data import reliability
        from .data.manifest import load_manifest as _load_manifest

        rows = _load_manifest(
            declared[task["manifest_artifact"]] / "manifest.csv"
        )
        label_values = np.array(
            [float(r[task["label"]]) for r in rows], dtype=float
        )
        sd_obs = float(np.std(label_values))
        recomputed = sd_obs * math.sqrt(2.0) * math.sqrt(
            1.0 - float(reliability.RELIABILITY_237)
        )
        if abs(recomputed - float(declared_threshold)) > 5e-6:
            raise ValueError(
                f"se_diff_threshold {float(declared_threshold):.6f} does not "
                f"match the value recomputed from this manifest "
                f"({recomputed:.6f}, sd_obs {sd_obs:.6f}). Criterion 4 "
                "requires the threshold to be DERIVED, not pinned -- a "
                "mismatch means the config and the cohort disagree."
            )
        ctx.log(
            f"se_diff RECOMPUTED from sd_obs {sd_obs:.6f} = "
            f"{recomputed:.6f}, matching the declared "
            f"{float(declared_threshold):.6f}"
        )
        min_separation = recomputed
    elif pair_source == "synth_within_face":
        # **[FOUND 2026-09-01 BY THE SETTINGS SWEEP.]** R-syn's pair
        # source is NOT IMPLEMENTED: nothing here reads the synthetic
        # set, so falling through would silently train this arm on
        # COHORT pairs -- making it R-all under another name, the exact
        # defect the sweep was run to end. The arm is already blocked by
        # its UNRULED sentinel; this refuses it a second time, on its
        # own terms, so the block does not depend on that sentinel
        # staying in place.
        raise ValueError(
            "pair_source 'synth_within_face' is NOT IMPLEMENTED: R-syn "
            "must draw WITHIN-FACE pairs from the synthetic TPS set, and "
            "no code here reads that set. Running it now would train on "
            "cohort pairs and report them as R-syn."
        )
    else:
        min_separation = 0.0
    ctx.log(
        f"pair_source={pair_source} -> min_separation={min_separation:.6f}"
    )

    backbone_config = {
        "objective": "pairwise_logistic",
        "bounded": bounded,
        "min_separation": min_separation,
        "learning_rate": task.get("learning_rate", 1e-3),
        "weight_decay": task.get("weight_decay", 0.01),
    }
    pooled_source = None
    if task.get("embeddings_artifact"):
        # [FIXED 2026-09-01] `geometry` is REQUIRED and was omitted. It is
        # not a new setting: `task["geometry"]` is already a field on this
        # spec and is `g1` in all six configs. It is also not merely
        # carried -- PooledSource hands it to embeddings.check_pairing,
        # which VERIFIES it against the artifact's own metadata, so a
        # wrong value is caught rather than silently consumed.
        #
        # Built exactly as task_train_cv builds it (run.py, the donor
        # path): imagenet has no checkpoint, and Road A consumes
        # "pooled". Both are the dataclass defaults and are left to it.
        pooled_source = PooledSource(
            directory=declared[task["embeddings_artifact"]],
            backbone=task["backbone"],
            init=task.get("init", "imagenet"),
            geometry=task["geometry"],
        )

    per_seed = {}
    for seed in [int(s) for s in task["seeds"]]:
        ctx.log(f"--- seed {seed} ---")
        result = phase3.run(
            manifest_dir=declared[task["manifest_artifact"]],
            staged_dir=declared[task["staged_artifact"]],
            geometry=task["geometry"],
            label=task["label"],
            backbone=task["backbone"],
            trainable="head",
            seed=seed,
            pooled_source=pooled_source,
            train_config=TrainConfig(
                max_epochs=task["max_epochs"],
                patience=task["patience"],
                inner_val_frac=task["inner_val_frac"],
                monitor=task["monitor"],
                seed=seed,
            ),
            backbone_config=backbone_config,
            log=ctx.log,
        )
        phase3.write_outputs(result, ctx, prefix=f"seed_{seed}__")
        per_seed[seed] = {
            "pcc": result.summary["oof"]["pcc"],
            "spearman": result.summary["oof"].get("spearman"),
        }
        # **The pair count, per arm, in the run log.** A count line here
        # would have made R-all and R-clear distinguishable at the first
        # launch instead of only in their configs.
        census = ranking_pair_census(
            np.array(
                [float(r[task["label"]]) for r in load_manifest(
                    declared[task["manifest_artifact"]] / "manifest.csv"
                )], dtype=float
            ),
            min_separation,
        )
        per_seed[seed]["pair_census"] = census
        ctx.log(
            f"  seed {seed}: OOF PCC {per_seed[seed]['pcc']:.4f} | "
            f"pairs trained {census['n_pairs_trained']} of "
            f"{census['n_pairs_total']} "
            f"({census['n_pairs_tied_dropped']} tied, "
            f"{census['n_pairs_below_threshold_dropped']} below threshold)"
        )

    with ctx.atomic("metrics.json", tier="SHAREABLE") as tmp:
        tmp.write_text(json.dumps(as_builtin({
            "arm": task["arm"],
            "pair_source": task["pair_source"],
            "min_separation": min_separation,
            "bounded": bounded,
            "monitor": task["monitor"],
            "per_seed": per_seed,
            "settings": {
                "loss": "pairwise_logistic",
                "targets": "hard",
                "tie_rule": "dropped",
                "pair_sampling": "all_pairs_per_epoch",
            },
            "declared_limitation": (
                None if bounded else
                phase22.MONITOR_BIND_RESOLVED["unbounded_arms"]
            ),
            "prediction_scale": (
                "label scale" if bounded else "RAW SCORES -- not label scale"
            ),
        }), indent=1), encoding="utf-8")


def task_tie_fraction(ctx: RunContext) -> None:
    """How many of the 27,966 pairs have EXACTLY EQUAL panel means?
    (``phase22.TIE_FRACTION_MEASUREMENT_DESIGNED``.)

    **A PRECONDITION of Phase 22's target-map ruling, not a Phase 22
    arm.** Hard targets assign every exact tie a definite WRONG order --
    ``m_i > m_j`` is false for a tie, so a tie silently receives P = 0
    rather than the 0.5 the panel supports, and no check would raise it.

    **The tie test is EXACT INTEGER EQUALITY on the recovered rater
    sums** (``phase22.TIE_COMPARISON_RULE_DECLARED``). ``soft_k * 5`` is
    an exact count, so each patient's five grades sum to an integer in
    [5, 25] and two patients tie iff those integers are equal. **No
    float comparison and no tolerance appear in the tie test.** The
    recovery is ``phase8c.rater_multiset``, which already ships and
    which **confirms the rater count from the data by refusing** a
    manifest that is not five raters.

    The float route would almost certainly agree -- the smallest
    non-zero separation is one rater-step, 0.2 -- but the integer route
    removes the question instead of bounding it, at no cost.

    POD ARITHMETIC: one manifest, 27,966 integer differences. No
    backbone, no GPU, no new artifact, and no gate.
    """
    import json
    import math

    from . import phase8c, phase22
    from .data.manifest import load_manifest

    task = ctx.config["task"]
    declared = {entry["name"]: entry for entry in ctx.inputs}
    rows = load_manifest(
        Path(declared[task["manifest_artifact"]]["path"]) / "manifest.csv"
    )
    if len(rows) != int(task["expect_patients"]):
        raise ValueError(
            f"{len(rows)} manifest rows, expected {task['expect_patients']}"
        )

    # ---- the integer form, recovered by the shipped function ----------
    # rater_multiset ASSERTS five raters and refuses otherwise, which is
    # how the rater count is confirmed rather than assumed.
    sums = np.array(
        [sum(phase8c.rater_multiset(row)) for row in rows], dtype=np.int64
    )
    n_raters = len(phase8c.rater_multiset(rows[0]))
    ctx.log(
        f"rater count CONFIRMED FROM THE DATA at {n_raters} for all "
        f"{len(rows)} patients -- rater_multiset refuses anything else"
    )

    # The two routes must agree: mean == sum / n_raters, exactly enough
    # that the integer form describes the same cohort the label does.
    means = np.array([float(row[task["label"]]) for row in rows], dtype=float)
    worst = float(np.max(np.abs(means - sums / n_raters)))
    if worst > 1e-9:
        raise ValueError(
            f"the '{task['label']}' column is not the rater sum over "
            f"{n_raters}: worst discrepancy {worst:.3e}. The integer form "
            "and the label describe different quantities, so a tie count "
            "from one would not be a tie count for the other"
        )
    ctx.log(f"label agrees with the integer form to {worst:.3e}")

    # ---- all C(n,2) separations, in integer rater-steps ---------------
    upper = np.triu_indices(len(sums), 1)
    steps = np.abs(sums[:, None] - sums[None, :])[upper]
    n_pairs = int(len(steps))
    if n_pairs != int(task["expect_pairs"]):
        raise ValueError(
            f"{n_pairs} pairs, expected {task['expect_pairs']}. This "
            "cohort's pair count disagrees with the separation run's; "
            "STOPPING rather than banking a tie fraction whose "
            "denominator is not the one on record"
        )

    n_ties = int(np.count_nonzero(steps == 0))
    tie_fraction = n_ties / n_pairs
    ctx.log(
        f"{n_ties} of {n_pairs} pairs are EXACT TIES "
        f"({tie_fraction:.4f}) -- integer equality, no tolerance"
    )

    # ---- the bottom of the distribution -------------------------------
    values, counts = np.unique(steps, return_counts=True)
    smallest = [
        {"rater_steps": int(v), "separation": float(v) / n_raters,
         "count": int(c), "fraction": float(c) / n_pairs}
        for v, c in list(zip(values, counts))[:int(task["report_smallest"])]
    ]
    for entry in smallest:
        ctx.log(
            f"  {entry['rater_steps']} rater-step(s) "
            f"(separation {entry['separation']:.4f}): {entry['count']} pairs"
        )

    # ---- cross-check against the separation run ------------------------
    # Same arithmetic as task_cohort_pair_separation: sd ddof=0, strict <.
    below = None
    expect_below = task.get("expect_below_se_diff")
    if expect_below is not None:
        from .data import reliability

        sd_obs = float(np.std(means))
        se_diff = sd_obs * math.sqrt(2.0) * math.sqrt(
            1.0 - float(reliability.RELIABILITY_237)
        )
        below = float(np.mean((steps / n_raters) < se_diff))
        if round(below, 4) != round(float(expect_below), 4):
            raise ValueError(
                f"below-SE_diff fraction {below:.4f} disagrees with the "
                f"separation run's {float(expect_below):.4f}. STOPPING: "
                "two measurements of the same cohort that differ here are "
                "not both right"
            )
        ctx.log(
            f"cross-check: below SE_diff ({se_diff:.6f}) = {below:.4f}, "
            f"reproducing the separation run"
        )

    # ---- the cell, against the REGISTERED boundary ---------------------
    boundary = float(task["boundary"])
    cell = ("ties_are_a_large_fraction" if tie_fraction >= boundary
            else "ties_are_a_small_fraction")
    ctx.log(
        f"tie fraction {tie_fraction:.4f} against the registered boundary "
        f"{boundary} -> {cell}"
    )
    ctx.log(
        phase22.TIE_FRACTION_MEASUREMENT_DESIGNED["readings"][cell]
    )

    with ctx.atomic("metrics.json", tier="SHAREABLE") as tmp:
        tmp.write_text(json.dumps(as_builtin({
            "n_patients": len(rows),
            "n_raters": n_raters,
            "n_pairs": n_pairs,
            "n_ties": n_ties,
            "tie_fraction": tie_fraction,
            "smallest_separations": smallest,
            "below_se_diff": below,
            "boundary": boundary,
            "cell_fired": cell,
            "comparison_rule": phase22.TIE_COMPARISON_RULE_DECLARED[
                "the_rule"],
            "reading": phase22.TIE_FRACTION_MEASUREMENT_DESIGNED[
                "readings"][cell],
        }), indent=1), encoding="utf-8")


def task_cross_arm_shrinkage(ctx: RunContext) -> None:
    """Is shrinkage a property of the arms, or of one run?
    (``phase21.CROSS_ARM_SHRINKAGE_REGISTERED``.)

    ``phase3.sanity_report`` has printed ``shrinkage =
    sd(prediction) / sd(truth)`` on every arm since 2026-07-28, but
    ``phase21.SHRINKAGE_MECHANISM``'s general claim rested on ONE SEED
    OF ONE ARM. This recomputes the same ratio across the ruled arm set
    from the cached CSVs -- the truth column is already in every one of
    them, so no new artifact and no new hash.

    **The denominator is a constant and that is VERIFIED, not assumed**:
    on the 237 cohort every arm scores the same truth vector, so the
    ratio is driven entirely by prediction spread.

    A reckoning input for Phase 22. DESCRIPTIVE, no ledger row.
    """
    import json

    from . import phase21
    from .data.manifest import load_manifest

    task = ctx.config["task"]
    declared = {entry["name"]: entry for entry in ctx.inputs}
    rows = load_manifest(
        Path(declared[task["manifest_artifact"]]["path"]) / "manifest.csv"
    )
    if len(rows) != int(task["expect_patients"]):
        raise ValueError(
            f"{len(rows)} manifest rows, expected {task['expect_patients']}"
        )
    patient_ids = [int(r["patient_id"]) for r in rows]
    manifest_truth = np.array([float(r["mean"]) for r in rows], dtype=float)

    by_seed, truth_by_seed = _p21_load_arm_predictions(
        ctx, task, declared, patient_ids, with_truth=True,
    )
    seeds = sorted(by_seed)
    names = sorted(by_seed[seeds[0]])

    # ---- the denominator, verified constant ---------------------------
    reference = truth_by_seed[seeds[0]][names[0]]
    mismatches = [
        f"{name} seed {seed}"
        for seed in seeds for name in names
        if not np.allclose(truth_by_seed[seed][name], reference, atol=1e-9)
    ]
    if mismatches:
        raise ValueError(
            f"the truth vector is NOT constant across arms: {mismatches[:5]} "
            "differ. sd(truth) is the denominator of every ratio here, and a "
            "varying denominator would make the arms incomparable"
        )
    if not np.allclose(reference, manifest_truth, atol=1e-9):
        raise ValueError(
            "the arms' truth column does not match the manifest's mean "
            "column; the ratio would be against a different label"
        )
    truth_is_constant = True
    sd_truth = float(np.std(reference))
    ctx.log(
        f"denominator VERIFIED constant across {len(names)} arms x "
        f"{len(seeds)} seeds and equal to the manifest: sd(truth) = "
        f"{sd_truth:.6f} (ddof=0). The ratio is driven ENTIRELY by "
        f"prediction spread"
    )

    # ---- the ratio, per arm per seed ----------------------------------
    by_arm = {}
    for name in names:
        per_seed = {
            str(seed): float(np.std(by_seed[seed][name]) / sd_truth)
            for seed in seeds
        }
        values = np.array(list(per_seed.values()))
        by_arm[name] = {
            "per_seed": per_seed,
            "mean": float(values.mean()),
            "sd": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "group": next(
                a["group"] for a in task["arms"] if a["name"] == name
            ),
        }
    means = np.array([by_arm[n]["mean"] for n in names])
    lowest = names[int(np.argmin(means))]
    highest = names[int(np.argmax(means))]
    ctx.log(
        f"shrinkage across {len(names)} arms: min {means.min():.4f} "
        f"({lowest}), median {float(np.median(means)):.4f}, max "
        f"{means.max():.4f} ({highest})"
    )

    by_group = {}
    for group in sorted({by_arm[n]["group"] for n in names}):
        values = np.array(
            [by_arm[n]["mean"] for n in names if by_arm[n]["group"] == group]
        )
        by_group[group] = {
            "n_arms": int(len(values)), "min": float(values.min()),
            "median": float(np.median(values)), "max": float(values.max()),
        }
        ctx.log(
            f"  {group}: {len(values)} arms, {values.min():.4f} - "
            f"{values.max():.4f} (median {float(np.median(values)):.4f})"
        )

    probe = task["probe_arm"]
    ctx.log(f"the probe ({probe}): shrinkage {by_arm[probe]['mean']:.4f}")

    # ---- the cell, on DECLARED thresholds, in DECLARED precedence -----
    fails_at = float(task["fails_to_shrink_at"])
    substantially_at = float(task["shrinks_substantially_at"])
    widely_at = float(task["varies_widely_at"])
    not_shrinking = [n for n in names if by_arm[n]["mean"] >= fails_at]
    # **[RULED 2026-09-01] The dispersion statistic is the SAMPLE SD of
    # the per-arm means, not their range.** The range was built,
    # simulated and REJECTED ON EVIDENCE: its estimand grows with arm
    # count, so at 64 arms it fires on genuinely homogeneous ones
    # (phase21.SHRINKAGE_DISPERSION_STATISTIC_RULED). The range is still
    # COMPUTED AND REPORTED below -- it is informative context -- but it
    # decides nothing.
    dispersion = float(np.std(means, ddof=1))
    spread_range = float(means.max() - means.min())
    if not_shrinking:
        cell = "some_arm_does_not_shrink"
    elif dispersion >= widely_at:
        cell = "shrinkage_varies_widely"
    elif float(means.max()) <= substantially_at:
        cell = "every_arm_shrinks_substantially"
    else:
        cell = None
    ctx.log(
        f"declared thresholds: fails_to_shrink >= {fails_at}, "
        f"substantially <= {substantially_at}, varies_widely SD >= "
        f"{widely_at}. Observed SD {dispersion:.4f} (ddof=1) over "
        f"{len(means)} arms, max {means.max():.4f}, range "
        f"{spread_range:.4f} (reported, not the cell driver), arms at or "
        f"above the fail line: {not_shrinking or 'none'}"
    )
    if cell is None:
        reading = (
            "NO COMMITTED CELL FIRES. Recorded as a dated OBSERVATION on "
            "phase17.UNPREDICTED_PATTERN's precedent; no cell is stretched "
            "to fit and no reading is invented after the numbers"
        )
        ctx.log("READING: " + reading)
    else:
        reading = _registered_reading(
            ctx,
            reading=phase21.CROSS_ARM_SHRINKAGE_REGISTERED["readings"][cell],
            joined=len(names), expected=int(task["expect_arms"]),
            what="the cross-arm shrinkage measurement",
        )

    summary = {
        "registration": phase21.CROSS_ARM_SHRINKAGE_REGISTERED,
        "status": (
            "DESCRIPTIVE reckoning input for Phase 22; no ledger row, no "
            "lock touched"
        ),
        "n_arms": len(names),
        "seeds": seeds,
        "truth_is_constant": truth_is_constant,
        "sd_truth": sd_truth,
        "sd_ddof": 0,
        "by_arm": by_arm,
        "by_group": by_group,
        "across_arms": {
            "min": float(means.min()), "min_arm": lowest,
            "median": float(np.median(means)),
            "max": float(means.max()), "max_arm": highest,
            "sd": dispersion,
            "sd_ddof": 1,
            "range": spread_range,
            "dispersion_statistic": (
                "SD of the per-arm means (ddof=1) -- the RULED statistic. "
                "The range is reported beside it and decides nothing "
                "(phase21.SHRINKAGE_DISPERSION_STATISTIC_RULED)"
            ),
        },
        "probe": {"name": probe, "shrinkage": by_arm[probe]["mean"]},
        "declared_thresholds": {
            "fails_to_shrink_at": fails_at,
            "shrinks_substantially_at": substantially_at,
            "varies_widely_at": widely_at,
        },
        "arms_not_shrinking": not_shrinking,
        "cell_fired": cell,
        "reading": reading,
    }
    with ctx.atomic("metrics.json", tier="SHAREABLE") as tmp:
        tmp.write_text(
            json.dumps(as_builtin(summary), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def task_p21_ensemble_probe(ctx: RunContext) -> None:
    """Phase 21 Arm A: does combining existing arms' PREDICTIONS exceed
    any of them individually? (``phase21.ARM_A_REGISTERED``.)

    **A simple mean, per seed, over the 64 ruled arms' per-patient OOF
    predictions.** No weights: a learned weighting fitted on the OOF
    predictions would be
    ``phase21.SELECTION_ON_EVALUATION_DATA_PROHIBITION`` with extra
    steps, and the unweighted mean has no free parameters to fit.

    The contrast is reported against **both anchors**: the probe (0.2520)
    under PLAN 4.3's full criterion, and the withdrawn Phase 7B concat
    result (+0.0073), because an ensemble gain quoted without the concat
    precedent beside it would read as though combination had never been
    tried.

    POD ARITHMETIC: cached CSVs and averages. No backbone, no GPU, no new
    artifact, and no gate -- nothing here touches a cohort pixel.
    """
    import json

    from . import phase21
    from .data.manifest import load_manifest
    from .eval import metrics
    from .train import phase3

    task = ctx.config["task"]
    declared = {entry["name"]: entry for entry in ctx.inputs}
    rows = load_manifest(
        Path(declared[task["manifest_artifact"]]["path"]) / "manifest.csv"
    )
    patient_ids = [int(r["patient_id"]) for r in rows]
    truth = np.array([float(r["mean"]) for r in rows])

    by_seed = _p21_load_arm_predictions(ctx, task, declared, patient_ids)
    seeds = sorted(by_seed)
    probe_arm = task["probe_arm"]

    ensemble_by_seed, probe_by_seed, per_arm = {}, {}, {}
    for seed in seeds:
        arms = by_seed[seed]
        stacked = np.vstack([arms[name] for name in sorted(arms)])
        ensemble = stacked.mean(axis=0)
        ensemble_by_seed[seed] = ensemble
        probe_by_seed[seed] = arms[probe_arm]
        for name, vector in arms.items():
            per_arm.setdefault(name, {})[seed] = float(metrics.pcc(truth, vector))
        ctx.log(
            f"  seed {seed}: ensemble PCC "
            f"{metrics.pcc(truth, ensemble):+.4f} over {len(arms)} arms"
        )

    ensemble_pcc = {s: float(metrics.pcc(truth, ensemble_by_seed[s]))
                    for s in seeds}
    probe_pcc = {s: per_arm[probe_arm][s] for s in seeds}
    ensemble_values = np.array([ensemble_pcc[s] for s in seeds])
    probe_values = np.array([probe_pcc[s] for s in seeds])

    # ---- PLAN 4.3, both conditions, through the shipped machinery ----
    per_seed_intervals = {}
    for seed in seeds:
        delta, lo, hi = metrics.paired_delta_bca(
            metrics.pcc, truth, ensemble_by_seed[seed], probe_by_seed[seed],
            n_boot=int(task["n_boot"]), seed=seed,
        )
        per_seed_intervals[str(seed)] = {
            "delta": delta, "lo": lo, "hi": hi,
            "excludes_zero": bool(lo > 0 or hi < 0),
        }
        ctx.log(
            f"  seed {seed}: ensemble - probe {delta:+.4f} "
            f"[{lo:+.4f}, {hi:+.4f}]"
        )
    n_excluding = sum(
        1 for v in per_seed_intervals.values() if v["excludes_zero"]
    )
    mean_delta = float(ensemble_values.mean() - probe_values.mean())
    threshold = phase3.combined_claimable_delta(
        float(ensemble_values.std(ddof=1)), len(seeds),
        float(probe_values.std(ddof=1)), len(seeds),
    )
    condition_1 = n_excluding == len(seeds)
    condition_2 = abs(mean_delta) > threshold["arm_means_95"]
    claimable = condition_1 and condition_2

    # ---- does it exceed any arm INDIVIDUALLY? ------------------------
    arm_means = {name: float(np.mean(list(v.values())))
                 for name, v in per_arm.items()}
    best_arm = max(arm_means, key=arm_means.get)
    ctx.log(
        f"ensemble mean {ensemble_values.mean():+.4f} against the best single "
        f"arm {best_arm} at {arm_means[best_arm]:+.4f}, and the probe at "
        f"{probe_values.mean():+.4f}"
    )

    # ---- both anchors, per the lock's criterion 2 --------------------
    concat_precedent = float(task["concat_precedent_delta"])
    ctx.log(
        f"against the withdrawn Phase 7B concat precedent (+{concat_precedent}): "
        f"this ensemble's gain over the probe is {mean_delta:+.4f}"
    )
    ctx.log("prohibition: " + phase21.SELECTION_ON_EVALUATION_DATA_PROHIBITION)

    if claimable:
        cell = "a_above_the_probe_and_claimable"
    elif mean_delta > 0:
        cell = "a_above_but_unresolved"
    else:
        cell = "a_at_or_below_the_probe"
    reading = _registered_reading(
        ctx, reading=phase21.READINGS_COMMITTED[cell],
        joined=len(patient_ids), expected=int(task["expect_patients"]),
        what="Phase 21 Arm A, the ensemble probe",
    )
    ctx.log("BINDING: " + phase21.PHASE_21_BINDING["the_binding"])

    summary = {
        "registration": phase21.ARM_A_REGISTERED,
        "arm_set": phase21.ARM_SET_RULED,
        "binding": phase21.PHASE_21_BINDING,
        "prohibition": phase21.SELECTION_ON_EVALUATION_DATA_PROHIBITION,
        "n_arms": len(task["arms"]),
        "n_patients": len(patient_ids),
        "seeds": seeds,
        "ensemble_pcc_by_seed": {str(s): ensemble_pcc[s] for s in seeds},
        "ensemble_pcc_mean": float(ensemble_values.mean()),
        "ensemble_pcc_sd": float(ensemble_values.std(ddof=1)),
        "probe_pcc_by_seed": {str(s): probe_pcc[s] for s in seeds},
        "probe_pcc_mean": float(probe_values.mean()),
        "arm_pcc_mean": arm_means,
        "best_single_arm": {"name": best_arm, "pcc_mean": arm_means[best_arm]},
        "exceeds_best_single_arm": bool(
            ensemble_values.mean() > arm_means[best_arm]
        ),
        "paired_vs_probe": {
            "per_seed": per_seed_intervals,
            "n_excluding_zero": n_excluding,
            "mean_delta": mean_delta,
            "threshold": threshold,
            "condition_1_all_seeds_exclude_zero": condition_1,
            "condition_2_delta_exceeds_combined_uncertainty": condition_2,
            "claimable": claimable,
        },
        "against_the_concat_precedent": {
            "phase_7b_concat_delta": concat_precedent,
            "status_there": "WITHDRAWN",
            "this_delta": mean_delta,
            "why_it_is_quoted": (
                "an ensemble gain quoted without the concat precedent "
                "beside it would read as though combination had never "
                "been tried (phase21.EXIT_CRITERIA criterion 2)"
            ),
        },
        "all_68_variant": (
            "NOT RUN: it would cross cohort sizes (64 arms on 237, 4 on "
            "236). Recorded with the reason, per EXIT_CRITERIA criterion 4"
        ),
        "reading": reading,
        "cell_fired": cell,
        "status": (
            "Arm A's ledger row is to be ruled and is NOT assumed "
            "here (phase21.EXIT_CRITERIA criterion 10)"
        ),
    }
    with ctx.atomic("metrics.json", tier="SHAREABLE") as tmp:
        tmp.write_text(
            json.dumps(as_builtin(summary), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def _p21_rater_disagreement(ctx, declared: dict, task: dict, rows: list):
    """Per-patient inter-rater sd, by BOTH routes, asserted equal.

    Route 1: ``soft_1..soft_5`` are the fraction of five raters awarding
    each grade, so ``soft_k * 5`` is an exact count and the rater
    multiset is recoverable exactly.

    Route 2: ``phase10_rater_grades`` reads the five rater columns from
    the score sheet directly.

    **Both, because the label join has produced two defects before**
    (``phase21.DISAGREEMENT_STATISTIC_PROPOSED``). One line of assertion
    costs nothing and closes the shape that produced them.
    """
    from .data import scoresheet as scoresheet_module

    n_raters = int(task["n_raters"])
    derived = []
    for row in rows:
        counts = [int(round(float(row[f"soft_{k}"]) * n_raters))
                  for k in range(1, 6)]
        if sum(counts) != n_raters:
            raise ValueError(
                f"patient {row['patient_id']}: soft labels give {sum(counts)} "
                f"raters, not {n_raters}"
            )
        grades = np.repeat(np.arange(1, 6), counts).astype(float)
        derived.append(float(grades.std(ddof=1)))
    derived = np.array(derived)

    if not task.get("scoresheet_artifact"):
        raise ValueError(
            "task.scoresheet_artifact is required: the derived rater sd is "
            "cross-checked against phase10_rater_grades, and the label join "
            "has produced two defects before"
        )
    by_id = {int(r["patient_id"]): r for r in rows}
    sheet = Path(declared[task["scoresheet_artifact"]]["path"])
    if task.get("scoresheet_file"):
        sheet = sheet / task["scoresheet_file"]
    patient_ids = [int(r["patient_id"]) for r in rows]
    per_rater = phase10_rater_grades(
        Path(declared[task["manifest_artifact"]]["path"]), sheet, patient_ids
    )
    matrix = np.vstack([np.asarray(per_rater[r], dtype=float)
                        for r in sorted(per_rater)])
    direct = matrix.std(axis=0, ddof=1)
    worst = float(np.max(np.abs(derived - direct)))
    if worst > 1e-9:
        raise ValueError(
            f"the two rater-sd routes disagree by up to {worst:.3e}: the "
            "soft-label reconstruction and phase10_rater_grades are not "
            "measuring the same panel. A FINDING, not a tolerance to widen"
        )
    ctx.log(
        f"rater sd by two routes agrees to {worst:.2e} "
        f"({len(sorted(per_rater))} rater columns, {scoresheet_module.__name__})"
    )
    return derived


def error_indicators(binarisation: str, residual: np.ndarray) -> np.ndarray:
    """The two registered binarisations, named EXPLICITLY.

    Neither is a default: an unrecognised name raises rather than
    falling through to whichever branch happens to be last, because a
    silent fallback would score one binarisation under the other's name
    -- the R2 shape, in the arm that exists to keep the two apart.

    **[MOVED TO MODULE LEVEL 2026-09-01]** It was nested inside
    ``task_p21_error_consistency``. Phase 22's diagnostic needs the same
    binarisations against a different comparison set, and a second
    implementation of either would be disqualifying, so the shipped one
    moved out and BOTH tasks call it. The body is unchanged.
    """
    if binarisation == "residual_sign":
        # Over-prediction is the "error" direction; the label is
        # arbitrary and kappa is symmetric under flipping it.
        return residual > 0
    if binarisation == "worst_quartile":
        # Worst quartile of |residual|: error rate 0.25 BY
        # CONSTRUCTION, which is what makes c_exp meaningful.
        magnitude = np.abs(residual)
        return magnitude >= np.quantile(magnitude, 0.75)
    raise ValueError(
        f"{binarisation!r} is not a registered binarisation; expected "
        "'residual_sign' or 'worst_quartile' "
        "(phase21.ARM_B_REGISTERED)"
    )


def error_consistency_matrix(errors: np.ndarray) -> np.ndarray:
    """Geirhos error consistency, pairwise over the rows of ``errors``.

    ``errors`` is (n_arms, n_patients) boolean. Returns the symmetric
    (n_arms, n_arms) kappa matrix with ones on the diagonal.

    **[MOVED TO MODULE LEVEL 2026-09-01]**, body unchanged, for the same
    reason as ``error_indicators``: one implementation, two callers.
    """
    n = len(errors)
    matrix = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            e1, e2 = errors[i], errors[j]
            p1, p2 = float(e1.mean()), float(e2.mean())
            c_exp = p1 * p2 + (1 - p1) * (1 - p2)
            c_obs = float(np.mean(e1 == e2))
            kappa = 0.0 if c_exp >= 1.0 else (c_obs - c_exp) / (1 - c_exp)
            matrix[i, j] = matrix[j, i] = kappa
    return matrix


def p23_contrast_record(
    contrast: dict, arm_a: dict, arm_b: dict,
    rho: float, half_width: float, threshold: float, full_n: int,
) -> dict:
    """One Phase 23 contrast's record, assembled where it can be TESTED.

    **[ADDED 2026-09-02.]** Extracted from ``task_p23_rope`` for one
    reason: the code -> record sweep could read the task's source for
    field NAMES but could not execute it without a cluster, so it
    asserted that ``observations`` reached metrics.json and stopped
    there. It did reach it, carrying nulls. This function takes two
    arms' per-fold structures and returns the row, so the sweep can run
    it on a fixture and assert what the fields CARRY
    (``phase23.THE_SWEEP_CHECKED_PRESENCE_NOT_CONTENT``).

    **It refuses a reduced n it cannot account for.** ``n`` plus the
    dropped cells plus the unpaired cells must equal ``full_n``; when
    they do not, something shortened the vector that this record has no
    words for, and a short n with no reason is exactly what must not be
    written again.
    """
    from . import rope

    a, b = contrast["a"], contrast["b"]
    report = rope.paired_fold_report(
        a, arm_a["pcc"], arm_a["detail"],
        b, arm_b["pcc"], arm_b["detail"],
    )
    x, n = report["x"], report["n"]
    probabilities = rope.probabilities(x, rho=rho, half_width=half_width)
    decision = rope.verdict(probabilities, threshold=threshold)
    observations = rope.describe(
        x, dropped=report["dropped"], unpaired=report["unpaired"]
    )
    accounted = n + observations["n_dropped"] + observations["n_unpaired"]
    if accounted != int(full_n):
        raise ValueError(
            f"contrast {contrast['key']!r} ({a} vs {b}) accounts for "
            f"{accounted} of {full_n} cells: n={n}, "
            f"{observations['n_dropped']} dropped, "
            f"{observations['n_unpaired']} unpaired. Seeds absent from "
            f"{a}: {arm_a['missing_seeds']}; from {b}: "
            f"{arm_b['missing_seeds']}. Refusing to write a reduced n "
            "with no reason beside it."
        )
    return {
        "row": contrast["row"],
        "a": a, "b": b,
        "n": n,
        "degrees_of_freedom": n - 1,
        "n_is_the_full_form": n == int(full_n),
        "probabilities": probabilities,
        "verdict": decision,
        # Criterion 9: described, gating nothing.
        "observations": observations,
        "cells_accounted": accounted,
    }


def task_p23_rope(ctx: RunContext) -> None:
    """Phase 23: P(rope) for each of the 45 contrasts in scope.

    **Benavoli's single-dataset Bayesian correlated t-test**, computed
    from banked per-fold PCC differences. The arithmetic is
    ``cleft.rope``; this task is plumbing.

    **Nothing banked is recomputed.** No delta, no threshold, no
    verdict, and no ledger row: the task reads prediction CSVs and
    produces posteriors (``phase23.EXIT_CRITERIA`` criterion 7).

    **The width is RECOMPUTED, never read as a constant.** The config's
    declared value is a CHECK, and a mismatch refuses the run --
    criterion 2, whose withdrawal clause is the strongest protection on
    the width.

    DESCRIPTIVE, no ledger row. POD ARITHMETIC over cached CSVs.
    """
    import json

    from . import phase23, rope
    from .cluster_csv import read_cluster_csv

    task = ctx.config["task"]
    declared = {entry["name"]: entry for entry in ctx.inputs}

    # ---- criterion 2: recompute the width, refuse a mismatch --------
    half_width = rope.rope_half_width(
        sd=float(task["probe_seed_sd"]), n_seeds=int(task["probe_n_seeds"])
    )
    declared_width = float(task["expect_rope_half_width"])
    # **EXACT, no tolerance on either side.** [2026-09-02: this compared
    # with 5e-7, which is still slack. The guard's purpose is refusing a
    # width nobody ruled, and any slack defeats it -- the config now
    # carries the formula's own value, so equality is the right test.
    # phase23.THE_WIDTH_WAS_A_ROUNDED_DISPLAY.]
    if half_width != declared_width:
        raise ValueError(
            f"the ROPE half-width recomputes to {half_width:.7f} but the "
            f"config declares {declared_width:.7f}. EXIT_CRITERIA "
            "criterion 2: the width is fixed before any posterior, and a "
            "moved width WITHDRAWS EVERY RESULT. Refusing rather than "
            "computing against a width nobody ruled."
        )
    rho = float(task["rho"])
    threshold = float(task["decision_threshold"])
    ctx.log(
        f"ROPE +/-{half_width:.6f} RECOMPUTED from sd "
        f"{task['probe_seed_sd']} n={task['probe_n_seeds']}, matching the "
        f"declared {declared_width}; rho={rho}, threshold P>{threshold}"
    )
    ctx.log(
        "attribution: width and threshold are THIS PROJECT'S; rho, the "
        "prior, the posterior form and P(rope)-as-integral are the "
        "source's (phase23.EXIT_CRITERIA['the_settings_as_ruled'])"
    )
    ctx.log(
        "rho's approximation travels: the heuristic was derived for "
        "random resamples and these are non-overlapping k-fold folds "
        "(literature passage 5, footnote 2)"
    )

    # ---- per-arm, per-seed, per-fold PCC ------------------------------
    def arm_folds(arm: str) -> dict:
        """An arm's ``seed -> {fold: pcc}``, from either declaration form.

        **[2026-09-02] Two forms, because the record holds both.** Some
        shipped configs declare each per-seed CSV as its own input
        (``<arm>__seed_<n>``); others declare the RUN DIRECTORY and let
        the reader find the files inside. The Phase 17, 16 and 7b arms
        are declared the second way and could not be reached at all by
        the first -- which is why the completion needed this, not a new
        hash for every CSV.

        The filename inside a declared directory is not constructed
        guesswork: ``phase3.write_outputs`` writes
        ``<prefix>predictions.csv`` and every task passes
        ``prefix=f"seed_{seed}__"``. A missing file still fails loudly
        at the read.
        """
        by_seed, detail, missing = {}, {}, []
        directory = declared.get(arm)
        for seed in [int(s) for s in task["seeds"]]:
            entry = declared.get(f"{arm}__seed_{seed}")
            if entry is not None:
                path = Path(entry["path"])
            elif directory is not None:
                path = (
                    Path(directory["path"]) / f"seed_{seed}__predictions.csv"
                )
                if not path.is_file():
                    # **[2026-09-02] Recorded, not merely skipped.** This
                    # shortens n exactly as a degenerate fold does, and
                    # used to leave no trace whatever.
                    missing.append(int(seed))
                    continue
            else:
                missing.append(int(seed))
                continue
            rows = read_cluster_csv(path, expect=("patient_id", "fold"))
            # per_fold_pcc reads its answer from per_fold_detail, so the
            # numbers and the causes cannot disagree about which folds
            # are degenerate.
            detail[seed] = rope.per_fold_detail(rows)
            by_seed[seed] = {f: d["pcc"] for f, d in detail[seed].items()}
        if not by_seed:
            raise ValueError(
                f"arm {arm!r} has no reachable per-seed CSVs. Declare "
                "either one input per seed named "
                f"{arm}__seed_<n>, or the run DIRECTORY as {arm!r}. "
                "Every contrast needs both arms' per-fold vectors."
            )
        return {"pcc": by_seed, "detail": detail, "missing_seeds": missing}

    # **[2026-09-02] Refuse a silent partial BEFORE any posterior.**
    # The 30-contrast config called itself incomplete in its own header;
    # the completed one must be unable to run short without saying so.
    # This checks the DECLARED list, so a truncated config fails here
    # rather than after computing whatever it happened to carry.
    if len(task["contrasts"]) != int(task["expect_contrasts"]):
        raise ValueError(
            f"the config declares {len(task['contrasts'])} contrasts but "
            f"expect_contrasts is {task['expect_contrasts']}. A partial "
            "run is refused: EXIT_CRITERIA criterion 8 fixes the scope, "
            "and a short config that ran anyway would report a subset "
            "as if it were the family."
        )

    cache: dict[str, dict] = {}
    results = {}
    for contrast in task["contrasts"]:
        key, a, b = contrast["key"], contrast["a"], contrast["b"]
        for arm in (a, b):
            if arm not in cache:
                cache[arm] = arm_folds(arm)
        results[key] = p23_contrast_record(
            contrast, cache[a], cache[b],
            rho=rho, half_width=half_width, threshold=threshold,
            full_n=int(task["full_n"]),
        )
        n = results[key]["n"]
        probabilities = results[key]["probabilities"]
        decision = results[key]["verdict"]
        ctx.log(
            f"  {key}: n={n} (df {n - 1})  P(left) "
            f"{probabilities['left']:.4f}  P(rope) "
            f"{probabilities['rope']:.4f}  P(right) "
            f"{probabilities['right']:.4f}  -> {decision.upper()}"
        )
        if n != int(task["full_n"]):
            ctx.log(
                f"      n != {task['full_n']}: this contrast's posterior "
                f"has {n - 1} degrees of freedom, not "
                f"{int(task['full_n']) - 1}"
            )
            # **[2026-09-02] The reason, beside the reduced n.** It used
            # to print the shortfall and not its cause, which is the
            # same silence metrics.json carried.
            for drop in results[key]["observations"]["dropped_folds"]:
                for cause in drop["causes"]:
                    ctx.log(
                        f"      DROPPED seed {drop['seed']} fold "
                        f"{drop['fold']}: {cause['arm']} PCC undefined "
                        f"-- {cause['reason']} (n={cause['n_patients']}, "
                        f"sd truth {cause['truth_sd']:.6f}, sd "
                        f"prediction {cause['prediction_sd']:.6f})"
                    )
            for gap in results[key]["observations"]["unpaired_folds"]:
                ctx.log(
                    f"      UNPAIRED seed {gap['seed']} fold "
                    f"{gap['fold']}: present in {gap['present_in']}, "
                    f"absent from {gap['absent_from']} -- NOT a "
                    "degeneracy"
                )

    if len(results) != int(task["expect_contrasts"]):
        raise ValueError(
            f"{len(results)} contrasts computed, expected "
            f"{task['expect_contrasts']}. EXIT_CRITERIA criterion 8: "
            "the scope is fixed and nothing is added or dropped."
        )
    counts = {v: 0 for v in rope.VERDICTS}
    for entry in results.values():
        counts[entry["verdict"]] += 1
    ctx.log(f"verdicts: {counts}")

    with ctx.atomic("metrics.json", tier="SHAREABLE") as tmp:
        tmp.write_text(json.dumps(as_builtin({
            "rope_half_width": half_width,
            "width_provenance": (
                "RECOMPUTED from combined_claimable_delta at run time; "
                "the config's value was a check, not the source"
            ),
            "rho": rho,
            "rho_approximation": (
                "derived for random resamples, applied to "
                "non-overlapping k-fold folds -- the paper's footnote 2"
            ),
            "decision_threshold": threshold,
            "attribution": {
                "ours": ["rope_half_width", "decision_threshold"],
                "source": ["rho", "prior", "posterior_form", "p_rope_integral"],
            },
            "n_contrasts": len(results),
            "verdict_counts": counts,
            "contrasts": results,
            "nothing_banked_was_recomputed": (
                "no delta, threshold or verdict from the ledger is "
                "recomputed here; the ledger is untouched"
            ),
            "no_decision_is_not_equivalence": (
                phase23.READINGS_COMMITTED["no_decision_is_NOT_equivalence"]
            ),
        }), indent=1), encoding="utf-8")


def task_p22_diagnostic(ctx: RunContext) -> None:
    """Phase 22's feature-difference diagnostic: does a ranking arm make
    the SAME errors as the 63 shrinking arms?
    (``phase22.FEATURE_DIFFERENCE_DIAGNOSTIC``.)

    **Registers nothing.** The readings and both thresholds were
    committed before any arm ran; this task reads them and reports which
    cell each arm fires.

    **The kappa arithmetic is the shipped one** -- ``error_indicators``
    and ``error_consistency_matrix``, the same functions
    ``task_p21_error_consistency`` calls. A second implementation of
    either would be disqualifying.

    **The thresholds are READ FROM phase21 AT RUN TIME**, never from the
    config, and the config's declared values are checked against them:
    EXIT_CRITERIA criterion 6 says that if either source figure has
    changed the phase STOPS rather than reading on.

    DESCRIPTIVE, no ledger row. POD ARITHMETIC over cached CSVs.
    """
    import json

    from . import phase21, phase22
    from .data.manifest import load_manifest

    task = ctx.config["task"]
    declared = {entry["name"]: entry for entry in ctx.inputs}
    rows = load_manifest(
        Path(declared[task["manifest_artifact"]]["path"]) / "manifest.csv"
    )
    patient_ids = [int(r["patient_id"]) for r in rows]
    truth = np.array([float(r["mean"]) for r in rows])

    # ---- criterion 6: thresholds from phase21, config only CHECKS ----
    high = {
        "residual_sign": float(
            phase21.ARM_B_OBSERVED["residual_sign"]["mean_kappa"]),
        "worst_quartile": float(
            phase21.ARM_B_OBSERVED["worst_quartile"]["mean_kappa"]),
    }
    low = {
        name: float(value) for name, value in
        phase21.ARM_A_THE_SHRINKAGE_CONTROL["kappa_measured"].items()
    }
    for binarisation, declared_high in task["expect_high_kappa"].items():
        if abs(high[binarisation] - float(declared_high)) > 1e-9:
            raise ValueError(
                f"HIGH threshold for {binarisation} is {high[binarisation]} "
                f"in phase21.ARM_B_OBSERVED but the config expects "
                f"{declared_high}. EXIT_CRITERIA criterion 6: the phase "
                "STOPS and the change is investigated."
            )
    for binarisation, declared_low in task["expect_low_kappa"].items():
        if abs(low[binarisation] - float(declared_low)) > 1e-9:
            raise ValueError(
                f"LOW threshold for {binarisation} is {low[binarisation]} "
                f"in phase21.ARM_A_THE_SHRINKAGE_CONTROL but the config "
                f"expects {declared_low}. EXIT_CRITERIA criterion 6: the "
                "phase STOPS and the change is investigated."
            )
    ctx.log(
        f"thresholds READ from phase21 -- HIGH {high}, LOW {low}; "
        "the config's values matched"
    )

    by_seed = _p21_load_arm_predictions(ctx, task, declared, patient_ids)
    seeds = sorted(by_seed)
    names = sorted(by_seed[seeds[0]])
    subjects = list(task["subject_arms"])
    comparison = [n for n in names if n not in subjects]
    missing = [n for n in subjects if n not in names]
    if missing:
        raise ValueError(
            f"subject_arms names {missing}, which the arm list does not "
            f"declare. Declared arms: {names}"
        )
    if len(comparison) != int(task["expect_comparison_arms"]):
        raise ValueError(
            f"{len(comparison)} comparison arms, expected "
            f"{task['expect_comparison_arms']}. The diagnostic is defined "
            "against the 63 SHRINKING arms; arm A is excluded because it "
            "is the reference the LOW threshold comes from, and putting it "
            "on both sides would compare it with itself."
        )
    ctx.log(
        f"{len(subjects)} subject arm(s) against {len(comparison)} "
        f"comparison arms, {len(seeds)} seeds"
    )

    index = {name: i for i, name in enumerate(names)}
    results = {}
    for binarisation in task["binarisations"]:
        matrices = []
        for seed in seeds:
            errors = np.vstack([
                error_indicators(binarisation, by_seed[seed][name] - truth)
                for name in names
            ])
            matrices.append(error_consistency_matrix(errors))
        mean_matrix = np.mean(matrices, axis=0)

        per_arm = {}
        for subject in subjects:
            row = mean_matrix[index[subject]]
            values = np.array([row[index[c]] for c in comparison])
            mean_kappa = float(values.mean())
            if mean_kappa >= high[binarisation]:
                cell = "kappa_at_or_above_the_all_pairs_mean"
            elif mean_kappa <= low[binarisation]:
                cell = "kappa_at_or_below_arm_as_level"
            else:
                cell = "kappa_between"
            per_arm[subject] = {
                "mean_kappa": mean_kappa,
                "sd": float(values.std(ddof=1)),
                "n_comparison_arms": len(comparison),
                "cell_fired": cell,
                "reading": phase22.FEATURE_DIFFERENCE_DIAGNOSTIC[
                    "readings"][cell],
            }
            ctx.log(
                f"  {binarisation} / {subject}: mean kappa "
                f"{mean_kappa:+.4f} against HIGH {high[binarisation]:+.4f} "
                f"and LOW {low[binarisation]:+.4f} -> {cell}"
            )
        results[binarisation] = per_arm

    with ctx.atomic("metrics.json", tier="SHAREABLE") as tmp:
        tmp.write_text(json.dumps(as_builtin({
            "thresholds": {"high": high, "low": low},
            "threshold_provenance": (
                "READ AT RUN TIME from phase21.ARM_B_OBSERVED and "
                "phase21.ARM_A_THE_SHRINKAGE_CONTROL, not from this config"
            ),
            "n_comparison_arms": len(comparison),
            "seeds": seeds,
            "by_binarisation": results,
            "reported_separately_never_averaged": (
                "the two binarisations answer different questions; "
                "collapsing them would be the R2 shape"
            ),
        }), indent=1), encoding="utf-8")


def task_rater_icc(ctx: RunContext) -> None:
    """The panel's ICCs, beside the banked Spearman-Brown reliability.

    **Why it needs the score sheet.** ICC(2,k) puts BETWEEN-RATER
    variance in its denominator, so it needs to know WHICH rater gave
    which grade. The manifest's ``soft_1..soft_5`` recover the per-patient
    multiset exactly and lose rater identity, so they cannot answer this
    (``phase21.INTER_RATER_DISAGREEMENT_MEASURED`` established the
    recovery; this is the case it does not cover).

    **The cross-check is the point of the first half.** Before any ICC,
    the recomputed Fleiss, QWK and mean inter-rater r are compared with
    the banked 237 constants. If they do not match, this is not the
    matrix the banked reliability came from and the ICC would not be
    comparable to 0.8158 -- so it refuses rather than reporting an ICC
    against a different panel.

    DESCRIPTIVE. No ledger row (``record_audit.ICC_PREDICTION_
    REGISTERED``).
    """
    import json

    from . import record_audit
    from .data import reliability as R
    from .data import scoresheet
    from .data.manifest import load_manifest

    task = ctx.config["task"]
    declared = {entry["name"]: entry for entry in ctx.inputs}
    manifest_dir = Path(declared[task["manifest_artifact"]]["path"])
    sheet_path = Path(declared[task["scoresheet_artifact"]]["path"])

    rows = load_manifest(manifest_dir / "manifest.csv")
    patient_ids = [int(row["patient_id"]) for row in rows]
    if len(patient_ids) != int(task["expect_patients"]):
        raise ValueError(
            f"{len(patient_ids)} manifest rows against the declared "
            f"{task['expect_patients']}"
        )

    by_rater = phase10_rater_grades(manifest_dir, sheet_path, patient_ids)
    if len(by_rater) != int(task["expect_raters"]):
        raise ValueError(
            f"{len(by_rater)} rater columns against the declared "
            f"{task['expect_raters']}"
        )
    # Column order is the score sheet's own, so the matrix is the one
    # every other reliability figure was computed from.
    matrix = np.column_stack([by_rater[name] for name in scoresheet.RATERS])
    matrix = matrix.astype(int)

    # ---- the cross-check, before any ICC ---------------------------
    recomputed = {
        "fleiss_kappa": R.fleiss_kappa(matrix),
        "mean_pairwise_qwk": R.mean_pairwise_qwk(matrix),
        "mean_inter_rater_r": R.mean_inter_rater_r(matrix),
    }
    banked = {
        "fleiss_kappa": R.FLEISS_237,
        "mean_pairwise_qwk": R.QWK_237,
        "mean_inter_rater_r": R.MEAN_R_237,
    }
    for name, value in recomputed.items():
        if abs(value - banked[name]) > 5e-5:
            raise ValueError(
                f"{name} recomputes to {value:.6f} against the banked "
                f"{banked[name]}. **This is not the matrix the banked "
                "reliability came from**, so an ICC from it would not be "
                "comparable to RELIABILITY_237 -- refusing rather than "
                "reporting one against a different panel."
            )
        ctx.log(f"cross-check {name}: {value:.4f} == banked {banked[name]}")

    # ---- the ICCs --------------------------------------------------
    icc = R.icc_two_way(matrix)
    spearman_brown = R.spearman_brown(recomputed["mean_inter_rater_r"], matrix.shape[1])
    gap = spearman_brown - icc["icc_2_k"]
    ctx.log(
        f"ICC(2,k) {icc['icc_2_k']:.4f}  ICC(3,k) {icc['icc_3_k']:.4f}  "
        f"Spearman-Brown {spearman_brown:.4f}"
    )
    ctx.log(
        f"the gap Spearman-Brown - ICC(2,k) = {gap:+.4f} -- "
        "the systematic rater offset r cannot see "
        "(record_audit.ICC_PREDICTION_REGISTERED)"
    )
    ctx.log(
        "PREDICTED BEFORE THE NUMBER: ICC(2,k) <= Spearman-Brown, because "
        "Pearson r is invariant to an additive offset and ICC(2,k) is not"
    )

    ceiling_range = sorted(
        [R.pcc_ceiling(max(icc["icc_2_k"], 0.0)), R.pcc_ceiling(spearman_brown)]
    )
    ctx.log(
        f"ceiling as a RANGE: [{ceiling_range[0]:.4f}, "
        f"{ceiling_range[1]:.4f}] -- the banked 0.9032 is the upper "
        "endpoint and is NOT withdrawn "
        "(record_audit.THE_CEILING_AS_A_RANGE)"
    )

    with ctx.atomic("metrics.json", tier="SHAREABLE") as tmp:
        tmp.write_text(json.dumps(as_builtin({
            "n_subjects": icc["n_subjects"],
            "n_raters": icc["n_raters"],
            "cross_check": {"recomputed": recomputed, "banked": banked},
            "icc": icc,
            "spearman_brown_k": spearman_brown,
            "gap_spearman_brown_minus_icc_2k": gap,
            "pcc_ceiling_range": ceiling_range,
            "the_form_is_not_chosen_here": (
                "ICC(2,k) treats the raters as a random sample, ICC(3,k) as "
                "the population of interest; both are reported and the "
                "choice is the maintainer's "
                "(record_audit.ICC_FORM_DISTINCTION)"
            ),
            "prediction": record_audit.ICC_PREDICTION_REGISTERED[
                "the_prediction"
            ],
        }), indent=1), encoding="utf-8")


def task_p25_contrasts(ctx: RunContext) -> None:
    """Phase 25's three contrasts, on the two arms and the probe.

    **``phase7b.paired_comparison`` UNCHANGED** -- the per-seed paired
    BCa and the two-condition verdict come from the shipped function,
    not from a second implementation.

    **The pair list is not a config field.** It comes from
    ``phase25.contrast_family()`` -- two primaries and one secondary,
    fixed before either arm ran -- so the family cannot grow through a
    config edit (``THE_FAMILY_OF_THREE``: no alpha correction BECAUSE
    nothing is selected).

    **``winner_sd`` is COMPUTED, not declared.** ``paired_comparison``
    derives the baseline's sd from the loaded predictions and takes the
    winner's as an argument; a declared constant on one side and a
    measured value on the other is two quantities under one name, and a
    declared sd can disagree with the artifact it describes. Both sides
    are computed here from each arm's own per-seed PCCs.

    DESCRIPTIVE. A ledger row only where a contrast is claimable
    (``EXIT_CRITERIA`` criterion 9).
    """
    import json

    from . import phase25
    from .data.manifest import load_manifest
    from .eval.metrics import pcc
    from .phase7b import paired_comparison

    task = ctx.config["task"]
    declared = {entry["name"]: entry for entry in ctx.inputs}
    rows = load_manifest(
        Path(declared[task["manifest_artifact"]]["path"]) / "manifest.csv"
    )
    patient_ids = [int(r["patient_id"]) for r in rows]
    if len(patient_ids) != int(task["expect_patients"]):
        raise ValueError(
            f"{len(patient_ids)} manifest rows against the declared "
            f"{task['expect_patients']}"
        )
    truth = np.array([float(r["mean"]) for r in rows])

    by_seed = _p21_load_arm_predictions(ctx, task, declared, patient_ids)
    seeds = sorted(by_seed)
    available = set(by_seed[seeds[0]])

    family = phase25.contrast_family()
    if len(family) != int(task["expect_contrasts"]):
        raise ValueError(
            f"the family is {len(family)} contrasts but expect_contrasts "
            f"is {task['expect_contrasts']}"
        )
    missing = sorted(
        {c["a"] for c in family} | {c["b"] for c in family}
    ) and [
        arm for c in family for arm in (c["a"], c["b"])
        if arm not in available
    ]
    if missing:
        raise ValueError(
            f"contrast members not declared: {sorted(set(missing))}. "
            "All three contrasts run or none do -- a partial family "
            "would report a subset as if it were the family."
        )

    # Each arm's own per-seed PCC, and its sd. **Measured from the
    # artifacts, never from a config field or a message.**
    arm_pcc = {
        arm: [float(pcc(truth, by_seed[seed][arm])) for seed in seeds]
        for arm in sorted(available)
    }
    arm_stats = {
        arm: {
            "per_seed_pcc": scores,
            "mean": float(np.mean(scores)),
            "sd": float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0,
        }
        for arm, scores in arm_pcc.items()
    }
    for arm, stats in sorted(arm_stats.items()):
        ctx.log(
            f"arm {arm}: mean {stats['mean']:.4f} sd {stats['sd']:.4f} "
            f"over {len(seeds)} seeds"
        )

    verdicts = {}
    for contrast in family:
        winner = {s: by_seed[s][contrast["a"]] for s in seeds}
        baseline = {s: by_seed[s][contrast["b"]] for s in seeds}
        result = paired_comparison(
            truth=truth,
            winner_by_seed=winner,
            baseline_by_seed=baseline,
            winner_sd=arm_stats[contrast["a"]]["sd"],
            n_boot=int(task["n_boot"]),
        )
        # **BOTH PLAN 4.3 conditions, recorded explicitly.** The Phase 22
        # null-condition defect must not recur: a verdict without both is
        # a verdict a reader has to infer from an adjacent field.
        condition_1 = result["all_seeds_exclude_zero_one_direction"]
        condition_2 = result["exceeds_threshold"]
        if condition_1 is None or condition_2 is None:
            raise ValueError(
                f"{contrast['key']}: paired_comparison returned a null "
                "condition. A verdict may not be recorded without both."
            )
        verdicts[contrast["key"]] = {
            "kind": contrast["kind"],
            "question": contrast["question"],
            "a": contrast["a"], "b": contrast["b"],
            "varies": contrast["varies"],
            "n_seeds": result["n_seeds"],
            "n_patients": len(patient_ids),
            "threshold": result["threshold"],
            "condition_1": condition_1,
            "condition_2": condition_2,
            "verdict": (
                "claimable" if result["claimable"]
                else "withdrawn" if condition_1 and not condition_2
                else "unresolved"
            ),
            "result": result,
        }
        ctx.log(
            f"  {contrast['key']}: delta "
            f"{result['mean_delta']:+.4f} | n={result['n_seeds']} seeds "
            f"x {len(patient_ids)} patients | condition_1 {condition_1} "
            f"condition_2 {condition_2} -> "
            f"{verdicts[contrast['key']]['verdict'].upper()}"
        )

    # **[CORRECTED 2026-09-02] This logged a +0.0656 delta that does not
    # exist** -- it came from arm figures that were in a message and in no
    # artifact (phase25.THE_FABRICATED_FIGURES_WITHDRAWN). A fabricated
    # number hardcoded into a task would have been printed into every run
    # of it as though the run had produced it. The band is real and stays;
    # the delta is now READ from the arms above.
    ctx.log(
        "REGISTERED BEFORE THESE NUMBERS: the cohort has never resolved a "
        "delta in the 0.04-0.10 band and the smallest ever resolved here "
        "is 0.1386, so seeds agreeing in direction may still return "
        "UNRESOLVED -- both PLAN 4.3 conditions decide, not the sweep "
        "(phase25.THE_DELTA_IS_INSIDE_THE_UNRESOLVED_BAND, WITHDRAWN for "
        "its subject; the band argument stands)"
    )

    with ctx.atomic("metrics.json", tier="SHAREABLE") as tmp:
        tmp.write_text(json.dumps(as_builtin({
            "family_size": len(family),
            "arm_stats": arm_stats,
            "verdicts": verdicts,
            "registered_before_the_numbers": (
                "phase25.THE_DELTA_IS_INSIDE_THE_UNRESOLVED_BAND; "
                "phase25.READINGS_COMMITTED; "
                "phase25.THE_ATTRIBUTION_PROBLEM"
            ),
        }), indent=1), encoding="utf-8")


def task_p22_contrasts(ctx: RunContext) -> None:
    """Phase 22's contrast family, on the evaluable subset.

    **``phase7b.paired_comparison`` UNCHANGED** -- the per-seed paired
    BCa and the two-condition verdict come from the shipped function,
    not from a second implementation.

    **The pair list is not a config field.** It is derived from
    ``phase22.contrast_family()`` and filtered to the contrasts whose
    both members are declared, so the family cannot grow through a
    config edit (``CONTRAST_FAMILY``: nothing added after the lock).

    DESCRIPTIVE, no ledger row.
    """
    import json

    from . import phase22
    from .data.manifest import load_manifest
    from .phase7b import paired_comparison

    task = ctx.config["task"]
    declared = {entry["name"]: entry for entry in ctx.inputs}
    rows = load_manifest(
        Path(declared[task["manifest_artifact"]]["path"]) / "manifest.csv"
    )
    patient_ids = [int(r["patient_id"]) for r in rows]
    truth = np.array([float(r["mean"]) for r in rows])

    by_seed = _p21_load_arm_predictions(ctx, task, declared, patient_ids)
    seeds = sorted(by_seed)
    available = set(by_seed[seeds[0]])

    family = phase22.contrast_family()
    if len(family) != 15:
        raise ValueError(
            f"the family is {len(family)} contrasts, not the locked 15"
        )
    evaluable = [
        c for c in family
        if c["a"] in available and c["b"] in available
    ]
    deferred = [c["key"] for c in family if c not in evaluable]
    ctx.log(
        f"{len(evaluable)} of {len(family)} contrasts evaluable; "
        f"{len(deferred)} DEFERRED (both members must be declared)"
    )
    if len(evaluable) != int(task["expect_evaluable"]):
        raise ValueError(
            f"{len(evaluable)} evaluable contrasts, expected "
            f"{task['expect_evaluable']}. A mismatch means the declared "
            "arms and the locked family disagree about what can be run."
        )

    verdicts = {}
    for contrast in evaluable:
        winner = {s: by_seed[s][contrast["a"]] for s in seeds}
        baseline = {s: by_seed[s][contrast["b"]] for s in seeds}
        result = paired_comparison(
            truth=truth,
            winner_by_seed=winner,
            baseline_by_seed=baseline,
            winner_sd=float(task["winner_sd"]),
            n_boot=int(task["n_boot"]),
        )
        # **[FIXED 2026-09-02] Both PLAN 4.3 conditions, recorded.**
        # The first version stored the verdict string and neither
        # condition, so all eight rows carried `condition_1: None` --
        # and a reader would have had to infer which condition failed
        # from `n_excluding_zero`, which is the read-it-from-an-
        # adjacent-field habit this project has been burned by.
        # Both come from paired_comparison's own result; neither is
        # recomputed here.
        condition_1 = result["all_seeds_exclude_zero_one_direction"]
        condition_2 = result["exceeds_threshold"]
        if condition_1 is None or condition_2 is None:
            raise ValueError(
                f"{contrast['key']}: paired_comparison returned a null "
                "condition. A verdict may not be recorded without both."
            )
        verdicts[contrast["key"]] = {
            "kind": contrast["kind"],
            "question": contrast["question"],
            "a": contrast["a"], "b": contrast["b"],
            "varies": contrast["varies"],
            "condition_1": condition_1,
            "condition_2": condition_2,
            "verdict": (
                "claimable" if result["claimable"]
                else "withdrawn" if condition_1 and not condition_2
                else "unresolved"
            ),
            "result": result,
        }
        claimable = bool(result.get("claimable"))
        ctx.log(
            f"  {contrast['key']}: delta "
            f"{result.get('mean_delta', float('nan')):+.4f} | "
            f"condition_1 {condition_1} condition_2 {condition_2} -> "
            f"{verdicts[contrast['key']]['verdict'].upper()}"
        )

    with ctx.atomic("metrics.json", tier="SHAREABLE") as tmp:
        tmp.write_text(json.dumps(as_builtin({
            "family_size": len(family),
            "evaluable": [c["key"] for c in evaluable],
            "deferred": deferred,
            "deferred_owe_a_verdict": (
                "EXIT_CRITERIA criterion 8: all fifteen need a verdict "
                "before the phase closes. These are DEFERRED, not "
                "withdrawn -- they wait on R-syn"
            ),
            "verdicts": verdicts,
        }), indent=1), encoding="utf-8")


def task_p21_error_consistency(ctx: RunContext) -> None:
    """Phase 21 Arm B: do architecturally diverse arms mispredict the
    SAME patients? (``phase21.ARM_B_REGISTERED``.)

    Geirhos, Meding & Wichmann (NeurIPS 2020) error consistency adapted
    to regression, on **two binarisations reported separately and never
    averaged together**: residual sign, and worst-quartile absolute
    residual.

    Deliverables: the pairwise consistency matrix, the per-patient
    hardness vector, and the tie-aware cross-reference against
    inter-rater disagreement with its permutation null at the OBSERVED
    tie structure (``phase21.DISAGREEMENT_STATISTIC_PROPOSED``).

    DESCRIPTIVE, no ledger row. POD ARITHMETIC over cached CSVs.
    """
    import json

    from . import phase21
    from .data.manifest import load_manifest
    from .phase11 import kendall_tau_b

    task = ctx.config["task"]
    declared = {entry["name"]: entry for entry in ctx.inputs}
    rows = load_manifest(
        Path(declared[task["manifest_artifact"]]["path"]) / "manifest.csv"
    )
    patient_ids = [int(r["patient_id"]) for r in rows]
    truth = np.array([float(r["mean"]) for r in rows])

    by_seed = _p21_load_arm_predictions(ctx, task, declared, patient_ids)
    seeds = sorted(by_seed)
    names = sorted(by_seed[seeds[0]])

    results = {}
    for binarisation in task["binarisations"]:
        matrices, hardness_per_seed = [], []
        for seed in seeds:
            errors = np.vstack([
                error_indicators(binarisation, by_seed[seed][name] - truth)
                for name in names
            ])
            hardness_per_seed.append(errors.sum(axis=0).astype(float))
            matrices.append(error_consistency_matrix(errors))
        mean_matrix = np.mean(matrices, axis=0)
        upper = mean_matrix[np.triu_indices(len(names), 1)]
        mean_kappa = float(upper.mean())
        hardness = np.mean(hardness_per_seed, axis=0)

        low, high = float(task["low_kappa"]), float(task["high_kappa"])
        cell = ("b_high_consistency" if mean_kappa >= high
                else "b_low_consistency" if mean_kappa <= low
                else "between")
        ctx.log(
            f"{binarisation}: mean pairwise kappa {mean_kappa:+.4f} "
            f"(sd {float(upper.std(ddof=1)):.4f}) over "
            f"{len(upper)} pairs; declared cells at <={low} and >={high}: "
            f"{cell.upper()}"
        )
        results[binarisation] = {
            "mean_kappa": mean_kappa,
            "kappa_sd_over_pairs": float(upper.std(ddof=1)),
            "n_pairs": int(len(upper)),
            "matrix": mean_matrix.tolist(),
            "arms": names,
            "hardness": hardness.tolist(),
            "cell": cell,
        }

    # ---- the cross-reference, tie-aware, null at the OBSERVED ties ---
    disagreement = _p21_rater_disagreement(ctx, declared, task, rows)
    levels, counts = np.unique(disagreement, return_counts=True)
    tie_structure = {
        "n_distinct_values": int(len(levels)),
        "largest_tie_group": int(counts.max()),
        "counts_by_value": {f"{v:.6f}": int(c) for v, c in zip(levels, counts)},
    }
    ctx.log(
        f"disagreement: {len(levels)} distinct values over "
        f"{len(disagreement)} patients, largest tie group {counts.max()}"
    )
    rng = np.random.default_rng(int(ctx.config["seed"]))
    for binarisation, block in results.items():
        hardness = np.array(block["hardness"])
        observed = kendall_tau_b(disagreement, hardness)
        work = hardness.copy()
        null = np.empty(int(task["n_permutations"]))
        for i in range(len(null)):
            rng.shuffle(work)
            null[i] = kendall_tau_b(disagreement, work)
        p_value = float(np.mean(np.abs(null) >= abs(observed)))
        block["cross_reference"] = {
            "statistic": "kendall_tau_b (phase11, corrected 2026-09-01)",
            "tau_b": observed,
            "permutation_p": p_value,
            "n_permutations": int(len(null)),
            "null_mean": float(null.mean()),
            "null_sd": float(null.std(ddof=1)),
            "tie_structure": tie_structure,
            "cell": (
                "b_hardness_correlates_with_disagreement"
                if p_value < float(task["alpha"])
                else "b_hardness_does_not_correlate_with_disagreement"
            ),
        }
        ctx.log(
            f"{binarisation}: tau-b(hardness, disagreement) {observed:+.4f}, "
            f"permutation p {p_value:.4f} over {len(null)} draws"
        )

    ctx.log("BINDING: " + phase21.PHASE_21_BINDING["the_binding"])
    ctx.log("  " + phase21.PHASE_21_BINDING["arm_b_cannot_unbound_phase_20"])

    summary = {
        "registration": phase21.ARM_B_REGISTERED,
        "thresholds": phase21.CONSISTENCY_THRESHOLDS_RULED,
        "statistic": phase21.DISAGREEMENT_STATISTIC_PROPOSED,
        "binding": phase21.PHASE_21_BINDING,
        "n_arms": len(names),
        "n_patients": len(patient_ids),
        "seeds": seeds,
        "by_binarisation": results,
        "patient_ids": patient_ids,
        "disagreement": disagreement.tolist(),
        "status": "DESCRIPTIVE, no ledger row (phase21.ARM_B_REGISTERED)",
    }
    with ctx.atomic("metrics.json", tier="CLUSTER-ONLY") as tmp:
        tmp.write_text(
            json.dumps(as_builtin(summary), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def task_metric_space_analysis(ctx: RunContext) -> None:
    """Phase 18: the five deliverables over the locked arm list
    (phase18.EXIT_CRITERIA). Pure CPU arithmetic over declared inputs --
    no patient images, no training, and NO GLOB anywhere: every arm
    reaches this task through the config's literal list, which the
    generator derives from phase18.ARM_LIST_LOCKED.
    """
    import json

    from . import classification, phase7b, phase18
    from .cluster_csv import PREDICTIONS_COLUMNS, read_cluster_csv
    from .data.manifest import load_manifest
    from .eval import metrics as frozen_metrics
    from .phase11 import kendall_tau_b

    task = ctx.config["task"]
    declared = {entry["name"]: entry for entry in ctx.inputs}
    manifest_dir = Path(declared[task["manifest_artifact"]]["path"])

    # ---- the truth, cross-checked ONCE and shared ---------------------
    rows = load_manifest(manifest_dir / "manifest.csv")
    patient_ids = [int(r["patient_id"]) for r in rows]
    truth_mean = np.array([float(r["mean"]) for r in rows])
    truth_class3 = np.array([int(r["class3"]) for r in rows])
    derived = frozen_metrics.to_3class(truth_mean)
    if int(np.sum(derived != truth_class3)):
        raise ValueError(
            "manifest class3 disagrees with the frozen collapse of the "
            "mean; the truth is not what this phase was locked against"
        )
    floor_class = classification.majority_baseline(truth_class3, 3)
    floor_iem = phase18.constant_predictor_iem(truth_mean)
    floor_macro = phase18.MACRO_F1_FLOOR

    # ---- D1 + D3 + D5: every locked arm, per seed, every metric -------
    table, vectors_by_arm = {}, {}
    index_of = {p: i for i, p in enumerate(patient_ids)}
    for arm in task["arms"]:
        run_dir = Path(declared[arm["input"]]["path"])
        # [2026-08-30, run p18-metric-space-2] The guard stays HARD;
        # the EXPECTATION is per group, from the lock: 237 for the
        # cohort arms, 236 for p12's both-views cohort (folder 238 has
        # no basal; phase12.STOP_1_MANIFEST "drop folder 238 -> 236
        # rows verbatim", arm A "RE-RUN on the 236 cohort"). Each arm
        # is scored on ITS documented rows (the ruling; to
        # revert, set every n_patients to 237 and drop the tau caveat).
        expected_rows = int(arm["n_patients"])
        arm_ids, per_seed, by_seed = None, {}, {}
        for seed in [int(s) for s in arm["seeds"]]:
            records = read_cluster_csv(
                run_dir / f"seed_{seed}__{arm['csv']}.csv",
                expect=PREDICTIONS_COLUMNS,
            )
            if len(records) != expected_rows:
                raise ValueError(
                    f"{arm['name']} seed {seed}: {len(records)} rows "
                    f"where the arm's documented shape is "
                    f"{expected_rows} (ARM_LIST_LOCKED)"
                )
            seed_ids = sorted(int(r["patient_id"]) for r in records)
            if arm_ids is None:
                arm_ids = seed_ids
                unknown = set(arm_ids) - set(patient_ids)
                if unknown:
                    raise ValueError(
                        f"{arm['name']}: patient ids "
                        f"{sorted(unknown)[:5]} are not in the manifest"
                    )
                rows_idx = [index_of[p] for p in arm_ids]
                arm_truth_mean = truth_mean[rows_idx]
                arm_truth_class3 = truth_class3[rows_idx]
            elif seed_ids != arm_ids:
                raise ValueError(
                    f"{arm['name']} seed {seed}: patient set differs "
                    "from this arm's other seeds"
                )
            by_id = {
                int(r["patient_id"]): float(r["prediction"])
                for r in records
            }
            predicted = np.array([by_id[p] for p in arm_ids])
            by_seed[seed] = predicted
            predicted_class = frozen_metrics.to_3class(predicted, clip=True)
            report = classification.prf_report(
                arm_truth_class3, predicted_class, 3, _check_phase10=False
            )
            per_seed[str(seed)] = {
                "pcc": frozen_metrics.pcc(arm_truth_mean, predicted),
                "spearman": frozen_metrics.spearman(arm_truth_mean, predicted),
                "accuracy": report["accuracy"],
                "macro_f1": report["f1_macro"],
                "per_class_f1": [
                    report["per_class"][str(c)]["f1"] for c in range(3)
                ],
                # [2026-08-30, run p18-metric-space] qwk_3cat's inputs
                # are GRADE SPACE (1-5): it collapses internally via
                # to_3class(clip=False), whose range guard refused the
                # first version's CLASS-SPACE arguments ([0,2]) on
                # every attempt -- the frozen metric's design working
                # against a bad caller. Grade-space in, default
                # clip_pred=True per its own contract: a regression
                # head may legitimately overshoot; the truth is never
                # clipped.
                "qwk_3cat": frozen_metrics.qwk_3cat(arm_truth_mean, predicted),
                "iem": phase18.iem_score(arm_truth_mean, predicted),
            }
        vectors_by_arm[arm["name"]] = by_seed
        pooled = {}
        for key in ("pcc", "spearman", "accuracy", "macro_f1",
                    "qwk_3cat", "iem"):
            values = [per_seed[str(s)][key] for s in arm["seeds"]]
            pooled[key] = {
                "mean": float(np.mean(values)),
                "sd": float(np.std(values, ddof=1)),
            }
        stacked = np.stack([by_seed[int(s)] for s in arm["seeds"]])
        table[arm["name"]] = {
            "group": arm["group"], "csv": arm["csv"],
            "n_seeds": len(arm["seeds"]),
            "n_patients": expected_rows,
            # [2026-08-30] The p17-IEM verification columns: IEM is a
            # SCALED error, so an arm's prediction distribution beside
            # the truth's is what separates a calibration mismatch
            # from a computation defect (phase18.P17_IEM_PATH_VERIFIED).
            "prediction_mean": float(stacked.mean()),
            "prediction_sd": float(stacked.std(ddof=1)),
            "truth_mean": float(arm_truth_mean.mean()),
            "truth_sd": float(arm_truth_mean.std(ddof=1)),
            "per_seed": per_seed, "pooled": pooled,
        }
        ctx.log(
            f"{arm['name']}: PCC {pooled['pcc']['mean']:+.4f} | macro F1 "
            f"{pooled['macro_f1']['mean']:.4f} | IEM "
            f"{pooled['iem']['mean']:.4f}"
        )

    # ---- the three rankings and their agreement -----------------------
    names = [arm["name"] for arm in task["arms"]]
    by_pcc = sorted(names, key=lambda n: -table[n]["pooled"]["pcc"]["mean"])
    by_f1 = sorted(
        names, key=lambda n: -table[n]["pooled"]["macro_f1"]["mean"]
    )
    by_iem = sorted(names, key=lambda n: table[n]["pooled"]["iem"]["mean"])
    by_qwk = sorted(
        names, key=lambda n: -table[n]["pooled"]["qwk_3cat"]["mean"]
    )

    def agreement(order_a, order_b):
        rank_a = {n: i for i, n in enumerate(order_a)}
        rank_b = {n: i for i, n in enumerate(order_b)}
        tau = kendall_tau_b(
            [rank_a[n] for n in names], [rank_b[n] for n in names]
        )
        return {
            "kendall_tau_b": tau,
            "top5_overlap": len(set(order_a[:5]) & set(order_b[:5])),
        }

    rankings = {
        "n_patients_caveat": (
            "four sets (the p12 view-ablation arms) rank on their "
            "documented 236 both-views cohort; every other set on 237 "
            "-- the tau readings mix the two cohort sizes by that much"
        ),
        "by_pcc_top5": by_pcc[:5],
        "by_macro_f1_top5": by_f1[:5],
        "by_iem_top5": by_iem[:5],
        "by_qwk_top5": by_qwk[:5],
        "pcc_vs_macro_f1": agreement(by_pcc, by_f1),
        "pcc_vs_iem": agreement(by_pcc, by_iem),
        "macro_f1_vs_iem": agreement(by_f1, by_iem),
        # [2026-08-30] D3 completed: the QWK ordering against BOTH
        # families, per exit criterion 3 (phase18.D3_COMPLETED).
        "pcc_vs_qwk": agreement(by_pcc, by_qwk),
        "macro_f1_vs_qwk": agreement(by_f1, by_qwk),
    }
    ctx.log(f"tau(PCC, macroF1) = "
            f"{rankings['pcc_vs_macro_f1']['kendall_tau_b']:.4f}; "
            f"tau(PCC, IEM) = {rankings['pcc_vs_iem']['kendall_tau_b']:.4f}")

    # ---- D4: floor-marked differences ---------------------------------
    floor = floor_macro["floor"]
    f1_means = {
        n: table[n]["pooled"]["macro_f1"]["mean"] for n in names
    }
    ordered = sorted(f1_means, key=lambda n: -f1_means[n])
    below_floor_pairs = sum(
        1 for a, b in zip(ordered, ordered[1:])
        if abs(f1_means[a] - f1_means[b]) < floor
    )
    d4 = {
        "derived_floor": floor_macro,
        "adjacent_pairs_below_floor": below_floor_pairs,
        "of_adjacent_pairs": len(names) - 1,
        "marking": (
            "every observed macro-F1 difference below the derived floor "
            "is UNINTERPRETABLE regardless of what the criterion says"
        ),
    }

    # ---- D2: the criterion under macro F1 on the named contrasts ------
    flip_table = {}
    for contrast in task["contrasts"]:
        for side in (contrast["winner"], contrast["baseline"]):
            if table[side]["n_patients"] != len(patient_ids):
                raise ValueError(
                    f"contrast {contrast['name']}: {side} is documented "
                    f"at {table[side]['n_patients']} rows; the named "
                    "contrasts pair on the full 237"
                )
        winner = vectors_by_arm[contrast["winner"]]
        baseline = vectors_by_arm[contrast["baseline"]]
        shared = sorted(set(winner) & set(baseline))
        winner_scores = [
            phase18.macro_f1_statistic(truth_mean, winner[s]) for s in shared
        ]
        result = phase7b.paired_comparison(
            truth=truth_mean,
            winner_by_seed={s: winner[s] for s in shared},
            baseline_by_seed={s: baseline[s] for s in shared},
            winner_sd=float(np.std(winner_scores, ddof=1)),
            n_boot=int(task["n_boot"]),
            statistic=phase18.macro_f1_statistic,
            statistic_name="macro_f1",
        )
        if result["statistic"] != "macro_f1":
            raise ValueError(
                "the metric-generic path was not exercised: "
                f"paired_comparison reports {result['statistic']!r}"
            )
        f1_verdict = ("claimable" if result["claimable"] else "unresolved")
        flip_table[contrast["name"]] = {
            "pcc_verdict_banked": contrast["pcc_verdict"],
            "f1_verdict": f1_verdict,
            "flipped": (
                f1_verdict != contrast["pcc_verdict"].split("_")[0]
            ),
            "result": result,
        }
        ctx.log(f"D2 {contrast['name']}: PCC "
                f"{contrast['pcc_verdict']} -> F1 {f1_verdict}")

    with ctx.atomic("metrics.json", tier="SHAREABLE") as tmp:
        tmp.write_text(json.dumps(as_builtin({
            "exit_criteria": phase18.EXIT_CRITERIA,
            "arm_list": phase18.ARM_LIST_LOCKED["locked"],
            "n_arms": len(names),
            "table": table,
            "rankings": rankings,
            "d2_flip_table": flip_table,
            "d2_banked_verdicts_untouched": (
                "the PCC column above is the ledger's, carried in the "
                "config -- never recomputed here"
            ),
            "d4": d4,
            "floors": {
                "class_majority": floor_class,
                "iem_constant_predictor": floor_iem,
                "macro_f1_single_patient": floor_macro,
            },
            "d5_caveats": phase18.DELIVERABLES_REGISTERED[
                "d5_defect_caveats_travel"],
            "cleftgnn_prohibition": phase18.DELIVERABLES_REGISTERED[
                "cleftgnn_iem_prohibition"],
            "standing_clause": phase18.DELIVERABLES_REGISTERED[
                "standing_clause_mse_objective"],
            "readings": {
                "d1": [phase18.DELIVERABLES_REGISTERED[k] for k in (
                    "d1_reading_concordant",
                    "d1_reading_discordant_with_mechanism",
                    "d1_reading_discordant_without_mechanism",
                )],
                "d3_registered_position": phase18.D3_COMPLETED[
                    "registered_position_qwk_over_macro_f1"],
                "d5": phase18.DELIVERABLES_REGISTERED["d5_readings"],
            },
            "d2_home": phase18.D2_HOME_RULED,
        }), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def task_classification_metrics(ctx: RunContext) -> None:
    """Post-Phase-15-closing addendum: precision, recall and F1 on the
    best arm's EXISTING predictions
    (classification.CLASSIFICATION_METRICS_SECONDARY).

    Pod arithmetic. No training, no inference, no GPU -- the five
    per-seed OOF CSVs the arm already wrote, discretised at the frozen
    2.5/3.5 thresholds and tabulated.

    **These figures are SECONDARY and DESCRIPTIVE.** They never become a
    claim and never become a ledger row; the readings that say so are
    written into this run's own metrics.json beside every number, so a
    figure cannot be lifted out of the artifact without them.
    """
    import json

    import numpy as np

    from . import classification, phase9
    from .cluster_csv import PREDICTIONS_COLUMNS, read_cluster_csv
    from .data.manifest import load_manifest

    task = ctx.config["task"]
    declared = {entry["name"]: Path(entry["path"]) for entry in ctx.inputs}
    run_dir = declared[task["arm_run"]]
    manifest_dir = declared[task["manifest_artifact"]]
    n_classes = int(task["n_classes"])
    seeds = [int(seed) for seed in task["seeds"]]

    # ---- the truth's classes come from the MANIFEST's own column ------
    # NOT re-derived from the grades here. `class3` is a stored column
    # whose schema says what it is -- "3-class collapse of the mean at
    # fixed thresholds 2.5/3.5" -- and a fourth local copy of that rule
    # is exactly the defect cluster_csv's docstring records three
    # instances of.
    rows = load_manifest(manifest_dir / "manifest.csv")
    class3_by_id = {int(r["patient_id"]): int(r["class3"]) for r in rows}
    ctx.log(f"manifest: {len(class3_by_id)} patients with a class3 column")

    # ---- the floor, and the assertion that it is the RIGHT floor ------
    cohort_truth = np.array(
        [class3_by_id[pid] for pid in sorted(class3_by_id)], dtype=int
    )
    floor = classification.majority_baseline(cohort_truth, n_classes)
    banked = float(task["expect_majority"])
    if abs(floor["accuracy"] - banked) > 5e-4:
        raise ValueError(
            f"majority baseline is {floor['accuracy']:.4f} but phase9 "
            f"banked {banked:.4f} for class3. The truth column is not "
            "the one this addendum was registered against -- most "
            "likely the MEDIAN grade collapsed rather than the MEAN "
            "(manifest schema: class3 is the collapse of the MEAN). "
            "Refusing: a metric quoted against the wrong floor is "
            "worse than no metric"
        )
    ctx.log(
        f"floor: majority {floor['accuracy']:.4f} (class "
        f"{floor['majority_class']}) reproduces phase9's {banked:.4f}; "
        f"its macro F1 is {floor['f1_macro']:.4f}"
    )

    # ---- per seed ------------------------------------------------------
    per_seed, seed_rows = {}, {}
    for seed in seeds:
        path = run_dir / f"seed_{seed}__predictions.csv"
        if not path.is_file():
            raise ValueError(
                f"{path.name} is missing from {run_dir.name}. This "
                "addendum reads what the arm already wrote and computes "
                "nothing it cannot read"
            )
        records = read_cluster_csv(path, expect=PREDICTIONS_COLUMNS)
        ids = [int(r["patient_id"]) for r in records]
        truth_continuous = np.array(
            [float(r["truth"]) for r in records], dtype=float
        )
        predictions = np.array(
            [float(r["prediction"]) for r in records], dtype=float
        )

        # The three-way consistency check: the CSV's own truth column,
        # collapsed by the FROZEN rule, must equal the manifest's stored
        # class3 for the same patient. If it does not, the arm did not
        # train on the label this addendum assumes.
        stored = np.array([class3_by_id[pid] for pid in ids], dtype=int)
        derived = classification.discretise(truth_continuous)
        disagreeing = int(np.sum(stored != derived))
        if disagreeing:
            raise ValueError(
                f"seed {seed}: {disagreeing} of {len(ids)} patients have "
                "a stored class3 that disagrees with their own truth "
                "column collapsed at 2.5/3.5. The arm's target and the "
                "manifest's class3 are not the same quantity"
            )

        report = classification.prf_report(
            stored, classification.discretise(predictions), n_classes
        )
        per_seed[str(seed)] = report
        seed_rows[seed] = (stored, classification.discretise(predictions))
        ctx.log(
            f"  seed {seed}: accuracy {report['accuracy']:.4f}, macro F1 "
            f"{report['f1_macro']:.4f}, weighted F1 "
            f"{report['f1_weighted']:.4f} (support {report['support']})"
        )

    # ---- pooled --------------------------------------------------------
    # POOLED means the five per-seed values summarised as mean and
    # five-seed SD -- the SAME shape every arm figure in this project
    # carries, so these sit beside the banked PCC without a second
    # convention. Averaging the PREDICTIONS across seeds first would
    # build a predictor that was never trained or evaluated, and its
    # metrics would not be comparable to anything banked.
    pooled = {}
    for key in (
        "accuracy", "precision_macro", "recall_macro", "f1_macro",
        "precision_weighted", "recall_weighted", "f1_weighted",
    ):
        values = [per_seed[str(seed)][key] for seed in seeds]
        pooled[key] = {
            "mean": float(np.mean(values)),
            "sd": float(np.std(values, ddof=1)),
            "n": len(values),
        }
    ctx.log(
        f"pooled: accuracy {pooled['accuracy']['mean']:.4f} "
        f"(sd {pooled['accuracy']['sd']:.4f}), macro F1 "
        f"{pooled['f1_macro']['mean']:.4f} "
        f"(sd {pooled['f1_macro']['sd']:.4f}) vs floor "
        f"{floor['accuracy']:.4f} / {floor['f1_macro']:.4f}"
    )

    # The supports are identical across seeds (same 237 patients, same
    # folds), so the support vector is stated ONCE as the cohort's --
    # and checked rather than assumed.
    supports = {tuple(per_seed[str(seed)]["support"]) for seed in seeds}
    if len(supports) != 1:
        raise ValueError(
            f"the five seeds disagree on class support: {sorted(supports)}. "
            "Every seed is out-of-fold over the same 237 patients"
        )

    summary = {
        "registration": classification.CLASSIFICATION_METRICS_SECONDARY,
        "three_not_five": classification.THREE_NOT_FIVE,
        "arm": task["arm"],
        "arm_run": run_dir.name,
        "n_classes": n_classes,
        "class_support": list(next(iter(supports))),
        "class_meaning": phase9.PROTOTYPE_CLASSIFIER_REGISTERED[
            "evaluation"
        ] if "evaluation" in phase9.PROTOTYPE_CLASSIFIER_REGISTERED else (
            "class3 at 2.5/3.5: 0 = grades 1-2 (low), 1 = grade 3 (mid), "
            "2 = grades 4-5 (high)"
        ),
        "per_seed": per_seed,
        "pooled": pooled,
        "pooled_means": (
            "the mean and five-seed SD OF THE PER-SEED METRICS -- the "
            "same shape the arm's banked PCC carries. Predictions were "
            "NOT averaged across seeds first"
        ),
        "majority_baseline": floor,
        "the_caveats": [
            classification.CLASSIFICATION_METRICS_SECONDARY[key]
            for key in (
                "caveat_a_support",
                "caveat_b_macro_averaging",
                "caveat_c_discretisation_is_reporting",
            )
        ],
        "not_a_claim": classification.CLASSIFICATION_METRICS_SECONDARY[
            "what_this_is_not"
        ],
        "no_ledger_row": (
            "NO LEDGER ROW. These figures describe predictions already "
            "banked under PCC; they add a second description, not a "
            "second result"
        ),
    }
    with ctx.atomic("metrics.json", tier="SHAREABLE") as tmp:
        tmp.write_text(
            json.dumps(as_builtin(summary), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def task_extract_mebeauty_embeddings(ctx: RunContext) -> None:
    """Phase 15, stop 4a: cleft-cohort embeddings through a MEBeauty
    checkpoint (phase15.STOP_4_BUILT).

    **The arm-to-run mapping is ASSERTED here**, from the pretraining
    run's own metrics.json, because it could not be confirmed on the
    laptop (phase15.STOP_3_PRETRAINING_BANKED). A swapped mapping fails
    at this line rather than propagating into a verdict.

    The set is written by MEBeauty's OWN writer into MEBeauty's OWN
    namespace: the ladder's ``embeddings.INITS`` stays closed, and the
    two lattices are never pooled (the ruling).

    CLUSTER-ONLY -- these are cleft-patient embeddings.
    """
    import json

    from . import mebeauty
    from . import phase15
    from .data.manifest import load_manifest
    from .provenance.hashing import hash_dir, hash_file
    from .train.extract import extract_features

    task = ctx.config["task"]
    declared = {entry["name"]: Path(entry["path"]) for entry in ctx.inputs}
    run_dir = declared[task["pretrain_run"]]
    init = task["init"]
    geometry = task["geometry"]
    declared_source = task["expect_source"]

    # ---- the mapping, asserted from the run's own record --------------
    metrics = json.loads(
        (run_dir / "metrics.json").read_text(encoding="utf-8")
    )
    mebeauty.assert_run_is_the_declared_arm(metrics, declared_source)
    variant = mebeauty.expected_mebeauty_variant(init, geometry)
    if variant != declared_source:
        raise ValueError(
            f"init {init!r} at {geometry!r} needs variant {variant!r} "
            f"but the config declares source {declared_source!r} -- the "
            "geometry-bound checkpoint rule (VARIANT_FOR_MEBEAUTY_INIT)"
        )
    ctx.log(
        f"arm mapping ASSERTED: run records source "
        f"{metrics.get('source')!r}, config declares {declared_source!r}, "
        f"init {init!r} at {geometry!r} -- agreed. Source-side test PCC "
        f"{metrics.get('source_side_test_pcc')} (PROVENANCE, not the "
        "verdict)"
    )

    checkpoint = run_dir / "pretrained.pt"
    if not checkpoint.is_file():
        raise ValueError(f"{checkpoint} does not exist")
    checkpoint_sha256 = hash_file(checkpoint)

    # ---- the cohort, in the manifest's own row order -------------------
    manifest_dir = declared[task["manifest_artifact"]]
    staged_dir = declared[task["staged_artifact"]]
    rows = load_manifest(manifest_dir / "manifest.csv")
    if len(rows) != int(task["expect_patients"]):
        raise ValueError(
            f"{len(rows)} manifest rows, expected {task['expect_patients']}"
        )
    patient_ids = [int(row["patient_id"]) for row in rows]
    images = np.load(staged_dir / f"staged_patient_{geometry}.npy")
    if images.shape[0] != len(patient_ids):
        raise ValueError(
            f"{images.shape[0]} staged images for {len(patient_ids)} "
            "patients -- the artifact and the manifest disagree"
        )

    values, metadata = extract_features(
        task["backbone"], init, images,
        checkpoint_path=checkpoint,
        batch_size=int(task.get("batch_size", 32)),
    )
    out_dir = ctx.repo_root / "data" / "embeddings" / task["out_version"]
    payload = mebeauty.save_set(
        out_dir, values, init=init, geometry=geometry, variant=variant,
        checkpoint_sha256=checkpoint_sha256, patient_ids=patient_ids,
        manifest_ids=patient_ids, run_name=ctx.run_dir.name,
    )
    # [2026-08-24] **TWO DIGESTS OVER TWO SCOPES, AND THE LOG NOW SAYS
    # WHICH IS WHICH.** `save_set` returns the PAYLOAD rollup -- the one
    # MANIFEST.json records -- while `declare_inputs.py` measures the
    # directory with MANIFEST.json in it. The first version printed the
    # payload digest unlabelled beside the word "rollup", which is a
    # fill waiting to happen: a wrong value that looks exactly like a
    # right one, and that guard 3 would then refuse with a mismatch
    # nobody could explain. `task_masked_scut` already prints both
    # under the repo's own names and `task_stage_mebeauty` already
    # recomputes after the manifest; this follows them.
    declare = hash_dir(out_dir)
    ctx.log(
        f"{task['out_version']}: {values.shape}; "
        f"rollup_sha256_for_configs {declare['rollup'][:8]} (DECLARE THIS) "
        f"/ payload_rollup {payload['rollup'][:8]} (in MANIFEST.json) "
        "-- CLUSTER-ONLY, MEBeauty namespace"
    )

    with ctx.atomic("metrics.json", tier="SHAREABLE") as tmp:
        tmp.write_text(
            json.dumps(as_builtin({
                "registration": phase15.STOP_4_BUILT,
                "init": init,
                "geometry": geometry,
                "variant": variant,
                "checkpoint_sha256": checkpoint_sha256,
                "source_side_test_pcc": metrics.get("source_side_test_pcc"),
                "artifact": task["out_version"],
                "payload_rollup": payload["rollup"],
                "rollup_sha256_for_configs": declare["rollup"],
                "two_digests": (
                    "payload_rollup covers values.npy + metadata.json and is "
                    "what MANIFEST.json records; rollup_sha256_for_configs "
                    "covers the directory AS IT STANDS, MANIFEST.json "
                    "included -- what declare_inputs.py measures and guard "
                    "3 verifies. **They cannot coincide**: the manifest "
                    "contains the payload digest, so hashing the "
                    "directory again necessarily gives a different value. "
                    "DECLARE THE SECOND ONE."
                ),
                "shape": list(values.shape),
                "extractor": metadata.get("backbone"),
                "namespace": "mebeauty -- never pooled with the ladder's",
                "the_verdict_is_not_here": (
                    "the probe on this set decides; the source-side PCC "
                    "above is provenance"
                ),
            }), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def task_probe_mebeauty(ctx: RunContext) -> None:
    """Phase 15, stop 4b: A's probe on a MEBeauty embedding set
    (phase15.STOP_4_BUILT).

    **Its own kind, by the ruling**: reusing ``train_cv`` would
    widen the ladder's ``init`` vocabulary, and a shared vocabulary is
    what lets an undeclared swap manufacture a factor effect -- the
    same property at the config layer that the ruling forbids at the
    set layer. Same ``phase3.run`` entry point, same derived recipe,
    second call site.

    The verdict is computed HERE with the threshold and comparator
    DECLARED IN THE CONFIG before any number existed, the
    within-geometry rule asserted, the ImageNet anchor in the reading,
    and the four caveats bound to the output.
    """
    import json

    from . import mebeauty
    from . import phase15
    from .data.manifest import load_manifest
    from .train import phase3
    from .train.harness import TrainConfig

    task = ctx.config["task"]
    declared = {entry["name"]: Path(entry["path"]) for entry in ctx.inputs}
    manifest_dir = declared[task["manifest_artifact"]]
    rows = load_manifest(manifest_dir / "manifest.csv")
    patient_ids = [int(row["patient_id"]) for row in rows]

    values, metadata = mebeauty.load_set(
        declared[task["mebeauty_embeddings"]], patient_ids
    )
    if metadata["init"] != task["init"] or (
        metadata["geometry"] != task["geometry"]
    ):
        raise ValueError(
            f"the set is {metadata['init']!r}@{metadata['geometry']!r} "
            f"and the config declares {task['init']!r}@"
            f"{task['geometry']!r} -- the arm and its features must agree"
        )
    ctx.log(
        f"probing {metadata['init']}@{metadata['geometry']} from variant "
        f"{metadata['variant']} ({values.shape[0]} patients x "
        f"{values.shape[1]}d), row order checked against cleft_v1"
    )

    seeds = [int(s) for s in task["seeds"]]
    common = dict(
        manifest_dir=manifest_dir,
        staged_dir=declared[task["staged_artifact"]],
        geometry=task["geometry"],
        label=task["label"],
        backbone=task["backbone"],
        trainable=task["trainable"],
        patch_scheme=task["patch_scheme"],
        feature_source=task["feature_source"],
        features_override=(values, {
            "source": "mebeauty set",
            "init": metadata["init"],
            "variant": metadata["variant"],
        }),
        backbone_config={
            "learning_rate": task["learning_rate"],
            "weight_decay": task["weight_decay"],
            "batch_size": task["batch_size"],
            "alpha": task.get("alpha", 1.0),
            "pooling": task.get("pooling", "mean"),
            "augmentation": task.get("augmentation"),
        },
        log=ctx.log,
    )
    results, pooled = [], []
    for seed in seeds:
        ctx.log(f"--- seed {seed} ---")
        result = phase3.run(
            **common, seed=seed,
            train_config=TrainConfig(
                max_epochs=task["max_epochs"],
                patience=task["patience"],
                inner_val_frac=task["inner_val_frac"],
                monitor=task["monitor"],
                seed=seed,
            ),
        )
        phase3.write_outputs(result, ctx, prefix=f"seed_{seed}__")
        results.append(result)
        pooled.append(result.summary["oof"]["pcc"])

    mean_pcc = float(np.mean(pooled))
    sd_pcc = float(np.std(pooled, ddof=1)) if len(pooled) > 1 else 0.0
    ctx.log(
        f"MEBeauty {metadata['init']}@{metadata['geometry']}: mean OOF "
        f"PCC {mean_pcc:+.4f} (sd {sd_pcc:.4f}) over {len(seeds)} seeds"
    )

    # ---- the verdict, against a comparator declared in the config -----
    verdict = mebeauty.verdict_delta(
        {"geometry": task["geometry"], "pcc": mean_pcc, "sd": sd_pcc},
        task["comparator_init"], task["comparator_geometry"],
    )
    threshold = phase3.combined_claimable_delta(
        sd_pcc, len(seeds),
        float(task["comparator_sd"]), int(task["comparator_n"]),
    )
    clears = abs(verdict["delta_vs_comparator"]) > threshold["arm_means_95"]
    ctx.log(
        f"vs {verdict['comparator']}: delta "
        f"{verdict['delta_vs_comparator']:+.4f} against threshold "
        f"{threshold['arm_means_95']:.4f} (PLAN 4.12.1, both arms' own "
        f"SDs) -- {'CLEARS' if clears else 'does NOT clear'} the bar"
    )
    ctx.log(
        f"vs ImageNet {verdict['imagenet_pcc']}: delta "
        f"{verdict['delta_vs_imagenet']:+.4f}. "
        + verdict["the_imagenet_anchor"]
    )
    caveats = phase15.caveats_for(metadata["variant"])
    for name, text in caveats.items():
        ctx.log(f"  caveat {name}: {text}")

    summary = {
        "registration": phase15.STOP_4_BUILT,
        "init": metadata["init"],
        "geometry": metadata["geometry"],
        "variant": metadata["variant"],
        "seeds": seeds,
        "pcc_by_seed": dict(zip(map(str, seeds), pooled)),
        "pcc_mean": mean_pcc,
        "pcc_sd": sd_pcc,
        "verdict": verdict,
        "threshold": threshold,
        "clears_the_bar": bool(clears),
        "caveats_bound": caveats,
        "status": (
            "DESCRIPTIVE ranking; PLAN 4.3 machinery only if a gap "
            "clears the paired bar, and COHORT_CANNOT_RESOLVE is the "
            "registered expectation. No ledger row"
        ),
    }
    with ctx.atomic("metrics.json", tier="SHAREABLE") as tmp:
        tmp.write_text(
            json.dumps(as_builtin(summary), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def task_contact_sheet(ctx: RunContext) -> None:
    """Phase 2's visual check. CLUSTER-ONLY PNGs plus a SHAREABLE report."""
    import json

    from .data.manifest import load_manifest
    from .geometry import contact
    from .geometry.patches import PatchConfig
    from .geometry.render import save_sheet

    task = ctx.config["task"]
    declared = {entry["name"]: Path(entry["path"]) for entry in ctx.inputs}

    artifact = declared[task["manifest_artifact"]]
    folders = declared[task["patient_folders"]]
    rows = load_manifest(artifact / "manifest.csv")
    selected = contact.select_patients(rows, task["n_patients"])
    ctx.log(f"rendering {len(selected)} patients from {artifact.name}")

    config = PatchConfig(
        boundary_rule=task.get("boundary_rule", "keep"),
        patch_seed=task.get("patch_seed", 1337),
    )

    results = []
    for geometry in task["geometries"]:
        result = contact.build_sheet(
            geometry,
            selected,
            folders,
            generator=task["generator"],
            config=config,
            columns=task.get("columns", 4),
        )
        # The filename contains "contact_sheet", so the tier guard refuses to let
        # it be marked SHAREABLE. It renders patient faces.
        target = ctx.path(f"contact_sheet_{geometry}.png", tier="CLUSTER-ONLY")
        save_sheet(
            result.sheet, target, labels=result.labels,
            columns=task.get("columns", 4), cell=result.cell,
        )
        ctx.log(f"  {geometry}: {target.name}")
        results.append(result)

    summary = contact.aggregate(results)
    ctx.path("metrics.json", tier="SHAREABLE").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print(json.dumps(summary, indent=2, sort_keys=True))
    for line in json.dumps(summary, indent=2, sort_keys=True).splitlines():
        ctx.log(line)


def task_anatomy_sweep(ctx: RunContext) -> None:
    """Render a grid of candidate anatomy placements on one sheet."""
    import json

    from .data.manifest import load_manifest
    from .geometry import contact
    from .geometry.render import save_sheet

    task = ctx.config["task"]
    declared = {entry["name"]: Path(entry["path"]) for entry in ctx.inputs}

    rows = load_manifest(declared[task["manifest_artifact"]] / "manifest.csv")
    selected = contact.select_patients(rows, task["n_patients"])
    variants = contact.variant_grid(
        [float(v) for v in task["v_offsets"]],
        [float(s) for s in task["scales"]],
        [float(s) for s in task.get("scales_x") or (1.0,)],
        [float(s) for s in task.get("box_scales_x") or (1.0,)],
    )
    ctx.log(f"sweeping {len(variants)} variants over {len(selected)} faces")

    result = contact.build_sweep(
        selected,
        declared[task["patient_folders"]],
        variants,
        geometry=task.get("geometry", "g2"),
    )
    target = ctx.path("contact_sheet_anatomy_sweep.png", tier="CLUSTER-ONLY")
    save_sheet(
        result.sheet, target, labels=result.labels,
        columns=len(selected), cell=result.cell,
    )

    clamped = contact.sweep_clamp_report(result)
    geometry_report = contact.sweep_geometry_report(
        variants, geometry=task.get("geometry", "g2")
    )
    summary = {
        "geometry": result.geometry,
        "n_variants": len(variants),
        "n_faces": len(selected),
        "variants": [v.label for v in variants],
        "layout": "one row per variant, one column per face, reading order",
        # A variant whose outermost regions were pulled back inside the frame is
        # not a candidate: it only looks wider. Boxes are labelled with "!" on
        # the sheet, and named here so the exclusion is not a judgement call.
        "clamped_by_variant": clamped,
        "variants_with_clamping": sorted(clamped),
        # Clamping and overlap per variant. Overlap matters because at some size
        # the 27 regions stop being distinct structures and become 27 views of
        # the same area -- which would leave the anatomy scheme producing 27
        # nodes while no longer testing what it claims to test.
        "per_variant": geometry_report,
        "candidates": [
            label for label, entry in geometry_report.items() if entry["candidate"]
        ],
    }
    ctx.path("metrics.json", tier="SHAREABLE").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    rendered = json.dumps(summary, indent=2, sort_keys=True)
    print(rendered)
    for line in rendered.splitlines():
        ctx.log(line)


def task_stage_and_patch(ctx: RunContext) -> None:
    """Phase 2 exit criteria 7, 8 and 10."""
    import json

    from .geometry import stage_build

    task = ctx.config["task"]
    declared = {entry["name"]: Path(entry["path"]) for entry in ctx.inputs}

    result = stage_build.build(
        manifest_dir=declared[task["manifest_artifact"]],
        patient_folders=declared[task["patient_folders"]],
        artifact_dir=ctx.repo_root / "data" / "staged" / task["out_version"],
        generators=tuple(task["generators"]),
        geometries=tuple(task["geometries"]),
        log=ctx.log,
    )

    ctx.path("metrics.json", tier="SHAREABLE").write_text(
        json.dumps(result.summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    rendered = json.dumps(result.summary, indent=2, sort_keys=True)
    print(rendered)
    for line in rendered.splitlines():
        ctx.log(line)


def task_segmentation_probe(ctx: RunContext) -> None:
    """Phase 4 §2: can lips be segmented from these crops at all?

    The gate that decides whether SymNose is built. CLUSTER-ONLY sheet, SHAREABLE
    verdict. Deliberately cheap -- a handful of patients, three standard methods,
    no pipeline.
    """
    import json

    from .data.manifest import load_manifest
    from .geometry import contact, segmentation
    from .geometry.render import Panel, cell_for, save_sheet, tile

    task = ctx.config["task"]
    declared = {entry["name"]: Path(entry["path"]) for entry in ctx.inputs}

    artifact = declared[task["manifest_artifact"]]
    folders = declared[task["patient_folders"]]
    rows = load_manifest(artifact / "manifest.csv")
    selected = contact.select_patients(rows, task["n_patients"])
    ctx.log(f"segmentation feasibility on {len(selected)} patients")

    spatial_prior = task.get("spatial_prior", segmentation.DEFAULT_SPATIAL_PRIOR)
    use_face_mask = bool(task.get("face_mask", True))
    relaxation = float(task.get("contour_relaxation", segmentation.CONTOUR_RELAXATION))
    normalisation = task.get("normalisation", segmentation.DEFAULT_NORMALISATION)
    min_depth = float(
        task.get("fissure_min_depth", segmentation.FISSURE_MIN_DEPTH)
    )
    split_rule = task.get("split_rule", segmentation.DEFAULT_SPLIT_RULE)
    ctx.log(
        f"spatial prior: {spatial_prior}, face mask: {use_face_mask}, "
        f"relaxation: {relaxation}, normalisation: {normalisation}, "
        f"fissure_min_depth: {min_depth}, split_rule: {split_rule}"
    )

    panels: list[Panel] = []
    labels: list[str] = []
    reports: list[dict] = []

    for row in selected:
        patient_id = int(row["patient_id"])
        image = segmentation.load_staged(
            folders, row, task.get("geometry", segmentation.DEFAULT_GEOMETRY)
        )
        region = segmentation.analysis_region(
            image, spatial_prior, use_face_mask=use_face_mask
        )

        # Both quantities, so the correction is visible rather than asserted:
        # `separability` is lip-versus-skin inside the region, `whole_frame` is
        # the run-1 number that turned out to measure face-versus-background.
        separability = segmentation.separability_of(image, region)
        whole_frame = segmentation.separability_of(image)

        # The original beside its candidates, so a reviewer can see what each
        # method claimed rather than only that it claimed something.
        panels.append(Panel(f"{patient_id} source", image))
        labels.append(f"{patient_id} source")

        per_method = {}
        results = segmentation.run_methods(
            image,
            spatial_prior,
            use_face_mask=use_face_mask,
            relaxation=relaxation,
            normalisation=normalisation,
            min_depth=min_depth,
            split_rule=split_rule,
        )
        for result in results:
            panels.append(
                Panel(result.method, segmentation.overlay(image, result.mask))
            )
            # "sat" marks a mask that filled the whole region it was allowed to
            # search -- plausible by the declared criteria, but nothing was
            # segmented. Visible on the sheet so it cannot be missed there either.
            if result.description.get("saturates_region"):
                verdict_mark = "sat"
            else:
                verdict_mark = "OK" if result.plausible else "no"
            labels.append(f"{patient_id} {result.method} [{verdict_mark}]")
            per_method[result.method] = {
                **result.description,
                "plausible": result.plausible,
                "reasons": result.reasons,
            }

        attribution = segmentation.attribute(results, image, region, split_rule)
        reports.append(
            {
                # Patient-keyed, so this list goes to a CLUSTER-ONLY file and the
                # SHAREABLE metrics.json carries only counts built from it.
                "patient_id": patient_id,
                "separability": round(separability, 4),
                "separability_whole_frame": round(whole_frame, 4),
                "methods": per_method,
                "attribution": attribution,
            }
        )
        ctx.log(
            f"  {patient_id}: separability {separability:.3f} in-region "
            f"({whole_frame:.3f} whole-frame), agreement "
            f"{attribution['instrument_comparator_iou']:.3f}, growth "
            f"{attribution['relaxation_growth']:.2f}x, {attribution['attribution']}, "
            f"plausible {sorted(m for m, v in per_method.items() if v['plausible'])}"
        )

    target = ctx.path("contact_sheet_segmentation.png", tier="CLUSTER-ONLY")
    save_sheet(
        tile(panels, columns=len(segmentation.METHODS) + 1),
        target,
        labels=labels,
        columns=len(segmentation.METHODS) + 1,
        # cell_for, not panels[0]: the caption grid must be the grid tile
        # laid out, or labels land on neighbouring panels (render.cell_for).
        cell=cell_for(panels),
    )
    ctx.log(f"sheet: {target.name}")

    # Patient-keyed, so CLUSTER-ONLY. This is where a failing patient is NAMED --
    # "one patient failed" is not actionable, "patient 143 failed, and the two
    # methods agreed on what they found" is.
    ctx.path("segmentation_per_patient.json", tier="CLUSTER-ONLY").write_text(
        json.dumps(reports, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    summary = segmentation.verdict(reports)
    summary["spatial_prior"] = spatial_prior
    summary["face_mask"] = use_face_mask
    summary["contour_relaxation"] = relaxation
    summary["normalisation"] = normalisation
    summary["fissure_min_depth"] = min_depth
    summary["split_rule"] = split_rule
    summary["mean_separability_whole_frame"] = round(
        sum(r["separability_whole_frame"] for r in reports) / len(reports), 4
    )
    ctx.path("metrics.json", tier="SHAREABLE").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    rendered = json.dumps(summary, indent=2, sort_keys=True)
    print(rendered)
    for line in rendered.splitlines():
        ctx.log(line)


def task_partition_sensitivity(ctx: RunContext) -> None:
    """Phase 4: how much does the estimate depend on how the data is partitioned?

    One arm, several partitionings of the same 237 patients. NOT fold-assignment
    variance -- the frozen generator is deterministic, so that cannot be measured
    here at all. See ``train.partition``.
    """
    import json

    from .train import partition, phase3
    from .train.harness import TrainConfig

    task = ctx.config["task"]
    declared = {entry["name"]: Path(entry["path"]) for entry in ctx.inputs}

    from .data.manifest import load_manifest

    manifest_dir = declared[task["manifest_artifact"]]
    staged_dir = declared[task["staged_artifact"]]

    images, _, patient_ids, _ = phase3.load_inputs(
        manifest_dir, staged_dir, task["geometry"], task["label"]
    )

    # STRATIFICATION IS ON class3, NOT on the label the arm trains on. The
    # generator refuses the raw 1-5 grade -- it has ~237 distinct values and
    # stratifying on those is meaningless -- so passing `labels` here aborts the
    # run at the first fold count.
    rows = load_manifest(manifest_dir / "manifest.csv")
    class_labels = [int(row["class3"]) for row in rows]

    fold_counts = tuple(task.get("fold_counts", partition.DEFAULT_FOLD_COUNTS))
    schemes = partition.partitionings(
        patient_ids, class_labels, fold_counts, task.get("include_lopo", True)
    )
    ctx.log(f"partitionings: {sorted(schemes)}")

    # ONCE, for every partitioning. The features do not depend on how the
    # patients are split, so a frozen backbone should meet each image exactly one
    # time in this whole task. Extraction was already outside the fold loop;
    # this takes it outside the PARTITIONING loop too.
    backbone_config = {
        "learning_rate": task.get("learning_rate", 1e-3),
        "weight_decay": task.get("weight_decay", 0.01),
        "batch_size": task.get("batch_size", 32),
    }
    features_once = phase3.prepare_features(
        images,
        task["backbone"],
        task.get("trainable", "head"),
        backbone_config,
        task["geometry"],
    )
    ctx.log(
        f"features extracted once for {len(schemes)} partitionings: "
        f"{features_once[1].get('features')} {features_once[1].get('shape')}"
    )

    results: dict[str, dict] = {}
    for name, assignments in schemes.items():
        sizes = partition.training_sizes(assignments)
        ctx.log(
            f"--- {name}: {sizes['n_folds']} folds, mean train "
            f"{sizes['mean_train_size']} ---"
        )
        result = phase3.run(
            manifest_dir=manifest_dir,
            staged_dir=staged_dir,
            geometry=task["geometry"],
            label=task["label"],
            backbone=task["backbone"],
            trainable=task.get("trainable", "head"),
            seed=ctx.config["seed"],
            assignments_override=assignments,
            features_override=features_once,
            train_config=TrainConfig(
                max_epochs=task["max_epochs"],
                patience=task["patience"],
                inner_val_frac=task["inner_val_frac"],
                monitor=task["monitor"],
                seed=ctx.config["seed"],
            ),
            backbone_config=backbone_config,
            log=ctx.log,
        )
        phase3.write_outputs(result, ctx, prefix=f"{name}__")
        results[name] = {
            "pcc": result.summary["oof"]["pcc"],
            "spearman": result.summary["oof"]["spearman"],
            "training_sizes": sizes,
        }
        ctx.log(f"  {name}: pcc {results[name]['pcc']:.4f}")

    summary = partition.sensitivity(results)
    summary["seed"] = ctx.config["seed"]
    summary["arm"] = task["backbone"]
    ctx.path("metrics.json", tier="SHAREABLE").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    rendered = json.dumps(summary, indent=2, sort_keys=True)
    print(rendered)
    for line in rendered.splitlines():
        ctx.log(line)


def task_masked_scut(ctx: RunContext) -> None:
    """Phase 5 SS2 / exit criterion 10: BUILD the masked SCUT artifact.

    **[MEASURED 2026-07-31] The artifact had never been written.** All three
    p5_masked_scut runs produced only the sheet and metrics.json -- the parity
    was measured and the sheet approved, but ``data/scut/masked_v1`` never
    existed on any machine, discovered only when the Phase 6 pretraining
    configs declared it as an input. The persistence step was gated on
    ``write_artifact`` and every run left it false.

    Two mechanical consequences, both here:

    * **a full build that persists nothing is refused** -- ``n_faces: 0`` with
      ``write_artifact: false`` would spend the whole cohort's compute on a
      PNG, three-for-three the observed failure mode;
    * **the write path carries the synthesis contract**: streamed to memmaps
      (one face resident, not 1.65 GiB of lists), checkpointed so a Run:AI
      pause resumes instead of restarting, built in ``<version>.inprogress``
      and renamed on completion, with ``MANIFEST.json`` (per-file hashes,
      payload rollup, generating run) written before the rename. metrics.json
      records the whole-directory rollup the pretraining configs must declare.

    SHAREABLE throughout -- SCUT is public, non-clinical data.
    """
    import json

    import numpy as np

    from .geometry import render
    from .geometry.staging import OUTPUT_SIZE as STAGED_SIZE
    from .provenance.hashing import hash_dir
    from .scut import dataset, landmarks, masked, placement
    from .train import checkpoint as ckpt

    task = ctx.config["task"]
    declared = {entry["name"]: Path(entry["path"]) for entry in ctx.inputs}
    root = declared[task["scut_root"]]

    write_artifact = bool(task.get("write_artifact", False))
    limit = int(task.get("n_faces", 0))
    # THE GUARD, before any data is touched: a config that declares an artifact
    # version and then builds the full cohort without persisting it produces
    # exactly the state this task was found in -- parity measured, sheet
    # approved, and the artifact nonexistent.
    if not write_artifact and not limit:
        raise ValueError(
            "this is a FULL build (n_faces: 0) with write_artifact: false -- "
            "every face would be built and nothing kept beyond the sheet. "
            "Phase 5 ran exactly this three times, and the pretraining configs "
            "then pointed at data/scut/masked_v1, which had never existed. Set "
            "write_artifact: true to build the artifact, or set n_faces to a "
            "small number for a sheet-only review pass."
        )

    stems = dataset.image_stems(root)
    if limit:
        stems = stems[:limit]
    ctx.log(f"{len(stems)} faces after excluding {dataset.EXCLUDED}")

    ar_mode = task.get("ar_sampling", masked.DEFAULT_AR_SAMPLING)
    ratios = masked.sample_aspect_ratios(len(stems), ctx.config["seed"], ar_mode)
    geometries = list(task.get("geometries", ["g1", "g2"]))
    checkpoint_every = int(task.get("checkpoint_every", 250))
    # Road B's resolution axis: the same frozen composition at a new size
    # argument. The default keeps every existing config byte-identical.
    size = int(task.get("size", STAGED_SIZE))

    n_faces = len(stems)
    version = task["out_version"]
    artifact = ctx.repo_root / "data" / "scut" / version
    working = artifact.with_name(artifact.name + ".inprogress")
    faces_path = working / "faces.jsonl"

    per_face: list[dict] = []
    arrays: dict[str, np.ndarray] = {}
    start_index = 0

    if write_artifact:
        if artifact.exists():
            raise ValueError(
                f"{artifact} already exists. Data artifacts are immutable "
                "(PLAN §2.6): create a new version, never overwrite."
            )
        working.mkdir(parents=True, exist_ok=True)

        found = ckpt.latest(working)
        if found is not None:
            state = ckpt.load(found)
            if (
                state.extra.get("n_faces") != n_faces
                or state.extra.get("geometries") != geometries
                # Old checkpoints predate the size field and were 224 builds.
                or state.extra.get("size", STAGED_SIZE) != size
            ):
                raise ValueError(
                    f"the checkpoint in {working} describes a different build "
                    f"({state.extra.get('n_faces')} faces, "
                    f"{state.extra.get('geometries')}); resuming it into this "
                    "one would interleave two artifacts. Delete the directory "
                    "and rebuild."
                )
            start_index = state.step
            # The journal length is DERIVED from the resume point, never read
            # from a second number that can disagree with it -- the synthesis
            # build published an index with 211 rows for a 200-face build by
            # trusting the recorded count. See task_asymmetry_synthesis.
            lines = (
                faces_path.read_text(encoding="utf-8").splitlines()
                if faces_path.exists()
                else []
            )
            if len(lines) < start_index:
                raise ValueError(
                    f"faces.jsonl has {len(lines)} rows but the checkpoint "
                    f"resumes at face {start_index}. The journal is short of "
                    f"its checkpoint; delete {working} and rebuild rather than "
                    "resuming into a gap."
                )
            faces_path.write_bytes(
                "".join(line + "\n" for line in lines[:start_index]).encode("utf-8")
            )
            ctx.log(f"RESUMING at face {start_index} of {n_faces}")

        for geometry in geometries:
            name = working / f"masked_{geometry}.npy"
            arrays[geometry] = np.lib.format.open_memmap(
                name,
                mode="r+" if name.exists() else "w+",
                dtype=np.uint8,
                shape=(n_faces, size, size, 3),
            )
        faces_path.touch()
        per_face = [
            json.loads(line)
            for line in faces_path.read_text(encoding="utf-8").splitlines()
        ]

    n_sheet = min(int(task.get("n_sheet", 6)), n_faces)
    #: Sheet variants for the no-artifact mode, where there is no memmap to
    #: read them back from.
    sheet_stash: dict[tuple[int, str], np.ndarray] = {}

    for index in range(start_index, n_faces):
        stem, ratio = stems[index], ratios[index]
        image = render.load_image(root / dataset.IMAGE_DIR / f"{stem}.jpg")
        points = landmarks.load_pts(root / dataset.LANDMARK_DIR / f"{stem}.pts")
        entry = {"stem": stem, "aspect_ratio": round(float(ratio), 6)}
        for geometry in geometries:
            variant, record = masked.build_one(
                image, points, float(ratio), geometry, size=size
            )
            entry[geometry] = record
            if write_artifact:
                arrays[geometry][index] = variant
            elif index < n_sheet:
                sheet_stash[(index, geometry)] = variant
        per_face.append(entry)

        if write_artifact:
            with faces_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry, sort_keys=True) + "\n")
            if (index + 1) % checkpoint_every == 0 or index + 1 == n_faces:
                # Flush the arrays BEFORE recording how far we got, or a
                # checkpoint could claim faces whose pixels never reached disk.
                for array in arrays.values():
                    array.flush()
                ckpt.save(
                    working / ckpt.CHECKPOINT_NAME,
                    ckpt.Checkpoint(
                        step=index + 1,
                        epoch=0,
                        arrays={},
                        # This build consumes NO streaming randomness: the
                        # ratios are drawn for every face up front from
                        # config.seed, so face N depends only on (stem, ratio).
                        # Recorded anyway -- the contract is uniform, and
                        # recording a constant is cheaper than discovering
                        # later that it was not one.
                        numpy_rng=ckpt.capture_numpy_rng(
                            np.random.default_rng(ctx.config["seed"])
                        ),
                        output_rows={"faces": len(per_face)},
                        extra={
                            "n_faces": n_faces,
                            "geometries": geometries,
                            "ar_sampling": ar_mode,
                            "size": size,
                        },
                    ),
                )
                ctx.log(f"checkpoint at face {index + 1}/{n_faces}")

    # ---- the sheet: placement beside masked output, one review pass ----
    #
    # Sources are reloaded for the first n_sheet faces because a resumed build
    # never produced the early ones in this process; variants come back off the
    # memmap (as COPIES -- an open view holds the mapping and breaks the
    # publish rename on Windows) or from the sheet-only stash.
    panels: list = []
    labels: list[str] = []
    for index in range(n_sheet):
        stem = stems[index]
        image = render.load_image(root / dataset.IMAGE_DIR / f"{stem}.jpg")
        points = landmarks.load_pts(root / dataset.LANDMARK_DIR / f"{stem}.pts")
        panels.append(
            render.Panel(
                f"{stem} placement",
                # At the build size, blocky-sharp via the frozen nearest
                # resize -- fills its cell instead of floating in grey, and
                # the masked panels' smoothing beside it IS the
                # interpolation (masked.placement_panel).
                masked.placement_panel(image, points, float(ratios[index]), size),
            )
        )
        labels.append(f"{stem} placement ar={ratios[index]:.3f}")
        for geometry in geometries:
            if write_artifact:
                variant = np.array(arrays[geometry][index], copy=True)
            else:
                variant = sheet_stash[(index, geometry)]
            panels.append(render.Panel(f"{stem} {geometry}", variant))
            labels.append(f"{stem} masked {geometry}")
        # The cleft trapezium at the median cleft AR, drawn from the recorded
        # geometry, so "does this look like a cleft crop" is a comparison rather
        # than a memory test. At the build's own size, so the panels compare
        # at one scale.
        panels.append(render.Panel("cleft shape", masked.cleft_reference_panel(size)))
        labels.append("expected cleft trapezium (median AR)")

    columns = len(geometries) + 2
    # NOT "contact_sheet_*": the tier guard treats that token as patient-keyed
    # and refuses to mark it SHAREABLE. That heuristic is right for every cleft
    # artifact and wrong here -- this sheet contains only public SCUT faces and
    # no clinical data at all. The name avoids the token DELIBERATELY, and the
    # guard stays untouched: it is frozen, and weakening a clinical-data
    # safeguard to accommodate a public dataset would be the wrong trade.
    target = ctx.path("masked_scut_sheet.png", tier="SHAREABLE")
    render.save_sheet(
        render.tile(panels, columns=columns),
        target,
        labels=labels,
        columns=columns,
        # [CORRECTED 2026-08-09] This passed panels[0]'s dimensions -- the
        # source-scale placement panel -- while tile laid out at the max
        # panel size, so at 768 every caption drifted onto the next face's
        # cell. The 224 sheet aligned only because the 350px placement panel
        # happened to BE the max there. One source of truth now
        # (render.cell_for), and the panels are uniform at the build size
        # anyway (masked.placement_panel).
        cell=render.cell_for(panels),
    )
    ctx.log(f"sheet: {target.name} ({n_sheet} faces)")

    # Every SCUT source is 350x350, so a build above that interpolates EVERY
    # face uniformly. The note travels in BOTH the run's metrics and the
    # artifact's MANIFEST -- a consumer opening the artifact must be able to
    # tell, without this run's records, that the pixels carry no texture
    # information beyond 350px (the write-up guard:
    # roadb.PHASE_6_STRUCTURE -- higher-resolution pretraining adapts
    # positional geometry, not finer detail).
    interpolation_note = (
        f"every SCUT source is 350x350; at {size} ALL faces are upsampled "
        f"{size / 350:.2f}x and carry no texture information beyond 350px. "
        "On the sheet the masked panels render NATIVE; their smoothing "
        "against the placement panel -- nearest-upscaled, so it stays "
        "blocky-sharp -- is that interpolation, visible"
    ) if size > 350 else None

    summary = {
        "n_faces": n_faces,
        "geometries": geometries,
        "ar_sampling": ar_mode,
        "size": size,
        "seed": ctx.config["seed"],
        "excluded": list(dataset.EXCLUDED),
        "cm152_disposition": dict(dataset.CM152_DISPOSITION),
        "levelling": placement.SCUT_HEAD_TILT_LEVELLING,
        "parity": {
            g: masked.parity_report([face[g] for face in per_face])
            for g in geometries
        },
    }
    if interpolation_note:
        summary["interpolation_note"] = interpolation_note

    # ---- publish ----
    if write_artifact:
        # Close the mappings before the rename: an open memmap holds the file
        # handle, and on Windows that fails the directory rename at the very
        # end, after the entire build.
        import gc

        for array in arrays.values():
            array.flush()
            mapping = getattr(array, "_mmap", None)
            if mapping is not None:
                mapping.close()
        arrays.clear()
        gc.collect()

        # The index must describe the arrays it sits beside: row i of every
        # array is stems[i], so the journal must list exactly the stems in
        # order. Checked before the rename so a corrupted index is never
        # published.
        journal_stems = [face["stem"] for face in per_face]
        if journal_stems != stems:
            raise ValueError(
                f"faces.jsonl lists {len(journal_stems)} stems and disagrees "
                "with the build order; the index does not describe the arrays "
                "beside it, so this artifact must not be published."
            )

        atomic_write_text(
            working / "faces.json",
            json.dumps(per_face, indent=2, sort_keys=True) + "\n",
        )
        # The JSONL form is the RESUME journal, not the artifact.
        faces_path.unlink(missing_ok=True)
        (working / ckpt.CHECKPOINT_NAME).unlink(missing_ok=True)

        # MANIFEST.json, the smoke/v1 convention: payload hashes computed
        # BEFORE the manifest is added, so the manifest describes the payload
        # without describing itself. The whole-directory rollup -- what a
        # consuming config declares, since hash_dir covers MANIFEST.json too --
        # is computed after, and recorded in metrics.json as the value to
        # paste into the p6 pretraining configs.
        payload = hash_dir(working)
        atomic_write_text(
            working / "MANIFEST.json",
            json.dumps(
                {
                    "artifact": f"scut/{version}",
                    "generated_by": "cleft.run task masked_scut",
                    "generating_run": ctx.run_dir.name,
                    "generator_git_sha8": ctx.git.sha8,
                    "seed": ctx.config["seed"],
                    "n_faces": n_faces,
                    "geometries": geometries,
                    "ar_sampling": ar_mode,
                    # The staging size and, above the 350px source size, the
                    # write-up guard -- IN the artifact, so a consumer
                    # opening it can tell what the pixels are without this
                    # run's metrics. The arrays are self-describing (their
                    # shape is the size); this is the human-facing record.
                    "size": size,
                    **(
                        {"interpolation_note": interpolation_note}
                        if interpolation_note else {}
                    ),
                    "numpy_version": np.__version__,
                    "payload_files": payload["files"],
                    "payload_rollup": payload["rollup"],
                    "payload_total_bytes": payload["total_bytes"],
                    "purpose": (
                        "masked SCUT variants for Phase 6 pretraining; row i "
                        "of each masked_<geometry>.npy is faces.json[i]"
                    ),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
        )
        declare = hash_dir(working)

        # The rename is what publishes: until it happens the final path does
        # not exist, so an interrupted build cannot be mistaken for a complete
        # one, and the immutability guard stays exactly as strict.
        if artifact.exists():
            raise ValueError(
                f"{artifact} appeared while this build was running. Data "
                "artifacts are immutable (PLAN §2.6): create a new version."
            )
        working.rename(artifact)
        ctx.log(f"artifact: {artifact}")

        summary["artifact"] = {
            "path": f"data/scut/{version}",
            "checkpoint_every": checkpoint_every,
            "resumed_at_face": start_index,
            "payload_rollup": payload["rollup"],
            "rollup_sha256_for_configs": declare["rollup"],
            "file_count": declare["file_count"],
            "total_bytes": declare["total_bytes"],
            "note": (
                "rollup_sha256_for_configs covers MANIFEST.json too, matching "
                "what guard 3 verifies -- it is the value the p6_pretrain "
                "masked configs declare for the masked_scut input."
            ),
        }

    ctx.path("metrics.json", tier="SHAREABLE").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    rendered = json.dumps(summary, indent=2, sort_keys=True)
    print(rendered)
    for line in rendered.splitlines():
        ctx.log(line)


def task_asymmetry_synthesis(ctx: RunContext) -> None:
    """Phase 5 §5: unilateral cleft-like asymmetry by TPS (PLAN §4.11 Q-b).

    Deformation only. **Four limitations are written into ``metrics.json`` before
    the arm runs**, in the SymNose construct-note pattern: no scar; the
    magnitude-to-grade mapping is an assumption and not a measurement; the midline
    is anchored; and the deformation cannot reach the philtrum, the region the
    relevance diagnostic ranks first. ``ARM_PREDICTION`` says in advance that the
    arm is not expected to succeed -- stated afterwards, that would be
    indistinguishable from explaining away a poor result.

    **Streamed and resumable.** Arrays go into memmaps as they are produced and a
    checkpoint records progress, so a Run:AI pause continues rather than silently
    restarting at face 0. The artifact is built in ``<version>.inprogress`` and
    renamed on completion, so the final path never exists half-built.

    SHAREABLE throughout: SCUT is public, non-clinical data.
    """
    import json

    import numpy as np

    from .geometry import render
    from .geometry.staging import OUTPUT_SIZE as OUTPUT_SIZE_SYNTH
    from .scut import dataset, landmarks, masked, placement, synthesis

    task = ctx.config["task"]
    declared = {entry["name"]: Path(entry["path"]) for entry in ctx.inputs}
    root = declared[task["scut_root"]]

    # Same guard as task_masked_scut, same failure mode: a full build whose
    # only outputs are a sheet and metrics. Worse here -- the full synthesis
    # build is the known pathologically slow one (~1,400 CPU-s/face on the
    # cluster), so the silent mode would burn hours and persist nothing.
    if not task.get("write_artifact", False) and not int(task.get("n_faces", 0)):
        raise ValueError(
            "this is a FULL build (n_faces: 0) with write_artifact: false -- "
            "every face would be built and nothing kept beyond the sheet. "
            "Phase 5's masked_scut ran exactly this three times and its "
            "artifact never existed. Set write_artifact: true to build the "
            "artifact, or set n_faces to a small number for a sheet-only "
            "review pass."
        )

    magnitudes = [float(m) for m in task.get("magnitudes", (0.0, 0.015, 0.025, 0.035))]
    if 0.0 not in magnitudes:
        raise ValueError(
            "magnitudes must include 0.0: the undeformed set is this arm's own "
            "reference, and a deformed-only set has nothing to be compared with."
        )

    stems = dataset.image_stems(root)
    limit = int(task.get("n_faces", 0))
    if limit:
        stems = stems[:limit]

    geometries_declared = list(task.get("geometries", ["g1", "g2"]))
    # **The cross product is DISK now, not memory.** Staged arrays are written
    # straight into per-(geometry, magnitude) memmaps as each face is produced, so
    # resident memory is one face's worth whatever the build size. An earlier
    # revision accumulated every array in RAM -- 6.6 GiB for the full set -- and
    # guarded it with a 3 GiB budget, which would have forced the cross product to
    # be cut to fit a laptop's memory. **A memory limit must not decide the
    # scientific design**, so the limit was removed rather than the magnitudes.
    #
    # The budget is kept, pointed at disk, and set above the real build: it exists
    # to catch a runaway cross product while the run is still cheap to re-scope,
    # not to constrain the intended one.
    per_array = 224 * 224 * 3
    estimated = len(stems) * len(magnitudes) * len(geometries_declared) * per_array
    budget = int(task.get("disk_budget_bytes", 16 * 1024**3))
    if estimated > budget:
        raise ValueError(
            f"this run would write {estimated / 1024**3:.1f} GiB of staged arrays "
            f"({len(stems)} faces x {len(magnitudes)} magnitudes x "
            f"{len(geometries_declared)} geometries x {per_array} bytes), over "
            f"the {budget / 1024**3:.1f} GiB budget. Reduce n_faces, magnitudes "
            "or geometries, or raise disk_budget_bytes deliberately. Refused here "
            "rather than after the images are read: a run that dies part-way has "
            "consumed the time and produced nothing."
        )

    seed = ctx.config["seed"]
    # [2026-08-29, Phase 17] The warp FAMILY is a config key: "tps" is
    # the parked module's spline (the default, so every shipped p5
    # config is byte-unchanged), "piecewise_affine" is arm B's family
    # (scut.piecewise, same control points, same signature). One key,
    # two families, everything downstream shared -- which is what makes
    # B-vs-C isolate the transformation family.
    warp_family = task.get("warp_family", "tps")
    if warp_family == "piecewise_affine":
        from .scut import piecewise

        warp_content = piecewise.warp_content
    else:
        warp_content = synthesis.warp_content
    ar_mode = task.get("ar_sampling", masked.DEFAULT_AR_SAMPLING)
    ratios = masked.sample_aspect_ratios(len(stems), seed, ar_mode)
    sides = synthesis.assign_sides(len(stems), seed)
    geometries = geometries_declared
    ctx.log(
        f"{len(stems)} faces, magnitudes {magnitudes}, sides "
        f"{synthesis.side_balance(sides)['n_left']}L/"
        f"{synthesis.side_balance(sides)['n_right']}R"
    )

    # ---- where the arrays go, and how a pause survives -------------------
    #
    # **The build is streamed and resumable, and the two are the same mechanism.**
    # Arrays go straight into memmaps as each face is produced; a checkpoint every
    # `checkpoint_every` faces records how far the build got. A Run:AI pause
    # recreates the pod and reattaches the run directory (PLAN §2.7), but WITHOUT
    # this the work restarts at face 0 with no symptom but wall time. See
    # docs/VERIFY_checkpoint.md.
    #
    # Written into `<version>.inprogress` and renamed on completion, so the final
    # artifact path never exists half-built. That is what lets a resume reuse the
    # partial work without weakening the immutability guard: the guard is on the
    # final name, and the final name appears only when the build is whole.
    from .train import checkpoint as ckpt

    write_artifact = bool(task.get("write_artifact", False))
    checkpoint_every = int(task.get("checkpoint_every", 250))
    version = task["out_version"]
    artifact = ctx.repo_root / "data" / "scut" / version
    working = artifact.with_name(artifact.name + ".inprogress")

    if write_artifact and artifact.exists():
        raise ValueError(
            f"{artifact} already exists. Data artifacts are immutable "
            "(PLAN §2.6): create a new version, never overwrite."
        )

    n_faces = len(stems)
    arrays: dict[tuple, np.ndarray] = {}
    start_index = 0
    per_face: list[dict] = []
    survival: list[dict] = []
    faces_path = working / "faces.jsonl"
    survival_path = working / "survival.jsonl"

    if write_artifact:
        working.mkdir(parents=True, exist_ok=True)
        found = ckpt.latest(working)
        if found is not None:
            state = ckpt.load(found)
            start_index = state.step

            # **The journal length is DERIVED from the resume point, not read
            # from a second number that can disagree with it.**
            #
            # [DEFECT 2026-07-30, found by the cluster gate] The first version
            # truncated to `output_rows[...]`, a count recorded beside `step`.
            # The two are meant to be equal and the arrays were rolled back
            # correctly, but `faces.json` came out with 211 rows for a 200-face
            # build and 11 duplicate stems -- so the two disagreed by exactly the
            # faces between the last checkpoint and the pause. The pixels were
            # right and the index beside them was wrong: an artifact
            # misreporting its own contents.
            #
            # Recording the same fact twice and trusting both is the error. There
            # is only one truth here -- face N's row is row N -- so the counts are
            # computed from `step` and the recorded ones are used only to REPORT
            # a disagreement, never to act on one.
            expected = {
                faces_path: start_index,
                survival_path: start_index * sum(1 for m in magnitudes if m > 0),
            }
            for path, keep in expected.items():
                lines = (
                    path.read_text(encoding="utf-8").splitlines()
                    if path.exists()
                    else []
                )
                if len(lines) < keep:
                    raise ValueError(
                        f"{path.name} has {len(lines)} rows but the checkpoint "
                        f"resumes at face {start_index}, which needs {keep}. The "
                        "journal is short of its checkpoint, so rows the build "
                        "believes are durable are missing. Delete "
                        f"{working} and rebuild rather than resuming into a gap."
                    )
                path.write_text(
                    "".join(line + "\n" for line in lines[:keep]), encoding="utf-8"
                )

            recorded = state.output_rows.get("faces")
            drift = "" if recorded == start_index else (
                f" [recorded output_rows.faces={recorded} disagreed with step="
                f"{start_index}; the derived count wins]"
            )
            ctx.log(
                f"RESUMING at face {start_index} of {n_faces}; journals rolled "
                f"back to {expected[faces_path]} face and "
                f"{expected[survival_path]} survival rows{drift}"
            )

        for geometry in geometries:
            for magnitude in magnitudes:
                name = working / f"synth_{geometry}_m{magnitude:.3f}.npy"
                arrays[(geometry, magnitude)] = np.lib.format.open_memmap(
                    name,
                    mode="r+" if name.exists() else "w+",
                    dtype=np.uint8,
                    shape=(n_faces, OUTPUT_SIZE_SYNTH, OUTPUT_SIZE_SYNTH, 3),
                )
        for path in (faces_path, survival_path):
            path.touch()
        per_face = [
            json.loads(line)
            for line in faces_path.read_text(encoding="utf-8").splitlines()
        ]
        survival = [
            json.loads(line)
            for line in survival_path.read_text(encoding="utf-8").splitlines()
        ]

    n_sheet = min(int(task.get("n_sheet", 6)), n_faces)
    sheet_geometry = geometries[0]
    sheet_arrays: dict[tuple, np.ndarray] = {}

    for index in range(start_index, n_faces):
        # [2026-08-29] Per-face wall clock, unconditional -- the
        # compute-gate A/B needs a timing sheet-only mode never logged
        # (phase17.COMPUTE_GATE_MEASUREMENT['timing_observability']).
        # Elapsed rides IN the message because ctx.log's one-second
        # timestamps cannot resolve a fast run. A log line, not a
        # behaviour change.
        face_started = time.perf_counter()
        stem, ratio, side = stems[index], ratios[index], sides[index]
        image = render.load_image(root / dataset.IMAGE_DIR / f"{stem}.jpg")
        points = landmarks.load_pts(root / dataset.LANDMARK_DIR / f"{stem}.pts")
        box = placement.crop_box(points, float(ratio))
        entry = {
            "stem": stem,
            "side": str(side),
            "aspect_ratio": round(float(ratio), 6),
            "midline_check": placement.mirror_pair_asymmetry(points),
            "magnitudes": {},
        }

        for magnitude in magnitudes:
            warped, record = warp_content(
                image, points, box, str(side), magnitude
            )
            for geometry in geometries:
                staged = masked.build_one(
                    warped, points, float(ratio), geometry
                )[0]
                if write_artifact:
                    arrays[(geometry, magnitude)][index] = staged
                if index < n_sheet and geometry == sheet_geometry:
                    sheet_arrays[(index, magnitude)] = staged
            entry["magnitudes"][str(magnitude)] = record

            if magnitude > 0:
                report = synthesis.landmark_asymmetry_survives(
                    points, box, str(side), magnitude
                )
                report["stem"] = stem
                survival.append(report)
                if write_artifact:
                    with survival_path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(report, sort_keys=True) + "\n")

        per_face.append(entry)
        ctx.log(
            f"face {index + 1}/{n_faces} in "
            f"{time.perf_counter() - face_started:.2f}s"
        )
        if write_artifact:
            with faces_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry, sort_keys=True) + "\n")

            if (index + 1) % checkpoint_every == 0 or index + 1 == n_faces:
                # Flush the arrays BEFORE recording how far we got, or a
                # checkpoint could claim faces whose pixels never reached disk.
                for array in arrays.values():
                    array.flush()
                ckpt.save(
                    working / ckpt.CHECKPOINT_NAME,
                    ckpt.Checkpoint(
                        step=index + 1,
                        epoch=0,
                        arrays={},
                        # This build consumes NO streaming randomness: the aspect
                        # ratios and the sides are both drawn for every face up
                        # front from config.seed, so face N's work depends only on
                        # (stem, ratio, side). The state is recorded anyway
                        # because the contract is uniform and a future streamed
                        # version would need it -- and because recording a
                        # constant is cheaper than discovering later that it was
                        # not one.
                        numpy_rng=ckpt.capture_numpy_rng(
                            np.random.default_rng(seed)
                        ),
                        output_rows={
                            "faces": len(per_face),
                            "survival": len(survival),
                        },
                        extra={"n_faces": n_faces, "magnitudes": magnitudes},
                    ),
                )
                ctx.log(f"checkpoint at face {index + 1}/{n_faces}")

    # ---- the sheet: undeformed beside each magnitude, one review pass ----
    #
    # Read back from the artifact when resuming, because a resumed run never
    # produced the early faces in this process. Falls back to what was held in
    # memory when no artifact is being written (the sheet-only mode).
    panels: list = []
    labels: list[str] = []
    for index in range(n_sheet):
        for magnitude in magnitudes:
            if (index, magnitude) in sheet_arrays:
                panel_image = sheet_arrays[(index, magnitude)]
            else:
                # **A COPY, not a view.** np.asarray on a memmap slice returns a
                # view that keeps the mapping open, and on Windows an open
                # mapping makes the publish rename fail with access denied --
                # after the whole build has run.
                panel_image = np.array(
                    arrays[(sheet_geometry, magnitude)][index], copy=True
                )
            panels.append(
                render.Panel(f"{stems[index]} m={magnitude}", panel_image)
            )
            labels.append(
                f"{stems[index]} {sides[index]} m={magnitude} {sheet_geometry}"
            )

    columns = len(magnitudes)
    # Deliberately not "contact_sheet_*": that token makes the frozen tier guard
    # treat the file as patient-keyed. Right for every cleft artifact, wrong for
    # public SCUT -- so the name avoids it and the guard stays untouched.
    target = ctx.path("asymmetry_synthesis_sheet.png", tier="SHAREABLE")
    render.save_sheet(
        render.tile(panels, columns=columns),
        target,
        labels=labels,
        columns=columns,
        # cell_for, not panels[0]: the caption grid must be the grid tile
        # laid out (render.cell_for).
        cell=render.cell_for(panels),
    )
    ctx.log(f"sheet: {target.name} ({n_sheet} faces x {len(magnitudes)} magnitudes)")

    if write_artifact:
        # **Close the mappings before the rename.** An open memmap holds the file
        # handle, and on Windows that makes the directory rename fail with access
        # denied -- at the very end, after the entire build. numpy has no public
        # close for a memmap, so the mapping is released explicitly where that is
        # possible and the references dropped either way.
        import gc

        for array in arrays.values():
            array.flush()
            mapping = getattr(array, "_mmap", None)
            if mapping is not None:
                mapping.close()
        arrays.clear()
        gc.collect()

        # **The index must describe the arrays it sits beside**, checked before
        # the rename so a corrupted index is never published. See
        # synthesis.assert_index_describes_arrays for what this caught.
        index_counts = synthesis.assert_index_describes_arrays(
            per_face, survival, n_faces, magnitudes
        )
        ctx.log(f"index verified: {index_counts}")

        atomic_write_text(
            working / "faces.json",
            json.dumps(per_face, indent=2, sort_keys=True) + "\n",
        )
        # The JSONL forms are the RESUME journal, not the artifact, and leaving
        # them behind would put two records of the same thing in a published
        # directory -- with no rule saying which is authoritative.
        for path in (faces_path, survival_path):
            path.unlink(missing_ok=True)
        (working / ckpt.CHECKPOINT_NAME).unlink(missing_ok=True)

        # **The rename is what publishes the artifact.** Until it happens the
        # final path does not exist, so an interrupted build cannot be mistaken
        # for a complete one -- and the immutability guard on the final name
        # stays exactly as strict as it was.
        if artifact.exists():
            raise ValueError(
                f"{artifact} appeared while this build was running. Data "
                "artifacts are immutable (PLAN §2.6): create a new version."
            )
        working.rename(artifact)
        ctx.log(f"artifact: {artifact}")

    # Worst case over every face and magnitude, because a mean would hide the
    # one face whose deformation drifted. Each of these is asserted by test on
    # fixtures; here it is measured on real faces.
    def worst(key: str) -> tuple[float, int]:
        values = [
            record["deformation"][key]
            for face in per_face
            for record in face["magnitudes"].values()
            if not record["identity"]
        ]
        # **Not rounded, and the count travels with it.** Rounded to 8 places the
        # real answer (~4e-11) printed as a flat 0.0, indistinguishable from a
        # quantity nobody computed -- and an aggregate over an empty list would
        # also have been 0.0. Both are failure mode 5 in PLAN R7's tally. The
        # count is what says the number came from somewhere.
        return (max(values) if values else None), len(values)

    ratios_seen = [
        value
        for report in survival
        for value in [report["min_survival_ratio"]]
        if value is not None
    ]
    summary = {
        # **Uppercase so it sorts FIRST**: metrics.json is dumped with
        # sort_keys=True, so "limitations" lands eighth among the lowercase keys
        # -- an earlier comment here claimed it was first, which it was not. The
        # prediction is what a reader must meet before the numbers, so it gets a
        # key that cannot be scrolled past (PLAN §3.2 pattern).
        "ARM_PREDICTION": {
            "predicted_to_succeed": False,
            "recorded": "before the arm ran",
            "reason": (
                "the deformation cannot touch the philtrum, which the univariate "
                "relevance diagnostic ranks FIRST of 22 at |Spearman| 0.151. The "
                "midline is anchored so the mirror-difference instrument keeps a "
                "fixed axis; see limitations.midline_is_anchored."
            ),
            "why_in_advance": (
                "stated afterwards this would be indistinguishable from "
                "explaining away a poor result"
            ),
            "read_next": "limitations",
        },
        # Properties of the instrument, not disclaimers about a component of it.
        "limitations": dict(synthesis.LIMITATIONS),
        "n_faces": len(stems),
        "magnitudes": magnitudes,
        "geometries": geometries,
        "ar_sampling": ar_mode,
        "seed": seed,
        "side_balance": synthesis.side_balance(sides),
        "directions": {k: list(v) for k, v in synthesis.DIRECTIONS.items()},
        "weights": dict(synthesis.WEIGHTS),
        "deformation_is_local": {
            # **Read pixel_change_y_min first.** The anchor-motion figures below
            # are guaranteed small by the fit -- every landmark except the four
            # targets is an anchor -- so they are a consistency check, not
            # evidence. The pixel extent is the measurement that can fail: the
            # eyes sit near 0.23 of the crop, so change reaching above ~0.35
            # means the deformation is not confined to the nasolabial region.
            "pixel_change_y_min_worst": min(
                (
                    record["deformation"]["pixel_change"]["y_min"]
                    for face in per_face
                    for record in face["magnitudes"].values()
                    if not record["identity"]
                ),
                default=None,
            ),
            "pixel_change_outside_box_count": sum(
                1
                for face in per_face
                for record in face["magnitudes"].values()
                if not record["identity"]
                and record["deformation"]["pixel_change"]["outside_box"]
            ),
            "max_unintended_motion_frac_width": worst(
                "max_unintended_motion_frac_width"
            )[0],
            "n_deformations_measured": worst("max_unintended_motion_frac_width")[1],
            # Beside the count, what it SHOULD be. A resumed run whose journal was
            # appended rather than rolled back reported 633 here against 600 --
            # and with nothing to compare against, 633 read as a fine number.
            "n_deformations_expected": n_faces
            * sum(1 for m in magnitudes if m > 0),
            "note": (
                "anchors, the contralateral landmarks and the midline landmarks "
                "must not move: a drifting frame is a reframed face rather than "
                "an asymmetric one, and a moving contralateral side is not a "
                "unilateral deformation at all. Reported UNROUNDED: at 8 decimal "
                "places the real answer (~4e-11, floating-point residue in the "
                "spline solve) printed as a flat 0.0, which is what a quantity "
                "nobody computed also prints as. n_deformations_measured is what "
                "says the number came from somewhere -- null with a count of 0 "
                "means nothing was measured, not that nothing moved."
            ),
        },
        "asymmetry_survives_g2": {
            "min_over_faces": round(min(ratios_seen), 6) if ratios_seen else None,
            "max_over_faces": round(max(ratios_seen), 6) if ratios_seen else None,
            "n_measured": len(ratios_seen),
            "n_expected": n_faces * sum(1 for m in magnitudes if m > 0),
            "note": (
                "EXACT landmark measurement, no resampling floor. Ratios ABOVE 1 "
                "are correct: G2 magnifies horizontal distance by "
                "1/(2*half_width), 1.66 at the brow to 1.0 at the lip. A ratio "
                "near zero would mean the asymmetry was destroyed and G2 unusable "
                "for this arm."
            ),
        },
        "levelling": placement.SCUT_HEAD_TILT_LEVELLING,
        "excluded": list(dataset.EXCLUDED),
    }
    ctx.path("metrics.json", tier="SHAREABLE").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    rendered = json.dumps(summary, indent=2, sort_keys=True)
    print(rendered)
    for line in rendered.splitlines():
        ctx.log(line)


def task_feature_relevance(ctx: RunContext) -> None:
    """Which regions' asymmetry tracks the grade, one feature at a time.

    A diagnostic over data that already exists: the mirror-difference index's 22
    features for all 237 patients. No fitting, no seeds, nothing tuned on it.
    """
    import json

    from .geometry import mirror
    from .relevance import report
    from .train import phase3

    task = ctx.config["task"]
    declared = {entry["name"]: Path(entry["path"]) for entry in ctx.inputs}

    images, labels, patient_ids, _ = phase3.load_inputs(
        declared[task["manifest_artifact"]],
        declared[task["staged_artifact"]],
        task.get("geometry", "g2"),
        task.get("label", "mean"),
    )
    ctx.log(f"{len(patient_ids)} patients, label {task.get('label', 'mean')!r}")

    features, names = mirror.feature_matrix(
        images, geometry=task.get("geometry", "g2")
    )
    summary = report(features, labels, names)
    summary["label"] = task.get("label", "mean")
    summary["geometry"] = task.get("geometry", "g2")

    for row in summary["ranked"][:8]:
        ctx.log(
            f"  {row['rank_by_abs_spearman']:2d}. {row['feature']:32s} "
            f"spearman {row['spearman']} pearson {row['pearson']}"
        )

    ctx.path("metrics.json", tier="SHAREABLE").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    rendered = json.dumps(summary, indent=2, sort_keys=True)
    print(rendered)
    for line in rendered.splitlines():
        ctx.log(line)


def task_symnose_audit(ctx: RunContext) -> None:
    """Phase 4 §3.2: does the segmentation hold at 237, or only at the gate's 9?

    Diagnostic only. Nothing is tuned; this reports what the shipped
    configuration does across the whole cohort, so a SymNose result of zero can
    be attributed before anything is adjusted.
    """
    import json

    from .geometry import symnose
    from .train import phase3

    task = ctx.config["task"]
    declared = {entry["name"]: Path(entry["path"]) for entry in ctx.inputs}

    geometry = task.get("geometry", symnose.REQUIRED_GEOMETRY)
    normalisation = task.get("normalisation", "none")
    split_rule = task.get("split_rule", "centroid")

    # The label is not a knob here: this audit never uses the labels array, so
    # `load_inputs` is given the manifest's primary column purely to satisfy its
    # column-existence check. Offering a config field for it would have implied
    # the audit was run against a chosen target.
    images, _, patient_ids, _ = phase3.load_inputs(
        declared[task["manifest_artifact"]],
        declared[task["staged_artifact"]],
        geometry,
        "mean",
    )
    ctx.log(f"auditing {len(patient_ids)} patients")

    aggregate, per_patient = symnose.audit(
        images, normalisation=normalisation, split_rule=split_rule
    )
    for record, patient_id in zip(per_patient, patient_ids):
        record["patient_id"] = patient_id

    # Patient-keyed, so the failing patients can be NAMED and pulled onto a
    # sheet. "62% fail" is not actionable; "these 89 fail, 74 of them on
    # area_min alone" is.
    ctx.path("symnose_audit_per_patient.json", tier="CLUSTER-ONLY").write_text(
        json.dumps(per_patient, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    # Every knob that changes the result, recorded with it. Two audits run at
    # different split_rule produce different numbers, and without these the
    # files give no way to tell why -- "a number used but not recorded is a
    # provenance gap" (module docstring).
    aggregate["geometry"] = geometry
    aggregate["normalisation"] = normalisation
    aggregate["split_rule"] = split_rule
    aggregate["construct_note"] = symnose.CONSTRUCT_NOTE
    aggregate["measured_result"] = dict(symnose.MEASURED_RESULT)
    ctx.path("metrics.json", tier="SHAREABLE").write_text(
        json.dumps(aggregate, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    rendered = json.dumps(aggregate, indent=2, sort_keys=True)
    print(rendered)
    for line in rendered.splitlines():
        ctx.log(line)


def task_pretrain(ctx: RunContext) -> None:
    """Phase 6: full fine-tuning on SCUT, one backbone x one source per run.

    The twelve pretraining runs all go through here. The loop, the checkpoint
    wiring and the band check live in ``train.pretrain``; this handler resolves
    the declared inputs, refuses a declaration that does not match the source,
    and writes the run record.

    SHAREABLE throughout: SCUT is public, non-clinical data. The test-side
    scores file avoids the token "predictions" DELIBERATELY -- the frozen tier
    guard treats that token as patient-keyed, which is right for every cleft
    artifact and wrong for public SCUT stems, so the name steps around it and
    the guard stays untouched (the masked_scut sheet took the same route).
    """
    import json

    from .train import pretrain

    task = ctx.config["task"]
    declared = {entry["name"]: Path(entry["path"]) for entry in ctx.inputs}
    source = task["source"]

    root = declared.get(task["scut_root"])
    if root is None:
        raise ValueError(
            f"task.scut_root names input {task['scut_root']!r}, which is not "
            f"declared in inputs: {sorted(declared)}"
        )

    # The masked artifact is referenced BY CONVENTIONAL NAME, not by a task
    # field: a field would give the original configs a knob that changes
    # nothing (schema.py's symnose note says why that is worse than no knob),
    # and the lattice a second field differing between neighbours.
    masked_dir = declared.get(pretrain.MASKED_INPUT_NAME)
    if source == "original" and masked_dir is not None:
        raise ValueError(
            f"source 'original' does not read {pretrain.MASKED_INPUT_NAME!r}, "
            "but the config declares it. An input the run never reads would "
            "still be hash-verified and recorded in inputs.json -- a "
            "provenance record claiming data fed a run it did not feed."
        )
    if source != "original" and masked_dir is None:
        raise ValueError(
            f"source {source!r} reads the masked SCUT artifact; declare an "
            f"input named {pretrain.MASKED_INPUT_NAME!r} pointing at "
            "data/scut/<version>."
        )

    # [2026-08-10] The pretrained init, by the same conventional-name route.
    # Until this existed the init was the one input guard 3 never verified --
    # re-downloaded every run, unrecorded, and when two ViT-512 runs diverged
    # no field could say whether their inits differed
    # (roadb.PRETRAINED_INIT_IS_NOW_DECLARABLE).
    init_dir = declared.get(pretrain.INIT_INPUT_NAME)
    if init_dir is not None and task["backbone"] in ("srgnn", "agnet"):
        raise ValueError(
            f"the graph backbones do not consume {pretrain.INIT_INPUT_NAME!r} "
            "yet -- their pretrained bodies are built inside their own "
            "assemblies, and the loading route is a recorded follow-up. An "
            "input the run never reads would still be hash-verified and "
            "recorded in inputs.json: a provenance record claiming weights "
            "fed a run they did not feed."
        )

    config = pretrain.PretrainConfig(
        epochs=task["epochs"],
        inner_val_frac=task["inner_val_frac"],
        seed=ctx.config["seed"],
        deterministic=task["deterministic"],
        monitor=task["monitor"],
        batch_size=task["batch_size"],
        learning_rate=task["learning_rate"],
        weight_decay=task["weight_decay"],
        checkpoint_every=task["checkpoint_every"],
    )
    ctx.log(
        f"pretrain: backbone {task['backbone']!r}, source {source!r}, "
        f"scheme {task['region_scheme']!r}, seed {config.seed}, "
        f"checkpoint every {config.checkpoint_every} epoch(s)"
    )

    result = pretrain.run_pretraining(
        scut_root=root,
        source=source,
        backbone=task["backbone"],
        region_scheme=task["region_scheme"],
        masked_dir=masked_dir,
        expect_train=task["expect_train"],
        expect_test=task["expect_test"],
        run_dir=ctx.run_dir,
        curves_path=ctx.path("curves.csv", tier="SHAREABLE"),
        pretrained_path=ctx.path(pretrain.PRETRAINED_NAME, tier="SHAREABLE"),
        config=config,
        init_dir=init_dir,
        log=ctx.log,
    )

    summary = result.summary
    # WHAT the run started from, in every record -- declared and verified,
    # or a runtime download whose bytes nothing captured. The absence used
    # to be silent; now it names itself.
    summary["pretrained_init"] = (
        {
            "declared": True,
            "input": pretrain.INIT_INPUT_NAME,
            "artifact": str(init_dir),
            "verified_by": "guard 3; rollup in inputs.json",
        }
        if init_dir is not None else
        {
            "declared": False,
            "note": (
                "init downloaded at run time -- an undeclared input; "
                "roadb.PRETRAINED_INIT_IS_NOW_DECLARABLE"
            ),
        }
    )
    from .provenance.hashing import hash_file

    summary["pretrained_checkpoint"] = {
        "file": pretrain.PRETRAINED_NAME,
        "sha256": hash_file(ctx.run_dir / pretrain.PRETRAINED_NAME),
        "format": "train.checkpoint v1 (load with checkpoint.load)",
    }

    with ctx.atomic("metrics.json", tier="SHAREABLE") as tmp:
        tmp.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n",
                       encoding="utf-8")

    # Public SCUT stems, so SHAREABLE -- and named to avoid the patient-keyed
    # token, see the docstring.
    with ctx.atomic("scut_test_scores.csv", tier="SHAREABLE") as tmp:
        with tmp.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["stem", "truth", "pred"])
            for stem, truth, prediction in result.test_table:
                writer.writerow([stem, f"{truth:.6f}", f"{prediction:.6f}"])

    ctx.log(
        f"test pcc {summary['test']['pcc']:.4f} "
        f"(band: {summary['band_check'].get('checkpoint_consumable')})"
    )
    rendered = json.dumps(summary, indent=2, sort_keys=True)
    print(rendered)
    for line in rendered.splitlines():
        ctx.log(line)


def task_augmentation_contact_sheet(ctx: RunContext) -> None:
    """Phase 7C exit criterion 5. CLUSTER-ONLY PNG plus a SHAREABLE report.

    **Arm 0 does not run until this is reviewed.** Every other check in this
    phase is mechanical and passes on a protect-list that covers the wrong
    pixels; this is the one that cannot.
    """
    import json

    from . import phase7c
    from .data.manifest import load_manifest
    from .geometry import contact
    from .geometry.render import save_sheet
    from .train import augment, augment_sheet, phase3

    task = ctx.config["task"]
    declared = {entry["name"]: Path(entry["path"]) for entry in ctx.inputs}
    manifest_dir = declared[task["manifest_artifact"]]
    staged_dir = declared[task["staged_artifact"]]
    geometry = task["geometry"]

    images, _, patient_ids, _ = phase3.load_inputs(
        manifest_dir, staged_dir, geometry, "mean"
    )
    geometry_rows = phase3.load_geometry_rows(staged_dir)

    # The same spread across reporting classes every Phase 2 sheet used, and
    # deterministic for the same reason: a sheet showing a different set each
    # run makes "the regions look right" unrepeatable.
    rows = load_manifest(manifest_dir / "manifest.csv")
    selected = contact.select_patients(rows, task["n_patients"])
    index_of = {int(pid): i for i, pid in enumerate(patient_ids)}
    missing = [r["patient_id"] for r in selected if int(r["patient_id"]) not in index_of]
    if missing:
        raise ValueError(f"selected patients absent from the staged tensor: {missing}")
    rows_to_render = [index_of[int(r["patient_id"])] for r in selected]

    # The FULL policy -- arm 5. The review is of the mask and of what
    # full-strength augmentation does to it.
    full = next(
        arm for arm in phase7c.arms() if arm["name"] == "p7c_5_region_full"
    )
    policy = augment.Policy.from_arm(full)
    ctx.log(
        f"rendering {len(rows_to_render)} patients at {geometry} under "
        f"{full['name']}"
    )

    sheet, report = augment_sheet.build_sheet(
        images, geometry_rows, patient_ids,
        rows_to_render=rows_to_render,
        geometry=geometry,
        policy=policy,
        protected_regions=phase7c.PROTECTED_REGIONS,
        photometric_settings=phase7c.PHOTOMETRIC,
        geometric_settings=phase7c.GEOMETRIC,
        inside_strength=phase7c.REGION_AWARE["inside_strength"],
        sigma_fraction=phase7c.BLEND["sigma_fraction_of_crop_width"],
        seed=int(ctx.config["seed"]),
    )
    save_sheet(
        sheet,
        ctx.path(f"augmentation_{geometry}.png", tier="CLUSTER-ONLY"),
        columns=3,
    )

    summary = augment_sheet.summarise(report)
    for entry in report:
        ctx.log(
            f"  patient {entry['patient_id']}: protected "
            f"{entry['protected_fraction_of_content']:.3f} of content, moved "
            f"inside {entry['moved_inside']:.2f} against outside "
            f"{entry['moved_outside']:.2f}  "
            f"(whole frame: {entry['moved_inside_whole_frame']:.2f} against "
            f"{entry['moved_outside_whole_frame']:.2f}, "
            f"content {entry['content_fraction_of_frame']:.2f} of frame)"
        )
    if summary["n_inverted"]:
        ctx.log(
            f"  WARNING: protection inverted on {summary['n_inverted']} "
            f"patient(s): {summary['patients_where_protection_inverted']}"
        )
    if summary["n_inverted_whole_frame"] > summary["n_inverted"]:
        ctx.log(
            f"  (whole-frame comparison would flag "
            f"{summary['n_inverted_whole_frame']}; the difference is white-pad "
            "dilution, not policy -- augment_sheet."
            "MOVED_OUTSIDE_WAS_DILUTED_BY_THE_PAD)"
        )

    with ctx.atomic("metrics.json", tier="SHAREABLE") as tmp:
        tmp.write_text(
            json.dumps(
                {
                    "summary": summary,
                    "per_patient": report,
                    "policy": full["name"],
                    "protected_regions": list(phase7c.PROTECTED_REGIONS),
                    "inside_strength": phase7c.REGION_AWARE["inside_strength"],
                    "sigma_fraction": phase7c.BLEND["sigma_fraction_of_crop_width"],
                    "geometry": geometry,
                    "gate": (
                        "arm 0 does not run until this sheet is reviewed. "
                        "Every other check in this phase passes on a "
                        "protect-list covering the wrong pixels"
                    ),
                },
                indent=2, sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )


def task_phase7b_search(ctx: RunContext) -> None:
    """Phase 7B: the pre-registered search, in ONE job.

    **The search is the artifact** (brief §4: report the search, not the
    winner), so it lives in one run directory with one config, one env.json
    and one inputs.json. Twenty-four directories would make "which twenty-four
    constituted the search" a join the reader performs, and that is exactly the
    join a twenty-fifth could slip through. The budget is also enforceable
    in-process here: ``phase7b.configurations()`` raises on a mismatch, so a
    single walk cannot exceed it.

    **Checkpointed per trial**, because PLAN §2.7's measured hazard is that run
    identity survives a pause and TRAINING DOES NOT -- the pod reattaches the
    directory and restarts at zero with no symptom but wall time.

    **The out-of-fold number is computed ONCE, for the selected configuration,
    after selection.** Never for the twenty-three others: an OOF score that is
    never computed cannot be ranked on, which is a stronger guarantee than one
    computed and refused.
    """
    import json

    from . import phase7b
    from .data.manifest import load_manifest
    from .eval.metrics import pcc
    from .train import phase3

    task = ctx.config["task"]
    declared = {entry["name"]: entry for entry in ctx.inputs}
    manifest_dir = Path(declared[task["manifest_artifact"]]["path"])
    staged_dir = Path(declared[task["staged_artifact"]]["path"])

    # **Refuse before spending anything if the budget is not satisfiable.** A
    # search that dies at trial 13 has burned cluster time and left a record
    # whose exhaustion window cannot be evaluated; 23 trials is not a smaller
    # search, it is a different one.
    sets = {name: entry for name, entry in declared.items() if name.startswith("set_")}
    available = {name[len("set_"):]: Path(entry["path"]) for name, entry in sets.items()}
    phase7b.verify_available(set(available))

    images, labels, patient_ids, _ = phase3.load_inputs(
        manifest_dir, staged_dir, task["geometry"], task["label"]
    )
    rows = load_manifest(manifest_dir / "manifest.csv")
    assignments = {int(row["patient_id"]): int(row["fold"]) for row in rows}

    features_by_set: dict[str, np.ndarray] = {}
    for name, path in sorted(available.items()):
        values, metadata = embeddings_module_load(path, patient_ids)
        features_by_set[name] = values
        ctx.log(
            f"  set {name}: shape {list(values.shape)}"
            + (
                f", block {metadata['block']} token {metadata['token']}"
                if metadata.get("block") is not None
                else ""
            )
        )

    seed = int(ctx.config["seed"])
    inner_val_frac = float(task["inner_val_frac"])

    def make_head(trial, *, seed, fold):
        return phase7b.head_for(trial, seed=seed, fold=fold)

    trials_path = ctx.run_dir / "trials.json"
    completed = []
    if trials_path.is_file():
        completed = json.loads(trials_path.read_text(encoding="utf-8"))
        ctx.log(f"RESUMING: {len(completed)} trials already recorded")

    def checkpoint(records):
        trials_path.write_text(
            json.dumps(records, indent=2) + "\n", encoding="utf-8"
        )

    def evaluate(trial):
        return phase7b.evaluate_configuration(
            trial,
            features_by_set=features_by_set,
            labels=labels,
            patient_ids=patient_ids,
            assignments=assignments,
            make_head=make_head,
            inner_val_frac=inner_val_frac,
            seed=seed,
        )

    ctx.log(f"SEARCH: {phase7b.SEARCH_BUDGET} configurations, inner-val only")
    result = phase7b.run_search(
        evaluate, completed=completed, checkpoint=checkpoint, log=ctx.log
    )
    selected = result["selected"]
    ctx.log(
        f"SELECTED trial {selected['index']} [{selected['axis']}] "
        f"inner_val_pcc {selected['inner_val_pcc']:.4f}"
    )
    ctx.log(f"EXHAUSTION: {result['exhaustion']}")

    # ---- only now does an out-of-fold number exist, and only for one arm ---
    ctx.log(
        f"WINNER at {len(task['seeds'])} seeds: its OWN band, and the FIRST "
        "out-of-fold number computed in this run"
    )
    winner = phase7b.evaluate_selected_oof(
        selected,
        features_by_set=features_by_set,
        labels=labels,
        patient_ids=patient_ids,
        assignments=assignments,
        make_head=make_head,
        inner_val_frac=inner_val_frac,
        seeds=[int(s) for s in task["seeds"]],
    )
    band = phase3.seed_variance(winner["pooled_pccs"])
    ctx.log(
        f"  winner OOF PCC {band['mean']:.4f} sd {band['sd']:.4f} "
        f"over {len(winner['pooled_pccs'])} seeds"
    )

    # **The winner's vectors are WRITTEN, in the baseline's exact layout.**
    # Otherwise the comparison could not be re-derived: the baseline's five
    # are on disk and the winner's would exist only for the length of this
    # process. Same marker, same columns, same six decimals as
    # phase3.write_outputs -- cluster_csv mirrors that writer rather than
    # replacing it, because gate 1 compares its output byte for byte.
    from .cluster_csv import write_predictions

    fold_of = {int(pid): assignments[int(pid)] for pid in winner["oof_ids"]}
    for seed, vector in winner["oof_by_seed"].items():
        write_predictions(
            ctx.path(f"seed_{seed}__predictions.csv", tier="CLUSTER-ONLY"),
            zip(winner["oof_ids"], winner["oof_truth"], vector,
                (fold_of[int(pid)] for pid in winner["oof_ids"])),
        )
    ctx.log(
        f"  wrote {len(winner['oof_by_seed'])} per-seed prediction files in "
        "the baseline's layout"
    )

    # ---- exit criterion 4: paired BCa against the untuned baseline ---------
    baseline = phase7b.BASELINE
    comparison = {
        "baseline": baseline,
        "note": (
            "the tuned arm reports its OWN sd; PLAN §4.12.1 forbids inheriting "
            "the untuned arm's"
        ),
    }
    baseline_paths = phase7b.baseline_paths_from_inputs(declared)
    if baseline_paths:
        baseline_by_seed = phase7b.load_baseline_predictions(
            baseline_paths,
            [entry["seed"] for entry in winner["seeds"]],
            winner["oof_ids"],
        )

        # **Verify the declared directory IS the 0.2520 arm before pairing
        # against it.** The five vectors' own PCCs must reproduce the recorded
        # baseline; if the wrong run were declared the comparison would still
        # complete, still produce intervals, and be against another arm
        # entirely. Checked rather than trusted, on the machine holding both.
        observed = [
            float(pcc(winner["oof_truth"], vector))
            for vector in baseline_by_seed.values()
        ]
        observed_mean = float(np.mean(observed))
        if abs(observed_mean - baseline["pcc"]) > 5e-3:
            raise ValueError(
                f"the declared baseline scores {observed_mean:.4f} over its "
                f"five seeds, but phase7b.BASELINE records {baseline['pcc']}. "
                "Either the wrong run directory is declared, or the recorded "
                "baseline is stale -- resolve it before pairing anything "
                "against it."
            )
        ctx.log(
            f"  baseline verified: {observed_mean:.4f} over "
            f"{len(observed)} seeds, against a recorded {baseline['pcc']}"
        )

        comparison["paired"] = phase7b.paired_comparison(
            truth=winner["oof_truth"],
            winner_by_seed=winner["oof_by_seed"],
            baseline_by_seed=baseline_by_seed,
            winner_sd=band["sd"],
        )
        for entry in comparison["paired"]["per_seed"]:
            ctx.log(
                f"    seed {entry['seed']:6d}: winner {entry['winner_pcc']:.4f} "
                f"baseline {entry['baseline_pcc']:.4f} delta "
                f"{entry['delta']:+.4f} [{entry['lo']:+.4f}, {entry['hi']:+.4f}]"
                + ("  excludes 0" if entry["excludes_zero"] else "")
            )
        ctx.log(f"  {comparison['paired']['reading']}")
    else:
        comparison["paired"] = None
        comparison["why_absent"] = (
            f"no {phase7b.BASELINE_INPUT_PREFIX}<seed> inputs declared; exit "
            "criterion 4 is NOT met by this run"
        )
        ctx.log(
            f"  NO paired BCa: no {phase7b.BASELINE_INPUT_PREFIX}<seed> inputs"
        )

    summary = {
        "protocol": result["protocol"],
        "trials": result["trials"],
        "selected": selected,
        "exhaustion": result["exhaustion"],
        "winner": {
            "configuration": selected["config"],
            "seeds": winner["seeds"],
            "pooled_pccs": winner["pooled_pccs"],
            "band": band,
        },
        "comparison": comparison,
        "oof_computed_for": (
            "the selected configuration only -- never for the other "
            f"{phase7b.SEARCH_BUDGET - 1}"
        ),
    }
    # `ctx.atomic`, as every other task writes metrics.json. This line read
    # `ctx.write_metrics(...)` -- a method RunContext does not have, and the
    # only occurrence of that name in the file. See tests/test_phase7b_task.py.
    with ctx.atomic("metrics.json", tier="SHAREABLE") as tmp:
        tmp.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )


def task_phase7c_paired(ctx: RunContext) -> None:
    """Phase 7C exit criterion 6: the per-seed paired BCa, over stored vectors.

    **This task fits nothing.** Every arm's out-of-fold vector is already on
    disk -- 35 CSVs, one per arm per seed -- and PLAN §4.3's condition 1 is a
    re-analysis of them. Phase 7C reported nine verdicts from condition 2
    alone; this computes the condition that was promised in
    ``phase7c.NOT_A_SEARCH['carried_over']`` and never run.

    The conditions are ANDed, so this can only ever WITHDRAW claims -- it
    cannot rescue the three nulls, and is not permitted to try. What it adds
    for them is the distinction between an underpowered contrast and an
    absent effect: see ``phase7c.PAIRED_BCA_IS_THE_MISSING_CONDITION``.

    **The declared vectors are verified against the recorded round before
    anything is paired.** Mean and SD, every arm. Thirty-five files is enough
    that one wrong path is a live risk, and the wrong path would still load,
    still pair, and still produce plausible intervals.
    """
    import json

    from . import phase7c

    task = ctx.config["task"]
    declared = {entry["name"]: entry for entry in ctx.inputs}
    round_label = task["round"]
    seeds = [int(s) for s in task.get("seeds", phase7c.SEEDS)]
    n_boot = int(task.get("n_boot", 10000))

    paths_by_arm = phase7c.oof_paths_from_inputs(declared)
    ctx.log(
        f"ROUND {round_label}: {len(paths_by_arm)} arms x {len(seeds)} seeds = "
        f"{sum(len(v) for v in paths_by_arm.values())} declared vectors"
    )

    loaded = phase7c.load_oof_vectors(paths_by_arm, seeds=seeds)
    ctx.log(
        f"  loaded {len(loaded['patient_ids'])} patients; the TRUTH column "
        "agrees across every vector"
    )

    verification = phase7c.verify_against_record(loaded, round_label=round_label)
    for entry in verification["arms"]:
        ctx.log(
            f"  {entry['arm']:24} pcc {entry['observed_pcc']:.4f} "
            f"(recorded {entry['recorded_pcc']}) sd {entry['observed_sd']:.4f} "
            f"(recorded {entry['recorded_sd']})"
        )
    ctx.log(
        f"  verified: all {len(verification['arms'])} arms reproduce round "
        f"{round_label} within {verification['tolerance']}"
    )

    ctx.log(f"PAIRED BCa: n_boot {n_boot}, per seed, paired over patients")
    matrix = phase7c.paired_matrix(
        loaded, round_label=round_label, n_boot=n_boot
    )

    for group, label in (
        (matrix["against_identity"], "against identity"),
        (matrix["comparisons"], "declared pair"),
    ):
        for name, entry in group.items():
            paired = entry["paired"]
            ctx.log(f"  [{label}] {name} -- delta is {entry['delta_is']}")
            for seed_entry in paired["per_seed"]:
                ctx.log(
                    f"    seed {seed_entry['seed']:6d}: "
                    f"{seed_entry['delta']:+.4f} "
                    f"[{seed_entry['lo']:+.4f}, {seed_entry['hi']:+.4f}]"
                    + ("  excludes 0" if seed_entry["excludes_zero"] else "")
                )
            ctx.log(
                f"    mean {paired['mean_delta']:+.4f}  "
                f"cond1 {entry['condition_1_paired_bca']}  "
                f"cond2 {entry['condition_2_exceeds_threshold']}  "
                f"=> claimable {entry['claimable']}"
                + (
                    "   ** VERDICT CHANGED **"
                    if entry.get("verdict_changed") else ""
                )
            )

    if matrix["verdicts_changed"]:
        ctx.log(
            "VERDICTS CHANGED by adding condition 1: "
            + ", ".join(matrix["verdicts_changed"])
        )
    else:
        ctx.log("no verdict changed: condition 1 agrees with condition 2 throughout")
    if matrix["underpowered_nulls"]:
        ctx.log(
            "UNDERPOWERED, not absent -- intervals exclude zero but the "
            "unpaired threshold is not cleared: "
            + ", ".join(matrix["underpowered_nulls"])
        )

    summary = {
        "round": round_label,
        "seeds": seeds,
        "n_boot": n_boot,
        "n_patients": len(loaded["patient_ids"]),
        "verification": verification,
        "arms": {
            name: {
                "pccs": arm["pccs"], "mean": arm["mean"],
                "sd": arm["sd"], "n_seeds": arm["n_seeds"],
            }
            for name, arm in loaded["arms"].items()
        },
        "matrix": matrix,
        "criterion": phase7c.PAIRED_BCA_IS_THE_MISSING_CONDITION,
    }
    with ctx.atomic("metrics.json", tier="SHAREABLE") as tmp:
        tmp.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )


def as_builtin(value):
    """Convert numpy scalars and arrays to Python types, recursively.

    **[MEASURED 2026-08-04] Fixed at the boundary rather than at each call
    site.** ``task_grad_cam`` crashed writing its report on an ``np.bool_``
    from a comparison -- ``x > threshold`` gives ``np.bool_``, not ``bool``.
    Casting at every comparison would fix that one and miss the next, and the
    report has three new fields and will grow.

    **``default=`` on ``json.dumps`` is NOT sufficient, and the reason is an
    asymmetry worth knowing.** It is only consulted for objects json cannot
    serialise:

        np.bool_    TypeError
        np.int64    TypeError
        np.float32  TypeError
        np.float64  SERIALISES SILENTLY -- it subclasses float

    So a handler would catch the loud cases and leave ``np.float64`` in the
    output: a value that serialises but is not a Python type, which is the
    quieter version of the same problem and survives a round trip looking
    fine. A coercion pass normalises all four.
    """
    if isinstance(value, dict):
        return {key: as_builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [as_builtin(item) for item in value]
    if isinstance(value, np.ndarray):
        return as_builtin(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value


def task_grad_cam(ctx: RunContext) -> None:
    """Phase 8: Grad-CAM for arm A, gated by the randomisation test.

    **Nothing is written until two gates pass.** The refit must reproduce arm
    A's 0.2520 (``phase8.REPRODUCE_GATE``), and every map must degrade under
    model-parameter randomisation (``phase8.RANDOMISATION_TEST``). A map that
    survives randomisation is an edge detector wearing an explanation's name.

    **The target layer is block 11, and that is measured rather than
    conventional.** At block 12's output the patch tokens feed nothing -- the
    head reads the CLS token after a per-token LayerNorm -- so their gradient
    is identically zero, and a zero map normalises float noise to full range.
    ``gradcam.cam`` refuses it. ``phase8.GRAD_CAM_TARGET``.
    """
    import json

    from . import gradcam, phase8
    from .data.manifest import load_manifest
    from .eval.metrics import pcc
    from .train import phase3
    from .train.torch_backbone import FrozenExtractor

    task = ctx.config["task"]
    declared = {entry["name"]: entry for entry in ctx.inputs}
    manifest_dir = Path(declared[task["manifest_artifact"]]["path"])
    staged_dir = Path(declared[task["staged_artifact"]]["path"])
    seeds = [int(s) for s in task["seeds"]]

    images, labels, patient_ids, _ = phase3.load_inputs(
        manifest_dir, staged_dir, task["geometry"], task["label"]
    )
    rows = load_manifest(manifest_dir / "manifest.csv")
    assignments = {int(row["patient_id"]): int(row["fold"]) for row in rows}

    # ---- gate 1: the refit must BE arm A ---------------------------------
    embeddings_dir = Path(declared[task["embeddings_artifact"]]["path"])
    features, _ = embeddings_module_load(embeddings_dir, patient_ids)
    heads, pooled = phase8_refit_heads(
        features, labels, patient_ids, assignments, seeds, task, ctx.log
    )
    band = phase3.seed_variance(pooled)
    gate = phase8.REPRODUCE_GATE
    ctx.log(
        f"REFIT: {band['mean']:.4f} sd {band['sd']:.4f} over {len(pooled)} "
        f"seeds, against a recorded {gate['expected_pcc']}"
    )
    if abs(band["mean"] - gate["expected_pcc"]) > gate["tolerance"]:
        raise ValueError(
            f"the refit scores {band['mean']:.4f} against arm A's recorded "
            f"{gate['expected_pcc']}. These are not arm A's heads, so any map "
            "would explain a different model -- refusing before computing one"
        )

    # ---- the stratified sample -------------------------------------------
    sample = phase8_stratified_sample(patient_ids, labels)
    ctx.log(
        f"SAMPLE: {len(sample)} patients, {phase8.STRATIFIED_SAMPLE['per_stratum']} "
        f"from each of {phase8.STRATIFIED_SAMPLE['n_strata']} strata, "
        "no substitution"
    )

    extractor = FrozenExtractor(task["backbone"], batch_size=task["batch_size"])
    index = {int(pid): i for i, pid in enumerate(patient_ids)}

    def maps_for(patient: int, weights_by_seed: dict) -> dict:
        """One patient's map per seed, from the fold-model that held them out."""
        pixels = np.asarray(images[index[patient]: index[patient] + 1])
        per_seed = {}
        for seed, weights in weights_by_seed.items():
            acts, grads = gradcam.token_activations_and_gradients(
                extractor, pixels, weights
            )
            per_seed[seed] = gradcam.map_for(acts[0], grads[0])
        return per_seed

    held_out = {
        patient: {
            seed: heads[(seed, assignments[patient])] for seed in seeds
        }
        for patient in sample
    }

    originals, stability = {}, {}
    for patient in sample:
        per_seed = maps_for(patient, held_out[patient])
        grids = [entry["grid"].ravel() for entry in per_seed.values()]
        originals[patient] = np.mean(grids, axis=0)
        # **Stability is measured, not averaged away.** A mean map over five
        # seeds looks authoritative and hides whether the five agreed.
        stability[patient] = [
            gradcam.similarity(grids[i], grids[j])
            for i in range(len(grids)) for j in range(i + 1, len(grids))
        ]
        ctx.log(
            f"  patient {patient}: seed agreement median "
            f"{float(np.median(stability[patient])):.3f}"
        )

    # ---- gate 2: the randomisation test -----------------------------------
    ordered_maps = [originals[p] for p in sample]
    baseline = gradcam.between_patient_baseline(ordered_maps)
    spread = gradcam.between_patient_distribution(ordered_maps)
    ctx.log(
        f"BETWEEN-PATIENT BASELINE: {baseline:.4f} (the pass criterion), from "
        f"{spread['n_pairs']} pairs -- mean {spread['mean']:.4f} sd "
        f"{spread['sd']:.4f}, q05 {spread['q05']:.4f} q95 {spread['q95']:.4f}, "
        f"range [{spread['min']:.4f}, {spread['max']:.4f}]"
    )

    curves = []
    for patient in sample:
        randomised = {}
        for stage in phase8_randomisation_stages(extractor):
            acts, grads = gradcam.token_activations_and_gradients(
                extractor, np.asarray(images[index[patient]: index[patient] + 1]),
                held_out[patient][seeds[0]],
            )
            randomised[stage] = gradcam.map_for(acts[0], grads[0])["grid"].ravel()
        curves.append(gradcam.randomisation_curve(originals[patient], randomised))
    separable = gradcam.separability(
        [curve["final"] for curve in curves], ordered_maps
    )
    verdict = gradcam.randomisation_verdict(
        curves, baseline, distribution=spread, separable=separable
    )
    ctx.log(f"RANDOMISATION: {verdict['reading']}")
    ctx.log(
        f"SEPARABILITY: observed {separable['observed_difference']:+.4f}, "
        f"95% CI [{separable['ci95'][0]:+.4f}, {separable['ci95'][1]:+.4f}] "
        f"(bootstrap over patients) -- {separable['reading']}"
    )

    summary = {
        "pre_registration": phase8.summary(),
        "refit": {"mean": band["mean"], "sd": band["sd"], "pooled": pooled},
        "sample": [int(p) for p in sample],
        "seed_stability": {
            str(patient): {
                "median": float(np.median(values)), "min": float(np.min(values)),
            }
            for patient, values in stability.items()
        },
        "randomisation": verdict,
        "maps_published": bool(verdict["passes"]),
    }
    with ctx.atomic("metrics.json", tier="SHAREABLE") as tmp:
        tmp.write_text(
            json.dumps(as_builtin(summary), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    # **[DECIDED] The npz is the artifact 'published' scopes over; the sheet
    # is not.** ``phase8.PUBLICATION_SCOPE``: the review asks whether the
    # METHOD puts maps on plausible anatomy, which is a different question
    # from the gate's and is worth answering whichever way the gate went. So
    # the sheet is always rendered and the npz is written only on a pass --
    # structurally, because a stamp is prose and the figure is what gets
    # reused (FROZEN_BACKBONE_CAVEAT's own lesson).
    if verdict["passes"]:
        npz = ctx.path("grad_cam_maps.npz", tier="CLUSTER-ONLY")
        np.savez_compressed(
            npz,
            patient_ids=np.asarray(sample, dtype=int),
            maps=np.stack([originals[p] for p in sample]),
            caveat=np.asarray(phase8.FROZEN_BACKBONE_CAVEAT["text"]),
            layer=np.asarray(phase8.GRAD_CAM_TARGET["layer"]),
            resolution_floor_px=np.asarray(phase8.RESOLUTION_FLOOR["floor_px"]),
        )
        ctx.log(f"  wrote {len(sample)} maps with the caveat attached")
    else:
        ctx.log(
            "  NOT writing grad_cam_maps.npz: the gate failed, and the npz is "
            "what 'published' scopes over (phase8.PUBLICATION_SCOPE)"
        )

    # ---- the sheet: the check no test substitutes for ---------------------
    from . import gradcam_sheet
    from .geometry.render import save_sheet

    masks = phase8_content_masks(
        staged_dir, task["geometry"], int(images.shape[1])
    )
    percentiles = verdict.get("final_percentiles") or []
    entries = []
    for position, patient in enumerate(sample):
        grid_map = gradcam.to_grid(originals[patient])
        content = masks.get(int(patient))
        entries.append(gradcam_sheet.per_patient(
            patient,
            np.asarray(images[index[patient]], dtype=np.uint8),
            grid_map,
            gradcam.upsample(grid_map),
            seed_agreement=stability[patient],
            content=content,
            # Marked on the patient's OWN panel, not only in the banner.
            survivor=(
                percentiles[position]
                if position in verdict["survivors"] and position < len(percentiles)
                else None
            ),
        ))
    sheet_path = ctx.path("grad_cam_sheet.png", tier="CLUSTER-ONLY")
    save_sheet(gradcam_sheet.build_sheet(entries), sheet_path)
    banner = gradcam_sheet.verdict_banner(verdict)
    ctx.log(f"  SHEET BANNER: {banner}")
    report = gradcam_sheet.summarise(entries)
    report["verdict_banner"] = banner
    report["maps_published"] = bool(verdict["passes"])
    report["publication_scope"] = phase8.PUBLICATION_SCOPE
    # **[ADDED 2026-08-08] The per-patient reports are serialised.** They were
    # aggregated away, so correcting the out-of-content expectation could not
    # be done as a re-analysis and needed a re-run. A statistic reported per
    # patient in the log and only in aggregate in the artifact cannot be
    # rechecked against a better baseline later.
    report["per_patient"] = [entry["report"] for entry in entries]
    report["edge_convergence"] = gradcam_sheet.edge_convergence(
        entries, finals=[curve["final"] for curve in curves]
    )
    ctx.log(
        f"  EDGE: outer-ring median {report['outer_ring_mass_median']:.3f} "
        f"against a uniform {report['outer_ring_uniform']:.3f}; "
        f"{report['n_ring_at_or_above_uniform']} of {report['n_patients']} at "
        "or above"
    )
    # **There is no single uniform value any more, which is the point of the
    # correction.** Each patient is scored against their own expectation, so
    # the log reports the median of those alongside the median ratio.
    if "out_of_content_ratio_median" in report:
        ctx.log(
            f"  PAD: out-of-content median "
            f"{report['out_of_content_mass_median']:.3f} against each "
            f"patient's OWN expectation (median "
            f"{report['out_of_content_expected_median']:.3f}); ratio median "
            f"{report['out_of_content_ratio_median']:.3f}; "
            f"{report['n_out_of_content_above_own_uniform']} of "
            f"{report['n_out_of_content_defined']} above their own"
        )
    if report.get("n_out_of_content_undefined"):
        ctx.log(
            f"  PAD: {report['n_out_of_content_undefined']} patient(s) have no "
            "out-of-content region -- the statistic is UNDEFINED there, not zero"
        )
    for name, entry in report["edge_convergence"].items():
        if isinstance(entry, dict) and "spearman" in entry:
            ctx.log(
                f"  CONVERGENCE {name}: rho {entry['spearman']:+.3f} "
                f"CI [{entry['ci95'][0]:+.3f}, {entry['ci95'][1]:+.3f}] -- "
                f"{entry['reading']}"
            )
    with ctx.atomic("grad_cam_sheet.json", tier="SHAREABLE") as tmp:
        tmp.write_text(json.dumps(as_builtin(report), indent=2, sort_keys=True) + "\n",
                       encoding="utf-8")
    ctx.log(
        f"  sheet: {report['n_patients']} patients, top-10 share median "
        f"{report[f'top_{gradcam_sheet.CONCENTRATION_AT[0]}_share_median']:.0%} "
        f"against a uniform {report['uniform_top_k_share']:.0%}; "
        f"{report['n_at_or_below_uniform']} at or below uniform"
    )
    for question in gradcam_sheet.REVIEW_QUESTIONS:
        ctx.log(f"    REVIEW: {question}")

    # **The raise comes LAST**, after metrics.json and the sheet are both on
    # disk. Failing earlier would have made the gate's outcome decide whether
    # the method could be reviewed at all, which is the tension
    # phase8.PUBLICATION_SCOPE resolves: the review is a check on the METHOD
    # and is worth answering whichever way the gate went.
    if not verdict["passes"]:
        raise ValueError(
            f"{len(verdict['survivors'])} of {verdict['n_patients']} maps "
            "survive model-parameter randomisation. NO MAPS ARE PUBLISHED -- "
            "grad_cam_maps.npz is not written. The sheet IS rendered, stamped "
            "with the verdict, for review of the method "
            "(phase8.PUBLICATION_SCOPE, exit criteria 2 and 7)"
        )


def task_grad_cam_softmax(ctx: RunContext) -> None:
    """Phase 8b: the Grad-CAM VARIANT -- softmax weighting, negatives
    dropped -- as a SECOND METHOD beside the original, with its own gate.

    **The existing Grad-CAM's problem was never the weighting -- it was
    gate estimability at n=15, and that verdict stands; changing the
    statistic after a failed gate is the move this project has corrected
    twice, which is why this is a second method with its own gate, both
    reported** (``phase8.GRAD_CAM_SOFTMAX_REGISTERED``).

    One forward-backward pass per patient per seed serves BOTH methods'
    aggregations -- the methods differ only downstream of the activations
    and gradients. The original's maps are recomputed through the
    original's own code path purely for the 8c comparison surface; its
    verdict is untouched and stamped, never revised.
    """
    import json

    from . import gradcam, gradcam_sheet, node_weights, phase8
    from .data.manifest import load_manifest
    from .train import phase3
    from .train.torch_backbone import FrozenExtractor

    task = ctx.config["task"]
    declared = {entry["name"]: entry for entry in ctx.inputs}
    manifest_dir = Path(declared[task["manifest_artifact"]]["path"])
    staged_dir = Path(declared[task["staged_artifact"]]["path"])
    seeds = [int(s) for s in task["seeds"]]

    images, labels, patient_ids, _ = phase3.load_inputs(
        manifest_dir, staged_dir, task["geometry"], task["label"]
    )
    rows = load_manifest(manifest_dir / "manifest.csv")
    assignments = {int(row["patient_id"]): int(row["fold"]) for row in rows}

    # ---- gate 1: the refit must BE arm A (REPRODUCE_GATE, unchanged) -----
    embeddings_dir = Path(declared[task["embeddings_artifact"]]["path"])
    features, _ = embeddings_module_load(embeddings_dir, patient_ids)
    heads, pooled = phase8_refit_heads(
        features, labels, patient_ids, assignments, seeds, task, ctx.log
    )
    band = phase3.seed_variance(pooled)
    gate = phase8.REPRODUCE_GATE
    ctx.log(
        f"REFIT: {band['mean']:.4f} sd {band['sd']:.4f} over {len(pooled)} "
        f"seeds, against a recorded {gate['expected_pcc']}"
    )
    if abs(band["mean"] - gate["expected_pcc"]) > gate["tolerance"]:
        raise ValueError(
            f"the refit scores {band['mean']:.4f} against arm A's recorded "
            f"{gate['expected_pcc']}. These are not arm A's heads, so any map "
            "would explain a different model -- refusing before computing one"
        )

    sample = phase8_stratified_sample(patient_ids, labels)
    extractor = FrozenExtractor(task["backbone"], batch_size=task["batch_size"])
    index = {int(pid): i for i, pid in enumerate(patient_ids)}
    held_out = {
        patient: {seed: heads[(seed, assignments[patient])] for seed in seeds}
        for patient in sample
    }

    # ---- both methods from ONE pass per (patient, seed) ------------------
    #
    # **Per-cell refusal handling (GRAD_CAM_SOFTMAX_REFUSAL_HANDLING).** A
    # cell whose variant map is all-zero after the ReLU records
    # ``all_zero: true`` as a DATUM and the walk continues -- fatal-as-was
    # could never measure prevalence, and prevalence is the finding (pod
    # probe: 19 of 75 cells). The ORIGINAL method computes FIRST, so the
    # shared zero-gradient guard (a wrong-layer fault, not a method
    # refusal) still fails the run fatally on the original's path.
    originals, variants, variant_stability = {}, {}, {}
    concentration_pairs, refusals = [], {}
    for patient in sample:
        pixels = np.asarray(images[index[patient]: index[patient] + 1])
        original_grids, variant_grids = [], []
        original_conc, variant_conc = [], []
        refused_cells = []
        for seed in seeds:
            acts, grads = gradcam.token_activations_and_gradients(
                extractor, pixels, held_out[patient][seed]
            )
            original = gradcam.map_for(acts[0], grads[0])
            original_grids.append(original["grid"].ravel())
            original_conc.append(gradcam_sheet.concentration(
                gradcam.cam(acts[0], grads[0])
            )[f"top_{gradcam_sheet.CONCENTRATION_AT[0]}_share"])
            try:
                variant = gradcam.map_for_softmax(acts[0], grads[0])
            except gradcam.GradCamError as exc:
                refused_cells.append({
                    "seed": int(seed), "all_zero": True, "reason": str(exc),
                })
                continue
            variant_grids.append(variant["grid"].ravel())
            # Concentration on the RAW (pre-normalisation) maps -- the
            # statistic's own rule -- through the one shared definition.
            variant_conc.append(gradcam_sheet.concentration(
                gradcam.cam_softmax(acts[0], grads[0])
            )[f"top_{gradcam_sheet.CONCENTRATION_AT[0]}_share"])
        refusals[patient] = refused_cells
        originals[patient] = np.mean(original_grids, axis=0)
        if variant_grids:
            # The patient's map is the mean over SURVIVING cells, count
            # stamped; agreement over surviving pairs, UNDEFINED below two.
            variants[patient] = np.mean(variant_grids, axis=0)
            variant_stability[patient] = [
                gradcam.similarity(variant_grids[i], variant_grids[j])
                for i in range(len(variant_grids))
                for j in range(i + 1, len(variant_grids))
            ]
        concentration_pairs.append({
            "patient": int(patient),
            "n_cells_surviving": len(variant_grids),
            "n_cells_refused": len(refused_cells),
            "original_top10": float(np.mean(original_conc)),
            "variant_top10": (
                float(np.mean(variant_conc)) if variant_conc else None
            ),
            "paired_difference": (
                float(np.mean(variant_conc) - np.mean(original_conc))
                if variant_conc else None
            ),
        })
        agreement = variant_stability.get(patient)
        ctx.log(
            f"  patient {patient}: {len(refused_cells)}/{len(seeds)} cells "
            "all-zero; variant seed agreement "
            + (
                f"median {float(np.median(agreement)):.3f}"
                if agreement else "UNDEFINED (fewer than two surviving cells)"
            )
            + (
                f", top10 {np.mean(variant_conc):.3f} vs original "
                f"{np.mean(original_conc):.3f}"
                if variant_conc else ", no surviving variant cell"
            )
        )

    total_cells = len(sample) * len(seeds)
    total_refused = sum(len(v) for v in refusals.values())
    surviving_patients = [p for p in sample if p in variants]
    excluded_patients = [int(p) for p in sample if p not in variants]
    if excluded_patients:
        ctx.log(
            f"  EXCLUDED (no surviving cell, no variant map): "
            f"{excluded_patients} -- stamped, per the registered handling"
        )
    ctx.log(
        f"REFUSALS: {total_refused} of {total_cells} cells; gate 2 runs on "
        f"{len(surviving_patients)} of {len(sample)} patients"
    )

    # ---- the committed concentration reading, looked UP not composed -----
    #
    # **The both-numbers rule** (GRAD_CAM_SOFTMAX_REFUSAL_HANDLING):
    # refusal count and survivor concentration are one mechanism and are
    # never reported apart -- the sentence carries both, always.
    paired = [
        entry["paired_difference"] for entry in concentration_pairs
        if entry["paired_difference"] is not None
    ]
    median_difference = float(np.median(paired)) if paired else None
    readings = phase8.GRAD_CAM_SOFTMAX_REGISTERED["concentration_readings"]
    concentration_verdict = {
        "refused_cells": f"{total_refused} of {total_cells}",
        "median_paired_difference_on_survivors": median_difference,
        "both_numbers_sentence": (
            f"{total_refused} of {total_cells} cells refuse; on the rest, "
            f"the median paired top-10 difference is "
            f"{median_difference:+.4f}"
            if median_difference is not None
            else f"{total_refused} of {total_cells} cells refuse; no cell "
            "survived to compare"
        ),
        "reading": (
            None if median_difference is None
            else readings["if_concentrates"] if median_difference > 0
            else readings["if_diffuse"]
        ),
        "degenerate_form_rule": phase8.GRAD_CAM_SOFTMAX_REFUSAL_HANDLING[
            "both_numbers_rule"
        ],
        "descriptive_never_a_gate": readings["descriptive_never_a_gate"],
        "per_patient": concentration_pairs,
    }
    ctx.log(f"CONCENTRATION: {concentration_verdict['both_numbers_sentence']}")

    # ---- gate 2: randomisation, VARIANT maps, VARIANT'S OWN baseline -----
    #
    # Over the SURVIVING patients, n stamped. The registered-though-
    # unreached boundary is coded, not narrated: below seven survivors the
    # phase-level gate is UNESTIMABLE -- the original method's failure mode
    # arriving by another road, stated as such rather than discovered after
    # (GRAD_CAM_SOFTMAX_REFUSAL_HANDLING.registered_though_unreached).
    # **[CORRECTED 2026-08-15, after the rerun reached this boundary] The
    # UNESTIMABLE declaration is a COMPLETION path, not a crash.** The first
    # implementation raised here, so a REGISTERED verdict terminated the
    # run as FAILED and the artifacts below never landed -- the same
    # altitude class as the first crash: the registration says declared and
    # stamped, and a raise is neither. The verdict is unchanged; only its
    # delivery moved (phase8.GRAD_CAM_SOFTMAX_REFUSAL_HANDLING.outcome).
    UNESTIMABLE_BELOW = 7  # n(n-1)/2 >= 20 pairs for 5% resolution
    unestimable = None
    if len(surviving_patients) < UNESTIMABLE_BELOW:
        unestimable = (
            f"{len(surviving_patients)} of {len(sample)} patients have a "
            "variant map, below the registered estimability floor of "
            f"{UNESTIMABLE_BELOW} (n(n-1)/2 >= 20 baseline pairs) -- the "
            "original method's failure mode arriving by another road"
        )
    ordered = [variants[p] for p in surviving_patients]
    baseline = gradcam.between_patient_baseline(ordered)
    spread = gradcam.between_patient_distribution(ordered)
    ctx.log(
        f"VARIANT BASELINE (its own, never inherited): {baseline:.4f} from "
        f"{spread['n_pairs']} pairs over {len(surviving_patients)} patients"
    )

    # **Clean snapshot restored before EVERY patient's walk** -- the
    # dirty-restart defect (phase8.ARM_A_RANDOMISATION_RESTARTS_DIRTY) is
    # structurally impossible here rather than merely avoided.
    import torch

    snapshot = {
        key: value.detach().cpu().clone()
        for key, value in extractor.model.state_dict().items()
    }
    finals, final_refusals = [], []
    finals_patients = []
    for patient in surviving_patients if unestimable is None else []:
        extractor.model.load_state_dict(snapshot)
        pixels = np.asarray(images[index[patient]: index[patient] + 1])
        last = None
        refused = False
        for _stage in phase8_randomisation_stages(extractor):
            acts, grads = gradcam.token_activations_and_gradients(
                extractor, pixels, held_out[patient][seeds[0]]
            )
            # **The same per-cell rule at the finals**: a randomised map
            # that refuses is recorded and the patient leaves the finals
            # with n stamped -- never imputed a similarity.
            try:
                last = gradcam.map_for_softmax(
                    acts[0], grads[0]
                )["grid"].ravel()
            except gradcam.GradCamError:
                refused = True
        if refused or last is None:
            final_refusals.append(int(patient))
            continue
        finals.append(gradcam.similarity(variants[patient], last))
        finals_patients.append(int(patient))
    extractor.model.load_state_dict(snapshot)
    if final_refusals:
        ctx.log(
            f"  randomised-final refusals: {final_refusals} -- excluded "
            "from the finals, n stamped (the registered per-cell rule)"
        )
    if unestimable is None and len(finals) < UNESTIMABLE_BELOW:
        unestimable = (
            f"{len(finals)} of {len(surviving_patients)} randomised finals "
            "are defined, below the registered estimability floor of "
            f"{UNESTIMABLE_BELOW} -- the original method's failure mode "
            "arriving by another road"
        )

    if unestimable is None:
        separable = gradcam.separability(finals, ordered)
        resolves = bool(separable["excludes_zero"])
        pairs = np.asarray(spread["pairs"], dtype=float)
        percentiles = node_weights.per_patient_percentiles(finals, pairs)
        gate_verdict = "RESOLVES" if resolves else "UNRESOLVED"
        ctx.log(
            f"RANDOMISATION (variant): observed "
            f"{separable['observed_difference']:+.4f}, CI "
            f"[{separable['ci95'][0]:+.4f}, {separable['ci95'][1]:+.4f}] over "
            f"{len(finals)} finals -- {gate_verdict}"
        )
    else:
        # A verdict, not a failure: the run completes, everything below is
        # written, and the claim-scoped npz correctly never exists.
        separable, resolves, percentiles = None, False, []
        gate_verdict = "UNESTIMABLE"
        ctx.log(f"RANDOMISATION (variant): UNESTIMABLE -- {unestimable}")

    summary = {
        "registration": phase8.GRAD_CAM_SOFTMAX_REGISTERED,
        "refusal_handling": phase8.GRAD_CAM_SOFTMAX_REFUSAL_HANDLING,
        "method": "grad_cam_softmax",
        "refit": {"mean": band["mean"], "sd": band["sd"], "pooled": pooled},
        "sample": [int(p) for p in sample],
        # **The refusal ledger, stamped** -- per-cell data, the counts the
        # both-numbers rule binds to every report, and the exclusions.
        "refusals": {
            "cells": f"{total_refused} of {total_cells}",
            "per_patient": {
                str(int(p)): refusals[p] for p in sample
            },
            "patients_excluded_no_surviving_cell": excluded_patients,
            "randomised_final_refusals": final_refusals,
        },
        "variant_seed_stability": {
            str(p): (
                {"median": float(np.median(v)), "min": float(np.min(v)),
                 "n_pairs": len(v)}
                if (v := variant_stability.get(p))
                else {"undefined": "fewer than two surviving cells"}
            )
            for p in surviving_patients
        },
        "concentration": concentration_verdict,
        "randomisation": {
            "criterion": "phase-level separability, per-patient percentiles",
            "verdict": gate_verdict,
            **(
                {"unestimable": unestimable} if unestimable is not None else {}
            ),
            "n_patients": len(surviving_patients),
            "n_finals": len(finals),
            "finals_patients": finals_patients,
            "own_baseline": float(baseline),
            "baseline_distribution": {
                k: v for k, v in spread.items() if k != "pairs"
            },
            "separability": separable,
            "resolves": resolves,
            "per_patient_percentiles": percentiles,
        },
        "original_method": {
            "verdict_untouched": (
                "the original's gate failed and its maps stay unpublished; "
                "its grids here are RECOMPUTED for the comparison surface "
                "only (phase8.GRAD_CAM_SOFTMAX_REGISTERED.artifacts)"
            ),
        },
        "maps_published": resolves,
    }
    with ctx.atomic("metrics.json", tier="SHAREABLE") as tmp:
        tmp.write_text(
            json.dumps(as_builtin(summary), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    # ---- the 8c comparison surface: ALWAYS written, stamped ---------------
    # Grids for SURVIVING patients; the refusal ledger rides beside them so
    # the surface can never show survivor concentration without the count.
    pair_npz = ctx.path("grad_cam_methods.npz", tier="CLUSTER-ONLY")
    np.savez_compressed(
        pair_npz,
        patient_ids=np.asarray(surviving_patients, dtype=int),
        original_grids=np.stack([originals[p] for p in surviving_patients]),
        variant_grids=np.stack([variants[p] for p in surviving_patients]),
        refused_cells_per_patient=np.asarray(
            [len(refusals[p]) for p in surviving_patients], dtype=int
        ),
        cells_refusing_total=np.asarray(f"{total_refused} of {total_cells}"),
        patients_excluded=np.asarray(excluded_patients, dtype=int),
        both_numbers_sentence=np.asarray(
            concentration_verdict["both_numbers_sentence"]
        ),
        original_top10=np.asarray(
            [e["original_top10"] for e in concentration_pairs
             if e["patient"] in set(map(int, surviving_patients))]
        ),
        variant_top10=np.asarray(
            [e["variant_top10"] for e in concentration_pairs
             if e["patient"] in set(map(int, surviving_patients))],
            dtype=float,
        ),
        original_verdict=np.asarray(
            "FAILED its own gate; maps unpublished; shown for method "
            "comparison under the phase-level licence"
        ),
        variant_verdict=np.asarray(
            gate_verdict if unestimable is None
            else f"UNESTIMABLE -- {unestimable}"
        ),
        caveat=np.asarray(phase8.FROZEN_BACKBONE_CAVEAT["text"]),
        layer=np.asarray(phase8.GRAD_CAM_TARGET["layer"]),
        resolution_floor_px=np.asarray(phase8.RESOLUTION_FLOOR["floor_px"]),
    )
    ctx.log("  wrote grad_cam_methods.npz -- the 8c surface, both stamps")

    # ---- the claim-scoped artifact: only on a pass -----------------------
    if resolves:
        npz = ctx.path("grad_cam_softmax_maps.npz", tier="CLUSTER-ONLY")
        np.savez_compressed(
            npz,
            patient_ids=np.asarray(surviving_patients, dtype=int),
            maps=np.stack([variants[p] for p in surviving_patients]),
            refused_cells_per_patient=np.asarray(
                [len(refusals[p]) for p in surviving_patients], dtype=int
            ),
            cells_refusing_total=np.asarray(
                f"{total_refused} of {total_cells}"
            ),
            caveat=np.asarray(phase8.FROZEN_BACKBONE_CAVEAT["text"]),
            layer=np.asarray(phase8.GRAD_CAM_TARGET["layer"]),
            resolution_floor_px=np.asarray(
                phase8.RESOLUTION_FLOOR["floor_px"]
            ),
        )
        ctx.log(
            f"  wrote {len(surviving_patients)} variant maps with the "
            "caveat and the refusal counts"
        )
    elif unestimable is not None:
        # **A registered verdict is a COMPLETION, not a failure.** The run
        # finalizes normally: the verdict is in metrics.json and stamped on
        # the comparison surface, and the claim-scoped npz correctly never
        # exists. Re-running would reproduce it deterministically -- there
        # is nothing here a retry loop should be asked to retry.
        ctx.log(
            "  NOT writing grad_cam_softmax_maps.npz: the gate is "
            f"UNESTIMABLE ({unestimable}); the npz is what 'published' "
            "scopes over. Declared and stamped, as registered -- the run "
            "completes"
        )
    else:
        ctx.log(
            "  NOT writing grad_cam_softmax_maps.npz: the gate did not "
            "resolve, and the npz is what 'published' scopes over"
        )
        raise ValueError(
            "the variant's maps do not separate from their own "
            "between-patient baseline under parameter randomisation. "
            "NOTHING IS PUBLISHED for the variant -- metrics.json and the "
            "stamped comparison surface ARE written, because the method "
            "comparison is worth reviewing whichever way the gate went"
        )


def task_phase8c_sheet(ctx: RunContext) -> None:
    """Phase 8c, the clinical delivery: one self-contained HTML page per
    patient, a technical appendix, and a cohort index.

    [REBUILT 2026-08-15 after the maintainer's eye review --
    ``phase8c.CLINICAL_DELIVERY_REBUILT``: the measured data was sound and
    the delivery failed its audience.] Plain language on the pages; every
    verdict stamp, ledger and verbatim caveat on the linked appendix. The
    animation run's APNGs are DECLARED INPUTS (one rolled-up run
    directory), so the pages embed verified bytes.
    """
    import csv
    import io
    import json

    from . import phase8c
    from .data.manifest import load_manifest
    from .geometry import render
    from .geometry.mapping import map_patches
    from .geometry.patch_features import staged_from_row
    from .roadb_regioncrop import protected_patches
    from .train import phase3

    task = ctx.config["task"]
    declared = {entry["name"]: entry for entry in ctx.inputs}
    manifest_dir = Path(declared[task["manifest_artifact"]]["path"])
    staged_dir = Path(declared[task["staged_artifact"]]["path"])
    anim_dir = Path(declared[task["animation_artifact"]]["path"])

    images, labels, patient_ids, _ = phase3.load_inputs(
        manifest_dir, staged_dir, task["geometry"], task["label"]
    )
    index = {int(p): i for i, p in enumerate(patient_ids)}
    geometry_rows = {
        int(row["patient_id"]): row
        for row in phase3.load_geometry_rows(staged_dir)
    }
    manifest_rows = {
        int(row["patient_id"]): row
        for row in load_manifest(manifest_dir / "manifest.csv")
    }

    methods = np.load(Path(declared[task["methods_artifact"]]["path"]))
    node = np.load(Path(declared[task["node_weights_artifact"]]["path"]))
    caveat = str(methods["caveat"])
    original_stamp = str(methods["original_verdict"])
    variant_stamp = str(methods["variant_verdict"])
    sentence = str(methods["both_numbers_sentence"])
    sample = [int(p) for p in methods["patient_ids"]]
    node_index = {int(p): i for i, p in enumerate(node["patient_ids"])}

    def read_rows(name: str):
        text = Path(declared[name]["path"]).read_text(encoding="utf-8")
        lines = [l for l in text.splitlines() if not l.startswith("#")]
        return list(csv.DictReader(io.StringIO("\n".join(lines))))

    predictions: dict[int, list[float]] = {}
    truth: dict[int, float] = {}
    for name in task["prediction_inputs"]:
        for row in read_rows(name):
            pid = int(row["patient_id"])
            predictions.setdefault(pid, []).append(float(row["prediction"]))
            truth[pid] = float(row["truth"])
    curve_rows: list[dict] = []
    for name in task["curve_inputs"]:
        for row in read_rows(name):
            curve_rows.append({**row, "series": name})

    patches = protected_patches()
    if len(patches) != 22:
        raise phase8c.Phase8cError(
            f"protected_patches() returned {len(patches)}, not the 22 the "
            "panel-1 decision names (phase8c.PANEL_1_NAMING)"
        )

    grid_side = int(round(np.sqrt(methods["original_grids"].shape[1])))

    def anim_uri(axis: str, patient: int) -> str:
        path = anim_dir / f"8c_anim_{axis}_p{patient}.png"
        if not path.is_file():
            raise phase8c.Phase8cError(
                f"the animation run is missing {path.name}; the clinical "
                "page cannot embed an animation that was never rendered"
            )
        return phase8c.file_data_uri(path)

    for patient in sample:
        position = sample.index(patient)
        face = np.asarray(images[index[patient]], dtype=np.uint8)
        row = geometry_rows[patient]
        staged = staged_from_row(face, row)
        overlay = render.render_overlay(staged, map_patches(staged, patches))
        original_grid = methods["original_grids"][position].reshape(
            grid_side, grid_side
        )
        hero = phase8c.composite_heat(face, original_grid)
        scores = phase8c.rater_multiset(manifest_rows[patient])
        prediction_sentence = (
            "Five raters scored this outcome -- a cleft patient, an "
            "orthodontist, a speech and language therapist, a plastic "
            "surgeon and a psychologist (green marks; the tall green "
            "line is their average). The model's score is shown "
            "beside them in orange -- one small mark per training run, "
            "the tall orange line their average."
        )
        strip_path = anim_dir / f"8c_seed_strip_p{patient}.png"
        if not strip_path.is_file():
            raise phase8c.Phase8cError(
                f"the animation run is missing {strip_path.name}"
            )
        html = phase8c.clinical_page(
            patient=patient,
            hero_uri=phase8c.png_data_uri(hero),
            train_anim_uri=anim_uri("train", patient),
            walk_anim_uri=anim_uri("walk", patient),
            depth_anim_uri=anim_uri("depth", patient),
            seed_strip_uri=phase8c.file_data_uri(strip_path),
            regions_uri=phase8c.png_data_uri(overlay),
            prediction_uri=phase8c.png_data_uri(
                phase8c.prediction_panel(scores, predictions[patient])
            ),
            prediction_sentence=prediction_sentence,
            appendix_href="8c_appendix.html",
            index_href="8c_index.html",
        )
        ctx.path(f"8c_patient_{patient}.html", tier="CLUSTER-ONLY").write_text(
            html, encoding="utf-8"
        )
        ctx.log(f"  patient {patient}: clinical page written")

    # The cohort scatter and the appendix figures.
    cohort_ids = sorted(predictions)
    errors, spreads, marked = [], [], []
    for pid in cohort_ids:
        scores = phase8c.rater_multiset(manifest_rows[pid])
        errors.append(abs(float(np.mean(predictions[pid])) - truth[pid]))
        spreads.append(max(scores) - min(scores))
        marked.append(pid in set(sample))
    scatter_uri = phase8c.png_data_uri(phase8c.cohort_scatter(
        np.asarray(errors), np.asarray(spreads), np.asarray(marked)
    ))
    exemplar = int(node["patient_ids"][0])
    flatness_uri = phase8c.png_data_uri(
        phase8c.flatness_panel(np.asarray(node["weights"][0]))
    )
    curves_uri = phase8c.png_data_uri(phase8c.curves_panel(curve_rows))
    refusal_summary = (
        f"Refused cells total: {str(methods['cells_refusing_total'])}; "
        f"per-sampled-patient counts "
        f"{[int(v) for v in methods['refused_cells_per_patient']]}. The "
        f"flatness exemplar is patient {exemplar}; every patient's weights "
        "are equally uniform."
    )
    ctx.path("8c_appendix.html", tier="CLUSTER-ONLY").write_text(
        phase8c.appendix_page(
            caveat_verbatim=caveat,
            original_stamp=original_stamp,
            variant_stamp=variant_stamp,
            both_numbers_sentence=sentence,
            refusal_summary=refusal_summary,
            flatness_uri=flatness_uri,
            curves_uri=curves_uri,
            scatter_uri=scatter_uri,
            index_href="8c_index.html",
        ),
        encoding="utf-8",
    )
    ctx.path("8c_index.html", tier="CLUSTER-ONLY").write_text(
        phase8c.index_page(
            patients=sample, scatter_uri=scatter_uri,
            appendix_href="8c_appendix.html",
        ),
        encoding="utf-8",
    )

    summary = {
        "design": phase8c.SHEET_DESIGN,
        "rebuild": phase8c.CLINICAL_DELIVERY_REBUILT,
        "panel_1": phase8c.PANEL_1_NAMING,
        "pages": len(sample),
        "cohort_patients": len(cohort_ids),
        "stamps_on_appendix": {
            "original": original_stamp, "variant": variant_stamp,
        },
        "both_numbers_sentence": sentence,
    }
    with ctx.atomic("metrics.json", tier="SHAREABLE") as tmp:
        tmp.write_text(
            json.dumps(as_builtin(summary), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    ctx.log(f"clinical delivery: {len(sample)} pages + appendix + index")


def task_phase8c_animation(ctx: RunContext) -> None:
    """Phase 8c, the animation: heat over three REAL axes, measured frames
    only (``phase8c.ANIMATION_REGISTERED``).

    (a) the training trajectory at the registered checkpoints, captured
    during the refit through the head's ``on_step`` hook; (b) the
    randomisation walk, original method, clean snapshot per patient;
    (d) depth, blocks 0..10. Axis (c) is the 5-frame STATIC seed strip --
    computed here because the sheet task computes no maps -- labelled
    unordered. Every animation's final frame is READ from
    ``grad_cam_methods.npz`` (the endpoint resolution).
    """
    import json

    from . import gradcam, phase8, phase8c
    from .data.manifest import load_manifest
    from .train import phase3
    from .train.torch_backbone import FrozenExtractor

    task = ctx.config["task"]
    declared = {entry["name"]: entry for entry in ctx.inputs}
    manifest_dir = Path(declared[task["manifest_artifact"]]["path"])
    staged_dir = Path(declared[task["staged_artifact"]]["path"])
    seeds = [int(s) for s in task["seeds"]]

    images, labels, patient_ids, _ = phase3.load_inputs(
        manifest_dir, staged_dir, task["geometry"], task["label"]
    )
    rows = load_manifest(manifest_dir / "manifest.csv")
    assignments = {int(r["patient_id"]): int(r["fold"]) for r in rows}

    methods = np.load(Path(declared[task["methods_artifact"]]["path"]))
    caveat = str(methods["caveat"])
    original_stamp = str(methods["original_verdict"])
    sample = [int(p) for p in methods["patient_ids"]]
    grid_side = int(round(np.sqrt(methods["original_grids"].shape[1])))

    # ---- refit with checkpoint capture (seeds[0] only) --------------------
    embeddings_dir = Path(declared[task["embeddings_artifact"]]["path"])
    features, _ = embeddings_module_load(embeddings_dir, patient_ids)
    captured: dict = {}

    def capturing_head(seed, backbone):
        if seed != seeds[0]:
            return
        state = {"epoch": 1, "fold_key": len(captured)}
        checkpoints: dict = {}
        captured[state["fold_key"]] = checkpoints

        def on_step(step):
            capture = (
                state["epoch"] == 1 and step in phase8c.A_EPOCH1_STEPS
            ) or step == backbone.max_steps
            if capture:
                checkpoints[f"e{state['epoch']}s{step}"] = (
                    backbone.head_weights().copy()
                )
            if step == backbone.max_steps:
                state["epoch"] += 1

        backbone.on_step = on_step

    heads, pooled = phase8_refit_heads(
        features, labels, patient_ids, assignments, seeds, task, ctx.log,
        on_head=capturing_head,
    )
    band = phase3.seed_variance(pooled)
    gate = phase8.REPRODUCE_GATE
    if abs(band["mean"] - gate["expected_pcc"]) > gate["tolerance"]:
        raise ValueError(
            f"the refit scores {band['mean']:.4f} against arm A's recorded "
            f"{gate['expected_pcc']}; these are not arm A's heads"
        )
    fold_order = sorted({assignments[int(p)] for p in patient_ids})
    checkpoints_by_fold = {
        fold_order[key]: checkpoints
        for key, checkpoints in captured.items()
    }

    extractor = FrozenExtractor(task["backbone"], batch_size=task["batch_size"])
    index = {int(p): i for i, p in enumerate(patient_ids)}

    def one_map(pixels, head_weights, block=None):
        acts, grads = gradcam.token_activations_and_gradients(
            extractor, pixels, head_weights, block=block
        )
        return gradcam.map_for(acts[0], grads[0])["grid"]

    def endpoint_frame(face, position):
        return phase8c.frame(
            face,
            methods["original_grids"][position].reshape(grid_side, grid_side),
            axis_label="endpoint -- the page's map (5-seed mean)",
        )

    frame_counts: dict = {}
    import torch

    snapshot = {
        key: value.detach().cpu().clone()
        for key, value in extractor.model.state_dict().items()
    }
    for patient in sample:
        position = sample.index(patient)
        fold = assignments[patient]
        pixels = np.asarray(images[index[patient]: index[patient] + 1])

        # (a) the training trajectory.
        face = np.asarray(images[index[patient]], dtype=np.uint8)
        frames = [phase8c.frame(
            face, None,
            axis_label="before training: the photograph -- the model has "
            "not yet learned where to look",
        )]
        for label, weights in checkpoints_by_fold[fold].items():
            frames.append(phase8c.frame(
                face, one_map(pixels, weights),
                axis_label=f"training checkpoint {label} (seed {seeds[0]})",
            ))
        frames.append(endpoint_frame(face, position))
        _write_axis(ctx, phase8c, "train", patient, frames)
        frame_counts.setdefault("train", []).append(len(frames))

        # (b) the randomisation walk, clean snapshot per patient.
        extractor.model.load_state_dict(snapshot)
        trained = heads[(seeds[0], fold)]
        frames = [phase8c.frame(
            face, one_map(pixels, trained),
            axis_label=f"trained (seed {seeds[0]})",
        )]
        for stage in phase8_randomisation_stages(extractor):
            frames.append(phase8c.frame(
                face, one_map(pixels, trained),
                axis_label=f"randomised: {stage}",
            ))
        extractor.model.load_state_dict(snapshot)
        frames.append(endpoint_frame(face, position))
        _write_axis(ctx, phase8c, "walk", patient, frames)
        frame_counts.setdefault("walk", []).append(len(frames))

        # (d) depth, blocks 0..10 -- ending ON the registered layer.
        frames = []
        for block in range(0, 11):
            suffix = " (the registered layer)" if block == 10 else ""
            frames.append(phase8c.frame(
                face, one_map(pixels, trained, block=block),
                axis_label=f"block {block}{suffix}",
            ))
        frames.append(endpoint_frame(face, position))
        _write_axis(ctx, phase8c, "depth", patient, frames)
        frame_counts.setdefault("depth", []).append(len(frames))

        # (c) the seed strip -- STATIC, labelled unordered.
        strip = phase8c.side_by_side([
            phase8c.frame(
                face, one_map(pixels, heads[(seed, fold)]),
                axis_label=f"seed {seed} (unordered)",
            )
            for seed in seeds
        ])
        from .geometry import render

        render.save_sheet(strip, ctx.path(
            f"8c_seed_strip_p{patient}.png", tier="CLUSTER-ONLY"
        ))
        ctx.log(f"  patient {patient}: axes written")

    summary = {
        "registration": phase8c.ANIMATION_REGISTERED,
        "refit": {"mean": band["mean"], "sd": band["sd"]},
        "sample": sample,
        "frames": {
            axis: {"per_patient": counts}
            for axis, counts in frame_counts.items()
        },
        "captured_checkpoints": {
            str(fold): list(checkpoints)
            for fold, checkpoints in checkpoints_by_fold.items()
        },
    }
    with ctx.atomic("metrics.json", tier="SHAREABLE") as tmp:
        tmp.write_text(
            json.dumps(as_builtin(summary), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def _write_axis(ctx, phase8c_module, axis: str, patient: int, frames) -> None:
    """PNG frames beside the APNG: the frames are the measured objects."""
    from .geometry import render

    for number, image in enumerate(frames):
        render.save_sheet(image, ctx.path(
            f"8c_anim_{axis}_p{patient}_f{number:02d}.png", tier="CLUSTER-ONLY"
        ))
    phase8c_module.write_animation(frames, ctx.path(
        f"8c_anim_{axis}_p{patient}.png", tier="CLUSTER-ONLY"
    ))


def phase8_refit_heads(features, labels, patient_ids, assignments, seeds, task, log,
                       on_head=None):
    """Arm A's thirty heads, refitted so their gradients can be taken.

    ``on_head(seed, backbone)`` [ADDED 2026-08-15, Phase 8c] is called once
    per created head, BEFORE training -- the animation task attaches its
    ``on_step`` checkpoint capture there. Observation only: a callback that
    changed the head would change what the refit gate certifies.

    **[MEASURED 2026-08-04] The first draft of this called
    ``phase3.run_cv(..., keep_heads=True)`` -- a function at a path that does
    not exist, with a parameter that does not exist, returning a field that
    does not exist.** Nothing here could have caught it: the refit needs torch
    and a model, so no laptop test executes this path. Third instance of that
    shape in this project, after ``ctx.write_metrics`` and
    ``extractor.preprocess``.

    The real API is ``harness.run_cv(make_backbone=...)``, and ``FoldRun``
    carries no head -- ``harness.py`` is FROZEN, so retention cannot be added
    there. **The heads are captured through the factory instead**: the harness
    calls ``make_backbone`` once per fold, so a factory that records what it
    creates holds every fitted head afterwards. Phase 7C's
    ``_AugmentingFactory`` did exactly this to recover a fold ordinal.
    """
    from .train import harness
    from .train.torch_backbone import EmbeddingHeadBackbone

    heads, pooled = {}, []
    for seed in seeds:
        created: list = []

        def make_backbone(_seed=seed, _created=created):
            backbone = EmbeddingHeadBackbone(
                learning_rate=float(task["learning_rate"]),
                weight_decay=float(task["weight_decay"]),
                seed=_seed,
            )
            if on_head is not None:
                on_head(_seed, backbone)
            _created.append(backbone)
            return backbone

        config = harness.TrainConfig(
            seed=seed,
            inner_val_frac=float(task["inner_val_frac"]),
            max_epochs=int(task["max_epochs"]),
            patience=int(task["patience"]),
            monitor=task["monitor"],
        )
        result = harness.run_cv(
            features=features, labels=labels, patient_ids=patient_ids,
            assignments=assignments, make_backbone=make_backbone, config=config,
        )
        pooled.append(float(result.metrics()["pcc"]))
        # The harness walks folds in sorted order, so the nth backbone it
        # asked for is the nth fold's. Asserted rather than assumed, because a
        # silent misalignment would explain each patient with another fold's
        # head and produce entirely ordinary maps.
        folds = [run.fold for run in result.folds]
        if len(created) != len(folds):
            raise ValueError(
                f"the harness created {len(created)} backbones for "
                f"{len(folds)} folds; the capture is misaligned"
            )
        for fold, backbone in zip(folds, created):
            heads[(seed, int(fold))] = backbone.head_weights()
        log(f"  seed {seed}: pooled {pooled[-1]:.4f}")
    return heads, pooled


def phase8_content_masks(staged_dir: Path, geometry: str, size: int) -> dict:
    """``{patient_id: content mask}`` from ``geometry.csv``.

    **[MEASURED 2026-08-04] This called ``load_manifest`` on geometry.csv and
    was refused** -- that function validates against the manifest schema, and
    geometry.csv carries ``content_x``/``content_y``/``content_w``/
    ``content_h`` and the rest. ``phase3.load_geometry_rows`` is the reader
    for this file, already migrated to ``cluster_csv.read_cluster_csv`` and
    already marker-safe. It is the fourth site to read it and must not become
    a fifth reader.

    **The SAME content mask Phase 7C's sheet builds**, from the same staged
    geometry -- a second definition of "on the face" would be a second thing
    to keep in step, and out-of-content mass is measured against the pad
    fraction that definition implies.

    Extracted from ``task_grad_cam`` so a laptop test can drive it with a real
    geometry.csv: the call sat on a torch-only path, but ``load_manifest`` is
    laptop-reachable and the defect was therefore catchable here (R10).
    """
    from .train.augment_sheet import content_mask
    from .train.phase3 import load_geometry_rows

    masks = {}
    for row in load_geometry_rows(staged_dir):
        masks[int(row["patient_id"])] = content_mask(
            size,
            (row["content_x"], row["content_y"], row["content_w"],
             row["content_h"]),
            geometry,
        )
    return masks


def phase8_stratified_sample(patient_ids, labels):
    """Three patients from each of five equal-count grade strata.

    ``phase8.STRATIFIED_SAMPLE``: fixed seed, and **no substitution** -- a
    patient whose map fails to render is reported, never replaced.
    Replacement-on-rejection is cherry-picking with an audit trail that looks
    principled.
    """
    from . import phase8

    rule = phase8.STRATIFIED_SAMPLE
    order = np.argsort(np.asarray(labels, dtype=float), kind="mergesort")
    strata = np.array_split(order, rule["n_strata"])
    rng = np.random.default_rng(rule["seed"])
    chosen = []
    for stratum in strata:
        take = rng.choice(stratum, size=rule["per_stratum"], replace=False)
        chosen.extend(int(patient_ids[i]) for i in sorted(take))
    return chosen


def phase8_noise_std(parameter, group_std: float) -> float:
    """The std the randomising noise is drawn at, for one parameter tensor.

    **[MEASURED 2026-08-14, torch 2.13.0 CPU] ``parameter.std()`` returns NaN
    for a single-element tensor** -- the default correction is 1, so the
    denominator is zero -- and ``normal_`` then raises ``normal expects std
    >= 0.0, but found std -nan``. SR-GNN has three such parameters
    (``self_attn.Wa.bias``, ``weighted_attn.W.bias``, ``classifier.bias``),
    and the first sits in the FIRST group ``explanation_source`` randomises.
    The generator would have died on stage 1 of arm B's only working mode.

    Arm A never met it: ViT-B/16's blocks have no scalar parameters, which
    is why ``phase8_randomisation_stages`` ran. That function is deliberately
    left alone -- it is arm A's, it already ran, and changing the noise it
    draws would make a re-run disagree with the recorded one for a reason
    unrelated to arm A.

    A tensor with no spread of its own (one element, or a constant) borrows
    the GROUP's spread rather than being skipped: skipping would leave a
    trained parameter in place inside a stage that claims to have randomised
    everything above it, which is a check quietly doing less than it says.
    """
    own = float(parameter.detach().std(unbiased=False))
    if own > 0.0:
        return own
    if group_std > 0.0:
        return group_std
    raise ValueError(
        f"a parameter of shape {tuple(parameter.shape)} has no spread and "
        "neither does its group, so randomising it would change nothing. A "
        "stage that randomises nothing reports a test it did not run"
    )


def phase8_node_weight_stages(model, mode: str):
    """Arm B's randomisation, cumulatively, in the one mode that can run.

    ``phase8.ARM_A_AND_B_RANDOMISATIONS_ARE_NOT_EQUIVALENT``: arm A
    randomised its frozen backbone, which for it was also nearly the whole
    model AND the source of its map. Arm B's frozen backbone, trained
    layers and explanation source are three different things.

    * ``explanation_source`` -- the trained graph layers, which is where
      ``region_w`` is actually produced. This one works.
    * ``component_role`` -- **REFUSED, and measured rather than argued.**
      See ``phase8.COMPONENT_ROLE_IS_A_NO_OP``: randomising all 20,806,952
      backbone parameters leaves ``region_w`` and the logits BITWISE
      identical, because ``forward_from_features`` enters below the backbone
      and the map arrives precomputed. Asking arm A's question of arm B
      needs a re-extraction, not a parameter walk.

    Noise is matched to each parameter's own spread and drawn from one
    seeded generator, as ``phase8_randomisation_stages`` does -- with the
    scalar-parameter correction ``phase8_noise_std`` documents.

    **Validates eagerly and returns the walk.** Written as a generator, the
    refusal below would not fire until the first ``next()``, so a caller that
    built the walk and never stepped it would see nothing -- and a config
    error would surface on the cluster rather than on the laptop. Nothing
    here needs torch, which is the other half of the point.
    """
    from . import node_weights, phase8

    if mode not in node_weights.RANDOMISATION_MODES:
        raise ValueError(
            f"unknown randomisation mode {mode!r}; expected one of "
            f"{node_weights.RANDOMISATION_MODES}"
        )
    if mode == "component_role":
        # **Refusing beats running a no-op.** Walking model.backbone here
        # would complete, yield stage labels, and produce similarity 1.0 for
        # every patient -- which the gate would read as "the explanation
        # survives randomisation", the worst available verdict, arrived at
        # by testing nothing.
        raise ValueError(
            "component_role cannot be run as a parameter randomisation on "
            "this arm. [MEASURED] randomising all "
            f"{phase8.COMPONENT_ROLE_IS_A_NO_OP['backbone_parameters']:,} "
            "backbone parameters leaves region_w and the logits BITWISE "
            "identical: the artifact-fed path enters below the backbone, so "
            "the frozen representation is an INPUT here rather than a "
            "computation. On an imagenet-init arm that backbone was never "
            "even loaded with ImageNet weights -- it is a randomly "
            "initialised module that nothing reads. Arm A's question needs "
            "the map RE-EXTRACTED from a randomised backbone -- declare the "
            "control artifact as randomised_embeddings_artifact and the mode "
            "runs by that route instead (extract.RANDOMISED_BACKBONE, "
            "phase8.COMPONENT_ROLE_CONTROL_IS_A_REEXTRACTION)."
        )

    # The trained machinery, in the order it runs: attention, then
    # propagation, then the readout. **[MEASURED] only two of these six can
    # move the explanation** -- region_w is produced by self_attn and
    # weighted_attn, and the other four sit downstream of it
    # (phase8.WHICH_STAGES_CAN_MOVE_THE_EXPLANATION). The gate reads the
    # FINAL stage, where all six are randomised, so this is about reading
    # the curve rather than about the verdict.
    groups = [
        (name, getattr(model, name))
        for name in (
            "self_attn", "gnn_mlp1", "gnn_mlp2", "gnn_out",
            "weighted_attn", "classifier",
        )
        if hasattr(model, name)
    ]
    if not groups:
        raise ValueError(
            f"mode {mode!r} selected no parameters to randomise; nothing "
            "would be tested"
        )
    return _phase8_walk_stages(groups, mode)


def _phase8_walk_stages(groups, mode: str):
    """The cumulative walk itself, once ``phase8_node_weight_stages`` has
    decided the mode can run and which modules it touches."""
    import torch

    generator = torch.Generator(device="cpu").manual_seed(1337)
    for name, module in groups:
        with torch.no_grad():
            parameters = list(module.parameters())
            flat = (
                torch.cat([p.detach().reshape(-1) for p in parameters])
                if parameters else None
            )
            group_std = 0.0 if flat is None else float(flat.std(unbiased=False))
            for parameter in parameters:
                noise = torch.empty(
                    parameter.shape, dtype=parameter.dtype, device="cpu"
                )
                noise.normal_(
                    0.0, phase8_noise_std(parameter, group_std),
                    generator=generator,
                )
                parameter.copy_(noise.to(parameter.device))
        yield f"{mode}:{name}_and_above"


def phase8_randomisation_stages(extractor):
    """Randomise the network's weights top-down, yielding a label per stage.

    Cumulative and from the top, as Adebayo et al. specify: each stage adds
    the next block down, so a map that only degrades once the whole network is
    randomised is visibly different from one that degrades immediately.

    **[MEASURED 2026-08-14] Arm A's caller walks this once PER PATIENT over
    one extractor that nothing restores**, so patients after the first see a
    fully randomised model at every stage
    (``phase8.ARM_A_RANDOMISATION_RESTARTS_DIRTY``). The verdict reads the
    final stage only and the per-stage curve is never serialised, so nothing
    reported is wrong -- and this is left as it ran rather than corrected
    into a run that no longer matches its record. Arm B's counterpart
    restores a clean snapshot before every walk.
    """
    import torch

    blocks = extractor.model.blocks
    generator = torch.Generator(device="cpu").manual_seed(1337)
    for depth in range(len(blocks) - 1, -1, -1):
        with torch.no_grad():
            for parameter in blocks[depth].parameters():
                noise = torch.empty(
                    parameter.shape, dtype=parameter.dtype, device="cpu"
                )
                noise.normal_(0.0, float(parameter.std()), generator=generator)
                parameter.copy_(noise.to(parameter.device))
        yield f"blocks_{depth}_and_above"


def phase8_held_out_region_weights(models, packed, patient_ids, assignments):
    """``(n_patients, n_nodes)`` -- every patient explained by the fold-model
    that HELD THEM OUT, and no patient explained twice.

    ``phase8.SRGNN_NODE_WEIGHTS_R10_READ.per_fold``: explaining a patient
    with a model that trained on them explains a memorised label. The rule is
    enforced by CONSTRUCTION here rather than checked afterwards -- each
    fold's model is asked only for its own held-out rows, and a patient no
    fold held out leaves a row unfilled, which raises.

    This is where the arms MATCH: arm A's Grad-CAM is per-fold under exactly
    the same rule.
    """
    row_of = {int(pid): i for i, pid in enumerate(patient_ids)}
    out = None
    covered: set[int] = set()
    for fold, instance in models:
        held = [int(pid) for pid in patient_ids if assignments[int(pid)] == fold]
        if not held:
            raise ValueError(
                f"fold {fold} holds out no patient, so its model explains "
                "nobody. Either the fold assignment moved or the models are "
                "misaligned with the folds"
            )
        rows = [row_of[pid] for pid in held]
        weights = instance.region_weights(np.asarray(packed)[rows])
        if out is None:
            out = np.zeros((len(patient_ids), weights.shape[1]), dtype=np.float64)
        elif weights.shape[1] != out.shape[1]:
            raise ValueError(
                f"fold {fold} returned {weights.shape[1]} nodes against "
                f"{out.shape[1]} from an earlier fold. Two folds ranking "
                "different numbers of regions cannot be averaged"
            )
        out[rows] = weights
        overlap = covered & set(held)
        if overlap:
            raise ValueError(
                f"patients {sorted(overlap)[:5]} are held out by more than "
                "one fold; the later model would silently overwrite the "
                "earlier explanation"
            )
        covered.update(held)
    missing = [int(pid) for pid in patient_ids if int(pid) not in covered]
    if missing:
        raise ValueError(
            f"{len(missing)} patient(s) were explained by no fold-model, e.g. "
            f"{missing[:5]}. Their rows would be zeros, which rank as a "
            "perfectly flat explanation rather than as an absence"
        )
    return out


def task_node_weights(ctx: RunContext) -> None:
    """Phase 8 arm B: SR-GNN's node weights, gated before any are shown.

    The regions-voting explanation supervision asked for
    (``node_weights``), read off arm B -- ``p7_d1_srgnn_imagenet_g1_native``,
    the interpretable arm at 0.1719.

    **GATE 1 -- the refit must BE arm B.** Ten seeds through the ordinary
    graph-cleft path; if the pooled mean misses 0.1719 by more than 2e-3
    these are not arm B's models and every weight would describe a different
    one (``phase8.SRGNN_NODE_WEIGHTS_R10_READ.refit_gate``).

    **GATE 2 -- the randomisation test, on the mode that can run.** Node
    weights are an explanation, so Adebayo's check applies to them
    (``phase8.NODE_WEIGHT_RANDOMISATION_REGISTERED``). ``component_role`` is
    refused as a measured no-op (``phase8.COMPONENT_ROLE_IS_A_NO_OP``);
    ``explanation_source`` randomises the parameters that actually produce
    ``region_w``.

    **Two samples, two questions** (``phase8.ARM_B_SAMPLE_DECIDED``): the
    arm's own gate runs on all 237, and the A<->B-scoped gate on the shared
    fifteen -- arm A's stratified sample, drawn by the same function so the
    two arms cannot silently disagree about which patients they mean.

    **26 regions, not 27.** The whole-image node is split off and reported
    separately: a shared near-constant element inflates the between-patient
    baseline, and that baseline is the bar a randomised vector must fall
    BELOW (``phase8.SRGNN_NODE_WEIGHTS_R10_READ``).
    """
    import json

    from . import gradcam, node_weights, phase8
    from .train import graph_cleft
    from .train.harness import TrainConfig

    task = ctx.config["task"]
    declared = {entry["name"]: entry for entry in ctx.inputs}
    gate_spec = phase8.SRGNN_NODE_WEIGHTS_R10_READ["refit_gate"]

    seeds = [int(s) for s in task["seeds"]]
    if len(seeds) != len(set(seeds)):
        raise ValueError(f"duplicate seed in {seeds}; each must fit its own models")
    if len(seeds) != gate_spec["seeds"]:
        raise ValueError(
            f"the refit gate is stated over {gate_spec['seeds']} seeds and "
            f"{len(seeds)} are declared. The gate's tolerance is calibrated "
            "for that count, and a mean over a different one is a different "
            "quantity wearing the same name"
        )

    checkpoint_path = None
    checkpoint_sha256 = None
    if task.get("checkpoint"):
        if task["checkpoint"] not in declared:
            raise ValueError(
                f"task.checkpoint names input {task['checkpoint']!r}, which is "
                f"not declared. Declared inputs: {sorted(declared)}"
            )
        checkpoint_path = Path(declared[task["checkpoint"]]["path"])
        checkpoint_sha256 = declared[task["checkpoint"]]["rollup"]

    manifest_dir = Path(declared[task["manifest_artifact"]]["path"])
    label = task.get("label", "mean")
    patient_ids, labels, assignments = graph_cleft.load_labels_and_folds(
        manifest_dir, label
    )
    row_of = {int(pid): i for i, pid in enumerate(patient_ids)}

    common = dict(
        manifest_dir=manifest_dir,
        staged_dir=Path(declared[task["staged_artifact"]]["path"]),
        embeddings_dir=Path(declared[task["embeddings_artifact"]]["path"]),
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=checkpoint_sha256,
        backbone=task["backbone"],
        init=task["init"],
        geometry=task["geometry"],
        region_scheme=task["region_scheme"],
        label=label,
        trainable=task.get("trainable", "graph_layers"),
        deterministic=task.get("deterministic"),
        backbone_config={
            "learning_rate": task.get("learning_rate", 1e-4),
            "weight_decay": task.get("weight_decay", 0.01),
            "batch_size": task.get("batch_size", 16),
        },
        log=ctx.log,
    )

    # ---- gate 1: the refit must BE arm B, and the weights ride along -------
    pooled: list[float] = []
    weights_by_seed: dict[int, np.ndarray] = {}
    retained: dict = {}
    snapshots: dict = {}
    for seed in seeds:
        capture: dict = {}

        def on_fold_models(
            *, models, packed, patient_ids, assignments, boxes, map_shape,
            _c=capture,
        ):
            _c.update(
                models=models, packed=packed, boxes=boxes, map_shape=map_shape,
                patient_ids=[int(pid) for pid in patient_ids],
                assignments={int(k): int(v) for k, v in assignments.items()},
            )

        result = graph_cleft.run(
            **common, seed=seed,
            train_config=TrainConfig(
                max_epochs=task["max_epochs"],
                patience=task["patience"],
                inner_val_frac=task["inner_val_frac"],
                monitor=task["monitor"],
                seed=seed,
            ),
            on_fold_models=on_fold_models,
        )
        pooled.append(float(result.summary["oof"]["pcc"]))
        if not capture:
            raise ValueError(
                "graph_cleft.run returned without handing back its fold "
                "models, so there is nothing to read weights from"
            )
        if capture["patient_ids"] != [int(pid) for pid in patient_ids]:
            raise ValueError(
                "the run's patient order differs from this task's; every row "
                "index would address a different patient"
            )
        weights_by_seed[seed] = phase8_held_out_region_weights(
            capture["models"], capture["packed"],
            capture["patient_ids"], capture["assignments"],
        )
        ctx.log(f"  seed {seed}: pooled {pooled[-1]:.4f}, weights read per fold")
        if seed == seeds[0]:
            # **The randomisation runs on seed 0's models, as arm A's does**,
            # and their trained state is snapshotted on the CPU first: the
            # stage walk overwrites parameters in place, and a clean restore
            # before every walk is what keeps arm A's dirty-restart defect
            # (phase8.ARM_A_RANDOMISATION_RESTARTS_DIRTY) structurally
            # impossible here rather than merely avoided.
            retained = capture
            snapshots = {
                fold: {
                    key: value.detach().cpu().clone()
                    for key, value in instance.model.state_dict().items()
                }
                for fold, instance in capture["models"]
            }
        else:
            capture.clear()  # release five models before fitting five more

    band = graph_cleft.seed_variance(pooled)
    ctx.log(
        f"REFIT: {band['mean']:.4f} sd {band['sd']:.4f} over {len(pooled)} "
        f"seeds, against a recorded {gate_spec['expected_pcc']}"
    )
    if abs(band["mean"] - gate_spec["expected_pcc"]) > gate_spec["tolerance"]:
        raise ValueError(
            f"the refit scores {band['mean']:.4f} against arm B's recorded "
            f"{gate_spec['expected_pcc']}. These are not arm B's models, so "
            "any node weight would describe a different one -- refusing "
            "before reading one"
        )

    # ---- the explanation: mean over seeds, with the agreement kept ---------
    roi_by_seed, whole_by_seed = [], []
    for seed in seeds:
        region, whole = node_weights.split_whole_image(weights_by_seed[seed])
        roi_by_seed.append(region)
        whole_by_seed.append(whole)
    roi = np.mean(roi_by_seed, axis=0)
    whole_image = np.mean(whole_by_seed, axis=0)

    # **Agreement is measured, not averaged away** -- a mean vector over ten
    # seeds looks authoritative and hides whether the ten agreed. Measured on
    # the 26 the gate ranks, in the same statistic, so it is comparable to
    # arm A's seed stability directly.
    agreement = {}
    for index, patient in enumerate(patient_ids):
        vectors = [region[index] for region in roi_by_seed]
        agreement[int(patient)] = [
            gradcam.similarity(vectors[i], vectors[j])
            for i in range(len(vectors)) for j in range(i + 1, len(vectors))
        ]
    medians = [float(np.median(values)) for values in agreement.values()]
    ctx.log(
        f"SEED AGREEMENT over {len(seeds)} seeds: median of medians "
        f"{float(np.median(medians)):.3f}, worst patient "
        f"{float(np.min(medians)):.3f}"
    )

    # ---- gate 2: the randomisation, on the mode that can run --------------
    shared = phase8_stratified_sample(patient_ids, labels)
    shared_rows = [row_of[int(pid)] for pid in shared]
    sample_rule = phase8.ARM_B_SAMPLE_DECIDED
    if len(patient_ids) != sample_rule["own_gate"]["n"]:
        raise ValueError(
            f"the cohort has {len(patient_ids)} patients against the "
            f"{sample_rule['own_gate']['n']} the gate was designed on. The "
            "baseline's resolution, and therefore what the interval can "
            "resolve, is a function of that count "
            "(phase8.NODE_WEIGHT_STATISTIC_DECIDED)"
        )
    if len(shared_rows) != sample_rule["a_vs_b_comparison"]["n"]:
        raise ValueError(
            f"the shared sample is {len(shared_rows)} patients against arm "
            f"A's {sample_rule['a_vs_b_comparison']['n']}; the comparison "
            "would not be on matched patients"
        )

    def restore_clean():
        """Every mode starts from the trained state, always."""
        for fold, instance in retained["models"]:
            instance.model.load_state_dict(snapshots[fold])

    def held_rows(fold):
        held = [
            int(pid) for pid in patient_ids
            if retained["assignments"][int(pid)] == fold
        ]
        return [row_of[pid] for pid in held]

    # ---- the control set: arm A's intervention, at the only place arm B
    # has one. The MODEL is untouched here -- what changes is the map it
    # reads (extract.RANDOMISED_BACKBONE).
    randomised_packed = None
    control_report = None
    if task.get("randomised_embeddings_artifact"):
        from . import embeddings as _embeddings

        control_dir = Path(
            declared[task["randomised_embeddings_artifact"]]["path"]
        )
        control_maps, control_metadata = _embeddings.load(
            control_dir, manifest_ids=[int(pid) for pid in patient_ids]
        )
        _embeddings.check_pairing(
            control_metadata, kind="feature_map",
            backbone=task["backbone"], init=task["init"],
            geometry=task["geometry"], region_scheme=task["region_scheme"],
            checkpoint_sha256=checkpoint_sha256,
            # **Both directions.** A real set arriving here would report the
            # explanation unchanged under randomisation -- similarity 1.0 --
            # from a test that never ran.
            expect_randomised=True,
        )
        if tuple(int(v) for v in control_maps.shape[1:]) != tuple(
            retained["map_shape"]
        ):
            raise ValueError(
                f"the control maps are {control_maps.shape[1:]} against the "
                f"arm's {tuple(retained['map_shape'])}; the trained layers "
                "would be reading a different representation shape"
            )
        randomised_packed = graph_cleft.pack(control_maps, retained["boxes"])
        if randomised_packed.shape != np.asarray(retained["packed"]).shape:
            raise ValueError(
                f"the packed control is {randomised_packed.shape} against "
                f"{np.asarray(retained['packed']).shape}; the rows must have "
                "the arm's own layout or the boxes address the wrong columns"
            )
        control_report = {
            "artifact": control_dir.name,
            "randomisation": control_metadata.get("randomisation"),
            "why_here": phase8.COMPONENT_ROLE_IS_A_NO_OP["what_would_restore_it"],
        }
        ctx.log(
            f"CONTROL SET: {control_dir.name}, "
            f"{control_metadata['randomisation']['n_parameters']:,} backbone "
            "parameters randomised at extraction"
        )

    by_mode, curves_by_mode, comparison_scoped = {}, {}, {}
    randomised_by_mode = {}
    modes = list(node_weights.PARAMETER_WALK_MODES) + (
        list(node_weights.CONTROL_SET_MODES) if randomised_packed is not None
        else []
    )
    for mode in modes:
        restore_clean()
        stage_curves: dict[str, dict[int, float]] = {}
        randomised_regions = np.zeros_like(roi)
        if mode in node_weights.CONTROL_SET_MODES:
            # One endpoint, no walk: the representation is either the real
            # one or noise, and the criterion reads the endpoint.
            weights = phase8_held_out_region_weights(
                retained["models"], randomised_packed,
                patient_ids, retained["assignments"],
            )
            randomised_regions, _ = node_weights.split_whole_image(weights)
            stage_curves[f"{mode}:backbone_randomised_at_extraction"] = {
                row: gradcam.similarity(roi[row], randomised_regions[row])
                for row in range(len(patient_ids))
            }
        else:
            for fold, instance in retained["models"]:
                rows = held_rows(fold)
                for stage in phase8_node_weight_stages(instance.model, mode):
                    weights = instance.region_weights(
                        np.asarray(retained["packed"])[rows]
                    )
                    stage_region, _ = node_weights.split_whole_image(weights)
                    per_row = stage_curves.setdefault(stage, {})
                    for position, row in enumerate(rows):
                        per_row[row] = gradcam.similarity(
                            roi[row], stage_region[position]
                        )
                        randomised_regions[row] = stage_region[position]
        if not stage_curves:
            raise ValueError(f"mode {mode!r} produced no stages")
        final_stage = list(stage_curves)[-1]
        if len(stage_curves[final_stage]) != len(patient_ids):
            raise ValueError(
                f"{len(stage_curves[final_stage])} of {len(patient_ids)} "
                f"patients have a randomised vector at {final_stage!r}"
            )
        finals = np.asarray(
            [stage_curves[final_stage][row] for row in range(len(patient_ids))]
        )
        randomised_by_mode[mode] = randomised_regions
        # **The randomised vectors travel WITH the gate**, so a pass produced
        # by uniform vectors cannot be read without the fact that produced it
        # (node_weights.degeneracy, phase8.THE_RANDOMISED_BACKBONE_COLLAPSES).
        by_mode[mode] = node_weights.gate(
            roi, finals, randomised=randomised_regions
        )
        comparison_scoped[mode] = node_weights.gate(
            roi[shared_rows], finals[shared_rows],
            randomised=randomised_regions[shared_rows],
        )
        curves_by_mode[mode] = {
            stage: float(np.median(list(values.values())))
            for stage, values in stage_curves.items()
        }
        collapse = by_mode[mode]["degeneracy"]
        if collapse["any"]:
            ctx.log(
                f"  DEGENERATE {mode}: {collapse['n_constant']} of "
                f"{collapse['n_patients']} randomised vectors are UNIFORM "
                f"over the regions -- {collapse['reading']}"
            )
        own, restricted = by_mode[mode], comparison_scoped[mode]
        ctx.log(
            f"RANDOMISATION {mode}: baseline median "
            f"{own['baseline_median']:.4f} over {own['n_baseline_pairs']} "
            f"pairs; separability {own['separability']['observed_difference']:+.4f} "
            f"CI [{own['separability']['ci95'][0]:+.4f}, "
            f"{own['separability']['ci95'][1]:+.4f}] -- "
            f"{'RESOLVES' if own['resolves'] else 'UNRESOLVED'} on "
            f"{own['n_patients']}, "
            f"{'RESOLVES' if restricted['resolves'] else 'UNRESOLVED'} on the "
            f"shared {restricted['n_patients']}"
        )
        for stage, value in curves_by_mode[mode].items():
            ctx.log(f"    stage {stage}: median similarity {value:+.3f}")

    verdict = node_weights.publishable(by_mode)
    ctx.log(
        f"GATE: {'publishable' if verdict['publishable'] else 'UNRESOLVED'} "
        f"-- ran {verdict['modes_run']}, resolved {verdict['modes_resolved']}, "
        f"not run {verdict['modes_not_run']}"
    )

    summary = {
        "pre_registration": phase8.summary(),
        "arm": {key: phase8.ARMS["B"][key] for key in ("stem", "pcc", "geometry")},
        "refit": {"mean": band["mean"], "sd": band["sd"], "pooled": pooled},
        "n_patients": len(patient_ids),
        "n_regions_ranked": int(roi.shape[1]),
        "whole_image_node": {
            "excluded_from_the_similarity": True,
            "why": phase8.SRGNN_NODE_WEIGHTS_R10_READ[
                "the_count_is_26_plus_the_whole_image"
            ]["why_it_matters_twice"],
            "median_weight": float(np.median(whole_image)),
            "min_weight": float(np.min(whole_image)),
            "max_weight": float(np.max(whole_image)),
        },
        "seed_agreement": {
            "n_seeds": len(seeds),
            "median_of_medians": float(np.median(medians)),
            "worst_patient_median": float(np.min(medians)),
            "per_patient_median": {
                str(patient): float(np.median(values))
                for patient, values in agreement.items()
            },
        },
        "randomisation": {
            "own_gate": by_mode,
            "a_vs_b_scoped": comparison_scoped,
            "stage_curve_medians": curves_by_mode,
            "which_stages_can_move_it": phase8.WHICH_STAGES_CAN_MOVE_THE_EXPLANATION,
            "how_each_mode_randomises": {
                "explanation_source": "a parameter walk at cleft time",
                "component_role": (
                    "the feature map RE-EXTRACTED from a randomised backbone; "
                    "the model is untouched (phase8.COMPONENT_ROLE_IS_A_NO_OP)"
                ),
            },
            "control_set": control_report,
            "modes_not_run": [
                mode for mode in node_weights.RANDOMISATION_MODES
                if mode not in by_mode
            ],
        },
        "publication": verdict,
        # **Available or not, and NOT the same as unresolved.** The comparison
        # mode needs its control artifact; without one it was not tested, and
        # the explanation_source figures answer a different question and
        # cannot stand in for it.
        "a_vs_b": {
            "available": node_weights.COMPARISON_MODE in by_mode,
            "comparison_mode": node_weights.COMPARISON_MODE,
            "how": (
                "the control set re-extracts the map from a randomised "
                "backbone -- arm A's own intervention, moved to where arm B's "
                "representation is computed"
            ),
            "why_it_needs_one": phase8.COMPONENT_ROLE_IS_A_NO_OP["consequence"],
            "not_the_same_as_unresolved": (
                "a test with no instrument is not a test that failed; the "
                "explanation_source figures answer a different question from "
                "arm A's and may not be quoted for it"
            ),
            **(
                {}
                if node_weights.COMPARISON_MODE in by_mode
                else {
                    "what_would_restore_it": phase8.COMPONENT_ROLE_IS_A_NO_OP[
                        "what_would_restore_it"
                    ]
                }
            ),
        },
        "sample": {
            "own_gate": len(patient_ids),
            "shared_with_arm_a": [int(pid) for pid in shared],
            "drawn_by": "phase8_stratified_sample -- the same call arm A made",
        },
    }
    with ctx.atomic("metrics.json", tier="SHAREABLE") as tmp:
        tmp.write_text(
            json.dumps(as_builtin(summary), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    # The npz is what "published" scopes over, exactly as arm A's is
    # (phase8.PUBLICATION_SCOPE): written only if the gate resolves.
    if verdict["publishable"]:
        boxes = getattr(retained["models"][0][1].model, "default_boxes", None)
        npz = ctx.path("node_weights.npz", tier="CLUSTER-ONLY")
        np.savez_compressed(
            npz,
            patient_ids=np.asarray([int(pid) for pid in patient_ids], dtype=int),
            weights=roi,
            whole_image_weight=whole_image,
            seed_agreement_median=np.asarray(medians),
            region_boxes=(
                np.zeros((0, 4)) if boxes is None
                else np.asarray(boxes.detach().cpu())
            ),
            caveat=np.asarray(phase8.FROZEN_BACKBONE_CAVEAT["text"]),
        )
        ctx.log(f"  wrote {len(patient_ids)} node-weight vectors with the caveat")
    else:
        ctx.log(
            "  NOT writing node_weights.npz: no randomisation resolved, and "
            "the npz is what 'published' scopes over"
        )

    rendered = json.dumps(as_builtin(summary), indent=2, sort_keys=True)
    for line in rendered.splitlines():
        ctx.log(line)

    if not verdict["publishable"]:
        raise ValueError(
            "arm B's node weights do not separate from the between-patient "
            f"baseline under {verdict['modes_run']}. NOTHING IS PUBLISHED -- "
            "node_weights.npz is not written. metrics.json IS, because the "
            "gate's outcome is the result (phase8.PUBLICATION_SCOPE)"
        )



def task_tsne(ctx: RunContext) -> None:
    """Phase 8, the last original item: t-SNE at the two registered
    perplexities with the k-NN companion BESIDE it.

    A figure, never evidence -- no gate, no claim, no npz on the claim
    surface. The registration is split across ``phase8.TSNE`` (perplexities,
    seed, display rule) and ``phase8.TSNE_COMPANION`` (embeddings, classes,
    k, tie rule, bootstrap, baselines -- the 2026-08-15 resolution), and
    everything numeric here is read from those records rather than from the
    config: a registered detail in a config field is a knob, and the layer
    rule exists because knobs get nudged.
    """
    import json

    from . import phase8, phase8c
    from .data.manifest import load_manifest
    from .geometry import render

    task = ctx.config["task"]
    declared = {entry["name"]: entry for entry in ctx.inputs}
    spec = phase8.TSNE_COMPANION

    embeddings_dir = Path(declared[task["embeddings_artifact"]]["path"])
    if embeddings_dir.name != spec["embeddings_set_name"]:
        raise ValueError(
            f"the registration names {spec['embeddings_set_name']!r} "
            f"(phase8.TSNE_COMPANION) but the declared embeddings input is "
            f"{embeddings_dir.name!r}. The subject of the figure is a "
            "registered detail, not a config choice."
        )

    manifest_dir = Path(declared[task["manifest_artifact"]]["path"])
    rows = load_manifest(manifest_dir / "manifest.csv")
    patient_ids = [int(row["patient_id"]) for row in rows]
    classes = np.array([int(row[spec["classes"]]) for row in rows])
    values, counts = np.unique(classes, return_counts=True)
    if tuple(int(c) for c in counts) != tuple(spec["expected_class_counts"]):
        raise ValueError(
            f"class counts {tuple(int(c) for c in counts)} do not match the "
            f"committed {spec['expected_class_counts']} "
            "(phase8.TSNE_COMPANION): the printed baselines would describe "
            "a different cohort than the points."
        )

    features, _ = embeddings_module_load(embeddings_dir, patient_ids)
    ctx.log(
        f"t-SNE subject: {embeddings_dir.name}, {features.shape[0]} patients "
        f"x {features.shape[1]} dims; companion k={spec['k']} on "
        f"{spec['classes']}"
    )

    # The companion FIRST -- the figure cannot be composed without it.
    accuracy, correct, _ = phase8.knn_loo_accuracy(
        features, classes, spec["k"]
    )
    interval = phase8.bootstrap_interval(
        correct, n_boot=spec["n_boot"], seed=spec["bootstrap_seed"]
    )
    chance = 1.0 / len(values)
    majority = float(counts.max()) / float(counts.sum())
    companion = {
        "k": spec["k"],
        "accuracy": accuracy,
        "interval": interval,
        "chance": chance,
        "majority": majority,
        "class_counts": tuple(int(c) for c in counts),
        "n_boot": spec["n_boot"],
    }
    ctx.log(
        f"companion: accuracy {accuracy:.3f} "
        f"[{interval[0]:.3f}, {interval[1]:.3f}], chance {chance:.3f}, "
        f"majority {majority:.3f}"
    )

    registration = phase8.TSNE
    embedded = {}
    for perplexity in registration["perplexities"]:
        embedded[perplexity] = phase8.tsne_embed(
            features, perplexity=perplexity, seed=registration["seed"]
        )
        ctx.log(f"  perplexity {perplexity}: embedded")

    figure = phase8c.tsne_figure(embedded, classes, companion)
    render.save_sheet(figure, ctx.path("tsne_figure.png", tier="CLUSTER-ONLY"))
    np.savez(
        ctx.path("tsne.npz", tier="CLUSTER-ONLY"),
        classes=classes,
        patient_ids=np.array(patient_ids),
        correct=correct,
        **{
            f"perplexity_{p}": xy for p, xy in embedded.items()
        },
    )

    summary = {
        "registration": registration,
        "companion_spec": spec,
        "companion": {
            "accuracy": accuracy,
            "interval_95": list(interval),
            "chance": chance,
            "majority": majority,
            "class_counts": [int(c) for c in counts],
        },
        "is_evidence": False,
    }
    with ctx.atomic("metrics.json", tier="SHAREABLE") as tmp:
        tmp.write_text(
            json.dumps(as_builtin(summary), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


class _PretrainGradCamAdapter:
    """The shape ``gradcam.token_activations_and_gradients`` expects, over a
    live ``TorchPretrainModel``: ``.model`` is the timm module and
    ``.preprocess`` builds the graph-capable batch. Reaching into the
    model's private fields is deliberate and local -- a public re-export
    would be a second definition of the training model's boundary."""

    def __init__(self, pretrain_model):
        self._pretrain_model = pretrain_model

    @property
    def model(self):
        return self._pretrain_model._model

    def preprocess(self, images):
        return self._pretrain_model._prepare(images)


def phase8_scut_face_sample(labels_by_stem: dict) -> dict:
    """The registered face sample (``phase8.SCUT_ANIMATION_FACES``): sorted
    stems, score-ordered, five equal-count strata, two seeded picks each.
    Reads only (stems, scores, seed), so both variants get the SAME faces."""
    from . import phase8

    rule = phase8.SCUT_ANIMATION_FACES
    stems = sorted(labels_by_stem)
    scores = np.array([float(labels_by_stem[stem]) for stem in stems])
    order = np.argsort(scores, kind="stable")
    strata = np.array_split(order, rule["n_strata"])
    rng = np.random.default_rng(rule["seed"])
    chosen: list[int] = []
    for stratum in strata:
        picks = rng.choice(len(stratum), size=rule["per_stratum"], replace=False)
        chosen.extend(int(stratum[p]) for p in sorted(picks))
    return {
        "stems": [stems[i] for i in chosen],
        "scores": [float(scores[i]) for i in chosen],
    }


def phase8_scut_capture_cell(face, weights, compute_map, label: str):
    """One face at one training state: (grid, frame image, status).

    **[2026-08-16, after the first launch crashed on it] The 8b per-cell
    refusal lesson, carried into the capture path
    (phase8.SCUT_ANIMATION_REFUSAL_HANDLING).** A zero head is the
    registered init state (status ``"zero_head"``, the epoch-0 photograph).
    The METHOD'S OWN refusal -- ``GradCamError`` out of the map chain, an
    all-zero map or zero gradients -- is a DATUM (status ``"all_zero"``):
    the frame is the photograph with the measured-refusal caption and the
    run continues. Anything else propagates: config errors fire in
    ``token_activations_and_gradients``, OUTSIDE ``compute_map``, so this
    except cannot swallow a wrong-block fault as a refusal.
    """
    from . import gradcam, phase8c

    if not np.any(weights):
        return None, phase8c.frame(face, None, axis_label=label), "zero_head"
    try:
        grid = compute_map()
    except gradcam.GradCamError:
        return None, phase8c.frame(
            face, None,
            axis_label=label + " -- no map: every cell zero after the ReLU "
            "(a measured refusal, recorded as a datum)",
        ), "all_zero"
    return grid, phase8c.frame(face, grid, axis_label=label), None


def task_scut_animation(ctx: RunContext) -> None:
    """Phase 8's final addition: the training animation on SCUT pretraining,
    one variant per run -- Grad-CAM evolving as the backbone itself learns
    beauty (``phase8.SCUT_ANIMATION_BUILT``).

    A full rerun of the declared Phase 6 cell through ``run_pretraining``'s
    epoch-end observation hook, frames rendered IN-RUN. Epoch 0 is the
    photograph (the head is zero-initialised -- no map exists yet); the
    terminal frame is rendered from the SHIPPED cell's selected weights,
    the declared deliverable. The self-gate is the CORRECTED one: byte
    reproduction of a ViT pretraining run does not exist
    (roadb.VIT_PRETRAINING_IS_NONDETERMINISTIC), so fingerprint mismatch
    refuses, an out-of-band test-PCC delta raises LAST, and per-array byte
    identity is reported, expected False. SHAREABLE throughout -- SCUT is
    public and the mask derives from aggregate cohort statistics.
    """
    import json

    from . import gradcam, phase8, phase8c
    from .geometry import render
    from .train import pretrain

    task = ctx.config["task"]
    declared = {entry["name"]: Path(entry["path"]) for entry in ctx.inputs}
    source = task["source"]

    root = declared.get(task["scut_root"])
    if root is None:
        raise ValueError(
            f"task.scut_root names input {task['scut_root']!r}, which is not "
            f"declared in inputs: {sorted(declared)}"
        )
    masked_dir = declared.get(pretrain.MASKED_INPUT_NAME)
    if source == "original" and masked_dir is not None:
        raise ValueError(
            f"source 'original' does not read {pretrain.MASKED_INPUT_NAME!r}, "
            "but the config declares it -- an input the run never reads would "
            "still enter inputs.json as if it fed the run (the task_pretrain "
            "rule, unchanged here)."
        )
    if source != "original" and masked_dir is None:
        raise ValueError(
            f"source {source!r} reads the masked SCUT artifact; declare "
            f"{pretrain.MASKED_INPUT_NAME!r}."
        )
    init_dir = declared.get(pretrain.INIT_INPUT_NAME)
    if init_dir is None:
        raise ValueError(
            f"this rerun declares its init ({pretrain.INIT_INPUT_NAME!r}, "
            "data/inits/init_vit_b16_v1): an undeclared download would put "
            "the one input guard 3 cannot verify back into the record."
        )
    shipped_path = declared.get(task["shipped_artifact"])
    if shipped_path is None:
        raise ValueError(
            f"task.shipped_artifact names input {task['shipped_artifact']!r}, "
            f"which is not declared in inputs: {sorted(declared)}"
        )

    # The shipped cell FIRST: a wrong artifact must refuse before thirty
    # epochs of rerun, not after.
    shipped = pretrain.ckpt.load(shipped_path)
    shipped_print = shipped.extra["fingerprint"]
    for key, expected in (
        ("backbone", task["backbone"]), ("source", source),
        ("region_scheme", task["region_scheme"]),
        ("seed", ctx.config["seed"]), ("monitor", task["monitor"]),
        ("n_test", task["expect_test"]),
    ):
        if shipped_print.get(key) != expected:
            raise ValueError(
                f"the shipped checkpoint's fingerprint says "
                f"{key}={shipped_print.get(key)!r} but this run declares "
                f"{expected!r}: this is not the cell the animation claims "
                "to retrace."
            )

    # The registered faces, identical in both variants by rule.
    _, test_labels_by_stem = pretrain.read_split_or_refuse(
        Path(root), expect_train=task["expect_train"],
        expect_test=task["expect_test"],
    )
    sample = phase8_scut_face_sample(test_labels_by_stem)
    ctx.log(
        f"faces by rule ({phase8.SCUT_ANIMATION_FACES['rule']}): "
        + ", ".join(
            f"{stem} ({score:.2f})"
            for stem, score in zip(sample["stems"], sample["scores"])
        )
    )
    faces, _ = pretrain.load_features(
        source=source, scut_root=Path(root), masked_dir=masked_dir,
        stems=sample["stems"], with_boxes=False, log=ctx.log,
    )
    faces = np.asarray(faces, dtype=np.uint8)

    budget = int(task["epochs"])

    refusal_ledger: dict = {}

    def capture_frames(model, tag: str, label: str) -> dict:
        """Observation only: torch RNG fenced, model handed back as taken.
        Returns each face's raw map grid (None where no map exists) so the
        endpoint comparison's caption can carry a MEASURED difference
        (phase8.SCUT_ANIMATION_CAPTION_RULE). A cell where the method
        refuses is a DATUM in the ledger, never a fatality
        (phase8.SCUT_ANIMATION_REFUSAL_HANDLING -- the first launch died
        here)."""
        import torch

        rng_state = torch.get_rng_state()
        cuda_states = (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        )
        grids: dict = {}
        try:
            adapter = _PretrainGradCamAdapter(model)
            adapter.model.eval()
            head = model._classifier()
            weights = head.weight.detach().cpu().numpy()[0]
            for position, stem in enumerate(sample["stems"]):
                if np.any(weights):
                    # Config errors (wrong block, no gradient reached) fire
                    # HERE, outside the refusal catch, and stay fatal.
                    acts, grads = gradcam.token_activations_and_gradients(
                        adapter, faces[position:position + 1], weights
                    )
                    compute = (
                        lambda a=acts, g=grads: gradcam.map_for(a[0], g[0])["grid"]
                    )
                else:
                    compute = None
                grid, image, status = phase8_scut_capture_cell(
                    faces[position], weights, compute, label
                )
                grids[stem] = grid
                if status == "all_zero":
                    refusal_ledger.setdefault(tag, []).append(stem)
                render.save_sheet(image, ctx.path(
                    f"scut_anim_{stem}_{tag}.png", tier="SHAREABLE"
                ))
        finally:
            torch.set_rng_state(rng_state)
            if cuda_states is not None:
                torch.cuda.set_rng_state_all(cuda_states)
        return grids

    captured_grids: dict = {}

    def on_epoch(epoch, model):
        label = (
            f"epoch 0 of {budget}: ImageNet-initialised backbone -- the "
            "beauty head starts at zero, so no map exists yet"
            if epoch == 0
            else f"epoch {epoch} of {budget}"
        )
        grids = capture_frames(model, f"e{epoch:02d}", label)
        if epoch == budget:
            # The final epoch always runs in the completing invocation --
            # the resume checkpoint stops at budget-1 -- so these grids
            # exist whenever the comparison below does.
            captured_grids["final"] = grids
        ctx.log(f"  frames captured at epoch {epoch}")

    made: dict = {}

    def factory():
        made["model"] = pretrain.make_model(task["backbone"], config, init_dir)
        return made["model"]

    config = pretrain.PretrainConfig(
        epochs=budget,
        inner_val_frac=task["inner_val_frac"],
        seed=ctx.config["seed"],
        deterministic=task["deterministic"],
        monitor=task["monitor"],
        batch_size=task["batch_size"],
        learning_rate=task["learning_rate"],
        weight_decay=task["weight_decay"],
        checkpoint_every=task["checkpoint_every"],
    )
    # ctx.path CLAIMS an output name, once per run: claimed HERE and reused
    # below for the self-gate's read-back. The first masked-G1 launch called
    # it a second time to load and died on the duplicate-claim refusal
    # (phase8.SCUT_ANIMATION_FIRST_LAUNCH, defect 2).
    pretrained_path = ctx.path(pretrain.PRETRAINED_NAME, tier="SHAREABLE")
    result = pretrain.run_pretraining(
        scut_root=Path(root),
        source=source,
        backbone=task["backbone"],
        region_scheme=task["region_scheme"],
        masked_dir=masked_dir,
        expect_train=task["expect_train"],
        expect_test=task["expect_test"],
        run_dir=ctx.run_dir,
        curves_path=ctx.path("curves.csv", tier="SHAREABLE"),
        pretrained_path=pretrained_path,
        config=config,
        model_factory=factory,
        init_dir=init_dir,
        log=ctx.log,
        on_epoch=on_epoch,
    )
    if result.paused:
        return

    # ---- the corrected self-gate (phase8.SCUT_ANIMATION_BUILT) ------------
    fresh = pretrain.ckpt.load(pretrained_path)
    if fresh.extra["fingerprint"] != shipped_print:
        raise ValueError(
            f"rerun fingerprint {fresh.extra['fingerprint']} does not equal "
            f"the shipped {shipped_print}: the rerun did not retrace the "
            "declared cell."
        )
    byte_identical = set(fresh.arrays) == set(shipped.arrays) and all(
        np.array_equal(fresh.arrays[key], shipped.arrays[key])
        for key in shipped.arrays
    )
    gate = phase8.SCUT_ANIMATION_BUILT["self_gate"]
    delta = abs(float(fresh.extra["test_pcc"]) - float(shipped.extra["test_pcc"]))
    verdict = "WITHIN_RECORDED_BAND" if delta <= gate["band"] else "OUT_OF_BAND"
    ctx.log(
        f"self-gate: shipped {shipped.extra['test_pcc']:.4f} vs rerun "
        f"{fresh.extra['test_pcc']:.4f} (delta {delta:.4f}, band "
        f"{gate['band']}), byte_identical={byte_identical} -- {verdict}"
    )

    # ---- the endpoint frame: the SHIPPED cell's selected weights ----------
    model = made["model"]
    model.load_state_arrays(shipped.arrays)
    captured_grids["shipped"] = capture_frames(
        model, "endpoint",
        f"the shipped Phase 6 cell's selected weights (epoch "
        f"{shipped.extra['selected_epoch']}, test PCC "
        f"{shipped.extra['test_pcc']:.4f}) -- the declared deliverable",
    )

    # ---- APNGs + the endpoint comparison (the registered caption rule) ----
    from PIL import Image

    rule = phase8.SCUT_ANIMATION_CAPTION_RULE
    comparisons: dict = {}
    out_of_mechanism: list = []
    for stem in sample["stems"]:
        frames = []
        for epoch in range(0, budget + 1):
            # READ of an output this run already claimed when it wrote the
            # frame (a resumed attempt's earlier frames included) -- claiming
            # again is the duplicate-claim refusal the first launch died on.
            path = ctx.run_dir / f"scut_anim_{stem}_e{epoch:02d}.png"
            if not path.is_file():
                raise ValueError(
                    f"frame missing for {stem} at epoch {epoch}: the axis "
                    "has a hole, and an animation over a gapped axis would "
                    "show a trajectory nothing measured."
                )
            frames.append(np.asarray(Image.open(path).convert("RGB")))
        endpoint_image = np.asarray(Image.open(
            ctx.run_dir / f"scut_anim_{stem}_endpoint.png"
        ).convert("RGB"))
        frames.append(endpoint_image)
        # The recipe sentence travels INSIDE the animation; the per-frame
        # PNGs stay the measured objects (SCUT_ANIMATION_CAPTION_RULE).
        phase8c.write_animation(
            phase8c.captioned_frames(frames, rule["recipe_sentence"]),
            ctx.path(f"scut_anim_{stem}.png", tier="SHAREABLE"),
        )

        # The comparison completes WITH its data either way: a side without
        # a map is captioned as the recorded refusal, and a refusal at the
        # final epoch or on the shipped weights is OUT OF the registered
        # early-epoch mechanism -- flagged for the raise-last below, never
        # a mid-assembly crash (SCUT_ANIMATION_REFUSAL_HANDLING boundary).
        final_grid = captured_grids["final"][stem]
        shipped_grid = captured_grids["shipped"][stem]
        if final_grid is None or shipped_grid is None:
            sides = [
                name for name, grid in (
                    ("final epoch", final_grid), ("shipped weights", shipped_grid),
                ) if grid is None
            ]
            out_of_mechanism.append(f"{stem}: no map on {' and '.join(sides)}")
            comparisons[stem] = None
            second_line = (
                f"no map existed on the {' and the '.join(sides)} side -- a "
                "measured refusal, recorded as a datum; no difference is "
                "quotable for this face"
            )
        else:
            difference = float(np.max(np.abs(
                np.asarray(final_grid) - np.asarray(shipped_grid)
            )))
            comparisons[stem] = difference
            second_line = (
                f"largest per-cell map difference on this face: "
                f"{difference:.3f} (grid scale 0..1)"
            )
        panel = phase8c.endpoint_comparison_panel(
            frames[budget], endpoint_image,
            [rule["comparison_sentence"], second_line],
        )
        render.save_sheet(panel, ctx.path(
            f"scut_anim_{stem}_endpoint_comparison.png", tier="SHAREABLE"
        ))
    refused_cells = sum(len(stems) for stems in refusal_ledger.values())
    ctx.log(f"animations assembled: {len(sample['stems'])} faces, "
            f"{budget + 2} frames each, captioned; endpoint comparisons "
            f"written; {refused_cells} refused cell(s) in the ledger")

    summary = {
        "registration": phase8.SCUT_ANIMATION_BUILT,
        "faces": sample,
        "source": source,
        "frames_per_face": budget + 2,
        "self_gate": {
            "shipped_test_pcc": float(shipped.extra["test_pcc"]),
            "rerun_test_pcc": float(fresh.extra["test_pcc"]),
            "delta": delta,
            "band": gate["band"],
            "verdict": verdict,
            "byte_identical": bool(byte_identical),
            "byte_identity_expectation": gate["byte_identity"],
        },
        "shipped_selected_epoch": int(shipped.extra["selected_epoch"]),
        "rerun_summary": result.summary,
        "governance": phase8.SCUT_ANIMATION_AUDIT["governance"],
        "caption_rule": rule,
        "endpoint_comparison": {
            "largest_per_cell_map_difference": comparisons,
            "basis": (
                "the recorded run-to-run nondeterminism made visual "
                "(SCUT_ANIMATION_CAPTION_RULE) -- shown, never smoothed"
            ),
        },
        # The 8b-shape ledger: which cells (face x state) the method
        # refused, with the registered mechanism beside the counts
        # (SCUT_ANIMATION_REFUSAL_HANDLING).
        "refusals": {
            "ledger": refusal_ledger,
            "refused_cells": refused_cells,
            "total_cells": len(sample["stems"]) * (budget + 2),
            "mechanism": phase8.SCUT_ANIMATION_REFUSAL_HANDLING["mechanism"],
            "out_of_mechanism": out_of_mechanism,
        },
    }
    with ctx.atomic("metrics.json", tier="SHAREABLE") as tmp:
        tmp.write_text(
            json.dumps(as_builtin(summary), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    # Raise LAST, after every artifact is written -- the arm-A precedent.
    # Both conditions are faults to diagnose, not registered completions:
    # out-of-band means the rerun is not behaving as a draw of the recorded
    # cell; an out-of-mechanism refusal means TRAINED weights refused to
    # map, which the registered early-epoch mechanism does not cover.
    late_faults = []
    if out_of_mechanism:
        late_faults.append(
            "refusals OUTSIDE the registered early-epoch mechanism: "
            + "; ".join(out_of_mechanism)
        )
    if verdict == "OUT_OF_BAND":
        late_faults.append(
            f"self-gate OUT_OF_BAND: |{fresh.extra['test_pcc']:.4f} - "
            f"{shipped.extra['test_pcc']:.4f}| = {delta:.4f} > "
            f"{gate['band']} (twice the recorded three-run spread)"
        )
    if late_faults:
        raise ValueError(
            " | ".join(late_faults)
            + ". Artifacts and metrics are on disk; the run needs diagnosis "
            "before its animation is shown to anyone."
        )



def task_prototypes(ctx: RunContext) -> None:
    """Phase 9: the prototypes -- medoids per grade in arm A's embedding
    space, validated exactly as registered (phase9.PROTOTYPES_REGISTERED).

    Primary: three class3 medoids, leave-one-out identity stability with
    the nearest-medoid accuracy companion (majority baseline printed),
    bootstrap identity persistence (2,000 resamples, seed 1337).
    Secondary: the five-grade (median) medoids, DESCRIPTIVE -- computed
    and shown, never validated. Medoids are patient faces: the sheet and
    the id-bearing json are CLUSTER-ONLY; metrics.json carries numbers
    only. The measured prior travels in the record: the t-SNE companion
    (0.409 vs majority 0.502) predicts weak neighbourhood structure, so
    unstable medoids corroborate rather than surprise -- both readings
    were committed before this task existed.
    """
    import json

    from . import phase8c, phase9
    from .data.manifest import load_manifest
    from .geometry import render
    from .train import phase3

    task = ctx.config["task"]
    declared = {entry["name"]: entry for entry in ctx.inputs}
    spec = phase9.PROTOTYPES_REGISTERED

    embeddings_dir = Path(declared[task["embeddings_artifact"]]["path"])
    if embeddings_dir.name != spec["space"]["set_name"]:
        raise ValueError(
            f"the registration names {spec['space']['set_name']!r} "
            f"(phase9.PROTOTYPES_REGISTERED) but the declared embeddings "
            f"input is {embeddings_dir.name!r}. The space is a registered "
            "detail, not a config choice."
        )

    manifest_dir = Path(declared[task["manifest_artifact"]]["path"])
    staged_dir = Path(declared[task["staged_artifact"]]["path"])
    rows = load_manifest(manifest_dir / "manifest.csv")
    patient_ids = np.array([int(row["patient_id"]) for row in rows])
    primary = np.array([int(row[spec["primary"]["column"]]) for row in rows])
    secondary = np.array([
        int(float(row[spec["secondary"]["column"]])) for row in rows
    ])
    features, _ = embeddings_module_load(embeddings_dir, [int(p) for p in patient_ids])
    features = np.asarray(features, dtype=np.float64)

    # ---- the primary: three class3 medoids, fully validated ---------------
    values = sorted(int(v) for v in np.unique(primary))
    primary_report: dict = {}
    for value in values:
        member_rows = np.flatnonzero(primary == value)
        stability = phase9.loo_identity_stability(
            features[member_rows], patient_ids[member_rows]
        )
        persistence = phase9.bootstrap_persistence(
            features[member_rows], patient_ids[member_rows],
            n_boot=spec["bootstrap"]["n_boot"], seed=spec["bootstrap"]["seed"],
        )
        if stability["medoid_id"] != persistence["medoid_id"]:
            raise ValueError(
                f"class {value}: the two validations name different "
                f"medoids ({stability['medoid_id']} vs "
                f"{persistence['medoid_id']}) -- one deterministic "
                "computation produced two answers, which is a fault, not "
                "a statistic"
            )
        primary_report[value] = {
            "medoid_patient_id": stability["medoid_id"],
            "n_members": stability["n"],
            "loo_identity_stability": stability["stability"],
            "loo_ceiling": stability["ceiling"],
            "bootstrap_persistence": persistence["persistence"],
        }
        ctx.log(
            f"class3={value}: medoid patient {stability['medoid_id']} "
            f"(n={stability['n']}), LOO stability "
            f"{stability['stability']:.3f} (ceiling "
            f"{stability['ceiling']:.3f}), bootstrap persistence "
            f"{persistence['persistence']:.3f}"
        )

    companion = phase9.nearest_medoid_accuracy(features, primary, patient_ids)
    ctx.log(
        f"nearest-medoid companion: accuracy {companion['accuracy']:.3f} "
        f"vs majority {companion['majority']:.3f}, chance "
        f"{companion['chance']:.3f}"
    )

    # ---- the secondary: five-grade (median) medoids, DESCRIPTIVE ----------
    secondary_report: dict = {}
    for value in range(1, 6):
        member_rows = np.flatnonzero(secondary == value)
        if not len(member_rows):
            secondary_report[value] = {"absent": True}
            continue
        medoid_row = member_rows[phase9.medoid_index(
            features[member_rows], patient_ids[member_rows]
        )]
        secondary_report[value] = {
            "medoid_patient_id": int(patient_ids[medoid_row]),
            "n_members": int(len(member_rows)),
            "flagged_single_member": bool(len(member_rows) == 1),
        }

    # ---- the sheet: medoid faces, CLUSTER-ONLY ----------------------------
    images, _, image_ids, _ = phase3.load_inputs(
        manifest_dir, staged_dir, task["geometry"], task["label"]
    )
    row_of = {int(p): i for i, p in enumerate(image_ids)}
    panels = []
    for value in values:
        entry = primary_report[value]
        face = np.asarray(images[row_of[entry["medoid_patient_id"]]], dtype=np.uint8)
        panels.append(phase8c.frame(
            face, None,
            axis_label=(
                f"class3={value} medoid -- patient "
                f"{entry['medoid_patient_id']} (n={entry['n_members']}, "
                f"LOO stability {entry['loo_identity_stability']:.2f}, "
                f"bootstrap {entry['bootstrap_persistence']:.2f})"
            ),
        ))
    sheet = phase8c.stack_panels([
        phase8c.side_by_side(panels),
        phase8c.label_strip(
            phase8c.side_by_side(panels).shape[1],
            [
                (
                    f"nearest-medoid companion: accuracy "
                    f"{companion['accuracy']:.3f}; baselines: majority "
                    f"{companion['majority']:.3f}, chance "
                    f"{companion['chance']:.3f} -- the number is unreadable "
                    "without this bar"
                ),
                (
                    "registered prior: the space's neighbourhoods predict "
                    "class3 worse than majority guessing (t-SNE companion "
                    "0.409 vs 0.502); unstable medoids corroborate, they "
                    "do not surprise"
                ),
            ],
        ),
    ], gap=4)
    render.save_sheet(sheet, ctx.path("prototypes_sheet.png", tier="CLUSTER-ONLY"))

    with ctx.atomic("prototypes.json", tier="CLUSTER-ONLY") as tmp:
        tmp.write_text(json.dumps(as_builtin({
            "primary": primary_report,
            "secondary": secondary_report,
            "companion": companion,
        }), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    summary = {
        "registration": spec,
        "primary_validation": {
            str(value): {
                "n_members": primary_report[value]["n_members"],
                "loo_identity_stability": primary_report[value][
                    "loo_identity_stability"
                ],
                "loo_ceiling": primary_report[value]["loo_ceiling"],
                "bootstrap_persistence": primary_report[value][
                    "bootstrap_persistence"
                ],
            }
            for value in values
        },
        "companion": companion,
        "secondary_is_descriptive": spec["secondary"]["status"],
        "note": "patient ids live in the CLUSTER-ONLY json, not here",
    }
    with ctx.atomic("metrics.json", tier="SHAREABLE") as tmp:
        tmp.write_text(
            json.dumps(as_builtin(summary), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )



def task_deall_reference(ctx: RunContext) -> None:
    """Phase 9: the 25-image EXTERNAL REFERENCE -- the frozen best arm
    scored on the Deall/CleftGNN benchmark set, never pooled with cohort
    results (phase9.DEALL_REFERENCE_REGISTERED; the reads that unblocked
    it in phase9.DEALL_REFERENCE_READS).

    Generalisation to a DIFFERENT label (the survey-design key's
    five-grade Score, not the cohort's panel mean), reported as external
    reference only -- and the first directly-placed number beside the
    group's published CleftGNN Table 5, on the same 25 images. The heads
    are arm A's, reproduced through the 0.2520 gate and then FROZEN: no
    refitting touches the 25, and all 25 heads (5 seeds x 5 folds) score
    every image because an external image belongs to no fold.
    """
    import json

    from . import phase8, phase8c, phase9
    from .data.manifest import load_manifest
    from .eval import metrics
    from .geometry import render, staging
    from .train import phase3
    from .train.torch_backbone import FrozenExtractor

    task = ctx.config["task"]
    declared = {entry["name"]: entry for entry in ctx.inputs}
    deall_dir = Path(declared[task["deall_artifact"]]["path"])
    embeddings_dir = Path(declared[task["embeddings_artifact"]]["path"])
    manifest_dir = Path(declared[task["manifest_artifact"]]["path"])
    staged_dir = Path(declared[task["staged_artifact"]]["path"])

    # ---- the 25-set first: a wrong folder refuses BEFORE the refit --------
    views = phase9.deall_partition_stems(
        [child.name for child in deall_dir.iterdir()]
    )
    if len(views["composite"]) != 25:
        raise ValueError(
            f"{len(views['composite'])} composite images in {deall_dir.name} "
            "where the registration fixes 25 "
            "(phase9.DEALL_REFERENCE_REGISTERED)"
        )
    labels_by_id = phase9.deall_read_labels(
        deall_dir / phase9.DEALL_LABELS_FILENAME
    )
    stems = [Path(name).stem for name in views["composite"]]
    mismatch = sorted(set(stems) ^ set(labels_by_id))
    if mismatch:
        raise ValueError(
            f"composites and labels disagree on {mismatch}: every "
            "composite must carry exactly one Score"
        )
    ctx.log(
        f"25-set verified: 25 composites, labels matched; "
        f"{len(views['lips'])} -lips and {len(views['nose'])} -nose views "
        "recorded present, not staged (the registration)"
    )

    # ---- arm A's heads, reproduced through the gate and FROZEN ------------
    images, labels, patient_ids, _ = phase3.load_inputs(
        manifest_dir, staged_dir, task["geometry"], task["label"]
    )
    rows = load_manifest(manifest_dir / "manifest.csv")
    assignments = {int(r["patient_id"]): int(r["fold"]) for r in rows}
    features, _ = embeddings_module_load(embeddings_dir, patient_ids)
    seeds = [int(s) for s in task["seeds"]]
    captured: dict = {}

    def on_head(seed, backbone):
        captured.setdefault(seed, []).append(backbone)

    _, pooled = phase8_refit_heads(
        features, labels, patient_ids, assignments, seeds, task, ctx.log,
        on_head=on_head,
    )
    band = phase3.seed_variance(pooled)
    gate = phase8.REPRODUCE_GATE
    if abs(band["mean"] - gate["expected_pcc"]) > gate["tolerance"]:
        raise ValueError(
            f"the refit scores {band['mean']:.4f} against arm A's recorded "
            f"{gate['expected_pcc']}; these are not arm A's heads and the "
            "external reference would describe a different model"
        )
    backbones = [b for seed in seeds for b in captured[seed]]
    n_folds = len({assignments[int(p)] for p in patient_ids})
    if len(backbones) != len(seeds) * n_folds:
        raise ValueError(
            f"{len(backbones)} heads captured for {len(seeds)} seeds x "
            f"{n_folds} folds; the capture is misaligned"
        )

    # ---- stage and extract the composites ---------------------------------
    # G1's whole-image path is the PLAIN staged square (stage_build takes
    # base.image for g1; the trapezium is patch geometry only) -- the READS
    # record's staging answer, applied.
    staged_faces = np.stack([
        staging.stage(render.load_image(deall_dir / name)).image
        for name in views["composite"]
    ])
    extractor = FrozenExtractor(task["backbone"], batch_size=task["batch_size"])
    deall_features = extractor(staged_faces)

    # Live-path parity, REPORTED: one cohort patient extracted through this
    # run's own path against their artifact row. A gross mismatch means the
    # 25 were embedded by a different boundary than the heads were fit on.
    parity = float(np.max(np.abs(
        extractor(np.asarray(images[:1]))[0]
        - np.asarray(features[0], dtype=np.float32)
    )))
    ctx.log(f"live-path parity (patient {int(patient_ids[0])}): "
            f"max|delta| {parity:.2e}")
    if parity > 1e-2:
        raise ValueError(
            f"live-path parity {parity:.2e} exceeds the gross-mismatch "
            "guard 1e-2: the extraction boundary does not reproduce the "
            "artifact features, so the 25-set's embeddings would not live "
            "in the heads' space"
        )

    # ---- score: all frozen heads on every image, mean and spread ----------
    per_head = np.stack([b.predict(deall_features) for b in backbones])
    means = per_head.mean(axis=0)
    spreads = per_head.std(axis=0)
    truth = np.array([float(labels_by_id[stem]) for stem in stems])
    pcc = metrics.pcc(truth, means)
    spearman = metrics.spearman(truth, means)
    ctx.log(
        f"external reference (n=25, different label): PCC {pcc:.4f}, "
        f"Spearman {spearman:.4f}; per-image head spread mean "
        f"{float(spreads.mean()):.4f}"
    )

    # ---- artifacts --------------------------------------------------------
    # The staged composites as one sheet -- documents exactly what the
    # backbone saw, and doubles as the eyeball criterion's record.
    frames = [
        phase8c.frame(
            staged_faces[position], None,
            axis_label=f"{stems[position]} -- Score {labels_by_id[stems[position]]}",
        )
        for position in range(len(stems))
    ]
    sheet = phase8c.stack_panels([
        phase8c.side_by_side(frames[start:start + 5])
        for start in range(0, len(frames), 5)
    ], gap=4)
    render.save_sheet(sheet, ctx.path("deall_staged_sheet.png", tier="CLUSTER-ONLY"))

    with ctx.atomic("deall_scores.csv", tier="CLUSTER-ONLY") as tmp:
        lines = ["photo_id,score,head_mean,head_sd"]
        lines += [
            f"{stems[i]},{labels_by_id[stems[i]]},{means[i]!r},{spreads[i]!r}"
            for i in range(len(stems))
        ]
        tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")

    summary = {
        "registration": phase9.DEALL_REFERENCE_REGISTERED["reading_registered"],
        "label_provenance": phase9.DEALL_REFERENCE_READS["score"],
        "external_reference": {
            "n": 25,
            "pcc": pcc,
            "spearman": spearman,
            "per_image_head_spread_mean": float(spreads.mean()),
            "never_pooled": (
                "external reference only; a different label (survey-design "
                "key), never pooled with cohort results"
            ),
            "beside": "CleftGNN Table 5 -- the same 25 images",
        },
        "refit_gate": {"pooled_mean": band["mean"], "sd": band["sd"]},
        "live_path_parity_max_abs": parity,
        "other_views": "-lips/-nose present and available, not staged",
    }
    with ctx.atomic("metrics.json", tier="SHAREABLE") as tmp:
        tmp.write_text(
            json.dumps(as_builtin(summary), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )



def task_prototype_classifier(ctx: RunContext) -> None:
    """Phase 9: the prototype classifier -- the actual reading of
    "prototypes" (phase9.PROTOTYPE_CLASSIFIER_REGISTERED).

    The 25 anchors' grades reach each cohort patient through raw feature
    distance: 4 cells per metric (k in {1,3} x plain/weighted), Euclidean
    primary and cosine secondary, ALL cells reported, none selected after
    the fact. NO training anywhere. The pre-committed prediction travels
    in the record: three convergent priors say near-or-below chance; if
    the classifier beats them, that matters more.
    """
    import json

    from . import phase9
    from .data.manifest import load_manifest
    from .eval import metrics
    from .geometry import render, staging
    from .train import phase3
    from .train.torch_backbone import FrozenExtractor

    task = ctx.config["task"]
    declared = {entry["name"]: entry for entry in ctx.inputs}
    deall_dir = Path(declared[task["deall_artifact"]]["path"])
    embeddings_dir = Path(declared[task["embeddings_artifact"]]["path"])
    manifest_dir = Path(declared[task["manifest_artifact"]]["path"])
    staged_dir = Path(declared[task["staged_artifact"]]["path"])
    spec = phase9.PROTOTYPE_CLASSIFIER_REGISTERED

    # ---- the anchors: partition, labels, the recorded trap guards ---------
    views = phase9.deall_partition_stems(
        [child.name for child in deall_dir.iterdir()]
    )
    if len(views["composite"]) != 25:
        raise ValueError(
            f"{len(views['composite'])} composite images where the "
            "registration fixes 25"
        )
    labels_by_id = phase9.deall_read_labels(
        deall_dir / phase9.DEALL_LABELS_FILENAME
    )
    stems = [Path(name).stem for name in views["composite"]]
    mismatch = sorted(set(stems) ^ set(labels_by_id))
    if mismatch:
        raise ValueError(f"composites and labels disagree on {mismatch}")
    anchor_grades = np.array([int(labels_by_id[stem]) for stem in stems])

    # ---- the cohort: cached embeddings, mean and class3 -------------------
    images, labels, patient_ids, _ = phase3.load_inputs(
        manifest_dir, staged_dir, task["geometry"], task["label"]
    )
    rows = load_manifest(manifest_dir / "manifest.csv")
    mean_by_id = {int(r["patient_id"]): float(r["mean"]) for r in rows}
    class3_by_id = {int(r["patient_id"]): int(r["class3"]) for r in rows}
    features, _ = embeddings_module_load(
        embeddings_dir, [int(p) for p in patient_ids]
    )
    features = np.asarray(features, dtype=np.float64)
    truth_mean = np.array([mean_by_id[int(p)] for p in patient_ids])
    truth_class3 = np.array([class3_by_id[int(p)] for p in patient_ids])
    values, counts = np.unique(truth_class3, return_counts=True)
    majority = float(counts.max()) / float(counts.sum())
    chance = 1.0 / len(values)

    # ---- the anchors' features: re-extracted live (the registration's
    # inputs correction -- the -3 run persisted scores, not features),
    # same path, parity check repeated.
    staged_faces = np.stack([
        staging.stage(render.load_image(deall_dir / name)).image
        for name in views["composite"]
    ])
    extractor = FrozenExtractor(task["backbone"], batch_size=task["batch_size"])
    anchor_features = np.asarray(extractor(staged_faces), dtype=np.float64)
    parity = float(np.max(np.abs(
        extractor(np.asarray(images[:1]))[0]
        - np.asarray(features[0], dtype=np.float32)
    )))
    ctx.log(f"live-path parity (patient {int(patient_ids[0])}): "
            f"max|delta| {parity:.2e}")
    if parity > 1e-2:
        raise ValueError(
            f"live-path parity {parity:.2e} exceeds 1e-2: the anchors "
            "would not live in the cohort embeddings' space"
        )

    # ---- every registered cell, nothing selected --------------------------
    report: dict = {}
    per_patient: dict = {}
    for metric_name in ("euclidean", "cosine"):
        consistency = phase9.anchor_self_consistency(
            anchor_features, anchor_grades, metric=metric_name
        )
        cells: dict = {}
        for k in spec["cells"]["k"]:
            for vote in spec["cells"]["votes"]:
                predicted = phase9.anchor_knn_grades(
                    anchor_features, anchor_grades, features,
                    k=int(k), weighted=(vote == "weighted"),
                    metric=metric_name,
                )
                predicted_class3 = np.array([
                    phase9.grade_to_class3(g) for g in predicted
                ])
                cell_name = f"k{k}_{vote}"
                cells[cell_name] = {
                    "pcc_vs_mean": metrics.pcc(truth_mean, predicted.astype(float)),
                    "spearman_vs_mean": metrics.spearman(
                        truth_mean, predicted.astype(float)
                    ),
                    "accuracy_class3": float(
                        (predicted_class3 == truth_class3).mean()
                    ),
                }
                per_patient[f"{metric_name}_{cell_name}"] = predicted
                ctx.log(
                    f"  {metric_name} {cell_name}: PCC "
                    f"{cells[cell_name]['pcc_vs_mean']:.4f}, acc3 "
                    f"{cells[cell_name]['accuracy_class3']:.4f} "
                    f"(majority {majority:.3f}, chance {chance:.3f})"
                )
        report[metric_name] = {
            "anchor_self_consistency": consistency,
            "cells": cells,
        }
        ctx.log(
            f"{metric_name} anchor self-consistency: "
            f"{consistency['agreeing']}/{consistency['n']} "
            f"({consistency['fraction']:.3f})"
        )

    # ---- artifacts --------------------------------------------------------
    with ctx.atomic("prototype_classifier_grades.csv", tier="CLUSTER-ONLY") as tmp:
        columns = sorted(per_patient)
        lines = ["patient_id,mean,class3," + ",".join(columns)]
        for row in range(len(patient_ids)):
            lines.append(
                f"{int(patient_ids[row])},{truth_mean[row]!r},"
                f"{int(truth_class3[row])},"
                + ",".join(str(int(per_patient[c][row])) for c in columns)
            )
        tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")

    summary = {
        "registration": spec,
        "n": int(len(patient_ids)),
        "baselines": {"majority": majority, "chance": chance},
        "results": report,
        "live_path_parity_max_abs": parity,
        "no_training": (
            "no parameter was fitted anywhere in this run; the heads "
            "never enter it"
        ),
    }
    with ctx.atomic("metrics.json", tier="SHAREABLE") as tmp:
        tmp.write_text(
            json.dumps(as_builtin(summary), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )



def task_cleftgnn_cv(ctx: RunContext) -> None:
    """Phase 10: the CleftGNN replication under this project's criterion
    (phase10.PHASE_10_REGISTERED, PHASE_10_UNBLOCKED, CLEFTGNN_BUILT).

    Train as they trained (CE on the resolved consensus grade, plain SGD
    lr 0.01 momentum 0, batch 16), evaluate as we evaluate (expected-value
    predictions against the panel mean, 5-fold OOF, five seeds, seed CSVs
    in the baseline layout for the paired BCa vs 0.2520). Both output
    readings written; neither selected after the fact. Every registered
    caveat is stamped into metrics.

    **[2026-08-17] The task serves BOTH cells**
    (``phase10.NOTEBOOK_RECIPE_CELL_REGISTERED``). ``task.recipe`` selects
    which of the group's artifacts the architecture is faithful to and
    ``task.optimizer`` which optimiser it runs; both DEFAULT to the
    manuscript's, so ``p10_cleftgnn.yaml`` is untouched and its numbers
    stand. Everything downstream of the model -- label, folds, seeds,
    early stopping, both readings, the CSV layout -- is shared code by
    construction, which is what makes the two cells comparable at all.
    """
    import json

    from . import phase10
    from .data import scoresheet
    from .data.manifest import load_manifest
    from .models import cleftgnn
    from .train import harness, phase3

    task = ctx.config["task"]
    declared = {entry["name"]: entry for entry in ctx.inputs}
    manifest_dir = Path(declared[task["manifest_artifact"]]["path"])
    staged_dir = Path(declared[task["staged_artifact"]]["path"])
    sheet_path = Path(declared[task["scoresheet_artifact"]]["path"])

    images, mean_labels, patient_ids, _ = phase3.load_inputs(
        manifest_dir, staged_dir, task["geometry"], task["label"]
    )
    rows = load_manifest(manifest_dir / "manifest.csv")
    by_id = {int(r["patient_id"]): r for r in rows}
    assignments = {int(r["patient_id"]): int(r["fold"]) for r in rows}

    # ---- the training label: the sheet's OWN Median column ----------------
    # phase10.CONSENSUS_LABEL_IS_THE_MEDIAN. The resolution lives in
    # median_by_patient (below), which Phase 11's pre-step also calls --
    # ONE implementation, because two copies of a label resolution is how
    # two runs quietly train against different labels.
    consensus = median_by_patient(by_id, sheet_path, patient_ids)
    grades, counts = np.unique(consensus.astype(int), return_counts=True)
    ctx.log(
        f"training label = the sheet's {scoresheet.MEDIAN_COLUMN} column, "
        f"verified against the five rater cells on all {len(consensus)} "
        "patients (any disagreement would have raised); grade counts "
        + ", ".join(f"{g}:{c}" for g, c in zip(grades, counts))
    )

    # ---- which cell is this? ----------------------------------------------
    recipe = str(task.get("recipe", "manuscript"))
    optimizer = str(task.get("optimizer", "sgd"))
    if recipe not in cleftgnn.RECIPES:
        raise ValueError(
            f"task.recipe {recipe!r} is not one of {cleftgnn.RECIPES}"
        )
    ctx.log(
        f"recipe = {recipe} ({optimizer} at lr {task['learning_rate']}); "
        + (
            "the manuscript cell -- ResNet-50, SABM eqs 7-9, additive "
            "fusion, both LayerNorms"
            if recipe == "manuscript" else
            "the NOTEBOOK cell -- frozen ViT-B/16 by their extraction "
            "path, their w_beta attention, multiplicative fusion "
            "f_t + f_t*v, NO LayerNorm anywhere "
            "(phase10.NOTEBOOK_RECIPE_CELL_REGISTERED)"
        )
    )

    # ---- five seeds through the frozen harness ----------------------------
    seeds = [int(s) for s in task["seeds"]]
    config_of = lambda seed: harness.TrainConfig(
        seed=seed,
        inner_val_frac=float(task["inner_val_frac"]),
        max_epochs=int(task["max_epochs"]),
        patience=int(task["patience"]),
        monitor=task["monitor"],
    )
    from .cluster_csv import write_predictions

    pooled_primary, pooled_top1 = [], []
    per_seed: dict = {}
    for seed in seeds:
        created: list = []

        def make_backbone(_seed=seed, _created=created):
            backbone = cleftgnn.CleftGNNBackbone(
                learning_rate=float(task["learning_rate"]),
                momentum=float(task.get("momentum", 0.0)),
                batch_size=int(task["batch_size"]),
                seed=_seed,
                record_stages=True,
                recipe=recipe,
                optimizer=optimizer,
            )
            _created.append(backbone)
            return backbone

        result = harness.run_cv(
            features=images, labels=consensus, patient_ids=patient_ids,
            assignments=assignments, make_backbone=make_backbone,
            config=config_of(seed),
        )
        # Evaluation labels are the MEAN -- recomposed by oof id.
        mean_of = {
            int(p): float(m) for p, m in zip(patient_ids, mean_labels)
        }
        truth = np.array([mean_of[int(p)] for p in result.oof_ids])
        expected = np.asarray(result.oof_predictions, dtype=float)
        from .eval import metrics
        primary_pcc = metrics.pcc(truth, expected)
        pooled_primary.append(primary_pcc)

        # Top-1 beside: the captured fold backbones on their own test ids.
        folds = [run.fold for run in result.folds]
        if len(created) != len(folds):
            raise ValueError(
                f"{len(created)} backbones for {len(folds)} folds; the "
                "capture is misaligned"
            )
        row_of = {int(p): i for i, p in enumerate(patient_ids)}
        top1_of: dict = {}
        for run, backbone in zip(result.folds, created):
            test_rows = [row_of[int(p)] for p in run.test_ids]
            probabilities = backbone.predict_probabilities(images[test_rows])
            for pid, probs in zip(run.test_ids, probabilities):
                top1_of[int(pid)] = int(np.argmax(probs)) + 1
        top1 = np.array([top1_of[int(p)] for p in result.oof_ids], dtype=float)
        top1_pcc = metrics.pcc(truth, top1)
        pooled_top1.append(top1_pcc)

        fold_of = {int(p): assignments[int(p)] for p in result.oof_ids}
        write_predictions(
            ctx.path(f"seed_{seed}__predictions.csv", tier="CLUSTER-ONLY"),
            zip(result.oof_ids, truth, expected,
                (fold_of[int(p)] for p in result.oof_ids)),
        )
        write_predictions(
            ctx.path(f"seed_{seed}__top1_grades.csv", tier="CLUSTER-ONLY"),
            zip(result.oof_ids, truth, top1,
                (fold_of[int(p)] for p in result.oof_ids)),
        )
        # **[2026-08-17] The curves and the stage table are WRITTEN, not
        # discarded.** The harness computes a per-epoch curve for every
        # fold and returns it in FoldRun.curve; this task used to throw
        # all 25 away, which is what made the monitor question
        # unanswerable (phase10.CURVE_WRITING_PROPOSED). The stage ratios
        # come from the fold's SELECTED epoch, so criterion (i) is
        # answered on the arm's own fold rather than assumed from the
        # probe's initialisation batch (phase10.CRITERION_I_UNREPORTED).
        with ctx.atomic(f"seed_{seed}__curves.csv", tier="SHAREABLE") as tmp:
            lines = ["fold,epoch,train_loss,inner_val_mse,inner_val_pcc"]
            for fold_run in result.folds:
                for row in fold_run.curve:
                    lines.append(
                        f"{fold_run.fold},{row['epoch']},{row['train_loss']!r},"
                        f"{row['inner_val_mse']!r},{row['inner_val_pcc']!r}"
                    )
            tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")

        # **[2026-08-17] EVERY epoch's table, not only the selected one**
        # (phase10.U_SHAPE_BELONGS_LATER, phase10.ALL_EPOCHS_RECORDED).
        # The backbone already computes a table per epoch at no extra
        # forward; writing only the selected row threw the rest away, and
        # the U-shape then had to be reconstructed by hand from a run that
        # had already measured it. A longer CSV, no computation, and a
        # ``selected`` column so the old reading is one filter away.
        with ctx.atomic(f"seed_{seed}__stage_ratios.csv", tier="SHAREABLE") as tmp:
            lines = [
                "fold,epoch,selected,selected_epoch,stage,magnitude,"
                "across_image_sd,ratio"
            ]
            for fold_run, fold_backbone in zip(result.folds, created):
                by_epoch = fold_backbone.stage_ratios_by_epoch
                if not by_epoch:
                    lines.append(
                        f"{fold_run.fold},,,{fold_run.selected_epoch},"
                        "NOT_RECORDED,,,"
                    )
                    continue
                for epoch in sorted(by_epoch):
                    chosen = int(epoch == fold_run.selected_epoch)
                    for stage, (magnitude, across, ratio) in by_epoch[
                        epoch
                    ].items():
                        lines.append(
                            f"{fold_run.fold},{epoch},{chosen},"
                            f"{fold_run.selected_epoch},{stage},"
                            f"{magnitude!r},{across!r},{ratio!r}"
                        )
            tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")

        fused_at_selection = [
            fold_backbone.stage_ratios_by_epoch.get(
                fold_run.selected_epoch, {}
            ).get("fused", (float("nan"),) * 3)[2]
            for fold_run, fold_backbone in zip(result.folds, created)
        ]
        ctx.log(
            f"  seed {seed}: fused ratio at each fold's selected epoch "
            + ", ".join(f"{value:.4f}" for value in fused_at_selection)
        )

        per_seed[seed] = {
            "expected_value_pcc_vs_mean": primary_pcc,
            "top1_pcc_vs_mean": top1_pcc,
            "selected_epochs": [run.selected_epoch for run in result.folds],
            "fused_ratio_at_selected_epoch": fused_at_selection,
        }
        ctx.log(
            f"seed {seed}: expected-value PCC {primary_pcc:.4f}, top-1 "
            f"{top1_pcc:.4f}"
        )

    band = phase3.seed_variance(pooled_primary)
    cell = (
        {
            "which": "the NOTEBOOK-RECIPE cell",
            "faithful_to": (
                "the notebook, the only executable artifact the group has "
                "shared -- Adam lr 0.001, multiplicative fusion, frozen "
                "ViT-B/16, batch 16, no LayerNorm anywhere"
            ),
            "registered": phase10.NOTEBOOK_RECIPE_CELL_REGISTERED,
            "deviations": phase10.NOTEBOOK_CELL_DEVIATIONS,
        }
        if recipe == "notebook"
        else {
            "which": "the MANUSCRIPT-RECIPE cell",
            "faithful_to": (
                "the manuscript, the publication under revision -- SGD "
                "lr 0.01, additive eq (9), ResNet-50, SABM eqs 7-9"
            ),
            "deviations": phase10.REGISTERED_DEVIATIONS,
        }
    )
    summary = {
        "recipe": recipe,
        "optimizer": optimizer,
        "cell": cell,
        "registration": phase10.PHASE_10_REGISTERED,
        "unblocked": phase10.PHASE_10_UNBLOCKED,
        "training_label": {
            "source": (
                f"the score sheet's {scoresheet.MEDIAN_COLUMN} column -- the "
                "group's consensus grade (phase10.CONSENSUS_LABEL_IS_THE_MEDIAN)"
            ),
            "n": int(len(consensus)),
            "verified": (
                "recomputed median of the five rater cells equals the "
                "published column on every row; disagreement is fatal"
            ),
            "grade_counts": {
                int(g): int(c) for g, c in zip(grades, counts)
            },
            "learnability_237": phase10.CONSENSUS_LABEL_IS_THE_MEDIAN[
                "learnability_237"
            ],
        },
        "per_seed": per_seed,
        "pooled": {
            "expected_value_pcc_vs_mean": {
                "mean": band["mean"], "sd": band["sd"],
            },
            "top1_pcc_vs_mean": {
                "mean": phase3.seed_variance(pooled_top1)["mean"],
                "sd": phase3.seed_variance(pooled_top1)["sd"],
            },
        },
        "caveats": phase10.CLEFTGNN_BUILT["caveats_travel"],
        "next": (
            "the paired BCa vs the 0.2520 arm consumes the per-seed CSVs; "
            "its config follows this run's directory"
        ),
    }
    with ctx.atomic("metrics.json", tier="SHAREABLE") as tmp:
        tmp.write_text(
            json.dumps(as_builtin(summary), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )



def median_by_patient(by_id: dict, sheet_path: Path, patient_ids):
    """The score sheet's OWN Median grade per patient, in ``patient_ids``
    order (``phase10.CONSENSUS_LABEL_IS_THE_MEDIAN``).

    ``by_id`` maps patient id to its manifest row, which is where the
    frontal photo id lives. The reader is ``data.scoresheet`` -- the
    module that already knows this sheet's identity -- and it recomputes
    the median of the five rater cells for every row, refusing any
    disagreement. So the precomputed column is verified, not trusted, and
    no derived file enters the label's provenance.

    **[EXTRACTED 2026-08-17]** It was inline in ``task_cleftgnn_cv``.
    Phase 11's pre-step scores IEM against this same grade
    (``phase11.PRE_STEP_REGISTERED``), and a second copy of a label
    resolution is how two runs quietly measure against different labels.
    """
    from .data import scoresheet

    median_by_photo = scoresheet.load_median(sheet_path)
    resolved = []
    for pid in (int(p) for p in patient_ids):
        frontal = int(by_id[pid]["frontal_id"])
        if frontal not in median_by_photo:
            raise ValueError(
                f"patient {pid}: frontal photo {frontal} has no row in the "
                f"score sheet's {scoresheet.MEDIAN_COLUMN!r} column"
            )
        resolved.append(median_by_photo[frontal])
    return np.array(resolved, dtype=float)


def phase10_rater_grades(manifest_dir: Path, sheet_path: Path, patient_ids):
    """``{rater: array of that rater's grade per patient}``, aligned to
    ``patient_ids`` through each patient's frontal photo id.

    The five rater columns come from ``data.scoresheet``, which owns this
    sheet's identity -- the R10 lesson from the Median column, applied
    rather than relearned (phase10.CONSENSUS_LABEL_IS_THE_MEDIAN).
    """
    from .data import scoresheet
    from .data.manifest import load_manifest

    sheet = scoresheet.load(sheet_path)
    rows = {int(r["patient_id"]): r for r in load_manifest(manifest_dir / "manifest.csv")}
    out = {}
    for position, rater in enumerate(scoresheet.RATERS):
        grades = []
        for pid in (int(p) for p in patient_ids):
            frontal = int(rows[pid]["frontal_id"])
            if frontal not in sheet.rows:
                raise ValueError(
                    f"patient {pid}: frontal photo {frontal} is not in the "
                    "score sheet"
                )
            grades.append(sheet.rows[frontal].grades[position])
        out[rater] = np.array(grades, dtype=float)
    return out


def task_rater_screen(ctx: RunContext) -> None:
    """Phase 10: the rater-specific screen (phase10.RATER_SCREEN_REGISTERED).

    The 0.2520 arm -- frozen ViT-B/16, imagenet, G1 -- refit once per
    rater, five seeds each, the standard 5-fold OOF criterion. The prior
    is committed BOTH WAYS in the registration before any number: every
    cell below 0.2520 (rater-specific modelling measured worse), or any
    cell beyond the arm's own seed band (prior refuted, the full ladder
    justified). Cheap: only the head refits, on declared embeddings.
    """
    import json

    from . import phase10
    from .data.manifest import load_manifest
    from .eval import metrics
    from .train import harness, phase3
    from .train.torch_backbone import EmbeddingHeadBackbone

    task = ctx.config["task"]
    declared = {entry["name"]: entry for entry in ctx.inputs}
    manifest_dir = Path(declared[task["manifest_artifact"]]["path"])
    staged_dir = Path(declared[task["staged_artifact"]]["path"])
    embeddings_dir = Path(declared[task["embeddings_artifact"]]["path"])
    sheet_path = Path(declared[task["scoresheet_artifact"]]["path"])

    _, _, patient_ids, _ = phase3.load_inputs(
        manifest_dir, staged_dir, task["geometry"], task["label"],
        require_images=False,
    )
    rows = load_manifest(manifest_dir / "manifest.csv")
    assignments = {int(r["patient_id"]): int(r["fold"]) for r in rows}
    features, _ = embeddings_module_load(embeddings_dir, patient_ids)
    by_rater = phase10_rater_grades(manifest_dir, sheet_path, patient_ids)
    seeds = [int(s) for s in task["seeds"]]

    spec = phase10.RATER_SCREEN_REGISTERED
    from .cluster_csv import write_predictions

    report = {}
    for rater, labels in by_rater.items():
        pooled = []
        for seed in seeds:
            config = harness.TrainConfig(
                seed=seed,
                inner_val_frac=float(task["inner_val_frac"]),
                max_epochs=int(task["max_epochs"]),
                patience=int(task["patience"]),
                monitor=task["monitor"],
            )
            result = harness.run_cv(
                features=features, labels=labels, patient_ids=patient_ids,
                assignments=assignments,
                make_backbone=lambda _seed=seed: EmbeddingHeadBackbone(
                    learning_rate=float(task["learning_rate"]),
                    weight_decay=float(task["weight_decay"]),
                    seed=_seed,
                ),
                config=config,
            )
            pooled.append(float(result.metrics()["pcc"]))
            fold_of = {int(p): assignments[int(p)] for p in result.oof_ids}
            safe = rater.split(" - ")[0].replace(" ", "_").lower()
            write_predictions(
                ctx.path(f"{safe}__seed_{seed}__predictions.csv",
                         tier="CLUSTER-ONLY"),
                zip(result.oof_ids, result.oof_truth, result.oof_predictions,
                    (fold_of[int(p)] for p in result.oof_ids)),
            )
        band = phase3.seed_variance(pooled)
        report[rater] = {
            "per_seed_pcc": pooled,
            "mean": band["mean"],
            "sd": band["sd"],
            "below_0_2520": bool(band["mean"] < 0.2520),
        }
        ctx.log(
            f"{rater}: pooled PCC {band['mean']:.4f} (sd {band['sd']:.4f}) "
            f"-- {'below' if band['mean'] < 0.2520 else 'AT OR ABOVE'} 0.2520"
        )

    # The registered readings, applied -- not chosen now.
    arm_sd = 0.0148
    threshold = 0.2520 + 2 * arm_sd
    above = [r for r, cell in report.items() if cell["mean"] > threshold]
    verdict = (
        "PRIOR_HELD -- every rater-specific cell below 0.2520"
        if all(cell["below_0_2520"] for cell in report.values())
        else "PRIOR_REFUTED -- a cell beyond the arm's seed band"
        if above else "MIXED -- no cell beyond the band, not all below"
    )
    ctx.log(f"screen verdict: {verdict}")

    summary = {
        "registration": spec,
        "per_rater": report,
        "comparison_point": {"arm": 0.2520, "sd": arm_sd, "band_top": threshold},
        "verdict": verdict,
        "beyond_band": above,
        "note": (
            "each rater is a DIFFERENT target, so these are not paired "
            "against the 0.2520 arm's vectors -- the reading is the "
            "registered one, level against the arm and its seed band"
        ),
    }
    with ctx.atomic("metrics.json", tier="SHAREABLE") as tmp:
        tmp.write_text(
            json.dumps(as_builtin(summary), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def task_cleftgnn_faithful(ctx: RunContext) -> None:
    """Phase 10: CleftGNN under THEIR protocol
    (phase10.FAITHFUL_ARM_REGISTERED) -- the cell beside the protocol arm.

    Rater-specific models, one per rater; an 85:15 stratified split; their
    metrics (Top-1 macro precision/recall/F1 and PCC); a single run with
    no intervals, which is their form and is recorded rather than
    repaired. Evaluated on both their test-set shapes: the held-out 15%
    and the 25-image benchmark against its Score. The two arms differ ONLY
    in protocol, so any gap between them is protocol, not architecture.
    """
    import json

    from sklearn.model_selection import train_test_split

    from . import phase9, phase10
    from .eval import metrics
    from .geometry import render, staging
    from .models import cleftgnn
    from .train import harness, phase3

    task = ctx.config["task"]
    declared = {entry["name"]: entry for entry in ctx.inputs}
    manifest_dir = Path(declared[task["manifest_artifact"]]["path"])
    staged_dir = Path(declared[task["staged_artifact"]]["path"])
    sheet_path = Path(declared[task["scoresheet_artifact"]]["path"])
    deall_dir = Path(declared[task["deall_artifact"]]["path"])

    images, _, patient_ids, _ = phase3.load_inputs(
        manifest_dir, staged_dir, task["geometry"], task["label"]
    )
    by_rater = phase10_rater_grades(manifest_dir, sheet_path, patient_ids)

    # Their Benchmark shape: the same 25 images, against Score.
    views = phase9.deall_partition_stems(
        [child.name for child in deall_dir.iterdir()]
    )
    if len(views["composite"]) != 25:
        raise ValueError(f"{len(views['composite'])} composites, expected 25")
    scores = phase9.deall_read_labels(deall_dir / phase9.DEALL_LABELS_FILENAME)
    benchmark_stems = [Path(name).stem for name in views["composite"]]
    benchmark_faces = np.stack([
        staging.stage(render.load_image(deall_dir / name)).image
        for name in views["composite"]
    ])
    benchmark_truth = np.array(
        [scores[stem] for stem in benchmark_stems], dtype=int
    )

    seed = int(ctx.config["seed"])
    report = {}
    for rater, labels in by_rater.items():
        grades = labels.astype(int)
        fit_rows, test_rows = train_test_split(
            np.arange(len(grades)), test_size=0.15, random_state=seed,
            stratify=grades,
        )
        # The registered deviation applies here too: inner-val early
        # stopping, carved from the 85% by the frozen splitter.
        inner_fit, inner_val = harness.inner_val_split(
            list(range(len(fit_rows))), float(task["inner_val_frac"]), seed, 0
        )
        train_rows = fit_rows[np.array(inner_fit, dtype=int)]
        val_rows = fit_rows[np.array(inner_val, dtype=int)]

        backbone = cleftgnn.CleftGNNBackbone(
            learning_rate=float(task["learning_rate"]),
            momentum=float(task["momentum"]),
            batch_size=int(task["batch_size"]),
            seed=seed,
        )
        backbone.reset(labels[train_rows])
        best, best_epoch, waited, best_state = None, 0, 0, None
        for epoch in range(1, int(task["max_epochs"]) + 1):
            backbone.train_epoch(images[train_rows], labels[train_rows])
            predicted = backbone.predict(images[val_rows])
            score = float(np.mean((predicted - labels[val_rows]) ** 2))
            if best is None or score < best - 1e-12:
                best, best_epoch, waited = score, epoch, 0
                best_state = {
                    k: v.detach().cpu().clone()
                    for k, v in backbone._model.state_dict().items()
                }
            else:
                waited += 1
                if waited >= int(task["patience"]):
                    break
        if best_state is not None:
            backbone._model.load_state_dict(best_state)
        ctx.log(f"{rater}: selected epoch {best_epoch} (inner-val MSE {best:.4f})")

        cells = {}
        for shape, faces, truth in (
            ("study_15pct", images[test_rows], grades[test_rows]),
            ("benchmark_25", benchmark_faces, benchmark_truth),
        ):
            probabilities = backbone.predict_probabilities(faces)
            top1 = np.argmax(probabilities, axis=1) + 1
            cell = phase10.top1_macro_prf(truth, top1)
            cell["pcc_top1"] = metrics.pcc(
                truth.astype(float), top1.astype(float)
            )
            cell["n"] = int(len(truth))
            cells[shape] = cell
            ctx.log(
                f"   {shape} (n={len(truth)}): F1 {cell['f1_macro']:.4f}, "
                f"PCC {cell['pcc_top1']:.4f}"
            )
        report[rater] = {"selected_epoch": best_epoch, **cells}

    summary = {
        "registration": phase10.FAITHFUL_ARM_REGISTERED,
        "per_rater": report,
        "their_598_cell": phase10.FAITHFUL_ARM_REGISTERED["their_598_cell"],
        "no_intervals": (
            "single run by construction -- their form, recorded rather "
            "than repaired; nothing here is claimable under this "
            "project's criterion"
        ),
        "purpose": phase10.FAITHFUL_ARM_REGISTERED["purpose"],
    }
    with ctx.atomic("metrics.json", tier="SHAREABLE") as tmp:
        tmp.write_text(
            json.dumps(as_builtin(summary), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


#: **[2026-08-31] The declared split-seed derivation for the Phase 10
#: annex.** Written down rather than only implemented, so the draw set
#: is auditable from the root without running anything
#: (``phase10_annex.DISTRIBUTION_ARM_REGISTERED["draws_enumerated"]``).
P10X_SPLIT_SEED_RULE = (
    "numpy SeedSequence(root_seed).generate_state(n_draws, dtype=uint32), "
    "in order; the resulting seeds are ENUMERATED in the config and the "
    "task refuses a list that does not re-derive from its declared root"
)


def p10x_split_seeds(root_seed: int, n_draws: int) -> list:
    """The ``P10X_SPLIT_SEED_RULE`` derivation, one place.

    ``SeedSequence`` is used rather than an RNG stream because its
    state derivation is a specified algorithm rather than a stream
    guarantee -- the draw set must be reproducible from the root years
    from now, not merely within one numpy generation.
    """
    state = np.random.SeedSequence(int(root_seed)).generate_state(
        int(n_draws), dtype=np.uint32
    )
    return [int(value) for value in state]


def p10x_shard_draws(n_draws: int, shard) -> list:
    """Which draw indices this shard runs: ``index % count``.

    Sharding by DRAW rather than by rater is deliberate -- every shard
    then yields all five raters on its own subset, so a lost shard
    costs draws rather than an entire rater's distribution
    (``phase10_annex.DISTRIBUTION_ARM_REGISTERED["compute_shape"]``).
    """
    count, index = int(shard["count"]), int(shard["index"])
    return [draw for draw in range(int(n_draws)) if draw % count == index]


def _p10x_draw_rows(n: int, split_seed: int, train_size: int, test_size: int):
    """The declared draw procedure, one implementation.

    Permutation of the cohort rows under ``split_seed``; the first
    ``test_size`` rows are the test side, the next ``train_size`` the
    train side, the remainder UNUSED. The annex runs on OUR cohort by
    construction -- deriving the 181-analogue set is not attempted
    (``phase10_annex.SUPERVISION_28_DEPENDENCY``: 181-subset-of-237 is not
    assumed without IDs).
    """
    if n < train_size + test_size:
        raise ValueError(
            f"cohort has {n} patients; a {train_size}/{test_size} draw "
            f"needs at least {train_size + test_size}"
        )
    perm = np.random.default_rng(int(split_seed)).permutation(n)
    return perm[test_size:test_size + train_size], perm[:test_size]


def _p10x_occupancy(values) -> dict:
    """Class occupancy over the five grades, as a plain mapping."""
    return {str(g): int(np.sum(values == g)) for g in range(1, 6)}


def _p10x_one_fit(
    *, images, grades, train_rows, test_rows, recipe, trainable,
    model_seed, log=None,
) -> dict:
    """**ONE fit, and the ONLY fit path in the annex.**

    Both ``task_p10x_gate_fullfit`` and ``task_p10x_distribution`` call
    this; neither constructs a model itself. That is the Phase 18 IEM
    lesson applied before it can be repeated, and
    ``phase10.PROBE_RECONSTRUCTED_THE_PIPELINE`` is the same shape: a
    second implementation of one computation is how a table starts
    describing something the model is not doing. Here the specific harm
    is that the gate's VERIFIED cost and parameter count would detach
    from the distribution's numbers with nothing to disagree.

    Last epoch IS the model -- no validation split, no early stopping,
    no checkpoint selection (``phase10.NOTEBOOK_BUDGET_AND_
    NORMALISATION``). ``model_seed`` is held FIXED across draws by the
    distribution arm: the annex varies the SPLIT and nothing else, so
    letting the initialisation move too would confound the two.
    """
    import time

    from . import phase10, phase11
    from .eval import metrics
    from .models import cleftgnn

    backbone = cleftgnn.CleftGNNBackbone(
        recipe=cleftgnn.ANNEX_RECIPE,
        optimizer=recipe["optimizer"],
        learning_rate=float(recipe["learning_rate"]),
        momentum=0.0,
        batch_size=int(recipe["batch_size"]),
        seed=int(model_seed),
        trainable=trainable,
    )
    backbone.reset(grades[train_rows].astype(float))
    report = getattr(backbone, "parameter_report", {})

    losses, per_epoch = [], []
    train_start = time.perf_counter()
    for epoch in range(1, int(recipe["epochs"]) + 1):
        tick = time.perf_counter()
        loss = backbone.train_epoch(
            images[train_rows], grades[train_rows].astype(float)
        )
        per_epoch.append(time.perf_counter() - tick)
        losses.append(float(loss))
        if log is not None:
            log(f"epoch {epoch}/{recipe['epochs']}: loss {loss:.4f} "
                f"({per_epoch[-1]:.1f}s)")
    train_total = time.perf_counter() - train_start

    eval_start = time.perf_counter()
    probabilities = backbone.predict_probabilities(images[test_rows])
    eval_seconds = time.perf_counter() - eval_start
    top1 = np.argmax(probabilities, axis=1) + 1
    truth = grades[test_rows]

    cell = phase10.top1_macro_prf(truth, top1)
    # PCC is undefined on a zero-variance vector; None is the honest
    # value there, never a number invented for the column.
    if float(np.std(truth)) == 0.0 or float(np.std(top1)) == 0.0:
        cell["pcc_top1"] = None
    else:
        cell["pcc_top1"] = metrics.pcc(
            truth.astype(float), top1.astype(float)
        )
    # IEM through phase11.iem under convention A, the answered
    # direction (phase11.IEM_DIRECTION_ANSWERED) -- and NEVER beside
    # CleftGNN's Table 2/4/6 (the standing prohibition).
    cell["iem_a_top1_mean"] = float(np.mean(
        phase11.iem(top1.astype(float) - truth.astype(float))
    ))

    # Floors beside every metric (phase10_annex.FLOORS_REGISTERED):
    # the majority-class predictor from the TRAIN side, and the
    # constant predictor's IEM. Train-side by construction -- a
    # test-side majority would peek at the answer.
    majority = int(np.argmax(np.bincount(grades[train_rows], minlength=6)[1:])) + 1
    constant = np.full(len(truth), majority, dtype=int)
    floors = {
        "majority_grade": majority,
        "majority_class": phase10.top1_macro_prf(truth, constant),
        "iem_constant_mean": float(np.mean(
            phase11.iem(constant.astype(float) - truth.astype(float))
        )),
        "pcc": (
            "undefined -- a constant predictor has zero variance "
            "(phase10_annex.FLOORS_REGISTERED)"
        ),
    }
    return {
        "metrics": cell,
        "floors": floors,
        "parameter_report": report,
        "loss_by_epoch": losses,
        "wall_clock_seconds": {
            "train_total": train_total,
            "train_per_epoch": per_epoch,
            "eval": eval_seconds,
        },
    }


def p10x_locate(values, point) -> dict:
    """Where ``point`` falls in a distribution, and by what percentile.

    ``where`` is one of ``inside`` / ``above_all`` / ``below_all`` /
    ``undefined``. The boundary belongs to ``inside``: equal to the
    maximum is not beyond it. ``undefined`` covers an empty
    distribution -- every draw's PCC undefined -- because reporting a
    percentile against nothing would be a number with no denominator.
    """
    defined = [float(v) for v in values if v is not None]
    if not defined:
        return {"value": float(point), "where": "undefined",
                "percentile": None, "n": 0}
    point = float(point)
    if point > max(defined):
        where = "above_all"
    elif point < min(defined):
        where = "below_all"
    else:
        where = "inside"
    below = sum(1 for v in defined if v < point)
    return {
        "value": point,
        "where": where,
        "percentile": 100.0 * below / len(defined),
        "n": len(defined),
    }


def p10x_apply_readings(grid) -> dict:
    """The two committed readings, applied MECHANICALLY.

    ``grid`` maps rater -> {point name -> ``where``}. Reading 1 fires
    only if every published point falls INSIDE every rater's
    distribution; reading 2 only if every one falls ABOVE all of them.
    **Anything else is the registered observation, not a third
    reading** (``phase10_annex.MIXED_PATTERN_REGISTERED``) -- including
    the ``below_all`` case, which the two readings do not cover and
    which was named before the run rather than decided after.
    """
    from . import phase10_annex

    wheres = {w for rater in grid.values() for w in rater.values()}
    readings = phase10_annex.READINGS_COMMITTED
    if wheres == {"inside"}:
        fired = "published_inside_the_distribution"
    elif wheres == {"above_all"}:
        fired = "published_outside_and_above"
    else:
        return {
            "fired": None,
            "observation": "MIXED_PATTERN_REGISTERED",
            "reading": None,
            "what_was_seen": sorted(wheres),
            "handling": phase10_annex.MIXED_PATTERN_REGISTERED["handling"],
        }
    return {
        "fired": fired,
        "observation": None,
        "reading": readings[fired],
        "what_was_seen": sorted(wheres),
    }


def task_p10x_gate_fullfit(ctx: RunContext) -> None:
    """[2026-08-30] Phase 10 annex: the COMPUTE GATE. One plain-random
    153/28 draw, one rater, one full-trainable ResNet-50 CleftGNN fit
    under the notebook's recipe and its CIFAR-10 statistics
    (``phase10_annex.COMPUTE_GATE_DESIGNED``; rulings
    ``RULING_A_ALL_FIVE_RATERS``, ``RULING_B_FULL_TRAINABLE``,
    ``NORMALISATION_RULED``).

    The gate exists to MEASURE the per-fit cost before any draw count N
    is declared -- the Phase 17 lesson, measure before declaring. It is
    NOT an arm: one draw by construction, nothing claimable, and the
    annex prohibition travels in its metrics. Last epoch IS the model
    -- no validation split, no early stopping, no checkpoint selection
    (``phase10.NOTEBOOK_BUDGET_AND_NORMALISATION``). A degenerate draw
    (test side missing a grade) is REPORTED, never discarded or
    redrawn: how often the design yields one is itself a finding.

    **[REFACTORED 2026-08-31, behaviour-preserving]** The fit moved into
    ``_p10x_one_fit`` so the distribution arm runs THIS path rather than
    a copy of it; what this task computes and reports is unchanged, and
    the suite asserts it.
    """
    import json

    from . import phase10_annex
    from .models import cleftgnn
    from .train import phase3

    task = ctx.config["task"]
    declared = {entry["name"]: entry for entry in ctx.inputs}
    manifest_dir = Path(declared[task["manifest_artifact"]]["path"])
    staged_dir = Path(declared[task["staged_artifact"]]["path"])
    sheet_path = Path(declared[task["scoresheet_artifact"]]["path"])

    if int(task["region_count"]) != cleftgnn.N_REGIONS:
        raise ValueError(
            f"task.region_count {task['region_count']} is not the model's "
            f"{cleftgnn.N_REGIONS}; the schema should have refused this"
        )

    # ``label`` loads for alignment only; the training label is the
    # rater's own integer grades, sheet identity, UNMATCHED to their
    # A-E (phase10_annex.RATER_PANEL_DISCREPANCY_EIGHTH).
    images, _, patient_ids, _ = phase3.load_inputs(
        manifest_dir, staged_dir, task["geometry"], task["label"]
    )
    grades = phase10_rater_grades(manifest_dir, sheet_path, patient_ids)[
        task["rater"]
    ].astype(int)

    n = len(patient_ids)
    train_size, test_size = int(task["train_size"]), int(task["test_size"])
    train_rows, test_rows = _p10x_draw_rows(
        n, int(task["split_seed"]), train_size, test_size
    )

    occupancy = {
        "train": _p10x_occupancy(grades[train_rows]),
        "test": _p10x_occupancy(grades[test_rows]),
    }
    degenerate = any(occupancy["test"][str(g)] == 0 for g in range(1, 6))
    ctx.log(
        f"draw at split_seed {task['split_seed']}: occupancy "
        f"train {occupancy['train']} test {occupancy['test']}"
        + (" -- DEGENERATE test side, reported not discarded"
           if degenerate else "")
    )

    fit = _p10x_one_fit(
        images=images, grades=grades,
        train_rows=train_rows, test_rows=test_rows,
        recipe=task["recipe"], trainable=task["trainable"],
        model_seed=int(ctx.config["seed"]), log=ctx.log,
    )
    cell, floors = fit["metrics"], fit["floors"]
    report = fit["parameter_report"]
    train_total = fit["wall_clock_seconds"]["train_total"]
    per_epoch = fit["wall_clock_seconds"]["train_per_epoch"]
    eval_seconds = fit["wall_clock_seconds"]["eval"]
    if report:
        ctx.log(
            f"{report.get('trainable_parameters')} of "
            f"{report.get('total_parameters')} parameters trainable "
            f"(trainable={report.get('trainable')})"
        )
    ctx.log(
        f"test n={test_size}: F1 {cell['f1_macro']:.4f} "
        f"(majority floor {floors['majority_class']['f1_macro']:.4f}), "
        f"train {train_total:.1f}s, eval {eval_seconds:.1f}s"
    )

    summary = {
        "gate": (
            "phase10_annex.COMPUTE_GATE_DESIGNED -- one draw, one "
            "rater, one full fit; measures cost before N is declared"
        ),
        "prohibition": phase10_annex.ANNEX_PROHIBITION,
        "rater": task["rater"],
        "rater_caveat": (
            "a sheet identity, UNMATCHED to their A-E "
            "(phase10_annex.RATER_PANEL_DISCREPANCY_EIGHTH)"
        ),
        "split": {
            "seed": int(task["split_seed"]),
            "cohort_n": n,
            "train_size": train_size,
            "test_size": test_size,
            "unused": n - train_size - test_size,
            "procedure": (
                "permutation under split_seed; first test_size rows "
                "test, next train_size train, remainder unused -- OUR "
                "cohort by construction, the 181-analogue not derived"
            ),
        },
        "occupancy": {
            **occupancy,
            "degenerate_test_side": degenerate,
            "macro_convention": (
                "truth-absent classes EXCLUDED from the macro average "
                "(phase10.top1_macro_prf); a degenerate draw is "
                "REPORTED, never discarded or redrawn"
            ),
        },
        "wall_clock_seconds": {
            "train_total": train_total,
            "train_per_epoch": per_epoch,
            "eval": eval_seconds,
        },
        "sizing": (
            "total fits = N x 5 raters; total cost ~ N x 5 x this "
            "per-fit wall-clock / cluster parallelism. N is declared "
            "only after this measured cost, by the ruling "
            "(COMPUTE_GATE_DESIGNED['n_is_gated'])"
        ),
        "metrics": cell,
        "floors": floors,
        "parameter_report": report,
        "recipe": task["recipe"],
        "model_selection": (
            "last epoch IS the model -- no validation split, no early "
            "stopping, no checkpoint selection "
            "(phase10.NOTEBOOK_BUDGET_AND_NORMALISATION)"
        ),
        "normalisation_note": (
            "the notebook's CIFAR-10 statistics -- a deliberately "
            "replicated defect, replicated knowingly, never 'a recipe' "
            "(phase10_annex.NORMALISATION_RULED)"
        ),
        "not_claimable": (
            "one draw by construction; nothing here enters any verdict "
            "or ledger row"
        ),
    }
    with ctx.atomic("metrics.json", tier="SHAREABLE") as tmp:
        tmp.write_text(
            json.dumps(as_builtin(summary), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def task_p10x_distribution(ctx: RunContext) -> None:
    """[2026-08-31] Phase 10 annex: THE DISTRIBUTION ARM. 500 draws x 5
    raters through the gate's own fit path
    (``phase10_annex.DISTRIBUTION_ARM_REGISTERED``, ``N_RULED``,
    ``EXIT_CRITERIA``).

    The registered question: **does a single 153/28 split identify a
    model's performance at this cohort size?** Everything except the
    split is fixed to theirs, so what this measures is the DESIGN.

    Held fixed across every fit so that only the split varies: the
    model seed, the recipe, the normalisation, the region count, the
    trainability. Degenerate draws are counted, reported as a per-rater
    frequency, and reported SEPARATELY -- never discarded, never
    silently averaged into a five-class figure they are not comparable
    with. Nothing here is claimable and nothing enters the ledger.
    """
    import csv
    import json
    import time

    from . import phase10_annex
    from .models import cleftgnn
    from .train import phase3

    task = ctx.config["task"]
    declared = {entry["name"]: entry for entry in ctx.inputs}
    manifest_dir = Path(declared[task["manifest_artifact"]]["path"])
    staged_dir = Path(declared[task["staged_artifact"]]["path"])
    sheet_path = Path(declared[task["scoresheet_artifact"]]["path"])

    if int(task["region_count"]) != cleftgnn.N_REGIONS:
        raise ValueError(
            f"task.region_count {task['region_count']} is not the model's "
            f"{cleftgnn.N_REGIONS}; the schema should have refused this"
        )

    images, _, patient_ids, _ = phase3.load_inputs(
        manifest_dir, staged_dir, task["geometry"], task["label"]
    )
    by_rater = phase10_rater_grades(manifest_dir, sheet_path, patient_ids)

    n = len(patient_ids)
    train_size, test_size = int(task["train_size"]), int(task["test_size"])
    seeds = [int(s) for s in task["split_seeds"]]
    draws = p10x_shard_draws(int(task["n_draws"]), task["shard"])
    ctx.log(
        f"shard {task['shard']['index']}/{task['shard']['count']}: "
        f"{len(draws)} of {task['n_draws']} draws x {len(task['raters'])} "
        f"raters = {len(draws) * len(task['raters'])} fits"
    )

    per_rater: dict = {rater: {
        "distributions": {
            key: [] for key in (
                "precision_macro", "recall_macro", "f1_macro", "accuracy",
                "pcc_top1", "iem_a_top1_mean",
            )
        },
        "floors": {
            "majority_class_f1_macro": [],
            "iem_constant_mean": [],
            # Registered as UNDEFINED rather than given a number: a
            # constant predictor has zero variance, so it has no PCC
            # (phase10_annex.FLOORS_REGISTERED). The string rides in
            # every rater's floors so no PCC figure is ever read
            # without it.
            "pcc": (
                "undefined -- a constant predictor has zero variance "
                "(phase10_annex.FLOORS_REGISTERED)"
            ),
        },
        "occupancy": [],
        "degenerate_flags": [],
    } for rater in task["raters"]}

    rows = []
    started = time.perf_counter()
    for draw_index in draws:
        split_seed = seeds[draw_index]
        train_rows, test_rows = _p10x_draw_rows(
            n, split_seed, train_size, test_size
        )
        for rater in task["raters"]:
            grades = by_rater[rater].astype(int)
            occupancy = _p10x_occupancy(grades[test_rows])
            degenerate = any(occupancy[str(g)] == 0 for g in range(1, 6))
            # The model seed is the CONFIG's, identical for every fit:
            # the annex varies the split and nothing else, so a moving
            # initialisation would confound the two.
            fit = _p10x_one_fit(
                images=images, grades=grades,
                train_rows=train_rows, test_rows=test_rows,
                recipe=task["recipe"], trainable=task["trainable"],
                model_seed=int(ctx.config["seed"]),
            )
            cell, floors = fit["metrics"], fit["floors"]
            slot = per_rater[rater]
            for key in slot["distributions"]:
                slot["distributions"][key].append(cell[key])
            slot["floors"]["majority_class_f1_macro"].append(
                floors["majority_class"]["f1_macro"]
            )
            slot["floors"]["iem_constant_mean"].append(
                floors["iem_constant_mean"]
            )
            slot["occupancy"].append(occupancy)
            slot["degenerate_flags"].append(degenerate)
            rows.append({
                "draw_index": draw_index,
                "split_seed": split_seed,
                "rater": rater,
                "degenerate_test_side": int(degenerate),
                **{f"occupancy_{g}": occupancy[str(g)] for g in range(1, 6)},
                "precision_macro": cell["precision_macro"],
                "recall_macro": cell["recall_macro"],
                "f1_macro": cell["f1_macro"],
                "accuracy": cell["accuracy"],
                "pcc_top1": "" if cell["pcc_top1"] is None else cell["pcc_top1"],
                "iem_a_top1_mean": cell["iem_a_top1_mean"],
                "floor_majority_f1_macro": floors["majority_class"]["f1_macro"],
                "floor_iem_constant_mean": floors["iem_constant_mean"],
            })
        ctx.log(
            f"draw {draw_index} (seed {split_seed}): "
            + ", ".join(
                f"{rater.split(' - ')[-1]} F1 "
                f"{per_rater[rater]['distributions']['f1_macro'][-1]:.4f}"
                for rater in task["raters"]
            )
        )
    elapsed = time.perf_counter() - started

    points = phase10_annex.PUBLISHED_REFERENCE_POINTS["points"]
    grid: dict = {}
    for rater, slot in per_rater.items():
        flags = slot["degenerate_flags"]
        f1 = slot["distributions"]["f1_macro"]
        slot["n_draws"] = len(flags)
        slot["degenerate"] = {
            "count": int(sum(flags)),
            "complete_count": int(len(flags) - sum(flags)),
            "frequency": (sum(flags) / len(flags)) if flags else 0.0,
            # Reported SEPARATELY, per EXIT_CRITERIA: a degenerate
            # draw's macro is an average over FEWER classes and is not
            # term-by-term comparable with a five-class one.
            "f1_macro_degenerate": [
                value for value, flag in zip(f1, flags) if flag
            ],
            "f1_macro_complete": [
                value for value, flag in zip(f1, flags) if not flag
            ],
        }
        pcc = slot["distributions"]["pcc_top1"]
        slot["pcc_undefined_draws"] = int(sum(1 for v in pcc if v is None))
        slot["published_position"] = {
            name: p10x_locate(pcc, value) for name, value in points.items()
        }
        grid[rater] = {
            name: position["where"]
            for name, position in slot["published_position"].items()
        }
        slot.pop("degenerate_flags")

    summary = {
        "arm": (
            "phase10_annex.DISTRIBUTION_ARM_REGISTERED -- 500 draws x 5 "
            "raters; the registered question is whether a single 153/28 "
            "split identifies performance at this cohort size"
        ),
        "prohibition": phase10_annex.ANNEX_PROHIBITION,
        "rater_caveat": (
            "five UNMATCHED raters -- sheet identities, never matched to "
            "their A-E (phase10_annex.RATER_PANEL_DISCREPANCY_EIGHTH); "
            "the caveat rides on every per-rater figure here"
        ),
        "normalisation_note": (
            "the notebook's CIFAR-10 statistics -- a deliberately "
            "replicated defect, replicated knowingly, never 'a recipe' "
            "(phase10_annex.NORMALISATION_RULED)"
        ),
        "supervision_dependency": (
            "[REPORTED, not measured] supervision has said the 28 test images "
            "come from the cohort; no 181-subset-of-237 assumption is "
            "made and these draws are from OUR cohort "
            "(phase10_annex.SUPERVISION_28_DEPENDENCY)"
        ),
        "shard": {
            "count": int(task["shard"]["count"]),
            "index": int(task["shard"]["index"]),
            "draws_in_shard": len(draws),
        },
        "split": {
            "root_seed": int(task["root_seed"]),
            "n_draws": int(task["n_draws"]),
            "seed_rule": P10X_SPLIT_SEED_RULE,
            "cohort_n": n,
            "train_size": train_size,
            "test_size": test_size,
            "unused_per_draw": n - train_size - test_size,
        },
        "fits_completed": len(rows),
        "wall_clock_seconds": {
            "total": elapsed,
            "per_fit_mean": elapsed / len(rows) if rows else None,
        },
        "per_rater": per_rater,
        "degenerate_note": (
            "phase10.top1_macro_prf EXCLUDES a truth-absent class from "
            "the macro average, so a degenerate draw's macro figure is "
            "an average over FEWER classes and is not term-by-term "
            "comparable with a five-class one. Degenerate draws are "
            "therefore reported separately as well as inside the "
            "all-draws distribution -- never discarded, never silently "
            "averaged in (phase10_annex.EXIT_CRITERIA)"
        ),
        "published_position_note": (
            "positions are computed against the PCC distributions only: "
            "the four citable published Study-Set reference points are "
            "PCCs, and no published Study macro-F1 exists in our record "
            "to locate against (phase10_annex."
            "PUBLISHED_REFERENCE_POINTS). Each position is a location "
            "within a distribution, never a matched rater-to-rater "
            "comparison"
        ),
        "published_points": points,
        "reading": p10x_apply_readings(grid),
        "model_selection": (
            "last epoch IS the model -- no validation split, no early "
            "stopping, no checkpoint selection "
            "(phase10.NOTEBOOK_BUDGET_AND_NORMALISATION)"
        ),
        "held_fixed_across_fits": (
            "model seed, recipe, normalisation, region count and "
            "trainability are identical in every fit -- ONLY the split "
            "varies, which is what makes this a measurement of the "
            "design rather than of anything else"
        ),
        "recipe": task["recipe"],
        "not_claimable": (
            "the annex characterises a DESIGN; no ledger row follows "
            "from it without its own registration and the standing "
            "two-condition criterion (phase10_annex.EXIT_CRITERIA)"
        ),
    }
    with ctx.atomic("metrics.json", tier="SHAREABLE") as tmp:
        tmp.write_text(
            json.dumps(as_builtin(summary), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    # Per-cell detail beside the aggregate: aggregate scalars only, no
    # patient identifiers, so it is SHAREABLE like metrics.json.
    with ctx.atomic("draws.csv", tier="SHAREABLE") as tmp:
        with tmp.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def task_iem_prestep(ctx: RunContext) -> None:
    """Phase 11's pre-step: the asymmetric error on predictions already on
    disk (``phase11.PRE_STEP_REGISTERED``). **Fits nothing.**

    Both direction conventions, every declared arm, ranked by IEM alone
    with the IEM-vs-PCC disagreement beside it. DESCRIPTIVE only: no
    claim, no ledger row for a ranking. Every registered exit criterion
    is computed here rather than asserted afterwards -- the crossover
    share, the residual skew, the power check, the [1,5] domain
    assertion.
    """
    import json

    from . import phase7c, phase11
    from .data.manifest import load_manifest
    from .eval import metrics

    task = ctx.config["task"]
    declared = {entry["name"]: entry for entry in ctx.inputs}
    manifest_dir = Path(declared[task["manifest_artifact"]]["path"])
    sheet_path = Path(declared[task["scoresheet_artifact"]]["path"])

    paths_by_arm = phase7c.oof_paths_from_inputs(declared)
    if not paths_by_arm:
        raise ValueError("no oof_<arm>_seed_<n> inputs declared")
    shared = sorted(int(s) for s in task["shared_seeds"])

    # Arms group by the seed band they actually declare: the transformer
    # arms ran five and the graph arms ten, and the loader takes one band
    # per call. Deriving the groups rather than assuming five is what
    # keeps a ten-seed arm from being scored on half its band in silence.
    bands: dict = {}
    for arm, by_seed in paths_by_arm.items():
        bands.setdefault(tuple(sorted(by_seed)), []).append(arm)
    ctx.log(
        f"{len(paths_by_arm)} arms, "
        f"{sum(len(v) for v in paths_by_arm.values())} vectors, "
        f"{len(bands)} seed bands: "
        + ", ".join(f"{len(a)} arms x {len(b)} seeds" for b, a in bands.items())
    )

    loaded: dict = {}
    patient_ids = truth_mean = None
    for band, arms in sorted(bands.items()):
        for seeds, tag in ((list(band), "own"), (shared, "shared5")):
            result = phase7c.load_oof_vectors(
                {arm: paths_by_arm[arm] for arm in arms}, seeds=seeds
            )
            # The loader cross-checks the truth column across every file
            # it reads; this cross-checks it BETWEEN calls, so two bands
            # scored against different labels cannot pass unnoticed.
            if patient_ids is None:
                patient_ids, truth_mean = result["patient_ids"], result["truth"]
            elif result["patient_ids"] != patient_ids:
                raise ValueError(
                    f"band {band} covers a different patient set from the "
                    "first band read"
                )
            else:
                drift = max(
                    abs(a - b) for a, b in zip(result["truth"], truth_mean)
                )
                if drift > 1e-9:
                    raise ValueError(
                        f"band {band} disagrees with the first band's TRUTH "
                        f"column by up to {drift:.3e}"
                    )
            for arm, cell in result["arms"].items():
                loaded.setdefault(arm, {})[tag] = cell

    # ---- G is the sheet's MEDIAN GRADE, not the CSVs' panel mean --------
    # phase11.PRE_STEP_REGISTERED: IEM is defined against a consensus
    # GRADE. The CSVs' truth column is the panel mean and is used ONLY as
    # the consistency check above -- never as G.
    rows = {int(r["patient_id"]): r for r in load_manifest(manifest_dir / "manifest.csv")}
    grades = median_by_patient(rows, sheet_path, patient_ids)
    ctx.log(
        f"G = the sheet's median grade on {len(grades)} patients "
        f"(mean {float(np.mean(grades)):.4f}); the CSVs' panel-mean truth "
        "column is the consistency check only, never G"
    )

    crossover = phase11.IEM_CROSSOVER
    per_arm: dict = {}
    domain_breaches: dict = {}
    for arm in sorted(loaded):
        cell: dict = {}
        for tag, band_cell in loaded[arm].items():
            by_seed = band_cell["by_seed"]
            seeds = band_cell["seeds"]
            residuals = np.array(
                [np.asarray(by_seed[s], dtype=float) - grades for s in seeds]
            )
            # Exit criterion 5: the domain [0,4] holds only if the
            # predictions do. Asserted, and a breach is a finding rather
            # than a silent clip.
            predictions = np.array([by_seed[s] for s in seeds], dtype=float)
            low, high = float(predictions.min()), float(predictions.max())
            if low < 1.0 or high > 5.0:
                domain_breaches[f"{arm}@{tag}"] = {
                    "min": low, "max": high,
                    "outside": int(
                        np.sum((predictions < 1.0) | (predictions > 5.0))
                    ),
                }
            per_convention = {}
            for convention in phase11.CONVENTIONS:
                by_seed_iem = [
                    float(np.mean(phase11.iem(row, convention)))
                    for row in residuals
                ]
                per_convention[convention] = {
                    "per_seed": by_seed_iem,
                    "mean": float(np.mean(by_seed_iem)),
                    "sd": float(np.std(by_seed_iem, ddof=1))
                    if len(by_seed_iem) > 1 else 0.0,
                }
            flat = residuals.reshape(-1)
            centred = flat - flat.mean()
            deviation = float(np.std(flat))
            cell[tag] = {
                "iem": per_convention,
                # The control that separates the target change from the
                # metric change (phase11.PRE_STEP_REGISTERED).
                "mae_vs_median": float(np.mean(np.abs(flat))),
                "pcc_vs_panel_mean": band_cell["mean"],
                "pcc_sd": band_cell["sd"],
                "n_seeds": band_cell["n_seeds"],
                "residual_skew": (
                    float(np.mean(centred ** 3) / deviation ** 3)
                    if deviation > 0 else float("nan")
                ),
                "below_crossover_share": float(
                    np.mean(np.abs(flat) < crossover)
                ),
                "prediction_range": (low, high),
            }
        per_arm[arm] = cell

    def _rank_block(tag: str) -> dict:
        arms = sorted(a for a in per_arm if tag in per_arm[a])
        iem_a = [
            per_arm[a][tag]["iem"]["a_manuscript_literal"]["mean"] for a in arms
        ]
        iem_b = [per_arm[a][tag]["iem"]["b_swapped"]["mean"] for a in arms]
        mae = [per_arm[a][tag]["mae_vs_median"] for a in arms]
        pcc_values = [per_arm[a][tag]["pcc_vs_panel_mean"] for a in arms]
        rank_a, rank_b = phase11.ranks(iem_a), phase11.ranks(iem_b)
        rank_mae = phase11.ranks(mae)
        # PCC is a similarity: negate so every ranking here is best-first.
        finite = [i for i, v in enumerate(pcc_values) if np.isfinite(v)]
        rank_pcc = phase11.ranks([-pcc_values[i] for i in finite])
        top5_a = {a for a, r in zip(arms, rank_a) if r <= 5}
        top5_b = {a for a, r in zip(arms, rank_b) if r <= 5}
        tau = phase11.kendall_tau_b(rank_a, rank_b)
        power = max(
            (abs(x - y) / abs(x) if x else float("inf"))
            for x, y in zip(iem_a, iem_b)
        )
        moved = top5_a != top5_b or tau < 0.90
        return {
            "arms": arms,
            "iem_a": iem_a, "iem_b": iem_b,
            "rank_a": rank_a, "rank_b": rank_b,
            "top5_a": sorted(top5_a), "top5_b": sorted(top5_b),
            "kendall_tau_b_a_vs_b": tau,
            "ranking_moved": bool(moved),
            "power_max_relative_gap": power,
            "power_sufficient": bool(power >= 0.01),
            "tau_iem_a_vs_mae": phase11.kendall_tau_b(rank_a, rank_mae),
            "tau_iem_a_vs_pcc": phase11.kendall_tau_b(
                [rank_a[i] for i in finite], rank_pcc
            ),
            "tau_mae_vs_pcc": phase11.kendall_tau_b(
                [rank_mae[i] for i in finite], rank_pcc
            ),
            "arms_with_non_finite_pcc": [
                arms[i] for i in range(len(arms)) if i not in finite
            ],
        }

    rankings = {tag: _rank_block(tag) for tag in ("own", "shared5")}
    for tag, block in rankings.items():
        # The registered readings, APPLIED -- not chosen now.
        if not block["power_sufficient"] and not block["ranking_moved"]:
            verdict = (
                "UNINFORMATIVE -- the power check failed (max relative gap "
                f"{block['power_max_relative_gap']:.4f} < 0.01), so a "
                "same-ranking result does NOT license reading 1"
            )
        elif block["ranking_moved"]:
            verdict = (
                "READING 2 -- the ranking MOVES between conventions, so "
                "the direction must be settled before any loss is built"
            )
        else:
            verdict = (
                "READING 1 -- the ambiguity does not bite for the ranking "
                "question; the pre-step's conclusion holds regardless of "
                "the supervision answer"
            )
        block["verdict"] = verdict
        ctx.log(f"[{tag}] tau {block['kendall_tau_b_a_vs_b']:.4f}; {verdict}")

    if domain_breaches:
        ctx.log(
            f"DOMAIN BREACH on {len(domain_breaches)} arm-bands: the "
            "paper's [0,4] does not hold by construction here "
            "(phase11.IEM_BOUNDS_ARE_A_DOMAIN) -- reported, not clipped"
        )

    with ctx.atomic("per_arm.csv", tier="SHAREABLE") as tmp:
        header = (
            "arm,band,n_seeds,iem_a,iem_a_sd,iem_b,iem_b_sd,mae_vs_median,"
            "pcc_vs_panel_mean,residual_skew,below_crossover_share,"
            "prediction_min,prediction_max"
        )
        lines = [header]
        for arm in sorted(per_arm):
            for tag in ("own", "shared5"):
                cell = per_arm[arm].get(tag)
                if cell is None:
                    continue
                a = cell["iem"]["a_manuscript_literal"]
                b = cell["iem"]["b_swapped"]
                lines.append(
                    f"{arm},{tag},{cell['n_seeds']},{a['mean']!r},"
                    f"{a['sd']!r},{b['mean']!r},{b['sd']!r},"
                    f"{cell['mae_vs_median']!r},"
                    f"{cell['pcc_vs_panel_mean']!r},"
                    f"{cell['residual_skew']!r},"
                    f"{cell['below_crossover_share']!r},"
                    f"{cell['prediction_range'][0]!r},"
                    f"{cell['prediction_range'][1]!r}"
                )
        tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")

    summary = {
        "registration": phase11.PRE_STEP_REGISTERED,
        "readings": phase11.PRE_STEP_READINGS,
        "exit_criteria": phase11.PHASE_11_EXIT_CRITERIA,
        "status": "DESCRIPTIVE ONLY -- no claim, no ledger row",
        "metric": {
            "equation": "manuscript eq (16), phase10.IEM_CARRIED_FOR_PHASE_11",
            "crossover": crossover,
            "crossover_finding": phase11.IEM_CROSSOVER_INVERTS,
            "bounds": phase11.IEM_BOUNDS_ARE_A_DOMAIN,
            "identity_open": phase11.CASE_IEM_IDENTITY_OPEN,
            "direction_inference": phase11.IEM_DIRECTION_INFERENCE,
        },
        "truth": {
            "G": "the score sheet's Median grade, verified row by row",
            "n": int(len(grades)),
            "not_the_panel_mean": (
                "the CSVs' truth column is the panel mean and was used "
                "ONLY as the cross-arm consistency check"
            ),
        },
        "per_arm": per_arm,
        "rankings": rankings,
        "domain_breaches": domain_breaches or "none -- every arm within [1,5]",
        "blocked": phase10_direction_block(),
    }
    with ctx.atomic("metrics.json", tier="SHAREABLE") as tmp:
        tmp.write_text(
            json.dumps(as_builtin(summary), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def phase10_direction_block() -> dict:
    """The block this phase's build waits on, carried into the run's own
    metrics so a reader of the artifact does not have to know to look."""
    from . import phase10

    return phase10.PHASE_11_BLOCKED_ON_IEM_DIRECTION


def task_iem_arm(ctx: RunContext) -> None:
    """Phase 11's loss arm, and its matched MSE control
    (``phase11.PHASE_11_BUILD_REGISTERED``). One task, one code path,
    ``task.loss`` the only thing that differs -- which is what makes the
    control matched rather than merely similar.

    Trains on the sheet's MEDIAN GRADE (what eq (16) is defined against),
    evaluates PCC against the PANEL MEAN (what everything else in this
    project is judged on), and reports BOTH readings always. The
    per-epoch diagnostic and its joint trigger ride the harness's own
    inner-val predictions at no extra forward.
    """
    import json

    from . import phase11
    from .data.manifest import load_manifest
    from .eval import metrics
    from .train import harness, phase3

    task = ctx.config["task"]
    loss = str(task["loss"])
    if loss not in phase11.LOSSES:
        raise ValueError(f"task.loss {loss!r} is not one of {phase11.LOSSES}")
    declared = {entry["name"]: entry for entry in ctx.inputs}
    manifest_dir = Path(declared[task["manifest_artifact"]]["path"])
    staged_dir = Path(declared[task["staged_artifact"]]["path"])
    embeddings_dir = Path(declared[task["embeddings_artifact"]]["path"])
    sheet_path = Path(declared[task["scoresheet_artifact"]]["path"])

    _, mean_labels, patient_ids, _ = phase3.load_inputs(
        manifest_dir, staged_dir, task["geometry"], task["label"],
        require_images=False,
    )
    rows = load_manifest(manifest_dir / "manifest.csv")
    assignments = {int(r["patient_id"]): int(r["fold"]) for r in rows}
    by_id = {int(r["patient_id"]): r for r in rows}
    features, _ = embeddings_module_load(embeddings_dir, patient_ids)

    # The TARGET is the median grade -- eq (16) is defined against a
    # consensus grade, and training against anything else would confound
    # the arm (phase11.PHASE_11_BUILD_REGISTERED). One resolution, shared
    # with Phase 10's arm.
    grades = median_by_patient(by_id, sheet_path, patient_ids)
    ctx.log(
        f"loss = {loss}; target = the sheet's median grade "
        f"(mean {float(np.mean(grades)):.4f}); evaluation = PCC against "
        "the panel mean, as everywhere"
    )

    seeds = [int(s) for s in task["seeds"]]
    mean_of = {int(p): float(m) for p, m in zip(patient_ids, mean_labels)}
    crossover = phase11.IEM_CROSSOVER
    per_seed: dict = {}
    curve_rows: list = []
    signatures: list = []
    pooled_pcc, pooled_iem = [], []
    from .cluster_csv import write_predictions

    for seed in seeds:
        created: list = []

        def make_head(_seed=seed, _created=created):
            head = phase11.make_head(
                loss=loss,
                learning_rate=float(task["learning_rate"]),
                weight_decay=float(task["weight_decay"]),
                seed=_seed,
            )
            _created.append(head)
            return head

        result = harness.run_cv(
            features=features, labels=grades, patient_ids=patient_ids,
            assignments=assignments, make_backbone=make_head,
            config=harness.TrainConfig(
                seed=seed,
                inner_val_frac=float(task["inner_val_frac"]),
                max_epochs=int(task["max_epochs"]),
                patience=int(task["patience"]),
                monitor=task["monitor"],
            ),
        )
        row_of = {int(p): i for i, p in enumerate(patient_ids)}
        truth = np.array([mean_of[int(p)] for p in result.oof_ids])
        predictions = np.asarray(result.oof_predictions, dtype=float)
        # BOTH readings, always, and the pair travels together: PCC
        # against the panel mean, IEM against the median grade the arm
        # actually trained on.
        seed_pcc = metrics.pcc(truth, predictions)
        oof_grades = grades[[row_of[int(p)] for p in result.oof_ids]]
        seed_iem = float(np.mean(phase11.iem(predictions - oof_grades)))
        pooled_pcc.append(seed_pcc)
        pooled_iem.append(seed_iem)

        row_of = {int(p): i for i, p in enumerate(patient_ids)}
        for fold_run, head in zip(result.folds, created):
            inner_rows = [row_of[int(p)] for p in fold_run.inner_val_ids]
            inner_labels = grades[inner_rows]
            for epoch in sorted(head.by_epoch):
                values = np.asarray(head.by_epoch[epoch], dtype=float)
                residual = values - inner_labels
                curve_rows.append(
                    f"{seed},{fold_run.fold},{epoch},"
                    f"{int(epoch == fold_run.selected_epoch)},"
                    f"{float(values.max() - values.min())!r},"
                    f"{float(np.mean(np.abs(residual) < crossover))!r},"
                    f"{float(metrics.pcc(inner_labels, values))!r}"
                )
            signature = phase11.degeneracy_signature(
                head.by_epoch, inner_labels, fold_run.selected_epoch
            )
            signature["seed"], signature["fold"] = seed, fold_run.fold
            signatures.append(signature)

        fold_of = {int(p): assignments[int(p)] for p in result.oof_ids}
        write_predictions(
            ctx.path(f"seed_{seed}__predictions.csv", tier="CLUSTER-ONLY"),
            zip(result.oof_ids, truth, predictions,
                (fold_of[int(p)] for p in result.oof_ids)),
        )
        per_seed[seed] = {
            "pcc_vs_panel_mean": seed_pcc,
            "iem_vs_median_grade": seed_iem,
            "selected_epochs": [f.selected_epoch for f in result.folds],
        }
        ctx.log(f"seed {seed}: PCC {seed_pcc:.4f}, IEM {seed_iem:.4f}")

    with ctx.atomic("degeneracy_curve.csv", tier="SHAREABLE") as tmp:
        header = (
            "seed,fold,epoch,selected,span,below_crossover_share,"
            "inner_val_pcc"
        )
        tmp.write_text("\n".join([header] + curve_rows) + "\n", encoding="utf-8")

    fired = [s for s in signatures if s.get("fired")]
    rule = phase11.DEGENERACY_APPLICATION_RULE
    ctx.log(
        f"degeneracy signature fired on {len(fired)} of {len(signatures)} "
        f"folds; condition 1 needs >= 13 "
        f"({'MET' if len(fired) >= 13 else 'not met'}). Condition 2 needs "
        "the matched control's count, which this run does not have"
    )

    summary = {
        "loss": loss,
        "registration": phase11.PHASE_11_BUILD_REGISTERED,
        "exit_criteria": phase11.PHASE_11_BUILD_EXIT_CRITERIA,
        "loss_form": phase11.IEM_LOSS_VERBATIM,
        "degeneracy_rule": rule,
        "defects_carried_with_every_number": {
            "value_inversion": phase11.IEM_CROSSOVER_INVERTS,
            "gradient_inversion": phase11.DEGENERACY_PULL_REGISTERED[
                "the_gradient_pull"
            ],
        },
        "target": "the sheet's median grade",
        "evaluation": "PCC against the panel mean",
        "per_seed": per_seed,
        "pooled": {
            "pcc_vs_panel_mean": phase3.seed_variance(pooled_pcc),
            "iem_vs_median_grade": phase3.seed_variance(pooled_iem),
        },
        "degeneracy": {
            "fired": len(fired),
            "folds": len(signatures),
            "condition_1_met": bool(len(fired) >= 13),
            "condition_2": (
                "needs the matched control's count -- this run is one arm "
                "of the pair, and the rule is applied across both"
            ),
            "per_fold": signatures,
        },
        "pairing": (
            "seed CSVs land in the baseline layout, so the IEM-vs-MSE "
            "paired BCa consumes them directly; the ladder comparison is "
            "DESCRIPTIVE only"
        ),
    }
    with ctx.atomic("metrics.json", tier="SHAREABLE") as tmp:
        tmp.write_text(
            json.dumps(as_builtin(summary), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


#: What every paired scope has always compared: PCC against the panel
#: mean the prediction CSVs carry. A pair that names no ``metrics`` gets
#: this one, which is why generalising cost no existing scope anything.
DEFAULT_PAIRED_METRIC = {"name": "pcc", "truth": "panel_mean"}


def paired_statistic(spec: dict, loaded: dict, declared: dict, task: dict):
    """``(statistic, truth)`` for one metric of a paired comparison.

    The METRIC is data in the pair enumeration -- a name and which truth
    vector it is defined against -- and this resolves it. Keeping the
    resolution here and the choice in the enumeration is the same split
    the pairs themselves use: the road's file says WHAT, this path does
    the arithmetic once.
    """
    from . import phase11
    from .data.manifest import load_manifest
    from .eval import metrics

    if spec["truth"] == "panel_mean":
        truth = loaded["truth"]
    elif spec["truth"] == "median_grade":
        # IEM is defined against a consensus GRADE, which the prediction
        # CSVs do not carry -- their truth column is the panel mean.
        manifest_dir = Path(declared[task["manifest_artifact"]]["path"])
        truth = median_by_patient(
            {
                int(row["patient_id"]): row
                for row in load_manifest(manifest_dir / "manifest.csv")
            },
            Path(declared[task["scoresheet_artifact"]]["path"]),
            loaded["patient_ids"],
        )
    else:
        raise ValueError(f"unknown paired truth {spec['truth']!r}")

    if spec["name"] == "pcc":
        return metrics.pcc, truth
    if spec["name"] == "iem":
        return (
            lambda g, p: float(
                np.mean(phase11.iem(np.asarray(p) - np.asarray(g)))
            ),
            truth,
        )
    raise ValueError(f"unknown paired statistic {spec['name']!r}")


def task_paired_claims(ctx: RunContext) -> None:
    """PLAN §4.3 condition 1 for the ladder's claims. Fits nothing.

    **[MEASURED 2026-08-04] Two of two phases audited had the same defect.**
    Phase 7C never computed condition 1 and it withdrew all nine of its
    verdicts; Phase 7B computed it, wrote it to its own metrics.json, and
    recorded "claimably worse" from condition 2 while condition 1 sat in the
    same file saying UNRESOLVED. The ladder's remaining claims are unchecked,
    and every one of them is a read away.

    **Seed bands are not uniform** -- transformer arms ran at five seeds and
    graph arms at ten -- so the run walks ``paired_claim_seed_groups`` rather
    than assuming one band. Truth is cross-checked WITHIN each group by the
    loader and BETWEEN groups here, because a label mismatch is the one
    corruption that would still produce ordinary-looking intervals.
    """
    import json

    from . import (
        ladder, phase7b, phase7c, phase10, phase11, phase12, phase27, roadb,
    )

    task = ctx.config["task"]
    scope = task["scope"]
    n_boot = int(task.get("n_boot", 10000))
    declared = {entry["name"]: entry for entry in ctx.inputs}

    # **The pair ENUMERATION is the only per-road part.** Road B's lives in
    # `roadb` because that road's structure belongs in its own file; the
    # derivations, the loader, the truth cross-check, the short-band refusal
    # and the BCa itself are this one path, walked by both roads. Writing a
    # second paired-comparison implementation is what R10 forbids -- and
    # this project has already paid for that class of duplication once.
    source = (
        roadb if scope.startswith("roadb_")
        # [2026-08-17] Phase 11's loss contrast joins on the same terms.
        else phase11 if scope == "p11_loss"
        # [2026-08-17] Phase 10's single pair joins on the same terms:
        # its enumeration lives in phase10 because that phase's structure
        # belongs in its own file, and everything downstream -- the
        # derivations, the loader, the truth cross-check, the short-band
        # refusal and the BCa -- is this one path
        # (phase10.PHASE_10_PAIRED_REGISTERED).
        else phase10 if scope == "p10"
        # [2026-08-23] Phase 12's three view contrasts, same terms.
        else phase12 if scope == "p12"
        # [2026-09-06] Phase 27's single pair, same terms. The arm is
        # the only one in this record fit on something other than the
        # cohort, and NONE of that reaches here: the pair is two stems
        # and five seeds like every other, which is the point of the
        # scope mechanism.
        else phase27 if scope == "p27"
        else ladder
    )
    pairs = source.paired_claim_pairs(scope)
    coverage = source.PAIRED_CLAIM_COVERAGE
    required = ladder.paired_claim_vectors(scope, pairs=pairs)
    stems = ladder.paired_claim_stems(scope, pairs=pairs)
    paths_by_stem = phase7c.oof_paths_from_inputs(declared)
    absent = [
        f"{stem}@{seed}" for stem, seed in required
        if seed not in paths_by_stem.get(stem, {})
    ]
    if absent:
        raise ValueError(
            f"{len(absent)} declared vectors missing, e.g. {absent[:5]}. "
            f"Scope {scope!r} needs {len(required)} across "
            f"{len(stems)} run directories"
        )
    ctx.log(
        f"SCOPE {scope}: {len(pairs)} pairs over "
        f"{len(stems)} run directories, "
        f"{len(required)} vectors, fits nothing"
    )

    groups = ladder.paired_claim_seed_groups(scope, pairs=pairs)
    truth_reference = None
    results, loaded_groups = {}, {}
    for seeds, group_pairs in sorted(groups.items()):
        stems = sorted({s for p in group_pairs for s in (p["a"], p["b"])})
        loaded = phase7c.load_oof_vectors(
            {stem: paths_by_stem[stem] for stem in stems}, seeds=list(seeds)
        )
        # **Between groups, not just within.** The loader guarantees one truth
        # per call; two calls could disagree, and a disagreement means the arms
        # were scored against different manifest columns -- which is exactly
        # why the label question is excluded (ladder.PAIRED_CLAIM_COVERAGE).
        if truth_reference is None:
            truth_reference = loaded["truth"]
        else:
            drift = max(
                abs(a - b) for a, b in zip(loaded["truth"], truth_reference)
            )
            if drift > 1e-9:
                raise ValueError(
                    f"seed group {seeds} is scored against a different truth "
                    f"vector, by up to {drift:.3e}. The arms were not "
                    "evaluated on the same target and nothing may be paired"
                )
        # **A short seed count must fail loudly, and here is why it is not
        # covered by the missing-vector check above.** Condition 1 is a
        # universal over the seeds PRESENT, so four intervals is a strictly
        # EASIER bar than ten -- a partial run that happens to satisfy it
        # would produce an ordinary-looking claim. A partial declared through
        # its own directory would already fail on missing files; this catches
        # the case where the paths resolve but the band is short anyway.
        # ladder.SIBLING_RUNS_AUDIT.
        for stem in stems:
            observed = loaded["arms"][stem]["n_seeds"]
            if observed != len(seeds):
                raise ValueError(
                    f"{stem} loaded {observed} seeds where the arm list says "
                    f"{len(seeds)}. Condition 1 is a universal over the seeds "
                    "present, so a short band is an EASIER bar -- refusing"
                )
        loaded_groups[seeds] = loaded
        ctx.log(f"  seeds {list(seeds)}: {len(stems)} arms loaded")
        for stem in stems:
            arm = loaded["arms"][stem]
            ctx.log(
                f"    {stem:44} pcc {arm['mean']:.4f} sd {arm['sd']:.4f} "
                f"n {arm['n_seeds']}"
            )

    for seeds, group_pairs in sorted(groups.items()):
        loaded = loaded_groups[seeds]
        for pair in group_pairs:
            a, b = pair["a"], pair["b"]
            # **[2026-08-17] The METRIC is part of the enumeration, not a
            # branch in the arithmetic.** A pair may name more than one
            # (phase11.PAIRED_CLAIM_COVERAGE: IEM alone would be marking
            # its own homework), and the default is the single PCC every
            # other scope has always run -- so this stays ONE call site
            # inside a loop rather than a second call site in an if,
            # which is what roadb's guard forbids and rightly.
            by_metric = {}
            for spec in pair.get("metrics", (DEFAULT_PAIRED_METRIC,)):
                statistic, metric_truth = paired_statistic(
                    spec, loaded, declared, task
                )
                winner = loaded["arms"][b]["by_seed"]
                by_metric[spec["name"]] = phase7b.paired_comparison(
                    truth=metric_truth,
                    winner_by_seed=winner,
                    baseline_by_seed=loaded["arms"][a]["by_seed"],
                    winner_sd=(
                        loaded["arms"][b]["sd"] if spec["name"] == "pcc"
                        else float(np.std(
                            [
                                statistic(metric_truth, winner[s])
                                for s in sorted(winner)
                            ],
                            ddof=1,
                        ))
                    ),
                    n_boot=n_boot,
                    statistic=statistic,
                    statistic_name=spec["name"],
                )
                by_metric[spec["name"]]["lower_is_better"] = bool(
                    spec.get("lower_is_better", False)
                )
                by_metric[spec["name"]]["truth_is"] = spec["truth"]
            paired = by_metric[
                pair.get("metrics", (DEFAULT_PAIRED_METRIC,))[0]["name"]
            ]
            condition_1 = bool(
                paired["n_excluding_zero"] == paired["n_seeds"]
                and paired["same_direction"]
            )
            margin = (
                abs(paired["mean_delta"]) / paired["threshold"]["arm_means_95"]
                if paired["threshold"]["arm_means_95"] else float("inf")
            )
            results[pair["key"]] = {
                "question": pair["question"], "varies": pair["varies"],
                "a": a, "b": b, "delta_is": f"{b} minus {a}",
                "n_seeds": len(seeds),
                "condition_1_paired_bca": condition_1,
                "condition_2_exceeds_threshold": bool(paired["exceeds_threshold"]),
                "claimable": bool(paired["claimable"]),
                "condition_2_margin": round(margin, 3),
                "recorded": pair.get("recorded"),
                "paired": paired,
                # Every scope gets this; for all but p11_loss it holds the
                # one PCC comparison "paired" already is.
                "by_metric": by_metric,
            }
            if len(by_metric) > 1:
                # The defects ride into the artifact rather than being
                # remembered by whoever reads it
                # (phase11.IEM_CROSSOVER_INVERTS).
                results[pair["key"]]["defects_carried"] = {
                    "value_inversion_below": phase11.IEM_CROSSOVER,
                    "gradient_inversion_below": 0.0719,
                    "record": (
                        "phase11.IEM_CROSSOVER_INVERTS, "
                        "phase11.DEGENERACY_PULL_REGISTERED"
                    ),
                }
            # **[FIXED 2026-08-17, phase11.THE_LOG_WAS_PARTIAL]** One line
            # PER METRIC. This printed only the first, so a p11 reader of
            # the log alone saw the PCC contrast and never learned the
            # artifact carried an IEM one -- an artifact that is right
            # while its log is partial is the same family as a
            # computed-then-discarded curve. The loop is over by_metric,
            # so a third metric would print itself without another fix.
            for name, comparison in by_metric.items():
                metric_condition_1 = bool(
                    comparison["n_excluding_zero"] == comparison["n_seeds"]
                    and comparison["same_direction"]
                )
                direction = (
                    "lower is better" if comparison.get("lower_is_better")
                    else "higher is better"
                )
                ctx.log(
                    f"  {pair['key'][:44]:44} [{name}] "
                    f"d {comparison['mean_delta']:+.4f}  "
                    f"{comparison['n_excluding_zero']}/"
                    f"{comparison['n_seeds']} exclude 0  "
                    f"cond1 {metric_condition_1}  cond2 "
                    f"{bool(comparison['exceeds_threshold'])}  "
                    f"=> {'CLAIMABLE' if comparison['claimable'] else 'unresolved'}"
                    f"  ({direction})"
                )

    # **Reported thinnest-first on the margin computed HERE**, not on the
    # recorded threshold -- the recorded numbers are the condition-2 values
    # under audit, and ranking the audit by them would be trusting the thing
    # being checked.
    order = sorted(results, key=lambda key: results[key]["condition_2_margin"])
    survived = [key for key in order if results[key]["claimable"]]
    withdrawn = [
        key for key in order
        if not results[key]["claimable"]
        and results[key]["condition_2_exceeds_threshold"]
    ]
    ctx.log(
        f"CONDITION 1: {len(survived)} of {len(results)} survive; "
        f"{len(withdrawn)} claimed on condition 2 and fail condition 1"
    )
    for key in withdrawn:
        ctx.log(f"  WITHDRAWN {key}")

    summary = {
        "scope": scope,
        "n_boot": n_boot,
        "n_pairs": len(results),
        "n_patients": len(truth_reference or []),
        "coverage": coverage,
        "by_margin_thinnest_first": order,
        "survived_condition_1": survived,
        "withdrawn_by_condition_1": withdrawn,
        "pairs": results,
    }
    with ctx.atomic("metrics.json", tier="SHAREABLE") as tmp:
        tmp.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )


def embeddings_module_load(path: Path, manifest_ids: list):
    from . import embeddings

    return embeddings.load(path, manifest_ids)


def task_extract_embeddings(ctx: RunContext) -> None:
    """Phase 6 §4: the required embedding sets, extracted over the 237.

    The set list is DERIVED from the arm list (``embedding_plan``) rather than
    cross-producted -- the shipped config enumerates exactly the derived sets,
    asserted by test, so what runs is written down and what is written down is
    what the arms need. Checkpoints arrive as declared file inputs named by
    ``extract.checkpoint_input_name``, so each set's provenance line runs
    config -> input -> pretraining run without a join.
    """
    import json

    from . import embedding_plan, embeddings
    from .data.manifest import load_manifest
    from .provenance.hashing import hash_dir
    from .train import extract, phase3

    task = ctx.config["task"]
    declared = {entry["name"]: entry for entry in ctx.inputs}
    manifest_dir = Path(declared[task["manifest_artifact"]]["path"])
    staged_dir = Path(declared[task["staged_artifact"]]["path"])

    # [2026-08-11] The declared init, by the conventional-name route
    # (pretrain.INIT_INPUT_NAME) -- the extract-task half of the init gate.
    # A declaration nothing consumes is refused HERE, before any file is
    # read: it would be hash-verified and recorded as feeding sets it never
    # fed. Per-set consumption rules (graphs refused, non-imagenet refused)
    # live in extract._load_model, at the failure site.
    from .train import pretrain as _pretrain_names

    init_entry = declared.get(_pretrain_names.INIT_INPUT_NAME)
    init_dir = Path(init_entry["path"]) if init_entry is not None else None
    if init_dir is not None:
        # [2026-08-11] The graph EXTRACTION route consumes declared inits
        # too (extract._load_graph_imagenet_init, measured mappings), so
        # the consumer test is any imagenet set at all.
        if not any(s["init"] == "imagenet" for s in task["sets"]):
            raise ValueError(
                f"{_pretrain_names.INIT_INPUT_NAME!r} is declared but no "
                "imagenet set in this config consumes it -- a provenance "
                "record claiming weights fed sets they did not feed."
            )

    out_root = ctx.repo_root / "data" / "embeddings" / task["out_version"]
    if out_root.exists():
        raise ValueError(
            f"{out_root} already exists. Data artifacts are immutable "
            "(PLAN §2.6): create a new version, never overwrite."
        )

    # The manifest read INDEPENDENTLY of load_inputs, so the row-order
    # assertion compares two reads rather than one read with itself.
    manifest_ids = [
        int(row["patient_id"]) for row in load_manifest(manifest_dir / "manifest.csv")
    ]

    # **Per-fold BN re-extraction turns each declared set into five.**
    # `extract.PER_FOLD_REEXTRACTION`: re-estimating the backbone's BatchNorm
    # statistics on cleft data makes the features depend on WHICH patients
    # were adapted on, so it must be done per fold on that fold's TRAINING
    # patients. Once over all 237 would put every test fold's statistics into
    # the representation used to predict it -- invisibly.
    per_fold = bool(task.get("per_fold_bn_reestimation", False))
    fold_of: dict[int, int] = {}
    if per_fold:
        from .data.manifest import load_manifest as _load_manifest

        fold_of = {
            int(row["patient_id"]): int(row["fold"])
            for row in _load_manifest(manifest_dir / "manifest.csv")
        }
        ctx.log(
            f"PER-FOLD BN re-extraction: {len(set(fold_of.values()))} folds, "
            "each adapted on its own training patients"
        )

    loaded: dict[str, tuple] = {}
    per_set = []
    for entry in task["sets"]:
        backbone = entry["backbone"]
        init = entry["init"]
        geometry = entry["geometry"]
        scheme = entry.get("pretrain_scheme")
        variant = embeddings.expected_variant(init, geometry)

        checkpoint_path = None
        checkpoint_hash = None
        if variant is not None:
            name = extract.checkpoint_input_name(
                {"backbone": backbone, "variant": variant, "pretrain_scheme": scheme}
            )
            if name not in declared:
                raise ValueError(
                    f"set {embedding_plan.set_name({**entry, 'pretrain_scheme': scheme})!r} "
                    f"needs a checkpoint input named {name!r}, which is not "
                    f"declared. Declared inputs: {sorted(declared)}"
                )
            checkpoint_path = Path(declared[name]["path"])
            # The VERIFIED hash (guard 3's own recomputation), not the
            # declaration -- the two agree or the run never started.
            checkpoint_hash = declared[name]["rollup"]

        if geometry not in loaded:
            loaded[geometry] = phase3.load_inputs(
                manifest_dir, staged_dir, geometry, "mean"
            )
        images, _, patient_ids, _ = loaded[geometry]

        # One pass for an ordinary set; one PER FOLD when re-estimating, each
        # adapted on that fold's TRAINING patients only.
        if per_fold:
            folds = sorted(set(fold_of[pid] for pid in patient_ids))
            passes = [
                (
                    fold,
                    np.array(
                        [
                            row
                            for row, pid in enumerate(patient_ids)
                            if fold_of[pid] != fold
                        ],
                        dtype=int,
                    ),
                )
                for fold in folds
            ]
        else:
            passes = [(None, None)]

        for fold, adapt_rows in passes:
            values, info = extract.extract_features(
                backbone, init, images,
                checkpoint_path=checkpoint_path,
                batch_size=task["batch_size"],
                adapt_rows=adapt_rows,
                block=entry.get("block"),
                token=entry.get("token"),
                # Only imagenet sets read it; _load_model refuses the rest.
                init_dir=init_dir if init == "imagenet" else None,
                # Arm B's component-role CONTROL (extract.RANDOMISED_BACKBONE).
                randomise=bool(entry.get("randomise", False)),
            )

            adapted_ids = (
                None
                if adapt_rows is None
                else [patient_ids[row] for row in adapt_rows]
            )
            set_entry = {
                **entry, "pretrain_scheme": scheme, "variant": variant,
                "fold": fold,
            }
            directory = out_root / embedding_plan.set_name(set_entry)
            # The mark comes from what the extraction REPORTED, not from what
            # the config asked for: a set named "randomised" whose features
            # are the real ones is the failure direction check_pairing's
            # second branch exists for.
            randomisation = info.get("randomisation")
            if bool(entry.get("randomise", False)) != bool(randomisation):
                raise ValueError(
                    f"set {directory.name!r} declares randomise="
                    f"{entry.get('randomise', False)!r} but the extraction "
                    f"reported randomisation={randomisation!r}. A control "
                    "and a real set must never be able to swap labels"
                )
            metadata = embeddings.save(
                directory, values,
                backbone=backbone,
                backbone_kind=info["backbone_kind"],
                init=init, geometry=geometry, variant=variant,
                checkpoint_sha256=checkpoint_hash,
                patient_ids=patient_ids, manifest_ids=manifest_ids,
                pretrain_scheme=scheme,
                fold=fold,
                adapted_on_patient_ids=adapted_ids,
                bn_reestimation=info.get("bn_reestimation"),
                block=entry.get("block"),
                token=entry.get("token"),
                randomisation=randomisation,
            )
            # **The leak, checked at BUILD time as well as at consumption.**
            # The consuming arm re-checks it (that is the one that reports the
            # number), but a set that leaks should never reach disk to be
            # found later.
            if fold is not None:
                embeddings.assert_no_test_fold_leak(
                    metadata,
                    [pid for pid in patient_ids if fold_of[pid] == fold],
                )
            ctx.log(
                f"  {directory.name}: kind {metadata['kind']}, "
                f"shape n={metadata['n_patients']} d={metadata['feature_dim']}"
                + ("" if fold is None else f", adapted on {len(adapted_ids)} rows")
            )
            per_set.append({
                "set": directory.name,
                "kind": metadata["kind"],
                "n_patients": metadata["n_patients"],
                "feature_dim": metadata["feature_dim"],
                "map_size": metadata.get("map_size"),
                "checkpoint_sha256": checkpoint_hash,
                "normalization_source": info.get("normalization_source"),
                "fold": fold,
                "n_adapted_on": None if adapted_ids is None else len(adapted_ids),
                "bn_reestimation": info.get("bn_reestimation"),
            })

    # **A produced set asserted BIT-FOR-BIT against an already-verified one.**
    # The pooling axis at block=depth, token=cls is the same representation
    # the final pooled extraction returns, so it is checked against a set that
    # already exists and whose hash guard 3 re-verified before this run
    # started. That validates the whole intermediate-block path against
    # known-good data, at build time, for free -- and it fails LOUDLY rather
    # than producing twelve plausible sets whose depth axis is subtly wrong.
    reproductions = []
    for claim in task.get("reproduces") or []:
        produced = out_root / claim["set"]
        if not produced.is_dir():
            raise ValueError(
                f"reproduces names {claim['set']!r}, which this run did not "
                f"produce. Produced: {sorted(p.name for p in out_root.iterdir())}"
            )
        if claim["input"] not in declared:
            raise ValueError(
                f"reproduces names input {claim['input']!r}, which is not "
                f"declared. Declared: {sorted(declared)}"
            )
        ours, _ = embeddings.load(produced)
        theirs, _ = embeddings.load(Path(declared[claim["input"]]["path"]))
        if ours.shape != theirs.shape or not np.array_equal(ours, theirs):
            raise ValueError(
                f"REPRODUCTION FAILED: {claim['set']} does not equal "
                f"{claim['input']}. Shapes {ours.shape} against {theirs.shape}. "
                "The intermediate-block path was expected to reproduce the "
                "final pooled extraction exactly at the last block with the "
                "class token; it does not, so every other depth in this batch "
                "is suspect and none of them should be consumed."
            )
        ctx.log(
            f"  REPRODUCES: {claim['set']} == {claim['input']} "
            f"(ARRAYS bit-identical, {ours.shape[0]} rows). Their directory "
            "ROLLUPS differ, and correctly so -- see the record."
        )
        reproductions.append({
            "set": claim["set"], "input": claim["input"],
            "bit_identical": True, "n_rows": int(ours.shape[0]),
            "compared": "values.npy, element for element",
            #: **The rollups of the two directories DIFFER, and that is
            #: correct.** A rollup covers every file including metadata.json,
            #: and the produced set records `block` and `token` where the
            #: reference records None -- so identical arrays land under
            #: different digests. Recorded here because a reader checking the
            #: two rollups instead of the two arrays would conclude the
            #: reproduction failed, and this is the artifact they would check.
            "why_the_rollups_differ": (
                "a rollup covers metadata.json, and this set records block "
                "and token where the reference does not. The arrays are what "
                "was compared and they are identical; rollups are not "
                "expected to match and their difference is not a failure"
            ),
        })

    rollup = hash_dir(out_root)
    summary = {
        "n_sets": len(per_set),
        "sets": per_set,
        "reproductions": reproductions,
        "out_version": task["out_version"],
        "artifact": {
            "path": f"data/embeddings/{task['out_version']}",
            "rollup_sha256_for_configs": rollup["rollup"],
            "file_count": rollup["file_count"],
            "total_bytes": rollup["total_bytes"],
        },
        "plan": embedding_plan.summary(),
        "row_order": "asserted against an independent manifest read, per set",
        "per_fold_bn_reestimation": per_fold,
        **(
            {
                "per_fold_note": (
                    "each set is adapted on its own fold's TRAINING patients "
                    "only; the test-fold leak is asserted at build AND at "
                    "consumption (extract.PER_FOLD_REEXTRACTION)"
                )
            }
            if per_fold
            else {}
        ),
    }
    ctx.path("metrics.json", tier="SHAREABLE").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    rendered = json.dumps(summary, indent=2, sort_keys=True)
    print(rendered)
    for line in rendered.splitlines():
        ctx.log(line)


def task_train_graph_cv(ctx: RunContext) -> None:
    """Phase 6/7: the graph cleft arm -- frozen feature maps, trained graph
    layers, the third regime. The seed-band gate runs through this task.

    Mirrors task_train_cv's shape: one job, one env.json, a seed list swept
    in-process with per-seed output prefixes, and gate-2-style seed variance
    reported when more than one seed runs.
    """
    import json

    from .train import graph_cleft, phase3
    from .train.harness import TrainConfig

    task = ctx.config["task"]
    declared = {entry["name"]: entry for entry in ctx.inputs}

    checkpoint_path = None
    checkpoint_sha256 = None
    if task.get("checkpoint"):
        if task["checkpoint"] not in declared:
            raise ValueError(
                f"task.checkpoint names input {task['checkpoint']!r}, which is "
                f"not declared. Declared inputs: {sorted(declared)}"
            )
        entry = declared[task["checkpoint"]]
        checkpoint_path = Path(entry["path"])
        # The VERIFIED hash -- guard 3 recomputed it before the run started.
        checkpoint_sha256 = entry["rollup"]

    seeds = [int(s) for s in (task.get("seeds") or [ctx.config["seed"]])]
    if len(seeds) != len(set(seeds)):
        raise ValueError(f"duplicate seed in {seeds}; each must give its own outputs")

    # One set for every fold, or one PER fold. Exactly one, and the loader
    # refuses both or neither -- an arm whose representation is ambiguous is
    # an arm nobody can describe afterwards.
    by_fold = task.get("embeddings_artifacts_by_fold")
    embeddings_dir = None
    embeddings_dirs = None
    if by_fold:
        missing = [name for name in by_fold.values() if name not in declared]
        if missing:
            raise ValueError(
                f"embeddings_artifacts_by_fold names undeclared input(s) "
                f"{missing}. Declared: {sorted(declared)}"
            )
        embeddings_dirs = {
            int(fold): Path(declared[name]["path"])
            for fold, name in by_fold.items()
        }
    else:
        if not task.get("embeddings_artifact"):
            raise ValueError(
                "declare embeddings_artifact (one set) or "
                "embeddings_artifacts_by_fold (one per fold)"
            )
        embeddings_dir = Path(declared[task["embeddings_artifact"]]["path"])

    # **[2026-08-14] `consume: nodes` is DECLARED but NOT WIRED, and this
    # refuses it here rather than after submission.**
    #
    # Road B Branch 3's graph arm crops regions BEFORE the backbone, so its
    # set is (n, regions, dim) and there is no map to pool them out of. The
    # path below is built for feature maps end to end: `pack` requires
    # (N, C, H, W), `boxes_for_arm` computes region boxes the node set
    # already embodies, and `make_backbone` constructs a graph head that
    # ROI-pools a map. Feeding nodes needs a second construction path, not
    # a reshape.
    #
    # Refused at LOAD time so the arm cannot sit in a queue and fail after
    # the wait -- the same reason _check_train_cv_init refuses at load.
    if task.get("consume") == "nodes":
        raise ValueError(
            "task.consume is 'nodes', which this task does not yet "
            "support: graph_cleft builds its nodes by ROI-pooling a "
            "feature map, and a region_vectors set has no map. The four "
            "other Branch 3 arms are unaffected and may run. See "
            "roadb.GRAPH_NODE_PATH_IS_NOT_WIRED for what remains."
        )

    common = dict(
        manifest_dir=Path(declared[task["manifest_artifact"]]["path"]),
        staged_dir=Path(declared[task["staged_artifact"]]["path"]),
        embeddings_dir=embeddings_dir,
        embeddings_dirs=embeddings_dirs,
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=checkpoint_sha256,
        backbone=task["backbone"],
        init=task["init"],
        geometry=task["geometry"],
        region_scheme=task["region_scheme"],
        label=task.get("label", "mean"),
        # The regime by default. "classifier" is the diagnostic that freezes
        # the graph layers too -- graph_cleft.FROZEN_GRAPH_DIAGNOSTIC, not a
        # ladder arm.
        trainable=task.get("trainable", "graph_layers"),
        # None means "not declared", which the run refuses for AG-Net. Every
        # other backbone is bitwise under the flag and leaves it alone.
        deterministic=task.get("deterministic"),
        backbone_config={
            "learning_rate": task.get("learning_rate", 1e-4),
            "weight_decay": task.get("weight_decay", 0.01),
            "batch_size": task.get("batch_size", 16),
        },
        log=ctx.log,
    )

    results, pooled = [], []
    for seed in seeds:
        ctx.log(f"--- seed {seed} ---")
        result = graph_cleft.run(
            **common,
            seed=seed,
            train_config=TrainConfig(
                max_epochs=task["max_epochs"],
                patience=task["patience"],
                inner_val_frac=task["inner_val_frac"],
                monitor=task["monitor"],
                seed=seed,
            ),
        )
        prefix = "" if len(seeds) == 1 else f"seed_{seed}__"
        phase3.write_outputs(result, ctx, prefix=prefix)
        results.append(result)
        pooled.append(result.summary["oof"]["pcc"])

    summary = results[0].summary if len(seeds) == 1 else {
        "seeds": seeds,
        "regime": "graph_third",
        "seed_band": graph_cleft.seed_variance(pooled),
        "pooled_pcc_by_seed": dict(zip(map(str, seeds), pooled)),
        "per_seed": {str(s): r.summary for s, r in zip(seeds, results)},
    }
    if len(seeds) > 1:
        ctx.path("seed_variance.json", tier="SHAREABLE").write_text(
            json.dumps(summary["seed_band"], indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    rendered = json.dumps(summary, indent=2, sort_keys=True)
    print(rendered)
    for line in rendered.splitlines():
        ctx.log(line)


def _p20_strata(ctx, task: dict, declared: dict, *, features=None):
    """Phase 20's strata, built ONCE before any seed.

    **The stratification is a locked design choice, not an estimate**
    (``phase20.STRATIFICATION_RULED``): the fitted value of the confound
    model over all five statistics, fitted IN-SAMPLE on all 237, cut into
    ``k_strata`` quantile bins. Fitting in-sample is legitimate precisely
    because nothing is read off the fit -- it decides only who may swap
    labels with whom. Per-seed out-of-fold strata would make the DESIGN
    move with the seed and confound it with the variation the arm exists
    to expose.

    Arm P is this at ``k = 1``: one stratum, everyone eligible. It is
    the same code path, not a second one.

    ``features`` may be supplied by a caller that has already computed
    the five statistics [ADDED 2026-08-31, for the stratification
    diagnostic], which needs them for its own fit as well. It is the
    SAME array either way -- ``_confound_statistics`` is deterministic --
    so this saves a second read of the staged tensor and changes no
    number.

    Returns ``(strata, report)``. The report carries the pre-registered
    fixed-point expectation and the degenerate-stratum census BEFORE any
    seed runs, so the prediction is on the record independently of what
    the counts turn out to be.
    """
    from . import phase20
    from .data.manifest import load_manifest
    from .train.ridge import RidgeBackbone

    block = task["permutation"]
    k = int(block["k_strata"])
    rows = load_manifest(declared[task["manifest_artifact"]] / "manifest.csv")
    patient_ids = [int(row["patient_id"]) for row in rows]

    if k == 1:
        # No fit is needed for one stratum, and running one would put a
        # ridge in the record that decides nothing.
        strata = phase20.quantile_strata(np.zeros(len(patient_ids)), 1)
        variable = "none -- Arm P is one stratum, everyone eligible"
    else:
        # The SAME five statistics the confound ceiling uses, through
        # the same extraction (_confound_statistics) -- not a second
        # copy, which is how two measurements quietly diverge.
        names = list(_decoder_statistic_names())
        if features is None:
            features, names = _confound_statistics(
                ctx, declared, task, patient_ids
            )
        labels = np.array([float(row[task["label"]]) for row in rows], dtype=float)
        model = RidgeBackbone(alpha=float(task.get("alpha", 1.0)), seed=0)
        model.reset(labels)
        model.train_epoch(features, labels)
        fitted = model.predict(features)
        strata = phase20.quantile_strata(fitted, k)
        variable = (
            f"the in-sample fitted value of a ridge over {len(names)} "
            f"statistics ({', '.join(names)}) against {task['label']!r}"
        )

    census = phase20.degenerate_strata(strata)
    report = {
        "registration": phase20.STRATIFICATION_RULED,
        "kind": block["kind"],
        "k_strata": k,
        "variable": variable,
        "target": task["label"],
        # Pre-registered, and written here BEFORE any seed draws, so the
        # prediction cannot be read as a description of the outcome.
        "expected_fixed_points": k,
        "expected_fixed_points_rule": (
            "E[fixed points] = 1 per stratum for ANY stratum size, so k "
            "strata expect exactly k (phase20.STRATIFICATION_RULED)"
        ),
        "degenerate_strata": census,
        "fixed_points_by_seed": {},
    }
    ctx.log(
        f"permutation: {block['kind']}, k = {k}, smallest stratum "
        f"{census['smallest']}, {census['n_immovable']} immovable; "
        f"expecting {k} unchanged labels per seed"
    )
    if census["n_immovable"]:
        ctx.log("  " + census["policy"])
    return strata, report


def task_p20_permutation(ctx: RunContext) -> None:
    """Phase 20 Arms P and S: the probe's own fit path, with the labels
    permuted (``phase20.ARM_P_REGISTERED``, ``STRATIFICATION_RULED``).

    **This function delegates and does nothing else, deliberately.** The
    control's whole value is that it is the SAME computation as the thing
    it controls: same features, same folds, same head, same pooling, same
    metric. A separate implementation here -- however careful -- would
    reintroduce exactly the defect Phase 7's probe was built to rule out
    (``phase3.PROBE_RECONSTRUCTED_THE_PIPELINE``), and a control that has
    drifted from its arm measures nothing.

    What makes it an arm is the config's ``permutation`` block, which
    ``task_train_cv`` reads. Nothing about the fit is decided here.
    """
    task_train_cv(ctx)


def task_p20_stratification_diagnostic(ctx: RunContext) -> None:
    """Phase 20: **did the stratification preserve what it exists to
    preserve?** (``phase20.STRATIFICATION_DIAGNOSTIC``.)

    Arm S shuffles labels within 10 quantile bins of the confound model's
    fitted value, which preserves confound structure **only between
    bins**. If the five statistics' predictive power lives substantially
    WITHIN bin, S destroyed the confound signal along with the cleft
    signal and measures **both-destroyed** rather than
    **cleft-destroyed** -- opposite conclusions from the same number,
    and S cannot tell them apart
    (``phase20.LOCK_LIMITATION_STRATIFICATION_UNVERIFIED``).

    **The measurement**: fit the five-statistic confound model against
    **Arm S's own permuted labels**, per seed, through **Arm C's recipe**
    -- closed-form ridge over ``cleft_v1``'s own folds, inner-val split
    per seed, alpha 1.0.

    **Nothing here is a second implementation.** The strata come from
    ``_p20_strata``, which is the function Arm S ran; the permutations
    from ``phase20.permute_within_strata`` at the same seeds; the
    statistics from ``_confound_statistics`` and the fit from
    ``_confound_ceiling_oof``, which are Arm C's. The label vector this
    scores is therefore byte-identically the one Arm S trained on -- by
    construction, not by resemblance.

    POD ARITHMETIC: five scalars and a closed-form ridge. No backbone, no
    GPU, no new artifact, no new hash -- and no gate, because nothing
    here generates or modifies a cohort pixel.
    """
    import json

    from . import phase20
    from .data.manifest import load_manifest

    task = ctx.config["task"]
    declared = {entry["name"]: Path(entry["path"]) for entry in ctx.inputs}

    rows = load_manifest(declared[task["manifest_artifact"]] / "manifest.csv")
    if len(rows) != int(task["expect_patients"]):
        raise ValueError(
            f"{len(rows)} manifest rows, expected {task['expect_patients']}"
        )
    patient_ids = [int(row["patient_id"]) for row in rows]
    fold_of = {int(row["patient_id"]): int(row["fold"]) for row in rows}
    # The SAME construction phase3.load_inputs uses -- a manifest column
    # read in manifest row order. A different order here would permute a
    # different vector and the numbers would still look plausible.
    labels = np.array([float(row[task["label"]]) for row in rows], dtype=float)

    features, names = _confound_statistics(ctx, declared, task, patient_ids)
    strata, strata_report = _p20_strata(ctx, task, declared, features=features)

    ceiling = float(task["ceiling_pcc"])
    inner_val_frac = float(task["inner_val_frac"])
    alpha = float(task["alpha"])
    by_seed, retained_by_seed = {}, {}
    for seed in (int(s) for s in task["seeds"]):
        permutation = phase20.permute_within_strata(strata, seed=seed)
        unchanged = phase20.fixed_points(permutation)
        strata_report["fixed_points_by_seed"][str(seed)] = unchanged
        scored = _confound_ceiling_oof(
            features, labels[permutation], patient_ids, fold_of,
            seeds=[seed], inner_val_frac=inner_val_frac, alpha=alpha,
            log=lambda _message: None,
        )[seed]
        by_seed[seed] = scored
        retained_by_seed[seed] = scored / ceiling
        ctx.log(
            f"  seed {seed}: confound PCC on Arm S's permuted labels "
            f"{scored:+.4f} = {retained_by_seed[seed] * 100:+.1f}% of the "
            f"{ceiling:+.4f} ceiling ({unchanged} labels unchanged)"
        )

    values = np.array(list(by_seed.values()))
    mean_pcc = float(values.mean())
    sd_pcc = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    retained = mean_pcc / ceiling
    fractions = np.array(list(retained_by_seed.values()))
    retained_sd = float(fractions.std(ddof=1)) if len(fractions) > 1 else 0.0

    survived_at = float(task["survived_fraction"])
    destroyed_at = float(task["destroyed_fraction"])
    if retained >= survived_at:
        cell = "near_the_ceiling"
    elif retained <= destroyed_at:
        cell = "near_zero"
    else:
        cell = "between"
    ctx.log(
        f"confound signal on the stratified permutation: mean {mean_pcc:+.4f} "
        f"(sd {sd_pcc:.4f}) against the {ceiling:+.4f} ceiling -- "
        f"{retained * 100:+.1f}% retained (sd {retained_sd * 100:.1f} points), "
        f"declared cells at <={destroyed_at} and >={survived_at}: {cell.upper()}"
    )
    reading = _registered_reading(
        ctx,
        reading=phase20.DIAGNOSTIC_READINGS_COMMITTED[cell],
        joined=len(patient_ids), expected=int(task["expect_patients"]),
        what="Phase 20, the stratification diagnostic",
    )
    # **The binding is logged on EVERY branch, including the favourable
    # one.** It is where the temptation is: "S is interpretable" is not
    # "S is the answer" (phase20.DIAGNOSTIC_BINDING).
    ctx.log("BINDING: " + phase20.DIAGNOSTIC_BINDING["the_binding"])
    ctx.log("  " + phase20.DIAGNOSTIC_BINDING["what_stands_in_the_way_even_then"])

    # The counts that prove this scored ARM S's permutation and not a
    # look-alike. They are written where a reader can check them, with
    # the comparison named -- an equality nobody is told to check is one
    # nobody checks.
    strata_report["cross_check"] = (
        "fixed_points_by_seed and degenerate_strata MUST equal Arm S's "
        "permutation.json exactly. If they differ, this run scored a "
        "different permutation from the one Arm S trained on and its "
        "number says nothing about Arm S"
    )
    summary = {
        "registration": phase20.STRATIFICATION_DIAGNOSTIC,
        "binding": phase20.DIAGNOSTIC_BINDING,
        "statistics": names,
        "n_patients": len(patient_ids),
        "label": task["label"],
        "pcc_by_seed": {str(s): v for s, v in by_seed.items()},
        "pcc_mean": mean_pcc,
        "pcc_sd": sd_pcc,
        "ceiling_pcc": ceiling,
        "retained_fraction_by_seed": {
            str(s): v for s, v in retained_by_seed.items()
        },
        "retained_fraction_mean": retained,
        "retained_fraction_sd": retained_sd,
        "declared_cells": {
            "near_zero_at_or_below": destroyed_at,
            "near_the_ceiling_at_or_above": survived_at,
        },
        "cell_fired": cell,
        "reading": reading,
        "permutation": strata_report,
        "status": (
            "DESCRIPTIVE, REPORT-NEVER-GATE: no ladder entry, no ledger "
            "row (phase20.LEDGER_RULED), no PLAN 4.3 machinery. It "
            "decides whether Arm S is INTERPRETABLE, not what Arm S shows"
        ),
    }
    with ctx.atomic("metrics.json", tier="SHAREABLE") as tmp:
        tmp.write_text(
            json.dumps(as_builtin(summary), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    ctx.path("permutation.json", tier="SHAREABLE").write_text(
        json.dumps(as_builtin(strata_report), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def task_train_cv(ctx: RunContext) -> None:
    """Phase 3: the minimal path, with gates 3-6 running inside the run."""
    import time

    from . import phase20
    from .train import phase3
    from .train.harness import TrainConfig

    from .train.pooled import PooledSource

    task = ctx.config["task"]
    entries = {entry["name"]: entry for entry in ctx.inputs}
    declared = {name: Path(entry["path"]) for name, entry in entries.items()}

    # Phase 7: the ladder arms consume the POOLED artifacts extraction
    # produced. Without this the arm runs live extraction, which builds
    # ViT-B/16 at ImageNet init and nothing else -- so no pretrained init was
    # reachable at all. The declarations here are CHECKED against the
    # artifact's own metadata, never trusted (train.pooled).
    pooled_source = None
    if task.get("concat_embeddings_artifacts"):
        # Arm 4: several pooled sets, one per declared backbone, in the
        # declared order. Each becomes its own PooledSource so every
        # constituent is row-order- and pairing-checked against its OWN
        # backbone (pooled.load_concat_features); the schema has already
        # refused this field on any backbone but "concat".
        sources = []
        for entry in task["concat_embeddings_artifacts"]:
            if entry["artifact"] not in entries:
                raise ValueError(
                    f"concat_embeddings_artifacts names input "
                    f"{entry['artifact']!r}, which is not declared. Declared "
                    f"inputs: {sorted(entries)}"
                )
            sources.append(PooledSource(
                directory=declared[entry["artifact"]],
                backbone=entry["backbone"],
                init=task.get("init", "imagenet"),
                geometry=task["geometry"],
                checkpoint_sha256=None,
                consume="pooled",
            ))
        pooled_source = sources
    elif task.get("embeddings_artifact"):
        name = task["embeddings_artifact"]
        if name not in entries:
            raise ValueError(
                f"task.embeddings_artifact names input {name!r}, which is not "
                f"declared. Declared inputs: {sorted(entries)}"
            )
        checkpoint_sha256 = None
        if task.get("checkpoint"):
            if task["checkpoint"] not in entries:
                raise ValueError(
                    f"task.checkpoint names input {task['checkpoint']!r}, which "
                    f"is not declared. Declared inputs: {sorted(entries)}"
                )
            # The VERIFIED hash -- guard 3 recomputed it before the run
            # started -- never the config's own declaration of it.
            checkpoint_sha256 = entries[task["checkpoint"]]["rollup"]
        pooled_source = PooledSource(
            directory=declared[name],
            backbone=task["backbone"],
            init=task.get("init", "imagenet"),
            geometry=task["geometry"],
            checkpoint_sha256=checkpoint_sha256,
            # Declared, never inferred: one region_vectors set feeds both
            # a concat arm and a graph arm (embeddings.as_consumed).
            consume=task.get("consume") or "pooled",
        )

    # One job, one environment, one env.json -- and one run directory per seed's
    # outputs by filename prefix. Ten separate workload creations would give ten
    # env.json files describing the same environment while making it easy for two
    # to differ in something nobody recorded.
    seeds = [int(s) for s in (task.get("seeds") or [ctx.config["seed"]])]
    if len(seeds) != len(set(seeds)):
        raise ValueError(f"duplicate seed in {seeds}; each must give its own outputs")

    common = dict(
        manifest_dir=declared[task["manifest_artifact"]],
        staged_dir=declared[task["staged_artifact"]],
        geometry=task["geometry"],
        label=task["label"],
        backbone=task["backbone"],
        trainable=task.get("trainable", "head"),
        patch_scheme=task.get("patch_scheme", "whole"),
        feature_source=task.get("feature_source", "mirror"),
        pooled_source=pooled_source,
        backbone_config={
            "learning_rate": task.get("learning_rate", 1e-3),
            "weight_decay": task.get("weight_decay", 0.01),
            "batch_size": task.get("batch_size", 16),
            "alpha": task.get("alpha", 1.0),
            "pooling": task.get("pooling", "mean"),
            #: **[MEASURED 2026-08-03] This key was missing and every Phase 7C
            #: arm ran unaugmented.** ``backbone_config`` is a WHITELIST, and
            #: ``phase3.prepare_features`` and ``make_factory`` both read
            #: ``augmentation`` out of it -- so the policy reached the run
            #: record and never the execution, and all seven arms produced
            #: byte-identical predictions.
            #:
            #: ``test_workflow_hygiene`` now asserts that every key phase3
            #: reads from this dict is supplied here, which is the general
            #: form of the defect.
            "augmentation": task.get("augmentation"),
        },
        log=ctx.log,
    )

    # **[2026-08-31] Phase 20's permutation control.** Declared, never
    # inferred; absent, everything below is byte-for-byte what it was.
    # The strata are built ONCE, before any seed, because the
    # stratification is a locked DESIGN CHOICE and must not move with
    # the seed (phase20.STRATIFICATION_RULED).
    strata = permutation_report = None
    if task.get("permutation"):
        strata, permutation_report = _p20_strata(ctx, task, declared)

    results, pooled, wall_clock = [], [], {}
    for seed in seeds:
        ctx.log(f"--- seed {seed} ---")
        label_permutation = None
        if strata is not None:
            label_permutation = phase20.permute_within_strata(strata, seed=seed)
            unchanged = phase20.fixed_points(label_permutation)
            permutation_report["fixed_points_by_seed"][str(seed)] = unchanged
            ctx.log(
                f"  permutation: {unchanged} of {len(strata)} labels "
                f"unchanged ({unchanged / len(strata) * 100:.2f}%), against "
                f"the pre-registered expectation of "
                f"{permutation_report['expected_fixed_points']} "
                "-- REPORTED, never redrawn"
            )
        started = time.perf_counter()
        result = phase3.run(
            **common,
            seed=seed,
            label_permutation=label_permutation,
            train_config=TrainConfig(
                max_epochs=task["max_epochs"],
                patience=task["patience"],
                inner_val_frac=task["inner_val_frac"],
                monitor=task["monitor"],
                seed=seed,
            ),
        )
        # [2026-08-31] The compute question, ruled: the timing is folded
        # into the runs rather than measured by a separate gate job
        # (phase20.COMPUTE_GATE_DESIGNED["ruled_2026_08_31"]). It costs
        # nothing and it is a measurement rather than an expectation.
        elapsed = time.perf_counter() - started
        wall_clock[str(seed)] = round(elapsed, 3)
        ctx.log(f"  seed {seed}: {elapsed:.1f}s wall-clock")
        prefix = "" if len(seeds) == 1 else f"seed_{seed}__"
        phase3.write_outputs(result, ctx, prefix=prefix)
        results.append(result)
        pooled.append(result.summary["oof"]["pcc"])

    import json

    summary = results[0].summary if len(seeds) == 1 else {
        "seeds": seeds,
        "gate2_seed_variance": phase3.seed_variance(pooled),
        "pooled_pcc_by_seed": dict(zip(map(str, seeds), pooled)),
        "per_seed": {str(s): r.summary for s, r in zip(seeds, results)},
    }
    summary["wall_clock_seconds_by_seed"] = wall_clock
    if permutation_report is not None:
        # Its OWN file, for the reason gate1_reproduction.json has one
        # (the [DEFECT 2026-08-01] note below): at a single seed
        # `summary` IS results[0].summary, whose metrics.json was
        # already written, so anything assigned here would reach the log
        # and never the record. The fixed-point counts are the phase's
        # pre-registered prediction and cannot be the thing that goes
        # missing.
        permutation_report["wall_clock_seconds_by_seed"] = wall_clock
        summary["permutation"] = permutation_report
        ctx.path("permutation.json", tier="SHAREABLE").write_text(
            json.dumps(
                as_builtin(permutation_report), indent=2, sort_keys=True
            ) + "\n",
            encoding="utf-8",
        )
    if len(seeds) > 1:
        ctx.path("seed_variance.json", tier="SHAREABLE").write_text(
            json.dumps(summary["gate2_seed_variance"], indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    # Gate 1, re-checked. Only the arm that PRODUCED the reference can reproduce
    # it -- for any other arm the numbers are expected to differ, and comparing
    # them would be the category error this project keeps finding. So the
    # comparison is made when the arm matches, and its absence is recorded with
    # the reason rather than left as a missing key.
    reference_arm = (
        task.get("patch_scheme", "whole") == "whole"
        and task["geometry"] == "g1"
        and task["backbone"] == "vit_b16"
        and task.get("trainable", "head") == "head"
        and task["label"] == "mean"
        and phase3.GATE1_REFERENCE["seed"] in seeds
        # **[2026-08-31] A permuted arm matches the reference on every
        # other field.** Arm P differs from the probe in the permutation
        # block ALONE -- which is the point -- so without this it would
        # claim the gate-1 comparison and be scored against the probe's
        # fingerprint. It would DIFFER, correctly, and the run would
        # report a determinism failure for a run that is behaving
        # exactly as designed.
        and not task.get("permutation")
        # And it extracted its own features. The reference was measured on
        # live extraction; an artifact-fed arm reaches its vectors through
        # extract.py instead, so even at imagenet init the two are not the
        # same computation and a mismatch would say nothing about
        # determinism. Every other field can match while this one decides.
        and pooled_source is None
    )
    if reference_arm:
        index = seeds.index(phase3.GATE1_REFERENCE["seed"])
        summary["gate1_reproduction"] = phase3.compare_to_gate1(
            results[index].summary["fingerprint"]
        )
        ctx.log(
            "gate 1 reproduction: "
            f"{'MATCHES' if summary['gate1_reproduction']['matches'] else 'DIFFERS'}"
        )
    else:
        summary["gate1_reproduction"] = {
            "checked": False,
            "why": (
                "GATE1_REFERENCE was measured on the whole-image vit_b16 head arm "
                "at g1 on the mean label, seed 1337, EXTRACTING ITS OWN FEATURES. "
                "This arm differs, so its numbers are expected to differ and "
                "comparing them would say nothing about determinism. Determinism "
                "for THIS arm is checked by running it twice and comparing "
                "fingerprint.predictions_sha256."
            ),
            "consumes_pooled_artifact": pooled_source is not None,
            # Named, so a Phase 20 arm's absent gate-1 check reads as a
            # design decision rather than an omission.
            "labels_permuted": bool(task.get("permutation")),
        }

    # **The verdict goes to its own SHAREABLE file, and it has to.**
    #
    # [DEFECT 2026-08-01] Everything above assigns into `summary` AFTER
    # `phase3.write_outputs` has already written metrics.json. At one seed
    # `summary` IS `results[0].summary`, so the mutation lands on an object
    # whose file was written moments earlier; at more than one seed `summary`
    # is a fresh dict that no code ever writes. Either way
    # `gate1_reproduction` reached **stdout and log.txt only** -- never
    # metrics.json, which is the file PLAN §2.4 makes the run's record and the
    # one a reader consults.
    #
    # That is this project's own failure shape in the most awkward possible
    # place: the determinism re-verification is closed by READING this
    # verdict, and the verdict was not being kept. The comment above even says
    # its absence is "recorded with the reason rather than left as a missing
    # key" -- into a dict that was then discarded.
    #
    # A dedicated file rather than a late re-write of metrics.json: it works
    # identically at one seed and at ten, it cannot be undone by ordering, and
    # it mirrors seed_variance.json, which exists for the same reason.
    ctx.path("gate1_reproduction.json", tier="SHAREABLE").write_text(
        json.dumps(summary["gate1_reproduction"], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    rendered = json.dumps(summary, indent=2, sort_keys=True)
    print(rendered)
    for line in rendered.splitlines():
        ctx.log(line)


from . import roadb_tasks as _roadb_tasks


def task_p26_calibration_table(ctx: RunContext) -> None:
    """Phase 26's six readouts, together, for every cell.

    **The union is the deliverable.** No record in this project reports
    PCC, shrinkage, RMSE, MAE, three-class accuracy and IEM for the same
    arm, and the maximum co-occurrence anywhere is three, at one seed
    (``phase26.THE_RECKONING``). Four come from the training path and
    three-class accuracy is a consolidation of shipped code. **IEM is the
    one gap**, so it is computed here through ``phase18.iem_score``,
    which is ``phase11.iem`` under convention A, reused and never
    reimplemented so the three measured defects stay attached to the
    figures they qualify.

    **The floor is recomputed here rather than transcribed.** It exists
    in the record only as a function whose one output went to a
    cluster-only file, and criterion 2 requires a floor beside every IEM
    figure. Each cell carries a floor from this run's own execution, and
    the Phase 18 value is compared against it when declared.

    **The anchor cell is a gate, not a result.** Weight decay 0.01 with
    patience 5 is the probe's own recipe and must reproduce its banked
    figure. If it does not, the run is wrong and no cell is read.

    DESCRIPTIVE. No ledger row unless a contrast is claimable, which the
    reckoning predicts none will be.
    """
    import json

    from . import phase18, phase26
    from .classification import prf_report
    from .data.manifest import load_manifest
    from .eval.metrics import mae, pcc, rmse, to_3class

    task = ctx.config["task"]
    declared = {entry["name"]: entry for entry in ctx.inputs}

    # **The runtime entry carries ``declared_rollup``.**
    # ``rollup_sha256`` is the CONFIG field name, and reading it here
    # would raise a KeyError on the cluster after the queue wait. That
    # is the Phase 25 class of defect, and the shipped hygiene guard
    # caught it before this ever ran.
    placeholder = "0" * 64
    pending = sorted(
        entry["name"] for entry in ctx.inputs
        if entry["declared_rollup"] == placeholder
    )
    if pending:
        raise ValueError(
            f"{len(pending)} run directory declaration(s) are still "
            f"placeholders: {pending}. Fill them with "
            "scripts/declare_inputs.py once the cells have run. The table "
            "refuses rather than reading a directory nobody declared."
        )

    rows = load_manifest(
        Path(declared[task["manifest_artifact"]]["path"]) / "manifest.csv"
    )
    patient_ids = [int(r["patient_id"]) for r in rows]
    if len(patient_ids) != int(task["expect_patients"]):
        raise ValueError(
            f"{len(patient_ids)} manifest rows against the declared "
            f"{task['expect_patients']}"
        )
    truth = np.array([float(r["mean"]) for r in rows])

    levels = {
        arm["name"]: (float(arm["weight_decay"]), int(arm["patience"]))
        for arm in task["arms"]
    }
    if len(levels) != int(task["expect_cells"]):
        raise ValueError(
            f"{len(levels)} cells declared against the expected "
            f"{task['expect_cells']}"
        )
    grid = phase26.THE_GRID_LOCKED
    declared_pairs = {
        (float(w), int(p))
        for w in grid["weight_decay_levels"] for p in grid["patience_levels"]
    }
    if set(levels.values()) != declared_pairs:
        raise ValueError(
            "the declared cells are not the locked grid. The grid is "
            "fixed at phase26.THE_GRID_LOCKED and a config may not add, "
            "drop or move a cell."
        )

    by_seed = _p21_load_arm_predictions(ctx, task, declared, patient_ids)
    seeds = sorted(by_seed)

    # **The floor, from this run's own execution.** It depends only on
    # the truth vector, so it is constant across cells by construction
    # and computing it once per cell is also a self-consistency check.
    floor = float(phase18.constant_predictor_iem(truth))
    sd_truth = float(np.std(truth, ddof=1))

    cells = {}
    for name in sorted(levels):
        per_seed = {"pcc": [], "shrinkage": [], "rmse": [], "mae": [],
                    "acc3": [], "iem": []}
        for seed in seeds:
            predicted = by_seed[seed][name]
            per_seed["pcc"].append(float(pcc(truth, predicted)))
            per_seed["rmse"].append(float(rmse(truth, predicted)))
            per_seed["mae"].append(float(mae(truth, predicted)))
            per_seed["shrinkage"].append(
                float(np.std(predicted, ddof=1)) / sd_truth
            )
            report = prf_report(
                to_3class(truth), to_3class(predicted, clip=True), 3
            )
            per_seed["acc3"].append(float(report["accuracy"]))
            per_seed["iem"].append(float(phase18.iem_score(truth, predicted)))

        decay, patience = levels[name]
        cells[name] = {
            "weight_decay": decay,
            "patience": patience,
            "per_seed": per_seed,
            "mean": {k: float(np.mean(v)) for k, v in per_seed.items()},
            "sd": {
                k: float(np.std(v, ddof=1)) if len(v) > 1 else 0.0
                for k, v in per_seed.items()
            },
            #: Criterion 2. Every IEM figure carries its floor.
            "iem_floor": floor,
        }

    # **Criterion 9: the anchor cell is the probe or the run is wrong.**
    anchor = task["anchor_cell"]
    if anchor not in cells:
        raise ValueError(f"the anchor cell {anchor!r} is not declared")
    got = cells[anchor]["mean"]["pcc"]
    want = float(task["anchor_pcc"])
    tolerance = 2.0 * float(task["anchor_sd"])
    anchor_ok = abs(got - want) <= tolerance
    if not anchor_ok:
        raise ValueError(
            f"the anchor cell {anchor!r} scored {got:.4f} against the "
            f"probe's declared {want:.4f}, outside two of its own seed sd "
            f"({tolerance:.4f}). The cell that IS the probe must "
            "reproduce the probe, so the run is wrong and no cell is read "
            "(phase26.EXIT_CRITERIA criterion 9)."
        )

    # **Criterion 2's cross-check.** Reported either way, and its absence
    # is reported as an absence rather than passed over.
    recovered = task.get("phase18_floor_iem")
    cross_check = {
        "recomputed_here": floor,
        "phase18_recovered": None if recovered is None else float(recovered),
        "agree": None,
        "note": (
            "the Phase 18 floor was NOT declared, so the cross-check is "
            "UNAVAILABLE rather than passed. Recover it from that run's "
            "metrics.json under floors.iem_constant_predictor and declare "
            "it as task.phase18_floor_iem."
        ),
    }
    if recovered is not None:
        delta = abs(floor - float(recovered))
        cross_check["agree"] = bool(delta < 1e-9)
        cross_check["absolute_difference"] = float(delta)
        cross_check["note"] = (
            "two independent computations of the same quantity. Agreement "
            "establishes the floor twice by different paths. A "
            "disagreement means the truth vector or the scorer differs "
            "between the two phases, which is itself a finding and not a "
            "failure of this phase."
        )

    # **Criterion 6.** Per-fit wall clock, read from each cell's own run
    # summary. Reported as unavailable per cell rather than defaulted,
    # because a missing timing is not a zero.
    #
    # **[DEFECT, FOUND 2026-09-06, NOT FIXED] this returns None for
    # every cell and always will.** Nothing in this repository writes a
    # file named ``summary.json``. ``task_train_cv`` computes
    # ``wall_clock_seconds_by_seed`` and renders it to the LOG, so the
    # timing is not readable from any artifact. Phase 26 closed its
    # criterion 6 by reading the sixteen run logs instead, and banked
    # the result at ``phase26.THE_RUNTIME_MEASURED``. **The fix is
    # owed and is not made here**: see
    # ``phase26.WHAT_CRITERION_3_DID_NOT_GET`` for the defect, its
    # class, and what it cost. A caller wanting timing from this task
    # must not read the null as a zero or as an absence of cost.
    wall_clock = {}
    for name in sorted(levels):
        summary = Path(declared[name]["path"]) / "summary.json"
        if not summary.is_file():
            wall_clock[name] = None
            continue
        payload = json.loads(summary.read_text(encoding="utf-8"))
        by_seed_clock = payload.get("wall_clock_seconds_by_seed")
        if not by_seed_clock:
            wall_clock[name] = None
            continue
        values = [float(v) for v in dict(by_seed_clock).values()]
        wall_clock[name] = {
            "seconds_by_seed": values,
            "seconds_total": float(sum(values)),
            #: The unit is the FOLD FIT. A per-cell total hides whether
            #: seeds or folds are doing the work.
            "seconds_per_fold_fit": float(sum(values)) / (len(values) * 5),
        }

    metrics = {
        "cells": cells,
        "iem_floor": floor,
        "floor_cross_check": cross_check,
        "anchor": {
            "cell": anchor, "declared": want, "measured": got,
            "within_two_seed_sd": anchor_ok,
        },
        "wall_clock": wall_clock,
        "grid": {
            "weight_decay_levels": list(grid["weight_decay_levels"]),
            "patience_levels": list(grid["patience_levels"]),
            "cells": len(cells),
        },
        "prohibition": phase26.THE_PROHIBITION,
        "iem_defects_travel": phase26.WHAT_THIS_PHASE_MAY_NOT_CLAIM[
            "4_the_IEM_defects_travel"
        ],
    }
    ctx.path("metrics.json", tier="SHAREABLE").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    ctx.log(
        f"sixteen cells, six readouts, floor {floor:.4f}, anchor "
        f"{anchor} at {got:.4f}"
    )

def task_p27_anchor_train(ctx: RunContext) -> None:
    """Phase 27: fit a linear head on the 25 anchor grades, evaluate on
    all 237 as pure test (``phase27.EXIT_CRITERIA``, locked 2026-09-06).

    **The separation between the two sets is STRUCTURAL, not
    documentary** (``phase27.THE_DISJOINTNESS_GUARD_DESIGNED``). The one
    precedent, ``task_tstr_regression``, rests its separation on a
    comment and statement ordering. This task rests it on four things
    that can each fail: the anchor loader's namespace refusal,
    ``phase27.disjoint_or_raise`` over the features, ``phase27.fit_head``
    having no parameter through which cohort truth could arrive, and an
    AST test on the call site below.

    **No inner validation, no monitor, no patience, no checkpoint
    selection** (``phase27.THE_INNER_VALIDATION_RULED``). The fit runs a
    fixed budget and the phase reads at two epochs declared in advance.
    """
    import json

    from . import phase16, phase27
    from . import embeddings as embeddings_module
    from .cluster_csv import write_predictions
    from .data.manifest import load_manifest
    from .eval import metrics as frozen_metrics

    task = ctx.config["task"]
    declared = {entry["name"]: entry for entry in ctx.inputs}
    seeds = [int(seed) for seed in task["seeds"]]
    readout_epochs = tuple(int(e) for e in task["readout_epochs"])
    if readout_epochs != phase27.READOUT_EPOCHS:
        raise ValueError(
            f"readout_epochs {readout_epochs} is not the declared "
            f"{phase27.READOUT_EPOCHS}: the epochs the phase reads at are "
            "fixed at the lock and are not a config choice"
        )

    # ---- TRAINING SET: the 25 anchors, and nothing else ---------------
    # Guard part one: load_anchor_set refuses any directory whose
    # metadata does not say namespace anchor_deall, and re-checks the 25
    # rows and the 3/7/6/6/3 spread. A config pointing this at the cohort
    # embeddings raises here rather than training on them.
    anchor_features, anchor_metadata = phase16.load_anchor_set(
        Path(declared[task["anchor_artifact"]]["path"])
    )
    anchor_grades = [float(g) for g in anchor_metadata["grades"]]
    anchor_stems = list(anchor_metadata["stems"])
    if len(anchor_grades) != int(task["expect_anchors"]):
        raise ValueError(
            f"{len(anchor_grades)} anchors where the lock fixes "
            f"{task['expect_anchors']}"
        )

    # ---- EVALUATION SET: all 237, pure test ---------------------------
    manifest_dir = Path(declared[task["manifest_artifact"]]["path"])
    embeddings_dir = Path(declared[task["embeddings_artifact"]]["path"])
    rows = load_manifest(manifest_dir / "manifest.csv")
    patient_ids = [int(row["patient_id"]) for row in rows]
    if len(patient_ids) != int(task["expect_patients"]):
        raise ValueError(
            f"{len(patient_ids)} patients where the lock fixes "
            f"{task['expect_patients']}"
        )
    truth_mean = np.array([float(row["mean"]) for row in rows])
    truth_class3 = np.array([int(row["class3"]) for row in rows])
    folds_by_patient = {int(r["patient_id"]): int(r["fold"]) for r in rows}
    cohort_features, _ = embeddings_module.load(embeddings_dir, patient_ids)

    # ---- Guard part two, BEFORE anything is fitted --------------------
    disjointness = phase27.disjoint_or_raise(
        anchor_features,
        cohort_features,
        train_ids=anchor_stems,
        eval_ids=patient_ids,
    )
    ctx.log(
        f"disjointness guard PASSED: {disjointness['n_train']} training "
        f"rows, {disjointness['n_eval']} evaluation rows, "
        f"{disjointness['feature_dim']}-d, no shared identifier and no "
        "identical feature row"
    )

    # ---- THE FIT. Guard part three is fit_head's signature: it takes
    # ---- features, targets and settings, and nothing else. Nothing
    # ---- derived from the cohort appears in this call.
    trajectory, states_by_seed = {}, {}
    for seed in seeds:
        per_epoch = {}
        for epoch, weight, bias in phase27.fit_head(
            anchor_features,
            anchor_grades,
            seed=seed,
            budget=int(task["max_epochs"]),
            learning_rate=float(task["learning_rate"]),
            weight_decay=float(task["weight_decay"]),
            max_steps=int(task["max_steps"]),
        ):
            per_epoch[epoch] = (weight, bias)
        states_by_seed[seed] = per_epoch
        ctx.log(f"  seed {seed}: {len(per_epoch)} epochs, no early stop")

    # ---- EVALUATION, after every head is trained and frozen -----------
    values, counts = np.unique(truth_class3, return_counts=True)
    chance, majority = 1.0 / len(values), float(counts.max() / counts.sum())
    offset = float(task["target_offset"])
    per_seed, predictions_by_seed = {}, {}
    for seed in seeds:
        trajectory[str(seed)] = {}
        for epoch, (weight, bias) in sorted(states_by_seed[seed].items()):
            predicted = phase27.predict(weight, bias, cohort_features)
            readout = {
                "pcc": frozen_metrics.pcc(truth_mean, predicted),
                "spearman": frozen_metrics.spearman(truth_mean, predicted),
                "rmse": frozen_metrics.rmse(truth_mean, predicted),
                "mae": frozen_metrics.mae(truth_mean, predicted),
                "acc3": float(np.mean(
                    frozen_metrics.to_3class(predicted, clip=True)
                    == truth_class3
                )),
                #: The collapse detector. Criterion 4 requires it, and it
                #: is what separates a degenerate fit from a genuine null
                #: (phase27.THE_DEGENERATE_SIGNATURE_REGISTERED).
                "prediction_sd": float(np.std(predicted)),
                "prediction_mean": float(np.mean(predicted)),
            }
            trajectory[str(seed)][str(epoch)] = readout
            if epoch in readout_epochs:
                per_seed.setdefault(str(epoch), {})[str(seed)] = readout
            if epoch == int(task["primary_epoch"]):
                predictions_by_seed[seed] = predicted

        write_predictions(
            ctx.path(f"seed_{seed}__predictions.csv", tier="CLUSTER-ONLY"),
            zip(patient_ids, truth_mean, predictions_by_seed[seed],
                (folds_by_patient[p] for p in patient_ids)),
        )

    def _band(epoch_key: str, field: str) -> dict:
        series = [per_seed[epoch_key][str(s)][field] for s in seeds]
        return {
            "mean": float(np.mean(series)),
            "sd": float(np.std(series, ddof=1)) if len(series) > 1 else 0.0,
            "min": float(np.min(series)),
            "max": float(np.max(series)),
            "n_seeds": len(series),
        }

    primary_key = str(int(task["primary_epoch"]))
    metrics = {
        "arm": task["arm"],
        "declared_readout_epochs": list(readout_epochs),
        "primary_epoch": int(task["primary_epoch"]),
        "per_seed_by_readout_epoch": per_seed,
        #: Criterion 7: this arm's OWN seed band, never the cohort arms'.
        "seed_band_by_readout_epoch": {
            key: {
                field: _band(key, field)
                for field in ("pcc", "spearman", "rmse", "mae", "acc3",
                              "prediction_sd")
            }
            for key in per_seed
        },
        "primary": {
            field: _band(primary_key, field)
            for field in ("pcc", "spearman", "rmse", "mae", "acc3",
                          "prediction_sd")
        },
        "baselines": {"chance": chance, "majority": majority},
        #: DESCRIPTIVE ONLY. Reading the best epoch off this is
        #: prohibited: phase21.SELECTION_ON_EVALUATION_DATA_PROHIBITION,
        #: carried at phase27.THE_TWO_READOUT_EPOCHS_DECLARED.
        "trajectory_DESCRIPTIVE_ONLY": trajectory,
        "disjointness_guard": disjointness,
        "the_crossing": {
            "training_target": "anchor Score, integer 1 to 5",
            "evaluation_target": "cohort panel mean, continuous on 0.2",
            "target_offset_declared": offset,
            "note": (
                "pcc and spearman cross the two targets cleanly; rmse, "
                "mae and acc3 carry the offset and are NOT comparable to "
                "a cohort-trained arm's "
                "(phase27.THE_LABEL_INCOMPARABILITY_ADDRESSED)"
            ),
        },
        "anchor_grade_mean": float(np.mean(anchor_grades)),
    }
    with ctx.atomic("metrics.json", tier="SHAREABLE") as tmp:
        tmp.write_text(
            json.dumps(as_builtin(metrics), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    ctx.log(
        f"primary (epoch {primary_key}): PCC "
        f"{metrics['primary']['pcc']['mean']:.4f} "
        f"sd {metrics['primary']['pcc']['sd']:.4f}, prediction sd "
        f"{metrics['primary']['prediction_sd']['mean']:.4f}"
    )


TASKS = {
    "smoke": task_smoke,
    "scan_folders": task_scan_folders,
    "inspect_scoresheet": task_inspect_scoresheet,
    "build_manifest": task_build_manifest,
    "build_views_manifest": task_build_views_manifest,
    "stage_basal_views": task_stage_basal_views,
    "extract_basal_embeddings": task_extract_basal_embeddings,
    "view_arm": task_view_arm,
    "extract_scut_embeddings": task_extract_scut_embeddings,
    "train_scut_decoder": task_train_scut_decoder,
    "reconstruct_cohort": task_reconstruct_cohort,
    "regional_comparison": task_regional_comparison,
    "asymmetry_retention": task_asymmetry_retention,
    "extract_reconstruction_embeddings": (
        task_extract_reconstruction_embeddings
    ),
    "confound_ceiling": task_confound_ceiling,
    "p20_permutation": task_p20_permutation,
    "p20_stratification_diagnostic": task_p20_stratification_diagnostic,
    "p21_ensemble_probe": task_p21_ensemble_probe,
    "p21_error_consistency": task_p21_error_consistency,
    "cohort_pair_separation": task_cohort_pair_separation,
    "tie_fraction": task_tie_fraction,
    "train_rank_cv": task_train_rank_cv,
    "p22_diagnostic": task_p22_diagnostic,
    "p22_contrasts": task_p22_contrasts,
    "p25_contrasts": task_p25_contrasts,
    "p26_calibration_table": task_p26_calibration_table,
    "p27_anchor_train": task_p27_anchor_train,
    "rater_icc": task_rater_icc,
    "p23_rope": task_p23_rope,
    "cross_arm_shrinkage": task_cross_arm_shrinkage,
    "extract_occluded_embeddings": task_extract_occluded_embeddings,
    "extract_occluded_recon_embeddings": (
        task_extract_occluded_recon_embeddings
    ),
    "asymmetry_encoding_probe": task_asymmetry_encoding_probe,
    "survey_mebeauty": task_survey_mebeauty,
    "verify_mebeauty_mapping": task_verify_mebeauty_mapping,
    "screen_mebeauty_landmarks": task_screen_mebeauty_landmarks,
    "stage_mebeauty": task_stage_mebeauty,
    "pretrain_mebeauty": task_pretrain_mebeauty,
    "extract_mebeauty_embeddings": task_extract_mebeauty_embeddings,
    "probe_mebeauty": task_probe_mebeauty,
    "classification_metrics": task_classification_metrics,
    "extract_anchor_embeddings": task_extract_anchor_embeddings,
    "anchor_loop": task_anchor_loop,
    "tstr_regression": task_tstr_regression,
    "siamese_contrastive": task_siamese_contrastive,
    "tstr_family_analysis": task_tstr_family_analysis,
    "metric_space_analysis": task_metric_space_analysis,
    "record_artifact_check": task_record_artifact_check,
    "contact_sheet": task_contact_sheet,
    "segmentation_probe": task_segmentation_probe,
    "symnose_audit": task_symnose_audit,
    "feature_relevance": task_feature_relevance,
    "masked_scut": task_masked_scut,
    "asymmetry_synthesis": task_asymmetry_synthesis,
    "pretrain": task_pretrain,
    "extract_embeddings": task_extract_embeddings,
    "phase7b_search": task_phase7b_search,
    "phase7c_paired": task_phase7c_paired,
    "paired_claims": task_paired_claims,
    "iem_prestep": task_iem_prestep,
    "iem_arm": task_iem_arm,
    "grad_cam": task_grad_cam,
    "grad_cam_softmax": task_grad_cam_softmax,
    "node_weights": task_node_weights,
    "tsne": task_tsne,
    "scut_animation": task_scut_animation,
    "prototypes": task_prototypes,
    "deall_reference": task_deall_reference,
    "prototype_classifier": task_prototype_classifier,
    "cleftgnn_cv": task_cleftgnn_cv,
    "cleftgnn_faithful": task_cleftgnn_faithful,
    "p10x_gate_fullfit": task_p10x_gate_fullfit,
    "p10x_distribution": task_p10x_distribution,
    "rater_screen": task_rater_screen,
    "phase8c_sheet": task_phase8c_sheet,
    "phase8c_animation": task_phase8c_animation,
    # Road B Phase 2 lives in its own module -- one phase's wiring, and run.py
    # already carries every other phase's. The registry stays the single place
    # a task kind resolves.
    "roadb_stage": _roadb_tasks.stage,
    "roadb_residual_gate": _roadb_tasks.residual_gate,
    "roadb_sheets": _roadb_tasks.sheets,
    "roadb_region_crop_sheet": _roadb_tasks.region_crop_sheet,
    "roadb_region_crop_extract": _roadb_tasks.region_crop_extract,
    "roadb_swin_build_gate": _roadb_tasks.swin_build_gate,
    "roadb_graph_geometry_gate": _roadb_tasks.graph_geometry_gate,
    "snapshot_pretrained_init": _roadb_tasks.snapshot_pretrained_init,
    "roadb_pretrain_build_gate": _roadb_tasks.pretrain_build_gate,
    "augmentation_contact_sheet": task_augmentation_contact_sheet,
    "train_graph_cv": task_train_graph_cv,
    "partition_sensitivity": task_partition_sensitivity,
    "anatomy_sweep": task_anatomy_sweep,
    "stage_and_patch": task_stage_and_patch,
    "train_cv": task_train_cv,
}


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> Path:
    """Run one config. Returns the run directory."""
    args = build_parser().parse_args(argv)

    with RunContext(args.config, args.out) as ctx:
        kind = ctx.config["task"]["kind"]
        if kind not in TASKS:
            raise ValueError(f"no task registered for kind {kind!r}")
        TASKS[kind](ctx)
        run_dir = ctx.run_dir

    print(run_dir)
    return run_dir


if __name__ == "__main__":  # pragma: no cover
    main()
