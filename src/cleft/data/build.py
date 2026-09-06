"""Phase 1: build the manifest, labels, reliability and folds from the real data.

Wiring only. Every step below is a module that has its own tests and, by now, its
own confirmation against the real cohort; this orders them and writes the
artifact.

Order matters, and it is: **validate everything before writing anything.** The
artifact directory is created last, once every assertion has passed, so a run
that fails leaves no half-built ``data/manifests/cleft_v1/`` for a later run to
mistake for a finished one.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..provenance import atomic_write_text, hash_dir
from . import folderscan, folds, labels, manifest, reliability, report, scoresheet, softlabels


class BuildError(RuntimeError):
    """The build cannot proceed. Always fatal, always specific."""


@dataclass
class BuildResult:
    manifest: manifest.Manifest
    summary: report.Phase1Summary
    artifact_dir: Path


def _rater_index(sheet: scoresheet.ScoreSheet, name: str) -> int:
    """Find a rater column by NAME, so a column reorder cannot change the target."""
    normalised = scoresheet.normalise_header(name).casefold()
    for index, rater in enumerate(sheet.raters):
        if scoresheet.normalise_header(rater).casefold() == normalised:
            return index
    raise BuildError(
        f"rater {name!r} is not a column in the score sheet. Columns are: "
        f"{list(sheet.raters)}"
    )


def _check_equivalence(primary: scoresheet.ScoreSheet, other: scoresheet.ScoreSheet) -> None:
    if set(primary.rows) != set(other.rows):
        raise BuildError(
            "the two workbook forms cover different photo ids: "
            f"{len(primary.rows)} vs {len(other.rows)}. They are documented as "
            "identical in content; investigate before building."
        )
    differing = [p for p in primary.rows if primary.rows[p].grades != other.rows[p].grades]
    if differing:
        raise BuildError(
            f"the two workbook forms disagree on {len(differing)} row(s), e.g. "
            f"{differing[:5]}. They are documented as identical in content. Do "
            "not pick one: find out why they differ."
        )


def build(
    *,
    primary_path: Path,
    patient_folders_path: Path,
    artifact_dir: Path,
    orthodontist_rater: str,
    n_folds: int,
    seed: int,
    expected: manifest.Counts,
    expected_scored_rows: int,
    equivalence_path: Path | None = None,
    soft_reference_path: Path | None = None,
    log=print,
) -> BuildResult:
    """Run the whole Phase 1 pipeline. Writes the artifact only if all of it passes."""
    if artifact_dir.exists():
        raise BuildError(
            f"{artifact_dir} already exists. Data artifacts are immutable: bump "
            "out_version and create the next one. Never modify a version that "
            "results may already cite."
        )

    # 1. Score sheet -------------------------------------------------------
    log(f"loading score sheet: {primary_path.name}")
    sheet = scoresheet.load(primary_path, expected_rows=expected_scored_rows)
    log(f"  {len(sheet.rows)} rows, {len(sheet.raters)} raters, {sheet.n_cells} cells")

    if equivalence_path is not None:
        log(f"checking equivalence against {equivalence_path.name}")
        _check_equivalence(sheet, scoresheet.load(equivalence_path))
        log("  identical")

    # 2. Folder tree -------------------------------------------------------
    log(f"scanning {patient_folders_path}")
    scan = folderscan.scan(patient_folders_path)
    log(
        f"  {len(scan.folders)} directories, {len(scan.with_images)} with images, "
        f"empty {scan.empty}, {scan.total_images} images, "
        f"{scan.skipped_clutter} clutter files skipped"
    )

    # 3. Manifest ----------------------------------------------------------
    scored_sequence = sheet.photo_ids()
    result = manifest.build(
        scan.folders,
        set(scored_sequence),
        scored_sequence=scored_sequence,
        expected=expected,
    )
    log(
        f"  {len(result.rows)} patients, {len(result.photoless_ids)} photo-less ids, "
        f"empty folders {result.empty_folders}"
    )
    log(f"  rule cross-check disagrees in: {sorted(result.rule_disagreements)}")

    # 4. Labels ------------------------------------------------------------
    frontal_ids = [row.frontal_id for row in result.rows]
    matrix = np.array([sheet.rows[i].grades for i in frontal_ids], dtype=np.int64)

    targets = {kind: labels.build_target(matrix, kind) for kind in labels.AGGREGATE_TARGETS}
    ortho_index = _rater_index(sheet, orthodontist_rater)
    targets["orthodontist"] = matrix[:, ortho_index].astype(float)
    soft = labels.soft_labels(matrix)
    class3 = labels.class3(targets["mean"])

    # 5. Soft-label cross-check -------------------------------------------
    if soft_reference_path is not None:
        log(f"cross-checking soft labels against {soft_reference_path.name}")
        computed = {pid: soft[i] for i, pid in enumerate(frontal_ids)}
        reference = softlabels.load_reference(soft_reference_path)
        # The reference covers all 251 scored rows; we hold the 237 with photos.
        softlabels.crosscheck(
            computed,
            {k: v for k, v in reference.items() if k in computed},
            require_full_coverage=True,
        )
        log("  agrees")

    # 6. Reliability and learnability -------------------------------------
    stats = reliability.summarise(matrix)
    learn = labels.learnability_table(matrix, orthodontist_index=ortho_index)

    # 7. Folds -------------------------------------------------------------
    patient_ids = [row.patient_id for row in result.rows]
    fold_result = folds.generate(patient_ids, class3, n_folds=n_folds, seed=seed)

    # 8. The summary -------------------------------------------------------
    summary = report.build_summary(
        counts=report.CohortCounts(
            patients=len(result.rows),
            frontal=sum(1 for r in result.rows if r.frontal_id is not None),
            basal=sum(1 for r in result.rows if r.basal_id is not None),
            photoless=len(result.photoless_ids),
            images=result.n_images,
            scored_rows=len(sheet.rows),
        ),
        rule_disagreements=result.rule_disagreements,
        reliability=stats,
        learnability=learn,
        expected_ordering=labels.EXPECTED_ORDERING,
        fold_sizes=[len(fold_result.test_ids(f)) for f in range(fold_result.n_folds)],
        fold_class_counts=fold_result.class_counts,
        rater_names=sheet.raters,
    )

    # 9. Everything passed. Only now does anything get written. -------------
    _write_artifact(
        artifact_dir=artifact_dir,
        rows=result.rows,
        targets=targets,
        soft=soft,
        class3=class3,
        fold_result=fold_result,
        manifest_result=result,
        summary=summary,
        seed=seed,
        sources={
            "primary_scoresheet": str(primary_path),
            "patient_folders": str(patient_folders_path),
            "equivalence_scoresheet": str(equivalence_path) if equivalence_path else None,
            "soft_label_reference": str(soft_reference_path) if soft_reference_path else None,
        },
    )
    log(f"artifact written: {artifact_dir}")
    return BuildResult(manifest=result, summary=summary, artifact_dir=artifact_dir)


def _write_artifact(
    *,
    artifact_dir: Path,
    rows,
    targets,
    soft,
    class3,
    fold_result,
    manifest_result,
    summary,
    seed,
    sources,
) -> None:
    artifact_dir.mkdir(parents=True)

    # Single source of truth for the layout, shared with manifest_schema() so the
    # documentation in MANIFEST.json cannot drift from the file it describes.
    header = [name for name, _ in manifest.MANIFEST_COLUMNS]
    lines = [
        "# CLUSTER-ONLY: patient-keyed, never leaves EHU infrastructure",
        ",".join(header),
    ]
    for index, row in enumerate(rows):
        lines.append(
            ",".join(
                [
                    str(row.patient_id),
                    str(row.frontal_id),
                    "" if row.basal_id is None else str(row.basal_id),
                    f"{targets['mean'][index]:.6f}",
                    f"{targets['median'][index]:.6f}",
                    f"{targets['mode'][index]:.6f}",
                    f"{targets['weighted_mean'][index]:.6f}",
                    f"{targets['orthodontist'][index]:.6f}",
                    *[f"{v:.6f}" for v in soft[index]],
                    str(int(class3[index])),
                    str(fold_result.assignments[row.patient_id]),
                ]
            )
        )
    atomic_write_text(artifact_dir / "manifest.csv", "\n".join(lines) + "\n")

    folds.save(fold_result, artifact_dir / "folds.json")

    atomic_write_text(
        artifact_dir / "photoless_ids.json",
        json.dumps(
            {
                "note": "scored rows with no photograph; recorded, never dropped",
                "count": len(manifest_result.photoless_ids),
                "ids": manifest_result.photoless_ids,
            },
            indent=2,
        )
        + "\n",
    )

    payload = hash_dir(artifact_dir)
    atomic_write_text(
        artifact_dir / "MANIFEST.json",
        json.dumps(
            {
                "artifact": artifact_dir.name,
                "seed": seed,
                "sources": sources,
                "payload_rollup": payload["rollup"],
                "payload_files": payload["files"],
                "payload_total_bytes": payload["total_bytes"],
                "summary": summary.as_dict(),
                # What the CSV columns mean, and the comment-line trap, recorded
                # in the artifact so a consumer never has to guess or experiment.
                "manifest_csv": manifest.manifest_schema(),
                "empty_folders": manifest_result.empty_folders,
                "rule_disagreements": sorted(manifest_result.rule_disagreements),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )
