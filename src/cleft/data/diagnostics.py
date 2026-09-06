"""Two cheap read-only diagnostics, to be run before the expensive build.

Both take seconds, write no artifact, and report aggregates only. They exist
because the folder scanner and the score-sheet loader have never seen the real
data, and finding a structural surprise costs one short job here instead of a
half-written artifact and a debugging session there.

**Why these two configs declare their paths under ``task`` rather than
``inputs``.** A declared input is an artifact whose identity is pinned by guard
3, and pinning requires knowing the hash — which is what the round trip in
``p1_build_manifest.yaml`` establishes. A diagnostic runs *before* that: its job
is to look at a path and describe it, and it consumes nothing and produces
nothing that any result depends on. Requiring the hash first would mean the
diagnostic could only run after the step it exists to de-risk.

That exception applies to diagnostics only. Anything that produces an artifact
declares its inputs and is hashed.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from . import folderscan, scoresheet


def _bar(count: int, total: int, width: int = 30) -> str:
    filled = 0 if total == 0 else round(width * count / total)
    return "#" * filled


# --------------------------------------------------------------------------
# folder tree
# --------------------------------------------------------------------------


def scan_folders_report(root: str | Path) -> tuple[dict, str]:
    """Inventory the patient folder tree. Aggregates plus folder numbers.

    Folder numbers are anonymised sequence ids, and the governing plan already
    publishes 52, 143 and 238. Gaps are listed because an unexpected one is the
    single most useful thing this can tell you.
    """
    root = Path(root)
    result = folderscan.scan(root)
    folders = result.folders

    numbers = sorted(folders)
    expected = set(range(numbers[0], numbers[-1] + 1))
    gaps = sorted(expected - set(numbers))
    per_folder = Counter(len(ids) for ids in folders.values())

    payload = {
        "root": str(root),
        "n_directories": len(folders),
        "n_with_images": len(result.with_images),
        "empty_folders": result.empty,
        "first_folder": numbers[0],
        "last_folder": numbers[-1],
        "gaps": gaps,
        "n_gaps": len(gaps),
        "images_per_folder": {str(k): v for k, v in sorted(per_folder.items())},
        "total_images": result.total_images,
        "single_image_folders": sorted(f for f, ids in folders.items() if len(ids) == 1),
        "skipped_clutter": result.skipped_clutter,
        "skipped_by_name": dict(sorted(result.skipped_names.items())),
    }

    lines = [
        "=" * 70,
        "FOLDER SCAN  [SHAREABLE - structure only, no image content]",
        "=" * 70,
        f"  root                 {root}",
        f"  directories found    {len(folders)}",
        f"  with images          {len(result.with_images)}",
        f"  present but EMPTY    {result.empty}",
        f"  numbered             {numbers[0]} .. {numbers[-1]}",
        f"  missing numbers      {gaps if gaps else 'none'}",
        f"  total images         {result.total_images}",
        "",
        "  images per folder:",
    ]
    for size, count in sorted(per_folder.items()):
        lines.append(f"    {size} image(s)        {count:>4}  {_bar(count, len(folders))}")

    lines += [
        "",
        f"  single-image folders {payload['single_image_folders']}",
        "",
        f"  clutter skipped      {result.skipped_clutter}",
    ]
    for name, count in sorted(result.skipped_names.items()):
        lines.append(f"    {name:<20} {count:>5}")

    lines += [
        "",
        "  EXPECTED: 238 directories, ALL present; folder 52 present but empty,",
        "            so 237 hold photographs. 473 images, two per folder except",
        "            238 which has one. Clutter: 473 AppleDouble sidecars",
        "            (._*.jpg, 4096 bytes each) + 3 Thumbs.db + 3 ._Thumbs.db.",
        "=" * 70,
    ]
    return payload, "\n".join(lines)


# --------------------------------------------------------------------------
# score sheet
# --------------------------------------------------------------------------


def compare_scoresheets(primary: Path, other: Path) -> tuple[dict, list[str]]:
    """Are the two workbook forms genuinely the same data?

    The text and integer workbooks are documented as identical in content, and
    the build asserts it. This turns that from two outputs to eyeball into one
    explicit line, before anything depends on it.
    """
    a = scoresheet.load(primary)
    b = scoresheet.load(other)

    shared = sorted(set(a.rows) & set(b.rows))
    only_a = sorted(set(a.rows) - set(b.rows))
    only_b = sorted(set(b.rows) - set(a.rows))
    differing = [pid for pid in shared if a.rows[pid].grades != b.rows[pid].grades]

    identical = not only_a and not only_b and not differing
    payload = {
        "other_file": other.name,
        "identical": identical,
        "n_shared_ids": len(shared),
        "n_only_in_primary": len(only_a),
        "n_only_in_other": len(only_b),
        "n_differing_rows": len(differing),
    }

    verdict = "IDENTICAL" if identical else "DIFFERENT"
    lines = [
        "",
        "-- equivalence check against the second workbook form ---------------",
        f"  other file           {other.name}",
        f"  shared photo ids     {len(shared)}",
        f"  only in primary      {len(only_a)}",
        f"  only in other        {len(only_b)}",
        f"  rows with different grades  {len(differing)}",
        f"  VERDICT              {verdict}",
    ]
    if not identical:
        lines += [
            "",
            "  The two forms are documented as identical in content. They are not.",
            "  Investigate before building anything: one of them is not the file",
            "  it is believed to be. Do not simply pick the primary.",
        ]
    return payload, lines


def inspect_scoresheet_report(
    path: str | Path, compare_with: str | Path | None = None
) -> tuple[dict, str]:
    """Load a workbook and describe it. Builds nothing."""
    path = Path(path)
    sheet = scoresheet.load(path)
    matrix = sheet.grade_matrix()

    per_rater = {}
    for index, rater in enumerate(sheet.raters):
        counts = Counter(int(v) for v in matrix[:, index])
        per_rater[rater] = {str(g): counts.get(g, 0) for g in (1, 2, 3, 4, 5)}

    overall = Counter(int(v) for v in matrix.ravel())
    ids = sheet.photo_ids()

    payload = {
        "path": str(path),
        "n_rows": len(sheet.rows),
        "n_raters": len(sheet.raters),
        "n_cells": sheet.n_cells,
        "n_missing": sheet.n_missing,
        "raters_normalised": list(sheet.raters),
        "grade_histogram": {str(g): overall.get(g, 0) for g in (1, 2, 3, 4, 5)},
        "per_rater_histogram": per_rater,
        "photo_id_min": ids[0],
        "photo_id_max": ids[-1],
    }

    lines = [
        "=" * 70,
        "SCORE SHEET  [SHAREABLE - counts only, no per-patient values]",
        "=" * 70,
        f"  file                 {path.name}",
        f"  rows                 {len(sheet.rows)}",
        f"  raters               {len(sheet.raters)}",
        f"  cells                {sheet.n_cells}   missing {sheet.n_missing}",
        f"  photo id range       {ids[0]} .. {ids[-1]}",
        "",
        "  rater columns after normalisation:",
    ]
    lines += [f"    {rater}" for rater in sheet.raters]

    lines += ["", "  grade histogram (all raters):"]
    total = sum(overall.values())
    words = {value: word for word, value in scoresheet.GRADE_WORDS.items()}
    for grade in (1, 2, 3, 4, 5):
        count = overall.get(grade, 0)
        share = 0.0 if total == 0 else 100.0 * count / total
        label = words.get(grade, "?").title()
        lines.append(
            f"    {grade} {label:<11} {count:>5}  {share:5.1f}%  {_bar(count, total)}"
        )

    lines += ["", "  per rater:"]
    for rater, counts in per_rater.items():
        rendered = "  ".join(f"{g}:{n:>4}" for g, n in counts.items())
        lines.append(f"    {rater:<44} {rendered}")

    if compare_with is not None:
        comparison, comparison_lines = compare_scoresheets(path, Path(compare_with))
        payload["equivalence"] = comparison
        lines += comparison_lines

    lines += [
        "",
        "  EXPECTED: 251 rows, 5 raters, 1255 cells, 0 missing.",
        "=" * 70,
    ]
    return payload, "\n".join(lines)
