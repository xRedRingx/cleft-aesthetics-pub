"""Derive ``cleft_v1_views``: the 236-patient both-views cohort.

**A derivation, not a rebuild** (``phase12.STOP_1_MANIFEST``). ``cleft_v1``
already carries the view pairing -- its frontal came from score-sheet
lookup and its ``basal_id`` is the folder's other image -- so the pairing
here is CARRIED from the artifact that measured it, never re-derived. A
second implementation of frontal-by-lookup is the duplication R10 forbids,
and it is also how folder 143's two views could quietly swap.

**What the derivation does**: drop the one patient without a basal view
(folder 238), carry every other row VERBATIM -- labels, soft labels,
class3 and the FOLD column included -- and refuse to produce anything if
the source does not look exactly like the cohort Phase 1 measured.

**Why folds are carried, not re-stratified.** Re-running StratifiedKFold
on 236 patients would reshuffle every assignment, so "arm A on 236 vs the
0.2520 arm on 237" would differ by a fold reshuffle AND a patient -- two
factors. Carrying the folds means the phase's baseline re-measurement
differs from the ladder by exactly one patient's removal, and every arm
inside the phase shares one fold structure by construction.
"""

from __future__ import annotations

from . import manifest

#: The one patient captured with a single view; excluded from EVERY arm of
#: the view ablation, the frontal-only arm included, so every contrast is
#: within-cohort and one-factor (``phase12.PHASE_12_REGISTERED``).
EXCLUDED_PATIENT = 238

#: Folder 143's pairing, asserted rather than trusted: the odd/even rule
#: inverts there, and swapping these two ids would score the basal as the
#: frontal with nothing downstream erroring.
EXCEPTION_PAIRING = {143: {"frontal_id": 524, "basal_id": 523}}

#: The derived cohort shape: every patient with both views.
VIEWS_COHORT = {"patients": 236, "frontal": 236, "basal": 236}

#: The source cohort shape this derivation refuses to run without.
SOURCE_ROWS = 237


class ViewsError(manifest.ManifestError):
    """A views-manifest invariant failed. Always fatal, never a warning."""


def views_manifest_schema() -> dict:
    """The layout of the derived manifest.csv -- ``cleft_v1``'s columns,
    same names and order, so every existing loader reads it unchanged.
    Only ``basal_id``'s contract tightens: never empty here."""
    schema = manifest.manifest_schema()
    schema["derived_from"] = "cleft_v1 -- rows carried verbatim, 238 dropped"
    schema["basal_id_contract"] = (
        "NEVER empty in this artifact: the both-views cohort excludes "
        f"folder {EXCLUDED_PATIENT}"
    )
    return schema


def derive(rows: list[dict]) -> list[dict]:
    """``cleft_v1``'s rows -> the 236 both-views rows, verbatim.

    Every invariant is checked against the SOURCE first, so a source that
    has drifted from the Phase 1 measurement stops the derivation instead
    of silently producing a plausible cohort of the wrong shape.
    """
    if len(rows) != SOURCE_ROWS:
        raise ViewsError(
            f"source manifest has {len(rows)} rows, expected {SOURCE_ROWS} "
            "(cleft_v1's measured cohort). This derivation is defined "
            "against that artifact and no other."
        )

    ids = [int(r["patient_id"]) for r in rows]
    if len(set(ids)) != len(ids):
        raise ViewsError("source manifest repeats a patient id")

    missing_basal = [int(r["patient_id"]) for r in rows if not r["basal_id"]]
    if missing_basal != [EXCLUDED_PATIENT]:
        raise ViewsError(
            f"patients without a basal view: {missing_basal}, expected "
            f"exactly [{EXCLUDED_PATIENT}]. A second single-view patient "
            "(or 238 growing a basal) means the data or an assumption has "
            "changed, and the cohort must not be derived until it is "
            "understood."
        )

    # Folder 143's pairing, asserted against the recorded exception.
    by_id = {int(r["patient_id"]): r for r in rows}
    for patient, expected in EXCEPTION_PAIRING.items():
        row = by_id[patient]
        actual = {
            "frontal_id": int(row["frontal_id"]),
            "basal_id": int(row["basal_id"]),
        }
        if actual != expected:
            raise ViewsError(
                f"folder {patient} pairing is {actual}, expected {expected}. "
                "Swapped views here score the basal as the frontal with "
                "nothing downstream erroring -- the exact failure the "
                "odd/even cross-check exists for."
            )

    kept = [r for r in rows if int(r["patient_id"]) != EXCLUDED_PATIENT]
    for row in kept:
        frontal, basal = int(row["frontal_id"]), int(row["basal_id"])
        if frontal == basal:
            raise ViewsError(
                f"patient {row['patient_id']}: frontal and basal are both "
                f"{frontal} -- one image cannot be two views"
            )

    counts = {
        "patients": len(kept),
        "frontal": sum(1 for r in kept if r["frontal_id"]),
        "basal": sum(1 for r in kept if r["basal_id"]),
    }
    if counts != VIEWS_COHORT:
        raise ViewsError(
            f"derived cohort is {counts}, expected {VIEWS_COHORT}"
        )

    # Dropping one patient must not empty or unbalance a fold into
    # unusability: every fold must still exist on both sides of every
    # split. (238's fold loses exactly one patient; the check is that no
    # fold vanishes, not that sizes stay equal.)
    folds = sorted({int(r["fold"]) for r in kept})
    if folds != sorted({int(r["fold"]) for r in rows}):
        raise ViewsError(
            f"dropping patient {EXCLUDED_PATIENT} removed an entire fold: "
            f"{folds} remain. The fold structure is carried from cleft_v1 "
            "and must survive the drop."
        )

    return kept


def pairing_summary(kept: list[dict]) -> dict:
    """What the derived artifact's MANIFEST.json records about the pairing
    -- the table's shape, its exceptions, and the carried-fold decision,
    so a consumer reads the provenance instead of rediscovering it."""
    per_fold: dict = {}
    for row in kept:
        per_fold[int(row["fold"])] = per_fold.get(int(row["fold"]), 0) + 1
    return {
        "patients": len(kept),
        "views_per_patient": 2,
        "excluded": {
            "patient": EXCLUDED_PATIENT,
            "reason": (
                "single view (frontal 581, no basal); excluded from ALL "
                "arms of the view ablation, the frontal-only arm included, "
                "so every contrast is within-cohort and one-factor"
            ),
        },
        "exception_pairing": {
            str(k): v for k, v in EXCEPTION_PAIRING.items()
        },
        "folds_carried_from": (
            "cleft_v1, verbatim -- re-stratifying on 236 would reshuffle "
            "every assignment and make the baseline re-measurement differ "
            "from the ladder by two factors instead of one"
        ),
        "fold_sizes": {str(k): per_fold[k] for k in sorted(per_fold)},
    }
