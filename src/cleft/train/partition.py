"""Sensitivity of the estimate to how the data is partitioned. Phase 4.

**What this measures, and what it does not.** It re-runs one arm across several
partitionings of the same 237 patients -- 5-fold, 6-fold, 10-fold, and
leave-one-patient-out -- and reports the spread of the pooled OOF PCC. That is
**sensitivity to partitioning**. It is *not* fold-assignment variance, and the
difference matters enough that the two names must not be swapped.

**Why not fold-assignment variance.** ``data.folds.generate`` runs the splitter
with ``shuffle=False``, and its docstring is explicit that the seed is recorded
but does not affect the partition: the assignment is a deterministic function of
(groups, labels, n_folds). So "regenerate at five fold-seeds" returns **one fold
set five times** and a spread of exactly zero -- which would read as *fold
assignment does not matter* while actually meaning *nothing was varied*.

Varying the fold **count** gives genuinely different partitions with the
deterministic stratification intact, no shuffling, and nothing touching the
frozen module. ``shuffle=False`` was a deliberate Phase 1 stratification fix and
is not being re-litigated here.

**The confound, stated rather than hidden.** Fold count also changes
training-set size: 5-fold trains on about 190 patients, 10-fold on about 213,
leave-one-out on 236. So the spread mixes *how the data is partitioned* with *how
much of it each model sees*. Neither is separable from the other in this design,
and both are reported per partitioning so the reader can see which is moving.

**Leave-one-patient-out is the reference point.** 237 folds, deterministic, and
**no assignment choice at all** -- there is nothing to stratify because every test
fold is one patient. Its distance from the 5-fold estimate is the single most
direct statement of how much the headline number depends on partitioning, and it
costs minutes over frozen embeddings.
"""

from __future__ import annotations

import numpy as np

from ..data import folds as folds_module

#: Fold counts to compare. All use the frozen generator unchanged.
DEFAULT_FOLD_COUNTS = (5, 6, 10)

#: The name the leave-one-patient-out partitioning is reported under.
LOPO = "lopo"


class PartitionError(RuntimeError):
    """A partitioning could not be built."""


#: **The gap this exercise identifies and does not close.**
#:
#: Three sources of variability bear on a claim in this project, and the design
#: sees two of them:
#:
#: * **patient sampling** -- captured by the paired BCa CI over patients (§4.3);
#: * **training procedure** -- captured empirically by the seed band (§4.12);
#: * **fold assignment** -- captured by **neither**.
#:
#: And the design cannot see the third: ``cleft_v1``'s folds are fixed, every arm
#: uses them, and the generator is deterministic, so no arm in the project ever
#: varies fold assignment while holding everything else constant. Nadeau-Bengio
#: was removed from the claim criterion partly because it addressed a test this
#: design does not run -- and the gap it was masking is exactly this one (§4.3).
#:
#: **Partition sensitivity is a bound on it, not a measurement of it.** If the
#: estimate barely moves across partitionings of very different shape, fold
#: assignment is unlikely to be moving it much either. If it moves a lot, the
#: cause is ambiguous between partitioning and training-set size. Either way this
#: belongs in limitations as a stated gap in the claim criterion, not as a
#: resolved question.
NOT_MEASURED = {
    "quantity": "fold-assignment variance",
    "captured_by_paired_bca": "patient sampling",
    "captured_by_seed_band": "training procedure",
    "captured_by_neither": "fold assignment",
    "why_the_design_cannot_see_it": (
        "cleft_v1's folds are fixed and every arm uses them, and the generator "
        "is deterministic (shuffle=False), so no arm varies fold assignment while "
        "holding everything else constant."
    ),
    "what_this_gives_instead": (
        "sensitivity to partitioning -- a BOUND on it, not a measurement of it, "
        "and confounded with training-set size because fold count changes both."
    ),
    "belongs_in": "limitations, as a stated gap in the claim criterion",
}


def assignments_for(
    patient_ids: list[int], class_labels, n_folds: int
) -> dict[int, int]:
    """Fold assignments from the **frozen** generator, at this fold count.

    ``class_labels`` must be the **3-class collapse** (``class3``), not the raw
    1-5 grade. The generator stratifies on it and refuses anything else -- the
    raw label has ~237 distinct values and stratifying on those is meaningless.
    Named ``class_labels`` rather than ``labels`` because the arm's *training*
    label and its *stratification* label are different columns of the same
    manifest, and passing the wrong one is a one-word mistake.
    """
    raw = [float(v) for v in class_labels]
    # Integrality first. ``int(2.7)`` is 2, so a truncating cast would accept the
    # raw 1-5 grade as class3 and stratify on a silently mangled column.
    fractional = [v for v in raw if v != int(v)]
    if fractional:
        raise PartitionError(
            f"stratification needs the 3-class collapse {folds_module.CLASSES}, "
            f"got non-integer values {fractional[:5]}. Pass the manifest's class3 "
            "column, not the raw grade the arm trains on."
        )

    values = [int(v) for v in raw]
    unexpected = sorted(set(values) - set(folds_module.CLASSES))
    if unexpected:
        raise PartitionError(
            f"stratification needs the 3-class collapse {folds_module.CLASSES}, "
            f"got {unexpected[:5]}. Pass the manifest's class3 column, not the "
            "raw grade the arm trains on."
        )
    result = folds_module.generate(patient_ids, values, n_folds=n_folds)
    return dict(result.assignments)


def leave_one_out_assignments(patient_ids: list[int]) -> dict[int, int]:
    """One patient per fold. **Not** built through the generator, deliberately.

    ``folds.generate`` requires every class to have at least ``n_folds`` members,
    which no 237-patient cohort satisfies at 237 folds -- and rightly so, because
    there is nothing to stratify. A leave-one-out partition has no assignment
    choice to make, so a stratifier has no job here and forcing one through it
    would be pretending otherwise.
    """
    ordered = sorted(int(pid) for pid in patient_ids)
    if len(set(ordered)) != len(ordered):
        raise PartitionError("patient ids must be unique for leave-one-out")
    return {pid: index for index, pid in enumerate(ordered)}


def training_sizes(assignments: dict[int, int]) -> dict:
    """How much data each model actually sees. The confound, quantified."""
    n = len(assignments)
    counts: dict[int, int] = {}
    for fold in assignments.values():
        counts[fold] = counts.get(fold, 0) + 1
    train_sizes = [n - size for size in counts.values()]
    return {
        "n_folds": len(counts),
        "mean_train_size": round(float(np.mean(train_sizes)), 1),
        "min_train_size": int(min(train_sizes)),
        "max_train_size": int(max(train_sizes)),
        "mean_test_size": round(float(np.mean(list(counts.values()))), 1),
    }


def partitionings(
    patient_ids: list[int],
    class_labels,
    fold_counts=DEFAULT_FOLD_COUNTS,
    include_lopo: bool = True,
) -> dict[str, dict[int, int]]:
    """Every partitioning to compare, keyed by name. ``class_labels`` is class3."""
    out: dict[str, dict[int, int]] = {}
    for n_folds in fold_counts:
        out[f"{n_folds}fold"] = assignments_for(patient_ids, class_labels, n_folds)
    if include_lopo:
        out[LOPO] = leave_one_out_assignments(patient_ids)
    return out


def sensitivity(results: dict[str, dict]) -> dict:
    """The spread across partitionings, with the reference comparison.

    ``results`` maps a partitioning name to at least ``{"pcc": float}`` plus the
    ``training_sizes`` block for that partitioning.
    """
    if not results:
        raise PartitionError("no partitionings to compare")

    values = {name: float(entry["pcc"]) for name, entry in results.items()}
    pccs = list(values.values())
    baseline = values.get("5fold")
    reference = values.get(LOPO)

    return {
        "pcc_by_partitioning": {k: round(v, 6) for k, v in values.items()},
        "n_partitionings": len(values),
        "range": round(max(pccs) - min(pccs), 6),
        "sd": round(float(np.std(pccs, ddof=1)), 6) if len(pccs) > 1 else 0.0,
        "min": round(min(pccs), 6),
        "max": round(max(pccs), 6),
        # The headline: how far the reference point with no assignment choice at
        # all sits from the number the project actually quotes.
        "lopo_minus_5fold": (
            round(reference - baseline, 6)
            if reference is not None and baseline is not None
            else None
        ),
        "training_sizes": {
            name: entry.get("training_sizes") for name, entry in results.items()
        },
        "measures": "sensitivity_to_partitioning",
        "does_not_measure": dict(NOT_MEASURED),
        "note": (
            "SENSITIVITY TO PARTITIONING, not fold-assignment variance -- see "
            "does_not_measure. Fold count changes the partition AND the "
            "training-set size (5-fold trains on ~190, 10-fold on ~213, "
            "leave-one-out on 236), so the spread mixes the two and neither is "
            "separable here. training_sizes is reported per partitioning so it is "
            "visible which is moving. "
            "lopo_minus_5fold is the most direct statement available: "
            "leave-one-patient-out has 237 folds, is deterministic, and involves "
            "NO assignment choice at all, so its distance from the quoted 5-fold "
            "number bounds how much that number depends on partitioning. "
            "COMPARE THE RANGE AGAINST THE SEED BAND (PLAN §4.12) before reading "
            "anything into it: a spread inside the band is not a finding."
        ),
    }
