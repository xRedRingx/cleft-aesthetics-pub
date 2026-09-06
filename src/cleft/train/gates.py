"""Gates 5 and 6, as callable checks rather than only as tests.

A gate that exists only in the test suite is not checked on the cluster run that
matters. These run inside the task, on the real cohort, and their results go into
the SHAREABLE block.

Gate 3 lives in ``harness.assert_epoch0_calibrated`` because it has to fire
mid-fold; gates 1 and 2 are cluster procedures rather than functions.
"""

from __future__ import annotations

import numpy as np

from ..eval import metrics


class GateError(RuntimeError):
    """A gate failed. A gate that 'mostly passes' is a gate that failed."""


# --------------------------------------------------------------------------
# gate 5 -- metric verification
# --------------------------------------------------------------------------


def verify_metrics(seed: int = 1337, n: int = 200) -> dict:
    """Check every reported metric against scipy/sklearn on synthetic data.

    Phase 0 already tests these; this runs the same comparison *inside the run*,
    so a cluster result carries evidence that its own metric code agreed with the
    reference implementations at the time it was produced -- not that it did on a
    laptop at some earlier commit.
    """
    from scipy import stats
    from sklearn.metrics import cohen_kappa_score

    rng = np.random.default_rng(seed)
    truth = rng.uniform(1.0, 5.0, size=n)
    pred = np.clip(truth + rng.normal(0.0, 0.8, size=n), 1.0, 5.0)

    checks = {
        "pcc": (metrics.pcc(truth, pred), float(stats.pearsonr(truth, pred).statistic)),
        "spearman": (
            metrics.spearman(truth, pred),
            float(stats.spearmanr(truth, pred).statistic),
        ),
        "qwk_3cat": (
            metrics.qwk_3cat(truth, pred),
            float(
                cohen_kappa_score(
                    metrics.to_3class(truth), metrics.to_3class(pred),
                    weights="quadratic",
                )
            ),
        ),
        "mae": (metrics.mae(truth, pred), float(np.abs(truth - pred).mean())),
        "rmse": (
            metrics.rmse(truth, pred),
            float(np.sqrt(((truth - pred) ** 2).mean())),
        ),
    }

    disagreements = {
        name: {"ours": ours, "reference": reference}
        for name, (ours, reference) in checks.items()
        if abs(ours - reference) > 1e-9
    }
    if disagreements:
        raise GateError(
            f"metric verification failed against scipy/sklearn: {disagreements}"
        )

    # The paired machinery: a variable against itself must not claim a delta.
    delta, lo, hi = metrics.paired_delta_bca(
        metrics.pcc, truth, pred, pred, n_boot=500, seed=seed
    )
    if not (lo <= 0.0 <= hi) or abs(delta) > 1e-12:
        raise GateError(
            f"paired delta of a variable against itself gave {delta} with CI "
            f"({lo}, {hi}); it must be 0 with an interval containing 0"
        )

    return {
        "n": n,
        "seed": seed,
        "metrics_verified": sorted(checks),
        "max_abs_difference": max(
            abs(ours - ref) for ours, ref in checks.values()
        ),
        "paired_self_delta": delta,
        "paired_self_ci": [lo, hi],
    }


# --------------------------------------------------------------------------
# gate 6 -- OOF integrity by reconstruction
# --------------------------------------------------------------------------


def reconstruct_oof(
    fold_record: dict,
    *,
    expected_patients: set[int],
    expected_assignments: dict[int, int],
    oof_ids: list[int],
) -> dict:
    """Rebuild the OOF structure from the fold artifact and check it.

    **Not by trusting an assert.** Phase 1 found ``csv.DictReader`` silently
    misreading the manifest because of the comment line, and the tests passed
    because they had been written around it. An assertion that has never been
    checked against an independent reconstruction is not evidence.

    So this takes the recorded per-fold membership and derives, from scratch:
    who was predicted, how often, whether anyone trained on themselves, and
    whether the fold assignment is the one ``cleft_v1`` specifies.
    """
    folds = fold_record.get("folds")
    if not folds:
        raise GateError("the fold record contains no folds")

    seen: dict[int, int] = {}
    problems: list[str] = []

    for entry in folds:
        fold = entry["fold"]
        train = set(entry["train_ids"])
        inner = set(entry["inner_val_ids"])
        test = set(entry["test_ids"])

        # 1. Nobody may be in their own training fold, directly or via inner-val.
        leaked = test & (train | inner)
        if leaked:
            problems.append(
                f"fold {fold}: {len(leaked)} patient(s) appear in both the test "
                f"fold and its own training data, e.g. {sorted(leaked)[:5]}"
            )

        # 2. Inner-val is drawn only from the training folds.
        if inner & test:
            problems.append(f"fold {fold}: inner-val overlaps the test fold")
        if train & inner:
            problems.append(f"fold {fold}: inner-val overlaps the training set")

        # 3. The recorded fold assignment matches the artifact's.
        for pid in test:
            if expected_assignments.get(pid) != fold:
                problems.append(
                    f"patient {pid} was tested in fold {fold} but cleft_v1 "
                    f"assigns it to fold {expected_assignments.get(pid)}"
                )
                break
            seen[pid] = seen.get(pid, 0) + 1

    # 4. Every patient predicted exactly once.
    predicted_once = {pid for pid, count in seen.items() if count == 1}
    repeated = sorted(pid for pid, count in seen.items() if count > 1)
    if repeated:
        problems.append(f"{len(repeated)} patient(s) predicted more than once: {repeated[:5]}")

    missing = sorted(expected_patients - predicted_once)
    if missing:
        problems.append(
            f"{len(missing)} patient(s) never appear in any test fold: {missing[:5]}"
        )
    extra = sorted(predicted_once - expected_patients)
    if extra:
        problems.append(f"{len(extra)} unexpected patient(s) predicted: {extra[:5]}")

    # 5. The pooled OOF vector matches the reconstruction.
    if sorted(oof_ids) != sorted(predicted_once):
        problems.append(
            f"the pooled OOF vector covers {len(set(oof_ids))} patients but the "
            f"fold record reconstructs {len(predicted_once)}"
        )
    if len(oof_ids) != len(set(oof_ids)):
        problems.append("the pooled OOF vector contains a duplicate patient")

    if problems:
        raise GateError(
            "OOF reconstruction failed:\n  " + "\n  ".join(problems)
        )

    return {
        "n_patients": len(predicted_once),
        "n_folds": len(folds),
        "fold_sizes": [len(entry["test_ids"]) for entry in folds],
        "inner_val_sizes": [len(entry["inner_val_ids"]) for entry in folds],
        "train_sizes": [len(entry["train_ids"]) for entry in folds],
        "every_patient_predicted_once": True,
        "no_patient_in_own_training_fold": True,
        "inner_val_from_training_folds_only": True,
        "assignments_match_manifest": True,
    }
