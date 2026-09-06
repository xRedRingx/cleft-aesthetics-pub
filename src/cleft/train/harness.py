"""The minimal training path: 5-fold CV over the Phase 1 folds.

Phase 3 §2. Scoped deliberately small -- one backbone, whole-image, one label --
because this exists to test the *harness*, not to produce a result.

**No torch anywhere in this module.** The backbone arrives through a factory, so
the harness can be exercised on the laptop with a stub and only ever meets
ViT-B/16 on the cluster. That also makes gate 3's teeth-test honest: the stub can
reproduce the actual defect rather than a contrived one.

The harness is *designed* for patches and multiple backbones without
*implementing* them -- ``features`` is an opaque array the backbone interprets.
The patch path slots in at Phase 7, and **determinism must be re-verified then**,
because gate 1 tests the harness and the harness will have changed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Protocol

import numpy as np

from ..eval import metrics


class HarnessError(RuntimeError):
    """The training path cannot proceed."""


class Backbone(Protocol):
    """What the harness needs from a model. Torch and stub both satisfy it."""

    def reset(self, train_labels: np.ndarray) -> None:
        """Initialise before any optimiser step.

        ``train_labels`` is passed so a head can be initialised to the
        training-fold mean, which is what gate 3 then checks.
        """

    def train_epoch(self, features: np.ndarray, labels: np.ndarray) -> float:
        """One pass. Returns mean training loss."""

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Predictions on the raw 1-5 scale."""


@dataclass(frozen=True)
class TrainConfig:
    max_epochs: int = 40
    #: Identical across every arm (PLAN §4.6). Not a per-arm knob.
    patience: int = 5
    inner_val_frac: float = 0.2
    seed: int = 1337
    #: What early stopping watches. MSE is the training objective; selecting on
    #: the reported metric would be selecting on the thing being claimed. Both
    #: are recorded per epoch regardless, because the void ladder's collapse was
    #: visible in PCC while the loss looked unremarkable.
    monitor: str = "inner_val_mse"
    #: Gate 3 tolerances, as a fraction. See ``assert_epoch0_calibrated``.
    epoch0_mean_tol: float = 0.5
    epoch0_mse_tol: float = 0.6

    def __post_init__(self) -> None:
        if self.monitor not in ("inner_val_mse", "inner_val_pcc"):
            raise HarnessError(f"unknown monitor {self.monitor!r}")
        if not 0.0 < self.inner_val_frac < 0.5:
            raise HarnessError(f"inner_val_frac out of range: {self.inner_val_frac}")
        if self.patience < 1 or self.max_epochs < 1:
            raise HarnessError("patience and max_epochs must be positive")


@dataclass
class FoldRun:
    fold: int
    train_ids: list[int]
    inner_val_ids: list[int]
    test_ids: list[int]
    selected_epoch: int
    epoch0_pred_mean: float
    epoch0_inner_val_mse: float
    inner_val_var: float
    inner_val_mean: float
    train_label_mean: float
    train_label_var: float
    curve: list[dict] = field(default_factory=list)
    test_predictions: np.ndarray | None = None


@dataclass
class CVResult:
    folds: list[FoldRun]
    oof_ids: list[int]
    oof_predictions: np.ndarray
    oof_truth: np.ndarray

    def metrics(self) -> dict:
        """Pooled OOF scalars. SHAREABLE."""
        return {
            "n": len(self.oof_ids),
            "pcc": metrics.pcc(self.oof_truth, self.oof_predictions),
            "spearman": metrics.spearman(self.oof_truth, self.oof_predictions),
            "qwk_3cat": metrics.qwk_3cat(self.oof_truth, self.oof_predictions),
            "mae": metrics.mae(self.oof_truth, self.oof_predictions),
            "rmse": metrics.rmse(self.oof_truth, self.oof_predictions),
            "selected_epochs": [f.selected_epoch for f in self.folds],
        }

    def fold_record(self) -> dict:
        """Exactly what gate 6 reconstructs from. CLUSTER-ONLY: patient-keyed."""
        return {
            "folds": [
                {
                    "fold": f.fold,
                    "train_ids": list(f.train_ids),
                    "inner_val_ids": list(f.inner_val_ids),
                    "test_ids": list(f.test_ids),
                    "selected_epoch": f.selected_epoch,
                }
                for f in self.folds
            ]
        }


# --------------------------------------------------------------------------
# gate 3 -- epoch-0 calibration
# --------------------------------------------------------------------------


def assert_epoch0_calibrated(
    *,
    pred_mean: float,
    inner_val_mse: float,
    inner_val_var: float,
    inner_val_mean: float,
    train_label_mean: float,
    train_label_var: float,
    mean_tol: float = 0.5,
    mse_tol: float = 0.6,
    where: str = "",
) -> None:
    """Gate 3. Before any optimiser step, an untrained head predicts the mean.

    **[MEASURED] why this exists.** In the void ladder, ImageNet SR-GNN arms
    started at ``inner_val_mse`` 1.12-8.63 against 0.40-0.88 for every other arm,
    because they had no BatchNorm running statistics to load. Twenty fold-runs of
    LayerNorm backbones were clean. One assertion here would have caught it on
    first appearance rather than after two void ladders.

    Two checks, deliberately different in kind:

    **Head position** -- ``pred_mean`` against the training-fold mean. This is the
    one that catches the defect: a head on the wrong scale predicts the wrong
    value.

    **Dispersion** -- the MSE against what a *constant* predictor at ``pred_mean``
    would score. For a constant prediction the identity is exact::

        mse = var(inner) + (mean(inner) - pred_mean)^2

    so the ratio is 1 whatever the sample size, and a departure means the epoch-0
    outputs are not constant -- an untrained head emitting varying garbage, which
    the mean check can miss when large positive and negative errors cancel.

    Both are sample-size robust, which two earlier drafts were not. Comparing the
    MSE against the raw *training* variance failed a clean stub at ratio 2.09;
    against the raw *inner-val* variance another failed at 3.21, because on a
    small inner-val split ``mean(inner) - mean(train)`` is large by sampling
    alone. That difference is not noise to be tolerated with a wider band -- it is
    computable, so it is computed.
    """
    label_sd = float(np.sqrt(train_label_var))
    prefix = f"{where}: " if where else ""

    mean_error = abs(pred_mean - train_label_mean)
    if label_sd > 0 and mean_error > mean_tol * label_sd:
        raise HarnessError(
            f"{prefix}epoch-0 predicted mean {pred_mean:.4f} is {mean_error:.4f} "
            f"from the training-fold mean {train_label_mean:.4f} "
            f"({mean_error / label_sd:.2f} SD, limit {mean_tol}). An untrained "
            "head should predict the mean; this one does not. In the void ladder "
            "this signature was missing BatchNorm running statistics, which left "
            "epoch-0 activations on the wrong scale and produced two void ladders "
            "before anyone looked at the label variance."
        )

    if train_label_var <= 0:
        raise HarnessError(f"{prefix}training-fold labels have zero variance")
    if inner_val_var <= 0:
        raise HarnessError(f"{prefix}inner-val labels have zero variance")

    # What a constant predictor at pred_mean would score. Exact, not an estimate.
    expected = inner_val_var + (inner_val_mean - pred_mean) ** 2
    if expected <= 0:
        raise HarnessError(f"{prefix}expected epoch-0 MSE is not positive")

    ratio = inner_val_mse / expected
    if not (1.0 - mse_tol) <= ratio <= (1.0 + mse_tol):
        raise HarnessError(
            f"{prefix}epoch-0 inner_val_mse {inner_val_mse:.4f} is {ratio:.2f}x the "
            f"{expected:.4f} a constant prediction at {pred_mean:.4f} would score, "
            f"outside [{1 - mse_tol:.2f}, {1 + mse_tol:.2f}]. An untrained head "
            "should emit one value; this one's outputs vary. In the void ladder "
            "the same root cause -- missing BatchNorm running statistics leaving "
            "epoch-0 activations on the wrong scale -- produced inner_val_mse of "
            "1.12-8.63 against 0.40-0.88 for every clean arm."
        )


# --------------------------------------------------------------------------
# the split
# --------------------------------------------------------------------------


def inner_val_split(
    train_ids: list[int], frac: float, seed: int, fold: int
) -> tuple[list[int], list[int]]:
    """Carve an inner-val set out of the TRAINING folds only.

    Not a leak: inner-val comes from the training folds, never from the test
    fold. Gate 6 reconstructs this independently and checks it.

    Seeded per fold so the five folds do not all hold out the same positions,
    and deterministic so a rerun reproduces the split exactly.
    """
    if len(train_ids) < 2:
        raise HarnessError(f"fold {fold}: too few training patients to split")
    order = np.random.default_rng(seed + fold).permutation(len(train_ids))
    n_val = max(1, int(round(len(train_ids) * frac)))
    if n_val >= len(train_ids):
        raise HarnessError(f"fold {fold}: inner_val_frac leaves no training data")

    val_index = set(order[:n_val].tolist())
    inner = [train_ids[i] for i in sorted(val_index)]
    inner_set = set(inner)
    return [pid for pid in train_ids if pid not in inner_set], inner


# --------------------------------------------------------------------------
# the loop
# --------------------------------------------------------------------------


def run_fold(
    *,
    fold: int,
    features: np.ndarray,
    labels: np.ndarray,
    patient_ids: list[int],
    assignments: dict[int, int],
    make_backbone: Callable[[], Backbone],
    config: TrainConfig,
    log=lambda *_: None,
) -> FoldRun:
    """One fold: split, initialise, check epoch 0, train with early stopping."""
    index_of = {pid: i for i, pid in enumerate(patient_ids)}
    test_ids = [pid for pid in patient_ids if assignments[pid] == fold]
    outer_train = [pid for pid in patient_ids if assignments[pid] != fold]
    if not test_ids:
        raise HarnessError(f"fold {fold} has no test patients")

    train_ids, inner_ids = inner_val_split(
        outer_train, config.inner_val_frac, config.seed, fold
    )

    def rows(ids):
        return np.array([index_of[pid] for pid in ids], dtype=int)

    train_rows, inner_rows, test_rows = rows(train_ids), rows(inner_ids), rows(test_ids)
    train_labels = labels[train_rows]
    inner_labels = labels[inner_rows]

    backbone = make_backbone()
    backbone.reset(train_labels)

    # --- gate 3, before any optimiser step --------------------------------
    epoch0 = backbone.predict(features[inner_rows])
    epoch0_mean = float(np.mean(epoch0))
    epoch0_mse = float(np.mean((epoch0 - inner_labels) ** 2))
    train_mean = float(np.mean(train_labels))
    train_var = float(np.var(train_labels))
    inner_var = float(np.var(inner_labels))

    assert_epoch0_calibrated(
        pred_mean=epoch0_mean,
        inner_val_mse=epoch0_mse,
        inner_val_var=inner_var,
        inner_val_mean=float(np.mean(inner_labels)),
        train_label_mean=train_mean,
        train_label_var=train_var,
        mean_tol=config.epoch0_mean_tol,
        mse_tol=config.epoch0_mse_tol,
        where=f"fold {fold}",
    )

    # --- train with inner-val early stopping ------------------------------
    curve: list[dict] = []
    best_score = None
    best_epoch = 0
    best_predictions = backbone.predict(features[test_rows])
    since_improvement = 0

    for epoch in range(1, config.max_epochs + 1):
        train_loss = backbone.train_epoch(features[train_rows], train_labels)
        predictions = backbone.predict(features[inner_rows])
        inner_mse = float(np.mean((predictions - inner_labels) ** 2))
        inner_pcc = metrics.pcc(inner_labels, predictions)

        curve.append(
            {
                "epoch": epoch,
                "train_loss": float(train_loss),
                "inner_val_mse": inner_mse,
                "inner_val_pcc": inner_pcc,
            }
        )

        score = inner_mse if config.monitor == "inner_val_mse" else -inner_pcc
        if np.isnan(score):
            score = float("inf")

        if best_score is None or score < best_score - 1e-12:
            best_score, best_epoch = score, epoch
            best_predictions = backbone.predict(features[test_rows])
            since_improvement = 0
        else:
            since_improvement += 1
            if since_improvement >= config.patience:
                log(f"  fold {fold}: stopped at epoch {epoch}, best {best_epoch}")
                break

    return FoldRun(
        fold=fold,
        train_ids=train_ids,
        inner_val_ids=inner_ids,
        test_ids=test_ids,
        selected_epoch=best_epoch,
        epoch0_pred_mean=epoch0_mean,
        epoch0_inner_val_mse=epoch0_mse,
        inner_val_var=inner_var,
        inner_val_mean=float(np.mean(inner_labels)),
        train_label_mean=train_mean,
        train_label_var=train_var,
        curve=curve,
        test_predictions=best_predictions,
    )


def run_cv(
    *,
    features: np.ndarray,
    labels: np.ndarray,
    patient_ids: list[int],
    assignments: dict[int, int],
    make_backbone: Callable[[], Backbone],
    config: TrainConfig | None = None,
    log=lambda *_: None,
) -> CVResult:
    """Every fold, pooled to out-of-fold predictions."""
    config = config or TrainConfig()
    if len(patient_ids) != len(labels) or len(patient_ids) != len(features):
        raise HarnessError(
            f"length mismatch: {len(patient_ids)} ids, {len(labels)} labels, "
            f"{len(features)} feature rows"
        )
    missing = [pid for pid in patient_ids if pid not in assignments]
    if missing:
        raise HarnessError(f"{len(missing)} patient(s) have no fold: {missing[:8]}")

    folds = sorted(set(assignments[pid] for pid in patient_ids))
    runs = [
        run_fold(
            fold=fold,
            features=features,
            labels=labels,
            patient_ids=patient_ids,
            assignments=assignments,
            make_backbone=make_backbone,
            config=config,
            log=log,
        )
        for fold in folds
    ]

    label_of = dict(zip(patient_ids, labels))
    oof_ids: list[int] = []
    oof_pred: list[float] = []
    for run in runs:
        oof_ids.extend(run.test_ids)
        oof_pred.extend(np.asarray(run.test_predictions).ravel().tolist())

    order = np.argsort(oof_ids)
    ordered_ids = [oof_ids[i] for i in order]
    return CVResult(
        folds=runs,
        oof_ids=ordered_ids,
        oof_predictions=np.array([oof_pred[i] for i in order], dtype=float),
        oof_truth=np.array([label_of[pid] for pid in ordered_ids], dtype=float),
    )
