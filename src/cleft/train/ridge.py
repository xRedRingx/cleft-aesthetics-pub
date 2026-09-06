"""Closed-form ridge, for arms whose features are already the model.

Phase 4 §3.1. The mirror-difference index produces 22 interpretable features; the
regressor over them should be the simplest thing that can map them to a grade, so
that a win or a loss is attributable to the *index* rather than to whatever the
head learned. Closed form, one fixed alpha, no iteration.

**Torch-free**, so the whole arm is exercised on the laptop. That is not a
convenience -- it is what lets the features and the regressor be tested against a
known asymmetry before any cluster time is spent.

**On seeds, and what this arm's SD measures.** A closed-form solve has no
initialisation and no early stopping, so the obvious expectation is that the seed
changes nothing and the variance is exactly zero. **That is not what happens
here**, and the reason is worth stating: ``harness.inner_val_split`` seeds off
``config.seed``, so the seed decides *which patients are held out of training*.
The fit therefore moves with the seed even though the solver is deterministic.

So this arm does have a seed band, and it measures **split sensitivity** rather
than the initialisation sensitivity the Phase 3 probe's band measures. Both are
"seed variance"; they are not the same quantity, and PLAN §4.12.1 is why each arm
reports its own rather than inheriting one.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


class RidgeError(RuntimeError):
    """The ridge arm cannot be fitted as asked."""


#: Fixed, and deliberately not selected against anything. Alpha could legitimately
#: be chosen on the inner-val split -- that is not leakage -- but every knob turned
#: is a degree of freedom the comparison has to carry, and the point of this arm is
#: to be the cheapest instrument that could work. If it needs tuning to beat the
#: probe, it did not beat the probe.
DEFAULT_ALPHA = 1.0


@dataclass
class RidgeBackbone:
    """Ridge regression on standardised features, satisfying the harness protocol.

    Fits in one step. ``train_epoch`` is idempotent by construction, so early
    stopping selects epoch 1 and stops at patience -- which is honest for a
    closed-form solver and is what the recorded ``selected_epoch`` should say.
    """

    alpha: float = DEFAULT_ALPHA
    seed: int = 1337
    parameter_report: dict = field(default_factory=dict)

    _weights: np.ndarray | None = field(default=None, repr=False)
    _bias: float = 0.0
    _centre: np.ndarray | None = field(default=None, repr=False)
    _scale: np.ndarray | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.alpha < 0:
            raise RidgeError(f"alpha must be non-negative, got {self.alpha}")

    # -- harness protocol --------------------------------------------------

    def reset(self, train_labels: np.ndarray) -> None:
        """Un-fitted, predicting the training-fold mean -- which is gate 3."""
        self._weights = None
        self._centre = None
        self._scale = None
        self._bias = float(np.mean(train_labels))

    def train_epoch(self, features: np.ndarray, labels: np.ndarray) -> float:
        design = _flatten(features)
        if self._centre is None:
            self._centre = design.mean(axis=0)
            spread = design.std(axis=0)
            # A feature that is constant across the training fold carries no
            # information; dividing by its zero spread would make it infinite
            # rather than useless.
            self._scale = np.where(spread > 1e-12, spread, 1.0)

        scaled = (design - self._centre) / self._scale
        target = np.asarray(labels, dtype=float)
        self._bias = float(target.mean())

        # Centred targets, so the penalty never touches the intercept -- shrinking
        # the intercept would pull every prediction toward zero rather than toward
        # the mean, which on a 1-5 scale is a large and entirely artificial bias.
        gram = scaled.T @ scaled + self.alpha * np.eye(scaled.shape[1])
        self._weights = np.linalg.solve(gram, scaled.T @ (target - self._bias))

        self.parameter_report = {
            "total_parameters": int(self._weights.size) + 1,
            "trainable_parameters": int(self._weights.size) + 1,
        }
        return float(np.mean((self.predict(features) - target) ** 2))

    def predict(self, features: np.ndarray) -> np.ndarray:
        design = _flatten(features)
        if self._weights is None or self._centre is None:
            return np.full(design.shape[0], self._bias, dtype=float)
        return ((design - self._centre) / self._scale) @ self._weights + self._bias


def _flatten(features: np.ndarray) -> np.ndarray:
    array = np.asarray(features, dtype=float)
    return array.reshape(array.shape[0], -1) if array.ndim > 1 else array[:, None]
