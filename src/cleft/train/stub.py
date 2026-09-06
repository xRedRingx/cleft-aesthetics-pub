"""A torch-free backbone, so the harness can be tested on the laptop.

Two jobs:

1. **Stand in for ViT** while the harness is exercised end to end. It learns a
   linear map on whatever feature vector it is given, which is enough to produce
   a curve, an early stop, and out-of-fold predictions.

2. **Reproduce the defect gate 3 exists to catch.** ``BrokenHeadBackbone`` starts
   with a head whose predictions sit far from the training-fold mean, which is
   what missing BatchNorm running statistics do to an ImageNet SR-GNN arm.

On (2): a contrived "some wrong number" would prove the assertion fires, not that
it fires on *this* defect. The void ladder's signature was specific -- epoch-0
``inner_val_mse`` of **1.12-8.63** against **0.40-0.88** for every clean arm, a
factor of roughly 2 to 20 above label variance. ``BrokenHeadBackbone`` is
parameterised to land in that band, so the teeth-test demonstrates the same
failure rather than a different one that happens to trip the same check.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

#: The measured void-ladder bands (PLAN §4.6, Phase 3 brief §3 gate 3).
CLEAN_EPOCH0_MSE = (0.40, 0.88)
BROKEN_EPOCH0_MSE = (1.12, 8.63)


def flatten(features: np.ndarray) -> np.ndarray:
    """(N, ...) -> (N, D) float. The stub does not care what the axes meant."""
    array = np.asarray(features, dtype=float)
    return array.reshape(array.shape[0], -1) if array.ndim > 1 else array[:, None]


@dataclass
class StubBackbone:
    """Gradient descent on a linear head. Deterministic, torch-free, and it learns.

    Full-batch GD rather than closed-form ridge: successive epochs improve
    smoothly and then plateau, which is the shape early stopping needs to have
    something real to stop on. A closed-form solver jumps to its optimum in one
    step and then only moves as the regulariser changes, which made the patience
    test unfalsifiable.

    Features are standardised on the first training batch and the statistics
    reused, so one learning rate works whatever scale the caller passes.
    """

    learning_rate: float = 0.15
    #: Same shape of report the torch backbones produce, so the parameter/policy
    #: consistency check is exercised on the laptop rather than only where torch
    #: is installed. The stub has no frozen backbone: it *is* the whole model.
    parameter_report: dict = field(default_factory=dict)
    _weights: np.ndarray | None = field(default=None, repr=False)
    _bias: float = 0.0
    _centre: np.ndarray | None = field(default=None, repr=False)
    _scale: np.ndarray | None = field(default=None, repr=False)

    def reset(self, train_labels: np.ndarray) -> None:
        # The head starts at the training-fold mean: an untrained model that
        # predicts the mean is exactly what gate 3 asserts.
        self._weights = None
        self._centre = None
        self._scale = None
        self._bias = float(np.mean(train_labels))

    def _standardise(self, design: np.ndarray) -> np.ndarray:
        if self._centre is None:
            self._centre = design.mean(axis=0)
            spread = design.std(axis=0)
            self._scale = np.where(spread > 1e-12, spread, 1.0)
        return (design - self._centre) / self._scale

    def train_epoch(self, features: np.ndarray, labels: np.ndarray) -> float:
        design = flatten(features)
        if self._centre is None:
            self._standardise(design)
        scaled = self._standardise(design)
        if self._weights is None:
            self._weights = np.zeros(scaled.shape[1], dtype=float)
            self.parameter_report = {
                "total_parameters": int(self._weights.size) + 1,
                "trainable_parameters": int(self._weights.size) + 1,
            }

        residual = scaled @ self._weights + self._bias - labels
        self._weights -= self.learning_rate * 2.0 * scaled.T @ residual / len(labels)
        self._bias -= self.learning_rate * 2.0 * float(residual.mean())
        return float(np.mean(residual**2))

    def predict(self, features: np.ndarray) -> np.ndarray:
        design = flatten(features)
        if self._weights is None or self._centre is None:
            return np.full(design.shape[0], self._bias, dtype=float)
        return self._standardise(design) @ self._weights + self._bias


@dataclass
class BrokenHeadBackbone(StubBackbone):
    """A head that does NOT start at the training mean.

    The SR-GNN signature: no BatchNorm running statistics to load, so epoch-0
    activations are on the wrong scale and the head's predictions are far from
    the label mean. ``offset`` is in label units and is chosen so epoch-0
    ``inner_val_mse`` lands in the measured broken band rather than merely being
    "wrong".
    """

    offset: float = 1.5

    def reset(self, train_labels: np.ndarray) -> None:
        super().reset(train_labels)
        self._bias = float(np.mean(train_labels)) + self.offset


def offset_for_target_mse(label_variance: float, target_mse: float) -> float:
    """The head offset that produces a given epoch-0 MSE.

    A constant prediction offset by ``d`` from the mean gives
    ``mse = variance + d^2``, so ``d = sqrt(target - variance)``. Used to place
    the broken stub inside the measured 1.12-8.63 band for a given cohort rather
    than guessing an offset and hoping.
    """
    if target_mse <= label_variance:
        raise ValueError(
            f"target MSE {target_mse} is not above the label variance "
            f"{label_variance}; no offset can produce it"
        )
    return float(np.sqrt(target_mse - label_variance))
