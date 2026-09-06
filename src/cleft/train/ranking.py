"""Pairwise-logistic ranking: Phase 22's six arms.

Every other arm in this project fits an ABSOLUTE target -- MSE on the raw
1-5 scale (``models/factory.py``), or LDL's distribution. This one fits an
ORDER: the only supervision is that patient i outranks patient j.

**The settings are RULED, not chosen here** (``phase22``):

* **Loss** -- pairwise logistic, hard targets (``LOSS_RULED``,
  ``TARGET_MAP_RULED``). No margin: the logistic needs none, which is one
  fewer undeclarable constant than a hinge.
* **Ties dropped** (``TIE_RULE_RULED``). A tie carries no order, so under
  a hard-target loss training on it teaches nothing true. Measured at
  2,345 of 27,966 pairs.
* **Pairs** -- all of them, every epoch, formed INSIDE ``train_epoch``
  (``PAIR_CONSTRUCTION_RULED``). Nothing is sampled, so there is no
  sampler seed to declare.
* **Head** -- the shipped scalar head, in two versions
  (``HEAD_BOUNDING_RULED``): **bounded** onto 1-5, and **unbounded**
  (order only).

**Fold honesty is STRUCTURAL, not policed.** The frozen harness calls
``backbone.train_epoch(features[train_rows], train_labels)`` -- only that
fold's training rows. Pairs formed inside that call are between two
training patients *by construction*, and no code path exists by which one
could span the train/test boundary. Nothing has to check that it does not.

**This module deliberately mirrors ``train/ldl.py``**, which is the
project's precedent for a non-MSE objective: the arithmetic that can be
checked without torch lives in module-level functions and is tested
directly; the fit loop is thin and imports torch inside its methods. And
it mirrors ``EmbeddingHeadBackbone``'s optimiser, learning rate, weight
decay and seeding, so a ranking arm varies **the objective and nothing
else** -- the same argument LDL makes for itself.

**Frozen apparatus is untouched.** ``train_epoch(features, labels) ->
float`` is the Protocol in ``train/harness.py``; eight backbones already
implement it and this is the ninth.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

#: The label scale every bounded score lives on (PLAN 4.6).
LABEL_MIN = 1.0
LABEL_MAX = 5.0


class RankingError(ValueError):
    """A ranking arm was asked for something it cannot honestly do."""


# --------------------------------------------------------------------------
# pair construction -- the ruled rule, testable without torch
# --------------------------------------------------------------------------


def ordered_pairs(
    labels: np.ndarray, min_separation: float = 0.0
) -> tuple[np.ndarray, np.ndarray]:
    """Every strictly-ordered pair of the rows given, ties DROPPED.

    Returns ``(high, low)`` index arrays into ``labels``, one entry per
    pair, with ``labels[high] > labels[low]`` throughout.

    **All pairs, every epoch** -- nothing sampled, so the result is a
    deterministic function of the labels and no sampler seed exists.

    **The indices are into the array passed in.** When the harness hands
    over one fold's training rows, that is the only thing this function
    can see, which is why fold honesty needs no check.

    **[ADDED 2026-09-01] ``min_separation`` IS R-clear.** A pair is kept
    only when ``|label_i - label_j| >= min_separation``. R-all passes
    ``0.0`` and keeps every ordered pair; R-clear passes SE_diff and
    keeps only pairs the panel can resolve. **This parameter did not
    exist until now**, which is why R-all and R-clear trained on
    identical pair sets and produced identical results
    (``phase22.PAIR_SOURCE_WAS_NOT_CONSUMED``).

    The comparison is ``>=``, matching the separation run's strict ``<``
    for "below SE_diff": a pair is below the threshold or it clears it,
    with no pair in both sets and none in neither.
    """
    values = np.asarray(labels, dtype=float)
    n = len(values)
    i, j = np.triu_indices(n, 1)
    difference = values[i] - values[j]
    # Ties dropped (phase22.TIE_RULE_RULED), then R-clear's restriction
    # (phase22.ARM_R_CLEAR_REGISTERED); the rest oriented high-first.
    ordered = (difference != 0.0) & (np.abs(difference) >= float(min_separation))
    i, j, difference = i[ordered], j[ordered], difference[ordered]
    high = np.where(difference > 0, i, j)
    low = np.where(difference > 0, j, i)
    return high, low


def pair_census(labels: np.ndarray, min_separation: float = 0.0) -> dict:
    """What the pair set cost, for the run log. Aggregates only.

    Separates the two exclusions, because they are two different
    rulings: ties are dropped from EVERY arm (``TIE_RULE_RULED``), and
    below-threshold pairs are dropped only by R-clear.
    """
    values = np.asarray(labels, dtype=float)
    total = len(values) * (len(values) - 1) // 2
    untied = len(ordered_pairs(values, 0.0)[0])
    kept = len(ordered_pairs(values, min_separation)[0])
    return {
        "n_rows": int(len(values)),
        "n_pairs_total": int(total),
        "n_pairs_tied_dropped": int(total - untied),
        "n_pairs_below_threshold_dropped": int(untied - kept),
        "n_pairs_trained": int(kept),
        "min_separation": float(min_separation),
        "tie_fraction": float(total - untied) / total if total else 0.0,
    }


def pairwise_logistic_loss(
    scores: np.ndarray, high: np.ndarray, low: np.ndarray
) -> float:
    """``mean softplus(-(s_high - s_low))`` -- the numpy reference.

    RankNet's cross-entropy at a hard target: the target probability is
    1, so the loss reduces to ``log(1 + exp(-gap))``. It reads GAPS, not
    levels, and **has no finite optimum** -- it keeps falling as the gap
    widens, which is what "aims at order alone" means
    (``phase22.ATTRACTOR_FINDING``).

    The torch path below computes the same quantity; this one exists so
    the arithmetic is checked without torch.
    """
    if len(high) == 0:
        return 0.0
    gap = np.asarray(scores, dtype=float)[high] - np.asarray(
        scores, dtype=float
    )[low]
    return float(np.mean(np.logaddexp(0.0, -gap)))


# --------------------------------------------------------------------------
# the two heads
# --------------------------------------------------------------------------


def parameter_report(bounded: bool, min_separation: float = 0.0) -> dict:
    """What the arm trains, as a record. Module-level so it is testable
    without torch, which is the whole point of this module's shape."""
    return {
        "loss": "pairwise_logistic",
        "targets": "hard",
        "tied_pairs": "dropped",
        "min_separation": float(min_separation),
        "head": "bounded_1_to_5" if bounded else "unbounded",
        "monitor": "inner_val_mse" if bounded else "inner_val_pcc",
        "pair_construction": "all_pairs_per_epoch_inside_train_epoch",
        "reported_scalar": (
            "the score itself" if bounded
            else "the RAW score -- NOT on the label scale"
        ),
    }


def bounded_scores(raw: np.ndarray) -> np.ndarray:
    """``1 + 4 * sigmoid(raw)`` -- monotone into [1, 5].

    Monotone, so the ordering the loss trains is preserved exactly; and
    on the label scale, so MSE monitoring means what it means for every
    other arm (``phase22.MONITOR_BIND_RESOLVED``).

    **It SATURATES.** Beyond about |raw| = 37 the logistic underflows in
    float64 and the score is exactly 1.0 or 5.0. That is harmless here --
    both are inside the label scale and both metrics accept them -- but
    it is why the range is the CLOSED [1, 5] and not the open interval:
    a bounded arm CAN emit an endpoint, and a reader checking the
    predictions should not treat one as a defect.
    """
    values = np.asarray(raw, dtype=float)
    span = LABEL_MAX - LABEL_MIN
    # Numerically stable logistic.
    return LABEL_MIN + span / (1.0 + np.exp(-values))


def bias_for_mean(mean: float, *, bounded: bool) -> float:
    """The bias at which an untrained head predicts ``mean``.

    **[CORRECTED 2026-09-01.** The original read: *"Gate 3 by
    construction, exactly as EmbeddingHeadBackbone does it: weights at
    zero and the bias placed so the epoch-0 prediction is the
    training-fold mean."* **Two things were wrong.** ``reset`` sets
    ``_weights`` to ``None``, not zero -- the width is unknown until
    features arrive. And while the bias arithmetic here was right, the
    head **could not predict at all** before its first ``train_epoch``,
    so "the untrained head predicts the training-fold mean" described a
    path that did not exist. The arithmetic held; the delivery did not.
    See ``phase22.GATE_3_CLAIM_CORRECTED``.**

    What is true now: ``reset`` places the bias here, ``_weights``
    stays ``None`` until the first ``train_epoch``, and ``predict``
    returns this bias for every row while that is so -- which makes the
    epoch-0 prediction the training-fold mean, as gate 3 requires. The
    frozen gate is satisfied rather than amended.
    """
    value = float(mean)
    if not bounded:
        return value
    span = LABEL_MAX - LABEL_MIN
    if not LABEL_MIN < value < LABEL_MAX:
        raise RankingError(
            f"a bounded head cannot predict {value}: it lives strictly "
            f"inside the label scale ({LABEL_MIN}, {LABEL_MAX}), so a "
            "training-fold mean at or beyond the endpoint has no bias. "
            "A cohort that produced one would not be this cohort."
        )
    fraction = (value - LABEL_MIN) / span
    return float(np.log(fraction / (1.0 - fraction)))


# --------------------------------------------------------------------------
# the backbone -- the ninth implementation of the frozen Protocol
# --------------------------------------------------------------------------


@dataclass
class RankingHeadBackbone:
    """A linear head over frozen embeddings, fitted under pairwise logistic.

    **Deliberately mirrors ``EmbeddingHeadBackbone``** -- same optimiser,
    learning rate, weight decay and seeding -- so the arm varies the
    OBJECTIVE and nothing else. A ranking arm on a different optimiser
    would confound "does ordering help" with "does this optimiser suit
    this problem", and the phase exists to answer the first.

    Torch, for that reason. The parts that can be checked without it --
    pair construction, the tie rule, the loss, both head maps and the
    gate-3 bias -- are the module-level functions above.
    """

    #: Scores on the 1-5 label scale, or order only. Both are built;
    #: the contrast between them is three of the fifteen (phase22).
    bounded: bool = True
    #: **R-clear's restriction, and the ONLY thing separating it from
    #: R-all.** 0.0 keeps every ordered pair; SE_diff keeps only pairs
    #: the panel can resolve. [ADDED 2026-09-01 -- until then the two
    #: arms trained on identical pair sets.]
    min_separation: float = 0.0
    learning_rate: float = 1e-3
    weight_decay: float = 0.01
    max_steps: int = 50
    seed: int = 1337
    parameter_report: dict = field(default_factory=dict)

    _weights: Any = field(default=None, repr=False)
    _bias: Any = field(default=None, repr=False)
    _optimizer: Any = field(default=None, repr=False)
    _device: Any = field(default=None, repr=False)
    _census: dict = field(default_factory=dict, repr=False)

    def reset(self, train_labels: np.ndarray) -> None:
        import torch

        torch.manual_seed(self.seed)
        self._device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        # Weights at zero, bias placed so the untrained head predicts the
        # training-fold mean -- gate 3, satisfied by construction.
        self._weights = None
        self._bias = torch.tensor(
            bias_for_mean(float(np.mean(train_labels)), bounded=self.bounded),
            device=self._device,
            requires_grad=True,
        )
        self._optimizer = None
        self.parameter_report = parameter_report(
            self.bounded, self.min_separation
        )

    def _ensure(self, dim: int) -> None:
        import torch

        if self._optimizer is not None:
            return
        self._weights = torch.zeros(
            dim, dtype=torch.float32, device=self._device, requires_grad=True
        )
        self._optimizer = torch.optim.AdamW(
            [self._weights, self._bias],
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        self.parameter_report["total_parameters"] = int(dim + 1)
        self.parameter_report["trainable_parameters"] = int(dim + 1)

    def _score(self, embeddings):
        raw = embeddings @ self._weights + self._bias
        if not self.bounded:
            return raw
        import torch

        span = LABEL_MAX - LABEL_MIN
        return LABEL_MIN + span * torch.sigmoid(raw)

    def train_epoch(self, features: np.ndarray, labels: np.ndarray) -> float:
        """One full-batch pass over EVERY ordered pair of these rows.

        ``features``/``labels`` are the fold's TRAINING rows and nothing
        else, so the pairs formed here are fold-honest by construction.
        """
        import torch

        embeddings = np.asarray(features, dtype=np.float32)
        self._ensure(embeddings.shape[1])
        high, low = ordered_pairs(labels, self.min_separation)
        self._census = pair_census(labels, self.min_separation)
        if len(high) == 0:
            # Every label in this fold is equal: no order to learn, and
            # no pair to learn it from. Reported, never silently zero.
            raise RankingError(
                f"no trainable pairs from {self._census['n_rows']} rows: "
                f"{self._census['n_pairs_tied_dropped']} dropped as ties and "
                f"{self._census['n_pairs_below_threshold_dropped']} as below "
                f"min_separation={self.min_separation}. A ranking arm has "
                "nothing to fit here."
            )
        x = torch.as_tensor(embeddings, device=self._device)
        high_t = torch.as_tensor(np.asarray(high), device=self._device)
        low_t = torch.as_tensor(np.asarray(low), device=self._device)

        last = 0.0
        for _ in range(self.max_steps):
            self._optimizer.zero_grad(set_to_none=True)
            scores = self._score(x)
            gap = scores[high_t] - scores[low_t]
            # softplus(-gap): RankNet's cross-entropy at a hard target.
            loss = torch.nn.functional.softplus(-gap).mean()
            loss.backward()
            self._optimizer.step()
            last = float(loss.detach().cpu())
        return last

    def census(self) -> dict:
        """The last epoch's pair census, for the run log. Empty before
        the first ``train_epoch``."""
        return dict(self._census)

    def predict(self, features: np.ndarray) -> np.ndarray:
        """**The harness calls this BEFORE any ``train_epoch``** -- gate 3
        runs on an untrained head, and again for ``best_predictions``
        before the loop starts.

        [FIXED 2026-09-01] At that point ``_weights`` is still ``None``,
        because the embedding width is not known until features arrive.
        ``EmbeddingHeadBackbone.predict`` handles exactly this by
        returning the bias for every row, and this mirrors it. **The
        check belongs here and not in ``_score``**: ``_score`` is the
        trained forward pass, and putting a None-check inside it would
        make every training step carry a branch that only the untrained
        case needs.
        """
        import torch

        with torch.no_grad():
            if self._weights is None:
                # reset() placed the bias so this equals the
                # training-fold mean -- which is what gate 3 asserts.
                raw = float(self._bias.item())
                value = (
                    float(bounded_scores(np.asarray([raw]))[0])
                    if self.bounded else raw
                )
                return np.full(len(features), value, dtype=float)
            x = torch.as_tensor(
                np.asarray(features, dtype=np.float32), device=self._device
            )
            return self._score(x).cpu().numpy()
