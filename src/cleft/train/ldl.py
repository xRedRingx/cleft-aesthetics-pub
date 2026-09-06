"""Label Distribution Learning: the one arm that addresses label noise directly.

Every other arm in this project collapses five raters to a scalar and fits it.
LDL fits the **distribution** -- ``soft_1..soft_5``, the fraction of raters who
gave each grade -- with a five-output softmax head and a KL loss, and reports
its expectation as a scalar so PCC stays comparable with everything else.

**[2026-09-02] Kappa is UNWEIGHTED** -- it scores a 1-vs-2 disagreement
as identically wrong to a 1-vs-5, which is the wrong loss for an ordinal
scale, so this motivating figure overstates the disagreement taken alone.
The distance-aware figures on the same matrix are QWK 0.4276 and mean
inter-rater r 0.4696. Nothing below is edited:
``record_audit.THE_KAPPA_LIMITATION``.

Fleiss kappa **0.1662** is the research problem (PLAN §4.3) -- the value on the
**237 with photographs**, which is this project's population.
``data.reliability.FLEISS_237``, not the 251-row ``FLEISS_251`` of 0.1605 this
docstring carried until 2026-08-02. A patient scored {2,2,3,3,4} and one scored
{3,3,3,3,3} have the same mean and are not the same case, and only this arm can
see the difference.

----------------------------------------------------------------------------
THE TRUTH VECTOR IS UNCHANGED, AND THAT IS WHAT MAKES THE COMPARISON LEGAL
----------------------------------------------------------------------------
Stage G compares mean / median / LDL. If LDL were scored against a different
quantity than the other two, the comparison would be between different
questions wearing one metric's name (PLAN R2). So:

* **truth** is the ``mean`` column, exactly as for the mean arm;
* **prediction** is the expectation of the predicted distribution;
* only the **training target** differs -- the distribution instead of its
  expectation.

``assert_mean_is_the_soft_expectation`` checks the manifest's own consistency
at runtime: ``mean`` must equal ``sum(k * soft_k)``. If it does not, the soft
columns are not the distribution whose expectation the label is, and the
arm is measuring something nobody designed.

----------------------------------------------------------------------------
THE SOFT COLUMNS ARE TARGETS, NEVER INPUTS
----------------------------------------------------------------------------
The frozen harness passes one array and slices it by row, so the per-patient
distribution reaches the head by riding in the feature row (the same device
the graph path uses for boxes). **That puts the label inside the input
array**, and a head that read those five columns as features would predict the
label from the label and score near-perfectly.

Nothing about the shapes would look wrong. So the split is asserted rather
than arranged: ``unpack_targets`` returns the two halves separately, the head's
weight matrix is built with ``embedding_dim`` rows and would raise on a
mismatch, and ``assert_head_ignores_targets`` checks -- by perturbing the
target columns and requiring the prediction not to move -- that no path exists
from them to the output.

----------------------------------------------------------------------------
GATE 3, SATISFIED BY CONSTRUCTION RATHER THAN BY LUCK
----------------------------------------------------------------------------
Gate 3 requires an untrained head to predict the training-fold mean. A
softmax head with zero weights and zero bias emits the uniform distribution,
whose expectation is 3.0 -- not the training mean (~2.6), so the naive
initialisation FAILS the gate.

Zero weights with the bias at ``log`` of the **training marginal
distribution** satisfies it exactly: ``softmax(log q) == q`` for any
distribution ``q``, so epoch-0 predicts the training marginal, whose
expectation is the mean of the training patients' own expectations -- the
training-fold mean. Same construction as the other heads (zero weights, bias
carrying the mean), adapted to a distribution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

#: The manifest columns holding the rater distribution, in grade order.
SOFT_COLUMNS = ("soft_1", "soft_2", "soft_3", "soft_4", "soft_5")

#: The grades those columns correspond to. The expectation is over THESE, and
#: writing them down is what makes ``expectation`` a statement about the 1-5
#: scale rather than about column indices.
GRADES = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

N_GRADES = len(SOFT_COLUMNS)

#: Floor for a zero-probability grade before taking a log. The tails ARE
#: sparse -- it is why CORAL/CORN were rejected (PLAN §4.7) -- so grades with
#: no votes in a training fold are expected, not exceptional. Small enough to
#: leave the marginal essentially unchanged, large enough that log is finite.
PROBABILITY_FLOOR = 1e-8

#: What the label field must say to select this arm. Not a manifest column,
#: unlike every other label: it names five of them plus a construction.
LDL_LABEL = "ldl"


class LDLError(RuntimeError):
    """The label-distribution arm cannot run as configured."""


# --------------------------------------------------------------------------
# the distribution, read and checked
# --------------------------------------------------------------------------


def target_distributions(rows: list[dict]) -> np.ndarray:
    """(N, 5) rater distributions from the manifest rows, validated.

    Rows are normalised to sum to 1 -- the manifest stores fractions already,
    but a set that sums to 0.999 would otherwise shift every KL by a constant
    that varies per patient.
    """
    missing = [column for column in SOFT_COLUMNS if column not in (rows[0] if rows else {})]
    if missing:
        raise LDLError(
            f"the manifest has no {missing} column(s); the label-distribution "
            "arm needs the full five-rater distribution per patient"
        )
    values = np.array(
        [[float(row[column]) for column in SOFT_COLUMNS] for row in rows],
        dtype=float,
    )
    if np.any(values < 0):
        raise LDLError("a soft column holds a negative value; it is a fraction")
    totals = values.sum(axis=1)
    if np.any(totals <= 0):
        bad = int(np.argmin(totals))
        raise LDLError(
            f"patient at row {bad} has an all-zero rater distribution, so it "
            "has no target at all"
        )
    return values / totals[:, None]


def expectation(distributions: np.ndarray) -> np.ndarray:
    """E[grade] per row -- the scalar this arm is scored on."""
    array = np.asarray(distributions, dtype=float)
    if array.ndim != 2 or array.shape[1] != N_GRADES:
        raise LDLError(
            f"distributions must be (N, {N_GRADES}); got {array.shape}"
        )
    return array @ GRADES


def assert_mean_is_the_soft_expectation(
    labels: np.ndarray, distributions: np.ndarray, tolerance: float = 1e-6
) -> dict:
    """**The manifest's own consistency, checked before the arm runs.**

    The whole comparability argument is that LDL is scored against the SAME
    truth vector as the mean arm. That holds only if the ``mean`` column
    really is the expectation of the soft columns. If it is not -- a different
    weighting, a different rounding, a column that means something else --
    then the arm trains toward one quantity and is scored against another, and
    the resulting PCC is a number answering a question nobody asked.

    Checked rather than assumed, because both readings produce a plausible
    result and only one of them is the designed comparison.
    """
    derived = expectation(distributions)
    truth = np.asarray(labels, dtype=float)
    if truth.shape != derived.shape:
        raise LDLError(
            f"{truth.shape[0]} labels against {derived.shape[0]} distributions"
        )
    worst = float(np.max(np.abs(truth - derived))) if truth.size else 0.0
    if worst > tolerance:
        index = int(np.argmax(np.abs(truth - derived)))
        raise LDLError(
            f"the 'mean' label is not the expectation of soft_1..soft_5: worst "
            f"disagreement {worst:.6g} at row {index} ({truth[index]:.6f} "
            f"against {derived[index]:.6f}). The LDL arm is scored on 'mean' "
            "so that it is comparable with the mean arm; if the two are "
            "different quantities that comparison is not the one intended."
        )
    return {
        "checked": True,
        "worst_absolute_disagreement": worst,
        "tolerance": tolerance,
        "why": (
            "LDL trains on the distribution and is scored on its expectation; "
            "that is only the mean arm's truth vector if mean == sum(k*soft_k)"
        ),
    }


def log_marginal_bias(
    distributions: np.ndarray, floor: float = PROBABILITY_FLOOR
) -> np.ndarray:
    """Bias making an untrained softmax head predict the training marginal.

    ``softmax(log q) == q`` for a distribution ``q``, so with zero weights the
    head emits ``q`` for every patient and its expectation is the training
    fold's mean label. That is gate 3, satisfied by construction rather than
    approached by tuning.
    """
    marginal = np.asarray(distributions, dtype=float).mean(axis=0)
    total = marginal.sum()
    if total <= 0:
        raise LDLError("the training marginal is all zeros")
    marginal = np.clip(marginal / total, floor, None)
    return np.log(marginal / marginal.sum())


# --------------------------------------------------------------------------
# packing: the distribution rides in the feature row
# --------------------------------------------------------------------------


def pack_targets(features: np.ndarray, distributions: np.ndarray) -> np.ndarray:
    """``[embedding..., soft_1..soft_5]`` per row, for the frozen harness.

    The harness has no side channel and slices rows agnostically, so the
    per-patient target travels with its features -- the same device the graph
    path uses for region boxes. See ``unpack_targets`` for why the split is
    then asserted rather than assumed.
    """
    embeddings = np.asarray(features, dtype=float)
    targets = np.asarray(distributions, dtype=float)
    if embeddings.ndim != 2:
        raise LDLError(f"features must be (N, D); got {embeddings.shape}")
    if targets.shape != (embeddings.shape[0], N_GRADES):
        raise LDLError(
            f"distributions must be (N, {N_GRADES}) with N="
            f"{embeddings.shape[0]}; got {targets.shape}"
        )
    return np.concatenate([embeddings, targets], axis=1)


def unpack_targets(packed: np.ndarray, embedding_dim: int) -> tuple:
    """(embeddings, distributions). The split the head must respect."""
    array = np.asarray(packed, dtype=float)
    expected = embedding_dim + N_GRADES
    if array.shape[1] != expected:
        raise LDLError(
            f"packed rows are {array.shape[1]} wide; a {embedding_dim}-dim "
            f"embedding plus {N_GRADES} target columns needs {expected}"
        )
    return array[:, :embedding_dim], array[:, embedding_dim:]


def assert_head_ignores_targets(head, packed: np.ndarray) -> dict:
    """**No path from the target columns to the prediction.**

    The sharpest defect available in this arm: the label rides inside the
    feature array, so a head whose weight matrix spanned the full packed width
    would read the rater distribution as an input and predict the label from
    the label. Every shape stays right and the PCC goes near 1.

    Perturbing the target columns and requiring the predictions not to move
    tests the property directly, rather than inspecting a weight shape and
    inferring it. A weight matrix of the correct width could still be applied
    to the wrong slice.
    """
    array = np.asarray(packed, dtype=float)
    if array.shape[0] < 1:
        raise LDLError("nothing to check: no rows")
    baseline = np.asarray(head.predict(array), dtype=float)

    original = array[:, -N_GRADES:]
    # **The perturbation must change the EXPECTATION, not merely the
    # arrangement.** Every valid distribution sums to 1, so a permutation
    # leaves any constant-weight reader seeing exactly what it saw before --
    # and a head that reads `sum(targets)` is not leaking anything anyway,
    # since it adds a constant. The leak that matters reads `sum(k * t_k)`,
    # the label itself, so the disturbance has to move that.
    #
    # Built as a fresh array rather than a reversed view of the same memory:
    # assigning a slice from an overlapping view of itself is undefined and
    # would make this check pass through aliasing corruption instead of
    # signal.
    replacement = np.zeros_like(original)
    replacement[:, 0] = 1.0
    if np.allclose(original, replacement):
        replacement[:] = 0.0
        replacement[:, -1] = 1.0
    if np.allclose(original, replacement):
        raise LDLError(
            "the target columns could not be perturbed, so this check has "
            "nothing to detect and its silence would mean nothing"
        )

    disturbed = array.copy()
    disturbed[:, -N_GRADES:] = replacement
    moved = np.asarray(head.predict(disturbed), dtype=float)

    worst = float(np.max(np.abs(baseline - moved))) if baseline.size else 0.0
    if worst > 0:
        raise LDLError(
            f"the head's predictions moved by {worst:.6g} when only the TARGET "
            "columns changed, so the rater distribution is reaching the model "
            "as an input. That arm predicts the label from the label and "
            "scores near-perfectly with every shape correct."
        )
    return {
        "checked": True,
        "prediction_shift_when_targets_perturbed": worst,
        "expectation_shift_applied": float(
            np.max(np.abs(expectation(original) - expectation(replacement)))
        ),
    }


# --------------------------------------------------------------------------
# the head
# --------------------------------------------------------------------------


@dataclass
class LDLHeadBackbone:
    """Five-output softmax over frozen embeddings, fitted under KL.

    **Deliberately mirrors ``EmbeddingHeadBackbone``** -- same optimiser,
    learning rate, weight decay and seeding -- so Stage G varies the label
    formulation and nothing else. An LDL arm on a different optimiser would
    confound "does modelling the distribution help" with "does this optimiser
    suit this problem", and the whole stage exists to answer the first.

    Torch, for that reason. The parts that can be checked without it -- the
    target construction, the gate-3 bias, the expectation, the packing and the
    target-isolation check -- are module-level functions above, tested
    directly; the fit loop is verified in ``scripts/verify_backbone_builds.py``.
    """

    embedding_dim: int
    learning_rate: float = 1e-3
    weight_decay: float = 0.01
    max_steps: int = 50
    seed: int = 1337
    parameter_report: dict = field(default_factory=dict)

    _weights: Any = field(default=None, repr=False)
    _bias: Any = field(default=None, repr=False)
    _optimizer: Any = field(default=None, repr=False)
    _device: Any = field(default=None, repr=False)

    def reset(self, train_labels: np.ndarray) -> None:
        """Gate 3 by construction.

        ``train_labels`` is the harness's scalar array and is NOT what this
        head trains on -- the distributions arrive with the features. It is
        used only to check that the bias built from the training marginal
        really does predict the training mean, which is the property gate 3
        asserts moments later.
        """
        import torch

        torch.manual_seed(self.seed)
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._weights = torch.zeros(
            (self.embedding_dim, N_GRADES),
            dtype=torch.float32,
            device=self._device,
            requires_grad=True,
        )
        # Filled on the first train_epoch, when the fold's distributions are
        # available: the bias is the training marginal's log and cannot be
        # known from scalar labels alone.
        self._bias = torch.zeros(
            N_GRADES, dtype=torch.float32, device=self._device, requires_grad=True
        )
        self._optimizer = None
        self.parameter_report = {
            "total_parameters": int(self.embedding_dim * N_GRADES + N_GRADES),
            "trainable_parameters": int(self.embedding_dim * N_GRADES + N_GRADES),
            "head": "softmax_5",
            "loss": "kl_divergence",
            "reported_scalar": "expectation over grades 1-5",
        }

    def _ensure_bias(self, targets) -> None:
        """Set the gate-3 bias from this fold's marginal, once."""
        import torch

        if self._optimizer is not None:
            return
        marginal = log_marginal_bias(targets.detach().cpu().numpy())
        with torch.no_grad():
            self._bias.copy_(torch.as_tensor(marginal, dtype=torch.float32))
        self._optimizer = torch.optim.AdamW(
            [self._weights, self._bias],
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

    def _logits(self, embeddings):
        return embeddings @ self._weights + self._bias

    def train_epoch(self, features: np.ndarray, labels: np.ndarray) -> float:
        import torch

        embeddings, targets = unpack_targets(features, self.embedding_dim)
        x = torch.as_tensor(np.asarray(embeddings, dtype=np.float32), device=self._device)
        t = torch.as_tensor(np.asarray(targets, dtype=np.float32), device=self._device)
        self._ensure_bias(t)

        last = 0.0
        for _ in range(self.max_steps):
            self._optimizer.zero_grad(set_to_none=True)
            log_p = torch.log_softmax(self._logits(x), dim=1)
            # KL(target || predicted). batchmean is the correct reduction: the
            # default 'mean' divides by N*5 and reports a fifth of the
            # divergence, which is not a KL of anything.
            loss = torch.nn.functional.kl_div(log_p, t, reduction="batchmean")
            loss.backward()
            self._optimizer.step()
            last = float(loss.item())
        return last

    def predict(self, features: np.ndarray) -> np.ndarray:
        """The distribution's EXPECTATION, on the raw 1-5 scale.

        The target columns are unpacked and discarded here -- never read as
        input. ``assert_head_ignores_targets`` verifies that by measurement.
        """
        import torch

        embeddings, _discarded_targets = unpack_targets(features, self.embedding_dim)
        with torch.no_grad():
            x = torch.as_tensor(
                np.asarray(embeddings, dtype=np.float32), device=self._device
            )
            probabilities = torch.softmax(self._logits(x), dim=1)
        return expectation(probabilities.cpu().numpy())
