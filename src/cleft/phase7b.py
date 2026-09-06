"""Phase 7B: the exhaustion argument, PRE-REGISTERED AS DATA.

**This module is the pre-registration.** Not a description of one -- the search
space enumerates here, the budget is a constant here, and the selection rule is
a function here, so the suite can check that what ran is what was declared. A
pre-registration in a markdown file drifts silently; one the tests read cannot.

----------------------------------------------------------------------------
WHY THE PROTOCOL, NOT THE NUMBER
----------------------------------------------------------------------------
Phase 7B asks **has this data been exhausted?** and two things wait on a
credible answer: the un-masked full-face images, which the surgical team will
release once the crop is shown to be used up, and a second facial-beauty
dataset, worth acquiring only after the first is fully used. **So a null here
is a result with consequences.** The phase succeeds if the search is thorough,
pre-registered and honestly reported, whatever number comes out.

**Every threshold in this project assumes each arm was measured once.**
Searching over configurations on the 237 and reporting the best biases the
result upward by roughly the spread of what was tried, and invalidates the seed
bands, the BCa criterion and the claimable-delta rule together. The protection
is that **selection reads inner-validation only** -- carved from the training
folds, disjoint from test by gate 6 -- and ``select`` refuses a trial that
carries a test-fold metric at all rather than trusting the caller.

----------------------------------------------------------------------------
THE MONITOR DIVERGES FROM THE LADDER, DELIBERATELY AND ON THE RECORD
----------------------------------------------------------------------------
[DECIDED 2026-08-02] Phase 7B runs **monitor and select on ``inner_val_pcc``**.
Every ladder arm early-stops on ``inner_val_mse``.

Two quantities on one split is exactly the confusion the gate-4 amendment
found -- there, patience fired on noise in a flat region and run length tracked
luck rather than scheme, and the replay showed that switching the monitored
quantity only moves where the noise bites. PCC is the primary metric, so a
phase whose whole output is a PCC comparison monitors PCC.

**The cost is declared because it is real: a tuned arm monitored differently
from the arm it is compared against carries an undeclared factor**, and the
paired BCa against the 0.2520 baseline therefore varies monitor as well as
configuration. It is named in ``BASELINE['also_varies']`` for the same reason
``ladder.LICENCES`` exists -- an extra factor is a recorded design decision or
it is a defect.
"""

from __future__ import annotations

from .train.phase3 import claimable_delta

#: The arm being tuned, and the only one (brief §3). Tuning several and
#: reporting the best reintroduces the selection problem this phase exists to
#: avoid.
BASELINE = {
    "arm": "p7_d1_vit_b16_imagenet_g1",
    "backbone": "vit_b16",
    "init": "imagenet",
    "geometry": "g1",
    "label": "mean",
    "pcc": 0.2520,
    "sd": 0.0148,
    "seeds": 5,
    "why_this_arm": (
        "best of the 45 cells that carry numbers, cheapest to iterate on, and "
        "claimably ahead of the nearest challenger -- though only at 1.05x its "
        "threshold, which is stated because it is thin"
    ),
    #: The undeclared-factor register for the headline comparison. See the
    #: module docstring: this phase monitors inner_val_pcc and the baseline
    #: was monitored on inner_val_mse.
    "also_varies": ["monitor", "head_fit_procedure"],
    "licence": {
        "monitor": (
            "PCC is the primary metric and this phase's output is a PCC "
            "comparison, so it monitors PCC. Declared rather than silent: the "
            "paired BCa against 0.2520 varies monitor as well as configuration"
        ),
        "head_fit_procedure": (
            "every trial fits in closed form because harness.run_fold computes "
            "the held-out fold's predictions as part of its job, and using it "
            "per trial would mean computing 24 out-of-fold vectors and "
            "declining to read them. See SEARCH_HEAD_PROCEDURE -- the winning "
            "configuration must be re-run through train_cv before its number "
            "is quoted against the ladder's 0.2520"
        ),
    },
}

#: **Fixed in advance. The phase stops being a search and becomes a fish if it
#: is exceeded** (brief §8). Twenty-four, allocated to put the most trials
#: where the hypothesis is strongest.
SEARCH_BUDGET = 24

#: Selection and monitoring both. See the module docstring for the divergence.
SELECTION_METRIC = "inner_val_pcc"
MONITOR = "inner_val_pcc"
LADDER_MONITOR = "inner_val_mse"

#: How many trailing trials must pass without a claimable gain for the search
#: to be called exhausted. A third of the budget.
EXHAUSTION_WINDOW = 8


def exhaustion_threshold() -> float:
    """The gain a trial must beat to count as an improvement.

    **Derived from the baseline arm's own SD, not written down** -- PLAN
    §4.12's rule, so a re-measured SD gives a re-derived criterion instead of a
    stale constant somebody has to remember. At sd 0.0148 over 5 seeds this is
    0.0183.
    """
    return claimable_delta(BASELINE["seeds"], BASELINE["sd"])


#: ---------------------------------------------------------------------------
#: THE SEARCH SPACE. Each axis is a hypothesis, not a knob.
#: ---------------------------------------------------------------------------

#: **Head architecture (6).** [REASONED] at 152 training samples the head is
#: not obviously the bottleneck, but it is the cheapest axis and costs one run
#: each. Reachable additively: ``harness.py`` is frozen but it is the TRAINING
#: LOOP -- the model is built in ``train/phase3.py`` and ``train/ridge.py``,
#: both unfrozen, so no part of this axis needs the freeze broken.
#:
#: ``linear`` is the baseline configuration and is deliberately IN the search,
#: so "no configuration beat the untuned arm" is an observable outcome rather
#: than a separate argument.
HEAD_AXIS = (
    {"head": "linear"},
    {"head": "linear_standardised"},
    {"head": "ridge", "alpha": 0.1},
    {"head": "ridge", "alpha": 1.0},
    {"head": "ridge", "alpha": 10.0},
    {"head": "mlp", "hidden": 64, "dropout": 0.2},
)

#: **Pooling source (12).** [LITERATURE] intermediate layers often transfer
#: better than the last for tasks unlike the pretraining objective, and
#: aesthetic assessment of a surgical outcome is very unlike ImageNet
#: classification. **The axis most is expected from**, which is why it gets
#: half the budget.
#:
#: Six depths crossed with CLS against mean-pooled patch tokens. ViT-B/16 has
#: 12 blocks; block 12 with CLS is the current representation.
POOLING_BLOCKS = (2, 4, 6, 8, 10, 12)
POOLING_TOKENS = ("cls", "mean_patch")
POOLING_AXIS = tuple(
    {"block": block, "token": token}
    for block in POOLING_BLOCKS
    for token in POOLING_TOKENS
)

#: **Ensembling, ViT+Swin only (6).** Both are ``pooled`` artifacts, both at
#: ImageNet/G1, both already extracted and declared -- no new sets.
#:
#: **Averaging is unavailable and that is a measured fact, not an omission**:
#: ``models.factory.BACKBONES`` gives ViT-B/16 768 dims and Swin-B 1024, so
#: there is nothing to average without a projection, and a projection is a
#: learned layer rather than an ensemble. Concatenation is the operation
#: available, at three normalisations.
ENSEMBLE_COMBINES = ("concat", "concat_standardised", "concat_l2norm")
ENSEMBLE_HEADS = ({"head": "linear"}, {"head": "ridge", "alpha": 1.0})
ENSEMBLE_AXIS = tuple(
    {"combine": combine, "with": ("vit_b16", "swin_b"), **head}
    for combine in ENSEMBLE_COMBINES
    for head in ENSEMBLE_HEADS
)

#: **Declared out of budget, with the reason each is out.** Recorded rather
#: than omitted: a search that quietly drops an axis is indistinguishable from
#: one that never considered it, and this phase is being read as an exhaustion
#: argument.
OUT_OF_BUDGET = {
    "input_resolution_384": (
        "the only axis needing RE-EXTRACTION -- a new artifact version, a timm "
        "model that supports 384, and cluster time. [REASONED] the crop is "
        "small and detail-dense so more tokens may help, but the sources are "
        "compressed JPEG and the extra tokens may be interpolation rather than "
        "information. Sequenced after this phase, not inside it"
    ),
    "graph_inclusive_ensembling": (
        "SR-GNN and AG-Net store feature_map (n, 2048, 7, 7) where the "
        "transformers store pooled (n, d) -- embeddings.KIND_FOR_BACKBONE_KIND, "
        "and check_pairing refuses a feature map at a probe arm by design. "
        "Including them needs a pooling rule those arms never used, which "
        "makes it a NEW REPRESENTATION rather than an ensemble of measured ones"
    ),
    "augmentation": (
        "STRUCTURALLY unavailable to this arm, not merely unbudgeted. The arm "
        "reads a stored embedding artifact extracted once; photometric "
        "augmentation acts on IMAGES, so it would require the live extraction "
        "path and a different arm entirely. No horizontal flip in any case -- "
        "not label-preserving when asymmetry is the signal, and laterality is "
        "unrecorded (PLAN §4.6)"
    ),
}


class Phase7BError(RuntimeError):
    """The pre-registered protocol was not followed."""


def configurations() -> list[dict]:
    """The 24 trials, in the FIXED ORDER ties are broken by.

    Order is head, then pooling, then ensembling -- cheapest first, and the
    baseline configuration first of all, so a tie between the baseline and
    anything else resolves to the baseline.
    """
    trials: list[dict] = []
    for axis, entries in (
        ("head", HEAD_AXIS),
        ("pooling", POOLING_AXIS),
        ("ensemble", ENSEMBLE_AXIS),
    ):
        for entry in entries:
            trials.append({"index": len(trials), "axis": axis, **entry})
    if len(trials) != SEARCH_BUDGET:
        raise Phase7BError(
            f"the space enumerates {len(trials)} configurations against a "
            f"declared budget of {SEARCH_BUDGET}. The budget is the "
            "pre-registration; the space must match it exactly"
        )
    return trials


#: Keys a trial result must never carry. Selection reads inner-validation
#: ONLY, and the cheapest way to guarantee that is to refuse to look at a
#: record containing anything else rather than to promise not to read it.
FORBIDDEN_IN_SELECTION = (
    "oof_pcc", "test_pcc", "pcc", "oof", "test", "predictions",
)


def select(trials: list[dict]) -> dict:
    """The pre-registered selection rule: best mean ``inner_val_pcc``, ties to
    the earlier configuration in ``configurations()`` order.

    **Refuses a trial carrying any test-fold quantity.** Gate 6 asserts
    inner-val is disjoint from test, but that is a property of the SPLIT; this
    is a property of the SELECTION, and they are different guarantees. A rule
    that merely promised to ignore an out-of-fold score would be one edit away
    from reading it.
    """
    if not trials:
        raise Phase7BError("no trials to select from")

    for trial in trials:
        leaked = sorted(set(trial) & set(FORBIDDEN_IN_SELECTION))
        if leaked:
            raise Phase7BError(
                f"trial {trial.get('index')} carries {leaked}, which selection "
                "must not see. Report the OOF number for the SELECTED "
                "configuration; never rank on it"
            )
        if SELECTION_METRIC not in trial:
            raise Phase7BError(
                f"trial {trial.get('index')} has no {SELECTION_METRIC}"
            )

    return min(
        trials, key=lambda t: (-float(t[SELECTION_METRIC]), int(t["index"]))
    )


def is_exhausted(trials: list[dict]) -> dict:
    """**What "exhausted" means, stated numerically before the search starts.**

    No configuration in the final ``EXHAUSTION_WINDOW`` trials improved the
    running best ``inner_val_pcc`` by more than ``exhaustion_threshold()``.

    Returns the verdict WITH its inputs, because "exhausted" reported without
    the window, the threshold and the best gain inside it is a claim nobody can
    check -- and this phase's output is being used to ask for a dataset.
    """
    if len(trials) < EXHAUSTION_WINDOW:
        raise Phase7BError(
            f"{len(trials)} trials against a window of {EXHAUSTION_WINDOW}; "
            "the criterion cannot be evaluated yet"
        )

    ordered = sorted(trials, key=lambda t: int(t["index"]))
    threshold = exhaustion_threshold()
    split = len(ordered) - EXHAUSTION_WINDOW
    best_before = max(float(t[SELECTION_METRIC]) for t in ordered[:split])

    gains = [
        float(t[SELECTION_METRIC]) - best_before for t in ordered[split:]
    ]
    best_gain = max(gains)
    return {
        "exhausted": bool(best_gain <= threshold),
        "window": EXHAUSTION_WINDOW,
        "threshold": round(threshold, 6),
        "best_before_window": round(best_before, 6),
        "best_gain_in_window": round(best_gain, 6),
        "n_trials": len(ordered),
        "reading": (
            "no trial in the final window improved the running best by more "
            "than the baseline arm's own claimable delta"
            if best_gain <= threshold
            else "the search was still improving when the budget ran out; "
            "report that, and do not extend the budget to reach a verdict"
        ),
    }


#: ---------------------------------------------------------------------------
#: WHAT EACH CONFIGURATION CONSUMES
#: ---------------------------------------------------------------------------

#: The tuned arm's own set, and the ensemble partner. Both extracted and
#: declared (``ladder.DECLARED_EMBEDDING_HASHES``).
BASE_SET = "vit_b16__imagenet__g1"
ENSEMBLE_PARTNER_SET = "swin_b__imagenet__g1"


#: The artifact version the pooling sets are extracted into. Their own, for
#: PLAN §2.6's reason: a published artifact cannot gain a set, and
#: ``embeddings_g1_ladder_v1`` already exists holding the D1 batch.
POOLING_EMBEDDINGS_VERSION = "embeddings_p7b_blocks_v1"


def pooling_set_name(block: int, token: str) -> str:
    """The directory name of one pooling set.

    **Delegates to ``embedding_plan.set_name``** rather than formatting the
    string here. The naming rule had four implementations once already
    (``ladder.embedding_set_name``'s docstring records the collapse); a fifth
    would be the version that eventually disagrees, and this one decides which
    artifact a trial reads.
    """
    from .embedding_plan import set_name

    return set_name({
        "backbone": "vit_b16", "init": "imagenet", "geometry": "g1",
        "pretrain_scheme": None, "block": block, "token": token,
    })


def required_sets(trial: dict) -> tuple[str, ...]:
    """Every embedding set one configuration reads."""
    if trial["axis"] == "pooling":
        return (pooling_set_name(trial["block"], trial["token"]),)
    if trial["axis"] == "ensemble":
        return tuple(
            BASE_SET if backbone == "vit_b16" else ENSEMBLE_PARTNER_SET
            for backbone in trial["with"]
        )
    return (BASE_SET,)


def all_required_sets() -> tuple[str, ...]:
    """The union over the whole budget, in a stable order."""
    seen: dict[str, None] = {}
    for trial in configurations():
        for name in required_sets(trial):
            seen[name] = None
    return tuple(seen)


def verify_available(declared: set) -> None:
    """**Refuse the search unless every configuration can run.**

    Guard-3 shaped, and for the same reason: a search that starts and dies at
    trial 13 has spent cluster time and produced a partial record whose
    exhaustion window cannot be evaluated. The budget is pre-registered as
    twenty-four; twenty-three is not a smaller search, it is a different one.
    """
    missing = [name for name in all_required_sets() if name not in declared]
    if missing:
        raise Phase7BError(
            f"{len(missing)} of {len(all_required_sets())} embedding sets are "
            f"not declared: {missing[:4]}{'...' if len(missing) > 4 else ''}.\n"
            "  The pooling axis needs intermediate-block sets that no "
            "extraction produces yet -- extract.extract_features has no block "
            "or token axis. Build that extraction before running the search; "
            "running a subset breaks both the budget and the exhaustion window."
        )


#: ---------------------------------------------------------------------------
#: THE TRIAL LOOP -- inner-validation only, structurally
#: ---------------------------------------------------------------------------

#: Everything a trial record may carry. **A whitelist rather than a
#: blacklist**: ``select`` refuses forbidden keys at the consuming end, and
#: this refuses them at the producing end, so an out-of-fold number is never
#: written down rather than written down and then not read. Those are
#: different guarantees and the second is much weaker.
TRIAL_RECORD_KEYS = ("index", "axis", "config", "inner_val_pcc", "n_folds")


def make_trial_record(trial: dict, inner_val_pcc: float, n_folds: int) -> dict:
    """One trial's record, carrying inner-val and nothing else."""
    record = {
        "index": int(trial["index"]),
        "axis": str(trial["axis"]),
        "config": {k: v for k, v in trial.items() if k not in ("index", "axis")},
        "inner_val_pcc": float(inner_val_pcc),
        "n_folds": int(n_folds),
    }
    stray = sorted(set(record) - set(TRIAL_RECORD_KEYS))
    if stray:
        raise Phase7BError(f"trial record carries {stray}, which is not declared")
    return record


def fit_feature_transform(kind: str, train_features):
    """A transform FITTED ON TRAINING ROWS ONLY, returned as a callable.

    **Standardisation fitted over all 237 would put the test fold's column
    means and variances into the representation used to predict it** -- a leak
    with no symptom: the shapes are right, the fit succeeds, the number is
    merely better than it should be. It is the same failure
    ``embeddings.assert_no_test_fold_leak`` exists for on the AdaBN path, in a
    place where nothing would check it.

    ``l2norm`` is per-row and fits nothing; ``identity`` and ``concat`` fit
    nothing either. Only standardisation has parameters, and they come from
    the training rows of the fold being scored.
    """
    import numpy as np

    if kind in ("identity", "concat", None):
        return lambda features: features
    if kind == "l2norm":
        def apply(features):
            norms = np.linalg.norm(features, axis=1, keepdims=True)
            return features / np.maximum(norms, 1e-12)
        return apply
    if kind in ("standardise", "concat_standardised", "linear_standardised"):
        mean = train_features.mean(axis=0, keepdims=True)
        std = train_features.std(axis=0, keepdims=True)
        std = np.maximum(std, 1e-12)
        return lambda features: (features - mean) / std
    raise Phase7BError(f"unknown feature transform {kind!r}")


def transform_for(trial: dict) -> str:
    """Which transform a configuration implies."""
    if trial["axis"] == "ensemble":
        return {
            "concat": "identity",
            "concat_standardised": "standardise",
            "concat_l2norm": "l2norm",
        }[trial["combine"]]
    if trial.get("head") == "linear_standardised":
        return "standardise"
    return "identity"


def assemble_features(trial: dict, features_by_set: dict):
    """The feature matrix one configuration sees, before the fold transform."""
    import numpy as np

    names = required_sets(trial)
    missing = [name for name in names if name not in features_by_set]
    if missing:
        raise Phase7BError(f"trial {trial['index']} needs {missing}")
    arrays = [np.asarray(features_by_set[name]) for name in names]
    for array in arrays:
        if array.ndim != 2:
            raise Phase7BError(
                f"trial {trial['index']} got a {array.ndim}-d array; the "
                "search consumes pooled sets only"
            )
    return arrays[0] if len(arrays) == 1 else np.concatenate(arrays, axis=1)


def evaluate_configuration(
    trial: dict,
    *,
    features_by_set: dict,
    labels,
    patient_ids: list,
    assignments: dict,
    make_head,
    inner_val_frac: float,
    seed: int,
) -> dict:
    """One configuration's mean inner-val PCC over the folds.

    **The test fold is never scored and never materialised as a row index.**
    Each fold's outer-train is split by ``harness.inner_val_split`` -- the same
    frozen function every ladder arm uses -- the head is fitted on the training
    part and scored on the inner-val part, and that is the whole of it. The
    only place a test id is computed is inside the assertion that inner-val
    does not intersect it, which is a check on the split rather than a use of
    the fold.
    """
    import numpy as np

    from .eval.metrics import pcc
    from .train.harness import inner_val_split

    features = assemble_features(trial, features_by_set)
    index_of = {pid: i for i, pid in enumerate(patient_ids)}
    labels = np.asarray(labels)

    scores = []
    for fold in sorted(set(assignments.values())):
        outer_train = [pid for pid in patient_ids if assignments[pid] != fold]
        train_ids, inner_ids = inner_val_split(
            outer_train, inner_val_frac, seed, fold
        )

        # The ONLY use of the fold's own patients: asserting they are absent
        # from what gets scored. Computed here and discarded.
        held_out = {pid for pid in patient_ids if assignments[pid] == fold}
        if held_out & set(inner_ids):
            raise Phase7BError(
                f"fold {fold}: inner-val intersects the held-out fold. "
                "Selection would be reading test patients"
            )

        train_rows = np.array([index_of[pid] for pid in train_ids], dtype=int)
        inner_rows = np.array([index_of[pid] for pid in inner_ids], dtype=int)

        transform = fit_feature_transform(
            transform_for(trial), features[train_rows]
        )
        train_features = transform(features[train_rows])
        inner_features = transform(features[inner_rows])

        head = make_head(trial, seed=seed, fold=fold)
        head.fit(train_features, labels[train_rows])
        scores.append(
            float(pcc(labels[inner_rows], head.predict(inner_features)))
        )

    return {"inner_val_pcc": float(np.mean(scores)), "n_folds": len(scores)}


#: **[DECIDED 2026-08-02] Every trial fits its head in CLOSED FORM, and the
#: ladder arm does not.**
#:
#: The ladder's 769-parameter head is trained by SGD with early stopping
#: through ``harness.run_fold`` -- which computes the held-out fold's
#: predictions as part of its job. Using it per trial would mean computing 24
#: out-of-fold vectors and declining to look at them, and "computed and
#: refused" is the weaker guarantee this protocol exists to avoid.
#:
#: So the search fits ridge-regularised least squares: deterministic, no early
#: stopping, no test-fold prediction anywhere in the path. **Trial 0 is the
#: closed-form counterpart of the arm's head, not the arm's head**, and that
#: is a real difference registered in ``BASELINE['also_varies']`` beside the
#: monitor.
#:
#: **What it costs, stated rather than buried:** the winner's OOF number
#: carries a head-fitting procedure the 0.2520 baseline did not use, so the
#: paired BCa varies configuration AND procedure AND monitor. What the search
#: cleanly answers is "does any configuration in the pre-registered space beat
#: the others on inner-val", which is the exhaustion question. Whether the
#: winner beats 0.2520 *as the ladder would have run it* needs the winning
#: configuration re-run through ``train_cv``, and that is a follow-up this
#: phase should declare rather than assume.
SEARCH_HEAD_PROCEDURE = {
    "fit": "ridge-regularised least squares, closed form",
    "ladder_arm_fit": "SGD with early stopping through harness.run_fold",
    "why": (
        "harness.run_fold computes the held-out fold's predictions as part of "
        "its job, so using it per trial would mean computing 24 out-of-fold "
        "vectors and declining to read them"
    ),
    "cost": (
        "the winner's OOF carries a procedure the 0.2520 baseline did not "
        "use. Re-run the winning configuration through train_cv before "
        "quoting it against the ladder"
    ),
    "default_alpha": 1e-6,
}


class _LeastSquaresHead:
    """Closed-form ridge. Deterministic, and it never sees a test fold."""

    def __init__(self, alpha: float, hidden: int | None = None, dropout: float = 0.0):
        self.alpha = float(alpha)
        self.hidden = hidden
        self.dropout = dropout
        self.weights = None
        self.projection = None

    def fit(self, features, labels):
        import numpy as np

        design = np.asarray(features, dtype=np.float64)
        if self.hidden:
            # A fixed random projection with a ReLU, fitted only in its output
            # layer. Not a trained MLP -- a trained one needs SGD, which is
            # the procedure this path exists to avoid. Declared as what it is.
            rng = np.random.default_rng(0)
            if self.projection is None:
                self.projection = rng.normal(
                    scale=1.0 / np.sqrt(design.shape[1]),
                    size=(design.shape[1], int(self.hidden)),
                )
            design = np.maximum(design @ self.projection, 0.0)
        design = np.concatenate([design, np.ones((len(design), 1))], axis=1)
        gram = design.T @ design + self.alpha * np.eye(design.shape[1])
        self.weights = np.linalg.solve(gram, design.T @ np.asarray(labels, float))
        return self

    def predict(self, features):
        import numpy as np

        design = np.asarray(features, dtype=np.float64)
        if self.hidden:
            design = np.maximum(design @ self.projection, 0.0)
        design = np.concatenate([design, np.ones((len(design), 1))], axis=1)
        return design @ self.weights


def head_for(trial: dict, *, seed: int, fold: int):
    """The head one configuration fits. See ``SEARCH_HEAD_PROCEDURE``."""
    head = trial.get("head", "linear")
    if head == "ridge":
        return _LeastSquaresHead(alpha=float(trial["alpha"]))
    if head == "mlp":
        return _LeastSquaresHead(
            alpha=SEARCH_HEAD_PROCEDURE["default_alpha"],
            hidden=int(trial["hidden"]),
            dropout=float(trial.get("dropout", 0.0)),
        )
    if head in ("linear", "linear_standardised"):
        return _LeastSquaresHead(alpha=SEARCH_HEAD_PROCEDURE["default_alpha"])
    raise Phase7BError(f"unknown head {head!r}")


def evaluate_selected_oof(
    selected: dict,
    *,
    features_by_set: dict,
    labels,
    patient_ids: list,
    assignments: dict,
    make_head,
    inner_val_frac: float,
    seeds: list,
) -> dict:
    """**The only out-of-fold computation in the whole phase**, for the one
    configuration selection already chose.

    Runs the winner at every seed, pooling each seed's held-out predictions
    into one 237-vector so the arm reports its OWN band (PLAN §4.12.1:
    inherited, never) and so a paired BCa has something to pair.
    """
    import numpy as np

    from .eval.metrics import pcc
    from .train.harness import inner_val_split

    trial = {"index": selected["index"], "axis": selected["axis"],
             **selected["config"]}
    features = assemble_features(trial, features_by_set)
    index_of = {pid: i for i, pid in enumerate(patient_ids)}
    labels = np.asarray(labels)

    pooled_pccs, per_seed = [], []
    #: **Every seed's vector is kept, not the first one.** An earlier version
    #: stored whichever seed ran first and called it "the OOF" -- which would
    #: have paired one arbitrary run of the tuned arm against the baseline and
    #: reported it as the arm. The comparison is per seed (``COMPARISON_RULE``)
    #: and needs all five on both sides.
    oof_by_seed: dict[int, list] = {}
    oof_ids = oof_truth = None
    for seed in seeds:
        predictions = np.full(len(patient_ids), np.nan)
        for fold in sorted(set(assignments.values())):
            outer_train = [pid for pid in patient_ids if assignments[pid] != fold]
            test_ids = [pid for pid in patient_ids if assignments[pid] == fold]
            train_ids, _ = inner_val_split(
                outer_train, inner_val_frac, seed, fold
            )
            train_rows = np.array([index_of[p] for p in train_ids], dtype=int)
            test_rows = np.array([index_of[p] for p in test_ids], dtype=int)

            transform = fit_feature_transform(
                transform_for(trial), features[train_rows]
            )
            head = make_head(trial, seed=seed, fold=fold)
            head.fit(transform(features[train_rows]), labels[train_rows])
            predictions[test_rows] = head.predict(transform(features[test_rows]))

        if np.isnan(predictions).any():
            raise Phase7BError("some patients received no out-of-fold prediction")
        pooled_pccs.append(float(pcc(labels, predictions)))
        per_seed.append({"seed": int(seed), "pooled_oof_pcc": pooled_pccs[-1]})
        oof_by_seed[int(seed)] = predictions.tolist()
        if oof_ids is None:
            oof_ids = list(patient_ids)
            oof_truth = labels.tolist()

    return {
        "seeds": per_seed,
        "pooled_pccs": pooled_pccs,
        "oof_ids": oof_ids,
        "oof_truth": oof_truth,
        "oof_by_seed": oof_by_seed,
        "head_procedure": SEARCH_HEAD_PROCEDURE,
    }


#: **[DECIDED 2026-08-02] The paired BCa is PER SEED, and the seeds are never
#: averaged into one vector.**
#:
#: The 0.2520 baseline is not one out-of-fold vector. Its run directory holds
#: **five**, one per seed, and the winner's side has the same shape -- so a
#: BCa comparing one averaged vector against another would compare two
#: artefacts rather than two arms.
#:
#: **Averaging is worse than merely inelegant: it is an ensemble.** Mean-
#: pooling five prediction vectors cancels independent error, so
#: ``PCC(truth, mean of five)`` is systematically HIGHER than the mean of the
#: five PCCs. The averaged baseline would therefore not score 0.2520 -- it
#: would silently replace the bar this project quotes everywhere with a larger
#: number nobody measured, and understate the tuned arm against a baseline
#: that never ran.
#:
#: **The pairing is real, not merely parallel.** Both arms draw
#: ``harness.inner_val_split(outer_train, frac, seed, fold)``, so at a given
#: seed they hold out the same inner-val patients and train on the same rows.
#: Same data conditions, different configuration, which is what a paired
#: comparison isolates. Pairing by seed number is therefore meaningful rather
#: than an arbitrary alignment of two lists.
#:
#: **The claim needs both conditions, as everywhere else** (PLAN §4.3): every
#: seed's interval excluding zero in the same direction, AND the mean delta
#: exceeding the two arms' combined seed uncertainty. Reporting "k of 5" when
#: k < 5 is the honest outcome and is what "unresolved" looks like here.
COMPARISON_RULE = {
    "shape": "per seed, paired over patients; five intervals, not one",
    "never": "averaging the seeds into one vector before the BCa",
    "why_not": (
        "averaging prediction vectors is an ENSEMBLE -- it cancels independent "
        "error, so the averaged vector's PCC exceeds the mean of the five. The "
        "averaged baseline would not score 0.2520, so the bar would move"
    ),
    "pairing": (
        "both arms split with inner_val_split(outer_train, frac, seed, fold), "
        "so one seed means the same held-out patients on both sides"
    ),
    "claimable_requires": [
        "every seed's BCa interval excludes zero, all in the same direction",
        "the mean delta exceeds combined_claimable_delta of the two arms",
    ],
}


#: **The baseline's five vectors are declared as FIVE FILE INPUTS**, named
#: ``baseline_oof_seed_<seed>``, mirroring the ``set_`` convention the
#: embedding sets already use.
#:
#: **[DECIDED 2026-08-02] Not the run directory, and the reason is the
#: worktree.** A run directory contains ``code/`` -- a git worktree at the
#: run's SHA (PLAN §2.4) -- so its rollup would cover the entire source tree.
#: Large, and worse, it would CHANGE whenever anything in that worktree was
#: touched, making the declared hash unstable for reasons having nothing to do
#: with the data being consumed. Guard 3 would then fail on a run whose
#: predictions were untouched, which trains people to re-paste hashes without
#: reading them.
#:
#: Scoping the hash to a glob inside the input declaration is not available:
#: ``provenance/hashing.py`` is FROZEN, and reaching into the provenance
#: apparatus to make a Phase 7B input tidier is exactly the change PLAN R5
#: says to escalate rather than make.
#:
#: Declaring the files individually is better on the merits anyway: each
#: seed's vector is independently hashed, guard 3 names WHICH one moved if one
#: does, and the provenance is per vector -- the same shape as the comparison
#: that consumes them.
BASELINE_INPUT_PREFIX = "baseline_oof_seed_"


def baseline_paths_from_inputs(declared: dict) -> dict:
    """``{seed: path}`` for every declared baseline prediction input."""
    paths: dict[int, str] = {}
    for name, entry in declared.items():
        if not name.startswith(BASELINE_INPUT_PREFIX):
            continue
        suffix = name[len(BASELINE_INPUT_PREFIX):]
        if not suffix.isdigit():
            raise Phase7BError(
                f"input {name!r} does not name a seed; the convention is "
                f"{BASELINE_INPUT_PREFIX}<seed>"
            )
        paths[int(suffix)] = entry["path"] if isinstance(entry, dict) else entry
    return paths


def load_baseline_predictions(paths_by_seed: dict, seeds: list, patient_ids: list) -> dict:
    """The baseline arm's out-of-fold vector for each seed.

    Row order is realigned by patient id rather than assumed -- the CSVs are
    written per seed and nothing guarantees they enumerate in the manifest's
    order, and a silent misalignment pairs shuffled vectors and still produces
    a plausible interval (``embeddings.assert_row_order``'s lesson, in the one
    place left that reads a CSV).
    """
    from pathlib import Path

    from .cluster_csv import PREDICTIONS_COLUMNS, read_cluster_csv

    wanted = [int(s) for s in seeds]
    absent = [seed for seed in wanted if seed not in paths_by_seed]
    if absent:
        raise Phase7BError(
            f"no {BASELINE_INPUT_PREFIX}<seed> input for seeds {absent}; "
            f"declared: {sorted(paths_by_seed)}. The comparison is per seed "
            "and needs the winner's seeds on both sides"
        )

    by_seed: dict[int, list] = {}
    for seed in wanted:
        path = Path(paths_by_seed[seed])
        if not path.is_file():
            raise Phase7BError(f"baseline predictions missing at {path}")
        # **Tier-marked CSV**: line 1 is `# CLUSTER-ONLY: patient-keyed`, and a
        # bare DictReader takes it as the header. Third time that marker has
        # cost a round -- cleft.cluster_csv is where the skip lives now.
        table = {
            int(row["patient_id"]): float(row["prediction"])
            for row in read_cluster_csv(path, expect=PREDICTIONS_COLUMNS)
        }
        missing = [pid for pid in patient_ids if pid not in table]
        if missing:
            raise Phase7BError(
                f"{path.name} is missing {len(missing)} patients, e.g. "
                f"{missing[:5]}; it cannot be paired with the winner"
            )
        by_seed[seed] = [table[pid] for pid in patient_ids]
    return by_seed


def paired_comparison(
    *, truth, winner_by_seed: dict, baseline_by_seed: dict,
    winner_sd: float, n_boot: int = 10000,
    statistic=None, statistic_name: str = "pcc",
) -> dict:
    """Per-seed paired BCa, and the two-condition verdict.

    Returns every seed's delta and interval, not just the summary -- the brief
    asks for the search to be reported rather than the winner, and the same
    applies to the comparison that concludes it.

    **[GENERALISED 2026-08-17]** ``statistic`` and ``statistic_name`` default
    to PCC, so every existing scope produces byte-identical output. The
    frozen ``paired_delta_bca`` was ALREADY metric-generic -- only this
    function hardcoded ``pcc`` -- so Phase 11's IEM contrast is served by
    passing the statistic rather than by a second paired implementation,
    which R10 forbids. The two-condition machinery is sign-agnostic: it
    asks whether every seed's interval excludes zero in ONE direction, and
    that is as true for an error (lower better) as for a correlation.
    **Which direction is good is the CALLER's to record**, and this
    function deliberately does not guess.
    """
    import numpy as np

    from .eval.metrics import paired_delta_bca, pcc
    from .train.phase3 import combined_claimable_delta

    statistic = pcc if statistic is None else statistic

    seeds = sorted(set(winner_by_seed) & set(baseline_by_seed))
    if seeds != sorted(winner_by_seed) or seeds != sorted(baseline_by_seed):
        raise Phase7BError(
            f"the two arms do not share their seeds: winner "
            f"{sorted(winner_by_seed)}, baseline {sorted(baseline_by_seed)}. "
            "Pairing by seed means the same held-out patients on both sides"
        )

    per_seed = []
    for seed in seeds:
        delta, lo, hi = paired_delta_bca(
            statistic, truth, winner_by_seed[seed], baseline_by_seed[seed],
            n_boot=n_boot, seed=seed,
        )
        per_seed.append({
            "seed": seed, "delta": delta, "lo": lo, "hi": hi,
            "excludes_zero": bool(lo > 0 or hi < 0),
            f"winner_{statistic_name}": float(
                statistic(truth, winner_by_seed[seed])
            ),
            f"baseline_{statistic_name}": float(
                statistic(truth, baseline_by_seed[seed])
            ),
        })

    deltas = [entry["delta"] for entry in per_seed]
    excluding = [entry for entry in per_seed if entry["excludes_zero"]]
    directions = {np.sign(entry["delta"]) for entry in excluding}
    baseline_scores = [
        entry[f"baseline_{statistic_name}"] for entry in per_seed
    ]
    baseline_sd = (
        float(np.std(baseline_scores, ddof=1)) if len(seeds) > 1 else 0.0
    )

    threshold = combined_claimable_delta(
        winner_sd, len(seeds), baseline_sd, len(seeds)
    )
    mean_delta = float(np.mean(deltas))
    all_exclude = len(excluding) == len(per_seed) and len(directions) == 1

    return {
        "rule": COMPARISON_RULE,
        # [2026-08-17] Which metric these numbers are. Every existing scope
        # reads "pcc", which is what they always were.
        "statistic": statistic_name,
        "per_seed": per_seed,
        "mean_delta": mean_delta,
        "n_excluding_zero": len(excluding),
        "n_seeds": len(seeds),
        # **[CORRECTED 2026-08-04] This was `len(directions) <= 1`, computed
        # over the EXCLUDING subset -- so when nothing excluded zero it
        # reported True on an empty set.** The ladder headline came back
        # `n_excluding_zero: 0, same_direction: true`, which reads as five
        # seeds agreeing when in fact the per-seed deltas ran -0.0027 to
        # +0.1040 and the sign flips. R7 instance 14: a field reporting
        # agreement exactly when there is no evidence for any.
        #
        # The VERDICT was never wrong -- `all_exclude` requires
        # `len(directions) == 1`, which an empty set fails. Only the reported
        # field misled. None here rather than False: "no seed excluded zero"
        # is not the same claim as "the excluding seeds disagreed".
        "same_direction": (len(directions) == 1) if excluding else None,
        # What a reader actually wants: do the five deltas point one way at
        # all, regardless of whether any interval excludes zero.
        "same_direction_all_seeds": len({np.sign(d) for d in deltas}) == 1,
        "baseline_sd_recomputed": baseline_sd,
        "threshold": threshold,
        # **[ADDED 2026-09-02] PLAN 4.3's CONDITION 1, returned rather
        # than discarded.** `all_exclude` was computed here and thrown
        # away, so every caller wanting it rebuilt the conjunction from
        # `n_excluding_zero`, `n_seeds` and `same_direction`. Two copies
        # of one rule, and a Phase 22 task that recorded neither
        # condition at all left eight verdicts carrying `None`.
        #
        # Additive: no existing value changes, and the components stay
        # for readers who want them. A test pins this field equal to
        # that derivation, so the two cannot drift.
        "all_seeds_exclude_zero_one_direction": bool(all_exclude),
        # PLAN 4.3's CONDITION 2.
        "exceeds_threshold": bool(abs(mean_delta) > threshold["arm_means_95"]),
        "claimable": bool(all_exclude and abs(mean_delta) > threshold["arm_means_95"]),
        "reading": (
            "claimable: every seed's interval excludes zero in one direction "
            "and the mean delta clears the combined uncertainty"
            if all_exclude and abs(mean_delta) > threshold["arm_means_95"]
            else f"UNRESOLVED: {len(excluding)} of {len(per_seed)} intervals "
            "exclude zero"
        ),
    }


def run_search(
    evaluate, *, completed=(), checkpoint=lambda _: None, log=lambda *_: None
) -> dict:
    """Walk the budget, record each trial, select, and evaluate exhaustion.

    ``completed`` carries trial records from an earlier attempt, so a resumed
    job skips what already ran, and ``checkpoint`` is called with the records
    so far after every trial. **PLAN §2.7's measured hazard is that run
    identity survives a pause and training does not** -- the pod reattaches the
    directory and starts again at zero, with no symptom but wall time. Twenty-
    four trials is long enough for that to cost a relaunch, and Phase 5 built
    checkpointing for exactly this.
    """
    done = {int(record["index"]): record for record in completed}
    records: list[dict] = []
    for trial in configurations():
        if trial["index"] in done:
            log(f"  trial {trial['index']:2d} [{trial['axis']}] RESUMED")
            records.append(done[trial["index"]])
            continue
        outcome = evaluate(trial)
        record = make_trial_record(
            trial, outcome["inner_val_pcc"], outcome["n_folds"]
        )
        log(
            f"  trial {record['index']:2d} [{record['axis']}] "
            f"inner_val_pcc {record['inner_val_pcc']:.4f}"
        )
        records.append(record)
        checkpoint(records)

    selected = select(records)
    return {
        "trials": records,
        "selected": selected,
        "exhaustion": is_exhausted(records),
        "budget": SEARCH_BUDGET,
        "protocol": summary(),
    }


#: **[MEASURED 2026-08-02, run p7b-search-2] The 24 trials, by axis.**
#:
#: Inner-validation only, as pre-registered. **[CORRECTED 2026-08-03] These
#: are ``p7b-search-3``, which completed cleanly.** ``p7b-search-2`` produced
#: the same trial scores and then died writing its summary
#: (``ctx.write_metrics``, a method RunContext does not have); it was re-run
#: rather than recovered, because a search reported from a crashed job is a
#: search somebody has to be told how to read. The re-run landed and this
#: record is no longer provisional.
#:
#: **POOLING -- refuted, decisively and monotonically.** The axis the brief
#: expected most from says the least::
#:
#:     block  2   -0.032 / -0.080      block  8   0.113 / 0.120
#:     block  4    0.009 /  0.029      block 10   0.186 / 0.193
#:     block  6    0.066 /  0.021      block 12   0.202 / 0.191
#:
#: Deeper is uniformly better all the way to the last block, and the best
#: pooling configuration IS trial 0 -- block 12 with the class token is the
#: baseline representation, which the extraction's bit-identity check
#: confirmed. **ImageNet's final representation is the right one for this
#: task**, which is the opposite of the [LITERATURE] prior that intermediate
#: layers transfer better for objectives unlike the pretraining one. Recorded
#: as a refuted prediction rather than an absent effect: it was specific, it
#: named the axis, and the axis answered.
#:
#: **HEAD -- null within threshold.** Trials 0-4 span 0.0139, inside the
#: 0.0183 exhaustion criterion. The MLP at 0.0437 is a capacity failure, not
#: an architecture finding: a fixed random projection through 64 units
#: discards most of a 768-dim signal at 152 training samples, and is recorded
#: as what it is (``SEARCH_HEAD_PROCEDURE`` -- it is not a trained MLP).
#:
#: **ENSEMBLE -- the only axis that moves.** Trial 22 (ViT+Swin, L2-normalised
#: concatenation, linear) at 0.2236 is **+0.0221 over trial 0**, clearing
#: 0.0183. Three of six ensemble configurations beat trial 0. The two using
#: standardised concatenation do not -- consistent with the head axis, where
#: ``linear_standardised`` also came in below plain ``linear``. **Two axes
#: independently say standardisation hurts here**, which is worth more than
#: either alone.
#:
#: **What this does NOT yet establish.** The winner's inner-val lead is not a
#: claim about out-of-fold performance: the OOF number and its paired BCa
#: against 0.2520 are computed after selection, and the comparison carries
#: the two declared divergences (``BASELINE['also_varies']``) -- so a lead
#: here still needs the winning configuration re-run through ``train_cv``
#: before it is quoted against the ladder.
#: **[WITHDRAWN 2026-08-04] "The tuned arm is claimably WORSE" is withdrawn.
#: The interval that refutes it was in this run's own metrics.json from the
#: day it was written.**
#:
#: ``comparison["paired"]``, computed by ``task_phase7b_search`` because the
#: five ``baseline_oof_seed_<seed>`` inputs were declared: ``n_excluding_zero:
#: 1``, ``n_seeds: 5``, ``claimable: false``. Per-seed deltas -0.024, -0.064,
#: -0.047, -0.045, -0.092; only seed 12345's interval excludes zero, at
#: [-0.176, -0.008].
#:
#: **This is a harder failure than Phase 7C's.** 7C never computed condition
#: 1. Here it was computed, written to the artifact, and this record was
#: authored from condition 2 while condition 1 sat in the same file saying
#: otherwise. Not uncomputed -- computed and not read. The verdict was
#: transcribed from the fields that confirmed it.
#:
#: **What survives, and it is the part that matters.** ``NOTHING BEAT 0.2520``
#: needs only that the winner is not claimably BETTER, and no interval can
#: overturn that -- the winner scores lower at every one of five seeds. The
#: exhaustion criterion, the 24-configuration budget, the inner-val selection
#: discipline and ``the_protocol_worked`` are all untouched: a +0.0221
#: inner-val lead still became a -0.0542 out-of-fold loss, and selecting on
#: inner-validation is still what made that visible.
#:
#: **What goes is the sharper sentence.** "Claimably worse" becomes "lower at
#: every seed, and this cohort cannot resolve a -0.054 difference" -- which is
#: a weaker claim about the arm and a stronger one about the cohort.
CLAIMABLY_WORSE_WITHDRAWN = {
    "withdrawn": "2026-08-04",
    "was": "the tuned arm is claimably WORSE than the untuned one",
    "now": (
        "UNRESOLVED -- lower at every seed, 1 of 5 intervals excludes zero"
    ),
    "condition_1": {
        "n_excluding_zero": 1, "n_seeds": 5, "same_direction": True,
        "claimable": False,
    },
    "the_sequence": (
        "computed and not read. task_phase7b_search wrote comparison['paired'] "
        "into metrics.json because the five baseline inputs were declared; "
        "this record was authored from condition 2 while condition 1 sat in "
        "the same file saying UNRESOLVED"
    ),
    "worse_than_7c": (
        "7C never computed condition 1. Here it existed in the artifact and "
        "the verdict was transcribed from the fields that confirmed it"
    ),
    "survives": (
        "NOTHING BEAT 0.2520 -- it needs only that the winner is not "
        "claimably BETTER, which no interval can overturn. Exhaustion, the "
        "budget, the selection discipline and the_protocol_worked are "
        "untouched"
    ),
    "for_the_write_up": (
        "a weaker claim about the arm and a stronger one about the cohort: "
        "237 patients cannot resolve a -0.054 difference"
    ),
}

SEARCH_AXIS_VERDICTS = {
    "measured": "2026-08-03",
    "run": "p7b-search-3",
    "selection_metric": SELECTION_METRIC,
    "superseded": (
        "p7b-search-2 produced the same trial scores and died writing its "
        "summary; re-run rather than recovered"
    ),
    "pooling": {
        "verdict": "REFUTED, monotonically -- deeper is uniformly better",
        "by_block": {
            2: (-0.0323, -0.0798), 4: (0.0094, 0.0292), 6: (0.0659, 0.0212),
            8: (0.1130, 0.1200), 10: (0.1858, 0.1931), 12: (0.2015, 0.1909),
        },
        "note": "(cls, mean_patch) at each depth",
        "best_is_trial_0": (
            "block 12 + cls IS the baseline representation, which the "
            "extraction's bit-identity check confirmed"
        ),
        "against_the_prior": (
            "[LITERATURE] said intermediate layers transfer better for "
            "objectives unlike the pretraining one. Here the final block wins "
            "at every depth compared"
        ),
    },
    "head": {
        "verdict": "NULL -- trials 0-4 span 0.0139, inside the 0.0183 criterion",
        "range": 0.0139,
        "mlp": (
            "0.0437, a capacity failure rather than an architecture finding: "
            "a fixed random projection through 64 units discards most of a "
            "768-dim signal at 152 training samples"
        ),
    },
    "ensemble": {
        "verdict": "the ONLY axis that moves",
        "best": {
            "trial": 22, "inner_val_pcc": 0.2236, "over_trial_0": 0.0221,
            "configuration": "vit+swin, concat_l2norm, linear",
            "clears": EXHAUSTION_WINDOW and 0.0183,
        },
        "n_beating_trial_0": 3,
        "standardisation_hurts": (
            "both concat_standardised configurations fall below trial 0, "
            "matching the head axis where linear_standardised also lost. Two "
            "axes independently say standardisation hurts on this data"
        ),
    },
    #: **[MEASURED 2026-08-03] The inner-val lead did not transfer, and the
    #: winner is CLAIMABLY WORSE than the arm it was tuning.**
    #:
    #: Trial 22 led inner-validation by +0.0221 and came in at **0.1978 OOF
    #: (sd 0.0215, five seeds)** against the baseline's 0.2520 -- a delta of
    #: **-0.0542** against a threshold of 0.0229 from the two arms' own SDs.
    #: It clears ``single_run_95`` (0.0512) as well, making it only the third
    #: delta in the project to survive without averaging.
    #:
    #: **Every seed falls short**: 0.2254, 0.1965, 0.2105, 0.1872, 0.1693 --
    #: the highest is 0.0266 below the baseline mean. So this is not a
    #: near-miss averaged away; no run of the tuned arm reached the untuned
    #: one.
    #:
    #: **This is the protocol working, not failing.** Selecting on
    #: inner-validation is what made the gap visible; had the search ranked on
    #: out-of-fold it would have reported a winner that does not exist, and
    #: the seed bands and the claim criterion would have been invalidated
    #: together. A +0.022 inner-val lead becoming a -0.054 out-of-fold loss is
    #: the exact quantity the discipline exists to expose.
    "outcome": {
        "winner": "trial 22 -- vit+swin, concat_l2norm, linear",
        "inner_val_pcc": 0.2236,
        "inner_val_lead": 0.0221,
        "oof_pcc": 0.1978,
        "oof_sd": 0.0215,
        "oof_per_seed": [0.2254, 0.1965, 0.2105, 0.1872, 0.1693],
        "against_baseline": {
            "baseline": 0.2520,
            "delta": -0.0542,
            "threshold": 0.0229,
            # **[WITHDRAWN 2026-08-04] This said `claimable: True` and read
            # "claimably WORSE". Both were condition 2 only.** PLAN §4.3 needs
            # both conditions; condition 1 was in this run's own metrics.json
            # the whole time, saying UNRESOLVED. See
            # CLAIMABLY_WORSE_WITHDRAWN.
            "claimable": False,
            "claimable_condition_2_only": True,
            "single_run_95": 0.0512,
            "survives_single_run_95": True,
            "condition_1": {
                "n_excluding_zero": 1,
                "n_seeds": 5,
                "same_direction": True,
                "per_seed_deltas": [-0.024, -0.064, -0.047, -0.045, -0.092],
                "only_interval_excluding_zero": {
                    "seed": 12345, "lo": -0.176, "hi": -0.008,
                },
            },
            "reading": (
                "UNRESOLVED. The tuned arm scores lower at every seed, but "
                "one of five intervals excludes zero -- this cohort cannot "
                "resolve a -0.054 difference"
            ),
            "reading_before_condition_1": (
                "the tuned arm is claimably WORSE than the untuned one"
            ),
        },
        "every_seed_short": True,
        "exhaustion": {
            "exhausted": True,
            "best_gain_in_window": 0.0082,
            "threshold": 0.0183,
            "window": EXHAUSTION_WINDOW,
        },
    },
    "verdict": (
        "NOTHING BEAT 0.2520. Twenty-four configurations searched, the "
        "exhaustion criterion met, and the winner claimably worse out of fold "
        "than the arm it tuned"
    ),
    "the_protocol_worked": (
        "a +0.0221 inner-val lead became a -0.0542 out-of-fold loss. "
        "Selecting on inner-validation is what made that visible; ranking on "
        "out-of-fold would have reported a winner that does not exist"
    ),
    "for_the_write_up": (
        "the winner's number carries two declared divergences from the ladder "
        "(BASELINE['also_varies']). It does not need re-running through "
        "train_cv to support the conclusion -- it lost, and re-running it "
        "under the ladder's own head could only move it further from a claim "
        "it already fails to make"
    ),
}


def summary() -> dict:
    """The pre-registration in one object, for the run record."""
    return {
        "budget": SEARCH_BUDGET,
        "n_configurations": len(configurations()),
        "by_axis": {
            "head": len(HEAD_AXIS),
            "pooling": len(POOLING_AXIS),
            "ensemble": len(ENSEMBLE_AXIS),
        },
        "selection_metric": SELECTION_METRIC,
        "monitor": MONITOR,
        "ladder_monitor": LADDER_MONITOR,
        "monitor_diverges": MONITOR != LADDER_MONITOR,
        "exhaustion": {
            "window": EXHAUSTION_WINDOW,
            "threshold": round(exhaustion_threshold(), 6),
            "derived_from": "the baseline arm's own SD over its own seed count",
        },
        "baseline": BASELINE,
        "out_of_budget": sorted(OUT_OF_BUDGET),
    }
