"""Measure where CleftGNN's predictions stop being usable. Pod-only.

    PYTHONPATH=src python scripts/p10_scale_probe.py \
        --config configs/p10_cleftgnn.yaml

**A probe, not a run** (the ``p8b_softmax_probe`` precedent): no
RunContext, no artifacts, nothing declared or claimed. It exists because
the mechanism cannot be measured on the laptop -- ``torchvision::roi_align``
has no CPU kernel there -- so the numbers this prints are the ones
``phase10.CLEFTGNN_SCALE_MEASURED`` could only measure through a declared
substitute.

What it prints, per ``phase10.P10_SCALE_PROBE``:

1. first-batch loss and max|grad| for the first fold;
2. whether logits are finite at epoch 0 and again after step 1;
3. the prediction vector's sd at epoch 0, after step 1, after epoch 1 --
   **a zero sd is COLLAPSE and a nan is BLOW-UP, and they are different
   findings** (``metrics.pcc`` returns nan for the first and raises for
   the second, which is exactly how the two arms differed);
4. |f_t| and |v_a| on REAL features, against the 21.5 measured offline;
5. the same numbers for one faithful-arm rater split, so "do the arms
   share the mechanism" is answered rather than argued.

**[EXTENDED 2026-08-17, again]** ``--gate3`` decomposes epoch-0
calibration across the config's seeds, for whichever cell the config
names -- magnitude of what the classifier receives, the class-to-class
spread of ``W . f``, softmax saturation, the epoch-0 mean against gate
3's own 0.5-SD limit, and the zeroed-weight control that isolates
``W . f``. It exists because the notebook cell died at gate 3 fold 0 and
the offline account of why needs confirming on real features
(``phase10.GATE_3_FAILED_ON_THE_NOTEBOOK_CELL``). The probe now reads
``task.recipe`` from the config and checks the stage table against
``cleftgnn.RECIPE_STAGES``, so it measures the cell it is pointed at.

**[EXTENDED 2026-08-17]** ``--stages`` prints the STAGE-VARIANCE TABLE on
real pretrained features: across-image sd relative to each stage's own
magnitude, from the input through to the logits. Offline (random-init
backbone, substitute op) that table located the collapse at the K==1
sigmoid -- ratio falling 0.648 -> 0.017 while the SABM path held 0.786
(``phase10.COLLAPSE_DIAGNOSED``). Their pre-activation scale depends on
real features, so this is the measurement that confirms or corrects the
diagnosis before anything is changed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import numpy as np  # noqa: E402
import yaml  # noqa: E402


def _describe(values) -> str:
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    if not finite.any():
        return "ALL NON-FINITE (blow-up)"
    unique = len(np.unique(np.round(values[finite], 8)))
    sd = float(np.std(values[finite]))
    tag = " COLLAPSE" if unique == 1 else ""
    missing = "" if finite.all() else f", {int((~finite).sum())} non-finite"
    return f"sd {sd:.6f}, unique {unique}{tag}{missing}"


def probe_split(backbone, images, labels, fit_rows, name, log=print):
    """The five numbers, for one fit split."""
    import torch

    log(f"\n--- {name}: n_fit {len(fit_rows)} ---")
    backbone.reset(labels[fit_rows])
    report = backbone.parameter_report
    log(f"frozen {report['frozen_backbone_parameters']:,} | "
        f"trainable {report['trainable_parameters']:,}")

    # (2)+(3) epoch 0.
    with torch.no_grad():
        for batch, _ in backbone._batches(images[fit_rows[:8]], None, False):
            logits = backbone._model(batch)
            log(f"epoch 0: logits finite {bool(torch.isfinite(logits).all())}, "
                f"|max| {float(logits.abs().max()):.4f}")
            break
    log(f"epoch 0: predictions {_describe(backbone.predict(images[fit_rows]))}")

    # (4) the scale on REAL features -- READ FROM THE MODEL'S OWN FORWARD,
    # never rebuilt (phase10.PROBE_RECONSTRUCTED_THE_PIPELINE).
    with torch.no_grad():
        stages: dict = {}
        for batch, _ in backbone._batches(images[fit_rows[:8]], None, False):
            backbone._model(batch, stages=stages)
            break
        log(f"scale on real features: |feature map| "
            f"{float(stages['backbone_map'].abs().max()):.3f}, |f_t| "
            f"{float(stages['gated_pooling_f_t'].abs().max()):.3f}, "
            f"|gnn_norm(f_t)| "
            f"{float(stages['gnn_norm_f_t'].abs().max()):.3f}, |v_a| "
            f"{float(stages['sabm_v_a'].abs().max()):.3f}")

    # (1)+(2)+(3) the first step, then the rest of epoch 1.
    backbone._model.train()
    backbone._model.backbone.eval()
    first = True
    total, seen = 0.0, 0
    for batch, target in backbone._batches(images[fit_rows], labels[fit_rows], True):
        backbone._optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.cross_entropy(backbone._model(batch), target)
        loss.backward()
        grads = [
            float(p.grad.abs().max())
            for p in backbone._model.parameters() if p.grad is not None
        ]
        if first:
            log(f"first batch: loss {float(loss.item()):.6f}, "
                f"max|grad| {max(grads):.4e}")
        backbone._optimizer.step()
        if first:
            with torch.no_grad():
                after = backbone._model(batch)
            log(f"after step 1: logits finite "
                f"{bool(torch.isfinite(after).all())}, |max| "
                f"{float(after.abs().max()):.4f}")
            log(f"after step 1: predictions "
                f"{_describe(backbone.predict(images[fit_rows]))}")
            first = False
        total += float(loss.item()) * len(target)
        seen += len(target)
    log(f"epoch 1: mean loss {total / max(seen, 1):.6f}")
    log(f"epoch 1: predictions {_describe(backbone.predict(images[fit_rows]))}")



#: The stage list and the ratio arithmetic both live in the MODEL
#: (``cleftgnn.STAGE_ORDER``, ``cleftgnn.stage_ratios``) -- a second copy
#: here is exactly what produced phase10.PROBE_RECONSTRUCTED_THE_PIPELINE.


def stage_variance(backbone, images, rows, log=print):
    """Across-image sd per stage, relative to that stage's magnitude.

    **[REBUILT 2026-08-17, phase10.PROBE_RECONSTRUCTED_THE_PIPELINE]**
    This reads the stages THE MODEL RECORDS DURING ITS OWN FORWARD. The
    previous version rebuilt the pipeline by hand and fell behind the
    model -- it never applied ``gnn_norm``, so a post-fix table came back
    identical to the pre-fix one. There is now exactly one implementation
    of this computation, and the instrument cannot describe an
    architecture the model is not running.

    A high ratio means the stage still distinguishes images; a collapse
    between two rows is where the head stops depending on its input
    (``phase10.COLLAPSE_DIAGNOSED``).
    """
    import torch

    stages: dict = {}
    log("\n--- STAGE VARIANCE on real features (from the model's own forward) ---")
    log("stage                              magnitude       variation")
    with torch.no_grad():
        for batch, _ in backbone._batches(images[rows[:20]], None, False):
            backbone._model(batch, stages=stages)
            break

    from cleft.models.cleftgnn import RECIPE_STAGES, stage_ratios

    # Recipe-aware, from the MODEL's own list: the notebook path has no
    # sabm_v_a, gnn_norm_f_t or after_layernorm, and the manuscript path
    # has no acm_v. Checking against all fourteen would have made this
    # probe unusable on the second cell -- or, worse, made it quietly
    # skip stages (phase10.NOTEBOOK_RECIPE_CELL_REGISTERED).
    expected = RECIPE_STAGES[backbone.recipe]
    missing = [name for name in expected if name not in stages]
    if missing:
        raise SystemExit(
            f"the {backbone.recipe} model did not report {missing}. The "
            "instrument reads the model's own stages; a renamed stage must "
            "be renamed there too, loudly, rather than silently dropped "
            "from the table."
        )
    for name, (magnitude, across, ratio) in stage_ratios(stages).items():
        flag = "  <-- SUSPECT" if name == "sigmoid_f_hat" else ""
        log(f"{name:34s} |mean| {magnitude:9.4f}  sd {across:9.4f}  "
            f"ratio {ratio:.4f}{flag}")

    # The readout uses the model's OWN expected_grade: writing the
    # softmax-weighted sum again here would be the very defect this
    # probe was rebuilt to remove (one computation, one implementation).
    from cleft.models.cleftgnn import expected_grade

    probabilities = torch.softmax(stages["logits"], dim=1).cpu().numpy()
    expected = expected_grade(probabilities)
    log(f"\nexpected grade across images: sd {float(expected.std()):.6f}, "
        f"range [{float(expected.min()):.4f}, {float(expected.max()):.4f}]")
    log("\nPre-registered (phase10.GNN_BRANCH_NORM): with the GNN branch "
        "normalised, fused should land in 0.4-0.7 (was 0.1315), logits "
        "several-fold above 0.0214, and gated_pooling_f_t's own ratio "
        "should still be ~0.0068. If fused does not clear ~0.4, the "
        "dilution was not the binding constraint and the sigmoid is.")


def gate3_decomposition(make_backbone, images, labels, fit_rows, seeds,
                        recipe, log=print):
    """Why gate 3 sees what it sees, decomposed, on real features.

    ``phase10.GATE_3_FAILED_ON_THE_NOTEBOOK_CELL``. Gate 3 asks that an
    untrained head predict the training-fold mean. The head's output is
    ``softmax(W . f + b)``, where ``b`` is the Laplace bias -- which alone
    reproduces the fold mean exactly -- so any departure is ``W . f``, and
    ``W . f`` scales with ``||f||``. This prints, per seed:

    * the magnitude of what the classifier actually receives;
    * the class-to-class spread of ``W . f``, against ``sd(W) * ||f||``;
    * the mean top-class probability, i.e. how saturated the softmax is;
    * the epoch-0 predicted mean and its distance from the fold mean in
      label SD, with gate 3's own 0.5 limit applied;
    * the SAME split with the classifier weight zeroed -- the control that
      isolates ``W . f`` from everything else.

    Run it across seeds because the offline measurement says the offset is
    a per-SEED constant rather than a per-fold accident: within a seed the
    epoch-0 mean barely moves between folds, and between seeds it wanders
    across the whole grade range. If that holds on real features, a seed
    does not fail one fold -- it fails all five.
    """
    import torch

    from cleft.models.cleftgnn import (
        CLASSIFIER_INPUT_STAGE, expected_grade,
    )

    fold_labels = labels[fit_rows]
    fold_mean = float(np.mean(fold_labels))
    fold_sd = float(np.std(fold_labels))
    log(f"\n--- GATE 3 DECOMPOSITION ({recipe}) ---")
    log(f"train fold: n {len(fit_rows)}, mean {fold_mean:.4f}, sd "
        f"{fold_sd:.4f}; gate 3 limit 0.5 SD = {0.5 * fold_sd:.4f}")
    log(f"{'seed':>6}{'rms(f)':>10}{'||f||':>10}{'sd(W.f)':>10}"
        f"{'predicted':>10}{'p_max':>8}{'mean':>9}{'SD off':>8}"
        f"{'  gate3':>8}{'W=0 mean':>10}")
    for seed in seeds:
        backbone = make_backbone(seed)
        backbone.reset(fold_labels)
        stage = CLASSIFIER_INPUT_STAGE[backbone.recipe]
        stages: dict = {}
        with torch.no_grad():
            for batch, _ in backbone._batches(images[fit_rows], None, False):
                logits = backbone._model(batch, stages=stages)
                break
            seen = stages[stage]
            rms = float(seen.pow(2).mean().sqrt())
            norm2 = float(seen.pow(2).sum(dim=1).sqrt().mean())
            spread = float(
                (logits - backbone._model.classifier.bias).std(dim=1).mean()
            )
            probabilities = torch.softmax(logits, dim=1)
            p_max = float(probabilities.max(dim=1).values.mean())
        predicted = backbone.predict(images[fit_rows])
        mean = float(np.mean(predicted))
        off = abs(mean - fold_mean) / fold_sd if fold_sd else float("nan")
        with torch.no_grad():
            backbone._model.classifier.weight.zero_()
        control = float(np.mean(backbone.predict(images[fit_rows])))
        log(f"{seed:>6}{rms:>10.4f}{norm2:>10.2f}{spread:>10.3f}"
            f"{_describe(predicted).split(',')[0]:>10}{p_max:>8.4f}"
            f"{mean:>9.4f}{off:>8.2f}"
            f"{('  pass' if off <= 0.5 else '  FAIL'):>8}{control:>10.4f}")
    closed = (len(fold_labels) * fold_mean + 15) / (len(fold_labels) + 5)
    log(f"\nW = 0 predicts (N*mean + 15)/(N + 5) = {closed:.4f} on this "
        f"fold, {abs(closed - fold_mean) / fold_sd:.3f} SD out -- the "
        "Laplace bias alone IS calibrated, so every departure above is "
        "W . f and nothing else.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/p10_cleftgnn.yaml")
    parser.add_argument(
        "--stages", action="store_true",
        help="print the stage-variance table on real features",
    )
    parser.add_argument(
        "--gate3", action="store_true",
        help="decompose epoch-0 calibration across the config's seeds",
    )
    args = parser.parse_args(argv)

    from cleft.data import scoresheet
    from cleft.data.manifest import load_manifest
    from cleft.models import cleftgnn
    from cleft.train import phase3

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    declared = {e["name"]: Path(e["path"]) for e in config["inputs"]}
    task = config["task"]
    manifest_dir = declared[task["manifest_artifact"]]
    images, _, patient_ids, _ = phase3.load_inputs(
        manifest_dir, declared[task["staged_artifact"]],
        task["geometry"], task["label"],
    )
    rows = {int(r["patient_id"]): r for r in load_manifest(manifest_dir / "manifest.csv")}
    sheet_path = declared[task["scoresheet_artifact"]]

    # The recipe comes from the config, exactly as the run reads it, so
    # this probe measures whichever cell it is pointed at rather than
    # always the manuscript one.
    recipe = str(task.get("recipe", "manuscript"))
    optimizer = str(task.get("optimizer", "sgd"))

    def backbone_of(seed=None):
        return cleftgnn.CleftGNNBackbone(
            learning_rate=float(task.get("learning_rate", 0.01)),
            momentum=float(task.get("momentum", 0.0)),
            batch_size=int(task.get("batch_size", 16)),
            seed=int(config["seed"] if seed is None else seed),
            recipe=recipe,
            optimizer=optimizer,
        )

    # The protocol arm's first fold: train on folds != 0.
    median = scoresheet.load_median(sheet_path)
    labels = np.array(
        [float(median[int(rows[int(p)]["frontal_id"])]) for p in patient_ids]
    )
    folds = np.array([int(rows[int(p)]["fold"]) for p in patient_ids])
    protocol_rows = np.flatnonzero(folds != 0)
    if args.gate3:
        gate3_decomposition(
            backbone_of, images, labels, protocol_rows,
            [int(s) for s in task.get("seeds", [config["seed"]])], recipe,
        )
        return 0
    if args.stages:
        staged = backbone_of()
        staged.reset(labels[protocol_rows])
        stage_variance(staged, images, protocol_rows)
        return 0
    probe_split(
        backbone_of(), images, labels, protocol_rows,
        "PROTOCOL ARM, fold 0 held out (median label)",
    )

    # One faithful-arm split: the first rater, 85% of the cohort.
    sheet = scoresheet.load(sheet_path)
    rater = scoresheet.RATERS[0]
    rater_labels = np.array([
        float(sheet.rows[int(rows[int(p)]["frontal_id"])].grades[0])
        for p in patient_ids
    ])
    rng = np.random.default_rng(int(config["seed"]))
    order = rng.permutation(len(patient_ids))
    probe_split(
        backbone_of(), images, rater_labels,
        order[: int(0.85 * len(order))],
        f"FAITHFUL ARM, {rater} (85% split)",
    )
    print("\nIf both splits show the same first-step signature, the arms "
          "share the mechanism and the faithful arm's collapse is ours, "
          "not their protocol's.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
