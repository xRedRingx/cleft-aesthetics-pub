"""The Phase 7B search TASK, end to end through ``run.main``.

**[MEASURED 2026-08-02] This file exists because it did not.** The search ran
all 24 trials, made its selection, fitted the winner at five seeds -- and then
died writing the summary, on ``ctx.write_metrics``, a method ``RunContext``
does not have. The name appeared exactly once in the whole of ``run.py``.

**Nothing caught it because nothing exercised it.** ``phase7b.py`` had thirty
tests of its functions and the handler that wires them together had none,
while every other task kind is driven end to end through ``run.main`` with the
real ``RunContext``. So the gap was not a permissive mock swallowing an
unknown attribute -- the project does not use one -- it was an absent test,
which is the harder kind to notice: a mock at least appears in a coverage
report.

The generalisation is in ``test_workflow_hygiene.py``: every ``ctx.<name>`` in
every task is checked against the real class, so a misnamed context call in
ANY handler fails on the laptop rather than after the compute.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from cleft import embeddings as emb
from cleft import phase7b

from fixtures import builders

from test_extract import declare, make_artifacts


def _write_sets(root: Path, patient_ids: list, dim: int = 6):
    """Every set the pre-registered space reads, as small pooled artifacts.

    The REAL space -- all 24 configurations -- because a shrunk one would
    exercise a different pre-registration, and the budget check is half of
    what makes this phase's null credible.
    """
    rng = np.random.default_rng(11)
    directory = root / "embeddings"
    for index, name in enumerate(phase7b.all_required_sets()):
        backbone = "swin_b" if name.startswith("swin_b") else "vit_b16"
        block = token = None
        if "__block" in name:
            suffix = name.rsplit("__block", 1)[1]
            depth, token = suffix.split("_", 1)
            block = int(depth)
        emb.save(
            directory / name,
            rng.normal(size=(len(patient_ids), dim)).astype(np.float32),
            backbone=backbone,
            backbone_kind="transformer",
            init="imagenet",
            geometry="g1",
            variant=None,
            checkpoint_sha256=None,
            patient_ids=list(patient_ids),
            manifest_ids=list(patient_ids),
            block=block,
            token=token,
        )
    return directory


def _run(tmp_path, clean_repo, *, with_baseline: bool, monkeypatch=None):
    manifest_dir, staged_dir = make_artifacts(tmp_path, n=12)
    patient_ids = list(range(1, 13))
    sets_root = _write_sets(tmp_path, patient_ids)

    inputs = [
        declare("manifest_v1", manifest_dir),
        declare("staged_v1", staged_dir),
    ]
    for name in phase7b.all_required_sets():
        inputs.append(declare(f"set_{name}", sets_root / name))

    seeds = [1337, 2024, 7]
    if with_baseline:
        from cleft.cluster_csv import write_predictions
        from cleft.data.manifest import load_manifest
        from cleft.eval.metrics import pcc

        truth = np.array(
            [float(row["mean"]) for row in load_manifest(manifest_dir / "manifest.csv")]
        )
        baseline = tmp_path / "baseline"
        baseline.mkdir()
        rng = np.random.default_rng(3)
        observed = []
        for seed in seeds:
            vector = truth + rng.normal(scale=0.4, size=len(truth))
            observed.append(float(pcc(truth, vector)))
            path = baseline / f"seed_{seed}__predictions.csv"
            write_predictions(
                path,
                [
                    (pid, t, v, pid % 5)
                    for pid, t, v in zip(patient_ids, truth, vector)
                ],
            )
            inputs.append(declare(f"baseline_oof_seed_{seed}", path))

        # **The recorded baseline is patched to what the fixture ACTUALLY
        # scores**, so the run's verification passes because the numbers
        # agree rather than because the check was disabled. Disabling it
        # would leave the one guard against pairing an arm against the wrong
        # run untested by the only test that reaches it.
        monkeypatch.setitem(phase7b.BASELINE, "pcc", float(np.mean(observed)))

    config = builders.write_config(
        tmp_path / "search.yaml",
        phase="p7b",
        inputs=inputs,
        task={
            "kind": "phase7b_search",
            "manifest_artifact": "manifest_v1",
            "staged_artifact": "staged_v1",
            "geometry": "g1",
            "label": "mean",
            "inner_val_frac": 0.25,
            "seeds": seeds,
        },
    )
    from cleft.run import main

    run_dir = main(["--config", str(config), "--out", str(tmp_path / "runs")])
    return Path(run_dir), seeds


def test_the_search_task_runs_end_to_end_and_writes_its_summary(
    tmp_path, clean_repo, monkeypatch
):
    """**The run that crashed.** Twenty-four trials, a selection, the winner's
    seeds, and then the summary write -- which is the step that failed on the
    cluster and the step no test reached."""
    monkeypatch.setenv("CLEFT_SCUT_ROOT", "/nonexistent/sentinel")
    run_dir, seeds = _run(tmp_path, clean_repo, with_baseline=False)
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))

    assert len(metrics["trials"]) == phase7b.SEARCH_BUDGET == 24
    assert metrics["selected"]["index"] in range(24)
    assert metrics["protocol"]["budget"] == 24
    assert "exhausted" in metrics["exhaustion"]

    # The winner's band is its own, over its own seeds.
    assert len(metrics["winner"]["pooled_pccs"]) == len(seeds)
    assert metrics["winner"]["band"]["sd"] is not None

    # Criterion 4 is reported UNMET rather than silently skipped.
    assert metrics["comparison"]["paired"] is None
    assert "NOT" in metrics["comparison"]["why_absent"]

    # The checkpoint file and the winner's per-seed predictions are on disk.
    trials = json.loads((run_dir / "trials.json").read_text(encoding="utf-8"))
    assert len(trials) == 24
    for seed in seeds:
        assert (run_dir / f"seed_{seed}__predictions.csv").is_file()


def test_no_trial_record_ever_carries_an_out_of_fold_number(
    tmp_path, clean_repo, monkeypatch
):
    """The discipline, checked on the ARTIFACT rather than on the function.
    ``trials.json`` is what a reader inspects, and an out-of-fold score
    reaching it would mean the search could have been ranked on one."""
    monkeypatch.setenv("CLEFT_SCUT_ROOT", "/nonexistent/sentinel")
    run_dir, _ = _run(tmp_path, clean_repo, with_baseline=False)
    trials = json.loads((run_dir / "trials.json").read_text(encoding="utf-8"))
    for record in trials:
        assert set(record) == set(phase7b.TRIAL_RECORD_KEYS)
        for forbidden in phase7b.FORBIDDEN_IN_SELECTION:
            assert forbidden not in record


def test_the_paired_comparison_runs_when_the_baseline_is_declared(
    tmp_path, clean_repo, monkeypatch
):
    """Exit criterion 4, per seed. The verification that the declared baseline
    IS the recorded arm is relaxed here -- these are synthetic vectors, not
    the 0.2520 run -- so the check is monkeypatched to the fixture's own
    value rather than skipped, which would leave it untested."""
    monkeypatch.setenv("CLEFT_SCUT_ROOT", "/nonexistent/sentinel")
    run_dir, seeds = _run(
        tmp_path, clean_repo, with_baseline=True, monkeypatch=monkeypatch
    )
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    paired = metrics["comparison"]["paired"]

    assert paired is not None
    assert len(paired["per_seed"]) == len(seeds)
    assert {entry["seed"] for entry in paired["per_seed"]} == set(seeds)
    assert paired["n_seeds"] == len(seeds)
    # The verdict must follow from its own two conditions.
    assert paired["claimable"] == (
        paired["n_excluding_zero"] == paired["n_seeds"]
        and paired["same_direction"]
        and paired["exceeds_threshold"]
    )


def test_pairing_against_the_wrong_run_is_refused(tmp_path, clean_repo, monkeypatch):
    """**The guard that a declared baseline IS the recorded arm.**

    Without it a wrong run directory completes, produces intervals, and
    compares the tuned arm against something else entirely -- every shape
    right, the number merely about a different comparison. Exercised by
    leaving the recorded baseline at its real 0.2520 while the fixture scores
    something else.
    """
    monkeypatch.setenv("CLEFT_SCUT_ROOT", "/nonexistent/sentinel")

    class _NoPatch:
        def setitem(self, *args, **kwargs):
            pass

    with pytest.raises(ValueError, match="the declared baseline scores"):
        _run(tmp_path, clean_repo, with_baseline=True, monkeypatch=_NoPatch())
