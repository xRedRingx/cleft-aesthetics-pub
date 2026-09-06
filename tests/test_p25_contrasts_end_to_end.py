"""Phase 25's contrast task, driven end to end against fixture runs.

**The fourth defect of one class in this phase alone was found at launch**
-- the init vocabulary, the checkpoint declaration, the missing config,
and then ``_p21_load_arm_predictions`` called with arm dicts lacking the
``csv`` key it opens files with. Each was one layer deeper than the last,
and each cost a cluster round trip.

This is the Phase 22 fixture pattern applied to the contrast task: three
small run directories, the real ``RunContext``, the real ``TASKS`` entry,
the real loader and the real ``phase7b.paired_comparison``. It fails in
under a second on the laptop for anything that would fail at launch.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import yaml

from cleft.cluster_csv import PREDICTIONS_COLUMNS
from cleft.eval.metrics import to_3class

N_PATIENTS = 40
SEEDS = (1337, 2024, 7, 99, 12345)
ARMS = ("probe", "p25_arm_d2", "p25_arm_d1")


def _manifest(tmp_path):
    """A small cohort with a real five-rater panel behind ``mean``."""
    rng = np.random.default_rng(25)
    patient_ids = list(range(1, N_PATIENTS + 1))
    sums = rng.integers(10, 16, size=N_PATIENTS)
    means = sums / 5.0
    classes = to_3class(means)

    directory = tmp_path / "manifest"
    directory.mkdir()
    header = (
        "patient_id,frontal_id,basal_id,mean,median,mode,weighted_mean,"
        "orthodontist,soft_1,soft_2,soft_3,soft_4,soft_5,class3,fold"
    )
    lines = ["# CLUSTER-ONLY: patient-keyed", header]
    for index, pid in enumerate(patient_ids):
        n_three = int(sums[index]) - 10
        n_two = 5 - n_three
        soft = [0.0, n_two / 5.0, n_three / 5.0, 0.0, 0.0]
        value = float(means[index])
        lines.append(
            f"{pid},{pid},{pid},{value:.4f},{round(value)},{round(value)},"
            f"{value:.4f},{round(value)},"
            + ",".join(f"{s:.4f}" for s in soft)
            + f",{int(classes[index])},{index % 5}"
        )
    (directory / "manifest.csv").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    return directory, patient_ids, np.array(means, dtype=float)


def _run_dir(tmp_path, name, patient_ids, truth, *, noise, rng):
    """A run directory holding one prediction CSV per seed.

    ``noise`` sets how far the arm's predictions sit from the truth, so
    the three fixture arms are genuinely different and the contrasts have
    something to order.
    """
    directory = tmp_path / "runs" / name
    directory.mkdir(parents=True)
    for seed in SEEDS:
        rows = ["# CLUSTER-ONLY: patient-keyed", ",".join(PREDICTIONS_COLUMNS)]
        predictions = truth + rng.normal(0.0, noise, size=len(truth))
        for index, pid in enumerate(patient_ids):
            rows.append(
                f"{pid},{truth[index]:.6f},{predictions[index]:.6f},"
                f"{index % 5}"
            )
        (directory / f"seed_{seed}__predictions.csv").write_text(
            "\n".join(rows) + "\n", encoding="utf-8"
        )
    return directory


def _config(tmp_path, manifest_dir, run_dirs, **overrides):
    from cleft.provenance import hash_dir

    inputs = [{
        "name": "manifest_v1", "path": str(manifest_dir),
        "rollup_sha256": hash_dir(manifest_dir)["rollup"],
    }]
    for name, directory in run_dirs.items():
        inputs.append({
            "name": name, "path": str(directory),
            "rollup_sha256": hash_dir(directory)["rollup"],
        })

    arms = []
    for name in ARMS:
        entry = {
            "name": name, "input": name, "seeds": list(SEEDS),
            "csv": "predictions", "n_patients": N_PATIENTS,
        }
        entry.update(overrides.pop(f"arm_{name}", {}))
        arms.append(entry)

    task = {
        "kind": "p25_contrasts",
        "manifest_artifact": "manifest_v1",
        "arms": arms,
        "seeds": list(SEEDS),
        "n_boot": 200,               # small: the shape is what is tested
        "expect_contrasts": 3,
        "expect_patients": N_PATIENTS,
    }
    task.update(overrides)
    config = {
        "schema_version": 1, "phase": "p25", "tier": "keeper", "seed": 1337,
        "inputs": inputs, "task": task,
    }
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def _cohort(tmp_path):
    manifest_dir, patient_ids, truth = _manifest(tmp_path)
    rng = np.random.default_rng(2509)
    run_dirs = {
        "probe": _run_dir(
            tmp_path, "probe", patient_ids, truth, noise=0.35, rng=rng
        ),
        "p25_arm_d2": _run_dir(
            tmp_path, "p25_arm_d2", patient_ids, truth, noise=0.90, rng=rng
        ),
        "p25_arm_d1": _run_dir(
            tmp_path, "p25_arm_d1", patient_ids, truth, noise=0.40, rng=rng
        ),
    }
    return manifest_dir, run_dirs, patient_ids


def test_the_contrast_task_runs_end_to_end_to_three_verdicts(
    tmp_path, clean_repo
):
    """**Pre-fix this raised** ``KeyError: 'csv'`` from
    ``_p21_load_arm_predictions`` at ``seed_{seed}__{arm['csv']}.csv`` --
    the cluster's failure, on the laptop, in under a second. **Post-fix
    it runs through to three verdicts.**

    It would also have caught the three earlier Phase 25 defects, each of
    which died one layer further in: an init outside the vocabulary at
    ``expected_variant``, a checkpoint requirement at ``check_pairing``,
    and a config that does not exist at ``load_config``.
    """
    from cleft.provenance import RunContext
    from cleft.run import TASKS

    manifest_dir, run_dirs, _ = _cohort(tmp_path)
    config = _config(tmp_path, manifest_dir, run_dirs)

    with RunContext(config, tmp_path / "out", repo_root=clean_repo) as ctx:
        TASKS["p25_contrasts"](ctx)

    metrics = json.loads(
        (ctx.run_dir / "metrics.json").read_text(encoding="utf-8")
    )
    assert metrics["family_size"] == 3
    assert set(metrics["verdicts"]) == {
        "d2-vs-probe", "d1-vs-probe", "d2-vs-d1"
    }

    # Every arm's own mean and sd are MEASURED from the fixtures, not
    # declared -- the reason winner_sd is not a config field.
    assert set(metrics["arm_stats"]) == set(ARMS)
    for name, stats in metrics["arm_stats"].items():
        assert len(stats["per_seed_pcc"]) == len(SEEDS), name
        assert stats["sd"] > 0.0, name
    # The fixture arms really are ordered, so the contrasts have signal.
    assert (metrics["arm_stats"]["probe"]["mean"]
            > metrics["arm_stats"]["p25_arm_d2"]["mean"])


def test_every_verdict_carries_both_conditions_and_its_n(
    tmp_path, clean_repo
):
    """The Phase 22 null-condition defect, checked on real output rather
    than in the task's source."""
    from cleft.provenance import RunContext
    from cleft.run import TASKS

    manifest_dir, run_dirs, _ = _cohort(tmp_path)
    config = _config(tmp_path, manifest_dir, run_dirs)
    with RunContext(config, tmp_path / "out", repo_root=clean_repo) as ctx:
        TASKS["p25_contrasts"](ctx)
    metrics = json.loads(
        (ctx.run_dir / "metrics.json").read_text(encoding="utf-8")
    )

    for key, row in metrics["verdicts"].items():
        assert isinstance(row["condition_1"], bool), key
        assert isinstance(row["condition_2"], bool), key
        assert row["n_seeds"] == len(SEEDS), key
        assert row["n_patients"] == N_PATIENTS, key
        assert row["threshold"], key
        assert row["verdict"] in ("claimable", "withdrawn", "unresolved"), key
        # The verdict follows from the two conditions, not from a third
        # place: claimable requires both.
        assert (row["verdict"] == "claimable") == (
            row["condition_1"] and row["condition_2"]
        ), key


def test_a_missing_arm_key_fails_here_rather_than_at_launch(
    tmp_path, clean_repo
):
    """**The regression that earned this file.** An arm dict without the
    loader's ``csv`` key must die in the suite."""
    from cleft.provenance import RunContext
    from cleft.run import TASKS

    manifest_dir, run_dirs, _ = _cohort(tmp_path)
    config = _config(tmp_path, manifest_dir, run_dirs)
    payload = yaml.safe_load(config.read_text(encoding="utf-8"))
    for arm in payload["task"]["arms"]:
        arm.pop("csv")
    config.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    # The schema now catches it BEFORE the task runs -- one layer earlier
    # than the KeyError the cluster saw.
    with pytest.raises(Exception) as caught:
        with RunContext(config, tmp_path / "out", repo_root=clean_repo) as ctx:
            TASKS["p25_contrasts"](ctx)
    assert "csv" in str(caught.value)


def test_a_partial_family_is_refused(tmp_path, clean_repo):
    """All three contrasts run or none do."""
    from cleft.provenance import RunContext
    from cleft.run import TASKS

    manifest_dir, run_dirs, _ = _cohort(tmp_path)
    run_dirs.pop("p25_arm_d1")
    config = _config(tmp_path, manifest_dir, run_dirs)
    payload = yaml.safe_load(config.read_text(encoding="utf-8"))
    payload["task"]["arms"] = [
        arm for arm in payload["task"]["arms"] if arm["name"] != "p25_arm_d1"
    ]
    config.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="not declared"):
        with RunContext(config, tmp_path / "out", repo_root=clean_repo) as ctx:
            TASKS["p25_contrasts"](ctx)


def test_every_call_the_task_makes_matches_its_callees_signature():
    """**The Phase 22 signature audit, applied to this task.** One pass
    over every call into shipped code, rather than one launch per defect.

    The ``csv`` KeyError was invisible to a signature check -- the call
    was well-formed and the DICT was short -- so this audits required
    dict keys as well as parameters.
    """
    import inspect
    import re

    from cleft import phase25
    from cleft import run as run_module
    from cleft.config.schema import TASK_SPECS
    from cleft.data.manifest import load_manifest
    from cleft.eval.metrics import pcc
    from cleft.phase7b import paired_comparison
    from cleft.provenance import RunContext

    task_src = inspect.getsource(run_module.task_p25_contrasts)

    # ---- parameters, against the real signatures --------------------
    loader = inspect.signature(run_module._p21_load_arm_predictions)
    assert list(loader.parameters)[:4] == [
        "ctx", "task", "declared", "patient_ids"
    ]
    assert "_p21_load_arm_predictions(ctx, task, declared, patient_ids)" in task_src

    comparison = inspect.signature(paired_comparison)
    passed = {"truth", "winner_by_seed", "baseline_by_seed", "winner_sd",
              "n_boot"}
    required = {
        name for name, p in comparison.parameters.items()
        if p.default is inspect.Parameter.empty
    }
    assert required <= passed, sorted(required - passed)
    assert passed <= set(comparison.parameters), sorted(
        passed - set(comparison.parameters)
    )
    # Everything it passes is keyword-only, as the callee requires.
    assert all(
        comparison.parameters[name].kind is inspect.Parameter.KEYWORD_ONLY
        for name in passed
    )

    assert list(inspect.signature(pcc).parameters) == ["truth", "pred"]
    assert list(inspect.signature(load_manifest).parameters) == ["path"]
    assert list(inspect.signature(phase25.contrast_family).parameters) == []
    assert list(inspect.signature(RunContext.atomic).parameters) == [
        "self", "name", "tier"
    ]

    # ---- required DICT KEYS, which is where this defect lived -------
    loader_src = inspect.getsource(run_module._p21_load_arm_predictions)
    arm_keys = set(re.findall(r"arm\[.([a-z_]+).\]", loader_src))
    assert arm_keys == {"name", "input", "seeds", "csv", "n_patients"}
    declared_arm_keys = set(TASK_SPECS["p25_contrasts"]["arms"].item_spec)
    assert arm_keys <= declared_arm_keys, sorted(arm_keys - declared_arm_keys)

    loader_task_keys = set(re.findall(r"task\[.([a-z_]+).\]", loader_src))
    task_keys = set(re.findall(r"task\[.([a-z_]+).\]", task_src))
    spec_keys = set(TASK_SPECS["p25_contrasts"])
    assert (loader_task_keys | task_keys) <= spec_keys, sorted(
        (loader_task_keys | task_keys) - spec_keys
    )

    # The contrast entries carry every key the task reads off them.
    read_off = set(re.findall(r"contrast\[.([a-z_]+).\]", task_src))
    for entry in phase25.contrast_family():
        assert read_off <= set(entry), sorted(read_off - set(entry))


def test_the_shipped_config_satisfies_the_audit():
    """The audit above is against the SPEC; this is against the file."""
    import re
    import inspect

    from cleft import run as run_module

    config = yaml.safe_load(
        (
            __import__("pathlib").Path(__file__).resolve().parents[1]
            / "configs" / "p25_contrasts.yaml"
        ).read_text(encoding="utf-8")
    )
    loader_src = inspect.getsource(run_module._p21_load_arm_predictions)
    arm_keys = set(re.findall(r"arm\[.([a-z_]+).\]", loader_src))
    for arm in config["task"]["arms"]:
        assert arm_keys <= set(arm), (arm["name"], sorted(arm_keys - set(arm)))
        assert arm["csv"] == "predictions"
        assert arm["n_patients"] == 237
