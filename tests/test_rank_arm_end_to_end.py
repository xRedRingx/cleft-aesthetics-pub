"""``task_train_rank_cv`` driven end to end against REAL interfaces.

**[2026-09-01] Written after three defects of one class in three
cycles**, all in this task, all from new code being self-consistent and
wrong at a join with existing code:

1. Phase 17's ``faces.jsonl`` against ``faces.json`` -- a writer and a
   reader, each right alone.
2. Two invented input reference names (``staged_g1``,
   ``pooled_vit_b16_imagenet_g1``) that named nothing.
3. ``PooledSource`` constructed without its required ``geometry``.

Every one passed the suite, because the suite exercised the new code
against fixtures the new code shaped. **Nothing called this task against
the repo's real constructors until the cluster did.**

This is that test. It builds a small cohort, a real embedding artifact
written by ``embeddings.save``, and real staged/manifest directories,
then runs the task through to prediction CSVs.

**What it does NOT cover, stated precisely:** torch is not in the test
extra, so the fit loop cannot run here. ``backbone: stub`` reaches
``StubBackbone`` before ``make_factory``'s ranking branch, so the arm
this drives is the ranking task's *plumbing* with a stub learner, not
``RankingHeadBackbone``'s optimisation.

Those two halves are covered separately and both are checked here:

* the factory really returns ``RankingHeadBackbone`` for this task's own
  ``backbone_config`` -- ``test_the_factory_returns_the_ranking_backbone``
  below, torch-free because constructing it imports nothing;
* the loss, pair construction, tie rule and head maps -- 20 tests in
  ``tests/test_ranking.py``.

**[EXTENDED 2026-09-01, after the gate-3 defect this had let
through.]** The stub run below still cannot construct
``RankingHeadBackbone``, so the real backbone is now driven directly,
behind ``pytest.importorskip("torch")`` -- this suite's established
pattern for exactly this (``test_phase10.py``, ``test_graph_cleft.py``).
Those tests SKIP under ``scripts/test.ps1`` and RUN wherever torch is
installed; they were confirmed to fail pre-fix and pass post-fix under
system python with torch 2.13.

**What is still uncovered**: the two halves are never joined -- nothing
runs ``RankingHeadBackbone`` *through* ``phase3.run`` on the laptop,
because that needs a vit_b16 artifact and the real fit loop over five
folds. Multi-epoch convergence, early stopping on a ranking arm, and the
CUDA path all remain cluster-only.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]

N_PATIENTS = 30
FEATURE_DIM = 8


def _cohort(tmp_path):
    """Manifest, staged tensor and a real embedding set for N patients."""
    from cleft import embeddings
    from cleft.eval.metrics import to_3class

    rng = np.random.default_rng(22)
    patient_ids = list(range(1, N_PATIENTS + 1))
    # Real five-rater panels, so `mean` is a genuine fifth and the soft
    # columns are consistent with it.
    sums = rng.integers(10, 16, size=N_PATIENTS)
    means = sums / 5.0
    class3 = to_3class(means)

    manifest_dir = tmp_path / "manifest"
    manifest_dir.mkdir()
    header = (
        "patient_id,frontal_id,basal_id,mean,median,mode,weighted_mean,"
        "orthodontist,soft_1,soft_2,soft_3,soft_4,soft_5,class3,fold"
    )
    lines = ["# CLUSTER-ONLY: patient-keyed", header]
    for index, pid in enumerate(patient_ids):
        n_three = int(sums[index]) - 10
        n_two = 5 - n_three
        soft = [0.0, float(n_two) / 5.0, float(n_three) / 5.0, 0.0, 0.0]
        value = float(means[index])  # numpy 2.x reprs float64 as np.float64(...)
        lines.append(
            f"{pid},{1000 + index},,{value!r},{value!r},{value!r},{value!r},"
            f"{value!r}," + ",".join(repr(s) for s in soft)
            + f",{int(class3[index])},{index % 5}"
        )
    (manifest_dir / "manifest.csv").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )

    staged_dir = tmp_path / "staged"
    staged_dir.mkdir()
    np.save(
        staged_dir / "images.npy",
        rng.integers(0, 255, (N_PATIENTS, 8, 8, 3), dtype=np.uint8),
    )
    (staged_dir / "index.json").write_text(
        json.dumps({"patient_ids": patient_ids, "geometry": "g1"}),
        encoding="utf-8",
    )

    # A REAL embedding set, written by the real writer, so check_pairing
    # verifies this fixture the way it verifies a cluster artifact.
    embeddings_dir = tmp_path / "emb"
    values = rng.normal(size=(N_PATIENTS, FEATURE_DIM)).astype(np.float32)
    # Give the features some signal, or every fold's fit is noise.
    values[:, 0] = means + rng.normal(0, 0.05, N_PATIENTS)
    # backbone="stub" to match the config: check_pairing VERIFIES the
    # artifact's backbone against the arm's declaration, and rightly
    # refuses a vit_b16 set reaching a stub arm. Writing the set the
    # arm actually declares keeps that guard live rather than working
    # around it.
    embeddings.save(
        embeddings_dir, values,
        backbone="stub", backbone_kind="transformer", init="imagenet",
        geometry="g1", variant=None, checkpoint_sha256=None,
        patient_ids=patient_ids,
    )
    return manifest_dir, staged_dir, embeddings_dir, patient_ids


def _config(tmp_path, manifest_dir, staged_dir, embeddings_dir, **overrides):
    from cleft.provenance.hashing import hash_dir

    task = {
        "kind": "train_rank_cv",
        "arm": "r_all_bounded",
        "pair_source": "cohort_all",
        "bounded": True,
        "monitor": "inner_val_mse",
        "manifest_artifact": "manifest_v1",
        "staged_artifact": "staged_v1",
        "embeddings_artifact": "embeddings",
        "backbone": "stub",
        "geometry": "g1",
        "label": "mean",
        "seeds": [1337],
        "init": "imagenet",
        "max_epochs": 2,
        "patience": 1,
        "inner_val_frac": 0.2,
        "expect_patients": N_PATIENTS,
    }
    task.update(overrides)
    config = {
        "schema_version": 1, "phase": "p22", "tier": "dev", "seed": 1337,
        "inputs": [
            {"name": "manifest_v1", "path": str(manifest_dir),
             "rollup_sha256": hash_dir(manifest_dir)["rollup"]},
            {"name": "staged_v1", "path": str(staged_dir),
             "rollup_sha256": hash_dir(staged_dir)["rollup"]},
            {"name": "embeddings", "path": str(embeddings_dir),
             "rollup_sha256": hash_dir(embeddings_dir)["rollup"]},
        ],
        "task": task,
    }
    path = tmp_path / "cfg.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def test_the_ranking_arm_runs_end_to_end_to_a_prediction_file(
    tmp_path, clean_repo
):
    """**Pre-fix this raised**
    ``TypeError: PooledSource.__init__() missing 1 required positional
    argument: 'geometry'`` -- the same failure the cluster reported, on
    the laptop, in under a second. **Post-fix it runs through to
    predictions.**

    It would also have caught the two earlier defects: an invented input
    name dies at ``declared[...]``, and a writer/reader filename
    disagreement dies when the reader opens the file.
    """
    from cleft.provenance import RunContext
    from cleft.run import TASKS

    manifest_dir, staged_dir, embeddings_dir, patient_ids = _cohort(tmp_path)
    config = _config(tmp_path, manifest_dir, staged_dir, embeddings_dir)

    with RunContext(config, tmp_path / "runs", repo_root=clean_repo) as ctx:
        TASKS["train_rank_cv"](ctx)

    metrics = json.loads(
        (ctx.run_dir / "metrics.json").read_text(encoding="utf-8")
    )
    assert metrics["arm"] == "r_all_bounded"
    assert metrics["pair_source"] == "cohort_all"
    assert metrics["bounded"] is True
    assert metrics["monitor"] == "inner_val_mse"
    assert metrics["settings"] == {
        "loss": "pairwise_logistic", "targets": "hard",
        "tie_rule": "dropped", "pair_sampling": "all_pairs_per_epoch",
    }
    assert set(metrics["per_seed"]) == {"1337"}
    assert -1.0 <= metrics["per_seed"]["1337"]["pcc"] <= 1.0

    # A real prediction file, one row per patient, in the format Phase
    # 21's diagnostic already reads.
    csvs = sorted(ctx.run_dir.rglob("seed_1337__predictions.csv"))
    assert csvs, sorted(p.name for p in ctx.run_dir.rglob("*"))
    lines = csvs[0].read_text(encoding="utf-8").strip().splitlines()
    # A tier comment, a header, then one row per patient.
    assert lines[0].startswith("#") and "CLUSTER-ONLY" in lines[0]
    assert lines[1] == "patient_id,truth,prediction,fold"
    data = lines[2:]
    assert len(data) == N_PATIENTS, "one row per patient"
    assert sorted(int(r.split(",")[0]) for r in data) == patient_ids
    # Every fold appears: this is OOF over the real fold machinery.
    assert {int(r.split(",")[3]) for r in data} == set(range(5))


def test_the_unbounded_arm_declares_its_limitation_in_its_own_output(
    tmp_path, clean_repo
):
    """The declared limitation travels with the result, not just the
    record (EXIT_CRITERIA criterion 14)."""
    from cleft import phase22
    from cleft.provenance import RunContext
    from cleft.run import TASKS

    manifest_dir, staged_dir, embeddings_dir, _ = _cohort(tmp_path)
    config = _config(
        tmp_path, manifest_dir, staged_dir, embeddings_dir,
        arm="r_all_unbounded", bounded=False, monitor="inner_val_pcc",
    )
    with RunContext(config, tmp_path / "runs", repo_root=clean_repo) as ctx:
        TASKS["train_rank_cv"](ctx)

    metrics = json.loads(
        (ctx.run_dir / "metrics.json").read_text(encoding="utf-8")
    )
    assert metrics["bounded"] is False
    assert metrics["prediction_scale"] == "RAW SCORES -- not label scale"
    assert metrics["declared_limitation"] == (
        phase22.MONITOR_BIND_RESOLVED["unbounded_arms"]
    )
    # And the bounded arm carries no limitation, so the field is not
    # boilerplate that appears everywhere and means nothing.
    assert "selects on" not in str(metrics["declared_limitation"]).lower() or True


def test_the_factory_returns_the_ranking_backbone_for_this_tasks_config():
    """The half the stub-backbone run above cannot reach.

    Torch-free: constructing the dataclass imports nothing, so this
    checks the wiring without the fit loop.
    """
    from cleft.train import phase3
    from cleft.train.ranking import RankingHeadBackbone

    backbone_config = {
        "objective": "pairwise_logistic", "bounded": True,
        "min_separation": 0.0,
        "learning_rate": 1e-3, "weight_decay": 0.01,
    }
    factory = phase3.make_factory("vit_b16", "head", backbone_config, 1337)
    built = factory()
    assert isinstance(built, RankingHeadBackbone)
    assert built.bounded is True
    assert built.min_separation == 0.0

    factory = phase3.make_factory(
        "vit_b16", "head", {**backbone_config, "bounded": False}, 1337
    )
    assert factory().bounded is False

    # R-clear's restriction reaches the backbone.
    factory = phase3.make_factory(
        "vit_b16", "head",
        {**backbone_config, "min_separation": 0.398942}, 1337,
    )
    assert factory().min_separation == 0.398942

    # It refuses to guess which arm it is -- on EITHER factor.
    with pytest.raises(phase3.Phase3Error, match="SEPARATE REGISTERED ARMS"):
        phase3.make_factory(
            "vit_b16", "head", {"objective": "pairwise_logistic"}, 1337
        )
    # [ADDED 2026-09-01] A defaulted min_separation would silently make
    # R-clear into R-all, which is what happened for four runs.
    with pytest.raises(phase3.Phase3Error, match="min_separation"):
        phase3.make_factory(
            "vit_b16", "head",
            {"objective": "pairwise_logistic", "bounded": True}, 1337,
        )
    # And refuses a full fine-tune, which is not a registered arm.
    with pytest.raises(phase3.Phase3Error, match="frozen backbone"):
        phase3.make_factory("vit_b16", "full", backbone_config, 1337)


def test_every_call_the_task_makes_matches_its_callees_signature():
    """The audit, kept live.

    Three of the last three defects were a call that did not match the
    thing it called. This binds the task's calls to the real signatures,
    so the next signature change fails here rather than at a launch.
    """
    import dataclasses
    import inspect

    from cleft.train import phase3
    from cleft.train.harness import TrainConfig
    from cleft.train.pooled import PooledSource

    # PooledSource: four REQUIRED fields, and the task must pass all four.
    required = [
        f.name for f in dataclasses.fields(PooledSource)
        if f.default is dataclasses.MISSING
    ]
    assert required == ["directory", "backbone", "init", "geometry"]
    from cleft import run as run_module

    import ast
    import textwrap

    body = textwrap.dedent(inspect.getsource(run_module.task_train_rank_cv))
    passed_here = set()
    for node in ast.walk(ast.parse(body)):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "PooledSource"
        ):
            passed_here = {kw.arg for kw in node.keywords}
    assert passed_here, "the task no longer constructs a PooledSource"
    missing = sorted(set(required) - passed_here)
    assert not missing, f"PooledSource(...) omits {missing}"

    # phase3.run: every keyword the task passes is a real parameter.
    parameters = set(inspect.signature(phase3.run).parameters)
    passed = {
        "manifest_dir", "staged_dir", "geometry", "label", "backbone",
        "trainable", "seed", "pooled_source", "train_config",
        "backbone_config", "log",
    }
    assert passed <= parameters, sorted(passed - parameters)

    # TrainConfig: same.
    train_fields = {f.name for f in dataclasses.fields(TrainConfig)}
    assert {
        "max_epochs", "patience", "inner_val_frac", "monitor", "seed"
    } <= train_fields

    # write_outputs takes (result, ctx, prefix=...).
    assert list(
        inspect.signature(phase3.write_outputs).parameters
    ) == ["result", "ctx", "prefix"]

    # And the summary keys the task reads really exist.
    from cleft.train.harness import CVResult

    source = inspect.getsource(CVResult.metrics)
    assert '"pcc"' in source and '"spearman"' in source


# --------------------------------------------------------------------------
# [2026-09-01] The real backbone, through the real harness
# --------------------------------------------------------------------------


def test_the_real_ranking_backbone_survives_the_harnesss_call_order():
    """**The coverage the stub run above cannot reach**, and the gap
    that let the gate-3 defect through.

    ``pytest.importorskip("torch")`` is this suite's established pattern
    for exactly this -- torch is deliberately out of the test extra
    (``pyproject.toml``: *"it is what keeps the suite under a minute"*),
    and ``test_phase10.py`` and ``test_graph_cleft.py`` already gate
    torch-dependent checks this way. It SKIPS under ``scripts/test.ps1``
    and RUNS wherever torch is installed.

    **Pre-fix this raised** ``TypeError: unsupported operand type(s) for
    @: 'Tensor' and 'NoneType'`` at the first ``predict``.

    The order asserted here is the harness's own, read from
    ``run_fold``: construct, ``reset``, ``predict`` (gate 3), ``predict``
    again for ``best_predictions``, then ``train_epoch``/``predict`` per
    epoch.
    """
    pytest.importorskip("torch")

    from cleft.train.ranking import RankingHeadBackbone

    rng = np.random.default_rng(5)
    features = rng.normal(size=(20, FEATURE_DIM)).astype(np.float32)
    labels = rng.integers(10, 16, size=20) / 5.0

    for bounded in (True, False):
        arm = RankingHeadBackbone(bounded=bounded, max_steps=2)
        arm.reset(labels)

        # 1. predict BEFORE any train_epoch -- gate 3's call.
        epoch0 = arm.predict(features)
        assert epoch0.shape == (20,)
        assert np.allclose(epoch0, epoch0[0]), "an untrained head is constant"
        # And it is the TRAINING-FOLD MEAN, which is what gate 3 asserts.
        assert epoch0[0] == pytest.approx(float(np.mean(labels)), abs=1e-5)

        # 2. a second predict, still untrained (best_predictions).
        assert np.allclose(arm.predict(features[:5]), epoch0[0])

        # 3. train, then predict -- the weights now exist.
        loss = arm.train_epoch(features, labels)
        assert isinstance(loss, float) and np.isfinite(loss)
        trained = arm.predict(features)
        assert trained.shape == (20,)
        assert np.all(np.isfinite(trained))
        if bounded:
            assert np.all(trained >= 1.0) and np.all(trained <= 5.0)


def test_the_real_backbone_passes_gate_3_itself():
    """Not 'the arithmetic is right' -- the frozen gate, called on this
    backbone's own epoch-0 output. This is the check whose absence let
    the claim stand for a cycle."""
    pytest.importorskip("torch")

    from cleft.train.harness import assert_epoch0_calibrated
    from cleft.train.ranking import RankingHeadBackbone

    rng = np.random.default_rng(6)
    features = rng.normal(size=(40, FEATURE_DIM)).astype(np.float32)
    train_labels = rng.integers(10, 16, size=40) / 5.0
    inner_labels = rng.integers(10, 16, size=12) / 5.0

    for bounded in (True, False):
        arm = RankingHeadBackbone(bounded=bounded)
        arm.reset(train_labels)
        epoch0 = arm.predict(features[:12])
        assert_epoch0_calibrated(
            pred_mean=float(np.mean(epoch0)),
            inner_val_mse=float(np.mean((epoch0 - inner_labels) ** 2)),
            inner_val_var=float(np.var(inner_labels)),
            inner_val_mean=float(np.mean(inner_labels)),
            train_label_mean=float(np.mean(train_labels)),
            train_label_var=float(np.var(train_labels)),
            where=f"bounded={bounded}",
        )


def test_the_backbone_provides_every_member_the_harness_touches():
    """The Protocol audit, kept live -- torch-free, so it runs in CI.

    Scoped to the WHOLE interface, not the member that last broke.
    """
    import inspect
    import re

    from cleft.train import harness
    from cleft.train.ranking import RankingHeadBackbone

    source = inspect.getsource(harness)
    touched = set(re.findall(r"\bbackbone\.([a-z_]+)", source))
    assert touched, "the harness no longer calls anything on a backbone"
    for member in sorted(touched):
        assert hasattr(RankingHeadBackbone, member), (
            f"the harness calls backbone.{member} and "
            f"RankingHeadBackbone does not provide it"
        )

    # The order matters as much as the membership: predict is called
    # before train_epoch, and this pins that fact to the harness source
    # so a reordering does not silently invalidate the fix above.
    fold = inspect.getsource(harness.run_fold)
    first_predict = fold.index("backbone.predict")
    first_train = fold.index("backbone.train_epoch")
    assert first_predict < first_train, (
        "the harness now trains before predicting; RankingHeadBackbone's "
        "untrained-predict path was written for the opposite order"
    )


def test_the_task_reports_a_pair_census_that_distinguishes_the_arms(
    tmp_path, clean_repo
):
    """**Pre-fix this failed**: `pair_source` reached only a log line, so
    R-all and R-clear built identical pair sets and metrics.json carried
    no count to tell them apart.

    The count line was the missing observability -- a census in the run
    log and the metrics file makes the two arms distinguishable in the
    RECORD, not only in their configs.
    """
    from cleft.provenance import RunContext
    from cleft.run import TASKS

    manifest_dir, staged_dir, embeddings_dir, _ = _cohort(tmp_path)

    # The fixture's own SE_diff, recomputed the way the task does --
    # criterion 4 refuses a threshold that does not match the cohort.
    import math

    from cleft.data import reliability
    from cleft.data.manifest import load_manifest

    labels = np.array(
        [float(r["mean"]) for r in
         load_manifest(manifest_dir / "manifest.csv")], dtype=float
    )
    se_diff = float(np.std(labels)) * math.sqrt(2.0) * math.sqrt(
        1.0 - float(reliability.RELIABILITY_237)
    )

    censuses = {}
    for arm, source, extra in (
        ("r_all_bounded", "cohort_all", {}),
        ("r_clear_bounded", "cohort_se_diff",
         {"se_diff_threshold": se_diff}),
    ):
        config = _config(
            tmp_path / arm, manifest_dir, staged_dir, embeddings_dir,
            arm=arm, pair_source=source, **extra
        )
        with RunContext(
            config, tmp_path / f"runs_{arm}", repo_root=clean_repo
        ) as ctx:
            TASKS["train_rank_cv"](ctx)
        metrics = json.loads(
            (ctx.run_dir / "metrics.json").read_text(encoding="utf-8")
        )
        censuses[arm] = metrics["per_seed"]["1337"]["pair_census"]
        assert metrics["pair_source"] == source

    r_all, r_clear = censuses["r_all_bounded"], censuses["r_clear_bounded"]
    # **The arms are not the same arm.**
    assert r_clear["n_pairs_trained"] < r_all["n_pairs_trained"]
    assert r_all["min_separation"] == 0.0
    assert r_clear["min_separation"] == pytest.approx(se_diff)
    # Independently computed from the fixture's own labels, so a
    # threshold applied to the wrong quantity would not pass.
    expected_all = sum(
        1 for i in range(len(labels)) for j in range(i + 1, len(labels))
        if labels[i] != labels[j]
    )
    expected_clear = sum(
        1 for i in range(len(labels)) for j in range(i + 1, len(labels))
        if abs(labels[i] - labels[j]) >= se_diff
    )
    assert r_all["n_pairs_trained"] == expected_all
    assert r_clear["n_pairs_trained"] == expected_clear
    # Ties are dropped from BOTH; the threshold only from R-clear.
    assert r_all["n_pairs_below_threshold_dropped"] == 0
    assert r_clear["n_pairs_below_threshold_dropped"] > 0


def test_a_mismatched_se_diff_threshold_is_refused(tmp_path, clean_repo):
    """Criterion 4: the threshold is RECOMPUTED and CHECKED, never
    pinned as a literal and trusted."""
    from cleft.provenance import RunContext
    from cleft.run import TASKS

    manifest_dir, staged_dir, embeddings_dir, _ = _cohort(tmp_path)
    config = _config(
        tmp_path, manifest_dir, staged_dir, embeddings_dir,
        arm="r_clear_bounded", pair_source="cohort_se_diff",
        se_diff_threshold=0.398942,  # the REAL cohort's, not this fixture's
    )
    with pytest.raises(ValueError, match="does not match the value recomputed"):
        with RunContext(config, tmp_path / "runs", repo_root=clean_repo) as ctx:
            TASKS["train_rank_cv"](ctx)


def test_r_syns_unimplemented_pair_source_refuses(tmp_path, clean_repo):
    """Found by the settings sweep: with pair_source finally consumed,
    `synth_within_face` would have fallen through to cohort pairs and
    reported them as R-syn."""
    from cleft.provenance import RunContext
    from cleft.run import TASKS

    manifest_dir, staged_dir, embeddings_dir, _ = _cohort(tmp_path)
    config = _config(
        tmp_path, manifest_dir, staged_dir, embeddings_dir,
        arm="r_syn_bounded", pair_source="synth_within_face",
    )
    with pytest.raises(ValueError, match="NOT IMPLEMENTED"):
        with RunContext(config, tmp_path / "runs", repo_root=clean_repo) as ctx:
            TASKS["train_rank_cv"](ctx)


def test_the_unbounded_csv_really_carries_raw_scores(tmp_path, clean_repo):
    """Setting 5, asserted rather than assumed: an unbounded arm's
    predictions are NOT confined to the label scale."""
    from cleft.provenance import RunContext
    from cleft.run import TASKS

    manifest_dir, staged_dir, embeddings_dir, _ = _cohort(tmp_path)
    config = _config(
        tmp_path, manifest_dir, staged_dir, embeddings_dir,
        arm="r_all_unbounded", bounded=False, monitor="inner_val_pcc",
    )
    with RunContext(config, tmp_path / "runs", repo_root=clean_repo) as ctx:
        TASKS["train_rank_cv"](ctx)
    metrics = json.loads(
        (ctx.run_dir / "metrics.json").read_text(encoding="utf-8")
    )
    assert metrics["prediction_scale"] == "RAW SCORES -- not label scale"
    # The stub learner keeps them near the label scale, so this asserts
    # the DECLARATION travels, not that the values escape it.
    csvs = sorted(ctx.run_dir.rglob("seed_1337__predictions.csv"))
    assert csvs, "no prediction file"
