"""Run identity across a Run:AI restart, and atomic writes.

Run:AI pauses a workload by deleting and recreating the pod. The entrypoint then
re-runs from the top. With a timestamp-derived run id there is no collision, so
guard 2 never fires, the job silently restarts from epoch 0, discards its
checkpoint, and leaves two directories for one logical run. On a long pretraining
job that is hours lost with no error anywhere.

The fix is to derive the run id from the job id, which survives pod recreation.
These tests encode that, and they encode what a resume is NOT allowed to do.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from cleft.provenance import (
    JOB_ID_ENV_VARS,
    GuardError,
    RunContext,
    atomic_write_text,
    job_token,
)

FIXED_NOW = datetime(2026, 7, 26, 12, 0, 0, tzinfo=timezone.utc)


@pytest.fixture
def no_job_id(monkeypatch):
    """The laptop case: make sure no stray job variable leaks into a test."""
    for name in JOB_ID_ENV_VARS:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def with_job_id(monkeypatch):
    for name in JOB_ID_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("RUNAI_JOB_ID", "job-abc123")
    return "job-abc123"


# --------------------------------------------------------------------------
# where the run id comes from
# --------------------------------------------------------------------------


def test_run_id_uses_the_job_id_when_present(write_config, out_root, clean_repo, with_job_id):
    cfg = write_config("p0_smoke.yaml", tier="keeper")
    with RunContext(cfg, out_root, repo_root=clean_repo) as ctx:
        stem, sha8, suffix = ctx.run_dir.name.split("__")
        assert suffix == with_job_id
        assert ctx.run_id_source == "RUNAI_JOB_ID"
        assert ctx.job_id == with_job_id


def test_run_id_falls_back_to_a_timestamp_on_the_laptop(
    write_config, out_root, clean_repo, no_job_id
):
    cfg = write_config("p0_smoke.yaml", tier="keeper")
    with RunContext(cfg, out_root, repo_root=clean_repo, now=FIXED_NOW) as ctx:
        assert ctx.run_id.endswith("__20260726T120000000Z")
        assert ctx.run_id_source == "timestamp"
        assert ctx.job_id is None


def test_explicit_cleft_job_id_wins(write_config, out_root, clean_repo, monkeypatch):
    """The workload spec can set this and not depend on Run:AI's variable names."""
    monkeypatch.setenv("RUNAI_JOB_ID", "from-runai")
    monkeypatch.setenv("CLEFT_JOB_ID", "from-spec")
    cfg = write_config(tier="keeper")
    with RunContext(cfg, out_root, repo_root=clean_repo) as ctx:
        assert ctx.job_id == "from-spec"
        assert ctx.run_id_source == "CLEFT_JOB_ID"


def test_pod_level_variables_are_never_used(monkeypatch):
    """A pod name changes when the pod is recreated. That is the bug, not the fix."""
    for name in JOB_ID_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    for name in ("POD_NAME", "HOSTNAME", "RUNAI_POD_NAME", "RUNAI_POD_UUID"):
        monkeypatch.setenv(name, "pod-xyz-recreated")

    token, source = job_token()
    assert token is None and source is None
    assert not any("POD" in name for name in JOB_ID_ENV_VARS)


def test_job_id_is_sanitised_into_one_path_component(
    write_config, out_root, clean_repo, monkeypatch
):
    monkeypatch.setenv("CLEFT_JOB_ID", "proj/26698617:train #4")
    cfg = write_config(tier="keeper")
    with RunContext(cfg, out_root, repo_root=clean_repo) as ctx:
        suffix = ctx.run_dir.name.split("__")[2]

    assert "/" not in suffix and "\\" not in suffix and " " not in suffix
    assert ctx.run_dir.parent.name == "p0", "the id must not create extra directories"


# --------------------------------------------------------------------------
# resume
# --------------------------------------------------------------------------


def test_restarted_pod_lands_in_the_same_directory(
    write_config, out_root, clean_repo, with_job_id
):
    """The whole point: one logical run, one directory, however many pods."""
    cfg = write_config(tier="keeper")

    with RunContext(cfg, out_root, repo_root=clean_repo) as first:
        first_dir = first.run_dir
        assert first.resumed is False
        assert first.attempt == 0

    with RunContext(cfg, out_root, repo_root=clean_repo) as second:
        assert second.run_dir == first_dir
        assert second.resumed is True
        assert second.attempt == 1


def test_each_attempt_records_its_own_environment(
    write_config, out_root, clean_repo, with_job_id
):
    """A resumed pod can land on a different node with a different GPU."""
    cfg = write_config(tier="keeper")
    with RunContext(cfg, out_root, repo_root=clean_repo) as ctx:
        run_dir = ctx.run_dir
    for _ in range(2):
        with RunContext(cfg, out_root, repo_root=clean_repo):
            pass

    env = json.loads((run_dir / "env.json").read_text(encoding="utf-8"))
    assert env["attempt"] == 0, "env.json stays the first-attempt record"

    resumes = json.loads((run_dir / "resumes.json").read_text(encoding="utf-8"))["resumes"]
    assert [entry["attempt"] for entry in resumes] == [1, 2]
    assert all(entry["env"]["run_id"] == env["run_id"] for entry in resumes)


def test_resume_refuses_a_changed_config(write_config, out_root, clean_repo, with_job_id):
    """A changed config is a new run, not a continuation of this one."""
    cfg = write_config(tier="keeper")
    with RunContext(cfg, out_root, repo_root=clean_repo):
        pass

    cfg.write_text(
        cfg.read_text(encoding="utf-8").replace("seed: 1337", "seed: 7"), encoding="utf-8"
    )
    with pytest.raises(GuardError, match="differs from the config"):
        RunContext(cfg, out_root, repo_root=clean_repo)


def test_a_new_commit_is_a_new_run_not_a_resume(
    write_config, out_root, clean_repo, with_job_id
):
    """The SHA is part of the run id, so relaunching after a commit forks cleanly.

    Arms run at different code states are not comparable, so they must not share
    a directory even when Run:AI reuses the job id.
    """
    from fixtures import builders

    cfg = write_config(tier="keeper")
    with RunContext(cfg, out_root, repo_root=clean_repo) as first:
        first_dir = first.run_dir

    (clean_repo / "src" / "thing.py").write_text("VALUE = 2\n", encoding="utf-8")
    builders.git(clean_repo, "add", "-A")
    builders.git(clean_repo, "commit", "-q", "-m", "moved on")

    with RunContext(cfg, out_root, repo_root=clean_repo) as second:
        assert second.run_dir != first_dir
        assert second.resumed is False


def test_resume_refuses_a_directory_recorded_at_another_sha(
    write_config, out_root, clean_repo, with_job_id
):
    """Defence in depth for a force-push, or a directory from another repo.

    The run id carries only the first 8 characters of the SHA, so this check is
    what stops a directory whose full SHA disagrees from being adopted.
    """
    cfg = write_config(tier="keeper")
    with RunContext(cfg, out_root, repo_root=clean_repo) as ctx:
        env_path = ctx.run_dir / "env.json"

    env = json.loads(env_path.read_text(encoding="utf-8"))
    env["git"]["sha"] = "0" * 40
    env_path.write_text(json.dumps(env, indent=2, sort_keys=True), encoding="utf-8")

    with pytest.raises(GuardError, match="was created at SHA"):
        RunContext(cfg, out_root, repo_root=clean_repo)


def test_resume_refuses_a_half_built_directory(
    write_config, out_root, clean_repo, with_job_id
):
    """A directory with no env.json was not created by a completed guard pass."""
    cfg = write_config(tier="keeper")
    planned = RunContext.plan_run_dir(cfg, out_root, repo_root=clean_repo)
    planned.mkdir(parents=True)

    with pytest.raises(GuardError, match="no env.json"):
        RunContext(cfg, out_root, repo_root=clean_repo)


def test_timestamped_run_still_aborts_on_collision(
    write_config, out_root, clean_repo, no_job_id
):
    """Without a job id there is no resume, so guard 2 stays a hard abort."""
    cfg = write_config(tier="keeper")
    RunContext(cfg, out_root, repo_root=clean_repo, now=FIXED_NOW).finalize()

    with pytest.raises(GuardError, match="cannot be a Run:AI restart"):
        RunContext(cfg, out_root, repo_root=clean_repo, now=FIXED_NOW)


def test_outputs_from_earlier_attempts_survive_in_the_index(
    write_config, out_root, clean_repo, with_job_id
):
    cfg = write_config(tier="keeper")
    with RunContext(cfg, out_root, repo_root=clean_repo) as ctx:
        ctx.path("metrics.json", tier="SHAREABLE").write_text("{}", encoding="utf-8")
        run_dir = ctx.run_dir

    with RunContext(cfg, out_root, repo_root=clean_repo) as ctx:
        ctx.path("curves.csv", tier="SHAREABLE").write_text("epoch\n", encoding="utf-8")

    index = json.loads((run_dir / "outputs.json").read_text(encoding="utf-8"))
    by_name = {entry["name"]: entry for entry in index["outputs"]}
    assert set(by_name) == {"metrics.json", "curves.csv"}
    assert by_name["metrics.json"]["attempt"] == 0
    assert by_name["curves.csv"]["attempt"] == 1


def test_worktree_is_not_recreated_on_resume(write_config, out_root, clean_repo, with_job_id):
    cfg = write_config(tier="keeper")
    with RunContext(cfg, out_root, repo_root=clean_repo) as ctx:
        code = ctx.run_dir / "code"
    assert code.is_dir()

    with RunContext(cfg, out_root, repo_root=clean_repo) as ctx:
        assert ctx.env["git"]["worktree_created"] is True
    assert code.is_dir()


# --------------------------------------------------------------------------
# atomic writes
# --------------------------------------------------------------------------


def test_atomic_write_leaves_no_partial_file_on_failure(
    write_config, out_root, clean_repo, no_job_id
):
    """A pause mid-write must not leave something that still parses."""
    cfg = write_config(tier="keeper")
    with RunContext(cfg, out_root, repo_root=clean_repo) as ctx:
        with pytest.raises(RuntimeError, match="pod killed"):
            with ctx.atomic("checkpoint.json", tier="SHAREABLE") as tmp:
                tmp.write_text('{"epoch": 4', encoding="utf-8")  # truncated on purpose
                raise RuntimeError("pod killed mid-write")

        assert not (ctx.run_dir / "checkpoint.json").exists()
        leftovers = [p.name for p in ctx.run_dir.glob("*.tmp")]
        assert not leftovers, f"temporary files left behind: {leftovers}"


def test_atomic_write_replaces_the_previous_file_wholesale(
    write_config, out_root, clean_repo, no_job_id
):
    cfg = write_config(tier="keeper")
    with RunContext(cfg, out_root, repo_root=clean_repo) as ctx:
        target = ctx.run_dir / "curves.csv"
        target.write_text("stale content that is much longer\n", encoding="utf-8")
        with ctx.atomic("curves.csv", tier="SHAREABLE") as tmp:
            tmp.write_text("epoch\n", encoding="utf-8")

        assert target.read_text(encoding="utf-8") == "epoch\n"


def test_atomic_write_respects_the_patient_keyed_rule(
    write_config, out_root, clean_repo, no_job_id
):
    cfg = write_config(tier="keeper")
    with RunContext(cfg, out_root, repo_root=clean_repo) as ctx:
        with pytest.raises(ValueError, match="CLUSTER-ONLY"):
            with ctx.atomic("predictions.csv", tier="SHAREABLE"):
                pass


def test_atomic_write_text_helper_is_all_or_nothing(tmp_path):
    target = tmp_path / "thing.json"
    atomic_write_text(target, '{"a": 1}')
    assert target.read_text(encoding="utf-8") == '{"a": 1}'
    assert not list(tmp_path.glob("*.tmp"))


# --------------------------------------------------------------------------
# keeper outside the pinned image
# --------------------------------------------------------------------------


def test_keeper_outside_the_pinned_image_warns_loudly(
    write_config, out_root, clean_repo, no_job_id, monkeypatch, capsys
):
    """It happened on the very first cluster attempt. Loud, and in env.json."""
    monkeypatch.delenv("CLEFT_IN_CONTAINER", raising=False)
    cfg = write_config(tier="keeper")
    with RunContext(cfg, out_root, repo_root=clean_repo) as ctx:
        run_dir = ctx.run_dir

    assert "PROVENANCE WARNING" in capsys.readouterr().err

    env = json.loads((run_dir / "env.json").read_text(encoding="utf-8"))
    assert env["keeper_outside_pinned_image"] is True
    assert any("OUTSIDE the pinned image" in w for w in env["provenance_warnings"])
    assert "WARNING" in (run_dir / "log.txt").read_text(encoding="utf-8")


def test_keeper_outside_the_pinned_image_is_not_fatal(
    write_config, out_root, clean_repo, no_job_id, monkeypatch
):
    """CPU-only keeper work such as building the manifest is legitimate."""
    monkeypatch.delenv("CLEFT_IN_CONTAINER", raising=False)
    cfg = write_config(tier="keeper")
    with RunContext(cfg, out_root, repo_root=clean_repo) as ctx:
        assert ctx.tier == "keeper"
        assert ctx.run_dir.is_dir()


def test_dev_run_outside_the_pinned_image_does_not_warn(
    write_config, out_root, clean_repo, no_job_id, monkeypatch
):
    monkeypatch.delenv("CLEFT_IN_CONTAINER", raising=False)
    cfg = write_config(tier="dev")
    with RunContext(cfg, out_root, repo_root=clean_repo) as ctx:
        assert ctx.warnings == []
        assert ctx.env["keeper_outside_pinned_image"] is False
