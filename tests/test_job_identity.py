"""Run identity from the scheduler, and what the scheduler actually provides.

Measured on the cluster 2026-07-27: there is NO ``RUNAI_JOB_ID`` and no
``RUNAI_JOB_UUID``. What exists is ``JOB_UUID`` / ``jobUUID`` (a real UUID) and
``RUNAI_JOB_NAME`` / ``JOB_NAME`` / ``jobName`` (a reusable workload name).

These tests pin the lookup order rather than the presence of any one variable,
because the variable names were guessed wrong once already.
"""

from __future__ import annotations

import json

import pytest

from cleft.provenance import JOB_ID_ENV_VARS, NAME_ONLY_JOB_VARS, RunContext, job_token
from cleft.provenance.context import _sanitize_token

CLUSTER_UUID = "6fb7f0fa-1c4d-4f0e-9a3b-2f7e8d1c5a90"


@pytest.fixture(autouse=True)
def clear_job_env(monkeypatch):
    for name in JOB_ID_ENV_VARS:
        monkeypatch.delenv(name, raising=False)


def test_no_job_variables_falls_back_to_a_timestamp():
    token, source = job_token()
    assert token is None and source is None


def test_the_variables_the_cluster_actually_sets_are_looked_up():
    """Guards against reverting to the names that do not exist."""
    for name in ("JOB_UUID", "jobUUID", "RUNAI_JOB_NAME", "JOB_NAME", "jobName"):
        assert name in JOB_ID_ENV_VARS, f"{name} is set on the cluster and must be read"


def test_uuid_wins_over_name(monkeypatch):
    """A workload name is reusable; a UUID is not. The UUID must be preferred."""
    monkeypatch.setenv("jobName", "test3")
    monkeypatch.setenv("RUNAI_JOB_NAME", "test3")
    monkeypatch.setenv("JOB_UUID", CLUSTER_UUID)

    token, source = job_token()
    assert source == "JOB_UUID"
    assert token == CLUSTER_UUID


def test_camelcase_uuid_is_also_read(monkeypatch):
    """Environment variables are case-sensitive; jobUUID is a distinct name."""
    monkeypatch.setenv("jobUUID", CLUSTER_UUID)
    token, source = job_token()
    assert source == "jobUUID"
    assert token == CLUSTER_UUID


def test_explicit_override_beats_everything(monkeypatch):
    monkeypatch.setenv("JOB_UUID", CLUSTER_UUID)
    monkeypatch.setenv("CLEFT_JOB_ID", "chosen-by-the-workload-spec")
    token, source = job_token()
    assert source == "CLEFT_JOB_ID"
    assert token == "chosen-by-the-workload-spec"


def test_name_only_variables_are_flagged_as_such():
    assert NAME_ONLY_JOB_VARS <= set(JOB_ID_ENV_VARS)
    assert "JOB_UUID" not in NAME_ONLY_JOB_VARS
    assert "jobName" in NAME_ONLY_JOB_VARS


def test_a_name_derived_run_id_warns_that_names_are_not_unique(
    write_config, out_root, clean_repo, monkeypatch
):
    monkeypatch.setenv("jobName", "test3")
    cfg = write_config(tier="keeper")

    with RunContext(cfg, out_root, repo_root=clean_repo) as ctx:
        env = json.loads((ctx.run_dir / "env.json").read_text(encoding="utf-8"))

    warnings = " ".join(env["provenance_warnings"])
    assert "workload NAME" in warnings
    assert "CLEFT_JOB_ID" in warnings


def test_a_uuid_derived_run_id_does_not_warn(
    write_config, out_root, clean_repo, monkeypatch
):
    monkeypatch.setenv("JOB_UUID", CLUSTER_UUID)
    cfg = write_config(tier="keeper")

    with RunContext(cfg, out_root, repo_root=clean_repo) as ctx:
        env = json.loads((ctx.run_dir / "env.json").read_text(encoding="utf-8"))

    assert not [w for w in env["provenance_warnings"] if "workload NAME" in w]
    assert env["run_id_source"] == "JOB_UUID"
    assert env["job_id"] == CLUSTER_UUID


def test_run_id_contains_the_job_id_so_a_restart_lands_in_the_same_directory(
    write_config, out_root, clean_repo, monkeypatch
):
    monkeypatch.setenv("JOB_UUID", CLUSTER_UUID)
    cfg = write_config(tier="keeper")

    with RunContext(cfg, out_root, repo_root=clean_repo) as first:
        first_dir = first.run_dir

    # A Run:AI pause deletes and recreates the pod; the entrypoint re-runs from
    # the top with the same job id.
    with RunContext(cfg, out_root, repo_root=clean_repo) as second:
        assert second.run_dir == first_dir
        assert second.resumed is True
        assert second.attempt == 1


def test_job_identity_variables_are_recorded(
    write_config, out_root, clean_repo, monkeypatch
):
    """The names were guessed wrong once; record what was actually present."""
    monkeypatch.setenv("JOB_UUID", CLUSTER_UUID)
    monkeypatch.setenv("jobName", "test3")
    cfg = write_config(tier="keeper")

    with RunContext(cfg, out_root, repo_root=clean_repo) as ctx:
        env = json.loads((ctx.run_dir / "env.json").read_text(encoding="utf-8"))

    assert env["job_identity"] == {"JOB_UUID": CLUSTER_UUID, "jobName": "test3"}


def test_job_identity_is_an_allowlist_not_an_environment_sweep(
    write_config, out_root, clean_repo, monkeypatch
):
    """A blanket sweep of the environment would capture credentials."""
    monkeypatch.setenv("JOB_UUID", CLUSTER_UUID)
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "must-never-be-recorded")
    monkeypatch.setenv("RUNAI_SOMETHING_ELSE", "also-not-recorded")
    cfg = write_config(tier="keeper")

    with RunContext(cfg, out_root, repo_root=clean_repo) as ctx:
        text = (ctx.run_dir / "env.json").read_text(encoding="utf-8")

    assert "must-never-be-recorded" not in text
    assert "also-not-recorded" not in text


def test_tokens_are_sanitised_into_one_path_component():
    assert "/" not in _sanitize_token("a/b")
    assert "\\" not in _sanitize_token("a\\b")
    assert _sanitize_token(CLUSTER_UUID) == CLUSTER_UUID, "a UUID must survive intact"
    assert _sanitize_token("  ") == "unnamed-job"
    assert len(_sanitize_token("x" * 200)) <= 48
