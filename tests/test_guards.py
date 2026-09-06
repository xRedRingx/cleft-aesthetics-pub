"""The three mechanical guards.

These are not rules to remember, they are hard failures in code. Chain 1 of the
previous failure (CLEFT_PIPELINE_PLAN_v1.md Part 1) was possible because a
dirty tree only warned. Every assertion here must stay a raise, never a log
line.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from cleft.provenance import GuardError, RunContext

FIXED_NOW = datetime(2026, 7, 26, 12, 0, 0, tzinfo=timezone.utc)


# --------------------------------------------------------------------------
# Guard 1 - dirty tree
# --------------------------------------------------------------------------


def test_dirty_tree_refused_from_keeper(write_config, out_root, dirty_repo):
    cfg = write_config(tier="keeper")
    with pytest.raises(GuardError, match="dirty"):
        RunContext(cfg, out_root, repo_root=dirty_repo)


def test_dirty_tree_allowed_in_dev_and_recorded(write_config, out_root, dirty_repo):
    cfg = write_config(tier="dev")
    with RunContext(cfg, out_root, repo_root=dirty_repo) as ctx:
        run_dir = ctx.run_dir
        assert ctx.tier == "dev"

    env = json.loads((run_dir / "env.json").read_text(encoding="utf-8"))
    assert env["git"]["dirty_tree"] is True
    # A dev run gets no worktree, and the reason is recorded rather than implied.
    assert env["git"]["worktree_created"] is False
    assert env["git"]["worktree_skipped_reason"]
    assert not (run_dir / "code").exists()


def test_modified_tracked_file_is_also_dirty(write_config, out_root, clean_repo):
    from fixtures import builders

    builders.dirty(clean_repo, "modified")
    cfg = write_config(tier="keeper")
    with pytest.raises(GuardError, match="dirty"):
        RunContext(cfg, out_root, repo_root=clean_repo)


def test_clean_tree_keeper_succeeds(write_config, out_root, clean_repo):
    cfg = write_config(tier="keeper")
    with RunContext(cfg, out_root, repo_root=clean_repo) as ctx:
        assert ctx.tier == "keeper"
        assert ctx.run_dir.parent.parent.name == "keeper"
        env = ctx.env

    assert env["git"]["dirty_tree"] is False
    assert env["git"]["worktree_created"] is True
    assert (ctx.run_dir / "code").is_dir()


def test_keeper_tier_cannot_be_smuggled_past_the_config(write_config, out_root, clean_repo):
    """An explicit tier argument that disagrees with the config is a hard error.

    Otherwise the config stops being the provenance record, which is the whole
    point of having one.
    """
    cfg = write_config(tier="dev")
    with pytest.raises(GuardError, match="tier"):
        RunContext(cfg, out_root, repo_root=clean_repo, tier="keeper")


# --------------------------------------------------------------------------
# Guard 2 - no clobber
# --------------------------------------------------------------------------


def test_existing_run_directory_aborts(write_config, out_root, clean_repo):
    cfg = write_config(tier="keeper")
    with RunContext(cfg, out_root, repo_root=clean_repo, now=FIXED_NOW) as ctx:
        first = ctx.run_dir

    assert first.exists()
    with pytest.raises(GuardError, match="exists"):
        RunContext(cfg, out_root, repo_root=clean_repo, now=FIXED_NOW)


def test_clobber_guard_fires_even_if_directory_is_empty(write_config, out_root, clean_repo):
    """An empty directory still means something else claimed that run id.

    Guards against the check being written as ``exists() and any(iterdir())``,
    which would silently adopt a half-built directory.
    """
    cfg = write_config(tier="keeper")
    planned = RunContext.plan_run_dir(cfg, out_root, repo_root=clean_repo, now=FIXED_NOW)
    planned.mkdir(parents=True)
    assert not any(planned.iterdir())

    with pytest.raises(GuardError, match="exists"):
        RunContext(cfg, out_root, repo_root=clean_repo, now=FIXED_NOW)


def test_two_runs_produce_two_directories(write_config, out_root, clean_repo):
    cfg = write_config(tier="keeper")
    seen = []
    for minute in (0, 1):
        now = FIXED_NOW.replace(minute=minute)
        with RunContext(cfg, out_root, repo_root=clean_repo, now=now) as ctx:
            seen.append(ctx.run_dir)

    assert seen[0] != seen[1]
    assert all(p.exists() for p in seen)


# --------------------------------------------------------------------------
# Guard 3 - input hash mismatch
# --------------------------------------------------------------------------


def test_declared_input_hash_mismatch_aborts(write_config, out_root, clean_repo, data_dir):
    cfg = write_config(
        tier="keeper",
        inputs=[
            {
                "name": "artifact",
                "path": str(data_dir),
                "rollup_sha256": "0" * 64,
            }
        ],
    )
    with pytest.raises(GuardError, match="hash"):
        RunContext(cfg, out_root, repo_root=clean_repo)


def test_matching_input_hash_is_accepted_and_recorded(
    write_config, out_root, clean_repo, data_dir
):
    from cleft.provenance import hash_dir

    actual = hash_dir(data_dir)
    cfg = write_config(
        tier="keeper",
        inputs=[
            {"name": "artifact", "path": str(data_dir), "rollup_sha256": actual["rollup"]}
        ],
    )
    with RunContext(cfg, out_root, repo_root=clean_repo) as ctx:
        run_dir = ctx.run_dir

    inputs = json.loads((run_dir / "inputs.json").read_text(encoding="utf-8"))
    entry = inputs["inputs"][0]
    assert entry["name"] == "artifact"
    assert entry["rollup"] == actual["rollup"]
    assert entry["file_count"] == actual["file_count"]
    assert entry["total_bytes"] == actual["total_bytes"]


def test_corrupting_one_byte_of_an_input_stops_the_next_run(
    write_config, out_root, clean_repo, data_dir
):
    """Exit criterion 5. One byte is enough."""
    from cleft.provenance import hash_dir

    declared = hash_dir(data_dir)["rollup"]
    cfg = write_config(
        tier="keeper",
        inputs=[{"name": "artifact", "path": str(data_dir), "rollup_sha256": declared}],
    )
    RunContext(cfg, out_root, repo_root=clean_repo, now=FIXED_NOW).finalize()

    victim = data_dir / "item_00.bin"
    payload = bytearray(victim.read_bytes())
    payload[0] ^= 0x01
    victim.write_bytes(bytes(payload))

    with pytest.raises(GuardError, match="hash"):
        RunContext(cfg, out_root, repo_root=clean_repo, now=FIXED_NOW.replace(minute=5))


def test_missing_input_path_aborts(write_config, out_root, clean_repo, tmp_path):
    cfg = write_config(
        tier="keeper",
        inputs=[
            {
                "name": "absent",
                "path": str(tmp_path / "does_not_exist"),
                "rollup_sha256": "0" * 64,
            }
        ],
    )
    with pytest.raises(GuardError, match="not exist|missing"):
        RunContext(cfg, out_root, repo_root=clean_repo)
