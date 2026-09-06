"""The run directory contract and the hashing that backs it.

CLEFT_PIPELINE_PLAN_v1.md 2.4. If a result cannot name the code, the
environment and the inputs that produced it, the result cannot be defended.
"""

from __future__ import annotations

import json
import re
import sys

import pytest

from cleft.provenance import RunContext, hash_dir, hash_file

# --------------------------------------------------------------------------
# run directory contract
# --------------------------------------------------------------------------

REQUIRED_MEMBERS = ["config.yaml", "env.json", "inputs.json", "log.txt", "code"]


def test_run_directory_contains_every_required_member(write_config, out_root, clean_repo):
    cfg = write_config(tier="keeper")
    with RunContext(cfg, out_root, repo_root=clean_repo) as ctx:
        run_dir = ctx.run_dir

    for member in REQUIRED_MEMBERS:
        assert (run_dir / member).exists(), f"missing {member}"


def test_config_copy_is_byte_identical(write_config, out_root, clean_repo):
    cfg = write_config(tier="keeper")
    with RunContext(cfg, out_root, repo_root=clean_repo) as ctx:
        copied = ctx.run_dir / "config.yaml"

    assert copied.read_bytes() == cfg.read_bytes()


def test_run_id_encodes_config_sha_and_timestamp(write_config, out_root, clean_repo):
    from fixtures import builders

    cfg = write_config("p0_smoke.yaml", tier="keeper")
    sha = builders.git(clean_repo, "rev-parse", "HEAD")
    with RunContext(cfg, out_root, repo_root=clean_repo) as ctx:
        stem, sha8, stamp = ctx.run_dir.name.split("__")

    assert stem == "p0_smoke"
    assert sha8 == sha[:8]
    # 20260726T123800123Z - millisecond precision, so that two legitimate runs
    # of the same config at the same SHA do not collide.
    assert stamp.endswith("Z")
    assert re.fullmatch(r"\d{8}T\d{9}Z", stamp), stamp


def test_run_directory_is_nested_by_tier_and_phase(write_config, out_root, clean_repo):
    cfg = write_config(tier="keeper", phase="p3")
    with RunContext(cfg, out_root, repo_root=clean_repo) as ctx:
        assert ctx.run_dir.parent.name == "p3"
        assert ctx.run_dir.parent.parent.name == "keeper"
        assert ctx.run_dir.parent.parent.parent == out_root


# --------------------------------------------------------------------------
# env.json
# --------------------------------------------------------------------------


def test_env_json_captures_the_full_environment(write_config, out_root, clean_repo):
    from fixtures import builders

    sha = builders.git(clean_repo, "rev-parse", "HEAD")
    cfg = write_config(tier="keeper")
    with RunContext(cfg, out_root, repo_root=clean_repo) as ctx:
        env = json.loads((ctx.run_dir / "env.json").read_text(encoding="utf-8"))

    assert env["git"]["sha"] == sha
    assert env["git"]["sha8"] == sha[:8]
    assert env["git"]["dirty_tree"] is False
    assert env["utc_timestamp"].endswith("Z")
    assert env["host"]["hostname"]
    assert env["host"]["platform"]
    assert env["tier"] == "keeper"
    assert env["phase"] == "p0"
    # Package versions are recorded, absent ones as null rather than omitted:
    # "the key was missing" and "the package was missing" must be different facts.
    for pkg in ("numpy", "scipy", "torch", "timm"):
        assert pkg in env["packages"]
    assert env["packages"]["numpy"]
    assert env["cuda"].keys() >= {"available", "version", "gpu_name", "gpu_count"}
    assert "digest" in env["image"] and "base_digest" in env["image"]


def test_env_json_records_the_actual_python_version_not_an_inferred_one(
    write_config, out_root, clean_repo
):
    """The container's python version is a measured fact, never read off a tag."""
    cfg = write_config(tier="keeper")
    with RunContext(cfg, out_root, repo_root=clean_repo) as ctx:
        env = json.loads((ctx.run_dir / "env.json").read_text(encoding="utf-8"))

    actual = "{}.{}.{}".format(*sys.version_info[:3])
    assert env["python"]["version"] == actual
    assert env["python"]["version_full"] == sys.version
    assert env["python"]["executable"] == sys.executable
    # The declared version from requirements.txt is carried alongside, with the
    # comparison already made so nobody has to make it by eye.
    assert env["python"]["declared"]
    assert isinstance(env["python"]["matches_declared"], bool)


def test_python_mismatch_is_fatal_inside_the_container(
    write_config, out_root, clean_repo, monkeypatch
):
    """On the laptop a mismatch is a recorded warning; in the image it aborts."""
    from cleft.provenance import GuardError

    monkeypatch.setenv("CLEFT_IN_CONTAINER", "1")
    monkeypatch.setattr("cleft.provenance.context.DECLARED_PYTHON", "3.0.0")
    cfg = write_config(tier="keeper")
    with pytest.raises(GuardError, match="python"):
        RunContext(cfg, out_root, repo_root=clean_repo)


def test_image_digest_is_read_from_the_environment(
    write_config, out_root, clean_repo, monkeypatch
):
    monkeypatch.setenv("CLEFT_IMAGE_DIGEST", "sha256:" + "ab" * 32)
    cfg = write_config(tier="keeper")
    with RunContext(cfg, out_root, repo_root=clean_repo) as ctx:
        env = json.loads((ctx.run_dir / "env.json").read_text(encoding="utf-8"))

    assert env["image"]["digest"] == "sha256:" + "ab" * 32


# --------------------------------------------------------------------------
# code/ worktree
# --------------------------------------------------------------------------


def test_code_worktree_is_at_the_captured_sha(write_config, out_root, clean_repo):
    from fixtures import builders

    sha = builders.git(clean_repo, "rev-parse", "HEAD")
    cfg = write_config(tier="keeper")
    with RunContext(cfg, out_root, repo_root=clean_repo) as ctx:
        code = ctx.run_dir / "code"

    assert code.is_dir()
    assert builders.git(code, "rev-parse", "HEAD") == sha
    assert (code / "src" / "thing.py").read_text(encoding="utf-8") == "VALUE = 1\n"


def test_worktree_survives_later_commits_on_the_branch(write_config, out_root, clean_repo):
    """The snapshot is of the SHA, not of the branch tip."""
    from fixtures import builders

    cfg = write_config(tier="keeper")
    with RunContext(cfg, out_root, repo_root=clean_repo) as ctx:
        code = ctx.run_dir / "code"

    (clean_repo / "src" / "thing.py").write_text("VALUE = 999\n", encoding="utf-8")
    builders.git(clean_repo, "add", "-A")
    builders.git(clean_repo, "commit", "-q", "-m", "later change")

    assert (code / "src" / "thing.py").read_text(encoding="utf-8") == "VALUE = 1\n"


# --------------------------------------------------------------------------
# output tiering
# --------------------------------------------------------------------------


def test_outputs_are_tier_marked_and_indexed(write_config, out_root, clean_repo):
    cfg = write_config(tier="keeper")
    with RunContext(cfg, out_root, repo_root=clean_repo) as ctx:
        ctx.path("metrics.json", tier="SHAREABLE").write_text("{}", encoding="utf-8")
        ctx.path("predictions.csv", tier="CLUSTER-ONLY").write_text("id\n", encoding="utf-8")
        run_dir = ctx.run_dir

    index = json.loads((run_dir / "outputs.json").read_text(encoding="utf-8"))
    tiers = {entry["name"]: entry["tier"] for entry in index["outputs"]}
    assert tiers == {"metrics.json": "SHAREABLE", "predictions.csv": "CLUSTER-ONLY"}


def test_unknown_output_tier_is_rejected(write_config, out_root, clean_repo):
    cfg = write_config(tier="keeper")
    with RunContext(cfg, out_root, repo_root=clean_repo) as ctx:
        with pytest.raises(ValueError, match="tier"):
            ctx.path("whatever.csv", tier="PUBLIC")


def test_patient_keyed_output_cannot_be_marked_shareable(write_config, out_root, clean_repo):
    """Data boundary: patient-keyed predictions never leave EHU infrastructure."""
    cfg = write_config(tier="keeper")
    with RunContext(cfg, out_root, repo_root=clean_repo) as ctx:
        with pytest.raises(ValueError, match="CLUSTER-ONLY"):
            ctx.path("predictions.csv", tier="SHAREABLE")


def test_log_is_written_to_the_run_directory(write_config, out_root, clean_repo):
    cfg = write_config(tier="keeper")
    with RunContext(cfg, out_root, repo_root=clean_repo) as ctx:
        ctx.log("a recorded line")
        run_dir = ctx.run_dir

    assert "a recorded line" in (run_dir / "log.txt").read_text(encoding="utf-8")


# --------------------------------------------------------------------------
# hashing
# --------------------------------------------------------------------------


def test_hashing_is_deterministic(data_dir):
    assert hash_dir(data_dir)["rollup"] == hash_dir(data_dir)["rollup"]


def test_hashing_is_content_sensitive(data_dir):
    before = hash_dir(data_dir)["rollup"]
    victim = data_dir / "item_01.bin"
    payload = bytearray(victim.read_bytes())
    payload[-1] ^= 0x01
    victim.write_bytes(bytes(payload))
    assert hash_dir(data_dir)["rollup"] != before


def test_hashing_is_order_independent(tmp_path):
    """Two directories with the same contents created in opposite order agree."""
    a = tmp_path / "a"
    b = tmp_path / "b"
    for root, order in ((a, ["x", "y", "z"]), (b, ["z", "y", "x"])):
        root.mkdir()
        for name in order:
            (root / f"{name}.txt").write_text(f"content {name}\n", encoding="utf-8")

    assert hash_dir(a)["rollup"] == hash_dir(b)["rollup"]


def test_renaming_a_file_changes_the_rollup(tmp_path):
    """Content-only hashing would call a mislabelled artifact identical."""
    root = tmp_path / "r"
    root.mkdir()
    (root / "one.txt").write_text("same\n", encoding="utf-8")
    before = hash_dir(root)["rollup"]
    (root / "one.txt").rename(root / "two.txt")
    assert hash_dir(root)["rollup"] != before


def test_rollup_carries_file_count_and_total_bytes(data_dir):
    """A truncated directory must be visible, not just differently hashed."""
    result = hash_dir(data_dir)
    assert result["file_count"] == 4
    assert result["total_bytes"] == sum(
        p.stat().st_size for p in data_dir.rglob("*") if p.is_file()
    )
    assert set(result["files"]) == {
        "item_00.bin",
        "item_01.bin",
        "item_02.bin",
        "nested/note.txt",
    }


def test_relative_paths_use_forward_slashes(data_dir):
    """The same artifact must hash identically on the laptop and on the cluster."""
    assert all("\\" not in rel for rel in hash_dir(data_dir)["files"])


def test_hash_file_matches_hashlib(tmp_path):
    import hashlib

    path = tmp_path / "f.bin"
    path.write_bytes(b"exactly these bytes")
    assert hash_file(path) == hashlib.sha256(b"exactly these bytes").hexdigest()


def test_a_second_claim_of_an_output_name_is_refused(write_config, out_root, clean_repo):
    """The masked-G1 SCUT animation crash (2026-08-16), as the API's own
    contract: ``ctx.path`` CLAIMS -- calling it again to LOAD what the run
    already wrote is refused. The pattern is claim once, keep the Path,
    and read own outputs through the variable or through ``ctx.run_dir``
    (which also reaches a resumed attempt's earlier files)."""
    cfg = write_config(tier="keeper")
    with RunContext(cfg, out_root, repo_root=clean_repo) as ctx:
        claimed = ctx.path("pretrained.npz", tier="SHAREABLE")
        claimed.write_bytes(b"weights")

        with pytest.raises(ValueError, match="already claimed"):
            ctx.path("pretrained.npz", tier="SHAREABLE")

        assert claimed.read_bytes() == b"weights"
        assert (ctx.run_dir / "pretrained.npz").read_bytes() == b"weights"
