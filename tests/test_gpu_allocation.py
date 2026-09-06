"""Fractional GPU allocation, recorded alongside the GPU name.

Measured on the cluster 2026-07-27: ``RUNAI_NUM_OF_GPUS=0.11``. Run:AI hands out
slices of a device, so two runs that both report "RTX PRO 6000" can have wildly
different memory ceilings and throughput. Recording the device name alone would
make them look like the same hardware budget.
"""

from __future__ import annotations

import json

import pytest

from cleft.provenance import RunContext
from cleft.provenance.context import _gpu_allocation


@pytest.fixture(autouse=True)
def clear_allocation(monkeypatch):
    monkeypatch.delenv("RUNAI_NUM_OF_GPUS", raising=False)


def test_the_allocation_is_not_cached_across_runs(monkeypatch):
    """It comes from an environment variable a different workload sets differently.

    The torch half of the CUDA block IS cached -- the hardware cannot change
    mid-process -- but caching the allocation with it made every run after the
    first report the first one's fraction. Caught by
    ``test_allocation_lands_in_env_json`` when the cache was first added.
    """
    from cleft.provenance.context import _cuda_info

    monkeypatch.setenv("RUNAI_NUM_OF_GPUS", "0.11")
    first = _cuda_info()["runai_num_of_gpus"]
    monkeypatch.setenv("RUNAI_NUM_OF_GPUS", "1")
    second = _cuda_info()["runai_num_of_gpus"]

    assert (first, second) == ("0.11", "1")


def test_absent_variable_records_nothing_rather_than_guessing():
    allocation = _gpu_allocation()
    assert allocation["runai_num_of_gpus"] is None
    assert allocation["gpus_requested"] is None
    assert allocation["fractional_gpu"] is False


def test_the_measured_cluster_value_is_parsed(monkeypatch):
    monkeypatch.setenv("RUNAI_NUM_OF_GPUS", "0.11")
    allocation = _gpu_allocation()
    assert allocation["runai_num_of_gpus"] == "0.11"
    assert allocation["gpus_requested"] == pytest.approx(0.11)
    assert allocation["fractional_gpu"] is True


def test_a_whole_device_is_not_fractional(monkeypatch):
    monkeypatch.setenv("RUNAI_NUM_OF_GPUS", "1")
    assert _gpu_allocation()["fractional_gpu"] is False


def test_multiple_devices_are_not_fractional(monkeypatch):
    monkeypatch.setenv("RUNAI_NUM_OF_GPUS", "4")
    allocation = _gpu_allocation()
    assert allocation["gpus_requested"] == pytest.approx(4.0)
    assert allocation["fractional_gpu"] is False


def test_unparseable_value_is_kept_raw_rather_than_dropped(monkeypatch):
    """Keep what the scheduler said even when it cannot be interpreted."""
    monkeypatch.setenv("RUNAI_NUM_OF_GPUS", "not-a-number")
    allocation = _gpu_allocation()
    assert allocation["runai_num_of_gpus"] == "not-a-number"
    assert allocation["gpus_requested"] is None
    assert allocation["fractional_gpu"] is False


def test_allocation_lands_in_env_json(write_config, out_root, clean_repo, monkeypatch):
    monkeypatch.setenv("RUNAI_NUM_OF_GPUS", "0.11")
    cfg = write_config(tier="keeper")

    with RunContext(cfg, out_root, repo_root=clean_repo) as ctx:
        env = json.loads((ctx.run_dir / "env.json").read_text(encoding="utf-8"))

    cuda = env["cuda"]
    assert cuda["runai_num_of_gpus"] == "0.11"
    assert cuda["gpus_requested"] == pytest.approx(0.11)
    assert cuda["fractional_gpu"] is True
    # Recorded beside the device identity, so the pair is readable together.
    assert "gpu_name" in cuda
    assert "gpu_total_memory_bytes" in cuda
