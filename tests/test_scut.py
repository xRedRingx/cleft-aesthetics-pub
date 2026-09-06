"""SCUT-FBP5500 inventory and landmarks (Phase 5).

SCUT is public and on the laptop, so these run against synthetic ``.pts`` bytes
in the same format as the real files -- the format is exact, the content is not
real data. The checks that need the real copy are the parity fingerprint and the
landmark layout, and both are exercised here against constructed inputs.
"""

from __future__ import annotations

import struct

import numpy as np
import pytest

from cleft.scut import dataset, landmarks


def pts_bytes(points) -> bytes:
    array = np.asarray(points, dtype="<f4")
    return struct.pack("<i", len(array)) + array.tobytes()


def a_face(n: int = 86) -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.uniform(50, 300, size=(n, 2))


# --------------------------------------------------------------------------
# the .pts format
# --------------------------------------------------------------------------


def test_a_landmark_file_round_trips(tmp_path):
    points = a_face()
    path = tmp_path / "AF1.pts"
    path.write_bytes(pts_bytes(points))
    assert landmarks.load_pts(path) == pytest.approx(points, abs=1e-3)


def test_the_empty_file_is_a_property_of_the_data_not_a_failure(tmp_path):
    """CM152 is shipped as 4 bytes with count 0 by the dataset authors. The
    error says so, because "landmark detection failed" is the wrong conclusion
    and the one someone would otherwise reach."""
    path = tmp_path / "CM152.pts"
    path.write_bytes(struct.pack("<i", 0))

    assert landmarks.is_empty(path)
    with pytest.raises(landmarks.LandmarkError, match="shipped empty"):
        landmarks.load_pts(path)


def test_a_truncated_file_is_refused(tmp_path):
    path = tmp_path / "AF2.pts"
    path.write_bytes(struct.pack("<i", 86) + b"\x00" * 40)
    with pytest.raises(landmarks.LandmarkError, match="bytes"):
        landmarks.load_pts(path)


def test_an_unexpected_count_is_refused(tmp_path):
    path = tmp_path / "AF3.pts"
    path.write_bytes(pts_bytes(a_face(68)))
    with pytest.raises(landmarks.LandmarkError, match="expected 86"):
        landmarks.load_pts(path)


# --------------------------------------------------------------------------
# the layout, derived rather than assumed
# --------------------------------------------------------------------------


def test_the_groups_cover_every_index_exactly_once():
    """A gap or an overlap would put the trapezium on the wrong anatomy for
    every one of 5,499 faces, and nothing would raise."""
    covered = [i for group in landmarks.LANDMARK_GROUPS.values() for i in group]
    assert sorted(covered) == list(range(landmarks.N_LANDMARKS))


def test_the_span_the_cleft_crop_covers_is_brow_to_mouth():
    assert landmarks.BROW_INDICES == landmarks.LANDMARK_GROUPS["brows"]
    assert landmarks.MOUTH_INDICES == landmarks.LANDMARK_GROUPS["mouth"]


def test_the_mouth_detail_sits_inside_the_mouth_group():
    mouth = set(landmarks.LANDMARK_GROUPS["mouth"])
    assert set(landmarks.MOUTH_CORNERS) <= mouth
    assert set(landmarks.UPPER_LIP_OUTER) <= mouth
    assert set(landmarks.LOWER_LIP_OUTER) <= mouth
    assert not set(landmarks.UPPER_LIP_OUTER) & set(landmarks.LOWER_LIP_OUTER)


def test_bounds_describes_the_extent():
    points = np.array([[10.0, 20.0], [30.0, 5.0]])
    assert landmarks.bounds(points) == {
        "x_min": 10.0, "x_max": 30.0, "y_min": 5.0, "y_max": 20.0, "n": 2
    }


def test_bounds_refuses_a_wrong_shape():
    with pytest.raises(landmarks.LandmarkError, match=r"\(N, 2\)"):
        landmarks.bounds(np.zeros((5, 3)))


def test_annotate_marks_the_points_and_leaves_the_rest():
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    marked = landmarks.annotate(image, np.array([[50.0, 50.0]]))
    assert marked[50, 50].any()
    assert not marked[10, 10].any()


# --------------------------------------------------------------------------
# the inventory
# --------------------------------------------------------------------------


def test_cm152_is_excluded_from_every_variant():
    """Masked and original SCUT must train on identical images, or every delta
    carries a one-image difference in training data."""
    assert "CM152" in dataset.EXCLUDED


def test_the_cm152_disposition_is_recorded():
    """[MEASURED 2026-07-28] It falls on the TEST side, so excluding it leaves
    the pretraining training set untouched."""
    disposition = dataset.CM152_DISPOSITION
    assert disposition["official_split_side"] == "test"
    assert disposition["train_after_exclusion"] == dataset.N_TRAIN_OFFICIAL
    assert disposition["test_after_exclusion"] == dataset.N_TEST_OFFICIAL - 1
    assert disposition["total_after_exclusion"] == dataset.N_IMAGES - 1
    assert "shipped empty" in disposition["reason"]


def test_the_sample_is_deterministic_and_evenly_spaced():
    """Two copies must sample the SAME files. A random sample would compare
    different files and prove nothing."""
    stems = [f"IMG{i:04d}" for i in range(500)]
    first = dataset.sampled_stems(stems)
    assert first == dataset.sampled_stems(stems)
    assert len(first) == dataset.SAMPLE_SIZE
    assert first[0] == stems[0]
    assert len(set(first)) == len(first)


def test_a_short_list_is_sampled_whole():
    stems = ["A", "B", "C"]
    assert dataset.sampled_stems(stems) == stems


# --------------------------------------------------------------------------
# copy parity
# --------------------------------------------------------------------------


def base_fingerprint() -> dict:
    return {
        "n_images": 5500,
        "n_landmarks": 5500,
        "image_bytes": 123456789,
        "landmark_bytes": 3806000,
        "sampled_image_sha256": {"AF1": "aaa", "CM1": "bbb"},
        "sampled_landmark_sha256": {"AF1": "ccc", "CM1": "ddd"},
    }


def test_identical_copies_match():
    report = dataset.compare_copies(base_fingerprint(), base_fingerprint())
    assert report["match"] is True
    assert report["differences"] == []


def test_a_missing_file_is_reported_specifically():
    """"They differ" is not actionable; naming what differs is."""
    remote = base_fingerprint()
    remote["n_landmarks"] = 5499
    report = dataset.compare_copies(base_fingerprint(), remote)
    assert report["match"] is False
    assert any("n_landmarks" in d for d in report["differences"])


def test_a_changed_file_is_named():
    remote = base_fingerprint()
    remote["sampled_image_sha256"]["CM1"] = "different"
    report = dataset.compare_copies(base_fingerprint(), remote)
    assert any("CM1 differs" in d for d in report["differences"])


def test_copies_that_sampled_different_files_are_flagged():
    """Comparing hashes of different files would silently prove nothing."""
    remote = base_fingerprint()
    remote["sampled_image_sha256"] = {"AF2": "aaa", "CM1": "bbb"}
    report = dataset.compare_copies(base_fingerprint(), remote)
    assert any("sampled different files" in d for d in report["differences"])


def test_every_difference_is_reported_not_just_the_first():
    remote = base_fingerprint()
    remote["n_images"] = 5499
    remote["image_bytes"] = 1
    remote["sampled_image_sha256"]["AF1"] = "x"
    report = dataset.compare_copies(base_fingerprint(), remote)
    assert report["n_differences"] >= 3


def test_the_parity_note_says_why_it_matters():
    report = dataset.compare_copies(base_fingerprint(), base_fingerprint())
    assert "different copy of the data than the one that trains" in report["note"]


# --------------------------------------------------------------------------
# Phase 5 exit criterion 9: the copies are byte-identical
# --------------------------------------------------------------------------


def test_the_landmark_byte_total_admits_exactly_one_short_file():
    """**A fourth independent confirmation that CM152.pts carries count 0**, and
    it is arithmetic rather than a hash.

    An 86-point `.pts` is `4 + 86*8 = 692` bytes, and `5499*692 + 4` is
    **exactly** the recorded 3,805,312. So the directory's byte count is only
    consistent with 5,499 complete point sets and exactly one 4-byte file.

    Stronger than the MD5 in one direction and weaker in another: the MD5 pins
    that ONE file's contents, while this pins that every OTHER file is a complete
    record and that no second truncated file is hiding in the set. Computed from
    `N_LANDMARKS`, so it cannot drift from the format it checks.
    """
    recorded = dataset.VERIFIED_COPIES["directories"][dataset.LANDMARK_DIR]
    assert dataset.PTS_BYTES_FULL == 4 + landmarks.N_LANDMARKS * 8
    implied = (recorded["n_files"] - 1) * dataset.PTS_BYTES_FULL + 4
    assert implied == recorded["total_bytes"] == 3_805_312


def test_the_empty_landmark_md5_is_the_md5_of_four_zero_bytes():
    """**A third independent confirmation**, after the `od` read and the parse
    failure. Recomputed here rather than trusted: the digest arrived by paste."""
    import hashlib

    expected = hashlib.md5(b"\x00" * 4).hexdigest()
    key = f"{dataset.LANDMARK_DIR}/{landmarks.EMPTY_LANDMARK_FILES[0]}.pts"
    assert dataset.VERIFIED_COPIES["md5"][key] == expected
    assert expected == "f1d3ff8443297732862df21dc4e57262"


def test_the_recorded_copy_verification_is_self_consistent():
    """Both directories, both totals, four digests, and the verdict."""
    record = dataset.VERIFIED_COPIES
    assert record["verdict"] == "byte-identical"
    assert set(record["directories"]) == {dataset.IMAGE_DIR, dataset.LANDMARK_DIR}
    for entry in record["directories"].values():
        assert entry["n_files"] == dataset.N_IMAGES
    assert len(record["md5"]) == 4
    assert all(len(digest) == 32 for digest in record["md5"].values())
    assert record["cluster_root"] == "/home/user/codex/scut/SCUT-FBP5500_v2"


def test_the_du_sb_hazard_is_recorded_as_a_quantity_confusion():
    """**[R2] The first comparison differed by 135,168 bytes in BOTH
    directories** -- identical across two directories of unrelated content, which
    file bytes cannot do. `du -sb` counts the directory itself; a files-only sum
    does not. Recorded because the tooling should not invite the comparison."""
    hazard = dataset.VERIFIED_COPIES["du_sb_hazard"]
    assert "135,168" in hazard
    assert "NOT the same quantity" in hazard

    commands = dataset.BYTE_SUM_COMMANDS
    assert "-type f" in commands["posix"]
    assert commands["do_not_use"].startswith("du -sb")
    assert "different quantities" in commands["why"]


def test_the_byte_sum_is_files_only_and_says_so(tmp_path):
    """The quantity is named in the result, so a reader cannot mistake it for
    `du -sb` output pasted from a cluster shell."""
    (tmp_path / "d").mkdir()
    (tmp_path / "d" / "a.bin").write_bytes(b"x" * 10)
    (tmp_path / "d" / "b.bin").write_bytes(b"y" * 25)
    (tmp_path / "d" / "sub").mkdir()
    (tmp_path / "d" / "sub" / "c.bin").write_bytes(b"z" * 100)

    result = dataset.directory_byte_sum(tmp_path, "d")
    assert result["n_files"] == 2, "subdirectories are not descended into"
    assert result["total_bytes"] == 35, "and their contents are not counted"
    assert "NOT du -sb" in result["quantity"]


def test_a_missing_directory_is_refused_not_reported_as_zero_bytes(tmp_path):
    """A zero byte sum for an absent directory would compare equal to another
    absent directory and read as a match."""
    with pytest.raises(dataset.DatasetError, match="does not exist"):
        dataset.directory_byte_sum(tmp_path, "absent")


def test_verification_against_the_record_names_every_mismatch(tmp_path):
    """The criterion is a COMPUTATION, not a remembered verdict -- so it must be
    able to fail, and say what failed."""
    root = tmp_path / "copy"
    (root / dataset.IMAGE_DIR).mkdir(parents=True)
    (root / dataset.LANDMARK_DIR).mkdir(parents=True)
    (root / dataset.IMAGE_DIR / "AF1.jpg").write_bytes(b"not the real image")

    result = dataset.verify_against_record(root)
    assert result["match"] is False
    joined = " ".join(result["differences"])
    assert "n_files" in joined
    assert "md5" in joined or "missing" in joined
    # The two arithmetic confirmations do not depend on the copy, so they hold
    # even against a wrong directory -- they check the RECORD's consistency.
    assert result["cm152_arithmetic_confirms_one_short_file"] is True
    assert result["cm152_md5_is_four_zero_bytes"] is True


def test_verification_detects_a_corrupted_record(tmp_path):
    """If the recorded byte total and the recorded file count stop agreeing with
    the .pts format, the arithmetic confirmation must fail rather than be
    quietly skipped."""
    import copy

    record = copy.deepcopy(dataset.VERIFIED_COPIES)
    record["directories"][dataset.LANDMARK_DIR]["total_bytes"] += 1
    result = dataset.verify_against_record(tmp_path, record)
    assert result["cm152_arithmetic_confirms_one_short_file"] is False
    assert any("byte arithmetic" in d for d in result["differences"])


def test_md5_is_local_because_provenance_is_frozen():
    """MD5 lives here, not in provenance/hashing.py: that package is a frozen
    TREE, so a new file in it would change the rollup. And nothing in the
    provenance chain uses MD5 -- artifact rollups stay SHA-256."""
    assert dataset.md5_file.__module__ == "cleft.scut.dataset"
    assert "FROZEN TREE" in dataset.md5_file.__doc__
    assert "SHA-256 and stay that way" in dataset.md5_file.__doc__


def test_an_unknown_split_side_is_refused(tmp_path):
    with pytest.raises(dataset.DatasetError, match="unknown split side"):
        dataset.read_split(tmp_path, "validation")


def test_a_missing_image_directory_is_refused(tmp_path):
    with pytest.raises(dataset.DatasetError, match="does not exist"):
        dataset.image_stems(tmp_path)
