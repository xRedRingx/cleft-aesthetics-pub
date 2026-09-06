"""Reading the patient folder tree.

This module has never seen the real data, so its job is to fail informatively.
Each test below is a surprise it might meet on the first cluster run.
"""

from __future__ import annotations

import pytest

from cleft.data import folderscan as F


def tree(root, layout):
    for folder, names in layout.items():
        directory = root / str(folder)
        directory.mkdir(parents=True, exist_ok=True)
        for name in names:
            (directory / name).write_bytes(b"x")
    return root


def test_reads_a_normal_tree(tmp_path):
    root = tree(tmp_path / "photos", {1: ["1001.jpg", "1002.jpg"], 238: ["581.jpg"]})
    assert F.scan(root).folders == {1: [1001, 1002], 238: [581]}


def test_image_ids_come_from_the_filename_stem(tmp_path):
    root = tree(tmp_path / "photos", {7: ["0523 frontal.jpg", "0524 basal.JPG"]})
    assert F.scan(root).folders[7] == [523, 524]


def test_extensions_are_case_insensitive(tmp_path):
    root = tree(tmp_path / "photos", {2: ["10.JPG", "11.Jpeg"]})
    assert F.scan(root).folders[2] == [10, 11]


def test_a_filename_with_no_number_is_rejected(tmp_path):
    root = tree(tmp_path / "photos", {3: ["frontal.jpg", "12.jpg"]})
    with pytest.raises(F.FolderScanError, match="no number"):
        F.scan(root)


def test_an_ambiguous_filename_is_rejected_rather_than_guessed(tmp_path):
    root = tree(tmp_path / "photos", {4: ["523_v2.jpg", "524.jpg"]})
    with pytest.raises(F.FolderScanError, match="ambiguous"):
        F.scan(root)


def test_a_repeated_number_in_one_stem_is_not_ambiguous(tmp_path):
    """'523 of 523.jpg' names one id twice, which is unambiguous."""
    root = tree(tmp_path / "photos", {5: ["523 of 523.jpg", "524.jpg"]})
    assert F.scan(root).folders[5] == [523, 524]


def test_a_folder_with_three_images_is_rejected_with_an_inventory(tmp_path):
    root = tree(tmp_path / "photos", {6: ["1.jpg", "2.jpg", "3.jpg"]})
    with pytest.raises(F.FolderScanError) as excinfo:
        F.scan(root)
    message = str(excinfo.value)
    assert "folder 6" in message
    assert "inventory" in message.lower(), "the message must list what it found"


def test_an_empty_folder_is_present_not_missing(tmp_path):
    """CORRECTED 2026-07-27: folder 52 EXISTS and is EMPTY.

    The plan previously recorded it as absent. The consequence is the same --
    237 patients have photographs -- but the condition is different, and code
    that inferred absence from a gap in the numbering would be wrong about the
    actual tree. An empty directory is a fact to report, not a hole to detect.
    """
    root = tree(tmp_path / "photos", {1: ["1.jpg", "2.jpg"], 3: ["3.jpg", "4.jpg"]})
    (root / "52").mkdir()

    result = F.scan(root)
    assert sorted(result.folders) == [1, 3, 52]
    assert result.folders[52] == []
    assert result.empty == [52]
    assert sorted(result.with_images) == [1, 3]
    assert result.total_images == 4


def test_a_folder_holding_only_clutter_counts_as_empty(tmp_path):
    root = tree(tmp_path / "photos", {1: ["1.jpg", "2.jpg"]})
    (root / "52").mkdir()
    (root / "52" / "Thumbs.db").write_bytes(b"x")
    (root / "52" / "._Thumbs.db").write_bytes(b"\x00" * 82)

    result = F.scan(root)
    assert result.empty == [52]


# --------------------------------------------------------------------------
# the clutter rule -- by basename, because extensions cannot catch it
# --------------------------------------------------------------------------


def test_appledouble_sidecars_are_skipped(tmp_path):
    """``._1001.jpg`` is a 4,096-byte metadata file with a .jpg extension.

    Extension filtering admits it, and the stem contains the real image id, so a
    naive scan reports every image twice and then fails on a duplicate id.
    """
    root = tree(
        tmp_path / "photos",
        {1: ["1001.jpg", "._1001.jpg", "1002.jpg", "._1002.jpg"]},
    )
    result = F.scan(root)
    assert result.folders[1] == [1001, 1002]
    assert result.skipped_clutter == 2
    assert result.skipped_names["._*"] == 2


def test_thumbs_db_and_its_sidecar_are_skipped(tmp_path):
    root = tree(tmp_path / "photos", {1: ["1.jpg", "2.jpg", "Thumbs.db", "._Thumbs.db"]})
    result = F.scan(root)
    assert result.folders[1] == [1, 2]
    assert result.skipped_clutter == 2


def test_skipped_clutter_is_counted_not_dropped_silently(tmp_path):
    root = tree(
        tmp_path / "photos",
        {1: ["1.jpg", "._1.jpg", ".DS_Store"], 2: ["2.jpg", "._2.jpg"]},
    )
    result = F.scan(root)
    assert result.skipped_clutter == 3
    assert dict(result.skipped_names) == {"._*": 2, ".DS_Store": 1}


def test_the_real_census_shape(tmp_path):
    """952 files = 473 images + 473 sidecars + 3 Thumbs.db + 3 ._Thumbs.db."""
    from fixtures import cohort as C

    built = C.build_cohort(tmp_path / "cohort", write_images=True)
    result = F.scan(built.images_dir)

    assert len(result.folders) == C.N_DIRECTORIES == 238
    assert len(result.with_images) == C.N_FOLDERS_WITH_PHOTOS == 237
    assert result.empty == [C.EMPTY_FOLDER]
    assert result.total_images == C.N_IMAGES == 473
    # 473 image sidecars + 3 Thumbs.db + 3 ._Thumbs.db
    assert result.skipped_clutter == 473 + 3 + 3


def test_zip_files_beside_the_extracted_tree_are_ignored(tmp_path):
    """The archives sit next to the extracted directories in the real bch/ tree."""
    root = tree(tmp_path / "photos", {1: ["1.jpg", "2.jpg"]})
    (root / "238 Cropped Anonymised Photos.zip").write_bytes(b"PK")
    assert F.scan(root).folders == {1: [1, 2]}


def test_pointing_at_a_directory_of_archives_says_so(tmp_path):
    root = tmp_path / "photos"
    root.mkdir()
    (root / "photos.zip").write_bytes(b"PK")
    with pytest.raises(F.FolderScanError, match="no numbered patient folders"):
        F.scan(root)


def test_a_non_numeric_subdirectory_is_reported(tmp_path):
    root = tree(tmp_path / "photos", {1: ["1.jpg", "2.jpg"]})
    (root / "All 25 images as JPG").mkdir()
    with pytest.raises(F.FolderScanError, match="not numbered patient folders"):
        F.scan(root)


def test_an_unexpected_extension_is_reported_not_ignored(tmp_path):
    root = tree(tmp_path / "photos", {1: ["1.jpg", "2.jpg", "notes.txt"]})
    with pytest.raises(F.FolderScanError, match="unexpected extension"):
        F.scan(root)


def test_os_clutter_is_ignored(tmp_path):
    root = tree(tmp_path / "photos", {1: ["1.jpg", "2.jpg", "Thumbs.db"]})
    assert F.scan(root).folders == {1: [1, 2]}


def test_a_loose_file_at_the_root_is_reported(tmp_path):
    root = tree(tmp_path / "photos", {1: ["1.jpg", "2.jpg"]})
    (root / "readme.md").write_bytes(b"x")
    with pytest.raises(F.FolderScanError, match="loose file"):
        F.scan(root)


def test_folders_containing_spaces_in_filenames(tmp_path):
    """Every real path has spaces; filenames may too."""
    root = tree(tmp_path / "238 Cropped Anonymised Photos", {1: ["1001 frontal.jpg"]})
    assert F.scan(root).folders == {1: [1001]}


def test_a_missing_root_is_rejected(tmp_path):
    with pytest.raises(F.FolderScanError, match="not a directory"):
        F.scan(tmp_path / "nope")

