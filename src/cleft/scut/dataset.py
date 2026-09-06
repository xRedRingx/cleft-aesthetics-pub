"""The SCUT-FBP5500 inventory: what is in it, what is excluded, and copy parity.

**SCUT is public and non-clinical, and it is on the laptop.** Every previous
phase was split laptop/cluster because the clinical cohort cannot leave EHU;
Phase 5's data has no such constraint, so geometry can be developed and checked
against real faces with real landmarks rather than fixtures.

Two things this module settles before any geometry is built.

**CM152 is excluded from every variant.** Its landmark file is shipped empty
(4 bytes, count 0). If original-SCUT and masked-SCUT differed by that one image,
every masked-vs-original delta would carry a one-image difference in training
data -- negligible in effect, free to remove, and it deletes a "why do these have
different n?" question from the write-up.

**[MEASURED 2026-07-28] It falls on the TEST side of the official 60/40 split**
(`test.txt`, rating 1.816667), so excluding it does not touch the pretraining
*training* set at all: 3300 train unchanged, 2200 -> 2199 test, 5499 total.

**The cluster copy must be verified against the laptop copy** before any geometry
built here is trusted. Geometry verified against a different copy of the data
than the one that trains is the class of gap this project keeps finding -- and it
is cheap to close: count, total bytes, and hashes of a deterministic sample.

**[MEASURED 2026-07-30] Done, and it passed: the copies are byte-identical.**
Phase 5 exit criterion 9. See ``VERIFIED_COPIES`` for the figures and
``verify_against_record`` to re-derive them, and note two things about how it is
recorded:

* the pasted cluster figures were **recomputed on the laptop copy before being
  written down**. A hash nobody recomputed is exactly what PLAN §7 warns about,
  and the record would otherwise be a transcription of a chat message.
* **compare files-only byte sums on both sides.** ``du -sb`` counts the directory
  as well, and the first attempt differed by 135,168 bytes in *both* directories
  -- identical across two directories of unrelated content, which file bytes
  cannot do. ``BYTE_SUM_COMMANDS`` pins the commands that measure the same thing.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from ..provenance.hashing import hash_file
from .landmarks import EMPTY_LANDMARK_FILES, N_LANDMARKS

#: [MEASURED] The dataset as shipped.
N_IMAGES = 5500
N_TRAIN_OFFICIAL = 3300
N_TEST_OFFICIAL = 2200

#: Excluded from every variant. See the module docstring.
EXCLUDED = tuple(EMPTY_LANDMARK_FILES)

#: [MEASURED 2026-07-28] Where the exclusion lands.
CM152_DISPOSITION = {
    "stem": "CM152",
    "reason": "landmark file shipped empty by the dataset authors (4 bytes, count 0)",
    "official_split_side": "test",
    "rating": 1.816667,
    "train_after_exclusion": 3300,
    "test_after_exclusion": 2199,
    "total_after_exclusion": 5499,
    "note": (
        "It is on the TEST side, so excluding it leaves the pretraining training "
        "set untouched. Both variants exclude it regardless, so masked and "
        "original SCUT train on identical images."
    ),
}

IMAGE_DIR = "Images"
LANDMARK_DIR = "facial landmark"
SPLIT_DIR = "train_test_files/split_of_60%training and 40%testing"

#: Where the cluster copy lives. [MEASURED 2026-07-30] Confirmed present with
#: ``Images/``, ``facial landmark/``, ``All_Ratings.xlsx``, ``README.txt`` and
#: ``train_test_files/``.
CLUSTER_ROOT = "/home/user/codex/scut/SCUT-FBP5500_v2"
LAPTOP_ROOT = "C:/data/scut-fbp5500/SCUT-FBP5500_v2"

#: A landmark file's size when it carries the full point set: a little-endian
#: ``int32`` count, then N x (x, y) ``float32``. Derived from ``N_LANDMARKS``
#: rather than written as 692, so the arithmetic cross-check below cannot drift
#: from the format it is checking.
PTS_BYTES_FULL = 4 + N_LANDMARKS * 8

#: **[MEASURED 2026-07-30] Phase 5 exit criterion 9: the two copies are
#: byte-identical.** Recorded as constants so the criterion is re-derivable
#: rather than asserted, and verified on the laptop copy before being written
#: here -- the figures arrived by paste from the cluster, and a pasted hash that
#: nobody recomputed is the thing PLAN §7 warns about.
#:
#: MD5 rather than SHA-256 only because that is what was run on the cluster side.
#: ``provenance/`` is a FROZEN TREE, so the MD5 helper lives in this module: a new
#: file under ``provenance/`` would change the package rollup and invalidate every
#: result measured with it. The freeze working as intended, not an inconvenience.
VERIFIED_COPIES = {
    "measured": "2026-07-30",
    "criterion": "Phase 5 exit criterion 9 -- cluster and laptop copies agree",
    "cluster_root": CLUSTER_ROOT,
    "laptop_root": LAPTOP_ROOT,
    "verdict": "byte-identical",
    #: **[R2] What was verified is NOT what the declared rollup covers, and the
    #: two must not be conflated.** The verification compared ``Images/`` and
    #: ``facial landmark/`` -- 11,000 files, 172,020,158 bytes. The
    #: ``rollup_sha256`` in the SCUT configs is a **whole-tree** hash:
    #: **11,017 files, 193,973,184 bytes**, so it additionally covers
    #: ``All_Ratings.xlsx``, ``Images_Sources.xlsx``, ``README.txt``,
    #: ``train_test_files.zip`` and the 13 files under ``train_test_files/``.
    #:
    #: **Seventeen files are inside the hash and outside the verification.** The
    #: cluster listing named ``Images/``, ``facial landmark/``,
    #: ``All_Ratings.xlsx``, ``README.txt`` and ``train_test_files/`` -- it did not
    #: mention ``Images_Sources.xlsx`` or ``train_test_files.zip``. That listing
    #: may simply have been abbreviated; the point is that **the byte-identity
    #: result does not settle the rollup**, and saying "the copies are identical so
    #: one declared hash is legitimate" moves between two quantities.
    #:
    #: **It is already checkable and about to be checked.** Guard 3 compares the
    #: whole-tree rollup on every run, and ``scripts/declare_inputs.py`` reports
    #: MATCHES or DISAGREES -- which it could not do on the cluster until the path
    #: resolution was fixed. If it DISAGREES, the 17 files above are where to look
    #: before suspecting the images.
    "verification_scope": {
        "verified": {
            "what": [IMAGE_DIR, LANDMARK_DIR],
            "n_files": 11_000,
            "total_bytes": 172_020_158,
        },
        "declared_rollup_covers": {
            "what": "the whole SCUT root, recursively",
            "n_files": 11_017,
            "total_bytes": 193_973_184,
            "extra_files": [
                "All_Ratings.xlsx",
                "Images_Sources.xlsx",
                "README.txt",
                "train_test_files.zip",
                "train_test_files/ (13 files)",
            ],
        },
        "gap": (
            "17 files are inside the declared rollup and outside the byte-identity "
            "verification. Guard 3 settles it on the first cluster run; if the "
            "rollup DISAGREES, look at those 17 before suspecting the images."
        ),
    },
    "directories": {
        IMAGE_DIR: {"n_files": 5500, "total_bytes": 168_214_846},
        LANDMARK_DIR: {"n_files": 5500, "total_bytes": 3_805_312},
    },
    "md5": {
        "Images/AF1.jpg": "7c9d8077f6227a5d2af1aff630570fe2",
        "Images/CM152.jpg": "e3ac2742a338f6f6125bab7b26273edf",
        "facial landmark/AF1.pts": "24b263a507f868e112fcaa0ecfa1d415",
        "facial landmark/CM152.pts": "f1d3ff8443297732862df21dc4e57262",
    },
    #: **Two further independent confirmations that CM152.pts carries count 0**,
    #: after the ``od`` read and the parse failure -- four in total, and they fail
    #: in different ways, which is the point of having four.
    "cm152_confirmations": {
        "md5_is_four_zero_bytes": (
            "f1d3ff8443297732862df21dc4e57262 is the MD5 of b'\\x00\\x00\\x00"
            "\\x00'. Verified by recomputation, not assumed. So the file is "
            "exactly four zero bytes: a count field of 0 and nothing after it."
        ),
        "landmark_total_bytes_is_only_consistent_with_one_short_file": (
            "an 86-point .pts is 4 + 86*8 = 692 bytes, and "
            "5499 * 692 + 4 = 3,805,312 -- EXACTLY the recorded total. The "
            "directory's byte count is therefore only consistent with 5,499 "
            "full files and exactly one 4-byte file. This is a STRONGER "
            "statement than the MD5 in one direction and weaker in another: the "
            "MD5 pins that ONE file's contents, while the arithmetic pins that "
            "every OTHER file is a complete 86-point record and that there is "
            "no second truncated file hiding in the set."
        ),
    },
    #: **[R2] The first comparison showed a 135,168-byte difference in BOTH
    #: directories, and it was not file content.** ``du -sb`` counts the directory
    #: itself alongside its files; a files-only sum does not. The tell was that
    #: the discrepancy was IDENTICAL in two directories holding completely
    #: different data -- file content cannot do that. Harmless here, and recorded
    #: because it is another instance of comparing two different quantities, and
    #: because tooling should not invite it: see ``BYTE_SUM_COMMANDS``.
    "du_sb_hazard": (
        "du -sb counts the directory inode/blocks as well as the files, so it "
        "is NOT the same quantity as a files-only byte sum. The first "
        "cross-platform comparison differed by 135,168 bytes in both "
        "directories -- identical across two directories of unrelated content, "
        "which is impossible for file bytes and is the signature of a "
        "per-directory constant. Use a files-only sum on BOTH sides, which is "
        "like-for-like by construction."
    ),
}

#: The two commands that produce the SAME quantity as
#: ``fingerprint()['image_bytes']`` -- a files-only byte sum. Recorded together
#: so the cluster side and the laptop side cannot drift into measuring different
#: things, which is exactly what happened with ``du -sb``.
BYTE_SUM_COMMANDS = {
    "posix": "find <dir> -type f -printf '%s\\n' | awk '{n++; t+=$1} END {print n, t}'",
    "powershell": (
        "Get-ChildItem <dir> -File | Measure-Object -Sum Length"
    ),
    "do_not_use": "du -sb <dir>",
    "why": (
        "du -sb includes the directory itself; the other two do not. They are "
        "different quantities, and comparing one against the other reports a "
        "difference that is not in the data. See VERIFIED_COPIES['du_sb_hazard']."
    ),
}


class DatasetError(RuntimeError):
    """The SCUT copy is not as expected."""


def image_stems(root: str | Path, exclude: bool = True) -> list[str]:
    """Every image stem, sorted, with the excluded ones dropped."""
    directory = Path(root) / IMAGE_DIR
    if not directory.is_dir():
        raise DatasetError(f"{directory} does not exist")
    stems = sorted(path.stem for path in directory.glob("*.jpg"))
    if exclude:
        stems = [stem for stem in stems if stem not in EXCLUDED]
    return stems


def read_split(root: str | Path, side: str) -> list[str]:
    """The official 60/40 split, as image stems."""
    if side not in ("train", "test"):
        raise DatasetError(f"unknown split side {side!r}; expected train or test")
    path = Path(root) / SPLIT_DIR / f"{side}.txt"
    if not path.is_file():
        raise DatasetError(f"{path} does not exist")

    stems = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            stems.append(Path(line.split()[0]).stem)
    return stems


def split_after_exclusion(root: str | Path) -> dict:
    """Both sides with the exclusions applied, and the counts that result."""
    sides = {}
    for side in ("train", "test"):
        stems = [s for s in read_split(root, side) if s not in EXCLUDED]
        sides[side] = stems
    return {
        "train": sides["train"],
        "test": sides["test"],
        "n_train": len(sides["train"]),
        "n_test": len(sides["test"]),
        "n_total": len(sides["train"]) + len(sides["test"]),
        "excluded": list(EXCLUDED),
    }


# --------------------------------------------------------------------------
# copy parity
# --------------------------------------------------------------------------

#: How many files to hash when fingerprinting a copy. Deterministic and evenly
#: spaced through the sorted list, so two copies sample the SAME files -- a
#: random sample would compare different files and prove nothing.
SAMPLE_SIZE = 12


def sampled_stems(stems: list[str], sample_size: int = SAMPLE_SIZE) -> list[str]:
    """An evenly spaced, deterministic sample of a sorted stem list."""
    if not stems:
        return []
    if len(stems) <= sample_size:
        return list(stems)
    step = len(stems) / sample_size
    return [stems[int(index * step)] for index in range(sample_size)]


def fingerprint(root: str | Path, sample_size: int = SAMPLE_SIZE) -> dict:
    """A comparable description of one copy of SCUT. SHAREABLE -- public data.

    Count, total bytes and hashes of a deterministic sample, for images and
    landmarks alike. Two copies agreeing on all three is strong evidence they are
    the same data; a full hash of 5,500 images would be stronger and is not worth
    the minutes on every check.
    """
    root = Path(root)
    images = Path(root) / IMAGE_DIR
    marks = Path(root) / LANDMARK_DIR
    for directory in (images, marks):
        if not directory.is_dir():
            raise DatasetError(f"{directory} does not exist")

    # Every file, including the excluded one: this fingerprints the COPY, not
    # the training set. A copy missing CM152 entirely is a different copy.
    image_files = sorted(images.glob("*.jpg"))
    mark_files = sorted(marks.glob("*.pts"))

    stems = [path.stem for path in image_files]
    sample = set(sampled_stems(stems, sample_size))

    return {
        "n_images": len(image_files),
        "n_landmarks": len(mark_files),
        "image_bytes": sum(path.stat().st_size for path in image_files),
        "landmark_bytes": sum(path.stat().st_size for path in mark_files),
        "sample_size": sample_size,
        "sampled_image_sha256": {
            path.stem: hash_file(path) for path in image_files if path.stem in sample
        },
        "sampled_landmark_sha256": {
            path.stem: hash_file(path) for path in mark_files if path.stem in sample
        },
        "excluded": list(EXCLUDED),
    }


def md5_file(path: str | Path) -> str:
    """MD5 of a file's bytes.

    Here rather than in ``provenance/hashing.py`` because that package is a
    FROZEN TREE -- adding a function to it would change the package rollup and
    invalidate every result measured with it. MD5 is used for one purpose only:
    matching the digests the cluster-side check produced. **Nothing in the
    provenance chain uses it**; artifact rollups are SHA-256 and stay that way.
    """
    digest = hashlib.md5()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def directory_byte_sum(root: str | Path, directory: str) -> dict:
    """Files-only count and byte sum for one directory. **Not** ``du -sb``.

    The same quantity as ``find <dir> -type f -printf '%s\\n'`` summed, and
    deliberately not the same as ``du -sb``, which also counts the directory
    itself. See ``VERIFIED_COPIES['du_sb_hazard']``: a first comparison differed
    by 135,168 bytes in both directories at once, which file content cannot do.
    """
    target = Path(root) / directory
    if not target.is_dir():
        raise DatasetError(f"{target} does not exist")
    files = [path for path in target.iterdir() if path.is_file()]
    return {
        "directory": directory,
        "n_files": len(files),
        "total_bytes": sum(path.stat().st_size for path in files),
        "quantity": "files-only byte sum (NOT du -sb)",
    }


def verify_against_record(root: str | Path, record: dict = VERIFIED_COPIES) -> dict:
    """Re-derive Phase 5 exit criterion 9 from a copy on disk.

    **The criterion is a computation, not a remembered verdict.** Every figure in
    ``VERIFIED_COPIES`` is recomputed and each mismatch is named. Also re-checks
    the two arithmetic confirmations of ``CM152.pts``, because those are the parts
    a reader is most likely to take on trust:

    * the landmark total equals ``(n - 1) * PTS_BYTES_FULL + 4``, so exactly one
      short file exists and every other is a complete point set;
    * ``md5(b'\\x00' * 4)`` is the recorded ``CM152.pts`` digest.
    """
    root = Path(root)
    differences: list[str] = []

    directories = {}
    for directory, expected in record["directories"].items():
        try:
            actual = directory_byte_sum(root, directory)
        except DatasetError as exc:
            differences.append(str(exc))
            continue
        directories[directory] = actual
        for key in ("n_files", "total_bytes"):
            if actual[key] != expected[key]:
                differences.append(
                    f"{directory} {key}: expected {expected[key]}, "
                    f"got {actual[key]}"
                )

    md5s = {}
    for relative, expected in record["md5"].items():
        path = root / relative
        if not path.is_file():
            differences.append(f"{relative} is missing")
            continue
        md5s[relative] = md5_file(path)
        if md5s[relative] != expected:
            differences.append(
                f"{relative} md5: expected {expected}, got {md5s[relative]}"
            )

    landmarks_recorded = record["directories"][LANDMARK_DIR]
    n_files = landmarks_recorded["n_files"]
    implied = (n_files - 1) * PTS_BYTES_FULL + 4
    arithmetic_holds = implied == landmarks_recorded["total_bytes"]
    if not arithmetic_holds:
        differences.append(
            f"landmark byte arithmetic: {n_files - 1} full files of "
            f"{PTS_BYTES_FULL} plus one of 4 gives {implied}, but the record "
            f"says {landmarks_recorded['total_bytes']}"
        )

    empty_md5 = hashlib.md5(b"\x00" * 4).hexdigest()
    empty_recorded = record["md5"][f"{LANDMARK_DIR}/{EMPTY_LANDMARK_FILES[0]}.pts"]
    empty_holds = empty_md5 == empty_recorded
    if not empty_holds:
        differences.append(
            f"md5 of four zero bytes is {empty_md5}, but the record has "
            f"{empty_recorded} for the empty landmark file"
        )

    return {
        "root": str(root),
        "match": not differences,
        "n_differences": len(differences),
        "differences": differences,
        "directories": directories,
        "md5": md5s,
        "cm152_arithmetic_confirms_one_short_file": arithmetic_holds,
        "cm152_md5_is_four_zero_bytes": empty_holds,
        "note": (
            "Phase 5 exit criterion 9, RE-DERIVED rather than recalled. The "
            "byte sums are files-only and therefore comparable with "
            "`find -type f -printf '%s\\n'` on the cluster side; they are NOT "
            "du -sb, which counts the directory too and differed by 135,168 "
            "bytes in both directories at once on the first attempt."
        ),
    }


def compare_copies(local: dict, remote: dict) -> dict:
    """Do two fingerprints describe the same data? Every difference reported.

    **Not a boolean.** "They differ" is not actionable; "the cluster copy has one
    fewer landmark file and these three images hash differently" is.
    """
    differences: list[str] = []
    for key in ("n_images", "n_landmarks", "image_bytes", "landmark_bytes"):
        if local.get(key) != remote.get(key):
            differences.append(f"{key}: local {local.get(key)} vs {remote.get(key)}")

    for field in ("sampled_image_sha256", "sampled_landmark_sha256"):
        left, right = local.get(field, {}), remote.get(field, {})
        if set(left) != set(right):
            differences.append(
                f"{field}: sampled different files -- local {sorted(set(left) - set(right))[:5]}, "
                f"remote {sorted(set(right) - set(left))[:5]}"
            )
        for stem in sorted(set(left) & set(right)):
            if left[stem] != right[stem]:
                differences.append(f"{field}: {stem} differs")

    return {
        "match": not differences,
        "n_differences": len(differences),
        "differences": differences,
        "note": (
            "Geometry verified against a different copy of the data than the one "
            "that trains is the class of gap this project keeps finding. This is "
            "cheap; run it before trusting any artifact built from either copy."
        ),
    }
