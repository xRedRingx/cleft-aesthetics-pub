"""Content hashing for immutable data artifacts.

Chain 2 of the previous failure: data directories were rebuilt in place, so
"which mask produced this checkpoint?" had no answer in the artifacts. A rollup
hash makes lineage a fact in a file rather than a memory.

Rollup design:

* SHA-256 per file, over bytes.
* Relative paths normalised to forward slashes, so the same artifact hashes
  identically on the Windows laptop and on the Linux cluster.
* Entries sorted by relative path, then folded into one digest, so the rollup is
  order-independent but sensitive to both content and naming.
* File count and total bytes are recorded alongside, so a truncated directory is
  visible rather than merely different.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from .. import clutter

CHUNK = 1 << 20

#: The clutter rule lives in ``cleft.clutter`` and is shared with the folder
#: scanner, so a file counted by one and skipped by the other cannot produce a
#: hash that depends on which tool last walked the directory.
#:
#: This matters concretely: the cohort carries a 4,096-byte AppleDouble sidecar
#: beside every image, and Windows regenerates Thumbs.db whenever a directory is
#: browsed. Hashing either would make guard 3 abort on a rollup change that has
#: nothing to do with the data.


def hash_file(path: str | Path) -> str:
    """SHA-256 of a file's bytes."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_junk(rel: Path) -> bool:
    return clutter.is_clutter(rel.name) or any(
        clutter.is_clutter_dir(part) for part in rel.parts[:-1]
    )


def hash_dir(root: str | Path) -> dict:
    """Hash a directory tree.

    Returns a mapping with ``rollup``, ``file_count``, ``total_bytes``,
    ``files`` (relative path -> per-file digest) and ``excluded`` (the junk
    filter that was applied, recorded so the result is auditable).
    """
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"input path does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"input path is not a directory: {root}")

    files: dict[str, str] = {}
    total_bytes = 0
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(root)
        if _is_junk(rel):
            continue
        files[rel.as_posix()] = hash_file(path)
        total_bytes += path.stat().st_size

    digest = hashlib.sha256()
    for rel in sorted(files):
        digest.update(f"{rel}\0{files[rel]}\n".encode("utf-8"))

    return {
        "rollup": digest.hexdigest(),
        "file_count": len(files),
        "total_bytes": total_bytes,
        "files": files,
        "excluded": clutter.describe(),
    }


def hash_path(path: str | Path) -> dict:
    """Hash a file or a directory, returning the same shape either way."""
    path = Path(path)
    if path.is_dir():
        return hash_dir(path)
    if path.is_file():
        return {
            "rollup": hash_file(path),
            "file_count": 1,
            "total_bytes": path.stat().st_size,
            "files": {path.name: hash_file(path)},
            "excluded": {"names": [], "dirs": []},
        }
    raise FileNotFoundError(f"input path does not exist: {path}")
