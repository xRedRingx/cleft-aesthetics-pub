"""The one filesystem-clutter rule, shared by every consumer.

Three places walk directories and must agree exactly on what is not data:

* ``provenance.hashing`` — the rollup that guard 3 checks
* ``data.folderscan`` — the patient folder inventory
* ``scripts/declare_inputs.py`` — via ``hash_path``, so it inherits the same rule

They must share one list, because a file counted by one and skipped by another
produces a hash that depends on which tool last looked at the directory.

**Two measured facts drive this rule** (cluster inventory, 2026-07-27):

``._<name>.jpg``
    Every image has a macOS AppleDouble sidecar of exactly 4,096 bytes. It has a
    ``.jpg`` extension and its stem contains the image id, so **extension
    filtering cannot catch it and stem parsing would read it as a duplicate
    image**. The rule has to be by basename prefix.

``Thumbs.db``
    Windows regenerates this whenever the directory is browsed. A rollup that
    included it would change without the data changing, and guard 3 would abort a
    legitimate run for a reason that has nothing to do with the cohort.

Skipped files are counted and reported, never dropped silently: "I ignored 479
files" is a fact worth seeing.
"""

from __future__ import annotations

#: Exact basenames that are never data.
CLUTTER_NAMES = frozenset({"Thumbs.db", ".DS_Store", "desktop.ini", ".Spotlight-V100"})

#: Basename prefixes that are never data. ``._`` is the AppleDouble sidecar
#: prefix; the suffix is whatever the file it shadows had.
CLUTTER_PREFIXES = ("._",)

#: Directory names that are never data.
CLUTTER_DIRS = frozenset({"__pycache__", ".git", ".ipynb_checkpoints", ".Trashes"})


def is_clutter(name: str) -> bool:
    """Is this basename filesystem clutter rather than data?

    By basename, deliberately. ``._1001.jpg`` is a 4,096-byte AppleDouble sidecar
    that an extension check would call an image.
    """
    if name in CLUTTER_NAMES:
        return True
    return any(name.startswith(prefix) for prefix in CLUTTER_PREFIXES)


def is_clutter_dir(name: str) -> bool:
    return name in CLUTTER_DIRS or is_clutter(name)


def describe() -> dict:
    """The rule, recorded alongside a hash so it is auditable."""
    return {
        "names": sorted(CLUTTER_NAMES),
        "prefixes": list(CLUTTER_PREFIXES),
        "dirs": sorted(CLUTTER_DIRS),
    }
