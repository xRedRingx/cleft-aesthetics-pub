"""Read the patient folder tree into ``{folder_id: [image_id, ...]}``.

This is the module most likely to meet a surprise. It has never seen the real
tree, and the fixtures cannot contain what nobody has looked at yet: stray files,
an unexpected extension, a nested directory, a filename whose number is not where
we assume.

So it is written to **fail with an inventory rather than a traceback**. When it
refuses, the message says what it found, which is what turns one cluster round
trip into a fix rather than a guessing game.

Two rules it will not bend:

* An image id is the integer in the filename stem. If a stem has no integer, or
  more than one candidate, that is an error and not a guess.
* A folder holds one or two images. Anything else stops the build.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from .. import clutter

#: Extensions treated as photographs. Case-insensitive.
IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"})

#: Archives sitting beside the extracted directories. Pointing at a .zip instead
#: of the extracted folder is an easy mistake and would otherwise present as
#: "0 images".
IGNORED_SUFFIXES = frozenset({".zip", ".7z", ".tar", ".gz"})

INTEGER = re.compile(r"\d+")


class FolderScanError(ValueError):
    """The folder tree is not the shape the manifest requires."""


def image_id_from(path: Path) -> int:
    """The integer in a filename stem, when there is exactly one."""
    found = INTEGER.findall(path.stem)
    if not found:
        raise FolderScanError(
            f"no number in filename {path.name!r} (folder {path.parent.name}); "
            "an image id is the integer in the stem"
        )
    if len({int(f) for f in found}) > 1:
        raise FolderScanError(
            f"ambiguous image id in {path.name!r} (folder {path.parent.name}): "
            f"found {found}. Exactly one integer is expected in the stem."
        )
    return int(found[0])


@dataclass
class ScanResult:
    """Every numeric directory, plus what was skipped getting there."""

    folders: dict[int, list[int]] = field(default_factory=dict)
    skipped_clutter: int = 0
    skipped_names: Counter = field(default_factory=Counter)

    @property
    def with_images(self) -> dict[int, list[int]]:
        return {f: ids for f, ids in self.folders.items() if ids}

    @property
    def empty(self) -> list[int]:
        """Directories that exist but hold no image. Folder 52 is one."""
        return sorted(f for f, ids in self.folders.items() if not ids)

    @property
    def total_images(self) -> int:
        return sum(len(ids) for ids in self.folders.values())


def scan(
    root: str | Path, *, max_images: int = 2, min_images: int = 0
) -> ScanResult:
    """Inventory every numeric directory under ``root``.

    ``min_images`` defaults to 0 because **folder 52 exists and is empty**. An
    empty directory is a different fact from a missing one, and the manifest
    needs to be able to tell them apart rather than infer absence from silence.
    """
    root = Path(root)
    if not root.is_dir():
        raise FolderScanError(f"patient folder root is not a directory: {root}")

    result = ScanResult()
    unnamed: list[str] = []
    oddities: list[str] = []

    for child in sorted(root.iterdir()):
        if clutter.is_clutter(child.name):
            result.skipped_clutter += 1
            result.skipped_names[child.name] += 1
            continue
        if child.suffix.lower() in IGNORED_SUFFIXES:
            continue
        if not child.is_dir():
            oddities.append(f"loose file at the root: {child.name}")
            continue
        if not child.name.isdigit():
            unnamed.append(child.name)
            continue

        images = []
        for entry in sorted(child.rglob("*")):
            if entry.is_dir():
                continue
            # By BASENAME, before any extension check: an AppleDouble sidecar is
            # named ._1001.jpg, so extension filtering would admit it and stem
            # parsing would read it as a second copy of image 1001.
            if clutter.is_clutter(entry.name):
                result.skipped_clutter += 1
                key = "._*" if entry.name.startswith("._") else entry.name
                result.skipped_names[key] += 1
                continue
            if entry.suffix.lower() in IGNORED_SUFFIXES:
                continue
            if entry.suffix.lower() not in IMAGE_SUFFIXES:
                oddities.append(f"{child.name}/{entry.name} (unexpected extension)")
                continue
            images.append(image_id_from(entry))

        ids = sorted(set(images))
        if len(ids) != len(images):
            raise FolderScanError(
                f"folder {child.name}: repeated image id among {sorted(images)}. "
                "If these are AppleDouble sidecars, the clutter rule has regressed."
            )
        if not min_images <= len(ids) <= max_images:
            raise FolderScanError(
                f"folder {child.name}: found {len(ids)} image(s) {ids}, expected "
                f"between {min_images} and {max_images}. Full inventory: "
                f"{[e.name for e in sorted(child.iterdir())]}"
            )
        result.folders[int(child.name)] = ids

    if unnamed:
        raise FolderScanError(
            f"{len(unnamed)} entry(ies) under {root} are not numbered patient "
            f"folders: {sorted(unnamed)[:10]}. If these are archives, point the "
            "config at the extracted directory instead."
        )
    if oddities:
        raise FolderScanError(
            f"unexpected entries under {root}: {oddities[:10]}"
            + (f" (+{len(oddities) - 10} more)" if len(oddities) > 10 else "")
        )
    if not result.folders:
        raise FolderScanError(
            f"no numbered patient folders under {root}. Is this the extracted "
            "directory, or the .zip beside it?"
        )
    return result
