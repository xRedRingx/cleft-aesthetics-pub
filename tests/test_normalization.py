"""Guards a known Stage-1 bug: hardcoded ImageNet normalization constants.

Normalization must come from the model factory, which reads it from the
pretrained config of the backbone actually being used. A hardcoded ImageNet
triple is silently wrong for any backbone that was not trained with it, and
silently right for the ones that were, which is why it survives review.

The factory does not exist yet (Phase 0 builds no models). The test is written
now so that it is already in place the moment it can fail.
"""

from __future__ import annotations

import re
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"

#: The only file permitted to name normalization constants.
FACTORY = "cleft/models/factory.py"

IMAGENET_LITERALS = [
    ("0.485", "0.456", "0.406"),  # mean
    ("0.229", "0.224", "0.225"),  # std
]


def _python_sources() -> list[Path]:
    return sorted(SRC.rglob("*.py"))


def _relative(path: Path) -> str:
    return path.relative_to(SRC).as_posix()


def test_imagenet_constants_appear_nowhere_outside_the_factory():
    offenders = []
    for path in _python_sources():
        if _relative(path) == FACTORY:
            continue
        text = path.read_text(encoding="utf-8")
        for triple in IMAGENET_LITERALS:
            if all(value in text for value in triple):
                offenders.append((_relative(path), triple))

    assert not offenders, (
        "hardcoded ImageNet normalization found outside "
        f"{FACTORY}: {offenders}. Read it from the model factory instead."
    )


def test_no_source_file_hardcodes_a_normalization_triple_literal():
    """Catches the same bug written with different numbers.

    Any three-float tuple assigned to something called mean/std is suspect
    outside the factory, whatever the values are.
    """
    pattern = re.compile(
        r"\b(mean|std)\s*=\s*[\(\[]\s*"
        r"\d+\.\d+\s*,\s*\d+\.\d+\s*,\s*\d+\.\d+\s*[\)\]]"
    )
    offenders = []
    for path in _python_sources():
        if _relative(path) == FACTORY:
            continue
        for lineno, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if pattern.search(line):
                offenders.append(f"{_relative(path)}:{lineno}")

    assert not offenders, f"normalization triple hardcoded at: {offenders}"


def test_the_allowlisted_factory_path_is_the_one_the_project_will_use():
    """If the factory moves, this test must be updated deliberately, not silently.

    Passing because the file is absent is acceptable in Phase 0; passing because
    it was renamed and the allowlist now guards nothing is not.
    """
    models_dir = SRC / "cleft" / "models"
    factory = SRC / FACTORY
    if not models_dir.exists():
        return
    candidates = [p for p in models_dir.glob("*factory*.py")]
    assert not candidates or factory in candidates, (
        f"a factory-like module exists but the allowlist points at {FACTORY}: "
        f"{[_relative(p) for p in candidates]}"
    )
