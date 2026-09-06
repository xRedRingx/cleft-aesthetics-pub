"""Parsing the two pinned dependency lists.

Shared by ``test_workflow_hygiene`` (do the two lists agree with each other?) and
``test_environment`` (does the environment actually running the tests agree with
them?). One parser, so the two answers cannot disagree for parsing reasons.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
REQUIREMENTS = REPO / "docker" / "requirements.txt"
PYPROJECT = REPO / "pyproject.toml"


def normalise(name: str) -> str:
    """PEP 503 name normalisation: PyYAML, pyyaml and py-yaml are one package."""
    return re.sub(r"[-_.]+", "-", name).lower()


def pinned(lines: list[str]) -> dict[str, str]:
    """Map normalised package name -> version, for ``name==version`` pins only."""
    out: dict[str, str] = {}
    for line in lines:
        stripped = line.split("#")[0].strip().strip('",\'')
        match = re.fullmatch(r"([A-Za-z0-9][A-Za-z0-9._-]*)==([^\s;]+)", stripped)
        if match:
            out[normalise(match.group(1))] = match.group(2)
    return out


def test_extra() -> dict[str, str]:
    """The [project.optional-dependencies].test pins.

    Read as raw text rather than through a toml parser so that this keeps working
    even when pyproject.toml stops being valid toml -- which is exactly when a
    hygiene test is most useful. ``test_pyproject_is_valid_toml_and_declares_the_test_extra``
    cross-checks the two readings against each other.
    """
    text = PYPROJECT.read_text(encoding="utf-8")
    block = re.search(r"^test\s*=\s*\[(.*?)\]", text, re.MULTILINE | re.DOTALL)
    assert block, "pyproject.toml has no [project.optional-dependencies].test list"
    return pinned(block.group(1).splitlines())


def image_pins() -> dict[str, str]:
    """The pins in docker/requirements.txt."""
    return pinned(REQUIREMENTS.read_text(encoding="utf-8").splitlines())


def declared_python() -> str | None:
    """The interpreter version declared in docker/requirements.txt."""
    match = re.search(
        r"^#\s*python:\s*(\d+\.\d+\.\d+)\s*$",
        REQUIREMENTS.read_text(encoding="utf-8"),
        re.MULTILINE,
    )
    return match.group(1) if match else None
