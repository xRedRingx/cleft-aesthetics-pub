"""Is this suite running against the dependencies the cluster runs?

R1: measure before asserting. Every assumption this project checked turned out
wrong, and this one already has -- the suite was green for days against numpy
2.2.6 on the laptop's system python while the cluster and the image run 1.26.4.
It passed because nothing in the suite touches numpy-2-incompatible code, which
is luck, not evidence.

A green run in the wrong environment is worse than a red one: it is false
confidence, and false confidence is what R1 exists to prevent. So this is a
runtime check rather than a paragraph in the README.

If this file is the only thing failing, the other 130-odd passes are not wrong,
they are *unverified against the real environment*. Build the pinned one and run
again; the recipe is in the failure message.
"""

from __future__ import annotations

import sys
from importlib.metadata import PackageNotFoundError, version

import pytest

from fixtures import pins

#: The packages that can silently change a numeric result. Named explicitly so a
#: reader knows why these matter more than, say, a pytest patch version.
NUMERICAL_STACK = ("numpy", "scipy", "scikit-learn")

RECIPE = """
Build the pinned environment on python 3.11 and run the suite there:

    py -3.11 -m venv .venv
    .venv/Scripts/pip install -e ".[test]"
    .venv/Scripts/python -m pytest

On Linux, .venv/bin/ instead of .venv/Scripts/.

Note that `pip install -e ".[test]"` cannot work on python 3.13: numpy 1.26.4
publishes no 3.13 wheels, so pip falls back to compiling it from source. The pin
is not the problem -- 3.13 is not the interpreter this project runs on.
""".rstrip()


def _installed(package: str) -> str | None:
    try:
        return version(package)
    except PackageNotFoundError:
        return None


def test_numerical_stack_matches_the_pinned_versions():
    """numpy, scipy and scikit-learn must be exactly what the image ships.

    `numpy<2` is a real ABI constraint, not a preference: torch-geometric and the
    pinned opencv build are compiled against the 1.x ABI, and a silent upgrade
    produces wrong numbers rather than an import error.
    """
    expected = pins.test_extra()
    wrong = {}
    for package in NUMERICAL_STACK:
        want = expected.get(pins.normalise(package))
        assert want, f"{package} is not pinned in the test extra"
        got = _installed(package)
        if got != want:
            wrong[package] = (got or "not installed", want)

    if wrong:
        detail = "\n".join(
            f"    {name:<14} installed {got:<12} pinned {want}"
            for name, (got, want) in sorted(wrong.items())
        )
        pytest.fail(
            "This suite is running against the WRONG numerical stack.\n\n"
            f"{detail}\n\n"
            "Whatever else passed in this run was verified against dependencies "
            "the cluster does not use, so it is not evidence that the code works "
            "there.\n"
            f"{RECIPE}",
            pytrace=False,
        )


def test_every_pinned_test_dependency_is_the_installed_one():
    """The rest of the extra too -- PyYAML and pytest itself.

    Less likely to change a number, but the rule is simply that the environment
    is the pinned environment; carving out exceptions is how drift starts.
    """
    expected = pins.test_extra()
    wrong = {
        name: (_installed(name) or "not installed", want)
        for name, want in sorted(expected.items())
        if _installed(name) != want
    }
    if wrong:
        detail = "\n".join(
            f"    {name:<14} installed {got:<12} pinned {want}"
            for name, (got, want) in wrong.items()
        )
        pytest.fail(f"installed packages differ from the test extra:\n\n{detail}\n{RECIPE}", pytrace=False)


def test_running_python_matches_the_declared_interpreter():
    """The suite must run on the interpreter series the image ships.

    Compared at major.minor: the image pins an exact patch version and
    RunContext enforces that inside the container, but CI and a local venv will
    legitimately sit on different 3.11.x patches.
    """
    declared = pins.declared_python()
    assert declared, "docker/requirements.txt declares no python version"

    want = ".".join(declared.split(".")[:2])
    got = "{}.{}".format(*sys.version_info[:2])

    if got != want:
        pytest.fail(
            f"This suite is running on python {got}; the image ships {want} "
            f"(declared {declared} in docker/requirements.txt).\n\n"
            "Python version changes behaviour that tests can depend on, and it "
            "determines which dependency wheels exist at all.\n"
            f"{RECIPE}",
            pytrace=False,
        )


def test_this_check_is_not_excluded_from_collection():
    """A check that stops running is indistinguishable from a check that passes.

    Skipping or ignoring this file would restore exactly the false confidence it
    exists to remove, so make that require deleting a test rather than adding a
    config line.
    """
    config = pins.PYPROJECT.read_text(encoding="utf-8")
    assert "test_environment" not in config, (
        "test_environment is named in pyproject.toml, which suggests it is being "
        "ignored or deselected. This check must always run."
    )
