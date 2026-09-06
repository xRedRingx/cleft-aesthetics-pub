"""The run-directory name, checked against itself.

PLAN §2.4's contract is ``<config-stem>__<sha8>__<job-id>``, so the name
carries the run's identity TWICE. Nothing compared the two halves until
2026-08-12, when a directory turned up whose stem said ``224`` and whose
job id said ``768``: a job launched under the wrong id that ran the 224
config (``roadb.MISTYPED_LAUNCH_DELETED``). It was caught by eye.

**Why the obvious rule is the wrong rule, measured before it was
written.** "The job id is the stem with underscores hyphenated" holds for
22 of the 83 run directories this repo declares and fails for 61 --
because job ids legitimately ABBREVIATE (``p6-pt-swin-g1-v2`` for
``p6_pretrain_swin_b_masked_g1``), carry attempt suffixes (``-2``,
``-v3``), and carry variant markers (``-unfused``). A guard that fires on
all 61 would be switched off within a day, and a switched-off guard is
worse than none.

**So the rule is: a job id may DROP what the stem says, but it may not
CONTRADICT it.** For each closed vocabulary below -- resolution,
backbone, geometry, init, region scheme -- any value the job id names
must also be named by the stem. Abbreviation passes; a 768 job id on a
224 config does not. That is exactly the failure that occurred, and the
axes are the discriminators whose confusion would produce a
plausible-looking wrong comparison rather than an obvious mess.
"""

from __future__ import annotations

import re


class RunNameError(ValueError):
    """A run-directory name that does not satisfy the contract."""


#: The closed vocabularies a job id may abbreviate away but never
#: contradict. Each is a set of TOKENS as they appear split on ``_``/``-``.
#:
#: Resolutions are Road B's axis; backbones include ``b16`` because
#: ``vit_b16`` splits into two tokens and a job id usually says only
#: ``vit``; the init axis omits ``scut`` (a qualifier, not a value) so
#: that ``p7-d-swin-masked`` matches ``..._scut_masked_...``.
RUN_DIR_AXES = {
    "resolution": {"224", "512", "768"},
    "backbone": {"vit", "b16", "swin", "srgnn", "agnet"},
    "geometry": {"g1", "g2"},
    "init": {"imagenet", "masked", "original"},
    "scheme": {"native", "anatomy", "grid", "random"},
    #: **[ADDED 2026-09-06, for Phase 27]** WHAT THE ARM TRAINED ON.
    #: Every axis above describes the FEATURES. None described the
    #: training population, because until Phase 27 every arm trained
    #: on the cohort and the axis would have had one value. An arm
    #: fit on the 25 anchors and one fit on the 237 would otherwise
    #: be indistinguishable by directory name
    #: (``phase27.WHAT_MUST_BE_BUILT`` item 4).
    "trained_on": {"cohort", "anchors", "synth"},
}

_SEPARATORS = re.compile(r"[-_]")


def parse_run_dir(name: str) -> tuple[str, str, str]:
    """``(config_stem, sha8, job_id)`` from a run-directory name.

    Raises rather than returning a sentinel: a name that does not carry
    the contract cannot be checked against itself, and silently skipping
    it is how the check would come to cover nothing.
    """
    parts = name.split("__")
    if len(parts) != 3:
        raise RunNameError(
            f"{name!r} is not <config-stem>__<sha8>__<job-id>: "
            f"{len(parts)} parts"
        )
    stem, sha, job = parts
    if not re.fullmatch(r"[0-9a-f]{8}", sha):
        raise RunNameError(f"{name!r}: {sha!r} is not an 8-char hex sha")
    if not stem or not job:
        raise RunNameError(f"{name!r}: empty stem or job id")
    return stem, sha, job


def _tokens(text: str) -> set[str]:
    return {token for token in _SEPARATORS.split(text) if token}


def job_id_contradictions(name: str) -> list[dict]:
    """Every axis on which the job id names something the stem does not.

    Empty means consistent -- either the job id agrees or it abbreviates.
    A non-empty result is the mistyped-launch shape: the two halves of
    the name describe different runs, and downstream readers take
    different halves (a human skims the job id; a declaration matches the
    whole string).
    """
    stem, _sha, job = parse_run_dir(name)
    stem_tokens, job_tokens = _tokens(stem), _tokens(job)
    found = []
    for axis, vocabulary in sorted(RUN_DIR_AXES.items()):
        in_job = job_tokens & vocabulary
        in_stem = stem_tokens & vocabulary
        extra = in_job - in_stem
        if extra:
            found.append({
                "axis": axis,
                "job_id_says": sorted(extra),
                "config_stem_says": sorted(in_stem) or ["nothing"],
            })
    return found


def check_run_dir(name: str) -> None:
    """Raise if the two halves of a run-directory name disagree."""
    contradictions = job_id_contradictions(name)
    if contradictions:
        detail = "; ".join(
            f"{item['axis']}: job id says {item['job_id_says']}, config "
            f"stem says {item['config_stem_says']}"
            for item in contradictions
        )
        raise RunNameError(
            f"{name!r}: the run-directory name contradicts itself -- "
            f"{detail}. PLAN 2.4 carries the identity twice and readers "
            "take different halves, so this is the shape that files one "
            "cell's result under another cell's name "
            "(roadb.MISTYPED_LAUNCH_DELETED)"
        )
